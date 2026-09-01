"""M2.3の訓練上限、評価分離、継続門を検証する。"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from typing import Any

import gymnasium as gym
import numpy as np

from general_terrain.student_prefix_rescue_env import (
    POST_CLEARANCE_RECOVERY_PHASE,
    PRE_HURDLE_PHASE,
)
from general_terrain.train_prefix_rescue_teacher import (
    RescueTrainingAuditWrapper,
    compute_rollout_budget,
    copy_teacher_initialization,
    evaluate_m2_3_gate,
    evaluate_rescue_teacher,
    sha256_file,
)


class FakeRescueTeacher:
    """評価時の循環状態と開始フラグを記録する模擬教師。"""

    def __init__(self) -> None:
        self.states: list[Any] = []
        self.episode_starts: list[bool] = []

    def predict(
        self,
        observation: np.ndarray,
        *,
        state: Any,
        episode_start: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.ndarray, int]:
        """ゼロ動作と増加する循環状態を返す。"""
        del observation, deterministic
        self.states.append(state)
        self.episode_starts.append(bool(episode_start[0]))
        next_state = 1 if state is None else int(state) + 1
        return np.zeros(6, dtype=np.float32), next_state


class FakeCurrentSpec:
    """評価結果に必要な開始位置だけを保持する。"""

    start_runway_voxels = 20


class FakeRescueEnvironment(gym.Env):
    """成功、転倒、停滞を乱数シードで切り替える模擬環境。"""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(95,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        self.current_spec = FakeCurrentSpec()
        self.seed_value = 0
        self.steps = 0
        self.received_seeds: list[int] = []

    def _info(self) -> dict[str, object]:
        """教師評価に必要な監査項目を返す。"""
        phase = (
            POST_CLEARANCE_RECOVERY_PHASE
            if self.seed_value == 1 and self.steps >= 2
            else PRE_HURDLE_PHASE
        )
        success = self.seed_value == 1 and self.steps >= 2
        hard_fall = self.seed_value == 2 and self.steps >= 1
        safe_stall = self.seed_value == 3 and self.steps >= 3
        return {
            "rescue_reset_seed": self.seed_value,
            "teacher_training_only": True,
            "student_controller_active": False,
            "student_observation_privileged": False,
            "orientation_error": 0.1,
            "stall_steps": 20 if safe_stall else 0,
            "upper_body_grounded": False,
            "rescue_phase": phase,
            "rescue_success": success,
            "course_complete": success,
            "hard_fall": hard_fall,
            "sequence_failed": False,
            "failure_reason": "stall_limit" if safe_stall else "",
            "raw_clearances": int(success),
            "recovered_obstacles": int(success),
            "max_x_position": float(self.steps),
            "rescue_steps": self.steps,
            "stall_limit_reached": safe_stall,
            "time_limit_reached": safe_stall,
            "rescue_step_limit_reached": False,
        }

    def reset(self, *, seed=None, options=None):
        """指定されたプレフィックス乱数シードで状態を初期化する。"""
        super().reset(seed=seed)
        self.seed_value = int(options["prefix_seed"])
        self.received_seeds.append(self.seed_value)
        self.steps = 0
        return np.zeros(95, dtype=np.float32), self._info()

    def step(self, action):
        """乱数シードに対応する終了結果へ一歩進める。"""
        del action
        self.steps += 1
        info = self._info()
        terminated = bool(info["rescue_success"] or info["hard_fall"])
        truncated = bool(info["stall_limit_reached"])
        return (
            np.zeros(95, dtype=np.float32),
            1.0,
            terminated,
            truncated,
            info,
        )


class M23RescueTeacherTests(unittest.TestCase):
    """M2.3が訓練専用教師の境界を守るか検査する。"""

    def test_rollout_budget_never_exceeds_five_thousand(self):
        self.assertEqual(compute_rollout_budget(5_000, 128), 4_992)
        self.assertLessEqual(compute_rollout_budget(5_000, 128), 5_000)
        with self.assertRaises(ValueError):
            compute_rollout_budget(5_001, 128)

    def test_initialization_copy_preserves_exact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "student.zip"
            destination = root / "teacher.zip"
            source.write_bytes(b"frozen-student")
            copied_hash = copy_teacher_initialization(source, destination)
            self.assertEqual(copied_hash, sha256_file(source))
            self.assertEqual(destination.read_bytes(), b"frozen-student")

    def test_evaluation_uses_only_explicit_train_reset_seeds(self):
        environment = FakeRescueEnvironment()
        teacher = FakeRescueTeacher()
        result = evaluate_rescue_teacher(
            environment,
            teacher,
            reset_seeds=(1, 2, 3),
        )
        self.assertEqual(environment.received_seeds, [1, 2, 3])
        self.assertEqual(result["split"], "train")
        self.assertEqual(result["validation_episodes"], 0)
        self.assertEqual(result["holdout_episodes"], 0)
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["hard_fall_count"], 1)
        self.assertEqual(result["safe_stall_count"], 1)
        self.assertEqual(result["phase_step_counts"][PRE_HURDLE_PHASE], 5)
        self.assertEqual(
            result["phase_step_counts"][POST_CLEARANCE_RECOVERY_PHASE],
            1,
        )
        self.assertEqual(teacher.states, [None, 1, None, None, 1, 2])
        self.assertEqual(
            teacher.episode_starts,
            [True, False, True, True, False, False],
        )

    def test_training_wrapper_counts_steps_and_terminal_results(self):
        base = FakeRescueEnvironment()
        wrapper = RescueTrainingAuditWrapper(base)
        wrapper.reset(options={"prefix_seed": 1})
        wrapper.step(np.zeros(6, dtype=np.float32))
        wrapper.step(np.zeros(6, dtype=np.float32))
        snapshot = wrapper.snapshot()
        self.assertEqual(snapshot["observed_step_count"], 2)
        self.assertEqual(snapshot["reset_seed_counts"], {"1": 1})
        self.assertEqual(snapshot["completed_episode_count"], 1)
        self.assertTrue(snapshot["completed_episodes"][0]["rescue_success"])

    def test_continue_gate_requires_success_safety_and_stall_improvement(self):
        passed = evaluate_m2_3_gate(
            {"success_count": 4, "hard_fall_count": 1, "safe_stall_count": 6}
        )
        self.assertTrue(passed["continue_to_m2_4"])
        failed = evaluate_m2_3_gate(
            {"success_count": 3, "hard_fall_count": 0, "safe_stall_count": 8}
        )
        self.assertFalse(failed["continue_to_m2_4"])


if __name__ == "__main__":
    unittest.main()
