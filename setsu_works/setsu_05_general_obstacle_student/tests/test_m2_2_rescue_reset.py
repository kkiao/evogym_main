"""M2.2の学生プレフィックスのリセット、段階分類、訓練隔離を検証する。"""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import general_terrain.audit_rescue_reset_env as audit_module
import general_terrain.student_prefix_rescue_env as rescue_env_module
from general_terrain.audit_rescue_reset_env import (
    audit_reset_determinism,
    run_action_smoke_test,
)
from general_terrain.rescue_reset_manifest import (
    RescueResetManifest,
    RescueResetSpec,
    load_rescue_reset_manifest,
)
from general_terrain.student_prefix_rescue_env import (
    HURDLE_DEFORMATION_PHASE,
    POST_CLEARANCE_RECOVERY_PHASE,
    POST_RECOVERY_STALL_PHASE,
    PRE_HURDLE_PHASE,
    StudentPrefixRescueEnv,
    classify_rescue_phase,
)


class FakePrefixStudent:
    """循環状態の連続性を記録する凍結模擬学生。"""

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


class FakePrefixEnvironment:
    """プレフィックス歩数だけ確定的に進む模擬物理環境。"""

    VOXEL_SIZE = 0.1

    def __init__(self, course) -> None:
        self.course = course
        self.unwrapped = self
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(95,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        self.schema = tuple(f"feature_{index}" for index in range(95))
        self.step_count = 0
        self.positions = np.asarray(
            [[0.1, 0.2], [0.2, 0.2]],
            dtype=np.float64,
        )
        self.closed = False
        self.info_overrides: dict[str, object] = {}

    def _observation(self) -> np.ndarray:
        """現在の歩数を先頭に持つ95次元観測を返す。"""
        observation = np.zeros(95, dtype=np.float32)
        observation[0] = float(self.step_count)
        return observation

    def _info(self) -> dict[str, object]:
        """救援環境に必要な監査項目を返す。"""
        info: dict[str, object] = {
            "course_id": self.course.course_id,
            "obstacle_count": 1,
            "x_position": 0.3 + 0.01 * self.step_count,
            "orientation_error": 0.0,
            "angular_velocity": 0.0,
            "stall_steps": 0,
            "upper_body_grounded": False,
            "hard_fall": False,
            "course_complete": False,
            "raw_clearances": 0,
            "recovered_obstacles": 0,
        }
        info.update(self.info_overrides)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """コースを受け取り歩数と物理座標を初期化する。"""
        del seed
        if options and "course" in options:
            self.course = options["course"]
        self.step_count = 0
        self.info_overrides = {}
        self.positions = np.asarray(
            [[0.1, 0.2], [0.2, 0.2]],
            dtype=np.float64,
        )
        return self._observation(), self._info()

    def step(self, action: np.ndarray):
        """有限の固定報酬で一歩だけ進む。"""
        del action
        self.step_count += 1
        self.positions = self.positions + np.asarray([[0.01], [0.0]])
        return self._observation(), 0.5, False, False, self._info()

    def object_pos_at_time(self, time_value: int, object_name: str) -> np.ndarray:
        """模擬ロボットの二次元座標を返す。"""
        del time_value, object_name
        return self.positions.copy()

    def get_time(self) -> int:
        """現在の模擬歩数を返す。"""
        return self.step_count

    def close(self) -> None:
        """環境が閉じられたことを記録する。"""
        self.closed = True


def make_spec() -> RescueResetSpec:
    """3ステップのプレフィックスを持つ模擬救援リセット地点を返す。"""
    return RescueResetSpec(
        seed=100009,
        start_runway_voxels=20,
        prefix_steps=3,
        trigger_reason="teacher_disagreement",
        x_position=0.33,
        orientation_error=0.0,
        angular_velocity=0.0,
        stall_steps=0,
        raw_clearances=0,
        recovered_obstacles=0,
    )


class M22RescueResetTests(unittest.TestCase):
    """救援リセット環境が学生プレフィックスと教師課題を分離するか検査する。"""

    def make_environment(self) -> tuple[StudentPrefixRescueEnv, FakePrefixStudent]:
        """模擬学生と模擬物理環境を組み合わせる。"""
        student = FakePrefixStudent()
        environment = StudentPrefixRescueEnv(
            student,
            (make_spec(),),
            enforce_reference_metrics=False,
            environment_factory=FakePrefixEnvironment,
        )
        return environment, student

    def test_reset_replays_exact_prefix_and_keeps_recurrent_state(self):
        environment, student = self.make_environment()
        try:
            observation, info = environment.reset(
                options={"prefix_seed": 100009}
            )
        finally:
            environment.close()
        self.assertEqual(float(observation[0]), 3.0)
        self.assertEqual(student.states, [None, 1, 2])
        self.assertEqual(student.episode_starts, [True, False, False])
        self.assertEqual(info["student_prefix_steps"], 3)
        self.assertTrue(info["teacher_training_only"])
        self.assertFalse(info["student_controller_active"])
        self.assertEqual(info["rescue_phase"], PRE_HURDLE_PHASE)

    def test_teacher_step_has_finite_reward_and_fixed_shapes(self):
        environment, _ = self.make_environment()
        try:
            observation, _ = environment.reset(
                options={"prefix_seed": 100009}
            )
            next_observation, reward, terminated, truncated, info = (
                environment.step(np.zeros(6, dtype=np.float32))
            )
        finally:
            environment.close()
        self.assertEqual(observation.shape, (95,))
        self.assertEqual(next_observation.shape, (95,))
        self.assertTrue(np.isfinite(reward))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["rescue_steps"], 1)

    def test_explicit_gym_seed_selects_same_reset_state(self):
        environment, _ = self.make_environment()
        try:
            first, _ = environment.reset(seed=7)
            second, _ = environment.reset(seed=7)
        finally:
            environment.close()
        np.testing.assert_array_equal(first, second)

    def test_success_and_hard_fall_have_explicit_teacher_termination(self):
        success_environment, _ = self.make_environment()
        try:
            success_environment.reset(options={"prefix_seed": 100009})
            success_environment.base_environment.info_overrides = {
                "raw_clearances": 1,
                "recovered_obstacles": 1,
                "course_complete": True,
            }
            _, success_reward, success_terminated, success_truncated, success_info = (
                success_environment.step(np.zeros(6, dtype=np.float32))
            )
        finally:
            success_environment.close()
        self.assertTrue(success_terminated)
        self.assertFalse(success_truncated)
        self.assertTrue(success_info["rescue_success"])
        self.assertGreater(success_reward, 25.0)

        fall_environment, _ = self.make_environment()
        try:
            fall_environment.reset(options={"prefix_seed": 100009})
            fall_environment.base_environment.info_overrides = {
                "hard_fall": True,
            }
            _, fall_reward, fall_terminated, fall_truncated, fall_info = (
                fall_environment.step(np.zeros(6, dtype=np.float32))
            )
        finally:
            fall_environment.close()
        self.assertTrue(fall_terminated)
        self.assertFalse(fall_truncated)
        self.assertFalse(fall_info["rescue_success"])
        self.assertLess(fall_reward, 0.0)

    def test_four_training_only_phases_are_distinct(self):
        environment = FakePrefixEnvironment(
            rescue_env_module.sample_curriculum_course(
                100009,
                "hurdle_single",
                "train",
            )
        )
        info = environment._info()
        self.assertEqual(classify_rescue_phase(environment, info), PRE_HURDLE_PHASE)
        obstacle_start = environment.course.obstacles[0].start_x * environment.VOXEL_SIZE
        environment.positions[0, 1] = obstacle_start
        self.assertEqual(
            classify_rescue_phase(environment, info),
            HURDLE_DEFORMATION_PHASE,
        )
        self.assertEqual(
            classify_rescue_phase(environment, {**info, "raw_clearances": 1}),
            POST_CLEARANCE_RECOVERY_PHASE,
        )
        self.assertEqual(
            classify_rescue_phase(
                environment,
                {
                    **info,
                    "raw_clearances": 1,
                    "recovered_obstacles": 1,
                    "stall_steps": 20,
                },
            ),
            POST_RECOVERY_STALL_PHASE,
        )

    def test_reset_audit_and_smoke_test_are_deterministic(self):
        environment, _ = self.make_environment()
        manifest = RescueResetManifest(
            version="test",
            stage="hurdle_single",
            split="train",
            source_profile="test",
            source_summary=Path("test.json"),
            source_summary_sha256="0" * 64,
            student_model_sha256="1" * 64,
            states=(make_spec(),),
            source_path=Path("manifest.json"),
            sha256="2" * 64,
        )
        try:
            rows = audit_reset_determinism(environment, manifest, repeats=2)
            smoke = run_action_smoke_test(
                environment,
                prefix_seed=100009,
                requested_steps=3,
            )
        finally:
            environment.close()
        self.assertTrue(rows[0]["deterministic"])
        self.assertEqual(rows[0]["maximum_observation_difference"], 0.0)
        self.assertEqual(smoke["completed_steps"], 3)
        self.assertTrue(smoke["all_rewards_finite"])

    def test_frozen_manifest_matches_train_seeds_and_source_hash(self):
        manifest = load_rescue_reset_manifest()
        self.assertEqual(len(manifest.states), 11)
        self.assertEqual(manifest.split, "train")
        self.assertEqual(
            {state.start_runway_voxels for state in manifest.states},
            set(range(20, 31)),
        )
        self.assertEqual(len(manifest.sha256), 64)
        self.assertEqual(len(manifest.source_summary_sha256), 64)

    def test_m2_2_runtime_contains_no_training_update_call(self):
        sources = "\n".join(
            Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")
            for module in (rescue_env_module, audit_module)
        )
        self.assertNotIn(".learn(", sources)
        self.assertNotIn(".backward(", sources)
        self.assertNotIn("optimizer", sources.lower())


if __name__ == "__main__":
    unittest.main()
