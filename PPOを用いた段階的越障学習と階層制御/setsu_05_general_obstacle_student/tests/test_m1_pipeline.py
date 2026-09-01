"""M1収集、循環履歴、教師隔離、固定乱数種契約を検証する。"""

from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from general_terrain.curriculum import get_curriculum_stage, sample_curriculum_course
from general_terrain.interactive_collection import (
    build_argument_parser as build_collection_parser,
    collect_rescue_batch,
    collect_rescue_episode,
)
from general_terrain.interactive_rescue import RescueConfig
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_only_evaluation import (
    build_argument_parser as build_evaluation_parser,
    evaluate_student_batch,
    evaluate_student_episode,
)
import general_terrain.student_only_evaluation as student_only_module


class FakeStudent:
    """循環状態の受渡しを記録する決定論的な模擬学生。"""

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
        """零動作を返し、呼出回数を次の循環状態として使う。"""
        del observation, deterministic
        self.states.append(state)
        self.episode_starts.append(bool(episode_start[0]))
        next_state = 1 if state is None else int(state) + 1
        return np.zeros(6, dtype=np.float32), next_state


class FakeTeacher:
    """常に一動作を返す訓練専用の模擬教師。"""

    def __init__(self) -> None:
        self.reset_count = 0
        self.predict_count = 0

    def reset(self, environment: Any) -> None:
        """教師初期化回数だけを記録する。"""
        del environment
        self.reset_count += 1

    def predict(
        self,
        environment: Any,
        observation: np.ndarray,
        info: dict[str, object],
    ) -> tuple[np.ndarray, str]:
        """固定教師動作と段階名を返す。"""
        del environment, observation, info
        self.predict_count += 1
        return np.ones(6, dtype=np.float32), "first_to_50"


class FakeTakeoverTeacher(FakeTeacher):
    """接管開始時の動作再計算と解放通知を記録する模擬教師。"""

    def __init__(self) -> None:
        super().__init__()
        self.start_count = 0
        self.release_count = 0

    def on_rescue_start(
        self,
        environment: Any,
        observation: np.ndarray,
        info: dict[str, object],
    ) -> tuple[np.ndarray, str]:
        """接管歩だけ別動作を返し、再初期化相当の呼出しを記録する。"""
        del environment, observation, info
        self.start_count += 1
        return np.full(6, 0.25, dtype=np.float32), "takeover_reset"

    def on_rescue_release(self, environment: Any) -> None:
        """学生への解放通知回数を記録する。"""
        del environment
        self.release_count += 1


class FakeEnvironment:
    """三歩成功または一歩失敗を再現する模擬環境。"""

    def __init__(self, course, *, fail: bool = False) -> None:
        self.course = course
        self.unwrapped = self
        self.fail = fail
        self.step_count = 0
        self.closed = False
        names = [f"feature_{index}" for index in range(95)]
        names[10] = "relative_terrain_height_+0"
        names[11] = "relative_terrain_height_+1"
        self.schema = tuple(names)

    @staticmethod
    def _observation() -> np.ndarray:
        """局所高低差を含む九十五次元模擬観測を返す。"""
        observation = np.zeros(95, dtype=np.float32)
        observation[10] = 0.2
        observation[11] = 0.1
        return observation

    def _info(self, **overrides) -> dict[str, object]:
        """収集と評価に必要な診断項目を揃える。"""
        info: dict[str, object] = {
            "course_id": self.course.course_id,
            "orientation_error": 0.0,
            "angular_velocity": 0.0,
            "stall_steps": 0,
            "upper_body_grounded": False,
            "x_position": 1.2,
            "max_x_position": 1.2,
            "obstacle_count": 1,
            "raw_clearances": 0,
            "recovered_obstacles": 0,
            "course_complete": False,
            "hard_fall": False,
            "failure_reason": "",
        }
        info.update(overrides)
        return info

    def reset(self, *, seed: int | None = None):
        """危険角度から始まる模擬回を初期化する。"""
        del seed
        self.step_count = 0
        return self._observation(), self._info(orientation_error=np.deg2rad(40.0), x_position=1.0)

    def step(self, action: np.ndarray):
        """教師接管後の安全回復または失敗終端を返す。"""
        del action
        self.step_count += 1
        if self.fail:
            return (
                self._observation(),
                0.0,
                True,
                False,
                self._info(hard_fall=True, failure_reason="orientation_hard_fall"),
            )
        complete = self.step_count >= 3
        return (
            self._observation(),
            0.0,
            complete,
            False,
            self._info(
                course_complete=complete,
                raw_clearances=int(complete),
                recovered_obstacles=int(complete),
                x_position=1.2 + 0.1 * self.step_count,
                max_x_position=1.2 + 0.1 * self.step_count,
            ),
        )

    def close(self) -> None:
        """一括処理が環境を閉じたことを記録する。"""
        self.closed = True


class M1PipelineTests(unittest.TestCase):
    """M1で追加した訓練収集と純学生評価の境界を検査する。"""

    def setUp(self) -> None:
        self.course = sample_curriculum_course(
            100009,
            "hurdle_single",
            "train",
        )
        self.rescue_config = RescueConfig(
            minimum_teacher_steps=2,
            release_safe_steps=2,
            release_progress=0.1,
            disagreement_streak_steps=2,
        )

    def test_collection_keeps_student_state_during_teacher_control(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "rescued.npz"
            student = FakeStudent()
            teacher = FakeTeacher()
            environment = FakeEnvironment(self.course)
            result = collect_rescue_episode(
                environment,
                student,
                teacher,
                seed=100009,
                output_path=output_path,
                rescue_config=self.rescue_config,
            )
            self.assertTrue(result.branch_accepted)
            self.assertEqual(result.rescue_count, 1)
            self.assertEqual(result.teacher_control_steps, 2)
            self.assertEqual(result.student_control_steps, 1)
            self.assertEqual(student.states, [None, 1, 2])
            self.assertEqual(student.episode_starts, [True, False, False])
            self.assertEqual(teacher.predict_count, 3)
            with np.load(output_path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(
                    arrays["teacher_mask"],
                    np.asarray([True, True, False]),
                )

    def test_collection_rejects_failed_teacher_rescue(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "failed.npz"
            result = collect_rescue_episode(
                FakeEnvironment(self.course, fail=True),
                FakeStudent(),
                FakeTeacher(),
                seed=100009,
                output_path=output_path,
                rescue_config=self.rescue_config,
            )
            self.assertFalse(result.branch_accepted)
            self.assertTrue(result.hard_fall)
            self.assertFalse(output_path.exists())

    def test_collection_recomputes_teacher_action_on_takeover(self):
        """開始判定後の再初期化動作が同じ接管歩へ保存される。"""
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "reset_takeover.npz"
            teacher = FakeTakeoverTeacher()
            result = collect_rescue_episode(
                FakeEnvironment(self.course),
                FakeStudent(),
                teacher,
                seed=100009,
                output_path=output_path,
                rescue_config=self.rescue_config,
            )
            self.assertTrue(result.branch_accepted)
            self.assertEqual(teacher.start_count, 1)
            self.assertEqual(teacher.release_count, 1)
            with np.load(output_path, allow_pickle=False) as arrays:
                np.testing.assert_array_equal(
                    arrays["teacher_actions"][0],
                    np.full(6, 0.25, dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    arrays["executed_actions"][0],
                    np.full(6, 0.25, dtype=np.float32),
                )
                self.assertEqual(arrays["teacher_stages"][0], "takeover_reset")

    def test_batch_refuses_stage_without_verified_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                collect_rescue_batch(
                    FakeStudent(),
                    FakeTeacher(),
                    seeds=(100009,),
                    stage=get_curriculum_stage("hurdle_double"),
                    output_dir=Path(directory) / "locked",
                )

    def test_student_only_evaluation_has_no_teacher_entry(self):
        signature = inspect.signature(evaluate_student_episode)
        self.assertNotIn("teacher", signature.parameters)
        option_names = {
            option
            for action in build_evaluation_parser()._actions
            for option in action.option_strings
        }
        self.assertFalse(any("teacher" in option for option in option_names))
        source = Path(inspect.getsourcefile(student_only_module)).read_text(encoding="utf-8")
        imported_modules = {
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        )
        self.assertFalse(
            any("portfolio_height1_teacher" in name for name in imported_modules)
        )

    def test_student_only_batch_reports_zero_interventions(self):
        student = FakeStudent()
        result = evaluate_student_batch(
            student,
            seeds=(200004, 200000),
            stage=get_curriculum_stage("hurdle_single"),
            environment_factory=lambda course: FakeEnvironment(course),
        )
        self.assertEqual(result["controller_mode"], "student_only")
        self.assertFalse(result["teacher_module_loaded"])
        self.assertEqual(result["teacher_interventions"], 0)
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(result["evaluation_episodes"], 2)

    def test_frozen_seed_manifest_is_disjoint_and_covers_all_positions(self):
        manifest = load_seed_manifest()
        self.assertEqual(manifest.stage, "hurdle_single")
        self.assertEqual(len(manifest.sha256), 64)
        sets = [set(manifest.for_split(split)) for split in ("train", "validation", "holdout")]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        for split in ("train", "validation", "holdout"):
            positions = {
                sample_curriculum_course(seed, "hurdle_single", split)
                .obstacles[0]
                .start_x
                for seed in manifest.for_split(split)
            }
            self.assertEqual(positions, set(range(20, 31)))

    def test_collection_and_evaluation_parsers_have_distinct_roles(self):
        collection_options = {
            option
            for action in build_collection_parser()._actions
            for option in action.option_strings
        }
        evaluation_options = {
            option
            for action in build_evaluation_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            collection_options - {"--rescue-profile"},
            evaluation_options,
        )
        self.assertIn("--rescue-profile", collection_options)
        self.assertNotIn("--rescue-profile", evaluation_options)
        self.assertNotIn("--split", collection_options)
        self.assertNotIn("--split", evaluation_options)


if __name__ == "__main__":
    unittest.main()
