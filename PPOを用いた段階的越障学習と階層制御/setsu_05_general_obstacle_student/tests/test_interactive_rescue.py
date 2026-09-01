"""連続救援、安定解放、成功分岐だけの保存を検証する。"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from general_terrain.interactive_rescue import (
    STUDENT_CONTROLLER,
    TEACHER_CONTROLLER,
    InteractiveRescueController,
    RescueConfig,
    SuccessfulRescueBuffer,
    local_terrain_is_visible,
)
from general_terrain.rescue_profiles import get_rescue_profile
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher


def make_info(**overrides) -> dict[str, object]:
    """救援状態機械用の安全な基準情報を返す。"""
    info: dict[str, object] = {
        "orientation_error": 0.0,
        "angular_velocity": 0.0,
        "stall_steps": 0,
        "upper_body_grounded": False,
        "x_position": 1.0,
        "raw_clearances": 0,
        "recovered_obstacles": 0,
        "obstacle_count": 1,
        "course_complete": False,
        "hard_fall": False,
        "course_id": "test_course",
    }
    info.update(overrides)
    return info


class InteractiveRescueTests(unittest.TestCase):
    """教師が危険時だけ連続制御し、失敗データを残さないか検査する。"""

    def setUp(self):
        self.student_action = np.zeros(6, dtype=np.float32)
        self.teacher_action = np.ones(6, dtype=np.float32)
        self.config = RescueConfig(
            disagreement_streak_steps=2,
            minimum_teacher_steps=2,
            release_safe_steps=2,
            release_progress=0.1,
        )

    def test_orientation_danger_starts_continuous_teacher_control(self):
        controller = InteractiveRescueController(self.config)
        start = controller.decide(
            make_info(orientation_error=self.config.entry_orientation),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        continued = controller.decide(
            make_info(orientation_error=self.config.warning_orientation),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        self.assertEqual(start.controller, TEACHER_CONTROLLER)
        self.assertEqual(start.event, "start")
        self.assertEqual(continued.controller, TEACHER_CONTROLLER)
        self.assertEqual(continued.event, "continue")

    def test_release_requires_safe_streak_and_forward_progress(self):
        controller = InteractiveRescueController(self.config)
        controller.decide(
            make_info(orientation_error=self.config.entry_orientation),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        first_safe = controller.decide(
            make_info(x_position=1.2),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        released = controller.decide(
            make_info(x_position=1.2),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        self.assertEqual(first_safe.controller, TEACHER_CONTROLLER)
        self.assertEqual(released.controller, STUDENT_CONTROLLER)
        self.assertEqual(released.event, "release")

    def test_disagreement_needs_visible_terrain_and_consecutive_steps(self):
        controller = InteractiveRescueController(self.config)
        hidden = controller.decide(
            make_info(),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        first = controller.decide(
            make_info(stall_steps=self.config.disagreement_minimum_stall_steps),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=True,
        )
        second = controller.decide(
            make_info(stall_steps=self.config.disagreement_minimum_stall_steps),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=True,
        )
        self.assertEqual(hidden.controller, STUDENT_CONTROLLER)
        self.assertEqual(first.controller, STUDENT_CONTROLLER)
        self.assertEqual(second.controller, TEACHER_CONTROLLER)
        self.assertEqual(second.reason, "teacher_disagreement")

    def test_global_disagreement_can_enter_before_terrain_is_visible(self):
        """明示された全局分岐だけが地形不可視時の早期接管を許可する。"""
        config = replace(
            self.config,
            disagreement_requires_local_terrain=False,
            disagreement_minimum_stall_steps=0,
        )
        controller = InteractiveRescueController(config)
        decisions = [
            controller.decide(
                make_info(),
                self.student_action,
                self.teacher_action,
                local_terrain_visible=False,
            )
            for _ in range(config.disagreement_streak_steps)
        ]
        self.assertEqual(decisions[0].controller, STUDENT_CONTROLLER)
        self.assertEqual(decisions[-1].controller, TEACHER_CONTROLLER)
        self.assertEqual(decisions[-1].reason, "teacher_disagreement")

    def test_student_prefix_budget_allows_exactly_one_student_action(self):
        """一歩上限は初歩を学生へ渡し次歩で教師を連続接管させる。"""
        config = replace(
            self.config,
            maximum_student_prefix_steps=1,
            disagreement_threshold=2.0,
        )
        controller = InteractiveRescueController(config)
        first = controller.decide(
            make_info(),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        second = controller.decide(
            make_info(),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        self.assertEqual(first.controller, STUDENT_CONTROLLER)
        self.assertEqual(second.controller, TEACHER_CONTROLLER)
        self.assertEqual(second.reason, "student_prefix_budget")

    def test_terrain_visibility_uses_only_observation_schema(self):
        schema = (
            "com_velocity_x",
            "relative_terrain_height_+0",
            "relative_terrain_height_+1",
            "previous_action_0",
        )
        flat = np.asarray([0.0, 0.2, 0.2, 0.0], dtype=np.float32)
        raised = np.asarray([0.0, 0.2, 0.1, 0.0], dtype=np.float32)
        self.assertFalse(local_terrain_is_visible(flat, schema))
        self.assertTrue(local_terrain_is_visible(raised, schema))

    def test_terrain_visibility_respects_preventive_window(self):
        schema = (
            "relative_terrain_height_+0",
            "relative_terrain_height_+6",
            "relative_terrain_height_+7",
        )
        observation = np.asarray([0.2, 0.2, 0.1], dtype=np.float32)
        self.assertFalse(
            local_terrain_is_visible(
                observation,
                schema,
                maximum_rise_offset=6,
            )
        )
        self.assertTrue(
            local_terrain_is_visible(
                observation,
                schema,
                maximum_rise_offset=7,
            )
        )

    def test_preventive_profile_can_enter_without_stall(self):
        config = get_rescue_profile("m2_1_near6")
        controller = InteractiveRescueController(config)
        decisions = [
            controller.decide(
                make_info(stall_steps=0),
                self.student_action,
                self.teacher_action,
                local_terrain_visible=True,
            )
            for _ in range(config.disagreement_streak_steps)
        ]
        self.assertTrue(
            all(item.controller == STUDENT_CONTROLLER for item in decisions[:-1])
        )
        self.assertEqual(decisions[-1].controller, TEACHER_CONTROLLER)
        self.assertEqual(decisions[-1].reason, "teacher_disagreement")

    def test_preventive_danger_does_not_enter_before_local_window(self):
        config = get_rescue_profile("m2_1_near6")
        decision = InteractiveRescueController(config).decide(
            make_info(orientation_error=config.entry_orientation),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        self.assertEqual(decision.controller, STUDENT_CONTROLLER)

    def test_pre_recovery_rescue_cannot_release_on_progress_alone(self):
        config = replace(
            get_rescue_profile("m2_1_near6"),
            minimum_teacher_steps=2,
            release_safe_steps=2,
        )
        controller = InteractiveRescueController(config)
        controller.decide(
            make_info(orientation_error=config.entry_orientation),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=True,
        )
        progress_only = [
            controller.decide(
                make_info(x_position=1.3),
                self.student_action,
                self.teacher_action,
                local_terrain_visible=False,
            )
            for _ in range(2)
        ]
        self.assertTrue(
            all(item.controller == TEACHER_CONTROLLER for item in progress_only)
        )
        first_recovered = controller.decide(
            make_info(x_position=1.3, recovered_obstacles=1),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        released = controller.decide(
            make_info(x_position=1.3, recovered_obstacles=1),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        self.assertEqual(first_recovered.controller, TEACHER_CONTROLLER)
        self.assertEqual(released.event, "release")

    def test_post_recovery_stall_has_distinct_reason(self):
        config = get_rescue_profile("m2_1_near6")
        decision = InteractiveRescueController(config).decide(
            make_info(
                recovered_obstacles=1,
                stall_steps=config.post_recovery_stall_steps,
            ),
            self.student_action,
            self.teacher_action,
            local_terrain_visible=False,
        )
        self.assertEqual(decision.controller, TEACHER_CONTROLLER)
        self.assertEqual(decision.reason, "post_recovery_stall")

    def test_recovered_episode_routes_teacher_to_flat_finish(self):
        class FlatController:
            """平地動作の呼出しだけを記録する。"""

            def __init__(self) -> None:
                self.calls = 0

            def predict_flat(self, environment) -> np.ndarray:
                """平地呼出し回数を増やし零動作を返す。"""
                del environment
                self.calls += 1
                return np.zeros(6, dtype=np.float32)

        controller = FlatController()
        teacher = PortfolioHeight1Teacher.__new__(PortfolioHeight1Teacher)
        teacher.controller = controller
        teacher.profile = "early_direct"
        teacher.phase = "prefix"
        teacher.robust_flat_model = None
        action, stage = teacher.predict(
            object(),
            np.zeros(95, dtype=np.float32),
            make_info(recovered_obstacles=1),
        )
        self.assertEqual(stage, "early_direct:flat_finish")
        self.assertEqual(controller.calls, 1)
        np.testing.assert_array_equal(action, np.zeros(6, dtype=np.float32))

    def test_buffer_rejects_failed_or_teacher_free_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.npz"
            buffer = SuccessfulRescueBuffer()
            decision = InteractiveRescueController(self.config).decide(
                make_info(),
                self.student_action,
                self.teacher_action,
                local_terrain_visible=False,
            )
            buffer.append(
                np.zeros(95, dtype=np.float32),
                self.student_action,
                self.teacher_action,
                self.student_action,
                decision,
                "flat",
            )
            self.assertFalse(buffer.commit(path, make_info(course_complete=True, recovered_obstacles=1)))
            self.assertFalse(path.exists())

    def test_buffer_rejects_teacher_rescue_that_did_not_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed_branch.npz"
            controller = InteractiveRescueController(self.config)
            decision = controller.decide(
                make_info(orientation_error=self.config.entry_orientation),
                self.student_action,
                self.teacher_action,
                local_terrain_visible=False,
            )
            buffer = SuccessfulRescueBuffer()
            buffer.append(
                np.zeros(95, dtype=np.float32),
                self.student_action,
                self.teacher_action,
                self.teacher_action,
                decision,
                "first_to_50",
            )
            accepted = buffer.commit(
                path,
                make_info(course_complete=False, hard_fall=True),
            )
            self.assertFalse(accepted)
            self.assertFalse(path.exists())

    def test_buffer_commits_only_successful_rescue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.npz"
            controller = InteractiveRescueController(self.config)
            decision = controller.decide(
                make_info(orientation_error=self.config.entry_orientation),
                self.student_action,
                self.teacher_action,
                local_terrain_visible=False,
            )
            buffer = SuccessfulRescueBuffer()
            buffer.append(
                np.zeros(95, dtype=np.float32),
                self.student_action,
                self.teacher_action,
                self.teacher_action,
                decision,
                "first_to_50",
            )
            accepted = buffer.commit(
                path,
                make_info(course_complete=True, recovered_obstacles=1),
                metadata={"split": "test"},
            )
            self.assertTrue(accepted)
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".json").exists())
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(arrays["observations"].shape, (1, 95))
                self.assertTrue(bool(arrays["teacher_mask"][0]))


if __name__ == "__main__":
    unittest.main()
