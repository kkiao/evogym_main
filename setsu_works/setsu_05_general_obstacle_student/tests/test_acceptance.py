"""軌跡検収器が速度停止を要求せず安全回復を判定するか検証する。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from general_terrain.acceptance import AcceptanceConfig, CourseAcceptanceTracker
from general_terrain.terrain import build_course


def make_positions(left_x: float) -> np.ndarray:
    """合成試験用の直立した四点身体座標を返す。"""
    return np.asarray(
        [
            [left_x, left_x + 0.1, left_x + 0.2, left_x + 0.3],
            [0.1, 0.2, 0.3, 0.4],
        ],
        dtype=float,
    )


class AcceptanceTrackerTests(unittest.TestCase):
    """完全通過、回復距離、連続安全姿勢、転倒を検証する。"""

    def setUp(self):
        self.course = build_course(
            ["low_hurdle"],
            split="train",
            seed=1,
            difficulty=1,
        )
        self.config = AcceptanceConfig(
            recovery_stable_steps=20,
            recovery_distance_voxels=5,
        )

    def test_clearance_alone_is_not_accepted(self):
        tracker = CourseAcceptanceTracker(self.course, self.config)
        snapshot = tracker.update(
            make_positions(2.2),
            orientation_error=0.0,
            upper_body_grounded=False,
        )
        self.assertEqual(snapshot.raw_clearances, 1)
        self.assertEqual(snapshot.recovered_obstacles, 0)
        self.assertFalse(snapshot.course_complete)

    def test_continuous_upright_motion_validates_recovery(self):
        tracker = CourseAcceptanceTracker(self.course, self.config)
        tracker.update(
            make_positions(2.2),
            orientation_error=0.0,
            upper_body_grounded=False,
        )
        snapshot = None
        for step in range(1, 21):
            snapshot = tracker.update(
                make_positions(2.2 + 0.03 * step),
                orientation_error=math.radians(10.0),
                upper_body_grounded=False,
            )
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.recovered_obstacles, 1)
        self.assertGreaterEqual(snapshot.current_recovery_distance, 0.5)

    def test_side_fall_resets_safe_streak(self):
        tracker = CourseAcceptanceTracker(self.course, self.config)
        tracker.update(
            make_positions(2.2),
            orientation_error=0.0,
            upper_body_grounded=False,
        )
        for step in range(1, 11):
            tracker.update(
                make_positions(2.2 + 0.03 * step),
                orientation_error=math.radians(10.0),
                upper_body_grounded=False,
            )
        snapshot = tracker.update(
            make_positions(2.55),
            orientation_error=math.radians(60.0),
            upper_body_grounded=False,
        )
        self.assertEqual(snapshot.current_safe_streak, 0)
        self.assertEqual(snapshot.recovered_obstacles, 0)

    def test_persistent_large_angle_is_a_hard_fall(self):
        tracker = CourseAcceptanceTracker(self.course, self.config)
        snapshot = None
        for _ in range(self.config.hard_fall_grace_steps):
            snapshot = tracker.update(
                make_positions(1.0),
                orientation_error=math.radians(85.0),
                upper_body_grounded=False,
            )
        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.hard_fall)
        self.assertEqual(snapshot.failure_reason, "orientation_hard_fall")

    def test_brief_upper_body_contact_is_recoverable(self):
        tracker = CourseAcceptanceTracker(self.course, self.config)
        snapshot = None
        for _ in range(self.config.upper_body_ground_grace_steps - 1):
            snapshot = tracker.update(
                make_positions(1.0),
                orientation_error=math.radians(20.0),
                upper_body_grounded=True,
            )
        self.assertIsNotNone(snapshot)
        self.assertFalse(snapshot.hard_fall)
        snapshot = tracker.update(
            make_positions(1.0),
            orientation_error=math.radians(20.0),
            upper_body_grounded=False,
        )
        self.assertFalse(snapshot.hard_fall)

    def test_persistent_upper_body_contact_is_a_hard_fall(self):
        tracker = CourseAcceptanceTracker(self.course, self.config)
        snapshot = None
        for _ in range(self.config.upper_body_ground_grace_steps):
            snapshot = tracker.update(
                make_positions(1.0),
                orientation_error=math.radians(20.0),
                upper_body_grounded=True,
            )
        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.hard_fall)
        self.assertEqual(snapshot.failure_reason, "upper_body_grounded")

    def test_one_recovery_cannot_validate_two_obstacles(self):
        course = build_course(
            ["low_hurdle", "low_hurdle"],
            split="train",
            seed=2,
            difficulty=1,
            gaps=[15],
        )
        tracker = CourseAcceptanceTracker(course, self.config)
        second_end = (course.obstacles[1].end_x + 1) * self.config.voxel_size
        snapshot = tracker.update(
            make_positions(second_end + 0.1),
            orientation_error=0.0,
            upper_body_grounded=False,
        )
        self.assertTrue(snapshot.sequence_failed)
        self.assertEqual(snapshot.recovered_obstacles, 0)
        self.assertEqual(snapshot.failure_reason, "next_obstacle_before_recovery")


if __name__ == "__main__":
    unittest.main()
