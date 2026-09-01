"""分離カリキュラムの順序、乱数性、昇格門限を検証する。"""

from __future__ import annotations

import unittest

from general_terrain.curriculum import (
    CURRICULUM_STAGES,
    TEACHER_BELOW_GATE,
    TEACHER_MISSING,
    TEACHER_VERIFIED,
    CurriculumGate,
    get_curriculum_stage,
    sample_curriculum_course,
)
from general_terrain.feasibility import assert_course_feasible
from general_terrain.terrain import MAX_GAP_VOXELS, MIN_GAP_VOXELS


class CurriculumTests(unittest.TestCase):
    """技能が早期混合されず、門限なしで昇格しないことを検査する。"""

    def test_stage_order_and_teacher_readiness_are_explicit(self):
        self.assertEqual(
            [stage.name for stage in CURRICULUM_STAGES],
            [
                "hurdle_single",
                "hurdle_double",
                "platform_single",
                "platform_double",
                "hurdle_platform_mixed",
            ],
        )
        self.assertEqual(get_curriculum_stage("hurdle_single").teacher_status, TEACHER_VERIFIED)
        self.assertEqual(get_curriculum_stage("hurdle_double").teacher_status, TEACHER_BELOW_GATE)
        self.assertEqual(get_curriculum_stage("platform_single").teacher_status, TEACHER_MISSING)

    def test_single_hurdle_never_samples_platform(self):
        for seed in range(50):
            course = sample_curriculum_course(seed, "hurdle_single", "train")
            self.assertEqual(
                [item.template.name for item in course.obstacles],
                ["low_hurdle"],
            )

    def test_double_hurdle_randomizes_only_position_and_gap(self):
        courses = [
            sample_curriculum_course(seed, "hurdle_double", "train")
            for seed in range(50)
        ]
        starts = {course.obstacles[0].start_x for course in courses}
        gaps = {
            course.obstacles[1].start_x - course.obstacles[0].end_x - 1
            for course in courses
        }
        self.assertGreater(len(starts), 1)
        self.assertGreater(len(gaps), 1)
        self.assertGreaterEqual(min(gaps), MIN_GAP_VOXELS)
        self.assertLessEqual(max(gaps), MAX_GAP_VOXELS)
        for course in courses:
            self.assertEqual(
                [item.template.name for item in course.obstacles],
                ["low_hurdle", "low_hurdle"],
            )
            assert_course_feasible(course)

    def test_mixed_stage_contains_both_skills(self):
        for seed in range(20):
            course = sample_curriculum_course(
                seed,
                "hurdle_platform_mixed",
                "validation",
            )
            self.assertEqual(
                {item.template.name for item in course.obstacles},
                {"low_hurdle", "low_platform_short"},
            )

    def test_gate_requires_nine_successes_and_zero_hard_falls(self):
        gate = CurriculumGate()
        self.assertTrue(
            gate.evaluate(
                {
                    "evaluation_episodes": 11,
                    "success_count": 9,
                    "hard_fall_count": 0,
                }
            ).passed
        )
        self.assertFalse(
            gate.evaluate(
                {
                    "evaluation_episodes": 11,
                    "success_count": 8,
                    "hard_fall_count": 0,
                }
            ).passed
        )
        self.assertFalse(
            gate.evaluate(
                {
                    "evaluation_episodes": 11,
                    "success_count": 9,
                    "hard_fall_count": 1,
                }
            ).passed
        )


if __name__ == "__main__":
    unittest.main()
