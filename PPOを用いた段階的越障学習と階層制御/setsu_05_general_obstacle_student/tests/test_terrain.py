"""手続き型場の能力包絡、再現性、区分分離を検証する。"""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from general_terrain.feasibility import assert_course_feasible, assert_template_feasible, course_errors, template_errors
from general_terrain.terrain import CATALOG, MAX_START_RUNWAY_VOXELS, MIN_GAP_VOXELS, START_RUNWAY_VOXELS, build_course, make_course_array, sample_course


class TerrainFeasibilityTests(unittest.TestCase):
    """全候補障害物と標本化コースの静的契約を検証する。"""

    def test_catalog_is_inside_capability_envelope(self):
        for template in CATALOG.values():
            assert_template_feasible(template)

    def test_height_three_is_rejected(self):
        invalid = replace(CATALOG["low_hurdle"], name="too_high", heights=(3,))
        self.assertTrue(template_errors(invalid))

    def test_vertical_high_wall_cannot_be_wide(self):
        invalid = replace(
            CATALOG["narrow_high_hurdle"],
            name="too_wide_vertical",
            heights=(2, 2, 2),
        )
        self.assertTrue(template_errors(invalid))

    def test_random_courses_are_reproducible(self):
        first = sample_course(77, 3, 6, "train")
        second = sample_course(77, 3, 6, "train")
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_random_courses_vary_first_obstacle_position(self):
        starts = {
            sample_course(seed, 1, 1, "train").obstacles[0].start_x
            for seed in range(20)
        }
        self.assertGreater(len(starts), 1)
        self.assertGreaterEqual(min(starts), START_RUNWAY_VOXELS)
        self.assertLessEqual(max(starts), MAX_START_RUNWAY_VOXELS)

    def test_random_courses_pass_feasibility(self):
        cases = (
            (101, 1, 4, "train"),
            (102, 2, 5, "train"),
            (103, 3, 6, "train"),
            (104, 3, 4, "validation"),
            (105, 3, 4, "holdout"),
        )
        for seed, difficulty, count, split in cases:
            assert_course_feasible(sample_course(seed, difficulty, count, split))

    def test_too_close_obstacles_are_rejected(self):
        course = build_course(
            ["low_hurdle", "low_platform_short"],
            split="train",
            seed=1,
            difficulty=1,
            gaps=[MIN_GAP_VOXELS - 1],
        )
        self.assertTrue(course_errors(course))

    def test_ground_is_continuous_and_obstacles_are_static(self):
        course = sample_course(88, 3, 6, "train")
        array = make_course_array(course)
        self.assertTrue(np.all(array[-1] == 5))
        self.assertTrue(np.all(np.isin(array, (0, 5))))

    def test_dataset_splits_are_disjoint(self):
        names = {
            split: {item.name for item in CATALOG.values() if item.split == split}
            for split in ("train", "validation", "holdout")
        }
        self.assertFalse(names["train"] & names["validation"])
        self.assertFalse(names["train"] & names["holdout"])
        self.assertFalse(names["validation"] & names["holdout"])


if __name__ == "__main__":
    unittest.main()
