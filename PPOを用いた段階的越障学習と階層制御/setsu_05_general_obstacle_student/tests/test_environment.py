"""統一環境の観測契約、地図交換、物理スモークを検証する。"""

from __future__ import annotations

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from general_terrain.environment import (
    PRIVILEGED_OBSERVATION_NAMES,
    STALL_TERMINATION_STEPS,
    TERRAIN_SCAN_SIZE,
    GeneralObstacleEnv,
)
from general_terrain.terrain import build_course, sample_course


class GeneralObstacleEnvironmentTests(unittest.TestCase):
    """同一クラスが全区分を扱い、方策へ地図正解を漏らさないか検査する。"""

    def test_gymnasium_contract(self):
        env = GeneralObstacleEnv(
            course=sample_course(11, 3, 3, "train"),
            resample_on_reset=False,
        )
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_schema_contains_no_privileged_fields(self):
        env = GeneralObstacleEnv(
            course=sample_course(12, 3, 3, "train"),
            resample_on_reset=False,
        )
        try:
            schema_text = " ".join(env.schema)
            for forbidden in PRIVILEGED_OBSERVATION_NAMES:
                self.assertNotIn(forbidden, schema_text)
        finally:
            env.close()

    def test_all_splits_share_shape_and_finite_observation(self):
        courses = (
            sample_course(21, 3, 3, "train"),
            sample_course(22, 3, 3, "validation"),
            sample_course(23, 3, 3, "holdout"),
        )
        shapes = set()
        for course in courses:
            env = GeneralObstacleEnv(course=course, resample_on_reset=False)
            try:
                observation, info = env.reset()
                shapes.add(observation.shape)
                self.assertTrue(np.all(np.isfinite(observation)))
                self.assertTrue(env.observation_space.contains(observation))
                self.assertEqual(info["course_split"], course.split)
            finally:
                env.close()
        self.assertEqual(len(shapes), 1)

    def test_reset_can_load_another_course_spec(self):
        first = sample_course(31, 3, 2, "train")
        second = sample_course(32, 3, 4, "holdout")
        env = GeneralObstacleEnv(course=first, resample_on_reset=False)
        try:
            first_observation, first_info = env.reset()
            second_observation, second_info = env.reset(options={"course": second})
            self.assertEqual(first_observation.shape, second_observation.shape)
            self.assertNotEqual(first_info["course_id"], second_info["course_id"])
            self.assertEqual(second_info["course_split"], "holdout")
        finally:
            env.close()

    def test_terrain_scan_distinguishes_local_shapes(self):
        low = build_course(
            ["low_hurdle"],
            split="train",
            seed=41,
            difficulty=1,
        )
        mound = build_course(
            ["triangular_mound"],
            split="train",
            seed=42,
            difficulty=2,
        )
        env = GeneralObstacleEnv(course=low, resample_on_reset=False)
        try:
            low_observation, _ = env.reset()
            mound_observation, _ = env.reset(options={"course": mound})
            self.assertFalse(
                np.allclose(
                    low_observation[-TERRAIN_SCAN_SIZE:],
                    mound_observation[-TERRAIN_SCAN_SIZE:],
                )
            )
        finally:
            env.close()

    def test_neutral_actions_keep_candidate_worlds_numerically_stable(self):
        courses = (
            sample_course(51, 3, 3, "train"),
            sample_course(52, 3, 3, "validation"),
            sample_course(53, 3, 3, "holdout"),
        )
        for course in courses:
            env = GeneralObstacleEnv(course=course, resample_on_reset=False)
            try:
                observation, _ = env.reset()
                action = np.full(env.action_space.shape, -0.2, dtype=np.float32)
                for _ in range(20):
                    observation, _, terminated, truncated, info = env.step(action)
                    self.assertTrue(np.all(np.isfinite(observation)))
                    self.assertFalse(info["simulation_unstable"])
                    if terminated or truncated:
                        break
            finally:
                env.close()

    def test_stationary_neutral_action_has_no_positive_reward_loophole(self):
        course = build_course(
            ["low_hurdle"],
            split="train",
            seed=61,
            difficulty=1,
        )
        env = GeneralObstacleEnv(course=course, resample_on_reset=False)
        try:
            env.reset()
            action = np.full(env.action_space.shape, -0.2, dtype=np.float32)
            rewards = []
            for _ in range(75):
                _, reward, terminated, truncated, _ = env.step(action)
                rewards.append(reward)
                if terminated or truncated:
                    break
            self.assertLess(sum(rewards), 0.0)
        finally:
            env.close()

    def test_stall_limit_uses_distinct_incomplete_reason(self):
        course = build_course(
            ["low_hurdle"],
            split="train",
            seed=63,
            difficulty=1,
        )
        env = GeneralObstacleEnv(course=course, resample_on_reset=False)
        try:
            env.reset()
            env._stall_steps = STALL_TERMINATION_STEPS - 1
            env._last_progress_x = 1_000_000.0
            action = np.full(env.action_space.shape, -0.2, dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertTrue(info["stall_limit_reached"])
            self.assertEqual(info["failure_reason"], "stall_limit")
        finally:
            env.close()

    def test_local_rise_metrics_use_nearby_profile_without_obstacle_labels(self):
        course = build_course(
            ["low_platform_short"],
            split="train",
            seed=62,
            difficulty=1,
            start_runway_voxels=20,
        )
        env = GeneralObstacleEnv(course=course, resample_on_reset=False)
        try:
            positions = np.asarray(
                [
                    [1.55, 1.65, 1.75, 1.85],
                    [0.1, 0.2, 0.3, 0.4],
                ],
                dtype=float,
            )
            rise_key, fraction, rear_progress, _ = env._local_rise_metrics(positions)
            self.assertEqual(rise_key, (20, 22))
            self.assertEqual(fraction, 0.0)
            self.assertEqual(rear_progress, 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
