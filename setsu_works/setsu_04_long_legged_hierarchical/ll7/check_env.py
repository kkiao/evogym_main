"""長脚形状、三身体幅間隔、厳格状態機械を学習前に検証する。"""

from __future__ import annotations

import argparse
import gc

import numpy as np
from evogym import get_full_connectivity, has_actuator, is_connected

from ll7.body import BODY_WIDTH_VOXELS, make_body
from ll7.curriculum import (
    CURRICULUM_LEVELS,
    GAP_VOXELS,
    get_course,
    make_course_array,
    obstacle_empty_gaps,
)
from ll7.environment import LANDING_STABLE_STEPS, LongLeggedCurriculumEnv
from ll7.experiment import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="检查新七级长腿机器人环境。")
    parser.add_argument("--steps", type=int, default=3)
    return parser.parse_args()


def check_state_machine():
    """完全通過、安定着地、再前進の順序を人工状態で確認する。"""
    body = make_body()
    env = LongLeggedCurriculumEnv(
        body,
        level=1,
        connections=get_full_connectivity(body),
    )
    try:
        env.reset(seed=7)
        positions = env.object_pos_at_time(env.get_time(), "robot").copy()
        obstacle_end = (get_course(1).obstacles[0].end_x + 1) * env.VOXEL_SIZE
        positions[0, :] += obstacle_end + 0.02 - float(np.min(positions[0]))
        events, failure = env._update_phase(positions)
        assert not failure
        assert events["new_strict_clearance"]
        assert env._phase == "landing"

        for _ in range(LANDING_STABLE_STEPS):
            events, failure = env._update_phase(positions)
        assert not failure
        assert events["new_stable_landing"]
        assert env._phase == "restart"

        positions[0, :] += BODY_WIDTH_VOXELS * env.VOXEL_SIZE + 0.01
        events, failure = env._update_phase(positions)
        assert not failure
        assert events["new_restart"]
        assert env._validated_count == 1
        assert env._phase == "completed"
    finally:
        env.close()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps 必须大于0。")

    body = make_body()
    assert is_connected(body)
    assert has_actuator(body)
    assert BODY_WIDTH_VOXELS == 5
    check_state_machine()
    print("state_machine=passed")

    for level in CURRICULUM_LEVELS:
        course = get_course(level)
        terrain = make_course_array(level)
        gaps = obstacle_empty_gaps(course)
        assert set(np.unique(terrain)).issubset({0, 5})
        assert all(gap == GAP_VOXELS for gap in gaps)
        env = make_env(body, level, course.max_steps)
        try:
            obs, info = env.reset(seed=7)
            assert obs.shape == env.observation_space.shape
            assert np.isfinite(obs).all()
            for _ in range(args.steps):
                obs, _, terminated, truncated, info = env.step(
                    np.zeros(env.action_space.shape, dtype=np.float32)
                )
                assert np.isfinite(obs).all()
                if terminated or truncated:
                    break
            print(
                f"level={level} obstacles={len(course.obstacles)} "
                f"gaps={list(gaps)} obs={obs.shape} action={env.action_space.shape} "
                f"phase={info['phase']}"
            )
        finally:
            env.close()
            gc.collect()


if __name__ == "__main__":
    main()

