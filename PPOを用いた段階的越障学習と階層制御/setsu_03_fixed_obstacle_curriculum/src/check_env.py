"""三形状・八段階・厳格通過判定を学習前に検証する。"""

from __future__ import annotations

import argparse
import gc

import numpy as np
from evogym import get_full_connectivity, has_actuator, is_connected

from src.bodies import BODY_NAMES, make_body
from src.curriculum import CURRICULUM_LEVELS, get_course, make_course_array
from src.environment import FixedCurriculumEnv
from src.experiment import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="检查三个身体和六级固定课程环境。")
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def check_strict_clearance():
    """一つでも身体点が壁の手前なら未通過になることを確認する。"""
    body = make_body("original")
    env = FixedCurriculumEnv(
        body,
        level=1,
        connections=get_full_connectivity(body),
    )
    try:
        positions = env.object_pos_at_time(env.get_time(), "robot").copy()
        obstacle_end = (get_course(1).obstacles[0].end_x + 1) * env.VOXEL_SIZE
        positions[0, :] = obstacle_end + 0.01
        assert env._count_cleared(positions) == 1
        positions[0, 0] = obstacle_end - 0.01
        assert env._count_cleared(positions) == 0
    finally:
        env.close()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps 必须大于0。")

    check_strict_clearance()
    print("strict_clearance=passed")
    for level in CURRICULUM_LEVELS:
        course = get_course(level)
        terrain = make_course_array(level)
        assert set(np.unique(terrain)).issubset({0, 5})
        print(
            f"level={level} name={course.name} obstacles={len(course.obstacles)} "
            f"finish_x={course.finish_x} max_steps={course.max_steps}"
        )
        for body_name in BODY_NAMES:
            body = make_body(body_name)
            assert is_connected(body)
            assert has_actuator(body)
            env = make_env(body, level, course.max_steps)
            try:
                obs, info = env.reset(seed=7)
                assert np.isfinite(obs).all()
                for _ in range(args.steps):
                    obs, _, terminated, truncated, info = env.step(
                        np.zeros(env.action_space.shape, dtype=np.float32)
                    )
                    assert np.isfinite(obs).all()
                    if terminated or truncated:
                        break
                print(
                    f"  {body_name}: obs={obs.shape} action={env.action_space.shape} "
                    f"finite={bool(np.isfinite(obs).all())} cleared={info['obstacles_cleared']}"
                )
            finally:
                env.close()
                gc.collect()


if __name__ == "__main__":
    main()
