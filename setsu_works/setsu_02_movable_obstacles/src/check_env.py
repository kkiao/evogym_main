"""二種類の形状と混合障害物環境を、学習せずにスモークテストする。"""

from __future__ import annotations

import argparse

import numpy as np

from src.bodies import BODY_NAMES, make_body
from src.course import OBSTACLES, course_metadata, make_course_array
from src.experiment import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="检查混合障碍环境。")
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps 必须大于0。")
    course = make_course_array()
    metadata = course_metadata()
    print(
        f"course={metadata['version']}, shape={course.shape}, "
        f"obstacles={len(OBSTACLES)}, finish_x={metadata['finish_x']}"
    )
    for body_name in BODY_NAMES:
        body = make_body(body_name)
        env = make_env(body, max_steps=max(args.steps, 2))
        try:
            obs, info = env.reset(seed=7)
            initial_x = info["x_position"]
            for _ in range(args.steps):
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                if terminated or truncated:
                    break
            print(
                f"{body_name}: obs={obs.shape}, action={env.action_space.shape}, "
                f"actuators={int(np.count_nonzero((body == 3) | (body == 4)))}, "
                f"x_delta={info['x_position'] - initial_x:.6f}, "
                f"cleared={info['obstacles_cleared']}, finite={bool(np.all(np.isfinite(obs)))}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
