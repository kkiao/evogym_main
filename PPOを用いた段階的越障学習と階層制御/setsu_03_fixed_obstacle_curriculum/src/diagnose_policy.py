"""一つ以上のPPOモデルについて固定障害物付近の詳細挙動を診断する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.bodies import BODY_NAMES, make_body
from src.curriculum import CURRICULUM_LEVELS, get_course
from src.experiment import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="诊断模型的抬升和部分越障程度。")
    parser.add_argument("--body-name", choices=BODY_NAMES, required=True)
    parser.add_argument("--level", type=int, choices=CURRICULUM_LEVELS, required=True)
    parser.add_argument("models", nargs="+", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    body = make_body(args.body_name)
    course = get_course(args.level)
    for path in args.models:
        model = PPO.load(path, device="cpu")
        env = make_env(body, args.level, course.max_steps)
        try:
            obs, info = env.reset(seed=10_000)
            total_reward = 0.0
            clearance_state = None
            for steps in range(1, course.max_steps + 1):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if clearance_state is None and info["obstacles_cleared"]:
                    raw = env.unwrapped
                    positions = raw.object_pos_at_time(raw.get_time(), "robot")
                    velocity = raw.get_vel_com_obs("robot")
                    clearance_state = {
                        "step": steps,
                        "com_x": float(np.mean(positions[0])),
                        "com_y": float(np.mean(positions[1])),
                        "bottom_y": float(np.min(positions[1])),
                        "width": float(np.ptp(positions[0])),
                        "height": float(np.ptp(positions[1])),
                        "velocity_x": float(velocity[0]),
                        "velocity_y": float(velocity[1]),
                    }
                if terminated or truncated:
                    break
            raw = env.unwrapped
            final_positions = raw.object_pos_at_time(raw.get_time(), "robot")
            final_velocity = raw.get_vel_com_obs("robot")
            print(
                f"{path.name}: return={total_reward:.4f} steps={steps} "
                f"max_x={info['max_x_position']:.4f} cleared={info['obstacles_cleared']} "
                f"max_com_y={info['maximum_com_y']:.4f} "
                f"max_bottom_y={info['maximum_bottom_y']:.4f} "
                f"crossed_fraction={raw._maximum_crossed_fraction:.4f} "
                f"crossing_score={raw._maximum_crossing_score:.4f} "
                f"stall={info['stall_steps']}"
            )
            if clearance_state is not None:
                print(
                    "  clearance: "
                    f"step={clearance_state['step']} "
                    f"com=({clearance_state['com_x']:.4f}, {clearance_state['com_y']:.4f}) "
                    f"bottom={clearance_state['bottom_y']:.4f} "
                    f"size=({clearance_state['width']:.4f}, {clearance_state['height']:.4f}) "
                    f"velocity=({clearance_state['velocity_x']:.4f}, "
                    f"{clearance_state['velocity_y']:.4f})"
                )
            print(
                "  final: "
                f"com=({float(np.mean(final_positions[0])):.4f}, "
                f"{float(np.mean(final_positions[1])):.4f}) "
                f"bottom={float(np.min(final_positions[1])):.4f} "
                f"size=({float(np.ptp(final_positions[0])):.4f}, "
                f"{float(np.ptp(final_positions[1])):.4f}) "
                f"velocity=({float(final_velocity[0]):.4f}, "
                f"{float(final_velocity[1]):.4f})"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
