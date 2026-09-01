"""歩行方策と固定壁通過方策を状態機械で組み合わせて評価する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.bodies import BODY_NAMES, make_body
from src.curriculum import CURRICULUM_LEVELS, get_course
from src.experiment import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="评估行走与越墙策略的分层组合。")
    parser.add_argument("--body-name", choices=BODY_NAMES, required=True)
    parser.add_argument("--level", type=int, choices=CURRICULUM_LEVELS, required=True)
    parser.add_argument("--walk-model", type=Path, required=True)
    parser.add_argument("--hurdle-model", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=(0.4, 0.6, 0.8))
    parser.add_argument("--cooldowns", nargs="+", type=int, default=(0, 50, 100))
    parser.add_argument(
        "--post-clear-modes",
        nargs="+",
        choices=("walk", "hurdle"),
        default=("walk", "hurdle"),
    )
    return parser.parse_args()


def evaluate_pair(
    body,
    level,
    walk_model,
    hurdle_model,
    threshold,
    cooldown_steps,
    post_clear_mode,
):
    """一組の切替設定で決定論的な一エピソードを実行する。"""
    course = get_course(level)
    env = make_env(body, level, course.max_steps)
    try:
        obs, info = env.reset(seed=10_000)
        total_reward = 0.0
        previous_cleared = 0
        cooldown = 0
        switch_count = 0
        previous_mode = None
        for steps in range(1, course.max_steps + 1):
            cleared = int(info["obstacles_cleared"])
            if cleared > previous_cleared:
                cooldown = cooldown_steps
                previous_cleared = cleared

            if cooldown > 0:
                mode = post_clear_mode
                cooldown = max(0, cooldown - 1)
            elif cleared >= len(course.obstacles):
                mode = "walk"
            else:
                obstacle = course.obstacles[cleared]
                distance = obstacle.start_x * env.unwrapped.VOXEL_SIZE - info["x_position"]
                mode = "hurdle" if distance <= threshold else "walk"

            model = hurdle_model if mode == "hurdle" else walk_model
            action, _ = model.predict(obs, deterministic=True)
            if previous_mode is not None and mode != previous_mode:
                switch_count += 1
            previous_mode = mode
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return {
            "threshold": threshold,
            "cooldown": cooldown_steps,
            "post_clear_mode": post_clear_mode,
            "return": total_reward,
            "steps": steps,
            "max_x": float(info["max_x_position"]),
            "cleared": int(info["obstacles_cleared"]),
            "success": bool(info["is_success"]),
            "switches": switch_count,
        }
    finally:
        env.close()


def main():
    args = parse_args()
    body = make_body(args.body_name)
    walk_model = PPO.load(args.walk_model, device="cpu")
    hurdle_model = PPO.load(args.hurdle_model, device="cpu")
    results = []
    for threshold in args.thresholds:
        for cooldown in args.cooldowns:
            for post_clear_mode in args.post_clear_modes:
                result = evaluate_pair(
                    body,
                    args.level,
                    walk_model,
                    hurdle_model,
                    threshold,
                    cooldown,
                    post_clear_mode,
                )
                results.append(result)
                print(
                    f"threshold={threshold:.2f} cooldown={cooldown} "
                    f"post={post_clear_mode} cleared={result['cleared']} "
                    f"max_x={result['max_x']:.4f} return={result['return']:.4f} "
                    f"success={result['success']} switches={result['switches']}"
                )
    best = max(results, key=lambda item: (item["cleared"], item["max_x"], item["return"]))
    print(f"best={best}")


if __name__ == "__main__":
    main()
