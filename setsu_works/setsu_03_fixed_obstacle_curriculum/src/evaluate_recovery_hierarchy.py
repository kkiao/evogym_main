"""越障教師と確率的回復方策を組み合わせた状態機械を評価する。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from src.bodies import BODY_NAMES, make_body
from src.curriculum import CURRICULUM_LEVELS, get_course
from src.experiment import PROJECT_ROOT, make_env, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="评估越障教师与恢复策略的完整组合。")
    parser.add_argument("--body-name", choices=BODY_NAMES, required=True)
    parser.add_argument("--level", type=int, choices=CURRICULUM_LEVELS, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--recovery-model", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=(0.35, 0.45, 0.55, 0.65))
    parser.add_argument("--post-clear-steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output-name", default="recovery_hierarchy_results")
    return parser.parse_args()


def run_episode(
    body,
    level,
    teacher,
    recovery,
    threshold,
    post_clear_steps,
    seed,
):
    """一つの固定乱数種で状態機械を最後まで実行する。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    course = get_course(level)
    env = make_env(body, level, course.max_steps)
    try:
        obs, info = env.reset(seed=seed)
        previous_cleared = 0
        post_clear_remaining = 0
        switch_count = 0
        previous_mode = None
        total_reward = 0.0
        for steps in range(1, course.max_steps + 1):
            cleared = int(info["obstacles_cleared"])
            if cleared > previous_cleared:
                post_clear_remaining = post_clear_steps
                previous_cleared = cleared

            if cleared == 0 or post_clear_remaining > 0:
                mode = "teacher"
                post_clear_remaining = max(0, post_clear_remaining - 1)
            elif cleared >= len(course.obstacles):
                mode = "recovery"
            else:
                obstacle = course.obstacles[cleared]
                obstacle_x = obstacle.start_x * env.unwrapped.VOXEL_SIZE
                distance = obstacle_x - float(info["x_position"])
                mode = "teacher" if distance <= threshold else "recovery"

            if mode == "teacher":
                action, _ = teacher.predict(obs, deterministic=True)
            else:
                action, _ = recovery.predict(obs, deterministic=False)
            if previous_mode is not None and mode != previous_mode:
                switch_count += 1
            previous_mode = mode

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return {
            "seed": seed,
            "threshold": threshold,
            "steps": steps,
            "return": total_reward,
            "max_x": float(info["max_x_position"]),
            "cleared": int(info["obstacles_cleared"]),
            "success": bool(info["is_success"]),
            "switches": switch_count,
        }
    finally:
        env.close()


def summarize(rows):
    """切替距離ごとの通過率と到達位置を集約する。"""
    summaries = []
    for threshold in sorted({row["threshold"] for row in rows}):
        selected = [row for row in rows if row["threshold"] == threshold]
        summaries.append(
            {
                "threshold": threshold,
                "episodes": len(selected),
                "two_wall_clear_rate": float(np.mean([row["cleared"] >= 2 for row in selected])),
                "success_rate": float(np.mean([row["success"] for row in selected])),
                "mean_max_x": float(np.mean([row["max_x"] for row in selected])),
                "best_max_x": float(np.max([row["max_x"] for row in selected])),
                "mean_return": float(np.mean([row["return"] for row in selected])),
            }
        )
    return summaries


def main():
    args = parse_args()
    body = make_body(args.body_name)
    teacher = PPO.load(args.teacher_model, device="cpu")
    recovery = PPO.load(args.recovery_model, device="cpu")
    rows = []
    for threshold in args.thresholds:
        for episode in range(args.episodes):
            row = run_episode(
                body,
                args.level,
                teacher,
                recovery,
                threshold,
                args.post_clear_steps,
                30_000 + episode,
            )
            rows.append(row)
            print(
                f"threshold={threshold:.2f} seed={row['seed']} "
                f"cleared={row['cleared']} max_x={row['max_x']:.4f} "
                f"success={row['success']}",
                flush=True,
            )

    output_dir = PROJECT_ROOT / "analysis"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"{args.output_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summaries = summarize(rows)
    write_json(
        output_dir / f"{args.output_name}.json",
        {
            "body_name": args.body_name,
            "level": args.level,
            "post_clear_steps": args.post_clear_steps,
            "teacher_model": str(args.teacher_model.resolve()),
            "recovery_model": str(args.recovery_model.resolve()),
            "summaries": summaries,
        },
    )
    for item in summaries:
        print(f"summary={item}", flush=True)


if __name__ == "__main__":
    main()
