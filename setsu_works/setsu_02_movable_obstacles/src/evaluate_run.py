"""学習前・ランダム行動・最良モデル・最終モデルを統一条件で比較する。"""

from __future__ import annotations

import argparse
import json

import numpy as np
from stable_baselines3 import PPO

from src.environment import LEGACY_ENVIRONMENT_VERSION
from src.experiment import PROJECT_DIR, evaluate_ppo, evaluate_random_actions, write_json_atomic


def parse_args():
    parser = argparse.ArgumentParser(description="统一评估一个多障碍PPO实验。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于0。")
    run_dir = PROJECT_DIR / "runs" / args.run_name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    body = np.load(run_dir / "body.npy")
    initial_model = PPO.load(run_dir / "initial_model.zip", device="cpu")
    best_model = PPO.load(run_dir / "best_model.zip", device="cpu")
    latest_model = PPO.load(run_dir / "latest_model.zip", device="cpu")
    environment_version = config.get("environment_version", LEGACY_ENVIRONMENT_VERSION)
    results = {
        "run_name": args.run_name,
        "algorithm": "PPO",
        "body_name": config["body_name"],
        "course_version": config["course_version"],
        "environment_version": environment_version,
        "evaluation_episodes": args.episodes,
        "max_steps": config["max_steps"],
        "seed": args.seed,
        "best_timesteps": summary["best_timesteps"],
        "latest_completed_timesteps": int(latest_model.num_timesteps),
        "initial_deterministic": evaluate_ppo(
            initial_model,
            body,
            args.episodes,
            config["max_steps"],
            args.seed,
            environment_version,
        ),
        "random_actions": evaluate_random_actions(
            body,
            args.episodes,
            config["max_steps"],
            args.seed + 10_000,
            environment_version,
        ),
        "best_deterministic": evaluate_ppo(
            best_model,
            body,
            args.episodes,
            config["max_steps"],
            args.seed,
            environment_version,
        ),
        "latest_deterministic": evaluate_ppo(
            latest_model,
            body,
            args.episodes,
            config["max_steps"],
            args.seed,
            environment_version,
        ),
    }
    output_path = run_dir / "comparison_evaluation.json"
    write_json_atomic(output_path, results)
    print(f"统一评估已保存：{output_path}")
    for name in (
        "initial_deterministic",
        "random_actions",
        "best_deterministic",
        "latest_deterministic",
    ):
        metrics = results[name]
        print(
            f"{name}: obstacles={metrics['mean_obstacles_cleared']:.2f}/7, "
            f"max_x={metrics['mean_max_x']:.3f}, return={metrics['mean_return']:.3f}, "
            f"success={metrics['success_rate']:.1%}"
        )


if __name__ == "__main__":
    main()
