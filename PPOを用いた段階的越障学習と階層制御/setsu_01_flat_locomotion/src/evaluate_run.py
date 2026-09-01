"""実験の学習前・学習後・ランダム行動の性能を統一条件で比較する。"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch

from src.policy import Policy
from src.reinforce import (
    evaluate_policy,
    evaluate_random_actions,
    evaluate_sampled_policy,
)
from src.train_ppo import evaluate_ppo


PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="统一评估一个 EvoGym 实验。")
    parser.add_argument("--run-name", required=True, help="runs 下的实验目录名。")
    parser.add_argument("--episodes", type=int, default=20, help="每项评估回合数。")
    parser.add_argument("--seed", type=int, default=20000, help="独立测试随机种子。")
    return parser.parse_args()


def load_state_policy(state_dict, observation_size, action_size):
    policy = Policy(observation_size, action_size)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def current_exploration_std(config, completed_episodes):
    progress = min(
        completed_episodes / config["exploration_decay_episodes"],
        1.0,
    )
    return config["exploration_std_start"] + (
        config["exploration_std_end"] - config["exploration_std_start"]
    ) * progress


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于 0。")

    run_dir = PROJECT_DIR / "runs" / args.run_name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    body = np.load(run_dir / "body.npy")
    if config["algorithm"] == "PPO":
        initial_model = PPO.load(run_dir / "initial_model.zip", device="cpu")
        best_model = PPO.load(run_dir / "best_model.zip", device="cpu")
        latest_model = PPO.load(run_dir / "latest_model.zip", device="cpu")
        max_steps = config["max_steps"]
        results = {
            "run_name": args.run_name,
            "algorithm": "PPO",
            "evaluation_episodes": args.episodes,
            "max_steps": max_steps,
            "seed": args.seed,
            "latest_completed_timesteps": latest_model.num_timesteps,
            "initial_deterministic": evaluate_ppo(
                initial_model, body, args.episodes, max_steps, args.seed
            ),
            "best_deterministic": evaluate_ppo(
                best_model, body, args.episodes, max_steps, args.seed
            ),
            "latest_deterministic": evaluate_ppo(
                latest_model, body, args.episodes, max_steps, args.seed
            ),
            "random_actions": evaluate_random_actions(
                config["task_name"], body, args.episodes, max_steps, args.seed
            ),
        }
        save_and_print_results(run_dir, results)
        return

    latest = torch.load(
        run_dir / "latest_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    initial_state = torch.load(
        run_dir / "initial_policy.pt",
        map_location="cpu",
        weights_only=True,
    )
    best_state = torch.load(
        run_dir / "best_policy.pt",
        map_location="cpu",
        weights_only=True,
    )
    observation_size = latest["observation_size"]
    action_size = latest["action_size"]
    initial_policy = load_state_policy(initial_state, observation_size, action_size)
    best_policy = load_state_policy(best_state, observation_size, action_size)
    latest_policy = load_state_policy(
        latest["policy_state_dict"],
        observation_size,
        action_size,
    )
    max_steps = config["max_steps"]
    sampled_std = current_exploration_std(config, latest["completed_episodes"])

    results = {
        "run_name": args.run_name,
        "evaluation_episodes": args.episodes,
        "max_steps": max_steps,
        "seed": args.seed,
        "latest_completed_episodes": latest["completed_episodes"],
        "latest_sampled_exploration_std": sampled_std,
        "initial_deterministic": evaluate_policy(
            config["task_name"], body, initial_policy, args.episodes, max_steps, args.seed
        ),
        "best_deterministic": evaluate_policy(
            config["task_name"], body, best_policy, args.episodes, max_steps, args.seed
        ),
        "latest_deterministic": evaluate_policy(
            config["task_name"], body, latest_policy, args.episodes, max_steps, args.seed
        ),
        "latest_sampled": evaluate_sampled_policy(
            config["task_name"],
            body,
            latest_policy,
            args.episodes,
            max_steps,
            sampled_std,
            args.seed,
        ),
        "random_actions": evaluate_random_actions(
            config["task_name"], body, args.episodes, max_steps, args.seed
        ),
    }
    save_and_print_results(run_dir, results)


def save_and_print_results(run_dir, results):
    output_path = run_dir / "comparison_evaluation.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"统一评估已保存：{output_path}")
    for name in (
        "initial_deterministic",
        "best_deterministic",
        "latest_deterministic",
        "latest_sampled",
        "random_actions",
    ):
        if name not in results:
            continue
        metrics = results[name]
        print(
            f"{name}: return={metrics['mean_return']:.6f} "
            f"± {metrics['std_return']:.6f}, speed={metrics['mean_speed']:.8f}"
        )


if __name__ == "__main__":
    main()
