"""提出用の最良平地方策を決定論的に再評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.reinforce import evaluate_random_actions
from src.train_ppo import evaluate_ppo


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """身体、評価回数、乱数種を読み取る。"""
    parser = argparse.ArgumentParser(
        description="提出済みの平地最良方策を教師なしで再評価する。"
    )
    parser.add_argument("--body", choices=("original", "layered"), required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    """保存済み最良方策とランダム行動を同一条件で比較する。"""
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("評価回数は一以上でなければならない。")
    result_dir = PROJECT_ROOT / "results" / args.body
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    body = np.load(result_dir / "body.npy")
    model = PPO.load(
        PROJECT_ROOT / "models" / f"{args.body}_best.zip",
        device="cpu",
    )
    result = {
        "body": args.body,
        "evaluation_episodes": args.episodes,
        "seed": args.seed,
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
        "best_deterministic": evaluate_ppo(
            model,
            body,
            args.episodes,
            int(config["max_steps"]),
            args.seed,
        ),
        "random_actions": evaluate_random_actions(
            str(config["task_name"]),
            body,
            args.episodes,
            int(config["max_steps"]),
            args.seed + 10_000,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
