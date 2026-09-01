"""最終統一環境で旧段階の5k刻みモデルを再評価する。"""

from __future__ import annotations

import argparse
import csv

import numpy as np
from stable_baselines3 import PPO

from src.environment import ENVIRONMENT_VERSIONS, MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION
from src.experiment import EVALUATION_FIELDS, PROJECT_DIR, append_evaluation, evaluate_ppo


def parse_args():
    parser = argparse.ArgumentParser(description="统一重评旧阶段检查点。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--environment-version",
        choices=ENVIRONMENT_VERSIONS,
        default=MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
    )
    parser.add_argument("--max-timesteps", type=int)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于0。")
    run_dir = PROJECT_DIR / "runs" / args.run_name
    import json

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    body = np.load(run_dir / "body.npy")
    candidates = [(0, run_dir / "initial_model.zip")]
    for path in (run_dir / "checkpoints").glob("model_*_steps.zip"):
        step = int(path.stem.split("_")[1])
        if step % 5_000 == 0:
            candidates.append((step, path))
    candidates = sorted(candidates)
    if args.max_timesteps is not None:
        candidates = [item for item in candidates if item[0] <= args.max_timesteps]
    if not candidates:
        raise ValueError("没有可重评检查点。")

    safe_version = args.environment_version.replace("/", "_")
    output_path = run_dir / f"reevaluation_{safe_version}_every_5k.csv"
    if output_path.exists():
        output_path.unlink()
    with output_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=EVALUATION_FIELDS).writeheader()
    for index, (step, model_path) in enumerate(candidates, 1):
        model = PPO.load(model_path, device="cpu")
        metrics = evaluate_ppo(
            model,
            body,
            args.episodes,
            config["max_steps"],
            args.seed,
            args.environment_version,
        )
        append_evaluation(output_path, step, metrics)
        print(
            f"{index}/{len(candidates)} step={step}: "
            f"obstacles={metrics['mean_obstacles_cleared']:.2f}, "
            f"max_x={metrics['mean_max_x']:.3f}"
        )
    print(f"统一重评表：{output_path}")


if __name__ == "__main__":
    main()
