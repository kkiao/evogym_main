"""保存済みPPOチェックポイントを同一の最終環境で5k刻みに再評価する。"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from stable_baselines3 import PPO

from src.bodies import make_body
from src.curriculum import CURRICULUM_LEVELS, get_course
from src.experiment import EVALUATION_FIELDS, RUNS_DIR, evaluate_ppo


def parse_args():
    parser = argparse.ArgumentParser(description="在统一最终环境中重新评估所有检查点。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--body-name", required=True)
    parser.add_argument("--level", type=int, choices=CURRICULUM_LEVELS, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=40_000)
    parser.add_argument("--output", default="reevaluation_final_environment_5k.csv")
    return parser.parse_args()


def checkpoint_paths(run_dir: Path):
    """初期モデルと利用可能な5kチェックポイントを時系列で返す。"""
    paths = [(0, run_dir / "initial_model.zip")]
    pattern = re.compile(r"model_(\d+)_steps\.zip$")
    for path in (run_dir / "checkpoints").glob("model_*_steps.zip"):
        match = pattern.fullmatch(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return sorted(paths)


def main():
    args = parse_args()
    run_dir = RUNS_DIR / args.run_name
    body = make_body(args.body_name)
    course = get_course(args.level)
    output_path = run_dir / args.output
    rows = []
    for timesteps, path in checkpoint_paths(run_dir):
        model = PPO.load(path, device="cpu")
        metrics = evaluate_ppo(
            model,
            body,
            args.level,
            args.episodes,
            course.max_steps,
            args.seed,
        )
        row = {"timesteps": timesteps, **metrics}
        rows.append(row)
        print(
            f"step={timesteps} return={metrics['mean_return']:.4f} "
            f"max_x={metrics['mean_max_x']:.4f} "
            f"cleared={metrics['mean_obstacles_cleared']:.2f} "
            f"success={metrics['success_rate']:.2f}",
            flush=True,
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"统一重评表：{output_path}", flush=True)


if __name__ == "__main__":
    main()
