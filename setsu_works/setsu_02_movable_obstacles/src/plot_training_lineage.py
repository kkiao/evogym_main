"""実際のモデル継承経路に沿って二段階の5k推移を連結する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment import MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION
from src.experiment import EVALUATION_FIELDS, PROJECT_DIR


COLORS = ("#1f77b4", "#ff7f0e")


def parse_args():
    parser = argparse.ArgumentParser(description="绘制最终统一标准的训练继承趋势。")
    parser.add_argument("--original-base", required=True)
    parser.add_argument("--original-final", required=True)
    parser.add_argument("--layered-base", required=True)
    parser.add_argument("--layered-final", required=True)
    parser.add_argument("--output-dir", default="final_training_lineage_5k")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as file:
        output = []
        for raw in csv.DictReader(file):
            row = {"timesteps": int(raw["timesteps"])}
            for name in EVALUATION_FIELDS:
                if name != "timesteps":
                    row[name] = float(raw[name])
            output.append(row)
    return output


def lineage(base_name: str, final_name: str, body_label: str) -> list[dict]:
    base_dir = PROJECT_DIR / "runs" / base_name
    final_dir = PROJECT_DIR / "runs" / final_name
    base_summary = json.loads((base_dir / "summary.json").read_text(encoding="utf-8"))
    source_step = int(base_summary["best_timesteps"])
    reevaluation = base_dir / (
        f"reevaluation_{MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION}_every_5k.csv"
    )
    base_rows = [row for row in read_rows(reevaluation) if row["timesteps"] <= source_step]
    final_rows = read_rows(final_dir / "ppo_evaluation_5k.csv")
    output = []
    for row in base_rows:
        output.append(
            {
                **row,
                "body": body_label,
                "phase": "movable_training_v5_rechecked",
                "phase_timesteps": row["timesteps"],
                "lineage_timesteps": row["timesteps"],
            }
        )
    for row in final_rows:
        if row["timesteps"] == 0:
            continue
        output.append(
            {
                **row,
                "body": body_label,
                "phase": "com_clearance_finetune_v6",
                "phase_timesteps": row["timesteps"],
                "lineage_timesteps": source_step + row["timesteps"],
            }
        )
    return output


def save_plot(path: Path, runs: list[tuple[str, list[dict]]], field: str, title: str, ylabel: str):
    figure, axis = plt.subplots(figsize=(10.8, 6.3))
    for (label, rows), color in zip(runs, COLORS):
        axis.plot(
            [row["lineage_timesteps"] for row in rows],
            [row[field] for row in rows],
            color=color,
            marker="o",
            markersize=3.4,
            linewidth=2,
            label=label,
        )
        transition = next(
            (row["lineage_timesteps"] for row in rows if row["phase"] == "com_clearance_finetune_v6"),
            None,
        )
        if transition is not None:
            axis.axvline(transition - 5_000, color=color, linestyle="--", alpha=0.28)
    axis.set_title(title, fontsize=15)
    axis.set_xlabel("PPO timesteps along the inherited policy lineage")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def main():
    args = parse_args()
    if Path(args.output_dir).name != args.output_dir:
        raise ValueError("--output-dir 不能包含路径。")
    original = lineage(args.original_base, args.original_final, "Original body (4 actuators)")
    layered = lineage(args.layered_base, args.layered_final, "Layered body (10 actuators)")
    runs = [("Original body (4 actuators)", original), ("Layered body (10 actuators)", layered)]
    output_dir = PROJECT_DIR / "runs" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = sorted(original + layered, key=lambda row: (row["body"], row["lineage_timesteps"]))
    fieldnames = [
        "body",
        "phase",
        "phase_timesteps",
        "lineage_timesteps",
        *[name for name in EVALUATION_FIELDS if name != "timesteps"],
    ]
    with (output_dir / "final_metrics_every_5k_long.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    save_plot(
        output_dir / "01_final_obstacles_cleared_every_5k.png",
        runs,
        "mean_obstacles_cleared",
        "Final obstacle capability along the inherited PPO training path",
        "Mean obstacles cleared (out of 7)",
    )
    save_plot(
        output_dir / "02_final_max_progress_every_5k.png",
        runs,
        "mean_max_x",
        "Maximum forward progress under one consistent final environment",
        "Mean maximum x-position",
    )
    save_plot(
        output_dir / "03_final_return_every_5k.png",
        runs,
        "mean_return",
        "PPO return under one consistent final environment",
        "Mean episode return",
    )
    print(f"最终继承路径5k表格和图表：{output_dir}")


if __name__ == "__main__":
    main()
