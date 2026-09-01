"""5k刻みの指標表と、指標ごとに分離した推移図を生成する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiment import EVALUATION_FIELDS, PROJECT_DIR


COLORS = ("#1f77b4", "#ff7f0e")


def parse_args():
    parser = argparse.ArgumentParser(description="绘制两身体障碍训练趋势。")
    parser.add_argument("--original-run", required=True)
    parser.add_argument("--layered-run", required=True)
    parser.add_argument("--bin-steps", type=int, default=5_000)
    parser.add_argument("--output-dir", default="obstacle_body_comparison")
    return parser.parse_args()


def read_evaluations(run_dir: Path, bin_steps: int) -> list[dict]:
    with (run_dir / "ppo_evaluation_5k.csv").open(newline="", encoding="utf-8") as file:
        rows = []
        for raw in csv.DictReader(file):
            step = int(raw["timesteps"])
            if step != 0 and step % bin_steps:
                continue
            row = {"timesteps": step}
            for name in EVALUATION_FIELDS:
                if name != "timesteps":
                    row[name] = float(raw[name])
            rows.append(row)
    return rows


def read_monitor(run_dir: Path) -> list[dict]:
    path = run_dir / "training.monitor.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        first = file.readline()
        if not first.startswith("#"):
            file.seek(0)
        rows = []
        cumulative_steps = 0
        for raw in csv.DictReader(file):
            length = int(float(raw["l"]))
            cumulative_steps += length
            rows.append(
                {
                    "end_step": cumulative_steps,
                    "return": float(raw["r"]),
                    "length": length,
                    "obstacles_cleared": float(raw.get("obstacles_cleared", 0) or 0),
                    "max_x": float(raw.get("max_x_position", 0) or 0),
                    "success": float(str(raw.get("is_success", "False")).lower() in {"true", "1"}),
                }
            )
    return rows


def aggregate_monitor(rows: list[dict], bin_steps: int) -> list[dict]:
    bins: dict[int, list[dict]] = {}
    for row in rows:
        boundary = int(math.ceil(row["end_step"] / bin_steps) * bin_steps)
        bins.setdefault(boundary, []).append(row)
    output = []
    for boundary, values in sorted(bins.items()):
        output.append(
            {
                "timesteps": boundary,
                "episodes": len(values),
                "mean_training_return": sum(row["return"] for row in values) / len(values),
                "mean_training_obstacles": sum(row["obstacles_cleared"] for row in values) / len(values),
                "mean_training_max_x": sum(row["max_x"] for row in values) / len(values),
                "training_success_rate": sum(row["success"] for row in values) / len(values),
            }
        )
    return output


def load_run(run_name: str, label: str, bin_steps: int) -> dict:
    run_dir = PROJECT_DIR / "runs" / run_name
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return {
        "name": run_name,
        "label": label,
        "config": config,
        "evaluations": read_evaluations(run_dir, bin_steps),
        "training": aggregate_monitor(read_monitor(run_dir), bin_steps),
    }


def write_prefixed_table(path: Path, runs: list[dict], key: str, fields: tuple[str, ...]) -> None:
    step_set = sorted({row["timesteps"] for run in runs for row in run[key]})
    maps = [{row["timesteps"]: row for row in run[key]} for run in runs]
    fieldnames = ["timesteps"]
    for run in runs:
        fieldnames.extend(f"{run['name']}__{field}" for field in fields)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for step in step_set:
            output = {"timesteps": step}
            for run, rows_by_step in zip(runs, maps):
                row = rows_by_step.get(step, {})
                for field in fields:
                    output[f"{run['name']}__{field}"] = row.get(field, "")
            writer.writerow(output)


def save_line_plot(
    output_path: Path,
    runs: list[dict],
    data_key: str,
    value_key: str,
    title: str,
    ylabel: str,
    *,
    percent: bool = False,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for run, color in zip(runs, COLORS):
        rows = run[data_key]
        x = [row["timesteps"] for row in rows]
        y = [100 * row[value_key] if percent else row[value_key] for row in rows]
        axis.plot(x, y, color=color, marker="o", markersize=3.5, linewidth=2, label=run["label"])
    axis.set_title(title, fontsize=15)
    axis.set_xlabel("Training timesteps (evaluated every 5k)")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=190)
    plt.close(figure)


def main():
    args = parse_args()
    if args.bin_steps <= 0:
        raise ValueError("--bin-steps 必须大于0。")
    if Path(args.output_dir).name != args.output_dir:
        raise ValueError("--output-dir 不能包含路径。")
    runs = [
        load_run(args.original_run, "Original body (4 actuators)", args.bin_steps),
        load_run(args.layered_run, "Layered body (10 actuators)", args.bin_steps),
    ]
    output_dir = PROJECT_DIR / "runs" / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_fields = tuple(name for name in EVALUATION_FIELDS if name != "timesteps")
    training_fields = (
        "episodes",
        "mean_training_return",
        "mean_training_obstacles",
        "mean_training_max_x",
        "training_success_rate",
    )
    write_prefixed_table(output_dir / "deterministic_metrics_every_5k.csv", runs, "evaluations", evaluation_fields)
    write_prefixed_table(output_dir / "training_metrics_5k_bins.csv", runs, "training", training_fields)

    save_line_plot(
        output_dir / "01_obstacles_cleared_trend_5k.png",
        runs,
        "evaluations",
        "mean_obstacles_cleared",
        "Obstacle-course capability during PPO training",
        "Mean obstacles cleared (out of 7)",
    )
    save_line_plot(
        output_dir / "02_maximum_progress_trend_5k.png",
        runs,
        "evaluations",
        "mean_max_x",
        "Maximum forward progress during deterministic evaluation",
        "Mean maximum x-position",
    )
    save_line_plot(
        output_dir / "03_evaluation_return_trend_5k.png",
        runs,
        "evaluations",
        "mean_return",
        "Deterministic evaluation return",
        "Mean episode return",
    )
    save_line_plot(
        output_dir / "04_success_rate_trend_5k.png",
        runs,
        "evaluations",
        "success_rate",
        "Full-course completion rate",
        "Success rate (%)",
        percent=True,
    )
    save_line_plot(
        output_dir / "05_stochastic_training_return_5k.png",
        runs,
        "training",
        "mean_training_return",
        "Stochastic training return (5k-step episode bins)",
        "Mean training episode return",
    )
    print(f"5k表格和分离趋势图已保存：{output_dir}")


if __name__ == "__main__":
    main()
