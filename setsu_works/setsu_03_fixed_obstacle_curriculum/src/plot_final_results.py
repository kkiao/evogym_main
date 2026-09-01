"""三形状の5k評価表、分離推移図、突破イベント図を生成する。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.experiment import PROJECT_ROOT


SERIES = (
    (
        "long_legged",
        "Long-legged (6 actuators)",
        "#2ca02c",
        PROJECT_ROOT
        / "runs"
        / "level6_dense_long_legged_seed7_v7"
        / "reevaluation_final_environment_5k.csv",
    ),
    (
        "original",
        "Original short-legged (4 actuators)",
        "#1f77b4",
        PROJECT_ROOT / "runs" / "level7_adjacent_original_seed7_v8" / "ppo_evaluation_5k.csv",
    ),
    (
        "layered",
        "Layered body (10 actuators)",
        "#ff7f0e",
        PROJECT_ROOT / "runs" / "level7_adjacent_layered_seed7_v8" / "ppo_evaluation_5k.csv",
    ),
)


def read_rows(path: Path):
    """評価CSVを数値辞書として読み込む。"""
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def save_plot(output, datasets, value_key, title, ylabel, percent=False, ylim=None):
    """一つの指標だけを含む比較推移図を保存する。"""
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    for (_, label, color, _), rows in zip(SERIES, datasets):
        x = [row["timesteps"] for row in rows]
        y = [row[value_key] * (100.0 if percent else 1.0) for row in rows]
        axis.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=3.2, label=label)
    axis.set_title(title, fontsize=15)
    axis.set_xlabel("PPO training timesteps (evaluation every 5k)")
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def write_combined_table(path, datasets):
    """三形状の主要指標を一行一時点の比較表へ統合する。"""
    maps = [{int(row["timesteps"]): row for row in rows} for rows in datasets]
    timesteps = sorted({step for mapping in maps for step in mapping})
    metrics = ("mean_return", "mean_max_x", "mean_obstacles_cleared", "success_rate")
    fields = ["timesteps"] + [
        f"{name}__{metric}"
        for name, _, _, _ in SERIES
        for metric in metrics
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in timesteps:
            output = {"timesteps": step}
            for (name, _, _, _), mapping in zip(SERIES, maps):
                row = mapping.get(step, {})
                for metric in metrics:
                    output[f"{name}__{metric}"] = row.get(metric, "")
            writer.writerow(output)


def save_breakthrough_plot(path, long_rows):
    """40k、45k、50kの突破と忘却だけを独立図として示す。"""
    selected = [row for row in long_rows if int(row["timesteps"]) in {40_000, 45_000, 50_000}]
    labels = [f"{int(row['timesteps'] / 1000)}k" for row in selected]
    cleared = [row["mean_obstacles_cleared"] for row in selected]
    colors = ["#9ecae1", "#2ca02c", "#d62728"]
    figure, axis = plt.subplots(figsize=(8.5, 5.6))
    bars = axis.bar(labels, cleared, color=colors, width=0.58)
    axis.set_title("Breakthrough and catastrophic forgetting (special event)", fontsize=14)
    axis.set_xlabel("Long-legged checkpoint")
    axis.set_ylabel("Strictly cleared fixed obstacles")
    axis.set_ylim(0, 2.35)
    axis.grid(axis="y", alpha=0.3)
    for bar, row in zip(bars, selected):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.06,
            f"cleared={row['mean_obstacles_cleared']:.0f}\nreturn={row['mean_return']:.1f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    datasets = [read_rows(path) for _, _, _, path in SERIES]
    output_dir = PROJECT_ROOT / "analysis" / "final_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_combined_table(output_dir / "metrics_every_5k.csv", datasets)
    save_plot(
        output_dir / "01_maximum_progress_trend_5k.png",
        datasets,
        "mean_max_x",
        "Maximum forward progress on the final fixed-obstacle course",
        "Mean maximum x-position",
    )
    save_plot(
        output_dir / "02_obstacles_cleared_trend_5k.png",
        datasets,
        "mean_obstacles_cleared",
        "Strict fixed-obstacle clearance",
        "Mean obstacles cleared (out of 2)",
        ylim=(-0.05, 2.15),
    )
    save_plot(
        output_dir / "03_evaluation_return_trend_5k.png",
        datasets,
        "mean_return",
        "Deterministic PPO evaluation return",
        "Mean episode return",
    )
    save_plot(
        output_dir / "04_success_rate_trend_5k.png",
        datasets,
        "success_rate",
        "Full clearance plus stable-landing success rate",
        "Success rate (%)",
        percent=True,
        ylim=(-3, 103),
    )
    save_breakthrough_plot(output_dir / "05_breakthrough_forgetting_event.png", datasets[0])
    print(f"最终表格和图表：{output_dir}")


if __name__ == "__main__":
    main()
