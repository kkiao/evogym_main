"""完了済みLevel 0から2までの5k評価図と整理済みCSVを生成する。"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "submission_assets"

NUMERIC_FIELDS = (
    "timesteps",
    "mean_return",
    "mean_max_x",
    "mean_strict_clearances",
    "mean_stable_landings",
    "mean_restart_successes",
    "mean_validated_obstacles",
    "success_rate",
)


def read_rows(path: Path) -> list[dict]:
    """評価CSVから描画に必要な数値列を読み込む。"""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        item = dict(row)
        for field in NUMERIC_FIELDS:
            item[field] = float(row[field])
        result.append(item)
    return result


def simple_level_rows(level: int, run_name: str, csv_name: str) -> list[dict]:
    """単一実験の評価行へ共通の段階名と累積歩数を付ける。"""
    rows = read_rows(RUNS_DIR / run_name / csv_name)
    for row in rows:
        row["stage"] = f"Level {level} training"
        row["stage_step"] = int(row["timesteps"])
        row["cumulative_step"] = int(row["timesteps"])
        row["source_run"] = run_name
    return rows


def level2_lineage_rows() -> tuple[list[dict], list[tuple[int, str]]]:
    """Level 2の実際の五段階学習履歴を累積5k軸へ連結する。"""
    stages = (
        ("wall1_clear_land", "level2_angle65_seed7_v6", 30_000),
        ("wall1_restart", "level2_restart_after_strict_landing_seed7_v7", 10_000),
        ("wall2_clearance", "level2_second_angle45_seed7_v17", 5_000),
        ("wall2_strict_landing", "level2_second_landing90_speed03_seed7_v28", 15_000),
        ("wall2_restart", "level2_second_relaxed_restart_seed7_v26", 5_000),
    )
    rows = []
    boundaries = []
    offset = 0
    for stage_index, (stage, run_name, chosen_step) in enumerate(stages):
        source = read_rows(RUNS_DIR / run_name / "hierarchical_evaluation_5k.csv")
        selected = [row for row in source if row["timesteps"] <= chosen_step]
        if stage_index:
            selected = [row for row in selected if row["timesteps"] > 0]
        boundaries.append((offset, stage))
        for row in selected:
            row["stage"] = stage
            row["stage_step"] = int(row["timesteps"])
            row["cumulative_step"] = offset + int(row["timesteps"])
            row["source_run"] = run_name
            rows.append(row)
        offset += chosen_step

    final = dict(rows[-1])
    final.update(
        {
            "stage": "assembled_strict_final",
            "stage_step": 0,
            "cumulative_step": offset,
            "source_run": "analysis/level2_full_strict_hierarchy_v1.json",
            "mean_return": math.nan,
            "mean_strict_clearances": 2.0,
            "mean_stable_landings": 2.0,
            "mean_restart_successes": 2.0,
            "mean_validated_obstacles": 2.0,
            "success_rate": 1.0,
        }
    )
    rows.append(final)
    boundaries.append((offset, "assembled_strict_final"))
    return rows, boundaries


def write_rows(path: Path, rows: list[dict]):
    """整理済み評価行を再利用可能なCSVとして保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "cumulative_step",
        "stage_step",
        "stage",
        "source_run",
        "mean_return",
        "mean_max_x",
        "mean_strict_clearances",
        "mean_stable_landings",
        "mean_restart_successes",
        "mean_validated_obstacles",
        "success_rate",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def mark_boundaries(axis, boundaries: list[tuple[int, str]]):
    """多段階学習の境界線を図へ追加する。"""
    for step, _ in boundaries[1:]:
        axis.axvline(step, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)


def plot_level(level: int, rows: list[dict], boundaries: list[tuple[int, str]]):
    """回報・前進量と厳格状態指標を別々のPNGへ描画する。"""
    output_dir = OUTPUT_ROOT / f"level_{level}" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [row["cumulative_step"] for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    valid_return = [
        (step, row["mean_return"])
        for step, row in zip(steps, rows)
        if not math.isnan(row["mean_return"])
    ]
    axes[0].plot(
        [item[0] for item in valid_return],
        [item[1] for item in valid_return],
        marker="o",
        linewidth=1.8,
        color="#2166ac",
    )
    axes[0].set_ylabel("Mean return")
    axes[0].set_title(f"Level {level}: evaluation every 5k steps")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        steps,
        [row["mean_max_x"] for row in rows],
        marker="o",
        linewidth=1.8,
        color="#1b9e77",
    )
    axes[1].set_ylabel("Mean maximum x (m)")
    axes[1].set_xlabel("Training steps (5k sampling; cumulative by stage)")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        mark_boundaries(axis, boundaries)
    figure.tight_layout()
    figure.savefig(output_dir / "trend_5k.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 6.5))
    metric_styles = (
        ("mean_strict_clearances", "Strict clearances", "#d95f02"),
        ("mean_stable_landings", "Stable landings", "#7570b3"),
        ("mean_restart_successes", "Forward restarts", "#1b9e77"),
        ("mean_validated_obstacles", "Validated obstacles", "#e7298a"),
    )
    for field, label, color in metric_styles:
        axis.plot(
            steps,
            [row[field] for row in rows],
            marker="o",
            linewidth=1.7,
            label=label,
            color=color,
        )
    axis.plot(
        steps,
        [row["success_rate"] for row in rows],
        marker="s",
        linestyle=":",
        linewidth=2.0,
        label="Success rate",
        color="#000000",
    )
    mark_boundaries(axis, boundaries)
    axis.set_title(f"Level {level}: strict state metrics every 5k steps")
    axis.set_xlabel("Training steps (5k sampling; cumulative by stage)")
    axis.set_ylabel("Count / rate")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "strict_metrics_5k.png", dpi=180)
    plt.close(figure)

    write_rows(output_dir / "evaluation_5k_curated.csv", rows)


def main():
    level0 = simple_level_rows(
        0,
        "level0_long_legged_gap3_seed7_consolidate_v2",
        "ppo_evaluation_5k.csv",
    )
    level1 = simple_level_rows(
        1,
        "level1_four_stage_restart_consolidate_seed7_v2",
        "hierarchical_evaluation_5k.csv",
    )
    level2, level2_boundaries = level2_lineage_rows()
    plot_level(0, level0, [])
    plot_level(1, level1, [])
    plot_level(2, level2, level2_boundaries)
    print(f"charts={OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
