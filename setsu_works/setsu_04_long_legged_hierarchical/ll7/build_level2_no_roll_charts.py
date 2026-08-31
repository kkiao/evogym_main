"""Level 2無横転最適化の5k推移図と品質図を別々に生成する。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
ASSET_DIR = PROJECT_ROOT / "submission_assets" / "level_2_no_roll"
CHART_DIR = ASSET_DIR / "charts"
GIF_DIR = ASSET_DIR / "gifs"


STAGES = (
    ("wall1_speed_6", "quality_l2_clearance_speed6_seed7_v5", "upright", 30_000),
    ("wall1_speed_4.5", "quality_l2_clearance_speed45_seed7_v6", "upright", 30_000),
    ("wall1_speed_3.5_a", "quality_l2_clearance_speed35_seed7_v9", "upright", 20_000),
    ("wall1_speed_3.5_b", "quality_l2_clearance_speed35_seed7_v10", "upright", 25_000),
    ("wall1_speed_3.2", "quality_l2_clearance_speed32_angle45_seed7_v12", "upright", 20_000),
    ("wall1_land_95", "quality_l2_wall1_speed32_land95_seed7_v15", "hier", 20_000),
    ("wall1_land_90", "quality_l2_wall1_speed32_land90_seed7_v16", "hier", 30_000),
    ("wall1_peak_90", "quality_l2_wall1_speed32_peakreward90_seed7_v18", "hier", 25_000),
    ("wall1_peak_89", "quality_l2_wall1_speed32_peakreward89_seed7_v19", "hier", 20_000),
    ("wall1_peak_88", "quality_l2_wall1_speed32_peakreward88_seed7_v20", "hier", 25_000),
    ("wall2_fraction", "quality_l2_second_fraction50_noroll_seed7_v23", "hier", 5_000),
    ("wall2_upright", "quality_l2_second_angle75_speed60_noroll_seed7_v24", "hier", 20_000),
    ("wall2_brake_0.5", "quality_l2_second_brake_speed05_seed7_v26", "hier", 35_000),
    ("wall2_brake_0.3", "quality_l2_second_brake_speed03_seed7_v27", "hier", 30_000),
)


def read_csv(path: Path) -> list[dict]:
    """UTF-8評価CSVを辞書行として読み込む。"""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str, default: float = math.nan) -> float:
    """欠損列を許容して数値へ変換する。"""
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def build_rows():
    """採用した各サブ課程を累積5k軸へ連結する。"""
    rows = []
    boundaries = []
    offset = 0
    for stage_index, (stage, run_name, kind, chosen_step) in enumerate(STAGES):
        csv_name = (
            "upright_clearance_evaluation_5k.csv"
            if kind == "upright"
            else "hierarchical_evaluation_5k.csv"
        )
        source = read_csv(RUNS_DIR / run_name / csv_name)
        selected = [row for row in source if number(row, "timesteps") <= chosen_step]
        if stage_index:
            selected = [row for row in selected if number(row, "timesteps") > 0]
        boundaries.append((offset, stage))
        for source_row in selected:
            stage_step = int(number(source_row, "timesteps"))
            if kind == "upright":
                strict = number(source_row, "clear_rate", 0.0)
                stable = restart = validated = 0.0
                clearance_angle = number(source_row, "mean_clearance_angle_deg")
                clearance_speed = number(source_row, "mean_clearance_speed")
            else:
                strict = number(source_row, "mean_strict_clearances", 0.0)
                stable = number(source_row, "mean_stable_landings", 0.0)
                restart = number(source_row, "mean_restart_successes", 0.0)
                validated = number(source_row, "mean_validated_obstacles", 0.0)
                clearance_angle = clearance_speed = math.nan
            rows.append(
                {
                    "cumulative_step": offset + stage_step,
                    "stage_step": stage_step,
                    "stage": stage,
                    "source_run": run_name,
                    "mean_return": number(source_row, "mean_return"),
                    "mean_max_x": number(source_row, "mean_max_x"),
                    "strict_clearances": strict,
                    "stable_landings": stable,
                    "restart_successes": restart,
                    "validated_obstacles": validated,
                    "success_rate": number(source_row, "success_rate", 0.0),
                    "clearance_angle_degrees": clearance_angle,
                    "clearance_speed": clearance_speed,
                }
            )
        offset += chosen_step
    final = dict(rows[-1])
    final.update(
        {
            "cumulative_step": offset,
            "stage_step": 0,
            "stage": "assembled_no_roll_final",
            "source_run": "analysis/quality_level2_full_noroll_with_old_restart.json",
            "mean_return": math.nan,
            "strict_clearances": 2.0,
            "stable_landings": 2.0,
            "restart_successes": 2.0,
            "validated_obstacles": 2.0,
            "success_rate": 1.0,
            "clearance_angle_degrees": 46.58,
            "clearance_speed": 1.55,
        }
    )
    rows.append(final)
    boundaries.append((offset, "assembled_no_roll_final"))
    return rows, boundaries


def write_rows(rows):
    """整理済み5k評価を再利用可能なCSVへ保存する。"""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / "evaluation_5k_curated.csv"
    fields = tuple(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mark_boundaries(axis, boundaries):
    """各サブ課程の開始位置を薄い縦線で示す。"""
    for step, _ in boundaries[1:]:
        axis.axvline(step, color="#888888", linestyle="--", linewidth=0.6, alpha=0.45)


def plot_training_trend(rows, boundaries):
    """回報と到達距離だけを連続的な学習推移図へ描く。"""
    steps = [row["cumulative_step"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    valid_return = [
        (row["cumulative_step"], row["mean_return"])
        for row in rows
        if not math.isnan(row["mean_return"])
    ]
    axes[0].plot(
        [item[0] for item in valid_return],
        [item[1] for item in valid_return],
        marker="o",
        markersize=3,
        linewidth=1.6,
        color="#2166ac",
    )
    axes[0].set_title("Level 2 no-roll optimization: evaluation every 5k steps")
    axes[0].set_ylabel("Mean return")
    axes[1].plot(
        steps,
        [row["mean_max_x"] for row in rows],
        marker="o",
        markersize=3,
        linewidth=1.6,
        color="#1b9e77",
    )
    axes[1].set_ylabel("Mean maximum x (m)")
    axes[1].set_xlabel("Cumulative training steps along selected curriculum")
    for axis in axes:
        axis.grid(alpha=0.25)
        mark_boundaries(axis, boundaries)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "training_trend_5k.png", dpi=190)
    plt.close(figure)


def plot_task_progress(rows, boundaries):
    """厳格状態機械の四計数を学習推移とは別の図へ描く。"""
    steps = [row["cumulative_step"] for row in rows]
    figure, axis = plt.subplots(figsize=(12, 6.5))
    styles = (
        ("strict_clearances", "Strict clearances", "#d95f02"),
        ("stable_landings", "Stable landings", "#7570b3"),
        ("restart_successes", "Forward restarts", "#1b9e77"),
        ("validated_obstacles", "Validated obstacles", "#e7298a"),
    )
    for field, label, color in styles:
        axis.plot(
            steps,
            [row[field] for row in rows],
            marker="o",
            markersize=3,
            linewidth=1.6,
            label=label,
            color=color,
        )
    axis.plot(
        steps,
        [row["success_rate"] for row in rows],
        linestyle=":",
        linewidth=2.0,
        color="#000000",
        label="Success rate",
    )
    mark_boundaries(axis, boundaries)
    axis.set_title("Level 2 no-roll optimization: strict task progress every 5k steps")
    axis.set_xlabel("Cumulative training steps along selected curriculum")
    axis.set_ylabel("Count / rate")
    axis.set_ylim(-0.08, 2.15)
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "task_progress_5k.png", dpi=190)
    plt.close(figure)


def plot_quality_nodes():
    """九つのGIF JSONから姿勢品質だけを独立した図へ描く。"""
    paths = sorted(GIF_DIR.glob("[0-9][0-9]_*.json"))
    if len(paths) != 9:
        raise RuntimeError(f"关键节点JSON数量应为9，实际为{len(paths)}。")
    labels = [path.stem[:2] for path in paths]
    maximum_angles = []
    wall2_angles = []
    wall2_speeds = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        maximum_angles.append(float(data["maximum_orientation_degrees"]))
        event = next(
            (item for item in data["strict_events"] if int(item["count"]) == 2),
            None,
        )
        wall2_angles.append(float(event["angle_degrees"]) if event else math.nan)
        wall2_speeds.append(float(event["speed"]) if event else math.nan)
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(labels, maximum_angles, marker="o", color="#d73027", linewidth=1.8)
    axes[0].axhline(90.0, color="#000000", linestyle="--", label="Side-roll boundary (90 deg)")
    axes[0].set_ylabel("Episode max angle (deg)")
    axes[0].legend()
    axes[1].plot(labels, wall2_angles, marker="o", color="#4575b4", linewidth=1.8)
    axes[1].axhline(45.0, color="#777777", linestyle="--", label="45 deg reference")
    axes[1].set_ylabel("Wall 2 clear angle (deg)")
    axes[1].legend()
    axes[2].plot(labels, wall2_speeds, marker="o", color="#1a9850", linewidth=1.8)
    axes[2].set_ylabel("Wall 2 clear speed")
    axes[2].set_xlabel("Nine key GIF nodes")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Level 2 no-roll posture quality at selected key nodes")
    figure.tight_layout()
    figure.savefig(CHART_DIR / "posture_quality_key_nodes.png", dpi=190)
    plt.close(figure)


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    rows, boundaries = build_rows()
    write_rows(rows)
    plot_training_trend(rows, boundaries)
    plot_task_progress(rows, boundaries)
    plot_quality_nodes()
    print(f"charts={CHART_DIR}", flush=True)


if __name__ == "__main__":
    main()
