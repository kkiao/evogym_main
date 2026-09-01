"""Level 2無側倒再最適化の5k推移と最終段階監査図を生成する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "submission_assets" / "level_2_true_no_side_fall_v2" / "charts"
ANALYSIS_PATH = PROJECT_ROOT / "analysis" / "level2_true_no_side_fall_v2_seed10000.json"

COURSES = (
    (
        "Direct 33% attempt",
        PROJECT_ROOT / "runs" / "true_noroll_v2_l2_wall2_to33_seed7_v32" / "second_crossing_evaluation_5k.csv",
    ),
    (
        "Safe 20%",
        PROJECT_ROOT / "runs" / "true_noroll_v2_l2_wall2_to20_seed7_v33" / "second_crossing_evaluation_5k.csv",
    ),
    (
        "Early handoff 33%",
        PROJECT_ROOT / "runs" / "true_noroll_v2_l2_wall2_early50_to33_seed7_v36" / "second_crossing_evaluation_5k.csv",
    ),
    (
        "33% to 50%",
        PROJECT_ROOT / "runs" / "true_noroll_v2_l2_wall2_33to50_seed7_v37" / "second_crossing_evaluation_5k.csv",
    ),
    (
        "50% to full",
        PROJECT_ROOT / "runs" / "true_noroll_v2_l2_wall2_50to100_seed7_v38" / "second_crossing_evaluation_5k.csv",
    ),
)


def read_rows(path):
    """5k評価CSVを数値辞書列として読み込む。"""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def configure_axes(ax, title, ylabel):
    """提出図で共通する目盛線と見出しを設定する。"""
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Training steps within each course")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def save_fraction_chart(course_rows):
    """各分割課程の通過割合を5k間隔の独立曲線で保存する。"""
    figure, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for label, rows in course_rows:
        ax.plot(
            [row["timesteps"] for row in rows],
            [100.0 * row["mean_fraction"] for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    for target in (20, 33.333, 50, 100):
        ax.axhline(target, color="gray", linestyle="--", linewidth=0.8, alpha=0.45)
    configure_axes(ax, "Level 2 curriculum progress (evaluated every 5k)", "Maximum crossed fraction (%)")
    figure.savefig(OUTPUT_DIR / "01_fraction_trend_every_5k.png", dpi=180)
    plt.close(figure)


def save_angle_chart(course_rows):
    """各分割課程の最大姿勢角を5k間隔の独立曲線で保存する。"""
    figure, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for label, rows in course_rows:
        ax.plot(
            [row["timesteps"] for row in rows],
            [row["mean_maximum_degrees"] for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.axhline(65.0, color="red", linestyle="--", linewidth=1.2, label="Training safety limit (65 deg)")
    configure_axes(ax, "Orientation trend by curriculum course (evaluated every 5k)", "Maximum orientation (deg)")
    figure.savefig(OUTPUT_DIR / "02_orientation_trend_every_5k.png", dpi=180)
    plt.close(figure)


def save_stage_audit(metrics):
    """旧接近段と本輪最適化段の接地回数を分離して可視化する。"""
    stage_quality = metrics["stage_quality"]
    stages = list(stage_quality)
    contacts = [stage_quality[name]["contact_steps"] for name in stages]
    colors = [
        "#9e9e9e" if name in {"first_approach", "first_to_50"} else "#2a9d8f"
        for name in stages
    ]
    labels = [name.replace("_", "\n") for name in stages]
    figure, ax = plt.subplots(figsize=(12.5, 6.2), constrained_layout=True)
    bars = ax.bar(labels, contacts, color=colors)
    ax.set_title("Final stage audit: torso-near-ground frames", fontsize=13, weight="bold")
    ax.set_ylabel("Detected frames")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    for bar, value in zip(bars, contacts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center", va="bottom", fontsize=8)
    ax.text(
        0.99,
        0.95,
        "Gray = legacy approach/first-half (not optimized in this round)\nGreen = optimized obstacle-control stages",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    figure.savefig(OUTPUT_DIR / "03_final_stage_contact_audit.png", dpi=180)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    course_rows = [(label, read_rows(path)) for label, path in COURSES]
    metrics = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    save_fraction_chart(course_rows)
    save_angle_chart(course_rows)
    save_stage_audit(metrics)
    readme = (
        "# Level 2 无侧倒优化图表\n\n"
        "- `01_fraction_trend_every_5k.png`：各分级课程每 5k 的越障比例。\n"
        "- `02_orientation_trend_every_5k.png`：各分级课程每 5k 的最大倾角。\n"
        "- `03_final_stage_contact_audit.png`：最终控制链逐阶段接触审计；灰色为未纳入本轮的旧接近段，绿色为本轮验收段。\n"
    )
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"charts=3 output={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
