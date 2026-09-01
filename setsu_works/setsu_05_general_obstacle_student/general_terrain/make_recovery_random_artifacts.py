"""高さ一教師門とランダム二障害初期訓練の検収資料を生成する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEACHER_SUMMARY = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_teacher_search"
    / "noisy_teacher_portfolio_seed7_v1"
    / "summary.json"
)
RANDOM_CSV_FILES = (
    PROJECT_ROOT
    / "runs"
    / "random_multi_obstacle_training"
    / "random_d1_two_obstacles_seed7_v1"
    / "evaluation_5k.csv",
    PROJECT_ROOT
    / "runs"
    / "random_multi_obstacle_training"
    / "random_d1_two_obstacles_resume20k_seed7_v2"
    / "evaluation_5k.csv",
)
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "recovery_gate_and_random_d1_seed7"


def read_random_history() -> list[dict[str, float]]:
    """複数回に分かれた継続訓練を累積歩数で一列へ統合する。"""
    rows_by_step: dict[int, dict[str, float]] = {}
    for path in RANDOM_CSV_FILES:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for source in csv.DictReader(handle):
                step = int(source["step"])
                rows_by_step[step] = {
                    "step": step,
                    "success_count": int(source["success_count"]),
                    "hard_fall_count": int(source["hard_fall_count"]),
                    "mean_raw_clearances": float(source["mean_raw_clearances"]),
                    "mean_recovered_obstacles": float(
                        source["mean_recovered_obstacles"]
                    ),
                    "mean_max_x": float(source["mean_max_x"]),
                }
    return [rows_by_step[key] for key in sorted(rows_by_step)]


def save_combined_csv(rows: list[dict[str, float]]) -> None:
    """五千歩間隔の統合指標を再利用可能なCSVで保存する。"""
    path = OUTPUT_DIR / "random_d1_evaluation_5k_combined.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_capability(rows: list[dict[str, float]]) -> None:
    """通過能力指標だけを安全指標から分離して描画する。"""
    steps = [item["step"] for item in rows]
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    axis.plot(
        steps,
        [item["mean_raw_clearances"] for item in rows],
        marker="o",
        linewidth=2.0,
        label="Mean raw clearances / 2",
    )
    axis.plot(
        steps,
        [item["mean_recovered_obstacles"] for item in rows],
        marker="s",
        linewidth=2.0,
        label="Mean stable recoveries / 2",
    )
    axis.plot(
        steps,
        [item["success_count"] / 11.0 for item in rows],
        marker="^",
        linewidth=2.0,
        label="Course-completion rate",
    )
    axis.axvline(20_000, color="#666666", linestyle="--", alpha=0.7)
    axis.annotate(
        "capability peak",
        xy=(20_000, rows[4]["mean_recovered_obstacles"]),
        xytext=(21_500, 0.42),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    axis.set_title("Random two-obstacle capability (5k evaluation)")
    axis.set_xlabel("Cumulative training steps")
    axis.set_ylabel("Count or rate")
    axis.set_ylim(-0.03, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "01_random_capability_5k.png", dpi=180)
    plt.close(figure)


def plot_safety(rows: list[dict[str, float]]) -> None:
    """硬側倒を独立図にして能力低下と混同しないようにする。"""
    steps = [item["step"] for item in rows]
    values = [item["hard_fall_count"] for item in rows]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.plot(steps, values, marker="o", color="#b23a48", linewidth=2.2)
    axis.fill_between(steps, values, color="#b23a48", alpha=0.15)
    axis.set_title("Random two-obstacle hard falls (lower is safer)")
    axis.set_xlabel("Cumulative training steps")
    axis.set_ylabel("Hard falls / 11 courses")
    axis.set_ylim(0, 11)
    axis.set_yticks(range(0, 12))
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "02_random_safety_5k.png", dpi=180)
    plt.close(figure)


def plot_teacher_gate(summary: dict[str, object]) -> None:
    """十一位置に対する教師の成功と安全停止を一覧表示する。"""
    validation = summary["validation"]
    positions = [int(item["position"]) for item in validation]
    success = [int(bool(item["course_complete"])) for item in validation]
    colors = ["#2a9d8f" if item else "#9aa0a6" for item in success]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.bar([str(item) for item in positions], success, color=colors)
    axis.axhline(1.0, color="#333333", linewidth=0.8)
    axis.set_title("Height-1 disturbed teacher gate: 9/11 success, 0 hard falls")
    axis.set_xlabel("Obstacle start position")
    axis.set_ylabel("Successful stable clearance")
    axis.set_ylim(0, 1.15)
    axis.set_yticks([0, 1])
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "00_height1_teacher_gate.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """図表、統合CSV、判定要約を一つの成果物フォルダへ出力する。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    teacher = json.loads(TEACHER_SUMMARY.read_text(encoding="utf-8"))
    history = read_random_history()
    save_combined_csv(history)
    plot_teacher_gate(teacher)
    plot_capability(history)
    plot_safety(history)
    capability_peak = max(
        history,
        key=lambda item: (
            item["mean_recovered_obstacles"],
            item["mean_raw_clearances"],
        ),
    )
    safety_peak = min(
        history,
        key=lambda item: (
            item["hard_fall_count"],
            -item["mean_recovered_obstacles"],
        ),
    )
    report = {
        "height1_disturbed_teacher_gate": {
            "success_count": teacher["success_count"],
            "hard_fall_count": teacher["hard_fall_count"],
            "passed": teacher["robustness_gate_passed"],
            "accepted_branch_count": len(teacher["solutions"]),
        },
        "random_two_obstacle_training": {
            "evaluated_through_step": history[-1]["step"],
            "capability_peak": capability_peak,
            "safety_peak": safety_peak,
            "final": history[-1],
            "full_course_success_observed": any(
                item["success_count"] > 0 for item in history
            ),
            "decision": "split_low_hurdle_then_platform_before_mixed_random",
        },
    }
    (OUTPUT_DIR / "phase_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
