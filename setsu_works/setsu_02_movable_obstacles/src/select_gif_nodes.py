"""5k刻みの評価点から能力変化が最も明確な九つのGIF節点を選ぶ。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.experiment import PROJECT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="选择关键GIF检查点。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--count", type=int, default=9)
    return parser.parse_args()


def read_available_rows(run_dir: Path) -> list[dict]:
    available = {0}
    for path in (run_dir / "checkpoints").glob("model_*_steps.zip"):
        try:
            available.add(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    with (run_dir / "ppo_evaluation_5k.csv").open(newline="", encoding="utf-8") as file:
        rows = [dict(row) for row in csv.DictReader(file)]
    numeric_fields = (
        "mean_return",
        "mean_max_x",
        "mean_obstacles_cleared",
        "mean_obstacle_fraction",
        "success_rate",
    )
    output = []
    for row in rows:
        row["timesteps"] = int(row["timesteps"])
        if row["timesteps"] not in available:
            continue
        for field in numeric_fields:
            row[field] = float(row[field])
        output.append(row)
    return sorted(output, key=lambda row: row["timesteps"])


def select_nodes(rows: list[dict], count: int) -> list[dict]:
    if count <= 0:
        raise ValueError("--count 必须大于0。")
    if len(rows) <= count:
        return [{**row, "selection_reason": "all_available"} for row in rows]

    selected: dict[int, dict] = {}

    def add(row, reason):
        current = selected.get(row["timesteps"])
        if current is None:
            selected[row["timesteps"]] = {**row, "selection_reason": reason}
        elif reason not in current["selection_reason"]:
            current["selection_reason"] += f";{reason}"

    add(rows[0], "untrained_baseline")
    global_best = max(
        rows,
        key=lambda row: (
            row["success_rate"],
            row["mean_obstacles_cleared"],
            row["mean_max_x"],
            row["mean_return"],
        ),
    )
    add(global_best, "best_capability")
    add(rows[-1], "latest_regular_checkpoint")

    # 初めて新しい通過数へ到達した点を、最も明確な能力の節目とする。
    highest_level = 0
    for row in rows[1:]:
        level = int(row["mean_obstacles_cleared"])
        if level > highest_level:
            add(row, f"first_reached_{level}_obstacles")
            highest_level = level

    # 隣接評価点間の能力変化で不足分を補い、機械的な等間隔抽出を避ける。
    jumps = []
    previous = rows[0]
    for row in rows[1:]:
        delta_obstacles = row["mean_obstacles_cleared"] - previous["mean_obstacles_cleared"]
        delta_x = row["mean_max_x"] - previous["mean_max_x"]
        delta_success = row["success_rate"] - previous["success_rate"]
        magnitude = 5.0 * abs(delta_obstacles) + abs(delta_x) + 8.0 * abs(delta_success)
        jumps.append((magnitude, row, delta_obstacles, delta_x))
        previous = row
    for _, row, delta_obstacles, delta_x in sorted(jumps, key=lambda item: item[0], reverse=True):
        if len(selected) >= count:
            break
        add(row, f"large_change_dobs={delta_obstacles:.2f}_dx={delta_x:.3f}")

    # 曲線がほぼ不変なら、既存節点から時間的に最も離れた点で診断系列を補う。
    while len(selected) < count:
        remaining = [row for row in rows if row["timesteps"] not in selected]
        if not remaining:
            break
        chosen = max(
            remaining,
            key=lambda row: min(
                abs(row["timesteps"] - step) for step in selected
            ),
        )
        add(chosen, "fills_largest_timeline_gap")

    if len(selected) > count:
        mandatory_steps = {rows[0]["timesteps"], global_best["timesteps"], rows[-1]["timesteps"]}
        ordered = sorted(selected.values(), key=lambda row: row["timesteps"])
        removable = [row for row in ordered if row["timesteps"] not in mandatory_steps]
        while len(ordered) > count and removable:
            victim = min(
                removable,
                key=lambda row: (
                    row["mean_obstacles_cleared"],
                    row["success_rate"],
                    row["mean_max_x"],
                ),
            )
            ordered.remove(victim)
            removable.remove(victim)
        return ordered
    return sorted(selected.values(), key=lambda row: row["timesteps"])


def save_selection(run_dir: Path, selected: list[dict]) -> None:
    json_path = run_dir / "selected_gif_nodes.json"
    json_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = list(selected[0].keys()) if selected else ["timesteps", "selection_reason"]
    with (run_dir / "selected_gif_nodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)


def main():
    args = parse_args()
    run_dir = PROJECT_DIR / "runs" / args.run_name
    rows = read_available_rows(run_dir)
    if not rows:
        raise ValueError("没有同时具备模型文件和评估数据的检查点。")
    selected = select_nodes(rows, args.count)
    save_selection(run_dir, selected)
    for index, row in enumerate(selected, 1):
        print(
            f"{index:02d}: step={row['timesteps']}, "
            f"obstacles={row['mean_obstacles_cleared']:.2f}, "
            f"max_x={row['mean_max_x']:.3f}, reason={row['selection_reason']}"
        )


if __name__ == "__main__":
    main()
