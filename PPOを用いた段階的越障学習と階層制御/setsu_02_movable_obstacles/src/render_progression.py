"""一つの形状について九つの主要能力節点を選択して描画する。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.experiment import PROJECT_DIR
from src.render_gif import render_checkpoint
from src.select_gif_nodes import read_available_rows, save_selection, select_nodes


def parse_args():
    parser = argparse.ArgumentParser(description="生成9阶段障碍训练GIF。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--fps", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = PROJECT_DIR / "runs" / args.run_name
    selected = select_nodes(read_available_rows(run_dir), args.count)
    if not selected:
        raise ValueError("没有可渲染检查点。")
    save_selection(run_dir, selected)
    output_dir = PROJECT_DIR / "submission_gifs" / f"{args.run_name}_key_{len(selected)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for index, row in enumerate(selected, 1):
        step = int(row["timesteps"])
        output_path = output_dir / f"{index:02d}_step_{step:07d}.gif"
        metrics = render_checkpoint(
            run_dir=run_dir,
            checkpoint="initial" if step == 0 else "latest",
            checkpoint_step=None if step == 0 else step,
            output_path=output_path,
            seed=args.seed,
            max_steps=None,
            frame_skip=args.frame_skip,
            fps=args.fps,
        )
        metric_rows.append(
            {
                "stage": index,
                "timesteps": step,
                "selection_reason": row["selection_reason"],
                "evaluation_mean_return": row["mean_return"],
                "evaluation_mean_max_x": row["mean_max_x"],
                "evaluation_mean_obstacles_cleared": row["mean_obstacles_cleared"],
                "evaluation_success_rate": row["success_rate"],
                "gif_return": metrics["return"],
                "gif_max_x": metrics["max_x"],
                "gif_obstacles_cleared": metrics["obstacles_cleared"],
                "gif_success": metrics["success"],
                "gif_file": output_path.name,
            }
        )
        print(f"已生成 {index}/{len(selected)}：{output_path.name}")
    with (output_dir / "key_gif_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)
    print(f"关键GIF目录：{output_dir}")


if __name__ == "__main__":
    main()
