"""主学習段階と最終判定段階を九つの主要GIFへ統合する。"""

from __future__ import annotations

import argparse
import csv
import json

from src.experiment import PROJECT_DIR
from src.render_gif import render_checkpoint
from src.select_gif_nodes import read_available_rows, select_nodes


def parse_args():
    parser = argparse.ArgumentParser(description="生成跨阶段9节点GIF。")
    parser.add_argument("--base-run", required=True, help="可推动障碍主训练运行。")
    parser.add_argument("--final-run", required=True, help="重心通过标准的最终验证运行。")
    parser.add_argument("--pre-run", help="可选：在主阶段前加入真正未训练策略。")
    parser.add_argument(
        "--base-steps",
        nargs="+",
        type=int,
        help="人工审核后的主阶段5k节点；省略时自动选择。",
    )
    parser.add_argument("--label", required=True, help="输出目录中的身体短名。")
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--frame-skip", type=int, default=6)
    parser.add_argument("--fps", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 2:
        raise ValueError("跨阶段序列至少需要2个GIF。")
    base_dir = PROJECT_DIR / "runs" / args.base_run
    final_dir = PROJECT_DIR / "runs" / args.final_run
    pre_count = 1 if args.pre_run else 0
    required_base_count = args.count - 1 - pre_count
    available_base_rows = read_available_rows(base_dir)
    if args.base_steps:
        rows_by_step = {row["timesteps"]: row for row in available_base_rows}
        missing = [step for step in args.base_steps if step not in rows_by_step]
        if missing:
            raise ValueError(f"主阶段缺少这些检查点或评估点：{missing}")
        base_nodes = [
            {**rows_by_step[step], "selection_reason": "manually_reviewed_key_change"}
            for step in args.base_steps
        ]
    else:
        base_nodes = select_nodes(available_base_rows, required_base_count)
    if len(base_nodes) != required_base_count:
        raise ValueError("主训练阶段没有足够的可用检查点。")
    final_summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    output_dir = PROJECT_DIR / "submission_gifs" / f"{args.label}_obstacles_key_{args.count}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    next_stage = 1
    if args.pre_run:
        pre_dir = PROJECT_DIR / "runs" / args.pre_run
        pre_output = output_dir / f"{next_stage:02d}_true_untrained.gif"
        pre_metrics = render_checkpoint(
            run_dir=pre_dir,
            checkpoint="initial",
            checkpoint_step=None,
            output_path=pre_output,
            seed=args.seed,
            max_steps=None,
            frame_skip=args.frame_skip,
            fps=args.fps,
        )
        rows.append(
            {
                "stage": next_stage,
                "source_run": args.pre_run,
                "training_timesteps": 0,
                "selection_reason": "true_untrained_policy",
                "return": pre_metrics["return"],
                "max_x": pre_metrics["max_x"],
                "obstacles_cleared": pre_metrics["obstacles_cleared"],
                "success": pre_metrics["success"],
                "gif_file": pre_output.name,
            }
        )
        print(f"训练前 {next_stage}/{args.count}：{pre_output.name}")
        next_stage += 1

    for node in base_nodes:
        stage = next_stage
        step = int(node["timesteps"])
        output_path = output_dir / f"{stage:02d}_base_step_{step:07d}.gif"
        metrics = render_checkpoint(
            run_dir=base_dir,
            checkpoint="initial" if step == 0 else "latest",
            checkpoint_step=None if step == 0 else step,
            output_path=output_path,
            seed=args.seed,
            max_steps=None,
            frame_skip=args.frame_skip,
            fps=args.fps,
        )
        rows.append(
            {
                "stage": stage,
                "source_run": args.base_run,
                "training_timesteps": step,
                "selection_reason": node["selection_reason"],
                "return": metrics["return"],
                "max_x": metrics["max_x"],
                "obstacles_cleared": metrics["obstacles_cleared"],
                "success": metrics["success"],
                "gif_file": output_path.name,
            }
        )
        print(f"主阶段 {stage}/{args.count}：{output_path.name}")
        next_stage += 1

    final_stage = args.count
    final_output = output_dir / f"{final_stage:02d}_final_best.gif"
    final_metrics = render_checkpoint(
        run_dir=final_dir,
        checkpoint="best",
        checkpoint_step=None,
        output_path=final_output,
        seed=args.seed,
        max_steps=None,
        frame_skip=args.frame_skip,
        fps=args.fps,
    )
    rows.append(
        {
            "stage": final_stage,
            "source_run": args.final_run,
            "training_timesteps": final_summary["best_timesteps"],
            "selection_reason": "final_best_with_com_clearance",
            "return": final_metrics["return"],
            "max_x": final_metrics["max_x"],
            "obstacles_cleared": final_metrics["obstacles_cleared"],
            "success": final_metrics["success"],
            "gif_file": final_output.name,
        }
    )
    print(f"最终阶段 {final_stage}/{args.count}：{final_output.name}")

    with (output_dir / "key_gif_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "README.md").write_text(
        "# 9个关键越障阶段\n\n"
        f"主阶段GIF来自 `{args.base_run}` 的每5k检查点，"
        f"第 {args.count} 个来自 `{args.final_run}` 的最佳模型。"
        "节点按首次达到新障碍数、能力跃迁和时间覆盖选择，并非机械等间隔。\n",
        encoding="utf-8",
    )
    print(f"跨阶段关键GIF目录：{output_dir}")


if __name__ == "__main__":
    main()
