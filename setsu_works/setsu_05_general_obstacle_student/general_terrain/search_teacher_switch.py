"""前半越壁方策から後半方策へ切り替える最良通過率を探索する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from general_terrain.closed_loop_height1_teacher import (
    ClosedLoopHeight1Teacher,
    run_teacher_episode,
)
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """候補切替率を全十一壁位置で検査し成功集合の合併を求める。"""
    parser = argparse.ArgumentParser(description="搜索越墙前后半动作的切换比例。")
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.25, 0.33, 0.4, 0.6, 0.67, 0.75, 0.85],
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "training_only_teacher" / "generated" / "switch_fraction_audit.json"),
    )
    parser.add_argument("--positions", nargs="+", type=int, default=list(range(20, 31)))
    parser.add_argument("--handoff-distances", nargs="+", type=float, default=[0.45])
    parser.add_argument("--robust-flat-model")
    args = parser.parse_args()

    rows = []
    coverage = {position: [] for position in args.positions}
    for handoff_distance in args.handoff_distances:
        for fraction in args.fractions:
            teacher = ClosedLoopHeight1Teacher(
                post_clear_mode="restart_then_flat",
                clearance_blend=1.0,
                handoff_distance=handoff_distance,
                adaptive_handoff=True,
                first_switch_fraction=fraction,
                robust_flat_model_path=(
                    Path(args.robust_flat_model) if args.robust_flat_model else None
                ),
            )
            episodes = []
            for index, position in enumerate(args.positions):
                course = build_course(
                    ["low_hurdle"],
                    split="teacher_switch_search",
                    seed=82_000 + index,
                    difficulty=1,
                    start_runway_voxels=position,
                )
                episode = run_teacher_episode(
                    teacher,
                    course,
                    seed=83_000 + index,
                    render=False,
                )
                episodes.append(episode)
                if episode["course_complete"]:
                    coverage[position].append(
                        {
                            "first_switch_fraction": fraction,
                            "handoff_distance": handoff_distance,
                        }
                    )
            successful = [
                int(episode["start_runway_voxels"])
                for episode in episodes
                if episode["course_complete"]
            ]
            rows.append(
                {
                    "first_switch_fraction": fraction,
                    "handoff_distance": handoff_distance,
                    "successful_positions": successful,
                    "episodes": episodes,
                }
            )
            print(
                json.dumps(
                    {
                        "fraction": fraction,
                        "handoff_distance": handoff_distance,
                        "successful_positions": successful,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    result = {
        "rows": rows,
        "coverage": {str(key): value for key, value in coverage.items()},
        "uncovered_positions": [key for key, value in coverage.items() if not value],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["uncovered_positions"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
