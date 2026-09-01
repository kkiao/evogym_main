"""二壁課程で第一壁と第二壁の教師設定対を段階的に探索する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from general_terrain.search_double_hurdle_teacher import (
    OUTPUT_ROOT,
    POSITIONS,
    SequentialHeight1Teacher,
    run_episode,
)
from general_terrain.search_noisy_teacher_portfolio import configurations


def main() -> None:
    """第一壁回復候補だけを残し、その候補へ第二壁設定を組み合わせる。"""
    parser = argparse.ArgumentParser(description="分阶段搜索連続する二つの低壁教师参数对。")
    parser.add_argument("--run-name", default="double_hurdle_pairs_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-successes", type=int, default=9)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    parser.add_argument("--candidate-limit", type=int, default=18)
    parser.add_argument("--resume-summary")
    parser.add_argument("--robust-second", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "branches").mkdir()
    first_candidates = [
        item for item in configurations() if not bool(item["robust_flat"])
    ][: args.candidate_limit]
    second_candidates = (
        [item for item in configurations() if bool(item["robust_flat"])]
        if args.robust_second
        else first_candidates
    )
    solutions: dict[int, dict[str, object]] = {}
    if args.resume_summary:
        previous = json.loads(Path(args.resume_summary).read_text(encoding="utf-8"))
        solutions = {
            int(position): item for position, item in previous["solutions"].items()
        }
    first_rows = []
    pair_rows = []
    for position in POSITIONS:
        if len(solutions) >= args.target_successes:
            break
        if position in solutions:
            continue
        episode_seed = args.seed + 91_000 + (position - POSITIONS[0])
        for first_index, first_configuration in enumerate(first_candidates):
            probe_teacher = SequentialHeight1Teacher(
                first_configuration,
                first_configuration,
            )
            first_result, _ = run_episode(
                probe_teacher,
                position=position,
                seed=episode_seed,
                noise_std=args.noise_std,
                noise_probability=args.noise_probability,
                save_trajectory=False,
                stop_after_recovered=1,
            )
            first_rows.append(
                {
                    "position": position,
                    "first_index": first_index,
                    "configuration": first_configuration,
                    "result": first_result,
                }
            )
            first_safe = bool(
                first_result["recovered_obstacles"] >= 1
                and not first_result["hard_fall"]
            )
            if not first_safe:
                continue
            print(
                json.dumps(
                    {
                        "position": position,
                        "safe_first_index": first_index,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            for second_index, second_configuration in enumerate(second_candidates):
                teacher = SequentialHeight1Teacher(
                    first_configuration,
                    second_configuration,
                )
                result, _ = run_episode(
                    teacher,
                    position=position,
                    seed=episode_seed,
                    noise_std=args.noise_std,
                    noise_probability=args.noise_probability,
                    save_trajectory=False,
                )
                pair_rows.append(
                    {
                        "position": position,
                        "first_index": first_index,
                        "second_index": second_index,
                        "result": result,
                    }
                )
                if result["course_complete"] and not result["hard_fall"]:
                    solutions[position] = {
                        "first_index": first_index,
                        "second_index": second_index,
                        "first_configuration": first_configuration,
                        "second_configuration": second_configuration,
                    }
                    print(
                        json.dumps(
                            {
                                "solved_position": position,
                                "first_index": first_index,
                                "second_index": second_index,
                                "solved_count": len(solutions),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    break
            if position in solutions:
                break
    validation = []
    for position in POSITIONS:
        solution = solutions.get(position)
        teacher = (
            SequentialHeight1Teacher(
                solution["first_configuration"],
                solution["second_configuration"],
            )
            if solution
            else None
        )
        episode_seed = args.seed + 91_000 + (position - POSITIONS[0])
        result, trajectory = run_episode(
            teacher,
            position=position,
            seed=episode_seed,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
            save_trajectory=solution is not None,
        )
        validation.append(result)
        if trajectory is not None and result["course_complete"] and not result["hard_fall"]:
            import numpy as np

            np.savez_compressed(
                output_dir / "branches" / f"x{position}_double_hurdle.npz",
                **trajectory,
            )
    success_count = sum(item["course_complete"] for item in validation)
    hard_fall_count = sum(item["hard_fall"] for item in validation)
    summary = {
        "course": "two_low_hurdles_gap_three_body_widths",
        "solutions": {str(key): value for key, value in solutions.items()},
        "validation": validation,
        "success_count": success_count,
        "hard_fall_count": hard_fall_count,
        "curriculum_gate_passed": bool(
            success_count >= args.target_successes and hard_fall_count == 0
        ),
        "first_stage_evaluations": len(first_rows),
        "pair_evaluations": len(pair_rows),
    }
    (output_dir / "first_rows.json").write_text(
        json.dumps(first_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pair_rows.json").write_text(
        json.dumps(pair_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
