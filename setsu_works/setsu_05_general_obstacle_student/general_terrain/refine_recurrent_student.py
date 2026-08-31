"""成功示範を高く保ちながら循環学生へ保守的DAgger修正を加える。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.train_recurrent_dagger_student import (
    POSITIONS,
    Sequence,
    collect_sequences,
    evaluate_student,
    make_course,
    recurrent_behavior_clone,
    save_round_sequences,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = (
    PROJECT_ROOT
    / "runs"
    / "height1_recurrent_dagger_student"
    / "height1_recurrent_dagger_seed7_v1"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "height1_recurrent_dagger_refinement"


def load_teacher_sequences(path: Path) -> list[Sequence]:
    """純教師反復の全十一位置配列を順序付きで読み込む。"""
    sequences = []
    for position in POSITIONS:
        data = np.load(path / f"x{position}.npz")
        sequences.append(
            (
                np.asarray(data["observations"], dtype=np.float32),
                np.asarray(data["actions"], dtype=np.float32),
            )
        )
    return sequences


def append_progress(path: Path, row: dict[str, object]) -> None:
    """保守的修正の採否をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def score(result: dict[str, object]) -> tuple[float, float, float]:
    """完走、越壁、安全の優先順で比較可能な三項組を返す。"""
    return (
        float(result["success_rate"]),
        float(result["clearance_rate"]),
        -float(result["hard_fall_rate"]),
    )


def main() -> None:
    """各候補を最良方策から作り、悪化時は自動的に採用を見送る。"""
    parser = argparse.ArgumentParser(description="保守地修正循环DAgger学生。")
    parser.add_argument("--run-name", default="height1_recurrent_refine_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--teacher-weight", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "datasets").mkdir()
    (output_dir / "checkpoints").mkdir()
    teacher_sequences = load_teacher_sequences(SOURCE_RUN / "datasets" / "round_00")
    source_model = SOURCE_RUN / "checkpoints" / "round_01.zip"
    environment = GeneralObstacleEnv(
        course=make_course(20, args.seed, "recurrent_refinement_model"),
        resample_on_reset=False,
    )
    try:
        best_model = RecurrentPPO.load(source_model, env=environment, device="cpu")
        best_result = evaluate_student(best_model, seed=args.seed + 50_000)
        best_score = score(best_result)
        best_model.save(output_dir / "best_model")
        history = [
            {
                "round": 0,
                "accepted": True,
                "score": list(best_score),
                "evaluation": best_result,
            }
        ]
        print(
            json.dumps(
                {"round": 0, "accepted": True, "score": list(best_score)},
                ensure_ascii=False,
            ),
            flush=True,
        )

        for round_index in range(1, args.rounds + 1):
            candidate = RecurrentPPO.load(
                output_dir / "best_model.zip",
                env=environment,
                device="cpu",
            )
            correction_sequences, collection = collect_sequences(
                candidate,
                seed=args.seed + 51_000 + round_index * 100,
                beta=args.beta,
            )
            save_round_sequences(
                output_dir / "datasets" / f"round_{round_index:02d}",
                correction_sequences,
            )
            training_sequences = teacher_sequences * args.teacher_weight
            training_sequences.extend(correction_sequences)
            losses = recurrent_behavior_clone(
                candidate,
                training_sequences,
                epochs=1,
                learning_rate=args.learning_rate,
            )
            result = evaluate_student(
                candidate,
                seed=args.seed + 50_000 + round_index * 10,
            )
            candidate_score = score(result)
            accepted = candidate_score > best_score
            candidate.save(output_dir / "checkpoints" / f"round_{round_index:02d}")
            if accepted:
                best_score = candidate_score
                best_result = result
                candidate.save(output_dir / "best_model")
            row = {
                "round": round_index,
                "accepted": accepted,
                "beta": args.beta,
                "collection_success_rate": collection["collection_success_rate"],
                "loss": losses[-1],
                "success_rate": result["success_rate"],
                "clearance_rate": result["clearance_rate"],
                "hard_fall_rate": result["hard_fall_rate"],
            }
            append_progress(output_dir / "progress.csv", row)
            history.append(
                {
                    **row,
                    "score": list(candidate_score),
                    "collection": collection,
                    "evaluation": result,
                }
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)

        summary = {
            "method": "conservative_recurrent_dagger_refinement",
            "source_model": str(source_model.resolve()),
            "observation_dimension": 95,
            "privileged_student_inputs": False,
            "best_score": list(best_score),
            "best_evaluation": best_result,
            "history": history,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
