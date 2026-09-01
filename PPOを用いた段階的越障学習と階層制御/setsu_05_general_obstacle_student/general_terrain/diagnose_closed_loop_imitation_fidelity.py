"""M2.3.3の閉ループ模倣保真度を重み更新なしで診断する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from general_terrain.audit_rescue_demonstrations import (
    KEY_TEACHER_PHASES,
    PHASE_CODES,
    find_true_segments,
    load_and_validate_branch_arrays,
    sha256_file,
)
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.student_prefix_rescue_env import (
    RESCUE_PHASES,
    classify_rescue_phase,
)
from general_terrain.train_phase_balanced_rescue_teacher import (
    PhaseBalancedSequence,
    load_phase_balanced_sequences,
    load_training_protocol,
)
from general_terrain.train_prefix_rescue_teacher import hash_policy_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_SUMMARY = (
    PROJECT_ROOT
    / "runs"
    / "phase_balanced_rescue_teacher"
    / "m2_3_2_phase_balanced_bc_seed7_v1"
    / "summary.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "closed_loop_imitation_fidelity"
ACTION_MSE_DIVERGENCE_THRESHOLD = 0.01
ACTION_MAX_DIVERGENCE_THRESHOLD = 0.25


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    """空配列を許容して最大絶対差を返す。"""
    if left.size == 0:
        return 0.0
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def first_threshold_crossing(
    values: np.ndarray,
    threshold: float,
) -> int | None:
    """閾値を初めて超える相対位置を返す。"""
    indices = np.flatnonzero(np.asarray(values, dtype=np.float64) > threshold)
    return int(indices[0]) if len(indices) else None


def summarize_action_errors(
    squared_errors: np.ndarray,
    maximum_errors: np.ndarray,
    phase_codes: np.ndarray,
) -> dict[str, object]:
    """動作誤差を全体、段階別、最初の明確な乖離へ集約する。"""
    squared = np.asarray(squared_errors, dtype=np.float64).reshape(-1)
    maximum = np.asarray(maximum_errors, dtype=np.float64).reshape(-1)
    phases = np.asarray(phase_codes, dtype=np.int8).reshape(-1)
    if not (len(squared) == len(maximum) == len(phases)):
        raise ValueError("動作誤差と段階コードの長さが一致しない。")
    if len(squared) < 1:
        raise ValueError("動作誤差系列は空にできない。")
    phase_rows: dict[str, dict[str, object]] = {}
    for phase in KEY_TEACHER_PHASES:
        mask = phases == PHASE_CODES[phase]
        if not np.any(mask):
            raise ValueError(f"必須段階の動作誤差が存在しない: {phase}")
        phase_rows[phase] = {
            "steps": int(np.count_nonzero(mask)),
            "mean_squared_error": float(np.mean(squared[mask])),
            "maximum_absolute_error": float(np.max(maximum[mask])),
        }
    mse_crossing = first_threshold_crossing(
        squared,
        ACTION_MSE_DIVERGENCE_THRESHOLD,
    )
    maximum_crossing = first_threshold_crossing(
        maximum,
        ACTION_MAX_DIVERGENCE_THRESHOLD,
    )
    return {
        "steps": len(squared),
        "mean_squared_error": float(np.mean(squared)),
        "maximum_absolute_error": float(np.max(maximum)),
        "phase_metrics": phase_rows,
        "divergence_thresholds": {
            "step_mean_squared_error": ACTION_MSE_DIVERGENCE_THRESHOLD,
            "maximum_absolute_action_error": ACTION_MAX_DIVERGENCE_THRESHOLD,
        },
        "first_mse_divergence_relative_step": mse_crossing,
        "first_maximum_error_divergence_relative_step": maximum_crossing,
    }


def evaluate_open_loop_sequence(
    model: Any,
    sequence: PhaseBalancedSequence,
) -> dict[str, object]:
    """保存観測上で空の循環状態から教師動作との差を測る。"""
    recurrent_state: Any = None
    episode_start = np.ones((1,), dtype=bool)
    squared_errors: list[float] = []
    maximum_errors: list[float] = []
    for observation, target in zip(sequence.observations, sequence.actions):
        action, recurrent_state = model.predict(
            observation,
            state=recurrent_state,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start[:] = False
        error = np.asarray(action, dtype=np.float64).reshape(-1) - np.asarray(
            target,
            dtype=np.float64,
        ).reshape(-1)
        squared_errors.append(float(np.mean(error**2)))
        maximum_errors.append(float(np.max(np.abs(error))))
    summary = summarize_action_errors(
        np.asarray(squared_errors),
        np.asarray(maximum_errors),
        sequence.phase_codes,
    )
    summary.update(
        {
            "seed": sequence.seed,
            "first_mse_divergence_absolute_step": (
                None
                if summary["first_mse_divergence_relative_step"] is None
                else int(summary["first_mse_divergence_relative_step"])
            ),
            "first_maximum_error_divergence_absolute_step": (
                None
                if summary["first_maximum_error_divergence_relative_step"] is None
                else int(summary["first_maximum_error_divergence_relative_step"])
            ),
        }
    )
    return summary


def evaluate_closed_loop_handoff(
    model: Any,
    candidate: Any,
) -> dict[str, object]:
    """保存学生前置動作の後を学習器だけで終端まで制御する。"""
    arrays, _ = load_and_validate_branch_arrays(candidate)
    takeover_step = find_true_segments(arrays["teacher_mask"].astype(bool))[0][0]
    course = sample_curriculum_course(candidate.seed, "hurdle_single", "train")
    if course.course_id != candidate.course_id:
        raise ValueError(f"再生成コースが示範と一致しない: {candidate.seed}")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    recurrent_state: Any = None
    episode_start = np.ones((1,), dtype=bool)
    phase_counts = {phase: 0 for phase in RESCUE_PHASES}
    maximum_saved_timeline_observation_difference = 0.0
    first_saved_timeline_observation_divergence: int | None = None
    terminated = False
    truncated = False
    try:
        observation, info = environment.reset(seed=candidate.seed)
        maximum_prefix_observation_difference = 0.0
        for index in range(takeover_step):
            difference = _maximum_difference(
                observation,
                arrays["observations"][index],
            )
            maximum_prefix_observation_difference = max(
                maximum_prefix_observation_difference,
                difference,
            )
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(arrays["executed_actions"][index], dtype=np.float32)
            )
            if terminated or truncated:
                raise RuntimeError(
                    f"学生前置再生が教師引き継ぎ前に終了した: {candidate.seed}"
                )
        takeover_observation_difference = _maximum_difference(
            observation,
            arrays["observations"][takeover_step],
        )
        if maximum_prefix_observation_difference != 0.0:
            raise RuntimeError(f"学生前置観測を精密再生できない: {candidate.seed}")
        if takeover_observation_difference != 0.0:
            raise RuntimeError(f"教師引き継ぎ状態を精密再生できない: {candidate.seed}")

        absolute_step = takeover_step
        learner_steps = 0
        while not (terminated or truncated):
            if absolute_step < len(arrays["observations"]):
                difference = _maximum_difference(
                    observation,
                    arrays["observations"][absolute_step],
                )
                maximum_saved_timeline_observation_difference = max(
                    maximum_saved_timeline_observation_difference,
                    difference,
                )
                if difference > 1e-4 and first_saved_timeline_observation_divergence is None:
                    first_saved_timeline_observation_divergence = learner_steps
            phase = classify_rescue_phase(environment, info)
            phase_counts[phase] += 1
            action, recurrent_state = model.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start[:] = False
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(action, dtype=np.float32).reshape(
                    environment.action_space.shape
                )
            )
            learner_steps += 1
            absolute_step += 1
    finally:
        environment.close()

    success = bool(
        info["course_complete"]
        and not info["hard_fall"]
        and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
    )
    return {
        "seed": candidate.seed,
        "profile": candidate.profile,
        "start_runway_voxels": candidate.start_runway_voxels,
        "takeover_step": takeover_step,
        "prefix_observation_exact": True,
        "takeover_observation_exact": True,
        "learner_recurrent_state_at_takeover": "empty",
        "learner_steps": learner_steps,
        "course_complete": bool(info["course_complete"]),
        "success": success,
        "hard_fall": bool(info["hard_fall"]),
        "safe_stall": bool(not success and not info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "phase_step_counts": phase_counts,
        "maximum_saved_timeline_observation_difference": (
            maximum_saved_timeline_observation_difference
        ),
        "first_saved_timeline_observation_divergence_relative_step": (
            first_saved_timeline_observation_divergence
        ),
    }


def _aggregate_open_loop(rows: list[dict[str, object]]) -> dict[str, object]:
    """四系列の重み付き全体値と段階別値をまとめる。"""
    total_steps = sum(int(row["steps"]) for row in rows)
    phase_metrics: dict[str, dict[str, object]] = {}
    for phase in KEY_TEACHER_PHASES:
        phase_steps = sum(
            int(row["phase_metrics"][phase]["steps"]) for row in rows
        )
        phase_mse = sum(
            int(row["phase_metrics"][phase]["steps"])
            * float(row["phase_metrics"][phase]["mean_squared_error"])
            for row in rows
        ) / phase_steps
        phase_metrics[phase] = {
            "steps": phase_steps,
            "mean_squared_error": phase_mse,
            "maximum_absolute_error": max(
                float(row["phase_metrics"][phase]["maximum_absolute_error"])
                for row in rows
            ),
        }
    return {
        "sequence_count": len(rows),
        "steps": total_steps,
        "mean_squared_error": sum(
            int(row["steps"]) * float(row["mean_squared_error"])
            for row in rows
        )
        / total_steps,
        "maximum_absolute_error": max(
            float(row["maximum_absolute_error"]) for row in rows
        ),
        "phase_metrics": phase_metrics,
    }


def diagnose(
    source_summary_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """凍結チェックポイントの開ループ誤差と閉ループ結果を保存する。"""
    from sb3_contrib import RecurrentPPO

    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if source_summary.get("checkpoint_disposition") != "quarantined_failed_m2_3_2":
        raise ValueError("M2.3.3は隔離済みM2.3.2チェックポイントだけを診断する。")
    checkpoint_path = Path(str(source_summary["teacher_checkpoint"])).resolve()
    expected_checkpoint_sha256 = str(source_summary["teacher_checkpoint_sha256"])
    if sha256_file(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("M2.3.2チェックポイントのハッシュが一致しない。")
    protocol = load_training_protocol(
        Path(str(source_summary["protocol"]["source_path"]))
    )
    sequences, dataset_metadata, manifest = load_phase_balanced_sequences(protocol)
    protected_paths = (
        source_summary_path.resolve(),
        checkpoint_path,
        protocol.source_path,
        protocol.student_model_path,
        protocol.demo_manifest_path,
        protocol.demo_audit_path,
        protocol.reset_manifest_path,
        *(candidate.branch_path for candidate in manifest.candidates),
    )
    hashes_before = {str(path): sha256_file(path) for path in protected_paths}
    model = RecurrentPPO.load(checkpoint_path, device="cpu")
    parameter_hash_before = hash_policy_parameters(model)
    open_loop_rows = [
        evaluate_open_loop_sequence(model, sequence) for sequence in sequences
    ]
    takeover_by_seed = {
        candidate.seed: find_true_segments(
            load_and_validate_branch_arrays(candidate)[0]["teacher_mask"].astype(bool)
        )[0][0]
        for candidate in manifest.candidates
    }
    for row in open_loop_rows:
        takeover_step = takeover_by_seed[int(row["seed"])]
        relative_mse = row["first_mse_divergence_relative_step"]
        relative_maximum = row["first_maximum_error_divergence_relative_step"]
        row["first_mse_divergence_absolute_step"] = (
            None if relative_mse is None else takeover_step + int(relative_mse)
        )
        row["first_maximum_error_divergence_absolute_step"] = (
            None
            if relative_maximum is None
            else takeover_step + int(relative_maximum)
        )
    closed_loop_rows = [
        evaluate_closed_loop_handoff(model, candidate)
        for candidate in manifest.candidates
    ]
    parameter_hash_after = hash_policy_parameters(model)
    if parameter_hash_after != parameter_hash_before:
        raise RuntimeError("M2.3.3診断中にモデル重みが変更された。")
    hashes_after = {str(path): sha256_file(path) for path in protected_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("M2.3.3診断中に凍結出典が変更された。")
    success_count = sum(bool(row["success"]) for row in closed_loop_rows)
    hard_fall_count = sum(bool(row["hard_fall"]) for row in closed_loop_rows)
    safe_stall_count = sum(bool(row["safe_stall"]) for row in closed_loop_rows)
    route = (
        "takeover_state_coverage_expansion"
        if success_count >= 3
        else "interactive_closed_loop_correction_aggregation"
    )
    result = {
        "method": "m2_3_3_read_only_closed_loop_imitation_fidelity",
        "stage": "hurdle_single",
        "split": "train",
        "teacher_training_only": True,
        "source_summary": str(source_summary_path.resolve()),
        "source_summary_sha256": hashes_before[str(source_summary_path.resolve())],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "dataset": dataset_metadata,
        "open_loop_action_fidelity": {
            "aggregate": _aggregate_open_loop(open_loop_rows),
            "sequences": open_loop_rows,
        },
        "closed_loop_original_takeover_states": {
            "episodes": len(closed_loop_rows),
            "success_count": success_count,
            "hard_fall_count": hard_fall_count,
            "safe_stall_count": safe_stall_count,
            "raw_clearance_count": sum(
                int(row["raw_clearances"]) > 0 for row in closed_loop_rows
            ),
            "recovery_count": sum(
                int(row["recovered_obstacles"]) > 0 for row in closed_loop_rows
            ),
            "rows": closed_loop_rows,
        },
        "decision_gate": {
            "minimum_success_count_for_coverage_route": 3,
            "success_count": success_count,
            "coverage_route_passed": success_count >= 3,
            "diagnosis": (
                "takeover_state_coverage_shortage"
                if success_count >= 3
                else "closed_loop_imitation_error_accumulation"
            ),
            "selected_next_route": route,
        },
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "ppo_training_steps": 0,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "teacher_interventions_in_student_test": 0,
        "model_parameter_hash_unchanged": True,
        "protected_source_files_unchanged": True,
        "eligible_for_m2_4": False,
        "eligible_for_final_student_test": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    """M2.3.3の出典と出力名だけを受け取る。"""
    parser = argparse.ArgumentParser(
        description="隔離済み模倣チェックポイントを閉ループ診断する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-summary", default=str(DEFAULT_SOURCE_SUMMARY))
    return parser


def main() -> None:
    """四つの元引き継ぎ状態を診断してJSONへ保存する。"""
    args = build_argument_parser().parse_args()
    result = diagnose(
        Path(args.source_summary),
        RUNS_ROOT / args.run_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
