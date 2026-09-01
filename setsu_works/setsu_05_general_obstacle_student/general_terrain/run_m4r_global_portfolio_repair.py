"""全局動作分岐の早期接管と検証済み組合せ教師を零更新で復核する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from general_terrain.audit_rescue_demonstrations import sha256_file
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher
from general_terrain.run_m4r_teacher_repair import (
    LearnerPrefixReplay,
    build_repair_gate,
    collect_trigger_replay,
    load_protocol,
    write_trigger_artifacts,
)
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_prefix_rescue_env import classify_rescue_phase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "config"
    / "m4r_learner_distribution_teacher_repair_protocol_v5.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "m4r_teacher_repair"


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    """二配列の最大絶対差を倍精度で返す。"""
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def _portfolio_route(position: int) -> dict[str, object]:
    """検証済み組合せ教師の位置別初期制御値を監査用に返す。"""
    profile = PortfolioHeight1Teacher._profile_for_position(position)
    switch_fraction = {
        "early_direct": 0.25,
        "mid_direct": 0.40,
        "tuned_direct": 0.38,
        "raw_recovery": 0.50,
        "half_recovery": 0.50,
    }[profile]
    return {
        "first_switch_fraction": switch_fraction,
        "post_clear_mode": "restart_then_flat",
        "handoff_distance": 0.40 if profile == "tuned_direct" else 0.45,
        "clearance_blend": 1.0,
        "clearance_family": "first",
        "adaptive_handoff": True,
    }


def evaluate_verified_portfolio(
    replay: LearnerPrefixReplay,
    *,
    prefix_offset: int = 0,
) -> dict[str, object]:
    """学習器快照から凍結済み組合せ教師を厳格終端まで実行する。"""
    prefix_steps = replay.trigger_step - prefix_offset
    if prefix_steps < 0:
        raise ValueError("組合せ教師の快照歩数は零以上でなければならない。")
    course = sample_curriculum_course(replay.seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    teacher = PortfolioHeight1Teacher()
    maximum_difference = 0.0
    phase_counts = {
        "pre_hurdle_safety_intercept": 0,
        "hurdle_contact_deformation": 0,
        "landing_recovery_stall": 0,
    }
    try:
        observation, info = environment.reset(seed=replay.seed)
        for index, action in enumerate(replay.actions[:prefix_steps]):
            maximum_difference = max(
                maximum_difference,
                _maximum_difference(observation, replay.observations[index]),
            )
            observation, _, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"学習器前置軌跡が快照前に終了した: {replay.seed}, {index}"
                )
        maximum_difference = max(
            maximum_difference,
            _maximum_difference(observation, replay.observations[prefix_steps]),
        )
        teacher.reset(environment)
        terminated = False
        truncated = False
        rescue_steps = 0
        final_stage = ""
        while not (terminated or truncated):
            phase = classify_rescue_phase(environment, info)
            if phase == "pre_hurdle":
                phase_counts["pre_hurdle_safety_intercept"] += 1
            elif phase == "hurdle_deformation":
                phase_counts["hurdle_contact_deformation"] += 1
            else:
                phase_counts["landing_recovery_stall"] += 1
            action, final_stage = teacher.predict(environment, observation, dict(info))
            observation, _, terminated, truncated, info = environment.step(action)
            rescue_steps += 1
    finally:
        environment.close()
    success = bool(
        info["course_complete"]
        and not info["hard_fall"]
        and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
    )
    return {
        "controller_mode": "verified_portfolio",
        "seed": replay.seed,
        "start_runway_voxels": replay.start_runway_voxels,
        "trigger_profile": replay.profile_name,
        "trigger_step": replay.trigger_step,
        "prefix_offset": prefix_offset,
        "prefix_steps": prefix_steps,
        "rescue_steps": rescue_steps,
        "success": success,
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "maximum_x_position": float(info["max_x_position"]),
        "final_stage": final_stage,
        "phase_step_counts": phase_counts,
        "maximum_prefix_observation_difference": maximum_difference,
        "prefix_replay_exact": maximum_difference == 0.0,
    }


def evaluate_profile(
    profile: Mapping[str, object],
    replays: tuple[LearnerPrefixReplay, ...],
    *,
    perturbation_offsets: tuple[int, ...],
    minimum_robust_successes: int,
    requirements: Mapping[str, object],
    sources_unchanged: bool,
) -> dict[str, object]:
    """一つの全局分岐開始条件を正確快照と早期快照で復核する。"""
    position_results: list[dict[str, object]] = []
    for replay in replays:
        exact = evaluate_verified_portfolio(replay)
        perturbations = [
            evaluate_verified_portfolio(replay, prefix_offset=offset)
            for offset in perturbation_offsets
            if replay.trigger_step - offset >= 1
        ]
        robust_successes = sum(bool(row["success"]) for row in perturbations)
        robust_hard_falls = sum(bool(row["hard_fall"]) for row in perturbations)
        robust = bool(
            exact["success"]
            and not exact["hard_fall"]
            and robust_successes >= minimum_robust_successes
            and robust_hard_falls == 0
        )
        status = (
            "robust_success"
            if robust
            else "exact_only_success"
            if exact["success"] and not exact["hard_fall"]
            else "safe_fallback"
            if not exact["hard_fall"]
            else "unsafe_fallback"
        )
        position_results.append(
            {
                "seed": replay.seed,
                "start_runway_voxels": replay.start_runway_voxels,
                "trigger_step": replay.trigger_step,
                "trigger_reason": replay.trigger_event["reason"],
                "trigger_upper_body_grounded": replay.trigger_event[
                    "upper_body_grounded"
                ],
                "attempt_count": 1,
                "selection_status": status,
                "selected": _portfolio_route(replay.start_runway_voxels),
                "selected_candidate_id": "verified_portfolio",
                "selected_exact_evaluation": exact,
                "selected_perturbation_evaluations": perturbations,
                "attempts": [],
            }
        )
    screen = {
        "profile": dict(profile),
        "profile_name": str(profile["name"]),
        "success_count": sum(
            bool(row["selected_exact_evaluation"]["success"])
            for row in position_results
        ),
        "hard_fall_count": sum(
            bool(row["selected_exact_evaluation"]["hard_fall"])
            for row in position_results
        ),
        "safe_stall_count": sum(
            not bool(row["selected_exact_evaluation"]["success"])
            and not bool(row["selected_exact_evaluation"]["hard_fall"])
            for row in position_results
        ),
        "grounded_trigger_count": sum(
            bool(replay.trigger_event["upper_body_grounded"])
            for replay in replays
        ),
        "mean_trigger_step": float(np.mean([replay.trigger_step for replay in replays])),
        "trigger_steps": {
            str(replay.start_runway_voxels): replay.trigger_step for replay in replays
        },
    }
    result = {
        "profile": dict(profile),
        "profile_name": str(profile["name"]),
        "controller_mode": "verified_portfolio",
        "screen": screen,
        "position_results": position_results,
    }
    result["repair_gate"] = build_repair_gate(
        result,
        requirements=requirements,
        sources_unchanged=sources_unchanged,
    )
    return result


def _source_entry(path: Path) -> dict[str, str]:
    """プロジェクト相対パスと現在ハッシュを目録形式で返す。"""
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(PROJECT_ROOT.resolve())),
        "sha256": sha256_file(resolved),
    }


def build_portfolio_manifest(
    protocol: Mapping[str, object],
    selected: Mapping[str, object],
    trigger_manifest_path: Path,
    gate: Mapping[str, object],
) -> dict[str, object]:
    """合格した全局開始条件と組合せ教師を訓練専用目録へ固定する。"""
    source_paths = [
        Path(protocol["probe_preflight_summary_path"]),
        Path(protocol["probe_student_model_path"]),
        Path(protocol["protected_original_student_model_path"]),
        Path(protocol["source_teacher_manifest_path"]),
        trigger_manifest_path,
    ]
    source_paths.extend(
        Path(row["resolved_path"])
        for row in protocol.get("portfolio_teacher_sources", [])
    )
    sources = []
    seen: set[Path] = set()
    for path in source_paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            sources.append(_source_entry(resolved))
    return {
        "version": "m4r_global_disagreement_portfolio_teacher_manifest_v1",
        "frozen": True,
        "stage": "hurdle_single",
        "split": "train",
        "teacher_training_only": True,
        "purpose": "training_demonstration_and_rescue_only",
        "controller_mode": "verified_portfolio",
        "trigger_profile": selected["profile"],
        "separate_recovery_courses": protocol["separate_recovery_courses"],
        "robust_flat_prefix_steps": 0,
        "robust_flat_teacher_model_path": None,
        "routes": {
            str(position): _portfolio_route(position) for position in range(20, 31)
        },
        "route_status": {
            str(row["start_runway_voxels"]): row["selection_status"]
            for row in selected["position_results"]
        },
        "repair_gate": dict(gate),
        "eligible_for_continuous_validation": bool(gate["gate_passed"]),
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "probe_student_weights_updated": False,
        "protected_original_student_weights_updated": False,
        "teacher_weights_updated": False,
        "sources": sources,
    }


def run(protocol: dict[str, object], output_dir: Path) -> dict[str, object]:
    """全局分岐開始条件を選別し合格時だけ組合せ教師目録を凍結する。"""
    from sb3_contrib import RecurrentPPO

    output_dir.mkdir(parents=True, exist_ok=False)
    seed_manifest = load_seed_manifest(Path(protocol["seed_manifest_path"]))
    seeds = seed_manifest.for_split("train")
    student = RecurrentPPO.load(Path(protocol["probe_student_model_path"]), device="cpu")
    monitor_teacher = PortfolioHeight1Teacher()
    protected = {
        Path(protocol["source_path"]),
        Path(protocol["seed_manifest_path"]),
        Path(protocol["probe_preflight_summary_path"]),
        Path(protocol["probe_student_model_path"]),
        Path(protocol["protected_original_student_model_path"]),
        Path(protocol["source_teacher_manifest_path"]),
        *(Path(path) for path in protocol["protocol_chain_paths"]),
        *(
            Path(row["resolved_path"])
            for row in protocol.get("portfolio_teacher_sources", [])
        ),
    }
    before = {str(path): sha256_file(path) for path in sorted(protected)}
    profile_results: list[dict[str, object]] = []
    profile_replays: dict[str, tuple[LearnerPrefixReplay, ...]] = {}
    for profile in protocol["trigger_profiles"]:
        replays = tuple(
            collect_trigger_replay(
                student,
                monitor_teacher,
                seed=seed,
                profile=profile,
            )
            for seed in seeds
        )
        name = str(profile["name"])
        profile_replays[name] = replays
        result = evaluate_profile(
            profile,
            replays,
            perturbation_offsets=tuple(
                int(value) for value in protocol["perturbation_prefix_offsets"]
            ),
            minimum_robust_successes=int(
                protocol["minimum_required_robust_offset_successes"]
            ),
            requirements=protocol["repair_gate"],
            sources_unchanged=before
            == {str(path): sha256_file(path) for path in sorted(protected)},
        )
        profile_results.append(result)
        print(
            json.dumps(
                {
                    "event": "global_portfolio_profile_evaluated",
                    "profile": name,
                    "mean_trigger_step": result["screen"]["mean_trigger_step"],
                    "success_count": result["repair_gate"][
                        "exact_state_success_count"
                    ],
                    "hard_fall_count": result["repair_gate"][
                        "exact_state_hard_fall_count"
                    ],
                    "robust_position_count": result["repair_gate"][
                        "robust_position_count"
                    ],
                    "gate_passed": result["repair_gate"]["gate_passed"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if bool(result["repair_gate"]["gate_passed"]):
            break
    selected = max(
        profile_results,
        key=lambda row: (
            int(row["repair_gate"]["gate_passed"]),
            int(row["repair_gate"]["exact_state_success_count"]),
            int(row["repair_gate"]["robust_position_count"]),
            -int(row["repair_gate"]["exact_state_hard_fall_count"]),
            -float(row["screen"]["mean_trigger_step"]),
        ),
    )
    selected_replays = profile_replays[str(selected["profile_name"])]
    trigger_manifest_path, trajectory_rows = write_trigger_artifacts(
        output_dir,
        selected["profile"],
        selected_replays,
        offsets=tuple(int(value) for value in protocol["perturbation_prefix_offsets"]),
    )
    teacher_manifest = build_portfolio_manifest(
        protocol,
        selected,
        trigger_manifest_path,
        selected["repair_gate"],
    )
    teacher_manifest_path = output_dir / "teacher_manifest.json"
    teacher_manifest_path.write_text(
        json.dumps(teacher_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    after = {str(path): sha256_file(path) for path in sorted(protected)}
    if before != after:
        raise RuntimeError("M4R全局組合せ復核中に凍結出典が変更された。")
    summary = {
        "method": "m4r_global_teacher_disagreement_verified_portfolio_repair",
        "run_name": output_dir.name,
        "stage": "hurdle_single",
        "split": "train",
        "protocol_path": str(protocol["source_path"]),
        "protocol_sha256": protocol["sha256"],
        "seed_manifest": seed_manifest.as_dict(),
        "profile_results": profile_results,
        "selected_profile_name": selected["profile_name"],
        "selected_profile": selected,
        "trigger_state_manifest_path": str(trigger_manifest_path.resolve()),
        "trigger_state_manifest_sha256": sha256_file(trigger_manifest_path),
        "trajectory_rows": trajectory_rows,
        "teacher_manifest_path": str(teacher_manifest_path.resolve()),
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "repair_gate": selected["repair_gate"],
        "eligible_for_continuous_validation": bool(
            selected["repair_gate"]["gate_passed"]
        ),
        "probe_student_weights_updated": False,
        "protected_original_student_weights_updated": False,
        "teacher_weights_updated": False,
        "probe_student_update_steps": 0,
        "protected_original_student_update_steps": 0,
        "teacher_weight_update_steps": 0,
        "ppo_training_steps": 0,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_files_unchanged": True,
        "eligible_for_student_update": False,
        "eligible_for_final_student_test": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """凍結規約と一意な出力名だけを受け取る。"""
    parser = argparse.ArgumentParser(
        description="全局動作分岐から検証済み組合せ教師へ早期接管する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """M4R全局分岐組合せ教師の零更新復核を実行する。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name)
    print(
        json.dumps(
            {
                "run_name": result["run_name"],
                "selected_profile_name": result["selected_profile_name"],
                "repair_gate": result["repair_gate"],
                "teacher_manifest_path": result["teacher_manifest_path"],
                "teacher_manifest_sha256": result["teacher_manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
