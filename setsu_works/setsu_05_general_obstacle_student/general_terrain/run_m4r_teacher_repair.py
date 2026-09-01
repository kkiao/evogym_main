"""M4Rの早期危険介入と学習器分布別救援経路を有界探索する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from general_terrain.audit_rescue_demonstrations import sha256_file
from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.interactive_rescue import (
    InteractiveRescueController,
    local_terrain_is_visible,
)
from general_terrain.m4r_learner_distribution_teacher import (
    trigger_config_from_payload,
)
from general_terrain.manifest_training_rescue_teacher import (
    ManifestTrainingRescueTeacher,
    TrainingTeacherManifest,
    load_training_teacher_manifest,
)
from general_terrain.search_takeover_recalibration_rescue import (
    RecalibratedControllerCandidate,
    generate_candidates,
)
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_prefix_rescue_env import classify_rescue_phase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "config"
    / "m4r_learner_distribution_teacher_repair_protocol_v4.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "m4r_teacher_repair"


@dataclass(frozen=True)
class LearnerPrefixReplay:
    """一つの危険開始条件までの学習器動作と全観測を保持する。"""

    profile_name: str
    seed: int
    start_runway_voxels: int
    actions: np.ndarray
    observations: np.ndarray
    monitor_teacher_actions: np.ndarray
    state_rows: tuple[dict[str, object], ...]
    trigger_event: dict[str, object]
    maximum_replay_observation_difference: float = 0.0

    @property
    def trigger_step(self) -> int:
        """教師が接管する直前までの学生動作歩数を返す。"""
        return len(self.actions)


def _resolve_project_path(value: str) -> Path:
    """規約内の相対パスをプロジェクト配下だけへ解決する。"""
    resolved = (PROJECT_ROOT / value).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M4R出典はプロジェクト配下でなければならない。")
    return resolved


def _load_protocol_chain(
    path: Path,
    *,
    seen: set[Path] | None = None,
) -> tuple[dict[str, object], tuple[Path, ...]]:
    """継承規約を根から読み込み循環と各基礎ハッシュを検査する。"""
    resolved = path.resolve()
    visited = set() if seen is None else set(seen)
    if resolved in visited:
        raise ValueError("M4R規約継承に循環がある。")
    visited.add(resolved)
    supplied = json.loads(resolved.read_text(encoding="utf-8"))
    if "base_protocol_path" not in supplied:
        return supplied, (resolved,)
    base_path = _resolve_project_path(str(supplied["base_protocol_path"]))
    if sha256_file(base_path) != str(supplied["base_protocol_sha256"]).lower():
        raise ValueError("M4R基礎規約ハッシュが一致しない。")
    base_payload, chain = _load_protocol_chain(base_path, seen=visited)
    return {**base_payload, **supplied}, (*chain, resolved)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、出典ハッシュ、零更新、教師隔離条件を検査する。"""
    source_path = path.resolve()
    payload, protocol_chain = _load_protocol_chain(source_path)
    if "minimum_robust_position_count" in payload:
        payload["repair_gate"] = {
            **payload["repair_gate"],
            "minimum_robust_position_count": int(
                payload["minimum_robust_position_count"]
            ),
        }
    if not bool(payload.get("frozen", False)):
        raise ValueError("M4R規約は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M4Rは単一低壁の訓練区分だけを使用できる。")
    zero_fields = (
        "validation_episodes",
        "holdout_episodes",
        "probe_student_update_steps",
        "protected_original_student_update_steps",
        "teacher_weight_update_steps",
        "ppo_training_steps",
        "teacher_interventions_in_final_student_test",
    )
    if any(int(payload[field]) != 0 for field in zero_fields):
        raise ValueError("M4Rでは学生更新、教師重み更新、評価教師介入を禁止する。")
    for field in (
        "validation_teacher_enabled",
        "holdout_teacher_enabled",
        "final_student_test_teacher_enabled",
    ):
        if bool(payload.get(field, True)):
            raise ValueError("検証、留保、最終学生試験では教師を停止しなければならない。")
    if int(payload["train_trigger_screen_episodes_per_profile"]) != 11:
        raise ValueError("各開始条件の訓練選別は十一回でなければならない。")
    if int(payload["route_candidate_limit_per_position"]) != 192:
        raise ValueError("位置ごとの救援経路候補上限は192でなければならない。")
    path_fields = (
        "seed_manifest_path",
        "probe_preflight_summary_path",
        "probe_student_model_path",
        "protected_original_student_model_path",
        "source_teacher_manifest_path",
    )
    if "robust_flat_teacher_model_path" in payload:
        path_fields = (
            *path_fields,
            "robust_flat_teacher_model_path",
            "robust_flat_teacher_summary_path",
        )
    resolved: dict[str, Path] = {}
    for field in path_fields:
        file_path = _resolve_project_path(str(payload[field]))
        expected = str(payload[f"{field.removesuffix('_path')}_sha256"]).lower()
        if sha256_file(file_path) != expected:
            raise ValueError(f"M4R出典ハッシュが一致しない: {file_path}")
        resolved[field] = file_path
    portfolio_sources: list[dict[str, object]] = []
    for source in payload.get("portfolio_teacher_sources", []):
        file_path = _resolve_project_path(str(source["path"]))
        expected = str(source["sha256"]).lower()
        if sha256_file(file_path) != expected:
            raise ValueError(f"M4R組合せ教師の出典ハッシュが一致しない: {file_path}")
        portfolio_sources.append({**source, "resolved_path": file_path})
    preflight = json.loads(
        resolved["probe_preflight_summary_path"].read_text(encoding="utf-8")
    )
    if bool(preflight["m4_probe_preflight_gate"]["gate_passed"]):
        raise ValueError("合格済み前置復核からM4Rへ切り替えることはできない。")
    if int(preflight["collection"]["collection_success_count"]) != 1:
        raise ValueError("M4R失敗基線は一成功でなければならない。")
    if "robust_flat_teacher_summary_path" in resolved:
        flat_summary = json.loads(
            resolved["robust_flat_teacher_summary_path"].read_text(encoding="utf-8")
        )
        evidence = payload["robust_flat_teacher_evidence"]
        best = next(
            row
            for row in flat_summary["evaluations"]
            if int(row["step"]) == int(flat_summary["best_step"])
        )
        if int(best["success_count"]) != int(evidence["evaluation_success_count"]):
            raise ValueError("頑健平地教師の成功根拠が凍結値と一致しない。")
        if int(best["hard_fall_count"]) != int(evidence["evaluation_hard_fall_count"]):
            raise ValueError("頑健平地教師の転倒根拠が凍結値と一致しない。")
    return {
        **payload,
        **resolved,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
        "protocol_chain_paths": protocol_chain,
        "portfolio_teacher_sources": portfolio_sources,
    }


def _state_row(step: int, info: Mapping[str, object]) -> dict[str, object]:
    """危険開始監査に必要な一歩分の物理指標を返す。"""
    return {
        "step": step,
        "x_position": float(info["x_position"]),
        "orientation_error": float(info["orientation_error"]),
        "angular_velocity": float(info["angular_velocity"]),
        "stall_steps": int(info["stall_steps"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "upper_body_grounded": bool(info["upper_body_grounded"]),
        "hard_fall": bool(info["hard_fall"]),
    }


def collect_trigger_replay(
    student: Any,
    monitor_teacher: ManifestTrainingRescueTeacher,
    *,
    seed: int,
    profile: Mapping[str, object],
) -> LearnerPrefixReplay:
    """学生を危険開始まで実行し接管前軌跡と安全快照をメモリへ保存する。"""
    course = sample_curriculum_course(seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    rescue = InteractiveRescueController(trigger_config_from_payload(profile))
    recurrent_state: Any = None
    episode_start = np.ones((1,), dtype=bool)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    monitor_actions: list[np.ndarray] = []
    state_rows: list[dict[str, object]] = []
    try:
        observation, info = environment.reset(seed=seed)
        monitor_teacher.reset(environment)
        schema = tuple(environment.unwrapped.schema)
        step = 0
        while True:
            student_action, recurrent_state = student.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            teacher_action, teacher_stage = monitor_teacher.predict(
                environment,
                observation,
                info,
            )
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            monitor_actions.append(
                np.asarray(teacher_action, dtype=np.float32).copy()
            )
            state_rows.append(_state_row(step, info))
            decision = rescue.decide(
                info,
                np.asarray(student_action, dtype=np.float32),
                np.asarray(teacher_action, dtype=np.float32),
                local_terrain_visible=local_terrain_is_visible(
                    observation,
                    schema,
                    maximum_rise_offset=(
                        trigger_config_from_payload(
                            profile
                        ).disagreement_maximum_rise_offset
                    ),
                ),
            )
            if decision.event == "start":
                trigger = {
                    **state_rows[-1],
                    "event": "start",
                    "reason": decision.reason,
                    "teacher_stage": teacher_stage,
                    "action_disagreement": float(
                        np.mean(
                            np.abs(
                                np.asarray(student_action, dtype=np.float32)
                                - np.asarray(teacher_action, dtype=np.float32)
                            )
                        )
                    ),
                }
                break
            actions.append(np.asarray(student_action, dtype=np.float32).copy())
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(student_action, dtype=np.float32)
            )
            episode_start[:] = False
            step += 1
            if terminated or truncated:
                raise RuntimeError(f"危険開始前に探針回が終了した: {seed}")
    finally:
        environment.close()
    return LearnerPrefixReplay(
        profile_name=str(profile["name"]),
        seed=seed,
        start_runway_voxels=int(course.obstacles[0].start_x),
        actions=np.asarray(actions, dtype=np.float32),
        observations=np.asarray(observations, dtype=np.float32),
        monitor_teacher_actions=np.asarray(monitor_actions, dtype=np.float32),
        state_rows=tuple(state_rows),
        trigger_event=trigger,
    )


def _make_controller(
    candidate: RecalibratedControllerCandidate,
    template: ClosedLoopHeight1Teacher,
) -> ClosedLoopHeight1Teacher:
    """読取専用専門方策を共有し候補制御値だけを複製する。"""
    controller = copy.copy(template)
    controller.post_clear_mode = candidate.post_clear_mode
    controller.clearance_blend = candidate.clearance_blend
    controller.handoff_distance = candidate.handoff_distance
    controller.adaptive_handoff = candidate.adaptive_handoff
    controller.clearance_family = candidate.clearance_family
    controller.first_switch_fraction = candidate.first_switch_fraction
    return controller


def evaluate_route(
    candidate: RecalibratedControllerCandidate,
    replay: LearnerPrefixReplay,
    template: ClosedLoopHeight1Teacher,
    *,
    prefix_offset: int = 0,
) -> dict[str, object]:
    """指定した接管前快照から一経路を厳格終端まで評価する。"""
    prefix_steps = replay.trigger_step - prefix_offset
    if prefix_steps < 1:
        raise ValueError("接管前快照は少なくとも一学生歩を保持しなければならない。")
    course = sample_curriculum_course(replay.seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    controller = _make_controller(candidate, template)
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
                float(
                    np.max(
                        np.abs(
                            np.asarray(observation, dtype=np.float64)
                            - replay.observations[index].astype(np.float64)
                        )
                    )
                ),
            )
            observation, _, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"探針前置軌跡が快照前に終了した: {replay.seed}, {index}"
                )
        maximum_difference = max(
            maximum_difference,
            float(
                np.max(
                    np.abs(
                        np.asarray(observation, dtype=np.float64)
                        - replay.observations[prefix_steps].astype(np.float64)
                    )
                )
            ),
        )
        controller.reset(environment)
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
            action, final_stage = controller.predict(environment, dict(info))
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
        "candidate_id": candidate.candidate_id,
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


def _fallback_candidate_for_position(position: int) -> RecalibratedControllerCandidate:
    """旧教師の未登録位置に使用された安全後備設定を返す。"""
    return RecalibratedControllerCandidate(
        first_switch_fraction=0.25,
        post_clear_mode="restart_then_flat",
        handoff_distance=0.45,
        clearance_blend=1.0,
        clearance_family="first",
        adaptive_handoff=True,
    )


def route_candidates(
    protocol: Mapping[str, object],
    source_manifest: TrainingTeacherManifest,
) -> tuple[RecalibratedControllerCandidate, ...]:
    """既存実証経路を先頭に置き残りを凍結192候補で補う。"""
    candidates: list[RecalibratedControllerCandidate] = []
    seen: set[str] = set()

    def add(candidate: RecalibratedControllerCandidate) -> None:
        """候補順を保ったまま重複識別子を除く。"""
        if candidate.candidate_id not in seen:
            seen.add(candidate.candidate_id)
            candidates.append(candidate)

    for route in source_manifest.routes.values():
        add(RecalibratedControllerCandidate(**asdict(route)))
    add(_fallback_candidate_for_position(21))
    search_protocol = {
        **dict(protocol["route_search"]),
        "candidate_limit_per_position": int(
            protocol["route_candidate_limit_per_position"]
        ),
    }
    for candidate in generate_candidates(search_protocol):
        add(candidate)
        if len(candidates) >= int(protocol["route_candidate_limit_per_position"]):
            break
    return tuple(candidates)


def select_route(
    replay: LearnerPrefixReplay,
    candidates: tuple[RecalibratedControllerCandidate, ...],
    template: ClosedLoopHeight1Teacher,
    *,
    perturbation_offsets: tuple[int, ...],
    minimum_robust_successes: int,
    maximum_attempts: int,
) -> dict[str, object]:
    """正確状態と早期快照の両方を安全に救う最初の経路を選ぶ。"""
    attempts: list[dict[str, object]] = []
    best_safe: tuple[tuple[int, int, int, float], RecalibratedControllerCandidate, dict[str, object]] | None = None
    first_exact: tuple[RecalibratedControllerCandidate, dict[str, object], list[dict[str, object]]] | None = None
    selected: RecalibratedControllerCandidate | None = None
    selected_exact: dict[str, object] | None = None
    selected_perturbations: list[dict[str, object]] = []
    for candidate in candidates[:maximum_attempts]:
        exact = evaluate_route(candidate, replay, template)
        perturbations: list[dict[str, object]] = []
        if bool(exact["success"]) and not bool(exact["hard_fall"]):
            for offset in perturbation_offsets:
                if replay.trigger_step > offset:
                    perturbations.append(
                        evaluate_route(
                            candidate,
                            replay,
                            template,
                            prefix_offset=offset,
                        )
                    )
            if first_exact is None:
                first_exact = (candidate, exact, perturbations)
            robust_successes = sum(bool(row["success"]) for row in perturbations)
            robust_hard_falls = sum(bool(row["hard_fall"]) for row in perturbations)
            if (
                robust_successes >= minimum_robust_successes
                and robust_hard_falls == 0
            ):
                selected = candidate
                selected_exact = exact
                selected_perturbations = perturbations
        rank = (
            int(bool(exact["success"])),
            int(exact["recovered_obstacles"]),
            int(exact["raw_clearances"]),
            float(exact["maximum_x_position"]),
        )
        if not bool(exact["hard_fall"]) and (best_safe is None or rank > best_safe[0]):
            best_safe = (rank, candidate, exact)
        attempts.append(
            {
                "candidate": asdict(candidate),
                "candidate_id": candidate.candidate_id,
                "exact_evaluation": exact,
                "perturbation_evaluations": perturbations,
            }
        )
        if first_exact is not None:
            break
    if selected is not None:
        status = "robust_success"
        chosen = selected
        chosen_exact = selected_exact
        chosen_perturbations = selected_perturbations
    elif first_exact is not None:
        status = "exact_only_success"
        chosen, chosen_exact, chosen_perturbations = first_exact
    elif best_safe is not None:
        status = "safe_fallback"
        _, chosen, chosen_exact = best_safe
        chosen_perturbations = []
    else:
        status = "unsafe_fallback"
        chosen = candidates[0]
        chosen_exact = attempts[0]["exact_evaluation"]
        chosen_perturbations = attempts[0]["perturbation_evaluations"]
    return {
        "seed": replay.seed,
        "start_runway_voxels": replay.start_runway_voxels,
        "trigger_step": replay.trigger_step,
        "trigger_reason": replay.trigger_event["reason"],
        "trigger_upper_body_grounded": replay.trigger_event[
            "upper_body_grounded"
        ],
        "attempt_count": len(attempts),
        "selection_status": status,
        "selected": asdict(chosen),
        "selected_candidate_id": chosen.candidate_id,
        "selected_exact_evaluation": chosen_exact,
        "selected_perturbation_evaluations": chosen_perturbations,
        "attempts": attempts,
    }


def screen_profile(
    profile: Mapping[str, object],
    replays: tuple[LearnerPrefixReplay, ...],
    source_manifest: TrainingTeacherManifest,
    template: ClosedLoopHeight1Teacher,
) -> dict[str, object]:
    """旧位置経路で一開始条件の安全性と修復前基線を測る。"""
    rows: list[dict[str, object]] = []
    for replay in replays:
        source_route = source_manifest.routes.get(replay.start_runway_voxels)
        candidate = (
            RecalibratedControllerCandidate(**asdict(source_route))
            if source_route is not None
            else _fallback_candidate_for_position(replay.start_runway_voxels)
        )
        rows.append(evaluate_route(candidate, replay, template))
    success_count = sum(bool(row["success"]) for row in rows)
    hard_fall_count = sum(bool(row["hard_fall"]) for row in rows)
    grounded_count = sum(
        bool(replay.trigger_event["upper_body_grounded"]) for replay in replays
    )
    return {
        "profile": dict(profile),
        "profile_name": profile["name"],
        "success_count": success_count,
        "hard_fall_count": hard_fall_count,
        "safe_stall_count": sum(
            not bool(row["success"]) and not bool(row["hard_fall"])
            for row in rows
        ),
        "grounded_trigger_count": grounded_count,
        "mean_trigger_step": float(np.mean([row.trigger_step for row in replays])),
        "trigger_steps": {
            str(row.start_runway_voxels): row.trigger_step for row in replays
        },
        "rows": rows,
    }


def build_repair_gate(
    profile_result: Mapping[str, object],
    *,
    requirements: Mapping[str, object],
    sources_unchanged: bool,
) -> dict[str, object]:
    """正確救援、早期快照、無転倒、出典不変から修復門を判定する。"""
    positions = list(profile_result["position_results"])
    exact_success_count = sum(
        bool(row["selected_exact_evaluation"]["success"]) for row in positions
    )
    exact_hard_fall_count = sum(
        bool(row["selected_exact_evaluation"]["hard_fall"]) for row in positions
    )
    robust_position_count = sum(
        row["selection_status"] == "robust_success" for row in positions
    )
    perturbation_hard_fall_count = sum(
        bool(item["hard_fall"])
        for row in positions
        for item in row["selected_perturbation_evaluations"]
    )
    grounded_count = int(profile_result["screen"]["grounded_trigger_count"])
    checks = {
        "exact_success_requirement_passed": exact_success_count
        >= int(requirements["minimum_exact_state_success_count"]),
        "exact_hard_fall_requirement_passed": exact_hard_fall_count
        <= int(requirements["maximum_exact_state_hard_fall_count"]),
        "robust_position_requirement_passed": robust_position_count
        >= int(requirements["minimum_robust_position_count"]),
        "perturbation_hard_fall_requirement_passed": perturbation_hard_fall_count
        <= int(requirements["maximum_perturbation_hard_fall_count"]),
        "ungrounded_trigger_requirement_passed": grounded_count == 0,
        "source_immutability_requirement_passed": sources_unchanged,
    }
    return {
        "gate_name": "m4r_learner_distribution_teacher_repair_gate_v1",
        **requirements,
        "exact_state_success_count": exact_success_count,
        "exact_state_hard_fall_count": exact_hard_fall_count,
        "robust_position_count": robust_position_count,
        "perturbation_hard_fall_count": perturbation_hard_fall_count,
        "grounded_trigger_count": grounded_count,
        **checks,
        "gate_passed": all(checks.values()),
        "eligible_for_continuous_validation": all(checks.values()),
    }


def write_trigger_artifacts(
    output_dir: Path,
    profile: Mapping[str, object],
    replays: tuple[LearnerPrefixReplay, ...],
    *,
    offsets: tuple[int, ...],
) -> tuple[Path, list[dict[str, object]]]:
    """選択開始条件の軌跡NPZと接管前安全快照目録を凍結保存する。"""
    trajectories_dir = output_dir / "probe_trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for replay in replays:
        trajectory_path = trajectories_dir / (
            f"seed_{replay.seed}_x{replay.start_runway_voxels}_prefix.npz"
        )
        np.savez_compressed(
            trajectory_path,
            observations=replay.observations,
            student_actions=replay.actions,
            monitor_teacher_actions=replay.monitor_teacher_actions,
        )
        snapshots = []
        for offset in (0, *offsets):
            index = replay.trigger_step - offset
            if index >= 0:
                snapshots.append(
                    {
                        "prefix_offset": offset,
                        **replay.state_rows[index],
                    }
                )
        rows.append(
            {
                "seed": replay.seed,
                "start_runway_voxels": replay.start_runway_voxels,
                "trigger_step": replay.trigger_step,
                "trigger_event": replay.trigger_event,
                "safe_snapshots": snapshots,
                "trajectory_path": str(trajectory_path.resolve()),
                "trajectory_sha256": sha256_file(trajectory_path),
                "observation_count": len(replay.observations),
                "student_action_count": len(replay.actions),
            }
        )
    manifest = {
        "version": "m4r_probe_trigger_state_manifest_v1",
        "frozen": True,
        "stage": "hurdle_single",
        "split": "train",
        "trigger_profile": dict(profile),
        "student_weights_updated": False,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "trajectories": rows,
    }
    path = output_dir / "trigger_state_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, rows


def build_teacher_manifest(
    protocol: Mapping[str, object],
    selected_profile: Mapping[str, object],
    trigger_manifest_path: Path,
    gate: Mapping[str, object],
) -> dict[str, object]:
    """修復門を通過した十一経路と三分離課程を訓練専用目録へ固定する。"""
    routes = {
        str(row["start_runway_voxels"]): row["selected"]
        for row in selected_profile["position_results"]
    }
    statuses = {
        str(row["start_runway_voxels"]): row["selection_status"]
        for row in selected_profile["position_results"]
    }
    source_fields = (
        "probe_preflight_summary_path",
        "probe_student_model_path",
        "protected_original_student_model_path",
        "source_teacher_manifest_path",
    )
    if "robust_flat_teacher_model_path" in protocol:
        source_fields = (
            *source_fields,
            "robust_flat_teacher_model_path",
            "robust_flat_teacher_summary_path",
        )
    sources = [
        {
            "path": str(Path(protocol[field]).resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(Path(protocol[field])),
        }
        for field in source_fields
    ]
    sources.append(
        {
            "path": str(trigger_manifest_path.resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(trigger_manifest_path),
        }
    )
    return {
        "version": "m4r_learner_distribution_training_teacher_manifest_v1",
        "frozen": True,
        "stage": "hurdle_single",
        "split": "train",
        "teacher_training_only": True,
        "purpose": "training_demonstration_and_rescue_only",
        "trigger_profile": selected_profile["profile"],
        "separate_recovery_courses": protocol["separate_recovery_courses"],
        "robust_flat_prefix_steps": int(
            selected_profile["robust_flat_prefix_steps"]
        ),
        "robust_flat_teacher_model_path": (
            str(
                Path(protocol["robust_flat_teacher_model_path"])
                .resolve()
                .relative_to(PROJECT_ROOT)
            )
            if "robust_flat_teacher_model_path" in protocol
            else None
        ),
        "routes": routes,
        "route_status": statuses,
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
    """開始条件選別、経路探索、快照凍結、修復門、出典不変を実行する。"""
    from sb3_contrib import RecurrentPPO

    output_dir.mkdir(parents=True, exist_ok=False)
    seed_manifest = load_seed_manifest(Path(protocol["seed_manifest_path"]))
    seeds = seed_manifest.for_split("train")
    source_manifest = load_training_teacher_manifest(
        Path(protocol["source_teacher_manifest_path"])
    )
    student = RecurrentPPO.load(Path(protocol["probe_student_model_path"]), device="cpu")
    robust_flat_path = (
        Path(protocol["robust_flat_teacher_model_path"])
        if "robust_flat_teacher_model_path" in protocol
        else None
    )
    monitor_teacher = ManifestTrainingRescueTeacher(
        source_manifest,
        flat_model_path=robust_flat_path,
    )
    template = ClosedLoopHeight1Teacher(
        post_clear_mode="restart_then_flat",
        clearance_blend=1.0,
        handoff_distance=0.45,
        adaptive_handoff=True,
        first_switch_fraction=0.25,
        robust_flat_model_path=robust_flat_path,
    )
    protected = (
        Path(protocol["source_path"]),
        Path(protocol["seed_manifest_path"]),
        Path(protocol["probe_preflight_summary_path"]),
        Path(protocol["probe_student_model_path"]),
        Path(protocol["protected_original_student_model_path"]),
        Path(protocol["source_teacher_manifest_path"]),
    )
    protected = (
        *protected,
        *tuple(Path(path) for path in protocol["protocol_chain_paths"]),
    )
    if robust_flat_path is not None:
        protected = (
            *protected,
            robust_flat_path,
            Path(protocol["robust_flat_teacher_summary_path"]),
        )
    before = {str(path): sha256_file(path) for path in protected}
    profile_replays: dict[str, tuple[LearnerPrefixReplay, ...]] = {}
    screens: list[dict[str, object]] = []
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
        profile_replays[str(profile["name"])] = replays
        screen = screen_profile(profile, replays, source_manifest, template)
        screens.append(screen)
        print(
            json.dumps(
                {
                    "event": "trigger_profile_screened",
                    "profile": profile["name"],
                    "success_count": screen["success_count"],
                    "hard_fall_count": screen["hard_fall_count"],
                    "grounded_trigger_count": screen["grounded_trigger_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    screens.sort(
        key=lambda row: (
            int(row["grounded_trigger_count"]),
            int(row["hard_fall_count"]),
            -int(row["success_count"]),
            float(row["mean_trigger_step"]),
        )
    )
    duration_screens: list[dict[str, object]] = []
    trigger_profile_limit = int(protocol["maximum_profiles_for_route_search"])
    for trigger_screen in screens[:trigger_profile_limit]:
        name = str(trigger_screen["profile_name"])
        for duration in protocol["robust_flat_prefix_step_candidates"]:
            duration_template = copy.copy(template)
            duration_template.robust_flat_max_steps = int(duration)
            duration_screen = screen_profile(
                trigger_screen["profile"],
                profile_replays[name],
                source_manifest,
                duration_template,
            )
            duration_screen["robust_flat_prefix_steps"] = int(duration)
            duration_screens.append(duration_screen)
            print(
                json.dumps(
                    {
                        "event": "flat_duration_screened",
                        "profile": name,
                        "robust_flat_prefix_steps": int(duration),
                        "success_count": duration_screen["success_count"],
                        "hard_fall_count": duration_screen["hard_fall_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    duration_screens.sort(
        key=lambda row: (
            int(row["hard_fall_count"]),
            -int(row["success_count"]),
            int(row["robust_flat_prefix_steps"]),
        )
    )
    candidates = route_candidates(protocol, source_manifest)
    searched_profiles: list[dict[str, object]] = []
    selected_profile: dict[str, object] | None = None
    selected_gate: dict[str, object] | None = None
    maximum_combinations = int(
        protocol["maximum_trigger_flat_duration_combinations_for_route_search"]
    )
    for screen in duration_screens[:maximum_combinations]:
        name = str(screen["profile_name"])
        replays = profile_replays[name]
        route_template = copy.copy(template)
        route_template.robust_flat_max_steps = int(
            screen["robust_flat_prefix_steps"]
        )
        quick_budget = int(protocol.get("quick_route_candidate_budget_per_position", 12))
        expanded_limit = int(
            protocol.get(
                "expanded_route_candidate_limit_per_position",
                protocol["route_candidate_limit_per_position"],
            )
        )
        deferred = {
            int(value)
            for value in protocol.get("deferred_historically_uncovered_positions", [])
        }
        position_results = []
        for replay in replays:
            row = select_route(
                replay,
                candidates,
                route_template,
                perturbation_offsets=tuple(
                    int(value) for value in protocol["perturbation_prefix_offsets"]
                ),
                minimum_robust_successes=int(
                    protocol["minimum_required_robust_offset_successes"]
                ),
                maximum_attempts=quick_budget,
            )
            position_results.append(row)
            print(
                json.dumps(
                    {
                        "event": "route_selected",
                        "profile": name,
                        "position": replay.start_runway_voxels,
                        "status": row["selection_status"],
                        "attempt_count": row["attempt_count"],
                        "exact_success": row["selected_exact_evaluation"]["success"],
                        "search_pass": "quick",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        exact_success_count = sum(
            bool(row["selected_exact_evaluation"]["success"])
            for row in position_results
        )
        expansion_order = [
            index
            for index, replay in enumerate(replays)
            if replay.start_runway_voxels not in deferred
            and not bool(
                position_results[index]["selected_exact_evaluation"]["success"]
            )
        ]
        expansion_order.extend(
            index
            for index, replay in enumerate(replays)
            if replay.start_runway_voxels in deferred
            and not bool(
                position_results[index]["selected_exact_evaluation"]["success"]
            )
        )
        minimum_exact = int(protocol["repair_gate"]["minimum_exact_state_success_count"])
        for index in expansion_order:
            if exact_success_count >= minimum_exact:
                break
            replay = replays[index]
            row = select_route(
                replay,
                candidates,
                route_template,
                perturbation_offsets=tuple(
                    int(value) for value in protocol["perturbation_prefix_offsets"]
                ),
                minimum_robust_successes=int(
                    protocol["minimum_required_robust_offset_successes"]
                ),
                maximum_attempts=expanded_limit,
            )
            position_results[index] = row
            exact_success_count = sum(
                bool(item["selected_exact_evaluation"]["success"])
                for item in position_results
            )
            print(
                json.dumps(
                    {
                        "event": "route_selected",
                        "profile": name,
                        "position": replay.start_runway_voxels,
                        "status": row["selection_status"],
                        "attempt_count": row["attempt_count"],
                        "exact_success": row["selected_exact_evaluation"]["success"],
                        "search_pass": "expanded",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        profile_result = {
            "profile": screen["profile"],
            "profile_name": name,
            "robust_flat_prefix_steps": int(
                screen["robust_flat_prefix_steps"]
            ),
            "screen": screen,
            "position_results": position_results,
        }
        after_search = {str(path): sha256_file(path) for path in protected}
        gate = build_repair_gate(
            profile_result,
            requirements=protocol["repair_gate"],
            sources_unchanged=before == after_search,
        )
        profile_result["repair_gate"] = gate
        searched_profiles.append(profile_result)
        if bool(gate["gate_passed"]):
            selected_profile = profile_result
            selected_gate = gate
            break
    if selected_profile is None:
        selected_profile = max(
            searched_profiles,
            key=lambda row: (
                int(row["repair_gate"]["exact_state_success_count"]),
                int(row["repair_gate"]["robust_position_count"]),
                -int(row["repair_gate"]["exact_state_hard_fall_count"]),
            ),
        )
        selected_gate = dict(selected_profile["repair_gate"])
    selected_replays = profile_replays[str(selected_profile["profile_name"])]
    trigger_manifest_path, trajectory_rows = write_trigger_artifacts(
        output_dir,
        selected_profile["profile"],
        selected_replays,
        offsets=tuple(int(value) for value in protocol["perturbation_prefix_offsets"]),
    )
    teacher_manifest = build_teacher_manifest(
        protocol,
        selected_profile,
        trigger_manifest_path,
        selected_gate,
    )
    teacher_manifest_path = output_dir / "teacher_manifest.json"
    teacher_manifest_path.write_text(
        json.dumps(teacher_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    after = {str(path): sha256_file(path) for path in protected}
    source_files_unchanged = before == after
    if not source_files_unchanged:
        raise RuntimeError("M4R探索中に凍結出典が変更された。")
    summary = {
        "method": "m4r_learner_distribution_teacher_repair",
        "run_name": output_dir.name,
        "stage": "hurdle_single",
        "split": "train",
        "protocol_path": str(protocol["source_path"]),
        "protocol_sha256": protocol["sha256"],
        "seed_manifest": seed_manifest.as_dict(),
        "source_teacher_manifest": source_manifest.as_dict(),
        "trigger_profile_screens": screens,
        "flat_duration_screens": duration_screens,
        "route_candidate_count": len(candidates),
        "searched_profiles": searched_profiles,
        "selected_profile_name": selected_profile["profile_name"],
        "selected_profile": selected_profile,
        "trigger_state_manifest_path": str(trigger_manifest_path.resolve()),
        "trigger_state_manifest_sha256": sha256_file(trigger_manifest_path),
        "trajectory_rows": trajectory_rows,
        "teacher_manifest_path": str(teacher_manifest_path.resolve()),
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "repair_gate": selected_gate,
        "eligible_for_continuous_validation": bool(selected_gate["gate_passed"]),
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
        "source_files_unchanged": source_files_unchanged,
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
        description="学習器誘導状態に対するM4R訓練専用教師を有界修復する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """M4Rの開始条件選別、経路探索、修復門保存を実行する。"""
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
