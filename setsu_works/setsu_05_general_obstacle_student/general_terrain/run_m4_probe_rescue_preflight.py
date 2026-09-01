"""学習器誘導状態からの教師救援を重み更新なしで収集し再検証する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from general_terrain.audit_rescue_demonstrations import find_true_segments, sha256_file
from general_terrain.curriculum import get_curriculum_stage, sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.interactive_collection import collect_rescue_batch
from general_terrain.manifest_training_rescue_teacher import (
    ManifestTrainingRescueTeacher,
    load_training_teacher_manifest,
)
from general_terrain.rescue_profiles import M2_4_MANIFEST_PROFILE, get_rescue_profile
from general_terrain.rescue_reset_manifest import RescueResetSpec, load_rescue_reset_manifest
from general_terrain.seed_manifest import load_seed_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "m4_probe_rescue_preflight_protocol_v1.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "m4_probe_rescue_collection"
REQUIRED_ARRAYS = {
    "observations",
    "student_actions",
    "teacher_actions",
    "executed_actions",
    "teacher_mask",
    "rescue_ids",
    "teacher_stages",
}
TRIGGER_FLOAT_FIELDS = (
    "x_position",
    "orientation_error",
    "angular_velocity",
)
TRIGGER_INTEGER_FIELDS = (
    "stall_steps",
    "raw_clearances",
    "recovered_obstacles",
)


def _resolve_project_path(value: str) -> Path:
    """規約内の相対パスをプロジェクト配下だけへ解決する。"""
    resolved = (PROJECT_ROOT / value).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M4前置復核の出典はプロジェクト配下でなければならない。")
    return resolved


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、出典ハッシュ、零更新、教師隔離条件を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M4前置復核規約は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M4前置復核は単一低壁の訓練区分だけを使用できる。")
    if payload.get("rescue_profile") != M2_4_MANIFEST_PROFILE:
        raise ValueError("M4前置復核の連続救援設定が凍結値と一致しない。")
    if int(payload["train_collection_episodes"]) != 11:
        raise ValueError("M4前置復核の訓練収集は十一回でなければならない。")
    zero_fields = (
        "validation_episodes",
        "holdout_episodes",
        "probe_student_update_steps",
        "protected_original_student_update_steps",
        "teacher_update_steps",
        "teacher_interventions_in_final_student_test",
    )
    if any(int(payload[field]) != 0 for field in zero_fields):
        raise ValueError("M4前置復核では評価アクセス、重み更新、最終介入を禁止する。")
    disabled_fields = (
        "validation_teacher_enabled",
        "holdout_teacher_enabled",
        "final_student_test_teacher_enabled",
    )
    if any(bool(payload.get(field, True)) for field in disabled_fields):
        raise ValueError("検証、留保、最終学生試験では教師を完全停止しなければならない。")
    path_fields = (
        "seed_manifest_path",
        "reset_manifest_path",
        "teacher_manifest_path",
        "m3_1_summary_path",
        "probe_student_model_path",
        "protected_original_student_model_path",
    )
    resolved_paths: dict[str, Path] = {}
    for field in path_fields:
        resolved = _resolve_project_path(str(payload[field]))
        expected_hash = str(payload[f"{field.removesuffix('_path')}_sha256"]).lower()
        if sha256_file(resolved) != expected_hash:
            raise ValueError(f"M4前置復核の出典ハッシュが一致しない: {resolved}")
        resolved_paths[field] = resolved
    m3_summary = json.loads(
        resolved_paths["m3_1_summary_path"].read_text(encoding="utf-8")
    )
    if bool(m3_summary.get("eligible_for_m4", True)):
        raise ValueError("M3.1不合格後だけが本前置復核の対象である。")
    if m3_summary.get("selected_student_path") is not None:
        raise ValueError("M3.1で正式採用済み学生がある場合は隔離探針を使用できない。")
    candidate_hashes = {
        str(item.get("checkpoint_sha256", "")).lower()
        for item in m3_summary.get("candidates", [])
    }
    if str(payload["probe_student_model_sha256"]).lower() not in candidate_hashes:
        raise ValueError("探針チェックポイントがM3.1候補集合に存在しない。")
    return {
        **payload,
        **resolved_paths,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    """二配列の最大絶対差を倍精度で返す。"""
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _load_branch(path: Path) -> dict[str, np.ndarray]:
    """保存分岐の必須配列、形状、有限性、制御器対応を検査する。"""
    with np.load(path, allow_pickle=False) as archive:
        if not REQUIRED_ARRAYS.issubset(set(archive.files)):
            raise ValueError(f"成功分岐の必須配列が不足している: {path}")
        arrays = {name: archive[name].copy() for name in REQUIRED_ARRAYS}
    length = len(arrays["observations"])
    expected_shapes = {
        "observations": (length, 95),
        "student_actions": (length, 6),
        "teacher_actions": (length, 6),
        "executed_actions": (length, 6),
        "teacher_mask": (length,),
        "rescue_ids": (length,),
        "teacher_stages": (length,),
    }
    for name, expected in expected_shapes.items():
        if arrays[name].shape != expected:
            raise ValueError(f"成功分岐の配列形状が不正である: {path}, {name}")
    for name in (
        "observations",
        "student_actions",
        "teacher_actions",
        "executed_actions",
    ):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"成功分岐に非有限値がある: {path}, {name}")
    mask = arrays["teacher_mask"].astype(bool)
    if not np.array_equal(arrays["executed_actions"][mask], arrays["teacher_actions"][mask]):
        raise ValueError(f"教師制御歩の実行動作がラベルと一致しない: {path}")
    if not np.array_equal(arrays["executed_actions"][~mask], arrays["student_actions"][~mask]):
        raise ValueError(f"学生制御歩の実行動作が予測と一致しない: {path}")
    if np.any(arrays["rescue_ids"][~mask] != 0):
        raise ValueError(f"学生制御歩に救援番号が混入している: {path}")
    return arrays


def audit_trigger_state(
    episode: Mapping[str, object],
    reference: RescueResetSpec,
    *,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """探針の救援開始状態を原学生境界と比較し誘導状態の変化を判定する。"""
    starts = [event for event in episode["events"] if event["event"] == "start"]
    if len(starts) != 1:
        return {
            "seed": int(episode["seed"]),
            "start_runway_voxels": int(episode["start_runway_voxels"]),
            "start_event_count": len(starts),
            "learner_induced_state_changed": False,
            "comparison_available": False,
            "failure_reason": "single_start_event_not_available",
        }
    event = starts[0]
    step_delta = int(event["step"]) - int(reference.prefix_steps)
    float_deltas = {
        field: float(event[field]) - float(getattr(reference, field))
        for field in TRIGGER_FLOAT_FIELDS
    }
    integer_deltas = {
        field: int(event[field]) - int(getattr(reference, field))
        for field in TRIGGER_INTEGER_FIELDS
    }
    changed = bool(
        step_delta != 0
        or any(abs(value) > tolerance for value in float_deltas.values())
        or any(value != 0 for value in integer_deltas.values())
    )
    return {
        "seed": int(episode["seed"]),
        "start_runway_voxels": int(episode["start_runway_voxels"]),
        "start_event_count": 1,
        "comparison_available": True,
        "learner_induced_state_changed": changed,
        "probe_trigger": dict(event),
        "original_student_reference": asdict(reference),
        "step_delta": step_delta,
        "float_deltas": float_deltas,
        "integer_deltas": integer_deltas,
    }


def replay_accepted_branch(episode: Mapping[str, object]) -> dict[str, object]:
    """保存動作を初期状態から再生し、全観測と安全成功終端を照合する。"""
    seed = int(episode["seed"])
    position = int(episode["start_runway_voxels"])
    branch_path = Path(str(episode["branch_path"]))
    sidecar_path = branch_path.with_suffix(".json")
    arrays = _load_branch(branch_path)
    mask = arrays["teacher_mask"].astype(bool)
    segments = find_true_segments(mask)
    starts = [event for event in episode["events"] if event["event"] == "start"]
    if len(starts) != 1:
        raise ValueError(f"成功分岐の救援開始イベントが一個ではない: {seed}")
    trigger_step = int(starts[0]["step"])
    if len(segments) != 1 or segments[0] != (trigger_step, len(mask)):
        raise ValueError(f"教師制御が開始点から終端まで単一連続区間ではない: {seed}")
    expected_prefix = (
        f"manifest_takeover_x{position}:"
        if position not in {21, 22}
        else f"manifest_fallback_x{position}:"
    )
    if not all(str(stage).startswith(expected_prefix) for stage in arrays["teacher_stages"][mask]):
        raise ValueError(f"成功分岐に目録外の教師段階が含まれている: {seed}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not bool(sidecar.get("accepted", False)):
        raise ValueError(f"成功分岐の側車記録が不合格を示している: {seed}")
    if sidecar["metadata"].get("stage") != "hurdle_single":
        raise ValueError(f"成功分岐の段階メタデータが不正である: {seed}")
    if sidecar["metadata"].get("split") != "train":
        raise ValueError(f"成功分岐の区分メタデータが不正である: {seed}")
    course = sample_curriculum_course(seed, "hurdle_single", "train")
    if int(course.obstacles[0].start_x) != position:
        raise ValueError(f"再生成コースの開始位置が一致しない: {seed}")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    maximum_difference = 0.0
    terminated = False
    truncated = False
    try:
        observation, info = environment.reset(seed=seed)
        for index, action in enumerate(arrays["executed_actions"]):
            maximum_difference = max(
                maximum_difference,
                _maximum_difference(
                    np.asarray(observation, dtype=np.float32),
                    arrays["observations"][index],
                ),
            )
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(action, dtype=np.float32)
            )
            if (terminated or truncated) and index != len(mask) - 1:
                raise RuntimeError(f"成功分岐が保存終端より早く終了した: {seed}")
        if not (terminated or truncated):
            raise RuntimeError(f"成功分岐が保存終端で終了しなかった: {seed}")
    finally:
        environment.close()
    replay_success = bool(
        info["course_complete"]
        and not info["hard_fall"]
        and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
    )
    return {
        "seed": seed,
        "start_runway_voxels": position,
        "steps": len(mask),
        "probe_student_prefix_steps": trigger_step,
        "teacher_steps": int(np.count_nonzero(mask)),
        "teacher_segments": [list(segment) for segment in segments],
        "single_continuous_teacher_segment": len(segments) == 1,
        "replay_success": replay_success,
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_observation_difference": maximum_difference,
        "all_step_observations_exact": maximum_difference == 0.0,
        "branch_path": str(branch_path.resolve()),
        "branch_sha256": sha256_file(branch_path),
        "sidecar_path": str(sidecar_path.resolve()),
        "sidecar_sha256": sha256_file(sidecar_path),
    }


def audit_failed_branch_absence(
    output_dir: Path,
    episodes: list[Mapping[str, object]],
) -> dict[str, object]:
    """不成功回に分岐ファイルがなく成功回だけが保存されたことを検査する。"""
    unexpected: list[str] = []
    expected: set[Path] = set()
    for episode in episodes:
        stem = (
            f"seed_{int(episode['seed'])}_"
            f"x{int(episode['start_runway_voxels'])}_rescued"
        )
        npz_path = output_dir / "branches" / f"{stem}.npz"
        json_path = output_dir / "branches" / f"{stem}.json"
        if bool(episode["branch_accepted"]):
            expected.update({npz_path.resolve(), json_path.resolve()})
        else:
            for path in (npz_path, json_path):
                if path.exists():
                    unexpected.append(str(path.resolve()))
    actual = {
        path.resolve()
        for path in (output_dir / "branches").iterdir()
        if path.suffix in {".npz", ".json"}
    }
    unexpected.extend(str(path) for path in sorted(actual - expected))
    missing = [str(path) for path in sorted(expected - actual)]
    return {
        "failed_branch_files_absent": not unexpected,
        "accepted_branch_files_complete": not missing,
        "unexpected_files": sorted(set(unexpected)),
        "missing_files": missing,
        "expected_file_count": len(expected),
        "actual_file_count": len(actual),
    }


def build_gate(
    collection: Mapping[str, object],
    replay_rows: list[Mapping[str, object]],
    trigger_rows: list[Mapping[str, object]],
    failed_file_audit: Mapping[str, object],
    *,
    requirements: Mapping[str, object],
    source_files_unchanged: bool,
) -> dict[str, object]:
    """成功、安全、誘導状態、厳密再生、出典不変から前置門を判定する。"""
    episodes = list(collection["episodes"])
    accepted_positions = {
        int(row["start_runway_voxels"])
        for row in episodes
        if bool(row["branch_accepted"])
    }
    changed_count = sum(
        bool(row["learner_induced_state_changed"]) for row in trigger_rows
    )
    timeout_count = sum(
        event["event"] == "teacher_timeout"
        for episode in episodes
        for event in episode["events"]
    )
    accepted_count = int(collection["accepted_branch_count"])
    checks = {
        "collection_success_requirement_passed": int(collection["collection_success_count"])
        >= int(requirements["minimum_collection_success_count"]),
        "hard_fall_requirement_passed": int(collection["hard_fall_count"])
        <= int(requirements["maximum_hard_fall_count"]),
        "accepted_branch_requirement_passed": accepted_count
        >= int(requirements["minimum_accepted_branch_count"]),
        "route_diversity_requirement_passed": len(accepted_positions)
        >= int(requirements["minimum_distinct_accepted_route_positions"]),
        "learner_induced_state_requirement_passed": changed_count
        >= int(requirements["minimum_learner_induced_changed_state_count"]),
        "exact_replay_requirement_passed": len(replay_rows) == accepted_count
        and all(
            bool(row["replay_success"])
            and bool(row["all_step_observations_exact"])
            for row in replay_rows
        ),
        "continuous_segment_requirement_passed": all(
            bool(row["single_continuous_teacher_segment"])
            for row in replay_rows
        ),
        "failed_branch_absence_requirement_passed": bool(
            failed_file_audit["failed_branch_files_absent"]
            and failed_file_audit["accepted_branch_files_complete"]
        ),
        "teacher_timeout_requirement_passed": timeout_count == 0,
        "source_immutability_requirement_passed": source_files_unchanged,
    }
    gate_passed = all(checks.values())
    return {
        "gate_name": "m4_learner_induced_rescue_preflight_gate_v1",
        **requirements,
        "accepted_route_positions": sorted(accepted_positions),
        "learner_induced_changed_state_count": changed_count,
        "teacher_timeout_event_count": timeout_count,
        **checks,
        "gate_passed": gate_passed,
        "eligible_for_later_bounded_aggregation_update": gate_passed,
        "direct_m4_weight_update_executed": False,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """重みを更新しないM4前置救援復核の引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="隔離探針の誘導状態から訓練専用教師救援を復核する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """十一訓練コースを収集し、零更新と成功分岐の厳密再生を保存する。"""
    from sb3_contrib import RecurrentPPO

    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    seed_manifest = load_seed_manifest(protocol["seed_manifest_path"])
    reset_manifest = load_rescue_reset_manifest(protocol["reset_manifest_path"])
    teacher_manifest = load_training_teacher_manifest(protocol["teacher_manifest_path"])
    train_seeds = seed_manifest.for_split("train")
    if tuple(state.seed for state in reset_manifest.states) != train_seeds:
        raise ValueError("前置復核の訓練順序が凍結救援境界目録と一致しない。")
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_paths = (
        Path(protocol["source_path"]),
        Path(protocol["seed_manifest_path"]),
        Path(protocol["reset_manifest_path"]),
        Path(protocol["teacher_manifest_path"]),
        Path(protocol["m3_1_summary_path"]),
        Path(protocol["probe_student_model_path"]),
        Path(protocol["protected_original_student_model_path"]),
    )
    before = {str(path): sha256_file(path) for path in protected_paths}
    probe_student = RecurrentPPO.load(protocol["probe_student_model_path"], device="cpu")
    teacher = ManifestTrainingRescueTeacher(teacher_manifest)
    collection = collect_rescue_batch(
        probe_student,
        teacher,
        seeds=train_seeds,
        stage=get_curriculum_stage("hurdle_single"),
        output_dir=output_dir,
        rescue_config=get_rescue_profile(M2_4_MANIFEST_PROFILE),
        rescue_profile_name=M2_4_MANIFEST_PROFILE,
    )
    collection_path = output_dir / "collection_summary.json"
    collection_path.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    specs = {state.seed: state for state in reset_manifest.states}
    trigger_rows = [
        audit_trigger_state(episode, specs[int(episode["seed"])])
        for episode in collection["episodes"]
    ]
    replay_rows = [
        replay_accepted_branch(episode)
        for episode in collection["episodes"]
        if bool(episode["branch_accepted"])
    ]
    failed_file_audit = audit_failed_branch_absence(
        output_dir,
        list(collection["episodes"]),
    )
    after = {str(path): sha256_file(path) for path in protected_paths}
    source_files_unchanged = before == after
    gate = build_gate(
        collection,
        replay_rows,
        trigger_rows,
        failed_file_audit,
        requirements=protocol["gate"],
        source_files_unchanged=source_files_unchanged,
    )
    branch_manifest = {
        "version": "m4_probe_accepted_correction_branch_manifest_v1",
        "frozen": True,
        "purpose": "learner_induced_state_teacher_correction_preflight_only",
        "stage": "hurdle_single",
        "split": "train",
        "probe_candidate_id": protocol["probe_candidate_id"],
        "probe_disposition": protocol["probe_disposition"],
        "probe_student_model_sha256": before[str(protocol["probe_student_model_path"])],
        "protected_original_student_model_sha256": before[
            str(protocol["protected_original_student_model_path"])
        ],
        "teacher_manifest_sha256": teacher_manifest.sha256,
        "probe_student_weights_updated": False,
        "protected_original_student_weights_updated": False,
        "teacher_weights_updated": False,
        "teacher_training_only": True,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "branches": replay_rows,
    }
    branch_manifest_path = output_dir / "accepted_probe_branch_manifest.json"
    branch_manifest_path.write_text(
        json.dumps(branch_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "method": "m4_learner_induced_state_teacher_rescue_preflight",
        "run_name": args.run_name,
        "purpose": protocol["purpose"],
        "stage": "hurdle_single",
        "split": "train",
        "protocol_path": str(protocol["source_path"]),
        "protocol_sha256": protocol["sha256"],
        "seed_manifest": seed_manifest.as_dict(),
        "reset_manifest": reset_manifest.as_dict(),
        "teacher_manifest": teacher_manifest.as_dict(),
        "probe_candidate_id": protocol["probe_candidate_id"],
        "probe_disposition": protocol["probe_disposition"],
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_files_unchanged": source_files_unchanged,
        "probe_student_model_unchanged": before[str(protocol["probe_student_model_path"])]
        == after[str(protocol["probe_student_model_path"])],
        "protected_original_student_model_unchanged": before[
            str(protocol["protected_original_student_model_path"])
        ]
        == after[str(protocol["protected_original_student_model_path"])],
        "probe_student_weights_updated": False,
        "protected_original_student_weights_updated": False,
        "teacher_weights_updated": False,
        "probe_student_update_steps": 0,
        "protected_original_student_update_steps": 0,
        "teacher_update_steps": 0,
        "ppo_training_steps": 0,
        "train_collection_episodes": 11,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "collection_summary_path": str(collection_path.resolve()),
        "collection_summary_sha256": sha256_file(collection_path),
        "collection": collection,
        "trigger_state_audit": trigger_rows,
        "replay_rows": replay_rows,
        "failed_branch_file_audit": failed_file_audit,
        "accepted_probe_branch_manifest_path": str(branch_manifest_path.resolve()),
        "accepted_probe_branch_manifest_sha256": sha256_file(branch_manifest_path),
        "m4_probe_preflight_gate": gate,
        "eligible_for_later_bounded_aggregation_update": bool(
            gate["eligible_for_later_bounded_aggregation_update"]
        ),
        "direct_m4_weight_update_executed": False,
        "eligible_for_final_student_test": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
