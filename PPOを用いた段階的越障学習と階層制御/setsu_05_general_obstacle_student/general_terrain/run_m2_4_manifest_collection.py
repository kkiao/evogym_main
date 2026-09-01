"""M2.4の訓練専用教師で十一コースを連続収集し厳密再生する。"""

from __future__ import annotations

import argparse
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
from general_terrain.rescue_profiles import (
    M2_4_MANIFEST_PROFILE,
    get_rescue_profile,
)
from general_terrain.rescue_reset_manifest import load_rescue_reset_manifest
from general_terrain.seed_manifest import load_seed_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m2_4_manifest_collection_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "interactive_rescue_collection"
REQUIRED_ARRAYS = {
    "observations",
    "student_actions",
    "teacher_actions",
    "executed_actions",
    "teacher_mask",
    "rescue_ids",
    "teacher_stages",
}


def _resolve_project_path(value: str) -> Path:
    """規約内の相対パスをプロジェクト配下だけへ解決する。"""
    resolved = (PROJECT_ROOT / value).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M2.4の出典はプロジェクト配下になければならない。")
    return resolved


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、全出典ハッシュ、訓練限定条件、合格門を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M2.4収集規約は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M2.4は単一低壁の訓練区分だけを使用できる。")
    if payload.get("rescue_profile") != M2_4_MANIFEST_PROFILE:
        raise ValueError("M2.4の連続接管設定が凍結値と一致しない。")
    if int(payload["train_episodes"]) != 11:
        raise ValueError("M2.4の訓練収集は十一回でなければならない。")
    zero_fields = (
        "validation_episodes",
        "holdout_episodes",
        "student_update_steps",
        "teacher_update_steps",
        "teacher_interventions_in_final_student_test",
    )
    if any(int(payload[field]) != 0 for field in zero_fields):
        raise ValueError("M2.4収集では評価アクセス、重み更新、最終介入を禁止する。")
    if bool(payload.get("final_student_test_teacher_enabled", True)):
        raise ValueError("最終学生試験では教師を完全停止しなければならない。")
    path_fields = (
        "seed_manifest_path",
        "reset_manifest_path",
        "teacher_manifest_path",
        "teacher_gate_summary_path",
        "source_student_model_path",
    )
    resolved_paths: dict[str, Path] = {}
    for field in path_fields:
        resolved = _resolve_project_path(str(payload[field]))
        expected_hash = str(payload[f"{field.removesuffix('_path')}_sha256"])
        if sha256_file(resolved) != expected_hash:
            raise ValueError(f"M2.4出典ハッシュが一致しない: {resolved}")
        resolved_paths[field] = resolved
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
    return float(
        np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
    )


def _load_branch(path: Path) -> dict[str, np.ndarray]:
    """成功分岐の必須配列、形状、有限性、制御器対応を検査する。"""
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
    if not np.array_equal(
        arrays["executed_actions"][mask],
        arrays["teacher_actions"][mask],
    ):
        raise ValueError(f"教師制御歩の実行動作がラベルと一致しない: {path}")
    if not np.array_equal(
        arrays["executed_actions"][~mask],
        arrays["student_actions"][~mask],
    ):
        raise ValueError(f"学生制御歩の実行動作が予測と一致しない: {path}")
    return arrays


def replay_accepted_branch(
    episode: Mapping[str, object],
    *,
    prefix_steps: int,
) -> dict[str, object]:
    """保存動作を初期状態から再生し、全観測と安全成功終端を照合する。"""
    seed = int(episode["seed"])
    branch_path = Path(str(episode["branch_path"]))
    sidecar_path = branch_path.with_suffix(".json")
    arrays = _load_branch(branch_path)
    mask = arrays["teacher_mask"].astype(bool)
    segments = find_true_segments(mask)
    if len(segments) != 1 or segments[0] != (prefix_steps, len(mask)):
        raise ValueError(f"連続教師区間が凍結接管境界と一致しない: {seed}")
    if not all(
        str(stage).startswith(f"manifest_takeover_x{episode['start_runway_voxels']}:")
        for stage in arrays["teacher_stages"][mask]
    ):
        raise ValueError(f"成功分岐に目録外の教師段階が含まれている: {seed}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar["metadata"].get("stage") != "hurdle_single":
        raise ValueError(f"成功分岐の段階メタデータが不正である: {seed}")
    if sidecar["metadata"].get("split") != "train":
        raise ValueError(f"成功分岐の区分メタデータが不正である: {seed}")
    course = sample_curriculum_course(seed, "hurdle_single", "train")
    if int(course.obstacles[0].start_x) != int(episode["start_runway_voxels"]):
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
        "start_runway_voxels": int(episode["start_runway_voxels"]),
        "steps": len(mask),
        "prefix_steps": prefix_steps,
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


def _trigger_alignment(
    episode: Mapping[str, object],
    *,
    prefix_steps: int,
    trigger_reason: str,
) -> bool:
    """収集開始イベントが凍結学生境界と理由へ一致するかを返す。"""
    starts = [event for event in episode["events"] if event["event"] == "start"]
    return bool(
        len(starts) == 1
        and int(starts[0]["step"]) == prefix_steps
        and str(starts[0]["reason"]) == trigger_reason
    )


def build_gate(
    collection: Mapping[str, object],
    replay_rows: list[Mapping[str, object]],
    *,
    requirements: Mapping[str, object],
    trigger_alignment: bool,
    source_files_unchanged: bool,
) -> dict[str, object]:
    """成功、安全、連続性、厳密再生、出典不変からM2.4門を判定する。"""
    episodes = collection["episodes"]
    accepted_positions = {
        int(row["start_runway_voxels"])
        for row in episodes
        if bool(row["branch_accepted"])
    }
    safe_stall_positions = {
        int(row["start_runway_voxels"])
        for row in episodes
        if not bool(row["course_complete"])
        and not bool(row["hard_fall"])
        and str(row["failure_reason"]) == "stall_limit"
    }
    expected_routes = set(int(value) for value in requirements["required_route_positions"])
    expected_stalls = set(
        int(value) for value in requirements["required_safe_stall_positions"]
    )
    checks = {
        "collection_success_requirement_passed": int(
            collection["collection_success_count"]
        )
        >= int(requirements["minimum_collection_success_count"]),
        "hard_fall_requirement_passed": int(collection["hard_fall_count"])
        <= int(requirements["maximum_hard_fall_count"]),
        "accepted_branch_requirement_passed": int(
            collection["accepted_branch_count"]
        )
        >= int(requirements["minimum_accepted_branch_count"]),
        "route_position_requirement_passed": accepted_positions == expected_routes,
        "safe_stall_requirement_passed": safe_stall_positions == expected_stalls,
        "exact_replay_requirement_passed": len(replay_rows)
        == int(requirements["required_exact_replay_count"])
        and all(
            bool(row["replay_success"])
            and bool(row["all_step_observations_exact"])
            for row in replay_rows
        ),
        "continuous_segment_requirement_passed": all(
            bool(row["single_continuous_teacher_segment"])
            for row in replay_rows
        ),
        "trigger_alignment_requirement_passed": trigger_alignment,
        "source_immutability_requirement_passed": source_files_unchanged,
    }
    gate_passed = all(checks.values())
    return {
        "gate_name": "m2_4_manifest_continuous_collection_gate_v1",
        **requirements,
        "accepted_route_positions": sorted(accepted_positions),
        "safe_stall_positions": sorted(safe_stall_positions),
        **checks,
        "gate_passed": gate_passed,
        "eligible_for_m2_5": gate_passed,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """正式訓練を伴わないM2.4収集の引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="凍結学生と訓練専用目録教師でM2.4分岐を収集する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """十一訓練コースを収集し、成功分岐の厳密再生と合格門を保存する。"""
    from sb3_contrib import RecurrentPPO

    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    seed_manifest = load_seed_manifest(protocol["seed_manifest_path"])
    reset_manifest = load_rescue_reset_manifest(protocol["reset_manifest_path"])
    teacher_manifest = load_training_teacher_manifest(
        protocol["teacher_manifest_path"]
    )
    train_seeds = seed_manifest.for_split("train")
    if tuple(state.seed for state in reset_manifest.states) != train_seeds:
        raise ValueError("M2.4の訓練順序が凍結救援境界と一致しない。")
    if reset_manifest.student_model_sha256 != protocol["source_student_model_sha256"]:
        raise ValueError("M2.4の学生ハッシュが救援境界目録と一致しない。")
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_paths = tuple(
        Path(protocol[field])
        for field in (
            "seed_manifest_path",
            "reset_manifest_path",
            "teacher_manifest_path",
            "teacher_gate_summary_path",
            "source_student_model_path",
        )
    )
    before = {str(path): sha256_file(path) for path in protected_paths}
    student = RecurrentPPO.load(protocol["source_student_model_path"], device="cpu")
    teacher = ManifestTrainingRescueTeacher(teacher_manifest)
    collection = collect_rescue_batch(
        student,
        teacher,
        seeds=train_seeds,
        stage=get_curriculum_stage("hurdle_single"),
        output_dir=output_dir,
        rescue_config=get_rescue_profile(M2_4_MANIFEST_PROFILE),
        rescue_profile_name=M2_4_MANIFEST_PROFILE,
    )
    (output_dir / "collection_summary.json").write_text(
        json.dumps(collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    specs = {state.seed: state for state in reset_manifest.states}
    trigger_alignment = all(
        _trigger_alignment(
            episode,
            prefix_steps=specs[int(episode["seed"])].prefix_steps,
            trigger_reason=specs[int(episode["seed"])].trigger_reason,
        )
        for episode in collection["episodes"]
    )
    replay_rows = [
        replay_accepted_branch(
            episode,
            prefix_steps=specs[int(episode["seed"])].prefix_steps,
        )
        for episode in collection["episodes"]
        if bool(episode["branch_accepted"])
    ]
    after = {str(path): sha256_file(path) for path in protected_paths}
    source_files_unchanged = after == before
    gate = build_gate(
        collection,
        replay_rows,
        requirements=protocol["gate"],
        trigger_alignment=trigger_alignment,
        source_files_unchanged=source_files_unchanged,
    )
    branch_manifest = {
        "version": "m2_4_accepted_rescue_branch_manifest_v1",
        "frozen": True,
        "stage": "hurdle_single",
        "split": "train",
        "teacher_training_only": True,
        "source_student_model_sha256": before[
            str(protocol["source_student_model_path"])
        ],
        "teacher_manifest_sha256": teacher_manifest.sha256,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "branches": replay_rows,
    }
    branch_manifest_path = output_dir / "accepted_branch_manifest.json"
    branch_manifest_path.write_text(
        json.dumps(branch_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "method": "m2_4_manifest_continuous_interactive_rescue_collection",
        "run_name": args.run_name,
        "stage": "hurdle_single",
        "split": "train",
        "protocol_path": str(protocol["source_path"]),
        "protocol_sha256": protocol["sha256"],
        "seed_manifest": seed_manifest.as_dict(),
        "reset_manifest": reset_manifest.as_dict(),
        "teacher_manifest": teacher_manifest.as_dict(),
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_files_unchanged": source_files_unchanged,
        "student_model_unchanged": before[
            str(protocol["source_student_model_path"])
        ]
        == after[str(protocol["source_student_model_path"])],
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "student_update_steps": 0,
        "teacher_update_steps": 0,
        "train_episodes": 11,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "collection": collection,
        "trigger_step_alignment": trigger_alignment,
        "replay_rows": replay_rows,
        "accepted_branch_manifest_path": str(branch_manifest_path.resolve()),
        "accepted_branch_manifest_sha256": sha256_file(branch_manifest_path),
        "m2_4_gate": gate,
        "eligible_for_m2_5": bool(gate["eligible_for_m2_5"]),
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
