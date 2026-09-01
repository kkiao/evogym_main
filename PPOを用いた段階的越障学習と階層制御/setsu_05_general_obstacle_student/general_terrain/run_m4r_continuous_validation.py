"""M4R訓練専用前動作屏障を十一種で連続収集し厳密再生する。"""

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
from general_terrain.m4r_learner_distribution_teacher import (
    M4RLearnerDistributionTeacher,
    load_m4r_teacher_manifest,
)
from general_terrain.run_m4_probe_rescue_preflight import (
    _load_branch,
    audit_failed_branch_absence,
)
from general_terrain.seed_manifest import load_seed_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "m4r_continuous_validation_protocol_v1.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "m4r_continuous_validation"


def _resolve_project_path(value: str) -> Path:
    """規約内の相対パスをプロジェクト配下だけへ解決する。"""
    resolved = (PROJECT_ROOT / value).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M4R連続復核の出典はプロジェクト配下でなければならない。")
    return resolved


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、修復合格、零更新、教師隔離、出典ハッシュを検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M4R連続復核規約は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M4R連続復核は単一低壁の訓練区分だけを使用できる。")
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
        raise ValueError("M4R連続復核では重み更新と評価教師介入を禁止する。")
    for field in (
        "validation_teacher_enabled",
        "holdout_teacher_enabled",
        "final_student_test_teacher_enabled",
    ):
        if bool(payload.get(field, True)):
            raise ValueError("検証、留保、最終学生試験では教師を停止しなければならない。")
    if int(payload["train_collection_episodes"]) != 11:
        raise ValueError("M4R連続復核は十一訓練種でなければならない。")
    fields = (
        "seed_manifest_path",
        "repair_protocol_path",
        "repair_summary_path",
        "teacher_manifest_path",
        "probe_student_model_path",
        "protected_original_student_model_path",
    )
    if "runner_path" in payload:
        fields = (*fields, "runner_path")
    resolved: dict[str, Path] = {}
    for field in fields:
        file_path = _resolve_project_path(str(payload[field]))
        expected = str(payload[f"{field.removesuffix('_path')}_sha256"]).lower()
        if sha256_file(file_path) != expected:
            raise ValueError(f"M4R連続復核の出典ハッシュが一致しない: {file_path}")
        resolved[field] = file_path
    repair = json.loads(resolved["repair_summary_path"].read_text(encoding="utf-8"))
    if not bool(repair["repair_gate"]["gate_passed"]):
        raise ValueError("M4R修復門を通過していない教師は連続復核へ使用できない。")
    if repair.get("method") != "m4r_global_teacher_disagreement_verified_portfolio_repair":
        raise ValueError("M4R修復要約の方式が前動作組合せ屏障と一致しない。")
    return {
        **payload,
        **resolved,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    """二配列の最大絶対差を倍精度で返す。"""
    if left.size == 0:
        return 0.0
    return float(
        np.max(
            np.abs(left.astype(np.float64) - right.astype(np.float64))
        )
    )


def replay_accepted_branch(episode: Mapping[str, object]) -> dict[str, object]:
    """保存された前動作屏障分岐を初期状態から全歩厳密再生する。"""
    seed = int(episode["seed"])
    position = int(episode["start_runway_voxels"])
    branch_path = Path(str(episode["branch_path"]))
    sidecar_path = branch_path.with_suffix(".json")
    arrays = _load_branch(branch_path)
    mask = arrays["teacher_mask"].astype(bool)
    segments = find_true_segments(mask)
    starts = [event for event in episode["events"] if event["event"] == "start"]
    if len(starts) != 1:
        raise ValueError(f"M4R成功分岐の救援開始イベントが一個ではない: {seed}")
    if int(starts[0]["step"]) != 0:
        raise ValueError(f"M4R前動作屏障が零歩で接管していない: {seed}")
    if str(starts[0]["reason"]) != "teacher_disagreement":
        raise ValueError(f"M4R前動作屏障の接管理由が不正である: {seed}")
    if len(segments) != 1 or segments[0] != (0, len(mask)):
        raise ValueError(f"M4R教師制御が零歩から終端まで連続していない: {seed}")
    if not all(
        str(stage).startswith("m4r_")
        for stage in arrays["teacher_stages"][mask]
    ):
        raise ValueError(f"M4R成功分岐に目録外の教師段階が含まれている: {seed}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not bool(sidecar.get("accepted", False)):
        raise ValueError(f"M4R成功分岐の側車記録が不合格を示している: {seed}")
    if sidecar["metadata"].get("stage") != "hurdle_single":
        raise ValueError(f"M4R成功分岐の段階メタデータが不正である: {seed}")
    if sidecar["metadata"].get("split") != "train":
        raise ValueError(f"M4R成功分岐の区分メタデータが不正である: {seed}")
    course = sample_curriculum_course(seed, "hurdle_single", "train")
    if int(course.obstacles[0].start_x) != position:
        raise ValueError(f"M4R再生成コースの開始位置が一致しない: {seed}")
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
                raise RuntimeError(f"M4R成功分岐が保存終端より早く終了した: {seed}")
        if not (terminated or truncated):
            raise RuntimeError(f"M4R成功分岐が保存終端で終了しなかった: {seed}")
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
        "student_prefix_steps": 0,
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


def build_gate(
    collection: Mapping[str, object],
    replay_rows: list[Mapping[str, object]],
    failed_file_audit: Mapping[str, object],
    *,
    requirements: Mapping[str, object],
    source_files_unchanged: bool,
) -> dict[str, object]:
    """連続成功、安全、零歩接管、厳密再生、出典不変から門を判定する。"""
    episodes = list(collection["episodes"])
    starts = [
        event
        for episode in episodes
        for event in episode["events"]
        if event["event"] == "start"
    ]
    pre_action_count = sum(
        int(event["step"]) == 0
        and str(event["reason"]) == "teacher_disagreement"
        and not bool(event["upper_body_grounded"])
        for event in starts
    )
    timeout_count = sum(
        event["event"] == "teacher_timeout"
        for episode in episodes
        for event in episode["events"]
    )
    accepted_count = int(collection["accepted_branch_count"])
    checks = {
        "collection_success_requirement_passed": int(
            collection["collection_success_count"]
        )
        >= int(requirements["minimum_collection_success_count"]),
        "hard_fall_requirement_passed": int(collection["hard_fall_count"])
        <= int(requirements["maximum_hard_fall_count"]),
        "accepted_branch_requirement_passed": accepted_count
        >= int(requirements["minimum_accepted_branch_count"]),
        "pre_action_trigger_requirement_passed": pre_action_count
        >= int(requirements["minimum_pre_action_trigger_count"]),
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
    return {
        "gate_name": "m4r_pre_action_shield_continuous_gate_v1",
        **requirements,
        "pre_action_trigger_count": pre_action_count,
        "teacher_timeout_event_count": timeout_count,
        **checks,
        "gate_passed": all(checks.values()),
        "eligible_for_training_demonstration_use": all(checks.values()),
        "eligible_as_off_distribution_physical_recovery_evidence": False,
        "direct_student_weight_update_executed": False,
    }


def run(protocol: dict[str, object], output_dir: Path) -> dict[str, object]:
    """十一訓練種の連続収集、成功分岐保存、厳密再生を零更新で実行する。"""
    from sb3_contrib import RecurrentPPO

    output_dir.mkdir(parents=True, exist_ok=False)
    seed_manifest = load_seed_manifest(Path(protocol["seed_manifest_path"]))
    seeds = seed_manifest.for_split("train")
    teacher_manifest = load_m4r_teacher_manifest(Path(protocol["teacher_manifest_path"]))
    protected = (
        Path(protocol["source_path"]),
        Path(protocol["seed_manifest_path"]),
        Path(protocol["repair_protocol_path"]),
        Path(protocol["repair_summary_path"]),
        Path(protocol["teacher_manifest_path"]),
        Path(protocol["probe_student_model_path"]),
        Path(protocol["protected_original_student_model_path"]),
        *(
            (Path(protocol["runner_path"]),)
            if "runner_path" in protocol
            else ()
        ),
    )
    before = {str(path): sha256_file(path) for path in protected}
    student = RecurrentPPO.load(Path(protocol["probe_student_model_path"]), device="cpu")
    teacher = M4RLearnerDistributionTeacher(teacher_manifest)
    collection = collect_rescue_batch(
        student,
        teacher,
        seeds=seeds,
        stage=get_curriculum_stage("hurdle_single"),
        output_dir=output_dir,
        rescue_config=teacher_manifest.trigger_config,
        rescue_profile_name="m4r_pre_action_portfolio_shield_v1",
    )
    collection_path = output_dir / "collection_summary.json"
    collection_path.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    replay_rows = [
        replay_accepted_branch(episode)
        for episode in collection["episodes"]
        if bool(episode["branch_accepted"])
    ]
    failed_file_audit = audit_failed_branch_absence(
        output_dir,
        list(collection["episodes"]),
    )
    after = {str(path): sha256_file(path) for path in protected}
    gate = build_gate(
        collection,
        replay_rows,
        failed_file_audit,
        requirements=protocol["gate"],
        source_files_unchanged=before == after,
    )
    branch_manifest = {
        "version": "m4r_pre_action_training_branch_manifest_v1",
        "frozen": True,
        "purpose": "training_demonstration_and_pre_action_intercept_only",
        "result_semantics": protocol["result_semantics"],
        "stage": "hurdle_single",
        "split": "train",
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
    branch_manifest_path = output_dir / "accepted_branch_manifest.json"
    branch_manifest_path.write_text(
        json.dumps(branch_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "method": "m4r_pre_action_teacher_shield_continuous_validation",
        "run_name": output_dir.name,
        "purpose": protocol["purpose"],
        "result_semantics": protocol["result_semantics"],
        "stage": "hurdle_single",
        "split": "train",
        "protocol_path": str(protocol["source_path"]),
        "protocol_sha256": protocol["sha256"],
        "seed_manifest": seed_manifest.as_dict(),
        "teacher_manifest": teacher_manifest.as_dict(),
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_files_unchanged": before == after,
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
        "replay_rows": replay_rows,
        "failed_branch_file_audit": failed_file_audit,
        "accepted_branch_manifest_path": str(branch_manifest_path.resolve()),
        "accepted_branch_manifest_sha256": sha256_file(branch_manifest_path),
        "continuous_gate": gate,
        "eligible_for_training_demonstration_use": bool(
            gate["eligible_for_training_demonstration_use"]
        ),
        "eligible_as_off_distribution_physical_recovery_evidence": False,
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
        description="M4R前動作屏障を十一訓練種で連続収集し厳密再生する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """M4R連続収集門を実行し最小結果を標準出力へ返す。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name)
    print(
        json.dumps(
            {
                "run_name": result["run_name"],
                "collection_success_count": result["collection"][
                    "collection_success_count"
                ],
                "hard_fall_count": result["collection"]["hard_fall_count"],
                "accepted_branch_count": result["collection"][
                    "accepted_branch_count"
                ],
                "continuous_gate": result["continuous_gate"],
                "accepted_branch_manifest_path": result[
                    "accepted_branch_manifest_path"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
