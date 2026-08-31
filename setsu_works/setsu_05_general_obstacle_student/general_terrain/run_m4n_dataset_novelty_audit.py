"""M4R前動作屏障分岐とM2.4救援分岐の新規性を零更新で監査する。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from general_terrain.audit_rescue_demonstrations import sha256_file
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.run_m4_probe_rescue_preflight import _load_branch
from general_terrain.student_prefix_rescue_env import (
    HURDLE_DEFORMATION_PHASE,
    PRE_HURDLE_PHASE,
    classify_rescue_phase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "m4n_dataset_novelty_protocol_v1.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "m4n_dataset_novelty"
THREE_COURSES = (
    "pre_hurdle_safety_intercept",
    "hurdle_contact_deformation",
    "landing_recovery_stall",
)


def _resolve_project_path(value: str) -> Path:
    """規約内の相対パスをプロジェクト配下だけへ解決する。"""
    resolved = (PROJECT_ROOT / value).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M4N出典はプロジェクト配下でなければならない。")
    return resolved


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、出典ハッシュ、零更新、教師隔離を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M4N規約は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M4Nは単一低壁の訓練区分だけを使用できる。")
    zero_fields = (
        "student_update_steps",
        "teacher_update_steps",
        "ppo_training_steps",
        "validation_episodes",
        "holdout_episodes",
        "teacher_interventions_in_final_student_test",
    )
    if any(int(payload[field]) != 0 for field in zero_fields):
        raise ValueError("M4Nでは重み更新、評価アクセス、最終教師介入を禁止する。")
    for field in (
        "validation_teacher_enabled",
        "holdout_teacher_enabled",
        "final_student_test_teacher_enabled",
    ):
        if bool(payload.get(field, True)):
            raise ValueError("検証、留保、最終学生試験では教師を停止しなければならない。")
    fields = (
        "seed_manifest_path",
        "m2_summary_path",
        "m2_branch_manifest_path",
        "m4r_summary_path",
        "m4r_branch_manifest_path",
        "one_step_failure_summary_path",
        "protected_original_student_model_path",
        "probe_student_model_path",
        "runner_path",
    )
    resolved: dict[str, Path] = {}
    for field in fields:
        file_path = _resolve_project_path(str(payload[field]))
        expected = str(payload[f"{field.removesuffix('_path')}_sha256"]).lower()
        if sha256_file(file_path) != expected:
            raise ValueError(f"M4N出典ハッシュが一致しない: {file_path}")
        resolved[field] = file_path
    m2_summary = json.loads(resolved["m2_summary_path"].read_text(encoding="utf-8"))
    if not bool(m2_summary["m2_4_gate"]["gate_passed"]):
        raise ValueError("M2.4成功分岐門が未通過である。")
    m4r_summary = json.loads(
        resolved["m4r_summary_path"].read_text(encoding="utf-8")
    )
    if not bool(m4r_summary["continuous_gate"]["gate_passed"]):
        raise ValueError("M4R連続屏障門が未通過である。")
    if bool(
        m4r_summary["continuous_gate"].get(
            "eligible_as_off_distribution_physical_recovery_evidence",
            True,
        )
    ):
        raise ValueError("M4R出典は離軌物理回復証拠として明示的に拒否されなければならない。")
    return {
        **payload,
        **resolved,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def _course_name(phase: str) -> str:
    """四監査相位をM4Rの三分離課程へ写像する。"""
    if phase == PRE_HURDLE_PHASE:
        return "pre_hurdle_safety_intercept"
    if phase == HURDLE_DEFORMATION_PHASE:
        return "hurdle_contact_deformation"
    return "landing_recovery_stall"


def _replay_and_label(
    row: Mapping[str, object],
    *,
    stride: int,
) -> dict[str, object]:
    """成功分岐を厳密再生し決定論的標本と三課程ラベルを返す。"""
    branch_path = Path(str(row["branch_path"]))
    sidecar_path = Path(str(row["sidecar_path"]))
    if sha256_file(branch_path) != str(row["branch_sha256"]).lower():
        raise ValueError(f"M4N分岐ハッシュが一致しない: {branch_path}")
    if sha256_file(sidecar_path) != str(row["sidecar_sha256"]).lower():
        raise ValueError(f"M4N側車ハッシュが一致しない: {sidecar_path}")
    arrays = _load_branch(branch_path)
    seed = int(row["seed"])
    position = int(row["start_runway_voxels"])
    course = sample_curriculum_course(seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    phases: list[str] = []
    maximum_difference = 0.0
    try:
        observation, info = environment.reset(seed=seed)
        for index, action in enumerate(arrays["executed_actions"]):
            maximum_difference = max(
                maximum_difference,
                float(
                    np.max(
                        np.abs(
                            np.asarray(observation, dtype=np.float64)
                            - arrays["observations"][index].astype(np.float64)
                        )
                    )
                ),
            )
            phases.append(_course_name(classify_rescue_phase(environment, info)))
            observation, _, terminated, truncated, info = environment.step(action)
            if (terminated or truncated) and index != len(arrays["executed_actions"]) - 1:
                raise RuntimeError(f"M4N分岐が保存終端より早く終了した: {seed}")
    finally:
        environment.close()
    if not bool(info["course_complete"]) or bool(info["hard_fall"]):
        raise RuntimeError(f"M4N出典分岐が厳格成功終端ではない: {seed}")
    indices = np.arange(0, len(phases), stride, dtype=np.int64)
    if indices[-1] != len(phases) - 1:
        indices = np.concatenate((indices, np.asarray([len(phases) - 1])))
    teacher_mask = arrays["teacher_mask"].astype(bool)
    action_disagreement = np.mean(
        np.abs(
            arrays["student_actions"].astype(np.float64)
            - arrays["teacher_actions"].astype(np.float64)
        ),
        axis=1,
    )
    return {
        "seed": seed,
        "position": position,
        "steps": len(phases),
        "student_executed_steps": int(np.count_nonzero(~teacher_mask)),
        "teacher_executed_steps": int(np.count_nonzero(teacher_mask)),
        "maximum_replay_observation_difference": maximum_difference,
        "phase_counts": dict(Counter(phases)),
        "sample_indices": indices,
        "sample_observations": arrays["observations"][indices].astype(np.float32),
        "sample_phases": np.asarray(phases, dtype="U40")[indices],
        "teacher_step_action_disagreement": action_disagreement[teacher_mask],
        "first_step_action_disagreement": float(action_disagreement[0]),
        "first_student_action_maximum_absolute": float(
            np.max(np.abs(arrays["student_actions"][0].astype(np.float64)))
        ),
        "first_teacher_action_maximum_absolute": float(
            np.max(np.abs(arrays["teacher_actions"][0].astype(np.float64)))
        ),
        "branch_path": str(branch_path.resolve()),
        "branch_sha256": sha256_file(branch_path),
    }


def _nearest_normalized_rms(
    queries: np.ndarray,
    references: np.ndarray,
    scale: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """標準化観測の最近傍RMS距離を固定チャンクで計算する。"""
    normalized_references = references.astype(np.float64) / scale
    reference_norms = np.sum(normalized_references**2, axis=1)
    output = np.empty(len(queries), dtype=np.float64)
    dimension = float(queries.shape[1])
    for start in range(0, len(queries), chunk_size):
        stop = min(len(queries), start + chunk_size)
        normalized_queries = queries[start:stop].astype(np.float64) / scale
        query_norms = np.sum(normalized_queries**2, axis=1)[:, None]
        squared = (
            query_norms
            + reference_norms[None, :]
            - 2.0 * normalized_queries @ normalized_references.T
        )
        output[start:stop] = np.sqrt(
            np.maximum(np.min(squared, axis=1), 0.0) / dimension
        )
    return output


def _calibration_threshold(
    old_rows: list[dict[str, object]],
    scale: np.ndarray,
    *,
    quantile: float,
    chunk_size: int,
) -> tuple[float, np.ndarray]:
    """旧分岐を一条ずつ除外した最近傍距離から新規性閾値を校正する。"""
    distances = []
    for index, row in enumerate(old_rows):
        references = np.concatenate(
            [
                other["sample_observations"]
                for other_index, other in enumerate(old_rows)
                if other_index != index
            ],
            axis=0,
        )
        distances.append(
            _nearest_normalized_rms(
                row["sample_observations"],
                references,
                scale,
                chunk_size=chunk_size,
            )
        )
    merged = np.concatenate(distances)
    return float(np.quantile(merged, quantile)), merged


def _distance_summary(values: np.ndarray, threshold: float) -> dict[str, object]:
    """最近傍距離列を分位点と新規標本率へ要約する。"""
    if len(values) == 0:
        return {
            "sample_count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "p95": None,
            "maximum": None,
            "novel_sample_count": 0,
            "novel_sample_ratio": 0.0,
            "exact_duplicate_sample_count": 0,
        }
    novel = values > threshold
    return {
        "sample_count": len(values),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
        "novel_sample_count": int(np.count_nonzero(novel)),
        "novel_sample_ratio": float(np.mean(novel)),
        "exact_duplicate_sample_count": int(np.count_nonzero(values <= 1e-12)),
    }


def build_gate(
    audit: Mapping[str, object],
    *,
    requirements: Mapping[str, object],
    source_files_unchanged: bool,
) -> dict[str, object]:
    """新規性、安全性、学習器実行状態、出典不変からM4A入口を判定する。"""
    required_positions = {int(value) for value in requirements["required_new_positions"]}
    position_rows = {
        int(position): row for position, row in audit["position_novelty"].items()
    }
    phase_counts = audit["new_phase_sample_counts"]
    checks = {
        "source_replay_requirement_passed": int(
            audit["maximum_source_replay_observation_difference"] == 0.0
        )
        == 1,
        "safe_branch_requirement_passed": int(audit["new_branch_count"])
        >= int(requirements["minimum_safe_new_branch_count"]),
        "overall_novelty_requirement_passed": float(
            audit["overall_novelty"]["novel_sample_ratio"]
        )
        >= float(requirements["minimum_overall_novel_sample_ratio"]),
        "new_position_requirement_passed": required_positions.issubset(position_rows)
        and all(
            float(position_rows[position]["novel_sample_ratio"])
            >= float(requirements["minimum_required_position_novel_sample_ratio"])
            for position in required_positions
        ),
        "three_course_coverage_requirement_passed": all(
            int(phase_counts.get(course, 0))
            >= int(requirements["minimum_samples_per_recovery_course"])
            for course in THREE_COURSES
        ),
        "action_disagreement_requirement_passed": float(
            audit["teacher_step_action_disagreement"]["mean"]
        )
        >= float(requirements["minimum_mean_teacher_student_action_disagreement"]),
        "learner_executed_state_requirement_passed": int(
            audit["new_student_executed_steps"]
        )
        >= int(requirements["minimum_learner_executed_steps_for_m4a"]),
        "source_immutability_requirement_passed": source_files_unchanged,
    }
    demonstration_checks = {
        key: value
        for key, value in checks.items()
        if key != "learner_executed_state_requirement_passed"
    }
    return {
        "gate_name": "m4n_dataset_novelty_and_interactive_aggregation_gate_v1",
        **requirements,
        **checks,
        "teacher_demonstration_archive_gate_passed": all(
            demonstration_checks.values()
        ),
        "gate_passed": all(checks.values()),
        "eligible_for_m4a": all(checks.values()),
        "student_weight_update_executed": False,
    }


def run(protocol: dict[str, object], output_dir: Path) -> dict[str, object]:
    """旧新分岐の厳密再生、距離校正、新規性門を零更新で実行する。"""
    output_dir.mkdir(parents=True, exist_ok=False)
    protected = (
        Path(protocol["source_path"]),
        Path(protocol["seed_manifest_path"]),
        Path(protocol["m2_summary_path"]),
        Path(protocol["m2_branch_manifest_path"]),
        Path(protocol["m4r_summary_path"]),
        Path(protocol["m4r_branch_manifest_path"]),
        Path(protocol["one_step_failure_summary_path"]),
        Path(protocol["protected_original_student_model_path"]),
        Path(protocol["probe_student_model_path"]),
        Path(protocol["runner_path"]),
    )
    before = {str(path): sha256_file(path) for path in protected}
    m2_manifest = json.loads(
        Path(protocol["m2_branch_manifest_path"]).read_text(encoding="utf-8")
    )
    m4r_manifest = json.loads(
        Path(protocol["m4r_branch_manifest_path"]).read_text(encoding="utf-8")
    )
    stride = int(protocol["sampling_stride"])
    old_rows = [
        _replay_and_label(row, stride=stride) for row in m2_manifest["branches"]
    ]
    new_rows = [
        _replay_and_label(row, stride=stride) for row in m4r_manifest["branches"]
    ]
    old_observations = np.concatenate(
        [row["sample_observations"] for row in old_rows], axis=0
    )
    lower = np.quantile(old_observations.astype(np.float64), 0.05, axis=0)
    upper = np.quantile(old_observations.astype(np.float64), 0.95, axis=0)
    scale = np.maximum(
        upper - lower,
        float(protocol["minimum_feature_scale"]),
    )
    threshold, calibration_distances = _calibration_threshold(
        old_rows,
        scale,
        quantile=float(protocol["calibration_quantile"]),
        chunk_size=int(protocol["distance_chunk_size"]),
    )
    new_observations = np.concatenate(
        [row["sample_observations"] for row in new_rows], axis=0
    )
    new_distances = _nearest_normalized_rms(
        new_observations,
        old_observations,
        scale,
        chunk_size=int(protocol["distance_chunk_size"]),
    )
    position_novelty = {}
    phase_novelty = {}
    offset = 0
    for row in new_rows:
        count = len(row["sample_observations"])
        distances = new_distances[offset : offset + count]
        position_novelty[str(row["position"])] = _distance_summary(
            distances,
            threshold,
        )
        offset += count
    new_phases = np.concatenate([row["sample_phases"] for row in new_rows])
    for course in THREE_COURSES:
        mask = new_phases == course
        phase_novelty[course] = _distance_summary(new_distances[mask], threshold)
    action_values = np.concatenate(
        [row["teacher_step_action_disagreement"] for row in new_rows]
    )
    first_step_rows = [
        {
            "seed": row["seed"],
            "position": row["position"],
            "student_teacher_mean_absolute_difference": row[
                "first_step_action_disagreement"
            ],
            "student_action_maximum_absolute": row[
                "first_student_action_maximum_absolute"
            ],
            "teacher_action_maximum_absolute": row[
                "first_teacher_action_maximum_absolute"
            ],
        }
        for row in new_rows
    ]
    one_step_summary = json.loads(
        Path(protocol["one_step_failure_summary_path"]).read_text(encoding="utf-8")
    )
    audit = {
        "old_branch_count": len(old_rows),
        "new_branch_count": len(new_rows),
        "old_total_steps": sum(int(row["steps"]) for row in old_rows),
        "new_total_steps": sum(int(row["steps"]) for row in new_rows),
        "old_sample_count": len(old_observations),
        "new_sample_count": len(new_observations),
        "old_student_executed_steps": sum(
            int(row["student_executed_steps"]) for row in old_rows
        ),
        "new_student_executed_steps": sum(
            int(row["student_executed_steps"]) for row in new_rows
        ),
        "new_teacher_executed_steps": sum(
            int(row["teacher_executed_steps"]) for row in new_rows
        ),
        "maximum_source_replay_observation_difference": max(
            float(row["maximum_replay_observation_difference"])
            for row in (*old_rows, *new_rows)
        ),
        "normalization": {
            "method": "m2_4_feature_p95_minus_p05_with_floor",
            "minimum_feature_scale": float(protocol["minimum_feature_scale"]),
        },
        "calibration": {
            "method": "m2_4_leave_one_branch_out_nearest_normalized_rms",
            "quantile": float(protocol["calibration_quantile"]),
            "threshold": threshold,
            "sample_count": len(calibration_distances),
            "median": float(np.median(calibration_distances)),
            "p95": float(np.quantile(calibration_distances, 0.95)),
            "maximum": float(np.max(calibration_distances)),
        },
        "overall_novelty": _distance_summary(new_distances, threshold),
        "position_novelty": position_novelty,
        "phase_novelty": phase_novelty,
        "new_phase_sample_counts": dict(Counter(new_phases.tolist())),
        "teacher_step_action_disagreement": {
            "step_count": len(action_values),
            "mean": float(np.mean(action_values)),
            "median": float(np.median(action_values)),
            "p90": float(np.quantile(action_values, 0.90)),
            "maximum": float(np.max(action_values)),
        },
        "first_step_action_rows": first_step_rows,
        "first_step_action_disagreement_mean": float(
            np.mean(
                [row["student_teacher_mean_absolute_difference"] for row in first_step_rows]
            )
        ),
        "one_step_physical_recovery_evidence": {
            "summary_path": str(Path(protocol["one_step_failure_summary_path"])),
            "exact_state_success_count": int(
                one_step_summary["repair_gate"]["exact_state_success_count"]
            ),
            "exact_state_hard_fall_count": int(
                one_step_summary["repair_gate"]["exact_state_hard_fall_count"]
            ),
            "eligible_for_continuous_validation": bool(
                one_step_summary["eligible_for_continuous_validation"]
            ),
        },
    }
    after = {str(path): sha256_file(path) for path in protected}
    gate = build_gate(
        audit,
        requirements=protocol["gate"],
        source_files_unchanged=before == after,
    )
    frozen_manifest = {
        "version": "m4n_dataset_novelty_audit_manifest_v1",
        "frozen": True,
        "stage": "hurdle_single",
        "split": "train",
        "sampling_stride": stride,
        "calibrated_novelty_threshold": threshold,
        "old_branches": [
            {
                key: row[key]
                for key in ("seed", "position", "steps", "branch_path", "branch_sha256")
            }
            for row in old_rows
        ],
        "new_branches": [
            {
                key: row[key]
                for key in ("seed", "position", "steps", "branch_path", "branch_sha256")
            }
            for row in new_rows
        ],
        "gate": gate,
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
    }
    manifest_path = output_dir / "dataset_novelty_manifest.json"
    manifest_path.write_text(
        json.dumps(frozen_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "method": "m4n_zero_update_dataset_novelty_audit",
        "run_name": output_dir.name,
        "stage": "hurdle_single",
        "split": "train",
        "protocol_path": str(protocol["source_path"]),
        "protocol_sha256": protocol["sha256"],
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_files_unchanged": before == after,
        "audit": audit,
        "m4n_gate": gate,
        "dataset_novelty_manifest_path": str(manifest_path.resolve()),
        "dataset_novelty_manifest_sha256": sha256_file(manifest_path),
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "student_update_steps": 0,
        "teacher_update_steps": 0,
        "ppo_training_steps": 0,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "eligible_for_m4a": bool(gate["eligible_for_m4a"]),
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
        description="M4R屏障分岐の状態新規性とM4A入口を零更新で監査する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """M4N新規性監査を実行し門の最小結果を出力する。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name)
    print(
        json.dumps(
            {
                "run_name": result["run_name"],
                "overall_novelty": result["audit"]["overall_novelty"],
                "new_student_executed_steps": result["audit"][
                    "new_student_executed_steps"
                ],
                "m4n_gate": result["m4n_gate"],
                "dataset_novelty_manifest_path": result[
                    "dataset_novelty_manifest_path"
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
