"""M2.3.1の成功救援示範を重み更新なしで再生し監査する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.student_prefix_rescue_env import (
    HURDLE_DEFORMATION_PHASE,
    POST_CLEARANCE_RECOVERY_PHASE,
    POST_RECOVERY_STALL_PHASE,
    PRE_HURDLE_PHASE,
    RESCUE_PHASES,
    classify_rescue_phase,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_MANIFEST = (
    PROJECT_ROOT / "config" / "m2_3_1_success_demo_manifest_v1.json"
)
DEFAULT_STUDENT_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "height1_recurrent_dagger_student"
    / "height1_recurrent_dagger_seed7_v1"
    / "best_model.zip"
)
AUDIT_ROOT = PROJECT_ROOT / "runs" / "rescue_demo_audit"
EXPECTED_SEEDS = {100004, 100001, 100000, 100008}
KEY_TEACHER_PHASES = (
    PRE_HURDLE_PHASE,
    HURDLE_DEFORMATION_PHASE,
    POST_CLEARANCE_RECOVERY_PHASE,
)
PHASE_CODES = {
    PRE_HURDLE_PHASE: 0,
    HURDLE_DEFORMATION_PHASE: 1,
    POST_CLEARANCE_RECOVERY_PHASE: 2,
    POST_RECOVERY_STALL_PHASE: 3,
}


@dataclass(frozen=True)
class RescueDemoCandidate:
    """一つの凍結成功救援分岐と出典情報を保持する。"""

    run_name: str
    profile: str
    seed: int
    start_runway_voxels: int
    course_id: str
    expected_steps: int
    expected_teacher_steps: int
    summary_path: Path
    summary_sha256: str
    branch_path: Path
    branch_sha256: str
    sidecar_path: Path
    sidecar_sha256: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存できる絶対パス付き辞書を返す。"""
        data = asdict(self)
        for name in ("summary_path", "branch_path", "sidecar_path"):
            data[name] = str(data[name])
        return data


@dataclass(frozen=True)
class RescueDemoManifest:
    """M2.3.1で許可された成功示範だけを保持する。"""

    version: str
    stage: str
    split: str
    purpose: str
    student_model_sha256: str
    excluded_sources: tuple[str, ...]
    candidates: tuple[RescueDemoCandidate, ...]
    source_path: Path
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """監査要約へ保存できる辞書を返す。"""
        return {
            "version": self.version,
            "stage": self.stage,
            "split": self.split,
            "purpose": self.purpose,
            "student_model_sha256": self.student_model_sha256,
            "excluded_sources": list(self.excluded_sources),
            "source_path": str(self.source_path),
            "sha256": self.sha256,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def sha256_file(path: Path) -> str:
    """指定ファイルのSHA-256を小文字で返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_file(relative_path: str) -> Path:
    """目録内の相対パスをプロジェクト配下の絶対パスへ変換する。"""
    resolved = (PROJECT_ROOT / relative_path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("示範ファイルはプロジェクト配下になければならない。")
    return resolved


def load_rescue_demo_manifest(
    path: Path = DEFAULT_DEMO_MANIFEST,
) -> RescueDemoManifest:
    """凍結示範目録を読み込み、範囲と全出典ハッシュを検査する。"""
    resolved_path = path.resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("成功救援示範目録は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M2.3.1は単一低壁の訓練区分だけを使用できる。")
    candidates: list[RescueDemoCandidate] = []
    for item in payload.get("candidates", []):
        candidate = RescueDemoCandidate(
            run_name=str(item["run_name"]),
            profile=str(item["profile"]),
            seed=int(item["seed"]),
            start_runway_voxels=int(item["start_runway_voxels"]),
            course_id=str(item["course_id"]),
            expected_steps=int(item["expected_steps"]),
            expected_teacher_steps=int(item["expected_teacher_steps"]),
            summary_path=_resolve_project_file(str(item["summary_path"])),
            summary_sha256=str(item["summary_sha256"]),
            branch_path=_resolve_project_file(str(item["branch_path"])),
            branch_sha256=str(item["branch_sha256"]),
            sidecar_path=_resolve_project_file(str(item["sidecar_path"])),
            sidecar_sha256=str(item["sidecar_sha256"]),
        )
        file_hashes = (
            (candidate.summary_path, candidate.summary_sha256),
            (candidate.branch_path, candidate.branch_sha256),
            (candidate.sidecar_path, candidate.sidecar_sha256),
        )
        for source_path, expected_hash in file_hashes:
            if sha256_file(source_path) != expected_hash:
                raise ValueError(f"成功救援出典のハッシュが一致しない: {source_path}")
        candidates.append(candidate)
    if len(candidates) != 4:
        raise ValueError("M2.3.1の候補示範は4本でなければならない。")
    if {candidate.seed for candidate in candidates} != EXPECTED_SEEDS:
        raise ValueError("M2.3.1の候補乱数シード集合が凍結値と一致しない。")
    if any(not candidate.run_name.endswith("_v2") for candidate in candidates):
        raise ValueError("無効なv1救援結果を示範へ含めることはできない。")
    excluded_sources = tuple(str(item) for item in payload["excluded_sources"])
    required_exclusions = {
        "m2_1_preventive_near6_v1",
        "all_failed_branches",
        "m2_3_prefix_rescue_teacher_seed7_v1",
    }
    if not required_exclusions.issubset(set(excluded_sources)):
        raise ValueError("失敗または無効な出典の除外指定が不足している。")
    return RescueDemoManifest(
        version=str(payload["version"]),
        stage=str(payload["stage"]),
        split=str(payload["split"]),
        purpose=str(payload["purpose"]),
        student_model_sha256=str(payload["student_model_sha256"]),
        excluded_sources=excluded_sources,
        candidates=tuple(candidates),
        source_path=resolved_path,
        sha256=sha256_file(resolved_path),
    )


def find_true_segments(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """真値が連続する区間を開始位置と終了位置の組で返す。"""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        if not value and start is not None:
            segments.append((start, index))
            start = None
    if start is not None:
        segments.append((start, len(values)))
    return tuple(segments)


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    """空配列にも対応して二配列の最大絶対差を返す。"""
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def load_and_validate_branch_arrays(
    candidate: RescueDemoCandidate,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """示範配列の形状、有限性、制御器対応と連続区間を検査する。"""
    required = {
        "observations",
        "student_actions",
        "teacher_actions",
        "executed_actions",
        "teacher_mask",
        "rescue_ids",
        "teacher_stages",
    }
    with np.load(candidate.branch_path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"示範配列項目が凍結契約と一致しない: {candidate.seed}")
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    steps = candidate.expected_steps
    expected_shapes = {
        "observations": (steps, 95),
        "student_actions": (steps, 6),
        "teacher_actions": (steps, 6),
        "executed_actions": (steps, 6),
        "teacher_mask": (steps,),
        "rescue_ids": (steps,),
        "teacher_stages": (steps,),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"示範配列の形状が不正である: {candidate.seed}, {name}")
    for name in (
        "observations",
        "student_actions",
        "teacher_actions",
        "executed_actions",
    ):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"示範配列に非有限値がある: {candidate.seed}, {name}")
    teacher_mask = arrays["teacher_mask"].astype(bool)
    if int(np.count_nonzero(teacher_mask)) != candidate.expected_teacher_steps:
        raise ValueError(f"教師制御歩数が出典と一致しない: {candidate.seed}")
    if np.any(arrays["rescue_ids"][teacher_mask] <= 0):
        raise ValueError(f"教師区間の救援番号が不正である: {candidate.seed}")
    if np.any(arrays["rescue_ids"][~teacher_mask] != 0):
        raise ValueError(f"学生区間に救援番号が混入している: {candidate.seed}")
    teacher_action_difference = _maximum_difference(
        arrays["executed_actions"][teacher_mask],
        arrays["teacher_actions"][teacher_mask],
    )
    student_action_difference = _maximum_difference(
        arrays["executed_actions"][~teacher_mask],
        arrays["student_actions"][~teacher_mask],
    )
    if teacher_action_difference != 0.0 or student_action_difference != 0.0:
        raise ValueError(f"実行動作と制御器動作が一致しない: {candidate.seed}")
    segments = find_true_segments(teacher_mask)
    if not segments:
        raise ValueError(f"教師制御区間が存在しない: {candidate.seed}")
    return arrays, {
        "observation_shape": [95],
        "action_shape": [6],
        "teacher_control_steps": int(np.count_nonzero(teacher_mask)),
        "student_control_steps": int(np.count_nonzero(~teacher_mask)),
        "teacher_fraction": float(np.mean(teacher_mask)),
        "teacher_segments": [
            {"start": start, "end_exclusive": end, "steps": end - start}
            for start, end in segments
        ],
        "maximum_teacher_action_difference": teacher_action_difference,
        "maximum_student_action_difference": student_action_difference,
    }


def validate_source_metadata(candidate: RescueDemoCandidate) -> dict[str, object]:
    """集計要約と側車JSONが同じ合格分岐を指すか検査する。"""
    summary = json.loads(candidate.summary_path.read_text(encoding="utf-8"))
    sidecar = json.loads(candidate.sidecar_path.read_text(encoding="utf-8"))
    if summary.get("stage") != "hurdle_single" or summary.get("split") != "train":
        raise ValueError(f"出典区分が訓練単一低壁ではない: {candidate.seed}")
    if summary.get("rescue_profile") != candidate.profile:
        raise ValueError(f"救援設定名が目録と一致しない: {candidate.seed}")
    matching = [
        episode
        for episode in summary["episodes"]
        if int(episode["seed"]) == candidate.seed
    ]
    if len(matching) != 1:
        raise ValueError(f"出典要約の候補回数が不正である: {candidate.seed}")
    episode = matching[0]
    if not bool(episode["course_complete"]) or bool(episode["hard_fall"]):
        raise ValueError(f"候補分岐が成功無転倒ではない: {candidate.seed}")
    if not bool(episode["branch_accepted"]):
        raise ValueError(f"候補分岐が保存対象として合格していない: {candidate.seed}")
    if int(episode["steps"]) != candidate.expected_steps:
        raise ValueError(f"候補分岐歩数が目録と一致しない: {candidate.seed}")
    if int(episode["teacher_control_steps"]) != candidate.expected_teacher_steps:
        raise ValueError(f"候補教師歩数が目録と一致しない: {candidate.seed}")
    if str(episode["course_id"]) != candidate.course_id:
        raise ValueError(f"候補コース識別子が目録と一致しない: {candidate.seed}")
    if not bool(sidecar["accepted"]):
        raise ValueError(f"側車JSONが不合格を示している: {candidate.seed}")
    if not bool(sidecar["course_complete"]) or bool(sidecar["hard_fall"]):
        raise ValueError(f"側車JSONの成功条件が不正である: {candidate.seed}")
    if int(sidecar["steps"]) != candidate.expected_steps:
        raise ValueError(f"側車JSONの歩数が目録と一致しない: {candidate.seed}")
    if int(sidecar["teacher_steps"]) != candidate.expected_teacher_steps:
        raise ValueError(f"側車JSONの教師歩数が目録と一致しない: {candidate.seed}")
    if sidecar["metadata"].get("split") != "train":
        raise ValueError(f"側車JSONが訓練区分ではない: {candidate.seed}")
    start_events = [event for event in episode["events"] if event["event"] == "start"]
    return {
        "summary_success_confirmed": True,
        "sidecar_success_confirmed": True,
        "recorded_rescue_count": int(episode["rescue_count"]),
        "recorded_start_events": start_events,
    }


def _phase_counts(labels: np.ndarray, mask: np.ndarray | None = None) -> dict[str, int]:
    """段階コード列から各段階の歩数を数える。"""
    selected = labels if mask is None else labels[np.asarray(mask, dtype=bool)]
    return {
        phase: int(np.count_nonzero(selected == code))
        for phase, code in PHASE_CODES.items()
    }


def _teacher_phase_segments(
    phase_codes: np.ndarray,
    teacher_mask: np.ndarray,
) -> list[dict[str, object]]:
    """教師制御内で同じ段階が連続する区間を返す。"""
    segments: list[dict[str, object]] = []
    active_start: int | None = None
    active_code: int | None = None
    inverse_codes = {code: phase for phase, code in PHASE_CODES.items()}
    for index in range(len(phase_codes) + 1):
        is_teacher = index < len(phase_codes) and bool(teacher_mask[index])
        code = int(phase_codes[index]) if is_teacher else None
        if active_start is not None and (not is_teacher or code != active_code):
            segments.append(
                {
                    "phase": inverse_codes[int(active_code)],
                    "start": active_start,
                    "end_exclusive": index,
                    "steps": index - active_start,
                }
            )
            active_start = None
            active_code = None
        if is_teacher and active_start is None:
            active_start = index
            active_code = code
    return segments


def replay_candidate(
    candidate: RescueDemoCandidate,
    *,
    stage: str,
    split: str,
    phase_index_dir: Path,
) -> dict[str, object]:
    """保存済み実行動作を初期状態から再生し全観測と終局を照合する。"""
    source_metadata = validate_source_metadata(candidate)
    arrays, array_audit = load_and_validate_branch_arrays(candidate)
    teacher_mask = arrays["teacher_mask"].astype(bool)
    first_teacher_step = find_true_segments(teacher_mask)[0][0]
    start_events = source_metadata["recorded_start_events"]
    if not start_events or int(start_events[0]["step"]) != first_teacher_step:
        raise ValueError(f"教師開始位置が出典イベントと一致しない: {candidate.seed}")
    course = sample_curriculum_course(candidate.seed, stage, split)
    if course.course_id != candidate.course_id:
        raise ValueError(f"再生成コースが出典と一致しない: {candidate.seed}")
    if course.obstacles[0].start_x != candidate.start_runway_voxels:
        raise ValueError(f"再生成開始位置が出典と一致しない: {candidate.seed}")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    phase_codes = np.empty(candidate.expected_steps, dtype=np.int8)
    maximum_observation_difference = 0.0
    terminated = False
    truncated = False
    try:
        observation, info = environment.reset(seed=candidate.seed)
        for index, action in enumerate(arrays["executed_actions"]):
            difference = _maximum_difference(
                np.asarray(observation, dtype=np.float32),
                arrays["observations"][index],
            )
            maximum_observation_difference = max(
                maximum_observation_difference,
                difference,
            )
            phase = classify_rescue_phase(environment, info)
            phase_codes[index] = PHASE_CODES[phase]
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(action, dtype=np.float32)
            )
            if (terminated or truncated) and index != candidate.expected_steps - 1:
                raise RuntimeError(
                    f"示範再生が保存終端より早く終了した: {candidate.seed}, {index + 1}"
                )
        if not (terminated or truncated):
            raise RuntimeError(f"示範再生が保存終端で終了しなかった: {candidate.seed}")
        replay_success = bool(
            info["course_complete"]
            and not info["hard_fall"]
            and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
        )
        result = {
            "run_name": candidate.run_name,
            "profile": candidate.profile,
            "seed": candidate.seed,
            "course_id": candidate.course_id,
            "start_runway_voxels": candidate.start_runway_voxels,
            "steps": candidate.expected_steps,
            "replay_success": replay_success,
            "course_complete": bool(info["course_complete"]),
            "hard_fall": bool(info["hard_fall"]),
            "failure_reason": str(info["failure_reason"]),
            "raw_clearances": int(info["raw_clearances"]),
            "recovered_obstacles": int(info["recovered_obstacles"]),
            "maximum_observation_difference": maximum_observation_difference,
            "all_step_observations_exact": maximum_observation_difference == 0.0,
            "all_values_finite": True,
            "source_metadata": source_metadata,
            "array_audit": array_audit,
            "all_phase_step_counts": _phase_counts(phase_codes),
            "teacher_phase_step_counts": _phase_counts(phase_codes, teacher_mask),
            "teacher_phase_segments": _teacher_phase_segments(
                phase_codes,
                teacher_mask,
            ),
        }
    finally:
        environment.close()
    phase_index_dir.mkdir(parents=True, exist_ok=True)
    phase_index_path = phase_index_dir / f"seed_{candidate.seed}_phase_index.npz"
    segments = find_true_segments(teacher_mask)
    np.savez_compressed(
        phase_index_path,
        phase_codes=phase_codes,
        teacher_indices=np.flatnonzero(teacher_mask).astype(np.int32),
        teacher_segment_starts=np.asarray([start for start, _ in segments], dtype=np.int32),
        teacher_segment_ends=np.asarray([end for _, end in segments], dtype=np.int32),
    )
    result["phase_index_path"] = str(phase_index_path.resolve())
    result["phase_index_sha256"] = sha256_file(phase_index_path)
    return result


def build_m2_3_1_gate(rows: list[Mapping[str, object]]) -> dict[str, object]:
    """再生成功、安全性、次元、段階網羅からM2.3.1門を判定する。"""
    teacher_phase_counts = {phase: 0 for phase in RESCUE_PHASES}
    for row in rows:
        for phase, count in row["teacher_phase_step_counts"].items():
            teacher_phase_counts[phase] += int(count)
    candidate_requirement = len(rows) == 4
    replay_requirement = bool(
        candidate_requirement and all(bool(row["replay_success"]) for row in rows)
    )
    hard_fall_requirement = all(not bool(row["hard_fall"]) for row in rows)
    exact_observation_requirement = all(
        bool(row["all_step_observations_exact"]) for row in rows
    )
    dimension_requirement = all(
        row["array_audit"]["observation_shape"] == [95]
        and row["array_audit"]["action_shape"] == [6]
        for row in rows
    )
    phase_coverage_requirement = all(
        teacher_phase_counts[phase] > 0 for phase in KEY_TEACHER_PHASES
    )
    gate_passed = bool(
        candidate_requirement
        and replay_requirement
        and hard_fall_requirement
        and exact_observation_requirement
        and dimension_requirement
        and phase_coverage_requirement
    )
    return {
        "gate_name": "m2_3_1_success_demo_audit_gate_v1",
        "required_candidate_count": 4,
        "required_replay_success_count": 4,
        "maximum_hard_fall_count": 0,
        "required_observation_shape": [95],
        "required_action_shape": [6],
        "required_teacher_phases": list(KEY_TEACHER_PHASES),
        "candidate_requirement_passed": candidate_requirement,
        "replay_requirement_passed": replay_requirement,
        "hard_fall_requirement_passed": hard_fall_requirement,
        "exact_observation_requirement_passed": exact_observation_requirement,
        "dimension_requirement_passed": dimension_requirement,
        "phase_coverage_requirement_passed": phase_coverage_requirement,
        "teacher_phase_step_counts": teacher_phase_counts,
        "gate_passed": gate_passed,
        "eligible_for_m2_3_2": gate_passed,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """重み更新を持たないM2.3.1監査引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="凍結済み成功救援示範を再生し段階網羅を監査する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_DEMO_MANIFEST))
    return parser


def main() -> None:
    """四つの成功分岐を再生し監査要約と段階索引を保存する。"""
    args = build_argument_parser().parse_args()
    manifest = load_rescue_demo_manifest(Path(args.manifest))
    student_hash_before = sha256_file(DEFAULT_STUDENT_MODEL)
    if student_hash_before != manifest.student_model_sha256:
        raise ValueError("凍結学生のハッシュが示範目録と一致しない。")
    source_hashes_before = {
        str(path): sha256_file(path)
        for candidate in manifest.candidates
        for path in (
            candidate.summary_path,
            candidate.branch_path,
            candidate.sidecar_path,
        )
    }
    output_dir = AUDIT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        replay_candidate(
            candidate,
            stage=manifest.stage,
            split=manifest.split,
            phase_index_dir=output_dir / "phase_indices",
        )
        for candidate in manifest.candidates
    ]
    source_hashes_after = {
        path: sha256_file(Path(path)) for path in source_hashes_before
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("M2.3.1中に出典示範ファイルが変更された。")
    student_hash_after = sha256_file(DEFAULT_STUDENT_MODEL)
    if student_hash_after != student_hash_before:
        raise RuntimeError("M2.3.1中に凍結学生ファイルが変更された。")
    gate = build_m2_3_1_gate(rows)
    summary = {
        "method": "m2_3_1_success_rescue_demonstration_audit",
        "stage": manifest.stage,
        "split": manifest.split,
        "purpose": manifest.purpose,
        "run_name": args.run_name,
        "manifest": manifest.as_dict(),
        "manifest_sha256": manifest.sha256,
        "candidate_count": len(rows),
        "replay_success_count": sum(bool(row["replay_success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "source_files_unchanged": source_hashes_after == source_hashes_before,
        "student_model_sha256_before": student_hash_before,
        "student_model_sha256_after": student_hash_after,
        "student_model_unchanged": student_hash_after == student_hash_before,
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "training_steps": 0,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "m2_3_failed_checkpoint_loaded": False,
        "observation_dimension": 95,
        "action_dimension": 6,
        "phase_code_mapping": PHASE_CODES,
        "candidates": rows,
        "m2_3_1_gate": gate,
        "eligible_for_m2_3_2": bool(gate["eligible_for_m2_3_2"]),
        "eligible_for_student_initialization": False,
        "eligible_for_final_student_test": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
