"""M2.3.5aの位相別成功軌跡リセットと旧教師近傍能力を監査する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from general_terrain.audit_rescue_demonstrations import (
    RescueDemoCandidate,
    load_and_validate_branch_arrays,
    load_rescue_demo_manifest,
    sha256_file,
)
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEMO_MANIFEST = (
    PROJECT_ROOT / "config" / "m2_3_1_success_demo_manifest_v1.json"
)
DEFAULT_DEMO_AUDIT = (
    PROJECT_ROOT
    / "runs"
    / "rescue_demo_audit"
    / "m2_3_1_success_demo_audit_v1"
    / "summary.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "phase_reset_curriculum_audit"
CURRICULUM_PHASES = (
    "pre_hurdle",
    "hurdle_deformation",
    "post_clearance_recovery",
    "stable_finish",
)
IMPULSE_MAGNITUDE = 0.01


@dataclass(frozen=True)
class PhaseResetSpec:
    """一つの成功軌跡上にある訓練専用位相リセット点を保持する。"""

    reset_id: str
    phase: str
    seed: int
    profile: str
    course_id: str
    start_runway_voxels: int
    source_step: int
    source_branch_path: str
    source_branch_sha256: str
    source_observation_sha256: str
    impulse_action_dimension: int
    impulse_sign: int

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存できる辞書を返す。"""
        return asdict(self)


def array_sha256(array: np.ndarray) -> str:
    """配列の型、形状、連続バイト列から安定したハッシュを返す。"""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def select_phase_reset_steps(
    teacher_phase_segments: list[dict[str, object]],
    *,
    stable_window_steps: int = 64,
) -> dict[str, int]:
    """三位相区間から開始近傍と終端安定窓の代表歩を選ぶ。"""
    if stable_window_steps < 1:
        raise ValueError("安定終端窓は一歩以上でなければならない。")
    segments = {str(row["phase"]): row for row in teacher_phase_segments}
    required = (
        "pre_hurdle",
        "hurdle_deformation",
        "post_clearance_recovery",
    )
    if any(phase not in segments for phase in required):
        raise ValueError("成功軌跡に必須位相区間がない。")
    pre = segments["pre_hurdle"]
    deformation = segments["hurdle_deformation"]
    recovery = segments["post_clearance_recovery"]
    recovery_start = int(recovery["start"])
    recovery_end = int(recovery["end_exclusive"])
    stable_start = max(recovery_start + 1, recovery_end - stable_window_steps)
    return {
        "pre_hurdle": int(pre["start"]),
        "hurdle_deformation": int(deformation["start"])
        + max(1, int(deformation["steps"]) // 4),
        "post_clearance_recovery": recovery_start
        + max(1, int(recovery["steps"]) // 4),
        "stable_finish": stable_start,
    }


def build_phase_reset_specs(
    candidates: tuple[RescueDemoCandidate, ...],
    audit: dict[str, object],
) -> tuple[PhaseResetSpec, ...]:
    """四成功軌跡から各四位相の凍結リセット仕様を作る。"""
    rows_by_seed = {int(row["seed"]): row for row in audit["candidates"]}
    specs: list[PhaseResetSpec] = []
    for candidate_index, candidate in enumerate(candidates):
        arrays, _ = load_and_validate_branch_arrays(candidate)
        steps = select_phase_reset_steps(
            rows_by_seed[candidate.seed]["teacher_phase_segments"]
        )
        for phase_index, phase in enumerate(CURRICULUM_PHASES):
            source_step = steps[phase]
            if not 0 <= source_step < candidate.expected_steps:
                raise ValueError(f"位相リセット歩が軌跡範囲外である: {candidate.seed}")
            specs.append(
                PhaseResetSpec(
                    reset_id=f"seed_{candidate.seed}_{phase}",
                    phase=phase,
                    seed=candidate.seed,
                    profile=candidate.profile,
                    course_id=candidate.course_id,
                    start_runway_voxels=candidate.start_runway_voxels,
                    source_step=source_step,
                    source_branch_path=str(candidate.branch_path.resolve()),
                    source_branch_sha256=candidate.branch_sha256,
                    source_observation_sha256=array_sha256(
                        arrays["observations"][source_step]
                    ),
                    impulse_action_dimension=(candidate_index + phase_index) % 6,
                    impulse_sign=1 if (candidate_index + phase_index) % 2 == 0 else -1,
                )
            )
    if len(specs) != 16:
        raise RuntimeError("位相リセット仕様は16個でなければならない。")
    return tuple(specs)


def _replay_to_spec(
    spec: PhaseResetSpec,
    candidate: RescueDemoCandidate,
    *,
    teacher: PortfolioHeight1Teacher | None,
) -> tuple[GeneralObstacleEnv, np.ndarray, dict[str, object]]:
    """成功軌跡を指定歩まで再生し必要なら教師内部状態も進める。"""
    arrays, _ = load_and_validate_branch_arrays(candidate)
    course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
    if course.course_id != spec.course_id:
        raise ValueError("位相リセットのコース識別子が出典と一致しない。")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    observation, info = environment.reset(seed=spec.seed)
    if teacher is not None:
        teacher.reset(environment)
    for step in range(spec.source_step):
        if teacher is not None:
            teacher.predict(environment, observation, info)
        observation, _, terminated, truncated, info = environment.step(
            np.asarray(arrays["executed_actions"][step], dtype=np.float32)
        )
        if terminated or truncated:
            environment.close()
            raise RuntimeError(
                f"位相リセット前に軌跡が終了した: {spec.reset_id}, {step + 1}"
            )
    return environment, np.asarray(observation, dtype=np.float32), dict(info)


def audit_reset_reproducibility(
    spec: PhaseResetSpec,
    candidate: RescueDemoCandidate,
) -> dict[str, object]:
    """同一位相リセットを二回再生し観測と全質点を比較する。"""
    observations: list[np.ndarray] = []
    points: list[np.ndarray] = []
    infos: list[dict[str, object]] = []
    for _ in range(2):
        environment, observation, info = _replay_to_spec(
            spec,
            candidate,
            teacher=None,
        )
        try:
            observations.append(observation)
            points.append(
                np.asarray(
                    environment.object_pos_at_time(
                        environment.get_time(),
                        "robot",
                    ),
                    dtype=np.float64,
                )
            )
            infos.append(info)
        finally:
            environment.close()
    saved_arrays, _ = load_and_validate_branch_arrays(candidate)
    source_observation = np.asarray(
        saved_arrays["observations"][spec.source_step],
        dtype=np.float32,
    )
    return {
        "reset_id": spec.reset_id,
        "phase": spec.phase,
        "seed": spec.seed,
        "source_step": spec.source_step,
        "observation_shape": list(observations[0].shape),
        "maximum_repeat_observation_difference": float(
            np.max(np.abs(observations[0] - observations[1]))
        ),
        "maximum_repeat_point_difference": float(
            np.max(np.abs(points[0] - points[1]))
        ),
        "maximum_source_observation_difference": float(
            np.max(np.abs(observations[0] - source_observation))
        ),
        "source_observation_sha256_matches": (
            array_sha256(observations[0]) == spec.source_observation_sha256
        ),
        "x_position": float(infos[0]["x_position"]),
        "orientation_error": float(infos[0]["orientation_error"]),
        "angular_velocity": float(infos[0]["angular_velocity"]),
        "stall_steps": int(infos[0]["stall_steps"]),
        "raw_clearances": int(infos[0]["raw_clearances"]),
        "recovered_obstacles": int(infos[0]["recovered_obstacles"]),
    }


def evaluate_old_teacher_from_spec(
    spec: PhaseResetSpec,
    candidate: RescueDemoCandidate,
    *,
    impulse_magnitude: float,
) -> dict[str, object]:
    """旧教師を位相状態から実行し単発近傍摂動への回復も測る。"""
    teacher = PortfolioHeight1Teacher()
    environment, observation, info = _replay_to_spec(
        spec,
        candidate,
        teacher=teacher,
    )
    terminated = False
    truncated = False
    steps = 0
    impulse = np.zeros(environment.action_space.shape, dtype=np.float32)
    if impulse_magnitude > 0.0:
        impulse[spec.impulse_action_dimension] = (
            spec.impulse_sign * impulse_magnitude
        )
    realized_impulse = np.zeros_like(impulse)
    try:
        while not (terminated or truncated):
            action, _ = teacher.predict(environment, observation, info)
            clean_action = np.asarray(action, dtype=np.float32)
            executed_action = clean_action
            if steps == 0 and impulse_magnitude > 0.0:
                executed_action = np.clip(
                    clean_action + impulse,
                    environment.action_space.low,
                    environment.action_space.high,
                ).astype(np.float32)
                realized_impulse = executed_action - clean_action
            observation, _, terminated, truncated, info = environment.step(
                executed_action
            )
            steps += 1
    finally:
        environment.close()
    success = bool(
        info["course_complete"]
        and not info["hard_fall"]
        and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
    )
    return {
        "reset_id": spec.reset_id,
        "phase": spec.phase,
        "seed": spec.seed,
        "source_step": spec.source_step,
        "impulse_magnitude_requested": impulse_magnitude,
        "impulse_action_dimension": spec.impulse_action_dimension,
        "impulse_sign": spec.impulse_sign,
        "realized_impulse": realized_impulse.tolist(),
        "steps": steps,
        "success": success,
        "course_complete": bool(info["course_complete"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
    }


def _group_teacher_results(rows: list[dict[str, object]]) -> dict[str, object]:
    """旧教師結果を全体と位相別の成功・転倒数へ集約する。"""
    phase_rows: dict[str, dict[str, int]] = {}
    for phase in CURRICULUM_PHASES:
        selected = [row for row in rows if row["phase"] == phase]
        phase_rows[phase] = {
            "episodes": len(selected),
            "success_count": sum(bool(row["success"]) for row in selected),
            "hard_fall_count": sum(bool(row["hard_fall"]) for row in selected),
        }
    return {
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "phase_results": phase_rows,
        "rows": rows,
    }


def run_audit(output_dir: Path) -> dict[str, object]:
    """位相目録、二重リセット、旧教師精密・近傍基線を一括生成する。"""
    manifest = load_rescue_demo_manifest(DEFAULT_DEMO_MANIFEST)
    audit = json.loads(DEFAULT_DEMO_AUDIT.read_text(encoding="utf-8"))
    if not bool(audit["m2_3_1_gate"]["gate_passed"]):
        raise ValueError("M2.3.5aは合格済み成功示範だけを使用できる。")
    protected_paths = (
        DEFAULT_DEMO_MANIFEST.resolve(),
        DEFAULT_DEMO_AUDIT.resolve(),
        *(candidate.branch_path for candidate in manifest.candidates),
    )
    hashes_before = {str(path): sha256_file(path) for path in protected_paths}
    specs = build_phase_reset_specs(manifest.candidates, audit)
    candidates_by_seed = {
        candidate.seed: candidate for candidate in manifest.candidates
    }
    reset_rows = [
        audit_reset_reproducibility(spec, candidates_by_seed[spec.seed])
        for spec in specs
    ]
    exact_rows = [
        evaluate_old_teacher_from_spec(
            spec,
            candidates_by_seed[spec.seed],
            impulse_magnitude=0.0,
        )
        for spec in specs
    ]
    impulse_rows = [
        evaluate_old_teacher_from_spec(
            spec,
            candidates_by_seed[spec.seed],
            impulse_magnitude=IMPULSE_MAGNITUDE,
        )
        for spec in specs
    ]
    hashes_after = {str(path): sha256_file(path) for path in protected_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("M2.3.5a中に凍結成功示範が変更された。")
    reset_gate = bool(
        all(row["observation_shape"] == [95] for row in reset_rows)
        and all(row["maximum_repeat_observation_difference"] == 0.0 for row in reset_rows)
        and all(row["maximum_repeat_point_difference"] == 0.0 for row in reset_rows)
        and all(row["maximum_source_observation_difference"] == 0.0 for row in reset_rows)
        and all(row["source_observation_sha256_matches"] for row in reset_rows)
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    phase_manifest_path = output_dir / "phase_reset_manifest.json"
    phase_manifest_payload = {
        "version": "m2_3_5a_phase_reset_manifest_v1",
        "frozen": True,
        "stage": "hurdle_single",
        "split": "train",
        "teacher_training_only": True,
        "source_demo_manifest": str(DEFAULT_DEMO_MANIFEST.resolve()),
        "source_demo_manifest_sha256": manifest.sha256,
        "source_demo_audit": str(DEFAULT_DEMO_AUDIT.resolve()),
        "source_demo_audit_sha256": sha256_file(DEFAULT_DEMO_AUDIT),
        "curriculum_phases": list(CURRICULUM_PHASES),
        "specs": [spec.as_dict() for spec in specs],
    }
    phase_manifest_path.write_text(
        json.dumps(phase_manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = {
        "method": "m2_3_5a_phase_separated_success_trajectory_reset_audit",
        "stage": "hurdle_single",
        "split": "train",
        "run_name": output_dir.name,
        "phase_reset_manifest": str(phase_manifest_path.resolve()),
        "phase_reset_manifest_sha256": sha256_file(phase_manifest_path),
        "reset_spec_count": len(specs),
        "phase_spec_counts": {
            phase: sum(spec.phase == phase for spec in specs)
            for phase in CURRICULUM_PHASES
        },
        "reset_reproducibility": {
            "rows": reset_rows,
            "gate_passed": reset_gate,
        },
        "old_teacher_exact_baseline": _group_teacher_results(exact_rows),
        "old_teacher_single_impulse_baseline": {
            "impulse_magnitude": IMPULSE_MAGNITUDE,
            **_group_teacher_results(impulse_rows),
        },
        "m2_3_5a_gate": {
            "reset_reproducibility_passed": reset_gate,
            "exact_teacher_success_requirement": 16,
            "exact_teacher_success_count": sum(
                bool(row["success"]) for row in exact_rows
            ),
            "exact_teacher_hard_fall_count": sum(
                bool(row["hard_fall"]) for row in exact_rows
            ),
            "gate_passed": bool(
                reset_gate
                and all(bool(row["success"]) for row in exact_rows)
                and not any(bool(row["hard_fall"]) for row in exact_rows)
            ),
        },
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "training_steps": 0,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "teacher_interventions_in_student_test": 0,
        "protected_source_files_unchanged": True,
        "eligible_for_final_student_test": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    """出力名だけを受け取るM2.3.5a引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="成功軌跡を四位相に分けてリセットと旧教師近傍能力を監査する。"
    )
    parser.add_argument("--run-name", required=True)
    return parser


def main() -> None:
    """M2.3.5a監査を実行し要約を保存する。"""
    args = build_argument_parser().parse_args()
    result = run_audit(RUNS_ROOT / args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
