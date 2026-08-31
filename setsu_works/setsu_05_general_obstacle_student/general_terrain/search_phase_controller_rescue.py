"""M2.3.5cの訓練専用位相救援制御器を有界探索する。"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from general_terrain.audit_phase_reset_curriculum import PhaseResetSpec
from general_terrain.audit_rescue_demonstrations import (
    RescueDemoCandidate,
    load_and_validate_branch_arrays,
    load_rescue_demo_manifest,
    sha256_file,
)
from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m2_3_5c_controller_rescue_search_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "phase_controller_rescue_search"


@dataclass(frozen=True)
class ControllerCandidate:
    """一つの閉ループ救援制御器設定を保持する。"""

    first_switch_fraction: float
    post_clear_mode: str
    handoff_distance: float
    clearance_blend: float

    @property
    def candidate_id(self) -> str:
        """設定値から安定した短い識別子を返す。"""
        mode = self.post_clear_mode.replace("_", "-")
        return (
            f"f{self.first_switch_fraction:.3f}_m{mode}_"
            f"h{self.handoff_distance:.3f}_b{self.clearance_blend:.3f}"
        )


def _resolve_project_path(value: str) -> Path:
    """プロジェクト外への参照を拒否してパスを解決する。"""
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("救援探索の出典はプロジェクト配下でなければならない。")
    return path


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、出典ハッシュ、探索上限、評価隔離を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M2.3.5c規約は凍結済みでなければならない。")
    path_fields = (
        "phase_reset_manifest_path",
        "failed_phase_teacher_summary_path",
        "source_student_model_path",
    )
    resolved = {name: _resolve_project_path(str(payload[name])) for name in path_fields}
    for name, resolved_path in resolved.items():
        hash_name = f"{name.removesuffix('_path')}_sha256"
        if sha256_file(resolved_path) != str(payload[hash_name]):
            raise ValueError(f"M2.3.5c出典ハッシュが一致しない: {resolved_path}")
    failed = json.loads(
        resolved["failed_phase_teacher_summary_path"].read_text(encoding="utf-8")
    )
    if bool(failed["gate"]["gate_passed"]):
        raise ValueError("合格済み候補がある場合は救援探索へ切り替えられない。")
    if int(payload["candidate_limit_per_position"]) != 160:
        raise ValueError("位置ごとの候補上限は160でなければならない。")
    if int(payload["validation_episodes"]) or int(payload["holdout_episodes"]):
        raise ValueError("M2.3.5cは検証区分または留保区分へアクセスできない。")
    return {
        **payload,
        **resolved,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def load_pre_hurdle_specs(path: Path) -> tuple[PhaseResetSpec, ...]:
    """凍結目録から四つの越壁前位相仕様だけを読み込む。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("位相リセット目録は凍結済みでなければならない。")
    specs = tuple(
        PhaseResetSpec(**row)
        for row in payload["specs"]
        if row["phase"] == "pre_hurdle"
    )
    if len(specs) != 4 or len({spec.start_runway_voxels for spec in specs}) != 4:
        raise ValueError("越壁前位相は異なる四位置でなければならない。")
    return specs


def generate_candidates(protocol: Mapping[str, object]) -> tuple[ControllerCandidate, ...]:
    """既存値の近傍から重複のない有界候補列を作る。"""
    fractions = tuple(float(value) for value in protocol["switch_fractions"])
    modes = tuple(str(value) for value in protocol["post_clear_modes"])
    distances = tuple(float(value) for value in protocol["handoff_distances"])
    blends = tuple(float(value) for value in protocol["clearance_blends"])
    candidates: list[ControllerCandidate] = []
    seen: set[str] = set()

    def add(rows: Iterable[ControllerCandidate]) -> None:
        """候補識別子を用いて順序を保ったまま重複を除く。"""
        for row in rows:
            if row.candidate_id not in seen:
                seen.add(row.candidate_id)
                candidates.append(row)

    add(
        ControllerCandidate(fraction, "restart_then_flat", 0.45, 1.0)
        for fraction in fractions
    )
    add(
        ControllerCandidate(fraction, mode, 0.45, 1.0)
        for mode in modes
        for fraction in fractions
    )
    add(
        ControllerCandidate(fraction, mode, distance, 1.0)
        for distance in distances
        for mode in modes[:3]
        for fraction in fractions
    )
    add(
        ControllerCandidate(fraction, mode, distance, blend)
        for blend in blends
        for distance in distances
        for mode in modes[:3]
        for fraction in fractions
    )
    return tuple(candidates[: int(protocol["candidate_limit_per_position"])])


def _make_controller(
    candidate: ControllerCandidate,
    template: ClosedLoopHeight1Teacher,
) -> ClosedLoopHeight1Teacher:
    """読取専用方策を共有し候補値だけを持つ制御器を生成する。"""
    controller = copy.copy(template)
    controller.post_clear_mode = candidate.post_clear_mode
    controller.clearance_blend = candidate.clearance_blend
    controller.handoff_distance = candidate.handoff_distance
    controller.adaptive_handoff = True
    controller.first_switch_fraction = candidate.first_switch_fraction
    return controller


def evaluate_candidate(
    candidate: ControllerCandidate,
    spec: PhaseResetSpec,
    source: RescueDemoCandidate,
    template: ClosedLoopHeight1Teacher,
    *,
    impulse_magnitude: float,
) -> dict[str, object]:
    """成功軌跡の位相状態から候補制御器を厳格終了まで実行する。"""
    arrays, _ = load_and_validate_branch_arrays(source)
    course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    controller = _make_controller(candidate, template)
    observation, info = environment.reset(seed=spec.seed)
    controller.reset(environment)
    try:
        for step in range(spec.source_step):
            controller.predict(environment, info)
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(arrays["executed_actions"][step], dtype=np.float32)
            )
            if terminated or truncated:
                raise RuntimeError(f"候補探索前の位相再生が終了した: {spec.reset_id}")
        terminated = False
        truncated = False
        steps = 0
        maximum_angle = float(info["orientation_error"])
        while not (terminated or truncated):
            action, stage = controller.predict(environment, info)
            executed = np.asarray(action, dtype=np.float32)
            if steps == 0 and impulse_magnitude > 0.0:
                impulse = np.zeros(environment.action_space.shape, dtype=np.float32)
                impulse[spec.impulse_action_dimension] = (
                    spec.impulse_sign * impulse_magnitude
                )
                executed = np.clip(
                    executed + impulse,
                    environment.action_space.low,
                    environment.action_space.high,
                ).astype(np.float32)
            observation, _, terminated, truncated, info = environment.step(executed)
            steps += 1
            maximum_angle = max(maximum_angle, float(info["orientation_error"]))
    finally:
        environment.close()
    success = bool(
        info["course_complete"]
        and not info["hard_fall"]
        and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
    )
    return {
        "candidate_id": candidate.candidate_id,
        "seed": spec.seed,
        "start_runway_voxels": spec.start_runway_voxels,
        "impulse_magnitude": impulse_magnitude,
        "steps": steps,
        "final_stage": stage,
        "success": success,
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "maximum_angle": maximum_angle,
    }


def select_position_candidate(
    spec: PhaseResetSpec,
    source: RescueDemoCandidate,
    candidates: tuple[ControllerCandidate, ...],
    template: ClosedLoopHeight1Teacher,
    *,
    impulse_magnitude: float,
) -> dict[str, object]:
    """摂動成功を先に確認し精密状態も成功する最初の候補を選ぶ。"""
    attempts: list[dict[str, object]] = []
    selected: ControllerCandidate | None = None
    selected_exact: dict[str, object] | None = None
    selected_impulse: dict[str, object] | None = None
    for candidate in candidates:
        impulse = evaluate_candidate(
            candidate,
            spec,
            source,
            template,
            impulse_magnitude=impulse_magnitude,
        )
        row: dict[str, object] = {
            "candidate": asdict(candidate),
            "candidate_id": candidate.candidate_id,
            "impulse": impulse,
            "exact": None,
        }
        if bool(impulse["success"]):
            exact = evaluate_candidate(
                candidate,
                spec,
                source,
                template,
                impulse_magnitude=0.0,
            )
            row["exact"] = exact
            if bool(exact["success"]):
                selected = candidate
                selected_exact = exact
                selected_impulse = impulse
                attempts.append(row)
                break
        attempts.append(row)
    return {
        "reset_id": spec.reset_id,
        "seed": spec.seed,
        "start_runway_voxels": spec.start_runway_voxels,
        "attempt_count": len(attempts),
        "selected": asdict(selected) if selected is not None else None,
        "selected_candidate_id": selected.candidate_id if selected is not None else None,
        "selected_exact": selected_exact,
        "selected_impulse": selected_impulse,
        "attempts": attempts,
    }


def evaluate_gate(
    position_results: list[dict[str, object]],
    requirements: Mapping[str, int],
) -> dict[str, object]:
    """四位置で精密状態と摂動状態が共に安全成功したか判定する。"""
    selected = [row for row in position_results if row["selected"] is not None]
    exact_success = sum(bool(row["selected_exact"]["success"]) for row in selected)
    impulse_success = sum(bool(row["selected_impulse"]["success"]) for row in selected)
    exact_hard = sum(bool(row["selected_exact"]["hard_fall"]) for row in selected)
    impulse_hard = sum(bool(row["selected_impulse"]["hard_fall"]) for row in selected)
    checks = {
        "exact_success": exact_success
        >= requirements["minimum_selected_exact_success_count"],
        "impulse_success": impulse_success
        >= requirements["minimum_selected_impulse_success_count"],
        "exact_hard_fall": exact_hard
        <= requirements["maximum_selected_exact_hard_fall_count"],
        "impulse_hard_fall": impulse_hard
        <= requirements["maximum_selected_impulse_hard_fall_count"],
    }
    return {
        "requirements": dict(requirements),
        "selected_position_count": len(selected),
        "exact_success_count": exact_success,
        "impulse_success_count": impulse_success,
        "exact_hard_fall_count": exact_hard,
        "impulse_hard_fall_count": impulse_hard,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def run(protocol: dict[str, object], output_dir: Path) -> dict[str, object]:
    """位置別候補探索、合格判定、出典不変確認を一括実行する。"""
    specs = load_pre_hurdle_specs(Path(protocol["phase_reset_manifest_path"]))
    manifest = load_rescue_demo_manifest()
    sources = {candidate.seed: candidate for candidate in manifest.candidates}
    candidates = generate_candidates(protocol)
    template = ClosedLoopHeight1Teacher(
        post_clear_mode="restart_then_flat",
        clearance_blend=1.0,
        handoff_distance=0.45,
        adaptive_handoff=True,
        first_switch_fraction=0.25,
    )
    protected = (
        Path(protocol["source_path"]),
        Path(protocol["phase_reset_manifest_path"]),
        Path(protocol["failed_phase_teacher_summary_path"]),
        Path(protocol["source_student_model_path"]),
        *(candidate.branch_path for candidate in manifest.candidates),
    )
    before = {str(path): sha256_file(path) for path in protected}
    output_dir.mkdir(parents=True, exist_ok=False)
    position_results = [
        select_position_candidate(
            spec,
            sources[spec.seed],
            candidates,
            template,
            impulse_magnitude=float(protocol["impulse_magnitude"]),
        )
        for spec in specs
    ]
    gate = evaluate_gate(position_results, protocol["gate"])
    after = {str(path): sha256_file(path) for path in protected}
    if before != after:
        raise RuntimeError("M2.3.5c中に凍結出典が変更された。")
    result = {
        "method": "m2_3_5c_position_routed_closed_loop_rescue_search",
        "stage": "hurdle_single",
        "split": "train",
        "run_name": output_dir.name,
        "protocol_sha256": protocol["sha256"],
        "teacher_training_only": True,
        "candidate_count": len(candidates),
        "candidate_limit_per_position": protocol["candidate_limit_per_position"],
        "position_results": position_results,
        "gate": gate,
        "checkpoint_disposition": (
            "phase_rescue_route_candidate"
            if bool(gate["gate_passed"])
            else "no_controller_route_found"
        ),
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "training_steps": 0,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "teacher_interventions_in_student_test": 0,
        "protected_source_files_unchanged": True,
        "eligible_for_m2_4": False,
        "eligible_for_final_student_test": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    """凍結規約と一意な出力名だけを受け取る。"""
    parser = argparse.ArgumentParser(
        description="訓練専用の位置別閉ループ救援制御器を有界探索する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """M2.3.5cの有界制御器救援探索を実行する。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
