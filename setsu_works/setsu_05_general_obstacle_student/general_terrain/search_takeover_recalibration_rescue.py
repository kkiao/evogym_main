"""M2.3.5eの学生引継ぎ状態専用再校正救援制御器を有界探索する。"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from general_terrain.audit_rescue_demonstrations import sha256_file
from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.rescue_reset_manifest import (
    RescueResetManifest,
    RescueResetSpec,
    load_rescue_reset_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "config"
    / "m2_3_5e_takeover_recalibration_search_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "takeover_recalibration_rescue_search"


@dataclass(frozen=True)
class RecalibratedControllerCandidate:
    """引継ぎ時に初期化する一つの閉ループ制御器設定を保持する。"""

    first_switch_fraction: float
    post_clear_mode: str
    handoff_distance: float
    clearance_blend: float
    clearance_family: str
    adaptive_handoff: bool

    @property
    def candidate_id(self) -> str:
        """設定値から安定した識別子を返す。"""
        mode = self.post_clear_mode.replace("_", "-")
        adaptive = "a1" if self.adaptive_handoff else "a0"
        return (
            f"{self.clearance_family}_f{self.first_switch_fraction:.3f}_"
            f"m{mode}_h{self.handoff_distance:.3f}_"
            f"b{self.clearance_blend:.3f}_{adaptive}"
        )


@dataclass(frozen=True)
class PrefixReplay:
    """一つの凍結学生引継ぎ状態へ至る決定論的動作列を保持する。"""

    spec: RescueResetSpec
    actions: np.ndarray


def _resolve_project_path(value: str) -> Path:
    """プロジェクト配下だけに限定して相対パスを解決する。"""
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("再校正探索の出典はプロジェクト配下でなければならない。")
    return path


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、失敗根拠、出典ハッシュ、評価隔離を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M2.3.5e規約は凍結済みでなければならない。")
    fields = (
        "failed_routed_teacher_summary_path",
        "reset_manifest_path",
        "source_student_model_path",
    )
    resolved = {name: _resolve_project_path(str(payload[name])) for name in fields}
    for name, resolved_path in resolved.items():
        hash_name = f"{name.removesuffix('_path')}_sha256"
        if sha256_file(resolved_path) != str(payload[hash_name]):
            raise ValueError(f"M2.3.5e出典ハッシュが一致しない: {resolved_path}")
    failed = json.loads(
        resolved["failed_routed_teacher_summary_path"].read_text(encoding="utf-8")
    )
    if bool(failed["gate"]["gate_passed"]):
        raise ValueError("合格済み教師から再校正探索へ切り替えることはできない。")
    if int(failed["reset_state_evaluation"]["success_count"]) != 1:
        raise ValueError("再校正探索の失敗基線は一成功でなければならない。")
    if int(payload["candidate_limit_per_position"]) != 192:
        raise ValueError("位置ごとの再校正候補上限は192でなければならない。")
    if not bool(payload["reinitialize_at_takeover"]):
        raise ValueError("M2.3.5eは引継ぎ時再初期化を必須とする。")
    if int(payload["validation_episodes"]) or int(payload["holdout_episodes"]):
        raise ValueError("M2.3.5eは検証区分または留保区分へアクセスできない。")
    return {
        **payload,
        **resolved,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def generate_candidates(
    protocol: Mapping[str, object],
) -> tuple[RecalibratedControllerCandidate, ...]:
    """第一・第二越壁系列と安全復帰値から有界候補列を作る。"""
    fractions = tuple(float(value) for value in protocol["switch_fractions"])
    modes = tuple(str(value) for value in protocol["post_clear_modes"])
    distances = tuple(float(value) for value in protocol["handoff_distances"])
    blends = tuple(float(value) for value in protocol["clearance_blends"])
    adaptive_values = tuple(bool(value) for value in protocol["adaptive_handoff_values"])
    candidates: list[RecalibratedControllerCandidate] = []
    seen: set[str] = set()

    def add(rows: Iterable[RecalibratedControllerCandidate]) -> None:
        """候補順を保ちながら識別子重複を除く。"""
        for row in rows:
            if row.candidate_id not in seen:
                seen.add(row.candidate_id)
                candidates.append(row)

    add(
        RecalibratedControllerCandidate(
            fraction,
            mode,
            0.45,
            1.0,
            "first",
            True,
        )
        for mode in modes
        for fraction in fractions
    )
    add(
        RecalibratedControllerCandidate(
            0.5,
            mode,
            distance,
            1.0,
            "second",
            adaptive,
        )
        for adaptive in adaptive_values
        for distance in distances
        for mode in modes
    )
    add(
        RecalibratedControllerCandidate(
            fraction,
            mode,
            0.45,
            blend,
            "first",
            True,
        )
        for blend in blends[1:]
        for mode in modes
        for fraction in fractions
    )
    add(
        RecalibratedControllerCandidate(
            0.5,
            mode,
            distance,
            blend,
            "second",
            adaptive,
        )
        for blend in blends[1:]
        for adaptive in adaptive_values
        for distance in distances
        for mode in modes
    )
    add(
        RecalibratedControllerCandidate(
            fraction,
            mode,
            distance,
            1.0,
            "first",
            True,
        )
        for distance in distances
        for mode in modes
        for fraction in fractions
    )
    return tuple(candidates[: int(protocol["candidate_limit_per_position"])])


def collect_prefix_replays(
    frozen_student: Any,
    manifest: RescueResetManifest,
) -> tuple[PrefixReplay, ...]:
    """凍結学生を各引継ぎ歩まで一度だけ実行し動作列を保存する。"""
    replays: list[PrefixReplay] = []
    for spec in manifest.states:
        course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        actions: list[np.ndarray] = []
        recurrent_state: Any = None
        episode_start = np.ones((1,), dtype=bool)
        try:
            observation, info = environment.reset(seed=spec.seed)
            for step in range(spec.prefix_steps):
                action, recurrent_state = frozen_student.predict(
                    observation,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                action_array = np.asarray(action, dtype=np.float32)
                actions.append(action_array.copy())
                observation, _, terminated, truncated, info = environment.step(
                    action_array
                )
                if terminated or truncated:
                    raise RuntimeError(
                        f"凍結学生が引継ぎ前に終了した: {spec.seed}, {step + 1}"
                    )
        finally:
            environment.close()
        if not np.isclose(float(info["x_position"]), spec.x_position, atol=1e-9):
            raise RuntimeError(f"学生引継ぎ状態が凍結値と一致しない: {spec.seed}")
        replays.append(PrefixReplay(spec=spec, actions=np.asarray(actions)))
    return tuple(replays)


def _make_controller(
    candidate: RecalibratedControllerCandidate,
    template: ClosedLoopHeight1Teacher,
) -> ClosedLoopHeight1Teacher:
    """読取専用方策を共有し候補値だけを複製する。"""
    controller = copy.copy(template)
    controller.post_clear_mode = candidate.post_clear_mode
    controller.clearance_blend = candidate.clearance_blend
    controller.handoff_distance = candidate.handoff_distance
    controller.adaptive_handoff = candidate.adaptive_handoff
    controller.clearance_family = candidate.clearance_family
    controller.first_switch_fraction = candidate.first_switch_fraction
    return controller


def evaluate_candidate(
    candidate: RecalibratedControllerCandidate,
    replay: PrefixReplay,
    template: ClosedLoopHeight1Teacher,
) -> dict[str, object]:
    """学生引継ぎ状態で制御器を再校正し厳格終了まで評価する。"""
    spec = replay.spec
    course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    controller = _make_controller(candidate, template)
    observation, info = environment.reset(seed=spec.seed)
    try:
        for step, action in enumerate(replay.actions, start=1):
            observation, _, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"再校正候補の学生再生が早期終了した: {spec.seed}, {step}"
                )
        controller.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        stage = ""
        while not (terminated or truncated):
            action, stage = controller.predict(environment, info)
            observation, _, terminated, truncated, info = environment.step(action)
            steps += 1
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
        "prefix_steps": spec.prefix_steps,
        "rescue_steps": steps,
        "final_stage": stage,
        "success": success,
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
    }


def select_candidate(
    replay: PrefixReplay,
    candidates: tuple[RecalibratedControllerCandidate, ...],
    template: ClosedLoopHeight1Teacher,
) -> dict[str, object]:
    """安全成功する最初の再校正候補を位置ごとに選ぶ。"""
    attempts: list[dict[str, object]] = []
    selected: RecalibratedControllerCandidate | None = None
    selected_evaluation: dict[str, object] | None = None
    for candidate in candidates:
        evaluation = evaluate_candidate(candidate, replay, template)
        attempts.append(
            {
                "candidate": asdict(candidate),
                "candidate_id": candidate.candidate_id,
                "evaluation": evaluation,
            }
        )
        if bool(evaluation["success"]) and not bool(evaluation["hard_fall"]):
            selected = candidate
            selected_evaluation = evaluation
            break
    return {
        "seed": replay.spec.seed,
        "start_runway_voxels": replay.spec.start_runway_voxels,
        "prefix_steps": replay.spec.prefix_steps,
        "attempt_count": len(attempts),
        "selected": asdict(selected) if selected is not None else None,
        "selected_candidate_id": selected.candidate_id if selected is not None else None,
        "selected_evaluation": selected_evaluation,
        "attempts": attempts,
    }


def evaluate_gate(
    position_results: list[dict[str, object]],
    requirements: Mapping[str, int],
) -> dict[str, object]:
    """選択済み引継ぎ状態の成功数と転倒数を判定する。"""
    selected = [row for row in position_results if row["selected"] is not None]
    success_count = sum(
        bool(row["selected_evaluation"]["success"]) for row in selected
    )
    hard_fall_count = sum(
        bool(row["selected_evaluation"]["hard_fall"]) for row in selected
    )
    checks = {
        "success": success_count >= requirements["minimum_selected_success_count"],
        "hard_fall": hard_fall_count
        <= requirements["maximum_selected_hard_fall_count"],
    }
    return {
        "requirements": dict(requirements),
        "selected_position_count": len(selected),
        "success_count": success_count,
        "hard_fall_count": hard_fall_count,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }


def run(protocol: dict[str, object], output_dir: Path) -> dict[str, object]:
    """前置動作保存、位置別探索、門判定、出典不変確認を実行する。"""
    from sb3_contrib import RecurrentPPO

    manifest = load_rescue_reset_manifest(Path(protocol["reset_manifest_path"]))
    frozen_student = RecurrentPPO.load(
        Path(protocol["source_student_model_path"]),
        device="cpu",
    )
    replays = collect_prefix_replays(frozen_student, manifest)
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
        Path(protocol["failed_routed_teacher_summary_path"]),
        Path(protocol["reset_manifest_path"]),
        Path(protocol["source_student_model_path"]),
    )
    before = {str(path): sha256_file(path) for path in protected}
    output_dir.mkdir(parents=True, exist_ok=False)
    position_results = [
        select_candidate(replay, candidates, template) for replay in replays
    ]
    gate = evaluate_gate(position_results, protocol["gate"])
    after = {str(path): sha256_file(path) for path in protected}
    if before != after:
        raise RuntimeError("M2.3.5e中に凍結出典が変更された。")
    result = {
        "method": "m2_3_5e_takeover_reinitialized_controller_search",
        "stage": "hurdle_single",
        "split": "train",
        "run_name": output_dir.name,
        "protocol_sha256": protocol["sha256"],
        "teacher_training_only": True,
        "reinitialize_at_takeover": True,
        "candidate_count": len(candidates),
        "candidate_limit_per_position": protocol["candidate_limit_per_position"],
        "position_results": position_results,
        "gate": gate,
        "checkpoint_disposition": (
            "takeover_rescue_route_candidate"
            if bool(gate["gate_passed"])
            else "no_takeover_recalibration_route_found"
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
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    """凍結規約と一意な出力名だけを受け取る。"""
    parser = argparse.ArgumentParser(
        description="学生引継ぎ時に再校正する訓練専用救援制御器を有界探索する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """M2.3.5eの引継ぎ再校正探索を実行する。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
