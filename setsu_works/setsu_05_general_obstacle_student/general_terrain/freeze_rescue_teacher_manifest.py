"""M2.3.5fの二課程救援教師目録を独立再生後に凍結する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from general_terrain.audit_rescue_demonstrations import sha256_file
from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher
from general_terrain.rescue_reset_manifest import load_rescue_reset_manifest
from general_terrain.search_takeover_recalibration_rescue import (
    PrefixReplay,
    RecalibratedControllerCandidate,
    collect_prefix_replays,
    evaluate_candidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m2_3_5f_teacher_manifest_gate_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "rescue_teacher_manifest"


def _resolve_project_path(value: str) -> Path:
    """プロジェクト配下だけに限定して相対パスを解決する。"""
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("教師目録の出典はプロジェクト配下でなければならない。")
    return path


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、二課程出典、再生回数、評価隔離を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M2.3.5f規約は凍結済みでなければならない。")
    fields = (
        "phase_teacher_summary_path",
        "takeover_search_summary_path",
        "reset_manifest_path",
        "source_student_model_path",
    )
    resolved = {name: _resolve_project_path(str(payload[name])) for name in fields}
    for name, resolved_path in resolved.items():
        hash_name = f"{name.removesuffix('_path')}_sha256"
        if sha256_file(resolved_path) != str(payload[hash_name]):
            raise ValueError(f"M2.3.5f出典ハッシュが一致しない: {resolved_path}")
    phase = json.loads(resolved["phase_teacher_summary_path"].read_text(encoding="utf-8"))
    takeover = json.loads(
        resolved["takeover_search_summary_path"].read_text(encoding="utf-8")
    )
    if int(phase["phase_exact_evaluation"]["success_count"]) != 16:
        raise ValueError("位相教師の精密基線が完全成功ではない。")
    if int(phase["phase_impulse_evaluation"]["success_count"]) != 16:
        raise ValueError("位相教師の摂動基線が完全成功ではない。")
    if not bool(takeover["gate"]["gate_passed"]):
        raise ValueError("学生引継ぎ再校正探索が合格していない。")
    if int(payload["independent_replay_repetitions"]) != 2:
        raise ValueError("独立再生は二回でなければならない。")
    if int(payload["validation_episodes"]) or int(payload["holdout_episodes"]):
        raise ValueError("M2.3.5fは検証区分または留保区分へアクセスできない。")
    return {
        **payload,
        **resolved,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def takeover_routes_from_search(
    summary: Mapping[str, object],
) -> dict[int, RecalibratedControllerCandidate]:
    """合格探索結果から安全成功した位置別設定だけを読み取る。"""
    routes: dict[int, RecalibratedControllerCandidate] = {}
    for row in summary["position_results"]:
        selected = row["selected"]
        if selected is None:
            continue
        evaluation = row["selected_evaluation"]
        if not bool(evaluation["success"]) or bool(evaluation["hard_fall"]):
            raise ValueError("選択済み引継ぎ経路が安全成功条件を満たさない。")
        routes[int(row["start_runway_voxels"])] = (
            RecalibratedControllerCandidate(**selected)
        )
    if len(routes) < 4:
        raise ValueError("引継ぎ経路は少なくとも四位置必要である。")
    return routes


def evaluate_fallback(
    replay: PrefixReplay,
) -> dict[str, object]:
    """未対応位置を引継ぎ時再初期化した旧教師で評価する。"""
    spec = replay.spec
    course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    teacher = PortfolioHeight1Teacher()
    observation, info = environment.reset(seed=spec.seed)
    try:
        for step, action in enumerate(replay.actions, start=1):
            observation, _, terminated, truncated, info = environment.step(action)
            if terminated or truncated:
                raise RuntimeError(
                    f"旧教師代替経路の学生再生が早期終了した: {spec.seed}, {step}"
                )
        teacher.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        stage = ""
        while not (terminated or truncated):
            action, stage = teacher.predict(environment, observation, info)
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
        "seed": spec.seed,
        "start_runway_voxels": spec.start_runway_voxels,
        "route_type": "portfolio_fallback",
        "rescue_steps": steps,
        "final_stage": stage,
        "success": success,
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
    }


def stable_outcome(row: Mapping[str, object]) -> tuple[object, ...]:
    """独立再生間で一致すべき終了結果を固定順で返す。"""
    return (
        bool(row["success"]),
        bool(row["hard_fall"]),
        str(row["failure_reason"]),
        int(row["raw_clearances"]),
        int(row["recovered_obstacles"]),
        int(row.get("rescue_steps", 0)),
    )


def evaluate_takeover_curriculum(
    routes: Mapping[int, RecalibratedControllerCandidate],
    replays: tuple[PrefixReplay, ...],
    *,
    repetitions: int,
) -> dict[str, object]:
    """十一引継ぎ状態を独立二回再生し経路再現性を検査する。"""
    template = ClosedLoopHeight1Teacher(
        post_clear_mode="restart_then_flat",
        clearance_blend=1.0,
        handoff_distance=0.45,
        adaptive_handoff=True,
        first_switch_fraction=0.25,
    )
    rows: list[dict[str, object]] = []
    all_reproducible = True
    for replay in replays:
        route = routes.get(replay.spec.start_runway_voxels)
        repeated: list[dict[str, object]] = []
        for _ in range(repetitions):
            if route is None:
                evaluation = evaluate_fallback(replay)
            else:
                evaluation = evaluate_candidate(route, replay, template)
                evaluation = {
                    **evaluation,
                    "route_type": "takeover_recalibrated_controller",
                }
            repeated.append(evaluation)
        reproducible = all(
            stable_outcome(row) == stable_outcome(repeated[0]) for row in repeated[1:]
        )
        all_reproducible = all_reproducible and reproducible
        rows.append(
            {
                "seed": replay.spec.seed,
                "start_runway_voxels": replay.spec.start_runway_voxels,
                "route_available": route is not None,
                "route": asdict(route) if route is not None else None,
                "reproducible": reproducible,
                "evaluation": repeated[0],
                "repetitions": repeated,
            }
        )
    primary = [row["evaluation"] for row in rows]
    return {
        "episodes": len(rows),
        "repetitions": repetitions,
        "all_reproducible": all_reproducible,
        "success_count": sum(bool(row["success"]) for row in primary),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in primary),
        "safe_stall_count": sum(
            not bool(row["success"]) and not bool(row["hard_fall"]) for row in primary
        ),
        "route_coverage_count": sum(row["route_available"] for row in rows),
        "rows": rows,
    }


def evaluate_gate(
    phase_exact: Mapping[str, object],
    phase_impulse: Mapping[str, object],
    takeover: Mapping[str, object],
    requirements: Mapping[str, object],
) -> dict[str, object]:
    """二課程の成功、安全、再現性条件を同時に判定する。"""
    checks = {
        "phase_exact_success": int(phase_exact["success_count"])
        >= int(requirements["minimum_phase_exact_success_count"]),
        "phase_impulse_success": int(phase_impulse["success_count"])
        >= int(requirements["minimum_phase_impulse_success_count"]),
        "phase_exact_hard_fall": int(phase_exact["hard_fall_count"])
        <= int(requirements["maximum_phase_exact_hard_fall_count"]),
        "phase_impulse_hard_fall": int(phase_impulse["hard_fall_count"])
        <= int(requirements["maximum_phase_impulse_hard_fall_count"]),
        "reset_success": int(takeover["success_count"])
        >= int(requirements["minimum_reset_state_success_count"]),
        "reset_hard_fall": int(takeover["hard_fall_count"])
        <= int(requirements["maximum_reset_state_hard_fall_count"]),
        "reset_reproducibility": bool(takeover["all_reproducible"])
        is bool(requirements["require_reset_reproducibility"]),
    }
    return {
        "requirements": dict(requirements),
        "checks": checks,
        "gate_passed": all(checks.values()),
        "eligible_for_m2_4": all(checks.values()),
    }


def run(protocol: dict[str, object], output_dir: Path) -> dict[str, object]:
    """独立再生、二課程門判定、教師目録凍結、出典不変確認を行う。"""
    from sb3_contrib import RecurrentPPO

    phase_summary = json.loads(
        Path(protocol["phase_teacher_summary_path"]).read_text(encoding="utf-8")
    )
    takeover_summary = json.loads(
        Path(protocol["takeover_search_summary_path"]).read_text(encoding="utf-8")
    )
    routes = takeover_routes_from_search(takeover_summary)
    manifest = load_rescue_reset_manifest(Path(protocol["reset_manifest_path"]))
    frozen_student = RecurrentPPO.load(
        Path(protocol["source_student_model_path"]),
        device="cpu",
    )
    replays = collect_prefix_replays(frozen_student, manifest)
    protected = (
        Path(protocol["source_path"]),
        Path(protocol["phase_teacher_summary_path"]),
        Path(protocol["takeover_search_summary_path"]),
        Path(protocol["reset_manifest_path"]),
        Path(protocol["source_student_model_path"]),
    )
    before = {str(path): sha256_file(path) for path in protected}
    output_dir.mkdir(parents=True, exist_ok=False)
    takeover_evaluation = evaluate_takeover_curriculum(
        routes,
        replays,
        repetitions=int(protocol["independent_replay_repetitions"]),
    )
    phase_exact = phase_summary["phase_exact_evaluation"]
    phase_impulse = phase_summary["phase_impulse_evaluation"]
    gate = evaluate_gate(phase_exact, phase_impulse, takeover_evaluation, protocol["gate"])
    teacher_manifest = {
        "version": "m2_3_5f_separated_training_teacher_manifest_v1",
        "frozen": True,
        "teacher_training_only": True,
        "stage": "hurdle_single",
        "split": "train",
        "phase_reset_curriculum": {
            "source_summary": str(Path(protocol["phase_teacher_summary_path"])),
            "source_summary_sha256": protocol["phase_teacher_summary_sha256"],
            "activation_phase": "pre_hurdle",
            "routes": phase_summary["routes"],
            "later_phase_fallback": "portfolio_height1_teacher_v1",
        },
        "student_takeover_curriculum": {
            "source_summary": str(Path(protocol["takeover_search_summary_path"])),
            "source_summary_sha256": protocol["takeover_search_summary_sha256"],
            "reinitialize_at_takeover": True,
            "routes": {str(position): asdict(route) for position, route in routes.items()},
            "uncovered_positions": sorted(set(range(20, 31)) - set(routes)),
            "uncovered_fallback": "portfolio_height1_teacher_v1",
        },
        "student_observation_privileged": False,
        "teacher_may_use_training_context": True,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "teacher_interventions_in_final_student_test": 0,
        "eligible_for_m2_4": bool(gate["eligible_for_m2_4"]),
    }
    manifest_path = output_dir / "teacher_manifest.json"
    manifest_path.write_text(
        json.dumps(teacher_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    after = {str(path): sha256_file(path) for path in protected}
    if before != after:
        raise RuntimeError("M2.3.5f中に凍結出典が変更された。")
    result = {
        "method": "m2_3_5f_separated_curriculum_teacher_manifest_gate",
        "stage": "hurdle_single",
        "split": "train",
        "run_name": output_dir.name,
        "protocol_sha256": protocol["sha256"],
        "teacher_training_only": True,
        "phase_exact_evaluation": {
            "episodes": phase_exact["episodes"],
            "success_count": phase_exact["success_count"],
            "hard_fall_count": phase_exact["hard_fall_count"],
            "source_summary_reused": True,
        },
        "phase_impulse_evaluation": {
            "episodes": phase_impulse["episodes"],
            "success_count": phase_impulse["success_count"],
            "hard_fall_count": phase_impulse["hard_fall_count"],
            "source_summary_reused": True,
        },
        "takeover_independent_evaluation": takeover_evaluation,
        "gate": gate,
        "teacher_manifest": str(manifest_path.resolve()),
        "teacher_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_disposition": (
            "m2_4_training_teacher_manifest"
            if bool(gate["eligible_for_m2_4"])
            else "quarantined_m2_3_5f_manifest"
        ),
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "training_steps": 0,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "teacher_interventions_in_student_test": 0,
        "protected_source_files_unchanged": True,
        "eligible_for_m2_4": bool(gate["eligible_for_m2_4"]),
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
        description="分離した二課程の訓練専用救援教師目録を凍結する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """M2.3.5fの独立再生と教師目録門を実行する。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
