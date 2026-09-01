"""M2.3.5dの位置・位相分離救援教師を三組の訓練状態で検収する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from general_terrain.audit_phase_reset_curriculum import (
    CURRICULUM_PHASES,
    PhaseResetSpec,
)
from general_terrain.audit_rescue_demonstrations import (
    RescueDemoCandidate,
    load_and_validate_branch_arrays,
    load_rescue_demo_manifest,
    sha256_file,
)
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.phase_routed_rescue_teacher import (
    PhaseRoutedRescueTeacher,
    route_configs_from_search,
)
from general_terrain.rescue_reset_manifest import (
    RescueResetManifest,
    load_rescue_reset_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m2_3_5d_routed_rescue_teacher_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "phase_routed_rescue_teacher"


def _resolve_project_path(value: str) -> Path:
    """プロジェクト配下だけに限定して相対パスを解決する。"""
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("経路救援教師の出典はプロジェクト配下でなければならない。")
    return path


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、全出典ハッシュ、訓練区分限定を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M2.3.5d規約は凍結済みでなければならない。")
    path_fields = (
        "controller_search_summary_path",
        "phase_reset_manifest_path",
        "reset_manifest_path",
        "source_student_model_path",
    )
    resolved = {name: _resolve_project_path(str(payload[name])) for name in path_fields}
    for name, resolved_path in resolved.items():
        hash_name = f"{name.removesuffix('_path')}_sha256"
        if sha256_file(resolved_path) != str(payload[hash_name]):
            raise ValueError(f"M2.3.5d出典ハッシュが一致しない: {resolved_path}")
    search = json.loads(
        resolved["controller_search_summary_path"].read_text(encoding="utf-8")
    )
    route_configs_from_search(search)
    if str(payload["route_activation_phase"]) != "pre_hurdle":
        raise ValueError("位置別経路は越壁前位相だけで起動しなければならない。")
    if int(payload["validation_episodes"]) or int(payload["holdout_episodes"]):
        raise ValueError("M2.3.5dは検証区分または留保区分へアクセスできない。")
    return {
        **payload,
        **resolved,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def load_phase_specs(path: Path) -> tuple[PhaseResetSpec, ...]:
    """凍結目録を十六個の位相リセット仕様へ変換する。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = tuple(PhaseResetSpec(**row) for row in payload["specs"])
    if not bool(payload.get("frozen", False)) or len(specs) != 16:
        raise ValueError("凍結位相リセット仕様は十六個でなければならない。")
    return specs


def replay_to_phase(
    teacher: PhaseRoutedRescueTeacher,
    spec: PhaseResetSpec,
    source: RescueDemoCandidate,
) -> tuple[GeneralObstacleEnv, np.ndarray, dict[str, object]]:
    """成功軌跡を位相地点まで再生し教師状態も毎歩追従させる。"""
    arrays, _ = load_and_validate_branch_arrays(source)
    course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    observation, info = environment.reset(seed=spec.seed)
    teacher.reset(environment)
    for step in range(spec.source_step):
        teacher.observe(environment, observation, info)
        observation, _, terminated, truncated, info = environment.step(
            np.asarray(arrays["executed_actions"][step], dtype=np.float32)
        )
        if terminated or truncated:
            environment.close()
            raise RuntimeError(f"位相地点前に再生が終了した: {spec.reset_id}")
    return environment, np.asarray(observation, dtype=np.float32), dict(info)


def evaluate_phase_specs(
    routes: Mapping[int, object],
    specs: tuple[PhaseResetSpec, ...],
    sources: Mapping[int, RescueDemoCandidate],
    *,
    impulse_magnitude: float,
) -> dict[str, object]:
    """十六位相を精密状態または単発摂動から厳格終了まで評価する。"""
    rows: list[dict[str, object]] = []
    for spec in specs:
        teacher = PhaseRoutedRescueTeacher(routes)
        environment, observation, info = replay_to_phase(
            teacher,
            spec,
            sources[spec.seed],
        )
        teacher.activate(environment, info)
        terminated = False
        truncated = False
        steps = 0
        stage = ""
        try:
            while not (terminated or truncated):
                action, stage = teacher.predict(environment, observation, info)
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
        finally:
            environment.close()
        success = bool(
            info["course_complete"]
            and not info["hard_fall"]
            and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
        )
        rows.append(
            {
                "reset_id": spec.reset_id,
                "phase": spec.phase,
                "seed": spec.seed,
                "start_runway_voxels": spec.start_runway_voxels,
                "activation_phase": teacher.activation_phase,
                "used_routed_controller": teacher.use_routed_controller,
                "steps": steps,
                "final_stage": stage,
                "success": success,
                "hard_fall": bool(info["hard_fall"]),
                "failure_reason": str(info["failure_reason"]),
                "raw_clearances": int(info["raw_clearances"]),
                "recovered_obstacles": int(info["recovered_obstacles"]),
            }
        )
    return {
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "routed_episode_count": sum(bool(row["used_routed_controller"]) for row in rows),
        "phase_results": {
            phase: {
                "episodes": sum(row["phase"] == phase for row in rows),
                "success_count": sum(
                    bool(row["success"]) and row["phase"] == phase for row in rows
                ),
                "hard_fall_count": sum(
                    bool(row["hard_fall"]) and row["phase"] == phase for row in rows
                ),
            }
            for phase in CURRICULUM_PHASES
        },
        "rows": rows,
    }


def evaluate_reset_states(
    routes: Mapping[int, object],
    frozen_student: Any,
    manifest: RescueResetManifest,
) -> dict[str, object]:
    """十一個の凍結学生前置状態から経路教師を厳格評価する。"""
    rows: list[dict[str, object]] = []
    for spec in manifest.states:
        course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        teacher = PhaseRoutedRescueTeacher(routes)
        recurrent_state: Any = None
        episode_start = np.ones((1,), dtype=bool)
        terminated = False
        truncated = False
        steps = 0
        stage = ""
        try:
            observation, info = environment.reset(seed=spec.seed)
            teacher.reset(environment)
            for prefix_step in range(spec.prefix_steps):
                teacher.observe(environment, observation, info)
                action, recurrent_state = frozen_student.predict(
                    observation,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                observation, _, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        f"学生前置再生が早期終了した: {spec.seed}, {prefix_step + 1}"
                    )
            teacher.activate(environment, info)
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
        rows.append(
            {
                "seed": spec.seed,
                "start_runway_voxels": spec.start_runway_voxels,
                "activation_phase": teacher.activation_phase,
                "used_routed_controller": teacher.use_routed_controller,
                "steps": steps,
                "final_stage": stage,
                "success": success,
                "hard_fall": bool(info["hard_fall"]),
                "safe_stall": bool(not success and not info["hard_fall"]),
                "failure_reason": str(info["failure_reason"]),
                "raw_clearances": int(info["raw_clearances"]),
                "recovered_obstacles": int(info["recovered_obstacles"]),
            }
        )
    return {
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "safe_stall_count": sum(bool(row["safe_stall"]) for row in rows),
        "routed_episode_count": sum(bool(row["used_routed_controller"]) for row in rows),
        "rows": rows,
    }


def evaluate_gate(
    exact: Mapping[str, object],
    impulse: Mapping[str, object],
    reset: Mapping[str, object],
    requirements: Mapping[str, int],
) -> dict[str, object]:
    """位相二組と学生前置状態の成功・安全条件を同時に判定する。"""
    checks = {
        "phase_exact_success": int(exact["success_count"])
        >= requirements["minimum_phase_exact_success_count"],
        "phase_impulse_success": int(impulse["success_count"])
        >= requirements["minimum_phase_impulse_success_count"],
        "phase_exact_hard_fall": int(exact["hard_fall_count"])
        <= requirements["maximum_phase_exact_hard_fall_count"],
        "phase_impulse_hard_fall": int(impulse["hard_fall_count"])
        <= requirements["maximum_phase_impulse_hard_fall_count"],
        "reset_success": int(reset["success_count"])
        >= requirements["minimum_reset_state_success_count"],
        "reset_hard_fall": int(reset["hard_fall_count"])
        <= requirements["maximum_reset_state_hard_fall_count"],
    }
    return {
        "requirements": dict(requirements),
        "checks": checks,
        "gate_passed": all(checks.values()),
        "eligible_for_m2_4": all(checks.values()),
    }


def run(protocol: dict[str, object], output_dir: Path) -> dict[str, object]:
    """経路読込、三組評価、門判定、出典不変確認を実行する。"""
    from sb3_contrib import RecurrentPPO

    search = json.loads(
        Path(protocol["controller_search_summary_path"]).read_text(encoding="utf-8")
    )
    routes = route_configs_from_search(search)
    specs = load_phase_specs(Path(protocol["phase_reset_manifest_path"]))
    demo_manifest = load_rescue_demo_manifest()
    sources = {candidate.seed: candidate for candidate in demo_manifest.candidates}
    reset_manifest = load_rescue_reset_manifest(Path(protocol["reset_manifest_path"]))
    protected = (
        Path(protocol["source_path"]),
        Path(protocol["controller_search_summary_path"]),
        Path(protocol["phase_reset_manifest_path"]),
        Path(protocol["reset_manifest_path"]),
        Path(protocol["source_student_model_path"]),
        *(candidate.branch_path for candidate in demo_manifest.candidates),
    )
    before = {str(path): sha256_file(path) for path in protected}
    output_dir.mkdir(parents=True, exist_ok=False)
    exact = evaluate_phase_specs(routes, specs, sources, impulse_magnitude=0.0)
    impulse = evaluate_phase_specs(
        routes,
        specs,
        sources,
        impulse_magnitude=float(protocol["phase_impulse_magnitude"]),
    )
    frozen_student = RecurrentPPO.load(
        Path(protocol["source_student_model_path"]),
        device="cpu",
    )
    reset = evaluate_reset_states(routes, frozen_student, reset_manifest)
    gate = evaluate_gate(exact, impulse, reset, protocol["gate"])
    after = {str(path): sha256_file(path) for path in protected}
    if before != after:
        raise RuntimeError("M2.3.5d中に凍結出典が変更された。")
    result = {
        "method": "m2_3_5d_phase_and_position_routed_rescue_teacher",
        "stage": "hurdle_single",
        "split": "train",
        "run_name": output_dir.name,
        "protocol_sha256": protocol["sha256"],
        "teacher_training_only": True,
        "routes": {
            str(position): {
                "first_switch_fraction": route.first_switch_fraction,
                "post_clear_mode": route.post_clear_mode,
                "handoff_distance": route.handoff_distance,
                "clearance_blend": route.clearance_blend,
            }
            for position, route in routes.items()
        },
        "phase_exact_evaluation": exact,
        "phase_impulse_evaluation": impulse,
        "reset_state_evaluation": reset,
        "gate": gate,
        "checkpoint_disposition": (
            "m2_4_training_teacher_candidate"
            if bool(gate["eligible_for_m2_4"])
            else "quarantined_m2_3_5d_routed_teacher"
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
        description="位置と位相を分離した訓練専用救援教師を検収する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """M2.3.5dの三組教師検収を実行する。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
