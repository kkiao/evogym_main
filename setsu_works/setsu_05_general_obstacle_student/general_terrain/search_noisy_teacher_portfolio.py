"""固定摂動下で安全成功する高さ一教師設定を位置別に探索する。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "noisy_teacher_search"
ROBUST_FLAT_MODEL = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_flat_runs"
    / "noisy_flat_seed7_v1"
    / "best_model.zip"
)
POSITIONS = tuple(range(20, 31))


def course(position: int, seed: int):
    """探索用の高さ一単壁コースを返す。"""
    return build_course(
        ["low_hurdle"],
        split="noisy_teacher_portfolio_search",
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )


def configurations() -> list[dict[str, object]]:
    """名義成功率の高い設定から順に有限探索表を構成する。"""
    rows = []
    fractions = (0.25, 0.33, 0.40, 0.20, 0.30, 0.35, 0.38, 0.45, 0.50)
    for post_clear_mode, handoffs in (
        ("restart_then_flat", (0.45, 0.40, 0.50, 0.35)),
        ("landing_then_restart", (0.45, 0.40, 0.50)),
    ):
        for handoff_distance, fraction in itertools.product(handoffs, fractions):
            rows.append(
                {
                    "post_clear_mode": post_clear_mode,
                    "handoff_distance": handoff_distance,
                    "first_switch_fraction": fraction,
                    "clearance_family": "first",
                    "robust_flat": False,
                }
            )
    for handoff_distance, fraction in itertools.product(
        (0.25, 0.30, 0.35, 0.40, 0.45),
        fractions,
    ):
        rows.append(
            {
                "post_clear_mode": "restart_then_flat",
                "handoff_distance": handoff_distance,
                "first_switch_fraction": fraction,
                "clearance_family": "first",
                "robust_flat": True,
            }
        )
    for handoff_distance, fraction in itertools.product((0.40, 0.45), fractions):
        rows.append(
            {
                "post_clear_mode": "restart_then_flat",
                "handoff_distance": handoff_distance,
                "first_switch_fraction": fraction,
                "clearance_family": "second",
                "robust_flat": False,
            }
        )
    return rows


def make_teacher(configuration: dict[str, object]) -> ClosedLoopHeight1Teacher:
    """一つの探索設定から閉ループ教師を生成する。"""
    return ClosedLoopHeight1Teacher(
        post_clear_mode=str(configuration["post_clear_mode"]),
        clearance_blend=1.0,
        handoff_distance=float(configuration["handoff_distance"]),
        adaptive_handoff=True,
        clearance_family=str(configuration["clearance_family"]),
        first_switch_fraction=float(configuration["first_switch_fraction"]),
        robust_flat_model_path=(
            ROBUST_FLAT_MODEL if bool(configuration["robust_flat"]) else None
        ),
    )


def run_episode(
    teacher: ClosedLoopHeight1Teacher | None,
    *,
    position: int,
    seed: int,
    noise_std: float,
    noise_probability: float,
    save_trajectory: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    """一設定を固定摂動で実行し安全性と必要なら軌跡を返す。"""
    environment = GeneralObstacleEnv(course=course(position, seed), resample_on_reset=False)
    rng = np.random.default_rng(seed + 5_000_000)
    observations = []
    actions = []
    stages = []
    disturbance_count = 0
    maximum_angle = 0.0
    upper_contact_steps = 0
    try:
        observation, info = environment.reset(seed=seed)
        if teacher is not None:
            teacher.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            if teacher is None:
                action = np.full(environment.action_space.shape, -0.2, dtype=np.float32)
                stage = "safe_stall"
            else:
                action, stage = teacher.predict(environment, info)
            if save_trajectory:
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                stages.append(stage)
            executed_action = np.asarray(action, dtype=np.float32)
            if rng.random() < noise_probability:
                executed_action = np.clip(
                    executed_action
                    + rng.normal(0.0, noise_std, size=executed_action.shape),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                disturbance_count += 1
            observation, _, terminated, truncated, info = environment.step(executed_action)
            steps += 1
            maximum_angle = max(maximum_angle, float(info["orientation_error"]))
            upper_contact_steps += int(bool(info["upper_body_grounded"]))
    finally:
        environment.close()
    result = {
        "position": position,
        "seed": seed,
        "steps": steps,
        "disturbance_count": disturbance_count,
        "course_complete": bool(info["course_complete"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_angle_degrees": math.degrees(maximum_angle),
        "upper_body_contact_steps": upper_contact_steps,
        "maximum_com_x": float(info["max_x_position"]),
    }
    trajectory = None
    if save_trajectory:
        trajectory = {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "stages": np.asarray(stages),
        }
    return result, trajectory


def main() -> None:
    """位置別成功教師を探索し九成功かつ零側倒の組合せを検収する。"""
    parser = argparse.ArgumentParser(description="搜索高度1扰动救回教师组合。")
    parser.add_argument("--run-name", default="noisy_teacher_portfolio_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    parser.add_argument("--target-successes", type=int, default=9)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "branches").mkdir()
    solutions: dict[int, dict[str, object]] = {}
    search_rows = []
    for configuration_index, configuration in enumerate(configurations()):
        if len(solutions) >= args.target_successes:
            break
        teacher = make_teacher(configuration)
        for position in POSITIONS:
            if position in solutions:
                continue
            seed = args.seed + 70_000 + (position - POSITIONS[0])
            result, _ = run_episode(
                teacher,
                position=position,
                seed=seed,
                noise_std=args.noise_std,
                noise_probability=args.noise_probability,
                save_trajectory=False,
            )
            search_rows.append(
                {
                    "configuration_index": configuration_index,
                    "configuration": configuration,
                    "result": result,
                }
            )
            if result["course_complete"] and not result["hard_fall"]:
                solutions[position] = {
                    "configuration_index": configuration_index,
                    "configuration": configuration,
                    "search_result": result,
                }
                print(
                    json.dumps(
                        {
                            "solved_position": position,
                            "configuration_index": configuration_index,
                            "solved_count": len(solutions),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    validation = []
    for position in POSITIONS:
        solution = solutions.get(position)
        teacher = make_teacher(solution["configuration"]) if solution else None
        seed = args.seed + 70_000 + (position - POSITIONS[0])
        result, trajectory = run_episode(
            teacher,
            position=position,
            seed=seed,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
            save_trajectory=solution is not None,
        )
        result["configuration_index"] = (
            int(solution["configuration_index"]) if solution else None
        )
        validation.append(result)
        if trajectory is not None and result["course_complete"] and not result["hard_fall"]:
            np.savez_compressed(
                output_dir / "branches" / f"x{position}_rescued.npz",
                **trajectory,
            )
    success_count = sum(item["course_complete"] for item in validation)
    hard_fall_count = sum(item["hard_fall"] for item in validation)
    summary = {
        "noise_std": args.noise_std,
        "noise_probability": args.noise_probability,
        "solutions": {str(key): value for key, value in solutions.items()},
        "validation": validation,
        "success_count": success_count,
        "hard_fall_count": hard_fall_count,
        "robustness_gate_passed": bool(
            success_count >= args.target_successes and hard_fall_count == 0
        ),
        "search_evaluations": len(search_rows),
    }
    (output_dir / "search_rows.json").write_text(
        json.dumps(search_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
