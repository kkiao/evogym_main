"""重要段階へ単発摂動を加え教師が安全回復した軌跡だけを分岐庫へ保存する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRANCH_ROOT = PROJECT_ROOT / "runs" / "height1_teacher_recovery_branches"
POSITIONS = tuple(range(20, 31))
TARGETS = (
    "approach_far",
    "approach_near",
    "cross_early",
    "cross_late",
    "recovery",
)
DIRECTIONS = (
    np.asarray([1.0, -1.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32),
    np.asarray([-0.5, 1.0, -1.0, 1.0, -0.5, 0.5], dtype=np.float32),
    np.asarray([1.0, 0.5, -1.0, -0.5, 0.5, -1.0], dtype=np.float32),
    np.asarray([-1.0, -0.5, 0.5, 1.0, -1.0, 0.5], dtype=np.float32),
)


def relative_wall_distance(environment: GeneralObstacleEnv) -> float:
    """現在質心から最初の壁前端までの符号付き相対距離を返す。"""
    base = environment.unwrapped
    positions = base.object_pos_at_time(base.get_time(), "robot")
    obstacle_start = base.course.obstacles[0].start_x * base.VOXEL_SIZE
    return float(obstacle_start - np.mean(positions[0]))


def should_inject(
    target: str,
    *,
    stage: str,
    stage_age: int,
    distance: float,
    info: dict[str, object],
) -> bool:
    """各回復能力を分離して測れる一回限りの注入時点を判定する。"""
    stage_name = stage.split(":", maxsplit=1)[-1]
    if target == "approach_far":
        return stage_name == "flat" and 1.15 <= distance <= 1.25
    if target == "approach_near":
        return stage_name == "flat" and 0.52 <= distance <= 0.62
    if target == "cross_early":
        return stage_name == "first_to_50" and stage_age == 12
    if target == "cross_late":
        return (
            stage_name in {"first_safe_to_full", "half_recovery"}
            and stage_age == 8
        )
    if target == "recovery":
        return (
            int(info["raw_clearances"]) >= 1
            and stage_name in {"raw_recovery", "half_recovery", "first_restart", "flat"}
            and stage_age == 12
        )
    raise ValueError(f"未知の摂動段階：{target}")


def run_branch(
    teacher: PortfolioHeight1Teacher,
    *,
    position: int,
    seed: int,
    target: str,
    magnitude: float,
    direction: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """一度だけ動作を乱し以後教師へ戻して全軌跡と厳格結果を返す。"""
    course = build_course(
        ["low_hurdle"],
        split="teacher_recovery_branch",
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    observations = []
    actions = []
    injected = False
    injection_step: int | None = None
    stage_age = 0
    previous_stage = ""
    maximum_angle = 0.0
    upper_contact_steps = 0
    try:
        observation, info = environment.reset(seed=seed)
        teacher.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            teacher_action, stage = teacher.predict(environment, observation, info)
            stage_age = stage_age + 1 if stage == previous_stage else 0
            previous_stage = stage
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(np.asarray(teacher_action, dtype=np.float32))
            executed_action = teacher_action
            if not injected and should_inject(
                target,
                stage=stage,
                stage_age=stage_age,
                distance=relative_wall_distance(environment),
                info=info,
            ):
                executed_action = np.clip(
                    teacher_action + magnitude * direction,
                    -1.0,
                    1.0,
                ).astype(np.float32)
                injected = True
                injection_step = steps
            observation, _, terminated, truncated, info = environment.step(
                executed_action
            )
            steps += 1
            maximum_angle = max(maximum_angle, float(info["orientation_error"]))
            upper_contact_steps += int(bool(info["upper_body_grounded"]))
    finally:
        environment.close()
    accepted = bool(
        injected
        and info["course_complete"]
        and not info["hard_fall"]
        and upper_contact_steps == 0
        and maximum_angle <= math.radians(45.0)
    )
    result = {
        "position": position,
        "seed": seed,
        "target": target,
        "magnitude": magnitude,
        "direction": direction.tolist(),
        "injected": injected,
        "injection_step": injection_step,
        "steps": steps,
        "accepted": accepted,
        "course_complete": bool(info["course_complete"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_angle_degrees": math.degrees(maximum_angle),
        "upper_body_contact_steps": upper_contact_steps,
        "maximum_com_x": float(info["max_x_position"]),
    }
    return (
        result,
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def main() -> None:
    """全候補を検査し合格分岐配列、索引、位置別被覆率を保存する。"""
    parser = argparse.ArgumentParser(description="生成教师成功救回的扰动分支。")
    parser.add_argument("--run-name", default="height1_recovery_branches_seed7_v1")
    parser.add_argument("--positions", nargs="+", type=int, default=list(POSITIONS))
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[0.005, 0.01])
    parser.add_argument("--direction-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--flat-model")
    args = parser.parse_args()
    output_dir = BRANCH_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "branches").mkdir()
    teacher = PortfolioHeight1Teacher(
        flat_model_path=Path(args.flat_model) if args.flat_model else None
    )
    rows = []
    branch_index = []
    candidate_index = 0
    for position in args.positions:
        for target in TARGETS:
            for magnitude in args.magnitudes:
                for direction_index, direction in enumerate(
                    DIRECTIONS[: args.direction_count]
                ):
                    seed = args.seed + 100_000 + candidate_index
                    result, observations, actions = run_branch(
                        teacher,
                        position=position,
                        seed=seed,
                        target=target,
                        magnitude=magnitude,
                        direction=direction,
                    )
                    result["direction_index"] = direction_index
                    rows.append(result)
                    if result["accepted"]:
                        name = (
                            f"x{position}_{target}_m{magnitude:.3f}"
                            f"_d{direction_index}.npz"
                        )
                        np.savez_compressed(
                            output_dir / "branches" / name,
                            observations=observations,
                            actions=actions,
                            injection_step=np.asarray(
                                [result["injection_step"]],
                                dtype=np.int32,
                            ),
                        )
                        branch_index.append({**result, "file": name})
                    candidate_index += 1
                    print(
                        json.dumps(
                            {
                                "position": position,
                                "target": target,
                                "magnitude": magnitude,
                                "direction": direction_index,
                                "accepted": result["accepted"],
                                "reason": result["failure_reason"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    coverage = {
        str(position): {
            target: sum(
                1
                for item in branch_index
                if item["position"] == position and item["target"] == target
            )
            for target in TARGETS
        }
        for position in args.positions
    }
    result = {
        "candidates": len(rows),
        "accepted_branches": len(branch_index),
        "acceptance_rate": len(branch_index) / max(1, len(rows)),
        "targets": list(TARGETS),
        "positions": args.positions,
        "coverage": coverage,
        "branch_index": branch_index,
        "episodes": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidates": result["candidates"],
                "accepted_branches": result["accepted_branches"],
                "acceptance_rate": result["acceptance_rate"],
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
