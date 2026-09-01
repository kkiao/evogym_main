"""九十五次元回復方策を含む統一高さ一教師を全位置で検収する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import build_course
from general_terrain.train_recovery_teacher import safe_flat_handoff


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOVERY_MODEL = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "recovery_runs"
    / "_smoke_recovery_teacher_seed7_v1"
    / "best_model.zip"
)


def run_episode(
    recovery_model: PPO,
    *,
    position: int,
    seed: int,
    output_gif: Path | None,
) -> dict[str, object]:
    """一位置で統一教師の全制御段階を最後まで実行する。"""
    course = build_course(
        ["low_hurdle"],
        split="unified_teacher_validation",
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array" if output_gif is not None else None,
    )
    prefix_teacher = ClosedLoopHeight1Teacher(
        post_clear_mode="restart_then_flat",
        clearance_blend=1.0,
        handoff_distance=0.45,
        adaptive_handoff=True,
    )
    frames: list[np.ndarray] = []
    phase = "prefix"
    phase_events = [{"step": 0, "phase": phase}]
    safe_streak = 0
    maximum_angle = 0.0
    upper_contact_steps = 0
    try:
        observation, info = environment.reset(seed=seed)
        prefix_teacher.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            if phase == "prefix":
                action, _ = prefix_teacher.predict(environment, info)
            elif phase == "recovery":
                action, _ = recovery_model.predict(observation, deterministic=True)
            else:
                action = prefix_teacher.predict_flat(environment)
            observation, _, terminated, truncated, info = environment.step(action)
            steps += 1
            maximum_angle = max(maximum_angle, float(info["orientation_error"]))
            upper_contact_steps += int(bool(info["upper_body_grounded"]))

            if phase == "prefix" and int(info["raw_clearances"]) >= 1:
                phase = "recovery"
                phase_events.append({"step": steps, "phase": phase})
            elif phase == "recovery":
                ready = safe_flat_handoff(environment, info)
                safe_streak = safe_streak + 1 if ready else 0
                if safe_streak >= 20:
                    phase = "flat_finish"
                    phase_events.append({"step": steps, "phase": phase})

            if output_gif is not None and steps % 5 == 0:
                frame = environment.render()
                if frame is not None:
                    frames.append(np.asarray(frame))
    finally:
        environment.close()
    if output_gif is not None and frames:
        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, frames, fps=12, loop=0)
    return {
        "position": position,
        "seed": seed,
        "steps": steps,
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "course_complete": bool(info["course_complete"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_angle_degrees": math.degrees(maximum_angle),
        "upper_body_contact_steps": upper_contact_steps,
        "maximum_com_x": float(info["max_x_position"]),
        "phase_events": phase_events,
        "gif": str(output_gif.resolve()) if output_gif is not None else None,
    }


def main() -> None:
    """二十格から三十格まで全十一位置の厳格合格率を保存する。"""
    parser = argparse.ArgumentParser(description="验收统一95维闭环矮墙教师。")
    parser.add_argument("--model", default=str(DEFAULT_RECOVERY_MODEL))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "training_only_teacher" / "generated" / "unified_height1_v1"),
    )
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recovery_model = PPO.load(Path(args.model), device="cpu")
    rows = []
    for index, position in enumerate(range(20, 31)):
        gif = output_dir / "gifs" / f"teacher_x{position}.gif" if args.render else None
        rows.append(
            run_episode(
                recovery_model,
                position=position,
                seed=70_000 + index,
                output_gif=gif,
            )
        )
    result = {
        "teacher": "relative_phase_prefix_plus_95d_recovery_plus_flat_finish",
        "recovery_model": str(Path(args.model).resolve()),
        "positions": list(range(20, 31)),
        "episodes": rows,
        "success_rate": float(np.mean([row["course_complete"] for row in rows])),
        "hard_fall_rate": float(np.mean([row["hard_fall"] for row in rows])),
        "all_positions_passed": all(row["course_complete"] for row in rows),
    }
    (output_dir / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
