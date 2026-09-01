"""改良版Level 2胴体非接地階層制御器を全区間で厳格検証する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.curriculum import get_course
from ll7.experiment import make_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATHS = {
    "approach": MODELS_DIR / "approach.zip",
    "first_to_50": MODELS_DIR / "first_to_50.zip",
    "first_safe_to_full": MODELS_DIR / "first_safe_to_full.zip",
    "first_landing": MODELS_DIR / "first_landing.zip",
    "first_restart": MODELS_DIR / "first_restart.zip",
    "second_to_33": MODELS_DIR / "second_to_33.zip",
    "second_to_50": MODELS_DIR / "second_to_50.zip",
    "second_to_full": MODELS_DIR / "second_to_full.zip",
    "second_landing": MODELS_DIR / "second_landing.zip",
    "second_restart": MODELS_DIR / "second_restart.zip",
}

ACTION_SCALES = {
    "first_landing": 0.05,
    "first_restart": 0.75,
    "second_landing": 0.1,
    "second_restart": 0.75,
}


def load_models():
    """改良版制御鎖で使う全PPO方策をCPUへ読み込む。"""
    return {
        name: PPO.load(path.resolve(), device="cpu")
        for name, path in MODEL_PATHS.items()
    }


def select_stage(info, course, env, second_handoff_started, second_stage):
    """状態機械と通過割合から現在の階層方策名を選ぶ。"""
    active = int(info["active_obstacle"])
    phase = str(info["phase"])
    if active == 0:
        if phase == "landing":
            return "first_landing", second_handoff_started, second_stage
        if phase == "restart":
            return "first_restart", second_handoff_started, second_stage
        if float(info["x_position"]) < 0.95:
            return "first_approach", second_handoff_started, second_stage
        if float(info["maximum_crossed_fraction"]) < 0.5:
            return "first_to_50", second_handoff_started, second_stage
        return "first_safe_to_full", second_handoff_started, second_stage

    if int(info["strict_clearances"]) >= 2:
        stage = "second_landing" if phase == "landing" else "second_restart"
        return stage, second_handoff_started, second_stage

    obstacle_x = course.obstacles[1].start_x * env.unwrapped.VOXEL_SIZE
    if obstacle_x - float(info["x_position"]) <= 0.5:
        second_handoff_started = True
    fraction = float(info["maximum_crossed_fraction"])
    if fraction >= 0.5:
        second_stage = "second_to_full"
    elif fraction >= 1.0 / 3.0:
        second_stage = "second_to_50"
    stage = "second_approach" if not second_handoff_started else second_stage
    return stage, second_handoff_started, second_stage


def run_final(seed=10_000, render_mode=None, frame_skip=3):
    """二障害物の全フレームを再生し、姿勢と胴体接地を集計する。"""
    models = load_models()
    course = get_course(2)
    env = make_env(make_body(), 2, course.max_steps, render_mode=render_mode)
    quality = [
        {"maximum_degrees": 0.0, "contact_steps": 0}
        for _ in course.obstacles
    ]
    strict_events = []
    stage_events = []
    stage_quality = {}
    frames = []
    second_handoff_started = False
    second_stage = "second_to_33"
    try:
        obs, info = env.reset(seed=seed)
        for completed_steps in range(1, course.max_steps + 1):
            stage, second_handoff_started, second_stage = select_stage(
                info,
                course,
                env,
                second_handoff_started,
                second_stage,
            )
            if not stage_events or stage_events[-1]["stage"] != stage:
                stage_events.append({"step": completed_steps - 1, "stage": stage})
            if render_mode is not None and (completed_steps - 1) % frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    frames.append((completed_steps - 1, stage, frame, dict(info)))

            model_key = "approach" if stage in {"first_approach", "second_approach"} else stage
            action, _ = models[model_key].predict(obs, deterministic=True)
            action_scale = ACTION_SCALES.get(stage, 1.0)
            active_before = min(int(info["active_obstacle"]), 1)
            obs, _, terminated, truncated, info = env.step(action_scale * action)

            angle = max(
                float(info["orientation_error"]),
                abs(float(info.get("unwrapped_orientation_error", 0.0))),
            )
            quality[active_before]["maximum_degrees"] = max(
                quality[active_before]["maximum_degrees"],
                math.degrees(angle),
            )
            quality[active_before]["contact_steps"] += int(
                bool(info.get("upper_body_grounded", False))
            )
            stage_row = stage_quality.setdefault(
                stage,
                {"steps": 0, "maximum_degrees": 0.0, "contact_steps": 0},
            )
            stage_row["steps"] += 1
            stage_row["maximum_degrees"] = max(
                stage_row["maximum_degrees"],
                math.degrees(angle),
            )
            stage_row["contact_steps"] += int(
                bool(info.get("upper_body_grounded", False))
            )
            if info.get("new_strict_clearance", False):
                strict_events.append(
                    {
                        "step": completed_steps,
                        "count": int(info["strict_clearances"]),
                        "angle_degrees": math.degrees(float(info["orientation_error"])),
                        "speed": float(info["com_speed"]),
                    }
                )
            if terminated or truncated:
                break
        if render_mode is not None:
            frame = env.render()
            if frame is not None:
                frames.append((completed_steps, stage, frame, dict(info)))
    finally:
        env.close()

    success = bool(info["is_success"])
    optimized_stages = (
        "first_safe_to_full",
        "first_landing",
        "first_restart",
        "second_to_33",
        "second_to_50",
        "second_to_full",
        "second_landing",
        "second_restart",
    )
    optimized_rows = [stage_quality[name] for name in optimized_stages if name in stage_quality]
    optimized_contact_steps = sum(row["contact_steps"] for row in optimized_rows)
    optimized_maximum_degrees = max(row["maximum_degrees"] for row in optimized_rows)
    result = {
        "seed": seed,
        "steps": completed_steps,
        "strict_clearances": int(info["strict_clearances"]),
        "stable_landings": int(info["stable_landings"]),
        "restart_successes": int(info["restart_successes"]),
        "validated_obstacles": int(info["validated_obstacles"]),
        "success": success,
        "failure_reason": str(info["failure_reason"]),
        "quality": quality,
        "overall_maximum_degrees": max(item["maximum_degrees"] for item in quality),
        "overall_contact_steps": sum(item["contact_steps"] for item in quality),
        "true_no_side_fall": bool(
            success
            and all(item["contact_steps"] == 0 for item in quality)
            and all(item["maximum_degrees"] < 50.0 for item in quality)
        ),
        "optimized_stages": list(optimized_stages),
        "optimized_maximum_degrees": optimized_maximum_degrees,
        "optimized_contact_steps": optimized_contact_steps,
        "obstacle_control_no_side_fall": bool(
            success
            and optimized_contact_steps == 0
            and optimized_maximum_degrees < 50.0
        ),
        "strict_events": strict_events,
        "stage_events": stage_events,
        "stage_quality": stage_quality,
        "action_scales": ACTION_SCALES,
        "model_paths": {
            name: str(path.relative_to(PROJECT_ROOT))
            for name, path in MODEL_PATHS.items()
        },
    }
    return result, frames


def main():
    parser = argparse.ArgumentParser(
        description="改良版Level 2の姿勢制御チェーンを厳格評価する。"
    )
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    result, _ = run_final(seed=args.seed)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
