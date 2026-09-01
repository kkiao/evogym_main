"""Level 2最終無側倒階層制御器を統一回放して厳格指標を返す。"""

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
RUNS_DIR = PROJECT_ROOT / "runs"

MODEL_PATHS = {
    "approach": RUNS_DIR / "_smoke_legacy_exact_v3" / "initial_model.zip",
    "first_clearance": RUNS_DIR / "true_noroll_l2_wall1_clear_speed28_early095_seed7_v7" / "checkpoints" / "model_45000_steps.zip",
    "first_brake": RUNS_DIR / "true_noroll_l2_wall1_residual65_scale55_seed7_v15" / "checkpoints" / "model_50000_steps.zip",
    "first_righting": RUNS_DIR / "true_noroll_l2_wall1_righting60_full_seed7_v16" / "checkpoints" / "model_15000_steps.zip",
    "first_restart": RUNS_DIR / "level2_restart_after_strict_landing_seed7_v7" / "checkpoints" / "model_10000_steps.zip",
    "second_to_33": RUNS_DIR / "true_noroll_l2_wall2_fraction33_progress_seed7_v23" / "checkpoints" / "model_15000_steps.zip",
    "second_to_50": RUNS_DIR / "true_noroll_l2_wall2_prefix33_to50_seed7_v27" / "checkpoints" / "model_40000_steps.zip",
    "second_to_full": RUNS_DIR / "true_noroll_l2_wall2_prefix50_to100_probe_seed7_v29" / "initial_model.zip",
    "second_landing": RUNS_DIR / "quality_l2_second_brake_speed03_seed7_v27" / "checkpoints" / "model_30000_steps.zip",
    "second_restart": RUNS_DIR / "level2_second_relaxed_restart_seed7_v26" / "checkpoints" / "model_5000_steps.zip",
}


def load_models():
    """最終制御鎖の全PPO方策をCPUへ読み込む。"""
    return {
        name: PPO.load(path.resolve(), device="cpu")
        for name, path in MODEL_PATHS.items()
    }


def run_final(seed: int = 10_000, render_mode: str | None = None, frame_skip: int = 3):
    """最終制御鎖を一回再生し、段階別安全指標と任意フレームを返す。"""
    models = load_models()
    course = get_course(2)
    env = make_env(
        make_body(),
        2,
        course.max_steps,
        render_mode=render_mode,
    )
    first_landing_mode = "first_brake"
    brake_ready_steps = 0
    second_stage = "second_to_33"
    second_handoff_started = False
    quality = [
        {"maximum_degrees": 0.0, "contact_steps": 0}
        for _ in course.obstacles
    ]
    strict_events = []
    stage_events = []
    frames = []
    try:
        obs, info = env.reset(seed=seed)
        for completed_steps in range(1, course.max_steps + 1):
            active = int(info["active_obstacle"])
            if active == 0:
                if info["phase"] == "landing":
                    stage = first_landing_mode
                    action_scale = 0.55 if stage == "first_brake" else 1.0
                elif info["phase"] == "restart":
                    stage = "first_restart"
                    action_scale = 0.5
                elif float(info["x_position"]) < 0.95:
                    stage = "first_approach"
                    action_scale = 1.0
                else:
                    stage = "first_clearance"
                    action_scale = 1.0
            else:
                if int(info["strict_clearances"]) < 2:
                    obstacle_x = course.obstacles[1].start_x * env.unwrapped.VOXEL_SIZE
                    if obstacle_x - float(info["x_position"]) <= 0.25:
                        second_handoff_started = True
                    stage = (
                        "second_approach"
                        if not second_handoff_started
                        else second_stage
                    )
                    action_scale = 1.0
                elif info["phase"] == "landing":
                    stage = "second_landing"
                    action_scale = 0.1
                else:
                    stage = "second_restart"
                    action_scale = 0.75

            if not stage_events or stage_events[-1]["stage"] != stage:
                stage_events.append({"step": completed_steps - 1, "stage": stage})
            if render_mode is not None and (completed_steps - 1) % frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    frames.append((completed_steps - 1, stage, frame, dict(info)))

            model_key = (
                "approach"
                if stage in {"first_approach", "second_approach"}
                else stage
            )
            action, _ = models[model_key].predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action_scale * action)

            index = min(int(info["active_obstacle"]), 1)
            if info["phase"] in {"landing", "restart"}:
                angle = max(
                    float(info["orientation_error"]),
                    abs(float(info.get("unwrapped_orientation_error", 0.0))),
                )
                quality[index]["maximum_degrees"] = max(
                    quality[index]["maximum_degrees"],
                    math.degrees(angle),
                )
                quality[index]["contact_steps"] += int(
                    bool(info.get("upper_body_grounded", False))
                )
            if info.get("new_strict_clearance", False):
                strict_events.append(
                    {
                        "step": completed_steps,
                        "count": int(info["strict_clearances"]),
                        "angle_degrees": math.degrees(
                            float(info["orientation_error"])
                        ),
                        "speed": float(info["com_speed"]),
                    }
                )

            if first_landing_mode == "first_brake" and active == 0 and info["phase"] == "landing":
                ready = bool(
                    float(info["com_speed"]) <= 0.10
                    and float(info["restart_space_margin"]) >= 0.0
                    and not info.get("upper_body_grounded", False)
                )
                brake_ready_steps = brake_ready_steps + 1 if ready else 0
                if brake_ready_steps >= 10:
                    first_landing_mode = "first_righting"
            if active == 1 and int(info["strict_clearances"]) < 2:
                fraction = float(info["maximum_crossed_fraction"])
                if fraction >= 0.5:
                    second_stage = "second_to_full"
                elif fraction >= 1.0 / 3.0:
                    second_stage = "second_to_50"
            if terminated or truncated:
                break
        if render_mode is not None:
            frame = env.render()
            if frame is not None:
                frames.append((completed_steps, stage, frame, dict(info)))
    finally:
        env.close()

    result = {
        "seed": seed,
        "steps": completed_steps,
        "strict_clearances": int(info["strict_clearances"]),
        "stable_landings": int(info["stable_landings"]),
        "restart_successes": int(info["restart_successes"]),
        "validated_obstacles": int(info["validated_obstacles"]),
        "success": bool(info["is_success"]),
        "failure_reason": info["failure_reason"],
        "quality": quality,
        "overall_maximum_degrees": max(item["maximum_degrees"] for item in quality),
        "overall_contact_steps": sum(item["contact_steps"] for item in quality),
        "true_no_side_fall": bool(
            info["is_success"]
            and all(item["contact_steps"] == 0 for item in quality)
            and all(item["maximum_degrees"] < 70.0 for item in quality)
        ),
        "strict_events": strict_events,
        "stage_events": stage_events,
        "model_paths": {name: str(path.resolve()) for name, path in MODEL_PATHS.items()},
    }
    return result, frames


def main():
    parser = argparse.ArgumentParser(description="统一验收最终Level 2无侧倒控制链。")
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output")
    args = parser.parse_args()
    result, _ = run_final(seed=args.seed)
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
