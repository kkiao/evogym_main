"""一つのPPO方策を厳格状態機械の各条件まで詳しく診断する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.curriculum import get_course
from ll7.environment import (
    LANDING_ANGLE_LIMIT,
    LANDING_ANGULAR_SPEED_LIMIT,
    LANDING_SPEED_LIMIT,
)
from ll7.experiment import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="诊断一个长腿越障策略。")
    parser.add_argument("--model", required=True)
    parser.add_argument("--approach-model")
    parser.add_argument("--clearance-model")
    parser.add_argument("--landing-model")
    parser.add_argument("--restart-model")
    parser.add_argument("--next-clearance-model")
    parser.add_argument("--next-landing-model")
    parser.add_argument("--prefix-validated", type=int, default=1)
    parser.add_argument("--prefix-clearances", type=int, default=2)
    parser.add_argument("--handoff-distance", type=float, default=0.25)
    parser.add_argument("--handoff-x", type=float)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--landing-angle-limit-degrees", type=float, default=35.0)
    parser.add_argument("--landing-speed-limit", type=float, default=0.15)
    parser.add_argument("--no-side-fall-angle-degrees", type=float, default=60.0)
    parser.add_argument("--landing-neutral-blend", type=float, default=0.0)
    parser.add_argument("--stop-after-stable-landings", type=int)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.landing_neutral_blend <= 1.0:
        raise ValueError("--landing-neutral-blend 必须在0到1之间。")
    model = PPO.load(Path(args.model), device="cpu")
    approach_model = (
        PPO.load(Path(args.approach_model), device="cpu")
        if args.approach_model
        else None
    )
    clearance_model = (
        PPO.load(Path(args.clearance_model), device="cpu")
        if args.clearance_model
        else None
    )
    landing_model = (
        PPO.load(Path(args.landing_model), device="cpu")
        if args.landing_model
        else None
    )
    restart_model = (
        PPO.load(Path(args.restart_model), device="cpu")
        if args.restart_model
        else None
    )
    next_clearance_model = (
        PPO.load(Path(args.next_clearance_model), device="cpu")
        if args.next_clearance_model
        else None
    )
    next_landing_model = (
        PPO.load(Path(args.next_landing_model), device="cpu")
        if args.next_landing_model
        else None
    )
    course = get_course(args.level)
    angle_limit = math.radians(args.landing_angle_limit_degrees)
    use_post_prefix_thresholds = next_clearance_model is not None
    env = make_env(
        make_body(),
        args.level,
        course.max_steps,
        landing_angle_limit=(LANDING_ANGLE_LIMIT if use_post_prefix_thresholds else angle_limit),
        landing_speed_limit=(LANDING_SPEED_LIMIT if use_post_prefix_thresholds else args.landing_speed_limit),
    )
    obs, info = env.reset(seed=args.seed)
    snapshots = []
    strict_events = []
    best_stable_steps = 0
    maximum_orientation_error = 0.0
    maximum_unwrapped_orientation_error = 0.0
    post_clearance_quality = [
        {
            "obstacle": index + 1,
            "maximum_orientation_error": 0.0,
            "maximum_unwrapped_orientation_error": 0.0,
            "upper_body_ground_contact_steps": 0,
        }
        for index in range(len(course.obstacles))
    ]
    clear_step = None
    completed_steps = 0
    try:
        for completed_steps in range(1, course.max_steps + 1):
            if (
                use_post_prefix_thresholds
                and int(info["validated_obstacles"]) >= args.prefix_validated
            ):
                env.unwrapped.landing_angle_limit = angle_limit
                env.unwrapped.landing_speed_limit = args.landing_speed_limit
            if (
                next_clearance_model is not None
                and int(info["validated_obstacles"]) >= args.prefix_validated
            ):
                if int(info["strict_clearances"]) < args.prefix_clearances:
                    active_model = next_clearance_model
                elif (
                    next_landing_model is not None
                    and int(info["stable_landings"]) < args.prefix_clearances
                ):
                    active_model = next_landing_model
                else:
                    active_model = model
            elif info["phase"] == "landing" and landing_model is not None:
                active_model = landing_model
            elif info["phase"] == "restart" and restart_model is not None:
                active_model = restart_model
            elif info["phase"] != "approach":
                active_model = model
            elif (
                approach_model is not None
                and (
                    (
                        args.handoff_x is not None
                        and float(info["x_position"]) < args.handoff_x
                    )
                    or (
                        args.handoff_x is None
                        and info["active_obstacle"] < len(course.obstacles)
                        and course.obstacles[info["active_obstacle"]].start_x * 0.1
                        - float(info["x_position"])
                        > args.handoff_distance
                    )
                )
            ):
                active_model = approach_model
            elif clearance_model is not None:
                active_model = clearance_model
            else:
                active_model = model
            action, _ = active_model.predict(obs, deterministic=True)
            if info["phase"] == "landing" and args.landing_neutral_blend > 0.0:
                # 着地中だけ中立動作へ混合し、安全な制動範囲を探索する。
                action = (1.0 - args.landing_neutral_blend) * np.asarray(action)
            obs, _, terminated, truncated, info = env.step(action)
            maximum_orientation_error = max(
                maximum_orientation_error,
                float(info["orientation_error"]),
            )
            maximum_unwrapped_orientation_error = max(
                maximum_unwrapped_orientation_error,
                abs(float(info.get("unwrapped_orientation_error", 0.0))),
            )
            active_index = int(info["active_obstacle"])
            if (
                info["phase"] in {"landing", "restart"}
                and active_index < len(post_clearance_quality)
            ):
                quality = post_clearance_quality[active_index]
                quality["maximum_orientation_error"] = max(
                    quality["maximum_orientation_error"],
                    float(info["orientation_error"]),
                )
                quality["maximum_unwrapped_orientation_error"] = max(
                    quality["maximum_unwrapped_orientation_error"],
                    abs(float(info.get("unwrapped_orientation_error", 0.0))),
                )
                if info.get("upper_body_grounded", False):
                    quality["upper_body_ground_contact_steps"] += 1
            if clear_step is None and info["strict_clearances"]:
                clear_step = completed_steps
            if info.get("new_strict_clearance", False):
                strict_events.append(
                    {
                        "step": completed_steps,
                        "count": int(info["strict_clearances"]),
                        "angle_deg": float(math.degrees(info["orientation_error"])),
                        "speed": float(info["com_speed"]),
                    }
                )
            if completed_steps % 25 == 0 or (
                clear_step is not None and completed_steps == clear_step
            ) or info.get("new_strict_clearance", False):
                snapshots.append(
                    {
                        "step": completed_steps,
                        "phase": info["phase"],
                        "x": info["x_position"],
                        "orientation_error": info["orientation_error"],
                        "angular_speed": info["angular_speed"],
                        "com_speed": info["com_speed"],
                        "bottom_y": info["bottom_y"],
                        "body_height": info["body_height"],
                        "stable_steps": info["landing_stable_steps"],
                    }
                )
            if clear_step is not None:
                best_stable_steps = max(best_stable_steps, info["landing_stable_steps"])
            if terminated or truncated:
                break
            if (
                args.stop_after_stable_landings is not None
                and int(info["stable_landings"])
                >= args.stop_after_stable_landings
            ):
                break
    finally:
        env.close()

    no_side_fall_limit = math.radians(args.no_side_fall_angle_degrees)
    for quality in post_clearance_quality:
        quality["maximum_orientation_degrees"] = math.degrees(
            quality["maximum_orientation_error"]
        )
        quality["maximum_unwrapped_orientation_degrees"] = math.degrees(
            quality["maximum_unwrapped_orientation_error"]
        )
        quality["passed"] = bool(
            max(
                quality["maximum_orientation_error"],
                quality["maximum_unwrapped_orientation_error"],
            )
            <= no_side_fall_limit
            and quality["upper_body_ground_contact_steps"] == 0
        )
    true_no_side_fall_success = bool(
        info["is_success"]
        and post_clearance_quality
        and all(item["passed"] for item in post_clearance_quality)
    )
    result = {
        "model": str(Path(args.model).resolve()),
        "approach_model": (
            str(Path(args.approach_model).resolve()) if args.approach_model else None
        ),
        "clearance_model": (
            str(Path(args.clearance_model).resolve()) if args.clearance_model else None
        ),
        "handoff_x": args.handoff_x,
        "level": args.level,
        "completed_steps": completed_steps,
        "clear_step": clear_step,
        "best_stable_steps": best_stable_steps,
        "maximum_orientation_error": maximum_orientation_error,
        "maximum_orientation_degrees": math.degrees(maximum_orientation_error),
        "maximum_unwrapped_orientation_degrees": math.degrees(
            maximum_unwrapped_orientation_error
        ),
        "no_side_fall_angle_limit_degrees": args.no_side_fall_angle_degrees,
        "landing_neutral_blend": args.landing_neutral_blend,
        "post_clearance_quality": post_clearance_quality,
        "true_no_side_fall_success": true_no_side_fall_success,
        "strict_events": strict_events,
        "limits": {
            "orientation_error": angle_limit,
            "angular_speed": LANDING_ANGULAR_SPEED_LIMIT,
            "com_speed": LANDING_SPEED_LIMIT,
            "training_com_speed": args.landing_speed_limit,
        },
        "final_info": info,
        "snapshots": snapshots,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
