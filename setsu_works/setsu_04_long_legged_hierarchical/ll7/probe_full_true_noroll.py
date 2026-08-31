"""新しい第一障害物制御鎖と第二障害物方策を完全なLevel 2で検査する。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.curriculum import get_course
from ll7.experiment import make_env


def load(path: str):
    """CPU上へ決定論的評価用PPO方策を読み込む。"""
    return PPO.load(Path(path), device="cpu")


def main():
    parser = argparse.ArgumentParser(description="验收完整Level 2无侧倒控制链。")
    for name in (
        "approach_model",
        "first_clearance_model",
        "first_brake_model",
        "first_righting_model",
        "first_restart_model",
        "second_clearance_model",
        "second_landing_model",
        "second_restart_model",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--max-steps", type=int, default=1_850)
    args = parser.parse_args()

    models = {name: load(getattr(args, name)) for name in vars(args) if name.endswith("_model")}
    course = get_course(2)
    env = make_env(make_body(), 2, course.max_steps)
    quality = [
        {"maximum_degrees": 0.0, "contact_steps": 0}
        for _ in course.obstacles
    ]
    brake_ready_steps = 0
    first_landing_mode = "brake"
    try:
        obs, info = env.reset(seed=10_000)
        for completed_steps in range(1, args.max_steps + 1):
            active = int(info["active_obstacle"])
            if active == 0:
                if info["phase"] == "landing":
                    if first_landing_mode == "brake":
                        model = models["first_brake_model"]
                        action_scale = 0.55
                    else:
                        model = models["first_righting_model"]
                        action_scale = 1.0
                elif info["phase"] == "restart":
                    model = models["first_restart_model"]
                    action_scale = 0.5
                else:
                    model = (
                        models["approach_model"]
                        if float(info["x_position"]) < 0.95
                        else models["first_clearance_model"]
                    )
                    action_scale = 1.0
            else:
                if int(info["strict_clearances"]) < 2:
                    obstacle_x = course.obstacles[1].start_x * env.unwrapped.VOXEL_SIZE
                    model = (
                        models["approach_model"]
                        if obstacle_x - float(info["x_position"]) > 0.25
                        else models["second_clearance_model"]
                    )
                elif info["phase"] == "landing":
                    model = models["second_landing_model"]
                else:
                    model = models["second_restart_model"]
                action_scale = 1.0
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action_scale * action)

            index = min(int(info["active_obstacle"]), len(quality) - 1)
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

            if active == 0 and first_landing_mode == "brake" and info["phase"] == "landing":
                ready = bool(
                    float(info["com_speed"]) <= 0.10
                    and float(info["restart_space_margin"]) >= 0.0
                    and not info.get("upper_body_grounded", False)
                )
                brake_ready_steps = brake_ready_steps + 1 if ready else 0
                if brake_ready_steps >= 10:
                    first_landing_mode = "righting"
            if terminated or truncated:
                break
        print(
            {
                "steps": completed_steps,
                "strict_clearances": int(info["strict_clearances"]),
                "stable_landings": int(info["stable_landings"]),
                "restart_successes": int(info["restart_successes"]),
                "validated_obstacles": int(info["validated_obstacles"]),
                "success": bool(info["is_success"]),
                "failure_reason": info["failure_reason"],
                "phase": info["phase"],
                "x_position": round(float(info["x_position"]), 3),
                "max_x_position": round(float(info["max_x_position"]), 3),
                "quality": quality,
                "final_degrees": round(
                    math.degrees(float(info["orientation_error"])),
                    2,
                ),
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
