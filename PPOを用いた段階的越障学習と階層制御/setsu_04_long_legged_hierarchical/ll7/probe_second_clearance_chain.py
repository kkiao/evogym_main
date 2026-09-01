"""第二障害物の段階通過制御鎖を完全通過まで厳格に検査する。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_true_noroll_second_crossing_env


def main():
    parser = argparse.ArgumentParser(description="验收第二墙分段完整越障链。")
    for name in (
        "approach_model",
        "first_clearance_model",
        "first_brake_model",
        "first_righting_model",
        "first_restart_model",
        "second_prefix_model",
        "second_prefix_model_2",
        "second_final_model",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args()
    model_names = (
        "approach_model",
        "first_clearance_model",
        "first_brake_model",
        "first_righting_model",
        "first_restart_model",
        "second_prefix_model",
        "second_prefix_model_2",
        "second_final_model",
    )
    models = {
        name: PPO.load(Path(getattr(args, name)), device="cpu")
        for name in model_names
    }
    env = make_true_noroll_second_crossing_env(
        make_body(),
        approach_model=models["approach_model"],
        first_clearance_model=models["first_clearance_model"],
        first_brake_model=models["first_brake_model"],
        first_righting_model=models["first_righting_model"],
        first_restart_model=models["first_restart_model"],
        target_fraction=1.0,
        agent_max_steps=500,
        prefix_max_steps=3_000,
        second_prefix_model=models["second_prefix_model"],
        second_prefix_fraction=1.0 / 3.0,
        second_prefix_model_2=models["second_prefix_model_2"],
        second_prefix_fraction_2=0.5,
    )
    try:
        obs, info = env.reset(seed=10_000)
        maximum_angle = float(info.get("second_prefix_maximum_orientation", 0.0))
        contact_steps = int(info.get("second_prefix_contact_steps", 0))
        for completed_steps in range(1, 501):
            action, _ = models["second_final_model"].predict(
                obs,
                deterministic=True,
            )
            obs, _, terminated, truncated, info = env.step(action)
            maximum_angle = max(
                maximum_angle,
                float(info["orientation_error"]),
                abs(float(info.get("unwrapped_orientation_error", 0.0))),
            )
            contact_steps += int(bool(info.get("upper_body_grounded", False)))
            if terminated or truncated:
                break
        print(
            {
                "steps_after_50_percent": completed_steps,
                "strict_clearances": int(info["strict_clearances"]),
                "fraction": float(info["maximum_crossed_fraction"]),
                "maximum_degrees": round(math.degrees(maximum_angle), 2),
                "contact_steps": contact_steps,
                "clearance_degrees": round(
                    math.degrees(float(info["orientation_error"])),
                    2,
                ),
                "clearance_speed": round(float(info["com_speed"]), 3),
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
