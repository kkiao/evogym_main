"""第二障害物完全通過後の着地方策を動作縮小率ごとに検査する。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_true_noroll_second_landing_env


def main():
    parser = argparse.ArgumentParser(description="诊断第二墙落地动作尺度。")
    for name in (
        "approach_model",
        "first_clearance_model",
        "first_brake_model",
        "first_righting_model",
        "first_restart_model",
        "second_prefix_model",
        "second_prefix_model_2",
        "second_final_clearance_model",
        "landing_model",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    args = parser.parse_args()
    names = (
        "approach_model",
        "first_clearance_model",
        "first_brake_model",
        "first_righting_model",
        "first_restart_model",
        "second_prefix_model",
        "second_prefix_model_2",
        "second_final_clearance_model",
        "landing_model",
    )
    models = {
        name: PPO.load(Path(getattr(args, name)), device="cpu") for name in names
    }
    for scale in args.scales:
        env = make_true_noroll_second_landing_env(
            make_body(),
            approach_model=models["approach_model"],
            first_clearance_model=models["first_clearance_model"],
            first_brake_model=models["first_brake_model"],
            first_righting_model=models["first_righting_model"],
            first_restart_model=models["first_restart_model"],
            second_prefix_model=models["second_prefix_model"],
            second_prefix_model_2=models["second_prefix_model_2"],
            second_final_clearance_model=models["second_final_clearance_model"],
            agent_max_steps=500,
            prefix_max_steps=3_000,
            agent_action_scale=scale,
        )
        try:
            obs, info = env.reset(seed=10_000)
            maximum_angle = float(info["orientation_error"])
            contact_steps = 0
            for completed_steps in range(1, 501):
                action, _ = models["landing_model"].predict(obs, deterministic=True)
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
                    "scale": scale,
                    "steps": completed_steps,
                    "maximum_degrees": round(math.degrees(maximum_angle), 2),
                    "contact_steps": contact_steps,
                    "stable_landings": int(info["stable_landings"]),
                    "final_degrees": round(
                        math.degrees(float(info["orientation_error"])),
                        2,
                    ),
                    "speed": round(float(info["com_speed"]), 3),
                    "failure_reason": info["failure_reason"],
                }
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
