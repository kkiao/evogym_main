"""第二障害物の安全着地後に最終再前進方策を縮小率ごとに検査する。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_true_noroll_second_restart_env


def main():
    parser = argparse.ArgumentParser(description="诊断第二墙后的最终再前进。")
    names = (
        "approach_model",
        "first_clearance_model",
        "first_brake_model",
        "first_righting_model",
        "first_restart_model",
        "second_prefix_model",
        "second_prefix_model_2",
        "second_final_clearance_model",
        "second_landing_model",
        "restart_model",
    )
    for name in names:
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    args = parser.parse_args()
    models = {
        name: PPO.load(Path(getattr(args, name)), device="cpu") for name in names
    }
    for scale in args.scales:
        env = make_true_noroll_second_restart_env(
            make_body(),
            approach_model=models["approach_model"],
            first_clearance_model=models["first_clearance_model"],
            first_brake_model=models["first_brake_model"],
            first_righting_model=models["first_righting_model"],
            first_restart_model=models["first_restart_model"],
            second_prefix_model=models["second_prefix_model"],
            second_prefix_model_2=models["second_prefix_model_2"],
            second_final_clearance_model=models["second_final_clearance_model"],
            second_landing_model=models["second_landing_model"],
            agent_max_steps=600,
            prefix_max_steps=3_000,
            second_landing_action_scale=0.1,
            agent_action_scale=scale,
        )
        try:
            obs, info = env.reset(seed=10_000)
            maximum_angle = float(info["orientation_error"])
            contact_steps = 0
            for completed_steps in range(1, 601):
                action, _ = models["restart_model"].predict(obs, deterministic=True)
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
                    "restart_successes": int(info["restart_successes"]),
                    "validated_obstacles": int(info["validated_obstacles"]),
                    "success": bool(info["is_success"]),
                    "final_degrees": round(
                        math.degrees(float(info["orientation_error"])),
                        2,
                    ),
                    "failure_reason": info["failure_reason"],
                }
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
