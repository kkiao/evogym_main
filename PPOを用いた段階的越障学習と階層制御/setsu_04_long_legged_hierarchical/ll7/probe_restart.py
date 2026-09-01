"""第一障害物の安全着地後に再前進方策を決定論的に検査する。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_three_stage_recovery_env


def main():
    parser = argparse.ArgumentParser(description="诊断无侧倒落地后的再前进策略。")
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--clearance-model", required=True)
    parser.add_argument("--brake-model", required=True)
    parser.add_argument("--righting-model", required=True)
    parser.add_argument("--restart-model", required=True)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()

    approach = PPO.load(Path(args.approach_model), device="cpu")
    clearance = PPO.load(Path(args.clearance_model), device="cpu")
    brake = PPO.load(Path(args.brake_model), device="cpu")
    righting = PPO.load(Path(args.righting_model), device="cpu")
    restart = PPO.load(Path(args.restart_model), device="cpu")
    env = make_three_stage_recovery_env(
        make_body(),
        2,
        approach,
        clearance,
        0.95,
        args.steps,
        1_800,
        target_validated=1,
        brake_model=brake,
        brake_action_scale=0.55,
        brake_target_speed=0.10,
        brake_stable_steps=10,
        righting_prefix_model=righting,
        righting_prefix_action_scale=1.0,
        agent_action_scale=args.action_scale,
    )
    try:
        obs, info = env.reset(seed=10_000)
        maximum_angle = float(info["orientation_error"])
        contact_steps = 0
        for completed_steps in range(1, args.steps + 1):
            action, _ = restart.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            maximum_angle = max(
                maximum_angle,
                float(info["orientation_error"]),
                abs(float(info.get("unwrapped_orientation_error", 0.0))),
            )
            contact_steps += int(bool(info.get("upper_body_grounded", False)))
            if terminated or truncated or int(info["validated_obstacles"]) >= 1:
                break
        print(
            {
                "steps": completed_steps,
                "maximum_degrees": round(math.degrees(maximum_angle), 2),
                "contact_steps": contact_steps,
                "restart_successes": int(info["restart_successes"]),
                "validated_obstacles": int(info["validated_obstacles"]),
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
