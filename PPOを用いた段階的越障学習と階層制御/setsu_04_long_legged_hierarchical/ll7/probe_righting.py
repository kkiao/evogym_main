"""安全制動後の扶正方策を複数の動作縮小率で決定論的に比較する。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_three_stage_recovery_env


def main():
    parser = argparse.ArgumentParser(description="比较扶正阶段动作尺度。")
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--clearance-model", required=True)
    parser.add_argument("--brake-model", required=True)
    parser.add_argument("--righting-model", required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--handoff-x", type=float, default=0.95)
    parser.add_argument("--brake-action-scale", type=float, default=0.55)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    approach = PPO.load(Path(args.approach_model), device="cpu")
    clearance = PPO.load(Path(args.clearance_model), device="cpu")
    brake = PPO.load(Path(args.brake_model), device="cpu")
    righting = PPO.load(Path(args.righting_model), device="cpu")
    for scale in args.scales:
        env = make_three_stage_recovery_env(
            make_body(),
            2,
            approach,
            clearance,
            args.handoff_x,
            args.steps,
            1_800,
            brake_model=brake,
            brake_action_scale=args.brake_action_scale,
            brake_target_speed=0.10,
            brake_stable_steps=10,
            agent_action_scale=scale,
        )
        try:
            obs, info = env.reset(seed=10_000)
            maximum_angle = max(
                float(info["orientation_error"]),
                float(info.get("brake_prefix_maximum_orientation", 0.0)),
            )
            contact_steps = int(info.get("brake_prefix_contact_steps", 0))
            for completed_steps in range(1, args.steps + 1):
                action, _ = righting.predict(obs, deterministic=True)
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
                    "restart_margin": round(
                        float(info["restart_space_margin"]),
                        3,
                    ),
                }
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
