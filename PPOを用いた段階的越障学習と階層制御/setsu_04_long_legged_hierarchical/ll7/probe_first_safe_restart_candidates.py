"""第一障害物の安全着地後に既存再前進方策を比較する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_first_safe_restart_env


def main():
    parser = argparse.ArgumentParser(description="比较安全落地后的再前进候选。")
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--clearance-model", required=True)
    parser.add_argument("--safe-clearance-model", required=True)
    parser.add_argument("--landing-model", required=True)
    parser.add_argument("--landing-scale", type=float, default=0.05)
    parser.add_argument("--candidate-model", action="append", required=True)
    parser.add_argument("--scales", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--prefix-max-steps", type=int, default=2_000)
    parser.add_argument("--output", default="analysis/first_safe_restart_candidates.json")
    args = parser.parse_args()

    approach = PPO.load(Path(args.approach_model).resolve(), device="cpu")
    clearance = PPO.load(Path(args.clearance_model).resolve(), device="cpu")
    safe_clearance = PPO.load(Path(args.safe_clearance_model).resolve(), device="cpu")
    landing = PPO.load(Path(args.landing_model).resolve(), device="cpu")
    results = []
    for candidate_path in args.candidate_model:
        model = PPO.load(Path(candidate_path).resolve(), device="cpu")
        for scale in [float(value) for value in args.scales.split(",")]:
            env = make_first_safe_restart_env(
                make_body(),
                approach_model=approach,
                clearance_model=clearance,
                safe_clearance_model=safe_clearance,
                landing_model=landing,
                prefix_fraction=0.5,
                agent_max_steps=args.max_steps,
                prefix_max_steps=args.prefix_max_steps,
                landing_action_scale=args.landing_scale,
                agent_action_scale=scale,
                max_orientation=math.radians(50.0),
                preferred_orientation=math.radians(15.0),
            )
            try:
                obs, info = env.reset(seed=10_000)
                maximum_angle = float(info["orientation_error"])
                contact_steps = 0
                for step in range(1, args.max_steps + 1):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    maximum_angle = max(
                        maximum_angle,
                        float(info["orientation_error"]),
                        float(info.get("unwrapped_orientation_error", 0.0)),
                    )
                    contact_steps += int(bool(info.get("upper_body_grounded", False)))
                    if terminated or truncated:
                        break
            finally:
                env.close()
            result = {
                "model": str(Path(candidate_path)),
                "scale": scale,
                "steps": step,
                "validated_obstacles": int(info["validated_obstacles"]),
                "restart_successes": int(info["restart_successes"]),
                "maximum_degrees": math.degrees(maximum_angle),
                "body_contact_steps": contact_steps,
                "recovery_progress": float(info["recovery_progress"]),
                "failure_reason": str(info.get("failure_reason", "")),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
