"""改良第一障害物後の第二障害物交接距離と既存方策を比較する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_improved_second_crossing_env


def main():
    parser = argparse.ArgumentParser(description="比较第二墙交接距离。")
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--first-half-model", required=True)
    parser.add_argument("--safe-clearance-model", required=True)
    parser.add_argument("--landing-model", required=True)
    parser.add_argument("--restart-model", required=True)
    parser.add_argument("--candidate-model", action="append", required=True)
    parser.add_argument("--distances", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", default="analysis/second_handoff_candidates.json")
    args = parser.parse_args()

    fixed = {
        "approach": PPO.load(Path(args.approach_model).resolve(), device="cpu"),
        "first_half": PPO.load(Path(args.first_half_model).resolve(), device="cpu"),
        "safe": PPO.load(Path(args.safe_clearance_model).resolve(), device="cpu"),
        "landing": PPO.load(Path(args.landing_model).resolve(), device="cpu"),
        "restart": PPO.load(Path(args.restart_model).resolve(), device="cpu"),
    }
    results = []
    for candidate_path in args.candidate_model:
        candidate = PPO.load(Path(candidate_path).resolve(), device="cpu")
        for distance in [float(value) for value in args.distances.split(",")]:
            env = make_improved_second_crossing_env(
                make_body(),
                approach_model=fixed["approach"],
                first_half_model=fixed["first_half"],
                first_safe_clearance_model=fixed["safe"],
                first_landing_model=fixed["landing"],
                first_restart_model=fixed["restart"],
                target_fraction=1.0,
                agent_max_steps=args.max_steps,
                prefix_max_steps=3_000,
                second_handoff_distance=distance,
                max_orientation=math.radians(65.0),
                preferred_orientation=math.radians(20.0),
            )
            try:
                obs, info = env.reset(seed=10_000)
                maximum_angle = float(info["orientation_error"])
                contact_steps = 0
                for step in range(1, args.max_steps + 1):
                    action, _ = candidate.predict(obs, deterministic=True)
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
            row = {
                "model": str(Path(candidate_path)),
                "distance": distance,
                "steps": step,
                "maximum_fraction": float(info["maximum_crossed_fraction"]),
                "strict_clearances": int(info["strict_clearances"]),
                "maximum_degrees": math.degrees(maximum_angle),
                "body_contact_steps": contact_steps,
                "failure_reason": str(info.get("failure_reason", "")),
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
