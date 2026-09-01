"""第一障害物の安全通過後に既存着地方策と動作縮尺を比較する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_first_safe_landing_env


def evaluate(model, scale, args, approach, clearance, safe_clearance):
    """一つの候補を決定論的に再生して厳格着地指標を返す。"""
    env = make_first_safe_landing_env(
        make_body(),
        approach_model=approach,
        clearance_model=clearance,
        safe_clearance_model=safe_clearance,
        prefix_fraction=0.5,
        agent_max_steps=args.max_steps,
        prefix_max_steps=args.prefix_max_steps,
        max_orientation=math.radians(args.max_orientation_degrees),
        preferred_orientation=math.radians(args.preferred_orientation_degrees),
        agent_action_scale=scale,
    )
    try:
        obs, info = env.reset(seed=10_000)
        maximum_angle = max(
            float(info["orientation_error"]),
            float(info.get("unwrapped_orientation_error", 0.0)),
        )
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
    return {
        "scale": scale,
        "steps": step,
        "strict_clearances": int(info["strict_clearances"]),
        "stable_landings": int(info["stable_landings"]),
        "maximum_degrees": math.degrees(maximum_angle),
        "body_contact_steps": contact_steps,
        "final_degrees": math.degrees(float(info["orientation_error"])),
        "final_speed": float(info["com_speed"]),
        "failure_reason": str(info.get("failure_reason", "")),
    }


def main():
    parser = argparse.ArgumentParser(description="比较安全越墙后的第一墙落地候选。")
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--clearance-model", required=True)
    parser.add_argument("--safe-clearance-model", required=True)
    parser.add_argument("--candidate-model", action="append", required=True)
    parser.add_argument("--scales", default="0.05,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--max-orientation-degrees", type=float, default=50.0)
    parser.add_argument("--preferred-orientation-degrees", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--prefix-max-steps", type=int, default=2_000)
    parser.add_argument("--output", default="analysis/first_safe_landing_candidates.json")
    args = parser.parse_args()

    approach = PPO.load(Path(args.approach_model).resolve(), device="cpu")
    clearance = PPO.load(Path(args.clearance_model).resolve(), device="cpu")
    safe_clearance = PPO.load(Path(args.safe_clearance_model).resolve(), device="cpu")
    scales = [float(value) for value in args.scales.split(",")]
    results = []
    for candidate_path in args.candidate_model:
        model = PPO.load(Path(candidate_path).resolve(), device="cpu")
        for scale in scales:
            result = evaluate(model, scale, args, approach, clearance, safe_clearance)
            result["model"] = str(Path(candidate_path))
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
