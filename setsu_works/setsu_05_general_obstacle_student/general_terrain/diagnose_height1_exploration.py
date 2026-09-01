"""高さ一パイロットの確率的探索が生んだ部分能力を診断する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import build_course


def main() -> None:
    """固定二形状を複数回再生し高さと通過の最大値を保存する。"""
    parser = argparse.ArgumentParser(description="诊断高度1试验的随机探索行为。")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    args = parser.parse_args()
    model = PPO.load(Path(args.model), device="cpu")
    rows = []
    for template_index, template_name in enumerate(("low_hurdle", "low_platform_short")):
        course = build_course(
            [template_name],
            split="exploration_diagnosis",
            seed=40_000 + template_index,
            difficulty=1,
            start_runway_voxels=20,
        )
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        try:
            for episode in range(args.episodes):
                observation, info = environment.reset(seed=41_000 + episode)
                positions = environment.object_pos_at_time(environment.get_time(), "robot")
                maximum_bottom_y = float(np.min(positions[1]))
                maximum_front_x = float(np.max(positions[0]))
                terminated = False
                truncated = False
                steps = 0
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=False)
                    observation, _, terminated, truncated, info = environment.step(action)
                    positions = environment.object_pos_at_time(environment.get_time(), "robot")
                    maximum_bottom_y = max(maximum_bottom_y, float(np.min(positions[1])))
                    maximum_front_x = max(maximum_front_x, float(np.max(positions[0])))
                    steps += 1
                rows.append(
                    {
                        "template": template_name,
                        "episode": episode,
                        "steps": steps,
                        "maximum_bottom_y": maximum_bottom_y,
                        "maximum_front_x": maximum_front_x,
                        "raw_clearances": int(info["raw_clearances"]),
                        "recovered_obstacles": int(info["recovered_obstacles"]),
                        "success": bool(info["course_complete"]),
                        "hard_fall": bool(info["hard_fall"]),
                    }
                )
        finally:
            environment.close()
    result = {
        "model": str(Path(args.model).resolve()),
        "episodes": len(rows),
        "raw_clearance_rate": float(np.mean([row["raw_clearances"] > 0 for row in rows])),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "hard_fall_rate": float(np.mean([row["hard_fall"] for row in rows])),
        "maximum_bottom_y": max(row["maximum_bottom_y"] for row in rows),
        "maximum_front_x": max(row["maximum_front_x"] for row in rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
