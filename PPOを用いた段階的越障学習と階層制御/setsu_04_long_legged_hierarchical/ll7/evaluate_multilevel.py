"""四段階方策を複数障害物コースで繰り返して厳格評価する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.curriculum import CURRICULUM_LEVELS, get_course
from ll7.experiment import evaluate_repeating_stages


def parse_args():
    parser = argparse.ArgumentParser(description="在多障碍课程上评估四阶段控制器。")
    parser.add_argument("--level", type=int, choices=CURRICULUM_LEVELS, required=True)
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--clearance-model", required=True)
    parser.add_argument("--landing-model", required=True)
    parser.add_argument("--restart-model", required=True)
    parser.add_argument("--handoff-distance", type=float, default=0.25)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--output")
    return parser.parse_args()


def main():
    args = parse_args()
    models = {
        name: PPO.load(Path(getattr(args, f"{name}_model")), device="cpu")
        for name in ("approach", "clearance", "landing", "restart")
    }
    course = get_course(args.level)
    metrics = evaluate_repeating_stages(
        models["approach"],
        models["clearance"],
        models["landing"],
        models["restart"],
        args.handoff_distance,
        make_body(),
        args.level,
        args.episodes,
        course.max_steps,
        args.seed,
    )
    result = {
        "level": args.level,
        "course": course.as_dict(),
        "handoff_distance": args.handoff_distance,
        "models": {
            name: str(Path(getattr(args, f"{name}_model")).resolve())
            for name in models
        },
        "metrics": metrics,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

