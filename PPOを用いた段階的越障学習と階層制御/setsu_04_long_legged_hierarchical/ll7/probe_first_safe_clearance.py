"""第一障害物の胴体非接地通過方策を複数シードとGIFで検証する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import make_first_early_landing_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_env(args, approach, clearance, render_mode=None):
    """学習時と同一の早期交接・安全制約を再構築する。"""
    return make_first_early_landing_env(
        make_body(),
        approach_model=approach,
        clearance_model=clearance,
        prefix_fraction=args.prefix_fraction,
        agent_max_steps=args.max_steps,
        prefix_max_steps=args.prefix_max_steps,
        max_orientation=math.radians(args.max_orientation_degrees),
        preferred_orientation=math.radians(args.preferred_orientation_degrees),
        target_clearance_only=True,
        render_mode=render_mode,
    )


def run_episode(env, model, seed, max_steps, max_orientation_degrees, collect_frames=False):
    """一回の決定論的軌跡から厳格指標と任意の画面列を返す。"""
    obs, info = env.reset(seed=seed)
    frames = []
    maximum_angle = max(
        float(info["orientation_error"]),
        float(info.get("unwrapped_orientation_error", 0.0)),
    )
    contact_steps = 0
    for step in range(1, max_steps + 1):
        if collect_frames:
            frame = env.render()
            frames.append((step - 1, frame, dict(info)))
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
    if collect_frames:
        frame = env.render()
        frames.append((step, frame, dict(info)))
    success = bool(
        int(info["strict_clearances"]) >= 1
        and contact_steps == 0
        and maximum_angle < math.radians(max_orientation_degrees)
    )
    return {
        "seed": seed,
        "success": success,
        "steps": step,
        "strict_clearances": int(info["strict_clearances"]),
        "maximum_degrees": math.degrees(maximum_angle),
        "body_contact_steps": contact_steps,
        "clearance_angle_degrees": math.degrees(float(info["orientation_error"])),
        "clearance_speed": float(info["com_speed"]),
    }, frames


def add_overlay(frame, step, info):
    """肉眼検査用GIFへ通過角度と胴体接地状態を表示する。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = (
        f"First wall safe clearance | step {step}\n"
        f"angle {math.degrees(float(info['orientation_error'])):.1f} deg | "
        f"speed {float(info['com_speed']):.2f}\n"
        f"strict clear {int(info['strict_clearances'])} | "
        f"torso contact {bool(info.get('upper_body_grounded', False))}"
    )
    box = draw.textbbox((0, 0), text)
    draw.rectangle((4, 4, box[2] + 18, box[3] + 16), fill="white")
    draw.text((12, 9), text, fill=(15, 15, 15))
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser(description="严格复验第一墙零躯干接触通行。")
    parser.add_argument("--model", required=True)
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--clearance-model", required=True)
    parser.add_argument("--output-dir", default="analysis/first_safe_clearance_v31")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    parser.add_argument("--max-orientation-degrees", type=float, default=50.0)
    parser.add_argument("--preferred-orientation-degrees", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--prefix-max-steps", type=int, default=2_000)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    model = PPO.load(Path(args.model).resolve(), device="cpu")
    approach = PPO.load(Path(args.approach_model).resolve(), device="cpu")
    clearance = PPO.load(Path(args.clearance_model).resolve(), device="cpu")
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args, approach, clearance)
    rows = []
    try:
        for index in range(args.episodes):
            row, _ = run_episode(
                env,
                model,
                10_000 + index,
                args.max_steps,
                args.max_orientation_degrees,
            )
            rows.append(row)
    finally:
        env.close()

    render_env = make_env(args, approach, clearance, render_mode="rgb_array")
    try:
        _, frames = run_episode(
            render_env,
            model,
            10_000,
            args.max_steps,
            args.max_orientation_degrees,
            collect_frames=True,
        )
    finally:
        render_env.close()
    images = [add_overlay(frame, step, info) for step, frame, info in frames]
    images = [images[0]] * 4 + images + [images[-1]] * 8
    imageio.mimsave(output_dir / "first_wall_safe_clearance.gif", images, fps=args.fps, loop=0)

    summary = {
        "episodes": args.episodes,
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "maximum_degrees": max(row["maximum_degrees"] for row in rows),
        "total_body_contact_steps": sum(row["body_contact_steps"] for row in rows),
        "episodes_detail": rows,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
