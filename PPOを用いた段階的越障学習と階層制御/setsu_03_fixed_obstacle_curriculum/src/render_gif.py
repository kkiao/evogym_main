"""最終固定障害物環境で任意のPPOチェックポイントをGIFとして描画する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from evogym import get_full_connectivity
from stable_baselines3 import PPO

from src.bodies import BODY_NAMES, make_body
from src.curriculum import CURRICULUM_LEVELS, get_course
from src.environment import FixedCurriculumEnv
from src.wrappers import ActionRescaleWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="渲染固定障碍 PPO 检查点 GIF。")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--body-name", choices=BODY_NAMES, required=True)
    parser.add_argument("--level", type=int, choices=CURRICULUM_LEVELS, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40_000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--fps", type=int, default=25)
    return parser.parse_args()


def add_overlay(frame: np.ndarray, label: str, info: dict, step: int, obstacle_count: int):
    """段階名と厳格通過指標を画面左上へ重ねる。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = (
        f"PPO stage: {label} | env step: {step}\n"
        f"fixed obstacles cleared: {info['obstacles_cleared']}/{obstacle_count} | "
        f"max x: {info['max_x_position']:.2f} | "
        f"stable landing: {bool(info['landing_stable'])}"
    )
    box = draw.textbbox((0, 0), text)
    draw.rectangle(
        (4, 4, box[2] - box[0] + 22, box[3] - box[1] + 18),
        fill=(255, 255, 255),
    )
    draw.text((13, 10), text, fill=(20, 20, 20))
    return np.asarray(image)


def render(model_path, body_name, level, label, output, seed, frame_skip, fps):
    """一つの決定論的エピソードを描画し、同名JSONへ指標を保存する。"""
    body = make_body(body_name)
    course = get_course(level)
    raw = FixedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
        render_mode="rgb_array",
    )
    env = ActionRescaleWrapper(gym.wrappers.TimeLimit(raw, max_episode_steps=course.max_steps))
    model = PPO.load(model_path, device="cpu")
    obs, info = env.reset(seed=seed)
    frames = []
    total_return = 0.0
    try:
        for step in range(1, course.max_steps + 1):
            if (step - 1) % frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(add_overlay(frame, label, info, step - 1, len(course.obstacles)))
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                break
        frame = env.render()
        if frame is not None:
            frames.append(add_overlay(frame, label, info, step, len(course.obstacles)))
    finally:
        env.close()
    if not frames:
        raise RuntimeError("环境没有返回可保存画面。")
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps, loop=0)
    metrics = {
        "label": label,
        "model": str(model_path.resolve()),
        "body_name": body_name,
        "level": level,
        "return": total_return,
        "steps": step,
        "max_x": float(info["max_x_position"]),
        "obstacles_cleared": int(info["obstacles_cleared"]),
        "landing_stable_steps": int(info["landing_stable_steps"]),
        "success": bool(info["is_success"]),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main():
    args = parse_args()
    metrics = render(
        args.model,
        args.body_name,
        args.level,
        args.label,
        args.output,
        args.seed,
        args.frame_skip,
        args.fps,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
