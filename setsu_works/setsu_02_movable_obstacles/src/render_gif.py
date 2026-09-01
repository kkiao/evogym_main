"""一つの学習チェックポイントを指標説明付きGIFとして描画する。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from src.environment import LEGACY_ENVIRONMENT_VERSION
from src.experiment import PROJECT_DIR, make_env


def parse_args():
    parser = argparse.ArgumentParser(description="渲染多障碍 PPO 策略 GIF。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint", choices=("initial", "best", "latest"), default="best")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output")
    return parser.parse_args()


def model_path_for(run_dir: Path, checkpoint: str, checkpoint_step: int | None) -> Path:
    if checkpoint_step is not None:
        return run_dir / "checkpoints" / f"model_{checkpoint_step}_steps.zip"
    return run_dir / f"{checkpoint}_model.zip"


def add_overlay(frame: np.ndarray, step_label: str, info: dict, episode_step: int) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = (
        f"PPO checkpoint: {step_label} | frame step: {episode_step}\n"
        f"obstacles: {info['obstacles_cleared']}/7 | "
        f"max x: {info['max_x_position']:.2f} | success: {bool(info['is_success'])}"
    )
    box = draw.textbbox((0, 0), text)
    width = box[2] - box[0] + 18
    height = box[3] - box[1] + 14
    draw.rectangle((4, 4, 4 + width, 4 + height), fill=(255, 255, 255, 220))
    draw.text((13, 10), text, fill=(20, 20, 20))
    return np.asarray(image)


def render_checkpoint(
    run_dir: Path,
    checkpoint: str,
    checkpoint_step: int | None,
    output_path: Path,
    seed: int,
    max_steps: int | None,
    frame_skip: int,
    fps: int,
) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    body = np.load(run_dir / "body.npy")
    limit = max_steps or int(config["max_steps"])
    environment_version = config.get("environment_version", LEGACY_ENVIRONMENT_VERSION)
    try:
        env = make_env(
            body,
            limit,
            render_mode="rgb_array",
            environment_version=environment_version,
        )
    except IndexError as error:
        if "invalid vector<bool> subscript" not in str(error):
            raise
        gc.collect()
        env = make_env(
            body,
            limit,
            render_mode="rgb_array",
            environment_version=environment_version,
        )
    path = model_path_for(run_dir, checkpoint, checkpoint_step)
    if not path.exists():
        env.close()
        raise FileNotFoundError(f"找不到模型：{path}")
    model = PPO.load(path, env=env, device="cpu")
    obs, info = env.reset(seed=seed)
    initial_x = float(info["x_position"])
    step_label = str(checkpoint_step) if checkpoint_step is not None else checkpoint
    frames = []
    total_return = 0.0
    steps = 0
    try:
        for episode_step in range(limit):
            if episode_step % frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(add_overlay(frame, step_label, info, episode_step))
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            steps += 1
            if terminated or truncated:
                break
        final_frame = env.render()
        if final_frame is not None:
            frames.append(add_overlay(final_frame, step_label, info, steps))
    finally:
        env.close()
    if not frames:
        raise RuntimeError("环境没有返回可保存画面。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    metrics = {
        "checkpoint": step_label,
        "model_path": str(path),
        "gif_path": str(output_path),
        "return": total_return,
        "steps": steps,
        "displacement": float(info["x_position"]) - initial_x,
        "max_x": float(info["max_x_position"]),
        "maximum_com_y": float(info["maximum_com_y"]),
        "maximum_bottom_y": float(info["maximum_bottom_y"]),
        "obstacles_cleared": int(info["obstacles_cleared"]),
        "success": bool(info["is_success"]),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main():
    args = parse_args()
    if args.checkpoint_step is not None and args.checkpoint_step <= 0:
        raise ValueError("--checkpoint-step 必须大于0。")
    if args.frame_skip <= 0 or args.fps <= 0:
        raise ValueError("--frame-skip 和 --fps 必须大于0。")
    run_dir = PROJECT_DIR / "runs" / args.run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"找不到实验目录：{run_dir}")
    stage = f"step_{args.checkpoint_step}" if args.checkpoint_step else args.checkpoint
    output_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / f"{stage}_obstacle_demo.gif"
    )
    metrics = render_checkpoint(
        run_dir,
        args.checkpoint,
        args.checkpoint_step,
        output_path,
        args.seed,
        args.max_steps,
        args.frame_skip,
        args.fps,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
