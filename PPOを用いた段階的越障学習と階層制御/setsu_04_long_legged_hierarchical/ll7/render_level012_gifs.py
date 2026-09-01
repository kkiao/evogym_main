"""既存チェックポイントからLevel 0から2の主要9段階GIFを生成する。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.curriculum import get_course
from ll7.experiment import make_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "submission_assets"
APPROACH_MODEL = RUNS_DIR / "_smoke_legacy_exact_v3" / "initial_model.zip"


def checkpoint(run_name: str, step: int) -> Path:
    """実験名と5k歩数からチェックポイント絶対パスを返す。"""
    return RUNS_DIR / run_name / "checkpoints" / f"model_{step}_steps.zip"


LEVEL0_NODES = (
    ("00_initial", "initial policy", RUNS_DIR / "level0_long_legged_gap3_seed7_consolidate_v2" / "initial_model.zip"),
    ("01_step_005000", "5k", checkpoint("level0_long_legged_gap3_seed7_consolidate_v2", 5_000)),
    ("02_step_010000", "10k", checkpoint("level0_long_legged_gap3_seed7_consolidate_v2", 10_000)),
    ("03_step_015000", "15k transition", checkpoint("level0_long_legged_gap3_seed7_consolidate_v2", 15_000)),
    ("04_step_020000", "20k first stable success", checkpoint("level0_long_legged_gap3_seed7_consolidate_v2", 20_000)),
    ("05_step_030000", "30k", checkpoint("level0_long_legged_gap3_seed7_consolidate_v2", 30_000)),
    ("06_step_040000", "40k", checkpoint("level0_long_legged_gap3_seed7_consolidate_v2", 40_000)),
    ("07_step_050000", "50k", checkpoint("level0_long_legged_gap3_seed7_consolidate_v2", 50_000)),
    ("08_step_060000", "60k final", checkpoint("level0_long_legged_gap3_seed7_consolidate_v2", 60_000)),
)

LEVEL1_RESTART_NODES = (
    ("00_initial", "restart initial", RUNS_DIR / "level1_four_stage_restart_consolidate_seed7_v2" / "initial_model.zip"),
    ("01_step_005000", "restart 5k", checkpoint("level1_four_stage_restart_consolidate_seed7_v2", 5_000)),
    ("02_step_010000", "restart 10k", checkpoint("level1_four_stage_restart_consolidate_seed7_v2", 10_000)),
    ("03_step_015000", "restart 15k", checkpoint("level1_four_stage_restart_consolidate_seed7_v2", 15_000)),
    ("04_step_020000", "restart 20k", checkpoint("level1_four_stage_restart_consolidate_seed7_v2", 20_000)),
    ("05_step_025000", "restart 25k", checkpoint("level1_four_stage_restart_consolidate_seed7_v2", 25_000)),
    ("06_step_030000", "restart 30k", checkpoint("level1_four_stage_restart_consolidate_seed7_v2", 30_000)),
    ("07_step_035000", "restart 35k first success", checkpoint("level1_four_stage_restart_consolidate_seed7_v2", 35_000)),
    ("08_step_040000", "restart 40k final", checkpoint("level1_four_stage_restart_consolidate_seed7_v2", 40_000)),
)

LEVEL1_CLEARANCE = RUNS_DIR / "level1_upright_clearance_seed7_v1" / "best_model.zip"
LEVEL1_LANDING = checkpoint("level1_three_stage_braking_seed7_v2", 55_000)
LEVEL1_RESTART_FALLBACK = RUNS_DIR / "level1_four_stage_restart_consolidate_seed7_v2" / "initial_model.zip"

LEVEL2_NODES = (
    {
        "name": "00_wall1_clear_005k",
        "label": "wall 1 clearance 5k",
        "first": checkpoint("level2_angle65_seed7_v6", 5_000),
        "first_restart": LEVEL1_RESTART_FALLBACK,
    },
    {
        "name": "01_wall1_clear_015k",
        "label": "wall 1 clearance 15k",
        "first": checkpoint("level2_angle65_seed7_v6", 15_000),
        "first_restart": LEVEL1_RESTART_FALLBACK,
    },
    {
        "name": "02_wall1_land_030k",
        "label": "wall 1 strict landing",
        "first": checkpoint("level2_angle65_seed7_v6", 30_000),
        "first_restart": LEVEL1_RESTART_FALLBACK,
    },
    {
        "name": "03_wall1_restart_005k",
        "label": "wall 1 restart 5k",
        "first": checkpoint("level2_angle65_seed7_v6", 30_000),
        "first_restart": checkpoint("level2_restart_after_strict_landing_seed7_v7", 5_000),
    },
    {
        "name": "04_wall1_restart_010k",
        "label": "wall 1 validated",
        "first": checkpoint("level2_angle65_seed7_v6", 30_000),
        "first_restart": checkpoint("level2_restart_after_strict_landing_seed7_v7", 10_000),
    },
    {
        "name": "05_wall2_clear_005k",
        "label": "wall 2 clearance",
        "first": checkpoint("level2_angle65_seed7_v6", 30_000),
        "first_restart": checkpoint("level2_restart_after_strict_landing_seed7_v7", 10_000),
        "second_clearance": checkpoint("level2_second_angle45_seed7_v17", 5_000),
    },
    {
        "name": "06_wall2_land_005k",
        "label": "wall 2 landing 5k",
        "first": checkpoint("level2_angle65_seed7_v6", 30_000),
        "first_restart": checkpoint("level2_restart_after_strict_landing_seed7_v7", 10_000),
        "second_clearance": checkpoint("level2_second_angle45_seed7_v17", 5_000),
        "second_landing": checkpoint("level2_second_landing90_speed03_seed7_v28", 5_000),
    },
    {
        "name": "07_wall2_land_015k",
        "label": "wall 2 strict landing",
        "first": checkpoint("level2_angle65_seed7_v6", 30_000),
        "first_restart": checkpoint("level2_restart_after_strict_landing_seed7_v7", 10_000),
        "second_clearance": checkpoint("level2_second_angle45_seed7_v17", 5_000),
        "second_landing": checkpoint("level2_second_landing90_speed03_seed7_v28", 15_000),
    },
    {
        "name": "08_full_strict_success",
        "label": "two-wall strict success",
        "first": checkpoint("level2_angle65_seed7_v6", 30_000),
        "first_restart": checkpoint("level2_restart_after_strict_landing_seed7_v7", 10_000),
        "second_clearance": checkpoint("level2_second_angle45_seed7_v17", 5_000),
        "second_landing": checkpoint("level2_second_landing90_speed03_seed7_v28", 15_000),
        "second_restart": checkpoint("level2_second_relaxed_restart_seed7_v26", 5_000),
    },
)


class ModelCache:
    """同じPPOファイルを一度だけCPUへ読み込む。"""

    def __init__(self):
        self._models = {}

    def get(self, path: Path):
        resolved = path.resolve()
        if resolved not in self._models:
            if not resolved.exists():
                raise FileNotFoundError(f"找不到模型：{resolved}")
            self._models[resolved] = PPO.load(resolved, device="cpu")
        return self._models[resolved]


def add_overlay(frame: np.ndarray, level: int, label: str, step: int, info: dict):
    """GIF左上へ段階名と厳格状態機械の計数を重ねる。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = (
        f"Level {level} | {label} | env step {step}\n"
        f"clear {info['strict_clearances']} | land {info['stable_landings']} | "
        f"restart {info['restart_successes']} | validated {info['validated_obstacles']} | "
        f"success {bool(info['is_success'])}"
    )
    box = draw.textbbox((0, 0), text)
    draw.rectangle((4, 4, box[2] + 18, box[3] + 16), fill=(255, 255, 255))
    draw.text((12, 9), text, fill=(15, 15, 15))
    return np.asarray(image)


def render_episode(level, label, output, controller, max_steps, frame_skip, fps):
    """一つの決定論的エピソードをGIFと指標JSONへ保存する。"""
    course = get_course(level)
    env = None
    frames = []
    total_return = 0.0
    try:
        env = make_env(
            make_body(),
            level,
            min(max_steps, course.max_steps),
            render_mode="rgb_array",
        )
        obs, info = env.reset(seed=7)
        for step in range(1, min(max_steps, course.max_steps) + 1):
            if (step - 1) % frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(add_overlay(frame, level, label, step - 1, info))
            action = controller(obs, env, info)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            if terminated or truncated:
                break
        frame = env.render()
        if frame is not None:
            frames.append(add_overlay(frame, level, label, step, info))
    finally:
        if env is not None:
            env.close()
        gc.collect()
    if not frames:
        raise RuntimeError("环境没有返回可保存画面。")
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=fps, loop=0)
    metrics = {
        "level": level,
        "label": label,
        "steps": step,
        "return": total_return,
        "strict_clearances": int(info["strict_clearances"]),
        "stable_landings": int(info["stable_landings"]),
        "restart_successes": int(info["restart_successes"]),
        "validated_obstacles": int(info["validated_obstacles"]),
        "max_x": float(info["max_x_position"]),
        "success": bool(info["is_success"]),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def predict(model, obs):
    """PPOの決定論的平均行動だけを返す。"""
    action, _ = model.predict(obs, deterministic=True)
    return action


def render_level0(cache, frame_skip, fps):
    """平地方策の9チェックポイントを描画する。"""
    results = []
    for name, label, path in LEVEL0_NODES:
        model = cache.get(path)
        controller = lambda obs, _env, _info, model=model: predict(model, obs)
        output = OUTPUT_ROOT / "level_0" / "gifs" / f"{name}.gif"
        results.append(render_episode(0, label, output, controller, 700, frame_skip, fps))
    return results


def render_level1(cache, frame_skip, fps):
    """固定した前三方策と学習中の再前進方策を組み合わせて描画する。"""
    approach = cache.get(APPROACH_MODEL)
    clearance = cache.get(LEVEL1_CLEARANCE)
    landing = cache.get(LEVEL1_LANDING)
    results = []
    for name, label, restart_path in LEVEL1_RESTART_NODES:
        restart = cache.get(restart_path)

        def controller(obs, env, info, restart=restart):
            if info["phase"] == "landing":
                model = landing
            elif info["phase"] == "restart":
                model = restart
            else:
                model = approach if float(info["x_position"]) < 1.15 else clearance
            return predict(model, obs)

        output = OUTPUT_ROOT / "level_1" / "gifs" / f"{name}.gif"
        results.append(render_episode(1, label, output, controller, 1_300, frame_skip, fps))
    return results


def render_level2(cache, frame_skip, fps):
    """二障害物の技能獲得順に9個の階層制御器を描画する。"""
    approach = cache.get(APPROACH_MODEL)
    results = []
    for node in LEVEL2_NODES:
        first = cache.get(node["first"])
        first_restart = cache.get(node["first_restart"])
        second_clearance = cache.get(node.get("second_clearance", node["first"]))
        second_landing = cache.get(node.get("second_landing", node.get("second_clearance", node["first"])))
        second_restart = cache.get(node.get("second_restart", node["first_restart"]))

        def controller(obs, env, info):
            raw = env.unwrapped
            if int(info["validated_obstacles"]) < 1:
                if info["phase"] == "landing":
                    model = first
                elif info["phase"] == "restart":
                    model = first_restart
                else:
                    obstacle_x = raw.course.obstacles[0].start_x * raw.VOXEL_SIZE
                    model = approach if obstacle_x - float(info["x_position"]) > 0.25 else first
            elif int(info["strict_clearances"]) < 2:
                model = second_clearance
            elif info["phase"] == "landing":
                model = second_landing
            else:
                model = second_restart
            return predict(model, obs)

        output = OUTPUT_ROOT / "level_2" / "gifs" / f"{node['name']}.gif"
        results.append(
            render_episode(2, node["label"], output, controller, 1_850, frame_skip, fps)
        )
    return results


def write_readme(level: int, results: list[dict]):
    """各GIFの順番と厳格結果を中国語Markdownへまとめる。"""
    gif_dir = OUTPUT_ROOT / f"level_{level}" / "gifs"
    gif_files = sorted(gif_dir.glob("*.gif"))
    lines = [
        f"# Level {level} 九个关键节点 GIF",
        "",
        "所有 GIF 均为已有检查点的确定性评估；文件名前两位是播放顺序。",
        "",
        "| 顺序 | 文件 | 越障 | 落地 | 恢复 | 完成 | success |",
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for index, (gif_path, metrics) in enumerate(zip(gif_files, results)):
        lines.append(
            f"| {index} | `{gif_path.name}` | {metrics['strict_clearances']} | "
            f"{metrics['stable_landings']} | {metrics['restart_successes']} | "
            f"{metrics['validated_obstacles']} | {metrics['success']} |"
        )
    (gif_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="生成 Level 0–2 的九节点 GIF。")
    parser.add_argument("--level", type=int, choices=(0, 1, 2))
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_skip <= 0 or args.fps <= 0:
        raise ValueError("--frame-skip 和 --fps 必须大于0。")
    cache = ModelCache()
    levels = (args.level,) if args.level is not None else (0, 1, 2)
    for level in levels:
        if level == 0:
            results = render_level0(cache, args.frame_skip, args.fps)
        elif level == 1:
            results = render_level1(cache, args.frame_skip, args.fps)
        else:
            results = render_level2(cache, args.frame_skip, args.fps)
        write_readme(level, results)
        print(f"level={level} gifs={len(results)}", flush=True)


if __name__ == "__main__":
    main()
