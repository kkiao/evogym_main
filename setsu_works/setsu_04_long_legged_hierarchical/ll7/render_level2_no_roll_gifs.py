"""Level 2無横転最適化の九つの主要段階をGIFへ書き出す。"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
OUTPUT_DIR = PROJECT_ROOT / "submission_assets" / "level_2_no_roll" / "gifs"
APPROACH = RUNS_DIR / "_smoke_legacy_exact_v3" / "initial_model.zip"


def checkpoint(run_name: str, step: int) -> Path:
    """実験名と歩数から保存済みPPOチェックポイントを返す。"""
    return RUNS_DIR / run_name / "checkpoints" / f"model_{step}_steps.zip"


OLD_FIRST = checkpoint("level2_angle65_seed7_v6", 30_000)
OLD_FIRST_RESTART = checkpoint("level2_restart_after_strict_landing_seed7_v7", 10_000)
OLD_SECOND_CLEAR = checkpoint("level2_second_angle45_seed7_v17", 5_000)
OLD_SECOND_LAND = checkpoint("level2_second_landing90_speed03_seed7_v28", 15_000)
OLD_SECOND_RESTART = checkpoint("level2_second_relaxed_restart_seed7_v26", 5_000)

FINAL_FIRST_CLEAR = checkpoint("quality_l2_clearance_speed32_angle45_seed7_v12", 20_000)
FINAL_FIRST_LAND = checkpoint("quality_l2_wall1_speed32_peakreward88_seed7_v20", 25_000)
FINAL_SECOND_CLEAR = checkpoint("quality_l2_second_angle75_speed60_noroll_seed7_v24", 20_000)
FINAL_SECOND_LAND = checkpoint("quality_l2_second_brake_speed03_seed7_v27", 30_000)


NODES = (
    {
        "name": "00_baseline_side_roll",
        "label": "baseline: side-roll recovery",
        "first_clear": OLD_FIRST,
        "first_land": OLD_FIRST,
        "first_restart": OLD_FIRST_RESTART,
        "second_clear": OLD_SECOND_CLEAR,
        "second_land": OLD_SECOND_LAND,
        "second_restart": OLD_SECOND_RESTART,
    },
    {
        "name": "01_wall1_speed45",
        "label": "wall 1: speed <= 4.5",
        "first_clear": checkpoint("quality_l2_clearance_speed45_seed7_v6", 30_000),
        "first_land": checkpoint("quality_l2_wall1_speed45_land85_seed7_v8", 50_000),
        "first_restart": OLD_FIRST_RESTART,
    },
    {
        "name": "02_wall1_speed35",
        "label": "wall 1: speed <= 3.5",
        "first_clear": checkpoint("quality_l2_clearance_speed35_seed7_v10", 25_000),
        "first_land": checkpoint("quality_l2_wall1_speed45_land85_seed7_v8", 50_000),
        "first_restart": OLD_FIRST_RESTART,
    },
    {
        "name": "03_wall1_no_roll",
        "label": "wall 1: no-roll + restart",
        "first_clear": FINAL_FIRST_CLEAR,
        "first_land": FINAL_FIRST_LAND,
        "first_restart": OLD_FIRST_RESTART,
    },
    {
        "name": "04_wall2_fraction50",
        "label": "wall 2: no-roll crossing found",
        "first_clear": FINAL_FIRST_CLEAR,
        "first_land": FINAL_FIRST_LAND,
        "first_restart": OLD_FIRST_RESTART,
        "second_clear": checkpoint("quality_l2_second_fraction50_noroll_seed7_v23", 5_000),
        "second_land": checkpoint("quality_l2_second_fraction50_noroll_seed7_v23", 5_000),
        "second_restart": OLD_SECOND_RESTART,
    },
    {
        "name": "05_wall2_upright_clear",
        "label": "wall 2: 46.58 deg clearance",
        "first_clear": FINAL_FIRST_CLEAR,
        "first_land": FINAL_FIRST_LAND,
        "first_restart": OLD_FIRST_RESTART,
        "second_clear": FINAL_SECOND_CLEAR,
        "second_land": checkpoint("quality_l2_second_fraction50_noroll_seed7_v23", 5_000),
        "second_restart": OLD_SECOND_RESTART,
    },
    {
        "name": "06_wall2_brake_speed05",
        "label": "wall 2: braking <= 0.5",
        "first_clear": FINAL_FIRST_CLEAR,
        "first_land": FINAL_FIRST_LAND,
        "first_restart": OLD_FIRST_RESTART,
        "second_clear": FINAL_SECOND_CLEAR,
        "second_land": checkpoint("quality_l2_second_brake_speed05_seed7_v26", 35_000),
        "second_restart": OLD_SECOND_RESTART,
    },
    {
        "name": "07_wall2_strict_landing",
        "label": "wall 2: strict landing <= 0.15",
        "first_clear": FINAL_FIRST_CLEAR,
        "first_land": FINAL_FIRST_LAND,
        "first_restart": OLD_FIRST_RESTART,
        "second_clear": FINAL_SECOND_CLEAR,
        "second_land": checkpoint("quality_l2_second_brake_speed03_seed7_v27", 15_000),
        "second_restart": OLD_SECOND_RESTART,
    },
    {
        "name": "08_full_level2_no_roll_success",
        "label": "final: 2/2/2/2 no-roll success",
        "first_clear": FINAL_FIRST_CLEAR,
        "first_land": FINAL_FIRST_LAND,
        "first_restart": OLD_FIRST_RESTART,
        "second_clear": FINAL_SECOND_CLEAR,
        "second_land": FINAL_SECOND_LAND,
        "second_restart": OLD_SECOND_RESTART,
    },
)


class ModelCache:
    """各PPOファイルを一度だけCPUへ読み込む。"""

    def __init__(self):
        self.models = {}

    def get(self, path: Path):
        resolved = path.resolve()
        if resolved not in self.models:
            if not resolved.exists():
                raise FileNotFoundError(f"找不到模型：{resolved}")
            self.models[resolved] = PPO.load(resolved, device="cpu")
        return self.models[resolved]


def predict(model, observation):
    """確率分布の平均に対応する決定論的行動を返す。"""
    action, _ = model.predict(observation, deterministic=True)
    return action


def add_overlay(frame, label, step, info, maximum_angle):
    """画面へ姿勢角と厳格状態機械の進行を表示する。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    angle = math.degrees(float(info["orientation_error"]))
    text = (
        f"Level 2 no-roll | {label} | env step {step}\n"
        f"angle {angle:.1f} deg | max {maximum_angle:.1f} deg | "
        f"speed {float(info['com_speed']):.2f}\n"
        f"clear {info['strict_clearances']} | land {info['stable_landings']} | "
        f"restart {info['restart_successes']} | validated {info['validated_obstacles']} | "
        f"success {bool(info['is_success'])}"
    )
    box = draw.textbbox((0, 0), text)
    draw.rectangle((4, 4, box[2] + 18, box[3] + 16), fill=(255, 255, 255))
    draw.text((12, 9), text, fill=(15, 15, 15))
    return np.asarray(image)


def render_node(cache, node, frame_skip, fps):
    """一つの階層制御器を決定論的に再生してGIFとJSONを保存する。"""
    approach = cache.get(APPROACH)
    first_clear = cache.get(node["first_clear"])
    first_land = cache.get(node["first_land"])
    first_restart = cache.get(node["first_restart"])
    second_clear = cache.get(node.get("second_clear", node["first_clear"]))
    second_land = cache.get(node.get("second_land", node.get("second_clear", node["first_land"])))
    second_restart = cache.get(node.get("second_restart", node["first_restart"]))
    env = None
    frames = []
    maximum_angle = 0.0
    strict_events = []
    total_return = 0.0
    course = get_course(2)
    try:
        env = make_env(make_body(), 2, course.max_steps, render_mode="rgb_array")
        observation, info = env.reset(seed=7)
        for step in range(1, 1_851):
            maximum_angle = max(
                maximum_angle,
                math.degrees(float(info["orientation_error"])),
            )
            if (step - 1) % frame_skip == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(
                        add_overlay(frame, node["label"], step - 1, info, maximum_angle)
                    )
            if int(info["validated_obstacles"]) < 1:
                if info["phase"] == "landing":
                    model = first_land
                elif info["phase"] == "restart":
                    model = first_restart
                else:
                    obstacle_x = env.unwrapped.course.obstacles[0].start_x * env.unwrapped.VOXEL_SIZE
                    model = approach if obstacle_x - float(info["x_position"]) > 0.25 else first_clear
            elif int(info["strict_clearances"]) < 2:
                model = second_clear
            elif info["phase"] == "landing":
                model = second_land
            else:
                model = second_restart
            observation, reward, terminated, truncated, info = env.step(
                predict(model, observation)
            )
            total_return += float(reward)
            maximum_angle = max(
                maximum_angle,
                math.degrees(float(info["orientation_error"])),
            )
            if info.get("new_strict_clearance", False):
                strict_events.append(
                    {
                        "count": int(info["strict_clearances"]),
                        "angle_degrees": math.degrees(float(info["orientation_error"])),
                        "speed": float(info["com_speed"]),
                    }
                )
            if terminated or truncated:
                break
        frame = env.render()
        if frame is not None:
            frames.append(add_overlay(frame, node["label"], step, info, maximum_angle))
    finally:
        if env is not None:
            env.close()
        gc.collect()
    if not frames:
        raise RuntimeError("环境没有返回可保存画面。")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{node['name']}.gif"
    imageio.mimsave(output, frames, fps=fps, loop=0)
    metrics = {
        "label": node["label"],
        "steps": step,
        "return": total_return,
        "maximum_orientation_degrees": maximum_angle,
        "strict_events": strict_events,
        "strict_clearances": int(info["strict_clearances"]),
        "stable_landings": int(info["stable_landings"]),
        "restart_successes": int(info["restart_successes"]),
        "validated_obstacles": int(info["validated_obstacles"]),
        "success": bool(info["is_success"]),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def write_readme(results):
    """九つのGIFと主要結果を中国語の表へまとめる。"""
    lines = [
        "# Level 2 无侧翻姿态优化：九个关键节点 GIF",
        "",
        "GIF 均为保存检查点的确定性回放。角度小于 90°表示全过程没有发生侧翻。",
        "",
        "| 顺序 | 文件 | 最大倾角 | 越障 | 落地 | 恢复 | 验证 | 成功 |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for index, (node, metrics) in enumerate(zip(NODES, results)):
        lines.append(
            f"| {index} | `{node['name']}.gif` | "
            f"{metrics['maximum_orientation_degrees']:.2f}° | "
            f"{metrics['strict_clearances']} | {metrics['stable_landings']} | "
            f"{metrics['restart_successes']} | {metrics['validated_obstacles']} | "
            f"{metrics['success']} |"
        )
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="生成Level 2无侧翻优化的九个GIF。")
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.frame_skip <= 0 or args.fps <= 0:
        raise ValueError("--frame-skip 和 --fps 必须大于0。")
    cache = ModelCache()
    results = [render_node(cache, node, args.frame_skip, args.fps) for node in NODES]
    write_readme(results)
    print(f"gifs={len(results)} output={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
