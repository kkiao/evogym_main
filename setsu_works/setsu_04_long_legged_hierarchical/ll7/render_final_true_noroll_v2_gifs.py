"""改良版Level 2制御鎖を九つの提出用GIFへ書き出す。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from ll7.final_true_noroll_v2 import run_final


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "submission_assets" / "level_2_true_no_side_fall_v2" / "gifs"

CLIPS = (
    ("00_first_approach_and_half", {"first_approach", "first_to_50"}),
    ("01_first_safe_clearance", {"first_safe_to_full"}),
    ("02_first_stable_landing", {"first_landing"}),
    ("03_first_restart_forward", {"first_restart"}),
    ("04_second_crossing_to_33", {"second_to_33"}),
    ("05_second_crossing_33_to_50", {"second_to_50"}),
    ("06_second_crossing_50_to_full", {"second_to_full"}),
    ("07_second_landing_and_restart", {"second_landing", "second_restart"}),
)


def overlay(frame, step, stage, info):
    """各フレームへ段階名と厳格状態指標を重ねる。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = (
        f"Level 2 no-side-fall v2 | {stage} | step {step}\n"
        f"angle {math.degrees(float(info['orientation_error'])):.1f} deg | "
        f"speed {float(info['com_speed']):.2f}\n"
        f"clear {info['strict_clearances']} | land {info['stable_landings']} | "
        f"restart {info['restart_successes']} | torso-near-ground "
        f"{bool(info.get('upper_body_grounded', False))}"
    )
    box = draw.textbbox((0, 0), text)
    draw.rectangle((4, 4, box[2] + 18, box[3] + 16), fill="white")
    draw.text((12, 9), text, fill=(15, 15, 15))
    return np.asarray(image)


def save_clip(path, selected, fps):
    """選択画面列を前後静止付きGIFとして保存する。"""
    if not selected:
        raise RuntimeError(f"GIF段階没有画面：{path.name}")
    images = [overlay(frame, step, stage, info) for step, stage, frame, info in selected]
    images = [images[0]] * 4 + images + [images[-1]] * 8
    imageio.mimsave(path, images, fps=fps, loop=0)


def main():
    parser = argparse.ArgumentParser(description="生成改良版 Level 2 九个 GIF。")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    result, frames = run_final(seed=10_000, render_mode="rgb_array", frame_skip=args.frame_skip)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, stages in CLIPS:
        selected = [item for item in frames if item[1] in stages]
        output = OUTPUT_DIR / f"{name}.gif"
        save_clip(output, selected, args.fps)
        entries.append({"file": output.name, "stages": sorted(stages)})
    full_output = OUTPUT_DIR / "08_full_level2_no_side_fall_v2.gif"
    save_clip(full_output, frames, args.fps)
    entries.append({"file": full_output.name, "stages": ["all"]})
    (OUTPUT_DIR / "final_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Level 2 无侧倒姿态优化 v2：9 个关键 GIF",
        "",
        f"两墙任务成功：{result['success']}，优化区间无侧倒：{result['obstacle_control_no_side_fall']}。",
        f"严格越墙/稳定落地/恢复前进/验证 = {result['strict_clearances']}/{result['stable_landings']}/{result['restart_successes']}/{result['validated_obstacles']}。",
        f"优化区间最大倾角 {result['optimized_maximum_degrees']:.2f}°，躯干近地 {result['optimized_contact_steps']} 帧。",
        "",
        "说明：00 保留旧接近与第一墙前半段作为对照，不计入本轮越障后姿态优化指标。",
        "",
    ]
    lines.extend(f"- `{item['file']}`" for item in entries)
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"gifs={len(entries)} output={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
