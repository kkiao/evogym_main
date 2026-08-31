"""最終無側倒Level 2制御鎖を九つの主要段階GIFへ書き出す。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from ll7.final_true_noroll import run_final


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "submission_assets" / "level_2_true_no_side_fall" / "gifs"

CLIPS = (
    ("00_first_controlled_clearance", {"first_approach", "first_clearance"}),
    ("01_first_safe_brake", {"first_brake"}),
    ("02_first_upright_recovery", {"first_righting"}),
    ("03_first_restart", {"first_restart"}),
    ("04_second_crossing_to_33", {"second_approach", "second_to_33"}),
    ("05_second_crossing_33_to_50", {"second_to_50"}),
    ("06_second_crossing_50_to_full", {"second_to_full"}),
    ("07_second_safe_landing", {"second_landing"}),
)


def overlay(frame, step, stage, info):
    """各フレームへ段階名と厳格状態機械の指標を重ねる。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    angle = math.degrees(float(info["orientation_error"]))
    text = (
        f"True no-side-fall Level 2 | {stage} | step {step}\n"
        f"angle {angle:.1f} deg | speed {float(info['com_speed']):.2f}\n"
        f"clear {info['strict_clearances']} | land {info['stable_landings']} | "
        f"restart {info['restart_successes']} | validated {info['validated_obstacles']}"
    )
    box = draw.textbbox((0, 0), text)
    draw.rectangle((4, 4, box[2] + 18, box[3] + 16), fill=(255, 255, 255))
    draw.text((12, 9), text, fill=(15, 15, 15))
    return np.asarray(image)


def save_clip(path, selected, fps):
    """選択済みフレーム列を前後静止付きGIFとして保存する。"""
    if not selected:
        raise RuntimeError(f"GIF段階没有画面：{path.name}")
    frames = [overlay(frame, step, stage, info) for step, stage, frame, info in selected]
    padded = [frames[0]] * 4 + frames + [frames[-1]] * 8
    imageio.mimsave(path, padded, fps=fps, loop=0)


def main():
    parser = argparse.ArgumentParser(description="生成最终无侧倒Level 2九个GIF。")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    result, frames = run_final(
        seed=10_000,
        render_mode="rgb_array",
        frame_skip=args.frame_skip,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, stages in CLIPS:
        selected = [item for item in frames if item[1] in stages]
        output = OUTPUT_DIR / f"{name}.gif"
        save_clip(output, selected, args.fps)
        entries.append({"file": output.name, "stages": sorted(stages)})
    full_output = OUTPUT_DIR / "08_full_level2_true_no_side_fall_success.gif"
    save_clip(full_output, frames, args.fps)
    entries.append({"file": full_output.name, "stages": ["all"]})
    (OUTPUT_DIR / "final_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Level 2 真正无侧倒优化：9个关键阶段 GIF",
        "",
        f"最终结果：成功={result['success']}，真正无侧倒={result['true_no_side_fall']}，",
        f"严格越障/稳定落地/恢复前进/验证障碍 = {result['strict_clearances']}/{result['stable_landings']}/{result['restart_successes']}/{result['validated_obstacles']}，",
        f"全过程最大倾角 {result['overall_maximum_degrees']:.2f}°，上半身触地 {result['overall_contact_steps']} 帧。",
        "",
    ]
    lines.extend(f"- `{index:02d}`：`{item['file']}`" for index, item in enumerate(entries))
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"gifs={len(entries)} output={OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
