"""M6L最良学生を位置二十の単壁と三体幅間隔の二壁で比較描画する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from general_terrain.body import BODY_WIDTH_VOXELS
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.render_m6l_best_student_showcase import (
    DEFAULT_MODEL,
    file_sha256,
    render_episode,
)
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "m6l_double_hurdle_reproduction_2026-09-01"
)


def render_double_hurdle(
    model: PPO,
    *,
    output_path: Path,
    frame_skip: int,
) -> dict[str, object]:
    """位置二十から三体幅間隔の同一低壁二つを学生だけで走行する。"""
    seed = 200_004
    gap_voxels = 3 * BODY_WIDTH_VOXELS
    course = build_course(
        ["low_hurdle", "low_hurdle"],
        split="m6l_double_hurdle_probe",
        seed=seed,
        difficulty=1,
        gaps=[gap_voxels],
        start_runway_voxels=20,
    )
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array",
    )
    frames: list[np.ndarray] = []
    first_clearance_step = -1
    first_recovery_step = -1
    second_clearance_step = -1
    second_recovery_step = -1
    try:
        observation, info = environment.reset(seed=seed)
        environment.default_viewer.set_target_rps(None)
        environment.default_viewer.set_resolution((600, 300))
        first_frame = environment.render()
        if first_frame is not None:
            frames.extend([np.asarray(first_frame).copy()] * 6)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            steps += 1
            raw_clearances = int(info["raw_clearances"])
            recoveries = int(info["recovered_obstacles"])
            if first_clearance_step < 0 and raw_clearances >= 1:
                first_clearance_step = steps
            if first_recovery_step < 0 and recoveries >= 1:
                first_recovery_step = steps
            if second_clearance_step < 0 and raw_clearances >= 2:
                second_clearance_step = steps
            if second_recovery_step < 0 and recoveries >= 2:
                second_recovery_step = steps
            if steps % frame_skip == 0 or terminated or truncated:
                frame = environment.render()
                if frame is not None:
                    frames.append(np.asarray(frame).copy())
        final_frame = environment.render()
        if final_frame is not None:
            frames.extend([np.asarray(final_frame).copy()] * 12)
    finally:
        environment.close()
    if not frames:
        raise RuntimeError("二壁回放の画像を取得できなかった。")
    imageio.mimsave(output_path, frames, fps=12, loop=0)
    return {
        "file": output_path.name,
        "seed": seed,
        "course_id": course.course_id,
        "body_width_voxels": BODY_WIDTH_VOXELS,
        "gap_body_widths": 3,
        "gap_voxels": gap_voxels,
        "obstacle_start_voxels": [
            int(obstacle.start_x) for obstacle in course.obstacles
        ],
        "steps": steps,
        "frames": len(frames),
        "first_clearance_step": first_clearance_step,
        "first_recovery_step": first_recovery_step,
        "second_clearance_step": second_clearance_step,
        "second_recovery_step": second_recovery_step,
        "course_complete": bool(info["course_complete"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "hard_fall": bool(info["hard_fall"]),
        "stall_limit_reached": bool(info["stall_limit_reached"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_com_x": float(info["max_x_position"]),
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
    }


def interpretation(result: dict[str, object]) -> str:
    """二壁結果を第一壁再現性と第二壁能力へ分類する。"""
    if bool(result["course_complete"]):
        return "double_hurdle_complete"
    if int(result["recovered_obstacles"]) >= 2:
        return "both_hurdles_recovered_but_course_incomplete"
    if int(result["raw_clearances"]) >= 2:
        return "both_hurdles_cleared_without_second_recovery"
    if int(result["recovered_obstacles"]) >= 1:
        return "first_hurdle_reproduced_and_recovered_second_failed"
    if int(result["raw_clearances"]) >= 1:
        return "first_hurdle_cleared_without_recovery"
    return "single_hurdle_success_not_reproduced_in_double_course"


def main() -> None:
    """単壁再現と二壁三体幅試験を同一モデルでGIF化する。"""
    parser = argparse.ArgumentParser(
        description="M6L最良学生の位置二十単壁と二壁を比較する。"
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame-skip", type=int, default=5)
    args = parser.parse_args()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model_hash_before = file_sha256(model_path)
    model = PPO.load(model_path, device="cpu")
    single_result = render_episode(
        model,
        seed=200_004,
        output_path=output_dir / "01_single_x20_reproduction.gif",
        frame_skip=int(args.frame_skip),
    )
    double_result = render_double_hurdle(
        model,
        output_path=output_dir / "02_double_x20_gap_3_body_widths.gif",
        frame_skip=int(args.frame_skip),
    )
    model_hash_after = file_sha256(model_path)
    if model_hash_after != model_hash_before:
        raise RuntimeError("M6L最良学生モデルが二壁試験中に変更された。")
    manifest = {
        "method": "m6l_x20_double_hurdle_three_body_gap_probe",
        "model_path": str(model_path),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "model_unchanged": True,
        "training_steps": 0,
        "weight_updates": 0,
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
        "single_x20_reproduction": single_result,
        "double_hurdle_probe": double_result,
        "interpretation": interpretation(double_result),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
