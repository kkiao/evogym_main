"""第一壁回復時の前方走査外へ第二壁を置いたM6L学生をGIF化する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from general_terrain.body import BODY_WIDTH_VOXELS
from general_terrain.environment import GeneralObstacleEnv, TERRAIN_LOOK_AHEAD
from general_terrain.render_m6l_best_student_showcase import DEFAULT_MODEL, file_sha256
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts" / "m6l_far_second_hurdle_probe_2026-09-01"
)
FIRST_OBSTACLE_START = 20
SECOND_OBSTACLE_START = 80


def render_far_second_hurdle(
    model: PPO,
    *,
    output_path: Path,
    frame_skip: int,
) -> dict[str, object]:
    """第二壁の初回可視化と各壁の通過回復時刻を記録して再生する。"""
    seed = 200_004
    gap_voxels = SECOND_OBSTACLE_START - FIRST_OBSTACLE_START - 1
    course = build_course(
        ["low_hurdle", "low_hurdle"],
        split="m6l_far_second_hurdle_probe",
        seed=seed,
        difficulty=1,
        gaps=[gap_voxels],
        start_runway_voxels=FIRST_OBSTACLE_START,
    )
    if int(course.obstacles[1].start_x) != SECOND_OBSTACLE_START:
        raise RuntimeError("第二壁の配置位置が要求値と一致しない。")
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array",
    )
    frames: list[np.ndarray] = []
    first_detection_step = -1
    first_detection_x = -1.0
    first_clearance_step = -1
    first_recovery_step = -1
    first_recovery_x = -1.0
    first_recovery_scan_end = -1
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
            center_column = int(
                math.floor(float(info["x_position"]) / environment.VOXEL_SIZE)
            )
            scan_end = center_column + TERRAIN_LOOK_AHEAD
            if first_detection_step < 0 and scan_end >= SECOND_OBSTACLE_START:
                first_detection_step = steps
                first_detection_x = float(info["x_position"])
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            steps += 1
            raw_clearances = int(info["raw_clearances"])
            recoveries = int(info["recovered_obstacles"])
            if first_clearance_step < 0 and raw_clearances >= 1:
                first_clearance_step = steps
            if first_recovery_step < 0 and recoveries >= 1:
                first_recovery_step = steps
                first_recovery_x = float(info["x_position"])
                first_recovery_scan_end = int(
                    math.floor(first_recovery_x / environment.VOXEL_SIZE)
                    + TERRAIN_LOOK_AHEAD
                )
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
        raise RuntimeError("遠隔第二壁回放の画像を取得できなかった。")
    imageio.mimsave(output_path, frames, fps=12, loop=0)
    invisible_at_first_recovery = bool(
        first_recovery_step >= 0
        and first_recovery_scan_end < SECOND_OBSTACLE_START
    )
    return {
        "file": output_path.name,
        "seed": seed,
        "course_id": course.course_id,
        "body_width_voxels": BODY_WIDTH_VOXELS,
        "terrain_look_ahead_voxels": TERRAIN_LOOK_AHEAD,
        "obstacle_start_voxels": [
            int(obstacle.start_x) for obstacle in course.obstacles
        ],
        "gap_voxels": gap_voxels,
        "gap_body_widths": gap_voxels / BODY_WIDTH_VOXELS,
        "steps": steps,
        "frames": len(frames),
        "first_clearance_step": first_clearance_step,
        "first_recovery_step": first_recovery_step,
        "first_recovery_x": first_recovery_x,
        "first_recovery_scan_end_voxel": first_recovery_scan_end,
        "second_obstacle_invisible_at_first_recovery": invisible_at_first_recovery,
        "invisible_margin_voxels_at_first_recovery": (
            SECOND_OBSTACLE_START - first_recovery_scan_end
            if first_recovery_step >= 0
            else -1
        ),
        "first_detection_step": first_detection_step,
        "first_detection_x": first_detection_x,
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


def main() -> None:
    """走査外第二壁の教師なし回放と監査一覧を生成する。"""
    parser = argparse.ArgumentParser(
        description="M6L学生を前方走査外の第二壁で再生する。"
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
    result = render_far_second_hurdle(
        model,
        output_path=output_dir / "01_far_second_hurdle_stall_before_first.gif",
        frame_skip=int(args.frame_skip),
    )
    model_hash_after = file_sha256(model_path)
    if model_hash_after != model_hash_before:
        raise RuntimeError("M6L最良学生モデルが遠隔第二壁試験中に変更された。")
    manifest = {
        "method": "m6l_far_second_hurdle_outside_recovery_scan_probe",
        "model_path": str(model_path),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "model_unchanged": True,
        "training_steps": 0,
        "weight_updates": 0,
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
        "result": result,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
