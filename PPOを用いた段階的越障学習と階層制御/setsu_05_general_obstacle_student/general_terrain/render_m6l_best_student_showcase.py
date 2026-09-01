"""M6L最良学生の代表的な教師なし挙動をGIFへ保存する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.train_m5_reverse_curriculum import flat_retention_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models"
    / "m6l_best_student_boundary_dependent.zip"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "student_showcase_recheck"
)


def file_sha256(path: Path) -> str:
    """指定ファイルのSHA-256を逐次読み込みで返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_episode(
    model: PPO,
    *,
    seed: int,
    output_path: Path,
    frame_skip: int,
    flat: bool = False,
) -> dict[str, object]:
    """固定コースを学生だけで再生し、判定値とGIFを返す。"""
    course = (
        flat_retention_course(seed)
        if flat
        else sample_curriculum_course(seed, "hurdle_single", "validation")
    )
    environment = GeneralObstacleEnv(
        course=course,
        resample_on_reset=False,
        render_mode="rgb_array",
    )
    frames: list[np.ndarray] = []
    try:
        observation, info = environment.reset(seed=seed)
        initial_x = float(info["x_position"])
        environment.default_viewer.set_target_rps(None)
        environment.default_viewer.set_resolution((600, 300))
        first_frame = environment.render()
        if first_frame is not None:
            frames.extend([np.asarray(first_frame).copy()] * 6)
        terminated = False
        truncated = False
        steps = 0
        flat_success = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            steps += 1
            if flat and float(info["x_position"]) - initial_x >= 2.4:
                flat_success = not bool(info["hard_fall"])
                terminated = True
                truncated = False
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
        raise RuntimeError("M6L学生回放の画像を取得できなかった。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=12, loop=0)
    return {
        "file": output_path.name,
        "seed": seed,
        "split": "flat_retention" if flat else "validation",
        "course_id": course.course_id,
        "start_runway_voxels": int(course.obstacles[0].start_x),
        "steps": steps,
        "frames": len(frames),
        "course_complete": bool(info["course_complete"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "hard_fall": bool(info["hard_fall"]),
        "stall_limit_reached": bool(info["stall_limit_reached"]),
        "flat_success": flat_success,
        "failure_reason": str(info["failure_reason"]),
        "maximum_com_x": float(info["max_x_position"]),
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
    }


def main() -> None:
    """成功、落地失敗、停止、平地保持の四回放をまとめて生成する。"""
    parser = argparse.ArgumentParser(
        description="M6L最良学生の教師なし代表GIFを生成する。"
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
    specifications = (
        ("01_validation_x20_complete.gif", 200_004, False),
        ("02_validation_x23_clear_then_hard_fall.gif", 200_018, False),
        ("03_validation_x21_stall_before_wall.gif", 200_000, False),
        ("04_flat_retention_success.gif", 840_001, True),
    )
    episodes = [
        render_episode(
            model,
            seed=seed,
            output_path=output_dir / name,
            frame_skip=int(args.frame_skip),
            flat=flat,
        )
        for name, seed, flat in specifications
    ]
    model_hash_after = file_sha256(model_path)
    if model_hash_after != model_hash_before:
        raise RuntimeError("M6L最良学生モデルが回放中に変更された。")
    manifest = {
        "method": "m6l_best_student_teacher_free_showcase",
        "model_path": str(model_path),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "model_unchanged": True,
        "training_steps": 0,
        "weight_updates": 0,
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
        "resolution": [600, 300],
        "fps": 12,
        "frame_skip": int(args.frame_skip),
        "episodes": episodes,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
