"""抗摂動越壁方策の通過直後を閉ループ教師で救回し分岐として保存する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX_MODEL = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_height1_runs"
    / "noisy_height1_demo_seed7_v3"
    / "best_model.zip"
)
OUTPUT_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "noisy_rescue_teacher"
POSITIONS = tuple(range(20, 31))


def course(position: int, seed: int):
    """検収用の高さ一単壁コースを返す。"""
    return build_course(
        ["low_hurdle"],
        split="noisy_rescue_teacher_validation",
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )


def crossed_fraction(environment: GeneralObstacleEnv) -> float:
    """全身体質点のうち壁後端を越えた割合を返す。"""
    base = environment.unwrapped
    positions = base.object_pos_at_time(base.get_time(), "robot")
    obstacle = base.course.obstacles[0]
    obstacle_end = (obstacle.end_x + 1) * base.VOXEL_SIZE
    return float(np.mean(positions[0] > obstacle_end))


def run_episode(
    prefix_model: PPO,
    rescue_teacher: PortfolioHeight1Teacher,
    *,
    position: int,
    seed: int,
    noise_std: float,
    noise_probability: float,
    rescue_fraction: float,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    """越壁までは学生候補を使い、完全通過後だけ教師に救回させる。"""
    environment = GeneralObstacleEnv(
        course=course(position, seed),
        resample_on_reset=False,
    )
    rng = np.random.default_rng(seed + 5_000_000)
    observations = []
    actions = []
    modes = []
    rescue_started = False
    rescue_start_step: int | None = None
    disturbance_count = 0
    maximum_angle = 0.0
    upper_contact_steps = 0
    try:
        observation, info = environment.reset(seed=seed)
        rescue_teacher.reset(environment)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            if crossed_fraction(environment) >= rescue_fraction:
                if not rescue_started:
                    rescue_started = True
                    rescue_start_step = steps
                action, stage = rescue_teacher.predict(
                    environment,
                    observation,
                    info,
                )
                mode = f"rescue:{stage}"
            else:
                action, _ = prefix_model.predict(observation, deterministic=True)
                mode = "prefix"
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(np.asarray(action, dtype=np.float32))
            modes.append(mode)
            executed_action = np.asarray(action, dtype=np.float32)
            if rng.random() < noise_probability:
                executed_action = np.clip(
                    executed_action
                    + rng.normal(0.0, noise_std, size=executed_action.shape),
                    -1.0,
                    1.0,
                ).astype(np.float32)
                disturbance_count += 1
            observation, _, terminated, truncated, info = environment.step(
                executed_action
            )
            steps += 1
            maximum_angle = max(maximum_angle, float(info["orientation_error"]))
            upper_contact_steps += int(bool(info["upper_body_grounded"]))
    finally:
        environment.close()
    result = {
        "position": position,
        "seed": seed,
        "steps": steps,
        "disturbance_count": disturbance_count,
        "rescue_fraction": rescue_fraction,
        "rescue_started": rescue_started,
        "rescue_start_step": rescue_start_step,
        "course_complete": bool(info["course_complete"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_angle_degrees": math.degrees(maximum_angle),
        "upper_body_contact_steps": upper_contact_steps,
        "maximum_com_x": float(info["max_x_position"]),
    }
    branch = None
    if result["course_complete"] and not result["hard_fall"]:
        branch = {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "modes": np.asarray(modes),
            "rescue_start_step": np.asarray([rescue_start_step], dtype=np.int32),
        }
    return result, branch


def main() -> None:
    """十一位置を検査し成功救回分岐と厳格門判定を保存する。"""
    parser = argparse.ArgumentParser(description="验收受扰动后由教师救回的高度1组合。")
    parser.add_argument("--run-name", default="noisy_rescue_seed7_v1")
    parser.add_argument("--prefix-model", default=str(DEFAULT_PREFIX_MODEL))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    parser.add_argument("--rescue-fraction", type=float, default=1.0)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "branches").mkdir()
    prefix_model = PPO.load(Path(args.prefix_model), device="cpu")
    rescue_teacher = PortfolioHeight1Teacher()
    episodes = []
    for index, position in enumerate(POSITIONS):
        result, branch = run_episode(
            prefix_model,
            rescue_teacher,
            position=position,
            seed=args.seed + 70_000 + index,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
            rescue_fraction=args.rescue_fraction,
        )
        episodes.append(result)
        if branch is not None:
            np.savez_compressed(
                output_dir / "branches" / f"x{position}_rescued.npz",
                **branch,
            )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    success_count = sum(item["course_complete"] for item in episodes)
    hard_fall_count = sum(item["hard_fall"] for item in episodes)
    summary = {
        "prefix_model": str(Path(args.prefix_model).resolve()),
        "noise_std": args.noise_std,
        "noise_probability": args.noise_probability,
        "rescue_fraction": args.rescue_fraction,
        "episodes": episodes,
        "success_count": success_count,
        "hard_fall_count": hard_fall_count,
        "robustness_gate_passed": bool(
            success_count >= 9 and hard_fall_count == 0
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
