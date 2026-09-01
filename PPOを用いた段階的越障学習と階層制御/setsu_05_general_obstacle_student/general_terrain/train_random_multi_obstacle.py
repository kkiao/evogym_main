"""高さ一安全門の後に難度一ランダム二障害コースを継続訓練する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import sample_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INITIAL_MODEL = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_height1_runs"
    / "noisy_height1_demo_seed7_v3"
    / "best_model.zip"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "random_multi_obstacle_training"


class RandomMultiObstacleEnv(gym.Wrapper):
    """ランダム二障害へ摂動、安全、段階通過報酬を追加する。"""

    def __init__(
        self,
        *,
        seed: int,
        obstacle_count: int,
        noise_std: float,
        noise_probability: float,
    ) -> None:
        environment = GeneralObstacleEnv(
            split="train",
            difficulty=1,
            obstacle_count=obstacle_count,
            base_seed=seed,
            resample_on_reset=True,
        )
        super().__init__(environment)
        self.base_seed = seed
        self.episode_index = 0
        self.noise_std = noise_std
        self.noise_probability = noise_probability
        self.rng = np.random.default_rng(seed)
        self.previous_raw_clearances = 0
        self.previous_recovered = 0
        self.disturbance_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """各回で新しい再現可能コースと摂動列を選ぶ。"""
        actual_seed = self.base_seed + self.episode_index if seed is None else seed
        self.episode_index += 1
        observation, info = self.env.reset(seed=int(actual_seed), options=options)
        self.rng = np.random.default_rng(int(actual_seed) + 9_000_000)
        self.previous_raw_clearances = int(info["raw_clearances"])
        self.previous_recovered = int(info["recovered_obstacles"])
        self.disturbance_count = 0
        return observation, info

    def step(self, action):
        """独立動作摂動を加え複数障害の安全回復を強く報酬化する。"""
        executed_action = np.asarray(action, dtype=np.float32)
        if self.rng.random() < self.noise_probability:
            executed_action = np.clip(
                executed_action
                + self.rng.normal(0.0, self.noise_std, size=executed_action.shape),
                -1.0,
                1.0,
            ).astype(np.float32)
            self.disturbance_count += 1
        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        raw_clearances = int(info["raw_clearances"])
        recovered = int(info["recovered_obstacles"])
        if raw_clearances > self.previous_raw_clearances:
            reward += 8.0
        if recovered > self.previous_recovered:
            reward += 30.0
        if bool(info["course_complete"]):
            reward += 100.0
        orientation_error = float(info["orientation_error"])
        if orientation_error > math.radians(35.0):
            reward -= 0.10 * (orientation_error / math.radians(35.0)) ** 2
        if bool(info["upper_body_grounded"]):
            reward -= 2.0
        if bool(info["hard_fall"]):
            reward -= 100.0
        self.previous_raw_clearances = raw_clearances
        self.previous_recovered = recovered
        info = dict(info)
        info["action_disturbance_count"] = self.disturbance_count
        return observation, float(reward), terminated, truncated, info


def evaluate(
    model: PPO,
    *,
    seed: int,
    obstacle_count: int,
    noise_std: float,
    noise_probability: float,
) -> dict[str, object]:
    """未再標本化の十一ランダムコースで完走、回復、側倒を測る。"""
    episodes = []
    for index in range(11):
        episode_seed = seed + index
        course = sample_course(
            episode_seed,
            difficulty=1,
            obstacle_count=obstacle_count,
            split="train",
        )
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        rng = np.random.default_rng(episode_seed + 9_000_000)
        disturbance_count = 0
        maximum_angle = 0.0
        upper_contact_steps = 0
        try:
            observation, info = environment.reset(seed=episode_seed)
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                if rng.random() < noise_probability:
                    action = np.clip(
                        action + rng.normal(0.0, noise_std, size=np.asarray(action).shape),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                    disturbance_count += 1
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                maximum_angle = max(maximum_angle, float(info["orientation_error"]))
                upper_contact_steps += int(bool(info["upper_body_grounded"]))
        finally:
            environment.close()
        episodes.append(
            {
                "course_id": course.course_id,
                "templates": [item.template.name for item in course.obstacles],
                "steps": steps,
                "disturbance_count": disturbance_count,
                "course_complete": bool(info["course_complete"]),
                "raw_clearances": int(info["raw_clearances"]),
                "recovered_obstacles": int(info["recovered_obstacles"]),
                "hard_fall": bool(info["hard_fall"]),
                "failure_reason": str(info["failure_reason"]),
                "maximum_angle_degrees": math.degrees(maximum_angle),
                "upper_body_contact_steps": upper_contact_steps,
                "maximum_com_x": float(info["max_x_position"]),
            }
        )
    return {
        "episodes": episodes,
        "success_count": sum(item["course_complete"] for item in episodes),
        "hard_fall_count": sum(item["hard_fall"] for item in episodes),
        "mean_raw_clearances": float(
            np.mean([item["raw_clearances"] for item in episodes])
        ),
        "mean_recovered_obstacles": float(
            np.mean([item["recovered_obstacles"] for item in episodes])
        ),
        "mean_max_x": float(np.mean([item["maximum_com_x"] for item in episodes])),
    }


def append_progress(path: Path, row: dict[str, object]) -> None:
    """五千歩ごとのランダムコース指標をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def score(result: dict[str, object]) -> tuple[int, int, float, float]:
    """完走、安全回復数、到達距離で最良モデルを比較する。"""
    return (
        int(result["success_count"]),
        -int(result["hard_fall_count"]),
        float(result["mean_recovered_obstacles"]),
        float(result["mean_max_x"]),
    )


def main() -> None:
    """高さ一候補から難度一ランダム二障害PPOを開始する。"""
    parser = argparse.ArgumentParser(description="训练随机双障碍统一学生。")
    parser.add_argument("--run-name", default="random_d1_two_obstacles_seed7_v1")
    parser.add_argument("--initial-model", default=str(DEFAULT_INITIAL_MODEL))
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--evaluate-every", type=int, default=5_000)
    parser.add_argument("--obstacle-count", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    environment = Monitor(
        RandomMultiObstacleEnv(
            seed=args.seed,
            obstacle_count=args.obstacle_count,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        ),
        filename=str(output_dir / "training.monitor.csv"),
    )
    try:
        model = PPO.load(Path(args.initial_model), env=environment, device="cpu")
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.ent_coef = 0.001
        initial = evaluate(
            model,
            seed=args.seed + 90_000,
            obstacle_count=args.obstacle_count,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        )
        best_score = score(initial)
        best_step = 0
        best_result = initial
        model.save(output_dir / "best_model")
        history = [{"step": args.step_offset, **initial}]
        append_progress(
            output_dir / "evaluation_5k.csv",
            {
                "step": args.step_offset,
                "success_count": initial["success_count"],
                "hard_fall_count": initial["hard_fall_count"],
                "mean_raw_clearances": initial["mean_raw_clearances"],
                "mean_recovered_obstacles": initial["mean_recovered_obstacles"],
                "mean_max_x": initial["mean_max_x"],
            },
        )
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        completed = args.step_offset
        while completed < args.total_steps:
            chunk = min(args.evaluate_every, args.total_steps - completed)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            completed += chunk
            result = evaluate(
                model,
                seed=args.seed + 90_000,
                obstacle_count=args.obstacle_count,
                noise_std=args.noise_std,
                noise_probability=args.noise_probability,
            )
            row = {"step": completed, **result}
            history.append(row)
            model.save(output_dir / "checkpoints" / f"model_{completed}_steps")
            candidate_score = score(result)
            if candidate_score > best_score:
                best_score = candidate_score
                best_step = completed
                best_result = result
                model.save(output_dir / "best_model")
            append_progress(
                output_dir / "evaluation_5k.csv",
                {
                    "step": completed,
                    "success_count": result["success_count"],
                    "hard_fall_count": result["hard_fall_count"],
                    "mean_raw_clearances": result["mean_raw_clearances"],
                    "mean_recovered_obstacles": result["mean_recovered_obstacles"],
                    "mean_max_x": result["mean_max_x"],
                },
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
        summary = {
            "difficulty": 1,
            "obstacle_count": args.obstacle_count,
            "completed_steps": completed,
            "best_step": best_step,
            "best_score": list(best_score),
            "best_evaluation": best_result,
            "history": history,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
