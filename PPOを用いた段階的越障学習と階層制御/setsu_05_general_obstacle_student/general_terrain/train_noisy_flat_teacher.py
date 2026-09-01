"""連続小摂動下でも助走を維持する九十五次元平地教師を訓練する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.height1_pilot import DEFAULT_FLAT_TEACHER, distill_flat_teacher
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "noisy_flat_runs"
TARGET_X = 3.0


def distant_wall_course(seed: int):
    """通常の学習時間内では壁へ到達しない長い平地コースを返す。"""
    return build_course(
        ["low_hurdle"],
        split="noisy_flat_teacher",
        seed=seed,
        difficulty=1,
        start_runway_voxels=80,
    )


class ActionNoiseWrapper(gym.Wrapper):
    """指定確率で正規化動作へ独立ガウス摂動を加える。"""

    def __init__(
        self,
        environment: gym.Env,
        *,
        seed: int,
        noise_std: float,
        noise_probability: float,
    ) -> None:
        super().__init__(environment)
        self.base_seed = seed
        self.noise_std = noise_std
        self.noise_probability = noise_probability
        self.rng = np.random.default_rng(seed)
        self.disturbance_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """回ごとに摂動乱数と回数を再初期化する。"""
        actual_seed = self.base_seed if seed is None else seed
        self.rng = np.random.default_rng(actual_seed + 3_000_000)
        self.disturbance_count = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """動作摂動を適用し診断情報へ累積回数を追加する。"""
        executed = np.asarray(action, dtype=np.float32)
        if self.rng.random() < self.noise_probability:
            executed = np.clip(
                executed
                + self.rng.normal(0.0, self.noise_std, size=executed.shape),
                -1.0,
                1.0,
            ).astype(np.float32)
            self.disturbance_count += 1
        observation, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        info["action_disturbance_count"] = self.disturbance_count
        return observation, reward, terminated, truncated, info


def make_environment(
    *,
    seed: int,
    noise_std: float,
    noise_probability: float,
) -> ActionNoiseWrapper:
    """長平地、固定時間上限、動作摂動を持つ一環境を生成する。"""
    base = GeneralObstacleEnv(
        course=distant_wall_course(seed),
        resample_on_reset=False,
        max_episode_steps=1_200,
    )
    return ActionNoiseWrapper(
        base,
        seed=seed,
        noise_std=noise_std,
        noise_probability=noise_probability,
    )


def evaluate(
    model: PPO,
    *,
    seed: int,
    noise_std: float,
    noise_probability: float,
) -> dict[str, object]:
    """十一独立摂動列で目標距離到達と無側倒を検査する。"""
    episodes = []
    for index in range(11):
        environment = make_environment(
            seed=seed + index,
            noise_std=noise_std,
            noise_probability=noise_probability,
        )
        try:
            observation, info = environment.reset(seed=seed + index)
            terminated = False
            truncated = False
            steps = 0
            reached_target = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                if float(info["max_x_position"]) >= TARGET_X:
                    reached_target = True
                    break
            episodes.append(
                {
                    "episode": index,
                    "steps": steps,
                    "reached_target": reached_target,
                    "hard_fall": bool(info["hard_fall"]),
                    "failure_reason": str(info["failure_reason"]),
                    "maximum_com_x": float(info["max_x_position"]),
                    "disturbance_count": int(
                        info.get("action_disturbance_count", 0)
                    ),
                }
            )
        finally:
            environment.close()
    success_count = sum(
        item["reached_target"] and not item["hard_fall"] for item in episodes
    )
    hard_fall_count = sum(item["hard_fall"] for item in episodes)
    return {
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "hard_fall_count": hard_fall_count,
        "hard_fall_rate": hard_fall_count / len(episodes),
        "mean_max_x": float(np.mean([item["maximum_com_x"] for item in episodes])),
    }


def append_progress(path: Path, row: dict[str, object]) -> None:
    """五千歩評価をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """平地模倣初期化後に摂動付きPPO更新を行い最良教師を保存する。"""
    parser = argparse.ArgumentParser(description="训练95维连续小扰动平地教师。")
    parser.add_argument("--run-name", default="noisy_flat_seed7_v1")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--evaluate-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    training_environment = Monitor(
        make_environment(
            seed=args.seed,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        ),
        filename=str(output_dir / "training.monitor.csv"),
    )
    try:
        model = PPO(
            "MlpPolicy",
            training_environment,
            learning_rate=5e-5,
            n_steps=512,
            batch_size=128,
            n_epochs=3,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.0,
            clip_range=0.1,
            policy_kwargs={"net_arch": [128, 128]},
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
        initialization = distill_flat_teacher(
            model,
            DEFAULT_FLAT_TEACHER,
            seed=args.seed,
            epochs=50,
            include_obstacle_teacher=False,
        )
        (output_dir / "initialization.json").write_text(
            json.dumps(initialization, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model.save(output_dir / "initial_model")
        initial = evaluate(
            model,
            seed=args.seed + 60_000,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        )
        best_key = (
            initial["success_rate"],
            -initial["hard_fall_rate"],
            initial["mean_max_x"],
        )
        best_step = 0
        model.save(output_dir / "best_model")
        evaluations = [{"step": 0, **initial}]
        append_progress(
            output_dir / "evaluation_5k.csv",
            {
                "step": 0,
                "success_rate": initial["success_rate"],
                "hard_fall_rate": initial["hard_fall_rate"],
                "mean_max_x": initial["mean_max_x"],
            },
        )
        print(json.dumps(evaluations[-1], ensure_ascii=False), flush=True)

        completed = 0
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
                seed=args.seed + 60_000,
                noise_std=args.noise_std,
                noise_probability=args.noise_probability,
            )
            row = {"step": completed, **result}
            evaluations.append(row)
            model.save(output_dir / "checkpoints" / f"model_{completed}_steps")
            key = (
                result["success_rate"],
                -result["hard_fall_rate"],
                result["mean_max_x"],
            )
            if key > best_key:
                best_key = key
                best_step = completed
                model.save(output_dir / "best_model")
            append_progress(
                output_dir / "evaluation_5k.csv",
                {
                    "step": completed,
                    "success_rate": result["success_rate"],
                    "hard_fall_rate": result["hard_fall_rate"],
                    "mean_max_x": result["mean_max_x"],
                },
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
        summary = {
            "completed_steps": completed,
            "best_step": best_step,
            "best_score": list(best_key),
            "noise_std": args.noise_std,
            "noise_probability": args.noise_probability,
            "target_x": TARGET_X,
            "evaluations": evaluations,
            "flat_robustness_gate_passed": bool(
                best_key[0] >= 9.0 / 11.0 and best_key[1] == 0.0
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        training_environment.close()


if __name__ == "__main__":
    main()
