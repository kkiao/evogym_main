"""統一環境の実通過直後状態から新しい着地回復教師を訓練する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "recovery_runs"
POSITIONS = tuple(range(20, 31))


def safe_flat_handoff(environment: GeneralObstacleEnv, info: dict[str, object]) -> bool:
    """平地歩行方策へ安全に制御を返せる姿勢かを判定する。"""
    velocity_y = float(environment.unwrapped.get_vel_com_obs("robot")[1])
    return bool(
        int(info["recovered_obstacles"]) >= 1
        and float(info["orientation_error"]) <= math.radians(20.0)
        and abs(float(info["angular_velocity"])) <= 0.015
        and abs(velocity_y) <= 2.0
        and not bool(info["upper_body_grounded"])
    )


class RecoveryPrefixEnv(gym.Wrapper):
    """閉ループ前綴で完全通過直後へ進め着地回復だけを学習させる。"""

    def __init__(
        self,
        *,
        seed: int,
        max_recovery_steps: int = 500,
        positions: tuple[int, ...] = POSITIONS,
        prefix_fraction: float = 1.0,
    ) -> None:
        initial_course = build_course(
            ["low_hurdle"],
            split="recovery_teacher_train",
            seed=seed,
            difficulty=1,
            start_runway_voxels=POSITIONS[0],
        )
        environment = GeneralObstacleEnv(
            course=initial_course,
            resample_on_reset=False,
        )
        super().__init__(environment)
        self.base_seed = seed
        self.positions = positions
        self.prefix_fraction = prefix_fraction
        self.episode_index = 0
        self.max_recovery_steps = max_recovery_steps
        self.prefix_teacher = ClosedLoopHeight1Teacher(
            post_clear_mode="restart_then_flat",
            clearance_blend=1.0,
            handoff_distance=0.45,
            adaptive_handoff=True,
        )
        self.recovery_steps = 0
        self.safe_handoff_streak = 0

    def _course(self, position: int, seed: int):
        """指定位置の再現可能な単一低壁コースを返す。"""
        return build_course(
            ["low_hurdle"],
            split="recovery_teacher_train",
            seed=seed,
            difficulty=1,
            start_runway_voxels=position,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """位置を循環し教師前綴で身体全体が越えた直後まで進める。"""
        del options
        episode_seed = self.base_seed + self.episode_index if seed is None else seed
        position = self.positions[self.episode_index % len(self.positions)]
        self.episode_index += 1
        course = self._course(position, int(episode_seed))
        observation, info = self.env.reset(
            seed=int(episode_seed),
            options={"course": course},
        )
        self.prefix_teacher.reset(self.env)
        prefix_steps = 0
        crossed_fraction = 0.0
        while (
            int(info["raw_clearances"]) < 1
            and crossed_fraction < self.prefix_fraction
        ):
            action, _ = self.prefix_teacher.predict(self.env, info)
            observation, _, terminated, truncated, info = self.env.step(action)
            prefix_steps += 1
            positions = self.env.unwrapped.object_pos_at_time(
                self.env.unwrapped.get_time(),
                "robot",
            )
            obstacle = self.env.unwrapped.course.obstacles[0]
            obstacle_end = (obstacle.end_x + 1) * self.env.unwrapped.VOXEL_SIZE
            crossed_fraction = float(np.mean(positions[0] > obstacle_end))
            if terminated or truncated:
                raise RuntimeError(
                    f"闭环前缀未完成越墙：位置={position}，原因={info['failure_reason']}"
                )
        self.recovery_steps = 0
        self.safe_handoff_streak = 0
        info = dict(info)
        info["prefix_steps"] = prefix_steps
        info["recovery_teacher_success"] = False
        return observation, info

    def step(self, action):
        """回復成功を追加報酬化し、短い着地課題として終了する。"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.recovery_steps += 1
        ready = safe_flat_handoff(self.env, info)
        self.safe_handoff_streak = self.safe_handoff_streak + 1 if ready else 0
        success = bool(
            self.safe_handoff_streak >= 20
            or info["course_complete"]
        )

        reward -= 0.002
        reward -= 0.02 * min(1.0, float(info["orientation_error"]) / math.pi)
        if bool(info["upper_body_grounded"]):
            reward -= 0.10
        if bool(info["hard_fall"]):
            reward -= 15.0
        if success:
            reward += 25.0
            terminated = True
            truncated = False
        elif self.recovery_steps >= self.max_recovery_steps and not terminated:
            truncated = True

        info = dict(info)
        info["recovery_steps"] = self.recovery_steps
        info["safe_handoff_streak"] = self.safe_handoff_streak
        info["recovery_teacher_success"] = success
        return observation, float(reward), terminated, truncated, info


def collect_bootstrap_dataset(
    *,
    seed: int,
    positions: tuple[int, ...] = POSITIONS,
    prefix_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """旧回復方策を実状態へ適用し新方策の初期教師データを集める。"""
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rows = []
    for index, position in enumerate(positions):
        course = build_course(
            ["low_hurdle"],
            split="recovery_bootstrap",
            seed=seed + index,
            difficulty=1,
            start_runway_voxels=position,
        )
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        teacher = ClosedLoopHeight1Teacher(
            post_clear_mode="restart_then_flat",
            clearance_blend=1.0,
            handoff_distance=0.45,
            adaptive_handoff=True,
        )
        try:
            observation, info = environment.reset(seed=seed + index)
            teacher.reset(environment)
            crossed_fraction = 0.0
            while (
                int(info["raw_clearances"]) < 1
                and crossed_fraction < prefix_fraction
            ):
                action, _ = teacher.predict(environment, info)
                observation, _, terminated, truncated, info = environment.step(action)
                robot_positions = environment.object_pos_at_time(
                    environment.get_time(),
                    "robot",
                )
                obstacle = environment.course.obstacles[0]
                obstacle_end = (obstacle.end_x + 1) * environment.VOXEL_SIZE
                crossed_fraction = float(
                    np.mean(robot_positions[0] > obstacle_end)
                )
                if terminated or truncated:
                    raise RuntimeError("回復初期化用の越壁前綴が失敗した。")
            safe_streak = 0
            collected = 0
            for _ in range(300):
                action, _ = teacher.predict(environment, info)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                observation, _, terminated, truncated, info = environment.step(action)
                collected += 1
                ready = safe_flat_handoff(environment, info)
                safe_streak = safe_streak + 1 if ready else 0
                if safe_streak >= 20 or terminated or truncated:
                    break
            rows.append(
                {
                    "position": position,
                    "rows": collected,
                    "safe_handoff": safe_streak >= 20,
                    "hard_fall": bool(info["hard_fall"]),
                }
            )
        finally:
            environment.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        {"episodes": rows, "rows": len(observations)},
    )


def behavior_clone(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    epochs: int = 80,
) -> float:
    """新九十五次元方策を実状態上の旧回復動作で初期化する。"""
    rng = np.random.default_rng(seed)
    parameters = list(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    batch_size = 128
    model.policy.train()
    for _ in range(epochs):
        for start in range(0, len(observations), batch_size):
            indices = rng.permutation(len(observations))[start : start + batch_size]
            batch_observations = torch.as_tensor(
                observations[indices],
                device=model.device,
            )
            batch_actions = torch.as_tensor(actions[indices], device=model.device)
            latent = model.policy.mlp_extractor.forward_actor(batch_observations)
            predictions = model.policy.action_net(latent)
            loss = torch.mean((predictions - batch_actions) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        all_observations = torch.as_tensor(observations, device=model.device)
        all_actions = torch.as_tensor(actions, device=model.device)
        latent = model.policy.mlp_extractor.forward_actor(all_observations)
        predictions = model.policy.action_net(latent)
        final_loss = float(torch.mean((predictions - all_actions) ** 2).cpu())
        model.policy.log_std.fill_(-2.5)
        return final_loss


def evaluate_recovery(
    model: PPO,
    *,
    seed: int,
    positions: tuple[int, ...] = POSITIONS,
    prefix_fraction: float = 1.0,
) -> dict[str, object]:
    """三壁位置の通過直後から決定論的回復率を測る。"""
    environment = RecoveryPrefixEnv(
        seed=seed,
        positions=positions,
        prefix_fraction=prefix_fraction,
    )
    rows = []
    try:
        for index, position in enumerate(positions):
            observation, info = environment.reset(seed=seed + index)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
            rows.append(
                {
                    "position": position,
                    "success": bool(info["recovery_teacher_success"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "steps": int(info["recovery_steps"]),
                    "maximum_com_x": float(info["max_x_position"]),
                }
            )
    finally:
        environment.close()
    return {
        "episodes": rows,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "hard_fall_rate": float(np.mean([row["hard_fall"] for row in rows])),
    }


class RecoveryEvaluationCallback(BaseCallback):
    """一定歩数ごとに回復教師を固定三位置で検査する。"""

    def __init__(
        self,
        output_dir: Path,
        every: int,
        seed: int,
        positions: tuple[int, ...],
        prefix_fraction: float,
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.every = every
        self.seed = seed
        self.positions = positions
        self.prefix_fraction = prefix_fraction
        self.next_step = every
        self.best_key = (-1.0, -1.0)
        self.best_step = 0
        self.rows = []

    def evaluate(self, step: int) -> dict[str, object]:
        """現方策を採点し安全優先の最良モデルを保存する。"""
        result = evaluate_recovery(
            self.model,
            seed=self.seed + 10_000,
            positions=self.positions,
            prefix_fraction=self.prefix_fraction,
        )
        row = {
            "step": step,
            "success_rate": result["success_rate"],
            "hard_fall_rate": result["hard_fall_rate"],
        }
        self.rows.append(row)
        csv_path = self.output_dir / "evaluation_5k.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        key = (float(result["success_rate"]), -float(result["hard_fall_rate"]))
        if key > self.best_key:
            self.best_key = key
            self.best_step = step
            self.model.save(self.output_dir / "best_model")
        print(json.dumps({"step": step, **result}, ensure_ascii=False), flush=True)
        return result

    def _on_step(self) -> bool:
        """次の評価境界に達した場合だけ検査を実行する。"""
        if self.num_timesteps >= self.next_step:
            self.model.save(
                self.output_dir / "checkpoints" / f"model_{self.next_step}_steps"
            )
            self.evaluate(self.next_step)
            self.next_step += self.every
        return True


def main() -> None:
    """新しい着地回復教師を模倣初期化後にPPOで改善する。"""
    parser = argparse.ArgumentParser(description="训练统一环境的落地恢复教师。")
    parser.add_argument("--run-name", default="recovery_teacher_seed7_v1")
    parser.add_argument("--total-steps", type=int, default=30_000)
    parser.add_argument("--evaluate-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--positions", nargs="+", type=int, default=list(POSITIONS))
    parser.add_argument("--prefix-fraction", type=float, default=1.0)
    args = parser.parse_args()

    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    positions = tuple(args.positions)
    environment = Monitor(
        RecoveryPrefixEnv(
            seed=args.seed,
            positions=positions,
            prefix_fraction=args.prefix_fraction,
        ),
        filename=str(output_dir / "training.monitor.csv"),
    )
    try:
        model = PPO(
            "MlpPolicy",
            environment,
            learning_rate=1e-5,
            n_steps=512,
            batch_size=64,
            n_epochs=1,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            clip_range=0.05,
            policy_kwargs={"net_arch": [128, 128]},
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
        observations, actions, dataset = collect_bootstrap_dataset(
            seed=args.seed,
            positions=positions,
            prefix_fraction=args.prefix_fraction,
        )
        cloning_loss = behavior_clone(
            model,
            observations,
            actions,
            seed=args.seed,
        )
        initialization = {
            "observation_dimension": 95,
            "dataset": dataset,
            "behavior_cloning_mse": cloning_loss,
            "privileged_inputs_in_student_observation": False,
        }
        (output_dir / "initialization.json").write_text(
            json.dumps(initialization, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model.save(output_dir / "initial_model")
        callback = RecoveryEvaluationCallback(
            output_dir,
            args.evaluate_every,
            args.seed,
            positions,
            args.prefix_fraction,
        )
        callback.init_callback(model)
        callback.evaluate(0)
        if args.total_steps > 0:
            model.learn(
                total_timesteps=args.total_steps,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        model.save(output_dir / "final_model")
        final = callback.evaluate(int(model.num_timesteps))
        summary = {
            "completed_steps": int(model.num_timesteps),
            "best_step": callback.best_step,
            "final": final,
            "teacher_gate_passed": bool(
                float(final["success_rate"]) == 1.0
                and float(final["hard_fall_rate"]) == 0.0
            ),
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
