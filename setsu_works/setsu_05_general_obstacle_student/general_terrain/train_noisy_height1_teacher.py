"""抗摂動平地方策を壁前状態から高さ一通過へ継続訓練する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.recurrent_prototype_student import SOURCE_DATASET
from general_terrain.terrain import build_course
from general_terrain.train_noisy_flat_teacher import (
    TARGET_X,
    distant_wall_course,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLAT_MODEL = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_flat_runs"
    / "noisy_flat_seed7_v1"
    / "best_model.zip"
)
RUNS_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "noisy_height1_runs"
POSITIONS = tuple(range(20, 31))
TEACHER_VALIDATION = (
    PROJECT_ROOT / "training_only_teacher" / "generated" / "portfolio_height1_teacher_v1" / "validation.json"
)


def course(position: int, seed: int, split: str):
    """指定位置の高さ一単壁コースを返す。"""
    return build_course(
        ["low_hurdle"],
        split=split,
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )


def wall_distance(environment: GeneralObstacleEnv) -> float:
    """質心から壁前端までの現在相対距離を返す。"""
    base = environment.unwrapped
    positions = base.object_pos_at_time(base.get_time(), "robot")
    start = base.course.obstacles[0].start_x * base.VOXEL_SIZE
    return float(start - np.mean(positions[0]))


class NearWallCurriculumEnv(gym.Wrapper):
    """抗摂動平地教師で壁前へ進めた後だけ学習方策へ制御を渡す。"""

    def __init__(
        self,
        *,
        flat_model_path: Path,
        seed: int,
        noise_std: float,
        noise_probability: float,
        handoff_distance: float = 0.5,
        max_control_steps: int = 800,
    ) -> None:
        environment = GeneralObstacleEnv(
            course=course(POSITIONS[0], seed, "noisy_height1_train"),
            resample_on_reset=False,
        )
        super().__init__(environment)
        self.flat_model = PPO.load(flat_model_path, device="cpu")
        self.base_seed = seed
        self.noise_std = noise_std
        self.noise_probability = noise_probability
        self.handoff_distance = handoff_distance
        self.max_control_steps = max_control_steps
        self.episode_index = 0
        self.control_steps = 0
        self.disturbance_count = 0
        self.rng = np.random.default_rng(seed)
        self.previous_raw_clearances = 0
        self.previous_recovered = 0
        self.maximum_bottom_y = 0.0
        self.maximum_crossed_fraction = 0.0
        self.maximum_rear_progress = 0.0
        self.previous_x = 0.0
        self.previous_orientation_error = 0.0

    def _disturb(self, action: np.ndarray) -> np.ndarray:
        """設定確率で動作へ小さな独立摂動を加える。"""
        result = np.asarray(action, dtype=np.float32)
        if self.rng.random() < self.noise_probability:
            result = np.clip(
                result + self.rng.normal(0.0, self.noise_std, size=result.shape),
                -1.0,
                1.0,
            ).astype(np.float32)
            self.disturbance_count += 1
        return result

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """位置を循環し抗摂動平地方策で安全な壁前状態まで進める。"""
        del options
        actual_seed = self.base_seed + self.episode_index if seed is None else seed
        position = POSITIONS[self.episode_index % len(POSITIONS)]
        self.episode_index += 1
        observation, info = self.env.reset(
            seed=int(actual_seed),
            options={
                "course": course(position, int(actual_seed), "noisy_height1_train")
            },
        )
        self.rng = np.random.default_rng(int(actual_seed) + 4_000_000)
        self.disturbance_count = 0
        prefix_steps = 0
        while wall_distance(self.env) > self.handoff_distance:
            action, _ = self.flat_model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = self.env.step(
                self._disturb(action)
            )
            prefix_steps += 1
            if terminated or truncated:
                raise RuntimeError(
                    f"抗摂動助走が壁前到達前に失敗した：{info['failure_reason']}"
                )
        self.control_steps = 0
        self.previous_raw_clearances = int(info["raw_clearances"])
        self.previous_recovered = int(info["recovered_obstacles"])
        positions = self.env.unwrapped.object_pos_at_time(
            self.env.unwrapped.get_time(),
            "robot",
        )
        _, crossed_fraction, rear_progress, bottom_y = (
            self.env.unwrapped._local_rise_metrics(positions)
        )
        self.maximum_bottom_y = bottom_y
        self.maximum_crossed_fraction = crossed_fraction
        self.maximum_rear_progress = rear_progress
        self.previous_x = float(np.mean(positions[0]))
        self.previous_orientation_error = float(info["orientation_error"])
        info = dict(info)
        info["prefix_steps"] = prefix_steps
        info["curriculum_position"] = position
        info["action_disturbance_count"] = self.disturbance_count
        return observation, info

    def step(self, action):
        """越壁進捗、安全回復、完走へ段階報酬を追加する。"""
        observation, reward, terminated, truncated, info = self.env.step(
            self._disturb(action)
        )
        self.control_steps += 1
        raw_clearances = int(info["raw_clearances"])
        recovered = int(info["recovered_obstacles"])
        positions = self.env.unwrapped.object_pos_at_time(
            self.env.unwrapped.get_time(),
            "robot",
        )
        current_x = float(np.mean(positions[0]))
        _, crossed_fraction, rear_progress, bottom_y = (
            self.env.unwrapped._local_rise_metrics(positions)
        )
        bottom_gain = max(0.0, bottom_y - self.maximum_bottom_y)
        crossing_gain = max(0.0, crossed_fraction - self.maximum_crossed_fraction)
        rear_gain = max(0.0, rear_progress - self.maximum_rear_progress)
        forward_gain = max(0.0, current_x - self.previous_x)
        orientation_error = float(info["orientation_error"])
        crossing_active = crossed_fraction >= 0.10 or raw_clearances >= 1
        angle_scale = np.deg2rad(45.0)
        reward += 60.0 * bottom_gain
        reward += 20.0 * crossing_gain
        reward += 30.0 * rear_gain
        reward += 10.0 * forward_gain
        if crossing_active:
            angle_ratio = orientation_error / angle_scale
            reward -= 0.15 * angle_ratio**2
            reward += 3.0 * (self.previous_orientation_error - orientation_error)
            if orientation_error <= np.deg2rad(20.0):
                reward += 0.03
        if raw_clearances > self.previous_raw_clearances:
            reward += 4.0
        if recovered > self.previous_recovered:
            reward += 50.0
        if bool(info["course_complete"]):
            reward += 100.0
        if bool(info["hard_fall"]):
            reward -= 100.0
        if bool(info["upper_body_grounded"]):
            reward -= 2.0
        self.previous_raw_clearances = raw_clearances
        self.previous_recovered = recovered
        self.maximum_bottom_y = max(self.maximum_bottom_y, bottom_y)
        self.maximum_crossed_fraction = max(
            self.maximum_crossed_fraction,
            crossed_fraction,
        )
        self.maximum_rear_progress = max(self.maximum_rear_progress, rear_progress)
        self.previous_x = current_x
        self.previous_orientation_error = orientation_error
        if self.control_steps >= self.max_control_steps and not terminated:
            reward -= 10.0
            truncated = True
        info = dict(info)
        info["control_steps"] = self.control_steps
        info["action_disturbance_count"] = self.disturbance_count
        info["height1_teacher_success"] = bool(info["course_complete"])
        return observation, float(reward), terminated, truncated, info


def evaluate(
    model: PPO,
    *,
    seed: int,
    noise_std: float,
    noise_probability: float,
) -> dict[str, object]:
    """学生と同じ全行程連続摂動条件で十一位置を評価する。"""
    episodes = []
    for index, position in enumerate(POSITIONS):
        environment = GeneralObstacleEnv(
            course=course(position, seed + index, "noisy_height1_validation"),
            resample_on_reset=False,
        )
        rng = np.random.default_rng(seed + index + 5_000_000)
        disturbance_count = 0
        try:
            observation, info = environment.reset(seed=seed + index)
            terminated = False
            truncated = False
            steps = 0
            maximum_angle = 0.0
            upper_contact_steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                if rng.random() < noise_probability:
                    action = np.clip(
                        action
                        + rng.normal(0.0, noise_std, size=np.asarray(action).shape),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                    disturbance_count += 1
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                maximum_angle = max(maximum_angle, float(info["orientation_error"]))
                upper_contact_steps += int(bool(info["upper_body_grounded"]))
            episodes.append(
                {
                    "position": position,
                    "steps": steps,
                    "disturbance_count": disturbance_count,
                    "course_complete": bool(info["course_complete"]),
                    "raw_clearances": int(info["raw_clearances"]),
                    "recovered_obstacles": int(info["recovered_obstacles"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "failure_reason": str(info["failure_reason"]),
                    "maximum_angle_degrees": float(np.degrees(maximum_angle)),
                    "upper_body_contact_steps": upper_contact_steps,
                    "maximum_com_x": float(info["max_x_position"]),
                }
            )
        finally:
            environment.close()
    success_count = sum(item["course_complete"] for item in episodes)
    clearance_count = sum(item["raw_clearances"] >= 1 for item in episodes)
    hard_fall_count = sum(item["hard_fall"] for item in episodes)
    return {
        "episodes": episodes,
        "success_count": success_count,
        "success_rate": success_count / len(episodes),
        "clearance_count": clearance_count,
        "clearance_rate": clearance_count / len(episodes),
        "hard_fall_count": hard_fall_count,
        "hard_fall_rate": hard_fall_count / len(episodes),
        "mean_max_x": float(np.mean([item["maximum_com_x"] for item in episodes])),
    }


def append_progress(path: Path, row: dict[str, object]) -> None:
    """五千歩ごとの全行程指標をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def collect_flat_anchor_dataset(
    model: PPO,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """抗摂動平地方策自身の訪問状態と決定論的動作を保存する。"""
    observations = []
    actions = []
    for episode in range(6):
        environment = GeneralObstacleEnv(
            course=distant_wall_course(seed + episode),
            resample_on_reset=False,
            max_episode_steps=1_200,
        )
        rng = np.random.default_rng(seed + episode + 6_000_000)
        try:
            observation, info = environment.reset(seed=seed + episode)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                executed = np.asarray(action, dtype=np.float32)
                if rng.random() < 0.01:
                    executed = np.clip(
                        executed
                        + rng.normal(0.0, 0.01, size=executed.shape),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                observation, _, terminated, truncated, info = environment.step(executed)
                if float(info["max_x_position"]) >= TARGET_X:
                    break
        finally:
            environment.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def collect_obstacle_demo_dataset() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """成功教師軌跡から壁前四十歩以降の越壁と回復区間だけを抽出する。"""
    validation = json.loads(TEACHER_VALIDATION.read_text(encoding="utf-8"))
    observations = []
    actions = []
    rows = []
    for episode in validation["episodes"]:
        position = int(episode["position"])
        data = np.load(SOURCE_DATASET / f"x{position}.npz")
        trajectory_observations = np.asarray(data["observations"], dtype=np.float32)
        trajectory_actions = np.asarray(data["actions"], dtype=np.float32)
        first_wall_stage = next(
            int(item["step"])
            for item in episode["events"]
            if not str(item["stage"]).endswith(":flat")
        )
        start = max(0, first_wall_stage - 40)
        observations.append(trajectory_observations[start:])
        actions.append(trajectory_actions[start:])
        rows.append(
            {
                "position": position,
                "start_step": start,
                "rows": len(trajectory_observations) - start,
            }
        )
    return (
        np.concatenate(observations),
        np.concatenate(actions),
        {"episodes": rows, "rows": int(sum(item["rows"] for item in rows))},
    )


def demonstration_initialize(
    model: PPO,
    *,
    seed: int,
    epochs: int = 20,
) -> dict[str, object]:
    """平地錨を保持しながら成功越壁区間の動作で方策を予熱する。"""
    flat_observations, flat_actions = collect_flat_anchor_dataset(model, seed=seed)
    obstacle_observations, obstacle_actions, obstacle_metadata = (
        collect_obstacle_demo_dataset()
    )
    obstacle_repeat = 3
    observations = np.concatenate(
        (flat_observations, np.repeat(obstacle_observations, obstacle_repeat, axis=0))
    )
    actions = np.concatenate(
        (flat_actions, np.repeat(obstacle_actions, obstacle_repeat, axis=0))
    )
    rng = np.random.default_rng(seed)
    parameters = list(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-4, weight_decay=1e-6)
    batch_size = 256
    losses = []
    model.policy.train()
    for _ in range(epochs):
        batch_losses = []
        permutation = rng.permutation(len(observations))
        for start in range(0, len(observations), batch_size):
            indices = permutation[start : start + batch_size]
            batch_observations = torch.as_tensor(
                observations[indices],
                dtype=torch.float32,
                device=model.device,
            )
            batch_actions = torch.as_tensor(
                actions[indices],
                dtype=torch.float32,
                device=model.device,
            )
            latent = model.policy.mlp_extractor.forward_actor(batch_observations)
            predictions = model.policy.action_net(latent)
            loss = torch.mean((predictions - batch_actions) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(batch_losses)))
    with torch.no_grad():
        model.policy.log_std.fill_(-0.5)
    return {
        "flat_anchor_rows": len(flat_observations),
        "obstacle_demo": obstacle_metadata,
        "obstacle_repeat": obstacle_repeat,
        "total_rows": len(observations),
        "epochs": epochs,
        "first_loss": losses[0],
        "final_loss": losses[-1],
    }


def main() -> None:
    """抗摂動平地重みから高さ一教師を継続学習し安全最良モデルを保存する。"""
    parser = argparse.ArgumentParser(description="从抗扰平地继续训练95维高度1教师。")
    parser.add_argument("--run-name", default="noisy_height1_seed7_v1")
    parser.add_argument("--flat-model", default=str(DEFAULT_FLAT_MODEL))
    parser.add_argument("--initial-model", default=None)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--evaluate-every", type=int, default=5_000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    parser.add_argument("--demo-initialize", action="store_true")
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    training_environment = Monitor(
        NearWallCurriculumEnv(
            flat_model_path=Path(args.flat_model),
            seed=args.seed,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        ),
        filename=str(output_dir / "training.monitor.csv"),
    )
    try:
        initial_model_path = Path(args.initial_model or args.flat_model)
        model = PPO.load(initial_model_path, env=training_environment, device="cpu")
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
        model.ent_coef = 0.001
        with np.errstate(all="ignore"):
            model.policy.log_std.data.fill_(-0.5)
        initialization = None
        if args.demo_initialize:
            initialization = demonstration_initialize(model, seed=args.seed)
            (output_dir / "initialization.json").write_text(
                json.dumps(initialization, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            model.save(output_dir / "demo_initialized_model")
        initial = evaluate(
            model,
            seed=args.seed + 70_000,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        )
        best_key = (
            initial["success_rate"],
            initial["clearance_rate"],
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
                "clearance_rate": initial["clearance_rate"],
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
                seed=args.seed + 70_000,
                noise_std=args.noise_std,
                noise_probability=args.noise_probability,
            )
            row = {"step": completed, **result}
            evaluations.append(row)
            model.save(output_dir / "checkpoints" / f"model_{completed}_steps")
            key = (
                result["success_rate"],
                result["clearance_rate"],
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
                    "clearance_rate": result["clearance_rate"],
                    "hard_fall_rate": result["hard_fall_rate"],
                    "mean_max_x": result["mean_max_x"],
                },
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if result["success_count"] >= 9 and result["hard_fall_count"] == 0:
                break
        summary = {
            "completed_steps": completed,
            "best_step": best_step,
            "best_score": list(best_key),
            "noise_std": args.noise_std,
            "noise_probability": args.noise_probability,
            "demonstration_initialization": initialization,
            "evaluations": evaluations,
            "robustness_gate_passed": bool(
                best_key[0] >= 9.0 / 11.0 and best_key[2] == 0.0
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
