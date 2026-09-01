"""高さ一の単一障害物で統一方策の小規模能力標定を実行する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
import math

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import CourseSpec, build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_FLAT_TEACHER = (
    REPOSITORY_ROOT
    / "setsu_04_long_legged_hierarchical"
    / "models"
    / "flat_level0_best.zip"
)
EVALUATION_COLUMNS = (
    "step",
    "group",
    "episodes",
    "success_rate",
    "raw_clearance_rate",
    "recovery_rate",
    "hard_fall_rate",
    "mean_max_x",
    "median_success_speed",
    "mean_steps",
)


def evaluation_courses() -> dict[str, tuple[CourseSpec, ...]]:
    """既知二形状と未見幅一形状の固定評価集合を返す。"""
    seen = []
    seed = 20_000
    for name in ("low_hurdle", "low_platform_short"):
        for start in (20, 25, 30):
            seen.append(
                build_course(
                    [name],
                    split="calibration_seen",
                    seed=seed,
                    difficulty=1,
                    start_runway_voxels=start,
                )
            )
            seed += 1
    unseen = tuple(
        build_course(
            ["low_platform_body_width"],
            split="calibration_unseen_width",
            seed=21_000 + index,
            difficulty=1,
            start_runway_voxels=start,
        )
        for index, start in enumerate((20, 25, 30))
    )
    return {"seen_height1": tuple(seen), "unseen_width": unseen}


class FixedCourseEvaluator:
    """同一チェックポイントを固定コース群で決定論的に採点する。"""

    def __init__(self, course_groups: dict[str, tuple[CourseSpec, ...]]) -> None:
        self.environments = {
            group: [
                GeneralObstacleEnv(course=course, resample_on_reset=False)
                for course in courses
            ]
            for group, courses in course_groups.items()
        }

    def close(self) -> None:
        """全評価環境のビューアと物理器を閉じる。"""
        for environments in self.environments.values():
            for environment in environments:
                environment.close()

    def evaluate_group(self, model: PPO, group: str) -> dict[str, float | int | str]:
        """一評価群の成功、安全、進捗、速度を集計する。"""
        episodes = []
        for index, environment in enumerate(self.environments[group]):
            observation, info = environment.reset(seed=30_000 + index)
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
            displacement = float(info["forward_displacement"])
            duration = max(steps / 50.0, 1e-6)
            episodes.append(
                {
                    "success": bool(info["course_complete"]),
                    "raw_clearance": int(info["raw_clearances"]) >= 1,
                    "recovery": int(info["recovered_obstacles"]) >= 1,
                    "hard_fall": bool(info["hard_fall"]),
                    "max_x": float(info["max_x_position"]),
                    "speed": displacement / duration,
                    "steps": steps,
                }
            )
        count = len(episodes)
        success_speeds = [row["speed"] for row in episodes if row["success"]]
        return {
            "group": group,
            "episodes": count,
            "success_rate": sum(row["success"] for row in episodes) / count,
            "raw_clearance_rate": sum(row["raw_clearance"] for row in episodes) / count,
            "recovery_rate": sum(row["recovery"] for row in episodes) / count,
            "hard_fall_rate": sum(row["hard_fall"] for row in episodes) / count,
            "mean_max_x": float(np.mean([row["max_x"] for row in episodes])),
            "median_success_speed": float(median(success_speeds)) if success_speeds else 0.0,
            "mean_steps": float(np.mean([row["steps"] for row in episodes])),
        }

    def evaluate(self, model: PPO, step: int) -> list[dict[str, float | int | str]]:
        """全評価群を採点し学習歩数を付加する。"""
        rows = []
        for group in self.environments:
            row = self.evaluate_group(model, group)
            row["step"] = step
            rows.append(row)
        return rows


def score_key(rows: list[dict[str, float | int | str]]) -> tuple[float, ...]:
    """安全成功を速度より優先するチェックポイント順位を返す。"""
    by_group = {str(row["group"]): row for row in rows}
    seen = by_group["seen_height1"]
    unseen = by_group["unseen_width"]
    return (
        float(seen["success_rate"]),
        float(seen["recovery_rate"]),
        float(seen["raw_clearance_rate"]),
        -float(seen["hard_fall_rate"]),
        float(unseen["success_rate"]),
        float(seen["mean_max_x"]),
    )


def flat_teacher_baseline_observation() -> np.ndarray:
    """旧平地環境の初期観測を定数入力折畳み用に取得する。"""
    from ll7.body import make_body as make_legacy_body
    from ll7.curriculum import get_course
    from ll7.experiment import make_env

    course = get_course(0)
    environment = make_env(make_legacy_body(), 0, course.max_steps)
    try:
        observation, _ = environment.reset(seed=7)
        return np.asarray(observation, dtype=np.float32)
    finally:
        environment.close()


def initialize_from_flat_teacher(model: PPO, teacher_path: Path) -> dict[str, object]:
    """旧特権入力を除外し共通身体制御重みだけを学生へ移植する。"""
    if not teacher_path.exists():
        raise FileNotFoundError(f"找不到平地教师模型：{teacher_path}")
    teacher = PPO.load(teacher_path, device="cpu")
    if teacher.observation_space.shape != (97,):
        raise ValueError("平地教师观测维度不是预期的97。")
    if model.observation_space.shape != (95,):
        raise ValueError("统一学生观测维度不是预期的95。")
    baseline = flat_teacher_baseline_observation()
    teacher_state = teacher.policy.state_dict()
    student_state = model.policy.state_dict()

    with torch.no_grad():
        for network_name in ("policy_net", "value_net"):
            first_weight_key = f"mlp_extractor.{network_name}.0.weight"
            first_bias_key = f"mlp_extractor.{network_name}.0.bias"
            old_weight = teacher_state[first_weight_key]
            new_weight = torch.zeros_like(student_state[first_weight_key])
            new_weight[:, 0:2] = old_weight[:, 0:2]
            new_weight[:, 2] = old_weight[:, 2]
            new_weight[:, 5:53] = old_weight[:, 3:51]
            for offset in range(-5, 21):
                old_floor_index = 51 + (offset + 20)
                new_terrain_index = 53 + (offset + 5)
                new_weight[:, new_terrain_index] = old_weight[:, old_floor_index]
            constant_floor_indices = list(range(51, 66))
            constant_input = torch.as_tensor(
                np.concatenate((baseline[constant_floor_indices], baseline[92:97])),
                dtype=old_weight.dtype,
            )
            constant_weight = torch.cat(
                (old_weight[:, constant_floor_indices], old_weight[:, 92:97]),
                dim=1,
            )
            new_bias = teacher_state[first_bias_key] + constant_weight @ constant_input
            student_state[first_weight_key] = new_weight
            student_state[first_bias_key] = new_bias

        transferable = (
            "log_std",
            "mlp_extractor.policy_net.2.weight",
            "mlp_extractor.policy_net.2.bias",
            "mlp_extractor.value_net.2.weight",
            "mlp_extractor.value_net.2.bias",
            "action_net.weight",
            "action_net.bias",
            "value_net.weight",
            "value_net.bias",
        )
        for key in transferable:
            student_state[key] = teacher_state[key].clone()
        model.policy.load_state_dict(student_state)

    return {
        "teacher_path": str(teacher_path.resolve()),
        "teacher_observation_shape": list(teacher.observation_space.shape),
        "student_observation_shape": list(model.observation_space.shape),
        "copied_common_inputs": ["com_velocity", "orientation_sine", "relative_body_points"],
        "mapped_dynamic_terrain_offsets": [-5, 20],
        "new_inputs_initialized_to_zero": ["orientation_cosine", "terrain_offsets_21_to_30"],
        "privileged_phase_inputs_copied": False,
        "teacher_floor_and_phase_constants_folded_into_bias": True,
    }


def legacy_surface_heights(course) -> np.ndarray:
    """旧コース定義から各列の地表高さを体素数で返す。"""
    heights = np.ones(course.width, dtype=np.float32)
    for obstacle in course.obstacles:
        for offset, height in enumerate(obstacle.heights):
            heights[obstacle.start_x + offset] += float(height)
    return heights


def student_observation_from_legacy_course(
    environment,
    upright_reference: float,
    surface_heights: np.ndarray,
    angular_velocity: float,
    previous_action: np.ndarray,
) -> np.ndarray:
    """旧物理状態と地表輪郭を新しい非特権観測形式へ変換する。"""
    base = environment.unwrapped
    positions = base.object_pos_at_time(base.get_time(), "robot")
    com_x = float(np.mean(positions[0]))
    com_y = float(np.mean(positions[1]))
    orientation = base.object_orientation_at_time(base.get_time(), "robot")
    angle = math.atan2(
        math.sin(orientation - upright_reference),
        math.cos(orientation - upright_reference),
    )
    center_column = int(math.floor(com_x / base.VOXEL_SIZE))
    terrain = []
    for offset in range(-5, 31):
        column = center_column + offset
        if column < 0 or column >= len(surface_heights):
            terrain.append(0.5)
        else:
            surface_y = float(surface_heights[column]) * base.VOXEL_SIZE
            terrain.append(float(np.clip(com_y - surface_y, -0.5, 0.5)))
    return np.concatenate(
        (
            base.get_vel_com_obs("robot"),
            np.asarray([math.sin(angle), math.cos(angle)]),
            np.asarray([angular_velocity]),
            base.get_relative_pos_obs("robot"),
            np.asarray(terrain),
            np.asarray(previous_action),
        )
    ).astype(np.float32)


def augment_distant_height1_terrain(
    observation: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """遠方の高さ一輪郭を加え歩行教師の位置記憶を弱める。"""
    augmented = observation.copy()
    start_offset = int(rng.integers(8, 28))
    width = int(rng.choice((1, 3, 5)))
    for offset in range(start_offset, min(31, start_offset + width)):
        index = 53 + (offset + 5)
        augmented[index] = float(np.clip(augmented[index] - 0.1, -0.5, 0.5))
    return augmented


def collect_flat_teacher_dataset(
    teacher: PPO,
    *,
    seed: int,
    augmentations_per_state: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """一つの安定歩行回放から地形拡張付き模倣データを作る。"""
    from ll7.body import make_body as make_legacy_body
    from ll7.curriculum import get_course
    from ll7.experiment import make_env

    course = get_course(0)
    surface_heights = legacy_surface_heights(course)
    environment = make_env(make_legacy_body(), 0, course.max_steps)
    observations = []
    actions = []
    rng = np.random.default_rng(seed)
    try:
        teacher_observation, info = environment.reset(seed=seed)
        upright_reference = environment.unwrapped.object_orientation_at_time(
            environment.unwrapped.get_time(),
            "robot",
        )
        previous_action = np.zeros(6, dtype=np.float32)
        for _ in range(course.max_steps):
            action, _ = teacher.predict(teacher_observation, deterministic=True)
            student_observation = student_observation_from_legacy_course(
                environment,
                upright_reference,
                surface_heights,
                float(info.get("angular_speed", 0.0)),
                previous_action,
            )
            observations.append(student_observation)
            actions.append(np.asarray(action, dtype=np.float32))
            for _ in range(augmentations_per_state):
                observations.append(
                    augment_distant_height1_terrain(student_observation, rng)
                )
                actions.append(np.asarray(action, dtype=np.float32))
            teacher_observation, _, terminated, truncated, info = environment.step(action)
            previous_action = np.asarray(action, dtype=np.float32)
            if terminated or truncated:
                break
    finally:
        environment.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def collect_safe_obstacle_teacher_dataset(
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """再現可能な二低壁成功制御鎖から新観測と教師行動を収集する。"""
    from ll7.body import make_body as make_legacy_body
    from ll7.curriculum import get_course
    from ll7.experiment import make_env
    from ll7.final_true_noroll_v2 import (
        ACTION_SCALES,
        load_models,
        select_stage,
    )

    models = load_models()
    course = get_course(2)
    surface_heights = legacy_surface_heights(course)
    environment = make_env(make_legacy_body(), 2, course.max_steps)
    observations = []
    actions = []
    second_handoff_started = False
    second_stage = "second_to_33"
    try:
        teacher_observation, info = environment.reset(seed=seed)
        upright_reference = environment.unwrapped.object_orientation_at_time(
            environment.unwrapped.get_time(),
            "robot",
        )
        previous_action = np.zeros(6, dtype=np.float32)
        for _ in range(course.max_steps):
            stage, second_handoff_started, second_stage = select_stage(
                info,
                course,
                environment,
                second_handoff_started,
                second_stage,
            )
            model_key = (
                "approach"
                if stage in {"first_approach", "second_approach"}
                else stage
            )
            action, _ = models[model_key].predict(
                teacher_observation,
                deterministic=True,
            )
            scaled_action = ACTION_SCALES.get(stage, 1.0) * action
            observations.append(
                student_observation_from_legacy_course(
                    environment,
                    upright_reference,
                    surface_heights,
                    float(info.get("angular_speed", 0.0)),
                    previous_action,
                )
            )
            actions.append(np.asarray(scaled_action, dtype=np.float32))
            teacher_observation, _, terminated, truncated, info = environment.step(
                scaled_action
            )
            previous_action = np.asarray(scaled_action, dtype=np.float32)
            if terminated or truncated:
                break
    finally:
        environment.close()
    metadata = {
        "rows": len(observations),
        "teacher_success": bool(info["is_success"]),
        "strict_clearances": int(info["strict_clearances"]),
        "stable_landings": int(info["stable_landings"]),
        "restart_successes": int(info["restart_successes"]),
    }
    if not metadata["teacher_success"]:
        raise RuntimeError("低墙教师轨迹未能复现成功，停止蒸馏。")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        metadata,
    )


def distill_flat_teacher(
    model: PPO,
    teacher_path: Path,
    *,
    seed: int,
    epochs: int = 40,
    include_obstacle_teacher: bool = False,
) -> dict[str, object]:
    """平地歩行を初期化し、必要な場合だけ旧越障軌跡も混合する。"""
    if not teacher_path.exists():
        raise FileNotFoundError(f"找不到平地教师模型：{teacher_path}")
    teacher = PPO.load(teacher_path, device="cpu")
    flat_observations, flat_actions = collect_flat_teacher_dataset(teacher, seed=seed)
    obstacle_repeat = 0
    obstacle_metadata: dict[str, object] = {
        "rows": 0,
        "teacher_success": None,
    }
    if include_obstacle_teacher:
        obstacle_observations, obstacle_actions, obstacle_metadata = (
            collect_safe_obstacle_teacher_dataset(seed=10_000)
        )
        obstacle_repeat = 2
        observations = np.concatenate(
            (
                flat_observations,
                np.repeat(obstacle_observations, obstacle_repeat, axis=0),
            ),
            axis=0,
        )
        actions = np.concatenate(
            (flat_actions, np.repeat(obstacle_actions, obstacle_repeat, axis=0)),
            axis=0,
        )
    else:
        observations = flat_observations
        actions = flat_actions
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(observations))
    validation_count = max(1, len(indices) // 10)
    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]
    parameters = list(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    batch_size = 256
    model.policy.train()
    for _ in range(epochs):
        shuffled = rng.permutation(training_indices)
        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start : start + batch_size]
            batch_observations = torch.as_tensor(
                observations[batch_indices],
                device=model.device,
            )
            batch_actions = torch.as_tensor(
                actions[batch_indices],
                device=model.device,
            )
            latent = model.policy.mlp_extractor.forward_actor(batch_observations)
            predictions = model.policy.action_net(latent)
            loss = torch.mean((predictions - batch_actions) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        validation_observations = torch.as_tensor(
            observations[validation_indices],
            device=model.device,
        )
        validation_actions = torch.as_tensor(
            actions[validation_indices],
            device=model.device,
        )
        validation_latent = model.policy.mlp_extractor.forward_actor(
            validation_observations
        )
        validation_predictions = model.policy.action_net(validation_latent)
        validation_loss = float(
            torch.mean((validation_predictions - validation_actions) ** 2).cpu()
        )
        model.policy.log_std.copy_(teacher.policy.log_std)
    return {
        "mode": (
            "flat_and_safe_low_wall_behavior_cloning"
            if include_obstacle_teacher
            else "flat_gait_behavior_cloning"
        ),
        "teacher_path": str(teacher_path.resolve()),
        "dataset_rows": int(len(observations)),
        "flat_teacher_rows": int(len(flat_observations)),
        "safe_obstacle_teacher_rows": int(obstacle_metadata["rows"]),
        "safe_obstacle_repeat": obstacle_repeat,
        "safe_obstacle_teacher": obstacle_metadata,
        "training_rows": int(len(training_indices)),
        "validation_rows": int(len(validation_indices)),
        "epochs": epochs,
        "validation_action_mse": validation_loss,
        "student_observation_shape": list(model.observation_space.shape),
        "teacher_observation_copied_to_student": False,
        "privileged_phase_inputs_copied": False,
        "terrain_augmentation": "height1_at_relative_offsets_8_to_30",
    }


class PilotEvaluationCallback(BaseCallback):
    """五千歩ごとに固定評価、表追記、チェックポイント保存を行う。"""

    def __init__(
        self,
        evaluator: FixedCourseEvaluator,
        output_dir: Path,
        evaluate_every: int,
    ) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.evaluate_every = evaluate_every
        self.next_evaluation = evaluate_every
        self.best_key: tuple[float, ...] | None = None
        self.best_step = 0
        self.rows: list[dict[str, float | int | str]] = []
        self.csv_path = output_dir / "evaluation_5k.csv"

    def _write_rows(self, rows: list[dict[str, float | int | str]]) -> None:
        """新しい評価行をCSVへ追記する。"""
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=EVALUATION_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def run_evaluation(self, step: int) -> list[dict[str, float | int | str]]:
        """現方策を評価し安全優先の最良モデルを更新する。"""
        rows = self.evaluator.evaluate(self.model, step)
        self.rows.extend(rows)
        self._write_rows(rows)
        key = score_key(rows)
        if self.best_key is None or key > self.best_key:
            self.best_key = key
            self.best_step = step
            self.model.save(self.output_dir / "best_model")
        print(json.dumps({"evaluation": rows, "best_step": self.best_step}, ensure_ascii=False), flush=True)
        return rows

    def _on_step(self) -> bool:
        """予定歩数へ到達した時だけ評価を発火する。"""
        if self.num_timesteps >= self.next_evaluation:
            step = self.next_evaluation
            self.model.save(self.output_dir / "checkpoints" / f"model_{step}_steps")
            self.run_evaluation(step)
            self.next_evaluation += self.evaluate_every
        return True


def main() -> None:
    """高さ一の小規模PPO標定を実行して全成果を新規フォルダへ保存する。"""
    parser = argparse.ArgumentParser(description="运行高度1单障碍小规模能力标定。")
    parser.add_argument("--run-name", default="height1_single_pilot_seed7")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--evaluate-every", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-flat-teacher", action="store_true")
    parser.add_argument(
        "--teacher-mode",
        choices=("distill-flat", "distill-mixed", "transplant"),
        default="distill-flat",
    )
    args = parser.parse_args()

    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    config = {
        "run_name": args.run_name,
        "total_steps": args.total_steps,
        "evaluate_every": args.evaluate_every,
        "seed": args.seed,
        "device": "cpu",
        "training_distribution": {
            "split": "train",
            "difficulty": 1,
            "obstacle_count": 1,
            "templates": ["low_hurdle", "low_platform_short"],
            "start_runway_voxels": [20, 30],
        },
        "formal_training": False,
        "purpose": "height1_single_obstacle_capability_calibration",
        "flat_teacher_initialization": not args.no_flat_teacher,
        "teacher_mode": "none" if args.no_flat_teacher else args.teacher_mode,
        "learning_rate": 1e-4,
        "n_steps": 2_048,
        "batch_size": 64,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "ent_coef": 0.005,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    base_environment = GeneralObstacleEnv(
        split="train",
        difficulty=1,
        obstacle_count=1,
        base_seed=args.seed,
        resample_on_reset=True,
    )
    training_environment = Monitor(
        base_environment,
        filename=str(output_dir / "training.monitor.csv"),
    )
    evaluator = FixedCourseEvaluator(evaluation_courses())
    try:
        model = PPO(
            "MlpPolicy",
            training_environment,
            learning_rate=1e-4,
            n_steps=2_048,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.005,
            policy_kwargs={"net_arch": [128, 128]},
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
        if args.no_flat_teacher:
            initialization = {"flat_teacher_initialization": False}
        elif args.teacher_mode in {"distill-flat", "distill-mixed"}:
            initialization = distill_flat_teacher(
                model,
                DEFAULT_FLAT_TEACHER,
                seed=args.seed,
                include_obstacle_teacher=args.teacher_mode == "distill-mixed",
            )
        else:
            initialization = initialize_from_flat_teacher(model, DEFAULT_FLAT_TEACHER)
        (output_dir / "initialization.json").write_text(
            json.dumps(initialization, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model.save(output_dir / "initial_model")
        callback = PilotEvaluationCallback(
            evaluator,
            output_dir,
            args.evaluate_every,
        )
        callback.init_callback(model)
        callback.run_evaluation(0)
        model.learn(
            total_timesteps=args.total_steps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        model.save(output_dir / "final_model")
        final_rows = callback.run_evaluation(int(model.num_timesteps))
        summary = {
            "completed_steps": int(model.num_timesteps),
            "best_step": callback.best_step,
            "best_score_key": list(callback.best_key or ()),
            "final_evaluation": final_rows,
            "capability_signal": bool(
                score_key(final_rows)[0] >= 0.5
                or score_key(final_rows)[1] >= 0.5
            ),
            "formal_gate_passed": bool(
                score_key(final_rows)[0] >= 0.9
                and score_key(final_rows)[3] >= -0.01
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2), flush=True)
    finally:
        evaluator.close()
        training_environment.close()


if __name__ == "__main__":
    main()
