"""M6Sで学生成功軌跡を錨にし越え前から安全な落地を訓練する。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import torch

from general_terrain.audit_m6_contact_bridge import actor_predictions
from general_terrain.curriculum import get_curriculum_stage, sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_only_evaluation import evaluate_student_batch
from general_terrain.train_m5_reverse_curriculum import (
    array_sha256,
    flat_retention_course,
    resolve_project_path,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "m6s_student_success_safety_protocol_v1.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "m6s_student_success_safety"
TRAINING_MODES = (
    "failed_prelanding",
    "successful_prelanding",
    "full_start",
    "flat",
)


@dataclass(frozen=True)
class M6SProtocol:
    """学生自体の成功錨、早期接管点、検査点上限を保持する。"""

    source_path: Path
    source_model_path: Path
    source_model_sha256: str
    source_summary_path: Path
    source_summary_sha256: str
    seed_manifest_path: Path
    seed_manifest_sha256: str
    device: str
    parallel_environments: int
    preclearance_offsets: tuple[int, ...]
    training_weights: dict[str, float]
    maximum_student_steps_per_episode: int
    checkpoint_interval_steps: int
    total_steps: int
    success_anchor_repeat: int
    flat_anchor_stride: int
    anchor_learning_rate: float
    anchor_epochs_per_checkpoint: int
    anchor_maximum_gradient_norm: float
    minimum_validation_raw_clearances: int
    minimum_flat_successes: int


@dataclass(frozen=True)
class PrelandingResetSpec:
    """学生の生越え前から制御を渡す一つの再生仕様を保持する。"""

    reset_id: str
    seed: int
    start_runway_voxels: int
    course_id: str
    prefix_steps: int
    clearance_step: int
    offset_before_clearance: int
    source_success: bool
    branch_path: Path
    branch_sha256: str
    expected_observation_sha256: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存できる辞書を返す。"""
        payload = asdict(self)
        payload["branch_path"] = str(self.branch_path.resolve())
        return payload


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> M6SProtocol:
    """M6S規約、学生のみ境界、全出典ハッシュを検査する。"""
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M6S規約は凍結済みでなければならない。")
    if str(payload["device"]) != "cpu" or int(payload["parallel_environments"]) != 8:
        raise ValueError("M6S正式訓練はCPU八並列に固定する。")
    if int(payload["teacher_actions_used"]) != 0:
        raise ValueError("M6Sでは教師動作を使用できない。")
    if bool(payload["validation_teacher_enabled"]):
        raise ValueError("M6S検証では教師を有効化できない。")
    if bool(payload["holdout_teacher_enabled"]):
        raise ValueError("M6S留出では教師を有効化できない。")
    if int(payload["holdout_episodes"]) != 0:
        raise ValueError("M6Sは留出区分へアクセスできない。")
    paths = {
        "source_model": resolve_project_path(str(payload["source_model_path"])),
        "source_summary": resolve_project_path(str(payload["source_summary_path"])),
        "seed_manifest": resolve_project_path(str(payload["seed_manifest_path"])),
    }
    for name, protected_path in paths.items():
        if sha256_file(protected_path) != str(payload[f"{name}_sha256"]):
            raise ValueError(f"M6S出典ハッシュが一致しない: {protected_path}")
    source_summary = json.loads(paths["source_summary"].read_text(encoding="utf-8"))
    if int(source_summary["best_student_validation"]["success_count"]) != 1:
        raise ValueError("M6S源学生は教師なし検証1成功でなければならない。")
    offsets = tuple(int(value) for value in payload["preclearance_offsets"])
    if offsets != tuple(sorted(offsets, reverse=True)) or min(offsets) < 1:
        raise ValueError("M6S越え前オフセットは正の降順でなければならない。")
    weights = {
        str(key): float(value) for key, value in payload["training_weights"].items()
    }
    if set(weights) != set(TRAINING_MODES) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("M6S訓練混合が不正である。")
    interval = int(payload["checkpoint_interval_steps"])
    total_steps = int(payload["total_steps"])
    if total_steps < interval * 3 or total_steps % interval != 0:
        raise ValueError("M6Sは三検査点以上かつ間隔の整数倍でなければならない。")
    return M6SProtocol(
        source_path=source_path,
        source_model_path=paths["source_model"],
        source_model_sha256=str(payload["source_model_sha256"]),
        source_summary_path=paths["source_summary"],
        source_summary_sha256=str(payload["source_summary_sha256"]),
        seed_manifest_path=paths["seed_manifest"],
        seed_manifest_sha256=str(payload["seed_manifest_sha256"]),
        device=str(payload["device"]),
        parallel_environments=int(payload["parallel_environments"]),
        preclearance_offsets=offsets,
        training_weights=weights,
        maximum_student_steps_per_episode=int(
            payload["maximum_student_steps_per_episode"]
        ),
        checkpoint_interval_steps=interval,
        total_steps=total_steps,
        success_anchor_repeat=int(payload["success_anchor_repeat"]),
        flat_anchor_stride=int(payload["flat_anchor_stride"]),
        anchor_learning_rate=float(payload["anchor_learning_rate"]),
        anchor_epochs_per_checkpoint=int(payload["anchor_epochs_per_checkpoint"]),
        anchor_maximum_gradient_norm=float(
            payload["anchor_maximum_gradient_norm"]
        ),
        minimum_validation_raw_clearances=int(
            payload["minimum_validation_raw_clearances"]
        ),
        minimum_flat_successes=int(payload["minimum_flat_successes"]),
    )


def collect_source_trajectories(
    model: PPO,
    train_seeds: tuple[int, ...],
    branches_dir: Path,
) -> tuple[dict[str, object], ...]:
    """訓練区分の源学生全軌跡を観測と動作ごと凍結する。"""
    branches_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in train_seeds:
        course = sample_curriculum_course(seed, "hurdle_single", "train")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        observations = []
        actions = []
        raw_clearance_step = -1
        recovery_step = -1
        try:
            observation, info = environment.reset(seed=seed)
            observations.append(np.asarray(observation, dtype=np.float32))
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                actions.append(np.asarray(action, dtype=np.float32))
                observation, _, terminated, truncated, info = environment.step(action)
                observations.append(np.asarray(observation, dtype=np.float32))
                if raw_clearance_step < 0 and int(info["raw_clearances"]) >= 1:
                    raw_clearance_step = len(actions)
                if recovery_step < 0 and int(info["recovered_obstacles"]) >= 1:
                    recovery_step = len(actions)
        finally:
            environment.close()
        branch_path = branches_dir / f"seed_{seed}_source_student.npz"
        np.savez_compressed(
            branch_path,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
        )
        rows.append(
            {
                "seed": seed,
                "course_id": course.course_id,
                "start_runway_voxels": int(course.obstacles[0].start_x),
                "steps": len(actions),
                "raw_clearance_step": raw_clearance_step,
                "recovery_step": recovery_step,
                "course_complete": bool(info["course_complete"]),
                "hard_fall": bool(info["hard_fall"]),
                "raw_clearances": int(info["raw_clearances"]),
                "recovered_obstacles": int(info["recovered_obstacles"]),
                "failure_reason": str(info["failure_reason"]),
                "branch_path": str(branch_path.resolve()),
                "branch_sha256": sha256_file(branch_path),
            }
        )
    return tuple(rows)


def build_prelanding_specs(
    trajectory_rows: tuple[dict[str, object], ...],
    offsets: tuple[int, ...],
) -> tuple[PrelandingResetSpec, ...]:
    """生越えを持つ学生軌跡から複数の早期接管点を作る。"""
    specs = []
    for row in trajectory_rows:
        clearance_step = int(row["raw_clearance_step"])
        if clearance_step < 1:
            continue
        branch_path = Path(str(row["branch_path"])).resolve()
        with np.load(branch_path, allow_pickle=False) as archive:
            observations = np.asarray(archive["observations"], dtype=np.float32)
        for offset in offsets:
            prefix_steps = max(0, clearance_step - offset)
            specs.append(
                PrelandingResetSpec(
                    reset_id=(
                        f"seed_{row['seed']}_prelanding_minus_{offset}_step_{prefix_steps}"
                    ),
                    seed=int(row["seed"]),
                    start_runway_voxels=int(row["start_runway_voxels"]),
                    course_id=str(row["course_id"]),
                    prefix_steps=prefix_steps,
                    clearance_step=clearance_step,
                    offset_before_clearance=offset,
                    source_success=bool(row["course_complete"]),
                    branch_path=branch_path,
                    branch_sha256=str(row["branch_sha256"]),
                    expected_observation_sha256=array_sha256(
                        observations[prefix_steps]
                    ),
                )
            )
    return tuple(specs)


def collect_anchor_dataset(
    source_model: PPO,
    trajectory_rows: tuple[dict[str, object], ...],
    *,
    success_repeat: int,
    flat_stride: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """唯一成功窓と平地歩行から源学生の自己模倣錨を構築する。"""
    success_row = next(row for row in trajectory_rows if bool(row["course_complete"]))
    with np.load(Path(str(success_row["branch_path"])), allow_pickle=False) as archive:
        observations = np.asarray(archive["observations"], dtype=np.float32)
        actions = np.asarray(archive["actions"], dtype=np.float32)
    clearance_step = int(success_row["raw_clearance_step"])
    success_start = max(0, clearance_step - 64)
    success_observations = observations[success_start:-1]
    success_actions = actions[success_start:]
    anchor_observations = [
        np.repeat(success_observations, success_repeat, axis=0)
    ]
    anchor_actions = [np.repeat(success_actions, success_repeat, axis=0)]
    flat_rows = 0
    for seed in (820_001, 820_002, 820_003):
        course = flat_retention_course(seed)
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        flat_observations = []
        flat_actions = []
        try:
            observation, info = environment.reset(seed=seed)
            initial_x = float(info["x_position"])
            steps = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = source_model.predict(observation, deterministic=True)
                if steps % flat_stride == 0:
                    flat_observations.append(np.asarray(observation, dtype=np.float32))
                    flat_actions.append(np.asarray(action, dtype=np.float32))
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                if float(info["x_position"]) - initial_x >= 2.4:
                    break
        finally:
            environment.close()
        anchor_observations.append(np.asarray(flat_observations, dtype=np.float32))
        anchor_actions.append(np.asarray(flat_actions, dtype=np.float32))
        flat_rows += len(flat_observations)
    return (
        np.concatenate(anchor_observations, axis=0),
        np.concatenate(anchor_actions, axis=0),
        {
            "success_window_rows": int(len(success_observations)),
            "success_window_repeat": success_repeat,
            "flat_rows": flat_rows,
        },
    )


class StudentPrelandingEnv(gym.Env):
    """失敗前と成功前の学生接管、完全開始、平地を混合する。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        specs: tuple[PrelandingResetSpec, ...],
        train_seeds: tuple[int, ...],
        weights: Mapping[str, float],
        *,
        seed: int,
        maximum_student_steps: int,
    ) -> None:
        self.specs = specs
        self.train_seeds = train_seeds
        self.weights = dict(weights)
        self.mode_names = tuple(self.weights)
        self.mode_probabilities = np.asarray(tuple(self.weights.values()), dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.maximum_student_steps = maximum_student_steps
        initial_course = sample_curriculum_course(train_seeds[0], "hurdle_single", "train")
        self.environment = GeneralObstacleEnv(course=initial_course, resample_on_reset=False)
        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space
        self.arrays_by_path: dict[Path, dict[str, np.ndarray]] = {}
        self.episode_index = 0
        self.current_mode = ""
        self.current_reset_id = ""
        self.student_steps = 0
        self.initial_x = 0.0
        self.previous_raw_clearances = 0
        self.previous_recovered_obstacles = 0

    def _arrays(self, spec: PrelandingResetSpec) -> dict[str, np.ndarray]:
        """学生源軌跡を作業環境ごとに一度だけ読み込む。"""
        if spec.branch_path not in self.arrays_by_path:
            if sha256_file(spec.branch_path) != spec.branch_sha256:
                raise ValueError("M6S学生源軌跡ハッシュが一致しない。")
            with np.load(spec.branch_path, allow_pickle=False) as archive:
                self.arrays_by_path[spec.branch_path] = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
        return self.arrays_by_path[spec.branch_path]

    def _select_spec(self, mode: str, reset_id: str | None) -> PrelandingResetSpec:
        """成功または失敗の指定集合から接管点を選ぶ。"""
        require_success = mode == "successful_prelanding"
        candidates = tuple(
            spec for spec in self.specs if spec.source_success == require_success
        )
        if reset_id is not None:
            return next(spec for spec in candidates if spec.reset_id == reset_id)
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """学生前置列または教師なし通常開始を再現する。"""
        super().reset(seed=seed)
        options = options or {}
        mode = str(
            options.get(
                "mode",
                self.rng.choice(self.mode_names, p=self.mode_probabilities),
            )
        )
        if mode not in TRAINING_MODES:
            raise ValueError(f"未知のM6S開始モード: {mode}")
        episode_seed = int(
            options.get(
                "course_seed",
                self.train_seeds[self.episode_index % len(self.train_seeds)],
            )
        )
        self.episode_index += 1
        prefix_steps = 0
        reset_id = ""
        if mode in {"failed_prelanding", "successful_prelanding"}:
            spec = self._select_spec(
                mode,
                str(options["reset_id"]) if "reset_id" in options else None,
            )
            course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
            if course.course_id != spec.course_id:
                raise ValueError("M6S接管点とコース識別子が一致しない。")
            observation, info = self.environment.reset(
                seed=spec.seed,
                options={"course": course},
            )
            arrays = self._arrays(spec)
            for step, action in enumerate(arrays["actions"][: spec.prefix_steps]):
                observation, _, terminated, truncated, info = self.environment.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        f"M6S学生前置列が早期終了した: {spec.reset_id}, {step + 1}"
                    )
            if array_sha256(np.asarray(observation, dtype=np.float32)) != (
                spec.expected_observation_sha256
            ):
                raise RuntimeError("M6S学生接管観測が凍結値と一致しない。")
            prefix_steps = spec.prefix_steps
            reset_id = spec.reset_id
        elif mode == "flat":
            course = flat_retention_course(episode_seed)
            observation, info = self.environment.reset(
                seed=episode_seed,
                options={"course": course},
            )
        else:
            course = sample_curriculum_course(episode_seed, "hurdle_single", "train")
            observation, info = self.environment.reset(
                seed=episode_seed,
                options={"course": course},
            )
        self.current_mode = mode
        self.current_reset_id = reset_id
        self.student_steps = 0
        self.initial_x = float(info["x_position"])
        self.previous_raw_clearances = int(info["raw_clearances"])
        self.previous_recovered_obstacles = int(info["recovered_obstacles"])
        enriched = dict(info)
        enriched.update(
            {
                "curriculum_mode": mode,
                "curriculum_reset_id": reset_id,
                "student_prefix_steps": prefix_steps,
                "student_prefix_source": "frozen_student_npz" if prefix_steps else "none",
                "self_demonstration_only": True,
                "teacher_module_loaded": False,
                "teacher_actions_used": 0,
                "student_control_started": True,
                "student_task_success": False,
            }
        )
        return np.asarray(observation, dtype=np.float32), enriched

    def step(self, action: np.ndarray):
        """学生動作だけを実行し越え、回復、完走を一度ずつ報酬化する。"""
        observation, base_reward, terminated, truncated, info = self.environment.step(
            np.asarray(action, dtype=np.float32)
        )
        self.student_steps += 1
        raw_now = int(info["raw_clearances"])
        recovery_now = int(info["recovered_obstacles"])
        raw_event = max(0, raw_now - self.previous_raw_clearances)
        recovery_event = max(0, recovery_now - self.previous_recovered_obstacles)
        self.previous_raw_clearances = raw_now
        self.previous_recovered_obstacles = recovery_now
        reward = float(base_reward) + 6.0 * raw_event + 24.0 * recovery_event
        if bool(info["course_complete"]):
            reward += 35.0
        if bool(info["hard_fall"]):
            reward -= 24.0
        flat_success = bool(
            self.current_mode == "flat"
            and float(info["x_position"]) - self.initial_x >= 2.4
            and not bool(info["hard_fall"])
        )
        if flat_success:
            reward += 15.0
            terminated = True
            truncated = False
        if self.student_steps >= self.maximum_student_steps and not terminated:
            truncated = True
        success = bool(flat_success or (info["course_complete"] and not info["hard_fall"]))
        enriched = dict(info)
        enriched.update(
            {
                "curriculum_mode": self.current_mode,
                "curriculum_reset_id": self.current_reset_id,
                "student_steps": self.student_steps,
                "self_demonstration_only": True,
                "teacher_module_loaded": False,
                "teacher_actions_used": 0,
                "student_task_success": success,
                "raw_clearance_event": raw_event,
                "recovery_event": recovery_event,
            }
        )
        return observation, float(reward), bool(terminated), bool(truncated), enriched

    def close(self) -> None:
        """内部物理環境を閉じる。"""
        self.environment.close()


def make_vector_environment(
    specs: tuple[PrelandingResetSpec, ...],
    train_seeds: tuple[int, ...],
    weights: Mapping[str, float],
    *,
    seed: int,
    maximum_student_steps: int,
    environment_count: int,
) -> VecEnv:
    """学生越え前環境をCPU一環境または八並列で作る。"""
    factories = []
    for index in range(environment_count):
        worker_seed = seed + index * 10_000

        def factory(actual_seed: int = worker_seed) -> StudentPrelandingEnv:
            """一作業プロセス専用の越え前環境を生成する。"""
            return StudentPrelandingEnv(
                specs,
                train_seeds,
                weights,
                seed=actual_seed,
                maximum_student_steps=maximum_student_steps,
            )

        factories.append(factory)
    if environment_count == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method="spawn")


def apply_self_anchor(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    maximum_gradient_norm: float,
) -> dict[str, object]:
    """源学生の成功窓と平地動作へactorを小さく引き戻す。"""
    parameters = list(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    observation_tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=model.device
    )
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=model.device)
    history = []
    model.policy.train()
    for epoch in range(1, epochs + 1):
        latent = model.policy.mlp_extractor.forward_actor(observation_tensor)
        predictions = model.policy.action_net(latent)
        loss = torch.mean((predictions - action_tensor) ** 2)
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, maximum_gradient_norm).cpu()
        )
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": gradient_norm,
            }
        )
    return {"history": history}


def evaluate_prelanding_offsets(
    model: PPO,
    specs: tuple[PrelandingResetSpec, ...],
    train_seeds: tuple[int, ...],
    *,
    offset: int,
    maximum_student_steps: int,
) -> dict[str, object]:
    """指定オフセットの全七位置から学生の越えと回復を評価する。"""
    selected = tuple(spec for spec in specs if spec.offset_before_clearance == offset)
    environment = StudentPrelandingEnv(
        specs,
        train_seeds,
        {"failed_prelanding": 0.85, "successful_prelanding": 0.15},
        seed=830_000,
        maximum_student_steps=maximum_student_steps,
    )
    rows = []
    try:
        for spec in selected:
            mode = "successful_prelanding" if spec.source_success else "failed_prelanding"
            observation, info = environment.reset(
                options={"mode": mode, "reset_id": spec.reset_id}
            )
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
            rows.append(
                {
                    "reset_id": spec.reset_id,
                    "start_runway_voxels": spec.start_runway_voxels,
                    "success": bool(info["student_task_success"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "raw_clearances": int(info["raw_clearances"]),
                    "recovered_obstacles": int(info["recovered_obstacles"]),
                }
            )
    finally:
        environment.close()
    return {
        "method": "student_prefix_then_student_prelanding",
        "offset_before_clearance": offset,
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "raw_clearance_count": sum(int(row["raw_clearances"]) >= 1 for row in rows),
        "recovery_count": sum(int(row["recovered_obstacles"]) >= 1 for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "teacher_module_loaded": False,
        "teacher_actions_used": 0,
        "rows": rows,
    }


def evaluate_flat_retention(
    model: PPO,
    specs: tuple[PrelandingResetSpec, ...],
    train_seeds: tuple[int, ...],
    *,
    maximum_student_steps: int,
) -> dict[str, object]:
    """三つの長助走で平地技能を学生だけで評価する。"""
    environment = StudentPrelandingEnv(
        specs,
        train_seeds,
        {"flat": 1.0},
        seed=840_000,
        maximum_student_steps=maximum_student_steps,
    )
    rows = []
    try:
        for seed in (840_001, 840_002, 840_003):
            observation, info = environment.reset(
                options={"mode": "flat", "course_seed": seed}
            )
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
            rows.append(
                {
                    "seed": seed,
                    "success": bool(info["student_task_success"]),
                    "hard_fall": bool(info["hard_fall"]),
                }
            )
    finally:
        environment.close()
    return {
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "teacher_module_loaded": False,
        "teacher_actions_used": 0,
        "rows": rows,
    }


def raw_clearance_count(result: Mapping[str, object]) -> int:
    """完全評価の各回から生越え回数を数える。"""
    return sum(int(row["raw_clearances"]) >= 1 for row in result["episodes"])


def eligible_checkpoint(
    validation: Mapping[str, object],
    flat: Mapping[str, object],
    protocol: M6SProtocol,
) -> bool:
    """越え率と平地を保持した検査点だけを選択対象にする。"""
    return bool(
        raw_clearance_count(validation) >= protocol.minimum_validation_raw_clearances
        and int(flat["success_count"]) >= protocol.minimum_flat_successes
        and int(flat["hard_fall_count"]) == 0
    )


def safety_key(
    validation: Mapping[str, object],
    prelanding: Mapping[str, object],
) -> tuple[float, ...]:
    """完走、回復、安全、早期接管の順で合格検査点を順位付けする。"""
    return (
        float(validation["success_count"]),
        float(validation["mean_recovered_obstacles"]),
        -float(validation["hard_fall_count"]),
        float(prelanding["recovery_count"]),
        float(validation["mean_max_x"]),
    )


def append_progress(path: Path, row: Mapping[str, object]) -> None:
    """M6S検査点の完全能力と安全能力をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """自己成功錨付きのCPU八並列PPOを複数検査点で実行する。"""
    parser = argparse.ArgumentParser(description="M6S学生成功軌跡安全定着を実行する。")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-name", default="m6s_student_success_safety_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--environment-count", type=int)
    parser.add_argument("--maximum-total-steps", type=int)
    args = parser.parse_args()
    protocol = load_protocol(Path(args.protocol))
    seed_manifest = load_seed_manifest(protocol.seed_manifest_path)
    if seed_manifest.sha256 != protocol.seed_manifest_sha256:
        raise ValueError("M6S乱数種目録の読み込み後ハッシュが一致しない。")
    train_seeds = seed_manifest.for_split("train")
    validation_seeds = seed_manifest.for_split("validation")
    environment_count = (
        protocol.parallel_environments
        if args.environment_count is None
        else int(args.environment_count)
    )
    if environment_count < 1 or environment_count > protocol.parallel_environments:
        raise ValueError("M6S環境数は1から8でなければならない。")
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    source_model = PPO.load(protocol.source_model_path, device="cpu")
    trajectory_rows = collect_source_trajectories(
        source_model,
        train_seeds,
        output_dir / "source_trajectories",
    )
    successful_rows = [row for row in trajectory_rows if bool(row["course_complete"])]
    clearance_rows = [row for row in trajectory_rows if int(row["raw_clearance_step"]) >= 1]
    if len(successful_rows) != 1 or len(clearance_rows) < 7:
        raise RuntimeError("M6S源学生の成功数または生越え数が凍結期待と一致しない。")
    specs = build_prelanding_specs(trajectory_rows, protocol.preclearance_offsets)
    if len(specs) != len(clearance_rows) * len(protocol.preclearance_offsets):
        raise RuntimeError("M6S越え前接管点数が不正である。")
    manifest_path = output_dir / "prelanding_reset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "m6s_student_prelanding_manifest_v1",
                "frozen": True,
                "split": "train",
                "source_model_path": str(protocol.source_model_path),
                "source_model_sha256": protocol.source_model_sha256,
                "teacher_actions_used": 0,
                "trajectory_rows": trajectory_rows,
                "specs": [spec.as_dict() for spec in specs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    anchor_observations, anchor_actions, anchor_metadata = collect_anchor_dataset(
        source_model,
        trajectory_rows,
        success_repeat=protocol.success_anchor_repeat,
        flat_stride=protocol.flat_anchor_stride,
    )
    protected_paths = (
        protocol.source_model_path,
        protocol.source_summary_path,
        protocol.seed_manifest_path,
    )
    hashes_before = {str(path): sha256_file(path) for path in protected_paths}
    active_environment = make_vector_environment(
        specs,
        train_seeds,
        protocol.training_weights,
        seed=args.seed,
        maximum_student_steps=protocol.maximum_student_steps_per_episode,
        environment_count=environment_count,
    )
    model = PPO.load(
        protocol.source_model_path,
        env=active_environment,
        device=protocol.device,
    )
    model.save(output_dir / "initial_student_copy")
    stage = get_curriculum_stage("hurdle_single")
    initial_validation = evaluate_student_batch(
        model,
        seeds=validation_seeds,
        stage=stage,
        split="validation",
    )
    initial_prelanding = evaluate_prelanding_offsets(
        model,
        specs,
        train_seeds,
        offset=max(protocol.preclearance_offsets),
        maximum_student_steps=protocol.maximum_student_steps_per_episode,
    )
    initial_flat = evaluate_flat_retention(
        model,
        specs,
        train_seeds,
        maximum_student_steps=protocol.maximum_student_steps_per_episode,
    )
    if not eligible_checkpoint(initial_validation, initial_flat, protocol):
        raise RuntimeError("M6S源学生が越え率または平地保持門を満たさない。")
    best_key = safety_key(initial_validation, initial_prelanding)
    best_step = 0
    model.save(output_dir / "best_student")
    history = []
    completed_steps = 0
    maximum_steps = (
        protocol.total_steps
        if args.maximum_total_steps is None
        else min(protocol.total_steps, int(args.maximum_total_steps))
    )
    try:
        while completed_steps < maximum_steps:
            chunk = min(
                protocol.checkpoint_interval_steps,
                maximum_steps - completed_steps,
            )
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            completed_steps += chunk
            anchor_result = apply_self_anchor(
                model,
                anchor_observations,
                anchor_actions,
                learning_rate=protocol.anchor_learning_rate,
                epochs=protocol.anchor_epochs_per_checkpoint,
                maximum_gradient_norm=protocol.anchor_maximum_gradient_norm,
            )
            model.save(checkpoints_dir / f"student_{completed_steps}_m6s_steps")
            validation = evaluate_student_batch(
                model,
                seeds=validation_seeds,
                stage=stage,
                split="validation",
            )
            prelanding = evaluate_prelanding_offsets(
                model,
                specs,
                train_seeds,
                offset=max(protocol.preclearance_offsets),
                maximum_student_steps=protocol.maximum_student_steps_per_episode,
            )
            flat = evaluate_flat_retention(
                model,
                specs,
                train_seeds,
                maximum_student_steps=protocol.maximum_student_steps_per_episode,
            )
            eligible = eligible_checkpoint(validation, flat, protocol)
            key = safety_key(validation, prelanding)
            if eligible and key > best_key:
                best_key = key
                best_step = completed_steps
                model.save(output_dir / "best_student")
            row = {
                "m6s_student_steps": completed_steps,
                "validation_success_count": validation["success_count"],
                "validation_raw_clearance_count": raw_clearance_count(validation),
                "validation_recovery_count": int(
                    round(float(validation["mean_recovered_obstacles"]) * 11)
                ),
                "validation_hard_fall_count": validation["hard_fall_count"],
                "validation_mean_max_x": validation["mean_max_x"],
                "prelanding_success_count": prelanding["success_count"],
                "prelanding_recovery_count": prelanding["recovery_count"],
                "prelanding_hard_fall_count": prelanding["hard_fall_count"],
                "flat_success_count": flat["success_count"],
                "eligible": eligible,
                "anchor_loss": anchor_result["history"][-1]["loss"],
            }
            history.append(
                {
                    **row,
                    "validation": validation,
                    "prelanding": prelanding,
                    "flat": flat,
                    "anchor": anchor_result,
                }
            )
            append_progress(output_dir / "checkpoint_progress.csv", row)
            print(json.dumps({"event": "checkpoint", **row}, ensure_ascii=False), flush=True)
        model.save(output_dir / "final_student")
    finally:
        active_environment.close()
    best_model = PPO.load(output_dir / "best_student.zip", device="cpu")
    final_model = PPO.load(output_dir / "final_student.zip", device="cpu")
    best_validation = evaluate_student_batch(
        best_model,
        seeds=validation_seeds,
        stage=stage,
        split="validation",
    )
    best_train = evaluate_student_batch(
        best_model,
        seeds=train_seeds,
        stage=stage,
        split="train",
    )
    best_prelanding = evaluate_prelanding_offsets(
        best_model,
        specs,
        train_seeds,
        offset=max(protocol.preclearance_offsets),
        maximum_student_steps=protocol.maximum_student_steps_per_episode,
    )
    best_flat = evaluate_flat_retention(
        best_model,
        specs,
        train_seeds,
        maximum_student_steps=protocol.maximum_student_steps_per_episode,
    )
    final_validation = evaluate_student_batch(
        final_model,
        seeds=validation_seeds,
        stage=stage,
        split="validation",
    )
    hashes_after = {str(path): sha256_file(path) for path in protected_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("M6S中に保護出典が変更された。")
    summary = {
        "method": "m6s_student_success_anchor_prelanding_safety_ppo",
        "run_name": args.run_name,
        "protocol_path": str(protocol.source_path),
        "protocol_sha256": sha256_file(protocol.source_path),
        "source_model_path": str(protocol.source_model_path),
        "source_model_sha256": protocol.source_model_sha256,
        "trajectory_manifest": str(manifest_path.resolve()),
        "trajectory_manifest_sha256": sha256_file(manifest_path),
        "source_trajectory_count": len(trajectory_rows),
        "source_success_count": len(successful_rows),
        "source_clearance_count": len(clearance_rows),
        "prelanding_spec_count": len(specs),
        "anchor_metadata": anchor_metadata,
        "anchor_rows": int(len(anchor_observations)),
        "single_shared_student": True,
        "device": protocol.device,
        "parallel_environments": environment_count,
        "completed_steps": completed_steps,
        "checkpoint_count": len(history),
        "best_step": best_step,
        "initial_validation": initial_validation,
        "initial_prelanding": initial_prelanding,
        "initial_flat_retention": initial_flat,
        "checkpoint_history": history,
        "best_student_validation": best_validation,
        "best_student_train": best_train,
        "best_student_prelanding": best_prelanding,
        "best_student_flat_retention": best_flat,
        "final_student_validation": final_validation,
        "teacher_module_loaded_in_student_evaluation": False,
        "teacher_actions_used": 0,
        "self_demonstration_source": "frozen_successful_student_trajectory",
        "validation_teacher_interventions": 0,
        "holdout_episodes": 0,
        "m7_gate": {
            "required_successes": 9,
            "maximum_hard_falls": 0,
            "success_count": int(best_validation["success_count"]),
            "hard_fall_count": int(best_validation["hard_fall_count"]),
            "passed": bool(
                int(best_validation["success_count"]) >= 9
                and int(best_validation["hard_fall_count"]) == 0
            ),
        },
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_sources_unchanged": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["m7_gate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
