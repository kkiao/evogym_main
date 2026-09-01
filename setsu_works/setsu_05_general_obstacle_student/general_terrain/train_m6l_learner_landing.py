"""M6Lで学生自身が誘導した越え直後状態から落地回復を訓練する。"""

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
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "m6l_learner_landing_protocol_v1.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "m6l_learner_landing"
TRAINING_MODES = ("learner_landing", "full_start", "flat")


@dataclass(frozen=True)
class LandingResetSpec:
    """凍結学生が初めて生越えした直後の再生仕様を保持する。"""

    reset_id: str
    seed: int
    course_id: str
    prefix_steps: int
    branch_path: Path
    branch_sha256: str
    expected_observation_sha256: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存できる辞書を返す。"""
        payload = asdict(self)
        payload["branch_path"] = str(self.branch_path.resolve())
        return payload


@dataclass(frozen=True)
class M6LProtocol:
    """学生誘導落地の出典、混合比、検査点上限を保持する。"""

    source_path: Path
    source_model_path: Path
    source_model_sha256: str
    source_summary_path: Path
    source_summary_sha256: str
    seed_manifest_path: Path
    seed_manifest_sha256: str
    device: str
    parallel_environments: int
    checkpoint_interval_steps: int
    total_steps: int
    maximum_student_steps_per_episode: int
    training_weights: dict[str, float]
    minimum_landing_specs: int


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> M6LProtocol:
    """M6L規約、学生のみ境界、全出典ハッシュを検査する。"""
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M6L規約は凍結済みでなければならない。")
    if str(payload["device"]) != "cpu" or int(payload["parallel_environments"]) != 8:
        raise ValueError("M6L正式訓練はCPU八並列に固定する。")
    if int(payload["teacher_actions_used"]) != 0:
        raise ValueError("M6Lでは教師動作を使用できない。")
    if bool(payload["validation_teacher_enabled"]):
        raise ValueError("M6L検証では教師を有効化できない。")
    if bool(payload["holdout_teacher_enabled"]):
        raise ValueError("M6L留出では教師を有効化できない。")
    if int(payload["holdout_episodes"]) != 0:
        raise ValueError("M6Lは留出区分へアクセスできない。")
    paths = {
        "source_model": resolve_project_path(str(payload["source_model_path"])),
        "source_summary": resolve_project_path(str(payload["source_summary_path"])),
        "seed_manifest": resolve_project_path(str(payload["seed_manifest_path"])),
    }
    for name, protected_path in paths.items():
        if sha256_file(protected_path) != str(payload[f"{name}_sha256"]):
            raise ValueError(f"M6L出典ハッシュが一致しない: {protected_path}")
    interval = int(payload["checkpoint_interval_steps"])
    total_steps = int(payload["total_steps"])
    if total_steps < interval * 3 or total_steps % interval != 0:
        raise ValueError("M6Lは三検査点以上かつ間隔の整数倍でなければならない。")
    weights = {
        str(key): float(value) for key, value in payload["training_weights"].items()
    }
    if set(weights) != set(TRAINING_MODES) or not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("M6L訓練混合が不正である。")
    return M6LProtocol(
        source_path=source_path,
        source_model_path=paths["source_model"],
        source_model_sha256=str(payload["source_model_sha256"]),
        source_summary_path=paths["source_summary"],
        source_summary_sha256=str(payload["source_summary_sha256"]),
        seed_manifest_path=paths["seed_manifest"],
        seed_manifest_sha256=str(payload["seed_manifest_sha256"]),
        device=str(payload["device"]),
        parallel_environments=int(payload["parallel_environments"]),
        checkpoint_interval_steps=interval,
        total_steps=total_steps,
        maximum_student_steps_per_episode=int(
            payload["maximum_student_steps_per_episode"]
        ),
        training_weights=weights,
        minimum_landing_specs=int(payload["minimum_landing_specs"]),
    )


def collect_landing_reset_specs(
    model: PPO,
    train_seeds: tuple[int, ...],
    branches_dir: Path,
) -> tuple[LandingResetSpec, ...]:
    """訓練種子だけから初回生越えまでの学生動作前置列を凍結する。"""
    branches_dir.mkdir(parents=True, exist_ok=False)
    specs = []
    for seed in train_seeds:
        course = sample_curriculum_course(seed, "hurdle_single", "train")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        actions = []
        try:
            observation, info = environment.reset(seed=seed)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                actions.append(np.asarray(action, dtype=np.float32))
                observation, _, terminated, truncated, info = environment.step(action)
                if int(info["raw_clearances"]) >= 1:
                    branch_path = branches_dir / f"seed_{seed}_first_clearance.npz"
                    np.savez_compressed(
                        branch_path,
                        actions=np.asarray(actions, dtype=np.float32),
                        expected_observation=np.asarray(observation, dtype=np.float32),
                    )
                    specs.append(
                        LandingResetSpec(
                            reset_id=f"seed_{seed}_student_first_clearance",
                            seed=seed,
                            course_id=course.course_id,
                            prefix_steps=len(actions),
                            branch_path=branch_path.resolve(),
                            branch_sha256=sha256_file(branch_path),
                            expected_observation_sha256=array_sha256(
                                np.asarray(observation, dtype=np.float32)
                            ),
                        )
                    )
                    break
        finally:
            environment.close()
    return tuple(specs)


class LearnerLandingEnv(gym.Env):
    """学生前置列、通常起点、平地保持を教師なしで混合する。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        specs: tuple[LandingResetSpec, ...],
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

    def _arrays(self, spec: LandingResetSpec) -> dict[str, np.ndarray]:
        """学生前置列を作業環境ごとに一度だけ読み込む。"""
        if spec.branch_path not in self.arrays_by_path:
            if sha256_file(spec.branch_path) != spec.branch_sha256:
                raise ValueError("M6L学生前置列ハッシュが一致しない。")
            with np.load(spec.branch_path, allow_pickle=False) as archive:
                self.arrays_by_path[spec.branch_path] = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
        return self.arrays_by_path[spec.branch_path]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """学生誘導落地または教師なし通常開始を再現する。"""
        super().reset(seed=seed)
        options = options or {}
        mode = str(
            options.get(
                "mode",
                self.rng.choice(self.mode_names, p=self.mode_probabilities),
            )
        )
        if mode not in TRAINING_MODES:
            raise ValueError(f"未知のM6L開始モード: {mode}")
        episode_seed = int(
            options.get(
                "course_seed",
                self.train_seeds[self.episode_index % len(self.train_seeds)],
            )
        )
        self.episode_index += 1
        prefix_steps = 0
        reset_id = ""
        if mode == "learner_landing":
            if "reset_id" in options:
                spec = next(
                    item for item in self.specs if item.reset_id == str(options["reset_id"])
                )
            else:
                spec = self.specs[int(self.rng.integers(0, len(self.specs)))]
            course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
            if course.course_id != spec.course_id:
                raise ValueError("M6L学生前置列とコース識別子が一致しない。")
            observation, info = self.environment.reset(
                seed=spec.seed,
                options={"course": course},
            )
            arrays = self._arrays(spec)
            for step, action in enumerate(arrays["actions"]):
                observation, _, terminated, truncated, info = self.environment.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        f"M6L学生前置列が早期終了した: {spec.reset_id}, {step + 1}"
                    )
            if array_sha256(np.asarray(observation, dtype=np.float32)) != (
                spec.expected_observation_sha256
            ):
                raise RuntimeError("M6L学生誘導観測が凍結値と一致しない。")
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
                "learner_prefix_steps": prefix_steps,
                "learner_prefix_source": "frozen_student_npz" if prefix_steps else "none",
                "teacher_module_loaded": False,
                "teacher_actions_used": 0,
                "student_control_started": True,
                "student_task_success": False,
            }
        )
        return np.asarray(observation, dtype=np.float32), enriched

    def step(self, action: np.ndarray):
        """学生動作だけを実行し新しい越えと回復を一度だけ報酬化する。"""
        observation, base_reward, terminated, truncated, info = self.environment.step(
            np.asarray(action, dtype=np.float32)
        )
        self.student_steps += 1
        raw_now = int(info["raw_clearances"])
        recovered_now = int(info["recovered_obstacles"])
        raw_event = max(0, raw_now - self.previous_raw_clearances)
        recovery_event = max(0, recovered_now - self.previous_recovered_obstacles)
        self.previous_raw_clearances = raw_now
        self.previous_recovered_obstacles = recovered_now
        reward = float(base_reward) + 6.0 * raw_event + 20.0 * recovery_event
        if bool(info["course_complete"]):
            reward += 30.0
        if bool(info["hard_fall"]):
            reward -= 20.0
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
    specs: tuple[LandingResetSpec, ...],
    train_seeds: tuple[int, ...],
    weights: Mapping[str, float],
    *,
    seed: int,
    maximum_student_steps: int,
    environment_count: int,
) -> VecEnv:
    """学生誘導落地をCPU一環境または八並列で作る。"""
    factories = []
    for index in range(environment_count):
        worker_seed = seed + index * 10_000

        def factory(actual_seed: int = worker_seed) -> LearnerLandingEnv:
            """一作業プロセス専用の落地環境を生成する。"""
            return LearnerLandingEnv(
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


def evaluate_landing_resets(
    model: PPO,
    specs: tuple[LandingResetSpec, ...],
    train_seeds: tuple[int, ...],
    *,
    maximum_student_steps: int,
) -> dict[str, object]:
    """全学生誘導落地点からの回復と完走を教師なしで評価する。"""
    environment = LearnerLandingEnv(
        specs,
        train_seeds,
        {"learner_landing": 1.0},
        seed=800_000,
        maximum_student_steps=maximum_student_steps,
    )
    rows = []
    try:
        for spec in specs:
            observation, info = environment.reset(
                options={"mode": "learner_landing", "reset_id": spec.reset_id}
            )
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
            rows.append(
                {
                    "reset_id": spec.reset_id,
                    "success": bool(info["student_task_success"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "raw_clearances": int(info["raw_clearances"]),
                    "recovered_obstacles": int(info["recovered_obstacles"]),
                    "maximum_com_x": float(info["max_x_position"]),
                }
            )
    finally:
        environment.close()
    return {
        "method": "learner_prefix_then_student_recovery",
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "recovery_count": sum(int(row["recovered_obstacles"]) >= 1 for row in rows),
        "teacher_module_loaded": False,
        "teacher_actions_used": 0,
        "rows": rows,
    }


def evaluate_flat_retention(
    model: PPO,
    specs: tuple[LandingResetSpec, ...],
    train_seeds: tuple[int, ...],
    *,
    maximum_student_steps: int,
) -> dict[str, object]:
    """三つの長助走で平地技能を教師なし評価する。"""
    environment = LearnerLandingEnv(
        specs,
        train_seeds,
        {"flat": 1.0},
        seed=810_000,
        maximum_student_steps=maximum_student_steps,
    )
    rows = []
    try:
        for seed in (810_001, 810_002, 810_003):
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


def checkpoint_key(
    validation: Mapping[str, object],
    landing: Mapping[str, object],
) -> tuple[float, ...]:
    """完走、回復、越え、安全、落地回復の順で検査点を順位付けする。"""
    return (
        float(validation["success_count"]),
        float(validation["mean_recovered_obstacles"]),
        float(validation["mean_raw_clearances"]),
        -float(validation["hard_fall_count"]),
        float(landing["recovery_count"]),
        float(validation["mean_max_x"]),
    )


def append_progress(path: Path, row: Mapping[str, object]) -> None:
    """M6L検査点の完全能力と落地能力をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """学生誘導落地を凍結し同一学生を複数検査点で訓練する。"""
    parser = argparse.ArgumentParser(description="M6L学生誘導落地回復を実行する。")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-name", default="m6l_learner_landing_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--environment-count", type=int)
    parser.add_argument("--maximum-total-steps", type=int)
    args = parser.parse_args()
    protocol = load_protocol(Path(args.protocol))
    seed_manifest = load_seed_manifest(protocol.seed_manifest_path)
    if seed_manifest.sha256 != protocol.seed_manifest_sha256:
        raise ValueError("M6L乱数種目録の読み込み後ハッシュが一致しない。")
    train_seeds = seed_manifest.for_split("train")
    validation_seeds = seed_manifest.for_split("validation")
    environment_count = (
        protocol.parallel_environments
        if args.environment_count is None
        else int(args.environment_count)
    )
    if environment_count < 1 or environment_count > protocol.parallel_environments:
        raise ValueError("M6L環境数は1から8でなければならない。")
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    source_model = PPO.load(protocol.source_model_path, device="cpu")
    specs = collect_landing_reset_specs(
        source_model,
        train_seeds,
        output_dir / "landing_branches",
    )
    if len(specs) < protocol.minimum_landing_specs:
        raise RuntimeError("M6L学生誘導落地点が最低数に達しない。")
    manifest_path = output_dir / "landing_reset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "m6l_learner_landing_manifest_v1",
                "frozen": True,
                "split": "train",
                "source_model_path": str(protocol.source_model_path),
                "source_model_sha256": protocol.source_model_sha256,
                "teacher_actions_used": 0,
                "specs": [spec.as_dict() for spec in specs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
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
    initial_landing = evaluate_landing_resets(
        model,
        specs,
        train_seeds,
        maximum_student_steps=protocol.maximum_student_steps_per_episode,
    )
    best_key = checkpoint_key(initial_validation, initial_landing)
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
            model.save(checkpoints_dir / f"student_{completed_steps}_m6l_steps")
            validation = evaluate_student_batch(
                model,
                seeds=validation_seeds,
                stage=stage,
                split="validation",
            )
            landing = evaluate_landing_resets(
                model,
                specs,
                train_seeds,
                maximum_student_steps=protocol.maximum_student_steps_per_episode,
            )
            key = checkpoint_key(validation, landing)
            if key > best_key:
                best_key = key
                best_step = completed_steps
                model.save(output_dir / "best_student")
            row = {
                "m6l_student_steps": completed_steps,
                "validation_success_count": validation["success_count"],
                "validation_hard_fall_count": validation["hard_fall_count"],
                "validation_mean_raw_clearances": validation[
                    "mean_raw_clearances"
                ],
                "validation_mean_recovered_obstacles": validation[
                    "mean_recovered_obstacles"
                ],
                "validation_mean_max_x": validation["mean_max_x"],
                "landing_success_count": landing["success_count"],
                "landing_recovery_count": landing["recovery_count"],
                "landing_hard_fall_count": landing["hard_fall_count"],
            }
            history.append({**row, "validation": validation, "landing": landing})
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
    best_landing = evaluate_landing_resets(
        best_model,
        specs,
        train_seeds,
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
        raise RuntimeError("M6L中に保護出典が変更された。")
    summary = {
        "method": "m6l_learner_induced_landing_recovery_ppo",
        "run_name": args.run_name,
        "protocol_path": str(protocol.source_path),
        "protocol_sha256": sha256_file(protocol.source_path),
        "source_model_path": str(protocol.source_model_path),
        "source_model_sha256": protocol.source_model_sha256,
        "landing_reset_manifest": str(manifest_path.resolve()),
        "landing_reset_manifest_sha256": sha256_file(manifest_path),
        "landing_spec_count": len(specs),
        "single_shared_student": True,
        "device": protocol.device,
        "parallel_environments": environment_count,
        "completed_steps": completed_steps,
        "checkpoint_count": len(history),
        "best_step": best_step,
        "initial_validation": initial_validation,
        "initial_landing": initial_landing,
        "checkpoint_history": history,
        "best_student_validation": best_validation,
        "best_student_train": best_train,
        "best_student_landing": best_landing,
        "best_student_flat_retention": best_flat,
        "final_student_validation": final_validation,
        "teacher_module_loaded_in_student_evaluation": False,
        "teacher_actions_used": 0,
        "validation_teacher_interventions": 0,
        "holdout_episodes": 0,
        "breakthrough": {
            "criterion": "at_least_one_teacher_free_validation_course_complete",
            "achieved": int(best_validation["success_count"]) >= 1,
            "validation_success_count": int(best_validation["success_count"]),
        },
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_sources_unchanged": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["breakthrough"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
