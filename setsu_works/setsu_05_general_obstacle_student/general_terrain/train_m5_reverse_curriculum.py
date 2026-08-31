"""M5の単一学生を逆向きカリキュラムと検査点階段で訓練する。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from general_terrain.curriculum import get_curriculum_stage, sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_only_evaluation import evaluate_student_batch
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "m5_reverse_curriculum_protocol_v1.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "m5_reverse_curriculum"
PHASE_NAMES = (
    "pre_hurdle",
    "hurdle_deformation",
    "post_clearance_recovery",
    "stable_finish",
)
ROLLIN_MODES = (*PHASE_NAMES, "dense_handoff")
TRAINING_MODES = (*ROLLIN_MODES, "flat", "full_start")


def sha256_file(path: Path) -> str:
    """ファイル内容のSHA-256を返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """配列の型、形状、連続バイト列から安定したハッシュを返す。"""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def resolve_project_path(value: str) -> Path:
    """相対パスをプロジェクト内だけで解決する。"""
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M5の出典はプロジェクト配下でなければならない。")
    return path


@dataclass(frozen=True)
class RollinSpec:
    """凍結成功軌跡上の一つの学生開始点を保持する。"""

    reset_id: str
    phase: str
    seed: int
    course_id: str
    source_step: int
    source_branch_path: Path
    source_branch_sha256: str
    source_observation_sha256: str


@dataclass(frozen=True)
class CurriculumBlock:
    """一つの訓練区間と混合する開始モードを保持する。"""

    name: str
    steps: int
    weights: dict[str, float]


@dataclass(frozen=True)
class M5Protocol:
    """M5の凍結出典、計算量、評価境界を保持する。"""

    source_path: Path
    source_model_path: Path
    source_model_sha256: str
    phase_reset_manifest_path: Path
    phase_reset_manifest_sha256: str
    seed_manifest_path: Path
    seed_manifest_sha256: str
    device: str
    parallel_environments: int
    checkpoint_interval_steps: int
    maximum_student_steps_per_episode: int
    extension_steps: int
    plateau_window_checkpoints: int
    curriculum: tuple[CurriculumBlock, ...]

    @property
    def planned_steps(self) -> int:
        """延長を除く学生制御歩数を返す。"""
        return sum(block.steps for block in self.curriculum)


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> M5Protocol:
    """凍結規約と全出典ハッシュを検査して読み込む。"""
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M5規約は凍結済みでなければならない。")
    if str(payload["device"]) != "cpu":
        raise ValueError("M5正式訓練はCPUに固定する。")
    if int(payload["parallel_environments"]) != 8:
        raise ValueError("M5正式訓練は8並列環境に固定する。")
    if int(payload["teacher_actions_after_student_takeover"]) != 0:
        raise ValueError("学生接管後の教師動作は0でなければならない。")
    if bool(payload["validation_teacher_enabled"]):
        raise ValueError("検証では教師を有効化できない。")
    if bool(payload["holdout_teacher_enabled"]):
        raise ValueError("留出評価では教師を有効化できない。")
    if int(payload["holdout_episodes"]) != 0:
        raise ValueError("M5は留出区分へアクセスできない。")
    source_model_path = resolve_project_path(str(payload["source_model_path"]))
    phase_manifest_path = resolve_project_path(
        str(payload["phase_reset_manifest_path"])
    )
    seed_manifest_path = resolve_project_path(str(payload["seed_manifest_path"]))
    protected = (
        (source_model_path, str(payload["source_model_sha256"])),
        (phase_manifest_path, str(payload["phase_reset_manifest_sha256"])),
        (seed_manifest_path, str(payload["seed_manifest_sha256"])),
    )
    for protected_path, expected_hash in protected:
        if sha256_file(protected_path) != expected_hash:
            raise ValueError(f"M5出典ハッシュが一致しない: {protected_path}")
    interval = int(payload["checkpoint_interval_steps"])
    blocks = tuple(
        CurriculumBlock(
            name=str(row["name"]),
            steps=int(row["steps"]),
            weights={str(key): float(value) for key, value in row["weights"].items()},
        )
        for row in payload["curriculum"]
    )
    expected_names = (
        "stable_finish",
        "post_clearance_recovery",
        "hurdle_deformation",
        "pre_hurdle",
        "full_start",
    )
    if tuple(block.name for block in blocks) != expected_names:
        raise ValueError("M5逆向きカリキュラムの順序が不正である。")
    for block in blocks:
        if block.steps < interval or block.steps % interval != 0:
            raise ValueError("各M5区間は検査点間隔の整数倍でなければならない。")
        if not set(block.weights).issubset(TRAINING_MODES):
            raise ValueError("M5区間に未知の開始モードがある。")
        if not np.isclose(sum(block.weights.values()), 1.0):
            raise ValueError("M5区間の混合確率合計は1でなければならない。")
    return M5Protocol(
        source_path=source_path,
        source_model_path=source_model_path,
        source_model_sha256=str(payload["source_model_sha256"]),
        phase_reset_manifest_path=phase_manifest_path,
        phase_reset_manifest_sha256=str(payload["phase_reset_manifest_sha256"]),
        seed_manifest_path=seed_manifest_path,
        seed_manifest_sha256=str(payload["seed_manifest_sha256"]),
        device=str(payload["device"]),
        parallel_environments=int(payload["parallel_environments"]),
        checkpoint_interval_steps=interval,
        maximum_student_steps_per_episode=int(
            payload["maximum_student_steps_per_episode"]
        ),
        extension_steps=int(payload["extension_steps"]),
        plateau_window_checkpoints=int(payload["plateau_window_checkpoints"]),
        curriculum=blocks,
    )


def load_rollin_specs(path: Path) -> tuple[RollinSpec, ...]:
    """凍結位相目録を教師モジュールなしで読み込む。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("位相開始点目録は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M5位相開始点は単一低壁の訓練区分でなければならない。")
    specs = tuple(
        RollinSpec(
            reset_id=str(row["reset_id"]),
            phase=str(row["phase"]),
            seed=int(row["seed"]),
            course_id=str(row["course_id"]),
            source_step=int(row["source_step"]),
            source_branch_path=Path(str(row["source_branch_path"])).resolve(),
            source_branch_sha256=str(row["source_branch_sha256"]),
            source_observation_sha256=str(row["source_observation_sha256"]),
        )
        for row in payload["specs"]
    )
    if len(specs) != 16:
        raise ValueError("M5位相開始点は16個でなければならない。")
    if {spec.phase for spec in specs} != set(PHASE_NAMES):
        raise ValueError("M5位相開始点の位相集合が不正である。")
    for spec in specs:
        if not spec.source_branch_path.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("M5軌跡分岐はプロジェクト配下でなければならない。")
        if sha256_file(spec.source_branch_path) != spec.source_branch_sha256:
            raise ValueError(f"M5軌跡分岐ハッシュが一致しない: {spec.reset_id}")
    return specs


def flat_retention_course(seed: int):
    """通常時間内に壁へ到達しない平地保持コースを返す。"""
    return build_course(
        ["low_hurdle"],
        split="m5_train_flat_retention",
        seed=seed,
        difficulty=1,
        start_runway_voxels=80,
    )


class M5ReverseCurriculumEnv(gym.Env):
    """成功軌跡の接近点と通常起点を混合して単一学生を訓練する。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        specs: tuple[RollinSpec, ...],
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
        self.worker_seed = seed
        self.maximum_student_steps = maximum_student_steps
        initial_course = sample_curriculum_course(
            train_seeds[0], "hurdle_single", "train"
        )
        self.environment = GeneralObstacleEnv(
            course=initial_course,
            resample_on_reset=False,
        )
        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space
        self.arrays_by_path: dict[Path, dict[str, np.ndarray]] = {}
        self.episode_index = 0
        self.student_steps = 0
        self.current_mode = ""
        self.current_reset_id = ""
        self.initial_x = 0.0
        self.initial_raw_clearances = 0
        self.initial_recovered_obstacles = 0

    def _arrays(self, spec: RollinSpec) -> dict[str, np.ndarray]:
        """一つの凍結軌跡を作業環境ごとに一度だけ読み込む。"""
        if spec.source_branch_path not in self.arrays_by_path:
            with np.load(spec.source_branch_path, allow_pickle=False) as archive:
                self.arrays_by_path[spec.source_branch_path] = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
        return self.arrays_by_path[spec.source_branch_path]

    def _select_spec(self, phase: str, reset_id: str | None) -> RollinSpec:
        """指定位相から再現可能に一つの開始点を選ぶ。"""
        candidates = tuple(spec for spec in self.specs if spec.phase == phase)
        if reset_id is not None:
            return next(spec for spec in candidates if spec.reset_id == reset_id)
        return candidates[int(self.rng.integers(0, len(candidates)))]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """教師問い合わせなしの凍結動作再生または通常起点を生成する。"""
        super().reset(seed=seed)
        options = options or {}
        requested_mode = options.get("mode")
        if requested_mode is None:
            mode = str(self.rng.choice(self.mode_names, p=self.mode_probabilities))
        else:
            mode = str(requested_mode)
        if mode not in TRAINING_MODES:
            raise ValueError(f"未知のM5開始モード: {mode}")
        episode_seed = int(
            options.get(
                "course_seed",
                self.train_seeds[self.episode_index % len(self.train_seeds)],
            )
        )
        self.episode_index += 1
        rollin_steps = 0
        reset_id = ""
        if mode in ROLLIN_MODES:
            spec = self._select_spec(
                mode,
                str(options["reset_id"]) if "reset_id" in options else None,
            )
            course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
            if course.course_id != spec.course_id:
                raise ValueError("M5開始点とコース識別子が一致しない。")
            observation, info = self.environment.reset(
                seed=spec.seed,
                options={"course": course},
            )
            arrays = self._arrays(spec)
            for step in range(spec.source_step):
                observation, _, terminated, truncated, info = self.environment.step(
                    np.asarray(arrays["executed_actions"][step], dtype=np.float32)
                )
                if terminated or truncated:
                    raise RuntimeError(
                        f"M5開始点再生が早期終了した: {spec.reset_id}, {step + 1}"
                    )
            if array_sha256(np.asarray(observation, dtype=np.float32)) != (
                spec.source_observation_sha256
            ):
                raise RuntimeError(f"M5開始点観測が出典と一致しない: {spec.reset_id}")
            rollin_steps = spec.source_step
            reset_id = spec.reset_id
        elif mode == "flat":
            course = flat_retention_course(episode_seed)
            observation, info = self.environment.reset(
                seed=episode_seed,
                options={"course": course},
            )
        else:
            course = sample_curriculum_course(
                episode_seed, "hurdle_single", "train"
            )
            observation, info = self.environment.reset(
                seed=episode_seed,
                options={"course": course},
            )
        self.student_steps = 0
        self.current_mode = mode
        self.current_reset_id = reset_id
        self.initial_x = float(info["x_position"])
        self.initial_raw_clearances = int(info["raw_clearances"])
        self.initial_recovered_obstacles = int(info["recovered_obstacles"])
        enriched = dict(info)
        enriched.update(
            {
                "curriculum_mode": mode,
                "curriculum_reset_id": reset_id,
                "training_rollin_steps": rollin_steps,
                "training_rollin_action_source": (
                    "frozen_success_npz" if rollin_steps else "none"
                ),
                "student_control_started": True,
                "teacher_module_loaded": False,
                "teacher_interventions": 0,
                "teacher_actions_after_student_takeover": 0,
                "student_task_success": False,
            }
        )
        return np.asarray(observation, dtype=np.float32), enriched

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """学生動作だけを実行し越え、回復、完走へ段階報酬を加える。"""
        observation, base_reward, terminated, truncated, info = self.environment.step(
            np.asarray(action, dtype=np.float32)
        )
        self.student_steps += 1
        raw_gain = max(0, int(info["raw_clearances"]) - self.initial_raw_clearances)
        recovery_gain = max(
            0,
            int(info["recovered_obstacles"]) - self.initial_recovered_obstacles,
        )
        reward = float(base_reward) + 4.0 * raw_gain + 8.0 * recovery_gain
        if bool(info["hard_fall"]):
            reward -= 15.0
        if bool(info["course_complete"]):
            reward += 25.0
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
                "student_control_started": True,
                "teacher_module_loaded": False,
                "teacher_interventions": 0,
                "teacher_actions_after_student_takeover": 0,
                "student_task_success": success,
                "raw_clearance_gain_after_takeover": raw_gain,
                "recovery_gain_after_takeover": recovery_gain,
            }
        )
        return (
            np.asarray(observation, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            enriched,
        )

    def close(self) -> None:
        """内部物理環境を閉じる。"""
        self.environment.close()


def make_vector_environment(
    specs: tuple[RollinSpec, ...],
    train_seeds: tuple[int, ...],
    weights: Mapping[str, float],
    *,
    seed: int,
    maximum_student_steps: int,
    environment_count: int,
) -> VecEnv:
    """Windows対応のCPU並列環境または単一試験環境を作る。"""
    factories = []
    for index in range(environment_count):
        worker_seed = seed + index * 10_000

        def factory(actual_seed: int = worker_seed) -> M5ReverseCurriculumEnv:
            """一作業プロセス専用の環境を生成する。"""
            return M5ReverseCurriculumEnv(
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


def evaluation_score(result: Mapping[str, object]) -> tuple[float, ...]:
    """安全完走を最優先する検査点順位を返す。"""
    return (
        float(result["success_count"]),
        float(result["mean_recovered_obstacles"]),
        float(result["mean_raw_clearances"]),
        -float(result["hard_fall_count"]),
        float(result["mean_max_x"]),
    )


def meaningful_improvement(
    earlier: Mapping[str, object],
    later: Mapping[str, object],
) -> bool:
    """偶然の微小変動を除く能力改善の有無を返す。"""
    return bool(
        int(later["success_count"]) > int(earlier["success_count"])
        or float(later["mean_recovered_obstacles"])
        > float(earlier["mean_recovered_obstacles"]) + 0.04
        or float(later["mean_raw_clearances"])
        > float(earlier["mean_raw_clearances"]) + 0.04
        or int(later["hard_fall_count"]) < int(earlier["hard_fall_count"])
        or float(later["mean_max_x"]) > float(earlier["mean_max_x"]) + 0.02
    )


def plateau_detected(
    evaluations: list[dict[str, object]],
    *,
    window: int,
) -> bool:
    """連続した複数検査点に能力改善がない場合だけ停滞と判定する。"""
    if len(evaluations) < window + 1:
        return False
    recent = evaluations[-(window + 1) :]
    return not any(
        meaningful_improvement(recent[index], recent[index + 1])
        for index in range(window)
    )


def evaluate_phase_starts(
    model: PPO,
    specs: tuple[RollinSpec, ...],
    train_seeds: tuple[int, ...],
    *,
    phase: str,
    maximum_student_steps: int,
) -> dict[str, object]:
    """訓練専用接近点からの学生単独後半能力を分離集計する。"""
    selected = tuple(spec for spec in specs if spec.phase == phase)
    environment = M5ReverseCurriculumEnv(
        specs,
        train_seeds,
        {phase: 1.0},
        seed=900_000,
        maximum_student_steps=maximum_student_steps,
    )
    episodes = []
    try:
        for spec in selected:
            observation, info = environment.reset(
                options={"mode": phase, "reset_id": spec.reset_id}
            )
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
            episodes.append(
                {
                    "reset_id": spec.reset_id,
                    "success": bool(info["student_task_success"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "raw_clearances": int(info["raw_clearances"]),
                    "recovered_obstacles": int(info["recovered_obstacles"]),
                    "maximum_com_x": float(info["max_x_position"]),
                    "teacher_interventions": 0,
                }
            )
    finally:
        environment.close()
    return {
        "method": "training_rollin_then_student_only",
        "phase": phase,
        "episodes": len(episodes),
        "success_count": sum(bool(row["success"]) for row in episodes),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in episodes),
        "teacher_module_loaded": False,
        "teacher_interventions_after_takeover": 0,
        "rows": episodes,
    }


def evaluate_flat_retention(
    model: PPO,
    specs: tuple[RollinSpec, ...],
    train_seeds: tuple[int, ...],
    *,
    maximum_student_steps: int,
) -> dict[str, object]:
    """三つの長助走で旧平地技能の保持を学生だけで測る。"""
    environment = M5ReverseCurriculumEnv(
        specs,
        train_seeds,
        {"flat": 1.0},
        seed=910_000,
        maximum_student_steps=maximum_student_steps,
    )
    episodes = []
    try:
        for seed in (910_001, 910_002, 910_003):
            observation, info = environment.reset(
                options={"mode": "flat", "course_seed": seed}
            )
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
            episodes.append(
                {
                    "seed": seed,
                    "success": bool(info["student_task_success"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "forward_displacement": float(info["forward_displacement"]),
                }
            )
    finally:
        environment.close()
    return {
        "episodes": len(episodes),
        "success_count": sum(bool(row["success"]) for row in episodes),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in episodes),
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
        "rows": episodes,
    }


def append_evaluation_csv(path: Path, row: Mapping[str, object]) -> None:
    """検査点の主要学生指標をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_argument_parser() -> argparse.ArgumentParser:
    """正式訓練と短い工学試験の引数を定義する。"""
    parser = argparse.ArgumentParser(description="M5逆向きカリキュラムを実行する。")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-name", default="m5_reverse_curriculum_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--environment-count", type=int)
    parser.add_argument("--maximum-blocks", type=int)
    parser.add_argument("--maximum-total-steps", type=int)
    parser.add_argument("--skip-extension", action="store_true")
    return parser


def main() -> None:
    """単一PPOを全逆向き区間で更新し学生単独検証を保存する。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    specs = load_rollin_specs(protocol.phase_reset_manifest_path)
    seed_manifest = load_seed_manifest(protocol.seed_manifest_path)
    if seed_manifest.sha256 != protocol.seed_manifest_sha256:
        raise ValueError("M5乱数種目録の読み込み後ハッシュが一致しない。")
    train_seeds = seed_manifest.for_split("train")
    validation_seeds = seed_manifest.for_split("validation")
    environment_count = (
        protocol.parallel_environments
        if args.environment_count is None
        else int(args.environment_count)
    )
    if environment_count < 1 or environment_count > protocol.parallel_environments:
        raise ValueError("M5環境数は1から8でなければならない。")
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    protected_paths = (
        protocol.source_model_path,
        protocol.phase_reset_manifest_path,
        protocol.seed_manifest_path,
        *(spec.source_branch_path for spec in specs),
    )
    protected_hashes_before = {
        str(path): sha256_file(path) for path in dict.fromkeys(protected_paths)
    }
    blocks = protocol.curriculum
    if args.maximum_blocks is not None:
        blocks = blocks[: int(args.maximum_blocks)]
    stage = get_curriculum_stage("hurdle_single")
    model: PPO | None = None
    active_environment: VecEnv | None = None
    completed_steps = 0
    checkpoint_index = 0
    validation_history: list[dict[str, object]] = []
    phase_history: list[dict[str, object]] = []
    curriculum_history: list[dict[str, object]] = []
    best_score: tuple[float, ...] | None = None
    best_step = 0
    stopped_by_user_cap = False
    try:
        initial_environment = make_vector_environment(
            specs,
            train_seeds,
            blocks[0].weights,
            seed=args.seed,
            maximum_student_steps=protocol.maximum_student_steps_per_episode,
            environment_count=environment_count,
        )
        active_environment = initial_environment
        model = PPO.load(
            protocol.source_model_path,
            env=active_environment,
            device=protocol.device,
        )
        if model.observation_space.shape != (95,) or model.action_space.shape != (6,):
            raise ValueError("M5初期学生の観測または動作形状が不正である。")
        model.save(output_dir / "initial_student_copy")
        initial_validation = evaluate_student_batch(
            model,
            seeds=validation_seeds,
            stage=stage,
            split="validation",
        )
        initial_validation["m5_student_steps"] = 0
        initial_validation["curriculum_block"] = "initial"
        validation_history.append(initial_validation)
        best_score = evaluation_score(initial_validation)
        model.save(output_dir / "best_student")
        append_evaluation_csv(
            output_dir / "validation_checkpoints.csv",
            {
                "m5_student_steps": 0,
                "curriculum_block": "initial",
                "success_count": initial_validation["success_count"],
                "hard_fall_count": initial_validation["hard_fall_count"],
                "mean_raw_clearances": initial_validation["mean_raw_clearances"],
                "mean_recovered_obstacles": initial_validation[
                    "mean_recovered_obstacles"
                ],
                "mean_max_x": initial_validation["mean_max_x"],
            },
        )
        print(
            json.dumps(
                {
                    "event": "initial_validation",
                    "m5_student_steps": 0,
                    "success_count": initial_validation["success_count"],
                    "hard_fall_count": initial_validation["hard_fall_count"],
                    "mean_raw_clearances": initial_validation["mean_raw_clearances"],
                    "mean_recovered_obstacles": initial_validation[
                        "mean_recovered_obstacles"
                    ],
                    "mean_max_x": initial_validation["mean_max_x"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        for block_index, block in enumerate(blocks):
            if block_index > 0:
                new_environment = make_vector_environment(
                    specs,
                    train_seeds,
                    block.weights,
                    seed=args.seed + block_index * 100_000,
                    maximum_student_steps=protocol.maximum_student_steps_per_episode,
                    environment_count=environment_count,
                )
                model.set_env(new_environment)
                active_environment.close()
                active_environment = new_environment
            block_completed = 0
            while block_completed < block.steps:
                if (
                    args.maximum_total_steps is not None
                    and completed_steps >= int(args.maximum_total_steps)
                ):
                    stopped_by_user_cap = True
                    break
                chunk = min(
                    protocol.checkpoint_interval_steps,
                    block.steps - block_completed,
                )
                if args.maximum_total_steps is not None:
                    chunk = min(chunk, int(args.maximum_total_steps) - completed_steps)
                if chunk <= 0:
                    stopped_by_user_cap = True
                    break
                model.learn(
                    total_timesteps=chunk,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                completed_steps += chunk
                block_completed += chunk
                checkpoint_index += 1
                checkpoint_path = (
                    checkpoints_dir / f"student_{completed_steps}_m5_steps"
                )
                model.save(checkpoint_path)
                validation = evaluate_student_batch(
                    model,
                    seeds=validation_seeds,
                    stage=stage,
                    split="validation",
                )
                validation["m5_student_steps"] = completed_steps
                validation["curriculum_block"] = block.name
                validation_history.append(validation)
                phase_result = evaluate_phase_starts(
                    model,
                    specs,
                    train_seeds,
                    phase=block.name if block.name in PHASE_NAMES else "pre_hurdle",
                    maximum_student_steps=protocol.maximum_student_steps_per_episode,
                )
                phase_result["m5_student_steps"] = completed_steps
                phase_result["curriculum_block"] = block.name
                phase_history.append(phase_result)
                score = evaluation_score(validation)
                if best_score is None or score > best_score:
                    best_score = score
                    best_step = completed_steps
                    model.save(output_dir / "best_student")
                csv_row = {
                    "m5_student_steps": completed_steps,
                    "curriculum_block": block.name,
                    "success_count": validation["success_count"],
                    "hard_fall_count": validation["hard_fall_count"],
                    "mean_raw_clearances": validation["mean_raw_clearances"],
                    "mean_recovered_obstacles": validation[
                        "mean_recovered_obstacles"
                    ],
                    "mean_max_x": validation["mean_max_x"],
                }
                append_evaluation_csv(
                    output_dir / "validation_checkpoints.csv", csv_row
                )
                print(
                    json.dumps(
                        {
                            "event": "checkpoint",
                            **csv_row,
                            "phase_success_count": phase_result["success_count"],
                            "phase_hard_fall_count": phase_result["hard_fall_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            curriculum_history.append(
                {
                    "name": block.name,
                    "planned_steps": block.steps,
                    "completed_steps": block_completed,
                    "weights": block.weights,
                }
            )
            if stopped_by_user_cap:
                break
        extension_used = False
        planned_training_complete = bool(
            not stopped_by_user_cap and len(curriculum_history) == len(blocks)
        )
        plateau = plateau_detected(
            validation_history,
            window=protocol.plateau_window_checkpoints,
        )
        if (
            planned_training_complete
            and not args.skip_extension
            and not plateau
            and int(validation_history[-1]["success_count"]) < 9
        ):
            extension_used = True
            full_block = protocol.curriculum[-1]
            if curriculum_history[-1]["name"] != "full_start":
                new_environment = make_vector_environment(
                    specs,
                    train_seeds,
                    full_block.weights,
                    seed=args.seed + 900_000,
                    maximum_student_steps=protocol.maximum_student_steps_per_episode,
                    environment_count=environment_count,
                )
                model.set_env(new_environment)
                active_environment.close()
                active_environment = new_environment
            extension_completed = 0
            while extension_completed < protocol.extension_steps:
                chunk = min(
                    protocol.checkpoint_interval_steps,
                    protocol.extension_steps - extension_completed,
                )
                model.learn(
                    total_timesteps=chunk,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                completed_steps += chunk
                extension_completed += chunk
                checkpoint_index += 1
                model.save(
                    checkpoints_dir / f"student_{completed_steps}_m5_steps"
                )
                validation = evaluate_student_batch(
                    model,
                    seeds=validation_seeds,
                    stage=stage,
                    split="validation",
                )
                validation["m5_student_steps"] = completed_steps
                validation["curriculum_block"] = "full_start_extension"
                validation_history.append(validation)
                phase_result = evaluate_phase_starts(
                    model,
                    specs,
                    train_seeds,
                    phase="pre_hurdle",
                    maximum_student_steps=protocol.maximum_student_steps_per_episode,
                )
                phase_result["m5_student_steps"] = completed_steps
                phase_result["curriculum_block"] = "full_start_extension"
                phase_history.append(phase_result)
                score = evaluation_score(validation)
                if best_score is None or score > best_score:
                    best_score = score
                    best_step = completed_steps
                    model.save(output_dir / "best_student")
                csv_row = {
                    "m5_student_steps": completed_steps,
                    "curriculum_block": "full_start_extension",
                    "success_count": validation["success_count"],
                    "hard_fall_count": validation["hard_fall_count"],
                    "mean_raw_clearances": validation["mean_raw_clearances"],
                    "mean_recovered_obstacles": validation[
                        "mean_recovered_obstacles"
                    ],
                    "mean_max_x": validation["mean_max_x"],
                }
                append_evaluation_csv(
                    output_dir / "validation_checkpoints.csv", csv_row
                )
                print(
                    json.dumps(
                        {
                            "event": "extension_checkpoint",
                            **csv_row,
                            "phase_success_count": phase_result["success_count"],
                            "phase_hard_fall_count": phase_result["hard_fall_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            plateau = plateau_detected(
                validation_history,
                window=protocol.plateau_window_checkpoints,
            )
        model.save(output_dir / "final_student")
        best_model = PPO.load(output_dir / "best_student.zip", device="cpu")
        final_validation = evaluate_student_batch(
            model,
            seeds=validation_seeds,
            stage=stage,
            split="validation",
        )
        best_validation = evaluate_student_batch(
            best_model,
            seeds=validation_seeds,
            stage=stage,
            split="validation",
        )
        train_evaluation = evaluate_student_batch(
            best_model,
            seeds=train_seeds,
            stage=stage,
            split="train",
        )
        phase_final = {
            phase: evaluate_phase_starts(
                best_model,
                specs,
                train_seeds,
                phase=phase,
                maximum_student_steps=protocol.maximum_student_steps_per_episode,
            )
            for phase in PHASE_NAMES
        }
        flat_retention = evaluate_flat_retention(
            best_model,
            specs,
            train_seeds,
            maximum_student_steps=protocol.maximum_student_steps_per_episode,
        )
        protected_hashes_after = {
            str(path): sha256_file(path) for path in dict.fromkeys(protected_paths)
        }
        if protected_hashes_after != protected_hashes_before:
            raise RuntimeError("M5中に保護出典が変更された。")
        summary = {
            "method": "m5_unified_student_reverse_curriculum_ppo",
            "run_name": args.run_name,
            "protocol_path": str(protocol.source_path),
            "protocol_sha256": sha256_file(protocol.source_path),
            "source_model_path": str(protocol.source_model_path),
            "source_model_sha256": protocol.source_model_sha256,
            "algorithm": "PPO",
            "device": protocol.device,
            "parallel_environments": environment_count,
            "single_shared_student": True,
            "teacher_role": "training_rollin_actions_only",
            "teacher_module_loaded_in_student_evaluation": False,
            "teacher_actions_after_student_takeover": 0,
            "validation_teacher_interventions": 0,
            "holdout_episodes": 0,
            "planned_steps": protocol.planned_steps,
            "completed_steps": completed_steps,
            "checkpoint_count": checkpoint_index,
            "best_step": best_step,
            "best_score": list(best_score or ()),
            "extension_used": extension_used,
            "plateau_detected_at_end": plateau,
            "stopped_by_user_cap": stopped_by_user_cap,
            "curriculum_history": curriculum_history,
            "initial_validation": validation_history[0],
            "validation_history": validation_history,
            "phase_checkpoint_history": phase_history,
            "final_student_validation": final_validation,
            "best_student_validation": best_validation,
            "best_student_train": train_evaluation,
            "best_student_phase_rollin_evaluation": phase_final,
            "best_student_flat_retention": flat_retention,
            "breakthrough": {
                "criterion": "at_least_one_teacher_free_validation_course_complete",
                "achieved": int(best_validation["success_count"]) >= 1,
                "validation_success_count": int(best_validation["success_count"]),
            },
            "protected_hashes_before": protected_hashes_before,
            "protected_hashes_after": protected_hashes_after,
            "protected_sources_unchanged": True,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary["breakthrough"], ensure_ascii=False), flush=True)
    finally:
        if active_environment is not None:
            active_environment.close()


if __name__ == "__main__":
    main()
