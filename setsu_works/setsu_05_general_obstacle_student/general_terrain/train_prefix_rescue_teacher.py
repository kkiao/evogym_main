"""学生プレフィックス分布で訓練専用救援教師の短期試行を実行する。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Protocol

import gymnasium as gym
import numpy as np

from general_terrain.rescue_reset_manifest import (
    DEFAULT_RESCUE_RESET_MANIFEST,
    RescueResetManifest,
    load_rescue_reset_manifest,
)
from general_terrain.student_prefix_rescue_env import (
    RESCUE_PHASES,
    RESCUE_REWARD_VERSION,
    StudentPrefixRescueEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "height1_recurrent_dagger_student"
    / "height1_recurrent_dagger_seed7_v1"
    / "best_model.zip"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "prefix_rescue_teacher"
M2_3_MAX_TRAINING_STEPS = 5_000
M2_1_BASELINE = {
    "profile": "m2_1_near6",
    "evaluation_episodes": 11,
    "success_count": 1,
    "hard_fall_count": 1,
    "safe_stall_count": 9,
}


class RecurrentController(Protocol):
    """循環方策による決定論的評価に必要な契約を定義する。"""

    def predict(
        self,
        observation: np.ndarray,
        *,
        state: Any,
        episode_start: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.ndarray, Any]:
        """現在の観測から動作と次の循環状態を返す。"""


@dataclass(frozen=True)
class RescueTeacherEpisodeResult:
    """一つの凍結引き継ぎ状態における教師評価結果を保持する。"""

    seed: int
    start_runway_voxels: int
    rescue_steps: int
    total_reward: float
    rescue_success: bool
    course_complete: bool
    hard_fall: bool
    sequence_failed: bool
    safe_stall: bool
    failure_reason: str
    raw_clearances: int
    recovered_obstacles: int
    final_phase: str
    phase_step_counts: dict[str, int]
    maximum_angle_degrees: float
    upper_body_contact_steps: int
    maximum_stall_steps: int
    maximum_com_x: float

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存できる辞書形式を返す。"""
        return asdict(self)


def _empty_phase_counts() -> dict[str, int]:
    """四つの救援段階をゼロで初期化する。"""
    return {phase: 0 for phase in RESCUE_PHASES}


def sha256_file(path: Path) -> str:
    """指定ファイルのSHA-256を小文字で返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_policy_parameters(model: Any) -> str:
    """方策の全パラメータとバッファから安定したハッシュを作る。"""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def compute_rollout_budget(max_training_steps: int, rollout_steps: int) -> int:
    """上限を超えない最大の完全ロールアウト歩数を返す。"""
    if not 1 <= max_training_steps <= M2_3_MAX_TRAINING_STEPS:
        raise ValueError("M2.3の訓練上限は1歩から5000歩まででなければならない。")
    if rollout_steps < 1:
        raise ValueError("ロールアウト歩数は1以上でなければならない。")
    budget = (max_training_steps // rollout_steps) * rollout_steps
    if budget < 1:
        raise ValueError("訓練上限は一回のロールアウトより小さくできない。")
    return budget


def copy_teacher_initialization(source: Path, destination: Path) -> str:
    """凍結学生を独立した教師初期化ファイルへ複製し検証する。"""
    source_hash = sha256_file(source)
    shutil.copy2(source, destination)
    copied_hash = sha256_file(destination)
    if copied_hash != source_hash:
        raise RuntimeError("教師初期化コピーのハッシュが凍結学生と一致しない。")
    return copied_hash


def _is_safe_stall(info: Mapping[str, object], *, maximum_stall_steps: int) -> bool:
    """転倒せず進行停止または時間切れになった失敗を判定する。"""
    if bool(info.get("rescue_success", False)) or bool(info.get("hard_fall", False)):
        return False
    return bool(
        maximum_stall_steps >= 20
        or info.get("stall_limit_reached", False)
        or info.get("time_limit_reached", False)
        or info.get("rescue_step_limit_reached", False)
    )


def evaluate_rescue_teacher(
    environment: Any,
    teacher: RecurrentController,
    *,
    reset_seeds: tuple[int, ...],
) -> dict[str, object]:
    """同じ凍結訓練状態だけで救援教師を決定論的に評価する。"""
    episodes: list[RescueTeacherEpisodeResult] = []
    total_phase_counts = _empty_phase_counts()
    phase_episode_counts = _empty_phase_counts()
    for reset_seed in reset_seeds:
        observation, info = environment.reset(
            options={"prefix_seed": int(reset_seed)}
        )
        if not bool(info.get("teacher_training_only", False)):
            raise RuntimeError("評価環境が教師訓練専用として識別されていない。")
        if bool(info.get("student_controller_active", True)):
            raise RuntimeError("教師評価中に学生制御が有効になっている。")
        if bool(info.get("student_observation_privileged", True)):
            raise RuntimeError("学生観測へ特権情報が混入している。")
        recurrent_state: Any = None
        episode_start = np.ones((1,), dtype=bool)
        terminated = False
        truncated = False
        rescue_steps = 0
        total_reward = 0.0
        maximum_angle = float(info.get("orientation_error", 0.0))
        maximum_stall_steps = int(info.get("stall_steps", 0))
        upper_body_contact_steps = 0
        phase_counts = _empty_phase_counts()
        visited_phases: set[str] = set()
        while not (terminated or truncated):
            action, recurrent_state = teacher.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            action_array = np.asarray(action, dtype=np.float32).reshape(
                environment.action_space.shape
            )
            observation, reward, terminated, truncated, info = environment.step(
                action_array
            )
            episode_start[:] = False
            rescue_steps += 1
            total_reward += float(reward)
            maximum_angle = max(
                maximum_angle,
                float(info.get("orientation_error", 0.0)),
            )
            maximum_stall_steps = max(
                maximum_stall_steps,
                int(info.get("stall_steps", 0)),
            )
            upper_body_contact_steps += int(
                bool(info.get("upper_body_grounded", False))
            )
            phase = str(info["rescue_phase"])
            if phase not in phase_counts:
                raise RuntimeError(f"未知の救援段階が返された: {phase}")
            phase_counts[phase] += 1
            total_phase_counts[phase] += 1
            visited_phases.add(phase)
        for phase in visited_phases:
            phase_episode_counts[phase] += 1
        success = bool(info.get("rescue_success", False))
        hard_fall = bool(info.get("hard_fall", False))
        episodes.append(
            RescueTeacherEpisodeResult(
                seed=int(reset_seed),
                start_runway_voxels=int(
                    environment.current_spec.start_runway_voxels
                ),
                rescue_steps=rescue_steps,
                total_reward=total_reward,
                rescue_success=success,
                course_complete=bool(info.get("course_complete", False)),
                hard_fall=hard_fall,
                sequence_failed=bool(info.get("sequence_failed", False)),
                safe_stall=_is_safe_stall(
                    info,
                    maximum_stall_steps=maximum_stall_steps,
                ),
                failure_reason=str(info.get("failure_reason", "")),
                raw_clearances=int(info.get("raw_clearances", 0)),
                recovered_obstacles=int(info.get("recovered_obstacles", 0)),
                final_phase=str(info["rescue_phase"]),
                phase_step_counts=phase_counts,
                maximum_angle_degrees=math.degrees(maximum_angle),
                upper_body_contact_steps=upper_body_contact_steps,
                maximum_stall_steps=maximum_stall_steps,
                maximum_com_x=float(info.get("max_x_position", 0.0)),
            )
        )
    serialized = [episode.as_dict() for episode in episodes]
    return {
        "method": "deterministic_prefix_rescue_teacher_evaluation",
        "controller_mode": "training_rescue_teacher_only",
        "stage": "hurdle_single",
        "split": "train",
        "evaluation_episodes": len(episodes),
        "teacher_training_only": True,
        "student_controller_active_after_reset": False,
        "student_observation_privileged": False,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "success_count": sum(episode.rescue_success for episode in episodes),
        "hard_fall_count": sum(episode.hard_fall for episode in episodes),
        "safe_stall_count": sum(episode.safe_stall for episode in episodes),
        "sequence_failure_count": sum(
            episode.sequence_failed for episode in episodes
        ),
        "raw_clearance_count": sum(
            episode.raw_clearances > 0 for episode in episodes
        ),
        "recovery_count": sum(
            episode.recovered_obstacles > 0 for episode in episodes
        ),
        "mean_total_reward": float(
            np.mean([episode.total_reward for episode in episodes])
        ),
        "mean_rescue_steps": float(
            np.mean([episode.rescue_steps for episode in episodes])
        ),
        "phase_step_counts": total_phase_counts,
        "phase_episode_counts": phase_episode_counts,
        "episodes": serialized,
    }


def evaluate_m2_3_gate(evaluation: Mapping[str, object]) -> dict[str, object]:
    """M2.3の継続門を完走、転倒、停滞の三条件で判定する。"""
    success_count = int(evaluation["success_count"])
    hard_fall_count = int(evaluation["hard_fall_count"])
    safe_stall_count = int(evaluation["safe_stall_count"])
    success_requirement_passed = success_count >= 4
    hard_fall_requirement_passed = hard_fall_count <= 1
    stall_improvement_passed = bool(
        success_count > int(M2_1_BASELINE["success_count"])
        and safe_stall_count < int(M2_1_BASELINE["safe_stall_count"])
    )
    return {
        "gate_name": "m2_3_short_pilot_continue_gate_v1",
        "required_success_count": 4,
        "maximum_hard_fall_count": 1,
        "baseline_safe_stall_count": int(M2_1_BASELINE["safe_stall_count"]),
        "success_requirement_passed": success_requirement_passed,
        "hard_fall_requirement_passed": hard_fall_requirement_passed,
        "stall_improvement_passed": stall_improvement_passed,
        "continue_to_m2_4": bool(
            success_requirement_passed
            and hard_fall_requirement_passed
            and stall_improvement_passed
        ),
    }


class RescueTrainingAuditWrapper(gym.Wrapper):
    """訓練動作ごとの救援段階と終了結果を方策外で記録する。"""

    def __init__(self, environment: gym.Env) -> None:
        super().__init__(environment)
        self.phase_step_counts = _empty_phase_counts()
        self.reset_seed_counts: Counter[int] = Counter()
        self.completed_episodes: list[dict[str, object]] = []

    def reset(self, **kwargs):
        """内部環境をリセットし、使用した訓練状態だけを記録する。"""
        observation, info = self.env.reset(**kwargs)
        self.reset_seed_counts[int(info["rescue_reset_seed"])] += 1
        return observation, info

    def step(self, action):
        """一歩進め、段階訪問と終了結果を監査用に蓄積する。"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        phase = str(info["rescue_phase"])
        self.phase_step_counts[phase] += 1
        if terminated or truncated:
            self.completed_episodes.append(
                {
                    "seed": int(info["rescue_reset_seed"]),
                    "rescue_steps": int(info["rescue_steps"]),
                    "rescue_success": bool(info.get("rescue_success", False)),
                    "hard_fall": bool(info.get("hard_fall", False)),
                    "sequence_failed": bool(info.get("sequence_failed", False)),
                    "failure_reason": str(info.get("failure_reason", "")),
                    "final_phase": phase,
                }
            )
        return observation, reward, terminated, truncated, info

    def snapshot(self) -> dict[str, object]:
        """JSON保存用の訓練監査要約を返す。"""
        return {
            "observed_step_count": sum(self.phase_step_counts.values()),
            "phase_step_counts": dict(self.phase_step_counts),
            "reset_seed_counts": {
                str(seed): count
                for seed, count in sorted(self.reset_seed_counts.items())
            },
            "completed_episode_count": len(self.completed_episodes),
            "completed_episodes": list(self.completed_episodes),
        }


def _make_rescue_environment(
    prefix_student: RecurrentController,
    manifest: RescueResetManifest,
    *,
    max_rescue_steps: int,
) -> StudentPrefixRescueEnv:
    """凍結目録から訓練専用救援環境を一つ作る。"""
    return StudentPrefixRescueEnv(
        prefix_student,
        manifest.states,
        stage=manifest.stage,
        max_rescue_steps=max_rescue_steps,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """M2.3の固定範囲だけを公開する引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="学生プレフィックス分布で訓練専用救援教師を短期訓練する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--student-model", default=str(DEFAULT_STUDENT_MODEL))
    parser.add_argument("--manifest", default=str(DEFAULT_RESCUE_RESET_MANIFEST))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-training-steps", type=int, default=5_000)
    parser.add_argument("--max-rescue-steps", type=int, default=800)
    return parser


def main() -> None:
    """独立コピーの作成、短期訓練、同一状態評価を順に実行する。"""
    import torch
    from sb3_contrib import RecurrentPPO

    args = build_argument_parser().parse_args()
    torch.set_num_threads(1)
    manifest_path = Path(args.manifest).resolve()
    student_path = Path(args.student_model).resolve()
    manifest = load_rescue_reset_manifest(manifest_path)
    if manifest.stage != "hurdle_single" or manifest.split != "train":
        raise ValueError("M2.3は単一低壁の訓練区分だけを使用できる。")
    student_hash_before = sha256_file(student_path)
    if student_hash_before != manifest.student_model_sha256:
        raise ValueError("凍結学生のハッシュがM2.2目録と一致しない。")

    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    initialization_path = output_dir / "teacher_init_from_student.zip"
    initialization_hash = copy_teacher_initialization(
        student_path,
        initialization_path,
    )
    run_config = {
        "method": "m2_3_prefix_rescue_teacher_short_pilot",
        "stage": manifest.stage,
        "split": manifest.split,
        "run_name": args.run_name,
        "seed": args.seed,
        "requested_max_training_steps": args.max_training_steps,
        "absolute_training_step_cap": M2_3_MAX_TRAINING_STEPS,
        "max_rescue_steps": args.max_rescue_steps,
        "student_model": str(student_path),
        "student_model_sha256": student_hash_before,
        "teacher_initialization": str(initialization_path.resolve()),
        "teacher_initialization_sha256": initialization_hash,
        "manifest": manifest.as_dict(),
        "validation_episodes": 0,
        "holdout_episodes": 0,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prefix_student = RecurrentPPO.load(student_path, device="cpu")
    teacher = RecurrentPPO.load(initialization_path, device="cpu")
    reset_seeds = tuple(spec.seed for spec in manifest.states)

    initial_evaluation_environment = _make_rescue_environment(
        prefix_student,
        manifest,
        max_rescue_steps=args.max_rescue_steps,
    )
    evaluation_start = time.perf_counter()
    try:
        initial_evaluation = evaluate_rescue_teacher(
            initial_evaluation_environment,
            teacher,
            reset_seeds=reset_seeds,
        )
    finally:
        initial_evaluation_environment.close()
    initial_evaluation_seconds = time.perf_counter() - evaluation_start

    base_training_environment = _make_rescue_environment(
        prefix_student,
        manifest,
        max_rescue_steps=args.max_rescue_steps,
    )
    training_environment = RescueTrainingAuditWrapper(base_training_environment)
    teacher.set_env(training_environment)
    teacher.set_random_seed(args.seed)
    rollout_budget = compute_rollout_budget(
        args.max_training_steps,
        int(teacher.n_steps),
    )
    parameter_hash_before = hash_policy_parameters(teacher)
    training_start = time.perf_counter()
    try:
        teacher.learn(
            total_timesteps=rollout_budget,
            reset_num_timesteps=True,
            progress_bar=False,
            log_interval=1,
        )
        actual_training_steps = int(teacher.num_timesteps)
        training_audit = training_environment.snapshot()
    finally:
        training_environment.close()
    training_seconds = time.perf_counter() - training_start
    if actual_training_steps > M2_3_MAX_TRAINING_STEPS:
        raise RuntimeError("実訓練歩数がM2.3の5000歩上限を超えた。")
    if int(training_audit["observed_step_count"]) != actual_training_steps:
        raise RuntimeError("環境監査歩数と方策訓練歩数が一致しない。")
    parameter_hash_after = hash_policy_parameters(teacher)
    checkpoint_path = output_dir / f"teacher_after_{actual_training_steps}_steps.zip"
    teacher.save(checkpoint_path)

    final_evaluation_environment = _make_rescue_environment(
        prefix_student,
        manifest,
        max_rescue_steps=args.max_rescue_steps,
    )
    final_evaluation_start = time.perf_counter()
    try:
        final_evaluation = evaluate_rescue_teacher(
            final_evaluation_environment,
            teacher,
            reset_seeds=reset_seeds,
        )
    finally:
        final_evaluation_environment.close()
    final_evaluation_seconds = time.perf_counter() - final_evaluation_start

    student_hash_after = sha256_file(student_path)
    if student_hash_after != student_hash_before:
        raise RuntimeError("M2.3中に凍結学生ファイルが変更された。")
    gate = evaluate_m2_3_gate(final_evaluation)
    summary = {
        **run_config,
        "status": "complete",
        "reward_version": RESCUE_REWARD_VERSION,
        "rollout_steps": int(teacher.n_steps),
        "actual_training_steps": actual_training_steps,
        "student_weights_updated": False,
        "teacher_weights_updated": True,
        "student_model_sha256_after": student_hash_after,
        "student_model_unchanged": student_hash_after == student_hash_before,
        "teacher_parameter_sha256_before": parameter_hash_before,
        "teacher_parameter_sha256_after": parameter_hash_after,
        "teacher_parameters_changed": parameter_hash_after != parameter_hash_before,
        "teacher_checkpoint": str(checkpoint_path.resolve()),
        "teacher_checkpoint_sha256": sha256_file(checkpoint_path),
        "m2_1_comparison_baseline": dict(M2_1_BASELINE),
        "initialization_evaluation": initial_evaluation,
        "training_audit": training_audit,
        "final_evaluation": final_evaluation,
        "m2_3_gate": gate,
        "checkpoint_disposition": (
            "m2_4_candidate"
            if bool(gate["continue_to_m2_4"])
            else "quarantined_failed_pilot"
        ),
        "eligible_for_m2_4": bool(gate["continue_to_m2_4"]),
        "eligible_for_student_initialization": False,
        "eligible_for_final_student_test": False,
        "timing_seconds": {
            "initial_evaluation": initial_evaluation_seconds,
            "training": training_seconds,
            "final_evaluation": final_evaluation_seconds,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
