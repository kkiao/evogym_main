"""M2.3.5bの位相別横断教師を模倣初期化し稠密報酬で短く修復する。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch

from general_terrain.audit_phase_reset_curriculum import (
    CURRICULUM_PHASES,
    PhaseResetSpec,
    _replay_to_spec,
)
from general_terrain.audit_rescue_demonstrations import (
    RescueDemoCandidate,
    load_and_validate_branch_arrays,
    load_rescue_demo_manifest,
    sha256_file,
)
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import (
    HALF_RECOVERY_MODEL,
    RAW_RECOVERY_MODEL,
    PortfolioHeight1Teacher,
)
from general_terrain.rescue_reset_manifest import (
    RescueResetManifest,
    load_rescue_reset_manifest,
)
from general_terrain.train_phase_balanced_rescue_teacher import (
    actor_trainable_parameters,
)
from general_terrain.train_prefix_rescue_teacher import hash_policy_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m2_3_5b_phase_crossing_teacher_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "phase_crossing_teacher"


@dataclass(frozen=True)
class CrossingSequence:
    """一つの位相リセットから生越えまでの教師系列を保持する。"""

    reset_id: str
    seed: int
    observations: np.ndarray
    actions: np.ndarray

    @property
    def steps(self) -> int:
        """系列の動作数を返す。"""
        return int(len(self.observations))


def _resolve_project_path(value: str) -> Path:
    """プロジェクト配下だけに限定して相対パスを解決する。"""
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("位相教師規約の出典はプロジェクト配下でなければならない。")
    return path


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結規約、M2.3.5a合格、全出典ハッシュと上限を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M2.3.5b規約は凍結済みでなければならない。")
    path_fields = (
        "phase_audit_summary_path",
        "phase_reset_manifest_path",
        "source_student_model_path",
        "reset_manifest_path",
    )
    resolved = {name: _resolve_project_path(str(payload[name])) for name in path_fields}
    for name in path_fields:
        expected = str(payload[f"{name.removesuffix('_path')}_sha256"])
        if sha256_file(resolved[name]) != expected:
            raise ValueError(f"M2.3.5b出典ハッシュが一致しない: {resolved[name]}")
    audit = json.loads(resolved["phase_audit_summary_path"].read_text(encoding="utf-8"))
    if not bool(audit["m2_3_5a_gate"]["gate_passed"]):
        raise ValueError("M2.3.5a位相リセット門が通過していない。")
    if tuple(payload["crossing_phases"]) != (
        "pre_hurdle",
        "hurdle_deformation",
    ):
        raise ValueError("横断教師の位相分離が凍結値と一致しない。")
    if tuple(payload["recovery_phases_use_old_teacher"]) != (
        "post_clearance_recovery",
        "stable_finish",
    ):
        raise ValueError("回復位相の旧教師経路が凍結値と一致しない。")
    if int(payload["behavior_cloning_epochs"]) != 8:
        raise ValueError("横断模倣初期化は8回でなければならない。")
    if int(payload["ppo_training_steps"]) != 4992:
        raise ValueError("横断PPO試行は4992歩でなければならない。")
    if int(payload["validation_episodes"]) != 0 or int(payload["holdout_episodes"]) != 0:
        raise ValueError("M2.3.5bは検証または留保区分へアクセスできない。")
    return {
        **payload,
        **resolved,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def load_phase_specs(path: Path) -> tuple[PhaseResetSpec, ...]:
    """凍結位相目録を読み込み16個の仕様へ変換する。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("位相リセット目録は凍結済みでなければならない。")
    specs = tuple(PhaseResetSpec(**item) for item in payload["specs"])
    if len(specs) != 16:
        raise ValueError("位相リセット仕様は16個でなければならない。")
    if {spec.phase for spec in specs} != set(CURRICULUM_PHASES):
        raise ValueError("位相リセット目録の位相集合が不正である。")
    return specs


def load_crossing_sequences(
    specs: tuple[PhaseResetSpec, ...],
    candidates: tuple[RescueDemoCandidate, ...],
) -> tuple[CrossingSequence, ...]:
    """横断二位相だけを生越え直前までの模倣系列として抽出する。"""
    candidate_by_seed = {candidate.seed: candidate for candidate in candidates}
    source_audit = json.loads(
        (
            PROJECT_ROOT
            / "runs"
            / "rescue_demo_audit"
            / "m2_3_1_success_demo_audit_v1"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    sequences: list[CrossingSequence] = []
    for spec in specs:
        if spec.phase not in {"pre_hurdle", "hurdle_deformation"}:
            continue
        candidate = candidate_by_seed[spec.seed]
        arrays, _ = load_and_validate_branch_arrays(candidate)
        row = next(item for item in source_audit["candidates"] if int(item["seed"]) == spec.seed)
        recovery_segment = next(
            segment
            for segment in row["teacher_phase_segments"]
            if segment["phase"] == "post_clearance_recovery"
        )
        end = int(recovery_segment["start"])
        if end - spec.source_step <= 32:
            raise ValueError(f"横断系列が短すぎる: {spec.reset_id}")
        sequences.append(
            CrossingSequence(
                reset_id=spec.reset_id,
                seed=spec.seed,
                observations=np.asarray(
                    arrays["observations"][spec.source_step:end],
                    dtype=np.float32,
                ),
                actions=np.asarray(
                    arrays["teacher_actions"][spec.source_step:end],
                    dtype=np.float32,
                ),
            )
        )
    if len(sequences) != 8:
        raise RuntimeError("横断模倣系列は8本でなければならない。")
    return tuple(sequences)


class PhaseCrossingEnv(gym.Env):
    """前壁位相と変形位相を分けて生越えだけを学習させる。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        specs: tuple[PhaseResetSpec, ...],
        *,
        impulse_magnitude: float,
        maximum_steps: int,
        seed: int,
    ) -> None:
        crossing_specs = tuple(
            spec
            for spec in specs
            if spec.phase in {"pre_hurdle", "hurdle_deformation"}
        )
        if len(crossing_specs) != 8:
            raise ValueError("横断訓練には二位相各四仕様が必要である。")
        self.specs = crossing_specs
        self.impulse_magnitude = impulse_magnitude
        self.maximum_steps = maximum_steps
        self.rng = np.random.default_rng(seed)
        self.episode_index = 0
        self.current_spec: PhaseResetSpec | None = None
        self.crossing_steps = 0
        first_course = sample_curriculum_course(
            crossing_specs[0].seed,
            "hurdle_single",
            "train",
        )
        self.environment = GeneralObstacleEnv(
            course=first_course,
            resample_on_reset=False,
        )
        self.observation_space = self.environment.observation_space
        self.action_space = self.environment.action_space
        self.arrays_by_path: dict[str, dict[str, np.ndarray]] = {}

    def _arrays(self, spec: PhaseResetSpec) -> dict[str, np.ndarray]:
        """出典分岐配列をパスごとに一度だけ読み込む。"""
        if spec.source_branch_path not in self.arrays_by_path:
            path = Path(spec.source_branch_path)
            if sha256_file(path) != spec.source_branch_sha256:
                raise ValueError("横断リセット出典のハッシュが一致しない。")
            with np.load(path, allow_pickle=False) as archive:
                self.arrays_by_path[spec.source_branch_path] = {
                    name: np.asarray(archive[name]).copy() for name in archive.files
                }
        return self.arrays_by_path[spec.source_branch_path]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """位相仕様を循環選択し末尾一動作だけを小さく乱して再生する。"""
        super().reset(seed=seed)
        if options and "reset_id" in options:
            reset_id = str(options["reset_id"])
            spec = next(spec for spec in self.specs if spec.reset_id == reset_id)
            randomized = bool(options.get("randomized", False))
        else:
            spec = self.specs[self.episode_index % len(self.specs)]
            self.episode_index += 1
            randomized = True
        course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
        observation, info = self.environment.reset(
            seed=spec.seed,
            options={"course": course},
        )
        arrays = self._arrays(spec)
        impulse = np.zeros(self.action_space.shape, dtype=np.float32)
        if randomized:
            dimension = int(self.rng.integers(0, self.action_space.shape[0]))
            impulse[dimension] = float(
                self.rng.uniform(-self.impulse_magnitude, self.impulse_magnitude)
            )
        for step in range(spec.source_step):
            action = np.asarray(arrays["executed_actions"][step], dtype=np.float32)
            if randomized and step == spec.source_step - 1:
                action = np.clip(
                    action + impulse,
                    self.action_space.low,
                    self.action_space.high,
                ).astype(np.float32)
            observation, _, terminated, truncated, info = self.environment.step(action)
            if terminated or truncated:
                raise RuntimeError(f"横断位相リセットが早期終了した: {spec.reset_id}")
        self.current_spec = spec
        self.crossing_steps = 0
        enriched = dict(info)
        enriched.update(
            {
                "teacher_training_only": True,
                "phase_training_task": spec.phase,
                "phase_training_success": False,
                "student_observation_privileged": False,
                "reset_impulse": impulse.tolist(),
            }
        )
        return np.asarray(observation, dtype=np.float32), enriched

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """生越え、姿勢安全、前進を稠密化し横断完了時に終了する。"""
        observation, base_reward, terminated, truncated, info = self.environment.step(
            np.asarray(action, dtype=np.float32)
        )
        self.crossing_steps += 1
        success = bool(int(info["raw_clearances"]) > 0 and not info["hard_fall"])
        reward = float(base_reward) - 0.005
        reward -= 0.02 * min(1.0, float(info["orientation_error"]) / math.pi)
        if bool(info["upper_body_grounded"]):
            reward -= 0.25
        if bool(info["hard_fall"]):
            reward -= 20.0
            terminated = True
            truncated = False
        if success:
            reward += 30.0
            terminated = True
            truncated = False
        elif self.crossing_steps >= self.maximum_steps and not terminated:
            truncated = True
        enriched = dict(info)
        enriched.update(
            {
                "teacher_training_only": True,
                "phase_training_task": self.current_spec.phase,
                "phase_training_success": success,
                "student_observation_privileged": False,
                "crossing_steps": self.crossing_steps,
            }
        )
        return (
            np.asarray(observation, dtype=np.float32),
            reward,
            bool(terminated),
            bool(truncated),
            enriched,
        )

    def close(self) -> None:
        """内部物理環境を閉じる。"""
        self.environment.close()


def train_crossing_behavior_clone(
    model: Any,
    sequences: tuple[CrossingSequence, ...],
    *,
    epochs: int,
    learning_rate: float,
    handoff_window_steps: int,
    maximum_gradient_norm: float,
    seed: int,
) -> dict[str, object]:
    """引き継ぎ窓と残り横断区間を等重みで模倣する。"""
    torch.manual_seed(seed)
    parameters = actor_trainable_parameters(model)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    history: list[dict[str, object]] = []
    optimizer_steps = 0
    model.policy.train()
    for epoch in range(1, epochs + 1):
        rows: list[dict[str, object]] = []
        for sequence in sequences:
            hidden = torch.zeros(
                model.policy.lstm_hidden_state_shape,
                dtype=torch.float32,
                device=model.device,
            )
            cell = torch.zeros_like(hidden)
            losses: list[torch.Tensor] = []
            for step in range(sequence.steps):
                observation = torch.as_tensor(
                    sequence.observations[step : step + 1],
                    dtype=torch.float32,
                    device=model.device,
                )
                target = torch.as_tensor(
                    sequence.actions[step : step + 1],
                    dtype=torch.float32,
                    device=model.device,
                )
                episode_start = torch.as_tensor(
                    [float(step == 0)],
                    dtype=torch.float32,
                    device=model.device,
                )
                distribution, (hidden, cell) = model.policy.get_distribution(
                    observation,
                    (hidden, cell),
                    episode_start,
                )
                prediction = distribution.distribution.mean
                losses.append(torch.mean((prediction - target) ** 2))
            stacked = torch.stack(losses)
            handoff_loss = stacked[:handoff_window_steps].mean()
            remainder_loss = stacked[handoff_window_steps:].mean()
            loss = 0.5 * (handoff_loss + remainder_loss)
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=maximum_gradient_norm,
            )
            optimizer.step()
            optimizer_steps += 1
            rows.append(
                {
                    "reset_id": sequence.reset_id,
                    "steps": sequence.steps,
                    "loss": float(loss.detach().cpu()),
                    "handoff_loss": float(handoff_loss.detach().cpu()),
                    "remainder_loss": float(remainder_loss.detach().cpu()),
                    "gradient_norm_before_clip": float(gradient_norm),
                }
            )
        history.append({"epoch": epoch, "sequences": rows})
    return {
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "sequence_presentations": epochs * len(sequences),
        "history": history,
    }


def evaluate_crossing_model(
    model: Any,
    specs: tuple[PhaseResetSpec, ...],
    candidates: tuple[RescueDemoCandidate, ...],
    *,
    impulse_magnitude: float,
    maximum_steps: int,
) -> dict[str, object]:
    """横断二位相の生越え能力を精密状態または単発摂動で測る。"""
    candidate_by_seed = {candidate.seed: candidate for candidate in candidates}
    rows: list[dict[str, object]] = []
    for spec in specs:
        if spec.phase not in {"pre_hurdle", "hurdle_deformation"}:
            continue
        environment, observation, info = _replay_to_spec(
            spec,
            candidate_by_seed[spec.seed],
            teacher=None,
        )
        recurrent_state: Any = None
        episode_start = np.ones((1,), dtype=bool)
        terminated = False
        truncated = False
        steps = 0
        try:
            while not (terminated or truncated) and steps < maximum_steps:
                action, recurrent_state = model.predict(
                    observation,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                executed_action = np.asarray(action, dtype=np.float32)
                if steps == 0 and impulse_magnitude > 0.0:
                    impulse = np.zeros(environment.action_space.shape, dtype=np.float32)
                    impulse[spec.impulse_action_dimension] = (
                        spec.impulse_sign * impulse_magnitude
                    )
                    executed_action = np.clip(
                        executed_action + impulse,
                        environment.action_space.low,
                        environment.action_space.high,
                    ).astype(np.float32)
                observation, _, terminated, truncated, info = environment.step(
                    executed_action
                )
                steps += 1
                if int(info["raw_clearances"]) > 0:
                    break
        finally:
            environment.close()
        success = bool(int(info["raw_clearances"]) > 0 and not info["hard_fall"])
        rows.append(
            {
                "reset_id": spec.reset_id,
                "phase": spec.phase,
                "seed": spec.seed,
                "steps": steps,
                "success": success,
                "hard_fall": bool(info["hard_fall"]),
                "failure_reason": str(info["failure_reason"]),
                "raw_clearances": int(info["raw_clearances"]),
            }
        )
    return {
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "rows": rows,
    }


def _candidate_key(
    exact: Mapping[str, object],
    impulse: Mapping[str, object],
) -> tuple[int, int, int]:
    """訓練位相内の成功を優先し転倒を最後に抑える選択キーを返す。"""
    return (
        int(exact["success_count"]) + int(impulse["success_count"]),
        -int(exact["hard_fall_count"]) - int(impulse["hard_fall_count"]),
        int(exact["success_count"]),
    )


def evaluate_composite_phase_specs(
    crossing_model: Any,
    specs: tuple[PhaseResetSpec, ...],
    candidates: tuple[RescueDemoCandidate, ...],
    *,
    impulse_magnitude: float,
) -> dict[str, object]:
    """横断専門モデルから旧教師回復へ切り替え16位相を完走評価する。"""
    candidate_by_seed = {candidate.seed: candidate for candidate in candidates}
    rows: list[dict[str, object]] = []
    for spec in specs:
        old_teacher = PortfolioHeight1Teacher()
        environment, observation, info = _replay_to_spec(
            spec,
            candidate_by_seed[spec.seed],
            teacher=old_teacher,
        )
        recurrent_state: Any = None
        episode_start = np.ones((1,), dtype=bool)
        terminated = False
        truncated = False
        steps = 0
        crossing_steps = 0
        try:
            while not (terminated or truncated):
                old_action, _ = old_teacher.predict(environment, observation, info)
                use_crossing = bool(
                    spec.phase in {"pre_hurdle", "hurdle_deformation"}
                    and int(info["raw_clearances"]) == 0
                )
                if use_crossing:
                    action, recurrent_state = crossing_model.predict(
                        observation,
                        state=recurrent_state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                    episode_start[:] = False
                    crossing_steps += 1
                else:
                    action = old_action
                executed_action = np.asarray(action, dtype=np.float32)
                if steps == 0 and impulse_magnitude > 0.0:
                    impulse = np.zeros(environment.action_space.shape, dtype=np.float32)
                    impulse[spec.impulse_action_dimension] = (
                        spec.impulse_sign * impulse_magnitude
                    )
                    executed_action = np.clip(
                        executed_action + impulse,
                        environment.action_space.low,
                        environment.action_space.high,
                    ).astype(np.float32)
                observation, _, terminated, truncated, info = environment.step(
                    executed_action
                )
                steps += 1
        finally:
            environment.close()
        success = bool(
            info["course_complete"]
            and not info["hard_fall"]
            and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
        )
        rows.append(
            {
                "reset_id": spec.reset_id,
                "phase": spec.phase,
                "seed": spec.seed,
                "steps": steps,
                "crossing_steps": crossing_steps,
                "success": success,
                "hard_fall": bool(info["hard_fall"]),
                "failure_reason": str(info["failure_reason"]),
                "raw_clearances": int(info["raw_clearances"]),
                "recovered_obstacles": int(info["recovered_obstacles"]),
            }
        )
    return {
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "phase_results": {
            phase: {
                "episodes": sum(row["phase"] == phase for row in rows),
                "success_count": sum(
                    bool(row["success"]) and row["phase"] == phase for row in rows
                ),
                "hard_fall_count": sum(
                    bool(row["hard_fall"]) and row["phase"] == phase for row in rows
                ),
            }
            for phase in CURRICULUM_PHASES
        },
        "rows": rows,
    }


def evaluate_composite_reset_states(
    crossing_model: Any,
    frozen_student: Any,
    reset_manifest: RescueResetManifest,
) -> dict[str, object]:
    """十一学生前置状態から横断専門モデルと旧教師回復を連結評価する。"""
    rows: list[dict[str, object]] = []
    for spec in reset_manifest.states:
        course = sample_curriculum_course(spec.seed, "hurdle_single", "train")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        old_teacher = PortfolioHeight1Teacher()
        student_state: Any = None
        student_episode_start = np.ones((1,), dtype=bool)
        crossing_state: Any = None
        crossing_episode_start = np.ones((1,), dtype=bool)
        terminated = False
        truncated = False
        crossing_steps = 0
        try:
            observation, info = environment.reset(seed=spec.seed)
            old_teacher.reset(environment)
            for _ in range(spec.prefix_steps):
                old_teacher.predict(environment, observation, info)
                action, student_state = frozen_student.predict(
                    observation,
                    state=student_state,
                    episode_start=student_episode_start,
                    deterministic=True,
                )
                student_episode_start[:] = False
                observation, _, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    raise RuntimeError(f"十一状態前置再生が早期終了した: {spec.seed}")
            while not (terminated or truncated):
                old_action, _ = old_teacher.predict(environment, observation, info)
                if int(info["raw_clearances"]) == 0:
                    action, crossing_state = crossing_model.predict(
                        observation,
                        state=crossing_state,
                        episode_start=crossing_episode_start,
                        deterministic=True,
                    )
                    crossing_episode_start[:] = False
                    crossing_steps += 1
                else:
                    action = old_action
                observation, _, terminated, truncated, info = environment.step(action)
        finally:
            environment.close()
        success = bool(
            info["course_complete"]
            and not info["hard_fall"]
            and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
        )
        rows.append(
            {
                "seed": spec.seed,
                "start_runway_voxels": spec.start_runway_voxels,
                "crossing_steps": crossing_steps,
                "success": success,
                "hard_fall": bool(info["hard_fall"]),
                "safe_stall": bool(not success and not info["hard_fall"]),
                "failure_reason": str(info["failure_reason"]),
                "raw_clearances": int(info["raw_clearances"]),
                "recovered_obstacles": int(info["recovered_obstacles"]),
            }
        )
    return {
        "episodes": len(rows),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "safe_stall_count": sum(bool(row["safe_stall"]) for row in rows),
        "raw_clearance_count": sum(int(row["raw_clearances"]) > 0 for row in rows),
        "recovery_count": sum(int(row["recovered_obstacles"]) > 0 for row in rows),
        "rows": rows,
    }


def evaluate_gate(
    exact: Mapping[str, object],
    impulse: Mapping[str, object],
    reset: Mapping[str, object],
    requirements: Mapping[str, int],
) -> dict[str, object]:
    """位相精密、位相近傍、十一状態の全条件を同時に判定する。"""
    checks = {
        "phase_exact_success": int(exact["success_count"])
        >= requirements["minimum_phase_exact_success_count"],
        "phase_impulse_success": int(impulse["success_count"])
        >= requirements["minimum_phase_impulse_success_count"],
        "phase_exact_hard_fall": int(exact["hard_fall_count"])
        <= requirements["maximum_phase_exact_hard_fall_count"],
        "phase_impulse_hard_fall": int(impulse["hard_fall_count"])
        <= requirements["maximum_phase_impulse_hard_fall_count"],
        "reset_success": int(reset["success_count"])
        >= requirements["minimum_reset_state_success_count"],
        "reset_hard_fall": int(reset["hard_fall_count"])
        <= requirements["maximum_reset_state_hard_fall_count"],
    }
    return {
        "requirements": dict(requirements),
        "checks": checks,
        "gate_passed": all(checks.values()),
        "eligible_for_m2_4": all(checks.values()),
    }


def run(protocol: dict[str, object], output_dir: Path, seed: int) -> dict[str, object]:
    """模倣、稠密PPO、候補選択、位相と十一状態の評価を実行する。"""
    from sb3_contrib import RecurrentPPO

    specs = load_phase_specs(Path(protocol["phase_reset_manifest_path"]))
    demo_manifest = load_rescue_demo_manifest()
    sequences = load_crossing_sequences(specs, demo_manifest.candidates)
    reset_manifest = load_rescue_reset_manifest(Path(protocol["reset_manifest_path"]))
    protected_paths = (
        Path(protocol["source_path"]),
        Path(protocol["phase_audit_summary_path"]),
        Path(protocol["phase_reset_manifest_path"]),
        Path(protocol["source_student_model_path"]),
        Path(protocol["reset_manifest_path"]),
        RAW_RECOVERY_MODEL.resolve(),
        HALF_RECOVERY_MODEL.resolve(),
        *(candidate.branch_path for candidate in demo_manifest.candidates),
    )
    hashes_before = {str(path): sha256_file(path) for path in protected_paths}
    output_dir.mkdir(parents=True, exist_ok=False)
    initialization = output_dir / "crossing_init_from_frozen_student.zip"
    shutil.copy2(Path(protocol["source_student_model_path"]), initialization)
    if sha256_file(initialization) != str(protocol["source_student_model_sha256"]):
        raise RuntimeError("横断教師初期化コピーのハッシュが一致しない。")
    model = RecurrentPPO.load(initialization, device="cpu")
    parameter_hash_before = hash_policy_parameters(model)
    bc_result = train_crossing_behavior_clone(
        model,
        sequences,
        epochs=int(protocol["behavior_cloning_epochs"]),
        learning_rate=float(protocol["behavior_cloning_learning_rate"]),
        handoff_window_steps=int(protocol["handoff_window_steps"]),
        maximum_gradient_norm=float(protocol["maximum_gradient_norm"]),
        seed=seed,
    )
    bc_checkpoint = output_dir / "crossing_after_phase_bc.zip"
    model.save(bc_checkpoint)
    bc_exact = evaluate_crossing_model(
        model,
        specs,
        demo_manifest.candidates,
        impulse_magnitude=0.0,
        maximum_steps=int(protocol["maximum_crossing_steps"]),
    )
    bc_impulse = evaluate_crossing_model(
        model,
        specs,
        demo_manifest.candidates,
        impulse_magnitude=float(protocol["random_prefix_impulse_magnitude"]),
        maximum_steps=int(protocol["maximum_crossing_steps"]),
    )
    training_environment = PhaseCrossingEnv(
        specs,
        impulse_magnitude=float(protocol["random_prefix_impulse_magnitude"]),
        maximum_steps=int(protocol["maximum_crossing_steps"]),
        seed=seed,
    )
    try:
        model.set_env(training_environment)
        model.learn(
            total_timesteps=int(protocol["ppo_training_steps"]),
            reset_num_timesteps=False,
            progress_bar=False,
        )
    finally:
        training_environment.close()
    ppo_checkpoint = output_dir / "crossing_after_dense_phase_ppo.zip"
    model.save(ppo_checkpoint)
    ppo_exact = evaluate_crossing_model(
        model,
        specs,
        demo_manifest.candidates,
        impulse_magnitude=0.0,
        maximum_steps=int(protocol["maximum_crossing_steps"]),
    )
    ppo_impulse = evaluate_crossing_model(
        model,
        specs,
        demo_manifest.candidates,
        impulse_magnitude=float(protocol["random_prefix_impulse_magnitude"]),
        maximum_steps=int(protocol["maximum_crossing_steps"]),
    )
    bc_key = _candidate_key(bc_exact, bc_impulse)
    ppo_key = _candidate_key(ppo_exact, ppo_impulse)
    if ppo_key > bc_key:
        selected_name = "dense_phase_ppo"
        selected_path = ppo_checkpoint
        selected_exact_crossing = ppo_exact
        selected_impulse_crossing = ppo_impulse
    else:
        selected_name = "phase_behavior_cloning"
        selected_path = bc_checkpoint
        selected_exact_crossing = bc_exact
        selected_impulse_crossing = bc_impulse
    selected_model = RecurrentPPO.load(selected_path, device="cpu")
    phase_exact = evaluate_composite_phase_specs(
        selected_model,
        specs,
        demo_manifest.candidates,
        impulse_magnitude=0.0,
    )
    phase_impulse = evaluate_composite_phase_specs(
        selected_model,
        specs,
        demo_manifest.candidates,
        impulse_magnitude=float(protocol["random_prefix_impulse_magnitude"]),
    )
    frozen_student = RecurrentPPO.load(
        Path(protocol["source_student_model_path"]),
        device="cpu",
    )
    reset_evaluation = evaluate_composite_reset_states(
        selected_model,
        frozen_student,
        reset_manifest,
    )
    gate = evaluate_gate(
        phase_exact,
        phase_impulse,
        reset_evaluation,
        protocol["gate"],
    )
    selected_output = output_dir / "selected_crossing_teacher.zip"
    shutil.copy2(selected_path, selected_output)
    hashes_after = {str(path): sha256_file(path) for path in protected_paths}
    if hashes_after != hashes_before:
        raise RuntimeError("M2.3.5b中に凍結出典が変更された。")
    result = {
        "method": "m2_3_5b_phase_separated_crossing_teacher_repair",
        "stage": "hurdle_single",
        "split": "train",
        "run_name": output_dir.name,
        "seed": seed,
        "protocol": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in protocol.items()
                if key != "source_path"
            },
            "source_path": str(protocol["source_path"]),
        },
        "protocol_sha256": protocol["sha256"],
        "teacher_training_only": True,
        "dataset": {
            "sequence_count": len(sequences),
            "steps": sum(sequence.steps for sequence in sequences),
            "rows": [
                {
                    "reset_id": sequence.reset_id,
                    "seed": sequence.seed,
                    "steps": sequence.steps,
                }
                for sequence in sequences
            ],
        },
        "behavior_cloning": bc_result,
        "candidate_evaluations": {
            "phase_behavior_cloning": {
                "key": list(bc_key),
                "exact_crossing": bc_exact,
                "impulse_crossing": bc_impulse,
                "checkpoint": str(bc_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(bc_checkpoint),
            },
            "dense_phase_ppo": {
                "key": list(ppo_key),
                "exact_crossing": ppo_exact,
                "impulse_crossing": ppo_impulse,
                "checkpoint": str(ppo_checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(ppo_checkpoint),
            },
        },
        "selected_candidate": selected_name,
        "selected_crossing_evaluation": {
            "exact": selected_exact_crossing,
            "impulse": selected_impulse_crossing,
        },
        "selected_checkpoint": str(selected_output.resolve()),
        "selected_checkpoint_sha256": sha256_file(selected_output),
        "phase_exact_composite_evaluation": phase_exact,
        "phase_impulse_composite_evaluation": phase_impulse,
        "reset_state_composite_evaluation": reset_evaluation,
        "gate": gate,
        "checkpoint_disposition": (
            "m2_4_candidate"
            if bool(gate["eligible_for_m2_4"])
            else "quarantined_m2_3_5b_phase_teacher"
        ),
        "source_student_parameter_hash": parameter_hash_before,
        "selected_teacher_parameter_hash": hash_policy_parameters(selected_model),
        "student_weights_updated": False,
        "teacher_weights_updated": True,
        "ppo_training_steps": int(protocol["ppo_training_steps"]),
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "teacher_interventions_in_student_test": 0,
        "protected_source_files_unchanged": True,
        "eligible_for_m2_4": bool(gate["eligible_for_m2_4"]),
        "eligible_for_final_student_test": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    """凍結規約、出力名、乱数種だけを受け取る。"""
    parser = argparse.ArgumentParser(
        description="位相別横断教師を模倣と稠密PPOで一回だけ修復する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    """M2.3.5bの有界位相教師試行を実行する。"""
    args = build_argument_parser().parse_args()
    torch.set_num_threads(1)
    protocol = load_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
