"""学生誤差尺度の教師摂動からM2.3.4の回復示範を一回だけ作る。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import torch

from general_terrain.audit_rescue_demonstrations import (
    KEY_TEACHER_PHASES,
    PHASE_CODES,
    RescueDemoCandidate,
    find_true_segments,
    load_and_validate_branch_arrays,
    load_rescue_demo_manifest,
    sha256_file,
)
from general_terrain.curriculum import sample_curriculum_course
from general_terrain.diagnose_closed_loop_imitation_fidelity import (
    evaluate_closed_loop_handoff,
)
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import (
    HALF_RECOVERY_MODEL,
    RAW_RECOVERY_MODEL,
    PortfolioHeight1Teacher,
)
from general_terrain.rescue_reset_manifest import load_rescue_reset_manifest
from general_terrain.student_prefix_rescue_env import (
    StudentPrefixRescueEnv,
    classify_rescue_phase,
)
from general_terrain.train_interactive_correction_rescue_teacher import (
    _evaluate_gate,
    _module_hashes,
)
from general_terrain.train_phase_balanced_rescue_teacher import (
    PhaseBalancedSequence,
    actor_trainable_parameters,
    load_phase_balanced_sequences,
    load_training_protocol,
)
from general_terrain.train_prefix_rescue_teacher import (
    evaluate_rescue_teacher,
    hash_policy_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m2_3_4_perturbed_recovery_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "perturbed_recovery_rescue_teacher"


@dataclass(frozen=True)
class PerturbedRecoveryProtocol:
    """摂動収集と境界重み更新の凍結条件を保持する。"""

    version: str
    stage: str
    split: str
    teacher_training_only: bool
    failed_interactive_summary_path: Path
    failed_interactive_summary_sha256: str
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    source_training_summary_path: Path
    source_training_summary_sha256: str
    demonstration_manifest_path: Path
    demonstration_manifest_sha256: str
    reset_manifest_path: Path
    reset_manifest_sha256: str
    collection_seeds: tuple[int, ...]
    maximum_collection_episodes: int
    noise_residual_rms_fraction: float
    maximum_absolute_noise: float
    minimum_accepted_branches_for_update: int
    handoff_window_steps: int
    epochs: int
    learning_rate: float
    maximum_gradient_norm: float
    weight_decay: float
    loss_groups: tuple[str, ...]
    ppo_training_steps: int
    validation_episodes: int
    holdout_episodes: int
    gate: dict[str, int]
    source_path: Path
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存できる辞書を返す。"""
        result = asdict(self)
        for name in (
            "failed_interactive_summary_path",
            "source_checkpoint_path",
            "source_training_summary_path",
            "demonstration_manifest_path",
            "reset_manifest_path",
            "source_path",
        ):
            result[name] = str(result[name])
        result["collection_seeds"] = list(self.collection_seeds)
        result["loss_groups"] = list(self.loss_groups)
        return result


def _resolve_project_path(value: str) -> Path:
    """プロジェクト内に限定して相対パスを解決する。"""
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("摂動回復の出典はプロジェクト配下でなければならない。")
    return path


def load_perturbed_recovery_protocol(
    path: Path = DEFAULT_PROTOCOL,
) -> PerturbedRecoveryProtocol:
    """凍結規約と失敗経路の切替根拠を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("摂動回復規約は凍結済みでなければならない。")
    protocol = PerturbedRecoveryProtocol(
        version=str(payload["version"]),
        stage=str(payload["stage"]),
        split=str(payload["split"]),
        teacher_training_only=bool(payload["teacher_training_only"]),
        failed_interactive_summary_path=_resolve_project_path(
            str(payload["failed_interactive_summary_path"])
        ),
        failed_interactive_summary_sha256=str(
            payload["failed_interactive_summary_sha256"]
        ),
        source_checkpoint_path=_resolve_project_path(
            str(payload["source_checkpoint_path"])
        ),
        source_checkpoint_sha256=str(payload["source_checkpoint_sha256"]),
        source_training_summary_path=_resolve_project_path(
            str(payload["source_training_summary_path"])
        ),
        source_training_summary_sha256=str(
            payload["source_training_summary_sha256"]
        ),
        demonstration_manifest_path=_resolve_project_path(
            str(payload["demonstration_manifest_path"])
        ),
        demonstration_manifest_sha256=str(
            payload["demonstration_manifest_sha256"]
        ),
        reset_manifest_path=_resolve_project_path(
            str(payload["reset_manifest_path"])
        ),
        reset_manifest_sha256=str(payload["reset_manifest_sha256"]),
        collection_seeds=tuple(int(seed) for seed in payload["collection_seeds"]),
        maximum_collection_episodes=int(payload["maximum_collection_episodes"]),
        noise_residual_rms_fraction=float(payload["noise_residual_rms_fraction"]),
        maximum_absolute_noise=float(payload["maximum_absolute_noise"]),
        minimum_accepted_branches_for_update=int(
            payload["minimum_accepted_branches_for_update"]
        ),
        handoff_window_steps=int(payload["handoff_window_steps"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        maximum_gradient_norm=float(payload["maximum_gradient_norm"]),
        weight_decay=float(payload["weight_decay"]),
        loss_groups=tuple(str(name) for name in payload["loss_groups"]),
        ppo_training_steps=int(payload["ppo_training_steps"]),
        validation_episodes=int(payload["validation_episodes"]),
        holdout_episodes=int(payload["holdout_episodes"]),
        gate={str(name): int(value) for name, value in payload["gate"].items()},
        source_path=source_path,
        sha256=sha256_file(source_path),
    )
    for source, expected in (
        (
            protocol.failed_interactive_summary_path,
            protocol.failed_interactive_summary_sha256,
        ),
        (protocol.source_checkpoint_path, protocol.source_checkpoint_sha256),
        (
            protocol.source_training_summary_path,
            protocol.source_training_summary_sha256,
        ),
        (
            protocol.demonstration_manifest_path,
            protocol.demonstration_manifest_sha256,
        ),
        (protocol.reset_manifest_path, protocol.reset_manifest_sha256),
    ):
        if sha256_file(source) != expected:
            raise ValueError(f"摂動回復出典ハッシュが一致しない: {source}")
    failed_summary = json.loads(
        protocol.failed_interactive_summary_path.read_text(encoding="utf-8")
    )
    if bool(failed_summary["interactive_route_has_net_improvement"]):
        raise ValueError("対話的訂正が改善しているため摂動経路へ切替できない。")
    if failed_summary["fallback_route"] != (
        "learner_error_scaled_perturbed_teacher_recovery"
    ):
        raise ValueError("対話的訂正要約が摂動回復を選択していない。")
    if protocol.maximum_collection_episodes != 4:
        raise ValueError("摂動回復の収集は最大4回でなければならない。")
    if protocol.noise_residual_rms_fraction != 0.25:
        raise ValueError("摂動比率は学生誤差RMSの0.25倍でなければならない。")
    if protocol.maximum_absolute_noise != 0.08:
        raise ValueError("一動作の摂動上限は0.08でなければならない。")
    if protocol.handoff_window_steps != 32:
        raise ValueError("引き継ぎ強調窓は32歩でなければならない。")
    if protocol.epochs != 4 or protocol.learning_rate != 5e-5:
        raise ValueError("摂動回復の更新回数または学習率が凍結値と異なる。")
    if protocol.ppo_training_steps != 0:
        raise ValueError("摂動回復でPPOを実行することはできない。")
    if protocol.validation_episodes != 0 or protocol.holdout_episodes != 0:
        raise ValueError("摂動回復は検証または留保区分へアクセスできない。")
    return protocol


def estimate_action_residual_rms(
    model: Any,
    sequences: tuple[PhaseBalancedSequence, ...],
) -> np.ndarray:
    """保存教師観測上の学生動作残差RMSを動作次元ごとに求める。"""
    residuals: list[np.ndarray] = []
    for sequence in sequences:
        recurrent_state: Any = None
        episode_start = np.ones((1,), dtype=bool)
        for observation, target in zip(sequence.observations, sequence.actions):
            action, recurrent_state = model.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start[:] = False
            residuals.append(
                np.asarray(action, dtype=np.float64).reshape(-1)
                - np.asarray(target, dtype=np.float64).reshape(-1)
            )
    matrix = np.asarray(residuals, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 6:
        raise ValueError("学生動作残差は六次元でなければならない。")
    return np.sqrt(np.mean(matrix**2, axis=0))


def collect_perturbed_teacher_recovery(
    teacher: PortfolioHeight1Teacher,
    candidate: RescueDemoCandidate,
    *,
    noise_standard_deviation: np.ndarray,
    maximum_absolute_noise: float,
    rng: np.random.Generator,
    output_path: Path,
) -> tuple[PhaseBalancedSequence | None, dict[str, object]]:
    """教師動作へ小摂動を加え、次状態からの閉ループ訂正を収集する。"""
    source_arrays, _ = load_and_validate_branch_arrays(candidate)
    takeover_step = find_true_segments(source_arrays["teacher_mask"].astype(bool))[0][0]
    course = sample_curriculum_course(candidate.seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    observations: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    phase_codes: list[int] = []
    terminated = False
    truncated = False
    try:
        observation, info = environment.reset(seed=candidate.seed)
        teacher.reset(environment)
        for index in range(takeover_step):
            teacher.predict(environment, observation, info)
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(source_arrays["executed_actions"][index], dtype=np.float32)
            )
            if terminated or truncated:
                raise RuntimeError(
                    f"摂動回復の前置再生が早期終了した: {candidate.seed}"
                )
        while not (terminated or truncated):
            teacher_action, _ = teacher.predict(environment, observation, info)
            noise = np.clip(
                rng.normal(
                    loc=0.0,
                    scale=noise_standard_deviation,
                    size=environment.action_space.shape,
                ),
                -maximum_absolute_noise,
                maximum_absolute_noise,
            ).astype(np.float32)
            clean_action = np.asarray(teacher_action, dtype=np.float32)
            executed_action = np.clip(
                clean_action + noise,
                environment.action_space.low,
                environment.action_space.high,
            ).astype(np.float32)
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            teacher_actions.append(clean_action.copy())
            executed_actions.append(executed_action.copy())
            noises.append((executed_action - clean_action).copy())
            phase_codes.append(PHASE_CODES[classify_rescue_phase(environment, info)])
            observation, _, terminated, truncated, info = environment.step(
                executed_action
            )
    finally:
        environment.close()
    phase_array = np.asarray(phase_codes, dtype=np.int8)
    phase_counts = {
        phase: int(np.count_nonzero(phase_array == PHASE_CODES[phase]))
        for phase in KEY_TEACHER_PHASES
    }
    success = bool(
        info["course_complete"]
        and not info["hard_fall"]
        and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
    )
    accepted = bool(success and all(value > 0 for value in phase_counts.values()))
    noise_array = np.asarray(noises, dtype=np.float32)
    metadata = {
        "accepted": accepted,
        "seed": candidate.seed,
        "profile": candidate.profile,
        "takeover_step": takeover_step,
        "steps": len(observations),
        "course_complete": bool(info["course_complete"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "phase_step_counts": phase_counts,
        "realized_noise_rms": (
            np.sqrt(np.mean(noise_array.astype(np.float64) ** 2, axis=0)).tolist()
            if len(noise_array)
            else [0.0] * 6
        ),
        "realized_maximum_absolute_noise": (
            float(np.max(np.abs(noise_array))) if len(noise_array) else 0.0
        ),
    }
    if not accepted:
        return None, metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.asarray(observations, dtype=np.float32),
        teacher_actions=np.asarray(teacher_actions, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        action_noise=noise_array,
        phase_codes=phase_array,
    )
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return (
        PhaseBalancedSequence(
            seed=candidate.seed,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(teacher_actions, dtype=np.float32),
            phase_codes=phase_array,
            source_branch_sha256=sha256_file(output_path),
            phase_index_sha256="perturbed_live_phase_labels",
        ),
        {
            **metadata,
            "branch_path": str(output_path.resolve()),
            "branch_sha256": sha256_file(output_path),
        },
    )


def _handoff_balanced_loss(
    step_losses: torch.Tensor,
    phase_codes: torch.Tensor,
    handoff_window_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """引き継ぎ窓と後続三段階を等重みで合成する。"""
    indices = torch.arange(len(step_losses), device=step_losses.device)
    masks = {
        "handoff_window": indices < handoff_window_steps,
        "remaining_pre_hurdle": (indices >= handoff_window_steps)
        & (phase_codes == PHASE_CODES["pre_hurdle"]),
        "hurdle_deformation": phase_codes == PHASE_CODES["hurdle_deformation"],
        "post_clearance_recovery": (
            phase_codes == PHASE_CODES["post_clearance_recovery"]
        ),
    }
    losses: dict[str, torch.Tensor] = {}
    for name, mask in masks.items():
        selected = step_losses[mask]
        if selected.numel() < 1:
            raise ValueError(f"引き継ぎ等重み損失に必須群がない: {name}")
        losses[name] = selected.mean()
    return torch.stack(list(losses.values())).mean(), losses


def train_handoff_balanced_aggregate(
    model: Any,
    sequences: tuple[PhaseBalancedSequence, ...],
    *,
    epochs: int,
    learning_rate: float,
    maximum_gradient_norm: float,
    weight_decay: float,
    handoff_window_steps: int,
    seed: int,
) -> dict[str, object]:
    """引き継ぎ直後を独立群として強調し四回だけ集約更新する。"""
    if epochs != 4:
        raise ValueError("摂動回復の集約更新は四回だけ許可される。")
    torch.manual_seed(seed)
    parameters = actor_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
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
            step_losses: list[torch.Tensor] = []
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
                step_losses.append(torch.mean((prediction - target) ** 2))
            loss, group_losses = _handoff_balanced_loss(
                torch.stack(step_losses),
                torch.as_tensor(
                    sequence.phase_codes,
                    dtype=torch.int64,
                    device=model.device,
                ),
                handoff_window_steps,
            )
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
                    "seed": sequence.seed,
                    "steps": sequence.steps,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "group_losses": {
                        name: float(value.detach().cpu())
                        for name, value in group_losses.items()
                    },
                }
            )
        history.append({"epoch": epoch, "sequences": rows})
    return {
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "sequence_presentations": epochs * len(sequences),
        "history": history,
    }


def run(protocol: PerturbedRecoveryProtocol, output_dir: Path, seed: int) -> dict[str, object]:
    """誤差推定、四回摂動収集、境界更新、二組評価を実行する。"""
    from sb3_contrib import RecurrentPPO

    source_summary = json.loads(
        protocol.source_training_summary_path.read_text(encoding="utf-8")
    )
    source_protocol = load_training_protocol(
        Path(str(source_summary["protocol"]["source_path"]))
    )
    original_sequences, _, manifest = load_phase_balanced_sequences(source_protocol)
    if manifest.sha256 != protocol.demonstration_manifest_sha256:
        raise ValueError("摂動回復の示範目録ハッシュが一致しない。")
    if tuple(candidate.seed for candidate in manifest.candidates) != protocol.collection_seeds:
        raise ValueError("摂動回復の乱数種順が示範目録と一致しない。")
    reset_manifest = load_rescue_reset_manifest(protocol.reset_manifest_path)
    if reset_manifest.sha256 != protocol.reset_manifest_sha256:
        raise ValueError("摂動回復の救援リセット目録ハッシュが一致しない。")
    protected_paths = (
        protocol.source_path,
        protocol.failed_interactive_summary_path,
        protocol.source_checkpoint_path,
        protocol.source_training_summary_path,
        protocol.demonstration_manifest_path,
        protocol.reset_manifest_path,
        RAW_RECOVERY_MODEL.resolve(),
        HALF_RECOVERY_MODEL.resolve(),
        *(candidate.branch_path for candidate in manifest.candidates),
    )
    protected_hashes_before = {
        str(path): sha256_file(path) for path in protected_paths
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    initialization = output_dir / "teacher_init_from_m2_3_2.zip"
    shutil.copy2(protocol.source_checkpoint_path, initialization)
    if sha256_file(initialization) != protocol.source_checkpoint_sha256:
        raise RuntimeError("摂動回復の初期化コピーが出典と一致しない。")
    learner = RecurrentPPO.load(initialization, device="cpu")
    residual_rms = estimate_action_residual_rms(learner, original_sequences)
    noise_standard_deviation = (
        residual_rms * protocol.noise_residual_rms_fraction
    )
    teacher = PortfolioHeight1Teacher()
    rng = np.random.default_rng(seed)
    perturbed_sequences: list[PhaseBalancedSequence] = []
    collection_rows: list[dict[str, object]] = []
    for candidate in manifest.candidates:
        sequence, row = collect_perturbed_teacher_recovery(
            teacher,
            candidate,
            noise_standard_deviation=noise_standard_deviation,
            maximum_absolute_noise=protocol.maximum_absolute_noise,
            rng=rng,
            output_path=(
                output_dir / "branches" / f"seed_{candidate.seed}_perturbed.npz"
            ),
        )
        collection_rows.append(row)
        if sequence is not None:
            perturbed_sequences.append(sequence)
    parameter_hash_before = hash_policy_parameters(learner)
    actor_hash_before, critic_hash_before = _module_hashes(learner)
    update_allowed = len(perturbed_sequences) >= (
        protocol.minimum_accepted_branches_for_update
    )
    if update_allowed:
        training = {
            "executed": True,
            **train_handoff_balanced_aggregate(
                learner,
                (*original_sequences, *perturbed_sequences),
                epochs=protocol.epochs,
                learning_rate=protocol.learning_rate,
                maximum_gradient_norm=protocol.maximum_gradient_norm,
                weight_decay=protocol.weight_decay,
                handoff_window_steps=protocol.handoff_window_steps,
                seed=seed,
            ),
            "original_sequence_count": len(original_sequences),
            "perturbed_sequence_count": len(perturbed_sequences),
        }
    else:
        training = {
            "executed": False,
            "reason": "insufficient_successful_perturbed_recovery_branches",
            "epochs": 0,
            "optimizer_steps": 0,
        }
    parameter_hash_after = hash_policy_parameters(learner)
    actor_hash_after, critic_hash_after = _module_hashes(learner)
    if bool(training["executed"]):
        if actor_hash_after == actor_hash_before:
            raise RuntimeError("摂動回復更新でactorが変更されなかった。")
        if critic_hash_after != critic_hash_before:
            raise RuntimeError("摂動回復更新でcriticが変更された。")
    checkpoint = output_dir / "teacher_after_perturbed_recovery.zip"
    learner.save(checkpoint)
    source_rows = [
        evaluate_closed_loop_handoff(learner, candidate)
        for candidate in manifest.candidates
    ]
    source_evaluation = {
        "episodes": len(source_rows),
        "success_count": sum(bool(row["success"]) for row in source_rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in source_rows),
        "safe_stall_count": sum(bool(row["safe_stall"]) for row in source_rows),
        "raw_clearance_count": sum(
            int(row["raw_clearances"]) > 0 for row in source_rows
        ),
        "recovery_count": sum(
            int(row["recovered_obstacles"]) > 0 for row in source_rows
        ),
        "rows": source_rows,
    }
    prefix_student = RecurrentPPO.load(source_protocol.student_model_path, device="cpu")
    reset_environment = StudentPrefixRescueEnv(
        prefix_student,
        reset_manifest.states,
        max_rescue_steps=800,
    )
    try:
        reset_evaluation = evaluate_rescue_teacher(
            reset_environment,
            learner,
            reset_seeds=tuple(spec.seed for spec in reset_manifest.states),
        )
    finally:
        reset_environment.close()
    gate = _evaluate_gate(source_evaluation, reset_evaluation, protocol.gate)
    protected_hashes_after = {
        str(path): sha256_file(path) for path in protected_paths
    }
    if protected_hashes_after != protected_hashes_before:
        raise RuntimeError("摂動回復中に凍結出典が変更された。")
    result = {
        "method": "m2_3_4_learner_error_scaled_perturbed_teacher_recovery",
        "stage": protocol.stage,
        "split": protocol.split,
        "run_name": output_dir.name,
        "seed": seed,
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.sha256,
        "teacher_training_only": True,
        "failed_interactive_checkpoint_loaded": False,
        "noise_calibration": {
            "learner_action_residual_rms": residual_rms.tolist(),
            "noise_standard_deviation": noise_standard_deviation.tolist(),
            "residual_rms_fraction": protocol.noise_residual_rms_fraction,
            "maximum_absolute_noise": protocol.maximum_absolute_noise,
        },
        "collection": {
            "episode_count": len(collection_rows),
            "accepted_branch_count": len(perturbed_sequences),
            "success_count": sum(bool(row["course_complete"]) for row in collection_rows),
            "hard_fall_count": sum(bool(row["hard_fall"]) for row in collection_rows),
            "rows": collection_rows,
        },
        "training": training,
        "source_state_evaluation": source_evaluation,
        "reset_state_evaluation": reset_evaluation,
        "gate": gate,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_disposition": (
            "m2_4_candidate"
            if bool(gate["eligible_for_m2_4"])
            else "quarantined_m2_3_4_perturbed_result"
        ),
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "actor_hash_before": actor_hash_before,
        "actor_hash_after": actor_hash_after,
        "critic_hash_before": critic_hash_before,
        "critic_hash_after": critic_hash_after,
        "critic_unchanged": critic_hash_before == critic_hash_after,
        "student_weights_updated": False,
        "teacher_weights_updated": bool(training["executed"]),
        "ppo_training_steps": 0,
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
    """凍結規約、出力名、摂動乱数種だけを受け取る。"""
    parser = argparse.ArgumentParser(
        description="学生誤差尺度の摂動教師回復を一回だけ実行する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    """M2.3.4の摂動回復収集、更新、評価を実行する。"""
    args = build_argument_parser().parse_args()
    torch.set_num_threads(1)
    protocol = load_perturbed_recovery_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
