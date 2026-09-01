"""M2.3.4の学生誘導状態を用いた対話的救援訂正を一回だけ実行する。"""

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
from general_terrain.interactive_rescue import (
    InteractiveRescueController,
    local_terrain_is_visible,
)
from general_terrain.portfolio_height1_teacher import (
    HALF_RECOVERY_MODEL,
    RAW_RECOVERY_MODEL,
    PortfolioHeight1Teacher,
)
from general_terrain.rescue_profiles import get_rescue_profile
from general_terrain.rescue_reset_manifest import load_rescue_reset_manifest
from general_terrain.student_prefix_rescue_env import (
    RESCUE_PHASES,
    StudentPrefixRescueEnv,
    classify_rescue_phase,
)
from general_terrain.train_phase_balanced_rescue_teacher import (
    PhaseBalancedSequence,
    actor_trainable_parameters,
    equal_phase_mean_loss,
    hash_modules,
    load_phase_balanced_sequences,
    load_training_protocol,
)
from general_terrain.train_prefix_rescue_teacher import (
    evaluate_rescue_teacher,
    hash_policy_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m2_3_4_interactive_correction_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "interactive_correction_rescue_teacher"


@dataclass(frozen=True)
class InteractiveCorrectionProtocol:
    """一回限りの収集、更新、評価境界を保持する。"""

    version: str
    stage: str
    split: str
    teacher_training_only: bool
    source_diagnostic_path: Path
    source_diagnostic_sha256: str
    source_training_summary_path: Path
    source_training_summary_sha256: str
    source_checkpoint_path: Path
    source_checkpoint_sha256: str
    demonstration_manifest_path: Path
    demonstration_manifest_sha256: str
    reset_manifest_path: Path
    reset_manifest_sha256: str
    collection_seeds: tuple[int, ...]
    maximum_collection_episodes: int
    collection_episodes_per_seed: int
    aggregate_original_demonstrations: bool
    epochs: int
    learning_rate: float
    maximum_gradient_norm: float
    weight_decay: float
    update_modules: tuple[str, ...]
    ppo_training_steps: int
    validation_episodes: int
    holdout_episodes: int
    gate: dict[str, int]
    source_path: Path
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存できる辞書へ変換する。"""
        result = asdict(self)
        for name in (
            "source_diagnostic_path",
            "source_training_summary_path",
            "source_checkpoint_path",
            "demonstration_manifest_path",
            "reset_manifest_path",
            "source_path",
        ):
            result[name] = str(result[name])
        result["collection_seeds"] = list(self.collection_seeds)
        result["update_modules"] = list(self.update_modules)
        return result


def _resolve_project_path(value: str) -> Path:
    """プロジェクト相対値を範囲検査済み絶対パスへ変換する。"""
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M2.3.4の出典はプロジェクト配下でなければならない。")
    return path


def load_interactive_correction_protocol(
    path: Path = DEFAULT_PROTOCOL,
) -> InteractiveCorrectionProtocol:
    """凍結規約を読み込み、出典ハッシュと実験上限を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M2.3.4規約は凍結済みでなければならない。")
    protocol = InteractiveCorrectionProtocol(
        version=str(payload["version"]),
        stage=str(payload["stage"]),
        split=str(payload["split"]),
        teacher_training_only=bool(payload["teacher_training_only"]),
        source_diagnostic_path=_resolve_project_path(
            str(payload["source_diagnostic_path"])
        ),
        source_diagnostic_sha256=str(payload["source_diagnostic_sha256"]),
        source_training_summary_path=_resolve_project_path(
            str(payload["source_training_summary_path"])
        ),
        source_training_summary_sha256=str(
            payload["source_training_summary_sha256"]
        ),
        source_checkpoint_path=_resolve_project_path(
            str(payload["source_checkpoint_path"])
        ),
        source_checkpoint_sha256=str(payload["source_checkpoint_sha256"]),
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
        collection_episodes_per_seed=int(payload["collection_episodes_per_seed"]),
        aggregate_original_demonstrations=bool(
            payload["aggregate_original_demonstrations"]
        ),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        maximum_gradient_norm=float(payload["maximum_gradient_norm"]),
        weight_decay=float(payload["weight_decay"]),
        update_modules=tuple(str(name) for name in payload["update_modules"]),
        ppo_training_steps=int(payload["ppo_training_steps"]),
        validation_episodes=int(payload["validation_episodes"]),
        holdout_episodes=int(payload["holdout_episodes"]),
        gate={str(name): int(value) for name, value in payload["gate"].items()},
        source_path=source_path,
        sha256=sha256_file(source_path),
    )
    expected_hashes = (
        (protocol.source_diagnostic_path, protocol.source_diagnostic_sha256),
        (
            protocol.source_training_summary_path,
            protocol.source_training_summary_sha256,
        ),
        (protocol.source_checkpoint_path, protocol.source_checkpoint_sha256),
        (
            protocol.demonstration_manifest_path,
            protocol.demonstration_manifest_sha256,
        ),
        (protocol.reset_manifest_path, protocol.reset_manifest_sha256),
    )
    for source, expected in expected_hashes:
        if sha256_file(source) != expected:
            raise ValueError(f"M2.3.4出典ハッシュが一致しない: {source}")
    diagnostic = json.loads(
        protocol.source_diagnostic_path.read_text(encoding="utf-8")
    )
    if diagnostic["decision_gate"]["selected_next_route"] != (
        "interactive_closed_loop_correction_aggregation"
    ):
        raise ValueError("M2.3.3は対話的訂正経路を選択していない。")
    if protocol.stage != "hurdle_single" or protocol.split != "train":
        raise ValueError("M2.3.4は訓練用単一低壁だけを使用できる。")
    if protocol.maximum_collection_episodes != 4:
        raise ValueError("M2.3.4の収集回数上限は4回でなければならない。")
    if protocol.collection_episodes_per_seed != 1:
        raise ValueError("各乱数種の収集は一回だけでなければならない。")
    if len(protocol.collection_seeds) != 4:
        raise ValueError("M2.3.4の収集乱数種は4個でなければならない。")
    if protocol.epochs != 2 or protocol.learning_rate != 5e-5:
        raise ValueError("M2.3.4の更新回数または学習率が凍結値と異なる。")
    if protocol.ppo_training_steps != 0:
        raise ValueError("M2.3.4でPPOを実行することはできない。")
    if protocol.validation_episodes != 0 or protocol.holdout_episodes != 0:
        raise ValueError("M2.3.4は検証または留保区分へアクセスできない。")
    return protocol


def _phase_counts(codes: np.ndarray) -> dict[str, int]:
    """段階コード列を四段階の歩数へ集計する。"""
    return {
        phase: int(np.count_nonzero(codes == PHASE_CODES[phase]))
        for phase in RESCUE_PHASES
    }


def _save_correction_branch(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> None:
    """成功した対話的訂正系列だけを配列と側車情報へ保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(dict(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_interactive_correction(
    learner: Any,
    teacher: PortfolioHeight1Teacher,
    candidate: RescueDemoCandidate,
    *,
    output_path: Path,
) -> tuple[PhaseBalancedSequence | None, dict[str, object]]:
    """元引き継ぎ状態から学習器を動かし危険時だけ教師へ連続移譲する。"""
    source_arrays, _ = load_and_validate_branch_arrays(candidate)
    takeover_step = find_true_segments(source_arrays["teacher_mask"].astype(bool))[0][0]
    course = sample_curriculum_course(candidate.seed, "hurdle_single", "train")
    environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
    rescue_config = get_rescue_profile(candidate.profile)
    rescue = InteractiveRescueController(rescue_config)
    observations: list[np.ndarray] = []
    learner_actions: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    teacher_mask: list[bool] = []
    phase_codes: list[int] = []
    events: list[dict[str, object]] = []
    recurrent_state: Any = None
    episode_start = np.ones((1,), dtype=bool)
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
                    f"対話的訂正の前置再生が早期終了した: {candidate.seed}"
                )
        schema = tuple(environment.unwrapped.schema)
        while not (terminated or truncated):
            learner_action, recurrent_state = learner.predict(
                observation,
                state=recurrent_state,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start[:] = False
            teacher_action, teacher_stage = teacher.predict(
                environment,
                observation,
                info,
            )
            decision = rescue.decide(
                info,
                np.asarray(learner_action, dtype=np.float32),
                np.asarray(teacher_action, dtype=np.float32),
                local_terrain_visible=local_terrain_is_visible(
                    observation,
                    schema,
                    maximum_rise_offset=(
                        rescue_config.disagreement_maximum_rise_offset
                    ),
                ),
            )
            executed_action = (
                teacher_action if decision.use_teacher else learner_action
            )
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            learner_actions.append(
                np.asarray(learner_action, dtype=np.float32).copy()
            )
            teacher_actions.append(
                np.asarray(teacher_action, dtype=np.float32).copy()
            )
            executed_actions.append(
                np.asarray(executed_action, dtype=np.float32).copy()
            )
            teacher_mask.append(decision.use_teacher)
            phase_codes.append(PHASE_CODES[classify_rescue_phase(environment, info)])
            if decision.event in {"start", "release"}:
                events.append(
                    {
                        "relative_step": len(observations) - 1,
                        "event": decision.event,
                        "reason": decision.reason,
                        "rescue_id": decision.rescue_id,
                        "teacher_stage": teacher_stage,
                    }
                )
            observation, _, terminated, truncated, info = environment.step(
                np.asarray(executed_action, dtype=np.float32)
            )
    finally:
        environment.close()
    phase_array = np.asarray(phase_codes, dtype=np.int8)
    counts = _phase_counts(phase_array)
    success = bool(
        info["course_complete"]
        and not info["hard_fall"]
        and int(info["recovered_obstacles"]) >= int(info["obstacle_count"])
    )
    required_phase_coverage = all(counts[phase] > 0 for phase in KEY_TEACHER_PHASES)
    accepted = bool(success and any(teacher_mask) and required_phase_coverage)
    arrays = {
        "observations": np.asarray(observations, dtype=np.float32),
        "learner_actions": np.asarray(learner_actions, dtype=np.float32),
        "teacher_actions": np.asarray(teacher_actions, dtype=np.float32),
        "executed_actions": np.asarray(executed_actions, dtype=np.float32),
        "teacher_mask": np.asarray(teacher_mask, dtype=bool),
        "phase_codes": phase_array,
    }
    metadata = {
        "accepted": accepted,
        "seed": candidate.seed,
        "profile": candidate.profile,
        "takeover_step": takeover_step,
        "steps": len(observations),
        "teacher_control_steps": int(np.count_nonzero(teacher_mask)),
        "learner_control_steps": int(len(observations) - np.count_nonzero(teacher_mask)),
        "course_complete": bool(info["course_complete"]),
        "hard_fall": bool(info["hard_fall"]),
        "failure_reason": str(info["failure_reason"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "phase_step_counts": counts,
        "required_phase_coverage": required_phase_coverage,
        "events": events,
    }
    if not accepted:
        return None, metadata
    _save_correction_branch(output_path, arrays, metadata)
    return (
        PhaseBalancedSequence(
            seed=candidate.seed,
            observations=arrays["observations"],
            actions=arrays["teacher_actions"],
            phase_codes=arrays["phase_codes"],
            source_branch_sha256=sha256_file(output_path),
            phase_index_sha256="interactive_live_phase_labels",
        ),
        {
            **metadata,
            "branch_path": str(output_path.resolve()),
            "branch_sha256": sha256_file(output_path),
        },
    )


def _module_hashes(model: Any) -> tuple[str, str]:
    """更新許可actorと更新禁止criticのハッシュを返す。"""
    actor_hash = hash_modules(
        (
            ("lstm_actor", model.policy.lstm_actor),
            ("policy_net", model.policy.mlp_extractor.policy_net),
            ("action_net", model.policy.action_net),
        )
    )
    critic_hash = hash_modules(
        (
            ("lstm_critic", model.policy.lstm_critic),
            ("value_net_body", model.policy.mlp_extractor.value_net),
            ("value_net_head", model.policy.value_net),
        )
    )
    return actor_hash, critic_hash


def train_correction_aggregate(
    model: Any,
    sequences: tuple[PhaseBalancedSequence, ...],
    *,
    epochs: int,
    learning_rate: float,
    maximum_gradient_norm: float,
    weight_decay: float,
    seed: int,
) -> dict[str, object]:
    """元示範と学生誘導訂正を系列単位で等しく二回だけ更新する。"""
    if epochs != 2:
        raise ValueError("対話的訂正更新は二回だけ許可される。")
    torch.manual_seed(seed)
    parameters = actor_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    required_codes = tuple(PHASE_CODES[phase] for phase in KEY_TEACHER_PHASES)
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
            loss, phase_losses = equal_phase_mean_loss(
                torch.stack(losses),
                torch.as_tensor(
                    sequence.phase_codes,
                    dtype=torch.int64,
                    device=model.device,
                ),
                required_phase_codes=required_codes,
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
                    "equal_phase_loss": float(loss.detach().cpu()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "phase_losses": {
                        phase: float(phase_losses[PHASE_CODES[phase]].detach().cpu())
                        for phase in KEY_TEACHER_PHASES
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


def _evaluate_gate(
    source_states: Mapping[str, object],
    reset_states: Mapping[str, object],
    gate: Mapping[str, int],
) -> dict[str, object]:
    """出典四状態と凍結十一状態の大段階出口を同時に判定する。"""
    checks = {
        "source_success": int(source_states["success_count"])
        >= gate["minimum_source_state_success_count"],
        "source_hard_fall": int(source_states["hard_fall_count"])
        <= gate["maximum_source_state_hard_fall_count"],
        "reset_success": int(reset_states["success_count"])
        >= gate["minimum_reset_state_success_count"],
        "reset_hard_fall": int(reset_states["hard_fall_count"])
        <= gate["maximum_reset_state_hard_fall_count"],
    }
    return {
        "requirements": dict(gate),
        "checks": checks,
        "gate_passed": all(checks.values()),
        "eligible_for_m2_4": all(checks.values()),
    }


def run(protocol: InteractiveCorrectionProtocol, output_dir: Path, seed: int) -> dict[str, object]:
    """四回収集、一回集約更新、二種類の訓練状態評価を実行する。"""
    from sb3_contrib import RecurrentPPO

    manifest = load_rescue_demo_manifest(protocol.demonstration_manifest_path)
    if manifest.sha256 != protocol.demonstration_manifest_sha256:
        raise ValueError("M2.3.4示範目録ハッシュが一致しない。")
    if tuple(candidate.seed for candidate in manifest.candidates) != protocol.collection_seeds:
        raise ValueError("M2.3.4収集乱数種順が示範目録と一致しない。")
    m2_3_2_summary = json.loads(
        protocol.source_training_summary_path.read_text(encoding="utf-8")
    )
    source_training_protocol = load_training_protocol(
        Path(str(m2_3_2_summary["protocol"]["source_path"]))
    )
    original_sequences, _, _ = load_phase_balanced_sequences(
        source_training_protocol
    )
    reset_manifest = load_rescue_reset_manifest(protocol.reset_manifest_path)
    if reset_manifest.sha256 != protocol.reset_manifest_sha256:
        raise ValueError("M2.3.4救援重置目録ハッシュが一致しない。")
    protected_paths = (
        protocol.source_path,
        protocol.source_diagnostic_path,
        protocol.source_training_summary_path,
        protocol.source_checkpoint_path,
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
    checkpoint_initialization = output_dir / "teacher_init_from_m2_3_2.zip"
    shutil.copy2(protocol.source_checkpoint_path, checkpoint_initialization)
    if sha256_file(checkpoint_initialization) != protocol.source_checkpoint_sha256:
        raise RuntimeError("M2.3.4初期化コピーのハッシュが一致しない。")
    learner = RecurrentPPO.load(checkpoint_initialization, device="cpu")
    teacher = PortfolioHeight1Teacher()
    parameter_hash_before = hash_policy_parameters(learner)
    actor_hash_before, critic_hash_before = _module_hashes(learner)
    correction_sequences: list[PhaseBalancedSequence] = []
    collection_rows: list[dict[str, object]] = []
    branches_dir = output_dir / "branches"
    for candidate in manifest.candidates:
        sequence, row = collect_interactive_correction(
            learner,
            teacher,
            candidate,
            output_path=branches_dir / f"seed_{candidate.seed}_correction.npz",
        )
        collection_rows.append(row)
        if sequence is not None:
            correction_sequences.append(sequence)
    if not correction_sequences:
        training_result: dict[str, object] = {
            "executed": False,
            "reason": "no_successful_interactive_correction_branch",
            "epochs": 0,
            "optimizer_steps": 0,
        }
    else:
        aggregate = (
            (*original_sequences, *correction_sequences)
            if protocol.aggregate_original_demonstrations
            else tuple(correction_sequences)
        )
        training_result = {
            "executed": True,
            **train_correction_aggregate(
                learner,
                tuple(aggregate),
                epochs=protocol.epochs,
                learning_rate=protocol.learning_rate,
                maximum_gradient_norm=protocol.maximum_gradient_norm,
                weight_decay=protocol.weight_decay,
                seed=seed,
            ),
            "original_sequence_count": len(original_sequences),
            "correction_sequence_count": len(correction_sequences),
        }
    parameter_hash_after = hash_policy_parameters(learner)
    actor_hash_after, critic_hash_after = _module_hashes(learner)
    if bool(training_result["executed"]):
        if actor_hash_after == actor_hash_before:
            raise RuntimeError("M2.3.4でactorが更新されなかった。")
        if critic_hash_after != critic_hash_before:
            raise RuntimeError("M2.3.4でcriticが変更された。")
    checkpoint = output_dir / "teacher_after_interactive_correction.zip"
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
    prefix_student = RecurrentPPO.load(
        source_training_protocol.student_model_path,
        device="cpu",
    )
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
    interactive_improved = bool(
        int(source_evaluation["success_count"]) > 0
        and int(source_evaluation["hard_fall_count"]) <= 0
        and int(reset_evaluation["hard_fall_count"])
        <= int(m2_3_2_summary["final_evaluation"]["hard_fall_count"])
    )
    protected_hashes_after = {
        str(path): sha256_file(path) for path in protected_paths
    }
    if protected_hashes_after != protected_hashes_before:
        raise RuntimeError("M2.3.4中に凍結出典が変更された。")
    result = {
        "method": "m2_3_4_interactive_closed_loop_correction_aggregation",
        "stage": protocol.stage,
        "split": protocol.split,
        "run_name": output_dir.name,
        "seed": seed,
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.sha256,
        "teacher_training_only": True,
        "collection": {
            "episode_count": len(collection_rows),
            "accepted_branch_count": len(correction_sequences),
            "success_count": sum(bool(row["course_complete"]) for row in collection_rows),
            "hard_fall_count": sum(bool(row["hard_fall"]) for row in collection_rows),
            "teacher_control_steps": sum(
                int(row["teacher_control_steps"]) for row in collection_rows
            ),
            "learner_control_steps": sum(
                int(row["learner_control_steps"]) for row in collection_rows
            ),
            "rows": collection_rows,
        },
        "training": training_result,
        "source_state_evaluation": source_evaluation,
        "reset_state_evaluation": reset_evaluation,
        "gate": gate,
        "interactive_route_has_net_improvement": interactive_improved,
        "fallback_route": (
            None
            if interactive_improved
            else "learner_error_scaled_perturbed_teacher_recovery"
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_disposition": (
            "m2_4_candidate"
            if bool(gate["eligible_for_m2_4"])
            else "quarantined_m2_3_4_interactive_result"
        ),
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "actor_hash_before": actor_hash_before,
        "actor_hash_after": actor_hash_after,
        "critic_hash_before": critic_hash_before,
        "critic_hash_after": critic_hash_after,
        "critic_unchanged": critic_hash_before == critic_hash_after,
        "student_weights_updated": False,
        "teacher_weights_updated": bool(training_result["executed"]),
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
    """凍結規約、出力名、再現乱数種だけを受け取る。"""
    parser = argparse.ArgumentParser(
        description="学生誘導状態の対話的救援訂正を一回だけ集約する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    """M2.3.4の有界収集、更新、評価を実行する。"""
    args = build_argument_parser().parse_args()
    torch.set_num_threads(1)
    protocol = load_interactive_correction_protocol(Path(args.protocol))
    result = run(protocol, RUNS_ROOT / args.run_name, args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
