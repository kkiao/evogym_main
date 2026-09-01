"""M2.3.2の段階等重み循環模倣で訓練専用救援教師を初期化する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from general_terrain.audit_rescue_demonstrations import (
    KEY_TEACHER_PHASES,
    PHASE_CODES,
    RescueDemoManifest,
    find_true_segments,
    load_and_validate_branch_arrays,
    load_rescue_demo_manifest,
    sha256_file,
)
from general_terrain.rescue_reset_manifest import (
    RescueResetManifest,
    load_rescue_reset_manifest,
)
from general_terrain.student_prefix_rescue_env import StudentPrefixRescueEnv
from general_terrain.train_prefix_rescue_teacher import (
    copy_teacher_initialization,
    evaluate_rescue_teacher,
    hash_policy_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m2_3_2_phase_balanced_bc_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "phase_balanced_rescue_teacher"


@dataclass(frozen=True)
class PhaseBalancedBCProtocol:
    """M2.3.2で変更を許さない訓練条件と出典を保持する。"""

    version: str
    stage: str
    split: str
    teacher_training_only: bool
    student_model_path: Path
    student_model_sha256: str
    demo_manifest_path: Path
    demo_manifest_sha256: str
    demo_audit_path: Path
    demo_audit_sha256: str
    reset_manifest_path: Path
    reset_manifest_sha256: str
    excluded_checkpoint_run: str
    demonstration_seeds: tuple[int, ...]
    required_phases: tuple[str, ...]
    epochs: int
    learning_rate: float
    maximum_gradient_norm: float
    weight_decay: float
    loss_aggregation: str
    recurrent_state_at_takeover: str
    update_modules: tuple[str, ...]
    ppo_training_steps: int
    validation_episodes: int
    holdout_episodes: int
    gate: dict[str, object]
    source_path: Path
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """JSONへ保存できる辞書形式を返す。"""
        data = asdict(self)
        for name in (
            "student_model_path",
            "demo_manifest_path",
            "demo_audit_path",
            "reset_manifest_path",
            "source_path",
        ):
            data[name] = str(data[name])
        data["demonstration_seeds"] = list(self.demonstration_seeds)
        data["required_phases"] = list(self.required_phases)
        data["update_modules"] = list(self.update_modules)
        return data


@dataclass(frozen=True)
class PhaseBalancedSequence:
    """一回の教師引き継ぎから完走までの連続模倣系列を保持する。"""

    seed: int
    observations: np.ndarray
    actions: np.ndarray
    phase_codes: np.ndarray
    source_branch_sha256: str
    phase_index_sha256: str

    @property
    def steps(self) -> int:
        """系列に含まれる教師動作数を返す。"""
        return int(len(self.observations))


def _resolve_project_file(relative_path: str) -> Path:
    """プロジェクト相対パスを安全な絶対パスへ変換する。"""
    resolved = (PROJECT_ROOT / relative_path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M2.3.2の出典はプロジェクト配下でなければならない。")
    return resolved


def load_training_protocol(
    path: Path = DEFAULT_PROTOCOL,
) -> PhaseBalancedBCProtocol:
    """凍結訓練規約を読み込み、値と全主要ハッシュを検査する。"""
    resolved_path = path.resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M2.3.2訓練規約は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M2.3.2は単一低壁の訓練区分だけを使用できる。")
    protocol = PhaseBalancedBCProtocol(
        version=str(payload["version"]),
        stage=str(payload["stage"]),
        split=str(payload["split"]),
        teacher_training_only=bool(payload["teacher_training_only"]),
        student_model_path=_resolve_project_file(str(payload["student_model_path"])),
        student_model_sha256=str(payload["student_model_sha256"]),
        demo_manifest_path=_resolve_project_file(str(payload["demo_manifest_path"])),
        demo_manifest_sha256=str(payload["demo_manifest_sha256"]),
        demo_audit_path=_resolve_project_file(str(payload["demo_audit_path"])),
        demo_audit_sha256=str(payload["demo_audit_sha256"]),
        reset_manifest_path=_resolve_project_file(str(payload["reset_manifest_path"])),
        reset_manifest_sha256=str(payload["reset_manifest_sha256"]),
        excluded_checkpoint_run=str(payload["excluded_checkpoint_run"]),
        demonstration_seeds=tuple(int(seed) for seed in payload["demonstration_seeds"]),
        required_phases=tuple(str(phase) for phase in payload["required_phases"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        maximum_gradient_norm=float(payload["maximum_gradient_norm"]),
        weight_decay=float(payload["weight_decay"]),
        loss_aggregation=str(payload["loss_aggregation"]),
        recurrent_state_at_takeover=str(payload["recurrent_state_at_takeover"]),
        update_modules=tuple(str(name) for name in payload["update_modules"]),
        ppo_training_steps=int(payload["ppo_training_steps"]),
        validation_episodes=int(payload["validation_episodes"]),
        holdout_episodes=int(payload["holdout_episodes"]),
        gate=dict(payload["gate"]),
        source_path=resolved_path,
        sha256=sha256_file(resolved_path),
    )
    expected_hashes = (
        (protocol.student_model_path, protocol.student_model_sha256),
        (protocol.demo_manifest_path, protocol.demo_manifest_sha256),
        (protocol.demo_audit_path, protocol.demo_audit_sha256),
        (protocol.reset_manifest_path, protocol.reset_manifest_sha256),
    )
    for source_path, expected_hash in expected_hashes:
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"M2.3.2出典ハッシュが一致しない: {source_path}")
    if protocol.epochs != 4:
        raise ValueError("M2.3.2の模倣回数は4でなければならない。")
    if not math.isclose(protocol.learning_rate, 1e-4, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("M2.3.2の学習率は1e-4でなければならない。")
    if not math.isclose(
        protocol.maximum_gradient_norm,
        0.5,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("M2.3.2の勾配上限は0.5でなければならない。")
    if protocol.ppo_training_steps != 0:
        raise ValueError("M2.3.2でPPOを実行することはできない。")
    if protocol.validation_episodes != 0 or protocol.holdout_episodes != 0:
        raise ValueError("M2.3.2は検証または留保区分へアクセスできない。")
    if protocol.required_phases != KEY_TEACHER_PHASES:
        raise ValueError("M2.3.2の等重み段階が凍結値と一致しない。")
    if protocol.recurrent_state_at_takeover != "empty":
        raise ValueError("教師接管時の循環状態は空でなければならない。")
    if protocol.excluded_checkpoint_run != "m2_3_prefix_rescue_teacher_seed7_v1":
        raise ValueError("M2.3失敗チェックポイントの除外指定が不正である。")
    expected_modules = (
        "lstm_actor",
        "mlp_extractor.policy_net",
        "action_net",
    )
    if protocol.update_modules != expected_modules:
        raise ValueError("M2.3.2の更新対象が凍結範囲と一致しない。")
    return protocol


def load_phase_balanced_sequences(
    protocol: PhaseBalancedBCProtocol,
) -> tuple[tuple[PhaseBalancedSequence, ...], dict[str, object], RescueDemoManifest]:
    """監査済み連続教師段と段階索引をハッシュ検証付きで読み込む。"""
    manifest = load_rescue_demo_manifest(protocol.demo_manifest_path)
    if manifest.sha256 != protocol.demo_manifest_sha256:
        raise ValueError("示範目録ハッシュがM2.3.2規約と一致しない。")
    if tuple(candidate.seed for candidate in manifest.candidates) != protocol.demonstration_seeds:
        raise ValueError("示範乱数シード順がM2.3.2規約と一致しない。")
    audit = json.loads(protocol.demo_audit_path.read_text(encoding="utf-8"))
    if not bool(audit["m2_3_1_gate"]["gate_passed"]):
        raise ValueError("M2.3.1データ門が通過していない。")
    if not bool(audit["eligible_for_m2_3_2"]):
        raise ValueError("M2.3.1監査がM2.3.2利用を許可していない。")
    if str(audit["manifest_sha256"]) != manifest.sha256:
        raise ValueError("M2.3.1監査と示範目録のハッシュが一致しない。")
    audit_rows = {int(row["seed"]): row for row in audit["candidates"]}
    sequences: list[PhaseBalancedSequence] = []
    total_phase_counts = {phase: 0 for phase in protocol.required_phases}
    for candidate in manifest.candidates:
        arrays, array_audit = load_and_validate_branch_arrays(candidate)
        segments = find_true_segments(arrays["teacher_mask"])
        if len(segments) != 1:
            raise ValueError(f"教師制御段は一つでなければならない: {candidate.seed}")
        start, end = segments[0]
        audit_row = audit_rows[candidate.seed]
        if audit_row["array_audit"]["teacher_segments"] != [
            {"start": start, "end_exclusive": end, "steps": end - start}
        ]:
            raise ValueError(f"教師制御段がM2.3.1監査と一致しない: {candidate.seed}")
        phase_index_path = Path(str(audit_row["phase_index_path"])).resolve()
        if sha256_file(phase_index_path) != str(audit_row["phase_index_sha256"]):
            raise ValueError(f"段階索引ハッシュが一致しない: {candidate.seed}")
        with np.load(phase_index_path, allow_pickle=False) as archive:
            phase_codes = np.asarray(archive["phase_codes"], dtype=np.int8)
            teacher_indices = np.asarray(archive["teacher_indices"], dtype=np.int32)
            segment_starts = np.asarray(
                archive["teacher_segment_starts"],
                dtype=np.int32,
            )
            segment_ends = np.asarray(
                archive["teacher_segment_ends"],
                dtype=np.int32,
            )
        if not np.array_equal(
            teacher_indices,
            np.flatnonzero(arrays["teacher_mask"]).astype(np.int32),
        ):
            raise ValueError(f"段階索引の教師位置が不正である: {candidate.seed}")
        if not np.array_equal(segment_starts, np.asarray([start], dtype=np.int32)):
            raise ValueError(f"段階索引の開始位置が不正である: {candidate.seed}")
        if not np.array_equal(segment_ends, np.asarray([end], dtype=np.int32)):
            raise ValueError(f"段階索引の終了位置が不正である: {candidate.seed}")
        sequence_phase_codes = phase_codes[start:end]
        for phase in protocol.required_phases:
            count = int(np.count_nonzero(sequence_phase_codes == PHASE_CODES[phase]))
            if count < 1:
                raise ValueError(f"教師系列に必須段階がない: {candidate.seed}, {phase}")
            total_phase_counts[phase] += count
        sequences.append(
            PhaseBalancedSequence(
                seed=candidate.seed,
                observations=np.asarray(
                    arrays["observations"][start:end],
                    dtype=np.float32,
                ),
                actions=np.asarray(
                    arrays["teacher_actions"][start:end],
                    dtype=np.float32,
                ),
                phase_codes=sequence_phase_codes.copy(),
                source_branch_sha256=candidate.branch_sha256,
                phase_index_sha256=str(audit_row["phase_index_sha256"]),
            )
        )
        if int(array_audit["teacher_control_steps"]) != end - start:
            raise ValueError(f"教師系列長が配列監査と一致しない: {candidate.seed}")
    audited_phase_counts = audit["m2_3_1_gate"]["teacher_phase_step_counts"]
    if any(
        total_phase_counts[phase] != int(audited_phase_counts[phase])
        for phase in protocol.required_phases
    ):
        raise ValueError("教師段階総数がM2.3.1監査と一致しない。")
    metadata = {
        "sequence_count": len(sequences),
        "total_teacher_steps_per_epoch": sum(sequence.steps for sequence in sequences),
        "phase_step_counts": total_phase_counts,
        "sequences": [
            {
                "seed": sequence.seed,
                "steps": sequence.steps,
                "phase_step_counts": {
                    phase: int(
                        np.count_nonzero(
                            sequence.phase_codes == PHASE_CODES[phase]
                        )
                    )
                    for phase in protocol.required_phases
                },
                "source_branch_sha256": sequence.source_branch_sha256,
                "phase_index_sha256": sequence.phase_index_sha256,
            }
            for sequence in sequences
        ],
    }
    return tuple(sequences), metadata, manifest


def equal_phase_mean_loss(
    step_losses: torch.Tensor,
    phase_codes: torch.Tensor,
    *,
    required_phase_codes: tuple[int, ...],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """各段階内で平均してから段階間を等重みで合成する。"""
    if step_losses.ndim != 1 or phase_codes.ndim != 1:
        raise ValueError("歩ごとの損失と段階コードは一次元でなければならない。")
    if len(step_losses) != len(phase_codes):
        raise ValueError("歩ごとの損失数と段階コード数が一致しない。")
    phase_losses: dict[int, torch.Tensor] = {}
    for code in required_phase_codes:
        selected = step_losses[phase_codes == code]
        if selected.numel() < 1:
            raise ValueError(f"等重み損失に必須段階がない: {code}")
        phase_losses[code] = selected.mean()
    return torch.stack(list(phase_losses.values())).mean(), phase_losses


def actor_trainable_parameters(model: Any) -> tuple[torch.nn.Parameter, ...]:
    """凍結規約で許可されたactor三部分のパラメータだけを返す。"""
    parameters: list[torch.nn.Parameter] = []
    parameters.extend(model.policy.lstm_actor.parameters())
    parameters.extend(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("actor更新対象に重複パラメータがある。")
    return tuple(parameters)


def hash_modules(modules: Iterable[tuple[str, torch.nn.Module]]) -> str:
    """指定した複数モジュールの状態から安定したハッシュを作る。"""
    digest = hashlib.sha256()
    for module_name, module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(module_name.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _actor_module_hash(model: Any) -> str:
    """更新を許可されたactor部分のハッシュを返す。"""
    return hash_modules(
        (
            ("lstm_actor", model.policy.lstm_actor),
            ("policy_net", model.policy.mlp_extractor.policy_net),
            ("action_net", model.policy.action_net),
        )
    )


def _critic_module_hash(model: Any) -> str:
    """更新禁止のcritic部分のハッシュを返す。"""
    return hash_modules(
        (
            ("lstm_critic", model.policy.lstm_critic),
            ("value_net_body", model.policy.mlp_extractor.value_net),
            ("value_net_head", model.policy.value_net),
        )
    )


def train_phase_balanced_clone(
    model: Any,
    sequences: tuple[PhaseBalancedSequence, ...],
    *,
    epochs: int,
    learning_rate: float,
    maximum_gradient_norm: float,
    weight_decay: float,
    seed: int,
) -> dict[str, object]:
    """系列順を保ち、各系列内の三段階を等重みで模倣更新する。"""
    if epochs != 4:
        raise ValueError("M2.3.2では4回以外の模倣更新を実行できない。")
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
        sequence_rows: list[dict[str, object]] = []
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
            stacked_losses = torch.stack(step_losses)
            phase_tensor = torch.as_tensor(
                sequence.phase_codes,
                dtype=torch.int64,
                device=model.device,
            )
            sequence_loss, phase_losses = equal_phase_mean_loss(
                stacked_losses,
                phase_tensor,
                required_phase_codes=required_codes,
            )
            optimizer.zero_grad()
            sequence_loss.backward()
            gradient_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=maximum_gradient_norm,
            )
            optimizer.step()
            optimizer_steps += 1
            sequence_rows.append(
                {
                    "seed": sequence.seed,
                    "steps": sequence.steps,
                    "equal_phase_loss": float(sequence_loss.detach().cpu()),
                    "phase_losses": {
                        phase: float(
                            phase_losses[PHASE_CODES[phase]].detach().cpu()
                        )
                        for phase in KEY_TEACHER_PHASES
                    },
                    "gradient_norm_before_clip": float(
                        gradient_norm_before_clip.detach().cpu()
                    ),
                }
            )
        history.append(
            {
                "epoch": epoch,
                "mean_equal_phase_loss": float(
                    np.mean([row["equal_phase_loss"] for row in sequence_rows])
                ),
                "mean_phase_losses": {
                    phase: float(
                        np.mean(
                            [row["phase_losses"][phase] for row in sequence_rows]
                        )
                    )
                    for phase in KEY_TEACHER_PHASES
                },
                "maximum_gradient_norm_before_clip": float(
                    max(row["gradient_norm_before_clip"] for row in sequence_rows)
                ),
                "sequences": sequence_rows,
            }
        )
    model.policy.set_training_mode(False)
    return {
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "supervised_action_presentations": sum(
            sequence.steps for sequence in sequences
        )
        * epochs,
        "history": history,
    }


def evaluate_m2_3_2_gate(
    evaluation: Mapping[str, object],
    gate_config: Mapping[str, object],
) -> dict[str, object]:
    """完走、転倒、停滞、回復段階訪問からM2.3.2門を判定する。"""
    success_count = int(evaluation["success_count"])
    hard_fall_count = int(evaluation["hard_fall_count"])
    safe_stall_count = int(evaluation["safe_stall_count"])
    recovery_phase_steps = int(
        evaluation["phase_step_counts"]["post_clearance_recovery"]
    )
    success_passed = success_count >= int(gate_config["minimum_success_count"])
    hard_fall_passed = hard_fall_count <= int(
        gate_config["maximum_hard_fall_count"]
    )
    stall_passed = safe_stall_count < int(
        gate_config["maximum_safe_stall_count_exclusive"]
    )
    recovery_phase_passed = bool(
        recovery_phase_steps > 0
        if gate_config["require_post_clearance_recovery_visitation"]
        else True
    )
    gate_passed = bool(
        success_passed
        and hard_fall_passed
        and stall_passed
        and recovery_phase_passed
    )
    return {
        "gate_name": "m2_3_2_phase_balanced_bc_gate_v1",
        "minimum_success_count": int(gate_config["minimum_success_count"]),
        "maximum_hard_fall_count": int(gate_config["maximum_hard_fall_count"]),
        "maximum_safe_stall_count_exclusive": int(
            gate_config["maximum_safe_stall_count_exclusive"]
        ),
        "require_post_clearance_recovery_visitation": bool(
            gate_config["require_post_clearance_recovery_visitation"]
        ),
        "success_requirement_passed": success_passed,
        "hard_fall_requirement_passed": hard_fall_passed,
        "stall_requirement_passed": stall_passed,
        "recovery_phase_requirement_passed": recovery_phase_passed,
        "gate_passed": gate_passed,
        "eligible_for_m2_4": gate_passed,
    }


def _make_evaluation_environment(
    prefix_student: Any,
    reset_manifest: RescueResetManifest,
) -> StudentPrefixRescueEnv:
    """M2.2と同じ11状態の教師専用評価環境を作る。"""
    return StudentPrefixRescueEnv(
        prefix_student,
        reset_manifest.states,
        stage=reset_manifest.stage,
        max_rescue_steps=800,
    )


def _protected_source_paths(
    protocol: PhaseBalancedBCProtocol,
    manifest: RescueDemoManifest,
    demo_audit: Mapping[str, object],
) -> tuple[Path, ...]:
    """訓練前後に不変ハッシュを要求する全出典パスを返す。"""
    paths = {
        protocol.source_path,
        protocol.student_model_path,
        protocol.demo_manifest_path,
        protocol.demo_audit_path,
        protocol.reset_manifest_path,
    }
    for candidate in manifest.candidates:
        paths.update(
            {
                candidate.summary_path,
                candidate.branch_path,
                candidate.sidecar_path,
            }
        )
    for row in demo_audit["candidates"]:
        paths.add(Path(str(row["phase_index_path"])).resolve())
    return tuple(sorted(paths))


def build_argument_parser() -> argparse.ArgumentParser:
    """凍結規約と出力名だけを受け取るM2.3.2引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="監査済み救援系列で段階等重み循環模倣を実行する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """新しい教師コピーを固定4回だけ模倣更新し11状態で評価する。"""
    from sb3_contrib import RecurrentPPO

    args = build_argument_parser().parse_args()
    torch.set_num_threads(1)
    protocol = load_training_protocol(Path(args.protocol))
    sequences, dataset_metadata, demo_manifest = load_phase_balanced_sequences(
        protocol
    )
    demo_audit = json.loads(protocol.demo_audit_path.read_text(encoding="utf-8"))
    reset_manifest = load_rescue_reset_manifest(protocol.reset_manifest_path)
    if reset_manifest.sha256 != protocol.reset_manifest_sha256:
        raise ValueError("M2.2救援リセット目録が凍結規約と一致しない。")
    protected_paths = _protected_source_paths(protocol, demo_manifest, demo_audit)
    protected_hashes_before = {
        str(path): sha256_file(path) for path in protected_paths
    }

    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    teacher_initialization_path = output_dir / "teacher_init_from_student.zip"
    teacher_initialization_sha256 = copy_teacher_initialization(
        protocol.student_model_path,
        teacher_initialization_path,
    )
    prefix_student = RecurrentPPO.load(protocol.student_model_path, device="cpu")
    teacher = RecurrentPPO.load(teacher_initialization_path, device="cpu")
    reset_seeds = tuple(spec.seed for spec in reset_manifest.states)

    initial_environment = _make_evaluation_environment(prefix_student, reset_manifest)
    initial_start = time.perf_counter()
    try:
        initial_evaluation = evaluate_rescue_teacher(
            initial_environment,
            teacher,
            reset_seeds=reset_seeds,
        )
    finally:
        initial_environment.close()
    initial_seconds = time.perf_counter() - initial_start

    full_parameter_hash_before = hash_policy_parameters(teacher)
    actor_hash_before = _actor_module_hash(teacher)
    critic_hash_before = _critic_module_hash(teacher)
    training_start = time.perf_counter()
    training_result = train_phase_balanced_clone(
        teacher,
        sequences,
        epochs=protocol.epochs,
        learning_rate=protocol.learning_rate,
        maximum_gradient_norm=protocol.maximum_gradient_norm,
        weight_decay=protocol.weight_decay,
        seed=args.seed,
    )
    training_seconds = time.perf_counter() - training_start
    full_parameter_hash_after = hash_policy_parameters(teacher)
    actor_hash_after = _actor_module_hash(teacher)
    critic_hash_after = _critic_module_hash(teacher)
    if actor_hash_after == actor_hash_before:
        raise RuntimeError("M2.3.2でactorパラメータが更新されなかった。")
    if critic_hash_after != critic_hash_before:
        raise RuntimeError("M2.3.2で更新禁止のcriticパラメータが変更された。")
    checkpoint_path = output_dir / "teacher_after_phase_balanced_bc.zip"
    teacher.save(checkpoint_path)

    final_environment = _make_evaluation_environment(prefix_student, reset_manifest)
    final_start = time.perf_counter()
    try:
        final_evaluation = evaluate_rescue_teacher(
            final_environment,
            teacher,
            reset_seeds=reset_seeds,
        )
    finally:
        final_environment.close()
    final_seconds = time.perf_counter() - final_start
    gate = evaluate_m2_3_2_gate(final_evaluation, protocol.gate)

    protected_hashes_after = {
        str(path): sha256_file(path) for path in protected_paths
    }
    if protected_hashes_after != protected_hashes_before:
        raise RuntimeError("M2.3.2中に凍結出典ファイルが変更された。")
    summary = {
        "method": "m2_3_2_phase_balanced_recurrent_behavior_cloning",
        "stage": protocol.stage,
        "split": protocol.split,
        "run_name": args.run_name,
        "seed": args.seed,
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.sha256,
        "teacher_training_only": True,
        "student_model": str(protocol.student_model_path),
        "student_model_sha256": protocol.student_model_sha256,
        "student_model_unchanged": True,
        "teacher_initialization": str(teacher_initialization_path.resolve()),
        "teacher_initialization_sha256": teacher_initialization_sha256,
        "m2_3_failed_checkpoint_loaded": False,
        "dataset": dataset_metadata,
        "behavior_cloning": training_result,
        "ppo_training_steps": 0,
        "student_weights_updated": False,
        "teacher_weights_updated": True,
        "full_parameter_sha256_before": full_parameter_hash_before,
        "full_parameter_sha256_after": full_parameter_hash_after,
        "actor_sha256_before": actor_hash_before,
        "actor_sha256_after": actor_hash_after,
        "actor_parameters_changed": actor_hash_after != actor_hash_before,
        "critic_sha256_before": critic_hash_before,
        "critic_sha256_after": critic_hash_after,
        "critic_parameters_unchanged": critic_hash_after == critic_hash_before,
        "teacher_checkpoint": str(checkpoint_path.resolve()),
        "teacher_checkpoint_sha256": sha256_file(checkpoint_path),
        "initialization_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "m2_3_2_gate": gate,
        "checkpoint_disposition": (
            "m2_4_candidate"
            if bool(gate["eligible_for_m2_4"])
            else "quarantined_failed_m2_3_2"
        ),
        "eligible_for_m2_4": bool(gate["eligible_for_m2_4"]),
        "eligible_for_student_initialization": False,
        "eligible_for_final_student_test": False,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "protected_source_files_unchanged": True,
        "timing_seconds": {
            "initial_evaluation": initial_seconds,
            "behavior_cloning": training_seconds,
            "final_evaluation": final_seconds,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
