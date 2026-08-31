"""M2.4の成功分岐だけでM3循環学生を分層模倣初期化する。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from general_terrain.audit_rescue_demonstrations import find_true_segments, sha256_file
from general_terrain.curriculum import get_curriculum_stage, sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.interactive_rescue import local_terrain_is_visible
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_only_evaluation import evaluate_student_batch
from general_terrain.student_prefix_rescue_env import (
    HURDLE_DEFORMATION_PHASE,
    POST_CLEARANCE_RECOVERY_PHASE,
    POST_RECOVERY_STALL_PHASE,
    PRE_HURDLE_PHASE,
    classify_rescue_phase,
)
from general_terrain.train_prefix_rescue_teacher import hash_policy_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT
    / "config"
    / "m3_stratified_student_initialization_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "m3_stratified_student"
STRATUM_NAMES = (
    "student_prefix_flat",
    "student_prefix_obstacle",
    "teacher_rescue_window",
    "teacher_recovery",
)
STRATUM_CODES = {name: index for index, name in enumerate(STRATUM_NAMES)}
REQUIRED_ARRAYS = {
    "observations",
    "student_actions",
    "teacher_actions",
    "executed_actions",
    "teacher_mask",
    "rescue_ids",
    "teacher_stages",
}


@dataclass(frozen=True)
class M3Protocol:
    """M3で変更を許さない出典、模倣範囲、評価隔離を保持する。"""

    version: str
    stage: str
    split: str
    source_student_model_path: Path
    source_student_model_sha256: str
    accepted_branch_manifest_path: Path
    accepted_branch_manifest_sha256: str
    m2_4_summary_path: Path
    m2_4_summary_sha256: str
    seed_manifest_path: Path
    seed_manifest_sha256: str
    required_branch_count: int
    required_positions: tuple[int, ...]
    excluded_positions: tuple[int, ...]
    required_strata: tuple[str, ...]
    loss_aggregation: str
    target_action_rule: str
    epochs: int
    learning_rate: float
    maximum_gradient_norm: float
    weight_decay: float
    update_modules: tuple[str, ...]
    ppo_training_steps: int
    train_student_only_evaluations: int
    validation_student_only_evaluations: int
    holdout_episodes: int
    gate: dict[str, object]
    source_path: Path
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """監査要約へ保存できる辞書を返す。"""
        data = asdict(self)
        for name in (
            "source_student_model_path",
            "accepted_branch_manifest_path",
            "m2_4_summary_path",
            "seed_manifest_path",
            "source_path",
        ):
            data[name] = str(data[name])
        data["required_positions"] = list(self.required_positions)
        data["excluded_positions"] = list(self.excluded_positions)
        data["required_strata"] = list(self.required_strata)
        data["update_modules"] = list(self.update_modules)
        return data


@dataclass(frozen=True)
class M3Sequence:
    """一つの成功回全体と分層コード、凍結出典を保持する。"""

    seed: int
    start_runway_voxels: int
    observations: np.ndarray
    actions: np.ndarray
    stratum_codes: np.ndarray
    teacher_mask: np.ndarray
    branch_path: Path
    branch_sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    maximum_replay_observation_difference: float

    @property
    def steps(self) -> int:
        """系列の総歩数を返す。"""
        return int(len(self.observations))


def _resolve_project_path(value: str) -> Path:
    """相対または絶対出典をプロジェクト配下だけへ解決する。"""
    source = Path(value)
    resolved = source.resolve() if source.is_absolute() else (PROJECT_ROOT / source).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M3の出典はプロジェクト配下になければならない。")
    return resolved


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> M3Protocol:
    """凍結M3規約と全主要出典ハッシュ、無教師評価条件を検査する。"""
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M3規約は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M3は単一低壁の訓練成功分岐だけを使用できる。")
    protocol = M3Protocol(
        version=str(payload["version"]),
        stage=str(payload["stage"]),
        split=str(payload["split"]),
        source_student_model_path=_resolve_project_path(
            str(payload["source_student_model_path"])
        ),
        source_student_model_sha256=str(payload["source_student_model_sha256"]),
        accepted_branch_manifest_path=_resolve_project_path(
            str(payload["accepted_branch_manifest_path"])
        ),
        accepted_branch_manifest_sha256=str(
            payload["accepted_branch_manifest_sha256"]
        ),
        m2_4_summary_path=_resolve_project_path(str(payload["m2_4_summary_path"])),
        m2_4_summary_sha256=str(payload["m2_4_summary_sha256"]),
        seed_manifest_path=_resolve_project_path(str(payload["seed_manifest_path"])),
        seed_manifest_sha256=str(payload["seed_manifest_sha256"]),
        required_branch_count=int(payload["required_branch_count"]),
        required_positions=tuple(int(value) for value in payload["required_positions"]),
        excluded_positions=tuple(int(value) for value in payload["excluded_positions"]),
        required_strata=tuple(str(value) for value in payload["required_strata"]),
        loss_aggregation=str(payload["loss_aggregation"]),
        target_action_rule=str(payload["target_action_rule"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        maximum_gradient_norm=float(payload["maximum_gradient_norm"]),
        weight_decay=float(payload["weight_decay"]),
        update_modules=tuple(str(value) for value in payload["update_modules"]),
        ppo_training_steps=int(payload["ppo_training_steps"]),
        train_student_only_evaluations=int(
            payload["train_student_only_evaluations"]
        ),
        validation_student_only_evaluations=int(
            payload["validation_student_only_evaluations"]
        ),
        holdout_episodes=int(payload["holdout_episodes"]),
        gate=dict(payload["gate"]),
        source_path=resolved,
        sha256=sha256_file(resolved),
    )
    protected = (
        (protocol.source_student_model_path, protocol.source_student_model_sha256),
        (
            protocol.accepted_branch_manifest_path,
            protocol.accepted_branch_manifest_sha256,
        ),
        (protocol.m2_4_summary_path, protocol.m2_4_summary_sha256),
        (protocol.seed_manifest_path, protocol.seed_manifest_sha256),
    )
    for source_path, expected_hash in protected:
        if sha256_file(source_path) != expected_hash:
            raise ValueError(f"M3出典ハッシュが一致しない: {source_path}")
    if protocol.required_branch_count != 9:
        raise ValueError("M3の成功分岐数は九本でなければならない。")
    if set(protocol.required_positions) != set(range(20, 31)) - {21, 22}:
        raise ValueError("M3の許可位置がM2.4成功集合と一致しない。")
    if protocol.excluded_positions != (21, 22):
        raise ValueError("M3の除外位置が凍結値と一致しない。")
    if protocol.required_strata != STRATUM_NAMES:
        raise ValueError("M3の分層集合が凍結値と一致しない。")
    if protocol.epochs != 4 or not math.isclose(
        protocol.learning_rate,
        5e-5,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("M3は学習率5e-5で四回だけ模倣しなければならない。")
    if not math.isclose(protocol.maximum_gradient_norm, 0.5):
        raise ValueError("M3の勾配上限は0.5でなければならない。")
    if protocol.ppo_training_steps != 0:
        raise ValueError("M3でPPOを実行することはできない。")
    expected_modules = (
        "lstm_actor",
        "mlp_extractor.policy_net",
        "action_net",
    )
    if protocol.update_modules != expected_modules:
        raise ValueError("M3の更新対象が許可されたactor範囲と一致しない。")
    if protocol.holdout_episodes != 0:
        raise ValueError("M3は留出区分へアクセスできない。")
    if bool(payload.get("validation_teacher_enabled", True)):
        raise ValueError("M3検証では教師を使用できない。")
    if bool(payload.get("holdout_teacher_enabled", True)):
        raise ValueError("M3留出では教師を使用できない。")
    if bool(payload.get("final_student_test_teacher_enabled", True)):
        raise ValueError("最終学生試験では教師を完全停止しなければならない。")
    if int(payload.get("teacher_interventions_in_student_evaluations", -1)) != 0:
        raise ValueError("M3学生評価の教師介入数は零でなければならない。")
    return protocol


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    """二配列の最大絶対差を倍精度で返す。"""
    if left.size == 0:
        return 0.0
    return float(
        np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
    )


def _load_branch_arrays(path: Path) -> dict[str, np.ndarray]:
    """成功分岐の必須配列、形状、有限性、実行動作対応を検査する。"""
    with np.load(path, allow_pickle=False) as archive:
        if not REQUIRED_ARRAYS.issubset(set(archive.files)):
            raise ValueError(f"M3成功分岐の必須配列が不足している: {path}")
        arrays = {name: archive[name].copy() for name in REQUIRED_ARRAYS}
    length = len(arrays["observations"])
    shapes = {
        "observations": (length, 95),
        "student_actions": (length, 6),
        "teacher_actions": (length, 6),
        "executed_actions": (length, 6),
        "teacher_mask": (length,),
        "rescue_ids": (length,),
        "teacher_stages": (length,),
    }
    if any(arrays[name].shape != shape for name, shape in shapes.items()):
        raise ValueError(f"M3成功分岐の配列形状が不正である: {path}")
    for name in (
        "observations",
        "student_actions",
        "teacher_actions",
        "executed_actions",
    ):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"M3成功分岐に非有限値がある: {path}, {name}")
    mask = arrays["teacher_mask"].astype(bool)
    if not np.array_equal(
        arrays["executed_actions"][mask],
        arrays["teacher_actions"][mask],
    ):
        raise ValueError(f"M3教師実行動作がラベルと一致しない: {path}")
    if not np.array_equal(
        arrays["executed_actions"][~mask],
        arrays["student_actions"][~mask],
    ):
        raise ValueError(f"M3学生前置動作が凍結予測と一致しない: {path}")
    return arrays


def _stratum_for_step(
    observation: np.ndarray,
    teacher_active: bool,
    *,
    environment: Any,
    info: Mapping[str, object],
    schema: tuple[str, ...],
) -> int:
    """学生前置と教師救援を四つの等重み分層へ分類する。"""
    if not teacher_active:
        visible = local_terrain_is_visible(
            observation,
            schema,
            maximum_rise_offset=6,
        )
        name = "student_prefix_obstacle" if visible else "student_prefix_flat"
        return STRATUM_CODES[name]
    phase = classify_rescue_phase(environment, info)
    if phase in {PRE_HURDLE_PHASE, HURDLE_DEFORMATION_PHASE}:
        return STRATUM_CODES["teacher_rescue_window"]
    if phase in {POST_CLEARANCE_RECOVERY_PHASE, POST_RECOVERY_STALL_PHASE}:
        return STRATUM_CODES["teacher_recovery"]
    raise RuntimeError(f"M3で未知の救援段階が返された: {phase}")


def load_m3_sequences(
    protocol: M3Protocol,
) -> tuple[tuple[M3Sequence, ...], dict[str, object], tuple[Path, ...]]:
    """九成功分岐を全初期状態から精密再生し四分層系列へ変換する。"""
    manifest = json.loads(
        protocol.accepted_branch_manifest_path.read_text(encoding="utf-8")
    )
    if not bool(manifest.get("frozen", False)):
        raise ValueError("M3接受分岐目録は凍結済みでなければならない。")
    if manifest.get("stage") != protocol.stage or manifest.get("split") != "train":
        raise ValueError("M3接受分岐目録の段階または区分が不正である。")
    if not bool(manifest.get("teacher_training_only", False)):
        raise ValueError("M3出典教師は訓練専用でなければならない。")
    disabled = (
        "validation_teacher_enabled",
        "holdout_teacher_enabled",
        "final_student_test_teacher_enabled",
    )
    if any(bool(manifest.get(name, True)) for name in disabled):
        raise ValueError("M3出典目録の評価教師隔離が不正である。")
    rows = manifest["branches"]
    if len(rows) != protocol.required_branch_count:
        raise ValueError("M3接受分岐目録は九本でなければならない。")
    if {int(row["start_runway_voxels"]) for row in rows} != set(
        protocol.required_positions
    ):
        raise ValueError("M3接受分岐位置が凍結成功集合と一致しない。")
    sequences: list[M3Sequence] = []
    protected_paths = {
        protocol.source_path,
        protocol.source_student_model_path,
        protocol.accepted_branch_manifest_path,
        protocol.m2_4_summary_path,
        protocol.seed_manifest_path,
    }
    total_counts = {name: 0 for name in STRATUM_NAMES}
    for row in rows:
        seed = int(row["seed"])
        position = int(row["start_runway_voxels"])
        if not bool(row["replay_success"]) or bool(row["hard_fall"]):
            raise ValueError(f"M3目録に失敗分岐が含まれている: {seed}")
        if not bool(row["all_step_observations_exact"]):
            raise ValueError(f"M3目録に非精密分岐が含まれている: {seed}")
        branch_path = _resolve_project_path(str(row["branch_path"]))
        sidecar_path = _resolve_project_path(str(row["sidecar_path"]))
        if sha256_file(branch_path) != str(row["branch_sha256"]):
            raise ValueError(f"M3成功分岐ハッシュが一致しない: {seed}")
        if sha256_file(sidecar_path) != str(row["sidecar_sha256"]):
            raise ValueError(f"M3分岐監査JSONハッシュが一致しない: {seed}")
        arrays = _load_branch_arrays(branch_path)
        mask = arrays["teacher_mask"].astype(bool)
        segments = find_true_segments(mask)
        prefix_steps = int(row["prefix_steps"])
        if len(segments) != 1 or segments[0] != (prefix_steps, len(mask)):
            raise ValueError(f"M3教師区間が凍結境界と一致しない: {seed}")
        course = sample_curriculum_course(seed, protocol.stage, "train")
        if int(course.obstacles[0].start_x) != position:
            raise ValueError(f"M3再生成コース位置が目録と一致しない: {seed}")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        stratum_codes = np.empty(len(mask), dtype=np.int8)
        maximum_difference = 0.0
        terminated = False
        truncated = False
        try:
            observation, info = environment.reset(seed=seed)
            schema = tuple(environment.unwrapped.schema)
            for index, action in enumerate(arrays["executed_actions"]):
                difference = _maximum_difference(
                    np.asarray(observation, dtype=np.float32),
                    arrays["observations"][index],
                )
                maximum_difference = max(maximum_difference, difference)
                stratum_codes[index] = _stratum_for_step(
                    np.asarray(observation, dtype=np.float32),
                    bool(mask[index]),
                    environment=environment,
                    info=info,
                    schema=schema,
                )
                observation, _, terminated, truncated, info = environment.step(
                    np.asarray(action, dtype=np.float32)
                )
                if (terminated or truncated) and index != len(mask) - 1:
                    raise RuntimeError(f"M3分岐が保存終端より早く終了した: {seed}")
            if not (terminated or truncated):
                raise RuntimeError(f"M3分岐が保存終端で終了しなかった: {seed}")
            if not bool(info["course_complete"]) or bool(info["hard_fall"]):
                raise RuntimeError(f"M3分岐の精密再生が安全成功しなかった: {seed}")
        finally:
            environment.close()
        if maximum_difference != 0.0:
            raise ValueError(f"M3分岐の逐歩観測が出典と一致しない: {seed}")
        counts = {
            name: int(np.count_nonzero(stratum_codes == code))
            for name, code in STRATUM_CODES.items()
        }
        if any(count < 1 for count in counts.values()):
            raise ValueError(f"M3分岐に空の必須分層がある: {seed}")
        for name, count in counts.items():
            total_counts[name] += count
        sequences.append(
            M3Sequence(
                seed=seed,
                start_runway_voxels=position,
                observations=np.asarray(arrays["observations"], dtype=np.float32),
                actions=np.asarray(arrays["executed_actions"], dtype=np.float32),
                stratum_codes=stratum_codes,
                teacher_mask=mask,
                branch_path=branch_path,
                branch_sha256=str(row["branch_sha256"]),
                sidecar_path=sidecar_path,
                sidecar_sha256=str(row["sidecar_sha256"]),
                maximum_replay_observation_difference=maximum_difference,
            )
        )
        protected_paths.update({branch_path, sidecar_path})
    metadata = {
        "sequence_count": len(sequences),
        "total_steps_per_epoch": sum(sequence.steps for sequence in sequences),
        "stratum_code_mapping": STRATUM_CODES,
        "stratum_step_counts": total_counts,
        "all_source_replays_exact": all(
            sequence.maximum_replay_observation_difference == 0.0
            for sequence in sequences
        ),
        "all_strata_nonempty_per_sequence": True,
        "sequences": [
            {
                "seed": sequence.seed,
                "start_runway_voxels": sequence.start_runway_voxels,
                "steps": sequence.steps,
                "teacher_steps": int(np.count_nonzero(sequence.teacher_mask)),
                "student_steps": int(np.count_nonzero(~sequence.teacher_mask)),
                "stratum_step_counts": {
                    name: int(
                        np.count_nonzero(sequence.stratum_codes == code)
                    )
                    for name, code in STRATUM_CODES.items()
                },
                "source_branch_path": str(sequence.branch_path),
                "source_branch_sha256": sequence.branch_sha256,
                "source_sidecar_path": str(sequence.sidecar_path),
                "source_sidecar_sha256": sequence.sidecar_sha256,
                "maximum_replay_observation_difference": (
                    sequence.maximum_replay_observation_difference
                ),
            }
            for sequence in sequences
        ],
    }
    return tuple(sequences), metadata, tuple(sorted(protected_paths))


def equal_stratum_mean_loss(
    step_losses: torch.Tensor,
    stratum_codes: torch.Tensor,
    *,
    required_codes: tuple[int, ...] = tuple(range(len(STRATUM_NAMES))),
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """各分層内を平均してから四分層を等重みで合成する。"""
    if step_losses.ndim != 1 or stratum_codes.ndim != 1:
        raise ValueError("M3の歩損失と分層コードは一次元でなければならない。")
    if len(step_losses) != len(stratum_codes):
        raise ValueError("M3の歩損失数と分層コード数が一致しない。")
    losses: dict[int, torch.Tensor] = {}
    for code in required_codes:
        selected = step_losses[stratum_codes == code]
        if selected.numel() < 1:
            raise ValueError(f"M3の必須分層が空である: {code}")
        losses[code] = selected.mean()
    return torch.stack(list(losses.values())).mean(), losses


def actor_trainable_parameters(model: Any) -> tuple[torch.nn.Parameter, ...]:
    """M3で更新を許可されたactor三部分のパラメータだけを返す。"""
    parameters: list[torch.nn.Parameter] = []
    parameters.extend(model.policy.lstm_actor.parameters())
    parameters.extend(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("M3のactor更新対象に重複がある。")
    return tuple(parameters)


def _hash_modules(modules: Iterable[tuple[str, torch.nn.Module]]) -> str:
    """指定モジュールの状態から安定したSHA-256を作る。"""
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


def _actor_hash(model: Any) -> str:
    """M3で更新可能なactor部分のハッシュを返す。"""
    return _hash_modules(
        (
            ("lstm_actor", model.policy.lstm_actor),
            ("policy_net", model.policy.mlp_extractor.policy_net),
            ("action_net", model.policy.action_net),
        )
    )


def _critic_hash(model: Any) -> str:
    """M3で更新禁止のcritic部分のハッシュを返す。"""
    return _hash_modules(
        (
            ("lstm_critic", model.policy.lstm_critic),
            ("value_net_body", model.policy.mlp_extractor.value_net),
            ("value_net_head", model.policy.value_net),
        )
    )


def _sequence_losses(
    model: Any,
    sequence: M3Sequence,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """一系列を先頭から通し、循環状態を保った四分層損失を返す。"""
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
    stacked = torch.stack(step_losses)
    codes = torch.as_tensor(
        sequence.stratum_codes,
        dtype=torch.int64,
        device=model.device,
    )
    return equal_stratum_mean_loss(stacked, codes)


def evaluate_dataset_loss(
    model: Any,
    sequences: tuple[M3Sequence, ...],
) -> dict[str, object]:
    """同じ九系列上の等分層模倣損失を更新なしで測定する。"""
    model.policy.set_training_mode(False)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for sequence in sequences:
            total, losses = _sequence_losses(model, sequence)
            rows.append(
                {
                    "seed": sequence.seed,
                    "equal_stratum_loss": float(total.detach().cpu()),
                    "stratum_losses": {
                        name: float(losses[code].detach().cpu())
                        for name, code in STRATUM_CODES.items()
                    },
                }
            )
    return {
        "mean_equal_stratum_loss": float(
            np.mean([row["equal_stratum_loss"] for row in rows])
        ),
        "mean_stratum_losses": {
            name: float(np.mean([row["stratum_losses"][name] for row in rows]))
            for name in STRATUM_NAMES
        },
        "all_losses_finite": bool(
            all(
                math.isfinite(float(row["equal_stratum_loss"]))
                and all(
                    math.isfinite(float(value))
                    for value in row["stratum_losses"].values()
                )
                for row in rows
            )
        ),
        "sequences": rows,
    }


def train_stratified_clone(
    model: Any,
    sequences: tuple[M3Sequence, ...],
    *,
    protocol: M3Protocol,
    seed: int,
) -> dict[str, object]:
    """全回履歴を保ち、九系列ごとに四分層等重みで四回だけ更新する。"""
    torch.manual_seed(seed)
    parameters = actor_trainable_parameters(model)
    optimizer = torch.optim.Adam(
        parameters,
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )
    history: list[dict[str, object]] = []
    optimizer_steps = 0
    model.policy.train()
    for epoch in range(1, protocol.epochs + 1):
        rows: list[dict[str, object]] = []
        for sequence in sequences:
            total, losses = _sequence_losses(model, sequence)
            optimizer.zero_grad()
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=protocol.maximum_gradient_norm,
            )
            optimizer.step()
            optimizer_steps += 1
            rows.append(
                {
                    "seed": sequence.seed,
                    "steps": sequence.steps,
                    "equal_stratum_loss": float(total.detach().cpu()),
                    "stratum_losses": {
                        name: float(losses[code].detach().cpu())
                        for name, code in STRATUM_CODES.items()
                    },
                    "gradient_norm_before_clip": float(
                        gradient_norm.detach().cpu()
                    ),
                }
            )
        history.append(
            {
                "epoch": epoch,
                "mean_equal_stratum_loss": float(
                    np.mean([row["equal_stratum_loss"] for row in rows])
                ),
                "mean_stratum_losses": {
                    name: float(
                        np.mean([row["stratum_losses"][name] for row in rows])
                    )
                    for name in STRATUM_NAMES
                },
                "maximum_gradient_norm_before_clip": float(
                    max(row["gradient_norm_before_clip"] for row in rows)
                ),
                "sequences": rows,
            }
        )
    model.policy.set_training_mode(False)
    return {
        "epochs": protocol.epochs,
        "optimizer_steps": optimizer_steps,
        "supervised_action_presentations": sum(
            sequence.steps for sequence in sequences
        )
        * protocol.epochs,
        "history": history,
    }


def _evaluation_numeric_safe(evaluation: Mapping[str, object]) -> bool:
    """学生単独評価の歩数と連続診断値がすべて有限かを返す。"""
    return bool(
        all(
            int(row["steps"]) > 0
            and math.isfinite(float(row["maximum_angle_degrees"]))
            and math.isfinite(float(row["maximum_com_x"]))
            for row in evaluation["episodes"]
        )
    )


def build_m3_gate(
    *,
    protocol: M3Protocol,
    dataset: Mapping[str, object],
    initial_loss: Mapping[str, object],
    final_loss: Mapping[str, object],
    actor_changed: bool,
    critic_unchanged: bool,
    source_files_unchanged: bool,
    evaluations: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    """データ、損失、更新範囲、無教師評価、留出隔離からM3門を判定する。"""
    checks = {
        "source_replay_requirement_passed": bool(
            dataset["all_source_replays_exact"]
        ),
        "stratum_requirement_passed": bool(
            dataset["all_strata_nonempty_per_sequence"]
        ),
        "finite_loss_requirement_passed": bool(
            initial_loss["all_losses_finite"] and final_loss["all_losses_finite"]
        ),
        "loss_improvement_requirement_passed": float(
            final_loss["mean_equal_stratum_loss"]
        )
        < float(initial_loss["mean_equal_stratum_loss"]),
        "actor_update_requirement_passed": actor_changed,
        "critic_freeze_requirement_passed": critic_unchanged,
        "source_immutability_requirement_passed": source_files_unchanged,
        "student_only_evaluation_requirement_passed": all(
            row["controller_mode"] == "student_only"
            and not bool(row["teacher_module_loaded"])
            and int(row["teacher_interventions"]) == 0
            and _evaluation_numeric_safe(row)
            for row in evaluations
        ),
        "holdout_isolation_requirement_passed": protocol.holdout_episodes == 0,
    }
    gate_passed = all(checks.values())
    return {
        "gate_name": "m3_stratified_recurrent_student_initialization_gate_v1",
        **protocol.gate,
        **checks,
        "gate_passed": gate_passed,
        "eligible_for_m4": gate_passed,
    }


def _write_dataset_indices(
    output_dir: Path,
    sequences: tuple[M3Sequence, ...],
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """各系列の分層索引と出典ハッシュを独立M3目録へ保存する。"""
    index_dir = output_dir / "dataset_indices"
    index_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    metadata_by_seed = {
        int(row["seed"]): row for row in metadata["sequences"]
    }
    for sequence in sequences:
        index_path = index_dir / f"seed_{sequence.seed}_strata.npz"
        np.savez_compressed(
            index_path,
            stratum_codes=sequence.stratum_codes,
            teacher_mask=sequence.teacher_mask,
            student_indices=np.flatnonzero(~sequence.teacher_mask).astype(np.int32),
            teacher_indices=np.flatnonzero(sequence.teacher_mask).astype(np.int32),
        )
        rows.append(
            {
                **metadata_by_seed[sequence.seed],
                "index_path": str(index_path.resolve()),
                "index_sha256": sha256_file(index_path),
            }
        )
    manifest = {
        "version": "m3_stratified_success_dataset_manifest_v1",
        "frozen": True,
        "stage": "hurdle_single",
        "split": "train",
        "source_branch_count": len(sequences),
        "excluded_positions": [21, 22],
        "target_action_rule": "recorded_executed_action",
        "stratum_code_mapping": STRATUM_CODES,
        "stratum_step_counts": metadata["stratum_step_counts"],
        "all_source_replays_exact": metadata["all_source_replays_exact"],
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "sequences": rows,
    }
    path = output_dir / "dataset_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        **manifest,
        "source_path": str(path.resolve()),
        "sha256": sha256_file(path),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """凍結規約、出力名、乱数種だけを受け取るM3引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="九成功分岐でM3循環学生を分層模倣初期化する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """独立学生コピーを四回だけ更新し、訓練と検証を学生単独で評価する。"""
    from sb3_contrib import RecurrentPPO

    args = build_argument_parser().parse_args()
    torch.set_num_threads(1)
    protocol = load_protocol(Path(args.protocol))
    sequences, dataset_metadata, protected_paths = load_m3_sequences(protocol)
    seed_manifest = load_seed_manifest(protocol.seed_manifest_path)
    if seed_manifest.sha256 != protocol.seed_manifest_sha256:
        raise ValueError("M3乱数種目録が凍結規約と一致しない。")
    m2_summary = json.loads(protocol.m2_4_summary_path.read_text(encoding="utf-8"))
    if not bool(m2_summary["m2_4_gate"]["gate_passed"]):
        raise ValueError("M2.4門が未通過のためM3を実行できない。")
    before = {str(path): sha256_file(path) for path in protected_paths}
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    initialization_path = output_dir / "student_init_from_frozen.zip"
    shutil.copy2(protocol.source_student_model_path, initialization_path)
    if sha256_file(initialization_path) != protocol.source_student_model_sha256:
        raise RuntimeError("M3学生初期コピーが凍結出典と一致しない。")
    dataset_manifest = _write_dataset_indices(
        output_dir,
        sequences,
        dataset_metadata,
    )
    baseline_student = RecurrentPPO.load(
        protocol.source_student_model_path,
        device="cpu",
    )
    candidate = RecurrentPPO.load(initialization_path, device="cpu")
    stage = get_curriculum_stage(protocol.stage)
    train_seeds = seed_manifest.for_split("train")
    validation_seeds = seed_manifest.for_split("validation")
    initial_eval_start = time.perf_counter()
    initial_train_evaluation = evaluate_student_batch(
        baseline_student,
        seeds=train_seeds,
        stage=stage,
        split="train",
    )
    initial_validation_evaluation = evaluate_student_batch(
        baseline_student,
        seeds=validation_seeds,
        stage=stage,
        split="validation",
    )
    initial_eval_seconds = time.perf_counter() - initial_eval_start
    full_hash_before = hash_policy_parameters(candidate)
    actor_hash_before = _actor_hash(candidate)
    critic_hash_before = _critic_hash(candidate)
    initial_loss = evaluate_dataset_loss(candidate, sequences)
    training_start = time.perf_counter()
    training = train_stratified_clone(
        candidate,
        sequences,
        protocol=protocol,
        seed=args.seed,
    )
    training_seconds = time.perf_counter() - training_start
    final_loss = evaluate_dataset_loss(candidate, sequences)
    full_hash_after = hash_policy_parameters(candidate)
    actor_hash_after = _actor_hash(candidate)
    critic_hash_after = _critic_hash(candidate)
    actor_changed = actor_hash_after != actor_hash_before
    critic_unchanged = critic_hash_after == critic_hash_before
    if not actor_changed:
        raise RuntimeError("M3でactorパラメータが更新されなかった。")
    if not critic_unchanged:
        raise RuntimeError("M3で更新禁止のcriticパラメータが変更された。")
    checkpoint_path = output_dir / "student_after_m3_initialization.zip"
    candidate.save(checkpoint_path)
    final_eval_start = time.perf_counter()
    final_train_evaluation = evaluate_student_batch(
        candidate,
        seeds=train_seeds,
        stage=stage,
        split="train",
    )
    final_validation_evaluation = evaluate_student_batch(
        candidate,
        seeds=validation_seeds,
        stage=stage,
        split="validation",
    )
    final_eval_seconds = time.perf_counter() - final_eval_start
    after = {str(path): sha256_file(path) for path in protected_paths}
    source_files_unchanged = after == before
    evaluations = (
        initial_train_evaluation,
        initial_validation_evaluation,
        final_train_evaluation,
        final_validation_evaluation,
    )
    gate = build_m3_gate(
        protocol=protocol,
        dataset=dataset_metadata,
        initial_loss=initial_loss,
        final_loss=final_loss,
        actor_changed=actor_changed,
        critic_unchanged=critic_unchanged,
        source_files_unchanged=source_files_unchanged,
        evaluations=evaluations,
    )
    summary = {
        "method": "m3_stratified_recurrent_student_initialization",
        "stage": protocol.stage,
        "split": protocol.split,
        "run_name": args.run_name,
        "seed": args.seed,
        "protocol": protocol.as_dict(),
        "protocol_sha256": protocol.sha256,
        "source_student_model": str(protocol.source_student_model_path),
        "source_student_model_sha256": protocol.source_student_model_sha256,
        "source_student_weights_updated": False,
        "candidate_student_weights_updated": True,
        "student_initialization": str(initialization_path.resolve()),
        "student_initialization_sha256": sha256_file(initialization_path),
        "dataset": dataset_manifest,
        "initial_dataset_loss": initial_loss,
        "behavior_cloning": training,
        "final_dataset_loss": final_loss,
        "ppo_training_steps": 0,
        "full_parameter_sha256_before": full_hash_before,
        "full_parameter_sha256_after": full_hash_after,
        "actor_sha256_before": actor_hash_before,
        "actor_sha256_after": actor_hash_after,
        "actor_parameters_changed": actor_changed,
        "critic_sha256_before": critic_hash_before,
        "critic_sha256_after": critic_hash_after,
        "critic_parameters_unchanged": critic_unchanged,
        "candidate_checkpoint": str(checkpoint_path.resolve()),
        "candidate_checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_train_student_only_evaluation": initial_train_evaluation,
        "initial_validation_student_only_evaluation": (
            initial_validation_evaluation
        ),
        "final_train_student_only_evaluation": final_train_evaluation,
        "final_validation_student_only_evaluation": final_validation_evaluation,
        "train_student_only_evaluation_episodes": 22,
        "validation_student_only_evaluation_episodes": 22,
        "holdout_episodes": 0,
        "teacher_interventions_in_student_evaluations": 0,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "source_hashes_before": before,
        "source_hashes_after": after,
        "protected_source_files_unchanged": source_files_unchanged,
        "m3_gate": gate,
        "checkpoint_disposition": (
            "m4_candidate" if gate["eligible_for_m4"] else "quarantined_failed_m3"
        ),
        "eligible_for_m4": bool(gate["eligible_for_m4"]),
        "eligible_for_final_student_test": False,
        "timing_seconds": {
            "initial_student_only_evaluations": initial_eval_seconds,
            "behavior_cloning": training_seconds,
            "final_student_only_evaluations": final_eval_seconds,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
