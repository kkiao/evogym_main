"""安全回帰したM3初版を隔離し、低変更量の学生候補を有界探索する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
import torch

from general_terrain.audit_rescue_demonstrations import sha256_file
from general_terrain.curriculum import get_curriculum_stage
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_only_evaluation import evaluate_student_batch
from general_terrain.train_m3_stratified_student import (
    M3Sequence,
    _actor_hash,
    _critic_hash,
    _sequence_losses,
    actor_trainable_parameters,
    evaluate_dataset_loss,
    load_m3_sequences,
    load_protocol as load_m3_protocol,
)
from general_terrain.train_prefix_rescue_teacher import hash_policy_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    PROJECT_ROOT / "config" / "m3_1_safe_initialization_search_protocol_v1.json"
)
RUNS_ROOT = PROJECT_ROOT / "runs" / "m3_safe_student_search"


def _resolve_project_path(value: str) -> Path:
    """規約内の相対出典をプロジェクト配下だけへ解決する。"""
    resolved = (PROJECT_ROOT / value).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M3.1の出典はプロジェクト配下になければならない。")
    return resolved


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict[str, object]:
    """凍結M3.1規約、失敗候補、候補上限、学生評価隔離を検査する。"""
    source_path = path.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M3.1規約は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M3.1は単一低壁の訓練出典だけを使用できる。")
    path_fields = (
        "source_student_model_path",
        "m3_protocol_path",
        "failed_m3_summary_path",
        "failed_m3_checkpoint_path",
    )
    resolved_paths: dict[str, Path] = {}
    for field in path_fields:
        resolved = _resolve_project_path(str(payload[field]))
        expected_hash = str(payload[f"{field.removesuffix('_path')}_sha256"])
        if sha256_file(resolved) != expected_hash:
            raise ValueError(f"M3.1出典ハッシュが一致しない: {resolved}")
        resolved_paths[field] = resolved
    candidates = payload["candidates"]
    if int(payload["candidate_limit"]) != 4 or len(candidates) != 4:
        raise ValueError("M3.1候補数は四つに限定しなければならない。")
    identifiers = [str(row["candidate_id"]) for row in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("M3.1候補識別子が重複している。")
    allowed_scopes = {"action_net", "policy_action", "full_actor"}
    for row in candidates:
        if str(row["update_scope"]) not in allowed_scopes:
            raise ValueError("M3.1候補の更新範囲が未知である。")
        if int(row["epochs"]) not in {1, 2}:
            raise ValueError("M3.1候補は一回または二回だけ更新できる。")
        if not 0.0 < float(row["learning_rate"]) <= 1e-5:
            raise ValueError("M3.1候補の学習率が許可上限を超えている。")
    if int(payload["ppo_training_steps"]) != 0:
        raise ValueError("M3.1でPPOを実行することはできない。")
    if payload["selection_split"] != "validation":
        raise ValueError("M3.1候補選択は凍結検証区分だけで行わなければならない。")
    if int(payload["holdout_episodes"]) != 0:
        raise ValueError("M3.1は留出区分へアクセスできない。")
    disabled = (
        "validation_teacher_enabled",
        "holdout_teacher_enabled",
        "final_student_test_teacher_enabled",
    )
    if any(bool(payload.get(field, True)) for field in disabled):
        raise ValueError("M3.1学生評価では教師を完全停止しなければならない。")
    if int(payload["teacher_interventions_in_student_evaluations"]) != 0:
        raise ValueError("M3.1学生評価の教師介入数は零でなければならない。")
    failed = json.loads(
        resolved_paths["failed_m3_summary_path"].read_text(encoding="utf-8")
    )
    initial = failed["initial_validation_student_only_evaluation"]
    final = failed["final_validation_student_only_evaluation"]
    if int(final["hard_fall_count"]) <= int(initial["hard_fall_count"]):
        raise ValueError("M3初版に安全回帰がないためM3.1を実行する根拠がない。")
    return {
        **payload,
        **resolved_paths,
        "source_path": source_path,
        "sha256": sha256_file(source_path),
    }


def _parameters_for_scope(model: Any, scope: str) -> tuple[torch.nn.Parameter, ...]:
    """候補ごとの許可範囲だけから重複しないパラメータ列を返す。"""
    if scope == "action_net":
        parameters = tuple(model.policy.action_net.parameters())
    elif scope == "policy_action":
        parameters = tuple(model.policy.mlp_extractor.policy_net.parameters()) + tuple(
            model.policy.action_net.parameters()
        )
    elif scope == "full_actor":
        parameters = actor_trainable_parameters(model)
    else:
        raise ValueError(f"未知のM3.1更新範囲: {scope}")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("M3.1更新対象に重複パラメータがある。")
    return parameters


def train_candidate(
    model: Any,
    sequences: tuple[M3Sequence, ...],
    configuration: Mapping[str, object],
    *,
    maximum_gradient_norm: float,
    weight_decay: float,
    seed: int,
) -> dict[str, object]:
    """指定範囲と低学習率で一候補を最大二回だけ分層模倣する。"""
    torch.manual_seed(seed)
    parameters = _parameters_for_scope(model, str(configuration["update_scope"]))
    optimizer = torch.optim.Adam(
        parameters,
        lr=float(configuration["learning_rate"]),
        weight_decay=weight_decay,
    )
    history: list[dict[str, object]] = []
    optimizer_steps = 0
    model.policy.train()
    for epoch in range(1, int(configuration["epochs"]) + 1):
        losses: list[float] = []
        gradients: list[float] = []
        for sequence in sequences:
            total, _ = _sequence_losses(model, sequence)
            optimizer.zero_grad()
            total.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=maximum_gradient_norm,
            )
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(total.detach().cpu()))
            gradients.append(float(gradient.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "mean_equal_stratum_loss": float(np.mean(losses)),
                "maximum_gradient_norm_before_clip": float(max(gradients)),
            }
        )
    model.policy.set_training_mode(False)
    return {
        "candidate_id": str(configuration["candidate_id"]),
        "update_scope": str(configuration["update_scope"]),
        "epochs": int(configuration["epochs"]),
        "learning_rate": float(configuration["learning_rate"]),
        "optimizer_steps": optimizer_steps,
        "supervised_action_presentations": sum(
            sequence.steps for sequence in sequences
        )
        * int(configuration["epochs"]),
        "history": history,
    }


def _candidate_score(evaluation: Mapping[str, object], loss: float) -> tuple[float, ...]:
    """安全門通過候補を能力、回復、転倒、進捗、損失の順に並べる。"""
    return (
        float(evaluation["success_count"]),
        float(evaluation["mean_recovered_obstacles"]),
        float(evaluation["mean_raw_clearances"]),
        -float(evaluation["hard_fall_count"]),
        float(evaluation["mean_max_x"]),
        -loss,
    )


def evaluate_safety_gate(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    initial_loss: float,
    final_loss: float,
    actor_changed: bool,
    critic_unchanged: bool,
) -> dict[str, object]:
    """検証成功、転倒、越壁、回復を凍結学生より悪化させない門を判定する。"""
    checks = {
        "success_non_regression_passed": int(candidate["success_count"])
        >= int(baseline["success_count"]),
        "hard_fall_non_regression_passed": int(candidate["hard_fall_count"])
        <= int(baseline["hard_fall_count"]),
        "raw_clearance_non_regression_passed": float(
            candidate["mean_raw_clearances"]
        )
        >= float(baseline["mean_raw_clearances"]),
        "recovery_non_regression_passed": float(
            candidate["mean_recovered_obstacles"]
        )
        >= float(baseline["mean_recovered_obstacles"]),
        "dataset_loss_decrease_passed": math.isfinite(final_loss)
        and final_loss < initial_loss,
        "actor_update_passed": actor_changed,
        "critic_freeze_passed": critic_unchanged,
        "student_only_evaluation_passed": (
            candidate["controller_mode"] == "student_only"
            and not bool(candidate["teacher_module_loaded"])
            and int(candidate["teacher_interventions"]) == 0
        ),
    }
    return {**checks, "gate_passed": all(checks.values())}


def build_argument_parser() -> argparse.ArgumentParser:
    """凍結規約、出力名、乱数種だけを受け取るM3.1引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="安全回帰したM3初版を隔離して低変更量学生を探索する。"
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    return parser


def main() -> None:
    """四候補を独立初期化し、無教師検証の非回帰門で一つだけ選択する。"""
    from sb3_contrib import RecurrentPPO

    args = build_argument_parser().parse_args()
    torch.set_num_threads(1)
    protocol = load_protocol(Path(args.protocol))
    m3_protocol = load_m3_protocol(protocol["m3_protocol_path"])
    sequences, dataset, m3_protected = load_m3_sequences(m3_protocol)
    seed_manifest = load_seed_manifest(m3_protocol.seed_manifest_path)
    stage = get_curriculum_stage("hurdle_single")
    validation_seeds = seed_manifest.for_split("validation")
    train_seeds = seed_manifest.for_split("train")
    protected_paths = set(m3_protected)
    protected_paths.update(
        {
            Path(protocol["source_path"]),
            Path(protocol["failed_m3_summary_path"]),
            Path(protocol["failed_m3_checkpoint_path"]),
        }
    )
    protected_paths = tuple(sorted(protected_paths))
    before = {str(path): sha256_file(path) for path in protected_paths}
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir()
    baseline_model = RecurrentPPO.load(
        protocol["source_student_model_path"],
        device="cpu",
    )
    baseline_validation = evaluate_student_batch(
        baseline_model,
        seeds=validation_seeds,
        stage=stage,
        split="validation",
    )
    baseline_loss = evaluate_dataset_loss(baseline_model, sequences)
    candidate_rows: list[dict[str, object]] = []
    search_start = time.perf_counter()
    for index, configuration in enumerate(protocol["candidates"]):
        model = RecurrentPPO.load(
            protocol["source_student_model_path"],
            device="cpu",
        )
        actor_before = _actor_hash(model)
        critic_before = _critic_hash(model)
        full_before = hash_policy_parameters(model)
        training = train_candidate(
            model,
            sequences,
            configuration,
            maximum_gradient_norm=float(protocol["maximum_gradient_norm"]),
            weight_decay=float(protocol["weight_decay"]),
            seed=args.seed + index,
        )
        final_loss = evaluate_dataset_loss(model, sequences)
        actor_after = _actor_hash(model)
        critic_after = _critic_hash(model)
        full_after = hash_policy_parameters(model)
        checkpoint = candidates_dir / f"{configuration['candidate_id']}.zip"
        model.save(checkpoint)
        validation = evaluate_student_batch(
            model,
            seeds=validation_seeds,
            stage=stage,
            split="validation",
        )
        safety_gate = evaluate_safety_gate(
            baseline_validation,
            validation,
            initial_loss=float(baseline_loss["mean_equal_stratum_loss"]),
            final_loss=float(final_loss["mean_equal_stratum_loss"]),
            actor_changed=actor_after != actor_before,
            critic_unchanged=critic_after == critic_before,
        )
        score = _candidate_score(
            validation,
            float(final_loss["mean_equal_stratum_loss"]),
        )
        candidate_rows.append(
            {
                "configuration": dict(configuration),
                "training": training,
                "initial_dataset_loss": baseline_loss,
                "final_dataset_loss": final_loss,
                "full_parameter_sha256_before": full_before,
                "full_parameter_sha256_after": full_after,
                "actor_sha256_before": actor_before,
                "actor_sha256_after": actor_after,
                "actor_parameters_changed": actor_after != actor_before,
                "critic_sha256_before": critic_before,
                "critic_sha256_after": critic_after,
                "critic_parameters_unchanged": critic_after == critic_before,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation_student_only_evaluation": validation,
                "safety_gate": safety_gate,
                "selection_score": list(score),
                "disposition": (
                    "safe_candidate"
                    if safety_gate["gate_passed"]
                    else "quarantined_safety_regression"
                ),
            }
        )
    search_seconds = time.perf_counter() - search_start
    passing = [row for row in candidate_rows if row["safety_gate"]["gate_passed"]]
    selected = max(passing, key=lambda row: tuple(row["selection_score"])) if passing else None
    selected_train_evaluation: dict[str, object] | None = None
    selected_path: Path | None = None
    if selected is not None:
        selected_path = output_dir / "selected_student.zip"
        shutil.copy2(Path(selected["checkpoint_path"]), selected_path)
        selected_model = RecurrentPPO.load(selected_path, device="cpu")
        selected_train_evaluation = evaluate_student_batch(
            selected_model,
            seeds=train_seeds,
            stage=stage,
            split="train",
        )
        selected["disposition"] = "selected_m4_candidate"
    after = {str(path): sha256_file(path) for path in protected_paths}
    source_files_unchanged = after == before
    m3_gate = {
        "gate_name": "m3_1_safe_initialization_selection_gate_v1",
        "candidate_limit_respected": len(candidate_rows)
        <= int(protocol["candidate_limit"]),
        "safe_candidate_count": len(passing),
        "safe_candidate_available": selected is not None,
        "source_files_unchanged": source_files_unchanged,
        "validation_teacher_interventions": sum(
            int(row["validation_student_only_evaluation"]["teacher_interventions"])
            for row in candidate_rows
        ),
        "holdout_episodes": 0,
        "gate_passed": bool(selected is not None and source_files_unchanged),
        "eligible_for_m4": bool(selected is not None and source_files_unchanged),
    }
    quarantine = {
        "version": "m3_v1_safety_regression_quarantine_v1",
        "failed_summary_path": str(protocol["failed_m3_summary_path"]),
        "failed_summary_sha256": protocol["failed_m3_summary_sha256"],
        "failed_checkpoint_path": str(protocol["failed_m3_checkpoint_path"]),
        "failed_checkpoint_sha256": protocol["failed_m3_checkpoint_sha256"],
        "reason": "student_only_hard_falls_increased_and_clearance_collapsed",
        "eligible_for_m4": False,
        "eligible_for_final_student_test": False,
    }
    quarantine_path = output_dir / "quarantined_m3_v1.json"
    quarantine_path.write_text(
        json.dumps(quarantine, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "method": "m3_1_safe_low_change_student_search",
        "stage": "hurdle_single",
        "split": "train",
        "run_name": args.run_name,
        "seed": args.seed,
        "protocol_path": str(protocol["source_path"]),
        "protocol_sha256": protocol["sha256"],
        "research_basis": [
            "DAgger: learner-induced state distribution requires corrective labels",
            "DART: off-policy cloning errors compound and recovery demonstrations matter",
        ],
        "failed_m3_v1_quarantine_path": str(quarantine_path.resolve()),
        "failed_m3_v1_quarantine_sha256": sha256_file(quarantine_path),
        "dataset": dataset,
        "baseline_dataset_loss": baseline_loss,
        "baseline_validation_student_only_evaluation": baseline_validation,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "selected_candidate_id": (
            selected["configuration"]["candidate_id"] if selected else None
        ),
        "selected_student_path": str(selected_path.resolve()) if selected_path else None,
        "selected_student_sha256": (
            sha256_file(selected_path) if selected_path else None
        ),
        "selected_validation_student_only_evaluation": (
            selected["validation_student_only_evaluation"] if selected else None
        ),
        "selected_train_student_only_evaluation": selected_train_evaluation,
        "source_student_weights_updated": False,
        "candidate_student_weights_updated": True,
        "ppo_training_steps": 0,
        "baseline_validation_episodes": 11,
        "candidate_validation_episodes": 44,
        "selected_train_diagnostic_episodes": 11 if selected else 0,
        "holdout_episodes": 0,
        "teacher_interventions_in_student_evaluations": 0,
        "validation_teacher_enabled": False,
        "holdout_teacher_enabled": False,
        "final_student_test_teacher_enabled": False,
        "source_hashes_before": before,
        "source_hashes_after": after,
        "protected_source_files_unchanged": source_files_unchanged,
        "m3_gate": m3_gate,
        "eligible_for_m4": m3_gate["eligible_for_m4"],
        "eligible_for_final_student_test": False,
        "timing_seconds": {"candidate_search": search_seconds},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
