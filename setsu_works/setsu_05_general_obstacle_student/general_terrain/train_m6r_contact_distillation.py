"""M6Rで十九歩の接触窓だけを蒸留しオンラインPPOで定着させる。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
import torch

from general_terrain.audit_m6_contact_bridge import actor_predictions
from general_terrain.curriculum import get_curriculum_stage
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_only_evaluation import evaluate_student_batch
from general_terrain.train_m5_reverse_curriculum import (
    evaluate_flat_retention,
    evaluate_phase_starts,
    load_rollin_specs,
    make_vector_environment,
    resolve_project_path,
    sha256_file,
)
from general_terrain.train_m6_dense_handoff import build_dense_specs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "m6r_contact_distillation_protocol_v1.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "m6r_contact_distillation"


@dataclass(frozen=True)
class DistillationCandidate:
    """一つの接触窓蒸留強度を保持する。"""

    name: str
    learning_rate: float
    epochs: int


@dataclass(frozen=True)
class M6RProtocol:
    """接触窓蒸留、旧挙動の錨、オンライン上限を保持する。"""

    source_path: Path
    source_model_path: Path
    source_model_sha256: str
    contact_bridge_audit_path: Path
    contact_bridge_audit_sha256: str
    phase_reset_manifest_path: Path
    phase_reset_manifest_sha256: str
    seed_manifest_path: Path
    seed_manifest_sha256: str
    start_fraction: float
    end_fraction: float
    anchor_stride: int
    anchor_loss_weight: float
    maximum_gradient_norm: float
    candidates: tuple[DistillationCandidate, ...]
    minimum_band_successes: int
    minimum_flat_successes: int
    online_parallel_environments: int
    online_checkpoint_interval_steps: int
    online_total_steps: int
    online_training_weights: dict[str, float]


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> M6RProtocol:
    """M6R規約、出典監査、教師禁止境界を検査する。"""
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M6R規約は凍結済みでなければならない。")
    if int(payload["teacher_actions_after_student_takeover"]) != 0:
        raise ValueError("M6R学生接管後の教師動作は0でなければならない。")
    if bool(payload["validation_teacher_enabled"]):
        raise ValueError("M6R検証では教師を有効化できない。")
    if bool(payload["holdout_teacher_enabled"]):
        raise ValueError("M6R留出では教師を有効化できない。")
    if int(payload["holdout_episodes"]) != 0:
        raise ValueError("M6Rは留出区分へアクセスできない。")
    paths = {
        "source_model": resolve_project_path(str(payload["source_model_path"])),
        "contact_bridge_audit": resolve_project_path(
            str(payload["contact_bridge_audit_path"])
        ),
        "phase_reset_manifest": resolve_project_path(
            str(payload["phase_reset_manifest_path"])
        ),
        "seed_manifest": resolve_project_path(str(payload["seed_manifest_path"])),
    }
    for name, protected_path in paths.items():
        if sha256_file(protected_path) != str(payload[f"{name}_sha256"]):
            raise ValueError(f"M6R出典ハッシュが一致しない: {protected_path}")
    audit = json.loads(paths["contact_bridge_audit"].read_text(encoding="utf-8"))
    if int(audit["success_count"]) != 4 or int(audit["hard_fall_count"]) != 0:
        raise ValueError("M6R接触橋は4/4成功かつ0硬転倒でなければならない。")
    if int(audit["student_weight_updates"]) != 0:
        raise ValueError("M6R接触橋監査は無更新でなければならない。")
    candidates = tuple(
        DistillationCandidate(
            name=str(row["name"]),
            learning_rate=float(row["learning_rate"]),
            epochs=int(row["epochs"]),
        )
        for row in payload["candidates"]
    )
    if len(candidates) < 3 or len({row.name for row in candidates}) != len(candidates):
        raise ValueError("M6R候補は三つ以上の固有名が必要である。")
    weights = {
        str(key): float(value)
        for key, value in payload["online_training_weights"].items()
    }
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("M6Rオンライン混合確率の合計は1でなければならない。")
    return M6RProtocol(
        source_path=source_path,
        source_model_path=paths["source_model"],
        source_model_sha256=str(payload["source_model_sha256"]),
        contact_bridge_audit_path=paths["contact_bridge_audit"],
        contact_bridge_audit_sha256=str(payload["contact_bridge_audit_sha256"]),
        phase_reset_manifest_path=paths["phase_reset_manifest"],
        phase_reset_manifest_sha256=str(payload["phase_reset_manifest_sha256"]),
        seed_manifest_path=paths["seed_manifest"],
        seed_manifest_sha256=str(payload["seed_manifest_sha256"]),
        start_fraction=float(payload["start_fraction"]),
        end_fraction=float(payload["end_fraction"]),
        anchor_stride=int(payload["anchor_stride"]),
        anchor_loss_weight=float(payload["anchor_loss_weight"]),
        maximum_gradient_norm=float(payload["maximum_gradient_norm"]),
        candidates=candidates,
        minimum_band_successes=int(payload["candidate_minimum_band_successes"]),
        minimum_flat_successes=int(payload["candidate_minimum_flat_successes"]),
        online_parallel_environments=int(payload["online_parallel_environments"]),
        online_checkpoint_interval_steps=int(
            payload["online_checkpoint_interval_steps"]
        ),
        online_total_steps=int(payload["online_total_steps"]),
        online_training_weights=weights,
    )


def build_distillation_dataset(
    original_specs,
    *,
    start_fraction: float,
    end_fraction: float,
    anchor_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """接触教師対と接触外の源学生錨観測を構築する。"""
    start_specs = build_dense_specs(original_specs, start_fraction)
    end_by_seed = {
        spec.seed: spec for spec in build_dense_specs(original_specs, end_fraction)
    }
    contact_observations = []
    contact_actions = []
    anchor_observations = []
    for start_spec in start_specs:
        end_spec = end_by_seed[start_spec.seed]
        with np.load(start_spec.source_branch_path, allow_pickle=False) as archive:
            observations = np.asarray(archive["observations"], dtype=np.float32)
            actions = np.asarray(archive["executed_actions"], dtype=np.float32)
        contact_observations.append(
            observations[start_spec.source_step : end_spec.source_step]
        )
        contact_actions.append(actions[start_spec.source_step : end_spec.source_step])
        anchor_observations.append(observations[: start_spec.source_step : anchor_stride])
        anchor_observations.append(observations[end_spec.source_step :: anchor_stride])
    return (
        np.concatenate(contact_observations, axis=0).astype(np.float32),
        np.concatenate(contact_actions, axis=0).astype(np.float32),
        np.concatenate(anchor_observations, axis=0).astype(np.float32),
    )


def module_sha256(modules: tuple[tuple[str, torch.nn.Module], ...]) -> str:
    """複数PyTorch部分の状態を安定したハッシュへ変換する。"""
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


def actor_hash(model: PPO) -> str:
    """更新対象actor二部分のハッシュを返す。"""
    return module_sha256(
        (
            ("policy_net", model.policy.mlp_extractor.policy_net),
            ("action_net", model.policy.action_net),
        )
    )


def critic_hash(model: PPO) -> str:
    """更新禁止critic二部分のハッシュを返す。"""
    return module_sha256(
        (
            ("value_net_body", model.policy.mlp_extractor.value_net),
            ("value_net_head", model.policy.value_net),
        )
    )


def distill_contact_window(
    model: PPO,
    contact_observations: np.ndarray,
    contact_actions: np.ndarray,
    anchor_observations: np.ndarray,
    anchor_actions: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    anchor_loss_weight: float,
    maximum_gradient_norm: float,
    seed: int,
) -> dict[str, object]:
    """接触誤差を減らしながら源学生の接触外動作を錨定する。"""
    torch.manual_seed(seed)
    parameters = list(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    contact_tensor = torch.as_tensor(
        contact_observations, dtype=torch.float32, device=model.device
    )
    target_tensor = torch.as_tensor(
        contact_actions, dtype=torch.float32, device=model.device
    )
    anchor_tensor = torch.as_tensor(
        anchor_observations, dtype=torch.float32, device=model.device
    )
    anchor_target_tensor = torch.as_tensor(
        anchor_actions, dtype=torch.float32, device=model.device
    )
    history = []
    model.policy.train()
    for epoch in range(1, epochs + 1):
        contact_latent = model.policy.mlp_extractor.forward_actor(contact_tensor)
        contact_predictions = model.policy.action_net(contact_latent)
        anchor_latent = model.policy.mlp_extractor.forward_actor(anchor_tensor)
        anchor_predictions_tensor = model.policy.action_net(anchor_latent)
        contact_loss = torch.mean((contact_predictions - target_tensor) ** 2)
        anchor_loss = torch.mean(
            (anchor_predictions_tensor - anchor_target_tensor) ** 2
        )
        loss = contact_loss + anchor_loss_weight * anchor_loss
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(parameters, maximum_gradient_norm).cpu()
        )
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "contact_loss": float(contact_loss.detach().cpu()),
                "anchor_loss": float(anchor_loss.detach().cpu()),
                "combined_loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": gradient_norm,
            }
        )
    return {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "anchor_loss_weight": anchor_loss_weight,
        "history": history,
    }


def candidate_gate(
    band_result: dict[str, object],
    flat_result: dict[str, object],
    protocol: M6RProtocol,
) -> bool:
    """接触改善と平地保持を同時に満たす候補だけを許可する。"""
    return bool(
        int(band_result["success_count"]) >= protocol.minimum_band_successes
        and int(flat_result["success_count"]) >= protocol.minimum_flat_successes
        and int(flat_result["hard_fall_count"]) == 0
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """M6R候補探索とオンライン定着の引数を定義する。"""
    parser = argparse.ArgumentParser(description="M6R接触窓定向蒸留を実行する。")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-name", default="m6r_contact_distillation_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-online", action="store_true")
    return parser


def main() -> None:
    """独立蒸留候補を選別し合格時だけCPU八並列PPOを続ける。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    original_specs = load_rollin_specs(protocol.phase_reset_manifest_path)
    dense_specs = build_dense_specs(original_specs, protocol.start_fraction)
    recovery_specs = tuple(
        spec for spec in original_specs if spec.phase == "post_clearance_recovery"
    )
    seed_manifest = load_seed_manifest(protocol.seed_manifest_path)
    if seed_manifest.sha256 != protocol.seed_manifest_sha256:
        raise ValueError("M6R乱数種目録の読み込み後ハッシュが一致しない。")
    train_seeds = seed_manifest.for_split("train")
    validation_seeds = seed_manifest.for_split("validation")
    protected_paths = (
        protocol.source_model_path,
        protocol.contact_bridge_audit_path,
        protocol.phase_reset_manifest_path,
        protocol.seed_manifest_path,
        *(spec.source_branch_path for spec in original_specs),
    )
    hashes_before = {str(path): sha256_file(path) for path in dict.fromkeys(protected_paths)}
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates_dir = output_dir / "candidates"
    candidates_dir.mkdir()
    checkpoints_dir = output_dir / "online_checkpoints"
    checkpoints_dir.mkdir()
    contact_observations, contact_actions, anchor_observations = (
        build_distillation_dataset(
            original_specs,
            start_fraction=protocol.start_fraction,
            end_fraction=protocol.end_fraction,
            anchor_stride=protocol.anchor_stride,
        )
    )
    source_model = PPO.load(protocol.source_model_path, device="cpu")
    anchor_actions = actor_predictions(source_model, anchor_observations)
    source_actor_hash = actor_hash(source_model)
    source_critic_hash = critic_hash(source_model)
    initial_contact_mse = float(
        np.mean((actor_predictions(source_model, contact_observations) - contact_actions) ** 2)
    )
    candidate_rows = []
    for index, candidate in enumerate(protocol.candidates):
        model = PPO.load(protocol.source_model_path, device="cpu")
        critic_before = critic_hash(model)
        training = distill_contact_window(
            model,
            contact_observations,
            contact_actions,
            anchor_observations,
            anchor_actions,
            epochs=candidate.epochs,
            learning_rate=candidate.learning_rate,
            anchor_loss_weight=protocol.anchor_loss_weight,
            maximum_gradient_norm=protocol.maximum_gradient_norm,
            seed=args.seed + index,
        )
        contact_mse = float(
            np.mean((actor_predictions(model, contact_observations) - contact_actions) ** 2)
        )
        anchor_mse = float(
            np.mean((actor_predictions(model, anchor_observations) - anchor_actions) ** 2)
        )
        band_result = evaluate_phase_starts(
            model,
            dense_specs,
            train_seeds,
            phase="dense_handoff",
            maximum_student_steps=1200,
        )
        flat_result = evaluate_flat_retention(
            model,
            original_specs,
            train_seeds,
            maximum_student_steps=1200,
        )
        gate_passed = candidate_gate(band_result, flat_result, protocol)
        candidate_path = candidates_dir / candidate.name
        model.save(candidate_path)
        row = {
            "name": candidate.name,
            "learning_rate": candidate.learning_rate,
            "epochs": candidate.epochs,
            "training": training,
            "contact_mse": contact_mse,
            "anchor_mse": anchor_mse,
            "band_result": band_result,
            "flat_result": flat_result,
            "gate_passed": gate_passed,
            "actor_changed": actor_hash(model) != source_actor_hash,
            "critic_unchanged": critic_hash(model) == critic_before == source_critic_hash,
            "model_path": str(candidate_path.with_suffix(".zip").resolve()),
        }
        candidate_rows.append(row)
        print(
            json.dumps(
                {
                    "event": "candidate",
                    "name": candidate.name,
                    "contact_mse": contact_mse,
                    "anchor_mse": anchor_mse,
                    "band_success_count": band_result["success_count"],
                    "band_hard_fall_count": band_result["hard_fall_count"],
                    "flat_success_count": flat_result["success_count"],
                    "gate_passed": gate_passed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    eligible = [row for row in candidate_rows if bool(row["gate_passed"])]
    selected = max(
        eligible,
        key=lambda row: (
            int(row["band_result"]["success_count"]),
            -int(row["band_result"]["hard_fall_count"]),
            -float(row["anchor_mse"]),
            -float(row["contact_mse"]),
        ),
        default=None,
    )
    online_history = []
    online_steps = 0
    active_environment: Any = None
    if selected is not None:
        selected_model = PPO.load(Path(str(selected["model_path"])), device="cpu")
        selected_model.save(output_dir / "selected_distilled_student")
        if not args.skip_online:
            training_specs = (*dense_specs, *recovery_specs)
            active_environment = make_vector_environment(
                training_specs,
                train_seeds,
                protocol.online_training_weights,
                seed=args.seed,
                maximum_student_steps=1200,
                environment_count=protocol.online_parallel_environments,
            )
            selected_model.set_env(active_environment)
            while online_steps < protocol.online_total_steps:
                selected_model.learn(
                    total_timesteps=protocol.online_checkpoint_interval_steps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                online_steps += protocol.online_checkpoint_interval_steps
                selected_model.save(
                    checkpoints_dir / f"student_{online_steps}_online_steps"
                )
                band_result = evaluate_phase_starts(
                    selected_model,
                    dense_specs,
                    train_seeds,
                    phase="dense_handoff",
                    maximum_student_steps=1200,
                )
                validation = evaluate_student_batch(
                    selected_model,
                    seeds=validation_seeds,
                    stage=get_curriculum_stage("hurdle_single"),
                    split="validation",
                )
                online_history.append(
                    {
                        "online_steps": online_steps,
                        "band_result": band_result,
                        "validation": validation,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "online_checkpoint",
                            "online_steps": online_steps,
                            "band_success_count": band_result["success_count"],
                            "band_hard_fall_count": band_result["hard_fall_count"],
                            "validation_success_count": validation["success_count"],
                            "validation_hard_fall_count": validation["hard_fall_count"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        selected_model.save(output_dir / "final_student")
    else:
        source_model.save(output_dir / "final_student")
    if active_environment is not None:
        active_environment.close()
    final_model = PPO.load(output_dir / "final_student.zip", device="cpu")
    final_band = evaluate_phase_starts(
        final_model,
        dense_specs,
        train_seeds,
        phase="dense_handoff",
        maximum_student_steps=1200,
    )
    final_validation = evaluate_student_batch(
        final_model,
        seeds=validation_seeds,
        stage=get_curriculum_stage("hurdle_single"),
        split="validation",
    )
    final_flat = evaluate_flat_retention(
        final_model,
        original_specs,
        train_seeds,
        maximum_student_steps=1200,
    )
    hashes_after = {str(path): sha256_file(path) for path in dict.fromkeys(protected_paths)}
    if hashes_after != hashes_before:
        raise RuntimeError("M6R中に保護出典が変更された。")
    summary = {
        "method": "m6r_targeted_contact_distillation_then_online_ppo",
        "run_name": args.run_name,
        "protocol_path": str(protocol.source_path),
        "protocol_sha256": sha256_file(protocol.source_path),
        "source_model_path": str(protocol.source_model_path),
        "source_model_sha256": protocol.source_model_sha256,
        "contact_rows": int(len(contact_observations)),
        "anchor_rows": int(len(anchor_observations)),
        "initial_contact_mse": initial_contact_mse,
        "candidate_rows": candidate_rows,
        "selected_candidate": None if selected is None else selected["name"],
        "online_steps": online_steps,
        "online_history": online_history,
        "final_band_result": final_band,
        "final_validation": final_validation,
        "final_flat_retention": final_flat,
        "teacher_role": "frozen_training_demonstration_only",
        "teacher_module_loaded_in_student_evaluation": False,
        "teacher_actions_after_student_takeover": 0,
        "validation_teacher_interventions": 0,
        "holdout_episodes": 0,
        "breakthrough": {
            "criterion": "at_least_one_teacher_free_validation_course_complete",
            "achieved": int(final_validation["success_count"]) >= 1,
            "validation_success_count": int(final_validation["success_count"]),
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
