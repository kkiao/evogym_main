"""M6で回復点と変形点の間を密な逆向き交接帯へ分割する。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from general_terrain.curriculum import get_curriculum_stage
from general_terrain.seed_manifest import load_seed_manifest
from general_terrain.student_only_evaluation import evaluate_student_batch
from general_terrain.train_m5_reverse_curriculum import (
    M5ReverseCurriculumEnv,
    RollinSpec,
    array_sha256,
    evaluate_flat_retention,
    evaluate_phase_starts,
    load_rollin_specs,
    make_vector_environment,
    resolve_project_path,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "config" / "m6_dense_handoff_protocol_v1.json"
RUNS_ROOT = PROJECT_ROOT / "runs" / "m6_dense_handoff"


@dataclass(frozen=True)
class M6Protocol:
    """密集交接訓練の出典、門限、計算量を保持する。"""

    source_path: Path
    source_model_path: Path
    source_model_sha256: str
    source_summary_path: Path
    source_summary_sha256: str
    phase_reset_manifest_path: Path
    phase_reset_manifest_sha256: str
    seed_manifest_path: Path
    seed_manifest_sha256: str
    device: str
    parallel_environments: int
    checkpoint_interval_steps: int
    maximum_checkpoints_per_band: int
    maximum_student_steps_per_episode: int
    fractions: tuple[float, ...]
    minimum_successes: int
    maximum_hard_falls: int
    training_weights: dict[str, float]


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> M6Protocol:
    """M6規約、教師禁止境界、全保護ハッシュを検査する。"""
    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M6規約は凍結済みでなければならない。")
    if str(payload["device"]) != "cpu" or int(payload["parallel_environments"]) != 8:
        raise ValueError("M6正式訓練はCPU八並列に固定する。")
    if int(payload["teacher_actions_after_student_takeover"]) != 0:
        raise ValueError("M6学生接管後の教師動作は0でなければならない。")
    if bool(payload["validation_teacher_enabled"]):
        raise ValueError("M6検証では教師を有効化できない。")
    if bool(payload["holdout_teacher_enabled"]):
        raise ValueError("M6留出では教師を有効化できない。")
    if int(payload["holdout_episodes"]) != 0:
        raise ValueError("M6は留出区分へアクセスできない。")
    paths = {
        "source_model": resolve_project_path(str(payload["source_model_path"])),
        "source_summary": resolve_project_path(str(payload["source_summary_path"])),
        "phase_reset_manifest": resolve_project_path(
            str(payload["phase_reset_manifest_path"])
        ),
        "seed_manifest": resolve_project_path(str(payload["seed_manifest_path"])),
    }
    for name, protected_path in paths.items():
        if sha256_file(protected_path) != str(payload[f"{name}_sha256"]):
            raise ValueError(f"M6出典ハッシュが一致しない: {protected_path}")
    fractions = tuple(
        float(value) for value in payload["dense_fractions_easy_to_hard"]
    )
    if fractions != tuple(sorted(fractions, reverse=True)):
        raise ValueError("M6密集割合は容易側から困難側へ並べなければならない。")
    if fractions[0] != 1.0 or fractions[-1] != 0.0:
        raise ValueError("M6密集割合は回復点1.0から変形点0.0まで必要である。")
    weights = {
        str(key): float(value) for key, value in payload["training_weights"].items()
    }
    if set(weights) != {"dense_handoff", "post_clearance_recovery", "flat"}:
        raise ValueError("M6訓練混合のモード集合が不正である。")
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("M6訓練混合確率の合計は1でなければならない。")
    return M6Protocol(
        source_path=source_path,
        source_model_path=paths["source_model"],
        source_model_sha256=str(payload["source_model_sha256"]),
        source_summary_path=paths["source_summary"],
        source_summary_sha256=str(payload["source_summary_sha256"]),
        phase_reset_manifest_path=paths["phase_reset_manifest"],
        phase_reset_manifest_sha256=str(payload["phase_reset_manifest_sha256"]),
        seed_manifest_path=paths["seed_manifest"],
        seed_manifest_sha256=str(payload["seed_manifest_sha256"]),
        device=str(payload["device"]),
        parallel_environments=int(payload["parallel_environments"]),
        checkpoint_interval_steps=int(payload["checkpoint_interval_steps"]),
        maximum_checkpoints_per_band=int(payload["maximum_checkpoints_per_band"]),
        maximum_student_steps_per_episode=int(
            payload["maximum_student_steps_per_episode"]
        ),
        fractions=fractions,
        minimum_successes=int(payload["band_pass_minimum_successes"]),
        maximum_hard_falls=int(payload["band_pass_maximum_hard_falls"]),
        training_weights=weights,
    )


def build_dense_specs(
    original_specs: tuple[RollinSpec, ...],
    fraction: float,
) -> tuple[RollinSpec, ...]:
    """各成功軌跡の変形点と回復点を線形補間して四開始点を作る。"""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("密集交接割合は0から1の範囲でなければならない。")
    seeds = sorted({spec.seed for spec in original_specs})
    dense_specs = []
    for seed in seeds:
        deformation = next(
            spec
            for spec in original_specs
            if spec.seed == seed and spec.phase == "hurdle_deformation"
        )
        recovery = next(
            spec
            for spec in original_specs
            if spec.seed == seed and spec.phase == "post_clearance_recovery"
        )
        if deformation.source_branch_path != recovery.source_branch_path:
            raise ValueError("M6補間端点の軌跡分岐が一致しない。")
        source_step = int(
            round(
                deformation.source_step
                + fraction * (recovery.source_step - deformation.source_step)
            )
        )
        with np.load(deformation.source_branch_path, allow_pickle=False) as archive:
            observation = np.asarray(archive["observations"][source_step], dtype=np.float32)
        dense_specs.append(
            RollinSpec(
                reset_id=(
                    f"seed_{seed}_dense_{fraction:.2f}_step_{source_step}"
                ),
                phase="dense_handoff",
                seed=seed,
                course_id=deformation.course_id,
                source_step=source_step,
                source_branch_path=deformation.source_branch_path,
                source_branch_sha256=deformation.source_branch_sha256,
                source_observation_sha256=array_sha256(observation),
            )
        )
    return tuple(dense_specs)


def band_passed(result: Mapping[str, object], protocol: M6Protocol) -> bool:
    """四軌跡の成功数と硬転倒数を密集帯門限へ照合する。"""
    return bool(
        int(result["success_count"]) >= protocol.minimum_successes
        and int(result["hard_fall_count"]) <= protocol.maximum_hard_falls
    )


def append_progress(path: Path, row: Mapping[str, object]) -> None:
    """M6検査点の境界能力と完全検証をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_argument_parser() -> argparse.ArgumentParser:
    """正式実行と小規模試験の引数を定義する。"""
    parser = argparse.ArgumentParser(description="M6密集交接橋接を実行する。")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-name", default="m6_dense_handoff_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--environment-count", type=int)
    parser.add_argument("--maximum-training-checkpoints", type=int)
    return parser


def main() -> None:
    """能力境界を先に走査し失敗帯だけを複数検査点で訓練する。"""
    args = build_argument_parser().parse_args()
    protocol = load_protocol(Path(args.protocol))
    original_specs = load_rollin_specs(protocol.phase_reset_manifest_path)
    seed_manifest = load_seed_manifest(protocol.seed_manifest_path)
    if seed_manifest.sha256 != protocol.seed_manifest_sha256:
        raise ValueError("M6乱数種目録の読み込み後ハッシュが一致しない。")
    train_seeds = seed_manifest.for_split("train")
    validation_seeds = seed_manifest.for_split("validation")
    environment_count = (
        protocol.parallel_environments
        if args.environment_count is None
        else int(args.environment_count)
    )
    if environment_count < 1 or environment_count > protocol.parallel_environments:
        raise ValueError("M6環境数は1から8でなければならない。")
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir()
    protected_paths = (
        protocol.source_model_path,
        protocol.source_summary_path,
        protocol.phase_reset_manifest_path,
        protocol.seed_manifest_path,
        *(spec.source_branch_path for spec in original_specs),
    )
    protected_hashes_before = {
        str(path): sha256_file(path) for path in dict.fromkeys(protected_paths)
    }
    model = PPO.load(protocol.source_model_path, device=protocol.device)
    preflight = []
    for fraction in protocol.fractions:
        dense_specs = build_dense_specs(original_specs, fraction)
        result = evaluate_phase_starts(
            model,
            dense_specs,
            train_seeds,
            phase="dense_handoff",
            maximum_student_steps=protocol.maximum_student_steps_per_episode,
        )
        row = {
            "fraction": fraction,
            "source_steps": [spec.source_step for spec in dense_specs],
            "passed": band_passed(result, protocol),
            **result,
        }
        preflight.append(row)
        print(
            json.dumps(
                {
                    "event": "preflight_band",
                    "fraction": fraction,
                    "success_count": result["success_count"],
                    "hard_fall_count": result["hard_fall_count"],
                    "passed": row["passed"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    first_failure_index = next(
        (
            index
            for index, row in enumerate(preflight)
            if index > 0 and bool(preflight[index - 1]["passed"]) and not bool(row["passed"])
        ),
        None,
    )
    if first_failure_index is None:
        if all(bool(row["passed"]) for row in preflight):
            first_failure_index = len(preflight) - 1
        else:
            raise RuntimeError("M6能力走査に容易側の合格帯がない。")
    active_environment: VecEnv | None = None
    total_steps = 0
    checkpoint_count = 0
    progress = []
    passed_fractions = [
        float(row["fraction"]) for row in preflight if bool(row["passed"])
    ]
    stalled_fraction: float | None = None
    best_bridge_key = (-1, -99, -1.0)
    best_bridge_step = 0
    stage = get_curriculum_stage("hurdle_single")
    initial_validation = evaluate_student_batch(
        model,
        seeds=validation_seeds,
        stage=stage,
        split="validation",
    )
    try:
        for fraction in protocol.fractions[first_failure_index:]:
            dense_specs = build_dense_specs(original_specs, fraction)
            recovery_specs = tuple(
                spec
                for spec in original_specs
                if spec.phase == "post_clearance_recovery"
            )
            training_specs = (*dense_specs, *recovery_specs)
            new_environment = make_vector_environment(
                training_specs,
                train_seeds,
                protocol.training_weights,
                seed=args.seed + int(round((1.0 - fraction) * 100_000)),
                maximum_student_steps=protocol.maximum_student_steps_per_episode,
                environment_count=environment_count,
            )
            model.set_env(new_environment)
            if active_environment is not None:
                active_environment.close()
            active_environment = new_environment
            band_succeeded = False
            for local_checkpoint in range(1, protocol.maximum_checkpoints_per_band + 1):
                if (
                    args.maximum_training_checkpoints is not None
                    and checkpoint_count >= int(args.maximum_training_checkpoints)
                ):
                    stalled_fraction = fraction
                    break
                model.learn(
                    total_timesteps=protocol.checkpoint_interval_steps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                total_steps += protocol.checkpoint_interval_steps
                checkpoint_count += 1
                model.save(checkpoints_dir / f"student_{total_steps}_m6_steps")
                band_result = evaluate_phase_starts(
                    model,
                    dense_specs,
                    train_seeds,
                    phase="dense_handoff",
                    maximum_student_steps=protocol.maximum_student_steps_per_episode,
                )
                validation = evaluate_student_batch(
                    model,
                    seeds=validation_seeds,
                    stage=stage,
                    split="validation",
                )
                passed = band_passed(band_result, protocol)
                row = {
                    "m6_student_steps": total_steps,
                    "fraction": fraction,
                    "local_checkpoint": local_checkpoint,
                    "band_success_count": band_result["success_count"],
                    "band_hard_fall_count": band_result["hard_fall_count"],
                    "band_passed": passed,
                    "validation_success_count": validation["success_count"],
                    "validation_hard_fall_count": validation["hard_fall_count"],
                    "validation_mean_raw_clearances": validation[
                        "mean_raw_clearances"
                    ],
                    "validation_mean_recovered_obstacles": validation[
                        "mean_recovered_obstacles"
                    ],
                    "validation_mean_max_x": validation["mean_max_x"],
                }
                progress.append({**row, "band_result": band_result, "validation": validation})
                append_progress(output_dir / "checkpoint_progress.csv", row)
                bridge_key = (
                    int(band_result["success_count"]),
                    -int(band_result["hard_fall_count"]),
                    -fraction,
                )
                if bridge_key > best_bridge_key:
                    best_bridge_key = bridge_key
                    best_bridge_step = total_steps
                    model.save(output_dir / "best_bridge_student")
                print(json.dumps({"event": "checkpoint", **row}, ensure_ascii=False), flush=True)
                if passed:
                    passed_fractions.append(fraction)
                    band_succeeded = True
                    model.save(output_dir / f"passed_fraction_{fraction:.2f}")
                    break
            if not band_succeeded:
                stalled_fraction = fraction
                break
        model.save(output_dir / "final_student")
        if not (output_dir / "best_bridge_student.zip").exists():
            model.save(output_dir / "best_bridge_student")
            best_bridge_step = total_steps
        best_model = PPO.load(output_dir / "best_bridge_student.zip", device="cpu")
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
        flat_retention = evaluate_flat_retention(
            best_model,
            original_specs,
            train_seeds,
            maximum_student_steps=protocol.maximum_student_steps_per_episode,
        )
        final_dense_scan = []
        for fraction in protocol.fractions:
            dense_specs = build_dense_specs(original_specs, fraction)
            result = evaluate_phase_starts(
                best_model,
                dense_specs,
                train_seeds,
                phase="dense_handoff",
                maximum_student_steps=protocol.maximum_student_steps_per_episode,
            )
            final_dense_scan.append(
                {
                    "fraction": fraction,
                    "passed": band_passed(result, protocol),
                    **result,
                }
            )
        protected_hashes_after = {
            str(path): sha256_file(path) for path in dict.fromkeys(protected_paths)
        }
        if protected_hashes_after != protected_hashes_before:
            raise RuntimeError("M6中に保護出典が変更された。")
        summary = {
            "method": "m6_dense_reverse_handoff_frontier_ppo",
            "run_name": args.run_name,
            "protocol_path": str(protocol.source_path),
            "protocol_sha256": sha256_file(protocol.source_path),
            "source_model_path": str(protocol.source_model_path),
            "source_model_sha256": protocol.source_model_sha256,
            "single_shared_student": True,
            "algorithm_changed_from_m5": False,
            "network_changed_from_m5": False,
            "reward_changed_from_m5": False,
            "primary_change": "dense_handoff_reset_distribution",
            "device": protocol.device,
            "parallel_environments": environment_count,
            "teacher_module_loaded_in_student_evaluation": False,
            "teacher_actions_after_student_takeover": 0,
            "validation_teacher_interventions": 0,
            "holdout_episodes": 0,
            "completed_steps": total_steps,
            "checkpoint_count": checkpoint_count,
            "best_bridge_step": best_bridge_step,
            "preflight_dense_scan": preflight,
            "training_progress": progress,
            "passed_fractions": sorted(set(passed_fractions), reverse=True),
            "stalled_fraction": stalled_fraction,
            "initial_validation": initial_validation,
            "final_student_validation": final_validation,
            "best_student_validation": best_validation,
            "best_student_train": train_evaluation,
            "best_student_flat_retention": flat_retention,
            "final_dense_scan": final_dense_scan,
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
