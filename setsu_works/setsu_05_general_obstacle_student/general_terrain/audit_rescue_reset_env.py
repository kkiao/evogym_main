"""M2.2の学生プレフィックス救援環境を訓練せず厳密に監査する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from general_terrain.environment import PRIVILEGED_OBSERVATION_NAMES
from general_terrain.rescue_reset_manifest import (
    DEFAULT_RESCUE_RESET_MANIFEST,
    RescueResetManifest,
    load_rescue_reset_manifest,
)
from general_terrain.student_prefix_rescue_env import StudentPrefixRescueEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = PROJECT_ROOT / "runs" / "rescue_reset_audit"


def _file_sha256(path: Path) -> str:
    """指定ファイルのSHA-256を小文字で返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    """連続配列の型、形状、値からSHA-256を返す。"""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def audit_reset_determinism(
    environment: StudentPrefixRescueEnv,
    manifest: RescueResetManifest,
    *,
    repeats: int = 2,
) -> list[dict[str, object]]:
    """各引き継ぎ状態を複数回再生し、観測と物理座標を比較する。"""
    if repeats < 2:
        raise ValueError("決定性監査の重置回数は二以上でなければならない。")
    rows: list[dict[str, object]] = []
    for spec in manifest.states:
        reference_observation: np.ndarray | None = None
        reference_positions: np.ndarray | None = None
        maximum_observation_difference = 0.0
        maximum_position_difference = 0.0
        observation_hashes: list[str] = []
        position_hashes: list[str] = []
        final_info: dict[str, object] | None = None
        for _ in range(repeats):
            observation, info = environment.reset(
                options={"prefix_seed": spec.seed}
            )
            base = environment.base_environment.unwrapped
            positions = np.asarray(
                base.object_pos_at_time(base.get_time(), "robot"),
                dtype=np.float64,
            )
            observation = np.asarray(observation, dtype=np.float32)
            observation_hashes.append(_array_sha256(observation))
            position_hashes.append(_array_sha256(positions))
            if reference_observation is None:
                reference_observation = observation.copy()
                reference_positions = positions.copy()
            else:
                maximum_observation_difference = max(
                    maximum_observation_difference,
                    float(np.max(np.abs(observation - reference_observation))),
                )
                maximum_position_difference = max(
                    maximum_position_difference,
                    float(np.max(np.abs(positions - reference_positions))),
                )
            final_info = dict(info)
        if final_info is None or reference_observation is None:
            raise RuntimeError("救援重置監査が実行されなかった。")
        deterministic = bool(
            maximum_observation_difference == 0.0
            and maximum_position_difference == 0.0
            and len(set(observation_hashes)) == 1
            and len(set(position_hashes)) == 1
        )
        rows.append(
            {
                "seed": spec.seed,
                "start_runway_voxels": spec.start_runway_voxels,
                "prefix_steps": spec.prefix_steps,
                "trigger_reason": spec.trigger_reason,
                "repeat_count": repeats,
                "observation_shape": list(reference_observation.shape),
                "observation_sha256": observation_hashes[0],
                "robot_positions_sha256": position_hashes[0],
                "maximum_observation_difference": maximum_observation_difference,
                "maximum_position_difference": maximum_position_difference,
                "deterministic": deterministic,
                "x_position": float(final_info["x_position"]),
                "orientation_error": float(final_info["orientation_error"]),
                "angular_velocity": float(final_info["angular_velocity"]),
                "stall_steps": int(final_info["stall_steps"]),
                "raw_clearances": int(final_info["raw_clearances"]),
                "recovered_obstacles": int(final_info["recovered_obstacles"]),
                "rescue_phase": str(final_info["rescue_phase"]),
            }
        )
    return rows


def run_action_smoke_test(
    environment: StudentPrefixRescueEnv,
    *,
    prefix_seed: int,
    requested_steps: int = 10,
) -> dict[str, object]:
    """ゼロ動作で教師ステップの入出力と有限報酬を確認する。"""
    if requested_steps < 1:
        raise ValueError("煙試験の歩数は一以上でなければならない。")
    observation, info = environment.reset(options={"prefix_seed": prefix_seed})
    rewards: list[float] = []
    phases = [str(info["rescue_phase"])]
    terminated = False
    truncated = False
    for _ in range(requested_steps):
        observation, reward, terminated, truncated, info = environment.step(
            np.zeros(environment.action_space.shape, dtype=np.float32)
        )
        rewards.append(float(reward))
        phases.append(str(info["rescue_phase"]))
        if terminated or truncated:
            break
    return {
        "prefix_seed": prefix_seed,
        "requested_steps": requested_steps,
        "completed_steps": len(rewards),
        "observation_shape": list(np.asarray(observation).shape),
        "action_shape": list(environment.action_space.shape),
        "all_rewards_finite": bool(np.all(np.isfinite(rewards))),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "phases": phases,
    }


def run_sb3_environment_check(environment: StudentPrefixRescueEnv) -> bool:
    """Stable-Baselines3の環境契約検査を学習なしで実行する。"""
    from stable_baselines3.common.env_checker import check_env

    check_env(environment, warn=True, skip_render_check=True)
    return True


def build_audit_summary(
    student: Any,
    manifest: RescueResetManifest,
    *,
    model_path: Path,
    repeats: int,
    smoke_steps: int,
) -> dict[str, object]:
    """実際のEvoGym環境におけるリセットの決定性と1ステップの入出力をまとめる。"""
    environment = StudentPrefixRescueEnv(student, manifest.states)
    try:
        rows = audit_reset_determinism(environment, manifest, repeats=repeats)
        sb3_environment_check_passed = run_sb3_environment_check(environment)
        smoke_seed = manifest.states[len(manifest.states) // 2].seed
        smoke = run_action_smoke_test(
            environment,
            prefix_seed=smoke_seed,
            requested_steps=smoke_steps,
        )
        schema = environment.schema
        observation_shape = list(environment.observation_space.shape)
        action_shape = list(environment.action_space.shape)
    finally:
        environment.close()
    privileged_fields = sorted(
        name for name in schema if name in PRIVILEGED_OBSERVATION_NAMES
    )
    deterministic_count = sum(bool(row["deterministic"]) for row in rows)
    return {
        "method": "frozen_student_prefix_rescue_reset_audit",
        "stage": manifest.stage,
        "split": manifest.split,
        "student_model": str(model_path.resolve()),
        "student_model_sha256": _file_sha256(model_path),
        "student_weights_updated": False,
        "teacher_weights_updated": False,
        "training_steps": 0,
        "teacher_environment_is_training_only": True,
        "validation_episodes": 0,
        "holdout_episodes": 0,
        "manifest": manifest.as_dict(),
        "reset_state_count": len(rows),
        "repeat_count_per_state": repeats,
        "deterministic_state_count": deterministic_count,
        "all_states_deterministic": deterministic_count == len(rows),
        "observation_shape": observation_shape,
        "action_shape": action_shape,
        "privileged_observation_fields": privileged_fields,
        "student_observation_privileged": bool(privileged_fields),
        "sb3_environment_check_passed": sb3_environment_check_passed,
        "states": rows,
        "smoke_test": smoke,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """重み更新を行わないM2.2監査引数を定義する。"""
    parser = argparse.ArgumentParser(
        description="凍結学生前缀救援環境の決定性を監査する。"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--reset-manifest",
        default=str(DEFAULT_RESCUE_RESET_MANIFEST),
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--smoke-steps", type=int, default=10)
    return parser


def main() -> None:
    """学生を凍結のまま読み込みM2.2監査JSONを保存する。"""
    from sb3_contrib import RecurrentPPO

    args = build_argument_parser().parse_args()
    model_path = Path(args.model)
    manifest = load_rescue_reset_manifest(Path(args.reset_manifest))
    model_hash = _file_sha256(model_path)
    if model_hash != manifest.student_model_sha256:
        raise ValueError("学生モデルのハッシュが救援重置目録と一致しない。")
    output_dir = AUDIT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    student = RecurrentPPO.load(model_path, device="cpu")
    summary = build_audit_summary(
        student,
        manifest,
        model_path=model_path,
        repeats=args.repeats,
        smoke_steps=args.smoke_steps,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
