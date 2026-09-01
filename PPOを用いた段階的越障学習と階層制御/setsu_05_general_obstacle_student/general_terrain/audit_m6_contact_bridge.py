"""M6で発見した狭い接触断層を凍結動作と学生接管で監査する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.train_m5_reverse_curriculum import (
    array_sha256,
    load_rollin_specs,
    sha256_file,
)
from general_terrain.train_m6_dense_handoff import (
    DEFAULT_PROTOCOL,
    build_dense_specs,
    load_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "m6_contact_bridge_audit"


def actor_predictions(model: PPO, observations: np.ndarray) -> np.ndarray:
    """決定論的actor平均動作を一括で返す。"""
    import torch

    with torch.no_grad():
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
        latent = model.policy.mlp_extractor.forward_actor(tensor)
        actions = model.policy.action_net(latent)
    return actions.detach().cpu().numpy().astype(np.float32)


def audit_contact_bridge(
    model: PPO,
    original_specs,
    *,
    start_fraction: float,
    end_fraction: float,
) -> dict[str, object]:
    """狭い凍結動作橋の誤差と橋通過後の学生完走を測る。"""
    start_specs = build_dense_specs(original_specs, start_fraction)
    end_specs = build_dense_specs(original_specs, end_fraction)
    end_by_seed = {spec.seed: spec for spec in end_specs}
    rows = []
    for start_spec in start_specs:
        end_spec = end_by_seed[start_spec.seed]
        if end_spec.source_step <= start_spec.source_step:
            raise ValueError("接触橋の終点は始点より後でなければならない。")
        with np.load(start_spec.source_branch_path, allow_pickle=False) as archive:
            observations = np.asarray(
                archive["observations"][start_spec.source_step : end_spec.source_step],
                dtype=np.float32,
            )
            target_actions = np.asarray(
                archive["executed_actions"][start_spec.source_step : end_spec.source_step],
                dtype=np.float32,
            )
            prefix_actions = np.asarray(
                archive["executed_actions"][: end_spec.source_step],
                dtype=np.float32,
            )
        predictions = actor_predictions(model, observations)
        course = sample_curriculum_course(start_spec.seed, "hurdle_single", "train")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        try:
            observation, info = environment.reset(seed=start_spec.seed)
            for step, action in enumerate(prefix_actions):
                observation, _, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        f"接触橋再生が早期終了した: {start_spec.seed}, {step + 1}"
                    )
            if array_sha256(np.asarray(observation, dtype=np.float32)) != (
                end_spec.source_observation_sha256
            ):
                raise RuntimeError("接触橋終点の観測が凍結軌跡と一致しない。")
            student_steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
                student_steps += 1
        finally:
            environment.close()
        rows.append(
            {
                "seed": start_spec.seed,
                "start_step": start_spec.source_step,
                "end_step": end_spec.source_step,
                "bridge_steps": end_spec.source_step - start_spec.source_step,
                "action_mse": float(np.mean((predictions - target_actions) ** 2)),
                "action_maximum_absolute_error": float(
                    np.max(np.abs(predictions - target_actions))
                ),
                "student_steps_after_bridge": student_steps,
                "success": bool(info["course_complete"] and not info["hard_fall"]),
                "hard_fall": bool(info["hard_fall"]),
                "raw_clearances": int(info["raw_clearances"]),
                "recovered_obstacles": int(info["recovered_obstacles"]),
            }
        )
    return {
        "method": "frozen_nominal_contact_bridge_then_student",
        "start_fraction": start_fraction,
        "end_fraction": end_fraction,
        "episodes": len(rows),
        "bridge_steps_mean": float(np.mean([row["bridge_steps"] for row in rows])),
        "action_mse_mean": float(np.mean([row["action_mse"] for row in rows])),
        "success_count": sum(bool(row["success"]) for row in rows),
        "hard_fall_count": sum(bool(row["hard_fall"]) for row in rows),
        "teacher_module_loaded": False,
        "frozen_demonstration_actions_before_student_takeover": int(
            sum(int(row["bridge_steps"]) for row in rows)
        ),
        "teacher_actions_after_student_takeover": 0,
        "rows": rows,
    }


def main() -> None:
    """凍結M5学生を更新せず接触断層の最小橋を監査する。"""
    parser = argparse.ArgumentParser(description="M6接触断層を無更新で監査する。")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--run-name", default="m6_contact_bridge_audit_v1")
    args = parser.parse_args()
    protocol = load_protocol(Path(args.protocol))
    original_specs = load_rollin_specs(protocol.phase_reset_manifest_path)
    protected_paths = (
        protocol.source_model_path,
        protocol.phase_reset_manifest_path,
        *(spec.source_branch_path for spec in original_specs),
    )
    hashes_before = {str(path): sha256_file(path) for path in dict.fromkeys(protected_paths)}
    model = PPO.load(protocol.source_model_path, device="cpu")
    result = audit_contact_bridge(
        model,
        original_specs,
        start_fraction=0.3,
        end_fraction=0.4,
    )
    hashes_after = {str(path): sha256_file(path) for path in dict.fromkeys(protected_paths)}
    if hashes_after != hashes_before:
        raise RuntimeError("接触橋監査中に保護出典が変更された。")
    summary = {
        **result,
        "run_name": args.run_name,
        "source_model_path": str(protocol.source_model_path),
        "source_model_sha256": protocol.source_model_sha256,
        "student_weight_updates": 0,
        "holdout_episodes": 0,
        "protected_hashes_before": hashes_before,
        "protected_hashes_after": hashes_after,
        "protected_sources_unchanged": True,
    }
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
