"""時系列を保持した模倣と保守的更新で循環着地回復教師を作る。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.environment import GeneralObstacleEnv
from general_terrain.terrain import build_course
from general_terrain.train_recovery_teacher import (
    POSITIONS,
    RecoveryPrefixEnv,
    safe_flat_handoff,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "recurrent_recovery_runs"
BRANCH_RUN = (
    PROJECT_ROOT
    / "runs"
    / "height1_teacher_recovery_branches"
    / "height1_recovery_branches_seed7_v1"
)


def collect_branch_sequences() -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """合格した摂動分岐を再生し完全通過後の救回区間を抽出する。"""
    summary = json.loads((BRANCH_RUN / "summary.json").read_text(encoding="utf-8"))
    sequences: list[tuple[np.ndarray, np.ndarray]] = []
    rows = []
    for item in summary["branch_index"]:
        position = int(item["position"])
        episode_seed = int(item["seed"])
        course = build_course(
            ["low_hurdle"],
            split="teacher_recovery_branch",
            seed=episode_seed,
            difficulty=1,
            start_runway_voxels=position,
        )
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        data = np.load(BRANCH_RUN / "branches" / str(item["file"]))
        trajectory_observations = np.asarray(data["observations"], dtype=np.float32)
        trajectory_actions = np.asarray(data["actions"], dtype=np.float32)
        injection_step = int(np.asarray(data["injection_step"])[0])
        direction = np.asarray(item["direction"], dtype=np.float32)
        magnitude = float(item["magnitude"])
        start_index: int | None = None
        safe_streak = 0
        end_index = len(trajectory_actions)
        try:
            _, info = environment.reset(seed=episode_seed)
            for step, teacher_action in enumerate(trajectory_actions):
                executed_action = teacher_action
                if step == injection_step:
                    executed_action = np.clip(
                        teacher_action + magnitude * direction,
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                _, _, terminated, truncated, info = environment.step(executed_action)
                if start_index is None and int(info["raw_clearances"]) >= 1:
                    start_index = step + 1
                if start_index is not None:
                    ready = safe_flat_handoff(environment, info)
                    safe_streak = safe_streak + 1 if ready else 0
                    if safe_streak >= 20 or terminated or truncated:
                        end_index = step + 1
                        break
            if start_index is None or bool(info["hard_fall"]):
                raise RuntimeError(f"合格分岐の再生に失敗した：{item['file']}")
            observations = trajectory_observations[start_index:end_index]
            actions = trajectory_actions[start_index:end_index]
            if len(observations) == 0:
                continue
            sequences.append((observations, actions))
            rows.append(
                {
                    "file": str(item["file"]),
                    "position": position,
                    "target": str(item["target"]),
                    "steps": len(observations),
                    "safe_handoff": safe_streak >= 20,
                }
            )
        finally:
            environment.close()
    return sequences, {"episodes": rows, "total_rows": sum(row["steps"] for row in rows)}


def collect_sequences(*, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """各壁位置の実通過直後から旧回復動作を時系列のまま収集する。"""
    sequences: list[tuple[np.ndarray, np.ndarray]] = []
    rows = []
    for index, position in enumerate(POSITIONS):
        course = build_course(
            ["low_hurdle"],
            split="recurrent_recovery_bootstrap",
            seed=seed + index,
            difficulty=1,
            start_runway_voxels=position,
        )
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        teacher = ClosedLoopHeight1Teacher(
            post_clear_mode="restart_then_flat",
            clearance_blend=1.0,
            handoff_distance=0.45,
            adaptive_handoff=True,
        )
        observations = []
        actions = []
        try:
            observation, info = environment.reset(seed=seed + index)
            teacher.reset(environment)
            while int(info["raw_clearances"]) < 1:
                action, _ = teacher.predict(environment, info)
                observation, _, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    raise RuntimeError("循环教师前缀在完整越墙前失败。")
            safe_streak = 0
            for _ in range(300):
                action, _ = teacher.predict(environment, info)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(np.asarray(action, dtype=np.float32))
                observation, _, terminated, truncated, info = environment.step(action)
                ready = safe_flat_handoff(environment, info)
                safe_streak = safe_streak + 1 if ready else 0
                if safe_streak >= 20 or terminated or truncated:
                    break
            accepted = bool(
                not info["hard_fall"]
                and (safe_streak >= 20 or info["course_complete"])
            )
            if accepted:
                sequences.append(
                    (
                        np.asarray(observations, dtype=np.float32),
                        np.asarray(actions, dtype=np.float32),
                    )
                )
            rows.append(
                {
                    "position": position,
                    "steps": len(observations),
                    "hard_fall": bool(info["hard_fall"]),
                    "accepted": accepted,
                }
            )
        finally:
            environment.close()
    branch_sequences, branch_metadata = collect_branch_sequences()
    sequences.extend(branch_sequences)
    return sequences, {
        "nominal_episodes": rows,
        "branch_dataset": branch_metadata,
        "sequence_count": len(sequences),
        "total_rows": sum(len(observations) for observations, _ in sequences),
    }


def recurrent_behavior_clone(
    model: RecurrentPPO,
    sequences: list[tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int,
) -> list[float]:
    """各軌跡内のLSTM状態を維持して教師動作平均へ回帰する。"""
    parameters = list(model.policy.lstm_actor.parameters())
    parameters.extend(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    losses = []
    model.policy.train()
    for _ in range(epochs):
        epoch_losses = []
        for observations, actions in sequences:
            hidden = torch.zeros(
                model.policy.lstm_hidden_state_shape,
                device=model.device,
            )
            cell = torch.zeros_like(hidden)
            sequence_loss = torch.zeros((), device=model.device)
            for step in range(len(observations)):
                observation = torch.as_tensor(
                    observations[step : step + 1],
                    device=model.device,
                )
                target = torch.as_tensor(
                    actions[step : step + 1],
                    device=model.device,
                )
                episode_start = torch.as_tensor(
                    [float(step == 0)],
                    device=model.device,
                )
                distribution, (hidden, cell) = model.policy.get_distribution(
                    observation,
                    (hidden, cell),
                    episode_start,
                )
                prediction = distribution.distribution.mean
                sequence_loss = sequence_loss + torch.mean((prediction - target) ** 2)
            sequence_loss = sequence_loss / max(len(observations), 1)
            optimizer.zero_grad()
            sequence_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            epoch_losses.append(float(sequence_loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    with torch.no_grad():
        model.policy.log_std.fill_(-2.5)
    return losses


def evaluate(model: RecurrentPPO, *, seed: int) -> dict[str, object]:
    """全十一位置でLSTM状態を保持した決定論的回復を検査する。"""
    environment = RecoveryPrefixEnv(seed=seed)
    rows = []
    try:
        for index, position in enumerate(POSITIONS):
            observation, info = environment.reset(seed=seed + index)
            state = None
            episode_start = np.ones((1,), dtype=bool)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, state = model.predict(
                    observation,
                    state=state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                observation, _, terminated, truncated, info = environment.step(action)
            rows.append(
                {
                    "position": position,
                    "success": bool(info["recovery_teacher_success"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "steps": int(info["recovery_steps"]),
                }
            )
    finally:
        environment.close()
    return {
        "episodes": rows,
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "hard_fall_rate": float(np.mean([row["hard_fall"] for row in rows])),
    }


def append_evaluation(path: Path, step: int, result: dict[str, object]) -> None:
    """循環方策の定期評価をCSVへ追記する。"""
    row = {
        "step": step,
        "success_rate": result["success_rate"],
        "hard_fall_rate": result["hard_fall_rate"],
    }
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """循環模倣を検収し、指定時だけ低雑音PPO更新を追加する。"""
    parser = argparse.ArgumentParser(description="训练循环落地恢复教师。")
    parser.add_argument("--run-name", default="recurrent_recovery_seed7_v1")
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--evaluate-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-model")
    args = parser.parse_args()

    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    environment = RecoveryPrefixEnv(seed=args.seed)
    try:
        if args.initial_model:
            model = RecurrentPPO.load(
                Path(args.initial_model),
                env=environment,
                device="cpu",
            )
            initialization = {
                "observation_dimension": 95,
                "continued_from": str(Path(args.initial_model).resolve()),
                "privileged_inputs_in_student_observation": False,
            }
        else:
            model = RecurrentPPO(
                "MlpLstmPolicy",
                environment,
                learning_rate=1e-5,
                n_steps=512,
                batch_size=64,
                n_epochs=1,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.0,
                clip_range=0.05,
                policy_kwargs={
                    "net_arch": [128, 128],
                    "lstm_hidden_size": 128,
                    "n_lstm_layers": 1,
                    "shared_lstm": False,
                    "enable_critic_lstm": True,
                },
                seed=args.seed,
                device="cpu",
                verbose=0,
            )
            sequences, dataset = collect_sequences(seed=args.seed)
            losses = recurrent_behavior_clone(
                model,
                sequences,
                epochs=args.bc_epochs,
            )
            initialization = {
                "observation_dimension": 95,
                "sequence_dataset": dataset,
                "bc_epochs": args.bc_epochs,
                "first_loss": losses[0],
                "final_loss": losses[-1],
                "log_standard_deviation": -2.5,
                "privileged_inputs_in_student_observation": False,
            }
        (output_dir / "initialization.json").write_text(
            json.dumps(initialization, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        model.save(output_dir / "initial_model")
        initial = evaluate(model, seed=args.seed + 20_000)
        append_evaluation(output_dir / "evaluation.csv", 0, initial)
        model.save(output_dir / "best_model")
        best_key = (float(initial["success_rate"]), -float(initial["hard_fall_rate"]))
        best_step = 0
        print(json.dumps({"step": 0, **initial}, ensure_ascii=False), flush=True)

        completed = 0
        while completed < args.total_steps:
            chunk = min(args.evaluate_every, args.total_steps - completed)
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            completed = int(model.num_timesteps)
            result = evaluate(model, seed=args.seed + 20_000)
            append_evaluation(output_dir / "evaluation.csv", completed, result)
            model.save(output_dir / "checkpoints" / f"model_{completed}_steps")
            key = (float(result["success_rate"]), -float(result["hard_fall_rate"]))
            if key > best_key:
                best_key = key
                best_step = completed
                model.save(output_dir / "best_model")
            print(json.dumps({"step": completed, **result}, ensure_ascii=False), flush=True)

        model.save(output_dir / "final_model")
        summary = {
            "completed_steps": completed,
            "best_step": best_step,
            "best_success_rate": best_key[0],
            "best_hard_fall_rate": -best_key[1],
            "initial": initial,
            "teacher_gate_passed": bool(best_key == (1.0, -0.0)),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
