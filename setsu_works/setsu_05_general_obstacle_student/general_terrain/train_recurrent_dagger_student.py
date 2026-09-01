"""時系列記憶付き学生を閉ループ教師との示範集約で訓練する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "height1_recurrent_dagger_student"
POSITIONS = tuple(range(20, 31))
Sequence = tuple[np.ndarray, np.ndarray]


def make_course(position: int, seed: int, split: str):
    """指定位置に矮壁を一つ置いた再現可能コースを返す。"""
    return build_course(
        ["low_hurdle"],
        split=split,
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )


def collect_sequences(
    model: RecurrentPPO | None,
    *,
    seed: int,
    beta: float,
) -> tuple[list[Sequence], dict[str, object]]:
    """学生のLSTM履歴を維持しながら訪問状態へ教師ラベルを付ける。"""
    rng = np.random.default_rng(seed)
    teacher = PortfolioHeight1Teacher()
    sequences: list[Sequence] = []
    episodes = []
    for index, position in enumerate(POSITIONS):
        environment = GeneralObstacleEnv(
            course=make_course(position, seed + index, "recurrent_dagger_collection"),
            resample_on_reset=False,
        )
        observations = []
        actions = []
        try:
            observation, info = environment.reset(seed=seed + index)
            teacher.reset(environment)
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            terminated = False
            truncated = False
            teacher_steps = 0
            steps = 0
            while not (terminated or truncated):
                teacher_action, _ = teacher.predict(environment, observation, info)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(np.asarray(teacher_action, dtype=np.float32))
                if model is None:
                    student_action = teacher_action
                else:
                    student_action, recurrent_state = model.predict(
                        observation,
                        state=recurrent_state,
                        episode_start=episode_start,
                        deterministic=True,
                    )
                episode_start[:] = False
                if model is None or rng.random() < beta:
                    executed_action = teacher_action
                    teacher_steps += 1
                else:
                    executed_action = student_action
                observation, _, terminated, truncated, info = environment.step(
                    executed_action
                )
                steps += 1
            sequences.append(
                (
                    np.asarray(observations, dtype=np.float32),
                    np.asarray(actions, dtype=np.float32),
                )
            )
            episodes.append(
                {
                    "position": position,
                    "steps": steps,
                    "teacher_action_fraction": teacher_steps / max(1, steps),
                    "course_complete": bool(info["course_complete"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "failure_reason": str(info["failure_reason"]),
                }
            )
        finally:
            environment.close()
    return sequences, {
        "beta": beta,
        "rows": int(sum(len(sequence[0]) for sequence in sequences)),
        "collection_success_rate": float(
            np.mean([episode["course_complete"] for episode in episodes])
        ),
        "episodes": episodes,
    }


def recurrent_behavior_clone(
    model: RecurrentPPO,
    sequences: list[Sequence],
    *,
    epochs: int,
    chunk_length: int = 128,
    learning_rate: float = 2e-4,
) -> list[float]:
    """切断逆伝播で履歴を保ちつつ教師の平均動作へ回帰する。"""
    parameters = list(model.policy.lstm_actor.parameters())
    parameters.extend(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-6)
    losses = []
    model.policy.train()
    for _ in range(epochs):
        epoch_losses = []
        for observations, actions in sequences:
            hidden = torch.zeros(model.policy.lstm_hidden_state_shape, device=model.device)
            cell = torch.zeros_like(hidden)
            for chunk_start in range(0, len(observations), chunk_length):
                chunk_end = min(len(observations), chunk_start + chunk_length)
                chunk_loss = torch.zeros((), device=model.device)
                for step in range(chunk_start, chunk_end):
                    observation = torch.as_tensor(
                        observations[step : step + 1],
                        dtype=torch.float32,
                        device=model.device,
                    )
                    target = torch.as_tensor(
                        actions[step : step + 1],
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
                    chunk_loss = chunk_loss + torch.mean((prediction - target) ** 2)
                chunk_loss = chunk_loss / max(1, chunk_end - chunk_start)
                optimizer.zero_grad()
                chunk_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
                hidden = hidden.detach()
                cell = cell.detach()
                epoch_losses.append(float(chunk_loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    with torch.no_grad():
        model.policy.log_std.fill_(-3.0)
    return losses


def evaluate_student(model: RecurrentPPO, *, seed: int) -> dict[str, object]:
    """循環状態を各回で初期化し学生だけの全十一位置成績を返す。"""
    episodes = []
    for index, position in enumerate(POSITIONS):
        environment = GeneralObstacleEnv(
            course=make_course(position, seed + index, "recurrent_dagger_validation"),
            resample_on_reset=False,
        )
        try:
            observation, info = environment.reset(seed=seed + index)
            recurrent_state = None
            episode_start = np.ones((1,), dtype=bool)
            terminated = False
            truncated = False
            steps = 0
            maximum_angle = 0.0
            upper_contact_steps = 0
            while not (terminated or truncated):
                action, recurrent_state = model.predict(
                    observation,
                    state=recurrent_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start[:] = False
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                maximum_angle = max(maximum_angle, float(info["orientation_error"]))
                upper_contact_steps += int(bool(info["upper_body_grounded"]))
            episodes.append(
                {
                    "position": position,
                    "steps": steps,
                    "course_complete": bool(info["course_complete"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "failure_reason": str(info["failure_reason"]),
                    "raw_clearances": int(info["raw_clearances"]),
                    "recovered_obstacles": int(info["recovered_obstacles"]),
                    "maximum_angle_degrees": math.degrees(maximum_angle),
                    "upper_body_contact_steps": upper_contact_steps,
                    "maximum_com_x": float(info["max_x_position"]),
                }
            )
        finally:
            environment.close()
    return {
        "episodes": episodes,
        "success_rate": float(np.mean([item["course_complete"] for item in episodes])),
        "clearance_rate": float(np.mean([item["raw_clearances"] >= 1 for item in episodes])),
        "hard_fall_rate": float(np.mean([item["hard_fall"] for item in episodes])),
    }


def append_progress(path: Path, row: dict[str, object]) -> None:
    """反復ごとの統一評価値をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_round_sequences(path: Path, sequences: list[Sequence]) -> None:
    """一反復の各位置軌跡を圧縮配列として個別保存する。"""
    path.mkdir(parents=True, exist_ok=True)
    for position, (observations, actions) in zip(POSITIONS, sequences):
        np.savez_compressed(
            path / f"x{position}.npz",
            observations=observations,
            actions=actions,
        )


def main() -> None:
    """循環模倣初期化と対話的示範集約を順に実行し最良学生を保存する。"""
    parser = argparse.ArgumentParser(description="训练带记忆的95维统一越墙学生。")
    parser.add_argument("--run-name", default="height1_recurrent_dagger_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-epochs", type=int, default=12)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.75, 0.5, 0.25])
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "datasets").mkdir()
    (output_dir / "checkpoints").mkdir()

    environment = GeneralObstacleEnv(
        course=make_course(20, args.seed, "recurrent_dagger_model_init"),
        resample_on_reset=False,
    )
    try:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            environment,
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
        all_sequences, collection = collect_sequences(
            None,
            seed=args.seed + 30_000,
            beta=1.0,
        )
        save_round_sequences(output_dir / "datasets" / "round_00", all_sequences)
        losses = recurrent_behavior_clone(
            model,
            all_sequences,
            epochs=args.initial_epochs,
        )
        evaluation = evaluate_student(model, seed=args.seed + 40_000)
        best_key = (
            evaluation["success_rate"],
            evaluation["clearance_rate"],
            -evaluation["hard_fall_rate"],
        )
        best_round = 0
        model.save(output_dir / "best_model")
        model.save(output_dir / "checkpoints" / "round_00")
        history = []

        def record(round_index: int, beta: float, round_collection, round_losses, result):
            """一反復の指標と詳細評価を保存する。"""
            row = {
                "round": round_index,
                "beta": beta,
                "dataset_rows": int(sum(len(item[0]) for item in all_sequences)),
                "sequence_count": len(all_sequences),
                "behavior_cloning_loss": round_losses[-1],
                "collection_success_rate": round_collection["collection_success_rate"],
                "student_success_rate": result["success_rate"],
                "student_clearance_rate": result["clearance_rate"],
                "student_hard_fall_rate": result["hard_fall_rate"],
            }
            append_progress(output_dir / "progress.csv", row)
            history.append({**row, "collection": round_collection, "evaluation": result})
            print(json.dumps(row, ensure_ascii=False), flush=True)

        record(0, 1.0, collection, losses, evaluation)
        for round_index, beta in enumerate(args.betas, start=1):
            new_sequences, collection = collect_sequences(
                model,
                seed=args.seed + 30_000 + 1_000 * round_index,
                beta=beta,
            )
            save_round_sequences(
                output_dir / "datasets" / f"round_{round_index:02d}",
                new_sequences,
            )
            all_sequences.extend(new_sequences)
            losses = recurrent_behavior_clone(
                model,
                all_sequences,
                epochs=args.update_epochs,
            )
            evaluation = evaluate_student(
                model,
                seed=args.seed + 40_000 + 100 * round_index,
            )
            model.save(output_dir / "checkpoints" / f"round_{round_index:02d}")
            key = (
                evaluation["success_rate"],
                evaluation["clearance_rate"],
                -evaluation["hard_fall_rate"],
            )
            if key > best_key:
                best_key = key
                best_round = round_index
                model.save(output_dir / "best_model")
            record(round_index, beta, collection, losses, evaluation)

        summary = {
            "method": "recurrent_interactive_dagger",
            "observation_dimension": 95,
            "action_dimension": 6,
            "privileged_student_inputs": False,
            "best_round": best_round,
            "best_key": list(best_key),
            "history": history,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
