"""閉ループ教師との対話的示範集約で九十五次元統一学生を訓練する。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from general_terrain.environment import GeneralObstacleEnv
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher
from general_terrain.terrain import build_course


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "height1_dagger_student"
POSITIONS = tuple(range(20, 31))


def make_course(position: int, seed: int, split: str):
    """指定壁位置を持つ再現可能な単一矮壁コースを返す。"""
    return build_course(
        ["low_hurdle"],
        split=split,
        seed=seed,
        difficulty=1,
        start_runway_voxels=position,
    )


def collect_dagger_round(
    model: PPO | None,
    *,
    seed: int,
    beta: float,
    positions: tuple[int, ...] = POSITIONS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """学生訪問状態で教師へ問い合わせ、実行成否と全ラベルを集める。"""
    rng = np.random.default_rng(seed)
    teacher = PortfolioHeight1Teacher()
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episodes = []
    for index, position in enumerate(positions):
        course = make_course(position, seed + index, "dagger_collection")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        try:
            observation, info = environment.reset(seed=seed + index)
            teacher.reset(environment)
            terminated = False
            truncated = False
            steps = 0
            teacher_steps = 0
            while not (terminated or truncated):
                teacher_action, _ = teacher.predict(environment, observation, info)
                observations.append(np.asarray(observation, dtype=np.float32))
                actions.append(np.asarray(teacher_action, dtype=np.float32))
                if model is None or rng.random() < beta:
                    executed_action = teacher_action
                    teacher_steps += 1
                else:
                    executed_action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(
                    executed_action
                )
                steps += 1
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
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        {
            "beta": beta,
            "rows": len(observations),
            "episodes": episodes,
            "collection_success_rate": float(
                np.mean([episode["course_complete"] for episode in episodes])
            ),
        },
    )


def supervised_update(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> float:
    """集約済み全データで学生の決定論的平均動作を回帰更新する。"""
    rng = np.random.default_rng(seed)
    parameters = list(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=3e-4, weight_decay=1e-6)
    batch_size = 256
    model.policy.train()
    for _ in range(epochs):
        permutation = rng.permutation(len(observations))
        for start in range(0, len(observations), batch_size):
            indices = permutation[start : start + batch_size]
            batch_observations = torch.as_tensor(
                observations[indices],
                dtype=torch.float32,
                device=model.device,
            )
            batch_actions = torch.as_tensor(
                actions[indices],
                dtype=torch.float32,
                device=model.device,
            )
            latent = model.policy.mlp_extractor.forward_actor(batch_observations)
            predictions = model.policy.action_net(latent)
            loss = torch.mean((predictions - batch_actions) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
    with torch.no_grad():
        sample_indices = np.arange(0, len(observations), max(1, len(observations) // 4096))
        sample_observations = torch.as_tensor(
            observations[sample_indices],
            dtype=torch.float32,
            device=model.device,
        )
        sample_actions = torch.as_tensor(
            actions[sample_indices],
            dtype=torch.float32,
            device=model.device,
        )
        latent = model.policy.mlp_extractor.forward_actor(sample_observations)
        predictions = model.policy.action_net(latent)
        model.policy.log_std.fill_(-3.0)
        return float(torch.mean((predictions - sample_actions) ** 2).cpu())


def evaluate_student(
    model: PPO,
    *,
    seed: int,
    positions: tuple[int, ...] = POSITIONS,
) -> dict[str, object]:
    """教師を使わず学生単独で全壁位置の厳格合格率を測る。"""
    episodes = []
    for index, position in enumerate(positions):
        course = make_course(position, seed + index, "dagger_validation")
        environment = GeneralObstacleEnv(course=course, resample_on_reset=False)
        try:
            observation, info = environment.reset(seed=seed + index)
            terminated = False
            truncated = False
            steps = 0
            maximum_angle = 0.0
            upper_contact_steps = 0
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
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
    """各集約反復の評価値をCSVへ追記する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """教師データ初期化後、学生訪問状態を反復集約して最良方策を保存する。"""
    parser = argparse.ArgumentParser(description="用交互式示范聚合训练统一95维学生。")
    parser.add_argument("--run-name", default="height1_dagger_seed7_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-epochs", type=int, default=30)
    parser.add_argument("--update-epochs", type=int, default=12)
    parser.add_argument(
        "--betas",
        nargs="+",
        type=float,
        default=[0.75, 0.5, 0.25, 0.0],
    )
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "datasets").mkdir()
    (output_dir / "checkpoints").mkdir()

    dummy_environment = gym.wrappers.TimeLimit(
        GeneralObstacleEnv(
            course=make_course(20, args.seed, "dagger_model_init"),
            resample_on_reset=False,
        ),
        max_episode_steps=2_000,
    )
    model = PPO(
        "MlpPolicy",
        dummy_environment,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    try:
        all_observations, all_actions, collection = collect_dagger_round(
            None,
            seed=args.seed + 1_000,
            beta=1.0,
        )
        np.savez_compressed(
            output_dir / "datasets" / "round_00.npz",
            observations=all_observations,
            actions=all_actions,
        )
        mse = supervised_update(
            model,
            all_observations,
            all_actions,
            seed=args.seed,
            epochs=args.initial_epochs,
        )
        evaluation = evaluate_student(model, seed=args.seed + 20_000)
        best_key = (
            evaluation["success_rate"],
            evaluation["clearance_rate"],
            -evaluation["hard_fall_rate"],
        )
        best_round = 0
        model.save(output_dir / "best_model")
        model.save(output_dir / "checkpoints" / "round_00")
        history = []

        def save_round(round_index: int, beta: float, round_collection, round_mse: float, result):
            """一反復分の要約をメモリとCSVへ同時保存する。"""
            row = {
                "round": round_index,
                "beta": beta,
                "environment_steps": int(len(all_observations)),
                "dataset_rows": int(len(all_observations)),
                "behavior_cloning_mse": round_mse,
                "collection_success_rate": round_collection["collection_success_rate"],
                "student_success_rate": result["success_rate"],
                "student_clearance_rate": result["clearance_rate"],
                "student_hard_fall_rate": result["hard_fall_rate"],
            }
            history.append({**row, "evaluation": result, "collection": round_collection})
            append_progress(output_dir / "progress.csv", row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

        save_round(0, 1.0, collection, mse, evaluation)
        for round_index, beta in enumerate(args.betas, start=1):
            new_observations, new_actions, collection = collect_dagger_round(
                model,
                seed=args.seed + 1_000 * (round_index + 1),
                beta=beta,
            )
            np.savez_compressed(
                output_dir / "datasets" / f"round_{round_index:02d}.npz",
                observations=new_observations,
                actions=new_actions,
            )
            all_observations = np.concatenate((all_observations, new_observations))
            all_actions = np.concatenate((all_actions, new_actions))
            mse = supervised_update(
                model,
                all_observations,
                all_actions,
                seed=args.seed + round_index,
                epochs=args.update_epochs,
            )
            evaluation = evaluate_student(
                model,
                seed=args.seed + 20_000 + 100 * round_index,
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
            save_round(round_index, beta, collection, mse, evaluation)

        summary = {
            "method": "interactive_dagger_behavior_cloning",
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
        dummy_environment.close()


if __name__ == "__main__":
    main()
