"""第一壁回復後の実状態から第二壁再起動方策を模倣とPPOで訓練する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from general_terrain.search_double_hurdle_teacher import (
    SequentialHeight1Teacher,
    course,
)
from general_terrain.search_noisy_teacher_portfolio import configurations
from general_terrain.environment import GeneralObstacleEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRST_ROWS = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "double_hurdle_search"
    / "double_hurdle_pairs_fast_seed7_v2"
    / "first_rows.json"
)
DOUBLE_BRANCHES = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "double_hurdle_search"
    / "double_hurdle_pairs_fast_seed7_v2"
    / "branches"
)
DEFAULT_INITIAL_MODEL = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "generated"
    / "noisy_height1_runs"
    / "noisy_height1_demo_seed7_v3"
    / "best_model.zip"
)
RUNS_ROOT = PROJECT_ROOT / "training_only_teacher" / "generated" / "interwall_restart_runs"
TARGET_POSITIONS = (21, 22, 25, 28)


def safe_first_configurations() -> dict[int, dict[str, object]]:
    """第一壁だけ安全回復した最初の設定を位置別に返す。"""
    rows = json.loads(FIRST_ROWS.read_text(encoding="utf-8"))
    selected: dict[int, dict[str, object]] = {}
    candidates = [item for item in configurations() if not bool(item["robust_flat"])][:18]
    for row in rows:
        position = int(row["position"])
        result = row["result"]
        if (
            position in TARGET_POSITIONS
            and position not in selected
            and int(result["recovered_obstacles"]) >= 1
            and not bool(result["hard_fall"])
        ):
            selected[position] = candidates[int(row["first_index"])]
    missing = sorted(set(TARGET_POSITIONS) - set(selected))
    if missing:
        raise RuntimeError(f"第一壁安全設定が不足している：{missing}")
    return selected


class InterWallRestartEnv(gym.Wrapper):
    """教師で第一壁回復まで進め第二壁区間だけを学習させる。"""

    def __init__(
        self,
        *,
        seed: int,
        noise_std: float,
        noise_probability: float,
    ) -> None:
        environment = GeneralObstacleEnv(
            course=course(TARGET_POSITIONS[0], seed),
            resample_on_reset=False,
        )
        super().__init__(environment)
        self.base_seed = seed
        self.episode_index = 0
        self.noise_std = noise_std
        self.noise_probability = noise_probability
        self.rng = np.random.default_rng(seed)
        self.first_configurations = safe_first_configurations()
        self.control_steps = 0
        self.previous_raw_clearances = 0
        self.previous_recovered = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """第一壁を教師で安全回復した直後の状態を作る。"""
        del options
        episode_seed = self.base_seed + self.episode_index if seed is None else seed
        position = TARGET_POSITIONS[self.episode_index % len(TARGET_POSITIONS)]
        self.episode_index += 1
        verified_prefix_seed = 91_007 + (position - 20)
        current_course = course(position, verified_prefix_seed)
        observation, info = self.env.reset(
            seed=verified_prefix_seed,
            options={"course": current_course},
        )
        first_configuration = self.first_configurations[position]
        teacher = SequentialHeight1Teacher(first_configuration, first_configuration)
        teacher.reset(self.env)
        prefix_rng = np.random.default_rng(verified_prefix_seed + 9_000_000)
        prefix_steps = 0
        while int(info["recovered_obstacles"]) < 1:
            action, _ = teacher.predict(self.env, info)
            executed_action = np.asarray(action, dtype=np.float32)
            if prefix_rng.random() < 0.01:
                executed_action = np.clip(
                    executed_action
                    + prefix_rng.normal(0.0, 0.01, size=executed_action.shape),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            observation, _, terminated, truncated, info = self.env.step(executed_action)
            prefix_steps += 1
            if terminated or truncated:
                raise RuntimeError(
                    f"第一壁教師前綴が失敗した：位置={position}、理由={info['failure_reason']}"
                )
        self.rng = np.random.default_rng(int(episode_seed) + 9_500_000)
        self.control_steps = 0
        self.previous_raw_clearances = int(info["raw_clearances"])
        self.previous_recovered = int(info["recovered_obstacles"])
        info = dict(info)
        info["prefix_steps"] = prefix_steps
        info["curriculum_position"] = position
        return observation, info

    def step(self, action):
        """第二壁の通過、回復、完走を安全優先で報酬化する。"""
        executed_action = np.asarray(action, dtype=np.float32)
        if self.rng.random() < self.noise_probability:
            executed_action = np.clip(
                executed_action
                + self.rng.normal(0.0, self.noise_std, size=executed_action.shape),
                -1.0,
                1.0,
            ).astype(np.float32)
        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        self.control_steps += 1
        raw_clearances = int(info["raw_clearances"])
        recovered = int(info["recovered_obstacles"])
        if raw_clearances > self.previous_raw_clearances:
            reward += 8.0
        if recovered > self.previous_recovered:
            reward += 40.0
        if bool(info["course_complete"]):
            reward += 100.0
        if bool(info["upper_body_grounded"]):
            reward -= 2.0
        if bool(info["hard_fall"]):
            reward -= 100.0
        self.previous_raw_clearances = raw_clearances
        self.previous_recovered = recovered
        if self.control_steps >= 900 and not terminated:
            truncated = True
        info = dict(info)
        info["interwall_success"] = bool(info["course_complete"])
        info["control_steps"] = self.control_steps
        return observation, float(reward), terminated, truncated, info


def load_second_wall_demonstrations() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """六つの双壁成功分岐から第二壁制御開始以降だけを抽出する。"""
    observations = []
    actions = []
    rows = []
    for branch_path in sorted(DOUBLE_BRANCHES.glob("x*_double_hurdle.npz")):
        data = np.load(branch_path)
        stages = np.asarray(data["stages"]).astype(str)
        second_indices = np.flatnonzero(np.char.startswith(stages, "obstacle_1:"))
        if len(second_indices) == 0:
            continue
        start = max(0, int(second_indices[0]) - 40)
        branch_observations = np.asarray(data["observations"], dtype=np.float32)[start:]
        branch_actions = np.asarray(data["actions"], dtype=np.float32)[start:]
        observations.append(branch_observations)
        actions.append(branch_actions)
        rows.append({"file": branch_path.name, "start": start, "rows": len(branch_actions)})
    if not observations:
        raise RuntimeError("第二壁成功示範が見つからない。")
    return (
        np.concatenate(observations),
        np.concatenate(actions),
        {"branches": rows, "rows": sum(item["rows"] for item in rows)},
    )


def behavior_clone(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> list[float]:
    """第二壁成功動作で方策平均だけを保守的に予熱する。"""
    rng = np.random.default_rng(seed)
    parameters = list(model.policy.mlp_extractor.policy_net.parameters())
    parameters.extend(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-5, weight_decay=1e-6)
    losses = []
    model.policy.train()
    for _ in range(epochs):
        epoch_losses = []
        permutation = rng.permutation(len(observations))
        for start in range(0, len(observations), 256):
            indices = permutation[start : start + 256]
            batch_observations = torch.as_tensor(
                observations[indices], dtype=torch.float32, device=model.device
            )
            batch_actions = torch.as_tensor(
                actions[indices], dtype=torch.float32, device=model.device
            )
            latent = model.policy.mlp_extractor.forward_actor(batch_observations)
            predictions = model.policy.action_net(latent)
            loss = torch.mean((predictions - batch_actions) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return losses


def evaluate(model: PPO, *, seed: int, noise_std: float, noise_probability: float):
    """四つの未解位置で第一壁教師後の第二壁成功率を測る。"""
    environment = InterWallRestartEnv(
        seed=seed,
        noise_std=noise_std,
        noise_probability=noise_probability,
    )
    episodes = []
    try:
        for index, position in enumerate(TARGET_POSITIONS):
            observation, info = environment.reset(seed=seed + index)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
            episodes.append(
                {
                    "position": position,
                    "course_complete": bool(info["course_complete"]),
                    "raw_clearances": int(info["raw_clearances"]),
                    "recovered_obstacles": int(info["recovered_obstacles"]),
                    "hard_fall": bool(info["hard_fall"]),
                    "failure_reason": str(info["failure_reason"]),
                    "control_steps": int(info["control_steps"]),
                    "maximum_com_x": float(info["max_x_position"]),
                }
            )
    finally:
        environment.close()
    return {
        "episodes": episodes,
        "success_count": sum(item["course_complete"] for item in episodes),
        "hard_fall_count": sum(item["hard_fall"] for item in episodes),
        "mean_recovered_obstacles": float(
            np.mean([item["recovered_obstacles"] for item in episodes])
        ),
    }


def append_progress(path: Path, row: dict[str, object]) -> None:
    """五千歩ごとの第二壁評価をCSVへ保存する。"""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """第二壁示範初期化後に四位置PPOを継続する。"""
    parser = argparse.ArgumentParser(description="训练双墙之间的再起步教师。")
    parser.add_argument("--run-name", default="interwall_restart_seed7_v1")
    parser.add_argument("--initial-model", default=str(DEFAULT_INITIAL_MODEL))
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--evaluate-every", type=int, default=5_000)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--noise-probability", type=float, default=0.01)
    args = parser.parse_args()
    output_dir = RUNS_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    environment = Monitor(
        InterWallRestartEnv(
            seed=args.seed,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        ),
        filename=str(output_dir / "training.monitor.csv"),
    )
    try:
        model = PPO.load(Path(args.initial_model), env=environment, device="cpu")
        observations, actions, dataset = load_second_wall_demonstrations()
        losses = behavior_clone(
            model,
            observations,
            actions,
            seed=args.seed,
            epochs=args.bc_epochs,
        )
        (output_dir / "initialization.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "bc_epochs": args.bc_epochs,
                    "first_loss": losses[0],
                    "final_loss": losses[-1],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        model.learning_rate = 1e-5
        model.lr_schedule = lambda _: 1e-5
        model.ent_coef = 0.001
        initial = evaluate(
            model,
            seed=args.seed + 91_000,
            noise_std=args.noise_std,
            noise_probability=args.noise_probability,
        )
        best_key = (
            initial["success_count"],
            -initial["hard_fall_count"],
            initial["mean_recovered_obstacles"],
        )
        best_step = 0
        model.save(output_dir / "best_model")
        history = [{"step": 0, **initial}]
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        completed = 0
        while completed < args.total_steps:
            chunk = min(args.evaluate_every, args.total_steps - completed)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
            completed += chunk
            result = evaluate(
                model,
                seed=args.seed + 91_000,
                noise_std=args.noise_std,
                noise_probability=args.noise_probability,
            )
            history.append({"step": completed, **result})
            model.save(output_dir / "checkpoints" / f"model_{completed}_steps")
            key = (
                result["success_count"],
                -result["hard_fall_count"],
                result["mean_recovered_obstacles"],
            )
            if key > best_key:
                best_key = key
                best_step = completed
                model.save(output_dir / "best_model")
            row = {
                "step": completed,
                "success_count": result["success_count"],
                "hard_fall_count": result["hard_fall_count"],
                "mean_recovered_obstacles": result["mean_recovered_obstacles"],
            }
            append_progress(output_dir / "evaluation_5k.csv", row)
            print(json.dumps(history[-1], ensure_ascii=False), flush=True)
            if result["success_count"] >= 3 and result["hard_fall_count"] == 0:
                break
        summary = {
            "completed_steps": completed,
            "best_step": best_step,
            "best_key": list(best_key),
            "history": history,
            "interwall_gate_passed": bool(best_key[0] >= 3 and best_key[1] == 0),
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
