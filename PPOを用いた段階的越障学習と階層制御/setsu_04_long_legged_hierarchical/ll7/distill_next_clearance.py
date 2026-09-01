"""次障害物を通過した確率的軌跡を決定論的方策へ蒸留する。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from evogym import get_full_connectivity
from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.environment import LongLeggedCurriculumEnv
from ll7.experiment import (
    ActionRescaleWrapper,
    RUNS_DIR,
    ValidatedFourStagePrefixWrapper,
    write_json,
)


class StrictClearanceGoalWrapper(gym.Wrapper):
    """指定数の物理的完全通過で短い蒸留試行を終了する。"""

    def __init__(self, env, target_clearances: int):
        super().__init__(env)
        self.target_clearances = target_clearances

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reached = int(info["strict_clearances"]) >= self.target_clearances
        info = dict(info)
        info["clearance_subgoal_success"] = reached
        if reached:
            reward += 100.0
            terminated = True
        return obs, reward, terminated, truncated, info


def parse_args():
    parser = argparse.ArgumentParser(description="蒸馏下一堵墙的成功随机轨迹。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--reuse-run")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--clearance-model", required=True)
    parser.add_argument("--landing-model", required=True)
    parser.add_argument("--restart-model", required=True)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--prefix-validated", type=int, default=1)
    parser.add_argument("--target-clearances", type=int, default=2)
    parser.add_argument("--max-angle-degrees", type=float, default=65.0)
    parser.add_argument("--max-clearance-speed", type=float, default=4.5)
    parser.add_argument("--handoff-distance", type=float, default=0.25)
    parser.add_argument("--success-episodes", type=int, default=20)
    parser.add_argument("--tail-steps", type=int, default=250)
    parser.add_argument("--max-attempts", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--agent-max-steps", type=int, default=1_500)
    parser.add_argument("--prefix-max-steps", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def make_env(body, models, args):
    """教師前綴と物理通過だけの短い終了条件を持つ環境を作る。"""
    approach, clearance, landing, restart = models
    env = LongLeggedCurriculumEnv(
        body=body,
        level=args.level,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(env)
    env = ValidatedFourStagePrefixWrapper(
        env,
        approach_model=approach,
        clearance_model=clearance,
        landing_model=landing,
        restart_model=restart,
        target_validated=args.prefix_validated,
        handoff_distance=args.handoff_distance,
        max_prefix_steps=args.prefix_max_steps,
    )
    env = StrictClearanceGoalWrapper(env, args.target_clearances)
    return gym.wrappers.TimeLimit(env, max_episode_steps=args.agent_max_steps)


def collect_successes(model, models, body, args):
    """成功した確率的試行だけから観測と行動を蓄積する。"""
    env = make_env(body, models, args)
    observations = []
    actions = []
    rows = []
    try:
        for attempt in range(1, args.max_attempts + 1):
            obs, info = env.reset(seed=args.seed + attempt)
            episode_observations = []
            episode_actions = []
            for step in range(1, args.agent_max_steps + 1):
                episode_observations.append(np.asarray(obs, dtype=np.float32).copy())
                action, _ = model.predict(obs, deterministic=False)
                action = np.asarray(action, dtype=np.float32)
                episode_actions.append(action.copy())
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            angle_deg = float(np.degrees(info["orientation_error"]))
            speed = float(info["com_speed"])
            quality_success = (
                info.get("clearance_subgoal_success", False)
                and angle_deg <= args.max_angle_degrees
                and speed <= args.max_clearance_speed
            )
            if quality_success:
                # 長い待機区間を除き、実際の通過直前動作だけを教師にする。
                observations.extend(episode_observations[-args.tail_steps :])
                actions.extend(episode_actions[-args.tail_steps :])
                rows.append(
                    {
                        "attempt": attempt,
                        "steps": step,
                        "angle_deg": angle_deg,
                        "speed": speed,
                        "max_x": float(info["max_x_position"]),
                    }
                )
                print(
                    f"[collect-next] success={len(rows)}/{args.success_episodes} "
                    f"attempt={attempt} angle={rows[-1]['angle_deg']:.1f} "
                    f"speed={rows[-1]['speed']:.2f}",
                    flush=True,
                )
                if len(rows) >= args.success_episodes:
                    break
    finally:
        env.close()
    if len(rows) < args.success_episodes:
        raise RuntimeError(
            f"只收集到 {len(rows)} 条成功轨迹，目标为 {args.success_episodes}。"
        )
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        rows,
    )


def load_success_tails(run_dir: Path, tail_steps: int, episode_index: int | None):
    """保存済み全軌跡から各成功試行の末尾だけを再利用する。"""
    data = np.load(run_dir / "successful_trajectories.npz")
    with (run_dir / "successful_episodes.csv").open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    observations = []
    actions = []
    offset = 0
    selected_rows = []
    for index, row in enumerate(rows, start=1):
        steps = int(row["steps"])
        start = offset + max(0, steps - tail_steps)
        end = offset + steps
        if episode_index is None or index == episode_index:
            observations.append(data["observations"][start:end])
            actions.append(data["actions"][start:end])
            selected_rows.append(row)
        offset = end
    if offset != data["observations"].shape[0]:
        raise RuntimeError("保存轨迹的回合长度与样本数不一致。")
    if not observations:
        raise ValueError("--episode-index 超出保存轨迹数量。")
    return np.concatenate(observations), np.concatenate(actions), selected_rows


def evaluate(model, models, body, args):
    """蒸留方策の決定論的な次障害物通過率を測る。"""
    env = make_env(body, models, args)
    rows = []
    try:
        for episode in range(args.eval_episodes):
            obs, info = env.reset(seed=10_000 + episode)
            for _ in range(args.agent_max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            quality_success = (
                info.get("clearance_subgoal_success", False)
                and float(np.degrees(info["orientation_error"]))
                <= args.max_angle_degrees
                and float(info["com_speed"]) <= args.max_clearance_speed
            )
            rows.append(
                {
                    "success": float(quality_success),
                    "angle_deg": float(np.degrees(info["orientation_error"])),
                    "speed": float(info["com_speed"]),
                    "max_x": float(info["max_x_position"]),
                }
            )
    finally:
        env.close()
    return {
        "clear_rate": float(np.mean([row["success"] for row in rows])),
        "mean_angle_deg": float(np.mean([row["angle_deg"] for row in rows])),
        "mean_speed": float(np.mean([row["speed"] for row in rows])),
        "mean_max_x": float(np.mean([row["max_x"] for row in rows])),
    }


def append_evaluation(path: Path, row: dict):
    """蒸留評価を固定列のCSVへ追記する。"""
    fields = ("epoch", "loss", "clear_rate", "mean_angle_deg", "mean_speed", "mean_max_x")
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fields})


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = RUNS_DIR / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"实验目录已有文件：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source_model).resolve()
    model_paths = tuple(
        Path(path).resolve()
        for path in (
            args.approach_model,
            args.clearance_model,
            args.landing_model,
            args.restart_model,
        )
    )
    source = PPO.load(source_path, device="cpu")
    models = tuple(PPO.load(path, device="cpu") for path in model_paths)
    body = make_body()

    if args.reuse_run:
        reuse_dir = Path(args.reuse_run)
        if not reuse_dir.is_absolute():
            reuse_dir = RUNS_DIR / reuse_dir
        observations, actions, episode_rows = load_success_tails(
            reuse_dir,
            args.tail_steps,
            args.episode_index,
        )
    else:
        observations, actions, episode_rows = collect_successes(source, models, body, args)
    np.savez_compressed(
        run_dir / "successful_trajectories.npz",
        observations=observations,
        actions=actions,
    )
    with (run_dir / "successful_episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ("attempt", "steps", "angle_deg", "speed", "max_x")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episode_rows)

    student = PPO.load(source_path, device="cpu")
    optimizer = torch.optim.Adam(student.policy.parameters(), lr=args.learning_rate)
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    sample_count = observations.shape[0]
    best_score = (-float("inf"),) * 3
    best_epoch = 0
    evaluation_path = run_dir / "distillation_evaluation.csv"

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(sample_count)
        losses = []
        student.policy.set_training_mode(True)
        for start in range(0, sample_count, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            distribution = student.policy.get_distribution(obs_tensor[indices])
            mean_actions = distribution.distribution.mean
            loss = torch.nn.functional.mse_loss(mean_actions, action_tensor[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            student.policy.set_training_mode(False)
            metrics = evaluate(student, models, body, args)
            mean_loss = float(np.mean(losses))
            append_evaluation(
                evaluation_path,
                {"epoch": epoch, "loss": mean_loss, **metrics},
            )
            score = (
                metrics["clear_rate"],
                metrics["mean_max_x"],
                -metrics["mean_angle_deg"],
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                student.save(run_dir / "best_model.zip")
            student.save(run_dir / "latest_model.zip")
            print(
                f"[distill-next] epoch={epoch} loss={mean_loss:.5f} "
                f"clear={metrics['clear_rate']:.2f} "
                f"angle={metrics['mean_angle_deg']:.1f}",
                flush=True,
            )

    write_json(
        run_dir / "summary.json",
        {
            "source_model": str(source_path),
            "teacher_models": [str(path) for path in model_paths],
            "success_episodes": len(episode_rows),
            "tail_steps": args.tail_steps,
            "sample_count": sample_count,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_score": list(best_score),
        },
    )


if __name__ == "__main__":
    main()
