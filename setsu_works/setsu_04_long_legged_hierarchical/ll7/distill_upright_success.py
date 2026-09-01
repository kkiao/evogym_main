"""PPOの成功探索軌跡を決定論的な直立低速通過方策へ蒸留する。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO

from ll7.body import make_body
from ll7.experiment import RUNS_DIR, make_upright_clearance_env, write_json
from ll7.train_upright_clearance import evaluate, score


def parse_args():
    parser = argparse.ArgumentParser(description="收集PPO成功轨迹并蒸馏为确定性策略。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--handoff-x", type=float, default=1.15)
    parser.add_argument("--max-clearance-speed", type=float, default=2.0)
    parser.add_argument("--success-episodes", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--agent-max-steps", type=int, default=350)
    parser.add_argument("--prefix-max-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def collect_successes(model, teacher_model, body, args):
    """成功した確率的エピソードだけから観測と行動を収集する。"""
    env = make_upright_clearance_env(
        body,
        level=1,
        teacher_model=teacher_model,
        handoff_x=args.handoff_x,
        agent_max_steps=args.agent_max_steps,
        prefix_max_steps=args.prefix_max_steps,
        max_clearance_speed=args.max_clearance_speed,
    )
    observations = []
    actions = []
    episode_rows = []
    rng = np.random.default_rng(args.seed)
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
            if info["upright_clearance_success"]:
                observations.extend(episode_observations)
                actions.extend(episode_actions)
                episode_rows.append(
                    {
                        "attempt": attempt,
                        "steps": step,
                        "angle_deg": float(np.degrees(info["clearance_angle"])),
                        "speed": float(info["clearance_speed"]),
                    }
                )
                print(
                    f"[collect] success={len(episode_rows)}/{args.success_episodes} "
                    f"attempt={attempt} angle={episode_rows[-1]['angle_deg']:.1f} "
                    f"speed={episode_rows[-1]['speed']:.2f}",
                    flush=True,
                )
                if len(episode_rows) >= args.success_episodes:
                    break
            rng.random()
    finally:
        env.close()
    if len(episode_rows) < args.success_episodes:
        raise RuntimeError(
            f"只收集到 {len(episode_rows)} 条成功轨迹，目标为 {args.success_episodes}。"
        )
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        episode_rows,
    )


def append_evaluation(path: Path, row: dict):
    """蒸留過程の決定論的評価をCSVへ追記する。"""
    fields = (
        "epoch",
        "loss",
        "clear_rate",
        "success_rate",
        "mean_clearance_angle_deg",
        "mean_clearance_speed",
        "mean_max_x",
    )
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
    teacher_path = Path(args.teacher_model).resolve()
    source_model = PPO.load(source_path, device="cpu")
    teacher_model = PPO.load(teacher_path, device="cpu")
    body = make_body()

    observations, actions, episode_rows = collect_successes(
        source_model,
        teacher_model,
        body,
        args,
    )
    np.savez_compressed(
        run_dir / "successful_trajectories.npz",
        observations=observations,
        actions=actions,
    )
    with (run_dir / "successful_episodes.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("attempt", "steps", "angle_deg", "speed"))
        writer.writeheader()
        writer.writerows(episode_rows)

    student = PPO.load(source_path, device="cpu")
    optimizer = torch.optim.Adam(student.policy.parameters(), lr=args.learning_rate)
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    evaluation_args = SimpleNamespace(
        eval_episodes=args.eval_episodes,
        handoff_x=args.handoff_x,
        agent_max_steps=args.agent_max_steps,
        prefix_max_steps=args.prefix_max_steps,
        max_clearance_speed=args.max_clearance_speed,
    )
    best_score = (-float("inf"),) * 7
    best_epoch = 0
    evaluation_path = run_dir / "distillation_evaluation.csv"
    sample_count = observations.shape[0]

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
            metrics = evaluate(student, teacher_model, body, evaluation_args)
            mean_loss = float(np.mean(losses))
            append_evaluation(
                evaluation_path,
                {"epoch": epoch, "loss": mean_loss, **metrics},
            )
            current_score = score(metrics, args.max_clearance_speed)
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                student.save(run_dir / "best_model.zip")
            student.save(run_dir / "latest_model.zip")
            print(
                f"[distill] epoch={epoch} loss={mean_loss:.5f} "
                f"success={metrics['success_rate']:.2f} "
                f"angle={metrics['mean_clearance_angle_deg']:.1f} "
                f"speed={metrics['mean_clearance_speed']:.2f}",
                flush=True,
            )

    write_json(
        run_dir / "summary.json",
        {
            "source_model": str(source_path),
            "teacher_model": str(teacher_path),
            "success_episodes": len(episode_rows),
            "sample_count": sample_count,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_score": list(best_score),
        },
    )


if __name__ == "__main__":
    main()

