"""固定Walker形状で再開可能かつ再現可能なREINFORCE方策を学習する。"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import evogym.envs  # noqa: F401 - EvoGym環境を登録するために読み込む
import gymnasium as gym
import numpy as np
import torch

from src.body import make_walker_body
from src.policy import Policy
from src.reinforce import (
    calculate_returns,
    choose_action,
    evaluate_policy,
    evaluate_random_actions,
    update_policy,
)


TASK_NAME = "Walker-v0"
PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs"

TRAINING_FIELDS = [
    "episode",
    "total_steps",
    "episode_return",
    "steps",
    "mean_speed",
    "exploration_std",
]
UPDATE_FIELDS = [
    "completed_episode",
    "total_steps",
    "batch_episodes",
    "mean_batch_return",
    "loss",
]
EVALUATION_FIELDS = [
    "completed_episode",
    "total_steps",
    "mean_return",
    "std_return",
    "min_return",
    "max_return",
    "mean_steps",
    "mean_speed",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="在固定 EvoGym Walker 身体上训练 REINFORCE 策略。"
    )
    parser.add_argument("--episodes", type=int, default=500, help="目标累计训练回合数。")
    parser.add_argument("--seed", type=int, default=7, help="训练随机种子。")
    parser.add_argument("--run-name", help="runs 下的实验目录名；默认含随机种子。")
    parser.add_argument("--resume", action="store_true", help="从已有实验的 latest_checkpoint.pt 续训。")
    parser.add_argument("--batch-episodes", type=int, default=5, help="每次策略更新包含的回合数。")
    parser.add_argument("--max-steps", type=int, default=500, help="每回合最大仿真步数。")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子。")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam 学习率。")
    parser.add_argument("--eval-interval", type=int, default=50, help="每隔多少训练回合评估一次。")
    parser.add_argument("--eval-episodes", type=int, default=5, help="每次确定性评估的回合数。")
    parser.add_argument("--eval-seed", type=int, default=1000, help="评估的起始随机种子。")
    parser.add_argument("--exploration-std-start", type=float, default=0.30, help="初始探索标准差。")
    parser.add_argument("--exploration-std-end", type=float, default=0.05, help="最终探索标准差。")
    parser.add_argument(
        "--exploration-decay-episodes",
        type=int,
        default=1000,
        help="探索标准差线性下降所需回合数。",
    )
    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError("--episodes 必须大于 0。")
    if args.batch_episodes <= 0 or args.max_steps <= 0:
        raise ValueError("--batch-episodes 和 --max-steps 必须大于 0。")
    if args.eval_interval <= 0 or args.eval_episodes <= 0:
        raise ValueError("--eval-interval 和 --eval-episodes 必须大于 0。")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma 必须位于 (0, 1]。")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate 必须大于 0。")
    if args.exploration_std_start <= 0 or args.exploration_std_end <= 0:
        raise ValueError("探索标准差必须大于 0。")
    if args.exploration_decay_episodes <= 0:
        raise ValueError("--exploration-decay-episodes 必须大于 0。")


def safe_run_name(run_name):
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name 必须是单个非空目录名，不能包含路径。")
    return run_name


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def ensure_csv(path, fieldnames):
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()


def append_csv(path, fieldnames, rows):
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(rows)


def save_torch_atomic(data, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary_path)
    temporary_path.replace(path)


def checkpoint_data(
    policy,
    optimizer,
    completed_episodes,
    total_steps,
    best_eval_return,
    best_completed_episode,
    observation_size,
    action_size,
):
    return {
        "schema_version": 1,
        "completed_episodes": completed_episodes,
        "total_steps": total_steps,
        "best_eval_return": best_eval_return,
        "best_completed_episode": best_completed_episode,
        "observation_size": observation_size,
        "action_size": action_size,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])


def exploration_std(episode_index, start, end, decay_episodes):
    progress = min(max(episode_index / decay_episodes, 0.0), 1.0)
    return start + (end - start) * progress


def evaluation_row(completed_episodes, total_steps, metrics):
    return {
        "completed_episode": completed_episodes,
        "total_steps": total_steps,
        "mean_return": metrics["mean_return"],
        "std_return": metrics["std_return"],
        "min_return": metrics["min_return"],
        "max_return": metrics["max_return"],
        "mean_steps": metrics["mean_steps"],
        "mean_speed": metrics["mean_speed"],
    }


def load_resume_config(args, config_path):
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    # 再開時は学習の意味を変える設定を引き継ぎ、--episodesだけを新しい累積目標とする。
    fixed_names = [
        "seed",
        "batch_episodes",
        "max_steps",
        "gamma",
        "learning_rate",
        "eval_interval",
        "eval_episodes",
        "eval_seed",
        "exploration_std_start",
        "exploration_std_end",
        "exploration_decay_episodes",
    ]
    for name in fixed_names:
        setattr(args, name, saved_config[name])
    return saved_config


def main():
    args = parse_args()
    args.run_name = safe_run_name(args.run_name or f"walker_fixed_seed{args.seed}")
    validate_args(args)

    RUNS_DIR.mkdir(exist_ok=True)
    run_dir = RUNS_DIR / args.run_name
    config_path = run_dir / "config.json"
    latest_path = run_dir / "latest_checkpoint.pt"
    best_checkpoint_path = run_dir / "best_checkpoint.pt"
    best_policy_path = run_dir / "best_policy.pt"
    training_path = run_dir / "training.csv"
    updates_path = run_dir / "updates.csv"
    evaluation_path = run_dir / "evaluation.csv"

    if args.resume:
        if not run_dir.is_dir() or not config_path.exists() or not latest_path.exists():
            raise FileNotFoundError(f"实验 {run_dir} 没有可续训的配置和检查点。")
        config = load_resume_config(args, config_path)
        validate_args(args)
        config["target_episodes"] = args.episodes
        config["last_resumed_at_utc"] = utc_now()
        write_json(config_path, config)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"实验目录 {run_dir} 已有文件。请换 --run-name，或使用 --resume。"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": 1,
            "algorithm": "REINFORCE",
            "task_name": TASK_NAME,
            "body_source": "src.body.make_walker_body",
            "run_name": args.run_name,
            "target_episodes": args.episodes,
            "seed": args.seed,
            "batch_episodes": args.batch_episodes,
            "max_steps": args.max_steps,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
            "random_baseline_seed": args.eval_seed + 10000,
            "exploration_std_start": args.exploration_std_start,
            "exploration_std_end": args.exploration_std_end,
            "exploration_decay_episodes": args.exploration_decay_episodes,
            "created_at_utc": utc_now(),
        }
        write_json(config_path, config)

    body = make_walker_body()
    body_path = run_dir / "body.npy"
    if args.resume:
        saved_body = np.load(body_path)
        if not np.array_equal(saved_body, body):
            raise ValueError("保存的身体与当前固定 Walker 身体不同，拒绝混合续训。")
    else:
        np.save(body_path, body)

    ensure_csv(training_path, TRAINING_FIELDS)
    ensure_csv(updates_path, UPDATE_FIELDS)
    ensure_csv(evaluation_path, EVALUATION_FIELDS)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(TASK_NAME, body=body, render_mode=None)
    first_obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    observation_size = len(first_obs)
    action_size = env.action_space.shape[0]
    policy = Policy(observation_size, action_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    if args.resume:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != 1:
            raise ValueError("无法识别检查点版本。")
        if checkpoint["observation_size"] != observation_size or checkpoint["action_size"] != action_size:
            raise ValueError("检查点的观测或动作维度与当前环境不一致。")
        policy.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_episodes = int(checkpoint["completed_episodes"])
        total_steps = int(checkpoint["total_steps"])
        best_eval_return = float(checkpoint["best_eval_return"])
        best_completed_episode = int(checkpoint["best_completed_episode"])
        restore_rng(checkpoint)
        print(f"已从第 {completed_episodes} 回合续训：{latest_path}")
    else:
        completed_episodes = 0
        total_steps = 0
        initial_metrics = evaluate_policy(
            TASK_NAME,
            body,
            policy,
            args.eval_episodes,
            args.max_steps,
            args.eval_seed,
        )
        random_metrics = evaluate_random_actions(
            TASK_NAME,
            body,
            args.eval_episodes,
            args.max_steps,
            args.eval_seed + 10000,
        )
        write_json(
            run_dir / "baselines.json",
            {
                "untrained_policy": initial_metrics,
                "random_actions": random_metrics,
            },
        )
        best_eval_return = initial_metrics["mean_return"]
        best_completed_episode = 0
        append_csv(
            evaluation_path,
            EVALUATION_FIELDS,
            [evaluation_row(0, 0, initial_metrics)],
        )
        save_torch_atomic(policy.state_dict(), run_dir / "initial_policy.pt")
        save_torch_atomic(policy.state_dict(), best_policy_path)
        initial_checkpoint = checkpoint_data(
            policy,
            optimizer,
            0,
            0,
            best_eval_return,
            best_completed_episode,
            observation_size,
            action_size,
        )
        save_torch_atomic(initial_checkpoint, latest_path)
        save_torch_atomic(initial_checkpoint, best_checkpoint_path)
        print(
            f"训练前评估：return={initial_metrics['mean_return']:.6f}, "
            f"speed={initial_metrics['mean_speed']:.8f}"
        )
        print(
            f"随机动作基线：return={random_metrics['mean_return']:.6f}, "
            f"speed={random_metrics['mean_speed']:.8f}"
        )

    if args.episodes <= completed_episodes:
        env.close()
        print(
            f"当前已完成 {completed_episodes} 回合，不少于目标 {args.episodes}；无需训练。"
        )
        return

    print(
        f"开始实验 {args.run_name}：第 {completed_episodes + 1}--{args.episodes} 回合，"
        f"batch={args.batch_episodes}, max_steps={args.max_steps}"
    )

    try:
        while completed_episodes < args.episodes:
            current_batch_size = min(
                args.batch_episodes,
                args.episodes - completed_episodes,
            )
            batch_log_probs = []
            batch_returns_tensors = []
            batch_training_rows = []
            batch_episode_returns = []

            for batch_index in range(current_batch_size):
                episode_index = completed_episodes + batch_index
                std = exploration_std(
                    episode_index,
                    args.exploration_std_start,
                    args.exploration_std_end,
                    args.exploration_decay_episodes,
                )
                obs, _ = env.reset(seed=args.seed + episode_index)
                rewards = []
                episode_log_probs = []
                episode_return = 0.0
                steps = 0

                for _ in range(args.max_steps):
                    action, log_prob = choose_action(
                        policy,
                        obs,
                        env.action_space.low,
                        env.action_space.high,
                        std,
                    )
                    obs, reward, terminated, truncated, _ = env.step(action)
                    rewards.append(float(reward))
                    episode_log_probs.append(log_prob)
                    episode_return += float(reward)
                    steps += 1
                    if terminated or truncated:
                        break

                batch_log_probs.extend(episode_log_probs)
                batch_returns_tensors.append(calculate_returns(rewards, args.gamma))
                batch_episode_returns.append(episode_return)
                batch_training_rows.append(
                    {
                        "episode": episode_index + 1,
                        "total_steps": total_steps + steps,
                        "episode_return": episode_return,
                        "steps": steps,
                        "mean_speed": episode_return / max(steps, 1),
                        "exploration_std": std,
                    }
                )
                total_steps += steps

            loss = update_policy(
                optimizer,
                batch_log_probs,
                torch.cat(batch_returns_tensors),
            )
            completed_episodes += current_batch_size
            append_csv(training_path, TRAINING_FIELDS, batch_training_rows)
            append_csv(
                updates_path,
                UPDATE_FIELDS,
                [
                    {
                        "completed_episode": completed_episodes,
                        "total_steps": total_steps,
                        "batch_episodes": current_batch_size,
                        "mean_batch_return": float(np.mean(batch_episode_returns)),
                        "loss": loss,
                    }
                ],
            )

            should_evaluate = (
                completed_episodes % args.eval_interval == 0
                or completed_episodes == args.episodes
            )
            if should_evaluate:
                metrics = evaluate_policy(
                    TASK_NAME,
                    body,
                    policy,
                    args.eval_episodes,
                    args.max_steps,
                    args.eval_seed,
                )
                append_csv(
                    evaluation_path,
                    EVALUATION_FIELDS,
                    [evaluation_row(completed_episodes, total_steps, metrics)],
                )
                if metrics["mean_return"] > best_eval_return:
                    best_eval_return = metrics["mean_return"]
                    best_completed_episode = completed_episodes
                    save_torch_atomic(policy.state_dict(), best_policy_path)
                    best_data = checkpoint_data(
                        policy,
                        optimizer,
                        completed_episodes,
                        total_steps,
                        best_eval_return,
                        best_completed_episode,
                        observation_size,
                        action_size,
                    )
                    save_torch_atomic(best_data, best_checkpoint_path)
                    print(f"  保存新最佳策略：return={best_eval_return:.6f}")
                print(
                    f"  确定性评估：return={metrics['mean_return']:.6f}, "
                    f"speed={metrics['mean_speed']:.8f}"
                )

            latest_data = checkpoint_data(
                policy,
                optimizer,
                completed_episodes,
                total_steps,
                best_eval_return,
                best_completed_episode,
                observation_size,
                action_size,
            )
            save_torch_atomic(latest_data, latest_path)
            write_json(
                run_dir / "summary.json",
                {
                    "run_name": args.run_name,
                    "completed_episodes": completed_episodes,
                    "target_episodes": args.episodes,
                    "total_steps": total_steps,
                    "best_eval_return": best_eval_return,
                    "best_completed_episode": best_completed_episode,
                    "updated_at_utc": utc_now(),
                },
            )
            print(
                f"回合 {completed_episodes}/{args.episodes}："
                f"batch_return={np.mean(batch_episode_returns):.6f}, loss={loss:.6f}"
            )
    finally:
        env.close()

    print(f"训练完成。结果目录：{run_dir}")


if __name__ == "__main__":
    main()
