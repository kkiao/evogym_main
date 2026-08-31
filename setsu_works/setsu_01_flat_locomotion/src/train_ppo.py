"""Stable-Baselines3 PPOで固定形状のEvoGym Walkerを学習する。"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import evogym.envs  # noqa: F401 - EvoGym環境を登録するために読み込む
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

from src.body import (
    make_layered_walker_body,
    make_soft_walker_body,
    make_walker_body,
)
from src.reinforce import evaluate_random_actions, summarize_rollouts
from src.wrappers import NormalizeActionSpace


TASK_NAME = "Walker-v0"
PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs"
EVALUATION_FIELDS = [
    "timesteps",
    "mean_return",
    "std_return",
    "min_return",
    "max_return",
    "mean_steps",
    "mean_speed",
]


def parse_args():
    parser = argparse.ArgumentParser(description="用 PPO 训练固定 EvoGym Walker。")
    parser.add_argument("--run-name", help="runs 下的实验目录名；默认含随机种子。")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--body-name",
        choices=["walker", "layered_walker"],
        default="walker",
        help="选择原始固定身体或黑/橙/蓝三层的第二种身体。",
    )
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="累计目标环境步数。")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=2048, help="每次 PPO 更新前收集的步数。")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5_000,
        help="确定性曲线评估间隔；GIF/模型检查点仍由 --checkpoint-interval 控制。",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=25_000,
        help="模型检查点间隔；默认每 25k 保存一次，供 GIF 使用。",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


def validate_args(args):
    positive_names = (
        "total_timesteps",
        "max_steps",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "eval_interval",
        "eval_episodes",
        "checkpoint_interval",
        "max_grad_norm",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于 0。")
    if args.n_steps % args.batch_size != 0:
        raise ValueError("--n-steps 必须能被 --batch-size 整除。")
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("--gamma 和 --gae-lambda 必须位于 (0, 1]。")
    if args.vf_coef < 0 or args.ent_coef < 0:
        raise ValueError("--vf-coef 和 --ent-coef 不能为负。")
    if not 0 < args.clip_range < 1:
        raise ValueError("--clip-range 必须位于 (0, 1)。")


def safe_run_name(run_name):
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name 必须是单个非空目录名，不能包含路径。")
    return run_name


def make_selected_body(body_name):
    if body_name == "walker":
        return make_walker_body(), "src.body.make_walker_body"
    if body_name == "soft_walker":
        # 保存済みの旧実験との互換性を保つが、新規学習用の選択肢には公開しない。
        return make_soft_walker_body(), "src.body.make_soft_walker_body"
    if body_name == "layered_walker":
        return make_layered_walker_body(), "src.body.make_layered_walker_body"
    raise ValueError(f"未知身体：{body_name}")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path, data):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_model_atomic(model, path):
    temporary_path = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary_path)
    temporary_path.replace(path)


def ensure_evaluation_csv(path):
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=EVALUATION_FIELDS).writeheader()


def append_evaluation(path, timesteps, metrics):
    row = {"timesteps": timesteps}
    row.update({name: metrics[name] for name in EVALUATION_FIELDS if name != "timesteps"})
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=EVALUATION_FIELDS).writerow(row)


def make_env(body, max_steps, monitor_path=None, append_monitor=False):
    env = gym.make(TASK_NAME, body=body, render_mode=None)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = NormalizeActionSpace(env)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            allow_early_resets=True,
            override_existing=not append_monitor,
        )
    return env


def evaluate_ppo(model, body, episodes, max_steps, seed):
    env = make_env(body, max_steps)
    returns = []
    steps_list = []
    try:
        for episode_number in range(episodes):
            obs, _ = env.reset(seed=seed + episode_number)
            episode_return = 0.0
            steps = 0
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            returns.append(episode_return)
            steps_list.append(steps)
    finally:
        env.close()
    return summarize_rollouts(returns, steps_list)


class PPOExperimentCallback(BaseCallback):
    def __init__(
        self,
        run_dir,
        body,
        max_steps,
        eval_interval,
        eval_episodes,
        eval_seed,
        checkpoint_interval,
        best_eval_return,
        best_timesteps,
        initial_timesteps,
        target_timesteps,
    ):
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.body = body
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.checkpoint_interval = checkpoint_interval
        self.best_eval_return = best_eval_return
        self.best_timesteps = best_timesteps
        self.target_timesteps = target_timesteps
        self.last_eval_timesteps = initial_timesteps
        self.next_eval = (initial_timesteps // eval_interval + 1) * eval_interval
        self.next_checkpoint = (
            initial_timesteps // checkpoint_interval + 1
        ) * checkpoint_interval
        self.evaluation_path = run_dir / "ppo_evaluation.csv"
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def _save_summary(self):
        write_json_atomic(
            self.run_dir / "summary.json",
            {
                "run_name": self.run_dir.name,
                "algorithm": "PPO",
                "completed_timesteps": self.num_timesteps,
                "target_timesteps": self.target_timesteps,
                "best_eval_return": self.best_eval_return,
                "best_timesteps": self.best_timesteps,
                "updated_at_utc": utc_now(),
            },
        )

    def _evaluate(self):
        metrics = evaluate_ppo(
            self.model,
            self.body,
            self.eval_episodes,
            self.max_steps,
            self.eval_seed,
        )
        append_evaluation(self.evaluation_path, self.num_timesteps, metrics)
        self.last_eval_timesteps = self.num_timesteps
        if metrics["mean_return"] > self.best_eval_return:
            self.best_eval_return = metrics["mean_return"]
            self.best_timesteps = self.num_timesteps
            save_model_atomic(self.model, self.run_dir / "best_model.zip")
            print(f"  PPO 保存新最佳模型：return={self.best_eval_return:.6f}")
        print(
            f"  PPO 评估 step={self.num_timesteps}: "
            f"return={metrics['mean_return']:.6f}, speed={metrics['mean_speed']:.8f}"
        )
        self._save_summary()

    def _on_step(self):
        if self.num_timesteps >= self.next_checkpoint:
            save_model_atomic(
                self.model,
                self.checkpoints_dir / f"model_{self.num_timesteps}_steps.zip",
            )
            # 既定の再開用モデルも更新し、予期せぬ中断後の手動コピーを不要にする。
            save_model_atomic(self.model, self.run_dir / "latest_model.zip")
            self._save_summary()
            self.next_checkpoint += self.checkpoint_interval
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
            self.next_eval += self.eval_interval
        return True

    def _on_training_end(self):
        if self.last_eval_timesteps != self.num_timesteps:
            self._evaluate()
        save_model_atomic(self.model, self.run_dir / "latest_model.zip")
        self._save_summary()


def load_resume_config(args, config_path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fixed_names = (
        "seed",
        "max_steps",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "vf_coef",
        "ent_coef",
        "clip_range",
        "max_grad_norm",
        "eval_interval",
        "eval_episodes",
        "eval_seed",
        "checkpoint_interval",
    )
    for name in fixed_names:
        setattr(args, name, config[name])
    args.body_name = config.get("body_name", "walker")
    config["target_timesteps"] = args.total_timesteps
    config["last_resumed_at_utc"] = utc_now()
    write_json_atomic(config_path, config)
    return config


def main():
    args = parse_args()
    args.run_name = safe_run_name(
        args.run_name or f"ppo_{args.body_name}_seed{args.seed}"
    )
    validate_args(args)

    RUNS_DIR.mkdir(exist_ok=True)
    run_dir = RUNS_DIR / args.run_name
    config_path = run_dir / "config.json"
    latest_path = run_dir / "latest_model.zip"
    evaluation_path = run_dir / "ppo_evaluation.csv"
    body_path = run_dir / "body.npy"

    if args.resume:
        if not config_path.exists() or not latest_path.exists():
            raise FileNotFoundError(f"实验 {run_dir} 没有可续训的 PPO 模型。")
        config = load_resume_config(args, config_path)
        validate_args(args)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"实验目录 {run_dir} 已有文件。请换 --run-name，或使用 --resume。"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": 1,
            "algorithm": "PPO",
            "task_name": TASK_NAME,
            "body_name": args.body_name,
            "body_source": make_selected_body(args.body_name)[1],
            "action_space": "normalized_-1_to_1",
            "run_name": args.run_name,
            "target_timesteps": args.total_timesteps,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "vf_coef": args.vf_coef,
            "ent_coef": args.ent_coef,
            "clip_range": args.clip_range,
            "max_grad_norm": args.max_grad_norm,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
            "checkpoint_interval": args.checkpoint_interval,
            "created_at_utc": utc_now(),
        }
        write_json_atomic(config_path, config)

    body, _ = make_selected_body(args.body_name)
    if args.resume:
        if not np.array_equal(np.load(body_path), body):
            raise ValueError("保存的身体与当前固定 Walker 身体不同。")
    else:
        np.save(body_path, body)
    ensure_evaluation_csv(evaluation_path)

    monitor_path = run_dir / "training.monitor.csv"
    train_env = make_env(
        body,
        args.max_steps,
        monitor_path,
        append_monitor=args.resume,
    )
    train_env.reset(seed=args.seed)
    train_env.action_space.seed(args.seed)

    if args.resume:
        model = PPO.load(latest_path, env=train_env, device="cpu")
        completed_timesteps = int(model.num_timesteps)
        rows = list(csv.DictReader(evaluation_path.open(encoding="utf-8")))
        if rows:
            best_row = max(rows, key=lambda row: float(row["mean_return"]))
            best_eval_return = float(best_row["mean_return"])
            best_timesteps = int(best_row["timesteps"])
        else:
            best_eval_return = -float("inf")
            best_timesteps = completed_timesteps
        print(f"PPO 从 {completed_timesteps} 步续训：{latest_path}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs={
                "activation_fn": torch.nn.Tanh,
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            },
            seed=args.seed,
            device="cpu",
            verbose=args.verbose,
        )
        completed_timesteps = 0
        initial_metrics = evaluate_ppo(
            model, body, args.eval_episodes, args.max_steps, args.eval_seed
        )
        random_metrics = evaluate_random_actions(
            TASK_NAME,
            body,
            args.eval_episodes,
            args.max_steps,
            args.eval_seed + 10000,
        )
        append_evaluation(evaluation_path, 0, initial_metrics)
        write_json_atomic(
            run_dir / "baselines.json",
            {
                "untrained_policy": initial_metrics,
                "random_actions": random_metrics,
            },
        )
        save_model_atomic(model, run_dir / "initial_model.zip")
        save_model_atomic(model, run_dir / "best_model.zip")
        save_model_atomic(model, latest_path)
        best_eval_return = initial_metrics["mean_return"]
        best_timesteps = 0
        print(
            f"PPO 训练前：return={initial_metrics['mean_return']:.6f}; "
            f"随机基线={random_metrics['mean_return']:.6f}"
        )

    if args.total_timesteps <= completed_timesteps:
        train_env.close()
        print(f"已完成 {completed_timesteps} 步，不少于目标；无需训练。")
        return

    callback = PPOExperimentCallback(
        run_dir=run_dir,
        body=body,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        checkpoint_interval=args.checkpoint_interval,
        best_eval_return=best_eval_return,
        best_timesteps=best_timesteps,
        initial_timesteps=completed_timesteps,
        target_timesteps=args.total_timesteps,
    )
    remaining_timesteps = args.total_timesteps - completed_timesteps
    print(
        f"开始 PPO 实验 {args.run_name}：{completed_timesteps} -> "
        f"{args.total_timesteps} steps"
    )
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            progress_bar=False,
        )
    finally:
        train_env.close()
    print(f"PPO 训练完成。结果目录：{run_dir}")


if __name__ == "__main__":
    main()
