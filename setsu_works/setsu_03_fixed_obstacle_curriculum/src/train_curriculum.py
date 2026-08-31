"""一種類の形状を一段階の固定障害物カリキュラムでPPO学習する。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.bodies import BODY_NAMES, make_body
from src.curriculum import CURRICULUM_LEVELS, COURSE_VERSION, get_course
from src.environment import REWARD_VERSION
from src.experiment import (
    RUNS_DIR,
    append_evaluation,
    evaluate_ppo,
    evaluate_random,
    make_env,
    read_evaluations,
    score_metrics,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="使用PPO训练一个固定障碍课程阶段。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--body-name", choices=BODY_NAMES, required=True)
    parser.add_argument("--level", type=int, choices=CURRICULUM_LEVELS, required=True)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-model")
    parser.add_argument("--init-log-std", type=float)
    parser.add_argument("--rehearsal-level", type=int, choices=CURRICULUM_LEVELS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--network-width", type=int, choices=(64, 128), default=64)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--verbose", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def validate_args(args):
    """ファイル作成前に数値設定と実験名を検証する。"""
    positive = (
        "total_timesteps",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "eval_interval",
        "eval_episodes",
        "torch_threads",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于0。")
    if args.n_steps % args.batch_size:
        raise ValueError("--n-steps 必须能被 --batch-size 整除。")
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("--gamma 和 --gae-lambda 必须位于(0, 1]。")
    if args.rehearsal_level is not None and args.rehearsal_level >= args.level:
        raise ValueError("--rehearsal-level 必须低于当前 --level。")
    run_path = Path(args.run_name)
    if run_path.name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name 必须是单个目录名。")


class EvaluationCallback(BaseCallback):
    """5k刻みで評価・チェックポイント・最良モデルを同時に保存する。"""

    def __init__(
        self,
        args,
        body,
        course,
        run_dir,
        evaluation_path,
        initial_timesteps,
        best_score,
        best_timesteps,
    ):
        super().__init__(verbose=0)
        self.args = args
        self.body = body
        self.course = course
        self.run_dir = run_dir
        self.evaluation_path = evaluation_path
        self.next_eval = (
            initial_timesteps // args.eval_interval + 1
        ) * args.eval_interval
        self.best_score = best_score
        self.best_timesteps = best_timesteps
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def _save_and_evaluate(self):
        metrics = evaluate_ppo(
            self.model,
            self.body,
            self.args.level,
            self.args.eval_episodes,
            self.course.max_steps,
            self.args.eval_seed,
        )
        row = {"timesteps": int(self.num_timesteps), **metrics}
        append_evaluation(self.evaluation_path, row)
        self.model.save(
            self.checkpoints_dir / f"model_{self.num_timesteps}_steps.zip"
        )
        self.model.save(self.run_dir / "latest_model.zip")
        current_score = score_metrics(metrics)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_timesteps = int(self.num_timesteps)
            self.model.save(self.run_dir / "best_model.zip")
        print(
            f"[{self.args.body_name} L{self.args.level}] step={self.num_timesteps} "
            f"return={metrics['mean_return']:.4f} "
            f"max_x={metrics['mean_max_x']:.4f} "
            f"cleared={metrics['mean_obstacles_cleared']:.2f} "
            f"success={metrics['success_rate']:.2f}",
            flush=True,
        )

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval:
            self._save_and_evaluate()
            self.next_eval += self.args.eval_interval
        return True


def transfer_policy(source_path: Path, target_model):
    """同一形状の前段階モデルから方策パラメータだけを安全に継承する。"""
    source = PPO.load(source_path, device="cpu")
    if source.action_space.shape != target_model.action_space.shape:
        raise ValueError("初始化模型与目标身体的动作维度不一致。")
    if source.observation_space.shape != target_model.observation_space.shape:
        raise ValueError("初始化模型与目标身体的观测维度不一致。")
    target_model.policy.load_state_dict(source.policy.state_dict())


def best_state(evaluation_path: Path):
    """既存評価から最良スコアと対応ステップを復元する。"""
    rows = read_evaluations(evaluation_path)
    if not rows:
        return (-float("inf"),) * 4, 0
    best = max(rows, key=score_metrics)
    return score_metrics(best), int(best["timesteps"])


def main():
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    course = get_course(args.level)
    body = make_body(args.body_name)
    run_dir = RUNS_DIR / args.run_name
    evaluation_path = run_dir / "ppo_evaluation_5k.csv"
    latest_path = run_dir / "latest_model.zip"

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"找不到续训模型：{latest_path}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"实验目录已有文件：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.rehearsal_level is None:
        train_env = make_env(
            body,
            args.level,
            course.max_steps,
            monitor_path=run_dir / "training",
            monitor_override=not args.resume,
        )
    else:
        rehearsal_course = get_course(args.rehearsal_level)

        def make_level_env(level, max_steps, monitor_name):
            """DummyVecEnv用に遅延生成される一つの監視付き環境を返す。"""
            return lambda: make_env(
                body,
                level,
                max_steps,
                monitor_path=run_dir / monitor_name,
                monitor_override=not args.resume,
            )

        train_env = DummyVecEnv(
            [
                make_level_env(
                    args.rehearsal_level,
                    rehearsal_course.max_steps,
                    f"training_level{args.rehearsal_level}",
                ),
                make_level_env(
                    args.level,
                    course.max_steps,
                    f"training_level{args.level}",
                ),
            ]
        )
    train_env.action_space.seed(args.seed)

    if args.resume:
        model = PPO.load(latest_path, env=train_env, device="cpu")
        completed_timesteps = int(model.num_timesteps)
        current_best_score, best_timesteps = best_state(evaluation_path)
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
            ent_coef=args.ent_coef,
            policy_kwargs={
                "activation_fn": torch.nn.Tanh,
                "net_arch": {
                    "pi": [args.network_width, args.network_width],
                    "vf": [args.network_width, args.network_width],
                },
            },
            seed=args.seed,
            device="cpu",
            verbose=args.verbose,
        )
        if args.init_model:
            transfer_policy(Path(args.init_model).resolve(), model)
            if args.init_log_std is not None:
                with torch.no_grad():
                    model.policy.log_std.fill_(args.init_log_std)
        completed_timesteps = 0
        initial = evaluate_ppo(
            model,
            body,
            args.level,
            args.eval_episodes,
            course.max_steps,
            args.eval_seed,
        )
        random = evaluate_random(
            body,
            args.level,
            args.eval_episodes,
            course.max_steps,
            args.eval_seed + 20_000,
        )
        append_evaluation(evaluation_path, {"timesteps": 0, **initial})
        model.save(run_dir / "initial_model.zip")
        model.save(run_dir / "best_model.zip")
        model.save(latest_path)
        current_best_score = score_metrics(initial)
        best_timesteps = 0
        write_json(run_dir / "baselines.json", {"initial": initial, "random": random})
        np.save(run_dir / "body.npy", body)
        write_json(run_dir / "course.json", course.as_dict())
        write_json(
            run_dir / "config.json",
            {
                "algorithm": "PPO",
                "course_version": COURSE_VERSION,
                "reward_version": REWARD_VERSION,
                "body_name": args.body_name,
                "curriculum_level": args.level,
                "run_name": args.run_name,
                "seed": args.seed,
                "target_timesteps": args.total_timesteps,
                "max_steps": course.max_steps,
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "ent_coef": args.ent_coef,
                "network_width": args.network_width,
                "eval_interval": args.eval_interval,
                "eval_episodes": args.eval_episodes,
                "rehearsal_level": args.rehearsal_level,
                "init_model": str(Path(args.init_model).resolve()) if args.init_model else None,
                "init_log_std": args.init_log_std,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    if completed_timesteps >= args.total_timesteps:
        print("已达到目标步数，无需继续训练。")
        train_env.close()
        return

    callback = EvaluationCallback(
        args,
        body,
        course,
        run_dir,
        evaluation_path,
        completed_timesteps,
        current_best_score,
        best_timesteps,
    )
    remaining = args.total_timesteps - completed_timesteps
    model.learn(
        total_timesteps=remaining,
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    model.save(latest_path)
    final_timesteps = int(model.num_timesteps)
    existing_steps = {row["timesteps"] for row in read_evaluations(evaluation_path)}
    if final_timesteps not in existing_steps:
        final_metrics = evaluate_ppo(
            model,
            body,
            args.level,
            args.eval_episodes,
            course.max_steps,
            args.eval_seed,
        )
        append_evaluation(
            evaluation_path,
            {"timesteps": final_timesteps, **final_metrics},
        )
        if score_metrics(final_metrics) > callback.best_score:
            callback.best_score = score_metrics(final_metrics)
            callback.best_timesteps = final_timesteps
            model.save(run_dir / "best_model.zip")

    write_json(
        run_dir / "summary.json",
        {
            "run_name": args.run_name,
            "body_name": args.body_name,
            "curriculum_level": args.level,
            "completed_timesteps": final_timesteps,
            "target_timesteps": args.total_timesteps,
            "best_score": list(callback.best_score),
            "best_timesteps": callback.best_timesteps,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    train_env.close()
    print(f"训练完成：{run_dir}", flush=True)


if __name__ == "__main__":
    main()
