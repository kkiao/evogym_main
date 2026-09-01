"""二種類の固定形状ソフトロボットをPPOで個別に学習し、連続混合障害物を通過させる。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

from src.bodies import BODY_NAMES, make_body
from src.course import (
    COURSE_VERSION,
    GROUNDED_ROBOT_START_Y,
    LEGACY_ROBOT_START_Y,
    course_metadata,
)
from src.environment import (
    DENSE_CROSSING_ENVIRONMENT_VERSION,
    ENVIRONMENT_VERSIONS,
    LEGACY_ENVIRONMENT_VERSION,
    MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
    MOVABLE_OBSTACLE_ENVIRONMENT_VERSION,
    SHAPED_ENVIRONMENT_VERSION,
    TRANSFER_ENVIRONMENT_VERSION,
)
from src.experiment import (
    EVALUATION_FIELDS,
    RUNS_DIR,
    append_evaluation,
    ensure_evaluation_csv,
    evaluate_ppo,
    evaluate_random_actions,
    make_env,
    safe_run_name,
    save_model_atomic,
    utc_now,
    write_json_atomic,
)


def parse_args():
    parser = argparse.ArgumentParser(description="用 PPO 训练软体机器人通过混合障碍路线。")
    parser.add_argument("--run-name", help="runs 下的独立实验目录名。")
    parser.add_argument("--body-name", choices=BODY_NAMES, default="original")
    parser.add_argument(
        "--environment-version",
        choices=ENVIRONMENT_VERSIONS,
        default=MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
        help="正式实验使用地面起步和局部塑形；旧100k基线仍可复现。",
    )
    parser.add_argument(
        "--init-model",
        help="可选：从同一身体的旧PPO模型复制兼容权重；原文件只读。",
    )
    parser.add_argument(
        "--network-width",
        type=int,
        choices=(64, 128),
        default=64,
        help="迁移时应与来源模型隐藏层宽度一致。",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--max-steps", type=int, default=1_200)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5_000,
        help="每5k进行一次确定性评估并写入表格。",
    )
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5_000,
        help="每5k保存模型，训练后才能从所有评估点挑选9个关键GIF。",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def validate_args(args) -> None:
    positive = (
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
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于0。")
    if args.n_steps % args.batch_size:
        raise ValueError("--n-steps 必须能被 --batch-size 整除。")
    if args.eval_interval % args.checkpoint_interval and args.checkpoint_interval % args.eval_interval:
        raise ValueError("评估与检查点间隔必须互为整数倍，避免评估点没有对应模型。")
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        raise ValueError("--gamma 和 --gae-lambda 必须位于 (0, 1]。")


class ObstacleExperimentCallback(BaseCallback):
    def __init__(
        self,
        run_dir,
        body,
        args,
        initial_timesteps,
        best_score,
        best_timesteps,
    ):
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.body = body
        self.args = args
        self.target_timesteps = args.total_timesteps
        self.best_score = best_score
        self.best_timesteps = best_timesteps
        self.last_eval_timesteps = initial_timesteps
        self.next_eval = (initial_timesteps // args.eval_interval + 1) * args.eval_interval
        self.next_checkpoint = (
            initial_timesteps // args.checkpoint_interval + 1
        ) * args.checkpoint_interval
        self.evaluation_path = run_dir / "ppo_evaluation_5k.csv"
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    @staticmethod
    def score(metrics: dict) -> tuple:
        # 完了率と通過数を優先し、その後に最遠位置と報酬を比較する。
        return (
            metrics["success_rate"],
            metrics["mean_obstacles_cleared"],
            metrics["mean_max_x"],
            metrics["mean_return"],
        )

    def _save_summary(self) -> None:
        write_json_atomic(
            self.run_dir / "summary.json",
            {
                "run_name": self.run_dir.name,
                "algorithm": "PPO",
                "course_version": COURSE_VERSION,
                "environment_version": self.args.environment_version,
                "completed_timesteps": self.num_timesteps,
                "target_timesteps": self.target_timesteps,
                "best_score": list(self.best_score),
                "best_timesteps": self.best_timesteps,
                "updated_at_utc": utc_now(),
            },
        )

    def _evaluate(self) -> None:
        metrics = evaluate_ppo(
            self.model,
            self.body,
            self.args.eval_episodes,
            self.args.max_steps,
            self.args.eval_seed,
            self.args.environment_version,
        )
        append_evaluation(self.evaluation_path, self.num_timesteps, metrics)
        self.last_eval_timesteps = self.num_timesteps
        score = self.score(metrics)
        if score > self.best_score:
            self.best_score = score
            self.best_timesteps = self.num_timesteps
            save_model_atomic(self.model, self.run_dir / "best_model.zip")
            print(
                f"  新最佳：step={self.num_timesteps}, "
                f"obstacles={metrics['mean_obstacles_cleared']:.2f}, "
                f"max_x={metrics['mean_max_x']:.3f}, success={metrics['success_rate']:.1%}"
            )
        else:
            print(
                f"  评估：step={self.num_timesteps}, "
                f"obstacles={metrics['mean_obstacles_cleared']:.2f}, "
                f"max_x={metrics['mean_max_x']:.3f}, return={metrics['mean_return']:.3f}"
            )
        self._save_summary()

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_checkpoint:
            save_model_atomic(
                self.model,
                self.checkpoints_dir / f"model_{self.num_timesteps}_steps.zip",
            )
            save_model_atomic(self.model, self.run_dir / "latest_model.zip")
            self._save_summary()
            self.next_checkpoint += self.args.checkpoint_interval
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
            self.next_eval += self.args.eval_interval
        return True

    def _on_training_end(self) -> None:
        if self.last_eval_timesteps != self.num_timesteps:
            self._evaluate()
        save_model_atomic(self.model, self.run_dir / "latest_model.zip")
        self._save_summary()


def read_best_state(evaluation_path):
    rows = list(csv.DictReader(evaluation_path.open(encoding="utf-8")))
    if not rows:
        return (-float("inf"),) * 4, 0
    best = max(
        rows,
        key=lambda row: (
            float(row["success_rate"]),
            float(row["mean_obstacles_cleared"]),
            float(row["mean_max_x"]),
            float(row["mean_return"]),
        ),
    )
    score = (
        float(best["success_rate"]),
        float(best["mean_obstacles_cleared"]),
        float(best["mean_max_x"]),
        float(best["mean_return"]),
    )
    return score, int(best["timesteps"])


def transfer_policy_weights(source_model_path, target_model, report_path):
    source_path = source_model_path.resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"找不到初始化模型：{source_path}")
    source_model = PPO.load(source_path, device="cpu")
    if source_model.action_space.shape != target_model.action_space.shape:
        raise ValueError(
            "初始化模型动作维度与当前身体不一致："
            f"{source_model.action_space.shape} != {target_model.action_space.shape}"
        )

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    exact_keys = []
    expanded_input_keys = []
    skipped_keys = []
    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            skipped_keys.append(key)
            continue
        if source_tensor.shape == target_tensor.shape:
            target_state[key] = source_tensor.detach().clone()
            exact_keys.append(key)
            continue
        if (
            source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and source_tensor.shape[0] == target_tensor.shape[0]
            and source_tensor.shape[1] < target_tensor.shape[1]
        ):
            expanded = torch.zeros_like(target_tensor)
            expanded[:, : source_tensor.shape[1]] = source_tensor
            target_state[key] = expanded
            expanded_input_keys.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue
        skipped_keys.append(key)
    target_model.policy.load_state_dict(target_state)
    report = {
        "source_model": str(source_path),
        "source_observation_shape": list(source_model.observation_space.shape),
        "target_observation_shape": list(target_model.observation_space.shape),
        "action_shape": list(target_model.action_space.shape),
        "exact_tensor_count": len(exact_keys),
        "expanded_input_tensors": expanded_input_keys,
        "skipped_tensors": skipped_keys,
        "note": "旧模型只读；新增地形输入列初始化为0。",
    }
    write_json_atomic(report_path, report)
    return report


def main():
    args = parse_args()
    validate_args(args)
    run_name = safe_run_name(args.run_name or f"ppo_obstacles_{args.body_name}_seed{args.seed}")
    run_dir = RUNS_DIR / run_name
    config_path = run_dir / "config.json"
    latest_path = run_dir / "latest_model.zip"
    evaluation_path = run_dir / "ppo_evaluation_5k.csv"
    body_path = run_dir / "body.npy"
    body = make_body(args.body_name)

    if args.resume:
        if not config_path.exists() or not latest_path.exists() or not body_path.exists():
            raise FileNotFoundError(f"实验 {run_dir} 没有完整续训文件。")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for name in (
            "body_name",
            "environment_version",
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
        ):
            if name == "environment_version":
                setattr(args, name, config.get(name, LEGACY_ENVIRONMENT_VERSION))
            else:
                setattr(args, name, config[name])
        validate_args(args)
        body = make_body(args.body_name)
        if not np.array_equal(np.load(body_path), body):
            raise ValueError("保存身体与当前固定身体定义不一致。")
        config["target_timesteps"] = args.total_timesteps
        config["last_resumed_at_utc"] = utc_now()
        write_json_atomic(config_path, config)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"实验目录 {run_dir} 已有文件；请更名或使用 --resume。")
        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(body_path, body)
        config = {
            "schema_version": 1,
            "algorithm": "PPO",
            "course_version": COURSE_VERSION,
            "environment_version": args.environment_version,
            "initialization_model": str(args.init_model) if args.init_model else None,
            "network_width": args.network_width,
            "body_name": args.body_name,
            "run_name": run_name,
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
        robot_start_y = (
            GROUNDED_ROBOT_START_Y
            if args.environment_version != LEGACY_ENVIRONMENT_VERSION
            else LEGACY_ROBOT_START_Y
        )
        write_json_atomic(run_dir / "course.json", course_metadata(robot_start_y))

    ensure_evaluation_csv(evaluation_path)
    train_env = make_env(
        body,
        args.max_steps,
        monitor_path=run_dir / "training.monitor.csv",
        append_monitor=args.resume,
        environment_version=args.environment_version,
    )
    train_env.reset(seed=args.seed)
    train_env.action_space.seed(args.seed)

    if args.resume:
        model = PPO.load(latest_path, env=train_env, device="cpu")
        completed_timesteps = int(model.num_timesteps)
        best_score, best_timesteps = read_best_state(evaluation_path)
        print(f"从 {completed_timesteps} 步续训：{run_dir}")
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
            report = transfer_policy_weights(
                Path(args.init_model),
                model,
                run_dir / "transfer_report.json",
            )
            print(
                "已复制旧行走策略权重："
                f"exact={report['exact_tensor_count']}, "
                f"expanded_inputs={len(report['expanded_input_tensors'])}"
            )
        completed_timesteps = 0
        initial = evaluate_ppo(
            model,
            body,
            args.eval_episodes,
            args.max_steps,
            args.eval_seed,
            args.environment_version,
        )
        random = evaluate_random_actions(
            body,
            args.eval_episodes,
            args.max_steps,
            args.eval_seed + 20_000,
            args.environment_version,
        )
        append_evaluation(evaluation_path, 0, initial)
        write_json_atomic(
            run_dir / "baselines.json",
            {"untrained_policy": initial, "random_actions": random},
        )
        save_model_atomic(model, run_dir / "initial_model.zip")
        save_model_atomic(model, run_dir / "best_model.zip")
        save_model_atomic(model, latest_path)
        best_score = ObstacleExperimentCallback.score(initial)
        best_timesteps = 0
        print(
            f"训练前：obstacles={initial['mean_obstacles_cleared']:.2f}, "
            f"max_x={initial['mean_max_x']:.3f}; "
            f"随机动作 obstacles={random['mean_obstacles_cleared']:.2f}"
        )

    if args.total_timesteps <= completed_timesteps:
        train_env.close()
        print(f"已有 {completed_timesteps} 步，不少于目标，无需续训。")
        return

    callback = ObstacleExperimentCallback(
        run_dir,
        body,
        args,
        completed_timesteps,
        best_score,
        best_timesteps,
    )
    remaining = args.total_timesteps - completed_timesteps
    print(f"开始训练 {run_name}：{completed_timesteps} -> {args.total_timesteps} steps")
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            reset_num_timesteps=not args.resume,
            progress_bar=False,
        )
    finally:
        train_env.close()
    print(f"训练阶段完成：{run_dir}")


if __name__ == "__main__":
    main()
