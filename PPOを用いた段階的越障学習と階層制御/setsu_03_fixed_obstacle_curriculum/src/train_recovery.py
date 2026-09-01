"""教師越障後の姿勢から回復し、次の固定障害物へ進むPPO方策を学習する。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.bodies import BODY_NAMES, make_body
from src.curriculum import CURRICULUM_LEVELS, COURSE_VERSION, get_course
from src.environment import REWARD_VERSION
from src.experiment import (
    RUNS_DIR,
    append_evaluation,
    make_teacher_prefix_env,
    score_metrics,
    summarize_episodes,
    write_json,
)
from src.train_curriculum import transfer_policy


def parse_args():
    parser = argparse.ArgumentParser(description="训练越墙后的恢复和下一障碍策略。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--body-name", choices=BODY_NAMES, required=True)
    parser.add_argument("--level", type=int, choices=CURRICULUM_LEVELS, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--target-cleared", type=int, default=1)
    parser.add_argument("--post-prefix-steps", type=int, default=0)
    parser.add_argument("--approach-model", type=Path)
    parser.add_argument("--approach-distance", type=float)
    parser.add_argument("--approach-max-steps", type=int, default=1_600)
    parser.add_argument("--goal-distance", type=float)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--agent-max-steps", type=int, default=1_200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--n-steps", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--stochastic-eval-episodes", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def evaluate_recovery(
    model,
    body,
    level,
    teacher,
    target_cleared,
    post_prefix_steps,
    goal_distance,
    episodes,
    max_steps,
    seed,
    deterministic=True,
    approach_model=None,
    approach_distance=None,
    approach_max_steps=1_600,
):
    """教師前置区間後の方策を指定された行動選択方式で評価する。"""
    results = []
    env = make_teacher_prefix_env(
        body,
        level,
        teacher,
        target_cleared,
        max_steps,
        post_prefix_steps=post_prefix_steps,
        approach_model=approach_model,
        approach_distance=approach_distance,
        approach_max_steps=approach_max_steps,
        recovery_goal_distance=goal_distance,
    )
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            total_reward = 0.0
            for steps in range(1, max_steps + 1):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            results.append(
                {
                    "return": total_reward,
                    "steps": steps,
                    "displacement": info["forward_displacement"],
                    "final_x": info["x_position"],
                    "max_x": info["max_x_position"],
                    "obstacles_cleared": info["obstacles_cleared"],
                    "obstacle_fraction": info["obstacle_fraction"],
                    "success": float(info["is_success"]),
                }
            )
    finally:
        env.close()
    return summarize_episodes(results)


class RecoveryEvaluationCallback(BaseCallback):
    """回復方策を5k刻みで評価し、技能を拡張した最良モデルを保存する。"""

    def __init__(
        self,
        args,
        body,
        teacher,
        approach,
        run_dir,
        evaluation_path,
        stochastic_evaluation_path,
        best_score,
    ):
        super().__init__(verbose=0)
        self.args = args
        self.body = body
        self.teacher = teacher
        self.approach = approach
        self.run_dir = run_dir
        self.evaluation_path = evaluation_path
        self.stochastic_evaluation_path = stochastic_evaluation_path
        self.best_score = best_score
        self.best_timesteps = 0
        self.next_eval = args.eval_interval
        self.checkpoints = run_dir / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval:
            metrics = evaluate_recovery(
                self.model,
                self.body,
                self.args.level,
                self.teacher,
                self.args.target_cleared,
                self.args.post_prefix_steps,
                self.args.goal_distance,
                self.args.eval_episodes,
                self.args.agent_max_steps,
                10_000,
                approach_model=self.approach,
                approach_distance=self.args.approach_distance,
                approach_max_steps=self.args.approach_max_steps,
            )
            append_evaluation(
                self.evaluation_path,
                {"timesteps": int(self.num_timesteps), **metrics},
            )
            stochastic_metrics = evaluate_recovery(
                self.model,
                self.body,
                self.args.level,
                self.teacher,
                self.args.target_cleared,
                self.args.post_prefix_steps,
                self.args.goal_distance,
                self.args.stochastic_eval_episodes,
                self.args.agent_max_steps,
                20_000,
                deterministic=False,
                approach_model=self.approach,
                approach_distance=self.args.approach_distance,
                approach_max_steps=self.args.approach_max_steps,
            )
            append_evaluation(
                self.stochastic_evaluation_path,
                {"timesteps": int(self.num_timesteps), **stochastic_metrics},
            )
            self.model.save(self.checkpoints / f"model_{self.num_timesteps}_steps.zip")
            self.model.save(self.run_dir / "latest_model.zip")
            current_score = score_metrics(stochastic_metrics)
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_timesteps = int(self.num_timesteps)
                self.model.save(self.run_dir / "best_model.zip")
            print(
                f"[{self.args.body_name} recovery L{self.args.level}] "
                f"step={self.num_timesteps} return={metrics['mean_return']:.4f} "
                f"max_x={metrics['mean_max_x']:.4f} "
                f"cleared={metrics['mean_obstacles_cleared']:.2f} "
                f"det_success={metrics['success_rate']:.2f} "
                f"stoch_success={stochastic_metrics['success_rate']:.2f}",
                flush=True,
            )
            self.next_eval += self.args.eval_interval
        return True


def main():
    args = parse_args()
    if args.level <= 0 or args.target_cleared <= 0:
        raise ValueError("恢复训练需要至少一个已由教师完成的障碍。")
    if args.n_steps % args.batch_size:
        raise ValueError("--n-steps 必须能被 --batch-size 整除。")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    body = make_body(args.body_name)
    course = get_course(args.level)
    teacher = PPO.load(args.teacher_model, device="cpu")
    approach = PPO.load(args.approach_model, device="cpu") if args.approach_model else None
    run_dir = RUNS_DIR / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"实验目录已有文件：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = run_dir / "ppo_evaluation_5k.csv"
    stochastic_evaluation_path = run_dir / "ppo_evaluation_stochastic_5k.csv"
    train_env = make_teacher_prefix_env(
        body,
        args.level,
        teacher,
        args.target_cleared,
        args.agent_max_steps,
        post_prefix_steps=args.post_prefix_steps,
        approach_model=approach,
        approach_distance=args.approach_distance,
        approach_max_steps=args.approach_max_steps,
        recovery_goal_distance=args.goal_distance,
        monitor_path=run_dir / "training",
    )

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
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        },
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    if args.init_model is not None:
        transfer_policy(args.init_model.resolve(), model)

    initial = evaluate_recovery(
        model,
        body,
        args.level,
        teacher,
        args.target_cleared,
        args.post_prefix_steps,
        args.goal_distance,
        args.eval_episodes,
        args.agent_max_steps,
        10_000,
        approach_model=approach,
        approach_distance=args.approach_distance,
        approach_max_steps=args.approach_max_steps,
    )
    append_evaluation(evaluation_path, {"timesteps": 0, **initial})
    initial_stochastic = evaluate_recovery(
        model,
        body,
        args.level,
        teacher,
        args.target_cleared,
        args.post_prefix_steps,
        args.goal_distance,
        args.stochastic_eval_episodes,
        args.agent_max_steps,
        20_000,
        deterministic=False,
        approach_model=approach,
        approach_distance=args.approach_distance,
        approach_max_steps=args.approach_max_steps,
    )
    append_evaluation(
        stochastic_evaluation_path,
        {"timesteps": 0, **initial_stochastic},
    )
    model.save(run_dir / "initial_model.zip")
    model.save(run_dir / "best_model.zip")
    np.save(run_dir / "body.npy", body)
    write_json(run_dir / "course.json", course.as_dict())
    write_json(
        run_dir / "config.json",
        {
            "algorithm": "PPO_teacher_prefix_recovery",
            "course_version": COURSE_VERSION,
            "reward_version": REWARD_VERSION,
            "body_name": args.body_name,
            "curriculum_level": args.level,
            "teacher_model": str(args.teacher_model.resolve()),
            "target_cleared": args.target_cleared,
            "post_prefix_steps": args.post_prefix_steps,
            "approach_model": str(args.approach_model.resolve()) if args.approach_model else None,
            "approach_distance": args.approach_distance,
            "approach_max_steps": args.approach_max_steps,
            "recovery_goal_distance": args.goal_distance,
            "init_model": str(args.init_model.resolve()) if args.init_model else None,
            "target_timesteps": args.total_timesteps,
            "agent_max_steps": args.agent_max_steps,
            "seed": args.seed,
            "deterministic_eval_episodes": args.eval_episodes,
            "stochastic_eval_episodes": args.stochastic_eval_episodes,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    callback = RecoveryEvaluationCallback(
        args,
        body,
        teacher,
        approach,
        run_dir,
        evaluation_path,
        stochastic_evaluation_path,
        score_metrics(initial_stochastic),
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        reset_num_timesteps=True,
        progress_bar=False,
    )
    model.save(run_dir / "latest_model.zip")
    write_json(
        run_dir / "summary.json",
        {
            "run_name": args.run_name,
            "completed_timesteps": int(model.num_timesteps),
            "best_timesteps": callback.best_timesteps,
            "best_score": list(callback.best_score),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    train_env.close()
    print(f"恢复训练完成：{run_dir}", flush=True)


if __name__ == "__main__":
    main()
