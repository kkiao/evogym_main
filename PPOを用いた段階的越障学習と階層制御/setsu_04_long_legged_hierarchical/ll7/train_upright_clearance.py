"""教師交接後の直立完全通過だけを短区間PPOで学習する。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from ll7.body import make_body
from ll7.curriculum import COURSE_VERSION, get_course
from ll7.environment import REWARD_VERSION
from ll7.experiment import (
    RUNS_DIR,
    make_true_noroll_second_clearance_env,
    make_upright_clearance_env,
    write_json,
)
from ll7.train_curriculum import transfer_policy


FIELDS = (
    "timesteps",
    "mean_return",
    "mean_steps",
    "clear_rate",
    "success_rate",
    "mean_clearance_angle_deg",
    "mean_clearance_speed",
    "mean_max_x",
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练以直立姿态完整越过墙后缘。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--init-model")
    parser.add_argument("--handoff-x", type=float, default=1.15)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--init-log-std", type=float, default=-0.8)
    parser.add_argument("--agent-max-steps", type=int, default=350)
    parser.add_argument("--max-clearance-speed", type=float)
    parser.add_argument("--max-clearance-angle-degrees", type=float, default=35.0)
    parser.add_argument("--prefix-max-steps", type=int, default=1_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--first-clearance-model")
    parser.add_argument("--first-brake-model")
    parser.add_argument("--first-righting-model")
    parser.add_argument("--first-restart-model")
    parser.add_argument("--second-handoff-distance", type=float, default=0.25)
    return parser.parse_args()


def make_task_env(body, teacher_model, args, prefix_models=None, monitor_path=None):
    """通常前綴または無側倒第一障害物前綴の環境を選択する。"""
    if prefix_models is not None:
        return make_true_noroll_second_clearance_env(
            body,
            approach_model=teacher_model,
            first_clearance_model=prefix_models["clearance"],
            first_brake_model=prefix_models["brake"],
            first_righting_model=prefix_models["righting"],
            first_restart_model=prefix_models["restart"],
            agent_max_steps=args.agent_max_steps,
            prefix_max_steps=args.prefix_max_steps,
            handoff_distance=args.second_handoff_distance,
            max_clearance_speed=args.max_clearance_speed,
            max_clearance_angle=np.radians(args.max_clearance_angle_degrees),
            monitor_path=monitor_path,
        )
    return make_upright_clearance_env(
        body,
        level=args.level,
        teacher_model=teacher_model,
        handoff_x=args.handoff_x,
        agent_max_steps=args.agent_max_steps,
        prefix_max_steps=args.prefix_max_steps,
        max_clearance_speed=args.max_clearance_speed,
        max_clearance_angle=np.radians(args.max_clearance_angle_degrees),
        monitor_path=monitor_path,
    )


def append_row(path: Path, row: dict):
    """直立通過評価を固定列順のCSVへ追記する。"""
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in FIELDS})


def evaluate(model, teacher_model, body, args, prefix_models=None) -> dict:
    """短区間課題を複数回決定論的に評価する。"""
    env = make_task_env(body, teacher_model, args, prefix_models=prefix_models)
    rows = []
    try:
        for episode in range(args.eval_episodes):
            obs, info = env.reset(seed=10_000 + episode)
            total_reward = 0.0
            completed_steps = 0
            for completed_steps in range(1, args.agent_max_steps + 1):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            cleared = int(info["strict_clearances"]) >= 1
            rows.append(
                {
                    "return": total_reward,
                    "steps": completed_steps,
                    "cleared": float(cleared),
                    "success": float(info["upright_clearance_success"]),
                    "angle": float(info["clearance_angle"]) if cleared else np.pi,
                    "speed": float(info["clearance_speed"]) if cleared else float("inf"),
                    "max_x": float(info["max_x_position"]),
                }
            )
    finally:
        env.close()
    return {
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "clear_rate": float(np.mean([row["cleared"] for row in rows])),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_clearance_angle_deg": float(
            np.degrees(np.mean([row["angle"] for row in rows]))
        ),
        "mean_clearance_speed": float(np.mean([row["speed"] for row in rows])),
        "mean_max_x": float(np.mean([row["max_x"] for row in rows])),
    }


def score(
    metrics: dict,
    max_clearance_speed: float | None = None,
    max_clearance_angle_degrees: float = 35.0,
) -> tuple[float, ...]:
    """直立成功、通過、角度、距離、回報の順で比較値を返す。"""
    angle = float(metrics["mean_clearance_angle_deg"])
    speed = float(metrics["mean_clearance_speed"])
    if max_clearance_speed is not None:
        violation = max(0.0, angle - max_clearance_angle_degrees) + 20.0 * max(
            0.0,
            speed - max_clearance_speed,
        )
    else:
        violation = max(0.0, angle - max_clearance_angle_degrees)
    return (
        float(metrics["success_rate"]),
        float(metrics["clear_rate"]),
        -violation,
        -angle,
        -speed,
        float(metrics["mean_max_x"]),
        float(metrics["mean_return"]),
    )


class EvaluationCallback(BaseCallback):
    """5k刻みで直立通過角を評価して最良モデルを保護する。"""

    def __init__(
        self,
        args,
        teacher_model,
        body,
        run_dir,
        evaluation_path,
        best_score,
        prefix_models=None,
    ):
        super().__init__(verbose=0)
        self.args = args
        self.teacher_model = teacher_model
        self.body = body
        self.run_dir = run_dir
        self.evaluation_path = evaluation_path
        self.best_score = best_score
        self.prefix_models = prefix_models
        self.best_timesteps = 0
        self.next_eval = args.eval_interval
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval:
            metrics = evaluate(
                self.model,
                self.teacher_model,
                self.body,
                self.args,
                prefix_models=self.prefix_models,
            )
            append_row(
                self.evaluation_path,
                {"timesteps": int(self.num_timesteps), **metrics},
            )
            self.model.save(
                self.checkpoints_dir / f"model_{self.num_timesteps}_steps.zip"
            )
            self.model.save(self.run_dir / "latest_model.zip")
            current_score = score(
                metrics,
                self.args.max_clearance_speed,
                self.args.max_clearance_angle_degrees,
            )
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_timesteps = int(self.num_timesteps)
                self.model.save(self.run_dir / "best_model.zip")
            print(
                f"[upright-clear] step={self.num_timesteps} "
                f"clear={metrics['clear_rate']:.2f} "
                f"upright={metrics['success_rate']:.2f} "
                f"angle={metrics['mean_clearance_angle_deg']:.1f}deg "
                f"speed={metrics['mean_clearance_speed']:.2f}",
                flush=True,
            )
            self.next_eval += self.args.eval_interval
        return True


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    teacher_path = Path(args.teacher_model).resolve()
    teacher_model = PPO.load(teacher_path, device="cpu")
    init_path = Path(args.init_model).resolve() if args.init_model else teacher_path
    prefix_paths = {
        "clearance": args.first_clearance_model,
        "brake": args.first_brake_model,
        "righting": args.first_righting_model,
        "restart": args.first_restart_model,
    }
    provided_prefix = [value is not None for value in prefix_paths.values()]
    if any(provided_prefix) and not all(provided_prefix):
        raise ValueError("第二障害物課程には第一障害物の四方策が全て必要です。")
    resolved_prefix_paths = (
        {key: Path(value).resolve() for key, value in prefix_paths.items()}
        if all(provided_prefix)
        else None
    )
    prefix_models = (
        {
            key: PPO.load(path, device="cpu")
            for key, path in resolved_prefix_paths.items()
        }
        if resolved_prefix_paths is not None
        else None
    )
    body = make_body()
    run_dir = RUNS_DIR / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"实验目录已有文件：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = run_dir / "upright_clearance_evaluation_5k.csv"

    env = make_task_env(
        body,
        teacher_model,
        args,
        prefix_models=prefix_models,
        monitor_path=run_dir / "training",
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=args.ent_coef,
        policy_kwargs={
            "activation_fn": torch.nn.Tanh,
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        },
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    transfer_policy(init_path, model)
    with torch.no_grad():
        model.policy.log_std.fill_(args.init_log_std)

    initial = evaluate(
        model,
        teacher_model,
        body,
        args,
        prefix_models=prefix_models,
    )
    append_row(evaluation_path, {"timesteps": 0, **initial})
    model.save(run_dir / "initial_model.zip")
    model.save(run_dir / "best_model.zip")
    best_score = score(
        initial,
        args.max_clearance_speed,
        args.max_clearance_angle_degrees,
    )
    np.save(run_dir / "body.npy", body)
    write_json(run_dir / "course.json", get_course(args.level).as_dict())
    write_json(
        run_dir / "config.json",
        {
            "algorithm": "PPO upright-clearance sub-curriculum",
            "level": args.level,
            "course_version": COURSE_VERSION,
            "reward_version": REWARD_VERSION,
            "teacher_model": str(teacher_path),
            "init_model": str(init_path),
            "first_prefix_models": (
                {key: str(path) for key, path in resolved_prefix_paths.items()}
                if resolved_prefix_paths is not None
                else None
            ),
            "handoff_x": args.handoff_x,
            "target_timesteps": args.total_timesteps,
            "agent_max_steps": args.agent_max_steps,
            "max_clearance_speed": args.max_clearance_speed,
            "max_clearance_angle_degrees": args.max_clearance_angle_degrees,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "ent_coef": args.ent_coef,
            "init_log_std": args.init_log_std,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    callback = EvaluationCallback(
        args,
        teacher_model,
        body,
        run_dir,
        evaluation_path,
        best_score,
        prefix_models=prefix_models,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    model.save(run_dir / "latest_model.zip")
    write_json(
        run_dir / "summary.json",
        {
            "completed_timesteps": int(model.num_timesteps),
            "best_timesteps": callback.best_timesteps,
            "best_score": list(callback.best_score),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    env.close()
    print(f"直立越障训练完成：{run_dir}", flush=True)


if __name__ == "__main__":
    main()
