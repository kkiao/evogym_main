"""第一障害物の途中交接から胴体非接地の完全通過と着地を共同学習する。"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from ll7.body import make_body
from ll7.curriculum import COURSE_VERSION, get_course
from ll7.environment import REWARD_VERSION
from ll7.experiment import RUNS_DIR, make_first_early_landing_env, write_json
from ll7.train_curriculum import transfer_policy


FIELDS = (
    "timesteps",
    "mean_return",
    "mean_steps",
    "mean_strict_clearances",
    "mean_stable_landings",
    "mean_maximum_degrees",
    "mean_body_contact_steps",
    "success_rate",
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练第一墙提前交接的真正无侧倒落地。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--clearance-model", required=True)
    parser.add_argument("--init-model", required=True)
    parser.add_argument("--prefix-fraction", type=float, default=0.5)
    parser.add_argument("--max-orientation-degrees", type=float, default=65.0)
    parser.add_argument("--preferred-orientation-degrees", type=float, default=20.0)
    parser.add_argument("--agent-action-scale", type=float, default=1.0)
    parser.add_argument("--target-clearance-only", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--agent-max-steps", type=int, default=700)
    parser.add_argument("--prefix-max-steps", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--init-log-std", type=float, default=-1.0)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def make_env(args, body, approach, clearance, monitor_path=None):
    """学習と評価で同じ早期交接・安全制約を構築する。"""
    return make_first_early_landing_env(
        body,
        approach_model=approach,
        clearance_model=clearance,
        prefix_fraction=args.prefix_fraction,
        agent_max_steps=args.agent_max_steps,
        prefix_max_steps=args.prefix_max_steps,
        max_orientation=math.radians(args.max_orientation_degrees),
        preferred_orientation=math.radians(args.preferred_orientation_degrees),
        agent_action_scale=args.agent_action_scale,
        target_clearance_only=args.target_clearance_only,
        monitor_path=monitor_path,
    )


def evaluate(model, args, body, approach, clearance):
    """決定論的に胴体接地と着地成功を測る。"""
    env = make_env(args, body, approach, clearance)
    rows = []
    try:
        for episode in range(args.eval_episodes):
            obs, info = env.reset(seed=10_000 + episode)
            total_return = 0.0
            maximum_angle = float(info["orientation_error"])
            contact_steps = 0
            for steps in range(1, args.agent_max_steps + 1):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_return += float(reward)
                maximum_angle = max(
                    maximum_angle,
                    float(info["orientation_error"]),
                    abs(float(info.get("unwrapped_orientation_error", 0.0))),
                )
                contact_steps += int(bool(info.get("upper_body_grounded", False)))
                if terminated or truncated:
                    break
            if args.target_clearance_only:
                success = bool(
                    int(info["strict_clearances"]) >= 1
                    and contact_steps == 0
                    and maximum_angle < math.radians(args.max_orientation_degrees)
                )
            else:
                success = bool(
                    int(info["stable_landings"]) >= 1
                    and contact_steps == 0
                    and maximum_angle < math.radians(args.max_orientation_degrees)
                )
            rows.append(
                {
                    "return": total_return,
                    "steps": steps,
                    "clearances": float(info["strict_clearances"]),
                    "landings": float(info["stable_landings"]),
                    "maximum_degrees": math.degrees(maximum_angle),
                    "contact_steps": float(contact_steps),
                    "success": float(success),
                }
            )
    finally:
        env.close()
    return {
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_strict_clearances": float(
            np.mean([row["clearances"] for row in rows])
        ),
        "mean_stable_landings": float(np.mean([row["landings"] for row in rows])),
        "mean_maximum_degrees": float(
            np.mean([row["maximum_degrees"] for row in rows])
        ),
        "mean_body_contact_steps": float(
            np.mean([row["contact_steps"] for row in rows])
        ),
        "success_rate": float(np.mean([row["success"] for row in rows])),
    }


def score(metrics):
    """真の成功、着地、通過、非接地、角度、回報の順で比較する。"""
    return (
        metrics["success_rate"],
        metrics["mean_stable_landings"],
        metrics["mean_strict_clearances"],
        -metrics["mean_body_contact_steps"],
        -metrics["mean_maximum_degrees"],
        metrics["mean_return"],
    )


def append_row(path, row):
    """5k評価を固定列CSVへ追記する。"""
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in FIELDS})


class Callback(BaseCallback):
    """5kごとに厳格な胴体非接地評価とモデル保存を行う。"""

    def __init__(self, args, body, approach, clearance, run_dir, csv_path, best):
        super().__init__(verbose=0)
        self.args = args
        self.body = body
        self.approach = approach
        self.clearance = clearance
        self.run_dir = run_dir
        self.csv_path = csv_path
        self.best = best
        self.best_timesteps = 0
        self.next_eval = args.eval_interval
        self.checkpoints = run_dir / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)

    def _on_step(self):
        while self.num_timesteps >= self.next_eval:
            metrics = evaluate(
                self.model,
                self.args,
                self.body,
                self.approach,
                self.clearance,
            )
            append_row(self.csv_path, {"timesteps": self.num_timesteps, **metrics})
            self.model.save(self.checkpoints / f"model_{self.num_timesteps}_steps.zip")
            self.model.save(self.run_dir / "latest_model.zip")
            current = score(metrics)
            if current > self.best:
                self.best = current
                self.best_timesteps = int(self.num_timesteps)
                self.model.save(self.run_dir / "best_model.zip")
            print(
                f"[early-landing] step={self.num_timesteps} "
                f"clear={metrics['mean_strict_clearances']:.2f} "
                f"land={metrics['mean_stable_landings']:.2f} "
                f"angle={metrics['mean_maximum_degrees']:.1f} "
                f"contact={metrics['mean_body_contact_steps']:.1f} "
                f"success={metrics['success_rate']:.2f}",
                flush=True,
            )
            self.next_eval += self.args.eval_interval
        return True


def main():
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    approach_path = Path(args.approach_model).resolve()
    clearance_path = Path(args.clearance_model).resolve()
    init_path = Path(args.init_model).resolve()
    approach = PPO.load(approach_path, device="cpu")
    clearance = PPO.load(clearance_path, device="cpu")
    body = make_body()
    run_dir = RUNS_DIR / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"实验目录已有文件：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "early_landing_evaluation_5k.csv"
    env = make_env(args, body, approach, clearance, run_dir / "training")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=1_024,
        batch_size=64,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
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
    initial = evaluate(model, args, body, approach, clearance)
    append_row(csv_path, {"timesteps": 0, **initial})
    model.save(run_dir / "initial_model.zip")
    model.save(run_dir / "best_model.zip")
    np.save(run_dir / "body.npy", body)
    write_json(run_dir / "course.json", get_course(2).as_dict())
    write_json(
        run_dir / "config.json",
        {
            "algorithm": "PPO first-obstacle early landing",
            "course_version": COURSE_VERSION,
            "reward_version": REWARD_VERSION,
            "approach_model": str(approach_path),
            "clearance_model": str(clearance_path),
            "init_model": str(init_path),
            "prefix_fraction": args.prefix_fraction,
            "max_orientation_degrees": args.max_orientation_degrees,
            "agent_action_scale": args.agent_action_scale,
            "target_clearance_only": args.target_clearance_only,
            "target_timesteps": args.total_timesteps,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    callback = Callback(
        args,
        body,
        approach,
        clearance,
        run_dir,
        csv_path,
        score(initial),
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
            "best_score": list(callback.best),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    env.close()
    print(f"第一障碍提前落地训练完成：{run_dir}", flush=True)


if __name__ == "__main__":
    main()
