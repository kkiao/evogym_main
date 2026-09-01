"""無側倒第一障害物前綴から第二障害物の部分通過をPPOで学習する。"""

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
from ll7.experiment import (
    RUNS_DIR,
    make_improved_second_crossing_env,
    make_true_noroll_second_crossing_env,
    write_json,
)
from ll7.train_curriculum import transfer_policy


FIELDS = (
    "timesteps",
    "mean_return",
    "mean_steps",
    "mean_fraction",
    "mean_strict_clearances",
    "mean_maximum_degrees",
    "mean_body_contact_steps",
    "success_rate",
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练第二墙分阶段部分跨越。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--approach-model", required=True)
    parser.add_argument("--first-clearance-model", required=True)
    parser.add_argument("--first-brake-model", required=True)
    parser.add_argument("--first-righting-model", required=True)
    parser.add_argument("--first-restart-model", required=True)
    parser.add_argument("--first-safe-clearance-model")
    parser.add_argument("--first-landing-model")
    parser.add_argument("--first-landing-scale", type=float, default=0.05)
    parser.add_argument("--first-restart-scale", type=float, default=0.75)
    parser.add_argument("--second-handoff-distance", type=float, default=0.25)
    parser.add_argument("--max-orientation-degrees", type=float, default=65.0)
    parser.add_argument("--preferred-orientation-degrees", type=float, default=20.0)
    parser.add_argument("--init-model", required=True)
    parser.add_argument("--second-prefix-model")
    parser.add_argument("--second-prefix-fraction", type=float)
    parser.add_argument("--second-prefix-model-2")
    parser.add_argument("--second-prefix-fraction-2", type=float)
    parser.add_argument("--target-fraction", type=float, required=True)
    parser.add_argument("--total-timesteps", type=int, default=60_000)
    parser.add_argument("--agent-max-steps", type=int, default=500)
    parser.add_argument("--prefix-max-steps", type=int, default=3_000)
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


def append_row(path: Path, row: dict):
    """固定列順で5k評価をCSVへ追記する。"""
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in FIELDS})


def make_env(args, body, models, monitor_path=None):
    """同じ検証済み第一障害物前綴を学習と評価で共有する。"""
    if "safe_clearance" in models:
        return make_improved_second_crossing_env(
            body,
            approach_model=models["approach"],
            first_half_model=models["clearance"],
            first_safe_clearance_model=models["safe_clearance"],
            first_landing_model=models["landing"],
            first_restart_model=models["restart"],
            target_fraction=args.target_fraction,
            agent_max_steps=args.agent_max_steps,
            prefix_max_steps=args.prefix_max_steps,
            second_handoff_distance=args.second_handoff_distance,
            first_landing_action_scale=args.first_landing_scale,
            first_restart_action_scale=args.first_restart_scale,
            second_prefix_model=models.get("second_prefix"),
            second_prefix_fraction=args.second_prefix_fraction,
            second_prefix_model_2=models.get("second_prefix_2"),
            second_prefix_fraction_2=args.second_prefix_fraction_2,
            max_orientation=math.radians(args.max_orientation_degrees),
            preferred_orientation=math.radians(args.preferred_orientation_degrees),
            monitor_path=monitor_path,
        )
    return make_true_noroll_second_crossing_env(
        body,
        approach_model=models["approach"],
        first_clearance_model=models["clearance"],
        first_brake_model=models["brake"],
        first_righting_model=models["righting"],
        first_restart_model=models["restart"],
        target_fraction=args.target_fraction,
        agent_max_steps=args.agent_max_steps,
        prefix_max_steps=args.prefix_max_steps,
        second_prefix_model=models.get("second_prefix"),
        second_prefix_fraction=args.second_prefix_fraction,
        second_prefix_model_2=models.get("second_prefix_2"),
        second_prefix_fraction_2=args.second_prefix_fraction_2,
        monitor_path=monitor_path,
    )


def evaluate(model, args, body, models):
    """決定論的に第二障害物の最大通過割合を測る。"""
    env = make_env(args, body, models)
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
            safe_success = bool(
                info["crossing_fraction_subgoal_success"]
                and contact_steps == 0
                and maximum_angle < math.radians(args.max_orientation_degrees)
            )
            rows.append(
                {
                    "return": total_return,
                    "steps": steps,
                    "fraction": float(info["maximum_crossed_fraction"]),
                    "clearances": float(info["strict_clearances"]),
                    "maximum_degrees": math.degrees(maximum_angle),
                    "contact_steps": float(contact_steps),
                    "success": float(safe_success),
                }
            )
    finally:
        env.close()
    return {
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_fraction": float(np.mean([row["fraction"] for row in rows])),
        "mean_strict_clearances": float(
            np.mean([row["clearances"] for row in rows])
        ),
        "mean_maximum_degrees": float(
            np.mean([row["maximum_degrees"] for row in rows])
        ),
        "mean_body_contact_steps": float(
            np.mean([row["contact_steps"] for row in rows])
        ),
        "success_rate": float(np.mean([row["success"] for row in rows])),
    }


def score(metrics):
    """部分通過成功、割合、厳格通過、回報の順で比較する。"""
    return (
        metrics["success_rate"],
        metrics["mean_fraction"],
        metrics["mean_strict_clearances"],
        -metrics["mean_body_contact_steps"],
        -metrics["mean_maximum_degrees"],
        metrics["mean_return"],
    )


class EvaluationCallback(BaseCallback):
    """5kごとに第二障害物の部分通過率を保存する。"""

    def __init__(self, args, body, models, run_dir, csv_path, best_score):
        super().__init__(verbose=0)
        self.args = args
        self.body = body
        self.models = models
        self.run_dir = run_dir
        self.csv_path = csv_path
        self.best_score = best_score
        self.best_timesteps = 0
        self.next_eval = args.eval_interval
        self.checkpoints = run_dir / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)

    def _on_step(self):
        while self.num_timesteps >= self.next_eval:
            metrics = evaluate(self.model, self.args, self.body, self.models)
            append_row(self.csv_path, {"timesteps": self.num_timesteps, **metrics})
            self.model.save(self.checkpoints / f"model_{self.num_timesteps}_steps.zip")
            self.model.save(self.run_dir / "latest_model.zip")
            current = score(metrics)
            if current > self.best_score:
                self.best_score = current
                self.best_timesteps = int(self.num_timesteps)
                self.model.save(self.run_dir / "best_model.zip")
            print(
                f"[second-crossing] step={self.num_timesteps} "
                f"fraction={metrics['mean_fraction']:.2f} "
                f"clear={metrics['mean_strict_clearances']:.2f} "
                f"angle={metrics['mean_maximum_degrees']:.1f} "
                f"contact={metrics['mean_body_contact_steps']:.1f} "
                f"success={metrics['success_rate']:.2f}",
                flush=True,
            )
            self.next_eval += self.args.eval_interval
        return True


def main():
    args = parse_args()
    if not 0.0 < args.target_fraction <= 1.0:
        raise ValueError("--target-fraction 必须在0到1之间。")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    paths = {
        "approach": Path(args.approach_model).resolve(),
        "clearance": Path(args.first_clearance_model).resolve(),
        "brake": Path(args.first_brake_model).resolve(),
        "righting": Path(args.first_righting_model).resolve(),
        "restart": Path(args.first_restart_model).resolve(),
    }
    if args.first_safe_clearance_model or args.first_landing_model:
        if not args.first_safe_clearance_model or not args.first_landing_model:
            raise ValueError("改良第一墙前缀必须同时提供安全越墙与落地模型。")
        paths["safe_clearance"] = Path(args.first_safe_clearance_model).resolve()
        paths["landing"] = Path(args.first_landing_model).resolve()
    if args.second_prefix_model:
        if args.second_prefix_fraction is None:
            raise ValueError("第二墙前缀模型必须提供 --second-prefix-fraction。")
        paths["second_prefix"] = Path(args.second_prefix_model).resolve()
    if args.second_prefix_model_2:
        if args.second_prefix_fraction_2 is None:
            raise ValueError("第二个第二墙前缀模型必须提供目标比例。")
        paths["second_prefix_2"] = Path(args.second_prefix_model_2).resolve()
    models = {key: PPO.load(path, device="cpu") for key, path in paths.items()}
    init_path = Path(args.init_model).resolve()
    body = make_body()
    run_dir = RUNS_DIR / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"实验目录已有文件：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "second_crossing_evaluation_5k.csv"
    env = make_env(args, body, models, monitor_path=run_dir / "training")
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
    initial = evaluate(model, args, body, models)
    append_row(csv_path, {"timesteps": 0, **initial})
    model.save(run_dir / "initial_model.zip")
    model.save(run_dir / "best_model.zip")
    np.save(run_dir / "body.npy", body)
    write_json(run_dir / "course.json", get_course(2).as_dict())
    write_json(
        run_dir / "config.json",
        {
            "algorithm": "PPO second-obstacle crossing sub-curriculum",
            "course_version": COURSE_VERSION,
            "reward_version": REWARD_VERSION,
            "prefix_models": {key: str(path) for key, path in paths.items()},
            "init_model": str(init_path),
            "target_fraction": args.target_fraction,
            "second_prefix_fraction": args.second_prefix_fraction,
            "second_prefix_fraction_2": args.second_prefix_fraction_2,
            "target_timesteps": args.total_timesteps,
            "learning_rate": args.learning_rate,
            "ent_coef": args.ent_coef,
            "init_log_std": args.init_log_std,
            "n_epochs": args.n_epochs,
            "clip_range": args.clip_range,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    callback = EvaluationCallback(
        args,
        body,
        models,
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
            "best_score": list(callback.best_score),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    env.close()
    print(f"第二障碍部分跨越训练完成：{run_dir}", flush=True)


if __name__ == "__main__":
    main()
