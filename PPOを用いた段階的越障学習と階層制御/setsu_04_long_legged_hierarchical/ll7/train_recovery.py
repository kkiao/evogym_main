"""教師越障後の直立着地と再前進をPPOで集中的に学習する。"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from ll7.body import BODY_NAME, make_body
from ll7.curriculum import COURSE_VERSION, get_course
from ll7.environment import REWARD_VERSION
from ll7.experiment import (
    RUNS_DIR,
    append_evaluation,
    evaluate_hierarchical,
    evaluate_four_stage,
    evaluate_prefix_clearance_landing_then_agent,
    evaluate_prefix_clearance_then_agent,
    evaluate_prefix_then_agent,
    evaluate_policy_list_then_agent,
    evaluate_three_stage,
    make_next_obstacle_env,
    make_restart_env,
    make_recovery_env,
    make_three_stage_recovery_env,
    read_evaluations,
    score_metrics,
    write_json,
)
from ll7.train_curriculum import transfer_policy


def parse_args():
    parser = argparse.ArgumentParser(description="集中训练越墙后的扶正、稳定落地和恢复前进。")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--pre-handoff-model")
    parser.add_argument("--landing-model")
    parser.add_argument("--restart-model")
    parser.add_argument("--next-clearance-model")
    parser.add_argument("--next-landing-model")
    parser.add_argument("--prefix-manifest")
    parser.add_argument("--init-model")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--network-width", type=int, choices=(64, 128), default=64)
    parser.add_argument("--init-log-std", type=float, default=-1.0)
    parser.add_argument("--agent-max-steps", type=int, default=1_000)
    parser.add_argument("--prefix-max-steps", type=int, default=1_000)
    parser.add_argument("--handoff-x", type=float)
    parser.add_argument("--target-validated", type=int)
    parser.add_argument("--target-clearances", type=int)
    parser.add_argument("--target-stable-landings", type=int)
    parser.add_argument("--target-crossing-fraction", type=float)
    parser.add_argument("--upright-clearance-speed", type=float)
    parser.add_argument("--upright-clearance-angle-degrees", type=float, default=35.0)
    parser.add_argument("--prefix-validated", type=int)
    parser.add_argument("--prefix-clearances", type=int)
    parser.add_argument("--handoff-distance", type=float, default=0.25)
    parser.add_argument("--training-angle-limit-degrees", type=float, default=35.0)
    parser.add_argument("--training-landing-speed-limit", type=float, default=0.15)
    parser.add_argument("--max-orientation-degrees", type=float)
    parser.add_argument("--preferred-orientation-degrees", type=float, default=35.0)
    parser.add_argument("--allow-upper-body-contact", action="store_true")
    parser.add_argument("--agent-action-scale", type=float, default=1.0)
    parser.add_argument("--brake-model")
    parser.add_argument("--brake-action-scale", type=float, default=1.0)
    parser.add_argument("--brake-target-speed", type=float, default=0.10)
    parser.add_argument("--brake-stable-steps", type=int, default=10)
    parser.add_argument("--righting-prefix-model")
    parser.add_argument("--righting-prefix-action-scale", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser.parse_args()


def validate_args(args):
    """回復学習の数値設定と実験名を作成前に検証する。"""
    for name in (
        "total_timesteps",
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "agent_max_steps",
        "prefix_max_steps",
        "eval_interval",
        "eval_episodes",
        "torch_threads",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于0。")
    if args.n_steps % args.batch_size:
        raise ValueError("--n-steps 必须能被 --batch-size 整除。")
    if not 0.0 < args.training_angle_limit_degrees <= 120.0:
        raise ValueError("--training-angle-limit-degrees 必须在0到120之间。")
    if args.training_landing_speed_limit <= 0.0:
        raise ValueError("--training-landing-speed-limit 必须大于0。")
    if (
        args.target_crossing_fraction is not None
        and not 0.0 < args.target_crossing_fraction <= 1.0
    ):
        raise ValueError("--target-crossing-fraction 必须在0到1之间。")
    if args.handoff_distance <= 0.0:
        raise ValueError("--handoff-distance 必须大于0。")
    if (
        args.max_orientation_degrees is not None
        and not 0.0 < args.max_orientation_degrees <= 180.0
    ):
        raise ValueError("--max-orientation-degrees 必须在0到180之间。")
    if not 0.0 < args.preferred_orientation_degrees < 90.0:
        raise ValueError("--preferred-orientation-degrees 必须在0到90之间。")
    if not 0.0 < args.upright_clearance_angle_degrees <= 90.0:
        raise ValueError("--upright-clearance-angle-degrees 必须在0到90之间。")
    if not 0.0 < args.agent_action_scale <= 1.0:
        raise ValueError("--agent-action-scale 必须大于0且不超过1。")
    if not 0.0 < args.brake_action_scale <= 1.0:
        raise ValueError("--brake-action-scale 必须大于0且不超过1。")
    if args.brake_target_speed <= 0.0 or args.brake_stable_steps <= 0:
        raise ValueError("制动速度和连续步数必须大于0。")
    if not 0.0 < args.righting_prefix_action_scale <= 1.0:
        raise ValueError("--righting-prefix-action-scale 必须大于0且不超过1。")
    if args.righting_prefix_model and not args.brake_model:
        raise ValueError("扶正前缀需要同时提供制动模型。")
    run_path = Path(args.run_name)
    if run_path.name != args.run_name or args.run_name in {"", ".", ".."}:
        raise ValueError("--run-name 必须是单个目录名。")


def load_prefix_obstacle_policies(path: Path):
    """実験ルート相対の方策一覧を読み込みPPO方策へ変換する。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[1]

    def load_model(model_path: str):
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return PPO.load(candidate.resolve(), device="cpu")

    policies = []
    for item in data["obstacles"]:
        policy = {
            "clearance": load_model(item["clearance_model"]),
            "landing": load_model(item["landing_model"]),
            "restart": load_model(item["restart_model"]),
            "visible_obstacle_count": item.get("visible_obstacle_count"),
        }
        if item.get("approach_model"):
            policy["approach"] = load_model(item["approach_model"])
        policies.append(policy)
    return policies


class RecoveryEvaluationCallback(BaseCallback):
    """5k刻みで階層制御を評価し、真の最良回復方策を保存する。"""

    def __init__(
        self,
        args,
        body,
        course,
        teacher_model,
        pre_handoff_model,
        landing_model,
        restart_model,
        next_clearance_model,
        next_landing_model,
        prefix_obstacle_policies,
        run_dir,
        evaluation_path,
        best_score,
    ):
        super().__init__(verbose=0)
        self.args = args
        self.body = body
        self.course = course
        self.teacher_model = teacher_model
        self.pre_handoff_model = pre_handoff_model
        self.landing_model = landing_model
        self.restart_model = restart_model
        self.next_clearance_model = next_clearance_model
        self.next_landing_model = next_landing_model
        self.prefix_obstacle_policies = prefix_obstacle_policies
        self.run_dir = run_dir
        self.evaluation_path = evaluation_path
        self.next_eval = args.eval_interval
        self.best_score = best_score
        self.best_timesteps = 0
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

    def _save_and_evaluate(self):
        if self.prefix_obstacle_policies is not None:
            metrics = evaluate_policy_list_then_agent(
                self.pre_handoff_model,
                self.prefix_obstacle_policies,
                self.model,
                self.args.prefix_validated,
                self.args.handoff_distance,
                self.body,
                self.args.level,
                self.args.eval_episodes,
                self.course.max_steps,
                self.args.eval_seed,
                next_clearance_model=self.next_clearance_model,
                next_landing_model=self.next_landing_model,
                prefix_clearances=self.args.prefix_clearances,
            )
        elif self.next_landing_model is not None:
            metrics = evaluate_prefix_clearance_landing_then_agent(
                self.pre_handoff_model,
                self.teacher_model,
                self.landing_model,
                self.restart_model,
                self.next_clearance_model,
                self.next_landing_model,
                self.model,
                self.args.prefix_validated,
                self.args.prefix_clearances,
                self.args.handoff_distance,
                self.body,
                self.args.level,
                self.args.eval_episodes,
                self.course.max_steps,
                self.args.eval_seed,
            )
        elif self.next_clearance_model is not None:
            metrics = evaluate_prefix_clearance_then_agent(
                self.pre_handoff_model,
                self.teacher_model,
                self.landing_model,
                self.restart_model,
                self.next_clearance_model,
                self.model,
                self.args.prefix_validated,
                self.args.prefix_clearances,
                self.args.handoff_distance,
                self.body,
                self.args.level,
                self.args.eval_episodes,
                self.course.max_steps,
                self.args.eval_seed,
            )
        elif self.restart_model is not None:
            metrics = evaluate_prefix_then_agent(
                self.pre_handoff_model,
                self.teacher_model,
                self.landing_model,
                self.restart_model,
                self.model,
                self.args.prefix_validated,
                self.args.handoff_distance,
                self.body,
                self.args.level,
                self.args.eval_episodes,
                self.course.max_steps,
                self.args.eval_seed,
            )
        elif self.landing_model is not None:
            metrics = evaluate_four_stage(
                self.pre_handoff_model,
                self.teacher_model,
                self.landing_model,
                self.model,
                self.args.handoff_x,
                self.body,
                self.args.level,
                self.args.eval_episodes,
                self.course.max_steps,
                self.args.eval_seed,
            )
        elif self.pre_handoff_model is None:
            metrics = evaluate_hierarchical(
                self.teacher_model,
                self.model,
                self.body,
                self.args.level,
                self.args.eval_episodes,
                self.course.max_steps,
                self.args.eval_seed,
                self.args.handoff_x,
            )
        else:
            metrics = evaluate_three_stage(
                self.pre_handoff_model,
                self.teacher_model,
                self.model,
                self.args.handoff_x,
                self.body,
                self.args.level,
                self.args.eval_episodes,
                self.course.max_steps,
                self.args.eval_seed,
                recovery_action_scale=self.args.agent_action_scale,
            )
        append_evaluation(
            self.evaluation_path,
            {"timesteps": int(self.num_timesteps), **metrics},
        )
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
            f"[recovery L{self.args.level}] step={self.num_timesteps} "
            f"strict={metrics['mean_strict_clearances']:.2f} "
            f"land={metrics['mean_stable_landings']:.2f} "
            f"restart={metrics['mean_restart_successes']:.2f} "
            f"validated={metrics['mean_validated_obstacles']:.2f} "
            f"success={metrics['success_rate']:.2f}",
            flush=True,
        )

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval:
            self._save_and_evaluate()
            self.next_eval += self.args.eval_interval
        return True


def main():
    args = parse_args()
    validate_args(args)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    body = make_body()
    course = get_course(args.level)
    teacher_path = Path(args.teacher_model).resolve()
    teacher_model = PPO.load(teacher_path, device="cpu")
    pre_handoff_path = (
        Path(args.pre_handoff_model).resolve() if args.pre_handoff_model else None
    )
    pre_handoff_model = (
        PPO.load(pre_handoff_path, device="cpu") if pre_handoff_path else None
    )
    landing_path = Path(args.landing_model).resolve() if args.landing_model else None
    landing_model = PPO.load(landing_path, device="cpu") if landing_path else None
    restart_path = Path(args.restart_model).resolve() if args.restart_model else None
    restart_model = PPO.load(restart_path, device="cpu") if restart_path else None
    next_clearance_path = (
        Path(args.next_clearance_model).resolve() if args.next_clearance_model else None
    )
    next_clearance_model = (
        PPO.load(next_clearance_path, device="cpu") if next_clearance_path else None
    )
    next_landing_path = (
        Path(args.next_landing_model).resolve() if args.next_landing_model else None
    )
    next_landing_model = (
        PPO.load(next_landing_path, device="cpu") if next_landing_path else None
    )
    prefix_manifest_path = (
        Path(args.prefix_manifest).resolve() if args.prefix_manifest else None
    )
    prefix_obstacle_policies = (
        load_prefix_obstacle_policies(prefix_manifest_path)
        if prefix_manifest_path
        else None
    )
    init_path = Path(args.init_model).resolve() if args.init_model else None
    brake_path = Path(args.brake_model).resolve() if args.brake_model else None
    brake_model = PPO.load(brake_path, device="cpu") if brake_path else None
    righting_prefix_path = (
        Path(args.righting_prefix_model).resolve()
        if args.righting_prefix_model
        else None
    )
    righting_prefix_model = (
        PPO.load(righting_prefix_path, device="cpu")
        if righting_prefix_path
        else None
    )
    run_dir = RUNS_DIR / args.run_name
    evaluation_path = run_dir / "hierarchical_evaluation_5k.csv"
    latest_path = run_dir / "latest_model.zip"
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"找不到续训模型：{latest_path}")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"实验目录已有文件：{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    if restart_model is not None:
        if (
            pre_handoff_model is None
            or landing_model is None
            or args.prefix_validated is None
            or sum(
                value is not None
                for value in (
                    args.target_validated,
                    args.target_clearances,
                    args.target_stable_landings,
                    args.upright_clearance_speed,
                    args.target_crossing_fraction,
                )
            )
            != 1
            or (next_clearance_model is not None and args.prefix_clearances is None)
            or (next_landing_model is not None and next_clearance_model is None)
        ):
            raise ValueError(
                "使用 --restart-model 时还必须提供前置、落地模型及前缀和目标数量。"
            )
        train_env = make_next_obstacle_env(
            body,
            args.level,
            approach_model=pre_handoff_model,
            clearance_model=teacher_model,
            landing_model=landing_model,
            restart_model=restart_model,
            prefix_validated=args.prefix_validated,
            target_validated=args.target_validated,
            target_clearances=args.target_clearances,
            target_stable_landings=args.target_stable_landings,
            upright_clearance_speed=args.upright_clearance_speed,
            target_crossing_fraction=args.target_crossing_fraction,
            handoff_distance=args.handoff_distance,
            agent_max_steps=args.agent_max_steps,
            prefix_max_steps=args.prefix_max_steps,
            landing_angle_limit=math.radians(args.training_angle_limit_degrees),
            landing_speed_limit=args.training_landing_speed_limit,
            next_clearance_model=next_clearance_model,
            next_landing_model=next_landing_model,
            prefix_clearances=args.prefix_clearances,
            upright_clearance_angle=math.radians(
                args.upright_clearance_angle_degrees
            ),
            prefix_obstacle_policies=prefix_obstacle_policies,
            max_orientation=(
                math.radians(args.max_orientation_degrees)
                if args.max_orientation_degrees is not None
                else None
            ),
            preferred_orientation=math.radians(
                args.preferred_orientation_degrees
            ),
            forbid_upper_body_contact=not args.allow_upper_body_contact,
            monitor_path=run_dir / "training",
        )
    elif landing_model is not None:
        if pre_handoff_model is None or args.handoff_x is None:
            raise ValueError("使用 --landing-model 时还必须提供前置模型和交接位置。")
        train_env = make_restart_env(
            body,
            args.level,
            approach_model=pre_handoff_model,
            clearance_model=teacher_model,
            landing_model=landing_model,
            handoff_x=args.handoff_x,
            agent_max_steps=args.agent_max_steps,
            prefix_max_steps=args.prefix_max_steps,
            target_validated=args.target_validated,
            target_stable_landings=args.target_stable_landings,
            max_orientation=(
                math.radians(args.max_orientation_degrees)
                if args.max_orientation_degrees is not None
                else None
            ),
            preferred_orientation=math.radians(
                args.preferred_orientation_degrees
            ),
            forbid_upper_body_contact=not args.allow_upper_body_contact,
            monitor_path=run_dir / "training",
        )
    elif pre_handoff_model is None:
        train_env = make_recovery_env(
            body,
            args.level,
            teacher_model,
            target_clearances=1,
            agent_max_steps=args.agent_max_steps,
            prefix_max_steps=args.prefix_max_steps,
            handoff_x=args.handoff_x,
            target_validated=args.target_validated,
            landing_angle_limit=math.radians(args.training_angle_limit_degrees),
            monitor_path=run_dir / "training",
        )
    else:
        if args.handoff_x is None:
            raise ValueError("使用 --pre-handoff-model 时必须提供 --handoff-x。")
        train_env = make_three_stage_recovery_env(
            body,
            args.level,
            approach_model=pre_handoff_model,
            clearance_model=teacher_model,
            handoff_x=args.handoff_x,
            agent_max_steps=args.agent_max_steps,
            prefix_max_steps=args.prefix_max_steps,
            target_validated=args.target_validated,
            target_stable_landings=args.target_stable_landings,
            max_orientation=(
                math.radians(args.max_orientation_degrees)
                if args.max_orientation_degrees is not None
                else None
            ),
            preferred_orientation=math.radians(
                args.preferred_orientation_degrees
            ),
            forbid_upper_body_contact=not args.allow_upper_body_contact,
            agent_action_scale=args.agent_action_scale,
            brake_model=brake_model,
            brake_action_scale=args.brake_action_scale,
            brake_target_speed=args.brake_target_speed,
            brake_stable_steps=args.brake_stable_steps,
            righting_prefix_model=righting_prefix_model,
            righting_prefix_action_scale=args.righting_prefix_action_scale,
            monitor_path=run_dir / "training",
        )
    train_env.action_space.seed(args.seed)

    if args.resume:
        model = PPO.load(latest_path, env=train_env, device="cpu")
        completed_timesteps = int(model.num_timesteps)
        rows = read_evaluations(evaluation_path)
        best_row = max(rows, key=score_metrics)
        best_score = score_metrics(best_row)
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
            verbose=0,
        )
        transfer_policy(
            init_path if init_path else (landing_path if landing_path else teacher_path),
            model,
        )
        with torch.no_grad():
            model.policy.log_std.fill_(args.init_log_std)
        completed_timesteps = 0
        if prefix_obstacle_policies is not None:
            initial = evaluate_policy_list_then_agent(
                pre_handoff_model,
                prefix_obstacle_policies,
                model,
                args.prefix_validated,
                args.handoff_distance,
                body,
                args.level,
                args.eval_episodes,
                course.max_steps,
                args.eval_seed,
                next_clearance_model=next_clearance_model,
                next_landing_model=next_landing_model,
                prefix_clearances=args.prefix_clearances,
            )
        elif next_landing_model is not None:
            initial = evaluate_prefix_clearance_landing_then_agent(
                pre_handoff_model,
                teacher_model,
                landing_model,
                restart_model,
                next_clearance_model,
                next_landing_model,
                model,
                args.prefix_validated,
                args.prefix_clearances,
                args.handoff_distance,
                body,
                args.level,
                args.eval_episodes,
                course.max_steps,
                args.eval_seed,
            )
        elif next_clearance_model is not None:
            initial = evaluate_prefix_clearance_then_agent(
                pre_handoff_model,
                teacher_model,
                landing_model,
                restart_model,
                next_clearance_model,
                model,
                args.prefix_validated,
                args.prefix_clearances,
                args.handoff_distance,
                body,
                args.level,
                args.eval_episodes,
                course.max_steps,
                args.eval_seed,
            )
        elif restart_model is not None:
            initial = evaluate_prefix_then_agent(
                pre_handoff_model,
                teacher_model,
                landing_model,
                restart_model,
                model,
                args.prefix_validated,
                args.handoff_distance,
                body,
                args.level,
                args.eval_episodes,
                course.max_steps,
                args.eval_seed,
            )
        elif landing_model is not None:
            initial = evaluate_four_stage(
                pre_handoff_model,
                teacher_model,
                landing_model,
                model,
                args.handoff_x,
                body,
                args.level,
                args.eval_episodes,
                course.max_steps,
                args.eval_seed,
            )
        elif pre_handoff_model is None:
            initial = evaluate_hierarchical(
                teacher_model,
                model,
                body,
                args.level,
                args.eval_episodes,
                course.max_steps,
                args.eval_seed,
                args.handoff_x,
            )
        else:
            initial = evaluate_three_stage(
                pre_handoff_model,
                teacher_model,
                model,
                args.handoff_x,
                body,
                args.level,
                args.eval_episodes,
                course.max_steps,
                args.eval_seed,
            )
        append_evaluation(evaluation_path, {"timesteps": 0, **initial})
        model.save(run_dir / "initial_model.zip")
        model.save(run_dir / "best_model.zip")
        model.save(latest_path)
        best_score = score_metrics(initial)
        np.save(run_dir / "body.npy", body)
        write_json(run_dir / "course.json", course.as_dict())
        write_json(
            run_dir / "config.json",
            {
                "algorithm": "PPO hierarchical recovery",
                "course_version": COURSE_VERSION,
                "reward_version": REWARD_VERSION,
                "body_name": BODY_NAME,
                "curriculum_level": args.level,
                "run_name": args.run_name,
                "teacher_model": str(teacher_path),
                "pre_handoff_model": (
                    str(pre_handoff_path) if pre_handoff_path else None
                ),
                "landing_model": str(landing_path) if landing_path else None,
                "restart_model": str(restart_path) if restart_path else None,
                "next_clearance_model": (
                    str(next_clearance_path) if next_clearance_path else None
                ),
                "next_landing_model": (
                    str(next_landing_path) if next_landing_path else None
                ),
                "prefix_manifest": (
                    str(prefix_manifest_path) if prefix_manifest_path else None
                ),
                "init_model": str(init_path) if init_path else None,
                "target_timesteps": args.total_timesteps,
                "agent_max_steps": args.agent_max_steps,
                "prefix_max_steps": args.prefix_max_steps,
                "handoff_x": args.handoff_x,
                "target_validated": args.target_validated,
                "target_clearances": args.target_clearances,
                "target_stable_landings": args.target_stable_landings,
                "target_crossing_fraction": args.target_crossing_fraction,
                "upright_clearance_speed": args.upright_clearance_speed,
                "upright_clearance_angle_degrees": (
                    args.upright_clearance_angle_degrees
                ),
                "prefix_validated": args.prefix_validated,
                "prefix_clearances": args.prefix_clearances,
                "handoff_distance": args.handoff_distance,
                "training_angle_limit_degrees": args.training_angle_limit_degrees,
                "training_landing_speed_limit": args.training_landing_speed_limit,
                "max_orientation_degrees": args.max_orientation_degrees,
                "preferred_orientation_degrees": (
                    args.preferred_orientation_degrees
                ),
                "allow_upper_body_contact": args.allow_upper_body_contact,
                "agent_action_scale": args.agent_action_scale,
                "brake_model": str(brake_path) if brake_path else None,
                "brake_action_scale": args.brake_action_scale,
                "brake_target_speed": args.brake_target_speed,
                "brake_stable_steps": args.brake_stable_steps,
                "righting_prefix_model": (
                    str(righting_prefix_path) if righting_prefix_path else None
                ),
                "righting_prefix_action_scale": args.righting_prefix_action_scale,
                "learning_rate": args.learning_rate,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "ent_coef": args.ent_coef,
                "network_width": args.network_width,
                "init_log_std": args.init_log_std,
                "eval_interval": args.eval_interval,
                "eval_episodes": args.eval_episodes,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    callback = RecoveryEvaluationCallback(
        args,
        body,
        course,
        teacher_model,
        pre_handoff_model,
        landing_model,
        restart_model,
        next_clearance_model,
        next_landing_model,
        prefix_obstacle_policies,
        run_dir,
        evaluation_path,
        best_score,
    )
    if completed_timesteps < args.total_timesteps:
        model.learn(
            total_timesteps=args.total_timesteps - completed_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )
    model.save(latest_path)
    final_timesteps = int(model.num_timesteps)
    write_json(
        run_dir / "summary.json",
        {
            "run_name": args.run_name,
            "completed_timesteps": final_timesteps,
            "target_timesteps": args.total_timesteps,
            "best_score": list(callback.best_score),
            "best_timesteps": callback.best_timesteps,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    train_env.close()
    print(f"恢复训练完成：{run_dir}", flush=True)


if __name__ == "__main__":
    main()
