"""カリキュラム環境の構築、評価、実験ファイル管理を共通化する。"""

from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from evogym import get_full_connectivity
from stable_baselines3.common.monitor import Monitor

from src.environment import FixedCurriculumEnv
from src.wrappers import (
    ActionRescaleWrapper,
    ApproachPrefixResetWrapper,
    RecoveryGoalWrapper,
    TeacherPrefixResetWrapper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
EVALUATION_FIELDS = (
    "timesteps",
    "mean_return",
    "std_return",
    "mean_steps",
    "mean_displacement",
    "mean_speed",
    "mean_final_x",
    "mean_max_x",
    "mean_obstacles_cleared",
    "mean_obstacle_fraction",
    "success_rate",
)


def make_env(
    body: np.ndarray,
    level: int,
    max_steps: int,
    monitor_path: Path | None = None,
    monitor_override: bool = True,
):
    """正規化行動と時間制限を備えた一つの環境を構築する。"""
    env = FixedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = ActionRescaleWrapper(env)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=("obstacles_cleared", "is_success", "max_x_position"),
            override_existing=monitor_override,
        )
    return env


def make_teacher_prefix_env(
    body: np.ndarray,
    level: int,
    teacher_model,
    target_cleared: int,
    agent_max_steps: int,
    post_prefix_steps: int = 0,
    approach_model=None,
    approach_distance: float | None = None,
    approach_max_steps: int = 1_600,
    recovery_goal_distance: float | None = None,
    monitor_path: Path | None = None,
):
    """教師前置区間の後だけを学習時間として数える環境を構築する。"""
    course_env = FixedCurriculumEnv(
        body=body,
        level=level,
        connections=get_full_connectivity(body),
    )
    env = ActionRescaleWrapper(course_env)
    env = TeacherPrefixResetWrapper(
        env,
        teacher_model=teacher_model,
        target_cleared=target_cleared,
        max_prefix_steps=getattr(course_env.course, "max_steps"),
        post_prefix_steps=post_prefix_steps,
    )
    if approach_model is not None:
        if approach_distance is None:
            raise ValueError("使用接近策略时必须提供接近距离。")
        env = ApproachPrefixResetWrapper(
            env,
            approach_model=approach_model,
            target_distance=approach_distance,
            max_approach_steps=approach_max_steps,
        )
    if recovery_goal_distance is not None:
        env = RecoveryGoalWrapper(env, recovery_goal_distance)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=agent_max_steps)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            info_keywords=("obstacles_cleared", "is_success", "max_x_position"),
        )
    return env


def write_json(path: Path, data: dict):
    """UTF-8のJSONを読みやすい形式で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_evaluation(path: Path, row: dict):
    """評価行を既定列順のCSVへ追記する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVALUATION_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row[key] for key in EVALUATION_FIELDS})


def read_evaluations(path: Path) -> list[dict]:
    """既存の評価CSVを数値辞書のリストとして読み込む。"""
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {key: float(value) if key != "timesteps" else int(float(value)) for key, value in row.items()}
        for row in rows
    ]


def summarize_episodes(episodes: list[dict]) -> dict:
    """複数エピソードの結果を学習判定用の共通指標へ集約する。"""
    def values(key: str) -> np.ndarray:
        return np.asarray([episode[key] for episode in episodes], dtype=float)

    returns = values("return")
    steps = values("steps")
    displacement = values("displacement")
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_steps": float(np.mean(steps)),
        "mean_displacement": float(np.mean(displacement)),
        "mean_speed": float(np.mean(displacement / np.maximum(steps, 1.0))),
        "mean_final_x": float(np.mean(values("final_x"))),
        "mean_max_x": float(np.mean(values("max_x"))),
        "mean_obstacles_cleared": float(np.mean(values("obstacles_cleared"))),
        "mean_obstacle_fraction": float(np.mean(values("obstacle_fraction"))),
        "success_rate": float(np.mean(values("success"))),
    }


def evaluate_controller(
    body: np.ndarray,
    level: int,
    controller: Callable,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict:
    """一つの制御器を独立エピソードで評価し、共通指標を返す。"""
    last_error = None
    for attempt in range(2):
        env = None
        try:
            env = make_env(body, level, max_steps)
            results = []
            for episode_number in range(episodes):
                obs, reset_info = env.reset(seed=seed + episode_number)
                total_reward = 0.0
                final_info = reset_info
                completed_steps = 0
                for completed_steps in range(1, max_steps + 1):
                    action = controller(obs, env)
                    obs, reward, terminated, truncated, final_info = env.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
                results.append(
                    {
                        "return": total_reward,
                        "steps": completed_steps,
                        "displacement": final_info["forward_displacement"],
                        "final_x": final_info["x_position"],
                        "max_x": final_info["max_x_position"],
                        "obstacles_cleared": final_info["obstacles_cleared"],
                        "obstacle_fraction": final_info["obstacle_fraction"],
                        "success": float(final_info["is_success"]),
                    }
                )
            return summarize_episodes(results)
        except Exception as exc:
            last_error = exc
            if "invalid vector<bool> subscript" not in str(exc) or attempt:
                raise
            gc.collect()
        finally:
            if env is not None:
                env.close()
    raise RuntimeError("环境评估失败。") from last_error


def evaluate_ppo(model, body, level, episodes, max_steps, seed) -> dict:
    """PPOの決定論的平均行動を評価する。"""
    def controller(obs, _env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def evaluate_random(body, level, episodes, max_steps, seed) -> dict:
    """一様ランダム行動を未学習基準として評価する。"""
    rng = np.random.default_rng(seed)

    def controller(_obs, env):
        return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)

    return evaluate_controller(body, level, controller, episodes, max_steps, seed)


def score_metrics(metrics: dict) -> tuple[float, float, float, float]:
    """成功率、通過数、最遠位置、報酬の優先順で比較可能な値を返す。"""
    return (
        float(metrics["success_rate"]),
        float(metrics["mean_obstacles_cleared"]),
        float(metrics["mean_max_x"]),
        float(metrics["mean_return"]),
    )
