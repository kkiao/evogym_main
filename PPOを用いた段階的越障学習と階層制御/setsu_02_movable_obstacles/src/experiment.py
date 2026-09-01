"""環境構築・評価・実験ファイル管理のためのツール。"""

from __future__ import annotations

import csv
import gc
from datetime import datetime, timezone
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from src.environment import MixedObstacleCourseEnv
from src.environment import MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION
from src.wrappers import NormalizeActionSpace


PROJECT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_DIR / "runs"
EVALUATION_FIELDS = (
    "timesteps",
    "mean_return",
    "std_return",
    "min_return",
    "max_return",
    "mean_steps",
    "mean_displacement",
    "mean_speed",
    "mean_final_x",
    "mean_max_x",
    "mean_obstacles_cleared",
    "mean_obstacle_fraction",
    "success_rate",
)
MONITOR_INFO_FIELDS = (
    "x_position",
    "max_x_position",
    "forward_displacement",
    "obstacles_cleared",
    "obstacle_fraction",
    "is_success",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run_name(run_name: str) -> str:
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name 必须是单个非空目录名，不能包含路径。")
    return run_name


def write_json_atomic(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_model_atomic(model, path: Path) -> None:
    temporary_path = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary_path)
    temporary_path.replace(path)


def make_env(
    body: np.ndarray,
    max_steps: int,
    render_mode: str | None = None,
    monitor_path: Path | None = None,
    append_monitor: bool = False,
    environment_version: str = MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
):
    env = MixedObstacleCourseEnv(
        body=body,
        environment_version=environment_version,
        render_mode=render_mode,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = NormalizeActionSpace(env)
    if monitor_path is not None:
        env = Monitor(
            env,
            filename=str(monitor_path),
            allow_early_resets=True,
            override_existing=not append_monitor,
            info_keywords=MONITOR_INFO_FIELDS,
        )
    return env


def summarize_episodes(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("至少需要一个评估回合。")

    def values(key: str) -> np.ndarray:
        return np.asarray([episode[key] for episode in episodes], dtype=float)

    returns = values("return")
    steps = values("steps")
    displacement = values("displacement")
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_steps": float(steps.mean()),
        "mean_displacement": float(displacement.mean()),
        "mean_speed": float(np.mean(displacement / np.maximum(steps, 1.0))),
        "mean_final_x": float(values("final_x").mean()),
        "mean_max_x": float(values("max_x").mean()),
        "mean_obstacles_cleared": float(values("obstacles_cleared").mean()),
        "mean_obstacle_fraction": float(values("obstacle_fraction").mean()),
        "success_rate": float(values("success").mean()),
    }


def evaluate_controller(
    body: np.ndarray,
    controller,
    episodes: int,
    max_steps: int,
    seed: int,
    environment_version: str = MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
) -> dict:
    """controller(obs, env)は正規化済みの行動を返す。"""
    try:
        env = make_env(body, max_steps, environment_version=environment_version)
    except IndexError as error:
        # Windows版EvoGymでは、多物体シミュレータの連続破棄・生成時に資源解放が遅れることがある。
        # 既知の特定エラーだけを対象にGC後一度再試行し、それ以外の例外はそのまま送出する。
        if "invalid vector<bool> subscript" not in str(error):
            raise
        gc.collect()
        env = make_env(body, max_steps, environment_version=environment_version)
    results = []
    try:
        for episode_number in range(episodes):
            obs, reset_info = env.reset(seed=seed + episode_number)
            initial_x = float(reset_info["x_position"])
            total_return = 0.0
            final_info = reset_info
            steps = 0
            for _ in range(max_steps):
                action = controller(obs, env)
                obs, reward, terminated, truncated, final_info = env.step(action)
                total_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            final_x = float(final_info["x_position"])
            results.append(
                {
                    "return": total_return,
                    "steps": steps,
                    "displacement": final_x - initial_x,
                    "final_x": final_x,
                    "max_x": float(final_info["max_x_position"]),
                    "obstacles_cleared": int(final_info["obstacles_cleared"]),
                    "obstacle_fraction": float(final_info["obstacle_fraction"]),
                    "success": bool(final_info["is_success"]),
                }
            )
    finally:
        env.close()
    return summarize_episodes(results)


def evaluate_ppo(
    model,
    body: np.ndarray,
    episodes: int,
    max_steps: int,
    seed: int,
    environment_version: str = MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
) -> dict:
    def controller(obs, _env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return evaluate_controller(
        body, controller, episodes, max_steps, seed, environment_version
    )


def evaluate_random_actions(
    body: np.ndarray,
    episodes: int,
    max_steps: int,
    seed: int,
    environment_version: str = MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
) -> dict:
    rng = np.random.default_rng(seed)

    def controller(_obs, env):
        return rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)

    return evaluate_controller(
        body, controller, episodes, max_steps, seed, environment_version
    )


def ensure_evaluation_csv(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=EVALUATION_FIELDS).writeheader()


def append_evaluation(path: Path, timesteps: int, metrics: dict) -> None:
    row = {"timesteps": timesteps}
    row.update({name: metrics[name] for name in EVALUATION_FIELDS if name != "timesteps"})
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=EVALUATION_FIELDS).writerow(row)
