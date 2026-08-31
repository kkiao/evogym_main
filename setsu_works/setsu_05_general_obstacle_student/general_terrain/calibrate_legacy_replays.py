"""既存三種類のLevel 2軌跡を再生し検収閾値を較正する。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ll7.body import make_body as make_legacy_body
from ll7.curriculum import get_course
from ll7.experiment import make_env
from ll7.final_true_noroll_v2 import (
    ACTION_SCALES as SAFE_ACTION_SCALES,
    MODEL_PATHS as SAFE_MODEL_PATHS,
    select_stage as select_safe_stage,
)
from ll7.render_level2_no_roll_gifs import APPROACH, NODES, ModelCache, predict
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "calibration" / "legacy_replay_calibration.json"
FRAME_RATE = 50


def state_row(step: int, info: dict) -> dict[str, object]:
    """旧環境の一時刻を較正用の最小状態へ変換する。"""
    return {
        "step": step,
        "angle_degrees": math.degrees(float(info["orientation_error"])),
        "upper_body_grounded": bool(info.get("upper_body_grounded", False)),
        "upper_body_min_y": float(info.get("upper_body_min_y", float("inf"))),
        "strict_clearances": int(info["strict_clearances"]),
        "validated_obstacles": int(info["validated_obstacles"]),
        "new_strict_clearance": bool(info.get("new_strict_clearance", False)),
        "new_restart": bool(info.get("new_restart", False)),
        "phase": str(info["phase"]),
        "x_position": float(info["x_position"]),
    }


def run_legacy_node(cache: ModelCache, node: dict) -> list[dict[str, object]]:
    """旧九段階図の一構成を全物理刻みで決定論的に再生する。"""
    approach = cache.get(APPROACH)
    first_clear = cache.get(node["first_clear"])
    first_land = cache.get(node["first_land"])
    first_restart = cache.get(node["first_restart"])
    second_clear = cache.get(node.get("second_clear", node["first_clear"]))
    second_land = cache.get(
        node.get("second_land", node.get("second_clear", node["first_land"]))
    )
    second_restart = cache.get(node.get("second_restart", node["first_restart"]))
    course = get_course(2)
    env = make_env(make_legacy_body(), 2, course.max_steps)
    rows = []
    try:
        observation, info = env.reset(seed=7)
        rows.append(state_row(0, info))
        for step in range(1, course.max_steps + 1):
            if int(info["validated_obstacles"]) < 1:
                if info["phase"] == "landing":
                    model = first_land
                elif info["phase"] == "restart":
                    model = first_restart
                else:
                    obstacle_x = (
                        env.unwrapped.course.obstacles[0].start_x
                        * env.unwrapped.VOXEL_SIZE
                    )
                    model = (
                        approach
                        if obstacle_x - float(info["x_position"]) > 0.25
                        else first_clear
                    )
            elif int(info["strict_clearances"]) < 2:
                model = second_clear
            elif info["phase"] == "landing":
                model = second_land
            else:
                model = second_restart
            observation, _, terminated, truncated, info = env.step(
                predict(model, observation)
            )
            rows.append(state_row(step, info))
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows


def load_safe_models() -> dict[str, PPO]:
    """真正無側倒制御鎖の保存方策をCPUへ読み込む。"""
    return {
        name: PPO.load(path.resolve(), device="cpu")
        for name, path in SAFE_MODEL_PATHS.items()
    }


def run_safe_success() -> list[dict[str, object]]:
    """上半身非接地の最終成功軌跡を全物理刻みで再生する。"""
    models = load_safe_models()
    course = get_course(2)
    env = make_env(make_legacy_body(), 2, course.max_steps)
    second_stage = "second_to_33"
    second_handoff_started = False
    rows = []
    try:
        observation, info = env.reset(seed=10_000)
        rows.append(state_row(0, info))
        for step in range(1, course.max_steps + 1):
            stage, second_handoff_started, second_stage = select_safe_stage(
                info,
                course,
                env,
                second_handoff_started,
                second_stage,
            )
            model_key = (
                "approach"
                if stage in {"first_approach", "second_approach"}
                else stage
            )
            action, _ = models[model_key].predict(observation, deterministic=True)
            action_scale = SAFE_ACTION_SCALES.get(stage, 1.0)
            observation, _, terminated, truncated, info = env.step(
                action_scale * action
            )
            rows.append(state_row(step, info))
            if terminated or truncated:
                break
    finally:
        env.close()
    return rows


def longest_streak(rows: list[dict[str, object]], predicate) -> int:
    """条件を満たす最長連続物理刻み数を返す。"""
    longest = 0
    current = 0
    for row in rows:
        if predicate(row):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def recovery_windows(
    rows: list[dict[str, object]],
    angle_limit: float,
    contact_height: float,
) -> list[dict[str, int]]:
    """各完全通過後から次完全通過前までの安全姿勢最長窓を返す。"""
    clearance_indices = [
        index for index, row in enumerate(rows) if row["new_strict_clearance"]
    ]
    results = []
    for obstacle_index, start in enumerate(clearance_indices):
        end = (
            clearance_indices[obstacle_index + 1]
            if obstacle_index + 1 < len(clearance_indices)
            else len(rows)
        )
        segment = rows[start:end]
        safe_longest = longest_streak(
            segment,
            lambda row: (
                float(row["angle_degrees"]) <= angle_limit
                and float(row["upper_body_min_y"]) > contact_height
            ),
        )
        results.append(
            {
                "obstacle": obstacle_index + 1,
                "clearance_step": int(rows[start]["step"]),
                "safe_streak_steps": safe_longest,
            }
        )
    return results


def summarize_trace(
    label: str,
    expected_class: str,
    rows: list[dict[str, object]],
    *,
    stable_steps: int,
    stable_angle: float,
    hard_angle: float,
    hard_angle_steps: int,
    contact_steps: int,
    contact_height: float,
) -> dict[str, object]:
    """一軌跡へ候補閾値を適用し期待分類との一致を返す。"""
    angle_streak = longest_streak(
        rows,
        lambda row: float(row["angle_degrees"]) >= hard_angle,
    )
    grounded_streak = longest_streak(
        rows,
        lambda row: float(row["upper_body_min_y"]) <= contact_height,
    )
    windows = recovery_windows(rows, stable_angle, contact_height)
    hard_fall = angle_streak >= hard_angle_steps or grounded_streak >= contact_steps
    recovery_pass = bool(windows) and all(
        row["safe_streak_steps"] >= stable_steps for row in windows
    )
    classified = "safe_success" if recovery_pass and not hard_fall else "unsafe"
    expected = "safe_success" if expected_class == "safe_success" else "unsafe"
    return {
        "label": label,
        "expected_class": expected_class,
        "steps": int(rows[-1]["step"]),
        "maximum_angle_degrees": max(float(row["angle_degrees"]) for row in rows),
        "minimum_upper_body_y": min(float(row["upper_body_min_y"]) for row in rows),
        "maximum_hard_angle_streak_steps": angle_streak,
        "maximum_hard_angle_streak_seconds": angle_streak / FRAME_RATE,
        "maximum_upper_body_ground_streak_steps": grounded_streak,
        "maximum_upper_body_ground_streak_seconds": grounded_streak / FRAME_RATE,
        "recovery_windows": windows,
        "hard_fall": hard_fall,
        "recovery_pass": recovery_pass,
        "classified_as": classified,
        "matches_expected": classified == expected,
    }


def main() -> None:
    """三軌跡を再生して採用可否と余裕をJSONへ保存する。"""
    parser = argparse.ArgumentParser(description="用旧回放校准统一验收阈值。")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    cache = ModelCache()
    traces = (
        ("upright_success", "safe_success", run_safe_success()),
        ("side_fall_false_positive", "side_fall", run_legacy_node(cache, NODES[-1])),
        ("roll_then_recover", "roll_recovery", run_legacy_node(cache, NODES[0])),
    )
    thresholds = {
        "stable_steps": 20,
        "stable_seconds": 20 / FRAME_RATE,
        "stable_angle_degrees": 45.0,
        "hard_fall_angle_degrees": 80.0,
        "hard_fall_angle_steps": 5,
        "hard_fall_angle_seconds": 5 / FRAME_RATE,
        "upper_body_ground_steps": 75,
        "upper_body_ground_seconds": 75 / FRAME_RATE,
        "upper_body_contact_height": 0.135,
        "frame_rate": FRAME_RATE,
    }
    summaries = [
        summarize_trace(
            label,
            expected,
            rows,
            stable_steps=thresholds["stable_steps"],
            stable_angle=thresholds["stable_angle_degrees"],
            hard_angle=thresholds["hard_fall_angle_degrees"],
            hard_angle_steps=thresholds["hard_fall_angle_steps"],
            contact_steps=thresholds["upper_body_ground_steps"],
            contact_height=thresholds["upper_body_contact_height"],
        )
        for label, expected, rows in traces
    ]
    result = {
        "calibration_source": "existing_level2_deterministic_replays",
        "training_executed": False,
        "thresholds": thresholds,
        "traces": summaries,
        "accepted": all(row["matches_expected"] for row in summaries),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
