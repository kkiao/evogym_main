"""訓練を行わず統一環境と検収規則の監査結果を保存する。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from general_terrain.environment import (
    PRIVILEGED_OBSERVATION_NAMES,
    GeneralObstacleEnv,
)
from general_terrain.terrain import sample_course


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "assets" / "environment_audit.json"


def audit_case(split: str, seed: int) -> dict[str, object]:
    """一区分の初期化、観測、二十刻み物理安定性を確認する。"""
    course = sample_course(seed, 3, 3, split)
    env = GeneralObstacleEnv(course=course, resample_on_reset=False)
    try:
        observation, info = env.reset()
        initial_finite = bool(np.all(np.isfinite(observation)))
        initial_in_space = bool(env.observation_space.contains(observation))
        action = np.full(env.action_space.shape, -0.2, dtype=np.float32)
        simulation_unstable = False
        executed_steps = 0
        for _ in range(20):
            observation, _, terminated, truncated, step_info = env.step(action)
            executed_steps += 1
            simulation_unstable = simulation_unstable or bool(
                step_info["simulation_unstable"]
            )
            if terminated or truncated:
                break
        schema_text = " ".join(env.schema)
        leaked_names = sorted(
            name for name in PRIVILEGED_OBSERVATION_NAMES if name in schema_text
        )
        return {
            "split": split,
            "course_id": info["course_id"],
            "course_width": course.width,
            "observation_shape": list(env.observation_space.shape),
            "action_shape": list(env.action_space.shape),
            "initial_observation_finite": initial_finite,
            "initial_observation_in_space": initial_in_space,
            "privileged_schema_names": leaked_names,
            "neutral_steps_executed": executed_steps,
            "simulation_unstable": simulation_unstable,
        }
    finally:
        env.close()


def main() -> None:
    """三つの分離区分を同一環境で監査しJSONへ出力する。"""
    cases = [
        audit_case("train", 801),
        audit_case("validation", 802),
        audit_case("holdout", 803),
    ]
    shared_observation_shape = len(
        {tuple(case["observation_shape"]) for case in cases}
    ) == 1
    shared_action_shape = len({tuple(case["action_shape"]) for case in cases}) == 1
    passed = bool(
        shared_observation_shape
        and shared_action_shape
        and all(case["initial_observation_finite"] for case in cases)
        and all(case["initial_observation_in_space"] for case in cases)
        and all(not case["privileged_schema_names"] for case in cases)
        and all(not case["simulation_unstable"] for case in cases)
    )
    result = {
        "audit_type": "environment_and_acceptance_no_training",
        "training_executed": False,
        "passed": passed,
        "shared_observation_shape": shared_observation_shape,
        "shared_action_shape": shared_action_shape,
        "cases": cases,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
