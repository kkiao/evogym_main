"""M6L位置二十成功に対する地図外観測の因果効果を二方向介入で検証する。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO

from general_terrain.curriculum import sample_curriculum_course
from general_terrain.environment import (
    GeneralObstacleEnv,
    TERRAIN_LOOK_AHEAD,
    TERRAIN_LOOK_BEHIND,
)
from general_terrain.render_m6l_best_student_showcase import DEFAULT_MODEL, file_sha256
from general_terrain.terrain import CourseSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "boundary_causal_validation_recheck"
)
BoundaryMode = Literal["sentinel", "flat"]


class BoundaryInterventionEnv(GeneralObstacleEnv):
    """物理地形を変えず仮想境界より先の走査表現だけを介入する。"""

    def __init__(
        self,
        *,
        course: CourseSpec,
        virtual_boundary_column: int,
        boundary_mode: BoundaryMode,
        render: bool,
    ) -> None:
        self.virtual_boundary_column = int(virtual_boundary_column)
        self.boundary_mode = boundary_mode
        super().__init__(
            course=course,
            resample_on_reset=False,
            render_mode="rgb_array" if render else None,
        )

    def _relative_terrain_scan(self, positions: np.ndarray) -> np.ndarray:
        """仮想境界外を固定標識または仮想平地として符号化する。"""
        com_x = float(np.mean(positions[0]))
        com_y = float(np.mean(positions[1]))
        center_column = int(math.floor(com_x / self.VOXEL_SIZE))
        values = []
        for offset in range(-TERRAIN_LOOK_BEHIND, TERRAIN_LOOK_AHEAD + 1):
            column = center_column + offset
            if column < 0:
                values.append(0.5)
                continue
            if column >= self.virtual_boundary_column:
                if self.boundary_mode == "sentinel":
                    values.append(0.5)
                else:
                    relative_distance = com_y - self.VOXEL_SIZE
                    values.append(float(np.clip(relative_distance, -0.5, 0.5)))
                continue
            surface_y = self._surface_heights[column] * self.VOXEL_SIZE
            relative_distance = com_y - surface_y
            values.append(float(np.clip(relative_distance, -0.5, 0.5)))
        return np.asarray(values, dtype=np.float32)


def make_courses() -> tuple[CourseSpec, CourseSpec]:
    """第一壁を同一に保った短地図と長平地地図を構築する。"""
    short_course = sample_curriculum_course(
        200_004,
        "hurdle_single",
        "validation",
    )
    long_course = replace(
        short_course,
        course_id="m6l_boundary_causal_long_flat_x20",
        split="boundary_causal_long_flat",
        width=106,
        finish_x=101,
    )
    return short_course, long_course


def run_condition(
    model: PPO,
    *,
    name: str,
    course: CourseSpec,
    virtual_boundary_column: int,
    boundary_mode: BoundaryMode,
    output_path: Path,
    frame_skip: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """一つの境界介入を第一壁回復または失敗まで決定論的に実行する。"""
    environment = BoundaryInterventionEnv(
        course=course,
        virtual_boundary_column=virtual_boundary_column,
        boundary_mode=boundary_mode,
        render=True,
    )
    frames: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    first_boundary_visible_step = -1
    first_boundary_visible_x = -1.0
    try:
        observation, info = environment.reset(seed=200_004)
        environment.default_viewer.set_target_rps(None)
        environment.default_viewer.set_resolution((600, 300))
        first_frame = environment.render()
        if first_frame is not None:
            frames.extend([np.asarray(first_frame).copy()] * 6)
        steps = 0
        recovered = False
        terminated = False
        truncated = False
        while not (terminated or truncated or recovered):
            center_column = int(
                math.floor(float(info["x_position"]) / environment.VOXEL_SIZE)
            )
            if (
                first_boundary_visible_step < 0
                and center_column + TERRAIN_LOOK_AHEAD
                >= virtual_boundary_column
            ):
                first_boundary_visible_step = steps
                first_boundary_visible_x = float(info["x_position"])
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            action, _ = model.predict(observation, deterministic=True)
            normalized_action = np.asarray(action, dtype=np.float32).copy()
            actions.append(normalized_action)
            observation, _, terminated, truncated, info = environment.step(
                normalized_action
            )
            steps += 1
            recovered = int(info["recovered_obstacles"]) >= 1
            if steps % frame_skip == 0 or terminated or truncated or recovered:
                frame = environment.render()
                if frame is not None:
                    frames.append(np.asarray(frame).copy())
        final_frame = environment.render()
        if final_frame is not None:
            frames.extend([np.asarray(final_frame).copy()] * 12)
    finally:
        environment.close()
    if not frames:
        raise RuntimeError(f"境界介入回放の画像が空である: {name}")
    imageio.mimsave(output_path, frames, fps=12, loop=0)
    row = {
        "name": name,
        "file": output_path.name,
        "physical_course_width": int(course.width),
        "physical_finish_x": int(course.finish_x),
        "virtual_boundary_column": virtual_boundary_column,
        "boundary_mode": boundary_mode,
        "steps": steps,
        "frames": len(frames),
        "first_boundary_visible_step": first_boundary_visible_step,
        "first_boundary_visible_x": first_boundary_visible_x,
        "first_hurdle_recovered": recovered,
        "course_complete": bool(info["course_complete"]),
        "raw_clearances": int(info["raw_clearances"]),
        "recovered_obstacles": int(info["recovered_obstacles"]),
        "hard_fall": bool(info["hard_fall"]),
        "stall_limit_reached": bool(info["stall_limit_reached"]),
        "failure_reason": str(info["failure_reason"]),
        "maximum_com_x": float(info["max_x_position"]),
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
    }
    return (
        row,
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def compare_traces(
    left_observations: np.ndarray,
    left_actions: np.ndarray,
    right_observations: np.ndarray,
    right_actions: np.ndarray,
) -> dict[str, object]:
    """二軌跡の共通区間における観測と動作の逐次差を返す。"""
    common = min(len(left_actions), len(right_actions))
    observation_delta = np.max(
        np.abs(left_observations[:common] - right_observations[:common]),
        axis=1,
    )
    action_delta = np.max(
        np.abs(left_actions[:common] - right_actions[:common]),
        axis=1,
    )
    differing_observation = np.flatnonzero(observation_delta > 1e-7)
    differing_action = np.flatnonzero(action_delta > 1e-7)
    return {
        "left_steps": len(left_actions),
        "right_steps": len(right_actions),
        "common_steps": common,
        "maximum_observation_absolute_difference": float(
            np.max(observation_delta) if common else 0.0
        ),
        "maximum_action_absolute_difference": float(
            np.max(action_delta) if common else 0.0
        ),
        "first_observation_difference_step": (
            int(differing_observation[0]) if differing_observation.size else -1
        ),
        "first_action_difference_step": (
            int(differing_action[0]) if differing_action.size else -1
        ),
        "traces_exactly_equal": bool(
            len(left_actions) == len(right_actions)
            and np.array_equal(left_observations, right_observations)
            and np.array_equal(left_actions, right_actions)
        ),
    }


def main() -> None:
    """四条件の双方向介入、軌跡等価性、因果判定を保存する。"""
    parser = argparse.ArgumentParser(
        description="M6L成功に対する地図外標識の因果効果を検証する。"
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame-skip", type=int, default=5)
    args = parser.parse_args()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    model_hash_before = file_sha256(model_path)
    model = PPO.load(model_path, device="cpu")
    short_course, long_course = make_courses()
    virtual_boundary = int(short_course.width)
    specifications = (
        (
            "short_physics_sentinel_observation",
            short_course,
            "sentinel",
            "01_short_native_sentinel.gif",
        ),
        (
            "short_physics_flat_observation",
            short_course,
            "flat",
            "02_short_counterfactual_flat.gif",
        ),
        (
            "long_physics_flat_observation",
            long_course,
            "flat",
            "03_long_native_flat.gif",
        ),
        (
            "long_physics_sentinel_observation",
            long_course,
            "sentinel",
            "04_long_counterfactual_sentinel.gif",
        ),
    )
    rows: dict[str, dict[str, object]] = {}
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, course, mode, filename in specifications:
        row, observations, actions = run_condition(
            model,
            name=name,
            course=course,
            virtual_boundary_column=virtual_boundary,
            boundary_mode=mode,
            output_path=output_dir / filename,
            frame_skip=int(args.frame_skip),
        )
        rows[name] = row
        traces[name] = (observations, actions)
    sentinel_equivalence = compare_traces(
        *traces["short_physics_sentinel_observation"],
        *traces["long_physics_sentinel_observation"],
    )
    flat_equivalence = compare_traces(
        *traces["short_physics_flat_observation"],
        *traces["long_physics_flat_observation"],
    )
    short_intervention = compare_traces(
        *traces["short_physics_sentinel_observation"],
        *traces["short_physics_flat_observation"],
    )
    long_intervention = compare_traces(
        *traces["long_physics_flat_observation"],
        *traces["long_physics_sentinel_observation"],
    )
    sentinel_success = all(
        bool(rows[name]["first_hurdle_recovered"])
        for name in (
            "short_physics_sentinel_observation",
            "long_physics_sentinel_observation",
        )
    )
    flat_failure = all(
        not bool(rows[name]["first_hurdle_recovered"])
        for name in (
            "short_physics_flat_observation",
            "long_physics_flat_observation",
        )
    )
    causal_pattern_confirmed = bool(
        sentinel_success
        and flat_failure
        and bool(sentinel_equivalence["traces_exactly_equal"])
        and bool(flat_equivalence["traces_exactly_equal"])
    )
    model_hash_after = file_sha256(model_path)
    if model_hash_after != model_hash_before:
        raise RuntimeError("M6L最良学生モデルが因果検証中に変更された。")
    summary = {
        "method": "two_by_two_bidirectional_boundary_observation_intervention",
        "model_path": str(model_path),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "model_unchanged": True,
        "training_steps": 0,
        "weight_updates": 0,
        "teacher_module_loaded": False,
        "teacher_interventions": 0,
        "controlled_factors": {
            "first_obstacle_start": int(short_course.obstacles[0].start_x),
            "first_obstacle_template": short_course.obstacles[0].template.name,
            "virtual_boundary_column": virtual_boundary,
            "short_physical_width": int(short_course.width),
            "long_physical_width": int(long_course.width),
            "terrain_look_ahead_voxels": TERRAIN_LOOK_AHEAD,
        },
        "conditions": rows,
        "same_sentinel_cross_physics_comparison": sentinel_equivalence,
        "same_flat_cross_physics_comparison": flat_equivalence,
        "short_physics_observation_intervention": short_intervention,
        "long_physics_observation_intervention": long_intervention,
        "sentinel_conditions_both_recovered": sentinel_success,
        "flat_conditions_both_failed": flat_failure,
        "causal_pattern_confirmed": causal_pattern_confirmed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
