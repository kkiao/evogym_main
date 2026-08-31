"""手続き型コースを同一方策インターフェースで実行する環境。"""

from __future__ import annotations

import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from evogym import EvoWorld, get_full_connectivity
from evogym.envs.base import EvoGymBase
from evogym.envs.traverse import StairsBase

from general_terrain.acceptance import AcceptanceConfig, CourseAcceptanceTracker
from general_terrain.body import BODY_HEIGHT_VOXELS, make_body
from general_terrain.feasibility import assert_course_feasible
from general_terrain.terrain import (
    ROBOT_START_X,
    ROBOT_START_Y,
    CourseSpec,
    make_course_array,
    sample_course,
)


ENVIRONMENT_VERSION = "general_relative_terrain_v1"
REWARD_VERSION = "generic_progress_safety_v4_rear_clearance_and_stall_reset"
TERRAIN_LOOK_BEHIND = 5
TERRAIN_LOOK_AHEAD = 30
TERRAIN_SCAN_SIZE = TERRAIN_LOOK_BEHIND + TERRAIN_LOOK_AHEAD + 1
UPPER_BODY_CONTACT_HEIGHT_VOXELS = 1.35
DEFAULT_MAX_STEPS_PER_VOXEL = 35
STALL_TERMINATION_STEPS = 250
PRIVILEGED_OBSERVATION_NAMES = frozenset(
    {
        "course_id",
        "split",
        "seed",
        "difficulty",
        "absolute_x",
        "active_obstacle",
        "obstacle_index",
        "phase",
        "finish_x",
    }
)


def observation_schema(
    robot_point_values: int,
    actuator_count: int,
) -> tuple[str, ...]:
    """観測ベクトルの各成分名を固定順で返す。"""
    names = [
        "com_velocity_x",
        "com_velocity_y",
        "orientation_sin",
        "orientation_cos",
        "angular_velocity",
    ]
    names.extend(f"relative_body_point_{index}" for index in range(robot_point_values))
    names.extend(
        f"relative_terrain_height_{offset:+d}"
        for offset in range(-TERRAIN_LOOK_BEHIND, TERRAIN_LOOK_AHEAD + 1)
    )
    names.extend(f"previous_action_{index}" for index in range(actuator_count))
    return tuple(names)


class GeneralObstacleEnv(StairsBase):
    """固定地図情報を観測へ含めない統一ランダム越障環境。"""

    metadata = {"render_modes": ["human", "screen", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        course: CourseSpec | None = None,
        *,
        split: str = "train",
        difficulty: int = 1,
        obstacle_count: int = 1,
        base_seed: int = 0,
        resample_on_reset: bool = True,
        max_episode_steps: int | None = None,
        render_mode: Optional[str] = None,
        render_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.split = split
        self.difficulty = difficulty
        self.obstacle_count = obstacle_count
        self.base_seed = base_seed
        self.resample_on_reset = resample_on_reset and course is None
        self._episode_index = 0
        self._first_reset = True
        self._explicit_max_episode_steps = max_episode_steps
        self._render_mode_requested = render_mode
        self._render_options_requested = render_options
        self.body = make_body()
        self.connections = get_full_connectivity(self.body)
        self.course = course or sample_course(
            base_seed,
            difficulty,
            obstacle_count,
            split,
        )
        assert_course_feasible(self.course)
        self.world = self._make_world(self.course)
        super().__init__(
            world=self.world,
            render_mode=render_mode,
            render_options=render_options,
        )

        actuator_count = self.get_actuator_indices("robot").size
        robot_point_values = self.object_pos_at_time(self.get_time(), "robot").size
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(actuator_count,),
            dtype=np.float32,
        )
        self._observation_schema = observation_schema(robot_point_values, actuator_count)
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(self._observation_schema),),
            dtype=np.float32,
        )
        self.acceptance_config = AcceptanceConfig(voxel_size=self.VOXEL_SIZE)
        self._acceptance = CourseAcceptanceTracker(self.course, self.acceptance_config)
        self._surface_heights = self._make_surface_heights(self.course)
        self._max_episode_steps = self._resolve_max_episode_steps()
        self._step_count = 0
        self._initial_com_x = 0.0
        self._maximum_com_x = 0.0
        self._body_height_reference = BODY_HEIGHT_VOXELS * self.VOXEL_SIZE
        self._upright_reference_angle = 0.0
        self._previous_action = np.zeros(actuator_count, dtype=np.float32)
        self._previous_orientation = 0.0
        self._angular_velocity = 0.0
        self._last_progress_x = 0.0
        self._stall_steps = 0
        self._local_rise_key: tuple[int, int] | None = None
        self._maximum_local_bottom_y = 0.0
        self._maximum_local_crossing_fraction = 0.0
        self._maximum_local_rear_progress = 0.0
        self._last_local_shaping_reward = 0.0
        self._upper_point_indices = np.asarray([], dtype=int)
        self._last_snapshot = None

    def _make_world(self, course: CourseSpec) -> EvoWorld:
        """一つのCourseSpecから静的地形と同一形状を構築する。"""
        world = EvoWorld()
        world.add_from_array("ground", make_course_array(course), 0, 0)
        world.add_from_array(
            "robot",
            self.body,
            ROBOT_START_X,
            ROBOT_START_Y,
            connections=self.connections,
        )
        return world

    @staticmethod
    def _make_surface_heights(course: CourseSpec) -> np.ndarray:
        """各列の地表高さを体素数で事前計算する。"""
        heights = np.ones(course.width, dtype=np.float32)
        for obstacle in course.obstacles:
            for offset, height in enumerate(obstacle.template.heights):
                heights[obstacle.start_x + offset] += float(height)
        return heights

    def _resolve_max_episode_steps(self) -> int:
        """明示上限またはコース長比例の安全な時間上限を返す。"""
        if self._explicit_max_episode_steps is not None:
            return int(self._explicit_max_episode_steps)
        return max(1000, self.course.width * DEFAULT_MAX_STEPS_PER_VOXEL)

    def _replace_course(self, course: CourseSpec) -> None:
        """ビューアを閉じ、同じインターフェースの新世界へ交換する。"""
        assert_course_feasible(course)
        self.default_viewer.close()
        self.course = course
        self.world = self._make_world(course)
        StairsBase.__init__(
            self,
            world=self.world,
            render_mode=self._render_mode_requested,
            render_options=self._render_options_requested,
        )
        self._surface_heights = self._make_surface_heights(course)
        self._acceptance = CourseAcceptanceTracker(course, self.acceptance_config)
        self._max_episode_steps = self._resolve_max_episode_steps()

    def _sample_episode_course(self, seed: int) -> CourseSpec:
        """設定済み分布から再現可能な一回分を抽出する。"""
        return sample_course(
            seed,
            self.difficulty,
            self.obstacle_count,
            self.split,
        )

    @staticmethod
    def _wrapped_angle_difference(angle: float, reference: float) -> float:
        """二角間の最短符号付き差を返す。"""
        return float(math.atan2(math.sin(angle - reference), math.cos(angle - reference)))

    def _orientation_error_signed(self) -> float:
        """初期直立姿勢からの符号付き角度誤差を返す。"""
        orientation = self.object_orientation_at_time(self.get_time(), "robot")
        return self._wrapped_angle_difference(orientation, self._upright_reference_angle)

    def _upper_body_grounded(self, positions: np.ndarray) -> bool:
        """初期上部に属する質点が地面近傍へ倒れたかを返す。"""
        if self._upper_point_indices.size == 0:
            return False
        minimum = float(np.min(positions[1, self._upper_point_indices]))
        return minimum <= UPPER_BODY_CONTACT_HEIGHT_VOXELS * self.VOXEL_SIZE

    def _relative_terrain_scan(self, positions: np.ndarray) -> np.ndarray:
        """絶対位置を除いた前後地表高さ走査を生成する。"""
        com_x = float(np.mean(positions[0]))
        com_y = float(np.mean(positions[1]))
        center_column = int(math.floor(com_x / self.VOXEL_SIZE))
        values = []
        for offset in range(-TERRAIN_LOOK_BEHIND, TERRAIN_LOOK_AHEAD + 1):
            column = center_column + offset
            if column < 0 or column >= self.course.width:
                values.append(0.5)
                continue
            surface_y = self._surface_heights[column] * self.VOXEL_SIZE
            relative_distance = com_y - surface_y
            values.append(float(np.clip(relative_distance, -0.5, 0.5)))
        return np.asarray(values, dtype=np.float32)

    def _observation(self, positions: np.ndarray | None = None) -> np.ndarray:
        """身体状態と相対地形だけから方策観測を構成する。"""
        if positions is None:
            positions = self.object_pos_at_time(self.get_time(), "robot")
        angle_error = self._orientation_error_signed()
        observation = np.concatenate(
            (
                self.get_vel_com_obs("robot"),
                np.asarray([math.sin(angle_error), math.cos(angle_error)]),
                np.asarray([self._angular_velocity]),
                self.get_relative_pos_obs("robot"),
                self._relative_terrain_scan(positions),
                self._previous_action,
            )
        )
        return observation.astype(np.float32)

    def _native_action(self, action: np.ndarray) -> np.ndarray:
        """正規化行動をEvoGymの伸縮倍率へ線形変換する。"""
        normalized = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        return 0.6 + 0.5 * (normalized + 1.0)

    def _local_rise_metrics(
        self,
        positions: np.ndarray,
    ) -> tuple[tuple[int, int] | None, float, float, float]:
        """近傍隆起、質点通過率、後端進捗、最低高さを返す。"""
        left_column = int(math.floor(float(np.min(positions[0])) / self.VOXEL_SIZE))
        front_column = int(math.floor(float(np.max(positions[0])) / self.VOXEL_SIZE))
        search_start = max(0, left_column - 2)
        search_end = min(self.course.width - 1, front_column + 8)
        raised = [
            column
            for column in range(search_start, search_end + 1)
            if self._surface_heights[column] > 1.0
        ]
        if not raised:
            return None, 0.0, 0.0, float(np.min(positions[1]))
        first = raised[0]
        while first - 1 >= 0 and self._surface_heights[first - 1] > 1.0:
            first -= 1
        end = first
        while (
            end + 1 < self.course.width
            and self._surface_heights[end + 1] > 1.0
        ):
            end += 1
        if float(np.min(positions[0])) > (end + 1) * self.VOXEL_SIZE:
            return None, 0.0, 0.0, float(np.min(positions[1]))
        crossing_fraction = float(
            np.mean(positions[0] > (end + 1) * self.VOXEL_SIZE)
        )
        rear_start = max(0, first - 2) * self.VOXEL_SIZE
        rear_target = (end + 1) * self.VOXEL_SIZE
        rear_progress = float(
            np.clip(
                (float(np.min(positions[0])) - rear_start)
                / max(rear_target - rear_start, self.VOXEL_SIZE),
                0.0,
                1.0,
            )
        )
        return (
            (first, end),
            crossing_fraction,
            rear_progress,
            float(np.min(positions[1])),
        )

    def _local_rise_shaping(self, positions: np.ndarray) -> float:
        """局所隆起付近の新しい持上げと通過割合だけを一度報酬化する。"""
        rise_key, crossing_fraction, rear_progress, bottom_y = (
            self._local_rise_metrics(positions)
        )
        if rise_key is None:
            self._local_rise_key = None
            self._maximum_local_bottom_y = bottom_y
            self._maximum_local_crossing_fraction = 0.0
            self._maximum_local_rear_progress = 0.0
            return 0.0
        if rise_key != self._local_rise_key:
            self._local_rise_key = rise_key
            self._maximum_local_bottom_y = bottom_y
            self._maximum_local_crossing_fraction = crossing_fraction
            self._maximum_local_rear_progress = rear_progress
            return 0.0
        bottom_gain = max(0.0, bottom_y - self._maximum_local_bottom_y)
        crossing_gain = max(
            0.0,
            crossing_fraction - self._maximum_local_crossing_fraction,
        )
        rear_gain = max(
            0.0,
            rear_progress - self._maximum_local_rear_progress,
        )
        self._maximum_local_bottom_y = max(self._maximum_local_bottom_y, bottom_y)
        self._maximum_local_crossing_fraction = max(
            self._maximum_local_crossing_fraction,
            crossing_fraction,
        )
        self._maximum_local_rear_progress = max(
            self._maximum_local_rear_progress,
            rear_progress,
        )
        return 10.0 * bottom_gain + 2.0 * crossing_gain + 6.0 * rear_gain

    def _diagnostic_info(
        self,
        positions: np.ndarray,
        *,
        simulation_unstable: bool,
        truncated: bool,
    ) -> dict[str, object]:
        """方策入力から分離した再現・採点情報を返す。"""
        snapshot = self._last_snapshot
        com_x = float(np.mean(positions[0]))
        info: dict[str, object] = {
            "environment_version": ENVIRONMENT_VERSION,
            "reward_version": REWARD_VERSION,
            "course_id": self.course.course_id,
            "course_split": self.course.split,
            "course_seed": self.course.seed,
            "course_difficulty": self.course.difficulty,
            "obstacle_count": len(self.course.obstacles),
            "step_count": self._step_count,
            "max_episode_steps": self._max_episode_steps,
            "x_position": com_x,
            "forward_displacement": com_x - self._initial_com_x,
            "max_x_position": self._maximum_com_x,
            "stall_steps": self._stall_steps,
            "orientation_error": abs(self._orientation_error_signed()),
            "angular_velocity": self._angular_velocity,
            "upper_body_grounded": self._upper_body_grounded(positions),
            "simulation_unstable": bool(simulation_unstable),
            "time_limit_reached": bool(truncated),
            "stall_limit_reached": bool(
                self._stall_steps >= STALL_TERMINATION_STEPS
            ),
            "local_rise_shaping_reward": self._last_local_shaping_reward,
        }
        if snapshot is not None:
            info.update(snapshot.as_dict())
        return info

    @property
    def schema(self) -> tuple[str, ...]:
        """観測監査用の読み取り専用成分名を返す。"""
        return self._observation_schema

    def reset(self, seed: int | None = None, options: dict | None = None):
        """必要なら新コースを抽出し、一回分の判定履歴を初期化する。"""
        gym.Env.reset(self, seed=seed)
        requested_course = options.get("course") if options else None
        should_replace = requested_course is not None
        if requested_course is not None and not isinstance(requested_course, CourseSpec):
            raise TypeError("options['course'] 必须是 CourseSpec。")

        if requested_course is None and self.resample_on_reset:
            if self._first_reset and seed is None:
                episode_seed = self.base_seed
            else:
                episode_seed = seed if seed is not None else self.base_seed + self._episode_index
            sampled = self._sample_episode_course(int(episode_seed))
            should_replace = sampled.as_dict() != self.course.as_dict()
            requested_course = sampled

        if should_replace:
            self._replace_course(requested_course)
        else:
            EvoGymBase.reset(self, seed=seed, options=options)

        self._first_reset = False
        self._episode_index += 1
        self._step_count = 0
        positions = self.object_pos_at_time(self.get_time(), "robot")
        com_x = float(np.mean(positions[0]))
        self._initial_com_x = com_x
        self._maximum_com_x = com_x
        self._body_height_reference = max(float(np.ptp(positions[1])), 1e-6)
        self._upright_reference_angle = self.object_orientation_at_time(
            self.get_time(),
            "robot",
        )
        self._previous_orientation = self._upright_reference_angle
        self._angular_velocity = 0.0
        minimum_y = float(np.min(positions[1]))
        upper_cutoff = minimum_y + 0.45 * self._body_height_reference
        self._upper_point_indices = np.flatnonzero(positions[1] >= upper_cutoff)
        self._previous_action.fill(0.0)
        self._last_progress_x = com_x
        self._stall_steps = 0
        self._local_rise_key = None
        self._maximum_local_bottom_y = float(np.min(positions[1]))
        self._maximum_local_crossing_fraction = 0.0
        self._maximum_local_rear_progress = 0.0
        self._last_local_shaping_reward = 0.0
        self._acceptance.reset()
        self._last_snapshot = self._acceptance.update(
            positions,
            orientation_error=0.0,
            upper_body_grounded=self._upper_body_grounded(positions),
        )
        observation = self._observation(positions)
        info = self._diagnostic_info(
            positions,
            simulation_unstable=False,
            truncated=False,
        )
        return observation, info

    def step(self, action: np.ndarray):
        """一物理刻みを進め、汎用報酬と独立検収結果を返す。"""
        normalized_action = np.clip(
            np.asarray(action, dtype=np.float32),
            -1.0,
            1.0,
        )
        positions_before = self.object_pos_at_time(self.get_time(), "robot")
        com_x_before = float(np.mean(positions_before[0]))
        simulation_unstable = super().step({"robot": self._native_action(normalized_action)})
        self._step_count += 1

        positions_after = self.object_pos_at_time(self.get_time(), "robot")
        com_x_after = float(np.mean(positions_after[0]))
        current_orientation = self.object_orientation_at_time(self.get_time(), "robot")
        self._angular_velocity = self._wrapped_angle_difference(
            current_orientation,
            self._previous_orientation,
        )
        self._previous_orientation = current_orientation
        self._maximum_com_x = max(self._maximum_com_x, com_x_after)
        if com_x_after > self._last_progress_x + 0.002:
            self._last_progress_x = com_x_after
            self._stall_steps = 0
        else:
            self._stall_steps += 1
        orientation_error = abs(self._orientation_error_signed())
        upper_body_grounded = self._upper_body_grounded(positions_after)
        self._last_snapshot = self._acceptance.update(
            positions_after,
            orientation_error=orientation_error,
            upper_body_grounded=upper_body_grounded,
            simulation_unstable=bool(simulation_unstable),
        )

        delta_x = float(np.clip(com_x_after - com_x_before, -0.05, 0.05))
        upright_score = math.cos(min(math.pi, orientation_error))
        effort = float(np.mean(normalized_action**2))
        action_change = float(np.mean((normalized_action - self._previous_action) ** 2))
        self._last_local_shaping_reward = self._local_rise_shaping(positions_after)
        reward = 10.0 * delta_x + self._last_local_shaping_reward - 0.001
        reward -= 0.002 * (1.0 - upright_score)
        reward -= 0.0002 * effort + 0.0005 * action_change
        if self._stall_steps >= 50:
            reward -= 0.002
        if upper_body_grounded:
            reward -= 0.05
        if self._last_snapshot.hard_fall:
            reward -= 5.0
        if self._last_snapshot.sequence_failed:
            reward -= 2.0
        if self._last_snapshot.course_complete:
            reward += 10.0
        self._previous_action = normalized_action.copy()

        terminated = bool(
            simulation_unstable
            or self._last_snapshot.hard_fall
            or self._last_snapshot.sequence_failed
            or self._last_snapshot.course_complete
        )
        stall_limit_reached = self._stall_steps >= STALL_TERMINATION_STEPS
        truncated = bool(
            (
                self._step_count >= self._max_episode_steps
                or stall_limit_reached
            )
            and not terminated
        )
        if truncated:
            incomplete_reason = "stall_limit" if stall_limit_reached else "time_limit"
            self._last_snapshot = self._acceptance.mark_timeout(incomplete_reason)

        observation = self._observation(positions_after)
        info = self._diagnostic_info(
            positions_after,
            simulation_unstable=bool(simulation_unstable),
            truncated=truncated,
        )
        return observation, float(reward), terminated, truncated, info
