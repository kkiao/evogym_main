"""ソフトロボットが任意の方法で連続混合障害物を通過するEvoGym環境。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from evogym import EvoWorld
from evogym.envs.traverse import StairsBase

from src.course import (
    FINISH_X,
    GROUNDED_ROBOT_START_Y,
    LEGACY_ROBOT_START_Y,
    OBSTACLES,
    ROBOT_START_X,
    make_course_array,
    rigid_obstacle_components,
)


LEGACY_ENVIRONMENT_VERSION = "baseline_v1_airdrop"
SHAPED_ENVIRONMENT_VERSION = "grounded_shaping_v2"
TRANSFER_ENVIRONMENT_VERSION = "grounded_transfer_shaping_v3"
DENSE_CROSSING_ENVIRONMENT_VERSION = "grounded_dense_crossing_v4"
MOVABLE_OBSTACLE_ENVIRONMENT_VERSION = "movable_mixed_obstacles_v5"
MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION = "movable_com_clear_v6"
ENVIRONMENT_VERSIONS = (
    LEGACY_ENVIRONMENT_VERSION,
    SHAPED_ENVIRONMENT_VERSION,
    TRANSFER_ENVIRONMENT_VERSION,
    DENSE_CROSSING_ENVIRONMENT_VERSION,
    MOVABLE_OBSTACLE_ENVIRONMENT_VERSION,
    MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
)


class MixedObstacleCourseEnv(StairsBase):
    """七組の障害物コース。通過方法を限定せず、回転や登攀も有効とする。"""

    metadata = {"render_modes": ["human", "screen", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        body: np.ndarray,
        connections: Optional[np.ndarray] = None,
        environment_version: str = SHAPED_ENVIRONMENT_VERSION,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):
        if environment_version not in ENVIRONMENT_VERSIONS:
            raise ValueError(f"未知环境版本：{environment_version}")
        self.environment_version = environment_version
        self.use_obstacle_shaping = environment_version != LEGACY_ENVIRONMENT_VERSION
        self.use_advanced_shaping = environment_version in {
            TRANSFER_ENVIRONMENT_VERSION,
            DENSE_CROSSING_ENVIRONMENT_VERSION,
        }
        self.use_dense_crossing = environment_version == DENSE_CROSSING_ENVIRONMENT_VERSION
        self.use_movable_obstacles = environment_version in {
            MOVABLE_OBSTACLE_ENVIRONMENT_VERSION,
            MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION,
        }
        self.use_com_clearance = (
            environment_version == MOVABLE_COM_CLEAR_ENVIRONMENT_VERSION
        )
        robot_start_y = (
            GROUNDED_ROBOT_START_Y
            if self.use_obstacle_shaping
            else LEGACY_ROBOT_START_Y
        )

        self.world = EvoWorld()
        self.world.add_from_array(
            "ground",
            make_course_array(include_task_obstacles=not self.use_movable_obstacles),
            0,
            0,
        )
        self.terrain_names = ["ground"]
        if self.use_movable_obstacles:
            for index, obstacle in enumerate(OBSTACLES):
                for component_number, (offset, component) in enumerate(
                    rigid_obstacle_components(obstacle),
                    1,
                ):
                    obstacle_name = f"obstacle_{index + 1}_{component_number}"
                    self.world.add_from_array(
                        obstacle_name,
                        component,
                        obstacle.start_x + offset,
                        1,
                    )
                    self.terrain_names.append(obstacle_name)
        self.world.add_from_array(
            "robot",
            body,
            ROBOT_START_X,
            robot_start_y,
            connections=connections,
        )
        super().__init__(
            world=self.world,
            render_mode=render_mode,
            render_options=render_options,
        )

        actuator_count = self.get_actuator_indices("robot").size
        robot_point_values = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 8
        self.action_space = gym.spaces.Box(
            low=0.6,
            high=1.6,
            shape=(actuator_count,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(3 + robot_point_values + (2 * self.sight_dist + 1),),
            dtype=np.float64,
        )

        self._initial_com_x = 0.0
        self._maximum_com_x = 0.0
        self._cleared_count = 0
        self._active_shaping_obstacle = 0
        self._maximum_near_obstacle_y = -float("inf")
        self._maximum_front_x = -float("inf")
        self._maximum_com_y = -float("inf")
        self._maximum_bottom_y = -float("inf")
        self._maximum_near_bottom_y = -float("inf")
        self._maximum_crossed_fraction = 0.0
        self._maximum_crossing_score = 0.0

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            (
                self.get_vel_com_obs("robot"),
                self.get_ort_obs("robot"),
                self.get_relative_pos_obs("robot"),
                self.get_floor_obs("robot", self.terrain_names, self.sight_dist),
            )
        )

    def _count_cleared(self, robot_positions: np.ndarray) -> int:
        # 可動障害物では押し出しや転倒を許し、重心が元の後端を越えれば通過とみなす。
        # 固定壁の対照実験では、全身の最左端が越える厳格な基準を使用する。
        progress_x = (
            float(np.mean(robot_positions[0]))
            if self.use_com_clearance
            else float(np.min(robot_positions[0]))
        )
        return sum(
            progress_x > (obstacle.end_x + 1) * self.VOXEL_SIZE
            for obstacle in OBSTACLES
        )

    def _info(self, robot_positions: np.ndarray, success: bool) -> dict:
        com_x = float(np.mean(robot_positions[0]))
        # エピソード中の最大通過数を報告し、通過後の一時的な後退で実績を消さない。
        cleared = self._cleared_count
        return {
            "x_position": com_x,
            "max_x_position": self._maximum_com_x,
            "forward_displacement": com_x - self._initial_com_x,
            "obstacles_cleared": cleared,
            "obstacle_fraction": cleared / len(OBSTACLES),
            "is_success": bool(success),
            "maximum_com_y": self._maximum_com_y,
            "maximum_bottom_y": self._maximum_bottom_y,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, _ = super().reset(seed=seed, options=options)
        positions = self.object_pos_at_time(self.get_time(), "robot")
        self._initial_com_x = float(np.mean(positions[0]))
        self._maximum_com_x = self._initial_com_x
        self._cleared_count = 0
        self._active_shaping_obstacle = 0
        self._maximum_near_obstacle_y = -float("inf")
        self._maximum_front_x = float(np.max(positions[0]))
        self._maximum_com_y = float(np.mean(positions[1]))
        self._maximum_bottom_y = float(np.min(positions[1]))
        self._maximum_near_bottom_y = -float("inf")
        self._maximum_crossed_fraction = 0.0
        self._maximum_crossing_score = 0.0
        return observation, self._info(positions, success=False)

    def _obstacle_shaping_reward(
        self,
        robot_positions: np.ndarray,
        obstacle_index: int,
    ) -> float:
        """障害物付近で初めて更新した高さと前端だけを報酬化し、往復による稼ぎを防ぐ。"""
        if not self.use_obstacle_shaping or obstacle_index >= len(OBSTACLES):
            return 0.0
        if obstacle_index != self._active_shaping_obstacle:
            self._active_shaping_obstacle = obstacle_index
            self._maximum_near_obstacle_y = -float("inf")
            self._maximum_front_x = float(np.max(robot_positions[0]))
            self._maximum_near_bottom_y = -float("inf")
            self._maximum_crossed_fraction = 0.0
            self._maximum_crossing_score = 0.0

        obstacle = OBSTACLES[obstacle_index]
        com_x = float(np.mean(robot_positions[0]))
        com_y = float(np.mean(robot_positions[1]))
        front_x = float(np.max(robot_positions[0]))
        obstacle_start = obstacle.start_x * self.VOXEL_SIZE
        obstacle_end = (obstacle.end_x + 1) * self.VOXEL_SIZE

        # 重心が次の障害物から約四マス以内に入ったとき、上昇・回転・登攀の探索を促す。
        near_obstacle = obstacle_start - 4 * self.VOXEL_SIZE <= com_x <= (
            obstacle_end + 3 * self.VOXEL_SIZE
        )
        reward = 0.0
        if near_obstacle:
            if self._maximum_near_obstacle_y == -float("inf"):
                self._maximum_near_obstacle_y = com_y
            height_gain = max(0.0, com_y - self._maximum_near_obstacle_y)
            self._maximum_near_obstacle_y = max(self._maximum_near_obstacle_y, com_y)
            reward += (8.0 if self.use_advanced_shaping else 4.0) * height_gain

            front_gain = max(0.0, front_x - self._maximum_front_x)
            self._maximum_front_x = max(self._maximum_front_x, front_x)
            reward += 0.25 * front_gain

            if self.use_advanced_shaping:
                bottom_y = float(np.min(robot_positions[1]))
                if self._maximum_near_bottom_y == -float("inf"):
                    self._maximum_near_bottom_y = bottom_y
                bottom_gain = max(0.0, bottom_y - self._maximum_near_bottom_y)
                self._maximum_near_bottom_y = max(self._maximum_near_bottom_y, bottom_y)
                reward += 12.0 * bottom_gain

                crossed_fraction = float(np.mean(robot_positions[0] > obstacle_end))
                fraction_gain = max(
                    0.0,
                    crossed_fraction - self._maximum_crossed_fraction,
                )
                self._maximum_crossed_fraction = max(
                    self._maximum_crossed_fraction,
                    crossed_fraction,
                )
                reward += 4.0 * fraction_gain

                if self.use_dense_crossing:
                    # 各身体点が障害物の前方から後方へ進む連続的な進捗を採点する。
                    # 過去最高値の更新だけを報酬化し、往復で繰り返し稼げないようにする。
                    scoring_start = obstacle_start - 3 * self.VOXEL_SIZE
                    scoring_end = obstacle_end + 2 * self.VOXEL_SIZE
                    point_progress = np.clip(
                        (robot_positions[0] - scoring_start)
                        / (scoring_end - scoring_start),
                        0.0,
                        1.0,
                    )
                    crossing_score = float(np.mean(point_progress))
                    crossing_gain = max(
                        0.0,
                        crossing_score - self._maximum_crossing_score,
                    )
                    self._maximum_crossing_score = max(
                        self._maximum_crossing_score,
                        crossing_score,
                    )
                    reward += 8.0 * crossing_gain
        return reward

    def step(self, action):
        positions_before = self.object_pos_at_time(self.get_time(), "robot")
        com_x_before = float(np.mean(positions_before[0]))

        simulation_unstable = super().step({"robot": action})

        positions_after = self.object_pos_at_time(self.get_time(), "robot")
        com_x_after = float(np.mean(positions_after[0]))
        self._maximum_com_x = max(self._maximum_com_x, com_x_after)
        self._maximum_com_y = max(
            self._maximum_com_y,
            float(np.mean(positions_after[1])),
        )
        self._maximum_bottom_y = max(
            self._maximum_bottom_y,
            float(np.min(positions_after[1])),
        )

        cleared_before = self._cleared_count
        shaping_reward = self._obstacle_shaping_reward(positions_after, cleared_before)
        cleared_after = self._count_cleared(positions_after)
        newly_cleared = max(0, cleared_after - self._cleared_count)
        self._cleared_count = max(self._cleared_count, cleared_after)

        # 基本変位報酬は公式Hurdlerに合わせ、一度限りの通過報酬を疎な節目として与える。
        reward = com_x_after - com_x_before
        reward += shaping_reward
        reward += (2.0 if self.use_obstacle_shaping else 0.5) * newly_cleared

        success = self._cleared_count == len(OBSTACLES) and (
            com_x_after > FINISH_X * self.VOXEL_SIZE
        )
        terminated = bool(simulation_unstable or success)
        if simulation_unstable:
            reward -= 3.0
        if success:
            reward += 3.0

        observation = self._observation()
        info = self._info(positions_after, success=success)
        info["newly_cleared"] = newly_cleared
        info["shaping_reward"] = shaping_reward
        info["simulation_unstable"] = bool(simulation_unstable)
        return observation, float(reward), terminated, False, info
