"""完全固定された障害物を厳格に通過するカリキュラムEvoGym環境。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from evogym import EvoWorld
from evogym.envs.traverse import StairsBase

from src.curriculum import ROBOT_START_X, ROBOT_START_Y, CourseSpec, get_course, make_course_array


REWARD_VERSION = "fixed_strict_curriculum_v8"
LANDING_STABLE_STEPS = 50
LANDING_SPEED_LIMIT = 0.12


class FixedCurriculumEnv(StairsBase):
    """静的障害物と全身通過判定を使用する一段階の学習環境。"""

    metadata = {"render_modes": ["human", "screen", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        body: np.ndarray,
        level: int,
        connections: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):
        self.course: CourseSpec = get_course(level)
        self.level = level
        self.world = EvoWorld()
        self.world.add_from_array("ground", make_course_array(level), 0, 0)
        self.world.add_from_array(
            "robot",
            body,
            ROBOT_START_X,
            ROBOT_START_Y,
            connections=connections,
        )
        super().__init__(
            world=self.world,
            render_mode=render_mode,
            render_options=render_options,
        )

        actuator_count = self.get_actuator_indices("robot").size
        robot_point_values = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 10
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
        self._maximum_com_y = 0.0
        self._maximum_bottom_y = 0.0
        self._cleared_count = 0
        self._active_obstacle = 0
        self._maximum_near_com_y = -float("inf")
        self._maximum_near_bottom_y = -float("inf")
        self._maximum_crossed_fraction = 0.0
        self._maximum_crossing_score = 0.0
        self._maximum_front_x = -float("inf")
        self._stall_steps = 0
        self._last_progress_x = 0.0
        self._landing_stable_steps = 0

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            (
                self.get_vel_com_obs("robot"),
                self.get_ort_obs("robot"),
                self.get_relative_pos_obs("robot"),
                self.get_floor_obs("robot", ["ground"], self.sight_dist),
            )
        )

    def _count_cleared(self, robot_positions: np.ndarray) -> int:
        """全身の最左端が後端を越えた固定障害物の数を返す。"""
        leftmost_x = float(np.min(robot_positions[0]))
        return sum(
            leftmost_x > (obstacle.end_x + 1) * self.VOXEL_SIZE
            for obstacle in self.course.obstacles
        )

    def _is_success(self, robot_positions: np.ndarray) -> bool:
        """現在の段階で要求される全課題を満たしたか判定する。"""
        com_x = float(np.mean(robot_positions[0]))
        if not self.course.obstacles:
            return com_x > self.course.finish_x * self.VOXEL_SIZE
        return (
            self._cleared_count == len(self.course.obstacles)
            and self._landing_stable_steps >= LANDING_STABLE_STEPS
        )

    def _update_landing_stability(self, robot_positions: np.ndarray):
        """全障害物通過後の低速接地姿勢が連続した歩数を更新する。"""
        if self._cleared_count != len(self.course.obstacles):
            self._landing_stable_steps = 0
            return
        velocity = self.get_vel_com_obs("robot")
        speed = float(np.linalg.norm(velocity))
        bottom_y = float(np.min(robot_positions[1]))
        body_height = float(np.ptp(robot_positions[1]))
        stable = (
            speed <= LANDING_SPEED_LIMIT
            and bottom_y <= 1.2 * self.VOXEL_SIZE
            and body_height >= 3.0 * self.VOXEL_SIZE
        )
        self._landing_stable_steps = self._landing_stable_steps + 1 if stable else 0

    def _info(self, robot_positions: np.ndarray, success: bool) -> dict:
        com_x = float(np.mean(robot_positions[0]))
        return {
            "curriculum_level": self.level,
            "x_position": com_x,
            "max_x_position": self._maximum_com_x,
            "forward_displacement": com_x - self._initial_com_x,
            "obstacles_cleared": self._cleared_count,
            "obstacle_fraction": (
                self._cleared_count / len(self.course.obstacles)
                if self.course.obstacles
                else 0.0
            ),
            "is_success": bool(success),
            "maximum_com_y": self._maximum_com_y,
            "maximum_bottom_y": self._maximum_bottom_y,
            "stall_steps": self._stall_steps,
            "landing_stable_steps": self._landing_stable_steps,
            "landing_stable": self._landing_stable_steps >= LANDING_STABLE_STEPS,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, _ = super().reset(seed=seed, options=options)
        positions = self.object_pos_at_time(self.get_time(), "robot")
        com_x = float(np.mean(positions[0]))
        self._initial_com_x = com_x
        self._maximum_com_x = com_x
        self._maximum_com_y = float(np.mean(positions[1]))
        self._maximum_bottom_y = float(np.min(positions[1]))
        self._cleared_count = 0
        self._active_obstacle = 0
        self._maximum_near_com_y = -float("inf")
        self._maximum_near_bottom_y = -float("inf")
        self._maximum_crossed_fraction = 0.0
        self._maximum_crossing_score = 0.0
        self._maximum_front_x = float(np.max(positions[0]))
        self._stall_steps = 0
        self._last_progress_x = com_x
        self._landing_stable_steps = 0
        return observation, self._info(positions, success=False)

    def _reset_obstacle_history(self, obstacle_index: int, positions: np.ndarray):
        """次の障害物へ進んだときに一度限りの整形報酬履歴を初期化する。"""
        self._active_obstacle = obstacle_index
        self._maximum_near_com_y = -float("inf")
        self._maximum_near_bottom_y = -float("inf")
        self._maximum_crossed_fraction = 0.0
        self._maximum_crossing_score = 0.0
        self._maximum_front_x = float(np.max(positions[0]))

    def _shaping_reward(self, positions: np.ndarray, obstacle_index: int) -> float:
        """次の固定障害物に対する上昇と部分通過の新記録だけを小さく報酬化する。"""
        if obstacle_index >= len(self.course.obstacles):
            return 0.0
        if obstacle_index != self._active_obstacle:
            self._reset_obstacle_history(obstacle_index, positions)

        obstacle = self.course.obstacles[obstacle_index]
        obstacle_start = obstacle.start_x * self.VOXEL_SIZE
        obstacle_end = (obstacle.end_x + 1) * self.VOXEL_SIZE
        com_x = float(np.mean(positions[0]))
        if not (
            obstacle_start - 4 * self.VOXEL_SIZE
            <= com_x
            <= obstacle_end + 3 * self.VOXEL_SIZE
        ):
            return 0.0

        com_y = float(np.mean(positions[1]))
        bottom_y = float(np.min(positions[1]))
        front_x = float(np.max(positions[0]))
        crossed_fraction = float(np.mean(positions[0] > obstacle_end))
        scoring_start = obstacle_start - 3 * self.VOXEL_SIZE
        scoring_end = obstacle_end + 2 * self.VOXEL_SIZE
        point_progress = np.clip(
            (positions[0] - scoring_start) / (scoring_end - scoring_start),
            0.0,
            1.0,
        )
        crossing_score = float(np.mean(point_progress))

        if self._maximum_near_com_y == -float("inf"):
            self._maximum_near_com_y = com_y
            self._maximum_near_bottom_y = bottom_y
            self._maximum_front_x = front_x

        com_height_gain = max(0.0, com_y - self._maximum_near_com_y)
        bottom_height_gain = max(0.0, bottom_y - self._maximum_near_bottom_y)
        front_gain = max(0.0, front_x - self._maximum_front_x)
        fraction_gain = max(
            0.0,
            crossed_fraction - self._maximum_crossed_fraction,
        )
        crossing_gain = max(
            0.0,
            crossing_score - self._maximum_crossing_score,
        )

        self._maximum_near_com_y = max(self._maximum_near_com_y, com_y)
        self._maximum_near_bottom_y = max(self._maximum_near_bottom_y, bottom_y)
        self._maximum_front_x = max(self._maximum_front_x, front_x)
        self._maximum_crossed_fraction = max(
            self._maximum_crossed_fraction,
            crossed_fraction,
        )
        self._maximum_crossing_score = max(
            self._maximum_crossing_score,
            crossing_score,
        )

        return (
            30.0 * com_height_gain
            + 50.0 * bottom_height_gain
            + 1.0 * front_gain
            + 20.0 * fraction_gain
            + 30.0 * crossing_gain
        )

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

        if com_x_after > self._last_progress_x + 0.01:
            self._last_progress_x = com_x_after
            self._stall_steps = 0
        else:
            self._stall_steps += 1

        cleared_before = self._cleared_count
        shaping_reward = self._shaping_reward(positions_after, cleared_before)
        cleared_after = self._count_cleared(positions_after)
        newly_cleared = max(0, cleared_after - self._cleared_count)
        self._cleared_count = max(self._cleared_count, cleared_after)
        self._update_landing_stability(positions_after)

        forward_scale = 5.0
        reward = forward_scale * (com_x_after - com_x_before)
        reward += shaping_reward
        reward += 50.0 * newly_cleared
        if self._stall_steps > 300:
            reward -= 0.001

        success = self._is_success(positions_after)
        terminated = bool(simulation_unstable or success)
        if simulation_unstable:
            reward -= 5.0
        if success:
            reward += 20.0

        observation = self._observation()
        info = self._info(positions_after, success)
        info["newly_cleared"] = newly_cleared
        info["shaping_reward"] = shaping_reward
        info["simulation_unstable"] = bool(simulation_unstable)
        return observation, float(reward), terminated, False, info
