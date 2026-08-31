"""長脚ロボット用の固定障害物・着地・再前進判定環境。"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from evogym import EvoWorld
from evogym.envs.traverse import StairsBase

from ll7.body import BODY_WIDTH_VOXELS
from ll7.curriculum import (
    ROBOT_START_X,
    ROBOT_START_Y,
    CourseSpec,
    get_course,
    make_course_array,
)


REWARD_VERSION = "strict_clear_land_restart_v10_true_no_side_fall"
SIGHT_DISTANCE = 20
LANDING_STABLE_STEPS = 50
LANDING_MAX_STEPS = 500
RESTART_MAX_STEPS = 500
LANDING_SPEED_LIMIT = 0.15
LANDING_ANGULAR_SPEED_LIMIT = 0.18
LANDING_ANGLE_LIMIT = math.radians(35.0)
MOVING_ANGLE_LIMIT = math.radians(50.0)
UPPER_BODY_CONTACT_HEIGHT = 1.35


class LongLeggedCurriculumEnv(StairsBase):
    """完全通過、直立着地、再前進を各障害物ごとに要求する環境。"""

    metadata = {"render_modes": ["human", "screen", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        body: np.ndarray,
        level: int,
        connections: Optional[np.ndarray] = None,
        landing_angle_limit: float = LANDING_ANGLE_LIMIT,
        landing_speed_limit: float = LANDING_SPEED_LIMIT,
        render_mode: Optional[str] = None,
        render_options: Optional[Dict[str, Any]] = None,
    ):
        self.course: CourseSpec = get_course(level)
        self.level = level
        self.landing_angle_limit = landing_angle_limit
        self.landing_speed_limit = landing_speed_limit
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
        self.sight_dist = SIGHT_DISTANCE
        self.action_space = gym.spaces.Box(
            low=0.6,
            high=1.6,
            shape=(actuator_count,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(3 + robot_point_values + (2 * self.sight_dist + 1) + 5,),
            dtype=np.float64,
        )

        self._initial_com_x = 0.0
        self._maximum_com_x = 0.0
        self._maximum_com_y = 0.0
        self._maximum_bottom_y = 0.0
        self._body_height_reference = 0.0
        self._upright_reference_angle = 0.0
        self._previous_orientation = 0.0
        self._unwrapped_orientation_error = 0.0
        self._angular_speed = 0.0
        self._upper_point_indices = np.asarray([], dtype=int)
        self._active_obstacle = 0
        self._strict_cleared_count = 0
        self._stable_landing_count = 0
        self._restart_count = 0
        self._validated_count = 0
        self._phase = "approach"
        self._phase_steps = 0
        self._landing_stable_steps = 0
        self._recovery_start_x = 0.0
        self._recovery_progress = 0.0
        self._maximum_near_com_y = -float("inf")
        self._maximum_near_bottom_y = -float("inf")
        self._maximum_crossed_fraction = 0.0
        self._maximum_crossing_score = 0.0
        self._maximum_front_x = -float("inf")
        self._stall_steps = 0
        self._last_progress_x = 0.0
        self._failure_reason = ""

    def _phase_observation(self) -> np.ndarray:
        """現在の評価段階と進捗を方策へ明示する。"""
        phases = ("approach", "landing", "restart")
        one_hot = [1.0 if self._phase == phase else 0.0 for phase in phases]
        stable_fraction = min(1.0, self._landing_stable_steps / LANDING_STABLE_STEPS)
        restart_distance = BODY_WIDTH_VOXELS * self.VOXEL_SIZE
        restart_fraction = min(1.0, self._recovery_progress / restart_distance)
        return np.asarray(one_hot + [stable_fraction, restart_fraction], dtype=float)

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            (
                self.get_vel_com_obs("robot"),
                self.get_ort_obs("robot"),
                self.get_relative_pos_obs("robot"),
                self.get_floor_obs("robot", ["ground"], self.sight_dist),
                self._phase_observation(),
            )
        )

    @staticmethod
    def _wrapped_angle_difference(angle: float, reference: float) -> float:
        """二つの角度の最短符号付き差を返す。"""
        return float(math.atan2(math.sin(angle - reference), math.cos(angle - reference)))

    def _orientation_error(self) -> float:
        """初期直立姿勢からの絶対角度誤差を返す。"""
        orientation = self.object_orientation_at_time(self.get_time(), "robot")
        return abs(self._wrapped_angle_difference(orientation, self._upright_reference_angle))

    def _upper_body_min_y(self, robot_positions: np.ndarray) -> float:
        """初期上部に属する質点の現在最小高さを返す。"""
        if self._upper_point_indices.size == 0:
            return float("inf")
        return float(np.min(robot_positions[1, self._upper_point_indices]))

    def _upper_body_grounded(self, robot_positions: np.ndarray) -> bool:
        """上半身質点が地面近傍まで倒れたかを判定する。"""
        return self._upper_body_min_y(robot_positions) <= (
            UPPER_BODY_CONTACT_HEIGHT * self.VOXEL_SIZE
        )

    def _is_upright(self, robot_positions: np.ndarray, angle_limit: float) -> bool:
        """角度と身体高さの両方から横転していないか判定する。"""
        body_height = float(np.ptp(robot_positions[1]))
        return (
            self._orientation_error() <= angle_limit
            and body_height >= 0.6 * self._body_height_reference
        )

    def _strictly_cleared(self, robot_positions: np.ndarray, obstacle_index: int) -> bool:
        """身体の最左点が指定障害物の後端を完全に越えたか判定する。"""
        if obstacle_index >= len(self.course.obstacles):
            return False
        obstacle = self.course.obstacles[obstacle_index]
        obstacle_end = (obstacle.end_x + 1) * self.VOXEL_SIZE
        return float(np.min(robot_positions[0])) > obstacle_end

    def _count_raw_strict_clearances(self, robot_positions: np.ndarray) -> int:
        """姿勢検証前に幾つの障害物を物理的に完全通過したか返す。"""
        leftmost_x = float(np.min(robot_positions[0]))
        return sum(
            leftmost_x > (obstacle.end_x + 1) * self.VOXEL_SIZE
            for obstacle in self.course.obstacles
        )

    def _landing_condition(self, robot_positions: np.ndarray) -> bool:
        """障害物後方で直立・接地・低速状態にあるか判定する。"""
        if not self._strictly_cleared(robot_positions, self._active_obstacle):
            return False
        speed = float(np.linalg.norm(self.get_vel_com_obs("robot")))
        bottom_y = float(np.min(robot_positions[1]))
        return (
            speed <= self.landing_speed_limit
            and abs(self._angular_speed) <= LANDING_ANGULAR_SPEED_LIMIT
            and bottom_y <= 1.25 * self.VOXEL_SIZE
            and self._is_upright(robot_positions, self.landing_angle_limit)
            and self._restart_space_margin(robot_positions) >= 0.0
        )

    def _restart_space_margin(self, robot_positions: np.ndarray) -> float:
        """次障害物までに一身体幅の再前進を残せる余白を返す。"""
        next_index = self._active_obstacle + 1
        if next_index >= len(self.course.obstacles):
            return float("inf")
        next_start = self.course.obstacles[next_index].start_x * self.VOXEL_SIZE
        restart_distance = BODY_WIDTH_VOXELS * self.VOXEL_SIZE
        return next_start - float(np.max(robot_positions[0])) - restart_distance

    def _reset_shaping_history(self, robot_positions: np.ndarray):
        """次の障害物向けの一度限り整形報酬履歴を初期化する。"""
        self._maximum_near_com_y = -float("inf")
        self._maximum_near_bottom_y = -float("inf")
        self._maximum_crossed_fraction = 0.0
        self._maximum_crossing_score = 0.0
        self._maximum_front_x = float(np.max(robot_positions[0]))

    def _shaping_reward(self, positions: np.ndarray) -> float:
        """接近段階だけで上昇と部分通過の新記録を小さく報酬化する。"""
        if self._phase != "approach" or self._active_obstacle >= len(self.course.obstacles):
            return 0.0
        obstacle = self.course.obstacles[self._active_obstacle]
        obstacle_start = obstacle.start_x * self.VOXEL_SIZE
        obstacle_end = (obstacle.end_x + 1) * self.VOXEL_SIZE
        com_x = float(np.mean(positions[0]))
        if not (
            obstacle_start - 5 * self.VOXEL_SIZE
            <= com_x
            <= obstacle_end + 3 * self.VOXEL_SIZE
        ):
            return 0.0

        com_y = float(np.mean(positions[1]))
        bottom_y = float(np.min(positions[1]))
        front_x = float(np.max(positions[0]))
        crossed_fraction = float(np.mean(positions[0] > obstacle_end))
        scoring_start = obstacle_start - 4 * self.VOXEL_SIZE
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
        fraction_gain = max(0.0, crossed_fraction - self._maximum_crossed_fraction)
        crossing_gain = max(0.0, crossing_score - self._maximum_crossing_score)

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
            20.0 * com_height_gain
            + 35.0 * bottom_height_gain
            + 1.0 * front_gain
            + 15.0 * fraction_gain
            + 20.0 * crossing_gain
        )

    def _update_phase(self, robot_positions: np.ndarray) -> tuple[dict, bool]:
        """厳格通過から再前進までの状態機械を一歩進める。"""
        events = {
            "new_strict_clearance": False,
            "new_stable_landing": False,
            "new_restart": False,
        }
        failure = False
        if not self.course.obstacles:
            return events, failure

        self._phase_steps += 1
        if self._phase == "approach":
            if self._strictly_cleared(robot_positions, self._active_obstacle):
                self._strict_cleared_count = max(
                    self._strict_cleared_count,
                    self._active_obstacle + 1,
                )
                self._phase = "landing"
                self._phase_steps = 0
                self._landing_stable_steps = 0
                events["new_strict_clearance"] = True

        elif self._phase == "landing":
            if self._landing_condition(robot_positions):
                self._landing_stable_steps += 1
            else:
                self._landing_stable_steps = 0
            if self._landing_stable_steps >= LANDING_STABLE_STEPS:
                self._stable_landing_count = max(
                    self._stable_landing_count,
                    self._active_obstacle + 1,
                )
                self._phase = "restart"
                self._phase_steps = 0
                self._recovery_start_x = float(np.mean(robot_positions[0]))
                self._recovery_progress = 0.0
                events["new_stable_landing"] = True
            elif self._phase_steps >= LANDING_MAX_STEPS:
                self._failure_reason = "landing_timeout"
                failure = True

        elif self._phase == "restart":
            com_x = float(np.mean(robot_positions[0]))
            self._recovery_progress = max(0.0, com_x - self._recovery_start_x)
            target_distance = BODY_WIDTH_VOXELS * self.VOXEL_SIZE
            restart_valid = (
                self._recovery_progress >= target_distance
                and self._strictly_cleared(robot_positions, self._active_obstacle)
                and self._is_upright(robot_positions, MOVING_ANGLE_LIMIT)
            )
            if self._active_obstacle + 1 < len(self.course.obstacles):
                next_start = (
                    self.course.obstacles[self._active_obstacle + 1].start_x
                    * self.VOXEL_SIZE
                )
                restart_valid = restart_valid and float(
                    np.max(robot_positions[0])
                ) < next_start
            if restart_valid:
                self._restart_count = max(self._restart_count, self._active_obstacle + 1)
                self._validated_count = self._active_obstacle + 1
                self._active_obstacle += 1
                self._phase_steps = 0
                self._landing_stable_steps = 0
                self._recovery_progress = 0.0
                events["new_restart"] = True
                if self._active_obstacle >= len(self.course.obstacles):
                    self._phase = "completed"
                else:
                    self._phase = "approach"
                    self._reset_shaping_history(robot_positions)
            elif self._phase_steps >= RESTART_MAX_STEPS:
                self._failure_reason = "restart_timeout"
                failure = True

        return events, failure

    def _is_success(self, robot_positions: np.ndarray) -> bool:
        """平地終点または全障害物の完全検証を達成したか判定する。"""
        if not self.course.obstacles:
            return float(np.mean(robot_positions[0])) > self.course.finish_x * self.VOXEL_SIZE
        return self._validated_count == len(self.course.obstacles)

    def _info(self, robot_positions: np.ndarray, success: bool) -> dict:
        """評価と可視化に必要な厳格指標を返す。"""
        com_x = float(np.mean(robot_positions[0]))
        speed = float(np.linalg.norm(self.get_vel_com_obs("robot")))
        total = len(self.course.obstacles)
        return {
            "curriculum_level": self.level,
            "x_position": com_x,
            "max_x_position": self._maximum_com_x,
            "forward_displacement": com_x - self._initial_com_x,
            "active_obstacle": self._active_obstacle,
            "phase": self._phase,
            "phase_steps": self._phase_steps,
            "strict_clearances": self._strict_cleared_count,
            "stable_landings": self._stable_landing_count,
            "restart_successes": self._restart_count,
            "validated_obstacles": self._validated_count,
            "obstacles_cleared": self._validated_count,
            "obstacle_fraction": self._validated_count / total if total else 0.0,
            "is_success": bool(success),
            "orientation_error": self._orientation_error(),
            "unwrapped_orientation_error": abs(self._unwrapped_orientation_error),
            "landing_angle_limit": self.landing_angle_limit,
            "landing_speed_limit": self.landing_speed_limit,
            "is_upright": self._is_upright(robot_positions, MOVING_ANGLE_LIMIT),
            "angular_speed": self._angular_speed,
            "com_speed": speed,
            "bottom_y": float(np.min(robot_positions[1])),
            "body_height": float(np.ptp(robot_positions[1])),
            "upper_body_min_y": self._upper_body_min_y(robot_positions),
            "upper_body_grounded": self._upper_body_grounded(robot_positions),
            "upper_body_contact_height": UPPER_BODY_CONTACT_HEIGHT * self.VOXEL_SIZE,
            "maximum_com_y": self._maximum_com_y,
            "maximum_bottom_y": self._maximum_bottom_y,
            "maximum_crossed_fraction": self._maximum_crossed_fraction,
            "maximum_crossing_score": self._maximum_crossing_score,
            "stall_steps": self._stall_steps,
            "landing_stable_steps": self._landing_stable_steps,
            "recovery_progress": self._recovery_progress,
            "restart_space_margin": self._restart_space_margin(robot_positions),
            "failure_reason": self._failure_reason,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)
        positions = self.object_pos_at_time(self.get_time(), "robot")
        com_x = float(np.mean(positions[0]))
        orientation = self.object_orientation_at_time(self.get_time(), "robot")
        self._initial_com_x = com_x
        self._maximum_com_x = com_x
        self._maximum_com_y = float(np.mean(positions[1]))
        self._maximum_bottom_y = float(np.min(positions[1]))
        self._body_height_reference = float(np.ptp(positions[1]))
        self._upright_reference_angle = orientation
        self._previous_orientation = orientation
        self._unwrapped_orientation_error = 0.0
        self._angular_speed = 0.0
        minimum_y = float(np.min(positions[1]))
        # 脚部を除く胴体下縁まで含め、角だけで地面を支える抜け道を防ぐ。
        upper_cutoff = minimum_y + 0.45 * self._body_height_reference
        self._upper_point_indices = np.flatnonzero(positions[1] >= upper_cutoff)
        self._active_obstacle = 0
        self._strict_cleared_count = 0
        self._stable_landing_count = 0
        self._restart_count = 0
        self._validated_count = 0
        self._phase = "approach"
        self._phase_steps = 0
        self._landing_stable_steps = 0
        self._recovery_start_x = com_x
        self._recovery_progress = 0.0
        self._stall_steps = 0
        self._last_progress_x = com_x
        self._failure_reason = ""
        self._reset_shaping_history(positions)
        return self._observation(), self._info(positions, success=False)

    def step(self, action):
        positions_before = self.object_pos_at_time(self.get_time(), "robot")
        com_x_before = float(np.mean(positions_before[0]))
        speed_before = float(np.linalg.norm(self.get_vel_com_obs("robot")))
        orientation_before = self.object_orientation_at_time(self.get_time(), "robot")
        orientation_error_before = abs(
            self._wrapped_angle_difference(
                orientation_before,
                self._upright_reference_angle,
            )
        )
        phase_before = self._phase
        simulation_unstable = super().step({"robot": action})
        positions_after = self.object_pos_at_time(self.get_time(), "robot")
        com_x_after = float(np.mean(positions_after[0]))
        current_orientation = self.object_orientation_at_time(self.get_time(), "robot")
        self._angular_speed = self._wrapped_angle_difference(
            current_orientation,
            self._previous_orientation,
        )
        self._unwrapped_orientation_error += self._angular_speed
        self._previous_orientation = current_orientation

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

        shaping_reward = self._shaping_reward(positions_after)
        events, phase_failure = self._update_phase(positions_after)
        raw_strict = self._count_raw_strict_clearances(positions_after)
        self._strict_cleared_count = max(self._strict_cleared_count, raw_strict)
        skipped_future_obstacle = raw_strict > self._active_obstacle + 1
        if skipped_future_obstacle:
            self._failure_reason = "skipped_unvalidated_obstacle"
            phase_failure = True

        delta_x = float(np.clip(com_x_after - com_x_before, -0.05, 0.05))
        forward_scale = {"approach": 6.0, "landing": 0.0, "restart": 8.0}.get(
            phase_before,
            2.0,
        )
        reward = forward_scale * delta_x + shaping_reward
        if (
            phase_before == "approach"
            and self._active_obstacle < len(self.course.obstacles)
        ):
            obstacle = self.course.obstacles[self._active_obstacle]
            obstacle_start = obstacle.start_x * self.VOXEL_SIZE
            obstacle_end = (obstacle.end_x + 1) * self.VOXEL_SIZE
            if obstacle_start - 3 * self.VOXEL_SIZE <= com_x_before <= obstacle_end:
                orientation_improvement = np.clip(
                    orientation_error_before - self._orientation_error(),
                    -0.2,
                    0.2,
                )
                reward += 3.0 * float(orientation_improvement)
        if phase_before == "landing":
            speed_after = float(np.linalg.norm(self.get_vel_com_obs("robot")))
            speed_improvement = np.clip(speed_before - speed_after, -0.5, 0.5)
            reward += 1.5 * float(speed_improvement)
            speed_excess = max(0.0, speed_after - self.landing_speed_limit)
            reward -= 0.01 * min(speed_excess, 10.0)
            orientation_improvement = np.clip(
                orientation_error_before - self._orientation_error(),
                -0.2,
                0.2,
            )
            reward += 12.0 * float(orientation_improvement)
            orientation_error = self._orientation_error()
            angle_excess = max(0.0, orientation_error - self.landing_angle_limit)
            reward += 0.25 * math.cos(min(math.pi, orientation_error))
            reward -= 0.50 * angle_excess / self.landing_angle_limit
            space_deficit = max(0.0, -self._restart_space_margin(positions_after))
            reward -= 0.20 * space_deficit / self.VOXEL_SIZE
            # 条件ごとの小報酬で、疎な安定着地判定までの方向を示す。
            reward += 0.03 if speed_after <= self.landing_speed_limit else 0.0
            reward += 0.05 if self._is_upright(
                positions_after,
                self.landing_angle_limit,
            ) else 0.0
            reward += 0.05 if self._restart_space_margin(positions_after) >= 0.0 else 0.0
            reward += 0.10 if self._landing_condition(positions_after) else -0.005
        if phase_before == "restart":
            orientation_improvement = np.clip(
                orientation_error_before - self._orientation_error(),
                -0.2,
                0.2,
            )
            reward += 8.0 * float(orientation_improvement)
            moving_angle_excess = max(
                0.0,
                self._orientation_error() - MOVING_ANGLE_LIMIT,
            )
            reward -= 0.10 * moving_angle_excess / MOVING_ANGLE_LIMIT
            if not self._is_upright(positions_after, MOVING_ANGLE_LIMIT):
                reward -= 0.02
        if events["new_strict_clearance"]:
            reward += 40.0 + 40.0 * math.cos(self._orientation_error())
        if events["new_stable_landing"]:
            reward += 50.0
        if events["new_restart"]:
            reward += 60.0
        if self._stall_steps > 350:
            reward -= 0.002

        success = self._is_success(positions_after)
        terminated = bool(simulation_unstable or phase_failure or success)
        if simulation_unstable:
            self._failure_reason = "simulation_unstable"
            reward -= 10.0
        if phase_failure:
            reward -= 20.0
        if success:
            reward += 100.0

        observation = self._observation()
        info = self._info(positions_after, success)
        info.update(events)
        info["shaping_reward"] = shaping_reward
        info["simulation_unstable"] = bool(simulation_unstable)
        info["skipped_future_obstacle"] = bool(skipped_future_obstacle)
        return observation, float(reward), terminated, False, info
