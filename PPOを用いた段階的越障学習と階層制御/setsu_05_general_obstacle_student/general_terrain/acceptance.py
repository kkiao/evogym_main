"""地図に依存しない通過、姿勢回復、完走判定を提供する。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from general_terrain.body import BODY_WIDTH_VOXELS
from general_terrain.terrain import CourseSpec


@dataclass(frozen=True)
class AcceptanceConfig:
    """速度を止めずに安全性を確認する共通判定閾値を保持する。"""

    voxel_size: float = 0.1
    recovery_stable_steps: int = 20
    recovery_distance_voxels: int = BODY_WIDTH_VOXELS
    upright_angle_limit: float = math.radians(45.0)
    hard_fall_angle_limit: float = math.radians(80.0)
    hard_fall_grace_steps: int = 5
    upper_body_ground_grace_steps: int = 75


@dataclass(frozen=True)
class AcceptanceSnapshot:
    """一時刻の検収状態を不変データとして返す。"""

    raw_clearances: int
    recovered_obstacles: int
    obstacle_fraction: float
    course_complete: bool
    hard_fall: bool
    sequence_failed: bool
    failure_reason: str
    current_safe_streak: int
    current_recovery_distance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_clearances": self.raw_clearances,
            "recovered_obstacles": self.recovered_obstacles,
            "obstacle_fraction": self.obstacle_fraction,
            "course_complete": self.course_complete,
            "hard_fall": self.hard_fall,
            "sequence_failed": self.sequence_failed,
            "failure_reason": self.failure_reason,
            "current_safe_streak": self.current_safe_streak,
            "current_recovery_distance": self.current_recovery_distance,
        }


class CourseAcceptanceTracker:
    """障害物名や訓練段階を方策へ渡さず軌跡だけを採点する。"""

    def __init__(
        self,
        course: CourseSpec,
        config: AcceptanceConfig = AcceptanceConfig(),
    ) -> None:
        self.course = course
        self.config = config
        self.reset()

    def reset(self) -> None:
        """一回分の内部集計を初期状態へ戻す。"""
        count = len(self.course.obstacles)
        self._cleared = [False] * count
        self._recovered = [False] * count
        self._clear_com_x = [0.0] * count
        self._safe_streak = [0] * count
        self._recovery_distance = [0.0] * count
        self._fall_angle_steps = 0
        self._upper_body_ground_steps = 0
        self._hard_fall = False
        self._sequence_failed = False
        self._failure_reason = ""

    def _raw_clearances(self, robot_positions: np.ndarray) -> int:
        """身体全体が後端を越えた障害物数を返す。"""
        leftmost_x = float(np.min(robot_positions[0]))
        return sum(
            leftmost_x > (obstacle.end_x + 1) * self.config.voxel_size
            for obstacle in self.course.obstacles
        )

    def _snapshot(self, course_complete: bool) -> AcceptanceSnapshot:
        """現在の集計値を公開用形式へ変換する。"""
        total = len(self.course.obstacles)
        recovered = sum(self._recovered)
        active = min(recovered, max(0, total - 1)) if total else 0
        return AcceptanceSnapshot(
            raw_clearances=sum(self._cleared),
            recovered_obstacles=recovered,
            obstacle_fraction=recovered / total if total else 1.0,
            course_complete=course_complete,
            hard_fall=self._hard_fall,
            sequence_failed=self._sequence_failed,
            failure_reason=self._failure_reason,
            current_safe_streak=self._safe_streak[active] if total else 0,
            current_recovery_distance=(
                self._recovery_distance[active] if total else 0.0
            ),
        )

    def update(
        self,
        robot_positions: np.ndarray,
        *,
        orientation_error: float,
        upper_body_grounded: bool,
        simulation_unstable: bool = False,
    ) -> AcceptanceSnapshot:
        """物理状態を一歩取り込み、完全通過後の回復を検証する。"""
        if simulation_unstable:
            self._hard_fall = True
            self._failure_reason = "simulation_unstable"

        if orientation_error >= self.config.hard_fall_angle_limit:
            self._fall_angle_steps += 1
        else:
            self._fall_angle_steps = 0
        if upper_body_grounded:
            self._upper_body_ground_steps += 1
        else:
            self._upper_body_ground_steps = 0

        if self._fall_angle_steps >= self.config.hard_fall_grace_steps:
            self._hard_fall = True
            self._failure_reason = self._failure_reason or "orientation_hard_fall"
        if (
            self._upper_body_ground_steps
            >= self.config.upper_body_ground_grace_steps
        ):
            self._hard_fall = True
            self._failure_reason = self._failure_reason or "upper_body_grounded"

        com_x = float(np.mean(robot_positions[0]))
        raw_count = self._raw_clearances(robot_positions)
        for index in range(raw_count):
            if not self._cleared[index]:
                self._cleared[index] = True
                self._clear_com_x[index] = com_x

        safe_pose = (
            orientation_error <= self.config.upright_angle_limit
            and not upper_body_grounded
        )
        recovery_target = (
            self.config.recovery_distance_voxels * self.config.voxel_size
        )
        active_index = sum(self._recovered)
        if active_index < len(self.course.obstacles):
            if raw_count > active_index + 1:
                self._sequence_failed = True
                self._failure_reason = (
                    self._failure_reason or "next_obstacle_before_recovery"
                )
            if self._cleared[active_index] and not self._sequence_failed:
                self._recovery_distance[active_index] = max(
                    self._recovery_distance[active_index],
                    com_x - self._clear_com_x[active_index],
                )
                self._safe_streak[active_index] = (
                    self._safe_streak[active_index] + 1 if safe_pose else 0
                )
                if (
                    self._safe_streak[active_index]
                    >= self.config.recovery_stable_steps
                    and self._recovery_distance[active_index] >= recovery_target
                ):
                    self._recovered[active_index] = True

        reached_finish = com_x >= self.course.finish_x * self.config.voxel_size
        all_recovered = all(self._recovered)
        course_complete = bool(
            reached_finish
            and all_recovered
            and safe_pose
            and not self._hard_fall
            and not self._sequence_failed
        )
        return self._snapshot(course_complete)

    def mark_timeout(self, reason: str = "time_limit") -> AcceptanceSnapshot:
        """時間または停滞上限による未完走理由を記録する。"""
        if not self._failure_reason:
            self._failure_reason = reason
        return self._snapshot(course_complete=False)
