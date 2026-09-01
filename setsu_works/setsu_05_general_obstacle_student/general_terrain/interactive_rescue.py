"""学生訪問状態から連続教師救援と成功分岐保存を管理する。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


STUDENT_CONTROLLER = "student"
TEACHER_CONTROLLER = "teacher"


@dataclass(frozen=True)
class RescueConfig:
    """救援開始、継続、解放に用いる保守的な閾値を保持する。"""

    entry_orientation: float = math.radians(35.0)
    warning_orientation: float = math.radians(20.0)
    exit_orientation: float = math.radians(20.0)
    entry_angular_velocity: float = 0.06
    exit_angular_velocity: float = 0.015
    entry_stall_steps: int = 60
    exit_stall_steps: int = 20
    disagreement_threshold: float = 0.35
    disagreement_minimum_stall_steps: int = 20
    disagreement_streak_steps: int = 5
    disagreement_maximum_rise_offset: int = 20
    disagreement_requires_local_terrain: bool = True
    maximum_student_prefix_steps: int | None = None
    post_recovery_stall_steps: int = 20
    require_recovery_before_release: bool = False
    pre_recovery_danger_requires_local_terrain: bool = False
    minimum_teacher_steps: int = 20
    release_safe_steps: int = 20
    release_progress: float = 0.15
    maximum_teacher_steps: int = 800

    def __post_init__(self) -> None:
        """開始条件が解放条件より緩くならないことを検査する。"""
        if self.entry_orientation <= self.exit_orientation:
            raise ValueError("救援開始角度は解放角度より大きくなければならない。")
        if self.warning_orientation > self.entry_orientation:
            raise ValueError("警戒角度は救援開始角度以下でなければならない。")
        if self.entry_angular_velocity <= self.exit_angular_velocity:
            raise ValueError("救援開始角速度は解放角速度より大きくなければならない。")
        integer_values = (
            self.entry_stall_steps,
            self.exit_stall_steps,
            self.disagreement_streak_steps,
            self.post_recovery_stall_steps,
            self.minimum_teacher_steps,
            self.release_safe_steps,
            self.maximum_teacher_steps,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("救援の歩数閾値はすべて一以上でなければならない。")
        if self.disagreement_minimum_stall_steps < 0:
            raise ValueError("動作分岐の最小停滞歩数は零以上でなければならない。")
        if self.disagreement_maximum_rise_offset < 1:
            raise ValueError("局所地形の最大前方偏移は一以上でなければならない。")
        if (
            self.maximum_student_prefix_steps is not None
            and self.maximum_student_prefix_steps < 1
        ):
            raise ValueError("学生前置歩数上限は一以上でなければならない。")


@dataclass(frozen=True)
class RescueDecision:
    """現在歩で使う制御器と救援状態遷移を公開する。"""

    controller: str
    event: str
    reason: str
    rescue_id: int
    teacher_steps: int
    timed_out: bool

    @property
    def use_teacher(self) -> bool:
        """この歩で教師動作を実行すべきかを返す。"""
        return self.controller == TEACHER_CONTROLLER


class InteractiveRescueController:
    """危険時に教師へ連続移譲し、安定回復後だけ学生へ戻す。"""

    def __init__(self, config: RescueConfig = RescueConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """一回分の介入履歴を初期化する。"""
        self.active = False
        self.rescue_id = 0
        self.teacher_steps = 0
        self.safe_streak = 0
        self.disagreement_streak = 0
        self.entry_x = 0.0
        self.entry_recovered = 0
        self.entry_obstacle_count = 0
        self.last_reason = ""
        self.student_prefix_steps = 0

    @staticmethod
    def _action_disagreement(
        student_action: np.ndarray,
        teacher_action: np.ndarray,
    ) -> float:
        """二動作の平均絶対差を正規化尺度で返す。"""
        student = np.asarray(student_action, dtype=np.float32)
        teacher = np.asarray(teacher_action, dtype=np.float32)
        if student.shape != teacher.shape:
            raise ValueError("学生動作と教師動作の形状が一致しない。")
        return float(np.mean(np.abs(student - teacher)))

    def _entry_reason(
        self,
        info: Mapping[str, object],
        student_action: np.ndarray,
        teacher_action: np.ndarray,
        *,
        local_terrain_visible: bool,
    ) -> str:
        """現在状態が救援開始条件を満たす理由を優先順で返す。"""
        orientation = float(info["orientation_error"])
        angular_velocity = abs(float(info["angular_velocity"]))
        stall_steps = int(info["stall_steps"])
        recovered = int(info["recovered_obstacles"])
        obstacle_count = int(info["obstacle_count"])
        pre_recovery_danger_allowed = bool(
            not self.config.pre_recovery_danger_requires_local_terrain
            or local_terrain_visible
            or int(info["raw_clearances"]) > 0
        )
        if bool(info["upper_body_grounded"]):
            return "upper_body_grounded"
        if (
            pre_recovery_danger_allowed
            and orientation >= self.config.entry_orientation
        ):
            return "orientation"
        if (
            pre_recovery_danger_allowed
            and orientation >= self.config.warning_orientation
            and angular_velocity >= self.config.entry_angular_velocity
        ):
            return "angular_velocity"
        if (
            recovered >= obstacle_count
            and stall_steps >= self.config.post_recovery_stall_steps
        ):
            return "post_recovery_stall"
        if stall_steps >= self.config.entry_stall_steps:
            return "stall"
        disagreement = self._action_disagreement(student_action, teacher_action)
        disagreement_allowed = bool(
            not self.config.disagreement_requires_local_terrain
            or local_terrain_visible
        )
        if (
            disagreement_allowed
            and stall_steps >= self.config.disagreement_minimum_stall_steps
            and disagreement >= self.config.disagreement_threshold
        ):
            self.disagreement_streak += 1
        else:
            self.disagreement_streak = 0
        if self.disagreement_streak >= self.config.disagreement_streak_steps:
            return "teacher_disagreement"
        return ""

    def _safe_for_release(self, info: Mapping[str, object]) -> bool:
        """教師から学生へ戻せる姿勢と進捗条件を判定する。"""
        progress = float(info["x_position"]) - self.entry_x
        recovered = int(info["recovered_obstacles"])
        progressed = progress >= self.config.release_progress
        recovered_new_obstacle = recovered > self.entry_recovered
        recovery_requirement_met = bool(
            not self.config.require_recovery_before_release
            or self.entry_recovered >= self.entry_obstacle_count
            or recovered_new_obstacle
        )
        return bool(
            float(info["orientation_error"]) <= self.config.exit_orientation
            and abs(float(info["angular_velocity"]))
            <= self.config.exit_angular_velocity
            and not bool(info["upper_body_grounded"])
            and int(info["stall_steps"]) <= self.config.exit_stall_steps
            and (progressed or recovered_new_obstacle)
            and recovery_requirement_met
        )

    def decide(
        self,
        info: Mapping[str, object],
        student_action: np.ndarray,
        teacher_action: np.ndarray,
        *,
        local_terrain_visible: bool,
    ) -> RescueDecision:
        """一歩分の制御器を選び、救援は一歩ごとに混合せず連続させる。"""
        if self.active:
            self.teacher_steps += 1
            self.safe_streak = self.safe_streak + 1 if self._safe_for_release(info) else 0
            may_release = bool(
                self.teacher_steps >= self.config.minimum_teacher_steps
                and self.safe_streak >= self.config.release_safe_steps
            )
            timed_out = self.teacher_steps >= self.config.maximum_teacher_steps
            if may_release:
                completed_steps = self.teacher_steps
                self.active = False
                self.teacher_steps = 0
                self.safe_streak = 0
                self.disagreement_streak = 0
                self.student_prefix_steps = 0
                return RescueDecision(
                    controller=STUDENT_CONTROLLER,
                    event="release",
                    reason=self.last_reason,
                    rescue_id=self.rescue_id,
                    teacher_steps=completed_steps,
                    timed_out=timed_out,
                )
            return RescueDecision(
                controller=TEACHER_CONTROLLER,
                event="continue",
                reason=self.last_reason,
                rescue_id=self.rescue_id,
                teacher_steps=self.teacher_steps,
                timed_out=timed_out,
            )

        reason = self._entry_reason(
            info,
            student_action,
            teacher_action,
            local_terrain_visible=local_terrain_visible,
        )
        if (
            not reason
            and self.config.maximum_student_prefix_steps is not None
            and self.student_prefix_steps >= self.config.maximum_student_prefix_steps
        ):
            reason = "student_prefix_budget"
        if reason:
            self.active = True
            self.rescue_id += 1
            self.teacher_steps = 1
            self.safe_streak = 0
            self.entry_x = float(info["x_position"])
            self.entry_recovered = int(info["recovered_obstacles"])
            self.entry_obstacle_count = int(info["obstacle_count"])
            self.last_reason = reason
            return RescueDecision(
                controller=TEACHER_CONTROLLER,
                event="start",
                reason=reason,
                rescue_id=self.rescue_id,
                teacher_steps=1,
                timed_out=False,
            )
        self.student_prefix_steps += 1
        return RescueDecision(
            controller=STUDENT_CONTROLLER,
            event="none",
            reason="",
            rescue_id=self.rescue_id,
            teacher_steps=0,
            timed_out=False,
        )


def local_terrain_is_visible(
    observation: np.ndarray,
    schema: tuple[str, ...],
    *,
    minimum_variation: float = 0.05,
    maximum_rise_offset: int = 20,
) -> bool:
    """学生観測だけから局所地形の高低差が見えているかを返す。"""
    if maximum_rise_offset < 1:
        raise ValueError("局所地形の最大前方偏移は一以上でなければならない。")
    indices = []
    for index, name in enumerate(schema):
        if not name.startswith("relative_terrain_height_"):
            continue
        offset = int(name.rsplit("_", maxsplit=1)[-1])
        if 0 <= offset <= maximum_rise_offset:
            indices.append(index)
    if not indices:
        raise ValueError("観測スキーマに相対地形走査が存在しない。")
    values = np.asarray(observation, dtype=np.float32)[indices]
    return float(np.ptp(values)) >= minimum_variation


class SuccessfulRescueBuffer:
    """一回分を一時保持し、成功救援だけを保存する。"""

    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.student_actions: list[np.ndarray] = []
        self.teacher_actions: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.teacher_mask: list[bool] = []
        self.rescue_ids: list[int] = []
        self.teacher_stages: list[str] = []

    def append(
        self,
        observation: np.ndarray,
        student_action: np.ndarray,
        teacher_action: np.ndarray,
        executed_action: np.ndarray,
        decision: RescueDecision,
        teacher_stage: str,
    ) -> None:
        """保存判定前の一歩分をメモリへ追加する。"""
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.student_actions.append(np.asarray(student_action, dtype=np.float32).copy())
        self.teacher_actions.append(np.asarray(teacher_action, dtype=np.float32).copy())
        self.executed_actions.append(np.asarray(executed_action, dtype=np.float32).copy())
        self.teacher_mask.append(decision.use_teacher)
        self.rescue_ids.append(decision.rescue_id if decision.use_teacher else 0)
        self.teacher_stages.append(str(teacher_stage))

    @property
    def teacher_steps(self) -> int:
        """教師が実際に制御した歩数を返す。"""
        return int(sum(self.teacher_mask))

    def commit(
        self,
        output_path: Path,
        final_info: Mapping[str, object],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> bool:
        """完走かつ無硬転倒の救援回だけをNPZとJSONへ保存する。"""
        accepted = bool(
            self.observations
            and self.teacher_steps > 0
            and bool(final_info["course_complete"])
            and not bool(final_info["hard_fall"])
            and int(final_info["recovered_obstacles"])
            >= int(final_info["obstacle_count"])
        )
        if not accepted:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            observations=np.asarray(self.observations, dtype=np.float32),
            student_actions=np.asarray(self.student_actions, dtype=np.float32),
            teacher_actions=np.asarray(self.teacher_actions, dtype=np.float32),
            executed_actions=np.asarray(self.executed_actions, dtype=np.float32),
            teacher_mask=np.asarray(self.teacher_mask, dtype=bool),
            rescue_ids=np.asarray(self.rescue_ids, dtype=np.int32),
            teacher_stages=np.asarray(self.teacher_stages, dtype=str),
        )
        report = {
            "accepted": True,
            "course_id": str(final_info["course_id"]),
            "steps": len(self.observations),
            "teacher_steps": self.teacher_steps,
            "teacher_fraction": self.teacher_steps / len(self.observations),
            "rescue_count": max(self.rescue_ids, default=0),
            "course_complete": True,
            "hard_fall": False,
            "recovered_obstacles": int(final_info["recovered_obstacles"]),
            "metadata": dict(metadata or {}),
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
