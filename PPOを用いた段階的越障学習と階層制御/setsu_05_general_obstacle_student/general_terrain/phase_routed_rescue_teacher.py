"""合格した位置別制御器を越壁前救援だけに限定して経路選択する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from general_terrain.closed_loop_height1_teacher import ClosedLoopHeight1Teacher
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher
from general_terrain.student_prefix_rescue_env import classify_rescue_phase


@dataclass(frozen=True)
class RoutedControllerConfig:
    """一つの開始位置に対応する合格済み制御器設定を保持する。"""

    first_switch_fraction: float
    post_clear_mode: str
    handoff_distance: float
    clearance_blend: float


def route_configs_from_search(
    summary: Mapping[str, object],
) -> dict[int, RoutedControllerConfig]:
    """合格済み探索結果から開始位置別設定を厳格に読み取る。"""
    if not bool(summary["gate"]["gate_passed"]):
        raise ValueError("不合格の制御器探索結果は経路教師に使用できない。")
    routes: dict[int, RoutedControllerConfig] = {}
    for row in summary["position_results"]:
        selected = row["selected"]
        if selected is None:
            raise ValueError("合格探索結果に未選択位置が含まれている。")
        position = int(row["start_runway_voxels"])
        routes[position] = RoutedControllerConfig(
            first_switch_fraction=float(selected["first_switch_fraction"]),
            post_clear_mode=str(selected["post_clear_mode"]),
            handoff_distance=float(selected["handoff_distance"]),
            clearance_blend=float(selected["clearance_blend"]),
        )
    if len(routes) != 4:
        raise ValueError("経路教師には四位置の合格設定が必要である。")
    return routes


class PhaseRoutedRescueTeacher:
    """越壁前だけ位置別救援を使い他位相では旧教師を維持する。"""

    def __init__(
        self,
        routes: Mapping[int, RoutedControllerConfig],
        *,
        flat_model_path: Path | None = None,
    ) -> None:
        self.routes = dict(routes)
        self.old_teacher = PortfolioHeight1Teacher(flat_model_path=flat_model_path)
        self.routed_controller = ClosedLoopHeight1Teacher(
            post_clear_mode="restart_then_flat",
            clearance_blend=1.0,
            handoff_distance=0.45,
            adaptive_handoff=True,
            first_switch_fraction=0.25,
        )
        self.position = -1
        self.route_available = False
        self.active = False
        self.use_routed_controller = False
        self.activation_phase = ""

    def reset(self, environment: object) -> None:
        """旧教師と位置別制御器を同じ初期物理状態で初期化する。"""
        self.position = int(environment.unwrapped.course.obstacles[0].start_x)
        self.old_teacher.reset(environment)
        self.route_available = self.position in self.routes
        if self.route_available:
            route = self.routes[self.position]
            self.routed_controller.post_clear_mode = route.post_clear_mode
            self.routed_controller.clearance_blend = route.clearance_blend
            self.routed_controller.handoff_distance = route.handoff_distance
            self.routed_controller.adaptive_handoff = True
            self.routed_controller.first_switch_fraction = route.first_switch_fraction
        self.routed_controller.reset(environment)
        self.active = False
        self.use_routed_controller = False
        self.activation_phase = ""

    def observe(
        self,
        environment: object,
        observation: np.ndarray,
        info: dict[str, object],
    ) -> None:
        """学生制御中も二つの教師状態を実物理状態へ追従させる。"""
        self.old_teacher.predict(environment, observation, info)
        self.routed_controller.predict(environment, info)

    def activate(self, environment: object, info: dict[str, object]) -> str:
        """介入開始位相を一度だけ判定し使用経路を固定する。"""
        if self.active:
            return self.activation_phase
        self.activation_phase = classify_rescue_phase(environment, info)
        self.use_routed_controller = bool(
            self.route_available and self.activation_phase == "pre_hurdle"
        )
        self.active = True
        return self.activation_phase

    def predict(
        self,
        environment: object,
        observation: np.ndarray,
        info: dict[str, object],
    ) -> tuple[np.ndarray, str]:
        """固定済み経路から訓練専用救援動作と監査段階名を返す。"""
        self.activate(environment, info)
        old_action, old_stage = self.old_teacher.predict(
            environment,
            observation,
            info,
        )
        routed_action, routed_stage = self.routed_controller.predict(environment, info)
        if self.use_routed_controller:
            return (
                np.asarray(routed_action, dtype=np.float32),
                f"routed_x{self.position}:{routed_stage}",
            )
        return (
            np.asarray(old_action, dtype=np.float32),
            f"portfolio:{old_stage}",
        )
