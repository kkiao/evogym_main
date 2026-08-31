"""学習器誘導状態専用の訓練救援教師目録を読み込み実行する。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from general_terrain.interactive_rescue import RescueConfig
from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher
from general_terrain.student_prefix_rescue_env import classify_rescue_phase


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class M4RRoute:
    """一つの学習器誘導位置に選択された救援制御器設定を保持する。"""

    first_switch_fraction: float
    post_clear_mode: str
    handoff_distance: float
    clearance_blend: float
    clearance_family: str
    adaptive_handoff: bool


@dataclass(frozen=True)
class M4RTeacherManifest:
    """M4R訓練専用教師の開始条件、経路、出典ハッシュを保持する。"""

    version: str
    stage: str
    split: str
    trigger_profile_name: str
    trigger_config: RescueConfig
    controller_mode: str
    robust_flat_teacher_model_path: Path | None
    robust_flat_prefix_steps: int
    routes: dict[int, M4RRoute]
    route_status: dict[int, str]
    source_path: Path
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """監査要約へ保存可能な辞書を返す。"""
        return {
            "version": self.version,
            "stage": self.stage,
            "split": self.split,
            "trigger_profile_name": self.trigger_profile_name,
            "trigger_config": asdict(self.trigger_config),
            "controller_mode": self.controller_mode,
            "robust_flat_teacher_model_path": (
                str(self.robust_flat_teacher_model_path)
                if self.robust_flat_teacher_model_path is not None
                else None
            ),
            "robust_flat_prefix_steps": self.robust_flat_prefix_steps,
            "routes": {
                str(position): asdict(route)
                for position, route in sorted(self.routes.items())
            },
            "route_status": {
                str(position): status
                for position, status in sorted(self.route_status.items())
            },
            "source_path": str(self.source_path),
            "sha256": self.sha256,
            "teacher_training_only": True,
            "validation_teacher_enabled": False,
            "holdout_teacher_enabled": False,
            "final_student_test_teacher_enabled": False,
            "teacher_interventions_in_final_student_test": 0,
        }


def _sha256(path: Path) -> str:
    """指定ファイルのSHA-256を小文字で返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trigger_config_from_payload(payload: Mapping[str, object]) -> RescueConfig:
    """凍結開始条件を連続終端保持用の救援設定へ変換する。"""
    return RescueConfig(
        entry_orientation=math.radians(float(payload["entry_orientation_degrees"])),
        warning_orientation=math.radians(
            float(payload["warning_orientation_degrees"])
        ),
        exit_orientation=math.radians(float(payload["exit_orientation_degrees"])),
        entry_angular_velocity=float(payload["entry_angular_velocity"]),
        exit_angular_velocity=float(payload["exit_angular_velocity"]),
        entry_stall_steps=60,
        exit_stall_steps=20,
        disagreement_threshold=float(payload["disagreement_threshold"]),
        disagreement_minimum_stall_steps=0,
        disagreement_streak_steps=int(payload["disagreement_streak_steps"]),
        disagreement_maximum_rise_offset=int(
            payload["disagreement_maximum_rise_offset"]
        ),
        disagreement_requires_local_terrain=bool(
            payload.get("disagreement_requires_local_terrain", True)
        ),
        maximum_student_prefix_steps=(
            int(payload["maximum_student_prefix_steps"])
            if payload.get("maximum_student_prefix_steps") is not None
            else None
        ),
        post_recovery_stall_steps=20,
        require_recovery_before_release=True,
        pre_recovery_danger_requires_local_terrain=bool(
            payload["pre_recovery_danger_requires_local_terrain"]
        ),
        minimum_teacher_steps=int(payload.get("minimum_teacher_steps", 20)),
        release_safe_steps=int(payload.get("release_safe_steps", 1200)),
        release_progress=0.15,
        maximum_teacher_steps=int(payload.get("maximum_teacher_steps", 1200)),
    )


def _validate_route(position: int, payload: Mapping[str, object]) -> M4RRoute:
    """一経路の型、位置、範囲を検査する。"""
    route = M4RRoute(
        first_switch_fraction=float(payload["first_switch_fraction"]),
        post_clear_mode=str(payload["post_clear_mode"]),
        handoff_distance=float(payload["handoff_distance"]),
        clearance_blend=float(payload["clearance_blend"]),
        clearance_family=str(payload["clearance_family"]),
        adaptive_handoff=bool(payload["adaptive_handoff"]),
    )
    if position not in range(20, 31):
        raise ValueError("M4R経路位置が許可範囲外である。")
    if not 0.1 <= route.first_switch_fraction <= 0.95:
        raise ValueError("M4R切替率が許可範囲外である。")
    if route.post_clear_mode not in {
        "flat",
        "restart",
        "restart_then_flat",
        "restart_brake_flat",
        "landing_then_restart",
    }:
        raise ValueError("M4R越壁後制御モードが未知である。")
    if not 0.0 < route.handoff_distance <= 1.0:
        raise ValueError("M4R引継ぎ距離が許可範囲外である。")
    if not 0.0 < route.clearance_blend <= 1.0:
        raise ValueError("M4R越壁動作混合率が許可範囲外である。")
    if route.clearance_family not in {"first", "second"}:
        raise ValueError("M4R越壁動作系列が未知である。")
    return route


def load_m4r_teacher_manifest(path: Path) -> M4RTeacherManifest:
    """凍結目録、全位置経路、出典不変、試験教師隔離を検査する。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("M4R教師目録はプロジェクト配下でなければならない。")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("M4R教師目録は凍結済みでなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("M4R教師は単一低壁の訓練区分だけを対象とする。")
    if not bool(payload.get("teacher_training_only", False)):
        raise ValueError("M4R教師は訓練専用でなければならない。")
    for field in (
        "validation_teacher_enabled",
        "holdout_teacher_enabled",
        "final_student_test_teacher_enabled",
    ):
        if bool(payload.get(field, True)):
            raise ValueError("検証、留保、最終学生試験では教師を停止しなければならない。")
    if int(payload.get("teacher_interventions_in_final_student_test", -1)) != 0:
        raise ValueError("最終学生試験の教師介入数は零でなければならない。")
    if not bool(payload.get("eligible_for_continuous_validation", False)):
        raise ValueError("修復門を通過していないM4R教師は連続復核へ使用できない。")
    for source in payload.get("sources", []):
        source_path = (PROJECT_ROOT / str(source["path"])).resolve()
        if not source_path.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError("M4R教師出典はプロジェクト配下でなければならない。")
        if _sha256(source_path) != str(source["sha256"]).lower():
            raise ValueError(f"M4R教師出典ハッシュが一致しない: {source_path}")
    routes = {
        int(position): _validate_route(int(position), route)
        for position, route in payload["routes"].items()
    }
    if set(routes) != set(range(20, 31)):
        raise ValueError("M4R教師目録は十一位置すべての安全経路を保持しなければならない。")
    route_status = {
        int(position): str(status)
        for position, status in payload["route_status"].items()
    }
    if set(route_status) != set(routes):
        raise ValueError("M4R経路状態と経路位置が一致しない。")
    trigger_payload = payload["trigger_profile"]
    robust_flat_value = payload.get("robust_flat_teacher_model_path")
    robust_flat_path = (
        (PROJECT_ROOT / str(robust_flat_value)).resolve()
        if robust_flat_value is not None
        else None
    )
    robust_flat_prefix_steps = int(payload.get("robust_flat_prefix_steps", 0))
    controller_mode = str(payload.get("controller_mode", "routed_closed_loop"))
    if controller_mode not in {"routed_closed_loop", "verified_portfolio"}:
        raise ValueError("M4R教師の実行制御方式が未知である。")
    if robust_flat_path is not None and robust_flat_prefix_steps < 1:
        raise ValueError("M4R頑健平地橋渡し歩数は一以上でなければならない。")
    return M4RTeacherManifest(
        version=str(payload["version"]),
        stage=str(payload["stage"]),
        split=str(payload["split"]),
        trigger_profile_name=str(trigger_payload["name"]),
        trigger_config=trigger_config_from_payload(trigger_payload),
        controller_mode=controller_mode,
        robust_flat_teacher_model_path=robust_flat_path,
        robust_flat_prefix_steps=robust_flat_prefix_steps,
        routes=routes,
        route_status=route_status,
        source_path=resolved,
        sha256=_sha256(resolved),
    )


class M4RLearnerDistributionTeacher:
    """監視中は旧教師、救援中は学習器分布専用経路を連続使用する。"""

    def __init__(self, manifest: M4RTeacherManifest) -> None:
        self.manifest = manifest
        self.portfolio = PortfolioHeight1Teacher(
            flat_model_path=manifest.robust_flat_teacher_model_path
        )
        self.position = -1
        self.route: M4RRoute | None = None
        self.active = False

    def reset(self, environment: object) -> None:
        """訓練コースだけを受け入れ一回分の監視状態を初期化する。"""
        course = environment.unwrapped.course
        if str(course.split) != "train_hurdle_single":
            raise ValueError("M4R教師は訓練区分以外で使用できない。")
        self.position = int(course.obstacles[0].start_x)
        self.route = self.manifest.routes[self.position]
        self.portfolio.reset(environment)
        self.active = False

    def _course_name(self, environment: object, info: Mapping[str, object]) -> str:
        """現在物理状態を三つの分離救援課程名へ写像する。"""
        phase = classify_rescue_phase(environment, info)
        if phase == "pre_hurdle":
            return "pre_hurdle_safety_intercept"
        if phase == "hurdle_deformation":
            return "hurdle_contact_deformation"
        return "landing_recovery_stall"

    def _routed_action(
        self,
        environment: object,
        info: Mapping[str, object],
    ) -> tuple[np.ndarray, str]:
        """選択済み経路から一歩分の動作と分離課程名を返す。"""
        controller = self.portfolio.controller
        if controller is None or self.route is None:
            raise RuntimeError("M4R位置別制御器が初期化されていない。")
        action, stage = controller.predict(environment, dict(info))
        course_name = self._course_name(environment, info)
        return (
            np.asarray(action, dtype=np.float32),
            f"m4r_{course_name}_x{self.position}:{stage}",
        )

    def _portfolio_action(
        self,
        environment: object,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> tuple[np.ndarray, str]:
        """凍結検証済み組合せ教師から一歩分の救援動作を返す。"""
        action, stage = self.portfolio.predict(
            environment,
            observation,
            dict(info),
        )
        course_name = self._course_name(environment, info)
        return (
            np.asarray(action, dtype=np.float32),
            f"m4r_{course_name}_x{self.position}:{stage}",
        )

    def predict(
        self,
        environment: object,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> tuple[np.ndarray, str]:
        """開始監視用ラベルまたは連続救援動作を返す。"""
        if self.active:
            if self.manifest.controller_mode == "verified_portfolio":
                return self._portfolio_action(environment, observation, info)
            return self._routed_action(environment, info)
        action, stage = self.portfolio.predict(
            environment,
            observation,
            dict(info),
        )
        return (
            np.asarray(action, dtype=np.float32),
            f"m4r_trigger_x{self.position}:{stage}",
        )

    def on_rescue_start(
        self,
        environment: object,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> tuple[np.ndarray, str]:
        """接管した同じ歩で選択経路を現在物理状態へ再初期化する。"""
        self.active = True
        if self.route is None or self.portfolio.controller is None:
            raise RuntimeError("M4R経路が救援開始前に設定されていない。")
        if self.manifest.controller_mode == "verified_portfolio":
            self.portfolio.reset(environment)
            return self._portfolio_action(environment, observation, info)
        controller = self.portfolio.controller
        controller.post_clear_mode = self.route.post_clear_mode
        controller.clearance_blend = self.route.clearance_blend
        controller.handoff_distance = self.route.handoff_distance
        controller.adaptive_handoff = self.route.adaptive_handoff
        controller.clearance_family = self.route.clearance_family
        controller.first_switch_fraction = self.route.first_switch_fraction
        controller.robust_flat_max_steps = self.manifest.robust_flat_prefix_steps
        controller.reset(environment)
        return self._routed_action(environment, info)

    def on_rescue_release(self, environment: object) -> None:
        """救援解放後は監視用教師を現在状態から再初期化する。"""
        self.portfolio.reset(environment)
        self.active = False
