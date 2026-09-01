"""凍結目録の位置別経路を訓練中の連続救援だけへ接続する。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from general_terrain.portfolio_height1_teacher import PortfolioHeight1Teacher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_MANIFEST = (
    PROJECT_ROOT
    / "training_only_teacher"
    / "teacher_manifest.json"
)


@dataclass(frozen=True)
class TakeoverRoute:
    """一つの開始位置で合格済みの引継ぎ制御器設定を保持する。"""

    first_switch_fraction: float
    post_clear_mode: str
    handoff_distance: float
    clearance_blend: float
    clearance_family: str
    adaptive_handoff: bool


@dataclass(frozen=True)
class TrainingTeacherManifest:
    """訓練専用教師の経路、隔離条件、出典ハッシュを保持する。"""

    version: str
    stage: str
    split: str
    routes: dict[int, TakeoverRoute]
    uncovered_positions: tuple[int, ...]
    source_path: Path
    sha256: str

    def as_dict(self) -> dict[str, object]:
        """監査要約へ保存できる辞書を返す。"""
        return {
            "version": self.version,
            "stage": self.stage,
            "split": self.split,
            "routes": {
                str(position): asdict(route)
                for position, route in sorted(self.routes.items())
            },
            "uncovered_positions": list(self.uncovered_positions),
            "source_path": str(self.source_path),
            "sha256": self.sha256,
            "teacher_training_only": True,
            "validation_teacher_enabled": False,
            "holdout_teacher_enabled": False,
            "final_student_test_teacher_enabled": False,
            "teacher_interventions_in_final_student_test": 0,
        }


def _sha256(path: Path) -> str:
    """指定ファイルのSHA-256を返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_route(position: int, payload: Mapping[str, object]) -> TakeoverRoute:
    """目録の一経路を型と有界範囲まで検査する。"""
    route = TakeoverRoute(
        first_switch_fraction=float(payload["first_switch_fraction"]),
        post_clear_mode=str(payload["post_clear_mode"]),
        handoff_distance=float(payload["handoff_distance"]),
        clearance_blend=float(payload["clearance_blend"]),
        clearance_family=str(payload["clearance_family"]),
        adaptive_handoff=bool(payload["adaptive_handoff"]),
    )
    if position not in range(20, 31):
        raise ValueError("引継ぎ経路の開始位置が許可範囲外である。")
    if not 0.1 <= route.first_switch_fraction <= 0.95:
        raise ValueError("引継ぎ経路の切替率が許可範囲外である。")
    if route.post_clear_mode not in {
        "flat",
        "restart",
        "restart_then_flat",
        "restart_brake_flat",
        "landing_then_restart",
    }:
        raise ValueError("引継ぎ経路の越壁後モードが未知である。")
    if not 0.0 < route.handoff_distance <= 1.0:
        raise ValueError("引継ぎ距離が許可範囲外である。")
    if not 0.0 < route.clearance_blend <= 1.0:
        raise ValueError("越壁動作混合率が許可範囲外である。")
    if route.clearance_family not in {"first", "second"}:
        raise ValueError("引継ぎ経路の越壁系列が未知である。")
    return route


def load_training_teacher_manifest(
    path: Path = DEFAULT_TEACHER_MANIFEST,
) -> TrainingTeacherManifest:
    """凍結目録を読み、訓練区分限定と最終試験隔離を厳格に検査する。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("訓練教師目録はプロジェクト配下になければならない。")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not bool(payload.get("frozen", False)):
        raise ValueError("訓練教師目録は凍結済みでなければならない。")
    if not bool(payload.get("teacher_training_only", False)):
        raise ValueError("教師目録は訓練専用でなければならない。")
    if payload.get("stage") != "hurdle_single" or payload.get("split") != "train":
        raise ValueError("教師目録は単一低壁の訓練区分だけを対象とする。")
    disabled_fields = (
        "validation_teacher_enabled",
        "holdout_teacher_enabled",
        "final_student_test_teacher_enabled",
    )
    if any(bool(payload.get(field, True)) for field in disabled_fields):
        raise ValueError("検証、留保、最終試験では教師を完全停止しなければならない。")
    if int(payload.get("teacher_interventions_in_final_student_test", -1)) != 0:
        raise ValueError("最終学生試験の教師介入数は零でなければならない。")
    takeover = payload["student_takeover_curriculum"]
    if not bool(takeover.get("reinitialize_at_takeover", False)):
        raise ValueError("引継ぎ時の教師再初期化が必須である。")
    routes = {
        int(position): _validate_route(int(position), route)
        for position, route in takeover["routes"].items()
    }
    uncovered = tuple(sorted(int(value) for value in takeover["uncovered_positions"]))
    if set(routes) & set(uncovered):
        raise ValueError("経路位置と未対応位置が重複している。")
    if set(routes) | set(uncovered) != set(range(20, 31)):
        raise ValueError("教師目録が全十一開始位置を網羅していない。")
    if len(routes) != 9 or uncovered != (21, 22):
        raise ValueError("凍結引継ぎ経路集合が九成功二未対応の基線と一致しない。")
    if takeover.get("uncovered_fallback") != "portfolio_height1_teacher_v1":
        raise ValueError("未対応位置の安全代替教師が凍結値と一致しない。")
    if not bool(payload.get("eligible_for_m2_4", False)):
        raise ValueError("M2.4未解禁の教師目録は収集へ使用できない。")
    return TrainingTeacherManifest(
        version=str(payload["version"]),
        stage=str(payload["stage"]),
        split=str(payload["split"]),
        routes=routes,
        uncovered_positions=uncovered,
        source_path=resolved,
        sha256=_sha256(resolved),
    )


class ManifestTrainingRescueTeacher:
    """学生制御中は旧教師を監視し、救援開始時だけ合格経路へ再初期化する。"""

    def __init__(
        self,
        manifest: TrainingTeacherManifest,
        *,
        flat_model_path: Path | None = None,
    ) -> None:
        """凍結目録と任意の訓練専用頑健平地教師を読み込む。"""
        self.manifest = manifest
        self.portfolio = PortfolioHeight1Teacher(flat_model_path=flat_model_path)
        self.position = -1
        self.route: TakeoverRoute | None = None
        self.active = False

    def reset(self, environment: object) -> None:
        """訓練コースだけを受け入れ、一回分の監視教師状態を初期化する。"""
        course = environment.unwrapped.course
        expected_split = f"{self.manifest.split}_{self.manifest.stage}"
        if str(course.split) != expected_split:
            raise ValueError("目録教師は訓練区分以外で使用できない。")
        self.position = int(course.obstacles[0].start_x)
        if self.position not in range(20, 31):
            raise ValueError("目録外の開始位置では訓練教師を使用できない。")
        self.route = self.manifest.routes.get(self.position)
        self.portfolio.reset(environment)
        self.active = False

    def _routed_action(
        self,
        environment: object,
        info: Mapping[str, object],
    ) -> tuple[np.ndarray, str]:
        """再初期化済み位置別制御器から一歩分の動作を返す。"""
        controller = self.portfolio.controller
        if controller is None or self.route is None:
            raise RuntimeError("位置別制御器が救援開始時に初期化されていない。")
        action, stage = controller.predict(environment, dict(info))
        return (
            np.asarray(action, dtype=np.float32),
            f"manifest_takeover_x{self.position}:{stage}",
        )

    def predict(
        self,
        environment: object,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> tuple[np.ndarray, str]:
        """開始判定用ラベルまたは連続救援中の位置別動作を返す。"""
        if self.active and self.route is not None:
            return self._routed_action(environment, info)
        action, stage = self.portfolio.predict(
            environment,
            observation,
            dict(info),
        )
        prefix = (
            f"manifest_fallback_x{self.position}"
            if self.active
            else f"manifest_trigger_x{self.position}"
        )
        return np.asarray(action, dtype=np.float32), f"{prefix}:{stage}"

    def on_rescue_start(
        self,
        environment: object,
        observation: np.ndarray,
        info: Mapping[str, object],
    ) -> tuple[np.ndarray, str]:
        """接管した同じ歩で教師を現在物理状態へ再初期化して動作を再計算する。"""
        self.active = True
        if self.route is None:
            self.portfolio.reset(environment)
            return self.predict(environment, observation, info)
        controller = self.portfolio.controller
        if controller is None:
            raise RuntimeError("位置別制御器の共有元が初期化されていない。")
        controller.post_clear_mode = self.route.post_clear_mode
        controller.clearance_blend = self.route.clearance_blend
        controller.handoff_distance = self.route.handoff_distance
        controller.adaptive_handoff = self.route.adaptive_handoff
        controller.clearance_family = self.route.clearance_family
        controller.first_switch_fraction = self.route.first_switch_fraction
        controller.reset(environment)
        return self._routed_action(environment, info)

    def on_rescue_release(self, environment: object) -> None:
        """救援解放後は現在状態から監視用旧教師を再初期化する。"""
        self.portfolio.reset(environment)
        self.active = False
