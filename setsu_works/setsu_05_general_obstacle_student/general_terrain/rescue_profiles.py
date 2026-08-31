"""M2とM2.1の境界付き救援設定を名前付きで管理する。"""

from __future__ import annotations

import math

from general_terrain.interactive_rescue import RescueConfig


M2_DEFAULT_PROFILE = "m2_default"
M2_1_PROFILE_NAMES = (
    "m2_1_near6",
    "m2_1_near8",
    "m2_1_near10",
)
M2_4_MANIFEST_PROFILE = "m2_4_manifest_hold"
RESCUE_PROFILE_NAMES = (
    M2_DEFAULT_PROFILE,
    *M2_1_PROFILE_NAMES,
    M2_4_MANIFEST_PROFILE,
)


def _preventive_profile(maximum_rise_offset: int) -> RescueConfig:
    """障害物直前の動作分岐で早期接管する共通設定を返す。"""
    return RescueConfig(
        entry_orientation=math.radians(25.0),
        warning_orientation=math.radians(15.0),
        exit_orientation=math.radians(15.0),
        entry_angular_velocity=0.04,
        exit_angular_velocity=0.015,
        entry_stall_steps=60,
        exit_stall_steps=20,
        disagreement_threshold=0.25,
        disagreement_minimum_stall_steps=0,
        disagreement_streak_steps=3,
        disagreement_maximum_rise_offset=maximum_rise_offset,
        post_recovery_stall_steps=20,
        require_recovery_before_release=True,
        pre_recovery_danger_requires_local_terrain=True,
        minimum_teacher_steps=20,
        release_safe_steps=20,
        release_progress=0.15,
        maximum_teacher_steps=800,
    )


def _manifest_hold_profile() -> RescueConfig:
    """凍結開始境界を保ち、成功または安全終端まで同一教師を連続使用する。"""
    profile = _preventive_profile(6)
    return RescueConfig(
        **{
            **profile.__dict__,
            "release_safe_steps": 1200,
            "maximum_teacher_steps": 1200,
        }
    )


def get_rescue_profile(name: str) -> RescueConfig:
    """名前付き設定を新規不変オブジェクトとして返す。"""
    if name == M2_DEFAULT_PROFILE:
        return RescueConfig()
    if name == M2_4_MANIFEST_PROFILE:
        return _manifest_hold_profile()
    offsets = {
        "m2_1_near6": 6,
        "m2_1_near8": 8,
        "m2_1_near10": 10,
    }
    try:
        return _preventive_profile(offsets[name])
    except KeyError as error:
        raise ValueError(f"未知の救援設定: {name}") from error
