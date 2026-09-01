"""M6密集交接帯の補間、門限、教師禁止境界を検査する。"""

from __future__ import annotations

from general_terrain.train_m5_reverse_curriculum import load_rollin_specs
from general_terrain.train_m6_dense_handoff import (
    DEFAULT_PROTOCOL,
    band_passed,
    build_dense_specs,
    load_protocol,
)


def test_dense_specs_interpolate_between_verified_endpoints() -> None:
    """密集開始歩が変形点と回復点の間へ厳密に入ることを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    original = load_rollin_specs(protocol.phase_reset_manifest_path)
    dense = build_dense_specs(original, 0.5)
    assert len(dense) == 4
    for spec in dense:
        deformation = next(
            item
            for item in original
            if item.seed == spec.seed and item.phase == "hurdle_deformation"
        )
        recovery = next(
            item
            for item in original
            if item.seed == spec.seed and item.phase == "post_clearance_recovery"
        )
        assert deformation.source_step < spec.source_step < recovery.source_step
        assert spec.phase == "dense_handoff"


def test_band_gate_requires_repeated_success_without_hard_fall() -> None:
    """一回だけの成功または硬転倒を帯通過として扱わないことを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert band_passed({"success_count": 3, "hard_fall_count": 0}, protocol)
    assert not band_passed({"success_count": 2, "hard_fall_count": 0}, protocol)
    assert not band_passed({"success_count": 4, "hard_fall_count": 1}, protocol)


def test_m6_keeps_teacher_disabled_after_takeover() -> None:
    """M6規約が学生接管後と検証時の教師を完全に禁止することを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol.device == "cpu"
    assert protocol.parallel_environments == 8
    assert protocol.maximum_checkpoints_per_band == 3
