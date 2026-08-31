"""M5逆向きカリキュラムの隔離、再生、判停規則を検査する。"""

from __future__ import annotations

from general_terrain.train_m5_reverse_curriculum import (
    DEFAULT_PROTOCOL,
    M5ReverseCurriculumEnv,
    load_protocol,
    load_rollin_specs,
    plateau_detected,
)
from general_terrain.seed_manifest import load_seed_manifest


def test_protocol_has_bounded_checkpoint_ladder() -> None:
    """正式規約がCPU八並列と複数検査点を固定することを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol.device == "cpu"
    assert protocol.parallel_environments == 8
    assert protocol.planned_steps == 196_608
    assert protocol.planned_steps // protocol.checkpoint_interval_steps == 16
    assert protocol.extension_steps == 49_152


def test_phase_rollin_starts_student_without_rescue() -> None:
    """精密再生後に教師介入ゼロで学生制御が始まることを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    specs = load_rollin_specs(protocol.phase_reset_manifest_path)
    seeds = load_seed_manifest(protocol.seed_manifest_path).for_split("train")
    spec = next(item for item in specs if item.phase == "pre_hurdle")
    environment = M5ReverseCurriculumEnv(
        specs,
        seeds,
        {"pre_hurdle": 1.0},
        seed=7,
        maximum_student_steps=32,
    )
    try:
        observation, info = environment.reset(
            options={"mode": "pre_hurdle", "reset_id": spec.reset_id}
        )
        assert observation.shape == (95,)
        assert info["training_rollin_steps"] == spec.source_step
        assert info["training_rollin_action_source"] == "frozen_success_npz"
        assert info["student_control_started"] is True
        assert info["teacher_module_loaded"] is False
        assert info["teacher_interventions"] == 0
        assert info["teacher_actions_after_student_takeover"] == 0
    finally:
        environment.close()


def test_plateau_requires_three_consecutive_non_improvements() -> None:
    """一回の失敗では捨てず三検査点の無改善だけを停滞とする。"""
    base = {
        "success_count": 0,
        "mean_recovered_obstacles": 0.0,
        "mean_raw_clearances": 0.0,
        "hard_fall_count": 7,
        "mean_max_x": 2.0,
    }
    assert not plateau_detected([base, dict(base)], window=3)
    assert plateau_detected([base, dict(base), dict(base), dict(base)], window=3)
    improved = dict(base)
    improved["mean_max_x"] = 2.03
    assert not plateau_detected(
        [base, dict(base), improved, dict(improved)],
        window=3,
    )
