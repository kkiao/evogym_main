"""M6L学生誘導落地の規約、順位、報酬境界を検査する。"""

from __future__ import annotations

from general_terrain.train_m6l_learner_landing import (
    DEFAULT_PROTOCOL,
    checkpoint_key,
    load_protocol,
)


def test_protocol_has_five_checkpoints_without_teacher() -> None:
    """M6Lが五検査点と教師動作ゼロを固定することを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol.device == "cpu"
    assert protocol.parallel_environments == 8
    assert protocol.total_steps // protocol.checkpoint_interval_steps == 5


def test_checkpoint_key_prefers_recovery_without_losing_clearance() -> None:
    """同じ越え数なら回復を増やす検査点が先になることを確認する。"""
    base = {
        "success_count": 0,
        "mean_recovered_obstacles": 0.0,
        "mean_raw_clearances": 0.5,
        "hard_fall_count": 5,
        "mean_max_x": 2.8,
    }
    recovered = dict(base)
    recovered["mean_recovered_obstacles"] = 0.1
    landing = {"recovery_count": 0}
    assert checkpoint_key(recovered, landing) > checkpoint_key(base, landing)
