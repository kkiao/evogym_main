"""M6Sの自己成功錨、越え率保持門、安全順位を検査する。"""

from __future__ import annotations

from general_terrain.train_m6s_student_success_safety import (
    DEFAULT_PROTOCOL,
    eligible_checkpoint,
    load_protocol,
    raw_clearance_count,
    safety_key,
)


def evaluation(successes: int, clearances: int, recoveries: int, falls: int):
    """小さな学生評価辞書を試験用に構築する。"""
    episodes = [
        {"raw_clearances": int(index < clearances)} for index in range(11)
    ]
    return {
        "success_count": successes,
        "mean_recovered_obstacles": recoveries / 11,
        "hard_fall_count": falls,
        "mean_max_x": 3.0,
        "episodes": episodes,
    }


def test_protocol_has_six_cpu_checkpoints_without_teacher() -> None:
    """M6SがCPU八並列と六検査点を固定することを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    assert protocol.device == "cpu"
    assert protocol.parallel_environments == 8
    assert protocol.total_steps // protocol.checkpoint_interval_steps == 6


def test_checkpoint_must_preserve_clearance_and_flat_gait() -> None:
    """硬転倒だけを減らす壁前停滞を合格させないことを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    flat = {"success_count": 3, "hard_fall_count": 0}
    retained = evaluation(1, 7, 1, 5)
    stalled = evaluation(0, 6, 0, 0)
    assert raw_clearance_count(retained) == 7
    assert eligible_checkpoint(retained, flat, protocol)
    assert not eligible_checkpoint(stalled, flat, protocol)


def test_safety_key_prefers_recovery_then_fewer_falls() -> None:
    """完走数が同じなら回復と安全を優先することを確認する。"""
    prelanding = {"recovery_count": 0}
    base = evaluation(1, 7, 1, 6)
    safer = evaluation(1, 7, 1, 5)
    recovered = evaluation(1, 7, 2, 6)
    assert safety_key(safer, prelanding) > safety_key(base, prelanding)
    assert safety_key(recovered, prelanding) > safety_key(safer, prelanding)
