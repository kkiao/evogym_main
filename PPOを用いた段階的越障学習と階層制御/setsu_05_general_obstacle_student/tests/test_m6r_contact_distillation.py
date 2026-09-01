"""M6R接触窓データ、候補門限、critic隔離を検査する。"""

from __future__ import annotations

from stable_baselines3 import PPO

from general_terrain.train_m5_reverse_curriculum import load_rollin_specs
from general_terrain.train_m6r_contact_distillation import (
    DEFAULT_PROTOCOL,
    build_distillation_dataset,
    candidate_gate,
    critic_hash,
    load_protocol,
)


def test_contact_dataset_is_narrow_and_anchored() -> None:
    """接触教師行が七十六歩で錨観測がより多いことを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    specs = load_rollin_specs(protocol.phase_reset_manifest_path)
    observations, actions, anchors = build_distillation_dataset(
        specs,
        start_fraction=protocol.start_fraction,
        end_fraction=protocol.end_fraction,
        anchor_stride=protocol.anchor_stride,
    )
    assert observations.shape == (76, 95)
    assert actions.shape == (76, 6)
    assert anchors.shape[0] > observations.shape[0]


def test_candidate_gate_needs_contact_and_flat_success() -> None:
    """接触改善だけで平地回帰候補を採用しないことを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    band = {"success_count": 1}
    assert candidate_gate(
        band,
        {"success_count": 3, "hard_fall_count": 0},
        protocol,
    )
    assert not candidate_gate(
        band,
        {"success_count": 2, "hard_fall_count": 0},
        protocol,
    )


def test_source_critic_hash_is_stable() -> None:
    """同じ源学生を二回読むとcriticハッシュが一致することを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    first = PPO.load(protocol.source_model_path, device="cpu")
    second = PPO.load(protocol.source_model_path, device="cpu")
    assert critic_hash(first) == critic_hash(second)
