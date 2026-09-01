"""M6接触橋監査のactor一括予測を検査する。"""

from __future__ import annotations

from stable_baselines3 import PPO

from general_terrain.audit_m6_contact_bridge import actor_predictions
from general_terrain.train_m6_dense_handoff import DEFAULT_PROTOCOL, load_protocol


def test_actor_predictions_match_policy_shape() -> None:
    """凍結学生の一括actor出力が六次元であることを確認する。"""
    protocol = load_protocol(DEFAULT_PROTOCOL)
    model = PPO.load(protocol.source_model_path, device="cpu")
    observations = model.observation_space.sample()[None, :]
    predictions = actor_predictions(model, observations)
    assert predictions.shape == (1, 6)
