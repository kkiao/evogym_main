"""M2.3.4対話的訂正規約と出口判定を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.train_interactive_correction_rescue_teacher import (
    _evaluate_gate,
    load_interactive_correction_protocol,
)


class InteractiveCorrectionTests(unittest.TestCase):
    """凍結上限と二組の成功・転倒条件を確認する。"""

    def test_frozen_protocol_loads_expected_boundary(self) -> None:
        """規約が四回収集、二回更新、PPO零を固定する。"""
        protocol = load_interactive_correction_protocol()
        self.assertEqual(protocol.maximum_collection_episodes, 4)
        self.assertEqual(protocol.epochs, 2)
        self.assertEqual(protocol.ppo_training_steps, 0)
        self.assertEqual(protocol.validation_episodes, 0)
        self.assertEqual(protocol.holdout_episodes, 0)

    def test_gate_requires_both_evaluation_sets(self) -> None:
        """出典状態だけ良くても十一リセット状態不足なら通過させない。"""
        gate = {
            "minimum_source_state_success_count": 3,
            "minimum_reset_state_success_count": 4,
            "maximum_source_state_hard_fall_count": 0,
            "maximum_reset_state_hard_fall_count": 1,
        }
        result = _evaluate_gate(
            {"success_count": 3, "hard_fall_count": 0},
            {"success_count": 3, "hard_fall_count": 0},
            gate,
        )
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["checks"]["reset_success"])

    def test_gate_passes_only_complete_boundary(self) -> None:
        """四条件を満たした時だけM2.4候補にする。"""
        gate = {
            "minimum_source_state_success_count": 3,
            "minimum_reset_state_success_count": 4,
            "maximum_source_state_hard_fall_count": 0,
            "maximum_reset_state_hard_fall_count": 1,
        }
        result = _evaluate_gate(
            {"success_count": 4, "hard_fall_count": 0},
            {"success_count": 5, "hard_fall_count": 1},
            gate,
        )
        self.assertTrue(result["gate_passed"])
        self.assertTrue(result["eligible_for_m2_4"])


if __name__ == "__main__":
    unittest.main()
