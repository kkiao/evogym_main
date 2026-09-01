"""M2.3.5b位相横断教師の規約、選択、門を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.train_phase_crossing_teacher import (
    _candidate_key,
    evaluate_gate,
    load_protocol,
)


class PhaseCrossingTeacherTests(unittest.TestCase):
    """有界試行と三組評価の全条件を小さい辞書で確認する。"""

    def test_protocol_freezes_training_and_data_boundaries(self) -> None:
        """模倣8回、PPO4992歩、検証と留保零を固定する。"""
        protocol = load_protocol()
        self.assertEqual(protocol["behavior_cloning_epochs"], 8)
        self.assertEqual(protocol["ppo_training_steps"], 4992)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)

    def test_candidate_key_prioritizes_success_then_safety(self) -> None:
        """二評価の成功総数を最優先し同数なら転倒を抑える。"""
        safe = _candidate_key(
            {"success_count": 6, "hard_fall_count": 0},
            {"success_count": 5, "hard_fall_count": 0},
        )
        unsafe = _candidate_key(
            {"success_count": 6, "hard_fall_count": 1},
            {"success_count": 5, "hard_fall_count": 0},
        )
        self.assertGreater(safe, unsafe)

    def test_gate_requires_phase_and_reset_results(self) -> None:
        """位相二組が合格しても十一状態不足ならM2.4を開けない。"""
        requirements = {
            "minimum_phase_exact_success_count": 16,
            "minimum_phase_impulse_success_count": 15,
            "maximum_phase_exact_hard_fall_count": 0,
            "maximum_phase_impulse_hard_fall_count": 0,
            "minimum_reset_state_success_count": 4,
            "maximum_reset_state_hard_fall_count": 1,
        }
        result = evaluate_gate(
            {"success_count": 16, "hard_fall_count": 0},
            {"success_count": 15, "hard_fall_count": 0},
            {"success_count": 3, "hard_fall_count": 0},
            requirements,
        )
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["eligible_for_m2_4"])


if __name__ == "__main__":
    unittest.main()
