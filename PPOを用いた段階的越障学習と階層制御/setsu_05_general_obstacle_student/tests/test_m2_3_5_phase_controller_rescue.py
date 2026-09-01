"""M2.3.5cの候補順序、規約、合格門を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.search_phase_controller_rescue import (
    evaluate_gate,
    generate_candidates,
    load_protocol,
)


class PhaseControllerRescueTests(unittest.TestCase):
    """有界探索が出典と評価隔離を守ることを確認する。"""

    def test_protocol_freezes_search_limit_and_split_isolation(self) -> None:
        """候補上限と検証・留保零を固定する。"""
        protocol = load_protocol()
        self.assertEqual(protocol["candidate_limit_per_position"], 160)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)

    def test_candidate_generation_is_bounded_and_unique(self) -> None:
        """生成候補が上限以内で識別子重複を持たない。"""
        protocol = load_protocol()
        candidates = generate_candidates(protocol)
        identifiers = [candidate.candidate_id for candidate in candidates]
        self.assertLessEqual(len(candidates), 160)
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_gate_requires_all_four_exact_and_impulse_successes(self) -> None:
        """一位置でも欠ける場合は位相救援門を閉じる。"""
        requirements = {
            "minimum_selected_exact_success_count": 4,
            "minimum_selected_impulse_success_count": 4,
            "maximum_selected_exact_hard_fall_count": 0,
            "maximum_selected_impulse_hard_fall_count": 0,
        }
        rows = [
            {
                "selected": {},
                "selected_exact": {"success": True, "hard_fall": False},
                "selected_impulse": {"success": True, "hard_fall": False},
            }
            for _ in range(3)
        ]
        result = evaluate_gate(rows, requirements)
        self.assertFalse(result["gate_passed"])
        self.assertEqual(result["selected_position_count"], 3)


if __name__ == "__main__":
    unittest.main()
