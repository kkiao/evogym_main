"""M2.3.5eの再校正候補、規約、合格門を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.search_takeover_recalibration_rescue import (
    evaluate_gate,
    generate_candidates,
    load_protocol,
)


class TakeoverRecalibrationTests(unittest.TestCase):
    """引継ぎ時再初期化が有界かつ訓練区分限定であることを確認する。"""

    def test_protocol_requires_reinitialization_and_split_isolation(self) -> None:
        """再初期化、候補上限、検証・留保零を固定する。"""
        protocol = load_protocol()
        self.assertTrue(protocol["reinitialize_at_takeover"])
        self.assertEqual(protocol["candidate_limit_per_position"], 192)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)

    def test_candidates_include_both_clearance_families(self) -> None:
        """第一・第二越壁系列を含み識別子重複を持たない。"""
        candidates = generate_candidates(load_protocol())
        identifiers = [candidate.candidate_id for candidate in candidates]
        self.assertEqual({candidate.clearance_family for candidate in candidates}, {"first", "second"})
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertLessEqual(len(candidates), 192)

    def test_gate_requires_four_safe_successes(self) -> None:
        """三成功では引継ぎ救援門を開かない。"""
        rows = [
            {
                "selected": {},
                "selected_evaluation": {"success": True, "hard_fall": False},
            }
            for _ in range(3)
        ]
        result = evaluate_gate(
            rows,
            {
                "minimum_selected_success_count": 4,
                "maximum_selected_hard_fall_count": 0,
            },
        )
        self.assertFalse(result["gate_passed"])


if __name__ == "__main__":
    unittest.main()
