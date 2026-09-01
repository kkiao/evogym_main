"""M2.3.5fの二課程教師目録と最終試験隔離を検査する。"""

from __future__ import annotations

import json
import unittest

from general_terrain.freeze_rescue_teacher_manifest import (
    evaluate_gate,
    load_protocol,
    takeover_routes_from_search,
)


class TeacherManifestGateTests(unittest.TestCase):
    """教師目録が独立再生と無教師最終試験を固定することを確認する。"""

    def test_protocol_requires_two_replays_and_closed_test_splits(self) -> None:
        """独立二回、検証零、留保零を固定する。"""
        protocol = load_protocol()
        self.assertEqual(protocol["independent_replay_repetitions"], 2)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)

    def test_takeover_routes_keep_nine_safe_positions(self) -> None:
        """合格探索から九位置の安全経路だけを読み込む。"""
        protocol = load_protocol()
        summary = json.loads(
            protocol["takeover_search_summary_path"].read_text(encoding="utf-8")
        )
        routes = takeover_routes_from_search(summary)
        self.assertEqual(len(routes), 9)
        self.assertEqual(set(range(20, 31)) - set(routes), {21, 22})

    def test_gate_requires_reproducible_takeover_results(self) -> None:
        """成功数が足りても再現性が偽ならM2.4を開けない。"""
        requirements = {
            "minimum_phase_exact_success_count": 16,
            "minimum_phase_impulse_success_count": 16,
            "maximum_phase_exact_hard_fall_count": 0,
            "maximum_phase_impulse_hard_fall_count": 0,
            "minimum_reset_state_success_count": 4,
            "maximum_reset_state_hard_fall_count": 1,
            "require_reset_reproducibility": True,
        }
        result = evaluate_gate(
            {"success_count": 16, "hard_fall_count": 0},
            {"success_count": 16, "hard_fall_count": 0},
            {"success_count": 9, "hard_fall_count": 0, "all_reproducible": False},
            requirements,
        )
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["eligible_for_m2_4"])


if __name__ == "__main__":
    unittest.main()
