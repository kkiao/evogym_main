"""M2.3.5dの位置経路読込、位相限定、合格門を検査する。"""

from __future__ import annotations

import json
import unittest

from general_terrain.evaluate_phase_routed_rescue_teacher import (
    evaluate_gate,
    load_protocol,
)
from general_terrain.phase_routed_rescue_teacher import route_configs_from_search


class PhaseRoutedRescueTeacherTests(unittest.TestCase):
    """合格済み経路だけを訓練評価へ接続することを確認する。"""

    def test_protocol_keeps_validation_and_holdout_closed(self) -> None:
        """経路起動位相、検証零、留保零を固定する。"""
        protocol = load_protocol()
        self.assertEqual(protocol["route_activation_phase"], "pre_hurdle")
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)

    def test_search_summary_provides_four_routes(self) -> None:
        """制御器探索の合格結果から四位置を読み込む。"""
        protocol = load_protocol()
        summary = json.loads(
            protocol["controller_search_summary_path"].read_text(encoding="utf-8")
        )
        routes = route_configs_from_search(summary)
        self.assertEqual(set(routes), {22, 25, 27, 28})

    def test_gate_requires_reset_success_after_phase_success(self) -> None:
        """位相二組が完全成功しても前置状態不足なら門を閉じる。"""
        requirements = {
            "minimum_phase_exact_success_count": 16,
            "minimum_phase_impulse_success_count": 16,
            "maximum_phase_exact_hard_fall_count": 0,
            "maximum_phase_impulse_hard_fall_count": 0,
            "minimum_reset_state_success_count": 4,
            "maximum_reset_state_hard_fall_count": 1,
        }
        result = evaluate_gate(
            {"success_count": 16, "hard_fall_count": 0},
            {"success_count": 16, "hard_fall_count": 0},
            {"success_count": 3, "hard_fall_count": 0},
            requirements,
        )
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["eligible_for_m2_4"])


if __name__ == "__main__":
    unittest.main()
