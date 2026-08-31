"""M2.3.5aの位相リセット選択と目録境界を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.audit_phase_reset_curriculum import (
    CURRICULUM_PHASES,
    select_phase_reset_steps,
)


class PhaseResetCurriculumTests(unittest.TestCase):
    """成功軌跡を重複しない四位相開始点へ変換する。"""

    def setUp(self) -> None:
        """三つの連続教師位相を固定入力として用意する。"""
        self.segments = [
            {
                "phase": "pre_hurdle",
                "start": 100,
                "end_exclusive": 180,
                "steps": 80,
            },
            {
                "phase": "hurdle_deformation",
                "start": 180,
                "end_exclusive": 280,
                "steps": 100,
            },
            {
                "phase": "post_clearance_recovery",
                "start": 280,
                "end_exclusive": 780,
                "steps": 500,
            },
        ]

    def test_selects_four_ordered_phase_resets(self) -> None:
        """引き継ぎ、変形、回復、終端安定の順で代表歩を返す。"""
        result = select_phase_reset_steps(self.segments, stable_window_steps=64)
        self.assertEqual(tuple(result), CURRICULUM_PHASES)
        self.assertEqual(result["pre_hurdle"], 100)
        self.assertEqual(result["hurdle_deformation"], 205)
        self.assertEqual(result["post_clearance_recovery"], 405)
        self.assertEqual(result["stable_finish"], 716)
        self.assertEqual(len(set(result.values())), 4)

    def test_rejects_missing_required_phase(self) -> None:
        """回復区間のない失敗軌跡を目録へ入れない。"""
        with self.assertRaises(ValueError):
            select_phase_reset_steps(self.segments[:-1])

    def test_rejects_empty_stable_window(self) -> None:
        """安定終端窓の零歩指定を拒否する。"""
        with self.assertRaises(ValueError):
            select_phase_reset_steps(self.segments, stable_window_steps=0)


if __name__ == "__main__":
    unittest.main()
