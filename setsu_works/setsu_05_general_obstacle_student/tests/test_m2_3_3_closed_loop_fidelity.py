"""M2.3.3閉ループ保真度診断の純粋集計処理を検査する。"""

from __future__ import annotations

import unittest

import numpy as np

from general_terrain.audit_rescue_demonstrations import PHASE_CODES
from general_terrain.diagnose_closed_loop_imitation_fidelity import (
    first_threshold_crossing,
    summarize_action_errors,
)


class ClosedLoopFidelityDiagnosticTests(unittest.TestCase):
    """閾値位置と段階別誤差を小さな固定配列で検査する。"""

    def test_first_threshold_crossing_is_strict(self) -> None:
        """閾値と同値の要素を乖離として扱わない。"""
        values = np.asarray([0.0, 0.01, 0.02], dtype=np.float64)
        self.assertEqual(first_threshold_crossing(values, 0.01), 2)
        self.assertIsNone(first_threshold_crossing(values, 0.02))

    def test_summary_keeps_equal_phase_metrics_separate(self) -> None:
        """三つの必須段階を系列長に関係なく個別集計する。"""
        squared = np.asarray([0.0, 0.02, 0.03, 0.04], dtype=np.float64)
        maximum = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        phases = np.asarray(
            [
                PHASE_CODES["pre_hurdle"],
                PHASE_CODES["hurdle_deformation"],
                PHASE_CODES["post_clearance_recovery"],
                PHASE_CODES["post_clearance_recovery"],
            ],
            dtype=np.int8,
        )
        result = summarize_action_errors(squared, maximum, phases)
        self.assertAlmostEqual(result["mean_squared_error"], 0.0225)
        self.assertEqual(result["first_mse_divergence_relative_step"], 1)
        self.assertEqual(
            result["first_maximum_error_divergence_relative_step"],
            2,
        )
        recovery = result["phase_metrics"]["post_clearance_recovery"]
        self.assertEqual(recovery["steps"], 2)
        self.assertAlmostEqual(recovery["mean_squared_error"], 0.035)

    def test_summary_rejects_missing_required_phase(self) -> None:
        """必須段階が欠けた入力を診断成功として扱わない。"""
        with self.assertRaises(ValueError):
            summarize_action_errors(
                np.asarray([0.0, 0.1]),
                np.asarray([0.0, 0.2]),
                np.asarray(
                    [
                        PHASE_CODES["pre_hurdle"],
                        PHASE_CODES["hurdle_deformation"],
                    ]
                ),
            )


if __name__ == "__main__":
    unittest.main()
