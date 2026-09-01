"""M2.3.4摂動回復の誤差尺度と境界損失を検査する。"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from general_terrain.audit_rescue_demonstrations import PHASE_CODES
from general_terrain.train_perturbed_recovery_rescue_teacher import (
    _handoff_balanced_loss,
    load_perturbed_recovery_protocol,
)


class PerturbedRecoveryTests(unittest.TestCase):
    """摂動上限と四群等重みが固定されることを確認する。"""

    def test_protocol_uses_bounded_student_error_noise(self) -> None:
        """学生誤差の四分の一と絶対上限を同時に固定する。"""
        protocol = load_perturbed_recovery_protocol()
        self.assertEqual(protocol.maximum_collection_episodes, 4)
        self.assertEqual(protocol.noise_residual_rms_fraction, 0.25)
        self.assertEqual(protocol.maximum_absolute_noise, 0.08)
        self.assertEqual(protocol.ppo_training_steps, 0)

    def test_handoff_window_has_equal_group_weight(self) -> None:
        """歩数の異なる四群を各群平均後に等しく合成する。"""
        step_losses = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
        phase_codes = torch.tensor(
            [
                PHASE_CODES["pre_hurdle"],
                PHASE_CODES["pre_hurdle"],
                PHASE_CODES["pre_hurdle"],
                PHASE_CODES["hurdle_deformation"],
                PHASE_CODES["post_clearance_recovery"],
            ]
        )
        loss, groups = _handoff_balanced_loss(step_losses, phase_codes, 2)
        self.assertAlmostEqual(float(groups["handoff_window"]), 2.0)
        self.assertAlmostEqual(float(groups["remaining_pre_hurdle"]), 5.0)
        self.assertAlmostEqual(float(groups["hurdle_deformation"]), 7.0)
        self.assertAlmostEqual(float(groups["post_clearance_recovery"]), 9.0)
        self.assertAlmostEqual(float(loss), 5.75)

    def test_handoff_loss_rejects_missing_group(self) -> None:
        """後続段階が欠けた系列を更新へ使用しない。"""
        with self.assertRaises(ValueError):
            _handoff_balanced_loss(
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor(
                    [
                        PHASE_CODES["pre_hurdle"],
                        PHASE_CODES["pre_hurdle"],
                        PHASE_CODES["hurdle_deformation"],
                    ]
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
