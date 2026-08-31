"""M3の成功分岐限定、四分層損失、更新範囲、教師隔離を検査する。"""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest

import torch

import general_terrain.train_m3_stratified_student as training_module
from general_terrain.repair_m3_safe_student import (
    evaluate_safety_gate,
    load_protocol as load_repair_protocol,
)
from general_terrain.train_m3_stratified_student import (
    STRATUM_NAMES,
    equal_stratum_mean_loss,
    load_m3_sequences,
    load_protocol,
)


class M3StratifiedStudentTests(unittest.TestCase):
    """M3が九成功分岐だけを有界な循環模倣へ使用することを確認する。"""

    def test_protocol_freezes_bounded_student_initialization(self) -> None:
        """四回、低学習率、PPO零、留出零、最終教師停止を固定する。"""
        protocol = load_protocol()
        self.assertEqual(protocol.required_branch_count, 9)
        self.assertEqual(protocol.required_strata, STRATUM_NAMES)
        self.assertEqual(protocol.epochs, 4)
        self.assertEqual(protocol.learning_rate, 5e-5)
        self.assertEqual(protocol.ppo_training_steps, 0)
        self.assertEqual(protocol.holdout_episodes, 0)
        self.assertEqual(protocol.excluded_positions, (21, 22))

    def test_loader_replays_nine_successes_and_populates_all_strata(self) -> None:
        """九系列の全観測が精密再生され、各系列の四分層が非空となる。"""
        sequences, metadata, _ = load_m3_sequences(load_protocol())
        self.assertEqual(len(sequences), 9)
        self.assertEqual(metadata["total_steps_per_epoch"], 9_493)
        self.assertTrue(metadata["all_source_replays_exact"])
        self.assertTrue(metadata["all_strata_nonempty_per_sequence"])
        self.assertTrue(
            all(value > 0 for value in metadata["stratum_step_counts"].values())
        )
        self.assertEqual(
            {sequence.start_runway_voxels for sequence in sequences},
            set(range(20, 31)) - {21, 22},
        )

    def test_equal_stratum_loss_ignores_raw_class_frequency(self) -> None:
        """歩数の多い分層が四分層平均を支配しない。"""
        step_losses = torch.tensor([1.0, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0])
        codes = torch.tensor([0, 0, 1, 2, 3, 3, 3])
        total, losses = equal_stratum_mean_loss(step_losses, codes)
        self.assertAlmostEqual(float(total), (1.0 + 2.0 + 4.0 + 8.0) / 4.0)
        self.assertEqual(float(losses[0]), 1.0)
        self.assertEqual(float(losses[3]), 8.0)

    def test_training_entry_contains_no_ppo_learning_call(self) -> None:
        """M3入口にPPO学習呼出しが存在しない。"""
        source = Path(inspect.getsourcefile(training_module)).read_text(
            encoding="utf-8"
        )
        self.assertNotIn(".learn(", source)
        self.assertNotIn("total_timesteps", source)

    def test_repair_protocol_is_low_change_and_holdout_free(self) -> None:
        """修復候補を四つ、最大二回、学習率1e-5以下、留出零に限定する。"""
        protocol = load_repair_protocol()
        self.assertEqual(protocol["candidate_limit"], 4)
        self.assertEqual(len(protocol["candidates"]), 4)
        self.assertTrue(
            all(row["epochs"] <= 2 for row in protocol["candidates"])
        )
        self.assertTrue(
            all(row["learning_rate"] <= 1e-5 for row in protocol["candidates"])
        )
        self.assertEqual(protocol["ppo_training_steps"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)

    def test_repair_gate_rejects_hard_fall_regression(self) -> None:
        """損失が下がっても学生単独の硬転倒増加を拒否する。"""
        baseline = {
            "success_count": 0,
            "hard_fall_count": 7,
            "mean_raw_clearances": 0.6,
            "mean_recovered_obstacles": 0.4,
        }
        candidate = {
            **baseline,
            "hard_fall_count": 8,
            "controller_mode": "student_only",
            "teacher_module_loaded": False,
            "teacher_interventions": 0,
        }
        gate = evaluate_safety_gate(
            baseline,
            candidate,
            initial_loss=0.02,
            final_loss=0.01,
            actor_changed=True,
            critic_unchanged=True,
        )
        self.assertFalse(gate["hard_fall_non_regression_passed"])
        self.assertFalse(gate["gate_passed"])


if __name__ == "__main__":
    unittest.main()
