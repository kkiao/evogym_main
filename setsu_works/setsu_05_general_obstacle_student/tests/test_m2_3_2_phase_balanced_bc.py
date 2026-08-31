"""M2.3.2の凍結規約、段階等重み損失、更新範囲を検証する。"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

import general_terrain.train_phase_balanced_rescue_teacher as training_module
from general_terrain.train_phase_balanced_rescue_teacher import (
    actor_trainable_parameters,
    equal_phase_mean_loss,
    evaluate_m2_3_2_gate,
    load_phase_balanced_sequences,
    load_training_protocol,
)


class M232PhaseBalancedBCTests(unittest.TestCase):
    """M2.3.2が固定模倣範囲から逸脱しないことを検査する。"""

    def test_protocol_freezes_four_epochs_and_excludes_failed_checkpoint(self):
        protocol = load_training_protocol()
        self.assertEqual(protocol.epochs, 4)
        self.assertEqual(protocol.learning_rate, 1e-4)
        self.assertEqual(protocol.maximum_gradient_norm, 0.5)
        self.assertEqual(protocol.ppo_training_steps, 0)
        self.assertEqual(protocol.validation_episodes, 0)
        self.assertEqual(protocol.holdout_episodes, 0)
        self.assertEqual(
            protocol.excluded_checkpoint_run,
            "m2_3_prefix_rescue_teacher_seed7_v1",
        )

    def test_loader_returns_four_contiguous_sequences_and_audited_counts(self):
        protocol = load_training_protocol()
        sequences, metadata, _ = load_phase_balanced_sequences(protocol)
        self.assertEqual(len(sequences), 4)
        self.assertEqual(metadata["total_teacher_steps_per_epoch"], 2_789)
        self.assertEqual(metadata["phase_step_counts"]["pre_hurdle"], 470)
        self.assertEqual(
            metadata["phase_step_counts"]["hurdle_deformation"],
            359,
        )
        self.assertEqual(
            metadata["phase_step_counts"]["post_clearance_recovery"],
            1_960,
        )

    def test_equal_phase_loss_does_not_weight_by_raw_step_count(self):
        step_losses = torch.tensor([1.0, 1.0, 3.0, 6.0, 6.0, 6.0])
        phase_codes = torch.tensor([0, 0, 1, 2, 2, 2])
        total, phase_losses = equal_phase_mean_loss(
            step_losses,
            phase_codes,
            required_phase_codes=(0, 1, 2),
        )
        self.assertAlmostEqual(
            float(total),
            (1.0 + 3.0 + 6.0) / 3.0,
            places=6,
        )
        self.assertEqual(float(phase_losses[0]), 1.0)
        self.assertEqual(float(phase_losses[1]), 3.0)
        self.assertEqual(float(phase_losses[2]), 6.0)

    def test_actor_parameter_scope_excludes_critic(self):
        actor_lstm = torch.nn.Linear(2, 2)
        policy_net = torch.nn.Linear(2, 2)
        action_net = torch.nn.Linear(2, 1)
        critic = torch.nn.Linear(2, 1)
        model = SimpleNamespace(
            policy=SimpleNamespace(
                lstm_actor=actor_lstm,
                mlp_extractor=SimpleNamespace(policy_net=policy_net),
                action_net=action_net,
                lstm_critic=critic,
            )
        )
        selected = {id(parameter) for parameter in actor_trainable_parameters(model)}
        expected = {
            id(parameter)
            for module in (actor_lstm, policy_net, action_net)
            for parameter in module.parameters()
        }
        critic_ids = {id(parameter) for parameter in critic.parameters()}
        self.assertEqual(selected, expected)
        self.assertTrue(selected.isdisjoint(critic_ids))

    def test_gate_requires_success_safety_stall_and_recovery_phase(self):
        protocol = load_training_protocol()
        passed = evaluate_m2_3_2_gate(
            {
                "success_count": 4,
                "hard_fall_count": 1,
                "safe_stall_count": 6,
                "phase_step_counts": {"post_clearance_recovery": 1},
            },
            protocol.gate,
        )
        self.assertTrue(passed["gate_passed"])
        failed = evaluate_m2_3_2_gate(
            {
                "success_count": 4,
                "hard_fall_count": 1,
                "safe_stall_count": 6,
                "phase_step_counts": {"post_clearance_recovery": 0},
            },
            protocol.gate,
        )
        self.assertFalse(failed["gate_passed"])

    def test_training_module_contains_no_ppo_update_entry(self):
        source = Path(inspect.getsourcefile(training_module)).read_text(
            encoding="utf-8"
        )
        self.assertNotIn(".learn(", source)
        self.assertNotIn("total_timesteps", source)


if __name__ == "__main__":
    unittest.main()
