"""M4N新規性監査の零更新契約とM4A入口門を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.run_m4n_dataset_novelty_audit import build_gate, load_protocol


class M4NDatasetNoveltyAuditTests(unittest.TestCase):
    """安全な新規教師標本だけでは対話集約を解禁しないことを検査する。"""

    def test_protocol_freezes_zero_updates_and_teacher_isolation(self) -> None:
        """M4Nが全重み更新と評価教師を禁止する。"""
        protocol = load_protocol()
        self.assertEqual(protocol["student_update_steps"], 0)
        self.assertEqual(protocol["teacher_update_steps"], 0)
        self.assertEqual(protocol["ppo_training_steps"], 0)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)
        self.assertFalse(protocol["validation_teacher_enabled"])
        self.assertFalse(protocol["holdout_teacher_enabled"])
        self.assertFalse(protocol["final_student_test_teacher_enabled"])
        self.assertEqual(protocol["teacher_interventions_in_final_student_test"], 0)

    def test_gate_keeps_m4a_locked_without_learner_executed_states(self) -> None:
        """新規教師軌跡があっても学生実行状態が零ならM4Aを拒否する。"""
        protocol = load_protocol()
        audit = {
            "maximum_source_replay_observation_difference": 0.0,
            "new_branch_count": 11,
            "overall_novelty": {"novel_sample_ratio": 0.5},
            "position_novelty": {
                "21": {"novel_sample_ratio": 0.5},
                "22": {"novel_sample_ratio": 0.5},
            },
            "new_phase_sample_counts": {
                "pre_hurdle_safety_intercept": 100,
                "hurdle_contact_deformation": 100,
                "landing_recovery_stall": 100,
            },
            "teacher_step_action_disagreement": {"mean": 0.2},
            "new_student_executed_steps": 0,
        }
        gate = build_gate(
            audit,
            requirements=protocol["gate"],
            source_files_unchanged=True,
        )
        self.assertTrue(gate["teacher_demonstration_archive_gate_passed"])
        self.assertFalse(gate["learner_executed_state_requirement_passed"])
        self.assertFalse(gate["eligible_for_m4a"])
        audit["new_student_executed_steps"] = 1
        passed = build_gate(
            audit,
            requirements=protocol["gate"],
            source_files_unchanged=True,
        )
        self.assertTrue(passed["eligible_for_m4a"])


if __name__ == "__main__":
    unittest.main()
