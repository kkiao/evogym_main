"""M4R前動作屏障の連続収集門と零更新契約を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.run_m4r_continuous_validation import build_gate, load_protocol


class M4RContinuousValidationTests(unittest.TestCase):
    """訓練専用屏障の成功、安全、厳密再生、評価隔離を検査する。"""

    def test_protocol_keeps_teacher_out_of_evaluation_and_zero_updates(self) -> None:
        """連続復核規約が評価教師と全重み更新を禁止する。"""
        protocol = load_protocol()
        self.assertEqual(protocol["train_collection_episodes"], 11)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)
        self.assertEqual(protocol["probe_student_update_steps"], 0)
        self.assertEqual(protocol["teacher_weight_update_steps"], 0)
        self.assertFalse(protocol["validation_teacher_enabled"])
        self.assertFalse(protocol["holdout_teacher_enabled"])
        self.assertFalse(protocol["final_student_test_teacher_enabled"])
        self.assertEqual(protocol["teacher_interventions_in_final_student_test"], 0)

    def test_gate_requires_nine_successes_zero_falls_and_eleven_intercepts(self) -> None:
        """九成功、零転倒、十一零歩接管、全分岐厳密再生を同時要求する。"""
        protocol = load_protocol()
        episodes = [
            {
                "events": [
                    {
                        "event": "start",
                        "step": 0,
                        "reason": "teacher_disagreement",
                        "upper_body_grounded": False,
                    }
                ]
            }
            for _ in range(11)
        ]
        collection = {
            "collection_success_count": 9,
            "hard_fall_count": 0,
            "accepted_branch_count": 9,
            "episodes": episodes,
        }
        replay_rows = [
            {
                "replay_success": True,
                "all_step_observations_exact": True,
                "single_continuous_teacher_segment": True,
            }
            for _ in range(9)
        ]
        files = {
            "failed_branch_files_absent": True,
            "accepted_branch_files_complete": True,
        }
        gate = build_gate(
            collection,
            replay_rows,
            files,
            requirements=protocol["gate"],
            source_files_unchanged=True,
        )
        self.assertTrue(gate["gate_passed"])
        self.assertFalse(gate["eligible_as_off_distribution_physical_recovery_evidence"])
        episodes[0]["events"][0]["step"] = 1
        failed = build_gate(
            collection,
            replay_rows,
            files,
            requirements=protocol["gate"],
            source_files_unchanged=True,
        )
        self.assertFalse(failed["gate_passed"])


if __name__ == "__main__":
    unittest.main()
