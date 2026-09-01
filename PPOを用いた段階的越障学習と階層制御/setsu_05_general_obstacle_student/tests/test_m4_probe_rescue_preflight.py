"""M4学習器誘導状態救援の零更新規約と判定門を検査する。"""

from __future__ import annotations

from pathlib import Path
import unittest

from general_terrain.rescue_reset_manifest import RescueResetSpec
from general_terrain.run_m4_probe_rescue_preflight import (
    audit_trigger_state,
    build_gate,
    load_protocol,
)


class M4ProbeRescuePreflightTests(unittest.TestCase):
    """隔離探針、訓練限定教師、厳密再生門の契約を確認する。"""

    def test_protocol_freezes_no_update_training_only_review(self) -> None:
        """十一訓練回、零評価回、零重み更新、零最終介入を固定する。"""
        protocol = load_protocol()
        self.assertEqual(protocol["train_collection_episodes"], 11)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)
        self.assertEqual(protocol["probe_student_update_steps"], 0)
        self.assertEqual(protocol["protected_original_student_update_steps"], 0)
        self.assertEqual(protocol["teacher_update_steps"], 0)
        self.assertFalse(protocol["validation_teacher_enabled"])
        self.assertFalse(protocol["holdout_teacher_enabled"])
        self.assertFalse(protocol["final_student_test_teacher_enabled"])
        self.assertEqual(protocol["teacher_interventions_in_final_student_test"], 0)
        self.assertIn("probe_only", protocol["probe_disposition"])

    def test_trigger_audit_detects_learner_induced_state_change(self) -> None:
        """開始歩または物理状態が異なる場合だけ誘導状態変化として数える。"""
        reference = RescueResetSpec(
            seed=100009,
            start_runway_voxels=20,
            prefix_steps=261,
            trigger_reason="teacher_disagreement",
            x_position=1.4,
            orientation_error=0.2,
            angular_velocity=0.01,
            stall_steps=0,
            raw_clearances=0,
            recovered_obstacles=0,
        )
        event = {
            "step": 262,
            "event": "start",
            "reason": "teacher_disagreement",
            "rescue_id": 1,
            "teacher_stage": "manifest_takeover_x20:approach",
            "x_position": 1.4,
            "orientation_error": 0.2,
            "angular_velocity": 0.01,
            "stall_steps": 0,
            "raw_clearances": 0,
            "recovered_obstacles": 0,
            "upper_body_grounded": False,
        }
        episode = {
            "seed": 100009,
            "start_runway_voxels": 20,
            "events": [event],
        }
        row = audit_trigger_state(episode, reference)
        self.assertTrue(row["comparison_available"])
        self.assertTrue(row["learner_induced_state_changed"])
        self.assertEqual(row["step_delta"], 1)
        event["step"] = 261
        unchanged = audit_trigger_state(episode, reference)
        self.assertFalse(unchanged["learner_induced_state_changed"])

    def test_gate_rejects_any_hard_fall(self) -> None:
        """十分な成功分岐があっても硬転倒が一回あれば前置門を閉じる。"""
        requirements = load_protocol()["gate"]
        episodes = [
            {
                "start_runway_voxels": position,
                "branch_accepted": True,
                "events": [{"event": "start"}],
            }
            for position in range(20, 24)
        ]
        collection = {
            "collection_success_count": 4,
            "hard_fall_count": 0,
            "accepted_branch_count": 4,
            "episodes": episodes,
        }
        replay_rows = [
            {
                "replay_success": True,
                "all_step_observations_exact": True,
                "single_continuous_teacher_segment": True,
            }
            for _ in range(4)
        ]
        trigger_rows = [
            {"learner_induced_state_changed": True} for _ in range(4)
        ]
        failed_file_audit = {
            "failed_branch_files_absent": True,
            "accepted_branch_files_complete": True,
        }
        passed = build_gate(
            collection,
            replay_rows,
            trigger_rows,
            failed_file_audit,
            requirements=requirements,
            source_files_unchanged=True,
        )
        self.assertTrue(passed["gate_passed"])
        collection["hard_fall_count"] = 1
        rejected = build_gate(
            collection,
            replay_rows,
            trigger_rows,
            failed_file_audit,
            requirements=requirements,
            source_files_unchanged=True,
        )
        self.assertFalse(rejected["gate_passed"])
        self.assertFalse(rejected["hard_fall_requirement_passed"])

    def test_runner_contains_no_training_call(self) -> None:
        """前置復核実行器に学習呼出しが存在しないことを静的に確認する。"""
        source_path = (
            Path(__file__).resolve().parents[1]
            / "general_terrain"
            / "run_m4_probe_rescue_preflight.py"
        )
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn(".learn(", source)


if __name__ == "__main__":
    unittest.main()
