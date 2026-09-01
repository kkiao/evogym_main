"""M4Rの零学生更新、早期介入、分離救援課程の契約を検査する。"""

from __future__ import annotations

import unittest
from pathlib import Path

from general_terrain.m4r_learner_distribution_teacher import (
    trigger_config_from_payload,
)
from general_terrain.manifest_training_rescue_teacher import (
    load_training_teacher_manifest,
)
from general_terrain.run_m4r_teacher_repair import (
    build_repair_gate,
    load_protocol,
    route_candidates,
)


class M4RTeacherRepairTests(unittest.TestCase):
    """教師修復だけを許可し学生更新と評価教師を禁止する。"""

    def test_protocol_freezes_three_courses_and_zero_updates(self) -> None:
        """三分離課程、零更新、零評価アクセス、零最終介入を固定する。"""
        protocol = load_protocol()
        self.assertEqual(
            set(protocol["separate_recovery_courses"]),
            {
                "pre_hurdle_safety_intercept",
                "hurdle_contact_deformation",
                "landing_recovery_stall",
            },
        )
        self.assertEqual(protocol["probe_student_update_steps"], 0)
        self.assertEqual(protocol["protected_original_student_update_steps"], 0)
        self.assertEqual(protocol["teacher_weight_update_steps"], 0)
        self.assertEqual(protocol["ppo_training_steps"], 0)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)
        self.assertFalse(protocol["final_student_test_teacher_enabled"])
        self.assertEqual(protocol["teacher_interventions_in_final_student_test"], 0)

    def test_trigger_profiles_hold_teacher_and_allow_global_risk(self) -> None:
        """危険姿勢は地形可視性に依存せず終端まで連続接管できる。"""
        protocol = load_protocol()
        for payload in protocol["trigger_profiles"]:
            config = trigger_config_from_payload(payload)
            self.assertFalse(config.pre_recovery_danger_requires_local_terrain)
            self.assertEqual(config.release_safe_steps, 1200)
            self.assertEqual(config.maximum_teacher_steps, 1200)
            self.assertLess(config.exit_orientation, config.entry_orientation)

    def test_v8_pre_action_shield_is_explicit_and_zero_update(self) -> None:
        """V8は前動作接管と十分な教師予算を明示し零更新を維持する。"""
        protocol = load_protocol(
            Path("config/m4r_learner_distribution_teacher_repair_protocol_v8.json")
        )
        self.assertEqual(protocol["controller_mode"], "verified_portfolio")
        for payload in protocol["trigger_profiles"]:
            config = trigger_config_from_payload(payload)
            self.assertFalse(config.disagreement_requires_local_terrain)
            self.assertEqual(config.disagreement_streak_steps, 1)
            self.assertIsNone(config.maximum_student_prefix_steps)
            self.assertEqual(config.maximum_teacher_steps, 2000)
            self.assertEqual(config.release_safe_steps, 2000)
        self.assertEqual(protocol["probe_student_update_steps"], 0)
        self.assertEqual(protocol["teacher_weight_update_steps"], 0)
        self.assertFalse(protocol["final_student_test_teacher_enabled"])

    def test_route_bank_is_bounded_and_starts_from_proven_routes(self) -> None:
        """経路候補は上限内で旧実証経路をすべて含む。"""
        protocol = load_protocol()
        manifest = load_training_teacher_manifest(
            protocol["source_teacher_manifest_path"]
        )
        candidates = route_candidates(protocol, manifest)
        self.assertLessEqual(
            len(candidates), protocol["route_candidate_limit_per_position"]
        )
        candidate_ids = {candidate.candidate_id for candidate in candidates}
        for route in manifest.routes.values():
            from general_terrain.search_takeover_recalibration_rescue import (
                RecalibratedControllerCandidate,
            )

            self.assertIn(
                RecalibratedControllerCandidate(**route.__dict__).candidate_id,
                candidate_ids,
            )

    def test_repair_gate_requires_nine_robust_positions_and_zero_falls(self) -> None:
        """正確成功だけでなく早期快照成功と無転倒を同時に要求する。"""
        protocol = load_protocol()
        positions = []
        for position in range(20, 31):
            robust = position < 29
            positions.append(
                {
                    "selection_status": (
                        "robust_success" if robust else "safe_fallback"
                    ),
                    "selected_exact_evaluation": {
                        "success": robust,
                        "hard_fall": False,
                    },
                    "selected_perturbation_evaluations": (
                        [{"hard_fall": False}] if robust else []
                    ),
                }
            )
        profile_result = {
            "screen": {"grounded_trigger_count": 0},
            "position_results": positions,
        }
        gate = build_repair_gate(
            profile_result,
            requirements=protocol["repair_gate"],
            sources_unchanged=True,
        )
        self.assertTrue(gate["gate_passed"])
        positions[0]["selected_exact_evaluation"]["hard_fall"] = True
        failed = build_repair_gate(
            profile_result,
            requirements=protocol["repair_gate"],
            sources_unchanged=True,
        )
        self.assertFalse(failed["gate_passed"])


if __name__ == "__main__":
    unittest.main()
