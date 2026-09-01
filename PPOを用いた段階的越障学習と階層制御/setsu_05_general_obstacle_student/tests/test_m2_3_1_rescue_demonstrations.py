"""M2.3.1の示範目録、連続区間、段階門を検証する。"""

from __future__ import annotations

import inspect
from pathlib import Path
import unittest

import numpy as np

import general_terrain.audit_rescue_demonstrations as audit_module
from general_terrain.audit_rescue_demonstrations import (
    HURDLE_DEFORMATION_PHASE,
    POST_CLEARANCE_RECOVERY_PHASE,
    PRE_HURDLE_PHASE,
    build_m2_3_1_gate,
    find_true_segments,
    load_and_validate_branch_arrays,
    load_rescue_demo_manifest,
    validate_source_metadata,
)


class M231RescueDemonstrationTests(unittest.TestCase):
    """成功分岐だけが次段階候補になることを検査する。"""

    def test_frozen_manifest_contains_only_four_valid_v2_candidates(self):
        manifest = load_rescue_demo_manifest()
        self.assertEqual(manifest.split, "train")
        self.assertEqual(len(manifest.candidates), 4)
        self.assertEqual(
            {candidate.seed for candidate in manifest.candidates},
            {100004, 100001, 100000, 100008},
        )
        self.assertTrue(
            all(candidate.run_name.endswith("_v2") for candidate in manifest.candidates)
        )
        self.assertIn("m2_1_preventive_near6_v1", manifest.excluded_sources)
        self.assertIn(
            "m2_3_prefix_rescue_teacher_seed7_v1",
            manifest.excluded_sources,
        )

    def test_true_segments_preserve_contiguous_teacher_sequences(self):
        mask = np.asarray([False, True, True, False, True, True, True], dtype=bool)
        self.assertEqual(find_true_segments(mask), ((1, 3), (4, 7)))
        self.assertEqual(find_true_segments(np.zeros(3, dtype=bool)), ())

    def test_candidate_arrays_match_controller_actions_and_dimensions(self):
        candidate = load_rescue_demo_manifest().candidates[0]
        arrays, audit = load_and_validate_branch_arrays(candidate)
        self.assertEqual(arrays["observations"].shape[1], 95)
        self.assertEqual(arrays["executed_actions"].shape[1], 6)
        self.assertEqual(audit["teacher_control_steps"], 647)
        self.assertEqual(audit["maximum_teacher_action_difference"], 0.0)
        self.assertEqual(audit["maximum_student_action_difference"], 0.0)

    def test_source_metadata_confirms_success_without_hard_fall(self):
        candidate = load_rescue_demo_manifest().candidates[0]
        metadata = validate_source_metadata(candidate)
        self.assertTrue(metadata["summary_success_confirmed"])
        self.assertTrue(metadata["sidecar_success_confirmed"])
        self.assertEqual(metadata["recorded_rescue_count"], 1)

    def test_gate_requires_all_replays_and_three_teacher_phases(self):
        rows = []
        for _ in range(4):
            rows.append(
                {
                    "replay_success": True,
                    "hard_fall": False,
                    "all_step_observations_exact": True,
                    "array_audit": {
                        "observation_shape": [95],
                        "action_shape": [6],
                    },
                    "teacher_phase_step_counts": {
                        PRE_HURDLE_PHASE: 10,
                        HURDLE_DEFORMATION_PHASE: 20,
                        POST_CLEARANCE_RECOVERY_PHASE: 30,
                        "post_recovery_stall": 0,
                    },
                }
            )
        self.assertTrue(build_m2_3_1_gate(rows)["gate_passed"])
        rows[0]["teacher_phase_step_counts"][POST_CLEARANCE_RECOVERY_PHASE] = 0
        for row in rows[1:]:
            row["teacher_phase_step_counts"][POST_CLEARANCE_RECOVERY_PHASE] = 0
        self.assertFalse(build_m2_3_1_gate(rows)["gate_passed"])

    def test_audit_module_contains_no_training_update_entry(self):
        source = Path(inspect.getsourcefile(audit_module)).read_text(encoding="utf-8")
        self.assertNotIn(".learn(", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("optimizer", source.lower())


if __name__ == "__main__":
    unittest.main()
