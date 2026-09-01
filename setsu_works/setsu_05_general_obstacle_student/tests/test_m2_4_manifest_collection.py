"""M2.4の凍結教師目録、連続接管設定、評価隔離を検査する。"""

from __future__ import annotations

import unittest

from general_terrain.manifest_training_rescue_teacher import (
    load_training_teacher_manifest,
)
from general_terrain.rescue_profiles import (
    M2_4_MANIFEST_PROFILE,
    get_rescue_profile,
)
from general_terrain.run_m2_4_manifest_collection import load_protocol


class ManifestCollectionTests(unittest.TestCase):
    """九経路二代替の目録と学生試験無教師契約を確認する。"""

    def test_manifest_is_training_only_and_covers_eleven_positions(self) -> None:
        """訓練経路と未対応位置の和集合が十一位置へ厳密に一致する。"""
        manifest = load_training_teacher_manifest()
        self.assertEqual(manifest.stage, "hurdle_single")
        self.assertEqual(manifest.split, "train")
        self.assertEqual(len(manifest.sha256), 64)
        self.assertEqual(set(manifest.routes), set(range(20, 31)) - {21, 22})
        self.assertEqual(manifest.uncovered_positions, (21, 22))
        exported = manifest.as_dict()
        self.assertFalse(exported["validation_teacher_enabled"])
        self.assertFalse(exported["holdout_teacher_enabled"])
        self.assertFalse(exported["final_student_test_teacher_enabled"])
        self.assertEqual(exported["teacher_interventions_in_final_student_test"], 0)

    def test_manifest_profile_keeps_original_entry_boundary(self) -> None:
        """M2.1開始境界を保ち、成功終端まで早期解放しない。"""
        source = get_rescue_profile("m2_1_near6")
        profile = get_rescue_profile(M2_4_MANIFEST_PROFILE)
        self.assertEqual(profile.entry_orientation, source.entry_orientation)
        self.assertEqual(profile.disagreement_threshold, source.disagreement_threshold)
        self.assertEqual(
            profile.disagreement_maximum_rise_offset,
            source.disagreement_maximum_rise_offset,
        )
        self.assertEqual(profile.release_safe_steps, 1200)
        self.assertEqual(profile.maximum_teacher_steps, 1200)

    def test_protocol_freezes_training_only_collection(self) -> None:
        """十一訓練回、零評価回、零重み更新、零最終介入を固定する。"""
        protocol = load_protocol()
        self.assertEqual(protocol["train_episodes"], 11)
        self.assertEqual(protocol["validation_episodes"], 0)
        self.assertEqual(protocol["holdout_episodes"], 0)
        self.assertEqual(protocol["student_update_steps"], 0)
        self.assertEqual(protocol["teacher_update_steps"], 0)
        self.assertFalse(protocol["final_student_test_teacher_enabled"])
        self.assertEqual(protocol["teacher_interventions_in_final_student_test"], 0)


if __name__ == "__main__":
    unittest.main()
