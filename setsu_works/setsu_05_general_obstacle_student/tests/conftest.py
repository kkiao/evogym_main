"""提出物に含めない中間学習証拠へ依存する履歴試験を収集対象外にする。"""

# これらの試験ソースは研究経緯の確認用に残すが、数GB規模の中間軌跡と
# 全チェックポイントを提出物へ含めないため、独立再現試験には数えない。
collect_ignore = [
    "test_m2_2_rescue_reset.py",
    "test_m2_3_1_rescue_demonstrations.py",
    "test_m2_3_2_phase_balanced_bc.py",
    "test_m2_3_4_interactive_correction.py",
    "test_m2_3_4_perturbed_recovery.py",
    "test_m2_3_5_phase_controller_rescue.py",
    "test_m2_3_5_phase_crossing_teacher.py",
    "test_m2_3_5_phase_routed_rescue_teacher.py",
    "test_m2_3_5_takeover_recalibration.py",
    "test_m2_3_5_teacher_manifest_gate.py",
    "test_m2_4_manifest_collection.py",
    "test_m3_stratified_student.py",
    "test_m4_probe_rescue_preflight.py",
    "test_m4n_dataset_novelty_audit.py",
    "test_m4r_continuous_validation.py",
    "test_m5_reverse_curriculum.py",
    "test_m6_contact_bridge_audit.py",
    "test_m6_dense_handoff.py",
    "test_m6l_learner_landing.py",
    "test_m6r_contact_distillation.py",
    "test_m6s_student_success_safety.py",
]
