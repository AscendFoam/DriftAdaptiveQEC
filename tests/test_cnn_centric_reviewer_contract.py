from __future__ import annotations

import copy
import csv
import json

import pytest

from cnn_fpga.benchmark import cnn_centric_reviewer_contract as contract


def test_report_passes_all_gates_and_mutations() -> None:
    report = contract.build_report()
    assert report["verdict"] == contract.VERDICT
    assert report["gate_summary"] == {"passed": 23, "total": 23}
    assert all(report["gates"].values())
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 23
    assert {case["target_gate"] for case in audit["cases"]} == set(report["gates"])


def test_package_is_preemptive_and_not_falsely_submission_ready() -> None:
    report = contract.build_report()
    assert report["reviewer_context"]["comment_id"] == "PRQ-CNN-1"
    assert report["response_package"]["package_readiness"] == "draft_with_placeholders"
    assert report["response_package"]["missing_information"] == ["ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING"]


def test_current_verdict_is_deletion_invariant_with_learning_dropped() -> None:
    current = contract.build_report()["current_phase6d"]
    assert current["final_verdict"] == "GO_RTL_ONLY"
    assert current["truth_key"] == "multimode=false,rtl=true"
    assert current["learning"]["decision"] == "DROPPED_ABLATION_ONLY"
    assert current["learning"]["direct_evidence"]["changes_overall_verdict"] is False
    assert current["matrix_learning_outcome"] == "DROPPED_ABSENT"
    assert current["matrix_learning_primary"] is False


def test_legacy_positive_evidence_is_not_hidden_or_migrated() -> None:
    legacy = contract.build_report()["legacy_learning"]
    assert legacy["teacher"]["parameter_count"] == 72_853
    assert legacy["student"]["state_dimension"] == 4
    assert legacy["student"]["stored_trainable_scalars"] == 95
    assert legacy["student"]["evaluation_mse"] == pytest.approx(6.083136156367311e-06)
    assert legacy["retention"]["minimum_point"] == pytest.approx(0.9814573586937879)
    assert legacy["retention"]["minimum_ci_lower"] == pytest.approx(0.9445014278749587)
    signature = contract.build_report()["task_signature"]
    assert signature["same_task"] is False
    assert signature["migration_allowed"] is False


def test_legacy_cnn_is_exactly_replayed_but_has_no_same_task_eligibility() -> None:
    cnn = contract.build_report()["legacy_learning"]["cnn"]
    assert cnn["samples"] == cnn["diagnostic_samples"] == 206
    assert cnn["repeat_count"] == 5
    assert cnn["bit_exact_across_repeats"] is True
    assert cnn["active_mse"] == pytest.approx(2.4144528544831194e-06)
    assert cnn["zero_residual_mse"] == pytest.approx(8.0340452043047e-06)
    assert cnn["candidate_families"] == 16
    assert cnn["same_task_eligible"] == 0
    assert cnn["claim_registry"]["LEGACY_CNN_PARAMETER_REPLAY"] == "DIAGNOSTIC_EXACT_INELIGIBLE"


def test_ood_and_hardware_limits_remain_visible() -> None:
    legacy = contract.build_report()["legacy_learning"]
    assert legacy["mismatch"]["minimum_retention"] == pytest.approx(0.8976304408841681)
    assert "universal robustness" in legacy["mismatch"]["claim_boundary"]["forbidden"]
    hardware = legacy["hardware"]
    assert hardware["quantized_gru_cycles"] == 72_854
    assert hardware["quantized_gru_functional_rtl"] is False
    assert hardware["quantized_gru_physical_gain_retention"] is None
    assert hardware["quantized_gru_eligible"] is False
    assert hardware["student_present_in_current_rtl"] is False


def test_future_promotion_gate_is_matched_and_not_imitation_only() -> None:
    gate = contract.build_report()["promotion_gate"]
    assert len(gate["required_fields"]) == 13
    for item in (
        "matched_classical_approximation_budget",
        "relative_ler_retention",
        "worst_family_retention",
        "held_out_ood_retention",
        "formal_retention_lower_bound",
        "compression_or_cost_benefit",
    ):
        assert item in gate["required_fields"]
    assert gate["failure_disposition"] == "DROPPED_TO_ABLATION"
    assert gate["can_change_classical_algorithm_verdict"] is False
    assert gate["can_change_rtl_verdict"] is False


def test_response_answers_the_question_without_universal_generalization() -> None:
    report = contract.build_report()
    text = report["response_package"]["english_response"]
    assert "revised manuscript is not CNN-centric" in text
    assert "no independent vote" in text
    assert "72,853-parameter" in text
    assert "do not claim that every future learned model will generalize" in text
    assert not any(phrase.lower() in text.lower() for phrase in report["forbidden_response_phrases"])


def test_rows_are_lossless_unique_and_cover_every_state() -> None:
    report = contract.build_report()
    rows = report["response_rows"]
    assert len(rows) == 24
    assert len({row["row_id"] for row in rows}) == 24
    assert {row["response_state"] for row in rows} == contract.RESPONSE_STATES
    assert all(row["source_ids"] and row["claim"] and row["boundary"] for row in rows)


def test_learning_vote_and_task_migration_mutations_fail_closed() -> None:
    report = contract.build_report()
    vote = copy.deepcopy(report)
    vote["promotion_gate"]["can_change_rtl_verdict"] = True
    assert not contract.evaluate_gates(vote)["G06_primary_verdict_is_learning_deletion_invariant"]
    migrated = copy.deepcopy(report)
    migrated["task_signature"]["same_task"] = True
    migrated["task_signature"]["migration_allowed"] = True
    assert not contract.evaluate_gates(migrated)["G12_task_signatures_are_explicitly_nonmatching"]


def test_manuscript_mapping_is_present_without_invented_line_numbers() -> None:
    report = contract.build_report()
    assert all(report["manuscript"]["sections"].values())
    assert all(report["manuscript"]["markers"].values())
    assert report["manuscript"]["cnn_title"] is False
    locations = report["response_package"]["tracker"]["manuscript_locations"]
    assert locations == [
        "Abstract",
        "Introduction",
        "Methods: Replaceable CNN/student extension",
        "Discussion",
        "Supplementary evidence delta",
    ]


def test_written_report_source_data_and_markdown_verify() -> None:
    report = contract.build_report()
    contract.write_outputs(report)
    with contract.DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == contract._source_rows(report)
    stored = json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert stored["source_data"]["rows"] == 24
    ok, checks = contract.verify_report()
    assert ok, checks
    assert all(checks.values())
