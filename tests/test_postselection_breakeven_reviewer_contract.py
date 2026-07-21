from __future__ import annotations

import copy
import csv
import json

from cnn_fpga.benchmark import postselection_breakeven_reviewer_contract as contract


def test_report_passes_all_gates_and_mutations() -> None:
    report = contract.build_report()
    assert report["verdict"] == contract.VERDICT
    assert report["gate_summary"] == {"passed": 24, "total": 24}
    assert all(report["gates"].values())
    assert report["semantic_mutation_audit"]["detected"] == report["semantic_mutation_audit"]["count"] == 24


def test_package_is_preemptive_and_keeps_placeholder() -> None:
    report = contract.build_report()
    assert report["reviewer_context"]["comment_id"] == "PRQ-BE-1"
    assert report["response_package"]["package_readiness"] == "draft_with_placeholders"
    assert report["response_package"]["missing_information"] == ["ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING"]


def test_phase6d_primary_uses_complete_denominator_and_keeps_no_go() -> None:
    primary = contract.build_report()["current_phase6d_primary"]
    assert primary["families"] == 13
    assert primary["physical_rounds"] == 79_872
    assert primary["postselection_used"] is False
    assert primary["hard_ood_rows_retained"]
    assert primary["relative_improvement"] == 0.0
    assert primary["verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM"


def test_historical_logical_channel_is_unconditional() -> None:
    channel = contract.build_report()["historical_logical_channel"]
    assert channel["lanes"] == 24
    assert channel["source_rows"] == 17_266
    assert channel["postselected_trajectories"] == 0
    assert channel["discarded_trajectories"] == 0
    assert channel["conditional_postselected_forbidden"]


def test_offline_postselection_reports_scale_acceptance_and_role() -> None:
    post = contract.build_report()["offline_postselection"]
    assert post["online_decoder"] is False
    assert post["primary_metric_eligible"] is False
    assert post["training_samples"] == 294_912
    assert post["evaluation_samples"] == 1_572_864
    assert post["target90"]["realized_survival_fraction"] == 0.8991082509358723


def test_unit_rejection_penalty_reverses_conditional_improvement() -> None:
    cost = contract.build_report()["rejection_cost"]
    target = cost["target90"]
    assert target["raw_error_rate"] == 0.013785044352213543
    assert target["conditional_error_rate"] == 0.0012424775303850107
    assert target["total_cost_by_rejection_penalty"]["1.00"] == 0.10200887086329927
    assert target["total_cost_by_rejection_penalty"]["1.00"] > target["raw_error_rate"]
    assert cost["targets"] == cost["targets_lower_conditional"] == cost["targets_worse_unit_penalty"] == 8


def test_cost_ledgers_do_not_mix_and_nulls_remain_null() -> None:
    cost = contract.build_report()["rejection_cost"]
    assert cost["postselection_joined_to_qec"] is False
    assert cost["global_cost_score"] is None
    assert cost["cross_lane_total"] is None
    assert len(cost["missing_fields"]) == 12
    assert all(row["value"] is None for row in cost["missing_fields"])


def test_break_even_taxonomy_keeps_only_finite_model_boundary() -> None:
    taxonomy = contract.build_report()["break_even_taxonomy"]
    assert taxonomy["wall_clock_operational_boundary"] == "ESTABLISHED_WITHIN_300US_FINITE_CUTOFF_MODEL"
    assert taxonomy["low_cutoff_counterexample_retained"]
    assert taxonomy["fit"] is taxonomy["ratio"] is None
    assert taxonomy["full_cost_operational_boundary"] == "NOT_ESTABLISHED"
    assert taxonomy["simulation_derived_coherence_gain"] == "NOT_ESTABLISHED"
    assert taxonomy["coherence_gain_value"] is None
    assert taxonomy["postselected_break_even"] == taxonomy["experimental_break_even"] == "NOT_ESTABLISHED"


def test_sivak_is_literature_only_and_nontransferable() -> None:
    report = contract.build_report()
    assert report["break_even_taxonomy"]["sivak_literature"] == {
        "value": 2.27,
        "uncertainty": 0.07,
        "evidence_grade": "LITERATURE_ONLY",
        "ranking_eligible": False,
        "same_task": False,
    }
    assert set(report["nontransfer_contract"].values()) == {False}
    mutated = copy.deepcopy(report)
    mutated["nontransfer_contract"]["sivak_to_project_break_even"] = True
    assert not contract.evaluate_gates(mutated)["G18_all_metric_and_evidence_transfers_forbidden"]


def test_response_is_numeric_direct_and_has_no_forbidden_claim() -> None:
    report = contract.build_report()
    text = report["response_package"]["english_response"]
    for token in ("79,872", "0.102009", "All eight targets", "NOT_ESTABLISHED", "2.27±0.07"):
        assert token in text
    assert not any(phrase.lower() in text.lower() for phrase in report["forbidden_response_phrases"])


def test_rows_are_unique_and_cover_all_states() -> None:
    rows = contract.build_report()["response_rows"]
    assert len(rows) == len({row["row_id"] for row in rows}) == 24
    assert {row["response_state"] for row in rows} == contract.RESPONSE_STATES
    assert all(row["source_ids"] and row["claim"] and row["boundary"] for row in rows)


def test_written_artifacts_verify_losslessly() -> None:
    report = contract.build_report()
    contract.write_outputs(report)
    with contract.DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == contract._source_rows(report)
    stored = json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert stored["source_data"]["rows"] == 24
    ok, checks = contract.verify_report()
    assert ok, checks
    assert all(checks.values())
