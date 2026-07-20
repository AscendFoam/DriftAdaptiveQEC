from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from cnn_fpga.benchmark.qec_postselection_cost import (
    CONTRACT_ID,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    implementation_sha256,
    run_report,
    validate_artifact_payload,
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_report()


def test_formal_cost_report_is_complete_and_passes_all_gates(payload: dict) -> None:
    assert payload["task_id"] == "T5.3.4"
    assert payload["contract_id"] == CONTRACT_ID
    assert payload["status"] == "PASS"
    assert len(payload["gates"]) == 27
    assert all(payload["gates"].values())
    assert len(payload["parent_audits"]) == 6
    assert len(payload["online_qec_cost_rows"]) == 6
    assert len(payload["postselection_cost_rows"]) == 8
    assert len(payload["missing_cost_fields"]) == 12
    assert validate_artifact_payload(payload) == payload["gates"]


def test_online_qec_costs_use_native_event_accounting(payload: dict) -> None:
    for row in payload["online_qec_cost_rows"]:
        assert row["horizon_us"] == 300.0
        assert row["full_cycles"] == 30
        assert row["half_cycles"] == 60
        assert row["measurement_events"] == 60
        assert row["reset_events"] == 60
        assert row["active_pulse_or_gate_count"] == 540
        assert row["passive_measurement_events"] == 0
        assert row["passive_reset_events"] == 0
        assert row["passive_active_gate_applications"] == 0
        assert row["scaled_reference_matches_channel_events"] is True
        assert row["postselection_applied"] is False
        assert row["postselection_acceptance_fraction"] == 1.0
        assert row["code_space_survival_is_acceptance"] is False


def test_online_rows_keep_metric_and_unmeasured_cost_boundaries(payload: dict) -> None:
    for row in payload["online_qec_cost_rows"]:
        assert row["final_active_average_fidelity"] > 0.0
        assert row["final_active_code_space_survival"] > row["final_active_average_fidelity"]
        assert row["achieved_logical_error_rate"] is None
        assert row["stored_control_scalars"] == 15
        assert row["analytic_macs_per_half_cycle"] == 0
        assert row["matched_controller_classical_latency_us"] is None
        assert row["active_pulse_duration_us"] is None
        assert row["active_pulse_energy"] is None
        assert row["full_cost_complete"] is False
        assert row["target_hardware_measured"] is False
        assert row["equivalent_squeezing_db"] == pytest.approx(6.360121703, abs=1e-9)


def test_postselection_conditional_improvement_is_not_free(payload: dict) -> None:
    for row in payload["postselection_cost_rows"]:
        assert row["conditional_error_rate"] < row["raw_error_rate"]
        assert row["acceptance_fraction"] + row["rejection_fraction"] == pytest.approx(1.0)
        assert row["accepted_failures_per_input"] == pytest.approx(
            row["acceptance_fraction"] * row["conditional_error_rate"]
        )
        assert row["total_cost_by_rejection_penalty"]["1.00"] > row["raw_error_rate"]
        assert row["raw_minus_conditional_seed_cluster_ci"]["ci_low"] > 0.0
        assert row["truth_upper_deployable"] is False
        assert row["online_correction_eligible"] is False
        assert row["primary_metric_eligible"] is False
        assert row["conditional_metric_online_eligible"] is False
        assert row["achieved_average_fidelity"] is None
        assert row["achieved_logical_error_rate"] is None
        assert row["measurement_events"] is None
        assert row["classical_latency_us"] is None


def test_separate_safety_campaign_and_missing_fields_are_not_joined(payload: dict) -> None:
    safety = payload["software_safety_cost_row"]
    assert safety["campaign_cycles"] == 767872
    assert safety["fallback_cycles"] == 11552
    assert safety["reset_request_cycles"] == 4
    assert safety["statistical_population_upper_bound"] is None
    assert safety["joined_to_online_channel_cost"] is False
    assert safety["joined_to_postselection_cost"] is False
    assert all(row["value"] is None for row in payload["missing_cost_fields"])


def test_cost_verdict_refuses_full_cost_and_break_even_promotion(payload: dict) -> None:
    verdict = payload["verdict"]
    assert verdict["postselection_targets_with_lower_conditional_error"] == 8
    assert verdict["postselection_targets_worse_at_unit_rejection_penalty"] == 8
    assert verdict["full_cost_operational_boundary"] == "NOT_ESTABLISHED"
    assert verdict["paper_defined_coherence_gain"] == "NOT_ESTABLISHED"
    assert verdict["postselected_break_even"] == "NOT_ESTABLISHED"
    assert payload["cost_contract"]["global_cost_score"] is None
    assert payload["cost_contract"]["cross_lane_total"] is None
    assert payload["cost_contract"]["postselection_joined_to_qec"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "parent_hash",
        "active_event_count",
        "passive_event_count",
        "squeezing",
        "code_survival_as_acceptance",
        "invent_ler",
        "invent_latency",
        "invent_pulse_duration",
        "qualify_full_cost_row",
        "break_acceptance_identity",
        "qualify_conditional_online",
        "invent_postselection_favg",
        "hide_unit_penalty",
        "deploy_truth_upper",
        "join_safety_campaign",
        "fill_missing_cost",
        "invent_global_score",
        "promote_full_cost_verdict",
        "promote_postselected_break_even",
        "promote_experimental_claim",
    ],
)
def test_semantic_cost_mutations_fail_closed(payload: dict, mutation: str) -> None:
    tampered = copy.deepcopy(payload)
    online = tampered["online_qec_cost_rows"][0]
    post = tampered["postselection_cost_rows"][0]
    if mutation == "parent_hash":
        tampered["parent_audits"]["T5.3.1"]["sha256"] = "0" * 64
    elif mutation == "active_event_count":
        online["active_pulse_or_gate_count"] -= 1
    elif mutation == "passive_event_count":
        online["passive_measurement_events"] = 1
    elif mutation == "squeezing":
        online["equivalent_squeezing_db"] = 10.0
    elif mutation == "code_survival_as_acceptance":
        online["code_space_survival_is_acceptance"] = True
    elif mutation == "invent_ler":
        online["achieved_logical_error_rate"] = 0.01
    elif mutation == "invent_latency":
        online["matched_controller_classical_latency_us"] = 1.0
    elif mutation == "invent_pulse_duration":
        online["active_pulse_duration_us"] = 1.0
    elif mutation == "qualify_full_cost_row":
        online["full_cost_complete"] = True
        online["full_cost_operational_boundary_qualified"] = True
    elif mutation == "break_acceptance_identity":
        post["rejection_fraction"] = 0.0
    elif mutation == "qualify_conditional_online":
        post["online_correction_eligible"] = True
        post["conditional_metric_online_eligible"] = True
    elif mutation == "invent_postselection_favg":
        post["achieved_average_fidelity"] = 0.99
    elif mutation == "hide_unit_penalty":
        post["total_costs"][-1] = 0.0
        post["total_cost_by_rejection_penalty"]["1.00"] = 0.0
    elif mutation == "deploy_truth_upper":
        post["truth_upper_deployable"] = True
    elif mutation == "join_safety_campaign":
        tampered["software_safety_cost_row"]["joined_to_online_channel_cost"] = True
    elif mutation == "fill_missing_cost":
        tampered["missing_cost_fields"][0]["value"] = 1.0
    elif mutation == "invent_global_score":
        tampered["cost_contract"]["global_cost_score"] = 0.5
    elif mutation == "promote_full_cost_verdict":
        tampered["verdict"]["full_cost_operational_boundary"] = "ESTABLISHED"
    elif mutation == "promote_postselected_break_even":
        tampered["verdict"]["postselected_break_even"] = "ESTABLISHED"
    elif mutation == "promote_experimental_claim":
        tampered["claim_boundary"]["experimental_break_even"] = True
    with pytest.raises(ValueError, match="stored gates"):
        validate_artifact_payload(tampered)


def test_formal_artifact_and_94_row_source_data_match_live_code() -> None:
    payload = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["status"] == "PASS"
    assert validate_artifact_payload(payload) == payload["gates"]
    data = DEFAULT_SOURCE_DATA.read_bytes()
    assert hashlib.sha256(data).hexdigest() == payload["source_data"]["sha256"]
    with DEFAULT_SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"] == 94
    assert {
        "contract",
        "parent",
        "standard_cost_reference",
        "online_qec_cost",
        "postselection_summary",
        "postselection_penalty",
        "software_safety_cost",
        "missing_cost_field",
        "gate",
    } <= {row["category"] for row in rows}

