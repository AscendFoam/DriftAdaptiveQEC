from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from cnn_fpga.benchmark.logical_channel_fidelity import (
    CONTRACT_ID,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    FidelityReportConfig,
    implementation_sha256,
    run_report,
    validate_artifact_payload,
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_report()


def test_formal_report_is_complete_and_all_semantic_gates_pass(payload: dict) -> None:
    assert payload["task_id"] == "T5.3.2"
    assert payload["contract_id"] == CONTRACT_ID
    assert payload["status"] == "PASS"
    assert len(payload["gates"]) == 23
    assert all(payload["gates"].values())
    assert len(payload["lanes"]) == 24
    assert len(payload["terminal_cutoff_intervals"]) == 30
    assert len(payload["matched_on_off_differences"]) == 3
    assert validate_artifact_payload(payload) == payload["gates"]


def test_tp_formula_overstatement_is_explicit_after_leakage(payload: dict) -> None:
    positive_gaps = []
    for lane in payload["lanes"].values():
        for point in lane["cycle_metrics"]:
            expected = (1.0 - point["mean_code_survival"]) / 3.0
            assert point["tp_formula_overstatement"] == pytest.approx(expected, abs=2.0e-9)
            assert point["direct_six_state_average_fidelity"] == pytest.approx(
                point["average_fidelity"], abs=2.0e-9
            )
            assert point["average_fidelity"] <= point["mean_code_survival"] + 2.0e-9
            positive_gaps.append(point["tp_formula_overstatement"])
    assert max(positive_gaps) > 0.2


def test_short_time_rate_preserves_active_transient_failure(payload: dict) -> None:
    for lane in payload["lanes"].values():
        rate = lane["short_time_effective_depolarization"]
        assert rate["exponential_fit_used"] is False
        assert rate["primary_rate_per_cycle"] == pytest.approx(
            10.0 * rate["primary_rate_per_us"], abs=2.0e-12
        )
        if lane["parent_lane_config"]["mode"] == "qec_off":
            assert rate["reliability_status"] == "reliable_discrete_short_time_proxy"
            assert rate["primary_lifetime_us"] is not None
        else:
            assert rate["reliability_status"] == "unreliable_cycle_scale_transient"
            assert (
                rate["first_three_monotone_nonincreasing"] is False
                or rate["relative_discretization_spread"] > 0.5
            )
            assert rate["primary_lifetime_us"] is None
            assert rate["algebraic_inverse_rate_us"] is not None
            assert rate["relative_discretization_spread"] > 0.5


def test_uncertainty_is_systematic_not_fake_statistical_ci(payload: dict) -> None:
    contract = payload["uncertainty_contract"]
    assert contract["statistical_standard_error"] is None
    assert contract["statistical_confidence_interval"] is None
    assert contract["cutoff_interval_is_ci"] is False
    assert contract["discretization_envelope_is_ci"] is False
    for row in payload["terminal_cutoff_intervals"]:
        assert row["lower_cutoff"] == 36
        assert row["higher_cutoff"] == 40
        assert row["is_confidence_interval"] is False
        assert row["statistical_confidence_level"] is None
        assert row["numerical_interval_min"] <= row["value_at_higher_cutoff"] <= row["numerical_interval_max"]


def test_low_cutoff_direction_reversal_and_no_gain_claim_are_retained(payload: dict) -> None:
    assert all(row["direction_reversal_preserved"] for row in payload["cutoff_direction_audit"])
    for row in payload["matched_on_off_differences"]:
        assert row["qec_on_minus_off_final_average_fidelity"] > 0.0
        assert row["qec_on_minus_off_short_time_rate_per_us"] > 0.0
        assert row["ratio_or_gain_reported"] is False
        assert row["operational_boundary_claim"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "parent_hash",
        "paper_hash",
        "tp_formula",
        "conditional_role",
        "fake_ci",
        "rate_fit",
        "qualify_active_lifetime",
        "drop_reversal",
        "invent_gain",
        "tamper_terminal",
    ],
)
def test_semantic_mutations_cannot_keep_stored_pass(payload: dict, mutation: str) -> None:
    bad = copy.deepcopy(payload)
    lane = next(iter(bad["lanes"].values()))
    if mutation == "parent_hash":
        bad["parent_audit"]["implementation_hash_matches"] = False
    elif mutation == "paper_hash":
        bad["paper_audit"]["sha256"] = "0" * 64
    elif mutation == "tp_formula":
        point = lane["cycle_metrics"][-1]
        point["average_fidelity"] = point["tp_assuming_average_fidelity"]
    elif mutation == "conditional_role":
        lane["cycle_metrics"][-1]["conditional_metric_role"] = "channel_fidelity"
    elif mutation == "fake_ci":
        bad["uncertainty_contract"]["statistical_confidence_interval"] = [0.1, 0.2]
    elif mutation == "rate_fit":
        lane["short_time_effective_depolarization"]["exponential_fit_used"] = True
    elif mutation == "qualify_active_lifetime":
        active = next(
            value
            for value in bad["lanes"].values()
            if value["parent_lane_config"]["mode"] == "qec_on"
        )
        active["short_time_effective_depolarization"]["primary_lifetime_us"] = 10.0
    elif mutation == "drop_reversal":
        bad["cutoff_direction_audit"][0]["direction_reversal_preserved"] = False
    elif mutation == "invent_gain":
        bad["matched_on_off_differences"][0]["ratio_or_gain_reported"] = True
    elif mutation == "tamper_terminal":
        bad["terminal_cutoff_intervals"][0]["absolute_spread"] += 0.1
    with pytest.raises(ValueError, match="stored gates"):
        validate_artifact_payload(bad)


def test_status_and_gate_values_are_recomputed(payload: dict) -> None:
    bad = copy.deepcopy(payload)
    bad["status"] = "FAIL"
    with pytest.raises(ValueError, match="status"):
        validate_artifact_payload(bad)
    bad = copy.deepcopy(payload)
    first = next(iter(bad["gates"]))
    bad["gates"][first] = False
    with pytest.raises(ValueError, match="stored gates"):
        validate_artifact_payload(bad)


def test_formal_artifact_source_hash_and_live_implementation() -> None:
    payload = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert validate_artifact_payload(payload) == payload["gates"]
    source = payload["source_data"]
    assert source["path"] == DEFAULT_SOURCE_DATA.as_posix()
    assert hashlib.sha256(DEFAULT_SOURCE_DATA.read_bytes()).hexdigest() == source["sha256"]
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == source["row_count"] == 5294
    assert {
        "contract",
        "parent",
        "paper",
        "channel_fidelity",
        "state_fidelity",
        "short_time_rate",
        "terminal_cutoff_interval",
        "matched_on_off_difference",
        "cutoff_direction_audit",
        "gate",
    } <= {row["category"] for row in rows}


def test_config_is_frozen_to_parent_cutoff_and_rate_contract() -> None:
    with pytest.raises(ValueError, match="T5.3.1"):
        FidelityReportConfig(parent_artifact="other.json")
    with pytest.raises(ValueError, match="36 and 40"):
        FidelityReportConfig(terminal_cutoffs=(32, 40))
    with pytest.raises(ValueError, match="estimator"):
        FidelityReportConfig(primary_rate_estimator="exponential_fit")
