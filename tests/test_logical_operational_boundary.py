from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from cnn_fpga.benchmark.logical_operational_boundary import (
    CONTRACT_ID,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    OperationalBoundaryConfig,
    implementation_sha256,
    run_report,
    validate_artifact_payload,
)


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_report()


def test_formal_report_has_all_comparisons_and_gates(payload: dict) -> None:
    assert payload["task_id"] == "T5.3.3"
    assert payload["contract_id"] == CONTRACT_ID
    assert payload["status"] == "PASS"
    assert len(payload["gates"]) == 25
    assert all(payload["gates"].values())
    assert len(payload["comparisons"]) == 12
    assert len(payload["terminal_qualification"]) == 3
    assert validate_artifact_payload(payload) == payload["gates"]


def test_terminal_cutoffs_have_stable_full_curve_boundaries(payload: dict) -> None:
    expected = {
        "high": (40.0, 60.0),
        "medium": (40.0, 90.0),
        "low": (60.0, 110.0),
    }
    for noise, (pointwise, cumulative) in expected.items():
        for cutoff in (36, 40):
            comparison = payload["comparisons"][f"cutoff{cutoff}:{noise}"]
            assert comparison["wall_clock_boundary_qualified"] is True
            assert comparison["boundary"]["sustained_dominance_time_us"] == pointwise
            assert comparison["boundary"]["cumulative_breakeven_time_us"] == cumulative
            assert comparison["boundary"]["terminal_advantage"] > 0.0
            assert comparison["boundary"]["terminal_cumulative_advantage_us"] > 0.0
        row = next(
            item for item in payload["terminal_qualification"]
            if item["noise_profile"] == noise
        )
        assert row["sustained_boundary_spread_us"] == 0.0
        assert row["cumulative_breakeven_spread_us"] == 0.0
        assert row["terminal_repeat_stable"] is True


def test_transient_and_low_cutoff_counterevidence_are_preserved(payload: dict) -> None:
    for noise in ("high", "medium", "low"):
        terminal = payload["comparisons"][f"cutoff40:{noise}"]
        assert terminal["boundary"]["initial_penalty_min"] < -0.25
        assert terminal["boundary"]["sign_reversal_count"] >= 1
        assert terminal["active_short_time_rate_status"] == "unreliable_cycle_scale_transient"
        assert terminal["active_qualified_short_time_lifetime_us"] is None

        low = payload["comparisons"][f"cutoff12:{noise}"]
        assert low["wall_clock_boundary_qualified"] is False
        assert low["boundary"]["sustained_dominance_time_us"] is None
        assert low["boundary"]["cumulative_breakeven_time_us"] is None
        assert low["boundary"]["terminal_advantage"] < 0.0


def test_boundary_is_wall_clock_only_not_paper_coherence_gain(payload: dict) -> None:
    verdict = payload["verdict"]
    assert verdict["wall_clock_operational_boundary"] == (
        "ESTABLISHED_WITHIN_300US_FINITE_CUTOFF_MODEL"
    )
    assert verdict["full_cost_operational_boundary"] == "NOT_ESTABLISHED_PENDING_T5.3.4"
    assert verdict["simulation_derived_coherence_gain"] == "NOT_ESTABLISHED"
    assert verdict["coherence_gain_value"] is None
    assert verdict["experimental_break_even"] == "NOT_ESTABLISHED"
    for comparison in payload["comparisons"].values():
        audit = comparison["matching_audit"]
        assert audit["wall_clock_matched"] is True
        assert audit["full_cost_matched"] is False
        assert audit["baseline_role"].endswith("not_best_passive_physical_qubit")
        assert comparison["full_cost_boundary_qualified"] is False
        assert comparison["coherence_gain_qualified"] is False
        assert comparison["boundary"]["ratio_reported"] is False
        assert comparison["boundary"]["exponential_fit_used"] is False


def test_uncertainty_remains_deterministic_not_fake_ci(payload: dict) -> None:
    uncertainty = payload["uncertainty_contract"]
    assert uncertainty["statistical_standard_error"] is None
    assert uncertainty["statistical_confidence_interval"] is None
    assert uncertainty["terminal_repeat_is_ci"] is False
    assert uncertainty["subgrid_interpolation_is_validation"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "channel_hash",
        "fidelity_hash",
        "paper_hash",
        "baseline_role",
        "trim_curve",
        "move_sustained_boundary",
        "move_cumulative_boundary",
        "hide_initial_penalty",
        "qualify_cutoff12",
        "tamper_terminal_table",
        "qualify_active_rate",
        "fake_full_cost",
        "invent_ratio",
        "invent_exponential_fit",
        "invent_coherence_gain",
        "invent_experimental_break_even",
        "invent_statistical_ci",
    ],
)
def test_semantic_mutations_fail_closed(payload: dict, mutation: str) -> None:
    tampered = copy.deepcopy(payload)
    comparison = tampered["comparisons"]["cutoff40:high"]
    if mutation == "channel_hash":
        tampered["parent_audits"]["channel"]["sha256"] = "0" * 64
    elif mutation == "fidelity_hash":
        tampered["parent_audits"]["fidelity"]["sha256"] = "0" * 64
    elif mutation == "paper_hash":
        tampered["paper_audit"]["sha256"] = "0" * 64
    elif mutation == "baseline_role":
        comparison["matching_audit"]["baseline_role"] = "best_passive_physical_qubit"
    elif mutation == "trim_curve":
        comparison["boundary"]["time_us"].pop()
    elif mutation == "move_sustained_boundary":
        comparison["boundary"]["sustained_dominance_time_us"] = 20.0
    elif mutation == "move_cumulative_boundary":
        comparison["boundary"]["cumulative_breakeven_time_us"] = 20.0
    elif mutation == "hide_initial_penalty":
        comparison["boundary"]["initial_penalty_min"] = 0.0
    elif mutation == "qualify_cutoff12":
        tampered["comparisons"]["cutoff12:high"]["wall_clock_boundary_qualified"] = True
    elif mutation == "tamper_terminal_table":
        tampered["terminal_qualification"][0]["sustained_boundary_time_us_at_40"] = 20.0
    elif mutation == "qualify_active_rate":
        comparison["active_short_time_rate_status"] = "reliable_discrete_short_time_proxy"
    elif mutation == "fake_full_cost":
        comparison["matching_audit"]["full_cost_matched"] = True
        comparison["full_cost_boundary_qualified"] = True
    elif mutation == "invent_ratio":
        comparison["boundary"]["ratio_reported"] = True
    elif mutation == "invent_exponential_fit":
        comparison["boundary"]["exponential_fit_used"] = True
    elif mutation == "invent_coherence_gain":
        tampered["verdict"]["simulation_derived_coherence_gain"] = "ESTABLISHED"
        tampered["verdict"]["coherence_gain_value"] = 2.0
    elif mutation == "invent_experimental_break_even":
        tampered["claim_boundary"]["experimental_break_even"] = True
    elif mutation == "invent_statistical_ci":
        tampered["uncertainty_contract"]["statistical_confidence_interval"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="stored gates"):
        validate_artifact_payload(tampered)


def test_formal_artifact_and_source_data_match_live_implementation() -> None:
    payload = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["status"] == "PASS"
    assert validate_artifact_payload(payload) == payload["gates"]
    source_bytes = DEFAULT_SOURCE_DATA.read_bytes()
    assert hashlib.sha256(source_bytes).hexdigest() == payload["source_data"]["sha256"]
    with DEFAULT_SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"] == 416
    categories = {row["category"] for row in rows}
    assert {
        "contract",
        "parent",
        "paper",
        "full_curve",
        "boundary_summary",
        "terminal_qualification",
        "gate",
    } <= categories


def test_formal_config_rejects_scope_drift() -> None:
    with pytest.raises(ValueError, match="channel artifact"):
        OperationalBoundaryConfig(channel_artifact="other.json")
    with pytest.raises(ValueError, match="fidelity artifact"):
        OperationalBoundaryConfig(fidelity_artifact="other.json")
    with pytest.raises(ValueError, match="cutoff scan"):
        OperationalBoundaryConfig(formal_cutoffs=(24, 36, 40))
    with pytest.raises(ValueError, match="terminal cutoff"):
        OperationalBoundaryConfig(terminal_cutoffs=(24, 40))
    with pytest.raises(ValueError, match="one 10 us"):
        OperationalBoundaryConfig(maximum_terminal_boundary_spread_us=20.0)

