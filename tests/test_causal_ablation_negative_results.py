from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import causal_ablation_negative_results as ablation


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/t5_4_3_causal_ablation_negative_results.json"
SOURCE = ROOT / "docs/t5_4_3_causal_ablation_negative_results_source_data.csv"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rehash(mutated: dict) -> dict:
    mutated["contract_sha256"] = ablation._canonical_sha256(
        ablation._contract_view(mutated)
    )
    return mutated


def test_formal_artifact_is_semantically_valid(report: dict) -> None:
    assert report["status"] == "PASS"
    assert report["gate_summary"] == {"passed": 18, "total": 18}
    assert ablation.validate_artifact(report) == ()


def test_all_six_mechanisms_are_native_lane_interventions(report: dict) -> None:
    assert set(report["lanes"]) == set(ablation.MECHANISMS)
    assert [row["mechanism"] for row in report["negative_result_table"]] == list(
        ablation.MECHANISMS
    )
    assert report["causal_contract"]["deployment_hidden_truth_inputs"] == []
    assert report["nonmixing_contract"]["cross_lane_aggregate"] is None
    assert report["nonmixing_contract"]["global_ranking"] is None


def test_all_parent_and_asset_bindings_are_live(report: dict) -> None:
    assert all(row["machine_pass"] for row in report["parent_bindings"])
    bindings = (
        report["parent_bindings"]
        + report["parent_source_bindings"]
        + report["implementation_bindings"]
        + report["cnn_asset_bindings"]
    )
    for row in bindings:
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert row["sha256"] == ablation._sha256(path)


def test_history_reversal_forces_claim_downgrade(report: dict) -> None:
    lane = report["lanes"]["history"]
    assert lane["intervention_changes_actions"] is True
    assert lane["splits"]["primary"]["benefit_interval"]["ci95_low"] > 0.0
    assert lane["splits"]["confirmation"]["benefit_interval"]["ci95_high"] < 0.0
    assert lane["result"] == "CROSS_CUTOFF_REVERSAL_NOT_SUPPORTED"
    assert lane["claim_decision"] == "DOWNGRADE_MEMORY_MECHANISM_NOT_SUPPORTED"


def test_cnn_off_is_exact_zero_and_scope_stays_parameter_only(report: dict) -> None:
    lane = report["lanes"]["cnn_residual"]
    assert len(lane["samples"]) == 206
    assert all(row["off_predicted_delta_b"] == [0.0, 0.0] for row in lane["samples"])
    assert lane["aggregate"]["active_mse"] == pytest.approx(
        lane["aggregate"]["preserved_evaluation_report_mse"], abs=1.0e-18
    )
    assert lane["aggregate"]["benefit_off_minus_active_mse"] > 0.0
    assert all(row["benefit_off_minus_active_mse"] > 0.0 for row in lane["scenario_rows"])
    assert "SINGLE_LEGACY_TEST_SPLIT" in lane["uncertainty_status"]
    assert "not LER or control gain" in lane["scope"]


def test_cnn_sample_mse_accounting_recomputes(report: dict) -> None:
    lane = report["lanes"]["cnn_residual"]
    active = np.mean([row["active_squared_error"] for row in lane["samples"]])
    off = np.mean([row["off_squared_error"] for row in lane["samples"]])
    assert active == pytest.approx(lane["aggregate"]["active_mse"], abs=1.0e-18)
    assert off == pytest.approx(lane["aggregate"]["off_mse"], abs=1.0e-18)
    assert off - active == pytest.approx(
        lane["aggregate"]["benefit_off_minus_active_mse"], abs=1.0e-18
    )


def test_regime_gain_keeps_detection_delay_cost(report: dict) -> None:
    lane = report["lanes"]["regime_state"]
    assert len(lane["seed_rows"]) == 8
    assert lane["benefit_interval"]["ci_low"] > 0.0
    assert lane["delay_cost_interval"]["ci_low"] > 0.0
    assert lane["claim_decision"] == "RETAIN_ESTIMATOR_PROPER_SCORE_ONLY"


def test_run_length_negative_result_is_not_hidden(report: dict) -> None:
    lane = report["lanes"]["run_length"]
    assert len(lane["cells"]) == 32
    assert lane["benefit_interval"]["ci_high"] < 0.0
    assert all(row["run_length_benefit_off_minus_active"] < 0.0 for row in lane["cells"])
    assert lane["claim_decision"] == "DOWNGRADE_RUN_LENGTH_PERFORMANCE_ACTIVE_IS_WORSE"


def test_parameter_update_is_a_bounded_component_result(report: dict) -> None:
    lane = report["lanes"]["parameter_update"]
    assert lane["benefit_interval"]["ci_low"] > 0.0
    assert all(row["run_length_bank_writes"] > 0 for row in lane["cells"])
    assert lane["cells"] == report["lanes"]["run_length"]["cells"]
    assert lane["claim_decision"] == "RETAIN_COMPONENT_EVENT_COST_ONLY"
    assert "not decoder or physical-memory gain" in lane["scope"]


def test_fallback_aggregate_and_harmful_lanes_are_both_retained(report: dict) -> None:
    lane = report["lanes"]["fallback"]
    assert lane["benefit_interval"]["ci_low"] > 0.0
    harmful = [
        row
        for row in lane["scenario_rows"]
        if row["benefit_interval"]["ci_high"] < 0.0
    ]
    assert {row["scenario_id"] for row in harmful} == {
        "compound_range_extrapolation"
    }
    assert lane["nominal_benefit_interval"]["estimate"] < 0.0
    accounting = lane["sample_accounting"]
    assert accounting["avoided_failure_count"] > accounting["induced_failure_count"] > 0
    assert accounting["unnecessary_fallback_count"] > 0


def test_source_csv_is_complete_and_byte_bound(report: dict) -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["source_data"]["row_count"] == 338
    assert report["source_data"]["csv_sha256"] == ablation._sha256(SOURCE)
    assert {row["row_type"] for row in rows} >= {
        "history_agent",
        "cnn_sample",
        "regime_seed",
        "event_cell",
        "fallback_seed",
        "claim_decision",
        "gate",
    }


@pytest.mark.parametrize(
    "mutation",
    (
        "hide_history_reversal",
        "make_cnn_off_nonzero",
        "hide_regime_delay",
        "hide_run_length_harm",
        "hide_fallback_harm",
    ),
)
def test_semantic_validator_rejects_hidden_negative_evidence(
    report: dict, mutation: str
) -> None:
    changed = copy.deepcopy(report)
    if mutation == "hide_history_reversal":
        changed["lanes"]["history"]["splits"]["confirmation"]["benefit_interval"][
            "ci95_high"
        ] = 0.1
    elif mutation == "make_cnn_off_nonzero":
        changed["lanes"]["cnn_residual"]["samples"][0]["off_predicted_delta_b"][0] = 1.0e-6
    elif mutation == "hide_regime_delay":
        changed["lanes"]["regime_state"]["delay_cost_interval"]["ci_low"] = -0.1
    elif mutation == "hide_run_length_harm":
        changed["lanes"]["run_length"]["benefit_interval"]["ci_high"] = 0.1
    elif mutation == "hide_fallback_harm":
        changed["lanes"]["fallback"]["scenario_rows"][-1]["benefit_interval"][
            "ci_high"
        ] = 0.1
    errors = ablation.validate_artifact(_rehash(changed))
    assert errors
    assert any("lane evidence" in error or "evidence gates" in error for error in errors)


def test_semantic_validator_rejects_cross_lane_score(report: dict) -> None:
    changed = copy.deepcopy(report)
    changed["nonmixing_contract"]["cross_lane_aggregate"] = 0.75
    changed["nonmixing_contract"]["global_ranking"] = list(ablation.MECHANISMS)
    errors = ablation.validate_artifact(_rehash(changed))
    assert "one or more evidence gates failed" in errors


def test_semantic_validator_rejects_device_claim(report: dict) -> None:
    changed = copy.deepcopy(report)
    changed["claim_boundary"]["device_calibrated"] = True
    changed["claim_boundary"]["hardware_measured"] = True
    errors = ablation.validate_artifact(_rehash(changed))
    assert "one or more evidence gates failed" in errors
