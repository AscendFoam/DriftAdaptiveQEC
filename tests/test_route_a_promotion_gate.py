from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import route_a_promotion_gate as gate


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(gate.DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def test_stored_report_is_current_and_fail_closed(report: dict) -> None:
    gate.verify_report(report)
    assert report["verdict"] == gate.VERDICT
    assert report["gate_summary"] == {"passed": 10, "failed": 0}


def test_raw_parent_analyses_recompute_without_retuning(report: dict) -> None:
    smooth = gate._recompute_smooth(gate._load(gate.SMOOTH_ARTIFACT))
    tail = gate._recompute_tail(gate._load(gate.TAIL_ARTIFACT))
    rtl = gate._recompute_rtl(gate._load(gate.RTL_ARTIFACT))
    assert smooth["raw_analysis_sha256"] == report["scientific_results"]["smooth"]["raw_analysis_sha256"]
    assert tail["raw_analysis_sha256"] == report["scientific_results"]["tail"]["raw_analysis_sha256"]
    assert rtl["trace_observed_sha256"] == report["scientific_results"]["rtl"]["trace_observed_sha256"]


def test_restricted_promotion_does_not_hide_smooth_counterevidence(report: dict) -> None:
    smooth = report["scientific_results"]["smooth"]
    assert smooth["primary_contrast"]["ci95_low"] > 0.0
    assert smooth["strongest_deployable"] == "window_map"
    assert not smooth["route_a_is_global_best_deployable"]
    assert not smooth["route_a_beats_static_average"]
    assert not smooth["route_a_beats_window_average"]
    assert smooth["holm_confirmed_families"] == ["periodic_drift"]


def test_tail_is_noninferiority_not_improvement(report: dict) -> None:
    result = report["scientific_results"]["tail"]
    assert result["tail_safety_gate_passes"]
    assert result["confirmed_average_improvement_families"] == []
    assert not result["broad_tail_improvement_confirmed"]
    assert set(result["exact_equal_average_families"]) == {
        "step_calibration_shift",
        "telegraph_drift",
        "readout_reset_fault",
        "leakage_persistence",
        "compound_ood",
    }


def test_claim_registry_has_explicit_negative_and_prohibited_states(report: dict) -> None:
    states = {row["claim_id"]: row["state"] for row in report["claim_registry"]}
    assert states["ROUTE_A_SYSTEM"] == "PROMOTED_RESTRICTED"
    assert states["GLOBAL_DEPLOYABLE_LER"] == "FALSIFIED"
    assert states["STATIC_GKP_SUPERIORITY"] == "FALSIFIED"
    assert states["TAIL_IMPROVEMENT"] == "NOT_ESTABLISHED"
    assert states["CNN_PRIMARY"] == "ABLATION_ONLY"
    assert states["HMM_ON_FPGA"] == "PROHIBITED"
    assert states["MEASURED_FPGA_SPEED"] == "PROHIBITED"


def test_semantic_mutations_are_all_detected(report: dict) -> None:
    audit = report["semantic_mutation_audit"]
    assert audit["detected"] == audit["count"] == 8
    assert all(row["rejected"] for row in audit["cases"])


def test_evaluator_rejects_scientific_and_claim_mutations(report: dict) -> None:
    primary = deepcopy(report)
    primary["scientific_results"]["smooth"]["primary_contrast"]["ci95_low"] = 0.0
    assert not gate.evaluate_gates(primary)["G04_locked_ewma_primary_lcb_strictly_positive"]

    rank = deepcopy(report)
    rank["promotion_decision"]["global_performance_rank"] = "promoted"
    assert not gate.evaluate_gates(rank)["G09_restricted_verdict_does_not_promote_falsified_claims"]
