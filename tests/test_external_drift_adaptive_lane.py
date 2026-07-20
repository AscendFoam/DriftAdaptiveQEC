from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cnn_fpga.benchmark import external_drift_adaptive_lane as lane


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_8_2_external_drift_adaptive_lane.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_current_state_verifier_accepts_integrity_pass_with_budget_failure() -> None:
    report = _report()
    lane.verify_report(report)
    assert report["evidence_integrity"]["passed"] is True
    assert report["gates"]["G08_external_budget_meets_common_caps"] is False
    assert report["verdict"].endswith("ROUTE_A_LOWER_LER_BUDGET_FAIL")


def test_external_and_parent_formal_traces_are_exactly_bound() -> None:
    report = _report()
    assert report["formal_results"]["trace_binding"] == {
        "external_trajectories": 504,
        "parent_trajectories": 504,
        "missing_parent_keys": 0,
        "input_sha256_mismatches": 0,
        "truth_sha256_mismatches": 0,
    }
    summaries = report["formal_results"]["method_summaries"]
    assert {row["decisions"] for row in summaries} == {24_772_608}


def test_external_code_changes_actions_and_is_not_an_ewma_noop() -> None:
    metrics = _report()["formal_results"]["external_detection_metrics"]
    assert metrics["detections"] > 0
    assert metrics["trajectories_differing_from_ewma"] == 102
    assert 0.0 < metrics["mean_window_bank_rate"] < 1.0


def test_online_bank_schedule_is_invariant_to_hidden_truth_labels() -> None:
    length = 4_000
    rng = np.random.default_rng(6802)
    truth = rng.integers(0, 4, size=length, dtype=np.uint8)
    window = rng.integers(0, 4, size=length, dtype=np.uint8)
    ewma = rng.integers(0, 4, size=length, dtype=np.uint8)
    common = {
        "seed": 1,
        "cell_id": "synthetic",
        "family": "synthetic",
        "scored_start_decision": 0,
        "observed_trace_sha256": "a" * 64,
        "truth_trace_sha256": "b" * 64,
    }
    base = {
        "truth": truth,
        "window_decisions": window,
        "ewma_decisions": ewma,
        "scores": [0.1],
        "boundaries": [2_000],
    }
    candidate = lane.BOCDCandidate(8, 1, 0.35)
    first = lane._run_candidate(
        {**base, "trajectory": SimpleNamespace(**common, labels=np.zeros(length // 32, dtype=np.uint8))},
        candidate,
        0.0,
        1.0,
        include_windows=False,
    )
    second = lane._run_candidate(
        {**base, "trajectory": SimpleNamespace(**common, labels=np.full(length // 32, 3, dtype=np.uint8))},
        candidate,
        0.0,
        1.0,
        include_windows=False,
    )
    for key in (
        "pauli_counts_I_Z_X_Y",
        "errors",
        "detection_count",
        "window_bank_rate",
        "error_trace_sha256",
    ):
        assert first[key] == second[key]
    assert [row["boundary_decision"] for row in first["detections"]] == [
        row["boundary_decision"] for row in second["detections"]
    ]


def test_pilot_selection_is_frozen_non_degenerate_and_disjoint() -> None:
    report = _report()
    selected = report["pilot_selection"]["selected"]
    assert selected["eligible"] is True
    assert selected["dynamic_detection_count"] == 1_478
    assert selected["nominal_window_bank_rate"] <= 0.10
    assert report["pilot_selection"]["selected_candidate_id"] == report["formal_results"]["candidate_id"]
    assert report["split_contract"]["all_rate_amplitude_duration_sets_disjoint"] is True


def test_budget_failure_is_one_observed_deadline_miss_not_hidden_by_p95() -> None:
    budget = _report()["formal_results"]["external_budget"]
    assert budget["update_count"] == 13_104
    assert budget["update_wallclock_p95_us"] < budget["common_update_wallclock_cap_us"]
    assert budget["update_wallclock_worst_us"] > budget["common_update_wallclock_cap_us"]
    assert budget["deadline_miss_count"] == len(budget["deadline_miss_witnesses"]) == 1


def test_claims_downgrade_external_outcome_after_budget_failure() -> None:
    claims = {row["claim_id"]: row["state"] for row in _report()["claim_registry"]}
    assert claims["EXTERNAL_BOCD_WRAPPER_PAIRED_OUTCOME"] == "ROUTE_A_LOWER_LER"
    assert claims["EXTERNAL_BOCD_MATCHED_BUDGET"] == "FAILED"
    assert claims["GENERAL_DRIFT_ADAPTIVE_SOTA"] == "PROHIBITED"
    assert claims["BHARDWAJ_EXACT_REPRODUCTION"] == "NOT_ESTABLISHED"


def test_each_semantic_mutation_targets_and_fails_its_own_gate() -> None:
    audit = _report()["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 10
    assert len({row["target_gate"] for row in audit["cases"]}) == 10
    assert all(row["rejected"] for row in audit["cases"])


def test_budget_claim_tampering_fails_claim_gate() -> None:
    report = deepcopy(_report())
    next(
        row
        for row in report["claim_registry"]
        if row["claim_id"] == "EXTERNAL_BOCD_MATCHED_BUDGET"
    )["state"] = "PASSED"
    assert lane.evaluate_gates(report)["G11_claim_scope_matches_results_without_general_sota"] is False

