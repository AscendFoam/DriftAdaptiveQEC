from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import route_a_final_evidence_gate as gate


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_9_3_route_a_final_evidence_gate.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _claims(report: dict | None = None) -> dict[str, dict]:
    current = _report() if report is None else report
    return {row["claim_id"]: row for row in current["claims"]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_final_report_recomputes_and_is_explicit_no_go() -> None:
    report = _report()
    gate.verify_report(report)
    assert report["verdict"] == "NO_GO_FULL_HIGH_LEVEL_PAPER_RESTRICTED_PREBOARD_DRAFT_ONLY"
    assert report["gate_summary"] == {"passed": 17, "failed": 0}
    assert report["paper_decision"]["full_cross_lane_high_level_paper"] == "NO_GO"
    assert report["paper_decision"]["phase7_main_figure_and_prose_freeze_allowed"] is False
    assert report["paper_decision"]["selected_downgrade"] == "RESTRICTED_PREBOARD_SYSTEM_DRAFT"


def test_positive_outcomes_are_narrow_and_negative_boundaries_remain_visible() -> None:
    claims = _claims()
    smooth = claims["SMOOTH_LOCKED_EWMA_ADVANTAGE"]
    assert smooth["final_state"] == "SUPPORTED_PAIRED_OUTCOME"
    assert smooth["current_result"]["primary_contrast"]["ci95_low"] > 0.0
    assert smooth["current_result"]["holm_confirmed_families"] == ["periodic_drift"]
    assert smooth["current_result"]["route_a_is_global_best"] is False

    static = claims["STATIC_GKP_SUPERIORITY"]
    assert static["final_state"] == "FALSIFIED"
    assert static["current_result"]["static_minus_route_a"]["ci95_high"] < 0.0

    tail = claims["TAIL_SAFETY_AND_IMPROVEMENT"]
    assert tail["final_state"] == "SAFETY_NONINFERIORITY_ONLY"
    assert tail["current_result"]["confirmed_average_improvement_families"] == []
    assert tail["current_result"]["broad_tail_improvement_confirmed"] is False


def test_external_nmf_and_speed_claims_fail_closed() -> None:
    claims = _claims()
    external = claims["GENERAL_DRIFT_EXTERNAL_COMPARISON"]
    paired = external["current_result"]["paired_outcome"]
    assert external["final_state"] == "PERFORMANCE_OUTCOME_BUDGET_FAIL"
    assert paired["external_minus_route_a"]["ci95_low"] > 0.0
    assert paired["external_update_worst_us"] > paired["wallclock_cap_us"]

    nmf = claims["PUVIANI_NMF_SURPASS"]
    assert nmf["final_state"] == "PROHIBITED_SOURCE_INCOMPLETE"
    assert nmf["current_result"]["paper_exact_passed"] == 0
    assert nmf["current_result"]["matched_metric_non_null_count"] == 0

    speed = claims["FPGA_SPEED_ADVANTAGE"]
    assert speed["final_state"] == "PROHIBITED_NO_SAME_TASK_BOARD_COMPARATOR"
    assert speed["current_result"]["same_task_external_comparator_count"] == 0
    assert speed["current_result"]["board_measured"] is False


def test_preboard_hardware_and_k4_are_supported_only_with_exact_boundaries() -> None:
    claims = _claims()
    k4 = claims["STATIC_K4_HARD_ACTION_EQUIVALENCE"]
    assert k4["final_state"] == "SUPPORTED_PREBOARD_NARROW"
    assert k4["current_result"]["domain_points"] == 1_048_576
    assert k4["current_result"]["hard_action_disagreements"] == 0

    hardware = claims["FPGA_DETERMINISTIC_ARCHITECTURE"]
    assert hardware["final_state"] == "SUPPORTED_PR_ESTIMATE"
    assert hardware["current_result"]["fmax_mhz"]["minimum"] >= 27.0
    assert hardware["current_result"]["clock_model_ns"] == pytest.approx(222.22222222222223)

    board = claims["BOARD_MEASURED_CORRECTNESS_LATENCY"]
    assert board["final_state"] == "BLOCKED_ALL_FIELDS_NULL"
    assert board["current_result"]["null_measured_fields"] == 42
    assert board["current_result"]["physical_prerequisites_failed"] == 6


def test_all_parent_source_and_implementation_bindings_are_live_and_csv_is_lossless() -> None:
    report = _report()
    bindings = [*report["parent_bindings"].values(), report["source_data"], report["implementation_binding"]]
    for binding in bindings:
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]

    with (ROOT / report["source_data"]["path"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 11
    assert {row["claim_id"] for row in rows} == gate.FINAL_CLAIM_IDS


def test_semantic_mutations_and_attempted_paper_promotion_are_rejected() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 17
    assert len(audit["cases"]) == 17
    assert all(row["rejected"] for row in audit["cases"])
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])

    forged = deepcopy(report)
    forged["paper_decision"]["full_cross_lane_high_level_paper"] = "GO"
    forged["paper_decision"]["phase7_main_figure_and_prose_freeze_allowed"] = True
    with pytest.raises(ValueError, match="gates/verdict"):
        gate.verify_report(forged)

