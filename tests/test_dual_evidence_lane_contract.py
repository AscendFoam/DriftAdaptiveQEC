from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import dual_evidence_lane_contract as contract


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))


def _lane(report: dict, lane_id: str) -> dict:
    return next(row for row in report["lanes"] if row["lane_id"] == lane_id)


def _claim(report: dict, claim_id: str) -> dict:
    return next(row for row in report["claims"] if row["claim_id"] == claim_id)


def test_repository_contract_verifies_end_to_end() -> None:
    assert contract.verify_report() == {
        "identity": True,
        "gates": True,
        "verdict": True,
        "analysis_hash": True,
    }
    report = _report()
    assert report["gate_summary"] == {"passed": 16, "failed": []}
    assert report["verdict"] == contract.VERDICT


def test_exactly_two_primary_lanes_and_one_dependent_learning_extension() -> None:
    report = _report()
    primary = {row["lane_id"] for row in report["lanes"] if row["role"] == "PRIMARY_EVIDENCE_LANE"}
    assert primary == contract.PRIMARY_LANE_IDS
    learned = _lane(report, contract.EXTENSION_LANE_ID)
    assert learned["role"] == "OPTIONAL_DEPENDENT_EXTENSION"
    assert learned["depends_on_lane"] == "MULTIMODE_SOFTWARE_ALGORITHM"
    assert learned["primary_metrics"] == []
    assert learned["primary_gate_ids"] == []
    assert learned["failure_disposition"] == "DROPPED_TO_ABLATION"


def test_all_task_signatures_are_complete_distinct_and_nonempty() -> None:
    report = _report()
    signatures = []
    for lane in report["lanes"]:
        assert tuple(lane["task_signature"]) == contract.TASK_SIGNATURE_FIELDS
        assert all(isinstance(value, str) and value for value in lane["task_signature"].values())
        signatures.append(json.dumps(lane["task_signature"], sort_keys=True))
    assert len(set(signatures)) == 3


def test_multimode_ler_and_single_mode_rtl_timing_cannot_be_conflated() -> None:
    report = _report()
    multimode = _lane(report, "MULTIMODE_SOFTWARE_ALGORITHM")
    rtl = _lane(report, "SINGLE_MODE_DETERMINISTIC_RTL")
    assert multimode["primary_metrics"] == ["per_round_p_L"]
    assert multimode["deployment_status"] == "SOFTWARE_ONLY_NOT_CURRENT_RTL"
    assert all("cycle" not in boundary.lower() for boundary in multimode["timing_boundaries"])
    assert rtl["frozen_fast_path"]["latency_cycles"] == 6
    assert rtl["frozen_fast_path"]["initiation_interval_cycles"] == 1
    assert rtl["deployment_status"] == "ACTUAL_SINGLE_MODE_RTL_PREBOARD"
    assert multimode["task_signature"] != rtl["task_signature"]


@pytest.mark.parametrize(
    ("mutation", "gate"),
    [
        ("multimode_rtl_timing", "G04_multimode_is_software_LER_only"),
        ("rtl_multimode_family", "G05_rtl_is_single_mode_six_cycle_ii1"),
        ("learning_primary_gate", "G06_learning_is_dependent_and_nonprimary"),
    ],
)
def test_direct_cross_lane_promotions_fail_closed(mutation: str, gate: str) -> None:
    report = deepcopy(_report())
    if mutation == "multimode_rtl_timing":
        _lane(report, "MULTIMODE_SOFTWARE_ALGORITHM")["timing_boundaries"].append("rtl_latency_cycles")
    elif mutation == "rtl_multimode_family":
        _lane(report, "SINGLE_MODE_DETERMINISTIC_RTL")["code_family"] = "multimode_surface_gkp_rtl"
    else:
        learned = _lane(report, contract.EXTENSION_LANE_ID)
        learned["primary_metrics"] = ["per_round_p_L"]
        learned["primary_gate_ids"] = ["T6.24.5"]
    assert contract.evaluate_gates(report, check_live_files=False)[gate] is False
    with pytest.raises(ValueError, match="verification failed"):
        contract.verify_report(report)


def test_integration_bridge_is_schema_reuse_not_multimode_rtl_deployment() -> None:
    report = _report()
    bridge = next(
        row for row in report["interfaces"]
        if row["interface_id"] == "IF-SLOW-PROPOSAL-TO-CANDIDATE-IMAGE"
    )
    assert bridge["deployment_implication"] == "CONTRACT_REUSE_ONLY_REQUIRES_SCHEMA_EQUIVALENCE"
    tampered = deepcopy(report)
    next(
        row for row in tampered["interfaces"]
        if row["interface_id"] == "IF-SLOW-PROPOSAL-TO-CANDIDATE-IMAGE"
    )["deployment_implication"] = "MULTIMODE_CURRENT_RTL"
    assert contract.evaluate_gates(tampered, check_live_files=False)[
        "G07_integration_bridge_does_not_imply_multimode_deployment"
    ] is False


def test_required_forbidden_transfers_cover_all_requested_and_derived_failure_modes() -> None:
    report = _report()
    transfers = {row["transfer_id"]: row for row in report["forbidden_transfers"]}
    assert set(transfers) == contract.FORBIDDEN_TRANSFER_IDS
    assert len(transfers) == 9
    assert transfers["FT-MM-LER-TO-CURRENT-RTL"]["rejection_code"] == "CROSS_LANE_IMPLEMENTATION_PROMOTION"
    assert transfers["FT-RTL-LATENCY-TO-MULTIMODE"]["rejection_code"] == "CROSS_LANE_TIMING_PROMOTION"
    assert transfers["FT-CNN-TO-ALGORITHM-SOTA"]["rejection_code"] == "SURROGATE_TO_PRIMARY_PROMOTION"
    assert transfers["FT-CNN-TO-RTL-SAFETY"]["rejection_code"] == "SURROGATE_TO_RTL_PROMOTION"
    assert transfers["FT-CROSS-LANE-WEIGHTED-SCORE"]["rejection_code"] == "GLOBAL_SCORE_PROHIBITED"


def test_claim_promotion_states_keep_multimode_conditional_and_board_null() -> None:
    report = _report()
    multimode = _claim(report, "C-MM-FROZEN-BENCHMARK-LER-SOTA")
    historical_rtl = _claim(report, "C-RTL-HISTORICAL-PREBOARD-IMPLEMENTATION")
    phase6d_rtl = _claim(report, "C-RTL-DETERMINISTIC-ATOMIC-FAIL-CLOSED")
    board = _claim(report, "C-BOARD-MEASURED-PERFORMANCE")
    assert multimode["state"] == "CONDITIONAL_FUTURE"
    assert multimode["required_gate"] == "T6.24.5"
    assert historical_rtl["state"] == "CURRENT_RESTRICTED"
    assert "RTL_PROPERTY_PROOF" not in historical_rtl["required_layers"]
    assert phase6d_rtl["state"] == "CONDITIONAL_FUTURE"
    assert phase6d_rtl["required_gate"] == "T6.25.4"
    assert board["state"] == "BLOCKED_NULL"
    assert board["current_value"] is None
    assert board["required_layers"] == ["BOARD_MEASURED"]
    assert "BOARD_MEASURED" not in report["current_evidence_layers"]


def test_board_value_or_opened_formal_promotion_is_rejected() -> None:
    report = deepcopy(_report())
    board = _claim(report, "C-BOARD-MEASURED-PERFORMANCE")
    board.update(state="CURRENT_RESTRICTED", current_value={"latency_ns": 12.0})
    assert contract.evaluate_gates(report, check_live_files=False)[
        "G10_board_measurement_remains_blocked_null"
    ] is False
    assert "FT-OPENED-DEVELOPMENT-TO-FORMAL" in {
        row["transfer_id"] for row in report["forbidden_transfers"]
    }


def test_source_data_is_lossless_and_every_payload_hash_recomputes() -> None:
    report = _report()
    with contract.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == contract._source_rows(report)
    assert len(rows) == report["source_data"]["rows"] == 31
    for row in rows:
        assert row["canonical_sha256"] == hashlib.sha256(
            row["payload_json"].encode("utf-8")
        ).hexdigest()
        assert json.loads(row["payload_json"])


def test_every_bound_parent_artifact_is_live_and_nonempty() -> None:
    report = _report()
    assert set(report["artifact_registry"]) == set(contract.ARTIFACT_PATHS)
    assert "unified_execution_source" in report["artifact_registry"]
    assert "unified_execution_report" not in report["artifact_registry"]
    for binding in report["artifact_registry"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert path.stat().st_size == binding["bytes"] > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]


def test_one_independent_semantic_mutation_targets_every_gate() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 16
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])


def test_human_contract_contains_every_atomic_identifier() -> None:
    report = _report()
    text = contract.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    ids = [row["lane_id"] for row in report["lanes"]]
    ids += [row["claim_id"] for row in report["claims"]]
    ids += [row["interface_id"] for row in report["interfaces"]]
    ids += [row["transfer_id"] for row in report["forbidden_transfers"]]
    assert all(f"`{value}`" in text for value in ids)
    assert "不互相补门" in text
