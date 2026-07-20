from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import route_a_board_measurement_gate as gate


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_9_2_route_a_board_measurement_blocker.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_blocker_report_recomputes_and_passes_integrity_gates() -> None:
    report = _report()
    gate.verify_report(report)
    assert report["verdict"] == "BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL"
    assert report["gate_summary"] == {"passed": 11, "failed": 0}
    assert report["execution_branch"] == "BLOCKED_NO_PHYSICAL_BOARD_BITSTREAM_OR_TRANSPORT"


def test_physical_prerequisites_are_explicitly_absent_not_silently_skipped() -> None:
    report = _report()
    external = [row for row in report["prerequisite_ledger"] if row["kind"] == "physical_external"]
    assert len(external) == 6
    assert all(row["passed"] is False and row["observed_path"] is None for row in external)
    assert {row["prerequisite"] for row in external} == set(gate.EXPECTED_EXTERNAL_ARTIFACTS)
    assert report["board_statuses"]["T6.1.1"] == "Blocked"
    assert report["board_statuses"]["T6.9.2"] == "Blocked"


def test_all_board_measurements_remain_null_and_pr_estimates_are_not_copied() -> None:
    report = _report()
    assert set(report["measured_results"]) == set(gate.MEASURED_FIELDS)
    assert len(report["measured_results"]) == 42
    assert all(value is None for value in report["measured_results"].values())
    assert report["non_substitution"]["pr_clock_model_ns"] == pytest.approx(222.22222222222223)
    assert report["non_substitution"]["copied_to_measured_source_to_action"] is False
    assert report["non_substitution"]["copied_to_measured_power"] is False
    assert report["claim_boundary"]["fpga_speed_advantage"] == "PROHIBITED"
    assert report["claim_boundary"]["zero_deadline_miss"] == "NOT_ESTABLISHED"


def test_bindings_recovery_contract_and_mutations_are_complete() -> None:
    report = _report()
    for binding in report["bindings"].values():
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]
    assert len(report["recovery_conditions"]) == 9
    assert all(row["required"] for row in report["recovery_conditions"])
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 11
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])


def test_any_invented_board_result_fails_closed() -> None:
    report = deepcopy(_report())
    report["measured_results"]["deadline_miss_count"] = 0
    report["claim_boundary"]["zero_deadline_miss"] = "ESTABLISHED"
    with pytest.raises(ValueError, match="gates/verdict"):
        gate.verify_report(report)
