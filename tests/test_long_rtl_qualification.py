from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from cnn_fpga.benchmark.long_rtl_qualification import (
    AbstractTransportAdapter,
    FAMILY_CYCLES,
    FAMILY_NAMES,
    REACHABLE_FAULT_BITS,
    ROOT,
    STRUCTURAL_ZERO_FAULT_BITS,
    TRACE_STRUCT,
    _fault_marker,
    evaluate_gates,
    legacy_crosscheck,
)


REPORT = ROOT / "docs/t6_2_2_long_rtl_qualification.json"
SOURCE_DATA = ROOT / "docs/t6_2_2_long_rtl_qualification_source_data.csv"


def _report() -> dict[str, object]:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_binary_contract_and_independent_short_crosscheck() -> None:
    assert TRACE_STRUCT.size == 82
    assert legacy_crosscheck(256) == {
        "rows": 256,
        "mismatches": 0,
        "first_mismatch": None,
    }


def test_abstract_transport_accounts_overflow_and_drains() -> None:
    adapter = AbstractTransportAdapter(capacity=2, deadline_budget=1)
    for cycle in range(6):
        adapter.cycle(cycle, _fault_marker(cycle), pause=True)
    for cycle in range(6, 24):
        adapter.cycle(cycle, None)
    summary = adapter.summary()
    assert summary["overflow_events"] == 4
    assert summary["accounted_overflow_events"] == 4
    assert summary["silent_overflow"] == 0
    assert summary["max_fifo_depth"] == 2
    assert summary["pending_fifo"] == 0
    assert summary["pending_markers"] == 0
    assert summary["explicit_fault_markers"] >= 4


def test_qualifying_artifact_recomputes_all_machine_gates() -> None:
    report = _report()
    assert report["verdict"] == "PASS_BOARD_INDEPENDENT_LONG_RTL_QUALIFICATION_READY_FOR_ROUTE_A"
    assert all(report["gates"].values())
    assert all(evaluate_gates(report).values())
    assert report["cycles_per_family"] == FAMILY_CYCLES
    assert report["aggregate_python"]["cycles"] == 1_000_000
    assert report["family_names"] == list(FAMILY_NAMES)
    assert report["legacy_crosscheck"]["rows"] >= 10_000
    assert report["legacy_crosscheck"]["mismatches"] == 0


def test_trace_and_every_cxxrtl_family_are_exact_and_hash_bound() -> None:
    report = _report()
    trace = ROOT / report["trace"]["path"]
    assert trace.stat().st_size == 1_000_000 * TRACE_STRUCT.size == 82_000_000
    assert _sha256(trace) == report["trace"]["sha256"]
    rows = report["cxxrtl_families"]
    assert len(rows) == 10
    assert sum(row["rows"] for row in rows) == 1_000_000
    assert all(row["rows"] == FAMILY_CYCLES for row in rows)
    assert all(row["mismatches"] == 0 for row in rows)
    assert all(row["undefined_actions"] == 0 for row in rows)
    assert all(row["actual_digest"] == row["expected_digest"] for row in rows)
    assert sum(row["shadow_mutations"] for row in rows) == 8
    assert sum(row["shadow_mutations_detected"] for row in rows) == 8


def test_fault_coverage_saturation_and_family_specific_recovery() -> None:
    report = _report()
    aggregate = report["aggregate_python"]
    by_id = {row["family_id"]: row for row in report["python_families"]}
    assert aggregate["undefined_actions"] == 0
    assert aggregate["output_crc_errors"] == 0
    assert aggregate["state_crc_errors"] == 0
    assert all(aggregate["fault_bits"][str(bit)] > 0 for bit in REACHABLE_FAULT_BITS)
    assert all(aggregate["fault_bits"][str(bit)] == 0 for bit in STRUCTURAL_ZERO_FAULT_BITS)
    assert all(value > 0 for value in aggregate["modes"].values())
    assert all(value > 0 for value in aggregate["health"].values())
    assert all(value > 0 for value in aggregate["actions"].values())
    assert all(by_id[index]["fault_to_healthy_recoveries"] > 0 for index in (2, 3, 4, 6, 7, 8, 9))
    assert by_id[9]["llr_min_hits"] > 0
    assert by_id[9]["llr_max_hits"] > 0
    assert all(value == 255 for value in by_id[9]["maxima"].values())


def test_transport_disturbances_are_explicit_bounded_and_recover() -> None:
    report = _report()
    by_id = {row["family_id"]: row for row in report["python_families"]}
    required = {
        6: ("pause_cycles", "backpressure_cycles", "overflow_events", "deadline_faults", "explicit_fault_markers"),
        7: ("pause_cycles", "drop_events", "duplicate_events", "reorder_events", "sequence_faults", "explicit_fault_markers"),
        8: ("pause_cycles", "overflow_events", "drop_events", "duplicate_events", "reorder_events", "sequence_faults", "deadline_faults", "explicit_fault_markers"),
    }
    for family_id, fields in required.items():
        family = by_id[family_id]
        transport = family["transport"]
        assert all(transport[field] > 0 for field in fields)
        assert transport["max_fifo_depth"] <= 8
        assert transport["pending_fifo"] == 0
        assert transport["pending_markers"] == 0
        assert transport["silent_overflow"] == 0
        assert family["final_valid_mode"] == 0
        assert family["final_valid_health"] == 0
        assert family["final_valid_fault_mask"] == 0
        assert family["last_healthy_output_cycle"] > family["last_fault_output_cycle"]


def test_commit_negative_paths_and_mutation_audits_are_real() -> None:
    report = _report()
    aggregate = report["aggregate_python"]
    assert aggregate["commit_acks"] > 0
    assert aggregate["commit_rejections"] > 0
    assert aggregate["rollback_commit_rejections"] > 0
    assert aggregate["untrusted_commit_rejections"] > 0
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 8
    assert all(row["rejected"] for row in audit["mutations"])


def test_source_data_has_one_complete_row_per_family() -> None:
    with SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert [row["family"] for row in rows] == list(FAMILY_NAMES)
    assert all(int(row["cycles"]) == FAMILY_CYCLES for row in rows)
    assert all(int(row["cxxrtl_mismatches"]) == 0 for row in rows)
    by_id = {int(row["family_id"]): row for row in rows}
    assert int(by_id[7]["drop_events"]) > 0
    assert int(by_id[7]["duplicate_events"]) > 0
    assert int(by_id[7]["reorder_events"]) > 0
    assert all(row["actual_digest"] == row["expected_digest"] for row in rows)

