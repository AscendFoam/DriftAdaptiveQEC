from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from cnn_fpga.benchmark.regime_aware_safe_policy_validation import run_validation


@pytest.fixture(scope="module")
def validation():
    root = Path("runs") / "test_t662_validation"
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    source_path = root / "source.csv"
    try:
        yield run_validation(report_path.resolve(), source_path.resolve()), source_path
    finally:
        report_path.unlink(missing_ok=True)
        source_path.unlink(missing_ok=True)
        root.rmdir()


def test_all_structural_gates_and_mutations_pass(validation) -> None:
    report, _ = validation
    assert report["gate_summary"] == {"passed": 20, "total": 20, "all_passed": True}
    assert all(row["passed"] for row in report["mutation_checks"])


def test_production_cadence_trace_is_continuous_and_six_cycle(validation) -> None:
    report, _ = validation
    assert report["contract"]["regime_window_cycles"] == 32
    assert report["contract"]["parameter_update_period_cycles"] == 4_000
    assert report["contract"]["fast_action_latency_cycles"] == 6
    assert report["trace"]["cycles"] == 20_061
    assert report["trace"]["commit_versions"] == [1, 2, 3, 4]
    assert report["trace"]["deferred_commit_cycles"] >= 3


def test_real_window_ewma_shadow_candidates_are_budgeted_and_fault_candidate_is_not_activated(
    validation,
) -> None:
    report, _ = validation
    candidates = report["trace"]["candidate_rows"]
    assert [row["cycle"] for row in candidates] == [4_000, 8_000, 12_000, 16_000]
    assert all(row["accepted"] and not row["deadline"]["host_budget_violation"] for row in candidates)
    compiled = report["candidate_provenance"]["compiled_candidates"]
    assert compiled["window_second"]["semantics_sha256"] != compiled["ewma_second"]["semantics_sha256"]
    assert report["gates"]["integrity_cancels_pending_window_semantics_before_version_reuse"]


def test_source_data_row_count_and_hash_are_recomputed(validation) -> None:
    report, source_path = validation
    with source_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 20_061
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["source_data"]["sha256"]
    assert all(row["source_to_action_cycles"] == "6" for row in rows)


def test_claim_boundary_remains_software_structural_only(validation) -> None:
    report, _ = validation
    closed = "|".join(report["claim_boundary"]["not_admitted"])
    assert "posterior calibration" in closed
    assert "LER superiority" in closed
    assert "board-measured" in closed
    assert report["posterior_fixture"]["status"].startswith("structural_branch_fixture")
