"""Tests for the independent fresh-twin post-outcome audit.

The module-scoped fixture performs exactly one full production audit.  This is
intentional: the audit streams the 528,384-row ledger and validates every
member of the 824 MB raw archive, so repeating it in each test would add cost
without adding coverage.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase9_fresh_twin_post_outcome_audit as audit


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return audit.write_artifacts(ROOT)


def _self_hash(document: dict[str, object]) -> str:
    payload = dict(document)
    payload.pop("analysis_sha256")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_live_audit_passes_all_independent_gates(report):
    assert report["verdict"] == audit.VERDICT
    assert report["formal_verdict"] == audit.FORMAL_NO_GO
    assert report["gate_summary"] == {
        "passed": 40,
        "total": 40,
        "all_passed": True,
    }


def test_live_audit_recomputes_full_formal_denominators(report):
    assert report["execution_summary"] == {
        "rows": 528_384,
        "chunks": 592,
        "attempt_events": 594,
        "exception_rows": 0,
        "archive_members": 594,
    }
    assert report["gate_outcome"]["total"] == 1_589
    assert report["gate_outcome"]["passed"] == 1_562
    assert report["gate_outcome"]["failed"] == 27
    assert report["gate_outcome"]["family_totals"] == (
        audit.EXPECTED_FAMILY_TOTALS
    )
    assert report["gate_outcome"]["failed_family_counts"] == (
        audit.EXPECTED_FAILED_FAMILIES
    )
    assert len(report["gate_outcome"]["failed_gate_ids"]) == 27


def test_mutation_suite_covers_semantics_and_three_release_branches(report):
    assert len(report["semantic_mutations"]) == 51
    assert len(report["branch_mutations"]) == 18
    assert report["mutation_summary"] == {
        "detected": 69,
        "total": 69,
        "all_detected": True,
    }
    assert {row["branch"] for row in report["branch_mutations"]} == {
        "pass",
        "no_go",
        "incomplete",
    }


@pytest.mark.parametrize(
    ("branch", "expected_verdict"),
    [
        ("pass", audit.FORMAL_PASS),
        ("no_go", audit.FORMAL_NO_GO),
        ("incomplete", audit.FORMAL_INCOMPLETE),
    ],
)
def test_all_release_branch_contracts_are_internally_valid(
    branch, expected_verdict
):
    fixture = audit._branch_fixture(branch)
    assert fixture["verdict"] == expected_verdict
    assert all(audit.audit_branch_fixture(fixture).values())


def test_no_go_is_fail_closed_and_all_claims_are_typed_null(report):
    assert report["qualified_claim"] is None
    assert set(report["claim_state"]) == set(audit.CLAIM_FIELDS)
    assert len(report["claim_state"]) == 15
    assert all(value is None for value in report["claim_state"].values())
    assert report["release_state"] == {
        task: False for task in audit.DOWNSTREAM_TASKS
    }
    assert report["historical_t9_2_4_no_go_preserved"] is True


def test_persisted_report_and_source_are_canonical_and_bound(report):
    persisted = json.loads((ROOT / audit.OUTPUT_PATH).read_text("utf-8"))
    assert persisted == report
    assert persisted["analysis_sha256"] == _self_hash(persisted)
    source = (ROOT / audit.SOURCE_PATH).read_bytes()
    assert len(source) == persisted["source_data"]["bytes"]
    assert hashlib.sha256(source).hexdigest() == (
        persisted["source_data"]["sha256"]
    )
    rows = list(csv.DictReader(io.StringIO(source.decode("utf-8"))))
    assert len(rows) == persisted["source_data"]["rows"]
    assert sum(row["record_type"] == "failed_formal_gate" for row in rows) == 27


def test_auditor_has_no_project_or_formal_pipeline_imports():
    source = (
        ROOT
        / "cnn_fpga/benchmark/phase9_fresh_twin_post_outcome_audit.py"
    ).read_text("utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = (
        "physics",
        "phase9_fresh_twin_qualification",
        "phase9_fresh_twin_formal_runner",
        "phase9_fresh_twin_verifier",
    )
    assert not [
        name for name in imported if any(token in name for token in forbidden)
    ]


def test_auditor_does_not_reference_old_formal_raw_evidence_literals():
    source = (
        ROOT
        / "cnn_fpga/benchmark/phase9_fresh_twin_post_outcome_audit.py"
    ).read_text("utf-8")
    prohibited = [
        "t9_2_4_dual_backend_" + "cell_ledger.csv",
        "t9_2_4_dual_backend_" + "raw_archive.zip",
        "t9_2_4_dual_backend_" + "qualification_source_data.csv",
    ]
    assert all(path not in source for path in prohibited)


def test_large_raw_evidence_paths_are_exactly_lfs_tracked():
    lines = (ROOT / ".gitattributes").read_text("utf-8").splitlines()
    expected = {
        "docs/t_risk_20260726_01_fresh_cell_ledger.csv "
        "filter=lfs diff=lfs merge=lfs -text",
        "docs/t_risk_20260726_01_fresh_raw_archive.zip "
        "filter=lfs diff=lfs merge=lfs -text",
    }
    assert expected <= set(lines)


def test_audit_source_compiles_and_public_contract_is_explicit():
    source_path = (
        ROOT
        / "cnn_fpga/benchmark/phase9_fresh_twin_post_outcome_audit.py"
    )
    compile(source_path.read_text("utf-8"), str(source_path), "exec")
    assert {
        "audit_branch_fixture",
        "audit_snapshot",
        "build_audit",
        "build_snapshot",
        "run_mutation_audit",
        "write_artifacts",
    } <= set(audit.__all__)


def test_attempt_chain_parser_detects_raw_event_tamper(tmp_path):
    events: list[dict[str, object]] = []
    previous = "0" * 64
    for index, kind in enumerate(("RUN_STARTED", "FINALIZED")):
        event: dict[str, object] = {
            "event_index": index,
            "event_kind": kind,
            "previous_event_sha256": previous,
            "payload": {"sentinel": index},
        }
        event["event_sha256"] = audit._sha(event)
        previous = str(event["event_sha256"])
        events.append(event)
    ledger = tmp_path / "attempt.jsonl"
    ledger.write_bytes(
        b"".join(audit._canonical(event) + b"\n" for event in events)
    )
    assert audit._scan_attempt_ledger(ledger)["chain_valid"] is True

    events[0]["payload"] = {"sentinel": "tampered"}
    ledger.write_bytes(
        b"".join(audit._canonical(event) + b"\n" for event in events)
    )
    assert audit._scan_attempt_ledger(ledger)["chain_valid"] is False


def test_gate_parser_recomputes_upper_and_lower_bounds(tmp_path):
    fields = (
        "gate_id",
        "family",
        "stage",
        "metric",
        "direction",
        "estimate",
        "standard_error",
        "bound",
        "margin",
        "cluster_count",
        "passed",
        "denominator",
    )
    rows = [
        {
            "gate_id": "upper",
            "family": "f",
            "stage": "s",
            "metric": "m",
            "direction": "upper",
            "estimate": "0.1",
            "standard_error": "0.01",
            "bound": str(0.1 + audit.Z_TOST * 0.01),
            "margin": "0.2",
            "cluster_count": "8",
            "passed": "True",
            "denominator": "all generated rows",
        },
        {
            "gate_id": "lower",
            "family": "f",
            "stage": "s",
            "metric": "m",
            "direction": "lower",
            "estimate": "0.9",
            "standard_error": "0.01",
            "bound": str(0.9 - audit.Z_TOST * 0.01),
            "margin": "0.8",
            "cluster_count": "8",
            "passed": "True",
            "denominator": "all generated rows",
        },
    ]
    ledger = tmp_path / "gates.csv"
    with ledger.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    scanned = audit._scan_gate_ledger(ledger)
    assert scanned["arithmetic_failures"] == []
    assert scanned["declared_pass_failures"] == []
    assert scanned["no_postselection_denominators"] is True

    rows[0]["bound"] = "0.0"
    rows[1]["passed"] = "False"
    with ledger.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tampered = audit._scan_gate_ledger(ledger)
    assert tampered["arithmetic_failures"] == ["upper"]
    assert tampered["declared_pass_failures"] == ["lower"]
