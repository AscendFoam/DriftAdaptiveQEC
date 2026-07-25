from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.phase9_dual_backend_post_outcome_audit import (
    EXPECTED_BLOCKED,
    EXPECTED_CLAIMS,
    EXPECTED_FAILURES,
    EXPECTED_RELEASED,
    SCHEMA,
    VERDICT,
    build_audit,
    build_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/t9_2_4_dual_backend_post_outcome_audit.json"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def test_live_formal_snapshot_is_complete_and_no_go() -> None:
    snapshot = build_snapshot(ROOT)
    assert snapshot["ledger_count"] == 16800
    assert snapshot["ledger_unique"] == 16800
    assert snapshot["source_gate_count"] == 1042
    assert snapshot["source_passed_count"] == 825
    assert snapshot["source_failed_count"] == 217
    assert snapshot["failure_counts"] == EXPECTED_FAILURES
    assert snapshot["released_tasks"] == sorted(EXPECTED_RELEASED)
    assert snapshot["blocked_tasks"] == sorted(EXPECTED_BLOCKED)
    assert snapshot["seed_overlap"] == 0
    assert snapshot["archive_id_coverage"] is True
    assert snapshot["worst_source_bound"] > snapshot["worst_tolerance"]


def test_audit_has_independent_gates_and_killable_mutations() -> None:
    audit = build_audit(ROOT)
    assert audit["schema_version"] == SCHEMA
    assert audit["verdict"] == VERDICT
    assert audit["gate_summary"] == {
        "passed": 25,
        "total": 25,
        "all_passed": True,
    }
    assert audit["mutation_summary"] == {
        "detected": 24,
        "total": 24,
        "all_detected": True,
    }
    assert all(value is None for value in audit["claim_state"].values())
    assert tuple(audit["claim_state"]) == EXPECTED_CLAIMS


def test_persisted_audit_is_canonical_and_live_bound() -> None:
    persisted = json.loads(OUTPUT.read_text(encoding="utf-8"))
    regenerated = build_audit(ROOT)
    assert persisted == regenerated
    stored = persisted.pop("analysis_sha256")
    assert stored == sha256(
        _canonical(persisted).encode("utf-8")
    ).hexdigest()


def test_auditor_does_not_import_physics_or_formal_verifier() -> None:
    source_path = (
        ROOT
        / "cnn_fpga/benchmark/phase9_dual_backend_post_outcome_audit.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith("physics") for name in imported)
    assert not any(
        "phase9_dual_backend_verifier" in name for name in imported
    )


def test_live_formal_outputs_make_child_seal_one_shot() -> None:
    from cnn_fpga.benchmark import phase9_twin_formal_runner_amendment

    config = json.loads(
        (
            ROOT
            / "configs/phase9/t9_2_4_formal_runner_amendment.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(
        ValueError,
        match="formal outputs already exist before child seal",
    ):
        phase9_twin_formal_runner_amendment.build_seal(ROOT, config)
