from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import claim_evidence_boundary_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(matrix.DEFAULT_REPORT.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repository_matrix_verifies_end_to_end() -> None:
    checks = matrix.verify_report()
    assert checks == {"identity": True, "gates": True, "verdict": True, "analysis_hash": True}
    report = _report()
    assert report["gate_summary"] == {"passed": 19, "failed": []}
    assert report["verdict"] == matrix.VERDICT


def test_all_v4_v5_and_secondary_claims_are_accounted_without_duplicates() -> None:
    report = _report()
    claims = report["claims"]
    assert len(claims) == len({row["claim_id"] for row in claims}) == 29
    assert report["claim_coverage"] == {
        "v4": sorted(matrix.V4_IDS), "v5": sorted(matrix.V5_IDS),
        "phase6c": sorted(matrix.PHASE6C_IDS), "total": 29,
    }
    assert {row["lane_id"] for row in claims if row["source_group"] == "PHASE6C_SECONDARY"} == matrix.LANES


def test_revoked_blocked_and_negative_claims_cannot_enter_title_or_abstract() -> None:
    report = _report()
    blocked = [row for row in report["claims"] if row["publication_state"] in {"MANDATORY_NEGATIVE", "PROHIBITED_POSITIVE", "BLOCKED"}]
    assert blocked
    assert all(row["assertion_polarity"] != "POSITIVE" for row in blocked)
    assert all(not row["placements"]["title"] and not row["placements"]["abstract"] for row in blocked)
    v5 = [row for row in report["claims"] if row["source_group"] == "V5_EARLY_STOP"]
    assert all(row["assertion_polarity"] != "POSITIVE" for row in v5)


def test_every_positive_claim_has_live_report_source_data_and_code() -> None:
    report = _report()
    artifacts = report["artifact_registry"]
    for row in report["claims"]:
        if row["assertion_polarity"] != "POSITIVE":
            continue
        for category in ("reports", "raw_data", "code"):
            assert row["evidence"][category]
            for artifact_id in row["evidence"][category]:
                binding = artifacts[artifact_id]
                path = ROOT / binding["path"]
                assert path.stat().st_size == binding["bytes"]
                assert _sha256(path) == binding["sha256"]
    secondary = [row for row in report["claims"] if row["source_group"] == "PHASE6C_SECONDARY"]
    assert all(len(row["evidence"]["reports"]) >= 2 for row in secondary)
    assert all(len(row["evidence"]["raw_data"]) >= 2 for row in secondary)
    assert all(len(row["evidence"]["code"]) >= 2 for row in secondary)


def test_source_csv_is_lossless_and_carries_hashes_for_each_claim() -> None:
    report = _report()
    with matrix.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 29
    assert {row["claim_id"] for row in rows} == {row["claim_id"] for row in report["claims"]}
    for row in rows:
        evidence_ids = json.loads(row["report_artifacts_json"]) + json.loads(row["raw_artifacts_json"]) + json.loads(row["code_artifacts_json"])
        hashes = json.loads(row["artifact_hashes_json"])
        assert set(hashes) == set(evidence_ids)
        assert all(len(value) == 64 for value in hashes.values())


def test_board_binding_ignores_unrelated_progress_but_detects_claim_prerequisite_change() -> None:
    text = matrix.BOARD.read_text(encoding="utf-8")
    original = matrix._board_binding(text)
    unrelated = matrix._board_binding(text.replace("| T7.1.2 | Todo |", "| T7.1.2 | In Progress |"))
    assert unrelated["canonical_sha256"] == original["canonical_sha256"]
    relevant = matrix._board_binding(text.replace("| T6.14.3 | Dropped |", "| T6.14.3 | Done |"))
    assert relevant["canonical_sha256"] != original["canonical_sha256"]


def test_post_route_estimate_cannot_be_promoted_to_board_measurement() -> None:
    report = deepcopy(_report())
    claim = next(row for row in report["claims"] if row["claim_id"] == "FPGA_DETERMINISTIC_ARCHITECTURE")
    claim["evidence_layers"]["current"].append("BOARD_MEASURED")
    assert matrix.evaluate_gates(report, check_live_files=False)["G11_board_measurement_is_blocked_and_all_fields_are_null"] is False
    with pytest.raises(ValueError, match="verification failed"):
        matrix.verify_report(report)


def test_one_substantive_mutation_per_gate_is_detected() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 19
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])
