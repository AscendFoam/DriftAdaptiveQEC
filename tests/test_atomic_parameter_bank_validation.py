from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from cnn_fpga.benchmark.atomic_parameter_bank_validation import (
    DEFAULT_CSV,
    DEFAULT_JSON,
    ROOT,
    SCHEMA_VERSION,
    _implementation_sha256,
    run_validation,
)


def test_validation_passes_all_gates_and_writes_7518_rows(tmp_path: Path) -> None:
    result = run_validation(
        json_path=tmp_path / "validation.json",
        csv_path=tmp_path / "source.csv",
    )

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == "PASS"
    assert len(result["gates"]) == 17
    assert all(gate["passed"] for gate in result["gates"])
    assert result["source_data"]["rows"] == 7518


def test_source_data_exhausts_every_prefix_cut_and_byte_flip(tmp_path: Path) -> None:
    csv_path = tmp_path / "source.csv"
    result = run_validation(json_path=tmp_path / "result.json", csv_path=csv_path)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    partial = [row for row in rows if row["scenario_family"] == "partial_cut"]
    corrupt = [
        row for row in rows if row["scenario_family"] == "single_byte_corruption"
    ]

    assert len(partial) == len(corrupt) == 3745
    assert [int(row["index"]) for row in partial] == list(range(3745))
    assert [int(row["index"]) for row in corrupt] == list(range(3745))
    assert {row["observed"] for row in partial} == {"payload_incomplete"}
    assert {row["observed"] for row in corrupt} == {"transfer_crc_mismatch"}
    assert result["summary"]["payload_lengths"][1] == 3745


def test_chunk_matrix_preserves_slots_until_finalize_and_confirms_readback(
    tmp_path: Path,
) -> None:
    result = run_validation(
        json_path=tmp_path / "result.json", csv_path=tmp_path / "source.csv"
    )
    chunk = result["summary"]["chunk_commit_matrix"]

    assert chunk["cases"] == 10
    assert chunk["chunk_sizes"] == [1, 7, 64, 511, 3745]
    assert chunk["all_intermediate_active_and_inactive_slots_immutable"] is True
    assert chunk["all_committed_and_confirmed"] is True
    assert chunk["all_unsafe_boundaries_deferred"] is True


def test_negative_matrix_has_integrity_cas_hysteresis_timestamp_and_stale_reasons(
    tmp_path: Path,
) -> None:
    result = run_validation(
        json_path=tmp_path / "result.json", csv_path=tmp_path / "source.csv"
    )
    negative = result["summary"]["negative_scenarios"]
    assert negative["all_active_unchanged"] is True
    assert {
        "hysteresis_not_satisfied",
        "hysteresis_invalidated",
        "timestamp_epoch_mismatch",
        "expected_active_version_mismatch",
        "transfer_crc_mismatch",
        "transfer_sha256_mismatch",
        "manifest_image_digest_mismatch",
        "manifest_crc_mismatch",
        "manifest_sha256_mismatch",
        "transaction_replay",
        "payload_stale",
        "payload_stale_before_commit",
    } <= set(negative["reasons"])


def test_double_bank_pipeline_and_concurrent_writer_evidence(tmp_path: Path) -> None:
    result = run_validation(
        json_path=tmp_path / "result.json", csv_path=tmp_path / "source.csv"
    )
    evidence = result["summary"]["double_bank_pipeline_and_race"]

    assert evidence["bank_sequence"] == ["A:v0", "B:v1", "A:v2"]
    assert evidence["inflight_versions"] == [0, 1]
    assert evidence["first_commit_confirmed"] is True
    assert evidence["second_commit_confirmed"] is True
    assert evidence["race_results"] == [
        "accepted",
        "writer_conflict_transfer_in_progress",
    ]


def test_manifest_and_claim_boundary_are_complete(tmp_path: Path) -> None:
    result = run_validation(
        json_path=tmp_path / "result.json", csv_path=tmp_path / "source.csv"
    )
    fields = set(result["manifest_fields"])
    assert {
        "expected_active_version",
        "new_version",
        "created_timestamp_ns",
        "apply_epoch",
        "payload_crc32",
        "payload_sha256",
        "image_crc32",
        "image_sha256",
        "manifest_crc32",
        "manifest_sha256",
    } <= fields
    assert "RTL atomicity" in result["claim_boundary"]["forbidden"]
    assert "not_rtl_or_board" in result["scope"]


def test_production_artifact_is_source_bound_when_present() -> None:
    if not (DEFAULT_JSON.exists() and DEFAULT_CSV.exists()):
        return
    artifact = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["implementation_sha256"] == _implementation_sha256()
    assert artifact["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_CSV.read_bytes()
    ).hexdigest()
    with DEFAULT_CSV.open(newline="", encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == artifact["source_data"]["rows"]


def test_human_contract_and_document_maps_are_synchronized_when_present() -> None:
    human_path = ROOT / "docs" / "atomic_parameter_bank.md"
    if not human_path.exists():
        return
    human = human_path.read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "protocol_hierarchy.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    benchmark = (ROOT / "cnn_fpga" / "benchmark" / "README.md").read_text(
        encoding="utf-8"
    )

    for fragment in ("7518", "3745", "CAS", "hysteresis", "ack", "readback"):
        assert fragment in human
    assert "3.36" in protocol and "Atomic" in protocol
    assert "atomic_parameter_bank.md" in readme
    assert "atomic_parameter_bank_validation.py" in benchmark
