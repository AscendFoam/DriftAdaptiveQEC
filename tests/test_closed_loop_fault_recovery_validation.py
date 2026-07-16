from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.closed_loop_fault_recovery_validation import (
    DEFAULT_CSV,
    DEFAULT_JSON,
    ROOT,
    SCENARIOS,
    SCHEMA_VERSION,
    FaultCampaignConfig,
    _implementation_sha256,
    run_validation,
)


def _artifact() -> dict:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


def test_fresh_production_validation_passes_all_17_gates(tmp_path: Path) -> None:
    result = run_validation(
        json_path=tmp_path / "validation.json",
        csv_path=tmp_path / "source.csv",
    )

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == "PASS"
    assert len(result["gates"]) == 17
    assert all(gate["passed"] for gate in result["gates"])
    assert result["summary"]["runs"] == 32
    assert result["summary"]["cycles_executed"] == 767_872
    assert result["source_data"]["rows"] == 436


def test_production_artifact_is_source_bound_and_complete() -> None:
    artifact = _artifact()
    assert artifact["status"] == "PASS"
    assert artifact["implementation_sha256"] == _implementation_sha256()
    assert artifact["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_CSV.read_bytes()
    ).hexdigest()
    with DEFAULT_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == artifact["source_data"]["rows"] == 436
    assert {row["record_type"] for row in rows} == {
        "cycle_segment",
        "run_summary",
        "update_attempt",
    }
    assert {row["scenario"] for row in rows} == set(SCENARIOS)


def test_required_scenario_outcomes_are_preserved_in_production_artifact() -> None:
    artifact = _artifact()
    per_run = artifact["summary"]["per_run"]
    grouped = {
        scenario: [row for row in per_run if row["scenario"] == scenario]
        for scenario in SCENARIOS
    }
    assert all(len(rows) == 4 for rows in grouped.values())
    assert all(row["undefined_action_count"] == 0 for row in per_run)
    assert all(row["blocking_fault_with_correction_count"] == 0 for row in per_run)
    assert all(row["frame_out_of_range_count"] == 0 for row in per_run)
    assert all(row["active_version_monotonic"] for row in per_run)

    assert all(
        row["fault_counts"]["leakage_observed"] >= 3
        and row["reset_request_cycles"] > 0
        for row in grouped["leakage_reset"]
    )
    assert all(
        row["fault_counts"]["parameter_stale"] > 0
        and row["commit_versions"] == [1, 2]
        and row["final_record"]["active_semantics_sha256"] == row["v1_semantics_sha256"]
        for row in grouped["host_timeout"]
    )
    assert all(
        row["ack_timeout_cycles"] > 0
        and row["confirmed_readbacks"] > 0
        for row in grouped["communication_pause_ack_loss"]
    )
    assert all(
        row["commit_epochs"] == [6200, 10200, 18200]
        and row["final_record"]["active_semantics_sha256"] == row["v0_semantics_sha256"]
        for row in grouped["post_commit_guard_republish"]
    )


def test_race_and_corrupt_transfer_evidence_is_transaction_scoped() -> None:
    per_run = _artifact()["summary"]["per_run"]
    for row in (item for item in per_run if item["scenario"] == "update_race"):
        initial = [
            item
            for item in row["update_attempts"]
            if item["transaction_id"].startswith("race-")
        ]
        assert len(initial) == 2
        assert sum(item["accepted"] for item in initial) == 1
        assert sum("writer_conflict" in item["reason"] for item in initial) == 1
    for row in (item for item in per_run if item["scenario"] == "corrupt_transfer"):
        corrupt = [
            item
            for item in row["update_attempts"]
            if item["transaction_id"].startswith("corrupt-")
        ]
        assert len(corrupt) == 1
        assert not corrupt[0]["accepted"]
        assert corrupt[0]["reason"] == "transfer_crc_mismatch"


def test_claim_boundary_does_not_promote_software_campaign_to_hardware() -> None:
    artifact = _artifact()
    assert "not_rtl_or_board" in artifact["scope"]
    forbidden = " ".join(artifact["claim_boundary"]["forbidden"]).lower()
    assert "rtl" in forbidden and "board" in forbidden and "physical" in forbidden


def test_campaign_config_rejects_demo_scale_or_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="at least 20,000"):
        FaultCampaignConfig(n_cycles=10_000)
    with pytest.raises(ValueError, match="four unique"):
        FaultCampaignConfig(seeds=(1, 1, 2, 3))


def test_human_contract_and_document_maps_are_synchronized_when_present() -> None:
    human_path = ROOT / "docs" / "closed_loop_fault_recovery.md"
    if not human_path.exists():
        return
    human = human_path.read_text(encoding="utf-8")
    protocol = (ROOT / "docs" / "protocol_hierarchy.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    benchmark = (ROOT / "cnn_fpga" / "benchmark" / "README.md").read_text(
        encoding="utf-8"
    )
    for fragment in ("767872", "32", "host timeout", "ack", "LKG", "rollback"):
        assert fragment in human
    assert "3.37" in protocol and "Fault recovery" in protocol
    assert "closed_loop_fault_recovery.md" in readme
    assert "closed_loop_fault_recovery_validation.py" in benchmark
