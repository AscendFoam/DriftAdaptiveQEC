from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.experimental_event_fsm_validation import (
    SCENARIOS,
    _implementation_sha256,
    run_validation,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "t4_2_2_experimental_event_fsm_validation.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_production_artifact_passes_all_twenty_gates_and_is_source_bound(
    artifact: dict,
) -> None:
    assert artifact["task_id"] == "T4.2.2"
    assert artifact["status"] == "PASS"
    assert artifact["implementation_sha256"] == _implementation_sha256()
    assert artifact["gate_summary"]["passed"] == 20
    assert artifact["gate_summary"]["failed"] == 0
    assert all(artifact["gate_summary"]["gates"].values())


def test_source_data_is_complete_has_all_modes_and_contains_no_truth_fields(
    artifact: dict,
) -> None:
    path = ROOT / artifact["source_data"]["path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == artifact["source_data"]["rows"] == 1024
    assert {row["scenario"] for row in rows} == set(SCENARIOS)
    assert {row["mode"] for row in rows} == {
        "normal",
        "x_recovery",
        "z_recovery",
        "hold",
        "reset_request",
        "fallback",
    }
    assert not any(
        token in column for column in rows[0] for token in ("truth", "hidden", "drift")
    )


def test_latency_and_initiation_interval_are_exact_not_average(artifact: dict) -> None:
    path = ROOT / artifact["source_data"]["path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert all(int(row["hardware_action_cycle"]) - int(row["source_cycle"]) == 6 for row in rows)
    for scenario in SCENARIOS:
        cycles = [
            int(row["hardware_action_cycle"])
            for row in rows
            if row["scenario"] == scenario
        ]
        assert cycles == list(range(6, 134))
    resource = artifact["resource_contract"]
    assert resource["map_plus_event_worst_case_latency_cycles"] == 6
    assert resource["initiation_interval_cycles"] == 1


def test_safe_modes_inhibit_flips_and_leave_every_frame_delta_zero(
    artifact: dict,
) -> None:
    path = ROOT / artifact["source_data"]["path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    safe = [row for row in rows if row["mode"] in {"hold", "reset_request", "fallback"}]
    pending_flips = [row for row in safe if row["map_logical_flip"] == "1"]
    assert safe and pending_flips
    assert all(row["correction_enable"] == "0" for row in safe)
    assert all(row["map_action_inhibited"] == "1" for row in pending_flips)
    for row in safe:
        assert row["pauli_frame_delta_x"] == row["pauli_frame_delta_z"] == "0"
        assert row["phase_frame_delta_x_code"] == row["phase_frame_delta_z_code"] == "0"


def test_all_six_counters_saturate_and_frame_wrap_is_nontrivial(artifact: dict) -> None:
    path = ROOT / artifact["source_data"]["path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for name in (
        "x_e_run",
        "z_e_run",
        "leakage_run",
        "leakage_clean_run",
        "health_good_run",
        "reset_wait_run",
    ):
        assert max(int(row[name]) for row in rows) == 7
    assert artifact["diagnostics"]["modulo_phase_wrap_count"] > 0


def test_bank_switch_is_monotonic_hash_bound_and_covers_degenerate_bank(
    artifact: dict,
) -> None:
    path = ROOT / artifact["source_data"]["path"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream) if row["scenario"] == "bank_version_switch"
        ]
    versions = [int(row["active_bank_version"]) for row in rows]
    hashes = artifact["diagnostics"]["image_sha256s"]
    assert versions == sorted(versions)
    assert set(versions) == set(range(8))
    assert all(row["map_image_sha256"] == hashes[int(row["active_bank_version"])] for row in rows)
    bank_zero = [row for row in rows if row["active_bank_version"] == "0"]
    assert bank_zero and all(row["map_logical_flip"] == "0" for row in bank_zero)


def test_negative_paths_are_transactional_and_hardware_claims_stay_null(
    artifact: dict,
) -> None:
    assert artifact["negative_audit"] and all(artifact["negative_audit"].values())
    resource = artifact["resource_contract"]
    for field in (
        "target_lut_count",
        "target_ff_count",
        "target_bram_count",
        "target_dsp_count",
        "fmax_mhz",
    ):
        assert resource[field] is None
    assert resource["rtl_measured"] is False
    assert resource["board_measured"] is False
    assert "complete conservative fallback" in artifact["claim_boundary"]["forbidden"]


def test_full_replay_regeneration_is_bit_deterministic(tmp_path: Path) -> None:
    payload = run_validation(
        cycles_per_scenario=128,
        json_path=tmp_path / "replay.json",
        csv_path=tmp_path / "replay.csv",
    )
    production = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["source_data"]["canonical_rows_sha256"] == production["source_data"]["canonical_rows_sha256"]
    assert payload["diagnostics"] == production["diagnostics"]
