from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.conservative_fallback_validation import (
    SCENARIOS,
    _implementation_sha256,
    run_validation,
)
from cnn_fpga.runtime.conservative_fallback import FAULT_BITS, FAULT_ORDER


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "t4_2_3_conservative_fallback_validation.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(artifact: dict) -> list[dict[str, str]]:
    with (ROOT / artifact["source_data"]["path"]).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_production_artifact_passes_all_twenty_gates_and_is_source_bound(
    artifact: dict,
) -> None:
    assert artifact["task_id"] == "T4.2.3"
    assert artifact["status"] == "PASS"
    assert artifact["implementation_sha256"] == _implementation_sha256()
    assert artifact["gate_summary"]["passed"] == 20
    assert artifact["gate_summary"]["failed"] == 0
    assert all(artifact["gate_summary"]["gates"].values())


def test_source_data_covers_sixteen_scenarios_and_all_fault_bits(
    artifact: dict, rows: list[dict[str, str]]
) -> None:
    assert len(rows) == artifact["source_data"]["rows"] == 4096
    assert {row["scenario"] for row in rows} == set(SCENARIOS)
    observed = {
        flag for row in rows for flag in row["fault_flags"].split("|") if flag
    }
    assert observed == set(FAULT_ORDER)
    assert [item["name"] for item in artifact["fault_registry"]] == list(FAULT_ORDER)
    assert [item["bit"] for item in artifact["fault_registry"]] == [
        FAULT_BITS[name] for name in FAULT_ORDER
    ]


def test_every_blocking_fault_is_frame_hold_without_map_or_frame_delta(
    rows: list[dict[str, str]],
) -> None:
    blocking = [
        row
        for row in rows
        if any(flag and flag != "leakage_observed" for flag in row["fault_flags"].split("|"))
    ]
    assert blocking
    for row in blocking:
        assert row["status"] == "fallback"
        assert row["hardware_mode"] == "fallback"
        assert row["conservative_action"] == "frame_hold"
        assert row["active_profile_id"] == "frame_hold_no_map"
        assert row["map_decision_accepted"] == "0"
        assert row["correction_enable"] == "0"
        assert row["pauli_frame_delta_x"] == row["pauli_frame_delta_z"] == "0"
        assert row["phase_frame_delta_x_code"] == row["phase_frame_delta_z_code"] == "0"


def test_ood_and_age_boundaries_are_exact(rows: list[dict[str, str]]) -> None:
    ood = [row for row in rows if row["scenario"] == "ood_boundary"]
    age = [row for row in rows if row["scenario"] == "parameter_age_boundary"]
    assert any(row["ood_score_code"] == "192" and not row["fault_flags"] for row in ood)
    assert all(
        ("ood_score_exceeded" in row["fault_flags"])
        == (int(row["ood_score_code"]) > 192)
        for row in ood
    )
    assert any(row["parameter_age_cycles"] == "64" and not row["fault_flags"] for row in age)
    assert all(
        ("parameter_stale" in row["fault_flags"])
        == (int(row["parameter_age_cycles"]) > 64)
        for row in age
    )


def test_version_commits_are_monotonic_and_faults_preserve_trusted_version(
    rows: list[dict[str, str]],
) -> None:
    switch = [row for row in rows if row["scenario"] == "valid_version_switch"]
    versions = [int(row["trusted_version_after"]) for row in switch]
    assert versions == sorted(versions)
    assert set(versions) == set(range(8))
    faults = [row for row in rows if row["scenario"] == "version_faults" and row["fault_flags"]]
    assert faults
    assert all(row["trusted_version_before"] == row["trusted_version_after"] for row in faults)


def test_hysteresis_counter_saturation_latency_and_reason_masks_are_exact(
    rows: list[dict[str, str]],
) -> None:
    recovery = [row for row in rows if row["scenario"] == "fallback_recovery"]
    assert all(row["status"] == "recovering" for row in recovery if int(row["scenario_offset"]) % 8 == 1)
    assert all(row["status"] == "healthy" for row in recovery if int(row["scenario_offset"]) % 8 == 2)
    saturation = [row for row in rows if row["scenario"] == "fault_counter_saturation"]
    assert max(int(row["fault_run"]) for row in saturation) == 255
    assert max(int(row["fault_cycle_count"]) for row in saturation) == 255
    assert all(int(row["hardware_action_cycle"]) - int(row["source_cycle"]) == 6 for row in rows)
    for row in rows:
        flags = [flag for flag in row["fault_flags"].split("|") if flag]
        assert int(row["fault_mask"]) == sum(FAULT_BITS[flag] for flag in flags)


def test_online_and_hardware_claim_boundaries_remain_fail_closed(artifact: dict) -> None:
    assert artifact["online_contract"]["hidden_truth_inputs"] == []
    assert not any(
        token in field
        for field in artifact["online_contract"]["input_fields"]
        for token in ("truth", "hidden", "drift", "recovery_depth")
    )
    resource = artifact["resource_contract"]
    for field in (
        "target_lut_count",
        "target_ff_count",
        "target_bram_count",
        "target_dsp_count",
        "fmax_mhz",
    ):
        assert resource[field] is None
    assert resource["rtl_measured"] is False and resource["board_measured"] is False
    assert "automatic bank rollback" in artifact["claim_boundary"]["forbidden"]


def test_full_replay_regeneration_is_bit_deterministic(tmp_path: Path) -> None:
    payload = run_validation(
        cycles_per_scenario=256,
        json_path=tmp_path / "replay.json",
        csv_path=tmp_path / "replay.csv",
    )
    production = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["source_data"]["canonical_rows_sha256"] == production["source_data"]["canonical_rows_sha256"]
    assert payload["diagnostics"] == production["diagnostics"]
