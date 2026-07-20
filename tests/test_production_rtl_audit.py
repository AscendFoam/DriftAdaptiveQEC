from __future__ import annotations

import csv
import json
from pathlib import Path
import zlib

from cnn_fpga.benchmark.production_rtl_audit import (
    CORE,
    DEFAULT_CSV,
    DEFAULT_JSON,
    DRIVER,
    TOP,
    VERDICT,
    build_management_trace,
)
from cnn_fpga.runtime.production_fast_path_management import (
    ProductionFastPathManagementReference,
    REJECT_ACTIVE_BANK,
    crc16_ccitt_little_endian,
    crc32_table_words,
)


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


def test_production_sources_are_substantive_and_synthesizable_contract_is_explicit() -> None:
    assert CORE.stat().st_size > 20_000
    assert TOP.stat().st_size > 20_000
    assert DRIVER.stat().st_size > 5_000
    source = TOP.read_text(encoding="utf-8")
    for token in (
        "cfg_word_count != 10'd514",
        "crc32_word22",
        "core_commit_valid = commit_pending && safe_boundary",
        "RETIRED_BANK_DRAIN_CYCLES",
        "management_snapshot_byte_index == 5'd17",
    ):
        assert token in source


def test_core_fails_closed_on_raw_config_and_commit_inputs() -> None:
    source = CORE.read_text(encoding="utf-8")
    assert "cfg_address <= 9'd256" in source
    assert "cfg_bank != active_bank" in source
    assert "requested_bank_trusted" in source
    assert "commit_bank != active_bank" in source
    assert "commit_version == (active_version + 16'd1)" in source


def test_production_thresholds_remove_demo_scale_version_and_age_limits() -> None:
    source = TOP.read_text(encoding="utf-8")
    assert "MAX_PARAMETER_AGE_CYCLES = 16'd8192" in source
    assert ".MAX_TRUSTED_BANK_VERSION(16'hffff)" in source
    assert "version_s4 > 16'd7" not in CORE.read_text(encoding="utf-8")


def test_management_snapshot_crc_is_byte_serial() -> None:
    source = TOP.read_text(encoding="utf-8")
    assert "crc16_144" not in source
    assert "management_snapshot_shift <= management_snapshot_shift >> 8" in source
    assert "crc16_byte(management_snapshot_crc, management_snapshot_octet)" in source


def test_crc32_word_packing_matches_zlib_for_boundaries() -> None:
    words = [0, 1, (1 << 21), (1 << 22) - 1, 0x155555]
    packed = b"".join(word.to_bytes(3, "little") for word in words)
    assert crc32_table_words(words) == zlib.crc32(packed) & 0xFFFFFFFF


def test_reference_rejects_active_bank_programming_without_mutation() -> None:
    reference = ProductionFastPathManagementReference()
    result = reference.step(
        {
            "cfg_begin_valid": 1,
            "cfg_begin_bank": 0,
            "cfg_expected_active_version": 0,
            "cfg_new_version": 1,
        }
    )
    assert result["management_reject"] == 1
    assert result["management_reject_reason"] == REJECT_ACTIVE_BANK
    assert reference.cfg_session_active == 0
    assert reference.bank0_trusted == 1


def test_fast_reference_trace_covers_full_transactions_and_five_snapshots() -> None:
    lines, expected, labels = build_management_trace()
    assert len(lines) == len(expected) == len(labels) == 1681
    assert sum(row["cfg_word_ack"] for row in expected) == 1543
    assert sum(row["management_state_valid"] for row in expected) == 5
    assert "bad_crc_finalize" in labels
    assert "drain_guard_reject" in labels


def test_frozen_cxxrtl_audit_passes_every_gate() -> None:
    report = _report()
    assert report["status"] == "PASS"
    assert report["verdict"] == VERDICT
    assert report["mismatch_count"] == 0
    assert report["cycle_rows"] == 1681
    assert report["gate_summary"]["passed"] == report["gate_summary"]["total"]
    assert all(report["gates"].values())


def test_frozen_audit_exercises_every_rejection_reason_and_rejects_mutations() -> None:
    report = _report()
    assert report["observed_reject_reasons"] == list(range(1, 12))
    assert len(report["mutation_audit"]) == 7
    assert all(row["rejected"] for row in report["mutation_audit"])


def test_cxxrtl_source_data_is_complete_exact_and_snapshot_crc_valid() -> None:
    rows = list(csv.DictReader(DEFAULT_CSV.open(encoding="utf-8", newline="")))
    assert len(rows) == 1681
    assert all(row["exact"] == "True" for row in rows)
    valid_rows = [row for row in rows if row["actual_management_state_valid"] == "1"]
    assert len(valid_rows) == 5
    for row in valid_rows:
        word = int(row["actual_management_state_word"], 16)
        payload = word & ((1 << 144) - 1)
        assert word >> 144 == crc16_ccitt_little_endian(payload, 18)


def test_audit_does_not_upgrade_evidence_to_transport_or_board_measurement() -> None:
    boundary = _report()["evidence_boundary"]
    assert boundary["synthesizable_rtl"] is True
    assert boundary["cycle_accurate_cxxrtl"] is True
    assert boundary["transport_or_cdc_validated"] is False
    assert boundary["target_place_route"] is False
    assert boundary["board_measured"] is False
    assert boundary["crc32_is_integrity_not_authentication"] is True

