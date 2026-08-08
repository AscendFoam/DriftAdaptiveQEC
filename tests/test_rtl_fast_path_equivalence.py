from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cnn_fpga.benchmark.rtl_fast_path_equivalence import (
    CORE,
    DEFAULT_JSON,
    DRIVER,
    MEMORY_MANIFEST,
    VERDICT,
    build_exhaustive_trace,
    build_fault_trace,
    compare_trace,
    discover_tools,
    mutation_audit,
)
from cnn_fpga.benchmark.bit_accurate_hardware_reference import load_frozen_images


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


def test_rtl_sources_exist_and_are_not_empty() -> None:
    assert CORE.stat().st_size > 20_000
    assert DRIVER.stat().st_size > 2_000


def test_core_has_full_pipeline_crc_fsm_and_bank_contract() -> None:
    source = CORE.read_text(encoding="utf-8")
    for token in (
        "crc16_42",
        "crc16_102",
        "crc16_216",
        "round_shift3_ties_even",
        "selected_commit",
        "MODE_RESET_REQUEST",
        "event_fault_mask",
        "pending_output_payload",
        "map_valid_debug <= v4",
    ):
        assert token in source


def test_rtl_uses_eight_mirrored_memories_for_legal_1r1w_ports() -> None:
    source = CORE.read_text(encoding="utf-8")
    assert source.count('ram_style = "block"') == 8
    assert source.count("$readmemh") == 8
    assert "2R+1W" in source


def test_no_online_real_or_division_operator_in_core() -> None:
    source = CORE.read_text(encoding="utf-8")
    assert " real " not in source
    assert "$ln" not in source and "$exp" not in source
    code = "\n".join(line.split("//", 1)[0] for line in source.splitlines())
    code_without_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    assert re.search(r"(?<![*/])/(?![/*])", code_without_strings) is None


def test_memory_manifest_matches_frozen_images() -> None:
    manifest = json.loads(MEMORY_MANIFEST.read_text(encoding="utf-8"))
    images = load_frozen_images()
    assert len(manifest["files"]) == 4
    for row in manifest["files"]:
        values = [int(line, 16) for line in (ROOT / row["path"]).read_text(encoding="ascii").splitlines()]
        expected = [value & ((1 << 22) - 1) for value in images[row["image_version"]].table_codes[row["phase"]]]
        assert values == expected


def test_fault_trace_exercises_all_registered_fault_classes() -> None:
    lines, records = build_fault_trace()
    assert len(lines) == len(records) == 226
    masks = [int(record.state_word_hex, 16) & ((1 << 216) - 1) for record in records]
    assert any(record.input_crc_ok is False for record in records)
    assert any(record.commit_status == "deferred" for record in records)
    assert any(record.commit_status == "committed" for record in records)
    assert any(mask != 0 for mask in masks)


def test_exhaustive_trace_contains_4096_valid_map_rows() -> None:
    _, records = build_exhaustive_trace()
    assert len(records) == 4102
    assert sum(record.map_valid for record in records) == 4096


def test_compare_rejects_row_count_mismatch() -> None:
    _, expected = build_fault_trace()
    actual = [
        {
            "cycle": "0", "commit_ack": "0", "active_version": "0",
            "map_valid": "0", "map_address": "0", "map_llr_twos": "0",
            "out_word_hex": expected[0].output_word_hex,
            "state_word_hex": expected[0].state_word_hex,
            "input_valid": "1", "input_word_hex": expected[0].input_word_hex,
        }
    ]
    _, mismatches = compare_trace("short", actual, expected)
    assert mismatches[0]["kind"] == "row_count"


def test_formal_report_passes_all_gates() -> None:
    report = _report()
    assert report["status"] == "PASS"
    assert report["verdict"] == VERDICT
    assert report["gate_summary"]["passed"] == report["gate_summary"]["total"]
    assert all(report["gates"].values())


def test_formal_report_has_zero_trace_mismatches() -> None:
    report = _report()
    assert [row["mismatch_count"] for row in report["scenarios"]] == [0, 0]
    assert report["scenarios"][1]["map_valid_rows"] == 4096


def test_formal_report_rejects_every_mutation() -> None:
    report = _report()
    assert len(report["mutation_audit"]) == 8
    assert all(row["rejected"] for row in report["mutation_audit"])


def test_formal_report_keeps_hardware_claims_false() -> None:
    boundary = _report()["evidence_boundary"]
    assert boundary["synthesizable_rtl"] is True
    assert boundary["cxxrtl_simulation"] is True
    assert boundary["target_device_synthesis"] is False
    assert boundary["target_device_place_route"] is False
    assert boundary["board_measured"] is False


def test_tools_are_discoverable() -> None:
    try:
        tools = discover_tools()
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(str(exc))
    assert tools["yosys"].is_file()
    assert tools["gpp"].is_file()
    assert (tools["include"] / "cxxrtl/cxxrtl.h").is_file()
