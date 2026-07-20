from __future__ import annotations

import copy
import csv
import json
import statistics
from pathlib import Path

import pytest

from cnn_fpga.benchmark.target_device_synthesis import (
    CORE,
    CORE_LATENCY_CYCLES,
    CST,
    DEFAULT_CSV,
    DEFAULT_JSON,
    DEVICE,
    FAMILY,
    INITIATION_INTERVAL_CYCLES,
    SDC,
    SEEDS,
    TARGET_MHZ,
    TOP,
    VERDICT,
    _read_tool_text,
    discover_tools,
    evaluate_gates,
)


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))


def test_source_and_constraints_exist_and_are_nontrivial() -> None:
    assert CORE.stat().st_size > 20_000
    assert TOP.stat().st_size > 4_000
    assert "create_clock" in SDC.read_text(encoding="utf-8")
    assert CST.read_text(encoding="utf-8").count("IO_LOC") == 18


def test_formal_report_passes_every_gate() -> None:
    report = _report()
    assert report["status"] == "PASS"
    assert report["verdict"] == VERDICT
    assert report["gate_summary"]["passed"] == report["gate_summary"]["total"] == 12
    assert all(report["gates"].values())
    assert all(evaluate_gates(report).values())


def test_target_contract_is_exact_and_not_a_generic_family_proxy() -> None:
    target = _report()["target_contract"]
    assert target["device"] == DEVICE == "GW2AR-LV18QN88C8/I7"
    assert target["family"] == FAMILY == "GW2A-18C"
    assert target["target_mhz"] == TARGET_MHZ == 27.0
    assert target["constraints"] == {
        "sdc": "cnn_fpga/rtl/tang_nano_20k_27mhz.sdc",
        "cst": "cnn_fpga/rtl/tang_nano_20k_synth_harness.cst",
    }


def test_yosys_synthesis_preserves_full_state_memories_and_dsps() -> None:
    synthesis = _report()["synthesis"]
    cells = synthesis["cell_counts"]
    assert synthesis["zero_structural_problems"] is True
    assert cells["SDPX9B"] == 8
    assert cells["MULT18X18"] == cells["MULT9X9"] == 1
    assert synthesis["register_count"] == 865
    assert synthesis["lut1_to_lut4_count"] == 0
    assert synthesis["lut_count_scope"] == "pre_abc9_log_unavailable_use_post_route_utilization"
    assert min(row["utilization"]["LUT4"]["used"] for row in _report()["place_route"]) > 3000
    assert len(synthesis["warnings"]) == 1
    assert "fault_counts" in synthesis["warnings"][0]


def test_three_independent_seeds_all_pass_27mhz() -> None:
    routes = _report()["place_route"]
    assert [row["seed"] for row in routes] == list(SEEDS) == [1, 7, 19]
    assert all(row["route_status"] == "PASS" for row in routes)
    assert all(row["timing_pass"] for row in routes)
    assert min(row["achieved_fmax_mhz"] for row in routes) > TARGET_MHZ


def test_fmax_summary_uses_all_seeds_without_best_seed_selection() -> None:
    report = _report()
    values = [row["achieved_fmax_mhz"] for row in report["place_route"]]
    summary = report["summary"]["fmax_mhz"]
    # Exact Fmax values legitimately change when the source-bound RTL is
    # regenerated.  The invariant is an all-seed summary and target margin,
    # not a stale router placement fingerprint.
    assert summary["minimum"] == min(values)
    assert summary["median"] == statistics.median(values)
    assert summary["maximum"] == max(values)
    assert summary["minimum"] > TARGET_MHZ
    assert report["summary"]["target_margin_mhz_worst_seed"] == pytest.approx(min(values) - 27.0)


def test_critical_path_decomposition_is_complete_for_every_seed() -> None:
    for row in _report()["place_route"]:
        critical = row["critical_path"]
        reconstructed = sum(
            critical[name]
            for name in ("clock_to_q_ns", "logic_ns", "routing_ns", "setup_ns")
        )
        assert reconstructed == pytest.approx(critical["period_ns"], abs=1e-9)
        assert critical["segment_count"] >= 70
        assert critical["start_cell"].startswith("core.")
        assert critical["end_cell"].startswith("fold5")


def test_worst_critical_path_is_not_hidden() -> None:
    routes = _report()["place_route"]
    worst = max(routes, key=lambda row: row["critical_path"]["period_ns"])
    assert worst["seed"] in SEEDS
    assert worst["critical_path"]["period_ns"] == max(
        row["critical_path"]["period_ns"] for row in routes
    )
    assert worst["critical_path"]["routing_ns"] > 0
    assert worst["critical_path"]["logic_ns"] > 0


def test_resources_are_reported_with_device_capacities() -> None:
    expected = {"BSRAM": 8, "DFF": 865, "MULT18X18": 1, "MULT9X9": 1, "IOB": 18, "ALU": 340}
    for route in _report()["place_route"]:
        for name, used in expected.items():
            assert route["utilization"][name]["used"] == used
            assert used <= route["utilization"][name]["available"]
        assert route["utilization"]["LUT4"]["used"] > 3000
        assert route["utilization"]["LUT4"]["used"] <= route["utilization"]["LUT4"]["available"]


def test_latency_is_cycle_based_and_recomputed_at_both_clocks() -> None:
    report = _report()
    latency = report["latency_estimate"]
    worst_fmax = report["summary"]["fmax_mhz"]["minimum"]
    assert latency["core_cycles"] == CORE_LATENCY_CYCLES == 6
    assert latency["initiation_interval_cycles"] == INITIATION_INTERVAL_CYCLES == 1
    assert latency["at_target_27mhz_ns"] == pytest.approx(6 * 1000 / 27)
    assert latency["at_worst_seed_fmax_ns"] == pytest.approx(6 * 1000 / worst_fmax)
    assert latency["excludes_harness_transport_adc_cdc_and_physical_actuation"] is True


def test_parent_equivalence_is_exact_and_source_bound() -> None:
    parent = _report()["parent_equivalence"]
    assert parent["status"] == "PASS"
    assert parent["mismatch_count"] == 0
    assert parent["map_valid_rows"] == 4316
    path = ROOT / parent["artifact"]["path"]
    assert path.stat().st_size == parent["artifact"]["bytes"]


def test_source_and_durable_artifact_hashes_are_live() -> None:
    report = _report()
    assert len(report["source_bindings"]) == 10
    assert len(report["durable_artifacts"]) == 7
    for row in report["source_bindings"] + report["durable_artifacts"]:
        path = ROOT / row["path"]
        assert path.is_file()
        assert path.stat().st_size == row["bytes"]


def test_raw_nextpnr_reports_retain_detailed_timing() -> None:
    for seed in SEEDS:
        path = ROOT / f"docs/t5_5_2_nextpnr_seed{seed:02d}_report.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert list(raw["fmax"]) == ["core.clk"]
        assert len(raw["critical_paths"]) == 1
        assert len(raw["critical_paths"][0]["path"]) >= 70
        assert raw["utilization"]["BSRAM"]["used"] == 8


def test_power_shell_utf16_log_is_decoded_without_silent_loss(tmp_path: Path) -> None:
    path = tmp_path / "native.log"
    path.write_text("Info: Program finished normally.\n", encoding="utf-16")
    assert _read_tool_text(path).splitlines() == ["Info: Program finished normally."]


def test_every_shortcut_mutation_is_rejected() -> None:
    mutations = _report()["mutation_audit"]
    assert len(mutations) == 9
    assert all(row["rejected"] and row["failed_gates"] for row in mutations)


def test_gate_recomputation_rejects_best_seed_only_report() -> None:
    report = copy.deepcopy(_report())
    report["place_route"] = [max(report["place_route"], key=lambda row: row["achieved_fmax_mhz"])]
    gates = evaluate_gates(report)
    assert gates["three_independent_place_route_seeds_pass_27mhz"] is False
    assert gates["min_median_max_fmax_are_recomputed_from_all_seeds"] is False


def test_evidence_boundary_does_not_upgrade_board_or_vendor_signoff() -> None:
    boundary = _report()["evidence_boundary"]
    assert boundary["target_device_synthesis"] is True
    assert boundary["target_device_place_route"] is True
    for name in (
        "bitstream_generated", "vendor_timing_signoff", "board_measured",
        "transport_implemented", "power_measured", "quantum_hardware_measured",
    ):
        assert boundary[name] is False


def test_source_data_row_count_and_gate_rows_are_consistent() -> None:
    report = _report()
    with DEFAULT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"]
    gate_rows = [row for row in rows if row["section"] == "gate"]
    assert len(gate_rows) == report["gate_summary"]["total"]
    assert all(row["value"] == "1" for row in gate_rows)


def test_fixed_toolchain_is_discoverable() -> None:
    try:
        tools = discover_tools()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert tools["yosys"].is_file()
    assert tools["nextpnr"].is_file()


def test_small_pin_harness_configuration_address_is_exhaustively_in_range() -> None:
    source = TOP.read_text(encoding="utf-8")
    assert "wire [8:0] cfg_safe_address" in source
    assert ".cfg_address(cfg_safe_address)" in source
    assert ".cfg_address(lfsr[8:0])" not in source
    reduced = [value if value <= 256 else value - 257 for value in range(512)]
    assert min(reduced) == 0
    assert max(reduced) == 256
    assert set(reduced) == set(range(257))
