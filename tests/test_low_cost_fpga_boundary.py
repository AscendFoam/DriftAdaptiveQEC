from __future__ import annotations

import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "low_cost_fpga_boundary.json"
DOC_PATH = ROOT / "docs" / "low_cost_fpga_boundary.md"
HIL_CONFIG_PATH = ROOT / "cnn_fpga" / "config" / "hardware_hil.yaml"
T72_GATE_PATH = (
    ROOT
    / "artifacts"
    / "t72_real_board_transfer_pack_provenance_hardening"
    / "current_host_regenerated_gate.json"
)


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_reference_target_does_not_claim_physical_board_or_price() -> None:
    contract = load_contract()
    selection = contract["selection"]
    budget = selection["budget_policy"]

    assert selection["board_model"] == "Sipeed Tang Nano 20K"
    assert selection["fpga_device"] == "GW2AR-LV18QN88C8/I7"
    assert selection["selection_status"] == "reference_target_frozen"
    assert selection["physical_unit_status"] == "not_procured_or_physically_verified"
    assert selection["evidence_level"] == "vendor_specification_only_not_project_measurement"
    assert budget["preferred_board_only_cny"] == 300
    assert budget["maximum_landed_cny"] == 350
    assert budget["current_quote_cny"] is None
    assert budget["price_claim_allowed"] is False


def test_vendor_resource_snapshot_is_complete_and_unambiguous() -> None:
    specs = load_contract()["vendor_specifications"]
    logic = specs["logic"]
    memory = specs["memory"]
    clocking = specs["clocking"]

    assert logic == {
        "lut4": 20736,
        "flip_flops": 15552,
        "ssram_bits": 41472,
        "bsram_vendor_kbits": 828,
        "bsram_blocks": 46,
        "multiplier_18x18": 48,
        "pll": 2,
        "io_banks": 8,
        "free_io": 34,
    }
    assert memory["sdr_sdram_mbits"] == 64
    assert memory["sdr_sdram_data_width_bits"] == 32
    assert memory["qspi_flash_mbits"] == 64
    assert clocking["onboard_crystal_mhz"] == 27
    assert clocking["project_bringup_clock_mhz"] == 27
    assert clocking["documented_extra_clock_outputs"] == 3


def test_interface_roles_fail_closed_before_transport_measurement() -> None:
    contract = load_contract()
    by_id = {row["id"]: row for row in contract["project_interface_freeze"]}

    assert set(by_id) == {"IF01", "IF02", "IF03", "IF04", "IF05"}
    assert "sustained replay throughput" in by_id["IF01"]["forbidden_inference"]
    assert "runtime histogram" in by_id["IF02"]["forbidden_inference"]
    assert "115200 baud" in by_id["IF03"]["forbidden_inference"]
    assert "before measurement" in by_id["IF04"]["forbidden_inference"]
    assert "analog ADC" in by_id["IF05"]["forbidden_inference"]
    assert contract["vendor_specifications"]["onboard_bridge"]["measured_application_throughput_mbps"] is None


def test_histogram_capacity_and_uart_serialization_are_recomputed() -> None:
    arithmetic = load_contract()["capacity_arithmetic_not_synthesis"]
    single_bytes = 32 * 32 * 2
    double_bits = single_bytes * 2 * 8
    bsram_bits = 828 * 1024
    uart_ms = single_bytes * 10 / 115200 * 1000
    required_bps = single_bytes * 10 / 0.020

    assert arithmetic["single_histogram_bytes"] == single_bytes
    assert arithmetic["double_histogram_bytes"] == single_bytes * 2
    assert arithmetic["double_histogram_bits"] == double_bits
    assert math.isclose(
        arithmetic["double_histogram_share_of_vendor_bsram_percent_if_kibits"],
        100 * double_bits / bsram_bits,
        rel_tol=0,
        abs_tol=5e-5,
    )
    assert math.isclose(arithmetic["uart_8n1_histogram_only_lower_bound_ms"], uart_ms, abs_tol=5e-5)
    assert arithmetic["minimum_uart_line_rate_for_histogram_only_bps_8n1"] == required_bps
    assert arithmetic["uart_8n1_histogram_only_lower_bound_ms"] > arithmetic["window_reference_period_ms"]


def test_affine_bank_capacity_uses_repository_q4_20_width() -> None:
    arithmetic = load_contract()["capacity_arithmetic_not_synthesis"]
    assert arithmetic["affine_bank_parameter_count"] == 6
    assert arithmetic["affine_parameter_bits_each"] == 25
    assert arithmetic["dual_affine_bank_payload_bits"] == 2 * 6 * 25
    assert "not synthesis" in arithmetic["interpretation"]


def test_measurement_boundary_excludes_quantum_front_end_claims() -> None:
    boundary = load_contract()["measurement_boundary"]
    allowed = " ".join(boundary["allowed_after_required_gates"]).lower()
    forbidden = " ".join(boundary["forbidden_without_external_quantum_instrumentation"]).lower()

    for token in ("core latency", "transport latency", "bit-for-bit", "fallback"):
        assert token in allowed
    for token in ("microwave", "quantum adc", "cavity", "transmon", "squeezing", "beyond-break-even"):
        assert token in forbidden
    assert boundary["canonical_claim"] == "low-cost FPGA digital control-plane reference target"


def test_current_repo_state_matches_recorded_placeholder_boundary() -> None:
    contract = load_contract()["current_repo_compatibility"]
    config = yaml.safe_load(HIL_CONFIG_PATH.read_text(encoding="utf-8"))
    gate = json.loads(T72_GATE_PATH.read_text(encoding="utf-8"))

    assert config["hil"]["backend"] == contract["observed_hil_backend"] == "mock"
    assert config["hil"]["board"] == contract["observed_hil_config_board"] == "ZCU111"
    assert config["dma"]["histogram_buffer_bytes"] == 4096
    assert contract["compatibility_with_tang_nano_20k"] == "not_directly_compatible"
    assert "UART/USB-SPI" in contract["required_future_adapter"]
    assert gate["final_gate_verdict"] == contract["current_real_board_gate"]
    assert gate["repo_execution_path_truth"]["status"] == "placeholder_only"


def test_sources_are_primary_vendor_pages_and_marked_with_access_date() -> None:
    sources = load_contract()["source_register"]
    assert {row["id"] for row in sources} == {"SRC01", "SRC02", "SRC03"}
    assert all(row["accessed"] == "2026-07-14" for row in sources)
    assert all("sipeed" in row["url"].lower() for row in sources)
    assert any(row["url"].endswith(".pdf") for row in sources)


def test_human_and_machine_contract_ids_stay_synchronized() -> None:
    contract = load_contract()
    text = DOC_PATH.read_text(encoding="utf-8")

    assert contract["schema_version"] in text
    assert contract["selection"]["board_model"] in text
    assert contract["selection"]["fpga_device"] in text
    for row in contract["project_interface_freeze"]:
        assert row["id"] in text
    assert contract["current_repo_compatibility"]["current_real_board_gate"] in text
    assert "不晋升 `claim_ladder` 的 CL2、\nCL3 或 CL4" in text
