from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.runtime.dual_latency_budget import (
    BudgetValidationError,
    REQUIRED_GATES,
    audit_budget,
    load_budget,
    validate_budget,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "dual_latency_budget.json"
REPORT = ROOT / "docs" / "dual_latency_budget.md"
VALIDATION = ROOT / "docs" / "t2_4_1_dual_latency_budget_validation.json"


def _payload() -> dict:
    return load_budget(ARTIFACT)


def _system(payload: dict, system_id: str) -> dict:
    return next(
        item
        for item in payload["lanes"]["literature_system"]["systems"]
        if item["system_id"] == system_id
    )


def _budget(system: dict, budget_id: str) -> dict:
    return next(item for item in system["budgets"] if item["budget_id"] == budget_id)


def test_checked_artifact_passes_all_executable_gates() -> None:
    result = validate_budget(artifact_path=ARTIFACT, root=ROOT)
    assert result["status"] == "PASS"
    assert result["gate_count"] == result["passed_gate_count"] == 23
    assert tuple(result["gates"]) == REQUIRED_GATES


def test_validation_snapshot_is_passed_and_hash_bound_to_current_artifact() -> None:
    snapshot = json.loads(VALIDATION.read_text(encoding="utf-8"))
    assert snapshot["audit_schema_version"] == "dual-latency-budget-audit-v1"
    assert snapshot["status"] == "PASS"
    assert snapshot["gate_count"] == snapshot["passed_gate_count"] == 23
    assert all(snapshot["gates"].values())
    assert snapshot["artifact_path"] == "docs/dual_latency_budget.json"
    assert snapshot["artifact_sha256"] == hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()


def test_artifact_has_exactly_two_noncomposable_lanes() -> None:
    lanes = _payload()["lanes"]
    assert set(lanes) == {"literature_system", "project_control_plane"}
    assert all(lane["composable_with_other_lane"] is False for lane in lanes.values())


def test_sivak_measurement_preserves_stages_without_fake_sum() -> None:
    measurement = _budget(_system(_payload(), "LIT-SIVAK-2023"), "LIT-SIVAK-MEASUREMENT")
    durations = {stage["stage_id"]: stage["duration_ns"] for stage in measurement["stages"]}
    assert list(durations.values()) == [700, 300, 1400, 332, 100]
    assert measurement["aggregate_duration_ns"] is None
    assert measurement["aggregation_status"] == "not_defined_due_to_overlap_semantics"


def test_sivak_scope_discrepancies_and_cycle_arithmetic_are_explicit() -> None:
    sivak = _system(_payload(), "LIT-SIVAK-2023")
    sbs = _budget(sivak, "LIT-SIVAK-SBS")
    reset = _budget(sivak, "LIT-SIVAK-RESET")
    cycle = _budget(sivak, "LIT-SIVAK-CONSTITUENT")
    assert (sbs["prose_sbs_ns"], sum(sbs["table_layer_durations_ns"])) == (1546, 1548)
    assert sbs["table_sbs_block_ns"] == 1596
    assert (reset["prose_subroutine_ns"], sum(reset["table_block_durations_ns"])) == (2332, 2380)
    assert sum(stage["duration_ns"] for stage in cycle["stages"]) == 4924
    assert cycle["full_xz_cycle_ns"] == 9848


def test_puviani_timing_is_model_assumption_and_sums_exactly() -> None:
    system = _system(_payload(), "LIT-PUVIANI-2025-MODEL")
    budget = _budget(system, "LIT-PUVIANI-MODEL-CYCLE")
    assert system["evidence_class"] == "external_model_assumption"
    assert sum(stage["duration_us"] for stage in budget["half_cycle_stages"]) == pytest.approx(5.0)
    assert budget["full_cycle_us"] == 2 * budget["half_cycle_us"] == 10.0
    assert "reset is numerical" in budget["scope_note"]


def test_project_cadence_and_slow_model_arithmetic_are_separate() -> None:
    project = _payload()["lanes"]["project_control_plane"]
    cadence = project["cadence"]
    model = project["software_latency_model"]
    assert cadence["window_content_duration_ms"] == pytest.approx(2048 * 5.0 / 1000)
    assert cadence["window_emission_interval_ms"] == pytest.approx(4000 * 5.0 / 1000)
    assert sum(stage["mean_us"] for stage in model["slow_path_stages"]) == 995.0
    assert model["fast_path"]["mean_us"] == 1.0
    assert cadence["fast_action_budget_us"] == 1.5


def test_uart_bounds_cover_target_and_current_software_payloads() -> None:
    project = _payload()["lanes"]["project_control_plane"]
    uart = next(item for item in project["transport"] if item["transport_id"] == "PROJECT-UART-8N1")
    assert uart["raw_uint16_histogram"]["serialization_lower_bound_ms"] == pytest.approx(
        1000 * 2048 * 10 / 115200, abs=1e-4
    )
    assert uart["software_float32_payload"]["serialization_lower_bound_ms"] == pytest.approx(
        1000 * 4096 * 10 / 115200, abs=1e-4
    )
    assert uart["raw_payload_meets_window_deadline"] is False
    assert uart["minimum_raw_line_rate_bps"] == 1_024_000


def test_all_target_board_and_physical_frontend_latency_fields_are_null() -> None:
    project = _payload()["lanes"]["project_control_plane"]
    fields = project["target_board_latency"] + project["physical_frontend"]
    assert fields
    assert all(field["value_us"] is None for field in fields)
    assert all(field["measured_on_target_board"] is False for field in fields)
    names = {field["name"] for field in fields}
    assert {
        "on_chip_core_latency",
        "measured_transport_latency",
        "end_to_end_digital_replay_latency",
        "quantum_measurement_latency",
        "high_speed_quantum_adc_acquisition_latency",
        "awg_or_dac_waveform_output_latency",
        "physical_action_latency",
    }.issubset(names)


def test_source_anchors_are_live_not_just_path_strings() -> None:
    result = audit_budget(artifact_path=ARTIFACT, root=ROOT)
    assert result["gates"]["source_anchors_resolve"] is True
    payload = _payload()
    anchors = _system(payload, "LIT-SIVAK-2023")["source_anchors"]
    assert any(anchor["line_start"] == 903 for anchor in anchors)


@pytest.mark.parametrize(
    ("mutator", "gate_name"),
    [
        (
            lambda payload: payload["cross_lane_comparison"].update({"ratio": 2.0}),
            "no_cross_lane_aggregate",
        ),
        (
            lambda payload: _system(payload, "LIT-SIVAK-2023").update(
                {"measured_on_target_board": True}
            ),
            "external_facts_not_target_measurements",
        ),
        (
            lambda payload: _budget(
                _system(payload, "LIT-SIVAK-2023"), "LIT-SIVAK-MEASUREMENT"
            ).update({"aggregate_duration_ns": 2832}),
            "sivak_measurement_aggregate_fail_closed",
        ),
        (
            lambda payload: payload["lanes"]["project_control_plane"]["cadence"].update(
                {"fast_cycle_period_us": 10.0}
            ),
            "project_config_bindings",
        ),
        (
            lambda payload: payload["lanes"]["project_control_plane"]["target_board_latency"][
                0
            ].update({"value_us": 0.332, "measured_on_target_board": True}),
            "target_board_latency_fields_null",
        ),
    ],
)
def test_critical_conflation_mutations_fail_closed(mutator, gate_name: str) -> None:
    payload = _payload()
    mutator(payload)
    result = audit_budget(payload, root=ROOT)
    assert result["status"] == "FAIL"
    assert result["gates"][gate_name] is False
    with pytest.raises(BudgetValidationError):
        validate_budget(payload, root=ROOT)


def test_unknown_evidence_class_fails_closed() -> None:
    payload = _payload()
    payload["lanes"]["project_control_plane"]["cadence"]["evidence_class"] = "measured_fpga"
    result = audit_budget(payload, root=ROOT)
    assert result["gates"]["evidence_classes_closed"] is False


def test_duplicate_nested_ids_fail_closed() -> None:
    payload = _payload()
    payload["lanes"]["project_control_plane"]["target_board_latency"][1]["field_id"] = (
        "PROJECT-TARGET-CORE-LATENCY"
    )
    result = audit_budget(payload, root=ROOT)
    assert result["gates"]["record_ids_unique"] is False


def test_source_fragment_drift_fails_closed() -> None:
    payload = _payload()
    _system(payload, "LIT-SIVAK-2023")["source_anchors"][0]["expected_fragment"] = (
        "THIS FRAGMENT DOES NOT EXIST"
    )
    result = audit_budget(payload, root=ROOT)
    assert result["gates"]["source_anchors_resolve"] is False


def test_declared_gate_list_cannot_omit_executable_gate() -> None:
    payload = _payload()
    payload["audit_contract"]["required_gates"].pop()
    result = audit_budget(payload, root=ROOT)
    assert result["status"] == "FAIL"
    assert any("required_gates" in failure for failure in result["failures"])


def test_public_runtime_lazy_exports_resolve() -> None:
    import cnn_fpga.runtime as runtime

    assert runtime.load_budget is load_budget
    assert runtime.audit_budget is audit_budget
    assert runtime.validate_budget is validate_budget
    assert runtime.BudgetValidationError is BudgetValidationError


def test_markdown_reports_every_machine_gate_and_claim_boundary() -> None:
    text = REPORT.read_text(encoding="utf-8")
    for phrase in (
        "不可相加、不可相减、不可求比",
        "700 ns",
        "332 ns",
        "4924 ns",
        "9848 ns",
        "995 us",
        "177.7778",
        "355.5556",
        "NO_GO_REAL_BOARD_HOST_OR_DEVICE_PATH_UNAVAILABLE",
        "ADC",
        "AWG/DAC",
        "physical action",
        "23 个 gate",
    ):
        assert phrase in text


def test_json_round_trip_is_stable_utf8() -> None:
    payload = _payload()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded == payload
