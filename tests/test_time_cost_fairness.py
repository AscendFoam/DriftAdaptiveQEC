from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.algorithm_success_falsification import FALLBACK_BRANCH_ID
from cnn_fpga.benchmark.time_cost_fairness import (
    CONTROLLER_STRATEGIES,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    PARENT_ARTIFACTS,
    build_report,
    current_parent_implementation_hashes,
    implementation_sha256,
    inspect_parent_integrity,
    load_parent_artifacts,
    validate_payload,
    write_artifacts,
)


def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict]:
    parents = load_parent_artifacts()
    return parents, inspect_parent_integrity(parents)


def test_committed_artifact_is_current_complete_and_source_bound() -> None:
    payload = _artifact()
    assert payload["task_id"] == "T5.1.5"
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["active_algorithm_branch"] == FALLBACK_BRANCH_ID
    assert payload["gate_summary"] == {
        "passed": len(payload["gates"]),
        "total": len(payload["gates"]),
        "failed": [],
    }
    assert len(payload["gates"]) == 18 and all(payload["gates"].values())
    assert payload["source_data"]["row_count"] == 537
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()
    assert validate_payload(payload) == ()


def test_parent_machine_gates_files_and_implementations_are_current() -> None:
    payload = _artifact()
    parents = load_parent_artifacts()
    implementation_hashes = current_parent_implementation_hashes()
    assert set(payload["parent_integrity"]) == set(PARENT_ARTIFACTS)
    for task_id, path in PARENT_ARTIFACTS.items():
        record = payload["parent_integrity"][task_id]
        assert record["machine_pass"] is True
        assert record["machine_gate_count"] >= 13
        assert record["all_declared_files_current"] is True
        assert record["implementation_current"] is True
        assert record["passed"] is True
        assert all(row["passed"] for row in record["declared_file_bindings"])
        binding = next(
            row for row in payload["artifact_bindings"] if row["task_id"] == task_id
        )
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if task_id in implementation_hashes:
            assert parents[task_id]["implementation_sha256"] == implementation_hashes[task_id]


def test_protocol_lane_has_exact_cycle_time_and_common_wallclock_arithmetic() -> None:
    rows = _artifact()["protocol_wallclock_rows"]
    assert len(rows) == 12
    assert {row["lane_id"] for row in rows} == {
        f"cutoff{cutoff}_{noise}"
        for cutoff in (12, 16)
        for noise in ("high", "medium", "low")
    }
    for row in rows:
        expected_cycle = 10.0 if row["strategy"] == "measurement_feedback" else 7.0
        expected_cycles = 70 if expected_cycle == 10.0 else 100
        assert row["cycle_duration_us"] == expected_cycle
        assert row["full_cycles"] == expected_cycles
        assert row["common_horizon_us"] == 700.0
        assert row["cycle_duration_us"] * row["full_cycles"] == 700.0
        assert row["logical_z_area_lifetime_protocol_cycles"] * expected_cycle == pytest.approx(
            row["logical_z_area_lifetime_us"]
        )
        assert row["fidelity_area_lifetime_protocol_cycles"] * expected_cycle == pytest.approx(
            row["fidelity_area_lifetime_us"]
        )
        assert row["online_classical_latency_us"] is None


def test_all_six_protocol_orderings_reverse_between_cycles_and_microseconds() -> None:
    rows = _artifact()["protocol_ordering_reversal"]
    assert len(rows) == 6
    for row in rows:
        assert 1.15 < row[
            "autonomous_to_feedback_logical_lifetime_protocol_cycle_ratio"
        ] < 1.35
        assert 0.80 < row["autonomous_to_feedback_logical_lifetime_us_ratio"] < 0.95
        assert row["protocol_cycle_favors_autonomous"] is True
        assert row["wallclock_favors_autonomous"] is False
        assert row["measurement_events_avoided"] == 140.0
        assert row["additional_autonomous_resets"] == 60.0
        assert row["additional_autonomous_active_gates"] == 540.0


def test_protocol_event_counts_are_native_not_equalized() -> None:
    rows = _artifact()["protocol_wallclock_rows"]
    for row in rows:
        if row["strategy"] == "measurement_feedback":
            assert (row["measurement_events"], row["reset_events"], row["active_gate_applications"]) == (
                140,
                140,
                1260,
            )
            assert row["measurements_per_100us"] == 20.0
        else:
            assert (row["measurement_events"], row["reset_events"], row["active_gate_applications"]) == (
                0,
                200,
                1800,
            )
            assert row["measurements_per_100us"] == 0.0
        assert row["resets_per_100us"] == row["reset_events"] / 7.0
        assert row["active_gates_per_100us"] == row["active_gate_applications"] / 7.0


def test_idle_reference_is_exact_zero_cost_but_not_ranked_at_700us() -> None:
    idle = _artifact()["idle_reference"]
    assert idle["horizon_us"] == 30.0
    assert idle["full_intervals"] == 3
    assert idle["measurement_events"] == 0
    assert idle["reset_events"] == 0
    assert idle["active_gate_applications"] == 0
    assert idle["classical_latency_us"] is None
    assert idle["ranked_with_700us_protocol_lane"] is False


def test_controller_lane_reports_both_time_units_events_and_analytic_cost() -> None:
    rows = _artifact()["matched_controller_rows"]
    assert len(rows) == 10
    assert {(row["cutoff"], row["strategy"]) for row in rows} == {
        (cutoff, strategy)
        for cutoff in (12, 16)
        for strategy in CONTROLLER_STRATEGIES
    }
    for row in rows:
        assert row["full_cycles"] == 10
        assert row["cycle_duration_us"] == 10.0
        assert row["simulated_physical_time_us"] == 100.0
        assert row["fidelity_effective_lifetime_us"] == pytest.approx(
            10.0 * row["fidelity_effective_lifetime_cycles"]
        )
        assert row["logical_z_effective_lifetime_us"] == pytest.approx(
            10.0 * row["logical_z_effective_lifetime_cycles"]
        )
        assert (row["measurement_events"], row["reset_events"], row["active_gate_applications"]) == (
            20,
            20,
            180,
        )
        assert row["multilevel_leakage_events"] is None
        assert row["classical_latency_us"] is None
        assert row["target_hardware_measured"] is False


def test_e_outcome_burden_is_separate_from_reset_count() -> None:
    rows = _artifact()["matched_controller_rows"]
    assert all(row["reset_events"] == 20 for row in rows)
    assert all(0.0 < row["observed_e_events"] < 7.0 for row in rows)
    assert all(row["observed_e_events"] != row["reset_events"] for row in rows)
    standard_primary = next(
        row
        for row in rows
        if row["cutoff"] == 12 and row["strategy"] == "standard"
    )
    handcrafted_primary = next(
        row
        for row in rows
        if row["cutoff"] == 12 and row["strategy"] == "handcrafted_recurrence"
    )
    assert standard_primary["observed_e_events"] == pytest.approx(6.6484375)
    assert handcrafted_primary["observed_e_events"] == pytest.approx(0.41796875)
    assert standard_primary["reset_events"] == handcrafted_primary["reset_events"]


def test_controller_costs_preserve_teacher_student_compression_and_no_fake_latency() -> None:
    rows = _artifact()["matched_controller_rows"]
    for cutoff in (12, 16):
        teacher = next(
            row
            for row in rows
            if row["cutoff"] == cutoff and row["strategy"] == "fresh_gru_teacher"
        )
        student = next(
            row
            for row in rows
            if row["cutoff"] == cutoff and row["strategy"] == "distilled_student"
        )
        assert teacher["stored_scalars"] == 72853
        assert teacher["analytic_macs_per_half_cycle"] == 72266
        assert teacher["deployable_in_parent"] is False
        assert student["stored_scalars"] == 95
        assert student["analytic_macs_per_half_cycle"] == 87
        assert student["deployable_in_parent"] is False
        assert teacher["classical_latency_us"] is None
        assert student["classical_latency_us"] is None


def test_two_cycle_control_reference_never_enters_ten_cycle_table() -> None:
    payload = _artifact()
    assert all(
        row["strategy"] != "finite_horizon_control_oracle"
        for row in payload["matched_controller_rows"]
    )
    exclusion = payload["excluded_control_reference"]
    assert exclusion["strategy"] == "finite_horizon_control_oracle"
    assert exclusion["excluded_from_ten_cycle_lane"] is True
    assert exclusion["available_horizon_cycles"] == 2
    assert "cannot" in exclusion["reason"]


def test_host_latency_rows_are_estimator_profiles_not_physical_or_board_results() -> None:
    rows = _artifact()["slow_loop_host_latency_rows"]
    assert len(rows) == 6
    assert {row["family"] for row in rows} == {
        "causal_tcn",
        "small_gru",
        "gaussian_hmm",
        "diagonal_kalman",
        "exponential_recurrence",
        "run_length_fsm",
    }
    for row in rows:
        assert row["update_period_cycles"] == 32
        assert 0.0 < row["host_batch_median_us_per_update"] < 5000.0
        assert row["latency_to_ceiling_fraction"] == pytest.approx(
            row["host_batch_median_us_per_update"] / 5000.0
        )
        assert row["physical_lifetime_cycles"] is None
        assert row["physical_lifetime_us"] is None
        assert row["measurement_events"] is None
        assert row["reset_events"] is None
        assert row["active_gate_applications"] is None
        assert row["target_hardware_measured"] is False


def test_configuration_assumptions_and_all_target_latency_nulls_are_preserved() -> None:
    latency = _artifact()["latency_contract"]
    assert latency["board_measurement_status"] == "not_started"
    software = latency["configured_software_latency_model"]
    assert software["fast_path_mean_us"] == 1.0
    assert software["slow_path_mean_total_us"] == 995.0
    assert software["evidence_class"] == "project_configuration_assumption"
    assert software["measured_on_target_board"] is False
    target_rows = latency["target_board_latency"] + latency["physical_frontend"]
    assert len(target_rows) == 7
    assert all(row["value_us"] is None for row in target_rows)
    assert all(row["measured_on_target_board"] is False for row in target_rows)
    assert latency["cross_lane_aggregate_latency_us"] is None
    assert latency["cross_lane_comparison_status"] == "forbidden_noncomparable_lanes"


def test_source_ledger_covers_every_lane_and_null_field() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == payload["source_data"]["row_count"] == 537
    assert {row["row_type"] for row in rows} == {
        "contract_gate",
        "controller_metric",
        "file_binding",
        "host_estimator_metric",
        "ordering_audit",
        "parent_artifact",
        "parent_gate",
        "protocol_metric",
        "target_latency_null",
    }
    assert sum(row["row_type"] == "protocol_metric" for row in rows) == 12 * 14
    assert sum(row["row_type"] == "controller_metric" for row in rows) == 10 * 14
    assert sum(row["row_type"] == "host_estimator_metric" for row in rows) == 6 * 7
    assert sum(row["row_type"] == "target_latency_null" for row in rows) == 7


@pytest.mark.parametrize(
    "mutation, expected_error",
    (
        ("drop_protocol_us", "cycle/us lifetime pair"),
        ("rescale_autonomous_cycle", "cycle arithmetic"),
        ("controller_latency_zero", "unmeasured controller latency"),
        ("e_as_reset", "physical event count"),
        ("include_control_oracle", "control reference entered"),
        ("slow_physical_lifetime", "estimator host profile"),
        ("target_core_latency", "null fields"),
        ("cross_lane_latency", "aggregate latency"),
        ("branch_rewrite", "fallback branch"),
    ),
)
def test_mutations_cannot_hide_cost_or_mix_latency_lanes(
    mutation: str, expected_error: str
) -> None:
    payload = copy.deepcopy(_artifact())
    if mutation == "drop_protocol_us":
        payload["protocol_wallclock_rows"][0]["logical_z_area_lifetime_us"] = None
    elif mutation == "rescale_autonomous_cycle":
        row = next(
            row
            for row in payload["protocol_wallclock_rows"]
            if row["strategy"] == "autonomous"
        )
        row["cycle_duration_us"] = 10.0
    elif mutation == "controller_latency_zero":
        payload["matched_controller_rows"][0]["classical_latency_us"] = 0.0
    elif mutation == "e_as_reset":
        row = payload["matched_controller_rows"][0]
        row["reset_events"] = row["observed_e_events"]
    elif mutation == "include_control_oracle":
        row = copy.deepcopy(payload["matched_controller_rows"][0])
        row["strategy"] = "finite_horizon_control_oracle"
        payload["matched_controller_rows"][0] = row
    elif mutation == "slow_physical_lifetime":
        payload["slow_loop_host_latency_rows"][0]["physical_lifetime_us"] = 42.0
    elif mutation == "target_core_latency":
        payload["latency_contract"]["target_board_latency"][0]["value_us"] = 0.25
    elif mutation == "cross_lane_latency":
        payload["latency_contract"]["cross_lane_aggregate_latency_us"] = 12.0
    elif mutation == "branch_rewrite":
        payload["active_algorithm_branch"] = "matched_learned_decoder_performance"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    assert any(expected_error in error for error in validate_payload(payload))


def test_stale_parent_fails_contract_without_changing_lane_data() -> None:
    parents, integrity = _inputs()
    integrity = copy.deepcopy(integrity)
    integrity["T3.2.8"]["passed"] = False
    result = build_report(parents, integrity)
    assert result["status"] == "FAIL"
    assert result["gates"]["all_parent_artifacts_are_passed_and_current"] is False
    assert len(result["protocol_wallclock_rows"]) == 12


def test_repeated_build_and_writer_preserve_contract(tmp_path: Path) -> None:
    parents, integrity = _inputs()
    first = build_report(copy.deepcopy(parents), copy.deepcopy(integrity))
    second = build_report(copy.deepcopy(parents), copy.deepcopy(integrity))
    assert first["contract_sha256"] == second["contract_sha256"]
    artifact = tmp_path / "fairness.json"
    source = tmp_path / "fairness.csv"
    written = write_artifacts(artifact_path=artifact, source_data_path=source)
    assert written["contract_sha256"] == first["contract_sha256"]
    assert written["source_data"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert json.loads(artifact.read_text(encoding="utf-8"))["status"] == "PASS"


def test_missing_parent_is_rejected() -> None:
    parents, integrity = _inputs()
    parents.pop("T3.2.8")
    with pytest.raises(ValueError, match="missing parent artifacts"):
        build_report(parents, integrity)
