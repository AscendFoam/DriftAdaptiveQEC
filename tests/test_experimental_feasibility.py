from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.algorithm_success_falsification import FALLBACK_BRANCH_ID
from cnn_fpga.benchmark.experimental_feasibility import (
    CONTROLLER_STRATEGIES,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    EXPECTED_COMPONENT_STATUS_COUNTS,
    EXPECTED_NONMIXING_CONTRACT,
    MISSING_EVIDENCE,
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


def _set_path(payload: dict, path: tuple[object, ...], value: object) -> None:
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


def test_committed_artifact_is_current_complete_and_source_bound() -> None:
    payload = _artifact()
    assert payload["task_id"] == "T5.1.6"
    assert payload["status"] == "PASS"
    assert payload["deployment_readiness"] == "NOT_ESTABLISHED"
    assert payload["active_algorithm_branch"] == FALLBACK_BRANCH_ID
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["gate_summary"] == {
        "passed": 21,
        "total": 21,
        "failed": [],
    }
    assert len(payload["gates"]) == 21 and all(payload["gates"].values())
    assert payload["source_data"]["row_count"] == 408
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()
    assert validate_payload(payload) == ()


def test_parent_machine_gates_files_and_implementations_are_current() -> None:
    payload = _artifact()
    parents = load_parent_artifacts()
    current = current_parent_implementation_hashes()
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
        assert parents[task_id]["implementation_sha256"] == current[task_id]


def test_controller_table_preserves_occupancy_reset_slew_and_null_fields() -> None:
    rows = _artifact()["controller_feasibility_rows"]
    assert len(rows) == 10
    assert {(row["cutoff"], row["strategy"]) for row in rows} == {
        (cutoff, strategy)
        for cutoff in (12, 16)
        for strategy in CONTROLLER_STRATEGIES
    }
    for row in rows:
        assert row["p_g"] + row["p_e"] == pytest.approx(1.0)
        assert row["expected_e_events"] == pytest.approx(20.0 * row["p_e"])
        assert row["multilevel_leakage_occupancy"] is None
        assert row["multilevel_leakage_events"] is None
        assert row["reset_events"] == 20
        assert row["expected_e_events"] != row["reset_events"]
        assert row["mean_parameter_slew_rms"] is not None
        assert row["parameter_saturation_rate"] is None
        assert row["classical_latency_us"] is None
        assert row["target_hardware_measured"] is False
        assert row["device_feasibility_status"] == "NOT_ESTABLISHED"
        assert row["peak_can_support_deployment_claim"] is False


def test_ground_occupancy_is_reported_without_promoting_it_to_leakage_evidence() -> None:
    rows = _artifact()["controller_feasibility_rows"]
    primary = {row["strategy"]: row for row in rows if row["cutoff"] == 12}
    assert primary["standard"]["p_g"] == pytest.approx(0.667578125)
    assert primary["exact_budget_mf"]["p_g"] == pytest.approx(0.82279296875)
    assert primary["fresh_gru_teacher"]["p_g"] == pytest.approx(0.86396484375)
    assert primary["handcrafted_recurrence"]["p_g"] == pytest.approx(0.9791015625)
    assert primary["distilled_student"]["p_g"] == pytest.approx(0.8642578125)
    assert _artifact()["nonmixing_contract"][
        "controller_occupancy_is_device_occupancy"
    ] is False


def test_peak_lifetimes_retain_cost_and_do_not_support_deployment_claim() -> None:
    peaks = [
        row
        for row in _artifact()["controller_feasibility_rows"]
        if row["peak_fidelity_lifetime_in_lane"]
    ]
    assert {(row["cutoff"], row["strategy"]) for row in peaks} == {
        (12, "exact_budget_mf"),
        (16, "fresh_gru_teacher"),
    }
    for peak in peaks:
        assert peak["stored_scalars"] == 72853
        assert peak["analytic_macs_per_half_cycle"] == 72266
        assert peak["parent_deployable_flag"] is False
        assert peak["classical_latency_us"] is None
        assert peak["peak_can_support_deployment_claim"] is False


def test_fault_campaign_preserves_scenario_denominators_and_burdens() -> None:
    rows = _artifact()["fault_campaign_rows"]
    assert len(rows) == 8
    by_scenario = {row["scenario"]: row for row in rows}
    assert all(row["run_count"] == 4 and row["cycles"] == 95984 for row in rows)
    assert by_scenario["burst"]["fallback_cycles"] == 260
    assert by_scenario["leakage_reset"]["fallback_cycles"] == 16
    assert by_scenario["leakage_reset"]["reset_request_cycles"] == 4
    assert by_scenario["host_timeout"]["fallback_cycles"] == 11232
    assert by_scenario["communication_pause_ack_loss"]["ack_timeout_cycles"] == 1596
    assert by_scenario["communication_pause_ack_loss"]["awaiting_readback_cycles"] == 1604
    assert by_scenario["corrupt_transfer"]["fallback_cycles"] == 32
    assert by_scenario["post_commit_guard_republish"]["fallback_cycles"] == 12
    for row in rows:
        assert row["fallback_rate"] == pytest.approx(row["fallback_cycles"] / 95984)
        assert row["reset_request_rate"] == pytest.approx(
            row["reset_request_cycles"] / 95984
        )
        assert row["unsafe_action_cycles"] == 0
        assert row["undefined_action_cycles"] == 0


def test_global_safety_rates_are_observed_campaign_rates_not_population_bounds() -> None:
    safety = _artifact()["safety_summary"]
    assert safety["campaign_cycles"] == 767872
    assert safety["scenario_count"] == 8
    assert safety["run_count"] == 32
    assert safety["fallback_cycles"] == 11552
    assert safety["fallback_rate"] == pytest.approx(11552 / 767872)
    assert safety["reset_request_cycles"] == 4
    assert safety["reset_request_rate"] == pytest.approx(4 / 767872)
    assert safety["unsafe_action_cycles"] == 0
    assert safety["observed_unsafe_action_rate"] == 0.0
    assert safety["undefined_action_cycles"] == 0
    assert safety["observed_undefined_action_rate"] == 0.0
    assert safety["statistical_population_upper_bound"] is None
    assert "not iid" in safety["upper_bound_reason"]


def test_component_fallback_and_student_fail_closed_contracts_are_preserved() -> None:
    payload = _artifact()
    component = payload["component_fallback"]
    assert component["cycles"] == 4096
    assert component["status_counts"] == EXPECTED_COMPONENT_STATUS_COUNTS
    assert component["healthy_fraction"] == pytest.approx(2050 / 4096)
    assert component["nonhealthy_fraction"] == pytest.approx(2046 / 4096)
    student = payload["student_fail_closed_contract"]
    assert student == {
        "safe_baseline": "reset state and exact zero physical residual",
        "leakage_resets_initial_state": True,
        "target_latency_cycles": None,
        "rtl_measured": False,
        "board_measured": False,
    }


def test_missing_evidence_and_nonmixing_ledgers_are_exact() -> None:
    payload = _artifact()
    assert tuple(payload["missing_evidence"]) == MISSING_EVIDENCE
    assert len(payload["missing_evidence"]) == 7
    assert all(row["status"] == "MISSING" for row in payload["missing_evidence"])
    assert payload["nonmixing_contract"] == EXPECTED_NONMIXING_CONTRACT
    assert all(value is False for value in payload["nonmixing_contract"].values())


def test_source_ledger_contains_all_row_types_and_exact_row_count() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 408
    assert {row["row_type"] for row in rows} == set(payload["source_data"]["row_types"])
    assert {row["row_type"] for row in rows} == {
        "contract_gate",
        "controller_feasibility",
        "fault_campaign",
        "file_binding",
        "missing_evidence",
        "parent_artifact",
        "parent_gate",
    }
    missing_rows = [row for row in rows if row["row_type"] == "missing_evidence"]
    assert len(missing_rows) == 7
    assert all(row["passed"] == "False" for row in missing_rows)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("controller_feasibility_rows", 0, "p_g"), 0.5),
        (("controller_feasibility_rows", 0, "multilevel_leakage_occupancy"), 0.0),
        (("controller_feasibility_rows", 0, "reset_events"), 19),
        (("controller_feasibility_rows", 0, "mean_parameter_slew_rms"), None),
        (("controller_feasibility_rows", 0, "parameter_saturation_rate"), 0.0),
        (("controller_feasibility_rows", 0, "classical_latency_us"), 0.0),
        (("controller_feasibility_rows", 0, "device_feasibility_status"), "READY"),
        (("controller_feasibility_rows", 0, "peak_can_support_deployment_claim"), True),
        (("fault_campaign_rows", 3, "fallback_cycles"), 0),
        (("fault_campaign_rows", 4, "ack_timeout_cycles"), 0),
        (("fault_campaign_rows", 2, "reset_request_cycles"), 0),
        (("fault_campaign_rows", 0, "unsafe_action_cycles"), 1),
        (("safety_summary", "fallback_cycles"), 0),
        (("safety_summary", "statistical_population_upper_bound"), 0.0),
        (("component_fallback", "status_counts", "fallback"), 1295),
        (("student_fail_closed_contract", "safe_baseline"), "zero residual only"),
        (("nonmixing_contract", "peak_lifetime_is_deployment_readiness"), True),
        (("active_algorithm_branch",), "learned_decoder"),
        (("deployment_readiness",), "READY"),
    ],
)
def test_semantic_validator_rejects_cost_hiding_and_evidence_invention(
    path: tuple[object, ...], replacement: object
) -> None:
    payload = copy.deepcopy(_artifact())
    _set_path(payload, path, replacement)
    assert validate_payload(payload)


def test_semantic_validator_rejects_missing_evidence_deletion() -> None:
    payload = copy.deepcopy(_artifact())
    payload["missing_evidence"].pop()
    assert "missing-evidence ledger changed" in validate_payload(payload)


def test_semantic_validator_rejects_wrong_peak_marker() -> None:
    payload = copy.deepcopy(_artifact())
    primary = [row for row in payload["controller_feasibility_rows"] if row["cutoff"] == 12]
    for row in primary:
        row["peak_fidelity_lifetime_in_lane"] = row["strategy"] == "standard"
    assert "marked controller peak is not the lane maximum" in validate_payload(payload)


def test_build_report_fails_closed_when_parent_is_missing() -> None:
    parents = load_parent_artifacts()
    integrity = inspect_parent_integrity(parents)
    parents.pop("T4.3.3")
    with pytest.raises(ValueError, match="missing parent artifacts"):
        build_report(parents, integrity)


def test_parent_integrity_rejects_stale_implementation_hashes() -> None:
    parents = load_parent_artifacts()
    stale = {task_id: "0" * 64 for task_id in parents}
    integrity = inspect_parent_integrity(parents, stale)
    assert all(record["machine_pass"] for record in integrity.values())
    assert all(record["implementation_current"] is False for record in integrity.values())
    assert all(record["passed"] is False for record in integrity.values())


def test_writer_is_deterministic_except_timestamp_and_output_binding(tmp_path: Path) -> None:
    first_json = tmp_path / "first.json"
    first_csv = tmp_path / "first.csv"
    second_json = tmp_path / "second.json"
    second_csv = tmp_path / "second.csv"
    first = write_artifacts(artifact_path=first_json, source_data_path=first_csv)
    second = write_artifacts(artifact_path=second_json, source_data_path=second_csv)
    assert first["status"] == second["status"] == "PASS"
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first["source_data"]["sha256"] == second["source_data"]["sha256"]
    assert validate_payload(first) == validate_payload(second) == ()
