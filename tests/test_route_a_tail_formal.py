from __future__ import annotations

import csv
import inspect
import json

import numpy as np

from cnn_fpga.benchmark.route_a_preregistration import (
    ABRUPT_OOD_FAMILIES,
    NOMINAL_FAMILY,
)
from cnn_fpga.benchmark.route_a_tail_formal import (
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    METHODS,
    PRIMARY_BASELINE,
    _run_trajectory,
    verify_report,
)


def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def test_committed_tail_artifact_recomputes_and_is_source_bound() -> None:
    payload = _artifact()
    verify_report(payload)
    assert payload["status"] == "PASS"
    assert payload["gate_summary"]["passed"] == 11
    assert payload["gate_summary"]["failed"] == 0
    assert payload["source_data_binding"]["row_count"] == 686_104
    assert payload["source_data_binding"]["expected_row_count"] == 686_104


def test_exact_untouched_24_seed_by_37_cell_matrix_and_no_reselection() -> None:
    payload = _artifact()
    rows = payload["trajectory_results"]
    assert payload["primary_baseline"] == PRIMARY_BASELINE == "ewma_adaptive_map"
    assert payload["formal_baseline_reselection"] is False
    assert len(payload["formal_design"]["seeds"]) == 24
    assert len(payload["formal_design"]["cells"]) == 37
    assert len(rows) == 888
    assert len({(row["seed"], row["cell_id"]) for row in rows}) == 888
    assert {row["family"] for row in rows} == set(
        (*ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY)
    )
    assert all(
        sum(row["family"] == family for row in rows) == 144
        for family in ABRUPT_OOD_FAMILIES
    )
    assert sum(row["family"] == NOMINAL_FAMILY for row in rows) == 24
    assert all(
        set(row["method_window_pauli_counts_class_order_I_Z_X_Y"])
        == set(METHODS)
        for row in rows
    )


def test_online_tail_router_does_not_consume_truth_values() -> None:
    source = inspect.getsource(_run_trajectory)
    online_prefix = source[: source.index("scored_truth =")]
    assert "truth[" not in online_prefix
    assert "local_truth" not in online_prefix
    assert 'decisions["proposed_route_a"]' in online_prefix
    assert "trajectory.labels[boundary_update]" in online_prefix
    # The label is copied only into an evaluation commit ledger after the
    # selected_expert decision has already been made; it never feeds the branch.
    selected_position = online_prefix.index("selected_expert =")
    label_position = online_prefix.index("trajectory.labels[boundary_update]")
    assert selected_position < label_position


def test_all_preregistered_tail_and_nominal_gates_pass_without_hiding_equality() -> None:
    payload = _artifact()
    analysis = payload["analysis"]
    assert analysis["tail_safety_gate_passes"] is True
    assert all(analysis["promotion_components"].values())
    assert len(analysis["family_paired_safety"]) == 6
    for row in analysis["family_paired_safety"]:
        assert row["passes_all_catastrophic_gates"] is True
        assert all(row["catastrophic_gate_components"].values())
    calibration = analysis["calibration_shift_strict_gate"]
    assert calibration["baseline_global_worst_error_count"] == 181
    assert calibration["proposed_global_worst_error_count"] == 181
    assert calibration["passes"] is True
    nominal = analysis["nominal_noninferiority_gate"]
    assert nominal["passes"] is True
    assert all(nominal["components"].values())


def test_tail_gate_is_noninferiority_not_static_or_adaptive_advantage() -> None:
    payload = _artifact()
    analysis = payload["analysis"]
    by_key = {
        (row["family"], row["method_id"]): row
        for row in analysis["family_method_summaries"]
    }
    # Step and telegraph are materially better under static MAP in the formal
    # data, even though the pilot-locked EWMA non-inferiority contrast passes.
    for family in ("step_calibration_shift", "telegraph_drift"):
        assert by_key[(family, "proposed_route_a")]["average_ler"] > by_key[
            (family, "static_joint_map")
        ]["average_ler"]
    assert by_key[("step_calibration_shift", "proposed_route_a")][
        "global_worst_window_error_count"
    ] == 181
    assert by_key[("step_calibration_shift", "static_joint_map")][
        "global_worst_window_error_count"
    ] == 32
    # Five families are exactly equal to EWMA in every registered safety
    # contrast; burst differs only by paired cancellations and one-window +1.
    exact_equal = 0
    for row in analysis["family_paired_safety"]:
        if (
            row["average_proposed_minus_baseline"]["estimate"] == 0.0
            and row["p95_proposed_minus_baseline"]["estimate"] == 0.0
            and row["seed_worst_proposed_minus_baseline"]["estimate"] == 0.0
            and row["max_single_window_excess_error_count"] == 0
        ):
            exact_equal += 1
    assert exact_equal == 5


def test_fallback_false_update_and_recovery_lag_are_not_silently_dropped() -> None:
    payload = _artifact()
    by_family = {
        row["family"]: row
        for row in payload["analysis"]["action_metrics_by_family"]
    }
    assert by_family[NOMINAL_FAMILY]["fallback_rate"] < 0.01
    assert by_family[NOMINAL_FAMILY]["unnecessary_fallback_rate"] < 0.0075
    for family in ABRUPT_OOD_FAMILIES:
        row = by_family[family]
        assert row["fallback_rate"] > 0.50
        assert row["unnecessary_fallback_rate"] > 0.50
        assert row["false_updates"] > 2_000
        assert row["commits"] == 3_456
        onset = row["events"]["tail_onset_to_fallback"]
        assert onset["right_censored"] == 0
    for family in ABRUPT_OOD_FAMILIES[1:]:
        recovery = by_family[family]["events"]["tail_recovery_to_open"]
        assert recovery["events"] > 0
        assert recovery["right_censored"] == 0
        assert recovery["p95_higher_decisions"] >= 256


def test_source_data_reconstructs_window_and_paired_action_totals() -> None:
    payload = _artifact()
    method_errors = {PRIMARY_BASELINE: 0, "proposed_route_a": 0}
    paired = np.zeros(4, dtype=np.int64)
    fallback = 0
    unnecessary = 0
    rows = 0
    with DEFAULT_SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            if row["row_type"] == "formal_window" and row["method_id"] in method_errors:
                method_errors[row["method_id"]] += sum(
                    int(row[key]) for key in ("n_X", "n_Y", "n_Z")
                )
            elif row["row_type"] == "paired_outcome_window":
                paired += np.asarray(
                    [int(row[key]) for key in ("n_I", "n_X", "n_Y", "n_Z")]
                )
            elif row["row_type"] == "action_window":
                fallback += int(row["n_I"])
                unnecessary += int(row["n_X"])
    assert rows == 686_104
    raw_rows = payload["trajectory_results"]
    expected_errors = {method: 0 for method in method_errors}
    for row in raw_rows:
        for method in expected_errors:
            counts = np.asarray(
                row["method_window_pauli_counts_class_order_I_Z_X_Y"][method]
            )
            expected_errors[method] += int(np.sum(counts[:, 1:]))
    assert method_errors == expected_errors
    assert int(paired[1]) == sum(
        row["avoided_errors"]
        for row in payload["analysis"]["action_metrics_by_family"]
    )
    assert int(paired[2]) == sum(
        row["induced_errors"]
        for row in payload["analysis"]["action_metrics_by_family"]
    )
    assert fallback == sum(
        sum(row["fallback_window_decision_counts"]) for row in raw_rows
    )
    assert unnecessary == sum(
        sum(row["unnecessary_fallback_window_decision_counts"]) for row in raw_rows
    )


def test_cache_replay_and_mutation_audits_are_complete() -> None:
    payload = _artifact()
    cache = payload["cache_audit"]
    assert cache["hits"] == 888
    assert cache["misses"] == 0
    assert len(cache["cache_keys"]) == len(set(cache["cache_keys"])) == 888
    assert len(payload["semantic_mutations"]) == 6
    assert all(row["rejected"] for row in payload["semantic_mutations"])
