from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np

from cnn_fpga.benchmark.route_a_smooth_formal import (
    BOOTSTRAP_REPLICATES,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    METHODS,
    PRIMARY_BASELINE,
    ROOT,
    SMOOTH_FAMILIES,
    _run_trajectory,
    variable_gaussian_oracle_decisions,
    verify_report,
)
from physics.ideal_gkp_decoder import map_decode_2d


def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def test_committed_formal_artifact_recomputes_and_is_source_bound() -> None:
    payload = _artifact()
    verify_report(payload)
    source = ROOT / payload["source_data_binding"]["path"]
    assert source == DEFAULT_SOURCE_DATA
    assert source.is_file()
    assert payload["source_data_binding"]["row_count"] == 498_240
    assert payload["gate_summary"]["passed"] == 12
    assert payload["gate_summary"]["failed"] == 0


def test_exact_untouched_formal_matrix_and_no_reselection() -> None:
    payload = _artifact()
    rows = payload["trajectory_results"]
    assert payload["primary_baseline"] == PRIMARY_BASELINE == "ewma_adaptive_map"
    assert payload["formal_baseline_reselection"] is False
    assert len(payload["formal_design"]["seeds"]) == 24
    assert len(payload["formal_design"]["cells"]) == 24
    assert len(rows) == 576
    assert len({(row["seed"], row["cell_id"]) for row in rows}) == 576
    assert {row["family"] for row in rows} == set(SMOOTH_FAMILIES)
    assert all(sum(row["family"] == family for row in rows) == 144 for family in SMOOTH_FAMILIES)
    assert all(set(row["method_window_pauli_counts_class_order_I_Z_X_Y"]) == set(METHODS) for row in rows)


def test_variable_state_oracle_exactly_matches_reference_map_decoder() -> None:
    rng = np.random.default_rng(2026071701)
    samples = 257
    lattice = float(np.sqrt(np.pi))
    residuals = rng.uniform(-0.5 * lattice, 0.5 * lattice, size=(samples, 2))
    mean = rng.normal(0.0, 0.25, size=(samples, 2))
    sigma_q = rng.uniform(0.18, 0.48, size=samples)
    sigma_p = rng.uniform(0.18, 0.48, size=samples)
    rho = rng.uniform(-0.80, 0.80, size=samples)
    actual = variable_gaussian_oracle_decisions(
        residuals, (mean[:, 0], mean[:, 1], sigma_q, sigma_p, rho)
    )
    expected = np.empty(samples, dtype=np.uint8)
    for index in range(samples):
        covariance = np.asarray(
            [
                [sigma_q[index] ** 2, rho[index] * sigma_q[index] * sigma_p[index]],
                [rho[index] * sigma_q[index] * sigma_p[index], sigma_p[index] ** 2],
            ]
        )
        expected[index] = map_decode_2d(
            residuals[index], covariance, mean=mean[index]
        ).logical_class
    assert np.array_equal(actual, expected)


def test_online_router_prefix_does_not_consume_logical_truth_values() -> None:
    source = inspect.getsource(_run_trajectory)
    online_prefix = source[: source.index("scored_truth =")]
    # Shape/cadence uses len(truth), but no logical truth value may select a
    # posterior, action, score, expert, bank or proposed decision.
    assert "truth[" not in online_prefix
    assert "local_truth" not in online_prefix
    assert 'decisions["proposed_route_a"]' in online_prefix
    assert "active_expert" in online_prefix


def test_pauli_channels_close_and_have_seed_cluster_intervals() -> None:
    payload = _artifact()
    summaries = payload["analysis"]["method_summaries"]
    assert len(summaries) == len(METHODS)
    for row in summaries:
        assert row["decisions"] == 28_311_552
        assert sum(row["pauli_counts_class_order_I_Z_X_Y"]) == row["decisions"]
        assert np.isclose(row["p_L"], row["p_X"] + row["p_Y"] + row["p_Z"])
        assert np.isclose(row["p_I"] + row["p_L"], 1.0)
        assert set(row["paired_formal_seed_cluster_ci95"]) == {"p_I", "p_X", "p_Y", "p_Z", "p_L"}
        assert all(
            0.0 <= interval[0] <= interval[1] <= 1.0
            for interval in row["paired_formal_seed_cluster_ci95"].values()
        )


def test_source_data_independently_reconstructs_primary_and_action_totals() -> None:
    payload = _artifact()
    formal_errors = {PRIMARY_BASELINE: 0, "proposed_route_a": 0}
    paired = np.zeros(4, dtype=np.int64)
    fallback = 0
    unnecessary = 0
    rows = 0
    with DEFAULT_SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            row_type = row["row_type"]
            if row_type == "formal_window" and row["method_id"] in formal_errors:
                formal_errors[row["method_id"]] += int(row["n_X"]) + int(row["n_Y"]) + int(row["n_Z"])
            elif row_type == "paired_outcome_window":
                paired += np.asarray([int(row[key]) for key in ("n_I", "n_X", "n_Y", "n_Z")])
            elif row_type == "action_window":
                fallback += int(row["n_I"])
                unnecessary += int(row["n_X"])
    assert rows == 498_240
    analysis = payload["analysis"]
    by_method = {row["method_id"]: row for row in analysis["method_summaries"]}
    assert formal_errors[PRIMARY_BASELINE] == by_method[PRIMARY_BASELINE]["errors"]
    assert formal_errors["proposed_route_a"] == by_method["proposed_route_a"]["errors"]
    assert int(paired[1]) == analysis["action_and_update_metrics"]["avoided_errors"]
    assert int(paired[2]) == analysis["action_and_update_metrics"]["induced_errors"]
    assert fallback == sum(sum(row["fallback_window_decision_counts"]) for row in payload["trajectory_results"])
    assert unnecessary == analysis["action_and_update_metrics"]["unnecessary_fallback_decisions"]


def test_primary_gate_passes_but_claim_does_not_hide_static_window_or_family_limits() -> None:
    payload = _artifact()
    analysis = payload["analysis"]
    primary = analysis["primary_contrast"]
    assert primary["ci95_low"] > 0.0
    assert primary["passes_95_lcb_strictly_greater_than_zero"] is True
    summaries = {row["method_id"]: row for row in analysis["method_summaries"]}
    assert summaries["proposed_route_a"]["p_L"] > summaries["static_joint_map"]["p_L"]
    assert summaries["proposed_route_a"]["p_L"] > summaries["window_map"]["p_L"]
    assert analysis["oracle_gap_closure"]["gap_closure"] < 0.0
    discoveries = {
        row["family"]
        for row in analysis["holm_smooth_family_superiority"]
        if row["reject_at_familywise_0_05"]
    }
    assert discoveries == {"periodic_drift"}
    assert analysis["bootstrap_contract"]["replicates"] == BOOTSTRAP_REPLICATES


def test_cache_replay_and_semantic_mutation_audits_are_complete() -> None:
    payload = _artifact()
    audit = payload["cache_audit"]
    assert audit["hits"] == 576
    assert audit["misses"] == 0
    assert len(audit["cache_keys"]) == len(set(audit["cache_keys"])) == 576
    mutations = payload["semantic_mutations"]
    assert len(mutations) == 6
    assert all(row["rejected"] for row in mutations)

