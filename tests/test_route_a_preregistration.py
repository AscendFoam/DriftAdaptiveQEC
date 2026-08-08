from __future__ import annotations

from copy import deepcopy
import json

import pytest

from cnn_fpga.benchmark.route_a_preregistration import (
    ABRUPT_OOD_FAMILIES,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    FUTURE_FORMAL_ARTIFACTS,
    MUTATIONS,
    NOMINAL_FAMILY,
    SMOOTH_FAMILIES,
    build_report,
    protocol_payload,
    recompute_gates,
    scenario_cells,
    split_specs,
    validate_protocol,
    verify_report,
)


def test_three_splits_have_disjoint_seeds_rates_amplitudes_and_durations() -> None:
    splits = split_specs()
    assert tuple(row.split_id for row in splits) == (
        "calibration",
        "pilot_validation",
        "formal_evaluation",
    )
    for attribute in (
        "seeds",
        "transition_rates_per_window",
        "amplitudes",
        "durations_windows",
    ):
        sets = [set(getattr(row, attribute)) for row in splits]
        assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3))
    assert [len(row.seeds) for row in splits] == [12, 12, 24]
    assert [row.scored_windows_per_cell for row in splits] == [48, 64, 96]
    assert all(row.decisions_per_window == 512 for row in splits)


def test_scenario_design_covers_every_family_and_exact_heldout_formal_cells() -> None:
    cells = scenario_cells()
    assert len(cells) == 143
    assert len({row["cell_id"] for row in cells}) == len(cells)
    assert set(row["family"] for row in cells) == set(
        (*SMOOTH_FAMILIES, *ABRUPT_OOD_FAMILIES, NOMINAL_FAMILY)
    )
    formal_dynamic = [
        row
        for row in cells
        if row["split_id"] == "formal_evaluation" and row["family"] != NOMINAL_FAMILY
    ]
    assert len(formal_dynamic) == 60
    for family in (*SMOOTH_FAMILIES, *ABRUPT_OOD_FAMILIES):
        family_rows = [row for row in formal_dynamic if row["family"] == family]
        assert len(family_rows) == 6
        assert len(
            {
                (
                    row["transition_rate_per_window"],
                    row["amplitude"],
                    row["duration_windows"],
                )
                for row in family_rows
            }
        ) == 6
        assert all(row["amplitude_semantics"] for row in family_rows)


def test_formal_workload_is_large_clustered_and_not_window_pseudoreplication() -> None:
    payload = protocol_payload()
    workload = payload["formal_workload"]
    assert workload == {
        "dynamic_cells": 60,
        "nominal_cells": 1,
        "seed_clusters": 24,
        "trajectories_per_method": 1464,
        "scored_decisions_per_method": 71_958_528,
        "seven_deployable_method_decisions": 503_709_696,
        "oracle_decisions_reported_separately": 71_958_528,
    }
    statistics = payload["statistics"]
    assert statistics["independent_cluster"].startswith("formal seed")
    assert statistics["formal_cluster_count"] == 24
    assert statistics["bootstrap"]["replicates"] == 20_000
    assert statistics["bootstrap"]["seed"] == 202607176999
    assert payload["metrics"]["window"].startswith("nonoverlapping 512 decisions")
    assert statistics["aggregation_weights"]["aggregate_smooth"].startswith(
        "four smooth family estimates equally weighted"
    )
    assert "post-result reweighting" in statistics["aggregation_weights"]["forbidden"]
    schedule = payload["trace_schedule"]
    assert schedule["shared_across_methods"] is True
    assert "SHA256" in schedule["domain_separated_seed_rule"]
    assert schedule["abrupt_onset"].endswith("offset in [0,7]")
    assert len(schedule["streams"]) == 5


def test_threshold_selection_is_common_validation_only_and_unselected_before_t663() -> None:
    selection = protocol_payload()["threshold_selection"]
    assert selection["selection_split"] == "pilot_validation"
    assert selection["formal_evaluation_access_allowed"] is False
    assert selection["one_tuple_shared_by_all_scenarios"] is True
    assert selection["per_scenario_thresholds_prohibited"] is True
    assert selection["selected_threshold_tuple"] is None
    assert selection["selected_strongest_deployable_baseline"] is None
    assert selection["lock_sha256"] is None
    assert len(selection["candidate_grid"]) == 6
    assert selection["failure_rule"].startswith("if no candidate passes")


def test_metrics_tail_catastrophic_nominal_and_multiplicity_are_exactly_frozen() -> None:
    payload = protocol_payload()
    metrics = payload["metrics"]
    assert metrics["p_l"].startswith("p_X+p_Y+p_Z")
    assert metrics["p95_window_ler"].endswith("method='higher' over registered 512-decision windows")
    assert "right-censored" in metrics["adaptation_lag"]
    assert "evaluation-only" in metrics["false_update"]

    gates = payload["acceptance_gates"]
    assert gates["primary_smooth"]["aggregate_paired_95_lcb_min_exclusive"] == 0.0
    assert gates["calibration_shift_tail"]["global_worst_error_count_proposed_max_relative_to_baseline"] == 0
    assert "55/512 > static 37/512" in gates["calibration_shift_tail"]["prior_counterexample_target"]
    catastrophic = gates["catastrophic_degradation_each_abrupt_ood_family"]
    assert catastrophic["average_ler_proposed_minus_baseline_95_ucb_max"] == 0.002
    assert catastrophic["p95_window_ler_proposed_minus_baseline_95_ucb_max"] == 4 / 512
    assert catastrophic["seed_worst_window_ler_proposed_minus_baseline_95_ucb_max"] == 8 / 512
    assert catastrophic["any_single_window_excess_error_count_max"] == 16
    nominal = gates["nominal_non_inferiority"]
    assert nominal["average_ler_proposed_minus_policy_off_95_ucb_max"] == 0.0005
    assert nominal["fallback_rate_max"] == 0.01
    assert nominal["induced_minus_avoided_rate_95_ucb_max"] == 0.00025
    multiplicity = payload["statistics"]["multiplicity"]
    assert multiplicity["method"] == "Holm step-down"
    assert multiplicity["familywise_alpha"] == 0.05


def test_prior_adaptive_design_is_disclosed_and_future_results_are_not_inputs() -> None:
    payload = protocol_payload()
    disclosure = payload["prior_evidence_disclosure"]
    assert len(disclosure) == 4
    assert all(row["exists"] and len(row["sha256"]) == 64 for row in disclosure)
    assert any("55/512" in row["disclosed_use"] for row in disclosure)
    freeze = payload["formal_result_access_at_freeze"]
    assert freeze["accessed"] is False
    assert tuple(freeze["future_artifact_paths"]) == FUTURE_FORMAL_ARTIFACTS
    assert freeze["paths_existing_when_frozen"] == []
    assert "not rechecked" in freeze["rule"]


def test_protocol_has_all_recomputable_gates_and_semantic_mutations_fail() -> None:
    report = build_report()
    verify_report(report)
    assert report["verdict"] == "PASS_ROUTE_A_RESULT_BLIND_PREREGISTRATION_FROZEN"
    gates = recompute_gates(report)
    assert len(gates) == 23
    assert all(gates.values())
    assert len(report["semantic_mutations"]) == len(MUTATIONS) == 12
    assert all(row["rejected"] for row in report["semantic_mutations"])
    assert report["threshold_selection"]["selected_threshold_tuple"] is None
    assert report["claim_boundary"]["not_claimed"] == [
        "threshold tuple selected",
        "strongest baseline selected",
        "formal evaluation executed",
        "Route-A performance or safety advantage",
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("splits", 2, "seeds"), list(range(202607176101, 202607176113))),
        (("threshold_selection", "formal_evaluation_access_allowed"), True),
        (("threshold_selection", "selected_threshold_tuple"), {"unsafe": 1}),
        (("statistics", "independent_cluster"), "window"),
        (("statistics", "bootstrap", "replicates"), 100),
        (("acceptance_gates", "calibration_shift_tail", "global_worst_error_count_proposed_max_relative_to_baseline"), 18),
        (("protocol_revision_policy",), "overwrite in place"),
    ),
)
def test_protocol_mutations_are_detected(path: tuple[object, ...], value: object) -> None:
    payload = protocol_payload()
    mutated = deepcopy(payload)
    target: object = mutated
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_protocol(mutated, verify_sources=False)


def test_committed_preregistration_artifacts_are_current() -> None:
    report = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    verify_report(report)
    rows = DEFAULT_SOURCE_DATA.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 3 + 143 + 23 + 12
    assert rows[0] == "row_type,split_or_family,item_id,value,detail"
