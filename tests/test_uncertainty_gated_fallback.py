from __future__ import annotations

import copy
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import uncertainty_gated_fallback as fallback


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / fallback.DEFAULT_ARTIFACT
SOURCE = ROOT / fallback.DEFAULT_SOURCE_DATA


def _load() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rehash(payload: dict) -> None:
    payload["contract_sha256"] = fallback._canonical_sha256(
        fallback._contract_view(payload)
    )


def test_formal_split_threshold_grid_and_workload_are_frozen() -> None:
    config = fallback.FallbackValidationConfig()
    assert len(config.calibration_seeds) == 8
    assert len(config.confirmation_seeds) == 12
    assert not set(config.calibration_seeds) & set(config.confirmation_seeds)
    assert config.threshold_grid == tuple(index / 40.0 for index in range(41))
    assert config.windows * config.evaluation_samples_per_window == 32_768
    with pytest.raises(ValueError, match="formal confirmation_seeds changed"):
        fallback.FallbackValidationConfig(
            confirmation_seeds=tuple(range(12))
        )
    with pytest.raises(ValueError, match="formal windows changed"):
        fallback.FallbackValidationConfig(windows=16)


def test_uncertainty_score_is_truth_free_finite_and_bounded() -> None:
    signature = inspect.signature(fallback._uncertainty_score)
    assert "truth" not in signature.parameters
    primary = np.array([[0.7, 0.1, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]])
    peers = (
        np.array([[0.6, 0.2, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1]]),
        np.array([[0.65, 0.15, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]]),
        np.array([[0.55, 0.25, 0.1, 0.1], [0.2, 0.2, 0.5, 0.1]]),
    )
    score, components = fallback._uncertainty_score(
        primary,
        peers,
        np.array([0, 0]),
        (np.array([0, 1]), np.array([0, 0]), np.array([0, 2])),
    )
    assert set(components) == set(fallback.SCORE_COMPONENTS)
    assert score.shape == (2,)
    assert np.all(np.isfinite(score)) and np.all((0.0 <= score) & (score <= 1.0))
    assert np.allclose(score, np.maximum.reduce(list(components.values())))


def test_counting_allows_both_avoided_and_induced_failures() -> None:
    row = fallback._count_threshold(
        0.5,
        np.array([0.9, 0.9, 0.1, 0.1]),
        np.array([True, False, True, False]),
        np.array([False, True, False, False]),
    )
    assert row["fallback_count"] == 2
    assert row["avoided_failure_count"] == 1
    assert row["induced_failure_count"] == 1
    assert row["gated_failure_count"] == row["primary_failure_count"]
    enriched = {**row, **fallback._rates(row)}
    assert fallback._count_row_is_valid(enriched)


def test_committed_artifact_recomputes_all_evidence_gates() -> None:
    artifact = _load()
    assert fallback.validate_artifact(artifact) == ()
    assert artifact["gates"] == fallback._compute_gates(
        artifact, fallback.load_parent_artifacts()
    )
    assert artifact["gate_summary"] == {"passed": 21, "total": 21}
    assert artifact["status"] == "PASS"


def test_parent_bindings_and_fresh_confirmation_clusters_are_exact() -> None:
    artifact = _load()
    assert {row["task_id"] for row in artifact["parent_bindings"]} == set(
        fallback.PARENT_ARTIFACTS
    )
    assert all(row["machine_pass"] for row in artifact["parent_bindings"])
    assert artifact["split_contract"]["confirmation_used_for_selection"] is False
    parent_seeds = set().union(
        *(
            fallback.held_parent._extract_seed_values(parent)
            for parent in fallback.load_parent_artifacts().values()
        )
    )
    assert not set(fallback.CONFIRMATION_SEEDS) & parent_seeds


def test_calibration_selection_is_seed_clustered_and_precedes_confirmation() -> None:
    calibration = _load()["calibration"]
    assert calibration["selected_threshold"] == 0.45
    assert len(calibration["threshold_curve"]) == 41
    assert len(calibration["threshold_seed_rows"]) == 41 * 8
    best = max(
        row["catastrophic_reduction_seed_cluster_ci"]["estimate"]
        for row in calibration["threshold_curve"]
    )
    selected = next(
        row
        for row in calibration["threshold_curve"]
        if row["threshold"] == calibration["selected_threshold"]
    )
    assert selected["catastrophic_reduction_seed_cluster_ci"]["estimate"] == best


def test_confirmatory_ood_aggregate_has_modest_positive_paired_reduction() -> None:
    lane = _load()["confirmation_ood"]
    metric = lane["metrics"]["absolute_catastrophic_reduction"]
    assert lane["sample_accounting"]["decisions"] == 1_179_648
    assert metric["estimate"] == pytest.approx(0.0010748969184027778)
    assert metric["ci_low"] > 0.0
    assert lane["metrics"]["gated_failure_rate"]["estimate"] < lane["metrics"][
        "primary_failure_rate"
    ]["estimate"]
    accounting = lane["sample_accounting"]
    assert accounting["primary_failure_count"] - accounting["gated_failure_count"] == (
        accounting["avoided_failure_count"] - accounting["induced_failure_count"]
    )


def test_scenario_heterogeneity_and_compound_harm_are_not_hidden() -> None:
    artifact = _load()
    summaries = artifact["confirmation_ood"]["scenario_summaries"]
    assert set(summaries) == set(fallback.CONFIRMATION_OOD_SCENARIOS)
    telegraph = summaries["stochastic_telegraph_unseen_family"]["metrics"][
        "absolute_catastrophic_reduction"
    ]
    compound = summaries["compound_range_extrapolation"]["metrics"][
        "absolute_catastrophic_reduction"
    ]
    sinusoidal = summaries["joint_sinusoidal_rotation_unseen_family"]["metrics"][
        "absolute_catastrophic_reduction"
    ]
    assert telegraph["ci_low"] > 0.0
    assert compound["ci_high"] < 0.0
    assert sinusoidal["ci_low"] <= 0.0 <= sinusoidal["ci_high"]
    assert artifact["claim_boundary"]["scenario_universal_benefit"] == "NOT_ESTABLISHED"


def test_fallback_costs_and_ungated_static_comparator_are_explicit() -> None:
    artifact = _load()
    lane = artifact["confirmation_ood"]
    accounting = lane["sample_accounting"]
    assert accounting["avoided_failure_count"] > accounting["induced_failure_count"] > 0
    assert accounting["unnecessary_fallback_count"] > 0
    assert accounting["selected_without_benefit_count"] > 0
    metrics = lane["metrics"]
    assert metrics["static_failure_rate"]["estimate"] > metrics[
        "gated_failure_rate"
    ]["estimate"]
    assert artifact["action_contract"]["always_static_role"] == (
        "ungated_last_known_good_comparator_not_selected_policy"
    )


def test_nominal_negative_control_retains_small_cost_and_confidence_interval() -> None:
    nominal = _load()["confirmation_nominal"]
    reduction = nominal["metrics"]["absolute_catastrophic_reduction"]
    assert nominal["scope"] == "in_distribution_negative_control"
    assert reduction["estimate"] < 0.0
    assert reduction["ci_low"] < 0.0 < reduction["ci_high"]
    assert nominal["metrics"]["fallback_rate"]["estimate"] > 0.0
    assert nominal["sample_accounting"]["induced_failure_count"] > nominal[
        "sample_accounting"
    ]["avoided_failure_count"]


def test_every_materialized_cell_threshold_and_window_recomputes() -> None:
    artifact = _load()
    cells = (
        *artifact["calibration"]["cells"],
        *artifact["confirmation_ood"]["cells"],
        *artifact["confirmation_nominal"]["cells"],
    )
    assert len(cells) == 24 + 36 + 12
    assert all(fallback._cell_accounting_is_valid(cell) for cell in cells)
    assert len({cell["trace_sha256"] for cell in artifact["confirmation_ood"]["cells"]}) == 36


@pytest.mark.parametrize(
    "mutation",
    ("threshold", "cell_accounting", "hide_negative_lane", "truth_input"),
)
def test_semantic_mutations_fail_closed_after_contract_rehash(mutation: str) -> None:
    artifact = copy.deepcopy(_load())
    if mutation == "threshold":
        artifact["calibration"]["selected_threshold"] = 0.475
    elif mutation == "cell_accounting":
        row = artifact["confirmation_ood"]["cells"][0]["threshold_rows"][0]
        row["avoided_failure_count"] += 1
    elif mutation == "hide_negative_lane":
        interval = artifact["confirmation_ood"]["scenario_summaries"][
            "compound_range_extrapolation"
        ]["metrics"]["absolute_catastrophic_reduction"]
        interval["ci_low"] = 0.001
        interval["ci_high"] = 0.002
    else:
        artifact["uncertainty_contract"]["hidden_truth_inputs"] = ["logical_truth"]
    _rehash(artifact)
    errors = fallback.validate_artifact(artifact)
    assert "stored gates do not match recomputed evidence gates" in errors


def test_source_data_is_complete_canonical_and_byte_hash_bound() -> None:
    artifact = _load()
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == artifact["source_data"]["row_count"] == 517
    assert fallback._sha256(SOURCE) == artifact["source_data"]["csv_sha256"]
    assert fallback._canonical_sha256(fallback.source_rows(artifact)) == artifact[
        "source_data"
    ]["rows_sha256"]
    assert {row["row_type"] for row in rows} == {
        "parent_binding",
        "calibration_threshold_seed",
        "calibration_threshold_aggregate",
        "confirmation_seed",
        "confirmation_aggregate",
        "confirmation_scenario_aggregate",
        "confirmation_cell",
        "gate",
    }


def test_claim_boundary_stays_at_syndrome_decision_level() -> None:
    artifact = _load()
    assert artifact["device_calibrated"] is False
    assert artifact["physical_memory_ler_established"] is False
    assert artifact["claim_boundary"]["fallback_scope"] == (
        "syndrome_decision_level_last_known_good_map_selection"
    )
    forbidden = artifact["claim_boundary"]["forbidden"]
    for phrase in ("physical-memory LER", "device catastrophic", "universal OOD"):
        assert phrase in forbidden
