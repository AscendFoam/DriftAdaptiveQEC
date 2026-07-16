from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.sliding_window_syndrome_estimator import (
    COMPARISON_ID,
    SLIDING_DESCRIPTOR,
    SLIDING_WINDOW_MAP_ID,
    SlidingWindowValidationConfig,
    _implementation_sha256,
    select_frozen_window,
    validate_sliding_window_registration,
)
from cnn_fpga.benchmark.standard_binning_baseline import major_comparison_registry


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_2_3_sliding_window_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_2_3_sliding_window_source_data.csv"


def _payload() -> dict[str, object]:
    return json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))


def test_descriptor_and_registry_freeze_roles_and_causal_budget() -> None:
    assert SLIDING_DESCRIPTOR.hidden_truth_inputs == ()
    assert SLIDING_DESCRIPTOR.update_timing.startswith("one_window_delay")
    validate_sliding_window_registration()
    entry = next(item for item in major_comparison_registry() if item.comparison_id == COMPARISON_ID)
    assert entry.method_ids == (
        "standard_binning",
        "static_training_average_map",
        "latest_window_periodic_moment_map",
        SLIDING_WINDOW_MAP_ID,
        "full_state_model_oracle_map",
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"training_seeds": (1, 2)}, "at least three"),
        ({"evaluation_seeds": (1, 2, 3, 4, 5)}, "at least six"),
        ({"training_seeds": (1, 2, 3), "evaluation_seeds": (3, 4, 5, 6, 7, 8)}, "disjoint"),
        ({"window_sample_candidates": (480, 576, 768, 1152)}, "first candidate"),
        ({"observation_samples_per_window": 383}, "divisible"),
        ({"confidence_level": 1.0}, "confidence"),
    ],
)
def test_validation_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        SlidingWindowValidationConfig(**kwargs)  # type: ignore[arg-type]


def test_production_artifact_is_source_bound_and_non_demo_scale() -> None:
    payload = _payload()
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_sha256()
    aggregate = payload["aggregate"]
    assert aggregate["evaluation_samples"] == 1_572_864
    assert aggregate["source_data_rows"] == 32
    assert payload["gate_summary"]["passed"] == 14
    assert payload["gate_summary"]["failed"] == 0


def test_training_selection_uses_full_grid_and_exact_argmin() -> None:
    payload = _payload()
    frozen = payload["frozen_window_selection"]
    candidates = payload["validation_config"]["window_sample_candidates"]
    scores = frozen["candidate_scores"]
    assert [row[0] for row in scores] == candidates
    expected = min(scores, key=lambda row: (row[1], row[0]))[0]
    assert frozen["selected_window_samples"] == expected
    assert frozen["selected_window_samples"] == 384
    assert len(frozen["training_trace_sha256"]) == 64
    assert payload["aggregate"]["evaluation_best_window_samples_diagnostic_only"] == 384


def test_negative_long_window_result_is_retained_without_promotion() -> None:
    payload = _payload()
    aggregate = payload["aggregate"]
    boundary = payload["claim_boundary"]
    assert boundary["selection_result"] == "latest_window_selected"
    assert boundary["latest_comparison_resolved"] is False
    assert boundary["evaluation_best_is_diagnostic_not_a_selector"] is True
    assert aggregate["latest_minus_selected_seed_cluster_ci"]["estimate"] == 0.0
    assert "universal optimal window" in boundary["forbidden"]
    for scenario in payload["scenarios"]:
        assert len(scenario["candidate_error_rates"]) == 6
        assert scenario["selected_window_samples"] == 384
        assert scenario["selected_error_rate"] == scenario["latest_window_error_rate"]


def test_selected_window_beats_static_with_proper_scores_and_oracle_stays_strict() -> None:
    payload = _payload()
    assert payload["aggregate"]["static_minus_selected_seed_cluster_ci"]["ci_low"] > 0.0
    for scenario in payload["scenarios"]:
        assert scenario["static_minus_selected_seed_cluster_ci"]["ci_low"] > 0.0
        assert scenario["selected_nll"] < scenario["static_nll"]
        assert scenario["selected_brier"] < scenario["static_brier"]
        assert scenario["oracle_error_rate"] < scenario["selected_error_rate"]


def test_source_data_has_unique_traces_and_recomputes_cluster_estimates() -> None:
    payload = _payload()
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 32
    assert len({row["trace_sha256"] for row in rows}) == 32
    assert all(len(row["trace_sha256"]) == 64 for row in rows)
    selected = payload["frozen_window_selection"]["selected_window_samples"]
    for row in rows:
        assert float(row["selected_error_rate"]) == float(row[f"window_{selected}_error_rate"])
        assert int(row["observation_samples_per_window"]) == 384
        assert int(row["updates_per_window"]) == 1
    cluster_values = []
    for seed in payload["validation_config"]["evaluation_seeds"]:
        cluster = [row for row in rows if int(row["base_evaluation_seed"]) == seed]
        cluster_values.append(np.mean([float(row["static_minus_selected_error_rate"]) for row in cluster]))
    assert np.mean(cluster_values) == pytest.approx(
        payload["aggregate"]["static_minus_selected_seed_cluster_ci"]["estimate"], abs=1e-15
    )


def test_cost_surface_is_monotone_and_hardware_is_null() -> None:
    costs = _payload()["cost_profiles"]
    storage = [row["stored_complex_values"] for row in costs]
    assert storage == sorted(storage)
    assert len(set(storage)) == len(storage)
    assert all(row["complex_exponentials_per_observation"] == 2 for row in costs)
    for row in costs:
        assert row["target_lut"] is None
        assert row["target_bram"] is None
        assert row["target_dsp"] is None
        assert row["target_fmax_hz"] is None
        assert row["target_measured"] is False
