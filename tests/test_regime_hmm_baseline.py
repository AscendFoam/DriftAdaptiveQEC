from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.regime_hmm_baseline import (
    REGIME_HMM_DESCRIPTOR,
    RegimeHMMValidationConfig,
    _implementation_sha256,
    _trajectory,
)
from cnn_fpga.decoder.regime_hmm import REGIME_CLASSES, RegimeEstimatorBudget


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_2_6_regime_hmm_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_2_6_regime_hmm_source_data.csv"


def _payload() -> dict[str, object]:
    return json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))


def test_descriptor_is_observed_host_estimator_not_decoder_or_controller() -> None:
    assert REGIME_HMM_DESCRIPTOR.online_hidden_truth_input == ()
    assert REGIME_HMM_DESCRIPTOR.logical_decoder is False
    assert REGIME_HMM_DESCRIPTOR.controller is False
    assert REGIME_HMM_DESCRIPTOR.hardware_measured is False
    assert REGIME_HMM_DESCRIPTOR.training_only_labels == REGIME_CLASSES
    assert "same_raw_window_shape" in REGIME_HMM_DESCRIPTOR.future_cnn_fairness


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"training_seeds": (1, 2)}, "at least 3"),
        ({"validation_seeds": (1, 2)}, "at least 3"),
        ({"evaluation_seeds": (1, 2, 3, 4, 5)}, "at least 6"),
        (
            {
                "training_seeds": (1, 2, 3),
                "validation_seeds": (3, 4, 5),
                "evaluation_seeds": (6, 7, 8, 9, 10, 11),
            },
            "pairwise disjoint",
        ),
        ({"windows_per_trajectory": 127}, "at least 128"),
        ({"budget": object()}, "RegimeEstimatorBudget"),
        ({"covariance_regularization_grid": ()}, "unique positive"),
        ({"transition_smoothing_grid": (0.0,)}, "unique positive"),
        ({"temperature_grid": (1.0, 1.0)}, "unique positive"),
    ],
)
def test_validation_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        RegimeHMMValidationConfig(**kwargs)  # type: ignore[arg-type]


def test_synthetic_trajectory_is_seed_deterministic_and_truth_separated() -> None:
    settings = RegimeHMMValidationConfig(windows_per_trajectory=128)
    first = _trajectory(777, settings)
    second = _trajectory(777, settings)

    assert first.deployable_trace_sha256 == second.deployable_trace_sha256
    assert first.truth_trace_sha256 == second.truth_trace_sha256
    assert first.labels == second.labels
    assert first.deployable_trace_sha256 != first.truth_trace_sha256
    assert first.features.shape == (128, 14)
    assert first.features.flags.writeable is False
    assert set(first.labels) == set(REGIME_CLASSES)
    assert all(window.cycles == 32 for window in first.windows)


def test_production_artifact_is_source_bound_non_demo_and_all_gates_pass() -> None:
    payload = _payload()
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_sha256()
    assert payload["evaluation"]["trajectories"] == 8
    assert payload["evaluation"]["windows"] == 4096
    assert payload["evaluation"]["cycles"] == 131_072
    assert payload["evaluation"]["source_data_rows"] == 4096
    assert payload["training_selection"]["training_trajectories"] == 3
    assert payload["training_selection"]["validation_trajectories"] == 3
    assert len(payload["training_selection"]["grid"]) == 54
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] == 15


def test_selected_hyperparameters_are_interior_and_evaluation_blind() -> None:
    payload = _payload()
    config = payload["validation_config"]
    selection = payload["training_selection"]
    assert selection["evaluation_truth_used"] is False
    assert min(config["covariance_regularization_grid"]) < selection[
        "selected_covariance_regularization"
    ] < max(config["covariance_regularization_grid"])
    assert min(config["transition_smoothing_grid"]) < selection[
        "selected_transition_smoothing"
    ] < max(config["transition_smoothing_grid"])
    assert min(config["temperature_grid"]) < selection[
        "selected_hmm_temperature"
    ] < max(config["temperature_grid"])


def test_hmm_improves_same_emission_memoryless_with_all_class_recall() -> None:
    evaluation = _payload()["evaluation"]
    aggregate = evaluation["aggregate"]
    comparisons = evaluation["paired_seed_cluster_comparisons"]
    assert comparisons["memoryless_minus_hmm_nll"]["ci_low"] > 0.0
    assert comparisons["memoryless_minus_hmm_brier"]["ci_low"] > 0.0
    assert comparisons["hmm_minus_memoryless_accuracy"]["ci_low"] >= 0.0
    assert min(aggregate["causal_hmm"]["class_recall"].values()) > 0.5
    assert aggregate["causal_hmm"]["false_switch_rate"] < aggregate[
        "memoryless_emission"
    ]["false_switch_rate"]
    assert aggregate["causal_hmm"]["mean_transition_detection_delay_windows"] > 0.0


def test_shared_budget_is_exact_and_not_presented_as_future_cnn_measurement() -> None:
    payload = _payload()
    budget = payload["shared_input_budget"]
    profile = payload["model"]["profile"]
    assert budget == {
        **budget,
        "window_cycles": 32,
        "update_period_cycles": 32,
        "raw_feature_count": 8,
        "summary_feature_count": 14,
        "max_macs_per_update": 4096,
        "max_float32_state_bytes": 4096,
    }
    assert profile["parameter_count_float_values"] == 896
    assert profile["float32_state_bytes_proxy"] == 3584
    assert profile["macs_per_update_proxy"] == 800
    assert profile["host_median_us_per_update"] > 0.0
    assert "future-CNN measured latency parity" in payload["claim_boundary"]["forbidden"]


def test_source_data_has_normalized_posteriors_unique_traces_and_all_classes() -> None:
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4096
    assert len({row["deployable_trace_sha256"] for row in rows}) == 8
    assert len({row["truth_trace_sha256"] for row in rows}) == 8
    assert {row["truth_regime"] for row in rows} == set(REGIME_CLASSES)
    for row in rows:
        for estimator in ("static_prior", "memoryless_emission", "causal_hmm"):
            probabilities = [float(row[f"{estimator}_p_{state}"]) for state in REGIME_CLASSES]
            assert sum(probabilities) == pytest.approx(1.0, abs=1.0e-10)
            assert row[f"{estimator}_prediction"] in REGIME_CLASSES


def test_transition_and_emission_model_are_complete_and_stochastic() -> None:
    model = _payload()["model"]
    transition = model["transition_matrix"]
    assert len(transition) == 4
    assert all(len(row) == 4 for row in transition)
    assert all(sum(row) == pytest.approx(1.0, abs=1.0e-12) for row in transition)
    assert len(model["emission_means_standardized"]) == 4
    assert all(len(row) == 14 for row in model["emission_means_standardized"])
    assert len(model["emission_covariances_standardized"]) == 4

