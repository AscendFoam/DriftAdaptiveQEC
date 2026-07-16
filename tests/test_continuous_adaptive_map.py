from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.continuous_adaptive_map import (
    ADAPTIVE_DESCRIPTOR,
    COMPARISON_ID,
    ContinuousAdaptiveValidationConfig,
    adaptive_cost_profile,
    continuous_drift_scenarios,
    validate_continuous_adaptive_registration,
)
from physics.constants import LATTICE_CONST


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_2_2_continuous_adaptive_map_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_2_2_continuous_adaptive_map_source_data.csv"


def _implementation_hash() -> str:
    paths = (
        ROOT / "cnn_fpga" / "benchmark" / "continuous_adaptive_map.py",
        ROOT / "cnn_fpga" / "decoder" / "periodic_adaptive_map.py",
        ROOT / "cnn_fpga" / "benchmark" / "standard_binning_baseline.py",
        ROOT / "cnn_fpga" / "benchmark" / "static_map_baseline.py",
        ROOT / "physics" / "ideal_gkp_decoder.py",
        ROOT / "physics" / "drift_processes.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_descriptor_and_registry_freeze_observed_only_role_contract() -> None:
    assert ADAPTIVE_DESCRIPTOR.hidden_truth_inputs == ()
    assert ADAPTIVE_DESCRIPTOR.consumed_observation_fields == (
        "residual_q",
        "residual_p",
    )
    assert ADAPTIVE_DESCRIPTOR.update_timing.startswith("one_window_delay")
    gates = validate_continuous_adaptive_registration()
    assert f"registry:{COMPARISON_ID}" in gates


def test_continuous_scenarios_have_no_step_jumps_and_remain_in_model_envelope() -> None:
    for scenario in continuous_drift_scenarios():
        states = scenario.states(48)
        means = np.asarray([state.mean for state in states]) / LATTICE_CONST
        covariance = np.asarray([state.covariance for state in states]) / LATTICE_CONST**2
        assert np.max(np.abs(np.diff(means, axis=0))) < 0.08
        assert np.max(np.abs(np.diff(covariance, axis=0))) < 0.02
        assert np.max(np.abs(means)) < 0.45
        assert all(state.loss_gamma == 0.0 and state.p_outlier == 0.0 for state in states)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"training_seeds": (1, 2)},
        {"evaluation_seeds": (4, 5, 6, 7, 8)},
        {"training_seeds": (1, 2, 3), "evaluation_seeds": (3, 4, 5, 6, 7, 8)},
        {"windows": 15},
        {"calibration_windows": 48},
        {"observation_samples_per_window": 127},
        {"ewma_alpha_candidates": (0.2, 0.4)},
        {"kalman_process_scale_candidates": (1.0,)},
        {"confidence_level": 1.0},
    ],
)
def test_validation_config_fails_closed(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        ContinuousAdaptiveValidationConfig(**kwargs)


def test_production_artifact_is_source_bound_and_non_demo_scale() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_hash()
    assert payload["aggregate"]["evaluation_samples"] == 1_572_864
    assert payload["aggregate"]["windows"] == 1536
    assert payload["aggregate"]["source_data_rows"] == 32
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] == 15
    assert payload["observation_budget"]["causal_delay_windows"] == 1
    assert payload["observation_budget"]["hidden_truth_inputs"] == []


def test_training_only_selection_has_full_grid_counterevidence_and_interior_optimum() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    frozen = payload["frozen_hyperparameters"]
    assert frozen["ewma_alpha"] == pytest.approx(0.85)
    assert frozen["kalman_process_scale"] == pytest.approx(2.0)
    assert frozen["kalman_measurement_scale"] == pytest.approx(0.75)
    assert len(frozen["ewma_candidate_scores"]) == 7
    assert len(frozen["kalman_candidate_scores"]) == 20
    assert min(frozen["ewma_candidate_scores"], key=lambda row: row[1])[0] == pytest.approx(
        frozen["ewma_alpha"]
    )
    best_kalman = min(frozen["kalman_candidate_scores"], key=lambda row: row[2])
    assert best_kalman[:2] == pytest.approx(
        [frozen["kalman_process_scale"], frozen["kalman_measurement_scale"]]
    )
    assert set(payload["validation_config"]["training_seeds"]).isdisjoint(
        payload["validation_config"]["evaluation_seeds"]
    )


def test_every_scenario_resolves_both_adaptive_gains_and_proper_scores() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    for scenario in payload["scenarios"]:
        assert scenario["static_minus_ewma_seed_cluster_ci"]["ci_low"] > 0.0
        assert scenario["static_minus_kalman_seed_cluster_ci"]["ci_low"] > 0.0
        assert scenario["static_minus_ewma_seed_cluster_ci"]["degrees_freedom"] == 7
        assert scenario["static_minus_kalman_seed_cluster_ci"]["interval_method"] == (
            "two_sided_student_t_cluster_mean"
        )
        assert min(scenario["ewma_error_rate"], scenario["kalman_error_rate"]) <= (
            scenario["window_error_rate"] + 1.0e-15
        )
        assert min(scenario["ewma_nll"], scenario["kalman_nll"]) < scenario["static_nll"]
        assert min(scenario["ewma_brier"], scenario["kalman_brier"]) < scenario[
            "static_brier"
        ]
        assert min(
            scenario["ewma_mean_tracking_rmse_lattice"],
            scenario["kalman_mean_tracking_rmse_lattice"],
        ) < scenario["window_mean_tracking_rmse_lattice"]
        assert min(
            scenario["ewma_covariance_tracking_rmse_lattice2"],
            scenario["kalman_covariance_tracking_rmse_lattice2"],
        ) < scenario["window_covariance_tracking_rmse_lattice2"]
        assert min(scenario["ewma_error_rate"], scenario["kalman_error_rate"]) > scenario[
            "oracle_error_rate"
        ]


def test_source_data_recomputes_seed_cluster_gains_and_trace_uniqueness() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 32
    assert len({row["trace_sha256"] for row in rows}) == 32
    assert all(int(row["observation_samples_per_window"]) == 384 for row in rows)
    for scenario in payload["scenarios"]:
        selected = [row for row in rows if row["scenario_id"] == scenario["scenario_id"]]
        assert len(selected) == 8
        for method in ("ewma", "kalman"):
            gains = np.asarray(
                [float(row[f"static_minus_{method}_error_rate"]) for row in selected]
            )
            assert np.mean(gains) == pytest.approx(
                scenario[f"static_minus_{method}_seed_cluster_ci"]["estimate"],
                abs=1.0e-15,
            )


def test_aggregate_seed_cluster_means_recompute_from_csv() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    suffixes = sorted({int(row["evaluation_seed"]) % 100_000 for row in rows})
    for method in ("ewma", "kalman"):
        cluster_means = []
        for suffix in suffixes:
            selected = [row for row in rows if int(row["evaluation_seed"]) % 100_000 == suffix]
            cluster_means.append(
                np.mean([float(row[f"static_minus_{method}_error_rate"]) for row in selected])
            )
        interval = payload["aggregate"][f"static_minus_{method}_seed_cluster_ci"]
        assert np.mean(cluster_means) == pytest.approx(interval["estimate"], abs=1.0e-15)
        assert interval["ci_low"] > 0.009


def test_cost_profile_is_deterministic_and_keeps_hardware_fields_null() -> None:
    profile = adaptive_cost_profile(ContinuousAdaptiveValidationConfig())
    assert profile.observation_samples_per_window == 384
    assert profile.complex_exponentials_per_observation == 2
    assert profile.complex_products_per_observation == 2
    assert profile.ewma_complex_state_values == 4
    assert profile.kalman_state_values == 10
    assert profile.kalman_covariance_values == 100
    assert profile.target_lut is None
    assert profile.target_bram is None
    assert profile.target_dsp is None
    assert profile.target_fmax_hz is None
    assert profile.target_measured is False


def test_claim_boundary_rejects_cnn_hardware_and_non_gaussian_overclaim() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    forbidden = payload["claim_boundary"]["forbidden"]
    assert "CNN superiority" in forbidden
    assert "loss/outlier/leakage" in forbidden
    assert "FPGA synthesis" in forbidden
    assert "device_calibration" in payload["descriptor"]["excluded_claims"]

