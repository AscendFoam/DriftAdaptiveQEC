from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.adaptive_drift_alignment import (
    AdaptiveAlignmentConfig,
    run_adaptive_drift_alignment,
)
from cnn_fpga.benchmark.static_map_baseline import (
    STATIC_MAP_DESCRIPTOR,
    STATIC_MAP_ID,
    StaticMAPParameters,
    StaticMAPValidationConfig,
    build_static_map_validation,
    fit_static_map_from_training_states,
    static_map_logical_class,
    validate_static_map_major_comparisons,
)
from cnn_fpga.benchmark.standard_binning_baseline import major_comparison_registry
from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState
from physics.ideal_gkp_decoder import map_decode_2d


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_1_2_static_map_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_1_2_static_map_source_data.csv"


def _state(
    *,
    mu_q: float,
    mu_p: float,
    sigma_q: float,
    sigma_p: float,
    rho: float = 0.0,
    p_outlier: float = 0.0,
    outlier_scale: float = 1.0,
    loss_gamma: float = 0.0,
) -> DriftState:
    return DriftState(
        mu_q=mu_q,
        mu_p=mu_p,
        sigma_q=sigma_q,
        sigma_p=sigma_p,
        rho=rho,
        p_outlier=p_outlier,
        outlier_scale=outlier_scale,
        loss_gamma=loss_gamma,
    )


def test_descriptor_freezes_training_only_and_no_update_semantics():
    descriptor = STATIC_MAP_DESCRIPTOR

    assert descriptor.baseline_id == STATIC_MAP_ID
    assert descriptor.deployable
    assert descriptor.evaluation_hidden_truth_inputs == ()
    assert not descriptor.update_during_evaluation
    assert "training_state_mean" in descriptor.training_inputs
    assert descriptor.evaluation_inputs == (
        "centered_modular_syndrome_q",
        "centered_modular_syndrome_p",
    )


def test_fit_uses_law_of_total_covariance_not_naive_covariance_average():
    first = _state(mu_q=-1.0, mu_p=0.5, sigma_q=0.2, sigma_p=0.3)
    second = _state(mu_q=1.0, mu_p=-0.5, sigma_q=0.4, sigma_p=0.5)

    fitted = fit_static_map_from_training_states(
        (first, second),
        training_protocol_id="unit-two-state",
    )

    mean = np.array([0.0, 0.0])
    expected = 0.5 * (
        first.mixture_covariance + np.outer(first.mean - mean, first.mean - mean)
    ) + 0.5 * (
        second.mixture_covariance + np.outer(second.mean - mean, second.mean - mean)
    )
    naive = 0.5 * (first.mixture_covariance + second.mixture_covariance)
    assert np.allclose(fitted.mean_array(), mean, atol=0.0, rtol=0.0)
    assert np.allclose(fitted.covariance_array(), expected, atol=1.0e-15, rtol=0.0)
    assert not np.allclose(fitted.covariance_array(), naive)


def test_weighted_fit_and_hash_are_deterministic_and_weight_sensitive():
    states = (
        _state(mu_q=0.0, mu_p=0.0, sigma_q=0.2, sigma_p=0.3),
        _state(mu_q=1.0, mu_p=-1.0, sigma_q=0.4, sigma_p=0.5),
    )
    first = fit_static_map_from_training_states(
        states,
        weights=[1.0, 3.0],
        training_protocol_id="weighted",
    )
    repeat = fit_static_map_from_training_states(
        states,
        weights=[1.0, 3.0],
        training_protocol_id="weighted",
    )
    changed = fit_static_map_from_training_states(
        states,
        weights=[3.0, 1.0],
        training_protocol_id="weighted",
    )

    assert first == repeat
    assert first.mean == pytest.approx((0.75, -0.75), abs=0.0)
    assert first.effective_training_weight == 4.0
    assert first.training_state_sha256 != changed.training_state_sha256


def test_fit_moment_matches_same_mean_outlier_mixture():
    state = _state(
        mu_q=0.1,
        mu_p=-0.2,
        sigma_q=0.3,
        sigma_p=0.4,
        p_outlier=0.25,
        outlier_scale=3.0,
    )
    fitted = fit_static_map_from_training_states(
        (state, state),
        training_protocol_id="mixture",
    )

    assert np.allclose(fitted.covariance_array(), state.mixture_covariance)
    assert not np.allclose(fitted.covariance_array(), state.covariance)


def test_fit_rejects_loss_instead_of_silent_displacement_proxy():
    lossy = _state(
        mu_q=0.0,
        mu_p=0.0,
        sigma_q=0.2,
        sigma_p=0.2,
        loss_gamma=0.01,
    )
    with pytest.raises(ValueError, match="nonzero loss_gamma"):
        fit_static_map_from_training_states(
            (lossy, lossy),
            training_protocol_id="lossy",
        )


@pytest.mark.parametrize(
    ("states", "weights", "protocol", "error", "message"),
    [
        ((), None, "x", ValueError, "at least two"),
        ((_state(mu_q=0, mu_p=0, sigma_q=1, sigma_p=1),), None, "x", ValueError, "at least two"),
        ((object(), object()), None, "x", TypeError, "DriftState"),
        (
            (
                _state(mu_q=0, mu_p=0, sigma_q=1, sigma_p=1),
                _state(mu_q=0, mu_p=0, sigma_q=1, sigma_p=1),
            ),
            [1.0],
            "x",
            ValueError,
            "one entry",
        ),
        (
            (
                _state(mu_q=0, mu_p=0, sigma_q=1, sigma_p=1),
                _state(mu_q=0, mu_p=0, sigma_q=1, sigma_p=1),
            ),
            [1.0, 0.0],
            "x",
            ValueError,
            "strictly positive",
        ),
        (
            (
                _state(mu_q=0, mu_p=0, sigma_q=1, sigma_p=1),
                _state(mu_q=0, mu_p=0, sigma_q=1, sigma_p=1),
            ),
            None,
            " ",
            ValueError,
            "must not be empty",
        ),
    ],
)
def test_fit_failure_branches(states, weights, protocol, error, message):
    with pytest.raises(error, match=message):
        fit_static_map_from_training_states(
            states,
            weights=weights,
            training_protocol_id=protocol,
        )


def test_static_decoder_matches_reference_and_is_chunk_invariant():
    states = (
        _state(mu_q=0.1, mu_p=-0.2, sigma_q=0.3, sigma_p=0.2, rho=0.2),
        _state(mu_q=0.4, mu_p=0.1, sigma_q=0.5, sigma_p=0.3, rho=-0.1),
    )
    parameters = fit_static_map_from_training_states(
        states,
        training_protocol_id="decode-reference",
    )
    rng = np.random.default_rng(312)
    syndrome = rng.uniform(
        -LATTICE_CONST / 2.0,
        np.nextafter(LATTICE_CONST / 2.0, -np.inf),
        size=(31, 2),
    )
    expected = np.asarray(
        map_decode_2d(
            syndrome,
            parameters.covariance_array(),
            mean=parameters.mean_array(),
        ).logical_class
    )

    assert np.array_equal(static_map_logical_class(syndrome, parameters), expected)
    assert np.array_equal(
        static_map_logical_class(syndrome, parameters, chunk_size=7),
        expected,
    )


def test_static_decoder_reuses_frozen_parameters_without_mutation():
    state = _state(mu_q=0.1, mu_p=-0.1, sigma_q=0.3, sigma_p=0.25)
    parameters = fit_static_map_from_training_states(
        (state, state),
        training_protocol_id="frozen",
    )
    before = parameters
    first = static_map_logical_class(np.zeros((8, 2)), parameters)
    second = static_map_logical_class(np.zeros((8, 2)), parameters)

    assert parameters == before
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("syndrome", "chunk_size", "error", "message"),
    [
        (np.zeros(3), 2_000, ValueError, "shape"),
        (np.array([[np.nan, 0.0]]), 2_000, ValueError, "finite"),
        (np.zeros((1, 2)), 0, ValueError, "positive"),
        (np.zeros((1, 2)), 1.5, TypeError, "integer"),
    ],
)
def test_static_decoder_failure_branches(syndrome, chunk_size, error, message):
    state = _state(mu_q=0, mu_p=0, sigma_q=0.3, sigma_p=0.3)
    parameters = fit_static_map_from_training_states(
        (state, state),
        training_protocol_id="decode-errors",
    )
    with pytest.raises(error, match=message):
        static_map_logical_class(syndrome, parameters, chunk_size=chunk_size)


def test_adaptive_main_comparison_uses_one_training_hash_across_eval_seeds():
    common = dict(
        windows=8,
        change_step=4,
        calibration_windows=2,
        observation_samples_per_window=400,
        evaluation_samples_per_window=400,
        histogram_bins=24,
        bootstrap_replicates=0,
        static_training_seed=999,
    )
    first = run_adaptive_drift_alignment(AdaptiveAlignmentConfig(seed=1001, **common))
    second = run_adaptive_drift_alignment(AdaptiveAlignmentConfig(seed=1002, **common))
    active = next(
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t1_3_4_adaptive_drift_alignment"
    )

    assert first.static_parameters == second.static_parameters
    assert first.static_training_state_sha256 == second.static_training_state_sha256
    assert first.trace_sha256 != second.trace_sha256
    assert first.comparison_method_ids == active.method_ids
    assert active.method_ids.count(STATIC_MAP_ID) == 1


def test_formal_static_map_validator_respects_task_specific_static_anchors():
    gates = validate_static_map_major_comparisons()
    formal_static = [
        entry
        for entry in major_comparison_registry()
        if entry.static_anchor_method_id == STATIC_MAP_ID
    ]

    assert len(gates) == len(formal_static)
    assert all(entry.method_ids.count(STATIC_MAP_ID) == 1 for entry in formal_static)
    memory = next(
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t3_2_1_memory_bayesian_episode_comparison"
    )
    assert memory.static_anchor_method_id == "final_outcome_static_periodic_bayes"
    assert STATIC_MAP_ID not in memory.method_ids


def test_production_gate_names_the_declared_formal_static_scope() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    gates = payload["gate_summary"]["gates"]
    assert "formal_static_map_present_in_declared_schemas" in gates
    assert "standard_and_static_present_in_all_required_schemas" not in gates


@pytest.mark.parametrize(
    "kwargs",
    [
        {"evaluation_seeds": (1, 2, 3)},
        {"evaluation_seeds": (1, 2, 3, 3)},
        {"evaluation_seeds": (1, 2, 3, 4), "training_seed": 4},
        {"evaluation_seeds": (1, 2, 3, 4), "confidence_level": 1.0},
    ],
)
def test_validation_config_rejects_ambiguous_split(kwargs):
    with pytest.raises((TypeError, ValueError)):
        StaticMAPValidationConfig(**kwargs)


def _implementation_hash() -> str:
    paths = (
        ROOT / "cnn_fpga" / "benchmark" / "static_map_baseline.py",
        ROOT / "cnn_fpga" / "benchmark" / "adaptive_drift_alignment.py",
        ROOT / "cnn_fpga" / "benchmark" / "standard_binning_baseline.py",
        ROOT / "physics" / "ideal_gkp_decoder.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_production_artifact_and_source_data_are_complete_and_source_bound():
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_hash()
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] == len(payload["gate_summary"]["gates"])
    parameters = payload["frozen_static_parameters"]
    covariance = np.asarray(parameters["covariance"], dtype=float)
    assert parameters["training_state_sha256"] == payload["aggregate"]["training_state_sha256"]
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    aggregate = payload["aggregate"]
    assert aggregate["evaluation_seeds"] == 8
    assert aggregate["paired_samples"] == 576_000
    assert aggregate["standard_minus_static"]["ci_low"] > 0.0
    assert aggregate["static_training_average_map_error_rate"] < aggregate[
        "standard_binning_error_rate"
    ]
    assert aggregate["full_state_model_oracle_map_error_rate"] < aggregate[
        "static_training_average_map_error_rate"
    ]
    assert len(rows) == 8
    assert len({row["evaluation_trace_sha256"] for row in rows}) == 8
    assert len({row["training_state_sha256"] for row in rows}) == 1


def test_validation_builder_reproduces_json_and_csv_semantics():
    payload, rows = build_static_map_validation()
    persisted = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert payload == persisted
    assert len(rows) == len(csv_rows)
    assert [int(row["evaluation_seed"]) for row in rows] == [
        int(row["evaluation_seed"]) for row in csv_rows
    ]


def test_parameter_dataclass_rejects_non_spd_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        StaticMAPParameters(
            mean=(0.0, 0.0),
            covariance=((1.0, 2.0), (2.0, 1.0)),
            training_windows=2,
            effective_training_weight=2.0,
            training_state_sha256="0" * 64,
            training_protocol_id="bad-covariance",
        )


def test_adaptive_config_rejects_training_eval_seed_alias_and_bad_schedule():
    with pytest.raises(ValueError, match="must differ"):
        AdaptiveAlignmentConfig(seed=10, static_training_seed=10)
    with pytest.raises(ValueError, match="static_training_change_step"):
        AdaptiveAlignmentConfig(static_training_windows=4, static_training_change_step=4)
