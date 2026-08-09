"""Tests for the T2.3.8 noise-transfer surrogate.

The numerical checks deliberately reconstruct probabilities and cell moments
without calling the production alias helper, so the validation is not
self-referential.
"""

from __future__ import annotations

import json
from math import exp, pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad

from physics.constants import LATTICE_CONST
from physics.noise_transfer_surrogate import (
    NOISE_TRANSFER_SCOPE,
    GKPNoiseTransferSurrogate,
    NoiseTransferConfig,
    NoiseTransferState,
    fock_q_variance_alignment,
    gaussian_alias_statistics,
    projector_delta_from_squeezing_db,
    run_noise_transfer_validation,
    squeezing_db_to_peak_variance,
    write_noise_transfer_validation,
)


def _normal_density(x: float, mean: float, variance: float) -> float:
    return exp(-0.5 * (x - mean) ** 2 / variance) / sqrt(2.0 * pi * variance)


def _independent_cell_integrals(
    mean: float, variance: float, spacing: float, indices: tuple[int, ...]
) -> tuple[np.ndarray, float]:
    probabilities = []
    conditioned_variance = 0.0
    for index in indices:
        lower = (index - 0.5) * spacing
        upper = (index + 0.5) * spacing
        probability = quad(
            _normal_density,
            lower,
            upper,
            args=(mean, variance),
            epsabs=2.0e-14,
        )[0]
        first = quad(
            lambda x: x * _normal_density(x, mean, variance),
            lower,
            upper,
            epsabs=2.0e-14,
        )[0]
        second = quad(
            lambda x: x * x * _normal_density(x, mean, variance),
            lower,
            upper,
            epsabs=2.0e-14,
        )[0]
        probabilities.append(probability)
        if probability > 1.0e-15:
            conditioned_variance += second - first * first / probability
    return np.asarray(probabilities), conditioned_variance / sum(probabilities)


@pytest.mark.parametrize(
    ("db", "expected"),
    [(0.0, 0.5), (3.0, 0.5 * 10.0**-0.3), (10.0, 0.05), (20.0, 0.005)],
)
def test_squeezing_db_conversion(db: float, expected: float) -> None:
    assert squeezing_db_to_peak_variance(db) == pytest.approx(expected)


@pytest.mark.parametrize("db", [3.0, 8.0, 10.0, 12.0, 20.0])
def test_projector_delta_reconstructs_peak_variance(db: float) -> None:
    delta = projector_delta_from_squeezing_db(db)
    assert np.tanh(delta * delta) / 2.0 == pytest.approx(
        squeezing_db_to_peak_variance(db), rel=2.0e-15
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: squeezing_db_to_peak_variance(-1.0), "squeezing_db"),
        (lambda: squeezing_db_to_peak_variance(41.0), "squeezing_db"),
        (lambda: projector_delta_from_squeezing_db(0.0), "squeezing_db"),
        (lambda: gaussian_alias_statistics(0.0, 0.0, 1.0), "variance"),
        (lambda: gaussian_alias_statistics(0.0, 1.0, 0.0), "spacing"),
    ],
)
def test_scalar_validation_is_fail_closed(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()


@pytest.mark.parametrize(
    ("mean", "variance"),
    [(0.0, 0.05), (0.37, 0.18), (-1.4, 0.7), (4.2, 1.3)],
)
def test_alias_probabilities_match_independent_quadrature(
    mean: float, variance: float
) -> None:
    result = gaussian_alias_statistics(mean, variance, LATTICE_CONST)
    independent, conditioned = _independent_cell_integrals(
        mean, variance, LATTICE_CONST, result.alias_indices
    )
    assert result.alias_probabilities == pytest.approx(independent, abs=2.0e-13)
    assert result.domain_conditioned_variance == pytest.approx(
        conditioned, abs=2.0e-12
    )
    assert result.probability_sum == pytest.approx(1.0, abs=2.0e-12)


def test_alias_symmetry_and_translation_covariance() -> None:
    positive = gaussian_alias_statistics(0.31, 0.24, LATTICE_CONST)
    negative = gaussian_alias_statistics(-0.31, 0.24, LATTICE_CONST)
    translated = gaussian_alias_statistics(
        0.31 + 2.0 * LATTICE_CONST, 0.24, LATTICE_CONST
    )
    positive_map = dict(zip(positive.alias_indices, positive.alias_probabilities))
    negative_map = dict(zip(negative.alias_indices, negative.alias_probabilities))
    translated_map = dict(zip(translated.alias_indices, translated.alias_probabilities))
    for index, probability in positive_map.items():
        assert negative_map.get(-index, 0.0) == pytest.approx(probability, abs=2.0e-15)
        assert translated_map.get(index + 2, 0.0) == pytest.approx(
            probability, abs=2.0e-15
        )
    assert translated.odd_alias_probability == pytest.approx(
        positive.odd_alias_probability, abs=2.0e-15
    )
    assert translated.ideal_center_folded_variance == pytest.approx(
        positive.ideal_center_folded_variance, abs=2.0e-14
    )


def test_odd_alias_probability_matches_large_monte_carlo() -> None:
    result = gaussian_alias_statistics(0.23, 0.31, LATTICE_CONST)
    rng = np.random.default_rng(2381)
    samples = rng.normal(0.23, sqrt(0.31), size=400_000)
    aliases = np.floor(samples / LATTICE_CONST + 0.5).astype(np.int64)
    empirical = float(np.mean(np.mod(np.abs(aliases), 2) == 1))
    standard_error = sqrt(
        result.odd_alias_probability * (1.0 - result.odd_alias_probability)
        / samples.size
    )
    assert abs(empirical - result.odd_alias_probability) < 5.0 * standard_error


def test_clipping_ratio_separates_narrow_and_broad_noise() -> None:
    narrow = gaussian_alias_statistics(0.0, 0.05, LATTICE_CONST)
    broad = gaussian_alias_statistics(0.0, 0.65, LATTICE_CONST)
    assert narrow.clipping_ratio > 0.99
    assert broad.clipping_ratio < 0.6
    assert broad.domain_conditioned_variance < broad.variance


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resource_covariance": ((1.0, 2.0), (0.0, 1.0))},
        {"resource_covariance": ((1.0, 2.0), (2.0, 1.0))},
        {"loss_transmissivity": -0.1},
        {"measurement_efficiency": 0.0},
        {"feedforward_gain": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))},
        {"scope": "silent claim promotion"},
    ],
)
def test_config_validation_rejects_invalid_or_scope_promoting_values(kwargs) -> None:
    with pytest.raises(ValueError):
        NoiseTransferConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lattice_index": (0.5, 1)},
        {"signal_offset": (0.0, np.nan)},
        {"fluctuation_covariance": ((1.0, 2.0), (2.0, 1.0))},
        {"logical_parity": (0, 2)},
    ],
)
def test_state_validation_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        NoiseTransferState(**kwargs)


def test_loss_propagation_has_analytic_signal_and_covariance() -> None:
    config = NoiseTransferConfig(
        resource_covariance=((0.07, 0.01), (0.01, 0.09)),
        loss_transmissivity=0.81,
        measurement_efficiency=1.0,
        feedforward_gain=((0.7, 0.1), (-0.2, 0.8)),
    )
    state = NoiseTransferState(
        lattice_index=(2, -1),
        signal_offset=(0.2, -0.3),
        fluctuation_covariance=((0.3, 0.08), (0.08, 0.2)),
    )
    result = GKPNoiseTransferSurrogate(config).propagate(state)
    lattice = LATTICE_CONST * np.asarray([2.0, -1.0])
    expected_bias = 0.9 * (lattice + np.asarray([0.2, -0.3])) - lattice
    expected_post_loss = (
        0.81 * np.asarray(state.fluctuation_covariance)
        + 0.19 * config.vacuum_variance * np.eye(2)
    )
    gain = np.asarray(config.feedforward_gain)
    residual = np.eye(2) - gain
    expected_output = residual @ expected_post_loss @ residual.T + gain @ np.asarray(config.resource_covariance) @ gain.T
    assert result.loss_bias == pytest.approx(expected_bias)
    assert result.post_loss_covariance == pytest.approx(expected_post_loss)
    assert result.output_signal_offset == pytest.approx(residual @ expected_bias)
    assert result.output_covariance == pytest.approx(expected_output)


def test_unity_gain_refreshes_input_noise_but_not_measurement_noise() -> None:
    config = NoiseTransferConfig(
        resource_covariance=((0.04, 0.01), (0.01, 0.06)),
        measurement_efficiency=0.8,
    )
    first = GKPNoiseTransferSurrogate(config).propagate(
        NoiseTransferState(fluctuation_covariance=((0.4, 0.12), (0.12, 0.3)))
    )
    second = GKPNoiseTransferSurrogate(config).propagate(
        NoiseTransferState(fluctuation_covariance=((0.08, -0.02), (-0.02, 0.11)))
    )
    assert np.asarray(first.output_covariance) == pytest.approx(
        np.asarray(first.measurement_equivalent_covariance)
    )
    assert np.asarray(second.output_covariance) == pytest.approx(
        np.asarray(second.measurement_equivalent_covariance)
    )
    assert np.asarray(first.output_covariance) == pytest.approx(
        np.asarray(second.output_covariance)
    )


def test_zero_gain_passes_post_loss_noise_without_resource_injection() -> None:
    config = NoiseTransferConfig(feedforward_gain=((0.0, 0.0), (0.0, 0.0)))
    result = GKPNoiseTransferSurrogate(config).propagate(NoiseTransferState())
    assert np.asarray(result.output_covariance) == pytest.approx(
        np.asarray(result.post_loss_covariance)
    )
    assert result.output_signal_offset == pytest.approx(result.loss_bias)


def test_measurement_inefficiency_adds_vacuum_equivalent_noise() -> None:
    efficient = GKPNoiseTransferSurrogate(
        NoiseTransferConfig(measurement_efficiency=1.0)
    ).propagate(NoiseTransferState())
    inefficient = GKPNoiseTransferSurrogate(
        NoiseTransferConfig(measurement_efficiency=0.5)
    ).propagate(NoiseTransferState())
    difference = np.asarray(inefficient.measurement_equivalent_covariance) - np.asarray(
        efficient.measurement_equivalent_covariance
    )
    assert difference == pytest.approx(
        NoiseTransferConfig().vacuum_variance * np.eye(2)
    )


def test_diagonal_decision_covariance_has_exact_pauli_product_law() -> None:
    result = GKPNoiseTransferSurrogate(NoiseTransferConfig()).propagate(
        NoiseTransferState()
    )
    jump = result.logical_jump
    assert jump.joint_rule.startswith("exact_axis_independence")
    assert sum(
        item
        for item in (
            jump.pauli_i_probability,
            jump.pauli_x_probability,
            jump.pauli_z_probability,
            jump.pauli_y_probability,
        )
        if item is not None
    ) == pytest.approx(1.0)
    assert jump.any_jump_probability == pytest.approx(
        1.0 - float(jump.pauli_i_probability)
    )


def test_correlated_decision_covariance_reports_only_safe_bounds() -> None:
    result = GKPNoiseTransferSurrogate(
        NoiseTransferConfig(resource_covariance=((0.08, 0.03), (0.03, 0.07)))
    ).propagate(
        NoiseTransferState(fluctuation_covariance=((0.12, -0.01), (-0.01, 0.1)))
    )
    jump = result.logical_jump
    assert jump.joint_rule.startswith("correlated_axes")
    assert jump.any_jump_probability is None
    assert jump.pauli_i_probability is None
    assert jump.any_jump_lower_bound <= jump.any_jump_upper_bound
    assert jump.any_jump_lower_bound == pytest.approx(
        max(jump.q_odd_probability, jump.p_odd_probability)
    )


def test_sampling_is_reproducible_and_residual_stays_in_voronoi_cell() -> None:
    model = GKPNoiseTransferSurrogate(NoiseTransferConfig())
    result = model.propagate(NoiseTransferState(logical_parity=(1, 0)))
    first = model.sample_step(result, np.random.default_rng(99))
    second = model.sample_step(result, np.random.default_rng(99))
    assert first == second
    assert all(abs(item) <= LATTICE_CONST / 2.0 for item in first.modular_residual)
    assert first.output_logical_parity == tuple(
        a ^ b for a, b in zip((1, 0), first.parity_jump)
    )


def test_squeezing_sweep_has_expected_validity_boundary() -> None:
    results = []
    for db in (3.0, 5.0, 8.0, 10.0, 12.0):
        variance = squeezing_db_to_peak_variance(db)
        result = GKPNoiseTransferSurrogate(
            NoiseTransferConfig(
                resource_covariance=((variance, 0.0), (0.0, variance)),
                loss_transmissivity=0.99,
                measurement_efficiency=0.97,
            )
        ).propagate(
            NoiseTransferState(
                fluctuation_covariance=((variance, 0.0), (0.0, variance))
            )
        )
        results.append(result)
    odd = [item.logical_jump.q_odd_probability for item in results]
    assert all(odd[index] > odd[index + 1] for index in range(len(odd) - 1))
    assert results[0].validity == "clipping_dominated"
    assert results[-2].validity == results[-1].validity == "localized"


@pytest.mark.parametrize("db", [10.0, 12.0])
def test_high_squeezing_state_and_fock_q_moments_align(db: float) -> None:
    point = fock_q_variance_alignment(
        db,
        cutoff=48,
        projection_grid_points=4097,
        quadrature_grid_points=8193,
    )
    assert point.maximum_proxy_to_direct_relative_error < 0.03
    assert point.maximum_fock_to_direct_relative_error < 0.08
    assert min(point.captured_probabilities) > 0.99


def test_low_squeezing_alignment_exposes_state_dependent_clipping() -> None:
    point = fock_q_variance_alignment(
        3.0,
        cutoff=30,
        projection_grid_points=4097,
        quadrature_grid_points=8193,
    )
    assert point.direct_state_relative_spread > 0.2
    assert point.maximum_proxy_to_direct_relative_error > 0.2
    assert point.maximum_fock_to_direct_relative_error < 0.01


def test_production_validation_and_writer(tmp_path) -> None:
    result = run_noise_transfer_validation(monte_carlo_samples=50_000, seed=2382)
    assert result.passed
    assert len(result.squeezing_sweep) == 5
    assert len(result.fock_alignment) == 3
    assert result.scope == NOISE_TRANSFER_SCOPE
    output = write_noise_transfer_validation(result, tmp_path / "validation.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert all(payload["checks"].values())


def test_validation_rejects_demo_sized_monte_carlo() -> None:
    with pytest.raises(ValueError, match="50000"):
        run_noise_transfer_validation(monte_carlo_samples=49_999)
