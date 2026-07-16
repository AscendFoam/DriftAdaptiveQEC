from __future__ import annotations

from dataclasses import replace
import json
from math import sqrt, tanh
from pathlib import Path

import numpy as np
import pytest

from physics.constants import LATTICE_CONST
from physics.finite_energy_gkp import damped_projector_state
from physics.finite_squeezing_noise import (
    FiniteSqueezingNoiseConfig,
    envelope_index_distribution,
    finite_squeezing_noise_budget,
    isolated_peak_variance,
    run_high_squeezing_limit_sweep,
    sample_finite_squeezing_noise,
    write_finite_squeezing_report,
)
from physics.ideal_gkp_decoder import gaussian_logical_flip_probability


def _zero_covariance() -> tuple[tuple[float, float], tuple[float, float]]:
    return ((0.0, 0.0), (0.0, 0.0))


def _quiet_config(**changes: object) -> FiniteSqueezingNoiseConfig:
    defaults: dict[str, object] = {
        "channel_covariance": _zero_covariance(),
        "data_delta": (0.0, 0.0),
        "ancilla_delta": (0.0, 0.0),
        "measurement_covariance": _zero_covariance(),
        "include_envelope": False,
        "samples": 20_000,
        "seed": 731,
    }
    defaults.update(changes)
    return FiniteSqueezingNoiseConfig(**defaults)


def test_isolated_peak_variance_is_derived_from_damped_projector_family() -> None:
    for delta in (0.18, 0.31, 0.55, 0.80):
        state = damped_projector_state("0", delta, tail_tolerance=1.0e-10)
        expected = state.amplitude_variance / 2.0
        assert isolated_peak_variance(delta) == pytest.approx(expected, abs=1.0e-15)
        assert expected == pytest.approx(tanh(delta * delta), abs=1.0e-15)
        assert isolated_peak_variance(
            delta, coordinate_chart="canonical_fock"
        ) == pytest.approx(tanh(delta * delta) / 2.0, abs=1.0e-15)
    assert isolated_peak_variance(0.0) == 0.0


def test_envelope_distribution_is_normalized_symmetric_and_non_gaussian() -> None:
    distribution = envelope_index_distribution(0.62, tail_tolerance=1.0e-15)
    assert np.sum(distribution.probabilities) == pytest.approx(1.0, abs=1.0e-15)
    assert distribution.mean_shift == pytest.approx(0.0, abs=1.0e-15)
    assert distribution.variance > 0.0
    assert distribution.contraction < 1.0
    assert distribution.captured_weight > 1.0 - 1.0e-12
    assert np.array_equal(distribution.indices, -distribution.indices[::-1])
    assert np.allclose(
        distribution.probabilities,
        distribution.probabilities[::-1],
        rtol=0.0,
        atol=1.0e-15,
    )
    positive_steps = np.diff(np.sort(distribution.shifts))
    assert np.allclose(
        positive_steps,
        positive_steps[0],
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    assert distribution.shifts.size > 5


def test_envelope_even_odd_classes_and_exact_ideal_endpoint() -> None:
    even = envelope_index_distribution(0.5, index_class="even")
    odd = envelope_index_distribution(0.5, index_class="odd")
    assert np.all(np.mod(even.indices, 2) == 0)
    assert np.all(np.mod(odd.indices, 2) == 1)
    assert odd.variance > even.variance

    ideal_all = envelope_index_distribution(0.0, index_class="all")
    ideal_odd = envelope_index_distribution(0.0, index_class="odd")
    assert ideal_all.variance == 0.0
    assert ideal_odd.variance == 0.0
    assert np.all(ideal_all.shifts == 0.0)
    assert np.all(ideal_odd.shifts == 0.0)


def test_covariance_budget_keeps_all_four_contribution_classes_separate() -> None:
    config = FiniteSqueezingNoiseConfig(samples=1_000)
    budget = finite_squeezing_noise_budget(config)
    assert np.array_equal(
        budget.physical_total,
        budget.channel + budget.data_gkp + budget.finite_energy_envelope,
    )
    assert np.array_equal(
        budget.observed_total,
        budget.physical_total + budget.ancilla_gkp + budget.measurement,
    )
    assert np.array_equal(
        budget.finite_squeezing_excess,
        budget.data_gkp + budget.ancilla_gkp + budget.finite_energy_envelope,
    )
    assert np.array_equal(budget.ideal_observed, budget.channel + budget.measurement)
    payload = budget.as_dict()
    assert set(payload["covariances"]) >= {
        "channel",
        "data_gkp",
        "ancilla_gkp",
        "measurement",
        "finite_energy_envelope",
    }


def test_zero_delta_returns_exact_ideal_budget_without_erasing_measurement() -> None:
    covariance = ((0.2, 0.04), (0.04, 0.1))
    measurement = ((0.03, -0.01), (-0.01, 0.02))
    config = _quiet_config(
        channel_covariance=covariance,
        measurement_covariance=measurement,
        include_envelope=True,
    )
    budget = finite_squeezing_noise_budget(config)
    assert np.array_equal(budget.data_gkp, np.zeros((2, 2)))
    assert np.array_equal(budget.ancilla_gkp, np.zeros((2, 2)))
    assert np.array_equal(budget.finite_energy_envelope, np.zeros((2, 2)))
    assert np.array_equal(budget.finite_squeezing_excess, np.zeros((2, 2)))
    assert np.array_equal(budget.physical_total, np.asarray(covariance))
    assert np.array_equal(
        budget.observed_total,
        np.asarray(covariance) + np.asarray(measurement),
    )


def test_sample_arrays_recompose_physical_observed_and_control_action_exactly() -> None:
    config = FiniteSqueezingNoiseConfig(samples=12_000, seed=19)
    batch = sample_finite_squeezing_noise(config)
    assert np.array_equal(
        batch.physical,
        batch.channel + batch.data_gkp + batch.finite_energy_envelope,
    )
    assert np.array_equal(
        batch.observed,
        batch.physical + batch.ancilla_gkp + batch.measurement,
    )
    manual_syndrome = batch.observed - np.floor(
        batch.observed / config.lattice + 0.5
    ) * config.lattice
    assert np.array_equal(batch.syndrome, manual_syndrome)
    assert np.array_equal(batch.correction, batch.syndrome)
    assert np.array_equal(batch.corrected_residual, batch.physical - batch.correction)
    manual_indices = np.floor(
        batch.corrected_residual / config.lattice + 0.5
    ).astype(np.int64)
    assert np.array_equal(batch.logical_parity, np.mod(manual_indices, 2))


def test_empirical_component_covariances_match_analytic_budget() -> None:
    config = FiniteSqueezingNoiseConfig(samples=260_000, seed=2021)
    batch = sample_finite_squeezing_noise(config)
    summary = batch.summary()
    pairs = {
        "channel": batch.budget.channel,
        "data_gkp": batch.budget.data_gkp,
        "ancilla_gkp": batch.budget.ancilla_gkp,
        "measurement": batch.budget.measurement,
        "finite_energy_envelope": batch.budget.finite_energy_envelope,
        "physical": batch.budget.physical_total,
        "observed": batch.budget.observed_total,
    }
    for name, expected in pairs.items():
        scale = max(float(np.linalg.norm(expected)), 1.0e-15)
        relative = float(
            np.linalg.norm(summary.empirical_covariances[name] - expected) / scale
        )
        assert relative < 0.025, (name, relative)
    assert summary.observed_covariance_relative_error < 0.015
    assert summary.physical_covariance_relative_error < 0.015


def test_channel_correlation_is_preserved_without_leaking_into_independent_terms() -> None:
    covariance = ((0.20, -0.09), (-0.09, 0.16))
    config = _quiet_config(
        channel_covariance=covariance,
        data_delta=(0.45, 0.35),
        ancilla_delta=(0.38, 0.28),
        samples=220_000,
        seed=121,
    )
    batch = sample_finite_squeezing_noise(config)
    channel_cov = np.cov(batch.channel, rowvar=False, ddof=1)
    data_cov = np.cov(batch.data_gkp, rowvar=False, ddof=1)
    ancilla_cov = np.cov(batch.ancilla_gkp, rowvar=False, ddof=1)
    assert channel_cov[0, 1] == pytest.approx(-0.09, abs=0.002)
    assert abs(data_cov[0, 1]) < 0.0015
    assert abs(ancilla_cov[0, 1]) < 0.0015


def test_envelope_ablation_changes_only_envelope_lane_under_fixed_seed() -> None:
    enabled = FiniteSqueezingNoiseConfig(
        data_delta=(0.75, 0.65),
        samples=30_000,
        seed=505,
        include_envelope=True,
    )
    with_envelope = sample_finite_squeezing_noise(enabled)
    without_envelope = sample_finite_squeezing_noise(
        replace(enabled, include_envelope=False)
    )
    assert np.array_equal(with_envelope.channel, without_envelope.channel)
    assert np.array_equal(with_envelope.data_gkp, without_envelope.data_gkp)
    assert np.array_equal(with_envelope.ancilla_gkp, without_envelope.ancilla_gkp)
    assert np.array_equal(with_envelope.measurement, without_envelope.measurement)
    assert np.any(with_envelope.finite_energy_envelope != 0.0)
    assert np.all(without_envelope.finite_energy_envelope == 0.0)
    assert not np.array_equal(with_envelope.observed, without_envelope.observed)


def test_ideal_gaussian_endpoint_matches_independent_analytic_logical_rate() -> None:
    sigma_q = 0.29 * LATTICE_CONST
    sigma_p = 0.23 * LATTICE_CONST
    config = _quiet_config(
        channel_covariance=((sigma_q**2, 0.0), (0.0, sigma_p**2)),
        samples=500_000,
        seed=808,
    )
    summary = sample_finite_squeezing_noise(config).summary()
    p_q = gaussian_logical_flip_probability(sigma_q)
    p_p = gaussian_logical_flip_probability(sigma_p)
    expected_any = 1.0 - (1.0 - p_q) * (1.0 - p_p)
    standard_error = sqrt(expected_any * (1.0 - expected_any) / config.samples)
    assert abs(summary.logical_error_rate - expected_any) < 5.0 * standard_error
    assert abs(summary.q_logical_error_rate - p_q) < 5.0 * sqrt(
        p_q * (1.0 - p_q) / config.samples
    )
    assert abs(summary.p_logical_error_rate - p_p) < 5.0 * sqrt(
        p_p * (1.0 - p_p) / config.samples
    )


def test_high_squeezing_sweep_converges_to_exact_ideal_endpoint() -> None:
    sigma_q = 0.13 * LATTICE_CONST
    sigma_p = 0.10 * LATTICE_CONST
    config = FiniteSqueezingNoiseConfig(
        channel_covariance=((sigma_q**2, 0.0), (0.0, sigma_p**2)),
        data_delta=(0.65, 0.55),
        ancilla_delta=(0.50, 0.42),
        measurement_covariance=_zero_covariance(),
        samples=160_000,
        seed=919,
    )
    result = run_high_squeezing_limit_sweep(
        config,
        scales=(1.0, 0.70, 0.40, 0.15, 0.0),
    )
    assert result.analytic_excess_strictly_decreases
    assert result.ideal_endpoint_exact
    assert result.broad_finite_squeezing_rate_above_ideal
    assert result.max_observed_covariance_relative_error < 0.025
    excess = np.array([point.finite_squeezing_excess_trace for point in result.points])
    assert np.all(np.diff(excess) < 0.0)
    assert excess[-1] == 0.0


def test_fixed_seed_replays_all_components_and_changed_seed_diverges() -> None:
    config = FiniteSqueezingNoiseConfig(samples=5_000, seed=44)
    first = sample_finite_squeezing_noise(config)
    second = sample_finite_squeezing_noise(config)
    changed = sample_finite_squeezing_noise(replace(config, seed=45))
    for name in (
        "channel",
        "data_gkp",
        "ancilla_gkp",
        "measurement",
        "finite_energy_envelope",
        "physical",
        "observed",
        "syndrome",
        "corrected_residual",
        "logical_parity",
    ):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name
    assert not np.array_equal(first.channel, changed.channel)
    assert not np.array_equal(first.logical_parity, changed.logical_parity)


def test_writer_emits_machine_readable_budget_checks_and_claim_boundary() -> None:
    config = FiniteSqueezingNoiseConfig(samples=8_000, seed=54)
    result = run_high_squeezing_limit_sweep(
        config,
        scales=(1.0, 0.5, 0.0),
    )
    output = Path("artifacts") / "test_t2_2_1_finite_squeezing.json"
    try:
        written = write_finite_squeezing_report(result, output)
        payload = json.loads(written.read_text(encoding="utf-8"))
        assert payload["checks"]["ideal_endpoint_exact"] is True
        assert payload["base_budget"]["scope"] == "decomposed_syndrome_level_effective_model"
        assert "data_gkp" in payload["base_budget"]["covariances"]
        assert "ancilla_gkp" in payload["base_budget"]["covariances"]
        assert "finite_energy_envelope" in payload["base_budget"]["covariances"]
        assert "Fock-space" in payload["claim_boundary"]["forbidden"]
    finally:
        output.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FiniteSqueezingNoiseConfig(channel_mean=(0.0,)),
        lambda: FiniteSqueezingNoiseConfig(channel_covariance=((1.0, 0.2), (0.0, 1.0))),
        lambda: FiniteSqueezingNoiseConfig(channel_covariance=((1.0, 2.0), (2.0, 1.0))),
        lambda: FiniteSqueezingNoiseConfig(measurement_covariance=((1.0, 2.0), (2.0, 1.0))),
        lambda: FiniteSqueezingNoiseConfig(data_delta=(-0.1, 0.2)),
        lambda: FiniteSqueezingNoiseConfig(ancilla_delta=(0.1, float("inf"))),
        lambda: FiniteSqueezingNoiseConfig(envelope_index_classes=("all", "bad")),
        lambda: FiniteSqueezingNoiseConfig(samples=999),
        lambda: FiniteSqueezingNoiseConfig(seed=-1),
        lambda: FiniteSqueezingNoiseConfig(lattice=0.0),
        lambda: FiniteSqueezingNoiseConfig(tail_tolerance=1.0),
    ],
)
def test_invalid_configs_fail_closed(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_invalid_envelope_and_sweep_requests_fail_closed() -> None:
    invalid_calls = [
        lambda: isolated_peak_variance(-0.1),
        lambda: envelope_index_distribution(-0.1),
        lambda: envelope_index_distribution(0.3, index_class="bad"),
        lambda: envelope_index_distribution(0.001, max_indices=3),
        lambda: finite_squeezing_noise_budget("bad"),
        lambda: sample_finite_squeezing_noise("bad"),
        lambda: run_high_squeezing_limit_sweep(scales=(1.0, 0.0)),
        lambda: run_high_squeezing_limit_sweep(scales=(1.0, 0.5, 0.6, 0.0)),
        lambda: run_high_squeezing_limit_sweep(scales=(1.0, 0.5, 0.1)),
    ]
    for call in invalid_calls:
        with pytest.raises((TypeError, ValueError)):
            call()
