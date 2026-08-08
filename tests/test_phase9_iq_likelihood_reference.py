from __future__ import annotations

from dataclasses import replace
import inspect
from math import exp, log, pi

import numpy as np
import pytest

from physics.phase9_backend_a import (
    BackendAConfig,
    Phase9BackendASimulator,
    backend_a_exogenous,
    diagnostic_action_word,
)
from physics.phase9_backend_b import (
    BackendBConfig,
    Phase9BackendBSimulator,
    backend_b_random_record,
    diagnostic_action_word_b,
)
from physics.phase9_iq_likelihood_reference import (
    INTEGRATION_CONVENTION,
    IQObservationReceipt,
    RAW_BASE_MEASURE,
    REFERENCE_ID,
    SIGMA_CONVENTION,
    affine_log_density_correction,
    component_log_likelihoods,
    evaluate_observation,
    evidence_and_posterior,
    integrated_marginal_cdf,
    integrated_mean_gain_jacobian,
    integrated_predictive_moments,
    pairwise_log_likelihood_ratios,
    per_complex_sample_log_score,
    residual_decomposed_log_likelihoods,
)


PRIORS = (0.2, 0.5, 0.3)
CENTERS = ((-0.8, 0.1), (0.7, -0.2), (0.05, 1.1))
SIGMA = 0.48
IQ_I = (-0.7, -0.2, 0.4, 0.9)
IQ_Q = (0.3, -0.4, 0.8, 0.1)


def test_reference_import_graph_is_independent_of_backends_and_rng() -> None:
    import physics.phase9_iq_likelihood_reference as reference

    source = inspect.getsource(reference)
    assert "phase9_backend_a" not in source
    assert "phase9_backend_b" not in source
    assert "numpy" not in source
    assert "random" not in source


def test_raw_likelihood_matches_manual_complete_density() -> None:
    observed = component_log_likelihoods(
        IQ_I,
        IQ_Q,
        centers=CENTERS,
        sigma=SIGMA,
    )
    expected = []
    for center_i, center_q in CENTERS:
        squared = sum(
            (sample_i - center_i) ** 2 + (sample_q - center_q) ** 2
            for sample_i, sample_q in zip(IQ_I, IQ_Q)
        )
        expected.append(
            -len(IQ_I) * log(2.0 * pi * SIGMA * SIGMA)
            - squared / (2.0 * SIGMA * SIGMA)
        )
    assert observed == pytest.approx(expected, abs=1.0e-14)
    assert residual_decomposed_log_likelihoods(
        IQ_I,
        IQ_Q,
        centers=CENTERS,
        sigma=SIGMA,
    ) == pytest.approx(observed, abs=1.0e-14)


def test_evidence_posterior_llr_and_receipt_are_complete_and_bound() -> None:
    receipt = evaluate_observation(
        IQ_I,
        IQ_Q,
        priors=PRIORS,
        centers=CENTERS,
        sigma=SIGMA,
    )
    weighted = [
        prior * exp(value)
        for prior, value in zip(
            PRIORS,
            receipt.component_log_likelihoods,
        )
    ]
    expected_evidence = log(sum(weighted))
    assert receipt.reference_id == REFERENCE_ID
    assert receipt.log_evidence == pytest.approx(expected_evidence, abs=1.0e-14)
    assert receipt.posterior == pytest.approx(
        [value / sum(weighted) for value in weighted],
        abs=1.0e-14,
    )
    assert receipt.pairwise_llr == pairwise_log_likelihood_ratios(
        receipt.component_log_likelihoods
    )
    assert len(receipt.semantic_sha256()) == 64
    changed = evaluate_observation(
        IQ_I[:-1] + (1.0,),
        IQ_Q,
        priors=PRIORS,
        centers=CENTERS,
        sigma=SIGMA,
    )
    assert changed.input_sha256 != receipt.input_sha256
    assert changed.semantic_sha256() != receipt.semantic_sha256()


def test_integrated_predictive_moments_include_shared_label_covariance() -> None:
    mean, covariance = integrated_predictive_moments(
        priors=PRIORS,
        centers=CENTERS,
        sigma=SIGMA,
        sample_count=4,
    )
    expected_mean = np.average(
        np.asarray(CENTERS),
        axis=0,
        weights=np.asarray(PRIORS),
    )
    centered = np.asarray(CENTERS) - expected_mean
    expected_covariance = (
        centered.T @ (np.asarray(PRIORS)[:, None] * centered)
        + (SIGMA * SIGMA / 4.0) * np.eye(2)
    )
    assert mean == pytest.approx(expected_mean, abs=1.0e-14)
    assert np.asarray(covariance) == pytest.approx(
        expected_covariance,
        abs=1.0e-14,
    )
    # A per-sample latent-label mutation would incorrectly divide this term.
    wrong = (
        centered.T @ (np.asarray(PRIORS)[:, None] * centered)
        + SIGMA * SIGMA * np.eye(2)
    ) / 4.0
    assert not np.allclose(covariance, wrong)


def test_integrated_marginal_cdf_uses_sigma_over_sqrt_sample_count() -> None:
    value = 0.15
    observed = integrated_marginal_cdf(
        value,
        axis=0,
        priors=PRIORS,
        centers=CENTERS,
        sigma=SIGMA,
        sample_count=16,
    )
    from math import erf, sqrt

    correct = sum(
        prior
        * 0.5
        * (
            1.0
            + erf(
                (value - center[0])
                / ((SIGMA / sqrt(16.0)) * sqrt(2.0))
            )
        )
        for prior, center in zip(PRIORS, CENTERS)
    )
    wrong_sigma = sum(
        prior
        * 0.5
        * (1.0 + erf((value - center[0]) / (SIGMA * sqrt(2.0))))
        for prior, center in zip(PRIORS, CENTERS)
    )
    assert observed == pytest.approx(correct, abs=1.0e-15)
    assert abs(observed - wrong_sigma) > 5.0e-3


def test_gain_jacobian_matches_finite_difference() -> None:
    analytic = integrated_mean_gain_jacobian(
        priors=PRIORS,
        centers=CENTERS,
    )
    epsilon = 1.0e-6
    plus, _ = integrated_predictive_moments(
        priors=PRIORS,
        centers=tuple(
            (epsilon * center_i + center_i, epsilon * center_q + center_q)
            for center_i, center_q in CENTERS
        ),
        sigma=SIGMA,
        sample_count=8,
    )
    minus, _ = integrated_predictive_moments(
        priors=PRIORS,
        centers=tuple(
            (center_i - epsilon * center_i, center_q - epsilon * center_q)
            for center_i, center_q in CENTERS
        ),
        sigma=SIGMA,
        sample_count=8,
    )
    finite_difference = tuple(
        (right - left) / (2.0 * epsilon)
        for right, left in zip(plus, minus)
    )
    assert analytic == pytest.approx(finite_difference, rel=2.0e-10)


def test_affine_gain_has_exact_two_dimensional_base_measure_jacobian() -> None:
    count = 8
    scalar_gain = 1.25
    observed = affine_log_density_correction(
        sample_count=count,
        gain_matrix=((scalar_gain, 0.0), (0.0, scalar_gain)),
    )
    assert observed == pytest.approx(-2.0 * count * log(scalar_gain))
    assert affine_log_density_correction(
        sample_count=count,
        gain_matrix=((1.0, 0.2), (-0.1, 0.9)),
    ) == pytest.approx(-count * log(0.92))
    assert per_complex_sample_log_score(-12.0, sample_count=8) == -1.5
    with pytest.raises(ValueError, match="nonsingular"):
        affine_log_density_correction(
            sample_count=count,
            gain_matrix=((1.0, 2.0), (2.0, 4.0)),
        )
    with pytest.raises(TypeError):
        affine_log_density_correction(
            sample_count=True,
            gain_matrix=((1.0, 0.0), (0.0, 1.0)),
        )


def test_scalar_gain_and_offset_transform_raw_density_exactly_once() -> None:
    gain = 1.25
    offset = (-0.31, 0.42)
    transformed_i = tuple(gain * value + offset[0] for value in IQ_I)
    transformed_q = tuple(gain * value + offset[1] for value in IQ_Q)
    transformed_centers = tuple(
        (gain * center_i + offset[0], gain * center_q + offset[1])
        for center_i, center_q in CENTERS
    )
    original = component_log_likelihoods(
        IQ_I, IQ_Q, centers=CENTERS, sigma=SIGMA
    )
    transformed = component_log_likelihoods(
        transformed_i,
        transformed_q,
        centers=transformed_centers,
        sigma=gain * SIGMA,
    )
    correction = affine_log_density_correction(
        sample_count=len(IQ_I),
        gain_matrix=((gain, 0.0), (0.0, gain)),
    )
    assert transformed == pytest.approx(
        [value + correction for value in original], abs=2.0e-14
    )
    # LLRs are invariant because the common base-measure term cancels.
    assert np.asarray(pairwise_log_likelihood_ratios(transformed)) == pytest.approx(
        np.asarray(pairwise_log_likelihood_ratios(original)), abs=2.0e-14
    )


def test_receipt_freezes_units_sigma_and_integration_semantics() -> None:
    receipt = evaluate_observation(
        IQ_I,
        IQ_Q,
        priors=PRIORS,
        centers=CENTERS,
        sigma=SIGMA,
    )
    assert receipt.raw_base_measure == RAW_BASE_MEASURE
    assert receipt.sigma_convention == SIGMA_CONVENTION
    assert receipt.integration_convention == INTEGRATION_CONVENTION
    with pytest.raises(ValueError, match="raw_base_measure"):
        replace(receipt, raw_base_measure="adc_codes")
    with pytest.raises(ValueError, match="sigma_convention"):
        replace(receipt, sigma_convention="complex_variance")
    with pytest.raises(ValueError, match="integration_convention"):
        replace(receipt, integration_convention="sum")


def test_label_permutation_is_evidence_invariant_and_posterior_equivariant() -> None:
    permutation = (2, 0, 1)
    original = evaluate_observation(
        IQ_I,
        IQ_Q,
        priors=PRIORS,
        centers=CENTERS,
        sigma=SIGMA,
    )
    permuted = evaluate_observation(
        IQ_I,
        IQ_Q,
        priors=tuple(PRIORS[index] for index in permutation),
        centers=tuple(CENTERS[index] for index in permutation),
        sigma=SIGMA,
    )
    assert permuted.log_evidence == pytest.approx(
        original.log_evidence,
        abs=1.0e-14,
    )
    assert permuted.component_log_likelihoods == pytest.approx(
        [original.component_log_likelihoods[index] for index in permutation]
    )
    assert permuted.posterior == pytest.approx(
        [original.posterior[index] for index in permutation]
    )


def test_mutated_normalization_sample_count_and_factor_two_are_detected() -> None:
    correct = component_log_likelihoods(
        IQ_I,
        IQ_Q,
        centers=CENTERS,
        sigma=SIGMA,
    )
    center_i, center_q = CENTERS[0]
    squared = sum(
        (sample_i - center_i) ** 2 + (sample_q - center_q) ** 2
        for sample_i, sample_q in zip(IQ_I, IQ_Q)
    )
    missing_normalization = -squared / (2.0 * SIGMA * SIGMA)
    wrong_sample_count = (
        -log(2.0 * pi * SIGMA * SIGMA)
        - squared / (2.0 * SIGMA * SIGMA)
    )
    wrong_factor_two = (
        -len(IQ_I) * log(2.0 * pi * SIGMA * SIGMA)
        - squared / (SIGMA * SIGMA)
    )
    assert abs(correct[0] - missing_normalization) > 0.1
    assert abs(correct[0] - wrong_sample_count) > 0.1
    assert abs(correct[0] - wrong_factor_two) > 0.1


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("sigma", True, TypeError),
        ("sigma", 0.0, ValueError),
        ("sigma", float("nan"), ValueError),
        ("priors", (0.2, 0.2, 0.2), ValueError),
        ("priors", (0.2, -0.1, 0.9), ValueError),
        ("iq_i", (0.0, float("inf")), ValueError),
    ],
)
def test_strict_invalid_inputs_are_rejected(field, value, exception) -> None:
    kwargs = {
        "iq_i": IQ_I,
        "iq_q": IQ_Q,
        "priors": PRIORS,
        "centers": CENTERS,
        "sigma": SIGMA,
    }
    kwargs[field] = value
    with pytest.raises(exception):
        evaluate_observation(**kwargs)


def test_receipt_rejects_bool_count_nonfinite_and_bad_hash() -> None:
    valid = evaluate_observation(
        IQ_I,
        IQ_Q,
        priors=PRIORS,
        centers=CENTERS,
        sigma=SIGMA,
    )
    with pytest.raises(TypeError):
        replace(valid, sample_count=True)
    with pytest.raises(ValueError):
        replace(valid, log_evidence=float("nan"))
    with pytest.raises(ValueError):
        replace(valid, input_sha256="0" * 63)


def _assert_backend_observation_matches_reference(
    *,
    iq_i,
    iq_q,
    priors,
    centers,
    sigma,
    expected_log_evidence,
    expected_posterior,
) -> None:
    receipt = evaluate_observation(
        iq_i,
        iq_q,
        priors=priors,
        centers=centers,
        sigma=sigma,
    )
    assert receipt.log_evidence == pytest.approx(
        expected_log_evidence,
        abs=2.0e-12,
    )
    assert receipt.posterior == pytest.approx(
        expected_posterior,
        abs=2.0e-12,
    )


def test_backend_a_real_step_matches_independent_reference() -> None:
    config = BackendAConfig(cutoff=8, substeps_per_segment=2, iq_samples=4)
    simulator = Phase9BackendASimulator(config)
    result = simulator.step(
        simulator.initialize_fock(),
        diagnostic_action_word("IDLE"),
        backend_a_exogenous(seed=81001, round_index=0, iq_samples=4),
        evaluator=None,
    )
    _assert_backend_observation_matches_reference(
        iq_i=result.observation.iq_i,
        iq_q=result.observation.iq_q,
        priors=result.truth.pre_measurement_level_probabilities,
        centers=config.iq_centers,
        sigma=config.iq_sigma,
        expected_log_evidence=result.observation.log_evidence_density,
        expected_posterior=result.observation.posterior_levels,
    )


def test_backend_b_real_step_matches_independent_reference() -> None:
    config = BackendBConfig(
        cutoff=8,
        split_steps_per_segment=2,
        iq_samples=4,
    )
    simulator = Phase9BackendBSimulator(config)
    result = simulator.step(
        simulator.initialize_fock(),
        diagnostic_action_word_b("IDLE"),
        backend_b_random_record(seed=82001, round_index=0, iq_samples=4),
        evaluator=None,
    )
    _assert_backend_observation_matches_reference(
        iq_i=result.observation.iq_i,
        iq_q=result.observation.iq_q,
        priors=result.truth.pre_measurement_levels,
        centers=config.iq_centers,
        sigma=config.iq_sigma,
        expected_log_evidence=result.observation.log_evidence_density,
        expected_posterior=result.observation.posterior_levels,
    )


def test_evidence_with_zero_prior_is_finite_and_normalized() -> None:
    logs = (-3.0, -4.0, -5.0)
    evidence, posterior = evidence_and_posterior(
        logs,
        priors=(1.0, 0.0, 0.0),
    )
    assert evidence == -3.0
    assert posterior == (1.0, 0.0, 0.0)
