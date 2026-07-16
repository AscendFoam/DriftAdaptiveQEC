from __future__ import annotations

from inspect import signature

import numpy as np
import pytest

from cnn_fpga.decoder.periodic_adaptive_map import (
    ConstantVelocityPeriodicKalman,
    LatestWindowPeriodicPredictor,
    PeriodicKalmanConfig,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    estimate_periodic_gaussian,
    periodic_characteristic_features,
    scaled_periodic_kalman_config,
    validate_residual_window,
)
from physics.constants import LATTICE_CONST


def _wrapped_gaussian(
    seed: int,
    samples: int,
    *,
    mean_fraction: tuple[float, float] = (0.12, -0.08),
    sigma_fraction: tuple[float, float] = (0.16, 0.12),
    rho: float = 0.45,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = np.asarray(mean_fraction) * LATTICE_CONST
    sq, sp = np.asarray(sigma_fraction) * LATTICE_CONST
    covariance = np.asarray([[sq * sq, rho * sq * sp], [rho * sq * sp, sp * sp]])
    values = rng.multivariate_normal(mean, covariance, size=samples)
    return (values + LATTICE_CONST / 2.0) % LATTICE_CONST - LATTICE_CONST / 2.0


def test_periodic_moments_recover_full_correlated_gaussian() -> None:
    residual = _wrapped_gaussian(10, 120_000)
    estimate = estimate_periodic_gaussian(residual)
    expected_mean = np.asarray([0.12, -0.08]) * LATTICE_CONST
    sq, sp = np.asarray([0.16, 0.12]) * LATTICE_CONST
    expected_covariance = np.asarray(
        [[sq * sq, 0.45 * sq * sp], [0.45 * sq * sp, sp * sp]]
    )
    mean_error = (estimate.mean_array() - expected_mean + LATTICE_CONST / 2.0) % (
        LATTICE_CONST
    ) - LATTICE_CONST / 2.0
    assert np.max(np.abs(mean_error)) < 0.0025 * LATTICE_CONST
    assert np.max(np.abs(estimate.covariance_array() - expected_covariance)) < (
        0.0012 * LATTICE_CONST**2
    )
    assert estimate.rho == pytest.approx(0.45, abs=0.025)
    assert estimate.joint_covariance_discrepancy < 0.0015 * LATTICE_CONST**2


def test_periodic_features_are_invariant_to_integer_lattice_translations() -> None:
    residual = _wrapped_gaussian(11, 4096)
    translated = residual + np.asarray([3.0, -2.0]) * LATTICE_CONST
    translated = (translated + LATTICE_CONST / 2.0) % LATTICE_CONST - LATTICE_CONST / 2.0
    config = PeriodicMomentConfig()
    assert np.allclose(
        periodic_characteristic_features(residual, config),
        periodic_characteristic_features(translated, config),
        rtol=0.0,
        atol=2e-15,
    )


def test_correlation_is_not_factorized_or_silently_zeroed() -> None:
    positive = estimate_periodic_gaussian(_wrapped_gaussian(12, 80_000, rho=0.75))
    negative = estimate_periodic_gaussian(_wrapped_gaussian(13, 80_000, rho=-0.65))
    assert positive.rho > 0.70
    assert negative.rho < -0.60
    assert abs(positive.covariance_array()[0, 1]) > 0.01 * LATTICE_CONST**2
    assert abs(negative.covariance_array()[0, 1]) > 0.01 * LATTICE_CONST**2


def test_latest_window_update_does_not_retroactively_change_prior_prediction() -> None:
    calibration = _wrapped_gaussian(14, 1024, mean_fraction=(-0.15, 0.1))
    current = _wrapped_gaussian(15, 1024, mean_fraction=(0.2, -0.12))
    predictor = LatestWindowPeriodicPredictor(calibration)
    before = predictor.prediction()
    predictor.update(current, window_id=0)
    after = predictor.prediction()
    assert before.window_id == -1
    assert after.window_id == 0
    assert np.linalg.norm(before.mean_array() - after.mean_array()) > 0.2 * LATTICE_CONST


def test_ewma_updates_the_four_complex_features_exactly() -> None:
    calibration = _wrapped_gaussian(16, 2048, mean_fraction=(-0.1, 0.08))
    current = _wrapped_gaussian(17, 2048, mean_fraction=(0.18, -0.12))
    config = PeriodicMomentConfig()
    predictor = PeriodicMomentEWMA(calibration, alpha=0.35, config=config)
    initial = predictor.feature_state
    observation = periodic_characteristic_features(current, config)
    predictor.update(current, window_id=2)
    assert np.allclose(
        predictor.feature_state,
        0.65 * initial + 0.35 * observation,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert predictor.prediction().window_id == 2


def test_alpha_one_ewma_matches_latest_window_periodic_moments() -> None:
    calibration = _wrapped_gaussian(18, 1024)
    current = _wrapped_gaussian(19, 1024, mean_fraction=(0.22, -0.18), rho=-0.4)
    ewma = PeriodicMomentEWMA(calibration, alpha=1.0)
    ewma.update(current, window_id=0)
    direct = estimate_periodic_gaussian(current, window_id=0)
    assert np.allclose(ewma.prediction().mean_array(), direct.mean_array(), atol=1.0e-15)
    assert np.allclose(
        ewma.prediction().covariance_array(), direct.covariance_array(), atol=1.0e-15
    )


def test_kalman_state_covariance_and_prediction_remain_finite_spd() -> None:
    calibration = _wrapped_gaussian(20, 2048, mean_fraction=(-0.18, 0.12))
    predictor = ConstantVelocityPeriodicKalman(
        calibration,
        kalman_config=scaled_periodic_kalman_config(
            process_scale=2.0, measurement_scale=0.75
        ),
    )
    for window in range(10):
        mean = (-0.18 + 0.025 * window, 0.12 - 0.018 * window)
        predictor.update(
            _wrapped_gaussian(100 + window, 2048, mean_fraction=mean),
            window_id=window,
        )
        prediction = predictor.prediction()
        assert np.all(np.isfinite(prediction.mean_array()))
        assert np.all(np.isfinite(prediction.covariance_array()))
        assert np.min(np.linalg.eigvalsh(prediction.covariance_array())) > 0.0
        assert np.min(np.linalg.eigvalsh(predictor.covariance)) > 0.0
        assert predictor.kalman_gain.shape == (10, 5)
        assert all(np.isfinite(prediction.resultants))


def test_constant_velocity_kalman_anticipates_linear_mean_better_than_latest_window() -> None:
    calibration = _wrapped_gaussian(30, 4096, mean_fraction=(-0.22, 0.12))
    latest = LatestWindowPeriodicPredictor(calibration)
    kalman = ConstantVelocityPeriodicKalman(
        calibration,
        kalman_config=scaled_periodic_kalman_config(
            process_scale=2.0, measurement_scale=0.75
        ),
    )
    for window in range(12):
        mean = (-0.22 + 0.025 * window, 0.12 - 0.015 * window)
        residual = _wrapped_gaussian(200 + window, 4096, mean_fraction=mean)
        latest.update(residual, window_id=window)
        kalman.update(residual, window_id=window)
    next_truth = np.asarray([-0.22 + 0.025 * 12, 0.12 - 0.015 * 12]) * LATTICE_CONST
    latest_error = np.linalg.norm(latest.prediction().mean_array() - next_truth)
    kalman_error = np.linalg.norm(kalman.prediction().mean_array() - next_truth)
    assert kalman_error < latest_error


def test_kalman_unwraps_mean_across_periodic_boundary_without_unit_cell_jump() -> None:
    calibration = _wrapped_gaussian(40, 4096, mean_fraction=(0.44, -0.1))
    kalman = ConstantVelocityPeriodicKalman(calibration)
    kalman.update(
        _wrapped_gaussian(41, 4096, mean_fraction=(-0.44, -0.1)),
        window_id=0,
    )
    # The internal unwrapped state follows the short +0.12L motion, not a -0.88L jump.
    assert kalman.state[0] > 0.40 * LATTICE_CONST


def test_online_predictor_signatures_expose_no_truth_or_drift_state() -> None:
    for method in (
        PeriodicMomentEWMA.update,
        ConstantVelocityPeriodicKalman.update,
        LatestWindowPeriodicPredictor.update,
    ):
        parameters = set(signature(method).parameters)
        assert parameters == {"self", "residuals", "window_id"}
        assert not parameters & {"truth", "state", "drift_state", "logical_class"}


@pytest.mark.parametrize(
    "call",
    [
        lambda: PeriodicMomentConfig(minimum_samples=7),
        lambda: PeriodicMomentConfig(variance_floor=1.0, variance_ceiling=0.5),
        lambda: PeriodicMomentConfig(resultant_floor=1.0),
        lambda: PeriodicMomentConfig(rho_clip=1.0),
        lambda: PeriodicMomentConfig(covariance_shrinkage=1.0),
        lambda: PeriodicKalmanConfig(process_std_position=(1.0,) * 4),
        lambda: PeriodicKalmanConfig(velocity_decay=1.1),
        lambda: scaled_periodic_kalman_config(process_scale=0.0, measurement_scale=1.0),
    ],
)
def test_configs_fail_closed(call) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()


def test_residual_validation_rejects_shape_nonfinite_range_and_short_windows() -> None:
    config = PeriodicMomentConfig(minimum_samples=64)
    with pytest.raises(ValueError, match="shape"):
        validate_residual_window(np.zeros((64, 3)), config)
    with pytest.raises(ValueError, match="fewer"):
        validate_residual_window(np.zeros((63, 2)), config)
    bad = np.zeros((64, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_residual_window(bad, config)
    with pytest.raises(ValueError, match="lattice/2"):
        validate_residual_window(np.full((64, 2), LATTICE_CONST / 2.0), config)
