from __future__ import annotations

import inspect

import numpy as np
import pytest

from cnn_fpga.decoder.periodic_adaptive_map import (
    PeriodicMomentConfig,
    estimate_periodic_gaussian,
    periodic_characteristic_features,
)
from cnn_fpga.decoder.sliding_window_syndrome import (
    SlidingWindowConfig,
    SlidingWindowPeriodicEstimator,
    validate_window_candidates,
)
from physics.constants import LATTICE_CONST


def _wrapped(seed: int, samples: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    covariance = LATTICE_CONST**2 * np.asarray([[0.030, -0.011], [-0.011, 0.024]])
    values = rng.multivariate_normal(
        np.asarray([0.16, -0.11]) * LATTICE_CONST,
        covariance,
        size=samples,
    )
    return (values + LATTICE_CONST / 2.0) % LATTICE_CONST - LATTICE_CONST / 2.0


def test_incremental_feature_sum_exactly_matches_raw_rolling_window() -> None:
    moment = PeriodicMomentConfig(minimum_samples=96)
    calibration = _wrapped(1, 1536)
    config = SlidingWindowConfig(
        window_samples=576,
        update_stride_samples=384,
        feature_chunk_samples=96,
    )
    estimator = SlidingWindowPeriodicEstimator(
        calibration,
        sliding_config=config,
        moment_config=moment,
    )
    raw = calibration[-576:].copy()
    assert np.allclose(
        estimator.feature_state,
        periodic_characteristic_features(raw, moment),
        rtol=0.0,
        atol=4e-16,
    )
    for window_id in range(5):
        update = _wrapped(10 + window_id, 384)
        raw = np.concatenate((raw, update), axis=0)[-576:]
        prediction = estimator.update(update, window_id=window_id)
        batch = estimate_periodic_gaussian(raw, moment, window_id=window_id)
        assert np.allclose(estimator.feature_state, batch.feature_moments, rtol=0.0, atol=4e-16)
        assert np.allclose(prediction.mean_array(), batch.mean_array(), rtol=0.0, atol=2e-15)
        assert np.allclose(
            prediction.covariance_array(), batch.covariance_array(), rtol=0.0, atol=2e-15
        )


def test_one_stride_candidate_matches_latest_batch_estimator() -> None:
    moment = PeriodicMomentConfig(minimum_samples=96)
    calibration = _wrapped(21, 1536)
    estimator = SlidingWindowPeriodicEstimator(
        calibration,
        sliding_config=SlidingWindowConfig(384),
        moment_config=moment,
    )
    for window_id in range(3):
        update = _wrapped(30 + window_id, 384)
        prediction = estimator.update(update, window_id=window_id)
        batch = estimate_periodic_gaussian(update, moment, window_id=window_id)
        assert np.allclose(prediction.mean_array(), batch.mean_array(), atol=2e-15, rtol=0.0)
        assert np.allclose(
            prediction.covariance_array(), batch.covariance_array(), atol=2e-15, rtol=0.0
        )


def test_long_window_retains_only_feature_chunks_not_raw_samples() -> None:
    estimator = SlidingWindowPeriodicEstimator(
        _wrapped(40, 1536),
        sliding_config=SlidingWindowConfig(1536),
        moment_config=PeriodicMomentConfig(minimum_samples=96),
    )
    chunks = list(estimator._chunks)
    assert len(chunks) == 16
    assert all(chunk.shape == (4,) and np.iscomplexobj(chunk) for chunk in chunks)
    assert estimator.retained_observations == 1536
    assert not any(
        isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[-1] == 2
        for value in estimator.__dict__.values()
    )


def test_storage_proxy_grows_but_per_observation_work_does_not() -> None:
    calibration = _wrapped(50, 1536)
    moment = PeriodicMomentConfig(minimum_samples=96)
    short = SlidingWindowPeriodicEstimator(
        calibration,
        sliding_config=SlidingWindowConfig(384),
        moment_config=moment,
    ).cost_profile()
    long = SlidingWindowPeriodicEstimator(
        calibration,
        sliding_config=SlidingWindowConfig(1536),
        moment_config=moment,
    ).cost_profile()
    assert long.stored_complex_values > short.stored_complex_values
    assert long.complex_exponentials_per_observation == short.complex_exponentials_per_observation == 2
    assert long.complex_products_per_observation == short.complex_products_per_observation == 2
    assert long.target_lut is None and long.target_measured is False


def test_update_rejects_wrong_stride_stale_gap_and_noninteger_ids() -> None:
    estimator = SlidingWindowPeriodicEstimator(
        _wrapped(60, 1536),
        sliding_config=SlidingWindowConfig(768),
        moment_config=PeriodicMomentConfig(minimum_samples=96),
    )
    with pytest.raises(ValueError, match="exactly"):
        estimator.update(_wrapped(61, 480), window_id=0)
    with pytest.raises(TypeError, match="integer"):
        estimator.update(_wrapped(62, 384), window_id=0.5)  # type: ignore[arg-type]
    estimator.update(_wrapped(63, 384), window_id=0)
    with pytest.raises(ValueError, match="sequential"):
        estimator.update(_wrapped(64, 384), window_id=0)
    with pytest.raises(ValueError, match="sequential"):
        estimator.update(_wrapped(65, 384), window_id=2)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"window_samples": 383}, "at least one stride"),
        ({"window_samples": 500}, "divisible"),
        ({"window_samples": 768, "update_stride_samples": 400}, "update_stride"),
        ({"window_samples": True}, "integer"),
    ],
)
def test_sliding_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        SlidingWindowConfig(**kwargs)  # type: ignore[arg-type]


def test_calibration_must_fill_window_and_updates_must_be_finite_wrapped() -> None:
    moment = PeriodicMomentConfig(minimum_samples=96)
    with pytest.raises(ValueError, match="shorter"):
        SlidingWindowPeriodicEstimator(
            _wrapped(70, 384),
            sliding_config=SlidingWindowConfig(768),
            moment_config=moment,
        )
    estimator = SlidingWindowPeriodicEstimator(
        _wrapped(71, 768),
        sliding_config=SlidingWindowConfig(768),
        moment_config=moment,
    )
    nonfinite = _wrapped(72, 384)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator.update(nonfinite, window_id=0)
    with pytest.raises(ValueError, match="lattice/2"):
        estimator.update(np.full((384, 2), LATTICE_CONST / 2.0), window_id=0)


def test_candidate_contract_rejects_demo_or_unordered_grids() -> None:
    assert validate_window_candidates(
        (384, 480, 768, 1536), update_stride_samples=384, feature_chunk_samples=96
    ) == (384, 480, 768, 1536)
    with pytest.raises(ValueError, match="at least four"):
        validate_window_candidates(
            (384, 768, 1536), update_stride_samples=384, feature_chunk_samples=96
        )
    with pytest.raises(ValueError, match="increasing"):
        validate_window_candidates(
            (384, 768, 480, 1536), update_stride_samples=384, feature_chunk_samples=96
        )
    with pytest.raises(ValueError, match="unique"):
        validate_window_candidates(
            (384, 480, 480, 1536), update_stride_samples=384, feature_chunk_samples=96
        )


def test_online_surface_has_no_truth_state_or_displacement_inputs() -> None:
    names = set(inspect.signature(SlidingWindowPeriodicEstimator.update).parameters)
    assert names == {"self", "residuals", "window_id"}
    forbidden = {"truth", "state", "drift", "displacement", "logical_parity"}
    assert not names & forbidden
