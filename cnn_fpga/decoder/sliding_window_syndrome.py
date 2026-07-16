"""Incremental periodic sliding-window syndrome estimator.

The estimator stores sufficient circular-feature sums in fixed-size chunks.
It receives the same number of new residuals at every update regardless of
history length, while older chunks leave the rolling window exactly.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cnn_fpga.decoder.periodic_adaptive_map import (
    PeriodicGaussianEstimate,
    PeriodicMomentConfig,
    estimate_from_characteristic_features,
    periodic_characteristic_features,
    validate_residual_window,
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class SlidingWindowConfig:
    window_samples: int
    update_stride_samples: int = 384
    feature_chunk_samples: int = 96

    def __post_init__(self) -> None:
        window = _positive_integer(self.window_samples, "window_samples")
        stride = _positive_integer(self.update_stride_samples, "update_stride_samples")
        chunk = _positive_integer(self.feature_chunk_samples, "feature_chunk_samples")
        if stride % chunk != 0:
            raise ValueError("update_stride_samples must be divisible by feature_chunk_samples")
        if window < stride or window % chunk != 0:
            raise ValueError(
                "window_samples must be at least one stride and divisible by feature_chunk_samples"
            )
        object.__setattr__(self, "window_samples", window)
        object.__setattr__(self, "update_stride_samples", stride)
        object.__setattr__(self, "feature_chunk_samples", chunk)

    @property
    def retained_chunks(self) -> int:
        return self.window_samples // self.feature_chunk_samples

    @property
    def chunks_per_update(self) -> int:
        return self.update_stride_samples // self.feature_chunk_samples


@dataclass(frozen=True)
class SlidingWindowCostProfile:
    window_samples: int
    update_stride_samples: int
    retained_feature_chunks: int
    complex_values_per_chunk: int = 4
    complex_accumulator_values: int = 4
    complex_exponentials_per_observation: int = 2
    complex_products_per_observation: int = 2
    target_lut: int | None = None
    target_ff: int | None = None
    target_bram: int | None = None
    target_dsp: int | None = None
    target_fmax_hz: float | None = None
    target_measured: bool = False
    scope: str = "incremental_sufficient_statistic_proxy_not_synthesis"

    @property
    def stored_complex_values(self) -> int:
        return self.retained_feature_chunks * self.complex_values_per_chunk + self.complex_accumulator_values


class SlidingWindowPeriodicEstimator:
    """Causal uniform sliding window over four joint circular moments."""

    def __init__(
        self,
        calibration_residuals: ArrayLike,
        *,
        sliding_config: SlidingWindowConfig,
        moment_config: PeriodicMomentConfig | None = None,
    ) -> None:
        if not isinstance(sliding_config, SlidingWindowConfig):
            raise TypeError("sliding_config must be SlidingWindowConfig")
        self.sliding_config = sliding_config
        self.moment_config = PeriodicMomentConfig() if moment_config is None else moment_config
        values = validate_residual_window(calibration_residuals, self.moment_config)
        if values.shape[0] < sliding_config.window_samples:
            raise ValueError("calibration_residuals are shorter than window_samples")
        retained = values[-sliding_config.window_samples :]
        self._chunks: deque[NDArray[np.complex128]] = deque()
        self._feature_sum = np.zeros(4, dtype=np.complex128)
        chunk = sliding_config.feature_chunk_samples
        for start in range(0, retained.shape[0], chunk):
            self._append_chunk(retained[start : start + chunk])
        self._last_window_id = -1

    def _append_chunk(self, residuals: NDArray[np.float64]) -> None:
        features = periodic_characteristic_features(residuals, self.moment_config)
        feature_sum = features * residuals.shape[0]
        self._chunks.append(feature_sum)
        self._feature_sum += feature_sum
        while len(self._chunks) > self.sliding_config.retained_chunks:
            self._feature_sum -= self._chunks.popleft()

    @property
    def feature_state(self) -> NDArray[np.complex128]:
        return self._feature_sum / self.sliding_config.window_samples

    @property
    def retained_observations(self) -> int:
        return len(self._chunks) * self.sliding_config.feature_chunk_samples

    def cost_profile(self) -> SlidingWindowCostProfile:
        return SlidingWindowCostProfile(
            window_samples=self.sliding_config.window_samples,
            update_stride_samples=self.sliding_config.update_stride_samples,
            retained_feature_chunks=self.sliding_config.retained_chunks,
        )

    def prediction(self) -> PeriodicGaussianEstimate:
        if self.retained_observations != self.sliding_config.window_samples:
            raise RuntimeError("sliding window is not fully initialized")
        return estimate_from_characteristic_features(
            self.feature_state,
            self.moment_config,
            source="periodic_uniform_sliding_window",
            window_id=self._last_window_id,
            observation_count=self.sliding_config.window_samples,
            update_applied=self._last_window_id >= 0,
        )

    def update(self, residuals: ArrayLike, *, window_id: int) -> PeriodicGaussianEstimate:
        update_id = _nonnegative_integer(window_id, "window_id")
        if update_id != self._last_window_id + 1:
            raise ValueError("window_id must be sequential with no duplicates or gaps")
        values = validate_residual_window(residuals, self.moment_config)
        if values.shape[0] != self.sliding_config.update_stride_samples:
            raise ValueError("residual update must contain exactly update_stride_samples")
        chunk = self.sliding_config.feature_chunk_samples
        for start in range(0, values.shape[0], chunk):
            self._append_chunk(values[start : start + chunk])
        self._last_window_id = update_id
        return self.prediction()


def validate_window_candidates(
    candidates: Sequence[int],
    *,
    update_stride_samples: int,
    feature_chunk_samples: int,
) -> tuple[int, ...]:
    values = tuple(_positive_integer(value, "window candidate") for value in candidates)
    if len(values) < 4 or len(set(values)) != len(values):
        raise ValueError("window candidates must contain at least four unique values")
    if tuple(sorted(values)) != values:
        raise ValueError("window candidates must be strictly increasing")
    for value in values:
        SlidingWindowConfig(
            window_samples=value,
            update_stride_samples=update_stride_samples,
            feature_chunk_samples=feature_chunk_samples,
        )
    return values


__all__ = [
    "SlidingWindowConfig",
    "SlidingWindowCostProfile",
    "SlidingWindowPeriodicEstimator",
    "validate_window_candidates",
]
