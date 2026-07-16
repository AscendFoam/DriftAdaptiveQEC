"""Causal periodic-moment estimators for adaptive Gaussian MAP decoding.

The input is a window of observed modular GKP syndromes in the half-open
fundamental cell.  First and joint circular characteristic moments identify
the mean and the full 2x2 covariance of a wrapped Gaussian without access to
lattice indices.  Three causal predictors are provided:

* latest-window periodic moments;
* EWMA of the four complex characteristic moments;
* a constant-velocity Kalman filter on ``(mu_q, mu_p, log var_q,
  log var_p, atanh rho)``.

These objects estimate decoder parameters only.  They never accept logical
truth, hidden ``DriftState`` objects, or evaluation outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atanh, isfinite, log, pi, sqrt, tanh
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physics.constants import LATTICE_CONST


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: object, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _tuple5(values: Sequence[float], name: str, *, positive: bool) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != 5:
        raise ValueError(f"{name} must contain exactly five values")
    result = tuple(_finite(value, f"{name}[{index}]") for index, value in enumerate(values))
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _wrap(values: ArrayLike, period: float) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    return np.mod(array + 0.5 * period, period) - 0.5 * period


def _stabilize_covariance(
    covariance: ArrayLike,
    *,
    variance_floor: float,
    variance_ceiling: float,
    rho_clip: float,
) -> NDArray[np.float64]:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite 2x2 matrix")
    symmetric = 0.5 * (matrix + matrix.T)
    var_q = float(np.clip(symmetric[0, 0], variance_floor, variance_ceiling))
    var_p = float(np.clip(symmetric[1, 1], variance_floor, variance_ceiling))
    limit = rho_clip * sqrt(var_q * var_p)
    cross = float(np.clip(symmetric[0, 1], -limit, limit))
    result = np.array([[var_q, cross], [cross, var_p]], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(result)
    eigenvalues = np.clip(eigenvalues, variance_floor, variance_ceiling)
    result = (eigenvectors * eigenvalues) @ eigenvectors.T
    result = 0.5 * (result + result.T)
    # The eigenvalue projection can slightly increase the correlation again.
    var_q = float(np.clip(result[0, 0], variance_floor, variance_ceiling))
    var_p = float(np.clip(result[1, 1], variance_floor, variance_ceiling))
    limit = rho_clip * sqrt(var_q * var_p)
    cross = float(np.clip(result[0, 1], -limit, limit))
    return np.array([[var_q, cross], [cross, var_p]], dtype=np.float64)


@dataclass(frozen=True)
class PeriodicMomentConfig:
    lattice: float = LATTICE_CONST
    minimum_samples: int = 64
    variance_floor: float = (0.035 * LATTICE_CONST) ** 2
    variance_ceiling: float = (0.45 * LATTICE_CONST) ** 2
    resultant_floor: float = 1.0e-3
    rho_clip: float = 0.95
    covariance_shrinkage: float = 0.02

    def __post_init__(self) -> None:
        lattice = _finite(self.lattice, "lattice")
        if lattice <= 0.0:
            raise ValueError("lattice must be positive")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(
            self,
            "minimum_samples",
            _positive_integer(self.minimum_samples, "minimum_samples", 8),
        )
        floor = _finite(self.variance_floor, "variance_floor")
        ceiling = _finite(self.variance_ceiling, "variance_ceiling")
        if floor <= 0.0 or ceiling <= floor:
            raise ValueError("variance bounds must satisfy 0 < floor < ceiling")
        object.__setattr__(self, "variance_floor", floor)
        object.__setattr__(self, "variance_ceiling", ceiling)
        resultant = _finite(self.resultant_floor, "resultant_floor")
        if not 0.0 < resultant < 1.0:
            raise ValueError("resultant_floor must lie in (0,1)")
        object.__setattr__(self, "resultant_floor", resultant)
        rho = _finite(self.rho_clip, "rho_clip")
        if not 0.0 < rho < 1.0:
            raise ValueError("rho_clip must lie in (0,1)")
        object.__setattr__(self, "rho_clip", rho)
        shrinkage = _finite(self.covariance_shrinkage, "covariance_shrinkage")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError("covariance_shrinkage must lie in [0,1)")
        object.__setattr__(self, "covariance_shrinkage", shrinkage)


@dataclass(frozen=True)
class PeriodicGaussianEstimate:
    mean: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    source: str
    window_id: int
    observation_count: int
    update_applied: bool
    resultants: tuple[float, float, float, float]
    joint_covariance_discrepancy: float
    feature_moments: tuple[complex, complex, complex, complex]

    def mean_array(self) -> NDArray[np.float64]:
        return np.asarray(self.mean, dtype=np.float64)

    def covariance_array(self) -> NDArray[np.float64]:
        return np.asarray(self.covariance, dtype=np.float64)

    @property
    def rho(self) -> float:
        covariance = self.covariance_array()
        return float(covariance[0, 1] / sqrt(covariance[0, 0] * covariance[1, 1]))


def validate_residual_window(
    residuals: ArrayLike,
    config: PeriodicMomentConfig,
) -> NDArray[np.float64]:
    if not isinstance(config, PeriodicMomentConfig):
        raise TypeError("config must be a PeriodicMomentConfig")
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("residuals must have shape (samples,2)")
    if values.shape[0] < config.minimum_samples:
        raise ValueError("residuals contain fewer than minimum_samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("residuals must contain only finite values")
    half = 0.5 * config.lattice
    if np.any(values < -half) or np.any(values >= half):
        raise ValueError("residuals must lie in [-lattice/2,lattice/2)")
    return values


def periodic_characteristic_features(
    residuals: ArrayLike,
    config: PeriodicMomentConfig,
) -> NDArray[np.complex128]:
    values = validate_residual_window(residuals, config)
    frequency = 2.0 * pi / config.lattice
    phase_q = np.exp(1j * frequency * values[:, 0])
    phase_p = np.exp(1j * frequency * values[:, 1])
    return np.asarray(
        [
            np.mean(phase_q),
            np.mean(phase_p),
            np.mean(phase_q * phase_p),
            np.mean(phase_q * np.conjugate(phase_p)),
        ],
        dtype=np.complex128,
    )


def _gaussian_characteristic_features(
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
    lattice: float,
) -> NDArray[np.complex128]:
    frequency = 2.0 * pi / lattice
    directions = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=np.float64,
    )
    phases = frequency * (directions @ mean)
    variances = np.einsum("ij,jk,ik->i", directions, covariance, directions)
    return np.exp(1j * phases - 0.5 * frequency * frequency * variances).astype(
        np.complex128
    )


def estimate_from_characteristic_features(
    features: ArrayLike,
    config: PeriodicMomentConfig,
    *,
    source: str,
    window_id: int,
    observation_count: int,
    update_applied: bool,
) -> PeriodicGaussianEstimate:
    values = np.asarray(features, dtype=np.complex128)
    if values.shape != (4,) or not np.all(np.isfinite(values.real)) or not np.all(
        np.isfinite(values.imag)
    ):
        raise ValueError("features must be four finite complex moments")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    count = _positive_integer(observation_count, "observation_count")
    resultants = np.abs(values)
    if resultants[0] <= config.resultant_floor or resultants[1] <= config.resultant_floor:
        raise ValueError("marginal circular moments are below resultant_floor")
    if max(resultants[2], resultants[3]) <= config.resultant_floor:
        raise ValueError("joint circular moments are below resultant_floor")
    clipped = np.clip(resultants, config.resultant_floor, 1.0)
    frequency = 2.0 * pi / config.lattice
    scale = frequency * frequency
    var_q_raw = -2.0 * log(float(clipped[0])) / scale
    var_p_raw = -2.0 * log(float(clipped[1])) / scale
    variance_plus = -2.0 * log(float(clipped[2])) / scale
    variance_minus = -2.0 * log(float(clipped[3])) / scale
    covariance_plus = 0.5 * (variance_plus - var_q_raw - var_p_raw)
    covariance_minus = 0.5 * (var_q_raw + var_p_raw - variance_minus)
    cross = 0.5 * (covariance_plus + covariance_minus)
    cross *= 1.0 - config.covariance_shrinkage
    covariance = _stabilize_covariance(
        [[var_q_raw, cross], [cross, var_p_raw]],
        variance_floor=config.variance_floor,
        variance_ceiling=config.variance_ceiling,
        rho_clip=config.rho_clip,
    )
    mean = _wrap(
        [np.angle(values[0]) / frequency, np.angle(values[1]) / frequency],
        config.lattice,
    )
    return PeriodicGaussianEstimate(
        mean=(float(mean[0]), float(mean[1])),
        covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
        source=source.strip(),
        window_id=int(window_id),
        observation_count=count,
        update_applied=bool(update_applied),
        resultants=tuple(float(item) for item in resultants),
        joint_covariance_discrepancy=float(abs(covariance_plus - covariance_minus)),
        feature_moments=tuple(complex(item) for item in values),
    )


def estimate_periodic_gaussian(
    residuals: ArrayLike,
    config: PeriodicMomentConfig | None = None,
    *,
    source: str = "periodic_window_moments",
    window_id: int = -1,
) -> PeriodicGaussianEstimate:
    actual = PeriodicMomentConfig() if config is None else config
    features = periodic_characteristic_features(residuals, actual)
    return estimate_from_characteristic_features(
        features,
        actual,
        source=source,
        window_id=window_id,
        observation_count=np.asarray(residuals).shape[0],
        update_applied=True,
    )


class LatestWindowPeriodicPredictor:
    def __init__(
        self,
        calibration_residuals: ArrayLike,
        config: PeriodicMomentConfig | None = None,
    ) -> None:
        self.config = PeriodicMomentConfig() if config is None else config
        self._estimate = estimate_periodic_gaussian(
            calibration_residuals,
            self.config,
            source="periodic_window_calibration",
            window_id=-1,
        )

    def prediction(self) -> PeriodicGaussianEstimate:
        return self._estimate

    def update(self, residuals: ArrayLike, *, window_id: int) -> PeriodicGaussianEstimate:
        self._estimate = estimate_periodic_gaussian(
            residuals,
            self.config,
            source="periodic_latest_window",
            window_id=window_id,
        )
        return self._estimate


class PeriodicMomentEWMA:
    def __init__(
        self,
        calibration_residuals: ArrayLike,
        *,
        alpha: float,
        config: PeriodicMomentConfig | None = None,
    ) -> None:
        self.config = PeriodicMomentConfig() if config is None else config
        smoothing = _finite(alpha, "alpha")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("alpha must lie in (0,1]")
        self.alpha = smoothing
        calibration = validate_residual_window(calibration_residuals, self.config)
        self._features = periodic_characteristic_features(calibration, self.config)
        self._observation_count = int(calibration.shape[0])
        self._window_id = -1

    @property
    def feature_state(self) -> NDArray[np.complex128]:
        return self._features.copy()

    def prediction(self) -> PeriodicGaussianEstimate:
        return estimate_from_characteristic_features(
            self._features,
            self.config,
            source="periodic_moment_ewma",
            window_id=self._window_id,
            observation_count=self._observation_count,
            update_applied=self._window_id >= 0,
        )

    def update(self, residuals: ArrayLike, *, window_id: int) -> PeriodicGaussianEstimate:
        values = validate_residual_window(residuals, self.config)
        features = periodic_characteristic_features(values, self.config)
        self._features = (1.0 - self.alpha) * self._features + self.alpha * features
        self._observation_count = int(values.shape[0])
        self._window_id = int(window_id)
        return self.prediction()


@dataclass(frozen=True)
class PeriodicKalmanConfig:
    process_std_position: tuple[float, ...] = (0.015, 0.015, 0.08, 0.08, 0.08)
    process_std_velocity: tuple[float, ...] = (0.004, 0.004, 0.02, 0.02, 0.02)
    measurement_std: tuple[float, ...] = (0.035, 0.035, 0.18, 0.18, 0.16)
    initial_velocity_std: tuple[float, ...] = (0.02, 0.02, 0.10, 0.10, 0.10)
    velocity_decay: float = 0.92
    covariance_floor: float = 1.0e-10

    def __post_init__(self) -> None:
        for name in (
            "process_std_position",
            "process_std_velocity",
            "measurement_std",
            "initial_velocity_std",
        ):
            object.__setattr__(self, name, _tuple5(getattr(self, name), name, positive=True))
        decay = _finite(self.velocity_decay, "velocity_decay")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("velocity_decay must lie in [0,1]")
        object.__setattr__(self, "velocity_decay", decay)
        floor = _finite(self.covariance_floor, "covariance_floor")
        if floor <= 0.0:
            raise ValueError("covariance_floor must be positive")
        object.__setattr__(self, "covariance_floor", floor)


def scaled_periodic_kalman_config(
    *,
    process_scale: float,
    measurement_scale: float,
) -> PeriodicKalmanConfig:
    process = _finite(process_scale, "process_scale")
    measurement = _finite(measurement_scale, "measurement_scale")
    if process <= 0.0 or measurement <= 0.0:
        raise ValueError("Kalman scales must be positive")
    base = PeriodicKalmanConfig()
    return PeriodicKalmanConfig(
        process_std_position=tuple(value * process for value in base.process_std_position),
        process_std_velocity=tuple(value * process for value in base.process_std_velocity),
        measurement_std=tuple(value * measurement for value in base.measurement_std),
        initial_velocity_std=base.initial_velocity_std,
        velocity_decay=base.velocity_decay,
        covariance_floor=base.covariance_floor,
    )


class ConstantVelocityPeriodicKalman:
    def __init__(
        self,
        calibration_residuals: ArrayLike,
        *,
        moment_config: PeriodicMomentConfig | None = None,
        kalman_config: PeriodicKalmanConfig | None = None,
    ) -> None:
        self.moment_config = (
            PeriodicMomentConfig() if moment_config is None else moment_config
        )
        self.kalman_config = (
            PeriodicKalmanConfig() if kalman_config is None else kalman_config
        )
        initial = estimate_periodic_gaussian(
            calibration_residuals,
            self.moment_config,
            source="periodic_kalman_calibration",
            window_id=-1,
        )
        coordinates = self._estimate_coordinates(initial, reference=None)
        self._state = np.concatenate((coordinates, np.zeros(5, dtype=np.float64)))
        measurement_variance = np.square(self.kalman_config.measurement_std)
        velocity_variance = np.square(self.kalman_config.initial_velocity_std)
        self._covariance = np.diag(np.concatenate((measurement_variance, velocity_variance)))
        self._transition = np.eye(10, dtype=np.float64)
        self._transition[:5, 5:] = np.eye(5, dtype=np.float64)
        self._transition[5:, 5:] *= self.kalman_config.velocity_decay
        self._observation = np.zeros((5, 10), dtype=np.float64)
        self._observation[:, :5] = np.eye(5, dtype=np.float64)
        self._process_covariance = np.diag(
            np.square(
                np.concatenate(
                    (
                        self.kalman_config.process_std_position,
                        self.kalman_config.process_std_velocity,
                    )
                )
            )
        )
        self._measurement_covariance = np.diag(
            np.square(self.kalman_config.measurement_std)
        )
        self._window_id = -1
        self._observation_count = int(np.asarray(calibration_residuals).shape[0])
        self._last_gain = np.zeros((10, 5), dtype=np.float64)

    @property
    def state(self) -> NDArray[np.float64]:
        return self._state.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        return self._covariance.copy()

    @property
    def kalman_gain(self) -> NDArray[np.float64]:
        return self._last_gain.copy()

    def _estimate_coordinates(
        self,
        estimate: PeriodicGaussianEstimate,
        *,
        reference: NDArray[np.float64] | None,
    ) -> NDArray[np.float64]:
        mean = estimate.mean_array().copy()
        if reference is not None:
            for axis in range(2):
                mean[axis] += round(
                    (float(reference[axis]) - float(mean[axis]))
                    / self.moment_config.lattice
                ) * self.moment_config.lattice
        covariance = estimate.covariance_array()
        rho = float(
            np.clip(
                covariance[0, 1] / sqrt(covariance[0, 0] * covariance[1, 1]),
                -self.moment_config.rho_clip,
                self.moment_config.rho_clip,
            )
        )
        return np.asarray(
            [
                mean[0],
                mean[1],
                log(float(covariance[0, 0])),
                log(float(covariance[1, 1])),
                atanh(rho),
            ],
            dtype=np.float64,
        )

    def _coordinates_estimate(
        self,
        coordinates: NDArray[np.float64],
        *,
        source: str,
        update_applied: bool,
    ) -> PeriodicGaussianEstimate:
        mean = _wrap(coordinates[:2], self.moment_config.lattice)
        variance = np.exp(
            np.clip(
                coordinates[2:4],
                log(self.moment_config.variance_floor),
                log(self.moment_config.variance_ceiling),
            )
        )
        rho = float(
            np.clip(
                tanh(float(coordinates[4])),
                -self.moment_config.rho_clip,
                self.moment_config.rho_clip,
            )
        )
        cross = rho * sqrt(float(variance[0] * variance[1]))
        covariance = _stabilize_covariance(
            [[variance[0], cross], [cross, variance[1]]],
            variance_floor=self.moment_config.variance_floor,
            variance_ceiling=self.moment_config.variance_ceiling,
            rho_clip=self.moment_config.rho_clip,
        )
        features = _gaussian_characteristic_features(
            np.asarray(mean, dtype=np.float64),
            covariance,
            self.moment_config.lattice,
        )
        return PeriodicGaussianEstimate(
            mean=(float(mean[0]), float(mean[1])),
            covariance=(
                (float(covariance[0, 0]), float(covariance[0, 1])),
                (float(covariance[1, 0]), float(covariance[1, 1])),
            ),
            source=source,
            window_id=self._window_id,
            observation_count=self._observation_count,
            update_applied=update_applied,
            resultants=tuple(float(value) for value in np.abs(features)),
            joint_covariance_discrepancy=0.0,
            feature_moments=tuple(complex(value) for value in features),
        )

    def _predicted_state_covariance(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        state = self._transition @ self._state
        covariance = (
            self._transition @ self._covariance @ self._transition.T
            + self._process_covariance
        )
        covariance = 0.5 * (covariance + covariance.T)
        return state, covariance

    def prediction(self) -> PeriodicGaussianEstimate:
        state, _ = self._predicted_state_covariance()
        return self._coordinates_estimate(
            state[:5],
            source="constant_velocity_periodic_kalman",
            update_applied=self._window_id >= 0,
        )

    def update(self, residuals: ArrayLike, *, window_id: int) -> PeriodicGaussianEstimate:
        values = validate_residual_window(residuals, self.moment_config)
        measurement_estimate = estimate_periodic_gaussian(
            values,
            self.moment_config,
            source="periodic_kalman_measurement",
            window_id=window_id,
        )
        predicted_state, predicted_covariance = self._predicted_state_covariance()
        measurement = self._estimate_coordinates(
            measurement_estimate,
            reference=predicted_state[:5],
        )
        innovation = measurement - self._observation @ predicted_state
        innovation_covariance = (
            self._observation @ predicted_covariance @ self._observation.T
            + self._measurement_covariance
        )
        gain = np.linalg.solve(
            innovation_covariance.T,
            (predicted_covariance @ self._observation.T).T,
        ).T
        updated_state = predicted_state + gain @ innovation
        identity = np.eye(10, dtype=np.float64)
        residual_operator = identity - gain @ self._observation
        # Joseph form preserves positive semidefiniteness under roundoff.
        updated_covariance = (
            residual_operator @ predicted_covariance @ residual_operator.T
            + gain @ self._measurement_covariance @ gain.T
        )
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(updated_covariance)
        eigenvalues = np.maximum(eigenvalues, self.kalman_config.covariance_floor)
        self._covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        self._state = updated_state
        self._window_id = int(window_id)
        self._observation_count = int(values.shape[0])
        self._last_gain = gain
        return self.prediction()


__all__ = [
    "PeriodicMomentConfig",
    "PeriodicGaussianEstimate",
    "validate_residual_window",
    "periodic_characteristic_features",
    "estimate_from_characteristic_features",
    "estimate_periodic_gaussian",
    "LatestWindowPeriodicPredictor",
    "PeriodicMomentEWMA",
    "PeriodicKalmanConfig",
    "scaled_periodic_kalman_config",
    "ConstantVelocityPeriodicKalman",
]
