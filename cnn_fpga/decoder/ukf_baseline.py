"""Unscented Kalman filter baseline with constant-velocity latent drift state."""

# 中文说明：
# - 该模块实现比 `EKF` 更强的一类经典滤波 baseline：`UKF`。
# - 与现有 `EKF` 相比，这里引入了“常速度”隐状态：
#   `[sigma, mu_q, mu_p, theta, v_sigma, v_mu_q, v_mu_p, v_theta]`
# - 这样可以更自然地表达漂移趋势，而不是只做纯随机游走。
# - 测量仍来自当前窗口 histogram 的 `Window Variance` 估计，因此保持严格因果、公平对比。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from cnn_fpga.decoder.param_mapper import NoisePrediction
from cnn_fpga.decoder.window_baseline import WindowVarianceBaseline, WindowVarianceConfig


def _stabilize_covariance(covariance: np.ndarray, floor: float) -> np.ndarray:
    """Project covariance to a symmetric positive-definite matrix."""
    symmetric = 0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, float(floor))
    stabilized = (eigenvectors * clipped) @ eigenvectors.T
    return 0.5 * (stabilized + stabilized.T)


def _wrap_theta_deg(theta_deg: float) -> float:
    return ((float(theta_deg) + 90.0) % 180.0) - 90.0


def _wrap_state_theta(state: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(state, dtype=float).copy()
    wrapped[3] = _wrap_theta_deg(wrapped[3])
    wrapped[7] = _wrap_theta_deg(wrapped[7])
    return wrapped


def _weighted_theta_mean_deg(values_deg: np.ndarray, weights: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(values_deg, dtype=float))
    sin_mean = float(np.sum(weights * np.sin(radians)))
    cos_mean = float(np.sum(weights * np.cos(radians)))
    if abs(sin_mean) < 1.0e-12 and abs(cos_mean) < 1.0e-12:
        return 0.0
    return _wrap_theta_deg(np.rad2deg(np.arctan2(sin_mean, cos_mean)))


@dataclass(frozen=True)
class UKFBaselineConfig:
    """Configuration of the unscented constant-velocity filter."""

    sigma_clip: Tuple[float, float]
    mu_clip: Tuple[float, float]
    theta_clip_deg: Tuple[float, float]
    initial_sigma: float
    initial_mu_q: float
    initial_mu_p: float
    initial_theta_deg: float
    initial_velocity_std_sigma: float
    initial_velocity_std_mu_q: float
    initial_velocity_std_mu_p: float
    initial_velocity_std_theta_deg: float
    process_std_sigma: float
    process_std_mu_q: float
    process_std_mu_p: float
    process_std_theta_deg: float
    process_std_v_sigma: float
    process_std_v_mu_q: float
    process_std_v_mu_p: float
    process_std_v_theta_deg: float
    measurement_std_sigma: float
    measurement_std_mu_q: float
    measurement_std_mu_p: float
    measurement_std_theta_deg: float
    velocity_decay: float
    covariance_floor: float
    alpha: float
    beta: float
    kappa: float

    def __post_init__(self) -> None:
        if self.sigma_clip[0] <= 0 or self.sigma_clip[0] > self.sigma_clip[1]:
            raise ValueError("sigma_clip must be positive and ordered")
        if self.mu_clip[0] > self.mu_clip[1]:
            raise ValueError("mu_clip must be ordered")
        if self.theta_clip_deg[0] > self.theta_clip_deg[1]:
            raise ValueError("theta_clip_deg must be ordered")
        if self.initial_sigma <= 0:
            raise ValueError("initial_sigma must be positive")
        for name, value in (
            ("initial_velocity_std_sigma", self.initial_velocity_std_sigma),
            ("initial_velocity_std_mu_q", self.initial_velocity_std_mu_q),
            ("initial_velocity_std_mu_p", self.initial_velocity_std_mu_p),
            ("initial_velocity_std_theta_deg", self.initial_velocity_std_theta_deg),
            ("process_std_sigma", self.process_std_sigma),
            ("process_std_mu_q", self.process_std_mu_q),
            ("process_std_mu_p", self.process_std_mu_p),
            ("process_std_theta_deg", self.process_std_theta_deg),
            ("process_std_v_sigma", self.process_std_v_sigma),
            ("process_std_v_mu_q", self.process_std_v_mu_q),
            ("process_std_v_mu_p", self.process_std_v_mu_p),
            ("process_std_v_theta_deg", self.process_std_v_theta_deg),
            ("measurement_std_sigma", self.measurement_std_sigma),
            ("measurement_std_mu_q", self.measurement_std_mu_q),
            ("measurement_std_mu_p", self.measurement_std_mu_p),
            ("measurement_std_theta_deg", self.measurement_std_theta_deg),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 <= self.velocity_decay <= 1.0):
            raise ValueError("velocity_decay must be within [0.0, 1.0]")
        if self.covariance_floor <= 0:
            raise ValueError("covariance_floor must be positive")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UKFBaselineConfig":
        slow_cfg = config.get("slow_loop", {})
        ukf_cfg = slow_cfg.get("ukf", {})
        window_cfg = WindowVarianceConfig.from_config(config)
        fixed_cfg = slow_cfg.get("fixed_baseline", {})
        return cls(
            sigma_clip=window_cfg.sigma_clip,
            mu_clip=window_cfg.mu_clip,
            theta_clip_deg=window_cfg.theta_clip_deg,
            initial_sigma=float(ukf_cfg.get("initial_sigma", fixed_cfg.get("sigma", window_cfg.sigma_clip[0]))),
            initial_mu_q=float(ukf_cfg.get("initial_mu_q", fixed_cfg.get("mu_q", 0.0))),
            initial_mu_p=float(ukf_cfg.get("initial_mu_p", fixed_cfg.get("mu_p", 0.0))),
            initial_theta_deg=float(ukf_cfg.get("initial_theta_deg", fixed_cfg.get("theta_deg", 0.0))),
            initial_velocity_std_sigma=float(ukf_cfg.get("initial_velocity_std_sigma", 0.01)),
            initial_velocity_std_mu_q=float(ukf_cfg.get("initial_velocity_std_mu_q", 0.01)),
            initial_velocity_std_mu_p=float(ukf_cfg.get("initial_velocity_std_mu_p", 0.01)),
            initial_velocity_std_theta_deg=float(ukf_cfg.get("initial_velocity_std_theta_deg", 1.0)),
            process_std_sigma=float(ukf_cfg.get("process_std_sigma", 0.008)),
            process_std_mu_q=float(ukf_cfg.get("process_std_mu_q", 0.008)),
            process_std_mu_p=float(ukf_cfg.get("process_std_mu_p", 0.008)),
            process_std_theta_deg=float(ukf_cfg.get("process_std_theta_deg", 0.6)),
            process_std_v_sigma=float(ukf_cfg.get("process_std_v_sigma", 0.002)),
            process_std_v_mu_q=float(ukf_cfg.get("process_std_v_mu_q", 0.002)),
            process_std_v_mu_p=float(ukf_cfg.get("process_std_v_mu_p", 0.002)),
            process_std_v_theta_deg=float(ukf_cfg.get("process_std_v_theta_deg", 0.2)),
            measurement_std_sigma=float(ukf_cfg.get("measurement_std_sigma", 0.08)),
            measurement_std_mu_q=float(ukf_cfg.get("measurement_std_mu_q", 0.05)),
            measurement_std_mu_p=float(ukf_cfg.get("measurement_std_mu_p", 0.05)),
            measurement_std_theta_deg=float(ukf_cfg.get("measurement_std_theta_deg", 4.0)),
            velocity_decay=float(ukf_cfg.get("velocity_decay", 1.0)),
            covariance_floor=float(ukf_cfg.get("covariance_floor", 1.0e-6)),
            alpha=float(ukf_cfg.get("alpha", 0.3)),
            beta=float(ukf_cfg.get("beta", 2.0)),
            kappa=float(ukf_cfg.get("kappa", 0.0)),
        )


class UKFBaseline:
    """Unscented filter with constant-velocity latent state and histogram-moment measurement."""

    def __init__(self, config: UKFBaselineConfig, *, measurement_baseline: WindowVarianceBaseline) -> None:
        self.config = config
        self.measurement_baseline = measurement_baseline
        self._dim = 8
        self._state = np.array(
            [
                np.clip(self.config.initial_sigma, *self.config.sigma_clip),
                np.clip(self.config.initial_mu_q, *self.config.mu_clip),
                np.clip(self.config.initial_mu_p, *self.config.mu_clip),
                np.clip(self.config.initial_theta_deg, *self.config.theta_clip_deg),
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        self._covariance = np.diag(
            [
                max(self.config.measurement_std_sigma**2, self.config.covariance_floor),
                max(self.config.measurement_std_mu_q**2, self.config.covariance_floor),
                max(self.config.measurement_std_mu_p**2, self.config.covariance_floor),
                max(self.config.measurement_std_theta_deg**2, self.config.covariance_floor),
                max(self.config.initial_velocity_std_sigma**2, self.config.covariance_floor),
                max(self.config.initial_velocity_std_mu_q**2, self.config.covariance_floor),
                max(self.config.initial_velocity_std_mu_p**2, self.config.covariance_floor),
                max(self.config.initial_velocity_std_theta_deg**2, self.config.covariance_floor),
            ]
        ).astype(float)
        self._process_cov = np.diag(
            [
                self.config.process_std_sigma**2,
                self.config.process_std_mu_q**2,
                self.config.process_std_mu_p**2,
                self.config.process_std_theta_deg**2,
                self.config.process_std_v_sigma**2,
                self.config.process_std_v_mu_q**2,
                self.config.process_std_v_mu_p**2,
                self.config.process_std_v_theta_deg**2,
            ]
        ).astype(float)
        self._measurement_cov = np.diag(
            [
                self.config.measurement_std_sigma**2,
                self.config.measurement_std_mu_q**2,
                self.config.measurement_std_mu_p**2,
                self.config.measurement_std_theta_deg**2,
            ]
        ).astype(float)
        self._covariance = _stabilize_covariance(self._covariance, self.config.covariance_floor)
        self._process_cov = _stabilize_covariance(self._process_cov, self.config.covariance_floor)
        self._measurement_cov = _stabilize_covariance(self._measurement_cov, self.config.covariance_floor)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "UKFBaseline":
        return cls(
            UKFBaselineConfig.from_config(config),
            measurement_baseline=WindowVarianceBaseline.from_config(config),
        )

    def _clip_state(self, state: np.ndarray) -> np.ndarray:
        clipped = np.asarray(state, dtype=float).copy()
        clipped[0] = float(np.clip(clipped[0], *self.config.sigma_clip))
        clipped[1] = float(np.clip(clipped[1], *self.config.mu_clip))
        clipped[2] = float(np.clip(clipped[2], *self.config.mu_clip))
        clipped[3] = float(np.clip(_wrap_theta_deg(clipped[3]), *self.config.theta_clip_deg))
        return _wrap_state_theta(clipped)

    def _process_model(self, state: np.ndarray) -> np.ndarray:
        next_state = np.asarray(state, dtype=float).copy()
        next_state[0] += next_state[4]
        next_state[1] += next_state[5]
        next_state[2] += next_state[6]
        next_state[3] = _wrap_theta_deg(next_state[3] + next_state[7])
        next_state[4:] *= self.config.velocity_decay
        return self._clip_state(next_state)

    def _measurement_model(self, state: np.ndarray) -> np.ndarray:
        clipped = self._clip_state(state)
        return np.array([clipped[0], clipped[1], clipped[2], clipped[3]], dtype=float)

    def _sigma_points(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self._dim
        alpha = self.config.alpha
        beta = self.config.beta
        kappa = self.config.kappa
        lam = alpha**2 * (n + kappa) - n
        scaled = (n + lam) * _stabilize_covariance(covariance, self.config.covariance_floor)
        jitter = np.eye(n, dtype=float) * self.config.covariance_floor
        chol = np.linalg.cholesky(scaled + jitter)
        points = [mean.copy()]
        for idx in range(n):
            column = chol[:, idx]
            points.append(self._clip_state(mean + column))
            points.append(self._clip_state(mean - column))
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - alpha**2 + beta)
        return np.asarray(points, dtype=float), wm, wc

    def _state_mean(self, sigma_points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mean = np.sum(weights[:, None] * sigma_points, axis=0)
        mean[3] = _weighted_theta_mean_deg(sigma_points[:, 3], weights)
        mean[7] = _weighted_theta_mean_deg(sigma_points[:, 7], weights)
        return self._clip_state(mean)

    def _measurement_mean(self, sigma_measurements: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mean = np.sum(weights[:, None] * sigma_measurements, axis=0)
        mean[3] = _weighted_theta_mean_deg(sigma_measurements[:, 3], weights)
        mean[3] = float(np.clip(mean[3], *self.config.theta_clip_deg))
        return mean

    def _state_residual(self, value: np.ndarray, mean: np.ndarray) -> np.ndarray:
        residual = np.asarray(value, dtype=float) - np.asarray(mean, dtype=float)
        residual[3] = _wrap_theta_deg(residual[3])
        residual[7] = _wrap_theta_deg(residual[7])
        return residual

    def _measurement_residual(self, value: np.ndarray, mean: np.ndarray) -> np.ndarray:
        residual = np.asarray(value, dtype=float) - np.asarray(mean, dtype=float)
        residual[3] = _wrap_theta_deg(residual[3])
        return residual

    def predict(self, histogram: np.ndarray, *, window_id: int | None = None) -> NoisePrediction:
        measurement = self.measurement_baseline.predict(histogram, window_id=window_id)
        sigma_points, wm, wc = self._sigma_points(self._state, self._covariance)
        predicted_points = np.asarray([self._process_model(point) for point in sigma_points], dtype=float)
        predicted_mean = self._state_mean(predicted_points, wm)

        predicted_cov = self._process_cov.copy()
        for idx, point in enumerate(predicted_points):
            residual = self._state_residual(point, predicted_mean)
            predicted_cov += wc[idx] * np.outer(residual, residual)
        predicted_cov = _stabilize_covariance(predicted_cov, self.config.covariance_floor)

        predicted_measurements = np.asarray(
            [self._measurement_model(point) for point in predicted_points],
            dtype=float,
        )
        measurement_mean = self._measurement_mean(predicted_measurements, wm)

        innovation_cov = self._measurement_cov.copy()
        cross_cov = np.zeros((self._dim, 4), dtype=float)
        for idx, (point, meas_point) in enumerate(zip(predicted_points, predicted_measurements)):
            state_residual = self._state_residual(point, predicted_mean)
            meas_residual = self._measurement_residual(meas_point, measurement_mean)
            innovation_cov += wc[idx] * np.outer(meas_residual, meas_residual)
            cross_cov += wc[idx] * np.outer(state_residual, meas_residual)
        innovation_cov = _stabilize_covariance(innovation_cov, self.config.covariance_floor)

        z = np.array([measurement.sigma, measurement.mu_q, measurement.mu_p, measurement.theta_deg], dtype=float)
        innovation = self._measurement_residual(z, measurement_mean)
        kalman_gain = np.linalg.solve(innovation_cov.T, cross_cov.T).T
        updated_state = predicted_mean + kalman_gain @ innovation
        updated_cov = predicted_cov - kalman_gain @ innovation_cov @ kalman_gain.T

        self._state = self._clip_state(updated_state)
        self._covariance = _stabilize_covariance(updated_cov, self.config.covariance_floor)

        metadata = {
            "window_id": window_id,
            "measurement_prediction": measurement.to_dict(),
            "innovation": {
                "sigma": float(innovation[0]),
                "mu_q": float(innovation[1]),
                "mu_p": float(innovation[2]),
                "theta_deg": float(innovation[3]),
            },
            "state_after_update": {
                "sigma": float(self._state[0]),
                "mu_q": float(self._state[1]),
                "mu_p": float(self._state[2]),
                "theta_deg": float(self._state[3]),
                "v_sigma": float(self._state[4]),
                "v_mu_q": float(self._state[5]),
                "v_mu_p": float(self._state[6]),
                "v_theta_deg": float(self._state[7]),
            },
            "kalman_gain_diag": np.diag(kalman_gain[:4, :4]).astype(float).tolist(),
            "covariance_diag": np.diag(self._covariance).astype(float).tolist(),
            "ukf_params": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "kappa": self.config.kappa,
                "velocity_decay": self.config.velocity_decay,
            },
        }
        return NoisePrediction(
            sigma=float(self._state[0]),
            mu_q=float(self._state[1]),
            mu_p=float(self._state[2]),
            theta_deg=float(self._state[3]),
            source="ukf_baseline",
            metadata=metadata,
        )
