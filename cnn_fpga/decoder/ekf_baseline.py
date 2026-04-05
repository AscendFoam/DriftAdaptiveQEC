"""Recursive EKF-style baseline on top of histogram moment observations."""

# 中文说明：
# - 该模块提供 P4 中的 `EKF` 正式基线。
# - 它不依赖训练模型，而是把窗口统计量视作观测，使用随机游走状态模型递推
#   `(sigma, mu_q, mu_p, theta_deg)`。
# - 这里的实现是工程上可行的“EKF风格”追踪器：观测提取是非线性的，
#   但滤波器内部采用线性化后的对角协方差更新，重点是提供一个稳定、可复现的
#   经典滤波 baseline。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from cnn_fpga.decoder.param_mapper import NoisePrediction
from cnn_fpga.decoder.window_baseline import WindowVarianceBaseline, WindowVarianceConfig


def _wrap_innovation_theta(theta_deg: float) -> float:
    """Wrap angle innovation to [-90, 90) to avoid EKF jumps."""
    return ((float(theta_deg) + 90.0) % 180.0) - 90.0


@dataclass(frozen=True)
class EKFBaselineConfig:
    """Configuration of the recursive baseline filter."""

    sigma_clip: tuple[float, float]
    mu_clip: tuple[float, float]
    theta_clip_deg: tuple[float, float]
    initial_sigma: float
    initial_mu_q: float
    initial_mu_p: float
    initial_theta_deg: float
    process_std_sigma: float
    process_std_mu_q: float
    process_std_mu_p: float
    process_std_theta_deg: float
    measurement_std_sigma: float
    measurement_std_mu_q: float
    measurement_std_mu_p: float
    measurement_std_theta_deg: float
    covariance_floor: float

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
            ("process_std_sigma", self.process_std_sigma),
            ("process_std_mu_q", self.process_std_mu_q),
            ("process_std_mu_p", self.process_std_mu_p),
            ("process_std_theta_deg", self.process_std_theta_deg),
            ("measurement_std_sigma", self.measurement_std_sigma),
            ("measurement_std_mu_q", self.measurement_std_mu_q),
            ("measurement_std_mu_p", self.measurement_std_mu_p),
            ("measurement_std_theta_deg", self.measurement_std_theta_deg),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.covariance_floor <= 0:
            raise ValueError("covariance_floor must be positive")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EKFBaselineConfig":
        slow_cfg = config.get("slow_loop", {})
        ekf_cfg = slow_cfg.get("ekf", {})
        window_cfg = WindowVarianceConfig.from_config(config)
        fixed_cfg = slow_cfg.get("fixed_baseline", {})
        return cls(
            sigma_clip=window_cfg.sigma_clip,
            mu_clip=window_cfg.mu_clip,
            theta_clip_deg=window_cfg.theta_clip_deg,
            initial_sigma=float(ekf_cfg.get("initial_sigma", fixed_cfg.get("sigma", window_cfg.sigma_clip[0]))),
            initial_mu_q=float(ekf_cfg.get("initial_mu_q", fixed_cfg.get("mu_q", 0.0))),
            initial_mu_p=float(ekf_cfg.get("initial_mu_p", fixed_cfg.get("mu_p", 0.0))),
            initial_theta_deg=float(ekf_cfg.get("initial_theta_deg", fixed_cfg.get("theta_deg", 0.0))),
            process_std_sigma=float(ekf_cfg.get("process_std_sigma", 0.015)),
            process_std_mu_q=float(ekf_cfg.get("process_std_mu_q", 0.01)),
            process_std_mu_p=float(ekf_cfg.get("process_std_mu_p", 0.01)),
            process_std_theta_deg=float(ekf_cfg.get("process_std_theta_deg", 1.0)),
            measurement_std_sigma=float(ekf_cfg.get("measurement_std_sigma", 0.08)),
            measurement_std_mu_q=float(ekf_cfg.get("measurement_std_mu_q", 0.05)),
            measurement_std_mu_p=float(ekf_cfg.get("measurement_std_mu_p", 0.05)),
            measurement_std_theta_deg=float(ekf_cfg.get("measurement_std_theta_deg", 4.0)),
            covariance_floor=float(ekf_cfg.get("covariance_floor", 1.0e-6)),
        )


class EKFBaseline:
    """Recursive histogram baseline using a diagonalized EKF-like update."""

    def __init__(self, config: EKFBaselineConfig, *, measurement_baseline: WindowVarianceBaseline) -> None:
        self.config = config
        self.measurement_baseline = measurement_baseline
        self._state = np.array(
            [
                np.clip(self.config.initial_sigma, *self.config.sigma_clip),
                np.clip(self.config.initial_mu_q, *self.config.mu_clip),
                np.clip(self.config.initial_mu_p, *self.config.mu_clip),
                np.clip(self.config.initial_theta_deg, *self.config.theta_clip_deg),
            ],
            dtype=float,
        )
        self._covariance = np.diag(
            [
                max(self.config.measurement_std_sigma**2, self.config.covariance_floor),
                max(self.config.measurement_std_mu_q**2, self.config.covariance_floor),
                max(self.config.measurement_std_mu_p**2, self.config.covariance_floor),
                max(self.config.measurement_std_theta_deg**2, self.config.covariance_floor),
            ]
        ).astype(float)
        self._process_cov = np.diag(
            [
                self.config.process_std_sigma**2,
                self.config.process_std_mu_q**2,
                self.config.process_std_mu_p**2,
                self.config.process_std_theta_deg**2,
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

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EKFBaseline":
        return cls(
            EKFBaselineConfig.from_config(config),
            measurement_baseline=WindowVarianceBaseline.from_config(config),
        )

    def _clip_state(self) -> None:
        self._state[0] = float(np.clip(self._state[0], *self.config.sigma_clip))
        self._state[1] = float(np.clip(self._state[1], *self.config.mu_clip))
        self._state[2] = float(np.clip(self._state[2], *self.config.mu_clip))
        self._state[3] = float(np.clip(self._state[3], *self.config.theta_clip_deg))

    def predict(self, histogram: np.ndarray, *, window_id: int | None = None) -> NoisePrediction:
        measurement = self.measurement_baseline.predict(histogram, window_id=window_id)
        predicted_state = self._state.copy()
        predicted_cov = self._covariance + self._process_cov

        z = np.array([measurement.sigma, measurement.mu_q, measurement.mu_p, measurement.theta_deg], dtype=float)
        innovation = z - predicted_state
        innovation[3] = _wrap_innovation_theta(innovation[3])

        innovation_cov = predicted_cov + self._measurement_cov
        kalman_gain = predicted_cov @ np.linalg.inv(innovation_cov)
        updated_state = predicted_state + kalman_gain @ innovation
        identity = np.eye(4, dtype=float)
        updated_cov = (identity - kalman_gain) @ predicted_cov

        self._state = updated_state
        self._covariance = 0.5 * (updated_cov + updated_cov.T)
        diag = np.maximum(np.diag(self._covariance), self.config.covariance_floor)
        self._covariance = np.diag(diag)
        self._clip_state()

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
            },
            "kalman_gain_diag": np.diag(kalman_gain).astype(float).tolist(),
            "covariance_diag": np.diag(self._covariance).astype(float).tolist(),
        }
        return NoisePrediction(
            sigma=float(self._state[0]),
            mu_q=float(self._state[1]),
            mu_p=float(self._state[2]),
            theta_deg=float(self._state[3]),
            source="ekf_baseline",
            metadata=metadata,
        )
