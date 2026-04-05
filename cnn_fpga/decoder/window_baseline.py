"""Window-statistics baseline built from histogram moments."""

# 中文说明：
# - 该模块提供 P4 中的 `Window Variance` 正式基线。
# - 它只使用当前窗口直方图，不使用任何未来信息，也不依赖训练模型。
# - 核心思想是：从窗口 histogram 重建均值/协方差，再将其压缩为
#   `(sigma, mu_q, mu_p, theta_deg)`，交给 `ParamMapper` 走与主模型一致的运行时链路。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from cnn_fpga.decoder.param_mapper import NoisePrediction


def _safe_clip_pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    raw = default if value is None else value
    lower = float(raw[0])
    upper = float(raw[1])
    if lower > upper:
        raise ValueError("clip lower bound must be <= upper bound")
    return lower, upper


def _wrap_theta_deg(theta_deg: float) -> float:
    """Normalize angle to [-90, 90) for ellipse-axis interpretation."""
    wrapped = ((float(theta_deg) + 90.0) % 180.0) - 90.0
    return wrapped


@dataclass(frozen=True)
class WindowVarianceConfig:
    """Configuration of histogram-moment estimation."""

    histogram_bins: int
    histogram_range_limit: float
    sigma_clip: Tuple[float, float]
    mu_clip: Tuple[float, float]
    theta_clip_deg: Tuple[float, float]
    theta_default_deg: float
    min_anisotropy_ratio: float
    sigma_ratio_p: float
    measurement_var_floor: float

    def __post_init__(self) -> None:
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")
        if self.histogram_range_limit <= 0:
            raise ValueError("histogram_range_limit must be positive")
        if self.sigma_clip[0] <= 0 or self.sigma_clip[0] > self.sigma_clip[1]:
            raise ValueError("sigma_clip must be positive and ordered")
        if self.mu_clip[0] > self.mu_clip[1]:
            raise ValueError("mu_clip must be ordered")
        if self.theta_clip_deg[0] > self.theta_clip_deg[1]:
            raise ValueError("theta_clip_deg must be ordered")
        if self.min_anisotropy_ratio < 0:
            raise ValueError("min_anisotropy_ratio must be non-negative")
        if self.sigma_ratio_p <= 0:
            raise ValueError("sigma_ratio_p must be positive")
        if self.measurement_var_floor < 0:
            raise ValueError("measurement_var_floor must be non-negative")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowVarianceConfig":
        fast_cfg = config.get("fast_loop", {})
        measurement_cfg = config.get("measurement", {})
        slow_cfg = config.get("slow_loop", {})
        mode_cfg = slow_cfg.get("window_variance", {})
        mapping_cfg = config.get("param_mapping", {})
        model_cfg = config.get("model", {})

        delta_eff = float(measurement_cfg.get("delta", 0.3))
        eta = float(measurement_cfg.get("measurement_efficiency", 0.95))
        if eta <= 0:
            raise ValueError("measurement_efficiency must be positive")
        sigma_meas = float(np.sqrt(max(0.0, (1.0 - eta) / (2.0 * eta))))

        hist_limit = float(
            fast_cfg.get("histogram_range_limit", fast_cfg.get("syndrome_limit", 1.0))
        )
        sigma_lower, sigma_upper = _safe_clip_pair(
            mode_cfg.get("sigma_clip"),
            (0.05, max(0.06, hist_limit)),
        )
        mu_lower, mu_upper = _safe_clip_pair(
            mode_cfg.get("mu_clip"),
            (-hist_limit, hist_limit),
        )
        theta_lower, theta_upper = _safe_clip_pair(
            mode_cfg.get("theta_clip_deg", mapping_cfg.get("theta_clip_deg", [-20.0, 20.0])),
            (-20.0, 20.0),
        )
        return cls(
            histogram_bins=int(fast_cfg.get("histogram_bins", 32)),
            histogram_range_limit=hist_limit,
            sigma_clip=(sigma_lower, sigma_upper),
            mu_clip=(mu_lower, mu_upper),
            theta_clip_deg=(theta_lower, theta_upper),
            theta_default_deg=float(mode_cfg.get("theta_default_deg", 0.0)),
            min_anisotropy_ratio=float(mode_cfg.get("min_anisotropy_ratio", 0.05)),
            sigma_ratio_p=float(mode_cfg.get("sigma_ratio_p", model_cfg.get("sigma_ratio_p", 1.0))),
            measurement_var_floor=float(sigma_meas**2 + delta_eff**2),
        )


@dataclass(frozen=True)
class HistogramMomentObservation:
    """Moment summary reconstructed from a normalized histogram."""

    valid: bool
    total_mass: float
    mean_q: float
    mean_p: float
    covariance: np.ndarray
    eigenvalues: np.ndarray
    principal_axis: np.ndarray
    anisotropy_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "total_mass": float(self.total_mass),
            "mean_q": float(self.mean_q),
            "mean_p": float(self.mean_p),
            "covariance": np.asarray(self.covariance, dtype=float).tolist(),
            "eigenvalues": np.asarray(self.eigenvalues, dtype=float).tolist(),
            "principal_axis": np.asarray(self.principal_axis, dtype=float).tolist(),
            "anisotropy_ratio": float(self.anisotropy_ratio),
        }


class HistogramMomentEstimator:
    """Reconstruct first/second moments from the window histogram."""

    def __init__(self, config: WindowVarianceConfig) -> None:
        self.config = config
        edges = np.linspace(
            -self.config.histogram_range_limit,
            self.config.histogram_range_limit,
            self.config.histogram_bins + 1,
            dtype=float,
        )
        self._centers = 0.5 * (edges[:-1] + edges[1:])

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HistogramMomentEstimator":
        return cls(WindowVarianceConfig.from_config(config))

    def observe(self, histogram: np.ndarray) -> HistogramMomentObservation:
        hist = np.asarray(histogram, dtype=float)
        if hist.shape != (self.config.histogram_bins, self.config.histogram_bins):
            raise ValueError(
                f"histogram must have shape {(self.config.histogram_bins, self.config.histogram_bins)}, "
                f"got {hist.shape}"
            )
        hist = np.nan_to_num(hist, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        total_mass = float(np.sum(hist))
        if total_mass <= 0.0:
            return HistogramMomentObservation(
                valid=False,
                total_mass=0.0,
                mean_q=0.0,
                mean_p=0.0,
                covariance=np.zeros((2, 2), dtype=float),
                eigenvalues=np.zeros(2, dtype=float),
                principal_axis=np.array([1.0, 0.0], dtype=float),
                anisotropy_ratio=0.0,
            )

        weights = hist / total_mass
        centers_q = self._centers[:, None]
        centers_p = self._centers[None, :]
        mean_q = float(np.sum(weights * centers_q))
        mean_p = float(np.sum(weights * centers_p))
        dq = centers_q - mean_q
        dp = centers_p - mean_p
        cov_qq = float(np.sum(weights * dq * dq))
        cov_pp = float(np.sum(weights * dp * dp))
        cov_qp = float(np.sum(weights * dq * dp))
        covariance = np.array([[cov_qq, cov_qp], [cov_qp, cov_pp]], dtype=float)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.maximum(eigvals, 0.0)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        principal_axis = eigvecs[:, 0]
        anisotropy_ratio = float(
            0.0
            if eigvals[0] <= 1.0e-12
            else np.clip((eigvals[0] - eigvals[1]) / eigvals[0], 0.0, 1.0)
        )
        return HistogramMomentObservation(
            valid=True,
            total_mass=total_mass,
            mean_q=mean_q,
            mean_p=mean_p,
            covariance=covariance,
            eigenvalues=eigvals,
            principal_axis=principal_axis,
            anisotropy_ratio=anisotropy_ratio,
        )


class WindowVarianceBaseline:
    """Current-window baseline without temporal filtering."""

    def __init__(self, config: WindowVarianceConfig) -> None:
        self.config = config
        self.moment_estimator = HistogramMomentEstimator(config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WindowVarianceBaseline":
        return cls(WindowVarianceConfig.from_config(config))

    def _fallback_prediction(self, *, source: str, metadata: Dict[str, Any]) -> NoisePrediction:
        return NoisePrediction(
            sigma=float(self.config.sigma_clip[0]),
            mu_q=0.0,
            mu_p=0.0,
            theta_deg=float(np.clip(self.config.theta_default_deg, *self.config.theta_clip_deg)),
            source=source,
            metadata=metadata,
        )

    def predict(self, histogram: np.ndarray, *, window_id: int | None = None) -> NoisePrediction:
        observation = self.moment_estimator.observe(histogram)
        metadata = {
            "window_id": window_id,
            "observation": observation.to_dict(),
            "measurement_var_floor": float(self.config.measurement_var_floor),
            "sigma_ratio_p": float(self.config.sigma_ratio_p),
        }
        if not observation.valid:
            metadata["fallback_reason"] = "empty_histogram"
            return self._fallback_prediction(source="window_variance", metadata=metadata)

        # 中文注释：
        # - 主模型运行时假设椭圆长短轴比例由 `sigma_ratio_p` 固定，因此这里用去偏后的总方差
        #   反推主轴 sigma，而不是直接把某一条轴的观测标准差当成 sigma。
        trace_signal = max(
            0.0,
            float(np.trace(observation.covariance) - 2.0 * self.config.measurement_var_floor),
        )
        sigma_est = float(np.sqrt(trace_signal / max(1.0e-8, 1.0 + self.config.sigma_ratio_p**2)))
        sigma_est = float(np.clip(sigma_est, self.config.sigma_clip[0], self.config.sigma_clip[1]))

        mu_q = float(np.clip(observation.mean_q, self.config.mu_clip[0], self.config.mu_clip[1]))
        mu_p = float(np.clip(observation.mean_p, self.config.mu_clip[0], self.config.mu_clip[1]))

        if observation.anisotropy_ratio < self.config.min_anisotropy_ratio:
            theta_deg = float(self.config.theta_default_deg)
            metadata["theta_fallback"] = "low_anisotropy"
        else:
            theta_deg = float(np.rad2deg(np.arctan2(observation.principal_axis[1], observation.principal_axis[0])))
            theta_deg = _wrap_theta_deg(theta_deg)
        theta_deg = float(np.clip(theta_deg, self.config.theta_clip_deg[0], self.config.theta_clip_deg[1]))

        metadata.update(
            {
                "sigma_estimate_raw": float(np.sqrt(max(0.0, trace_signal / max(1.0e-8, 1.0 + self.config.sigma_ratio_p**2)))),
                "mu_q_estimate_raw": float(observation.mean_q),
                "mu_p_estimate_raw": float(observation.mean_p),
                "theta_estimate_raw_deg": float(theta_deg),
            }
        )
        return NoisePrediction(
            sigma=sigma_est,
            mu_q=mu_q,
            mu_p=mu_p,
            theta_deg=theta_deg,
            source="window_variance",
            metadata=metadata,
        )
