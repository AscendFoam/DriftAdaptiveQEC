"""Bootstrap particle-filter baseline on top of histogram-moment observations."""

# 中文说明：
# - 该模块实现一条更强的经典滤波 baseline：`Particle Filter`。
# - 它不依赖线性化或高斯近似，适合用于补充 `EKF/UKF` 之后的论文级对照。
# - 当前实现保持与现有工程主线一致：
#   1. 状态量仍是 `(sigma, mu_q, mu_p, theta_deg)`
#   2. 观测仍来自当前窗口 histogram 的 `Window Variance`
#   3. 因此保持严格因果、公平比较

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from cnn_fpga.decoder.param_mapper import NoisePrediction
from cnn_fpga.decoder.window_baseline import WindowVarianceBaseline, WindowVarianceConfig


def _wrap_theta_deg(theta_deg: np.ndarray | float) -> np.ndarray | float:
    wrapped = (np.asarray(theta_deg, dtype=float) + 90.0) % 180.0 - 90.0
    if np.isscalar(theta_deg):
        return float(wrapped)
    return wrapped


def _weighted_theta_mean_deg(values_deg: np.ndarray, weights: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(values_deg, dtype=float))
    sin_mean = float(np.sum(weights * np.sin(radians)))
    cos_mean = float(np.sum(weights * np.cos(radians)))
    if abs(sin_mean) < 1.0e-12 and abs(cos_mean) < 1.0e-12:
        return 0.0
    return float(_wrap_theta_deg(np.rad2deg(np.arctan2(sin_mean, cos_mean))))


@dataclass(frozen=True)
class ParticleFilterBaselineConfig:
    """Configuration of the bootstrap particle filter baseline."""

    sigma_clip: tuple[float, float]
    mu_clip: tuple[float, float]
    theta_clip_deg: tuple[float, float]
    n_particles: int
    initial_sigma: float
    initial_mu_q: float
    initial_mu_p: float
    initial_theta_deg: float
    initial_std_sigma: float
    initial_std_mu_q: float
    initial_std_mu_p: float
    initial_std_theta_deg: float
    process_std_sigma: float
    process_std_mu_q: float
    process_std_mu_p: float
    process_std_theta_deg: float
    measurement_std_sigma: float
    measurement_std_mu_q: float
    measurement_std_mu_p: float
    measurement_std_theta_deg: float
    resample_ess_ratio: float
    rejuvenation_std_sigma: float
    rejuvenation_std_mu_q: float
    rejuvenation_std_mu_p: float
    rejuvenation_std_theta_deg: float

    def __post_init__(self) -> None:
        if self.sigma_clip[0] <= 0 or self.sigma_clip[0] > self.sigma_clip[1]:
            raise ValueError("sigma_clip must be positive and ordered")
        if self.mu_clip[0] > self.mu_clip[1]:
            raise ValueError("mu_clip must be ordered")
        if self.theta_clip_deg[0] > self.theta_clip_deg[1]:
            raise ValueError("theta_clip_deg must be ordered")
        if self.n_particles < 32:
            raise ValueError("n_particles must be >= 32")
        if self.initial_sigma <= 0:
            raise ValueError("initial_sigma must be positive")
        for name, value in (
            ("initial_std_sigma", self.initial_std_sigma),
            ("initial_std_mu_q", self.initial_std_mu_q),
            ("initial_std_mu_p", self.initial_std_mu_p),
            ("initial_std_theta_deg", self.initial_std_theta_deg),
            ("process_std_sigma", self.process_std_sigma),
            ("process_std_mu_q", self.process_std_mu_q),
            ("process_std_mu_p", self.process_std_mu_p),
            ("process_std_theta_deg", self.process_std_theta_deg),
            ("measurement_std_sigma", self.measurement_std_sigma),
            ("measurement_std_mu_q", self.measurement_std_mu_q),
            ("measurement_std_mu_p", self.measurement_std_mu_p),
            ("measurement_std_theta_deg", self.measurement_std_theta_deg),
            ("rejuvenation_std_sigma", self.rejuvenation_std_sigma),
            ("rejuvenation_std_mu_q", self.rejuvenation_std_mu_q),
            ("rejuvenation_std_mu_p", self.rejuvenation_std_mu_p),
            ("rejuvenation_std_theta_deg", self.rejuvenation_std_theta_deg),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 < self.resample_ess_ratio <= 1.0):
            raise ValueError("resample_ess_ratio must be within (0, 1]")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ParticleFilterBaselineConfig":
        slow_cfg = config.get("slow_loop", {})
        pf_cfg = slow_cfg.get("particle_filter", {})
        window_cfg = WindowVarianceConfig.from_config(config)
        fixed_cfg = slow_cfg.get("fixed_baseline", {})
        return cls(
            sigma_clip=window_cfg.sigma_clip,
            mu_clip=window_cfg.mu_clip,
            theta_clip_deg=window_cfg.theta_clip_deg,
            n_particles=int(pf_cfg.get("n_particles", 256)),
            initial_sigma=float(pf_cfg.get("initial_sigma", fixed_cfg.get("sigma", window_cfg.sigma_clip[0]))),
            initial_mu_q=float(pf_cfg.get("initial_mu_q", fixed_cfg.get("mu_q", 0.0))),
            initial_mu_p=float(pf_cfg.get("initial_mu_p", fixed_cfg.get("mu_p", 0.0))),
            initial_theta_deg=float(pf_cfg.get("initial_theta_deg", fixed_cfg.get("theta_deg", 0.0))),
            initial_std_sigma=float(pf_cfg.get("initial_std_sigma", 0.06)),
            initial_std_mu_q=float(pf_cfg.get("initial_std_mu_q", 0.04)),
            initial_std_mu_p=float(pf_cfg.get("initial_std_mu_p", 0.04)),
            initial_std_theta_deg=float(pf_cfg.get("initial_std_theta_deg", 5.0)),
            process_std_sigma=float(pf_cfg.get("process_std_sigma", 0.012)),
            process_std_mu_q=float(pf_cfg.get("process_std_mu_q", 0.01)),
            process_std_mu_p=float(pf_cfg.get("process_std_mu_p", 0.01)),
            process_std_theta_deg=float(pf_cfg.get("process_std_theta_deg", 0.9)),
            measurement_std_sigma=float(pf_cfg.get("measurement_std_sigma", 0.08)),
            measurement_std_mu_q=float(pf_cfg.get("measurement_std_mu_q", 0.05)),
            measurement_std_mu_p=float(pf_cfg.get("measurement_std_mu_p", 0.05)),
            measurement_std_theta_deg=float(pf_cfg.get("measurement_std_theta_deg", 4.0)),
            resample_ess_ratio=float(pf_cfg.get("resample_ess_ratio", 0.5)),
            rejuvenation_std_sigma=float(pf_cfg.get("rejuvenation_std_sigma", 0.003)),
            rejuvenation_std_mu_q=float(pf_cfg.get("rejuvenation_std_mu_q", 0.0025)),
            rejuvenation_std_mu_p=float(pf_cfg.get("rejuvenation_std_mu_p", 0.0025)),
            rejuvenation_std_theta_deg=float(pf_cfg.get("rejuvenation_std_theta_deg", 0.25)),
        )


class ParticleFilterBaseline:
    """Bootstrap particle filter using histogram moments as measurement."""

    def __init__(
        self,
        config: ParticleFilterBaselineConfig,
        *,
        measurement_baseline: WindowVarianceBaseline,
        seed: int | None = None,
    ) -> None:
        self.config = config
        self.measurement_baseline = measurement_baseline
        self._rng = np.random.default_rng(seed)
        self._particles = np.zeros((self.config.n_particles, 4), dtype=float)
        self._weights = np.full(self.config.n_particles, 1.0 / self.config.n_particles, dtype=float)
        self._initialize_particles()

    @classmethod
    def from_config(cls, config: Dict[str, Any], seed: int | None = None) -> "ParticleFilterBaseline":
        return cls(
            ParticleFilterBaselineConfig.from_config(config),
            measurement_baseline=WindowVarianceBaseline.from_config(config),
            seed=seed,
        )

    def _clip_particles(self) -> None:
        self._particles[:, 0] = np.clip(self._particles[:, 0], *self.config.sigma_clip)
        self._particles[:, 1] = np.clip(self._particles[:, 1], *self.config.mu_clip)
        self._particles[:, 2] = np.clip(self._particles[:, 2], *self.config.mu_clip)
        self._particles[:, 3] = np.clip(_wrap_theta_deg(self._particles[:, 3]), *self.config.theta_clip_deg)

    def _initialize_particles(self) -> None:
        self._particles[:, 0] = self.config.initial_sigma + self._rng.normal(
            0.0,
            self.config.initial_std_sigma,
            size=self.config.n_particles,
        )
        self._particles[:, 1] = self.config.initial_mu_q + self._rng.normal(
            0.0,
            self.config.initial_std_mu_q,
            size=self.config.n_particles,
        )
        self._particles[:, 2] = self.config.initial_mu_p + self._rng.normal(
            0.0,
            self.config.initial_std_mu_p,
            size=self.config.n_particles,
        )
        self._particles[:, 3] = self.config.initial_theta_deg + self._rng.normal(
            0.0,
            self.config.initial_std_theta_deg,
            size=self.config.n_particles,
        )
        self._clip_particles()

    def _predict_particles(self) -> None:
        self._particles[:, 0] += self._rng.normal(0.0, self.config.process_std_sigma, size=self.config.n_particles)
        self._particles[:, 1] += self._rng.normal(0.0, self.config.process_std_mu_q, size=self.config.n_particles)
        self._particles[:, 2] += self._rng.normal(0.0, self.config.process_std_mu_p, size=self.config.n_particles)
        self._particles[:, 3] += self._rng.normal(0.0, self.config.process_std_theta_deg, size=self.config.n_particles)
        self._clip_particles()

    def _systematic_resample(self) -> None:
        cumulative = np.cumsum(self._weights)
        positions = (self._rng.random() + np.arange(self.config.n_particles, dtype=float)) / self.config.n_particles
        indices = np.searchsorted(cumulative, positions, side="left")
        self._particles = self._particles[indices].copy()
        self._particles[:, 0] += self._rng.normal(0.0, self.config.rejuvenation_std_sigma, size=self.config.n_particles)
        self._particles[:, 1] += self._rng.normal(0.0, self.config.rejuvenation_std_mu_q, size=self.config.n_particles)
        self._particles[:, 2] += self._rng.normal(0.0, self.config.rejuvenation_std_mu_p, size=self.config.n_particles)
        self._particles[:, 3] += self._rng.normal(0.0, self.config.rejuvenation_std_theta_deg, size=self.config.n_particles)
        self._clip_particles()
        self._weights.fill(1.0 / self.config.n_particles)

    def _estimate_state(self) -> np.ndarray:
        estimate = np.sum(self._weights[:, None] * self._particles, axis=0)
        estimate[3] = _weighted_theta_mean_deg(self._particles[:, 3], self._weights)
        estimate[0] = float(np.clip(estimate[0], *self.config.sigma_clip))
        estimate[1] = float(np.clip(estimate[1], *self.config.mu_clip))
        estimate[2] = float(np.clip(estimate[2], *self.config.mu_clip))
        estimate[3] = float(np.clip(estimate[3], *self.config.theta_clip_deg))
        return estimate

    def _update_weights(self, measurement_vector: np.ndarray) -> Dict[str, float]:
        sigma_res = (measurement_vector[0] - self._particles[:, 0]) / self.config.measurement_std_sigma
        mu_q_res = (measurement_vector[1] - self._particles[:, 1]) / self.config.measurement_std_mu_q
        mu_p_res = (measurement_vector[2] - self._particles[:, 2]) / self.config.measurement_std_mu_p
        theta_delta = np.asarray(_wrap_theta_deg(measurement_vector[3] - self._particles[:, 3]), dtype=float)
        theta_res = theta_delta / self.config.measurement_std_theta_deg
        log_likelihood = -0.5 * (sigma_res**2 + mu_q_res**2 + mu_p_res**2 + theta_res**2)
        log_likelihood -= float(np.max(log_likelihood))
        likelihood = np.exp(log_likelihood)
        weighted = self._weights * likelihood
        total = float(np.sum(weighted))
        if not np.isfinite(total) or total <= 0.0:
            self._weights.fill(1.0 / self.config.n_particles)
            return {
                "weight_reset": 1.0,
                "effective_sample_size": float(self.config.n_particles),
            }
        self._weights = weighted / total
        ess = float(1.0 / np.sum(self._weights**2))
        return {
            "weight_reset": 0.0,
            "effective_sample_size": ess,
        }

    def predict(self, histogram: np.ndarray, *, window_id: int | None = None) -> NoisePrediction:
        measurement = self.measurement_baseline.predict(histogram, window_id=window_id)
        measurement_vector = np.array(
            [measurement.sigma, measurement.mu_q, measurement.mu_p, measurement.theta_deg],
            dtype=float,
        )
        self._predict_particles()
        update_meta = self._update_weights(measurement_vector)
        ess_threshold = self.config.resample_ess_ratio * self.config.n_particles
        resampled = update_meta["effective_sample_size"] < ess_threshold
        if resampled:
            self._systematic_resample()
        estimate = self._estimate_state()
        metadata = {
            "window_id": window_id,
            "measurement_prediction": measurement.to_dict(),
            "particle_filter": {
                "n_particles": self.config.n_particles,
                "effective_sample_size": float(update_meta["effective_sample_size"]),
                "ess_threshold": float(ess_threshold),
                "resampled": bool(resampled),
                "weight_reset": bool(update_meta["weight_reset"] > 0.5),
            },
            "state_after_update": {
                "sigma": float(estimate[0]),
                "mu_q": float(estimate[1]),
                "mu_p": float(estimate[2]),
                "theta_deg": float(estimate[3]),
            },
        }
        return NoisePrediction(
            sigma=float(estimate[0]),
            mu_q=float(estimate[1]),
            mu_p=float(estimate[2]),
            theta_deg=float(estimate[3]),
            source="particle_filter_baseline",
            metadata=metadata,
        )


@dataclass(frozen=True)
class ParticleFilterResidualBBaselineConfig:
    """Configuration of teacher-guided residual-b particle filter."""

    n_particles: int
    residual_clip_b: float
    initial_std_b: float
    process_std_b: float
    measurement_std_b: float
    resample_ess_ratio: float
    rejuvenation_std_b: float

    def __post_init__(self) -> None:
        if self.n_particles < 32:
            raise ValueError("n_particles must be >= 32")
        for name, value in (
            ("residual_clip_b", self.residual_clip_b),
            ("initial_std_b", self.initial_std_b),
            ("process_std_b", self.process_std_b),
            ("measurement_std_b", self.measurement_std_b),
            ("rejuvenation_std_b", self.rejuvenation_std_b),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 < self.resample_ess_ratio <= 1.0):
            raise ValueError("resample_ess_ratio must be within (0, 1]")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ParticleFilterResidualBBaselineConfig":
        slow_cfg = config.get("slow_loop", {})
        pf_cfg = slow_cfg.get("particle_filter_residual_b", {})
        hybrid_cfg = slow_cfg.get("hybrid_residual_b", {})
        return cls(
            n_particles=int(pf_cfg.get("n_particles", 256)),
            residual_clip_b=float(pf_cfg.get("residual_clip_b", hybrid_cfg.get("residual_clip_b", 0.12))),
            initial_std_b=float(pf_cfg.get("initial_std_b", 0.03)),
            process_std_b=float(pf_cfg.get("process_std_b", 0.01)),
            measurement_std_b=float(pf_cfg.get("measurement_std_b", 0.04)),
            resample_ess_ratio=float(pf_cfg.get("resample_ess_ratio", 0.5)),
            rejuvenation_std_b=float(pf_cfg.get("rejuvenation_std_b", 0.004)),
        )


@dataclass(frozen=True)
class ParticleFilterResidualBResult:
    """One causal prediction/update step of the residual-b particle filter."""

    delta_b_pred: np.ndarray
    delta_b_obs: np.ndarray
    metadata: Dict[str, Any]


class ParticleFilterResidualBBaseline:
    """Teacher-guided residual-b particle filter.

    中文说明：
    - 该版本不再直接估计绝对噪声参数，而是围绕 teacher baseline 的 `b`
      去追踪更稳定、更贴近运行时语义的 `delta_b`。
    - 这样它和 `Hybrid Residual-B / RLS Residual-B` 处于更公平的同口径比较。
    """

    def __init__(self, config: ParticleFilterResidualBBaselineConfig, *, seed: int | None = None) -> None:
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._particles = np.zeros((self.config.n_particles, 2), dtype=float)
        self._weights = np.full(self.config.n_particles, 1.0 / self.config.n_particles, dtype=float)
        self._initialize_particles()

    @classmethod
    def from_config(cls, config: Dict[str, Any], seed: int | None = None) -> "ParticleFilterResidualBBaseline":
        return cls(ParticleFilterResidualBBaselineConfig.from_config(config), seed=seed)

    def _initialize_particles(self) -> None:
        self._particles[:, 0] = self._rng.normal(0.0, self.config.initial_std_b, size=self.config.n_particles)
        self._particles[:, 1] = self._rng.normal(0.0, self.config.initial_std_b, size=self.config.n_particles)
        self._particles = np.clip(self._particles, -self.config.residual_clip_b, self.config.residual_clip_b)

    def _predict_particles(self) -> None:
        self._particles[:, 0] += self._rng.normal(0.0, self.config.process_std_b, size=self.config.n_particles)
        self._particles[:, 1] += self._rng.normal(0.0, self.config.process_std_b, size=self.config.n_particles)
        self._particles = np.clip(self._particles, -self.config.residual_clip_b, self.config.residual_clip_b)

    def _systematic_resample(self) -> None:
        cumulative = np.cumsum(self._weights)
        positions = (self._rng.random() + np.arange(self.config.n_particles, dtype=float)) / self.config.n_particles
        indices = np.searchsorted(cumulative, positions, side="left")
        self._particles = self._particles[indices].copy()
        self._particles[:, 0] += self._rng.normal(0.0, self.config.rejuvenation_std_b, size=self.config.n_particles)
        self._particles[:, 1] += self._rng.normal(0.0, self.config.rejuvenation_std_b, size=self.config.n_particles)
        self._particles = np.clip(self._particles, -self.config.residual_clip_b, self.config.residual_clip_b)
        self._weights.fill(1.0 / self.config.n_particles)

    def _update_weights(self, delta_b_obs: np.ndarray) -> Dict[str, float]:
        residual = (delta_b_obs.reshape(1, 2) - self._particles) / self.config.measurement_std_b
        log_likelihood = -0.5 * np.sum(residual**2, axis=1)
        log_likelihood -= float(np.max(log_likelihood))
        likelihood = np.exp(log_likelihood)
        weighted = self._weights * likelihood
        total = float(np.sum(weighted))
        if not np.isfinite(total) or total <= 0.0:
            self._weights.fill(1.0 / self.config.n_particles)
            return {
                "weight_reset": 1.0,
                "effective_sample_size": float(self.config.n_particles),
            }
        self._weights = weighted / total
        ess = float(1.0 / np.sum(self._weights**2))
        return {
            "weight_reset": 0.0,
            "effective_sample_size": ess,
        }

    def predict(
        self,
        *,
        teacher_b: np.ndarray,
        measurement_b: np.ndarray,
        window_id: int | None = None,
    ) -> ParticleFilterResidualBResult:
        teacher_b = np.asarray(teacher_b, dtype=float).reshape(2)
        measurement_b = np.asarray(measurement_b, dtype=float).reshape(2)
        delta_b_obs = np.clip(measurement_b - teacher_b, -self.config.residual_clip_b, self.config.residual_clip_b)
        self._predict_particles()
        update_meta = self._update_weights(delta_b_obs)
        ess_threshold = self.config.resample_ess_ratio * self.config.n_particles
        resampled = update_meta["effective_sample_size"] < ess_threshold
        if resampled:
            self._systematic_resample()
        delta_b_pred = np.sum(self._weights[:, None] * self._particles, axis=0)
        delta_b_pred = np.clip(delta_b_pred, -self.config.residual_clip_b, self.config.residual_clip_b)
        metadata = {
            "window_id": window_id,
            "teacher_b": teacher_b.astype(float).tolist(),
            "measurement_b": measurement_b.astype(float).tolist(),
            "delta_b_obs": delta_b_obs.astype(float).tolist(),
            "delta_b_pred": delta_b_pred.astype(float).tolist(),
            "particle_filter": {
                "n_particles": self.config.n_particles,
                "effective_sample_size": float(update_meta["effective_sample_size"]),
                "ess_threshold": float(ess_threshold),
                "resampled": bool(resampled),
                "weight_reset": bool(update_meta["weight_reset"] > 0.5),
                "residual_clip_b": self.config.residual_clip_b,
            },
        }
        return ParticleFilterResidualBResult(
            delta_b_pred=np.asarray(delta_b_pred, dtype=float),
            delta_b_obs=np.asarray(delta_b_obs, dtype=float),
            metadata=metadata,
        )
