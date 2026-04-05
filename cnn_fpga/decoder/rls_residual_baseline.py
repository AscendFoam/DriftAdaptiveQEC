"""Online RLS residual baseline on top of teacher + current-window observation."""

# 中文说明：
# - 该模块实现一个不依赖训练集的强经典 baseline：`RLS Residual-B`。
# - 它保持当前工程主线的语义不变：
#   1. teacher baseline 先给出稳定的 `(K, b)`
#   2. 当前窗口 `Window Variance` 给出更“瞬时”的观测
#   3. RLS 在线学习“观测 residual 应该如何修正 teacher 的 b”
# - 与 `Hybrid Residual-B` 相比，它不使用神经网络，而是使用递推最小二乘在线拟合
#   `delta_b = W @ phi`，因此可作为更强、也更可解释的经典对照组。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from cnn_fpga.decoder.param_mapper import NoisePrediction


@dataclass(frozen=True)
class RLSResidualBBaselineConfig:
    """Configuration of the online residual-b adaptive baseline."""

    teacher_mode: str
    forgetting_factor: float
    init_covariance: float
    covariance_floor: float
    residual_clip_b: float
    sigma_scale: float
    mu_scale: float
    theta_scale_deg: float
    b_scale: float

    def __post_init__(self) -> None:
        if self.teacher_mode not in {"window_variance", "ekf", "ukf", "particle_filter"}:
            raise ValueError(f"unsupported_teacher_mode:{self.teacher_mode}")
        if not (0.9 <= self.forgetting_factor <= 1.0):
            raise ValueError("forgetting_factor must be within [0.9, 1.0]")
        if self.init_covariance <= 0.0:
            raise ValueError("init_covariance must be positive")
        if self.covariance_floor <= 0.0:
            raise ValueError("covariance_floor must be positive")
        if self.residual_clip_b <= 0.0:
            raise ValueError("residual_clip_b must be positive")
        if self.sigma_scale <= 0.0 or self.mu_scale <= 0.0 or self.theta_scale_deg <= 0.0 or self.b_scale <= 0.0:
            raise ValueError("all feature scales must be positive")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RLSResidualBBaselineConfig":
        slow_cfg = config.get("slow_loop", {})
        mode_cfg = slow_cfg.get("rls_residual_b", {})
        fast_cfg = config.get("fast_loop", {})
        mapping_cfg = config.get("param_mapping", {})
        hist_limit = float(fast_cfg.get("histogram_range_limit", fast_cfg.get("syndrome_limit", 1.0)))
        theta_clip = mapping_cfg.get("theta_clip_deg", [-20.0, 20.0])
        correction_limit = float(fast_cfg.get("correction_limit", 1.0))
        return cls(
            teacher_mode=str(mode_cfg.get("teacher_mode", "ekf")).lower(),
            forgetting_factor=float(mode_cfg.get("forgetting_factor", 0.995)),
            init_covariance=float(mode_cfg.get("init_covariance", 25.0)),
            covariance_floor=float(mode_cfg.get("covariance_floor", 1.0e-6)),
            residual_clip_b=float(mode_cfg.get("residual_clip_b", slow_cfg.get("hybrid_residual_b", {}).get("residual_clip_b", 0.12))),
            sigma_scale=float(mode_cfg.get("sigma_scale", 0.5)),
            mu_scale=float(mode_cfg.get("mu_scale", max(1.0e-6, hist_limit))),
            theta_scale_deg=float(mode_cfg.get("theta_scale_deg", max(abs(float(theta_clip[0])), abs(float(theta_clip[1])), 1.0))),
            b_scale=float(mode_cfg.get("b_scale", max(1.0e-6, correction_limit))),
        )


@dataclass(frozen=True)
class RLSResidualBResult:
    """One causal prediction/update step of the RLS residual baseline."""

    delta_b_pred: np.ndarray
    delta_b_target: np.ndarray
    feature_vector: np.ndarray
    metadata: Dict[str, Any]


class RLSResidualBBaseline:
    """Online recursive least-squares residual corrector for decoder bias `b`."""

    def __init__(self, config: RLSResidualBBaselineConfig) -> None:
        self.config = config
        self._feature_dim = 17
        self._weights = np.zeros((2, self._feature_dim), dtype=float)
        self._covariance = np.stack(
            [np.eye(self._feature_dim, dtype=float) * self.config.init_covariance for _ in range(2)],
            axis=0,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RLSResidualBBaseline":
        return cls(RLSResidualBBaselineConfig.from_config(config))

    def _feature_vector(
        self,
        *,
        teacher_prediction: NoisePrediction,
        measurement_prediction: NoisePrediction,
        teacher_b: np.ndarray,
        measurement_b: np.ndarray,
    ) -> np.ndarray:
        teacher_b = np.asarray(teacher_b, dtype=float).reshape(2)
        measurement_b = np.asarray(measurement_b, dtype=float).reshape(2)
        delta_prediction = np.array(
            [
                measurement_prediction.sigma - teacher_prediction.sigma,
                measurement_prediction.mu_q - teacher_prediction.mu_q,
                measurement_prediction.mu_p - teacher_prediction.mu_p,
                measurement_prediction.theta_deg - teacher_prediction.theta_deg,
            ],
            dtype=float,
        )
        delta_b_obs = measurement_b - teacher_b
        return np.array(
            [
                1.0,
                teacher_prediction.sigma / self.config.sigma_scale,
                teacher_prediction.mu_q / self.config.mu_scale,
                teacher_prediction.mu_p / self.config.mu_scale,
                teacher_prediction.theta_deg / self.config.theta_scale_deg,
                measurement_prediction.sigma / self.config.sigma_scale,
                measurement_prediction.mu_q / self.config.mu_scale,
                measurement_prediction.mu_p / self.config.mu_scale,
                measurement_prediction.theta_deg / self.config.theta_scale_deg,
                delta_prediction[0] / self.config.sigma_scale,
                delta_prediction[1] / self.config.mu_scale,
                delta_prediction[2] / self.config.mu_scale,
                delta_prediction[3] / self.config.theta_scale_deg,
                teacher_b[0] / self.config.b_scale,
                teacher_b[1] / self.config.b_scale,
                delta_b_obs[0] / self.config.b_scale,
                delta_b_obs[1] / self.config.b_scale,
            ],
            dtype=float,
        )

    def _update_dim(self, dim: int, phi: np.ndarray, target: float) -> Dict[str, float]:
        weights = self._weights[dim]
        covariance = self._covariance[dim]
        lambda_ = self.config.forgetting_factor

        prediction = float(weights @ phi)
        innovation = float(target - prediction)
        denom = float(lambda_ + phi @ covariance @ phi)
        gain = (covariance @ phi) / max(denom, 1.0e-12)
        updated_weights = weights + gain * innovation
        updated_cov = (covariance - np.outer(gain, phi) @ covariance) / lambda_
        updated_cov = 0.5 * (updated_cov + updated_cov.T)
        diag = np.maximum(np.diag(updated_cov), self.config.covariance_floor)
        self._weights[dim] = updated_weights
        self._covariance[dim] = updated_cov
        np.fill_diagonal(self._covariance[dim], diag)
        return {
            "prediction_raw": prediction,
            "innovation": innovation,
            "gain_norm": float(np.linalg.norm(gain)),
            "weight_norm": float(np.linalg.norm(updated_weights)),
        }

    def predict(
        self,
        *,
        teacher_prediction: NoisePrediction,
        measurement_prediction: NoisePrediction,
        teacher_b: np.ndarray,
        measurement_b: np.ndarray,
        window_id: int | None = None,
    ) -> RLSResidualBResult:
        teacher_b = np.asarray(teacher_b, dtype=float).reshape(2)
        measurement_b = np.asarray(measurement_b, dtype=float).reshape(2)
        phi = self._feature_vector(
            teacher_prediction=teacher_prediction,
            measurement_prediction=measurement_prediction,
            teacher_b=teacher_b,
            measurement_b=measurement_b,
        )

        prediction_raw = self._weights @ phi
        delta_b_pred = np.clip(prediction_raw, -self.config.residual_clip_b, self.config.residual_clip_b)
        delta_b_target = np.clip(measurement_b - teacher_b, -self.config.residual_clip_b, self.config.residual_clip_b)

        update_meta = [
            self._update_dim(dim, phi, float(delta_b_target[dim]))
            for dim in range(2)
        ]
        metadata = {
            "window_id": window_id,
            "teacher_mode": self.config.teacher_mode,
            "residual_clip_b": self.config.residual_clip_b,
            "forgetting_factor": self.config.forgetting_factor,
            "delta_b_pred_raw": np.asarray(prediction_raw, dtype=float).tolist(),
            "delta_b_pred_clipped": np.asarray(delta_b_pred, dtype=float).tolist(),
            "delta_b_target": np.asarray(delta_b_target, dtype=float).tolist(),
            "feature_vector": np.asarray(phi, dtype=float).tolist(),
            "feature_dim": int(phi.size),
            "update": {
                "q": update_meta[0],
                "p": update_meta[1],
            },
        }
        return RLSResidualBResult(
            delta_b_pred=np.asarray(delta_b_pred, dtype=float),
            delta_b_target=np.asarray(delta_b_target, dtype=float),
            feature_vector=np.asarray(phi, dtype=float),
            metadata=metadata,
        )
