"""Map slow-loop noise predictions to runtime linear decoder parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from physics.gkp_state import LATTICE_CONST

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams


def analyze_decoder_aggressiveness(
    K: np.ndarray,
    b: np.ndarray,
    *,
    gain_upper_bound: float,
    correction_limit: float,
    aggressive_gain_ratio: float,
    aggressive_bias_ratio: float,
) -> Dict[str, Any]:
    """Heuristically assess whether runtime decoder parameters are overly aggressive.

    中文说明：
    - 这里的“预测参数过激进”不是严格物理定义，而是面向工程调参与诊断的启发式指标。
    - 目标不是替代 `LER`，而是帮助区分：
      1. 是直方图输入本身超范围；
      2. 还是 K,b 导致校正量太大；
      3. 还是二者同时存在。
    """

    K = np.asarray(K, dtype=float).reshape(2, 2)
    b = np.asarray(b, dtype=float).reshape(2)
    gain_upper_bound = max(1.0e-8, float(gain_upper_bound))
    correction_limit = max(1.0e-8, float(correction_limit))
    aggressive_gain_ratio = max(0.0, float(aggressive_gain_ratio))
    aggressive_bias_ratio = max(0.0, float(aggressive_bias_ratio))

    sym_k = 0.5 * (K + K.T)
    eigvals = np.linalg.eigvalsh(sym_k)
    max_gain = float(np.max(np.abs(eigvals)))
    bias_norm = float(np.linalg.norm(b))

    gain_threshold = aggressive_gain_ratio * gain_upper_bound
    bias_threshold = aggressive_bias_ratio * correction_limit
    gain_flag = bool(max_gain >= gain_threshold)
    bias_flag = bool(bias_norm >= bias_threshold)

    return {
        "max_gain": max_gain,
        "bias_norm": bias_norm,
        "gain_threshold": gain_threshold,
        "bias_threshold": bias_threshold,
        "gain_flag": gain_flag,
        "bias_flag": bias_flag,
        "aggressive": bool(gain_flag or bias_flag),
    }


@dataclass(frozen=True)
class NoisePrediction:
    """Predicted noise parameters for one slow-loop update."""

    sigma: float
    mu_q: float
    mu_p: float
    theta_deg: float
    source: str = "unknown"
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma": float(self.sigma),
            "mu_q": float(self.mu_q),
            "mu_p": float(self.mu_p),
            "theta_deg": float(self.theta_deg),
            "source": self.source,
            "metadata": {} if self.metadata is None else dict(self.metadata),
        }


@dataclass(frozen=True)
class ParamMapperConfig:
    """Configuration for deterministic (sigma, mu, theta) -> (K, b) mapping."""

    alpha_bias: float = 1.0
    beta_smoothing: float = 0.2
    gain_clip: Tuple[float, float] = (0.2, 1.2)
    gain_scale: float = 1.0
    theta_clip_deg: Tuple[float, float] = (-20.0, 20.0)
    sigma_meas: float = 0.0
    delta_eff: float = 0.3
    sigma_ratio_p: float = 1.0
    correction_limit: float = LATTICE_CONST
    aggressive_gain_ratio: float = 0.9
    aggressive_bias_ratio: float = 0.25
    var_signal: float = (LATTICE_CONST / 2.0) ** 2 / 3.0

    def __post_init__(self) -> None:
        if self.alpha_bias < 0:
            raise ValueError("alpha_bias must be non-negative")
        if not (0.0 <= self.beta_smoothing <= 1.0):
            raise ValueError("beta_smoothing must be in [0, 1]")
        if self.gain_clip[0] > self.gain_clip[1]:
            raise ValueError("gain_clip lower bound must be <= upper bound")
        if self.gain_scale <= 0:
            raise ValueError("gain_scale must be positive")
        if self.theta_clip_deg[0] > self.theta_clip_deg[1]:
            raise ValueError("theta_clip_deg lower bound must be <= upper bound")
        if self.sigma_meas < 0 or self.delta_eff < 0 or self.var_signal <= 0 or self.sigma_ratio_p <= 0:
            raise ValueError("sigma_meas, delta_eff, sigma_ratio_p and var_signal must be valid positive values")
        if self.correction_limit <= 0:
            raise ValueError("correction_limit must be positive")
        if self.aggressive_gain_ratio < 0 or self.aggressive_bias_ratio < 0:
            raise ValueError("aggressive ratios must be non-negative")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ParamMapperConfig":
        mapping_cfg = config.get("param_mapping", {})
        measurement_cfg = config.get("measurement", {})
        model_cfg = config.get("model", {})
        fast_cfg = config.get("fast_loop", {})

        delta = float(measurement_cfg.get("delta", 0.3))
        eta = float(measurement_cfg.get("measurement_efficiency", 0.95))
        if eta <= 0:
            raise ValueError("measurement_efficiency must be positive")
        var_inefficiency = (1.0 - eta) / (2.0 * eta)
        sigma_meas = float(np.sqrt(max(0.0, var_inefficiency)))

        gain_clip_raw = mapping_cfg.get("gain_clip", [0.2, 1.2])
        theta_clip_raw = mapping_cfg.get("theta_clip_deg", [-20.0, 20.0])
        return cls(
            alpha_bias=float(mapping_cfg.get("alpha_bias", 1.0)),
            beta_smoothing=float(mapping_cfg.get("beta_smoothing", 0.2)),
            gain_clip=(float(gain_clip_raw[0]), float(gain_clip_raw[1])),
            gain_scale=float(mapping_cfg.get("gain_scale", 1.0)),
            theta_clip_deg=(float(theta_clip_raw[0]), float(theta_clip_raw[1])),
            sigma_meas=sigma_meas,
            delta_eff=delta,
            sigma_ratio_p=float(mapping_cfg.get("sigma_ratio_p", model_cfg.get("sigma_ratio_p", 1.0))),
            correction_limit=float(fast_cfg.get("correction_limit", LATTICE_CONST)),
            aggressive_gain_ratio=float(mapping_cfg.get("aggressive_gain_ratio", 0.9)),
            aggressive_bias_ratio=float(mapping_cfg.get("aggressive_bias_ratio", 0.25)),
        )


class ParamMapper:
    """Deterministically maps predicted noise parameters to runtime decoder parameters."""

    def __init__(self, config: ParamMapperConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ParamMapper":
        return cls(ParamMapperConfig.from_config(config))

    def map_prediction(
        self,
        prediction: NoisePrediction,
        previous_params: DecoderRuntimeParams | None = None,
    ) -> DecoderRuntimeParams:
        sigma_hat = max(0.0, float(prediction.sigma))
        mu_q_hat = float(prediction.mu_q)
        mu_p_hat = float(prediction.mu_p)
        theta_deg = float(
            np.clip(
                prediction.theta_deg,
                self.config.theta_clip_deg[0],
                self.config.theta_clip_deg[1],
            )
        )
        theta_rad = np.deg2rad(theta_deg)
        rotation = np.array(
            [
                [np.cos(theta_rad), -np.sin(theta_rad)],
                [np.sin(theta_rad), np.cos(theta_rad)],
            ],
            dtype=float,
        )
        sigma_q = max(1.0e-8, sigma_hat)
        sigma_p = max(1.0e-8, sigma_hat * self.config.sigma_ratio_p)
        error_cov_principal = np.diag([sigma_q**2, sigma_p**2])
        error_cov = rotation @ error_cov_principal @ rotation.T

        noise_var = self.config.sigma_meas**2 + self.config.delta_eff**2
        measurement_cov = np.eye(2, dtype=float) * noise_var

        # 中文注释：
        # - 更合理的线性估计器应基于实验室坐标系下的协方差，而不是“gain * 旋转矩阵”。
        # - K = C (C + R)^-1 会自然形成对称增益矩阵；旋转只通过 C 进入，而不直接再对 syndrome 旋转一次。
        gain_matrix_raw = error_cov @ np.linalg.inv(error_cov + measurement_cov)
        eigvals_raw, eigvecs = np.linalg.eigh(gain_matrix_raw)
        eigvals_clipped = np.clip(eigvals_raw, self.config.gain_clip[0], self.config.gain_clip[1])
        # 中文注释：
        # - `gain_scale` 是面向工程调优的整体保守系数，用于在不破坏协方差方向结构的前提下统一收紧校正强度。
        eigvals_scaled = np.clip(eigvals_clipped * self.config.gain_scale, 0.0, self.config.gain_clip[1])
        k_target = eigvecs @ np.diag(eigvals_scaled) @ eigvecs.T

        mu_vec = np.array([mu_q_hat, mu_p_hat], dtype=float)
        # 中文注释：
        # - 若 syndrome 的先验均值为 mu，则后验线性估计的偏置应为 (I-K)mu，而不是 -mu。
        # - 旧实现的负号会在非零均值漂移时把校正方向推错。
        b_target = self.config.alpha_bias * (np.eye(2, dtype=float) - k_target) @ mu_vec

        if previous_params is None:
            k_next = k_target
            b_next = b_target
        else:
            beta = self.config.beta_smoothing
            k_next = (1.0 - beta) * previous_params.K + beta * k_target
            b_next = (1.0 - beta) * previous_params.b + beta * b_target

        aggressiveness = analyze_decoder_aggressiveness(
            k_next,
            b_next,
            gain_upper_bound=self.config.gain_clip[1],
            correction_limit=self.config.correction_limit,
            aggressive_gain_ratio=self.config.aggressive_gain_ratio,
            aggressive_bias_ratio=self.config.aggressive_bias_ratio,
        )

        metadata = {
            "prediction": prediction.to_dict(),
            "error_cov": error_cov.tolist(),
            "measurement_cov": measurement_cov.tolist(),
            "gain_eigvals_raw": eigvals_raw.tolist(),
            "gain_eigvals_clipped": eigvals_clipped.tolist(),
            "gain_eigvals_scaled": eigvals_scaled.tolist(),
            "gain_scale": self.config.gain_scale,
            "theta_deg_clipped": theta_deg,
            "sigma_ratio_p": self.config.sigma_ratio_p,
            "param_aggressiveness": aggressiveness,
        }
        return DecoderRuntimeParams(K=k_next, b=b_next, metadata=metadata)
