"""Decoder-side helpers for runtime parameter generation."""

# 中文说明：
# - 该子模块负责把慢回路估计的噪声参数映射为快回路使用的线性解码参数。
# - 当前首版先实现 (sigma, mu_q, mu_p, theta_deg) -> (K, b) 的确定性映射。

from .linear_runtime import FixedPointFormat, LinearRuntime, LinearRuntimeConfig, LinearRuntimeResult
from .ekf_baseline import EKFBaseline, EKFBaselineConfig
from .param_mapper import NoisePrediction, ParamMapper, ParamMapperConfig
from .particle_filter_baseline import (
    ParticleFilterBaseline,
    ParticleFilterBaselineConfig,
    ParticleFilterResidualBBaseline,
    ParticleFilterResidualBBaselineConfig,
    ParticleFilterResidualBResult,
)
from .rls_residual_baseline import RLSResidualBBaseline, RLSResidualBBaselineConfig, RLSResidualBResult
from .ukf_baseline import UKFBaseline, UKFBaselineConfig
from .window_baseline import (
    HistogramMomentEstimator,
    HistogramMomentObservation,
    WindowVarianceBaseline,
    WindowVarianceConfig,
)

__all__ = [
    "EKFBaseline",
    "EKFBaselineConfig",
    "FixedPointFormat",
    "HistogramMomentEstimator",
    "HistogramMomentObservation",
    "LinearRuntime",
    "LinearRuntimeConfig",
    "LinearRuntimeResult",
    "NoisePrediction",
    "ParamMapper",
    "ParamMapperConfig",
    "ParticleFilterBaseline",
    "ParticleFilterBaselineConfig",
    "ParticleFilterResidualBBaseline",
    "ParticleFilterResidualBBaselineConfig",
    "ParticleFilterResidualBResult",
    "RLSResidualBBaseline",
    "RLSResidualBBaselineConfig",
    "RLSResidualBResult",
    "UKFBaseline",
    "UKFBaselineConfig",
    "WindowVarianceBaseline",
    "WindowVarianceConfig",
]
