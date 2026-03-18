"""
Physics Layer for Drift-Adaptive QEC

This module provides physically accurate simulations of:
- Approximate GKP state preparation
- Quantum noise channels (photon loss, thermal noise, etc.)
- Syndrome measurement with finite squeezing effects
- Error correction protocols
- Logical error tracking

中文说明：
- 该模块是物理仿真层统一入口，集中导出最常用类和函数。
- 目标是让上层实验脚本通过稳定API访问噪声、测量、纠错与逻辑错误统计能力。
"""

from .gkp_state import ApproximateGKPState, GKPStateFactory
from .noise_channels import QuantumNoiseChannel, PhotonLossChannel, ThermalNoiseChannel
from .syndrome_measurement import SyndromeMeasurement, RealisticSyndromeMeasurement
from .error_correction import GKPErrorCorrector, LinearDecoder
from .logical_tracking import LogicalErrorTracker

__all__ = [
    'ApproximateGKPState',
    'GKPStateFactory',
    'QuantumNoiseChannel',
    'PhotonLossChannel',
    'ThermalNoiseChannel',
    'SyndromeMeasurement',
    'RealisticSyndromeMeasurement',
    'GKPErrorCorrector',
    'LinearDecoder',
    'LogicalErrorTracker',
]
