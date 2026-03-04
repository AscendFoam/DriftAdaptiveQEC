"""
Physics Layer for Drift-Adaptive QEC

This module provides physically accurate simulations of:
- Approximate GKP state preparation
- Quantum noise channels (photon loss, thermal noise, etc.)
- Syndrome measurement with finite squeezing effects
- Error correction protocols
- Logical error tracking
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
