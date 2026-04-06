"""Physics Layer for Drift-Adaptive QEC."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "ApproximateGKPState": "physics.gkp_state",
    "GKPStateFactory": "physics.gkp_state",
    "QuantumNoiseChannel": "physics.noise_channels",
    "PhotonLossChannel": "physics.noise_channels",
    "ThermalNoiseChannel": "physics.noise_channels",
    "SyndromeMeasurement": "physics.syndrome_measurement",
    "RealisticSyndromeMeasurement": "physics.syndrome_measurement",
    "GKPErrorCorrector": "physics.error_correction",
    "LinearDecoder": "physics.error_correction",
    "LogicalErrorTracker": "physics.logical_tracking",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
