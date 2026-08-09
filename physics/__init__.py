"""Small, lazy convenience API for the physics package.

New code should import from the concrete physics module.  The package root keeps
only the core state/noise/measurement/correction/tracking names documented by
the original public entry point.
"""

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
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
