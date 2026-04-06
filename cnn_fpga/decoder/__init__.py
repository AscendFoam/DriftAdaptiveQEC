"""Decoder-side helpers for runtime parameter generation."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "FixedPointFormat": "cnn_fpga.decoder.linear_runtime",
    "LinearRuntime": "cnn_fpga.decoder.linear_runtime",
    "LinearRuntimeConfig": "cnn_fpga.decoder.linear_runtime",
    "LinearRuntimeResult": "cnn_fpga.decoder.linear_runtime",
    "EKFBaseline": "cnn_fpga.decoder.ekf_baseline",
    "EKFBaselineConfig": "cnn_fpga.decoder.ekf_baseline",
    "NoisePrediction": "cnn_fpga.decoder.param_mapper",
    "ParamMapper": "cnn_fpga.decoder.param_mapper",
    "ParamMapperConfig": "cnn_fpga.decoder.param_mapper",
    "ParticleFilterBaseline": "cnn_fpga.decoder.particle_filter_baseline",
    "ParticleFilterBaselineConfig": "cnn_fpga.decoder.particle_filter_baseline",
    "ParticleFilterResidualBBaseline": "cnn_fpga.decoder.particle_filter_baseline",
    "ParticleFilterResidualBBaselineConfig": "cnn_fpga.decoder.particle_filter_baseline",
    "ParticleFilterResidualBResult": "cnn_fpga.decoder.particle_filter_baseline",
    "RLSResidualBBaseline": "cnn_fpga.decoder.rls_residual_baseline",
    "RLSResidualBBaselineConfig": "cnn_fpga.decoder.rls_residual_baseline",
    "RLSResidualBResult": "cnn_fpga.decoder.rls_residual_baseline",
    "UKFBaseline": "cnn_fpga.decoder.ukf_baseline",
    "UKFBaselineConfig": "cnn_fpga.decoder.ukf_baseline",
    "HistogramMomentEstimator": "cnn_fpga.decoder.window_baseline",
    "HistogramMomentObservation": "cnn_fpga.decoder.window_baseline",
    "WindowVarianceBaseline": "cnn_fpga.decoder.window_baseline",
    "WindowVarianceConfig": "cnn_fpga.decoder.window_baseline",
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
