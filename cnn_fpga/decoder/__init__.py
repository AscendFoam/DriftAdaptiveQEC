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
    "AxisMAPModel": "cnn_fpga.decoder.parametric_map_lut",
    "compile_active_param_bank": "cnn_fpga.decoder.parametric_map_lut",
    "compile_parametric_map_lut": "cnn_fpga.decoder.parametric_map_lut",
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
    "GaussianRegimeHMM": "cnn_fpga.decoder.regime_hmm",
    "RegimeEstimatorBudget": "cnn_fpga.decoder.regime_hmm",
    "RegimeObservationWindow": "cnn_fpga.decoder.regime_hmm",
    "fit_supervised_gaussian_hmm": "cnn_fpga.decoder.regime_hmm",
    "summarize_regime_window": "cnn_fpga.decoder.regime_hmm",
    "ContinuousNoiseCalibration": "cnn_fpga.decoder.hybrid_state_output",
    "RegimePosteriorOutput": "cnn_fpga.decoder.hybrid_state_output",
    "LeakageRecoveryOutput": "cnn_fpga.decoder.hybrid_state_output",
    "UncertaintyOutput": "cnn_fpga.decoder.hybrid_state_output",
    "ParameterBankRecommendation": "cnn_fpga.decoder.hybrid_state_output",
    "HybridStateOutput": "cnn_fpga.decoder.hybrid_state_output",
    "HybridStateEstimatorConfig": "cnn_fpga.decoder.hybrid_state_output",
    "HybridStateEstimator": "cnn_fpga.decoder.hybrid_state_output",
    "stage_parameter_bank_recommendation": "cnn_fpga.decoder.hybrid_state_output",
    "CalibrationRecord": "cnn_fpga.decoder.hybrid_multiobjective",
    "FrozenCalibration": "cnn_fpga.decoder.hybrid_multiobjective",
    "MultiObjectiveWeights": "cnn_fpga.decoder.hybrid_multiobjective",
    "TrainingNormalizers": "cnn_fpga.decoder.hybrid_multiobjective",
    "calibration_manifest": "cnn_fpga.decoder.hybrid_multiobjective",
    "evaluate_multiobjective_loss": "cnn_fpga.decoder.hybrid_multiobjective",
    "fit_training_normalizers": "cnn_fpga.decoder.hybrid_multiobjective",
    "fit_validation_calibration": "cnn_fpga.decoder.hybrid_multiobjective",
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
