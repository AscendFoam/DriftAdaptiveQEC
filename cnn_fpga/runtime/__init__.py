"""Runtime scaffolding for dual-loop CNN-FPGA scheduling."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "LatencyInjector": "cnn_fpga.runtime.latency_injector",
    "LatencySample": "cnn_fpga.runtime.latency_injector",
    "StageLatencySpec": "cnn_fpga.runtime.latency_injector",
    "FastLoopConfig": "cnn_fpga.runtime.fast_loop_emulator",
    "FastLoopEmulator": "cnn_fpga.runtime.fast_loop_emulator",
    "NoiseProvider": "cnn_fpga.runtime.fast_loop_emulator",
    "ArtifactHistogramPredictor": "cnn_fpga.runtime.inference_service",
    "BaseHistogramPredictor": "cnn_fpga.runtime.inference_service",
    "InferenceService": "cnn_fpga.runtime.inference_service",
    "InferenceServiceConfig": "cnn_fpga.runtime.inference_service",
    "InferenceServiceError": "cnn_fpga.runtime.inference_service",
    "InProcInferenceService": "cnn_fpga.runtime.inference_service",
    "SubprocessInferenceService": "cnn_fpga.runtime.inference_service",
    "TFLiteHistogramPredictor": "cnn_fpga.runtime.inference_service",
    "build_inference_service": "cnn_fpga.runtime.inference_service",
    "resolve_model_path": "cnn_fpga.runtime.inference_service",
    "CommitResult": "cnn_fpga.runtime.param_bank",
    "DecoderRuntimeParams": "cnn_fpga.runtime.param_bank",
    "ParamBank": "cnn_fpga.runtime.param_bank",
    "PendingCommit": "cnn_fpga.runtime.param_bank",
    "DualLoopScheduler": "cnn_fpga.runtime.scheduler",
    "SchedulerConfig": "cnn_fpga.runtime.scheduler",
    "SchedulerEvent": "cnn_fpga.runtime.scheduler",
    "SlowUpdateJob": "cnn_fpga.runtime.scheduler",
    "WindowFrame": "cnn_fpga.runtime.scheduler",
    "SlowLoopRuntime": "cnn_fpga.runtime.slow_loop_runtime",
    "SlowLoopRuntimeConfig": "cnn_fpga.runtime.slow_loop_runtime",
    "SlowLoopRuntimeError": "cnn_fpga.runtime.slow_loop_runtime",
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
