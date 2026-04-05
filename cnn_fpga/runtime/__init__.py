"""Runtime scaffolding for dual-loop CNN-FPGA scheduling."""

# 中文说明：
# - 该子模块用于承载“快回路持续运行 + 慢回路窗口更新”的调度骨架。
# - 当前首版聚焦双 bank 参数切换、时延注入和双回路调度，不直接依赖真实板卡。

from .latency_injector import LatencyInjector, LatencySample, StageLatencySpec
from .fast_loop_emulator import FastLoopConfig, FastLoopEmulator, NoiseProvider
from .inference_service import (
    ArtifactHistogramPredictor,
    BaseHistogramPredictor,
    InferenceService,
    InferenceServiceConfig,
    InferenceServiceError,
    InProcInferenceService,
    SubprocessInferenceService,
    TFLiteHistogramPredictor,
    build_inference_service,
    resolve_model_path,
)
from .param_bank import CommitResult, DecoderRuntimeParams, ParamBank, PendingCommit
from .scheduler import DualLoopScheduler, SchedulerConfig, SchedulerEvent, SlowUpdateJob, WindowFrame
from .slow_loop_runtime import SlowLoopRuntime, SlowLoopRuntimeConfig, SlowLoopRuntimeError

__all__ = [
    "ArtifactHistogramPredictor",
    "BaseHistogramPredictor",
    "CommitResult",
    "DecoderRuntimeParams",
    "DualLoopScheduler",
    "FastLoopConfig",
    "FastLoopEmulator",
    "InferenceService",
    "InferenceServiceConfig",
    "InferenceServiceError",
    "InProcInferenceService",
    "LatencyInjector",
    "LatencySample",
    "NoiseProvider",
    "ParamBank",
    "PendingCommit",
    "SchedulerConfig",
    "SchedulerEvent",
    "SlowUpdateJob",
    "SlowLoopRuntime",
    "SlowLoopRuntimeConfig",
    "SlowLoopRuntimeError",
    "StageLatencySpec",
    "SubprocessInferenceService",
    "TFLiteHistogramPredictor",
    "WindowFrame",
    "build_inference_service",
    "resolve_model_path",
]
