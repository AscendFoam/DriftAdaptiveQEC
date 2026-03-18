"""Runtime scaffolding for dual-loop CNN-FPGA scheduling."""

# 中文说明：
# - 该子模块用于承载“快回路持续运行 + 慢回路窗口更新”的调度骨架。
# - 当前首版聚焦双 bank 参数切换、时延注入和双回路调度，不直接依赖真实板卡。

from .latency_injector import LatencyInjector, LatencySample, StageLatencySpec
from .param_bank import CommitResult, DecoderRuntimeParams, ParamBank, PendingCommit
from .scheduler import DualLoopScheduler, SchedulerConfig, SchedulerEvent, SlowUpdateJob, WindowFrame

__all__ = [
    "CommitResult",
    "DecoderRuntimeParams",
    "DualLoopScheduler",
    "LatencyInjector",
    "LatencySample",
    "ParamBank",
    "PendingCommit",
    "SchedulerConfig",
    "SchedulerEvent",
    "SlowUpdateJob",
    "StageLatencySpec",
    "WindowFrame",
]
