"""Dual-loop scheduler scaffold for CNN-FPGA runtime emulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

from .latency_injector import LatencyInjector, LatencySample
from .param_bank import DecoderRuntimeParams, ParamBank


SlowPathFn = Callable[["WindowFrame", DecoderRuntimeParams], DecoderRuntimeParams]


@dataclass(frozen=True)
class SchedulerConfig:
    """Timing and buffering rules for the dual-loop scheduler."""

    t_fast_us: float = 5.0
    window_size: int = 2048
    slow_update_period_us: float = 20_000.0
    window_stride: Optional[int] = None
    max_pending_windows: int = 2
    commit_delay_cycles: int = 1
    fast_path_budget_us: float = 1.5
    slow_path_budget_us: float = 5_000.0
    guard_cycles_after_commit: int = 0

    def __post_init__(self) -> None:
        if self.t_fast_us <= 0:
            raise ValueError("t_fast_us must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.slow_update_period_us <= 0:
            raise ValueError("slow_update_period_us must be positive")
        if self.max_pending_windows <= 0:
            raise ValueError("max_pending_windows must be positive")
        if self.commit_delay_cycles <= 0:
            raise ValueError("commit_delay_cycles must be positive")
        if self.window_stride is not None and self.window_stride <= 0:
            raise ValueError("window_stride must be positive when provided")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SchedulerConfig":
        hardware = config.get("hardware_defaults", {})
        timing = config.get("timing", {})
        runtime_cfg = config.get("runtime", {})

        t_fast_us = float(runtime_cfg.get("t_fast_us", hardware.get("t_fast_us", 5.0)))
        window_size = int(runtime_cfg.get("window_size", hardware.get("window_size", 2048)))
        t_slow_update_ms = float(
            runtime_cfg.get("t_slow_update_ms", hardware.get("t_slow_update_ms", 20.0))
        )
        raw_stride = runtime_cfg.get("window_stride", None)
        return cls(
            t_fast_us=t_fast_us,
            window_size=window_size,
            slow_update_period_us=t_slow_update_ms * 1000.0,
            window_stride=None if raw_stride is None else int(raw_stride),
            max_pending_windows=int(runtime_cfg.get("max_pending_windows", 2)),
            commit_delay_cycles=int(runtime_cfg.get("commit_delay_cycles", 1)),
            fast_path_budget_us=float(
                runtime_cfg.get("fast_cycle_budget_us", timing.get("fast_cycle_budget_us", 1.5))
            ),
            slow_path_budget_us=float(
                runtime_cfg.get("slow_update_budget_us", timing.get("slow_update_budget_us", 5000.0))
            ),
            guard_cycles_after_commit=int(runtime_cfg.get("guard_cycles_after_commit", 0)),
        )

    @property
    def window_duration_us(self) -> float:
        return self.window_size * self.t_fast_us

    @property
    def resolved_window_stride(self) -> int:
        if self.window_stride is not None:
            return self.window_stride
        if self.slow_update_period_us >= self.window_duration_us:
            return self.window_size
        return max(1, self.window_size // 4)


@dataclass
class WindowFrame:
    """One histogram window produced by the fast loop."""

    window_id: int
    start_epoch: int
    end_epoch: int
    ready_time_us: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
            "ready_time_us": self.ready_time_us,
            "payload": dict(self.payload),
        }


@dataclass
class SlowUpdateJob:
    """A slow-loop update currently in flight."""

    job_id: int
    window: WindowFrame
    started_epoch: int
    started_time_us: float
    ready_time_us: float
    latency: LatencySample
    proposed_params: DecoderRuntimeParams

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "window": self.window.to_dict(),
            "started_epoch": self.started_epoch,
            "started_time_us": self.started_time_us,
            "ready_time_us": self.ready_time_us,
            "latency": self.latency.to_dict(),
            "proposed_params": self.proposed_params.to_dict(),
        }


@dataclass
class SchedulerEvent:
    """Structured event emitted by the dual-loop scheduler."""

    kind: str
    epoch_id: int
    time_us: float
    details: Dict[str, Any] = field(default_factory=dict)


def _default_slow_path(window: WindowFrame, active_params: DecoderRuntimeParams) -> DecoderRuntimeParams:
    metadata = dict(active_params.metadata)
    metadata.update(
        {
            "runtime_mode": "passthrough",
            "source_window_id": window.window_id,
            "source_window_end_epoch": window.end_epoch,
        }
    )
    return DecoderRuntimeParams(K=active_params.K.copy(), b=active_params.b.copy(), metadata=metadata)


class DualLoopScheduler:
    """Cycle-based dual-loop scheduler with staged parameter commits."""

    def __init__(
        self,
        config: SchedulerConfig,
        *,
        param_bank: Optional[ParamBank] = None,
        latency_injector: Optional[LatencyInjector] = None,
        slow_path_fn: Optional[SlowPathFn] = None,
    ) -> None:
        self.config = config
        self.param_bank = param_bank or ParamBank()
        self.latency_injector = latency_injector or LatencyInjector()
        self.slow_path_fn = slow_path_fn or _default_slow_path

        self.epoch_id = self.param_bank.epoch_id
        self.time_us = self.epoch_id * self.config.t_fast_us
        self.window_stride = self.config.resolved_window_stride

        self._window_queue: Deque[WindowFrame] = deque()
        self._next_window_emit_epoch = self.config.window_size
        self._window_counter = 0
        self._job_counter = 0
        self._slow_job: Optional[SlowUpdateJob] = None
        self._next_slow_start_time_us = 0.0
        self._guard_until_epoch = 0

        self.fast_cycle_budget_violations = 0
        self.slow_update_budget_violations = 0
        self.dropped_windows = 0
        self.event_log: List[SchedulerEvent] = []

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        *,
        param_bank: Optional[ParamBank] = None,
        latency_injector: Optional[LatencyInjector] = None,
        slow_path_fn: Optional[SlowPathFn] = None,
    ) -> "DualLoopScheduler":
        experiment = config.get("experiment", {})
        seed = int(experiment.get("seed", 1234))
        return cls(
            SchedulerConfig.from_config(config),
            param_bank=param_bank,
            latency_injector=latency_injector or LatencyInjector.from_config(config, seed=seed),
            slow_path_fn=slow_path_fn,
        )

    @property
    def slow_job(self) -> Optional[SlowUpdateJob]:
        return self._slow_job

    @property
    def pending_windows(self) -> int:
        return len(self._window_queue)

    def _record(self, event: SchedulerEvent, events: List[SchedulerEvent]) -> None:
        self.event_log.append(event)
        events.append(event)

    def _emit_window(self, window_payload: Optional[Dict[str, Any]], events: List[SchedulerEvent]) -> None:
        self._window_counter += 1
        frame = WindowFrame(
            window_id=self._window_counter,
            start_epoch=self.epoch_id - self.config.window_size + 1,
            end_epoch=self.epoch_id,
            ready_time_us=self.time_us,
            payload=dict(window_payload or {}),
        )

        if len(self._window_queue) >= self.config.max_pending_windows:
            dropped = self._window_queue.popleft()
            self.dropped_windows += 1
            self._record(
                SchedulerEvent(
                    kind="window_dropped",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={"dropped_window_id": dropped.window_id},
                ),
                events,
            )

        self._window_queue.append(frame)
        self._record(
            SchedulerEvent(
                kind="window_ready",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details=frame.to_dict(),
            ),
            events,
        )
        self._next_window_emit_epoch += self.window_stride

    def _maybe_finish_slow_job(self, events: List[SchedulerEvent]) -> None:
        if self._slow_job is None:
            return
        if self._slow_job.ready_time_us > self.time_us:
            return

        finished_job = self._slow_job
        self._slow_job = None
        self._record(
            SchedulerEvent(
                kind="slow_update_finished",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details=finished_job.to_dict(),
            ),
            events,
        )

        pending = self.param_bank.stage_update(
            finished_job.proposed_params,
            commit_epoch=self.epoch_id + self.config.commit_delay_cycles,
            staged_epoch=self.epoch_id,
            metadata={
                "job_id": finished_job.job_id,
                "window_id": finished_job.window.window_id,
            },
        )
        self._record(
            SchedulerEvent(
                kind="params_staged",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details={
                    "target_bank": pending.target_bank,
                    "commit_epoch": pending.commit_epoch,
                    "version": pending.version,
                    "window_id": finished_job.window.window_id,
                },
            ),
            events,
        )

    def _maybe_start_slow_job(self, events: List[SchedulerEvent]) -> None:
        if self._slow_job is not None:
            return
        if self.param_bank.has_pending_commit:
            return
        if not self._window_queue:
            return
        if self.time_us < self._next_slow_start_time_us:
            return
        if self.epoch_id < self._guard_until_epoch:
            return

        window = self._window_queue.popleft()
        active_params = self.param_bank.read_active()
        try:
            proposed_params = self.slow_path_fn(window, active_params)
        except Exception as exc:  # pragma: no cover - defensive path
            self._next_slow_start_time_us = self.time_us + self.config.slow_update_period_us
            self._record(
                SchedulerEvent(
                    kind="slow_update_failed",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={"window_id": window.window_id, "error": str(exc)},
                ),
                events,
            )
            return

        latency = self.latency_injector.sample_slow_update()
        self._job_counter += 1
        self._slow_job = SlowUpdateJob(
            job_id=self._job_counter,
            window=window,
            started_epoch=self.epoch_id,
            started_time_us=self.time_us,
            ready_time_us=self.time_us + latency.total_us,
            latency=latency,
            proposed_params=proposed_params,
        )
        self._next_slow_start_time_us = self.time_us + self.config.slow_update_period_us
        if latency.total_us > self.config.slow_path_budget_us:
            self.slow_update_budget_violations += 1
            self._record(
                SchedulerEvent(
                    kind="slow_budget_violation",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "job_id": self._slow_job.job_id,
                        "latency_us": latency.total_us,
                        "budget_us": self.config.slow_path_budget_us,
                    },
                ),
                events,
            )

        self._record(
            SchedulerEvent(
                kind="slow_update_started",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details=self._slow_job.to_dict(),
            ),
            events,
        )

    def tick(self, *, window_payload: Optional[Dict[str, Any]] = None) -> List[SchedulerEvent]:
        """Advance one fast-path cycle and emit scheduler events."""

        events: List[SchedulerEvent] = []
        self.epoch_id += 1
        self.time_us = self.epoch_id * self.config.t_fast_us

        fast_cycle_latency_us = self.latency_injector.sample_fast_cycle()
        if fast_cycle_latency_us > self.config.fast_path_budget_us:
            self.fast_cycle_budget_violations += 1
            self._record(
                SchedulerEvent(
                    kind="fast_budget_violation",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "latency_us": fast_cycle_latency_us,
                        "budget_us": self.config.fast_path_budget_us,
                    },
                ),
                events,
            )

        commit_result = self.param_bank.commit_if_ready(self.epoch_id)
        if commit_result is not None:
            self._guard_until_epoch = self.epoch_id + self.config.guard_cycles_after_commit
            self._record(
                SchedulerEvent(
                    kind="commit_applied",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "activated_bank": commit_result.activated_bank,
                        "version": commit_result.version,
                        "guard_until_epoch": self._guard_until_epoch,
                    },
                ),
                events,
            )

        self._maybe_finish_slow_job(events)

        if self.epoch_id >= self._next_window_emit_epoch:
            self._emit_window(window_payload, events)

        self._maybe_start_slow_job(events)
        return events

    def run(
        self,
        n_cycles: int,
        *,
        window_payload_factory: Optional[Callable[[int, int], Dict[str, Any]]] = None,
    ) -> List[SchedulerEvent]:
        if n_cycles <= 0:
            raise ValueError("n_cycles must be positive")

        collected: List[SchedulerEvent] = []
        for _ in range(n_cycles):
            payload = None
            will_emit_window = self.epoch_id + 1 >= self._next_window_emit_epoch
            if will_emit_window and window_payload_factory is not None:
                payload = window_payload_factory(self._window_counter + 1, self.epoch_id + 1)
            collected.extend(self.tick(window_payload=payload))
        return collected

    def snapshot(self) -> Dict[str, Any]:
        return {
            "epoch_id": self.epoch_id,
            "time_us": self.time_us,
            "window_stride": self.window_stride,
            "pending_windows": len(self._window_queue),
            "slow_job_inflight": None if self._slow_job is None else self._slow_job.to_dict(),
            "next_window_emit_epoch": self._next_window_emit_epoch,
            "next_slow_start_time_us": self._next_slow_start_time_us,
            "fast_cycle_budget_violations": self.fast_cycle_budget_violations,
            "slow_update_budget_violations": self.slow_update_budget_violations,
            "dropped_windows": self.dropped_windows,
            "param_bank": self.param_bank.snapshot(),
        }
