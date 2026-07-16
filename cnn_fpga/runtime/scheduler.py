"""Dual-loop scheduler scaffold for CNN-FPGA runtime emulation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence

import numpy as np

from .latency_injector import LatencyContext, LatencyInjector, LatencySample
from .param_bank import (
    DecoderRuntimeParams,
    ParamBank,
    ParameterUpdateConflictError,
    PendingCommit,
)


SlowPathFn = Callable[["WindowFrame", DecoderRuntimeParams], DecoderRuntimeParams]
FastPathFn = Callable[[int, float, bool], Optional[Dict[str, Any]]]


def _summarize_value(value: Any) -> Any:
    """Convert runtime payloads to JSON-friendly summaries for event logs."""
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return value.tolist()
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "mean": float(np.mean(value)),
            "std": float(np.std(value)),
        }
    if isinstance(value, dict):
        return {str(key): _summarize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    window_deadline_us: Optional[float] = None

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
        if self.window_deadline_us is not None and self.window_deadline_us <= 0:
            raise ValueError("window_deadline_us must be positive when provided")

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
            window_deadline_us=float(
                runtime_cfg.get("window_deadline_us", t_slow_update_ms * 1000.0)
            ),
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

    @property
    def resolved_window_deadline_us(self) -> float:
        return (
            self.slow_update_period_us
            if self.window_deadline_us is None
            else self.window_deadline_us
        )


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
            "payload": _summarize_value(self.payload),
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
    active_params: DecoderRuntimeParams
    proposed_params: Optional[DecoderRuntimeParams] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "window": self.window.to_dict(),
            "started_epoch": self.started_epoch,
            "started_time_us": self.started_time_us,
            "ready_time_us": self.ready_time_us,
            "latency": self.latency.to_dict(),
            "active_params": self.active_params.to_dict(),
            "proposed_params": None if self.proposed_params is None else self.proposed_params.to_dict(),
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
        self._communication_available = True
        self._communication_pause_started_epoch: Optional[int] = None
        self._communication_pause_started_time_us: Optional[float] = None

        self.fast_cycle_budget_violations = 0
        self.slow_update_budget_violations = 0
        self.window_deadline_misses = 0
        self.dropped_windows = 0
        self.fifo_overflows = 0
        self.input_bursts = 0
        self.communication_pauses = 0
        self.communication_paused_cycles = 0
        self.parameter_update_conflicts = 0
        self.last_fast_cycle_latency_us: Optional[float] = None
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

    def _enqueue_frame(
        self,
        frame: WindowFrame,
        events: List[SchedulerEvent],
        *,
        source: str,
    ) -> None:
        queue_depth_before = len(self._window_queue)
        dropped: Optional[WindowFrame] = None
        if queue_depth_before >= self.config.max_pending_windows:
            dropped = self._window_queue.popleft()
            self.dropped_windows += 1
            self.fifo_overflows += 1
            self._record(
                SchedulerEvent(
                    kind="fifo_overflow",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "capacity": self.config.max_pending_windows,
                        "queue_depth_before": queue_depth_before,
                        "dropped_window_id": dropped.window_id,
                        "accepted_window_id": frame.window_id,
                        "source": source,
                    },
                ),
                events,
            )
            self._record(
                SchedulerEvent(
                    kind="window_dropped",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "dropped_window_id": dropped.window_id,
                        "reason": "fifo_overflow_drop_oldest",
                        "source": source,
                    },
                ),
                events,
            )

        self._window_queue.append(frame)
        details = frame.to_dict()
        details.update(
            {
                "source": source,
                "queue_depth_after": len(self._window_queue),
                "fifo_overflowed": dropped is not None,
            }
        )
        self._record(
            SchedulerEvent(
                kind="window_ready",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details=details,
            ),
            events,
        )

    def _emit_window(self, window_payload: Optional[Dict[str, Any]], events: List[SchedulerEvent]) -> None:
        self._window_counter += 1
        frame = WindowFrame(
            window_id=self._window_counter,
            start_epoch=self.epoch_id - self.config.window_size + 1,
            end_epoch=self.epoch_id,
            ready_time_us=self.time_us,
            payload=dict(window_payload or {}),
        )
        self._enqueue_frame(frame, events, source="regular_cadence")
        self._next_window_emit_epoch += self.window_stride

    def inject_window_burst(
        self,
        payloads: Sequence[Dict[str, Any]],
    ) -> List[SchedulerEvent]:
        """Inject several externally arrived windows at the current cycle."""

        burst = [dict(payload) for payload in payloads]
        if len(burst) < 2:
            raise ValueError("input burst must contain at least two windows")
        events: List[SchedulerEvent] = []
        self.input_bursts += 1
        self._record(
            SchedulerEvent(
                kind="input_burst",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details={
                    "window_count": len(burst),
                    "queue_depth_before": len(self._window_queue),
                    "capacity": self.config.max_pending_windows,
                },
            ),
            events,
        )
        for payload in burst:
            self._window_counter += 1
            frame = WindowFrame(
                window_id=self._window_counter,
                start_epoch=max(1, self.epoch_id - self.config.window_size + 1),
                end_epoch=self.epoch_id,
                ready_time_us=self.time_us,
                payload=payload,
            )
            self._enqueue_frame(frame, events, source="injected_burst")
        return events

    def _update_communication_state(
        self,
        available: bool,
        events: List[SchedulerEvent],
    ) -> None:
        state = bool(available)
        if not state:
            self.communication_paused_cycles += 1
        if state == self._communication_available:
            return
        self._communication_available = state
        if not state:
            self.communication_pauses += 1
            self._communication_pause_started_epoch = self.epoch_id
            self._communication_pause_started_time_us = self.time_us
            self._record(
                SchedulerEvent(
                    kind="communication_pause_started",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={"pending_windows": len(self._window_queue)},
                ),
                events,
            )
            return

        start_epoch = self._communication_pause_started_epoch
        start_time = self._communication_pause_started_time_us
        self._record(
            SchedulerEvent(
                kind="communication_pause_ended",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details={
                    "start_epoch": start_epoch,
                    "duration_cycles": None if start_epoch is None else self.epoch_id - start_epoch,
                    "duration_us": None if start_time is None else self.time_us - start_time,
                    "pending_windows": len(self._window_queue),
                },
            ),
            events,
        )
        self._communication_pause_started_epoch = None
        self._communication_pause_started_time_us = None

    def _maybe_finish_slow_job(self, events: List[SchedulerEvent]) -> None:
        if self._slow_job is None:
            return
        if not self._communication_available:
            return
        if self._slow_job.ready_time_us > self.time_us:
            return

        finished_job = self._slow_job
        self._slow_job = None
        try:
            proposed_params = self.slow_path_fn(finished_job.window, finished_job.active_params)
        except Exception as exc:  # pragma: no cover - defensive path
            error_reason = getattr(exc, "reason", str(exc))
            self._record(
                SchedulerEvent(
                    kind="slow_update_failed",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "job_id": finished_job.job_id,
                        "window_id": finished_job.window.window_id,
                        "reason": error_reason,
                    },
                ),
                events,
            )
            return

        finished_job.proposed_params = proposed_params
        window_age_us = self.time_us - finished_job.window.ready_time_us
        deadline_us = self.config.resolved_window_deadline_us
        deadline_missed = window_age_us > deadline_us
        if deadline_missed:
            self.window_deadline_misses += 1
            self._record(
                SchedulerEvent(
                    kind="window_deadline_miss",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "job_id": finished_job.job_id,
                        "window_id": finished_job.window.window_id,
                        "window_age_us": window_age_us,
                        "deadline_us": deadline_us,
                        "queue_wait_us": finished_job.started_time_us
                        - finished_job.window.ready_time_us,
                        "service_latency_us": finished_job.latency.total_us,
                    },
                ),
                events,
            )
        finished_details = finished_job.to_dict()
        finished_details.update(
            {
                "window_age_us": window_age_us,
                "window_deadline_us": deadline_us,
                "window_deadline_missed": deadline_missed,
            }
        )
        self._record(
            SchedulerEvent(
                kind="slow_update_finished",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details=finished_details,
            ),
            events,
        )

        try:
            pending = self.param_bank.stage_update(
                finished_job.proposed_params,
                commit_epoch=self.epoch_id + self.config.commit_delay_cycles,
                staged_epoch=self.epoch_id,
                metadata={
                    "job_id": finished_job.job_id,
                    "window_id": finished_job.window.window_id,
                },
            )
        except ParameterUpdateConflictError as exc:
            self.parameter_update_conflicts += 1
            self._record(
                SchedulerEvent(
                    kind="parameter_update_conflict",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "job_id": finished_job.job_id,
                        "window_id": finished_job.window.window_id,
                        "reason": str(exc),
                        "active_version": self.param_bank.active_version,
                    },
                ),
                events,
            )
            return
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
        if not self._communication_available:
            return
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

        latency = self.latency_injector.sample_slow_update(
            context=LatencyContext(
                pending_windows=len(self._window_queue),
                slow_job_inflight=self._slow_job is not None,
                pending_commit=self.param_bank.has_pending_commit,
                recent_slow_budget_violations=self.slow_update_budget_violations,
            )
        )
        self._job_counter += 1
        self._slow_job = SlowUpdateJob(
            job_id=self._job_counter,
            window=window,
            started_epoch=self.epoch_id,
            started_time_us=self.time_us,
            ready_time_us=self.time_us + latency.total_us,
            latency=latency,
            active_params=active_params,
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

    def tick(
        self,
        *,
        window_payload: Optional[Dict[str, Any]] = None,
        communication_available: bool = True,
    ) -> List[SchedulerEvent]:
        """Advance one fast-path cycle and emit scheduler events."""
        return self.tick_with_fast_path(
            window_payload=window_payload,
            fast_path_fn=None,
            communication_available=communication_available,
        )

    def tick_with_fast_path(
        self,
        *,
        window_payload: Optional[Dict[str, Any]] = None,
        fast_path_fn: Optional[FastPathFn] = None,
        communication_available: bool = True,
    ) -> List[SchedulerEvent]:
        """Advance one fast-path cycle and optionally execute the fast-loop callback."""
        events: List[SchedulerEvent] = []
        self.epoch_id += 1
        self.time_us = self.epoch_id * self.config.t_fast_us
        self._update_communication_state(communication_available, events)

        fast_cycle_latency_us = self.latency_injector.sample_fast_cycle()
        self.last_fast_cycle_latency_us = fast_cycle_latency_us
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

        will_emit_window = self.epoch_id >= self._next_window_emit_epoch
        if fast_path_fn is not None:
            fast_payload = fast_path_fn(self.epoch_id, self.time_us, will_emit_window)
            if will_emit_window and fast_payload is not None:
                if window_payload is not None:
                    raise ValueError("window_payload and fast_path_fn both produced window payload")
                window_payload = fast_payload

        if will_emit_window:
            self._emit_window(window_payload, events)

        self._maybe_start_slow_job(events)
        return events

    def run(
        self,
        n_cycles: int,
        *,
        window_payload_factory: Optional[Callable[[int, int], Dict[str, Any]]] = None,
        fast_path_fn: Optional[FastPathFn] = None,
        communication_available_fn: Optional[Callable[[int, float], bool]] = None,
    ) -> List[SchedulerEvent]:
        if n_cycles <= 0:
            raise ValueError("n_cycles must be positive")

        collected: List[SchedulerEvent] = []
        for _ in range(n_cycles):
            payload = None
            will_emit_window = self.epoch_id + 1 >= self._next_window_emit_epoch
            if will_emit_window and window_payload_factory is not None:
                payload = window_payload_factory(self._window_counter + 1, self.epoch_id + 1)
            next_epoch = self.epoch_id + 1
            next_time_us = next_epoch * self.config.t_fast_us
            communication_available = (
                True
                if communication_available_fn is None
                else bool(communication_available_fn(next_epoch, next_time_us))
            )
            collected.extend(
                self.tick_with_fast_path(
                    window_payload=payload,
                    fast_path_fn=fast_path_fn,
                    communication_available=communication_available,
                )
            )
        return collected

    def stage_external_update(
        self,
        params: DecoderRuntimeParams,
        *,
        commit_epoch: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[PendingCommit], List[SchedulerEvent]]:
        """Stage an external update or emit an explicit conflict event."""

        events: List[SchedulerEvent] = []
        try:
            pending = self.param_bank.stage_update(
                params,
                commit_epoch=commit_epoch,
                staged_epoch=self.epoch_id,
                metadata=dict(metadata or {}),
            )
        except ParameterUpdateConflictError as exc:
            self.parameter_update_conflicts += 1
            self._record(
                SchedulerEvent(
                    kind="parameter_update_conflict",
                    epoch_id=self.epoch_id,
                    time_us=self.time_us,
                    details={
                        "commit_epoch": commit_epoch,
                        "reason": str(exc),
                        "active_version": self.param_bank.active_version,
                    },
                ),
                events,
            )
            return None, events
        self._record(
            SchedulerEvent(
                kind="external_params_staged",
                epoch_id=self.epoch_id,
                time_us=self.time_us,
                details={
                    "target_bank": pending.target_bank,
                    "commit_epoch": pending.commit_epoch,
                    "version": pending.version,
                },
            ),
            events,
        )
        return pending, events

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
            "window_deadline_misses": self.window_deadline_misses,
            "dropped_windows": self.dropped_windows,
            "fifo_overflows": self.fifo_overflows,
            "input_bursts": self.input_bursts,
            "communication_available": self._communication_available,
            "communication_pauses": self.communication_pauses,
            "communication_paused_cycles": self.communication_paused_cycles,
            "parameter_update_conflicts": self.parameter_update_conflicts,
            "last_fast_cycle_latency_us": self.last_fast_cycle_latency_us,
            "param_bank": self.param_bank.snapshot(),
        }
