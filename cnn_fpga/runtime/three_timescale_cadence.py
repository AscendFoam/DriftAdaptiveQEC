"""Executable cadence and adaptation-lag contract for the three runtime timescales.

The contract intentionally separates the local event reaction from window-driven
host adaptation.  All epochs are one-based fast-cycle boundaries, matching
``DualLoopScheduler``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal


MODEL_SCOPE = "software_cadence_contract_not_rtl_or_board_measurement"
EvidencePolicy = Literal["first_influenced_window", "first_full_post_change_window"]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _positive_fraction(value: object, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be numeric")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _exact_cycle_ratio(period_us: object, fast_us: object, name: str) -> int:
    ratio = _positive_fraction(period_us, name) / _positive_fraction(fast_us, "t_fast_us")
    if ratio.denominator != 1:
        raise ValueError(f"{name} must be an integer multiple of t_fast_us")
    return int(ratio)


@dataclass(frozen=True)
class ThreeTimescaleCadenceConfig:
    """Frozen production-reference cadence, not a measured hardware timing claim."""

    t_fast_us: float = 5.0
    window_size: int = 2048
    window_stride: int = 4000
    slow_update_period_us: float = 20_000.0
    slow_service_us: float = 995.0
    commit_delay_cycles: int = 1
    event_register_cycles: int = 1
    max_parameter_age_cycles: int = 8192
    recalibration_period_us: float = 60_000_000.0
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        _positive_fraction(self.t_fast_us, "t_fast_us")
        _positive_fraction(self.slow_update_period_us, "slow_update_period_us")
        _positive_fraction(self.slow_service_us, "slow_service_us")
        _positive_fraction(self.recalibration_period_us, "recalibration_period_us")
        for name in (
            "window_size",
            "window_stride",
            "commit_delay_cycles",
            "event_register_cycles",
            "max_parameter_age_cycles",
        ):
            _positive_int(getattr(self, name), name)
        if self.window_size > self.window_stride:
            raise ValueError("window_size must not exceed window_stride in the frozen cadence")
        if self.event_register_cycles != 1:
            raise ValueError("the T4.2 health/event action register is exactly one cycle")
        if self.commit_delay_cycles != 1:
            raise ValueError("the frozen scheduler stages for the next cycle boundary")
        if self.slow_period_cycles != self.window_stride:
            raise ValueError("slow update period and window stride must be phase-locked")
        if self.recalibration_period_cycles % self.slow_period_cycles != 0:
            raise ValueError("minute recalibration must align to a slow/window boundary")
        if self.max_parameter_age_cycles < 2 * self.slow_period_cycles:
            raise ValueError("max parameter age must tolerate one missed slow update")
        if self.max_parameter_age_cycles >= 1 << 16:
            raise ValueError("max parameter age must fit the T4.2 16-bit age word")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")

    @property
    def slow_period_cycles(self) -> int:
        return _exact_cycle_ratio(
            self.slow_update_period_us, self.t_fast_us, "slow_update_period_us"
        )

    @property
    def slow_service_cycles(self) -> int:
        ratio = _positive_fraction(self.slow_service_us, "slow_service_us") / _positive_fraction(
            self.t_fast_us, "t_fast_us"
        )
        # A zero-latency job would still finish on the next scheduler tick because
        # job completion is checked before a new job is started.  Service is positive
        # here, but max(1, ...) preserves that scheduler ordering explicitly.
        return max(1, math.ceil(ratio))

    @property
    def recalibration_period_cycles(self) -> int:
        return _exact_cycle_ratio(
            self.recalibration_period_us, self.t_fast_us, "recalibration_period_us"
        )

    @property
    def window_content_us(self) -> float:
        return float(_positive_fraction(self.t_fast_us, "t_fast_us") * self.window_size)


@dataclass(frozen=True)
class AdaptationLagRecord:
    """One phase-resolved local-event and host-adaptation schedule."""

    evidence_policy: EvidencePolicy
    onset_epoch: int
    event_action_epoch: int
    window_id: int
    window_start_epoch: int
    window_end_epoch: int
    post_change_samples: int
    slow_start_epoch: int
    slow_finish_epoch: int
    stage_epoch: int
    commit_epoch: int
    first_use_epoch: int
    evidence_wait_cycles: int
    queue_wait_cycles: int
    service_cycles: int
    commit_wait_cycles: int
    first_use_wait_cycles: int
    total_lag_cycles: int
    total_lag_us: float
    event_lag_cycles: int
    event_lag_us: float

    def __post_init__(self) -> None:
        if self.event_action_epoch - self.onset_epoch != self.event_lag_cycles:
            raise ValueError("event lag decomposition is inconsistent")
        component_sum = (
            self.evidence_wait_cycles
            + self.queue_wait_cycles
            + self.service_cycles
            + self.commit_wait_cycles
            + self.first_use_wait_cycles
        )
        if component_sum != self.total_lag_cycles:
            raise ValueError("adaptation lag decomposition is inconsistent")
        if self.first_use_epoch - self.onset_epoch != self.total_lag_cycles:
            raise ValueError("first-use epoch and total lag are inconsistent")

    def to_dict(self) -> dict[str, int | float | str]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class RecalibrationTrigger:
    epoch: int
    time_us: float
    kinds: tuple[str, ...]

    def to_dict(self) -> dict[str, int | float | list[str]]:
        return {"epoch": self.epoch, "time_us": self.time_us, "kinds": list(self.kinds)}


class ThreeTimescaleCadence:
    """Integer-epoch cadence calculator tied to the scheduler's real operation order."""

    def __init__(self, config: ThreeTimescaleCadenceConfig | None = None) -> None:
        self.config = ThreeTimescaleCadenceConfig() if config is None else config
        if not isinstance(self.config, ThreeTimescaleCadenceConfig):
            raise TypeError("config must be ThreeTimescaleCadenceConfig")

    def adaptation_schedule(
        self,
        onset_epoch: int,
        *,
        evidence_policy: EvidencePolicy = "first_influenced_window",
    ) -> AdaptationLagRecord:
        onset = _positive_int(onset_epoch, "onset_epoch")
        if evidence_policy == "first_influenced_window":
            required_samples = 1
        elif evidence_policy == "first_full_post_change_window":
            required_samples = self.config.window_size
        else:
            raise ValueError(f"unsupported evidence_policy: {evidence_policy}")

        required_end = onset + required_samples - 1
        if required_end <= self.config.window_size:
            offset = 0
        else:
            offset = (
                required_end - self.config.window_size + self.config.window_stride - 1
            ) // self.config.window_stride
        window_end = self.config.window_size + offset * self.config.window_stride
        window_start = window_end - self.config.window_size + 1
        post_change_samples = window_end - max(window_start, onset) + 1
        if post_change_samples < required_samples:
            raise RuntimeError("internal window-alignment error")

        # In the frozen production configuration each emitted window is consumed in
        # the same epoch: window stride equals the slow-start period and service is
        # shorter than that period.  The scheduler then finishes/stages before the
        # fast callback, commits at the following boundary, and that callback is the
        # first user of the new version.
        slow_start = window_end
        slow_finish = slow_start + self.config.slow_service_cycles
        stage_epoch = slow_finish
        commit_epoch = stage_epoch + self.config.commit_delay_cycles
        first_use = commit_epoch
        evidence_wait = window_end - onset
        queue_wait = slow_start - window_end
        use_wait = first_use - commit_epoch
        total = first_use - onset
        fast_us = float(self.config.t_fast_us)
        return AdaptationLagRecord(
            evidence_policy=evidence_policy,
            onset_epoch=onset,
            event_action_epoch=onset + self.config.event_register_cycles,
            window_id=offset + 1,
            window_start_epoch=window_start,
            window_end_epoch=window_end,
            post_change_samples=post_change_samples,
            slow_start_epoch=slow_start,
            slow_finish_epoch=slow_finish,
            stage_epoch=stage_epoch,
            commit_epoch=commit_epoch,
            first_use_epoch=first_use,
            evidence_wait_cycles=evidence_wait,
            queue_wait_cycles=queue_wait,
            service_cycles=self.config.slow_service_cycles,
            commit_wait_cycles=self.config.commit_delay_cycles,
            first_use_wait_cycles=use_wait,
            total_lag_cycles=total,
            total_lag_us=total * fast_us,
            event_lag_cycles=self.config.event_register_cycles,
            event_lag_us=self.config.event_register_cycles * fast_us,
        )

    def phase_sweep(self, *, evidence_policy: EvidencePolicy) -> tuple[AdaptationLagRecord, ...]:
        """Sweep every onset phase exactly once over a slow/window stride."""

        return tuple(
            self.adaptation_schedule(epoch, evidence_policy=evidence_policy)
            for epoch in range(1, self.config.window_stride + 1)
        )

    def recalibration_schedule(
        self,
        run_end_epoch: int,
        *,
        run_start_epoch: int = 1,
        include_end_of_run: bool = True,
    ) -> tuple[RecalibrationTrigger, ...]:
        """Return minute-boundary and explicit end-of-run administrative triggers.

        These are *due* signals for the host calibration lane.  They do not mutate an
        active bank and therefore do not bypass validation/commit rules.
        """

        start = _positive_int(run_start_epoch, "run_start_epoch")
        end = _positive_int(run_end_epoch, "run_end_epoch")
        if end < start:
            raise ValueError("run_end_epoch must be >= run_start_epoch")
        period = self.config.recalibration_period_cycles
        first_index = max(1, (start + period - 1) // period)
        kinds_by_epoch: dict[int, list[str]] = {}
        for epoch in range(first_index * period, end + 1, period):
            kinds_by_epoch.setdefault(epoch, []).append("periodic_minute")
        if include_end_of_run:
            kinds_by_epoch.setdefault(end, []).append("end_of_run")
        return tuple(
            RecalibrationTrigger(
                epoch=epoch,
                time_us=epoch * float(self.config.t_fast_us),
                kinds=tuple(kinds_by_epoch[epoch]),
            )
            for epoch in sorted(kinds_by_epoch)
        )
