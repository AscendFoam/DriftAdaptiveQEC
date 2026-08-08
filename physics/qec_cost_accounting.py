"""Unit-safe helpers for T5.3.4 QEC and post-selection cost ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log10
from typing import Any, Iterable


def _finite_probability(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def squeezing_db_from_projector_delta(projector_delta: float) -> float:
    """Return ``-10 log10(2 Delta^2)`` under the repository convention."""

    delta = float(projector_delta)
    if not isfinite(delta) or delta <= 0.0:
        raise ValueError("projector_delta must be finite and positive")
    return -10.0 * log10(2.0 * delta * delta)


@dataclass(frozen=True)
class ScaledProtocolCost:
    horizon_us: float
    full_cycles: int
    half_cycles: int
    measurement_events: int
    reset_events: int
    active_gate_applications: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_us": self.horizon_us,
            "full_cycles": self.full_cycles,
            "half_cycles": self.half_cycles,
            "measurement_events": self.measurement_events,
            "reset_events": self.reset_events,
            "active_gate_applications": self.active_gate_applications,
        }


def scale_measurement_feedback_cost(
    *,
    horizon_us: float,
    cycle_duration_us: float,
    measurements_per_full_cycle: int,
    resets_per_full_cycle: int,
    active_gates_per_full_cycle: int,
    tolerance: float = 1.0e-9,
) -> ScaledProtocolCost:
    """Scale integer event counts to an exactly divisible wall-clock horizon."""

    horizon = float(horizon_us)
    duration = float(cycle_duration_us)
    if not isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon_us must be finite and positive")
    if not isfinite(duration) or duration <= 0.0:
        raise ValueError("cycle_duration_us must be finite and positive")
    ratio = horizon / duration
    cycles = int(round(ratio))
    if abs(ratio - cycles) > tolerance:
        raise ValueError("horizon must contain an integer number of full cycles")
    counts = []
    for value, name in (
        (measurements_per_full_cycle, "measurements_per_full_cycle"),
        (resets_per_full_cycle, "resets_per_full_cycle"),
        (active_gates_per_full_cycle, "active_gates_per_full_cycle"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        counts.append(value)
    return ScaledProtocolCost(
        horizon_us=horizon,
        full_cycles=cycles,
        half_cycles=2 * cycles,
        measurement_events=cycles * counts[0],
        reset_events=cycles * counts[1],
        active_gate_applications=cycles * counts[2],
    )


@dataclass(frozen=True)
class PostselectionCost:
    acceptance_fraction: float
    rejection_fraction: float
    raw_error_rate: float
    conditional_error_rate: float
    accepted_failures_per_input: float
    rejection_penalties: tuple[float, ...]
    total_costs: tuple[float, ...]
    break_even_rejection_penalty: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "acceptance_fraction": self.acceptance_fraction,
            "rejection_fraction": self.rejection_fraction,
            "raw_error_rate": self.raw_error_rate,
            "conditional_error_rate": self.conditional_error_rate,
            "accepted_failures_per_input": self.accepted_failures_per_input,
            "rejection_penalties": list(self.rejection_penalties),
            "total_costs": list(self.total_costs),
            "total_cost_by_rejection_penalty": {
                f"{penalty:.2f}": cost
                for penalty, cost in zip(
                    self.rejection_penalties, self.total_costs, strict=True
                )
            },
            "break_even_rejection_penalty": self.break_even_rejection_penalty,
            "cost_definition": (
                "accepted_failures_per_input + rejection_penalty * rejection_fraction"
            ),
            "conditional_metric_online_eligible": False,
        }


def postselection_cost(
    *,
    acceptance_fraction: float,
    raw_error_rate: float,
    conditional_error_rate: float,
    rejection_penalties: Iterable[float] = (0.0, 0.25, 0.5, 1.0),
    tolerance: float = 1.0e-12,
) -> PostselectionCost:
    """Price rejection without mistaking conditional error for input-level cost."""

    acceptance = _finite_probability(acceptance_fraction, "acceptance_fraction")
    raw = _finite_probability(raw_error_rate, "raw_error_rate")
    conditional = _finite_probability(conditional_error_rate, "conditional_error_rate")
    if conditional > raw + tolerance:
        raise ValueError("conditional_error_rate cannot exceed raw_error_rate in this diagnostic")
    penalties = tuple(float(item) for item in rejection_penalties)
    if not penalties or any(not isfinite(item) or item < 0.0 for item in penalties):
        raise ValueError("rejection penalties must be a nonempty finite nonnegative sequence")
    if len(set(penalties)) != len(penalties) or tuple(sorted(penalties)) != penalties:
        raise ValueError("rejection penalties must be unique and sorted")
    rejection = 1.0 - acceptance
    accepted_failures = acceptance * conditional
    total = tuple(accepted_failures + penalty * rejection for penalty in penalties)
    if rejection <= tolerance:
        break_even = None
    else:
        break_even = (raw - accepted_failures) / rejection
    return PostselectionCost(
        acceptance_fraction=acceptance,
        rejection_fraction=rejection,
        raw_error_rate=raw,
        conditional_error_rate=conditional,
        accepted_failures_per_input=accepted_failures,
        rejection_penalties=penalties,
        total_costs=total,
        break_even_rejection_penalty=break_even,
    )


__all__ = [
    "PostselectionCost",
    "ScaledProtocolCost",
    "postselection_cost",
    "scale_measurement_feedback_cost",
    "squeezing_db_from_projector_delta",
]

