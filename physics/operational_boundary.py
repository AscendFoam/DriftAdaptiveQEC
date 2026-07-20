"""Full-curve operational-boundary metrics for matched logical channels.

This module deliberately does not define a coherence-gain ratio.  Such a ratio
requires a qualified short-time rate for the active channel and a best-passive
physical-qubit reference.  T5.3.1/T5.3.2 provide neither: they provide an
active nominal-sBs channel and a matched idle evolution of the same encoded
state.  The defensible estimand is therefore a sampled, wall-clock matched
operational boundary on the complete leakage-inclusive average-fidelity curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class OperationalBoundary:
    time_us: NDArray[np.float64]
    active_fidelity: NDArray[np.float64]
    passive_fidelity: NDArray[np.float64]
    pointwise_advantage: NDArray[np.float64]
    cumulative_advantage_us: NDArray[np.float64]
    first_positive_index: int | None
    last_negative_index: int | None
    sustained_dominance_index: int | None
    sustained_dominance_time_us: float | None
    sustained_linear_crossing_time_us: float | None
    cumulative_breakeven_index: int | None
    cumulative_breakeven_time_us: float | None
    cumulative_linear_crossing_time_us: float | None
    initial_penalty_min: float
    initial_penalty_time_us: float
    terminal_advantage: float
    terminal_cumulative_advantage_us: float
    sign_reversal_count: int
    grid_step_us: float
    horizon_us: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_us": self.time_us.tolist(),
            "active_fidelity": self.active_fidelity.tolist(),
            "passive_fidelity": self.passive_fidelity.tolist(),
            "pointwise_advantage": self.pointwise_advantage.tolist(),
            "cumulative_advantage_us": self.cumulative_advantage_us.tolist(),
            "first_positive_index": self.first_positive_index,
            "last_negative_index": self.last_negative_index,
            "sustained_dominance_index": self.sustained_dominance_index,
            "sustained_dominance_time_us": self.sustained_dominance_time_us,
            "sustained_linear_crossing_time_us": self.sustained_linear_crossing_time_us,
            "cumulative_breakeven_index": self.cumulative_breakeven_index,
            "cumulative_breakeven_time_us": self.cumulative_breakeven_time_us,
            "cumulative_linear_crossing_time_us": self.cumulative_linear_crossing_time_us,
            "initial_penalty_min": self.initial_penalty_min,
            "initial_penalty_time_us": self.initial_penalty_time_us,
            "terminal_advantage": self.terminal_advantage,
            "terminal_cumulative_advantage_us": self.terminal_cumulative_advantage_us,
            "sign_reversal_count": self.sign_reversal_count,
            "grid_step_us": self.grid_step_us,
            "horizon_us": self.horizon_us,
            "boundary_definition": (
                "first sampled time after the last active-minus-passive fidelity "
                "disadvantage; active must remain noninferior through the full horizon"
            ),
            "cumulative_definition": (
                "first sampled time after which the trapezoidal integral of active-minus-"
                "passive fidelity remains nonnegative through the full horizon"
            ),
            "interpolation_role": "linear_between_sample_diagnostic_not_subgrid_validation",
            "ratio_reported": False,
            "exponential_fit_used": False,
        }


def _curve(name: str, values: ArrayLike, *, size: int | None = None) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if size is not None and array.size != size:
        raise ValueError("time and fidelity vectors must have equal length")
    if array.size < 3:
        raise ValueError(f"{name} must be a one-dimensional vector with at least 3 points")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _linear_zero_time(
    time: NDArray[np.float64], values: NDArray[np.float64], index: int | None
) -> float | None:
    if index is None:
        return None
    if index == 0 or values[index] == 0.0:
        return float(time[index])
    left = index - 1
    y0 = float(values[left])
    y1 = float(values[index])
    if y0 * y1 > 0.0 or y1 == y0:
        return None
    fraction = -y0 / (y1 - y0)
    return float(time[left] + fraction * (time[index] - time[left]))


def _class_sign(values: NDArray[np.float64], tolerance: float) -> list[int]:
    result: list[int] = []
    for value in values:
        if value > tolerance:
            sign = 1
        elif value < -tolerance:
            sign = -1
        else:
            continue
        if not result or result[-1] != sign:
            result.append(sign)
    return result


def matched_operational_boundary(
    time_us: ArrayLike,
    active_fidelity: ArrayLike,
    passive_fidelity: ArrayLike,
    *,
    tolerance: float = 2.0e-9,
) -> OperationalBoundary:
    """Evaluate pointwise and cumulative full-curve active/passive boundaries.

    The sampled sustained boundary is conservative: a transient improvement is
    not sufficient, and the active curve must remain noninferior at every later
    sampled horizon.  The cumulative boundary additionally repays any initial
    fidelity deficit.  Neither quantity is a lifetime or coherence-gain ratio.
    """

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    time = _curve("time_us", time_us)
    active = _curve("active_fidelity", active_fidelity, size=time.size)
    passive = _curve("passive_fidelity", passive_fidelity, size=time.size)
    if time[0] != 0.0 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time_us must start at zero and increase strictly")
    steps = np.diff(time)
    if not np.allclose(steps, steps[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("formal operational boundary requires a uniform time grid")
    if np.any(active < -tolerance) or np.any(active > 1.0 + tolerance):
        raise ValueError("active fidelity must lie in [0,1]")
    if np.any(passive < -tolerance) or np.any(passive > 1.0 + tolerance):
        raise ValueError("passive fidelity must lie in [0,1]")
    if abs(active[0] - passive[0]) > tolerance or abs(active[0] - 1.0) > tolerance:
        raise ValueError("matched curves must begin from the same unit-fidelity state")

    advantage = active - passive
    increments = 0.5 * (advantage[:-1] + advantage[1:]) * steps
    cumulative = np.concatenate((np.array([0.0]), np.cumsum(increments)))

    positive = np.flatnonzero(advantage > tolerance)
    negative = np.flatnonzero(advantage < -tolerance)
    first_positive = None if positive.size == 0 else int(positive[0])
    last_negative = None if negative.size == 0 else int(negative[-1])
    if last_negative is None:
        sustained = 0
    elif last_negative + 1 >= time.size:
        sustained = None
    else:
        sustained = last_negative + 1

    cumulative_negative = np.flatnonzero(cumulative < -tolerance)
    if cumulative_negative.size == 0:
        cumulative_boundary = 0
    elif int(cumulative_negative[-1]) + 1 >= time.size:
        cumulative_boundary = None
    else:
        cumulative_boundary = int(cumulative_negative[-1]) + 1

    minimum_index = int(np.argmin(advantage))
    signs = _class_sign(advantage, tolerance)
    return OperationalBoundary(
        time_us=time.copy(),
        active_fidelity=active.copy(),
        passive_fidelity=passive.copy(),
        pointwise_advantage=advantage,
        cumulative_advantage_us=cumulative,
        first_positive_index=first_positive,
        last_negative_index=last_negative,
        sustained_dominance_index=sustained,
        sustained_dominance_time_us=None if sustained is None else float(time[sustained]),
        sustained_linear_crossing_time_us=_linear_zero_time(time, advantage, sustained),
        cumulative_breakeven_index=cumulative_boundary,
        cumulative_breakeven_time_us=(
            None if cumulative_boundary is None else float(time[cumulative_boundary])
        ),
        cumulative_linear_crossing_time_us=_linear_zero_time(
            time, cumulative, cumulative_boundary
        ),
        initial_penalty_min=float(advantage[minimum_index]),
        initial_penalty_time_us=float(time[minimum_index]),
        terminal_advantage=float(advantage[-1]),
        terminal_cumulative_advantage_us=float(cumulative[-1]),
        sign_reversal_count=max(0, len(signs) - 1),
        grid_step_us=float(steps[0]),
        horizon_us=float(time[-1]),
    )


__all__ = ["OperationalBoundary", "matched_operational_boundary"]
