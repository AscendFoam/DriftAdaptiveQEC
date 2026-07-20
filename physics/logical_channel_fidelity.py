"""Fidelity and short-time rate metrics for CPTNI logical subchannels.

For a trace-preserving qubit channel the familiar identity-target relation is
``F_avg=(2 F_e+1)/3``.  A projected code-space channel with leakage is trace
non-increasing, so silently using the TP constant ``1`` overstates fidelity.
For the unnormalized CPTNI map used by T5.3.1 the correct relations are

``F_e = Tr(R)/4`` and ``F_avg = (2 F_e + R_II)/3``.

Here ``F_avg`` is the Haar average of the unnormalized output overlap; mapping
all leaked weight to an orthogonal erasure flag gives the same value.  It is
therefore leakage-inclusive.  Per-state normalization is exposed only as a
diagnostic because state-dependent survival makes the conditional map nonlinear.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physics.fock_logical_channel import STATE_LABELS, logical_eigenstate_density


@dataclass(frozen=True)
class CPTNIFidelityMetrics:
    entanglement_fidelity: float
    average_fidelity: float
    mean_code_survival: float
    tp_assuming_average_fidelity: float
    tp_formula_overstatement: float
    direct_six_state_average_fidelity: float | None
    direct_six_state_mean_survival: float | None
    mean_conditional_state_fidelity: float | None
    minimum_conditional_state_fidelity: float | None
    maximum_conditional_state_fidelity: float | None
    minimum_state_survival: float | None
    maximum_state_survival: float | None
    six_state_ptm_residual: float | None
    six_state_survival_residual: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entanglement_fidelity": self.entanglement_fidelity,
            "average_fidelity": self.average_fidelity,
            "mean_code_survival": self.mean_code_survival,
            "tp_assuming_average_fidelity": self.tp_assuming_average_fidelity,
            "tp_formula_overstatement": self.tp_formula_overstatement,
            "direct_six_state_average_fidelity": self.direct_six_state_average_fidelity,
            "direct_six_state_mean_survival": self.direct_six_state_mean_survival,
            "mean_conditional_state_fidelity": self.mean_conditional_state_fidelity,
            "minimum_conditional_state_fidelity": self.minimum_conditional_state_fidelity,
            "maximum_conditional_state_fidelity": self.maximum_conditional_state_fidelity,
            "minimum_state_survival": self.minimum_state_survival,
            "maximum_state_survival": self.maximum_state_survival,
            "six_state_ptm_residual": self.six_state_ptm_residual,
            "six_state_survival_residual": self.six_state_survival_residual,
            "metric_scope": "leakage-inclusive unnormalized CPTNI identity-target fidelity",
            "conditional_metric_role": "diagnostic_only_not_a_linear_channel_fidelity",
        }


def _validate_ptm(ptm: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(ptm, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("ptm must be a finite real 4x4 matrix")
    return matrix


def cptni_identity_fidelity(
    ptm: ArrayLike,
    *,
    outputs: Mapping[str, ArrayLike] | None = None,
    tolerance: float = 2.0e-9,
) -> CPTNIFidelityMetrics:
    """Compute leakage-inclusive identity-target ``F_e`` and ``F_avg``."""

    matrix = _validate_ptm(ptm)
    r_ii = float(matrix[0, 0])
    entanglement = float(np.trace(matrix) / 4.0)
    average = float((2.0 * entanglement + r_ii) / 3.0)
    tp_assuming = float((2.0 * entanglement + 1.0) / 3.0)
    overstatement = float(tp_assuming - average)
    for name, value in (
        ("R_II", r_ii),
        ("entanglement fidelity", entanglement),
        ("average fidelity", average),
    ):
        if value < -tolerance or value > 1.0 + tolerance:
            raise ValueError(f"{name} lies outside the CPTNI fidelity range")
    if average > r_ii + tolerance:
        raise ValueError("average unnormalized overlap cannot exceed mean survival")
    if abs(overstatement - (1.0 - r_ii) / 3.0) > tolerance:
        raise RuntimeError("TP-formula overstatement identity failed")

    if outputs is None:
        return CPTNIFidelityMetrics(
            entanglement_fidelity=entanglement,
            average_fidelity=average,
            mean_code_survival=r_ii,
            tp_assuming_average_fidelity=tp_assuming,
            tp_formula_overstatement=overstatement,
            direct_six_state_average_fidelity=None,
            direct_six_state_mean_survival=None,
            mean_conditional_state_fidelity=None,
            minimum_conditional_state_fidelity=None,
            maximum_conditional_state_fidelity=None,
            minimum_state_survival=None,
            maximum_state_survival=None,
            six_state_ptm_residual=None,
            six_state_survival_residual=None,
        )

    if set(outputs) != set(STATE_LABELS):
        raise ValueError("outputs must contain exactly the six registered Pauli eigenstates")
    overlaps = []
    survivals = []
    conditional = []
    for label in STATE_LABELS:
        output = np.asarray(outputs[label], dtype=np.complex128)
        if output.shape != (2, 2) or not np.all(np.isfinite(output)):
            raise ValueError(f"output {label} must be a finite 2x2 matrix")
        if np.linalg.norm(output - output.conj().T, ord="fro") > tolerance:
            raise ValueError(f"output {label} must be Hermitian")
        survival = float(np.trace(output).real)
        overlap = float(np.trace(logical_eigenstate_density(label) @ output).real)
        if survival <= tolerance:
            raise ValueError("conditional state fidelity is undefined at zero survival")
        if overlap < -tolerance or overlap > survival + tolerance:
            raise ValueError("state overlap must lie between zero and survival")
        overlaps.append(overlap)
        survivals.append(survival)
        conditional.append(overlap / survival)
    direct_average = float(np.mean(overlaps))
    direct_survival = float(np.mean(survivals))
    fidelity_residual = abs(direct_average - average)
    survival_residual = abs(direct_survival - r_ii)
    if fidelity_residual > tolerance:
        raise ValueError("six-state direct average does not reproduce PTM average fidelity")
    if survival_residual > tolerance:
        raise ValueError("six-state mean survival does not reproduce R_II")
    return CPTNIFidelityMetrics(
        entanglement_fidelity=entanglement,
        average_fidelity=average,
        mean_code_survival=r_ii,
        tp_assuming_average_fidelity=tp_assuming,
        tp_formula_overstatement=overstatement,
        direct_six_state_average_fidelity=direct_average,
        direct_six_state_mean_survival=direct_survival,
        mean_conditional_state_fidelity=float(np.mean(conditional)),
        minimum_conditional_state_fidelity=float(np.min(conditional)),
        maximum_conditional_state_fidelity=float(np.max(conditional)),
        minimum_state_survival=float(np.min(survivals)),
        maximum_state_survival=float(np.max(survivals)),
        six_state_ptm_residual=fidelity_residual,
        six_state_survival_residual=survival_residual,
    )


@dataclass(frozen=True)
class ShortTimeDepolarization:
    step_us: float
    one_step_rate_per_us: float
    three_point_rate_per_us: float
    four_point_rate_per_us: float
    primary_rate_per_us: float
    primary_rate_per_cycle: float
    primary_lifetime_us: float | None
    primary_lifetime_cycles: float | None
    algebraic_inverse_rate_us: float | None
    algebraic_inverse_rate_cycles: float | None
    discretization_rate_min_per_us: float
    discretization_rate_max_per_us: float
    discretization_spread_per_us: float
    initial_curvature_per_us2: float
    initial_average_fidelity: float
    first_cycle_average_fidelity: float
    second_cycle_average_fidelity: float
    first_three_monotone_nonincreasing: bool
    relative_discretization_spread: float
    reliability_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_us": self.step_us,
            "one_step_rate_per_us": self.one_step_rate_per_us,
            "three_point_rate_per_us": self.three_point_rate_per_us,
            "four_point_rate_per_us": self.four_point_rate_per_us,
            "primary_rate_per_us": self.primary_rate_per_us,
            "primary_rate_per_cycle": self.primary_rate_per_cycle,
            "primary_lifetime_us": self.primary_lifetime_us,
            "primary_lifetime_cycles": self.primary_lifetime_cycles,
            "algebraic_inverse_rate_us": self.algebraic_inverse_rate_us,
            "algebraic_inverse_rate_cycles": self.algebraic_inverse_rate_cycles,
            "discretization_rate_min_per_us": self.discretization_rate_min_per_us,
            "discretization_rate_max_per_us": self.discretization_rate_max_per_us,
            "discretization_spread_per_us": self.discretization_spread_per_us,
            "initial_curvature_per_us2": self.initial_curvature_per_us2,
            "initial_average_fidelity": self.initial_average_fidelity,
            "first_cycle_average_fidelity": self.first_cycle_average_fidelity,
            "second_cycle_average_fidelity": self.second_cycle_average_fidelity,
            "first_three_monotone_nonincreasing": self.first_three_monotone_nonincreasing,
            "relative_discretization_spread": self.relative_discretization_spread,
            "reliability_status": self.reliability_status,
            "primary_definition": "second-order three-point forward derivative of F_avg at t=0, Gamma=-2 dF_avg/dt",
            "uncertainty_role": "one/three/four-point spread is time-discretization sensitivity, not a statistical confidence interval",
            "exponential_fit_used": False,
        }


def short_time_effective_depolarization(
    time_us: ArrayLike,
    average_fidelity: ArrayLike,
    *,
    tolerance: float = 2.0e-9,
) -> ShortTimeDepolarization:
    """Estimate ``Gamma=-2 dF_avg/dt|0`` without an exponential fit."""

    times = np.asarray(time_us, dtype=np.float64)
    values = np.asarray(average_fidelity, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape or times.size < 4:
        raise ValueError("time_us and average_fidelity must be aligned vectors with >=4 points")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("time and fidelity must be finite")
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time grid must start at zero and increase strictly")
    steps = np.diff(times)
    if not np.allclose(steps, steps[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("short-time finite differences require a uniform time grid")
    if abs(values[0] - 1.0) > tolerance:
        raise ValueError("initial average fidelity must equal one")
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("average fidelity values must lie in [0,1]")
    step = float(steps[0])
    derivative_one = (values[1] - values[0]) / step
    derivative_three = (-3.0 * values[0] + 4.0 * values[1] - values[2]) / (2.0 * step)
    derivative_four = (
        -11.0 * values[0] + 18.0 * values[1] - 9.0 * values[2] + 2.0 * values[3]
    ) / (6.0 * step)
    rates = -2.0 * np.array(
        [derivative_one, derivative_three, derivative_four], dtype=np.float64
    )
    if not np.all(np.isfinite(rates)):
        raise RuntimeError("short-time rate calculation produced non-finite values")
    primary = float(rates[1])
    algebraic_lifetime_us = None if primary <= 0.0 else 1.0 / primary
    curvature = float((values[0] - 2.0 * values[1] + values[2]) / (step * step))
    spread = float(np.max(rates) - np.min(rates))
    relative_spread = spread / max(abs(primary), 1.0e-15)
    monotone = bool(values[0] + tolerance >= values[1] >= values[2] - tolerance)
    reliable = bool(primary > 0.0 and monotone and relative_spread <= 0.25)
    reliability_status = (
        "reliable_discrete_short_time_proxy"
        if reliable
        else "unreliable_cycle_scale_transient"
    )
    qualified_lifetime_us = algebraic_lifetime_us if reliable else None
    return ShortTimeDepolarization(
        step_us=step,
        one_step_rate_per_us=float(rates[0]),
        three_point_rate_per_us=primary,
        four_point_rate_per_us=float(rates[2]),
        primary_rate_per_us=primary,
        primary_rate_per_cycle=primary * step,
        primary_lifetime_us=qualified_lifetime_us,
        primary_lifetime_cycles=(
            None if qualified_lifetime_us is None else qualified_lifetime_us / step
        ),
        algebraic_inverse_rate_us=algebraic_lifetime_us,
        algebraic_inverse_rate_cycles=(
            None if algebraic_lifetime_us is None else algebraic_lifetime_us / step
        ),
        discretization_rate_min_per_us=float(np.min(rates)),
        discretization_rate_max_per_us=float(np.max(rates)),
        discretization_spread_per_us=spread,
        initial_curvature_per_us2=curvature,
        initial_average_fidelity=float(values[0]),
        first_cycle_average_fidelity=float(values[1]),
        second_cycle_average_fidelity=float(values[2]),
        first_three_monotone_nonincreasing=monotone,
        relative_discretization_spread=relative_spread,
        reliability_status=reliability_status,
    )


def terminal_cutoff_interval(
    lower_value: float,
    higher_value: float,
    *,
    lower_cutoff: int,
    higher_cutoff: int,
) -> dict[str, Any]:
    """Return a deterministic two-cutoff numerical interval, never a CI."""

    values = np.asarray([lower_value, higher_value], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("cutoff values must be finite")
    if not 0 < int(lower_cutoff) < int(higher_cutoff):
        raise ValueError("cutoffs must be positive and strictly increasing")
    low = float(np.min(values))
    high = float(np.max(values))
    spread = high - low
    midpoint = 0.5 * (high + low)
    return {
        "lower_cutoff": int(lower_cutoff),
        "higher_cutoff": int(higher_cutoff),
        "value_at_lower_cutoff": float(values[0]),
        "value_at_higher_cutoff": float(values[1]),
        "numerical_interval_min": low,
        "numerical_interval_max": high,
        "absolute_spread": spread,
        "relative_half_spread": None if abs(midpoint) <= 1.0e-15 else 0.5 * spread / abs(midpoint),
        "uncertainty_type": "deterministic_terminal_cutoff_sensitivity",
        "statistical_confidence_level": None,
        "is_confidence_interval": False,
        "infinite_cutoff_claim": False,
    }


__all__ = [
    "CPTNIFidelityMetrics",
    "ShortTimeDepolarization",
    "cptni_identity_fidelity",
    "short_time_effective_depolarization",
    "terminal_cutoff_interval",
]
