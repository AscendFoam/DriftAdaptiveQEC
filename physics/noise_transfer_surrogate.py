"""Heisenberg-inspired signal/noise/jump surrogate for GKP correction.

The model follows the separation advocated by Ralph et al., Entropy 26, 874
(2024): discrete lattice signal, continuous fluctuation covariance and discrete
domain-alias errors are propagated as distinct objects.  It is a low-cost
middle-fidelity proxy, not an SBS Kraus map or a pulse/device simulation.

The stochastic state is stored on two independently decoder-standardized axes
with cell spacing ``sqrt(2*pi)`` and vacuum variance ``1``.  Those axes are a
classical normalization, not a joint canonical operator pair.  Fock/Fourier
comparisons are converted to canonical ``[x,p]=i`` explicitly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import atanh, exp, floor, isfinite, pi, sqrt
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from ._shared.numerics import hermite_functions as _hermite_functions
from .constants import LATTICE_CONST
from .finite_energy_gkp import damped_projector_state
from .fock_density_model import FiniteCutoffFockModel
from .quadrature_conventions import (
    CANONICAL_LOGICAL_CELL_SPACING,
    QuadratureAxis,
    QuadratureChartName,
    chart,
)
from ._shared.validation import finite_positive as _finite_positive


RealMatrix = NDArray[np.float64]
Axis = Literal["q", "p"]
Validity = Literal["localized", "clipping_dominated"]

NOISE_TRANSFER_SCOPE = (
    "Heisenberg-inspired operational-coordinate signal/fluctuation/logical-jump "
    "surrogate; not an SBS Kraus recovery, pulse/transmon model, device calibration, "
    "or replacement for Fock cross-validation"
)


def _unit_interval(value: float, name: str, *, strictly_positive: bool = False) -> float:
    result = float(value)
    lower_ok = result > 0.0 if strictly_positive else result >= 0.0
    if not isfinite(result) or not lower_ok or result > 1.0:
        bracket = "(0, 1]" if strictly_positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {bracket}")
    return result


def _vector2(value: ArrayLike, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-2 vector")
    return float(array[0]), float(array[1])


def _matrix2(value: ArrayLike, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2, 2) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2x2 matrix")
    return tuple(tuple(float(item) for item in row) for row in array)  # type: ignore[return-value]


def _covariance2(
    value: ArrayLike, name: str
) -> tuple[tuple[float, float], tuple[float, float]]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 2x2 covariance")
    if np.linalg.norm(matrix - matrix.T, ord="fro") > 1.0e-12:
        raise ValueError(f"{name} must be symmetric")
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if float(np.min(values)) < -1.0e-12:
        raise ValueError(f"{name} must be positive semidefinite")
    cleaned = 0.5 * (matrix + matrix.T)
    return tuple(tuple(float(item) for item in row) for row in cleaned)  # type: ignore[return-value]


def squeezing_db_to_peak_variance(
    squeezing_db: float,
    *,
    vacuum_variance: float = 0.5,
    coordinate_chart: QuadratureChartName = "canonical_fock",
    axis: QuadratureAxis = "q",
) -> float:
    """Convert dB squeezing to a chart-qualified probability-peak variance.

    ``vacuum_variance`` is always the canonical ``[x,p]=i`` variance.  The
    registered chart scale is then squared, so decoder-standardized axes have
    vacuum variance one instead of one half.
    """

    db = float(squeezing_db)
    if not isfinite(db) or db < 0.0 or db > 40.0:
        raise ValueError("squeezing_db must lie in [0, 40]")
    vacuum = _finite_positive(vacuum_variance, "vacuum_variance")
    if axis not in {"q", "p"}:
        raise ValueError("axis must be q or p")
    registered = chart(coordinate_chart)
    scale = (
        registered.canonical_scale_q
        if axis == "q"
        else registered.canonical_scale_p
    )
    return float(scale * scale * vacuum * 10.0 ** (-db / 10.0))


def projector_delta_from_squeezing_db(squeezing_db: float) -> float:
    """Map dB to the repository damped-projector Delta convention.

    ``FiniteEnergyGKPState`` has isolated probability-peak variance
    ``tanh(Delta^2)/2``.  With repository vacuum variance 1/2 this gives
    ``tanh(Delta^2)=10^(-dB/10)``.
    """

    db = float(squeezing_db)
    if not isfinite(db) or db <= 0.0 or db > 40.0:
        raise ValueError("squeezing_db must lie in (0, 40]")
    ratio = 10.0 ** (-db / 10.0)
    return sqrt(atanh(ratio))


@dataclass(frozen=True)
class GaussianAliasStatistics:
    mean: float
    variance: float
    spacing: float
    alias_indices: tuple[int, ...]
    alias_probabilities: tuple[float, ...]
    probability_sum: float
    truncation_tail_bound: float
    central_probability: float
    odd_alias_probability: float
    ideal_center_folded_mean: float
    ideal_center_folded_variance: float
    domain_conditioned_variance: float
    clipping_ratio: float


def gaussian_alias_statistics(
    mean: float,
    variance: float,
    spacing: float,
    *,
    tail_tolerance: float = 1.0e-12,
) -> GaussianAliasStatistics:
    """Exact one-dimensional Gaussian cell probabilities and folded moments."""

    location = float(mean)
    if not isfinite(location):
        raise ValueError("mean must be finite")
    var = _finite_positive(variance, "variance")
    cell = _finite_positive(spacing, "spacing")
    tolerance = float(tail_tolerance)
    if not isfinite(tolerance) or not 0.0 < tolerance < 1.0e-3:
        raise ValueError("tail_tolerance must lie in (0, 1e-3)")
    sigma = sqrt(var)
    z_limit = float(ndtri(1.0 - tolerance / 2.0))
    lower = location - z_limit * sigma
    upper = location + z_limit * sigma
    n_min = int(floor(lower / cell - 0.5)) - 1
    n_max = int(floor(upper / cell + 0.5)) + 1
    indices = np.arange(n_min, n_max + 1, dtype=np.int64)
    lower_edges = (indices.astype(np.float64) - 0.5) * cell
    upper_edges = (indices.astype(np.float64) + 0.5) * cell
    alpha = (lower_edges - location) / sigma
    beta = (upper_edges - location) / sigma
    probabilities = ndtr(beta) - ndtr(alpha)
    normalizer = sqrt(2.0 * pi)
    phi_alpha = np.exp(-0.5 * alpha * alpha) / normalizer
    phi_beta = np.exp(-0.5 * beta * beta) / normalizer
    first = location * probabilities + sigma * (phi_alpha - phi_beta)
    second = (
        (location * location + var) * probabilities
        + 2.0 * location * sigma * (phi_alpha - phi_beta)
        + var * (alpha * phi_alpha - beta * phi_beta)
    )
    centers = indices.astype(np.float64) * cell
    folded_first = float(np.sum(first - centers * probabilities))
    folded_second = float(
        np.sum(second - 2.0 * centers * first + centers * centers * probabilities)
    )
    probability_sum = float(np.sum(probabilities))
    folded_mean = folded_first / probability_sum
    folded_variance = max(folded_second / probability_sum - folded_mean**2, 0.0)
    conditioned = 0.0
    for probability, first_moment, second_moment in zip(
        probabilities, first, second
    ):
        if probability > tolerance * 1.0e-3:
            conditioned += float(
                second_moment - first_moment * first_moment / probability
            )
    conditioned /= probability_sum
    central = float(probabilities[indices == 0][0]) if np.any(indices == 0) else 0.0
    odd = float(np.sum(probabilities[np.mod(np.abs(indices), 2) == 1]))
    tail = max(0.0, 1.0 - probability_sum)
    return GaussianAliasStatistics(
        mean=location,
        variance=var,
        spacing=cell,
        alias_indices=tuple(int(item) for item in indices),
        alias_probabilities=tuple(float(item) for item in probabilities),
        probability_sum=probability_sum,
        truncation_tail_bound=tail,
        central_probability=central,
        odd_alias_probability=odd,
        ideal_center_folded_mean=folded_mean,
        ideal_center_folded_variance=folded_variance,
        domain_conditioned_variance=conditioned,
        clipping_ratio=conditioned / var,
    )


@dataclass(frozen=True)
class NoiseTransferConfig:
    domain_spacing: float = LATTICE_CONST
    resource_covariance: tuple[tuple[float, float], tuple[float, float]] = (
        (0.05, 0.0),
        (0.0, 0.05),
    )
    loss_transmissivity: float = 1.0
    measurement_efficiency: float = 1.0
    feedforward_gain: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
    )
    vacuum_variance: float = 1.0
    alias_tail_tolerance: float = 1.0e-12
    localization_probability_gate: float = 0.95
    clipping_ratio_gate: float = 0.90
    scope: str = NOISE_TRANSFER_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "domain_spacing", _finite_positive(self.domain_spacing, "domain_spacing")
        )
        object.__setattr__(
            self,
            "resource_covariance",
            _covariance2(self.resource_covariance, "resource_covariance"),
        )
        object.__setattr__(
            self,
            "loss_transmissivity",
            _unit_interval(self.loss_transmissivity, "loss_transmissivity"),
        )
        object.__setattr__(
            self,
            "measurement_efficiency",
            _unit_interval(
                self.measurement_efficiency,
                "measurement_efficiency",
                strictly_positive=True,
            ),
        )
        object.__setattr__(
            self, "feedforward_gain", _matrix2(self.feedforward_gain, "feedforward_gain")
        )
        object.__setattr__(
            self,
            "vacuum_variance",
            _finite_positive(self.vacuum_variance, "vacuum_variance"),
        )
        tolerance = float(self.alias_tail_tolerance)
        if not isfinite(tolerance) or not 0.0 < tolerance < 1.0e-3:
            raise ValueError("alias_tail_tolerance must lie in (0, 1e-3)")
        object.__setattr__(self, "alias_tail_tolerance", tolerance)
        for name in ("localization_probability_gate", "clipping_ratio_gate"):
            value = _unit_interval(getattr(self, name), name, strictly_positive=True)
            object.__setattr__(self, name, value)
        if self.scope != NOISE_TRANSFER_SCOPE:
            raise ValueError("scope must preserve the fail-closed surrogate boundary")


@dataclass(frozen=True)
class NoiseTransferState:
    lattice_index: tuple[int, int] = (0, 0)
    signal_offset: tuple[float, float] = (0.0, 0.0)
    fluctuation_covariance: tuple[tuple[float, float], tuple[float, float]] = (
        (0.05, 0.0),
        (0.0, 0.05),
    )
    logical_parity: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if (
            len(self.lattice_index) != 2
            or any(not isinstance(item, (int, np.integer)) or isinstance(item, bool) for item in self.lattice_index)
        ):
            raise ValueError("lattice_index must contain two integers")
        object.__setattr__(
            self, "lattice_index", tuple(int(item) for item in self.lattice_index)
        )
        object.__setattr__(self, "signal_offset", _vector2(self.signal_offset, "signal_offset"))
        object.__setattr__(
            self,
            "fluctuation_covariance",
            _covariance2(self.fluctuation_covariance, "fluctuation_covariance"),
        )
        if len(self.logical_parity) != 2 or any(int(item) not in (0, 1) for item in self.logical_parity):
            raise ValueError("logical_parity must contain two bits")
        object.__setattr__(
            self, "logical_parity", tuple(int(item) for item in self.logical_parity)
        )


@dataclass(frozen=True)
class NoiseTransferAxisDiagnostic:
    axis: Axis
    alias_statistics: GaussianAliasStatistics


@dataclass(frozen=True)
class NoiseTransferLogicalJump:
    q_odd_probability: float
    p_odd_probability: float
    any_jump_lower_bound: float
    any_jump_upper_bound: float
    any_jump_probability: float | None
    pauli_i_probability: float | None
    pauli_x_probability: float | None
    pauli_z_probability: float | None
    pauli_y_probability: float | None
    joint_rule: str


@dataclass(frozen=True)
class NoiseTransferStepResult:
    input_state: NoiseTransferState
    lattice_signal: tuple[float, float]
    loss_bias: tuple[float, float]
    post_loss_covariance: tuple[tuple[float, float], tuple[float, float]]
    measurement_equivalent_covariance: tuple[tuple[float, float], tuple[float, float]]
    decision_covariance: tuple[tuple[float, float], tuple[float, float]]
    output_signal_offset: tuple[float, float]
    output_covariance: tuple[tuple[float, float], tuple[float, float]]
    axis_diagnostics: tuple[NoiseTransferAxisDiagnostic, NoiseTransferAxisDiagnostic]
    logical_jump: NoiseTransferLogicalJump
    validity: Validity
    scope: str = NOISE_TRANSFER_SCOPE


@dataclass(frozen=True)
class NoiseTransferSample:
    decision_sample: tuple[float, float]
    alias_indices: tuple[int, int]
    modular_residual: tuple[float, float]
    parity_jump: tuple[int, int]
    output_logical_parity: tuple[int, int]


class GKPNoiseTransferSurrogate:
    """Propagate factorized signal, covariance and alias-jump diagnostics."""

    def __init__(self, config: NoiseTransferConfig) -> None:
        if not isinstance(config, NoiseTransferConfig):
            raise TypeError("config must be a NoiseTransferConfig")
        self.config = config

    def propagate(self, state: NoiseTransferState) -> NoiseTransferStepResult:
        if not isinstance(state, NoiseTransferState):
            raise TypeError("state must be a NoiseTransferState")
        spacing = self.config.domain_spacing
        lattice = spacing * np.asarray(state.lattice_index, dtype=np.float64)
        offset = np.asarray(state.signal_offset, dtype=np.float64)
        eta = self.config.loss_transmissivity
        attenuated = sqrt(eta) * (lattice + offset)
        loss_bias = attenuated - lattice
        input_covariance = np.asarray(state.fluctuation_covariance, dtype=np.float64)
        identity = np.eye(2, dtype=np.float64)
        post_loss = eta * input_covariance + (1.0 - eta) * self.config.vacuum_variance * identity
        efficiency = self.config.measurement_efficiency
        resource = np.asarray(self.config.resource_covariance, dtype=np.float64)
        measurement = resource + (
            (1.0 - efficiency) / efficiency
        ) * self.config.vacuum_variance * identity
        decision = post_loss + measurement
        gain = np.asarray(self.config.feedforward_gain, dtype=np.float64)
        residual_gain = identity - gain
        output_signal = residual_gain @ loss_bias
        output_covariance = (
            residual_gain @ post_loss @ residual_gain.T
            + gain @ measurement @ gain.T
        )
        output_covariance = 0.5 * (output_covariance + output_covariance.T)
        axis_records = []
        for index, axis in enumerate(("q", "p")):
            statistics = gaussian_alias_statistics(
                float(loss_bias[index]),
                float(decision[index, index]),
                spacing,
                tail_tolerance=self.config.alias_tail_tolerance,
            )
            axis_records.append(
                NoiseTransferAxisDiagnostic(axis=axis, alias_statistics=statistics)
            )
        q_odd = axis_records[0].alias_statistics.odd_alias_probability
        p_odd = axis_records[1].alias_statistics.odd_alias_probability
        lower = max(q_odd, p_odd)
        upper = min(1.0, q_odd + p_odd)
        if abs(float(decision[0, 1])) <= 1.0e-14:
            p_i = (1.0 - q_odd) * (1.0 - p_odd)
            p_x = q_odd * (1.0 - p_odd)
            p_z = (1.0 - q_odd) * p_odd
            p_y = q_odd * p_odd
            any_jump: float | None = 1.0 - p_i
            joint_rule = "exact_axis_independence_for_diagonal_decision_covariance"
        else:
            p_i = p_x = p_z = p_y = None
            any_jump = None
            joint_rule = "correlated_axes_report_marginals_and_frechet_bounds_only"
        jump = NoiseTransferLogicalJump(
            q_odd_probability=q_odd,
            p_odd_probability=p_odd,
            any_jump_lower_bound=lower,
            any_jump_upper_bound=upper,
            any_jump_probability=any_jump,
            pauli_i_probability=p_i,
            pauli_x_probability=p_x,
            pauli_z_probability=p_z,
            pauli_y_probability=p_y,
            joint_rule=joint_rule,
        )
        localized = all(
            item.alias_statistics.central_probability
            >= self.config.localization_probability_gate
            and item.alias_statistics.clipping_ratio >= self.config.clipping_ratio_gate
            for item in axis_records
        )
        return NoiseTransferStepResult(
            input_state=state,
            lattice_signal=_vector2(lattice, "lattice_signal"),
            loss_bias=_vector2(loss_bias, "loss_bias"),
            post_loss_covariance=_covariance2(post_loss, "post_loss_covariance"),
            measurement_equivalent_covariance=_covariance2(
                measurement, "measurement_equivalent_covariance"
            ),
            decision_covariance=_covariance2(decision, "decision_covariance"),
            output_signal_offset=_vector2(output_signal, "output_signal_offset"),
            output_covariance=_covariance2(output_covariance, "output_covariance"),
            axis_diagnostics=(axis_records[0], axis_records[1]),
            logical_jump=jump,
            validity="localized" if localized else "clipping_dominated",
        )

    def sample_step(
        self,
        result: NoiseTransferStepResult,
        rng: np.random.Generator,
    ) -> NoiseTransferSample:
        if not isinstance(result, NoiseTransferStepResult):
            raise TypeError("result must be a NoiseTransferStepResult")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        sample = rng.multivariate_normal(
            mean=np.asarray(result.loss_bias, dtype=np.float64),
            cov=np.asarray(result.decision_covariance, dtype=np.float64),
        )
        spacing = self.config.domain_spacing
        aliases = np.floor(sample / spacing + 0.5).astype(np.int64)
        residual = sample - spacing * aliases
        parity_jump = np.mod(np.abs(aliases), 2).astype(np.int64)
        output_parity = np.bitwise_xor(
            np.asarray(result.input_state.logical_parity, dtype=np.int64),
            parity_jump,
        )
        return NoiseTransferSample(
            decision_sample=_vector2(sample, "decision_sample"),
            alias_indices=(int(aliases[0]), int(aliases[1])),
            modular_residual=_vector2(residual, "modular_residual"),
            parity_jump=(int(parity_jump[0]), int(parity_jump[1])),
            output_logical_parity=(int(output_parity[0]), int(output_parity[1])),
        )


def _domain_conditioned_variance_from_density(
    coordinate: NDArray[np.float64],
    density: NDArray[np.float64],
    spacing: float,
) -> float:
    mass = float(np.trapz(density, coordinate))
    if not isfinite(mass) or mass <= 0.0:
        raise RuntimeError("density has invalid mass")
    probability = density / mass
    indices = np.floor(coordinate / spacing + 0.5).astype(np.int64)
    variance = 0.0
    for index in np.unique(indices):
        mask = indices == index
        if int(np.sum(mask)) < 2:
            continue
        local_x = coordinate[mask]
        local_p = probability[mask]
        local_mass = float(np.trapz(local_p, local_x))
        if local_mass <= 1.0e-15:
            continue
        first = float(np.trapz(local_x * local_p, local_x))
        second = float(np.trapz(local_x * local_x * local_p, local_x))
        variance += second - first * first / local_mass
    return max(float(variance), 0.0)


@dataclass(frozen=True)
class NoiseTransferFockAlignmentPoint:
    squeezing_db: float
    projector_delta: float
    proxy_peak_variance: float
    direct_state_variances: tuple[float, float, float, float]
    fock_variances: tuple[float, float, float, float]
    captured_probabilities: tuple[float, float, float, float]
    maximum_proxy_to_direct_relative_error: float
    maximum_fock_to_direct_relative_error: float
    direct_state_relative_spread: float


def fock_q_variance_alignment(
    squeezing_db: float,
    *,
    cutoff: int = 48,
    projection_grid_points: int = 8193,
    quadrature_grid_points: int = 16385,
) -> NoiseTransferFockAlignmentPoint:
    """Compare proxy peak variance with state-level and Fock q-domain moments."""

    if not isinstance(cutoff, (int, np.integer)) or not 12 <= int(cutoff) <= 48:
        raise ValueError("cutoff must be an integer in [12, 48]")
    if (
        not isinstance(projection_grid_points, (int, np.integer))
        or int(projection_grid_points) < 1025
        or int(projection_grid_points) % 2 == 0
    ):
        raise ValueError("projection_grid_points must be an odd integer >= 1025")
    if (
        not isinstance(quadrature_grid_points, (int, np.integer))
        or int(quadrature_grid_points) < 4097
        or int(quadrature_grid_points) % 2 == 0
    ):
        raise ValueError("quadrature_grid_points must be an odd integer >= 4097")
    delta = projector_delta_from_squeezing_db(squeezing_db)
    target = squeezing_db_to_peak_variance(
        squeezing_db, coordinate_chart="canonical_fock"
    )
    model = FiniteCutoffFockModel(int(cutoff))
    direct_values = []
    fock_values = []
    captured = []
    for label in ("0", "1", "+", "-"):
        source = damped_projector_state(label, delta)
        direct_coordinate = np.linspace(
            -source.support_radius,
            source.support_radius,
            int(quadrature_grid_points),
            dtype=np.float64,
        )
        direct_density = np.asarray(source.probability_density(direct_coordinate))
        direct_values.append(
            _domain_conditioned_variance_from_density(
                direct_coordinate, direct_density, LATTICE_CONST
            )
            / 2.0
        )
        preparation = model.prepare_damped_projector_gkp(
            label,
            delta,
            grid_points=int(projection_grid_points),
        )
        captured.append(preparation.captured_probability)
        fock_extent = min(
            preparation.q_extent,
            sqrt(2.0 * int(cutoff)) + 6.0,
        )
        fock_coordinate = np.linspace(
            -fock_extent,
            fock_extent,
            int(quadrature_grid_points),
            dtype=np.float64,
        )
        functions = _hermite_functions(fock_coordinate, int(cutoff))
        coefficients = preparation.coefficients / np.linalg.norm(preparation.coefficients)
        wavefunction = coefficients @ functions
        fock_values.append(
            _domain_conditioned_variance_from_density(
                fock_coordinate,
                np.abs(wavefunction) ** 2,
                CANONICAL_LOGICAL_CELL_SPACING,
            )
        )
    direct = np.asarray(direct_values)
    fock = np.asarray(fock_values)
    proxy_error = float(np.max(np.abs(direct - target) / target))
    fock_error = float(np.max(np.abs(fock - direct) / np.maximum(direct, 1.0e-15)))
    spread = float((np.max(direct) - np.min(direct)) / target)
    return NoiseTransferFockAlignmentPoint(
        squeezing_db=float(squeezing_db),
        projector_delta=delta,
        proxy_peak_variance=target,
        direct_state_variances=tuple(float(item) for item in direct),  # type: ignore[arg-type]
        fock_variances=tuple(float(item) for item in fock),  # type: ignore[arg-type]
        captured_probabilities=tuple(float(item) for item in captured),  # type: ignore[arg-type]
        maximum_proxy_to_direct_relative_error=proxy_error,
        maximum_fock_to_direct_relative_error=fock_error,
        direct_state_relative_spread=spread,
    )


@dataclass(frozen=True)
class NoiseTransferSweepPoint:
    squeezing_db: float
    decision_variance: float
    central_probability: float
    odd_alias_probability: float
    clipping_ratio: float
    validity: Validity


@dataclass(frozen=True)
class NoiseTransferValidationResult:
    squeezing_sweep: tuple[NoiseTransferSweepPoint, ...]
    fock_alignment: tuple[
        NoiseTransferFockAlignmentPoint,
        NoiseTransferFockAlignmentPoint,
        NoiseTransferFockAlignmentPoint,
    ]
    monte_carlo_samples: int
    monte_carlo_max_z_score: float
    monte_carlo_covariance_relative_error: float
    unity_gain_refresh_error: float
    inefficient_measurement_variance_increase: float
    loss_state_bias_ratio: float
    low_squeezing_clipping_gap: float
    checks: dict[str, bool]
    scope: str = NOISE_TRANSFER_SCOPE

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def _isotropic_state(squeezing_db: float) -> NoiseTransferState:
    variance = squeezing_db_to_peak_variance(
        squeezing_db, coordinate_chart="decoder_standardized"
    )
    return NoiseTransferState(
        fluctuation_covariance=((variance, 0.0), (0.0, variance))
    )


def _isotropic_config(
    squeezing_db: float,
    *,
    measurement_efficiency: float = 0.97,
    loss_transmissivity: float = 0.99,
    gain: float = 1.0,
) -> NoiseTransferConfig:
    variance = squeezing_db_to_peak_variance(
        squeezing_db, coordinate_chart="decoder_standardized"
    )
    return NoiseTransferConfig(
        resource_covariance=((variance, 0.0), (0.0, variance)),
        loss_transmissivity=loss_transmissivity,
        measurement_efficiency=measurement_efficiency,
        feedforward_gain=((gain, 0.0), (0.0, gain)),
    )


def run_noise_transfer_validation(
    *,
    monte_carlo_samples: int = 200_000,
    seed: int = 2380,
) -> NoiseTransferValidationResult:
    if not isinstance(monte_carlo_samples, (int, np.integer)) or int(monte_carlo_samples) < 50_000:
        raise ValueError("monte_carlo_samples must be an integer >= 50000")
    sweep = []
    results: dict[float, NoiseTransferStepResult] = {}
    for db in (3.0, 5.0, 8.0, 10.0, 12.0):
        result = GKPNoiseTransferSurrogate(_isotropic_config(db)).propagate(
            _isotropic_state(db)
        )
        results[db] = result
        axis = result.axis_diagnostics[0].alias_statistics
        sweep.append(
            NoiseTransferSweepPoint(
                squeezing_db=db,
                decision_variance=axis.variance,
                central_probability=axis.central_probability,
                odd_alias_probability=axis.odd_alias_probability,
                clipping_ratio=axis.clipping_ratio,
                validity=result.validity,
            )
        )
    alignment = tuple(
        fock_q_variance_alignment(db) for db in (3.0, 10.0, 12.0)
    )

    representative = results[10.0]
    mean = np.asarray(representative.loss_bias)
    covariance = np.asarray(representative.decision_covariance)
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean, covariance, size=int(monte_carlo_samples))
    aliases = np.floor(samples / LATTICE_CONST + 0.5).astype(np.int64)
    empirical_central = np.mean(aliases == 0, axis=0)
    empirical_odd = np.mean(np.mod(np.abs(aliases), 2) == 1, axis=0)
    predicted = []
    empirical = []
    for axis_index in range(2):
        statistics = representative.axis_diagnostics[axis_index].alias_statistics
        predicted.extend([statistics.central_probability, statistics.odd_alias_probability])
        empirical.extend([empirical_central[axis_index], empirical_odd[axis_index]])
    predicted_array = np.asarray(predicted)
    empirical_array = np.asarray(empirical)
    standard_error = np.sqrt(
        np.maximum(predicted_array * (1.0 - predicted_array), 1.0e-15)
        / float(monte_carlo_samples)
    )
    max_z = float(np.max(np.abs(empirical_array - predicted_array) / standard_error))
    empirical_covariance = np.cov(samples, rowvar=False, ddof=0)
    covariance_error = float(
        np.linalg.norm(empirical_covariance - covariance, ord="fro")
        / np.linalg.norm(covariance, ord="fro")
    )

    refresh = GKPNoiseTransferSurrogate(_isotropic_config(10.0, gain=1.0)).propagate(
        NoiseTransferState(
            fluctuation_covariance=((0.4, 0.08), (0.08, 0.3))
        )
    )
    refresh_error = float(
        np.linalg.norm(
            np.asarray(refresh.output_covariance)
            - np.asarray(refresh.measurement_equivalent_covariance),
            ord="fro",
        )
    )
    efficient = GKPNoiseTransferSurrogate(
        _isotropic_config(10.0, measurement_efficiency=1.0)
    ).propagate(_isotropic_state(10.0))
    inefficient = GKPNoiseTransferSurrogate(
        _isotropic_config(10.0, measurement_efficiency=0.75)
    ).propagate(_isotropic_state(10.0))
    efficiency_increase = float(
        np.trace(np.asarray(inefficient.decision_covariance))
        - np.trace(np.asarray(efficient.decision_covariance))
    )
    loss_proxy = GKPNoiseTransferSurrogate(
        _isotropic_config(10.0, loss_transmissivity=0.9)
    )
    near = loss_proxy.propagate(
        NoiseTransferState(
            lattice_index=(1, 0),
            fluctuation_covariance=_isotropic_state(10.0).fluctuation_covariance,
        )
    )
    far = loss_proxy.propagate(
        NoiseTransferState(
            lattice_index=(3, 0),
            fluctuation_covariance=_isotropic_state(10.0).fluctuation_covariance,
        )
    )
    bias_ratio = abs(far.loss_bias[0]) / abs(near.loss_bias[0])
    low = results[3.0].axis_diagnostics[0].alias_statistics
    clipping_gap = low.variance - low.domain_conditioned_variance
    odd_probabilities = [item.odd_alias_probability for item in sweep]
    checks = {
        "squeezing_reduces_logical_jump_monotonically": all(
            odd_probabilities[index] > odd_probabilities[index + 1]
            for index in range(len(odd_probabilities) - 1)
        ),
        "ten_and_twelve_db_are_localized": all(
            results[db].validity == "localized" for db in (10.0, 12.0)
        ),
        "three_db_is_clipping_dominated": results[3.0].validity == "clipping_dominated",
        "high_squeezing_proxy_matches_state_density": all(
            item.maximum_proxy_to_direct_relative_error < 0.03
            for item in alignment[1:]
        ),
        "high_squeezing_fock_matches_state_density": all(
            item.maximum_fock_to_direct_relative_error < 0.08
            for item in alignment[1:]
        ),
        "low_squeezing_exposes_state_dependence": alignment[0].direct_state_relative_spread > 0.2,
        "low_squeezing_exposes_clipping_gap": clipping_gap > 0.02,
        "monte_carlo_alias_rates_match_exact": max_z < 5.0,
        "monte_carlo_decision_covariance_matches": covariance_error < 0.01,
        "unity_gain_refreshes_continuous_noise": refresh_error < 1.0e-12,
        "measurement_inefficiency_increases_variance": efficiency_increase > 0.0,
        "loss_bias_scales_with_lattice_signal": abs(bias_ratio - 3.0) < 1.0e-12,
        "alias_probability_truncation_is_controlled": all(
            item.alias_statistics.truncation_tail_bound < 2.0e-12
            for result in results.values()
            for item in result.axis_diagnostics
        ),
        "diagonal_covariance_has_exact_pauli_distribution": all(
            result.logical_jump.any_jump_probability is not None
            for result in results.values()
        ),
    }
    return NoiseTransferValidationResult(
        squeezing_sweep=tuple(sweep),
        fock_alignment=(alignment[0], alignment[1], alignment[2]),
        monte_carlo_samples=int(monte_carlo_samples),
        monte_carlo_max_z_score=max_z,
        monte_carlo_covariance_relative_error=covariance_error,
        unity_gain_refresh_error=refresh_error,
        inefficient_measurement_variance_increase=efficiency_increase,
        loss_state_bias_ratio=float(bias_ratio),
        low_squeezing_clipping_gap=float(clipping_gap),
        checks=checks,
    )


def write_noise_transfer_validation(
    result: NoiseTransferValidationResult, output: str | Path
) -> Path:
    if not isinstance(result, NoiseTransferValidationResult):
        raise TypeError("result must be a NoiseTransferValidationResult")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["passed"] = result.passed
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monte-carlo-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=2380)
    arguments = parser.parse_args()
    result = run_noise_transfer_validation(
        monte_carlo_samples=arguments.monte_carlo_samples,
        seed=arguments.seed,
    )
    write_noise_transfer_validation(result, arguments.output)
    print(json.dumps({"passed": result.passed, "checks": result.checks}, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "NOISE_TRANSFER_SCOPE",
    "GaussianAliasStatistics",
    "NoiseTransferConfig",
    "NoiseTransferState",
    "NoiseTransferAxisDiagnostic",
    "NoiseTransferLogicalJump",
    "NoiseTransferStepResult",
    "NoiseTransferSample",
    "GKPNoiseTransferSurrogate",
    "NoiseTransferFockAlignmentPoint",
    "NoiseTransferSweepPoint",
    "NoiseTransferValidationResult",
    "squeezing_db_to_peak_variance",
    "projector_delta_from_squeezing_db",
    "gaussian_alias_statistics",
    "fock_q_variance_alignment",
    "run_noise_transfer_validation",
    "write_noise_transfer_validation",
]
