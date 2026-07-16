"""T3.2.1 bounded-episode memory-assisted periodic Bayesian decoder.

The decoder maintains a probability mass function for the cumulative two-axis
displacement on the ``2L x 2L`` logical torus.  A wrapped correlated-Gaussian
transition propagates the state; a wrapped modular-syndrome likelihood updates
it.  The four parity regions of the torus are therefore the four logical
cosets.  Starting from a known episode origin makes all earlier observations
causally useful without exposing the simulator's lattice indices or truth.

This is inspired by the multi-round Bayesian-estimation mechanism of Wan,
Neville and Kolthammer (arXiv:1912.00829v3), but it is deliberately not called
an exact reproduction of their finite-energy Glancy-Knill circuit.  The local
validation model is a syndrome-level Gaussian random walk with no intermediate
correction, a bounded history episode, and a final logical-coset decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from math import ceil, isfinite, log2, pi, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

from physics.constants import LATTICE_CONST
from physics.syndrome_stream import ObservedSyndromeStep


MEMORY_BAYES_ID = "periodic_memory_assisted_bayes"
STATIC_FINAL_BAYES_ID = "final_outcome_static_periodic_bayes"
TRUTH_REFERENCE_ID = "full_episode_logical_truth_reference"
MODEL_SCOPE = (
    "bounded_episode_correlated_gaussian_periodic_bayes_"
    "not_wan_finite_energy_circuit_reproduction"
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return _finite(values[0], f"{name}[0]"), _finite(values[1], f"{name}[1]")


def _covariance(values: ArrayLike, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 2x2 matrix")
    scale = max(float(np.max(np.abs(matrix))), np.finfo(np.float64).tiny)
    if not np.allclose(matrix, matrix.T, rtol=1.0e-12, atol=1.0e-14 * scale):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be strictly positive definite") from exc
    return matrix


def _wrap(values: NDArray[np.float64], period: float) -> NDArray[np.float64]:
    return np.mod(values + 0.5 * period, period) - 0.5 * period


@dataclass(frozen=True)
class BayesianObservationBudget:
    history_cycles: int = 20
    consumed_per_cycle_fields: tuple[str, ...] = ("residual_q", "residual_p")
    available_per_cycle_fields: tuple[str, ...] = (
        "analog_q",
        "analog_p",
        "residual_q",
        "residual_p",
        "syndrome_x",
        "syndrome_z",
        "phase_x_rad",
        "phase_z_rad",
        "x_e_run",
        "z_e_run",
        "leakage_run",
        "valid",
    )
    episode_start_state: str = "known_zero_logical_torus_origin"
    action_timing: str = "one_final_logical_coset_decision_after_history_episode"
    hidden_truth_inputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "history_cycles",
            _integer(self.history_cycles, "history_cycles", 2),
        )
        if not self.consumed_per_cycle_fields:
            raise ValueError("consumed_per_cycle_fields must not be empty")
        if not set(self.consumed_per_cycle_fields).issubset(
            self.available_per_cycle_fields
        ):
            raise ValueError("consumed fields must be a subset of available fields")
        if self.hidden_truth_inputs:
            raise ValueError("Bayesian observation budget must not expose hidden truth")


@dataclass(frozen=True)
class MemoryBayesianDescriptor:
    baseline_id: str = MEMORY_BAYES_ID
    label: str = "Bounded-episode periodic memory-assisted Bayesian decoder"
    task_owner: str = "T3.2.1"
    comparison_role: str = "deployable_history_aware_logical_decoder_baseline"
    deployable_algorithm: bool = True
    paper_inspiration: str = "Wan et al., arXiv:1912.00829v3, multi-round Bayesian estimation"
    exact_paper_reproduction: bool = False
    state_rule: str = (
        "predict and update a joint 2D posterior on the 2L logical torus; "
        "logical-class probabilities are posterior masses of four parity regions"
    )
    evidence_scope: str = MODEL_SCOPE
    excluded_claims: tuple[str, ...] = (
        "finite_energy_Glancy_Knill_circuit_fidelity_reproduction",
        "device_calibrated_decoder",
        "FPGA_synthesis_or_latency_measurement",
    )


MEMORY_BAYES_DESCRIPTOR = MemoryBayesianDescriptor()


@dataclass(frozen=True)
class PeriodicBayesConfig:
    lattice: float = LATTICE_CONST
    grid_size: int = 128
    process_mean: tuple[float, float] = (0.0, 0.0)
    process_covariance: tuple[tuple[float, float], tuple[float, float]] = (
        (0.20 * LATTICE_CONST * 0.20 * LATTICE_CONST, 0.0),
        (0.0, 0.20 * LATTICE_CONST * 0.20 * LATTICE_CONST),
    )
    measurement_covariance: tuple[tuple[float, float], tuple[float, float]] = (
        (0.10 * LATTICE_CONST * 0.10 * LATTICE_CONST, 0.0),
        (0.0, 0.10 * LATTICE_CONST * 0.10 * LATTICE_CONST),
    )
    tail_sigma: float = 8.0
    observation_budget: BayesianObservationBudget = BayesianObservationBudget()
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        spacing = _finite(self.lattice, "lattice")
        if spacing <= 0.0:
            raise ValueError("lattice must be positive")
        object.__setattr__(self, "lattice", spacing)
        size = _integer(self.grid_size, "grid_size", 32)
        if size % 2:
            raise ValueError("grid_size must be even")
        if size > 512:
            raise ValueError("grid_size must not exceed 512")
        object.__setattr__(self, "grid_size", size)
        object.__setattr__(self, "process_mean", _pair(self.process_mean, "process_mean"))
        process = _covariance(self.process_covariance, "process_covariance")
        measurement = _covariance(
            self.measurement_covariance, "measurement_covariance"
        )
        object.__setattr__(
            self,
            "process_covariance",
            tuple(tuple(float(value) for value in row) for row in process),
        )
        object.__setattr__(
            self,
            "measurement_covariance",
            tuple(tuple(float(value) for value in row) for row in measurement),
        )
        tail = _finite(self.tail_sigma, "tail_sigma")
        if tail < 5.0:
            raise ValueError("tail_sigma must be at least 5")
        object.__setattr__(self, "tail_sigma", tail)
        if not isinstance(self.observation_budget, BayesianObservationBudget):
            raise TypeError("observation_budget must be a BayesianObservationBudget")
        grid_step = 2.0 * spacing / size
        for name, covariance in (
            ("process", process),
            ("measurement", measurement),
        ):
            minimum_sigma = sqrt(float(np.min(np.linalg.eigvalsh(covariance))))
            if minimum_sigma < 0.45 * grid_step:
                raise ValueError(
                    f"{name} covariance is under-resolved by grid_size; increase noise or grid_size"
                )
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")


@dataclass(frozen=True)
class BayesianBatchResult:
    logical_class: NDArray[np.int64]
    logical_posterior: NDArray[np.float64]
    residual_circular_mean: NDArray[np.float64]
    residual_resultant_length: NDArray[np.float64]
    posterior_entropy_bits: NDArray[np.float64]
    posterior_mass: NDArray[np.float64]
    cycles_consumed: int


@dataclass(frozen=True)
class BayesianCostProfile:
    grid_size: int
    logical_torus_cells: int
    history_cycles: int
    raw_history_value_bits: int
    raw_history_storage_bits: int
    posterior_storage_bits: int
    transition_kernel_storage_bits: int
    real_fft_points_per_cycle: int
    complex_fft_butterfly_proxy_per_cycle: int
    transition_kernel_quadratic_forms_once: int
    likelihood_template_quadratic_forms_once: int
    likelihood_observation_quantizations_per_cycle: int
    likelihood_table_lookups_per_cycle: int
    logical_mass_accumulations_per_decision: int
    target_lut: int | None = None
    target_ff: int | None = None
    target_bram: int | None = None
    target_dsp: int | None = None
    target_fmax_hz: float | None = None
    target_measured: bool = False
    scope: str = "deterministic_grid_operation_storage_proxy_not_synthesis"


def bayesian_cost_profile(
    config: PeriodicBayesConfig,
    *,
    value_bits: int = 24,
) -> BayesianCostProfile:
    if not isinstance(config, PeriodicBayesConfig):
        raise TypeError("config must be a PeriodicBayesConfig")
    bits = _integer(value_bits, "value_bits", 8)
    cells = config.grid_size**2
    fft_proxy = int(2 * cells * ceil(log2(cells)))
    history = config.observation_budget.history_cycles
    process_aliases = (2 * _periodic_alias_radius(
        np.asarray(config.process_covariance, dtype=np.float64),
        period=2.0 * config.lattice,
        tail_sigma=config.tail_sigma,
    ) + 1) ** 2
    measurement_aliases = (2 * _periodic_alias_radius(
        np.asarray(config.measurement_covariance, dtype=np.float64),
        period=config.lattice,
        tail_sigma=config.tail_sigma,
    ) + 1) ** 2
    return BayesianCostProfile(
        grid_size=config.grid_size,
        logical_torus_cells=cells,
        history_cycles=history,
        raw_history_value_bits=bits,
        raw_history_storage_bits=history * 2 * bits,
        posterior_storage_bits=cells * bits,
        transition_kernel_storage_bits=cells * bits,
        real_fft_points_per_cycle=2 * cells,
        complex_fft_butterfly_proxy_per_cycle=fft_proxy,
        transition_kernel_quadratic_forms_once=cells * process_aliases,
        likelihood_template_quadratic_forms_once=cells * measurement_aliases,
        likelihood_observation_quantizations_per_cycle=2,
        likelihood_table_lookups_per_cycle=cells,
        logical_mass_accumulations_per_decision=cells - 4,
    )


def _periodic_alias_radius(
    covariance: NDArray[np.float64],
    *,
    period: float,
    tail_sigma: float,
) -> int:
    sigma_max = sqrt(float(np.max(np.linalg.eigvalsh(covariance))))
    # ``delta`` is first wrapped to half a period.  Shift r+1 is therefore at
    # least (r+1/2) periods away, so this is the smallest conservative radius
    # that covers ``tail_sigma`` standard deviations.  Radius one also keeps
    # the nearest periodic images for narrow kernels.
    return max(1, int(ceil(tail_sigma * sigma_max / period - 0.5)))


def _periodic_gaussian_values(
    delta_q: NDArray[np.float64],
    delta_p: NDArray[np.float64],
    covariance: NDArray[np.float64],
    *,
    period: float,
    tail_sigma: float,
) -> NDArray[np.float64]:
    inverse = np.linalg.inv(covariance)
    radius = _periodic_alias_radius(
        covariance,
        period=period,
        tail_sigma=tail_sigma,
    )
    wrapped_q = _wrap(delta_q, period)
    wrapped_p = _wrap(delta_p, period)
    result = np.zeros(np.broadcast_shapes(delta_q.shape, delta_p.shape), dtype=np.float64)
    for shift_q in range(-radius, radius + 1):
        q = wrapped_q + shift_q * period
        for shift_p in range(-radius, radius + 1):
            p = wrapped_p + shift_p * period
            quadratic = (
                inverse[0, 0] * q * q
                + 2.0 * inverse[0, 1] * q * p
                + inverse[1, 1] * p * p
            )
            result += np.exp(-0.5 * quadratic)
    return result


class _PeriodicBayesGrid:
    def __init__(self, config: PeriodicBayesConfig) -> None:
        self.config = config
        self.size = config.grid_size
        self.lattice = config.lattice
        self.period = 2.0 * self.lattice
        indices = np.arange(self.size, dtype=np.float64)
        self.grid_step = self.period / self.size
        state_coordinates = (indices + 0.5) * self.grid_step
        state_coordinates = np.where(
            state_coordinates < self.lattice,
            state_coordinates,
            state_coordinates - self.period,
        )
        delta_coordinates = indices * self.grid_step
        delta_coordinates = np.where(
            delta_coordinates < self.lattice,
            delta_coordinates,
            delta_coordinates - self.period,
        )
        self.coordinates = state_coordinates
        self.q_grid, self.p_grid = np.meshgrid(
            state_coordinates, state_coordinates, indexing="ij"
        )
        delta_q, delta_p = np.meshgrid(
            delta_coordinates, delta_coordinates, indexing="ij"
        )
        covariance = np.asarray(config.process_covariance, dtype=np.float64)
        mean = np.asarray(config.process_mean, dtype=np.float64)
        kernel = _periodic_gaussian_values(
            delta_q - mean[0],
            delta_p - mean[1],
            covariance,
            period=self.period,
            tail_sigma=config.tail_sigma,
        )
        kernel_sum = float(np.sum(kernel))
        if not isfinite(kernel_sum) or kernel_sum <= 0.0:
            raise RuntimeError("transition kernel has no finite positive mass")
        self.transition_kernel = kernel / kernel_sum
        self.transition_fft = np.fft.fft2(self.transition_kernel)
        measurement = _periodic_gaussian_values(
            -self.q_grid,
            -self.p_grid,
            np.asarray(config.measurement_covariance, dtype=np.float64),
            period=self.lattice,
            tail_sigma=config.tail_sigma,
        )
        measurement_max = float(np.max(measurement))
        if not isfinite(measurement_max) or measurement_max <= 0.0:
            raise RuntimeError("measurement likelihood template is numerically empty")
        self.measurement_template = measurement / measurement_max
        parity_q = np.mod(
            np.floor(state_coordinates / self.lattice + 0.5).astype(np.int64), 2
        )
        parity_p = parity_q.copy()
        self.logical_class_grid = 2 * parity_q[:, None] + parity_p[None, :]

    def initial_mass(self, batch: int) -> NDArray[np.float64]:
        result = np.zeros((batch, self.size, self.size), dtype=np.float64)
        # The known zero origin lies at the intersection of four cell-centred
        # bins.  Bilinear splitting preserves q/p and sign symmetry instead of
        # introducing a grid-resolution-dependent half-cell bias.
        for q_index in (0, self.size - 1):
            for p_index in (0, self.size - 1):
                result[:, q_index, p_index] = 0.25
        return result

    def predict(self, posterior: NDArray[np.float64]) -> NDArray[np.float64]:
        transformed = np.fft.fft2(posterior, axes=(-2, -1))
        predicted = np.fft.ifft2(
            transformed * self.transition_fft[None, :, :], axes=(-2, -1)
        ).real
        predicted = np.maximum(predicted, 0.0)
        totals = np.sum(predicted, axis=(-2, -1), keepdims=True)
        if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
            raise RuntimeError("Bayesian prediction lost probability mass")
        return predicted / totals

    def likelihood(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        # The latent state already lives on a finite grid.  Quantizing the
        # observed residual to that same grid makes every likelihood a cyclic
        # shift of one precomputed wrapped-Gaussian template, avoiding a large
        # transcendental recomputation at each cycle.  The maximum input
        # quantization error is explicitly bounded by grid_step/2 and is
        # audited against a finer reference grid in the production harness.
        shifts = np.floor(observations / self.grid_step + 0.5).astype(np.int64)
        indices = np.arange(self.size, dtype=np.int64)
        q_index = np.mod(indices[None, :, None] - shifts[:, 0, None, None], self.size)
        p_index = np.mod(indices[None, None, :] - shifts[:, 1, None, None], self.size)
        return self.measurement_template[q_index, p_index]

    def update(
        self,
        prior: NDArray[np.float64],
        observations: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        weighted = prior * self.likelihood(observations)
        evidence = np.sum(weighted, axis=(-2, -1), keepdims=True)
        if np.any(~np.isfinite(evidence)) or np.any(evidence <= 0.0):
            raise RuntimeError("Bayesian update has zero or non-finite evidence")
        return weighted / evidence

    def summarize(
        self,
        posterior: NDArray[np.float64],
        cycles: int,
    ) -> BayesianBatchResult:
        logical = np.stack(
            [
                np.sum(posterior[:, self.logical_class_grid == label], axis=-1)
                for label in range(4)
            ],
            axis=-1,
        )
        logical /= np.sum(logical, axis=-1, keepdims=True)
        decision = np.argmax(logical, axis=-1).astype(np.int64)
        residual = _wrap(self.coordinates, self.lattice)
        angle = 2.0 * pi * residual / self.lattice
        marginal_q = np.sum(posterior, axis=-1)
        marginal_p = np.sum(posterior, axis=-2)
        means: list[NDArray[np.float64]] = []
        lengths: list[NDArray[np.float64]] = []
        for marginal in (marginal_q, marginal_p):
            cosine = marginal @ np.cos(angle)
            sine = marginal @ np.sin(angle)
            lengths.append(np.sqrt(cosine * cosine + sine * sine))
            means.append(np.arctan2(sine, cosine) * self.lattice / (2.0 * pi))
        safe = np.clip(posterior, np.finfo(np.float64).tiny, 1.0)
        entropy = -np.sum(posterior * np.log2(safe), axis=(-2, -1))
        return BayesianBatchResult(
            logical_class=decision,
            logical_posterior=logical,
            residual_circular_mean=np.stack(means, axis=-1),
            residual_resultant_length=np.stack(lengths, axis=-1),
            posterior_entropy_bits=entropy,
            posterior_mass=posterior,
            cycles_consumed=cycles,
        )


def _validate_history(
    residual_history: ArrayLike,
    config: PeriodicBayesConfig,
) -> NDArray[np.float64]:
    values = np.asarray(residual_history, dtype=np.float64)
    if values.ndim == 2 and values.shape[-1] == 2:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("residual_history must have shape (cycles,2) or (batch,cycles,2)")
    if not np.all(np.isfinite(values)):
        raise ValueError("residual_history must contain only finite values")
    cycles = values.shape[1]
    if cycles < 1:
        raise ValueError("residual_history must contain at least one cycle")
    if cycles > config.observation_budget.history_cycles:
        raise ValueError("residual_history exceeds the registered history budget")
    half = config.lattice / 2.0
    if np.any(values < -half) or np.any(values >= half):
        raise ValueError(
            "residual_history must lie in the half-open interval [-lattice/2,lattice/2)"
        )
    return values


def periodic_memory_bayes_decode(
    residual_history: ArrayLike,
    config: PeriodicBayesConfig | None = None,
) -> BayesianBatchResult:
    actual = PeriodicBayesConfig() if config is None else config
    if not isinstance(actual, PeriodicBayesConfig):
        raise TypeError("config must be a PeriodicBayesConfig or None")
    history = _validate_history(residual_history, actual)
    grid = _PeriodicBayesGrid(actual)
    posterior = grid.initial_mass(history.shape[0])
    for cycle in range(history.shape[1]):
        posterior = grid.update(grid.predict(posterior), history[:, cycle, :])
    return grid.summarize(posterior, history.shape[1])


def final_outcome_static_bayes_decode(
    final_residual: ArrayLike,
    rounds: int,
    config: PeriodicBayesConfig | None = None,
) -> BayesianBatchResult:
    actual = PeriodicBayesConfig() if config is None else config
    if not isinstance(actual, PeriodicBayesConfig):
        raise TypeError("config must be a PeriodicBayesConfig or None")
    count = _integer(rounds, "rounds", 1)
    if count > actual.observation_budget.history_cycles:
        raise ValueError("rounds exceeds the registered history budget")
    values = np.asarray(final_residual, dtype=np.float64)
    if values.ndim == 1 and values.shape == (2,):
        values = values[None, :]
    if values.ndim != 2 or values.shape[-1] != 2:
        raise ValueError("final_residual must have shape (2,) or (batch,2)")
    _validate_history(values[:, None, :], actual)
    grid = _PeriodicBayesGrid(actual)
    posterior = grid.initial_mass(values.shape[0])
    # This is the exact same discretized H-step transition prior as the memory
    # decoder, but only the final observation is made available.
    spectrum = np.fft.fft2(posterior, axes=(-2, -1))
    prior = np.fft.ifft2(
        spectrum * (grid.transition_fft[None, :, :] ** count), axes=(-2, -1)
    ).real
    prior = np.maximum(prior, 0.0)
    prior /= np.sum(prior, axis=(-2, -1), keepdims=True)
    posterior = grid.update(prior, values)
    return grid.summarize(posterior, count)


def decode_observed_episode(
    observations: Sequence[ObservedSyndromeStep],
    config: PeriodicBayesConfig | None = None,
) -> BayesianBatchResult:
    actual = PeriodicBayesConfig() if config is None else config
    if not isinstance(actual, PeriodicBayesConfig):
        raise TypeError("config must be a PeriodicBayesConfig or None")
    if isinstance(observations, (str, bytes)):
        raise TypeError("observations must be a sequence of ObservedSyndromeStep")
    items = tuple(observations)
    if not items:
        raise ValueError("observations must not be empty")
    if any(not isinstance(item, ObservedSyndromeStep) for item in items):
        raise TypeError("every observation must be an ObservedSyndromeStep")
    start = items[0].cycle_index
    for offset, item in enumerate(items):
        if item.cycle_index != start + offset:
            raise ValueError("observation cycle indices must be consecutive")
        if not item.valid:
            raise ValueError("invalid observations are not accepted by this baseline")
        analog = np.asarray(item.analog_syndrome, dtype=np.float64)
        residual = np.asarray(item.residual_syndrome, dtype=np.float64)
        if not np.allclose(_wrap(analog, actual.lattice), residual, atol=1.0e-10):
            raise ValueError("analog_syndrome does not wrap to residual_syndrome")
    history = np.asarray([item.residual_syndrome for item in items], dtype=np.float64)
    return periodic_memory_bayes_decode(history, actual)


@dataclass(frozen=True)
class BayesianValidationScenario:
    scenario_id: str
    mean_lattice: tuple[float, float]
    sigma_q_lattice: float
    sigma_p_lattice: float
    rho: float
    measurement_sigma_lattice: float

    def config(
        self,
        *,
        grid_size: int,
        history_cycles: int,
    ) -> PeriodicBayesConfig:
        mean = tuple(value * LATTICE_CONST for value in self.mean_lattice)
        sigma_q = self.sigma_q_lattice * LATTICE_CONST
        sigma_p = self.sigma_p_lattice * LATTICE_CONST
        covariance = (
            (sigma_q**2, self.rho * sigma_q * sigma_p),
            (self.rho * sigma_q * sigma_p, sigma_p**2),
        )
        measurement_variance = (self.measurement_sigma_lattice * LATTICE_CONST) ** 2
        return PeriodicBayesConfig(
            grid_size=grid_size,
            process_mean=mean,
            process_covariance=covariance,
            measurement_covariance=(
                (measurement_variance, 0.0),
                (0.0, measurement_variance),
            ),
            observation_budget=BayesianObservationBudget(history_cycles=history_cycles),
        )


def bayesian_validation_scenarios() -> tuple[BayesianValidationScenario, ...]:
    return (
        BayesianValidationScenario("quiet_isotropic", (0.0, 0.0), 0.12, 0.12, 0.0, 0.08),
        BayesianValidationScenario("measurement_limited", (0.0, 0.0), 0.16, 0.16, 0.0, 0.16),
        BayesianValidationScenario("correlated", (0.0, 0.0), 0.15, 0.20, 0.70, 0.10),
        BayesianValidationScenario("biased_correlated", (0.045, -0.035), 0.17, 0.13, -0.55, 0.10),
    )


@dataclass(frozen=True)
class BayesianValidationConfig:
    evaluation_seeds: tuple[int, ...] = tuple(range(20260761, 20260769))
    episodes_per_seed: int = 128
    history_cycles: int = 20
    grid_size: int = 128
    reference_grid_size: int = 256
    reference_episodes_per_seed: int = 64
    confidence_level: float = 0.95
    batch_size: int = 64

    def __post_init__(self) -> None:
        seeds = tuple(self.evaluation_seeds)
        if len(seeds) < 6 or len(set(seeds)) != len(seeds):
            raise ValueError("evaluation_seeds must contain at least six unique values")
        if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in seeds):
            raise TypeError("evaluation seeds must be integers")
        if any(int(seed) < 0 or int(seed) >= 2**64 for seed in seeds):
            raise ValueError("evaluation seeds must lie in [0,2**64)")
        for name, minimum in (
            ("episodes_per_seed", 128),
            ("history_cycles", 8),
            ("grid_size", 32),
            ("reference_grid_size", 32),
            ("reference_episodes_per_seed", 16),
            ("batch_size", 1),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.grid_size % 2 or self.reference_grid_size % 2:
            raise ValueError("validation grid sizes must be even")
        if self.reference_grid_size <= self.grid_size:
            raise ValueError("reference_grid_size must exceed grid_size")
        if self.reference_episodes_per_seed > self.episodes_per_seed:
            raise ValueError("reference episodes must not exceed production episodes")
        confidence = _finite(self.confidence_level, "confidence_level")
        if not 0.5 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0.5,1)")
        object.__setattr__(self, "confidence_level", confidence)


def _logical_classes(total: NDArray[np.float64], lattice: float) -> NDArray[np.int64]:
    indices = np.floor(total / lattice + 0.5).astype(np.int64)
    parity = np.mod(indices, 2)
    return (2 * parity[..., 0] + parity[..., 1]).astype(np.int64)


def _simulate_episodes(
    scenario: BayesianValidationScenario,
    seed: int,
    episodes: int,
    cycles: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], str]:
    rng = np.random.default_rng(seed)
    process_config = scenario.config(grid_size=64, history_cycles=cycles)
    increments = rng.multivariate_normal(
        np.asarray(process_config.process_mean),
        np.asarray(process_config.process_covariance),
        size=(episodes, cycles),
    )
    cumulative = np.cumsum(increments, axis=1)
    measurement = rng.multivariate_normal(
        np.zeros(2),
        np.asarray(process_config.measurement_covariance),
        size=(episodes, cycles),
    )
    residual = _wrap(cumulative + measurement, process_config.lattice)
    truth = _logical_classes(cumulative[:, -1, :], process_config.lattice)
    digest = hashlib.sha256()
    digest.update(np.asarray(residual, dtype="<f8").tobytes())
    digest.update(np.asarray(truth, dtype="<i8").tobytes())
    return residual, truth, digest.hexdigest()


def _decode_in_batches(
    history: NDArray[np.float64],
    config: PeriodicBayesConfig,
    batch_size: int,
    *,
    memory: bool,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    decisions: list[NDArray[np.int64]] = []
    posteriors: list[NDArray[np.float64]] = []
    for start in range(0, history.shape[0], batch_size):
        batch = history[start : start + batch_size]
        if memory:
            result = periodic_memory_bayes_decode(batch, config)
        else:
            result = final_outcome_static_bayes_decode(
                batch[:, -1, :], batch.shape[1], config
            )
        decisions.append(result.logical_class)
        posteriors.append(result.logical_posterior)
    return np.concatenate(decisions), np.concatenate(posteriors)


def _proper_scores(
    posterior: NDArray[np.float64], truth: NDArray[np.int64]
) -> tuple[float, float]:
    selected = posterior[np.arange(truth.size), truth]
    nll = float(np.mean(-np.log(np.clip(selected, np.finfo(float).tiny, 1.0))))
    one_hot = np.eye(4, dtype=np.float64)[truth]
    brier = float(np.mean(np.sum((posterior - one_hot) ** 2, axis=-1)))
    return nll, brier


def _mean_interval(values: Sequence[float], confidence: float) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    estimate = float(np.mean(array))
    if array.size < 2:
        standard_error = 0.0
        critical = 0.0
        degrees_freedom = 0
    else:
        standard_error = float(np.std(array, ddof=1) / sqrt(array.size))
        degrees_freedom = int(array.size - 1)
        critical = float(student_t.ppf(0.5 + confidence / 2.0, degrees_freedom))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - critical * standard_error,
        "ci_high": estimate + critical * standard_error,
        "cluster_unit": "evaluation_seed",
        "interval_method": "two_sided_student_t_cluster_mean",
        "degrees_freedom": degrees_freedom,
    }


def validate_memory_bayesian_registration() -> tuple[str, ...]:
    from cnn_fpga.benchmark.standard_binning_baseline import (
        major_comparison_registry,
        validate_major_comparison_registry,
    )

    gates = validate_major_comparison_registry()
    match = [
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t3_2_1_memory_bayesian_episode_comparison"
    ]
    if len(match) != 1:
        raise ValueError("T3.2.1 comparison must be registered exactly once")
    entry = match[0]
    expected = (
        "standard_binning",
        STATIC_FINAL_BAYES_ID,
        MEMORY_BAYES_ID,
        TRUTH_REFERENCE_ID,
    )
    if entry.method_ids != expected or entry.standard_binning_policy != "required":
        raise ValueError("T3.2.1 comparison registration drifted from its method contract")
    return gates


def _implementation_sha256() -> str:
    paths = (
        Path(__file__),
        Path(__file__).with_name("standard_binning_baseline.py"),
        Path(__file__).parents[2] / "physics" / "syndrome_stream.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_memory_bayesian_validation(
    config: BayesianValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = BayesianValidationConfig() if config is None else config
    if not isinstance(settings, BayesianValidationConfig):
        raise TypeError("config must be a BayesianValidationConfig or None")
    registry_gates = validate_memory_bayesian_registration()
    scenario_payloads: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    all_seed_gains: list[float] = []
    trace_hashes: list[str] = []
    grid_tvs: list[float] = []
    grid_disagreements: list[float] = []
    grid_error_rate_deltas: list[float] = []
    for scenario_index, scenario in enumerate(bayesian_validation_scenarios()):
        production_config = scenario.config(
            grid_size=settings.grid_size,
            history_cycles=settings.history_cycles,
        )
        reference_config = scenario.config(
            grid_size=settings.reference_grid_size,
            history_cycles=settings.history_cycles,
        )
        seed_records: list[dict[str, object]] = []
        for seed_index, seed in enumerate(settings.evaluation_seeds):
            derived_seed = int(seed + 10_000 * scenario_index)
            history, truth, trace_hash = _simulate_episodes(
                scenario,
                derived_seed,
                settings.episodes_per_seed,
                settings.history_cycles,
            )
            memory_decision, memory_posterior = _decode_in_batches(
                history,
                production_config,
                settings.batch_size,
                memory=True,
            )
            static_decision, static_posterior = _decode_in_batches(
                history,
                production_config,
                settings.batch_size,
                memory=False,
            )
            standard_decision = np.zeros_like(truth)
            memory_error = float(np.mean(memory_decision != truth))
            static_error = float(np.mean(static_decision != truth))
            standard_error = float(np.mean(standard_decision != truth))
            memory_nll, memory_brier = _proper_scores(memory_posterior, truth)
            static_nll, static_brier = _proper_scores(static_posterior, truth)
            reference_evaluated = seed_index == 0
            if reference_evaluated:
                subset = history[: settings.reference_episodes_per_seed]
                subset_truth = truth[: settings.reference_episodes_per_seed]
                reference_result = periodic_memory_bayes_decode(subset, reference_config)
                production_subset = periodic_memory_bayes_decode(subset, production_config)
                tv: float | None = float(
                    np.mean(
                        0.5
                        * np.sum(
                            np.abs(
                                reference_result.logical_posterior
                                - production_subset.logical_posterior
                            ),
                            axis=-1,
                        )
                    )
                )
                disagreement: float | None = float(
                    np.mean(
                        reference_result.logical_class
                        != production_subset.logical_class
                    )
                )
                grid_error_delta: float | None = abs(
                    float(np.mean(reference_result.logical_class != subset_truth))
                    - float(np.mean(production_subset.logical_class != subset_truth))
                )
                grid_tvs.append(tv)
                grid_disagreements.append(disagreement)
                grid_error_rate_deltas.append(grid_error_delta)
            else:
                tv = None
                disagreement = None
                grid_error_delta = None
            gain = static_error - memory_error
            all_seed_gains.append(gain)
            trace_hashes.append(trace_hash)
            record = {
                "scenario_id": scenario.scenario_id,
                "evaluation_seed": derived_seed,
                "episodes": settings.episodes_per_seed,
                "history_cycles": settings.history_cycles,
                "trace_sha256": trace_hash,
                "standard_error_rate": standard_error,
                "static_final_error_rate": static_error,
                "memory_bayesian_error_rate": memory_error,
                "static_minus_memory_error_rate": gain,
                "static_nll": static_nll,
                "memory_nll": memory_nll,
                "static_brier": static_brier,
                "memory_brier": memory_brier,
                "grid_reference_evaluated": reference_evaluated,
                "grid_reference_tv_mean": tv,
                "grid_reference_decision_disagreement": disagreement,
                "grid_reference_error_rate_delta_abs": grid_error_delta,
            }
            seed_records.append(record)
            source_rows.append(record)
        gains = [float(record["static_minus_memory_error_rate"]) for record in seed_records]
        scenario_payloads.append(
            {
                "scenario": asdict(scenario),
                "episodes": settings.episodes_per_seed * len(settings.evaluation_seeds),
                "cycles": (
                    settings.episodes_per_seed
                    * len(settings.evaluation_seeds)
                    * settings.history_cycles
                ),
                "unique_trace_hashes": len(
                    {str(record["trace_sha256"]) for record in seed_records}
                ),
                "standard_error_rate": float(
                    np.mean([float(record["standard_error_rate"]) for record in seed_records])
                ),
                "static_final_error_rate": float(
                    np.mean([float(record["static_final_error_rate"]) for record in seed_records])
                ),
                "memory_bayesian_error_rate": float(
                    np.mean(
                        [float(record["memory_bayesian_error_rate"]) for record in seed_records]
                    )
                ),
                "static_minus_memory_seed_cluster_ci": _mean_interval(
                    gains, settings.confidence_level
                ),
                "memory_nll": float(
                    np.mean([float(record["memory_nll"]) for record in seed_records])
                ),
                "static_nll": float(
                    np.mean([float(record["static_nll"]) for record in seed_records])
                ),
                "memory_brier": float(
                    np.mean([float(record["memory_brier"]) for record in seed_records])
                ),
                "static_brier": float(
                    np.mean([float(record["static_brier"]) for record in seed_records])
                ),
                "grid_reference_tv_mean_max_seed": max(
                    float(record["grid_reference_tv_mean"])
                    for record in seed_records
                    if record["grid_reference_evaluated"]
                ),
                "grid_reference_decision_disagreement_max_seed": max(
                    float(record["grid_reference_decision_disagreement"])
                    for record in seed_records
                    if record["grid_reference_evaluated"]
                ),
                "grid_reference_error_rate_delta_abs_max_seed": max(
                    float(record["grid_reference_error_rate_delta_abs"])
                    for record in seed_records
                    if record["grid_reference_evaluated"]
                ),
            }
        )
    gates = {
        "descriptor_has_no_hidden_truth_input": (
            not MEMORY_BAYES_DESCRIPTOR.exact_paper_reproduction
            and not BayesianObservationBudget().hidden_truth_inputs
        ),
        "history_and_observation_budget_are_explicit": all(
            row["history_cycles"] == settings.history_cycles for row in source_rows
        ),
        "comparison_registry_requires_standard_anchor": any(
            gate == "registry:t3_2_1_memory_bayesian_episode_comparison"
            for gate in registry_gates
        ),
        "all_traces_are_unique": len(set(trace_hashes)) == len(trace_hashes),
        "memory_gain_resolved_in_every_scenario": all(
            item["static_minus_memory_seed_cluster_ci"]["ci_low"] > 0.0
            for item in scenario_payloads
        ),
        "memory_improves_both_proper_scores_in_every_scenario": all(
            item["memory_nll"] < item["static_nll"]
            and item["memory_brier"] < item["static_brier"]
            for item in scenario_payloads
        ),
        "aggregate_memory_gain_resolved": _mean_interval(
            all_seed_gains, settings.confidence_level
        )["ci_low"]
        > 0.0,
        "production_grid_matches_finer_reference": (
            max(grid_tvs) <= 0.025 and max(grid_error_rate_deltas) <= 0.025
        ),
        "cost_profile_remains_not_synthesis": (
            not bayesian_cost_profile(
                bayesian_validation_scenarios()[0].config(
                    grid_size=settings.grid_size,
                    history_cycles=settings.history_cycles,
                )
            ).target_measured
        ),
    }
    failures = [name for name, passed in gates.items() if not passed]
    payload = {
        "schema_version": "t3.2.1-memory-assisted-periodic-bayes-v1",
        "task_id": "T3.2.1",
        "status": "PASS" if not failures else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "descriptor": asdict(MEMORY_BAYES_DESCRIPTOR),
        "observation_budget": asdict(
            BayesianObservationBudget(history_cycles=settings.history_cycles)
        ),
        "paper_provenance": {
            "title": "Memory-assisted decoder for approximate Gottesman-Kitaev-Preskill codes",
            "authors": "Wan, Neville, Kolthammer",
            "doi": "10.1103/PhysRevResearch.2.043280",
            "arxiv": "1912.00829v3",
            "source_version_used": "open arXiv v3 TeX/PDF",
            "transferred_mechanism": "multi-round Bayesian posterior and final correction",
            "not_transferred": "finite-energy wavefunction, Glancy-Knill circuit, paper fidelity numbers",
        },
        "validation_config": asdict(settings),
        "scenarios": scenario_payloads,
        "aggregate": {
            "scenarios": len(scenario_payloads),
            "evaluation_seeds_per_scenario": len(settings.evaluation_seeds),
            "episodes": len(scenario_payloads)
            * len(settings.evaluation_seeds)
            * settings.episodes_per_seed,
            "cycles": len(scenario_payloads)
            * len(settings.evaluation_seeds)
            * settings.episodes_per_seed
            * settings.history_cycles,
            "source_data_rows": len(source_rows),
            "static_minus_memory_seed_cluster_ci": _mean_interval(
                [
                    float(
                        np.mean(
                            [
                                row["static_minus_memory_error_rate"]
                                for row in source_rows
                                if row["evaluation_seed"] % 10_000
                                == seed % 10_000
                            ]
                        )
                    )
                    for seed in settings.evaluation_seeds
                ],
                settings.confidence_level,
            ),
            "max_grid_reference_tv_mean": max(grid_tvs),
            "max_grid_reference_decision_disagreement": max(grid_disagreements),
            "max_grid_reference_error_rate_delta_abs": max(grid_error_rate_deltas),
        },
        "cost_profile": asdict(
            bayesian_cost_profile(
                bayesian_validation_scenarios()[0].config(
                    grid_size=settings.grid_size,
                    history_cycles=settings.history_cycles,
                )
            )
        ),
        "gate_summary": {
            "passed": len(gates) - len(failures),
            "failed": len(failures),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "within the registered bounded correlated-Gaussian modular-syndrome episodes, "
                "the observed-only periodic Bayesian filter uses the full causal history and "
                "is compared with a final-outcome static Bayesian decoder on the same traces"
            ),
            "forbidden": (
                "exact reproduction of Wan finite-energy circuit fidelity, universal history "
                "gain, device calibration, FPGA synthesis, or real quantum-hardware decoding"
            ),
        },
    }
    return json.loads(json.dumps(payload, ensure_ascii=False)), source_rows


def write_memory_bayesian_validation(
    json_path: str | Path,
    csv_path: str | Path,
    config: BayesianValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_memory_bayesian_validation(config)
    output_json = Path(json_path)
    output_csv = Path(csv_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate the T3.2.1 memory-assisted periodic Bayesian decoder"
    )
    parser.add_argument(
        "--json", default="docs/t3_2_1_memory_bayesian_validation.json"
    )
    parser.add_argument(
        "--csv", default="docs/t3_2_1_memory_bayesian_source_data.csv"
    )
    arguments = parser.parse_args()
    result = write_memory_bayesian_validation(arguments.json, arguments.csv)
    print(json.dumps(result["gate_summary"], ensure_ascii=False))


__all__ = [
    "MEMORY_BAYES_ID",
    "STATIC_FINAL_BAYES_ID",
    "TRUTH_REFERENCE_ID",
    "MODEL_SCOPE",
    "BayesianObservationBudget",
    "MemoryBayesianDescriptor",
    "MEMORY_BAYES_DESCRIPTOR",
    "PeriodicBayesConfig",
    "BayesianBatchResult",
    "BayesianCostProfile",
    "bayesian_cost_profile",
    "periodic_memory_bayes_decode",
    "final_outcome_static_bayes_decode",
    "decode_observed_episode",
    "BayesianValidationScenario",
    "bayesian_validation_scenarios",
    "BayesianValidationConfig",
    "validate_memory_bayesian_registration",
    "build_memory_bayesian_validation",
    "write_memory_bayesian_validation",
]
