"""有限 squeezing 的分解式 syndrome-level effective noise model。

本模块不把所有非理想性压成单个 ``sigma_eff``。一次两正交分量观测写成

``physical = channel + data_peak + envelope``

``observed = physical + ancilla_peak + measurement``。

其中 data/ancilla peak variance 来自 T1.2.1 damped-projector family 的隔离
probability peak variance ``tanh(Delta**2)/2``。finite-energy envelope 不是再次
添加一个随手指定的 Gaussian：它用孤立峰近似下的离散格点权重
``exp[-tanh(Delta**2) * (m*lattice)**2]`` 与 Mehler center contraction
``sech(Delta**2)`` 生成非高斯 shift。该 envelope 项是 effective、logical-state-
averaged comb approximation，不是 Fock-space density-matrix evolution。

``Delta=0`` 被明确定义为 ideal/high-squeezing endpoint；此时 data、ancilla 与
envelope 三项严格为零，而 channel 和 classical measurement 保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import json
from math import ceil, cosh, erfc, exp, isfinite, log, pi, sqrt, tanh
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .constants import LATTICE_CONST
from .quadrature_conventions import QuadratureAxis, QuadratureChartName, chart


EnvelopeIndexClass = Literal["all", "even", "odd"]
_AXES = 2


def _finite_pair(values: object, name: str, *, nonnegative: bool) -> tuple[float, float]:
    try:
        pair = tuple(float(value) for value in values)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two numeric values") from exc
    if len(pair) != _AXES or not all(isfinite(value) for value in pair):
        raise ValueError(f"{name} must contain two finite values")
    if nonnegative and any(value < 0.0 for value in pair):
        raise ValueError(f"{name} values must be non-negative")
    return pair  # type: ignore[return-value]


def _covariance_tuple(values: object, name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    try:
        covariance = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 2x2 covariance") from exc
    if covariance.shape != (_AXES, _AXES) or not np.all(np.isfinite(covariance)):
        raise ValueError(f"{name} must be a finite 2x2 covariance")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-14):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = 1.0e-13 * max(1.0, float(np.max(np.abs(covariance))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    covariance = 0.5 * (covariance + covariance.T)
    covariance[np.abs(covariance) < tolerance] = 0.0
    return (
        (float(covariance[0, 0]), float(covariance[0, 1])),
        (float(covariance[1, 0]), float(covariance[1, 1])),
    )


def _covariance_array(
    values: tuple[tuple[float, float], tuple[float, float]],
) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def _sech_square_delta(delta: float) -> float:
    epsilon = delta * delta
    if epsilon < 350.0:
        return 1.0 / cosh(epsilon)
    decayed = exp(-epsilon)
    return 2.0 * decayed / (1.0 + decayed * decayed)


def isolated_peak_variance(
    delta: float,
    *,
    coordinate_chart: QuadratureChartName = "decoder_standardized",
    axis: QuadratureAxis = "q",
) -> float:
    """返回 damped-projector 隔离 probability peak variance。

    ``Delta=0`` 是 ideal endpoint。非零值使用
    canonical ``[x,p]=i`` 中
    ``sigma_peak^2=tanh(Delta^2)/2``。默认返回 decoder-standardized
    axis 的方差，因此额外乘坐标 scale 的平方；两个 decoder axis 只是独立的
    classical normalization，不应解释为 joint quantum operators。
    """

    value = float(delta)
    if not isfinite(value) or value < 0.0:
        raise ValueError("delta must be finite and non-negative")
    epsilon = value * value
    if not isfinite(epsilon):
        raise ValueError("delta squared must be representable")
    if axis not in {"q", "p"}:
        raise ValueError("axis must be q or p")
    registered = chart(coordinate_chart)
    scale = (
        registered.canonical_scale_q
        if axis == "q"
        else registered.canonical_scale_p
    )
    return scale * scale * 0.5 * tanh(epsilon)


@dataclass(frozen=True)
class EnvelopeIndexDistribution:
    """finite-energy envelope 的离散 lattice-index effective distribution。"""

    delta: float
    lattice: float
    index_class: EnvelopeIndexClass
    indices: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    shifts: NDArray[np.float64]
    contraction: float
    mean_shift: float
    variance: float
    captured_weight: float
    scope: str = "isolated_peak_incoherent_envelope_effective_model"

    def sample(self, size: int, rng: np.random.Generator) -> NDArray[np.float64]:
        if not isinstance(size, int) or size < 1:
            raise ValueError("size must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        choices = rng.choice(self.shifts.size, size=size, p=self.probabilities)
        return self.shifts[choices]


def envelope_index_distribution(
    delta: float,
    *,
    lattice: float = LATTICE_CONST,
    index_class: EnvelopeIndexClass = "all",
    tail_tolerance: float = 1.0e-14,
    max_indices: int = 20_001,
) -> EnvelopeIndexDistribution:
    """构造 envelope lattice-index 分布及 center-contraction shift。

    ``all`` 是 logical-state-averaged effective comb；``even/odd`` 可用于显式
    parity sensitivity。权重对应 T1.2.1 component amplitude 的平方，并忽略相邻
    peaks 的微小 coherent overlap；该边界保存在 ``scope`` 字段中。
    """

    value = float(delta)
    spacing = float(lattice)
    tolerance = float(tail_tolerance)
    if not isfinite(value) or value < 0.0:
        raise ValueError("delta must be finite and non-negative")
    if not isfinite(spacing) or spacing <= 0.0:
        raise ValueError("lattice must be finite and positive")
    if index_class not in {"all", "even", "odd"}:
        raise ValueError("index_class must be 'all', 'even', or 'odd'")
    if not isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tail_tolerance must lie strictly between 0 and 1")
    if not isinstance(max_indices, int) or max_indices < 3:
        raise ValueError("max_indices must be an integer >= 3")

    if value == 0.0:
        if index_class == "odd":
            indices = np.asarray([-1, 1], dtype=np.int64)
            probabilities = np.asarray([0.5, 0.5], dtype=np.float64)
        else:
            indices = np.asarray([0], dtype=np.int64)
            probabilities = np.asarray([1.0], dtype=np.float64)
        shifts = np.zeros(indices.size, dtype=np.float64)
        return EnvelopeIndexDistribution(
            delta=0.0,
            lattice=spacing,
            index_class=index_class,
            indices=indices,
            probabilities=probabilities,
            shifts=shifts,
            contraction=1.0,
            mean_shift=0.0,
            variance=0.0,
            captured_weight=1.0,
        )

    width = tanh(value * value)
    radius_value = sqrt(log(1.0 / tolerance) / (width * spacing * spacing))
    if not isfinite(radius_value):
        raise ValueError("envelope support is not representable")
    radius = int(ceil(radius_value)) + 2
    indices = np.arange(-radius, radius + 1, dtype=np.int64)
    if index_class == "even":
        indices = indices[np.mod(indices, 2) == 0]
    elif index_class == "odd":
        indices = indices[np.mod(indices, 2) == 1]
    if indices.size == 0 or indices.size > max_indices:
        raise ValueError(
            "envelope distribution requires too many lattice indices; use delta=0 for the exact ideal endpoint or relax the requested support"
        )

    coordinates = indices.astype(np.float64) * spacing
    log_weights = -width * coordinates * coordinates
    maximum_log_weight = float(np.max(log_weights))
    log_weights -= maximum_log_weight
    unnormalized = np.exp(log_weights)
    normalization = float(np.sum(unnormalized))
    if not isfinite(normalization) or normalization <= 0.0:
        raise RuntimeError("envelope weights have zero or non-finite mass")
    probabilities = unnormalized / normalization
    contraction = _sech_square_delta(value)
    shifts = (contraction - 1.0) * coordinates
    mean_shift = float(np.dot(probabilities, shifts))
    variance = float(np.dot(probabilities, (shifts - mean_shift) ** 2))
    # 对所有整数格点的 omitted tail 给保守积分上界；even/odd subset 的 tail
    # 不会比 all-integer tail 更大。除以 exp(max_log_weight) 后与上面的 scaled
    # unnormalized weights 使用同一尺度。
    exponent = width * spacing * spacing
    all_integer_tail_bound = (
        sqrt(pi / exponent) * erfc(sqrt(exponent) * radius)
    )
    scaled_tail_bound = all_integer_tail_bound / exp(maximum_log_weight)
    captured_weight = normalization / (normalization + scaled_tail_bound)
    return EnvelopeIndexDistribution(
        delta=value,
        lattice=spacing,
        index_class=index_class,
        indices=indices,
        probabilities=probabilities,
        shifts=shifts,
        contraction=contraction,
        mean_shift=mean_shift,
        variance=variance,
        captured_weight=captured_weight,
    )


@dataclass(frozen=True)
class FiniteSqueezingNoiseConfig:
    """两正交分量 finite-squeezing effective simulation protocol。"""

    channel_mean: tuple[float, float] = (0.0, 0.0)
    channel_covariance: tuple[tuple[float, float], tuple[float, float]] = (
        ((0.14 * LATTICE_CONST) ** 2, 0.2 * 0.14 * 0.10 * LATTICE_CONST**2),
        (0.2 * 0.14 * 0.10 * LATTICE_CONST**2, (0.10 * LATTICE_CONST) ** 2),
    )
    data_delta: tuple[float, float] = (0.50, 0.42)
    ancilla_delta: tuple[float, float] = (0.36, 0.32)
    measurement_covariance: tuple[tuple[float, float], tuple[float, float]] = (
        ((0.03 * LATTICE_CONST) ** 2, 0.0),
        (0.0, (0.025 * LATTICE_CONST) ** 2),
    )
    envelope_index_classes: tuple[EnvelopeIndexClass, EnvelopeIndexClass] = (
        "all",
        "all",
    )
    include_envelope: bool = True
    lattice: float = LATTICE_CONST
    tail_tolerance: float = 1.0e-14
    max_envelope_indices: int = 20_001
    samples: int = 250_000
    seed: int = 2026071421

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel_mean",
            _finite_pair(self.channel_mean, "channel_mean", nonnegative=False),
        )
        object.__setattr__(
            self,
            "channel_covariance",
            _covariance_tuple(self.channel_covariance, "channel_covariance"),
        )
        object.__setattr__(
            self,
            "measurement_covariance",
            _covariance_tuple(self.measurement_covariance, "measurement_covariance"),
        )
        object.__setattr__(
            self,
            "data_delta",
            _finite_pair(self.data_delta, "data_delta", nonnegative=True),
        )
        object.__setattr__(
            self,
            "ancilla_delta",
            _finite_pair(self.ancilla_delta, "ancilla_delta", nonnegative=True),
        )
        try:
            classes = tuple(self.envelope_index_classes)
        except TypeError as exc:
            raise ValueError("envelope_index_classes must contain two entries") from exc
        if len(classes) != _AXES or any(
            value not in {"all", "even", "odd"} for value in classes
        ):
            raise ValueError(
                "envelope_index_classes must contain two all/even/odd entries"
            )
        object.__setattr__(self, "envelope_index_classes", classes)
        if not isinstance(self.include_envelope, bool):
            raise TypeError("include_envelope must be bool")
        if not isfinite(self.lattice) or self.lattice <= 0.0:
            raise ValueError("lattice must be finite and positive")
        if not isfinite(self.tail_tolerance) or not 0.0 < self.tail_tolerance < 1.0:
            raise ValueError("tail_tolerance must lie strictly between 0 and 1")
        if not isinstance(self.max_envelope_indices, int) or self.max_envelope_indices < 3:
            raise ValueError("max_envelope_indices must be an integer >= 3")
        if not isinstance(self.samples, int) or self.samples < 1_000:
            raise ValueError("samples must be an integer >= 1000")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class FiniteSqueezingNoiseBudget:
    """可逐项审计的 analytic covariance budget。"""

    channel: NDArray[np.float64]
    data_gkp: NDArray[np.float64]
    ancilla_gkp: NDArray[np.float64]
    measurement: NDArray[np.float64]
    finite_energy_envelope: NDArray[np.float64]
    physical_total: NDArray[np.float64]
    observed_total: NDArray[np.float64]
    ideal_observed: NDArray[np.float64]
    finite_squeezing_excess: NDArray[np.float64]
    envelope_distributions: tuple[EnvelopeIndexDistribution, EnvelopeIndexDistribution]
    scope: str = "decomposed_syndrome_level_effective_model"

    def as_dict(self) -> dict[str, object]:
        names = (
            "channel",
            "data_gkp",
            "ancilla_gkp",
            "measurement",
            "finite_energy_envelope",
            "physical_total",
            "observed_total",
            "ideal_observed",
            "finite_squeezing_excess",
        )
        return {
            "scope": self.scope,
            "covariances": {
                name: np.asarray(getattr(self, name), dtype=np.float64).tolist()
                for name in names
            },
            "trace_contributions": {
                name: float(np.trace(np.asarray(getattr(self, name))))
                for name in (
                    "channel",
                    "data_gkp",
                    "ancilla_gkp",
                    "measurement",
                    "finite_energy_envelope",
                    "finite_squeezing_excess",
                    "observed_total",
                )
            },
            "envelope": [
                {
                    "axis": axis,
                    "delta": distribution.delta,
                    "index_class": distribution.index_class,
                    "contraction": distribution.contraction,
                    "mean_shift": distribution.mean_shift,
                    "variance": distribution.variance,
                    "support_size": int(distribution.indices.size),
                    "captured_weight_lower_bound": distribution.captured_weight,
                    "scope": distribution.scope,
                }
                for axis, distribution in zip(("q", "p"), self.envelope_distributions)
            ],
        }


def finite_squeezing_noise_budget(
    config: FiniteSqueezingNoiseConfig,
) -> FiniteSqueezingNoiseBudget:
    if not isinstance(config, FiniteSqueezingNoiseConfig):
        raise TypeError("config must be a FiniteSqueezingNoiseConfig")
    channel = _covariance_array(config.channel_covariance)
    measurement = _covariance_array(config.measurement_covariance)
    data_gkp = np.diag([isolated_peak_variance(value) for value in config.data_delta])
    ancilla_gkp = np.diag(
        [isolated_peak_variance(value) for value in config.ancilla_delta]
    )
    distributions = tuple(
        envelope_index_distribution(
            delta,
            lattice=config.lattice,
            index_class=index_class,
            tail_tolerance=config.tail_tolerance,
            max_indices=config.max_envelope_indices,
        )
        for delta, index_class in zip(config.data_delta, config.envelope_index_classes)
    )
    envelope = np.diag(
        [distribution.variance for distribution in distributions]
        if config.include_envelope
        else [0.0, 0.0]
    )
    physical_total = channel + data_gkp + envelope
    observed_total = physical_total + ancilla_gkp + measurement
    ideal_observed = channel + measurement
    finite_excess = data_gkp + ancilla_gkp + envelope
    return FiniteSqueezingNoiseBudget(
        channel=channel,
        data_gkp=data_gkp,
        ancilla_gkp=ancilla_gkp,
        measurement=measurement,
        finite_energy_envelope=envelope,
        physical_total=physical_total,
        observed_total=observed_total,
        ideal_observed=ideal_observed,
        finite_squeezing_excess=finite_excess,
        envelope_distributions=distributions,  # type: ignore[arg-type]
    )


def _sample_gaussian(
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
    size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    standard = rng.standard_normal((size, _AXES))
    return mean + standard @ factor.T


def _centered_wrap(values: NDArray[np.float64], lattice: float) -> NDArray[np.float64]:
    indices = np.floor(values / lattice + 0.5)
    return values - indices * lattice


def _logical_parity(values: NDArray[np.float64], lattice: float) -> NDArray[np.int64]:
    indices = np.floor(values / lattice + 0.5).astype(np.int64)
    return np.mod(indices, 2)


@dataclass(frozen=True)
class FiniteSqueezingNoiseSummary:
    samples: int
    seed: int
    logical_error_rate: float
    logical_error_ci_low: float
    logical_error_ci_high: float
    q_logical_error_rate: float
    p_logical_error_rate: float
    empirical_covariances: dict[str, NDArray[np.float64]]
    observed_covariance_relative_error: float
    physical_covariance_relative_error: float
    evidence_scope: str = "finite_squeezing_effective_monte_carlo"

    def as_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "seed": self.seed,
            "logical_error_rate": self.logical_error_rate,
            "logical_error_ci": [
                self.logical_error_ci_low,
                self.logical_error_ci_high,
            ],
            "axis_logical_error_rate": {
                "q": self.q_logical_error_rate,
                "p": self.p_logical_error_rate,
            },
            "empirical_covariances": {
                name: covariance.tolist()
                for name, covariance in self.empirical_covariances.items()
            },
            "observed_covariance_relative_error": self.observed_covariance_relative_error,
            "physical_covariance_relative_error": self.physical_covariance_relative_error,
            "evidence_scope": self.evidence_scope,
        }


@dataclass(frozen=True)
class FiniteSqueezingNoiseBatch:
    """一次 paired Monte Carlo 的分解 component arrays 与 correction outcome。"""

    config: FiniteSqueezingNoiseConfig
    budget: FiniteSqueezingNoiseBudget
    channel: NDArray[np.float64]
    data_gkp: NDArray[np.float64]
    ancilla_gkp: NDArray[np.float64]
    measurement: NDArray[np.float64]
    finite_energy_envelope: NDArray[np.float64]
    physical: NDArray[np.float64]
    observed: NDArray[np.float64]
    syndrome: NDArray[np.float64]
    correction: NDArray[np.float64]
    corrected_residual: NDArray[np.float64]
    logical_parity: NDArray[np.int64]

    @property
    def logical_error_mask(self) -> NDArray[np.bool_]:
        return np.any(self.logical_parity != 0, axis=1)

    def summary(self) -> FiniteSqueezingNoiseSummary:
        components = {
            "channel": self.channel,
            "data_gkp": self.data_gkp,
            "ancilla_gkp": self.ancilla_gkp,
            "measurement": self.measurement,
            "finite_energy_envelope": self.finite_energy_envelope,
            "physical": self.physical,
            "observed": self.observed,
        }
        empirical = {
            name: np.cov(values, rowvar=False, ddof=1)
            for name, values in components.items()
        }
        logical = self.logical_error_mask
        count = int(np.sum(logical))
        rate = count / self.config.samples
        z = 1.96
        denominator = 1.0 + z * z / self.config.samples
        center = (rate + z * z / (2.0 * self.config.samples)) / denominator
        radius = (
            z
            * sqrt(
                rate * (1.0 - rate) / self.config.samples
                + z * z / (4.0 * self.config.samples * self.config.samples)
            )
            / denominator
        )
        observed_scale = max(float(np.linalg.norm(self.budget.observed_total)), 1.0e-15)
        physical_scale = max(float(np.linalg.norm(self.budget.physical_total)), 1.0e-15)
        return FiniteSqueezingNoiseSummary(
            samples=self.config.samples,
            seed=self.config.seed,
            logical_error_rate=rate,
            logical_error_ci_low=max(0.0, center - radius),
            logical_error_ci_high=min(1.0, center + radius),
            q_logical_error_rate=float(np.mean(self.logical_parity[:, 0] != 0)),
            p_logical_error_rate=float(np.mean(self.logical_parity[:, 1] != 0)),
            empirical_covariances=empirical,
            observed_covariance_relative_error=float(
                np.linalg.norm(empirical["observed"] - self.budget.observed_total)
                / observed_scale
            ),
            physical_covariance_relative_error=float(
                np.linalg.norm(empirical["physical"] - self.budget.physical_total)
                / physical_scale
            ),
        )


def sample_finite_squeezing_noise(
    config: FiniteSqueezingNoiseConfig | None = None,
) -> FiniteSqueezingNoiseBatch:
    """采样 decomposed physical/observation noise 并执行标准 wrapped correction。"""

    actual = FiniteSqueezingNoiseConfig() if config is None else config
    if not isinstance(actual, FiniteSqueezingNoiseConfig):
        raise TypeError("config must be a FiniteSqueezingNoiseConfig or None")
    budget = finite_squeezing_noise_budget(actual)
    child_sequences = np.random.SeedSequence(actual.seed).spawn(6)
    rng_channel, rng_data, rng_ancilla, rng_measurement, rng_eq, rng_ep = (
        np.random.default_rng(sequence) for sequence in child_sequences
    )
    zeros = np.zeros(_AXES, dtype=np.float64)
    channel = _sample_gaussian(
        np.asarray(actual.channel_mean, dtype=np.float64),
        budget.channel,
        actual.samples,
        rng_channel,
    )
    data = _sample_gaussian(zeros, budget.data_gkp, actual.samples, rng_data)
    ancilla = _sample_gaussian(zeros, budget.ancilla_gkp, actual.samples, rng_ancilla)
    measurement = _sample_gaussian(
        zeros,
        budget.measurement,
        actual.samples,
        rng_measurement,
    )
    envelope = np.zeros((actual.samples, _AXES), dtype=np.float64)
    if actual.include_envelope:
        envelope[:, 0] = budget.envelope_distributions[0].sample(
            actual.samples, rng_eq
        )
        envelope[:, 1] = budget.envelope_distributions[1].sample(
            actual.samples, rng_ep
        )
    physical = channel + data + envelope
    observed = physical + ancilla + measurement
    syndrome = _centered_wrap(observed, actual.lattice)
    correction = syndrome.copy()
    corrected_residual = physical - correction
    parity = _logical_parity(corrected_residual, actual.lattice)
    return FiniteSqueezingNoiseBatch(
        config=actual,
        budget=budget,
        channel=channel,
        data_gkp=data,
        ancilla_gkp=ancilla,
        measurement=measurement,
        finite_energy_envelope=envelope,
        physical=physical,
        observed=observed,
        syndrome=syndrome,
        correction=correction,
        corrected_residual=corrected_residual,
        logical_parity=parity,
    )


@dataclass(frozen=True)
class FiniteSqueezingSweepPoint:
    scale: float
    data_delta: tuple[float, float]
    ancilla_delta: tuple[float, float]
    finite_squeezing_excess_trace: float
    observed_covariance_trace: float
    observed_covariance_relative_error: float
    logical_error_rate: float
    logical_error_ci_low: float
    logical_error_ci_high: float


@dataclass(frozen=True)
class FiniteSqueezingSweepResult:
    base_config: FiniteSqueezingNoiseConfig
    scales: tuple[float, ...]
    points: tuple[FiniteSqueezingSweepPoint, ...]
    analytic_excess_strictly_decreases: bool
    ideal_endpoint_exact: bool
    broad_finite_squeezing_rate_above_ideal: bool
    max_observed_covariance_relative_error: float
    evidence_scope: str = "high_squeezing_limit_effective_validation"

    def as_dict(self) -> dict[str, object]:
        return {
            "base_config": {
                "channel_mean": list(self.base_config.channel_mean),
                "channel_covariance": [list(row) for row in self.base_config.channel_covariance],
                "data_delta": list(self.base_config.data_delta),
                "ancilla_delta": list(self.base_config.ancilla_delta),
                "measurement_covariance": [
                    list(row) for row in self.base_config.measurement_covariance
                ],
                "include_envelope": self.base_config.include_envelope,
                "lattice": self.base_config.lattice,
                "samples": self.base_config.samples,
                "seed": self.base_config.seed,
            },
            "scales": list(self.scales),
            "points": [
                {
                    "scale": point.scale,
                    "data_delta": list(point.data_delta),
                    "ancilla_delta": list(point.ancilla_delta),
                    "finite_squeezing_excess_trace": point.finite_squeezing_excess_trace,
                    "observed_covariance_trace": point.observed_covariance_trace,
                    "observed_covariance_relative_error": point.observed_covariance_relative_error,
                    "logical_error_rate": point.logical_error_rate,
                    "logical_error_ci": [
                        point.logical_error_ci_low,
                        point.logical_error_ci_high,
                    ],
                }
                for point in self.points
            ],
            "checks": {
                "analytic_excess_strictly_decreases": self.analytic_excess_strictly_decreases,
                "ideal_endpoint_exact": self.ideal_endpoint_exact,
                "broad_finite_squeezing_rate_above_ideal": self.broad_finite_squeezing_rate_above_ideal,
                "max_observed_covariance_relative_error": self.max_observed_covariance_relative_error,
            },
            "evidence_scope": self.evidence_scope,
            "claim_boundary": {
                "allowed": "decomposed finite-squeezing effective noise and exact ideal endpoint",
                "forbidden": "Fock-space fidelity, device-calibrated squeezing, or experimental logical lifetime",
            },
        }


def run_high_squeezing_limit_sweep(
    config: FiniteSqueezingNoiseConfig | None = None,
    *,
    scales: tuple[float, ...] = (1.0, 0.75, 0.50, 0.25, 0.10, 0.0),
) -> FiniteSqueezingSweepResult:
    """用相同 component RNG streams 验证 ``Delta->0`` 的 ideal endpoint。"""

    base = FiniteSqueezingNoiseConfig() if config is None else config
    if not isinstance(base, FiniteSqueezingNoiseConfig):
        raise TypeError("config must be a FiniteSqueezingNoiseConfig or None")
    try:
        actual_scales = tuple(float(value) for value in scales)
    except (TypeError, ValueError) as exc:
        raise ValueError("scales must contain finite numeric values") from exc
    if len(actual_scales) < 3 or not all(
        isfinite(value) and 0.0 <= value <= 1.0 for value in actual_scales
    ):
        raise ValueError("scales must contain at least three finite values in [0, 1]")
    if not all(left > right for left, right in zip(actual_scales, actual_scales[1:])):
        raise ValueError("scales must be strictly decreasing")
    if actual_scales[-1] != 0.0:
        raise ValueError("scales must end at the exact ideal endpoint 0")

    points: list[FiniteSqueezingSweepPoint] = []
    budgets: list[FiniteSqueezingNoiseBudget] = []
    for scale in actual_scales:
        point_config = replace(
            base,
            data_delta=tuple(scale * value for value in base.data_delta),
            ancilla_delta=tuple(scale * value for value in base.ancilla_delta),
        )
        batch = sample_finite_squeezing_noise(point_config)
        summary = batch.summary()
        budgets.append(batch.budget)
        points.append(
            FiniteSqueezingSweepPoint(
                scale=scale,
                data_delta=point_config.data_delta,
                ancilla_delta=point_config.ancilla_delta,
                finite_squeezing_excess_trace=float(
                    np.trace(batch.budget.finite_squeezing_excess)
                ),
                observed_covariance_trace=float(np.trace(batch.budget.observed_total)),
                observed_covariance_relative_error=summary.observed_covariance_relative_error,
                logical_error_rate=summary.logical_error_rate,
                logical_error_ci_low=summary.logical_error_ci_low,
                logical_error_ci_high=summary.logical_error_ci_high,
            )
        )

    excess = [point.finite_squeezing_excess_trace for point in points]
    final = budgets[-1]
    ideal_endpoint_exact = bool(
        np.array_equal(final.finite_squeezing_excess, np.zeros((_AXES, _AXES)))
        and np.array_equal(final.observed_total, final.ideal_observed)
        and np.array_equal(final.physical_total, final.channel)
    )
    return FiniteSqueezingSweepResult(
        base_config=base,
        scales=actual_scales,
        points=tuple(points),
        analytic_excess_strictly_decreases=all(
            right < left for left, right in zip(excess, excess[1:])
        ),
        ideal_endpoint_exact=ideal_endpoint_exact,
        broad_finite_squeezing_rate_above_ideal=(
            points[0].logical_error_ci_low > points[-1].logical_error_ci_high
        ),
        max_observed_covariance_relative_error=max(
            point.observed_covariance_relative_error for point in points
        ),
    )


def write_finite_squeezing_report(
    result: FiniteSqueezingSweepResult,
    output_path: str | Path,
) -> Path:
    if not isinstance(result, FiniteSqueezingSweepResult):
        raise TypeError("result must be a FiniteSqueezingSweepResult")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    payload["base_budget"] = finite_squeezing_noise_budget(result.base_config).as_dict()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--samples", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=2026071421)
    arguments = parser.parse_args()
    config = FiniteSqueezingNoiseConfig(samples=arguments.samples, seed=arguments.seed)
    result = run_high_squeezing_limit_sweep(config)
    write_finite_squeezing_report(result, arguments.output)
    print(json.dumps(result.as_dict()["checks"], ensure_ascii=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "EnvelopeIndexClass",
    "EnvelopeIndexDistribution",
    "FiniteSqueezingNoiseConfig",
    "FiniteSqueezingNoiseBudget",
    "FiniteSqueezingNoiseSummary",
    "FiniteSqueezingNoiseBatch",
    "FiniteSqueezingSweepPoint",
    "FiniteSqueezingSweepResult",
    "isolated_peak_variance",
    "envelope_index_distribution",
    "finite_squeezing_noise_budget",
    "sample_finite_squeezing_noise",
    "run_high_squeezing_limit_sweep",
    "write_finite_squeezing_report",
]
