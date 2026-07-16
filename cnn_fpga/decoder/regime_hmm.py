"""Observed-window Gaussian HMM for causal four-regime estimation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from physics.constants import LATTICE_CONST


REGIME_CLASSES = ("normal", "burst", "leakage", "calibration_shift")
RAW_FEATURE_NAMES = (
    "residual_q",
    "residual_p",
    "x_is_e",
    "z_is_e",
    "any_leakage",
    "quadrature_phase_bit",
    "valid",
    "deadline_ok",
)
SUMMARY_FEATURE_NAMES = (
    "mean_q",
    "mean_p",
    "variance_q",
    "variance_p",
    "covariance_qp",
    "mean_abs_q",
    "mean_abs_p",
    "tail_fraction",
    "x_e_fraction",
    "z_e_fraction",
    "leakage_fraction",
    "valid_fraction",
    "deadline_ok_fraction",
    "phase_selected_mean",
)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real") from exc
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not isfinite(result) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    if not isfinite(maximum):
        raise FloatingPointError("logsumexp received no finite value")
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _normalize_log(log_values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(log_values, dtype=np.float64) - _logsumexp(log_values)
    posterior = np.exp(normalized)
    if not np.all(np.isfinite(posterior)) or not np.isclose(
        np.sum(posterior), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise FloatingPointError("posterior normalization failed")
    return posterior


@dataclass(frozen=True)
class RegimeObservationWindow:
    window_index: int
    start_cycle: int
    values: np.ndarray

    def __post_init__(self) -> None:
        window_index = _integer(self.window_index, "window_index")
        start_cycle = _integer(self.start_cycle, "start_cycle")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(RAW_FEATURE_NAMES):
            raise ValueError(
                f"values must have shape (cycles, {len(RAW_FEATURE_NAMES)})"
            )
        if values.shape[0] < 2:
            raise ValueError("a regime window must contain at least two cycles")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        binary = values[:, 2:]
        if np.any((binary != 0.0) & (binary != 1.0)):
            raise ValueError("event/phase/health columns must be binary")
        object.__setattr__(self, "window_index", window_index)
        object.__setattr__(self, "start_cycle", start_cycle)
        object.__setattr__(self, "values", _readonly(values))

    @property
    def cycles(self) -> int:
        return int(self.values.shape[0])

    @property
    def end_cycle(self) -> int:
        return self.start_cycle + self.cycles - 1


def summarize_regime_window(
    window: RegimeObservationWindow,
    *,
    tail_threshold: float = 0.35 * LATTICE_CONST,
) -> np.ndarray:
    if not isinstance(window, RegimeObservationWindow):
        raise TypeError("window must be RegimeObservationWindow")
    threshold = _positive(tail_threshold, "tail_threshold")
    values = window.values
    residual = values[:, :2]
    covariance = np.cov(residual, rowvar=False, ddof=1)
    phase_selected = np.where(values[:, 5] == 0.0, residual[:, 0], residual[:, 1])
    summary = np.asarray(
        [
            np.mean(residual[:, 0]),
            np.mean(residual[:, 1]),
            covariance[0, 0],
            covariance[1, 1],
            covariance[0, 1],
            np.mean(np.abs(residual[:, 0])),
            np.mean(np.abs(residual[:, 1])),
            np.mean(np.max(np.abs(residual), axis=1) >= threshold),
            np.mean(values[:, 2]),
            np.mean(values[:, 3]),
            np.mean(values[:, 4]),
            np.mean(values[:, 6]),
            np.mean(values[:, 7]),
            np.mean(phase_selected),
        ],
        dtype=np.float64,
    )
    if summary.shape != (len(SUMMARY_FEATURE_NAMES),) or not np.all(np.isfinite(summary)):
        raise FloatingPointError("window summary is invalid")
    return summary


@dataclass(frozen=True)
class RegimeEstimatorBudget:
    window_cycles: int = 32
    update_period_cycles: int = 32
    raw_feature_count: int = len(RAW_FEATURE_NAMES)
    summary_feature_count: int = len(SUMMARY_FEATURE_NAMES)
    max_macs_per_update: int = 4096
    max_float32_state_bytes: int = 4096
    future_comparator_contract: str = "same_raw_window_shape_update_period_and_budget_for_T4.1_CNN_TCN_GRU"

    def __post_init__(self) -> None:
        for name in (
            "window_cycles",
            "update_period_cycles",
            "raw_feature_count",
            "summary_feature_count",
            "max_macs_per_update",
            "max_float32_state_bytes",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))
        if self.window_cycles != self.update_period_cycles:
            raise ValueError("current baseline requires non-overlapping one-update-per-window cadence")
        if self.raw_feature_count != len(RAW_FEATURE_NAMES):
            raise ValueError("raw_feature_count must match RAW_FEATURE_NAMES")
        if self.summary_feature_count != len(SUMMARY_FEATURE_NAMES):
            raise ValueError("summary_feature_count must match SUMMARY_FEATURE_NAMES")


@dataclass(frozen=True)
class GaussianRegimeHMM:
    standardization_mean: np.ndarray
    standardization_scale: np.ndarray
    emission_means: np.ndarray
    emission_covariances: np.ndarray
    emission_precisions: np.ndarray
    emission_log_determinants: np.ndarray
    transition_matrix: np.ndarray
    initial_probabilities: np.ndarray
    class_prior_probabilities: np.ndarray
    covariance_regularization: float
    transition_smoothing: float

    def __post_init__(self) -> None:
        classes = len(REGIME_CLASSES)
        features = len(SUMMARY_FEATURE_NAMES)
        shapes = {
            "standardization_mean": (features,),
            "standardization_scale": (features,),
            "emission_means": (classes, features),
            "emission_covariances": (classes, features, features),
            "emission_precisions": (classes, features, features),
            "emission_log_determinants": (classes,),
            "transition_matrix": (classes, classes),
            "initial_probabilities": (classes,),
            "class_prior_probabilities": (classes,),
        }
        for name, shape in shapes.items():
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, _readonly(values))
        if np.any(self.standardization_scale <= 0.0):
            raise ValueError("standardization_scale must be positive")
        if np.any(self.initial_probabilities <= 0.0) or not np.isclose(
            np.sum(self.initial_probabilities), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("initial_probabilities must be positive and sum to one")
        if np.any(self.class_prior_probabilities <= 0.0) or not np.isclose(
            np.sum(self.class_prior_probabilities), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("class_prior_probabilities must be positive and sum to one")
        if np.any(self.transition_matrix <= 0.0) or not np.allclose(
            np.sum(self.transition_matrix, axis=1), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("transition_matrix rows must be positive and sum to one")
        for covariance, precision in zip(
            self.emission_covariances, self.emission_precisions, strict=True
        ):
            if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
                raise ValueError("emission covariance must be positive definite")
            if not np.allclose(covariance @ precision, np.eye(features), atol=1.0e-7):
                raise ValueError("emission precision must invert covariance")
        object.__setattr__(
            self,
            "covariance_regularization",
            _positive(self.covariance_regularization, "covariance_regularization"),
        )
        object.__setattr__(
            self,
            "transition_smoothing",
            _positive(self.transition_smoothing, "transition_smoothing"),
        )

    @property
    def parameter_count(self) -> int:
        return int(
            self.standardization_mean.size
            + self.standardization_scale.size
            + self.emission_means.size
            + self.emission_precisions.size
            + self.emission_log_determinants.size
            + self.transition_matrix.size
            + self.initial_probabilities.size
            + self.class_prior_probabilities.size
        )

    @property
    def macs_per_update_proxy(self) -> int:
        classes = len(REGIME_CLASSES)
        features = len(SUMMARY_FEATURE_NAMES)
        return classes * features * features + classes * classes

    def standardize(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(SUMMARY_FEATURE_NAMES):
            raise ValueError("features have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("features must be finite")
        return (values - self.standardization_mean) / self.standardization_scale

    def emission_log_likelihood(self, features: np.ndarray) -> np.ndarray:
        standardized = self.standardize(features)
        difference = standardized[:, None, :] - self.emission_means[None, :, :]
        quadratic = np.einsum(
            "nki,kij,nkj->nk",
            difference,
            self.emission_precisions,
            difference,
            optimize=True,
        )
        constant = len(SUMMARY_FEATURE_NAMES) * np.log(2.0 * np.pi)
        result = -0.5 * (
            quadratic + self.emission_log_determinants[None, :] + constant
        )
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("emission log likelihood is non-finite")
        return result

    def memoryless_posterior(
        self, features: np.ndarray, *, temperature: float = 1.0
    ) -> np.ndarray:
        actual_temperature = _positive(temperature, "temperature")
        emissions = self.emission_log_likelihood(features)
        log_prior = np.log(self.class_prior_probabilities)
        return np.vstack(
            [_normalize_log((row + log_prior) / actual_temperature) for row in emissions]
        )

    def filter_sequence(
        self, features: np.ndarray, *, temperature: float = 1.0
    ) -> np.ndarray:
        actual_temperature = _positive(temperature, "temperature")
        emissions = self.emission_log_likelihood(features)
        output = np.empty_like(emissions)
        previous = self.initial_probabilities
        for index, emission in enumerate(emissions):
            prediction = previous @ self.transition_matrix
            unnormalized = np.log(prediction) + emission
            posterior = _normalize_log(unnormalized)
            output[index] = _normalize_log(
                np.log(np.clip(posterior, np.finfo(np.float64).tiny, 1.0))
                / actual_temperature
            )
            previous = posterior
        return output


def _labels_to_indices(labels: Sequence[str], expected: int) -> np.ndarray:
    if len(labels) != expected:
        raise ValueError("label count must match feature rows")
    lookup = {name: index for index, name in enumerate(REGIME_CLASSES)}
    try:
        return np.asarray([lookup[label] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unknown regime label {exc.args[0]!r}") from exc


def fit_supervised_gaussian_hmm(
    feature_sequences: Sequence[np.ndarray],
    label_sequences: Sequence[Sequence[str]],
    *,
    covariance_regularization: float = 0.02,
    transition_smoothing: float = 1.0,
) -> GaussianRegimeHMM:
    if len(feature_sequences) != len(label_sequences) or not feature_sequences:
        raise ValueError("feature_sequences and label_sequences must be nonempty and aligned")
    regularization = _positive(covariance_regularization, "covariance_regularization")
    smoothing = _positive(transition_smoothing, "transition_smoothing")
    feature_arrays = []
    label_arrays = []
    for features, labels in zip(feature_sequences, label_sequences, strict=True):
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(SUMMARY_FEATURE_NAMES):
            raise ValueError("every feature sequence must have the canonical summary width")
        if len(array) < 2 or not np.all(np.isfinite(array)):
            raise ValueError("every feature sequence must have at least two finite rows")
        feature_arrays.append(array)
        label_arrays.append(_labels_to_indices(labels, len(array)))
    all_features = np.vstack(feature_arrays)
    all_labels = np.concatenate(label_arrays)
    counts = np.bincount(all_labels, minlength=len(REGIME_CLASSES))
    if np.any(counts < len(SUMMARY_FEATURE_NAMES) + 2):
        raise ValueError("each regime needs more samples than the feature dimension")
    standard_mean = np.mean(all_features, axis=0)
    standard_scale = np.std(all_features, axis=0, ddof=1)
    standard_scale = np.where(standard_scale < 1.0e-8, 1.0, standard_scale)
    standardized = (all_features - standard_mean) / standard_scale
    emission_means = np.empty((len(REGIME_CLASSES), len(SUMMARY_FEATURE_NAMES)))
    emission_covariances = np.empty(
        (len(REGIME_CLASSES), len(SUMMARY_FEATURE_NAMES), len(SUMMARY_FEATURE_NAMES))
    )
    for state_index in range(len(REGIME_CLASSES)):
        selected = standardized[all_labels == state_index]
        emission_means[state_index] = np.mean(selected, axis=0)
        covariance = np.cov(selected, rowvar=False, ddof=1)
        emission_covariances[state_index] = covariance + regularization * np.eye(
            len(SUMMARY_FEATURE_NAMES)
        )
    emission_precisions = np.linalg.inv(emission_covariances)
    signs, log_determinants = np.linalg.slogdet(emission_covariances)
    if np.any(signs <= 0.0):
        raise ValueError("regularized emission covariance is not positive definite")
    transition_counts = np.full(
        (len(REGIME_CLASSES), len(REGIME_CLASSES)), smoothing, dtype=np.float64
    )
    initial_counts = np.full(len(REGIME_CLASSES), smoothing, dtype=np.float64)
    for labels in label_arrays:
        initial_counts[labels[0]] += 1.0
        for previous, current in zip(labels[:-1], labels[1:], strict=True):
            transition_counts[previous, current] += 1.0
    transition_matrix = transition_counts / np.sum(transition_counts, axis=1, keepdims=True)
    initial_probabilities = initial_counts / np.sum(initial_counts)
    class_prior_probabilities = (counts + smoothing) / (
        np.sum(counts) + smoothing * len(REGIME_CLASSES)
    )
    return GaussianRegimeHMM(
        standardization_mean=standard_mean,
        standardization_scale=standard_scale,
        emission_means=emission_means,
        emission_covariances=emission_covariances,
        emission_precisions=emission_precisions,
        emission_log_determinants=log_determinants,
        transition_matrix=transition_matrix,
        initial_probabilities=initial_probabilities,
        class_prior_probabilities=class_prior_probabilities,
        covariance_regularization=regularization,
        transition_smoothing=smoothing,
    )


def posterior_dict(posterior: Sequence[float]) -> Mapping[str, float]:
    values = np.asarray(posterior, dtype=np.float64)
    if values.shape != (len(REGIME_CLASSES),) or np.any(values < 0.0) or not np.isclose(
        np.sum(values), 1.0, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("posterior must be a normalized four-state vector")
    return MappingProxyType(
        {name: float(values[index]) for index, name in enumerate(REGIME_CLASSES)}
    )


__all__ = [
    "REGIME_CLASSES",
    "RAW_FEATURE_NAMES",
    "SUMMARY_FEATURE_NAMES",
    "RegimeObservationWindow",
    "RegimeEstimatorBudget",
    "GaussianRegimeHMM",
    "fit_supervised_gaussian_hmm",
    "posterior_dict",
    "summarize_regime_window",
]
