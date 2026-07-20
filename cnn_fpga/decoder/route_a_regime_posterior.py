"""Causal observed-only posterior model for the Route-A policy schema.

The older regime-HMM baseline uses ``normal/burst/leakage/calibration_shift``.
Route A deliberately separates leakage into the deterministic event/reset FSM
and therefore needs the distinct posterior order
``normal/smooth/calibration_shift/burst``.  This module makes that order part of
the serialized model contract so an array from the legacy HMM cannot be
silently reinterpreted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from cnn_fpga.decoder.regime_hmm import SUMMARY_FEATURE_NAMES


ROUTE_A_POSTERIOR_CLASSES = (
    "normal",
    "smooth",
    "calibration_shift",
    "burst",
)
MODEL_SCHEMA = "route-a-observed-gaussian-hmm-v1"
EVENT_MODEL_SCHEMA = "route-a-observed-tail-event-logit-v1"


def _positive(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _readonly(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _logsumexp_rows(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1, keepdims=True)
    shifted = np.exp(values - maximum)
    return maximum[:, 0] + np.log(np.sum(shifted, axis=1))


def _normalize_log_vector(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    posterior = weights / np.sum(weights)
    if not np.all(np.isfinite(posterior)) or not np.isclose(
        np.sum(posterior), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise FloatingPointError("posterior normalization failed")
    return posterior


def temperature_scale(posterior: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrate posterior confidence without altering HMM state recursion."""

    actual = _positive(temperature, "temperature")
    values = np.asarray(posterior, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(ROUTE_A_POSTERIOR_CLASSES):
        raise ValueError("posterior has the wrong shape")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("posterior must be finite and non-negative")
    row_sums = np.sum(values, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("posterior rows must sum to one before temperature scaling")
    # A valid Gaussian likelihood can underflow to an exact zero for a remote
    # class.  Clip only for the log transform, then renormalize.  Rejecting the
    # row would selectively discard the hardest/OOD evidence.
    clipped = np.clip(values, np.finfo(np.float64).tiny, 1.0)
    clipped /= np.sum(clipped, axis=1, keepdims=True)
    logits = np.log(clipped) / actual
    logits -= _logsumexp_rows(logits)[:, None]
    result = np.exp(logits)
    if not np.allclose(np.sum(result, axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise FloatingPointError("temperature-scaled posterior did not normalize")
    return result


@dataclass(frozen=True)
class RouteAPosteriorModel:
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
    class_order: tuple[str, ...] = ROUTE_A_POSTERIOR_CLASSES
    schema_version: str = MODEL_SCHEMA

    def __post_init__(self) -> None:
        classes = len(ROUTE_A_POSTERIOR_CLASSES)
        features = len(SUMMARY_FEATURE_NAMES)
        if tuple(self.class_order) != ROUTE_A_POSTERIOR_CLASSES:
            raise ValueError("Route-A posterior class order mismatch")
        if self.schema_version != MODEL_SCHEMA:
            raise ValueError("Route-A posterior model schema mismatch")
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
            object.__setattr__(self, name, _readonly(getattr(self, name), shape, name))
        if np.any(self.standardization_scale <= 0.0):
            raise ValueError("standardization scale must be positive")
        if np.any(self.transition_matrix <= 0.0) or not np.allclose(
            np.sum(self.transition_matrix, axis=1), 1.0, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("transition rows must be positive and normalized")
        for name in ("initial_probabilities", "class_prior_probabilities"):
            values = getattr(self, name)
            if np.any(values <= 0.0) or not np.isclose(
                np.sum(values), 1.0, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(f"{name} must be positive and normalized")
        for covariance, precision in zip(
            self.emission_covariances, self.emission_precisions, strict=True
        ):
            if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
                raise ValueError("emission covariance must be positive definite")
            if not np.allclose(covariance @ precision, np.eye(features), atol=1.0e-7):
                raise ValueError("emission precision does not invert covariance")
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
        classes = len(ROUTE_A_POSTERIOR_CLASSES)
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
            raise FloatingPointError("emission likelihood is non-finite")
        return result

    def filter_base(self, features: np.ndarray) -> np.ndarray:
        """Return causal untempered forward posteriors."""

        emissions = self.emission_log_likelihood(features)
        output = np.empty_like(emissions)
        previous = self.initial_probabilities
        for index, emission in enumerate(emissions):
            prediction = previous @ self.transition_matrix
            previous = _normalize_log_vector(np.log(prediction) + emission)
            output[index] = previous
        return output

    def filter_sequence(self, features: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
        return temperature_scale(self.filter_base(features), temperature)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "class_order": list(self.class_order),
            "summary_feature_names": list(SUMMARY_FEATURE_NAMES),
            "standardization_mean": self.standardization_mean.tolist(),
            "standardization_scale": self.standardization_scale.tolist(),
            "emission_means": self.emission_means.tolist(),
            "emission_covariances": self.emission_covariances.tolist(),
            "emission_precisions": self.emission_precisions.tolist(),
            "emission_log_determinants": self.emission_log_determinants.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
            "initial_probabilities": self.initial_probabilities.tolist(),
            "class_prior_probabilities": self.class_prior_probabilities.tolist(),
            "covariance_regularization": self.covariance_regularization,
            "transition_smoothing": self.transition_smoothing,
            "parameter_count": self.parameter_count,
            "macs_per_update_proxy": self.macs_per_update_proxy,
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RouteAPosteriorModel":
        if payload.get("schema_version") != MODEL_SCHEMA:
            raise ValueError("Route-A posterior checkpoint schema mismatch")
        if tuple(payload.get("class_order", ())) != ROUTE_A_POSTERIOR_CLASSES:
            raise ValueError("Route-A posterior checkpoint class order mismatch")
        arrays = (
            "standardization_mean",
            "standardization_scale",
            "emission_means",
            "emission_covariances",
            "emission_precisions",
            "emission_log_determinants",
            "transition_matrix",
            "initial_probabilities",
            "class_prior_probabilities",
        )
        return cls(
            **{name: np.asarray(payload[name], dtype=np.float64) for name in arrays},
            covariance_regularization=float(payload["covariance_regularization"]),
            transition_smoothing=float(payload["transition_smoothing"]),
        )


@dataclass(frozen=True)
class ObservedTailEventModel:
    """Small observed-summary logit used by the independent event fallback."""

    standardization_mean: np.ndarray
    standardization_scale: np.ndarray
    weights: np.ndarray
    bias: float
    schema_version: str = EVENT_MODEL_SCHEMA

    def __post_init__(self) -> None:
        width = len(SUMMARY_FEATURE_NAMES)
        object.__setattr__(
            self,
            "standardization_mean",
            _readonly(self.standardization_mean, (width,), "event standardization_mean"),
        )
        object.__setattr__(
            self,
            "standardization_scale",
            _readonly(self.standardization_scale, (width,), "event standardization_scale"),
        )
        object.__setattr__(self, "weights", _readonly(self.weights, (2 * width,), "event weights"))
        if np.any(self.standardization_scale <= 0.0):
            raise ValueError("event standardization scale must be positive")
        actual_bias = float(self.bias)
        if not isfinite(actual_bias):
            raise ValueError("event bias must be finite")
        object.__setattr__(self, "bias", actual_bias)
        if self.schema_version != EVENT_MODEL_SCHEMA:
            raise ValueError("event model schema mismatch")

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(SUMMARY_FEATURE_NAMES):
            raise ValueError("event features have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("event features must be finite")
        standardized = (values - self.standardization_mean) / self.standardization_scale
        expanded = np.concatenate((standardized, standardized * standardized), axis=1)
        logits = expanded @ self.weights + self.bias
        result = np.empty_like(logits)
        positive = logits >= 0.0
        result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_value = np.exp(logits[~positive])
        result[~positive] = exp_value / (1.0 + exp_value)
        if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
            raise FloatingPointError("event probability is invalid")
        return result

    def score_codes(self, features: np.ndarray) -> np.ndarray:
        return np.rint(255.0 * self.probabilities(features)).astype(np.uint8)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "summary_feature_names": list(SUMMARY_FEATURE_NAMES),
            "standardization_mean": self.standardization_mean.tolist(),
            "standardization_scale": self.standardization_scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feature_transform": "concat(z,z_squared)_without_cross_terms",
            "parameter_count": len(self.weights) + 1,
            "macs_per_update_proxy": len(self.weights),
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ObservedTailEventModel":
        if payload.get("schema_version") != EVENT_MODEL_SCHEMA:
            raise ValueError("event checkpoint schema mismatch")
        return cls(
            standardization_mean=np.asarray(payload["standardization_mean"], dtype=np.float64),
            standardization_scale=np.asarray(payload["standardization_scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
        )


def fit_route_a_posterior_model(
    feature_sequences: Sequence[np.ndarray],
    label_sequences: Sequence[Sequence[str] | np.ndarray],
    *,
    covariance_regularization: float,
    transition_smoothing: float,
) -> RouteAPosteriorModel:
    if not feature_sequences or len(feature_sequences) != len(label_sequences):
        raise ValueError("feature and label sequences must be nonempty and aligned")
    regularization = _positive(covariance_regularization, "covariance_regularization")
    smoothing = _positive(transition_smoothing, "transition_smoothing")
    lookup = {name: index for index, name in enumerate(ROUTE_A_POSTERIOR_CLASSES)}
    feature_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []
    for features, labels in zip(feature_sequences, label_sequences, strict=True):
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(SUMMARY_FEATURE_NAMES):
            raise ValueError("every feature sequence must have the canonical width")
        if len(array) != len(labels) or len(array) < 2 or not np.all(np.isfinite(array)):
            raise ValueError("every feature/label sequence must contain aligned finite rows")
        raw_labels = np.asarray(labels)
        if np.issubdtype(raw_labels.dtype, np.integer):
            indices = np.asarray(raw_labels, dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= len(ROUTE_A_POSTERIOR_CLASSES)):
                raise ValueError("integer Route-A posterior label is out of range")
        else:
            try:
                indices = np.asarray([lookup[str(label)] for label in labels], dtype=np.int64)
            except KeyError as exc:
                raise ValueError(f"unknown Route-A posterior label {exc.args[0]!r}") from exc
        feature_arrays.append(array)
        label_arrays.append(indices)
    all_features = np.vstack(feature_arrays)
    all_labels = np.concatenate(label_arrays)
    counts = np.bincount(all_labels, minlength=len(ROUTE_A_POSTERIOR_CLASSES))
    if np.any(counts < len(SUMMARY_FEATURE_NAMES) + 2):
        raise ValueError("each Route-A class needs more rows than the feature dimension")
    mean = np.mean(all_features, axis=0)
    scale = np.std(all_features, axis=0, ddof=1)
    scale = np.where(scale < 1.0e-8, 1.0, scale)
    standardized = (all_features - mean) / scale
    classes = len(ROUTE_A_POSTERIOR_CLASSES)
    width = len(SUMMARY_FEATURE_NAMES)
    emission_means = np.empty((classes, width), dtype=np.float64)
    emission_covariances = np.empty((classes, width, width), dtype=np.float64)
    for class_index in range(classes):
        selected = standardized[all_labels == class_index]
        emission_means[class_index] = np.mean(selected, axis=0)
        emission_covariances[class_index] = np.cov(selected, rowvar=False, ddof=1) + regularization * np.eye(width)
    emission_precisions = np.linalg.inv(emission_covariances)
    signs, log_determinants = np.linalg.slogdet(emission_covariances)
    if np.any(signs <= 0.0):
        raise ValueError("regularized covariance is not positive definite")
    transition_counts = np.full((classes, classes), smoothing, dtype=np.float64)
    initial_counts = np.full(classes, smoothing, dtype=np.float64)
    for labels in label_arrays:
        initial_counts[labels[0]] += 1.0
        np.add.at(transition_counts, (labels[:-1], labels[1:]), 1.0)
    return RouteAPosteriorModel(
        standardization_mean=mean,
        standardization_scale=scale,
        emission_means=emission_means,
        emission_covariances=emission_covariances,
        emission_precisions=emission_precisions,
        emission_log_determinants=log_determinants,
        transition_matrix=transition_counts / np.sum(transition_counts, axis=1, keepdims=True),
        initial_probabilities=initial_counts / np.sum(initial_counts),
        class_prior_probabilities=(counts + smoothing) / (np.sum(counts) + smoothing * classes),
        covariance_regularization=regularization,
        transition_smoothing=smoothing,
    )


__all__ = [
    "EVENT_MODEL_SCHEMA",
    "MODEL_SCHEMA",
    "ObservedTailEventModel",
    "ROUTE_A_POSTERIOR_CLASSES",
    "RouteAPosteriorModel",
    "fit_route_a_posterior_model",
    "temperature_scale",
]
