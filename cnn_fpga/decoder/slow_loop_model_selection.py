"""Matched-budget causal slow-loop regime estimators for T4.1.1.

All families consume the same bounded history of T3.2.6 observed-window
summaries and return the same four-state posterior.  Hidden regime labels are
training/evaluation targets only; none of the inference APIs accepts truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

import numpy as np

from cnn_fpga.decoder.regime_hmm import (
    REGIME_CLASSES,
    SUMMARY_FEATURE_NAMES,
    GaussianRegimeHMM,
)
# DLEnv links NumPy and torch to different OpenMP runtimes.  Construct the
# protocol constants before importing torch, matching the rest of the physics
# stack's safe import order and avoiding a later process abort during pytest
# collection.
from physics import sbs_error_space as _sbs_error_space  # noqa: F401

try:  # Keep non-neural artifact inspection usable in the base environment.
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised by base-environment collection.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


MODEL_FAMILIES = (
    "causal_tcn",
    "small_gru",
    "gaussian_hmm",
    "diagonal_kalman",
    "exponential_recurrence",
    "run_length_fsm",
)


def _integer(value: object, name: str, minimum: int = 1) -> int:
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


def _finite_matrix(values: object, name: str, width: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != width or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape (n, {width})")
    return result


def _readonly(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SlowLoopSelectionBudget:
    window_cycles: int = 32
    history_windows: int = 8
    update_period_cycles: int = 32
    summary_feature_count: int = len(SUMMARY_FEATURE_NAMES)
    class_count: int = len(REGIME_CLASSES)
    max_macs_per_update: int = 4096
    max_model_and_state_bytes: int = 4096
    max_transient_workspace_bytes: int = 4096
    host_software_latency_ceiling_us: float = 5000.0

    def __post_init__(self) -> None:
        for name in (
            "window_cycles",
            "history_windows",
            "update_period_cycles",
            "summary_feature_count",
            "class_count",
            "max_macs_per_update",
            "max_model_and_state_bytes",
            "max_transient_workspace_bytes",
        ):
            minimum = 2 if name == "history_windows" else 1
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.window_cycles != self.update_period_cycles:
            raise ValueError("one slow-loop update must correspond to one non-overlapping window")
        if self.summary_feature_count != len(SUMMARY_FEATURE_NAMES):
            raise ValueError("summary_feature_count must match the canonical summary schema")
        if self.class_count != len(REGIME_CLASSES):
            raise ValueError("class_count must match REGIME_CLASSES")
        object.__setattr__(
            self,
            "host_software_latency_ceiling_us",
            _positive(self.host_software_latency_ceiling_us, "host_software_latency_ceiling_us"),
        )


@dataclass(frozen=True)
class ModelResourceProfile:
    family: str
    trainable_or_fitted_float_values: int
    runtime_state_float_values: int
    model_and_state_bytes: int
    transient_workspace_float_values: int
    transient_workspace_bytes: int
    macs_per_update_proxy: int

    def __post_init__(self) -> None:
        if self.family not in MODEL_FAMILIES:
            raise ValueError("unknown model family")
        for name in (
            "trainable_or_fitted_float_values",
            "runtime_state_float_values",
            "model_and_state_bytes",
            "transient_workspace_float_values",
            "transient_workspace_bytes",
            "macs_per_update_proxy",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 0))
        expected = 4 * (
            self.trainable_or_fitted_float_values + self.runtime_state_float_values
        )
        if self.model_and_state_bytes != expected:
            raise ValueError("model_and_state_bytes must count float32 model and runtime state")
        if self.transient_workspace_bytes != 4 * self.transient_workspace_float_values:
            raise ValueError("transient_workspace_bytes must count float32 scratch values")

    def within(self, budget: SlowLoopSelectionBudget) -> bool:
        if not isinstance(budget, SlowLoopSelectionBudget):
            raise TypeError("budget must be SlowLoopSelectionBudget")
        return (
            self.model_and_state_bytes <= budget.max_model_and_state_bytes
            and self.transient_workspace_bytes <= budget.max_transient_workspace_bytes
            and self.macs_per_update_proxy <= budget.max_macs_per_update
        )


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        width = len(SUMMARY_FEATURE_NAMES)
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if mean.shape != (width,) or scale.shape != (width,):
            raise ValueError("standardizer arrays have the wrong shape")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("standardizer arrays must be finite with positive scale")
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "scale", _readonly(scale))

    @classmethod
    def fit(cls, sequences: Sequence[np.ndarray]) -> "FeatureStandardizer":
        if not sequences:
            raise ValueError("at least one training sequence is required")
        rows = np.vstack(
            [_finite_matrix(item, "training sequence", len(SUMMARY_FEATURE_NAMES)) for item in sequences]
        )
        scale = np.std(rows, axis=0, ddof=1)
        return cls(np.mean(rows, axis=0), np.where(scale < 1.0e-8, 1.0, scale))

    def transform(self, values: object) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != (len(SUMMARY_FEATURE_NAMES),) or not np.all(np.isfinite(array)):
            raise ValueError("values must be finite with canonical summary width")
        return (array - self.mean) / self.scale


def bounded_histories(
    feature_sequences: Sequence[np.ndarray],
    *,
    history_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return histories plus sequence/local indices without crossing trajectories."""

    history = _integer(history_windows, "history_windows", 2)
    rows: list[np.ndarray] = []
    sequence_indices: list[int] = []
    local_indices: list[int] = []
    for sequence_index, values in enumerate(feature_sequences):
        array = _finite_matrix(values, "feature sequence", len(SUMMARY_FEATURE_NAMES))
        if len(array) < history:
            raise ValueError("every feature sequence must cover the full history")
        for stop in range(history, len(array) + 1):
            rows.append(np.array(array[stop - history : stop], copy=True))
            sequence_indices.append(sequence_index)
            local_indices.append(stop - 1)
    result = np.asarray(rows, dtype=np.float64)
    return (
        result,
        np.asarray(sequence_indices, dtype=np.int64),
        np.asarray(local_indices, dtype=np.int64),
    )


def labels_for_histories(
    label_sequences: Sequence[Sequence[str]],
    sequence_indices: np.ndarray,
    local_indices: np.ndarray,
) -> np.ndarray:
    if sequence_indices.shape != local_indices.shape or sequence_indices.ndim != 1:
        raise ValueError("history indices must be aligned vectors")
    lookup = {name: index for index, name in enumerate(REGIME_CLASSES)}
    output = []
    for sequence, local in zip(sequence_indices, local_indices, strict=True):
        try:
            output.append(lookup[label_sequences[int(sequence)][int(local)]])
        except (IndexError, KeyError) as exc:
            raise ValueError("label sequences do not align with histories") from exc
    return np.asarray(output, dtype=np.int64)


def softmax_logits(logits: object, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(REGIME_CLASSES) or not np.all(np.isfinite(values)):
        raise ValueError("logits must be a finite class matrix")
    actual_temperature = _positive(temperature, "temperature")
    scaled = values / actual_temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


def temper_posterior(posterior: object, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(posterior, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(REGIME_CLASSES):
        raise ValueError("posterior must be a class matrix")
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("posterior entries must be finite and positive")
    if not np.allclose(np.sum(values, axis=1), 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("posterior rows must sum to one")
    return softmax_logits(np.log(values), temperature)


class RollingGaussianHMMAdapter:
    """Exact last-H filter using cached emissions instead of raw-window replay.

    Only the newest 14-feature emission is evaluated at each update.  The
    four-value emission vectors for the bounded history are cached, then the
    inexpensive 4x4 forward recursion is replayed.  This makes the registered
    MAC/state profile executable rather than a paper-only estimate.
    """

    def __init__(
        self,
        model: GaussianRegimeHMM,
        *,
        history_windows: int,
        temperature: float = 1.0,
    ) -> None:
        if not isinstance(model, GaussianRegimeHMM):
            raise TypeError("model must be GaussianRegimeHMM")
        self.model = model
        self.history_windows = _integer(history_windows, "history_windows", 2)
        self.temperature = _positive(temperature, "temperature")
        self._emissions: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return len(self._emissions) == self.history_windows

    @property
    def cached_emission_count(self) -> int:
        return len(self._emissions)

    def reset(self) -> None:
        self._emissions.clear()

    def step(self, summary_features: object) -> np.ndarray:
        values = np.asarray(summary_features, dtype=np.float64)
        if values.shape != (len(SUMMARY_FEATURE_NAMES),) or not np.all(np.isfinite(values)):
            raise ValueError("summary_features must be one finite canonical summary row")
        emission = self.model.emission_log_likelihood(values[None, :])[0]
        self._emissions.append(np.array(emission, copy=True))
        if len(self._emissions) > self.history_windows:
            del self._emissions[0]
        previous = np.array(self.model.initial_probabilities, copy=True)
        for cached in self._emissions:
            prediction = previous @ self.model.transition_matrix
            logits = np.log(prediction) + cached
            previous = softmax_logits(logits[None, :])[0]
        return temper_posterior(previous[None, :], self.temperature)[0]


@dataclass(frozen=True)
class DiagonalGaussianHead:
    means: np.ndarray
    variances: np.ndarray
    priors: np.ndarray

    def __post_init__(self) -> None:
        classes = len(REGIME_CLASSES)
        means = np.asarray(self.means, dtype=np.float64)
        variances = np.asarray(self.variances, dtype=np.float64)
        priors = np.asarray(self.priors, dtype=np.float64)
        if means.ndim != 2 or means.shape[0] != classes:
            raise ValueError("means must have one row per regime")
        if variances.shape != means.shape or np.any(variances <= 0.0):
            raise ValueError("variances must be positive and align with means")
        if priors.shape != (classes,) or np.any(priors <= 0.0) or not np.isclose(
            np.sum(priors), 1.0, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError("priors must be positive and normalized")
        if not all(np.all(np.isfinite(item)) for item in (means, variances, priors)):
            raise ValueError("head parameters must be finite")
        object.__setattr__(self, "means", _readonly(means))
        object.__setattr__(self, "variances", _readonly(variances))
        object.__setattr__(self, "priors", _readonly(priors))

    @classmethod
    def fit(
        cls,
        states: object,
        labels: object,
        *,
        variance_floor: float,
    ) -> "DiagonalGaussianHead":
        values = np.asarray(states, dtype=np.float64)
        truth = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or len(values) != len(truth) or not np.all(np.isfinite(values)):
            raise ValueError("states and labels must be aligned finite arrays")
        floor = _positive(variance_floor, "variance_floor")
        means = []
        variances = []
        counts = []
        for index in range(len(REGIME_CLASSES)):
            selected = values[truth == index]
            if len(selected) < 2:
                raise ValueError("each regime requires at least two fitted states")
            means.append(np.mean(selected, axis=0))
            variances.append(np.var(selected, axis=0, ddof=1) + floor)
            counts.append(len(selected) + 1.0)
        priors = np.asarray(counts, dtype=np.float64)
        priors /= np.sum(priors)
        return cls(np.asarray(means), np.asarray(variances), priors)

    @property
    def parameter_count(self) -> int:
        return int(self.means.size + self.variances.size + self.priors.size)

    def logits(self, states: object) -> np.ndarray:
        values = np.asarray(states, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.means.shape[1] or not np.all(np.isfinite(values)):
            raise ValueError("states have the wrong shape")
        difference = values[:, None, :] - self.means[None, :, :]
        return -0.5 * np.sum(
            difference * difference / self.variances[None, :, :]
            + np.log(self.variances[None, :, :]),
            axis=2,
        ) + np.log(self.priors)[None, :]


def exponential_states(histories: object, decay: float) -> np.ndarray:
    values = np.asarray(histories, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(SUMMARY_FEATURE_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError("histories have the wrong shape")
    actual_decay = _positive(decay, "decay", allow_zero=True)
    if actual_decay >= 1.0:
        raise ValueError("decay must be less than one")
    state = np.array(values[:, 0, :], copy=True)
    for index in range(1, values.shape[1]):
        state = actual_decay * state + (1.0 - actual_decay) * values[:, index, :]
    return state


def diagonal_kalman_states(histories: object, process_variance: float, measurement_variance: float) -> np.ndarray:
    values = np.asarray(histories, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(SUMMARY_FEATURE_NAMES) or not np.all(np.isfinite(values)):
        raise ValueError("histories have the wrong shape")
    process = _positive(process_variance, "process_variance")
    measurement = _positive(measurement_variance, "measurement_variance")
    mean = np.array(values[:, 0, :], copy=True)
    variance = np.full_like(mean, measurement)
    for index in range(1, values.shape[1]):
        predicted_variance = variance + process
        gain = predicted_variance / (predicted_variance + measurement)
        mean = mean + gain * (values[:, index, :] - mean)
        variance = (1.0 - gain) * predicted_variance
    return mean


def run_length_fsm_posterior(
    instantaneous: object,
    *,
    enter_run: int,
    confidence: float,
) -> np.ndarray:
    values = np.asarray(instantaneous, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(REGIME_CLASSES):
        raise ValueError("instantaneous posterior must have shape (n, history, classes)")
    if np.any(values <= 0.0) or not np.allclose(np.sum(values, axis=2), 1.0, atol=1.0e-10):
        raise ValueError("instantaneous posterior must be finite, positive and normalized")
    threshold = _integer(enter_run, "enter_run")
    actual_confidence = _positive(confidence, "confidence")
    if not 1.0 / len(REGIME_CLASSES) < actual_confidence < 1.0:
        raise ValueError("confidence must lie between uniform and one")
    output = np.empty((len(values), len(REGIME_CLASSES)), dtype=np.float64)
    for row_index, history in enumerate(values):
        active = int(np.argmax(history[0]))
        candidate = active
        run = 0
        for posterior in history[1:]:
            proposed = int(np.argmax(posterior))
            if proposed == active:
                candidate = active
                run = 0
            elif proposed == candidate:
                run += 1
            else:
                candidate = proposed
                run = 1
            if run >= threshold:
                active = candidate
                run = 0
        base = np.array(history[-1], copy=True)
        # Sum the non-active entries directly: ``1 - p_active`` loses all
        # meaningful digits for highly confident diagonal-Gaussian emissions.
        others = float(np.sum(np.delete(base, active), dtype=np.float64))
        result = np.zeros(len(REGIME_CLASSES), dtype=np.float64)
        result[active] = actual_confidence
        if others <= 1.0e-15:
            result += (1.0 - actual_confidence) / (len(REGIME_CLASSES) - 1)
            result[active] = actual_confidence
        else:
            for index in range(len(REGIME_CLASSES)):
                if index != active:
                    result[index] = (1.0 - actual_confidence) * base[index] / others
        result /= np.sum(result, dtype=np.float64)
        output[row_index] = result
    return output


if nn is not None:

    class CausalTCN(nn.Module):
        def __init__(self, input_features: int = len(SUMMARY_FEATURE_NAMES), channels: int = 7) -> None:
            super().__init__()
            if input_features != len(SUMMARY_FEATURE_NAMES) or channels != 7:
                raise ValueError("T4.1.1 freezes the 14x7 causal TCN architecture")
            self.conv1 = nn.Conv1d(input_features, channels, kernel_size=3, dilation=1)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=2)
            self.head = nn.Linear(channels, len(REGIME_CLASSES))

        @property
        def parameter_count(self) -> int:
            return int(sum(parameter.numel() for parameter in self.parameters()))

        def forward(self, histories: Any) -> Any:
            if histories.ndim != 3 or histories.shape[2] != len(SUMMARY_FEATURE_NAMES):
                raise ValueError("histories must have shape (batch, history, features)")
            values = histories.transpose(1, 2)
            values = F.relu(self.conv1(F.pad(values, (2, 0))))
            values = F.relu(self.conv2(F.pad(values, (4, 0))))
            return self.head(values[:, :, -1])


    class SmallGRU(nn.Module):
        def __init__(self, input_features: int = len(SUMMARY_FEATURE_NAMES), hidden_size: int = 5) -> None:
            super().__init__()
            if input_features != len(SUMMARY_FEATURE_NAMES) or hidden_size != 5:
                raise ValueError("T4.1.1 freezes the 14x5 small-GRU architecture")
            self.gru = nn.GRU(input_features, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, len(REGIME_CLASSES))

        @property
        def parameter_count(self) -> int:
            return int(sum(parameter.numel() for parameter in self.parameters()))

        def forward(self, histories: Any) -> Any:
            if histories.ndim != 3 or histories.shape[2] != len(SUMMARY_FEATURE_NAMES):
                raise ValueError("histories must have shape (batch, history, features)")
            output, _ = self.gru(histories)
            return self.head(output[:, -1, :])

else:  # pragma: no cover - import contract in the base environment.
    CausalTCN = None  # type: ignore[assignment]
    SmallGRU = None  # type: ignore[assignment]


def resource_profiles(budget: SlowLoopSelectionBudget | None = None) -> dict[str, ModelResourceProfile]:
    contract = SlowLoopSelectionBudget() if budget is None else budget
    if not isinstance(contract, SlowLoopSelectionBudget):
        raise TypeError("budget must be SlowLoopSelectionBudget")
    buffer_floats = contract.history_windows * contract.summary_feature_count
    history = contract.history_windows
    standardizer_floats = 2 * contract.summary_feature_count
    # Neural parameter counts are analytic so artifact inspection does not require torch.
    tcn_parameters = 7 * 14 * 3 + 7 + 7 * 7 * 3 + 7 + 7 * 4 + 4
    # torch.nn.GRU stores independent input/hidden biases for each of 3 gates.
    gru_parameters = 3 * (14 * 5 + 5 * 5 + 2 * 5) + 5 * 4 + 4
    def _profile(
        family: str,
        parameters: int,
        runtime: int,
        transient: int,
        macs: int,
    ) -> ModelResourceProfile:
        return ModelResourceProfile(
            family,
            parameters,
            runtime,
            4 * (parameters + runtime),
            transient,
            4 * transient,
            macs,
        )
    profiles = {
        "causal_tcn": _profile(
            "causal_tcn", standardizer_floats + tcn_parameters, buffer_floats,
            2 * history * 7 + 4,
            history * 7 * 14 * 3 + history * 7 * 7 * 3 + 7 * 4,
        ),
        "small_gru": _profile(
            "small_gru", standardizer_floats + gru_parameters, buffer_floats,
            6 * 5 + 5 + 4,
            history * 3 * (14 * 5 + 5 * 5) + 5 * 4,
        ),
        "gaussian_hmm": _profile(
            "gaussian_hmm", 896, history * len(REGIME_CLASSES) + len(REGIME_CLASSES),
            14 + 3 * len(REGIME_CLASSES),
            4 * 14 * 14 + history * 4 * 4 + 14,
        ),
        "diagonal_kalman": _profile(
            "diagonal_kalman", standardizer_floats + 28 + 56 + 56 + 4,
            buffer_floats + 28, 3 * 14 + 4,
            history * 14 * 8 + 4 * 14 * 3,
        ),
        "exponential_recurrence": _profile(
            "exponential_recurrence", standardizer_floats + 1 + 56 + 56 + 4,
            buffer_floats + 14, 14 + 4,
            history * 14 * 3 + 4 * 14 * 3,
        ),
        "run_length_fsm": _profile(
            "run_length_fsm", standardizer_floats + 56 + 56 + 4 + 2,
            buffer_floats + 3, 14 + 4,
            history * (4 * 14 * 3 + 8),
        ),
    }
    return profiles


__all__ = [
    "MODEL_FAMILIES",
    "SlowLoopSelectionBudget",
    "ModelResourceProfile",
    "FeatureStandardizer",
    "DiagonalGaussianHead",
    "CausalTCN",
    "SmallGRU",
    "bounded_histories",
    "labels_for_histories",
    "softmax_logits",
    "temper_posterior",
    "RollingGaussianHMMAdapter",
    "exponential_states",
    "diagonal_kalman_states",
    "run_length_fsm_posterior",
    "resource_profiles",
]
