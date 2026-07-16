"""Strict-split multi-objective loss and calibration for the slow loop.

This module owns metric/calibration semantics, not model fitting.  Training
records determine robust scales, validation records determine calibration
parameters, and evaluation records are consumed only after all parameters are
frozen.  Simulator truth is permitted in :class:`CalibrationRecord` because
the record is an offline evaluator object; it is never a deployable input.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite, log
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .hybrid_state_output import CONTINUOUS_PARAMETER_NAMES
from .regime_hmm import REGIME_CLASSES


OBJECTIVE_NAMES = (
    "state_estimation",
    "oracle_gap",
    "regime_detection",
    "uncertainty_calibration",
    "false_fallback",
    "update_cost",
)
SPLITS = ("training", "validation", "evaluation")


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(values: object, length: int, name: str, *, positive: bool = False) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive")
    return tuple(float(value) for value in array)


def _hash_ids(records: Sequence["CalibrationRecord"]) -> str:
    digest = hashlib.sha256()
    for record_id in sorted(record.record_id for record in records):
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class MultiObjectiveWeights:
    state_estimation: float = 0.24
    oracle_gap: float = 0.22
    regime_detection: float = 0.16
    uncertainty_calibration: float = 0.16
    false_fallback: float = 0.12
    update_cost: float = 0.10

    def __post_init__(self) -> None:
        values = np.asarray([getattr(self, name) for name in OBJECTIVE_NAMES], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("all objective weights must be finite and positive")
        if not np.isclose(np.sum(values), 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("objective weights must sum to one")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in OBJECTIVE_NAMES}

    def without(self, objective: str) -> Mapping[str, float]:
        if objective not in OBJECTIVE_NAMES:
            raise ValueError(f"unknown objective {objective!r}")
        values = self.as_dict()
        values[objective] = 0.0
        denominator = sum(values.values())
        return MappingProxyType({name: value / denominator for name, value in values.items()})


@dataclass(frozen=True)
class CalibrationRecord:
    """One future-aligned offline calibration/evaluation example."""

    record_id: str
    split: str
    seed: int
    prediction: tuple[float, ...]
    target: tuple[float, ...]
    uncertainty_standard_errors: tuple[float, ...]
    regime_probabilities: tuple[float, ...]
    regime_label: str
    candidate_failures: int
    oracle_failures: int
    oracle_trials: int
    fallback_score: float
    fallback_required: bool
    update_cost: float
    scope: str = "offline_future_aligned_calibration_record"

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise ValueError("record_id must be nonempty")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise TypeError("seed must be a nonnegative integer")
        object.__setattr__(
            self,
            "prediction",
            _vector(self.prediction, len(CONTINUOUS_PARAMETER_NAMES), "prediction"),
        )
        object.__setattr__(
            self,
            "target",
            _vector(self.target, len(CONTINUOUS_PARAMETER_NAMES), "target"),
        )
        object.__setattr__(
            self,
            "uncertainty_standard_errors",
            _vector(
                self.uncertainty_standard_errors,
                len(CONTINUOUS_PARAMETER_NAMES),
                "uncertainty_standard_errors",
                positive=True,
            ),
        )
        probabilities = np.asarray(self.regime_probabilities, dtype=np.float64)
        if probabilities.shape != (len(REGIME_CLASSES),) or not np.all(np.isfinite(probabilities)):
            raise ValueError("regime_probabilities has the wrong shape or nonfinite values")
        if np.any(probabilities <= 0.0) or not np.isclose(
            np.sum(probabilities), 1.0, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("regime_probabilities must be positive and normalized")
        object.__setattr__(self, "regime_probabilities", tuple(float(value) for value in probabilities))
        if self.regime_label not in REGIME_CLASSES:
            raise ValueError(f"regime_label must be one of {REGIME_CLASSES}")
        for name in ("candidate_failures", "oracle_failures", "oracle_trials"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"{name} must be a nonnegative integer")
        if self.oracle_trials < 1:
            raise ValueError("oracle_trials must be positive")
        if max(self.candidate_failures, self.oracle_failures) > self.oracle_trials:
            raise ValueError("failure counts cannot exceed oracle_trials")
        fallback_score = _finite(self.fallback_score, "fallback_score")
        if not 0.0 <= fallback_score <= 1.0:
            raise ValueError("fallback_score must lie in [0, 1]")
        object.__setattr__(self, "fallback_score", fallback_score)
        if not isinstance(self.fallback_required, (bool, np.bool_)):
            raise TypeError("fallback_required must be boolean")
        object.__setattr__(self, "fallback_required", bool(self.fallback_required))
        update_cost = _finite(self.update_cost, "update_cost")
        if update_cost < 0.0:
            raise ValueError("update_cost must be nonnegative")
        object.__setattr__(self, "update_cost", update_cost)
        if self.scope != "offline_future_aligned_calibration_record":
            raise ValueError("record scope must remain offline and future-aligned")


@dataclass(frozen=True)
class TrainingNormalizers:
    state_scales: tuple[float, ...]
    objective_scales: Mapping[str, float]
    training_record_ids_sha256: str
    training_seeds: tuple[int, ...]
    source_split: str = "training_only"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_scales",
            _vector(self.state_scales, len(CONTINUOUS_PARAMETER_NAMES), "state_scales", positive=True),
        )
        if set(self.objective_scales) != set(OBJECTIVE_NAMES):
            raise ValueError("objective_scales must cover every objective exactly once")
        scales = {name: _finite(self.objective_scales[name], name) for name in OBJECTIVE_NAMES}
        if any(value <= 0.0 for value in scales.values()):
            raise ValueError("objective scales must be positive")
        object.__setattr__(self, "objective_scales", MappingProxyType(scales))
        if self.source_split != "training_only":
            raise ValueError("normalizers must be training-only")


@dataclass(frozen=True)
class FrozenCalibration:
    regime_temperature: float
    regime_uniform_mix: float
    uncertainty_scale: float
    fallback_threshold: float
    minimum_unsafe_recall: float
    training_record_ids_sha256: str
    validation_record_ids_sha256: str
    training_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    selection_scope: str = "validation_only_after_training_normalizers"

    def __post_init__(self) -> None:
        for name in ("regime_temperature", "uncertainty_scale"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        uniform_mix = _finite(self.regime_uniform_mix, "regime_uniform_mix")
        if not 0.0 <= uniform_mix <= 1.0:
            raise ValueError("regime_uniform_mix must lie in [0, 1]")
        object.__setattr__(self, "regime_uniform_mix", uniform_mix)
        for name in ("fallback_threshold", "minimum_unsafe_recall"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        if set(self.training_seeds) & set(self.validation_seeds):
            raise ValueError("training and validation seeds must be disjoint")
        if self.selection_scope != "validation_only_after_training_normalizers":
            raise ValueError("calibration selection scope is invalid")


def _require_split(records: Sequence[CalibrationRecord], split: str) -> tuple[CalibrationRecord, ...]:
    result = tuple(records)
    if not result or any(record.split != split for record in result):
        raise ValueError(f"records must be nonempty and exclusively {split!r}")
    ids = [record.record_id for record in result]
    if len(ids) != len(set(ids)):
        raise ValueError("record IDs must be unique")
    return result


def _tempered(probabilities: Sequence[float], temperature: float) -> np.ndarray:
    logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-15, 1.0))
    logits /= temperature
    logits -= np.max(logits)
    values = np.exp(logits)
    return values / np.sum(values)


def _regime_scores(
    records: Sequence[CalibrationRecord], temperature: float, uniform_mix: float = 0.0
) -> tuple[float, float, float]:
    labels = {name: index for index, name in enumerate(REGIME_CLASSES)}
    nll = []
    brier = []
    correct = []
    for record in records:
        probability = _tempered(record.regime_probabilities, temperature)
        probability = (1.0 - uniform_mix) * probability + uniform_mix / len(REGIME_CLASSES)
        target = labels[record.regime_label]
        one_hot = np.eye(len(REGIME_CLASSES), dtype=np.float64)[target]
        nll.append(-log(max(float(probability[target]), 1.0e-15)))
        brier.append(float(np.sum((probability - one_hot) ** 2)))
        correct.append(int(np.argmax(probability) == target))
    return float(np.mean(nll)), float(np.mean(brier)), float(np.mean(correct))


def _uncertainty_scores(
    records: Sequence[CalibrationRecord],
    state_scales: Sequence[float],
    uncertainty_scale: float,
) -> tuple[float, float, float]:
    prediction = np.asarray([record.prediction for record in records])
    target = np.asarray([record.target for record in records])
    standard_error = np.asarray([record.uncertainty_standard_errors for record in records])
    scales = np.asarray(state_scales, dtype=np.float64)
    sigma = standard_error * uncertainty_scale
    normalized_sigma = sigma / scales[None, :]
    normalized_error = (target - prediction) / scales[None, :]
    z = normalized_error / normalized_sigma
    nll = 0.5 * (z * z + np.log(2.0 * np.pi * normalized_sigma * normalized_sigma))
    coverage68 = float(np.mean(np.abs(z) <= 1.0))
    coverage95 = float(np.mean(np.abs(z) <= 1.959963984540054))
    return float(np.mean(nll)), coverage68, coverage95


def _fallback_scores(
    records: Sequence[CalibrationRecord], threshold: float
) -> tuple[float, float, float, float]:
    required = np.asarray([record.fallback_required for record in records], dtype=bool)
    selected = np.asarray([record.fallback_score >= threshold for record in records], dtype=bool)
    safe = ~required
    false_rate = float(np.mean(selected[safe])) if np.any(safe) else 0.0
    miss_rate = float(np.mean(~selected[required])) if np.any(required) else 0.0
    recall = 1.0 - miss_rate
    # Missing a genuinely unsafe future is deliberately more expensive than a
    # conservative false fallback; this prevents the all-clear degenerate fit.
    cost = false_rate + 4.0 * miss_rate
    return cost, false_rate, miss_rate, recall


def _raw_objectives(
    records: Sequence[CalibrationRecord],
    state_scales: Sequence[float],
    *,
    regime_temperature: float,
    regime_uniform_mix: float,
    uncertainty_scale: float,
    fallback_threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    prediction = np.asarray([record.prediction for record in records])
    target = np.asarray([record.target for record in records])
    scales = np.asarray(state_scales, dtype=np.float64)
    error = np.abs((prediction - target) / scales[None, :])
    # Smooth-L1 is quadratic near zero and linear in the heavy synthetic tail.
    state = float(np.mean(np.where(error < 1.0, 0.5 * error * error, error - 0.5)))
    gap = float(
        np.mean(
            [
                abs(record.candidate_failures - record.oracle_failures) / record.oracle_trials
                for record in records
            ]
        )
    )
    regime_nll, regime_brier, accuracy = _regime_scores(
        records, regime_temperature, regime_uniform_mix
    )
    uncertainty_nll, coverage68, coverage95 = _uncertainty_scores(
        records, state_scales, uncertainty_scale
    )
    fallback, false_rate, miss_rate, recall = _fallback_scores(records, fallback_threshold)
    update = float(np.mean([record.update_cost for record in records]))
    raw = {
        "state_estimation": state,
        "oracle_gap": gap,
        "regime_detection": regime_nll + regime_brier,
        "uncertainty_calibration": uncertainty_nll,
        "false_fallback": fallback,
        "update_cost": update,
    }
    diagnostics = {
        "regime_nll": regime_nll,
        "regime_brier": regime_brier,
        "regime_accuracy": accuracy,
        "uncertainty_marginal_68_coverage": coverage68,
        "uncertainty_marginal_95_coverage": coverage95,
        "false_fallback_rate": false_rate,
        "missed_required_fallback_rate": miss_rate,
        "required_fallback_recall": recall,
    }
    return raw, diagnostics


def fit_training_normalizers(records: Sequence[CalibrationRecord]) -> TrainingNormalizers:
    training = _require_split(records, "training")
    targets = np.asarray([record.target for record in training], dtype=np.float64)
    median = np.median(targets, axis=0)
    mad = 1.4826 * np.median(np.abs(targets - median[None, :]), axis=0)
    span = np.quantile(targets, 0.95, axis=0) - np.quantile(targets, 0.05, axis=0)
    state_scales = np.maximum.reduce((mad, span / 3.29, np.full(len(median), 1.0e-4)))
    raw, _ = _raw_objectives(
        training,
        state_scales,
        regime_temperature=1.0,
        regime_uniform_mix=0.0,
        uncertainty_scale=1.0,
        fallback_threshold=0.58,
    )
    objective_scales = {name: max(abs(value), 1.0e-4) for name, value in raw.items()}
    return TrainingNormalizers(
        state_scales=tuple(float(value) for value in state_scales),
        objective_scales=objective_scales,
        training_record_ids_sha256=_hash_ids(training),
        training_seeds=tuple(sorted({record.seed for record in training})),
    )


def fit_validation_calibration(
    validation_records: Sequence[CalibrationRecord],
    normalizers: TrainingNormalizers,
    *,
    regime_temperature_grid: Sequence[float] = (
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
    ),
    regime_uniform_mix_grid: Sequence[float] = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0),
    uncertainty_scale_grid: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    fallback_threshold_grid: Sequence[float] = tuple(np.linspace(0.25, 0.95, 29)),
    minimum_unsafe_recall: float = 0.90,
) -> FrozenCalibration:
    validation = _require_split(validation_records, "validation")
    validation_seeds = tuple(sorted({record.seed for record in validation}))
    if set(normalizers.training_seeds) & set(validation_seeds):
        raise ValueError("training and validation seeds overlap")
    temperature_candidates = tuple(_finite(value, "regime temperature") for value in regime_temperature_grid)
    uniform_mix_candidates = tuple(_finite(value, "regime uniform mix") for value in regime_uniform_mix_grid)
    uncertainty_candidates = tuple(_finite(value, "uncertainty scale") for value in uncertainty_scale_grid)
    threshold_candidates = tuple(_finite(value, "fallback threshold") for value in fallback_threshold_grid)
    if not temperature_candidates or min(temperature_candidates) <= 0.0:
        raise ValueError("regime temperature grid must be nonempty and positive")
    if (
        not uniform_mix_candidates
        or min(uniform_mix_candidates) < 0.0
        or max(uniform_mix_candidates) > 1.0
    ):
        raise ValueError("regime uniform mix grid must lie in [0, 1]")
    if not uncertainty_candidates or min(uncertainty_candidates) <= 0.0:
        raise ValueError("uncertainty scale grid must be nonempty and positive")
    if not threshold_candidates or min(threshold_candidates) < 0.0 or max(threshold_candidates) > 1.0:
        raise ValueError("fallback threshold grid must lie in [0, 1]")
    recall_floor = _finite(minimum_unsafe_recall, "minimum_unsafe_recall")
    if not 0.0 <= recall_floor <= 1.0:
        raise ValueError("minimum_unsafe_recall must lie in [0, 1]")

    temperature, uniform_mix = min(
        (
            (temperature_value, mix_value)
            for temperature_value in temperature_candidates
            for mix_value in uniform_mix_candidates
        ),
        key=lambda values: (
            _regime_scores(validation, values[0], values[1])[0],
            values[1],
            values[0],
        ),
    )
    uncertainty_scale = min(
        uncertainty_candidates,
        key=lambda value: (_uncertainty_scores(validation, normalizers.state_scales, value)[0], value),
    )
    feasible = [
        value
        for value in threshold_candidates
        if _fallback_scores(validation, value)[3] + 1.0e-12 >= recall_floor
    ]
    if not feasible:
        # Fail safe: use the threshold with maximum recall, not the deceptively
        # low false-fallback threshold with missed unsafe futures.
        feasible = list(threshold_candidates)
        fallback_threshold = min(
            feasible,
            key=lambda value: (-_fallback_scores(validation, value)[3], _fallback_scores(validation, value)[0], value),
        )
    else:
        fallback_threshold = min(
            feasible,
            key=lambda value: (_fallback_scores(validation, value)[0], -value),
        )
    return FrozenCalibration(
        regime_temperature=temperature,
        regime_uniform_mix=uniform_mix,
        uncertainty_scale=uncertainty_scale,
        fallback_threshold=fallback_threshold,
        minimum_unsafe_recall=recall_floor,
        training_record_ids_sha256=normalizers.training_record_ids_sha256,
        validation_record_ids_sha256=_hash_ids(validation),
        training_seeds=normalizers.training_seeds,
        validation_seeds=validation_seeds,
    )


def evaluate_multiobjective_loss(
    records: Sequence[CalibrationRecord],
    normalizers: TrainingNormalizers,
    calibration: FrozenCalibration,
    weights: MultiObjectiveWeights | None = None,
) -> dict[str, object]:
    evaluation = _require_split(records, "evaluation")
    evaluation_seeds = tuple(sorted({record.seed for record in evaluation}))
    if set(evaluation_seeds) & (set(calibration.training_seeds) | set(calibration.validation_seeds)):
        raise ValueError("evaluation seeds overlap training or validation")
    if calibration.training_record_ids_sha256 != normalizers.training_record_ids_sha256:
        raise ValueError("calibration and normalizers have different training provenance")
    actual_weights = MultiObjectiveWeights() if weights is None else weights
    if not isinstance(actual_weights, MultiObjectiveWeights):
        raise TypeError("weights must be MultiObjectiveWeights")
    raw, diagnostics = _raw_objectives(
        evaluation,
        normalizers.state_scales,
        regime_temperature=calibration.regime_temperature,
        regime_uniform_mix=calibration.regime_uniform_mix,
        uncertainty_scale=calibration.uncertainty_scale,
        fallback_threshold=calibration.fallback_threshold,
    )
    normalized = {
        name: raw[name] / normalizers.objective_scales[name] for name in OBJECTIVE_NAMES
    }
    weighted = {
        name: normalized[name] * actual_weights.as_dict()[name] for name in OBJECTIVE_NAMES
    }
    total = float(sum(weighted.values()))
    ablations = {}
    for omitted in OBJECTIVE_NAMES:
        ablated_weights = actual_weights.without(omitted)
        ablated_total = float(sum(normalized[name] * ablated_weights[name] for name in OBJECTIVE_NAMES))
        ablations[omitted] = {
            "evaluation_total": ablated_total,
            "delta_from_full": ablated_total - total,
            "omitted_raw": raw[omitted],
            "omitted_normalized": normalized[omitted],
            "interpretation": "frozen-output_leave_one_objective_out_not_retrained_causal_ablation",
        }
    return {
        "record_count": len(evaluation),
        "record_ids_sha256": _hash_ids(evaluation),
        "evaluation_seeds": list(evaluation_seeds),
        "raw_objectives": raw,
        "normalized_objectives": normalized,
        "weights": actual_weights.as_dict(),
        "weighted_objectives": weighted,
        "total_loss": total,
        "diagnostics": diagnostics,
        "leave_one_objective_out": ablations,
        "selection_provenance": {
            "normalizers": normalizers.source_split,
            "calibration": calibration.selection_scope,
            "evaluation_used_for_selection": False,
        },
    }


def score_calibration_records(
    records: Sequence[CalibrationRecord],
    normalizers: TrainingNormalizers,
    calibration: FrozenCalibration,
) -> dict[str, object]:
    """Score one homogeneous split without changing frozen parameters."""

    values = tuple(records)
    if not values:
        raise ValueError("records must be nonempty")
    split = values[0].split
    homogeneous = _require_split(values, split)
    raw, diagnostics = _raw_objectives(
        homogeneous,
        normalizers.state_scales,
        regime_temperature=calibration.regime_temperature,
        regime_uniform_mix=calibration.regime_uniform_mix,
        uncertainty_scale=calibration.uncertainty_scale,
        fallback_threshold=calibration.fallback_threshold,
    )
    return {
        "split": split,
        "record_count": len(homogeneous),
        "record_ids_sha256": _hash_ids(homogeneous),
        "raw_objectives": raw,
        "diagnostics": diagnostics,
    }


def calibration_manifest(
    normalizers: TrainingNormalizers,
    calibration: FrozenCalibration,
    weights: MultiObjectiveWeights | None = None,
) -> dict[str, object]:
    actual_weights = MultiObjectiveWeights() if weights is None else weights
    payload = {
        "schema_version": "t4.1.4-multiobjective-calibration-v1",
        "continuous_parameter_order": list(CONTINUOUS_PARAMETER_NAMES),
        "regime_class_order": list(REGIME_CLASSES),
        "objective_order": list(OBJECTIVE_NAMES),
        "weights": actual_weights.as_dict(),
        "state_scales": list(normalizers.state_scales),
        "objective_scales": dict(normalizers.objective_scales),
        "regime_temperature": calibration.regime_temperature,
        "regime_uniform_mix": calibration.regime_uniform_mix,
        "uncertainty_scale": calibration.uncertainty_scale,
        "fallback_threshold": calibration.fallback_threshold,
        "minimum_unsafe_recall": calibration.minimum_unsafe_recall,
        "training_record_ids_sha256": calibration.training_record_ids_sha256,
        "validation_record_ids_sha256": calibration.validation_record_ids_sha256,
        "training_seeds": list(calibration.training_seeds),
        "validation_seeds": list(calibration.validation_seeds),
        "selection_scope": calibration.selection_scope,
        "deployable": False,
        "truth_use": "offline_targets_and_scores_only",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


__all__ = [
    "OBJECTIVE_NAMES",
    "SPLITS",
    "CalibrationRecord",
    "FrozenCalibration",
    "MultiObjectiveWeights",
    "TrainingNormalizers",
    "calibration_manifest",
    "evaluate_multiobjective_loss",
    "fit_training_normalizers",
    "fit_validation_calibration",
    "score_calibration_records",
]
