"""Observed-score post-selection diagnostics with explicit rejection cost.

This module does not implement an online correction action.  It evaluates an
offline accept/reject diagnostic from decoder posterior confidence and keeps a
truth-only upper bound confined to the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _probability(value: object, name: str, *, allow_endpoints: bool = True) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real probability")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real probability") from exc
    lower_ok = result >= 0.0 if allow_endpoints else result > 0.0
    upper_ok = result <= 1.0 if allow_endpoints else result < 1.0
    if not isfinite(result) or not lower_ok or not upper_ok:
        interval = "[0,1]" if allow_endpoints else "(0,1)"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def posterior_error_risk(posterior: ArrayLike) -> NDArray[np.float64]:
    """Return ``1-max_c p(c|observed syndrome)`` without hidden labels."""

    probabilities = np.asarray(posterior, dtype=np.float64)
    if probabilities.ndim < 1 or probabilities.shape[-1] < 2:
        raise ValueError("posterior must have shape (...,classes) with at least two classes")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("posterior must contain only finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("posterior probabilities must be nonnegative")
    normalization = np.sum(probabilities, axis=-1)
    if not np.allclose(normalization, 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("posterior probabilities must sum to one")
    return np.asarray(1.0 - np.max(probabilities, axis=-1), dtype=np.float64)


def calibrate_survival_thresholds(
    training_scores: ArrayLike,
    target_survivals: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    scores = np.asarray(training_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size < 1024:
        raise ValueError("training_scores must be a 1D non-demo array with at least 1024 values")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("training_scores must be finite and lie in [0,1]")
    targets = tuple(_probability(value, "target survival", allow_endpoints=False) for value in target_survivals)
    if len(targets) < 5 or len(set(targets)) != len(targets):
        raise ValueError("target_survivals must contain at least five unique values")
    if tuple(sorted(targets, reverse=True)) != targets:
        raise ValueError("target_survivals must be strictly decreasing")
    thresholds: list[tuple[float, float]] = []
    for target in targets:
        threshold = float(np.quantile(scores, target, method="higher"))
        thresholds.append((target, threshold))
    return tuple(thresholds)


@dataclass(frozen=True)
class PostselectionMetrics:
    threshold: float
    total_samples: int
    accepted_samples: int
    rejected_samples: int
    survival_fraction: float
    rejection_fraction: float
    raw_error_rate: float
    conditional_error_rate: float
    accepted_failures_per_input: float
    rejected_failure_capture_fraction: float
    rejected_error_rate: float
    truth_upper_conditional_error_rate: float
    truth_upper_accepted_failures_per_input: float
    random_rejection_expected_conditional_error_rate: float
    random_rejection_expected_accepted_failures_per_input: float
    break_even_rejection_penalty: float
    total_cost_by_rejection_penalty: Mapping[str, float]


def evaluate_postselection(
    observed_scores: ArrayLike,
    failures: ArrayLike,
    *,
    threshold: float,
    rejection_penalties: Sequence[float],
) -> PostselectionMetrics:
    scores = np.asarray(observed_scores, dtype=np.float64)
    error = np.asarray(failures)
    if scores.ndim != 1 or error.ndim != 1 or scores.shape != error.shape:
        raise ValueError("observed_scores and failures must be same-shape 1D arrays")
    if scores.size < 256:
        raise ValueError("post-selection evaluation requires at least 256 samples")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("observed_scores must be finite and lie in [0,1]")
    if error.dtype != np.bool_:
        raise TypeError("failures must be a boolean array")
    cutoff = float(threshold)
    if not isfinite(cutoff):
        raise ValueError("threshold must be finite")
    penalties = tuple(_probability(value, "rejection penalty") for value in rejection_penalties)
    if len(penalties) < 4 or len(set(penalties)) != len(penalties):
        raise ValueError("rejection_penalties must contain at least four unique values")
    if tuple(sorted(penalties)) != penalties or penalties[0] != 0.0 or penalties[-1] != 1.0:
        raise ValueError("rejection_penalties must increase from 0 to 1")

    accepted = scores <= cutoff
    accepted_count = int(np.count_nonzero(accepted))
    total = int(scores.size)
    if accepted_count == 0:
        raise ValueError("threshold rejects every sample")
    rejected_count = total - accepted_count
    total_failures = int(np.count_nonzero(error))
    accepted_failures = int(np.count_nonzero(error & accepted))
    rejected_failures = total_failures - accepted_failures
    survival = accepted_count / total
    rejection = rejected_count / total
    raw_error = total_failures / total
    conditional = accepted_failures / accepted_count
    accepted_per_input = accepted_failures / total
    captured = 0.0 if total_failures == 0 else rejected_failures / total_failures
    rejected_error = 0.0 if rejected_count == 0 else rejected_failures / rejected_count

    # Nondeployable diagnostic upper bound at the *same realized coverage*:
    # reject failures first, then correct samples.  No truth label enters the
    # observed score or threshold above.
    truth_upper_accepted_failures = max(0, total_failures - rejected_count)
    truth_upper_conditional = truth_upper_accepted_failures / accepted_count
    truth_upper_per_input = truth_upper_accepted_failures / total
    costs = {
        f"{penalty:.2f}": accepted_per_input + penalty * rejection
        for penalty in penalties
    }
    return PostselectionMetrics(
        threshold=cutoff,
        total_samples=total,
        accepted_samples=accepted_count,
        rejected_samples=rejected_count,
        survival_fraction=survival,
        rejection_fraction=rejection,
        raw_error_rate=raw_error,
        conditional_error_rate=conditional,
        accepted_failures_per_input=accepted_per_input,
        rejected_failure_capture_fraction=captured,
        rejected_error_rate=rejected_error,
        truth_upper_conditional_error_rate=truth_upper_conditional,
        truth_upper_accepted_failures_per_input=truth_upper_per_input,
        random_rejection_expected_conditional_error_rate=raw_error,
        random_rejection_expected_accepted_failures_per_input=survival * raw_error,
        break_even_rejection_penalty=rejected_error,
        total_cost_by_rejection_penalty=costs,
    )


def binary_score_auc(observed_scores: ArrayLike, failures: ArrayLike) -> float:
    """Tie-aware AUROC for the hypothesis that larger score predicts failure."""

    scores = np.asarray(observed_scores, dtype=np.float64)
    error = np.asarray(failures)
    if scores.ndim != 1 or error.shape != scores.shape or error.dtype != np.bool_:
        raise ValueError("scores/failures must be same-shape 1D arrays with boolean failures")
    positive = int(np.count_nonzero(error))
    negative = int(scores.size - positive)
    if positive == 0 or negative == 0:
        raise ValueError("AUROC requires both failure and success samples")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    rank_sum = float(np.sum(ranks[error]))
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


__all__ = [
    "PostselectionMetrics",
    "posterior_error_risk",
    "calibrate_survival_thresholds",
    "evaluate_postselection",
    "binary_score_auc",
]
