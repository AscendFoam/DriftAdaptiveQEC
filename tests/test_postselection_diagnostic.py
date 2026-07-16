from __future__ import annotations

import inspect

import numpy as np
import pytest

from cnn_fpga.decoder.postselection_diagnostic import (
    binary_score_auc,
    calibrate_survival_thresholds,
    evaluate_postselection,
    posterior_error_risk,
)


def test_posterior_error_risk_uses_only_max_probability() -> None:
    posterior = np.asarray([[0.7, 0.2, 0.08, 0.02], [0.25, 0.25, 0.25, 0.25]])
    assert np.allclose(posterior_error_risk(posterior), [0.3, 0.75])
    assert set(inspect.signature(posterior_error_risk).parameters) == {"posterior"}


@pytest.mark.parametrize(
    "posterior,match",
    [
        (np.asarray(1.0), "shape"),
        (np.asarray([[0.6, 0.5]]), "sum"),
        (np.asarray([[1.1, -0.1]]), "nonnegative"),
        (np.asarray([[np.nan, np.nan]]), "finite"),
    ],
)
def test_posterior_error_risk_fails_closed(posterior: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        posterior_error_risk(posterior)


def test_training_thresholds_are_monotone_and_deterministic() -> None:
    scores = np.linspace(0.0, 0.8, 4096) ** 2
    targets = (0.98, 0.95, 0.9, 0.8, 0.7, 0.5)
    first = calibrate_survival_thresholds(scores, targets)
    second = calibrate_survival_thresholds(scores, targets)
    assert first == second
    assert [row[0] for row in first] == list(targets)
    assert all(first[index][1] > first[index + 1][1] for index in range(len(first) - 1))


def test_threshold_calibration_rejects_demo_and_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="non-demo"):
        calibrate_survival_thresholds(np.linspace(0, 1, 100), (0.9, 0.8, 0.7, 0.6, 0.5))
    with pytest.raises(ValueError, match="strictly decreasing"):
        calibrate_survival_thresholds(np.linspace(0, 1, 2048), (0.9, 0.7, 0.8, 0.6, 0.5))
    with pytest.raises(ValueError, match="unique"):
        calibrate_survival_thresholds(np.linspace(0, 1, 2048), (0.9, 0.8, 0.8, 0.6, 0.5))
    with pytest.raises(ValueError, match=r"\(0,1\)"):
        calibrate_survival_thresholds(np.linspace(0, 1, 2048), (1.0, 0.9, 0.8, 0.7, 0.5))


def _perfect_risk_case() -> tuple[np.ndarray, np.ndarray]:
    scores = np.linspace(0.0, 1.0, 1000, endpoint=False)
    failures = scores >= 0.9
    return scores, failures.astype(np.bool_)


def test_metrics_count_survival_failure_capture_and_truth_upper_bound() -> None:
    scores, failures = _perfect_risk_case()
    result = evaluate_postselection(
        scores,
        failures,
        threshold=0.899,
        rejection_penalties=(0.0, 0.25, 0.5, 1.0),
    )
    assert result.accepted_samples == 900
    assert result.rejected_samples == 100
    assert result.survival_fraction == pytest.approx(0.9)
    assert result.raw_error_rate == pytest.approx(0.1)
    assert result.conditional_error_rate == 0.0
    assert result.rejected_failure_capture_fraction == 1.0
    assert result.rejected_error_rate == 1.0
    assert result.truth_upper_conditional_error_rate == 0.0
    assert result.break_even_rejection_penalty == 1.0


def test_total_cost_identity_prevents_free_rejection() -> None:
    rng = np.random.default_rng(12)
    scores = rng.random(4096)
    failures = (rng.random(4096) < (0.02 + 0.4 * scores)).astype(np.bool_)
    result = evaluate_postselection(
        scores,
        failures,
        threshold=0.8,
        rejection_penalties=(0.0, 0.25, 0.5, 1.0),
    )
    for penalty in (0.0, 0.25, 0.5, 1.0):
        expected = result.accepted_failures_per_input + penalty * result.rejection_fraction
        assert result.total_cost_by_rejection_penalty[f"{penalty:.2f}"] == pytest.approx(expected)
    assert result.total_cost_by_rejection_penalty["1.00"] >= result.raw_error_rate
    assert result.random_rejection_expected_conditional_error_rate == result.raw_error_rate


def test_truth_upper_is_coverage_matched_not_free_zero_error() -> None:
    rng = np.random.default_rng(15)
    scores = rng.random(1000)
    failures = np.zeros(1000, dtype=np.bool_)
    failures[:400] = True
    result = evaluate_postselection(
        scores,
        failures,
        threshold=float(np.quantile(scores, 0.9)),
        rejection_penalties=(0.0, 0.25, 0.5, 1.0),
    )
    assert result.rejected_samples == 100
    assert result.truth_upper_accepted_failures_per_input == pytest.approx(0.3)
    assert result.truth_upper_conditional_error_rate == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "scores,failures,threshold,penalties,match",
    [
        (np.zeros(255), np.zeros(255, dtype=np.bool_), 0.5, (0, 0.25, 0.5, 1), "at least 256"),
        (np.zeros(300), np.zeros(299, dtype=np.bool_), 0.5, (0, 0.25, 0.5, 1), "same-shape"),
        (np.zeros(300), np.zeros(300, dtype=np.int8), 0.5, (0, 0.25, 0.5, 1), "boolean"),
        (np.zeros(300), np.zeros(300, dtype=np.bool_), -1.0, (0, 0.25, 0.5, 1), "rejects every"),
        (np.zeros(300), np.zeros(300, dtype=np.bool_), 0.5, (0, 0.5, 1), "at least four"),
    ],
)
def test_postselection_evaluator_fails_closed(
    scores: np.ndarray,
    failures: np.ndarray,
    threshold: float,
    penalties: tuple[float, ...],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        evaluate_postselection(scores, failures, threshold=threshold, rejection_penalties=penalties)


def test_auc_is_tie_aware_and_requires_both_classes() -> None:
    scores, failures = _perfect_risk_case()
    assert binary_score_auc(scores, failures) == pytest.approx(1.0)
    tied_scores = np.zeros(1000)
    tied_failures = np.zeros(1000, dtype=np.bool_)
    tied_failures[:100] = True
    assert binary_score_auc(tied_scores, tied_failures) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="both"):
        binary_score_auc(scores, np.zeros(1000, dtype=np.bool_))
