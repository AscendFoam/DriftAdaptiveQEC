from dataclasses import replace

import numpy as np
import pytest

from cnn_fpga.decoder.hybrid_multiobjective import (
    OBJECTIVE_NAMES,
    CalibrationRecord,
    MultiObjectiveWeights,
    calibration_manifest,
    evaluate_multiobjective_loss,
    fit_training_normalizers,
    fit_validation_calibration,
    score_calibration_records,
)
from cnn_fpga.decoder.regime_hmm import REGIME_CLASSES


def _records(split: str, seed: int, count: int = 16) -> tuple[CalibrationRecord, ...]:
    values = []
    for index in range(count):
        label_index = index % len(REGIME_CLASSES)
        probabilities = np.full(len(REGIME_CLASSES), 0.05)
        probabilities[label_index] = 0.85
        target = np.linspace(0.1, 0.9, 9) + 0.01 * index
        prediction = target + (0.02 if index % 2 else -0.015)
        required = index % 3 == 0
        values.append(
            CalibrationRecord(
                record_id=f"{split}-{seed}-{index}",
                split=split,
                seed=seed,
                prediction=tuple(prediction),
                target=tuple(target),
                uncertainty_standard_errors=(0.02,) * 9,
                regime_probabilities=tuple(probabilities),
                regime_label=REGIME_CLASSES[label_index],
                candidate_failures=2 + index % 2,
                oracle_failures=1,
                oracle_trials=16,
                fallback_score=0.85 if required else 0.20,
                fallback_required=required,
                update_cost=0.05 + 0.01 * (index % 4),
            )
        )
    return tuple(values)


def _fit():
    training = _records("training", 101)
    validation = _records("validation", 202)
    normalizers = fit_training_normalizers(training)
    calibration = fit_validation_calibration(validation, normalizers)
    return training, validation, normalizers, calibration


def test_weights_cover_exactly_six_positive_normalized_objectives() -> None:
    weights = MultiObjectiveWeights()
    assert tuple(weights.as_dict()) == OBJECTIVE_NAMES
    assert sum(weights.as_dict().values()) == pytest.approx(1.0)
    for objective in OBJECTIVE_NAMES:
        ablated = weights.without(objective)
        assert ablated[objective] == 0.0
        assert sum(ablated.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"state_estimation": 0.0, "oracle_gap": 0.46}, "positive"),
        ({"state_estimation": 0.25}, "sum to one"),
    ],
)
def test_weights_fail_closed(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MultiObjectiveWeights(**kwargs)


def test_calibration_record_rejects_nonpositive_probability_and_online_scope() -> None:
    record = _records("training", 1, 1)[0]
    with pytest.raises(ValueError, match="positive and normalized"):
        replace(record, regime_probabilities=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="offline"):
        replace(record, scope="deployable")


def test_training_normalizers_are_robust_positive_and_training_only() -> None:
    training, _, normalizers, _ = _fit()
    assert normalizers.source_split == "training_only"
    assert normalizers.training_seeds == (101,)
    assert min(normalizers.state_scales) > 0.0
    assert set(normalizers.objective_scales) == set(OBJECTIVE_NAMES)
    assert len(normalizers.training_record_ids_sha256) == 64
    with pytest.raises(ValueError, match="training"):
        fit_training_normalizers(tuple(replace(row, split="validation") for row in training))


def test_validation_calibration_is_seed_disjoint_and_safety_constrained() -> None:
    _, validation, normalizers, calibration = _fit()
    assert calibration.validation_seeds == (202,)
    assert calibration.training_seeds == (101,)
    report = score_calibration_records(validation, normalizers, calibration)
    assert report["diagnostics"]["required_fallback_recall"] >= 0.90
    assert 0.0 <= calibration.regime_uniform_mix <= 1.0
    overlapping = tuple(replace(row, seed=101) for row in validation)
    with pytest.raises(ValueError, match="overlap"):
        fit_validation_calibration(overlapping, normalizers)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"regime_temperature_grid": ()}, "temperature grid"),
        ({"regime_uniform_mix_grid": (-0.1, 0.0)}, "uniform mix"),
        ({"uncertainty_scale_grid": (0.0,)}, "uncertainty scale"),
        ({"fallback_threshold_grid": (1.1,)}, "fallback threshold"),
    ],
)
def test_validation_grids_fail_closed(kwargs: dict[str, object], match: str) -> None:
    _, validation, normalizers, _ = _fit()
    with pytest.raises(ValueError, match=match):
        fit_validation_calibration(validation, normalizers, **kwargs)


def test_frozen_evaluation_reports_every_objective_and_ablation() -> None:
    _, _, normalizers, calibration = _fit()
    evaluation = _records("evaluation", 303)
    report = evaluate_multiobjective_loss(evaluation, normalizers, calibration)
    assert set(report["raw_objectives"]) == set(OBJECTIVE_NAMES)
    assert set(report["weighted_objectives"]) == set(OBJECTIVE_NAMES)
    assert set(report["leave_one_objective_out"]) == set(OBJECTIVE_NAMES)
    assert report["selection_provenance"]["evaluation_used_for_selection"] is False
    assert np.isfinite(report["total_loss"])


def test_evaluation_seed_overlap_is_rejected() -> None:
    _, _, normalizers, calibration = _fit()
    with pytest.raises(ValueError, match="overlap"):
        evaluate_multiobjective_loss(_records("evaluation", 202), normalizers, calibration)


def test_score_requires_one_homogeneous_split() -> None:
    training, validation, normalizers, calibration = _fit()
    with pytest.raises(ValueError, match="exclusively"):
        score_calibration_records((training[0], validation[0]), normalizers, calibration)


def test_calibration_manifest_is_deterministic_truth_bounded_and_hash_bound() -> None:
    _, _, normalizers, calibration = _fit()
    first = calibration_manifest(normalizers, calibration)
    second = calibration_manifest(normalizers, calibration)
    assert first == second
    assert first["deployable"] is False
    assert first["truth_use"] == "offline_targets_and_scores_only"
    assert len(first["manifest_sha256"]) == 64

