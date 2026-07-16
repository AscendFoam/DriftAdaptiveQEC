from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from cnn_fpga.decoder.regime_hmm import (
    RAW_FEATURE_NAMES,
    REGIME_CLASSES,
    SUMMARY_FEATURE_NAMES,
    RegimeEstimatorBudget,
    RegimeObservationWindow,
    fit_supervised_gaussian_hmm,
    posterior_dict,
    summarize_regime_window,
)


def _raw(cycles: int = 32) -> np.ndarray:
    values = np.zeros((cycles, len(RAW_FEATURE_NAMES)), dtype=float)
    values[:, 0] = np.linspace(-0.2, 0.3, cycles)
    values[:, 1] = np.linspace(0.1, -0.1, cycles)
    values[::4, 2] = 1.0
    values[1::5, 3] = 1.0
    values[2::11, 4] = 1.0
    values[:, 5] = np.arange(cycles) & 1
    values[:, 6:] = 1.0
    return values


def _training_data() -> tuple[list[np.ndarray], list[list[str]]]:
    rng = np.random.default_rng(326)
    centers = np.zeros((4, len(SUMMARY_FEATURE_NAMES)))
    centers[0, :4] = (0.0, 0.0, -1.0, -1.0)
    centers[1, :4] = (0.0, 0.0, 1.2, 1.0)
    centers[2, 10] = 2.5
    centers[3, :2] = (2.0, -2.0)
    features: list[np.ndarray] = []
    labels: list[list[str]] = []
    orders = ((0, 1, 2, 3), (2, 0, 3, 1), (3, 1, 0, 2))
    for order in orders:
        sequence_labels = [REGIME_CLASSES[index] for index in order for _ in range(32)]
        sequence_features = np.vstack(
            [centers[REGIME_CLASSES.index(label)] for label in sequence_labels]
        )
        sequence_features += rng.normal(0.0, 0.12, size=sequence_features.shape)
        features.append(sequence_features)
        labels.append(sequence_labels)
    return features, labels


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RegimeObservationWindow(0, 0, np.zeros((1, 8))),
        lambda: RegimeObservationWindow(0, 0, np.zeros((2, 7))),
        lambda: RegimeObservationWindow(0, 0, np.full((2, 8), np.nan)),
        lambda: RegimeObservationWindow(0, 0, np.full((2, 8), 0.5)),
        lambda: RegimeObservationWindow(True, 0, _raw()),
        lambda: RegimeEstimatorBudget(window_cycles=31, update_period_cycles=32),
        lambda: RegimeEstimatorBudget(raw_feature_count=7),
        lambda: RegimeEstimatorBudget(summary_feature_count=13),
    ],
)
def test_window_and_budget_contracts_fail_closed(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_window_is_copy_isolated_and_summary_matches_direct_statistics() -> None:
    source = _raw()
    window = RegimeObservationWindow(3, 96, source)
    source[:] = 9.0
    summary = summarize_regime_window(window)

    assert window.cycles == 32
    assert window.end_cycle == 127
    assert window.values.flags.writeable is False
    assert summary.shape == (len(SUMMARY_FEATURE_NAMES),)
    assert summary[0] == pytest.approx(np.mean(np.linspace(-0.2, 0.3, 32)))
    assert summary[8] == pytest.approx(0.25)
    assert summary[10] == pytest.approx(3 / 32)
    assert summary[11] == 1.0
    assert summary[12] == 1.0


def test_supervised_fit_builds_full_covariance_stochastic_hmm() -> None:
    features, labels = _training_data()
    model = fit_supervised_gaussian_hmm(features, labels)

    assert model.emission_covariances.shape == (4, 14, 14)
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
    assert np.isclose(model.initial_probabilities.sum(), 1.0)
    assert np.isclose(model.class_prior_probabilities.sum(), 1.0)
    assert np.min(np.linalg.eigvalsh(model.emission_covariances)) > 0.0
    assert model.parameter_count == 896
    assert model.macs_per_update_proxy == 800
    assert model.parameter_count * 4 <= RegimeEstimatorBudget().max_float32_state_bytes
    assert model.macs_per_update_proxy <= RegimeEstimatorBudget().max_macs_per_update


def test_filter_and_memoryless_posteriors_are_normalized_and_informative() -> None:
    features, labels = _training_data()
    model = fit_supervised_gaussian_hmm(features, labels)
    hmm = model.filter_sequence(features[0])
    memoryless = model.memoryless_posterior(features[0])
    truth = np.asarray([REGIME_CLASSES.index(label) for label in labels[0]])

    assert hmm.shape == memoryless.shape == (128, 4)
    assert np.allclose(hmm.sum(axis=1), 1.0)
    assert np.allclose(memoryless.sum(axis=1), 1.0)
    assert np.mean(np.argmax(hmm, axis=1) == truth) > 0.95
    assert np.mean(np.argmax(memoryless, axis=1) == truth) > 0.95


def test_forward_filter_is_strictly_prefix_causal() -> None:
    features, labels = _training_data()
    model = fit_supervised_gaussian_hmm(features, labels)
    full = model.filter_sequence(features[1], temperature=1.25)
    for stop in (1, 7, 33, 79, 128):
        prefix = model.filter_sequence(features[1][:stop], temperature=1.25)
        np.testing.assert_allclose(prefix, full[:stop], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_class", "each regime"),
        ("unknown_label", "unknown regime"),
        ("misaligned", "aligned"),
        ("wrong_width", "canonical summary width"),
        ("zero_regularization", "positive"),
    ],
)
def test_fit_rejects_malformed_or_underidentified_training_data(
    mutation: str, match: str
) -> None:
    features, labels = _training_data()
    kwargs: dict[str, object] = {}
    if mutation == "missing_class":
        labels = [["normal" if label == "leakage" else label for label in row] for row in labels]
    elif mutation == "unknown_label":
        labels[0][0] = "secret"
    elif mutation == "misaligned":
        labels = labels[:-1]
    elif mutation == "wrong_width":
        features[0] = features[0][:, :-1]
    elif mutation == "zero_regularization":
        kwargs["covariance_regularization"] = 0.0
    with pytest.raises((TypeError, ValueError), match=match):
        fit_supervised_gaussian_hmm(features, labels, **kwargs)  # type: ignore[arg-type]


def test_prediction_apis_reject_nonfinite_wrong_shape_and_bad_temperature() -> None:
    features, labels = _training_data()
    model = fit_supervised_gaussian_hmm(features, labels)
    with pytest.raises(ValueError, match="wrong shape"):
        model.filter_sequence(np.zeros((2, 13)))
    with pytest.raises(ValueError, match="finite"):
        model.memoryless_posterior(np.full((2, 14), np.nan))
    with pytest.raises(ValueError, match="positive"):
        model.filter_sequence(features[0], temperature=0.0)


def test_posterior_dict_is_normalized_immutable_and_complete() -> None:
    result = posterior_dict((0.1, 0.2, 0.3, 0.4))
    assert tuple(result) == REGIME_CLASSES
    assert sum(result.values()) == pytest.approx(1.0)
    with pytest.raises(TypeError):
        result["normal"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="normalized"):
        posterior_dict((0.1, 0.2, 0.3, 0.3))


def test_online_window_schema_contains_no_truth_label_or_regime() -> None:
    names = {field.name for field in fields(RegimeObservationWindow)}
    assert names == {"window_index", "start_cycle", "values"}
    assert not any("truth" in name or "label" in name or "regime" in name for name in names)
