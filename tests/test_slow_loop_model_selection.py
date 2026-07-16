from __future__ import annotations

import numpy as np
import pytest

from cnn_fpga.decoder.slow_loop_model_selection import (
    MODEL_FAMILIES,
    CausalTCN,
    DiagonalGaussianHead,
    FeatureStandardizer,
    RollingGaussianHMMAdapter,
    SlowLoopSelectionBudget,
    SmallGRU,
    bounded_histories,
    diagonal_kalman_states,
    exponential_states,
    labels_for_histories,
    resource_profiles,
    run_length_fsm_posterior,
    softmax_logits,
    temper_posterior,
)
from cnn_fpga.decoder.regime_hmm import REGIME_CLASSES, fit_supervised_gaussian_hmm


def _sequences() -> list[np.ndarray]:
    rng = np.random.default_rng(411)
    return [rng.normal(size=(12, 14)), rng.normal(size=(11, 14))]


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: SlowLoopSelectionBudget(window_cycles=31), "one slow-loop update"),
        (lambda: SlowLoopSelectionBudget(history_windows=1), "at least 2"),
        (lambda: SlowLoopSelectionBudget(summary_feature_count=13), "canonical"),
        (lambda: SlowLoopSelectionBudget(class_count=3), "REGIME_CLASSES"),
        (lambda: SlowLoopSelectionBudget(max_macs_per_update=True), "integer"),
        (lambda: SlowLoopSelectionBudget(host_software_latency_ceiling_us=0.0), "positive"),
    ],
)
def test_budget_contract_fails_closed(factory: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        factory()  # type: ignore[operator]


def test_bounded_histories_never_cross_trajectory_and_align_labels() -> None:
    sequences = _sequences()
    histories, sequence_index, local_index = bounded_histories(sequences, history_windows=8)
    labels = [
        ["normal", "burst", "leakage", "calibration_shift"] * 3,
        ["leakage"] * 11,
    ]
    truth = labels_for_histories(labels, sequence_index, local_index)

    assert histories.shape == (9, 8, 14)
    assert np.array_equal(histories[0], sequences[0][:8])
    assert np.array_equal(histories[4], sequences[0][4:12])
    assert np.array_equal(histories[5], sequences[1][:8])
    assert truth[0] == 3
    assert np.all(truth[5:] == 2)


def test_training_standardizer_is_copy_isolated_and_finite() -> None:
    sequences = _sequences()
    original = sequences[0].copy()
    standardizer = FeatureStandardizer.fit(sequences)
    transformed = standardizer.transform(original)
    sequences[0][:] = 999.0

    assert standardizer.mean.flags.writeable is False
    assert standardizer.scale.flags.writeable is False
    assert transformed.shape == original.shape
    assert np.all(np.isfinite(transformed))
    assert not np.allclose(standardizer.mean, 999.0)


def test_all_six_families_fit_the_shared_resource_envelope() -> None:
    budget = SlowLoopSelectionBudget()
    profiles = resource_profiles(budget)

    assert tuple(profiles) == MODEL_FAMILIES
    assert all(profile.within(budget) for profile in profiles.values())
    assert profiles["gaussian_hmm"].model_and_state_bytes == 3728
    assert profiles["gaussian_hmm"].macs_per_update_proxy == 926
    assert profiles["causal_tcn"].macs_per_update_proxy == 3556
    assert profiles["small_gru"].macs_per_update_proxy == 2300
    assert max(profile.model_and_state_bytes for profile in profiles.values()) <= 4096
    assert max(profile.transient_workspace_bytes for profile in profiles.values()) <= 4096


def test_rolling_hmm_emission_cache_is_exactly_equivalent_to_last8_replay() -> None:
    rng = np.random.default_rng(413)
    centers = rng.normal(size=(4, 14))
    feature_sequences = []
    label_sequences = []
    for offset in range(3):
        labels = [REGIME_CLASSES[(index // 20 + offset) % 4] for index in range(160)]
        features = np.vstack([centers[REGIME_CLASSES.index(label)] for label in labels])
        features += rng.normal(0.0, 0.15, size=features.shape)
        feature_sequences.append(features)
        label_sequences.append(labels)
    model = fit_supervised_gaussian_hmm(feature_sequences, label_sequences)
    adapter = RollingGaussianHMMAdapter(model, history_windows=8, temperature=1.25)
    outputs = np.vstack([adapter.step(row) for row in feature_sequences[0]])

    for stop in (8, 19, 61, 160):
        replay = model.filter_sequence(feature_sequences[0][stop - 8 : stop], temperature=1.25)[-1]
        np.testing.assert_allclose(outputs[stop - 1], replay, rtol=1.0e-13, atol=1.0e-13)
    assert adapter.ready
    assert adapter.cached_emission_count == 8
    adapter.reset()
    assert not adapter.ready and adapter.cached_emission_count == 0


def test_exponential_and_kalman_filters_are_bounded_prefix_causal() -> None:
    histories, _, _ = bounded_histories(_sequences(), history_windows=8)
    recurrence = exponential_states(histories, 0.7)
    kalman = diagonal_kalman_states(histories, 0.1, 1.0)

    assert recurrence.shape == kalman.shape == (9, 14)
    changed = histories.copy()
    changed[:, -1] += 1.0
    assert not np.allclose(exponential_states(changed, 0.7), recurrence)
    assert not np.allclose(diagonal_kalman_states(changed, 0.1, 1.0), kalman)
    with pytest.raises(ValueError, match="less than one"):
        exponential_states(histories, 1.0)


def test_diagonal_head_and_temperature_return_proper_probabilities() -> None:
    rng = np.random.default_rng(412)
    states = np.vstack([rng.normal(index, 0.2, size=(20, 3)) for index in range(4)])
    labels = np.repeat(np.arange(4), 20)
    head = DiagonalGaussianHead.fit(states, labels, variance_floor=0.01)
    posterior = softmax_logits(head.logits(states), temperature=1.25)
    reheated = temper_posterior(posterior, temperature=2.0)

    assert head.parameter_count == 28
    assert np.allclose(posterior.sum(axis=1), 1.0)
    assert np.allclose(reheated.sum(axis=1), 1.0)
    assert np.mean(np.argmax(posterior, axis=1) == labels) > 0.95
    assert np.mean(np.max(reheated, axis=1)) < np.mean(np.max(posterior, axis=1))


def test_run_length_fsm_requires_persistence_and_has_no_zero_probability() -> None:
    instantaneous = np.full((1, 8, 4), 0.01)
    instantaneous[:, :, 0] = 0.97
    instantaneous[0, 5:, 0] = 0.01
    instantaneous[0, 5:, 2] = 0.97
    result = run_length_fsm_posterior(instantaneous, enter_run=2, confidence=0.8)

    assert np.argmax(result[0]) == 2
    assert np.all(result > 0.0)
    assert np.sum(result) == pytest.approx(1.0)


def test_neural_architectures_have_exact_analytic_counts_and_causal_last_output() -> None:
    torch = pytest.importorskip("torch")
    assert CausalTCN is not None and SmallGRU is not None
    tcn = CausalTCN()
    gru = SmallGRU()
    values = torch.randn(3, 8, 14)

    assert tcn.parameter_count == 487
    assert gru.parameter_count == 339
    assert tcn(values).shape == gru(values).shape == (3, 4)
    prefix_changed = values.clone()
    prefix_changed[:, -1, :] += 2.0
    assert not torch.equal(tcn(values), tcn(prefix_changed))
    assert not torch.equal(gru(values), gru(prefix_changed))
