from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cnn_fpga.benchmark.latest_outcome_markovian_baseline import train_agent
from physics.latest_outcome_markovian import (
    ARCHITECTURE_ID,
    COMPUTE_CONTRACT,
    BudgetMatchedMarkovianPolicy,
    ObservedOutcome,
    audit_latest_only_behavior,
    build_budget_matched_policy,
)
from physics.nmf_directional_ranking import DirectionalRankingConfig, build_policy

torch = pytest.importorskip("torch")


def tiny_config(**overrides: object) -> DirectionalRankingConfig:
    values: dict[str, object] = {
        "cutoff": 6,
        "confirmation_cutoff": 6,
        "full_cycles": 2,
        "train_epochs": 1,
        "train_batch_size": 2,
        "validation_batch_size": 2,
        "test_batch_size": 2,
        "confirmation_batch_size": 2,
        "validation_interval": 1,
        "training_seeds": (11,),
        "validation_seeds": (101,),
        "test_seeds": (211,),
        "confirmation_seeds": (307,),
        "bootstrap_repetitions": 100,
        "device": "cpu",
    }
    values.update(overrides)
    return DirectionalRankingConfig(**values)


def build(seed: int = 17) -> BudgetMatchedMarkovianPolicy:
    return build_budget_matched_policy(device="cpu", dtype=torch.float64, seed=seed)


def test_parameter_and_dense_mac_budgets_exactly_match_history_gru() -> None:
    config = tiny_config()
    markovian = build()
    history = build_policy("nmf", config, 17)
    assert markovian.parameter_count == history.parameter_count == 72_853
    assert markovian.front_parameter_count == COMPUTE_CONTRACT.front_parameter_count == 390
    assert markovian.dense_mac_count == COMPUTE_CONTRACT.total_dense_macs == 72_266
    assert COMPUTE_CONTRACT.front_dense_macs == 330


def test_no_capacity_padding_parameter_is_outside_forward_modules() -> None:
    model = build()
    expected = {
        "feature_pair_scale",
        "outcome_encoder.weight", "outcome_encoder.bias",
        "adapter_down.weight", "adapter_down.bias",
        "adapter_up.weight", "adapter_up.bias",
        "adapter_norm.weight", "adapter_norm.bias",
        "dense1.weight", "dense1.bias", "dense2.weight", "dense2.bias",
        "output.weight", "output.bias",
    }
    assert set(dict(model.named_parameters())) == expected
    assert ARCHITECTURE_ID.startswith("LATEST3-STATIC390")


def test_initialization_is_seed_reproducible_and_seed_sensitive() -> None:
    first = build(17)
    second = build(17)
    third = build(19)
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    assert any(
        not torch.equal(value, third.state_dict()[name])
        for name, value in first.state_dict().items()
    )


def test_earlier_history_is_bit_exactly_irrelevant() -> None:
    model = build()
    histories = torch.tensor(
        [[0, 0, 0, 1], [1, 1, 0, 1], [2, 2, 2, 1]], dtype=torch.int64
    )
    # Compare equal-shape calls so BLAS batch-width roundoff cannot be confused
    # with a history dependency.
    outputs = [model(histories[index : index + 1], 4) for index in range(3)]
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])


def test_all_three_observed_tokens_are_executable_and_distinct() -> None:
    model = build()
    tokens = torch.tensor(
        [ObservedOutcome.G, ObservedOutcome.E, ObservedOutcome.LEAKAGE],
        dtype=torch.int64,
    )
    outputs = model.forward_latest(tokens)
    assert outputs.shape == (3, 15)
    assert torch.all(torch.isfinite(outputs))
    assert all(
        torch.max(torch.abs(outputs[left] - outputs[right])).detach().item() > 0.0
        for left, right in ((0, 1), (0, 2), (1, 2))
    )


def test_start_token_is_explicit_zero_vector_and_batch_preserving() -> None:
    model = build()
    history = torch.empty((4, 0), dtype=torch.int64)
    output = model(history, 0)
    direct = model.forward_latest(None, batch_size=4)
    assert output.shape == (4, 15)
    assert torch.equal(output, direct)
    assert torch.equal(output[0], output[-1])


@pytest.mark.parametrize(
    "tokens,error",
    [
        (torch.tensor([-1], dtype=torch.int64), ValueError),
        (torch.tensor([3], dtype=torch.int64), ValueError),
        (torch.tensor([0.0]), TypeError),
        (torch.tensor([[0]], dtype=torch.int64), ValueError),
        (torch.tensor([], dtype=torch.int64), ValueError),
    ],
)
def test_invalid_latest_tokens_fail_closed(tokens: torch.Tensor, error: type[Exception]) -> None:
    with pytest.raises(error):
        build().forward_latest(tokens)


def test_invalid_history_contract_fails_closed() -> None:
    model = build()
    with pytest.raises(TypeError):
        model(torch.tensor([0], dtype=torch.int64), 1)
    with pytest.raises(ValueError, match="width"):
        model(torch.tensor([[0, 1]], dtype=torch.int64), 1)
    with pytest.raises(ValueError, match="nonnegative"):
        model(torch.tensor([[]], dtype=torch.int64), -1)


def test_policy_has_no_rollout_state_and_call_order_does_not_matter() -> None:
    model = build()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    first = model.forward_latest(torch.tensor([0, 1, 2], dtype=torch.int64))
    _ = model.forward_latest(torch.tensor([2, 0], dtype=torch.int64))
    second = model.forward_latest(torch.tensor([0, 1, 2], dtype=torch.int64))
    assert torch.equal(first, second)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
    assert audit_latest_only_behavior(model)["has_no_recurrent_state_attribute"] is True


def test_every_parameter_element_gets_gradient_when_all_tokens_are_exercised() -> None:
    model = build()
    output = model.forward_latest(torch.tensor([0, 1, 2], dtype=torch.int64))
    weights = torch.arange(1, 16, dtype=torch.float64)[None, :]
    loss = torch.sum(output * weights) + torch.sum(output**2)
    loss.backward()
    missing = {
        name: int(torch.count_nonzero(parameter.grad == 0).item())
        for name, parameter in model.named_parameters()
        if parameter.grad is None or torch.any(parameter.grad == 0)
    }
    assert missing == {}


def test_tiny_training_uses_validation_selection_and_expected_leakage_gap() -> None:
    config = tiny_config()
    model, record = train_agent(11, config)
    assert model.parameter_count == 72_853
    assert record["epochs_executed"] == 1
    assert record["validation_seeds_used_for_checkpoint_selection_only"] == [101]
    assert set(record["training_trajectory_seeds"]).isdisjoint({101, 211, 307})
    coverage = record["gradient_coverage"]
    assert coverage["total_parameter_elements"] == 72_853
    # The two-level simulator never emits leakage, so precisely the ten
    # leakage-column encoder weights may remain unseen in production training.
    assert coverage["covered_parameter_elements"] >= 72_843
    assert np.isfinite(record["best_validation_score"])
