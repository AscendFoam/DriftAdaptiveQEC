from __future__ import annotations

import pytest

from physics.memory_specific_ablation import (
    FrozenMemoryInterventionPolicy,
    MemoryInterventionSpec,
)
from physics.nmf_directional_ranking import (
    DirectionalRankingConfig,
    build_policy,
    state_dict_sha256,
)


torch = pytest.importorskip("torch")


def _parent() -> object:
    config = DirectionalRankingConfig(
        cutoff=6,
        confirmation_cutoff=6,
        full_cycles=2,
        train_epochs=1,
        train_batch_size=1,
        validation_batch_size=1,
        test_batch_size=1,
        confirmation_batch_size=1,
        validation_interval=1,
        bootstrap_repetitions=10,
        training_seeds=(1,),
        validation_seeds=(2,),
        test_seeds=(3,),
        confirmation_seeds=(4,),
        device="cpu",
    )
    return build_policy("nmf", config, 7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "unknown"},
        {"mode": "history_truncation"},
        {"mode": "history_truncation", "history_length": 0},
        {"mode": "periodic_hidden_reset", "reset_period": True},
        {"mode": "history_shuffle", "shuffle_seed": -1},
        {"mode": "full_history", "history_length": 2},
    ],
)
def test_spec_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryInterventionSpec(**kwargs)  # type: ignore[arg-type]


def test_full_history_view_is_bit_exact_to_parent() -> None:
    parent = _parent()
    view = FrozenMemoryInterventionPolicy(parent, MemoryInterventionSpec("full_history"))
    history = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.int64)
    assert torch.equal(view(history, 4), parent(history, 4))


def test_shuffle_is_deterministic_count_preserving_and_nontrivial() -> None:
    parent = _parent()
    view = FrozenMemoryInterventionPolicy(
        parent, MemoryInterventionSpec("history_shuffle", shuffle_seed=17)
    )
    history = torch.tensor([[0, 0, 1, 0, 1, 1]], dtype=torch.int64)
    first = view.transformed_history(history, 6)
    second = view.transformed_history(history, 6)
    assert torch.equal(first, second)
    assert sorted(first[0].tolist()) == sorted(history[0].tolist())
    assert view.history_indices(6) != tuple(range(6))


def test_shuffle_is_prefix_consistent_when_a_new_observation_arrives() -> None:
    parent = _parent()
    view = FrozenMemoryInterventionPolicy(
        parent, MemoryInterventionSpec("history_shuffle", shuffle_seed=17)
    )
    shorter = view.history_indices(12)
    longer = view.history_indices(13)
    assert tuple(index for index in longer if index < 12) == shorter


def test_shuffle_and_all_views_are_prefix_causal() -> None:
    parent = _parent()
    prefix = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=torch.int64)
    for spec in (
        MemoryInterventionSpec("history_shuffle", shuffle_seed=29),
        MemoryInterventionSpec("history_truncation", history_length=2),
        MemoryInterventionSpec("periodic_hidden_reset", reset_period=3),
        MemoryInterventionSpec("last_outcome_only"),
    ):
        view = FrozenMemoryInterventionPolicy(parent, spec)
        assert torch.equal(view(prefix, 4)[0], view(prefix, 4)[1])
        with pytest.raises(ValueError, match="width"):
            view(prefix, 3)


def test_truncation_uses_exact_sliding_tail() -> None:
    parent = _parent()
    view = FrozenMemoryInterventionPolicy(
        parent, MemoryInterventionSpec("history_truncation", history_length=3)
    )
    assert view.history_indices(2) == (0, 1)
    assert view.history_indices(6) == (3, 4, 5)


def test_periodic_reset_is_block_based_not_sliding_window() -> None:
    parent = _parent()
    view = FrozenMemoryInterventionPolicy(
        parent, MemoryInterventionSpec("periodic_hidden_reset", reset_period=3)
    )
    assert view.history_indices(3) == (0, 1, 2)
    assert view.history_indices(4) == (3,)
    assert view.history_indices(6) == (3, 4, 5)
    assert view.history_indices(7) == (6,)


def test_reset_period_one_equals_latest_only_at_every_depth() -> None:
    parent = _parent()
    reset = FrozenMemoryInterventionPolicy(
        parent, MemoryInterventionSpec("periodic_hidden_reset", reset_period=1)
    )
    latest = FrozenMemoryInterventionPolicy(
        parent, MemoryInterventionSpec("last_outcome_only")
    )
    history = torch.tensor([[0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.int64)
    for depth in range(5):
        assert reset.history_indices(depth) == latest.history_indices(depth)
        assert torch.equal(reset(history[:, :depth], depth), latest(history[:, :depth], depth))


def test_interventions_do_not_mutate_or_add_trainable_parent_weights() -> None:
    parent = _parent()
    before = state_dict_sha256(parent.state_dict())
    view = FrozenMemoryInterventionPolicy(
        parent, MemoryInterventionSpec("history_truncation", history_length=2)
    )
    history = torch.tensor([[0, 1, 1, 0]], dtype=torch.int64)
    with torch.no_grad():
        view(history, 4)
    assert view.parameter_count == parent.parameter_count == 72_853
    assert state_dict_sha256(parent.state_dict()) == before


def test_parent_and_spec_types_are_checked() -> None:
    parent = _parent()
    with pytest.raises(TypeError, match="parent"):
        FrozenMemoryInterventionPolicy(object(), MemoryInterventionSpec("full_history"))
    with pytest.raises(TypeError, match="spec"):
        FrozenMemoryInterventionPolicy(parent, object())  # type: ignore[arg-type]
