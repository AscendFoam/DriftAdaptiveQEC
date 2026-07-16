from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from physics.trajectory_lookup_control_oracle import (
    ACTION_CONTRACT_ID,
    CONTROL_ORACLE_ROLE_ID,
    CausalHistoryLookupPolicy,
    TimeIndexedOpenLoopPolicy,
    TrajectoryLookupConfig,
    build_policy,
    enumerate_terminal_trajectories,
    evaluate_exact_policy,
    exact_expected_fidelity_tensor,
    expand_open_loop_as_lookup,
    history_node_count,
    load_policy_from_state,
    optimize_policy_once,
    resource_growth_row,
    standard_nominal_policy,
    terminal_branch_count,
    validate_production_design,
)

torch = pytest.importorskip("torch")


def pilot_config(**changes: object) -> TrajectoryLookupConfig:
    base = TrajectoryLookupConfig(
        full_cycles=1,
        cutoff=6,
        confirmation_cutoff=6,
        epochs=6,
        restart_seeds=(7,),
        device="cpu",
        real_dtype="float64",
    )
    return replace(base, **changes)


def test_binary_tree_counts_and_action_scalars_are_exact() -> None:
    assert history_node_count(4) == 15
    assert terminal_branch_count(4) == 16
    row = resource_growth_row(2, cutoff=12)
    assert row["half_cycles"] == 4
    assert row["causal_history_nodes"] == 15
    assert row["lookup_action_scalars"] == 225
    assert row["terminal_branches"] == 16


def test_resource_projection_is_exponential_not_linear() -> None:
    rows = [resource_growth_row(cycle) for cycle in range(1, 6)]
    assert [row["terminal_branches"] for row in rows] == [4, 16, 64, 256, 1024]
    assert [row["causal_history_nodes"] for row in rows] == [3, 15, 63, 255, 1023]
    assert rows[-1]["lookup_action_scalars"] == 15_345


def test_config_preserves_control_oracle_and_action_contract() -> None:
    config = pilot_config()
    assert config.role_id == CONTROL_ORACLE_ROLE_ID
    assert config.action_contract_id == ACTION_CONTRACT_ID
    with pytest.raises(ValueError, match="role_id"):
        replace(config, role_id="decoder_oracle")
    with pytest.raises(ValueError, match="explicit"):
        replace(config, full_cycles=4)


def test_production_gate_rejects_pilot_and_accepts_registered_design() -> None:
    with pytest.raises(ValueError, match="production"):
        validate_production_design(pilot_config())
    validate_production_design(TrajectoryLookupConfig(device="cpu"))


def test_terminal_enumeration_has_all_unique_binary_histories() -> None:
    outcomes = enumerate_terminal_trajectories(4)
    assert tuple(outcomes.shape) == (16, 4)
    assert len({tuple(row) for row in outcomes.tolist()}) == 16
    assert outcomes[0].tolist() == [0, 0, 0, 0]
    assert outcomes[-1].tolist() == [1, 1, 1, 1]


def test_lookup_node_mapping_covers_every_prefix_once() -> None:
    policy = CausalHistoryLookupPolicy(
        4, device="cpu", dtype=torch.float64
    )
    outcomes = enumerate_terminal_trajectories(4)
    covered: set[int] = set()
    for depth in range(4):
        indices = policy.node_indices(outcomes[:, :depth], depth)
        expected = set(range(2**depth - 1, 2 ** (depth + 1) - 1))
        assert set(indices.tolist()) == expected
        covered.update(indices.tolist())
    assert covered == set(range(15))


def test_lookup_is_prefix_invariant_to_unobserved_suffix() -> None:
    policy = CausalHistoryLookupPolicy(
        4, device="cpu", dtype=torch.float64
    )
    with torch.no_grad():
        policy.raw_table.copy_(
            torch.arange(15 * 15, dtype=torch.float64).reshape(15, 15)
        )
    outcomes = enumerate_terminal_trajectories(4)
    depth = 2
    actions = policy(outcomes[:, :depth], depth)
    for prefix in ((0, 0), (0, 1), (1, 0), (1, 1)):
        rows = [i for i, row in enumerate(outcomes.tolist()) if tuple(row[:depth]) == prefix]
        assert torch.all(actions[rows] == actions[rows[0]])


def test_lookup_rejects_future_width_and_nonbinary_history() -> None:
    policy = CausalHistoryLookupPolicy(4, device="cpu", dtype=torch.float64)
    with pytest.raises(ValueError, match="width"):
        policy(torch.zeros((2, 2), dtype=torch.int64), 1)
    with pytest.raises(ValueError, match="g=0 or e=1"):
        policy(torch.tensor([[0, 2]], dtype=torch.int64), 2)
    with pytest.raises(ValueError, match="outside"):
        policy(torch.zeros((1, 4), dtype=torch.int64), 4)


def test_time_indexed_open_loop_ignores_history_values() -> None:
    policy = TimeIndexedOpenLoopPolicy(2, device="cpu", dtype=torch.float64)
    with torch.no_grad():
        policy.raw_table[1].copy_(torch.arange(15, dtype=torch.float64))
    history = torch.tensor([[0], [1]], dtype=torch.int64)
    actions = policy(history, 1)
    assert torch.all(actions[0] == actions[1])


def test_open_loop_policy_embeds_exactly_in_lookup_tree() -> None:
    config = pilot_config()
    open_loop = build_policy(
        config,
        "time_indexed_open_loop",
        seed=29,
        initialization_std=0.04,
    )
    expanded = expand_open_loop_as_lookup(config, open_loop.raw_table.detach().cpu())
    lookup = build_policy(
        config,
        "causal_history_lookup",
        seed=0,
        initialization_std=0.0,
    )
    with torch.no_grad():
        lookup.raw_table.copy_(expanded)
    open_value = evaluate_exact_policy(config, open_loop).expected_fidelity
    lookup_value = evaluate_exact_policy(config, lookup).expected_fidelity
    assert lookup_value == pytest.approx(open_value, abs=2.0e-12)


def test_exact_standard_evaluation_normalizes_all_branches() -> None:
    config = pilot_config()
    result = evaluate_exact_policy(config, standard_nominal_policy(config))
    assert result.family == "standard_nominal"
    assert result.trajectory_probability_sum == pytest.approx(1.0, abs=2.0e-12)
    assert len(result.branch_rows) == 4
    assert result.minimum_trajectory_probability > 0.0
    assert result.maximum_trace_error < 2.0e-12
    assert result.minimum_final_eigenvalue > -2.0e-11


def test_exact_objective_gradient_reaches_every_history_node() -> None:
    config = pilot_config()
    policy = build_policy(
        config,
        "causal_history_lookup",
        seed=11,
        initialization_std=0.02,
    )
    objective = exact_expected_fidelity_tensor(config, policy)
    objective.backward()
    node_norms = torch.linalg.vector_norm(policy.raw_table.grad, dim=1)
    assert torch.all(torch.isfinite(node_norms))
    assert torch.all(node_norms > 1.0e-12)


def test_lookup_gradient_matches_central_finite_difference() -> None:
    config = pilot_config()
    policy = build_policy(
        config,
        "causal_history_lookup",
        seed=13,
        initialization_std=0.02,
    )
    objective = exact_expected_fidelity_tensor(config, policy)
    objective.backward()
    automatic = float(policy.raw_table.grad[1, 3])
    step = 1.0e-5
    with torch.no_grad():
        original = float(policy.raw_table[1, 3])
        policy.raw_table[1, 3] = original + step
        plus = float(exact_expected_fidelity_tensor(config, policy))
        policy.raw_table[1, 3] = original - step
        minus = float(exact_expected_fidelity_tensor(config, policy))
        policy.raw_table[1, 3] = original
    finite = (plus - minus) / (2.0 * step)
    assert automatic == pytest.approx(finite, rel=2.0e-4, abs=2.0e-7)


@pytest.mark.parametrize("family", ["time_indexed_open_loop", "causal_history_lookup"])
def test_short_exact_optimization_is_real_and_changes_all_nodes(family: str) -> None:
    config = pilot_config(epochs=8)
    run, state = optimize_policy_once(
        config,
        family,
        seed=17,
        initialization_std=0.02,
    )
    assert run.best_expected_fidelity > run.initial_expected_fidelity + 1.0e-5
    assert run.gradient_covered_nodes == run.action_node_count
    assert run.changed_nodes == run.action_node_count
    assert len(run.trace) == config.epochs + 1
    replay = load_policy_from_state(config, state)
    replay_value = evaluate_exact_policy(config, replay).expected_fidelity
    assert replay_value == pytest.approx(run.best_expected_fidelity, abs=2.0e-12)


def test_checkpoint_shape_mismatch_fails_closed() -> None:
    config = pilot_config()
    state = {
        "family": "causal_history_lookup",
        "seed": 0,
        "raw_table": np.zeros((1, 15)),
    }
    with pytest.raises(ValueError, match="shape"):
        load_policy_from_state(config, state)


def test_optimizer_rejects_malformed_warm_start() -> None:
    config = pilot_config(epochs=1)
    with pytest.raises(ValueError, match="initial_raw_table shape"):
        optimize_policy_once(
            config,
            "causal_history_lookup",
            seed=3,
            initialization_std=0.0,
            initial_raw_table=np.zeros((1, 15)),
        )
