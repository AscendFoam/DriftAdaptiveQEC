from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from physics.exponential_recurrence_control import (
    EXPONENTIAL_RECURRENCE_SCOPE,
    ExponentialRecurrenceConfig,
    FixedPointExponentialPolicy,
    build_policy,
    exact_expected_fidelity,
    load_policy_state,
    optimize_recurrence_once,
    state_dict_sha256,
    validate_production_design,
)

torch = pytest.importorskip("torch")


def pilot_config(**changes: object) -> ExponentialRecurrenceConfig:
    base = ExponentialRecurrenceConfig(
        full_cycles=1,
        cutoff=6,
        confirmation_cutoff=6,
        phase_one_epochs=8,
        refinement_epochs=4,
        restart_seeds=(7,),
        device="cpu",
    )
    return replace(base, **changes)


def test_registered_policy_has_75_trainable_and_105_stored_scalars() -> None:
    policy = build_policy(pilot_config(), seed=1, initialization_std=0.0)
    assert policy.parameter_count == 75
    assert policy.stored_scalar_count == 105
    assert "uncalibrated leakage-safe branch" in EXPONENTIAL_RECURRENCE_SCOPE


def test_decay_values_are_strictly_stable_and_branch_specific() -> None:
    config = pilot_config()
    policy = build_policy(config, seed=2, initialization_std=0.03)
    decay = policy.all_decays().detach().numpy()
    assert decay.shape == (3, 15)
    assert np.all(decay > 0.0)
    assert np.all(decay < 1.0)
    assert np.allclose(decay[2], config.leakage_decay)


@pytest.mark.parametrize("outcome", [0, 1, 2])
def test_repeated_outcome_matches_closed_form_saturation(outcome: int) -> None:
    config = pilot_config()
    policy = build_policy(config, seed=3, initialization_std=0.0)
    with torch.no_grad():
        policy.initial_raw.copy_(torch.linspace(-0.2, 0.2, 15))
        policy.ge_saturation_raw[0].fill_(0.4)
        policy.ge_saturation_raw[1].fill_(-0.3)
        policy.leakage_saturation_raw.fill_(0.1)
    history = torch.full((1, 2), outcome, dtype=torch.int64)
    actual = policy.state_after_history(history)[0]
    initial = policy.initial_raw
    saturation = policy.all_saturations()[outcome]
    decay = policy.all_decays()[outcome]
    expected = decay**2 * initial + (1.0 - decay**2) * saturation
    assert torch.allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_outcome_switch_uses_previous_state_not_original_state() -> None:
    policy = build_policy(pilot_config(), seed=4, initialization_std=0.0)
    with torch.no_grad():
        policy.initial_raw.fill_(0.2)
        policy.ge_saturation_raw[0].fill_(0.8)
        policy.ge_saturation_raw[1].fill_(-0.6)
    decay = policy.all_decays()
    after_g = decay[0] * policy.initial_raw + (1.0 - decay[0]) * policy.all_saturations()[0]
    expected = decay[1] * after_g + (1.0 - decay[1]) * policy.all_saturations()[1]
    actual = policy.state_after_history(torch.tensor([[0, 1]]))[0]
    assert torch.allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_policy_is_prefix_causal_and_rejects_future_width() -> None:
    policy = build_policy(pilot_config(), seed=5, initialization_std=0.03)
    full = torch.tensor([[0, 1], [0, 0], [1, 1], [1, 0]], dtype=torch.int64)
    first = policy(full[:, :1], 1)
    assert torch.allclose(first[0], first[1])
    assert torch.allclose(first[2], first[3])
    with pytest.raises(ValueError, match="width"):
        policy(full, 1)
    with pytest.raises(ValueError, match="g=0, e=1, leakage=2"):
        policy.state_after_history(torch.tensor([[3]], dtype=torch.int64))


def test_fixed_point_recurrence_tracks_float_for_mixed_g_e_leakage() -> None:
    config = pilot_config()
    policy = build_policy(config, seed=6, initialization_std=0.08)
    fixed = FixedPointExponentialPolicy(policy)
    histories = torch.tensor(
        [[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [2, 0]], dtype=torch.int64
    )
    floating = policy.state_after_history(histories)
    quantized = fixed.state_after_history(histories)
    assert torch.max(torch.abs(floating.detach() - quantized)).item() < 2.0e-4


def test_fixed_point_rejects_insufficient_state_width() -> None:
    policy = build_policy(pilot_config(), seed=6, initialization_std=0.0)
    with pytest.raises(ValueError, match="state_total_bits"):
        FixedPointExponentialPolicy(
            policy, state_fraction_bits=14, state_total_bits=15
        )


def test_exact_objective_has_finite_nonzero_gradient() -> None:
    config = pilot_config(full_cycles=2)
    policy = build_policy(config, seed=8, initialization_std=0.03)
    objective = exact_expected_fidelity(config, policy)
    objective.backward()
    gradients = torch.cat([parameter.grad.reshape(-1) for parameter in policy.parameters()])
    assert torch.all(torch.isfinite(gradients))
    assert torch.linalg.vector_norm(gradients).item() > 1.0e-6
    assert int(torch.sum(torch.abs(gradients) > 1.0e-14)) == 75


def test_short_optimization_is_real_and_checkpoint_roundtrips() -> None:
    config = pilot_config(full_cycles=2, phase_one_epochs=10)
    run, state = optimize_recurrence_once(
        config,
        seed=9,
        phase="test",
        epochs=10,
        learning_rate=0.02,
        initialization_std=0.03,
    )
    assert run.best_expected_fidelity > run.initial_expected_fidelity + 1.0e-4
    assert run.gradient_covered_scalars == 75
    assert run.changed_scalars == 75
    replay = load_policy_state(config, state)
    replay_value = float(exact_expected_fidelity(config, replay).detach())
    assert replay_value == pytest.approx(run.best_expected_fidelity, abs=2.0e-12)
    assert state_dict_sha256(state) == state_dict_sha256(replay.state_dict())


def test_fixed_point_physical_objective_is_close_for_short_policy() -> None:
    config = pilot_config()
    policy = build_policy(config, seed=10, initialization_std=0.03)
    fixed = FixedPointExponentialPolicy(policy)
    floating = float(exact_expected_fidelity(config, policy).detach())
    quantized = float(exact_expected_fidelity(config, fixed))
    assert abs(floating - quantized) < 3.0e-4


def test_production_design_gate_rejects_pilot_and_accepts_default() -> None:
    with pytest.raises(ValueError, match="production"):
        validate_production_design(pilot_config())
    validate_production_design(ExponentialRecurrenceConfig(device="cpu"))
