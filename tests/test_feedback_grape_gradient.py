from __future__ import annotations

import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch is optional in the minimal recovery interpreter", allow_module_level=True)

from physics.feedback_grape_gradient import (
    COMPACT_POLICY_PARAMETER_NAMES,
    FEEDBACK_GRAPE_GRADIENT_SCOPE,
    CompactHistoryPolicy,
    GradientValidationConfig,
    default_policy_parameters,
    enumerate_binary_trajectories,
    exact_gradient_decomposition,
    finite_difference_gradient,
    finite_difference_step_sweep,
    monte_carlo_gradient_validation,
    run_feedback_grape_gradient_validation,
)

import torch


@pytest.fixture(scope="module")
def config() -> GradientValidationConfig:
    return GradientValidationConfig(
        cutoff=6,
        monte_carlo_batch_size=128,
        monte_carlo_repeats=8,
    )


@pytest.fixture(scope="module")
def exact(config: GradientValidationConfig):
    return exact_gradient_decomposition(config)


def test_config_freezes_small_model_and_rejects_scope_erasure() -> None:
    config = GradientValidationConfig()
    assert config.half_cycles == 2
    assert config.branch_count == 4
    assert config.protocol_id == "PROTO-SBS-MAIN"
    assert "not teacher training" in config.scope


@pytest.mark.parametrize(
    "change, error",
    [
        ({"cutoff": 3}, ValueError),
        ({"cutoff": 17}, ValueError),
        ({"full_cycles": 0}, ValueError),
        ({"full_cycles": 3}, ValueError),
        ({"finite_difference_step": 0.0}, ValueError),
        ({"finite_difference_step": 0.01}, ValueError),
        ({"monte_carlo_batch_size": 63}, ValueError),
        ({"monte_carlo_repeats": 7}, ValueError),
        ({"seed": True}, TypeError),
        ({"device": "tpu"}, ValueError),
        ({"protocol_id": "wrong"}, ValueError),
        ({"scope": "demo"}, ValueError),
    ],
)
def test_config_negative_paths(change: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        GradientValidationConfig(**change)


def test_binary_trajectory_enumeration_is_complete_and_ordered() -> None:
    outcomes = enumerate_binary_trajectories(2)
    assert outcomes.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    four = enumerate_binary_trajectories(4)
    assert four.shape == (16, 4)
    assert len({tuple(row) for row in four.tolist()}) == 16
    with pytest.raises(ValueError):
        enumerate_binary_trajectories(13)


def test_default_policy_parameters_are_three_finite_trainable_scalars() -> None:
    theta = default_policy_parameters()
    assert theta.shape == (3,)
    assert theta.requires_grad
    assert torch.all(torch.isfinite(theta))
    assert COMPACT_POLICY_PARAMETER_NAMES == (
        "static_control_residual",
        "latest_outcome_response",
        "history_mean_response",
    )


def test_compact_policy_is_causal_and_history_sensitive() -> None:
    theta = default_policy_parameters()
    policy = CompactHistoryPolicy(theta)
    initial = policy(torch.empty((2, 0), dtype=torch.int64), 0)
    after = policy(torch.tensor([[0], [1]], dtype=torch.int64), 1)
    assert initial.shape == (2, 15)
    assert after.shape == (2, 15)
    assert torch.allclose(initial[0], initial[1])
    assert not torch.allclose(after[0], after[1])
    after.sum().backward()
    assert theta.grad is not None
    assert torch.all(torch.isfinite(theta.grad))


def test_compact_policy_rejects_wrong_parameter_and_history_shapes() -> None:
    with pytest.raises(TypeError):
        CompactHistoryPolicy(np.zeros(3))
    with pytest.raises(ValueError):
        CompactHistoryPolicy(torch.zeros(2))
    with pytest.raises(ValueError):
        CompactHistoryPolicy(torch.tensor([0.0, float("nan"), 0.0]))
    policy = CompactHistoryPolicy(default_policy_parameters())
    with pytest.raises(TypeError):
        policy([[]], 0)
    with pytest.raises(ValueError):
        policy(torch.empty((1, 0), dtype=torch.int64), 1)


def test_exact_decomposition_normalizes_every_branch(exact) -> None:
    assert exact.trajectory_probability_sum == pytest.approx(1.0, abs=2.0e-12)
    assert 0.0 < exact.expected_return < 1.0


def test_both_feedback_grape_terms_are_nonzero(exact) -> None:
    assert np.linalg.norm(exact.reward_path_gradient) > 1.0e-8
    assert np.linalg.norm(exact.score_path_gradient) > 1.0e-8
    assert np.sign(exact.reward_path_gradient[1]) == np.sign(exact.score_path_gradient[1])


def test_exact_gradient_equals_reward_plus_score(exact) -> None:
    np.testing.assert_allclose(
        exact.exact_gradient,
        np.asarray(exact.reward_path_gradient) + np.asarray(exact.score_path_gradient),
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    assert exact.decomposition_absolute_error < 2.0e-12


def test_constant_baseline_is_exactly_unbiased_under_enumeration(exact) -> None:
    np.testing.assert_allclose(
        exact.baseline_score_gradient,
        exact.score_path_gradient,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    assert exact.baseline_invariance_error < 2.0e-12


def test_probability_score_expectation_is_zero(exact) -> None:
    np.testing.assert_allclose(
        exact.probability_normalization_score,
        0.0,
        atol=2.0e-12,
    )
    assert exact.score_normalization_error < 2.0e-12


def test_exact_decomposition_requires_trainable_theta(
    config: GradientValidationConfig,
) -> None:
    with pytest.raises(ValueError):
        exact_gradient_decomposition(config, default_policy_parameters(requires_grad=False))


def test_central_finite_difference_matches_autograd(
    config: GradientValidationConfig, exact
) -> None:
    result = finite_difference_gradient(
        config,
        exact.exact_gradient,
        exact.reward_path_gradient,
        exact.score_path_gradient,
    )
    assert result.step == 1.0e-5
    assert result.maximum_absolute_error < 1.0e-8
    assert result.relative_l2_error < 1.0e-7
    np.testing.assert_allclose(
        result.autograd_gradient,
        result.finite_difference_gradient,
        atol=1.0e-8,
        rtol=1.0e-7,
    )
    assert result.reward_path_relative_l2_error < 1.0e-7
    assert result.score_path_relative_l2_error < 1.0e-7
    np.testing.assert_allclose(
        result.reward_path_autograd_gradient,
        result.reward_path_finite_difference_gradient,
        atol=1.0e-8,
        rtol=1.0e-7,
    )
    np.testing.assert_allclose(
        result.score_path_autograd_gradient,
        result.score_path_finite_difference_gradient,
        atol=1.0e-8,
        rtol=1.0e-7,
    )


def test_finite_difference_step_sweep_is_stable(
    config: GradientValidationConfig, exact
) -> None:
    results = finite_difference_step_sweep(config, exact)
    assert [item.step for item in results] == [3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5]
    assert max(item.relative_l2_error for item in results) < 1.0e-6
    assert max(item.reward_path_relative_l2_error for item in results) < 1.0e-6
    assert max(item.score_path_relative_l2_error for item in results) < 1.0e-6


def test_finite_difference_rejects_gradient_shape(
    config: GradientValidationConfig,
) -> None:
    with pytest.raises(ValueError):
        finite_difference_gradient(
            config,
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        )


def test_two_cycle_sixteen_branch_decomposition_still_closes() -> None:
    two_cycle = GradientValidationConfig(
        cutoff=4,
        full_cycles=2,
        monte_carlo_batch_size=64,
        monte_carlo_repeats=8,
    )
    result = exact_gradient_decomposition(two_cycle)
    assert result.trajectory_probability_sum == pytest.approx(1.0, abs=2.0e-11)
    assert result.decomposition_absolute_error < 2.0e-10
    assert result.score_normalization_error < 2.0e-10
    assert np.linalg.norm(result.score_path_gradient) > 1.0e-8


def test_sampled_two_term_estimator_matches_exact_and_baseline_reduces_variance(
    config: GradientValidationConfig, exact
) -> None:
    result = monte_carlo_gradient_validation(
        config, exact.exact_gradient, exact.expected_return
    )
    assert result.total_trajectories == 1024
    assert result.maximum_absolute_z_score < 3.5
    assert result.baseline_variance_ratio < 0.2
    assert result.baseline_score_trace_variance < result.plain_score_trace_variance
    assert 0.0 < result.mean_ground_outcome_fraction < 1.0


def test_monte_carlo_gradient_rejects_wrong_exact_shape(
    config: GradientValidationConfig, exact
) -> None:
    with pytest.raises(ValueError):
        monte_carlo_gradient_validation(config, exact.exact_gradient[:2], exact.expected_return)


def test_validation_payload_is_evidence_backed_and_fail_closed(tmp_path) -> None:
    target = tmp_path / "gradient.json"
    payload = run_feedback_grape_gradient_validation(
        device="cpu",
        output=target,
        monte_carlo_batch_size=128,
        monte_carlo_repeats=8,
    )
    assert payload["status"] == "PASS"
    assert len(payload["checks"]) == 15
    assert all(payload["checks"].values())
    assert payload["finite_difference"]["relative_l2_error"] < 1.0e-7
    assert payload["finite_difference"]["reward_path_relative_l2_error"] < 1.0e-7
    assert payload["finite_difference"]["score_path_relative_l2_error"] < 1.0e-7
    assert len(payload["finite_difference_step_sweep"]) == 4
    assert payload["exact_decomposition"]["decomposition_absolute_error"] < 1.0e-12
    assert payload["monte_carlo"]["baseline_variance_ratio"] < 0.2
    assert "not an optimized or trained RNN teacher" in payload["forbidden_claims"]
    assert payload["scope"] == FEEDBACK_GRAPE_GRADIENT_SCOPE
    assert target.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_exact_gradient_matches_cpu(exact) -> None:
    cuda_config = GradientValidationConfig(
        cutoff=6,
        device="cuda",
        monte_carlo_batch_size=64,
        monte_carlo_repeats=8,
    )
    cuda = exact_gradient_decomposition(cuda_config)
    np.testing.assert_allclose(cuda.exact_gradient, exact.exact_gradient, atol=2.0e-11, rtol=2.0e-11)
    assert cuda.decomposition_absolute_error < 2.0e-10


def test_scope_forbids_downstream_claims() -> None:
    lowered = FEEDBACK_GRAPE_GRADIENT_SCOPE.lower()
    for phrase in (
        "not teacher training",
        "feasibility envelope",
        "protocol ranking",
        "device calibration",
        "hardware evidence",
    ):
        assert phrase in lowered
