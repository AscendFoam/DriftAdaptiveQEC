from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from physics.differentiable_sbs_trajectory import nominal_sbs_parameters
from physics.puviani_paper_causal import (
    PAPER_CAUSAL_FEEDBACK_TIMELINE,
    PaperCausalDirectionalRankingConfig,
    PaperCausalMFPolicy,
    PaperCausalNMFPolicy,
    PaperCausalSBSConfig,
    PaperCausalSBSTrajectorySimulator,
    applied_control_penalties,
    applied_control_penalties_per_trajectory,
    build_policy,
)

torch = pytest.importorskip("torch")

from cnn_fpga.benchmark.puviani_paper_constrained_artifacts import (
    _feedback_grape_loss,
)


def _simulator(*, batch_size: int = 1) -> PaperCausalSBSTrajectorySimulator:
    return PaperCausalSBSTrajectorySimulator(
        PaperCausalSBSConfig(
            cutoff=4,
            full_cycles=1,
            batch_size=batch_size,
            grid_points=1025,
            device="cpu",
            real_dtype="float64",
        )
    )


class _TracingPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, torch.Tensor]] = []

    def reset_rollout(self, *, batch_size: int, device: object, dtype: object) -> None:
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def __call__(self, history: torch.Tensor, half_index: int) -> torch.Tensor:
        return self.step_rollout(history, half_index)

    def step_rollout(self, history: torch.Tensor, half_index: int) -> torch.Tensor:
        self.calls.append((half_index, history.detach().clone()))
        value = torch.full(
            (history.shape[0], 15),
            0.01 * half_index,
            dtype=torch.float64,
        )
        if half_index:
            value[:, 0] += 0.1 * history[:, -1].to(torch.float64)
            value[:, 14] -= 0.05 * history[:, -1].to(torch.float64)
        return value


class _CompactCausalGradientPolicy:
    """Small history-sensitive policy used only for exact gradient truth."""

    def __init__(self, theta: torch.Tensor) -> None:
        self.theta = theta
        control_index = torch.arange(
            1, 16, dtype=theta.dtype, device=theta.device
        )
        self.static_basis = 0.22 * torch.sin(0.41 * control_index)
        self.latest_basis = 0.18 * torch.cos(0.67 * (control_index - 0.5))
        self.memory_basis = 0.16 * torch.sin(0.29 * (control_index + 0.7))

    def __call__(
        self, history: torch.Tensor, half_index: int
    ) -> torch.Tensor:
        if history.ndim != 2 or history.shape[1] != half_index:
            raise ValueError("history must contain exactly the causal prefix")
        if half_index == 0:
            latest = torch.zeros(
                (history.shape[0], 1),
                dtype=self.theta.dtype,
                device=self.theta.device,
            )
            history_mean = latest
        else:
            # Match the production GQF convention: g=0 -> +1, e=1 -> -1.
            signed = 1.0 - 2.0 * history.to(self.theta.dtype)
            latest = signed[:, -1:]
            history_mean = signed.mean(dim=1, keepdim=True)
        return (
            self.theta[0] * self.static_basis[None, :]
            + self.theta[1] * latest * self.latest_basis[None, :]
            + self.theta[2] * history_mean * self.memory_basis[None, :]
        )


def _four_branch_gradient_tree(
    simulator: PaperCausalSBSTrajectorySimulator,
    theta: torch.Tensor,
) -> object:
    branches = torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int64
    )
    return simulator.run(
        control_policy=_CompactCausalGradientPolicy(theta),
        forced_outcomes=branches,
        seed=271828,
        record_cycle_metrics=False,
    )


def test_paper_causal_policy_is_called_once_for_every_prefix_and_cached() -> None:
    simulator = _simulator(batch_size=2)
    policy = _TracingPolicy()
    result = simulator.run(
        control_policy=policy,
        forced_outcomes=[[0, 1], [1, 0]],
    )

    assert [index for index, _ in policy.calls] == [0, 1, 2]
    assert [tuple(history.shape) for _, history in policy.calls] == [
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    assert policy.calls[1][1].tolist() == [[0], [1]]
    assert policy.calls[2][1].tolist() == [[0, 1], [1, 0]]
    assert result.decision_physical_controls.shape == (2, 3, 15)
    assert result.layer_applied_physical_controls.shape == (2, 2, 14)
    assert result.virtual_rotation_applied_physical_controls.shape == (2, 2, 1)
    assert torch.equal(
        result.terminal_virtual_rotation,
        result.decision_physical_controls[:, -1, 14],
    )
    assert result.feedback_timeline == PAPER_CAUSAL_FEEDBACK_TIMELINE
    assert result.resource_profile.simulated_physical_time_ns == 10_000


def test_operation_trace_matches_measure_idle_reset_next_vr_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = _simulator()
    operations: list[str] = []

    original_idle = simulator._apply_idle
    original_layer = simulator._layer
    original_measure = simulator._measure_collapsed
    original_reset = simulator._reset_measured_ancilla
    original_rotation = simulator._virtual_rotation

    def idle(state: object, phase_id: str) -> object:
        operations.append(f"idle:{phase_id}")
        return original_idle(state, phase_id)

    def layer(state: object, controls: object, layer_index: int) -> object:
        operations.append(f"layer:{layer_index}")
        return original_layer(state, controls, layer_index)

    def measure(*args: object, **kwargs: object) -> object:
        operations.append("measure")
        return original_measure(*args, **kwargs)

    def reset(*args: object, **kwargs: object) -> object:
        operations.append("reset")
        return original_reset(*args, **kwargs)

    def rotation(*args: object, **kwargs: object) -> object:
        operations.append("virtual_rotation")
        return original_rotation(*args, **kwargs)

    monkeypatch.setattr(simulator, "_apply_idle", idle)
    monkeypatch.setattr(simulator, "_layer", layer)
    monkeypatch.setattr(simulator, "_measure_collapsed", measure)
    monkeypatch.setattr(simulator, "_reset_measured_ancilla", reset)
    monkeypatch.setattr(simulator, "_virtual_rotation", rotation)
    simulator.run(forced_outcomes=[[0, 1]])

    one_half = [
        "idle:entering_cycle",
        "layer:1",
        "idle:layer_1",
        "layer:2",
        "idle:layer_2",
        "layer:3",
        "idle:layer_3",
        "layer:4",
        "idle:layer_4",
        "measure",
        "idle:measurement_and_reset",
        "reset",
        "virtual_rotation",
        "idle:virtual_rotation_and_idle",
    ]
    assert operations == one_half + one_half


def test_paper_causal_four_branch_probability_is_normalized() -> None:
    simulator = _simulator(batch_size=4)
    result = simulator.run(
        forced_outcomes=[[0, 0], [0, 1], [1, 0], [1, 1]]
    )
    assert result.trajectory_probability.sum().item() == pytest.approx(
        1.0, abs=2.0e-10
    )
    assert result.maximum_trace_error < 1.0e-10
    assert result.minimum_final_eigenvalue > -1.0e-10


def test_open_loop_requires_all_h_plus_one_decisions() -> None:
    simulator = _simulator()
    with pytest.raises(ValueError, match="paper-causal raw_corrections"):
        simulator.run(torch.zeros((1, 2, 15), dtype=torch.float64))
    result = simulator.run(
        torch.zeros((1, 3, 15), dtype=torch.float64),
        forced_outcomes=[[0, 1]],
    )
    assert result.decision_physical_controls.shape == (1, 3, 15)


def test_gqf_sign_zero_sentinel_and_cached_gru_match_manual_unroll() -> None:
    config = PaperCausalDirectionalRankingConfig(
        cutoff=4,
        confirmation_cutoff=4,
        full_cycles=1,
        train_epochs=1,
        train_batch_size=2,
        validation_batch_size=2,
        test_batch_size=2,
        confirmation_batch_size=2,
        validation_interval=1,
        training_seeds=(11,),
        validation_seeds=(101,),
        test_seeds=(211,),
        confirmation_seeds=(307,),
        bootstrap_repetitions=10,
        device="cpu",
    )
    nmf = build_policy("nmf", config, 11)
    assert isinstance(nmf, PaperCausalNMFPolicy)
    nmf.reset_rollout(batch_size=2, device=torch.device("cpu"), dtype=torch.float64)
    empty = torch.empty((2, 0), dtype=torch.int64)
    cached_zero = nmf.step_rollout(empty, 0)
    hidden0 = torch.zeros((2, 10), dtype=torch.float64)
    hidden1 = nmf.gru(torch.zeros((2, 1), dtype=torch.float64), hidden0)
    manual_zero = nmf.output(torch.tanh(nmf.dense2(torch.tanh(nmf.dense1(hidden1)))))
    assert torch.equal(cached_zero, manual_zero)

    history = torch.tensor([[0], [1]], dtype=torch.int64)
    cached_one = nmf.step_rollout(history, 1)
    token = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    hidden2 = nmf.gru(token, hidden1)
    manual_one = nmf.output(torch.tanh(nmf.dense2(torch.tanh(nmf.dense1(hidden2)))))
    assert torch.equal(cached_one, manual_one)
    assert torch.equal(cached_one, nmf(history, 1))
    with pytest.raises(RuntimeError, match="exactly once in order"):
        nmf.step_rollout(history, 1)

    nmf.reset_rollout(batch_size=2, device=torch.device("cpu"), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="exactly once in order"):
        nmf.step_rollout(history, 1)

    mf = build_policy("mf", config, 11)
    assert isinstance(mf, PaperCausalMFPolicy)
    manual_mf = mf.output(
        torch.tanh(mf.dense2(torch.tanh(mf.dense1(token))))
    )
    assert torch.equal(mf(history, 1), manual_mf)


def test_unused_d0_vr_and_terminal_layers_have_zero_gradient_and_penalty() -> None:
    simulator = _simulator()
    raw = torch.linspace(
        -0.08, 0.08, steps=45, dtype=torch.float64
    ).reshape(1, 3, 15)
    raw.requires_grad_(True)
    result = simulator.run(raw, forced_outcomes=[[0, 1]])
    loss = result.reward.mean() + 0.01 * result.log_probability.mean()
    gradient = torch.autograd.grad(loss, raw)[0]
    assert gradient[0, 0, 14].item() == 0.0
    assert torch.count_nonzero(gradient[0, -1, :14]).item() == 0
    assert torch.linalg.vector_norm(gradient[0, :-1, :14]).item() > 1.0e-9
    assert torch.linalg.vector_norm(gradient[0, 1:, 14]).item() > 1.0e-9

    nominal = nominal_sbs_parameters(dtype=torch.float64)
    baseline = applied_control_penalties(result, nominal)
    mutated_decisions = result.decision_physical_controls.clone()
    mutated_decisions[:, 0, 14] += 100.0
    mutated_decisions[:, -1, :14] -= 100.0
    mutated = replace(result, decision_physical_controls=mutated_decisions)
    changed = applied_control_penalties(mutated, nominal)
    assert torch.equal(baseline[0], changed[0])
    assert torch.equal(baseline[1], changed[1])


def test_h2_exact_tree_closes_feedback_grape_and_baseline_identities() -> None:
    simulator = _simulator(batch_size=4)
    theta = torch.tensor(
        [0.11, -0.17, 0.09], dtype=torch.float64, requires_grad=True
    )
    result = _four_branch_gradient_tree(simulator, theta)
    probability = result.trajectory_probability
    reward = result.reward
    log_probability = result.log_probability

    assert result.outcomes.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert probability.detach().sum().item() == pytest.approx(1.0, abs=2.0e-12)
    torch.testing.assert_close(
        torch.prod(result.conditional_probabilities, dim=1),
        probability,
        atol=2.0e-14,
        rtol=2.0e-14,
    )

    expected_reward = torch.sum(probability * reward)
    exact_gradient = torch.autograd.grad(
        expected_reward, theta, retain_graph=True
    )[0]
    reward_path_gradient = torch.autograd.grad(
        torch.sum(probability.detach() * reward), theta, retain_graph=True
    )[0]

    def score_gradient(baseline: float) -> torch.Tensor:
        return torch.autograd.grad(
            torch.sum(
                probability.detach()
                * (reward.detach() - baseline)
                * log_probability
            ),
            theta,
            retain_graph=True,
        )[0]

    score_at_low_baseline = score_gradient(-0.23)
    score_at_high_baseline = score_gradient(0.71)
    normalization_score = torch.autograd.grad(
        torch.sum(probability.detach() * log_probability),
        theta,
    )[0]

    torch.testing.assert_close(
        exact_gradient,
        reward_path_gradient + score_at_low_baseline,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    torch.testing.assert_close(
        score_at_low_baseline,
        score_at_high_baseline,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    torch.testing.assert_close(
        normalization_score,
        torch.zeros_like(normalization_score),
        atol=2.0e-12,
        rtol=0.0,
    )
    assert torch.linalg.vector_norm(reward_path_gradient).item() > 1.0e-9
    assert torch.linalg.vector_norm(score_at_low_baseline).item() > 1.0e-9


def test_h2_training_loss_finite_difference_and_d33_score_omission() -> None:
    simulator = _simulator(batch_size=4)
    theta = torch.tensor(
        [0.11, -0.17, 0.09], dtype=torch.float64, requires_grad=True
    )
    result = _four_branch_gradient_tree(simulator, theta)
    probability = result.trajectory_probability
    reward = result.reward
    nominal = nominal_sbs_parameters(dtype=torch.float64)
    residual, slew = applied_control_penalties_per_trajectory(result, nominal)
    baseline = 0.37
    residual_weight = 1.0e-5
    slew_weight = 1.0e-5
    detached_branch_weights = probability.detach()

    (
        implemented_loss,
        _,
        _,
        weighted_residual,
        weighted_slew,
    ) = _feedback_grape_loss(
        result,
        baseline=baseline,
        nominal=nominal,
        residual_l2_weight=residual_weight,
        slew_l2_weight=slew_weight,
        weights=detached_branch_weights,
    )
    torch.testing.assert_close(
        weighted_residual,
        torch.sum(detached_branch_weights * residual),
        atol=1.0e-15,
        rtol=1.0e-14,
    )
    torch.testing.assert_close(
        weighted_slew,
        torch.sum(detached_branch_weights * slew),
        atol=1.0e-15,
        rtol=1.0e-14,
    )

    control_cost = residual_weight * residual + slew_weight * slew
    true_expected_regularized_loss = torch.sum(
        probability * (-reward + control_cost)
    )
    true_gradient = torch.autograd.grad(
        true_expected_regularized_loss, theta, retain_graph=True
    )[0]
    implemented_gradient = torch.autograd.grad(
        implemented_loss, theta, retain_graph=True
    )[0]
    missing_regularizer_score = torch.autograd.grad(
        torch.sum(
            probability.detach()
            * control_cost.detach()
            * result.log_probability
        ),
        theta,
    )[0]

    # D33 is an explicit boundary, not a claimed unbiased regularized loss:
    # the production helper omits this likelihood-ratio contribution.
    torch.testing.assert_close(
        true_gradient,
        implemented_gradient + missing_regularizer_score,
        atol=2.0e-12,
        rtol=2.0e-10,
    )
    assert torch.linalg.vector_norm(missing_regularizer_score).item() > 1.0e-10
    assert not torch.allclose(
        true_gradient, implemented_gradient, atol=1.0e-10, rtol=0.0
    )

    base_probability = probability.detach()
    base_reward = reward.detach()

    def finite_difference_objectives(values: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            perturbed = _four_branch_gradient_tree(simulator, values)
            perturbed_residual, perturbed_slew = (
                applied_control_penalties_per_trajectory(perturbed, nominal)
            )
            perturbed_cost = (
                residual_weight * perturbed_residual
                + slew_weight * perturbed_slew
            )
            true_value = torch.sum(
                perturbed.trajectory_probability
                * (-perturbed.reward + perturbed_cost)
            )
            # ``reward.detach()`` in the production helper is frozen at the
            # differentiation point.  Recomputing detach at theta +/- h would
            # therefore be the wrong finite-difference objective.
            local_surrogate = torch.sum(
                base_probability
                * (
                    -perturbed.reward
                    - (base_reward - baseline) * perturbed.log_probability
                    + perturbed_cost
                )
            )
        return float(true_value), float(local_surrogate)

    step = 3.0e-5
    true_finite_difference: list[float] = []
    implemented_finite_difference: list[float] = []
    base_values = theta.detach()
    for index in range(base_values.numel()):
        upper = base_values.clone()
        lower = base_values.clone()
        upper[index] += step
        lower[index] -= step
        upper_true, upper_implemented = finite_difference_objectives(upper)
        lower_true, lower_implemented = finite_difference_objectives(lower)
        true_finite_difference.append((upper_true - lower_true) / (2.0 * step))
        implemented_finite_difference.append(
            (upper_implemented - lower_implemented) / (2.0 * step)
        )

    np.testing.assert_allclose(
        true_gradient.detach().cpu().numpy(),
        np.asarray(true_finite_difference),
        atol=2.0e-8,
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        implemented_gradient.detach().cpu().numpy(),
        np.asarray(implemented_finite_difference),
        atol=2.0e-8,
        rtol=2.0e-6,
    )
