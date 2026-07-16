from __future__ import annotations

import importlib.util
from math import pi, sqrt

import numpy as np
import pytest

if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch is optional in the minimal recovery interpreter", allow_module_level=True)

from physics.differentiable_sbs_trajectory import (
    DIFFERENTIABLE_SBS_SCOPE,
    PARAMETER_NAMES,
    POROTTI_S1_PROFILE_ID,
    DifferentiableSBSConfig,
    DifferentiableSBSTimingProfile,
    DifferentiableSBSTrajectorySimulator,
    nominal_sbs_parameters,
    run_differentiable_sbs_validation,
)

import torch


@pytest.fixture(scope="module")
def simulator() -> DifferentiableSBSTrajectorySimulator:
    return DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(cutoff=6, batch_size=4, grid_points=2049)
    )


def test_timing_profile_is_exact_and_fail_closed() -> None:
    profile = DifferentiableSBSTimingProfile()
    assert profile.profile_id == POROTTI_S1_PROFILE_ID
    assert [item.duration_ns for item in profile.phases] == [100, 500, 700, 300, 100, 2300, 1000]
    assert profile.half_cycle_duration_ns == 5000
    assert profile.full_cycle_duration_ns == 10000
    assert "not_target_hardware" in profile.evidence_scope


@pytest.mark.parametrize(
    "change, error",
    [
        ({"cutoff": 3}, ValueError),
        ({"cutoff": 49}, ValueError),
        ({"full_cycles": 0}, ValueError),
        ({"full_cycles": 11}, ValueError),
        ({"batch_size": 0}, ValueError),
        ({"batch_size": 4097}, ValueError),
        ({"grid_points": 1024}, ValueError),
        ({"cavity_lifetime_us": 0.0}, ValueError),
        ({"ancilla_t1_us": 20.0, "ancilla_t2_us": 41.0}, ValueError),
        ({"device": "tpu"}, ValueError),
        ({"real_dtype": "float16"}, ValueError),
        ({"probability_floor": 0.0}, ValueError),
        ({"protocol_id": "wrong"}, ValueError),
        ({"scope": "demo"}, ValueError),
    ],
)
def test_config_rejects_invalid_or_scope_erasing_inputs(change: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        DifferentiableSBSConfig(**change)


def test_parameter_registry_matches_paper_table_s4() -> None:
    parameters = nominal_sbs_parameters().cpu().numpy()
    assert len(PARAMETER_NAMES) == 15
    assert parameters.shape == (15,)
    np.testing.assert_allclose(
        parameters,
        [
            pi / 2,
            pi / 2,
            0.0,
            0.2,
            0.0,
            -pi / 2,
            sqrt(2 * pi),
            0.0,
            0.0,
            pi / 2,
            0.0,
            0.2,
            pi / 2,
            -pi / 2,
            pi / 2,
        ],
        atol=1.0e-14,
    )


def test_bounded_control_map_is_differentiable_and_obeys_paper_bounds(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    raw = torch.full(
        (4, 2, 15), 5.0, dtype=torch.float64, requires_grad=True
    )
    physical = simulator.bounded_physical_controls(raw)
    nominal = nominal_sbs_parameters(dtype=torch.float64)
    correction = physical - nominal[None, None, :]
    assert torch.all(correction[..., :-1] < 2.0)
    assert torch.all(correction[..., :-1] > 1.999)
    assert torch.all(correction[..., -1] < 1.0)
    assert torch.all(correction[..., -1] > 0.999)
    physical.sum().backward()
    assert raw.grad is not None
    assert torch.all(torch.isfinite(raw.grad))


def test_broadcast_control_shape_and_invalid_control_inputs(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    two_dimensional = torch.zeros((2, 15), dtype=torch.float64)
    expanded = simulator.bounded_physical_controls(two_dimensional)
    assert expanded.shape == (4, 2, 15)
    assert torch.allclose(expanded[0], expanded[3])
    with pytest.raises(TypeError):
        simulator.bounded_physical_controls(np.zeros((4, 2, 15)))
    with pytest.raises(ValueError):
        simulator.bounded_physical_controls(torch.zeros((4, 15)))
    with pytest.raises(ValueError):
        simulator.bounded_physical_controls(
            torch.full((4, 2, 15), float("nan"))
        )


def test_all_gate_families_are_unitary_at_finite_cutoff(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    residuals = simulator.gate_unitarity_residuals()
    assert set(residuals) == {"R1", "ECD1", "fixed_D", "VR"}
    assert max(residuals.values()) < 1.0e-10


def test_gate_definitions_match_independent_matrix_exponential(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    scipy = pytest.importorskip("scipy.linalg")
    beta = torch.full((4,), 0.17 + 0.09j, dtype=torch.complex128)
    displacement = simulator._batch_displacement(beta / 2)[0].detach().cpu().numpy()
    a = simulator.a.detach().cpu().numpy()
    expected = scipy.expm((0.17 + 0.09j) / 2 * a.conj().T - (0.17 - 0.09j) / 2 * a)
    np.testing.assert_allclose(displacement, expected, atol=2.0e-12, rtol=2.0e-12)
    ecd = simulator._batch_ecd(beta)[0].detach().cpu().numpy().reshape(6, 2, 6, 2)
    np.testing.assert_allclose(ecd[:, 0, :, 1], expected, atol=2.0e-12)
    np.testing.assert_allclose(ecd[:, 1, :, 0], expected.conj().T, atol=2.0e-12)
    np.testing.assert_allclose(ecd[:, 0, :, 0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(ecd[:, 1, :, 1], 0.0, atol=1.0e-14)


def test_each_analytic_idle_channel_is_cptp(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    residuals = simulator.idle_completeness_residuals()
    assert len(residuals) == 21
    assert max(residuals.values()) < 1.0e-10


def test_initial_joint_state_is_normalized_positive_and_ancilla_ground(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    state = simulator._initial_joint_density()
    assert state.shape == (4, 12, 12)
    np.testing.assert_allclose(
        torch.diagonal(state, dim1=-2, dim2=-1).sum(-1).cpu(), 1.0, atol=1.0e-12
    )
    assert torch.min(torch.linalg.eigvalsh(state)).item() > -1.0e-12
    blocks = state.reshape(4, 6, 2, 6, 2)
    assert torch.linalg.matrix_norm(blocks[:, :, 1, :, 1]).max().item() < 1.0e-14


def test_all_two_measurement_branches_sum_to_one_and_remain_physical(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    branches = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    result = simulator.run(forced_outcomes=branches)
    assert result.outcomes.tolist() == branches.tolist()
    assert result.conditional_probabilities.shape == (4, 2)
    assert result.trajectory_probability.detach().sum().item() == pytest.approx(
        1.0, abs=2.0e-10
    )
    assert torch.allclose(
        result.trajectory_probability, torch.exp(result.log_probability), atol=1.0e-12
    )
    assert result.maximum_trace_error < 1.0e-10
    assert result.maximum_hermiticity_error < 1.0e-10
    assert result.minimum_final_eigenvalue > -1.0e-10
    assert torch.all((result.reward >= 0.0) & (result.reward <= 1.0))


def test_random_sampling_is_seeded_and_exactly_replayable(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    random_a = simulator.run(seed=771)
    random_b = simulator.run(seed=771)
    assert torch.equal(random_a.outcomes, random_b.outcomes)
    assert torch.allclose(random_a.reward, random_b.reward, atol=1.0e-13)
    replay = simulator.run(forced_outcomes=random_a.outcomes, seed=999)
    assert torch.equal(random_a.outcomes, replay.outcomes)
    assert torch.allclose(
        random_a.trajectory_probability, replay.trajectory_probability, atol=1.0e-13
    )
    assert torch.allclose(random_a.final_joint_density, replay.final_joint_density, atol=1.0e-13)


def test_sampling_frequency_matches_known_ancilla_probability() -> None:
    batch = 3000
    model = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(cutoff=4, batch_size=batch, grid_points=1025)
    )
    cavity = model.target_cavity_density[None].expand(batch, -1, -1)
    ancilla = torch.tensor([[0.73, 0.0], [0.0, 0.27]], dtype=torch.complex128)
    state = torch.einsum("bij,kl->bikjl", cavity, ancilla).reshape(batch, 8, 8)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(221)
    _, outcomes, probability = model._measure_and_reset(state, None, generator)
    g_rate = torch.mean((outcomes == 0).double()).item()
    assert g_rate == pytest.approx(0.73, abs=0.025)
    expected = 0.27 + 0.46 * (outcomes == 0).double()
    assert torch.allclose(probability, expected, atol=1.0e-12)


def test_reward_and_log_probability_paths_both_connect_to_controls(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    raw = torch.linspace(-0.05, 0.05, 4 * 2 * 15, dtype=torch.float64).reshape(4, 2, 15)
    raw.requires_grad_(True)
    result = simulator.run(raw, forced_outcomes=[[0, 0], [0, 1], [1, 0], [1, 1]])
    reward_gradient = torch.autograd.grad(result.reward.mean(), raw, retain_graph=True)[0]
    probability_gradient = torch.autograd.grad(result.log_probability.mean(), raw)[0]
    assert torch.all(torch.isfinite(reward_gradient))
    assert torch.all(torch.isfinite(probability_gradient))
    assert torch.linalg.vector_norm(reward_gradient).item() > 1.0e-8
    assert torch.linalg.vector_norm(probability_gradient).item() > 1.0e-8


def test_controls_change_trajectory_probability_and_reward(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    forced = [[0, 0], [0, 1], [1, 0], [1, 1]]
    baseline = simulator.run(forced_outcomes=forced)
    perturbed_raw = torch.zeros((4, 2, 15), dtype=torch.float64)
    perturbed_raw[..., 2] = 0.12
    perturbed_raw[..., 7] = -0.08
    perturbed = simulator.run(perturbed_raw, forced_outcomes=forced)
    assert not torch.allclose(
        baseline.trajectory_probability, perturbed.trajectory_probability, atol=1.0e-8
    )
    assert not torch.allclose(baseline.reward, perturbed.reward, atol=1.0e-8)


def test_history_conditioned_policy_is_causal_and_differentiable(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    bias = torch.zeros(15, dtype=torch.float64, requires_grad=True)
    direction = torch.linspace(-0.1, 0.1, 15, dtype=torch.float64)
    seen_histories: list[torch.Tensor] = []

    def policy(history: torch.Tensor, half_index: int) -> torch.Tensor:
        seen_histories.append(history.detach().clone())
        if half_index == 0:
            latest = torch.zeros((4, 1), dtype=torch.float64)
        else:
            latest = 2.0 * history[:, -1:].double() - 1.0
        return bias[None, :] + latest * direction[None, :]

    branches = [[0, 0], [0, 1], [1, 0], [1, 1]]
    result = simulator.run(control_policy=policy, forced_outcomes=branches)
    assert seen_histories[0].shape == (4, 0)
    assert seen_histories[1].tolist() == [[0], [0], [1], [1]]
    assert not torch.allclose(
        result.physical_controls[0, 1], result.physical_controls[2, 1]
    )
    assert result.trajectory_probability.detach().sum().item() == pytest.approx(
        1.0, abs=2.0e-10
    )
    (result.reward.mean() + 0.01 * result.log_probability.mean()).backward()
    assert bias.grad is not None
    assert torch.all(torch.isfinite(bias.grad))
    assert torch.linalg.vector_norm(bias.grad).item() > 1.0e-8
    assert result.resource_profile.control_source == "history_conditioned_policy"


def test_policy_and_raw_controls_are_mutually_exclusive(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    with pytest.raises(ValueError):
        simulator.run(torch.zeros((4, 2, 15)), control_policy=lambda history, index: torch.zeros(15))
    with pytest.raises(TypeError):
        simulator.run(control_policy=object())
    with pytest.raises(TypeError):
        simulator.run(control_policy=lambda history, index: np.zeros(15))
    with pytest.raises(ValueError):
        simulator.run(control_policy=lambda history, index: torch.zeros(14))


def test_two_full_cycles_preserve_density_and_report_physical_time() -> None:
    model = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(
            cutoff=4, full_cycles=2, batch_size=1, grid_points=1025
        )
    )
    result = model.run(forced_outcomes=[[0, 1, 0, 1]])
    assert result.outcomes.shape == (1, 4)
    assert result.resource_profile.simulated_physical_time_ns == 20000
    assert result.resource_profile.half_cycles == 4
    assert result.maximum_trace_error < 1.0e-10
    assert result.minimum_final_eigenvalue > -1.0e-10


def test_resource_profile_reports_model_work_not_hardware_measurement(
    simulator: DifferentiableSBSTrajectorySimulator,
) -> None:
    result = simulator.run(forced_outcomes=[[0, 0], [0, 1], [1, 0], [1, 1]])
    resource = result.resource_profile
    assert resource.joint_dimension == 12
    assert resource.trainable_controls == 4 * 2 * 15
    assert resource.control_source == "open_loop_raw_corrections"
    assert resource.matrix_exponentials == 4 * 2 * 4
    assert resource.unitary_applications == 4 * 2 * 9
    assert resource.idle_windows == 4 * 2 * 7
    assert resource.cptp_channel_applications == 4 * 2 * 21
    assert resource.state_tensor_bytes == 4 * 12 * 12 * 16
    assert resource.target_hardware_measured is False
    assert result.scope == DIFFERENTIABLE_SBS_SCOPE


@pytest.mark.parametrize(
    "forced",
    [
        [[0, 0]],
        [[0, 0], [0, 1], [1, 0], [2, 1]],
        [[0], [1], [0], [1]],
    ],
)
def test_forced_outcomes_fail_closed(
    simulator: DifferentiableSBSTrajectorySimulator, forced: list[list[int]]
) -> None:
    with pytest.raises(ValueError):
        simulator.run(forced_outcomes=forced)


def test_seed_must_be_an_integer(simulator: DifferentiableSBSTrajectorySimulator) -> None:
    with pytest.raises(TypeError):
        simulator.run(seed=1.2)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        simulator.run(seed=True)  # type: ignore[arg-type]


def test_validation_payload_has_artifact_backed_non_demo_gates(tmp_path) -> None:
    output = tmp_path / "validation.json"
    payload = run_differentiable_sbs_validation(device="cpu", cutoff=6, output=output)
    assert payload["status"] == "PASS"
    assert len(payload["checks"]) == 17
    assert all(payload["checks"].values())
    assert payload["metrics"]["branch_probability_sum"] == pytest.approx(1.0, abs=2.0e-9)
    assert payload["metrics"]["gradient_norm"] > 1.0e-8
    assert payload["resource_profile"]["target_hardware_measured"] is False
    assert "reserved for T2.3.5" in " ".join(payload["forbidden_claims"])
    assert output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_smoke_preserves_gradient_and_density() -> None:
    model = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(
            cutoff=4,
            batch_size=1,
            grid_points=1025,
            device="cuda",
        )
    )
    raw = torch.zeros((1, 2, 15), dtype=torch.float64, device="cuda", requires_grad=True)
    result = model.run(raw, forced_outcomes=[[0, 1]])
    result.reward.mean().backward()
    assert raw.grad is not None and torch.all(torch.isfinite(raw.grad))
    assert result.maximum_trace_error < 1.0e-10
    assert result.resource_profile.cuda_peak_allocated_bytes is not None
