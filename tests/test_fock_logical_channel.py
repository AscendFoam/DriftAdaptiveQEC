from __future__ import annotations

import math

import numpy as np
import pytest

from physics.fock_logical_channel import (
    AXIS_STATE_PAIRS,
    FockLogicalChannelConfig,
    FockLogicalChannelSimulator,
    PAULIS,
    STATE_LABELS,
    finite_horizon_pauli_lifetime,
    logical_eigenstate_density,
    reconstruct_code_subchannel,
)


def _mapped_outputs(channel) -> dict[str, np.ndarray]:
    return {
        label: np.asarray(channel(logical_eigenstate_density(label)), dtype=np.complex128)
        for label in STATE_LABELS
    }


def test_six_state_identity_reconstructs_exact_ptm_and_physicality() -> None:
    result = reconstruct_code_subchannel(_mapped_outputs(lambda rho: rho))
    np.testing.assert_allclose(result.ptm, np.eye(4), atol=1.0e-15)
    assert result.minimum_choi_eigenvalue >= -1.0e-15
    np.testing.assert_allclose(result.tni_effect_eigenvalues, [1.0, 1.0], atol=1.0e-15)
    assert result.mean_leakage == pytest.approx(0.0, abs=1.0e-15)
    assert result.off_diagonal_pauli_norm == pytest.approx(0.0, abs=1.0e-15)
    assert result.passed_physicality


def test_uniform_erasure_keeps_missing_trace_instead_of_postselecting() -> None:
    survival = 0.73
    result = reconstruct_code_subchannel(
        _mapped_outputs(lambda rho: survival * rho)
    )
    np.testing.assert_allclose(result.ptm, survival * np.eye(4), atol=1.0e-15)
    assert result.mean_leakage == pytest.approx(1.0 - survival)
    np.testing.assert_allclose(
        result.tni_effect_eigenvalues, [survival, survival], atol=1.0e-15
    )
    assert result.passed_physicality


def test_coherent_z_rotation_is_visible_as_non_pauli_off_diagonal_ptm() -> None:
    theta = 0.31
    unitary = np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])
    result = reconstruct_code_subchannel(
        _mapped_outputs(lambda rho: unitary @ rho @ unitary.conj().T)
    )
    expected = np.eye(4)
    expected[1, 1] = math.cos(theta)
    expected[1, 2] = -math.sin(theta)
    expected[2, 1] = math.sin(theta)
    expected[2, 2] = math.cos(theta)
    np.testing.assert_allclose(result.ptm, expected, atol=2.0e-15)
    assert result.off_diagonal_pauli_norm > 0.4
    assert result.coherent_rotation_norm > 0.4
    assert result.passed_physicality


def test_amplitude_damping_exposes_nonunital_and_state_dependent_survival_terms() -> None:
    gamma = 0.2
    k0 = np.diag([1.0, math.sqrt(1.0 - gamma)])
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]])
    result = reconstruct_code_subchannel(
        _mapped_outputs(lambda rho: k0 @ rho @ k0.T + k1 @ rho @ k1.T)
    )
    assert result.nonunital_code_flow_norm == pytest.approx(gamma, abs=1.0e-14)
    assert result.state_dependent_survival_norm == pytest.approx(0.0, abs=1.0e-14)
    assert result.passed_physicality


def test_pair_sum_inconsistency_and_schema_errors_fail_closed() -> None:
    outputs = _mapped_outputs(lambda rho: rho)
    outputs["x_plus"] = 0.9 * outputs["x_plus"]
    result = reconstruct_code_subchannel(outputs)
    assert result.pair_sum_linearity_residual > 0.05
    assert not result.passed_physicality
    with pytest.raises(ValueError, match="six-state outputs mismatch"):
        reconstruct_code_subchannel({key: outputs[key] for key in STATE_LABELS[:-1]})
    bad = _mapped_outputs(lambda rho: rho)
    bad["z_plus"] = np.array([[1.0, 0.2], [0.0, 0.0]])
    with pytest.raises(ValueError, match="Hermitian"):
        reconstruct_code_subchannel(bad)


def test_finite_horizon_lifetime_reports_area_crossing_and_revival_without_clipping() -> None:
    cycles = np.arange(5)
    times = 10.0 * cycles
    signal = np.exp(-cycles / 2.0)
    metric = finite_horizon_pauli_lifetime(cycles, times, signal)
    assert metric["e_fold_status"] == "observed"
    assert metric["e_fold_crossing_cycles"] == pytest.approx(2.0, abs=0.08)
    assert metric["e_fold_crossing_us"] == pytest.approx(20.0, abs=0.8)
    assert metric["truncated_signed_area_us"] == pytest.approx(
        10.0 * metric["truncated_signed_area_cycles"]
    )
    revival = finite_horizon_pauli_lifetime(
        cycles, times, [1.0, 0.8, 0.85, 0.5, 0.55]
    )
    assert revival["revival_step_count"] == 2
    assert revival["e_fold_status"] == "right_censored"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mode": "invalid"}, "mode"),
        ({"mode": "qec_on", "full_cycles": 0}, "full_cycles"),
        ({"mode": "qec_off", "cutoff": 3}, "cutoff"),
        ({"mode": "qec_off", "cycle_duration_us": 9.0}, "10 us"),
        ({"mode": "qec_on", "real_dtype": "float32"}, "float64"),
        ({"mode": "qec_on", "scope": "promoted"}, "scope"),
    ],
)
def test_config_failure_paths(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FockLogicalChannelConfig(**kwargs)


def test_matched_idle_identity_limit_preserves_all_six_states_and_ptm() -> None:
    result = FockLogicalChannelSimulator(
        FockLogicalChannelConfig(
            mode="qec_off",
            full_cycles=2,
            cutoff=6,
            cavity_lifetime_us=1.0e15,
            ancilla_t1_us=1.0e15,
            ancilla_t2_us=1.0e15,
        )
    ).run()
    assert result.projected_outputs.shape == (3, 6, 2, 2)
    np.testing.assert_allclose(result.ptm, np.broadcast_to(np.eye(4), (3, 4, 4)), atol=2.0e-12)
    np.testing.assert_allclose(result.survival, 1.0, atol=2.0e-12)
    assert all(point.passed_physicality for point in result.tomography)
    assert result.event_accounting["active_gate_applications"] == 0
    assert result.event_accounting["discarded_trajectories"] == 0


def test_finite_loss_idle_lane_is_cptni_and_retains_leakage() -> None:
    result = FockLogicalChannelSimulator(
        FockLogicalChannelConfig(
            mode="qec_off",
            full_cycles=3,
            cutoff=6,
            cavity_lifetime_us=80.0,
        )
    ).run()
    assert np.max(result.leakage[-1]) > 1.0e-3
    assert all(point.passed_physicality for point in result.tomography)
    assert result.maximum_physical_trace_error < 2.0e-12
    assert result.minimum_physical_eigenvalue > -2.0e-10
    for index, axis in enumerate(("X", "Y", "Z"), start=1):
        np.testing.assert_allclose(
            result.ptm[:, index, index],
            [result.tomography[cycle].ptm[index, index] for cycle in range(4)],
        )
        assert result.pauli_lifetimes[axis]["horizon_cycles"] == 3.0


def test_active_lane_runs_six_state_channel_without_truth_or_postselection_shortcuts() -> None:
    result = FockLogicalChannelSimulator(
        FockLogicalChannelConfig(mode="qec_on", full_cycles=2, cutoff=6)
    ).run()
    np.testing.assert_allclose(result.ptm[0], np.eye(4), atol=2.0e-12)
    assert all(point.passed_physicality for point in result.tomography)
    assert result.event_accounting["measurement_events"] == 4
    assert result.event_accounting["reset_events"] == 4
    assert result.event_accounting["active_gate_applications"] == 36
    assert result.event_accounting["postselected_trajectories"] == 0
    assert result.event_accounting["discarded_trajectories"] == 0
    assert result.maximum_physical_trace_error < 2.0e-12
    assert np.all(np.isfinite(result.conditional_bloch))


def test_axis_pairs_are_complete_and_disjoint() -> None:
    flattened = [label for pair in AXIS_STATE_PAIRS.values() for label in pair]
    assert sorted(flattened) == sorted(STATE_LABELS)
    assert len(set(flattened)) == 6
    np.testing.assert_allclose(PAULIS[0], np.eye(2))
