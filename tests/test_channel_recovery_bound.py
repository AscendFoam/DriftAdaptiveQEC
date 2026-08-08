from __future__ import annotations

from math import exp

import numpy as np
import pytest

from physics.channel_recovery_bound import (
    encoded_channel_kraus,
    evaluate_encoded_channel_recovery,
    finite_cutoff_gkp_isometry,
    near_optimal_fidelity_from_qec,
    partial_trace_recovery_output,
    petz_recovery_diagnostics,
    pure_loss_kraus,
    qec_matrix,
    recovery_objective_matrix,
    solve_optimal_recovery_sdp,
)


def _amplitude_damping(gamma: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex128),
        np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128),
    )


def test_identity_channel_saturates_petz_theorem_and_sdp() -> None:
    identity = np.eye(2, dtype=np.complex128)
    result = evaluate_encoded_channel_recovery(
        identity,
        (identity,),
        solve_sdp=True,
    )
    assert result.petz["petz_fidelity"] == pytest.approx(1.0, abs=2.0e-12)
    assert result.petz["theorem_optimal_upper"] == pytest.approx(1.0, abs=2.0e-12)
    assert result.sdp is not None
    assert result.sdp["intersection_certified_lower"] == pytest.approx(1.0, abs=2.0e-8)
    assert result.sdp["intersection_certified_upper"] == pytest.approx(1.0, abs=2.0e-8)


def test_amplitude_damping_petz_and_repaired_sdp_double_bound() -> None:
    identity = np.eye(2, dtype=np.complex128)
    encoded = encoded_channel_kraus(identity, _amplitude_damping(0.2))
    near = near_optimal_fidelity_from_qec(
        qec_matrix(encoded),
        logical_dimension=2,
        kraus_count=2,
    )
    direct = petz_recovery_diagnostics(encoded)
    certificate = solve_optimal_recovery_sdp(encoded)
    assert direct["direct_petz_fidelity"] == pytest.approx(
        near["petz_fidelity"], abs=2.0e-9
    )
    assert near["petz_fidelity"] <= certificate["repaired_primal_fidelity_lower"] + 2.0e-8
    assert certificate["repaired_primal_fidelity_lower"] <= certificate[
        "repaired_dual_fidelity_upper"
    ] + 2.0e-8
    assert certificate["repaired_dual_fidelity_upper"] <= near[
        "theorem_optimal_upper"
    ] + 2.0e-7
    assert certificate["repaired_primal_tp_residual"] <= 2.0e-8
    assert certificate["repaired_primal_minimum_eigenvalue"] >= -2.0e-8
    assert certificate["repaired_dual_minimum_slack_eigenvalue"] >= 0.9e-10


def test_qec_petz_fidelity_is_invariant_to_unitary_kraus_mixing() -> None:
    operators = encoded_channel_kraus(
        np.eye(2, dtype=np.complex128), _amplitude_damping(0.17)
    )
    unitary = np.array(
        [[1.0, 1.0j], [1.0j, 1.0]], dtype=np.complex128
    ) / np.sqrt(2.0)
    mixed = tuple(
        sum(
            (unitary[row, column] * operators[column] for column in range(2)),
            start=np.zeros((2, 2), dtype=np.complex128),
        )
        for row in range(2)
    )
    original = near_optimal_fidelity_from_qec(
        qec_matrix(operators), logical_dimension=2, kraus_count=2
    )
    rotated = near_optimal_fidelity_from_qec(
        qec_matrix(mixed), logical_dimension=2, kraus_count=2
    )
    assert rotated["petz_fidelity"] == pytest.approx(
        original["petz_fidelity"], abs=2.0e-9
    )


def test_recovery_objective_uses_output_fast_choi_convention() -> None:
    identity = np.eye(2, dtype=np.complex128)
    objective = recovery_objective_matrix((identity,))
    vector = identity.reshape(-1, order="F")
    choi = np.outer(vector, vector.conj())
    assert np.trace(objective @ choi).real == pytest.approx(1.0, abs=2.0e-12)
    assert partial_trace_recovery_output(
        choi, logical_dimension=2, physical_dimension=2
    ) == pytest.approx(np.eye(2), abs=2.0e-12)


def test_finite_cutoff_pure_loss_is_exactly_tp_and_has_expected_no_loss_entry() -> None:
    operators = pure_loss_kraus(
        8,
        duration_us=10.0,
        cavity_lifetime_us=245.0,
    )
    effect = sum(
        (operator.conj().T @ operator for operator in operators),
        start=np.zeros((8, 8), dtype=np.complex128),
    )
    eta = exp(-10.0 / 245.0)
    assert effect == pytest.approx(np.eye(8), abs=2.0e-12)
    assert operators[0][7, 7] == pytest.approx(eta ** 3.5, abs=2.0e-12)


def test_registered_gkp_isometry_matches_live_trajectory_basis() -> None:
    pytest.importorskip("torch")
    from physics.differentiable_sbs_trajectory import (
        DifferentiableSBSConfig,
        DifferentiableSBSTrajectorySimulator,
    )

    expected = finite_cutoff_gkp_isometry(4, 0.34)
    engine = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(
            cutoff=4,
            full_cycles=1,
            batch_size=1,
            projector_delta=0.34,
            device="cpu",
            real_dtype="float64",
        )
    )
    actual = engine.logical_isometry.detach().cpu().numpy()
    assert actual == pytest.approx(expected, abs=2.0e-12)


def test_cutoff4_real_gkp_petz_matches_direct_and_sdp() -> None:
    isometry = finite_cutoff_gkp_isometry(4, 0.34)
    result = evaluate_encoded_channel_recovery(
        isometry,
        pure_loss_kraus(4, duration_us=10.0, cavity_lifetime_us=245.0),
        solve_sdp=True,
    )
    assert result.petz["petz_fidelity"] == pytest.approx(
        result.petz_recovery["direct_petz_fidelity"], abs=2.0e-9
    )
    assert result.sdp is not None
    assert result.petz["petz_fidelity"] <= result.sdp[
        "intersection_certified_lower"
    ] + 2.0e-8
    assert result.sdp["intersection_width"] <= 2.0e-7


@pytest.mark.parametrize(
    "call",
    [
        lambda: finite_cutoff_gkp_isometry(3, 0.34),
        lambda: finite_cutoff_gkp_isometry(4, 0.34, grid_points=8192),
        lambda: pure_loss_kraus(4, duration_us=0.0, cavity_lifetime_us=245.0),
        lambda: encoded_channel_kraus(
            np.eye(2, dtype=np.complex128),
            (0.9 * np.eye(2, dtype=np.complex128),),
        ),
        lambda: solve_optimal_recovery_sdp(
            (np.eye(2, dtype=np.complex128),), solver="SCS"
        ),
    ],
)
def test_invalid_or_non_tp_inputs_fail_closed(call) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()
