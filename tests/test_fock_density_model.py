from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from physics.finite_energy_gkp import damped_projector_state
from physics.fock_density_model import (
    FOCK_MODEL_SCOPE,
    FiniteCutoffDensity,
    FiniteCutoffFockModel,
    FockDensityValidationResult,
    run_fock_density_validation,
    write_fock_density_validation,
)


def _minimum_eigenvalue(state: FiniteCutoffDensity) -> float:
    return float(np.min(np.linalg.eigvalsh(state.matrix)))


def _assert_physical(state: FiniteCutoffDensity) -> None:
    assert np.trace(state.matrix) == pytest.approx(1.0, abs=1.0e-11)
    assert np.linalg.norm(state.matrix - state.matrix.conj().T) < 1.0e-11
    assert _minimum_eigenvalue(state) > -1.0e-9


def test_density_from_ket_normalizes_and_freezes_storage() -> None:
    state = FiniteCutoffDensity.from_ket([1.0, 1.0j, 0.0])
    _assert_physical(state)
    assert state.matrix[0, 0] == pytest.approx(0.5)
    assert not state.matrix.flags.writeable
    with pytest.raises(ValueError):
        state.matrix[0, 0] = 0.0


@pytest.mark.parametrize(
    "matrix,cutoff",
    [
        (np.eye(3), 2),
        (np.array([[1.0, 0.2], [0.0, 0.0]]), 2),
        (np.diag([1.1, -0.1]), 2),
        (np.diag([0.6, 0.6]), 2),
    ],
)
def test_density_rejects_shape_nonhermitian_nonpositive_and_nonunit_trace(
    matrix: np.ndarray, cutoff: int
) -> None:
    with pytest.raises(ValueError):
        FiniteCutoffDensity(matrix, cutoff)


def test_model_operators_obey_truncated_number_relation() -> None:
    model = FiniteCutoffFockModel(8)
    assert np.allclose(model.adag, model.a.conj().T)
    assert np.allclose(model.number, np.diag(np.arange(8)))
    commutator = model.a @ model.adag - model.adag @ model.a
    assert np.allclose(np.diag(commutator)[:-1], 1.0)
    assert commutator[-1, -1] == pytest.approx(-7.0)


def test_basis_state_diagnostics_are_exact() -> None:
    model = FiniteCutoffFockModel(10)
    diagnostics = model.diagnostics(model.basis_state(6))
    assert diagnostics.mean_photon_number == pytest.approx(6.0)
    assert diagnostics.purity == pytest.approx(1.0)
    assert diagnostics.minimum_eigenvalue == pytest.approx(0.0)


def test_gkp_projection_has_even_parity_and_explicit_capture() -> None:
    model = FiniteCutoffFockModel(18)
    result = model.prepare_damped_projector_gkp("0", 0.45, grid_points=4097)
    _assert_physical(result.state)
    assert result.captured_probability > 0.999
    assert np.max(np.abs(result.coefficients[1::2])) < 1.0e-10
    assert result.source_model == "damped_projector"
    assert result.logical_state == "0"


def test_gkp_projection_converges_in_grid_and_cutoff() -> None:
    source = damped_projector_state("0", 0.45)
    extent = max(10.0, source.support_radius)
    model18 = FiniteCutoffFockModel(18)
    coarse = model18.project_finite_energy_gkp(source, grid_points=4097, q_extent=extent)
    fine = model18.project_finite_energy_gkp(source, grid_points=8193, q_extent=extent)
    overlap = abs(
        np.vdot(
            coarse.coefficients / np.linalg.norm(coarse.coefficients),
            fine.coefficients / np.linalg.norm(fine.coefficients),
        )
    ) ** 2
    model24 = FiniteCutoffFockModel(24)
    higher = model24.project_finite_energy_gkp(source, grid_points=4097, q_extent=extent)
    assert overlap > 1.0 - 1.0e-12
    assert higher.captured_probability > coarse.captured_probability


def test_projection_rejects_even_or_too_small_grid_and_wrong_source() -> None:
    model = FiniteCutoffFockModel(8)
    source = damped_projector_state("0", 0.5)
    with pytest.raises(ValueError):
        model.project_finite_energy_gkp(source, grid_points=1024)
    with pytest.raises(ValueError):
        model.project_finite_energy_gkp(source, grid_points=513)
    with pytest.raises(TypeError):
        model.project_finite_energy_gkp(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_projection_rejects_invalid_source_coordinate_scale(scale: float) -> None:
    model = FiniteCutoffFockModel(18)
    source = damped_projector_state("0", 0.45)
    with pytest.raises(ValueError):
        model.project_finite_energy_gkp(
            source,
            grid_points=2049,
            source_coordinate_scale=scale,
        )


def test_source_coordinate_scale_applies_normalized_canonical_dilation() -> None:
    model = FiniteCutoffFockModel(40)
    source = damped_projector_state("0", 0.45)
    operational = model.project_finite_energy_gkp(
        source,
        grid_points=8193,
        source_coordinate_scale=1.0,
    )
    canonical = model.project_finite_energy_gkp(
        source,
        grid_points=8193,
        source_coordinate_scale=np.sqrt(2.0),
    )
    x = (model.a + model.adag) / np.sqrt(2.0)
    operational_second_moment = float(
        np.trace(operational.state.matrix @ x @ x).real
    )
    canonical_second_moment = float(np.trace(canonical.state.matrix @ x @ x).real)
    assert canonical.source_coordinate_scale == pytest.approx(np.sqrt(2.0))
    assert canonical.q_extent == pytest.approx(operational.q_extent / np.sqrt(2.0))
    assert canonical_second_moment / operational_second_moment == pytest.approx(
        0.5, abs=0.003
    )
    assert canonical.captured_probability > 0.999


def test_displacement_is_unitary_and_roundtrip_is_exact_at_cutoff() -> None:
    model = FiniteCutoffFockModel(16)
    operator = model.displacement_operator(0.31 - 0.17j)
    assert np.linalg.norm(operator.conj().T @ operator - model.identity) < 1.0e-12
    state = model.basis_state(3)
    restored = model.displace(model.displace(state, 0.31 - 0.17j), -0.31 + 0.17j)
    assert np.linalg.norm(restored.matrix - state.matrix) < 1.0e-12


def test_displaced_vacuum_has_coherent_mean_away_from_cutoff() -> None:
    model = FiniteCutoffFockModel(20)
    alpha = 0.4 + 0.2j
    displaced = model.displace(model.basis_state(0), alpha)
    assert model.diagnostics(displaced).mean_photon_number == pytest.approx(
        abs(alpha) ** 2, abs=1.0e-12
    )


def test_pure_loss_fock_population_is_binomial() -> None:
    model = FiniteCutoffFockModel(10)
    eta = 0.6
    output = model.pure_loss(model.basis_state(4), eta)
    expected = np.array(
        [
            0.4**4,
            4 * eta * 0.4**3,
            6 * eta**2 * 0.4**2,
            4 * eta**3 * 0.4,
            eta**4,
        ]
    )
    assert np.allclose(np.diag(output.matrix)[:5].real, expected, atol=1.0e-13)
    _assert_physical(output)


def test_pure_loss_endpoints_are_identity_and_vacuum() -> None:
    model = FiniteCutoffFockModel(8)
    state = model.displace(model.basis_state(0), 0.5)
    assert np.allclose(model.pure_loss(state, 1.0).matrix, state.matrix)
    vacuum = model.pure_loss(state, 0.0)
    assert np.allclose(vacuum.matrix, model.basis_state(0).matrix)


def test_thermal_zero_time_is_identity_and_vacuum_mean_is_analytic() -> None:
    model = FiniteCutoffFockModel(18)
    vacuum = model.basis_state(0)
    assert np.array_equal(
        model.thermal_excitation(vacuum, rate_time=0.0, bath_occupation=2.0).matrix,
        vacuum.matrix,
    )
    duration = 0.25
    occupation = 0.4
    output = model.thermal_excitation(
        vacuum, rate_time=duration, bath_occupation=occupation
    )
    expected = occupation * (1.0 - np.exp(-duration))
    assert model.diagnostics(output).mean_photon_number == pytest.approx(expected, abs=1.0e-10)
    _assert_physical(output)


def test_zero_temperature_thermal_lindblad_matches_loss_transmissivity() -> None:
    model = FiniteCutoffFockModel(12)
    state = model.displace(model.basis_state(1), 0.16 - 0.07j)
    duration = 0.19
    lindblad = model.thermal_excitation(
        state, rate_time=duration, bath_occupation=0.0
    )
    kraus = model.pure_loss(state, np.exp(-duration))
    assert np.linalg.norm(lindblad.matrix - kraus.matrix, ord="fro") < 1.0e-11


def test_phase_diffusion_damps_only_coherences_by_exact_factor() -> None:
    model = FiniteCutoffFockModel(8)
    ket = np.zeros(8, dtype=np.complex128)
    ket[1] = ket[5] = 1.0
    state = FiniteCutoffDensity.from_ket(ket)
    variance = 0.07
    output = model.phase_diffusion(state, variance)
    assert np.diag(output.matrix).real == pytest.approx(np.diag(state.matrix).real)
    assert output.matrix[1, 5] == pytest.approx(
        state.matrix[1, 5] * np.exp(-0.5 * variance * 16.0)
    )


def test_kerr_preserves_all_fock_populations_and_low_levels() -> None:
    model = FiniteCutoffFockModel(12)
    state = model.displace(model.basis_state(0), 0.7)
    output = model.kerr(state, 0.15)
    assert np.diag(output.matrix) == pytest.approx(np.diag(state.matrix))
    assert output.matrix[0, 1] == pytest.approx(state.matrix[0, 1])
    assert not np.allclose(output.matrix, state.matrix)


def test_modular_effects_are_positive_and_complete() -> None:
    model = FiniteCutoffFockModel(14)
    plus, minus = model.modular_effects(0.2 + 0.3j, contrast=0.83, phase=0.2)
    assert np.allclose(plus + minus, model.identity)
    assert np.min(np.linalg.eigvalsh(plus)) > -1.0e-12
    assert np.min(np.linalg.eigvalsh(minus)) > -1.0e-12


def test_conditional_measurement_probabilities_and_backaction_are_physical() -> None:
    model = FiniteCutoffFockModel(14)
    state = model.displace(model.basis_state(0), 0.4j)
    plus = model.modular_measurement(state, 0.3, "plus", contrast=0.9)
    minus = model.modular_measurement(state, 0.3, "minus", contrast=0.9)
    assert plus.probability + minus.probability == pytest.approx(1.0, abs=1.0e-12)
    _assert_physical(plus.state)
    _assert_physical(minus.state)
    nonselective = model.nonselective_modular_measurement(state, 0.3, contrast=0.9)
    mixture = plus.probability * plus.state.matrix + minus.probability * minus.state.matrix
    assert np.allclose(nonselective.matrix, mixture, atol=1.0e-12)


def test_zero_contrast_measurement_is_uninformative_and_nondisturbing() -> None:
    model = FiniteCutoffFockModel(10)
    state = model.displace(model.basis_state(0), 0.31)
    plus = model.modular_measurement(state, 0.4j, "plus", contrast=0.0)
    minus = model.modular_measurement(state, 0.4j, "minus", contrast=0.0)
    assert plus.probability == pytest.approx(0.5)
    assert minus.probability == pytest.approx(0.5)
    assert np.allclose(plus.state.matrix, state.matrix)
    assert np.allclose(minus.state.matrix, state.matrix)


def test_high_fock_proxy_is_cptp_and_handles_top_boundary() -> None:
    model = FiniteCutoffFockModel(7)
    probability = 0.3
    shifted = model.high_fock_leakage_proxy(model.basis_state(2), probability)
    assert np.diag(shifted.matrix).real == pytest.approx([0, 0, 0.7, 0.3, 0, 0, 0])
    top = model.high_fock_leakage_proxy(model.basis_state(6), probability)
    assert np.allclose(top.matrix, model.basis_state(6).matrix)
    _assert_physical(shifted)
    _assert_physical(top)


def test_composed_noise_stack_remains_physical() -> None:
    model = FiniteCutoffFockModel(18)
    state = model.prepare_damped_projector_gkp("0", 0.5, grid_points=4097).state
    state = model.displace(state, 0.12 - 0.04j)
    state = model.pure_loss(state, 0.96)
    state = model.thermal_excitation(state, rate_time=0.03, bath_occupation=0.08)
    state = model.phase_diffusion(state, 0.002)
    state = model.kerr(state, 0.01)
    state = model.nonselective_modular_measurement(state, 0.15j, contrast=0.92)
    state = model.high_fock_leakage_proxy(state, 0.01)
    _assert_physical(state)


@pytest.mark.parametrize(
    "operation",
    [
        lambda model, state: model.pure_loss(state, -0.1),
        lambda model, state: model.pure_loss(state, 1.1),
        lambda model, state: model.thermal_excitation(
            state, rate_time=-0.1, bath_occupation=0.0
        ),
        lambda model, state: model.phase_diffusion(state, -0.1),
        lambda model, state: model.modular_effects(0.1, contrast=1.1),
        lambda model, state: model.high_fock_leakage_proxy(state, -0.1),
    ],
)
def test_channels_reject_unphysical_parameters(operation) -> None:
    model = FiniteCutoffFockModel(8)
    with pytest.raises(ValueError):
        operation(model, model.basis_state(0))


def test_model_rejects_cutoff_mismatch() -> None:
    model = FiniteCutoffFockModel(8)
    with pytest.raises(ValueError):
        model.displace(FiniteCutoffFockModel(9).basis_state(0), 0.1)


def test_production_validation_covers_convergence_analytics_and_positivity() -> None:
    result = run_fock_density_validation(
        cutoffs=(18, 24, 30, 36), projector_delta=0.45, grid_points=4097
    )
    assert result.passed
    assert result.captured_probabilities[-1] > 0.9999
    assert result.adjacent_embedded_fidelities[-1] > 0.99999
    assert len(result.checks) == 10
    assert "transmon levels" in result.scope


def test_validation_writer_emits_machine_readable_pass_and_scope() -> None:
    result = FockDensityValidationResult(
        cutoffs=(8, 10, 12),
        captured_probabilities=(0.8, 0.9, 0.95),
        adjacent_embedded_fidelities=(0.9, 0.95),
        displacement_roundtrip_error=0.0,
        loss_mean_error=0.0,
        thermal_vacuum_mean_error=0.0,
        phase_coherence_error=0.0,
        kerr_population_error=0.0,
        measurement_probability_error=0.0,
        minimum_output_eigenvalue=0.0,
        checks={"gate": True},
    )
    output_path = Path.cwd() / ".pytest_fock_density_validation.json"
    try:
        output = write_fock_density_validation(result, output_path)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["passed"] is True
        assert payload["scope"] == FOCK_MODEL_SCOPE
    finally:
        output_path.unlink(missing_ok=True)


def test_scope_fail_closes_device_and_transmon_claims() -> None:
    lowered = FOCK_MODEL_SCOPE.lower()
    assert "no transmon" in lowered
    assert "device calibration" in lowered
    assert "no" in lowered and "hardware claim" in lowered
