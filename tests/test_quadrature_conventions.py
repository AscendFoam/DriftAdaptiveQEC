from __future__ import annotations

import json
from math import pi, sqrt

import numpy as np
import pytest

from physics.cross_fidelity_validation import fock_folded_map_response
from physics.finite_energy_gkp import damped_projector_state
from physics.fock_density_model import FiniteCutoffFockModel
from physics.noise_transfer_surrogate import (
    projector_delta_from_squeezing_db,
    squeezing_db_to_peak_variance,
)
from physics.quadrature_conventions import (
    CANONICAL_LOGICAL_CELL_SPACING,
    CANONICAL_STABILIZER_SPACING,
    DECODER_LOGICAL_CELL_SPACING,
    DECODER_STANDARDIZATION_SCALE,
    DISPLACEMENT_LOGICAL_AMPLITUDE,
    PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE,
    SYMPLECTIC_P_LOGICAL_CELL_SPACING,
    SYMPLECTIC_Q_LOGICAL_CELL_SPACING,
    chart,
    chart_transform,
    convert_axis_sigma,
    convert_covariance,
    convert_phase_vector,
    logical_cell_spacing,
    run_quadrature_convention_validation,
    wavefunction_dilation,
    write_quadrature_convention_validation,
)


def test_square_gkp_canonical_and_displacement_constants_are_distinct_semantics() -> None:
    assert CANONICAL_LOGICAL_CELL_SPACING == pytest.approx(sqrt(pi))
    assert CANONICAL_STABILIZER_SPACING == pytest.approx(2.0 * sqrt(pi))
    assert CANONICAL_LOGICAL_CELL_SPACING * CANONICAL_STABILIZER_SPACING == pytest.approx(
        2.0 * pi
    )
    assert DISPLACEMENT_LOGICAL_AMPLITUDE == pytest.approx(sqrt(pi / 2.0))
    assert PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE == pytest.approx(sqrt(2.0 * pi))
    assert 2.0 * DISPLACEMENT_LOGICAL_AMPLITUDE == pytest.approx(
        PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE
    )


def test_equal_sqrt_two_pi_numbers_do_not_share_a_contract() -> None:
    assert DECODER_LOGICAL_CELL_SPACING == pytest.approx(
        PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE
    )
    assert "classical" in chart("decoder_standardized").intended_use
    assert "Weyl" in chart("displacement_amplitude").intended_use


def test_decoder_isotropic_scaling_is_explicitly_non_symplectic() -> None:
    registered = chart("decoder_standardized")
    assert registered.commutator_multiplier == pytest.approx(2.0)
    assert not registered.quantum_symplectic
    with pytest.raises(ValueError, match="quantum symplectic"):
        chart_transform(
            "canonical_fock",
            "decoder_standardized",
            require_quantum_symplectic=True,
        )


def test_displacement_amplitudes_are_not_a_canonical_pair() -> None:
    registered = chart("displacement_amplitude")
    assert registered.commutator_multiplier == pytest.approx(0.5)
    assert not registered.quantum_symplectic


def test_anisotropic_bridge_preserves_commutator_and_cell_area() -> None:
    registered = chart("symplectic_bridge")
    assert registered.commutator_multiplier == pytest.approx(1.0)
    assert registered.quantum_symplectic
    assert np.linalg.det(
        chart_transform(
            "canonical_fock",
            "symplectic_bridge",
            require_quantum_symplectic=True,
        )
    ) == pytest.approx(1.0)
    assert SYMPLECTIC_Q_LOGICAL_CELL_SPACING * SYMPLECTIC_P_LOGICAL_CELL_SPACING == pytest.approx(
        pi
    )


@pytest.mark.parametrize(
    ("chart_name", "q_spacing", "p_spacing"),
    [
        ("canonical_fock", sqrt(pi), sqrt(pi)),
        ("decoder_standardized", sqrt(2.0 * pi), sqrt(2.0 * pi)),
        ("symplectic_bridge", sqrt(2.0 * pi), sqrt(pi / 2.0)),
        ("displacement_amplitude", sqrt(pi / 2.0), sqrt(pi / 2.0)),
    ],
)
def test_registered_logical_cell_spacings(
    chart_name: str, q_spacing: float, p_spacing: float
) -> None:
    assert logical_cell_spacing(chart_name, "q") == pytest.approx(q_spacing)  # type: ignore[arg-type]
    assert logical_cell_spacing(chart_name, "p") == pytest.approx(p_spacing)  # type: ignore[arg-type]


def test_phase_vector_and_covariance_symplectic_roundtrip() -> None:
    vector = np.array([0.7, -1.3])
    covariance = np.array([[0.4, 0.12], [0.12, 0.7]])
    mapped_vector = convert_phase_vector(
        vector,
        "canonical_fock",
        "symplectic_bridge",
        require_quantum_symplectic=True,
    )
    mapped_covariance = convert_covariance(
        covariance,
        "canonical_fock",
        "symplectic_bridge",
        require_quantum_symplectic=True,
    )
    assert convert_phase_vector(
        mapped_vector,
        "symplectic_bridge",
        "canonical_fock",
        require_quantum_symplectic=True,
    ) == pytest.approx(vector, abs=1.0e-14)
    assert convert_covariance(
        mapped_covariance,
        "symplectic_bridge",
        "canonical_fock",
        require_quantum_symplectic=True,
    ) == pytest.approx(covariance, abs=1.0e-14)


@pytest.mark.parametrize(
    "bad_covariance",
    [
        [[1.0, 0.0]],
        [[1.0, 0.2], [0.1, 1.0]],
        [[1.0, 0.0], [0.0, -0.1]],
        [[1.0, float("nan")], [float("nan"), 1.0]],
    ],
)
def test_covariance_conversion_rejects_invalid_inputs(bad_covariance) -> None:
    with pytest.raises(ValueError):
        convert_covariance(
            bad_covariance, "canonical_fock", "symplectic_bridge"
        )


def test_wavefunction_dilation_has_correct_coordinate_and_jacobian() -> None:
    coordinate_factor, amplitude_factor = wavefunction_dilation(
        "decoder_standardized", "canonical_fock", "q"
    )
    assert coordinate_factor == pytest.approx(sqrt(2.0))
    assert amplitude_factor == pytest.approx(2.0**0.25)
    x = np.linspace(-9.0, 9.0, 100_001)
    decoder_wavefunction_at_source_coordinate = (
        pi ** (-0.25)
        / sqrt(DECODER_STANDARDIZATION_SCALE)
        * np.exp(
            -0.5
            * (coordinate_factor * x / DECODER_STANDARDIZATION_SCALE) ** 2
        )
    )
    psi = amplitude_factor * decoder_wavefunction_at_source_coordinate
    assert np.trapz(psi * psi, x) == pytest.approx(1.0, abs=1.0e-10)


@pytest.mark.parametrize("label", ["0", "1", "+", "-"])
def test_damped_projector_operational_state_is_exact_canonical_dilation(label: str) -> None:
    delta = 0.34
    canonical = damped_projector_state(
        label, delta, lattice=CANONICAL_LOGICAL_CELL_SPACING
    )
    decoder = damped_projector_state(label, delta)
    x = np.linspace(-8.0, 8.0, 8193)
    transformed = sqrt(DECODER_STANDARDIZATION_SCALE) * decoder.wavefunction(
        DECODER_STANDARDIZATION_SCALE * x
    )
    assert transformed == pytest.approx(canonical.wavefunction(x), abs=2.0e-14)


def test_damped_projector_peak_variance_and_db_mapping_are_chart_qualified() -> None:
    db = 10.0
    delta = projector_delta_from_squeezing_db(db)
    canonical = damped_projector_state(
        "0", delta, lattice=CANONICAL_LOGICAL_CELL_SPACING
    )
    decoder = damped_projector_state("0", delta)
    assert canonical.amplitude_variance / 2.0 == pytest.approx(
        squeezing_db_to_peak_variance(db, coordinate_chart="canonical_fock")
    )
    assert decoder.amplitude_variance / 2.0 == pytest.approx(
        squeezing_db_to_peak_variance(
            db, coordinate_chart="decoder_standardized"
        )
    )


def test_axis_sigma_conversion_scales_variance_consistently() -> None:
    canonical_sigma = 0.31
    decoder_sigma = convert_axis_sigma(
        canonical_sigma, "canonical_fock", "decoder_standardized", "q"
    )
    assert decoder_sigma == pytest.approx(sqrt(2.0) * canonical_sigma)
    assert convert_axis_sigma(
        decoder_sigma, "decoder_standardized", "canonical_fock", "q"
    ) == pytest.approx(canonical_sigma)


def test_registered_fock_preparation_matches_direct_canonical_source() -> None:
    model = FiniteCutoffFockModel(42)
    delta = 0.34
    registered = model.prepare_damped_projector_gkp("0", delta, grid_points=4097)
    direct = model.project_finite_energy_gkp(
        damped_projector_state(
            "0", delta, lattice=CANONICAL_LOGICAL_CELL_SPACING
        ),
        grid_points=4097,
    )
    registered_coefficients = registered.coefficients / np.linalg.norm(
        registered.coefficients
    )
    direct_coefficients = direct.coefficients / np.linalg.norm(direct.coefficients)
    assert abs(np.vdot(registered_coefficients, direct_coefficients)) ** 2 == pytest.approx(
        1.0, abs=2.0e-13
    )


def test_registered_fock_preparation_rejects_ambiguous_scale() -> None:
    with pytest.raises(ValueError, match="registered"):
        FiniteCutoffFockModel(24).prepare_damped_projector_gkp(
            "0", 0.34, source_coordinate_scale=1.0
        )


@pytest.mark.parametrize("db", [10.0, 12.0])
def test_canonical_fock_qp_fourier_alias_rates_align_at_high_squeezing(db: float) -> None:
    delta = projector_delta_from_squeezing_db(db)
    peak_decoder = squeezing_db_to_peak_variance(
        db, coordinate_chart="decoder_standardized"
    )
    external_decoder = sqrt(0.18**2 + 0.06**2 + peak_decoder)
    q_error = np.mean(
        [
            fock_folded_map_response(
                label,
                delta,
                external_decoder,
                quadrature="q",
                cutoff=48,
                points_per_cell=256,
            ).map_error_probability
            for label in ("0", "1")
        ]
    )
    p_error = np.mean(
        [
            fock_folded_map_response(
                label,
                delta,
                external_decoder,
                quadrature="p",
                cutoff=48,
                points_per_cell=256,
            ).map_error_probability
            for label in ("+", "-")
        ]
    )
    assert abs(p_error - q_error) < 2.0e-6


def test_legacy_ambiguous_fourier_path_remains_negative_provenance() -> None:
    db = 10.0
    delta = projector_delta_from_squeezing_db(db)
    external_decoder = sqrt(
        0.18**2
        + 0.06**2
        + squeezing_db_to_peak_variance(
            db, coordinate_chart="decoder_standardized"
        )
    )
    q = fock_folded_map_response(
        "0", delta, external_decoder, quadrature="q", points_per_cell=256
    )
    legacy_p = np.mean(
        [
            fock_folded_map_response(
                label,
                delta,
                external_decoder,
                quadrature="p",
                points_per_cell=256,
                coordinate_contract="legacy_ambiguous_operational_fourier",
            ).map_error_probability
            for label in ("+", "-")
        ]
    )
    assert legacy_p - q.map_error_probability > 0.4


def test_fock_response_rejects_mixed_coordinate_contracts() -> None:
    with pytest.raises(ValueError, match=r"sqrt\(pi\)"):
        fock_folded_map_response(
            "0",
            0.34,
            0.2,
            domain_spacing=DECODER_LOGICAL_CELL_SPACING,
        )
    with pytest.raises(ValueError, match="source_coordinate_scale"):
        fock_folded_map_response(
            "0", 0.34, 0.2, source_coordinate_scale=1.0
        )


def test_validation_and_json_writer_preserve_all_fail_closed_checks(tmp_path) -> None:
    result = run_quadrature_convention_validation()
    assert result.passed
    assert len(result.checks) == 15
    assert all(result.checks.values())
    output = write_quadrature_convention_validation(result, tmp_path / "contract.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"]
    assert payload["decoder_commutator_multiplier"] == pytest.approx(2.0)
    assert payload["symplectic_bridge_commutator_multiplier"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: chart("unknown"),
        lambda: logical_cell_spacing("canonical_fock", "bad"),
        lambda: convert_axis_sigma(-0.1, "canonical_fock", "decoder_standardized", "q"),
        lambda: convert_phase_vector([1.0], "canonical_fock", "symplectic_bridge"),
    ],
)
def test_chart_api_rejects_invalid_inputs(call) -> None:
    with pytest.raises(ValueError):
        call()
