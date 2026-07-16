"""Independent tests for the T2.3.3 four-lane cross-fidelity validator."""

from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pytest

import physics
from physics.cross_fidelity_validation import (
    CROSS_FIDELITY_SCOPE,
    DEFAULT_DB_GRID,
    CrossFidelityConfig,
    evaluate_cross_fidelity_point,
    fock_folded_map_response,
    independent_axis_pauli_metrics,
    run_cross_fidelity_validation,
    write_cross_fidelity_validation,
)
from physics.finite_energy_gkp import damped_projector_state
from physics.logical_channel import (
    finite_energy_parity_response_1d,
    parity_confusion_from_response,
)
from physics.noise_transfer_surrogate import (
    projector_delta_from_squeezing_db,
    squeezing_db_to_peak_variance,
)


@pytest.fixture(scope="module")
def production_result():
    return run_cross_fidelity_validation(
        CrossFidelityConfig(effective_samples=100_000, seed=2026071434)
    )


def test_default_config_freezes_common_contract() -> None:
    config = CrossFidelityConfig()
    assert config.squeezing_db == DEFAULT_DB_GRID
    assert config.channel_sigma == pytest.approx(0.18)
    assert config.measurement_sigma == pytest.approx(0.06)
    assert config.fock_cutoff == 48
    assert config.effective_samples == 200_000
    assert config.scope == CROSS_FIDELITY_SCOPE


def test_point_api_supports_disjoint_holdout_without_relaxing_calibration_grid() -> None:
    config = CrossFidelityConfig(effective_samples=100_000, seed=2026071607)
    point = evaluate_cross_fidelity_point(10.25, config)
    assert point.squeezing_db == pytest.approx(10.25)
    assert point.region == "high_squeezing"
    assert point.effective.samples == 100_000
    assert config.squeezing_db == DEFAULT_DB_GRID
    with pytest.raises(ValueError):
        evaluate_cross_fidelity_point(0.0, config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"squeezing_db": (3.0, 10.0, 12.0)},
        {"channel_sigma": 0.0},
        {"measurement_sigma": -0.1},
        {"fock_cutoff": 23},
        {"fock_protocol_cutoff": 49},
        {"fock_points_per_cell": 255},
        {"projection_grid_points": 2048},
        {"syndrome_points": 511},
        {"effective_samples": 99_999},
        {"seed": -1},
        {"scope": "all models are equivalent"},
    ],
)
def test_config_rejects_demo_or_scope_promoting_values(kwargs) -> None:
    with pytest.raises(ValueError):
        CrossFidelityConfig(**kwargs)


@pytest.mark.parametrize(
    ("q", "p", "identity", "favg"),
    [
        (0.0, 0.0, 1.0, 1.0),
        (0.1, 0.2, 0.72, (2.0 * 0.72 + 1.0) / 3.0),
        (0.5, 0.5, 0.25, 0.5),
        (1.0, 0.0, 0.0, 1.0 / 3.0),
    ],
)
def test_independent_axis_pauli_metrics_have_analytic_identity(
    q: float, p: float, identity: float, favg: float
) -> None:
    metrics = independent_axis_pauli_metrics(q, p, construction="unit test")
    assert metrics.correct_coset_occupancy == pytest.approx(identity)
    assert metrics.logical_error_rate == pytest.approx(1.0 - identity)
    assert metrics.average_fidelity == pytest.approx(favg)


@pytest.mark.parametrize("q,p", [(-0.1, 0.0), (0.0, 1.1), (np.nan, 0.2)])
def test_independent_axis_pauli_metrics_reject_invalid_rates(q: float, p: float) -> None:
    with pytest.raises(ValueError):
        independent_axis_pauli_metrics(q, p, construction="invalid")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"logical_label": "+", "quadrature": "q"},
        {"logical_label": "0", "quadrature": "p"},
        {"logical_label": "bad", "quadrature": "q"},
        {"logical_label": "0", "quadrature": "q", "cutoff": 10},
        {"logical_label": "0", "quadrature": "q", "points_per_cell": 127},
        {"logical_label": "0", "quadrature": "q", "displacement_sigma": -0.1},
    ],
)
def test_fock_response_rejects_invalid_metric_contract(kwargs) -> None:
    defaults = {
        "logical_label": "0",
        "projector_delta": projector_delta_from_squeezing_db(10.0),
        "displacement_sigma": 0.2,
        "quadrature": "q",
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        fock_folded_map_response(**defaults)


@pytest.mark.parametrize("db", [3.0, 10.0])
def test_fock_q_response_matches_independent_direct_state_density(db: float) -> None:
    delta = projector_delta_from_squeezing_db(db)
    peak_variance = squeezing_db_to_peak_variance(
        db, coordinate_chart="decoder_standardized"
    )
    external = sqrt(0.18**2 + 0.06**2 + peak_variance)
    fock_errors = []
    direct_errors = []
    for label in ("0", "1"):
        fock = fock_folded_map_response(
            label,
            delta,
            external,
            cutoff=48,
            points_per_cell=512,
            projection_grid_points=4097,
        )
        direct = finite_energy_parity_response_1d(
            damped_projector_state(label, delta),
            displacement_sigma=external,
            points=2048,
        )
        fock_errors.append(fock.map_error_probability)
        direct_errors.append(parity_confusion_from_response(direct).error_probability)
        assert fock.reconstructed_mass == pytest.approx(1.0, abs=2.0e-12)
    tolerance = 1.0e-5 if db == 3.0 else 3.0e-5
    assert np.mean(fock_errors) == pytest.approx(
        np.mean(direct_errors), abs=tolerance
    )


def test_fock_twelve_db_tail_sensitive_alias_error_improves_with_cutoff() -> None:
    delta = projector_delta_from_squeezing_db(12.0)
    external = sqrt(
        0.18**2
        + 0.06**2
        + squeezing_db_to_peak_variance(
            12.0, coordinate_chart="decoder_standardized"
        )
    )
    errors = []
    captures = []
    for cutoff in (24, 36, 48):
        response = fock_folded_map_response(
            "0",
            delta,
            external,
            cutoff=cutoff,
            points_per_cell=256,
            projection_grid_points=4097,
        )
        errors.append(response.map_error_probability)
        captures.append(response.captured_probability)
    assert errors[0] > errors[1] > errors[2]
    assert captures[0] < captures[1] < captures[2]


def test_canonical_fourier_alignment_and_legacy_failure_are_both_exposed() -> None:
    delta = projector_delta_from_squeezing_db(10.0)
    external = sqrt(
        0.18**2
        + 0.06**2
        + squeezing_db_to_peak_variance(
            10.0, coordinate_chart="decoder_standardized"
        )
    )
    q = fock_folded_map_response(
        "0", delta, external, quadrature="q", points_per_cell=256
    )
    p = fock_folded_map_response(
        "+", delta, external, quadrature="p", points_per_cell=256
    )
    legacy = np.mean(
        [
            fock_folded_map_response(
                label,
                delta,
                external,
                quadrature="p",
                points_per_cell=256,
                coordinate_contract="legacy_ambiguous_operational_fourier",
            ).map_error_probability
            for label in ("+", "-")
        ]
    )
    assert abs(p.map_error_probability - q.map_error_probability) < 2.0e-6
    assert legacy - q.map_error_probability > 0.4


def test_production_result_passes_all_registered_gates(production_result) -> None:
    assert production_result.passed
    assert len(production_result.checks) == 15
    assert all(production_result.checks.values())
    assert production_result.scope == CROSS_FIDELITY_SCOPE


def test_production_regions_are_explicit(production_result) -> None:
    assert [item.region for item in production_result.points] == [
        "low_squeezing_clipping",
        "low_squeezing_clipping",
        "transition",
        "high_squeezing",
        "high_squeezing",
    ]


@pytest.mark.parametrize(
    "lane_getter",
    [
        lambda point: point.fock.two_axis_pauli_metrics,
        lambda point: point.effective.pauli,
        lambda point: point.noise_transfer.pauli,
        lambda point: point.syndrome.square_symmetry_projection,
    ],
)
def test_common_lane_LER_occupancy_fidelity_directions(
    production_result, lane_getter
) -> None:
    metrics = [lane_getter(point) for point in production_result.points]
    assert all(
        metrics[index].logical_error_rate > metrics[index + 1].logical_error_rate
        for index in range(len(metrics) - 1)
    )
    assert all(
        metrics[index].correct_coset_occupancy
        < metrics[index + 1].correct_coset_occupancy
        for index in range(len(metrics) - 1)
    )
    assert all(
        metrics[index].average_fidelity < metrics[index + 1].average_fidelity
        for index in range(len(metrics) - 1)
    )


def test_native_occupancy_metrics_are_directional_not_numerically_conflated(
    production_result,
) -> None:
    fock = [item.fock.protocol.average_code_survival for item in production_result.points]
    effective = [item.effective.central_domain_occupancy for item in production_result.points]
    noise = [item.noise_transfer.central_domain_occupancy for item in production_result.points]
    assert all(a < b for a, b in zip(fock, fock[1:]))
    assert all(a < b for a, b in zip(effective, effective[1:]))
    assert all(a < b for a, b in zip(noise, noise[1:]))
    assert not np.allclose(fock, effective, rtol=0.0, atol=1.0e-4)
    assert abs(fock[-1] - effective[-1]) > 5.0e-3


def test_high_and_low_regions_have_opposite_agreement_semantics(production_result) -> None:
    assert production_result.maximum_high_squeezing_noise_syndrome_q_ler_gap < 1.0e-4
    assert production_result.maximum_high_squeezing_fock_syndrome_q_ler_gap < 5.0e-4
    assert production_result.maximum_high_squeezing_canonical_fock_qp_ler_gap < 1.0e-5
    assert production_result.minimum_high_squeezing_legacy_p_minus_q_ler_gap > 0.4
    assert production_result.low_squeezing_noise_syndrome_q_ler_gap > 1.0e-2
    assert production_result.points[0].noise_transfer.validity == "clipping_dominated"
    assert all(
        point.noise_transfer.validity == "localized"
        for point in production_result.points[-2:]
    )


def test_effective_monte_carlo_matches_high_squeezing_exact_proxy(production_result) -> None:
    assert production_result.maximum_high_squeezing_effective_noise_z_score < 5.0
    assert all(point.effective.samples == 100_000 for point in production_result.points)


def test_cutoff_sweep_is_retained_in_machine_result(production_result) -> None:
    sweep = production_result.twelve_db_fock_cutoff_sweep
    assert [item.cutoff for item in sweep] == [24, 30, 36, 42, 48]
    assert all(a.q_axis_ler > b.q_axis_ler for a, b in zip(sweep, sweep[1:]))
    assert all(
        a.minimum_captured_probability < b.minimum_captured_probability
        for a, b in zip(sweep, sweep[1:])
    )


def test_error_attribution_table_covers_all_noncomparable_boundaries(
    production_result,
) -> None:
    identifiers = {item.attribution_id for item in production_result.attributions}
    assert identifiers == {
        "XA-LOW-CLIPPING",
        "XA-HIGH-CUTOFF",
        "XA-P-COORDINATE",
        "XA-OCCUPANCY-SEMANTICS",
    }
    p_entry = next(
        item
        for item in production_result.attributions
        if item.attribution_id == "XA-P-COORDINATE"
    )
    assert "not promoted to a coherent joint-axis correlation claim" in p_entry.reporting_consequence


def test_writer_preserves_checks_and_negative_evidence(production_result, tmp_path) -> None:
    output = write_cross_fidelity_validation(
        production_result, tmp_path / "cross_fidelity.json"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert all(payload["checks"].values())
    assert len(payload["attributions"]) == 4
    assert abs(payload["points"][-1]["fock"]["p_minus_q_ler_gap"]) < 1.0e-5
    assert payload["points"][-1]["fock"]["legacy_p_minus_q_ler_gap"] > 0.4


def test_public_lazy_exports_preserve_fail_closed_scope() -> None:
    assert physics.CrossFidelityConfig is CrossFidelityConfig
    assert physics.CROSS_FIDELITY_SCOPE == CROSS_FIDELITY_SCOPE
