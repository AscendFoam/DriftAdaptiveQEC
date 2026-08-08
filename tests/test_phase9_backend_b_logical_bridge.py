from __future__ import annotations

import ast
from dataclasses import replace
from math import cosh, exp, factorial, log, pi, sqrt, tanh
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_hermite

from physics.fock_sbs_cycle import SBSFockCycleConfig, SBSFockOneRoundSimulator
from physics.phase9_backend_b import (
    BACKEND_B_ID,
    BACKEND_B_LIKELIHOOD_ID,
    BACKEND_B_RNG_ID,
    BACKEND_B_SOLVER_ID,
    BackendBConfig,
    Phase9BackendBSimulator,
)
from physics.phase9_backend_b_logical_bridge import (
    MEHLER_BRIDGE_DERIVATION,
    MEHLER_BRIDGE_ID,
    MEHLER_BRIDGE_SCOPE,
    MehlerLogicalBridgeConfig,
    Phase9BackendBMehlerBridgeSimulator,
    analytic_mehler_isometry,
    gaussian_component_fock_coefficient,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "physics/phase9_backend_b_logical_bridge.py"


def _numerical_coefficient(order: int, center: float, variance: float) -> float:
    normalization = pi ** (-0.25) / sqrt(
        (2.0**order) * factorial(order)
    )

    def integrand(q: float) -> float:
        return (
            normalization
            * eval_hermite(order, q)
            * exp(-0.5 * q * q)
            * exp(-0.5 * (q - center) ** 2 / variance)
        )

    value, error = quad(
        integrand,
        -12.0,
        12.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=300,
        points=[center],
    )
    assert error < 2.0e-8
    return float(value)


@pytest.mark.parametrize("variance", [0.08, 0.2, 0.7, 1.4])
@pytest.mark.parametrize("center", [-2.1, -0.3, 0.0, 1.7])
@pytest.mark.parametrize("order", list(range(8)))
def test_closed_form_coefficients_match_independent_quadrature(
    order: int,
    center: float,
    variance: float,
) -> None:
    analytic = gaussian_component_fock_coefficient(
        order,
        center,
        variance,
    )
    numerical = _numerical_coefficient(order, center, variance)
    assert analytic == pytest.approx(numerical, abs=2.0e-11)


@pytest.mark.parametrize("cutoff", [8, 12, 16, 24])
def test_analytic_mehler_isometry_is_orthonormal(cutoff: int) -> None:
    isometry = analytic_mehler_isometry(
        cutoff,
        MehlerLogicalBridgeConfig(),
    )
    assert isometry.shape == (cutoff, 2)
    assert not isometry.flags.writeable
    assert np.linalg.norm(
        isometry.conj().T @ isometry - np.eye(2),
        ord="fro",
    ) < 1.0e-11


@pytest.mark.parametrize("cutoff", [8, 12])
def test_bridge_matches_registered_damped_projector_convention(
    cutoff: int,
) -> None:
    # The comparison is allowed only in tests/qualification.  The bridge
    # implementation itself has no backend-A or repository-projector import.
    reference = SBSFockOneRoundSimulator(
        SBSFockCycleConfig(
            cutoff=cutoff,
            projector_delta=0.34,
            grid_points=8193,
        )
    ).code_basis.isometry
    independent = analytic_mehler_isometry(
        cutoff,
        MehlerLogicalBridgeConfig(
            projector_delta=0.34,
            tail_tolerance=1.0e-12,
        ),
    )
    singular_values = np.linalg.svd(
        reference.conj().T @ independent,
        compute_uv=False,
    )
    projector_frobenius = np.linalg.norm(
        reference @ reference.conj().T
        - independent @ independent.conj().T,
        ord="fro",
    )
    assert float(np.min(singular_values)) >= 0.95
    assert float(projector_frobenius) <= 0.30
    assert float(projector_frobenius) < 1.0e-9


def test_bridge_repairs_detected_truncated_exponential_ordering_error() -> None:
    cutoff = 12
    reference = SBSFockOneRoundSimulator(
        SBSFockCycleConfig(
            cutoff=cutoff,
            projector_delta=0.34,
            grid_points=8193,
        )
    ).code_basis.isometry
    released_native = Phase9BackendBSimulator(
        BackendBConfig(cutoff=cutoff)
    )._comb_isometry()
    amended = analytic_mehler_isometry(
        cutoff,
        MehlerLogicalBridgeConfig(),
    )
    reference_projector = reference @ reference.conj().T
    native_error = np.linalg.norm(
        reference_projector
        - released_native @ released_native.conj().T,
        ord="fro",
    )
    amended_error = np.linalg.norm(
        reference_projector - amended @ amended.conj().T,
        ord="fro",
    )
    assert native_error > 0.30
    assert amended_error < 1.0e-9


def test_bridge_changes_only_logical_isometry_identity() -> None:
    base = BackendBConfig(cutoff=8)
    native = Phase9BackendBSimulator(base)
    bridge = Phase9BackendBMehlerBridgeSimulator(
        base,
        bridge_config=MehlerLogicalBridgeConfig(),
    )
    assert native.config == bridge.config
    diagnostics = bridge.logical_bridge_diagnostics()
    assert diagnostics["bridge_id"] == MEHLER_BRIDGE_ID
    assert diagnostics["backend_id"] == BACKEND_B_ID
    assert diagnostics["transition_solver_id"] == BACKEND_B_SOLVER_ID
    assert diagnostics["rng_id"] == BACKEND_B_RNG_ID
    assert diagnostics["likelihood_id"] == BACKEND_B_LIKELIHOOD_ID
    assert diagnostics["isometry_orthonormality_frobenius"] < 1.0e-11
    for parity, support in enumerate(diagnostics["supports"]):
        assert support == sorted(support)
        assert support == [-value for value in reversed(support)]
        assert all(value % 2 == parity for value in support)
    captured = diagnostics["captured_probabilities"]
    assert len(captured) == 2
    assert all(0.0 < value <= 1.0 for value in captured)
    assert len(diagnostics["raw_gram_eigenvalues"]) == 2


@pytest.mark.parametrize("delta", [0.22, 0.34, 0.52, 0.8])
def test_bridge_uses_mehler_identities_for_multiple_deltas(delta: float) -> None:
    simulator = Phase9BackendBMehlerBridgeSimulator(
        BackendBConfig(cutoff=12),
        bridge_config=MehlerLogicalBridgeConfig(
            projector_delta=delta,
            tail_tolerance=1.0e-10,
        ),
    )
    diagnostics = simulator.logical_bridge_diagnostics()
    variance = tanh(delta * delta)
    assert diagnostics["amplitude_variance"] == pytest.approx(
        variance,
        abs=1.0e-15,
    )
    assert diagnostics["center_contraction"] == pytest.approx(
        1.0 / cosh(delta * delta),
        abs=1.0e-15,
    )
    assert -0.5 * log(variance) > 0.0


def test_tail_tolerance_expands_support_without_changing_physical_formula() -> None:
    loose = Phase9BackendBMehlerBridgeSimulator(
        BackendBConfig(cutoff=12),
        bridge_config=MehlerLogicalBridgeConfig(tail_tolerance=1.0e-6),
    ).logical_bridge_diagnostics()
    strict = Phase9BackendBMehlerBridgeSimulator(
        BackendBConfig(cutoff=12),
        bridge_config=MehlerLogicalBridgeConfig(tail_tolerance=1.0e-14),
    ).logical_bridge_diagnostics()
    assert all(
        strict_count > loose_count
        for strict_count, loose_count in zip(
            strict["component_counts"],
            loose["component_counts"],
        )
    )
    loose_iso = analytic_mehler_isometry(
        12,
        MehlerLogicalBridgeConfig(tail_tolerance=1.0e-6),
    )
    strict_iso = analytic_mehler_isometry(
        12,
        MehlerLogicalBridgeConfig(tail_tolerance=1.0e-14),
    )
    assert np.linalg.norm(
        loose_iso @ loose_iso.conj().T
        - strict_iso @ strict_iso.conj().T,
        ord="fro",
    ) < 1.0e-6


def test_bridge_configuration_fails_closed() -> None:
    config = MehlerLogicalBridgeConfig()
    for field, value in (
        ("projector_delta", 0.0),
        ("projector_delta", float("nan")),
        ("tail_tolerance", 1.0),
        ("tail_tolerance", True),
        ("bridge_id", "forged"),
        ("derivation", "shortcut"),
        ("scope", "performance claim"),
    ):
        with pytest.raises((TypeError, ValueError)):
            replace(config, **{field: value})
    with pytest.raises(TypeError):
        analytic_mehler_isometry(8, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        gaussian_component_fock_coefficient(0, 0.0, 0.0)
    with pytest.raises(ValueError):
        gaussian_component_fock_coefficient(0, float("inf"), 0.2)


def test_bridge_source_is_independent_of_backend_a_and_projector_runtime() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = (
        "phase9_backend_a",
        "finite_energy_gkp",
        "fock_density_model",
        "fock_sbs_cycle",
    )
    assert all(
        token not in module
        for module in imported
        for token in forbidden
    )


def test_bridge_constants_are_exactly_frozen() -> None:
    assert MEHLER_BRIDGE_ID.endswith("-V1")
    assert "closed-form integral" in MEHLER_BRIDGE_DERIVATION
    assert "no lifetime" in MEHLER_BRIDGE_SCOPE
