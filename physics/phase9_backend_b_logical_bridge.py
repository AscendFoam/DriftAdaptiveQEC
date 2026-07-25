"""Independent finite-energy logical bridge for Phase-9 backend B.

The released T9.2.3 backend-B transition solver is intentionally immutable.
Its development logical basis used matrix exponentials *after* truncating the
oscillator Hilbert space.  That construction is useful as an independent
smoke evaluator, but it is not the same numerical operation as projecting an
infinite-dimensional finite-energy GKP wavefunction into a finite Fock basis.

T9.2.4 compares two transition solvers at a shared physical convention.  This
module therefore replaces only backend B's logical isometry.  It derives every
Fock coefficient from the closed-form Hermite--Gaussian integral for the
Mehler-damped comb.  It does not import backend A, ``finite_energy_gkp``,
``fock_density_model`` or ``fock_sbs_cycle`` and does not alter backend B's
Hamiltonian, channels, likelihood, sampler or addressed RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cosh, exp, factorial, isfinite, log, pi, sqrt, tanh
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .phase9_backend_b import (
    BACKEND_B_ID,
    BACKEND_B_LIKELIHOOD_ID,
    BACKEND_B_RNG_ID,
    BACKEND_B_SOLVER_ID,
    BackendBConfig,
    Phase9BackendBSimulator,
)


ComplexMatrix = NDArray[np.complex128]

MEHLER_BRIDGE_ID = "PHASE9-BACKEND-B-ANALYTIC-MEHLER-FOCK-BRIDGE-V1"
MEHLER_BRIDGE_DERIVATION = (
    "closed-form integral of normalized Fock Hermite functions against each "
    "Mehler-contracted Gaussian comb component; project infinite-dimensional "
    "wavefunction first, then truncate"
)
MEHLER_BRIDGE_SCOPE = (
    "T9.2.4 synthetic dual-backend logical-coordinate bridge only; preserves "
    "released backend-B transition/IQ/RNG solver and carries no lifetime, "
    "break-even, hardware, official-Puviani, external-SOTA or rank claim"
)


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be real")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real") from exc
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must lie in [{minimum},{maximum}]")
    return number


def _readonly(value: np.ndarray) -> ComplexMatrix:
    result = np.array(value, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MehlerLogicalBridgeConfig:
    """Frozen physical convention for the T9.2.4 backend-B logical bridge."""

    projector_delta: float = 0.34
    tail_tolerance: float = 1.0e-12
    bridge_id: str = MEHLER_BRIDGE_ID
    derivation: str = MEHLER_BRIDGE_DERIVATION
    scope: str = MEHLER_BRIDGE_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projector_delta",
            _finite_positive(self.projector_delta, "projector_delta"),
        )
        tolerance = _finite_positive(self.tail_tolerance, "tail_tolerance")
        if tolerance >= 1.0:
            raise ValueError("tail_tolerance must lie strictly below one")
        object.__setattr__(self, "tail_tolerance", tolerance)
        if self.bridge_id != MEHLER_BRIDGE_ID:
            raise ValueError("bridge_id is immutable")
        if self.derivation != MEHLER_BRIDGE_DERIVATION:
            raise ValueError("derivation is immutable")
        if self.scope != MEHLER_BRIDGE_SCOPE:
            raise ValueError("scope is immutable")

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "projector_delta": self.projector_delta,
            "tail_tolerance": self.tail_tolerance,
            "bridge_id": self.bridge_id,
            "derivation": self.derivation,
            "scope": self.scope,
        }


def gaussian_component_fock_coefficient(
    n: int,
    center: float,
    amplitude_variance: float,
) -> float:
    r"""Return ``<n| exp(-(q-center)^2/(2 variance))>``.

    The Gaussian component is deliberately left unnormalised because all
    components are combined and the complete logical column is normalised
    afterwards.  The expression follows from the physicists' Hermite
    generating function

    ``exp(2 q t - t^2) = sum_n H_n(q) t^n / n!``.

    This route never constructs a truncated displacement or squeezing
    exponential, which is the exact numerical-ordering issue the bridge fixes.
    """

    order = _integer(n, "n", 0, 128)
    q0 = float(center)
    variance = _finite_positive(amplitude_variance, "amplitude_variance")
    if not isfinite(q0):
        raise ValueError("center must be finite")

    linear = 2.0 * q0 / (variance + 1.0)
    quadratic = (variance - 1.0) / (variance + 1.0)
    if order == 0:
        polynomial = 1.0
    elif order == 1:
        polynomial = linear
    else:
        previous_previous = 1.0
        previous = linear
        for current_order in range(1, order):
            current = (
                linear * previous
                + 2.0
                * quadratic
                * current_order
                * previous_previous
            )
            previous_previous, previous = previous, current
        polynomial = previous
    gaussian_integral_prefactor = sqrt(
        2.0 * pi * variance / (variance + 1.0)
    )
    result = (
        pi ** (-0.25)
        / sqrt((2.0**order) * factorial(order))
        * exp(-q0 * q0 / (2.0 * (variance + 1.0)))
        * gaussian_integral_prefactor
        * polynomial
    )
    if not isfinite(result):
        raise RuntimeError("analytic Hermite-Gaussian coefficient is non-finite")
    return float(result)


def _symmetric_parity_indices(
    amplitude_variance: float,
    tail_tolerance: float,
    parity: int,
) -> tuple[int, ...]:
    maximum = int(
        ceil(
            sqrt(
                2.0
                * log(1.0 / tail_tolerance)
                / (pi * amplitude_variance)
            )
        )
    ) + 2
    if maximum > 255:
        raise ValueError(
            "tail_tolerance/projector_delta require more than 511 peaks"
        )
    return tuple(
        index
        for index in range(-maximum, maximum + 1)
        if index % 2 == parity
    )


def _raw_mehler_columns(
    cutoff: int,
    config: MehlerLogicalBridgeConfig,
) -> tuple[ComplexMatrix, dict[str, object]]:
    """Return raw projected comb columns and truncation diagnostics."""

    dimension = _integer(cutoff, "cutoff", 8, 128)
    if not isinstance(config, MehlerLogicalBridgeConfig):
        raise TypeError("config must be MehlerLogicalBridgeConfig")

    epsilon = config.projector_delta**2
    amplitude_variance = tanh(epsilon)
    center_contraction = 1.0 / cosh(epsilon)
    raw_columns: list[np.ndarray] = []
    captured_probabilities: list[float] = []
    supports: list[tuple[int, ...]] = []
    infinite_norms: list[float] = []
    for logical_bit in (0, 1):
        coefficients = np.zeros(dimension, dtype=np.float64)
        indices = _symmetric_parity_indices(
            amplitude_variance,
            config.tail_tolerance,
            logical_bit,
        )
        supports.append(indices)
        centers = np.array(
            [
                center_contraction * index * sqrt(pi)
                for index in indices
            ],
            dtype=np.float64,
        )
        ideal_centers = np.array(
            [index * sqrt(pi) for index in indices],
            dtype=np.float64,
        )
        weights = np.exp(
            -0.5 * amplitude_variance * ideal_centers**2
        )
        for ideal_center, contracted_center, envelope in zip(
            ideal_centers,
            centers,
            weights,
        ):
            coefficients += envelope * np.array(
                [
                    gaussian_component_fock_coefficient(
                        order,
                        contracted_center,
                        amplitude_variance,
                    )
                    for order in range(dimension)
                ],
                dtype=np.float64,
            )
        difference = centers[:, np.newaxis] - centers[np.newaxis, :]
        component_overlap = sqrt(pi * amplitude_variance) * np.exp(
            -difference**2 / (4.0 * amplitude_variance)
        )
        infinite_norm_squared = float(
            weights @ component_overlap @ weights
        )
        captured_norm_squared = float(np.vdot(coefficients, coefficients).real)
        if (
            not isfinite(infinite_norm_squared)
            or infinite_norm_squared <= 1.0e-15
            or not isfinite(captured_norm_squared)
            or captured_norm_squared <= 1.0e-15
        ):
            raise RuntimeError("analytic Mehler comb has zero or non-finite norm")
        captured = captured_norm_squared / infinite_norm_squared
        if not 0.0 < captured <= 1.0 + 1.0e-9:
            raise RuntimeError("analytic Mehler captured probability is invalid")
        captured_probabilities.append(min(captured, 1.0))
        infinite_norms.append(sqrt(infinite_norm_squared))
        raw_columns.append(coefficients / sqrt(captured_norm_squared))

    raw = np.column_stack(raw_columns).astype(np.complex128)
    gram = raw.conj().T @ raw
    diagnostics: dict[str, object] = {
        "amplitude_variance": amplitude_variance,
        "center_contraction": center_contraction,
        "supports": [list(values) for values in supports],
        "component_counts": [len(values) for values in supports],
        "captured_probabilities": captured_probabilities,
        "infinite_comb_norms": infinite_norms,
        "raw_overlap_real": float(gram[0, 1].real),
        "raw_overlap_imag": float(gram[0, 1].imag),
        "raw_gram_eigenvalues": [
            float(value) for value in np.linalg.eigvalsh(gram)
        ],
    }
    return _readonly(raw), diagnostics


def analytic_mehler_isometry(
    cutoff: int,
    config: MehlerLogicalBridgeConfig,
) -> ComplexMatrix:
    """Construct the orthonormal finite-Fock logical isometry independently."""

    raw, _ = _raw_mehler_columns(cutoff, config)
    gram = raw.conj().T @ raw
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if float(np.min(eigenvalues)) <= 1.0e-10:
        raise RuntimeError("analytic Mehler logical comb is singular")
    inverse_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))
    ) @ eigenvectors.conj().T
    isometry = raw @ inverse_root
    if np.linalg.norm(
        isometry.conj().T @ isometry - np.eye(2),
        ord="fro",
    ) > 1.0e-10:
        raise RuntimeError("analytic Mehler isometry orthonormality failure")
    return _readonly(isometry)


class Phase9BackendBMehlerBridgeSimulator(Phase9BackendBSimulator):
    """Released backend-B solver with the sealed analytic logical bridge."""

    def __init__(
        self,
        config: BackendBConfig,
        *,
        bridge_config: MehlerLogicalBridgeConfig,
    ) -> None:
        if not isinstance(bridge_config, MehlerLogicalBridgeConfig):
            raise TypeError(
                "bridge_config must be MehlerLogicalBridgeConfig"
            )
        self.bridge_config = bridge_config
        super().__init__(config)

    def _comb_isometry(self) -> ComplexMatrix:
        if self._logical_isometry is None:
            self._logical_isometry = analytic_mehler_isometry(
                self.cutoff,
                self.bridge_config,
            )
        return self._logical_isometry

    def logical_bridge_diagnostics(self) -> dict[str, object]:
        isometry = self._comb_isometry()
        _, basis_diagnostics = _raw_mehler_columns(
            self.cutoff,
            self.bridge_config,
        )
        return {
            "bridge_id": self.bridge_config.bridge_id,
            "backend_id": BACKEND_B_ID,
            "projector_delta": self.bridge_config.projector_delta,
            "tail_tolerance": self.bridge_config.tail_tolerance,
            **basis_diagnostics,
            "isometry_orthonormality_frobenius": float(
                np.linalg.norm(
                    isometry.conj().T @ isometry - np.eye(2),
                    ord="fro",
                )
            ),
            "transition_solver_id": BACKEND_B_SOLVER_ID,
            "rng_id": BACKEND_B_RNG_ID,
            "likelihood_id": BACKEND_B_LIKELIHOOD_ID,
            "scope": self.bridge_config.scope,
        }


__all__ = [
    "MEHLER_BRIDGE_DERIVATION",
    "MEHLER_BRIDGE_ID",
    "MEHLER_BRIDGE_SCOPE",
    "MehlerLogicalBridgeConfig",
    "Phase9BackendBMehlerBridgeSimulator",
    "analytic_mehler_isometry",
    "gaussian_component_fock_coefficient",
]
