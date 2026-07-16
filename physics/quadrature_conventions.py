"""GKP quadrature charts and normalization bridges.

The repository historically used ``sqrt(2*pi)`` both for a decoder-standardized
logical-cell spacing and for the ``l_S`` displacement-amplitude constant printed
in grid-code papers.  Those numbers are equal but their meanings are not.  This
module keeps four charts explicit and prevents a non-symplectic classical decoder
normalization from being passed off as a canonical oscillator coordinate.

The canonical convention is

``x=(a+a^dagger)/sqrt(2)``, ``p=i(a^dagger-a)/sqrt(2)``, ``[x,p]=i``.

For the square qubit GKP code, adjacent logical cosets are separated by
``sqrt(pi)`` in canonical ``x`` or ``p`` and stabilizers translate by
``2*sqrt(pi)``.  In displacement-amplitude coordinates ``alpha=(x+i*p)/sqrt(2)``,
the stabilizer amplitude is ``l_S=sqrt(2*pi)`` while the logical displacement is
``l_S/2=sqrt(pi/2)``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import erf, isfinite, pi, sqrt
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


QuadratureChartName = Literal[
    "canonical_fock",
    "decoder_standardized",
    "symplectic_bridge",
    "displacement_amplitude",
]
QuadratureAxis = Literal["q", "p"]
RealMatrix = NDArray[np.float64]

CANONICAL_COMMUTATOR = 1.0
CANONICAL_LOGICAL_CELL_SPACING = sqrt(pi)
CANONICAL_STABILIZER_SPACING = 2.0 * sqrt(pi)
DECODER_STANDARDIZATION_SCALE = sqrt(2.0)
DECODER_LOGICAL_CELL_SPACING = sqrt(2.0 * pi)
SYMPLECTIC_Q_LOGICAL_CELL_SPACING = sqrt(2.0 * pi)
SYMPLECTIC_P_LOGICAL_CELL_SPACING = sqrt(pi / 2.0)
DISPLACEMENT_LOGICAL_AMPLITUDE = sqrt(pi / 2.0)
PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE = sqrt(2.0 * pi)


@dataclass(frozen=True)
class QuadratureChart:
    name: QuadratureChartName
    canonical_scale_q: float
    canonical_scale_p: float
    quantum_symplectic: bool
    intended_use: str

    @property
    def canonical_map(self) -> RealMatrix:
        """Return ``z_chart = canonical_map @ z_canonical``."""

        matrix = np.diag((self.canonical_scale_q, self.canonical_scale_p)).astype(
            np.float64
        )
        matrix.setflags(write=False)
        return matrix

    @property
    def commutator_multiplier(self) -> float:
        return self.canonical_scale_q * self.canonical_scale_p

    @property
    def logical_cell_spacings(self) -> tuple[float, float]:
        return (
            self.canonical_scale_q * CANONICAL_LOGICAL_CELL_SPACING,
            self.canonical_scale_p * CANONICAL_LOGICAL_CELL_SPACING,
        )


CHARTS: dict[QuadratureChartName, QuadratureChart] = {
    "canonical_fock": QuadratureChart(
        name="canonical_fock",
        canonical_scale_q=1.0,
        canonical_scale_p=1.0,
        quantum_symplectic=True,
        intended_use="Fock operators, Fourier transforms and physical covariance",
    ),
    "decoder_standardized": QuadratureChart(
        name="decoder_standardized",
        canonical_scale_q=DECODER_STANDARDIZATION_SCALE,
        canonical_scale_p=DECODER_STANDARDIZATION_SCALE,
        quantum_symplectic=False,
        intended_use=(
            "two independently standardized classical syndrome axes only; never a "
            "quantum phase-space operator chart"
        ),
    ),
    "symplectic_bridge": QuadratureChart(
        name="symplectic_bridge",
        canonical_scale_q=DECODER_STANDARDIZATION_SCALE,
        canonical_scale_p=1.0 / DECODER_STANDARDIZATION_SCALE,
        quantum_symplectic=True,
        intended_use=(
            "commutator-preserving anisotropic bridge with q scaled like the decoder"
        ),
    ),
    "displacement_amplitude": QuadratureChart(
        name="displacement_amplitude",
        canonical_scale_q=1.0 / DECODER_STANDARDIZATION_SCALE,
        canonical_scale_p=1.0 / DECODER_STANDARDIZATION_SCALE,
        quantum_symplectic=False,
        intended_use=(
            "real and imaginary parts of alpha in D(alpha); Weyl amplitudes, not a "
            "canonical quadrature pair"
        ),
    ),
}


def chart(name: QuadratureChartName) -> QuadratureChart:
    try:
        return CHARTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown quadrature chart: {name!r}") from exc


def chart_transform(
    source: QuadratureChartName,
    target: QuadratureChartName,
    *,
    require_quantum_symplectic: bool = False,
) -> RealMatrix:
    """Return the linear map ``z_target = M @ z_source``.

    ``require_quantum_symplectic`` rejects either endpoint when it is only a
    classical normalization or a displacement-amplitude coordinate.  A map
    between two quantum charts then has determinant one by construction.
    """

    source_chart = chart(source)
    target_chart = chart(target)
    if require_quantum_symplectic and (
        not source_chart.quantum_symplectic or not target_chart.quantum_symplectic
    ):
        raise ValueError(
            "quantum symplectic conversion requires canonical_fock or "
            "symplectic_bridge endpoints"
        )
    result = target_chart.canonical_map @ np.linalg.inv(source_chart.canonical_map)
    if require_quantum_symplectic and not np.isclose(
        np.linalg.det(result), 1.0, rtol=0.0, atol=1.0e-14
    ):
        raise RuntimeError("registered quantum chart transform is not symplectic")
    result.setflags(write=False)
    return result


def convert_phase_vector(
    value: ArrayLike,
    source: QuadratureChartName,
    target: QuadratureChartName,
    *,
    require_quantum_symplectic: bool = False,
) -> RealMatrix:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape[-1:] != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError("phase vector must be finite with final dimension 2")
    transform = chart_transform(
        source, target, require_quantum_symplectic=require_quantum_symplectic
    )
    return np.asarray(vector @ transform.T, dtype=np.float64)


def convert_covariance(
    covariance: ArrayLike,
    source: QuadratureChartName,
    target: QuadratureChartName,
    *,
    require_quantum_symplectic: bool = False,
) -> RealMatrix:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be a finite 2x2 matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12:
        raise ValueError("covariance must be positive semidefinite")
    transform = chart_transform(
        source, target, require_quantum_symplectic=require_quantum_symplectic
    )
    return np.asarray(transform @ matrix @ transform.T, dtype=np.float64)


def convert_axis_sigma(
    sigma: float,
    source: QuadratureChartName,
    target: QuadratureChartName,
    axis: QuadratureAxis,
) -> float:
    value = float(sigma)
    if not isfinite(value) or value < 0.0:
        raise ValueError("sigma must be finite and non-negative")
    if axis not in {"q", "p"}:
        raise ValueError("axis must be q or p")
    index = 0 if axis == "q" else 1
    transform = chart_transform(source, target)
    return abs(float(transform[index, index])) * value


def wavefunction_dilation(
    source: QuadratureChartName,
    target: QuadratureChartName,
    axis: QuadratureAxis,
) -> tuple[float, float]:
    """Return ``(source_coordinate_factor, amplitude_factor)``.

    To evaluate a normalized source wavefunction in the target chart use
    ``psi_target(y) = amplitude_factor * psi_source(source_coordinate_factor*y)``.
    This helper applies only to the diagonal chart maps registered above.
    """

    if axis not in {"q", "p"}:
        raise ValueError("axis must be q or p")
    index = 0 if axis == "q" else 1
    source_scale = (chart(source).canonical_scale_q, chart(source).canonical_scale_p)[
        index
    ]
    target_scale = (chart(target).canonical_scale_q, chart(target).canonical_scale_p)[
        index
    ]
    coordinate_factor = source_scale / target_scale
    return coordinate_factor, sqrt(abs(coordinate_factor))


def logical_cell_spacing(name: QuadratureChartName, axis: QuadratureAxis) -> float:
    if axis not in {"q", "p"}:
        raise ValueError("axis must be q or p")
    return chart(name).logical_cell_spacings[0 if axis == "q" else 1]


def _gaussian_parity_error(spacing: float, sigma: float) -> float:
    if sigma == 0.0:
        return 0.0
    total = 0.0
    radius = max(12, int(np.ceil(8.0 * sigma / spacing)) + 4)
    root_two_sigma = sqrt(2.0) * sigma
    for alias in range(-radius, radius + 1):
        if alias % 2 == 0:
            continue
        lower = (alias - 0.5) * spacing
        upper = (alias + 0.5) * spacing
        total += 0.5 * (erf(upper / root_two_sigma) - erf(lower / root_two_sigma))
    return total


@dataclass(frozen=True)
class QuadratureConventionValidation:
    canonical_logical_cell_spacing: float
    canonical_stabilizer_spacing: float
    decoder_logical_cell_spacing: float
    paper_stabilizer_displacement_amplitude: float
    decoder_commutator_multiplier: float
    symplectic_bridge_commutator_multiplier: float
    maximum_vector_roundtrip_error: float
    maximum_covariance_roundtrip_error: float
    decoder_wavefunction_norm_error: float
    canonical_q_variance_error: float
    canonical_fourier_p_variance_error: float
    decoder_q_variance_error: float
    parity_alias_invariance_error: float
    fourier_roundtrip_error: float
    logical_peak_roundtrip_error: float
    fourier_reciprocal_spacing_error: float
    checks: dict[str, bool]
    scope: str = (
        "normalization contract only; decoder-standardized axes are classical and "
        "do not constitute a canonical quantum phase-space chart"
    )

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def run_quadrature_convention_validation() -> QuadratureConventionValidation:
    vector = np.array([0.73, -1.11], dtype=np.float64)
    covariance = np.array([[0.31, 0.07], [0.07, 0.46]], dtype=np.float64)
    bridged = convert_phase_vector(
        vector,
        "canonical_fock",
        "symplectic_bridge",
        require_quantum_symplectic=True,
    )
    vector_back = convert_phase_vector(
        bridged,
        "symplectic_bridge",
        "canonical_fock",
        require_quantum_symplectic=True,
    )
    covariance_bridged = convert_covariance(
        covariance,
        "canonical_fock",
        "symplectic_bridge",
        require_quantum_symplectic=True,
    )
    covariance_back = convert_covariance(
        covariance_bridged,
        "symplectic_bridge",
        "canonical_fock",
        require_quantum_symplectic=True,
    )

    q_decoder = np.linspace(-10.0, 10.0, 200_001, dtype=np.float64)
    coordinate_factor, amplitude_factor = wavefunction_dilation(
        "canonical_fock", "decoder_standardized", "q"
    )
    psi_decoder = (
        pi ** (-0.25)
        * amplitude_factor
        * np.exp(-0.5 * (coordinate_factor * q_decoder) ** 2)
    )
    density_decoder = psi_decoder * psi_decoder
    decoder_norm = float(np.trapz(density_decoder, q_decoder))
    decoder_variance = float(
        np.trapz(q_decoder * q_decoder * density_decoder, q_decoder)
    )
    q_canonical = np.linspace(-12.0, 12.0, 32_768, endpoint=False)
    canonical_step = float(q_canonical[1] - q_canonical[0])
    psi_canonical = pi ** (-0.25) * np.exp(-0.5 * q_canonical**2)
    canonical_density = psi_canonical * psi_canonical
    canonical_variance = float(
        np.sum(q_canonical * q_canonical * canonical_density) * canonical_step
    )
    raw_fourier = np.fft.fft(psi_canonical)
    fourier_wavefunction = (
        np.fft.fftshift(raw_fourier) * canonical_step / sqrt(2.0 * pi)
    )
    momentum = np.fft.fftshift(
        2.0 * pi * np.fft.fftfreq(q_canonical.size, d=canonical_step)
    )
    momentum_step = float(momentum[1] - momentum[0])
    momentum_density = np.abs(fourier_wavefunction) ** 2
    momentum_variance = float(
        np.sum(momentum * momentum * momentum_density) * momentum_step
    )
    fourier_roundtrip_error = float(
        np.max(np.abs(np.fft.ifft(raw_fourier) - psi_canonical))
    )

    canonical_peaks = np.column_stack(
        (
            np.arange(-4, 5, dtype=np.float64)
            * CANONICAL_LOGICAL_CELL_SPACING,
            np.zeros(9, dtype=np.float64),
        )
    )
    decoder_peaks = convert_phase_vector(
        canonical_peaks, "canonical_fock", "decoder_standardized"
    )
    peak_roundtrip = convert_phase_vector(
        decoder_peaks, "decoder_standardized", "canonical_fock"
    )
    peak_roundtrip_error = float(np.max(np.abs(peak_roundtrip - canonical_peaks)))
    reciprocal_error = abs(
        2.0 * pi / CANONICAL_LOGICAL_CELL_SPACING
        - CANONICAL_STABILIZER_SPACING
    )

    canonical_sigma = 0.37
    decoder_sigma = convert_axis_sigma(
        canonical_sigma, "canonical_fock", "decoder_standardized", "q"
    )
    canonical_alias = _gaussian_parity_error(
        CANONICAL_LOGICAL_CELL_SPACING, canonical_sigma
    )
    decoder_alias = _gaussian_parity_error(
        DECODER_LOGICAL_CELL_SPACING, decoder_sigma
    )
    vector_error = float(np.max(np.abs(vector_back - vector)))
    covariance_error = float(np.max(np.abs(covariance_back - covariance)))
    norm_error = abs(decoder_norm - 1.0)
    canonical_variance_error = abs(canonical_variance - 0.5)
    momentum_variance_error = abs(momentum_variance - 0.5)
    decoder_variance_error = abs(decoder_variance - 1.0)
    alias_error = abs(canonical_alias - decoder_alias)
    decoder_multiplier = chart("decoder_standardized").commutator_multiplier
    bridge_multiplier = chart("symplectic_bridge").commutator_multiplier

    checks = {
        "canonical_logical_times_stabilizer_is_two_pi": abs(
            CANONICAL_LOGICAL_CELL_SPACING * CANONICAL_STABILIZER_SPACING
            - 2.0 * pi
        )
        < 1.0e-14,
        "paper_lS_is_displacement_stabilizer_not_decoder_cell_semantics": abs(
            PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE
            - DECODER_LOGICAL_CELL_SPACING
        )
        < 1.0e-14,
        "displacement_logical_amplitude_is_half_lS": abs(
            2.0 * DISPLACEMENT_LOGICAL_AMPLITUDE
            - PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE
        )
        < 1.0e-14,
        "decoder_isotropic_chart_is_not_symplectic": abs(decoder_multiplier - 2.0)
        < 1.0e-14
        and not chart("decoder_standardized").quantum_symplectic,
        "anisotropic_bridge_preserves_commutator": abs(bridge_multiplier - 1.0)
        < 1.0e-14,
        "phase_vector_roundtrip": vector_error < 1.0e-14,
        "covariance_roundtrip": covariance_error < 1.0e-14,
        "wavefunction_jacobian_preserves_norm": norm_error < 1.0e-10,
        "gaussian_moments_scale_with_chart": canonical_variance_error < 1.0e-15
        and decoder_variance_error < 1.0e-9,
        "canonical_fourier_gaussian_preserves_qp_moments": momentum_variance_error
        < 1.0e-10,
        "discrete_fourier_roundtrip": fourier_roundtrip_error < 1.0e-12,
        "logical_peak_chart_roundtrip": peak_roundtrip_error < 1.0e-14,
        "fourier_reciprocal_lattice_matches_stabilizer_spacing": reciprocal_error
        < 1.0e-14,
        "parity_alias_probability_is_chart_invariant": alias_error < 1.0e-14,
        "symplectic_cell_area_is_pi": abs(
            SYMPLECTIC_Q_LOGICAL_CELL_SPACING
            * SYMPLECTIC_P_LOGICAL_CELL_SPACING
            - pi
        )
        < 1.0e-14,
    }
    return QuadratureConventionValidation(
        canonical_logical_cell_spacing=CANONICAL_LOGICAL_CELL_SPACING,
        canonical_stabilizer_spacing=CANONICAL_STABILIZER_SPACING,
        decoder_logical_cell_spacing=DECODER_LOGICAL_CELL_SPACING,
        paper_stabilizer_displacement_amplitude=PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE,
        decoder_commutator_multiplier=decoder_multiplier,
        symplectic_bridge_commutator_multiplier=bridge_multiplier,
        maximum_vector_roundtrip_error=vector_error,
        maximum_covariance_roundtrip_error=covariance_error,
        decoder_wavefunction_norm_error=norm_error,
        canonical_q_variance_error=canonical_variance_error,
        canonical_fourier_p_variance_error=momentum_variance_error,
        decoder_q_variance_error=decoder_variance_error,
        parity_alias_invariance_error=alias_error,
        fourier_roundtrip_error=fourier_roundtrip_error,
        logical_peak_roundtrip_error=peak_roundtrip_error,
        fourier_reciprocal_spacing_error=reciprocal_error,
        checks=checks,
    )


def write_quadrature_convention_validation(
    result: QuadratureConventionValidation, output: str | Path
) -> Path:
    if not isinstance(result, QuadratureConventionValidation):
        raise TypeError("result must be a QuadratureConventionValidation")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["passed"] = result.passed
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_quadrature_convention_validation()
    if args.output is not None:
        write_quadrature_convention_validation(result, args.output)
    print(json.dumps({"passed": result.passed, "checks": result.checks}, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_COMMUTATOR",
    "CANONICAL_LOGICAL_CELL_SPACING",
    "CANONICAL_STABILIZER_SPACING",
    "CHARTS",
    "DECODER_LOGICAL_CELL_SPACING",
    "DECODER_STANDARDIZATION_SCALE",
    "DISPLACEMENT_LOGICAL_AMPLITUDE",
    "PAPER_STABILIZER_DISPLACEMENT_AMPLITUDE",
    "QuadratureAxis",
    "QuadratureChart",
    "QuadratureChartName",
    "QuadratureConventionValidation",
    "SYMPLECTIC_P_LOGICAL_CELL_SPACING",
    "SYMPLECTIC_Q_LOGICAL_CELL_SPACING",
    "chart",
    "chart_transform",
    "convert_axis_sigma",
    "convert_covariance",
    "convert_phase_vector",
    "logical_cell_spacing",
    "run_quadrature_convention_validation",
    "wavefunction_dilation",
    "write_quadrature_convention_validation",
]
