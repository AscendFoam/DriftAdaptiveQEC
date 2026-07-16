"""Finite-cutoff oscillator density-matrix reference for approximate GKP checks.

The implementation is intentionally independent from the fast effective model.  It
uses a truncated harmonic-oscillator basis and explicit quantum channels so that
trace, positivity, cutoff convergence and simple analytic limits can be audited.
The ``high_fock_leakage_proxy`` is only a cavity high-occupation stressor; it is
not a transmon ``|f>`` model and must not be reported as device leakage physics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import comb, exp, isfinite, pi, sqrt
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye as sparse_eye, kron as sparse_kron
from scipy.sparse.linalg import expm_multiply

from .finite_energy_gkp import (
    FiniteEnergyGKPState,
    LogicalState,
    damped_projector_state,
)
from .quadrature_conventions import DECODER_STANDARDIZATION_SCALE


ComplexMatrix = NDArray[np.complex128]
ComplexVector = NDArray[np.complex128]
MeasurementOutcome = Literal["plus", "minus"]

FOCK_MODEL_SCOPE = (
    "finite-cutoff single-oscillator reference; no transmon levels, pulse Hamiltonian, "
    "device calibration, or hardware claim"
)


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _probability(value: float, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive_int(value: int, name: str, *, minimum: int = 2, maximum: int = 96) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _readonly(array: ArrayLike, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _minimum_eigenvalue(matrix: ComplexMatrix) -> float:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    return float(np.min(np.linalg.eigvalsh(hermitian)))


@dataclass(frozen=True)
class FockDiagnostics:
    cutoff: int
    trace_real: float
    trace_imag: float
    hermiticity_error: float
    minimum_eigenvalue: float
    purity: float
    mean_photon_number: float


@dataclass(frozen=True)
class FiniteCutoffDensity:
    """Validated normalized density matrix with immutable storage."""

    matrix: ComplexMatrix
    cutoff: int

    def __post_init__(self) -> None:
        cutoff = _positive_int(self.cutoff, "cutoff")
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        if matrix.shape != (cutoff, cutoff):
            raise ValueError("matrix shape must equal (cutoff, cutoff)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must contain only finite values")
        hermiticity_error = float(np.linalg.norm(matrix - matrix.conj().T, ord="fro"))
        if hermiticity_error > 1.0e-9:
            raise ValueError("matrix must be Hermitian")
        trace = complex(np.trace(matrix))
        if abs(trace.imag) > 1.0e-10 or abs(trace.real - 1.0) > 1.0e-9:
            raise ValueError("matrix must have unit trace")
        minimum = _minimum_eigenvalue(matrix)
        if minimum < -1.0e-9:
            raise ValueError("matrix must be positive semidefinite")
        cleaned = 0.5 * (matrix + matrix.conj().T)
        object.__setattr__(self, "matrix", _readonly(cleaned, dtype=np.complex128))
        object.__setattr__(self, "cutoff", cutoff)

    @classmethod
    def from_ket(cls, ket: ArrayLike) -> "FiniteCutoffDensity":
        vector = np.asarray(ket, dtype=np.complex128)
        if vector.ndim != 1:
            raise ValueError("ket must be one-dimensional")
        cutoff = _positive_int(vector.size, "ket length")
        if not np.all(np.isfinite(vector)):
            raise ValueError("ket must contain only finite values")
        norm = float(np.vdot(vector, vector).real)
        if not isfinite(norm) or norm <= 0.0:
            raise ValueError("ket must have finite nonzero norm")
        normalized = vector / sqrt(norm)
        return cls(np.outer(normalized, normalized.conj()), cutoff)


@dataclass(frozen=True)
class FockPreparationResult:
    state: FiniteCutoffDensity
    coefficients: ComplexVector
    captured_probability: float
    q_extent: float
    grid_points: int
    source_model: str
    logical_state: str
    source_coordinate_scale: float

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.complex128)
        if coefficients.shape != (self.state.cutoff,):
            raise ValueError("coefficient shape must match state cutoff")
        object.__setattr__(self, "coefficients", _readonly(coefficients, dtype=np.complex128))


@dataclass(frozen=True)
class ModularMeasurementResult:
    outcome: MeasurementOutcome
    probability: float
    state: FiniteCutoffDensity
    effect: ComplexMatrix

    def __post_init__(self) -> None:
        if self.outcome not in {"plus", "minus"}:
            raise ValueError("outcome must be 'plus' or 'minus'")
        _probability(self.probability, "probability")
        effect = np.asarray(self.effect, dtype=np.complex128)
        if effect.shape != self.state.matrix.shape:
            raise ValueError("effect shape must match state")
        object.__setattr__(self, "effect", _readonly(effect, dtype=np.complex128))


@dataclass(frozen=True)
class FockDensityValidationResult:
    cutoffs: tuple[int, ...]
    captured_probabilities: tuple[float, ...]
    adjacent_embedded_fidelities: tuple[float, ...]
    displacement_roundtrip_error: float
    loss_mean_error: float
    thermal_vacuum_mean_error: float
    phase_coherence_error: float
    kerr_population_error: float
    measurement_probability_error: float
    minimum_output_eigenvalue: float
    checks: dict[str, bool]
    scope: str = FOCK_MODEL_SCOPE

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


class FiniteCutoffFockModel:
    """Explicit operators and channels for one truncated oscillator."""

    def __init__(self, cutoff: int) -> None:
        self.cutoff = _positive_int(cutoff, "cutoff")
        annihilation = np.zeros((self.cutoff, self.cutoff), dtype=np.complex128)
        for n in range(1, self.cutoff):
            annihilation[n - 1, n] = sqrt(float(n))
        self.a = _readonly(annihilation, dtype=np.complex128)
        self.adag = _readonly(annihilation.conj().T, dtype=np.complex128)
        self.number = _readonly(self.adag @ self.a, dtype=np.complex128)
        self.identity = _readonly(np.eye(self.cutoff, dtype=np.complex128), dtype=np.complex128)

    def _require_state(self, state: FiniteCutoffDensity) -> None:
        if not isinstance(state, FiniteCutoffDensity):
            raise TypeError("state must be a FiniteCutoffDensity")
        if state.cutoff != self.cutoff:
            raise ValueError("state cutoff does not match model cutoff")

    def diagnostics(self, state: FiniteCutoffDensity) -> FockDiagnostics:
        self._require_state(state)
        matrix = state.matrix
        trace = complex(np.trace(matrix))
        return FockDiagnostics(
            cutoff=self.cutoff,
            trace_real=float(trace.real),
            trace_imag=float(trace.imag),
            hermiticity_error=float(np.linalg.norm(matrix - matrix.conj().T, ord="fro")),
            minimum_eigenvalue=_minimum_eigenvalue(matrix),
            purity=float(np.trace(matrix @ matrix).real),
            mean_photon_number=float(np.trace(matrix @ self.number).real),
        )

    def basis_state(self, n: int) -> FiniteCutoffDensity:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or not 0 <= int(n) < self.cutoff:
            raise ValueError("n must be an integer inside the Fock cutoff")
        ket = np.zeros(self.cutoff, dtype=np.complex128)
        ket[int(n)] = 1.0
        return FiniteCutoffDensity.from_ket(ket)

    def project_finite_energy_gkp(
        self,
        source: FiniteEnergyGKPState,
        *,
        grid_points: int = 8193,
        q_extent: float | None = None,
        source_coordinate_scale: float = 1.0,
    ) -> FockPreparationResult:
        """Project a normalized source wavefunction onto canonical Fock functions.

        ``source_coordinate_scale`` defines ``q_source = scale * q_fock``.  The
        transformed wavefunction is therefore ``sqrt(scale) * psi_source(scale*q)``.
        Keeping this bridge explicit prevents the repository's operational GKP
        syndrome coordinate from being silently identified with the canonical
        oscillator coordinate ``x=(a+a†)/sqrt(2)`` used by the SBS paper.
        """

        if not isinstance(source, FiniteEnergyGKPState):
            raise TypeError("source must be a FiniteEnergyGKPState")
        if not isinstance(grid_points, (int, np.integer)) or isinstance(grid_points, bool):
            raise ValueError("grid_points must be an odd integer >= 1025")
        points = int(grid_points)
        if points < 1025 or points % 2 == 0:
            raise ValueError("grid_points must be an odd integer >= 1025")
        scale = float(source_coordinate_scale)
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("source_coordinate_scale must be finite and positive")
        extent = (
            max(10.0 / scale, source.support_radius / scale)
            if q_extent is None
            else float(q_extent)
        )
        if not isfinite(extent) or extent <= 0.0:
            raise ValueError("q_extent must be finite and positive")

        q = np.linspace(-extent, extent, points, dtype=np.float64)
        psi = sqrt(scale) * np.asarray(source.wavefunction(scale * q), dtype=np.float64)
        phi_previous = np.zeros_like(q)
        phi_current = pi ** (-0.25) * np.exp(-0.5 * q * q)
        coefficients = np.empty(self.cutoff, dtype=np.complex128)
        for n in range(self.cutoff):
            coefficients[n] = complex(simpson(phi_current * psi, x=q))
            phi_next = sqrt(2.0 / (n + 1.0)) * q * phi_current
            if n > 0:
                phi_next -= sqrt(n / (n + 1.0)) * phi_previous
            phi_previous, phi_current = phi_current, phi_next
        captured = float(np.vdot(coefficients, coefficients).real)
        if not isfinite(captured) or captured <= 0.0 or captured > 1.0 + 5.0e-5:
            raise RuntimeError("Fock projection has invalid captured probability")
        state = FiniteCutoffDensity.from_ket(coefficients)
        return FockPreparationResult(
            state=state,
            coefficients=coefficients,
            captured_probability=min(captured, 1.0),
            q_extent=extent,
            grid_points=points,
            source_model=source.model,
            logical_state=source.logical_state,
            source_coordinate_scale=scale,
        )

    def prepare_damped_projector_gkp(
        self,
        logical_state: LogicalState,
        projector_delta: float,
        *,
        grid_points: int = 8193,
        q_extent: float | None = None,
        source_coordinate_scale: float | None = None,
    ) -> FockPreparationResult:
        """Prepare the standard square-code state in canonical Fock coordinates.

        The analytic source is stored in the repository's decoder-standardized
        q chart.  Its coordinate, peak width and envelope are a ``sqrt(2)``
        dilation of canonical ``x``.  Only that registered bridge is accepted
        here; exploratory arbitrary dilations remain available through the lower
        level :meth:`project_finite_energy_gkp` API.
        """

        scale = (
            DECODER_STANDARDIZATION_SCALE
            if source_coordinate_scale is None
            else float(source_coordinate_scale)
        )
        if not isfinite(scale) or not np.isclose(
            scale,
            DECODER_STANDARDIZATION_SCALE,
            rtol=0.0,
            atol=1.0e-14,
        ):
            raise ValueError(
                "damped-projector Fock preparation requires the registered "
                "decoder-to-canonical q scale sqrt(2)"
            )
        source = damped_projector_state(logical_state, projector_delta)
        return self.project_finite_energy_gkp(
            source,
            grid_points=grid_points,
            q_extent=q_extent,
            source_coordinate_scale=scale,
        )

    def displacement_operator(self, alpha: complex) -> ComplexMatrix:
        value = complex(alpha)
        if not isfinite(value.real) or not isfinite(value.imag):
            raise ValueError("alpha must be finite")
        return expm(value * self.adag - value.conjugate() * self.a)

    def displace(self, state: FiniteCutoffDensity, alpha: complex) -> FiniteCutoffDensity:
        self._require_state(state)
        operator = self.displacement_operator(alpha)
        return FiniteCutoffDensity(operator @ state.matrix @ operator.conj().T, self.cutoff)

    def pure_loss(self, state: FiniteCutoffDensity, transmissivity: float) -> FiniteCutoffDensity:
        """Exact truncated-basis bosonic pure-loss Kraus map."""

        self._require_state(state)
        eta = _probability(transmissivity, "transmissivity")
        output = np.zeros_like(state.matrix)
        for lost in range(self.cutoff):
            kraus = np.zeros_like(state.matrix)
            for n in range(lost, self.cutoff):
                kraus[n - lost, n] = (
                    sqrt(float(comb(n, lost)))
                    * (1.0 - eta) ** (0.5 * lost)
                    * eta ** (0.5 * (n - lost))
                )
            output += kraus @ state.matrix @ kraus.conj().T
        return FiniteCutoffDensity(output, self.cutoff)

    @staticmethod
    def _lindblad_superoperator(collapse: csr_matrix, cutoff: int) -> csr_matrix:
        identity = sparse_eye(cutoff, dtype=np.complex128, format="csr")
        cd_c = collapse.getH() @ collapse
        return (
            sparse_kron(collapse.conjugate(), collapse, format="csr")
            - 0.5 * sparse_kron(identity, cd_c, format="csr")
            - 0.5 * sparse_kron(cd_c.transpose(), identity, format="csr")
        )

    def thermal_excitation(
        self,
        state: FiniteCutoffDensity,
        *,
        rate_time: float,
        bath_occupation: float,
    ) -> FiniteCutoffDensity:
        """Finite-cutoff thermal Lindblad evolution via sparse exponential action."""

        self._require_state(state)
        duration = _finite_nonnegative(rate_time, "rate_time")
        occupation = _finite_nonnegative(bath_occupation, "bath_occupation")
        if self.cutoff > 48:
            raise ValueError("thermal sparse propagator is limited to cutoff <= 48")
        if duration == 0.0:
            return FiniteCutoffDensity(state.matrix, self.cutoff)
        annihilation = csr_matrix(self.a)
        creation = csr_matrix(self.adag)
        generator = (occupation + 1.0) * self._lindblad_superoperator(annihilation, self.cutoff)
        if occupation > 0.0:
            generator += occupation * self._lindblad_superoperator(creation, self.cutoff)
        vector = np.asarray(state.matrix).reshape(-1, order="F")
        evolved = expm_multiply(duration * generator, vector).reshape(
            (self.cutoff, self.cutoff), order="F"
        )
        evolved = 0.5 * (evolved + evolved.conj().T)
        trace = float(np.trace(evolved).real)
        if trace <= 0.0 or not isfinite(trace):
            raise RuntimeError("thermal evolution produced invalid trace")
        evolved /= trace
        return FiniteCutoffDensity(evolved, self.cutoff)

    def phase_diffusion(self, state: FiniteCutoffDensity, variance: float) -> FiniteCutoffDensity:
        """Gaussian random phase with supplied angle variance."""

        self._require_state(state)
        angle_variance = _finite_nonnegative(variance, "variance")
        indices = np.arange(self.cutoff, dtype=np.float64)
        differences = indices[:, np.newaxis] - indices[np.newaxis, :]
        damping = np.exp(-0.5 * angle_variance * differences * differences)
        return FiniteCutoffDensity(state.matrix * damping, self.cutoff)

    def kerr(self, state: FiniteCutoffDensity, strength: float) -> FiniteCutoffDensity:
        """Apply ``exp[-i strength n(n-1)/2]``."""

        self._require_state(state)
        value = float(strength)
        if not isfinite(value):
            raise ValueError("strength must be finite")
        n = np.arange(self.cutoff, dtype=np.float64)
        phases = np.exp(-0.5j * value * n * (n - 1.0))
        return FiniteCutoffDensity(phases[:, None] * state.matrix * phases.conj()[None, :], self.cutoff)

    def modular_effects(
        self,
        beta: complex,
        *,
        contrast: float = 1.0,
        phase: float = 0.0,
    ) -> tuple[ComplexMatrix, ComplexMatrix]:
        """Binary POVM from a Hermitian modular-displacement observable."""

        visibility = _probability(contrast, "contrast")
        angle = float(phase)
        if not isfinite(angle):
            raise ValueError("phase must be finite")
        displacement = np.exp(-1j * angle) * self.displacement_operator(beta)
        observable = 0.5 * visibility * (displacement + displacement.conj().T)
        plus = 0.5 * (self.identity + observable)
        minus = self.identity - plus
        return plus, minus

    @staticmethod
    def _positive_square_root(effect: ComplexMatrix) -> ComplexMatrix:
        values, vectors = np.linalg.eigh(0.5 * (effect + effect.conj().T))
        if float(np.min(values)) < -1.0e-10:
            raise RuntimeError("measurement effect is not positive")
        return (vectors * np.sqrt(np.clip(values, 0.0, None))) @ vectors.conj().T

    def modular_measurement(
        self,
        state: FiniteCutoffDensity,
        beta: complex,
        outcome: MeasurementOutcome,
        *,
        contrast: float = 1.0,
        phase: float = 0.0,
    ) -> ModularMeasurementResult:
        self._require_state(state)
        if outcome not in {"plus", "minus"}:
            raise ValueError("outcome must be 'plus' or 'minus'")
        plus, minus = self.modular_effects(beta, contrast=contrast, phase=phase)
        effect = plus if outcome == "plus" else minus
        probability = float(np.trace(effect @ state.matrix).real)
        if probability <= 1.0e-15:
            raise ValueError("requested outcome has zero probability")
        root = self._positive_square_root(effect)
        posterior = root @ state.matrix @ root.conj().T / probability
        return ModularMeasurementResult(
            outcome=outcome,
            probability=probability,
            state=FiniteCutoffDensity(posterior, self.cutoff),
            effect=effect,
        )

    def nonselective_modular_measurement(
        self,
        state: FiniteCutoffDensity,
        beta: complex,
        *,
        contrast: float = 1.0,
        phase: float = 0.0,
    ) -> FiniteCutoffDensity:
        self._require_state(state)
        effects = self.modular_effects(beta, contrast=contrast, phase=phase)
        output = np.zeros_like(state.matrix)
        for effect in effects:
            root = self._positive_square_root(effect)
            output += root @ state.matrix @ root.conj().T
        return FiniteCutoffDensity(output, self.cutoff)

    def high_fock_leakage_proxy(
        self,
        state: FiniteCutoffDensity,
        probability: float,
    ) -> FiniteCutoffDensity:
        """CPTP one-quantum upward-shift stressor, not transmon leakage."""

        self._require_state(state)
        chance = _probability(probability, "probability")
        jump = np.zeros_like(state.matrix)
        for n in range(self.cutoff - 1):
            jump[n + 1, n] = sqrt(chance)
        stay = np.eye(self.cutoff, dtype=np.complex128) * sqrt(1.0 - chance)
        stay[-1, -1] = 1.0
        output = stay @ state.matrix @ stay.conj().T + jump @ state.matrix @ jump.conj().T
        return FiniteCutoffDensity(output, self.cutoff)


def run_fock_density_validation(
    *,
    cutoffs: Sequence[int] = (18, 24, 30, 36),
    projector_delta: float = 0.45,
    grid_points: int = 8193,
) -> FockDensityValidationResult:
    """Run independent convergence, analytic-channel and CPTP gates."""

    validated = tuple(_positive_int(value, "cutoff") for value in cutoffs)
    if len(validated) < 3 or any(b <= a for a, b in zip(validated, validated[1:])):
        raise ValueError("cutoffs must contain at least three strictly increasing values")
    source = damped_projector_state("0", projector_delta)
    extent = max(
        10.0 / DECODER_STANDARDIZATION_SCALE,
        source.support_radius / DECODER_STANDARDIZATION_SCALE,
    )
    preparations = tuple(
        FiniteCutoffFockModel(cutoff).project_finite_energy_gkp(
            source,
            grid_points=grid_points,
            q_extent=extent,
            source_coordinate_scale=DECODER_STANDARDIZATION_SCALE,
        )
        for cutoff in validated
    )
    captured = tuple(item.captured_probability for item in preparations)
    fidelities: list[float] = []
    for lower, upper in zip(preparations, preparations[1:]):
        padded = np.zeros(upper.state.cutoff, dtype=np.complex128)
        padded[: lower.state.cutoff] = lower.coefficients / np.linalg.norm(lower.coefficients)
        upper_normalized = upper.coefficients / np.linalg.norm(upper.coefficients)
        fidelities.append(float(abs(np.vdot(padded, upper_normalized)) ** 2))

    model = FiniteCutoffFockModel(validated[-1])
    test_state = model.prepare_damped_projector_gkp(
        "0", projector_delta, grid_points=grid_points, q_extent=extent
    ).state
    alpha = 0.17 - 0.09j
    roundtrip = model.displace(model.displace(test_state, alpha), -alpha)
    roundtrip_error = float(np.linalg.norm(roundtrip.matrix - test_state.matrix, ord="fro"))

    fock_n = min(5, model.cutoff - 2)
    number_state = model.basis_state(fock_n)
    eta = 0.73
    loss_state = model.pure_loss(number_state, eta)
    loss_error = abs(model.diagnostics(loss_state).mean_photon_number - eta * fock_n)

    vacuum = model.basis_state(0)
    rate_time = 0.21
    bath = 0.37
    thermal = model.thermal_excitation(vacuum, rate_time=rate_time, bath_occupation=bath)
    thermal_expected = bath * (1.0 - exp(-rate_time))
    thermal_error = abs(model.diagnostics(thermal).mean_photon_number - thermal_expected)

    coherence_ket = np.zeros(model.cutoff, dtype=np.complex128)
    coherence_ket[0] = coherence_ket[3] = 1.0 / sqrt(2.0)
    coherence = FiniteCutoffDensity.from_ket(coherence_ket)
    variance = 0.08
    diffused = model.phase_diffusion(coherence, variance)
    expected_coherence = 0.5 * exp(-0.5 * variance * 9.0)
    phase_error = abs(diffused.matrix[0, 3] - expected_coherence)

    kerr = model.kerr(test_state, 0.031)
    kerr_error = float(
        np.max(np.abs(np.diag(kerr.matrix).real - np.diag(test_state.matrix).real))
    )

    plus = model.modular_measurement(test_state, 0.22j, "plus", contrast=0.91)
    minus = model.modular_measurement(test_state, 0.22j, "minus", contrast=0.91)
    probability_error = abs(plus.probability + minus.probability - 1.0)
    leaked = model.high_fock_leakage_proxy(test_state, 0.07)
    minimum_output = min(
        model.diagnostics(loss_state).minimum_eigenvalue,
        model.diagnostics(thermal).minimum_eigenvalue,
        model.diagnostics(diffused).minimum_eigenvalue,
        model.diagnostics(kerr).minimum_eigenvalue,
        model.diagnostics(plus.state).minimum_eigenvalue,
        model.diagnostics(minus.state).minimum_eigenvalue,
        model.diagnostics(leaked).minimum_eigenvalue,
    )

    checks = {
        "captured_probability_increases": all(b > a for a, b in zip(captured, captured[1:])),
        "highest_cutoff_captures_99_999_percent": captured[-1] > 0.99999,
        "adjacent_embedding_converges_above_99_999_percent": fidelities[-1]
        > 0.99999,
        "displacement_roundtrip": roundtrip_error < 1.0e-10,
        "pure_loss_analytic_mean": loss_error < 1.0e-11,
        "thermal_vacuum_analytic_mean": thermal_error < 2.0e-8,
        "phase_diffusion_analytic_coherence": float(abs(phase_error)) < 1.0e-12,
        "kerr_preserves_populations": kerr_error < 1.0e-12,
        "measurement_probabilities_complete": probability_error < 1.0e-11,
        "all_outputs_positive": minimum_output > -1.0e-9,
    }
    return FockDensityValidationResult(
        cutoffs=validated,
        captured_probabilities=captured,
        adjacent_embedded_fidelities=tuple(fidelities),
        displacement_roundtrip_error=roundtrip_error,
        loss_mean_error=float(loss_error),
        thermal_vacuum_mean_error=float(thermal_error),
        phase_coherence_error=float(abs(phase_error)),
        kerr_population_error=kerr_error,
        measurement_probability_error=float(probability_error),
        minimum_output_eigenvalue=float(minimum_output),
        checks=checks,
    )


def write_fock_density_validation(
    result: FockDensityValidationResult,
    output: str | Path,
) -> Path:
    if not isinstance(result, FockDensityValidationResult):
        raise TypeError("result must be a FockDensityValidationResult")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["passed"] = result.passed
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[18, 24, 30, 36])
    parser.add_argument("--projector-delta", type=float, default=0.45)
    parser.add_argument("--grid-points", type=int, default=8193)
    arguments = parser.parse_args()
    result = run_fock_density_validation(
        cutoffs=arguments.cutoffs,
        projector_delta=arguments.projector_delta,
        grid_points=arguments.grid_points,
    )
    write_fock_density_validation(result, arguments.output)
    print(json.dumps({"passed": result.passed, "checks": result.checks}, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "FOCK_MODEL_SCOPE",
    "FockDiagnostics",
    "FiniteCutoffDensity",
    "FockPreparationResult",
    "ModularMeasurementResult",
    "FockDensityValidationResult",
    "FiniteCutoffFockModel",
    "run_fock_density_validation",
    "write_fock_density_validation",
]
