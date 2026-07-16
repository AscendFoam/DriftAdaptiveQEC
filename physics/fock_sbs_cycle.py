"""One paper-aligned SBS QEC cycle in a finite Fock basis.

The constituent Kraus operators are the analytic operators printed in the
Supplementary Information of Sivak *et al.*, ``Real-time quantum error
correction beyond break-even`` (arXiv:2211.09116, SBS protocol section).  A
finite Fock cutoff does not preserve the infinite-dimensional Weyl algebra, so
the raw truncated operators are generally not trace preserving.  This module
keeps those raw operators for audit and applies the shared right completion

    K_b -> K_b (sum_b K_b^dagger K_b)^(-1/2)

to obtain a finite-dimensional CPTP instrument.  The completion is reported,
not hidden.  The model remains a single-oscillator reference: it is not a
pulse/ECD Hamiltonian, explicit transmon simulation, or hardware claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from math import isfinite, pi, sqrt
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fock_density_model import FiniteCutoffDensity, FiniteCutoffFockModel, FockDiagnostics
from .sbs_error_space import PauliFrame, SBS_PROTOCOL_ID


ComplexMatrix = NDArray[np.complex128]
Quadrature = Literal["X", "Z"]
BinaryOutcome = Literal["g", "e"]
LogicalLabel = Literal["0", "1", "+", "-", "+i", "-i", "mixed"]
CompletionPolicy = Literal["shared_right_inverse_sqrt"]

PAPER_CANONICAL_SOURCE_SCALE = sqrt(2.0)
FOCK_SBS_CYCLE_SCOPE = (
    "finite-cutoff single-oscillator implementation of the analytic SBS Kraus map; "
    "raw truncation diagnostics plus audited CPTP completion; no pulse-level ECD, "
    "explicit transmon, device calibration, or hardware claim"
)

_PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_IDENTITY_2 = np.eye(2, dtype=np.complex128)


def _readonly(array: ArrayLike, *, dtype: np.dtype = np.complex128) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _unit_interval(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _hermitian_function(matrix: ComplexMatrix, function: object) -> ComplexMatrix:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    values, vectors = np.linalg.eigh(hermitian)
    evaluated = function(values)  # type: ignore[operator]
    return _readonly((vectors * evaluated) @ vectors.conj().T)


def logical_density(label: LogicalLabel) -> ComplexMatrix:
    """Return a Pauli eigenstate or the maximally mixed logical state."""

    states: Mapping[str, ComplexMatrix] = {
        "0": 0.5 * (_IDENTITY_2 + _PAULI_Z),
        "1": 0.5 * (_IDENTITY_2 - _PAULI_Z),
        "+": 0.5 * (_IDENTITY_2 + _PAULI_X),
        "-": 0.5 * (_IDENTITY_2 - _PAULI_X),
        "+i": 0.5 * (_IDENTITY_2 + _PAULI_Y),
        "-i": 0.5 * (_IDENTITY_2 - _PAULI_Y),
        "mixed": 0.5 * _IDENTITY_2,
    }
    if label not in states:
        raise ValueError("unknown logical-state label")
    return _readonly(states[label])


def _validate_logical_density(matrix: ArrayLike) -> ComplexMatrix:
    state = np.asarray(matrix, dtype=np.complex128)
    if state.shape != (2, 2) or not np.all(np.isfinite(state)):
        raise ValueError("logical density must be a finite 2x2 matrix")
    if np.linalg.norm(state - state.conj().T, ord="fro") > 1.0e-10:
        raise ValueError("logical density must be Hermitian")
    if abs(complex(np.trace(state)) - 1.0) > 1.0e-10:
        raise ValueError("logical density must have unit trace")
    if float(np.min(np.linalg.eigvalsh(state))) < -1.0e-10:
        raise ValueError("logical density must be positive semidefinite")
    return _readonly(0.5 * (state + state.conj().T))


@dataclass(frozen=True)
class FockIdleConfig:
    displacement: complex = 0.0j
    loss_transmissivity: float = 1.0
    thermal_rate_time: float = 0.0
    thermal_bath_occupation: float = 0.0
    phase_diffusion_variance: float = 0.0
    kerr_strength: float = 0.0
    high_fock_proxy_probability: float = 0.0

    def __post_init__(self) -> None:
        displacement = complex(self.displacement)
        if not isfinite(displacement.real) or not isfinite(displacement.imag):
            raise ValueError("displacement must be finite")
        object.__setattr__(self, "displacement", displacement)
        object.__setattr__(
            self,
            "loss_transmissivity",
            _unit_interval(self.loss_transmissivity, "loss_transmissivity"),
        )
        for name in ("thermal_rate_time", "thermal_bath_occupation", "phase_diffusion_variance"):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        kerr = float(self.kerr_strength)
        if not isfinite(kerr):
            raise ValueError("kerr_strength must be finite")
        object.__setattr__(self, "kerr_strength", kerr)
        object.__setattr__(
            self,
            "high_fock_proxy_probability",
            _unit_interval(self.high_fock_proxy_probability, "high_fock_proxy_probability"),
        )


@dataclass(frozen=True)
class SBSFockCycleConfig:
    cutoff: int = 24
    projector_delta: float = 0.34
    grid_points: int = 8193
    readout_confusion: tuple[tuple[float, float], tuple[float, float]] = (
        (0.99, 0.01),
        (0.02, 0.98),
    )
    controller_residual_phase_by_observed: tuple[
        tuple[float, float], tuple[float, float]
    ] = ((0.0, 0.0), (0.0, 0.0))
    idle: FockIdleConfig = FockIdleConfig()
    completion_policy: CompletionPolicy = "shared_right_inverse_sqrt"
    protocol_id: str = SBS_PROTOCOL_ID
    scope: str = FOCK_SBS_CYCLE_SCOPE

    def __post_init__(self) -> None:
        if not isinstance(self.cutoff, (int, np.integer)) or isinstance(self.cutoff, bool):
            raise ValueError("cutoff must be an integer")
        cutoff = int(self.cutoff)
        if not 8 <= cutoff <= 48:
            raise ValueError("cutoff must lie in [8, 48]")
        object.__setattr__(self, "cutoff", cutoff)
        delta = float(self.projector_delta)
        if not isfinite(delta) or delta <= 0.0:
            raise ValueError("projector_delta must be finite and positive")
        object.__setattr__(self, "projector_delta", delta)
        if (
            not isinstance(self.grid_points, (int, np.integer))
            or isinstance(self.grid_points, bool)
            or int(self.grid_points) < 1025
            or int(self.grid_points) % 2 == 0
        ):
            raise ValueError("grid_points must be an odd integer >= 1025")
        object.__setattr__(self, "grid_points", int(self.grid_points))
        confusion = np.asarray(self.readout_confusion, dtype=np.float64)
        if confusion.shape != (2, 2) or not np.all(np.isfinite(confusion)):
            raise ValueError("readout_confusion must be a finite 2x2 matrix")
        if np.any(confusion < 0.0) or np.any(confusion > 1.0):
            raise ValueError("readout_confusion entries must lie in [0, 1]")
        if not np.allclose(np.sum(confusion, axis=1), 1.0, atol=1.0e-12):
            raise ValueError("readout_confusion rows must sum to one")
        object.__setattr__(
            self,
            "readout_confusion",
            tuple(tuple(float(value) for value in row) for row in confusion),
        )
        phases = np.asarray(self.controller_residual_phase_by_observed, dtype=np.float64)
        if phases.shape != (2, 2) or not np.all(np.isfinite(phases)):
            raise ValueError("controller_residual_phase_by_observed must be finite 2x2")
        if np.any(np.abs(phases) > pi):
            raise ValueError("controller residual phases must lie in [-pi, pi]")
        object.__setattr__(
            self,
            "controller_residual_phase_by_observed",
            tuple(tuple(float(value) for value in row) for row in phases),
        )
        if not isinstance(self.idle, FockIdleConfig):
            raise TypeError("idle must be a FockIdleConfig")
        if self.completion_policy != "shared_right_inverse_sqrt":
            raise ValueError("only the audited shared-right completion is supported")
        if self.protocol_id != SBS_PROTOCOL_ID:
            raise ValueError("T2.3.2 implements only the frozen SBS main protocol")
        if self.scope != FOCK_SBS_CYCLE_SCOPE:
            raise ValueError("scope must preserve the fail-closed model boundary")


@dataclass(frozen=True)
class FockCodeBasis:
    isometry: ComplexMatrix
    projector: ComplexMatrix
    raw_gram: ComplexMatrix
    raw_overlap: complex
    captured_probabilities: tuple[float, float]
    cutoff: int
    projector_delta: float
    source_coordinate_scale: float = PAPER_CANONICAL_SOURCE_SCALE

    def __post_init__(self) -> None:
        isometry = np.asarray(self.isometry, dtype=np.complex128)
        projector = np.asarray(self.projector, dtype=np.complex128)
        gram = np.asarray(self.raw_gram, dtype=np.complex128)
        if isometry.shape != (self.cutoff, 2):
            raise ValueError("isometry shape must be (cutoff, 2)")
        if projector.shape != (self.cutoff, self.cutoff) or gram.shape != (2, 2):
            raise ValueError("code-basis matrix shapes are inconsistent")
        if np.linalg.norm(isometry.conj().T @ isometry - _IDENTITY_2) > 1.0e-9:
            raise ValueError("code isometry columns must be orthonormal")
        if np.linalg.norm(projector @ projector - projector) > 1.0e-9:
            raise ValueError("code projector must be idempotent")
        object.__setattr__(self, "isometry", _readonly(isometry))
        object.__setattr__(self, "projector", _readonly(projector))
        object.__setattr__(self, "raw_gram", _readonly(gram))


@dataclass(frozen=True)
class SBSKrausDiagnostics:
    quadrature: Quadrature
    raw_completeness_frobenius_error: float
    raw_completeness_operator_error: float
    raw_code_subspace_completeness_error: float
    raw_gram_minimum_eigenvalue: float
    raw_gram_maximum_eigenvalue: float
    raw_gram_condition_number: float
    completion_pair_frobenius_change: float
    completed_completeness_frobenius_error: float
    completed_completeness_operator_error: float


@dataclass(frozen=True)
class FockLogicalProjection:
    code_survival_probability: float
    leakage_probability: float
    raw_logical_density: ComplexMatrix
    frame_corrected_logical_density: ComplexMatrix
    raw_purity: float
    frame: PauliFrame

    def __post_init__(self) -> None:
        survival = _unit_interval(self.code_survival_probability, "code_survival_probability")
        leakage = _unit_interval(self.leakage_probability, "leakage_probability")
        if abs(survival + leakage - 1.0) > 1.0e-9:
            raise ValueError("survival and leakage must sum to one")
        object.__setattr__(self, "raw_logical_density", _readonly(self.raw_logical_density))
        object.__setattr__(
            self,
            "frame_corrected_logical_density",
            _readonly(self.frame_corrected_logical_density),
        )


@dataclass(frozen=True)
class SBSFockObservedConstituent:
    quadrature: Quadrature
    observed_outcome: BinaryOutcome
    chronological_index: int
    controller_residual_phase: float
    input_frame: PauliFrame
    output_frame: PauliFrame


@dataclass(frozen=True)
class SBSFockTruthConstituent:
    quadrature: Quadrature
    hidden_outcome: BinaryOutcome
    hidden_probability: float
    observation_probability_given_hidden: float
    code_survival_before: float
    code_survival_after_hidden_kraus: float


@dataclass(frozen=True)
class SBSFockCycleBranch:
    probability: float
    chronological_observed_outcomes: tuple[BinaryOutcome, BinaryOutcome]
    chronological_hidden_outcomes: tuple[BinaryOutcome, BinaryOutcome]
    observed_kraus_label: str
    hidden_kraus_label: str
    observed_constituents: tuple[SBSFockObservedConstituent, SBSFockObservedConstituent]
    truth_constituents: tuple[SBSFockTruthConstituent, SBSFockTruthConstituent]
    final_state: FiniteCutoffDensity
    logical_projection: FockLogicalProjection | None
    input_frame: PauliFrame
    output_frame: PauliFrame


@dataclass(frozen=True)
class SBSFockExactCycleResult:
    initial_state: FiniteCutoffDensity
    post_idle_state: FiniteCutoffDensity
    post_idle_diagnostics: FockDiagnostics
    branches: tuple[SBSFockCycleBranch, ...]
    unconditional_state: FiniteCutoffDensity
    unconditional_projection: FockLogicalProjection
    total_probability: float
    input_frame: PauliFrame
    output_frame: PauliFrame
    protocol_id: str = SBS_PROTOCOL_ID
    scope: str = FOCK_SBS_CYCLE_SCOPE


@dataclass(frozen=True)
class SBSCutoffValidationPoint:
    cutoff: int
    average_conditional_logical_fidelity: float
    average_code_survival: float
    average_code_weighted_fidelity: float
    maximum_raw_code_completeness_error: float
    maximum_completed_completeness_error: float


@dataclass(frozen=True)
class SBSFockCycleValidationResult:
    cutoff: int
    projector_delta: float
    source_coordinate_scale: float
    code_raw_overlap_abs: float
    code_isometry_error: float
    kraus_diagnostics: tuple[SBSKrausDiagnostics, SBSKrausDiagnostics]
    exact_branch_count: int
    exact_probability_error: float
    minimum_branch_eigenvalue: float
    monte_carlo_max_z_score: float
    clean_average_conditional_logical_fidelity: float
    clean_average_code_survival: float
    clean_average_code_weighted_fidelity: float
    noisy_average_conditional_logical_fidelity: float
    noisy_average_code_survival: float
    noisy_average_code_weighted_fidelity: float
    one_cycle_photon_error_code_survival_gain: float
    cutoff_sweep: tuple[SBSCutoffValidationPoint, ...]
    checks: dict[str, bool]
    scope: str = FOCK_SBS_CYCLE_SCOPE

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


class SBSFockOneRoundSimulator:
    """Exact hidden/observed branching for one analytic SBS X-then-Z cycle."""

    def __init__(self, config: SBSFockCycleConfig) -> None:
        if not isinstance(config, SBSFockCycleConfig):
            raise TypeError("config must be an SBSFockCycleConfig")
        self.config = config
        self.model = FiniteCutoffFockModel(config.cutoff)
        self.code_basis = self._build_code_basis()
        self.x = _readonly((self.model.a + self.model.adag) / sqrt(2.0))
        self.p = _readonly(1.0j * (self.model.adag - self.model.a) / sqrt(2.0))
        self._raw_kraus: dict[str, tuple[ComplexMatrix, ComplexMatrix]] = {}
        self._kraus: dict[str, tuple[ComplexMatrix, ComplexMatrix]] = {}
        self._kraus_diagnostics: dict[str, SBSKrausDiagnostics] = {}
        for quadrature in ("X", "Z"):
            raw = self._paper_constituent_kraus(quadrature)
            completed, diagnostics = self._complete_constituent(raw, quadrature)
            self._raw_kraus[quadrature] = raw
            self._kraus[quadrature] = completed
            self._kraus_diagnostics[quadrature] = diagnostics
        self._controller_unities: dict[tuple[str, str], ComplexMatrix] = {}
        phases = np.asarray(config.controller_residual_phase_by_observed)
        number = np.arange(config.cutoff, dtype=np.float64)
        for q_index, quadrature in enumerate(("X", "Z")):
            for outcome_index, outcome in enumerate(("g", "e")):
                angle = float(phases[q_index, outcome_index])
                self._controller_unities[(quadrature, outcome)] = _readonly(
                    np.diag(np.exp(1.0j * angle * number))
                )

    def _build_code_basis(self) -> FockCodeBasis:
        preparations = [
            self.model.prepare_damped_projector_gkp(
                label,
                self.config.projector_delta,
                grid_points=self.config.grid_points,
                source_coordinate_scale=PAPER_CANONICAL_SOURCE_SCALE,
            )
            for label in ("0", "1")
        ]
        raw = np.column_stack(
            [item.coefficients / np.linalg.norm(item.coefficients) for item in preparations]
        )
        gram = raw.conj().T @ raw
        values, vectors = np.linalg.eigh(gram)
        if float(np.min(values)) <= 1.0e-10:
            raise RuntimeError("finite-cutoff logical codewords are linearly dependent")
        inverse_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
        isometry = raw @ inverse_sqrt
        projector = isometry @ isometry.conj().T
        return FockCodeBasis(
            isometry=isometry,
            projector=projector,
            raw_gram=gram,
            raw_overlap=complex(gram[0, 1]),
            captured_probabilities=tuple(
                float(item.captured_probability) for item in preparations
            ),
            cutoff=self.config.cutoff,
            projector_delta=self.config.projector_delta,
        )

    def _paper_constituent_kraus(
        self, quadrature: Quadrature
    ) -> tuple[ComplexMatrix, ComplexMatrix]:
        """Evaluate the paper's printed analytic Kraus pair at finite cutoff."""

        delta_squared = self.config.projector_delta**2
        root_pi = sqrt(pi)
        if quadrature == "X":
            substituted_x, substituted_p = self.x, self.p
        elif quadrature == "Z":
            substituted_x, substituted_p = -self.p, self.x
        else:
            raise ValueError("quadrature must be 'X' or 'Z'")
        cosine_p = _hermitian_function(root_pi * substituted_p, np.cos)
        sine_p = _hermitian_function(root_pi * substituted_p, np.sin)
        cosine_small_x = _hermitian_function(root_pi * delta_squared * substituted_x, np.cos)
        sine_small_x = _hermitian_function(root_pi * delta_squared * substituted_x, np.sin)
        k_g = cosine_p @ cosine_small_x + np.sin(pi * delta_squared / 2.0) * cosine_p
        k_e = (
            -np.cos(pi * delta_squared / 2.0) * sine_p
            + 1.0j * cosine_p @ sine_small_x
        )
        return _readonly(k_g), _readonly(k_e)

    def _complete_constituent(
        self,
        raw: tuple[ComplexMatrix, ComplexMatrix],
        quadrature: Quadrature,
    ) -> tuple[tuple[ComplexMatrix, ComplexMatrix], SBSKrausDiagnostics]:
        gram = sum(operator.conj().T @ operator for operator in raw)
        gram = 0.5 * (gram + gram.conj().T)
        values, vectors = np.linalg.eigh(gram)
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if minimum <= 1.0e-12:
            raise RuntimeError("raw finite-cutoff SBS Kraus Gram matrix is singular")
        inverse_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
        completed = tuple(_readonly(operator @ inverse_sqrt) for operator in raw)
        completed_gram = sum(operator.conj().T @ operator for operator in completed)
        raw_residual = gram - self.model.identity
        completed_residual = completed_gram - self.model.identity
        code_residual = (
            self.code_basis.isometry.conj().T
            @ raw_residual
            @ self.code_basis.isometry
        )
        pair_change = sqrt(
            sum(float(np.linalg.norm(new - old, ord="fro") ** 2) for old, new in zip(raw, completed))
        )
        diagnostics = SBSKrausDiagnostics(
            quadrature=quadrature,
            raw_completeness_frobenius_error=float(np.linalg.norm(raw_residual, ord="fro")),
            raw_completeness_operator_error=float(np.linalg.norm(raw_residual, ord=2)),
            raw_code_subspace_completeness_error=float(np.linalg.norm(code_residual, ord="fro")),
            raw_gram_minimum_eigenvalue=minimum,
            raw_gram_maximum_eigenvalue=maximum,
            raw_gram_condition_number=maximum / minimum,
            completion_pair_frobenius_change=pair_change,
            completed_completeness_frobenius_error=float(
                np.linalg.norm(completed_residual, ord="fro")
            ),
            completed_completeness_operator_error=float(np.linalg.norm(completed_residual, ord=2)),
        )
        return (completed[0], completed[1]), diagnostics

    def constituent_kraus(
        self, quadrature: Quadrature, *, raw: bool = False
    ) -> tuple[ComplexMatrix, ComplexMatrix]:
        if quadrature not in {"X", "Z"}:
            raise ValueError("quadrature must be 'X' or 'Z'")
        return self._raw_kraus[quadrature] if raw else self._kraus[quadrature]

    def kraus_diagnostics(self, quadrature: Quadrature) -> SBSKrausDiagnostics:
        if quadrature not in {"X", "Z"}:
            raise ValueError("quadrature must be 'X' or 'Z'")
        return self._kraus_diagnostics[quadrature]

    def _require_state(self, state: FiniteCutoffDensity) -> None:
        if not isinstance(state, FiniteCutoffDensity):
            raise TypeError("state must be a FiniteCutoffDensity")
        if state.cutoff != self.config.cutoff:
            raise ValueError("state cutoff does not match cycle cutoff")

    def initialize(
        self,
        logical_state: LogicalLabel | ArrayLike,
        *,
        frame: PauliFrame | None = None,
    ) -> FiniteCutoffDensity:
        logical = (
            logical_density(logical_state)
            if isinstance(logical_state, str)
            else _validate_logical_density(logical_state)
        )
        active_frame = PauliFrame() if frame is None else frame
        if not isinstance(active_frame, PauliFrame):
            raise TypeError("frame must be a PauliFrame")
        physical_logical = active_frame.unitary @ logical @ active_frame.unitary.conj().T
        matrix = self.code_basis.isometry @ physical_logical @ self.code_basis.isometry.conj().T
        return FiniteCutoffDensity(matrix, self.config.cutoff)

    def logical_project(
        self,
        state: FiniteCutoffDensity,
        *,
        frame: PauliFrame,
    ) -> FockLogicalProjection:
        self._require_state(state)
        if not isinstance(frame, PauliFrame):
            raise TypeError("frame must be a PauliFrame")
        encoded = self.code_basis.isometry.conj().T @ state.matrix @ self.code_basis.isometry
        survival = float(np.trace(encoded).real)
        if survival <= 1.0e-14:
            raise RuntimeError("logical projection has numerically zero code survival")
        raw = encoded / survival
        frame_unitary = frame.unitary
        corrected = frame_unitary.conj().T @ raw @ frame_unitary
        clipped = min(max(survival, 0.0), 1.0)
        return FockLogicalProjection(
            code_survival_probability=clipped,
            leakage_probability=1.0 - clipped,
            raw_logical_density=raw,
            frame_corrected_logical_density=corrected,
            raw_purity=float(np.trace(raw @ raw).real),
            frame=frame,
        )

    def _logical_project_if_supported(
        self,
        state: FiniteCutoffDensity,
        *,
        frame: PauliFrame,
    ) -> FockLogicalProjection | None:
        """Return ``None`` when a conditional branch has zero code support."""

        encoded = self.code_basis.isometry.conj().T @ state.matrix @ self.code_basis.isometry
        if float(np.trace(encoded).real) <= 1.0e-14:
            return None
        return self.logical_project(state, frame=frame)

    def apply_idle(self, state: FiniteCutoffDensity) -> FiniteCutoffDensity:
        self._require_state(state)
        idle = self.config.idle
        result = state
        if idle.displacement != 0.0j:
            result = self.model.displace(result, idle.displacement)
        if idle.loss_transmissivity != 1.0:
            result = self.model.pure_loss(result, idle.loss_transmissivity)
        if idle.thermal_rate_time != 0.0:
            result = self.model.thermal_excitation(
                result,
                rate_time=idle.thermal_rate_time,
                bath_occupation=idle.thermal_bath_occupation,
            )
        if idle.phase_diffusion_variance != 0.0:
            result = self.model.phase_diffusion(result, idle.phase_diffusion_variance)
        if idle.kerr_strength != 0.0:
            result = self.model.kerr(result, idle.kerr_strength)
        if idle.high_fock_proxy_probability != 0.0:
            result = self.model.high_fock_leakage_proxy(
                result, idle.high_fock_proxy_probability
            )
        return result

    def _code_survival(self, state: FiniteCutoffDensity) -> float:
        return float(np.trace(self.code_basis.projector @ state.matrix).real)

    @staticmethod
    def _kraus_label(chronological_outcomes: Sequence[BinaryOutcome]) -> str:
        if len(chronological_outcomes) != 2:
            raise ValueError("full cycle requires exactly X then Z outcomes")
        x_outcome, z_outcome = chronological_outcomes
        return f"K_{z_outcome}{x_outcome}"

    @staticmethod
    def _after_constituent(frame: PauliFrame, quadrature: Quadrature) -> PauliFrame:
        return frame.after_x_constituent() if quadrature == "X" else frame.after_z_constituent()

    def _apply_unitary(
        self, state: FiniteCutoffDensity, unitary: ComplexMatrix
    ) -> FiniteCutoffDensity:
        return FiniteCutoffDensity(
            unitary @ state.matrix @ unitary.conj().T,
            self.config.cutoff,
        )

    def _controller_phase(self, quadrature: Quadrature, outcome: BinaryOutcome) -> float:
        q_index = 0 if quadrature == "X" else 1
        outcome_index = 0 if outcome == "g" else 1
        return float(self.config.controller_residual_phase_by_observed[q_index][outcome_index])

    def _constituent_branches(
        self,
        state: FiniteCutoffDensity,
        quadrature: Quadrature,
        input_frame: PauliFrame,
        chronological_index: int,
    ) -> list[
        tuple[
            float,
            SBSFockObservedConstituent,
            SBSFockTruthConstituent,
            FiniteCutoffDensity,
            PauliFrame,
        ]
    ]:
        before = self._code_survival(state)
        output_frame = self._after_constituent(input_frame, quadrature)
        confusion = np.asarray(self.config.readout_confusion, dtype=np.float64)
        results = []
        for hidden_index, hidden in enumerate(("g", "e")):
            kraus = self._kraus[quadrature][hidden_index]
            unnormalized = kraus @ state.matrix @ kraus.conj().T
            hidden_probability = float(np.trace(unnormalized).real)
            if hidden_probability <= 1.0e-15:
                continue
            hidden_state = FiniteCutoffDensity(
                unnormalized / hidden_probability, self.config.cutoff
            )
            after_hidden = self._code_survival(hidden_state)
            for observed_index, observed in enumerate(("g", "e")):
                observation_probability = float(confusion[hidden_index, observed_index])
                if observation_probability <= 0.0:
                    continue
                phase = self._controller_phase(quadrature, observed)
                acted = self._apply_unitary(
                    hidden_state, self._controller_unities[(quadrature, observed)]
                )
                observed_record = SBSFockObservedConstituent(
                    quadrature=quadrature,
                    observed_outcome=observed,
                    chronological_index=chronological_index,
                    controller_residual_phase=phase,
                    input_frame=input_frame,
                    output_frame=output_frame,
                )
                truth_record = SBSFockTruthConstituent(
                    quadrature=quadrature,
                    hidden_outcome=hidden,
                    hidden_probability=hidden_probability,
                    observation_probability_given_hidden=observation_probability,
                    code_survival_before=before,
                    code_survival_after_hidden_kraus=after_hidden,
                )
                results.append(
                    (
                        hidden_probability * observation_probability,
                        observed_record,
                        truth_record,
                        acted,
                        output_frame,
                    )
                )
        return results

    def run_exact_cycle(
        self,
        initial_state: FiniteCutoffDensity,
        *,
        input_frame: PauliFrame | None = None,
    ) -> SBSFockExactCycleResult:
        self._require_state(initial_state)
        frame = PauliFrame() if input_frame is None else input_frame
        if not isinstance(frame, PauliFrame):
            raise TypeError("input_frame must be a PauliFrame")
        post_idle = self.apply_idle(initial_state)
        branches: list[SBSFockCycleBranch] = []
        unconditional = np.zeros_like(initial_state.matrix)
        for x_weight, x_observed, x_truth, x_state, x_frame in self._constituent_branches(
            post_idle, "X", frame, 0
        ):
            for z_weight, z_observed, z_truth, z_state, z_frame in self._constituent_branches(
                x_state, "Z", x_frame, 1
            ):
                probability = x_weight * z_weight
                observed_pair = (x_observed.observed_outcome, z_observed.observed_outcome)
                hidden_pair = (x_truth.hidden_outcome, z_truth.hidden_outcome)
                branch = SBSFockCycleBranch(
                    probability=probability,
                    chronological_observed_outcomes=observed_pair,
                    chronological_hidden_outcomes=hidden_pair,
                    observed_kraus_label=self._kraus_label(observed_pair),
                    hidden_kraus_label=self._kraus_label(hidden_pair),
                    observed_constituents=(x_observed, z_observed),
                    truth_constituents=(x_truth, z_truth),
                    final_state=z_state,
                    logical_projection=self._logical_project_if_supported(
                        z_state, frame=z_frame
                    ),
                    input_frame=frame,
                    output_frame=z_frame,
                )
                branches.append(branch)
                unconditional += probability * z_state.matrix
        total = float(sum(branch.probability for branch in branches))
        if abs(total - 1.0) > 1.0e-9:
            raise RuntimeError("exact SBS branch probabilities do not sum to one")
        unconditional_state = FiniteCutoffDensity(unconditional / total, self.config.cutoff)
        output_frame = frame.after_full_sbs_cycle()
        return SBSFockExactCycleResult(
            initial_state=initial_state,
            post_idle_state=post_idle,
            post_idle_diagnostics=self.model.diagnostics(post_idle),
            branches=tuple(branches),
            unconditional_state=unconditional_state,
            unconditional_projection=self.logical_project(
                unconditional_state, frame=output_frame
            ),
            total_probability=total,
            input_frame=frame,
            output_frame=output_frame,
        )

    @staticmethod
    def sample_branch(
        exact_result: SBSFockExactCycleResult,
        rng: np.random.Generator,
    ) -> SBSFockCycleBranch:
        if not isinstance(exact_result, SBSFockExactCycleResult):
            raise TypeError("exact_result must be an SBSFockExactCycleResult")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator")
        probabilities = np.asarray(
            [branch.probability for branch in exact_result.branches], dtype=np.float64
        )
        probabilities /= np.sum(probabilities)
        index = int(rng.choice(len(probabilities), p=probabilities))
        return exact_result.branches[index]


def _average_metrics(simulator: SBSFockOneRoundSimulator) -> tuple[float, float, float]:
    conditional_fidelities = []
    survivals = []
    weighted_fidelities = []
    for label in ("0", "1", "+", "-", "+i", "-i"):
        target = logical_density(label)
        result = simulator.run_exact_cycle(simulator.initialize(label))
        projection = result.unconditional_projection
        fidelity = float(
            np.trace(projection.frame_corrected_logical_density @ target).real
        )
        conditional_fidelities.append(fidelity)
        survivals.append(projection.code_survival_probability)
        weighted_fidelities.append(projection.code_survival_probability * fidelity)
    return (
        float(np.mean(conditional_fidelities)),
        float(np.mean(survivals)),
        float(np.mean(weighted_fidelities)),
    )


def _cutoff_point(cutoff: int, delta: float, grid_points: int) -> SBSCutoffValidationPoint:
    simulator = SBSFockOneRoundSimulator(
        SBSFockCycleConfig(
            cutoff=cutoff,
            projector_delta=delta,
            grid_points=grid_points,
            readout_confusion=((1.0, 0.0), (0.0, 1.0)),
        )
    )
    conditional, survival, weighted = _average_metrics(simulator)
    diagnostics = [simulator.kraus_diagnostics(q) for q in ("X", "Z")]
    return SBSCutoffValidationPoint(
        cutoff=cutoff,
        average_conditional_logical_fidelity=conditional,
        average_code_survival=survival,
        average_code_weighted_fidelity=weighted,
        maximum_raw_code_completeness_error=max(
            item.raw_code_subspace_completeness_error for item in diagnostics
        ),
        maximum_completed_completeness_error=max(
            item.completed_completeness_frobenius_error for item in diagnostics
        ),
    )


def run_sbs_fock_cycle_validation(
    *,
    cutoff: int = 24,
    projector_delta: float = 0.34,
    grid_points: int = 4097,
    monte_carlo_samples: int = 100_000,
    seed: int = 2301,
) -> SBSFockCycleValidationResult:
    if not isinstance(monte_carlo_samples, (int, np.integer)) or int(monte_carlo_samples) < 10_000:
        raise ValueError("monte_carlo_samples must be an integer >= 10000")
    clean_config = SBSFockCycleConfig(
        cutoff=cutoff,
        projector_delta=projector_delta,
        grid_points=grid_points,
        readout_confusion=((1.0, 0.0), (0.0, 1.0)),
    )
    clean = SBSFockOneRoundSimulator(clean_config)
    clean_conditional, clean_survival, clean_weighted = _average_metrics(clean)
    noisy = SBSFockOneRoundSimulator(
        replace(
            clean_config,
            readout_confusion=((0.985, 0.015), (0.025, 0.975)),
            controller_residual_phase_by_observed=((0.0, 0.035), (0.0, -0.035)),
            idle=FockIdleConfig(
                displacement=0.025 + 0.018j,
                loss_transmissivity=0.992,
                thermal_rate_time=0.012,
                thermal_bath_occupation=0.08,
                phase_diffusion_variance=0.0015,
                kerr_strength=0.002,
                high_fock_proxy_probability=0.002,
            ),
        )
    )
    noisy_conditional, noisy_survival, noisy_weighted = _average_metrics(noisy)
    representative = noisy.run_exact_cycle(noisy.initialize("+i"))
    probabilities = np.asarray(
        [branch.probability for branch in representative.branches], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    counts = np.bincount(
        rng.choice(len(probabilities), size=int(monte_carlo_samples), p=probabilities),
        minlength=len(probabilities),
    )
    empirical = counts / float(monte_carlo_samples)
    standard_errors = np.sqrt(
        np.maximum(probabilities * (1.0 - probabilities), 1.0e-15)
        / float(monte_carlo_samples)
    )
    max_z = float(np.max(np.abs(empirical - probabilities) / standard_errors))

    base = clean.initialize("0")
    survival_gains = []
    for error in (clean.model.a, clean.model.adag):
        disturbed_matrix = error @ base.matrix @ error.conj().T
        disturbed_matrix /= float(np.trace(disturbed_matrix).real)
        disturbed = FiniteCutoffDensity(disturbed_matrix, cutoff)
        before = clean._code_survival(disturbed)
        after = clean.run_exact_cycle(disturbed).unconditional_projection.code_survival_probability
        survival_gains.append(after - before)
    photon_error_gain = float(np.mean(survival_gains))

    minimum_eigenvalue = min(
        float(np.min(np.linalg.eigvalsh(branch.final_state.matrix)))
        for branch in representative.branches
    )
    code_error = float(
        np.linalg.norm(clean.code_basis.isometry.conj().T @ clean.code_basis.isometry - _IDENTITY_2)
    )
    diagnostics = tuple(clean.kraus_diagnostics(q) for q in ("X", "Z"))
    sweep_cutoffs = tuple(sorted(set((18, 24, 30, 36, 42, int(cutoff)))))
    cutoff_sweep = tuple(
        _cutoff_point(item, projector_delta, grid_points) for item in sweep_cutoffs
    )
    final_survival_change = abs(
        cutoff_sweep[-1].average_code_survival
        - cutoff_sweep[-2].average_code_survival
    )
    final_conditional_change = abs(
        cutoff_sweep[-1].average_conditional_logical_fidelity
        - cutoff_sweep[-2].average_conditional_logical_fidelity
    )
    checks = {
        "canonical_coordinate_bridge_is_explicit": (
            clean.code_basis.source_coordinate_scale == PAPER_CANONICAL_SOURCE_SCALE
        ),
        "code_basis_orthonormal": code_error < 1.0e-10,
        "raw_truncation_defect_is_detected": all(
            item.raw_completeness_frobenius_error > 1.0e-3 for item in diagnostics
        ),
        "raw_code_subspace_defect_is_bounded": all(
            item.raw_code_subspace_completeness_error < 0.2 for item in diagnostics
        ),
        "finite_cutoff_completion_is_nonsingular": all(
            item.raw_gram_minimum_eigenvalue > 1.0e-6 for item in diagnostics
        ),
        "completed_kraus_pairs_are_trace_preserving": all(
            item.completed_completeness_frobenius_error < 1.0e-10
            for item in diagnostics
        ),
        "exact_sixteen_branch_hidden_observed_instrument": len(representative.branches) == 16,
        "exact_branch_probability_complete": abs(representative.total_probability - 1.0) < 1.0e-10,
        "all_branch_states_positive": minimum_eigenvalue > -1.0e-9,
        "monte_carlo_matches_exact_branches": max_z < 5.0,
        "clean_cycle_preserves_conditional_logical_information": clean_conditional > 0.99,
        "clean_cycle_retains_code_weighted_fidelity": clean_weighted > 0.75,
        "registered_noise_does_not_improve_code_weighted_fidelity": noisy_weighted <= clean_weighted,
        "photon_loss_and_gain_are_pumped_toward_code": photon_error_gain > 0.4,
        "cutoff_sweep_conditional_fidelity_converges": final_conditional_change < 1.0e-3,
        "cutoff_sweep_survival_is_stable": final_survival_change < 0.01,
    }
    return SBSFockCycleValidationResult(
        cutoff=cutoff,
        projector_delta=projector_delta,
        source_coordinate_scale=PAPER_CANONICAL_SOURCE_SCALE,
        code_raw_overlap_abs=float(abs(clean.code_basis.raw_overlap)),
        code_isometry_error=code_error,
        kraus_diagnostics=(diagnostics[0], diagnostics[1]),
        exact_branch_count=len(representative.branches),
        exact_probability_error=abs(representative.total_probability - 1.0),
        minimum_branch_eigenvalue=minimum_eigenvalue,
        monte_carlo_max_z_score=max_z,
        clean_average_conditional_logical_fidelity=clean_conditional,
        clean_average_code_survival=clean_survival,
        clean_average_code_weighted_fidelity=clean_weighted,
        noisy_average_conditional_logical_fidelity=noisy_conditional,
        noisy_average_code_survival=noisy_survival,
        noisy_average_code_weighted_fidelity=noisy_weighted,
        one_cycle_photon_error_code_survival_gain=photon_error_gain,
        cutoff_sweep=cutoff_sweep,
        checks=checks,
    )


def write_sbs_fock_cycle_validation(
    result: SBSFockCycleValidationResult, output: str | Path
) -> Path:
    if not isinstance(result, SBSFockCycleValidationResult):
        raise TypeError("result must be an SBSFockCycleValidationResult")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["passed"] = result.passed
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=24)
    parser.add_argument("--projector-delta", type=float, default=0.34)
    parser.add_argument("--grid-points", type=int, default=4097)
    parser.add_argument("--monte-carlo-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=2301)
    arguments = parser.parse_args()
    result = run_sbs_fock_cycle_validation(
        cutoff=arguments.cutoff,
        projector_delta=arguments.projector_delta,
        grid_points=arguments.grid_points,
        monte_carlo_samples=arguments.monte_carlo_samples,
        seed=arguments.seed,
    )
    write_sbs_fock_cycle_validation(result, arguments.output)
    print(json.dumps({"passed": result.passed, "checks": result.checks}, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "PAPER_CANONICAL_SOURCE_SCALE",
    "FOCK_SBS_CYCLE_SCOPE",
    "FockIdleConfig",
    "SBSFockCycleConfig",
    "FockCodeBasis",
    "SBSKrausDiagnostics",
    "FockLogicalProjection",
    "SBSFockObservedConstituent",
    "SBSFockTruthConstituent",
    "SBSFockCycleBranch",
    "SBSFockExactCycleResult",
    "SBSCutoffValidationPoint",
    "SBSFockCycleValidationResult",
    "SBSFockOneRoundSimulator",
    "logical_density",
    "run_sbs_fock_cycle_validation",
    "write_sbs_fock_cycle_validation",
]
