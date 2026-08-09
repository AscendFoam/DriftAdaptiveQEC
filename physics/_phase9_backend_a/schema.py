"""Public data contracts and validation helpers for Phase-9 backend A."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite, pi
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..phase9_twin_contract import (
    ActionWord, NominalAction, execute_representative_probe, representative_action_probes,
)


ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]

BACKEND_A_ID = "PHASE9-BACKEND-A-JOINT-FOCK-QUTRIT-GKSL-V1"
MAX_SUPPORTED_CUTOFF = 32
MAX_EXACT_CHOI_CUTOFF = 8


def _supported_cutoff_cap() -> int:
    # Audited high-cutoff adapters mutate this value on the legacy façade.
    facade = sys.modules.get("physics.phase9_backend_a")
    return int(getattr(facade, "MAX_SUPPORTED_CUTOFF", MAX_SUPPORTED_CUTOFF))


BACKEND_A_SCOPE = (
    "finite-cutoff oscillator x qutrit synthetic qualification backend; "
    "dimensionless Hamiltonian/GKSL/IQ/reset parameters; no device-calibrated, "
    "lifetime, break-even, hardware, official-Puviani, external-SOTA or rank claim"
)
DEFAULT_PARAMETER_PROVENANCE = (
    "SYNTHETIC_DIMENSIONLESS_PHASE9_BACKEND_A_QUALIFICATION_NOT_DEVICE_CALIBRATED"
)
ANCILLA_LEVELS = ("g", "e", "f")
DRIFT_FIELDS = (
    "drive_q", "drive_p", "readout_i", "readout_q", "leakage_detuning",
)


def _readonly(
    value: ArrayLike,
    *,
    dtype: np.dtype[Any] = np.dtype(np.complex128),
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _probability(value: object, name: str) -> float:
    result = _nonnegative(value, name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive(value: object, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _exact_int(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return result


def _validated_tuple(
    value: Iterable[object],
    name: str,
    *,
    length: int,
    validator: Any,
) -> tuple[float, ...]:
    try:
        result = tuple(validator(item, name) for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of length {length}") from exc
    if len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


def _density_diagnostics(matrix: ComplexMatrix) -> dict[str, float]:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    trace = complex(np.trace(matrix))
    eigenvalues = np.linalg.eigvalsh(hermitian)
    return {
        "trace_real": float(trace.real),
        "trace_imag": float(trace.imag),
        "hermiticity_frobenius": float(
            np.linalg.norm(matrix - matrix.conj().T, ord="fro")
        ),
        "minimum_eigenvalue": float(np.min(eigenvalues)),
        "maximum_eigenvalue": float(np.max(eigenvalues)),
        "purity": float(np.trace(hermitian @ hermitian).real),
    }


def _validated_density(
    value: ArrayLike,
    dimension: int,
    name: str,
    *,
    trace_tolerance: float = 2.0e-9,
    positivity_tolerance: float = 2.0e-9,
) -> ComplexMatrix:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    diagnostics = _density_diagnostics(matrix)
    if diagnostics["hermiticity_frobenius"] > 2.0e-9:
        raise ValueError(f"{name} must be Hermitian")
    if (
        abs(diagnostics["trace_real"] - 1.0) > trace_tolerance
        or abs(diagnostics["trace_imag"]) > trace_tolerance
    ):
        raise ValueError(f"{name} must have unit trace")
    if diagnostics["minimum_eigenvalue"] < -positivity_tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return _readonly(0.5 * (matrix + matrix.conj().T))


def _trace_distance(left: ComplexMatrix, right: ComplexMatrix) -> float:
    delta = 0.5 * ((left - right) + (left - right).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))


@dataclass(frozen=True)
class BackendAConfig:
    """Dimensionless, explicit parameters for backend A.

    Frequencies and rates are angular-frequency multiples in one synthetic
    round-time unit.  No value is presented as a calibrated device parameter.
    """

    cutoff: int = 8
    substeps_per_segment: int = 2
    action_duration: float = 0.06
    ramsey_pulse_duration: float = 0.03
    sense_duration: float = 0.08
    ramsey_angle: float = pi / 2.0
    action_displacement: float = 0.16
    dispersive_chi: float = 0.32
    self_kerr: float = 0.006
    oscillator_loss_rate: float = 0.012
    oscillator_dephasing_rate: float = 0.003
    ancilla_ge_relax_rate: float = 0.010
    ancilla_fe_relax_rate: float = 0.016
    ancilla_ge_excitation_rate: float = 0.001
    ancilla_dephasing_rate: float = 0.006
    pulse_leakage_crosstalk: float = 0.035
    measurement_leakage_coupling: float = 0.018
    action_leakage_coupling: float = 0.012
    iq_samples: int = 8
    iq_sigma: float = 0.48
    iq_centers: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ] = ((-0.85, 0.0), (0.85, 0.0), (0.0, 1.15))
    reset_success_e: float = 0.985
    reset_success_f: float = 0.82
    reset_ack_error: float = 0.002
    drift_retention: tuple[float, float, float, float, float] = (
        0.985,
        0.985,
        0.992,
        0.992,
        0.975,
    )
    drift_noise_std: tuple[float, float, float, float, float] = (
        0.0015,
        0.0015,
        0.0020,
        0.0020,
        0.0010,
    )
    drift_action_kick: float = 0.0025
    drift_readout_heating: float = 0.0018
    drift_leakage_heating: float = 0.0012
    reset_action_energy: float = 1.5
    leakage_age_threshold: float = 0.05
    logical_projector_delta: float = 0.34
    logical_grid_points: int = 2049
    parameter_provenance: str = DEFAULT_PARAMETER_PROVENANCE
    backend_id: str = BACKEND_A_ID
    scope: str = BACKEND_A_SCOPE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cutoff",
            _exact_int(
                self.cutoff,
                "cutoff",
                minimum=2,
                maximum=_supported_cutoff_cap(),
            ),
        )
        object.__setattr__(
            self,
            "substeps_per_segment",
            _exact_int(
                self.substeps_per_segment,
                "substeps_per_segment",
                minimum=1,
                maximum=128,
            ),
        )
        for name in (
            "action_duration",
            "ramsey_pulse_duration",
            "sense_duration",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self,
            "ramsey_angle",
            _finite(self.ramsey_angle, "ramsey_angle"),
        )
        object.__setattr__(
            self,
            "action_displacement",
            _nonnegative(self.action_displacement, "action_displacement"),
        )
        for name in (
            "dispersive_chi",
            "self_kerr",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in (
            "oscillator_loss_rate",
            "oscillator_dephasing_rate",
            "ancilla_ge_relax_rate",
            "ancilla_fe_relax_rate",
            "ancilla_ge_excitation_rate",
            "ancilla_dephasing_rate",
            "pulse_leakage_crosstalk",
            "measurement_leakage_coupling",
            "action_leakage_coupling",
            "drift_action_kick",
            "drift_readout_heating",
            "drift_leakage_heating",
            "reset_action_energy",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "iq_samples",
            _exact_int(self.iq_samples, "iq_samples", minimum=1, maximum=4096),
        )
        object.__setattr__(self, "iq_sigma", _positive(self.iq_sigma, "iq_sigma"))
        centers = np.asarray(self.iq_centers, dtype=np.float64)
        if centers.shape != (3, 2) or not np.all(np.isfinite(centers)):
            raise ValueError("iq_centers must have finite shape (3, 2)")
        object.__setattr__(
            self,
            "iq_centers",
            tuple(tuple(float(item) for item in row) for row in centers),
        )
        for name in (
            "reset_success_e",
            "reset_success_f",
            "reset_ack_error",
            "leakage_age_threshold",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "drift_retention",
            _validated_tuple(
                self.drift_retention,
                "drift_retention",
                length=5,
                validator=_probability,
            ),
        )
        object.__setattr__(
            self,
            "drift_noise_std",
            _validated_tuple(
                self.drift_noise_std,
                "drift_noise_std",
                length=5,
                validator=_nonnegative,
            ),
        )
        object.__setattr__(
            self,
            "logical_projector_delta",
            _positive(self.logical_projector_delta, "logical_projector_delta"),
        )
        object.__setattr__(
            self,
            "logical_grid_points",
            _exact_int(
                self.logical_grid_points,
                "logical_grid_points",
                minimum=1025,
                maximum=32769,
            ),
        )
        if self.logical_grid_points % 2 == 0:
            raise ValueError("logical_grid_points must be odd")
        if (
            self.action_duration == 0.0
            and (
                self.action_displacement > 0.0
                or self.action_leakage_coupling > 0.0
            )
        ):
            raise ValueError(
                "nonzero action pulse requires positive action_duration"
            )
        if (
            self.ramsey_pulse_duration == 0.0
            and self.ramsey_angle != 0.0
        ):
            raise ValueError(
                "nonzero Ramsey angle requires positive pulse duration"
            )
        if (
            not isinstance(self.parameter_provenance, str)
            or not self.parameter_provenance.strip()
        ):
            raise ValueError("parameter_provenance must be non-empty")
        if self.backend_id != BACKEND_A_ID:
            raise ValueError("backend_id must preserve the backend-A identity")
        if self.scope != BACKEND_A_SCOPE:
            raise ValueError("scope must preserve the fail-closed claim boundary")

    def semantic_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["iq_centers"] = [list(row) for row in self.iq_centers]
        data["drift_retention"] = list(self.drift_retention)
        data["drift_noise_std"] = list(self.drift_noise_std)
        return data

    def semantic_sha256(self) -> str:
        return sha256(
            json.dumps(
                self.semantic_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class BackendAQualificationThresholds:
    """Pre-outcome implementation tolerances for T9.2.2."""

    choi_minimum_eigenvalue: float = -2.0e-10
    choi_tp_frobenius: float = 2.0e-10
    choi_hermiticity_frobenius: float = 2.0e-10
    density_trace_error: float = 5.0e-9
    density_hermiticity_frobenius: float = 5.0e-9
    density_minimum_eigenvalue: float = -5.0e-9
    instrument_completeness_frobenius: float = 1.0e-12
    probability_sum_error: float = 2.0e-10
    zero_noise_idle_trace_distance: float = 5.0e-10
    ideal_action_trace_distance: float = 2.0e-5
    limit_population_minimum: float = 1.0 - 2.0e-10
    measurement_coherence_ratio: float = 1.0e-3
    measurement_posterior_peak: float = 0.999
    syndrome_state_dependence_minimum: float = 0.05
    syndrome_backaction_trace_distance_minimum: float = 0.01
    action_induced_f_population_minimum: float = 0.10
    action_state_trace_distance_minimum: float = 1.0e-3
    action_drift_l2_minimum: float = 1.0e-5
    different_seed_iq_difference_minimum: float = 1.0e-6
    step_size_trace_distance: float = 1.5e-4
    step_size_error_ratio: float = 0.30
    fock_cutoff_trace_distance: float = 2.0e-4
    six_state_initial_fidelity: float = 1.0 - 2.0e-9

    def __post_init__(self) -> None:
        nonnegative = (
            "choi_tp_frobenius",
            "choi_hermiticity_frobenius",
            "density_trace_error",
            "density_hermiticity_frobenius",
            "instrument_completeness_frobenius",
            "probability_sum_error",
            "zero_noise_idle_trace_distance",
            "ideal_action_trace_distance",
            "measurement_coherence_ratio",
            "syndrome_state_dependence_minimum",
            "syndrome_backaction_trace_distance_minimum",
            "action_induced_f_population_minimum",
            "action_state_trace_distance_minimum",
            "action_drift_l2_minimum",
            "different_seed_iq_difference_minimum",
            "step_size_trace_distance",
            "step_size_error_ratio",
            "fock_cutoff_trace_distance",
        )
        for name in nonnegative:
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        for name in (
            "limit_population_minimum",
            "measurement_posterior_peak",
            "six_state_initial_fidelity",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name),
            )
        for name in (
            "choi_minimum_eigenvalue",
            "density_minimum_eigenvalue",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
        if self.choi_minimum_eigenvalue > 0.0:
            raise ValueError("choi_minimum_eigenvalue tolerance must be <= 0")
        if self.density_minimum_eigenvalue > 0.0:
            raise ValueError("density_minimum_eigenvalue tolerance must be <= 0")

    def semantic_dict(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class BackendADriftState:
    drive_q: float = 0.0
    drive_p: float = 0.0
    readout_i: float = 0.0
    readout_q: float = 0.0
    leakage_detuning: float = 0.0

    def __post_init__(self) -> None:
        for name in DRIFT_FIELDS:
            object.__setattr__(self, name, _finite(getattr(self, name), name))

    def vector(self) -> RealVector:
        return _readonly(
            [getattr(self, name) for name in DRIFT_FIELDS],
            dtype=np.dtype(np.float64),
        )

    @classmethod
    def from_vector(cls, value: ArrayLike) -> "BackendADriftState":
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (5,) or not np.all(np.isfinite(vector)):
            raise ValueError("drift vector must be finite with shape (5,)")
        return cls(*[float(item) for item in vector])


@dataclass(frozen=True)
class BackendAState:
    joint_density: ComplexMatrix
    cutoff: int
    drift: BackendADriftState = BackendADriftState()
    leakage_age: int = 0
    round_index: int = 0

    def __post_init__(self) -> None:
        cutoff = _exact_int(
            self.cutoff,
            "cutoff",
            minimum=2,
            maximum=_supported_cutoff_cap(),
        )
        object.__setattr__(
            self,
            "joint_density",
            _validated_density(
                self.joint_density,
                cutoff * 3,
                "joint_density",
            ),
        )
        object.__setattr__(self, "cutoff", cutoff)
        if not isinstance(self.drift, BackendADriftState):
            raise TypeError("drift must be BackendADriftState")
        object.__setattr__(
            self,
            "leakage_age",
            _exact_int(
                self.leakage_age,
                "leakage_age",
                minimum=0,
                maximum=65535,
            ),
        )
        object.__setattr__(
            self,
            "round_index",
            _exact_int(
                self.round_index,
                "round_index",
                minimum=0,
                maximum=(1 << 63) - 1,
            ),
        )


@dataclass(frozen=True)
class BackendAExogenous:
    """One round of explicit exogenous randomness for CRN interventions."""

    emission_uniform: float
    iq_standard_i: tuple[float, ...]
    iq_standard_q: tuple[float, ...]
    reset_uniform: float
    reset_ack_uniform: float
    drift_standard: tuple[float, float, float, float, float]
    seed: int
    round_index: int

    def __post_init__(self) -> None:
        for name in (
            "emission_uniform",
            "reset_uniform",
            "reset_ack_uniform",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
            object.__setattr__(self, name, value)
        for name in ("iq_standard_i", "iq_standard_q"):
            values = tuple(_finite(item, name) for item in getattr(self, name))
            if not values:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, values)
        if len(self.iq_standard_i) != len(self.iq_standard_q):
            raise ValueError("I/Q exogenous arrays must have equal length")
        object.__setattr__(
            self,
            "drift_standard",
            tuple(_finite(item, "drift_standard") for item in self.drift_standard),
        )
        if len(self.drift_standard) != 5:
            raise ValueError("drift_standard must contain five values")
        object.__setattr__(
            self,
            "seed",
            _exact_int(self.seed, "seed", minimum=0, maximum=(1 << 63) - 1),
        )
        object.__setattr__(
            self,
            "round_index",
            _exact_int(
                self.round_index,
                "round_index",
                minimum=0,
                maximum=(1 << 63) - 1,
            ),
        )


def backend_a_exogenous(
    *,
    seed: int,
    round_index: int,
    iq_samples: int,
) -> BackendAExogenous:
    """Generate an addressable round-level random record.

    ``SeedSequence([seed, round_index, constant])`` makes a round replayable
    without consuming hidden mutable RNG state.  Counterfactual actions can
    therefore use exactly the same exogenous record.
    """

    seed_value = _exact_int(seed, "seed", minimum=0, maximum=(1 << 63) - 1)
    round_value = _exact_int(
        round_index,
        "round_index",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    samples = _exact_int(iq_samples, "iq_samples", minimum=1, maximum=4096)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed_value, round_value, 0xA921])
    )
    return BackendAExogenous(
        emission_uniform=float(rng.random()),
        iq_standard_i=tuple(float(item) for item in rng.standard_normal(samples)),
        iq_standard_q=tuple(float(item) for item in rng.standard_normal(samples)),
        reset_uniform=float(rng.random()),
        reset_ack_uniform=float(rng.random()),
        drift_standard=tuple(float(item) for item in rng.standard_normal(5)),
        seed=seed_value,
        round_index=round_value,
    )


@dataclass(frozen=True)
class BackendAObservation:
    """Analog synthetic observation before the T9.2.6 fixed-point frontend."""

    iq_i: RealVector
    iq_q: RealVector
    integrated_i: float
    integrated_q: float
    log_evidence_density: float
    posterior_levels: tuple[float, float, float]
    leakage_confidence_analog: float
    reset_ack: str
    source: str = "synthetic_backend_a_analog_pre_frontend"

    def __post_init__(self) -> None:
        i_values = np.asarray(self.iq_i, dtype=np.float64)
        q_values = np.asarray(self.iq_q, dtype=np.float64)
        if (
            i_values.ndim != 1
            or q_values.shape != i_values.shape
            or i_values.size == 0
            or not np.all(np.isfinite(i_values))
            or not np.all(np.isfinite(q_values))
        ):
            raise ValueError("iq_i/iq_q must be equal non-empty finite vectors")
        object.__setattr__(
            self,
            "iq_i",
            _readonly(i_values, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "iq_q",
            _readonly(q_values, dtype=np.dtype(np.float64)),
        )
        for name in ("integrated_i", "integrated_q", "log_evidence_density"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        posterior = tuple(
            _probability(item, "posterior_levels")
            for item in self.posterior_levels
        )
        if len(posterior) != 3 or abs(sum(posterior) - 1.0) > 2.0e-10:
            raise ValueError("posterior_levels must be a normalized triple")
        object.__setattr__(self, "posterior_levels", posterior)
        leakage = _probability(
            self.leakage_confidence_analog,
            "leakage_confidence_analog",
        )
        if abs(leakage - posterior[2]) > 2.0e-12:
            raise ValueError("leakage confidence must equal the f posterior")
        object.__setattr__(self, "leakage_confidence_analog", leakage)
        if self.reset_ack not in {"none", "success", "failure"}:
            raise ValueError("reset_ack must be none/success/failure")
        if self.source != "synthetic_backend_a_analog_pre_frontend":
            raise ValueError("backend A cannot relabel analog IQ as recorded/live")


@dataclass(frozen=True)
class BackendATruthRecord:
    """Non-deployable physics truth for validation and backend-B comparison."""

    sampled_emission_level: str
    pre_measurement_level_probabilities: tuple[float, float, float]
    post_measurement_level_probabilities: tuple[float, float, float]
    reset_hidden_outcome: str
    pre_reset_level_probabilities: tuple[float, float, float]
    post_reset_level_probabilities: tuple[float, float, float]
    action_code: str
    action_alpha_real: float
    action_alpha_imag: float
    drift_before: tuple[float, float, float, float, float]
    drift_after: tuple[float, float, float, float, float]
    density_diagnostics: Mapping[str, float]
    namespace: str = "BACKEND_LATENT_AND_EVALUATOR_TRUTH_NOT_DEPLOYABLE"

    def __post_init__(self) -> None:
        if self.sampled_emission_level not in ANCILLA_LEVELS:
            raise ValueError("sampled_emission_level must be g/e/f")
        if self.reset_hidden_outcome not in {"none", "success", "failure"}:
            raise ValueError("invalid reset hidden outcome")
        if self.action_code not in {item.name for item in NominalAction}:
            raise ValueError("invalid action code")
        for name in (
            "pre_measurement_level_probabilities",
            "post_measurement_level_probabilities",
            "pre_reset_level_probabilities",
            "post_reset_level_probabilities",
        ):
            probabilities = tuple(
                _probability(item, name) for item in getattr(self, name)
            )
            if len(probabilities) != 3 or abs(sum(probabilities) - 1.0) > 2.0e-9:
                raise ValueError(f"{name} must be a normalized triple")
            object.__setattr__(self, name, probabilities)
        for name in ("action_alpha_real", "action_alpha_imag"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in ("drift_before", "drift_after"):
            values = tuple(_finite(item, name) for item in getattr(self, name))
            if len(values) != 5:
                raise ValueError(f"{name} must contain five values")
            object.__setattr__(self, name, values)
        diagnostics = dict(self.density_diagnostics)
        required = {
            "trace_real",
            "trace_imag",
            "hermiticity_frobenius",
            "minimum_eigenvalue",
            "maximum_eigenvalue",
            "purity",
        }
        if set(diagnostics) != required:
            raise ValueError("density diagnostics schema mismatch")
        object.__setattr__(self, "density_diagnostics", diagnostics)


@dataclass(frozen=True)
class BackendAEvaluatorState:
    target_label: str
    target_density: ComplexMatrix
    pauli_x: int = 0
    pauli_z: int = 0

    def __post_init__(self) -> None:
        if self.target_label not in {"0", "1", "+", "-", "+i", "-i", "mixed"}:
            raise ValueError("target_label must be a six-state label or mixed")
        object.__setattr__(
            self,
            "target_density",
            _validated_density(self.target_density, 2, "target_density"),
        )
        object.__setattr__(
            self,
            "pauli_x",
            _exact_int(self.pauli_x, "pauli_x", minimum=0, maximum=1),
        )
        object.__setattr__(
            self,
            "pauli_z",
            _exact_int(self.pauli_z, "pauli_z", minimum=0, maximum=1),
        )

    def after(self, action: ActionWord) -> "BackendAEvaluatorState":
        return replace(
            self,
            pauli_x=self.pauli_x ^ int(action.pauli_dx),
            pauli_z=self.pauli_z ^ int(action.pauli_dz),
        )


@dataclass(frozen=True)
class BackendALogicalRecord:
    code_survival_probability: float
    raw_logical_density: ComplexMatrix
    frame_corrected_logical_density: ComplexMatrix
    bloch_xyz: tuple[float, float, float]
    target_fidelity: float
    logical_error: bool
    evaluator_state: BackendAEvaluatorState
    namespace: str = "EVALUATOR_TRUTH_NOT_DEPLOYABLE"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "code_survival_probability",
            _probability(
                self.code_survival_probability,
                "code_survival_probability",
            ),
        )
        object.__setattr__(
            self,
            "raw_logical_density",
            _validated_density(
                self.raw_logical_density,
                2,
                "raw_logical_density",
            ),
        )
        object.__setattr__(
            self,
            "frame_corrected_logical_density",
            _validated_density(
                self.frame_corrected_logical_density,
                2,
                "frame_corrected_logical_density",
            ),
        )
        bloch = tuple(_finite(item, "bloch_xyz") for item in self.bloch_xyz)
        if len(bloch) != 3 or any(abs(item) > 1.0 + 2.0e-9 for item in bloch):
            raise ValueError("bloch_xyz must be a physical three-vector")
        object.__setattr__(self, "bloch_xyz", bloch)
        object.__setattr__(
            self,
            "target_fidelity",
            _probability(self.target_fidelity, "target_fidelity"),
        )
        if type(self.logical_error) is not bool:
            raise TypeError("logical_error must be an exact bool")
        if not isinstance(self.evaluator_state, BackendAEvaluatorState):
            raise TypeError("evaluator_state must be BackendAEvaluatorState")


@dataclass(frozen=True)
class BackendARoundResult:
    state: BackendAState
    observation: BackendAObservation
    truth: BackendATruthRecord
    logical: BackendALogicalRecord | None
    action_word: ActionWord
    exogenous: BackendAExogenous

    def __post_init__(self) -> None:
        if not isinstance(self.state, BackendAState):
            raise TypeError("state must be BackendAState")
        if not isinstance(self.observation, BackendAObservation):
            raise TypeError("observation must be BackendAObservation")
        if not isinstance(self.truth, BackendATruthRecord):
            raise TypeError("truth must be BackendATruthRecord")
        if self.logical is not None and not isinstance(
            self.logical,
            BackendALogicalRecord,
        ):
            raise TypeError("logical must be BackendALogicalRecord or None")
        if not isinstance(self.action_word, ActionWord):
            raise TypeError("action_word must be ActionWord")
        if not isinstance(self.exogenous, BackendAExogenous):
            raise TypeError("exogenous must be BackendAExogenous")


@dataclass(frozen=True)
class BackendATrajectory:
    rounds: tuple[BackendARoundResult, ...]
    initial_state: BackendAState
    final_state: BackendAState
    seed: int
    backend_id: str = BACKEND_A_ID


@dataclass(frozen=True)
class ChannelDiagnostics:
    dimension: int
    choi_minimum_eigenvalue: float
    choi_trace: float
    trace_preservation_frobenius: float
    hermiticity_frobenius: float

    @property
    def cp(self) -> bool:
        return self.choi_minimum_eigenvalue >= -2.0e-10

    @property
    def tp(self) -> bool:
        return self.trace_preservation_frobenius <= 2.0e-10


@dataclass(frozen=True)
class BackendAQualification:
    backend_id: str
    scope: str
    config_sha256: str
    metrics: Mapping[str, float | int | str | bool]
    checks: Mapping[str, bool]
    claim_state: Mapping[str, object]
    verdict: str

    @property
    def passed(self) -> bool:
        return (
            self.verdict == "QUALIFIED_BACKEND_A_ONLY"
            and bool(self.checks)
            and all(self.checks.values())
            and all(value is None for value in self.claim_state.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "scope": self.scope,
            "config_sha256": self.config_sha256,
            "metrics": dict(self.metrics),
            "checks": dict(self.checks),
            "claim_state": dict(self.claim_state),
            "verdict": self.verdict,
            "passed": self.passed,
        }


@lru_cache(maxsize=64)
def diagnostic_action_word(action_name: str) -> ActionWord:
    """Return a real T9.2.1 probe receipt for a primitive action."""

    if action_name not in {item.name for item in NominalAction if item != NominalAction.INVALID}:
        raise ValueError("unknown diagnostic action name")
    for probe in representative_action_probes():
        if probe.expected_terminal == action_name:
            return execute_representative_probe(probe)[-1].recurrence.action_word
    raise ValueError(f"no T9.2.1 representative probe emits {action_name}")
