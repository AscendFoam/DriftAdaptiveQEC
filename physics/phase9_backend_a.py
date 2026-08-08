"""Phase-9 backend A: joint finite-Fock oscillator and qutrit dynamics.

This module is the first exact physics backend for the Phase-9 digital twin.  It
does not add noise to a logical label.  Instead, every round evolves a joint
``cutoff x {g,e,f}`` density matrix through

* a time-dependent recovery Hamiltonian selected by the frozen T9.2.1
  :class:`~physics.phase9_twin_contract.ActionWord`;
* a GKSL master equation with oscillator loss/dephasing and qutrit relaxation,
  excitation and dephasing;
* a Ramsey-like qutrit syndrome interaction;
* a continuous-IQ diagonal Kraus instrument, whose likelihood produces the
  measurement backaction;
* a conditional reset instrument whose failed branch preserves ``e/f``;
* an action-conditioned latent drift recurrence.

The logical state and logical error are reconstructed only in the evaluator
namespace.  They never enter the state transition, IQ sampler, reset sampler or
action selection.  Parameters are dimensionless synthetic qualification
parameters unless an explicit future device-calibration provenance replaces
them.  Consequently this backend can qualify a simulator implementation but
cannot establish device fidelity, lifetime, break-even, FPGA or SOTA claims.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import json
from math import exp, isfinite, log, pi, sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm
from scipy.sparse import csr_matrix, eye as sparse_eye, kron as sparse_kron
from scipy.sparse.linalg import expm_multiply

from .fock_density_model import FiniteCutoffDensity, FiniteCutoffFockModel
from .fock_sbs_cycle import (
    SBSFockCycleConfig,
    SBSFockOneRoundSimulator,
    logical_density,
)
from .phase9_twin_contract import (
    ActionWord,
    NominalAction,
    execute_representative_probe,
    representative_action_probes,
)


ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]

BACKEND_A_ID = "PHASE9-BACKEND-A-JOINT-FOCK-QUTRIT-GKSL-V1"
MAX_SUPPORTED_CUTOFF = 32
MAX_EXACT_CHOI_CUTOFF = 8
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
    "drive_q",
    "drive_p",
    "readout_i",
    "readout_q",
    "leakage_detuning",
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
                maximum=MAX_SUPPORTED_CUTOFF,
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
            maximum=MAX_SUPPORTED_CUTOFF,
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


class Phase9BackendASimulator:
    """Joint oscillator-qutrit trajectory simulator."""

    def __init__(self, config: BackendAConfig) -> None:
        if not isinstance(config, BackendAConfig):
            raise TypeError("config must be BackendAConfig")
        self.config = config
        self.cutoff = config.cutoff
        self.ancilla_dimension = 3
        self.dimension = self.cutoff * self.ancilla_dimension
        self.oscillator = FiniteCutoffFockModel(self.cutoff)

        self.i_o = self.oscillator.identity
        self.i_a = np.eye(3, dtype=np.complex128)
        self.i_joint = np.eye(self.dimension, dtype=np.complex128)
        self.a = self.oscillator.a
        self.adag = self.oscillator.adag
        self.number = self.oscillator.number
        self.q = (self.a + self.adag) / sqrt(2.0)
        self.p = 1.0j * (self.adag - self.a) / sqrt(2.0)

        self.level_kets = tuple(
            np.eye(3, dtype=np.complex128)[:, index]
            for index in range(3)
        )
        self.level_projectors = tuple(
            np.outer(ket, ket.conj()) for ket in self.level_kets
        )
        g, e, f = self.level_kets
        self.sigma_ge = np.outer(g, e.conj())
        self.sigma_eg = self.sigma_ge.conj().T
        self.sigma_ef = np.outer(e, f.conj())
        self.sigma_fe = self.sigma_ef.conj().T
        self.y_ge = -1.0j * self.sigma_ge + 1.0j * self.sigma_eg
        self.x_ef = self.sigma_ef + self.sigma_fe

        self._joint_a = self._tensor(self.a, self.i_a)
        self._joint_number = self._tensor(self.number, self.i_a)
        self._joint_q = self._tensor(self.q, self.i_a)
        self._joint_p = self._tensor(self.p, self.i_a)
        self._joint_level_projectors = tuple(
            self._tensor(self.i_o, projector)
            for projector in self.level_projectors
        )
        self._joint_y_ge = self._tensor(self.i_o, self.y_ge)
        self._joint_x_ef = self._tensor(self.i_o, self.x_ef)

        self._collapse_operators = self._build_collapse_operators()
        self._dissipator = self._build_dissipator(self._collapse_operators)
        self._logical_simulator: SBSFockOneRoundSimulator | None = None

    @staticmethod
    def _tensor(left: ComplexMatrix, right: ComplexMatrix) -> ComplexMatrix:
        return np.kron(left, right)

    def _build_collapse_operators(self) -> tuple[ComplexMatrix, ...]:
        config = self.config
        rows: list[ComplexMatrix] = []

        def add(rate: float, operator: ComplexMatrix) -> None:
            if rate > 0.0:
                rows.append(sqrt(rate) * operator)

        add(config.oscillator_loss_rate, self._joint_a)
        add(config.oscillator_dephasing_rate, self._joint_number)
        add(
            config.ancilla_ge_relax_rate,
            self._tensor(self.i_o, self.sigma_ge),
        )
        add(
            config.ancilla_fe_relax_rate,
            self._tensor(self.i_o, self.sigma_ef),
        )
        add(
            config.ancilla_ge_excitation_rate,
            self._tensor(self.i_o, self.sigma_eg),
        )
        ancilla_phase = (
            -self.level_projectors[0]
            + self.level_projectors[1]
            + 2.0 * self.level_projectors[2]
        )
        add(
            config.ancilla_dephasing_rate,
            self._tensor(self.i_o, ancilla_phase),
        )
        return tuple(_readonly(operator) for operator in rows)

    def _build_dissipator(
        self,
        collapse_operators: Sequence[ComplexMatrix],
    ) -> csr_matrix:
        dimension = self.dimension
        identity = sparse_eye(dimension, dtype=np.complex128, format="csr")
        dissipator = csr_matrix(
            (dimension * dimension, dimension * dimension),
            dtype=np.complex128,
        )
        for operator in collapse_operators:
            collapse = csr_matrix(operator)
            gram = collapse.getH() @ collapse
            dissipator = (
                dissipator
                + sparse_kron(
                    collapse.conjugate(),
                    collapse,
                    format="csr",
                )
                - 0.5 * sparse_kron(identity, gram, format="csr")
                - 0.5
                * sparse_kron(gram.transpose(), identity, format="csr")
            )
        return dissipator.tocsr()

    def liouvillian(self, hamiltonian: ArrayLike) -> csr_matrix:
        h = np.asarray(hamiltonian, dtype=np.complex128)
        if h.shape != (self.dimension, self.dimension):
            raise ValueError("hamiltonian shape mismatch")
        if not np.all(np.isfinite(h)):
            raise ValueError("hamiltonian must be finite")
        if np.linalg.norm(h - h.conj().T, ord="fro") > 1.0e-10:
            raise ValueError("hamiltonian must be Hermitian")
        identity = sparse_eye(
            self.dimension,
            dtype=np.complex128,
            format="csr",
        )
        h_sparse = csr_matrix(h)
        commutator = -1.0j * (
            sparse_kron(identity, h_sparse, format="csr")
            - sparse_kron(h_sparse.transpose(), identity, format="csr")
        )
        return (commutator + self._dissipator).tocsr()

    def _base_hamiltonian(self, drift: BackendADriftState) -> ComplexMatrix:
        ancilla_dispersion = (
            self.level_projectors[1] + 2.0 * self.level_projectors[2]
        )
        kerr_operator = self.number @ (self.number - self.i_o)
        return _readonly(
            self.config.self_kerr * self._tensor(kerr_operator, self.i_a)
            + self.config.dispersive_chi
            * self._tensor(self.number, ancilla_dispersion)
            + drift.drive_q * self._joint_q
            + drift.drive_p * self._joint_p
            + drift.leakage_detuning * self._joint_level_projectors[2]
        )

    @staticmethod
    def _pulse_envelope(fraction: float) -> float:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("pulse fraction must lie in [0,1]")
        # Integral over [0,1] equals one.
        return 0.5 * pi * np.sin(pi * fraction)

    def _action_alpha(self, action: ActionWord) -> complex:
        if not isinstance(action, ActionWord):
            raise TypeError("action must be ActionWord")
        amplitude = self.config.action_displacement / sqrt(2.0)
        return complex(
            amplitude * int(action.pauli_dx),
            amplitude * int(action.pauli_dz),
        )

    def _action_energy(self, action: ActionWord) -> float:
        code = NominalAction(action.action_code)
        if code == NominalAction.RESET:
            return self.config.reset_action_energy
        if code == NominalAction.XZ:
            return 2.0
        if code in {NominalAction.X, NominalAction.Z}:
            return 1.0
        if code == NominalAction.HOLD:
            return 0.05
        if code == NominalAction.LKG_HOLD:
            return 0.02
        return 0.0

    def _evolve_segment(
        self,
        density: ComplexMatrix,
        duration: float,
        hamiltonian_at_fraction: Any,
    ) -> ComplexMatrix:
        if duration == 0.0:
            return _readonly(density)
        substeps = self.config.substeps_per_segment
        step = duration / substeps
        vector = np.asarray(density, dtype=np.complex128).reshape(
            self.dimension * self.dimension,
            order="F",
        )
        for index in range(substeps):
            midpoint = (index + 0.5) / substeps
            hamiltonian = np.asarray(
                hamiltonian_at_fraction(midpoint),
                dtype=np.complex128,
            )
            generator = self.liouvillian(hamiltonian)
            vector = expm_multiply(generator * step, vector)
        matrix = vector.reshape(
            (self.dimension, self.dimension),
            order="F",
        )
        matrix = 0.5 * (matrix + matrix.conj().T)
        trace = complex(np.trace(matrix))
        if abs(trace.imag) > 2.0e-9 or trace.real <= 0.0:
            raise RuntimeError("GKSL propagation produced an invalid trace")
        if abs(trace.real - 1.0) > 5.0e-8:
            raise RuntimeError(
                "GKSL propagation violated trace preservation before cleanup"
            )
        matrix = matrix / trace.real
        return _validated_density(
            matrix,
            self.dimension,
            "evolved_density",
            trace_tolerance=5.0e-9,
            positivity_tolerance=5.0e-9,
        )

    def _apply_action(
        self,
        density: ComplexMatrix,
        drift: BackendADriftState,
        action: ActionWord,
    ) -> ComplexMatrix:
        base = self._base_hamiltonian(drift)
        alpha = self._action_alpha(action)
        duration = self.config.action_duration
        drive = np.zeros_like(base)
        if duration > 0.0 and alpha != 0.0j:
            drive = (
                1.0j
                * (
                    alpha * self._tensor(self.adag, self.i_a)
                    - alpha.conjugate() * self._tensor(self.a, self.i_a)
                )
                / duration
            )
        leakage = np.zeros_like(base)
        if duration > 0.0:
            leakage = (
                self.config.action_leakage_coupling
                * self._action_energy(action)
                * self._joint_x_ef
                / duration
            )
        return self._evolve_segment(
            density,
            duration,
            lambda fraction: base
            + self._pulse_envelope(fraction) * (drive + leakage),
        )

    def _ramsey_pulse(
        self,
        density: ComplexMatrix,
        drift: BackendADriftState,
        angle: float,
    ) -> ComplexMatrix:
        duration = self.config.ramsey_pulse_duration
        if duration == 0.0 or angle == 0.0:
            return _readonly(density)
        base = self._base_hamiltonian(drift)
        ge_drive = angle * self._joint_y_ge / (2.0 * duration)
        leakage_drive = (
            abs(angle)
            * self.config.pulse_leakage_crosstalk
            * self._joint_x_ef
            / (2.0 * duration)
        )
        return self._evolve_segment(
            density,
            duration,
            lambda fraction: base
            + self._pulse_envelope(fraction)
            * (ge_drive + leakage_drive),
        )

    def _sense(
        self,
        density: ComplexMatrix,
        drift: BackendADriftState,
    ) -> ComplexMatrix:
        base = self._base_hamiltonian(drift)
        interaction = (
            self.config.measurement_leakage_coupling * self._joint_x_ef
        )
        return self._evolve_segment(
            density,
            self.config.sense_duration,
            lambda _fraction: base + interaction,
        )

    def ancilla_density(self, density: ArrayLike) -> ComplexMatrix:
        matrix = np.asarray(density, dtype=np.complex128)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("joint density shape mismatch")
        tensor = matrix.reshape(
            self.cutoff,
            3,
            self.cutoff,
            3,
        )
        result = np.trace(tensor, axis1=0, axis2=2)
        return _readonly(0.5 * (result + result.conj().T))

    def oscillator_density(self, density: ArrayLike) -> FiniteCutoffDensity:
        matrix = np.asarray(density, dtype=np.complex128)
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("joint density shape mismatch")
        tensor = matrix.reshape(
            self.cutoff,
            3,
            self.cutoff,
            3,
        )
        result = np.trace(tensor, axis1=1, axis2=3)
        result = 0.5 * (result + result.conj().T)
        result /= float(np.trace(result).real)
        return FiniteCutoffDensity(result, self.cutoff)

    def level_probabilities(
        self,
        density: ArrayLike,
    ) -> tuple[float, float, float]:
        ancilla = self.ancilla_density(density)
        probabilities = np.real(np.diag(ancilla))
        probabilities = np.maximum(probabilities, 0.0)
        probabilities /= np.sum(probabilities)
        return tuple(float(item) for item in probabilities)

    @staticmethod
    def _sample_categorical(
        probabilities: Sequence[float],
        uniform: float,
    ) -> int:
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += float(probability)
            if uniform < cumulative or index == len(probabilities) - 1:
                return index
        raise AssertionError("categorical sampler failed")

    def _measure_iq(
        self,
        density: ComplexMatrix,
        drift: BackendADriftState,
        exogenous: BackendAExogenous,
    ) -> tuple[
        ComplexMatrix,
        BackendAObservation,
        str,
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        if len(exogenous.iq_standard_i) != self.config.iq_samples:
            raise ValueError("exogenous IQ sample count does not match config")
        pre_probabilities = self.level_probabilities(density)
        sampled_index = self._sample_categorical(
            pre_probabilities,
            exogenous.emission_uniform,
        )
        centers = np.asarray(self.config.iq_centers, dtype=np.float64).copy()
        centers[:, 0] += drift.readout_i
        centers[:, 1] += drift.readout_q
        sigma = self.config.iq_sigma
        iq_i = (
            centers[sampled_index, 0]
            + sigma * np.asarray(exogenous.iq_standard_i, dtype=np.float64)
        )
        iq_q = (
            centers[sampled_index, 1]
            + sigma * np.asarray(exogenous.iq_standard_q, dtype=np.float64)
        )
        squared = (
            (iq_i[None, :] - centers[:, 0, None]) ** 2
            + (iq_q[None, :] - centers[:, 1, None]) ** 2
        )
        log_likelihood = (
            -0.5 * np.sum(squared, axis=1) / (sigma * sigma)
            - self.config.iq_samples * log(2.0 * pi * sigma * sigma)
        )
        maximum = float(np.max(log_likelihood))
        amplitudes = np.exp(0.5 * (log_likelihood - maximum))
        ancilla_kraus = np.diag(amplitudes.astype(np.complex128))
        kraus = self._tensor(self.i_o, ancilla_kraus)
        unnormalized = kraus @ density @ kraus.conj().T
        scaled_evidence = float(np.trace(unnormalized).real)
        if not isfinite(scaled_evidence) or scaled_evidence <= 0.0:
            raise RuntimeError("IQ measurement produced zero/invalid evidence")
        post = unnormalized / scaled_evidence
        post = _validated_density(
            post,
            self.dimension,
            "post_measurement_density",
            positivity_tolerance=5.0e-9,
        )
        posterior = self.level_probabilities(post)
        observation = BackendAObservation(
            iq_i=iq_i,
            iq_q=iq_q,
            integrated_i=float(np.mean(iq_i)),
            integrated_q=float(np.mean(iq_q)),
            log_evidence_density=maximum + log(scaled_evidence),
            posterior_levels=posterior,
            leakage_confidence_analog=posterior[2],
            reset_ack="none",
        )
        return (
            post,
            observation,
            ANCILLA_LEVELS[sampled_index],
            pre_probabilities,
            posterior,
        )

    def measurement_completeness_error(self) -> float:
        # Each level-conditioned 2D Gaussian integrates to one.  The remaining
        # operator identity is computed rather than asserted.
        completeness = sum(self._joint_level_projectors)
        return float(
            np.linalg.norm(completeness - self.i_joint, ord="fro")
        )

    def reset_kraus(self) -> dict[str, tuple[ComplexMatrix, ...]]:
        g, e, f = self.level_kets
        p_e = self.config.reset_success_e
        p_f = self.config.reset_success_f
        success = (
            self._tensor(self.i_o, np.outer(g, g.conj())),
            sqrt(p_e) * self._tensor(self.i_o, np.outer(g, e.conj())),
            sqrt(p_f) * self._tensor(self.i_o, np.outer(g, f.conj())),
        )
        failure_operator = (
            sqrt(1.0 - p_e) * np.outer(e, e.conj())
            + sqrt(1.0 - p_f) * np.outer(f, f.conj())
        )
        failure = (self._tensor(self.i_o, failure_operator),)
        return {
            "success": tuple(_readonly(item) for item in success),
            "failure": tuple(_readonly(item) for item in failure),
        }

    def reset_completeness_error(self) -> float:
        gram = np.zeros_like(self.i_joint)
        for group in self.reset_kraus().values():
            for operator in group:
                gram += operator.conj().T @ operator
        return float(np.linalg.norm(gram - self.i_joint, ord="fro"))

    def _reset(
        self,
        density: ComplexMatrix,
        exogenous: BackendAExogenous,
    ) -> tuple[ComplexMatrix, str, str]:
        branches: dict[str, ComplexMatrix] = {}
        probabilities: dict[str, float] = {}
        for outcome, operators in self.reset_kraus().items():
            branch = sum(
                operator @ density @ operator.conj().T
                for operator in operators
            )
            probability = float(np.trace(branch).real)
            branches[outcome] = branch
            probabilities[outcome] = max(probability, 0.0)
        total = sum(probabilities.values())
        if abs(total - 1.0) > 2.0e-9:
            raise RuntimeError("reset instrument probabilities do not sum to one")
        hidden = (
            "success"
            if exogenous.reset_uniform < probabilities["success"]
            else "failure"
        )
        probability = probabilities[hidden]
        if probability <= 0.0:
            raise RuntimeError("selected zero-probability reset branch")
        post = _validated_density(
            branches[hidden] / probability,
            self.dimension,
            "post_reset_density",
            positivity_tolerance=5.0e-9,
        )
        observed = hidden
        if exogenous.reset_ack_uniform < self.config.reset_ack_error:
            observed = "failure" if hidden == "success" else "success"
        return post, hidden, observed

    def _update_drift(
        self,
        drift: BackendADriftState,
        action: ActionWord,
        exogenous: BackendAExogenous,
    ) -> BackendADriftState:
        before = drift.vector()
        retention = np.asarray(
            self.config.drift_retention,
            dtype=np.float64,
        )
        noise = np.asarray(
            self.config.drift_noise_std,
            dtype=np.float64,
        ) * np.asarray(exogenous.drift_standard, dtype=np.float64)
        energy = self._action_energy(action)
        kick = np.array(
            [
                self.config.drift_action_kick * int(action.pauli_dx),
                self.config.drift_action_kick * int(action.pauli_dz),
                self.config.drift_readout_heating * energy,
                -0.5 * self.config.drift_readout_heating * energy,
                self.config.drift_leakage_heating * energy,
            ],
            dtype=np.float64,
        )
        return BackendADriftState.from_vector(retention * before + kick + noise)

    def _logical_engine(self) -> SBSFockOneRoundSimulator:
        if self.cutoff < 8:
            raise ValueError("logical GKP projection requires cutoff >= 8")
        if self._logical_simulator is None:
            self._logical_simulator = SBSFockOneRoundSimulator(
                SBSFockCycleConfig(
                    cutoff=self.cutoff,
                    projector_delta=self.config.logical_projector_delta,
                    grid_points=self.config.logical_grid_points,
                )
            )
        return self._logical_simulator

    def initialize_logical(
        self,
        label: str,
        *,
        ancilla_level: str = "g",
        drift: BackendADriftState | None = None,
    ) -> tuple[BackendAState, BackendAEvaluatorState]:
        if ancilla_level not in ANCILLA_LEVELS:
            raise ValueError("ancilla_level must be g/e/f")
        logical_engine = self._logical_engine()
        oscillator_state = logical_engine.initialize(label)
        ancilla = self.level_projectors[ANCILLA_LEVELS.index(ancilla_level)]
        state = BackendAState(
            joint_density=self._tensor(oscillator_state.matrix, ancilla),
            cutoff=self.cutoff,
            drift=BackendADriftState() if drift is None else drift,
        )
        evaluator = BackendAEvaluatorState(
            target_label=label,
            target_density=logical_density(label),
        )
        return state, evaluator

    def initialize_fock(
        self,
        *,
        oscillator_ket: ArrayLike | None = None,
        ancilla_state: str | ArrayLike = "g",
        drift: BackendADriftState | None = None,
    ) -> BackendAState:
        if oscillator_ket is None:
            ket = np.zeros(self.cutoff, dtype=np.complex128)
            ket[0] = 1.0
        else:
            ket = np.asarray(oscillator_ket, dtype=np.complex128)
        if ket.shape != (self.cutoff,) or not np.all(np.isfinite(ket)):
            raise ValueError("oscillator_ket must be finite with shape (cutoff,)")
        norm = float(np.vdot(ket, ket).real)
        if norm <= 0.0:
            raise ValueError("oscillator_ket must have nonzero norm")
        ket = ket / sqrt(norm)
        oscillator = np.outer(ket, ket.conj())
        if isinstance(ancilla_state, str):
            if ancilla_state not in ANCILLA_LEVELS:
                raise ValueError("ancilla_state must be g/e/f")
            ancilla = self.level_projectors[
                ANCILLA_LEVELS.index(ancilla_state)
            ]
        else:
            value = np.asarray(ancilla_state, dtype=np.complex128)
            if value.shape == (3,):
                value = value / sqrt(float(np.vdot(value, value).real))
                ancilla = np.outer(value, value.conj())
            else:
                ancilla = _validated_density(value, 3, "ancilla_state")
        return BackendAState(
            joint_density=self._tensor(oscillator, ancilla),
            cutoff=self.cutoff,
            drift=BackendADriftState() if drift is None else drift,
        )

    def logical_record(
        self,
        state: BackendAState,
        evaluator: BackendAEvaluatorState,
    ) -> BackendALogicalRecord:
        if state.cutoff != self.cutoff:
            raise ValueError("state cutoff mismatch")
        if not isinstance(evaluator, BackendAEvaluatorState):
            raise TypeError("evaluator must be BackendAEvaluatorState")
        oscillator = self.oscillator_density(state.joint_density)
        engine = self._logical_engine()
        encoded = (
            engine.code_basis.isometry.conj().T
            @ oscillator.matrix
            @ engine.code_basis.isometry
        )
        survival = float(np.trace(encoded).real)
        if survival <= 1.0e-12:
            raise RuntimeError("logical code survival is numerically zero")
        raw = 0.5 * (encoded / survival + (encoded / survival).conj().T)
        pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        frame = np.eye(2, dtype=np.complex128)
        if evaluator.pauli_x:
            frame = pauli_x @ frame
        if evaluator.pauli_z:
            frame = pauli_z @ frame
        corrected = frame.conj().T @ raw @ frame
        pauli_y = np.array(
            [[0.0, -1.0j], [1.0j, 0.0]],
            dtype=np.complex128,
        )
        bloch = tuple(
            float(np.trace(corrected @ operator).real)
            for operator in (pauli_x, pauli_y, pauli_z)
        )
        fidelity = float(
            np.trace(corrected @ evaluator.target_density).real
        )
        fidelity = min(max(fidelity, 0.0), 1.0)
        return BackendALogicalRecord(
            code_survival_probability=min(max(survival, 0.0), 1.0),
            raw_logical_density=raw,
            frame_corrected_logical_density=corrected,
            bloch_xyz=bloch,
            target_fidelity=fidelity,
            logical_error=bool(fidelity < 0.5),
            evaluator_state=evaluator,
        )

    def step(
        self,
        state: BackendAState,
        action: ActionWord,
        exogenous: BackendAExogenous,
        *,
        evaluator: BackendAEvaluatorState | None = None,
    ) -> BackendARoundResult:
        if not isinstance(state, BackendAState):
            raise TypeError("state must be BackendAState")
        if state.cutoff != self.cutoff:
            raise ValueError("state cutoff mismatch")
        if not isinstance(action, ActionWord):
            raise TypeError("action must be a semantic+CRC-valid ActionWord")
        # Round-addressed randomness is part of the causal contract.
        if exogenous.round_index != state.round_index:
            raise ValueError("exogenous round index must equal state round index")
        if len(exogenous.iq_standard_i) != self.config.iq_samples:
            raise ValueError("exogenous IQ length mismatch")

        density = self._apply_action(
            state.joint_density,
            state.drift,
            action,
        )
        density = self._ramsey_pulse(
            density,
            state.drift,
            self.config.ramsey_angle,
        )
        density = self._sense(density, state.drift)
        density = self._ramsey_pulse(
            density,
            state.drift,
            -self.config.ramsey_angle,
        )
        (
            density,
            observation,
            sampled_level,
            pre_measurement,
            post_measurement,
        ) = self._measure_iq(density, state.drift, exogenous)
        pre_reset = self.level_probabilities(density)
        reset_hidden = "none"
        reset_observed = "none"
        if action.reset_request:
            density, reset_hidden, reset_observed = self._reset(
                density,
                exogenous,
            )
            observation = replace(observation, reset_ack=reset_observed)
        post_reset = self.level_probabilities(density)
        drift_after = self._update_drift(state.drift, action, exogenous)
        leakage_age = (
            min(state.leakage_age + 1, 65535)
            if post_reset[2] >= self.config.leakage_age_threshold
            else 0
        )
        next_state = BackendAState(
            joint_density=density,
            cutoff=self.cutoff,
            drift=drift_after,
            leakage_age=leakage_age,
            round_index=state.round_index + 1,
        )
        next_evaluator = evaluator.after(action) if evaluator is not None else None
        logical = (
            self.logical_record(next_state, next_evaluator)
            if next_evaluator is not None
            else None
        )
        alpha = self._action_alpha(action)
        truth = BackendATruthRecord(
            sampled_emission_level=sampled_level,
            pre_measurement_level_probabilities=pre_measurement,
            post_measurement_level_probabilities=post_measurement,
            reset_hidden_outcome=reset_hidden,
            pre_reset_level_probabilities=pre_reset,
            post_reset_level_probabilities=post_reset,
            action_code=NominalAction(action.action_code).name,
            action_alpha_real=float(alpha.real),
            action_alpha_imag=float(alpha.imag),
            drift_before=tuple(float(item) for item in state.drift.vector()),
            drift_after=tuple(float(item) for item in drift_after.vector()),
            density_diagnostics=_density_diagnostics(density),
        )
        return BackendARoundResult(
            state=next_state,
            observation=observation,
            truth=truth,
            logical=logical,
            action_word=action,
            exogenous=exogenous,
        )

    def simulate(
        self,
        initial_state: BackendAState,
        actions: Sequence[ActionWord],
        *,
        seed: int,
        evaluator: BackendAEvaluatorState | None = None,
    ) -> BackendATrajectory:
        state = initial_state
        active_evaluator = evaluator
        rounds: list[BackendARoundResult] = []
        for action in actions:
            exogenous = backend_a_exogenous(
                seed=seed,
                round_index=state.round_index,
                iq_samples=self.config.iq_samples,
            )
            result = self.step(
                state,
                action,
                exogenous,
                evaluator=active_evaluator,
            )
            rounds.append(result)
            state = result.state
            if result.logical is not None:
                active_evaluator = result.logical.evaluator_state
        return BackendATrajectory(
            rounds=tuple(rounds),
            initial_state=initial_state,
            final_state=state,
            seed=seed,
        )

    def channel_diagnostics(
        self,
        hamiltonian: ArrayLike,
        duration: float,
    ) -> ChannelDiagnostics:
        """Numerically construct the Choi matrix of one GKSL segment.

        This is intentionally used with a small cutoff in qualification.  It is
        an implementation check of vectorization, trace preservation and CP,
        not a statement that one successful output-state test proves CP.
        """

        if self.config.cutoff > MAX_EXACT_CHOI_CUTOFF:
            raise RuntimeError(
                "exact Choi construction is restricted to cutoff "
                f"<= {MAX_EXACT_CHOI_CUTOFF}; use scalable state/channel "
                "diagnostics at high cutoff"
            )
        time = _nonnegative(duration, "duration")
        generator = self.liouvillian(hamiltonian)
        superoperator = expm(generator.toarray() * time)
        dimension = self.dimension
        choi = np.zeros(
            (dimension * dimension, dimension * dimension),
            dtype=np.complex128,
        )
        for row in range(dimension):
            for column in range(dimension):
                basis = np.zeros(
                    (dimension, dimension),
                    dtype=np.complex128,
                )
                basis[row, column] = 1.0
                output = (
                    superoperator
                    @ basis.reshape(dimension * dimension, order="F")
                ).reshape((dimension, dimension), order="F")
                input_basis = np.zeros_like(basis)
                input_basis[row, column] = 1.0
                choi += np.kron(input_basis, output)
        hermiticity = float(
            np.linalg.norm(choi - choi.conj().T, ord="fro")
        )
        choi_hermitian = 0.5 * (choi + choi.conj().T)
        tensor = choi_hermitian.reshape(
            dimension,
            dimension,
            dimension,
            dimension,
        )
        partial_output = np.trace(tensor, axis1=1, axis2=3)
        return ChannelDiagnostics(
            dimension=dimension,
            choi_minimum_eigenvalue=float(
                np.min(np.linalg.eigvalsh(choi_hermitian))
            ),
            choi_trace=float(np.trace(choi_hermitian).real),
            trace_preservation_frobenius=float(
                np.linalg.norm(
                    partial_output - np.eye(dimension),
                    ord="fro",
                )
            ),
            hermiticity_frobenius=hermiticity,
        )


def _noise_free_config(
    base: BackendAConfig,
    **overrides: object,
) -> BackendAConfig:
    values: dict[str, object] = {
        "oscillator_loss_rate": 0.0,
        "oscillator_dephasing_rate": 0.0,
        "ancilla_ge_relax_rate": 0.0,
        "ancilla_fe_relax_rate": 0.0,
        "ancilla_ge_excitation_rate": 0.0,
        "ancilla_dephasing_rate": 0.0,
        "pulse_leakage_crosstalk": 0.0,
        "measurement_leakage_coupling": 0.0,
        "action_leakage_coupling": 0.0,
        "dispersive_chi": 0.0,
        "self_kerr": 0.0,
        "ramsey_angle": 0.0,
        "sense_duration": 0.0,
        "iq_centers": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        "reset_ack_error": 0.0,
        "drift_retention": (1.0, 1.0, 1.0, 1.0, 1.0),
        "drift_noise_std": (0.0, 0.0, 0.0, 0.0, 0.0),
        "drift_action_kick": 0.0,
        "drift_readout_heating": 0.0,
        "drift_leakage_heating": 0.0,
    }
    values.update(overrides)
    return replace(base, **values)


def _embedded_density_distance(
    low: FiniteCutoffDensity,
    high: FiniteCutoffDensity,
) -> float:
    if low.cutoff >= high.cutoff:
        raise ValueError("low cutoff must be smaller than high cutoff")
    embedded = np.zeros_like(high.matrix)
    embedded[: low.cutoff, : low.cutoff] = low.matrix
    return _trace_distance(embedded, high.matrix)


def run_backend_a_qualification(
    config: BackendAConfig | None = None,
    thresholds: BackendAQualificationThresholds | None = None,
) -> BackendAQualification:
    """Execute the T9.2.2 implementation-level qualification suite."""

    base = BackendAConfig() if config is None else config
    if not isinstance(base, BackendAConfig):
        raise TypeError("config must be BackendAConfig or None")
    limits = (
        BackendAQualificationThresholds()
        if thresholds is None
        else thresholds
    )
    if not isinstance(limits, BackendAQualificationThresholds):
        raise TypeError(
            "thresholds must be BackendAQualificationThresholds or None"
        )
    metrics: dict[str, float | int | str | bool] = {}
    checks: dict[str, bool] = {}

    # 1. Small-system exact Choi test of the actual vectorized GKSL code path.
    cp_config = replace(
        base,
        cutoff=2,
        iq_samples=min(base.iq_samples, 4),
        logical_grid_points=1025,
    )
    cp_simulator = Phase9BackendASimulator(cp_config)
    cp_hamiltonian = cp_simulator._base_hamiltonian(BackendADriftState())
    channel = cp_simulator.channel_diagnostics(cp_hamiltonian, 0.07)
    metrics.update(
        {
            "choi_dimension": channel.dimension,
            "choi_minimum_eigenvalue": channel.choi_minimum_eigenvalue,
            "choi_trace": channel.choi_trace,
            "choi_tp_frobenius": channel.trace_preservation_frobenius,
            "choi_hermiticity_frobenius": channel.hermiticity_frobenius,
        }
    )
    checks["gksl_channel_cp"] = (
        channel.choi_minimum_eigenvalue
        >= limits.choi_minimum_eigenvalue
    )
    checks["gksl_channel_tp"] = (
        channel.trace_preservation_frobenius
        <= limits.choi_tp_frobenius
    )
    checks["gksl_choi_hermitian"] = (
        channel.hermiticity_frobenius
        <= limits.choi_hermiticity_frobenius
    )

    # 2. One full joint round: density and instrument invariants.
    simulator = Phase9BackendASimulator(base)
    initial, evaluator = simulator.initialize_logical("0")
    idle = diagnostic_action_word("IDLE")
    exogenous = backend_a_exogenous(
        seed=731,
        round_index=0,
        iq_samples=base.iq_samples,
    )
    normal = simulator.step(
        initial,
        idle,
        exogenous,
        evaluator=evaluator,
    )
    diagnostics = _density_diagnostics(normal.state.joint_density)
    metrics.update(
        {
            "full_round_trace_error": abs(
                diagnostics["trace_real"] - 1.0
            )
            + abs(diagnostics["trace_imag"]),
            "full_round_hermiticity_frobenius": diagnostics[
                "hermiticity_frobenius"
            ],
            "full_round_minimum_eigenvalue": diagnostics[
                "minimum_eigenvalue"
            ],
            "measurement_completeness_frobenius": simulator.measurement_completeness_error(),
            "reset_completeness_frobenius": simulator.reset_completeness_error(),
            "full_round_posterior_sum_error": abs(
                sum(normal.observation.posterior_levels) - 1.0
            ),
            "full_round_code_survival": (
                normal.logical.code_survival_probability
                if normal.logical is not None
                else -1.0
            ),
        }
    )
    checks["full_round_trace"] = (
        metrics["full_round_trace_error"]
        <= limits.density_trace_error
    )
    checks["full_round_hermiticity"] = (
        diagnostics["hermiticity_frobenius"]
        <= limits.density_hermiticity_frobenius
    )
    checks["full_round_positive"] = (
        diagnostics["minimum_eigenvalue"]
        >= limits.density_minimum_eigenvalue
    )
    checks["measurement_instrument_complete"] = (
        metrics["measurement_completeness_frobenius"]
        <= limits.instrument_completeness_frobenius
    )
    checks["reset_instrument_complete"] = (
        metrics["reset_completeness_frobenius"]
        <= limits.instrument_completeness_frobenius
    )
    checks["probabilities_normalized"] = (
        metrics["full_round_posterior_sum_error"]
        <= limits.probability_sum_error
    )
    checks["logical_tracking_finite"] = (
        normal.logical is not None
        and 0.0 <= normal.logical.code_survival_probability <= 1.0
        and 0.0 <= normal.logical.target_fidelity <= 1.0
    )

    # 3. Zero-noise identity and ideal displacement limits.
    # Use a fine pulse discretization for the continuum ideal-action limit.
    # The separate convergence test below verifies that this is not merely a
    # hand-picked one-grid agreement.
    limit_config = _noise_free_config(base, substeps_per_segment=64)
    limit_simulator = Phase9BackendASimulator(limit_config)
    vacuum = limit_simulator.initialize_fock()
    limit_exogenous = backend_a_exogenous(
        seed=11,
        round_index=0,
        iq_samples=limit_config.iq_samples,
    )
    zero = limit_simulator.step(vacuum, idle, limit_exogenous)
    zero_distance = _trace_distance(
        vacuum.joint_density,
        zero.state.joint_density,
    )
    x_action = diagnostic_action_word("X")
    acted = limit_simulator.step(vacuum, x_action, limit_exogenous)
    alpha = limit_simulator._action_alpha(x_action)
    displacement = limit_simulator.oscillator.displacement_operator(alpha)
    expected = np.kron(
        displacement
        @ limit_simulator.oscillator_density(vacuum.joint_density).matrix
        @ displacement.conj().T,
        limit_simulator.level_projectors[0],
    )
    ideal_action_distance = _trace_distance(
        expected,
        acted.state.joint_density,
    )
    metrics["zero_noise_idle_trace_distance"] = zero_distance
    metrics["ideal_action_trace_distance"] = ideal_action_distance
    checks["zero_noise_idle_limit"] = (
        zero_distance <= limits.zero_noise_idle_trace_distance
    )
    checks["ideal_action_limit"] = (
        ideal_action_distance <= limits.ideal_action_trace_distance
    )

    # 4. Reset success/failure limits and f-state persistence.
    reset_word = diagnostic_action_word("RESET")
    success_config = _noise_free_config(
        base,
        reset_success_e=1.0,
        reset_success_f=1.0,
    )
    success_simulator = Phase9BackendASimulator(success_config)
    f_state = success_simulator.initialize_fock(ancilla_state="f")
    reset_exogenous = BackendAExogenous(
        emission_uniform=0.5,
        iq_standard_i=(0.0,) * success_config.iq_samples,
        iq_standard_q=(0.0,) * success_config.iq_samples,
        reset_uniform=0.5,
        reset_ack_uniform=0.5,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=17,
        round_index=0,
    )
    reset_success = success_simulator.step(
        f_state,
        reset_word,
        reset_exogenous,
    )
    success_g = success_simulator.level_probabilities(
        reset_success.state.joint_density
    )[0]
    failure_config = _noise_free_config(
        base,
        reset_success_e=0.0,
        reset_success_f=0.0,
    )
    failure_simulator = Phase9BackendASimulator(failure_config)
    failure_f = failure_simulator.initialize_fock(ancilla_state="f")
    reset_failure = failure_simulator.step(
        failure_f,
        reset_word,
        reset_exogenous,
    )
    failure_f_probability = failure_simulator.level_probabilities(
        reset_failure.state.joint_density
    )[2]
    persistent = failure_simulator.step(
        failure_f,
        idle,
        reset_exogenous,
    )
    persistent_f_probability = failure_simulator.level_probabilities(
        persistent.state.joint_density
    )[2]
    metrics.update(
        {
            "large_reset_g_probability": success_g,
            "failed_reset_f_probability": failure_f_probability,
            "no_reset_f_persistence_probability": persistent_f_probability,
        }
    )
    checks["large_reset_limit"] = (
        success_g >= limits.limit_population_minimum
        and reset_success.truth.reset_hidden_outcome == "success"
        and reset_success.observation.reset_ack == "success"
    )
    checks["reset_failure_preserves_f"] = (
        failure_f_probability >= limits.limit_population_minimum
        and reset_failure.truth.reset_hidden_outcome == "failure"
        and reset_failure.observation.reset_ack == "failure"
    )
    checks["f_state_persistence"] = (
        persistent_f_probability >= limits.limit_population_minimum
    )

    # 5. IQ backaction must alter the quantum state through a Kraus update.
    measurement_config = _noise_free_config(
        base,
        iq_sigma=0.24,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.6)),
    )
    measurement_simulator = Phase9BackendASimulator(measurement_config)
    ge_plus = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / sqrt(2.0)
    coherent_ancilla = measurement_simulator.initialize_fock(
        ancilla_state=ge_plus,
    )
    measurement_exogenous = BackendAExogenous(
        emission_uniform=0.1,
        iq_standard_i=(0.0,) * measurement_config.iq_samples,
        iq_standard_q=(0.0,) * measurement_config.iq_samples,
        reset_uniform=0.5,
        reset_ack_uniform=0.5,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=23,
        round_index=0,
    )
    measured = measurement_simulator.step(
        coherent_ancilla,
        idle,
        measurement_exogenous,
    )
    before_coherence = abs(
        measurement_simulator.ancilla_density(
            coherent_ancilla.joint_density
        )[0, 1]
    )
    after_coherence = abs(
        measurement_simulator.ancilla_density(
            measured.state.joint_density
        )[0, 1]
    )
    posterior_peak = max(measured.observation.posterior_levels)
    metrics.update(
        {
            "measurement_coherence_before": float(before_coherence),
            "measurement_coherence_after": float(after_coherence),
            "measurement_posterior_peak": posterior_peak,
        }
    )
    checks["iq_drives_measurement_backaction"] = (
        after_coherence
        < before_coherence * limits.measurement_coherence_ratio
        and posterior_peak > limits.measurement_posterior_peak
    )

    # 6. The Ramsey interaction must make the IQ instrument a syndrome
    # measurement of the oscillator, rather than an ancilla-only label sensor.
    syndrome_config = _noise_free_config(
        base,
        ramsey_angle=pi / 2.0,
        ramsey_pulse_duration=0.03,
        sense_duration=0.8,
        dispersive_chi=1.0,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
        substeps_per_segment=16,
    )
    syndrome_simulator = Phase9BackendASimulator(syndrome_config)
    ket_zero = np.zeros(syndrome_config.cutoff, dtype=np.complex128)
    ket_zero[0] = 1.0
    ket_one = np.zeros(syndrome_config.cutoff, dtype=np.complex128)
    ket_one[1] = 1.0
    syndrome_exogenous = BackendAExogenous(
        emission_uniform=0.1,
        iq_standard_i=(0.0,) * syndrome_config.iq_samples,
        iq_standard_q=(0.0,) * syndrome_config.iq_samples,
        reset_uniform=0.5,
        reset_ack_uniform=0.5,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=29,
        round_index=0,
    )
    syndrome_zero = syndrome_simulator.step(
        syndrome_simulator.initialize_fock(oscillator_ket=ket_zero),
        idle,
        syndrome_exogenous,
    )
    syndrome_one = syndrome_simulator.step(
        syndrome_simulator.initialize_fock(oscillator_ket=ket_one),
        idle,
        syndrome_exogenous,
    )
    level_tv = 0.5 * float(
        np.sum(
            np.abs(
                np.asarray(
                    syndrome_zero.truth.pre_measurement_level_probabilities
                )
                - np.asarray(
                    syndrome_one.truth.pre_measurement_level_probabilities
                )
            )
        )
    )
    superposition = (ket_zero + ket_one) / sqrt(2.0)
    superposition_state = syndrome_simulator.initialize_fock(
        oscillator_ket=superposition
    )
    syndrome_measured = syndrome_simulator.step(
        superposition_state,
        idle,
        syndrome_exogenous,
    )
    oscillator_backaction = _trace_distance(
        syndrome_simulator.oscillator_density(
            superposition_state.joint_density
        ).matrix,
        syndrome_simulator.oscillator_density(
            syndrome_measured.state.joint_density
        ).matrix,
    )
    metrics.update(
        {
            "syndrome_fock0_vs_fock1_level_tv": level_tv,
            "syndrome_oscillator_backaction_trace_distance": oscillator_backaction,
        }
    )
    checks["ramsey_syndrome_state_dependence"] = (
        level_tv > limits.syndrome_state_dependence_minimum
    )
    checks["syndrome_measurement_backacts_on_oscillator"] = (
        oscillator_backaction
        > limits.syndrome_backaction_trace_distance_minimum
    )

    # 7. The explicit e<->f Hamiltonian must create action-dependent leakage,
    # not merely increment a classical leakage label.
    leakage_config = _noise_free_config(
        base,
        action_leakage_coupling=0.8,
        action_duration=0.1,
        substeps_per_segment=32,
    )
    leakage_simulator = Phase9BackendASimulator(leakage_config)
    leakage_initial = leakage_simulator.initialize_fock(
        ancilla_state="e"
    )
    leakage_exogenous = BackendAExogenous(
        emission_uniform=0.2,
        iq_standard_i=(0.0,) * leakage_config.iq_samples,
        iq_standard_q=(0.0,) * leakage_config.iq_samples,
        reset_uniform=0.2,
        reset_ack_uniform=0.2,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=30,
        round_index=0,
    )
    leakage_idle = leakage_simulator.step(
        leakage_initial,
        idle,
        leakage_exogenous,
    )
    leakage_x = leakage_simulator.step(
        leakage_initial,
        x_action,
        leakage_exogenous,
    )
    action_f_difference = (
        leakage_simulator.level_probabilities(
            leakage_x.state.joint_density
        )[2]
        - leakage_simulator.level_probabilities(
            leakage_idle.state.joint_density
        )[2]
    )
    metrics["action_induced_f_population_difference"] = action_f_difference
    checks["action_induces_physical_f_population"] = (
        action_f_difference
        > limits.action_induced_f_population_minimum
    )

    # 8. Same addressable randomness, different action: both quantum state and
    # latent drift must change.  This rejects independent label-noise shortcuts.
    intervention_config = replace(
        base,
        ramsey_angle=0.0,
        sense_duration=0.0,
        iq_centers=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    intervention_simulator = Phase9BackendASimulator(intervention_config)
    intervention_initial = intervention_simulator.initialize_fock()
    intervention_exogenous = backend_a_exogenous(
        seed=991,
        round_index=0,
        iq_samples=intervention_config.iq_samples,
    )
    intervention_idle = intervention_simulator.step(
        intervention_initial,
        idle,
        intervention_exogenous,
    )
    intervention_x = intervention_simulator.step(
        intervention_initial,
        x_action,
        intervention_exogenous,
    )
    state_intervention = _trace_distance(
        intervention_idle.state.joint_density,
        intervention_x.state.joint_density,
    )
    drift_intervention = float(
        np.linalg.norm(
            intervention_idle.state.drift.vector()
            - intervention_x.state.drift.vector()
        )
    )
    metrics.update(
        {
            "action_intervention_state_trace_distance": state_intervention,
            "action_intervention_drift_l2": drift_intervention,
            "action_intervention_shared_exogenous": (
                intervention_idle.exogenous == intervention_x.exogenous
            ),
        }
    )
    checks["action_changes_quantum_transition"] = (
        state_intervention
        > limits.action_state_trace_distance_minimum
    )
    checks["action_changes_latent_drift"] = (
        drift_intervention > limits.action_drift_l2_minimum
    )
    checks["intervention_uses_common_randomness"] = (
        intervention_idle.exogenous == intervention_x.exogenous
    )

    # 9. Seed determinism and genuine stochastic sensitivity.
    deterministic_one = simulator.simulate(
        initial,
        (idle, x_action, idle),
        seed=404,
        evaluator=evaluator,
    )
    deterministic_two = simulator.simulate(
        initial,
        (idle, x_action, idle),
        seed=404,
        evaluator=evaluator,
    )
    stochastic_other = simulator.simulate(
        initial,
        (idle, x_action, idle),
        seed=405,
        evaluator=evaluator,
    )
    deterministic_density_error = float(
        np.max(
            np.abs(
                deterministic_one.final_state.joint_density
                - deterministic_two.final_state.joint_density
            )
        )
    )
    deterministic_iq_error = max(
        float(
            np.max(
                np.abs(
                    left.observation.iq_i - right.observation.iq_i
                )
            )
        )
        for left, right in zip(
            deterministic_one.rounds,
            deterministic_two.rounds,
        )
    )
    seed_sensitivity = float(
        np.max(
            np.abs(
                deterministic_one.rounds[0].observation.iq_i
                - stochastic_other.rounds[0].observation.iq_i
            )
        )
    )
    metrics.update(
        {
            "seed_repeat_density_max_error": deterministic_density_error,
            "seed_repeat_iq_max_error": deterministic_iq_error,
            "different_seed_iq_max_difference": seed_sensitivity,
        }
    )
    checks["seed_determinism"] = (
        deterministic_density_error == 0.0
        and deterministic_iq_error == 0.0
    )
    checks["different_seed_changes_observation"] = (
        seed_sensitivity
        > limits.different_seed_iq_difference_minimum
    )

    # 10. Step-size convergence of a non-commuting drift + shaped action pulse.
    convergence_base = _noise_free_config(
        base,
        dispersive_chi=0.23,
        self_kerr=0.015,
        ramsey_angle=0.0,
        sense_duration=0.0,
        drift_action_kick=0.0,
    )
    pre_coarse_simulator = Phase9BackendASimulator(
        replace(convergence_base, substeps_per_segment=8)
    )
    coarse_simulator = Phase9BackendASimulator(
        replace(convergence_base, substeps_per_segment=16)
    )
    fine_simulator = Phase9BackendASimulator(
        replace(convergence_base, substeps_per_segment=32)
    )
    convergence_drift = BackendADriftState(drive_q=0.07, drive_p=-0.04)
    pre_coarse_initial = pre_coarse_simulator.initialize_fock(
        drift=convergence_drift
    )
    coarse_initial = coarse_simulator.initialize_fock(drift=convergence_drift)
    fine_initial = fine_simulator.initialize_fock(drift=convergence_drift)
    convergence_exogenous = BackendAExogenous(
        emission_uniform=0.2,
        iq_standard_i=(0.0,) * convergence_base.iq_samples,
        iq_standard_q=(0.0,) * convergence_base.iq_samples,
        reset_uniform=0.2,
        reset_ack_uniform=0.2,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=55,
        round_index=0,
    )
    pre_coarse = pre_coarse_simulator.step(
        pre_coarse_initial,
        x_action,
        convergence_exogenous,
    )
    coarse = coarse_simulator.step(
        coarse_initial,
        x_action,
        convergence_exogenous,
    )
    fine = fine_simulator.step(
        fine_initial,
        x_action,
        convergence_exogenous,
    )
    step_distance = _trace_distance(
        coarse.state.joint_density,
        fine.state.joint_density,
    )
    previous_step_distance = _trace_distance(
        pre_coarse.state.joint_density,
        coarse.state.joint_density,
    )
    convergence_ratio = step_distance / previous_step_distance
    metrics["step_size_8_vs_16_trace_distance"] = previous_step_distance
    metrics["step_size_16_vs_32_trace_distance"] = step_distance
    metrics["step_size_error_ratio"] = convergence_ratio
    checks["step_size_convergence"] = (
        step_distance <= limits.step_size_trace_distance
        and convergence_ratio <= limits.step_size_error_ratio
    )

    # 11. Fock-cutoff convergence on a low-energy physical trajectory.
    low_config = replace(convergence_base, cutoff=8, substeps_per_segment=12)
    high_config = replace(convergence_base, cutoff=12, substeps_per_segment=12)
    low_simulator = Phase9BackendASimulator(low_config)
    high_simulator = Phase9BackendASimulator(high_config)
    low_initial = low_simulator.initialize_fock(drift=convergence_drift)
    high_initial = high_simulator.initialize_fock(drift=convergence_drift)
    cutoff_exogenous_low = BackendAExogenous(
        emission_uniform=0.2,
        iq_standard_i=(0.0,) * low_config.iq_samples,
        iq_standard_q=(0.0,) * low_config.iq_samples,
        reset_uniform=0.2,
        reset_ack_uniform=0.2,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=56,
        round_index=0,
    )
    cutoff_exogenous_high = replace(cutoff_exogenous_low)
    low_result = low_simulator.step(
        low_initial,
        x_action,
        cutoff_exogenous_low,
    )
    high_result = high_simulator.step(
        high_initial,
        x_action,
        cutoff_exogenous_high,
    )
    cutoff_distance = _embedded_density_distance(
        low_simulator.oscillator_density(low_result.state.joint_density),
        high_simulator.oscillator_density(high_result.state.joint_density),
    )
    metrics["fock_cutoff_8_vs_12_trace_distance"] = cutoff_distance
    checks["fock_cutoff_convergence"] = (
        cutoff_distance <= limits.fock_cutoff_trace_distance
    )

    # 12. All six evaluator states initialize with exact logical fidelity.
    initial_fidelities: list[float] = []
    for label in ("0", "1", "+", "-", "+i", "-i"):
        state, truth = simulator.initialize_logical(label)
        record = simulator.logical_record(state, truth)
        initial_fidelities.append(record.target_fidelity)
    minimum_initial_fidelity = min(initial_fidelities)
    metrics["six_state_initial_minimum_fidelity"] = minimum_initial_fidelity
    checks["six_state_logical_projection"] = (
        minimum_initial_fidelity
        >= limits.six_state_initial_fidelity
    )

    claim_state = {
        "backend_b_qualified": None,
        "dual_backend_agreement": None,
        "round_ler": None,
        "six_state_lifetime": None,
        "physical_break_even": None,
        "official_puviani_exact": None,
        "puviani_nmf_surpass": None,
        "external_sota": None,
        "hardware_measured": None,
        "rank": None,
    }
    verdict = (
        "QUALIFIED_BACKEND_A_ONLY"
        if checks and all(checks.values())
        else "NO_GO_BACKEND_A_QUALIFICATION"
    )
    return BackendAQualification(
        backend_id=BACKEND_A_ID,
        scope=BACKEND_A_SCOPE,
        config_sha256=base.semantic_sha256(),
        metrics=metrics,
        checks=checks,
        claim_state=claim_state,
        verdict=verdict,
    )


__all__ = [
    "ANCILLA_LEVELS",
    "BACKEND_A_ID",
    "BACKEND_A_SCOPE",
    "BackendAConfig",
    "BackendADriftState",
    "BackendAEvaluatorState",
    "BackendAExogenous",
    "BackendALogicalRecord",
    "BackendAObservation",
    "BackendAQualification",
    "BackendAQualificationThresholds",
    "BackendARoundResult",
    "BackendAState",
    "BackendATrajectory",
    "BackendATruthRecord",
    "ChannelDiagnostics",
    "DEFAULT_PARAMETER_PROVENANCE",
    "MAX_SUPPORTED_CUTOFF",
    "MAX_EXACT_CHOI_CUTOFF",
    "Phase9BackendASimulator",
    "backend_a_exogenous",
    "diagnostic_action_word",
    "run_backend_a_qualification",
]
