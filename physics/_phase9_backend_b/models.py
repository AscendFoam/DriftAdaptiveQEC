"""Validated data contracts and addressed RNG for Phase-9 backend B."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import blake2b, sha256
import json
from math import cos, isfinite, log, pi, sin, sqrt
import random
from sys import modules
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..phase9_twin_contract import (
    ActionWord,
    NominalAction,
)


ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]

BACKEND_B_ID = "PHASE9-BACKEND-B-DENSE-STRANG-ANALYTIC-KRAUS-V1"
MAX_SUPPORTED_CUTOFF = 32
MAX_EXACT_CHOI_CUTOFF = 8
BACKEND_B_SCOPE = (
    "independent dense-unitary/analytic-Kraus finite-Fock x qutrit synthetic "
    "qualification backend; no backend-A kernel, device calibration, lifetime, "
    "break-even, hardware, official-Puviani, external-SOTA or rank claim"
)
BACKEND_B_SOLVER_ID = "DENSE_EXPM_STRANG_PLUS_ANALYTIC_CHANNELS"
BACKEND_B_RNG_ID = "BLAKE2B_ADDRESS_PYTHON_RANDOM_BOX_MULLER"
BACKEND_B_LIKELIHOOD_ID = "INDEPENDENT_PRODUCT_COMPLEX_GAUSSIAN_LOG_LIKELIHOOD"
BACKEND_B_LOGICAL_ID = "INDEPENDENT_SQUEEZED_COHERENT_COMB_FOCK_PROJECTOR"
DEFAULT_PARAMETER_PROVENANCE = (
    "SYNTHETIC_DIMENSIONLESS_PHASE9_BACKEND_B_QUALIFICATION_NOT_DEVICE_CALIBRATED"
)
ANCILLA_LEVELS = ("g", "e", "f")
DRIFT_FIELDS = (
    "drive_q",
    "drive_p",
    "readout_i",
    "readout_q",
    "leakage_detuning",
)


def _runtime_cutoff_cap() -> int:
    facade = modules.get("physics.phase9_backend_b")
    return getattr(facade, "MAX_SUPPORTED_CUTOFF", MAX_SUPPORTED_CUTOFF)


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
        raise TypeError(f"{name} must be real")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive(value: object, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _probability(value: object, name: str) -> float:
    number = _nonnegative(value, name)
    if number > 1.0:
        raise ValueError(f"{name} must lie in [0,1]")
    return number


def _integer(
    value: object,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must lie in [{minimum},{maximum}]")
    return number


def _set_validated(instance: object, name: str, validator: Any) -> None:
    value = validator(getattr(instance, name), name)
    object.__setattr__(instance, name, value)


def _set_integer(
    instance: object,
    name: str,
    minimum: int,
    maximum: int,
) -> None:
    value = _integer(getattr(instance, name), name, minimum, maximum)
    object.__setattr__(instance, name, value)


def _tuple_of(
    value: Iterable[object],
    name: str,
    length: int,
    validator: Any,
) -> tuple[float, ...]:
    try:
        result = tuple(validator(item, name) for item in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be iterable") from exc
    if len(result) != length:
        raise ValueError(f"{name} must have length {length}")
    return result


def _diagnostics(matrix: ComplexMatrix) -> dict[str, float]:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    trace = complex(np.trace(matrix))
    return {
        "trace_real": float(trace.real),
        "trace_imag": float(trace.imag),
        "hermiticity_frobenius": float(
            np.linalg.norm(matrix - matrix.conj().T, ord="fro")
        ),
        "minimum_eigenvalue": float(
            np.min(np.linalg.eigvalsh(hermitian))
        ),
        "purity": float(np.trace(hermitian @ hermitian).real),
    }


def _density(
    value: ArrayLike,
    dimension: int,
    name: str,
    *,
    tolerance: float = 5.0e-9,
) -> ComplexMatrix:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} has wrong shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    data = _diagnostics(matrix)
    if data["hermiticity_frobenius"] > tolerance:
        raise ValueError(f"{name} must be Hermitian")
    if (
        abs(data["trace_real"] - 1.0) > tolerance
        or abs(data["trace_imag"]) > tolerance
    ):
        raise ValueError(f"{name} must have unit trace")
    if data["minimum_eigenvalue"] < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return _readonly(0.5 * (matrix + matrix.conj().T))


def _trace_distance(left: ComplexMatrix, right: ComplexMatrix) -> float:
    delta = 0.5 * ((left - right) + (left - right).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))


@dataclass(frozen=True)
class BackendBConfig:
    cutoff: int = 8
    split_steps_per_segment: int = 8
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
    comb_squeezing: float = 0.90
    comb_envelope: float = 0.11
    comb_half_width: int = 2
    parameter_provenance: str = DEFAULT_PARAMETER_PROVENANCE
    backend_id: str = BACKEND_B_ID
    scope: str = BACKEND_B_SCOPE

    def __post_init__(self) -> None:
        _set_integer(self, "cutoff", 2, _runtime_cutoff_cap())
        _set_integer(self, "split_steps_per_segment", 1, 256)
        for name in (
            "action_duration",
            "ramsey_pulse_duration",
            "sense_duration",
        ):
            _set_validated(self, name, _nonnegative)
        for name in ("ramsey_angle", "dispersive_chi", "self_kerr"):
            _set_validated(self, name, _finite)
        for name in (
            "action_displacement",
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
            "comb_squeezing",
            "comb_envelope",
        ):
            _set_validated(self, name, _nonnegative)
        _set_integer(self, "iq_samples", 1, 4096)
        _set_validated(self, "iq_sigma", _positive)
        centers = np.asarray(self.iq_centers, dtype=np.float64)
        if centers.shape != (3, 2) or not np.all(np.isfinite(centers)):
            raise ValueError("iq_centers must have finite shape (3,2)")
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
            _set_validated(self, name, _probability)
        object.__setattr__(
            self,
            "drift_retention",
            _tuple_of(
                self.drift_retention,
                "drift_retention",
                5,
                _probability,
            ),
        )
        object.__setattr__(
            self,
            "drift_noise_std",
            _tuple_of(
                self.drift_noise_std,
                "drift_noise_std",
                5,
                _nonnegative,
            ),
        )
        _set_integer(self, "comb_half_width", 1, 8)
        if (
            self.action_duration == 0.0
            and (
                self.action_displacement > 0.0
                or self.action_leakage_coupling > 0.0
            )
        ):
            raise ValueError(
                "nonzero action pulse requires positive duration"
            )
        if (
            self.ramsey_pulse_duration == 0.0
            and self.ramsey_angle != 0.0
        ):
            raise ValueError(
                "nonzero Ramsey pulse requires positive duration"
            )
        if (
            not isinstance(self.parameter_provenance, str)
            or not self.parameter_provenance.strip()
        ):
            raise ValueError("parameter_provenance must be non-empty")
        if self.backend_id != BACKEND_B_ID:
            raise ValueError("backend_id is immutable")
        if self.scope != BACKEND_B_SCOPE:
            raise ValueError("scope is immutable")

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
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class BackendBQualificationThresholds:
    choi_minimum_eigenvalue: float = -3.0e-9
    choi_tp_frobenius: float = 3.0e-9
    full_round_trace_error: float = 1.0e-8
    full_round_minimum_eigenvalue: float = -1.0e-8
    analytic_loss_mean_error: float = 2.0e-10
    analytic_relaxation_error: float = 2.0e-10
    instrument_completeness: float = 1.0e-12
    ideal_action_trace_distance: float = 1.0e-4
    limit_population_minimum: float = 1.0 - 2.0e-10
    measurement_posterior_peak: float = 0.999
    syndrome_state_dependence_minimum: float = 0.05
    syndrome_backaction_minimum: float = 0.01
    action_f_population_minimum: float = 0.10
    action_state_distance_minimum: float = 1.0e-3
    action_drift_minimum: float = 1.0e-5
    rng_sensitivity_minimum: float = 1.0e-6
    split_distance: float = 3.0e-4
    split_ratio: float = 0.35
    cutoff_distance: float = 5.0e-4
    six_state_initial_fidelity: float = 1.0 - 2.0e-9

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            number = _finite(value, name)
            if "minimum_eigenvalue" in name:
                if number > 0.0:
                    raise ValueError(f"{name} must be <= 0")
            elif name in {
                "limit_population_minimum",
                "measurement_posterior_peak",
                "six_state_initial_fidelity",
            }:
                _probability(number, name)
            elif number < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, number)

    def semantic_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class BackendBDrift:
    drive_q: float = 0.0
    drive_p: float = 0.0
    readout_i: float = 0.0
    readout_q: float = 0.0
    leakage_detuning: float = 0.0

    def __post_init__(self) -> None:
        for name in DRIFT_FIELDS:
            _set_validated(self, name, _finite)

    def vector(self) -> RealVector:
        return _readonly(
            [getattr(self, name) for name in DRIFT_FIELDS],
            dtype=np.dtype(np.float64),
        )

    @classmethod
    def from_vector(cls, value: ArrayLike) -> "BackendBDrift":
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (5,) or not np.all(np.isfinite(vector)):
            raise ValueError("drift vector must be finite shape (5,)")
        return cls(*[float(item) for item in vector])


@dataclass(frozen=True)
class BackendBState:
    joint_density: ComplexMatrix
    cutoff: int
    drift: BackendBDrift = BackendBDrift()
    leakage_age: int = 0
    round_index: int = 0

    def __post_init__(self) -> None:
        cutoff = _integer(
            self.cutoff,
            "cutoff",
            2,
            _runtime_cutoff_cap(),
        )
        object.__setattr__(
            self,
            "joint_density",
            _density(self.joint_density, cutoff * 3, "joint_density"),
        )
        object.__setattr__(self, "cutoff", cutoff)
        if not isinstance(self.drift, BackendBDrift):
            raise TypeError("drift must be BackendBDrift")
        _set_integer(self, "leakage_age", 0, 65535)
        _set_integer(self, "round_index", 0, (1 << 63) - 1)


@dataclass(frozen=True)
class BackendBRandomRecord:
    component_uniform: float
    iq_normal_i: tuple[float, ...]
    iq_normal_q: tuple[float, ...]
    reset_uniform: float
    ack_uniform: float
    drift_normal: tuple[float, float, float, float, float]
    seed: int
    round_index: int
    rng_id: str = BACKEND_B_RNG_ID

    def __post_init__(self) -> None:
        for name in ("component_uniform", "reset_uniform", "ack_uniform"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0,1)")
            object.__setattr__(self, name, value)
        for name in ("iq_normal_i", "iq_normal_q"):
            values = tuple(_finite(item, name) for item in getattr(self, name))
            if not values:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, values)
        if len(self.iq_normal_i) != len(self.iq_normal_q):
            raise ValueError("I/Q normal arrays must have equal length")
        drift = tuple(_finite(item, "drift_normal") for item in self.drift_normal)
        if len(drift) != 5:
            raise ValueError("drift_normal must have length 5")
        object.__setattr__(self, "drift_normal", drift)
        _set_integer(self, "seed", 0, (1 << 63) - 1)
        _set_integer(self, "round_index", 0, (1 << 63) - 1)
        if self.rng_id != BACKEND_B_RNG_ID:
            raise ValueError("rng_id is immutable")


def _box_muller(generator: random.Random) -> tuple[float, float]:
    u1 = max(generator.random(), 2.0**-53)
    u2 = generator.random()
    radius = sqrt(-2.0 * log(u1))
    angle = 2.0 * pi * u2
    return radius * cos(angle), radius * sin(angle)


def backend_b_random_record(
    *,
    seed: int,
    round_index: int,
    iq_samples: int,
) -> BackendBRandomRecord:
    seed_value = _integer(seed, "seed", 0, (1 << 63) - 1)
    round_value = _integer(
        round_index,
        "round_index",
        0,
        (1 << 63) - 1,
    )
    samples = _integer(iq_samples, "iq_samples", 1, 4096)
    address = (
        f"{BACKEND_B_RNG_ID}|{seed_value}|{round_value}".encode("ascii")
    )
    derived = int.from_bytes(blake2b(address, digest_size=16).digest(), "big")
    generator = random.Random(derived)
    normals: list[float] = []
    while len(normals) < 2 * samples + 5:
        normals.extend(_box_muller(generator))
    return BackendBRandomRecord(
        component_uniform=generator.random(),
        iq_normal_i=tuple(normals[:samples]),
        iq_normal_q=tuple(normals[samples : 2 * samples]),
        reset_uniform=generator.random(),
        ack_uniform=generator.random(),
        drift_normal=tuple(normals[2 * samples : 2 * samples + 5]),
        seed=seed_value,
        round_index=round_value,
    )


@dataclass(frozen=True)
class BackendBObservation:
    iq_i: RealVector
    iq_q: RealVector
    integrated_i: float
    integrated_q: float
    log_evidence_density: float
    posterior_levels: tuple[float, float, float]
    leakage_confidence_analog: float
    reset_ack: str
    source: str = "synthetic_backend_b_analog_pre_frontend"

    def __post_init__(self) -> None:
        i_values = np.asarray(self.iq_i, dtype=np.float64)
        q_values = np.asarray(self.iq_q, dtype=np.float64)
        if (
            i_values.ndim != 1
            or i_values.size == 0
            or q_values.shape != i_values.shape
            or not np.all(np.isfinite(i_values))
            or not np.all(np.isfinite(q_values))
        ):
            raise ValueError("I/Q arrays must be equal non-empty finite vectors")
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
            _set_validated(self, name, _finite)
        if abs(self.integrated_i - float(np.mean(i_values))) > 2.0e-12:
            raise ValueError("integrated_i must equal the I sample mean")
        if abs(self.integrated_q - float(np.mean(q_values))) > 2.0e-12:
            raise ValueError("integrated_q must equal the Q sample mean")
        posterior = tuple(
            _probability(item, "posterior_levels")
            for item in self.posterior_levels
        )
        if len(posterior) != 3 or abs(sum(posterior) - 1.0) > 2.0e-9:
            raise ValueError("posterior_levels must be normalized")
        object.__setattr__(self, "posterior_levels", posterior)
        leakage = _probability(
            self.leakage_confidence_analog,
            "leakage_confidence_analog",
        )
        object.__setattr__(self, "leakage_confidence_analog", leakage)
        if abs(leakage - posterior[2]) > 2.0e-12:
            raise ValueError("leakage confidence must equal f posterior")
        if self.reset_ack not in {"none", "success", "failure"}:
            raise ValueError("invalid reset acknowledgement")
        if self.source != "synthetic_backend_b_analog_pre_frontend":
            raise ValueError("backend B cannot relabel IQ as recorded/live")


@dataclass(frozen=True)
class BackendBTruth:
    sampled_component: str
    pre_measurement_levels: tuple[float, float, float]
    post_measurement_levels: tuple[float, float, float]
    reset_hidden_outcome: str
    pre_reset_levels: tuple[float, float, float]
    post_reset_levels: tuple[float, float, float]
    action_code: str
    drift_before: tuple[float, ...]
    drift_after: tuple[float, ...]
    density_diagnostics: Mapping[str, float]
    namespace: str = "BACKEND_B_LATENT_TRUTH_NOT_DEPLOYABLE"

    def __post_init__(self) -> None:
        if self.sampled_component not in ANCILLA_LEVELS:
            raise ValueError("sampled_component must be g/e/f")
        for name in (
            "pre_measurement_levels",
            "post_measurement_levels",
            "pre_reset_levels",
            "post_reset_levels",
        ):
            values = tuple(
                _probability(item, name) for item in getattr(self, name)
            )
            if len(values) != 3 or abs(sum(values) - 1.0) > 2.0e-9:
                raise ValueError(f"{name} must be a normalized qutrit law")
            object.__setattr__(self, name, values)
        if self.reset_hidden_outcome not in {"none", "success", "failure"}:
            raise ValueError("invalid hidden reset outcome")
        if self.action_code not in {
            item.name for item in NominalAction
            if item != NominalAction.INVALID
        }:
            raise ValueError("invalid action code")
        for name in ("drift_before", "drift_after"):
            values = tuple(_finite(item, name) for item in getattr(self, name))
            if len(values) != len(DRIFT_FIELDS):
                raise ValueError(f"{name} must have five fields")
            object.__setattr__(self, name, values)
        diagnostics = {
            str(key): _finite(value, f"density_diagnostics[{key!r}]")
            for key, value in self.density_diagnostics.items()
        }
        required = {
            "trace_real",
            "trace_imag",
            "hermiticity_frobenius",
            "minimum_eigenvalue",
            "purity",
        }
        if set(diagnostics) != required:
            raise ValueError("density diagnostics schema mismatch")
        object.__setattr__(
            self,
            "density_diagnostics",
            MappingProxyType(diagnostics),
        )
        if self.namespace != "BACKEND_B_LATENT_TRUTH_NOT_DEPLOYABLE":
            raise ValueError("latent truth namespace is immutable")


@dataclass(frozen=True)
class BackendBEvaluator:
    target_label: str
    target_density: ComplexMatrix
    pauli_x: int = 0
    pauli_z: int = 0

    def __post_init__(self) -> None:
        if self.target_label not in {"0", "1", "+", "-", "+i", "-i"}:
            raise ValueError("target_label must be a six-state label")
        identity = np.eye(2, dtype=np.complex128)
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array(
            [[0.0, -1.0j], [1.0j, 0.0]],
            dtype=np.complex128,
        )
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        expected = {
            "0": 0.5 * (identity + z),
            "1": 0.5 * (identity - z),
            "+": 0.5 * (identity + x),
            "-": 0.5 * (identity - x),
            "+i": 0.5 * (identity + y),
            "-i": 0.5 * (identity - y),
        }[self.target_label]
        target = _density(self.target_density, 2, "target_density")
        if not np.allclose(target, expected, atol=2.0e-12, rtol=0.0):
            raise ValueError("target_density does not match target_label")
        object.__setattr__(
            self,
            "target_density",
            target,
        )
        _set_integer(self, "pauli_x", 0, 1)
        _set_integer(self, "pauli_z", 0, 1)

    def after(self, action: ActionWord) -> "BackendBEvaluator":
        return replace(
            self,
            pauli_x=self.pauli_x ^ int(action.pauli_dx),
            pauli_z=self.pauli_z ^ int(action.pauli_dz),
        )


@dataclass(frozen=True)
class BackendBLogical:
    code_survival: float
    raw_density: ComplexMatrix
    corrected_density: ComplexMatrix
    bloch_xyz: tuple[float, float, float]
    target_fidelity: float
    logical_error: bool
    evaluator: BackendBEvaluator
    namespace: str = "BACKEND_B_EVALUATOR_TRUTH_NOT_DEPLOYABLE"

    def __post_init__(self) -> None:
        _set_validated(self, "code_survival", _probability)
        object.__setattr__(
            self,
            "raw_density",
            _density(self.raw_density, 2, "raw_density"),
        )
        object.__setattr__(
            self,
            "corrected_density",
            _density(self.corrected_density, 2, "corrected_density"),
        )
        bloch = tuple(_finite(item, "bloch_xyz") for item in self.bloch_xyz)
        if len(bloch) != 3 or np.linalg.norm(bloch) > 1.0 + 2.0e-8:
            raise ValueError("bloch_xyz must lie in the Bloch ball")
        object.__setattr__(self, "bloch_xyz", bloch)
        _set_validated(self, "target_fidelity", _probability)
        if not isinstance(self.logical_error, bool):
            raise TypeError("logical_error must be bool")
        if self.logical_error != (self.target_fidelity < 0.5):
            raise ValueError("logical_error must be derived from fidelity")
        if not isinstance(self.evaluator, BackendBEvaluator):
            raise TypeError("evaluator must be BackendBEvaluator")
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array(
            [[0.0, -1.0j], [1.0j, 0.0]],
            dtype=np.complex128,
        )
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        expected_bloch = tuple(
            float(np.trace(self.corrected_density @ operator).real)
            for operator in (x, y, z)
        )
        if not np.allclose(bloch, expected_bloch, atol=2.0e-10, rtol=0.0):
            raise ValueError("bloch_xyz is inconsistent with corrected_density")
        if self.namespace != "BACKEND_B_EVALUATOR_TRUTH_NOT_DEPLOYABLE":
            raise ValueError("evaluator truth namespace is immutable")


@dataclass(frozen=True)
class BackendBRound:
    state: BackendBState
    observation: BackendBObservation
    truth: BackendBTruth
    logical: BackendBLogical | None
    action: ActionWord
    random_record: BackendBRandomRecord

    def __post_init__(self) -> None:
        if not isinstance(self.state, BackendBState):
            raise TypeError("state must be BackendBState")
        if not isinstance(self.observation, BackendBObservation):
            raise TypeError("observation must be BackendBObservation")
        if not isinstance(self.truth, BackendBTruth):
            raise TypeError("truth must be BackendBTruth")
        if self.logical is not None and not isinstance(
            self.logical,
            BackendBLogical,
        ):
            raise TypeError("logical must be BackendBLogical or None")
        if not isinstance(self.action, ActionWord):
            raise TypeError("action must be ActionWord")
        if not isinstance(self.random_record, BackendBRandomRecord):
            raise TypeError("random_record must be BackendBRandomRecord")
        if self.state.round_index != self.random_record.round_index + 1:
            raise ValueError("round record/state index mismatch")
        if self.truth.action_code != NominalAction(
            self.action.action_code
        ).name:
            raise ValueError("truth/action mismatch")
        if not np.allclose(
            self.observation.posterior_levels,
            self.truth.post_measurement_levels,
            atol=2.0e-10,
            rtol=0.0,
        ):
            raise ValueError("observation/truth posterior mismatch")
        if not np.allclose(
            self.truth.drift_after,
            self.state.drift.vector(),
            atol=2.0e-12,
            rtol=0.0,
        ):
            raise ValueError("truth/state drift mismatch")


@dataclass(frozen=True)
class BackendBTrajectory:
    rounds: tuple[BackendBRound, ...]
    initial_state: BackendBState
    final_state: BackendBState
    seed: int

    def __post_init__(self) -> None:
        rounds = tuple(self.rounds)
        if any(not isinstance(item, BackendBRound) for item in rounds):
            raise TypeError("rounds must contain BackendBRound values")
        object.__setattr__(self, "rounds", rounds)
        if not isinstance(self.initial_state, BackendBState):
            raise TypeError("initial_state must be BackendBState")
        if not isinstance(self.final_state, BackendBState):
            raise TypeError("final_state must be BackendBState")
        _set_integer(self, "seed", 0, (1 << 63) - 1)
        expected = self.initial_state.round_index
        for item in rounds:
            if item.random_record.seed != self.seed:
                raise ValueError("trajectory seed mismatch")
            if item.random_record.round_index != expected:
                raise ValueError("trajectory round sequence is not contiguous")
            expected += 1
        terminal = rounds[-1].state if rounds else self.initial_state
        if terminal is not self.final_state and (
            terminal.round_index != self.final_state.round_index
            or terminal.leakage_age != self.final_state.leakage_age
            or terminal.drift != self.final_state.drift
            or not np.array_equal(
                terminal.joint_density,
                self.final_state.joint_density,
            )
        ):
            raise ValueError("trajectory final state mismatch")


@dataclass(frozen=True)
class BackendBQualification:
    config_sha256: str
    metrics: Mapping[str, float | int | str | bool]
    checks: Mapping[str, bool]
    claim_state: Mapping[str, object]
    verdict: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.config_sha256, str)
            or len(self.config_sha256) != 64
        ):
            raise ValueError("config_sha256 must be a SHA-256 hex digest")
        try:
            int(self.config_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "config_sha256 must be a SHA-256 hex digest"
            ) from exc
        metrics: dict[str, float | int | str | bool] = {}
        for key, value in self.metrics.items():
            if isinstance(value, (bool, np.bool_)):
                normalized: float | int | str | bool = bool(value)
            elif isinstance(value, (int, np.integer)):
                normalized = int(value)
            elif isinstance(value, (float, np.floating)):
                normalized = _finite(value, f"metrics[{key!r}]")
            elif isinstance(value, str):
                normalized = value
            else:
                raise TypeError("qualification metric has unsupported type")
            metrics[str(key)] = normalized
        checks = {
            str(key): bool(value)
            for key, value in self.checks.items()
            if isinstance(value, (bool, np.bool_))
        }
        if not metrics or not checks:
            raise ValueError("qualification metrics/checks cannot be empty")
        if len(checks) != len(self.checks):
            raise TypeError("qualification checks must be bool")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(
            self,
            "claim_state",
            MappingProxyType(dict(self.claim_state)),
        )
        if self.verdict not in {
            "QUALIFIED_BACKEND_B_ONLY",
            "NO_GO_BACKEND_B_QUALIFICATION",
        }:
            raise ValueError("invalid qualification verdict")

    @property
    def passed(self) -> bool:
        return (
            self.verdict == "QUALIFIED_BACKEND_B_ONLY"
            and all(self.checks.values())
            and all(value is None for value in self.claim_state.values())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": BACKEND_B_ID,
            "scope": BACKEND_B_SCOPE,
            "solver_id": BACKEND_B_SOLVER_ID,
            "rng_id": BACKEND_B_RNG_ID,
            "likelihood_id": BACKEND_B_LIKELIHOOD_ID,
            "logical_id": BACKEND_B_LOGICAL_ID,
            "config_sha256": self.config_sha256,
            "metrics": dict(self.metrics),
            "checks": dict(self.checks),
            "claim_state": dict(self.claim_state),
            "verdict": self.verdict,
            "passed": self.passed,
        }
