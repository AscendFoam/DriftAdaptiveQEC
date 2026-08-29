"""Phase-9 backend B: independently implemented dense split-channel solver.

Backend B intentionally does not import backend A, the repository Fock density
model, the SBS cycle implementation, or the finite-energy GKP projector.  Its
independence mechanisms are:

* dense ``scipy.linalg.expm`` unitary half-steps with a Strang split;
* explicit analytic pure-loss, amplitude-damping and dephasing channels;
* a BLAKE2b-addressed Python ``random.Random`` stream and manual Box--Muller
  normals (not NumPy ``SeedSequence``/``Generator``);
* an independently written IQ mixture likelihood and Kraus update;
* an independently constructed squeezed coherent-comb logical basis.

It implements the same dimensionless Phase-9 physical semantics needed for a
future distributional comparison with backend A, but shares no transition
kernel, likelihood function, logical projector, RNG stream, or truth cache.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import blake2b, sha256
import json
from math import comb, cos, exp, isfinite, log, pi, sin, sqrt
import random
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import expm

from .phase9_twin_contract import (
    ActionWord,
    NominalAction,
    execute_representative_probe,
    representative_action_probes,
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
    hermitian = 0.5 * (matrix + matrix.conj().T)
    trace = complex(np.trace(matrix))
    if np.linalg.norm(matrix - matrix.conj().T, ord="fro") > tolerance:
        raise ValueError(f"{name} must be Hermitian")
    if (
        abs(trace.real - 1.0) > tolerance
        or abs(trace.imag) > tolerance
    ):
        raise ValueError(f"{name} must have unit trace")
    if float(np.min(np.linalg.eigvalsh(hermitian))) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return _readonly(hermitian)


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
        object.__setattr__(
            self,
            "cutoff",
            _integer(
                self.cutoff,
                "cutoff",
                2,
                MAX_SUPPORTED_CUTOFF,
            ),
        )
        object.__setattr__(
            self,
            "split_steps_per_segment",
            _integer(
                self.split_steps_per_segment,
                "split_steps_per_segment",
                1,
                256,
            ),
        )
        for name in (
            "action_duration",
            "ramsey_pulse_duration",
            "sense_duration",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        for name in ("ramsey_angle", "dispersive_chi", "self_kerr"):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
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
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "iq_samples",
            _integer(self.iq_samples, "iq_samples", 1, 4096),
        )
        object.__setattr__(
            self,
            "iq_sigma",
            _positive(self.iq_sigma, "iq_sigma"),
        )
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
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name),
            )
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
        object.__setattr__(
            self,
            "comb_half_width",
            _integer(self.comb_half_width, "comb_half_width", 1, 8),
        )
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
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )

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
            MAX_SUPPORTED_CUTOFF,
        )
        object.__setattr__(
            self,
            "joint_density",
            _density(self.joint_density, cutoff * 3, "joint_density"),
        )
        object.__setattr__(self, "cutoff", cutoff)
        if not isinstance(self.drift, BackendBDrift):
            raise TypeError("drift must be BackendBDrift")
        object.__setattr__(
            self,
            "leakage_age",
            _integer(self.leakage_age, "leakage_age", 0, 65535),
        )
        object.__setattr__(
            self,
            "round_index",
            _integer(
                self.round_index,
                "round_index",
                0,
                (1 << 63) - 1,
            ),
        )


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
        object.__setattr__(
            self,
            "seed",
            _integer(self.seed, "seed", 0, (1 << 63) - 1),
        )
        object.__setattr__(
            self,
            "round_index",
            _integer(
                self.round_index,
                "round_index",
                0,
                (1 << 63) - 1,
            ),
        )
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
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name),
            )
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
        object.__setattr__(
            self,
            "pauli_x",
            _integer(self.pauli_x, "pauli_x", 0, 1),
        )
        object.__setattr__(
            self,
            "pauli_z",
            _integer(self.pauli_z, "pauli_z", 0, 1),
        )

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
        object.__setattr__(
            self,
            "code_survival",
            _probability(self.code_survival, "code_survival"),
        )
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
        object.__setattr__(
            self,
            "target_fidelity",
            _probability(self.target_fidelity, "target_fidelity"),
        )
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
        object.__setattr__(
            self,
            "seed",
            _integer(self.seed, "seed", 0, (1 << 63) - 1),
        )
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


def diagnostic_action_word_b(action_name: str) -> ActionWord:
    if action_name not in {
        item.name for item in NominalAction if item != NominalAction.INVALID
    }:
        raise ValueError("unknown action")
    for probe in representative_action_probes():
        if probe.expected_terminal == action_name:
            return execute_representative_probe(probe)[-1].recurrence.action_word
    raise ValueError("T9.2.1 has no matching diagnostic action")


class Phase9BackendBSimulator:
    def __init__(self, config: BackendBConfig) -> None:
        if not isinstance(config, BackendBConfig):
            raise TypeError("config must be BackendBConfig")
        self.config = config
        self.cutoff = config.cutoff
        self.dimension = self.cutoff * 3
        self.i_o = np.eye(self.cutoff, dtype=np.complex128)
        self.i_a = np.eye(3, dtype=np.complex128)
        self.i_joint = np.eye(self.dimension, dtype=np.complex128)
        a = np.zeros((self.cutoff, self.cutoff), dtype=np.complex128)
        for number in range(1, self.cutoff):
            a[number - 1, number] = sqrt(float(number))
        self.a = a
        self.adag = a.conj().T
        self.number = self.adag @ self.a
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
        self.ge_lower = np.outer(g, e.conj())
        self.ge_raise = self.ge_lower.conj().T
        self.ef_lower = np.outer(e, f.conj())
        self.ef_raise = self.ef_lower.conj().T
        self.y_ge = -1.0j * self.ge_lower + 1.0j * self.ge_raise
        self.x_ef = self.ef_lower + self.ef_raise
        self.joint_a = np.kron(self.a, self.i_a)
        self.joint_adag = np.kron(self.adag, self.i_a)
        self.joint_number = np.kron(self.number, self.i_a)
        self.joint_q = np.kron(self.q, self.i_a)
        self.joint_p = np.kron(self.p, self.i_a)
        self.joint_y_ge = np.kron(self.i_o, self.y_ge)
        self.joint_x_ef = np.kron(self.i_o, self.x_ef)
        self.joint_projectors = tuple(
            np.kron(self.i_o, projector)
            for projector in self.level_projectors
        )
        dispersion = self.level_projectors[1] + 2.0 * self.level_projectors[2]
        kerr = self.number @ (self.number - self.i_o)
        self._base_kerr_term = self.config.self_kerr * np.kron(kerr, self.i_a)
        self._base_dispersive_term = self.config.dispersive_chi * np.kron(
            self.number,
            dispersion,
        )
        self._logical_isometry: ComplexMatrix | None = None

    @staticmethod
    def _apply_kraus(
        matrix: ComplexMatrix,
        operators: Sequence[ComplexMatrix],
    ) -> ComplexMatrix:
        result = np.zeros_like(matrix)
        for operator in operators:
            result += operator @ matrix @ operator.conj().T
        return result

    @lru_cache(maxsize=8)
    def _pure_loss_coefficients(self, duration: float) -> tuple[RealVector, ...]:
        transmissivity = exp(-self.config.oscillator_loss_rate * duration)
        rows: list[RealVector] = []
        for lost in range(self.cutoff):
            coefficients = np.array(
                [
                    sqrt(comb(initial, lost))
                    * (1.0 - transmissivity) ** (0.5 * lost)
                    * transmissivity ** (0.5 * (initial - lost))
                    for initial in range(lost, self.cutoff)
                ],
                dtype=np.float64,
            )
            rows.append(_readonly(coefficients))
        return tuple(rows)

    @lru_cache(maxsize=8)
    def _pure_loss_operators(self, duration: float) -> tuple[ComplexMatrix, ...]:
        rows: list[ComplexMatrix] = []
        for lost, coefficients in enumerate(
            self._pure_loss_coefficients(duration)
        ):
            operator = np.zeros(
                (self.cutoff, self.cutoff),
                dtype=np.complex128,
            )
            initial = np.arange(lost, self.cutoff)
            operator[initial - lost, initial] = coefficients
            rows.append(_readonly(np.kron(operator, self.i_a)))
        return tuple(rows)

    def _apply_pure_loss(
        self,
        matrix: ComplexMatrix,
        duration: float,
    ) -> ComplexMatrix:
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        result = np.zeros_like(tensor)
        for lost, coefficients in enumerate(
            self._pure_loss_coefficients(duration)
        ):
            remaining = self.cutoff - lost
            result[:remaining, :, :remaining, :] += (
                coefficients[:, None, None, None]
                * tensor[lost:, :, lost:, :]
                * coefficients[None, None, :, None]
            )
        return result.reshape(self.dimension, self.dimension)

    @lru_cache(maxsize=32)
    def _local_amplitude_operators(
        self,
        source: int,
        target: int,
        probability: float,
    ) -> tuple[ComplexMatrix, ComplexMatrix]:
        no_jump = np.eye(3, dtype=np.complex128)
        no_jump[source, source] = sqrt(1.0 - probability)
        jump = np.zeros((3, 3), dtype=np.complex128)
        jump[target, source] = sqrt(probability)
        return (
            _readonly(np.kron(self.i_o, no_jump)),
            _readonly(np.kron(self.i_o, jump)),
        )

    def _apply_local_amplitude(
        self,
        matrix: ComplexMatrix,
        source: int,
        target: int,
        probability: float,
    ) -> ComplexMatrix:
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        diagonal = np.ones(3, dtype=np.float64)
        diagonal[source] = sqrt(1.0 - probability)
        result = (
            tensor
            * diagonal[None, :, None, None]
            * diagonal[None, None, None, :]
        )
        result[:, target, :, target] += (
            probability * tensor[:, source, :, source]
        )
        return result.reshape(self.dimension, self.dimension)

    @lru_cache(maxsize=8)
    def _oscillator_dephasing_factor(self, duration: float) -> ComplexMatrix:
        indices = np.arange(self.cutoff, dtype=np.float64)
        return _readonly(
            np.exp(
                -0.5
                * self.config.oscillator_dephasing_rate
                * duration
                * (indices[:, None] - indices[None, :]) ** 2
            )
        )

    def _dephase_oscillator(
        self,
        matrix: ComplexMatrix,
        duration: float,
    ) -> ComplexMatrix:
        if self.config.oscillator_dephasing_rate == 0.0:
            return matrix
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        factor = self._oscillator_dephasing_factor(duration)
        tensor = tensor * factor[:, None, :, None]
        return tensor.reshape(self.dimension, self.dimension)

    @lru_cache(maxsize=8)
    def _ancilla_dephasing_factor(self, duration: float) -> ComplexMatrix:
        weights = np.array([-1.0, 1.0, 2.0], dtype=np.float64)
        return _readonly(
            np.exp(
                -0.5
                * self.config.ancilla_dephasing_rate
                * duration
                * (weights[:, None] - weights[None, :]) ** 2
            )
        )

    def _dephase_ancilla(
        self,
        matrix: ComplexMatrix,
        duration: float,
    ) -> ComplexMatrix:
        if self.config.ancilla_dephasing_rate == 0.0:
            return matrix
        factor = self._ancilla_dephasing_factor(duration)
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        tensor = tensor * factor[None, :, None, :]
        return tensor.reshape(self.dimension, self.dimension)

    def _noise_channels(
        self,
        matrix: ComplexMatrix,
        duration: float,
    ) -> ComplexMatrix:
        result = matrix
        if self.config.oscillator_loss_rate > 0.0:
            result = self._apply_pure_loss(result, duration)
        result = self._dephase_oscillator(result, duration)
        local_channels = (
            (1, 0, self.config.ancilla_ge_relax_rate),
            (2, 1, self.config.ancilla_fe_relax_rate),
            (0, 1, self.config.ancilla_ge_excitation_rate),
        )
        for source, target, rate in local_channels:
            if rate > 0.0:
                probability = 1.0 - exp(-rate * duration)
                result = self._apply_local_amplitude(
                    result,
                    source,
                    target,
                    probability,
                )
        result = self._dephase_ancilla(result, duration)
        return result

    def channel_completeness_errors(self, duration: float) -> dict[str, float]:
        errors: dict[str, float] = {}
        gram = np.zeros_like(self.i_joint)
        for operator in self._pure_loss_operators(duration):
            gram += operator.conj().T @ operator
        errors["pure_loss"] = float(
            np.linalg.norm(gram - self.i_joint, ord="fro")
        )
        for label, source, target, rate in (
            ("ge_relax", 1, 0, self.config.ancilla_ge_relax_rate),
            ("fe_relax", 2, 1, self.config.ancilla_fe_relax_rate),
            ("ge_excite", 0, 1, self.config.ancilla_ge_excitation_rate),
        ):
            probability = 1.0 - exp(-rate * duration)
            operators = self._local_amplitude_operators(
                source,
                target,
                probability,
            )
            local_gram = sum(
                operator.conj().T @ operator for operator in operators
            )
            errors[label] = float(
                np.linalg.norm(local_gram - self.i_joint, ord="fro")
            )
        return errors

    def _base_hamiltonian(self, drift: BackendBDrift) -> ComplexMatrix:
        return (
            self._base_kerr_term
            + self._base_dispersive_term
            + drift.drive_q * self.joint_q
            + drift.drive_p * self.joint_p
            + drift.leakage_detuning * self.joint_projectors[2]
        )

    @staticmethod
    def _envelope(fraction: float) -> float:
        return 0.5 * pi * sin(pi * fraction)

    def _split_segment(
        self,
        density: ComplexMatrix,
        duration: float,
        hamiltonian_at: Any,
    ) -> ComplexMatrix:
        if duration == 0.0:
            return _readonly(density)
        steps = self.config.split_steps_per_segment
        dt = duration / steps
        result = np.asarray(density, dtype=np.complex128)
        half_cache: dict[bytes, ComplexMatrix] = {}
        for index in range(steps):
            midpoint = (index + 0.5) / steps
            hamiltonian = np.asarray(
                hamiltonian_at(midpoint),
                dtype=np.complex128,
            )
            if np.linalg.norm(
                hamiltonian - hamiltonian.conj().T,
                ord="fro",
            ) > 1.0e-10:
                raise ValueError("Hamiltonian must be Hermitian")
            cache_key = hamiltonian.tobytes(order="C")
            half = half_cache.get(cache_key)
            if half is None:
                half = _readonly(expm(-0.5j * dt * hamiltonian))
                half_cache[cache_key] = half
            result = half @ result @ half.conj().T
            result = self._noise_channels(result, dt)
            result = half @ result @ half.conj().T
        raw_trace = complex(np.trace(result))
        if (
            abs(raw_trace.imag) > 5.0e-9
            or abs(raw_trace.real - 1.0) > 1.0e-8
        ):
            raise RuntimeError(
                "dense split propagation violated trace preservation"
            )
        result = result / raw_trace.real
        return _density(result, self.dimension, "split_output", tolerance=1.0e-8)

    def _action_alpha(self, action: ActionWord) -> complex:
        scale = self.config.action_displacement / sqrt(2.0)
        return complex(
            scale * int(action.pauli_dx),
            scale * int(action.pauli_dz),
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

    def _apply_action(
        self,
        density: ComplexMatrix,
        drift: BackendBDrift,
        action: ActionWord,
    ) -> ComplexMatrix:
        base = self._base_hamiltonian(drift)
        duration = self.config.action_duration
        alpha = self._action_alpha(action)
        drive = np.zeros_like(base)
        leakage = np.zeros_like(base)
        if duration > 0.0:
            drive = 1.0j * (
                alpha * self.joint_adag
                - alpha.conjugate() * self.joint_a
            ) / duration
            leakage = (
                self.config.action_leakage_coupling
                * self._action_energy(action)
                * self.joint_x_ef
                / duration
            )
        return self._split_segment(
            density,
            duration,
            lambda fraction: base
            + self._envelope(fraction) * (drive + leakage),
        )

    def _pulse(
        self,
        density: ComplexMatrix,
        drift: BackendBDrift,
        angle: float,
    ) -> ComplexMatrix:
        duration = self.config.ramsey_pulse_duration
        if duration == 0.0 or angle == 0.0:
            return _readonly(density)
        base = self._base_hamiltonian(drift)
        ge = angle * self.joint_y_ge / (2.0 * duration)
        ef = (
            abs(angle)
            * self.config.pulse_leakage_crosstalk
            * self.joint_x_ef
            / (2.0 * duration)
        )
        return self._split_segment(
            density,
            duration,
            lambda fraction: base + self._envelope(fraction) * (ge + ef),
        )

    def _sense(
        self,
        density: ComplexMatrix,
        drift: BackendBDrift,
    ) -> ComplexMatrix:
        base = self._base_hamiltonian(drift)
        return self._split_segment(
            density,
            self.config.sense_duration,
            lambda _fraction: base
            + self.config.measurement_leakage_coupling * self.joint_x_ef,
        )

    def ancilla_density(self, density: ArrayLike) -> ComplexMatrix:
        matrix = np.asarray(density, dtype=np.complex128)
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        result = np.trace(tensor, axis1=0, axis2=2)
        return _readonly(0.5 * (result + result.conj().T))

    def oscillator_density(self, density: ArrayLike) -> ComplexMatrix:
        matrix = np.asarray(density, dtype=np.complex128)
        tensor = matrix.reshape(self.cutoff, 3, self.cutoff, 3)
        result = np.trace(tensor, axis1=1, axis2=3)
        result = 0.5 * (result + result.conj().T)
        trace = float(np.trace(result).real)
        if abs(trace - 1.0) > 1.0e-8:
            raise RuntimeError("oscillator partial trace is not normalized")
        return _readonly(result / trace)

    def level_probabilities(
        self,
        density: ArrayLike,
    ) -> tuple[float, float, float]:
        diagonal = np.real(np.diag(self.ancilla_density(density)))
        diagonal = np.maximum(diagonal, 0.0)
        diagonal /= np.sum(diagonal)
        return tuple(float(item) for item in diagonal)

    @staticmethod
    def _categorical(probabilities: Sequence[float], uniform: float) -> int:
        cumulative = 0.0
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if uniform < cumulative or index == len(probabilities) - 1:
                return index
        raise AssertionError("categorical failure")

    def _measure(
        self,
        density: ComplexMatrix,
        drift: BackendBDrift,
        record: BackendBRandomRecord,
    ) -> tuple[
        ComplexMatrix,
        BackendBObservation,
        str,
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        if len(record.iq_normal_i) != self.config.iq_samples:
            raise ValueError("random IQ length mismatch")
        prior = self.level_probabilities(density)
        component = self._categorical(prior, record.component_uniform)
        centers = np.asarray(self.config.iq_centers, dtype=np.float64).copy()
        centers[:, 0] += drift.readout_i
        centers[:, 1] += drift.readout_q
        sigma = self.config.iq_sigma
        i_values = (
            centers[component, 0]
            + sigma * np.asarray(record.iq_normal_i, dtype=np.float64)
        )
        q_values = (
            centers[component, 1]
            + sigma * np.asarray(record.iq_normal_q, dtype=np.float64)
        )
        log_likelihood: list[float] = []
        normalization = -log(2.0 * pi * sigma * sigma)
        for center_i, center_q in centers:
            total = 0.0
            for sample_i, sample_q in zip(i_values, q_values):
                total += normalization - (
                    (sample_i - center_i) ** 2
                    + (sample_q - center_q) ** 2
                ) / (2.0 * sigma * sigma)
            log_likelihood.append(total)
        maximum = max(log_likelihood)
        amplitudes = np.array(
            [exp(0.5 * (item - maximum)) for item in log_likelihood],
            dtype=np.complex128,
        )
        operator = np.kron(self.i_o, np.diag(amplitudes))
        unnormalized = operator @ density @ operator.conj().T
        evidence_scaled = float(np.trace(unnormalized).real)
        if evidence_scaled <= 0.0 or not isfinite(evidence_scaled):
            raise RuntimeError("measurement evidence invalid")
        post = _density(
            unnormalized / evidence_scaled,
            self.dimension,
            "measurement_output",
            tolerance=1.0e-8,
        )
        posterior = self.level_probabilities(post)
        observation = BackendBObservation(
            iq_i=i_values,
            iq_q=q_values,
            integrated_i=float(np.mean(i_values)),
            integrated_q=float(np.mean(q_values)),
            log_evidence_density=maximum + log(evidence_scaled),
            posterior_levels=posterior,
            leakage_confidence_analog=posterior[2],
            reset_ack="none",
        )
        return post, observation, ANCILLA_LEVELS[component], prior, posterior

    def measurement_completeness_error(self) -> float:
        return float(
            np.linalg.norm(
                sum(self.joint_projectors) - self.i_joint,
                ord="fro",
            )
        )

    @lru_cache(maxsize=1)
    def _reset_operators(self) -> Mapping[str, tuple[ComplexMatrix, ...]]:
        g, e, f = self.level_kets
        success = (
            np.kron(self.i_o, np.outer(g, g.conj())),
            sqrt(self.config.reset_success_e)
            * np.kron(self.i_o, np.outer(g, e.conj())),
            sqrt(self.config.reset_success_f)
            * np.kron(self.i_o, np.outer(g, f.conj())),
        )
        failed_local = (
            sqrt(1.0 - self.config.reset_success_e)
            * np.outer(e, e.conj())
            + sqrt(1.0 - self.config.reset_success_f)
            * np.outer(f, f.conj())
        )
        return MappingProxyType(
            {
                "success": tuple(_readonly(item) for item in success),
                "failure": (_readonly(np.kron(self.i_o, failed_local)),),
            }
        )

    def reset_completeness_error(self) -> float:
        gram = np.zeros_like(self.i_joint)
        for operators in self._reset_operators().values():
            for operator in operators:
                gram += operator.conj().T @ operator
        return float(np.linalg.norm(gram - self.i_joint, ord="fro"))

    def _reset(
        self,
        density: ComplexMatrix,
        record: BackendBRandomRecord,
    ) -> tuple[ComplexMatrix, str, str]:
        branches: dict[str, ComplexMatrix] = {}
        probabilities: dict[str, float] = {}
        for outcome, operators in self._reset_operators().items():
            branch = self._apply_kraus(density, operators)
            branches[outcome] = branch
            probabilities[outcome] = max(float(np.trace(branch).real), 0.0)
        if abs(sum(probabilities.values()) - 1.0) > 2.0e-9:
            raise RuntimeError("reset probability normalization failed")
        hidden = (
            "success"
            if record.reset_uniform < probabilities["success"]
            else "failure"
        )
        selected = probabilities[hidden]
        if selected <= 0.0:
            raise RuntimeError("selected impossible reset outcome")
        post = _density(
            branches[hidden] / selected,
            self.dimension,
            "reset_output",
            tolerance=1.0e-8,
        )
        observed = hidden
        if record.ack_uniform < self.config.reset_ack_error:
            observed = "failure" if hidden == "success" else "success"
        return post, hidden, observed

    def _drift_update(
        self,
        drift: BackendBDrift,
        action: ActionWord,
        record: BackendBRandomRecord,
    ) -> BackendBDrift:
        retention = np.asarray(self.config.drift_retention)
        noise = np.asarray(self.config.drift_noise_std) * np.asarray(
            record.drift_normal
        )
        energy = self._action_energy(action)
        forcing = np.array(
            [
                self.config.drift_action_kick * int(action.pauli_dx),
                self.config.drift_action_kick * int(action.pauli_dz),
                self.config.drift_readout_heating * energy,
                -0.5 * self.config.drift_readout_heating * energy,
                self.config.drift_leakage_heating * energy,
            ]
        )
        return BackendBDrift.from_vector(
            retention * drift.vector() + forcing + noise
        )

    def _comb_isometry(self) -> ComplexMatrix:
        if self.cutoff < 8:
            raise ValueError("logical comb requires cutoff >= 8")
        if self._logical_isometry is not None:
            return self._logical_isometry
        squeeze_generator = 0.5 * self.config.comb_squeezing * (
            self.a @ self.a - self.adag @ self.adag
        )
        squeezed_vacuum = expm(squeeze_generator)[:, 0]
        columns: list[ComplexMatrix] = []
        for bit in (0, 1):
            vector = np.zeros(self.cutoff, dtype=np.complex128)
            for index in range(
                -self.config.comb_half_width,
                self.config.comb_half_width + 1,
            ):
                q_position = (2 * index + bit) * sqrt(pi)
                alpha = q_position / sqrt(2.0)
                displacement = expm(
                    alpha * self.adag - alpha * self.a
                )
                weight = exp(
                    -0.5 * self.config.comb_envelope * q_position**2
                )
                vector += weight * (displacement @ squeezed_vacuum)
            vector /= sqrt(float(np.vdot(vector, vector).real))
            columns.append(vector)
        raw = np.column_stack(columns)
        gram = raw.conj().T @ raw
        values, vectors = np.linalg.eigh(gram)
        if float(np.min(values)) <= 1.0e-10:
            raise RuntimeError("independent logical comb is singular")
        inverse_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
        self._logical_isometry = _readonly(raw @ inverse_root)
        return self._logical_isometry

    @staticmethod
    def _logical_target(label: str) -> ComplexMatrix:
        identity = np.eye(2, dtype=np.complex128)
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        values = {
            "0": 0.5 * (identity + z),
            "1": 0.5 * (identity - z),
            "+": 0.5 * (identity + x),
            "-": 0.5 * (identity - x),
            "+i": 0.5 * (identity + y),
            "-i": 0.5 * (identity - y),
        }
        if label not in values:
            raise ValueError("unknown logical label")
        return values[label]

    def initialize_logical(
        self,
        label: str,
        *,
        ancilla_level: str = "g",
        drift: BackendBDrift | None = None,
    ) -> tuple[BackendBState, BackendBEvaluator]:
        if ancilla_level not in ANCILLA_LEVELS:
            raise ValueError("ancilla level invalid")
        logical = self._logical_target(label)
        isometry = self._comb_isometry()
        oscillator = isometry @ logical @ isometry.conj().T
        ancilla = self.level_projectors[ANCILLA_LEVELS.index(ancilla_level)]
        return (
            BackendBState(
                joint_density=np.kron(oscillator, ancilla),
                cutoff=self.cutoff,
                drift=BackendBDrift() if drift is None else drift,
            ),
            BackendBEvaluator(
                target_label=label,
                target_density=logical,
            ),
        )

    def initialize_fock(
        self,
        *,
        oscillator_ket: ArrayLike | None = None,
        ancilla_state: str | ArrayLike = "g",
        drift: BackendBDrift | None = None,
    ) -> BackendBState:
        if oscillator_ket is None:
            ket = np.zeros(self.cutoff, dtype=np.complex128)
            ket[0] = 1.0
        else:
            ket = np.asarray(oscillator_ket, dtype=np.complex128)
        if ket.shape != (self.cutoff,) or not np.all(np.isfinite(ket)):
            raise ValueError("oscillator ket shape/finite failure")
        ket_norm = float(np.vdot(ket, ket).real)
        if ket_norm <= 1.0e-15:
            raise ValueError("oscillator ket must have nonzero norm")
        ket = ket / sqrt(ket_norm)
        oscillator = np.outer(ket, ket.conj())
        if isinstance(ancilla_state, str):
            if ancilla_state not in ANCILLA_LEVELS:
                raise ValueError("ancilla state invalid")
            ancilla = self.level_projectors[
                ANCILLA_LEVELS.index(ancilla_state)
            ]
        else:
            value = np.asarray(ancilla_state, dtype=np.complex128)
            if value.shape == (3,):
                value_norm = float(np.vdot(value, value).real)
                if not np.all(np.isfinite(value)) or value_norm <= 1.0e-15:
                    raise ValueError(
                        "ancilla ket must be finite and have nonzero norm"
                    )
                value = value / sqrt(value_norm)
                ancilla = np.outer(value, value.conj())
            else:
                ancilla = _density(value, 3, "ancilla_state")
        return BackendBState(
            joint_density=np.kron(oscillator, ancilla),
            cutoff=self.cutoff,
            drift=BackendBDrift() if drift is None else drift,
        )

    def logical_record(
        self,
        state: BackendBState,
        evaluator: BackendBEvaluator,
    ) -> BackendBLogical:
        isometry = self._comb_isometry()
        oscillator = self.oscillator_density(state.joint_density)
        encoded = isometry.conj().T @ oscillator @ isometry
        survival = float(np.trace(encoded).real)
        if survival <= 1.0e-12:
            raise RuntimeError("logical support vanished")
        raw = encoded / survival
        x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        frame = np.eye(2, dtype=np.complex128)
        if evaluator.pauli_x:
            frame = x @ frame
        if evaluator.pauli_z:
            frame = z @ frame
        corrected = frame.conj().T @ raw @ frame
        fidelity = min(
            max(float(np.trace(corrected @ evaluator.target_density).real), 0.0),
            1.0,
        )
        return BackendBLogical(
            code_survival=min(max(survival, 0.0), 1.0),
            raw_density=_density(raw, 2, "raw_logical"),
            corrected_density=_density(corrected, 2, "corrected_logical"),
            bloch_xyz=tuple(
                float(np.trace(corrected @ operator).real)
                for operator in (x, y, z)
            ),
            target_fidelity=fidelity,
            logical_error=bool(fidelity < 0.5),
            evaluator=evaluator,
        )

    def step(
        self,
        state: BackendBState,
        action: ActionWord,
        random_record: BackendBRandomRecord,
        *,
        evaluator: BackendBEvaluator | None = None,
    ) -> BackendBRound:
        if state.cutoff != self.cutoff:
            raise ValueError("state cutoff mismatch")
        if not isinstance(action, ActionWord):
            raise TypeError("action must be ActionWord")
        if random_record.round_index != state.round_index:
            raise ValueError("random record round mismatch")
        density = self._apply_action(
            state.joint_density,
            state.drift,
            action,
        )
        density = self._pulse(
            density,
            state.drift,
            self.config.ramsey_angle,
        )
        density = self._sense(density, state.drift)
        density = self._pulse(
            density,
            state.drift,
            -self.config.ramsey_angle,
        )
        (
            density,
            observation,
            sampled,
            pre_measurement,
            post_measurement,
        ) = self._measure(density, state.drift, random_record)
        pre_reset = self.level_probabilities(density)
        hidden_reset = "none"
        observed_reset = "none"
        if action.reset_request:
            density, hidden_reset, observed_reset = self._reset(
                density,
                random_record,
            )
            observation = replace(
                observation,
                reset_ack=observed_reset,
            )
        post_reset = self.level_probabilities(density)
        drift_after = self._drift_update(
            state.drift,
            action,
            random_record,
        )
        leakage_age = (
            min(state.leakage_age + 1, 65535)
            if post_reset[2] >= self.config.leakage_age_threshold
            else 0
        )
        next_state = BackendBState(
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
        truth = BackendBTruth(
            sampled_component=sampled,
            pre_measurement_levels=pre_measurement,
            post_measurement_levels=post_measurement,
            reset_hidden_outcome=hidden_reset,
            pre_reset_levels=pre_reset,
            post_reset_levels=post_reset,
            action_code=NominalAction(action.action_code).name,
            drift_before=tuple(float(item) for item in state.drift.vector()),
            drift_after=tuple(float(item) for item in drift_after.vector()),
            density_diagnostics=_diagnostics(density),
        )
        return BackendBRound(
            state=next_state,
            observation=observation,
            truth=truth,
            logical=logical,
            action=action,
            random_record=random_record,
        )

    def simulate(
        self,
        initial: BackendBState,
        actions: Sequence[ActionWord],
        *,
        seed: int,
        evaluator: BackendBEvaluator | None = None,
    ) -> BackendBTrajectory:
        state = initial
        active_evaluator = evaluator
        rows: list[BackendBRound] = []
        for action in actions:
            record = backend_b_random_record(
                seed=seed,
                round_index=state.round_index,
                iq_samples=self.config.iq_samples,
            )
            result = self.step(
                state,
                action,
                record,
                evaluator=active_evaluator,
            )
            rows.append(result)
            state = result.state
            if result.logical is not None:
                active_evaluator = result.logical.evaluator
        return BackendBTrajectory(
            rounds=tuple(rows),
            initial_state=initial,
            final_state=state,
            seed=seed,
        )

    def split_channel_choi(
        self,
        hamiltonian: ComplexMatrix,
        duration: float,
    ) -> tuple[float, float, float]:
        if self.config.cutoff > MAX_EXACT_CHOI_CUTOFF:
            raise RuntimeError(
                "exact Choi construction is restricted to cutoff "
                f"<= {MAX_EXACT_CHOI_CUTOFF}; use scalable state/channel "
                "diagnostics at high cutoff"
            )
        dimension = self.dimension
        half = expm(-0.5j * duration * hamiltonian)

        def channel(matrix: ComplexMatrix) -> ComplexMatrix:
            result = half @ matrix @ half.conj().T
            result = self._noise_channels(result, duration)
            return half @ result @ half.conj().T

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
                input_basis = np.zeros_like(basis)
                input_basis[row, column] = 1.0
                choi += np.kron(input_basis, channel(basis))
        choi = 0.5 * (choi + choi.conj().T)
        tensor = choi.reshape(
            dimension,
            dimension,
            dimension,
            dimension,
        )
        partial = np.trace(tensor, axis1=1, axis2=3)
        return (
            float(np.min(np.linalg.eigvalsh(choi))),
            float(np.linalg.norm(partial - np.eye(dimension), ord="fro")),
            float(np.linalg.norm(choi - choi.conj().T, ord="fro")),
        )


def _noise_free(config: BackendBConfig, **overrides: object) -> BackendBConfig:
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
    return replace(config, **values)


def _zero_record(config: BackendBConfig, round_index: int = 0) -> BackendBRandomRecord:
    return BackendBRandomRecord(
        component_uniform=0.2,
        iq_normal_i=(0.0,) * config.iq_samples,
        iq_normal_q=(0.0,) * config.iq_samples,
        reset_uniform=0.2,
        ack_uniform=0.2,
        drift_normal=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=0,
        round_index=round_index,
    )


def _embedded_distance(
    low: ComplexMatrix,
    high: ComplexMatrix,
) -> float:
    embedded = np.zeros_like(high)
    embedded[: low.shape[0], : low.shape[1]] = low
    return _trace_distance(embedded, high)


def run_backend_b_qualification(
    config: BackendBConfig | None = None,
    thresholds: BackendBQualificationThresholds | None = None,
) -> BackendBQualification:
    base = BackendBConfig() if config is None else config
    limits = (
        BackendBQualificationThresholds()
        if thresholds is None
        else thresholds
    )
    if not isinstance(base, BackendBConfig):
        raise TypeError("config must be BackendBConfig")
    if not isinstance(limits, BackendBQualificationThresholds):
        raise TypeError(
            "thresholds must be BackendBQualificationThresholds"
        )
    metrics: dict[str, float | int | str | bool] = {}
    checks: dict[str, bool] = {}

    small_config = replace(base, cutoff=2, iq_samples=min(4, base.iq_samples))
    small = Phase9BackendBSimulator(small_config)
    hamiltonian = small._base_hamiltonian(BackendBDrift())
    choi_minimum, choi_tp, choi_hermiticity = small.split_channel_choi(
        hamiltonian,
        0.07,
    )
    metrics.update(
        {
            "choi_dimension": small.dimension,
            "choi_minimum_eigenvalue": choi_minimum,
            "choi_tp_frobenius": choi_tp,
            "choi_hermiticity_frobenius": choi_hermiticity,
        }
    )
    checks["split_channel_cp"] = (
        choi_minimum >= limits.choi_minimum_eigenvalue
    )
    checks["split_channel_tp"] = choi_tp <= limits.choi_tp_frobenius
    checks["split_channel_hermitian"] = (
        choi_hermiticity <= limits.choi_tp_frobenius
    )

    simulator = Phase9BackendBSimulator(base)
    initial, evaluator = simulator.initialize_logical("0")
    idle = diagnostic_action_word_b("IDLE")
    x_action = diagnostic_action_word_b("X")
    reset_action = diagnostic_action_word_b("RESET")
    record = backend_b_random_record(
        seed=701,
        round_index=0,
        iq_samples=base.iq_samples,
    )
    normal = simulator.step(initial, idle, record, evaluator=evaluator)
    data = _diagnostics(normal.state.joint_density)
    trace_error = abs(data["trace_real"] - 1.0) + abs(data["trace_imag"])
    metrics.update(
        {
            "full_round_trace_error": trace_error,
            "full_round_minimum_eigenvalue": data["minimum_eigenvalue"],
            "full_round_hermiticity_frobenius": data[
                "hermiticity_frobenius"
            ],
            "measurement_completeness": simulator.measurement_completeness_error(),
            "reset_completeness": simulator.reset_completeness_error(),
            "maximum_channel_completeness_error": max(
                simulator.channel_completeness_errors(0.03).values()
            ),
        }
    )
    checks["full_round_trace"] = trace_error <= limits.full_round_trace_error
    checks["full_round_positive"] = (
        data["minimum_eigenvalue"] >= limits.full_round_minimum_eigenvalue
    )
    checks["full_round_hermitian"] = (
        data["hermiticity_frobenius"] <= limits.full_round_trace_error
    )
    checks["analytic_channels_complete"] = (
        metrics["maximum_channel_completeness_error"]
        <= limits.instrument_completeness
    )
    checks["measurement_instrument_complete"] = (
        metrics["measurement_completeness"]
        <= limits.instrument_completeness
    )
    checks["reset_instrument_complete"] = (
        metrics["reset_completeness"] <= limits.instrument_completeness
    )

    # Closed-form references: pure loss mean photon and qutrit e relaxation.
    loss_config = _noise_free(
        base,
        oscillator_loss_rate=0.37,
        action_duration=0.06,
    )
    loss_simulator = Phase9BackendBSimulator(loss_config)
    ket_two = np.zeros(loss_config.cutoff, dtype=np.complex128)
    ket_two[2] = 1.0
    loss_state = loss_simulator.initialize_fock(oscillator_ket=ket_two)
    loss_time = 0.23
    loss_output = loss_simulator._noise_channels(
        loss_state.joint_density,
        loss_time,
    )
    loss_oscillator = loss_simulator.oscillator_density(loss_output)
    loss_mean = float(np.trace(loss_oscillator @ loss_simulator.number).real)
    loss_expected = 2.0 * exp(-0.37 * loss_time)
    loss_error = abs(loss_mean - loss_expected)
    relaxation_config = _noise_free(
        base,
        ancilla_ge_relax_rate=0.41,
    )
    relaxation_simulator = Phase9BackendBSimulator(relaxation_config)
    relaxation_state = relaxation_simulator.initialize_fock(
        ancilla_state="e"
    )
    relaxation_time = 0.19
    relaxation_output = relaxation_simulator._noise_channels(
        relaxation_state.joint_density,
        relaxation_time,
    )
    e_population = relaxation_simulator.level_probabilities(
        relaxation_output
    )[1]
    e_expected = exp(-0.41 * relaxation_time)
    relaxation_error = abs(e_population - e_expected)
    metrics["analytic_loss_mean_error"] = loss_error
    metrics["analytic_relaxation_population_error"] = relaxation_error
    checks["pure_loss_closed_form"] = (
        loss_error <= limits.analytic_loss_mean_error
    )
    checks["relaxation_closed_form"] = (
        relaxation_error <= limits.analytic_relaxation_error
    )

    ideal_config = _noise_free(base, split_steps_per_segment=64)
    ideal = Phase9BackendBSimulator(ideal_config)
    vacuum = ideal.initialize_fock()
    zero = ideal.step(vacuum, idle, _zero_record(ideal_config))
    zero_distance = _trace_distance(
        vacuum.joint_density,
        zero.state.joint_density,
    )
    acted = ideal.step(vacuum, x_action, _zero_record(ideal_config))
    alpha = ideal._action_alpha(x_action)
    displacement = expm(alpha * ideal.adag - alpha.conjugate() * ideal.a)
    expected = np.kron(
        displacement
        @ ideal.oscillator_density(vacuum.joint_density)
        @ displacement.conj().T,
        ideal.level_projectors[0],
    )
    ideal_distance = _trace_distance(expected, acted.state.joint_density)
    metrics["zero_noise_idle_trace_distance"] = zero_distance
    metrics["ideal_action_trace_distance"] = ideal_distance
    checks["zero_noise_idle_limit"] = zero_distance <= 1.0e-12
    checks["ideal_action_limit"] = (
        ideal_distance <= limits.ideal_action_trace_distance
    )

    success_config = _noise_free(
        base,
        reset_success_e=1.0,
        reset_success_f=1.0,
    )
    success_simulator = Phase9BackendBSimulator(success_config)
    f_initial = success_simulator.initialize_fock(ancilla_state="f")
    success = success_simulator.step(
        f_initial,
        reset_action,
        _zero_record(success_config),
    )
    success_g = success_simulator.level_probabilities(
        success.state.joint_density
    )[0]
    failure_config = _noise_free(
        base,
        reset_success_e=0.0,
        reset_success_f=0.0,
    )
    failure_simulator = Phase9BackendBSimulator(failure_config)
    failure_initial = failure_simulator.initialize_fock(ancilla_state="f")
    failed = failure_simulator.step(
        failure_initial,
        reset_action,
        _zero_record(failure_config),
    )
    persisted = failure_simulator.step(
        failure_initial,
        idle,
        _zero_record(failure_config),
    )
    failed_f = failure_simulator.level_probabilities(
        failed.state.joint_density
    )[2]
    persisted_f = failure_simulator.level_probabilities(
        persisted.state.joint_density
    )[2]
    metrics["large_reset_g_probability"] = success_g
    metrics["failed_reset_f_probability"] = failed_f
    metrics["no_reset_f_probability"] = persisted_f
    checks["large_reset_limit"] = (
        success_g >= limits.limit_population_minimum
        and success.truth.reset_hidden_outcome == "success"
    )
    checks["failed_reset_preserves_f"] = (
        failed_f >= limits.limit_population_minimum
        and failed.truth.reset_hidden_outcome == "failure"
    )
    checks["f_state_persistence"] = (
        persisted_f >= limits.limit_population_minimum
    )

    measurement_config = _noise_free(
        base,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
    )
    measurement_simulator = Phase9BackendBSimulator(measurement_config)
    plus_ge = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / sqrt(2.0)
    coherent = measurement_simulator.initialize_fock(
        ancilla_state=plus_ge
    )
    measured = measurement_simulator.step(
        coherent,
        idle,
        _zero_record(measurement_config),
    )
    before_coherence = abs(
        measurement_simulator.ancilla_density(coherent.joint_density)[0, 1]
    )
    after_coherence = abs(
        measurement_simulator.ancilla_density(
            measured.state.joint_density
        )[0, 1]
    )
    posterior_peak = max(measured.observation.posterior_levels)
    metrics["measurement_coherence_ratio"] = float(
        after_coherence / before_coherence
    )
    metrics["measurement_posterior_peak"] = posterior_peak
    checks["iq_kraus_backaction"] = (
        posterior_peak > limits.measurement_posterior_peak
        and after_coherence < before_coherence * 1.0e-3
    )

    syndrome_config = _noise_free(
        base,
        ramsey_angle=pi / 2.0,
        ramsey_pulse_duration=0.03,
        sense_duration=0.8,
        dispersive_chi=1.0,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
        split_steps_per_segment=32,
    )
    syndrome = Phase9BackendBSimulator(syndrome_config)
    ket_zero = np.zeros(syndrome.cutoff, dtype=np.complex128)
    ket_one = np.zeros(syndrome.cutoff, dtype=np.complex128)
    ket_zero[0] = 1.0
    ket_one[1] = 1.0
    syndrome_record = _zero_record(syndrome_config)
    zero_result = syndrome.step(
        syndrome.initialize_fock(oscillator_ket=ket_zero),
        idle,
        syndrome_record,
    )
    one_result = syndrome.step(
        syndrome.initialize_fock(oscillator_ket=ket_one),
        idle,
        syndrome_record,
    )
    level_tv = 0.5 * float(
        np.sum(
            np.abs(
                np.asarray(zero_result.truth.pre_measurement_levels)
                - np.asarray(one_result.truth.pre_measurement_levels)
            )
        )
    )
    superposition = (ket_zero + ket_one) / sqrt(2.0)
    super_state = syndrome.initialize_fock(oscillator_ket=superposition)
    super_result = syndrome.step(
        super_state,
        idle,
        syndrome_record,
    )
    syndrome_backaction = _trace_distance(
        syndrome.oscillator_density(super_state.joint_density),
        syndrome.oscillator_density(super_result.state.joint_density),
    )
    metrics["syndrome_fock0_vs_fock1_level_tv"] = level_tv
    metrics["syndrome_oscillator_backaction_trace_distance"] = (
        syndrome_backaction
    )
    checks["ramsey_syndrome_state_dependence"] = (
        level_tv > limits.syndrome_state_dependence_minimum
    )
    checks["syndrome_backacts_on_oscillator"] = (
        syndrome_backaction > limits.syndrome_backaction_minimum
    )

    leakage_config = _noise_free(
        base,
        action_leakage_coupling=0.8,
        action_duration=0.1,
        split_steps_per_segment=64,
    )
    leakage = Phase9BackendBSimulator(leakage_config)
    e_initial = leakage.initialize_fock(ancilla_state="e")
    leakage_idle = leakage.step(
        e_initial,
        idle,
        _zero_record(leakage_config),
    )
    leakage_x = leakage.step(
        e_initial,
        x_action,
        _zero_record(leakage_config),
    )
    f_difference = (
        leakage.level_probabilities(leakage_x.state.joint_density)[2]
        - leakage.level_probabilities(leakage_idle.state.joint_density)[2]
    )
    metrics["action_induced_f_population_difference"] = f_difference
    checks["action_induces_f_population"] = (
        f_difference > limits.action_f_population_minimum
    )

    intervention_config = replace(
        base,
        ramsey_angle=0.0,
        sense_duration=0.0,
        iq_centers=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    intervention = Phase9BackendBSimulator(intervention_config)
    intervention_initial = intervention.initialize_fock()
    common_record = backend_b_random_record(
        seed=801,
        round_index=0,
        iq_samples=intervention_config.iq_samples,
    )
    idle_result = intervention.step(
        intervention_initial,
        idle,
        common_record,
    )
    x_result = intervention.step(
        intervention_initial,
        x_action,
        common_record,
    )
    state_distance = _trace_distance(
        idle_result.state.joint_density,
        x_result.state.joint_density,
    )
    drift_distance = float(
        np.linalg.norm(
            idle_result.state.drift.vector()
            - x_result.state.drift.vector()
        )
    )
    metrics["action_intervention_state_trace_distance"] = state_distance
    metrics["action_intervention_drift_l2"] = drift_distance
    metrics["action_intervention_shared_random_record"] = (
        idle_result.random_record == x_result.random_record
    )
    checks["action_changes_quantum_state"] = (
        state_distance > limits.action_state_distance_minimum
    )
    checks["action_changes_drift"] = (
        drift_distance > limits.action_drift_minimum
    )
    checks["common_random_record_intervention"] = (
        idle_result.random_record == x_result.random_record
    )

    actions = (idle, x_action, idle)
    replay_one = simulator.simulate(initial, actions, seed=901, evaluator=evaluator)
    replay_two = simulator.simulate(initial, actions, seed=901, evaluator=evaluator)
    other = simulator.simulate(initial, actions, seed=902, evaluator=evaluator)
    replay_density_error = float(
        np.max(
            np.abs(
                replay_one.final_state.joint_density
                - replay_two.final_state.joint_density
            )
        )
    )
    replay_iq_error = max(
        float(
            np.max(
                np.abs(left.observation.iq_i - right.observation.iq_i)
            )
        )
        for left, right in zip(replay_one.rounds, replay_two.rounds)
    )
    seed_difference = float(
        np.max(
            np.abs(
                replay_one.rounds[0].observation.iq_i
                - other.rounds[0].observation.iq_i
            )
        )
    )
    metrics["seed_replay_density_error"] = replay_density_error
    metrics["seed_replay_iq_error"] = replay_iq_error
    metrics["different_seed_iq_difference"] = seed_difference
    checks["seed_determinism"] = (
        replay_density_error == 0.0 and replay_iq_error == 0.0
    )
    checks["seed_sensitivity"] = (
        seed_difference > limits.rng_sensitivity_minimum
    )

    convergence_base = _noise_free(
        base,
        dispersive_chi=0.23,
        self_kerr=0.015,
        ramsey_angle=0.0,
        sense_duration=0.0,
        drift_action_kick=0.0,
    )
    convergence_drift = BackendBDrift(drive_q=0.07, drive_p=-0.04)
    convergence_record = _zero_record(convergence_base)
    outputs: dict[int, ComplexMatrix] = {}
    for steps in (8, 16, 32):
        active_config = replace(
            convergence_base,
            split_steps_per_segment=steps,
        )
        active = Phase9BackendBSimulator(active_config)
        active_state = active.initialize_fock(drift=convergence_drift)
        outputs[steps] = active.step(
            active_state,
            x_action,
            convergence_record,
        ).state.joint_density
    distance_8_16 = _trace_distance(outputs[8], outputs[16])
    distance_16_32 = _trace_distance(outputs[16], outputs[32])
    ratio = distance_16_32 / distance_8_16
    metrics["split_8_vs_16_trace_distance"] = distance_8_16
    metrics["split_16_vs_32_trace_distance"] = distance_16_32
    metrics["split_error_ratio"] = ratio
    checks["split_step_convergence"] = (
        distance_16_32 <= limits.split_distance
        and ratio <= limits.split_ratio
    )

    cutoff_outputs: dict[int, ComplexMatrix] = {}
    for cutoff in (8, 12):
        active_config = replace(
            convergence_base,
            cutoff=cutoff,
            split_steps_per_segment=24,
        )
        active = Phase9BackendBSimulator(active_config)
        active_state = active.initialize_fock(drift=convergence_drift)
        cutoff_outputs[cutoff] = active.oscillator_density(
            active.step(
                active_state,
                x_action,
                _zero_record(active_config),
            ).state.joint_density
        )
    cutoff_distance = _embedded_distance(
        cutoff_outputs[8],
        cutoff_outputs[12],
    )
    metrics["fock_cutoff_8_vs_12_trace_distance"] = cutoff_distance
    checks["fock_cutoff_convergence"] = (
        cutoff_distance <= limits.cutoff_distance
    )

    fidelities: list[float] = []
    for label in ("0", "1", "+", "-", "+i", "-i"):
        state, truth = simulator.initialize_logical(label)
        fidelities.append(simulator.logical_record(state, truth).target_fidelity)
    minimum_fidelity = min(fidelities)
    metrics["six_state_initial_minimum_fidelity"] = minimum_fidelity
    checks["independent_six_state_logical_projection"] = (
        minimum_fidelity >= limits.six_state_initial_fidelity
    )

    claim_state = {
        "backend_a_b_agreement": None,
        "dual_backend_qualified": None,
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
        "QUALIFIED_BACKEND_B_ONLY"
        if checks and all(checks.values())
        else "NO_GO_BACKEND_B_QUALIFICATION"
    )
    return BackendBQualification(
        config_sha256=base.semantic_sha256(),
        metrics=metrics,
        checks=checks,
        claim_state=claim_state,
        verdict=verdict,
    )


__all__ = [
    "ANCILLA_LEVELS",
    "BACKEND_B_ID",
    "BACKEND_B_LIKELIHOOD_ID",
    "BACKEND_B_LOGICAL_ID",
    "BACKEND_B_RNG_ID",
    "BACKEND_B_SCOPE",
    "BACKEND_B_SOLVER_ID",
    "MAX_SUPPORTED_CUTOFF",
    "MAX_EXACT_CHOI_CUTOFF",
    "BackendBConfig",
    "BackendBDrift",
    "BackendBEvaluator",
    "BackendBLogical",
    "BackendBObservation",
    "BackendBQualification",
    "BackendBRandomRecord",
    "BackendBRound",
    "BackendBState",
    "BackendBQualificationThresholds",
    "BackendBTrajectory",
    "BackendBTruth",
    "Phase9BackendBSimulator",
    "backend_b_random_record",
    "diagnostic_action_word_b",
    "run_backend_b_qualification",
]
