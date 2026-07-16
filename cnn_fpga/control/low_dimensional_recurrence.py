"""Pure-NumPy low-dimensional exponential recurrence student.

The state update is interpretable and outcome-specific::

    z[t+1] = a[m] * z[t] + (1 - a[m]) * z_inf[m]

The fifteen raw control coordinates are an affine head over ``z``.  Physical
residuals are always mapped through the canonical hard bounds and ``tanh``.
Leakage or any health failure resets the state and returns exact zero residual.
This online module has no torch, physics simulator, teacher, or hidden-truth
dependency.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


CONTROL_PARAMETER_NAMES = (
    "layer1_phi",
    "layer1_theta",
    "layer1_beta_real",
    "layer1_beta_imag",
    "layer2_phi",
    "layer2_theta",
    "layer2_beta_real",
    "layer2_beta_imag",
    "layer3_phi",
    "layer3_theta",
    "layer3_beta_real",
    "layer3_beta_imag",
    "layer4_phi",
    "layer4_theta",
    "virtual_rotation",
)
RESIDUAL_BOUNDS = (2.0,) * 14 + (1.0,)
ONLINE_INPUT_FIELDS = (
    "cycle_index",
    "observed_outcome",
    "valid",
    "crc_ok",
    "parameter_fresh",
    "deadline_ok",
)
SCHEMA_VERSION = "t4.4.3-low-dimensional-exponential-student-v1"
ONLINE_SCOPE = "observed_ge_health_only_low_dimensional_exponential_candidate"


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _vector(values: Any, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return tuple(float(value) for value in array)


def _matrix(values: Any, shape: tuple[int, int], name: str) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}")
    return tuple(tuple(float(value) for value in row) for row in array)


def _sha256_string(value: Any, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


@dataclass(frozen=True)
class LowDimensionalResourceProfile:
    state_dimension: int
    stored_trainable_scalars: int
    persistent_state_scalars: int
    multiply_adds_per_healthy_step: int
    nonlinearities_per_healthy_step: int = 15
    stored_scalar_bytes_float32: int = 0
    target_latency_cycles: None = None
    rtl_measured: bool = False
    board_measured: bool = False

    @classmethod
    def exact(cls, state_dimension: int) -> "LowDimensionalResourceProfile":
        dimension = _integer(state_dimension, "state_dimension", 1)
        scalars = 5 * dimension + 15 * dimension + 15
        return cls(
            state_dimension=dimension,
            stored_trainable_scalars=scalars,
            persistent_state_scalars=dimension,
            multiply_adds_per_healthy_step=18 * dimension + 15,
            stored_scalar_bytes_float32=4 * scalars,
        )

    def __post_init__(self) -> None:
        dimension = _integer(self.state_dimension, "state_dimension", 1)
        expected_scalars = 20 * dimension + 15
        expected = {
            "stored_trainable_scalars": expected_scalars,
            "persistent_state_scalars": dimension,
            "multiply_adds_per_healthy_step": 18 * dimension + 15,
            "nonlinearities_per_healthy_step": 15,
            "stored_scalar_bytes_float32": 4 * expected_scalars,
        }
        for field, value in expected.items():
            if getattr(self, field) != value:
                raise ValueError(f"resource field {field} does not match exact recurrence")
        if self.target_latency_cycles is not None:
            raise ValueError("latency remains null until RTL/synthesis evidence")
        if self.rtl_measured or self.board_measured:
            raise ValueError("software student cannot claim RTL or board measurement")


@dataclass(frozen=True)
class LowDimensionalRecurrenceArtifact:
    state_dimension: int
    initial_state: tuple[float, ...]
    outcome_decays: tuple[tuple[float, ...], ...]
    outcome_saturations: tuple[tuple[float, ...], ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]
    residual_bounds: tuple[float, ...]
    teacher_checkpoint_sha256: str
    teacher_state_sha256: str
    teacher_analysis_sha256: str
    training_dataset_sha256: str
    validation_dataset_sha256: str
    selected_dimension: int
    selected_restart: int
    validation_mse: float
    artifact_sha256: str
    resource_profile: LowDimensionalResourceProfile
    schema_version: str = SCHEMA_VERSION
    online_scope: str = ONLINE_SCOPE

    @staticmethod
    def _payload(
        *,
        state_dimension: int,
        initial_state: Sequence[float],
        outcome_decays: Sequence[Sequence[float]],
        outcome_saturations: Sequence[Sequence[float]],
        output_weights: Sequence[Sequence[float]],
        output_bias: Sequence[float],
        residual_bounds: Sequence[float],
        teacher_checkpoint_sha256: str,
        teacher_state_sha256: str,
        teacher_analysis_sha256: str,
        training_dataset_sha256: str,
        validation_dataset_sha256: str,
        selected_dimension: int,
        selected_restart: int,
        validation_mse: float,
        resource_profile: LowDimensionalResourceProfile,
    ) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "online_scope": ONLINE_SCOPE,
            "control_parameter_names": list(CONTROL_PARAMETER_NAMES),
            "online_input_fields": list(ONLINE_INPUT_FIELDS),
            "state_dimension": int(state_dimension),
            "initial_state": list(initial_state),
            "outcome_decays": [list(row) for row in outcome_decays],
            "outcome_saturations": [list(row) for row in outcome_saturations],
            "output_weights": [list(row) for row in output_weights],
            "output_bias": list(output_bias),
            "residual_bounds": list(residual_bounds),
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            "teacher_state_sha256": teacher_state_sha256,
            "teacher_analysis_sha256": teacher_analysis_sha256,
            "training_dataset_sha256": training_dataset_sha256,
            "validation_dataset_sha256": validation_dataset_sha256,
            "selected_dimension": int(selected_dimension),
            "selected_restart": int(selected_restart),
            "validation_mse": float(validation_mse),
            "resource_profile": asdict(resource_profile),
            "offline_teacher_object_embedded": False,
            "torch_runtime_dependency": False,
            "leakage_policy": "reset_state_and_exact_zero_residual",
        }

    @classmethod
    def create(
        cls,
        *,
        initial_state: Sequence[float],
        outcome_decays: Sequence[Sequence[float]],
        outcome_saturations: Sequence[Sequence[float]],
        output_weights: Sequence[Sequence[float]],
        output_bias: Sequence[float],
        teacher_checkpoint_sha256: str,
        teacher_state_sha256: str,
        teacher_analysis_sha256: str,
        training_dataset_sha256: str,
        validation_dataset_sha256: str,
        selected_dimension: int,
        selected_restart: int,
        validation_mse: float,
    ) -> "LowDimensionalRecurrenceArtifact":
        dimension = _integer(selected_dimension, "selected_dimension", 1)
        initial = _vector(initial_state, dimension, "initial_state")
        decays = _matrix(outcome_decays, (2, dimension), "outcome_decays")
        saturations = _matrix(
            outcome_saturations, (2, dimension), "outcome_saturations"
        )
        weights = _matrix(
            output_weights, (len(CONTROL_PARAMETER_NAMES), dimension), "output_weights"
        )
        bias = _vector(output_bias, len(CONTROL_PARAMETER_NAMES), "output_bias")
        bounds = tuple(RESIDUAL_BOUNDS)
        decay_array = np.asarray(decays)
        if np.any(decay_array <= 0.0) or np.any(decay_array >= 1.0):
            raise ValueError("outcome decays must lie strictly inside (0,1)")
        mse = float(validation_mse)
        if not isfinite(mse) or mse < 0.0:
            raise ValueError("validation_mse must be finite and nonnegative")
        hashes = tuple(
            _sha256_string(value, name)
            for value, name in (
                (teacher_checkpoint_sha256, "teacher_checkpoint_sha256"),
                (teacher_state_sha256, "teacher_state_sha256"),
                (teacher_analysis_sha256, "teacher_analysis_sha256"),
                (training_dataset_sha256, "training_dataset_sha256"),
                (validation_dataset_sha256, "validation_dataset_sha256"),
            )
        )
        restart = _integer(selected_restart, "selected_restart")
        resource = LowDimensionalResourceProfile.exact(dimension)
        payload = cls._payload(
            state_dimension=dimension,
            initial_state=initial,
            outcome_decays=decays,
            outcome_saturations=saturations,
            output_weights=weights,
            output_bias=bias,
            residual_bounds=bounds,
            teacher_checkpoint_sha256=hashes[0],
            teacher_state_sha256=hashes[1],
            teacher_analysis_sha256=hashes[2],
            training_dataset_sha256=hashes[3],
            validation_dataset_sha256=hashes[4],
            selected_dimension=dimension,
            selected_restart=restart,
            validation_mse=mse,
            resource_profile=resource,
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(
            state_dimension=dimension,
            initial_state=initial,
            outcome_decays=decays,
            outcome_saturations=saturations,
            output_weights=weights,
            output_bias=bias,
            residual_bounds=bounds,
            teacher_checkpoint_sha256=hashes[0],
            teacher_state_sha256=hashes[1],
            teacher_analysis_sha256=hashes[2],
            training_dataset_sha256=hashes[3],
            validation_dataset_sha256=hashes[4],
            selected_dimension=dimension,
            selected_restart=restart,
            validation_mse=mse,
            artifact_sha256=hashlib.sha256(canonical).hexdigest(),
            resource_profile=resource,
        )

    def __post_init__(self) -> None:
        dimension = _integer(self.state_dimension, "state_dimension", 1)
        if self.selected_dimension != dimension:
            raise ValueError("selected_dimension must equal state_dimension")
        object.__setattr__(self, "initial_state", _vector(self.initial_state, dimension, "initial_state"))
        object.__setattr__(
            self,
            "outcome_decays",
            _matrix(self.outcome_decays, (2, dimension), "outcome_decays"),
        )
        object.__setattr__(
            self,
            "outcome_saturations",
            _matrix(self.outcome_saturations, (2, dimension), "outcome_saturations"),
        )
        object.__setattr__(
            self,
            "output_weights",
            _matrix(
                self.output_weights,
                (len(CONTROL_PARAMETER_NAMES), dimension),
                "output_weights",
            ),
        )
        object.__setattr__(
            self,
            "output_bias",
            _vector(self.output_bias, len(CONTROL_PARAMETER_NAMES), "output_bias"),
        )
        object.__setattr__(
            self,
            "residual_bounds",
            _vector(self.residual_bounds, len(CONTROL_PARAMETER_NAMES), "residual_bounds"),
        )
        if self.residual_bounds != RESIDUAL_BOUNDS:
            raise ValueError("residual_bounds must preserve the canonical action box")
        if np.any(np.asarray(self.outcome_decays) <= 0.0) or np.any(
            np.asarray(self.outcome_decays) >= 1.0
        ):
            raise ValueError("outcome decays must lie strictly inside (0,1)")
        for name in (
            "teacher_checkpoint_sha256",
            "teacher_state_sha256",
            "teacher_analysis_sha256",
            "training_dataset_sha256",
            "validation_dataset_sha256",
            "artifact_sha256",
        ):
            _sha256_string(getattr(self, name), name)
        _integer(self.selected_restart, "selected_restart")
        mse = float(self.validation_mse)
        if not isfinite(mse) or mse < 0.0:
            raise ValueError("validation_mse must be finite and nonnegative")
        object.__setattr__(self, "validation_mse", mse)
        if self.resource_profile != LowDimensionalResourceProfile.exact(dimension):
            raise ValueError("resource_profile does not match exact recurrence")
        if self.schema_version != SCHEMA_VERSION or self.online_scope != ONLINE_SCOPE:
            raise ValueError("student schema/scope is invalid")
        payload = self._payload(
            state_dimension=dimension,
            initial_state=self.initial_state,
            outcome_decays=self.outcome_decays,
            outcome_saturations=self.outcome_saturations,
            output_weights=self.output_weights,
            output_bias=self.output_bias,
            residual_bounds=self.residual_bounds,
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            teacher_state_sha256=self.teacher_state_sha256,
            teacher_analysis_sha256=self.teacher_analysis_sha256,
            training_dataset_sha256=self.training_dataset_sha256,
            validation_dataset_sha256=self.validation_dataset_sha256,
            selected_dimension=self.selected_dimension,
            selected_restart=self.selected_restart,
            validation_mse=self.validation_mse,
            resource_profile=self.resource_profile,
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != self.artifact_sha256:
            raise ValueError("student artifact hash mismatch")

    def to_dict(self) -> dict[str, object]:
        payload = self._payload(
            state_dimension=self.state_dimension,
            initial_state=self.initial_state,
            outcome_decays=self.outcome_decays,
            outcome_saturations=self.outcome_saturations,
            output_weights=self.output_weights,
            output_bias=self.output_bias,
            residual_bounds=self.residual_bounds,
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            teacher_state_sha256=self.teacher_state_sha256,
            teacher_analysis_sha256=self.teacher_analysis_sha256,
            training_dataset_sha256=self.training_dataset_sha256,
            validation_dataset_sha256=self.validation_dataset_sha256,
            selected_dimension=self.selected_dimension,
            selected_restart=self.selected_restart,
            validation_mse=self.validation_mse,
            resource_profile=self.resource_profile,
        )
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LowDimensionalRecurrenceArtifact":
        if not isinstance(payload, Mapping):
            raise TypeError("student artifact payload must be a mapping")
        if tuple(payload.get("control_parameter_names", ())) != CONTROL_PARAMETER_NAMES:
            raise ValueError("control parameter order mismatch")
        if tuple(payload.get("online_input_fields", ())) != ONLINE_INPUT_FIELDS:
            raise ValueError("online input fields mismatch")
        if payload.get("offline_teacher_object_embedded") is not False:
            raise ValueError("online artifact must not embed the teacher")
        if payload.get("torch_runtime_dependency") is not False:
            raise ValueError("online artifact must not depend on torch")
        if payload.get("leakage_policy") != "reset_state_and_exact_zero_residual":
            raise ValueError("leakage policy mismatch")
        resource = payload.get("resource_profile")
        if not isinstance(resource, Mapping):
            raise ValueError("resource_profile must be a mapping")
        return cls(
            state_dimension=payload.get("state_dimension", 0),  # type: ignore[arg-type]
            initial_state=payload.get("initial_state", ()),  # type: ignore[arg-type]
            outcome_decays=payload.get("outcome_decays", ()),  # type: ignore[arg-type]
            outcome_saturations=payload.get("outcome_saturations", ()),  # type: ignore[arg-type]
            output_weights=payload.get("output_weights", ()),  # type: ignore[arg-type]
            output_bias=payload.get("output_bias", ()),  # type: ignore[arg-type]
            residual_bounds=payload.get("residual_bounds", ()),  # type: ignore[arg-type]
            teacher_checkpoint_sha256=str(payload.get("teacher_checkpoint_sha256", "")),
            teacher_state_sha256=str(payload.get("teacher_state_sha256", "")),
            teacher_analysis_sha256=str(payload.get("teacher_analysis_sha256", "")),
            training_dataset_sha256=str(payload.get("training_dataset_sha256", "")),
            validation_dataset_sha256=str(payload.get("validation_dataset_sha256", "")),
            selected_dimension=payload.get("selected_dimension", 0),  # type: ignore[arg-type]
            selected_restart=payload.get("selected_restart", -1),  # type: ignore[arg-type]
            validation_mse=payload.get("validation_mse", -1.0),  # type: ignore[arg-type]
            artifact_sha256=str(payload.get("artifact_sha256", "")),
            resource_profile=LowDimensionalResourceProfile(**dict(resource)),
            schema_version=str(payload.get("schema_version", "")),
            online_scope=str(payload.get("online_scope", "")),
        )


@dataclass(frozen=True)
class LowDimensionalObservation:
    cycle_index: int
    observed_outcome: str
    valid: bool = True
    crc_ok: bool = True
    parameter_fresh: bool = True
    deadline_ok: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index"))
        if self.observed_outcome not in {"g", "e", "leakage"}:
            raise ValueError("observed_outcome must be g, e, or leakage")
        for name in ("valid", "crc_ok", "parameter_fresh", "deadline_ok"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a strict boolean")


@dataclass(frozen=True)
class LowDimensionalDecision:
    cycle_index: int
    raw_control_residual: tuple[float, ...]
    physical_control_residual: tuple[float, ...]
    state: tuple[float, ...]
    used_safe_baseline: bool
    reason: str
    student_artifact_sha256: str


class LowDimensionalRecurrenceStudent:
    def __init__(self, artifact: LowDimensionalRecurrenceArtifact) -> None:
        if not isinstance(artifact, LowDimensionalRecurrenceArtifact):
            raise TypeError("student requires LowDimensionalRecurrenceArtifact")
        self.artifact = artifact
        self._initial = np.asarray(artifact.initial_state, dtype=np.float64)
        self._decays = np.asarray(artifact.outcome_decays, dtype=np.float64)
        self._saturations = np.asarray(artifact.outcome_saturations, dtype=np.float64)
        self._weights = np.asarray(artifact.output_weights, dtype=np.float64)
        self._bias = np.asarray(artifact.output_bias, dtype=np.float64)
        self._bounds = np.asarray(artifact.residual_bounds, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self._state = self._initial.copy()
        self._next_cycle = 0

    @property
    def state(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._state)

    def _decision(
        self, cycle_index: int, *, safe: bool, reason: str
    ) -> LowDimensionalDecision:
        if safe:
            raw = np.zeros(len(CONTROL_PARAMETER_NAMES), dtype=np.float64)
            residual = np.zeros_like(raw)
        else:
            raw = self._weights @ self._state + self._bias
            residual = self._bounds * np.tanh(raw)
        return LowDimensionalDecision(
            cycle_index=cycle_index,
            raw_control_residual=tuple(float(value) for value in raw),
            physical_control_residual=tuple(float(value) for value in residual),
            state=self.state,
            used_safe_baseline=safe,
            reason=reason,
            student_artifact_sha256=self.artifact.artifact_sha256,
        )

    def initial_decision(self) -> LowDimensionalDecision:
        return self._decision(
            0, safe=False, reason="low_dimensional_student_initial_state"
        )

    def step(self, observation: LowDimensionalObservation) -> LowDimensionalDecision:
        if not isinstance(observation, LowDimensionalObservation):
            raise TypeError("step accepts only LowDimensionalObservation")
        if observation.cycle_index != self._next_cycle:
            raise ValueError("observations must be contiguous and start at cycle zero")
        self._next_cycle += 1
        health_failures = [
            name
            for name in ("valid", "crc_ok", "parameter_fresh", "deadline_ok")
            if not getattr(observation, name)
        ]
        safe = observation.observed_outcome == "leakage" or bool(health_failures)
        if safe:
            self._state = self._initial.copy()
            reason = (
                "observed_leakage_reset_and_safe_baseline"
                if observation.observed_outcome == "leakage"
                else "health_gate_reset:" + "+".join(health_failures)
            )
            return self._decision(observation.cycle_index, safe=True, reason=reason)
        index = 0 if observation.observed_outcome == "g" else 1
        self._state = self._decays[index] * self._state + (
            1.0 - self._decays[index]
        ) * self._saturations[index]
        return self._decision(
            observation.cycle_index,
            safe=False,
            reason="low_dimensional_exponential_recurrence",
        )


def online_contract() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "schema_version": SCHEMA_VERSION,
            "online_scope": ONLINE_SCOPE,
            "input_fields": ONLINE_INPUT_FIELDS,
            "control_parameter_names": CONTROL_PARAMETER_NAMES,
            "residual_bounds": RESIDUAL_BOUNDS,
            "torch_runtime_dependency": False,
            "physics_runtime_dependency": False,
            "teacher_runtime_dependency": False,
            "safe_baseline": "reset state and exact zero physical residual",
            "target_latency_cycles": None,
            "rtl_measured": False,
            "board_measured": False,
        }
    )


__all__ = [
    "CONTROL_PARAMETER_NAMES",
    "LowDimensionalDecision",
    "LowDimensionalObservation",
    "LowDimensionalRecurrenceArtifact",
    "LowDimensionalRecurrenceStudent",
    "LowDimensionalResourceProfile",
    "ONLINE_INPUT_FIELDS",
    "ONLINE_SCOPE",
    "RESIDUAL_BOUNDS",
    "SCHEMA_VERSION",
    "online_contract",
]
