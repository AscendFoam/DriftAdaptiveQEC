"""Offline-teacher / deterministic-online-student separation contract.

The online module intentionally has no torch, simulator, hidden-state, oracle,
or teacher-model dependency.  It consumes one observed outcome plus explicit
health flags and either advances a compact recurrence or emits the registered
zero-residual safe baseline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Sequence

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
OBSERVED_OUTCOMES = ("g", "e", "leakage")
ONLINE_INPUT_FIELDS = (
    "cycle_index",
    "observed_outcome",
    "valid",
    "crc_ok",
    "parameter_fresh",
    "deadline_ok",
)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _vector(values: object, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(value) for value in array)


def _matrix(values: object, shape: tuple[int, int], name: str) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    return tuple(tuple(float(value) for value in row) for row in array)


@dataclass(frozen=True)
class StudentResourceProfile:
    stored_scalars: int = 105
    state_scalars: int = 15
    multiplications_per_healthy_step: int = 15
    additions_per_healthy_step: int = 30
    comparisons_per_step: int = 21
    parameter_bytes_float32: int = 420
    target_latency_cycles: int | None = None
    rtl_measured: bool = False
    board_measured: bool = False

    def __post_init__(self) -> None:
        expected = {
            "stored_scalars": 105,
            "state_scalars": 15,
            "multiplications_per_healthy_step": 15,
            "additions_per_healthy_step": 30,
            "comparisons_per_step": 21,
            "parameter_bytes_float32": 420,
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("student resource counts must match the exact recurrence")
        if self.target_latency_cycles is not None:
            raise ValueError("target latency remains null until RTL/synthesis evidence")
        if self.rtl_measured or self.board_measured:
            raise ValueError("software student cannot claim RTL or board measurement")


@dataclass(frozen=True)
class DistilledStudentArtifact:
    initial_state: tuple[float, ...]
    outcome_saturations: tuple[tuple[float, ...], ...]
    outcome_decays: tuple[tuple[float, ...], ...]
    leakage_safe_saturation: tuple[float, ...]
    leakage_safe_decay: tuple[float, ...]
    raw_clip: float
    teacher_checkpoint_sha256: str
    teacher_model_sha256s: tuple[str, ...]
    training_dataset_sha256: str
    validation_dataset_sha256: str
    selected_restart: int
    artifact_sha256: str
    resource_profile: StudentResourceProfile = StudentResourceProfile()
    schema_version: str = "t4.1.5-distilled-recurrence-student-v1"
    online_scope: str = "observed_outcome_health_only_deterministic_student_candidate"

    @staticmethod
    def _payload(
        *,
        initial_state: Sequence[float],
        outcome_saturations: Sequence[Sequence[float]],
        outcome_decays: Sequence[Sequence[float]],
        leakage_safe_saturation: Sequence[float],
        leakage_safe_decay: Sequence[float],
        raw_clip: float,
        teacher_checkpoint_sha256: str,
        teacher_model_sha256s: Sequence[str],
        training_dataset_sha256: str,
        validation_dataset_sha256: str,
        selected_restart: int,
        resource_profile: StudentResourceProfile,
    ) -> dict[str, object]:
        return {
            "schema_version": "t4.1.5-distilled-recurrence-student-v1",
            "online_scope": "observed_outcome_health_only_deterministic_student_candidate",
            "control_parameter_names": list(CONTROL_PARAMETER_NAMES),
            "initial_state": list(initial_state),
            "outcome_saturations": [list(row) for row in outcome_saturations],
            "outcome_decays": [list(row) for row in outcome_decays],
            "leakage_safe_saturation": list(leakage_safe_saturation),
            "leakage_safe_decay": list(leakage_safe_decay),
            "raw_clip": raw_clip,
            "teacher_checkpoint_sha256": teacher_checkpoint_sha256,
            "teacher_model_sha256s": list(teacher_model_sha256s),
            "training_dataset_sha256": training_dataset_sha256,
            "validation_dataset_sha256": validation_dataset_sha256,
            "selected_restart": selected_restart,
            "resource_profile": asdict(resource_profile),
            "online_input_fields": list(ONLINE_INPUT_FIELDS),
            "offline_teacher_object_embedded": False,
        }

    @classmethod
    def create(
        cls,
        *,
        initial_state: Sequence[float],
        outcome_saturations: Sequence[Sequence[float]],
        outcome_decays: Sequence[Sequence[float]],
        raw_clip: float,
        teacher_checkpoint_sha256: str,
        teacher_model_sha256s: Sequence[str],
        training_dataset_sha256: str,
        validation_dataset_sha256: str,
        selected_restart: int,
        leakage_safe_decay: float = 0.25,
    ) -> "DistilledStudentArtifact":
        size = len(CONTROL_PARAMETER_NAMES)
        initial = _vector(initial_state, size, "initial_state")
        saturations = _matrix(outcome_saturations, (2, size), "outcome_saturations")
        decays = _matrix(outcome_decays, (2, size), "outcome_decays")
        if np.any(np.asarray(decays) <= 0.0) or np.any(np.asarray(decays) >= 1.0):
            raise ValueError("learned g/e decays must lie strictly inside (0,1)")
        clip = float(raw_clip)
        if not isfinite(clip) or clip <= 0.0:
            raise ValueError("raw_clip must be finite and positive")
        safe_decay = float(leakage_safe_decay)
        if not isfinite(safe_decay) or not 0.0 < safe_decay < 1.0:
            raise ValueError("leakage_safe_decay must lie in (0,1)")
        safe_saturation = (0.0,) * size
        safe_decays = (safe_decay,) * size
        model_hashes = tuple(str(value) for value in teacher_model_sha256s)
        if len(model_hashes) < 3 or len(set(model_hashes)) != len(model_hashes):
            raise ValueError("at least three unique teacher model hashes are required")
        hashes = (
            str(teacher_checkpoint_sha256),
            *model_hashes,
            str(training_dataset_sha256),
            str(validation_dataset_sha256),
        )
        if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes):
            raise ValueError("all provenance hashes must be lowercase SHA-256")
        restart = _integer(selected_restart, "selected_restart")
        resource = StudentResourceProfile()
        payload = cls._payload(
            initial_state=initial,
            outcome_saturations=saturations,
            outcome_decays=decays,
            leakage_safe_saturation=safe_saturation,
            leakage_safe_decay=safe_decays,
            raw_clip=clip,
            teacher_checkpoint_sha256=hashes[0],
            teacher_model_sha256s=model_hashes,
            training_dataset_sha256=hashes[-2],
            validation_dataset_sha256=hashes[-1],
            selected_restart=restart,
            resource_profile=resource,
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return cls(
            initial_state=initial,
            outcome_saturations=saturations,
            outcome_decays=decays,
            leakage_safe_saturation=safe_saturation,
            leakage_safe_decay=safe_decays,
            raw_clip=clip,
            teacher_checkpoint_sha256=hashes[0],
            teacher_model_sha256s=model_hashes,
            training_dataset_sha256=hashes[-2],
            validation_dataset_sha256=hashes[-1],
            selected_restart=restart,
            artifact_sha256=hashlib.sha256(canonical).hexdigest(),
            resource_profile=resource,
        )

    def __post_init__(self) -> None:
        size = len(CONTROL_PARAMETER_NAMES)
        object.__setattr__(self, "initial_state", _vector(self.initial_state, size, "initial_state"))
        object.__setattr__(
            self,
            "outcome_saturations",
            _matrix(self.outcome_saturations, (2, size), "outcome_saturations"),
        )
        object.__setattr__(
            self,
            "outcome_decays",
            _matrix(self.outcome_decays, (2, size), "outcome_decays"),
        )
        object.__setattr__(
            self,
            "leakage_safe_saturation",
            _vector(self.leakage_safe_saturation, size, "leakage_safe_saturation"),
        )
        object.__setattr__(
            self,
            "leakage_safe_decay",
            _vector(self.leakage_safe_decay, size, "leakage_safe_decay"),
        )
        if np.any(np.asarray(self.outcome_decays) <= 0.0) or np.any(
            np.asarray(self.outcome_decays) >= 1.0
        ):
            raise ValueError("learned g/e decays must lie strictly inside (0,1)")
        if any(value != 0.0 for value in self.leakage_safe_saturation):
            raise ValueError("leakage safe saturation must be the zero-residual baseline")
        if (
            np.any(np.asarray(self.leakage_safe_decay) <= 0.0)
            or np.any(np.asarray(self.leakage_safe_decay) >= 1.0)
            or len(set(self.leakage_safe_decay)) != 1
        ):
            raise ValueError("leakage safe decay must be uniform and inside (0,1)")
        clip = float(self.raw_clip)
        if not isfinite(clip) or clip <= 0.0:
            raise ValueError("raw_clip must be finite and positive")
        object.__setattr__(self, "raw_clip", clip)
        _integer(self.selected_restart, "selected_restart")
        hashes = (
            self.teacher_checkpoint_sha256,
            *self.teacher_model_sha256s,
            self.training_dataset_sha256,
            self.validation_dataset_sha256,
            self.artifact_sha256,
        )
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in hashes
        ):
            raise ValueError("all provenance hashes must be lowercase SHA-256")
        if len(self.teacher_model_sha256s) < 3 or len(set(self.teacher_model_sha256s)) != len(
            self.teacher_model_sha256s
        ):
            raise ValueError("at least three unique teacher model hashes are required")
        if self.resource_profile != StudentResourceProfile():
            raise ValueError("resource profile does not match exact recurrence")
        if self.schema_version != "t4.1.5-distilled-recurrence-student-v1" or self.online_scope != (
            "observed_outcome_health_only_deterministic_student_candidate"
        ):
            raise ValueError("student schema/scope is invalid")
        payload = self._payload(
            initial_state=self.initial_state,
            outcome_saturations=self.outcome_saturations,
            outcome_decays=self.outcome_decays,
            leakage_safe_saturation=self.leakage_safe_saturation,
            leakage_safe_decay=self.leakage_safe_decay,
            raw_clip=self.raw_clip,
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            teacher_model_sha256s=self.teacher_model_sha256s,
            training_dataset_sha256=self.training_dataset_sha256,
            validation_dataset_sha256=self.validation_dataset_sha256,
            selected_restart=self.selected_restart,
            resource_profile=self.resource_profile,
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != self.artifact_sha256:
            raise ValueError("student artifact hash mismatch")

    def to_dict(self) -> dict[str, object]:
        payload = self._payload(
            initial_state=self.initial_state,
            outcome_saturations=self.outcome_saturations,
            outcome_decays=self.outcome_decays,
            leakage_safe_saturation=self.leakage_safe_saturation,
            leakage_safe_decay=self.leakage_safe_decay,
            raw_clip=self.raw_clip,
            teacher_checkpoint_sha256=self.teacher_checkpoint_sha256,
            teacher_model_sha256s=self.teacher_model_sha256s,
            training_dataset_sha256=self.training_dataset_sha256,
            validation_dataset_sha256=self.validation_dataset_sha256,
            selected_restart=self.selected_restart,
            resource_profile=self.resource_profile,
        )
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DistilledStudentArtifact":
        if not isinstance(payload, Mapping):
            raise TypeError("student artifact payload must be a mapping")
        if tuple(payload.get("control_parameter_names", ())) != CONTROL_PARAMETER_NAMES:
            raise ValueError("control parameter order does not match the online contract")
        if payload.get("offline_teacher_object_embedded") is not False:
            raise ValueError("online artifact must not embed an offline teacher object")
        if tuple(payload.get("online_input_fields", ())) != ONLINE_INPUT_FIELDS:
            raise ValueError("online input fields do not match the observed-health contract")
        resource_payload = payload.get("resource_profile")
        if not isinstance(resource_payload, Mapping):
            raise ValueError("resource_profile must be a mapping")
        return cls(
            initial_state=payload.get("initial_state", ()),  # type: ignore[arg-type]
            outcome_saturations=payload.get("outcome_saturations", ()),  # type: ignore[arg-type]
            outcome_decays=payload.get("outcome_decays", ()),  # type: ignore[arg-type]
            leakage_safe_saturation=payload.get("leakage_safe_saturation", ()),  # type: ignore[arg-type]
            leakage_safe_decay=payload.get("leakage_safe_decay", ()),  # type: ignore[arg-type]
            raw_clip=payload.get("raw_clip", 0.0),  # type: ignore[arg-type]
            teacher_checkpoint_sha256=str(payload.get("teacher_checkpoint_sha256", "")),
            teacher_model_sha256s=tuple(payload.get("teacher_model_sha256s", ())),  # type: ignore[arg-type]
            training_dataset_sha256=str(payload.get("training_dataset_sha256", "")),
            validation_dataset_sha256=str(payload.get("validation_dataset_sha256", "")),
            selected_restart=payload.get("selected_restart", -1),  # type: ignore[arg-type]
            artifact_sha256=str(payload.get("artifact_sha256", "")),
            resource_profile=StudentResourceProfile(**dict(resource_payload)),
            schema_version=str(payload.get("schema_version", "")),
            online_scope=str(payload.get("online_scope", "")),
        )


@dataclass(frozen=True)
class StudentObservation:
    cycle_index: int
    observed_outcome: str
    valid: bool = True
    crc_ok: bool = True
    parameter_fresh: bool = True
    deadline_ok: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index"))
        if self.observed_outcome not in OBSERVED_OUTCOMES:
            raise ValueError(f"observed_outcome must be one of {OBSERVED_OUTCOMES}")
        for name in ("valid", "crc_ok", "parameter_fresh", "deadline_ok"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean")
            object.__setattr__(self, name, bool(getattr(self, name)))


@dataclass(frozen=True)
class StudentDecision:
    cycle_index: int
    raw_control_residual: tuple[float, ...]
    used_safe_baseline: bool
    reason: str
    student_artifact_sha256: str
    deterministic: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index"))
        object.__setattr__(
            self,
            "raw_control_residual",
            _vector(self.raw_control_residual, len(CONTROL_PARAMETER_NAMES), "raw_control_residual"),
        )
        if not isinstance(self.used_safe_baseline, bool):
            raise TypeError("used_safe_baseline must be boolean")
        if not self.reason:
            raise ValueError("decision reason must be nonempty")
        if not self.deterministic:
            raise ValueError("online student decision must be deterministic")


class DistilledRecurrenceStudent:
    """105-scalar deterministic online recurrence with fail-safe baseline."""

    def __init__(self, artifact: DistilledStudentArtifact) -> None:
        if not isinstance(artifact, DistilledStudentArtifact):
            raise TypeError("online student requires DistilledStudentArtifact, not a teacher/model object")
        self.artifact = artifact
        self._initial = np.asarray(artifact.initial_state, dtype=np.float64)
        self._saturations = np.asarray(artifact.outcome_saturations, dtype=np.float64)
        self._decays = np.asarray(artifact.outcome_decays, dtype=np.float64)
        self._safe_saturation = np.asarray(artifact.leakage_safe_saturation, dtype=np.float64)
        self._safe_decay = np.asarray(artifact.leakage_safe_decay, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self._state = np.array(self._initial, copy=True)
        self._next_cycle = 0

    @property
    def state(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._state)

    def initial_decision(self) -> StudentDecision:
        return StudentDecision(
            cycle_index=0,
            raw_control_residual=tuple(
                float(value) for value in np.clip(self._state, -self.artifact.raw_clip, self.artifact.raw_clip)
            ),
            used_safe_baseline=False,
            reason="distilled_student_initial_state",
            student_artifact_sha256=self.artifact.artifact_sha256,
        )

    def step(self, observation: StudentObservation) -> StudentDecision:
        if not isinstance(observation, StudentObservation):
            raise TypeError("online step accepts only StudentObservation")
        if observation.cycle_index != self._next_cycle:
            raise ValueError("student observations must be contiguous and start at cycle zero")
        self._next_cycle += 1
        health_failures = [
            name
            for name in ("valid", "crc_ok", "parameter_fresh", "deadline_ok")
            if not getattr(observation, name)
        ]
        safe = observation.observed_outcome == "leakage" or bool(health_failures)
        if safe:
            self._state = self._safe_decay * self._state + (1.0 - self._safe_decay) * self._safe_saturation
            reason = (
                "observed_leakage_safe_baseline"
                if observation.observed_outcome == "leakage"
                else "health_gate:" + "+".join(health_failures)
            )
            output = self._safe_saturation
        else:
            index = 0 if observation.observed_outcome == "g" else 1
            self._state = self._decays[index] * self._state + (
                1.0 - self._decays[index]
            ) * self._saturations[index]
            reason = "distilled_student_recurrence"
            output = np.clip(self._state, -self.artifact.raw_clip, self.artifact.raw_clip)
        return StudentDecision(
            cycle_index=observation.cycle_index,
            raw_control_residual=tuple(float(value) for value in output),
            used_safe_baseline=safe,
            reason=reason,
            student_artifact_sha256=self.artifact.artifact_sha256,
        )


def online_contract() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "online_input_fields": ONLINE_INPUT_FIELDS,
            "control_parameter_names": CONTROL_PARAMETER_NAMES,
            "teacher_model_runtime_dependency": False,
            "torch_runtime_dependency": False,
            "simulator_truth_runtime_dependency": False,
            "safe_baseline": "zero_raw_residual_nominal_sbs_controls_downstream",
            "rtl_measured": False,
            "board_measured": False,
        }
    )


__all__ = [
    "CONTROL_PARAMETER_NAMES",
    "OBSERVED_OUTCOMES",
    "ONLINE_INPUT_FIELDS",
    "DistilledRecurrenceStudent",
    "DistilledStudentArtifact",
    "StudentDecision",
    "StudentObservation",
    "StudentResourceProfile",
    "online_contract",
]
