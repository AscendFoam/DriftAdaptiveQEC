"""Observed-only, causal experimental history schema for T4.1.2.

The builder joins real syndrome, action, soft-information and scheduler
producers after a cycle completes.  It never accepts simulator truth objects;
padding is represented by a separate mask so zero rows cannot masquerade as a
physical ``g/normal`` observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.runtime.run_length_fsm import (
    FSM_MODES,
    NORMAL,
    RunLengthFSMDecision,
)
from cnn_fpga.runtime.scheduler import SchedulerEvent
from physics.drift_processes import DriftState
from physics.ideal_gkp_decoder import llr_1d
from physics.syndrome_stream import ObservedSyndromeStep, SyndromeTruthStep


UPDATE_STATUSES = ("none", "staged", "committed", "conflict", "failed", "stale")
FORBIDDEN_INPUT_TOKENS = (
    "truth",
    "hidden",
    "targetregime",
    "targetlabel",
    "predictiontarget",
    "supervisiontarget",
    "oracle",
    "teacher",
    "label",
    "regime",
    "driftstate",
    "logical",
    "recoverydepth",
    "leakagekind",
    "leakagehazard",
    "outliercomponent",
    "channeldisplacement",
    "premeasurementshift",
    "physicalresidual",
)

FEATURE_GROUPS = MappingProxyType(
    {
        "analog_syndrome": ("analog_q", "analog_p"),
        "residual_syndrome": ("residual_q", "residual_p"),
        "observed_outcome": tuple(
            f"syndrome_{axis}_{outcome}"
            for axis in ("x", "z")
            for outcome in ("g", "e", "leakage")
        ),
        "quadrature_phase": (
            "phase_x_sin",
            "phase_x_cos",
            "phase_z_sin",
            "phase_z_cos",
        ),
        "recent_action": (
            "action_q",
            "action_p",
            *(f"action_mode_{mode}" for mode in FSM_MODES),
            "action_bank_switched",
            "action_local_safe",
            "action_bank_conflict",
        ),
        "soft_information": (
            "llr_q",
            "llr_p",
            "llr_q_saturated",
            "llr_p_saturated",
        ),
        "run_length": (
            "x_e_run",
            "z_e_run",
            "leakage_run",
            "x_e_run_saturated",
            "z_e_run_saturated",
            "leakage_run_saturated",
        ),
        "deadline_health": (
            "fast_deadline_ok",
            "slow_deadline_ok",
            "communication_available",
            "window_age_us",
            "window_age_valid",
        ),
        "parameter_update": (
            *(f"update_status_{status}" for status in UPDATE_STATUSES),
            "update_applied",
            "pending_update",
            "active_bank_version",
            "active_bank_version_saturated",
            "pending_window_count",
            "pending_window_count_saturated",
        ),
        "record_health": ("valid", "crc_ok"),
    }
)
FEATURE_NAMES = tuple(name for group in FEATURE_GROUPS.values() for name in group)
DEPLOYABLE_LLR_SOURCES = (
    "registered_observed_calibration",
    "online_observed_estimator",
    "fixed_deployment_calibration",
)
DEPLOYABLE_ACTION_SOURCES = (
    "run_length_fsm_decision",
    "registered_fast_path_decision",
    "explicit_neutral_action",
)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _pair(value: object, name: str, *, positive: bool = False) -> tuple[float, float]:
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must contain q and p") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain q and p")
    result = (_finite(values[0], f"{name}[0]"), _finite(values[1], f"{name}[1]"))
    if positive and any(item <= 0.0 for item in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def audit_mapping_for_information_leakage(value: object, *, path: str = "root") -> None:
    """Reject hidden/truth-bearing objects or nested field names, fail closed."""

    if isinstance(value, (SyndromeTruthStep, DriftState)):
        raise ValueError(f"forbidden truth object at {path}: {type(value).__name__}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            hit = next((token for token in FORBIDDEN_INPUT_TOKENS if token in normalized), None)
            if hit is not None:
                raise ValueError(f"forbidden information token {hit!r} at {path}.{key}")
            audit_mapping_for_information_leakage(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            audit_mapping_for_information_leakage(item, path=f"{path}[{index}]")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError(f"metadata array must be finite numeric data at {path}")
        return
    if isinstance(value, str):
        normalized = _normalized_key(value)
        hit = next((token for token in FORBIDDEN_INPUT_TOKENS if token in normalized), None)
        if hit is not None:
            raise ValueError(f"forbidden information token {hit!r} in string at {path}")
        return
    if value is None or isinstance(value, (bool, int, float, np.generic)):
        if isinstance(value, (float, np.floating)) and not isfinite(float(value)):
            raise ValueError(f"metadata scalar must be finite at {path}")
        return
    raise TypeError(f"unsupported metadata object at {path}: {type(value).__name__}")


@dataclass(frozen=True)
class ExperimentalHistoryConfig:
    history_cycles: int = 256
    llr_clip: float = 16.0
    run_length_clip: int = 255
    bank_version_clip: int = 65_535
    pending_window_clip: int = 255

    def __post_init__(self) -> None:
        object.__setattr__(self, "history_cycles", _integer(self.history_cycles, "history_cycles", 2))
        clip = _finite(self.llr_clip, "llr_clip")
        if clip <= 0.0:
            raise ValueError("llr_clip must be positive")
        object.__setattr__(self, "llr_clip", clip)
        for name in ("run_length_clip", "bank_version_clip", "pending_window_clip"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))


@dataclass(frozen=True)
class DeployableLLRContext:
    sigma: tuple[float, float]
    mean: tuple[float, float] = (0.0, 0.0)
    source: str = "registered_observed_calibration"
    estimator_version: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _pair(self.sigma, "sigma", positive=True))
        object.__setattr__(self, "mean", _pair(self.mean, "mean"))
        if self.source not in DEPLOYABLE_LLR_SOURCES:
            raise ValueError(f"source must be one of {DEPLOYABLE_LLR_SOURCES}")
        object.__setattr__(self, "estimator_version", _integer(self.estimator_version, "estimator_version"))


@dataclass(frozen=True)
class ObservedActionRecord:
    cycle_index: int
    correction: tuple[float, float]
    mode: str
    parameter_bank_version: int
    bank_switched: bool
    local_safe_rom_used: bool
    bank_conflict: bool
    source: str = "run_length_fsm_decision"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index"))
        object.__setattr__(self, "correction", _pair(self.correction, "correction"))
        if self.mode not in FSM_MODES:
            raise ValueError(f"mode must be one of {FSM_MODES}")
        if self.source not in DEPLOYABLE_ACTION_SOURCES:
            raise ValueError(f"source must be one of {DEPLOYABLE_ACTION_SOURCES}")
        object.__setattr__(self, "parameter_bank_version", _integer(self.parameter_bank_version, "parameter_bank_version"))
        for name in ("bank_switched", "local_safe_rom_used", "bank_conflict"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if self.bank_conflict and not self.local_safe_rom_used:
            raise ValueError("bank conflict must use local-safe ROM")

    @classmethod
    def from_fsm_decision(cls, decision: RunLengthFSMDecision) -> "ObservedActionRecord":
        if not isinstance(decision, RunLengthFSMDecision):
            raise TypeError("decision must be RunLengthFSMDecision")
        return cls(
            cycle_index=decision.cycle_index,
            correction=decision.correction,
            mode=decision.mode,
            parameter_bank_version=decision.parameter_bank_version,
            bank_switched=decision.bank_switched,
            local_safe_rom_used=decision.local_safe_rom_used,
            bank_conflict=decision.bank_conflict,
        )

    @classmethod
    def neutral(cls, cycle_index: int) -> "ObservedActionRecord":
        return cls(cycle_index, (0.0, 0.0), NORMAL, 0, False, False, False, "explicit_neutral_action")


@dataclass(frozen=True)
class HistoryRuntimeStatus:
    cycle_index: int
    fast_deadline_ok: bool
    slow_deadline_ok: bool
    communication_available: bool
    update_status: str
    update_applied: bool
    pending_update: bool
    active_bank_version: int
    pending_window_count: int
    window_age_us: float = 0.0
    window_age_valid: bool = False
    crc_ok: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index"))
        for name in (
            "fast_deadline_ok",
            "slow_deadline_ok",
            "communication_available",
            "update_applied",
            "pending_update",
            "window_age_valid",
            "crc_ok",
        ):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if self.update_status not in UPDATE_STATUSES:
            raise ValueError(f"update_status must be one of {UPDATE_STATUSES}")
        object.__setattr__(self, "active_bank_version", _integer(self.active_bank_version, "active_bank_version"))
        object.__setattr__(self, "pending_window_count", _integer(self.pending_window_count, "pending_window_count"))
        age = _finite(self.window_age_us, "window_age_us")
        if age < 0.0:
            raise ValueError("window_age_us must be non-negative")
        if not self.window_age_valid and age != 0.0:
            raise ValueError("window_age_us must be zero when not valid")
        object.__setattr__(self, "window_age_us", age)
        if self.update_status == "committed" and not self.update_applied:
            raise ValueError("committed status requires update_applied=true")


def runtime_status_from_scheduler(
    cycle_index: int,
    events: Sequence[SchedulerEvent],
    snapshot: Mapping[str, Any],
    *,
    crc_ok: bool = True,
) -> HistoryRuntimeStatus:
    cycle = _integer(cycle_index, "cycle_index")
    if not isinstance(snapshot, Mapping):
        raise TypeError("snapshot must be a scheduler snapshot mapping")
    audit_mapping_for_information_leakage(snapshot, path="scheduler_snapshot")
    kinds = []
    window_age = 0.0
    window_age_valid = False
    for event in events:
        if not isinstance(event, SchedulerEvent):
            raise TypeError("events must contain SchedulerEvent values")
        if event.epoch_id != cycle:
            raise ValueError("scheduler event epoch must align with cycle_index")
        audit_mapping_for_information_leakage(event.details, path=f"event[{event.kind}].details")
        kinds.append(event.kind)
        if event.kind in {"slow_update_finished", "window_deadline_miss"} and "window_age_us" in event.details:
            window_age = _finite(event.details["window_age_us"], "window_age_us")
            window_age_valid = True
    kind_set = set(kinds)
    priority = (
        ("conflict", {"parameter_update_conflict"}),
        ("failed", {"slow_update_failed"}),
        ("stale", {"window_deadline_miss"}),
        ("committed", {"commit_applied"}),
        ("staged", {"params_staged", "external_params_staged"}),
    )
    update_status = next(
        (status for status, triggers in priority if kind_set & triggers),
        "none",
    )
    param_bank = snapshot.get("param_bank")
    if not isinstance(param_bank, Mapping):
        raise ValueError("snapshot must contain param_bank mapping")
    return HistoryRuntimeStatus(
        cycle_index=cycle,
        fast_deadline_ok="fast_budget_violation" not in kind_set,
        slow_deadline_ok=not bool(kind_set & {"slow_budget_violation", "window_deadline_miss"}),
        communication_available=bool(snapshot.get("communication_available", False)),
        update_status=update_status,
        # ``update_status`` is the highest-priority diagnostic observed in this
        # cycle, while ``update_applied`` records the independent actuation
        # fact.  A commit and a later conflict may legitimately coexist in one
        # scheduler tick; collapsing the latter into ``False`` would erase a
        # deployable causal input.
        update_applied="commit_applied" in kind_set,
        pending_update=param_bank.get("pending_commit") is not None,
        active_bank_version=_integer(param_bank.get("active_version"), "active_version"),
        pending_window_count=_integer(snapshot.get("pending_windows"), "pending_windows"),
        window_age_us=window_age,
        window_age_valid=window_age_valid,
        crc_ok=_boolean(crc_ok, "crc_ok"),
    )


@dataclass(frozen=True)
class ExperimentalHistorySample:
    end_cycle: int
    values: np.ndarray
    mask: np.ndarray
    cycle_indices: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    schema_version: str = "t4.1.2-experimental-history-v1"

    def __post_init__(self) -> None:
        end_cycle = _integer(self.end_cycle, "end_cycle")
        values = np.asarray(self.values, dtype=np.float64)
        mask = np.asarray(self.mask, dtype=np.float64)
        indices = np.asarray(self.cycle_indices, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError("values must have shape (history, feature_count)")
        if mask.shape != (len(values),) or indices.shape != (len(values),):
            raise ValueError("mask and cycle_indices must align with history rows")
        if not np.all(np.isfinite(values)) or np.any((mask != 0.0) & (mask != 1.0)):
            raise ValueError("values must be finite and mask must be binary")
        if not np.any(mask == 1.0) or indices[-1] != end_cycle:
            raise ValueError("history must contain the declared end cycle")
        first_valid = int(np.argmax(mask == 1.0))
        if np.any(mask[:first_valid] != 0.0) or np.any(mask[first_valid:] != 1.0):
            raise ValueError("padding must be a single left prefix")
        if np.any(values[:first_valid] != 0.0) or np.any(indices[:first_valid] != -1):
            raise ValueError("padded rows must be zero with cycle index -1")
        valid_indices = indices[first_valid:]
        if not np.array_equal(valid_indices, np.arange(valid_indices[0], end_cycle + 1)):
            raise ValueError("valid history cycle indices must be contiguous")
        for name, array in (("values", values), ("mask", mask), ("cycle_indices", indices)):
            copy = np.array(array, copy=True)
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        object.__setattr__(self, "end_cycle", end_cycle)
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("feature_names must equal the frozen schema")


class ExperimentalHistoryBuilder:
    def __init__(self, config: ExperimentalHistoryConfig | None = None) -> None:
        self.config = ExperimentalHistoryConfig() if config is None else config
        if not isinstance(self.config, ExperimentalHistoryConfig):
            raise TypeError("config must be ExperimentalHistoryConfig")
        self._rows: list[np.ndarray] = []
        self._cycles: list[int] = []

    @property
    def row_count(self) -> int:
        return len(self._rows)

    def _row(
        self,
        observed: ObservedSyndromeStep,
        action: ObservedActionRecord,
        llr_context: DeployableLLRContext,
        runtime: HistoryRuntimeStatus,
    ) -> np.ndarray:
        if not isinstance(observed, ObservedSyndromeStep):
            raise TypeError("observed must be ObservedSyndromeStep")
        if not isinstance(action, ObservedActionRecord):
            raise TypeError("action must be ObservedActionRecord")
        if not isinstance(llr_context, DeployableLLRContext):
            raise TypeError("llr_context must be DeployableLLRContext")
        if not isinstance(runtime, HistoryRuntimeStatus):
            raise TypeError("runtime must be HistoryRuntimeStatus")
        if observed.observation_scope != "deployable_observed_syndrome":
            raise ValueError("observed record must declare deployable_observed_syndrome scope")
        cycle = _integer(observed.cycle_index, "observed.cycle_index")
        _boolean(observed.valid, "observed.valid")
        for name in ("x_e_run", "z_e_run", "leakage_run"):
            _integer(getattr(observed, name), f"observed.{name}")
        if action.cycle_index != cycle or runtime.cycle_index != cycle:
            raise ValueError("observed, action and runtime cycles must align")
        outcomes = []
        for axis_value in (observed.syndrome.x, observed.syndrome.z):
            outcomes.extend(float(axis_value == item) for item in ("g", "e", "leakage"))
        phases = []
        for phase in observed.quadrature_phases_rad:
            actual = _finite(phase, "quadrature_phase")
            phases.extend((float(np.sin(actual)), float(np.cos(actual))))
        mode = [float(action.mode == item) for item in FSM_MODES]
        raw_llr = np.asarray(
            [
                llr_1d(observed.residual_syndrome[0], llr_context.sigma[0], mean=llr_context.mean[0]),
                llr_1d(observed.residual_syndrome[1], llr_context.sigma[1], mean=llr_context.mean[1]),
            ],
            dtype=np.float64,
        )
        llr_saturated = np.abs(raw_llr) > self.config.llr_clip
        llr = np.clip(raw_llr, -self.config.llr_clip, self.config.llr_clip)
        raw_runs = np.asarray((observed.x_e_run, observed.z_e_run, observed.leakage_run), dtype=np.int64)
        run_saturated = raw_runs > self.config.run_length_clip
        runs = np.minimum(raw_runs, self.config.run_length_clip)
        update = [float(runtime.update_status == item) for item in UPDATE_STATUSES]
        bank_version_saturated = runtime.active_bank_version > self.config.bank_version_clip
        pending_window_saturated = runtime.pending_window_count > self.config.pending_window_clip
        row = np.asarray(
            [
                *observed.analog_syndrome,
                *observed.residual_syndrome,
                *outcomes,
                *phases,
                *action.correction,
                *mode,
                float(action.bank_switched),
                float(action.local_safe_rom_used),
                float(action.bank_conflict),
                *llr,
                *llr_saturated.astype(float),
                *runs.astype(float),
                *run_saturated.astype(float),
                float(runtime.fast_deadline_ok),
                float(runtime.slow_deadline_ok),
                float(runtime.communication_available),
                runtime.window_age_us,
                float(runtime.window_age_valid),
                *update,
                float(runtime.update_applied),
                float(runtime.pending_update),
                float(min(runtime.active_bank_version, self.config.bank_version_clip)),
                float(bank_version_saturated),
                float(min(runtime.pending_window_count, self.config.pending_window_clip)),
                float(pending_window_saturated),
                float(observed.valid),
                float(runtime.crc_ok),
            ],
            dtype=np.float64,
        )
        if row.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(row)):
            raise RuntimeError("constructed feature row violates frozen schema")
        return row

    def append(
        self,
        observed: ObservedSyndromeStep,
        action: ObservedActionRecord,
        llr_context: DeployableLLRContext,
        runtime: HistoryRuntimeStatus,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentalHistorySample:
        if metadata is not None:
            audit_mapping_for_information_leakage(metadata, path="history_metadata")
        cycle = (
            _integer(observed.cycle_index, "observed.cycle_index")
            if isinstance(observed, ObservedSyndromeStep)
            else -1
        )
        expected = 0 if not self._cycles else self._cycles[-1] + 1
        if cycle != expected:
            raise ValueError(f"cycle_index must be contiguous: expected {expected}, got {cycle}")
        row = self._row(observed, action, llr_context, runtime)
        self._rows.append(row)
        self._cycles.append(cycle)
        if len(self._rows) > self.config.history_cycles:
            del self._rows[0]
            del self._cycles[0]
        return self.snapshot()

    def snapshot(self) -> ExperimentalHistorySample:
        if not self._rows:
            raise RuntimeError("cannot snapshot an empty history")
        padding = self.config.history_cycles - len(self._rows)
        values = np.zeros((self.config.history_cycles, len(FEATURE_NAMES)), dtype=np.float64)
        mask = np.zeros(self.config.history_cycles, dtype=np.float64)
        cycle_indices = np.full(self.config.history_cycles, -1, dtype=np.int64)
        values[padding:] = np.asarray(self._rows)
        mask[padding:] = 1.0
        cycle_indices[padding:] = np.asarray(self._cycles)
        return ExperimentalHistorySample(
            end_cycle=self._cycles[-1],
            values=values,
            mask=mask,
            cycle_indices=cycle_indices,
        )


def schema_provenance() -> dict[str, object]:
    return {
        "schema_version": "t4.1.2-experimental-history-v1",
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "groups": {name: list(fields) for name, fields in FEATURE_GROUPS.items()},
        "mask_contract": "left padding is all-zero and mask=0; physical rows have mask=1",
        "cycle_alignment": "post-cycle observed syndrome plus same-cycle applied action and scheduler status; predicts only future slow state",
        "producers": {
            "syndrome": "physics.syndrome_stream.ObservedSyndromeStep",
            "action": "cnn_fpga.runtime.run_length_fsm.RunLengthFSMDecision",
            "llr": "physics.ideal_gkp_decoder.llr_1d with DeployableLLRContext",
            "runtime_status": "cnn_fpga.runtime.scheduler.SchedulerEvent plus DualLoopScheduler.snapshot",
        },
        "forbidden_tokens": list(FORBIDDEN_INPUT_TOKENS),
        "forbidden_object_types": ["SyndromeTruthStep", "DriftState"],
        "hardware_measured": False,
    }


__all__ = [
    "UPDATE_STATUSES",
    "DEPLOYABLE_LLR_SOURCES",
    "DEPLOYABLE_ACTION_SOURCES",
    "FORBIDDEN_INPUT_TOKENS",
    "FEATURE_GROUPS",
    "FEATURE_NAMES",
    "ExperimentalHistoryConfig",
    "DeployableLLRContext",
    "ObservedActionRecord",
    "HistoryRuntimeStatus",
    "ExperimentalHistorySample",
    "ExperimentalHistoryBuilder",
    "audit_mapping_for_information_leakage",
    "runtime_status_from_scheduler",
    "schema_provenance",
]
