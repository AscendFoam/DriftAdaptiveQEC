"""Deterministic run-length event FSM backed by the real double ParamBank."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np

from cnn_fpga.runtime.param_bank import (
    DecoderRuntimeParams,
    ParamBank,
    ParameterUpdateConflictError,
)


NORMAL = "normal"
X_RECOVERY = "x_recovery"
Z_RECOVERY = "z_recovery"
LEAKAGE_HOLD = "leakage_hold"
FALLBACK = "fallback"
FSM_MODES = (NORMAL, X_RECOVERY, Z_RECOVERY, LEAKAGE_HOLD, FALLBACK)
OBSERVED_CLASSES = ("g", "e", "leakage")


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


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


@dataclass(frozen=True)
class RunLengthFSMConfig:
    counter_bits: int = 3
    e_enter_run: int = 2
    leakage_enter_run: int = 1
    leakage_clear_run: int = 2
    fallback_clear_run: int = 2
    correction_limit: float = 1.0

    def __post_init__(self) -> None:
        bits = _integer(self.counter_bits, "counter_bits", 2)
        if bits > 16:
            raise ValueError("counter_bits must not exceed 16")
        object.__setattr__(self, "counter_bits", bits)
        for name in (
            "e_enter_run",
            "leakage_enter_run",
            "leakage_clear_run",
            "fallback_clear_run",
        ):
            value = _integer(getattr(self, name), name, 1)
            if value > 2**bits - 1:
                raise ValueError(f"{name} exceeds the saturating counter range")
            object.__setattr__(self, name, value)
        limit = _finite(self.correction_limit, "correction_limit")
        if limit <= 0.0:
            raise ValueError("correction_limit must be positive")
        object.__setattr__(self, "correction_limit", limit)

    @property
    def counter_max(self) -> int:
        return 2**self.counter_bits - 1


@dataclass(frozen=True)
class RunLengthFSMInput:
    cycle_index: int
    residual: tuple[float, float]
    syndrome_x: str
    syndrome_z: str
    quadrature_phase_bit: int
    valid: bool = True
    crc_ok: bool = True
    parameter_fresh: bool = True
    deadline_ok: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index"))
        if len(self.residual) != 2:
            raise ValueError("residual must contain q and p")
        residual = (_finite(self.residual[0], "residual[0]"), _finite(self.residual[1], "residual[1]"))
        object.__setattr__(self, "residual", residual)
        for name in ("syndrome_x", "syndrome_z"):
            value = getattr(self, name)
            if value not in OBSERVED_CLASSES:
                raise ValueError(f"{name} must be one of {OBSERVED_CLASSES}")
        phase = _integer(self.quadrature_phase_bit, "quadrature_phase_bit")
        if phase not in (0, 1):
            raise ValueError("quadrature_phase_bit must be 0 (X) or 1 (Z)")
        object.__setattr__(self, "quadrature_phase_bit", phase)
        for name in ("valid", "crc_ok", "parameter_fresh", "deadline_ok"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))

    @property
    def health_ok(self) -> bool:
        return self.valid and self.crc_ok and self.parameter_fresh and self.deadline_ok


@dataclass(frozen=True)
class RunLengthFSMState:
    cycle_index: int = -1
    mode: str = NORMAL
    x_e_run: int = 0
    z_e_run: int = 0
    leakage_run: int = 0
    leakage_clean_run: int = 0
    health_good_run: int = 0
    transition_count: int = 0
    fallback_count: int = 0
    parameter_bank_version: int = 0
    last_reason: str = "initial"


@dataclass(frozen=True)
class RunLengthFSMDecision:
    cycle_index: int
    mode: str
    requested_mode: str
    reason: str
    correction: tuple[float, float]
    parameter_bank_version: int
    bank_switched: bool
    bank_sync_required: bool
    local_safe_rom_used: bool
    bank_conflict: bool
    x_e_run: int
    z_e_run: int
    leakage_run: int
    leakage_clean_run: int
    health_good_run: int
    quadrature_phase_bit: int


class RunLengthParameterTable:
    """Immutable mode-to-parameter ROM used to stage atomic bank switches."""

    def __init__(self, entries: Mapping[str, DecoderRuntimeParams] | None = None) -> None:
        actual = self.default_entries() if entries is None else dict(entries)
        if set(actual) != set(FSM_MODES):
            raise ValueError(f"parameter table must contain exactly {FSM_MODES}")
        self._entries = MappingProxyType({mode: actual[mode].copy() for mode in FSM_MODES})

    @staticmethod
    def default_entries() -> dict[str, DecoderRuntimeParams]:
        def params(mode: str, diagonal: tuple[float, float]) -> DecoderRuntimeParams:
            return DecoderRuntimeParams(
                K=np.diag(diagonal),
                b=np.zeros(2),
                metadata={"mode": mode, "source": "t3.2.5-local-parameter-rom"},
            )

        return {
            NORMAL: params(NORMAL, (1.0, 1.0)),
            X_RECOVERY: params(X_RECOVERY, (0.65, 1.0)),
            Z_RECOVERY: params(Z_RECOVERY, (1.0, 0.65)),
            LEAKAGE_HOLD: params(LEAKAGE_HOLD, (0.0, 0.0)),
            FALLBACK: params(FALLBACK, (0.5, 0.5)),
        }

    def params(self, mode: str) -> DecoderRuntimeParams:
        if mode not in self._entries:
            raise ValueError(f"unknown FSM mode {mode!r}")
        return self._entries[mode].copy()


class RunLengthParameterBankFSM:
    """Observed-only FSM with saturating counters and atomic mode commits."""

    def __init__(
        self,
        config: RunLengthFSMConfig | None = None,
        *,
        parameter_table: RunLengthParameterTable | None = None,
        param_bank: ParamBank | None = None,
    ) -> None:
        self.config = RunLengthFSMConfig() if config is None else config
        if not isinstance(self.config, RunLengthFSMConfig):
            raise TypeError("config must be RunLengthFSMConfig")
        self.parameter_table = RunLengthParameterTable() if parameter_table is None else parameter_table
        if not isinstance(self.parameter_table, RunLengthParameterTable):
            raise TypeError("parameter_table must be RunLengthParameterTable")
        self.param_bank = ParamBank(self.parameter_table.params(NORMAL)) if param_bank is None else param_bank
        if not isinstance(self.param_bank, ParamBank):
            raise TypeError("param_bank must be ParamBank")
        self._state = RunLengthFSMState(parameter_bank_version=self.param_bank.active_version)
        self._history: list[RunLengthFSMDecision] = []

    @property
    def state(self) -> RunLengthFSMState:
        return self._state

    @property
    def history(self) -> tuple[RunLengthFSMDecision, ...]:
        return tuple(self._history)

    def _sat(self, value: int) -> int:
        return min(self.config.counter_max, value + 1)

    def _select_mode(
        self,
        event: RunLengthFSMInput,
        *,
        x_run: int,
        z_run: int,
        leakage_run: int,
        leakage_clean_run: int,
        health_good_run: int,
    ) -> tuple[str, str]:
        if not event.health_ok:
            failed = [
                name
                for name, passed in (
                    ("valid", event.valid),
                    ("crc", event.crc_ok),
                    ("fresh", event.parameter_fresh),
                    ("deadline", event.deadline_ok),
                )
                if not passed
            ]
            return FALLBACK, "health_fault:" + "+".join(failed)
        if leakage_run >= self.config.leakage_enter_run:
            return LEAKAGE_HOLD, "leakage_run_threshold"
        if self._state.mode == LEAKAGE_HOLD and leakage_clean_run < self.config.leakage_clear_run:
            return LEAKAGE_HOLD, "leakage_clear_hysteresis"
        if self._state.mode == FALLBACK and health_good_run < self.config.fallback_clear_run:
            return FALLBACK, "fallback_clear_hysteresis"
        x_ready = x_run >= self.config.e_enter_run
        z_ready = z_run >= self.config.e_enter_run
        if x_ready and z_ready:
            if event.quadrature_phase_bit == 0:
                return X_RECOVERY, "both_e_runs_phase_x_priority"
            return Z_RECOVERY, "both_e_runs_phase_z_priority"
        if x_ready:
            return X_RECOVERY, "x_e_run_threshold"
        if z_ready:
            return Z_RECOVERY, "z_e_run_threshold"
        return NORMAL, "no_event_threshold"

    def step(self, event: RunLengthFSMInput) -> RunLengthFSMDecision:
        if not isinstance(event, RunLengthFSMInput):
            raise TypeError("event must be RunLengthFSMInput")
        if event.cycle_index != self._state.cycle_index + 1:
            raise ValueError("cycle_index must be sequential with no replay or gaps")

        # Apply an already-staged external update first; a future pending update
        # remains pending and may cause the explicit conflict path below.
        self.param_bank.commit_if_ready(event.cycle_index)
        x_run = self._sat(self._state.x_e_run) if event.syndrome_x == "e" else 0
        z_run = self._sat(self._state.z_e_run) if event.syndrome_z == "e" else 0
        leakage = event.syndrome_x == "leakage" or event.syndrome_z == "leakage"
        leakage_run = self._sat(self._state.leakage_run) if leakage else 0
        leakage_clean_run = 0 if leakage else self._sat(self._state.leakage_clean_run)
        health_good_run = self._sat(self._state.health_good_run) if event.health_ok else 0
        requested_mode, reason = self._select_mode(
            event,
            x_run=x_run,
            z_run=z_run,
            leakage_run=leakage_run,
            leakage_clean_run=leakage_clean_run,
            health_good_run=health_good_run,
        )
        active_mode = self.param_bank.read_active().metadata.get("mode")
        bank_sync_required = active_mode != requested_mode
        conflict = False
        local_safe = False
        action_mode = requested_mode
        if bank_sync_required:
            try:
                self.param_bank.stage_update(
                    self.parameter_table.params(requested_mode),
                    commit_epoch=event.cycle_index,
                    metadata={"fsm_mode": requested_mode, "reason": reason},
                )
                result = self.param_bank.commit_if_ready(event.cycle_index)
                if result is None:
                    raise RuntimeError("same-cycle FSM bank commit did not activate")
            except ParameterUpdateConflictError:
                conflict = True
                local_safe = True
                action_mode = FALLBACK
                reason = (
                    "parameter_bank_conflict_local_safe_fallback:"
                    f"requested={requested_mode}"
                )

        params = (
            self.parameter_table.params(FALLBACK)
            if local_safe
            else self.param_bank.read_active()
        )
        correction_array = params.K @ np.asarray(event.residual, dtype=float) + params.b
        correction_array = np.clip(
            correction_array,
            -self.config.correction_limit,
            self.config.correction_limit,
        )
        transitioned = action_mode != self._state.mode
        transition_count = self._state.transition_count + int(transitioned)
        fallback_count = self._state.fallback_count + int(action_mode == FALLBACK)
        self._state = RunLengthFSMState(
            cycle_index=event.cycle_index,
            mode=action_mode,
            x_e_run=x_run,
            z_e_run=z_run,
            leakage_run=leakage_run,
            leakage_clean_run=leakage_clean_run,
            health_good_run=health_good_run,
            transition_count=transition_count,
            fallback_count=fallback_count,
            parameter_bank_version=self.param_bank.active_version,
            last_reason=reason,
        )
        decision = RunLengthFSMDecision(
            cycle_index=event.cycle_index,
            mode=action_mode,
            requested_mode=requested_mode,
            reason=reason,
            correction=(float(correction_array[0]), float(correction_array[1])),
            parameter_bank_version=self.param_bank.active_version,
            bank_switched=bank_sync_required and not conflict,
            bank_sync_required=bank_sync_required,
            local_safe_rom_used=local_safe,
            bank_conflict=conflict,
            x_e_run=x_run,
            z_e_run=z_run,
            leakage_run=leakage_run,
            leakage_clean_run=leakage_clean_run,
            health_good_run=health_good_run,
            quadrature_phase_bit=event.quadrature_phase_bit,
        )
        self._history.append(decision)
        return decision


__all__ = [
    "NORMAL",
    "X_RECOVERY",
    "Z_RECOVERY",
    "LEAKAGE_HOLD",
    "FALLBACK",
    "FSM_MODES",
    "RunLengthFSMConfig",
    "RunLengthFSMInput",
    "RunLengthFSMState",
    "RunLengthFSMDecision",
    "RunLengthParameterTable",
    "RunLengthParameterBankFSM",
]
