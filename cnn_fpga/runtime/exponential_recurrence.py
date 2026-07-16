"""Interpretable exponential-saturation event controller and fixed-point mirror.

Each observed branch updates a scalar evidence state through

    s[t+1] = a[m] * s[t] + (1-a[m]) * s_inf[m].

The controller carries independent X, Z, and leakage evidence.  It consumes
only the public ``RunLengthFSMInput`` observation contract and uses the same
atomic double-buffered parameter bank as the run-length comparator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np

from cnn_fpga.runtime.param_bank import ParamBank, ParameterUpdateConflictError
from cnn_fpga.runtime.run_length_fsm import (
    FALLBACK,
    LEAKAGE_HOLD,
    NORMAL,
    OBSERVED_CLASSES,
    X_RECOVERY,
    Z_RECOVERY,
    RunLengthFSMInput,
    RunLengthParameterTable,
)


ARITHMETIC_MODES = ("float64", "fixed_point")


def _unit_interval(value: object, name: str, *, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be real")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be real") from exc
    lower = result > 0.0 if strict else result >= 0.0
    upper = result < 1.0 if strict else result <= 1.0
    if not isfinite(result) or not lower or not upper:
        interval = "(0,1)" if strict else "[0,1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _integer(value: object, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if not lower <= parsed <= upper:
        raise ValueError(f"{name} must lie in [{lower},{upper}]")
    return parsed


@dataclass(frozen=True)
class ExponentialEventControllerConfig:
    saturation_g: float = 0.0
    saturation_e: float = 1.0
    saturation_leakage: float = 1.0
    decay_g: float = 0.65
    decay_e: float = 0.55
    decay_leakage: float = 0.20
    recovery_enter: float = 0.60
    recovery_exit: float = 0.25
    leakage_enter: float = 0.50
    leakage_exit: float = 0.15
    initial_state: float = 0.0
    correction_limit: float = 1.0
    state_fraction_bits: int = 16
    decay_fraction_bits: int = 18
    state_total_bits: int = 20

    def __post_init__(self) -> None:
        for name in (
            "saturation_g",
            "saturation_e",
            "saturation_leakage",
            "recovery_enter",
            "recovery_exit",
            "leakage_enter",
            "leakage_exit",
            "initial_state",
        ):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))
        for name in ("decay_g", "decay_e", "decay_leakage"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name, strict=True))
        if not self.recovery_exit < self.recovery_enter:
            raise ValueError("recovery_exit must be below recovery_enter")
        if not self.leakage_exit < self.leakage_enter:
            raise ValueError("leakage_exit must be below leakage_enter")
        limit = float(self.correction_limit)
        if not isfinite(limit) or limit <= 0.0:
            raise ValueError("correction_limit must be finite and positive")
        object.__setattr__(self, "correction_limit", limit)
        fraction = _integer(self.state_fraction_bits, "state_fraction_bits", 4, 24)
        decay_fraction = _integer(self.decay_fraction_bits, "decay_fraction_bits", 8, 24)
        total = _integer(self.state_total_bits, "state_total_bits", fraction + 2, 31)
        object.__setattr__(self, "state_fraction_bits", fraction)
        object.__setattr__(self, "decay_fraction_bits", decay_fraction)
        object.__setattr__(self, "state_total_bits", total)

    @property
    def saturations(self) -> tuple[float, float, float]:
        return (self.saturation_g, self.saturation_e, self.saturation_leakage)

    @property
    def decays(self) -> tuple[float, float, float]:
        return (self.decay_g, self.decay_e, self.decay_leakage)


@dataclass(frozen=True)
class ExponentialEventDecision:
    cycle_index: int
    mode: str
    requested_mode: str
    reason: str
    x_state: float
    z_state: float
    leakage_state: float
    state_codes: tuple[int, int, int]
    bank_switched: bool
    bank_conflict: bool
    local_safe_rom_used: bool
    correction: tuple[float, float]
    arithmetic: str


class ExponentialSaturationKernel:
    """Three-state recurrence with float64 and deterministic integer paths."""

    def __init__(self, config: ExponentialEventControllerConfig, arithmetic: str) -> None:
        if not isinstance(config, ExponentialEventControllerConfig):
            raise TypeError("config must be ExponentialEventControllerConfig")
        if arithmetic not in ARITHMETIC_MODES:
            raise ValueError(f"arithmetic must be one of {ARITHMETIC_MODES}")
        self.config = config
        self.arithmetic = arithmetic
        self.state_scale = 2**config.state_fraction_bits
        self.decay_scale = 2**config.decay_fraction_bits
        self.state_min = -(2 ** (config.state_total_bits - 1))
        self.state_max = 2 ** (config.state_total_bits - 1) - 1
        self.saturation_codes = tuple(self._state_code(value) for value in config.saturations)
        self.decay_codes = tuple(int(np.rint(value * self.decay_scale)) for value in config.decays)
        initial_code = self._state_code(config.initial_state)
        self._codes = [initial_code, initial_code, initial_code]
        self._states = [config.initial_state, config.initial_state, config.initial_state]

    def _state_code(self, value: float) -> int:
        return int(np.clip(np.rint(value * self.state_scale), self.state_min, self.state_max))

    @staticmethod
    def _round_divide_signed(numerator: int, denominator: int) -> int:
        if numerator >= 0:
            return (numerator + denominator // 2) // denominator
        return -((-numerator + denominator // 2) // denominator)

    @property
    def state_codes(self) -> tuple[int, int, int]:
        if self.arithmetic == "fixed_point":
            return tuple(self._codes)
        return tuple(self._state_code(value) for value in self._states)

    @property
    def states(self) -> tuple[float, float, float]:
        if self.arithmetic == "fixed_point":
            return tuple(code / self.state_scale for code in self._codes)
        return tuple(self._states)

    def _update_one(self, slot: int, outcome: str) -> None:
        if outcome not in OBSERVED_CLASSES:
            raise ValueError(f"outcome must be one of {OBSERVED_CLASSES}")
        branch = OBSERVED_CLASSES.index(outcome)
        if self.arithmetic == "fixed_point":
            decay = self.decay_codes[branch]
            numerator = decay * self._codes[slot] + (self.decay_scale - decay) * self.saturation_codes[branch]
            code = self._round_divide_signed(numerator, self.decay_scale)
            self._codes[slot] = int(np.clip(code, self.state_min, self.state_max))
        else:
            decay = self.config.decays[branch]
            saturation = self.config.saturations[branch]
            self._states[slot] = decay * self._states[slot] + (1.0 - decay) * saturation

    def step(self, syndrome_x: str, syndrome_z: str) -> tuple[float, float, float]:
        if syndrome_x not in OBSERVED_CLASSES or syndrome_z not in OBSERVED_CLASSES:
            raise ValueError("syndrome branches must be g, e, or leakage")
        # Leakage evidence is binary: ordinary g/e observations are both
        # leakage-clean.  Mapping a normal ``e`` to the leakage-e branch would
        # spuriously couple recovery evidence into the safety hold.
        aggregate = "leakage" if "leakage" in (syndrome_x, syndrome_z) else "g"
        self._update_one(0, syndrome_x)
        self._update_one(1, syndrome_z)
        self._update_one(2, aggregate)
        return self.states


class ExponentialRecurrenceEventController:
    """Observed-only recurrence controller with atomic mode commits."""

    def __init__(
        self,
        config: ExponentialEventControllerConfig | None = None,
        *,
        arithmetic: Literal["float64", "fixed_point"] = "float64",
        parameter_table: RunLengthParameterTable | None = None,
        param_bank: ParamBank | None = None,
    ) -> None:
        self.config = ExponentialEventControllerConfig() if config is None else config
        if not isinstance(self.config, ExponentialEventControllerConfig):
            raise TypeError("config must be ExponentialEventControllerConfig")
        self.kernel = ExponentialSaturationKernel(self.config, arithmetic)
        self.arithmetic = arithmetic
        self.parameter_table = RunLengthParameterTable() if parameter_table is None else parameter_table
        if not isinstance(self.parameter_table, RunLengthParameterTable):
            raise TypeError("parameter_table must be RunLengthParameterTable")
        self.param_bank = ParamBank(self.parameter_table.params(NORMAL)) if param_bank is None else param_bank
        if not isinstance(self.param_bank, ParamBank):
            raise TypeError("param_bank must be ParamBank")
        self._cycle_index = -1
        self._mode = NORMAL
        self._history: list[ExponentialEventDecision] = []

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def history(self) -> tuple[ExponentialEventDecision, ...]:
        return tuple(self._history)

    @property
    def bank_writes(self) -> int:
        return int(sum(decision.bank_switched for decision in self._history))

    def _threshold_code(self, value: float) -> int:
        return int(np.rint(value * self.kernel.state_scale))

    def _above(self, slot: int, threshold: float) -> bool:
        if self.arithmetic == "fixed_point":
            return self.kernel.state_codes[slot] >= self._threshold_code(threshold)
        return self.kernel.states[slot] >= threshold

    def _below_or_equal(self, slot: int, threshold: float) -> bool:
        if self.arithmetic == "fixed_point":
            return self.kernel.state_codes[slot] <= self._threshold_code(threshold)
        return self.kernel.states[slot] <= threshold

    def _select_mode(self, event: RunLengthFSMInput) -> tuple[str, str]:
        if not event.health_ok:
            return FALLBACK, "health_fault"
        leakage_ready = self._above(2, self.config.leakage_enter)
        leakage_held = self._mode == LEAKAGE_HOLD and not self._below_or_equal(2, self.config.leakage_exit)
        if leakage_ready or leakage_held:
            return LEAKAGE_HOLD, "leakage_exponential_threshold"
        threshold_x = self.config.recovery_exit if self._mode == X_RECOVERY else self.config.recovery_enter
        threshold_z = self.config.recovery_exit if self._mode == Z_RECOVERY else self.config.recovery_enter
        x_ready = self._above(0, threshold_x)
        z_ready = self._above(1, threshold_z)
        if x_ready and z_ready:
            return (
                (X_RECOVERY, "both_exponential_scores_phase_x_priority")
                if event.quadrature_phase_bit == 0
                else (Z_RECOVERY, "both_exponential_scores_phase_z_priority")
            )
        if x_ready:
            return X_RECOVERY, "x_exponential_threshold"
        if z_ready:
            return Z_RECOVERY, "z_exponential_threshold"
        return NORMAL, "scores_below_threshold"

    def step(self, event: RunLengthFSMInput) -> ExponentialEventDecision:
        if not isinstance(event, RunLengthFSMInput):
            raise TypeError("event must be RunLengthFSMInput")
        if event.cycle_index != self._cycle_index + 1:
            raise ValueError("cycle_index must be sequential with no replay or gaps")
        self.param_bank.commit_if_ready(event.cycle_index)
        self.kernel.step(event.syndrome_x, event.syndrome_z)
        requested, reason = self._select_mode(event)
        active_mode = self.param_bank.read_active().metadata.get("mode")
        switched = False
        conflict = False
        local_safe = False
        action_mode = requested
        if active_mode != requested:
            try:
                self.param_bank.stage_update(
                    self.parameter_table.params(requested),
                    commit_epoch=event.cycle_index,
                    metadata={"controller": "exponential_recurrence", "reason": reason},
                )
                committed = self.param_bank.commit_if_ready(event.cycle_index)
                if committed is None:
                    raise RuntimeError("same-cycle recurrence bank commit did not activate")
                switched = True
            except ParameterUpdateConflictError:
                conflict = True
                local_safe = True
                action_mode = FALLBACK
                reason = "parameter_bank_conflict_local_safe_fallback"
        params = self.parameter_table.params(FALLBACK) if local_safe else self.param_bank.read_active()
        correction = params.K @ np.asarray(event.residual, dtype=np.float64) + params.b
        correction = np.clip(correction, -self.config.correction_limit, self.config.correction_limit)
        states = self.kernel.states
        decision = ExponentialEventDecision(
            cycle_index=event.cycle_index,
            mode=action_mode,
            requested_mode=requested,
            reason=reason,
            x_state=states[0],
            z_state=states[1],
            leakage_state=states[2],
            state_codes=self.kernel.state_codes,
            bank_switched=switched,
            bank_conflict=conflict,
            local_safe_rom_used=local_safe,
            correction=(float(correction[0]), float(correction[1])),
            arithmetic=self.arithmetic,
        )
        self._cycle_index = event.cycle_index
        self._mode = action_mode
        self._history.append(decision)
        return decision


__all__ = [
    "ARITHMETIC_MODES",
    "ExponentialEventControllerConfig",
    "ExponentialEventDecision",
    "ExponentialSaturationKernel",
    "ExponentialRecurrenceEventController",
]
