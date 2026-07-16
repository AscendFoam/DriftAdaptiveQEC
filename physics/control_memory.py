"""T2.1.2 observed-only 多轮控制 memory。

该模块把 T2.1.1 的 deployable ``ObservedSyndromeStep`` 与实际执行的控制决定
组合成确定性 memory update。它跟踪 modular residual 的 nearest-lift、实际上一轮
correction、confidence、Pauli/phase frame、e/leakage runs、parameter-bank version
和 deadline 状态；不接收 ``SyndromeTruthStep`` 或 hidden regime。

``ControlDecision.applied_correction`` 沿用仓库 ``LogicalErrorTracker``/fast loop
约定：它是从 estimated error 中减去的 correction command。deadline miss 并不自动
把它清零，因为安全 fallback 仍可能按时执行本地动作；T4 才定义
deadline-to-fallback policy。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .constants import LATTICE_CONST
from .sbs_error_space import PauliFrame, SBS_PROTOCOL_ID
from .syndrome_stream import ObservedSyndromeStep


MODEL_SCOPE = "observed_only_multiround_control_memory_not_fallback_policy"


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return _finite(values[0], f"{name}[0]"), _finite(values[1], f"{name}[1]")


def _confidence_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    pair = _pair(values, name)
    if any(not 0.0 <= value <= 1.0 for value in pair):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return pair


def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _wrap(values: np.ndarray, lattice: float) -> np.ndarray:
    return np.mod(values + 0.5 * lattice, lattice) - 0.5 * lattice


def _wrap_phase(value: float) -> float:
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped)


@dataclass(frozen=True)
class ControlMemoryConfig:
    lattice: float = LATTICE_CONST
    counter_max: int = 65_535
    start_cycle_index: int = 0
    initial_parameter_bank_version: int = 0
    residual_consistency_atol: float = 1.0e-10
    strict_observed_run_validation: bool = True
    require_valid_observation: bool = True
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        lattice = _finite(self.lattice, "lattice")
        if lattice <= 0.0:
            raise ValueError("lattice must be positive")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "counter_max", _integer(self.counter_max, "counter_max", 1))
        object.__setattr__(
            self,
            "start_cycle_index",
            _integer(self.start_cycle_index, "start_cycle_index"),
        )
        object.__setattr__(
            self,
            "initial_parameter_bank_version",
            _integer(
                self.initial_parameter_bank_version,
                "initial_parameter_bank_version",
            ),
        )
        tolerance = _finite(self.residual_consistency_atol, "residual_consistency_atol")
        if tolerance < 0.0:
            raise ValueError("residual_consistency_atol must be non-negative")
        object.__setattr__(self, "residual_consistency_atol", tolerance)
        for name in ("strict_observed_run_validation", "require_valid_observation"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")


@dataclass(frozen=True)
class ControlDecision:
    """本周期实际执行结果；不是尚未落地的 proposal。

    ``applied_correction`` 的符号约定与仓库 fast loop 一致：post-action residual
    等于 estimated error 减去 correction，因此抵消正 residual 时 correction 为正。
    """

    applied_correction: tuple[float, float] = (0.0, 0.0)
    confidence: tuple[float, float] = (0.0, 0.0)
    pauli_frame_delta: PauliFrame = field(default_factory=PauliFrame)
    phase_frame_delta_rad: tuple[float, float] = (0.0, 0.0)
    parameter_bank_version: int = 0
    deadline_missed: bool = False
    control_mode: str = "normal"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied_correction",
            _pair(self.applied_correction, "applied_correction"),
        )
        object.__setattr__(
            self,
            "confidence",
            _confidence_pair(self.confidence, "confidence"),
        )
        if not isinstance(self.pauli_frame_delta, PauliFrame):
            raise TypeError("pauli_frame_delta must be a PauliFrame")
        object.__setattr__(
            self,
            "phase_frame_delta_rad",
            _pair(self.phase_frame_delta_rad, "phase_frame_delta_rad"),
        )
        object.__setattr__(
            self,
            "parameter_bank_version",
            _integer(self.parameter_bank_version, "parameter_bank_version"),
        )
        object.__setattr__(
            self,
            "deadline_missed",
            _boolean(self.deadline_missed, "deadline_missed"),
        )
        object.__setattr__(self, "control_mode", _text(self.control_mode, "control_mode"))


@dataclass(frozen=True)
class ControlMemoryState:
    cycle_index: int = -1
    cycle_count: int = 0
    accumulated_residual_shift: tuple[float, float] = (0.0, 0.0)
    previous_correction: tuple[float, float] = (0.0, 0.0)
    confidence: tuple[float, float] = (0.0, 0.0)
    pauli_frame: PauliFrame = field(default_factory=PauliFrame)
    phase_frame_rad: tuple[float, float] = (0.0, 0.0)
    x_e_run: int = 0
    z_e_run: int = 0
    leakage_run: int = 0
    parameter_bank_version: int = 0
    deadline_missed: bool = False
    deadline_miss_run: int = 0
    deadline_miss_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.cycle_index, bool) or not isinstance(
            self.cycle_index, (int, np.integer)
        ):
            raise TypeError("cycle_index must be an integer")
        if int(self.cycle_index) < -1:
            raise ValueError("cycle_index must be at least -1")
        object.__setattr__(self, "cycle_index", int(self.cycle_index))
        object.__setattr__(self, "cycle_count", _integer(self.cycle_count, "cycle_count"))
        object.__setattr__(
            self,
            "accumulated_residual_shift",
            _pair(self.accumulated_residual_shift, "accumulated_residual_shift"),
        )
        object.__setattr__(
            self,
            "previous_correction",
            _pair(self.previous_correction, "previous_correction"),
        )
        object.__setattr__(self, "confidence", _confidence_pair(self.confidence, "confidence"))
        if not isinstance(self.pauli_frame, PauliFrame):
            raise TypeError("pauli_frame must be a PauliFrame")
        phases = _pair(self.phase_frame_rad, "phase_frame_rad")
        object.__setattr__(
            self,
            "phase_frame_rad",
            (_wrap_phase(phases[0]), _wrap_phase(phases[1])),
        )
        for name in (
            "x_e_run",
            "z_e_run",
            "leakage_run",
            "parameter_bank_version",
            "deadline_miss_run",
            "deadline_miss_count",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(
            self,
            "deadline_missed",
            _boolean(self.deadline_missed, "deadline_missed"),
        )

    @property
    def minimum_confidence(self) -> float:
        return min(self.confidence)

    def as_deployable_dict(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "cycle_count": self.cycle_count,
            "accumulated_residual_q": self.accumulated_residual_shift[0],
            "accumulated_residual_p": self.accumulated_residual_shift[1],
            "previous_correction_q": self.previous_correction[0],
            "previous_correction_p": self.previous_correction[1],
            "confidence_q": self.confidence[0],
            "confidence_p": self.confidence[1],
            "minimum_confidence": self.minimum_confidence,
            "pauli_frame_x": self.pauli_frame.x,
            "pauli_frame_z": self.pauli_frame.z,
            "phase_frame_x_rad": self.phase_frame_rad[0],
            "phase_frame_z_rad": self.phase_frame_rad[1],
            "x_e_run": self.x_e_run,
            "z_e_run": self.z_e_run,
            "leakage_run": self.leakage_run,
            "parameter_bank_version": self.parameter_bank_version,
            "deadline_missed": self.deadline_missed,
            "deadline_miss_run": self.deadline_miss_run,
            "deadline_miss_count": self.deadline_miss_count,
            "memory_scope": MODEL_SCOPE,
        }


@dataclass(frozen=True)
class ControlMemoryUpdate:
    observation: ObservedSyndromeStep
    decision: ControlDecision
    previous_state: ControlMemoryState
    current_state: ControlMemoryState
    lifted_observation_shift: tuple[float, float]
    residual_alias_indices: tuple[int, int]
    parameter_bank_changed: bool

    def as_deployable_dict(self) -> dict[str, object]:
        record = self.current_state.as_deployable_dict()
        record.update(
            {
                "drift_step": self.observation.drift_step,
                "time": self.observation.time,
                "syndrome_x": self.observation.syndrome.x,
                "syndrome_z": self.observation.syndrome.z,
                "lifted_observation_q": self.lifted_observation_shift[0],
                "lifted_observation_p": self.lifted_observation_shift[1],
                "residual_alias_q": self.residual_alias_indices[0],
                "residual_alias_p": self.residual_alias_indices[1],
                "parameter_bank_changed": self.parameter_bank_changed,
                "control_mode": self.decision.control_mode,
            }
        )
        return record


@dataclass(frozen=True)
class ControlMemoryTrajectory:
    updates: tuple[ControlMemoryUpdate, ...]
    initial_state: ControlMemoryState
    final_state: ControlMemoryState
    protocol_id: str = SBS_PROTOCOL_ID
    model_scope: str = MODEL_SCOPE

    def deployable_records(self) -> tuple[dict[str, object], ...]:
        return tuple(update.as_deployable_dict() for update in self.updates)


class MultiRoundControlMemory:
    """严格因果的多轮 observed-only memory updater。"""

    def __init__(
        self,
        config: ControlMemoryConfig | None = None,
        *,
        initial_state: ControlMemoryState | None = None,
    ) -> None:
        self.config = ControlMemoryConfig() if config is None else config
        if not isinstance(self.config, ControlMemoryConfig):
            raise TypeError("config must be a ControlMemoryConfig or None")
        if initial_state is None:
            self._initial_state = ControlMemoryState(
                cycle_index=self.config.start_cycle_index - 1,
                parameter_bank_version=self.config.initial_parameter_bank_version,
            )
        else:
            if not isinstance(initial_state, ControlMemoryState):
                raise TypeError("initial_state must be a ControlMemoryState or None")
            if initial_state.cycle_index < self.config.start_cycle_index - 1:
                raise ValueError("initial_state precedes configured start_cycle_index")
            self._initial_state = initial_state
        self._state = self._initial_state
        self._history: list[ControlMemoryUpdate] = []

    @property
    def state(self) -> ControlMemoryState:
        return self._state

    @property
    def history(self) -> tuple[ControlMemoryUpdate, ...]:
        return tuple(self._history)

    def reset(self) -> ControlMemoryState:
        self._state = self._initial_state
        self._history.clear()
        return self._state

    def _saturating_increment(self, value: int) -> int:
        return min(self.config.counter_max, value + 1)

    def _validate_observation(self, observation: ObservedSyndromeStep) -> None:
        if not isinstance(observation, ObservedSyndromeStep):
            raise TypeError("observation must be an ObservedSyndromeStep")
        expected_cycle = self._state.cycle_index + 1
        if observation.cycle_index != expected_cycle:
            raise ValueError(
                f"observation cycle_index must be {expected_cycle}, got {observation.cycle_index}"
            )
        if self.config.require_valid_observation and not observation.valid:
            raise ValueError("invalid observation rejected by require_valid_observation")
        analog = np.asarray(observation.analog_syndrome, dtype=np.float64)
        residual = np.asarray(observation.residual_syndrome, dtype=np.float64)
        wrapped = _wrap(analog, self.config.lattice)
        if not np.allclose(
            wrapped,
            residual,
            rtol=0.0,
            atol=self.config.residual_consistency_atol,
        ):
            raise ValueError("observed residual is inconsistent with wrapped analog syndrome")

    def update(
        self,
        observation: ObservedSyndromeStep,
        decision: ControlDecision,
    ) -> ControlMemoryUpdate:
        self._validate_observation(observation)
        if not isinstance(decision, ControlDecision):
            raise TypeError("decision must be a ControlDecision")
        previous = self._state
        if decision.parameter_bank_version < previous.parameter_bank_version:
            raise ValueError("parameter_bank_version rollback is not allowed")

        x_e_run = (
            self._saturating_increment(previous.x_e_run)
            if observation.syndrome.x == "e"
            else 0
        )
        z_e_run = (
            self._saturating_increment(previous.z_e_run)
            if observation.syndrome.z == "e"
            else 0
        )
        leakage_run = (
            self._saturating_increment(previous.leakage_run)
            if "leakage" in observation.syndrome.as_tuple()
            else 0
        )
        if self.config.strict_observed_run_validation:
            supplied = (
                min(observation.x_e_run, self.config.counter_max),
                min(observation.z_e_run, self.config.counter_max),
                min(observation.leakage_run, self.config.counter_max),
            )
            expected = (x_e_run, z_e_run, leakage_run)
            if supplied != expected:
                raise ValueError(
                    f"observed run counters are not causal: supplied={supplied}, expected={expected}"
                )

        residual = np.asarray(observation.residual_syndrome, dtype=np.float64)
        reference = np.asarray(previous.accumulated_residual_shift, dtype=np.float64)
        aliases = np.floor((reference - residual) / self.config.lattice + 0.5).astype(
            np.int64
        )
        lifted = residual + aliases * self.config.lattice
        correction = np.asarray(decision.applied_correction, dtype=np.float64)
        accumulated = lifted - correction
        if not np.all(np.isfinite(accumulated)):
            raise ValueError("accumulated residual shift overflowed")

        phase_previous = np.asarray(previous.phase_frame_rad, dtype=np.float64)
        phase_delta = np.asarray(decision.phase_frame_delta_rad, dtype=np.float64)
        phase_frame = tuple(_wrap_phase(float(value)) for value in phase_previous + phase_delta)
        pauli_frame = PauliFrame(
            x=previous.pauli_frame.x ^ decision.pauli_frame_delta.x,
            z=previous.pauli_frame.z ^ decision.pauli_frame_delta.z,
        )
        deadline_miss_run = (
            self._saturating_increment(previous.deadline_miss_run)
            if decision.deadline_missed
            else 0
        )
        deadline_miss_count = (
            self._saturating_increment(previous.deadline_miss_count)
            if decision.deadline_missed
            else previous.deadline_miss_count
        )

        current = ControlMemoryState(
            cycle_index=observation.cycle_index,
            cycle_count=previous.cycle_count + 1,
            accumulated_residual_shift=(float(accumulated[0]), float(accumulated[1])),
            previous_correction=decision.applied_correction,
            confidence=decision.confidence,
            pauli_frame=pauli_frame,
            phase_frame_rad=(float(phase_frame[0]), float(phase_frame[1])),
            x_e_run=x_e_run,
            z_e_run=z_e_run,
            leakage_run=leakage_run,
            parameter_bank_version=decision.parameter_bank_version,
            deadline_missed=decision.deadline_missed,
            deadline_miss_run=deadline_miss_run,
            deadline_miss_count=deadline_miss_count,
        )
        update = ControlMemoryUpdate(
            observation=observation,
            decision=decision,
            previous_state=previous,
            current_state=current,
            lifted_observation_shift=(float(lifted[0]), float(lifted[1])),
            residual_alias_indices=(int(aliases[0]), int(aliases[1])),
            parameter_bank_changed=(
                decision.parameter_bank_version != previous.parameter_bank_version
            ),
        )
        self._state = current
        self._history.append(update)
        return update

    def run(
        self,
        observations: Sequence[ObservedSyndromeStep],
        decisions: Sequence[ControlDecision],
    ) -> ControlMemoryTrajectory:
        if isinstance(observations, (str, bytes)) or isinstance(decisions, (str, bytes)):
            raise TypeError("observations and decisions must be sequences")
        if len(observations) != len(decisions):
            raise ValueError("observations and decisions must have equal length")
        call_initial = self._state
        updates = tuple(
            self.update(observation, decision)
            for observation, decision in zip(observations, decisions)
        )
        return ControlMemoryTrajectory(
            updates=updates,
            initial_state=call_initial,
            final_state=self._state,
        )
