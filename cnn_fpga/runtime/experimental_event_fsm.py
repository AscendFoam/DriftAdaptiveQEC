"""Observed-only event FSM and frame-action contract for T4.2.2."""

from __future__ import annotations

from dataclasses import dataclass

from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTDecision


NORMAL = "normal"
X_RECOVERY = "x_recovery"
Z_RECOVERY = "z_recovery"
HOLD = "hold"
RESET_REQUEST = "reset_request"
FALLBACK = "fallback"
EVENT_MODES = (NORMAL, X_RECOVERY, Z_RECOVERY, HOLD, RESET_REQUEST, FALLBACK)
SAFE_MODES = (HOLD, RESET_REQUEST, FALLBACK)
OBSERVED_CLASSES = ("g", "e", "leakage")
MODEL_SCOPE = "observed_event_integer_frame_contract_not_rtl_or_device"


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class ExperimentalEventFSMConfig:
    counter_bits: int = 3
    e_enter_run: int = 2
    reset_request_run: int = 2
    leakage_clear_run: int = 2
    fallback_clear_run: int = 2
    phase_frame_bits: int = 8
    map_pipeline_latency_cycles: int = 5
    event_action_latency_cycles: int = 1
    start_event_cycle: int = 5
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        for name, minimum in (
            ("counter_bits", 2),
            ("e_enter_run", 1),
            ("reset_request_run", 1),
            ("leakage_clear_run", 1),
            ("fallback_clear_run", 1),
            ("phase_frame_bits", 3),
            ("map_pipeline_latency_cycles", 1),
            ("event_action_latency_cycles", 1),
            ("start_event_cycle", 0),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.counter_bits > 16:
            raise ValueError("counter_bits must not exceed 16")
        if self.phase_frame_bits > 24:
            raise ValueError("phase_frame_bits must not exceed 24")
        if self.event_action_latency_cycles != 1:
            raise ValueError("event_action_latency_cycles must be exactly one")
        for name in (
            "e_enter_run",
            "reset_request_run",
            "leakage_clear_run",
            "fallback_clear_run",
        ):
            if getattr(self, name) > self.counter_max:
                raise ValueError(f"{name} exceeds saturating counter range")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")

    @property
    def counter_max(self) -> int:
        return (1 << self.counter_bits) - 1

    @property
    def phase_modulus(self) -> int:
        return 1 << self.phase_frame_bits

    @property
    def logical_half_turn_code(self) -> int:
        return 1 << (self.phase_frame_bits - 1)


@dataclass(frozen=True)
class ExperimentalEventInput:
    cycle_index: int
    syndrome_x: str
    syndrome_z: str
    quadrature_phase_bit: int
    map_decision: ParametricMAPLUTDecision | None
    active_bank_version: int
    reset_ack: bool = False
    valid: bool = True
    crc_ok: bool = True
    parameter_fresh: bool = True
    deadline_ok: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index"))
        for name in ("syndrome_x", "syndrome_z"):
            value = getattr(self, name)
            if value not in OBSERVED_CLASSES:
                raise ValueError(f"{name} must be one of {OBSERVED_CLASSES}")
        phase = _integer(self.quadrature_phase_bit, "quadrature_phase_bit")
        if phase not in (0, 1):
            raise ValueError("quadrature_phase_bit must be 0 (X) or 1 (Z)")
        object.__setattr__(self, "quadrature_phase_bit", phase)
        if self.map_decision is not None and not isinstance(
            self.map_decision, ParametricMAPLUTDecision
        ):
            raise TypeError("map_decision must be ParametricMAPLUTDecision or None")
        object.__setattr__(
            self,
            "active_bank_version",
            _integer(self.active_bank_version, "active_bank_version"),
        )
        for name in ("reset_ack", "valid", "crc_ok", "parameter_fresh", "deadline_ok"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))

    @property
    def health_ok(self) -> bool:
        return self.valid and self.crc_ok and self.parameter_fresh and self.deadline_ok

    @property
    def leakage_observed(self) -> bool:
        return "leakage" in (self.syndrome_x, self.syndrome_z)


@dataclass(frozen=True)
class ExperimentalEventFSMState:
    cycle_index: int
    mode: str = NORMAL
    x_e_run: int = 0
    z_e_run: int = 0
    leakage_run: int = 0
    leakage_clean_run: int = 0
    health_good_run: int = 0
    reset_wait_run: int = 0
    pauli_frame_x: bool = False
    pauli_frame_z: bool = False
    phase_frame_x_code: int = 0
    phase_frame_z_code: int = 0
    active_bank_version: int = 0
    transition_count: int = 0
    reset_request_count: int = 0
    fallback_count: int = 0
    frame_update_count: int = 0
    last_reason: str = "initial"


@dataclass(frozen=True)
class ExperimentalHardwareAction:
    source_cycle: int
    action_cycle: int
    mode: str
    reason: str
    correction_enable: bool
    reset_request: bool
    map_action_inhibited: bool
    map_logical_action: str
    pauli_frame_delta_x: bool
    pauli_frame_delta_z: bool
    phase_frame_delta_x_code: int
    phase_frame_delta_z_code: int
    pauli_frame_x: bool
    pauli_frame_z: bool
    phase_frame_x_code: int
    phase_frame_z_code: int
    x_e_run: int
    z_e_run: int
    leakage_run: int
    leakage_clean_run: int
    health_good_run: int
    reset_wait_run: int
    active_bank_version: int
    map_image_sha256: str
    model_scope: str = MODEL_SCOPE


class ExperimentalEventFSM:
    """Six-mode deterministic FSM with atomic integer frame updates."""

    def __init__(self, config: ExperimentalEventFSMConfig | None = None) -> None:
        self.config = ExperimentalEventFSMConfig() if config is None else config
        if not isinstance(self.config, ExperimentalEventFSMConfig):
            raise TypeError("config must be ExperimentalEventFSMConfig")
        self._initial_state = ExperimentalEventFSMState(
            cycle_index=self.config.start_event_cycle - 1
        )
        self._state = self._initial_state
        self._history: list[ExperimentalHardwareAction] = []

    @property
    def state(self) -> ExperimentalEventFSMState:
        return self._state

    @property
    def history(self) -> tuple[ExperimentalHardwareAction, ...]:
        return tuple(self._history)

    def reset(self) -> ExperimentalEventFSMState:
        self._state = self._initial_state
        self._history.clear()
        return self._state

    def _sat(self, value: int) -> int:
        return min(self.config.counter_max, value + 1)

    def _validate_event(self, event: ExperimentalEventInput) -> None:
        if not isinstance(event, ExperimentalEventInput):
            raise TypeError("event must be ExperimentalEventInput")
        if event.cycle_index != self._state.cycle_index + 1:
            raise ValueError("cycle_index must be sequential with no replay or gaps")
        decision = event.map_decision
        if decision is None:
            if event.health_ok:
                raise ValueError("healthy event requires an aligned MAP decision")
            return
        if decision.valid_cycle != event.cycle_index:
            raise ValueError("MAP decision valid_cycle is not aligned to event cycle")
        if (
            decision.valid_cycle - decision.input_cycle
            != self.config.map_pipeline_latency_cycles
        ):
            raise ValueError("MAP decision latency does not match registered pipeline")
        if decision.quadrature_phase_bit != event.quadrature_phase_bit:
            raise ValueError("MAP decision phase does not match event phase")
        if decision.active_bank_version != event.active_bank_version:
            raise ValueError("MAP decision bank version does not match event version")
        if event.active_bank_version < self._state.active_bank_version:
            raise ValueError("active bank version rollback is forbidden")
        expected_action = (
            decision.phase_label if decision.logical_flip else "I"
        )
        if decision.logical_action != expected_action:
            raise ValueError("MAP logical action is inconsistent with logical_flip/phase")
        if decision.logical_flip != (decision.llr_code < 0):
            raise ValueError("MAP logical_flip is inconsistent with LLR sign")

    def _select_mode(
        self,
        event: ExperimentalEventInput,
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
        if event.reset_ack:
            if self._state.mode != RESET_REQUEST:
                return FALLBACK, "unexpected_reset_ack"
            return HOLD, "reset_acknowledged_post_reset_hold"
        if self._state.mode == RESET_REQUEST:
            return RESET_REQUEST, "reset_request_sticky_until_ack"
        if event.leakage_observed:
            if leakage_run >= self.config.reset_request_run:
                return RESET_REQUEST, "persistent_leakage_reset_threshold"
            return HOLD, "leakage_observed_hold"
        if self._state.mode == HOLD and leakage_clean_run < self.config.leakage_clear_run:
            return HOLD, "post_leakage_clean_hysteresis"
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

    def step(self, event: ExperimentalEventInput) -> ExperimentalHardwareAction:
        self._validate_event(event)
        previous = self._state
        if event.health_ok:
            x_run = self._sat(previous.x_e_run) if event.syndrome_x == "e" else 0
            z_run = self._sat(previous.z_e_run) if event.syndrome_z == "e" else 0
            leakage_run = (
                self._sat(previous.leakage_run) if event.leakage_observed else 0
            )
            leakage_clean_run = (
                0
                if event.leakage_observed
                else self._sat(previous.leakage_clean_run)
            )
            health_good_run = self._sat(previous.health_good_run)
        else:
            x_run = z_run = leakage_run = leakage_clean_run = health_good_run = 0
        reset_wait_run = (
            self._sat(previous.reset_wait_run)
            if previous.mode == RESET_REQUEST and not event.reset_ack
            else 0
        )
        mode, reason = self._select_mode(
            event,
            x_run=x_run,
            z_run=z_run,
            leakage_run=leakage_run,
            leakage_clean_run=leakage_clean_run,
            health_good_run=health_good_run,
        )

        inhibited = mode in SAFE_MODES
        decision = event.map_decision
        apply_map = bool(not inhibited and decision is not None and decision.logical_flip)
        delta_x = bool(apply_map and decision is not None and decision.phase_label == "X")
        delta_z = bool(apply_map and decision is not None and decision.phase_label == "Z")
        half_turn = self.config.logical_half_turn_code
        phase_delta_x = half_turn if delta_x else 0
        phase_delta_z = half_turn if delta_z else 0
        pauli_x = previous.pauli_frame_x ^ delta_x
        pauli_z = previous.pauli_frame_z ^ delta_z
        phase_x = (previous.phase_frame_x_code + phase_delta_x) % self.config.phase_modulus
        phase_z = (previous.phase_frame_z_code + phase_delta_z) % self.config.phase_modulus
        transitioned = mode != previous.mode
        frame_updated = delta_x or delta_z
        self._state = ExperimentalEventFSMState(
            cycle_index=event.cycle_index,
            mode=mode,
            x_e_run=x_run,
            z_e_run=z_run,
            leakage_run=leakage_run,
            leakage_clean_run=leakage_clean_run,
            health_good_run=health_good_run,
            reset_wait_run=reset_wait_run,
            pauli_frame_x=pauli_x,
            pauli_frame_z=pauli_z,
            phase_frame_x_code=phase_x,
            phase_frame_z_code=phase_z,
            active_bank_version=event.active_bank_version,
            transition_count=previous.transition_count + int(transitioned),
            reset_request_count=previous.reset_request_count + int(mode == RESET_REQUEST),
            fallback_count=previous.fallback_count + int(mode == FALLBACK),
            frame_update_count=previous.frame_update_count + int(frame_updated),
            last_reason=reason,
        )
        action = ExperimentalHardwareAction(
            source_cycle=(
                decision.input_cycle
                if decision is not None
                else event.cycle_index - self.config.map_pipeline_latency_cycles
            ),
            action_cycle=event.cycle_index + self.config.event_action_latency_cycles,
            mode=mode,
            reason=reason,
            correction_enable=not inhibited,
            reset_request=mode == RESET_REQUEST,
            map_action_inhibited=bool(inhibited and decision is not None and decision.logical_flip),
            map_logical_action=decision.logical_action if decision is not None else "I",
            pauli_frame_delta_x=delta_x,
            pauli_frame_delta_z=delta_z,
            phase_frame_delta_x_code=phase_delta_x,
            phase_frame_delta_z_code=phase_delta_z,
            pauli_frame_x=pauli_x,
            pauli_frame_z=pauli_z,
            phase_frame_x_code=phase_x,
            phase_frame_z_code=phase_z,
            x_e_run=x_run,
            z_e_run=z_run,
            leakage_run=leakage_run,
            leakage_clean_run=leakage_clean_run,
            health_good_run=health_good_run,
            reset_wait_run=reset_wait_run,
            active_bank_version=event.active_bank_version,
            map_image_sha256=decision.image_sha256 if decision is not None else "",
        )
        self._history.append(action)
        return action


__all__ = [
    "EVENT_MODES",
    "FALLBACK",
    "HOLD",
    "MODEL_SCOPE",
    "NORMAL",
    "OBSERVED_CLASSES",
    "RESET_REQUEST",
    "SAFE_MODES",
    "X_RECOVERY",
    "Z_RECOVERY",
    "ExperimentalEventFSM",
    "ExperimentalEventFSMConfig",
    "ExperimentalEventFSMState",
    "ExperimentalEventInput",
    "ExperimentalHardwareAction",
]
