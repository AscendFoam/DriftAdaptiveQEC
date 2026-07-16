"""Traceable conservative health/fallback policy for the integer fast path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cnn_fpga.runtime.experimental_event_fsm import (
    FALLBACK,
    HOLD,
    RESET_REQUEST,
    ExperimentalEventFSM,
    ExperimentalEventFSMConfig,
    ExperimentalEventInput,
    ExperimentalHardwareAction,
)
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTDecision


MODEL_SCOPE = "observed_health_and_integrity_fallback_contract_not_rtl_or_device"
HEALTHY = "healthy"
DEGRADED = "degraded"
RECOVERING = "recovering"
FALLBACK_ACTIVE = "fallback"
RESET_REQUIRED = "reset_required"
HEALTH_STATUSES = (HEALTHY, DEGRADED, RECOVERING, FALLBACK_ACTIVE, RESET_REQUIRED)

FAULT_ORDER = (
    "observation_invalid",
    "ood_score_exceeded",
    "input_crc_mismatch",
    "image_crc_mismatch",
    "image_sha256_mismatch",
    "unknown_bank_version",
    "bank_version_mismatch",
    "bank_version_rollback",
    "parameter_stale",
    "deadline_miss",
    "map_decision_missing",
    "map_alignment_or_action_invalid",
    "unexpected_reset_ack",
    "leakage_observed",
)
FAULT_BITS = {name: 1 << index for index, name in enumerate(FAULT_ORDER)}
NONBLOCKING_FLAGS = frozenset({"leakage_observed"})


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


def _hex_digest(value: object, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value.lower()


@dataclass(frozen=True)
class TrustedParameterImage:
    active_bank_version: int
    image_crc32: str
    image_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "active_bank_version",
            _integer(self.active_bank_version, "active_bank_version"),
        )
        object.__setattr__(
            self, "image_crc32", _hex_digest(self.image_crc32, "image_crc32", 8)
        )
        object.__setattr__(
            self,
            "image_sha256",
            _hex_digest(self.image_sha256, "image_sha256", 64),
        )


@dataclass(frozen=True)
class ConservativeFallbackConfig:
    ood_score_bits: int = 8
    ood_threshold_code: int = 192
    max_parameter_age_cycles: int = 64
    health_counter_bits: int = 8
    initial_active_bank_version: int = 0
    safe_profile_id: str = "frame_hold_no_map"
    model_scope: str = MODEL_SCOPE

    def __post_init__(self) -> None:
        for name, minimum in (
            ("ood_score_bits", 2),
            ("ood_threshold_code", 0),
            ("max_parameter_age_cycles", 0),
            ("health_counter_bits", 2),
            ("initial_active_bank_version", 0),
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))
        if self.ood_score_bits > 16:
            raise ValueError("ood_score_bits must not exceed 16")
        if self.health_counter_bits > 24:
            raise ValueError("health_counter_bits must not exceed 24")
        if self.ood_threshold_code > self.ood_code_max:
            raise ValueError("ood_threshold_code exceeds configured score width")
        if not isinstance(self.safe_profile_id, str) or not self.safe_profile_id:
            raise ValueError("safe_profile_id must be non-empty")
        if self.model_scope != MODEL_SCOPE:
            raise ValueError(f"model_scope must be {MODEL_SCOPE!r}")

    @property
    def ood_code_max(self) -> int:
        return (1 << self.ood_score_bits) - 1

    @property
    def health_counter_max(self) -> int:
        return (1 << self.health_counter_bits) - 1


@dataclass(frozen=True)
class ConservativeFallbackInput:
    cycle_index: int
    syndrome_x: str
    syndrome_z: str
    quadrature_phase_bit: int
    map_decision: ParametricMAPLUTDecision | None
    expected_active_bank_version: int
    reported_image_crc32: str
    reported_image_sha256: str
    parameter_age_cycles: int
    ood_score_code: int
    reset_ack: bool = False
    observation_valid: bool = True
    input_crc_ok: bool = True
    deadline_ok: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cycle_index", _integer(self.cycle_index, "cycle_index"))
        for name in ("syndrome_x", "syndrome_z"):
            if getattr(self, name) not in ("g", "e", "leakage"):
                raise ValueError(f"{name} must be g, e, or leakage")
        phase = _integer(self.quadrature_phase_bit, "quadrature_phase_bit")
        if phase not in (0, 1):
            raise ValueError("quadrature_phase_bit must be 0 or 1")
        object.__setattr__(self, "quadrature_phase_bit", phase)
        if self.map_decision is not None and not isinstance(
            self.map_decision, ParametricMAPLUTDecision
        ):
            raise TypeError("map_decision must be ParametricMAPLUTDecision or None")
        object.__setattr__(
            self,
            "expected_active_bank_version",
            _integer(self.expected_active_bank_version, "expected_active_bank_version"),
        )
        object.__setattr__(
            self,
            "reported_image_crc32",
            _hex_digest(self.reported_image_crc32, "reported_image_crc32", 8),
        )
        object.__setattr__(
            self,
            "reported_image_sha256",
            _hex_digest(self.reported_image_sha256, "reported_image_sha256", 64),
        )
        object.__setattr__(
            self,
            "parameter_age_cycles",
            _integer(self.parameter_age_cycles, "parameter_age_cycles"),
        )
        object.__setattr__(
            self, "ood_score_code", _integer(self.ood_score_code, "ood_score_code")
        )
        for name in ("reset_ack", "observation_valid", "input_crc_ok", "deadline_ok"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))


@dataclass(frozen=True)
class ConservativeFallbackState:
    cycle_index: int
    trusted_active_bank_version: int
    status: str = HEALTHY
    fault_run: int = 0
    good_run: int = 0
    fault_cycle_count: int = 0
    leakage_cycle_count: int = 0
    per_flag_cycle_counts: tuple[int, ...] = (0,) * len(FAULT_ORDER)
    last_fault_mask: int = 0
    last_primary_reason: str = "initial"


@dataclass(frozen=True)
class ConservativeFallbackAction:
    cycle_index: int
    hardware_action: ExperimentalHardwareAction
    status: str
    fault_flags: tuple[str, ...]
    fault_mask: int
    primary_reason: str
    reason_trace: str
    conservative_action: str
    map_decision_accepted: bool
    trusted_active_bank_version: int
    active_profile_id: str
    fault_run: int
    good_run: int
    fault_cycle_count: int
    leakage_cycle_count: int
    per_flag_cycle_counts: tuple[int, ...]
    model_scope: str = MODEL_SCOPE


class ConservativeFallbackController:
    """Pre-validates operational health and advances T4.2.2 without unsafe map use."""

    def __init__(
        self,
        trusted_images: Sequence[TrustedParameterImage],
        config: ConservativeFallbackConfig | None = None,
        event_config: ExperimentalEventFSMConfig | None = None,
    ) -> None:
        self.config = ConservativeFallbackConfig() if config is None else config
        if not isinstance(self.config, ConservativeFallbackConfig):
            raise TypeError("config must be ConservativeFallbackConfig")
        images = tuple(trusted_images)
        if not images or not all(isinstance(item, TrustedParameterImage) for item in images):
            raise ValueError("trusted_images must contain TrustedParameterImage records")
        versions = [item.active_bank_version for item in images]
        if len(set(versions)) != len(versions):
            raise ValueError("trusted image versions must be unique")
        self._trusted_images = {item.active_bank_version: item for item in images}
        if self.config.initial_active_bank_version not in self._trusted_images:
            raise ValueError("initial active bank version is not registered")
        self._fsm = ExperimentalEventFSM(event_config)
        if self._fsm.config.map_pipeline_latency_cycles != 5:
            raise ValueError("fallback controller requires the registered five-cycle MAP path")
        self._initial_state = ConservativeFallbackState(
            cycle_index=self._fsm.config.start_event_cycle - 1,
            trusted_active_bank_version=self.config.initial_active_bank_version,
        )
        self._state = self._initial_state
        self._history: list[ConservativeFallbackAction] = []

    @property
    def state(self) -> ConservativeFallbackState:
        return self._state

    @property
    def history(self) -> tuple[ConservativeFallbackAction, ...]:
        return tuple(self._history)

    @property
    def trusted_images(self) -> tuple[TrustedParameterImage, ...]:
        return tuple(self._trusted_images[key] for key in sorted(self._trusted_images))

    @property
    def event_config(self) -> ExperimentalEventFSMConfig:
        return self._fsm.config

    def reset(self) -> ConservativeFallbackState:
        self._fsm.reset()
        self._state = self._initial_state
        self._history.clear()
        return self._state

    def _sat(self, value: int) -> int:
        return min(self.config.health_counter_max, value + 1)

    def _map_is_aligned(self, event: ConservativeFallbackInput) -> bool:
        decision = event.map_decision
        if decision is None:
            return False
        try:
            expected_action = decision.phase_label if decision.logical_flip else "I"
            return bool(
                decision.valid_cycle == event.cycle_index
                and decision.valid_cycle - decision.input_cycle
                == self._fsm.config.map_pipeline_latency_cycles
                and decision.quadrature_phase_bit == event.quadrature_phase_bit
                and decision.phase_label
                == ("X" if event.quadrature_phase_bit == 0 else "Z")
                and decision.logical_action == expected_action
                and decision.logical_flip == (decision.llr_code < 0)
            )
        except (TypeError, ValueError):
            return False

    def _fault_flags(self, event: ConservativeFallbackInput) -> tuple[str, ...]:
        flags: set[str] = set()
        if not event.observation_valid:
            flags.add("observation_invalid")
        if event.ood_score_code > self.config.ood_threshold_code:
            flags.add("ood_score_exceeded")
        if not event.input_crc_ok:
            flags.add("input_crc_mismatch")
        trusted = self._trusted_images.get(event.expected_active_bank_version)
        if trusted is None:
            flags.add("unknown_bank_version")
        else:
            if event.reported_image_crc32 != trusted.image_crc32:
                flags.add("image_crc_mismatch")
            if event.reported_image_sha256 != trusted.image_sha256:
                flags.add("image_sha256_mismatch")
        if event.expected_active_bank_version < self._state.trusted_active_bank_version:
            flags.add("bank_version_rollback")
        if event.parameter_age_cycles > self.config.max_parameter_age_cycles:
            flags.add("parameter_stale")
        if not event.deadline_ok:
            flags.add("deadline_miss")
        if event.map_decision is None:
            flags.add("map_decision_missing")
        else:
            if event.map_decision.active_bank_version != event.expected_active_bank_version:
                flags.add("bank_version_mismatch")
            if trusted is not None and event.map_decision.image_sha256 != trusted.image_sha256:
                flags.add("image_sha256_mismatch")
            if not self._map_is_aligned(event):
                flags.add("map_alignment_or_action_invalid")
        if event.reset_ack and self._fsm.state.mode != RESET_REQUEST:
            flags.add("unexpected_reset_ack")
        if "leakage" in (event.syndrome_x, event.syndrome_z):
            flags.add("leakage_observed")
        return tuple(name for name in FAULT_ORDER if name in flags)

    def step(self, event: ConservativeFallbackInput) -> ConservativeFallbackAction:
        if not isinstance(event, ConservativeFallbackInput):
            raise TypeError("event must be ConservativeFallbackInput")
        if event.cycle_index != self._state.cycle_index + 1:
            raise ValueError("cycle_index must be sequential with no replay or gaps")
        if event.ood_score_code > self.config.ood_code_max:
            raise ValueError("ood_score_code exceeds configured score width")

        flags = self._fault_flags(event)
        blocking = tuple(name for name in flags if name not in NONBLOCKING_FLAGS)
        map_accepted = not blocking and event.map_decision is not None
        trusted_version = self._state.trusted_active_bank_version
        if map_accepted:
            trusted_version = event.expected_active_bank_version
        fsm_event = ExperimentalEventInput(
            cycle_index=event.cycle_index,
            syndrome_x=event.syndrome_x,
            syndrome_z=event.syndrome_z,
            quadrature_phase_bit=event.quadrature_phase_bit,
            map_decision=event.map_decision if map_accepted else None,
            active_bank_version=trusted_version,
            reset_ack=event.reset_ack and "unexpected_reset_ack" not in blocking,
            valid=not any(
                name in blocking
                for name in (
                    "observation_invalid",
                    "ood_score_exceeded",
                    "map_decision_missing",
                    "map_alignment_or_action_invalid",
                    "unexpected_reset_ack",
                )
            ),
            crc_ok=not any(
                name in blocking
                for name in (
                    "input_crc_mismatch",
                    "image_crc_mismatch",
                    "image_sha256_mismatch",
                )
            ),
            parameter_fresh=not any(
                name in blocking
                for name in (
                    "unknown_bank_version",
                    "bank_version_mismatch",
                    "bank_version_rollback",
                    "parameter_stale",
                )
            ),
            deadline_ok="deadline_miss" not in blocking,
        )
        hardware = self._fsm.step(fsm_event)
        previous = self._state
        fault_run = self._sat(previous.fault_run) if blocking else 0
        good_run = self._sat(previous.good_run) if not flags else 0
        counts = list(previous.per_flag_cycle_counts)
        for name in flags:
            index = FAULT_ORDER.index(name)
            counts[index] = self._sat(counts[index])
        mask = sum(FAULT_BITS[name] for name in flags)

        if hardware.mode == RESET_REQUEST:
            status = RESET_REQUIRED
        elif blocking:
            status = FALLBACK_ACTIVE
        elif hardware.mode == FALLBACK:
            status = RECOVERING
        elif flags or hardware.mode == HOLD:
            status = DEGRADED
        else:
            status = HEALTHY
        if flags:
            primary = flags[0]
        elif hardware.mode == FALLBACK:
            primary = "fallback_clear_hysteresis"
        else:
            primary = hardware.reason
        if hardware.mode == RESET_REQUEST:
            conservative_action = "reset_request"
        elif hardware.mode in (HOLD, FALLBACK):
            conservative_action = "frame_hold"
        else:
            conservative_action = "use_validated_map"
        active_profile = (
            self.config.safe_profile_id
            if conservative_action == "frame_hold" and hardware.mode == FALLBACK
            else f"trusted_map_v{trusted_version}"
        )
        self._state = ConservativeFallbackState(
            cycle_index=event.cycle_index,
            trusted_active_bank_version=trusted_version,
            status=status,
            fault_run=fault_run,
            good_run=good_run,
            fault_cycle_count=(
                self._sat(previous.fault_cycle_count) if blocking else previous.fault_cycle_count
            ),
            leakage_cycle_count=(
                self._sat(previous.leakage_cycle_count)
                if "leakage_observed" in flags
                else previous.leakage_cycle_count
            ),
            per_flag_cycle_counts=tuple(counts),
            last_fault_mask=mask,
            last_primary_reason=primary,
        )
        action = ConservativeFallbackAction(
            cycle_index=event.cycle_index,
            hardware_action=hardware,
            status=status,
            fault_flags=flags,
            fault_mask=mask,
            primary_reason=primary,
            reason_trace="|".join((*flags, f"fsm:{hardware.reason}")),
            conservative_action=conservative_action,
            map_decision_accepted=map_accepted,
            trusted_active_bank_version=trusted_version,
            active_profile_id=active_profile,
            fault_run=fault_run,
            good_run=good_run,
            fault_cycle_count=self._state.fault_cycle_count,
            leakage_cycle_count=self._state.leakage_cycle_count,
            per_flag_cycle_counts=self._state.per_flag_cycle_counts,
        )
        self._history.append(action)
        return action


__all__ = [
    "DEGRADED",
    "FALLBACK_ACTIVE",
    "FAULT_BITS",
    "FAULT_ORDER",
    "HEALTHY",
    "HEALTH_STATUSES",
    "MODEL_SCOPE",
    "RECOVERING",
    "RESET_REQUIRED",
    "ConservativeFallbackAction",
    "ConservativeFallbackConfig",
    "ConservativeFallbackController",
    "ConservativeFallbackInput",
    "ConservativeFallbackState",
    "TrustedParameterImage",
]
