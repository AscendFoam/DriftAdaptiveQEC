"""Integer, cycle-accurate reference for the Route-A FPGA policy overlay.

The slow loop exports a quantized four-class posterior in the frozen order
``normal, smooth, calibration_shift, burst``.  This module intentionally does
not recompute the HMM on the FPGA fast path.  It implements the deployable
posterior/action contract, hysteresis, leakage/integrity latches and the
trusted EWMA/Window bank router using integer arithmetic only.
"""

from __future__ import annotations

from dataclasses import dataclass

from cnn_fpga.runtime.fast_production_core_reference import crc16_int_little_endian


ACTION_OPEN = 0
ACTION_TAIL_EWMA = 1
ACTION_UNCERTAIN_EWMA = 2
ACTION_LEAKAGE_RESET = 3
ACTION_INTEGRITY_ROLLBACK = 4

REASON_ADAPTIVE_READY = 0
REASON_RAW_TAIL = 1
REASON_OOD_EVENT = 2
REASON_TAIL_LATCHED = 3
REASON_POSTERIOR_UNCERTAIN = 4
REASON_LEAKAGE = 5
REASON_INTEGRITY = 6
REASON_POSTERIOR_SUM = 7
REASON_VERSION = 8

EWMA_BANK = 0
WINDOW_BANK = 1

TAIL_ENTER_CODE = 230
TAIL_EXIT_CODE = 51
UNCERTAINTY_CODE_LIMIT = 64
OOD_CODE_LIMIT = 192
SMOOTH_BANK_CODE_MIN = 77
ENTER_HYSTERESIS = 2
RECOVERY_HYSTERESIS = 8


@dataclass(slots=True, frozen=True)
class RouteAPolicyInputs:
    posterior_valid: int = 0
    p_normal: int = 255
    p_smooth: int = 0
    p_calibration: int = 0
    p_burst: int = 0
    ood_code: int = 0
    router_boundary: int = 0
    window_prequential_win: int = 0
    integrity_fault: int = 0
    version_fault: int = 0
    integrity_clear: int = 0
    leakage_event: int = 0
    reset_ack: int = 0
    lkg_bank: int = EWMA_BANK


@dataclass(slots=True, frozen=True)
class RouteAPolicyCycleOutput:
    action: int
    reason: int
    selected_bank: int
    tail_latched: int
    leakage_latched: int
    integrity_latched: int
    enter_run: int
    recovery_run: int
    auto_commit_valid: int
    auto_commit_bank: int
    auto_commit_version: int
    commit_pending: int
    action_word: int
    state_word: int
    version_word: int


def _check_u8(value: int, name: str) -> int:
    value = int(value)
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must fit uint8")
    return value


class RouteAFixedPolicyReference:
    """Register-accurate model of :mod:`route_a_policy_overlay.sv`."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.action = ACTION_OPEN
        self.reason = REASON_ADAPTIVE_READY
        self.tail_latched = 0
        self.leakage_latched = 0
        self.integrity_latched = 0
        self.enter_run = 0
        self.recovery_run = 0
        self.selected_bank = EWMA_BANK
        self.commit_pending = 0
        self.pending_bank = EWMA_BANK
        self.policy_update_count = 0
        self.fallback_count = 0
        self.rollback_count = 0
        self.last_source_version = 0
        self._pipe: list[tuple[int, int, int, int, int] | None] = [None] * 6

    def peek_auto_commit(
        self, *, safe_boundary: int, active_bank: int, active_version: int
    ) -> tuple[int, int, int]:
        """Return the combinational pre-edge auto-commit request."""

        valid = int(
            bool(self.commit_pending)
            and bool(safe_boundary)
            and self.pending_bank != int(active_bank)
            and int(active_version) != 0xFFFF
        )
        return valid, self.pending_bank, (int(active_version) + 1) & 0xFFFF

    @staticmethod
    def _decision(
        inputs: RouteAPolicyInputs,
        *,
        action: int,
        reason: int,
        tail_latched: int,
        leakage_latched: int,
        integrity_latched: int,
        enter_run: int,
        recovery_run: int,
        active_bank: int,
        lkg_bank: int,
    ) -> tuple[int, int, int, int, int, int, int]:
        values = tuple(
            _check_u8(value, name)
            for value, name in zip(
                (inputs.p_normal, inputs.p_smooth, inputs.p_calibration, inputs.p_burst, inputs.ood_code),
                ("p_normal", "p_smooth", "p_calibration", "p_burst", "ood_code"),
                strict=True,
            )
        )
        p_normal, p_smooth, p_calibration, p_burst, ood = values
        posterior_sum_fault = bool(inputs.posterior_valid and sum(values[:4]) != 255)
        version_fault = bool(
            inputs.version_fault or active_bank not in (0, 1) or lkg_bank not in (0, 1)
        )
        integrity_now = bool(inputs.integrity_fault or posterior_sum_fault or version_fault)

        next_integrity = int(integrity_latched or integrity_now)
        if (
            next_integrity
            and inputs.integrity_clear
            and not integrity_now
            and active_bank == lkg_bank
        ):
            next_integrity = 0
        next_leakage = int(leakage_latched or inputs.leakage_event)
        if next_leakage and inputs.reset_ack and not inputs.leakage_event:
            next_leakage = 0

        next_tail = int(tail_latched)
        next_enter = int(enter_run)
        next_recovery = int(recovery_run)
        raw_tail = False
        event_alert = False
        adaptive_ready = False
        if inputs.posterior_valid:
            tail = p_calibration + p_burst
            adaptive = p_normal + p_smooth
            uncertainty = 255 - max(values[:4])
            raw_tail = tail >= TAIL_ENTER_CODE
            event_alert = ood > OOD_CODE_LIMIT
            adaptive_ready = adaptive >= TAIL_ENTER_CODE and uncertainty < UNCERTAINTY_CODE_LIMIT
            next_enter = min(3, next_enter + 1) if (raw_tail or event_alert) else 0
            next_recovery = min(15, next_recovery + 1) if tail <= TAIL_EXIT_CODE and adaptive_ready else 0
            if next_enter >= ENTER_HYSTERESIS:
                next_tail = 1
            elif next_recovery >= RECOVERY_HYSTERESIS:
                next_tail = 0

        next_action = int(action)
        next_reason = int(reason)
        if next_integrity:
            next_action = ACTION_INTEGRITY_ROLLBACK
            if version_fault:
                next_reason = REASON_VERSION
            elif posterior_sum_fault:
                next_reason = REASON_POSTERIOR_SUM
            else:
                next_reason = REASON_INTEGRITY
        elif next_leakage:
            next_action = ACTION_LEAKAGE_RESET
            next_reason = REASON_LEAKAGE
        elif inputs.posterior_valid:
            if raw_tail:
                next_action, next_reason = ACTION_TAIL_EWMA, REASON_RAW_TAIL
            elif event_alert:
                next_action, next_reason = ACTION_TAIL_EWMA, REASON_OOD_EVENT
            elif next_tail:
                next_action, next_reason = ACTION_TAIL_EWMA, REASON_TAIL_LATCHED
            elif not adaptive_ready:
                next_action, next_reason = ACTION_UNCERTAIN_EWMA, REASON_POSTERIOR_UNCERTAIN
            else:
                next_action, next_reason = ACTION_OPEN, REASON_ADAPTIVE_READY
        return (
            next_action,
            next_reason,
            next_tail,
            next_leakage,
            next_integrity,
            next_enter,
            next_recovery,
        )

    def step(
        self,
        inputs: RouteAPolicyInputs,
        *,
        sample_valid: int,
        safe_boundary: int,
        active_bank: int,
        active_version: int,
        core_output_word: int,
        visible_active_bank: int | None = None,
        visible_active_version: int | None = None,
    ) -> RouteAPolicyCycleOutput:
        active_bank = int(active_bank)
        active_version = int(active_version)
        lkg_bank = int(inputs.lkg_bank)
        if not 0 <= active_version <= 0xFFFF:
            raise ValueError("active_version must fit uint16")
        visible_active_bank = active_bank if visible_active_bank is None else int(visible_active_bank)
        visible_active_version = (
            active_version if visible_active_version is None else int(visible_active_version)
        )

        due = self._pipe.pop(0)
        decision = self._decision(
            inputs,
            action=self.action,
            reason=self.reason,
            tail_latched=self.tail_latched,
            leakage_latched=self.leakage_latched,
            integrity_latched=self.integrity_latched,
            enter_run=self.enter_run,
            recovery_run=self.recovery_run,
            active_bank=active_bank,
            lkg_bank=lkg_bank,
        )
        (
            next_action,
            next_reason,
            next_tail,
            next_leakage,
            next_integrity,
            next_enter,
            next_recovery,
        ) = decision

        desired_bank = self.selected_bank
        if next_action == ACTION_INTEGRITY_ROLLBACK:
            desired_bank = lkg_bank
        elif next_action in (ACTION_TAIL_EWMA, ACTION_UNCERTAIN_EWMA, ACTION_LEAKAGE_RESET):
            desired_bank = EWMA_BANK
        elif inputs.router_boundary:
            desired_bank = int(
                bool(inputs.window_prequential_win)
                and inputs.p_smooth >= SMOOTH_BANK_CODE_MIN
            )

        old_pending = self.commit_pending
        old_pending_bank = self.pending_bank
        auto_commit_valid, auto_commit_bank, auto_commit_version = self.peek_auto_commit(
            safe_boundary=safe_boundary,
            active_bank=active_bank,
            active_version=active_version,
        )

        pending = old_pending
        pending_bank = old_pending_bank
        if old_pending and active_bank == old_pending_bank:
            pending = 0
        if desired_bank != active_bank:
            pending = 1
            pending_bank = desired_bank
        elif pending_bank == active_bank:
            pending = 0

        if inputs.posterior_valid:
            self.policy_update_count = min(0xFFFF, self.policy_update_count + 1)
        if next_action != ACTION_OPEN:
            self.fallback_count = min(0xFFFF, self.fallback_count + 1)
        if next_action == ACTION_INTEGRITY_ROLLBACK and self.action != ACTION_INTEGRITY_ROLLBACK:
            self.rollback_count = min(0xFF, self.rollback_count + 1)

        self.action = next_action
        self.reason = next_reason
        self.tail_latched = next_tail
        self.leakage_latched = next_leakage
        self.integrity_latched = next_integrity
        self.enter_run = next_enter
        self.recovery_run = next_recovery
        self.selected_bank = desired_bank
        self.commit_pending = pending
        self.pending_bank = pending_bank

        self._pipe.append(
            (next_action, next_reason, desired_bank, active_bank, active_version)
            if sample_valid
            else None
        )

        core_payload = core_output_word & ((1 << 102) - 1)
        core_valid = core_payload & 1
        action_payload = 0
        if due is not None and core_valid:
            due_action, due_reason, due_bank, due_active_bank, due_version = due
            self.last_source_version = due_version
            action_payload |= 1
            action_payload |= due_action << 1
            action_payload |= due_reason << 4
            action_payload |= due_bank << 8
            action_payload |= int(due_action != ACTION_OPEN) << 9
            action_payload |= ((core_payload >> 4) & 1) << 10
            action_payload |= ((core_payload >> 5) & 1) << 11
            action_payload |= ((core_payload >> 7) & 3) << 12
            action_payload |= ((core_payload >> 1) & 7) << 14
            action_payload |= ((core_payload >> 47) & 7) << 17
            action_payload |= ((core_payload >> 50) & 0x3FFF) << 20
            action_payload |= due_version << 34
            action_payload |= due_active_bank << 50
            action_payload |= (self.policy_update_count & 0x1FFF) << 51
        action_word = action_payload | (crc16_int_little_endian(action_payload, 8) << 64)

        state_payload = 0
        state_payload |= self.action
        state_payload |= self.reason << 3
        state_payload |= self.tail_latched << 7
        state_payload |= self.leakage_latched << 8
        state_payload |= self.integrity_latched << 9
        state_payload |= self.enter_run << 10
        state_payload |= self.recovery_run << 12
        state_payload |= self.selected_bank << 16
        state_payload |= self.commit_pending << 17
        state_payload |= self.pending_bank << 18
        state_payload |= visible_active_bank << 19
        state_payload |= visible_active_version << 20
        state_payload |= self.policy_update_count << 36
        state_payload |= self.fallback_count << 52
        state_payload |= self.rollback_count << 68
        state_word = state_payload | (crc16_int_little_endian(state_payload, 10) << 80)

        version_payload = 0
        version_payload |= visible_active_version
        version_payload |= self.last_source_version << 16
        version_payload |= self.selected_bank << 32
        version_payload |= visible_active_bank << 33
        version_payload |= self.pending_bank << 34
        version_payload |= self.commit_pending << 35
        version_payload |= (self.policy_update_count & 0xFFF) << 36
        version_word = version_payload | (crc16_int_little_endian(version_payload, 6) << 48)

        return RouteAPolicyCycleOutput(
            action=self.action,
            reason=self.reason,
            selected_bank=self.selected_bank,
            tail_latched=self.tail_latched,
            leakage_latched=self.leakage_latched,
            integrity_latched=self.integrity_latched,
            enter_run=self.enter_run,
            recovery_run=self.recovery_run,
            auto_commit_valid=auto_commit_valid,
            auto_commit_bank=auto_commit_bank,
            auto_commit_version=auto_commit_version,
            commit_pending=self.commit_pending,
            action_word=action_word,
            state_word=state_word,
            version_word=version_word,
        )


__all__ = [
    "ACTION_INTEGRITY_ROLLBACK",
    "ACTION_LEAKAGE_RESET",
    "ACTION_OPEN",
    "ACTION_TAIL_EWMA",
    "ACTION_UNCERTAIN_EWMA",
    "ENTER_HYSTERESIS",
    "EWMA_BANK",
    "OOD_CODE_LIMIT",
    "RECOVERY_HYSTERESIS",
    "RouteAFixedPolicyReference",
    "RouteAPolicyCycleOutput",
    "RouteAPolicyInputs",
    "SMOOTH_BANK_CODE_MIN",
    "TAIL_ENTER_CODE",
    "TAIL_EXIT_CODE",
    "UNCERTAINTY_CODE_LIMIT",
    "WINDOW_BANK",
]
