"""Independent cycle reference for the converged single-mode production top.

The core and Route-A policy references predate the converged RTL.  This module
adds a separately written admission/management specification and composes all
three at the clock boundary.  It intentionally does not parse Verilog or reuse
the RTL manager implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cnn_fpga.runtime.fast_production_core_reference import (
    FastCoreCycleOutput,
    FastProductionCoreReference,
)
from cnn_fpga.runtime.route_a_fixed_policy_reference import (
    ACTION_OPEN,
    RouteAFixedPolicyReference,
    RouteAPolicyCycleOutput,
    RouteAPolicyInputs,
)


REJECT_NONE = 0x00
REJECT_CONFLICT = 0x01
REJECT_BUSY = 0x02
REJECT_ACTIVE_BANK = 0x03
REJECT_VERSION = 0x04
REJECT_DRAIN_GUARD = 0x05
REJECT_NO_SESSION = 0x06
REJECT_WORD_ORDER = 0x07
REJECT_CRC32 = 0x08
REJECT_INCOMPLETE = 0x09
REJECT_NO_PENDING = 0x0A
REJECT_UNTRUSTED = 0x0B


def _sat16(value: int) -> int:
    return min(0xFFFF, int(value) + 1)


def crc16_byte(crc_in: int, octet: int) -> int:
    crc = (int(crc_in) ^ ((int(octet) & 0xFF) << 8)) & 0xFFFF
    for _ in range(8):
        crc = (((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)) & 0xFFFF
    return crc


def crc32_byte(crc_in: int, octet: int) -> int:
    crc = int(crc_in) & 0xFFFFFFFF
    data = int(octet) & 0xFF
    for _ in range(8):
        crc = ((crc >> 1) ^ 0xEDB88320) if ((crc ^ data) & 1) else (crc >> 1)
        data >>= 1
    return crc & 0xFFFFFFFF


def crc32_word22(crc_in: int, word: int) -> int:
    value = int(word) & 0x3FFFFF
    crc = crc32_byte(crc_in, value)
    crc = crc32_byte(crc, value >> 8)
    return crc32_byte(crc, (value >> 16) & 0x3F)


def image_crc32(words: list[int] | tuple[int, ...]) -> int:
    crc = 0xFFFFFFFF
    for word in words:
        crc = crc32_word22(crc, word)
    return crc ^ 0xFFFFFFFF


@dataclass(slots=True, frozen=True)
class ConvergedInputs:
    in_valid: int = 0
    in_word: int = 0
    safe_boundary: int = 1
    cfg_begin_valid: int = 0
    cfg_begin_bank: int = 0
    cfg_expected_active_version: int = 0
    cfg_new_image_version: int = 0
    cfg_expected_crc32: int = 0
    cfg_word_valid: int = 0
    cfg_word_phase: int = 0
    cfg_word_address: int = 0
    cfg_word_data: int = 0
    cfg_finalize_valid: int = 0
    cfg_abort_valid: int = 0
    host_commit_valid: int = 0
    host_commit_bank: int = 0
    host_expected_active_version: int = 0
    host_new_activation_version: int = 0
    commit_cancel_valid: int = 0
    management_snapshot_request: int = 0
    posterior: RouteAPolicyInputs = RouteAPolicyInputs()


@dataclass(slots=True, frozen=True)
class AdmissionOutput:
    host_commit_blocked: int
    effective_commit_valid: int
    effective_commit_source_policy: int
    effective_commit_bank: int
    effective_expected_active_version: int
    effective_new_activation_version: int


@dataclass(slots=True, frozen=True)
class ManagerCombinational:
    core_cfg_we: int
    core_cfg_bank: int
    core_cfg_phase: int
    core_cfg_address: int
    core_cfg_data: int
    bank0_trusted: int
    bank1_trusted: int
    core_commit_valid: int
    core_commit_bank: int
    core_commit_version: int
    management_ready: int


@dataclass(slots=True, frozen=True)
class ConvergedCycleOutput:
    pulses: dict[str, int]
    core: FastCoreCycleOutput
    route: RouteAPolicyCycleOutput
    management_state_word: int
    manager_debug: dict[str, int]
    core_interface_debug: dict[str, int]
    admission_debug: AdmissionOutput


def admit_commit(
    inputs: ConvergedInputs,
    *,
    policy_commit_valid: int,
    policy_commit_bank: int,
    policy_commit_version: int,
    policy_commit_pending: int,
    policy_action: int,
) -> AdmissionOutput:
    host_allowed = int(
        bool(inputs.host_commit_valid)
        and not bool(policy_commit_valid)
        and not bool(policy_commit_pending)
        and int(policy_action) == ACTION_OPEN
    )
    source_policy = int(bool(policy_commit_valid))
    return AdmissionOutput(
        host_commit_blocked=int(bool(inputs.host_commit_valid) and not host_allowed),
        effective_commit_valid=int(bool(policy_commit_valid) or bool(host_allowed)),
        effective_commit_source_policy=source_policy,
        effective_commit_bank=(
            int(policy_commit_bank) if source_policy else int(inputs.host_commit_bank)
        ),
        effective_expected_active_version=(
            (int(policy_commit_version) - 1) & 0xFFFF
            if source_policy
            else int(inputs.host_expected_active_version)
        ),
        effective_new_activation_version=(
            int(policy_commit_version)
            if source_policy
            else int(inputs.host_new_activation_version)
        ),
    )


class IndependentParameterBankManager:
    """Clock-accurate executable specification of the production manager."""

    def __init__(self, *, words_per_phase: int = 257, drain_cycles: int = 6) -> None:
        if words_per_phase <= 0 or words_per_phase > 257:
            raise ValueError("words_per_phase must be in [1,257]")
        if drain_cycles < 0 or drain_cycles > 15:
            raise ValueError("drain_cycles must fit four bits")
        self.words_per_phase = int(words_per_phase)
        self.drain_cycles = int(drain_cycles)
        self.reset()

    def reset(self) -> None:
        self.bank_trusted = [1, 1]
        self.bank_image_version = [0, 1]
        self.cfg_session_active = 0
        self.cfg_staged_bank = 0
        self.cfg_staged_image_version = 0
        self.cfg_staged_expected_crc32 = 0
        self.cfg_running_crc32 = 0xFFFFFFFF
        self.cfg_next_phase = 0
        self.cfg_next_address = 0
        self.cfg_word_count = 0
        self.cfg_all_words_received = 0
        self.commit_pending = 0
        self.commit_pending_bank = 0
        self.commit_pending_version = 0
        self.commit_pending_source_policy = 0
        self.retired_bank_drain_count = 0
        self.protocol_fault_sticky = 0
        self.management_reject_count = 0
        self.crc_failure_count = 0
        self.management_snapshot_busy = 0
        self.management_snapshot_byte_index = 0
        self.management_snapshot_payload = 0
        self.management_snapshot_shift = 0
        self.management_snapshot_crc = 0xFFFF
        self.management_state_word = 0
        self.pulses = self._blank_pulses()

    @staticmethod
    def _blank_pulses() -> dict[str, int]:
        return {
            "cfg_begin_ack": 0,
            "cfg_word_ack": 0,
            "cfg_finalize_ack": 0,
            "cfg_abort_ack": 0,
            "commit_request_ack": 0,
            "commit_request_ack_source_policy": 0,
            "commit_complete": 0,
            "commit_complete_source_policy": 0,
            "commit_cancel_ack": 0,
            "management_snapshot_ack": 0,
            "management_state_valid": 0,
            "management_reject": 0,
            "management_reject_reason": REJECT_NONE,
        }

    @staticmethod
    def _request_count(inputs: ConvergedInputs, admission: AdmissionOutput) -> int:
        return sum(
            int(bool(value))
            for value in (
                inputs.cfg_begin_valid,
                inputs.cfg_word_valid,
                inputs.cfg_finalize_valid,
                inputs.cfg_abort_valid,
                admission.effective_commit_valid,
                inputs.commit_cancel_valid,
                inputs.management_snapshot_request,
            )
        )

    def _word_exact(self, inputs: ConvergedInputs, core_active_bank: int) -> bool:
        return bool(
            self.cfg_session_active
            and int(inputs.cfg_word_phase) == self.cfg_next_phase
            and int(inputs.cfg_word_address) == self.cfg_next_address
            and int(inputs.cfg_word_address) < self.words_per_phase
            and self.cfg_staged_bank != int(core_active_bank)
            and not self.commit_pending
        )

    def peek(
        self,
        inputs: ConvergedInputs,
        *,
        admission: AdmissionOutput,
        core_active_bank: int,
        core_active_version: int,
    ) -> ManagerCombinational:
        conflict = self._request_count(inputs, admission) > 1
        word_exact = self._word_exact(inputs, core_active_bank)
        commit_valid = int(
            bool(self.commit_pending)
            and bool(inputs.safe_boundary)
            and self.commit_pending_bank != int(core_active_bank)
            and int(core_active_version) != 0xFFFF
            and self.commit_pending_version == int(core_active_version) + 1
            and bool(self.bank_trusted[self.commit_pending_bank])
        )
        return ManagerCombinational(
            core_cfg_we=int(bool(inputs.cfg_word_valid) and not conflict and word_exact),
            core_cfg_bank=self.cfg_staged_bank,
            core_cfg_phase=int(inputs.cfg_word_phase),
            core_cfg_address=int(inputs.cfg_word_address),
            core_cfg_data=int(inputs.cfg_word_data) & 0x3FFFFF,
            bank0_trusted=self.bank_trusted[0],
            bank1_trusted=self.bank_trusted[1],
            core_commit_valid=commit_valid,
            core_commit_bank=self.commit_pending_bank,
            core_commit_version=self.commit_pending_version,
            management_ready=int(
                not self.cfg_session_active
                and not self.commit_pending
                and not self.management_snapshot_busy
                and self.retired_bank_drain_count == 0
            ),
        )

    def _state_payload(self, core_active_bank: int, core_active_version: int) -> int:
        payload = int(core_active_bank)
        payload |= int(core_active_version) << 1
        payload |= self.bank_trusted[0] << 17
        payload |= self.bank_trusted[1] << 18
        payload |= self.bank_image_version[0] << 19
        payload |= self.bank_image_version[1] << 35
        payload |= self.cfg_session_active << 51
        payload |= self.cfg_staged_bank << 52
        payload |= self.cfg_word_count << 53
        payload |= self.cfg_next_phase << 63
        payload |= self.cfg_next_address << 64
        payload |= self.cfg_all_words_received << 73
        payload |= self.commit_pending << 74
        payload |= self.commit_pending_bank << 75
        payload |= self.commit_pending_version << 76
        payload |= self.commit_pending_source_policy << 92
        payload |= self.retired_bank_drain_count << 93
        payload |= int(self.pulses["management_reject_reason"]) << 97
        payload |= self.management_reject_count << 105
        payload |= self.crc_failure_count << 121
        payload |= self.protocol_fault_sticky << 137
        payload |= self.management_snapshot_busy << 138
        return payload

    def _reject(self, reason: int, *, sticky: bool = False, crc: bool = False) -> None:
        self.pulses["management_reject"] = 1
        self.pulses["management_reject_reason"] = int(reason)
        self.management_reject_count = _sat16(self.management_reject_count)
        if sticky:
            self.protocol_fault_sticky = 1
        if crc:
            self.crc_failure_count = _sat16(self.crc_failure_count)

    def step(
        self,
        inputs: ConvergedInputs,
        *,
        admission: AdmissionOutput,
        core_active_bank: int,
        core_active_version: int,
        core_commit_ack: int,
    ) -> None:
        # Conditions in the RTL always block observe the pre-edge state.
        old: dict[str, Any] = {
            "cfg_session_active": self.cfg_session_active,
            "cfg_all_words_received": self.cfg_all_words_received,
            "cfg_staged_bank": self.cfg_staged_bank,
            "commit_pending": self.commit_pending,
            "snapshot_busy": self.management_snapshot_busy,
            "snapshot_index": self.management_snapshot_byte_index,
            "snapshot_payload": self.management_snapshot_payload,
            "snapshot_shift": self.management_snapshot_shift,
            "snapshot_crc": self.management_snapshot_crc,
            "drain": self.retired_bank_drain_count,
            "running_crc": self.cfg_running_crc32,
            "word_count": self.cfg_word_count,
            "next_phase": self.cfg_next_phase,
            "next_address": self.cfg_next_address,
            "reject_reason": self.pulses["management_reject_reason"],
        }
        state_payload_pre = self._state_payload(core_active_bank, core_active_version)
        conflict = self._request_count(inputs, admission) > 1
        word_exact = self._word_exact(inputs, core_active_bank)
        self.pulses = self._blank_pulses()

        if old["snapshot_busy"]:
            crc_next = crc16_byte(old["snapshot_crc"], old["snapshot_shift"] & 0xFF)
            self.management_snapshot_crc = crc_next
            self.management_snapshot_shift = old["snapshot_shift"] >> 8
            if old["snapshot_index"] == 17:
                self.management_state_word = (crc_next << 144) | old["snapshot_payload"]
                self.management_snapshot_busy = 0
                self.pulses["management_state_valid"] = 1
            else:
                self.management_snapshot_byte_index = old["snapshot_index"] + 1

        if old["drain"]:
            self.retired_bank_drain_count = old["drain"] - 1

        if core_commit_ack:
            self.commit_pending = 0
            self.pulses["commit_complete"] = 1
            self.pulses["commit_complete_source_policy"] = self.commit_pending_source_policy
            self.retired_bank_drain_count = self.drain_cycles

        if conflict:
            self._reject(REJECT_CONFLICT, sticky=True)
            if old["cfg_session_active"]:
                self.cfg_session_active = 0
                self.cfg_all_words_received = 0
                self.bank_trusted[old["cfg_staged_bank"]] = 0
            self.commit_pending = 0
        elif inputs.management_snapshot_request:
            if old["snapshot_busy"]:
                self._reject(REJECT_BUSY)
            else:
                self.management_snapshot_payload = state_payload_pre
                self.management_snapshot_shift = state_payload_pre
                self.management_snapshot_crc = 0xFFFF
                self.management_snapshot_byte_index = 0
                self.management_snapshot_busy = 1
                self.pulses["management_snapshot_ack"] = 1
        elif inputs.cfg_abort_valid:
            if old["cfg_session_active"]:
                self.cfg_session_active = 0
                self.cfg_all_words_received = 0
                self.pulses["cfg_abort_ack"] = 1
                self.bank_trusted[old["cfg_staged_bank"]] = 0
            else:
                self._reject(REJECT_NO_SESSION, sticky=True)
        elif inputs.cfg_begin_valid:
            maximum_image_version = max(self.bank_image_version)
            if old["cfg_session_active"] or old["commit_pending"] or old["snapshot_busy"]:
                self._reject(REJECT_BUSY)
            elif old["drain"]:
                self._reject(REJECT_DRAIN_GUARD)
            elif int(inputs.cfg_begin_bank) == int(core_active_bank):
                self._reject(REJECT_ACTIVE_BANK, sticky=True)
            elif (
                int(core_active_version) == 0xFFFF
                or int(inputs.cfg_expected_active_version) != int(core_active_version)
                or maximum_image_version == 0xFFFF
                or int(inputs.cfg_new_image_version) <= maximum_image_version
            ):
                self._reject(REJECT_VERSION)
            else:
                self.cfg_session_active = 1
                self.cfg_staged_bank = int(inputs.cfg_begin_bank)
                self.cfg_staged_image_version = int(inputs.cfg_new_image_version)
                self.cfg_staged_expected_crc32 = int(inputs.cfg_expected_crc32) & 0xFFFFFFFF
                self.cfg_running_crc32 = 0xFFFFFFFF
                self.cfg_next_phase = 0
                self.cfg_next_address = 0
                self.cfg_word_count = 0
                self.cfg_all_words_received = 0
                self.pulses["cfg_begin_ack"] = 1
                self.bank_trusted[self.cfg_staged_bank] = 0
        elif inputs.cfg_word_valid:
            if not old["cfg_session_active"]:
                self._reject(REJECT_NO_SESSION, sticky=True)
            elif not word_exact or old["cfg_all_words_received"]:
                self._reject(REJECT_WORD_ORDER, sticky=True)
                self.cfg_session_active = 0
                self.cfg_all_words_received = 0
                self.bank_trusted[old["cfg_staged_bank"]] = 0
            else:
                self.cfg_running_crc32 = crc32_word22(old["running_crc"], inputs.cfg_word_data)
                self.cfg_word_count = (old["word_count"] + 1) & 0x3FF
                self.pulses["cfg_word_ack"] = 1
                if not old["next_phase"] and old["next_address"] == self.words_per_phase - 1:
                    self.cfg_next_phase = 1
                    self.cfg_next_address = 0
                elif old["next_phase"] and old["next_address"] == self.words_per_phase - 1:
                    self.cfg_all_words_received = 1
                else:
                    self.cfg_next_address = old["next_address"] + 1
        elif inputs.cfg_finalize_valid:
            if not old["cfg_session_active"]:
                self._reject(REJECT_NO_SESSION)
            elif not old["cfg_all_words_received"] or old["word_count"] != 2 * self.words_per_phase:
                self._reject(REJECT_INCOMPLETE, sticky=True)
                self.cfg_session_active = 0
                self.cfg_all_words_received = 0
                self.bank_trusted[old["cfg_staged_bank"]] = 0
            elif (old["running_crc"] ^ 0xFFFFFFFF) != self.cfg_staged_expected_crc32:
                self._reject(REJECT_CRC32, sticky=True, crc=True)
                self.cfg_session_active = 0
                self.cfg_all_words_received = 0
                self.bank_trusted[old["cfg_staged_bank"]] = 0
            else:
                self.cfg_session_active = 0
                self.cfg_all_words_received = 0
                self.pulses["cfg_finalize_ack"] = 1
                self.bank_trusted[old["cfg_staged_bank"]] = 1
                self.bank_image_version[old["cfg_staged_bank"]] = self.cfg_staged_image_version
        elif inputs.commit_cancel_valid:
            if old["commit_pending"]:
                self.commit_pending = 0
                self.pulses["commit_cancel_ack"] = 1
            else:
                self._reject(REJECT_NO_PENDING)
        elif admission.effective_commit_valid:
            target = int(admission.effective_commit_bank)
            if old["cfg_session_active"] or old["commit_pending"] or old["snapshot_busy"]:
                self._reject(REJECT_BUSY)
            elif old["drain"]:
                self._reject(REJECT_DRAIN_GUARD)
            elif target == int(core_active_bank):
                self._reject(REJECT_ACTIVE_BANK)
            elif not self.bank_trusted[target]:
                self._reject(REJECT_UNTRUSTED)
            elif (
                int(core_active_version) == 0xFFFF
                or int(admission.effective_expected_active_version) != int(core_active_version)
                or int(admission.effective_new_activation_version) != int(core_active_version) + 1
            ):
                self._reject(REJECT_VERSION)
            else:
                self.commit_pending = 1
                self.commit_pending_bank = target
                self.commit_pending_version = int(admission.effective_new_activation_version)
                self.commit_pending_source_policy = int(admission.effective_commit_source_policy)
                self.pulses["commit_request_ack"] = 1
                self.pulses["commit_request_ack_source_policy"] = int(
                    admission.effective_commit_source_policy
                )

    def debug(self) -> dict[str, int]:
        return {
            "cfg_session_active": self.cfg_session_active,
            "cfg_staged_bank": self.cfg_staged_bank,
            "cfg_word_count": self.cfg_word_count,
            "cfg_all_words_received": self.cfg_all_words_received,
            "commit_pending": self.commit_pending,
            "commit_pending_bank": self.commit_pending_bank,
            "commit_pending_version": self.commit_pending_version,
            "commit_pending_source_policy": self.commit_pending_source_policy,
            "retired_bank_drain_count": self.retired_bank_drain_count,
            "bank0_trusted": self.bank_trusted[0],
            "bank1_trusted": self.bank_trusted[1],
            "bank0_image_version": self.bank_image_version[0],
            "bank1_image_version": self.bank_image_version[1],
        }


class ConvergedProductionReference:
    """Independent composition with explicit pre/post-edge signal ordering."""

    def __init__(self, tables: list[list[list[int]]]) -> None:
        self.core = FastProductionCoreReference(tables)
        self.policy = RouteAFixedPolicyReference()
        self.manager = IndependentParameterBankManager()
        self._registered_core_commit_ack = 0

    def reset(self) -> None:
        self.core.reset()
        self.policy.reset()
        self.manager.reset()
        self._registered_core_commit_ack = 0

    def step(self, inputs: ConvergedInputs) -> ConvergedCycleOutput:
        pre_bank = self.core.active_bank
        pre_version = self.core.active_version
        policy_valid, policy_bank, policy_version = self.policy.peek_auto_commit(
            safe_boundary=inputs.safe_boundary,
            active_bank=pre_bank,
            active_version=pre_version,
        )
        admission_pre = admit_commit(
            inputs,
            policy_commit_valid=policy_valid,
            policy_commit_bank=policy_bank,
            policy_commit_version=policy_version,
            policy_commit_pending=self.policy.commit_pending,
            policy_action=self.policy.action,
        )
        manager_pre = self.manager.peek(
            inputs,
            admission=admission_pre,
            core_active_bank=pre_bank,
            core_active_version=pre_version,
        )
        core = self.core.step(
            in_valid=inputs.in_valid,
            in_word=inputs.in_word,
            safe_boundary=inputs.safe_boundary,
            commit_valid=manager_pre.core_commit_valid,
            commit_bank=manager_pre.core_commit_bank,
            commit_version=manager_pre.core_commit_version,
            cfg_we=manager_pre.core_cfg_we,
            cfg_bank=manager_pre.core_cfg_bank,
            cfg_phase=manager_pre.core_cfg_phase,
            cfg_address=manager_pre.core_cfg_address,
            cfg_data=manager_pre.core_cfg_data,
            bank0_trusted=manager_pre.bank0_trusted,
            bank1_trusted=manager_pre.bank1_trusted,
        )
        self.manager.step(
            inputs,
            admission=admission_pre,
            core_active_bank=pre_bank,
            core_active_version=pre_version,
            core_commit_ack=self._registered_core_commit_ack,
        )
        route = self.policy.step(
            inputs.posterior,
            sample_valid=inputs.in_valid,
            safe_boundary=inputs.safe_boundary,
            active_bank=pre_bank,
            active_version=pre_version,
            core_output_word=core.output_word,
            visible_active_bank=core.active_bank,
            visible_active_version=core.active_version,
        )
        self._registered_core_commit_ack = core.commit_ack

        post_policy_valid, post_policy_bank, post_policy_version = self.policy.peek_auto_commit(
            safe_boundary=inputs.safe_boundary,
            active_bank=core.active_bank,
            active_version=core.active_version,
        )
        admission_post = admit_commit(
            inputs,
            policy_commit_valid=post_policy_valid,
            policy_commit_bank=post_policy_bank,
            policy_commit_version=post_policy_version,
            policy_commit_pending=self.policy.commit_pending,
            policy_action=self.policy.action,
        )
        manager_post = self.manager.peek(
            inputs,
            admission=admission_post,
            core_active_bank=core.active_bank,
            core_active_version=core.active_version,
        )
        pulses = dict(self.manager.pulses)
        pulses["host_commit_ack"] = int(
            bool(pulses["commit_request_ack"])
            and not bool(pulses["commit_request_ack_source_policy"])
        )
        pulses["policy_commit_ack"] = int(
            bool(pulses["commit_request_ack"])
            and bool(pulses["commit_request_ack_source_policy"])
        )
        pulses["management_ready"] = manager_post.management_ready
        pulses["host_commit_blocked"] = admission_post.host_commit_blocked
        return ConvergedCycleOutput(
            pulses=pulses,
            core=core,
            route=route,
            management_state_word=self.manager.management_state_word,
            manager_debug=self.manager.debug(),
            core_interface_debug={
                "core_cfg_we": manager_post.core_cfg_we,
                "core_cfg_bank": manager_post.core_cfg_bank,
                "core_commit_valid": manager_post.core_commit_valid,
                "core_commit_bank": manager_post.core_commit_bank,
                "core_commit_version": manager_post.core_commit_version,
            },
            admission_debug=admission_post,
        )


__all__ = [
    "AdmissionOutput",
    "ConvergedCycleOutput",
    "ConvergedInputs",
    "ConvergedProductionReference",
    "IndependentParameterBankManager",
    "REJECT_ACTIVE_BANK",
    "REJECT_BUSY",
    "REJECT_CONFLICT",
    "REJECT_CRC32",
    "REJECT_DRAIN_GUARD",
    "REJECT_INCOMPLETE",
    "REJECT_NONE",
    "REJECT_NO_PENDING",
    "REJECT_NO_SESSION",
    "REJECT_UNTRUSTED",
    "REJECT_VERSION",
    "REJECT_WORD_ORDER",
    "admit_commit",
    "crc16_byte",
    "crc32_byte",
    "crc32_word22",
    "image_crc32",
]
