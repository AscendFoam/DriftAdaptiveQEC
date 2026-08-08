"""Independent cycle model for the T6.2.1 production RTL management shell.

This model intentionally does not call the RTL or duplicate its source parser.
It provides a small executable contract for configuration ordering, CRC32,
compare-and-swap commit, safe-boundary deferral, and the retired-bank guard.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Mapping
import zlib


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

MANAGEMENT_SIGNAL_DEFAULTS: dict[str, int] = {
    "safe_boundary": 1,
    "cfg_begin_valid": 0,
    "cfg_begin_bank": 0,
    "cfg_expected_active_version": 0,
    "cfg_new_version": 0,
    "cfg_expected_crc32": 0,
    "cfg_word_valid": 0,
    "cfg_word_phase": 0,
    "cfg_word_address": 0,
    "cfg_word_data": 0,
    "cfg_finalize_valid": 0,
    "cfg_abort_valid": 0,
    "commit_request_valid": 0,
    "commit_request_bank": 0,
    "commit_expected_active_version": 0,
    "commit_new_version": 0,
    "commit_cancel_valid": 0,
    "management_snapshot_request": 0,
}


def crc32_table_words(words: list[int] | tuple[int, ...]) -> int:
    """Return zlib-compatible CRC32 over packed unsigned 22-bit words."""

    payload = bytearray()
    for word in words:
        if not 0 <= int(word) < (1 << 22):
            raise ValueError(f"table word outside unsigned 22-bit range: {word}")
        payload.extend(int(word).to_bytes(3, "little"))
    return zlib.crc32(payload) & 0xFFFFFFFF


def crc16_ccitt_little_endian(payload: int, byte_count: int) -> int:
    crc = 0xFFFF
    for byte_index in range(byte_count):
        octet = (payload >> (8 * byte_index)) & 0xFF
        crc ^= octet << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _sat16_inc(value: int) -> int:
    return min(0xFFFF, value + 1)


@dataclass
class ProductionFastPathManagementReference:
    """Register-accurate reference for ``gkp_fast_path_production_top``."""

    retired_bank_drain_cycles: int = 6
    active_bank: int = 0
    active_version: int = 0
    bank0_trusted: int = 1
    bank1_trusted: int = 1
    bank0_version: int = 0
    bank1_version: int = 1
    cfg_session_active: int = 0
    cfg_staged_bank: int = 0
    cfg_staged_version: int = 0
    cfg_staged_expected_crc32: int = 0
    cfg_running_crc32: int = 0xFFFFFFFF
    cfg_next_phase: int = 0
    cfg_next_address: int = 0
    cfg_word_count: int = 0
    cfg_all_words_received: int = 0
    commit_pending: int = 0
    commit_pending_bank: int = 0
    commit_pending_version: int = 0
    retired_bank_drain_count: int = 0
    protocol_fault_sticky: int = 0
    management_reject_reason: int = REJECT_NONE
    management_reject_count: int = 0
    crc_failure_count: int = 0
    management_snapshot_busy: int = 0
    management_snapshot_byte_index: int = 0
    management_snapshot_payload: int = 0
    management_snapshot_crc: int = 0xFFFF
    management_state_word_reg: int = 0
    _core_commit_ack: int = 0

    @staticmethod
    def _crc32_word22(crc: int, word: int) -> int:
        return zlib.crc32(int(word).to_bytes(3, "little"), crc ^ 0xFFFFFFFF) ^ 0xFFFFFFFF

    def _bank_trusted(self, bank: int) -> int:
        return self.bank1_trusted if bank else self.bank0_trusted

    def _bank_version(self, bank: int) -> int:
        return self.bank1_version if bank else self.bank0_version

    @staticmethod
    def _crc16_byte(crc: int, octet: int) -> int:
        crc ^= (octet & 0xFF) << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        return crc

    @staticmethod
    def _invalidate_staged(next_state: "ProductionFastPathManagementReference", bank: int) -> None:
        if bank:
            next_state.bank1_trusted = 0
        else:
            next_state.bank0_trusted = 0

    @staticmethod
    def _reject(
        next_state: "ProductionFastPathManagementReference",
        pulses: dict[str, int],
        reason: int,
        *,
        sticky: bool = False,
    ) -> None:
        pulses["management_reject"] = 1
        next_state.management_reject_reason = reason
        next_state.management_reject_count = _sat16_inc(next_state.management_reject_count)
        if sticky:
            next_state.protocol_fault_sticky = 1

    def step(self, signals: Mapping[str, int] | None = None) -> dict[str, int]:
        values = MANAGEMENT_SIGNAL_DEFAULTS.copy()
        if signals:
            unknown = set(signals) - set(values)
            if unknown:
                raise KeyError(f"unknown management signals: {sorted(unknown)}")
            values.update({key: int(value) for key, value in signals.items()})

        old = copy(self)
        nxt = copy(self)
        pulses = {
            "cfg_begin_ack": 0,
            "cfg_word_ack": 0,
            "cfg_finalize_ack": 0,
            "cfg_abort_ack": 0,
            "commit_request_ack": 0,
            "commit_complete": 0,
            "commit_cancel_ack": 0,
            "management_snapshot_ack": 0,
            "management_state_valid": 0,
            "management_reject": 0,
        }

        request_count = sum(
            values[name]
            for name in (
                "cfg_begin_valid",
                "cfg_word_valid",
                "cfg_finalize_valid",
                "cfg_abort_valid",
                "commit_request_valid",
                "commit_cancel_valid",
                "management_snapshot_request",
            )
        )
        conflict = request_count > 1

        # The core and wrapper are distinct synchronous blocks.  The wrapper
        # observes the core acknowledgement one clock after the core switches.
        selected_commit = bool(
            old.commit_pending
            and values["safe_boundary"]
            and old._bank_trusted(old.commit_pending_bank)
            and old.commit_pending_bank != old.active_bank
            and old.active_version != 0xFFFF
            and old.commit_pending_version == old.active_version + 1
        )
        nxt._core_commit_ack = int(selected_commit)
        if selected_commit:
            nxt.active_bank = old.commit_pending_bank
            nxt.active_version = old.commit_pending_version

        if old.retired_bank_drain_count:
            nxt.retired_bank_drain_count = old.retired_bank_drain_count - 1
        if old._core_commit_ack:
            nxt.commit_pending = 0
            pulses["commit_complete"] = 1
            nxt.retired_bank_drain_count = self.retired_bank_drain_cycles

        if old.management_snapshot_busy:
            octet = (old.management_snapshot_payload >> (8 * old.management_snapshot_byte_index)) & 0xFF
            crc_next = self._crc16_byte(old.management_snapshot_crc, octet)
            nxt.management_snapshot_crc = crc_next
            if old.management_snapshot_byte_index == 17:
                nxt.management_state_word_reg = old.management_snapshot_payload | (crc_next << 144)
                nxt.management_snapshot_busy = 0
                pulses["management_state_valid"] = 1
            else:
                nxt.management_snapshot_byte_index = old.management_snapshot_byte_index + 1

        if conflict:
            self._reject(nxt, pulses, REJECT_CONFLICT, sticky=True)
            if old.cfg_session_active:
                nxt.cfg_session_active = 0
                nxt.cfg_all_words_received = 0
                self._invalidate_staged(nxt, old.cfg_staged_bank)
            nxt.commit_pending = 0
        elif values["management_snapshot_request"]:
            if old.management_snapshot_busy:
                self._reject(nxt, pulses, REJECT_BUSY)
            else:
                nxt.management_snapshot_payload = old.management_state_payload()
                nxt.management_snapshot_crc = 0xFFFF
                nxt.management_snapshot_byte_index = 0
                nxt.management_snapshot_busy = 1
                pulses["management_snapshot_ack"] = 1
        elif values["cfg_abort_valid"]:
            if old.cfg_session_active:
                nxt.cfg_session_active = 0
                nxt.cfg_all_words_received = 0
                self._invalidate_staged(nxt, old.cfg_staged_bank)
                pulses["cfg_abort_ack"] = 1
            else:
                self._reject(nxt, pulses, REJECT_NO_SESSION, sticky=True)
        elif values["cfg_begin_valid"]:
            if old.cfg_session_active or old.commit_pending:
                self._reject(nxt, pulses, REJECT_BUSY)
            elif old.retired_bank_drain_count:
                self._reject(nxt, pulses, REJECT_DRAIN_GUARD)
            elif values["cfg_begin_bank"] == old.active_bank:
                self._reject(nxt, pulses, REJECT_ACTIVE_BANK, sticky=True)
            elif (
                old.active_version == 0xFFFF
                or values["cfg_expected_active_version"] != old.active_version
                or values["cfg_new_version"] != old.active_version + 1
            ):
                self._reject(nxt, pulses, REJECT_VERSION)
            else:
                nxt.cfg_session_active = 1
                nxt.cfg_staged_bank = values["cfg_begin_bank"]
                nxt.cfg_staged_version = values["cfg_new_version"]
                nxt.cfg_staged_expected_crc32 = values["cfg_expected_crc32"] & 0xFFFFFFFF
                nxt.cfg_running_crc32 = 0xFFFFFFFF
                nxt.cfg_next_phase = 0
                nxt.cfg_next_address = 0
                nxt.cfg_word_count = 0
                nxt.cfg_all_words_received = 0
                self._invalidate_staged(nxt, values["cfg_begin_bank"])
                pulses["cfg_begin_ack"] = 1
        elif values["cfg_word_valid"]:
            exact = bool(
                old.cfg_session_active
                and values["cfg_word_phase"] == old.cfg_next_phase
                and values["cfg_word_address"] == old.cfg_next_address
                and values["cfg_word_address"] <= 256
                and old.cfg_staged_bank != old.active_bank
                and not old.commit_pending
            )
            if not old.cfg_session_active:
                self._reject(nxt, pulses, REJECT_NO_SESSION, sticky=True)
            elif not exact or old.cfg_all_words_received:
                self._reject(nxt, pulses, REJECT_WORD_ORDER, sticky=True)
                nxt.cfg_session_active = 0
                nxt.cfg_all_words_received = 0
                self._invalidate_staged(nxt, old.cfg_staged_bank)
            else:
                nxt.cfg_running_crc32 = self._crc32_word22(
                    old.cfg_running_crc32, values["cfg_word_data"]
                )
                nxt.cfg_word_count = old.cfg_word_count + 1
                pulses["cfg_word_ack"] = 1
                if not old.cfg_next_phase and old.cfg_next_address == 256:
                    nxt.cfg_next_phase = 1
                    nxt.cfg_next_address = 0
                elif old.cfg_next_phase and old.cfg_next_address == 256:
                    nxt.cfg_all_words_received = 1
                else:
                    nxt.cfg_next_address = old.cfg_next_address + 1
        elif values["cfg_finalize_valid"]:
            if not old.cfg_session_active:
                self._reject(nxt, pulses, REJECT_NO_SESSION)
            elif not old.cfg_all_words_received or old.cfg_word_count != 514:
                self._reject(nxt, pulses, REJECT_INCOMPLETE, sticky=True)
                nxt.cfg_session_active = 0
                nxt.cfg_all_words_received = 0
                self._invalidate_staged(nxt, old.cfg_staged_bank)
            elif (old.cfg_running_crc32 ^ 0xFFFFFFFF) != old.cfg_staged_expected_crc32:
                self._reject(nxt, pulses, REJECT_CRC32, sticky=True)
                nxt.crc_failure_count = _sat16_inc(old.crc_failure_count)
                nxt.cfg_session_active = 0
                nxt.cfg_all_words_received = 0
                self._invalidate_staged(nxt, old.cfg_staged_bank)
            else:
                nxt.cfg_session_active = 0
                nxt.cfg_all_words_received = 0
                pulses["cfg_finalize_ack"] = 1
                if old.cfg_staged_bank:
                    nxt.bank1_trusted = 1
                    nxt.bank1_version = old.cfg_staged_version
                else:
                    nxt.bank0_trusted = 1
                    nxt.bank0_version = old.cfg_staged_version
        elif values["commit_cancel_valid"]:
            if old.commit_pending:
                nxt.commit_pending = 0
                pulses["commit_cancel_ack"] = 1
            else:
                self._reject(nxt, pulses, REJECT_NO_PENDING)
        elif values["commit_request_valid"]:
            bank = values["commit_request_bank"]
            if old.cfg_session_active or old.commit_pending:
                self._reject(nxt, pulses, REJECT_BUSY)
            elif old.retired_bank_drain_count:
                self._reject(nxt, pulses, REJECT_DRAIN_GUARD)
            elif bank == old.active_bank:
                self._reject(nxt, pulses, REJECT_ACTIVE_BANK)
            elif not old._bank_trusted(bank):
                self._reject(nxt, pulses, REJECT_UNTRUSTED)
            elif (
                old.active_version == 0xFFFF
                or values["commit_expected_active_version"] != old.active_version
                or values["commit_new_version"] != old.active_version + 1
                or old._bank_version(bank) != values["commit_new_version"]
            ):
                self._reject(nxt, pulses, REJECT_VERSION)
            else:
                nxt.commit_pending = 1
                nxt.commit_pending_bank = bank
                nxt.commit_pending_version = values["commit_new_version"]
                pulses["commit_request_ack"] = 1

        self.__dict__.update(nxt.__dict__)
        return {
            **pulses,
            "management_reject_reason": self.management_reject_reason,
            "cfg_session_active": self.cfg_session_active,
            "commit_pending": self.commit_pending,
            "management_snapshot_busy": self.management_snapshot_busy,
            "active_bank": self.active_bank,
            "active_version": self.active_version,
            "management_state_word": self.management_state_word_reg,
        }

    def management_state_payload(self) -> int:
        payload = 0
        payload |= self.active_bank
        payload |= self.active_version << 1
        payload |= self.bank0_trusted << 17
        payload |= self.bank1_trusted << 18
        payload |= self.bank0_version << 19
        payload |= self.bank1_version << 35
        payload |= self.cfg_session_active << 51
        payload |= self.cfg_staged_bank << 52
        payload |= self.cfg_word_count << 53
        payload |= self.cfg_next_phase << 63
        payload |= self.cfg_next_address << 64
        payload |= self.cfg_all_words_received << 73
        payload |= self.commit_pending << 74
        payload |= self.commit_pending_bank << 75
        payload |= self.commit_pending_version << 76
        payload |= self.retired_bank_drain_count << 92
        payload |= self.management_reject_reason << 96
        payload |= self.management_reject_count << 104
        payload |= self.crc_failure_count << 120
        payload |= self.protocol_fault_sticky << 136
        payload |= self.management_snapshot_busy << 137
        return payload

    def current_management_state_word(self) -> int:
        """Convenience value expected from a newly completed snapshot."""

        payload = self.management_state_payload()
        return payload | (crc16_ccitt_little_endian(payload, 18) << 144)
