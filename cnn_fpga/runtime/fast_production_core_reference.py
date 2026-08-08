"""High-throughput independent integer reference for production fast-path RTL.

The implementation is intentionally handwritten from the frozen word and FSM
contract.  It does not call ``BitAccurateHardwareReference`` or parse RTL.
T6.2.2 cross-checks it against both that legacy golden and CXXRTL before using
it to generate million-cycle binary qualification vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LLR_MIN = -(1 << 21)
LLR_MAX = (1 << 21) - 1
LLR_MASK = (1 << 22) - 1

MODE_NORMAL = 0
MODE_X_RECOVERY = 1
MODE_Z_RECOVERY = 2
MODE_HOLD = 3
MODE_RESET_REQUEST = 4
MODE_FALLBACK = 5

HEALTH_HEALTHY = 0
HEALTH_DEGRADED = 1
HEALTH_RECOVERING = 2
HEALTH_FALLBACK = 3
HEALTH_RESET_REQUIRED = 4


def _crc16_table() -> tuple[int, ...]:
    values = []
    for octet in range(256):
        crc = octet << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        values.append(crc)
    return tuple(values)


CRC16_TABLE = _crc16_table()


def crc16_int_little_endian(payload: int, byte_count: int) -> int:
    crc = 0xFFFF
    for byte_index in range(byte_count):
        octet = (payload >> (8 * byte_index)) & 0xFF
        crc = ((crc << 8) & 0xFFFF) ^ CRC16_TABLE[((crc >> 8) ^ octet) & 0xFF]
    return crc


def encode_fast_input_word(
    *,
    syndrome_code: int,
    syndrome_x_code: int,
    syndrome_z_code: int,
    phase: int,
    ood_score: int,
    parameter_age: int,
    reset_ack: int = 0,
    observation_valid: int = 1,
    deadline_ok: int = 1,
) -> int:
    fields = (
        (syndrome_code, 10),
        (syndrome_x_code, 2),
        (syndrome_z_code, 2),
        (phase, 1),
        (ood_score, 8),
        (parameter_age, 16),
        (reset_ack, 1),
        (observation_valid, 1),
        (deadline_ok, 1),
    )
    payload = 0
    offset = 0
    for value, width in fields:
        if not 0 <= int(value) < (1 << width):
            raise ValueError(f"input field {value} does not fit {width} bits")
        payload |= int(value) << offset
        offset += width
    return payload | (crc16_int_little_endian(payload, 6) << 42)


def corrupt_input_crc(word: int, mask: int = 1 << 42) -> int:
    if not 0 <= word < (1 << 58):
        raise ValueError("input word must be 58 bits")
    return word ^ mask


def _signed22(raw: int) -> int:
    raw &= LLR_MASK
    return raw - (1 << 22) if raw & (1 << 21) else raw


def _round_shift3_ties_even(value: int) -> int:
    negative = value < 0
    magnitude = -value if negative else value
    quotient, remainder = divmod(magnitude, 8)
    if remainder > 4 or (remainder == 4 and quotient & 1):
        quotient += 1
    return -quotient if negative else quotient


def _sat3(value: int) -> int:
    return min(7, value + 1)


def _sat8(value: int) -> int:
    return min(255, value + 1)


def load_frozen_rtl_tables(root: Path) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    generated = root / "cnn_fpga/rtl/generated"
    banks = []
    for bank in (0, 1):
        phases = []
        for phase in ("x", "z"):
            path = generated / f"t5_5_1_bank{bank}_{phase}.mem"
            raw = [int(line, 16) for line in path.read_text(encoding="ascii").splitlines()]
            if len(raw) != 257:
                raise ValueError(f"{path} must contain 257 words")
            phases.append(tuple(_signed22(value) for value in raw))
        banks.append((phases[0], phases[1]))
    return tuple(banks)


@dataclass(slots=True, frozen=True)
class FastCoreCycleOutput:
    commit_ack: int
    active_bank: int
    active_version: int
    map_valid: int
    map_address: int
    map_llr_twos: int
    output_word: int
    state_word: int
    fault_mask: int
    correction_enable: int
    reset_request: int


@dataclass(slots=True, frozen=True)
class _Request:
    address: int
    llr_bits: int
    x_code: int
    z_code: int
    phase: int
    ood: int
    age: int
    reset_ack: int
    observation_valid: int
    deadline_ok: int
    crc_ok: int
    bank_trusted: int
    version: int


class FastProductionCoreReference:
    """Register-accurate core model optimized for long qualification traces."""

    def __init__(
        self,
        tables: Sequence[Sequence[Sequence[int]]],
        *,
        ood_threshold: int = 192,
        max_parameter_age: int = 8192,
        max_trusted_version: int = 0xFFFF,
    ) -> None:
        if len(tables) != 2 or any(len(bank) != 2 for bank in tables):
            raise ValueError("tables must be [2 banks][2 phases][257 entries]")
        self.tables = [
            [list(map(int, bank[0])), list(map(int, bank[1]))]
            for bank in tables
        ]
        if any(len(phase) != 257 for bank in self.tables for phase in bank):
            raise ValueError("each table must contain 257 entries")
        self.ood_threshold = int(ood_threshold)
        self.max_parameter_age = int(max_parameter_age)
        self.max_trusted_version = int(max_trusted_version)
        self._reset_registers()

    def _reset_registers(self) -> None:
        self.active_bank = 0
        self.active_version = 0
        self.pipeline: list[_Request | None] = [None] * 5
        self.pending_output_payload = 0
        self.map_address = 0
        self.map_llr_twos = 0
        self.mode = MODE_NORMAL
        self.x_e_run = 0
        self.z_e_run = 0
        self.leakage_run = 0
        self.leakage_clean_run = 0
        self.health_good_run = 0
        self.reset_wait_run = 0
        self.pauli_frame_x = 0
        self.pauli_frame_z = 0
        self.phase_frame_x = 0
        self.phase_frame_z = 0
        self.trusted_active_version = 0
        self.health_status = HEALTH_HEALTHY
        self.fault_run = 0
        self.good_run = 0
        self.fault_cycle_count = 0
        self.leakage_cycle_count = 0
        self.fault_counts = [0] * 14
        self.last_fault_mask = 0

    def reset(self) -> None:
        """Apply synchronous core reset while preserving inferred RAM contents."""

        self._reset_registers()

    def _llr(self, bank: int, phase: int, code: int) -> tuple[int, int]:
        address = code >> 2
        fraction = code & 3
        table = self.tables[bank][phase]
        y0 = table[address]
        y1 = table[address + 1]
        interpolation = y0 + _round_shift3_ties_even((y1 - y0) * ((fraction << 1) | 1))
        interpolation = min(LLR_MAX, max(LLR_MIN, interpolation))
        return address, interpolation & LLR_MASK

    def _state_payload(self) -> int:
        payload = self.mode
        payload |= self.x_e_run << 3
        payload |= self.z_e_run << 6
        payload |= self.leakage_run << 9
        payload |= self.leakage_clean_run << 12
        payload |= self.health_good_run << 15
        payload |= self.reset_wait_run << 18
        payload |= self.pauli_frame_x << 21
        payload |= self.pauli_frame_z << 22
        payload |= self.phase_frame_x << 23
        payload |= self.phase_frame_z << 31
        payload |= self.trusted_active_version << 39
        payload |= self.health_status << 55
        payload |= self.fault_run << 58
        payload |= self.good_run << 66
        payload |= self.fault_cycle_count << 74
        payload |= self.leakage_cycle_count << 82
        for index, count in enumerate(self.fault_counts):
            payload |= count << (90 + index * 8)
        payload |= self.last_fault_mask << 202
        return payload

    def step(
        self,
        *,
        in_valid: int,
        in_word: int,
        safe_boundary: int = 1,
        commit_valid: int = 0,
        commit_bank: int = 0,
        commit_version: int = 0,
        cfg_we: int = 0,
        cfg_bank: int = 0,
        cfg_phase: int = 0,
        cfg_address: int = 0,
        cfg_data: int = 0,
        bank0_trusted: int = 1,
        bank1_trusted: int = 1,
    ) -> FastCoreCycleOutput:
        emitted_payload = self.pending_output_payload
        self.pending_output_payload = 0
        output_word = emitted_payload | (crc16_int_little_endian(emitted_payload, 13) << 102)

        old_active_bank = self.active_bank
        requested_trusted = bank1_trusted if commit_bank else bank0_trusted
        selected_commit = int(
            bool(commit_valid)
            and bool(safe_boundary)
            and bool(requested_trusted)
            and commit_bank != old_active_bank
            and self.active_version != 0xFFFF
            and commit_version == self.active_version + 1
        )
        cfg_allowed = int(
            bool(cfg_we)
            and cfg_address <= 256
            and cfg_bank != old_active_bank
            and not selected_commit
        )
        if selected_commit:
            self.active_bank = int(commit_bank)
            self.active_version = int(commit_version)
        if cfg_allowed:
            self.tables[int(cfg_bank)][int(cfg_phase)][int(cfg_address)] = _signed22(cfg_data)

        due = self.pipeline.pop(0)
        request: _Request | None = None
        if in_valid:
            payload = in_word & ((1 << 42) - 1)
            stored_crc = (in_word >> 42) & 0xFFFF
            x_code = (payload >> 10) & 3
            z_code = (payload >> 12) & 3
            phase = (payload >> 14) & 1
            address, llr_bits = self._llr(self.active_bank, phase, payload & 0x3FF)
            request = _Request(
                address=address,
                llr_bits=llr_bits,
                x_code=x_code,
                z_code=z_code,
                phase=phase,
                ood=(payload >> 15) & 0xFF,
                age=(payload >> 23) & 0xFFFF,
                reset_ack=(payload >> 39) & 1,
                observation_valid=((payload >> 40) & 1) and x_code != 3 and z_code != 3,
                deadline_ok=(payload >> 41) & 1,
                crc_ok=int(stored_crc == crc16_int_little_endian(payload, 6)),
                bank_trusted=int(bank1_trusted if self.active_bank else bank0_trusted),
                version=self.active_version,
            )
        self.pipeline.append(request)

        map_valid = int(due is not None)
        cycle_fault_mask = 0
        correction_enable = 0
        reset_request = 0
        if due is not None:
            self.map_address = due.address
            self.map_llr_twos = due.llr_bits
            leakage_observed = due.x_code == 2 or due.z_code == 2
            cycle_fault_mask = (
                (int(leakage_observed) << 13)
                | (int(bool(due.reset_ack) and self.mode != MODE_RESET_REQUEST) << 12)
                | (int(not due.deadline_ok) << 9)
                | (int(due.age > self.max_parameter_age) << 8)
                | (int(due.version < self.trusted_active_version) << 7)
                | (int(due.version > self.max_trusted_version) << 5)
                | (int(not due.bank_trusted) << 4)
                | (int(not due.bank_trusted) << 3)
                | (int(not due.crc_ok) << 2)
                | (int(due.ood > self.ood_threshold) << 1)
                | int(not due.observation_valid)
            )
            blocking = bool(cycle_fault_mask & 0x1FFF)
            map_accepted = not blocking
            flip = bool(due.llr_bits & (1 << 21))
            trusted_version = due.version if map_accepted else self.trusted_active_version
            next_x = 0 if blocking else (_sat3(self.x_e_run) if due.x_code == 1 else 0)
            next_z = 0 if blocking else (_sat3(self.z_e_run) if due.z_code == 1 else 0)
            next_leakage = 0 if blocking else (_sat3(self.leakage_run) if leakage_observed else 0)
            next_leakage_clean = 0 if blocking else (0 if leakage_observed else _sat3(self.leakage_clean_run))
            next_health_good = 0 if blocking else _sat3(self.health_good_run)
            next_reset_wait = _sat3(self.reset_wait_run) if self.mode == MODE_RESET_REQUEST and not due.reset_ack else 0

            if blocking:
                event_mode = MODE_FALLBACK
            elif due.reset_ack:
                event_mode = MODE_HOLD if self.mode == MODE_RESET_REQUEST else MODE_FALLBACK
            elif self.mode == MODE_RESET_REQUEST:
                event_mode = MODE_RESET_REQUEST
            elif leakage_observed:
                event_mode = MODE_RESET_REQUEST if next_leakage >= 2 else MODE_HOLD
            elif self.mode == MODE_HOLD and next_leakage_clean < 2:
                event_mode = MODE_HOLD
            elif self.mode == MODE_FALLBACK and next_health_good < 2:
                event_mode = MODE_FALLBACK
            elif next_x >= 2 and next_z >= 2:
                event_mode = MODE_Z_RECOVERY if due.phase else MODE_X_RECOVERY
            elif next_x >= 2:
                event_mode = MODE_X_RECOVERY
            elif next_z >= 2:
                event_mode = MODE_Z_RECOVERY
            else:
                event_mode = MODE_NORMAL

            inhibited = event_mode in (MODE_HOLD, MODE_RESET_REQUEST, MODE_FALLBACK)
            apply_map = map_accepted and not inhibited and flip
            delta_x = int(apply_map and not due.phase)
            delta_z = int(apply_map and due.phase)
            pauli_x = self.pauli_frame_x ^ delta_x
            pauli_z = self.pauli_frame_z ^ delta_z
            phase_x = (self.phase_frame_x + (0x80 if delta_x else 0)) & 0xFF
            phase_z = (self.phase_frame_z + (0x80 if delta_z else 0)) & 0xFF
            action_code = 0 if not map_accepted or not flip else (2 if due.phase else 1)
            if event_mode == MODE_RESET_REQUEST:
                event_health = HEALTH_RESET_REQUIRED
            elif blocking:
                event_health = HEALTH_FALLBACK
            elif event_mode == MODE_FALLBACK:
                event_health = HEALTH_RECOVERING
            elif cycle_fault_mask or event_mode == MODE_HOLD:
                event_health = HEALTH_DEGRADED
            else:
                event_health = HEALTH_HEALTHY

            self.mode = event_mode
            self.x_e_run = next_x
            self.z_e_run = next_z
            self.leakage_run = next_leakage
            self.leakage_clean_run = next_leakage_clean
            self.health_good_run = next_health_good
            self.reset_wait_run = next_reset_wait
            self.pauli_frame_x = pauli_x
            self.pauli_frame_z = pauli_z
            self.phase_frame_x = phase_x
            self.phase_frame_z = phase_z
            self.trusted_active_version = trusted_version
            self.health_status = event_health
            self.fault_run = _sat8(self.fault_run) if blocking else 0
            self.good_run = _sat8(self.good_run) if cycle_fault_mask == 0 else 0
            self.fault_cycle_count = _sat8(self.fault_cycle_count) if blocking else self.fault_cycle_count
            self.leakage_cycle_count = _sat8(self.leakage_cycle_count) if cycle_fault_mask & (1 << 13) else self.leakage_cycle_count
            self.last_fault_mask = cycle_fault_mask
            for index in range(14):
                if cycle_fault_mask & (1 << index):
                    self.fault_counts[index] = _sat8(self.fault_counts[index])

            correction_enable = int(not inhibited)
            reset_request = int(event_mode == MODE_RESET_REQUEST)
            pending = 1
            pending |= event_mode << 1
            pending |= correction_enable << 4
            pending |= reset_request << 5
            pending |= int(map_accepted and inhibited and flip) << 6
            pending |= action_code << 7
            pending |= delta_x << 9
            pending |= delta_z << 10
            pending |= pauli_x << 11
            pending |= pauli_z << 12
            pending |= phase_x << 13
            pending |= phase_z << 21
            pending |= next_x << 29
            pending |= next_z << 32
            pending |= next_leakage << 35
            pending |= next_leakage_clean << 38
            pending |= next_health_good << 41
            pending |= next_reset_wait << 44
            pending |= event_health << 47
            pending |= cycle_fault_mask << 50
            pending |= trusted_version << 64
            pending |= due.llr_bits << 80
            self.pending_output_payload = pending

        state_payload = self._state_payload()
        state_word = state_payload | (crc16_int_little_endian(state_payload, 27) << 216)
        return FastCoreCycleOutput(
            commit_ack=selected_commit,
            active_bank=self.active_bank,
            active_version=self.active_version,
            map_valid=map_valid,
            map_address=self.map_address,
            map_llr_twos=self.map_llr_twos,
            output_word=output_word,
            state_word=state_word,
            fault_mask=cycle_fault_mask,
            correction_enable=correction_enable,
            reset_request=reset_request,
        )
