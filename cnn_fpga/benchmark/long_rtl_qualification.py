"""T6.2.2 million-cycle board-independent fast-path qualification.

The runner generates ten deterministic trace families with an independent
integer golden model, streams every visible RTL output/state field into a
compact binary contract, and asks CXXRTL to compare every cycle.  FIFO and
communication disturbances are an explicitly abstract receiver-side model;
they are not a board transport implementation or measurement.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
from dataclasses import dataclass, field
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark.bit_accurate_hardware_reference import load_frozen_images
from cnn_fpga.benchmark.rtl_fast_path_equivalence import discover_tools
from cnn_fpga.runtime.bit_accurate_hardware_reference import BitAccurateHardwareReference
from cnn_fpga.runtime.fast_production_core_reference import (
    FastProductionCoreReference,
    LLR_MASK,
    LLR_MAX,
    LLR_MIN,
    MODE_RESET_REQUEST,
    corrupt_input_crc,
    crc16_int_little_endian,
    encode_fast_input_word,
    load_frozen_rtl_tables,
)


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
TOP = ROOT / "cnn_fpga/rtl/gkp_fast_path_qualification_top.sv"
DRIVER = ROOT / "cnn_fpga/rtl/long_qualification_cxxrtl_driver.cc"
T621_REPORT = ROOT / "docs/t6_2_1_production_rtl_audit.json"
DEFAULT_BUILD = ROOT / "build/t6_2_2_long_qualification"
DEFAULT_JSON = ROOT / "docs/t6_2_2_long_rtl_qualification.json"
DEFAULT_CSV = ROOT / "docs/t6_2_2_long_rtl_qualification_source_data.csv"
VERDICT = "PASS_BOARD_INDEPENDENT_LONG_RTL_QUALIFICATION_READY_FOR_ROUTE_A"
NON_QUALIFYING_VERDICT = "NON_QUALIFYING_SHORT_RUN"
FAMILY_CYCLES = 100_000
FAMILY_NAMES = (
    "nominal_random",
    "boundary_and_frame_wrap",
    "leakage_reset_hysteresis",
    "integrity_ood_stale",
    "deadline_pause_recovery",
    "version_trust_commit_race",
    "fifo_overflow_backpressure",
    "drop_duplicate_reorder",
    "compound_fault_recovery",
    "saturation_extreme_lut",
)

# Must match the packed C++ TraceRow exactly.  The expected output/state words
# are little-endian byte arrays, like CXXRTL's underlying wire chunks.
TRACE_STRUCT = struct.Struct("<6BH3BHI2BQ2BHBHI15s29s")
assert TRACE_STRUCT.size == 82

REACHABLE_FAULT_BITS = (0, 1, 2, 3, 4, 8, 9, 12, 13)
STRUCTURAL_ZERO_FAULT_BITS = (5, 6, 7, 10, 11)
MODE_NAMES = ("normal", "x_recovery", "z_recovery", "hold", "reset_request", "fallback")
HEALTH_NAMES = ("healthy", "degraded", "recovering", "fallback", "reset_required")
ACTION_NAMES = ("I", "X", "Z")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: Sequence[str | Path],
    *,
    env: Mapping[str, str],
    timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        env=dict(env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=True,
    )


class XorShift32:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF or 1

    def next(self) -> int:
        value = self.state
        value ^= (value << 13) & 0xFFFFFFFF
        value ^= value >> 17
        value ^= (value << 5) & 0xFFFFFFFF
        self.state = value & 0xFFFFFFFF
        return self.state


def _normal_word(cycle: int, rng: XorShift32) -> int:
    value = rng.next()
    return encode_fast_input_word(
        syndrome_code=value & 0x3FF,
        syndrome_x_code=1 if cycle % 19 in (0, 1, 2) else 0,
        syndrome_z_code=1 if cycle % 23 in (0, 1, 2) else 0,
        phase=(value >> 10) & 1,
        ood_score=(value >> 11) & 0x7F,
        parameter_age=(value >> 18) & 0xFFF,
    )


def _fault_marker(cycle: int) -> int:
    return encode_fast_input_word(
        syndrome_code=cycle & 0x3FF,
        syndrome_x_code=0,
        syndrome_z_code=0,
        phase=cycle & 1,
        ood_score=255,
        parameter_age=0xFFFF,
        observation_valid=0,
        deadline_ok=0,
    )


def _set_deadline(word: int, deadline_ok: int) -> int:
    payload = word & ((1 << 42) - 1)
    if deadline_ok:
        payload |= 1 << 41
    else:
        payload &= ~(1 << 41)
    return payload | (crc16_int_little_endian(payload, 6) << 42)


@dataclass(slots=True)
class _Packet:
    sequence: int
    created_cycle: int
    word: int


@dataclass(slots=True)
class AbstractTransportAdapter:
    """Finite FIFO plus receiver sequence checks; never a physical link model."""

    capacity: int = 8
    deadline_budget: int = 32
    queue: list[_Packet] = field(default_factory=list)
    marker_reasons: list[str] = field(default_factory=list)
    next_sequence: int = 0
    expected_sequence: int = 0
    stats: dict[str, int] = field(default_factory=lambda: {
        "source_packets": 0,
        "delivered_packets": 0,
        "pause_cycles": 0,
        "backpressure_cycles": 0,
        "overflow_events": 0,
        "accounted_overflow_events": 0,
        "drop_events": 0,
        "duplicate_events": 0,
        "reorder_events": 0,
        "sequence_faults": 0,
        "deadline_faults": 0,
        "explicit_fault_markers": 0,
        "max_fifo_depth": 0,
    })

    def _overflow(self) -> None:
        self.stats["overflow_events"] += 1
        self.stats["accounted_overflow_events"] += 1
        self.marker_reasons.append("overflow")

    def cycle(
        self,
        cycle: int,
        source_word: int | None,
        *,
        pause: bool = False,
        drop: bool = False,
        duplicate: bool = False,
        reorder: bool = False,
    ) -> int | None:
        if source_word is not None:
            packet = _Packet(self.next_sequence, cycle, source_word)
            self.next_sequence += 1
            self.stats["source_packets"] += 1
            if drop:
                self.stats["drop_events"] += 1
                self.marker_reasons.append("drop")
            elif len(self.queue) >= self.capacity:
                self._overflow()
            else:
                self.queue.append(packet)
                if duplicate:
                    self.stats["duplicate_events"] += 1
                    if len(self.queue) >= self.capacity:
                        self._overflow()
                    else:
                        self.queue.append(_Packet(packet.sequence, cycle, packet.word))
                if reorder and len(self.queue) >= 2:
                    self.stats["reorder_events"] += 1
                    self.queue[-1], self.queue[-2] = self.queue[-2], self.queue[-1]
            self.stats["max_fifo_depth"] = max(self.stats["max_fifo_depth"], len(self.queue))

        if pause:
            self.stats["pause_cycles"] += 1
            self.stats["backpressure_cycles"] += int(bool(self.queue or self.marker_reasons))
            return None

        # Preserve forward progress of accepted packets.  Explicit error tokens
        # are drained after the data queue and during the source-off tail.
        if self.queue:
            packet = self.queue.pop(0)
            if packet.sequence != self.expected_sequence:
                self.stats["sequence_faults"] += 1
                self.expected_sequence = max(self.expected_sequence, packet.sequence + 1)
                self.stats["explicit_fault_markers"] += 1
                return _fault_marker(cycle)
            self.expected_sequence += 1
            self.stats["delivered_packets"] += 1
            age = cycle - packet.created_cycle
            if age > self.deadline_budget:
                self.stats["deadline_faults"] += 1
                return _set_deadline(packet.word, 0)
            return packet.word
        if self.marker_reasons:
            self.marker_reasons.pop(0)
            self.stats["explicit_fault_markers"] += 1
            return _fault_marker(cycle)
        return None

    def summary(self) -> dict[str, int]:
        result = dict(self.stats)
        result["pending_fifo"] = len(self.queue)
        result["pending_markers"] = len(self.marker_reasons)
        result["silent_overflow"] = (
            result["overflow_events"] - result["accounted_overflow_events"]
        )
        return result


def _blank_signals() -> dict[str, int]:
    return {
        "in_valid": 0,
        "in_word": 0,
        "safe_boundary": 1,
        "commit_valid": 0,
        "commit_bank": 0,
        "commit_version": 0,
        "cfg_we": 0,
        "cfg_bank": 0,
        "cfg_phase": 0,
        "cfg_address": 0,
        "cfg_data": 0,
        "bank0_trusted": 1,
        "bank1_trusted": 1,
    }


def _output_fields(word: int) -> dict[str, int]:
    payload = word & ((1 << 102) - 1)
    return {
        "valid": payload & 1,
        "mode": (payload >> 1) & 7,
        "correction_enable": (payload >> 4) & 1,
        "reset_request": (payload >> 5) & 1,
        "held_action": (payload >> 6) & 1,
        "action": (payload >> 7) & 3,
        "health": (payload >> 47) & 7,
        "fault_mask": (payload >> 50) & 0x3FFF,
        "active_version": (payload >> 64) & 0xFFFF,
        "llr_twos": (payload >> 80) & LLR_MASK,
        "crc_ok": int(((word >> 102) & 0xFFFF) == crc16_int_little_endian(payload, 13)),
    }


def _state_fields(word: int) -> dict[str, Any]:
    payload = word & ((1 << 216) - 1)
    counts = [((payload >> (90 + index * 8)) & 0xFF) for index in range(14)]
    return {
        "mode": payload & 7,
        "fault_run": (payload >> 58) & 0xFF,
        "good_run": (payload >> 66) & 0xFF,
        "fault_cycle_count": (payload >> 74) & 0xFF,
        "leakage_cycle_count": (payload >> 82) & 0xFF,
        "fault_counts": counts,
        "last_fault_mask": (payload >> 202) & 0x3FFF,
        "crc_ok": int(((word >> 216) & 0xFFFF) == crc16_int_little_endian(payload, 27)),
    }


def _new_family_stats(family_id: int, name: str) -> dict[str, Any]:
    return {
        "family_id": family_id,
        "family": name,
        "cycles": 0,
        "input_valid": 0,
        "output_valid": 0,
        "reset_events": 0,
        "commit_attempts": 0,
        "commit_acks": 0,
        "commit_rejections": 0,
        "rollback_commit_rejections": 0,
        "untrusted_commit_rejections": 0,
        "commit_cfg_races": 0,
        "output_crc_errors": 0,
        "state_crc_errors": 0,
        "undefined_actions": 0,
        "llr_min_hits": 0,
        "llr_max_hits": 0,
        "counter_saturation_hits": 0,
        "fault_to_healthy_recoveries": 0,
        "last_fault_output_cycle": -1,
        "last_healthy_output_cycle": -1,
        "final_valid_mode": -1,
        "final_valid_health": -1,
        "final_valid_fault_mask": -1,
        "_awaiting_recovery": False,
        "modes": {name: 0 for name in MODE_NAMES},
        "health": {name: 0 for name in HEALTH_NAMES},
        "actions": {name: 0 for name in ACTION_NAMES},
        "fault_bits": {str(index): 0 for index in range(14)},
        "maxima": {
            "fault_run": 0,
            "good_run": 0,
            "fault_cycle_count": 0,
            "leakage_cycle_count": 0,
            "per_fault_count": 0,
        },
        "transport": {},
    }


def _stimulus(
    family_id: int,
    cycle: int,
    cycles: int,
    model: FastProductionCoreReference,
    rng: XorShift32,
    transport: AbstractTransportAdapter | None,
) -> tuple[dict[str, int], dict[str, int]]:
    signals = _blank_signals()
    tags: dict[str, int] = {}
    active_source_end = max(0, cycles - max(2_000, cycles // 5))

    if family_id == 0:
        if cycle % 97:
            signals["in_valid"] = 1
            signals["in_word"] = _normal_word(cycle, rng)

    elif family_id == 1:
        codes = (0, 1, 2, 3, 4, 7, 511, 512, 1019, 1020, 1021, 1022, 1023)
        code = codes[cycle % len(codes)]
        signals["in_valid"] = int(cycle % 101 != 0)
        signals["in_word"] = encode_fast_input_word(
            syndrome_code=code,
            syndrome_x_code=1 if cycle % 8 < 3 else 0,
            syndrome_z_code=1 if cycle % 10 < 3 else 0,
            phase=cycle & 1,
            ood_score=0,
            parameter_age=cycle & 0x1FFF,
        )

    elif family_id == 2:
        local = cycle % 64
        leakage = local in (0, 1, 2, 3)
        signals["in_valid"] = 1
        signals["in_word"] = encode_fast_input_word(
            syndrome_code=(cycle * 37) & 0x3FF,
            syndrome_x_code=2 if leakage else (1 if local in (20, 21) else 0),
            syndrome_z_code=2 if local in (2, 3) else 0,
            phase=cycle & 1,
            ood_score=0,
            parameter_age=cycle & 0x7FF,
            reset_ack=int(local in (12, 13, 14)),
        )

    elif family_id == 3:
        local = cycle % 12
        kwargs: dict[str, int] = {
            "syndrome_code": (cycle * 29) & 0x3FF,
            "syndrome_x_code": 0,
            "syndrome_z_code": 0,
            "phase": cycle & 1,
            "ood_score": 0,
            "parameter_age": cycle & 0xFFF,
        }
        if local == 0:
            kwargs["observation_valid"] = 0
        elif local == 1:
            kwargs["syndrome_x_code"] = 3
        elif local == 2:
            kwargs["ood_score"] = 255
        elif local == 3:
            kwargs["parameter_age"] = 0xFFFF
        elif local == 4:
            kwargs["deadline_ok"] = 0
        elif local == 6:
            signals["bank0_trusted"] = 0
        elif local == 7:
            kwargs["reset_ack"] = 1
        elif local == 8:
            kwargs["syndrome_z_code"] = 3
        word = encode_fast_input_word(**kwargs)
        if local == 5:
            word = corrupt_input_crc(word)
        signals["in_valid"] = 1
        signals["in_word"] = word

    elif family_id == 4:
        local = cycle % 128
        if local >= 24:
            signals["in_valid"] = 1
            signals["in_word"] = encode_fast_input_word(
                syndrome_code=(cycle * 61) & 0x3FF,
                syndrome_x_code=int(local in (24, 25, 26)),
                syndrome_z_code=int(local in (32, 33, 34)),
                phase=cycle & 1,
                ood_score=0,
                parameter_age=0,
                deadline_ok=int(local >= 32),
            )
        if cycle == cycles // 2:
            tags["reset_before"] = 1

    elif family_id == 5:
        local = cycle % 10_000
        if local < 12:
            target = 1 - model.active_bank
            expected = model.active_version + 1
            if local == 6:
                signals.update(commit_valid=1, commit_bank=target, commit_version=expected, safe_boundary=0)
                tags["safe_boundary_reject"] = 1
            elif local == 7:
                signals.update(commit_valid=1, commit_bank=target, commit_version=expected)
                signals["bank1_trusted" if target else "bank0_trusted"] = 0
                tags["untrusted_commit"] = 1
            elif local == 8:
                signals.update(commit_valid=1, commit_bank=target, commit_version=min(0xFFFF, expected + 1))
                tags["version_reject"] = 1
            elif local == 9:
                signals.update(commit_valid=1, commit_bank=target, commit_version=max(0, model.active_version - 1))
                tags["rollback_commit"] = 1
            elif local == 10:
                signals.update(
                    commit_valid=1,
                    commit_bank=target,
                    commit_version=expected,
                    cfg_we=1,
                    cfg_bank=target,
                    cfg_phase=0,
                    cfg_address=0,
                    cfg_data=model.tables[target][0][0] & LLR_MASK,
                )
                tags["commit_cfg_race"] = 1
            elif local == 11:
                signals.update(
                    commit_valid=1,
                    commit_bank=model.active_bank,
                    commit_version=min(0xFFFF, model.active_version + 1),
                )
                tags["same_bank_reject"] = 1
        else:
            signals["in_valid"] = 1
            signals["in_word"] = _normal_word(cycle, rng)

    elif family_id in (6, 7, 8):
        assert transport is not None
        source_word: int | None = None
        if cycle < active_source_end:
            if family_id == 8 and cycle % 257 == 0:
                source_word = corrupt_input_crc(encode_fast_input_word(
                    syndrome_code=cycle & 0x3FF,
                    syndrome_x_code=2 if cycle % 514 == 0 else 0,
                    syndrome_z_code=0,
                    phase=cycle & 1,
                    ood_score=255,
                    parameter_age=0xFFFF,
                    deadline_ok=0,
                ))
            else:
                source_word = _normal_word(cycle, rng)
        pause = False
        drop = duplicate = reorder = False
        if family_id == 6:
            pause = cycle < active_source_end and cycle % 5_000 < 300
        elif family_id == 7:
            pause = cycle < active_source_end and cycle % 7_000 < 17
            drop = bool(source_word is not None and cycle > 0 and cycle % 4_093 == 0)
            duplicate = bool(source_word is not None and cycle > 0 and cycle % 5_003 == 0)
            # Offset 18 is the first accepted enqueue after the registered
            # 17-cycle pause burst, when at least two packets are resident.
            reorder = bool(source_word is not None and cycle > 0 and cycle % 7_000 == 18)
        else:
            pause = cycle < active_source_end and cycle % 3_001 < 41
            drop = bool(source_word is not None and cycle > 0 and cycle % 2_003 == 0)
            duplicate = bool(source_word is not None and cycle > 0 and cycle % 3_007 == 0)
            reorder = bool(source_word is not None and cycle > 0 and cycle % 4_009 == 0)
            if cycle % 2_500 < 11:
                signals["bank0_trusted"] = 0
        delivered = transport.cycle(
            cycle,
            source_word,
            pause=pause,
            drop=drop,
            duplicate=duplicate,
            reorder=reorder,
        )
        if delivered is not None:
            signals["in_valid"] = 1
            signals["in_word"] = delivered
        elif (
            cycle >= active_source_end
            and not transport.queue
            and not transport.marker_reasons
        ):
            # Once every accepted/error token is drained, continue explicit
            # healthy observations so recovery is observed rather than merely
            # inferred from an idle core left in fallback.
            signals["in_valid"] = 1
            signals["in_word"] = encode_fast_input_word(
                syndrome_code=cycle & 0x3FF,
                syndrome_x_code=0,
                syndrome_z_code=0,
                phase=cycle & 1,
                ood_score=0,
                parameter_age=0,
            )

    elif family_id == 9:
        if cycle < 514:
            phase = int(cycle >= 257)
            address = cycle if phase == 0 else cycle - 257
            signals.update(
                cfg_we=1,
                cfg_bank=1,
                cfg_phase=phase,
                cfg_address=address,
                cfg_data=(LLR_MIN if address < 128 else LLR_MAX) & LLR_MASK,
            )
        elif cycle == 520:
            signals.update(commit_valid=1, commit_bank=1, commit_version=1)
            tags["extreme_bank_commit"] = 1
        elif cycle > 526:
            local = (cycle - 527) % 1_200
            code = 0 if local < 600 else 1023
            kwargs = {
                "syndrome_code": code,
                "syndrome_x_code": 0,
                "syndrome_z_code": 0,
                "phase": cycle & 1,
                "ood_score": 0,
                "parameter_age": 0,
            }
            if local < 300:
                kwargs.update(
                    syndrome_x_code=2,
                    syndrome_z_code=2,
                    ood_score=255,
                    parameter_age=0xFFFF,
                    observation_valid=0,
                    deadline_ok=0,
                )
            elif local >= 900:
                kwargs["syndrome_x_code"] = 2
                kwargs["reset_ack"] = int(model.mode == MODE_RESET_REQUEST and local % 16 == 0)
            word = encode_fast_input_word(**kwargs)
            if local < 300:
                word = corrupt_input_crc(word)
            signals["in_valid"] = 1
            signals["in_word"] = word

    return signals, tags


def _update_stats(
    stats: dict[str, Any],
    signals: Mapping[str, int],
    tags: Mapping[str, int],
    output: Any,
    reset_before: int,
    cycle: int,
) -> None:
    stats["cycles"] += 1
    stats["input_valid"] += int(signals["in_valid"])
    stats["reset_events"] += reset_before
    if signals["commit_valid"]:
        stats["commit_attempts"] += 1
        stats["commit_acks"] += output.commit_ack
        stats["commit_rejections"] += 1 - output.commit_ack
        stats["rollback_commit_rejections"] += int(bool(tags.get("rollback_commit")) and not output.commit_ack)
        stats["untrusted_commit_rejections"] += int(bool(tags.get("untrusted_commit")) and not output.commit_ack)
        stats["commit_cfg_races"] += int(bool(tags.get("commit_cfg_race")))

    decoded = _output_fields(output.output_word)
    state = _state_fields(output.state_word)
    stats["output_crc_errors"] += 1 - decoded["crc_ok"]
    stats["state_crc_errors"] += 1 - state["crc_ok"]
    if decoded["valid"]:
        stats["output_valid"] += 1
        if decoded["mode"] < len(MODE_NAMES):
            stats["modes"][MODE_NAMES[decoded["mode"]]] += 1
        if decoded["health"] < len(HEALTH_NAMES):
            stats["health"][HEALTH_NAMES[decoded["health"]]] += 1
        if decoded["action"] < len(ACTION_NAMES):
            stats["actions"][ACTION_NAMES[decoded["action"]]] += 1
        stats["undefined_actions"] += int(
            decoded["mode"] >= len(MODE_NAMES)
            or decoded["health"] >= len(HEALTH_NAMES)
            or decoded["action"] >= len(ACTION_NAMES)
        )
        for index in range(14):
            stats["fault_bits"][str(index)] += (decoded["fault_mask"] >> index) & 1
        stats["final_valid_mode"] = decoded["mode"]
        stats["final_valid_health"] = decoded["health"]
        stats["final_valid_fault_mask"] = decoded["fault_mask"]
        if decoded["fault_mask"]:
            stats["last_fault_output_cycle"] = cycle
            stats["_awaiting_recovery"] = True
        elif decoded["mode"] == 0 and decoded["health"] == 0:
            stats["last_healthy_output_cycle"] = cycle
            if stats["_awaiting_recovery"]:
                stats["fault_to_healthy_recoveries"] += 1
                stats["_awaiting_recovery"] = False
    if output.map_valid:
        stats["llr_min_hits"] += int(output.map_llr_twos == (LLR_MIN & LLR_MASK))
        stats["llr_max_hits"] += int(output.map_llr_twos == (LLR_MAX & LLR_MASK))

    maxima = stats["maxima"]
    for key in ("fault_run", "good_run", "fault_cycle_count", "leakage_cycle_count"):
        maxima[key] = max(maxima[key], state[key])
    maxima["per_fault_count"] = max(maxima["per_fault_count"], *state["fault_counts"])
    stats["counter_saturation_hits"] += int(
        255 in (
            state["fault_run"],
            state["good_run"],
            state["fault_cycle_count"],
            state["leakage_cycle_count"],
            *state["fault_counts"],
        )
    )


def generate_trace(build_dir: Path, cycles_per_family: int) -> tuple[Path, list[dict[str, Any]], float]:
    build_dir.mkdir(parents=True, exist_ok=True)
    trace_path = build_dir / "qualification_trace.bin"
    tables = load_frozen_rtl_tables(ROOT)
    family_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with trace_path.open("wb", buffering=4 * 1024 * 1024) as handle:
        for family_id, name in enumerate(FAMILY_NAMES):
            model = FastProductionCoreReference(tables)
            rng = XorShift32(0xA5C31E27 ^ (family_id * 0x9E3779B9))
            transport = AbstractTransportAdapter() if family_id in (6, 7, 8) else None
            stats = _new_family_stats(family_id, name)
            for cycle in range(cycles_per_family):
                signals, tags = _stimulus(
                    family_id, cycle, cycles_per_family, model, rng, transport
                )
                reset_before = int(cycle == 0 or bool(tags.get("reset_before")))
                if reset_before and cycle != 0:
                    model.reset()
                output = model.step(**signals)
                _update_stats(stats, signals, tags, output, reset_before, cycle)
                handle.write(TRACE_STRUCT.pack(
                    family_id,
                    reset_before,
                    signals["in_valid"],
                    signals["safe_boundary"],
                    signals["commit_valid"],
                    signals["commit_bank"],
                    signals["commit_version"],
                    signals["cfg_we"],
                    signals["cfg_bank"],
                    signals["cfg_phase"],
                    signals["cfg_address"],
                    signals["cfg_data"],
                    signals["bank0_trusted"],
                    signals["bank1_trusted"],
                    signals["in_word"],
                    output.commit_ack,
                    output.active_bank,
                    output.active_version,
                    output.map_valid,
                    output.map_address,
                    output.map_llr_twos,
                    output.output_word.to_bytes(15, "little"),
                    output.state_word.to_bytes(29, "little"),
                ))
            if transport is not None:
                stats["transport"] = transport.summary()
            stats.pop("_awaiting_recovery")
            family_reports.append(stats)
    return trace_path, family_reports, time.perf_counter() - started


def legacy_crosscheck(rows: int = 10_000) -> dict[str, Any]:
    tables = load_frozen_rtl_tables(ROOT)
    fast = FastProductionCoreReference(tables)
    legacy = BitAccurateHardwareReference(
        load_frozen_images(), max_parameter_age_cycles=8192
    )
    rng = XorShift32(0xD17A5EED)
    mismatches = 0
    first_mismatch: dict[str, Any] | None = None
    for cycle in range(rows):
        word = _normal_word(cycle, rng)
        fast_row = fast.step(in_valid=1, in_word=word)
        legacy_row = legacy.step_word(word)
        fields_match = (
            fast_row.output_word == int(legacy_row.output_word_hex, 16)
            and fast_row.state_word == int(legacy_row.state_word_hex, 16)
            and fast_row.active_version == legacy_row.active_version
            and fast_row.map_valid == int(legacy_row.map_valid)
            and (
                not fast_row.map_valid
                or (
                    fast_row.map_address == legacy_row.map_address
                    and fast_row.map_llr_twos == (int(legacy_row.map_llr_code) & LLR_MASK)
                )
            )
        )
        if not fields_match:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = {"cycle": cycle, "fast_output": hex(fast_row.output_word), "legacy_output": legacy_row.output_word_hex}
    return {"rows": rows, "mismatches": mismatches, "first_mismatch": first_mismatch}


def build_cxxrtl(build_dir: Path) -> dict[str, Any]:
    tools = discover_tools()
    temp_dir = build_dir / "temp"
    cache_dir = build_dir / "yowasp_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = build_dir / "gkp_fast_path_qualification_model.cc"
    executable = build_dir / "long_qualification_trace.exe"
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(cache_dir)
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["PATH"] = str(tools["gpp"].parent) + os.pathsep + env.get("PATH", "")
    yosys_script = (
        f"read_verilog -sv {_relative(CORE)} {_relative(TOP)}; "
        "hierarchy -check -top gkp_fast_path_qualification_top; proc; check; stat; "
        f"write_cxxrtl -O0 -g0 {_relative(model)}"
    )
    started = time.perf_counter()
    yosys_run = _run((tools["yosys"], "-Q", "-p", yosys_script), env=env)
    yosys_seconds = time.perf_counter() - started
    yosys_log = build_dir / "yosys_cxxrtl.log"
    yosys_log.write_text(yosys_run.stdout + yosys_run.stderr, encoding="utf-8")
    started = time.perf_counter()
    compile_run = _run((
        tools["gpp"], "-std=c++17", "-O3", "-DNDEBUG",
        "-I", tools["include"], "-I", build_dir, DRIVER, "-o", executable,
    ), env=env, timeout=1800)
    compile_seconds = time.perf_counter() - started
    compile_log = build_dir / "gpp_compile.log"
    compile_log.write_text(compile_run.stdout + compile_run.stderr, encoding="utf-8")
    memory_match = re.search(r"\n\s*(\d+) memories\n\s*(\d+) memory bits", yosys_run.stdout)
    return {
        "executable": executable,
        "environment": env,
        "yosys_version": _run((tools["yosys"], "-V"), env=env).stdout.strip(),
        "gpp_version": _run((tools["gpp"], "--version"), env=env).stdout.splitlines()[0],
        "yosys_seconds": yosys_seconds,
        "compile_seconds": compile_seconds,
        "model_bytes": model.stat().st_size,
        "model_sha256": _sha256(model),
        "executable_bytes": executable.stat().st_size,
        "executable_sha256": _sha256(executable),
        "structural_check_zero_problems": "Found and reported 0 problems" in yosys_run.stdout,
        "memory_count": None if memory_match is None else int(memory_match.group(1)),
        "memory_bits": None if memory_match is None else int(memory_match.group(2)),
        "yosys_log": _relative(yosys_log),
        "compile_log": _relative(compile_log),
    }


def _parse_cxxrtl_stdout(stdout: str) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(io.StringIO(stdout)))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append({
            "family_id": int(row["family_id"]),
            "family": FAMILY_NAMES[int(row["family_id"])],
            "rows": int(row["rows"]),
            "mismatches": int(row["mismatches"]),
            "output_valid": int(row["output_valid"]),
            "blocking_fault_outputs": int(row["blocking_fault_outputs"]),
            "undefined_actions": int(row["undefined_actions"]),
            "shadow_mutations": int(row["shadow_mutations"]),
            "shadow_mutations_detected": int(row["shadow_mutations_detected"]),
            "actual_digest": row["actual_digest"],
            "expected_digest": row["expected_digest"],
        })
    return parsed


def run_cxxrtl(executable: Path, env: Mapping[str, str], trace_path: Path) -> tuple[list[dict[str, Any]], float, str]:
    """Qualify independent reset-delimited families in parallel processes."""

    started = time.perf_counter()
    workers = min(len(FAMILY_NAMES), max(1, os.cpu_count() or 1))
    completed_by_family: dict[int, subprocess.CompletedProcess[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run,
                (executable, trace_path, family_id),
                env=env,
                timeout=7200,
            ): family_id
            for family_id in range(len(FAMILY_NAMES))
        }
        for future in as_completed(futures):
            family_id = futures[future]
            completed_by_family[family_id] = future.result()
    seconds = time.perf_counter() - started
    parsed: list[dict[str, Any]] = []
    stderr_parts: list[str] = []
    for family_id in range(len(FAMILY_NAMES)):
        completed = completed_by_family[family_id]
        family_rows = _parse_cxxrtl_stdout(completed.stdout)
        if len(family_rows) != 1 or family_rows[0]["family_id"] != family_id:
            raise RuntimeError(f"CXXRTL family {family_id} returned an invalid summary")
        parsed.extend(family_rows)
        if completed.stderr:
            stderr_parts.append(f"[family {family_id}]\n{completed.stderr}")
    return parsed, seconds, "".join(stderr_parts)


def _aggregate_python(families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "cycles": sum(int(row["cycles"]) for row in families),
        "output_valid": sum(int(row["output_valid"]) for row in families),
        "undefined_actions": sum(int(row["undefined_actions"]) for row in families),
        "output_crc_errors": sum(int(row["output_crc_errors"]) for row in families),
        "state_crc_errors": sum(int(row["state_crc_errors"]) for row in families),
        "reset_events": sum(int(row["reset_events"]) for row in families),
        "commit_attempts": sum(int(row["commit_attempts"]) for row in families),
        "commit_acks": sum(int(row["commit_acks"]) for row in families),
        "commit_rejections": sum(int(row["commit_rejections"]) for row in families),
        "rollback_commit_rejections": sum(int(row["rollback_commit_rejections"]) for row in families),
        "untrusted_commit_rejections": sum(int(row["untrusted_commit_rejections"]) for row in families),
        "llr_min_hits": sum(int(row["llr_min_hits"]) for row in families),
        "llr_max_hits": sum(int(row["llr_max_hits"]) for row in families),
        "counter_saturation_hits": sum(int(row["counter_saturation_hits"]) for row in families),
        "modes": {name: sum(int(row["modes"][name]) for row in families) for name in MODE_NAMES},
        "health": {name: sum(int(row["health"][name]) for row in families) for name in HEALTH_NAMES},
        "actions": {name: sum(int(row["actions"][name]) for row in families) for name in ACTION_NAMES},
        "fault_bits": {str(index): sum(int(row["fault_bits"][str(index)]) for row in families) for index in range(14)},
        "silent_overflow": sum(int(row.get("transport", {}).get("silent_overflow", 0)) for row in families),
        "transport_pending": sum(
            int(row.get("transport", {}).get("pending_fifo", 0))
            + int(row.get("transport", {}).get("pending_markers", 0))
            for row in families
        ),
    }


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    python = report["aggregate_python"]
    cxxrtl = report["cxxrtl_families"]
    cycles_per_family = int(report["cycles_per_family"])
    qualifying = cycles_per_family >= FAMILY_CYCLES
    by_id = {int(row["family_id"]): row for row in report["python_families"]}
    transport6 = by_id[6].get("transport", {})
    transport7 = by_id[7].get("transport", {})
    transport8 = by_id[8].get("transport", {})
    forbidden = ("board-measured", "physical transport", "measured power", "bitstream-qualified")
    return {
        "qualifying_scale": qualifying and len(report["python_families"]) == 10 and min(int(row["cycles"]) for row in report["python_families"]) >= FAMILY_CYCLES and int(python["cycles"]) >= 1_000_000,
        "legacy_independent_crosscheck": int(report["legacy_crosscheck"]["mismatches"]) == 0 and int(report["legacy_crosscheck"]["rows"]) >= 10_000,
        "yosys_structural_check": bool(report["toolchain"]["structural_check_zero_problems"]),
        "all_cycle_bit_exact": len(cxxrtl) == 10 and sum(int(row["mismatches"]) for row in cxxrtl) == 0 and all(row["actual_digest"] == row["expected_digest"] for row in cxxrtl),
        "cxxrtl_full_scale": len(cxxrtl) == 10 and min(int(row["rows"]) for row in cxxrtl) >= FAMILY_CYCLES and sum(int(row["rows"]) for row in cxxrtl) >= 1_000_000,
        "no_undefined_or_crc_error": int(python["undefined_actions"]) == 0 and int(python["output_crc_errors"]) == 0 and int(python["state_crc_errors"]) == 0 and sum(int(row["undefined_actions"]) for row in cxxrtl) == 0,
        "all_modes_health_actions": all(int(value) > 0 for value in python["modes"].values()) and all(int(value) > 0 for value in python["health"].values()) and all(int(value) > 0 for value in python["actions"].values()),
        "reachable_fault_coverage": all(int(python["fault_bits"][str(index)]) > 0 for index in REACHABLE_FAULT_BITS) and all(int(python["fault_bits"][str(index)]) == 0 for index in STRUCTURAL_ZERO_FAULT_BITS),
        "version_commit_negative_paths": int(python["rollback_commit_rejections"]) > 0 and int(python["untrusted_commit_rejections"]) > 0 and int(python["commit_acks"]) > 0 and int(python["commit_rejections"]) > 0,
        "family_target_coverage": (
            all(int(by_id[0]["fault_bits"][str(index)]) == 0 for index in range(14))
            and int(by_id[2]["modes"]["reset_request"]) > 0
            and int(by_id[2]["modes"]["hold"]) > 0
            and int(by_id[2]["fault_bits"]["13"]) > 0
            and all(int(by_id[3]["fault_bits"][str(index)]) > 0 for index in (0, 1, 2, 3, 4, 8, 9, 12))
            and int(by_id[4]["fault_bits"]["9"]) > 0
            and all(int(by_id[index]["fault_to_healthy_recoveries"]) > 0 for index in (2, 3, 4, 6, 7, 8, 9))
        ),
        "saturation_and_extreme_llr": (
            int(by_id[9]["llr_min_hits"]) > 0
            and int(by_id[9]["llr_max_hits"]) > 0
            and all(int(value) == 255 for value in by_id[9]["maxima"].values())
        ),
        "transport_fail_closed": (
            int(python["silent_overflow"]) == 0
            and int(python["transport_pending"]) == 0
            and all(int(row.get("transport", {}).get("max_fifo_depth", 0)) <= 8 for row in report["python_families"])
            and all(int(transport6.get(name, 0)) > 0 for name in ("pause_cycles", "backpressure_cycles", "overflow_events", "deadline_faults", "explicit_fault_markers"))
            and all(int(transport7.get(name, 0)) > 0 for name in ("pause_cycles", "drop_events", "duplicate_events", "reorder_events", "sequence_faults", "explicit_fault_markers"))
            and all(int(transport8.get(name, 0)) > 0 for name in ("pause_cycles", "overflow_events", "drop_events", "duplicate_events", "reorder_events", "sequence_faults", "deadline_faults", "explicit_fault_markers"))
            and all(int(by_id[index]["final_valid_mode"]) == 0 and int(by_id[index]["final_valid_health"]) == 0 and int(by_id[index]["final_valid_fault_mask"]) == 0 for index in (6, 7, 8))
        ),
        "cxxrtl_shadow_mutations": sum(int(row["shadow_mutations"]) for row in cxxrtl) == 8 and sum(int(row["shadow_mutations_detected"]) for row in cxxrtl) == 8,
        "evidence_scope_lint": not any(token in str(report["evidence_scope"]).lower() for token in forbidden),
        "t621_anchor_present": bool(report["t6_2_1_anchor"]["present"]) and bool(report["t6_2_1_anchor"]["pass"]),
    }


def semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    mutations: list[dict[str, Any]] = []

    def attempt(name: str, mutator: Any) -> None:
        candidate = copy.deepcopy(report)
        mutator(candidate)
        rejected = not all(evaluate_gates(candidate).values())
        mutations.append({"mutation": name, "rejected": rejected})

    attempt("hide_one_cxxrtl_mismatch", lambda value: value["cxxrtl_families"][0].update(mismatches=1))
    attempt("reduce_one_family_below_1e5", lambda value: value["python_families"][0].update(cycles=99_999))
    attempt("insert_undefined_action", lambda value: value["aggregate_python"].update(undefined_actions=1))
    attempt("insert_silent_overflow", lambda value: value["aggregate_python"].update(silent_overflow=1))
    attempt("erase_reachable_fault_branch", lambda value: value["aggregate_python"]["fault_bits"].update({"9": 0}))
    attempt("erase_rollback_rejection", lambda value: value["aggregate_python"].update(rollback_commit_rejections=0))
    attempt("promote_to_board_claim", lambda value: value.update(evidence_scope="board-measured physical transport"))
    attempt("erase_reorder_injection", lambda value: value["python_families"][7]["transport"].update(reorder_events=0))
    return {
        "mutations": mutations,
        "count": len(mutations),
        "detected": sum(int(row["rejected"]) for row in mutations),
    }


def write_source_data(path: Path, report: Mapping[str, Any]) -> None:
    cxxrtl_by_id = {int(row["family_id"]): row for row in report["cxxrtl_families"]}
    fieldnames = (
        "family_id", "family", "cycles", "input_valid", "output_valid", "cxxrtl_mismatches",
        "undefined_actions", "output_crc_errors", "state_crc_errors", "commit_attempts", "commit_acks",
        "commit_rejections", "rollback_commit_rejections", "llr_min_hits", "llr_max_hits",
        "counter_saturation_hits", "fault_to_healthy_recoveries", "last_fault_output_cycle",
        "last_healthy_output_cycle", "fault_mask_coverage_hex", "max_fifo_depth", "pause_cycles",
        "backpressure_cycles", "overflow_events", "drop_events", "duplicate_events", "reorder_events",
        "sequence_faults", "deadline_faults", "explicit_fault_markers", "silent_overflow",
        "pending_transport", "actual_digest", "expected_digest",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family in report["python_families"]:
            transport = family.get("transport", {})
            cxx = cxxrtl_by_id[int(family["family_id"])]
            coverage = sum((int(family["fault_bits"][str(index)]) > 0) << index for index in range(14))
            writer.writerow({
                "family_id": family["family_id"],
                "family": family["family"],
                "cycles": family["cycles"],
                "input_valid": family["input_valid"],
                "output_valid": family["output_valid"],
                "cxxrtl_mismatches": cxx["mismatches"],
                "undefined_actions": family["undefined_actions"],
                "output_crc_errors": family["output_crc_errors"],
                "state_crc_errors": family["state_crc_errors"],
                "commit_attempts": family["commit_attempts"],
                "commit_acks": family["commit_acks"],
                "commit_rejections": family["commit_rejections"],
                "rollback_commit_rejections": family["rollback_commit_rejections"],
                "llr_min_hits": family["llr_min_hits"],
                "llr_max_hits": family["llr_max_hits"],
                "counter_saturation_hits": family["counter_saturation_hits"],
                "fault_to_healthy_recoveries": family["fault_to_healthy_recoveries"],
                "last_fault_output_cycle": family["last_fault_output_cycle"],
                "last_healthy_output_cycle": family["last_healthy_output_cycle"],
                "fault_mask_coverage_hex": f"0x{coverage:04x}",
                "max_fifo_depth": transport.get("max_fifo_depth", 0),
                "pause_cycles": transport.get("pause_cycles", 0),
                "backpressure_cycles": transport.get("backpressure_cycles", 0),
                "overflow_events": transport.get("overflow_events", 0),
                "drop_events": transport.get("drop_events", 0),
                "duplicate_events": transport.get("duplicate_events", 0),
                "reorder_events": transport.get("reorder_events", 0),
                "sequence_faults": transport.get("sequence_faults", 0),
                "deadline_faults": transport.get("deadline_faults", 0),
                "explicit_fault_markers": transport.get("explicit_fault_markers", 0),
                "silent_overflow": transport.get("silent_overflow", 0),
                "pending_transport": transport.get("pending_fifo", 0) + transport.get("pending_markers", 0),
                "actual_digest": cxx["actual_digest"],
                "expected_digest": cxx["expected_digest"],
            })


def run_qualification(
    *,
    build_dir: Path = DEFAULT_BUILD,
    artifact_path: Path = DEFAULT_JSON,
    source_data_path: Path = DEFAULT_CSV,
    cycles_per_family: int = FAMILY_CYCLES,
) -> dict[str, Any]:
    if cycles_per_family < 1_000:
        raise ValueError("cycles_per_family must be at least 1000")
    build_dir.mkdir(parents=True, exist_ok=True)
    trace_path, families, generation_seconds = generate_trace(build_dir, cycles_per_family)
    crosscheck = legacy_crosscheck()
    toolchain = build_cxxrtl(build_dir)
    cxxrtl_rows, cxxrtl_seconds, cxxrtl_stderr = run_cxxrtl(
        toolchain["executable"], toolchain["environment"], trace_path
    )
    stderr_path = build_dir / "cxxrtl_stderr.log"
    stderr_path.write_text(cxxrtl_stderr, encoding="utf-8")
    t621: dict[str, Any] = {"present": T621_REPORT.is_file(), "pass": False, "sha256": None}
    if T621_REPORT.is_file():
        anchor = json.loads(T621_REPORT.read_text(encoding="utf-8"))
        t621.update({
            "pass": all(bool(value) for value in anchor.get("gates", {}).values()),
            "sha256": _sha256(T621_REPORT),
            "verdict": anchor.get("verdict"),
        })
    public_toolchain = {key: value for key, value in toolchain.items() if key not in ("executable", "environment")}
    report: dict[str, Any] = {
        "task_id": "T6.2.2",
        "schema_version": "t6.2.2-long-rtl-qualification-v1",
        "evidence_scope": "board-independent software golden and CXXRTL qualification with abstract receiver/FIFO disturbance model only",
        "cycles_per_family": cycles_per_family,
        "family_names": list(FAMILY_NAMES),
        "trace": {
            "path": _relative(trace_path),
            "rows": cycles_per_family * len(FAMILY_NAMES),
            "row_bytes": TRACE_STRUCT.size,
            "bytes": trace_path.stat().st_size,
            "sha256": _sha256(trace_path),
        },
        "python_families": families,
        "aggregate_python": _aggregate_python(families),
        "legacy_crosscheck": crosscheck,
        "cxxrtl_families": cxxrtl_rows,
        "toolchain": public_toolchain,
        "timing_seconds": {
            "trace_generation": generation_seconds,
            "yosys": toolchain["yosys_seconds"],
            "compile": toolchain["compile_seconds"],
            "cxxrtl": cxxrtl_seconds,
        },
        "t6_2_1_anchor": t621,
        "fault_contract": {
            "reachable_fault_bits_required_nonzero": list(REACHABLE_FAULT_BITS),
            "structural_zero_fault_bits_required_zero": list(STRUCTURAL_ZERO_FAULT_BITS),
            "structural_zero_reason": {
                "5": "MAX_TRUSTED_BANK_VERSION is uint16 max in production qualification",
                "6": "decision version is latched with the request by construction",
                "7": "ordered II=1 pipeline plus safe drained commits prevents a newer accepted version overtaking an older request",
                "10": "MAP decision exists whenever v4 is consumed",
                "11": "MAP alignment/action is construction checked",
            },
            "rollback_scope": "rollback is represented by rejected monotonic-CAS commit attempts, not forged as an unreachable datapath state",
        },
        "transport_contract": "abstract bounded FIFO/receiver sequence checker; physical transport, CDC, pins, bitstream and board timing are excluded",
    }
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = semantic_mutation_audit(report)
    report["gates"]["semantic_mutations"] = (
        report["semantic_mutation_audit"]["detected"]
        == report["semantic_mutation_audit"]["count"]
        == 8
    )
    report["verdict"] = VERDICT if all(report["gates"].values()) else NON_QUALIFYING_VERDICT if cycles_per_family < FAMILY_CYCLES else "FAIL_LONG_RTL_QUALIFICATION"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_source_data(source_data_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--cycles-per-family", type=int, default=FAMILY_CYCLES)
    args = parser.parse_args(argv)
    report = run_qualification(
        build_dir=args.build_dir,
        artifact_path=args.artifact,
        source_data_path=args.source_data,
        cycles_per_family=args.cycles_per_family,
    )
    print(json.dumps({
        "verdict": report["verdict"],
        "cycles": report["aggregate_python"]["cycles"],
        "cxxrtl_mismatches": sum(row["mismatches"] for row in report["cxxrtl_families"]),
        "gates": report["gates"],
        "artifact": _relative(args.artifact),
    }, indent=2))
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
