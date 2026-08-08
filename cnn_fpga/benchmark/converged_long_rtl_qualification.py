"""T6.25.3 current-source million-cycle qualification of the converged top."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark.long_rtl_qualification import AbstractTransportAdapter, XorShift32
from cnn_fpga.benchmark.rtl_fast_path_equivalence import discover_tools
from cnn_fpga.runtime.converged_production_reference import (
    ConvergedCycleOutput,
    ConvergedInputs,
    ConvergedProductionReference,
    REJECT_ACTIVE_BANK,
    REJECT_BUSY,
    REJECT_CONFLICT,
    REJECT_CRC32,
    REJECT_DRAIN_GUARD,
    REJECT_INCOMPLETE,
    REJECT_NO_PENDING,
    REJECT_NO_SESSION,
    REJECT_UNTRUSTED,
    REJECT_VERSION,
    REJECT_WORD_ORDER,
    image_crc32,
)
from cnn_fpga.runtime.fast_production_core_reference import (
    LLR_MASK,
    corrupt_input_crc,
    crc16_int_little_endian,
    encode_fast_input_word,
    load_frozen_rtl_tables,
)
from cnn_fpga.runtime.route_a_fixed_policy_reference import (
    ACTION_INTEGRITY_ROLLBACK,
    ACTION_LEAKAGE_RESET,
    ACTION_OPEN,
    ACTION_TAIL_EWMA,
    ACTION_UNCERTAIN_EWMA,
    RouteAPolicyInputs,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
REFERENCE = ROOT / "cnn_fpga/runtime/converged_production_reference.py"
CORE_REFERENCE = ROOT / "cnn_fpga/runtime/fast_production_core_reference.py"
POLICY_REFERENCE = ROOT / "cnn_fpga/runtime/route_a_fixed_policy_reference.py"
TOP = ROOT / "cnn_fpga/rtl/gkp_route_a_converged_production_top.sv"
CORE = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
POLICY = ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv"
ADMISSION = ROOT / "cnn_fpga/rtl/route_a_commit_admission.sv"
MANAGER = ROOT / "cnn_fpga/rtl/gkp_parameter_bank_manager.sv"
DRIVER = ROOT / "cnn_fpga/rtl/converged_long_cxxrtl_driver.cc"
FORMAL_REPORT = ROOT / "docs/t6_25_2_converged_rtl_formal.json"
CONFIG = ROOT / "configs/phase6d/t6_25_3_converged_long_rtl.json"
DEFAULT_BUILD = ROOT / "build/t6_25_3_converged_long_rtl"
REPORT = ROOT / "docs/t6_25_3_converged_long_rtl.json"
SOURCE_DATA = ROOT / "docs/t6_25_3_converged_long_rtl_source_data.csv"
MARKDOWN = ROOT / "docs/converged_long_rtl_qualification.md"
FAMILY_CYCLES = 100_000
VERDICT = "PASS_EXACT_CONVERGED_TOP_MILLION_CYCLE_CXXRTL_QUALIFICATION"
SHORT_VERDICT = "NON_QUALIFYING_SHORT_CURRENT_SOURCE_REPLAY"
EXPECTED_BYTES = 148
INPUT_STRUCT = struct.Struct("<3BQ3BHHI2BHI4BHH16B148s")
assert INPUT_STRUCT.size == 202

FAMILY_NAMES = (
    "nominal_ii1",
    "smooth_router_commits",
    "tail_hysteresis",
    "calibration_integrity",
    "telegraph_commit_race",
    "burst_crc_deadline",
    "leakage_reset",
    "management_full_image",
    "management_fault_snapshot",
    "compound_transport",
)
ACTION_NAMES = ("open", "tail_ewma", "uncertain_ewma", "leakage_reset", "integrity_rollback")
REASON_NAMES = (
    "adaptive_ready", "raw_tail", "ood_event", "tail_latched",
    "posterior_uncertain", "leakage", "integrity", "posterior_sum", "version",
)
REJECT_NAMES = {
    REJECT_CONFLICT: "conflict",
    REJECT_BUSY: "busy",
    REJECT_ACTIVE_BANK: "active_bank",
    REJECT_VERSION: "version",
    REJECT_DRAIN_GUARD: "drain_guard",
    REJECT_NO_SESSION: "no_session",
    REJECT_WORD_ORDER: "word_order",
    REJECT_CRC32: "crc32",
    REJECT_INCOMPLETE: "incomplete",
    REJECT_NO_PENDING: "no_pending",
    REJECT_UNTRUSTED: "untrusted",
}
# The converged manager makes several raw-core fault modes unreachable by
# construction.  Keep injectable faults and composition-protected faults
# separate: forcing an untrusted active bank here would recreate the raw-pin
# bypass that T6.25.2 removed.
INJECTABLE_CORE_FAULT_BITS = (0, 1, 2, 8, 9, 12, 13)
COMPOSITION_PROTECTED_CORE_FAULT_BITS = (3, 4, 5, 6, 7, 10, 11)
MEMORY_FILES = tuple(
    ROOT / f"cnn_fpga/rtl/generated/t5_5_1_bank{bank}_{phase}.mem"
    for bank in range(2) for phase in ("x", "z")
)


class IntegrityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing binding {path}")
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _binding_live(row: Mapping[str, Any]) -> bool:
    path = ROOT / str(row["path"])
    return path.is_file() and _sha256(path) == row["sha256"] and path.stat().st_size == int(row["bytes"])


def _run(
    command: Sequence[str | Path], *, env: Mapping[str, str], timeout: int = 3600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(value) for value in command], cwd=ROOT, env=dict(env), text=True,
        encoding="utf-8", errors="replace", capture_output=True, check=True,
        timeout=timeout,
    )


def _normal_word(cycle: int, rng: XorShift32, *, x: int = 0, z: int = 0) -> int:
    value = rng.next()
    return encode_fast_input_word(
        syndrome_code=(value ^ (cycle * 73)) & 0x3FF,
        syndrome_x_code=x,
        syndrome_z_code=z,
        phase=(value >> 10) & 1,
        ood_score=(value >> 11) & 0x7F,
        parameter_age=(value >> 18) & 0x1FFF,
    )


def _posterior_inputs(family_id: int, cycle: int) -> RouteAPolicyInputs:
    local = cycle % 4096
    values: dict[str, int] = {
        "posterior_valid": 1,
        "p_normal": 255,
        "p_smooth": 0,
        "p_calibration": 0,
        "p_burst": 0,
        "ood_code": 0,
        "router_boundary": 0,
        "window_prequential_win": 0,
        "integrity_fault": 0,
        "version_fault": 0,
        "integrity_clear": 0,
        "leakage_event": 0,
        "reset_ack": 0,
        "lkg_bank": 0,
    }
    if family_id == 1:
        values.update(p_normal=178, p_smooth=77)
        if local == 0:
            values.update(router_boundary=1, window_prequential_win=(cycle // 4096) & 1)
    elif family_id == 2:
        step = cycle % 256
        if step in (0, 1):
            values.update(p_normal=25, p_smooth=0, p_calibration=230, p_burst=0)
        elif 2 <= step <= 9:
            values.update(p_normal=203, p_smooth=0, p_calibration=52, p_burst=0)
        elif step == 20:
            values.update(p_normal=191, p_smooth=64)
        elif step == 21:
            values.update(p_normal=235, p_smooth=20, ood_code=193)
    elif family_id == 3:
        if local == 0:
            values.update(p_normal=100, p_smooth=100, p_calibration=30, p_burst=24)
        elif local == 8:
            values["integrity_clear"] = 1
        elif local == 512:
            values["integrity_fault"] = 1
        elif local == 520:
            values["integrity_clear"] = 1
        elif local == 1024:
            values["version_fault"] = 1
        elif local == 1032:
            values["integrity_clear"] = 1
    elif family_id == 4:
        if (cycle // 512) & 1:
            values.update(p_normal=25, p_smooth=0, p_calibration=0, p_burst=230)
        else:
            values.update(p_normal=178, p_smooth=77)
            if cycle % 512 == 0:
                values.update(router_boundary=1, window_prequential_win=(cycle // 1024) & 1)
    elif family_id == 5:
        values.update(p_normal=25, p_smooth=0, p_calibration=0, p_burst=230)
    elif family_id == 6:
        if local in (0, 1, 2):
            values["leakage_event"] = 1
        elif local == 16:
            values["reset_ack"] = 1
    return RouteAPolicyInputs(**values)


def _blank_inputs(cycle: int, cycles: int, rng: XorShift32, family_id: int) -> ConvergedInputs:
    return ConvergedInputs(
        in_valid=int(cycle < cycles - 8),
        in_word=_normal_word(cycle, rng),
        safe_boundary=1,
        posterior=_posterior_inputs(family_id, cycle),
    )


def _full_image_words(tables: list[list[list[int]]], bank: int) -> list[int]:
    return [int(word) & LLR_MASK for phase in range(2) for word in tables[bank][phase]]


def _apply_image_transaction(
    inputs: ConvergedInputs,
    *,
    cycle: int,
    start: int,
    bank: int,
    image_version: int,
    words: list[int],
    expected_active_version: int,
    corrupt_crc: bool = False,
) -> ConvergedInputs:
    if cycle == start:
        crc = image_crc32(words) ^ int(corrupt_crc)
        return replace(
            inputs,
            cfg_begin_valid=1,
            cfg_begin_bank=bank,
            cfg_expected_active_version=expected_active_version,
            cfg_new_image_version=image_version,
            cfg_expected_crc32=crc,
        )
    word_index = cycle - (start + 1)
    if 0 <= word_index < len(words):
        return replace(
            inputs,
            cfg_word_valid=1,
            cfg_word_phase=int(word_index >= 257),
            cfg_word_address=word_index % 257,
            cfg_word_data=words[word_index],
        )
    if cycle == start + 1 + len(words):
        return replace(inputs, cfg_finalize_valid=1)
    return inputs


def _management_stimulus(
    family_id: int,
    cycle: int,
    inputs: ConvergedInputs,
    reference: ConvergedProductionReference,
    tables: list[list[list[int]]],
) -> tuple[ConvergedInputs, dict[str, int]]:
    tags: dict[str, int] = {}
    words = _full_image_words(tables, 1)
    if family_id == 7:
        inputs = _apply_image_transaction(
            inputs, cycle=cycle, start=32, bank=1, image_version=2,
            words=words, expected_active_version=0,
        )
        if cycle == 32:
            tags["full_image_begin"] = 1
        if cycle == 547:
            tags["full_image_finalize"] = 1
        if cycle == 560:
            inputs = replace(
                inputs, host_commit_valid=1, host_commit_bank=1,
                host_expected_active_version=reference.core.active_version,
                host_new_activation_version=(reference.core.active_version + 1) & 0xFFFF,
            )
            tags["host_commit_attempt"] = 1
        if cycle == 700:
            inputs = replace(inputs, management_snapshot_request=1)
            tags["snapshot_request"] = 1
        if cycle == 701:
            inputs = replace(inputs, management_snapshot_request=1)
            tags["snapshot_busy_request"] = 1
        if cycle == 1000:
            inputs = replace(
                inputs, safe_boundary=0, host_commit_valid=1,
                host_commit_bank=1 - reference.core.active_bank,
                host_expected_active_version=reference.core.active_version,
                host_new_activation_version=(reference.core.active_version + 1) & 0xFFFF,
            )
            tags["cancel_setup"] = 1
        if cycle == 1001:
            inputs = replace(inputs, commit_cancel_valid=1)
            tags["cancel_attempt"] = 1
    elif family_id == 8:
        if cycle == 10:
            inputs = replace(inputs, cfg_begin_valid=1, management_snapshot_request=1)
            tags["conflict"] = 1
        elif cycle in (20, 21):
            inputs = replace(inputs, management_snapshot_request=1)
            tags["snapshot_request"] = 1
        elif cycle == 50:
            inputs = replace(
                inputs, cfg_begin_valid=1, cfg_begin_bank=reference.core.active_bank,
                cfg_expected_active_version=reference.core.active_version,
                cfg_new_image_version=2, cfg_expected_crc32=0,
            )
            tags["active_bank_begin"] = 1
        elif cycle == 60:
            inputs = replace(
                inputs, cfg_begin_valid=1, cfg_begin_bank=1 - reference.core.active_bank,
                cfg_expected_active_version=(reference.core.active_version + 1) & 0xFFFF,
                cfg_new_image_version=2, cfg_expected_crc32=0,
            )
            tags["version_begin"] = 1
        elif cycle == 70:
            inputs = replace(
                inputs, host_commit_valid=1, host_commit_bank=1 - reference.core.active_bank,
                host_expected_active_version=reference.core.active_version,
                host_new_activation_version=0,
            )
            tags["zero_version_attempt"] = 1
        elif cycle == 80:
            inputs = replace(
                inputs, host_commit_valid=1, host_commit_bank=reference.core.active_bank,
                host_expected_active_version=reference.core.active_version,
                host_new_activation_version=(reference.core.active_version + 1) & 0xFFFF,
            )
            tags["active_bank_commit"] = 1
        elif cycle == 100:
            inputs = replace(inputs, cfg_word_valid=1, cfg_word_address=0, cfg_word_data=0)
            tags["word_without_session"] = 1
        elif cycle == 110:
            inputs = replace(
                inputs, cfg_begin_valid=1, cfg_begin_bank=1,
                cfg_expected_active_version=reference.core.active_version,
                cfg_new_image_version=2, cfg_expected_crc32=0,
            )
        elif cycle == 111:
            inputs = replace(inputs, cfg_word_valid=1, cfg_word_address=1, cfg_word_data=0)
            tags["word_order"] = 1
        inputs = _apply_image_transaction(
            inputs, cycle=cycle, start=200, bank=1, image_version=3,
            words=words, expected_active_version=0, corrupt_crc=True,
        )
        if cycle == 715:
            tags["bad_crc_finalize"] = 1
        if cycle == 730:
            inputs = replace(
                inputs, host_commit_valid=1, host_commit_bank=1,
                host_expected_active_version=reference.core.active_version,
                host_new_activation_version=(reference.core.active_version + 1) & 0xFFFF,
            )
            tags["untrusted_commit"] = 1
        if cycle == 800:
            inputs = replace(
                inputs, cfg_begin_valid=1, cfg_begin_bank=1,
                cfg_expected_active_version=reference.core.active_version,
                cfg_new_image_version=4, cfg_expected_crc32=0,
            )
        elif cycle == 801:
            inputs = replace(inputs, cfg_word_valid=1, cfg_word_address=0, cfg_word_data=words[0])
        elif cycle == 802:
            inputs = replace(inputs, cfg_finalize_valid=1)
            tags["incomplete_finalize"] = 1
        if cycle == 900:
            inputs = replace(inputs, commit_cancel_valid=1)
            tags["cancel_without_pending"] = 1
        if cycle == 950:
            inputs = replace(
                inputs, cfg_begin_valid=1, cfg_begin_bank=1,
                cfg_expected_active_version=reference.core.active_version,
                cfg_new_image_version=5, cfg_expected_crc32=0,
            )
        elif cycle == 951:
            inputs = replace(inputs, cfg_abort_valid=1)
            tags["abort_session"] = 1
        elif cycle == 960:
            inputs = replace(inputs, cfg_abort_valid=1)
            tags["abort_without_session"] = 1
    return inputs, tags


def _stimulus(
    family_id: int,
    cycle: int,
    cycles: int,
    rng: XorShift32,
    reference: ConvergedProductionReference,
    tables: list[list[list[int]]],
    transport: AbstractTransportAdapter | None,
) -> tuple[ConvergedInputs, dict[str, int]]:
    inputs = _blank_inputs(cycle, cycles, rng, family_id)
    tags: dict[str, int] = {}
    if family_id == 4 and cycle % 512 in (1, 2, 3):
        inputs = replace(
            inputs, host_commit_valid=1, host_commit_bank=1 - reference.core.active_bank,
            host_expected_active_version=reference.core.active_version,
            host_new_activation_version=(reference.core.active_version + 1) & 0xFFFF,
        )
        tags["host_commit_attempt"] = 1
    elif family_id == 5:
        local = cycle % 256
        if local == 0:
            inputs = replace(inputs, in_word=corrupt_input_crc(inputs.in_word))
        elif local == 1:
            payload = inputs.in_word & ((1 << 42) - 1) & ~(1 << 41)
            inputs = replace(inputs, in_word=payload | (crc16_int_little_endian(payload, 6) << 42))
        elif local == 2:
            inputs = replace(inputs, in_word=encode_fast_input_word(
                syndrome_code=cycle & 0x3FF, syndrome_x_code=0, syndrome_z_code=0,
                phase=cycle & 1, ood_score=0, parameter_age=0xFFFF,
            ))
        elif local == 3:
            inputs = replace(inputs, in_word=encode_fast_input_word(
                syndrome_code=cycle & 0x3FF, syndrome_x_code=0, syndrome_z_code=0,
                phase=cycle & 1, ood_score=0, parameter_age=0, observation_valid=0,
            ))
        elif local == 4:
            inputs = replace(inputs, in_word=encode_fast_input_word(
                syndrome_code=cycle & 0x3FF, syndrome_x_code=0, syndrome_z_code=0,
                phase=cycle & 1, ood_score=255, parameter_age=0,
            ))
        elif local == 5:
            # Core-level protocol violation: an unsolicited reset ACK while
            # the event FSM is not in RESET_REQUEST must fail closed (bit 12).
            inputs = replace(inputs, in_word=encode_fast_input_word(
                syndrome_code=cycle & 0x3FF, syndrome_x_code=0, syndrome_z_code=0,
                phase=cycle & 1, ood_score=0, parameter_age=0, reset_ack=1,
            ))
    elif family_id == 6:
        local = cycle % 4096
        if local in (0, 1, 2):
            inputs = replace(inputs, in_word=encode_fast_input_word(
                syndrome_code=cycle & 0x3FF, syndrome_x_code=2, syndrome_z_code=0,
                phase=cycle & 1, ood_score=0, parameter_age=0,
            ))
        elif local == 16:
            inputs = replace(inputs, in_word=encode_fast_input_word(
                syndrome_code=cycle & 0x3FF, syndrome_x_code=0, syndrome_z_code=0,
                phase=cycle & 1, ood_score=0, parameter_age=0, reset_ack=1,
            ))
    elif family_id in (7, 8):
        inputs, management_tags = _management_stimulus(
            family_id, cycle, inputs, reference, tables
        )
        tags.update(management_tags)
    elif family_id == 9:
        assert transport is not None
        active_end = cycles - max(4000, cycles // 10)
        source = _normal_word(cycle, rng) if cycle < active_end else None
        local = cycle % 5000
        delivered = transport.cycle(
            cycle, source,
            pause=bool(cycle < active_end and local < 81),
            drop=bool(source is not None and local == 1003),
            duplicate=bool(source is not None and local == 2003),
            reorder=bool(source is not None and local == 82),
        )
        if delivered is None:
            if cycle < cycles - 8 and cycle >= active_end and not transport.queue and not transport.marker_reasons:
                delivered = _normal_word(cycle, rng)
            else:
                inputs = replace(inputs, in_valid=0, in_word=0)
        if delivered is not None:
            inputs = replace(inputs, in_valid=1, in_word=delivered)
            payload = delivered & ((1 << 42) - 1)
            marker = ((payload >> 40) & 3) != 3
            if marker:
                inputs = replace(inputs, posterior=replace(inputs.posterior, integrity_fault=1))
                tags["transport_fault_token"] = 1
    return inputs, tags


def _word_crc_ok(word: int, payload_bits: int, byte_count: int) -> bool:
    payload = int(word) & ((1 << payload_bits) - 1)
    return ((int(word) >> payload_bits) & 0xFFFF) == crc16_int_little_endian(payload, byte_count)


def _append_int(buffer: bytearray, value: int, width: int) -> None:
    buffer.extend(int(value).to_bytes(width, "little", signed=False))


def _expected_bytes(output: ConvergedCycleOutput) -> bytes:
    pulses = output.pulses
    buffer = bytearray()
    for key in (
        "cfg_begin_ack", "cfg_word_ack", "cfg_finalize_ack", "cfg_abort_ack",
        "host_commit_ack", "policy_commit_ack", "commit_complete",
        "commit_complete_source_policy", "commit_cancel_ack",
        "management_snapshot_ack", "management_state_valid", "management_reject",
        "management_reject_reason", "management_ready", "host_commit_blocked",
    ):
        _append_int(buffer, pulses[key], 1)
    _append_int(buffer, output.core.output_word, 15)
    _append_int(buffer, output.core.state_word, 29)
    _append_int(buffer, output.route.action_word, 10)
    _append_int(buffer, output.route.state_word, 12)
    _append_int(buffer, output.route.version_word, 8)
    _append_int(buffer, output.management_state_word, 20)
    _append_int(buffer, output.core.map_valid, 1)
    _append_int(buffer, output.core.map_llr_twos, 4)
    _append_int(buffer, output.core.map_address, 2)
    _append_int(buffer, output.core.active_version, 2)
    _append_int(buffer, output.core.active_bank, 1)
    _append_int(buffer, output.route.action, 1)
    _append_int(buffer, output.route.reason, 1)
    _append_int(buffer, output.route.selected_bank, 1)
    _append_int(buffer, output.route.commit_pending, 1)
    manager = output.manager_debug
    for key, width in (
        ("commit_pending", 1), ("commit_pending_bank", 1),
        ("commit_pending_version", 2), ("commit_pending_source_policy", 1),
        ("cfg_session_active", 1), ("cfg_staged_bank", 1), ("cfg_word_count", 2),
        ("cfg_all_words_received", 1), ("retired_bank_drain_count", 1),
        ("bank0_trusted", 1), ("bank1_trusted", 1),
        ("bank0_image_version", 2), ("bank1_image_version", 2),
    ):
        _append_int(buffer, manager[key], width)
    interface = output.core_interface_debug
    for key, width in (
        ("core_cfg_we", 1), ("core_cfg_bank", 1), ("core_commit_valid", 1),
        ("core_commit_bank", 1), ("core_commit_version", 2),
    ):
        _append_int(buffer, interface[key], width)
    _append_int(buffer, output.admission_debug.effective_commit_valid, 1)
    _append_int(buffer, output.admission_debug.effective_commit_source_policy, 1)
    _require(len(buffer) == EXPECTED_BYTES, f"expected pack is {len(buffer)}, not {EXPECTED_BYTES}")
    return bytes(buffer)


def _pack_trace_row(family_id: int, reset_before: int, inputs: ConvergedInputs, expected: bytes) -> bytes:
    p = inputs.posterior
    return INPUT_STRUCT.pack(
        family_id, reset_before, inputs.in_valid, inputs.in_word, inputs.safe_boundary,
        inputs.cfg_begin_valid, inputs.cfg_begin_bank, inputs.cfg_expected_active_version,
        inputs.cfg_new_image_version, inputs.cfg_expected_crc32, inputs.cfg_word_valid,
        inputs.cfg_word_phase, inputs.cfg_word_address, inputs.cfg_word_data,
        inputs.cfg_finalize_valid, inputs.cfg_abort_valid, inputs.host_commit_valid,
        inputs.host_commit_bank, inputs.host_expected_active_version,
        inputs.host_new_activation_version, inputs.commit_cancel_valid,
        inputs.management_snapshot_request, p.posterior_valid, p.p_normal, p.p_smooth,
        p.p_calibration, p.p_burst, p.ood_code, p.router_boundary,
        p.window_prequential_win, p.integrity_fault, p.version_fault,
        p.integrity_clear, p.leakage_event, p.reset_ack, p.lkg_bank, expected,
    )


def _new_stats(family_id: int) -> dict[str, Any]:
    return {
        "family_id": family_id, "family": FAMILY_NAMES[family_id], "cycles": 0,
        "input_valid": 0, "output_valid": 0, "route_valid": 0, "map_valid": 0,
        "latency_violations": 0, "map_latency_violations": 0,
        "route_alignment_violations": 0, "ii1_input_pairs": 0, "ii1_output_pairs": 0,
        "undefined_actions": 0, "crc_errors": 0, "version_transitions": 0,
        "silent_version_wraps": 0, "max_active_version": 0,
        "actions": {name: 0 for name in ACTION_NAMES},
        "reasons": {name: 0 for name in REASON_NAMES},
        "reject_reasons": {name: 0 for name in REJECT_NAMES.values()},
        "core_fault_bits": {str(bit): 0 for bit in range(14)},
        "cfg_begin_acks": 0, "cfg_word_acks": 0, "cfg_finalize_acks": 0,
        "cfg_abort_acks": 0, "host_commit_acks": 0, "policy_commit_acks": 0,
        "host_commit_blocks": 0, "commit_completes": 0,
        "host_commit_completes": 0, "policy_commit_completes": 0,
        "cancel_acks": 0, "snapshot_acks": 0, "snapshot_valids": 0,
        "management_rejects": 0, "trust_zero_cycles": 0,
        "fault_to_clean_recoveries": 0, "route_to_open_recoveries": 0,
        "zero_version_attempts": 0, "full_image_begins": 0,
        "full_image_finalizes": 0, "transport": {},
        "_pending_inputs": [], "_pending_maps": [], "_previous_input": 0,
        "_previous_output": 0, "_previous_version": None,
        "_awaiting_fault_recovery": False, "_awaiting_route_recovery": False,
    }


def _update_stats(
    stats: dict[str, Any], cycle: int, inputs: ConvergedInputs,
    output: ConvergedCycleOutput, tags: Mapping[str, int],
) -> None:
    stats["cycles"] += 1
    stats["input_valid"] += inputs.in_valid
    if inputs.in_valid:
        stats["_pending_inputs"].append(cycle)
        stats["_pending_maps"].append(cycle)
    if stats["_previous_input"] and inputs.in_valid:
        stats["ii1_input_pairs"] += 1
    core_payload = output.core.output_word & ((1 << 102) - 1)
    out_valid = core_payload & 1
    route_valid = output.route.action_word & 1
    stats["output_valid"] += out_valid
    stats["route_valid"] += route_valid
    stats["map_valid"] += output.core.map_valid
    if out_valid:
        due = stats["_pending_inputs"].pop(0) if stats["_pending_inputs"] else None
        stats["latency_violations"] += int(due is None or cycle - due != 6)
    if output.core.map_valid:
        due = stats["_pending_maps"].pop(0) if stats["_pending_maps"] else None
        stats["map_latency_violations"] += int(due is None or cycle - due != 5)
    stats["route_alignment_violations"] += int(route_valid != out_valid)
    if stats["_previous_output"] and out_valid:
        stats["ii1_output_pairs"] += 1
    stats["_previous_input"] = inputs.in_valid
    stats["_previous_output"] = out_valid
    version = output.core.active_version
    previous = stats["_previous_version"]
    if previous is not None:
        stats["version_transitions"] += int(version != previous)
        stats["silent_version_wraps"] += int(version < previous)
    stats["_previous_version"] = version
    stats["max_active_version"] = max(stats["max_active_version"], version)
    stats["actions"][ACTION_NAMES[output.route.action]] += 1
    stats["reasons"][REASON_NAMES[output.route.reason]] += 1
    stats["undefined_actions"] += int(output.route.action > 4 or output.route.reason > 8)
    stats["crc_errors"] += sum((
        int(not _word_crc_ok(output.core.output_word, 102, 13)),
        int(not _word_crc_ok(output.core.state_word, 216, 27)),
        int(not _word_crc_ok(output.route.action_word, 64, 8)),
        int(not _word_crc_ok(output.route.state_word, 80, 10)),
        int(not _word_crc_ok(output.route.version_word, 48, 6)),
        int(bool(output.pulses["management_state_valid"]) and not _word_crc_ok(output.management_state_word, 144, 18)),
    ))
    fault_mask = (core_payload >> 50) & 0x3FFF if out_valid else 0
    if out_valid:
        for bit in range(14):
            stats["core_fault_bits"][str(bit)] += (fault_mask >> bit) & 1
        if fault_mask:
            stats["_awaiting_fault_recovery"] = True
        elif stats["_awaiting_fault_recovery"]:
            stats["fault_to_clean_recoveries"] += 1
            stats["_awaiting_fault_recovery"] = False
    if output.route.action != ACTION_OPEN:
        stats["_awaiting_route_recovery"] = True
    elif stats["_awaiting_route_recovery"]:
        stats["route_to_open_recoveries"] += 1
        stats["_awaiting_route_recovery"] = False
    pulses = output.pulses
    for target, key in (
        ("cfg_begin_acks", "cfg_begin_ack"), ("cfg_word_acks", "cfg_word_ack"),
        ("cfg_finalize_acks", "cfg_finalize_ack"), ("cfg_abort_acks", "cfg_abort_ack"),
        ("host_commit_acks", "host_commit_ack"), ("policy_commit_acks", "policy_commit_ack"),
        ("host_commit_blocks", "host_commit_blocked"), ("commit_completes", "commit_complete"),
        ("cancel_acks", "commit_cancel_ack"), ("snapshot_acks", "management_snapshot_ack"),
        ("snapshot_valids", "management_state_valid"), ("management_rejects", "management_reject"),
    ):
        stats[target] += int(pulses[key])
    if pulses["commit_complete"]:
        stats["policy_commit_completes" if pulses["commit_complete_source_policy"] else "host_commit_completes"] += 1
    if pulses["management_reject"] and pulses["management_reject_reason"] in REJECT_NAMES:
        stats["reject_reasons"][REJECT_NAMES[pulses["management_reject_reason"]]] += 1
    stats["trust_zero_cycles"] += int(
        not output.manager_debug["bank0_trusted"] or not output.manager_debug["bank1_trusted"]
    )
    stats["zero_version_attempts"] += tags.get("zero_version_attempt", 0)
    stats["full_image_begins"] += tags.get("full_image_begin", 0)
    stats["full_image_finalizes"] += tags.get("full_image_finalize", 0)


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    _require(not stats["_pending_inputs"], f"{stats['family']} has unflushed output inputs")
    _require(not stats["_pending_maps"], f"{stats['family']} has unflushed map inputs")
    for key in tuple(stats):
        if key.startswith("_"):
            stats.pop(key)
    return stats


def generate_trace(
    build_dir: Path, cycles_per_family: int,
) -> tuple[Path, list[dict[str, Any]], float]:
    trace = build_dir / "converged_long_trace.bin"
    tables = load_frozen_rtl_tables(ROOT)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with trace.open("wb", buffering=4 * 1024 * 1024) as stream:
        for family_id in range(len(FAMILY_NAMES)):
            reference = ConvergedProductionReference(tables)
            rng = XorShift32(0x243F6A88 ^ (family_id * 0x9E3779B9))
            transport = AbstractTransportAdapter() if family_id == 9 else None
            stats = _new_stats(family_id)
            for cycle in range(cycles_per_family):
                inputs, tags = _stimulus(
                    family_id, cycle, cycles_per_family, rng, reference, tables, transport
                )
                output = reference.step(inputs)
                _update_stats(stats, cycle, inputs, output, tags)
                stream.write(_pack_trace_row(family_id, int(cycle == 0), inputs, _expected_bytes(output)))
            if transport is not None:
                stats["transport"] = transport.summary()
            reports.append(_finalize_stats(stats))
    return trace, reports, time.perf_counter() - started


def build_cxxrtl(build_dir: Path) -> dict[str, Any]:
    tools = discover_tools()
    temp = build_dir / "temp"
    cache = build_dir / "yowasp_cache"
    temp.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    model = build_dir / "converged_long_model.cc"
    executable = build_dir / "converged_long_trace.exe"
    env = os.environ.copy()
    env.update(YOWASP_CACHE_DIR=str(cache), TEMP=str(temp), TMP=str(temp))
    env["PATH"] = str(tools["gpp"].parent) + os.pathsep + env.get("PATH", "")
    sources = " ".join(_relative(path) for path in (CORE, POLICY, ADMISSION, MANAGER, TOP))
    script = (
        f"read_verilog -sv {sources}; "
        "hierarchy -check -top gkp_route_a_converged_production_top; "
        "proc; opt -full; check; stat; "
        f"write_cxxrtl -O3 -g0 {_relative(model)}"
    )
    started = time.perf_counter()
    yosys = _run((tools["yosys"], "-Q", "-p", script), env=env, timeout=1800)
    yosys_seconds = time.perf_counter() - started
    (build_dir / "yosys_cxxrtl.log").write_text(yosys.stdout + yosys.stderr, encoding="utf-8")
    started = time.perf_counter()
    compiled = _run((
        tools["gpp"], "-std=c++17", "-O3", "-DNDEBUG", "-I", tools["include"],
        "-I", build_dir, DRIVER, "-o", executable,
    ), env=env, timeout=1800)
    compile_seconds = time.perf_counter() - started
    (build_dir / "gpp_compile.log").write_text(compiled.stdout + compiled.stderr, encoding="utf-8")
    return {
        "executable": executable,
        "environment": env,
        "yosys_version": _run((tools["yosys"], "-V"), env=env).stdout.strip(),
        "gpp_version": _run((tools["gpp"], "--version"), env=env).stdout.splitlines()[0],
        "yosys_seconds": round(yosys_seconds, 3),
        "compile_seconds": round(compile_seconds, 3),
        "structural_check_zero_problems": "Found and reported 0 problems" in yosys.stdout,
        "model": _binding(model),
        "executable_binding": _binding(executable),
        "yosys_log": _relative(build_dir / "yosys_cxxrtl.log"),
        "compile_log": _relative(build_dir / "gpp_compile.log"),
    }


def _parse_cxxrtl(stdout: str) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(stdout)))
    _require(len(rows) == 1, "invalid CXXRTL family summary")
    return {
        key: value if key in ("actual_digest", "expected_digest") else int(value)
        for key, value in rows[0].items()
    }


def run_cxxrtl(
    executable: Path, env: Mapping[str, str], trace: Path,
) -> tuple[list[dict[str, Any]], float, str]:
    completed: dict[int, subprocess.CompletedProcess[str]] = {}
    started = time.perf_counter()
    workers = min(10, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run, (executable, trace, family), env=env, timeout=7200): family
            for family in range(10)
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    rows: list[dict[str, Any]] = []
    stderr: list[str] = []
    for family in range(10):
        row = _parse_cxxrtl(completed[family].stdout)
        row["family"] = FAMILY_NAMES[family]
        rows.append(row)
        if completed[family].stderr:
            stderr.append(f"[family {family}]\n{completed[family].stderr}")
    return rows, time.perf_counter() - started, "".join(stderr)


def _aggregate(families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "cycles", "input_valid", "output_valid", "route_valid", "map_valid",
        "latency_violations", "map_latency_violations", "route_alignment_violations",
        "ii1_input_pairs", "ii1_output_pairs", "undefined_actions", "crc_errors",
        "version_transitions", "silent_version_wraps", "cfg_begin_acks", "cfg_word_acks",
        "cfg_finalize_acks", "cfg_abort_acks", "host_commit_acks", "policy_commit_acks",
        "host_commit_blocks", "commit_completes", "host_commit_completes",
        "policy_commit_completes", "cancel_acks", "snapshot_acks", "snapshot_valids",
        "management_rejects", "trust_zero_cycles", "fault_to_clean_recoveries",
        "route_to_open_recoveries", "zero_version_attempts", "full_image_begins",
        "full_image_finalizes",
    )
    result = {key: sum(int(row[key]) for row in families) for key in scalar_keys}
    result["max_active_version"] = max(int(row["max_active_version"]) for row in families)
    result["actions"] = {
        name: sum(int(row["actions"][name]) for row in families) for name in ACTION_NAMES
    }
    result["reasons"] = {
        name: sum(int(row["reasons"][name]) for row in families) for name in REASON_NAMES
    }
    result["reject_reasons"] = {
        name: sum(int(row["reject_reasons"][name]) for row in families)
        for name in REJECT_NAMES.values()
    }
    result["core_fault_bits"] = {
        str(bit): sum(int(row["core_fault_bits"][str(bit)]) for row in families)
        for bit in range(14)
    }
    result["silent_overflow"] = sum(
        int(row.get("transport", {}).get("silent_overflow", 0)) for row in families
    )
    result["pending_transport"] = sum(
        int(row.get("transport", {}).get("pending_fifo", 0))
        + int(row.get("transport", {}).get("pending_markers", 0))
        for row in families
    )
    return result


def _formal_anchor() -> dict[str, Any]:
    report = json.loads(FORMAL_REPORT.read_text(encoding="utf-8"))
    bindings = {row["path"]: row for row in report["bindings"]}
    required = (TOP, CORE, POLICY, ADMISSION, MANAGER)
    exact = all(
        _relative(path) in bindings
        and bindings[_relative(path)]["sha256"] == _sha256(path)
        and int(bindings[_relative(path)]["bytes"]) == path.stat().st_size
        for path in required
    )
    return {
        "path": _relative(FORMAL_REPORT), "sha256": _sha256(FORMAL_REPORT),
        "verdict": report["verdict"], "exact_required_source_bindings": exact,
        "actual_core_atomic_commit_returncode": report["formal_results"]["actual_core_atomic_commit"]["returncode"],
        "near_wrap_witness_found": report["formal_results"]["cover_near_wrap_reject"]["model_found"],
        "mutation_closure": report["mutation_summary"],
    }


def evaluate_gates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    aggregate = report["aggregate_python"]
    cxx = report["cxxrtl_families"]
    full_scale = int(report["cycles_per_family"]) >= FAMILY_CYCLES
    transport = report["python_families"][9]["transport"]
    gates = [
        ("qualifying_scale", full_scale and len(cxx) == 10 and int(aggregate["cycles"]) >= 1_000_000),
        ("exact_t6_25_2_source_topology_anchor", report["formal_anchor"]["exact_required_source_bindings"] and report["formal_anchor"]["verdict"] == "PASS_CONVERGED_PRODUCTION_TOP_PROPERTY_COVER_MUTATION_CLOSED"),
        ("yosys_structural_check_clean", report["toolchain"]["structural_check_zero_problems"]),
        ("all_public_outputs_bit_exact", len(cxx) == 10 and sum(int(row["mismatches"]) for row in cxx) == 0 and all(row["actual_digest"] == row["expected_digest"] for row in cxx)),
        ("cxxrtl_full_scale", full_scale and min(int(row["rows"]) for row in cxx) >= FAMILY_CYCLES and sum(int(row["rows"]) for row in cxx) >= 1_000_000),
        ("all_expected_bytes_mutation_checked", sum(int(row["shadow_mutations"]) for row in cxx) == EXPECTED_BYTES and sum(int(row["shadow_mutations_detected"]) for row in cxx) == EXPECTED_BYTES),
        ("six_cycle_source_to_action_and_five_cycle_map", int(aggregate["latency_violations"]) == 0 and int(aggregate["map_latency_violations"]) == 0 and int(aggregate["route_alignment_violations"]) == 0 and int(aggregate["input_valid"]) == int(aggregate["output_valid"]) == int(aggregate["route_valid"]) == int(aggregate["map_valid"]) and sum(int(row["latency_violations"] + row["map_latency_violations"] + row["route_alignment_violations"]) for row in cxx) == 0 and all(int(rtl["output_valid"]) == int(py["output_valid"]) and int(rtl["route_valid"]) == int(py["route_valid"]) and int(rtl["map_valid"]) == int(py["map_valid"]) for rtl, py in zip(cxx, report["python_families"], strict=True))),
        ("ii1_has_no_bubbles", int(aggregate["ii1_input_pairs"]) > 800_000 and int(aggregate["ii1_input_pairs"]) == int(aggregate["ii1_output_pairs"]) and sum(int(row["ii1_input_pairs"] - row["ii1_output_pairs"]) for row in cxx) == 0),
        ("no_undefined_action_or_crc_error", int(aggregate["undefined_actions"]) == 0 and int(aggregate["crc_errors"]) == 0 and sum(int(row["undefined_actions"]) for row in cxx) == 0),
        ("transport_faults_accounted_and_drained", int(aggregate["silent_overflow"]) == 0 and int(aggregate["pending_transport"]) == 0 and int(transport.get("accounted_overflow_events", -1)) == int(transport.get("overflow_events", -2)) and all(int(transport.get(key, 0)) > 0 for key in ("pause_cycles", "overflow_events", "drop_events", "duplicate_events", "reorder_events", "sequence_faults", "deadline_faults", "explicit_fault_markers"))),
        ("all_policy_actions_and_reasons_covered", all(int(value) > 0 for value in aggregate["actions"].values()) and all(int(value) > 0 for value in aggregate["reasons"].values())),
        ("all_manager_reject_reasons_covered", all(int(value) > 0 for value in aggregate["reject_reasons"].values())),
        ("full_image_abort_snapshot_cancel_paths_covered", int(aggregate["cfg_begin_acks"]) > 0 and int(aggregate["cfg_word_acks"]) >= 1028 and int(aggregate["cfg_finalize_acks"]) > 0 and int(aggregate["cfg_abort_acks"]) > 0 and int(aggregate["snapshot_acks"]) > 0 and int(aggregate["snapshot_valids"]) > 0 and int(aggregate["cancel_acks"]) > 0 and int(aggregate["trust_zero_cycles"]) > 0),
        ("host_policy_atomic_commit_sources_covered", int(aggregate["host_commit_acks"]) > 0 and int(aggregate["policy_commit_acks"]) > 0 and int(aggregate["host_commit_completes"]) > 0 and int(aggregate["policy_commit_completes"]) > 0 and int(aggregate["version_transitions"]) > 0),
        ("core_fault_and_recovery_covered", all(int(aggregate["core_fault_bits"][str(bit)]) > 0 for bit in INJECTABLE_CORE_FAULT_BITS) and all(int(aggregate["core_fault_bits"][str(bit)]) == 0 for bit in COMPOSITION_PROTECTED_CORE_FAULT_BITS) and int(aggregate["fault_to_clean_recoveries"]) > 0 and int(aggregate["route_to_open_recoveries"]) > 0),
        ("no_silent_version_wrap_with_formal_near_wrap_anchor", int(aggregate["silent_version_wraps"]) == 0 and sum(int(row["silent_version_wraps"]) for row in cxx) == 0 and int(aggregate["zero_version_attempts"]) > 0 and report["formal_anchor"]["actual_core_atomic_commit_returncode"] == 0 and report["formal_anchor"]["near_wrap_witness_found"] is True),
        ("all_source_bindings_live", all(_binding_live(row) for row in report["bindings"])),
        ("claim_boundary_preboard_only", report["claim_boundary"] == {"board_measurement": None, "measured_latency_jitter_power": None, "fastest_or_sota": False, "multimode_decoder_in_rtl": False}),
    ]
    return [{"gate": name, "passed": bool(passed)} for name, passed in gates]


def semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    mutations: list[dict[str, Any]] = []

    def attempt(name: str, mutate: Any) -> None:
        candidate = copy.deepcopy(report)
        mutate(candidate)
        candidate["semantic_mutations"] = {"detected": 21, "total": 21}
        rejected = not all(row["passed"] for row in evaluate_gates(candidate))
        mutations.append({"mutation": name, "rejected": rejected})

    attempt("hide_mismatch", lambda x: x["cxxrtl_families"][0].update(mismatches=1))
    attempt("short_family", lambda x: x.update(cycles_per_family=99_999))
    attempt("erase_comparator_byte", lambda x: x["cxxrtl_families"][0].update(shadow_mutations_detected=147))
    attempt("change_latency", lambda x: x["aggregate_python"].update(latency_violations=1))
    attempt("drop_final_output", lambda x: x["aggregate_python"].update(output_valid=x["aggregate_python"]["output_valid"] - 1))
    attempt("insert_ii1_bubble", lambda x: x["aggregate_python"].update(ii1_output_pairs=x["aggregate_python"]["ii1_output_pairs"] - 1))
    attempt("undefined_action", lambda x: x["aggregate_python"].update(undefined_actions=1))
    attempt("silent_overflow", lambda x: x["aggregate_python"].update(silent_overflow=1))
    attempt("erase_policy_reason", lambda x: x["aggregate_python"]["reasons"].update(version=0))
    attempt("erase_reject_reason", lambda x: x["aggregate_python"]["reject_reasons"].update(crc32=0))
    attempt("erase_full_image", lambda x: x["aggregate_python"].update(cfg_finalize_acks=0))
    attempt("erase_snapshot", lambda x: x["aggregate_python"].update(snapshot_valids=0))
    attempt("erase_host_commit", lambda x: x["aggregate_python"].update(host_commit_acks=0))
    attempt("erase_policy_commit", lambda x: x["aggregate_python"].update(policy_commit_acks=0))
    attempt("erase_injectable_fault", lambda x: x["aggregate_python"]["core_fault_bits"].update({"12": 0}))
    attempt("insert_composition_protected_fault", lambda x: x["aggregate_python"]["core_fault_bits"].update({"4": 1}))
    attempt("erase_fault_recovery", lambda x: x["aggregate_python"].update(fault_to_clean_recoveries=0))
    attempt("insert_version_wrap", lambda x: x["aggregate_python"].update(silent_version_wraps=1))
    attempt("break_formal_anchor", lambda x: x["formal_anchor"].update(exact_required_source_bindings=False))
    attempt("corrupt_binding", lambda x: x["bindings"][0].update(sha256="0" * 64))
    attempt("promote_board_claim", lambda x: x["claim_boundary"].update(board_measurement={"latency_ns": 1}))
    return {"detected": sum(int(row["rejected"]) for row in mutations), "total": len(mutations), "mutations": mutations}


def _validate_report(report: Mapping[str, Any], *, check_files: bool = True) -> None:
    _require(report["task_id"] == "T6.25.3", "wrong task")
    _require(report["verdict"] == VERDICT, "wrong verdict")
    _require(report["gate_summary"] == {"passed": 19, "total": 19}, "gate closure failed")
    _require(report["semantic_mutations"] == {"detected": 21, "total": 21}, "mutation closure failed")
    recomputed_gates = evaluate_gates(report)
    _require(report["gates"][:-1] == recomputed_gates, "stored gates differ from recomputation")
    recomputed_mutations = semantic_mutation_audit(report)
    _require(report["semantic_mutation_results"] == recomputed_mutations["mutations"], "stored semantic mutations differ from recomputation")
    _require(all(row["passed"] for row in report["gates"]), "failed gate stored")
    _require(all(row["rejected"] for row in report["semantic_mutation_results"]), "surviving semantic mutation")
    _require(sum(int(row["mismatches"]) for row in report["cxxrtl_families"]) == 0, "stored mismatch")
    _require(report["trace"]["bytes"] == report["trace"]["rows"] * INPUT_STRUCT.size, "trace size mismatch")
    if check_files:
        _require(all(_binding_live(row) for row in report["bindings"]), "live binding mismatch")
        _require(_binding_live(report["toolchain"]["model"]), "generated CXXRTL model binding mismatch")
        _require(_binding_live(report["toolchain"]["executable_binding"]), "CXXRTL executable binding mismatch")
        formal = ROOT / report["formal_anchor"]["path"]
        _require(formal.is_file() and _sha256(formal) == report["formal_anchor"]["sha256"], "formal anchor binding mismatch")
        trace = ROOT / report["trace"]["path"]
        _require(trace.is_file() and _sha256(trace) == report["trace"]["sha256"], "trace binding mismatch")


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    cxx = {int(row["family_id"]): row for row in report["cxxrtl_families"]}
    rows: list[dict[str, Any]] = []
    for family in report["python_families"]:
        rtl = cxx[int(family["family_id"])]
        rows.append({
            "section": "family", "key": family["family"], "metric": "cycles",
            "value": family["cycles"],
            "detail": f"mismatch={rtl['mismatches']};latency={rtl['latency_violations']};digest={rtl['actual_digest']}",
        })
    for gate in report["gates"]:
        rows.append({"section": "gate", "key": gate["gate"], "metric": "passed", "value": gate["passed"], "detail": report["verdict"]})
    for mutation in report["semantic_mutation_results"]:
        rows.append({"section": "mutation", "key": mutation["mutation"], "metric": "rejected", "value": mutation["rejected"], "detail": "independent gate recomputation"})
    for binding in report["bindings"]:
        rows.append({"section": "binding", "key": binding["path"], "metric": "sha256", "value": binding["sha256"], "detail": binding["bytes"]})
    return rows


def _write_outputs(report: Mapping[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("section", "key", "metric", "value", "detail"))
        writer.writeheader()
        writer.writerows(_source_rows(report))
    aggregate = report["aggregate_python"]
    MARKDOWN.write_text(f"""# T6.25.3 converged top 百万周期 CXXRTL 资格验证

## 结论

**`{report['verdict']}`**。10 个 family 各 {report['cycles_per_family']:,} cycles，聚合 {aggregate['cycles']:,} cycles；新 trace 对 T6.25.2 exact converged top 的全部 148-byte 公开输出向量逐周期比较，bit mismatch、undefined action、CRC error、silent overflow 与 silent version wrap 均为 0。

## 硬件执行合同

- source-to-action 恰为 6 cycles，MAP debug 恰为 5 cycles；连续输入的 II=1 pair 为 {aggregate['ii1_input_pairs']:,}，输出 pair 数相同，无 bubble。
- 完整镜像事务实际传输 257×2 个 22-bit words，覆盖 CRC32、inactive write、trust、host/policy commit、cancel、drain、snapshot 与全部 11 类 reject reason。
- 可从封装端注入的 core fault 位均被命中并恢复；由 converged manager/数据通路结构排除的 fault 位始终为 0，未通过重建 raw bypass 伪造覆盖。
- CXXRTL comparator 对 148 个 expected bytes 逐字节 shadow mutation，{sum(int(row['shadow_mutations_detected']) for row in report['cxxrtl_families'])}/148 被检测；21/21 report semantic mutations 被独立 gate 重算拒绝。
- 版本长轨无下降；near-wrap 不是靠百万周期从 0 暴力递增，而是绑定同一源码 T6.25.2 的 actual-core arbitrary-state atomic proof 与 near-wrap witness。

## 边界

这仍是 two-state、pre-board CXXRTL。真实 transport/CDC/pins/bitstream、板测 latency/jitter/deadline/power、跨工作 fastest/SOTA 与 multimode decoder in RTL 均未建立。
""", encoding="utf-8")


def run_qualification(
    *, build_dir: Path = DEFAULT_BUILD, cycles_per_family: int = FAMILY_CYCLES,
    write_outputs: bool = True,
) -> dict[str, Any]:
    if cycles_per_family < 2000:
        raise ValueError("cycles_per_family must be >= 2000")
    build_dir.mkdir(parents=True, exist_ok=True)
    trace, families, generation_seconds = generate_trace(build_dir, cycles_per_family)
    toolchain_private = build_cxxrtl(build_dir)
    cxx, cxx_seconds, stderr = run_cxxrtl(
        Path(toolchain_private["executable"]), toolchain_private["environment"], trace
    )
    (build_dir / "cxxrtl_stderr.log").write_text(stderr, encoding="utf-8")
    toolchain = {key: value for key, value in toolchain_private.items() if key not in ("executable", "environment")}
    bindings = [_binding(path) for path in (
        CONFIG, RUNNER, REFERENCE, CORE_REFERENCE, POLICY_REFERENCE, DRIVER,
        TOP, CORE, POLICY, ADMISSION, MANAGER, *MEMORY_FILES,
    )]
    report: dict[str, Any] = {
        "task_id": "T6.25.3",
        "schema_version": "t6.25.3-converged-long-rtl-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycles_per_family": cycles_per_family,
        "family_names": list(FAMILY_NAMES),
        "trace": {
            "path": _relative(trace), "rows": cycles_per_family * len(FAMILY_NAMES),
            "row_bytes": INPUT_STRUCT.size, "expected_output_bytes": EXPECTED_BYTES,
            "bytes": trace.stat().st_size, "sha256": _sha256(trace),
            "role": "raw per-cycle input plus independent expected public-output vector",
        },
        "python_families": families,
        "aggregate_python": _aggregate(families),
        "cxxrtl_families": cxx,
        "formal_anchor": _formal_anchor(),
        "toolchain": toolchain,
        "timing_seconds": {
            "trace_generation": round(generation_seconds, 3),
            "yosys": toolchain["yosys_seconds"], "compile": toolchain["compile_seconds"],
            "cxxrtl": round(cxx_seconds, 3),
        },
        "bindings": bindings,
        "claim_boundary": {
            "board_measurement": None, "measured_latency_jitter_power": None,
            "fastest_or_sota": False, "multimode_decoder_in_rtl": False,
        },
    }
    report["gates"] = evaluate_gates(report)
    audit = semantic_mutation_audit(report) if cycles_per_family >= FAMILY_CYCLES else {"detected": 0, "total": 0, "mutations": []}
    report["semantic_mutations"] = {"detected": audit["detected"], "total": audit["total"]}
    report["semantic_mutation_results"] = audit["mutations"]
    # Mutation closure is itself the nineteenth gate at qualifying scale.
    report["gates"].append({"gate": "all_semantic_mutations_rejected", "passed": audit["detected"] == audit["total"] == 21})
    report["gate_summary"] = {
        "passed": sum(int(row["passed"]) for row in report["gates"]),
        "total": len(report["gates"]),
    }
    if all(row["passed"] for row in report["gates"]):
        report["verdict"] = VERDICT
    elif cycles_per_family < FAMILY_CYCLES and all(
        row["passed"] for row in report["gates"] if row["gate"] not in (
            "qualifying_scale", "cxxrtl_full_scale", "ii1_has_no_bubbles",
            "all_semantic_mutations_rejected",
        )
    ):
        report["verdict"] = SHORT_VERDICT
    else:
        report["verdict"] = "FAIL_CLOSED_CONVERGED_LONG_RTL_QUALIFICATION"
    canonical = copy.deepcopy(report)
    canonical.pop("generated_at_utc", None)
    report["analysis_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if write_outputs and cycles_per_family >= FAMILY_CYCLES:
        _validate_report(report)
        _write_outputs(report)
    return report


def verify() -> dict[str, Any]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    _validate_report(report)
    canonical = copy.deepcopy(report)
    expected = canonical.pop("analysis_sha256")
    canonical.pop("generated_at_utc", None)
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    _require(actual == expected, "analysis hash mismatch")
    _require(SOURCE_DATA.is_file() and SOURCE_DATA.stat().st_size > 0, "source data missing")
    _require(MARKDOWN.is_file() and MARKDOWN.stat().st_size > 0, "markdown missing")
    return {"verdict": report["verdict"], "gates": report["gate_summary"], "analysis_sha256": expected}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--cycles-per-family", type=int, default=FAMILY_CYCLES)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return 0
    report = run_qualification(build_dir=args.build_dir, cycles_per_family=args.cycles_per_family)
    print(json.dumps({
        "verdict": report["verdict"], "gates": report["gate_summary"],
        "cycles": report["aggregate_python"]["cycles"],
        "mismatches": sum(int(row["mismatches"]) for row in report["cxxrtl_families"]),
        "trace": report["trace"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] in (VERDICT, SHORT_VERDICT) else 1


if __name__ == "__main__":
    raise SystemExit(main())
