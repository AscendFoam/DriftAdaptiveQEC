"""T6.7.3 long fixed-point/CXXRTL qualification of the integrated Route-A stack.

The benchmark reuses the T6.2.2 production core and adds the frozen V4
integer policy boundary.  Four smooth and six abrupt/fault families are reset
delimited and independently checked.  The binary trace is the per-cycle Source
Data; the CSV is a compact family-level index into that raw artifact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
from dataclasses import replace
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

import numpy as np

from cnn_fpga.benchmark.long_rtl_qualification import (
    AbstractTransportAdapter,
    XorShift32,
    _fault_marker,
    _normal_word,
)
from cnn_fpga.benchmark.rtl_fast_path_equivalence import discover_tools
from cnn_fpga.benchmark.route_a_posterior_calibration import (
    RouteAPosteriorCalibrationConfig,
    _trajectory,
)
from cnn_fpga.benchmark.route_a_smooth_formal import (
    DEFAULT_CACHE_DIR as SMOOTH_CACHE_DIR,
    _formal_cells_and_seeds as smooth_cells_and_seeds,
    _load_models,
    _load_parents,
    _run_trajectory as run_smooth_trajectory,
)
from cnn_fpga.benchmark.route_a_tail_formal import (
    DEFAULT_CACHE_DIR as TAIL_CACHE_DIR,
    _formal_cells_and_seeds as tail_cells_and_seeds,
    _run_trajectory as run_tail_trajectory,
)
from cnn_fpga.benchmark.unified_comparator_runner import materialize_qualification_trace
from cnn_fpga.decoder.route_a_regime_posterior import temperature_scale
from cnn_fpga.runtime.fast_production_core_reference import (
    FastProductionCoreReference,
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
    RouteAFixedPolicyReference,
    RouteAPolicyInputs,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CORE = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
POLICY = ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv"
TOP = ROOT / "cnn_fpga/rtl/route_a_integrated_qualification_top.sv"
DRIVER = ROOT / "cnn_fpga/rtl/route_a_integrated_cxxrtl_driver.cc"
REFERENCE = ROOT / "cnn_fpga/runtime/route_a_fixed_policy_reference.py"
SMOOTH_FORMAL_RUNNER = ROOT / "cnn_fpga/benchmark/route_a_smooth_formal.py"
TAIL_FORMAL_RUNNER = ROOT / "cnn_fpga/benchmark/route_a_tail_formal.py"
POSTERIOR_MODEL = ROOT / "cnn_fpga/decoder/route_a_regime_posterior.py"
T622 = ROOT / "docs/t6_2_2_long_rtl_qualification.json"
LOCK = ROOT / "docs/t6_6_3_route_a_posterior_threshold_lock.json"
DEFAULT_BUILD = ROOT / "build/t6_7_3_route_a_integrated_rtl"
DEFAULT_JSON = ROOT / "docs/t6_7_3_route_a_integrated_rtl_qualification.json"
DEFAULT_CSV = ROOT / "docs/t6_7_3_route_a_integrated_rtl_source_data.csv"
FAMILY_CYCLES = 100_000
VERDICT = "PASS_ROUTE_A_INTEGRATED_LONG_RTL_QUALIFICATION"
FAMILY_NAMES = (
    "smooth_mean",
    "smooth_variance",
    "smooth_correlation",
    "smooth_periodic",
    "fault_step",
    "fault_telegraph",
    "fault_burst",
    "fault_readout_reset",
    "fault_leakage",
    "fault_compound",
)

# See route_a_integrated_cxxrtl_driver.cc.  Every field is byte-packed little
# endian and the assertion prevents a silent Python/C++ ABI disagreement.
TRACE_STRUCT = struct.Struct("<6BH3BHI2BQ14B2BHBHI15s29s10s12s8s5B")
assert TRACE_STRUCT.size == 131

ACTION_NAMES = ("open", "tail_ewma", "uncertain_ewma", "leakage_reset", "integrity_rollback")
REASON_NAMES = (
    "adaptive_ready", "raw_tail", "ood_event", "tail_latched",
    "posterior_uncertain", "leakage", "integrity", "posterior_sum", "version",
)
REQUIRED_CORE_FAULT_BITS = (0, 1, 2, 4, 8, 9, 12, 13)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str | Path], *, env: Mapping[str, str], timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command], cwd=ROOT, env=dict(env), text=True,
        encoding="utf-8", errors="replace", capture_output=True,
        timeout=timeout, check=True,
    )


def _quantize_posterior(posterior: np.ndarray) -> np.ndarray:
    """Largest-remainder uint8 quantizer with an exact sum of 255 per row."""

    values = np.asarray(posterior, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.all(np.isfinite(values)):
        raise ValueError("unified posterior must be finite Nx4")
    if np.max(np.abs(np.sum(values, axis=1) - 1.0)) > 1e-9 or np.any(values < 0.0):
        raise ValueError("unified posterior rows must be normalized probabilities")
    scaled = values * 255.0
    quantized = np.floor(scaled).astype(np.uint8)
    remainders = 255 - np.sum(quantized.astype(np.int16), axis=1)
    fractions = scaled - np.floor(scaled)
    for row_index, remainder in enumerate(remainders.astype(int)):
        if remainder:
            order = np.argsort(-fractions[row_index], kind="stable")
            quantized[row_index, order[:remainder]] += 1
    if not np.all(np.sum(quantized.astype(np.int16), axis=1) == 255):
        raise RuntimeError("posterior quantizer did not preserve the exact simplex")
    return quantized


def _build_unified_replay_streams(
    cycles_per_family: int,
) -> tuple[list[dict[str, np.ndarray | set[int]]], dict[str, Any]]:
    """Materialize actual frozen-HMM streams from T6.7 formal trajectories.

    Two reset-delimited, hash-bound trajectories per family provide more than
    100k decision cycles.  Directed boundary injections are applied later and
    explicitly counted; all remaining posterior cycles come from this replay.
    """

    parents = _load_parents()
    model, event_model, temperature, _ = _load_models(parents["threshold_lock"])
    calibration = np.asarray(
        materialize_qualification_trace()[0].calibration_residuals,
        dtype=np.float64,
    )
    settings = RouteAPosteriorCalibrationConfig()
    smooth_cells, smooth_seeds = smooth_cells_and_seeds()
    tail_cells, tail_seeds = tail_cells_and_seeds()
    cell_sets = (smooth_cells, tail_cells)
    seed_sets = (smooth_seeds, tail_seeds)
    formal_names = (
        "mean_drift", "variance_drift", "correlation_drift", "periodic_drift",
        "step_calibration_shift", "telegraph_drift", "burst_outlier",
        "readout_reset_fault", "leakage_persistence", "compound_ood",
    )
    streams: list[dict[str, np.ndarray | set[int]]] = []
    manifest_rows: list[dict[str, Any]] = []
    total_cache_hits = 0
    for family_id, formal_name in enumerate(formal_names):
        lane = 0 if family_id < 4 else 1
        matching = [row for row in cell_sets[lane] if str(row["family"]) == formal_name]
        if len(matching) != 6:
            raise ValueError(f"formal replay family {formal_name} is not the frozen six-cell design")
        cell = matching[0]
        posterior_parts: list[np.ndarray] = []
        ood_parts: list[np.ndarray] = []
        valid_parts: list[np.ndarray] = []
        boundary_parts: list[np.ndarray] = []
        preference_parts: list[np.ndarray] = []
        reset_cycles: set[int] = set()
        segment_rows: list[dict[str, Any]] = []
        offset = 0
        for seed in seed_sets[lane][:2]:
            trajectory = _trajectory(cell, int(seed), settings, keep_decisions=False)
            posterior_float = temperature_scale(
                model.filter_base(trajectory.features), temperature
            )
            posterior_code = _quantize_posterior(posterior_float)
            deterministic_ood = np.asarray(trajectory.ood_score_codes, dtype=np.uint8)
            event_ood = event_model.score_codes(trajectory.features)
            ood_update = np.maximum(deterministic_ood, event_ood).astype(np.uint8)
            if len(posterior_code) != len(ood_update):
                raise RuntimeError("unified posterior/OOD update lengths differ")
            run = run_smooth_trajectory if lane == 0 else run_tail_trajectory
            cache_dir = SMOOTH_CACHE_DIR if lane == 0 else TAIL_CACHE_DIR
            result, cache_hit = run(
                cell, int(seed), parents, calibration, cache_dir
            )
            total_cache_hits += int(cache_hit)
            if result["input_sha256"] != trajectory.observed_trace_sha256:
                raise RuntimeError("formal replay trajectory hash differs from cached result")
            cycles = len(posterior_code) * 32
            posterior_cycle = np.repeat(posterior_code, 32, axis=0)
            ood_cycle = np.repeat(ood_update, 32)
            posterior_valid = np.zeros(cycles, dtype=np.uint8)
            posterior_valid[::32] = 1
            router_boundary = np.zeros(cycles, dtype=np.uint8)
            window_win = np.zeros(cycles, dtype=np.uint8)
            for commit in result["commit_rows"]:
                decision = int(commit["commit_decision"])
                if 0 <= decision < cycles:
                    router_boundary[decision] = 1
                    window_win[decision] = int(commit["preferred_expert"] == "window_map")
            if offset:
                reset_cycles.add(offset)
            take = min(cycles, max(0, cycles_per_family - offset))
            posterior_parts.append(posterior_cycle[:take])
            ood_parts.append(ood_cycle[:take])
            valid_parts.append(posterior_valid[:take])
            boundary_parts.append(router_boundary[:take])
            preference_parts.append(window_win[:take])
            segment_rows.append({
                "cell_id": str(cell["cell_id"]),
                "seed": int(seed),
                "start_cycle": offset,
                "stop_cycle_exclusive": offset + take,
                "observed_trace_sha256": trajectory.observed_trace_sha256,
                "posterior_float_sha256": hashlib.sha256(
                    np.asarray(posterior_float, dtype="<f8").tobytes()
                ).hexdigest(),
                "posterior_uint8_sha256": hashlib.sha256(posterior_code.tobytes()).hexdigest(),
                "formal_cache_hit": bool(cache_hit),
            })
            offset += take
            if offset >= cycles_per_family:
                break
        if offset != cycles_per_family:
            raise RuntimeError(f"unified replay family {formal_name} supplies only {offset} cycles")
        stream: dict[str, np.ndarray | set[int]] = {
            "posterior": np.concatenate(posterior_parts),
            "ood": np.concatenate(ood_parts),
            "posterior_valid": np.concatenate(valid_parts),
            "router_boundary": np.concatenate(boundary_parts),
            "window_win": np.concatenate(preference_parts),
            "reset_cycles": reset_cycles,
        }
        stream_hash = hashlib.sha256()
        for field in ("posterior", "ood", "posterior_valid", "router_boundary", "window_win"):
            stream_hash.update(np.asarray(stream[field]).tobytes())
        streams.append(stream)
        manifest_rows.append({
            "family_id": family_id,
            "qualification_family": FAMILY_NAMES[family_id],
            "formal_family": formal_name,
            "cycles": cycles_per_family,
            "segments": segment_rows,
            "segment_reset_cycles": sorted(reset_cycles),
            "stream_sha256": stream_hash.hexdigest(),
        })
    return streams, {
        "schema_version": "t6.7.3-unified-posterior-replay-v1",
        "source": "frozen T6.7.1/T6.7.2 formal observed trajectories and V4 HMM/event model",
        "trajectory_count": 20,
        "formal_cache_hits": total_cache_hits,
        "formal_cache_misses": 20 - total_cache_hits,
        "families": manifest_rows,
    }


def _blank() -> dict[str, int]:
    return {
        "in_valid": 1, "in_word": 0, "safe_boundary": 1,
        "host_commit_valid": 0, "host_commit_bank": 0, "host_commit_version": 0,
        "cfg_we": 0, "cfg_bank": 0, "cfg_phase": 0, "cfg_address": 0, "cfg_data": 0,
        "bank0_trusted": 1, "bank1_trusted": 1,
    }


def _posterior_for(
    family_id: int,
    cycle: int,
    replay: Sequence[Mapping[str, np.ndarray | set[int]]],
) -> tuple[RouteAPolicyInputs, dict[str, int]]:
    stream = replay[family_id]
    p = tuple(int(value) for value in np.asarray(stream["posterior"])[cycle])
    ood = int(np.asarray(stream["ood"])[cycle])
    posterior_valid = int(np.asarray(stream["posterior_valid"])[cycle])
    router_boundary = int(np.asarray(stream["router_boundary"])[cycle])
    window_win = int(np.asarray(stream["window_win"])[cycle])
    tags: dict[str, int] = {"unified_replay_cycle": 1}

    local = cycle % 8_192
    if family_id == 2 and local < 320:
        cases = (
            ((26, 0, 229, 0), 0, "tail_229"),
            ((25, 0, 230, 0), 0, "tail_230"),
            ((235, 20, 0, 0), 192, "ood_192"),
            ((235, 20, 0, 0), 193, "ood_193"),
            ((192, 63, 0, 0), 0, "uncertainty_63"),
            ((191, 64, 0, 0), 0, "uncertainty_64"),
            ((179, 76, 0, 0), 0, "smooth_76"),
            ((178, 77, 0, 0), 0, "smooth_77"),
            ((204, 0, 51, 0), 0, "tail_exit_51"),
            ((203, 0, 52, 0), 0, "tail_exit_52"),
        )
        p, ood, tag = cases[local // 32]
        tags[tag] = posterior_valid
        router_boundary = posterior_valid
        window_win = 1
        tags["unified_replay_cycle"] = 0
        tags["directed_boundary_cycle"] = 1
    elif family_id == 7 and cycle % 4_096 == 0:
        # Establish Window as the active expert independently of the natural
        # readout/reset posterior.  Sixteen cycles later the version fault
        # must race an untrusted auto-LKG commit against a blocked host commit.
        p = (20, 225, 5, 5)
        ood = 0
        posterior_valid = 1
        router_boundary = 1
        window_win = 1
        tags["unified_replay_cycle"] = 0
        tags["directed_boundary_cycle"] = 1
    elif family_id == 6 and local == 320:
        p = (100, 100, 30, 24)  # Invalid sum 254 must fail closed.
        tags["posterior_sum_fault"] = posterior_valid
        tags["unified_replay_cycle"] = 0
        tags["directed_boundary_cycle"] = 1

    integrity_fault = int(family_id == 6 and local == 320 and posterior_valid)
    version_fault = int(family_id == 7 and cycle % 4_096 == 16)
    integrity_clear = int(
        (family_id == 6 and local == 512 and posterior_valid)
        or (family_id == 7 and cycle % 4_096 == 24)
        or (family_id == 9 and cycle % 5_000 == 320)
    )
    leakage_event = int(family_id == 8 and cycle % 4_096 in (0, 1, 2))
    reset_ack = int(family_id == 8 and cycle % 4_096 == 16)
    if family_id == 9 and cycle % 5_000 in (64, 65):
        leakage_event = 1
    if family_id == 9 and cycle % 5_000 == 80:
        reset_ack = 1
    return RouteAPolicyInputs(
        posterior_valid=posterior_valid,
        p_normal=p[0], p_smooth=p[1], p_calibration=p[2], p_burst=p[3],
        ood_code=ood, router_boundary=router_boundary,
        window_prequential_win=window_win, integrity_fault=integrity_fault,
        version_fault=version_fault, integrity_clear=integrity_clear,
        leakage_event=leakage_event, reset_ack=reset_ack, lkg_bank=0,
    ), tags


def _core_stimulus(
    family_id: int,
    cycle: int,
    cycles: int,
    rng: XorShift32,
    core: FastProductionCoreReference,
    policy_inputs: RouteAPolicyInputs,
    transport: AbstractTransportAdapter | None,
) -> tuple[dict[str, int], dict[str, int]]:
    signals = _blank()
    tags: dict[str, int] = {}
    x_code = 0
    z_code = 0
    reset_ack = policy_inputs.reset_ack
    observation_valid = 1
    deadline_ok = 1
    parameter_age = cycle & 0x1FFF

    if family_id == 5 and cycle % 2_048 < 12:
        signals["safe_boundary"] = 0
    if family_id == 7:
        local = cycle % 4_096
        if local == 0:
            observation_valid = 0
        elif local == 1:
            deadline_ok = 0
        elif local == 2:
            parameter_age = 0xFFFF
        elif local == 3:
            reset_ack = 1
        elif local == 4:
            signals["bank0_trusted" if core.active_bank == 0 else "bank1_trusted"] = 0
        if local in (8, 9, 10):
            signals["safe_boundary"] = 0
        if local == 17:
            # Force one auto-LKG commit rejection while a blocked host request
            # races it; trust is restored on the next cycle and the pending
            # fail-closed request must then complete.
            signals["bank0_trusted"] = 0
        if local in (9, 17):
            signals.update(
                host_commit_valid=1,
                host_commit_bank=1 - core.active_bank,
                host_commit_version=(core.active_version + 1) & 0xFFFF,
            )
            if local == 9:
                signals.update(
                    cfg_we=1,
                    cfg_bank=1 - core.active_bank,
                    cfg_phase=0,
                    cfg_address=0,
                    cfg_data=core.tables[1 - core.active_bank][0][0] & LLR_MASK,
                )
                tags["commit_cfg_race"] = 1
        elif local == 10:
            signals.update(
                host_commit_valid=1,
                host_commit_bank=1 - core.active_bank,
                host_commit_version=max(0, core.active_version - 1),
            )
            tags["rollback_attempt"] = 1
    elif family_id == 8:
        if policy_inputs.leakage_event:
            x_code = 2
    elif family_id == 9:
        assert transport is not None
        active_end = cycles - max(4_000, cycles // 10)
        source = _normal_word(cycle, rng) if cycle < active_end else None
        local = cycle % 5_000
        delivered = transport.cycle(
            cycle,
            source,
            pause=bool(cycle < active_end and local < 81),
            drop=bool(source is not None and local == 1_003),
            duplicate=bool(source is not None and local == 2_003),
            reorder=bool(source is not None and local == 82),
        )
        if delivered is None:
            signals["in_valid"] = 0
            signals["in_word"] = 0
            return signals, tags
        signals["in_word"] = delivered
        marker = delivered == _fault_marker(cycle) or ((delivered >> 40) & 3) != 3
        tags["transport_fault_token"] = int(marker)
        return signals, tags

    signals["in_word"] = encode_fast_input_word(
        syndrome_code=(rng.next() ^ (cycle * 73)) & 0x3FF,
        syndrome_x_code=x_code,
        syndrome_z_code=z_code,
        phase=cycle & 1,
        ood_score=policy_inputs.ood_code,
        parameter_age=parameter_age,
        reset_ack=reset_ack,
        observation_valid=observation_valid,
        deadline_ok=deadline_ok,
    )
    if family_id == 7 and cycle % 4_096 == 5:
        signals["in_word"] = corrupt_input_crc(signals["in_word"])
        tags["crc_fault"] = 1
    return signals, tags


def _word_crc_ok(word: int, payload_bits: int, byte_count: int) -> bool:
    payload = word & ((1 << payload_bits) - 1)
    return ((word >> payload_bits) & 0xFFFF) == crc16_int_little_endian(payload, byte_count)


def _new_stats(family_id: int) -> dict[str, Any]:
    return {
        "family_id": family_id, "family": FAMILY_NAMES[family_id], "cycles": 0,
        "input_valid": 0, "output_valid": 0, "core_crc_errors": 0,
        "route_action_crc_errors": 0, "route_state_crc_errors": 0,
        "route_version_crc_errors": 0, "undefined_actions": 0,
        "actions": {name: 0 for name in ACTION_NAMES},
        "reasons": {name: 0 for name in REASON_NAMES},
        "selected_banks": {"ewma": 0, "window": 0},
        "core_fault_bits": {str(index): 0 for index in range(14)},
        "posterior_updates": 0, "router_boundaries": 0, "auto_commit_requests": 0,
        "host_commit_attempts": 0, "host_auto_commit_collisions": 0,
        "host_commit_policy_blocks": 0,
        "unified_replay_cycles": 0, "directed_boundary_cycles": 0,
        "commit_acks": 0, "commit_rejections": 0, "commit_cfg_races": 0,
        "rollback_attempts": 0, "tail_latch_entries": 0, "tail_recoveries": 0,
        "leakage_entries": 0, "integrity_entries": 0,
        "threshold_hits": {name: 0 for name in (
            "tail_229", "tail_230", "ood_192", "ood_193", "uncertainty_63",
            "uncertainty_64", "smooth_76", "smooth_77", "tail_exit_51", "tail_exit_52",
        )},
        "transport": {}, "_previous_action": ACTION_OPEN, "_previous_tail": 0,
    }


def generate_trace(
    build_dir: Path,
    cycles_per_family: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any], float]:
    trace_path = build_dir / "route_a_integrated_trace.bin"
    tables = load_frozen_rtl_tables(ROOT)
    replay, replay_manifest = _build_unified_replay_streams(cycles_per_family)
    reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    with trace_path.open("wb", buffering=4 * 1024 * 1024) as handle:
        for family_id in range(len(FAMILY_NAMES)):
            core = FastProductionCoreReference(tables)
            policy = RouteAFixedPolicyReference()
            rng = XorShift32(0x6A09E667 ^ (family_id * 0x9E3779B9))
            transport = AbstractTransportAdapter() if family_id == 9 else None
            stats = _new_stats(family_id)
            for cycle in range(cycles_per_family):
                replay_resets = replay[family_id]["reset_cycles"]
                reset_before = int(
                    cycle == 0
                    or cycle in replay_resets
                    or (family_id == 9 and cycle == cycles_per_family // 2)
                )
                if reset_before and cycle:
                    core.reset()
                    policy.reset()
                policy_inputs, posterior_tags = _posterior_for(family_id, cycle, replay)
                signals, core_tags = _core_stimulus(
                    family_id, cycle, cycles_per_family, rng, core, policy_inputs, transport
                )
                if core_tags.get("transport_fault_token"):
                    policy_inputs = replace(policy_inputs, integrity_fault=1)
                pre_bank = core.active_bank
                pre_version = core.active_version
                auto_valid, auto_bank, auto_version = policy.peek_auto_commit(
                    safe_boundary=signals["safe_boundary"],
                    active_bank=pre_bank, active_version=pre_version,
                )
                host_allowed = int(
                    bool(signals["host_commit_valid"])
                    and policy.action == ACTION_OPEN
                    and not policy.commit_pending
                    and signals["host_commit_bank"] == policy.selected_bank
                )
                host_blocked = int(signals["host_commit_valid"] and not host_allowed)
                effective_valid = auto_valid or host_allowed
                effective_bank = auto_bank if auto_valid else signals["host_commit_bank"]
                effective_version = auto_version if auto_valid else signals["host_commit_version"]
                output = core.step(
                    in_valid=signals["in_valid"], in_word=signals["in_word"],
                    safe_boundary=signals["safe_boundary"], commit_valid=effective_valid,
                    commit_bank=effective_bank, commit_version=effective_version,
                    cfg_we=signals["cfg_we"], cfg_bank=signals["cfg_bank"],
                    cfg_phase=signals["cfg_phase"], cfg_address=signals["cfg_address"],
                    cfg_data=signals["cfg_data"], bank0_trusted=signals["bank0_trusted"],
                    bank1_trusted=signals["bank1_trusted"],
                )
                route = policy.step(
                    policy_inputs, sample_valid=signals["in_valid"],
                    safe_boundary=signals["safe_boundary"], active_bank=pre_bank,
                    active_version=pre_version, core_output_word=output.output_word,
                    visible_active_bank=output.active_bank,
                    visible_active_version=output.active_version,
                )

                stats["cycles"] += 1
                stats["input_valid"] += signals["in_valid"]
                stats["posterior_updates"] += policy_inputs.posterior_valid
                stats["router_boundaries"] += policy_inputs.router_boundary
                stats["auto_commit_requests"] += auto_valid
                stats["host_commit_attempts"] += signals["host_commit_valid"]
                stats["host_auto_commit_collisions"] += int(auto_valid and signals["host_commit_valid"])
                stats["host_commit_policy_blocks"] += host_blocked
                stats["unified_replay_cycles"] += posterior_tags.get("unified_replay_cycle", 0)
                stats["directed_boundary_cycles"] += posterior_tags.get("directed_boundary_cycle", 0)
                stats["commit_acks"] += output.commit_ack
                stats["commit_rejections"] += int(effective_valid and not output.commit_ack)
                stats["commit_cfg_races"] += core_tags.get("commit_cfg_race", 0)
                stats["rollback_attempts"] += core_tags.get("rollback_attempt", 0)
                stats["actions"][ACTION_NAMES[route.action]] += 1
                stats["reasons"][REASON_NAMES[route.reason]] += 1
                stats["selected_banks"]["window" if route.selected_bank else "ewma"] += 1
                stats["undefined_actions"] += int(route.action > 4 or route.reason > 8)
                stats["tail_latch_entries"] += int(route.tail_latched and not stats["_previous_tail"])
                stats["tail_recoveries"] += int(not route.tail_latched and stats["_previous_tail"])
                stats["leakage_entries"] += int(route.action == ACTION_LEAKAGE_RESET and stats["_previous_action"] != ACTION_LEAKAGE_RESET)
                stats["integrity_entries"] += int(route.action == ACTION_INTEGRITY_ROLLBACK and stats["_previous_action"] != ACTION_INTEGRITY_ROLLBACK)
                stats["_previous_action"] = route.action
                stats["_previous_tail"] = route.tail_latched
                for key, value in posterior_tags.items():
                    if key in stats["threshold_hits"]:
                        stats["threshold_hits"][key] += value
                stats["route_action_crc_errors"] += int(not _word_crc_ok(route.action_word, 64, 8))
                stats["route_state_crc_errors"] += int(not _word_crc_ok(route.state_word, 80, 10))
                stats["route_version_crc_errors"] += int(not _word_crc_ok(route.version_word, 48, 6))
                core_payload = output.output_word & ((1 << 102) - 1)
                stats["core_crc_errors"] += int(not _word_crc_ok(output.output_word, 102, 13))
                if core_payload & 1:
                    stats["output_valid"] += 1
                    mask = (core_payload >> 50) & 0x3FFF
                    for bit in range(14):
                        stats["core_fault_bits"][str(bit)] += (mask >> bit) & 1

                pi = policy_inputs
                handle.write(TRACE_STRUCT.pack(
                    family_id, reset_before, signals["in_valid"], signals["safe_boundary"],
                    signals["host_commit_valid"], signals["host_commit_bank"],
                    signals["host_commit_version"], signals["cfg_we"], signals["cfg_bank"],
                    signals["cfg_phase"], signals["cfg_address"], signals["cfg_data"],
                    signals["bank0_trusted"], signals["bank1_trusted"], signals["in_word"],
                    pi.posterior_valid, pi.p_normal, pi.p_smooth, pi.p_calibration, pi.p_burst,
                    pi.ood_code, pi.router_boundary, pi.window_prequential_win,
                    pi.integrity_fault, pi.version_fault, pi.integrity_clear,
                    pi.leakage_event, pi.reset_ack, pi.lkg_bank,
                    output.commit_ack, output.active_bank, output.active_version,
                    output.map_valid, output.map_address, output.map_llr_twos,
                    output.output_word.to_bytes(15, "little"),
                    output.state_word.to_bytes(29, "little"),
                    route.action_word.to_bytes(10, "little"),
                    route.state_word.to_bytes(12, "little"),
                    route.version_word.to_bytes(8, "little"),
                    route.action, route.reason, route.selected_bank, route.commit_pending,
                    host_blocked,
                ))
            if transport is not None:
                stats["transport"] = transport.summary()
            stats.pop("_previous_action")
            stats.pop("_previous_tail")
            reports.append(stats)
    return trace_path, reports, replay_manifest, time.perf_counter() - started


def build_cxxrtl(build_dir: Path) -> dict[str, Any]:
    tools = discover_tools()
    temp = build_dir / "temp"
    cache = build_dir / "yowasp_cache"
    temp.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    model = build_dir / "route_a_integrated_qualification_model.cc"
    executable = build_dir / "route_a_integrated_trace.exe"
    env = os.environ.copy()
    env.update(YOWASP_CACHE_DIR=str(cache), TEMP=str(temp), TMP=str(temp))
    env["PATH"] = str(tools["gpp"].parent) + os.pathsep + env.get("PATH", "")
    script = (
        f"read_verilog -sv {_relative(CORE)} {_relative(POLICY)} {_relative(TOP)}; "
        "hierarchy -check -top route_a_integrated_qualification_top; proc; check; stat; "
        f"write_cxxrtl -O0 -g0 {_relative(model)}"
    )
    started = time.perf_counter()
    yosys = _run((tools["yosys"], "-Q", "-p", script), env=env, timeout=1800)
    yosys_seconds = time.perf_counter() - started
    (build_dir / "yosys_cxxrtl.log").write_text(yosys.stdout + yosys.stderr, encoding="utf-8")
    started = time.perf_counter()
    compile_run = _run((
        tools["gpp"], "-std=c++17", "-O3", "-DNDEBUG", "-I", tools["include"],
        "-I", build_dir, DRIVER, "-o", executable,
    ), env=env, timeout=1800)
    compile_seconds = time.perf_counter() - started
    (build_dir / "gpp_compile.log").write_text(compile_run.stdout + compile_run.stderr, encoding="utf-8")
    return {
        "executable": executable, "environment": env,
        "yosys_version": _run((tools["yosys"], "-V"), env=env).stdout.strip(),
        "gpp_version": _run((tools["gpp"], "--version"), env=env).stdout.splitlines()[0],
        "yosys_seconds": yosys_seconds, "compile_seconds": compile_seconds,
        "structural_check_zero_problems": "Found and reported 0 problems" in yosys.stdout,
        "model_bytes": model.stat().st_size, "model_sha256": _sha256(model),
        "executable_bytes": executable.stat().st_size, "executable_sha256": _sha256(executable),
        "yosys_log": _relative(build_dir / "yosys_cxxrtl.log"),
        "compile_log": _relative(build_dir / "gpp_compile.log"),
    }


def _parse_cxxrtl(stdout: str) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(stdout)))
    if len(rows) != 1:
        raise RuntimeError("invalid CXXRTL summary")
    row = rows[0]
    return {
        key: int(value) if key not in ("actual_digest", "expected_digest") else value
        for key, value in row.items()
    }


def run_cxxrtl(executable: Path, env: Mapping[str, str], trace: Path) -> tuple[list[dict[str, Any]], float, str]:
    started = time.perf_counter()
    completed: dict[int, subprocess.CompletedProcess[str]] = {}
    with ThreadPoolExecutor(max_workers=min(10, max(1, os.cpu_count() or 1))) as pool:
        futures = {
            pool.submit(_run, (executable, trace, family), env=env, timeout=7200): family
            for family in range(10)
        }
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
    rows = []
    stderr = []
    for family in range(10):
        row = _parse_cxxrtl(completed[family].stdout)
        row["family"] = FAMILY_NAMES[family]
        rows.append(row)
        if completed[family].stderr:
            stderr.append(f"[family {family}]\n{completed[family].stderr}")
    return rows, time.perf_counter() - started, "".join(stderr)


def _aggregate(families: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        key: sum(int(row[key]) for row in families)
        for key in (
            "cycles", "input_valid", "output_valid", "core_crc_errors",
            "route_action_crc_errors", "route_state_crc_errors", "route_version_crc_errors",
            "undefined_actions", "posterior_updates", "router_boundaries",
            "auto_commit_requests", "host_commit_attempts", "host_auto_commit_collisions",
            "host_commit_policy_blocks",
            "unified_replay_cycles", "directed_boundary_cycles",
            "commit_acks", "commit_rejections", "commit_cfg_races", "rollback_attempts",
            "tail_latch_entries", "tail_recoveries", "leakage_entries", "integrity_entries",
        )
    }
    result["actions"] = {name: sum(int(row["actions"][name]) for row in families) for name in ACTION_NAMES}
    result["reasons"] = {name: sum(int(row["reasons"][name]) for row in families) for name in REASON_NAMES}
    result["selected_banks"] = {name: sum(int(row["selected_banks"][name]) for row in families) for name in ("ewma", "window")}
    result["core_fault_bits"] = {str(bit): sum(int(row["core_fault_bits"][str(bit)]) for row in families) for bit in range(14)}
    result["threshold_hits"] = {name: sum(int(row["threshold_hits"][name]) for row in families) for name in families[0]["threshold_hits"]}
    result["silent_overflow"] = sum(int(row.get("transport", {}).get("silent_overflow", 0)) for row in families)
    result["pending_transport"] = sum(
        int(row.get("transport", {}).get("pending_fifo", 0)) +
        int(row.get("transport", {}).get("pending_markers", 0)) for row in families
    )
    return result


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    families = report["python_families"]
    aggregate = report["aggregate_python"]
    cxx = report["cxxrtl_families"]
    transport = families[9]["transport"]
    forbidden = ("board-measured", "measured latency", "physical transport", "measured power")
    return {
        "qualifying_scale": len(families) == 10 and min(int(row["cycles"]) for row in families) >= 100_000 and int(aggregate["cycles"]) >= 1_000_000,
        "t622_anchor": bool(report["t6_2_2_anchor"]["pass"]),
        "v4_lock_anchor": bool(report["v4_lock_anchor"]["valid"]),
        "yosys_structural_check": bool(report["toolchain"]["structural_check_zero_problems"]),
        "all_cycle_bit_exact": len(cxx) == 10 and sum(int(row["mismatches"]) for row in cxx) == 0 and all(row["actual_digest"] == row["expected_digest"] for row in cxx),
        "cxxrtl_full_scale": len(cxx) == 10 and min(int(row["rows"]) for row in cxx) >= 100_000 and sum(int(row["rows"]) for row in cxx) >= 1_000_000,
        "no_undefined_or_crc_error": int(aggregate["undefined_actions"]) == 0 and sum(int(aggregate[key]) for key in ("core_crc_errors", "route_action_crc_errors", "route_state_crc_errors", "route_version_crc_errors")) == 0 and sum(int(row["undefined_action"] + row["action_crc_errors"] + row["state_crc_errors"] + row["version_crc_errors"]) for row in cxx) == 0,
        "all_policy_actions_and_reasons": all(int(value) > 0 for value in aggregate["actions"].values()) and all(int(value) > 0 for value in aggregate["reasons"].values()),
        "posterior_boundary_coverage": all(int(value) > 0 for value in aggregate["threshold_hits"].values()),
        "unified_runner_replay_provenance": (
            int(aggregate["unified_replay_cycles"]) >= int(aggregate["cycles"]) * 95 // 100
            and report["unified_replay"]["trajectory_count"] == 20
            and report["unified_replay"]["formal_cache_hits"] == 20
            and report["unified_replay"]["formal_cache_misses"] == 0
            and len(report["unified_replay"]["families"]) == 10
            and all(len(row["segments"]) == 2 for row in report["unified_replay"]["families"])
        ),
        "hysteresis_and_router_coverage": int(aggregate["tail_latch_entries"]) > 0 and int(aggregate["tail_recoveries"]) > 0 and all(int(value) > 0 for value in aggregate["selected_banks"].values()) and int(aggregate["auto_commit_requests"]) > 0,
        "commit_rollback_race_coverage": int(aggregate["host_commit_attempts"]) > 0 and int(aggregate["host_auto_commit_collisions"]) > 0 and int(aggregate["host_commit_policy_blocks"]) > 0 and int(aggregate["commit_acks"]) > 0 and int(aggregate["commit_rejections"]) > 0 and int(aggregate["commit_cfg_races"]) > 0 and int(aggregate["rollback_attempts"]) > 0,
        "event_integrity_coverage": int(aggregate["leakage_entries"]) > 0 and int(aggregate["integrity_entries"]) > 0 and all(int(aggregate["core_fault_bits"][str(bit)]) > 0 for bit in REQUIRED_CORE_FAULT_BITS),
        "transport_fail_closed": int(aggregate["silent_overflow"]) == 0 and int(aggregate["pending_transport"]) == 0 and all(int(transport.get(key, 0)) > 0 for key in ("pause_cycles", "overflow_events", "drop_events", "duplicate_events", "reorder_events", "sequence_faults", "deadline_faults", "explicit_fault_markers")),
        "cxxrtl_comparator_mutations": sum(int(row["shadow_mutations"]) for row in cxx) == 130 and sum(int(row["shadow_mutations_detected"]) for row in cxx) == 130,
        "source_hash_binding": all(
            (ROOT / path).is_file() and _sha256(ROOT / path) == digest
            for path, digest in report["source_hashes"].items()
        ),
        "evidence_scope_lint": not any(token in report["evidence_scope"].lower() for token in forbidden),
    }


def semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    mutations: list[dict[str, Any]] = []
    def attempt(name: str, fn: Any) -> None:
        candidate = copy.deepcopy(report)
        fn(candidate)
        mutations.append({"mutation": name, "rejected": not all(evaluate_gates(candidate).values())})
    attempt("hide_rtl_mismatch", lambda x: x["cxxrtl_families"][0].update(mismatches=1))
    attempt("short_family", lambda x: x["python_families"][0].update(cycles=99_999))
    attempt("undefined_action", lambda x: x["aggregate_python"].update(undefined_actions=1))
    attempt("silent_overflow", lambda x: x["aggregate_python"].update(silent_overflow=1))
    attempt("erase_tail_recovery", lambda x: x["aggregate_python"].update(tail_recoveries=0))
    attempt("erase_commit_collision", lambda x: x["aggregate_python"].update(host_auto_commit_collisions=0))
    attempt("erase_crc_branch", lambda x: x["aggregate_python"]["core_fault_bits"].update({"2": 0}))
    attempt("erase_boundary", lambda x: x["aggregate_python"]["threshold_hits"].update({"tail_230": 0}))
    attempt("erase_unified_replay", lambda x: x["aggregate_python"].update(unified_replay_cycles=0))
    attempt("erase_version_reason", lambda x: x["aggregate_python"]["reasons"].update({"version": 0}))
    attempt("promote_board_claim", lambda x: x.update(evidence_scope="board-measured physical transport"))
    attempt("corrupt_source_hash", lambda x: x["source_hashes"].update({next(iter(x["source_hashes"])): "0" * 64}))
    return {"count": len(mutations), "detected": sum(int(row["rejected"]) for row in mutations), "mutations": mutations}


def write_source_data(path: Path, report: Mapping[str, Any]) -> None:
    cxx = {int(row["family_id"]): row for row in report["cxxrtl_families"]}
    fields = (
        "family_id", "family", "cycles", "input_valid", "output_valid", "rtl_mismatches",
        "undefined_actions", "core_crc_errors", "route_crc_errors", "posterior_updates",
        "router_boundaries", "auto_commit_requests", "host_commit_attempts",
        "host_auto_commit_collisions", "host_commit_policy_blocks", "unified_replay_cycles",
        "directed_boundary_cycles", "commit_acks", "commit_rejections",
        "tail_latch_entries", "tail_recoveries", "leakage_entries", "integrity_entries",
        "ewma_cycles", "window_cycles", "fault_bit_mask_hex", "transport_overflow_events",
        "transport_sequence_faults", "silent_overflow", "actual_digest", "expected_digest",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["python_families"]:
            rtl = cxx[int(row["family_id"])]
            fault_mask = sum((int(row["core_fault_bits"][str(bit)]) > 0) << bit for bit in range(14))
            transport = row.get("transport", {})
            writer.writerow({
                "family_id": row["family_id"], "family": row["family"], "cycles": row["cycles"],
                "input_valid": row["input_valid"], "output_valid": row["output_valid"],
                "rtl_mismatches": rtl["mismatches"], "undefined_actions": row["undefined_actions"],
                "core_crc_errors": row["core_crc_errors"],
                "route_crc_errors": row["route_action_crc_errors"] + row["route_state_crc_errors"] + row["route_version_crc_errors"],
                "posterior_updates": row["posterior_updates"], "router_boundaries": row["router_boundaries"],
                "auto_commit_requests": row["auto_commit_requests"], "host_commit_attempts": row["host_commit_attempts"],
                "host_auto_commit_collisions": row["host_auto_commit_collisions"],
                "host_commit_policy_blocks": row["host_commit_policy_blocks"],
                "unified_replay_cycles": row["unified_replay_cycles"],
                "directed_boundary_cycles": row["directed_boundary_cycles"],
                "commit_acks": row["commit_acks"], "commit_rejections": row["commit_rejections"],
                "tail_latch_entries": row["tail_latch_entries"], "tail_recoveries": row["tail_recoveries"],
                "leakage_entries": row["leakage_entries"], "integrity_entries": row["integrity_entries"],
                "ewma_cycles": row["selected_banks"]["ewma"], "window_cycles": row["selected_banks"]["window"],
                "fault_bit_mask_hex": f"0x{fault_mask:04x}",
                "transport_overflow_events": transport.get("overflow_events", 0),
                "transport_sequence_faults": transport.get("sequence_faults", 0),
                "silent_overflow": transport.get("silent_overflow", 0),
                "actual_digest": rtl["actual_digest"], "expected_digest": rtl["expected_digest"],
            })


def run_qualification(
    *, build_dir: Path = DEFAULT_BUILD, artifact_path: Path = DEFAULT_JSON,
    source_data_path: Path = DEFAULT_CSV, cycles_per_family: int = FAMILY_CYCLES,
) -> dict[str, Any]:
    if cycles_per_family < 2_000:
        raise ValueError("cycles_per_family must be >= 2000")
    build_dir.mkdir(parents=True, exist_ok=True)
    trace, families, replay_manifest, generation_seconds = generate_trace(build_dir, cycles_per_family)
    toolchain = build_cxxrtl(build_dir)
    cxx, cxx_seconds, stderr = run_cxxrtl(toolchain["executable"], toolchain["environment"], trace)
    (build_dir / "cxxrtl_stderr.log").write_text(stderr, encoding="utf-8")

    t622 = json.loads(T622.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    public_toolchain = {key: value for key, value in toolchain.items() if key not in ("executable", "environment")}
    report: dict[str, Any] = {
        "task_id": "T6.7.3", "schema_version": "t6.7.3-route-a-integrated-rtl-v1",
        "evidence_scope": "board-independent integer golden, CXXRTL and synthesis-structural qualification; abstract FIFO receiver only",
        "cycles_per_family": cycles_per_family, "family_names": list(FAMILY_NAMES),
        "trace": {"path": _relative(trace), "rows": cycles_per_family * 10,
                  "row_bytes": TRACE_STRUCT.size, "bytes": trace.stat().st_size,
                  "sha256": _sha256(trace), "role": "raw per-cycle Source Data"},
        "python_families": families, "aggregate_python": _aggregate(families),
        "unified_replay": replay_manifest,
        "cxxrtl_families": cxx, "toolchain": public_toolchain,
        "timing_seconds": {"trace_generation": generation_seconds, "yosys": toolchain["yosys_seconds"],
                           "compile": toolchain["compile_seconds"], "cxxrtl": cxx_seconds},
        "t6_2_2_anchor": {"path": _relative(T622), "sha256": _sha256(T622),
                          "verdict": t622.get("verdict"), "pass": all(t622.get("gates", {}).values())},
        "v4_lock_anchor": {"path": _relative(LOCK), "sha256": _sha256(LOCK),
                           "lock_sha256": lock.get("threshold_lock", {}).get("lock_sha256"),
                           "valid": lock.get("threshold_lock", {}).get("lock_sha256") == "9347edb270bbeb3f50d8bd8aceaeefd8003e118f1e88712dd5265519bb0f67aa"},
        "integer_contract": {"posterior_order": ["normal", "smooth", "calibration_shift", "burst"],
                             "tail_enter": "sum(tail)>=230/255", "tail_exit": "sum(tail)<=51/255",
                             "adaptive_ready": "sum(adaptive)>=230/255 and 255-max(p)<64",
                             "ood": ">192", "enter_hysteresis": 2, "recovery_hysteresis": 8,
                             "ewma_bank": 0, "window_bank": 1,
                             "slow_loop_boundary": "posterior inference remains software; only quantized observed-only contract crosses FPGA boundary"},
        "source_hashes": {_relative(path): _sha256(path) for path in (
            RUNNER, CORE, POLICY, TOP, DRIVER, REFERENCE,
            SMOOTH_FORMAL_RUNNER, TAIL_FORMAL_RUNNER, POSTERIOR_MODEL,
        )},
        "transport_contract": "bounded FIFO/receiver disturbance abstraction inherited from T6.2.2; no CDC, pins or physical link claim",
    }
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = semantic_mutation_audit(report)
    report["gates"]["semantic_mutations"] = report["semantic_mutation_audit"]["detected"] == report["semantic_mutation_audit"]["count"] == 12
    report["verdict"] = VERDICT if all(report["gates"].values()) else ("NON_QUALIFYING_SHORT_RUN" if cycles_per_family < FAMILY_CYCLES else "FAIL_ROUTE_A_INTEGRATED_LONG_RTL_QUALIFICATION")
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
    report = run_qualification(build_dir=args.build_dir, artifact_path=args.artifact,
                               source_data_path=args.source_data, cycles_per_family=args.cycles_per_family)
    print(json.dumps({"verdict": report["verdict"], "cycles": report["aggregate_python"]["cycles"],
                      "mismatches": sum(row["mismatches"] for row in report["cxxrtl_families"]),
                      "gates": report["gates"], "artifact": _relative(args.artifact)}, indent=2))
    return 0 if all(report["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
