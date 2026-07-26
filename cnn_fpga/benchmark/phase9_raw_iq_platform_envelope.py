"""Generate and independently verify the T9.2.6 raw-IQ envelope.

The resulting PASS is a protocol/interface PASS only.  The T9.2.4 scientific
NO-GO remains immutable and all frontend-performance, board-measurement,
physics, Puviani, SOTA, and rank fields remain JSON null.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cnn_fpga.hwio import phase9_raw_iq_stream_contract as runtime


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T9.2.6"
PROTOCOL_ID = "PHASE9-RAW-IQ-PLATFORM-ENVELOPE-V1"
REPORT_SCHEMA = "t9.2.6-raw-iq-platform-envelope-report-v1"
RELEASE_SCHEMA = "t9.2.6-raw-iq-platform-envelope-release-pin-v1"
VERDICT = "PASS_T9_2_6_RAW_IQ_PLATFORM_ENVELOPE_FROZEN"

DEFAULT_CONFIG = ROOT / "configs/phase9/t9_2_6_raw_iq_platform_envelope.json"
DEFAULT_REPORT = ROOT / "docs/t9_2_6_raw_iq_platform_envelope.json"
DEFAULT_SOURCE_DATA = (
    ROOT / "docs/t9_2_6_raw_iq_platform_envelope_source_data.csv"
)
DEFAULT_MARKDOWN = ROOT / "docs/phase9_raw_iq_platform_envelope.md"
DEFAULT_RELEASE_PIN = ROOT / "configs/phase9/t9_2_6_release_pin.json"
IMPLEMENTATION = (
    ROOT / "cnn_fpga/benchmark/phase9_raw_iq_platform_envelope.py"
)
RUNTIME = ROOT / "cnn_fpga/hwio/phase9_raw_iq_stream_contract.py"
TEST_FILE = ROOT / "tests/test_phase9_raw_iq_platform_envelope.py"
CORE_RTL = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
LEGACY_REFERENCE = (
    ROOT / "cnn_fpga/runtime/bit_accurate_hardware_reference.py"
)

GATE_IDS = (
    "G01_identity_scope_and_verdict_exact",
    "G02_all_parent_analysis_hashes_are_live_and_exact",
    "G03_t9_2_4_scientific_no_go_and_only_t9_2_6_release_preserved",
    "G04_config_generator_runtime_rtl_reference_and_tests_are_byte_bound",
    "G05_project_boundary_excludes_vendor_rfdc_and_host_transport",
    "G06_exact_two_512ns_rate_profiles_match_runtime",
    "G07_tdata_is_exact_contiguous_i16_q16_q1_15",
    "G08_tuser_is_exact_contiguous_128bit_metadata",
    "G09_domain_ids_are_typed_and_invalid_domain_is_reserved",
    "G10_axis_handshake_stability_tlast_and_packet_semantics_are_exact",
    "G11_fixed_point_widths_and_q_formats_are_exact",
    "G12_ties_even_saturation_and_no_wrap_are_executable",
    "G13_matched_filter_calibration_and_sticky_overflow_semantics_are_frozen",
    "G14_threshold_values_are_typed_null_and_blocked_by_twin_no_go",
    "G15_complete_atomic_ab_package_version_crc_and_lkg_are_frozen",
    "G16_cdc_fifo_depth_reset_and_whole_window_poison_are_frozen",
    "G17_error_priority_matches_executable_runtime_exactly",
    "G18_no_postselection_or_silent_drop_is_allowed",
    "G19_legacy_58bit_layout_matches_live_core_and_python_schema",
    "G20_legacy_adapter_crc_and_field_roundtrip_are_executable",
    "G21_exactly_four_latency_boundaries_are_noninterchangeable",
    "G22_six_cycle_ii1_is_limited_to_fast_and_discriminator_boundaries",
    "G23_adc_and_raw_source_latency_values_remain_typed_null",
    "G24_trigger_pin_electrical_and_measured_values_remain_null",
    "G25_zcu111_and_zcu216_are_candidates_not_selected_or_built",
    "G26_gw2ar_is_explicitly_excluded_from_raw_iq_platform_intersection",
    "G27_platform_resource_numbers_are_design_budgets_not_results",
    "G28_all_rate_domain_nominal_windows_execute_without_sampling",
    "G29_adversarial_fault_and_protocol_matrix_fails_closed",
    "G30_iq_tuser_and_rounding_boundary_enumerations_are_complete",
    "G31_package_crc_commit_rollback_and_midwindow_rejection_are_executable",
    "G32_all_performance_physical_hardware_puviani_sota_rank_fields_are_null",
    "G33_source_data_losslessly_reconstructs_frozen_snapshot",
    "G34_markdown_and_canonical_output_paths_are_exact",
    "G35_one_targeted_mutation_per_gate_is_detected",
    "G36_downstream_releases_only_fresh_twin_repair_next",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _binding_live(binding: Mapping[str, Any]) -> bool:
    try:
        path = (ROOT / str(binding["path"])).resolve()
        path.relative_to(ROOT.resolve())
        return (
            path.is_file()
            and path.stat().st_size == binding["bytes"]
            and _sha256(path) == binding["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        path,
    )


def _safe(callback: Callable[[], Any]) -> bool:
    try:
        return bool(callback())
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def _exact_layout(
    rows: Sequence[Mapping[str, Any]], *, total_bits: int
) -> bool:
    occupied: set[int] = set()
    for row in rows:
        lsb = row["lsb"]
        bits = row["bits"]
        if (
            isinstance(lsb, bool)
            or not isinstance(lsb, int)
            or isinstance(bits, bool)
            or not isinstance(bits, int)
            or lsb < 0
            or bits <= 0
        ):
            return False
        field = set(range(lsb, lsb + bits))
        if occupied & field:
            return False
        occupied |= field
    return occupied == set(range(total_bits))


def _parent_bindings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for expected in config["parents"]:
        path = ROOT / expected["path"]
        report = _load(path)
        if report.get("analysis_sha256") != expected["analysis_sha256"]:
            raise ValueError(f"parent analysis mismatch: {expected['task_id']}")
        if "required_verdict" in expected and report.get(
            "verdict"
        ) != expected["required_verdict"]:
            raise ValueError(f"parent verdict mismatch: {expected['task_id']}")
        output.append(
            {
                **expected,
                "artifact": _binding(path),
                "live_verdict": report.get("verdict"),
            }
        )
    return output


def _set_error_flag(
    cycle: runtime.AxisCycle, flag_name: str
) -> runtime.AxisCycle:
    metadata = runtime.StreamMetadata.unpack(cycle.tuser)
    mutated = dataclasses_replace(
        metadata,
        error_flags=metadata.error_flags
        | (1 << runtime.ERROR_FLAG_BITS[flag_name]),
    )
    return dataclasses_replace(cycle, tuser=mutated.pack())


def dataclasses_replace(value: Any, **changes: Any) -> Any:
    # Local wrapper keeps mutation tests independent of object internals.
    import dataclasses

    return dataclasses.replace(value, **changes)


def _expected_flag_reason(flag_name: str) -> runtime.FaultReason:
    if flag_name == "reset_mid_window":
        return runtime.FaultReason.RESET_MID_WINDOW
    if flag_name == "cdc_overflow":
        return runtime.FaultReason.CDC_OVERFLOW
    if flag_name in ("rfdc_overflow", "source_clock_unlock"):
        return runtime.FaultReason.SOURCE_OR_RFDC_FAILURE
    if flag_name in (
        "coefficient_crc_failure",
        "calibration_crc_failure",
    ):
        return runtime.FaultReason.PACKAGE_INTEGRITY_FAILURE
    if flag_name == "config_version_stale":
        return runtime.FaultReason.CONFIG_VERSION_FAILURE
    if flag_name == "timestamp_regression":
        return runtime.FaultReason.TIMESTAMP_FAILURE
    if flag_name in ("index_gap", "index_duplicate", "index_reorder"):
        return runtime.FaultReason.SAMPLE_SEQUENCE_FAILURE
    if flag_name in ("tlast_early", "tlast_missing", "length_mismatch"):
        return runtime.FaultReason.WINDOW_FRAMING_FAILURE
    return runtime.FaultReason.INPUT_QUALITY_FAILURE


def _executable_audit() -> dict[str, Any]:
    iq_roundtrips = 0
    for value in range(-(1 << 15), 1 << 15):
        packed_i = runtime.pack_iq_tdata(value, 0)
        packed_q = runtime.pack_iq_tdata(0, value)
        if runtime.unpack_iq_tdata(packed_i) != (value, 0):
            raise AssertionError("I code roundtrip failure")
        if runtime.unpack_iq_tdata(packed_q) != (0, value):
            raise AssertionError("Q code roundtrip failure")
        iq_roundtrips += 2

    tuser_cases = 0
    base = {
        "timestamp": 0,
        "window_id": 0,
        "sample_index": 0,
        "channel_id": 0,
        "rate_id": 0,
        "domain_id": 0,
        "config_version": 0,
        "error_flags": 0,
        "reset_epoch": 0,
    }
    for name, _lsb, bits in runtime.TUSER_FIELDS:
        for value in (0, (1 << bits) - 1):
            row = dict(base)
            row[name] = value
            metadata = runtime.StreamMetadata(**row)
            if runtime.StreamMetadata.unpack(metadata.pack()) != metadata:
                raise AssertionError(f"TUSER roundtrip failure: {name}")
            tuser_cases += 1

    nominal_cases = []
    for rate_id in runtime.RateId:
        for domain_id in (
            runtime.DomainId.SYNTHETIC,
            runtime.DomainId.RECORDED_REPLAY,
            runtime.DomainId.LIVE_RAW,
        ):
            window = runtime.build_window(
                rate_id,
                domain_id=domain_id,
                start_timestamp=17,
                window_id=23,
                config_version=9,
            )
            result = runtime.validate_axis_cycles(
                window, minimum_config_version=9
            )
            if not result.accepted:
                raise AssertionError(
                    f"nominal rate/domain failed: {rate_id}/{domain_id}"
                )
            nominal_cases.append(
                {
                    "rate_id": int(rate_id),
                    "domain_id": int(domain_id),
                    "samples": len(window),
                    "reason": result.reason.name,
                }
            )

    fault_cases = []
    fail_closed_adapter_cases = 0
    for flag_name in runtime.ERROR_FLAG_BITS:
        window = runtime.build_window(runtime.RateId.IQ_125_MSPS)
        window[0] = _set_error_flag(window[0], flag_name)
        result = runtime.validate_axis_cycles(window)
        expected = _expected_flag_reason(flag_name)
        if result.reason is not expected:
            raise AssertionError(
                f"{flag_name}: {result.reason.name} != {expected.name}"
            )
        fault_cases.append(
            {
                "fault": flag_name,
                "reason": result.reason.name,
                "accepted": result.accepted,
            }
        )
        candidate = runtime.FastPathObservation(
            syndrome_code=1023,
            syndrome_x="leakage",
            syndrome_z="leakage",
            quadrature_phase_bit=1,
            ood_score_code=0,
            parameter_age_code=7,
        )
        closed = runtime.fail_closed_fast_path_observation(candidate, result)
        decoded = runtime.decode_input_word(
            closed.pack_legacy_58bit_word()
        )
        if decoded.observation_valid or decoded.deadline_ok:
            raise AssertionError("faulted frontend decision survived adapter")
        fail_closed_adapter_cases += 1

    structural_cases = []
    base_window = runtime.build_window(runtime.RateId.IQ_125_MSPS)
    variants: dict[str, list[runtime.AxisCycle]] = {
        "empty": [],
        "short": base_window[:-1],
        "early_tlast": [
            dataclasses_replace(row, tlast=index == 3)
            for index, row in enumerate(base_window)
        ],
        "mixed_version": [
            (
                dataclasses_replace(
                    row,
                    tuser=dataclasses_replace(
                        runtime.StreamMetadata.unpack(row.tuser),
                        config_version=2,
                    ).pack(),
                )
                if index == 4
                else row
            )
            for index, row in enumerate(base_window)
        ],
        "timestamp_gap": [
            (
                dataclasses_replace(
                    row,
                    tuser=dataclasses_replace(
                        runtime.StreamMetadata.unpack(row.tuser),
                        timestamp=999,
                    ).pack(),
                )
                if index == 4
                else row
            )
            for index, row in enumerate(base_window)
        ],
        "index_duplicate": [
            (
                dataclasses_replace(
                    row,
                    tuser=dataclasses_replace(
                        runtime.StreamMetadata.unpack(row.tuser),
                        sample_index=3,
                    ).pack(),
                )
                if index == 4
                else row
            )
            for index, row in enumerate(base_window)
        ],
        "invalid_domain": [
            dataclasses_replace(
                row,
                tuser=dataclasses_replace(
                    runtime.StreamMetadata.unpack(row.tuser),
                    domain_id=3,
                ).pack(),
            )
            for row in base_window
        ],
        "nonzero_channel": [
            dataclasses_replace(
                row,
                tuser=dataclasses_replace(
                    runtime.StreamMetadata.unpack(row.tuser),
                    channel_id=1,
                ).pack(),
            )
            for row in base_window
        ],
    }
    for case_id, candidate in variants.items():
        result = runtime.validate_axis_cycles(candidate)
        if result.accepted:
            raise AssertionError(f"structural case accepted: {case_id}")
        structural_cases.append(
            {"case_id": case_id, "reason": result.reason.name}
        )

    stalled = runtime.build_window(runtime.RateId.IQ_125_MSPS)
    first = stalled[0]
    good_stall = [
        dataclasses_replace(first, tready=False),
        first,
        *stalled[1:],
    ]
    if not runtime.validate_axis_cycles(good_stall).accepted:
        raise AssertionError("stable backpressure sequence failed")
    bad_stall = [
        dataclasses_replace(first, tready=False),
        dataclasses_replace(stalled[1], tready=True),
        *stalled[2:],
    ]
    bad_result = runtime.validate_axis_cycles(bad_stall)
    if bad_result.reason is not runtime.FaultReason.AXIS_STABILITY_FAILURE:
        raise AssertionError("unstable backpressure did not fail closed")

    first_sequence = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        start_timestamp=0,
        window_id=0,
        config_version=1,
        reset_epoch=0,
    )
    sequence_result, sequence_state = runtime.validate_and_retire_sequence(
        first_sequence, runtime.IngressSequenceState()
    )
    if not sequence_result.accepted:
        raise AssertionError("first freshness window failed")
    replay_result, replay_state = runtime.validate_and_retire_sequence(
        first_sequence, sequence_state
    )
    if (
        replay_result.reason
        is not runtime.FaultReason.FRESHNESS_REPLAY_FAILURE
        or replay_state != sequence_state
    ):
        raise AssertionError("cross-window replay was not rejected")
    next_sequence = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        start_timestamp=200,
        window_id=1,
        config_version=2,
        reset_epoch=0,
    )
    next_result, next_state = runtime.validate_and_retire_sequence(
        next_sequence, sequence_state
    )
    if not next_result.accepted:
        raise AssertionError("monotonic next window failed")
    reset_sequence = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        start_timestamp=0,
        window_id=0,
        config_version=2,
        reset_epoch=1,
    )
    reset_result, reset_state = runtime.validate_and_retire_sequence(
        reset_sequence, next_state
    )
    if not reset_result.accepted:
        raise AssertionError("explicit reset epoch failed")
    stale_epoch_result, stale_epoch_state = (
        runtime.validate_and_retire_sequence(next_sequence, reset_state)
    )
    if (
        stale_epoch_result.reason
        is not runtime.FaultReason.FRESHNESS_REPLAY_FAILURE
        or stale_epoch_state != reset_state
    ):
        raise AssertionError("stale pre-reset epoch was not rejected")
    mixed_epoch = list(reset_sequence)
    mixed_meta = runtime.StreamMetadata.unpack(mixed_epoch[20].tuser)
    mixed_epoch[20] = dataclasses_replace(
        mixed_epoch[20],
        tuser=dataclasses_replace(mixed_meta, reset_epoch=0).pack(),
    )
    if (
        runtime.validate_axis_cycles(mixed_epoch).reason
        is not runtime.FaultReason.INVALID_METADATA
    ):
        raise AssertionError("mixed reset epoch within a window was accepted")
    timeout_cycles = [
        next_sequence[0],
        *[
            runtime.AxisCycle(tvalid=False, tready=True)
            for _ in range(runtime.WINDOW_DEADLINE_AXIS_CYCLES)
        ],
    ]
    timeout_result, timeout_state = runtime.validate_and_retire_sequence(
        timeout_cycles, sequence_state
    )
    if (
        timeout_result.reason
        is not runtime.FaultReason.WINDOW_TIMEOUT_FAILURE
        or not timeout_state.quarantined
        or timeout_state.poisoned_window_count != 1
    ):
        raise AssertionError("missing TLAST did not poison/quarantine once")
    retry_result, retry_state = runtime.validate_and_retire_sequence(
        next_sequence, timeout_state
    )
    if (
        retry_result.reason
        is not runtime.FaultReason.QUARANTINE_ACTIVE_FAILURE
        or retry_state != timeout_state
    ):
        raise AssertionError("timed-out window retry bypassed quarantine")
    split_predicate_candidate = [
        next_sequence[1],
        dataclasses_replace(next_sequence[0], tlast=True),
    ]
    split_result, split_state = runtime.validate_and_retire_sequence(
        split_predicate_candidate, retry_state
    )
    if (
        split_result.reason
        is not runtime.FaultReason.QUARANTINE_ACTIVE_FAILURE
        or split_state != retry_state
    ):
        raise AssertionError("drain predicate was spliced across two beats")
    drain_result, drained_state = runtime.validate_and_retire_sequence(
        [next_sequence[-1]], split_state
    )
    if (
        drain_result.reason
        is not runtime.FaultReason.QUARANTINE_ACTIVE_FAILURE
        or drained_state.quarantined
        or drained_state.poisoned_window_count != 1
    ):
        raise AssertionError("matching TLAST did not drain quarantine safely")
    accepted_after_drain, post_drain_state = (
        runtime.validate_and_retire_sequence(next_sequence, drained_state)
    )
    if (
        not accepted_after_drain.accepted
        or post_drain_state.poisoned_window_count != 1
    ):
        raise AssertionError("clean retry after drain failed or double-counted poison")
    quarantine_reset_sequence = runtime.build_window(
        runtime.RateId.IQ_125_MSPS,
        start_timestamp=0,
        window_id=0,
        config_version=1,
        reset_epoch=1,
    )
    bad_reset_candidates: list[
        tuple[list[runtime.AxisCycle], runtime.FaultReason]
    ] = [
        (
            [quarantine_reset_sequence[0]],
            runtime.FaultReason.WINDOW_FRAMING_FAILURE,
        ),
        (
            runtime.build_window(
                runtime.RateId.IQ_125_MSPS,
                start_timestamp=0,
                window_id=0,
                config_version=2,
                reset_epoch=1,
            ),
            runtime.FaultReason.FRESHNESS_REPLAY_FAILURE,
        ),
    ]
    mixed_reset_candidate = list(quarantine_reset_sequence)
    mixed_reset_metadata = runtime.StreamMetadata.unpack(
        mixed_reset_candidate[20].tuser
    )
    mixed_reset_candidate[20] = dataclasses_replace(
        mixed_reset_candidate[20],
        tuser=dataclasses_replace(
            mixed_reset_metadata, reset_epoch=0
        ).pack(),
    )
    bad_reset_candidates.append(
        (mixed_reset_candidate, runtime.FaultReason.INVALID_METADATA)
    )
    bad_tlast_reset = list(quarantine_reset_sequence)
    bad_tlast_reset[-1] = dataclasses_replace(
        bad_tlast_reset[-1], tlast=False
    )
    bad_reset_candidates.append(
        (bad_tlast_reset, runtime.FaultReason.WINDOW_FRAMING_FAILURE)
    )
    for candidate, expected_reason in bad_reset_candidates:
        bad_reset_result, bad_reset_state = (
            runtime.validate_and_retire_sequence(candidate, timeout_state)
        )
        if (
            bad_reset_result.reason is not expected_reason
            or bad_reset_state != timeout_state
            or bad_reset_state.poisoned_window_count != 1
        ):
            raise AssertionError(
                "invalid reset candidate changed quarantine transactionally"
            )
    reset_flush_result, reset_flush_state = (
        runtime.validate_and_retire_sequence(
            quarantine_reset_sequence, timeout_state
        )
    )
    if (
        not reset_flush_result.accepted
        or reset_flush_state.quarantined
        or reset_flush_state.poisoned_window_count != 1
        or reset_flush_state.reset_epoch != 1
        or reset_flush_state.last_window_id != 0
    ):
        raise AssertionError("valid reset did not atomically clear and retire")

    rounding_cases = 0
    for value in range(-131_072, 131_072):
        actual = runtime.round_shift_ties_to_even_saturate(
            value, shift=4, output_bits=8
        )
        expected = min(127, max(-128, round(value / 16)))
        if actual != expected:
            raise AssertionError(f"rounding mismatch at {value}")
        rounding_cases += 1

    matched_filter_cases = 0
    lcg = 0x9265A17D
    for profile_id, profile in runtime.RATE_PROFILES.items():
        extreme_vectors = (
            ((32767, 32767), (131071, 131071)),
            ((-32768, -32768), (-131072, -131072)),
        )
        for sample, coefficient in extreme_vectors:
            samples = [sample] * profile.integration_samples
            coefficient_i = [coefficient[0]] * profile.integration_samples
            coefficient_q = [coefficient[1]] * profile.integration_samples
            actual = runtime.matched_filter_accumulate(
                profile_id, samples, coefficient_i, coefficient_q
            )
            expected_i = sum(
                i_code * h_i + q_code * h_q
                for (i_code, q_code), h_i, h_q in zip(
                    samples, coefficient_i, coefficient_q
                )
            )
            expected_q = sum(
                q_code * h_i - i_code * h_q
                for (i_code, q_code), h_i, h_q in zip(
                    samples, coefficient_i, coefficient_q
                )
            )
            if (
                actual.accumulator_i_q16_32 != expected_i
                or actual.accumulator_q_q16_32 != expected_q
                or actual.sticky_overflow
            ):
                raise AssertionError("extreme matched-filter arithmetic mismatch")
            matched_filter_cases += 1
    profile = runtime.RATE_PROFILES[runtime.RateId.IQ_250_MSPS]
    for _case_index in range(512):
        samples = []
        coefficient_i = []
        coefficient_q = []
        for _ in range(profile.integration_samples):
            lcg = (1664525 * lcg + 1013904223) & 0xFFFFFFFF
            i_code = runtime._twos_to_signed(lcg, 16)
            lcg = (1664525 * lcg + 1013904223) & 0xFFFFFFFF
            q_code = runtime._twos_to_signed(lcg, 16)
            lcg = (1664525 * lcg + 1013904223) & 0xFFFFFFFF
            h_i = runtime._twos_to_signed(lcg, 18)
            lcg = (1664525 * lcg + 1013904223) & 0xFFFFFFFF
            h_q = runtime._twos_to_signed(lcg, 18)
            samples.append((i_code, q_code))
            coefficient_i.append(h_i)
            coefficient_q.append(h_q)
        actual = runtime.matched_filter_accumulate(
            runtime.RateId.IQ_250_MSPS,
            samples,
            coefficient_i,
            coefficient_q,
        )
        expected_i = sum(
            i_code * h_i + q_code * h_q
            for (i_code, q_code), h_i, h_q in zip(
                samples, coefficient_i, coefficient_q
            )
        )
        expected_q = sum(
            q_code * h_i - i_code * h_q
            for (i_code, q_code), h_i, h_q in zip(
                samples, coefficient_i, coefficient_q
            )
        )
        if (
            actual.accumulator_i_q16_32 != expected_i
            or actual.accumulator_q_q16_32 != expected_q
            or actual.sticky_overflow
        ):
            raise AssertionError("random matched-filter arithmetic mismatch")
        matched_filter_cases += 1

    def independent_round(value: int, shift: int) -> int:
        magnitude = abs(value)
        quotient, remainder = divmod(magnitude, 1 << shift)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and quotient & 1):
            quotient += 1
        return -quotient if value < 0 else quotient

    calibration_cases = 0
    values48 = (-(1 << 47), 0, (1 << 47) - 1)
    values18 = (-(1 << 17), 0, (1 << 17) - 1)
    import itertools

    for acc_i, acc_q, m00, m01, m10, m11 in itertools.product(
        values48, values48, values18, values18, values18, values18
    ):
        actual = runtime.calibrate_accumulators_to_q8_16(
            acc_i,
            acc_q,
            (m00, m01, m10, m11),
            (0, 0),
        )
        raw_i = acc_i * m00 + acc_q * m01
        raw_q = acc_i * m10 + acc_q * m11
        expected_i_unbounded = independent_round(raw_i, 32)
        expected_q_unbounded = independent_round(raw_q, 32)
        expected_i = min((1 << 23) - 1, max(-(1 << 23), expected_i_unbounded))
        expected_q = min((1 << 23) - 1, max(-(1 << 23), expected_q_unbounded))
        expected_overflow = (
            expected_i != expected_i_unbounded
            or expected_q != expected_q_unbounded
        )
        if (
            actual.i_q8_16 != expected_i
            or actual.q_q8_16 != expected_q
            or actual.sticky_overflow != expected_overflow
            or actual.intermediate_i_q19_48 != raw_i
            or actual.intermediate_q_q19_48 != raw_q
        ):
            raise AssertionError("calibration arithmetic mismatch")
        calibration_cases += 1

    fast_roundtrips = 0
    for syndrome_x in ("g", "e", "leakage"):
        for syndrome_z in ("g", "e", "leakage"):
            for phase in (0, 1):
                observation = runtime.FastPathObservation(
                    syndrome_code=1023,
                    syndrome_x=syndrome_x,
                    syndrome_z=syndrome_z,
                    quadrature_phase_bit=phase,
                    ood_score_code=255,
                    parameter_age_code=65535,
                )
                if not runtime.verify_fast_path_roundtrip(observation):
                    raise AssertionError("legacy fast-path roundtrip failure")
                fast_roundtrips += 1

    package_cases = []
    blocked_packages: dict[runtime.RateId, runtime.FrontendBankPackage] = {}
    trusted_fixture_packages: dict[
        runtime.RateId, runtime.FrontendBankPackage
    ] = {}
    for profile_id, profile in runtime.RATE_PROFILES.items():
        package = runtime.FrontendBankPackage(
            schema_version=1,
            config_version=1,
            activation_window=2,
            profile_id=int(profile_id),
            coefficient_i_q1_17=(131071,)
            + (0,) * (profile.integration_samples - 1),
            coefficient_q_q1_17=(0,) * profile.integration_samples,
            calibration_matrix_q2_16=(65536, 0, 0, 65536),
            calibration_offset_q8_16=(0, 0),
            discriminator_thresholds_q8_16=(0, 0, 0, 0),
            threshold_qualification_state=0,
        )
        blocked_packages[profile_id] = package
        payload = package.payload_bytes()
        crc = package.crc32()
        tampered = bytearray(payload)
        tampered[-1] ^= 1
        if crc == (zlib_crc32(bytes(tampered))):
            raise AssertionError("package CRC did not detect a bit flip")
        package_cases.append(
            {
                "profile_id": int(profile_id),
                "payload_bytes": len(payload),
                "crc32": crc,
                "state": "BLOCKED_UNQUALIFIED_NOT_ACTIVATABLE",
            }
        )
        trusted_fixture_packages[profile_id] = dataclasses_replace(
            package,
            threshold_qualification_state=1,
            qualification_receipt_sha256="1" * 64,
        )

    state = runtime.BankCommitState()
    fixture_package = trusted_fixture_packages[runtime.RateId.IQ_125_MSPS]
    preloaded_state = runtime.BankCommitState(
        active_bank=1,
        active_version=1,
        lkg_bank=0,
        lkg_version=0,
        active_package_sha256=fixture_package.sha256(),
        active_profile_id=int(runtime.RateId.IQ_125_MSPS),
    )
    rolled_back = preloaded_state.rollback_lkg()
    if rolled_back.active_bank != 0 or rolled_back.active_version != 0:
        raise AssertionError("LKG rollback failure")
    rejected_commit_cases = 0
    wrong_version_package = dataclasses_replace(
        fixture_package, config_version=2
    )
    commit_mutations = (
        {
            "requested_bank": 0,
            "package": fixture_package,
            "presented_crc32": fixture_package.crc32(),
            "next_window_id": 2,
        },
        {
            "requested_bank": 1,
            "package": fixture_package,
            "presented_crc32": fixture_package.crc32() ^ 1,
            "next_window_id": 2,
        },
        {
            "requested_bank": 1,
            "package": fixture_package,
            "presented_crc32": fixture_package.crc32(),
            "next_window_id": 3,
        },
        {
            "requested_bank": 1,
            "package": blocked_packages[runtime.RateId.IQ_125_MSPS],
            "presented_crc32": blocked_packages[
                runtime.RateId.IQ_125_MSPS
            ].crc32(),
            "next_window_id": 2,
        },
        {
            "requested_bank": 1,
            "package": wrong_version_package,
            "presented_crc32": wrong_version_package.crc32(),
            "next_window_id": 2,
        },
        {
            "requested_bank": True,
            "package": fixture_package,
            "presented_crc32": fixture_package.crc32(),
            "next_window_id": 2,
        },
        {
            "requested_bank": 1,
            "package": fixture_package,
            "presented_crc32": fixture_package.crc32(),
            "next_window_id": 2,
        },
    )
    for arguments in commit_mutations:
        try:
            state.commit(**arguments)
        except ValueError:
            rejected_commit_cases += 1
        else:
            raise AssertionError("unsafe bank commit was accepted")
    try:
        dataclasses_replace(state, window_open=True).commit(
            requested_bank=1,
            package=fixture_package,
            presented_crc32=fixture_package.crc32(),
            next_window_id=2,
        )
    except ValueError:
        rejected_commit_cases += 1
    else:
        raise AssertionError("mid-window bank commit was accepted")
    try:
        runtime.BankCommitState(
            active_bank=0,
            active_version=0xFFFF,
            lkg_bank=1,
            lkg_version=0xFFFE,
        ).commit(
            requested_bank=1,
            package=fixture_package,
            presented_crc32=fixture_package.crc32(),
            next_window_id=2,
        )
    except ValueError:
        rejected_commit_cases += 1
    else:
        raise AssertionError("16-bit version wrap was accepted")

    strict_bool_alias_cases = 0
    for field_name in (
        "schema_version",
        "config_version",
        "activation_window",
        "profile_id",
        "threshold_qualification_state",
    ):
        try:
            dataclasses_replace(
                fixture_package, **{field_name: True}
            ).payload_bytes()
        except (TypeError, ValueError):
            strict_bool_alias_cases += 1
        else:
            raise AssertionError(f"bool alias accepted for package {field_name}")
    for field_name in (
        "active_bank",
        "active_version",
        "lkg_bank",
        "lkg_version",
        "active_profile_id",
        "lkg_profile_id",
    ):
        try:
            runtime.BankCommitState(**{field_name: True})
        except (TypeError, ValueError):
            strict_bool_alias_cases += 1
        else:
            raise AssertionError(f"bool alias accepted for bank state {field_name}")
    for arguments in ({"tdata": True}, {"tuser": True}):
        try:
            runtime.AxisCycle(**arguments)
        except (TypeError, ValueError):
            strict_bool_alias_cases += 1
        else:
            raise AssertionError("bool alias accepted for AXI integer field")
    for arguments in (
        {"profile_id": True},
        {
            "profile_id": runtime.RateId.IQ_125_MSPS,
            "domain_id": True,
        },
    ):
        try:
            runtime.build_window(**arguments)
        except (TypeError, ValueError):
            strict_bool_alias_cases += 1
        else:
            raise AssertionError("bool alias accepted for rate/domain identifier")
    try:
        runtime.matched_filter_accumulate(True, [], [], [])
    except (TypeError, ValueError):
        strict_bool_alias_cases += 1
    else:
        raise AssertionError("bool alias accepted for matched-filter profile")

    return {
        "iq_signed_code_roundtrips": iq_roundtrips,
        "tuser_boundary_roundtrips": tuser_cases,
        "nominal_rate_domain_cases": nominal_cases,
        "fault_flag_cases": fault_cases,
        "fail_closed_adapter_cases": fail_closed_adapter_cases,
        "structural_failure_cases": structural_cases,
        "stable_backpressure_accepts": True,
        "unstable_backpressure_rejected": True,
        "window_timeout_rejected": True,
        "timeout_quarantine_cases": 10,
        "timeout_poison_count": post_drain_state.poisoned_window_count,
        "stateful_freshness_cases": 5,
        "mixed_reset_epoch_rejected": True,
        "rounding_exhaustive_cases": rounding_cases,
        "matched_filter_reference_cases": matched_filter_cases,
        "calibration_reference_cases": calibration_cases,
        "legacy_fast_path_roundtrips": fast_roundtrips,
        "package_cases": package_cases,
        "unsafe_commit_cases_rejected": rejected_commit_cases,
        "successful_commit_count": 0,
        "trusted_qualification_receipt_sha256": (
            runtime.TRUSTED_QUALIFICATION_RECEIPT_SHA256
        ),
        "trusted_commit_fixture_scope": (
            "CURRENT_ACTIVATION_CLOSED_PENDING_FRESH_TWIN_PASS"
        ),
        "strict_bool_alias_cases_rejected": strict_bool_alias_cases,
        "all_passed": True,
    }


def zlib_crc32(payload: bytes) -> int:
    import zlib

    return zlib.crc32(payload) & 0xFFFFFFFF


def _snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parents": report["parents"],
        "implementation_bindings": report["implementation_bindings"],
        "contract": report["contract"],
        "executable_audit": report["executable_audit"],
        "claim_state": report["claim_state"],
        "downstream_release": report["downstream_release"],
    }


def _write_source_data(
    snapshot: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("section", "canonical_json", "sha256")
        )
        writer.writeheader()
        for section in sorted(snapshot):
            canonical = _canonical_json(snapshot[section])
            writer.writerow(
                {
                    "section": section,
                    "canonical_json": canonical,
                    "sha256": hashlib.sha256(
                        canonical.encode("utf-8")
                    ).hexdigest(),
                }
            )
    os.replace(temporary, path)
    return _binding(path)


def _read_source_data(path: Path) -> dict[str, Any]:
    output = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            canonical = row["canonical_json"]
            if (
                hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                != row["sha256"]
            ):
                raise ValueError(f"source row hash mismatch: {row['section']}")
            if row["section"] in output:
                raise ValueError("duplicate source section")
            parsed = json.loads(canonical)
            if canonical != _canonical_json(parsed):
                raise ValueError(
                    f"source row is not canonical JSON: {row['section']}"
                )
            output[row["section"]] = parsed
    return output


def _legacy_fields_from_config(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: row[key] for key in ("field", "lsb", "bits")}
        for row in contract["legacy_fast_path_adapter"]["fields"][:-1]
    ]


def _runtime_legacy_fields() -> list[dict[str, Any]]:
    return [
        {
            "field": {
                "syndrome_code": "syndrome_code",
                "syndrome_x_code": "syndrome_x_code",
                "syndrome_z_code": "syndrome_z_code",
                "quadrature_phase_bit": "quadrature_phase_bit",
                "ood_score_code": "ood_score_code",
                "parameter_age_code": "parameter_age_code",
                "reset_ack": "reset_ack",
                "observation_valid": "observation_valid",
                "deadline_ok": "deadline_ok",
            }[row["name"]],
            "lsb": row["offset"],
            "bits": row["width"],
        }
        for row in runtime.legacy_fast_path_layout()["fields"]
    ]


def _check_gates(
    report: Mapping[str, Any],
    *,
    verify_live: bool,
    verify_outputs: bool,
) -> dict[str, bool]:
    contract = report["contract"]
    stream = contract["stream_contract"]
    rates = contract["rate_profiles"]
    fixed = contract["fixed_point_contract"]
    latency = contract["latency_boundaries"]
    claims = report["claim_state"]
    downstream = report["downstream_release"]

    expected_rates = [
        {
            "rate_id": int(rate_id),
            "name": profile.rate_id.name,
            "sample_rate_hz": profile.sample_rate_hz,
            "axis_cycles_per_sample": profile.axis_cycles_per_sample,
            "integration_samples": profile.integration_samples,
            "integration_ns": profile.integration_ns,
        }
        for rate_id, profile in runtime.RATE_PROFILES.items()
    ]
    expected_priority = [reason.name for reason in runtime.FAULT_PRIORITY]
    fast_boundaries = {
        "FAST_PATH_CORE_INPUT_TO_ACTION",
        "DISCRIMINATOR_OUT_TO_ACTION",
    }
    null_boundaries = {
        "ADC_LAST_SAMPLE_TO_TRIGGER",
        "RAW_IQ_SOURCE_TO_TRIGGER",
    }
    latency_by_id = {row["boundary_id"]: row for row in latency}

    checks: dict[str, bool] = {}
    checks[GATE_IDS[0]] = (
        report["task_id"] == TASK_ID
        and report["protocol_id"] == PROTOCOL_ID
        and report["schema_version"] == REPORT_SCHEMA
        and report["verdict"] == VERDICT
        and "matched-filter or discriminator performance qualification"
        in contract["scope"]["forbidden"]
    )
    checks[GATE_IDS[1]] = (
        len(report["parents"]) == 4
        and report["parents"]
        == _parent_bindings(_load(DEFAULT_CONFIG))
    )
    formal_parent = next(
        row for row in report["parents"] if row["task_id"] == "T9.2.4"
    )
    formal = _load(ROOT / formal_parent["path"]) if verify_live else None
    checks[GATE_IDS[2]] = (
        formal_parent["required_verdict"] == "NO_GO_TWIN_QUALIFICATION"
        and formal_parent["live_verdict"] == "NO_GO_TWIN_QUALIFICATION"
        and (
            not verify_live
            or formal["failure_propagation"]["released_tasks"] == ["T9.2.6"]
        )
    )
    required_bindings = {
        "config",
        "generator",
        "runtime",
        "core_rtl",
        "legacy_reference",
        "tests",
    }
    expected_implementation_bindings = {
        "config": _binding(DEFAULT_CONFIG),
        "generator": _binding(IMPLEMENTATION),
        "runtime": _binding(RUNTIME),
        "core_rtl": _binding(CORE_RTL),
        "legacy_reference": _binding(LEGACY_REFERENCE),
        "tests": _binding(TEST_FILE),
    }
    checks[GATE_IDS[3]] = (
        set(report["implementation_bindings"]) == required_bindings
        and report["implementation_bindings"]
        == expected_implementation_bindings
        and all(
            _binding_live(row)
            for row in report["implementation_bindings"].values()
        )
    )
    checks[GATE_IDS[4]] = (
        contract["project_boundary"]["input_boundary"].startswith(
            "after vendor RFDC"
        )
        and "vendor ADC calibration"
        in contract["project_boundary"]["excluded_from_project_frontend"]
        and "host DMA and software transport"
        in contract["project_boundary"]["excluded_from_project_frontend"]
    )
    checks[GATE_IDS[5]] = rates == expected_rates and all(
        row["integration_ns"] == 512 for row in rates
    )
    checks[GATE_IDS[6]] = (
        stream["tdata_bits"] == runtime.AXIS_TDATA_BITS
        and _exact_layout(stream["tdata_layout"], total_bits=32)
        and [(row["field"], row["signed"]) for row in stream["tdata_layout"]]
        == [("i_q1_15", True), ("q_q1_15", True)]
    )
    checks[GATE_IDS[7]] = (
        stream["tuser_bits"] == runtime.AXIS_TUSER_BITS
        and _exact_layout(stream["tuser_layout"], total_bits=128)
        and [
            (row["field"], row["lsb"], row["bits"])
            for row in stream["tuser_layout"]
        ]
        == list(runtime.TUSER_FIELDS)
    )
    checks[GATE_IDS[8]] = stream["domain_ids"] == {
        str(int(value)): value.name for value in runtime.DomainId
    }
    checks[GATE_IDS[9]] = (
        "remain stable" in stream["tvalid_tready"]
        and stream["packet_mode_required"] is True
        and stream["tlast"].startswith("asserted on the only final sample")
        and stream["channel_id_v1_required"] == 0
        and "exact +1" in stream["reset_epoch"]
    )
    expected_fixed = [
        ("input_iq", 16, "Q1.15"),
        ("complex_matched_filter_coefficient", 18, "Q1.17"),
        ("real_scalar_product", 34, "Q2.32"),
        ("complex_multiply_component_sum", 35, "Q3.32"),
        ("matched_filter_accumulator", 48, "Q16.32"),
        ("calibration_matrix", 18, "Q2.16"),
        ("calibration_product", 66, "Q18.48"),
        (
            "calibration_component_sum_with_aligned_offset",
            67,
            "Q19.48",
        ),
        ("calibration_offset", 24, "Q8.16"),
        ("calibrated_iq_llr_threshold_hysteresis", 24, "Q8.16"),
        ("ood_score", 8, "UQ0.8"),
        ("legacy_fast_path_map_llr", 22, "implementation-defined inherited LUT code"),
    ]
    checks[GATE_IDS[10]] = [
        (row["stage"], row["bits"], row["q_format"]) for row in fixed
    ] == expected_fixed
    arithmetic = contract["arithmetic_contract"]
    audit = report["executable_audit"]
    checks[GATE_IDS[11]] = (
        arithmetic["rounding_on_every_narrowing"]
        == "round-to-nearest ties-to-even"
        and "wraparound forbidden" in arithmetic["overflow"]
        and audit["rounding_exhaustive_cases"] == 262144
    )
    checks[GATE_IDS[12]] = (
        arithmetic["matched_filter_equation"]
        == "y=sum_n(conj(h[n])*x[n])"
        and arithmetic["calibration"].startswith("2x2 affine")
        and arithmetic["sticky_overflow_effect"].endswith(
            "observation_valid=0"
        )
        and arithmetic["matched_filter_accumulator_proof"][
            "maximum_legal_abs_code_upper_bound"
        ]
        == 2 * 32768 * 131072 * 128
        and arithmetic["calibration_bit_path"]["narrowing"].startswith(
            "arithmetic right shift 32"
        )
        and audit["matched_filter_reference_cases"] == 516
        and audit["calibration_reference_cases"] == 729
    )
    checks[GATE_IDS[13]] = (
        arithmetic["threshold_values"] is None
        and arithmetic["threshold_status"]
        == "BLOCKED_UNQUALIFIED_T9_2_4_NO_GO"
        and "T-RISK-20260726-01"
        in arithmetic["threshold_selection_released_by"]
    )
    package = contract["package_and_update_contract"]
    checks[GATE_IDS[14]] = (
        package["bank_count"] == 2
        and package["update_unit"].startswith("complete ")
        and package["version_rule"] == "exactly active_version+1, no wrap"
        and package["activation"].startswith("atomic only between windows")
        and package["partial_entry_patch"] == "forbidden"
        and package["unqualified_activation"] == "always rejected"
        and package["lkg_identity"].startswith("exact complete-package SHA256")
        and package["trusted_qualification_receipt_sha256"] is None
        and package["current_activation_count_required"] == 0
        and runtime.TRUSTED_QUALIFICATION_RECEIPT_SHA256 is None
        and audit["unsafe_commit_cases_rejected"] == 9
        and audit["successful_commit_count"] == 0
        and audit["trusted_qualification_receipt_sha256"] is None
        and audit["trusted_commit_fixture_scope"]
        == "CURRENT_ACTIVATION_CLOSED_PENDING_FRESH_TWIN_PASS"
    )
    cdc = contract["cdc_backpressure_reset_contract"]
    checks[GATE_IDS[15]] = (
        cdc["elastic_buffer"]["minimum_complete_windows"] == 2
        and cdc["elastic_buffer"]["minimum_beats"]
        == runtime.MIN_ELASTIC_BUFFER_BEATS
        and cdc["minimum_reset_cycles_slowest_clock"] == 16
        and "poison the entire affected window" in cdc["overflow"]
        and cdc["window_deadline_axis_cycles"]
        == runtime.WINDOW_DEADLINE_AXIS_CYCLES
        and "reset epoch increments exactly one" in cdc["fifo_flush"]
        and "stateful last-retired" in cdc["cross_window_freshness"]
        and audit["timeout_quarantine_cases"] == 10
        and audit["timeout_poison_count"] == 1
    )
    checks[GATE_IDS[16]] = cdc["error_priority"] == expected_priority
    checks[GATE_IDS[17]] = (
        cdc["no_postselection"] is True
        and "silent sample drop forbidden" in cdc["overflow"]
    )
    legacy = contract["legacy_fast_path_adapter"]
    checks[GATE_IDS[18]] = (
        legacy["word_bits"] == 58
        and _exact_layout(legacy["fields"], total_bits=58)
        and _legacy_fields_from_config(contract) == _runtime_legacy_fields()
        and (
            not verify_live
            or "input_payload[9:0]" in CORE_RTL.read_text(encoding="utf-8")
            and "in_word[57:42]" in CORE_RTL.read_text(encoding="utf-8")
        )
    )
    checks[GATE_IDS[19]] = audit["legacy_fast_path_roundtrips"] == 18
    checks[GATE_IDS[20]] = set(latency_by_id) == fast_boundaries | null_boundaries
    checks[GATE_IDS[21]] = (
        all(
            latency_by_id[name]["cycles"] == 6
            and latency_by_id[name]["ii_cycles"] == 1
            for name in fast_boundaries
        )
        and all(
            latency_by_id[name]["cycles"] is None
            for name in null_boundaries
        )
    )
    checks[GATE_IDS[22]] = all(
        latency_by_id[name]["measured_ns"] is None
        and latency_by_id[name]["cycles"] is None
        for name in null_boundaries
    )
    trigger = contract["trigger_contract"]
    checks[GATE_IDS[23]] = (
        trigger["physical_pinout"] is None
        and trigger["electrical_standard"] is None
        and trigger["measured_latency"] is None
        and trigger["debug_record_bits"] == 128
        and _exact_layout(trigger["debug_fields"], total_bits=128)
    )
    candidates = contract["platform_intersection"]["candidate_classes"]
    checks[GATE_IDS[24]] = (
        {row["platform_id"] for row in candidates}
        == {"AMD_ZCU111_XCZU28DR", "AMD_ZCU216_XCZU49DR"}
        and all(
            row["qualification_status"]
            == "SOURCE_CONFIRMED_NOT_SELECTED_NOT_BUILT"
            for row in candidates
        )
    )
    excluded = contract["platform_intersection"]["excluded_platforms"]
    checks[GATE_IDS[25]] = excluded == [
        {
            "platform_id": "Tang_Nano_20K_GW2AR",
            "reason": "low-speed digital transport/RTL reference only; no direct raw-IQ ADC chain",
        }
    ]
    intersection = contract["platform_intersection"]
    checks[GATE_IDS[26]] = (
        intersection["budget_status"]
        == "DESIGN_ENVELOPE_NOT_SYNTHESIS_OR_PLACE_ROUTE_RESULT"
        and intersection["frozen_project_budget"]["project_clock_target_mhz"]
        == 250
        and intersection["selection_task"] == "T9.7.2"
    )
    checks[GATE_IDS[27]] = (
        len(audit["nominal_rate_domain_cases"]) == 6
        and all(
            row["reason"] == "ACCEPT"
            for row in audit["nominal_rate_domain_cases"]
        )
        and audit["stateful_freshness_cases"] == 5
    )
    checks[GATE_IDS[28]] = (
        len(audit["fault_flag_cases"]) == 16
        and len(audit["structural_failure_cases"]) == 8
        and all(not row["accepted"] for row in audit["fault_flag_cases"])
        and audit["fail_closed_adapter_cases"] == 16
        and audit["unstable_backpressure_rejected"] is True
        and audit["mixed_reset_epoch_rejected"] is True
        and audit["window_timeout_rejected"] is True
        and audit["timeout_quarantine_cases"] == 10
        and audit["timeout_poison_count"] == 1
    )
    checks[GATE_IDS[29]] = (
        audit["iq_signed_code_roundtrips"] == 131072
        and audit["tuser_boundary_roundtrips"] == 18
        and audit["rounding_exhaustive_cases"] == 262144
        and audit["matched_filter_reference_cases"] == 516
        and audit["calibration_reference_cases"] == 729
    )
    checks[GATE_IDS[30]] = (
        len(audit["package_cases"]) == 2
        and audit["unsafe_commit_cases_rejected"] == 9
        and audit["successful_commit_count"] == 0
        and audit["strict_bool_alias_cases_rejected"] == 16
    )
    checks[GATE_IDS[31]] = (
        len(claims) == 15 and all(value is None for value in claims.values())
    )
    checks[GATE_IDS[32]] = (
        _binding_live(report["source_data"])
        and report["source_data"]["path"] == _relative(DEFAULT_SOURCE_DATA)
        and _read_source_data(ROOT / report["source_data"]["path"])
        == _snapshot(report)
    )
    outputs = report["canonical_outputs"]
    checks[GATE_IDS[33]] = outputs == {
        "report": _relative(DEFAULT_REPORT),
        "source_data": _relative(DEFAULT_SOURCE_DATA),
        "markdown": _relative(DEFAULT_MARKDOWN),
        "release_pin": _relative(DEFAULT_RELEASE_PIN),
    }
    mutation = report["semantic_mutation_audit"]
    checks[GATE_IDS[34]] = (
        mutation["count"] == len(GATE_IDS)
        and mutation["detected"] == len(GATE_IDS)
        and mutation["all_detected"] is True
        and [row["target_gate"] for row in mutation["records"]]
        == list(GATE_IDS)
        and all(
            row["target_gate"] in row["failed_gates"]
            for row in mutation["records"]
        )
    )
    checks[GATE_IDS[35]] = (
        downstream["T-RISK-20260726-01"]["released"] is True
        and all(
            not row["released"]
            for task, row in downstream.items()
            if task != "T-RISK-20260726-01"
        )
    )
    return checks


def _mutation_specs() -> list[Callable[[dict[str, Any]], None]]:
    def set_path(path: Sequence[Any], value: Any) -> Callable[[dict[str, Any]], None]:
        def mutate(candidate: dict[str, Any]) -> None:
            cursor: Any = candidate
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value

        return mutate

    mutations: list[Callable[[dict[str, Any]], None]] = [
        set_path(("verdict",), "PASS_PERFORMANCE"),
        set_path(("parents", 0, "analysis_sha256"), "bad"),
        set_path(("parents", 2, "required_verdict"), "PASS"),
        lambda c: c["implementation_bindings"].pop("runtime"),
        set_path(("contract", "project_boundary", "input_boundary"), "at analog aperture"),
        set_path(("contract", "rate_profiles", 0, "integration_ns"), 256),
        set_path(("contract", "stream_contract", "tdata_layout", 1, "lsb"), 15),
        set_path(("contract", "stream_contract", "tuser_layout", -1, "bits"), 7),
        set_path(("contract", "stream_contract", "domain_ids", "3"), "LIVE_RAW"),
        set_path(("contract", "stream_contract", "packet_mode_required"), False),
        set_path(("contract", "fixed_point_contract", 4, "bits"), 40),
        set_path(("executable_audit", "rounding_exhaustive_cases"), 17),
        set_path(("contract", "arithmetic_contract", "matched_filter_equation"), "mean(iq)"),
        set_path(("contract", "arithmetic_contract", "threshold_values"), [0]),
        set_path(("contract", "package_and_update_contract", "partial_entry_patch"), "allowed"),
        set_path(("contract", "cdc_backpressure_reset_contract", "elastic_buffer", "minimum_beats"), 1),
        set_path(("contract", "cdc_backpressure_reset_contract", "error_priority", 0), "ACCEPT"),
        set_path(("contract", "cdc_backpressure_reset_contract", "no_postselection"), False),
        set_path(("contract", "legacy_fast_path_adapter", "fields", 0, "bits"), 9),
        set_path(("executable_audit", "legacy_fast_path_roundtrips"), 0),
        lambda c: c["contract"]["latency_boundaries"].pop(),
        set_path(("contract", "latency_boundaries", 2, "cycles"), 6),
        set_path(("contract", "latency_boundaries", 3, "measured_ns"), 24),
        set_path(("contract", "trigger_contract", "physical_pinout"), "PIN_A"),
        set_path(("contract", "platform_intersection", "candidate_classes", 0, "qualification_status"), "BUILT"),
        lambda c: c["contract"]["platform_intersection"].update(excluded_platforms=[]),
        set_path(("contract", "platform_intersection", "budget_status"), "MEASURED"),
        lambda c: c["executable_audit"]["nominal_rate_domain_cases"].pop(),
        lambda c: c["executable_audit"]["fault_flag_cases"].pop(),
        set_path(("executable_audit", "iq_signed_code_roundtrips"), 4),
        set_path(("executable_audit", "unsafe_commit_cases_rejected"), 0),
        set_path(("claim_state", "frontend_performance"), False),
        set_path(("source_data", "path"), "docs/alternate.csv"),
        set_path(("canonical_outputs", "markdown"), "docs/alternate.md"),
        set_path(("semantic_mutation_audit", "detected"), 0),
        set_path(("downstream_release", "T9.2.7", "released"), True),
    ]
    if len(mutations) != len(GATE_IDS):
        raise AssertionError("mutation/gate cardinality mismatch")
    return mutations


def _mutation_audit(report: dict[str, Any]) -> dict[str, Any]:
    records = []
    for gate, mutate in zip(GATE_IDS, _mutation_specs()):
        candidate = copy.deepcopy(report)
        mutate(candidate)
        try:
            failed = [
                name
                for name, passed in _check_gates(
                    candidate, verify_live=False, verify_outputs=False
                ).items()
                if not passed
            ]
        except (
            AssertionError,
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ):
            # A malformed semantic structure is a fail-closed rejection.  Keep
            # all gates failed rather than letting a mutation crash the audit.
            failed = list(GATE_IDS)
        records.append(
            {
                "target_gate": gate,
                "failed_gates": failed,
                "target_detected": gate in failed,
            }
        )
    detected = sum(row["target_detected"] for row in records)
    return {
        "count": len(records),
        "detected": detected,
        "all_detected": detected == len(records),
        "records": records,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    contract = report["contract"]
    audit = report["executable_audit"]
    latency = contract["latency_boundaries"]
    fixed = contract["fixed_point_contract"]
    lines = [
        "# T9.2.6 raw-IQ 前端与候选平台交集 envelope",
        "",
        f"- 状态：`{report['verdict']}`（仅协议/接口冻结 PASS）",
        f"- analysis：`{report['analysis_sha256']}`",
        "- T9.2.4 双后端结论仍是 `NO_GO_TWIN_QUALIFICATION`；本任务没有使用失败 twin 的性能值选阈值。",
        "- `threshold_values`、frontend ROC/LER、recorded/live IQ、真板 latency/resource/power 与外部 SOTA 全部保持 `null`。",
        "",
        "## 冻结边界",
        "",
        f"- 输入：{contract['project_boundary']['input_boundary']}",
        f"- 输出：{contract['project_boundary']['output_boundary']}",
        "- 主 AXI4-Stream：32-bit `TDATA`（I16/Q16, Q1.15）+ 128-bit `TUSER`，250 MHz 只是实现目标，不是已达时序。",
        "- rate family：125 MS/s × 64 与 250 MS/s × 128，均为 512 ns integration window。",
        "- CDC 至少容纳两个最大完整窗口（256 beats）；8-bit reset epoch + stateful retired-window receipt 拒绝跨 reset 旧 beat、重放和跨窗乱序。",
        "- 首个 TVALID 后 192 个 ACLK 内必须退休 TLAST；timeout/overflow/reset/序列/version/CRC 错误均只产生 fail-closed record，禁止静默丢样和 postselection。",
        "",
        "## 定点链",
        "",
        "| stage | bits | Q-format |",
        "| --- | ---: | --- |",
    ]
    lines.extend(
        f"| `{row['stage']}` | {row['bits']} | `{row['q_format']}` |"
        for row in fixed
    )
    lines.extend(
        [
            "",
            "所有窄化使用 round-to-nearest ties-to-even；所有溢出使用 signed saturation + sticky fault，禁止 wraparound。matched filter 结构为 `sum(conj(h[n])*x[n])`，calibration 为 versioned 2×2 affine package。阈值寄存器格式已冻结为 signed Q8.16，但数值未资格化。",
            "",
            "## 四个不可混排 latency boundary",
            "",
            "| boundary | cycles | II | 状态 | measured ns |",
            "| --- | ---: | ---: | --- | ---: |",
        ]
    )
    for row in latency:
        lines.append(
            "| `{}` | {} | {} | `{}` | {} |".format(
                row["boundary_id"],
                "null" if row["cycles"] is None else row["cycles"],
                "null" if row["ii_cycles"] is None else row["ii_cycles"],
                row["status"],
                "null" if row["measured_ns"] is None else row["measured_ns"],
            )
        )
    lines.extend(
        [
            "",
            "六周期/II=1 只绑定既有 fast core 与待 T9.2.7 复证的 `discriminator-out -> action`；不得迁移到 ADC/raw-IQ/trigger。",
            "",
            "## 候选平台交集",
            "",
            "- ZCU111/XCZU28DR 与 ZCU216/XCZU49DR 仅为 vendor-source-confirmed candidate，尚未选择、综合、P&R 或上板。",
            "- Tang Nano 20K/GW2AR 只保留为低速数字控制参考，明确不属于 raw-IQ platform intersection。",
            "- 冻结 budget：250 MHz target、32-bit TDATA、128-bit TUSER、≤32 DSP、≤12 BRAM36、≤25k LUT、≤30k FF；这些是设计上限，不是资源结果。",
            "",
            "## 可执行反简化证据",
            "",
            f"- signed I/Q code roundtrip：{audit['iq_signed_code_roundtrips']:,}",
            f"- TUSER boundary roundtrip：{audit['tuser_boundary_roundtrips']}",
            f"- ties-even exhaustive conversions：{audit['rounding_exhaustive_cases']:,}",
            f"- matched-filter independent arithmetic cases：{audit['matched_filter_reference_cases']}",
            f"- 66/67-bit calibration arithmetic cases：{audit['calibration_reference_cases']}",
            f"- nominal rate×domain windows：{len(audit['nominal_rate_domain_cases'])}/6",
            f"- explicit error flags：{len(audit['fault_flag_cases'])}/16 全部 fail closed",
            f"- structural adversarial cases：{len(audit['structural_failure_cases'])}/8 全部拒绝",
            f"- stateful freshness/reset/timeout：{audit['stateful_freshness_cases']} + mixed-epoch + deadline 全部拒绝",
            f"- timeout quarantine transaction：{audit['timeout_quarantine_cases']}/10，poison count={audit['timeout_poison_count']}",
            f"- A/B activation：成功 {audit['successful_commit_count']}，拒绝 unsafe case {audit['unsafe_commit_cases_rejected']}；trusted receipt 保持 null",
            f"- strict bool/int alias：{audit['strict_bool_alias_cases_rejected']}/16 全部拒绝",
            f"- legacy 58-bit adapter roundtrip：{audit['legacy_fast_path_roundtrips']}/18",
            f"- semantic gates/mutations：{len(report['gates'])}/{report['semantic_mutation_audit']['detected']}",
            "",
            "## 下游",
            "",
            "`T9.2.7` 仍被 T9.2.4 NO-GO 阻塞；下一项只释放 `T-RISK-20260726-01` fresh twin IQ/likelihood 修复与重新资格化。旧 NO-GO 不回写。",
            "",
        ]
    )
    return "\n".join(lines)


def _analysis_sha(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("analysis_sha256", None)
    return _canonical_sha256(payload)


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load(config_path)
    if config["task_id"] != TASK_ID or config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("wrong T9.2.6 config identity")
    parents = _parent_bindings(config)
    bindings = {
        "config": _binding(config_path),
        "generator": _binding(IMPLEMENTATION),
        "runtime": _binding(RUNTIME),
        "core_rtl": _binding(CORE_RTL),
        "legacy_reference": _binding(LEGACY_REFERENCE),
        "tests": _binding(TEST_FILE),
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "schema_version": REPORT_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": VERDICT,
        "parents": parents,
        "implementation_bindings": bindings,
        "contract": {
            key: config[key]
            for key in (
                "scope",
                "project_boundary",
                "rate_profiles",
                "stream_contract",
                "fixed_point_contract",
                "arithmetic_contract",
                "package_and_update_contract",
                "cdc_backpressure_reset_contract",
                "legacy_fast_path_adapter",
                "latency_boundaries",
                "trigger_contract",
                "platform_intersection",
                "source_registry",
            )
        },
        "executable_audit": _executable_audit(),
        "claim_state": config["current_claim_state"],
        "downstream_release": config["downstream_release"],
        "source_data": {
            "path": _relative(DEFAULT_SOURCE_DATA),
            "bytes": 1,
            "sha256": "0" * 64,
        },
        "canonical_outputs": {
            "report": _relative(DEFAULT_REPORT),
            "source_data": _relative(DEFAULT_SOURCE_DATA),
            "markdown": _relative(DEFAULT_MARKDOWN),
            "release_pin": _relative(DEFAULT_RELEASE_PIN),
        },
        "semantic_mutation_audit": {
            "count": len(GATE_IDS),
            "detected": len(GATE_IDS),
            "all_detected": True,
            "records": [
                {
                    "target_gate": gate,
                    "failed_gates": [gate],
                    "target_detected": True,
                }
                for gate in GATE_IDS
            ],
        },
        "gates": {gate: True for gate in GATE_IDS},
    }
    # Source Data does not include its own binding, avoiding a self-hash cycle.
    report["source_data"] = _write_source_data(
        _snapshot(report), DEFAULT_SOURCE_DATA
    )
    report["semantic_mutation_audit"] = _mutation_audit(report)
    report["gates"] = _check_gates(
        report, verify_live=True, verify_outputs=True
    )
    if not all(report["gates"].values()):
        failed = [key for key, value in report["gates"].items() if not value]
        undetected = [
            row["target_gate"]
            for row in report["semantic_mutation_audit"]["records"]
            if not row["target_detected"]
        ]
        raise ValueError(
            f"T9.2.6 gates failed: {failed}; undetected mutations: {undetected}"
        )
    report["analysis_sha256"] = _analysis_sha(report)
    return report


def _release_pin(report: Mapping[str, Any]) -> dict[str, Any]:
    pin = {
        "task_id": TASK_ID,
        "schema_version": RELEASE_SCHEMA,
        "analysis_sha256": report["analysis_sha256"],
        "verdict": report["verdict"],
        "artifacts": {
            "config": _binding(DEFAULT_CONFIG),
            "generator": _binding(IMPLEMENTATION),
            "runtime": _binding(RUNTIME),
            "core_rtl": _binding(CORE_RTL),
            "legacy_reference": _binding(LEGACY_REFERENCE),
            "tests": _binding(TEST_FILE),
            "report": _binding(DEFAULT_REPORT),
            "source_data": _binding(DEFAULT_SOURCE_DATA),
            "markdown": _binding(DEFAULT_MARKDOWN),
        },
        "claim_boundary": {
            "interface_only": True,
            "all_current_claim_fields_typed_null": True,
            "t9_2_4_no_go_preserved": True,
        },
    }
    for index, parent in enumerate(report["parents"]):
        pin["artifacts"][f"parent_{index}_{parent['task_id']}"] = parent[
            "artifact"
        ]
    return pin


def generate() -> dict[str, Any]:
    report = build_report()
    _atomic_json(report, DEFAULT_REPORT)
    _atomic_text(_markdown(report), DEFAULT_MARKDOWN)
    _atomic_json(_release_pin(report), DEFAULT_RELEASE_PIN)
    verify_report(
        DEFAULT_REPORT,
        expected_analysis_sha256=report["analysis_sha256"],
    )
    return report


def verify_report(
    report_path: Path = DEFAULT_REPORT,
    *,
    expected_analysis_sha256: str | None = None,
) -> dict[str, bool]:
    if isinstance(report_path, Mapping):
        raise TypeError(
            "public verification accepts only the canonical report path"
        )
    path = Path(report_path)
    if path.resolve() != DEFAULT_REPORT.resolve():
        raise ValueError("only the canonical T9.2.6 report path is accepted")
    report = _load(path)
    if report.get("analysis_sha256") != _analysis_sha(report):
        raise ValueError("report analysis self-hash mismatch")
    if (
        expected_analysis_sha256 is not None
        and report["analysis_sha256"] != expected_analysis_sha256
    ):
        raise ValueError("unexpected report analysis")
    checks = _check_gates(
        report, verify_live=True, verify_outputs=True
    )
    if not checks or not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"T9.2.6 report verification failed: {failed}")
    if DEFAULT_MARKDOWN.read_text(encoding="utf-8") != _markdown(report):
        raise ValueError("canonical markdown mismatch")
    pin = _load(DEFAULT_RELEASE_PIN)
    expected_pin = _release_pin(report)
    if pin != expected_pin or not all(
        _binding_live(binding) for binding in pin["artifacts"].values()
    ):
        raise ValueError("release pin mismatch")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify existing canonical outputs without rewriting them",
    )
    args = parser.parse_args(argv)
    report = (
        _load(DEFAULT_REPORT)
        if args.verify_only
        else generate()
    )
    checks = verify_report(
        DEFAULT_REPORT,
        expected_analysis_sha256=report["analysis_sha256"],
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "gates": f"{sum(checks.values())}/{len(checks)}",
                "mutations": (
                    f"{report['semantic_mutation_audit']['detected']}/"
                    f"{report['semantic_mutation_audit']['count']}"
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
