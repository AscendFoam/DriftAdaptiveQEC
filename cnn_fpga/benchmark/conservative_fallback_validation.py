"""T4.2.3 production validation for traceable conservative fast-path fallback."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.conservative_fallback import (
    FALLBACK_ACTIVE,
    FAULT_BITS,
    FAULT_ORDER,
    HEALTHY,
    RECOVERING,
    RESET_REQUIRED,
    ConservativeFallbackConfig,
    ConservativeFallbackController,
    ConservativeFallbackInput,
    TrustedParameterImage,
)
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTImage,
    ParametricMAPLUTInput,
    ParametricMAPLUTRuntime,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs" / "t4_2_3_conservative_fallback_validation.json"
DEFAULT_CSV = ROOT / "docs" / "t4_2_3_conservative_fallback_source_data.csv"
SCHEMA_VERSION = "t4.2.3-conservative-fallback-validation-v1"
SCENARIOS = (
    "nominal_map",
    "ood_boundary",
    "leakage_reset",
    "parameter_age_boundary",
    "input_crc_fault",
    "image_crc_fault",
    "image_sha_fault",
    "valid_version_switch",
    "version_faults",
    "deadline_fault",
    "map_missing",
    "map_alignment_fault",
    "unexpected_reset_ack",
    "simultaneous_faults",
    "fallback_recovery",
    "fault_counter_saturation",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/conservative_fallback.py",
        "cnn_fpga/runtime/experimental_event_fsm.py",
        "cnn_fpga/runtime/parametric_map_lut.py",
        "cnn_fpga/benchmark/conservative_fallback_validation.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _images() -> tuple[ParametricMAPLUTImage, ...]:
    config = ParametricMAPLUTConfig()
    return tuple(
        compile_parametric_map_lut(params, active_bank_version=index, config=config)
        for index, (params, _) in enumerate(registered_parameter_profiles(config))
    )


def _trusted(images: Sequence[ParametricMAPLUTImage]) -> tuple[TrustedParameterImage, ...]:
    return tuple(
        TrustedParameterImage(image.active_bank_version, image.image_crc32, image.image_sha256)
        for image in images
    )


def _action_codes(
    images: Sequence[ParametricMAPLUTImage],
) -> dict[tuple[int, int, bool], int]:
    codes: dict[tuple[int, int, bool], int] = {}
    for image in images:
        runtime = ParametricMAPLUTRuntime(image)
        for phase in (0, 1):
            for flip in (False, True):
                for code in range(image.config.adc_levels):
                    decision = runtime.decode_code(
                        ParametricMAPLUTInput(0, code, phase, image.active_bank_version)
                    )
                    if decision.logical_flip is flip:
                        codes[(image.active_bank_version, phase, flip)] = code
                        break
    return codes


def _scenario_spec(scenario: str, offset: int) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "image_index": 1,
        "reported_image_index": 1,
        "expected_version": 1,
        "phase": offset % 2,
        "flip": offset % 3 == 0,
        "syndrome_x": "g",
        "syndrome_z": "g",
        "map_kind": "valid",
        "age": 0,
        "ood": 0,
        "reset_ack": False,
        "observation_valid": True,
        "input_crc_ok": True,
        "deadline_ok": True,
        "crc_corrupt": False,
        "sha_corrupt": False,
    }
    position = offset % 8
    if scenario == "nominal_map":
        return spec
    if scenario == "ood_boundary":
        spec["ood"] = (191, 192, 193, 255)[offset % 4]
        return spec
    if scenario == "leakage_reset":
        if position in (0, 1):
            spec["syndrome_x"] = spec["syndrome_z"] = "leakage"
        if position == 3:
            spec["reset_ack"] = True
        spec["flip"] = True
        return spec
    if scenario == "parameter_age_boundary":
        spec["age"] = (63, 64, 65, 128)[offset % 4]
        return spec
    if scenario == "input_crc_fault":
        spec["input_crc_ok"] = offset % 4 != 0
        return spec
    if scenario == "image_crc_fault":
        spec["crc_corrupt"] = offset % 4 == 0
        return spec
    if scenario == "image_sha_fault":
        spec["sha_corrupt"] = offset % 4 == 0
        return spec
    if scenario == "valid_version_switch":
        index = min(offset // 32, 7)
        spec.update(image_index=index, reported_image_index=index, expected_version=index)
        return spec
    if scenario == "version_faults":
        position = offset % 16
        spec.update(image_index=2, reported_image_index=2, expected_version=2)
        if position == 1:
            spec.update(image_index=1, reported_image_index=1, expected_version=1)
        elif position == 2:
            spec.update(reported_image_index=7, expected_version=7)
        elif position == 3:
            spec["expected_version"] = 8
        return spec
    if scenario == "deadline_fault":
        spec["deadline_ok"] = offset % 4 != 0
        return spec
    if scenario == "map_missing":
        spec["map_kind"] = "missing" if offset % 4 == 0 else "valid"
        return spec
    if scenario == "map_alignment_fault":
        spec["map_kind"] = "corrupt" if offset % 4 == 0 else "valid"
        return spec
    if scenario == "unexpected_reset_ack":
        spec["reset_ack"] = offset % 4 == 0
        return spec
    if scenario == "simultaneous_faults":
        spec.update(
            expected_version=8,
            map_kind="missing",
            age=65,
            ood=255,
            observation_valid=False,
            input_crc_ok=False,
            deadline_ok=False,
            syndrome_x="leakage",
        )
        return spec
    if scenario == "fallback_recovery":
        if position == 0:
            spec["deadline_ok"] = False
        return spec
    if scenario == "fault_counter_saturation":
        spec["map_kind"] = "missing"
        return spec
    raise ValueError(f"unknown scenario {scenario!r}")


def _run_scenarios(*, cycles_per_scenario: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cycles_per_scenario < 256:
        raise ValueError("cycles_per_scenario must be at least 256")
    images = _images()
    runtimes = tuple(ParametricMAPLUTRuntime(image) for image in images)
    codes = _action_codes(images)
    trusted = _trusted(images)
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        controller = ConservativeFallbackController(trusted)
        for offset in range(cycles_per_scenario):
            spec = _scenario_spec(scenario, offset)
            image = images[spec["image_index"]]
            runtime = runtimes[spec["image_index"]]
            phase = int(spec["phase"])
            flip = bool(spec["flip"])
            key = (image.active_bank_version, phase, flip)
            if key not in codes:
                flip = False
                key = (image.active_bank_version, phase, False)
            event_cycle = controller.event_config.start_event_cycle + offset
            decision = runtime.decode_code(
                ParametricMAPLUTInput(
                    event_cycle - image.config.pipeline_latency_cycles,
                    codes[key],
                    phase,
                    image.active_bank_version,
                )
            )
            if spec["map_kind"] == "missing":
                decision = None
            elif spec["map_kind"] == "corrupt":
                decision = replace(decision, logical_action="I" if decision.logical_action != "I" else decision.phase_label)
            reported = images[spec["reported_image_index"]]
            reported_crc = "0" * 8 if spec["crc_corrupt"] else reported.image_crc32
            reported_sha = "0" * 64 if spec["sha_corrupt"] else reported.image_sha256
            before_version = controller.state.trusted_active_bank_version
            event = ConservativeFallbackInput(
                cycle_index=event_cycle,
                syndrome_x=spec["syndrome_x"],
                syndrome_z=spec["syndrome_z"],
                quadrature_phase_bit=phase,
                map_decision=decision,
                expected_active_bank_version=spec["expected_version"],
                reported_image_crc32=reported_crc,
                reported_image_sha256=reported_sha,
                parameter_age_cycles=spec["age"],
                ood_score_code=spec["ood"],
                reset_ack=spec["reset_ack"],
                observation_valid=spec["observation_valid"],
                input_crc_ok=spec["input_crc_ok"],
                deadline_ok=spec["deadline_ok"],
            )
            action = controller.step(event)
            hardware = action.hardware_action
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_offset": offset,
                    "source_cycle": hardware.source_cycle,
                    "event_cycle": event_cycle,
                    "hardware_action_cycle": hardware.action_cycle,
                    "syndrome_x": event.syndrome_x,
                    "syndrome_z": event.syndrome_z,
                    "phase_bit": phase,
                    "ood_score_code": event.ood_score_code,
                    "parameter_age_cycles": event.parameter_age_cycles,
                    "expected_active_bank_version": event.expected_active_bank_version,
                    "trusted_version_before": before_version,
                    "trusted_version_after": action.trusted_active_bank_version,
                    "status": action.status,
                    "fault_flags": "|".join(action.fault_flags),
                    "fault_mask": action.fault_mask,
                    "primary_reason": action.primary_reason,
                    "reason_trace": action.reason_trace,
                    "conservative_action": action.conservative_action,
                    "active_profile_id": action.active_profile_id,
                    "map_decision_present": int(event.map_decision is not None),
                    "map_decision_accepted": int(action.map_decision_accepted),
                    "map_logical_flip": int(bool(event.map_decision and event.map_decision.logical_flip)),
                    "hardware_mode": hardware.mode,
                    "hardware_reason": hardware.reason,
                    "correction_enable": int(hardware.correction_enable),
                    "reset_request": int(hardware.reset_request),
                    "map_action_inhibited": int(hardware.map_action_inhibited),
                    "pauli_frame_delta_x": int(hardware.pauli_frame_delta_x),
                    "pauli_frame_delta_z": int(hardware.pauli_frame_delta_z),
                    "phase_frame_delta_x_code": hardware.phase_frame_delta_x_code,
                    "phase_frame_delta_z_code": hardware.phase_frame_delta_z_code,
                    "fault_run": action.fault_run,
                    "good_run": action.good_run,
                    "fault_cycle_count": action.fault_cycle_count,
                    "leakage_cycle_count": action.leakage_cycle_count,
                    "per_flag_cycle_counts": ";".join(map(str, action.per_flag_cycle_counts)),
                    "observation_valid": int(event.observation_valid),
                    "input_crc_ok": int(event.input_crc_ok),
                    "deadline_ok": int(event.deadline_ok),
                    "reported_image_crc32": event.reported_image_crc32,
                    "reported_image_sha256": event.reported_image_sha256,
                }
            )
    diagnostics = {
        "cycles_per_scenario": cycles_per_scenario,
        "trusted_images": [asdict(item) for item in trusted],
        "fault_flag_counts": {
            flag: sum(flag in row["fault_flags"].split("|") for row in rows)
            for flag in FAULT_ORDER
        },
        "status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in sorted({row["status"] for row in rows})
        },
    }
    return rows, diagnostics


def _rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resource_contract() -> dict[str, Any]:
    return {
        "identity": "exact_health_state_and_reason_word_proxy_not_synthesis",
        "fault_flag_bits": len(FAULT_ORDER),
        "health_counter_bits_each": 8,
        "health_counter_count": 4 + len(FAULT_ORDER),
        "map_plus_health_event_latency_cycles": 6,
        "initiation_interval_cycles": 1,
        "target_lut_count": None,
        "target_ff_count": None,
        "target_bram_count": None,
        "target_dsp_count": None,
        "fmax_mhz": None,
        "rtl_measured": False,
        "board_measured": False,
    }


def run_validation(
    *,
    cycles_per_scenario: int = 256,
    json_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    rows, diagnostics = _run_scenarios(cycles_per_scenario=cycles_per_scenario)
    repeat, repeat_diagnostics = _run_scenarios(cycles_per_scenario=cycles_per_scenario)
    blocking_rows = [
        row
        for row in rows
        if any(
            flag and flag != "leakage_observed"
            for flag in row["fault_flags"].split("|")
        )
    ]
    leakage_only = [row for row in rows if row["fault_flags"] == "leakage_observed"]
    version_switch = [row for row in rows if row["scenario"] == "valid_version_switch"]
    version_faults = [row for row in rows if row["scenario"] == "version_faults"]
    ood_rows = [row for row in rows if row["scenario"] == "ood_boundary"]
    age_rows = [row for row in rows if row["scenario"] == "parameter_age_boundary"]
    recovery = [row for row in rows if row["scenario"] == "fallback_recovery"]
    saturation = [row for row in rows if row["scenario"] == "fault_counter_saturation"]
    observed_flags = {
        flag for row in rows for flag in row["fault_flags"].split("|") if flag
    }
    resource = _resource_contract()
    gates = {
        "all_sixteen_registered_scenarios_are_complete": len(rows)
        == len(SCENARIOS) * cycles_per_scenario
        and {row["scenario"] for row in rows} == set(SCENARIOS),
        "all_fourteen_health_and_integrity_flags_are_exercised": observed_flags
        == set(FAULT_ORDER),
        "every_blocking_fault_enters_frame_hold_fallback": blocking_rows
        and all(
            row["status"] == FALLBACK_ACTIVE
            and row["hardware_mode"] == "fallback"
            and row["conservative_action"] == "frame_hold"
            for row in blocking_rows
        ),
        "blocking_faults_never_accept_map_or_update_frames": all(
            row["map_decision_accepted"] == 0
            and row["pauli_frame_delta_x"] == 0
            and row["pauli_frame_delta_z"] == 0
            and row["phase_frame_delta_x_code"] == 0
            and row["phase_frame_delta_z_code"] == 0
            for row in blocking_rows
        ),
        "leakage_uses_hold_reset_and_ack_without_map_mutation": leakage_only
        and any(row["status"] == RESET_REQUIRED for row in rows if row["scenario"] == "leakage_reset")
        and all(row["map_decision_accepted"] == 1 for row in leakage_only)
        and all(row["pauli_frame_delta_x"] == row["pauli_frame_delta_z"] == 0 for row in leakage_only),
        "ood_threshold_is_inclusive_at_192_and_fails_at_193": any(
            row["ood_score_code"] == 192 and not row["fault_flags"] for row in ood_rows
        )
        and all(
            ("ood_score_exceeded" in row["fault_flags"]) == (row["ood_score_code"] > 192)
            for row in ood_rows
        ),
        "parameter_age_64_is_current_and_65_is_stale": any(
            row["parameter_age_cycles"] == 64 and not row["fault_flags"] for row in age_rows
        )
        and all(
            ("parameter_stale" in row["fault_flags"])
            == (row["parameter_age_cycles"] > 64)
            for row in age_rows
        ),
        "input_crc_image_crc_and_sha_faults_are_distinct": all(
            diagnostics["fault_flag_counts"][flag] > 0
            for flag in ("input_crc_mismatch", "image_crc_mismatch", "image_sha256_mismatch")
        ),
        "all_eight_registered_versions_commit_monotonically": {
            row["trusted_version_after"] for row in version_switch
        }
        == set(range(8))
        and [row["trusted_version_after"] for row in version_switch]
        == sorted(row["trusted_version_after"] for row in version_switch),
        "rollback_unknown_and_mismatched_versions_preserve_trusted_bank": all(
            row["trusted_version_after"] == row["trusted_version_before"]
            for row in version_faults
            if any(
                flag in row["fault_flags"]
                for flag in (
                    "bank_version_rollback",
                    "unknown_bank_version",
                    "bank_version_mismatch",
                )
            )
        )
        and all(
            diagnostics["fault_flag_counts"][flag] > 0
            for flag in (
                "bank_version_rollback",
                "unknown_bank_version",
                "bank_version_mismatch",
            )
        ),
        "deadline_missing_map_alignment_and_ack_faults_are_traced": all(
            diagnostics["fault_flag_counts"][flag] > 0
            for flag in (
                "deadline_miss",
                "map_decision_missing",
                "map_alignment_or_action_invalid",
                "unexpected_reset_ack",
            )
        ),
        "simultaneous_fault_mask_and_reason_trace_preserve_every_flag": all(
            row["fault_mask"]
            == sum(FAULT_BITS[flag] for flag in row["fault_flags"].split("|") if flag)
            and all(flag in row["reason_trace"] for flag in row["fault_flags"].split("|") if flag)
            for row in rows
        ),
        "fallback_clear_requires_one_recovering_then_second_good_cycle": any(
            row["scenario_offset"] % 8 == 1 and row["status"] == RECOVERING
            for row in recovery
        )
        and all(
            row["status"] == HEALTHY
            for row in recovery
            if row["scenario_offset"] % 8 == 2
        ),
        "fault_and_per_reason_counters_reach_uint8_saturation": max(
            row["fault_run"] for row in saturation
        )
        == 255
        and max(row["fault_cycle_count"] for row in saturation) == 255
        and max(
            int(row["per_flag_cycle_counts"].split(";")[FAULT_ORDER.index("map_decision_missing")])
            for row in saturation
        )
        == 255,
        "healthy_path_accepts_map_and_updates_both_axes": any(
            row["map_decision_accepted"] and row["pauli_frame_delta_x"] for row in rows
        )
        and any(row["map_decision_accepted"] and row["pauli_frame_delta_z"] for row in rows),
        "map_plus_health_event_latency_is_exactly_six_cycles": all(
            row["hardware_action_cycle"] - row["source_cycle"] == 6 for row in rows
        ),
        "health_pipeline_has_initiation_interval_one": all(
            [row["hardware_action_cycle"] for row in rows if row["scenario"] == scenario]
            == list(range(6, 6 + cycles_per_scenario))
            for scenario in SCENARIOS
        ),
        "replay_is_bit_deterministic": _rows_sha256(rows) == _rows_sha256(repeat)
        and diagnostics == repeat_diagnostics,
        "online_contract_contains_no_truth_or_hidden_fields": not any(
            token in name
            for name in ConservativeFallbackInput.__dataclass_fields__
            for token in ("truth", "hidden", "drift", "recovery_depth")
        ),
        "resource_and_hardware_fields_remain_non_measured": all(
            resource[field] is None
            for field in (
                "target_lut_count",
                "target_ff_count",
                "target_bram_count",
                "target_dsp_count",
                "fmax_mhz",
            )
        )
        and not resource["rtl_measured"]
        and not resource["board_measured"],
    }
    output_csv = DEFAULT_CSV if csv_path is None else Path(csv_path)
    _write_csv(rows, output_csv)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "T4.2.3",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "config": asdict(ConservativeFallbackConfig()),
        "fault_registry": [{"name": name, "bit": FAULT_BITS[name]} for name in FAULT_ORDER],
        "scenarios": list(SCENARIOS),
        "rows": len(rows),
        "diagnostics": diagnostics,
        "resource_contract": resource,
        "source_data": {
            "path": str(output_csv.relative_to(ROOT)).replace("\\", "/"),
            "rows": len(rows),
            "sha256": _sha256_file(output_csv),
            "canonical_rows_sha256": _rows_sha256(rows),
        },
        "online_contract": {
            "input_fields": list(ConservativeFallbackInput.__dataclass_fields__),
            "step_signature": list(inspect.signature(ConservativeFallbackController.step).parameters),
            "hidden_truth_inputs": [],
        },
        "gate_summary": {
            "passed": sum(gates.values()),
            "failed": len(gates) - sum(gates.values()),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": "traceable observed-health and integrity frame-hold/reset software fallback contract",
            "forbidden": "device-calibrated recovery efficacy, automatic bank rollback, complete transport watchdog, bit-accurate RTL, synthesis/post-route timing, FPGA, or board measurement",
        },
    }
    output_json = DEFAULT_JSON if json_path is None else Path(json_path)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles-per-scenario", type=int, default=256)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    payload = run_validation(
        cycles_per_scenario=args.cycles_per_scenario,
        json_path=args.json,
        csv_path=args.csv,
    )
    print(json.dumps(payload["gate_summary"], indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
