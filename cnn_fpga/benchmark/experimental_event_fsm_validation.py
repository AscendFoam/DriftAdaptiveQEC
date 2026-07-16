"""T4.2.2 production validation for observed-event FSM and frame actions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from cnn_fpga.benchmark.parametric_map_lut_validation import (
    registered_parameter_profiles,
)
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.experimental_event_fsm import (
    EVENT_MODES,
    FALLBACK,
    HOLD,
    MODEL_SCOPE,
    NORMAL,
    RESET_REQUEST,
    X_RECOVERY,
    Z_RECOVERY,
    ExperimentalEventFSM,
    ExperimentalEventFSMConfig,
    ExperimentalEventInput,
)
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTImage,
    ParametricMAPLUTInput,
    ParametricMAPLUTRuntime,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "docs" / "t4_2_2_experimental_event_fsm_validation.json"
DEFAULT_CSV = ROOT / "docs" / "t4_2_2_experimental_event_fsm_source_data.csv"
SCHEMA_VERSION = "t4.2.2-experimental-event-fsm-validation-v1"
SCENARIOS = (
    "nominal_frame",
    "x_recovery_saturation",
    "z_recovery_saturation",
    "dual_e_phase_tie",
    "leakage_reset_handshake",
    "health_fault_recovery",
    "counter_saturation",
    "bank_version_switch",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/experimental_event_fsm.py",
        "cnn_fpga/runtime/parametric_map_lut.py",
        "cnn_fpga/decoder/parametric_map_lut.py",
        "cnn_fpga/benchmark/experimental_event_fsm_validation.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _images() -> tuple[ParametricMAPLUTImage, ...]:
    config = ParametricMAPLUTConfig()
    profiles = registered_parameter_profiles(config)
    return tuple(
        compile_parametric_map_lut(params, active_bank_version=index, config=config)
        for index, (params, _) in enumerate(profiles)
    )


def _action_codes(
    images: Sequence[ParametricMAPLUTImage],
) -> dict[tuple[int, int, bool], int]:
    result: dict[tuple[int, int, bool], int] = {}
    for image in images:
        runtime = ParametricMAPLUTRuntime(image)
        for phase in (0, 1):
            for flip in (False, True):
                for code in range(image.config.adc_levels):
                    decision = runtime.decode_code(
                        ParametricMAPLUTInput(0, code, phase, image.active_bank_version)
                    )
                    if decision.logical_flip is flip:
                        result[(image.active_bank_version, phase, flip)] = code
                        break
    return result


def _scenario_input(scenario: str, offset: int) -> dict[str, Any]:
    base: dict[str, Any] = {
        "phase": offset % 2,
        "flip": offset % 3 == 0,
        "syndrome_x": "g",
        "syndrome_z": "g",
        "reset_ack": False,
        "valid": True,
        "crc_ok": True,
        "parameter_fresh": True,
        "deadline_ok": True,
        # Bank 0 is the deliberately weak static profile and its quantized support
        # is all-I.  Use bank 1 for action-path scenarios; the switch scenario
        # still traverses every registered version, including the all-I bank.
        "image_index": 1,
    }
    if scenario == "nominal_frame":
        return base
    if scenario == "x_recovery_saturation":
        base.update(
            phase=0,
            syndrome_x="e" if offset % 16 < 10 else "g",
            flip=offset % 4 == 0,
        )
        return base
    if scenario == "z_recovery_saturation":
        base.update(
            phase=1,
            syndrome_z="e" if offset % 16 < 10 else "g",
            flip=offset % 5 == 0,
        )
        return base
    if scenario == "dual_e_phase_tie":
        observed = "e" if offset % 12 < 8 else "g"
        base.update(syndrome_x=observed, syndrome_z=observed, flip=offset % 4 == 1)
        return base
    if scenario == "leakage_reset_handshake":
        phase = offset % 12
        if phase in (0, 1):
            base.update(syndrome_x="leakage", syndrome_z="leakage")
        if phase == 3:
            base["reset_ack"] = True
        base["flip"] = True
        return base
    if scenario == "health_fault_recovery":
        position = offset % 24
        if position == 0:
            base["valid"] = False
        elif position == 4:
            base["crc_ok"] = False
        elif position == 8:
            base["parameter_fresh"] = False
        elif position == 12:
            base["deadline_ok"] = False
        base["flip"] = position in (0, 4, 8, 12)
        return base
    if scenario == "counter_saturation":
        if offset < 32:
            base.update(syndrome_x="e", syndrome_z="e")
        else:
            base.update(syndrome_x="leakage", syndrome_z="leakage", flip=True)
        return base
    if scenario == "bank_version_switch":
        base.update(image_index=min(offset // 16, 7), flip=offset % 2 == 0)
        return base
    raise ValueError(f"unknown scenario {scenario!r}")


def _run_scenarios(
    *, cycles_per_scenario: int = 128
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if cycles_per_scenario < 32:
        raise ValueError("cycles_per_scenario must be at least 32")
    images = _images()
    runtimes = tuple(ParametricMAPLUTRuntime(image) for image in images)
    codes = _action_codes(images)
    rows: list[dict[str, Any]] = []
    transition_pairs: set[tuple[str, str]] = set()
    modulo_wrap_count = 0
    for scenario in SCENARIOS:
        fsm = ExperimentalEventFSM()
        previous_mode = fsm.state.mode
        previous_phase = (fsm.state.phase_frame_x_code, fsm.state.phase_frame_z_code)
        for offset in range(cycles_per_scenario):
            spec = _scenario_input(scenario, offset)
            runtime = runtimes[spec["image_index"]]
            image = runtime.image
            event_cycle = fsm.config.start_event_cycle + offset
            source_cycle = event_cycle - image.config.pipeline_latency_cycles
            phase = spec["phase"]
            requested_flip = bool(spec["flip"])
            action_key = (image.active_bank_version, phase, requested_flip)
            if action_key not in codes:
                available = sorted(
                    flip
                    for version, registered_phase, flip in codes
                    if version == image.active_bank_version and registered_phase == phase
                )
                if len(available) != 1:
                    raise RuntimeError("registered bank/phase has no deterministic action")
                requested_flip = available[0]
                action_key = (image.active_bank_version, phase, requested_flip)
            code = codes[action_key]
            map_decision = runtime.decode_code(
                ParametricMAPLUTInput(
                    source_cycle, code, phase, image.active_bank_version
                )
            )
            event = ExperimentalEventInput(
                cycle_index=event_cycle,
                syndrome_x=spec["syndrome_x"],
                syndrome_z=spec["syndrome_z"],
                quadrature_phase_bit=phase,
                map_decision=map_decision,
                active_bank_version=image.active_bank_version,
                reset_ack=spec["reset_ack"],
                valid=spec["valid"],
                crc_ok=spec["crc_ok"],
                parameter_fresh=spec["parameter_fresh"],
                deadline_ok=spec["deadline_ok"],
            )
            action = fsm.step(event)
            transition_pairs.add((previous_mode, action.mode))
            current_phase = (action.phase_frame_x_code, action.phase_frame_z_code)
            if (
                (action.phase_frame_delta_x_code and current_phase[0] < previous_phase[0])
                or (action.phase_frame_delta_z_code and current_phase[1] < previous_phase[1])
            ):
                modulo_wrap_count += 1
            previous_mode = action.mode
            previous_phase = current_phase
            row = {
                "scenario": scenario,
                "scenario_offset": offset,
                "source_cycle": action.source_cycle,
                "map_valid_cycle": map_decision.valid_cycle,
                "hardware_action_cycle": action.action_cycle,
                "syndrome_x": event.syndrome_x,
                "syndrome_z": event.syndrome_z,
                "phase_bit": phase,
                "map_llr_code": map_decision.llr_code,
                "map_logical_action": map_decision.logical_action,
                "map_logical_flip": int(map_decision.logical_flip),
                "requested_map_logical_flip": int(bool(spec["flip"])),
                "active_bank_version": action.active_bank_version,
                "mode": action.mode,
                "reason": action.reason,
                "correction_enable": int(action.correction_enable),
                "reset_request": int(action.reset_request),
                "reset_ack": int(event.reset_ack),
                "map_action_inhibited": int(action.map_action_inhibited),
                "pauli_frame_delta_x": int(action.pauli_frame_delta_x),
                "pauli_frame_delta_z": int(action.pauli_frame_delta_z),
                "phase_frame_delta_x_code": action.phase_frame_delta_x_code,
                "phase_frame_delta_z_code": action.phase_frame_delta_z_code,
                "pauli_frame_x": int(action.pauli_frame_x),
                "pauli_frame_z": int(action.pauli_frame_z),
                "phase_frame_x_code": action.phase_frame_x_code,
                "phase_frame_z_code": action.phase_frame_z_code,
                "x_e_run": action.x_e_run,
                "z_e_run": action.z_e_run,
                "leakage_run": action.leakage_run,
                "leakage_clean_run": action.leakage_clean_run,
                "health_good_run": action.health_good_run,
                "reset_wait_run": action.reset_wait_run,
                "valid": int(event.valid),
                "crc_ok": int(event.crc_ok),
                "parameter_fresh": int(event.parameter_fresh),
                "deadline_ok": int(event.deadline_ok),
                "map_image_sha256": action.map_image_sha256,
            }
            rows.append(row)
    diagnostics = {
        "transition_pairs": sorted(f"{left}->{right}" for left, right in transition_pairs),
        "modulo_phase_wrap_count": modulo_wrap_count,
        "image_sha256s": [image.image_sha256 for image in images],
        "cycles_per_scenario": cycles_per_scenario,
    }
    return rows, diagnostics


def _rows_sha256(rows: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _negative_audit() -> dict[str, bool]:
    image = _images()[1]
    runtime = ParametricMAPLUTRuntime(image)
    code = _action_codes((image,))[(1, 0, True)]
    decision = runtime.decode_code(ParametricMAPLUTInput(0, code, 0, 1))
    valid = ExperimentalEventInput(5, "g", "g", 0, decision, 1)
    cases = {
        "cycle_gap_rejected": replace(valid, cycle_index=6),
        "phase_mismatch_rejected": replace(valid, quadrature_phase_bit=1),
        "version_mismatch_rejected": replace(valid, active_bank_version=2),
        "logical_action_mismatch_rejected": replace(
            valid, map_decision=replace(decision, logical_action="I")
        ),
        "llr_sign_mismatch_rejected": replace(
            valid, map_decision=replace(decision, logical_flip=False, logical_action="I")
        ),
    }
    results = {}
    for name, event in cases.items():
        fsm = ExperimentalEventFSM()
        before = fsm.state
        try:
            fsm.step(event)
        except (TypeError, ValueError):
            results[name] = fsm.state == before and fsm.history == ()
        else:
            results[name] = False
    return results


def _resource_contract(config: ExperimentalEventFSMConfig) -> dict[str, Any]:
    state_bits = (
        3
        + 6 * config.counter_bits
        + 2
        + 2 * config.phase_frame_bits
        + 16
    )
    return {
        "identity": "exact_state_and_action_word_proxy_not_synthesis",
        "mode_encoding_bits": 3,
        "saturating_counter_count": 6,
        "counter_bits_each": config.counter_bits,
        "pauli_frame_bits": 2,
        "phase_frame_bits_each": config.phase_frame_bits,
        "bank_version_bits_assumed": 16,
        "minimum_live_state_bits": state_bits,
        "event_action_register_cycles": config.event_action_latency_cycles,
        "map_plus_event_worst_case_latency_cycles": (
            config.map_pipeline_latency_cycles + config.event_action_latency_cycles
        ),
        "initiation_interval_cycles": 1,
        "target_lut_count": None,
        "target_ff_count": None,
        "target_bram_count": None,
        "target_dsp_count": None,
        "fmax_mhz": None,
        "rtl_measured": False,
        "board_measured": False,
    }


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_validation(
    *,
    cycles_per_scenario: int = 128,
    json_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    rows, diagnostics = _run_scenarios(cycles_per_scenario=cycles_per_scenario)
    repeated_rows, repeated_diagnostics = _run_scenarios(
        cycles_per_scenario=cycles_per_scenario
    )
    deterministic = (
        _rows_sha256(rows) == _rows_sha256(repeated_rows)
        and diagnostics == repeated_diagnostics
    )
    modes = {row["mode"] for row in rows}
    reasons = {row["reason"] for row in rows}
    safe_rows = [row for row in rows if row["mode"] in (HOLD, RESET_REQUEST, FALLBACK)]
    safe_flips = [row for row in safe_rows if row["map_logical_flip"]]
    version_rows = [row for row in rows if row["scenario"] == "bank_version_switch"]
    version_sequence = [row["active_bank_version"] for row in version_rows]
    expected_health_fragments = {
        "health_fault:valid",
        "health_fault:crc",
        "health_fault:fresh",
        "health_fault:deadline",
    }
    required_transitions = {
        "normal->x_recovery",
        "normal->z_recovery",
        "normal->hold",
        "hold->reset_request",
        "reset_request->hold",
        "hold->normal",
        "normal->fallback",
        "fallback->normal",
    }
    negatives = _negative_audit()
    config = ExperimentalEventFSMConfig()
    resources = _resource_contract(config)
    forbidden_columns = {
        "truth",
        "logical_truth",
        "hidden_state",
        "drift_state",
        "recovery_depth",
    }
    gates = {
        "all_eight_registered_scenarios_are_complete": len(rows)
        == len(SCENARIOS) * cycles_per_scenario
        and {row["scenario"] for row in rows} == set(SCENARIOS),
        "all_six_modes_are_exercised": modes == set(EVENT_MODES),
        "all_required_mode_transitions_are_exercised": required_transitions.issubset(
            set(diagnostics["transition_pairs"])
        ),
        "g_e_and_leakage_observations_are_exercised": {
            row["syndrome_x"] for row in rows
        }
        | {row["syndrome_z"] for row in rows}
        == {"g", "e", "leakage"},
        "all_six_counters_reach_three_bit_saturation": all(
            max(row[name] for row in rows) == 7
            for name in (
                "x_e_run",
                "z_e_run",
                "leakage_run",
                "leakage_clean_run",
                "health_good_run",
                "reset_wait_run",
            )
        ),
        "both_phase_tie_break_directions_are_exercised": (
            "both_e_runs_phase_x_priority" in reasons
            and "both_e_runs_phase_z_priority" in reasons
        ),
        "reset_request_is_sticky_and_acknowledged": (
            "reset_request_sticky_until_ack" in reasons
            and "reset_acknowledged_post_reset_hold" in reasons
        ),
        "all_health_fault_reasons_and_clear_hysteresis_are_exercised": (
            expected_health_fragments.issubset(reasons)
            and "fallback_clear_hysteresis" in reasons
        ),
        "safe_modes_disable_correction": all(
            row["correction_enable"] == 0 for row in safe_rows
        ),
        "safe_modes_inhibit_every_pending_map_flip": safe_flips
        and all(row["map_action_inhibited"] == 1 for row in safe_flips),
        "safe_modes_never_mutate_frames": all(
            row["pauli_frame_delta_x"] == 0
            and row["pauli_frame_delta_z"] == 0
            and row["phase_frame_delta_x_code"] == 0
            and row["phase_frame_delta_z_code"] == 0
            for row in safe_rows
        ),
        "pauli_and_phase_frames_update_on_both_axes": (
            any(row["pauli_frame_delta_x"] for row in rows)
            and any(row["pauli_frame_delta_z"] for row in rows)
            and diagnostics["modulo_phase_wrap_count"] > 0
        ),
        "map_plus_event_latency_is_exactly_six_cycles": all(
            row["hardware_action_cycle"] - row["source_cycle"] == 6 for row in rows
        ),
        "event_action_pipeline_has_initiation_interval_one": all(
            [row["hardware_action_cycle"] for row in rows if row["scenario"] == scenario]
            == list(range(6, 6 + cycles_per_scenario))
            for scenario in SCENARIOS
        ),
        "active_bank_versions_advance_monotonically_without_mix": version_sequence
        == sorted(version_sequence)
        and set(version_sequence) == set(range(8)),
        "map_image_hash_matches_selected_version": all(
            row["map_image_sha256"]
            == diagnostics["image_sha256s"][row["active_bank_version"]]
            for row in rows
        ),
        "negative_alignment_and_action_paths_are_transactional": all(
            negatives.values()
        ),
        "replay_is_bit_deterministic": deterministic,
        "source_data_contains_no_truth_or_hidden_fields": not (
            set(rows[0]) & forbidden_columns
        )
        and not any(
            token in name for name in rows[0] for token in ("truth", "hidden", "drift")
        ),
        "resource_and_timing_fields_remain_non_measured": (
            resources["target_lut_count"] is None
            and resources["target_ff_count"] is None
            and resources["fmax_mhz"] is None
            and not resources["rtl_measured"]
            and not resources["board_measured"]
        ),
    }
    output_csv = DEFAULT_CSV if csv_path is None else Path(csv_path)
    _write_csv(rows, output_csv)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "task_id": "T4.2.2",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "config": asdict(config),
        "scenarios": list(SCENARIOS),
        "rows": len(rows),
        "mode_counts": {
            mode: sum(row["mode"] == mode for row in rows) for mode in EVENT_MODES
        },
        "reason_counts": {
            reason: sum(row["reason"] == reason for row in rows)
            for reason in sorted(reasons)
        },
        "diagnostics": diagnostics,
        "negative_audit": negatives,
        "resource_contract": resources,
        "source_data": {
            "path": str(output_csv.relative_to(ROOT)).replace("\\", "/"),
            "rows": len(rows),
            "sha256": _sha256_file(output_csv),
            "canonical_rows_sha256": _rows_sha256(rows),
        },
        "online_contract": {
            "input_fields": list(ExperimentalEventInput.__dataclass_fields__),
            "step_signature": list(inspect.signature(ExperimentalEventFSM.step).parameters),
            "model_scope": MODEL_SCOPE,
            "hidden_truth_inputs": [],
        },
        "gate_summary": {
            "passed": sum(gates.values()),
            "failed": len(gates) - sum(gates.values()),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": "observed-only six-mode integer event/frame software contract connected to version-bound MAP decisions",
            "forbidden": "device-calibrated recovery efficacy, complete conservative fallback policy, bit-accurate RTL, synthesis/post-route timing, FPGA, or board measurement",
        },
    }
    output_json = DEFAULT_JSON if json_path is None else Path(json_path)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles-per-scenario", type=int, default=128)
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
