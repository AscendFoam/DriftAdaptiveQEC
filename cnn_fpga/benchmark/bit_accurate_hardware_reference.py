"""T5.5.1 formal validation of the packed cycle-accurate RTL golden model."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.runtime.atomic_parameter_bank import (
    AtomicParameterBankConfig,
    deserialize_parameter_image,
)
from cnn_fpga.runtime.bit_accurate_hardware_reference import (
    INPUT_SCHEMA,
    MODEL_SCOPE,
    OUTPUT_SCHEMA,
    STATE_SCHEMA,
    BitAccurateHardwareReference,
    HardwareTraceRecord,
    decode_input_word,
    encode_input_word,
    hardware_reference_contract,
    pack_parameter_bundle,
    pack_parameter_image,
    unpack_parameter_bundle,
    unpack_parameter_image,
)
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTImage,
    ParametricMAPLUTInput,
    ParametricMAPLUTRuntime,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.5.1"
SCHEMA_VERSION = "t5.5.1-bit-accurate-hardware-reference-validation-v1"
PROTOCOL_ID = "PACKED-WORD-TRUE-PIPELINE-ATOMIC-BANK-RTL-GOLDEN-V1"
DEFAULT_ARTIFACT = Path("docs/t5_5_1_bit_accurate_hardware_reference.json")
DEFAULT_SOURCE_DATA = Path(
    "docs/t5_5_1_bit_accurate_hardware_reference_source_data.csv"
)
DEFAULT_TRACE = Path("docs/t5_5_1_bit_accurate_golden_trace.csv")
DEFAULT_BUNDLE = Path("docs/t5_5_1_bit_accurate_parameter_bank.bin")
BANK_IMAGES = Path("docs/t4_2_1_parametric_map_lut_bank_images.json")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T4.2.1": Path("docs/t4_2_1_parametric_map_lut_validation.json"),
    "T4.2.2": Path("docs/t4_2_2_experimental_event_fsm_validation.json"),
    "T4.2.3": Path("docs/t4_2_3_conservative_fallback_validation.json"),
    "T4.2.4": Path("docs/t4_2_4_fast_path_fixed_point_validation.json"),
    "T4.3.2": Path("docs/t4_3_2_atomic_parameter_bank_validation.json"),
    "T4.3.3": Path("docs/t4_3_3_closed_loop_fault_recovery_validation.json"),
    "T5.4.6": Path("docs/t5_4_6_randomized_model_mismatch.json"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/runtime/bit_accurate_hardware_reference.py"),
    Path("cnn_fpga/runtime/parametric_map_lut.py"),
    Path("cnn_fpga/runtime/experimental_event_fsm.py"),
    Path("cnn_fpga/runtime/conservative_fallback.py"),
    Path("cnn_fpga/runtime/atomic_parameter_bank.py"),
    Path("cnn_fpga/benchmark/bit_accurate_hardware_reference.py"),
)


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _parent_pass(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") == "PASS" or payload.get("passed") is True:
        return True
    summary = payload.get("gate_summary")
    if isinstance(summary, Mapping) and summary.get("passed") == summary.get("total"):
        return True
    gates = payload.get("gates")
    if isinstance(gates, Mapping):
        return bool(gates) and all(gates.values())
    if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
        return bool(gates) and all(bool(row.get("passed")) for row in gates)
    return False


def _parent_bindings() -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "machine_pass": _parent_pass(_load_json(path)),
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    ]


def _implementation_bindings() -> list[dict[str, Any]]:
    return [
        {"path": path.as_posix(), "sha256": _sha256(path)}
        for path in IMPLEMENTATION_PATHS
    ]


def load_frozen_images() -> tuple[ParametricMAPLUTImage, ...]:
    payload = _load_json(BANK_IMAGES)
    images = []
    for row in payload["images"]:
        canonical = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        images.append(deserialize_parameter_image(canonical))
    result = tuple(images)
    if len(result) != 8 or [image.active_bank_version for image in result] != list(range(8)):
        raise ValueError("frozen T4.2.1 image registry must contain versions 0..7")
    return result


def _manual_rounded_shift(value: int, bits: int) -> int:
    if bits == 0:
        return value
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, 1 << bits)
    half = 1 << (bits - 1)
    if remainder > half or (remainder == half and quotient % 2 == 1):
        quotient += 1
    return sign * quotient


def exhaustive_map_rows(
    images: Sequence[ParametricMAPLUTImage],
) -> list[dict[str, Any]]:
    rows = []
    for image in images:
        runtime = ParametricMAPLUTRuntime(image)
        config = image.config
        for phase in (0, 1):
            table = image.table_codes[phase]
            for code in range(config.adc_levels):
                decision = runtime.decode_code(
                    ParametricMAPLUTInput(
                        0, code, phase, image.active_bank_version
                    )
                )
                address = code >> config.fraction_bits
                fraction = code & ((1 << config.fraction_bits) - 1)
                numerator = (fraction << 1) + 1
                manual = table[address] + _manual_rounded_shift(
                    (table[address + 1] - table[address]) * numerator,
                    config.fraction_bits + 1,
                )
                manual = max(config.llr_min_code, min(config.llr_max_code, manual))
                rows.append(
                    {
                        "image_version": image.active_bank_version,
                        "phase": phase,
                        "syndrome_code": code,
                        "address": address,
                        "fraction_code": fraction,
                        "runtime_llr_code": decision.llr_code,
                        "manual_llr_code": manual,
                        "runtime_action": decision.logical_action,
                        "manual_action": "X" if phase == 0 and manual < 0 else (
                            "Z" if phase == 1 and manual < 0 else "I"
                        ),
                        "exact": decision.llr_code == manual,
                    }
                )
    return rows


def _input_word(cycle: int) -> int:
    syndrome_x = "e" if 200 <= cycle <= 211 else "g"
    syndrome_z = "g"
    if cycle in (160, 161):
        syndrome_x = "leakage"
    word = encode_input_word(
        syndrome_code=(73 * cycle + 19) % 1024,
        syndrome_x=syndrome_x,
        syndrome_z=syndrome_z,
        quadrature_phase_bit=cycle % 2,
        ood_score_code=255 if cycle == 180 else 16,
        parameter_age_code=cycle % 32,
        reset_ack=cycle == 162,
        observation_valid=cycle != 120,
        deadline_ok=cycle != 140,
    )
    if cycle == 100:
        word ^= 1
    return word


def run_trace(
    images: Sequence[ParametricMAPLUTImage],
    *,
    input_cycles: int,
    commit_apply_cycle: int,
) -> tuple[BitAccurateHardwareReference, dict[str, Any]]:
    reference = BitAccurateHardwareReference(images)
    staging = reference.stage_packed_update(
        pack_parameter_image(images[1]),
        transaction_id="t551-v1",
        selection_key="bank-v1",
        source_window_id=2,
        created_cycle=0,
        apply_cycle=commit_apply_cycle,
    )
    for cycle in range(input_cycles):
        reference.step_word(
            _input_word(cycle),
            safe_boundary=cycle != commit_apply_cycle,
        )
    for _ in range(6):
        reference.step_word(None, safe_boundary=True)
    return reference, staging


def _decode_output(record: HardwareTraceRecord) -> dict[str, int]:
    values, crc_ok, _, _ = OUTPUT_SCHEMA.unpack(int(record.output_word_hex, 16))
    if not crc_ok:
        raise RuntimeError("generated output word failed its CRC")
    return values


def summarize_trace(
    trace: Sequence[HardwareTraceRecord],
    *,
    input_cycles: int,
    commit_apply_cycle: int,
) -> dict[str, Any]:
    outputs = [row for row in trace if row.output_valid]
    by_source = {int(row.output_source_cycle): row for row in outputs}
    latencies = sorted(
        {row.hardware_cycle - int(row.output_source_cycle) for row in outputs}
    )
    commit_rows = [row for row in trace if row.commit_status != "none"]
    old_output = _decode_output(by_source[commit_apply_cycle])
    new_output = _decode_output(by_source[commit_apply_cycle + 1])
    crc_output = _decode_output(by_source[100])
    invalid_output = _decode_output(by_source[120])
    deadline_output = _decode_output(by_source[140])
    ood_output = _decode_output(by_source[180])
    leakage_output = _decode_output(by_source[161])
    reset_ack_output = _decode_output(by_source[162])
    saturated_output = _decode_output(by_source[211])
    return {
        "hardware_cycles": len(trace),
        "input_cycles": input_cycles,
        "output_valid_cycles": len(outputs),
        "warmup_invalid_outputs": sum(
            not row.output_valid for row in trace[:6]
        ),
        "output_source_cycle_min": min(by_source),
        "output_source_cycle_max": max(by_source),
        "output_source_cycles_unique": len(by_source),
        "source_to_output_latencies": latencies,
        "all_output_crc_ok": all(row.output_crc_ok for row in trace),
        "all_state_crc_ok": all(row.state_crc_ok for row in trace),
        "bad_input_crc_source_100_detected": trace[100].input_crc_ok is False,
        "bad_input_crc_output_fault_mask": crc_output["fault_mask"],
        "invalid_observation_output_fault_mask": invalid_output["fault_mask"],
        "deadline_output_fault_mask": deadline_output["fault_mask"],
        "ood_output_fault_mask": ood_output["fault_mask"],
        "persistent_leakage_reset_request": bool(leakage_output["reset_request"]),
        "reset_ack_clears_request": not bool(reset_ack_output["reset_request"]),
        "x_e_run_saturates_at_7": saturated_output["x_e_run"] == 7,
        "commit_rows": [
            {
                "cycle": row.hardware_cycle,
                "status": row.commit_status,
                "reason": row.commit_reason,
                "active_version": row.active_version,
            }
            for row in commit_rows
        ],
        "deferred_commit_cycle": next(
            row.hardware_cycle for row in commit_rows if row.commit_status == "deferred"
        ),
        "committed_cycle": next(
            row.hardware_cycle for row in commit_rows if row.commit_status == "committed"
        ),
        "inflight_old_version_at_source_4000": old_output["active_version"],
        "post_commit_version_at_source_4001": new_output["active_version"],
        "active_versions_monotonic": all(
            left.active_version <= right.active_version
            for left, right in zip(trace, trace[1:])
        ),
        "final_trace_sha256": trace[-1].trace_chain_sha256,
    }


def _input_crc_audit() -> dict[str, Any]:
    nominal = encode_input_word(
        syndrome_code=517,
        syndrome_x="e",
        syndrome_z="leakage",
        quadrature_phase_bit=1,
        ood_score_code=193,
        parameter_age_code=65535,
        reset_ack=True,
        observation_valid=True,
        deadline_ok=True,
    )
    decoded = decode_input_word(nominal)
    rejected = 0
    for bit in range(INPUT_SCHEMA.word_bits):
        if not decode_input_word(nominal ^ (1 << bit)).input_crc_ok:
            rejected += 1
    return {
        "word_bits": INPUT_SCHEMA.word_bits,
        "nominal_crc_ok": decoded.input_crc_ok,
        "single_bit_flips_tested": INPUT_SCHEMA.word_bits,
        "single_bit_flips_detected": rejected,
    }


def _binary_audit(images: Sequence[ParametricMAPLUTImage]) -> dict[str, Any]:
    payloads = [pack_parameter_image(image) for image in images]
    bundle = pack_parameter_bundle(images)
    first = payloads[0]
    corrupted_rejected = 0
    for index in range(len(first)):
        changed = bytearray(first)
        changed[index] ^= 1
        try:
            unpack_parameter_image(bytes(changed))
        except ValueError:
            corrupted_rejected += 1
    prefixes_rejected = 0
    for length in range(len(first)):
        try:
            unpack_parameter_image(first[:length])
        except ValueError:
            prefixes_rejected += 1
    return {
        "image_count": len(images),
        "bytes_per_image": len(first),
        "all_roundtrip_exact": all(
            unpack_parameter_image(payload) == image
            and pack_parameter_image(unpack_parameter_image(payload)) == payload
            for payload, image in zip(payloads, images)
        ),
        "unique_image_sha256": len({hashlib.sha256(payload).hexdigest() for payload in payloads}),
        "proper_prefixes_tested": len(first),
        "proper_prefixes_rejected": prefixes_rejected,
        "single_byte_corruptions_tested": len(first),
        "single_byte_corruptions_rejected": corrupted_rejected,
        "bundle_bytes": len(bundle),
        "bundle_roundtrip_exact": unpack_parameter_bundle(bundle) == tuple(images),
        "bundle_sha256": hashlib.sha256(bundle).hexdigest(),
    }


def _online_ast_audit() -> dict[str, Any]:
    source = inspect.getsource(BitAccurateHardwareReference.step_word)
    tree = ast.parse(textwrap.dedent(source))
    division_nodes = sum(
        isinstance(node, (ast.Div, ast.FloorDiv)) for node in ast.walk(tree)
    )
    forbidden_calls = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"float", "exp", "log", "sqrt"}
        }
    )
    return {
        "division_nodes": division_nodes,
        "forbidden_numeric_calls": forbidden_calls,
        "online_input_fields": [field.name for field in INPUT_SCHEMA.fields],
        "truth_tokens_present": any(
            token in source for token in ("truth", "hidden_state", "drift_state")
        ),
    }


def _trace_chain_valid(trace: Sequence[HardwareTraceRecord]) -> bool:
    chain = bytes(32)
    for record in trace:
        chain = hashlib.sha256(
            chain
            + json.dumps(
                record.unsigned_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).digest()
        if chain.hex() != record.trace_chain_sha256:
            return False
    return True


@dataclass(frozen=True)
class HardwareReferenceConfig:
    input_cycles: int = 4110
    commit_apply_cycle: int = 4000

    def __post_init__(self) -> None:
        if self.input_cycles < 4102:
            raise ValueError("formal trace must extend beyond the bank commit")
        if self.commit_apply_cycle != 4000:
            raise ValueError("formal commit boundary is frozen at cycle 4000")


def _compute_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    contract = report["contract"]
    binary = report["binary_parameter_audit"]
    crc = report["input_crc_audit"]
    exhaustive = report["exhaustive_map_summary"]
    trace = report["trace_summary"]
    repeat = report["repeatability"]
    ast_audit = report["online_ast_audit"]
    hardware = report["claim_boundary"]["hardware_fields"]
    return {
        "all_parent_artifacts_are_live_machine_passes": all(
            row["machine_pass"] for row in report["parent_bindings"]
        ),
        "selected_word_widths_rounding_saturation_and_latency_are_frozen": (
            contract["input_word"]["word_bits"] == 58
            and contract["output_word"]["word_bits"] == 118
            and contract["state_word"]["word_bits"] == 232
            and contract["source_to_output_cycles"] == 6
            and contract["initiation_interval_cycles"] == 1
            and contract["parameter_image"]["logical_llr_bits"] == 22
            and contract["parameter_image"]["table_container_bits"] == 24
        ),
        "all_eight_binary_parameter_images_and_bundle_roundtrip_exactly": (
            binary["image_count"] == 8
            and binary["all_roundtrip_exact"]
            and binary["unique_image_sha256"] == 8
            and binary["bundle_roundtrip_exact"]
        ),
        "binary_parameter_image_prefixes_and_byte_corruptions_fail_closed": (
            binary["proper_prefixes_tested"] == binary["proper_prefixes_rejected"]
            and binary["single_byte_corruptions_tested"]
            == binary["single_byte_corruptions_rejected"]
        ),
        "input_crc_detects_every_single_bit_flip": (
            crc["nominal_crc_ok"]
            and crc["single_bit_flips_tested"] == crc["word_bits"]
            and crc["single_bit_flips_detected"] == crc["word_bits"]
        ),
        "all_16384_map_codes_match_independent_integer_reconstruction": (
            exhaustive["rows"] == 16_384
            and exhaustive["llr_code_mismatch_count"] == 0
            and exhaustive["action_mismatch_count"] == 0
        ),
        "true_pipeline_has_exact_six_cycle_latency_and_ii_one": (
            trace["output_valid_cycles"] == trace["input_cycles"]
            and trace["output_source_cycles_unique"] == trace["input_cycles"]
            and trace["output_source_cycle_min"] == 0
            and trace["output_source_cycle_max"] == trace["input_cycles"] - 1
            and trace["source_to_output_latencies"] == [6]
            and trace["warmup_invalid_outputs"] == 6
        ),
        "output_and_state_words_have_valid_crc_on_every_cycle": (
            trace["all_output_crc_ok"] and trace["all_state_crc_ok"]
        ),
        "atomic_commit_defers_then_switches_at_safe_boundary": (
            trace["deferred_commit_cycle"] == 4000
            and trace["committed_cycle"] == 4001
            and trace["active_versions_monotonic"]
        ),
        "inflight_old_image_and_post_commit_new_image_are_both_latched": (
            trace["inflight_old_version_at_source_4000"] == 0
            and trace["post_commit_version_at_source_4001"] == 1
        ),
        "crc_invalid_deadline_ood_and_leakage_faults_are_not_silent": (
            trace["bad_input_crc_source_100_detected"]
            and trace["bad_input_crc_output_fault_mask"] != 0
            and trace["invalid_observation_output_fault_mask"] != 0
            and trace["deadline_output_fault_mask"] != 0
            and trace["ood_output_fault_mask"] != 0
            and trace["persistent_leakage_reset_request"]
            and trace["reset_ack_clears_request"]
        ),
        "event_counter_saturates_without_wrap": trace["x_e_run_saturates_at_7"],
        "trace_chain_is_deterministic_under_full_rerun": (
            repeat["trace_hash_equal"]
            and repeat["trace_rows_equal"]
            and repeat["staging_equal"]
            and report["golden_trace"]["chain_valid"]
            and report["golden_trace"]["final_chain_sha256"]
            == trace["final_trace_sha256"]
        ),
        "online_step_is_integer_only_and_truth_free": (
            ast_audit["division_nodes"] == 0
            and ast_audit["forbidden_numeric_calls"] == []
            and ast_audit["truth_tokens_present"] is False
        ),
        "trace_schema_and_parameter_bank_artifacts_are_byte_bound": all(
            report[name].get("sha256")
            for name in ("golden_trace", "parameter_bundle", "source_data")
        ),
        "rtl_synthesis_timing_resource_and_board_claims_remain_closed": (
            hardware["rtl_generated"] is False
            and hardware["synthesized"] is False
            and hardware["board_measured"] is False
            and all(
                hardware[name] is None
                for name in (
                    "fmax_mhz",
                    "target_lut_count",
                    "target_ff_count",
                    "target_bram_count",
                    "target_dsp_count",
                )
            )
        ),
    }


def _contract_view(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "task_id",
        "protocol_id",
        "status",
        "verdict",
        "config",
        "parent_bindings",
        "implementation_bindings",
        "contract",
        "binary_parameter_audit",
        "input_crc_audit",
        "exhaustive_map_summary",
        "trace_summary",
        "trace_samples",
        "online_ast_audit",
        "repeatability",
        "claim_boundary",
        "golden_trace",
        "parameter_bundle",
        "source_data",
        "gates",
        "gate_summary",
    )
    return {key: report[key] for key in keys}


def validate_artifact(
    report: Mapping[str, Any], *, check_files: bool = True
) -> tuple[str, ...]:
    errors = []
    if report.get("schema_version") != SCHEMA_VERSION or report.get("task_id") != TASK_ID:
        errors.append("schema/task mismatch")
    try:
        gates = _compute_gates(report)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"gate recomputation failed: {exc}")
        gates = {}
    if gates != report.get("gates"):
        errors.append("stored gates differ from semantic recomputation")
    summary = {"passed": sum(gates.values()), "total": len(gates)}
    if report.get("gate_summary") != summary:
        errors.append("gate summary mismatch")
    status = "PASS" if gates and all(gates.values()) else "FAIL"
    if report.get("status") != status:
        errors.append("status mismatch")
    expected_verdict = (
        "BIT_ACCURATE_PYTHON_RTL_GOLDEN_FROZEN_HARDWARE_UNMEASURED"
        if status == "PASS"
        else "INCOMPLETE_BIT_ACCURATE_REFERENCE"
    )
    if report.get("verdict") != expected_verdict:
        errors.append("verdict mismatch")
    if report.get("contract_sha256") != _canonical_sha256(_contract_view(report)):
        errors.append("contract hash mismatch")
    if check_files:
        for row in report.get("parent_bindings", ()): 
            if row["sha256"] != _sha256(row["path"]):
                errors.append(f"parent hash mismatch: {row['task_id']}")
        for row in report.get("implementation_bindings", ()): 
            if row["sha256"] != _sha256(row["path"]):
                errors.append(f"implementation hash mismatch: {row['path']}")
        for name in ("golden_trace", "parameter_bundle", "source_data"):
            binding = report.get(name, {})
            path = binding.get("path", "")
            if not _repo_path(path).is_file():
                errors.append(f"{name} file missing")
            elif binding.get("sha256") != _sha256(path):
                errors.append(f"{name} hash mismatch")
    return tuple(errors)


def _write_trace(path: Path, trace: Sequence[HardwareTraceRecord]) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.to_dict() for row in trace]
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=target.parent, encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=target.parent) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=target.parent, encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def source_rows(
    report: Mapping[str, Any], exhaustive_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append(row_type: str, item_id: str, metric: str, value: Any, detail: Any) -> None:
        rows.append(
            {
                "row_type": row_type,
                "item_id": item_id,
                "metric": metric,
                "value": value,
                "detail_json": json.dumps(detail, sort_keys=True),
            }
        )

    for row in report["parent_bindings"]:
        append("parent_binding", row["task_id"], "machine_pass", int(row["machine_pass"]), row)
    for row in report["implementation_bindings"]:
        append("implementation_binding", row["path"], "sha256_bound", 1, row)
    for schema_name in ("input_word", "output_word", "state_word"):
        schema = report["contract"][schema_name]
        for field in schema["fields"]:
            append("word_field", f"{schema_name}:{field['name']}", "width", field["width"], field)
    for row in report["parameter_image_rows"]:
        append("parameter_image", f"v{row['version']}", "packed_bytes", row["packed_bytes"], row)
    for row in exhaustive_rows:
        append(
            "exhaustive_map_code",
            f"v{row['image_version']}:p{row['phase']}:c{row['syndrome_code']}",
            "llr_code",
            row["runtime_llr_code"],
            row,
        )
    for row in report["trace_samples"]:
        append("trace_sample", f"cycle-{row['hardware_cycle']}", "output_valid", int(row["output_valid"]), row)
    for name, passed in report["gates"].items():
        append("gate", name, "passed", int(passed), {"gate": name, "passed": passed})
    return rows


def _write_source(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=target.parent, encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def run_bit_accurate_hardware_reference(
    config: HardwareReferenceConfig | None = None,
    *,
    artifact_path: Path = DEFAULT_ARTIFACT,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
    trace_path: Path = DEFAULT_TRACE,
    bundle_path: Path = DEFAULT_BUNDLE,
) -> dict[str, Any]:
    actual = config or HardwareReferenceConfig()
    images = load_frozen_images()
    binary_audit = _binary_audit(images)
    input_crc = _input_crc_audit()
    exhaustive = exhaustive_map_rows(images)
    reference, staging = run_trace(
        images,
        input_cycles=actual.input_cycles,
        commit_apply_cycle=actual.commit_apply_cycle,
    )
    repeat, repeat_staging = run_trace(
        images,
        input_cycles=actual.input_cycles,
        commit_apply_cycle=actual.commit_apply_cycle,
    )
    trace = reference.trace
    trace_summary = summarize_trace(
        trace,
        input_cycles=actual.input_cycles,
        commit_apply_cycle=actual.commit_apply_cycle,
    )
    _write_trace(trace_path, trace)
    bundle = pack_parameter_bundle(images)
    _atomic_bytes(bundle_path, bundle)
    sample_cycles = (0, 5, 6, 100, 106, 120, 126, 140, 146, 160, 167, 168, 211, 217, 4000, 4001, 4006, 4007, len(trace) - 1)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING",
        "verdict": "PENDING",
        "config": asdict(actual),
        "parent_bindings": _parent_bindings(),
        "implementation_bindings": _implementation_bindings(),
        "contract": hardware_reference_contract(),
        "binary_parameter_audit": binary_audit,
        "parameter_image_rows": [
            {
                "version": image.active_bank_version,
                "packed_bytes": len(pack_parameter_image(image)),
                "packed_sha256": hashlib.sha256(pack_parameter_image(image)).hexdigest(),
                "image_crc32": image.image_crc32,
                "image_sha256": image.image_sha256,
            }
            for image in images
        ],
        "input_crc_audit": input_crc,
        "exhaustive_map_summary": {
            "rows": len(exhaustive),
            "llr_code_mismatch_count": sum(not row["exact"] for row in exhaustive),
            "action_mismatch_count": sum(row["runtime_action"] != row["manual_action"] for row in exhaustive),
            "rows_sha256": _canonical_sha256(exhaustive),
        },
        "trace_summary": trace_summary,
        "trace_samples": [trace[cycle].to_dict() for cycle in sample_cycles],
        "online_ast_audit": _online_ast_audit(),
        "repeatability": {
            "trace_hash_equal": reference.final_trace_sha256 == repeat.final_trace_sha256,
            "trace_rows_equal": [row.to_dict() for row in trace]
            == [row.to_dict() for row in repeat.trace],
            "staging_equal": staging == repeat_staging,
            "first_trace_sha256": reference.final_trace_sha256,
            "repeat_trace_sha256": repeat.final_trace_sha256,
        },
        "claim_boundary": {
            "allowed": "packed-word cycle-accurate deterministic Python RTL golden reference",
            "forbidden": "RTL generated, synthesized, timing closed, routed, device calibrated, transported, FPGA or board measured",
            "hardware_fields": hardware_reference_contract()["hardware_fields"],
        },
        "golden_trace": {
            "path": Path(trace_path).as_posix(),
            "rows": len(trace),
            "sha256": _sha256(trace_path),
            "final_chain_sha256": reference.final_trace_sha256,
            "chain_valid": _trace_chain_valid(trace),
        },
        "parameter_bundle": {
            "path": Path(bundle_path).as_posix(),
            "bytes": len(bundle),
            "sha256": _sha256(bundle_path),
            "roundtrip_exact": unpack_parameter_bundle(bundle) == images,
        },
        "source_data": {
            "path": Path(source_data_path).as_posix(),
            "rows": 0,
            "rows_sha256": None,
            "sha256": None,
        },
    }
    report["gates"] = _compute_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
    }
    rows = source_rows(report, exhaustive)
    _write_source(source_data_path, rows)
    report["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "rows": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "sha256": _sha256(source_data_path),
    }
    report["gates"] = _compute_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    report["verdict"] = (
        "BIT_ACCURATE_PYTHON_RTL_GOLDEN_FROZEN_HARDWARE_UNMEASURED"
        if report["status"] == "PASS"
        else "INCOMPLETE_BIT_ACCURATE_REFERENCE"
    )
    rows = source_rows(report, exhaustive)
    _write_source(source_data_path, rows)
    report["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "rows": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "sha256": _sha256(source_data_path),
    }
    report["contract_sha256"] = _canonical_sha256(_contract_view(report))
    errors = validate_artifact(report)
    if errors:
        raise RuntimeError("invalid T5.5.1 artifact: " + "; ".join(errors))
    _atomic_json(artifact_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    report = run_bit_accurate_hardware_reference(
        artifact_path=args.artifact,
        source_data_path=args.source_data,
        trace_path=args.trace,
        bundle_path=args.bundle,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["verdict"],
                "gate_summary": report["gate_summary"],
                "trace_rows": report["golden_trace"]["rows"],
                "source_rows": report["source_data"]["rows"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_BUNDLE",
    "DEFAULT_SOURCE_DATA",
    "DEFAULT_TRACE",
    "HardwareReferenceConfig",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "TASK_ID",
    "exhaustive_map_rows",
    "load_frozen_images",
    "run_bit_accurate_hardware_reference",
    "run_trace",
    "summarize_trace",
    "validate_artifact",
]
