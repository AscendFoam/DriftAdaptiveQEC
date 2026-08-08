"""T6.2.1 structural and cycle-exact audit of the production RTL shell."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

from cnn_fpga.benchmark.rtl_fast_path_equivalence import discover_tools
from cnn_fpga.runtime.production_fast_path_management import (
    MANAGEMENT_SIGNAL_DEFAULTS,
    ProductionFastPathManagementReference,
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
    crc32_table_words,
)


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
TOP = ROOT / "cnn_fpga/rtl/gkp_fast_path_production_top.sv"
DRIVER = ROOT / "cnn_fpga/rtl/production_management_cxxrtl_driver.cc"
DEFAULT_BUILD = ROOT / "build/t6_2_1_production_rtl"
DEFAULT_JSON = ROOT / "docs/t6_2_1_production_rtl_audit.json"
DEFAULT_CSV = ROOT / "docs/t6_2_1_production_rtl_audit_source_data.csv"
VERDICT = "PASS_PRODUCTION_RTL_SHELL_READY_FOR_T6_2_2_LONG_TRACE"


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
    input_text: str | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        env=dict(env),
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def build_cxxrtl(build_dir: Path) -> dict[str, Any]:
    tools = discover_tools()
    build_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = build_dir / "temp"
    cache_dir = build_dir / "yowasp_cache"
    temp_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    model = build_dir / "gkp_fast_path_production_model.cc"
    executable = build_dir / "production_management_trace.exe"
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(cache_dir)
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["PATH"] = str(tools["gpp"].parent) + os.pathsep + env.get("PATH", "")

    yosys_script = (
        f"read_verilog -sv {_relative(CORE)} {_relative(TOP)}; "
        "hierarchy -check -top gkp_fast_path_production_top; proc; check; stat; "
        f"write_cxxrtl -O0 -g0 {_relative(model)}"
    )
    started = time.perf_counter()
    yosys_run = _run((tools["yosys"], "-Q", "-p", yosys_script), env=env)
    yosys_seconds = time.perf_counter() - started
    yosys_log = build_dir / "yosys_cxxrtl.log"
    yosys_log.write_text(yosys_run.stdout + yosys_run.stderr, encoding="utf-8")

    compile_command = (
        tools["gpp"],
        "-std=c++17",
        # CXXRTL is an executable semantic oracle here, not a throughput
        # benchmark.  -O0 keeps the 5+ MB generated model build bounded.
        "-O0",
        "-I",
        tools["include"],
        "-I",
        build_dir,
        DRIVER,
        "-o",
        executable,
    )
    started = time.perf_counter()
    compile_run = _run(compile_command, env=env)
    compile_seconds = time.perf_counter() - started
    compile_log = build_dir / "gpp_compile.log"
    compile_log.write_text(compile_run.stdout + compile_run.stderr, encoding="utf-8")
    stat_match = re.search(r"\n\s*(\d+) memories\n\s*(\d+) memory bits", yosys_run.stdout)
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
        "memory_count": None if stat_match is None else int(stat_match.group(1)),
        "memory_bits": None if stat_match is None else int(stat_match.group(2)),
        "yosys_log": _relative(yosys_log),
        "compile_log": _relative(compile_log),
    }


def _encode_driver_line(cycle: int, values: Mapping[str, int]) -> str:
    return " ".join(
        str(item)
        for item in (
            cycle,
            values["safe_boundary"],
            values["cfg_begin_valid"],
            values["cfg_begin_bank"],
            values["cfg_expected_active_version"],
            values["cfg_new_version"],
            f"{values['cfg_expected_crc32']:08x}",
            values["cfg_word_valid"],
            values["cfg_word_phase"],
            values["cfg_word_address"],
            f"{values['cfg_word_data']:06x}",
            values["cfg_finalize_valid"],
            values["cfg_abort_valid"],
            values["commit_request_valid"],
            values["commit_request_bank"],
            values["commit_expected_active_version"],
            values["commit_new_version"],
            values["commit_cancel_valid"],
            values["management_snapshot_request"],
        )
    )


def build_management_trace() -> tuple[list[str], list[dict[str, int]], list[str]]:
    reference = ProductionFastPathManagementReference()
    lines: list[str] = []
    expected: list[dict[str, int]] = []
    labels: list[str] = []

    def emit(label: str, **overrides: int) -> None:
        values = MANAGEMENT_SIGNAL_DEFAULTS.copy()
        values.update(overrides)
        cycle = len(lines)
        lines.append(_encode_driver_line(cycle, values))
        expected.append(reference.step(values))
        labels.append(label)

    words = [((index * 0x2C9277 + 0x13579B) ^ (index << 7)) & 0x3FFFFF for index in range(514)]
    table_crc = crc32_table_words(words)

    def begin(bank: int, active_version: int, new_version: int, crc: int, label: str) -> None:
        emit(
            label,
            cfg_begin_valid=1,
            cfg_begin_bank=bank,
            cfg_expected_active_version=active_version,
            cfg_new_version=new_version,
            cfg_expected_crc32=crc,
        )

    def stream_table(label_prefix: str) -> None:
        for index, word in enumerate(words):
            phase = 0 if index < 257 else 1
            address = index if phase == 0 else index - 257
            emit(
                f"{label_prefix}_word_{index}",
                cfg_word_valid=1,
                cfg_word_phase=phase,
                cfg_word_address=address,
                cfg_word_data=word,
            )

    def snapshot(label_prefix: str, *, retry_while_busy: bool = False) -> None:
        emit(f"{label_prefix}_snapshot_request", management_snapshot_request=1)
        for index in range(18):
            emit(
                f"{label_prefix}_snapshot_{'valid' if index == 17 else index}",
                management_snapshot_request=int(retry_while_busy and index == 0),
            )

    emit("reset_release_idle")
    snapshot("initial", retry_while_busy=True)
    emit(
        "request_conflict",
        cfg_begin_valid=1,
        cfg_begin_bank=1,
        cfg_word_valid=1,
    )
    begin(0, 0, 1, table_crc, "active_bank_reject")
    begin(1, 9, 10, table_crc, "version_reject")

    begin(1, 0, 1, table_crc, "v1_begin")
    stream_table("v1")
    emit("v1_finalize", cfg_finalize_valid=1)
    snapshot("v1_finalized")
    emit(
        "commit_v1_request_cancel_path",
        safe_boundary=0,
        commit_request_valid=1,
        commit_request_bank=1,
        commit_expected_active_version=0,
        commit_new_version=1,
    )
    emit(
        "commit_busy_reject",
        safe_boundary=0,
        commit_request_valid=1,
        commit_request_bank=1,
        commit_expected_active_version=0,
        commit_new_version=1,
    )
    emit("commit_v1_cancel", safe_boundary=0, commit_cancel_valid=1)
    emit(
        "commit_v1_request",
        safe_boundary=0,
        commit_request_valid=1,
        commit_request_bank=1,
        commit_expected_active_version=0,
        commit_new_version=1,
    )
    emit("commit_v1_deferred", safe_boundary=0)
    emit("commit_v1_switch", safe_boundary=1)
    emit("commit_v1_complete", safe_boundary=1)
    begin(0, 1, 2, table_crc, "drain_guard_reject")
    for index in range(5):
        emit(f"drain_v1_{index}")
    snapshot("v1_committed")

    begin(0, 1, 2, table_crc, "order_fault_begin")
    emit("order_fault", cfg_word_valid=1, cfg_word_phase=0, cfg_word_address=511, cfg_word_data=1)
    begin(0, 1, 2, table_crc, "incomplete_begin")
    emit("incomplete_first_word", cfg_word_valid=1, cfg_word_phase=0, cfg_word_address=0, cfg_word_data=words[0])
    emit("incomplete_finalize", cfg_finalize_valid=1)

    begin(0, 1, 2, table_crc ^ 1, "bad_crc_begin")
    stream_table("bad_crc")
    emit("bad_crc_finalize", cfg_finalize_valid=1)
    snapshot("bad_crc")

    begin(0, 1, 2, table_crc, "v2_begin")
    stream_table("v2")
    emit("v2_finalize", cfg_finalize_valid=1)
    emit(
        "commit_v2_request",
        safe_boundary=0,
        commit_request_valid=1,
        commit_request_bank=0,
        commit_expected_active_version=1,
        commit_new_version=2,
    )
    emit("commit_v2_switch", safe_boundary=1)
    emit("commit_v2_complete", safe_boundary=1)
    for index in range(6):
        emit(f"drain_v2_{index}")

    begin(1, 2, 3, table_crc, "abort_begin")
    emit("abort_session", cfg_abort_valid=1)
    emit(
        "untrusted_commit_reject",
        commit_request_valid=1,
        commit_request_bank=1,
        commit_expected_active_version=2,
        commit_new_version=3,
    )
    emit("word_without_session", cfg_word_valid=1, cfg_word_phase=0, cfg_word_address=0, cfg_word_data=0)
    emit("cancel_without_pending", commit_cancel_valid=1)
    begin(1, 2, 3, table_crc, "conflict_aborts_session_begin")
    emit(
        "conflict_aborts_session",
        cfg_word_valid=1,
        cfg_word_phase=0,
        cfg_word_address=0,
        cfg_word_data=words[0],
        cfg_abort_valid=1,
    )
    snapshot("final")
    return lines, expected, labels


def run_driver(
    executable: Path,
    env: Mapping[str, str],
    lines: Sequence[str],
) -> tuple[list[dict[str, str]], float]:
    started = time.perf_counter()
    completed = _run(
        (executable,),
        env=env,
        input_text="\n".join(lines) + "\n",
        timeout=900,
    )
    runtime = time.perf_counter() - started
    return list(csv.DictReader(io.StringIO(completed.stdout))), runtime


EXPECTED_FIELDS = (
    "cfg_begin_ack",
    "cfg_word_ack",
    "cfg_finalize_ack",
    "cfg_abort_ack",
    "commit_request_ack",
    "commit_complete",
    "commit_cancel_ack",
    "management_snapshot_ack",
    "management_state_valid",
    "management_reject",
    "management_reject_reason",
    "cfg_session_active",
    "commit_pending",
    "management_snapshot_busy",
    "active_bank",
    "active_version",
    "management_state_word",
)


def compare_trace(
    actual: Sequence[Mapping[str, str]],
    expected: Sequence[Mapping[str, int]],
    labels: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (len(actual) == len(expected) == len(labels)):
        return [], [{"kind": "row_count", "actual": len(actual), "expected": len(expected)}]
    comparisons: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for cycle, (actual_row, expected_row, label) in enumerate(zip(actual, expected, labels)):
        checks: dict[str, bool] = {}
        row: dict[str, Any] = {"cycle": cycle, "label": label}
        for field in EXPECTED_FIELDS:
            actual_key = "management_state_word_hex" if field == "management_state_word" else field
            actual_value = int(actual_row[actual_key], 16) if field == "management_state_word" else int(actual_row[actual_key])
            expected_value = int(expected_row[field])
            checks[field] = actual_value == expected_value
            row[f"actual_{field}"] = f"{actual_value:040x}" if field == "management_state_word" else actual_value
            row[f"expected_{field}"] = f"{expected_value:040x}" if field == "management_state_word" else expected_value
        row["exact"] = all(checks.values())
        comparisons.append(row)
        if not row["exact"]:
            mismatches.append(
                {
                    "cycle": cycle,
                    "label": label,
                    "failed_checks": [field for field, passed in checks.items() if not passed],
                }
            )
    return comparisons, mismatches


def mutation_audit(
    actual: Sequence[dict[str, str]],
    expected: Sequence[Mapping[str, int]],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    label_index = {label: index for index, label in enumerate(labels)}
    mutations: list[tuple[str, list[dict[str, str]]]] = []
    for name, label, field, transform in (
        ("drop_crc_reject", "bad_crc_finalize", "management_reject", lambda _value: "0"),
        ("wrong_crc_reason", "bad_crc_finalize", "management_reject_reason", lambda _value: "7"),
        ("early_commit_complete", "commit_v1_switch", "commit_complete", lambda _value: "1"),
        ("drop_finalize_ack", "v2_finalize", "cfg_finalize_ack", lambda _value: "0"),
        ("active_bank_bit_flip", "commit_v2_switch", "active_bank", lambda value: str(int(value) ^ 1)),
        (
            "management_state_bit_flip",
            "conflict_aborts_session",
            "management_state_word_hex",
            lambda value: f"{int(value, 16) ^ 1:040x}",
        ),
    ):
        changed = copy.deepcopy(list(actual))
        index = label_index[label]
        changed[index][field] = transform(changed[index][field])
        mutations.append((name, changed))
    dropped = copy.deepcopy(list(actual))
    dropped.pop(label_index["order_fault"])
    mutations.append(("drop_trace_row", dropped))
    rows = []
    for name, changed in mutations:
        _, mismatches = compare_trace(changed, expected, labels)
        rows.append({"mutation": name, "rejected": bool(mismatches), "mismatch_count": len(mismatches)})
    return rows


def _source_contract() -> dict[str, bool]:
    core = CORE.read_text(encoding="utf-8")
    top = TOP.read_text(encoding="utf-8")
    return {
        "core_rejects_out_of_range_config_address": "cfg_address <= 9'd256" in core,
        "core_rejects_active_bank_write": "cfg_bank != active_bank" in core,
        "core_commit_requires_trust_inactive_bank_and_monotonic_version": all(
            token in core
            for token in (
                "requested_bank_trusted",
                "commit_bank != active_bank",
                "commit_version == (active_version + 16'd1)",
            )
        ),
        "production_age_ceiling_is_8192_not_demo_64": "MAX_PARAMETER_AGE_CYCLES = 16'd8192" in top,
        "production_supports_full_uint16_version_range": ".MAX_TRUSTED_BANK_VERSION(16'hffff)" in top,
        "strict_514_word_transaction_and_crc32": all(
            token in top
            for token in (
                "cfg_word_count != 10'd514",
                "crc32_word22",
                "cfg_running_crc32 ^ 32'hffffffff",
            )
        ),
        "safe_boundary_commit_cancel_and_drain_guard_present": all(
            token in top
            for token in (
                "core_commit_valid = commit_pending && safe_boundary",
                "commit_cancel_valid",
                "RETIRED_BANK_DRAIN_CYCLES",
            )
        ),
        "management_snapshot_crc_is_byte_serial_not_long_combinational_chain": all(
            token in top
            for token in (
                "crc16_byte(management_snapshot_crc, management_snapshot_octet)",
                "management_snapshot_shift <= management_snapshot_shift >> 8",
                "management_snapshot_byte_index == 5'd17",
                "management_state_valid <= 1'b1",
            )
        ) and "crc16_144" not in top,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields = ["cycle", "label", "exact"]
    for field in EXPECTED_FIELDS:
        fields.extend((f"actual_{field}", f"expected_{field}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def run_audit(
    *,
    build_dir: Path = DEFAULT_BUILD,
    output_json: Path = DEFAULT_JSON,
    output_csv: Path = DEFAULT_CSV,
) -> dict[str, Any]:
    build = build_cxxrtl(build_dir)
    lines, expected, labels = build_management_trace()
    actual, runtime_seconds = run_driver(build["executable"], build["environment"], lines)
    comparisons, mismatches = compare_trace(actual, expected, labels)
    mutations = mutation_audit(actual, expected, labels)
    source_contract = _source_contract()
    observed_reasons = sorted(
        {
            row["expected_management_reject_reason"]
            for row in comparisons
            if row["expected_management_reject"]
        }
    )
    required_reasons = sorted(
        {
            REJECT_CONFLICT,
            REJECT_BUSY,
            REJECT_ACTIVE_BANK,
            REJECT_VERSION,
            REJECT_DRAIN_GUARD,
            REJECT_NO_SESSION,
            REJECT_WORD_ORDER,
            REJECT_CRC32,
            REJECT_INCOMPLETE,
            REJECT_NO_PENDING,
            REJECT_UNTRUSTED,
        }
    )
    gates = {
        "yosys_elaboration_and_structural_check_zero_problems": build["structural_check_zero_problems"],
        "independent_reference_matches_every_cycle_and_state_bit": not mismatches,
        "full_514_word_good_and_bad_crc_transactions_executed": (
            len(lines) >= 1500
            and any(row["label"] == "bad_crc_finalize" for row in comparisons)
            and any(row["label"] == "v2_finalize" for row in comparisons)
        ),
        "all_fail_closed_rejection_classes_observed": observed_reasons == required_reasons,
        "safe_boundary_defer_cancel_commit_and_drain_executed": all(
            label in labels
            for label in (
                "commit_v1_deferred",
                "commit_v1_cancel",
                "commit_v1_switch",
                "commit_v1_complete",
                "drain_guard_reject",
            )
        ),
        "coherent_crc_state_snapshots_complete_in_bounded_18_cycles": (
            sum(int(row["management_state_valid"]) for row in actual) == 5
            and sum(int(row["management_snapshot_ack"]) for row in actual) == 5
            and "initial_snapshot_0" in labels
        ),
        "all_source_contract_guards_present": all(source_contract.values()),
        "all_semantic_output_mutations_are_rejected": all(row["rejected"] for row in mutations),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "task_id": "T6.2.1",
        "status": status,
        "verdict": VERDICT if status == "PASS" else "FAIL_PRODUCTION_RTL_AUDIT",
        "cycle_rows": len(lines),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:20],
        "runtime_seconds": runtime_seconds,
        "observed_reject_reasons": observed_reasons,
        "required_reject_reasons": required_reasons,
        "gate_summary": {"passed": sum(gates.values()), "total": len(gates)},
        "gates": gates,
        "source_contract": source_contract,
        "mutation_audit": mutations,
        "build": {key: value for key, value in build.items() if key not in {"executable", "environment"}},
        "artifacts": {
            "source_data_csv": _relative(output_csv),
            "core_rtl": _relative(CORE),
            "production_top_rtl": _relative(TOP),
            "cxxrtl_driver": _relative(DRIVER),
        },
        "evidence_boundary": {
            "synthesizable_rtl": True,
            "cycle_accurate_cxxrtl": True,
            "board_independent_management_contract": True,
            "transport_or_cdc_validated": False,
            "target_place_route": False,
            "board_measured": False,
            "crc32_is_integrity_not_authentication": True,
        },
    }
    _write_csv(output_csv, comparisons)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    report = run_audit(build_dir=args.build_dir, output_json=args.output_json, output_csv=args.output_csv)
    print(json.dumps({key: report[key] for key in ("task_id", "status", "verdict", "cycle_rows", "mismatch_count")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
