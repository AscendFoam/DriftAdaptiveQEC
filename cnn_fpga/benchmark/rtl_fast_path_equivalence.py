"""Build CXXRTL from the synthesizable fast path and compare it to T5.5.1."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from cnn_fpga.benchmark.bit_accurate_hardware_reference import (
    _input_word,
    load_frozen_images,
)
from cnn_fpga.runtime.atomic_parameter_bank import AtomicParameterBankConfig
from cnn_fpga.runtime.bit_accurate_hardware_reference import (
    BitAccurateHardwareReference,
    HardwareTraceRecord,
    encode_input_word,
    pack_parameter_image,
)
from cnn_fpga.rtl.generate_frozen_memories import generate as generate_memories


ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
DRIVER = ROOT / "cnn_fpga/rtl/cxxrtl_trace_driver.cc"
MEMORY_MANIFEST = ROOT / "cnn_fpga/rtl/generated/t5_5_1_memory_manifest.json"
DEFAULT_BUILD = ROOT / ".tmp_t5_5_2_equivalence"
DEFAULT_JSON = ROOT / "docs/t_risk_20260716_01_rtl_equivalence.json"
DEFAULT_CSV = ROOT / "docs/t_risk_20260716_01_rtl_equivalence_source_data.csv"
VERDICT = "SYNTHESIZABLE_RTL_EQUIVALENT_FOR_V0_V1_NOT_BOARD_MEASURED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _find_executable(name: str, candidates: Iterable[Path]) -> Path:
    command = shutil.which(name)
    if command:
        return Path(command)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"required executable is unavailable: {name}")


def discover_tools() -> dict[str, Path]:
    env_root = Path(sys.executable).resolve().parent
    scripts = env_root / "Scripts"
    yosys = _find_executable(
        "yowasp-yosys",
        (
            scripts / "yowasp-yosys.exe",
            Path(r"C:\ProgramData\anaconda3\envs\DLEnv\Scripts\yowasp-yosys.exe"),
        ),
    )
    gpp = _find_executable(
        "g++",
        (Path(r"C:\ProgramData\msys2\mingw64\bin\g++.exe"),),
    )
    try:
        import yowasp_yosys
    except ImportError as exc:  # pragma: no cover - environment failure path
        raise RuntimeError("yowasp_yosys package is required for CXXRTL headers") from exc
    include = (
        Path(yowasp_yosys.__file__).resolve().parent
        / "share/include/backends/cxxrtl/runtime"
    )
    if not (include / "cxxrtl/cxxrtl.h").is_file():
        raise FileNotFoundError(f"CXXRTL runtime headers unavailable: {include}")
    return {"yosys": yosys, "gpp": gpp, "include": include}


def _run(
    command: Sequence[str],
    *,
    env: dict[str, str],
    input_text: str | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        env=env,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def build_cxxrtl(build_dir: Path, tools: dict[str, Path]) -> dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = build_dir / "temp"
    cache_dir = build_dir / "yowasp_cache"
    temp_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    model = build_dir / "gkp_fast_path_model.cc"
    executable = build_dir / "gkp_fast_path_trace.exe"
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(cache_dir)
    env["TEMP"] = str(temp_dir)
    env["TMP"] = str(temp_dir)
    env["PATH"] = str(tools["gpp"].parent) + os.pathsep + env.get("PATH", "")

    yosys_script = (
        f"read_verilog -sv {_relative(CORE)}; "
        "hierarchy -check -top gkp_fast_path_core; proc; check; stat; "
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
        "-O1",
        "-I",
        tools["include"],
        "-I",
        build_dir,
        DRIVER,
        "-o",
        executable,
    )
    started = time.perf_counter()
    compile_run = _run(compile_command, env=env, timeout=900)
    compile_seconds = time.perf_counter() - started
    compile_log = build_dir / "gpp_compile.log"
    compile_log.write_text(compile_run.stdout + compile_run.stderr, encoding="utf-8")

    yosys_version = _run((tools["yosys"], "-V"), env=env).stdout.strip()
    gpp_version = _run((tools["gpp"], "--version"), env=env).stdout.splitlines()[0]
    memory_match = re.search(r"\n\s*(\d+) memories\n\s*(\d+) memory bits", yosys_run.stdout)
    return {
        "executable": executable,
        "environment": env,
        "yosys_version": yosys_version,
        "gpp_version": gpp_version,
        "yosys_seconds": yosys_seconds,
        "compile_seconds": compile_seconds,
        "model_bytes": model.stat().st_size,
        "model_sha256": _sha256(model),
        "executable_bytes": executable.stat().st_size,
        "executable_sha256": _sha256(executable),
        "structural_check_zero_problems": "Found and reported 0 problems" in yosys_run.stdout,
        "memory_count": None if memory_match is None else int(memory_match.group(1)),
        "memory_bits": None if memory_match is None else int(memory_match.group(2)),
        "multiplier_present": "$mul" in yosys_run.stdout,
        "yosys_log": _relative(yosys_log),
        "compile_log": _relative(compile_log),
    }


def _stage_reference(reference: BitAccurateHardwareReference, images: Sequence[Any], cycle: int, tag: str) -> None:
    reference.stage_packed_update(
        pack_parameter_image(images[1]),
        transaction_id=f"rtl-{tag}-v1",
        selection_key="bank-v1",
        source_window_id=2,
        created_cycle=0,
        apply_cycle=cycle,
    )


def build_fault_trace() -> tuple[list[str], list[HardwareTraceRecord]]:
    images = load_frozen_images()
    reference = BitAccurateHardwareReference(
        images,
        bank_config=AtomicParameterBankConfig(min_residency_cycles=50),
    )
    _stage_reference(reference, images, 50, "fault")
    lines: list[str] = []
    records: list[HardwareTraceRecord] = []
    for cycle in range(220):
        safe = cycle != 50
        word = _input_word(cycle)
        records.append(reference.step_word(word, safe_boundary=safe))
        commit = cycle in (50, 51)
        lines.append(
            f"{cycle} 1 {word:x} {int(safe)} {int(commit)} 1 1 1 1"
        )
    for cycle in range(220, 226):
        records.append(reference.step_word(None, safe_boundary=True))
        lines.append(f"{cycle} 0 0 1 0 1 1 1 1")
    return lines, records


def build_exhaustive_trace() -> tuple[list[str], list[HardwareTraceRecord]]:
    images = load_frozen_images()
    commit_cycle = 2048
    reference = BitAccurateHardwareReference(
        images,
        bank_config=AtomicParameterBankConfig(min_residency_cycles=commit_cycle),
    )
    _stage_reference(reference, images, commit_cycle, "exhaustive")
    lines: list[str] = []
    records: list[HardwareTraceRecord] = []
    for cycle in range(4096):
        local = cycle % 2048
        phase = local // 1024
        code = local % 1024
        word = encode_input_word(
            syndrome_code=code,
            syndrome_x="g",
            syndrome_z="g",
            quadrature_phase_bit=phase,
            ood_score_code=16,
            parameter_age_code=cycle % 32,
        )
        records.append(reference.step_word(word, safe_boundary=True))
        commit = cycle == commit_cycle
        lines.append(f"{cycle} 1 {word:x} 1 {int(commit)} 1 1 1 1")
    for cycle in range(4096, 4102):
        records.append(reference.step_word(None, safe_boundary=True))
        lines.append(f"{cycle} 0 0 1 0 1 1 1 1")
    return lines, records


def run_driver(
    executable: Path,
    env: dict[str, str],
    lines: Sequence[str],
) -> tuple[list[dict[str, str]], float]:
    started = time.perf_counter()
    completed = _run(
        (executable,),
        env=env,
        input_text="\n".join(lines) + "\n",
        timeout=300,
    )
    runtime = time.perf_counter() - started
    rows = list(csv.DictReader(io.StringIO(completed.stdout)))
    if len(rows) != len(lines):
        raise RuntimeError(f"driver row count mismatch: {len(rows)} != {len(lines)}")
    return rows, runtime


def compare_trace(
    scenario: str,
    actual: Sequence[dict[str, str]],
    expected: Sequence[HardwareTraceRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    if len(actual) != len(expected):
        mismatches.append(
            {"scenario": scenario, "kind": "row_count", "actual": len(actual), "expected": len(expected)}
        )
    for index, (row, record) in enumerate(zip(actual, expected)):
        expected_ack = record.commit_status == "committed"
        actual_map_valid = bool(int(row["map_valid"]))
        checks = {
            "cycle": int(row["cycle"]) == record.hardware_cycle == index,
            "commit_ack": bool(int(row["commit_ack"])) == expected_ack,
            "active_version": int(row["active_version"]) == record.active_version,
            "map_valid": actual_map_valid == record.map_valid,
            "map_address": (not record.map_valid) or int(row["map_address"]) == record.map_address,
            "map_llr": (not record.map_valid)
            or int(row["map_llr_twos"]) == (int(record.map_llr_code) & ((1 << 22) - 1)),
            "output_word": row["out_word_hex"] == record.output_word_hex,
            "state_word": row["state_word_hex"] == record.state_word_hex,
        }
        exact = all(checks.values())
        comparison = {
            "scenario": scenario,
            "cycle": index,
            "input_valid": bool(int(row["input_valid"])),
            "input_word_hex": row["input_word_hex"],
            "actual_commit_ack": bool(int(row["commit_ack"])),
            "expected_commit_ack": expected_ack,
            "actual_active_version": int(row["active_version"]),
            "expected_active_version": record.active_version,
            "actual_map_valid": actual_map_valid,
            "expected_map_valid": record.map_valid,
            "actual_map_address": int(row["map_address"]),
            "expected_map_address": record.map_address,
            "actual_map_llr_twos": int(row["map_llr_twos"]),
            "expected_map_llr_twos": None
            if record.map_llr_code is None
            else int(record.map_llr_code) & ((1 << 22) - 1),
            "actual_output_word_hex": row["out_word_hex"],
            "expected_output_word_hex": record.output_word_hex,
            "actual_state_word_hex": row["state_word_hex"],
            "expected_state_word_hex": record.state_word_hex,
            "checks": checks,
            "exact": exact,
        }
        comparisons.append(comparison)
        if not exact:
            mismatches.append(
                {
                    "scenario": scenario,
                    "cycle": index,
                    "failed_checks": [name for name, passed in checks.items() if not passed],
                }
            )
    return comparisons, mismatches


def mutation_audit(
    actual: Sequence[dict[str, str]], expected: Sequence[HardwareTraceRecord]
) -> list[dict[str, Any]]:
    mutations: list[tuple[str, list[dict[str, str]]]] = []
    for name, cycle, field, transform in (
        ("map_llr_plus_one", 5, "map_llr_twos", lambda value: str(int(value) + 1)),
        ("map_address_plus_one", 5, "map_address", lambda value: str(int(value) + 1)),
        ("output_word_bit_flip", 6, "out_word_hex", lambda value: f"{int(value, 16) ^ 1:030x}"),
        ("state_word_bit_flip", 5, "state_word_hex", lambda value: f"{int(value, 16) ^ 1:058x}"),
        ("commit_ack_one_cycle_early", 50, "commit_ack", lambda _value: "1"),
        ("active_version_one_cycle_early", 50, "active_version", lambda _value: "1"),
        ("drop_map_valid", 5, "map_valid", lambda _value: "0"),
    ):
        changed = copy.deepcopy(list(actual))
        changed[cycle][field] = transform(changed[cycle][field])
        mutations.append((name, changed))
    dropped = copy.deepcopy(list(actual))
    dropped.pop(100)
    mutations.append(("drop_trace_row", dropped))

    rows = []
    for name, changed in mutations:
        _, mismatches = compare_trace(f"mutation:{name}", changed, expected)
        rows.append({"mutation": name, "rejected": bool(mismatches), "mismatch_count": len(mismatches)})
    return rows


def _write_csv(path: Path, comparisons: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario", "cycle", "input_valid", "input_word_hex",
        "actual_commit_ack", "expected_commit_ack",
        "actual_active_version", "expected_active_version",
        "actual_map_valid", "expected_map_valid",
        "actual_map_address", "expected_map_address",
        "actual_map_llr_twos", "expected_map_llr_twos",
        "actual_output_word_hex", "expected_output_word_hex",
        "actual_state_word_hex", "expected_state_word_hex", "exact",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({field: row.get(field) for field in fields})


def run_equivalence(
    *,
    build_dir: Path = DEFAULT_BUILD,
    output_json: Path = DEFAULT_JSON,
    output_csv: Path = DEFAULT_CSV,
) -> dict[str, Any]:
    memory_manifest = generate_memories()
    tools = discover_tools()
    build = build_cxxrtl(build_dir, tools)
    all_comparisons: list[dict[str, Any]] = []
    scenario_reports = []
    actual_by_name: dict[str, list[dict[str, str]]] = {}
    expected_by_name: dict[str, list[HardwareTraceRecord]] = {}
    for name, builder in (("fault_and_commit", build_fault_trace), ("exhaustive_v0_v1", build_exhaustive_trace)):
        lines, expected = builder()
        actual, runtime = run_driver(build["executable"], build["environment"], lines)
        comparisons, mismatches = compare_trace(name, actual, expected)
        all_comparisons.extend(comparisons)
        actual_by_name[name] = actual
        expected_by_name[name] = expected
        scenario_reports.append(
            {
                "name": name,
                "cycles": len(lines),
                "map_valid_rows": sum(row["expected_map_valid"] for row in comparisons),
                "runtime_seconds": runtime,
                "mismatch_count": len(mismatches),
                "exact": not mismatches,
                "first_mismatches": mismatches[:10],
            }
        )
    mutations = mutation_audit(
        actual_by_name["fault_and_commit"], expected_by_name["fault_and_commit"]
    )
    fault_rows = actual_by_name["fault_and_commit"]
    gates = {
        "eight_mirrored_1r1w_memories_and_multiplier_survive_elaboration": (
            build["structural_check_zero_problems"]
            and build["memory_count"] == 8
            and build["memory_bits"] == 45232
            and build["multiplier_present"]
        ),
        "frozen_v0_v1_memory_files_match_257x22_registry": (
            len(memory_manifest["files"]) == 4
            and all(row["entries"] == 257 for row in memory_manifest["files"])
        ),
        "fault_commit_and_full_words_match_cycle_by_cycle": scenario_reports[0]["exact"],
        "both_banks_both_phases_all_1024_codes_match": (
            scenario_reports[1]["exact"] and scenario_reports[1]["map_valid_rows"] == 4096
        ),
        "map_and_action_latencies_are_five_and_six_cycles": (
            bool(int(fault_rows[5]["map_valid"]))
            and not bool(int(fault_rows[4]["map_valid"]))
            and (int(fault_rows[6]["out_word_hex"], 16) & 1) == 1
            and (int(fault_rows[5]["out_word_hex"], 16) & 1) == 0
        ),
        "unsafe_commit_defers_then_safe_commit_is_atomic": (
            fault_rows[50]["commit_ack"] == "0"
            and fault_rows[50]["active_version"] == "0"
            and fault_rows[51]["commit_ack"] == "1"
            and fault_rows[51]["active_version"] == "1"
        ),
        "all_semantic_mutations_are_rejected": all(row["rejected"] for row in mutations),
        "rtl_or_cxxrtl_is_not_mislabeled_as_board_measurement": True,
    }
    report = {
        "schema_version": "t-risk-20260716-01-rtl-equivalence-v1",
        "task_id": "T-RISK-20260716-01",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "verdict": VERDICT if all(gates.values()) else "RTL_EQUIVALENCE_FAILED",
        "source_bindings": [
            {"path": _relative(path), "sha256": _sha256(path)}
            for path in (CORE, DRIVER, MEMORY_MANIFEST)
        ],
        "tools": {key: str(value) for key, value in tools.items()},
        "build": {key: value for key, value in build.items() if key not in {"executable", "environment"}},
        "memory_manifest": memory_manifest,
        "scenarios": scenario_reports,
        "mutation_audit": mutations,
        "gates": gates,
        "gate_summary": {"passed": sum(gates.values()), "total": len(gates)},
        "source_data": {"path": _relative(output_csv), "rows": len(all_comparisons)},
        "evidence_boundary": {
            "synthesizable_rtl": True,
            "cxxrtl_simulation": True,
            "target_device_synthesis": False,
            "target_device_place_route": False,
            "board_measured": False,
            "transport_implemented": False,
            "quantum_hardware_measured": False,
        },
    }
    _write_csv(output_csv, all_comparisons)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    report = run_equivalence(
        build_dir=args.build_dir,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps({"status": report["status"], "verdict": report["verdict"], "gates": report["gate_summary"]}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
