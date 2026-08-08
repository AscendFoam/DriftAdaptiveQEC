"""Freeze target-device synthesis and post-route timing evidence for T5.5.2."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CORE = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
TOP = ROOT / "cnn_fpga/rtl/gkp_fast_path_synth_top.sv"
SDC = ROOT / "cnn_fpga/rtl/tang_nano_20k_27mhz.sdc"
CST = ROOT / "cnn_fpga/rtl/tang_nano_20k_synth_harness.cst"
MEMORY_MANIFEST = ROOT / "cnn_fpga/rtl/generated/t5_5_1_memory_manifest.json"
EQUIVALENCE_REPORT = ROOT / "docs/t_risk_20260716_01_rtl_equivalence.json"
DEFAULT_BUILD = ROOT / ".tmp_t552_build"
DEFAULT_JSON = ROOT / "docs/t5_5_2_target_device_synthesis.json"
DEFAULT_CSV = ROOT / "docs/t5_5_2_target_device_synthesis_source_data.csv"
DEVICE = "GW2AR-LV18QN88C8/I7"
FAMILY = "GW2A-18C"
TOP_MODULE = "gkp_fast_path_synth_top"
TARGET_MHZ = 27.0
SEEDS = (1, 7, 19)
CORE_LATENCY_CYCLES = 6
INITIATION_INTERVAL_CYCLES = 1
VERDICT = "TARGET_DEVICE_POST_ROUTE_ESTIMATE_PASSES_27MHZ_NOT_BOARD_MEASURED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _read_tool_text(path: Path) -> str:
    """Read native tool logs, including PowerShell's UTF-16 redirection output."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _find_executable(name: str, candidates: Iterable[Path]) -> Path:
    command = shutil.which(name)
    if command:
        return Path(command)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"required executable is unavailable: {name}")


def discover_tools() -> dict[str, Path]:
    scripts = Path(sys.executable).resolve().parent / "Scripts"
    yosys = _find_executable(
        "yowasp-yosys",
        (scripts / "yowasp-yosys.exe", Path(r"C:\ProgramData\anaconda3\envs\DLEnv\Scripts\yowasp-yosys.exe")),
    )
    nextpnr = _find_executable(
        "yowasp-nextpnr-himbaechel-gowin",
        (
            scripts / "yowasp-nextpnr-himbaechel-gowin.exe",
            Path(r"C:\ProgramData\anaconda3\envs\DLEnv\Scripts\yowasp-nextpnr-himbaechel-gowin.exe"),
        ),
    )
    return {"yosys": yosys, "nextpnr": nextpnr}


def _run(command: Sequence[str], *, env: dict[str, str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=True,
    )


def _seed_paths(build_dir: Path, seed: int) -> dict[str, Path]:
    if seed == 1:
        return {
            "report": build_dir / "nextpnr_27mhz_report.json",
            "log": build_dir / "nextpnr_27mhz.log",
            "routed": build_dir / "gkp_fast_path_routed.json",
        }
    tag = f"{seed:02d}"
    return {
        "report": build_dir / f"nextpnr_seed{tag}_report.json",
        "log": build_dir / f"nextpnr_seed{tag}.log",
        "routed": build_dir / f"gkp_fast_path_routed_seed{tag}.json",
    }


def tool_commands(build_dir: Path, tools: dict[str, Path]) -> dict[str, Any]:
    netlist = build_dir / "gkp_fast_path_gw2a.json"
    yosys_script = (
        f"read_verilog -sv {_relative(CORE)} {_relative(TOP)}; "
        f"hierarchy -check -top {TOP_MODULE}; proc; check; "
        f"synth_gowin -family gw2a -no-rw-check -top {TOP_MODULE} -json {_relative(netlist)}; "
        "stat"
    )
    synthesis = [str(tools["yosys"]), "-Q", "-p", yosys_script]
    routes: dict[str, list[str]] = {}
    for seed in SEEDS:
        paths = _seed_paths(build_dir, seed)
        routes[str(seed)] = [
            str(tools["nextpnr"]),
            "--device", DEVICE,
            "-o", f"family={FAMILY}",
            "-o", f"cst={_relative(CST)}",
            "--json", _relative(netlist),
            "--top", TOP_MODULE,
            "--freq", f"{TARGET_MHZ:g}",
            "--sdc", _relative(SDC),
            "--seed", str(seed),
            "--report", _relative(paths["report"]),
            "--detailed-timing-report",
            "--write", _relative(paths["routed"]),
        ]
    return {"synthesis": synthesis, "place_route": routes}


def run_toolchain(build_dir: Path, tools: dict[str, Path]) -> dict[str, float]:
    build_dir.mkdir(parents=True, exist_ok=True)
    cache = build_dir / "yowasp_cache"
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(cache)
    commands = tool_commands(build_dir, tools)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    result = _run(commands["synthesis"], env=env)
    timings["synthesis_seconds"] = time.perf_counter() - started
    (build_dir / "yosys_synth_pipelined_harness.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )

    def route_seed(seed: int) -> tuple[int, float, bool]:
        started = time.perf_counter()
        try:
            result = _run(commands["place_route"][str(seed)], env=env, timeout=1200)
        except subprocess.CalledProcessError as exc:
            _seed_paths(build_dir, seed)["log"].write_text(
                (exc.stdout or "") + (exc.stderr or ""), encoding="utf-8"
            )
            return seed, time.perf_counter() - started, False
        _seed_paths(build_dir, seed)["log"].write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        return seed, time.perf_counter() - started, True

    failed_seeds = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = [executor.submit(route_seed, seed) for seed in SEEDS]
        for future in futures:
            seed, elapsed, succeeded = future.result()
            timings[f"place_route_seed_{seed}_seconds"] = elapsed
            if not succeeded:
                failed_seeds.append(seed)
    for seed in failed_seeds:
        started = time.perf_counter()
        result = _run(commands["place_route"][str(seed)], env=env, timeout=1200)
        timings[f"place_route_seed_{seed}_seconds"] += time.perf_counter() - started
        _seed_paths(build_dir, seed)["log"].write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
    return timings


def parse_yosys_log(path: Path) -> dict[str, Any]:
    text = _read_tool_text(path)
    names = (
        "LUT1", "LUT2", "LUT3", "LUT4", "MUX2_LUT5", "MUX2_LUT6",
        "MUX2_LUT7", "MUX2_LUT8", "ALU", "SDPX9B", "MULT18X18", "MULT9X9",
        "DFF", "DFFE", "DFFR", "DFFRE", "DFFS", "IBUF", "OBUF",
    )
    cells: dict[str, int] = {}
    for name in names:
        matches = re.findall(rf"^\s*(\d+)\s+{re.escape(name)}\s*$", text, flags=re.MULTILINE)
        cells[name] = int(matches[-1]) if matches else 0
    warnings = re.findall(r"^Warning:\s*(.+(?:\n\s+.+)?)$", text, flags=re.MULTILINE)
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "zero_structural_problems": "Found and reported 0 problems" in text,
        "cell_counts": cells,
        "register_count": sum(cells[name] for name in ("DFF", "DFFE", "DFFR", "DFFRE", "DFFS")),
        "lut1_to_lut4_count": sum(cells[name] for name in ("LUT1", "LUT2", "LUT3", "LUT4")),
        "lut_count_scope": "pre_abc9_log_unavailable_use_post_route_utilization",
        "warnings": warnings,
    }


def parse_nextpnr(report_path: Path, log_path: Path, seed: int) -> dict[str, Any]:
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if len(raw["fmax"]) != 1:
        raise ValueError(f"expected one clock in {report_path}, got {list(raw['fmax'])}")
    clock_name, fmax = next(iter(raw["fmax"].items()))
    if len(raw["critical_paths"]) != 1:
        raise ValueError(f"expected one critical path in {report_path}")
    critical = raw["critical_paths"][0]
    segments = critical["path"]
    delays = {
        kind: sum(float(row["delay"]) for row in segments if row["type"] == kind)
        for kind in ("clk-to-q", "logic", "routing", "setup")
    }
    total_ns = sum(float(row["delay"]) for row in segments)
    log = _read_tool_text(log_path)
    utilization = {
        name: {"used": int(value["used"]), "available": int(value["available"])}
        for name, value in raw["utilization"].items()
    }
    return {
        "seed": seed,
        "clock_name": clock_name,
        "target_mhz": float(fmax["constraint"]),
        "achieved_fmax_mhz": float(fmax["achieved"]),
        "timing_pass": float(fmax["achieved"]) >= float(fmax["constraint"]),
        "route_status": "PASS" if "Program finished normally." in log else "FAIL",
        "critical_path": {
            "period_ns": total_ns,
            "segment_count": len(segments),
            "clock_to_q_ns": delays["clk-to-q"],
            "logic_ns": delays["logic"],
            "routing_ns": delays["routing"],
            "setup_ns": delays["setup"],
            "start_cell": segments[0]["from"].get("cell"),
            "start_location": segments[0]["from"].get("loc"),
            "end_cell": segments[-1]["to"].get("cell"),
            "end_location": segments[-1]["to"].get("loc"),
            "from_clock": critical["from"],
            "to_clock": critical["to"],
        },
        "utilization": utilization,
        "report_artifact": {
            "path": _relative(report_path), "sha256": _sha256(report_path), "bytes": report_path.stat().st_size,
        },
        "log_artifact": {
            "path": _relative(log_path), "sha256": _sha256(log_path), "bytes": log_path.stat().st_size,
        },
        "log_contract": {
            "exact_device_loaded": f"device '{DEVICE}'" in log,
            "clock_constrained_to_27mhz": "27.00 MHz" in log,
            "program_finished_normally": "Program finished normally." in log,
        },
    }


def _bind(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _copy_durable_artifacts(build_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    docs = ROOT / "docs"
    synthesis_source = build_dir / "yosys_synth_pipelined_harness.log"
    synthesis_dest = docs / "t5_5_2_yosys_synthesis.log"
    synthesis_dest.write_text(_read_tool_text(synthesis_source), encoding="utf-8")
    copied = [_bind(synthesis_dest)]
    for seed in SEEDS:
        paths = _seed_paths(build_dir, seed)
        tag = f"{seed:02d}"
        for kind, suffix in (("report", "report.json"), ("log", "place_route.log")):
            destination = docs / f"t5_5_2_nextpnr_seed{tag}_{suffix}"
            if kind == "log":
                destination.write_text(_read_tool_text(paths[kind]), encoding="utf-8")
            else:
                shutil.copyfile(paths[kind], destination)
            copied.append(_bind(destination))
    return synthesis_dest, copied


def _artifact_matches(row: dict[str, Any]) -> bool:
    path = ROOT / row["path"]
    return path.is_file() and path.stat().st_size == row["bytes"] and _sha256(path) == row["sha256"]


def evaluate_gates(report: dict[str, Any]) -> dict[str, bool]:
    sources_ok = all(_artifact_matches(row) for row in report["source_bindings"])
    artifacts_ok = all(_artifact_matches(row) for row in report["durable_artifacts"])
    parent = report["parent_equivalence"]
    parent_ok = (
        parent["status"] == "PASS"
        and parent["mismatch_count"] == 0
        and _artifact_matches(parent["artifact"])
    )
    target = report["target_contract"]
    exact_target = (
        target["device"] == DEVICE
        and target["family"] == FAMILY
        and target["top"] == TOP_MODULE
        and target["target_mhz"] == TARGET_MHZ
        and target["constraints"]["sdc"] == _relative(SDC)
        and target["constraints"]["cst"] == _relative(CST)
    )
    synthesis = report["synthesis"]
    cells = synthesis["cell_counts"]
    synthesis_nonempty = (
        synthesis["zero_structural_problems"]
        and cells["SDPX9B"] == 8
        and cells["MULT18X18"] >= 1
        and cells["MULT9X9"] >= 1
        and synthesis["register_count"] >= 800
    )
    routes = report["place_route"]
    route_set = sorted(row["seed"] for row in routes) == list(SEEDS)
    every_route_passes = route_set and all(
        row["route_status"] == "PASS"
        and row["timing_pass"]
        and row["achieved_fmax_mhz"] >= TARGET_MHZ
        and row["clock_name"] == "core.clk"
        and all(row["log_contract"].values())
        for row in routes
    )
    selected_resources = ("LUT4", "DFF", "BSRAM", "MULT18X18", "MULT9X9", "IOB", "ALU")
    resources_valid = every_route_passes and all(
        all(
            row["utilization"][name]["used"] > 0
            and row["utilization"][name]["used"] <= row["utilization"][name]["available"]
            for name in selected_resources
        )
        for row in routes
    )
    critical_paths_valid = every_route_passes and all(
        row["critical_path"]["period_ns"] > 0
        and row["critical_path"]["logic_ns"] > 0
        and row["critical_path"]["routing_ns"] > 0
        and row["critical_path"]["segment_count"] >= 10
        and row["critical_path"]["start_cell"].startswith("core.")
        and row["critical_path"]["end_cell"].startswith("fold")
        for row in routes
    )
    summary = report["summary"]
    fmax_values = [row["achieved_fmax_mhz"] for row in routes]
    summary_recomputed = (
        abs(summary["fmax_mhz"]["minimum"] - min(fmax_values)) < 1e-9
        and abs(summary["fmax_mhz"]["median"] - statistics.median(fmax_values)) < 1e-9
        and abs(summary["fmax_mhz"]["maximum"] - max(fmax_values)) < 1e-9
    )
    latency = report["latency_estimate"]
    latency_valid = (
        latency["core_cycles"] == CORE_LATENCY_CYCLES
        and latency["initiation_interval_cycles"] == INITIATION_INTERVAL_CYCLES
        and abs(latency["at_target_27mhz_ns"] - CORE_LATENCY_CYCLES * 1000.0 / TARGET_MHZ) < 1e-9
        and abs(latency["at_worst_seed_fmax_ns"] - CORE_LATENCY_CYCLES * 1000.0 / min(fmax_values)) < 1e-9
    )
    boundary = report["evidence_boundary"]
    boundary_valid = (
        boundary["synthesizable_rtl"] is True
        and boundary["target_device_synthesis"] is True
        and boundary["target_device_place_route"] is True
        and boundary["bitstream_generated"] is False
        and boundary["board_measured"] is False
        and boundary["transport_implemented"] is False
        and boundary["vendor_timing_signoff"] is False
        and boundary["quantum_hardware_measured"] is False
    )
    return {
        "parent_rtl_equivalence_is_source_bound_and_exact": parent_ok,
        "all_source_and_constraint_hashes_match": sources_ok,
        "exact_gw2ar_qn88_target_and_27mhz_constraints_are_frozen": exact_target,
        "synthesis_is_nonempty_and_preserves_eight_brams_and_dsps": synthesis_nonempty,
        "three_independent_place_route_seeds_pass_27mhz": every_route_passes,
        "reported_resources_are_nonzero_and_within_device_capacity": resources_valid,
        "critical_paths_have_core_sources_and_observable_harness_endpoints": critical_paths_valid,
        "min_median_max_fmax_are_recomputed_from_all_seeds": summary_recomputed,
        "six_cycle_latency_and_ii_one_arithmetic_are_exact": latency_valid,
        "all_durable_tool_reports_and_logs_match_their_hashes": artifacts_ok,
        "post_route_estimate_is_not_mislabeled_as_vendor_or_board_measurement": boundary_valid,
    }


def mutation_audit(report: dict[str, Any]) -> list[dict[str, Any]]:
    mutations: list[tuple[str, Any]] = [
        ("wrong_device", lambda r: r["target_contract"].__setitem__("device", "GW1N-DEMO")),
        ("drop_seed_19", lambda r: r["place_route"].pop()),
        ("fmax_below_target", lambda r: r["place_route"][0].__setitem__("achieved_fmax_mhz", 26.0)),
        ("erase_bram_usage", lambda r: r["place_route"][0]["utilization"]["BSRAM"].__setitem__("used", 0)),
        ("erase_critical_routing", lambda r: r["place_route"][0]["critical_path"].__setitem__("routing_ns", 0.0)),
        ("break_parent_hash", lambda r: r["parent_equivalence"]["artifact"].__setitem__("sha256", "0" * 64)),
        ("break_latency_arithmetic", lambda r: r["latency_estimate"].__setitem__("at_target_27mhz_ns", 6.0)),
        ("claim_board_measurement", lambda r: r["evidence_boundary"].__setitem__("board_measured", True)),
        ("break_tool_report_hash", lambda r: r["durable_artifacts"][0].__setitem__("sha256", "f" * 64)),
    ]
    rows = []
    for name, mutate in mutations:
        candidate = copy.deepcopy(report)
        mutate(candidate)
        gates = evaluate_gates(candidate)
        rows.append({
            "mutation": name,
            "rejected": not all(gates.values()),
            "failed_gates": [gate for gate, passed in gates.items() if not passed],
        })
    return rows


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_text(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout + result.stderr).strip()


def build_report(build_dir: Path, tools: dict[str, Path], timings: dict[str, float] | None = None) -> dict[str, Any]:
    synthesis_log, artifacts = _copy_durable_artifacts(build_dir)
    routes = []
    for seed in SEEDS:
        tag = f"{seed:02d}"
        routes.append(parse_nextpnr(
            ROOT / f"docs/t5_5_2_nextpnr_seed{tag}_report.json",
            ROOT / f"docs/t5_5_2_nextpnr_seed{tag}_place_route.log",
            seed,
        ))
    fmax = [row["achieved_fmax_mhz"] for row in routes]
    parent_raw = json.loads(EQUIVALENCE_REPORT.read_text(encoding="utf-8"))
    parent_mismatches = sum(row["mismatch_count"] for row in parent_raw["scenarios"])
    source_bindings = [_bind(path) for path in (RUNNER, CORE, TOP, SDC, CST, MEMORY_MANIFEST)]
    for row in json.loads(MEMORY_MANIFEST.read_text(encoding="utf-8"))["files"]:
        source_bindings.append(_bind(ROOT / row["path"]))
    netlist = build_dir / "gkp_fast_path_gw2a.json"
    routed_digests = [_bind(_seed_paths(build_dir, seed)["routed"]) for seed in SEEDS]
    version_env = os.environ.copy()
    version_cache = build_dir / "yowasp_cache"
    version_cache.mkdir(parents=True, exist_ok=True)
    version_env["YOWASP_CACHE_DIR"] = str(version_cache)
    report: dict[str, Any] = {
        "schema_version": "t5.5.2-target-device-synthesis-v1",
        "task_id": "T5.5.2",
        "status": "PENDING",
        "verdict": VERDICT,
        "target_contract": {
            "board_class": "Sipeed Tang Nano 20K",
            "device": DEVICE,
            "family": FAMILY,
            "top": TOP_MODULE,
            "target_mhz": TARGET_MHZ,
            "constraints": {"sdc": _relative(SDC), "cst": _relative(CST)},
            "harness_scope": "small-pin activity/observability harness, not T6 transport",
        },
        "source_bindings": source_bindings,
        "parent_equivalence": {
            "artifact": _bind(EQUIVALENCE_REPORT),
            "status": parent_raw["status"],
            "verdict": parent_raw["verdict"],
            "mismatch_count": parent_mismatches,
            "map_valid_rows": sum(row["map_valid_rows"] for row in parent_raw["scenarios"]),
        },
        "tools": {
            "yosys_executable": str(tools["yosys"]),
            "nextpnr_executable": str(tools["nextpnr"]),
            "yosys_version": _version_text(_run((tools["yosys"], "-V"), env=version_env)),
            "nextpnr_version": _version_text(_run((tools["nextpnr"], "--version"), env=version_env)),
            "packages": {
                name: _package_version(name)
                for name in ("yowasp-yosys", "yowasp-nextpnr-himbaechel-gowin", "apycula", "yowasp-runtime")
            },
            "commands": tool_commands(build_dir, tools),
            "recorded_runtime_seconds": timings or {},
        },
        "synthesis": parse_yosys_log(synthesis_log),
        "place_route": routes,
        "summary": {
            "seed_count": len(routes),
            "seeds": list(SEEDS),
            "fmax_mhz": {
                "minimum": min(fmax),
                "median": statistics.median(fmax),
                "maximum": max(fmax),
                "spread": max(fmax) - min(fmax),
            },
            "target_margin_mhz_worst_seed": min(fmax) - TARGET_MHZ,
            "all_seeds_pass_target": all(value >= TARGET_MHZ for value in fmax),
        },
        "latency_estimate": {
            "core_cycles": CORE_LATENCY_CYCLES,
            "initiation_interval_cycles": INITIATION_INTERVAL_CYCLES,
            "at_target_27mhz_ns": CORE_LATENCY_CYCLES * 1000.0 / TARGET_MHZ,
            "initiation_interval_at_target_ns": 1000.0 / TARGET_MHZ,
            "at_worst_seed_fmax_ns": CORE_LATENCY_CYCLES * 1000.0 / min(fmax),
            "excludes_harness_transport_adc_cdc_and_physical_actuation": True,
        },
        "build_digests_not_required_for_checkout": {
            "synthesis_netlist": _bind(netlist),
            "routed_netlists": routed_digests,
        },
        "durable_artifacts": artifacts,
        "evidence_boundary": {
            "synthesizable_rtl": True,
            "target_device_synthesis": True,
            "target_device_place_route": True,
            "bitstream_generated": False,
            "vendor_timing_signoff": False,
            "board_measured": False,
            "transport_implemented": False,
            "power_measured": False,
            "quantum_hardware_measured": False,
        },
    }
    report["gates"] = evaluate_gates(report)
    report["mutation_audit"] = mutation_audit(report)
    mutation_gate = all(row["rejected"] for row in report["mutation_audit"])
    report["gates"]["all_semantic_shortcut_mutations_are_rejected"] = mutation_gate
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    return report


def source_data_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(section: str, name: str, value: Any, *, seed: Any = "", unit: str = "", path: str = "", detail: str = "") -> None:
        rows.append({"section": section, "seed": seed, "name": name, "value": value, "unit": unit, "path": path, "detail": detail})
    for row in report["source_bindings"]:
        add("source_binding", "sha256", row["sha256"], path=row["path"], detail=f"bytes={row['bytes']}")
    for name, value in report["synthesis"]["cell_counts"].items():
        add("synthesis_cell", name, value, unit="cells")
    for route in report["place_route"]:
        seed = route["seed"]
        add("timing", "achieved_fmax", route["achieved_fmax_mhz"], seed=seed, unit="MHz")
        for name, value in route["critical_path"].items():
            add("critical_path", name, value, seed=seed, unit="ns" if name.endswith("_ns") else "")
        for name, value in route["utilization"].items():
            add("resource", name, value["used"], seed=seed, unit="cells", detail=f"available={value['available']}")
    for name, value in report["gates"].items():
        add("gate", name, int(value), detail="PASS" if value else "FAIL")
    for row in report["mutation_audit"]:
        add("mutation", row["mutation"], int(row["rejected"]), detail=";".join(row["failed_gates"]))
    for row in report["durable_artifacts"]:
        add("artifact", "sha256", row["sha256"], path=row["path"], detail=f"bytes={row['bytes']}")
    return rows


def write_outputs(report: dict[str, Any], json_path: Path, csv_path: Path) -> None:
    rows = source_data_rows(report)
    report["source_data"] = {"path": _relative(csv_path), "rows": len(rows)}
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "seed", "name", "value", "unit", "path", "detail"))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--run-tools", action="store_true", help="rerun synthesis and all P&R seeds before reporting")
    args = parser.parse_args(argv)
    build_dir = args.build_dir.resolve()
    tools = discover_tools()
    timings = run_toolchain(build_dir, tools) if args.run_tools else None
    report = build_report(build_dir, tools, timings)
    write_outputs(report, args.output_json.resolve(), args.output_csv.resolve())
    print(json.dumps({
        "status": report["status"],
        "verdict": report["verdict"],
        "fmax_mhz": report["summary"]["fmax_mhz"],
        "gates": report["gate_summary"],
        "output": _relative(args.output_json.resolve()),
    }, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
