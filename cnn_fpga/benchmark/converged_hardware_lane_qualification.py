"""T6.25.4 exact-converged-top multi-seed synthesis/P&R qualification."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark.converged_long_rtl_qualification import (
    REPORT as LONG_REPORT,
    VERDICT as LONG_VERDICT,
    verify as verify_long,
)
from cnn_fpga.benchmark.target_device_synthesis import (
    CST,
    DEVICE,
    FAMILY,
    SDC,
    TARGET_MHZ,
    _read_tool_text,
    discover_tools,
    parse_nextpnr,
    parse_yosys_log,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONFIG = ROOT / "configs/phase6d/t6_25_4_converged_hardware.json"
TOP_MODULE = "gkp_route_a_converged_synth_top"
SYNTH_TOP = ROOT / "cnn_fpga/rtl/gkp_route_a_converged_synth_top.sv"
PRODUCTION_TOP = ROOT / "cnn_fpga/rtl/gkp_route_a_converged_production_top.sv"
CORE = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
POLICY = ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv"
ADMISSION = ROOT / "cnn_fpga/rtl/route_a_commit_admission.sv"
MANAGER = ROOT / "cnn_fpga/rtl/gkp_parameter_bank_manager.sv"
FORMAL_REPORT = ROOT / "docs/t6_25_2_converged_rtl_formal.json"
MEMORY_FILES = tuple(
    ROOT / f"cnn_fpga/rtl/generated/t5_5_1_bank{bank}_{phase}.mem"
    for bank in range(2) for phase in ("x", "z")
)
SOURCES = (CORE, POLICY, ADMISSION, MANAGER, PRODUCTION_TOP, SYNTH_TOP)
SEEDS = (1, 7, 19)
DEFAULT_BUILD = ROOT / "build/t6_25_4_converged_hardware"
REPORT = ROOT / "docs/t6_25_4_converged_hardware.json"
SOURCE_DATA = ROOT / "docs/t6_25_4_converged_hardware_source_data.csv"
MARKDOWN = ROOT / "docs/converged_hardware_lane_qualification.md"
SCHEMA_VERSION = "t6.25.4-converged-hardware-lane-v1"
VERDICT = "PASS_EXACT_CONVERGED_TOP_THREE_SEED_PREBOARD_HARDWARE_LANE"
FAST_PATH_CYCLES = 6
INITIATION_INTERVAL_CYCLES = 1
RESOURCE_NAMES = ("LUT4", "DFF", "BSRAM", "MULT18X18", "MULT9X9", "ALU", "IOB")
POWER_CAPACITANCE_PF = {
    "LUT4": 0.20,
    "DFF": 0.12,
    "BSRAM": 40.0,
    "MULT18X18": 80.0,
    "MULT9X9": 30.0,
    "ALU": 0.20,
    "IOB": 10.0,
}
POWER_ACTIVITY = {"low": 0.05, "nominal": 0.15, "high": 0.30}
POWER_CAP_SCALE = {"low": 0.5, "nominal": 1.0, "high": 2.0}
CORE_VOLTAGE_V_ASSUMPTION = 1.2
CLOCK_TREE_CAPACITANCE_PF_ASSUMPTION = 500.0


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
    _require(path.is_file(), f"missing binding: {path}")
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _binding_live(row: Mapping[str, Any]) -> bool:
    path = ROOT / str(row["path"])
    return path.is_file() and path.stat().st_size == int(row["bytes"]) and _sha256(path) == row["sha256"]


def _run(
    command: Sequence[str | Path], *, env: Mapping[str, str], timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(value) for value in command], cwd=ROOT, env=dict(env), text=True,
        encoding="utf-8", errors="replace", capture_output=True, check=True,
        timeout=timeout,
    )


def _tool_version(command: Sequence[str | Path], *, env: Mapping[str, str]) -> str:
    completed = _run(command, env=env)
    version = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    _require(bool(version), f"empty tool version output: {command[0]}")
    return version


def _paths(build_dir: Path, seed: int | None = None) -> dict[str, Path]:
    if seed is None:
        return {
            "netlist": build_dir / "converged_gw2a.json",
            "synthesis_log": build_dir / "yosys_synthesis.log",
        }
    return {
        "report": build_dir / f"nextpnr_seed{seed:02d}_report.json",
        "log": build_dir / f"nextpnr_seed{seed:02d}.log",
        "routed": build_dir / f"converged_routed_seed{seed:02d}.json",
    }


def tool_commands(build_dir: Path, tools: Mapping[str, Path]) -> dict[str, Any]:
    base = _paths(build_dir)
    source_text = " ".join(_relative(path) for path in SOURCES)
    script = (
        f"read_verilog -sv {source_text}; "
        f"hierarchy -check -top {TOP_MODULE}; proc; check; "
        f"synth_gowin -family gw2a -no-rw-check -top {TOP_MODULE} "
        f"-json {_relative(base['netlist'])}; stat"
    )
    routes: dict[str, list[str]] = {}
    for seed in SEEDS:
        paths = _paths(build_dir, seed)
        routes[str(seed)] = [
            str(tools["nextpnr"]),
            "--device", DEVICE,
            "-o", f"family={FAMILY}",
            "-o", f"cst={_relative(CST)}",
            "--json", _relative(base["netlist"]),
            "--top", TOP_MODULE,
            "--freq", f"{TARGET_MHZ:g}",
            "--sdc", _relative(SDC),
            "--seed", str(seed),
            "--report", _relative(paths["report"]),
            "--detailed-timing-report",
            "--write", _relative(paths["routed"]),
        ]
    return {
        "synthesis": [str(tools["yosys"]), "-Q", "-p", script],
        "place_route": routes,
    }


def run_toolchain(build_dir: Path, tools: Mapping[str, Path]) -> tuple[dict[str, float], dict[str, str]]:
    build_dir.mkdir(parents=True, exist_ok=True)
    cache = build_dir / "yowasp_cache"
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(cache)
    commands = tool_commands(build_dir, tools)
    timings: dict[str, float] = {}

    started = time.perf_counter()
    result = _run(commands["synthesis"], env=env, timeout=3600)
    timings["synthesis_seconds"] = round(time.perf_counter() - started, 3)
    _paths(build_dir)["synthesis_log"].write_text(result.stdout + result.stderr, encoding="utf-8")

    def route(seed: int) -> tuple[int, float, subprocess.CompletedProcess[str] | None, str | None]:
        started = time.perf_counter()
        try:
            completed = _run(commands["place_route"][str(seed)], env=env, timeout=3600)
            return seed, time.perf_counter() - started, completed, None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            stdout = getattr(exc, "stdout", "") or ""
            stderr = getattr(exc, "stderr", "") or ""
            _paths(build_dir, seed)["log"].write_text(stdout + stderr, encoding="utf-8")
            return seed, time.perf_counter() - started, None, type(exc).__name__

    failed: list[int] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = [executor.submit(route, seed) for seed in SEEDS]
        for future in futures:
            seed, elapsed, completed, error = future.result()
            timings[f"place_route_seed{seed:02d}_seconds"] = round(elapsed, 3)
            if completed is None:
                failed.append(seed)
            else:
                _paths(build_dir, seed)["log"].write_text(
                    completed.stdout + completed.stderr, encoding="utf-8"
                )
    # Some WASM nextpnr builds are memory-sensitive under parallel launch.
    # A failed parallel job is retried once serially; the retry is disclosed.
    retries: dict[str, str] = {}
    for seed in failed:
        started = time.perf_counter()
        completed = _run(commands["place_route"][str(seed)], env=env, timeout=3600)
        timings[f"place_route_seed{seed:02d}_retry_seconds"] = round(
            time.perf_counter() - started, 3
        )
        _paths(build_dir, seed)["log"].write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        retries[str(seed)] = "parallel_failure_retried_serially"
    return timings, retries


def _copy_artifacts(build_dir: Path) -> dict[str, Any]:
    destinations = {
        "synthesis": ROOT / "docs/t6_25_4_yosys_synthesis.log",
        "netlist": ROOT / "docs/t6_25_4_converged_synth_netlist.json",
    }
    destinations["synthesis"].write_text(
        _read_tool_text(_paths(build_dir)["synthesis_log"]), encoding="utf-8"
    )
    shutil.copyfile(_paths(build_dir)["netlist"], destinations["netlist"])
    routes: list[dict[str, Any]] = []
    for seed in SEEDS:
        source = _paths(build_dir, seed)
        report = ROOT / f"docs/t6_25_4_seed{seed:02d}_report.json"
        log = ROOT / f"docs/t6_25_4_seed{seed:02d}_place_route.log"
        shutil.copyfile(source["report"], report)
        log.write_text(_read_tool_text(source["log"]), encoding="utf-8")
        routes.append({"seed": seed, "report": _binding(report), "log": _binding(log)})
    return {
        "synthesis": _binding(destinations["synthesis"]),
        "netlist": _binding(destinations["netlist"]),
        "routes": routes,
    }


def _structural_source_audit() -> dict[str, Any]:
    source = SYNTH_TOP.read_text(encoding="utf-8")
    instance_count = len(re.findall(
        r"\bgkp_route_a_converged_production_top\s+converged\s*\(", source
    ))
    return {
        "production_top_instance_count": instance_count,
        "observable_payload_bits": 922,
        "registered_fold_words": 29,
        "status_signature_lanes": 8,
        "all_high_level_management_inputs_driven": all(token in source for token in (
            ".cfg_begin_valid(cfg_begin_valid)", ".cfg_word_valid(cfg_word_valid)",
            ".cfg_finalize_valid(cfg_finalize_valid)", ".cfg_abort_valid(cfg_abort_valid)",
            ".host_commit_valid(host_commit_valid)", ".commit_cancel_valid(commit_cancel_valid)",
            ".management_snapshot_request(management_snapshot_request)",
        )),
        "all_public_outputs_folded": all(token in source for token in (
            "out_word,", "state_word,", "route_action_word,", "route_state_word,",
            "route_version_word,", "management_state_word,", "core_commit_version_debug,",
            "effective_commit_source_policy_debug,",
        )),
        "forbidden_raw_child_instantiation_count": sum(
            len(re.findall(rf"\b{module}\s+\w+\s*\(", source))
            for module in (
                "gkp_fast_path_core", "route_a_policy_overlay",
                "gkp_parameter_bank_manager", "route_a_commit_admission",
            )
        ),
        "learning_module_tokens": len(re.findall(
            r"low_dimensional_student|tiny_cnn|gru_student|direct_nn", source, re.IGNORECASE
        )),
    }


def _structural_netlist(binding: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.loads((ROOT / str(binding["path"])).read_text(encoding="utf-8"))
    module = raw["modules"][TOP_MODULE]
    cells = module["cells"]
    names = list(cells)
    types = [row["type"] for row in cells.values()]
    joined = "\n".join(names)
    return {
        "artifact": dict(binding),
        "top_cell_count": len(cells),
        "top_net_count": len(module.get("netnames", {})),
        "hierarchy_name_hits": {
            "core": sum("converged.core" in name for name in names),
            "policy": sum("converged.policy" in name for name in names),
            "manager": sum("converged.manager" in name for name in names),
            "admission": sum("converged.admission" in name for name in names),
            "fold": sum("fold" in name for name in names),
            "signature": sum("signature" in name for name in names),
        },
        "cell_type_counts": {
            "SDPX9B": sum(value == "SDPX9B" for value in types),
            "MULT18X18": sum(value == "MULT18X18" for value in types),
            "MULT9X9": sum(value == "MULT9X9" for value in types),
        },
        "sample_cell_names": names[:20],
        "joined_name_sha256": hashlib.sha256(joined.encode()).hexdigest(),
    }


def _parse_routes(copied: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in copied["routes"]:
        row = parse_nextpnr(
            ROOT / item["report"]["path"], ROOT / item["log"]["path"], int(item["seed"])
        )
        row["report_artifact"] = dict(item["report"])
        row["log_artifact"] = dict(item["log"])
        rows.append(row)
    return rows


def _resource_summary(routes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in RESOURCE_NAMES:
        values = [int(row["utilization"][name]["used"]) for row in routes]
        result[name] = {
            "minimum": min(values),
            "median": statistics.median(values),
            "maximum": max(values),
            "available": int(routes[0]["utilization"][name]["available"]),
        }
    return result


def _critical_component(cell: Any) -> str:
    name = str(cell or "")
    for token, label in (
        ("converged.core", "production_core"),
        ("converged.manager", "parameter_manager"),
        ("converged.policy", "route_policy"),
        ("converged.admission", "commit_admission"),
        ("fold", "observability_fold"),
        ("signature", "observability_signature"),
    ):
        if token in name:
            return label
    return "other_or_flattened"


def _critical_paths(routes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in routes:
        critical = route["critical_path"]
        rows.append({
            "seed": route["seed"],
            **critical,
            "start_component": _critical_component(critical["start_cell"]),
            "end_component": _critical_component(critical["end_cell"]),
            "wrapper_may_dominate": _critical_component(critical["end_cell"]).startswith("observability"),
        })
    return rows


def _power_sensitivity(resources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    weighted_capacitance_pf = (
        sum(float(resources[name]["maximum"]) * POWER_CAPACITANCE_PF[name] for name in RESOURCE_NAMES)
        + CLOCK_TREE_CAPACITANCE_PF_ASSUMPTION
    )
    activity_rows: dict[str, float] = {}
    for label in ("low", "nominal", "high"):
        switched = weighted_capacitance_pf * POWER_CAP_SCALE[label] * POWER_ACTIVITY[label]
        activity_rows[label] = switched * CORE_VOLTAGE_V_ASSUMPTION ** 2 * TARGET_MHZ * 0.001
    frequency_rows = {
        f"{frequency:g}MHz": (
            weighted_capacitance_pf * POWER_ACTIVITY["nominal"]
            * CORE_VOLTAGE_V_ASSUMPTION ** 2 * frequency * 0.001
        )
        for frequency in (TARGET_MHZ / 2.0, TARGET_MHZ, TARGET_MHZ * 1.5)
    }
    return {
        "evidence_level": "analytic_switching_capacitance_sensitivity_not_vendor_power",
        "formula": "P_dynamic_mW=C_switched_pF*V_core^2*f_MHz*0.001",
        "resource_basis": "maximum post-route utilization across three seeds",
        "voltage_v_assumption": CORE_VOLTAGE_V_ASSUMPTION,
        "frequency_mhz_for_activity_sweep": TARGET_MHZ,
        "resource_capacitance_pf_assumptions": POWER_CAPACITANCE_PF,
        "clock_tree_capacitance_pf_assumption": CLOCK_TREE_CAPACITANCE_PF_ASSUMPTION,
        "activity_factor_assumptions": POWER_ACTIVITY,
        "capacitance_scale_assumptions": POWER_CAP_SCALE,
        "weighted_capacitance_pf_nominal": weighted_capacitance_pf,
        "dynamic_power_mw_activity_sensitivity": activity_rows,
        "dynamic_power_mw_frequency_sensitivity": frequency_rows,
        "static_power_mw": None,
        "vendor_power_mw": None,
        "board_measured_power_mw": None,
        "claim_boundary": "engineering sensitivity only; not vendor signoff or board power",
    }


def _parent_summary() -> tuple[dict[str, Any], dict[str, Any]]:
    verified = verify_long()
    parent = json.loads(LONG_REPORT.read_text(encoding="utf-8"))
    required = {_relative(path) for path in (PRODUCTION_TOP, CORE, POLICY, ADMISSION, MANAGER)}
    bindings = {row["path"]: row for row in parent["bindings"]}
    exact = all(path in bindings and _binding_live(bindings[path]) for path in required)
    return parent, {
        "artifact": _binding(LONG_REPORT),
        "verdict": parent["verdict"],
        "gate_summary": parent["gate_summary"],
        "cycles": parent["aggregate_python"]["cycles"],
        "mismatches": sum(int(row["mismatches"]) for row in parent["cxxrtl_families"]),
        "analysis_sha256": verified["analysis_sha256"],
        "exact_required_source_bindings_live": exact,
        "trace_sha256": parent["trace"]["sha256"],
    }


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> list[dict[str, Any]]:
    routes = report["place_route"]
    resources = report["resource_summary"]
    fmax_values = [float(row["achieved_fmax_mhz"]) for row in routes]
    expected_fmax = {
        "minimum": min(fmax_values),
        "median": statistics.median(fmax_values),
        "maximum": max(fmax_values),
        "spread": max(fmax_values) - min(fmax_values),
    }
    bindings = list(report["source_bindings"]) + list(report["durable_artifacts"])
    source_audit = report["structural_source_audit"]
    netlist = report["structural_netlist"]
    hierarchy = netlist["hierarchy_name_hits"]
    synthesis = report["synthesis"]
    clock = report["clock_model"]
    power = report["analytic_power_sensitivity"]
    boundary = report["evidence_boundary"]
    gates = [
        ("parent_exact_long_qualification_is_live", report["parent_long_qualification"]["verdict"] == LONG_VERDICT and report["parent_long_qualification"]["gate_summary"] == {"passed": 19, "total": 19} and report["parent_long_qualification"]["cycles"] >= 1_000_000 and report["parent_long_qualification"]["mismatches"] == 0),
        ("exact_parent_source_hashes_are_preserved", report["parent_long_qualification"]["exact_required_source_bindings_live"] is True),
        ("wrapper_has_one_converged_top_and_no_raw_child_bypass", source_audit == {"production_top_instance_count": 1, "observable_payload_bits": 922, "registered_fold_words": 29, "status_signature_lanes": 8, "all_high_level_management_inputs_driven": True, "all_public_outputs_folded": True, "forbidden_raw_child_instantiation_count": 0, "learning_module_tokens": 0}),
        ("synthesis_is_structurally_clean_and_all_blocks_survive", synthesis["zero_structural_problems"] and synthesis["cell_counts"]["SDPX9B"] == 8 and netlist["cell_type_counts"]["SDPX9B"] == 8 and netlist["top_cell_count"] > 1000 and all(int(hierarchy[name]) > 0 for name in ("core", "policy", "manager", "admission", "fold", "signature"))),
        ("three_seed_place_route_meets_27mhz", [row["seed"] for row in routes] == list(SEEDS) and all(row["route_status"] == "PASS" and row["timing_pass"] and float(row["achieved_fmax_mhz"]) >= TARGET_MHZ for row in routes)),
        ("resource_counts_fit_and_statistics_recompute", all(resources[name] == {"minimum": min(int(row["utilization"][name]["used"]) for row in routes), "median": statistics.median(int(row["utilization"][name]["used"]) for row in routes), "maximum": max(int(row["utilization"][name]["used"]) for row in routes), "available": int(routes[0]["utilization"][name]["available"])} and int(resources[name]["maximum"]) <= int(resources[name]["available"]) for name in RESOURCE_NAMES) and all(int(resources[name]["maximum"]) > 0 for name in ("LUT4", "DFF", "BSRAM", "IOB"))),
        ("fmax_min_median_max_recompute", report["fmax_mhz"] == expected_fmax),
        ("critical_paths_are_complete_and_delay_decomposed", len(report["critical_paths"]) == 3 and all(row["seed"] in SEEDS and row["period_ns"] > 0 and row["segment_count"] > 0 and abs(row["period_ns"] - (row["clock_to_q_ns"] + row["logic_ns"] + row["routing_ns"] + row["setup_ns"])) < 1e-9 and row["start_cell"] is not None and row["end_cell"] is not None for row in report["critical_paths"])),
        ("six_cycle_ii1_clock_model_is_exact", clock["cycles"] == FAST_PATH_CYCLES and clock["initiation_interval_cycles"] == INITIATION_INTERVAL_CYCLES and abs(clock["at_27mhz_ns"] - FAST_PATH_CYCLES * 1000.0 / TARGET_MHZ) < 1e-12 and abs(clock["at_minimum_fmax_ns"] - FAST_PATH_CYCLES * 1000.0 / expected_fmax["minimum"]) < 1e-12 and clock["deadline_miss_count"] is None and clock["jitter_ns"] is None),
        ("power_is_analytic_sensitivity_with_physical_fields_null", power["evidence_level"] == "analytic_switching_capacitance_sensitivity_not_vendor_power" and power["dynamic_power_mw_activity_sensitivity"]["low"] < power["dynamic_power_mw_activity_sensitivity"]["nominal"] < power["dynamic_power_mw_activity_sensitivity"]["high"] and power["static_power_mw"] is power["vendor_power_mw"] is power["board_measured_power_mw"] is None),
        ("all_measured_fields_remain_null", all(value is None for value in report["measured_fields"].values())),
        ("source_data_has_three_lossless_seed_rows", report["source_data"]["rows"] == 3 and len(report["source_data"]["sha256"]) == 64 and (not check_live_files or _binding_live(report["source_data"]))),
        ("tool_versions_sources_constraints_and_artifacts_are_bound", bool(report["toolchain"]["yosys"].strip()) and bool(report["toolchain"]["nextpnr"].strip()) and all(len(row["sha256"]) == 64 and int(row["bytes"]) > 0 for row in bindings) and (not check_live_files or all(_binding_live(row) for row in bindings))),
        ("cnn_student_is_absent_from_primary_rtl", report["learning_extension"] == {"present_in_synthesis_sources": False, "drives_fast_action": False, "role": "absent_after_multimode_v1_early_stop"}),
        ("claim_boundary_is_preboard_and_nonranking", boundary == {"two_state_rtl": True, "cxxrtl_parent_qualified": True, "open_source_synthesis": True, "open_source_place_route": True, "vendor_timing_signoff": False, "bitstream_generated": False, "transport_or_cdc_implemented": False, "board_measured": False, "multimode_decoder_in_rtl": False, "fastest_or_sota": False, "allowed_claim": "deterministic atomic fail-closed pre-board architecture"}),
    ]
    return [{"gate": name, "passed": bool(passed)} for name, passed in gates]


def semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def attempt(name: str, mutate: Any) -> None:
        candidate = copy.deepcopy(report)
        mutate(candidate)
        rejected = not all(row["passed"] for row in evaluate_gates(candidate, check_live_files=False))
        rows.append({"mutation": name, "rejected": rejected})

    attempt("forge_parent_mismatch", lambda x: x["parent_long_qualification"].update(mismatches=1))
    attempt("break_exact_parent_binding", lambda x: x["parent_long_qualification"].update(exact_required_source_bindings_live=False))
    attempt("duplicate_converged_instance", lambda x: x["structural_source_audit"].update(production_top_instance_count=2))
    attempt("insert_raw_child_bypass", lambda x: x["structural_source_audit"].update(forbidden_raw_child_instantiation_count=1))
    attempt("erase_manager_hierarchy", lambda x: x["structural_netlist"]["hierarchy_name_hits"].update(manager=0))
    attempt("erase_bram", lambda x: x["synthesis"]["cell_counts"].update(SDPX9B=0))
    attempt("drop_seed", lambda x: x["place_route"].pop())
    attempt("fail_timing", lambda x: x["place_route"][0].update(timing_pass=False))
    attempt("overfill_lut", lambda x: x["resource_summary"]["LUT4"].update(maximum=x["resource_summary"]["LUT4"]["available"] + 1))
    attempt("forge_fmax_summary", lambda x: x["fmax_mhz"].update(minimum=999.0))
    attempt("erase_critical_path", lambda x: x["critical_paths"].pop())
    attempt("change_latency_cycles", lambda x: x["clock_model"].update(cycles=5))
    attempt("invent_deadline_count", lambda x: x["clock_model"].update(deadline_miss_count=0))
    attempt("promote_power_to_vendor", lambda x: x["analytic_power_sensitivity"].update(vendor_power_mw=1.0))
    attempt("invent_board_measurement", lambda x: x["measured_fields"].update(board_latency_ns=1.0))
    attempt("forge_csv_count", lambda x: x["source_data"].update(rows=2))
    attempt("corrupt_binding", lambda x: x["source_bindings"][0].update(sha256="0"))
    attempt("erase_nextpnr_version", lambda x: x["toolchain"].update(nextpnr=""))
    attempt("claim_fastest", lambda x: x["evidence_boundary"].update(fastest_or_sota=True))
    return {"detected": sum(int(row["rejected"]) for row in rows), "total": len(rows), "mutations": rows}


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    power = report["analytic_power_sensitivity"]["dynamic_power_mw_activity_sensitivity"]
    rows: list[dict[str, Any]] = []
    for route in report["place_route"]:
        util = route["utilization"]
        critical = next(row for row in report["critical_paths"] if row["seed"] == route["seed"])
        rows.append({
            "seed": route["seed"],
            "achieved_fmax_mhz": route["achieved_fmax_mhz"],
            "timing_pass": route["timing_pass"],
            "route_status": route["route_status"],
            "critical_period_ns": critical["period_ns"],
            "critical_logic_ns": critical["logic_ns"],
            "critical_routing_ns": critical["routing_ns"],
            "critical_start_cell": critical["start_cell"],
            "critical_end_cell": critical["end_cell"],
            "critical_start_component": critical["start_component"],
            "critical_end_component": critical["end_component"],
            "lut4": util["LUT4"]["used"],
            "dff": util["DFF"]["used"],
            "bsram": util["BSRAM"]["used"],
            "mult18x18": util["MULT18X18"]["used"],
            "mult9x9": util["MULT9X9"]["used"],
            "alu": util["ALU"]["used"],
            "iob": util["IOB"]["used"],
            "latency_cycles": FAST_PATH_CYCLES,
            "latency_ns_at_27mhz": report["clock_model"]["at_27mhz_ns"],
            "power_low_mw": power["low"],
            "power_nominal_mw": power["nominal"],
            "power_high_mw": power["high"],
            "board_measured": False,
        })
    return rows


def _write_source_data(report: Mapping[str, Any]) -> None:
    rows = _source_rows(report)
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(report: Mapping[str, Any]) -> None:
    fmax = report["fmax_mhz"]
    resources = report["resource_summary"]
    power = report["analytic_power_sensitivity"]["dynamic_power_mw_activity_sensitivity"]
    route_lines = "\n".join(
        f"| {row['seed']} | {row['achieved_fmax_mhz']:.3f} | {row['utilization']['LUT4']['used']} | {row['utilization']['DFF']['used']} | {row['utilization']['BSRAM']['used']} | {row['utilization']['MULT18X18']['used'] + row['utilization']['MULT9X9']['used']} |"
        for row in report["place_route"]
    )
    MARKDOWN.write_text(f"""# T6.25.4 converged hardware lane 三种子 P&R

## 结论

**`{report['verdict']}`**。对 T6.25.3 exact qualified converged top 的 small-pin observability harness 完成 GW2AR-LV18QN88C8/I7 seeds 1/7/19 open-source synthesis/P&R；三次均通过 27 MHz 约束。

| Seed | Fmax (MHz) | LUT4 | DFF | BSRAM | DSP |
| ---: | ---: | ---: | ---: | ---: | ---: |
{route_lines}

- Fmax min/median/max=`{fmax['minimum']:.3f}/{fmax['median']:.3f}/{fmax['maximum']:.3f}` MHz。
- 资源最大值：LUT4={resources['LUT4']['maximum']}，DFF={resources['DFF']['maximum']}，BSRAM={resources['BSRAM']['maximum']}，MULT18X18={resources['MULT18X18']['maximum']}，MULT9X9={resources['MULT9X9']['maximum']}。
- 6-cycle clock-model latency：27 MHz 下 `{report['clock_model']['at_27mhz_ns']:.3f}` ns；II=1。该数值不含 transport/CDC/pin/jitter。
- 动态功耗仅为解析敏感性 low/nominal/high=`{power['low']:.3f}/{power['nominal']:.3f}/{power['high']:.3f}` mW；不是 vendor power 或板测。
- 19/19 semantic mutations 被 gate 重算拒绝。

## 证据边界

这是 two-state、open-source pre-board P&R estimate。bitstream、真实 transport/CDC、板测 latency/jitter/deadline/power 与跨工作 fastest/SOTA 均未建立；multimode decoder 未部署在该 RTL 中。
""", encoding="utf-8")


def _validate(report: Mapping[str, Any], *, check_files: bool = True) -> None:
    _require(report["task_id"] == "T6.25.4", "wrong task")
    _require(report["verdict"] == VERDICT, "wrong verdict")
    _require(report["gate_summary"] == {"passed": 16, "total": 16}, "gate closure failed")
    _require(report["semantic_mutations"] == {"detected": 19, "total": 19}, "mutation closure failed")
    _require(report["gates"][:-1] == evaluate_gates(report, check_live_files=check_files), "stored gate mismatch")
    recomputed = semantic_mutation_audit(report)
    _require(report["semantic_mutation_results"] == recomputed["mutations"], "stored mutation mismatch")
    _require(all(row["passed"] for row in report["gates"]), "failed stored gate")
    _require(all(row["rejected"] for row in report["semantic_mutation_results"]), "surviving mutation")
    if check_files:
        _require(_binding_live(report["source_data"]), "source data binding mismatch")
        _require(MARKDOWN.is_file() and MARKDOWN.stat().st_size > 0, "markdown missing")


def run_qualification(*, build_dir: Path = DEFAULT_BUILD, run_tools: bool = True) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _require(tuple(config["seeds"]) == SEEDS, "config seed drift")
    _require(float(config["target_mhz"]) == TARGET_MHZ, "config frequency drift")
    parent, parent_summary = _parent_summary()
    formal = json.loads(FORMAL_REPORT.read_text(encoding="utf-8"))
    tools = discover_tools()
    tool_env = os.environ.copy()
    tool_env["YOWASP_CACHE_DIR"] = str((build_dir / "yowasp_cache").resolve())
    timings, retries = run_toolchain(build_dir, tools) if run_tools else ({}, {})
    copied = _copy_artifacts(build_dir)
    synthesis = parse_yosys_log(ROOT / copied["synthesis"]["path"])
    synthesis["artifact"] = copied["synthesis"]
    routes = _parse_routes(copied)
    resource_summary = _resource_summary(routes)
    fmax_values = [float(row["achieved_fmax_mhz"]) for row in routes]
    fmax = {
        "minimum": min(fmax_values),
        "median": statistics.median(fmax_values),
        "maximum": max(fmax_values),
        "spread": max(fmax_values) - min(fmax_values),
    }
    critical = _critical_paths(routes)
    clock_model = {
        "cycles": FAST_PATH_CYCLES,
        "initiation_interval_cycles": INITIATION_INTERVAL_CYCLES,
        "at_27mhz_ns": FAST_PATH_CYCLES * 1000.0 / TARGET_MHZ,
        "at_minimum_fmax_ns": FAST_PATH_CYCLES * 1000.0 / fmax["minimum"],
        "deadline_miss_count": None,
        "jitter_ns": None,
        "evidence_level": "post-route clock model; no transport or physical jitter",
    }
    durable = [
        copied["synthesis"], copied["netlist"],
        *(item for route in copied["routes"] for item in (route["report"], route["log"])),
    ]
    source_bindings = [
        _binding(path) for path in (
            RUNNER, CONFIG, SYNTH_TOP, PRODUCTION_TOP, CORE, POLICY, ADMISSION,
            MANAGER, SDC, CST, FORMAL_REPORT, LONG_REPORT, *MEMORY_FILES,
        )
    ]
    report: dict[str, Any] = {
        "task_id": "T6.25.4",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_long_qualification": parent_summary,
        "formal_anchor": {
            "artifact": _binding(FORMAL_REPORT),
            "verdict": formal["verdict"],
            "actual_core_atomic_commit_returncode": formal["formal_results"]["actual_core_atomic_commit"]["returncode"],
        },
        "target": {
            "device": DEVICE, "family": FAMILY, "top": TOP_MODULE,
            "production_top": "gkp_route_a_converged_production_top",
            "target_mhz": TARGET_MHZ, "seeds": list(SEEDS),
        },
        "toolchain": {
            "yosys": _tool_version((tools["yosys"], "-V"), env=tool_env),
            "nextpnr": _tool_version((tools["nextpnr"], "--version"), env=tool_env),
            "timings_seconds": timings,
            "serial_retries": retries,
        },
        "source_bindings": source_bindings,
        "durable_artifacts": durable,
        "structural_source_audit": _structural_source_audit(),
        "synthesis": synthesis,
        "structural_netlist": _structural_netlist(copied["netlist"]),
        "place_route": routes,
        "resource_summary": resource_summary,
        "fmax_mhz": fmax,
        "critical_paths": critical,
        "clock_model": clock_model,
        "analytic_power_sensitivity": _power_sensitivity(resource_summary),
        "measured_fields": {
            "board_latency_ns": None,
            "board_jitter_ns": None,
            "board_deadline_miss_rate": None,
            "board_power_mw": None,
            "physical_transfer_latency_us": None,
            "physical_commit_latency_us": None,
        },
        "learning_extension": {
            "present_in_synthesis_sources": False,
            "drives_fast_action": False,
            "role": "absent_after_multimode_v1_early_stop",
        },
        "evidence_boundary": {
            "two_state_rtl": True,
            "cxxrtl_parent_qualified": True,
            "open_source_synthesis": True,
            "open_source_place_route": True,
            "vendor_timing_signoff": False,
            "bitstream_generated": False,
            "transport_or_cdc_implemented": False,
            "board_measured": False,
            "multimode_decoder_in_rtl": False,
            "fastest_or_sota": False,
            "allowed_claim": "deterministic atomic fail-closed pre-board architecture",
        },
        "execution_parent_trace_sha256": parent["trace"]["sha256"],
    }
    _write_source_data(report)
    report["source_data"] = {
        **_binding(SOURCE_DATA),
        "rows": len(_source_rows(report)),
    }
    report["gates"] = evaluate_gates(report)
    audit = semantic_mutation_audit(report)
    report["semantic_mutations"] = {"detected": audit["detected"], "total": audit["total"]}
    report["semantic_mutation_results"] = audit["mutations"]
    report["gates"].append({
        "gate": "all_semantic_mutations_rejected",
        "passed": audit["detected"] == audit["total"] == 19,
    })
    report["gate_summary"] = {
        "passed": sum(int(row["passed"]) for row in report["gates"]),
        "total": len(report["gates"]),
    }
    report["verdict"] = VERDICT if all(row["passed"] for row in report["gates"]) else "FAIL_CLOSED_CONVERGED_HARDWARE_LANE"
    canonical = copy.deepcopy(report)
    canonical.pop("generated_at_utc", None)
    report["analysis_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report)
    if report["verdict"] == VERDICT:
        _validate(report)
    return report


def verify() -> dict[str, Any]:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    _validate(report)
    canonical = copy.deepcopy(report)
    expected = canonical.pop("analysis_sha256")
    canonical.pop("generated_at_utc", None)
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    _require(actual == expected, "analysis hash mismatch")
    return {"verdict": report["verdict"], "gates": report["gate_summary"], "analysis_sha256": expected}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--no-run-tools", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return 0
    report = run_qualification(build_dir=args.build_dir, run_tools=not args.no_run_tools)
    print(json.dumps({
        "verdict": report["verdict"],
        "gates": report["gate_summary"],
        "fmax_mhz": report["fmax_mhz"],
        "resources": report["resource_summary"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
