"""T6.9.1 integrated Route-A synthesis/P&R and preboard Pareto evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any, Mapping, Sequence

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
TASK_ID = "T6.9.1"
SCHEMA_VERSION = "t6.9.1-route-a-hardware-pareto-v1"
TOP_MODULE = "route_a_hardware_pareto_synth_top"
SEEDS = (1, 7, 19)
PROFILES = {
    "route_a_core_no_student": 0,
    "route_a_plus_student_sidecar": 1,
}
SOURCES = (
    ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv",
    ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv",
    ROOT / "cnn_fpga/rtl/route_a_integrated_qualification_top.sv",
    ROOT / "cnn_fpga/rtl/low_dimensional_student_kernel.sv",
    ROOT / "cnn_fpga/rtl/route_a_hardware_pareto_synth_top.sv",
)
PARENT_RTL = ROOT / "docs/t6_7_3_route_a_integrated_rtl_qualification.json"
EXECUTION_CONTRACT = ROOT / "docs/t6_5_2_unified_execution_contract.json"
STUDENT_EQUIVALENCE = ROOT / "docs/t5_5_3_student_rtl_equivalence.json"
BASE_SYNTHESIS = ROOT / "docs/t5_5_2_target_device_synthesis.json"
DEFAULT_BUILD = ROOT / ".tmp_t691_build"
DEFAULT_ARTIFACT = ROOT / "docs/t6_9_1_route_a_hardware_pareto.json"
SOURCE_CSV = ROOT / "docs/t6_9_1_route_a_hardware_pareto_source_data.csv"
FAST_PATH_CYCLES = 6
INITIATION_INTERVAL_CYCLES = 1
STUDENT_CYCLES = 64

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.is_file() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _paths(build_dir: Path, profile: str, seed: int | None = None) -> dict[str, Path]:
    root = build_dir / profile
    if seed is None:
        return {
            "root": root,
            "netlist": root / "route_a_gw2a.json",
            "synth_log": root / "yosys_synthesis.log",
        }
    return {
        "report": root / f"nextpnr_seed{seed:02d}_report.json",
        "log": root / f"nextpnr_seed{seed:02d}.log",
        "routed": root / f"route_a_routed_seed{seed:02d}.json",
    }


def _commands(build_dir: Path, tools: Mapping[str, Path]) -> dict[str, Any]:
    commands: dict[str, Any] = {"synthesis": {}, "place_route": {}}
    source_text = " ".join(_relative(path) for path in SOURCES)
    for profile, enabled in PROFILES.items():
        paths = _paths(build_dir, profile)
        script = (
            f"read_verilog -sv {source_text}; "
            f"chparam -set ENABLE_STUDENT {enabled} {TOP_MODULE}; "
            f"hierarchy -check -top {TOP_MODULE}; proc; check; "
            f"synth_gowin -family gw2a -no-rw-check -top {TOP_MODULE} -json {_relative(paths['netlist'])}; stat"
        )
        commands["synthesis"][profile] = [str(tools["yosys"]), "-Q", "-p", script]
        commands["place_route"][profile] = {}
        for seed in SEEDS:
            seed_paths = _paths(build_dir, profile, seed)
            commands["place_route"][profile][str(seed)] = [
                str(tools["nextpnr"]),
                "--device", DEVICE,
                "-o", f"family={FAMILY}",
                "-o", f"cst={_relative(CST)}",
                "--json", _relative(paths["netlist"]),
                "--top", TOP_MODULE,
                "--freq", f"{TARGET_MHZ:g}",
                "--sdc", _relative(SDC),
                "--seed", str(seed),
                "--report", _relative(seed_paths["report"]),
                "--detailed-timing-report",
                "--write", _relative(seed_paths["routed"]),
            ]
    return commands


def _run(command: Sequence[str], env: Mapping[str, str], timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in command], cwd=ROOT, env=dict(env), text=True,
        encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, check=True,
    )


def run_toolchain(build_dir: Path, tools: Mapping[str, Path]) -> dict[str, float]:
    build_dir.mkdir(parents=True, exist_ok=True)
    cache = build_dir / "yowasp_cache"
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["YOWASP_CACHE_DIR"] = str(cache)
    commands = _commands(build_dir, tools)
    timings: dict[str, float] = {}

    for profile in PROFILES:
        paths = _paths(build_dir, profile)
        paths["root"].mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = _run(commands["synthesis"][profile], env)
        timings[f"synthesis_{profile}_seconds"] = time.perf_counter() - started
        paths["synth_log"].write_text(result.stdout + result.stderr, encoding="utf-8")

    jobs = [(profile, seed) for profile in PROFILES for seed in SEEDS]

    def route(job: tuple[str, int]) -> tuple[str, int, float, str | None]:
        profile, seed = job
        started = time.perf_counter()
        try:
            result = _run(commands["place_route"][profile][str(seed)], env)
            error = None
        except subprocess.CalledProcessError as exc:
            result = exc
            error = f"returncode={exc.returncode}"
        seed_paths = _paths(build_dir, profile, seed)
        seed_paths["log"].write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
        return profile, seed, time.perf_counter() - started, error

    failures: list[tuple[str, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        for profile, seed, elapsed, error in executor.map(route, jobs):
            timings[f"place_route_{profile}_seed{seed}_seconds"] = elapsed
            if error:
                failures.append((profile, seed))
    for profile, seed in failures:
        started = time.perf_counter()
        result = _run(commands["place_route"][profile][str(seed)], env)
        timings[f"place_route_{profile}_seed{seed}_retry_seconds"] = time.perf_counter() - started
        seed_paths = _paths(build_dir, profile, seed)
        seed_paths["log"].write_text(result.stdout + result.stderr, encoding="utf-8")
    return timings


def _copy_artifacts(build_dir: Path) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for profile in PROFILES:
        profile_rows = {"synthesis": None, "netlist": None, "routes": []}
        source = _paths(build_dir, profile)["synth_log"]
        destination = ROOT / "docs" / f"t6_9_1_{profile}_yosys_synthesis.log"
        destination.write_text(_read_tool_text(source), encoding="utf-8")
        profile_rows["synthesis"] = _binding(destination)
        netlist_destination = ROOT / "docs" / f"t6_9_1_{profile}_synth_netlist.json"
        shutil.copyfile(_paths(build_dir, profile)["netlist"], netlist_destination)
        profile_rows["netlist"] = _binding(netlist_destination)
        for seed in SEEDS:
            seed_paths = _paths(build_dir, profile, seed)
            report_dest = ROOT / "docs" / f"t6_9_1_{profile}_seed{seed:02d}_report.json"
            log_dest = ROOT / "docs" / f"t6_9_1_{profile}_seed{seed:02d}_place_route.log"
            shutil.copyfile(seed_paths["report"], report_dest)
            log_dest.write_text(_read_tool_text(seed_paths["log"]), encoding="utf-8")
            profile_rows["routes"].append({"seed": seed, "report": _binding(report_dest), "log": _binding(log_dest)})
        copied[profile] = profile_rows
    return copied


def _parse_route(report_binding: Mapping[str, Any], log_binding: Mapping[str, Any], seed: int) -> dict[str, Any]:
    row = parse_nextpnr(ROOT / report_binding["path"], ROOT / log_binding["path"], seed)
    row["report_artifact"] = dict(report_binding)
    row["log_artifact"] = dict(log_binding)
    return row


def _structural_netlist(path: Path, enable_student: bool) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    module = raw["modules"][TOP_MODULE]
    cell_names = list(module["cells"])
    cell_types = [cell["type"] for cell in module["cells"].values()]
    joined_names = "\n".join(cell_names)
    return {
        "artifact": _binding(path),
        "top_cell_count": len(cell_names),
        "top_net_count": len(module.get("netnames", {})),
        "student_parameter": enable_student,
        "student_hierarchy_present": "student_sidecar" in joined_names,
        "policy_hierarchy_present": "integrated.policy" in joined_names or "policy" in joined_names,
        "core_hierarchy_present": "integrated.core" in joined_names or "core" in joined_names,
        "sdpx9b_cells": sum(cell == "SDPX9B" for cell in cell_types),
        "mult18x18_cells": sum(cell == "MULT18X18" for cell in cell_types),
        "mult9x9_cells": sum(cell == "MULT9X9" for cell in cell_types),
    }


def _max_resources(routes: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        name: {
            "used": max(int(row["utilization"][name]["used"]) for row in routes),
            "available": int(routes[0]["utilization"][name]["available"]),
        }
        for name in RESOURCE_NAMES
    }


def _power_estimate(resources: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    weighted_capacitance_pf = sum(
        float(resources[name]["used"]) * POWER_CAPACITANCE_PF[name]
        for name in RESOURCE_NAMES
    ) + CLOCK_TREE_CAPACITANCE_PF_ASSUMPTION
    estimates: dict[str, float] = {}
    for label in ("low", "nominal", "high"):
        switched_cap_pf = weighted_capacitance_pf * POWER_CAP_SCALE[label] * POWER_ACTIVITY[label]
        estimates[label] = switched_cap_pf * CORE_VOLTAGE_V_ASSUMPTION ** 2 * TARGET_MHZ * 0.001
    return {
        "evidence_level": "analytic_switching_capacitance_sensitivity_not_vendor_power",
        "formula": "P_dynamic_mW=C_switched_pF*V_core^2*f_MHz*0.001",
        "voltage_v_assumption": CORE_VOLTAGE_V_ASSUMPTION,
        "frequency_mhz": TARGET_MHZ,
        "resource_capacitance_pf_assumptions": POWER_CAPACITANCE_PF,
        "clock_tree_capacitance_pf_assumption": CLOCK_TREE_CAPACITANCE_PF_ASSUMPTION,
        "activity_factor_assumptions": POWER_ACTIVITY,
        "capacitance_scale_assumptions": POWER_CAP_SCALE,
        "weighted_capacitance_pf_nominal": weighted_capacitance_pf,
        "dynamic_power_mw_sensitivity": estimates,
        "static_power_mw": None,
        "vendor_power_mw": None,
        "board_measured_power_mw": None,
        "claim_boundary": "engineering sensitivity only; do not compare with literature or use as device power signoff",
    }


def _profile_summary(
    profile: str,
    enabled: bool,
    synthesis_binding: Mapping[str, Any],
    netlist_binding: Mapping[str, Any],
    route_bindings: list[Mapping[str, Any]],
    build_dir: Path,
    deadline_us: float,
) -> dict[str, Any]:
    synthesis = parse_yosys_log(ROOT / synthesis_binding["path"])
    synthesis["artifact"] = dict(synthesis_binding)
    routes = [_parse_route(item["report"], item["log"], int(item["seed"])) for item in route_bindings]
    fmax = [float(row["achieved_fmax_mhz"]) for row in routes]
    resources = _max_resources(routes)
    structural = _structural_netlist(ROOT / str(netlist_binding["path"]), enabled)
    latency_target_ns = FAST_PATH_CYCLES * 1000.0 / TARGET_MHZ
    latency_worst_fmax_ns = FAST_PATH_CYCLES * 1000.0 / min(fmax)
    return {
        "profile_id": profile,
        "student_sidecar_enabled": enabled,
        "student_drives_fast_action": False,
        "student_cycles": STUDENT_CYCLES if enabled else None,
        "student_latency_us_at_27mhz": STUDENT_CYCLES / TARGET_MHZ if enabled else None,
        "synthesis": synthesis,
        "structural_netlist": structural,
        "place_route": routes,
        "summary": {
            "seeds": list(SEEDS),
            "fmax_mhz": {"minimum": min(fmax), "median": statistics.median(fmax), "maximum": max(fmax), "spread": max(fmax) - min(fmax)},
            "resources_max_across_seeds": resources,
        },
        "critical_path_classification": {
            "all_start_in_integrated_core": all(str(row["critical_path"]["start_cell"]).startswith("integrated.core.") for row in routes),
            "all_end_in_observability_fold": all(str(row["critical_path"]["end_cell"]).startswith("fold") for row in routes),
            "interpretation": "conservative full-observability wrapper path from integrated core telemetry/state into registered fold; not a measured board source-to-action path",
        },
        "source_to_action_latency_model": {
            "cycles": FAST_PATH_CYCLES,
            "initiation_interval_cycles": INITIATION_INTERVAL_CYCLES,
            "at_enforced_27mhz_ns": latency_target_ns,
            "at_worst_seed_achieved_fmax_ns": latency_worst_fmax_ns,
            "fast_action_budget_us_assumption": deadline_us,
            "deadline_margin_us_at_27mhz": deadline_us - latency_target_ns / 1000.0,
            "deadline_margin_us_at_worst_seed_fmax": deadline_us - latency_worst_fmax_ns / 1000.0,
            "deadline_miss_count": None,
            "evidence_level": "P&R clock-model estimate; transport and physical-board jitter excluded",
        },
        "dynamic_power_estimate": _power_estimate(resources),
    }


def _write_csv(profiles: list[Mapping[str, Any]]) -> None:
    fields = [
        "profile_id", "student_sidecar_enabled", "seed", "achieved_fmax_mhz", "timing_pass",
        "latency_cycles", "latency_ns_at_27mhz", "deadline_margin_us_at_27mhz",
        "lut4", "dff", "bsram", "mult18x18", "mult9x9", "alu", "iob",
        "dynamic_power_low_mw", "dynamic_power_nominal_mw", "dynamic_power_high_mw",
        "resource_evidence_level", "power_evidence_level", "board_measured",
    ]
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            power = profile["dynamic_power_estimate"]["dynamic_power_mw_sensitivity"]
            latency = profile["source_to_action_latency_model"]
            for route in profile["place_route"]:
                util = route["utilization"]
                writer.writerow({
                    "profile_id": profile["profile_id"],
                    "student_sidecar_enabled": profile["student_sidecar_enabled"],
                    "seed": route["seed"],
                    "achieved_fmax_mhz": route["achieved_fmax_mhz"],
                    "timing_pass": route["timing_pass"],
                    "latency_cycles": latency["cycles"],
                    "latency_ns_at_27mhz": latency["at_enforced_27mhz_ns"],
                    "deadline_margin_us_at_27mhz": latency["deadline_margin_us_at_27mhz"],
                    "lut4": util["LUT4"]["used"], "dff": util["DFF"]["used"],
                    "bsram": util["BSRAM"]["used"], "mult18x18": util["MULT18X18"]["used"],
                    "mult9x9": util["MULT9X9"]["used"], "alu": util["ALU"]["used"],
                    "iob": util["IOB"]["used"],
                    "dynamic_power_low_mw": power["low"],
                    "dynamic_power_nominal_mw": power["nominal"],
                    "dynamic_power_high_mw": power["high"],
                    "resource_evidence_level": "three_seed_open_source_post_route_estimate",
                    "power_evidence_level": profile["dynamic_power_estimate"]["evidence_level"],
                    "board_measured": False,
                })


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    profiles = {row["profile_id"]: row for row in report["profiles"]}
    core = profiles.get("route_a_core_no_student", {})
    student = profiles.get("route_a_plus_student_sidecar", {})
    bindings = list(report["source_bindings"]) + list(report["durable_artifacts"])
    route_rows = [route for profile in profiles.values() for route in profile["place_route"]]
    core_res = core.get("summary", {}).get("resources_max_across_seeds", {})
    student_res = student.get("summary", {}).get("resources_max_across_seeds", {})
    power_rows = [profile["dynamic_power_estimate"] for profile in profiles.values()]
    return {
        "G01_parent_integrated_rtl_is_million_cycle_bit_exact": report["parents"]["t6_7_3"]["cycles"] == 1_000_000 and report["parents"]["t6_7_3"]["mismatches"] == report["parents"]["t6_7_3"]["undefined_actions"] == report["parents"]["t6_7_3"]["silent_overflow"] == 0,
        "G02_six_cycle_and_1p5us_contract_is_live": report["contract"]["source_to_action_cycles"] == 6 and report["contract"]["fast_action_budget_us_assumption"] == 1.5 and report["contract"]["board_deadline_field_must_be_null_before_measurement"] is True,
        "G03_all_sources_parents_constraints_and_artifacts_are_hash_bound": all(len(row["sha256"]) == 64 for row in bindings) and (not check_live_files or all(_live(row) for row in bindings)),
        "G04_two_real_elaboration_profiles_use_one_integrated_top": set(profiles) == set(PROFILES) and core.get("student_sidecar_enabled") is False and student.get("student_sidecar_enabled") is True and report["target"]["top"] == TOP_MODULE,
        "G05_policy_core_and_eight_ab_map_brams_survive_synthesis": all(profile["synthesis"]["zero_structural_problems"] and profile["synthesis"]["cell_counts"]["SDPX9B"] == 8 and profile["structural_netlist"]["sdpx9b_cells"] == 8 and profile["structural_netlist"]["top_cell_count"] > 100 and profile["structural_netlist"]["policy_hierarchy_present"] and profile["structural_netlist"]["core_hierarchy_present"] for profile in profiles.values()),
        "G06_optional_student_is_real_sidecar_and_not_fast_action": core.get("structural_netlist", {}).get("student_hierarchy_present") is False and student.get("structural_netlist", {}).get("student_hierarchy_present") is True and student.get("synthesis", {}).get("cell_counts", {}).get("MULT18X18", 0) > core.get("synthesis", {}).get("cell_counts", {}).get("MULT18X18", 0) and all(profile["student_drives_fast_action"] is False for profile in profiles.values()),
        "G07_six_place_route_runs_pass_target_timing": len(route_rows) == 6 and sorted((row["seed"] for row in route_rows)) == [1, 1, 7, 7, 19, 19] and all(row["route_status"] == "PASS" and row["timing_pass"] and row["achieved_fmax_mhz"] >= TARGET_MHZ for row in route_rows) and all(profile["critical_path_classification"]["all_start_in_integrated_core"] and profile["critical_path_classification"]["all_end_in_observability_fold"] for profile in profiles.values()),
        "G08_resources_are_nonzero_fit_and_student_increment_is_observed": all(core_res[name]["used"] > 0 and core_res[name]["used"] <= core_res[name]["available"] and student_res[name]["used"] > 0 and student_res[name]["used"] <= student_res[name]["available"] for name in RESOURCE_NAMES) and student_res["LUT4"]["used"] > core_res["LUT4"]["used"] and student_res["DFF"]["used"] > core_res["DFF"]["used"],
        "G09_fmax_min_median_max_are_recomputed": all(profile["summary"]["fmax_mhz"] == {"minimum": min(row["achieved_fmax_mhz"] for row in profile["place_route"]), "median": statistics.median(row["achieved_fmax_mhz"] for row in profile["place_route"]), "maximum": max(row["achieved_fmax_mhz"] for row in profile["place_route"]), "spread": max(row["achieved_fmax_mhz"] for row in profile["place_route"]) - min(row["achieved_fmax_mhz"] for row in profile["place_route"])} for profile in profiles.values()),
        "G10_six_cycle_latency_model_and_positive_deadline_margin_are_exact": all(profile["source_to_action_latency_model"]["cycles"] == 6 and profile["source_to_action_latency_model"]["initiation_interval_cycles"] == 1 and abs(profile["source_to_action_latency_model"]["at_enforced_27mhz_ns"] - 6 * 1000 / 27) < 1e-12 and profile["source_to_action_latency_model"]["deadline_margin_us_at_27mhz"] > 0 and profile["source_to_action_latency_model"]["deadline_miss_count"] is None for profile in profiles.values()),
        "G11_power_is_sensitivity_estimate_with_null_vendor_and_board_values": all(row["evidence_level"] == "analytic_switching_capacitance_sensitivity_not_vendor_power" and row["dynamic_power_mw_sensitivity"]["low"] < row["dynamic_power_mw_sensitivity"]["nominal"] < row["dynamic_power_mw_sensitivity"]["high"] and row["static_power_mw"] is row["vendor_power_mw"] is row["board_measured_power_mw"] is None for row in power_rows),
        "G12_pareto_selects_nonlearning_fast_path_without_hiding_sidecar": report["pareto_decision"] == {"selected_profile": "route_a_core_no_student", "student_profile_role": "optional_ablation_sidecar_not_primary_action_path", "selection_reason": "same six-cycle fast action with lower resources; CNN/student lacks matched primary evidence", "student_profile_reported": True},
        "G13_source_csv_contains_all_six_runs": report["source_data"]["rows"] == 6 and len(report["source_data"]["sha256"]) == 64 and (not check_live_files or _sha256(ROOT / report["source_data"]["path"]) == report["source_data"]["sha256"]),
        "G14_evidence_boundary_forbids_measured_speed_power_or_deadline": report["evidence_boundary"] == {"cxxrtl_qualified": True, "open_source_synthesis": True, "open_source_place_route": True, "vendor_timing_signoff": False, "bitstream_generated": False, "transport_implemented": False, "board_measured": False, "board_deadline_miss": None, "measured_source_to_action_ns": None, "measured_power_mw": None, "speed_advantage": "PROHIBITED_PENDING_T6.9.2"},
        "G15_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 15,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 15, "detected": 15, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    def profile(x: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
        return next(row for row in x["profiles"] if row["profile_id"] == profile_id)

    attempt("forge_parent_mismatch", "G01_parent_integrated_rtl_is_million_cycle_bit_exact", lambda x: x["parents"]["t6_7_3"].update(mismatches=1))
    attempt("change_cycle_contract", "G02_six_cycle_and_1p5us_contract_is_live", lambda x: x["contract"].update(source_to_action_cycles=5))
    attempt("forge_source_hash", "G03_all_sources_parents_constraints_and_artifacts_are_hash_bound", lambda x: x["source_bindings"][0].update(sha256="0"))
    attempt("drop_student_profile", "G04_two_real_elaboration_profiles_use_one_integrated_top", lambda x: x["profiles"].pop())
    attempt("erase_bram", "G05_policy_core_and_eight_ab_map_brams_survive_synthesis", lambda x: profile(x, "route_a_core_no_student")["synthesis"]["cell_counts"].update(SDPX9B=0))
    attempt("claim_student_fast_action", "G06_optional_student_is_real_sidecar_and_not_fast_action", lambda x: profile(x, "route_a_plus_student_sidecar").update(student_drives_fast_action=True))
    attempt("drop_route_seed", "G07_six_place_route_runs_pass_target_timing", lambda x: profile(x, "route_a_core_no_student")["place_route"].pop())
    attempt("hide_student_increment", "G08_resources_are_nonzero_fit_and_student_increment_is_observed", lambda x: profile(x, "route_a_plus_student_sidecar")["summary"]["resources_max_across_seeds"]["LUT4"].update(used=1))
    attempt("forge_fmax_summary", "G09_fmax_min_median_max_are_recomputed", lambda x: profile(x, "route_a_core_no_student")["summary"]["fmax_mhz"].update(minimum=999.0))
    attempt("invent_deadline_count", "G10_six_cycle_latency_model_and_positive_deadline_margin_are_exact", lambda x: profile(x, "route_a_core_no_student")["source_to_action_latency_model"].update(deadline_miss_count=0))
    attempt("promote_power_to_vendor", "G11_power_is_sensitivity_estimate_with_null_vendor_and_board_values", lambda x: profile(x, "route_a_core_no_student")["dynamic_power_estimate"].update(vendor_power_mw=1.0))
    attempt("select_unqualified_student", "G12_pareto_selects_nonlearning_fast_path_without_hiding_sidecar", lambda x: x["pareto_decision"].update(selected_profile="route_a_plus_student_sidecar"))
    attempt("forge_csv_count", "G13_source_csv_contains_all_six_runs", lambda x: x["source_data"].update(rows=1))
    attempt("claim_board_speed", "G14_evidence_boundary_forbids_measured_speed_power_or_deadline", lambda x: x["evidence_boundary"].update(board_measured=True, speed_advantage="ESTABLISHED"))
    attempt("forge_mutation_count", "G15_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 15, "detected": 14, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report(build_dir: Path, *, run_tools: bool) -> dict[str, Any]:
    tools = discover_tools()
    timings = run_toolchain(build_dir, tools) if run_tools else {}
    copied = _copy_artifacts(build_dir)
    parent = _load(PARENT_RTL)
    contract = _load(EXECUTION_CONTRACT)
    student_eq = _load(STUDENT_EQUIVALENCE)
    base = _load(BASE_SYNTHESIS)
    deadline_us = float(contract["contract"]["budget"]["fast_action_budget_us_assumption"])
    profiles = [
        _profile_summary(profile, bool(enabled), copied[profile]["synthesis"], copied[profile]["netlist"], copied[profile]["routes"], build_dir, deadline_us)
        for profile, enabled in PROFILES.items()
    ]
    _write_csv(profiles)
    durable = [
        item for profile in copied.values()
        for item in ([profile["synthesis"], profile["netlist"]] + [part for route in profile["routes"] for part in (route["report"], route["log"])])
    ]
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {"device": DEVICE, "family": FAMILY, "top": TOP_MODULE, "target_mhz": TARGET_MHZ, "seeds": list(SEEDS), "profiles": PROFILES},
        "parents": {
            "t6_7_3": {"artifact": _binding(PARENT_RTL), "verdict": parent["verdict"], "cycles": parent["aggregate_python"]["cycles"], "mismatches": sum(row["mismatches"] for row in parent["cxxrtl_families"]), "undefined_actions": parent["aggregate_python"]["undefined_actions"], "silent_overflow": parent["aggregate_python"]["silent_overflow"]},
            "student_equivalence": {"artifact": _binding(STUDENT_EQUIVALENCE), "status": student_eq["status"], "mismatch_count": student_eq["trace"]["mismatch_count"]},
            "base_t5_5_2": {"artifact": _binding(BASE_SYNTHESIS), "verdict": base["verdict"]},
        },
        "contract": {
            "artifact": _binding(EXECUTION_CONTRACT),
            "contract_sha256": contract["contract_sha256"],
            "source_to_action_cycles": contract["contract"]["deadline"]["logical_action_valid_exactly_at_input_plus_cycles"],
            "fast_action_budget_us_assumption": deadline_us,
            "board_deadline_field_must_be_null_before_measurement": contract["contract"]["deadline"]["board_deadline_field_must_be_null_before_measurement"],
        },
        "source_bindings": [_binding(Path(__file__)), *[_binding(path) for path in SOURCES], _binding(CST), _binding(SDC), _binding(PARENT_RTL), _binding(EXECUTION_CONTRACT), _binding(STUDENT_EQUIVALENCE), _binding(BASE_SYNTHESIS)],
        "toolchain": {"yosys": {"path": str(tools["yosys"]), "version": _read_tool_text(Path(tools["yosys"]))[:0] if False else "captured in synthesis logs"}, "nextpnr": {"path": str(tools["nextpnr"]), "version": "captured in route logs"}, "commands": _commands(build_dir, tools)},
        "timing_seconds": timings,
        "profiles": profiles,
        "pareto_decision": {"selected_profile": "route_a_core_no_student", "student_profile_role": "optional_ablation_sidecar_not_primary_action_path", "selection_reason": "same six-cycle fast action with lower resources; CNN/student lacks matched primary evidence", "student_profile_reported": True},
        "evidence_boundary": {"cxxrtl_qualified": True, "open_source_synthesis": True, "open_source_place_route": True, "vendor_timing_signoff": False, "bitstream_generated": False, "transport_implemented": False, "board_measured": False, "board_deadline_miss": None, "measured_source_to_action_ns": None, "measured_power_mw": None, "speed_advantage": "PROHIBITED_PENDING_T6.9.2"},
        "durable_artifacts": durable,
        "source_data": {**_binding(SOURCE_CSV), "rows": 6},
    }
    report["semantic_mutation_audit"] = {"count": 15, "detected": 15, "cases": []}
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": sum(not value for value in report["gates"].values())}
    report["verdict"] = "PASS_ROUTE_A_INTEGRATED_THREE_SEED_PR_ESTIMATE_NOT_BOARD_MEASURED" if report["gate_summary"]["failed"] == 0 else "FAIL_ROUTE_A_HARDWARE_PARETO"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    summary = {"passed": sum(gates.values()), "failed": sum(not value for value in gates.values())}
    if report.get("gates") != gates or report.get("gate_summary") != summary or summary["failed"] != 0 or report.get("verdict") != "PASS_ROUTE_A_INTEGRATED_THREE_SEED_PR_ESTIMATE_NOT_BOARD_MEASURED":
        raise ValueError("T6.9.1 hardware Pareto gates/verdict do not pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or verify T6.9.1 integrated Route-A P&R/Pareto")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--reuse-build", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify_report(_load(args.verify))
        print(f"verified {args.verify}")
        return
    report = build_report(args.build_dir, run_tools=not args.reuse_build)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_report(report)
    print(json.dumps({"output": _relative(args.output), "verdict": report["verdict"], "gate_summary": report["gate_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
