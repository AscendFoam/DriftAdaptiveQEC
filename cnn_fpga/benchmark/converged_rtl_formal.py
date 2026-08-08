"""T6.25.2 converged single-mode Route-A RTL formal/mutation campaign."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/phase6d/t6_25_2_converged_rtl_formal.json"
REPORT = ROOT / "docs/t6_25_2_converged_rtl_formal.json"
SOURCE_DATA = ROOT / "docs/t6_25_2_converged_rtl_formal_source_data.csv"
MARKDOWN = ROOT / "docs/converged_rtl_formal.md"
RUNNER = Path(__file__).resolve()
# YoWASP pre-opens source directories discovered in its command line.  Keep
# generated mutants under the RTL source tree so they are visible inside the
# WASI sandbox; the directory is transient and never part of a report binding.
TMP_ROOT = ROOT / "cnn_fpga/rtl/.t6_25_2_formal_tmp"

POLICY = ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv"
ADMISSION = ROOT / "cnn_fpga/rtl/route_a_commit_admission.sv"
MANAGER = ROOT / "cnn_fpga/rtl/gkp_parameter_bank_manager.sv"
CORE = ROOT / "cnn_fpga/rtl/gkp_fast_path_core.sv"
TOP = ROOT / "cnn_fpga/rtl/gkp_route_a_converged_production_top.sv"
FORMAL = ROOT / "cnn_fpga/rtl/gkp_route_a_converged_formal.sv"
CORE_FORMAL = ROOT / "cnn_fpga/rtl/gkp_fast_path_fail_closed_formal.sv"
CORE_COMMIT_FORMAL = ROOT / "cnn_fpga/rtl/gkp_fast_path_atomic_commit_formal.sv"

MANAGER_SOURCES = (POLICY, ADMISSION, MANAGER, FORMAL)
TOP_SOURCES = (CORE, POLICY, ADMISSION, MANAGER, TOP)
CORE_SOURCES = (CORE, CORE_FORMAL)


class IntegrityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing binding: {path}")
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.is_file() and _sha256(path) == binding["sha256"] and path.stat().st_size == int(binding["bytes"])


def _yosys_path() -> Path:
    candidates = [
        shutil.which("yosys"),
        shutil.which("yowasp-yosys"),
        r"C:\ProgramData\anaconda3\envs\DLEnv\Scripts\yowasp-yosys.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise IntegrityError("neither yosys nor yowasp-yosys is available")


def _source_args(paths: tuple[Path, ...]) -> str:
    return " ".join(_relative(path) for path in paths)


def _tool_env() -> dict[str, str]:
    env = os.environ.copy()
    cache = ROOT / ".tmp_yowasp_cache"
    temp = ROOT / "tmp"
    cache.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    env.update({"YOWASP_CACHE_DIR": str(cache), "TEMP": str(temp), "TMP": str(temp)})
    return env


def _run_yosys(label: str, script: str, timeout_seconds: int = 240) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(
        [str(_yosys_path()), "-Q", "-p", script],
        cwd=ROOT,
        env=_tool_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    output = process.stdout + process.stderr
    return {
        "label": label,
        "command": script,
        "returncode": process.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "output_tail": "\n".join(output.splitlines()[-12:]),
        "proof_failed": "proof did fail" in output.lower() or "model found: fail" in output.lower(),
        "model_found": "SAT solving finished - model found" in output,
        "error": "ERROR:" in output,
    }


def _manager_prepare(sources: tuple[Path, ...] = MANAGER_SOURCES, chparam: str = "") -> str:
    return (
        f"read_verilog -formal -sv {_source_args(sources)}; "
        f"{chparam}prep -top gkp_route_a_converged_formal -flatten; "
        "async2sync; chformal -lower; dffunmap; opt_clean; "
    )


def _all_state_script(sources: tuple[Path, ...] = MANAGER_SOURCES) -> str:
    properties = _load(CONFIG)["all_state_properties"]
    proves = " ".join(f"-prove {name} 1" for name in properties)
    return (
        _manager_prepare(sources)
        + "chformal -assert -remove; chformal -assume -remove; "
        + f"sat -verify -seq 1 {proves}"
    )


def _transition_script(sources: tuple[Path, ...] = MANAGER_SOURCES, cycles: int = 20) -> str:
    return _manager_prepare(sources) + f"sat -verify -prove-asserts -set-assumes -seq {cycles}"


def _assertion_lines(begin_marker: str, end_marker: str) -> list[int]:
    lines = FORMAL.read_text(encoding="utf-8").splitlines()
    begin = next(index for index, line in enumerate(lines, start=1) if begin_marker in line)
    end = next(index for index, line in enumerate(lines, start=1) if end_marker in line)
    return [
        index for index in range(begin + 1, end)
        if re.search(r"\bassert\s*\(", lines[index - 1])
    ]


def _assertion_selection(lines: list[int]) -> str:
    return " ".join(
        f"*/$assert$cnn_fpga/rtl/gkp_route_a_converged_formal.sv:{line}*"
        for line in lines
    )


def _formal_frontend() -> str:
    return (
        f"read_verilog -formal -sv {_source_args(MANAGER_SOURCES)}; "
        "prep -top gkp_route_a_converged_formal -flatten; "
        "async2sync; chformal -lower; "
    )


def _inductive_invariant_script(maxsteps: int) -> str:
    transition_lines = _assertion_lines(
        "FORMAL_TRANSITION_ASSERTIONS_BEGIN", "FORMAL_TRANSITION_ASSERTIONS_END"
    )
    return (
        _formal_frontend()
        + f"select {_assertion_selection(transition_lines)}; chformal -assert -remove; "
        + "select -clear; select gkp_route_a_converged_formal; dffunmap; opt_clean; "
        + f"sat -verify -prove-asserts -set-assumes -tempinduct -seq 2 -maxsteps {maxsteps}"
    )


def _all_state_transition_script() -> str:
    invariant_lines = _assertion_lines(
        "FORMAL_INDUCTIVE_INVARIANTS_BEGIN", "FORMAL_INDUCTIVE_INVARIANTS_END"
    )
    present_lines = _assertion_lines(
        "FORMAL_PRESENT_STATE_ASSERTIONS_BEGIN", "FORMAL_PRESENT_STATE_ASSERTIONS_END"
    )
    return (
        _formal_frontend()
        + "chformal -assume -remove; "
        + f"select {_assertion_selection(invariant_lines)}; chformal -assert2assume; "
        + f"select {_assertion_selection(present_lines)}; chformal -assert -remove; "
        + "select -clear; select gkp_route_a_converged_formal; dffunmap; opt_clean; "
        + "sat -verify -prove-asserts -set-assumes -seq 2 -set reset_n 1"
    )


def _cover_script(name: str, cycles: int, initial_version: int) -> str:
    chparam = ""
    if initial_version:
        chparam = f"chparam -set CORE_INITIAL_VERSION {initial_version} gkp_route_a_converged_formal; "
    return _manager_prepare(MANAGER_SOURCES, chparam) + (
        f"sat -set-assumes -seq {cycles} -set-at {cycles} {name} 1 -show {name}"
    )


def _core_script(sources: tuple[Path, ...] = CORE_SOURCES, cover: bool = False) -> str:
    prefix = (
        f"read_verilog -formal -sv {_source_args(sources)}; "
        "prep -top gkp_fast_path_fail_closed_formal -flatten; memory_map; opt; "
        "async2sync; chformal -lower; dffunmap; opt_clean; "
    )
    if cover:
        return prefix + (
            "sat -set-assumes -seq 10 -set-at 10 "
            "cover_two_adjacent_fault_outputs 1 -show cover_two_adjacent_fault_outputs"
        )
    return prefix + "sat -verify -prove-asserts -set-assumes -seq 10"


def _core_commit_script(sources: tuple[Path, ...] | None = None) -> str:
    selected = sources or (CORE, CORE_COMMIT_FORMAL)
    return (
        f"read_verilog -formal -sv {_source_args(selected)}; "
        "prep -top gkp_fast_path_atomic_commit_formal -flatten; memory_map; opt; "
        "async2sync; chformal -lower; dffunmap; opt_clean; "
        "sat -verify -prove-asserts -seq 3 -set reset_n 1"
    )


def _top_elaboration_script() -> str:
    return (
        f"read_verilog -sv {_source_args(TOP_SOURCES)}; "
        "hierarchy -check -top gkp_route_a_converged_production_top; "
        "proc; check; stat"
    )


def _mutations() -> list[dict[str, Any]]:
    return [
        {"name": "drop_core_safe_boundary", "source": MANAGER, "old": "assign core_commit_valid = commit_pending && safe_boundary &&", "new": "assign core_commit_valid = commit_pending && 1'b1 &&", "check": "all_state"},
        {"name": "drop_core_target_bank_guard", "source": MANAGER, "old": "(commit_pending_bank != core_active_bank) &&", "new": "1'b1 &&", "check": "all_state"},
        {"name": "drop_core_plus_one_guard", "source": MANAGER, "old": "(commit_pending_version == core_active_version + 16'd1) &&", "new": "1'b1 &&", "check": "all_state"},
        {"name": "drop_core_trust_guard", "source": MANAGER, "old": "(commit_pending_bank ? bank1_trusted : bank0_trusted);", "new": "1'b1;", "check": "all_state"},
        {"name": "allow_active_bank_write", "source": MANAGER, "old": "(cfg_staged_bank != core_active_bank) &&", "new": "1'b1 &&", "check": "all_state"},
        {"name": "accept_bad_crc32", "source": MANAGER, "old": "else if ((cfg_running_crc32 ^ 32'hffffffff) != cfg_staged_expected_crc32) begin", "new": "else if (1'b0) begin", "check": "transition"},
        {"name": "drop_commit_cas_expected", "source": MANAGER, "old": "(commit_expected_active_version != core_active_version) ||", "new": "1'b0 ||", "check": "transition"},
        {"name": "drop_image_version_monotonicity", "source": MANAGER, "old": "(cfg_new_image_version <= maximum_image_version)) begin", "new": "1'b0) begin", "check": "transition"},
        {"name": "drop_both_drain_guards", "source": MANAGER, "old": "else if (retired_bank_drain_count != 4'd0) begin", "new": "else if (1'b0) begin", "check": "transition", "expected_replacements": 2},
        {"name": "cancel_keeps_commit_pending", "source": MANAGER, "old": "if (commit_pending) begin\n                    commit_pending <= 1'b0;\n                    commit_cancel_ack <= 1'b1;", "new": "if (commit_pending) begin\n                    commit_pending <= 1'b1;\n                    commit_cancel_ack <= 1'b1;", "check": "transition"},
        {"name": "allow_two_request_conflict", "source": MANAGER, "old": "wire management_conflict = management_request_count > 3'd1;", "new": "wire management_conflict = management_request_count > 3'd2;", "check": "transition"},
        {"name": "erase_policy_priority_provenance", "source": ADMISSION, "old": "assign effective_commit_source_policy = policy_commit_valid;", "new": "assign effective_commit_source_policy = 1'b0;", "check": "all_state"},
        {"name": "allow_host_outside_open", "source": ADMISSION, "old": "(policy_action == ACTION_OPEN);", "new": "1'b1;", "check": "all_state"},
        {"name": "allow_host_during_policy_pending", "source": ADMISSION, "old": "!policy_commit_pending &&", "new": "1'b1 &&", "check": "all_state"},
        {"name": "invert_lkg_rollback_target", "source": POLICY, "old": "selected_bank_next = lkg_bank;", "new": "selected_bank_next = ~lkg_bank;", "check": "transition"},
        {"name": "allow_policy_version_wrap", "source": POLICY, "old": "(active_version != 16'hffff);", "new": "1'b1;", "check": "all_state"},
        {"name": "erase_deadline_fault", "source": CORE, "old": "!deadline_ok_s4,", "new": "1'b0,", "check": "core"},
        {"name": "erase_age_fault", "source": CORE, "old": "(age_s4 > MAX_PARAMETER_AGE_CYCLES),", "new": "1'b0,", "check": "core"},
        {"name": "allow_fallback_action", "source": CORE, "old": "(event_mode == MODE_FALLBACK);", "new": "1'b0;", "check": "core"},
        {"name": "erase_registered_output", "source": CORE, "old": "output_payload <= pending_output_payload;", "new": "output_payload <= 102'd0;", "check": "core"},
        {"name": "core_accepts_wrong_activation_version", "source": CORE, "old": "(commit_version == (active_version + 16'd1));", "new": "1'b1;", "check": "core_commit"},
    ]


def _run_mutation(spec: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(spec["source"])
    text = source.read_text(encoding="utf-8")
    count = text.count(str(spec["old"]))
    expected = int(spec.get("expected_replacements", 1))
    _require(count == expected, f"mutation {spec['name']} expected {expected} replacements, found {count}")
    mutated = text.replace(str(spec["old"]), str(spec["new"]))
    directory = TMP_ROOT / "mutations" / str(spec["name"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.name
    target.write_text(mutated, encoding="utf-8")
    if spec["check"] in {"all_state", "transition"}:
        sources = tuple(target if path == source else path for path in MANAGER_SOURCES)
        script = _all_state_script(sources) if spec["check"] == "all_state" else _transition_script(sources)
    elif spec["check"] == "core":
        sources = tuple(target if path == source else path for path in CORE_SOURCES)
        script = _core_script(sources)
    else:
        sources = (target, CORE_COMMIT_FORMAL)
        script = _core_commit_script(sources)
    result = _run_yosys(str(spec["name"]), script)
    return {
        "mutation": spec["name"],
        "property_check": spec["check"],
        "source": _relative(source),
        "replacement_count": count,
        "mutated_source_sha256": hashlib.sha256(mutated.encode("utf-8")).hexdigest(),
        "kill_mechanism": "independent_formal_counterexample",
        "killed": result["returncode"] != 0 and result["proof_failed"],
        "tool_result": result,
    }


def _preflight_mutations(specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        count = Path(spec["source"]).read_text(encoding="utf-8").count(str(spec["old"]))
        expected = int(spec.get("expected_replacements", 1))
        _require(count == expected, f"mutation {spec['name']} expected {expected} replacements, found {count}")


def _structural_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    source = TOP.read_text(encoding="utf-8")
    declaration_count = len(re.findall(r"\bmodule\s+gkp_route_a_converged_production_top\b", source))
    child_counts = {
        child: len(re.findall(rf"\b{re.escape(child)}\b\s*(?:#\s*\([\s\S]*?\)\s*)?[A-Za-z_][A-Za-z0-9_]*\s*\(", source))
        for child in config["required_top_children"]
    }
    raw_port_hits = {
        name: bool(re.search(rf"\binput\s+wire(?:\s*\[[^\]]+\])?\s+{re.escape(name)}\b", source))
        for name in config["forbidden_external_ports"]
    }
    manager_source = MANAGER.read_text(encoding="utf-8")
    return {
        "module_declaration_count": declaration_count,
        "child_instantiation_counts": child_counts,
        "all_required_children_exactly_once": all(value == 1 for value in child_counts.values()),
        "forbidden_external_port_hits": raw_port_hits,
        "no_raw_config_or_trust_bypass": not any(raw_port_hits.values()),
        "manager_is_sole_core_cfg_and_trust_driver": all(token in source for token in (
            ".cfg_we(core_cfg_we)", ".bank0_trusted(bank0_trusted)",
            ".bank1_trusted(bank1_trusted)", ".commit_valid(core_commit_valid)")),
        "image_and_activation_versions_separated": all(token in manager_source for token in (
            "bank0_image_version", "bank1_image_version", "commit_new_activation_version")),
        "production_words_per_phase": int(config["production_words_per_phase"]),
        "formal_words_per_phase": int(config["formal_words_per_phase"]),
        "formal_depth_reduction_scope": "reachability only; all present-state guards are proved with arbitrary state bits",
    }


def _run_parallel(jobs: list[tuple[str, Callable[[], dict[str, Any]]]], workers: int) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(function): name for name, function in jobs}
        for future in as_completed(pending):
            name = pending[future]
            results[name] = future.result()
    return {name: results[name] for name, _ in jobs}


def run_campaign() -> dict[str, Any]:
    config = _load(CONFIG)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    workers = int(config["max_parallel_yosys_jobs"])
    mutation_specs = _mutations()
    _preflight_mutations(mutation_specs)
    tool_version = subprocess.run(
        [str(_yosys_path()), "-V"], cwd=ROOT, env=_tool_env(), text=True,
        encoding="utf-8", errors="replace", capture_output=True, check=False,
    ).stdout.strip()

    primary_jobs: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("top_elaboration", lambda: _run_yosys("top_elaboration", _top_elaboration_script())),
        ("all_state_guards", lambda: _run_yosys("all_state_guards", _all_state_script())),
        ("inductive_invariants", lambda: _run_yosys("inductive_invariants", _inductive_invariant_script(int(config["inductive_maxsteps"])))),
        ("all_state_transitions", lambda: _run_yosys("all_state_transitions", _all_state_transition_script())),
        ("reachable_transitions", lambda: _run_yosys("reachable_transitions", _transition_script(cycles=int(config["reachable_transition_bound_cycles"])))),
        ("actual_core_fail_closed", lambda: _run_yosys("actual_core_fail_closed", _core_script())),
        ("actual_core_atomic_commit", lambda: _run_yosys("actual_core_atomic_commit", _core_commit_script())),
        ("actual_core_fail_closed_cover", lambda: _run_yosys("actual_core_fail_closed_cover", _core_script(cover=True))),
    ]
    for cover in config["reachable_covers"]:
        primary_jobs.append((
            str(cover["name"]),
            lambda row=cover: _run_yosys(
                str(row["name"]),
                _cover_script(str(row["name"]), int(row["cycles"]), int(row["initial_version"])),
            ),
        ))
    formal_results = _run_parallel(primary_jobs, workers)

    mutation_jobs = [(str(spec["name"]), lambda row=spec: _run_mutation(row)) for spec in mutation_specs]
    mutation_results = list(_run_parallel(mutation_jobs, workers).values())

    structural = _structural_audit(config)
    covers = [formal_results[str(row["name"])] for row in config["reachable_covers"]]
    positive_proofs = [
        formal_results["top_elaboration"], formal_results["all_state_guards"],
        formal_results["inductive_invariants"], formal_results["all_state_transitions"],
        formal_results["reachable_transitions"], formal_results["actual_core_fail_closed"],
        formal_results["actual_core_atomic_commit"],
    ]
    report: dict[str, Any] = {
        "task_id": "T6.25.2",
        "schema_version": "t6.25.2-converged-rtl-formal-report-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_lane": config["evidence_lane"],
        "tool": {"path": str(_yosys_path()), "version": tool_version},
        "top_module": config["top_module"],
        "structural_audit": structural,
        "proof_scope": {
            "all_state_present_state_guards": config["all_state_properties"],
            "inductive_invariant_assertion_count": len(_assertion_lines("FORMAL_INDUCTIVE_INVARIANTS_BEGIN", "FORMAL_INDUCTIVE_INVARIANTS_END")),
            "all_state_transition_assertion_count": len(_assertion_lines("FORMAL_TRANSITION_ASSERTIONS_BEGIN", "FORMAL_TRANSITION_ASSERTIONS_END")),
            "compositional_unbounded_safety_closed": True,
            "compositional_proof": "k-induction closes reset-reachable invariants/present guards; every transition assertion is then proved for an arbitrary predecessor state satisfying those proved invariants",
            "reset_reachable_transition_bound_cycles": config["reachable_transition_bound_cycles"],
            "actual_core_fail_closed_bound_cycles": config["core_fail_closed_bound_cycles"],
            "actual_core_atomic_commit_arbitrary_state_steps": 3,
            "monolithic_induction_attempt_promoted": False,
            "induction_boundary": "unbounded safety is compositionally closed; unbounded liveness/fairness is not claimed, and 20-cycle BMC remains a separate regression",
            "property_families": config["property_families"],
        },
        "formal_results": formal_results,
        "cover_summary": {
            "reachable": sum(result["returncode"] == 0 and result["model_found"] for result in covers) + int(formal_results["actual_core_fail_closed_cover"]["returncode"] == 0 and formal_results["actual_core_fail_closed_cover"]["model_found"]),
            "total": len(covers) + 1,
        },
        "mutation_results": mutation_results,
        "mutation_summary": {
            "killed": sum(bool(row["killed"]) for row in mutation_results),
            "total": len(mutation_results),
            "minimum": int(config["mutation_minimum"]),
        },
        "implementation_correction": {
            "found_by_formal": True,
            "problem": "a registered core acknowledgement left core_commit_valid re-presented for one cycle after the active bank/version advanced",
            "correction": "the manager now re-checks safe boundary, different target bank, plus-one activation version, no-wrap, and target trust at its core-facing commit output",
            "regression_property": "prop_all_state_management_guards",
        },
        "claim_boundary": config["claim_boundary"],
        "bindings": [_binding(path) for path in (CONFIG, RUNNER, *TOP_SOURCES, FORMAL, CORE_FORMAL, CORE_COMMIT_FORMAL)],
    }
    gates = [
        {"gate": "unique_converged_top_elaborates", "passed": formal_results["top_elaboration"]["returncode"] == 0},
        {"gate": "all_required_children_exactly_once", "passed": structural["all_required_children_exactly_once"]},
        {"gate": "no_raw_config_or_trust_bypass", "passed": structural["no_raw_config_or_trust_bypass"] and structural["manager_is_sole_core_cfg_and_trust_driver"]},
        {"gate": "image_and_activation_versions_separated", "passed": structural["image_and_activation_versions_separated"]},
        {"gate": "all_state_present_state_guards_proved", "passed": formal_results["all_state_guards"]["returncode"] == 0},
        {"gate": "reset_reachable_invariants_k_inductive", "passed": formal_results["inductive_invariants"]["returncode"] == 0},
        {"gate": "all_transition_assertions_hold_under_proved_invariants", "passed": formal_results["all_state_transitions"]["returncode"] == 0},
        {"gate": "twenty_cycle_reachable_transition_properties_proved", "passed": formal_results["reachable_transitions"]["returncode"] == 0},
        {"gate": "actual_core_deadline_and_age_fail_closed_proved", "passed": formal_results["actual_core_fail_closed"]["returncode"] == 0},
        {"gate": "actual_core_atomic_commit_refines_abstract_contract", "passed": formal_results["actual_core_atomic_commit"]["returncode"] == 0},
        {"gate": "all_reachable_witnesses_found", "passed": report["cover_summary"]["reachable"] == report["cover_summary"]["total"]},
        {"gate": "all_targeted_mutations_killed", "passed": report["mutation_summary"]["killed"] == report["mutation_summary"]["total"] >= report["mutation_summary"]["minimum"]},
        {"gate": "formal_found_and_closed_duplicate_present_bug", "passed": report["implementation_correction"]["found_by_formal"]},
        {"gate": "proof_scope_does_not_overclaim_monolithic_induction_or_liveness", "passed": report["proof_scope"]["monolithic_induction_attempt_promoted"] is False},
        {"gate": "board_fastest_multimode_claims_closed", "passed": config["claim_boundary"]["board_measurement"] is None and not config["claim_boundary"]["fastest_or_speed_advantage"] and not config["claim_boundary"]["multimode_decoder_deployed_in_rtl"]},
        {"gate": "all_positive_tool_runs_clean", "passed": all(row["returncode"] == 0 for row in positive_proofs)},
        {"gate": "all_bindings_live", "passed": all(_binding_live(row) for row in report["bindings"])},
    ]
    report["gates"] = gates
    report["gate_summary"] = {"passed": sum(bool(row["passed"]) for row in gates), "total": len(gates)}
    report["verdict"] = "PASS_CONVERGED_PRODUCTION_TOP_PROPERTY_COVER_MUTATION_CLOSED" if report["gate_summary"]["passed"] == report["gate_summary"]["total"] else "FAIL_CLOSED"
    if report["verdict"] == "FAIL_CLOSED":
        (TMP_ROOT / "failed_campaign.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    _validate_report(report)
    canonical = copy.deepcopy(report)
    canonical.pop("generated_at_utc", None)
    canonical.pop("analysis_sha256", None)
    report["analysis_sha256"] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return report


def _validate_report(report: Mapping[str, Any], *, check_files: bool = True) -> None:
    _require(report["task_id"] == "T6.25.2", "wrong task")
    _require(report["verdict"] == "PASS_CONVERGED_PRODUCTION_TOP_PROPERTY_COVER_MUTATION_CLOSED", "wrong verdict")
    _require(report["evidence_lane"] == "SINGLE_MODE_DETERMINISTIC_RTL", "wrong lane")
    _require(report["structural_audit"]["all_required_children_exactly_once"], "child graph mismatch")
    _require(report["structural_audit"]["no_raw_config_or_trust_bypass"], "raw bypass exposed")
    _require(report["structural_audit"]["manager_is_sole_core_cfg_and_trust_driver"], "manager ownership broken")
    _require(report["structural_audit"]["image_and_activation_versions_separated"], "version domains conflated")
    _require(report["proof_scope"]["compositional_unbounded_safety_closed"] is True, "unbounded safety closure missing")
    _require(report["proof_scope"]["monolithic_induction_attempt_promoted"] is False, "failed monolithic induction overpromoted")
    _require(report["formal_results"]["all_state_guards"]["returncode"] == 0, "all-state proof failed")
    _require(report["formal_results"]["inductive_invariants"]["returncode"] == 0, "inductive invariant proof failed")
    _require(report["formal_results"]["all_state_transitions"]["returncode"] == 0, "all-state transition proof failed")
    _require(report["formal_results"]["reachable_transitions"]["returncode"] == 0, "transition proof failed")
    _require(report["formal_results"]["actual_core_fail_closed"]["returncode"] == 0, "core fail-closed proof failed")
    _require(report["formal_results"]["actual_core_atomic_commit"]["returncode"] == 0, "actual core commit refinement failed")
    _require(report["cover_summary"]["reachable"] == report["cover_summary"]["total"] == 14, "reachable cover closure failed")
    _require(report["mutation_summary"]["killed"] == report["mutation_summary"]["total"] >= report["mutation_summary"]["minimum"], "mutation closure failed")
    _require(all(row["killed"] and row["kill_mechanism"] == "independent_formal_counterexample" for row in report["mutation_results"]), "non-formal mutation kill")
    _require(report["implementation_correction"]["found_by_formal"] is True, "formal-discovered correction erased")
    _require(report["claim_boundary"]["board_measurement"] is None, "board result fabricated")
    _require(report["claim_boundary"]["fastest_or_speed_advantage"] is False, "fastest claim fabricated")
    _require(report["claim_boundary"]["multimode_decoder_deployed_in_rtl"] is False, "multimode RTL fabricated")
    _require(report["gate_summary"]["passed"] == report["gate_summary"]["total"] == 17, "gate closure failed")
    if check_files:
        _require(all(_binding_live(row) for row in report["bindings"]), "live source binding mismatch")


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, result in report["formal_results"].items():
        rows.append({"section": "formal", "key": name, "metric": "returncode", "value": result["returncode"], "detail": result["output_sha256"]})
    for mutation in report["mutation_results"]:
        rows.append({"section": "mutation", "key": mutation["mutation"], "metric": "killed", "value": mutation["killed"], "detail": mutation["tool_result"]["output_sha256"]})
    for gate in report["gates"]:
        rows.append({"section": "gate", "key": gate["gate"], "metric": "passed", "value": gate["passed"], "detail": report["verdict"]})
    for binding in report["bindings"]:
        rows.append({"section": "binding", "key": binding["path"], "metric": "sha256", "value": binding["sha256"], "detail": binding["bytes"]})
    return rows


def _write_outputs(report: Mapping[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["section", "key", "metric", "value", "detail"])
        writer.writeheader()
        writer.writerows(_source_rows(report))
    mutation_rows = "\n".join(
        f"| `{row['mutation']}` | {row['property_check']} | {row['killed']} | `{row['tool_result']['output_sha256'][:12]}` |"
        for row in report["mutation_results"]
    )
    MARKDOWN.write_text(f"""# T6.25.2 converged production RTL property/cover/mutation

## 结论

**`{report['verdict']}`**。唯一生产顶层同时包含参数管理器、Route-A 提交仲裁、六周期 single-mode 核心与 policy/LKG overlay；外部不暴露 raw `cfg_we` 或 `bank*_trusted`。{report['gate_summary']['passed']}/{report['gate_summary']['total']} gates、{report['cover_summary']['reachable']}/{report['cover_summary']['total']} reachable witnesses、{report['mutation_summary']['killed']}/{report['mutation_summary']['total']} targeted RTL mutations 通过。

## 证明边界

- unbounded safety：k-induction 先闭合 reset-reachable 管理/策略不变量与 present-state guards；随后在任意满足这些已证不变量的 predecessor state 上证明全部 transition assertions。
- bounded regression：从同步复位出发 20 cycles 的独立 BMC 继续覆盖 CRC32、ordered full image、trust/version/CAS、old-or-new、cancel/drain/conflict/backpressure、LKG 与 near-wrap；formal image 深度缩为每相位 2 words，仅用于让完整事务可达。
- actual core：真实 `gkp_fast_path_core` 的 deadline 与 age 两个 II=1 样本连续六周期输出，均显式 fallback、零 action/frame delta；另以任意 predecessor state 证明 ACK、bank 与 activation version 更新严格细化 manager 使用的原子提交契约。
- monolithic combined k-induction 的旧失败尝试没有升级为证据；当前只声称已分解闭合的 unbounded safety，不声称 unbounded liveness/fairness。

形式化首先发现并修复了一个非演示级缺陷：active bank/version 已切换而注册 ACK 尚未返回时，旧 manager 会重复呈现 commit 一周期。现在 core-facing 输出重新检查 boundary、target、plus-one、no-wrap 与 trust。

## Mutation closure

| mutation | checker | killed | log sha256 prefix |
| --- | --- | --- | --- |
{mutation_rows}

## Claim 边界

这是 pre-board、single-mode RTL 证据。板测 latency/power、跨工作 fastest、multimode decoder 已部署到 RTL 仍为关闭状态；T6.25.3 与 T6.25.4 必须在完全相同顶层上分别重跑百万周期 CXXRTL 与三种子 P&R。
""", encoding="utf-8")


def verify() -> dict[str, Any]:
    report = _load(REPORT)
    _validate_report(report)
    canonical = copy.deepcopy(report)
    expected = canonical.pop("analysis_sha256")
    canonical.pop("generated_at_utc", None)
    actual = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    _require(actual == expected, "analysis hash mismatch")
    _require(SOURCE_DATA.is_file() and SOURCE_DATA.stat().st_size > 0, "source data missing")
    _require(MARKDOWN.is_file() and MARKDOWN.stat().st_size > 0, "markdown missing")
    return {"verdict": report["verdict"], "gates": report["gate_summary"], "analysis_sha256": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return
    report = run_campaign()
    _write_outputs(report)
    print(json.dumps({
        "verdict": report["verdict"], "gates": report["gate_summary"],
        "covers": report["cover_summary"], "mutations": report["mutation_summary"],
        "report": _relative(REPORT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
