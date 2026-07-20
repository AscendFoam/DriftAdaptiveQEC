"""Fail-closed prerequisite gate for the T6.9.2 physical-board experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.9.2"
SCHEMA_VERSION = "t6.9.2-board-measurement-blocker-v2"
BOARD = ROOT / "docs/new_task_board.md"
PARETO = ROOT / "docs/t6_9_1_route_a_hardware_pareto.json"
NORMALIZATION = ROOT / "docs/t6_8_6_fpga_decoder_normalization.json"
DEFAULT_ARTIFACT = ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json"

EXPECTED_EXTERNAL_ARTIFACTS = {
    "actual_board_inventory": "configs/hardware/t6_1_1_actual_board.json",
    "transport_adapter_qualification": "docs/t6_1_2_transport_adapter_qualification.json",
    "timestamp_measurement_method": "docs/t6_1_3_board_timestamp_method.json",
    "board_correctness_smoke": "docs/t6_2_3_board_correctness_smoke.json",
    "board_hil_qualification": "docs/t6_4_route_a_board_hil_qualification.json",
    "bitstream_manifest": "docs/t6_9_2_bitstream_manifest.json",
}

MEASURED_FIELDS = (
    "bitstream_sha256", "source_sha256", "board_serial", "board_clock_mhz",
    "cycles", "bit_mismatch_count", "undefined_action_count", "silent_overflow_count",
    "deadline_miss_count", "zero_event_95pct_upper_bound",
    "core_latency_p50_ns", "core_latency_p95_ns", "core_latency_p99_ns", "core_latency_worst_ns",
    "transport_latency_p50_ns", "transport_latency_p95_ns", "transport_latency_p99_ns", "transport_latency_worst_ns",
    "source_to_action_p50_ns", "source_to_action_p95_ns", "source_to_action_p99_ns", "source_to_action_worst_ns",
    "end_to_end_p50_ns", "end_to_end_p95_ns", "end_to_end_p99_ns", "end_to_end_worst_ns",
    "initiation_interval_ns", "fmax_mhz", "jitter_p95_ns", "jitter_p99_ns", "jitter_worst_ns",
    "lut", "ff", "bram", "dsp", "power_idle_mw", "power_dynamic_mw", "power_total_mw",
    "same_task_external_comparator_count", "speed_advantage_effect_ns", "speed_advantage_ci95_low_ns", "speed_advantage_ci95_high_ns",
)

BOARD_STATUS_TASKS = (
    "T6.1.1", "T6.1.2", "T6.1.3", "T6.2.3", "T6.4.1", "T6.4.2",
    "T6.4.3", "T6.9.1", "T6.9.2",
)
EXPECTED_BOARD_STATUSES = {
    "T6.1.1": "Blocked", "T6.1.2": "Todo", "T6.1.3": "Todo",
    "T6.2.3": "Todo", "T6.4.1": "Todo", "T6.4.2": "Todo",
    "T6.4.3": "Todo", "T6.9.1": "Done", "T6.9.2": "Blocked",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.is_file() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _statuses(text: str) -> dict[str, str]:
    rows = re.findall(r"^\| (T[^| ]+) \| ([^|]+) \|", text, flags=re.MULTILINE)
    result: dict[str, str] = {}
    for task, status in rows:
        result.setdefault(task.strip(), status.strip())
    return result


def _board_status_binding(text: str) -> dict[str, Any]:
    statuses = _statuses(text)
    selected = {task: statuses[task] for task in BOARD_STATUS_TASKS}
    return {
        "path": _relative(BOARD),
        "task_ids": list(BOARD_STATUS_TASKS),
        "statuses": selected,
        "canonical_sha256": _canonical_sha256(selected),
    }


def _board_status_binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    if not path.is_file() or binding.get("task_ids") != list(BOARD_STATUS_TASKS):
        return False
    try:
        live = _board_status_binding(path.read_text(encoding="utf-8"))
    except (KeyError, OSError, UnicodeError):
        return False
    return (
        binding.get("statuses") == live["statuses"]
        and binding.get("canonical_sha256") == live["canonical_sha256"]
    )


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    prereqs = report["prerequisite_ledger"]
    measured = report["measured_results"]
    bindings = report["bindings"]
    failed_external = [row for row in prereqs if row["kind"] == "physical_external" and not row["passed"]]
    return {
        "G01_preboard_pareto_parent_passes_and_is_live": report["parent_t6_9_1"]["verdict"] == "PASS_ROUTE_A_INTEGRATED_THREE_SEED_PR_ESTIMATE_NOT_BOARD_MEASURED" and (not check_live_files or _live(bindings["pareto"])),
        "G02_board_and_hardware_upstream_tasks_are_not_done": report["board_statuses"] == EXPECTED_BOARD_STATUSES and report["board_status_binding"]["statuses"] == report["board_statuses"] and (not check_live_files or _board_status_binding_live(report["board_status_binding"])),
        "G03_all_six_physical_prerequisites_are_absent": len(failed_external) == len(EXPECTED_EXTERNAL_ARTIFACTS) == 6 and all(row["expected_path"] == EXPECTED_EXTERNAL_ARTIFACTS[row["prerequisite"]] and row["observed_path"] is None for row in failed_external),
        "G04_execution_branch_is_blocked_before_board_run": report["execution_branch"] == "BLOCKED_NO_PHYSICAL_BOARD_BITSTREAM_OR_TRANSPORT" and report["measurement_run_manifest"] is None and report["measurement_raw_data"] is None,
        "G05_every_measured_field_is_explicit_null": set(measured) == set(MEASURED_FIELDS) and all(value is None for value in measured.values()),
        "G06_no_preboard_value_is_copied_into_measured_fields": report["non_substitution"]["pr_clock_model_ns"] == 222.22222222222223 and report["non_substitution"]["copied_to_measured_source_to_action"] is False and measured["source_to_action_p50_ns"] is None,
        "G07_fpga_speed_claim_remains_prohibited": report["claim_boundary"] == {"board_correctness": "NOT_RUN_BLOCKED", "zero_deadline_miss": "NOT_ESTABLISHED", "measured_source_to_action": "UNDEFINED", "measured_power": "UNDEFINED", "fpga_speed_advantage": "PROHIBITED", "fastest_or_sota": "PROHIBITED"},
        "G08_recovery_requires_all_physical_evidence_not_one_file": len(report["recovery_conditions"]) == 9 and all(condition["required"] is True for condition in report["recovery_conditions"]),
        "G09_board_normalization_parent_is_live_and_same_task_count_zero": report["normalization_anchor"]["same_task_external_comparator_count"] == 0 and (not check_live_files or _live(bindings["normalization"])),
        "G10_all_current_bindings_are_live": set(bindings) == {"implementation", "pareto", "normalization"} and all(len(row["sha256"]) == 64 for row in bindings.values()) and (not check_live_files or all(_live(row) for row in bindings.values())),
        "G11_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 11,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 11, "detected": 11, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("promote_pareto_parent", "G01_preboard_pareto_parent_passes_and_is_live", lambda x: x["parent_t6_9_1"].update(verdict="MEASURED"))
    attempt("mark_board_done", "G02_board_and_hardware_upstream_tasks_are_not_done", lambda x: x["board_statuses"].update({"T6.1.1": "Done"}))
    attempt("invent_board_inventory", "G03_all_six_physical_prerequisites_are_absent", lambda x: x["prerequisite_ledger"][1].update(passed=True, observed_path="fake.json"))
    attempt("invent_measurement_manifest", "G04_execution_branch_is_blocked_before_board_run", lambda x: x.update(measurement_run_manifest={"run": "fake"}))
    attempt("copy_cycle_count", "G05_every_measured_field_is_explicit_null", lambda x: x["measured_results"].update(cycles=1_000_000))
    attempt("copy_pr_latency", "G06_no_preboard_value_is_copied_into_measured_fields", lambda x: x["measured_results"].update(source_to_action_p50_ns=222.22222222222223))
    attempt("claim_speed", "G07_fpga_speed_claim_remains_prohibited", lambda x: x["claim_boundary"].update(fpga_speed_advantage="ESTABLISHED"))
    attempt("drop_recovery_condition", "G08_recovery_requires_all_physical_evidence_not_one_file", lambda x: x["recovery_conditions"].pop())
    attempt("invent_same_task_comparator", "G09_board_normalization_parent_is_live_and_same_task_count_zero", lambda x: x["normalization_anchor"].update(same_task_external_comparator_count=1))
    attempt("forge_binding", "G10_all_current_bindings_are_live", lambda x: x["bindings"]["pareto"].update(sha256="0"))
    attempt("forge_mutation_count", "G11_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 11, "detected": 10, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    board_text = BOARD.read_text(encoding="utf-8")
    statuses = _statuses(board_text)
    pareto = _load(PARETO)
    normalization = _load(NORMALIZATION)
    prereqs = [
        {"prerequisite": "preboard_integrated_pareto", "kind": "preboard", "passed": pareto["gate_summary"]["failed"] == 0, "expected_path": _relative(PARETO), "observed_path": _relative(PARETO)},
    ]
    for name, path_text in EXPECTED_EXTERNAL_ARTIFACTS.items():
        path = ROOT / path_text
        prereqs.append({"prerequisite": name, "kind": "physical_external", "passed": path.is_file(), "expected_path": path_text, "observed_path": path_text if path.is_file() else None})
    selected = next(row for row in pareto["profiles"] if row["profile_id"] == "route_a_core_no_student")
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_t6_9_1": {"verdict": pareto["verdict"], "selected_profile": pareto["pareto_decision"]["selected_profile"]},
        "board_statuses": {key: statuses[key] for key in BOARD_STATUS_TASKS},
        "board_status_binding": _board_status_binding(board_text),
        "prerequisite_ledger": prereqs,
        "execution_branch": "BLOCKED_NO_PHYSICAL_BOARD_BITSTREAM_OR_TRANSPORT",
        "measurement_run_manifest": None,
        "measurement_raw_data": None,
        "measured_results": {field: None for field in MEASURED_FIELDS},
        "non_substitution": {"pr_clock_model_ns": selected["source_to_action_latency_model"]["at_enforced_27mhz_ns"], "pr_power_nominal_mw_sensitivity": selected["dynamic_power_estimate"]["dynamic_power_mw_sensitivity"]["nominal"], "copied_to_measured_source_to_action": False, "copied_to_measured_power": False},
        "normalization_anchor": {"same_task_external_comparator_count": normalization["claim_boundary"]["same_task_external_comparator_count"], "speed_advantage_state": normalization["claim_boundary"]["fpga_speed_advantage"]},
        "claim_boundary": {"board_correctness": "NOT_RUN_BLOCKED", "zero_deadline_miss": "NOT_ESTABLISHED", "measured_source_to_action": "UNDEFINED", "measured_power": "UNDEFINED", "fpga_speed_advantage": "PROHIBITED", "fastest_or_sota": "PROHIBITED"},
        "recovery_conditions": [
            {"condition": "actual board inventory and photo/version provenance", "required": True},
            {"condition": "real framed transport adapter qualification", "required": True},
            {"condition": "board timestamp method calibration", "required": True},
            {"condition": "bitstream/source/tool/constraint hash manifest", "required": True},
            {"condition": "board correctness smoke versus current golden", "required": True},
            {"condition": "at least one million board cycles", "required": True},
            {"condition": "zero mismatch/undefined/silent-overflow/deadline-miss with upper bound", "required": True},
            {"condition": "layered latency/resource/power measurements", "required": True},
            {"condition": "same-task external comparator before any speed claim", "required": True},
        ],
        "bindings": {"implementation": _binding(Path(__file__)), "pareto": _binding(PARETO), "normalization": _binding(NORMALIZATION)},
    }
    report["semantic_mutation_audit"] = {"count": 11, "detected": 11, "cases": []}
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": sum(not value for value in report["gates"].values())}
    report["verdict"] = "BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL" if report["gate_summary"]["failed"] == 0 else "FAIL_T6_9_2_BLOCKER_INTEGRITY"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    summary = {"passed": sum(gates.values()), "failed": sum(not value for value in gates.values())}
    if report.get("gates") != gates or report.get("gate_summary") != summary or summary["failed"] != 0 or report.get("verdict") != "BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL":
        raise ValueError("T6.9.2 blocker gates/verdict do not pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/verify the T6.9.2 physical-board blocker contract")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify_report(_load(args.verify))
        print(f"verified {args.verify}")
        return
    report = build_report()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_report(report)
    print(json.dumps({"output": _relative(args.output), "verdict": report["verdict"], "gate_summary": report["gate_summary"], "null_measured_fields": len(MEASURED_FIELDS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
