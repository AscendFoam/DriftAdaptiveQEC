from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.8.6"
SCHEMA_VERSION = "t6.8.6-fpga-decoder-normalization-v1"
SOURCE_LEDGER = ROOT / "configs" / "literature" / "t6_8_6_fpga_decoder_sources.json"
T5_REPORT = ROOT / "docs" / "t5_5_2_target_device_synthesis.json"
T67_REPORT = ROOT / "docs" / "t6_7_3_route_a_integrated_rtl_qualification.json"
SOURCE_CSV = ROOT / "docs" / "t6_8_6_fpga_decoder_normalization_source_data.csv"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_8_6_fpga_decoder_normalization.json"

NUMERIC_FIELDS = (
    "reported_latency_ns",
    "decoder_core_ns",
    "source_to_action_ns",
    "closed_loop_ns",
    "average_per_round_ns",
    "iteration_ns",
    "ii_ns",
    "throughput_period_ns",
    "clock_mhz",
    "latency_cycles",
    "lut",
    "lut_percent",
    "logic_units",
    "ff",
    "ff_percent",
    "bram",
    "bram_percent",
    "bram_36k",
    "bram_18k",
    "dsp",
    "dsp_percent",
    "memory_bytes",
    "power_w",
)

REQUIRED_ROW_FIELDS = {
    "row_id",
    "source_id",
    "code_family",
    "decoder",
    "input_semantics",
    "problem_size",
    "noise_model",
    "hardware_platform",
    "device",
    "latency_evidence",
    "resource_evidence",
    "qpu_in_loop",
    "physical_board_executed",
    "latency_boundary",
    "latency_statistic",
    "precision",
    "logic_unit_type",
    "memory_reported",
    "direct_speed_comparable_to_project",
    "incomparability_reasons",
    "numeric_sources",
    *NUMERIC_FIELDS,
}

EXTERNAL_ROW_IDS = {
    "lilliput_d5_m2",
    "helios_d21",
    "collision_clustering_d21",
    "local_clustering_d17_adaptive_hl",
    "overwater_nn_d5",
    "caune_stability8_9round_feedback",
    "maurer_gross_int4_x",
    "yang_nn_d3_closed_loop",
}
PROJECT_ROW_IDS = {"project_t5_fast_path_core_pr", "project_t6_route_a_integrated_cxxrtl"}


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


def _row_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["row_id"]): row for row in report["rows"]}


def _all_numeric_fields_typed_and_sourced(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        locators = row["numeric_sources"]
        for field in NUMERIC_FIELDS:
            value = row[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                return False
            if value is not None and (field not in locators or not str(locators[field]).strip()):
                return False
            if value is None and field in locators:
                return False
        if set(locators) - set(NUMERIC_FIELDS):
            return False
    return True


def _project_anchor_values(t5: Mapping[str, Any], t67: Mapping[str, Any]) -> dict[str, Any]:
    run = t5["place_route"][0]
    util = run["utilization"]
    return {
        "t5_status": t5["status"],
        "t5_verdict": t5["verdict"],
        "t5_target_mhz": t5["target_contract"]["target_mhz"],
        "t5_core_cycles": t5["latency_estimate"]["core_cycles"],
        "t5_core_ns": t5["latency_estimate"]["at_target_27mhz_ns"],
        "t5_ii_ns": t5["latency_estimate"]["initiation_interval_at_target_ns"],
        "t5_lut4": util["LUT4"]["used"],
        "t5_lut4_available": util["LUT4"]["available"],
        "t5_dff": util["DFF"]["used"],
        "t5_dff_available": util["DFF"]["available"],
        "t5_bsram": util["BSRAM"]["used"],
        "t5_bsram_available": util["BSRAM"]["available"],
        "t67_verdict": t67["verdict"],
        "t67_cycles": t67["aggregate_python"]["cycles"],
        "t67_silent_overflow": t67["aggregate_python"]["silent_overflow"],
        "t67_undefined_actions": t67["aggregate_python"]["undefined_actions"],
    }


def _anchors_match_rows(report: Mapping[str, Any]) -> bool:
    rows = _row_map(report)
    core = rows.get("project_t5_fast_path_core_pr", {})
    route = rows.get("project_t6_route_a_integrated_cxxrtl", {})
    anchor = report["project_anchor_verification"]
    comparisons = [
        core.get("clock_mhz") == anchor["t5_target_mhz"],
        core.get("latency_cycles") == anchor["t5_core_cycles"],
        math.isclose(core.get("reported_latency_ns", math.nan), anchor["t5_core_ns"], rel_tol=0.0, abs_tol=1e-12),
        math.isclose(core.get("ii_ns", math.nan), anchor["t5_ii_ns"], rel_tol=0.0, abs_tol=1e-12),
        core.get("lut") == anchor["t5_lut4"],
        core.get("ff") == anchor["t5_dff"],
        core.get("bram") == anchor["t5_bsram"],
        route.get("latency_cycles") == 6,
        anchor["t5_status"] == "PASS",
        anchor["t67_verdict"] == "PASS_ROUTE_A_INTEGRATED_LONG_RTL_QUALIFICATION",
        anchor["t67_cycles"] == 1_000_000,
        anchor["t67_silent_overflow"] == anchor["t67_undefined_actions"] == 0,
    ]
    derived = [
        math.isclose(core.get("lut_percent", math.nan), 100.0 * anchor["t5_lut4"] / anchor["t5_lut4_available"], rel_tol=0.0, abs_tol=1e-12),
        math.isclose(core.get("ff_percent", math.nan), 100.0 * anchor["t5_dff"] / anchor["t5_dff_available"], rel_tol=0.0, abs_tol=1e-12),
        math.isclose(core.get("bram_percent", math.nan), 100.0 * anchor["t5_bsram"] / anchor["t5_bsram_available"], rel_tol=0.0, abs_tol=1e-12),
    ]
    return all(comparisons + derived)


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    rows = report["rows"]
    row_ids = {row["row_id"] for row in rows}
    source_ids = {source["source_id"] for source in report["sources"]}
    source_by_id = {source["source_id"]: source for source in report["sources"]}
    integrated = _row_map(report).get("project_t6_route_a_integrated_cxxrtl", {})
    time_and_resource_fields = (
        "reported_latency_ns", "decoder_core_ns", "source_to_action_ns", "closed_loop_ns",
        "average_per_round_ns", "iteration_ns", "ii_ns", "throughput_period_ns", "clock_mhz",
        "lut", "lut_percent", "logic_units", "ff", "ff_percent", "bram", "bram_percent",
        "bram_36k", "bram_18k", "dsp", "dsp_percent", "memory_bytes", "power_w",
    )
    return {
        "G01_primary_sources_and_formal_refresh_are_explicit": report["frozen_at"] == "2026-07-18" and len(report["sources"]) == 10 and all(source["primary"] is True and source["version"] for source in report["sources"]) and all("doi:" in source["formal_identifier"] for source in report["sources"][:4]),
        "G02_concrete_implementation_rows_have_complete_schema": len(rows) == 10 and row_ids == EXTERNAL_ROW_IDS | PROJECT_ROW_IDS and all(set(row) == REQUIRED_ROW_FIELDS for row in rows),
        "G03_every_non_null_numeric_value_has_a_primary_locator": _all_numeric_fields_typed_and_sourced(rows),
        "G04_absent_values_are_json_null_not_placeholders": all(all(value is None or isinstance(value, (int, float)) and not isinstance(value, bool) for value in (row[field] for field in NUMERIC_FIELDS)) for row in rows) and all("NR" not in json.dumps(row) and "N/A" not in json.dumps(row) for row in rows),
        "G05_latency_boundary_and_statistic_are_never_implicit": all(row["latency_boundary"] and row["latency_statistic"] and row["latency_evidence"] for row in rows),
        "G06_hardware_and_resource_evidence_levels_are_separate": all(row["resource_evidence"] and isinstance(row["qpu_in_loop"], bool) and isinstance(row["physical_board_executed"], bool) for row in rows),
        "G07_no_external_cross_code_speed_row_is_marked_comparable": all(row["direct_speed_comparable_to_project"] is False and len(row["incomparability_reasons"]) >= 4 for row in rows if row["row_id"] in EXTERNAL_ROW_IDS),
        "G08_real_qpu_closed_loop_is_not_conflated_with_synthetic_fpga": {row["row_id"] for row in rows if row["qpu_in_loop"]} == {"caune_stability8_9round_feedback", "yang_nn_d3_closed_loop"} and all((row["closed_loop_ns"] is not None) == row["qpu_in_loop"] for row in rows),
        "G09_project_rows_recompute_from_hash_bound_live_reports": _anchors_match_rows(report) and all(len(report["bindings"][key]["sha256"]) == 64 for key in ("source_ledger", "t5_report", "t67_report", "source_csv", "implementation")),
        "G10_source_csv_is_complete_and_hash_bound": report["source_data"]["rows"] == len(rows) and len(report["source_data"]["sha256"]) == 64 and report["source_data"]["path"].endswith(".csv"),
        "G11_claim_boundary_forbids_fastest_or_sota": report["claim_boundary"] == {"same_task_external_comparator_count": 0, "fpga_speed_advantage": "UNESTABLISHED", "fastest_or_sota": "PROHIBITED", "real_board_source_to_action": "PENDING_T6.9.2", "allowed_statement": "deterministic six-cycle preboard Route-A architecture with no cross-code speed ranking"},
        "G12_integrated_route_a_time_resources_remain_null_until_t6_9_1": integrated.get("latency_cycles") == 6 and all(integrated.get(field) is None for field in time_and_resource_fields) and integrated.get("resource_evidence") == "NOT_YET_INTEGRATED_PLACE_AND_ROUTE",
        "G13_target_specific_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 13,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 13, "detected": 13, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("downgrade_primary_source", "G01_primary_sources_and_formal_refresh_are_explicit", lambda x: x["sources"][0].update(primary=False))
    attempt("drop_required_field", "G02_concrete_implementation_rows_have_complete_schema", lambda x: x["rows"][0].pop("latency_boundary"))
    attempt("remove_numeric_locator", "G03_every_non_null_numeric_value_has_a_primary_locator", lambda x: x["rows"][0]["numeric_sources"].pop("reported_latency_ns"))
    attempt("replace_null_with_NR", "G04_absent_values_are_json_null_not_placeholders", lambda x: x["rows"][0].update(source_to_action_ns="NR"))
    attempt("erase_latency_boundary", "G05_latency_boundary_and_statistic_are_never_implicit", lambda x: x["rows"][1].update(latency_boundary=""))
    attempt("erase_resource_evidence", "G06_hardware_and_resource_evidence_levels_are_separate", lambda x: x["rows"][2].update(resource_evidence=""))
    attempt("promote_cross_code_speed_comparison", "G07_no_external_cross_code_speed_row_is_marked_comparable", lambda x: x["rows"][1].update(direct_speed_comparable_to_project=True))
    attempt("mark_synthetic_as_real_qpu", "G08_real_qpu_closed_loop_is_not_conflated_with_synthetic_fpga", lambda x: x["rows"][2].update(qpu_in_loop=True))
    attempt("forge_project_core_latency", "G09_project_rows_recompute_from_hash_bound_live_reports", lambda x: _row_map(x)["project_t5_fast_path_core_pr"].update(reported_latency_ns=1.0))
    attempt("truncate_csv_hash", "G10_source_csv_is_complete_and_hash_bound", lambda x: x["source_data"].update(sha256="0"))
    attempt("claim_fastest", "G11_claim_boundary_forbids_fastest_or_sota", lambda x: x["claim_boundary"].update(fastest_or_sota="ESTABLISHED"))
    attempt("invent_integrated_fmax", "G12_integrated_route_a_time_resources_remain_null_until_t6_9_1", lambda x: _row_map(x)["project_t6_route_a_integrated_cxxrtl"].update(clock_mhz=27.0))
    attempt("forge_mutation_count", "G13_target_specific_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 13, "detected": 12, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def _write_source_csv(rows: list[Mapping[str, Any]]) -> None:
    fieldnames = [field for field in rows[0] if field != "numeric_sources"] + ["numeric_sources_json"]
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            flattened["incomparability_reasons"] = json.dumps(row["incomparability_reasons"], ensure_ascii=False, separators=(",", ":"))
            flattened["numeric_sources_json"] = json.dumps(flattened.pop("numeric_sources"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            writer.writerow(flattened)


def build_report() -> dict[str, Any]:
    ledger = json.loads(SOURCE_LEDGER.read_text(encoding="utf-8"))
    t5 = json.loads(T5_REPORT.read_text(encoding="utf-8"))
    t67 = json.loads(T67_REPORT.read_text(encoding="utf-8"))
    rows = ledger["rows"]
    _write_source_csv(rows)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_at": ledger["frozen_at"],
        "scope": ledger["scope"],
        "taxonomy": {
            "latency_boundaries": ["decoder core", "average per measurement round", "source-to-action", "closed loop", "iteration only"],
            "statistics": ["deterministic/fixed", "mean", "post-implementation estimate", "convergence-conditioned mean"],
            "hardware_evidence": ["post-implementation estimate", "FPGA with synthetic input", "real-QPU closed loop", "preboard CXXRTL"],
            "normalization_rule": "No ranking unless code family, input/action semantics, problem size, precision, boundary, statistic and hardware evidence all match.",
        },
        "sources": ledger["sources"],
        "rows": rows,
        "project_anchor_verification": _project_anchor_values(t5, t67),
        "comparison_eligibility": {
            "external_rows": len(EXTERNAL_ROW_IDS),
            "same_task_external_rows": [],
            "reason": "Every external row differs in code family, decoder task/boundary, problem size, or hardware evidence; the project also lacks integrated real-board measurements.",
        },
        "claim_boundary": {
            "same_task_external_comparator_count": 0,
            "fpga_speed_advantage": "UNESTABLISHED",
            "fastest_or_sota": "PROHIBITED",
            "real_board_source_to_action": "PENDING_T6.9.2",
            "allowed_statement": "deterministic six-cycle preboard Route-A architecture with no cross-code speed ranking",
        },
        "bindings": {
            "implementation": _binding(Path(__file__)),
            "source_ledger": _binding(SOURCE_LEDGER),
            "t5_report": _binding(T5_REPORT),
            "t67_report": _binding(T67_REPORT),
            "source_csv": _binding(SOURCE_CSV),
        },
        "source_data": {"path": _relative(SOURCE_CSV), "sha256": _sha256(SOURCE_CSV), "bytes": SOURCE_CSV.stat().st_size, "rows": len(rows)},
    }
    report["semantic_mutation_audit"] = {"count": 13, "detected": 13, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": sum(not value for value in report["gates"].values())}
    report["verdict"] = "PASS_FPGA_DECODER_NORMALIZATION_NO_SPEED_CLAIM" if all(report["gates"].values()) else "FAIL_FPGA_DECODER_NORMALIZATION"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    expected = "PASS_FPGA_DECODER_NORMALIZATION_NO_SPEED_CLAIM" if all(gates.values()) else "FAIL_FPGA_DECODER_NORMALIZATION"
    if report.get("gates") != gates or report.get("verdict") != expected or not all(gates.values()):
        raise ValueError("T6.8.6 gates/verdict do not recompute")
    for item in report["bindings"].values():
        path = ROOT / item["path"]
        if not path.is_file() or _sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"T6.8.6 bound artifact drifted: {item['path']}")
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != report["source_data"]["rows"] or _sha256(SOURCE_CSV) != report["source_data"]["sha256"]:
        raise ValueError("T6.8.6 source CSV drifted")
    t5 = json.loads(T5_REPORT.read_text(encoding="utf-8"))
    t67 = json.loads(T67_REPORT.read_text(encoding="utf-8"))
    if report["project_anchor_verification"] != _project_anchor_values(t5, t67):
        raise ValueError("T6.8.6 project live anchors drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    report = build_report()
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_report(json.loads(args.artifact.read_text(encoding="utf-8")))
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "rows": len(report["rows"]), "same_task_external": 0, "speed_advantage": report["claim_boundary"]["fpga_speed_advantage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
