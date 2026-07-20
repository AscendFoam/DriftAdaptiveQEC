"""T7.1.3 evidence-bounded contracts for manuscript Figures 3 and 4."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.benchmark import claim_evidence_boundary_matrix as claim_matrix
from cnn_fpga.benchmark import route_a_board_measurement_gate as board_gate
from cnn_fpga.benchmark import route_a_hardware_pareto as hardware_pareto
from cnn_fpga.benchmark import route_a_integrated_rtl_qualification as rtl_qualification
from cnn_fpga.benchmark import route_a_promotion_gate as promotion_gate
from cnn_fpga.benchmark import route_a_smooth_formal as smooth_formal
from cnn_fpga.benchmark import route_a_tail_formal as tail_formal
from cnn_fpga.benchmark import secondary_evidence_integrity_gate as secondary_gate
from cnn_fpga.benchmark import static_gkp_same_model_lane as static_lane


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.1.3"
SCHEMA_VERSION = "t7.1.3-main-result-figure-contract-v1"
VERDICT = "PASS_MAIN_FIGURES_3_4_RESTRICTED_PREBOARD_RESULTS"

DEFAULT_REPORT = ROOT / "docs/t7_1_3_main_result_figure_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_1_3_main_result_figure_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/main_figures_3_4_contract.md"
FIGURE_DIR = ROOT / "docs/figures/t7_1_3_main_figures"
DEFAULT_MANIFEST = FIGURE_DIR / "figure_manifest.json"

SOURCES = {
    "claim_report": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix.json",
    "claim_source": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix_source_data.csv",
    "claim_code": ROOT / "cnn_fpga/benchmark/claim_evidence_boundary_matrix.py",
    "smooth_report": ROOT / "docs/t6_7_1_smooth_formal_matrix.json",
    "smooth_source": ROOT / "docs/t6_7_1_smooth_formal_matrix_source_data.csv",
    "smooth_code": ROOT / "cnn_fpga/benchmark/route_a_smooth_formal.py",
    "tail_report": ROOT / "docs/t6_7_2_abrupt_ood_tail_formal_matrix.json",
    "tail_source": ROOT / "docs/t6_7_2_abrupt_ood_tail_formal_matrix_source_data.csv",
    "tail_code": ROOT / "cnn_fpga/benchmark/route_a_tail_formal.py",
    "promotion_report": ROOT / "docs/t6_7_4_route_a_promotion_gate.json",
    "promotion_source": ROOT / "docs/t6_7_4_route_a_promotion_gate_source_data.csv",
    "promotion_code": ROOT / "cnn_fpga/benchmark/route_a_promotion_gate.py",
    "static_report": ROOT / "docs/t6_8_1_static_gkp_same_model_lane.json",
    "static_source": ROOT / "docs/t6_8_1_static_gkp_same_model_lane_source_data.csv",
    "static_code": ROOT / "cnn_fpga/benchmark/static_gkp_same_model_lane.py",
    "rtl_report": ROOT / "docs/t6_7_3_route_a_integrated_rtl_qualification.json",
    "rtl_source": ROOT / "docs/t6_7_3_route_a_integrated_rtl_source_data.csv",
    "rtl_code": ROOT / "cnn_fpga/rtl/route_a_policy_overlay.sv",
    "pr_report": ROOT / "docs/t6_9_1_route_a_hardware_pareto.json",
    "pr_source": ROOT / "docs/t6_9_1_route_a_hardware_pareto_source_data.csv",
    "pr_code": ROOT / "cnn_fpga/benchmark/route_a_hardware_pareto.py",
    "board_report": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
    "board_code": ROOT / "cnn_fpga/benchmark/route_a_board_measurement_gate.py",
    "secondary_report": ROOT / "docs/t6_19_3_secondary_evidence_integrity.json",
    "secondary_source": ROOT / "docs/t6_19_3_secondary_evidence_integrity_source_data.csv",
    "secondary_code": ROOT / "cnn_fpga/benchmark/secondary_evidence_integrity_gate.py",
    "v5_report": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json",
    "implementation": Path(__file__).resolve(),
}

FIGURE_OUTPUTS = (
    "figure3_v4_results.svg", "figure3_v4_results.pdf",
    "figure3_v4_results.png", "figure3_v4_results.tiff",
    "figure4_preboard_evidence.svg", "figure4_preboard_evidence.pdf",
    "figure4_preboard_evidence.png", "figure4_preboard_evidence.tiff",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.is_file() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def _verify_integrated_rtl(report: Mapping[str, Any]) -> bool:
    live = rtl_qualification.evaluate_gates(report)
    stored = dict(report["gates"])
    mutation_gate = stored.pop("semantic_mutations", None)
    audit = report["semantic_mutation_audit"]
    return (
        stored == live and all(live.values()) and mutation_gate is True
        and audit["count"] == audit["detected"] == 12
        and report["verdict"] == "PASS_ROUTE_A_INTEGRATED_LONG_RTL_QUALIFICATION"
    )


def _parent_verification() -> dict[str, bool]:
    smooth = _load(SOURCES["smooth_report"])
    tail = _load(SOURCES["tail_report"])
    promotion = _load(SOURCES["promotion_report"])
    static = _load(SOURCES["static_report"])
    rtl = _load(SOURCES["rtl_report"])
    pr = _load(SOURCES["pr_report"])
    board = _load(SOURCES["board_report"])
    calls = {
        "claim_matrix": lambda: claim_matrix.verify_report(path=SOURCES["claim_report"]),
        "smooth_formal": lambda: smooth_formal.verify_report(smooth),
        "tail_formal": lambda: tail_formal.verify_report(tail),
        "promotion": lambda: promotion_gate.verify_report(promotion),
        "static_lane": lambda: static_lane.verify_report(static),
        "integrated_rtl": lambda: _verify_integrated_rtl(rtl),
        "hardware_pareto": lambda: hardware_pareto.verify_report(pr),
        "board_blocker": lambda: board_gate.verify_report(board),
        "secondary_integrity": lambda: secondary_gate.verify_report(report_path=SOURCES["secondary_report"]),
    }
    checks: dict[str, bool] = {}
    for key, call in calls.items():
        try:
            result = call()
            checks[key] = result is not False
        except Exception:
            checks[key] = False
    return checks


def _record(
    figure: str, panel: str, record_id: str, metric: str, method: str, family: str,
    value: Any, unit: str, status: str, evidence_layer: str,
    source_ids: Sequence[str], selector: str, claim_ids: Sequence[str],
    lower: Any = None, upper: Any = None, metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "figure": figure, "panel": panel, "record_id": record_id,
        "metric": metric, "method": method, "family": family,
        "value": value, "lower": lower, "upper": upper, "unit": unit,
        "status": status, "evidence_layer": evidence_layer,
        "source_ids": list(source_ids), "selector": selector,
        "claim_ids": list(claim_ids), "metadata": dict(metadata or {}),
    }


def _build_records() -> list[dict[str, Any]]:
    smooth = _load(SOURCES["smooth_report"])["analysis"]
    tail = _load(SOURCES["tail_report"])["analysis"]
    static = _load(SOURCES["static_report"])
    rtl = _load(SOURCES["rtl_report"])
    pr = _load(SOURCES["pr_report"])
    board = _load(SOURCES["board_report"])
    records: list[dict[str, Any]] = []

    for row in smooth["method_summaries"]:
        method = row["method_id"]
        status = "NONDEPLOYABLE_UPPER_BOUND" if method == "hidden_state_oracle" else ("PROPOSED_RESTRICTED" if method == "proposed_route_a" else "DEPLOYABLE_BASELINE")
        records.append(_record(
            "Figure 3", "a", f"f3a_{method}", "average_ler_equal_family_seed", method, "smooth_aggregate",
            row["p_L"], "probability", status, "PROJECT_NATIVE_SIMULATION",
            ("smooth_report", "smooth_source", "smooth_code"), f"analysis.method_summaries[{method}].p_L",
            ("SMOOTH_LOCKED_EWMA_ADVANTAGE", "STATIC_GKP_SUPERIORITY"),
            row["paired_formal_seed_cluster_ci95"]["p_L"][0], row["paired_formal_seed_cluster_ci95"]["p_L"][1],
            {"p95_window_ler": row["p95_window_ler"], "global_worst_window_ler": row["global_worst_window_ler"], "errors": row["errors"]},
        ))

    primary = smooth["primary_contrast"]
    records.append(_record(
        "Figure 3", "b", "f3b_ewma_minus_route", "paired_ler_difference", "ewma_adaptive_map minus proposed_route_a", "smooth_aggregate",
        primary["estimate"], "absolute LER", "POSITIVE_RESTRICTED", "PROJECT_NATIVE_SIMULATION",
        ("smooth_report", "smooth_source", "promotion_report"), "analysis.primary_contrast", ("SMOOTH_LOCKED_EWMA_ADVANTAGE",),
        primary["ci95_low"], primary["ci95_high"],
    ))
    static_contrast = static["paired_static_contrast"]
    records.append(_record(
        "Figure 3", "b", "f3b_static_minus_route", "paired_ler_difference", "static_joint_map minus proposed_route_a", "smooth_aggregate",
        static_contrast["estimate"], "absolute LER", "NEGATIVE_RESULT", "PROJECT_NATIVE_SIMULATION",
        ("static_report", "static_source", "static_code"), "paired_static_contrast", ("STATIC_GKP_SUPERIORITY",),
        static_contrast["ci95_low"], static_contrast["ci95_high"],
    ))
    gap = static["oracle_gap"]
    records.append(_record(
        "Figure 3", "b", "f3b_oracle_gap", "static_to_oracle_gap_closure", "proposed_route_a", "smooth_aggregate",
        gap["gap_closure"], "fraction", "NEGATIVE_RESULT", "PROJECT_NATIVE_SIMULATION",
        ("static_report", "static_source"), "oracle_gap", ("STATIC_GKP_SUPERIORITY",), gap["gap_closure_ci95"][0], gap["gap_closure_ci95"][1],
    ))

    tail_rows = {(row["family"], row["method_id"]): row for row in tail["family_method_summaries"]}
    action_rows = {row["family"]: row for row in tail["action_metrics_by_family"]}
    families = ("step_calibration_shift", "telegraph_drift", "burst_outlier", "readout_reset_fault", "leakage_persistence", "compound_ood")
    for family in families:
        for method in ("ewma_adaptive_map", "proposed_route_a"):
            row = tail_rows[(family, method)]
            records.append(_record(
                "Figure 3", "c", f"f3c_{family}_{method}", "global_worst_window_error_count", method, family,
                row["global_worst_window_error_count"], "errors / 512", "SAFETY_NONINFERIORITY_NOT_IMPROVEMENT", "PROJECT_NATIVE_SIMULATION",
                ("tail_report", "tail_source", "tail_code"), f"analysis.family_method_summaries[{family},{method}]", ("TAIL_SAFETY_AND_IMPROVEMENT",),
                metadata={"average_ler": row["average_ler"], "p95_window_ler": row["seed_mean_p95_window_ler"], "global_worst_window_ler": row["global_worst_window_ler"]},
            ))
        action = action_rows[family]
        recovery = action["events"]["tail_recovery_to_open"]
        records.append(_record(
            "Figure 3", "d", f"f3d_{family}_fallback", "fallback_rate", "proposed_route_a", family,
            action["fallback_rate"], "fraction", "COST_NOT_BENEFIT", "PROJECT_NATIVE_SIMULATION",
            ("tail_report", "tail_source", "promotion_report"), f"analysis.action_metrics_by_family[{family}].fallback_rate", ("TAIL_SAFETY_AND_IMPROVEMENT",),
            metadata={"unnecessary_fallback_rate": action["unnecessary_fallback_rate"], "false_updates": action["false_updates"], "recovery_p95_decisions": recovery["p95_higher_decisions"]},
        ))
    nominal = action_rows["nominal_static"]
    records.append(_record(
        "Figure 3", "d", "f3d_nominal_fallback", "fallback_rate", "proposed_route_a", "nominal_static",
        nominal["fallback_rate"], "fraction", "NOMINAL_NONINFERIORITY_COST", "PROJECT_NATIVE_SIMULATION",
        ("tail_report", "tail_source", "promotion_report"), "analysis.nominal_noninferiority_gate", ("TAIL_SAFETY_AND_IMPROVEMENT",),
        metadata={"unnecessary_fallback_rate": nominal["unnecessary_fallback_rate"], "average_difference": 0.0},
    ))

    aggregate = rtl["aggregate_python"]
    for metric, value, unit, selector in (
        ("qualified_cycles", aggregate["cycles"], "cycles", "aggregate_python.cycles"),
        ("rtl_mismatches", sum(int(row["mismatches"]) for row in rtl["cxxrtl_families"]), "count", "cxxrtl_families[*].mismatches"),
        ("undefined_actions", aggregate["undefined_actions"], "count", "aggregate_python.undefined_actions"),
        ("silent_overflow", aggregate["silent_overflow"], "count", "aggregate_python.silent_overflow"),
    ):
        records.append(_record(
            "Figure 4", "a", f"f4a_{metric}", metric, "route_a_integrated_rtl", "preboard_long_sequence", value, unit,
            "PREBOARD_CORRECTNESS", "CXXRTL_PREBOARD", ("rtl_report", "rtl_source", "rtl_code"), selector, ("FPGA_DETERMINISTIC_ARCHITECTURE",),
        ))
    selected = next(row for row in pr["profiles"] if row["profile_id"] == pr["pareto_decision"]["selected_profile"])
    records.extend([
        _record("Figure 4", "b", "f4b_latency_cycles", "source_to_action_latency", "selected_no_student", "clock_model", selected["source_to_action_latency_model"]["cycles"], "cycles", "CLOCK_MODEL_NOT_BOARD", "POST_ROUTE_ESTIMATE", ("pr_report", "pr_source", "pr_code"), "profiles.route_a_core_no_student.source_to_action_latency_model.cycles", ("FPGA_DETERMINISTIC_ARCHITECTURE",)),
        _record("Figure 4", "b", "f4b_ii", "initiation_interval", "selected_no_student", "clock_model", selected["source_to_action_latency_model"]["initiation_interval_cycles"], "cycles", "CLOCK_MODEL_NOT_BOARD", "POST_ROUTE_ESTIMATE", ("pr_report", "pr_source", "pr_code"), "profiles.route_a_core_no_student.source_to_action_latency_model.initiation_interval_cycles", ("FPGA_DETERMINISTIC_ARCHITECTURE",)),
        _record("Figure 4", "b", "f4b_27mhz_ns", "source_to_action_latency", "selected_no_student", "27_MHz_assumption", selected["source_to_action_latency_model"]["at_enforced_27mhz_ns"], "ns", "ANALYTIC_CLOCK_CONVERSION", "POST_ROUTE_ESTIMATE", ("pr_report", "pr_source", "pr_code"), "profiles.route_a_core_no_student.source_to_action_latency_model.at_enforced_27mhz_ns", ("FPGA_DETERMINISTIC_ARCHITECTURE",)),
    ])
    for profile in pr["profiles"]:
        for seed in profile["place_route"]:
            records.append(_record(
                "Figure 4", "c", f"f4c_{profile['profile_id']}_{seed['seed']}", "achieved_fmax", profile["profile_id"], f"seed_{seed['seed']}",
                seed["achieved_fmax_mhz"], "MHz", "POST_ROUTE_ESTIMATE", "POST_ROUTE_ESTIMATE",
                ("pr_report", "pr_source", "pr_code"), f"profiles.{profile['profile_id']}.place_route.seed_{seed['seed']}", ("FPGA_DETERMINISTIC_ARCHITECTURE",),
                metadata={"timing_pass": seed["timing_pass"], "target_mhz": seed["target_mhz"]},
            ))
    for profile in pr["profiles"]:
        resources = profile["summary"]["resources_max_across_seeds"]
        for resource in ("LUT4", "DFF", "BSRAM", "MULT18X18", "MULT9X9"):
            records.append(_record(
                "Figure 4", "d", f"f4d_{profile['profile_id']}_{resource.lower()}", "resource_utilization", profile["profile_id"], resource,
                resources[resource]["used"], "count", "POST_ROUTE_ESTIMATE", "POST_ROUTE_ESTIMATE",
                ("pr_report", "pr_source", "pr_code"), f"profiles.{profile['profile_id']}.summary.resources_max_across_seeds.{resource}", ("FPGA_DETERMINISTIC_ARCHITECTURE",),
                metadata={"available": resources[resource]["available"], "fraction": resources[resource]["used"] / resources[resource]["available"]},
            ))
    records.extend([
        _record("Figure 4", "e", "f4e_board_null", "board_measured_null_fields", "physical_board", "T6.9.2", sum(value is None for value in board["measured_results"].values()), "fields", "BLOCKED_ALL_NULL", "BOARD_MEASURED", ("board_report", "board_code"), "measured_results", ("BOARD_MEASURED_CORRECTNESS_LATENCY", "FPGA_SPEED_ADVANTAGE")),
        _record("Figure 4", "e", "f4e_v5_dropped", "v5_quantized_formal_cxxrtl_pr", "V5", "T6.10.2--T6.15.4", None, "N/A", "NOT_RUN_DROPPED", "CXXRTL_PREBOARD", ("v5_report", "claim_report"), "dropped_tasks", ("V5-QUANTIZED-RETENTION", "V5-LONG-CXXRTL", "V5-FORMAL-ATOMIC-SAFETY", "V5-MULTISEED-PR")),
        _record("Figure 3", "e", "f3e_secondary_supplement", "phase6c_positive_results", "Phase6C", "task-local lanes", None, "Supplement only", "EXCLUDED_FROM_MAIN_RANKING", "PROJECT_NATIVE_SIMULATION", ("secondary_report", "secondary_source", "secondary_code"), "claim_boundary", ("P6C-MULTIMODE-POSTERIOR-WEIGHTED", "P6C-AQEC-WALLCLOCK")),
    ])
    return records


def _write_source_data(records: Sequence[Mapping[str, Any]], artifacts: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    fields = ["figure", "panel", "record_id", "metric", "method", "family", "value_json", "lower_json", "upper_json", "unit", "status", "evidence_layer", "source_ids_json", "source_hashes_json", "selector", "claim_ids_json", "metadata_json"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({
                **{key: row[key] for key in ("figure", "panel", "record_id", "metric", "method", "family", "unit", "status", "evidence_layer", "selector")},
                "value_json": json.dumps(row["value"], ensure_ascii=False), "lower_json": json.dumps(row["lower"], ensure_ascii=False), "upper_json": json.dumps(row["upper"], ensure_ascii=False),
                "source_ids_json": json.dumps(row["source_ids"], separators=(",", ":")),
                "source_hashes_json": json.dumps({key: artifacts[key]["sha256"] for key in row["source_ids"]}, sort_keys=True, separators=(",", ":")),
                "claim_ids_json": json.dumps(row["claim_ids"], separators=(",", ":")), "metadata_json": json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            })
    temporary.replace(path)


def _csv_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["record_id"] for row in csv.DictReader(handle)]


def _render_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join([
        "# T7.1.3 主图 3--4 冻结合同", "",
        "## Figure contract", "",
        "- Backend：Python / matplotlib only；183 mm 双栏，白底、editable SVG/PDF、600-dpi LZW-TIFF。",
        "- Fig.3 核心结论：Route-A 只在 pilot-locked EWMA aggregate 上有窄优势；Window 更低、static 对比和 oracle-gap closure 为负，abrupt/OOD 只建立安全/非劣且 fallback 代价高。",
        "- Fig.4 核心结论：现有证据证明 fixed-point/CXXRTL 的百万周期确定性与三 seed open-source P&R estimate；不证明板上 latency/jitter/deadline/power 或速度优势。", "",
        "## Panel map", "",
        "- Fig.3a：七方法 smooth aggregate LER；oracle 明示 nondeployable。",
        "- Fig.3b：EWMA 与 static 的 paired contrast，以及负的 static-to-oracle gap closure。",
        "- Fig.3c：六 abrupt/OOD family 的 global worst-window counts，Route-A 与 locked EWMA 重合而非改善。",
        "- Fig.3d：fallback rate 与 recovery lag 成本；Fig.3e 将 Phase 6C task-local positive 移至 Supplement。",
        "- Fig.4a--d：million-cycle correctness、6-cycle/II=1 clock model、三 seed Fmax 与完整 profile resources。",
        "- Fig.4e：42 个 board fields null；V5 quantized/formal/CXXRTL/P&R 为 not run/dropped。", "",
        "## Reviewer-risk checks", "",
        "1. `strongest deployable = Window` 与 `Route-A is not global best` 直接写在 Fig.3，不以 EWMA 预注册对比替代全方法排序。",
        "2. tail plot 使用同轴 paired points；重合表示 safety/non-inferiority，不标 improvement。",
        "3. host/software、CXXRTL、post-route estimate 与 board measured 分层；222.222 ns 只标 27-MHz clock conversion。",
        "4. student profile 只作为 optional sidecar resource context；不驱动 fast action。", "",
        "## Figure legends", "",
        "**Fig. 3 | Restricted V4 performance and safety results.** Smooth results use 24 formal seed clusters and equal family weights. Route-A improves only over the pilot-locked EWMA aggregate; Window remains the strongest deployable method, the paired static contrast is negative, and static-to-oracle gap closure is below zero. Across six abrupt/OOD families, worst-window outcomes establish safety/non-inferiority rather than improvement and require substantial fallback. Phase 6C task-local results are excluded from the main ranking. Source data are provided.", "",
        "**Fig. 4 | Pre-board deterministic execution evidence and physical-measurement boundary.** The integer/CXXRTL path completes 1,000,000 cycles without mismatch, undefined action or silent overflow. The integrated selected profile has a six-cycle, II=1 clock-model path and three-seed open-source P&R estimates; the student is an optional sidecar. All 42 physical-board fields remain null, and V5 hardware stages were not run. Source data are provided.", "",
    ])


def evaluate_gates(report: Mapping[str, Any], check_live_files: bool = True) -> dict[str, bool]:
    records = {row["record_id"]: row for row in report["records"]}
    artifacts = report["artifact_registry"]
    source_path = ROOT / report["source_data"]["path"]
    markdown_path = ROOT / report["markdown"]["path"]
    valid_claims = {row["claim_id"] for row in _load(SOURCES["claim_report"])["claims"]}
    return {
        "G01_all_parent_verifiers_pass_live": len(report["parent_verification"]) == 9 and all(report["parent_verification"].values()),
        "G02_all_artifacts_and_record_sources_are_live": all((not check_live_files or _live(binding)) for binding in artifacts.values()) and all(row["source_ids"] and set(row["source_ids"]) <= set(artifacts) for row in records.values()),
        "G03_two_figures_have_exact_contract_and_panel_maps": report["figures"]["Figure 3"]["width_mm"] == 183 and report["figures"]["Figure 3"]["height_mm"] == 137 and set(report["figures"]["Figure 3"]["panel_map"]) == set("abcde") and report["figures"]["Figure 4"]["width_mm"] == 183 and report["figures"]["Figure 4"]["height_mm"] == 127 and set(report["figures"]["Figure 4"]["panel_map"]) == set("abcde"),
        "G04_records_are_unique_complete_and_hash_traceable": len(records) == len(report["records"]) == 55 and all(row["selector"] and row["claim_ids"] for row in records.values()),
        "G05_smooth_all_methods_and_strongest_window_are_explicit": sum(key.startswith("f3a_") for key in records) == 7 and report["result_boundary"]["strongest_deployable"] == "window_map" and report["result_boundary"]["route_a_global_best"] is False,
        "G06_ewma_positive_static_negative_and_oracle_negative_are_exact": records["f3b_ewma_minus_route"]["lower"] > 0 and records["f3b_static_minus_route"]["upper"] < 0 and records["f3b_oracle_gap"]["upper"] < 0,
        "G07_tail_is_six_family_noninferiority_not_improvement": sum(key.startswith("f3c_") for key in records) == 12 and all(records[f"f3c_{family}_ewma_adaptive_map"]["value"] == records[f"f3c_{family}_proposed_route_a"]["value"] for family in report["tail_families"]) and report["result_boundary"]["broad_tail_improvement"] is False,
        "G08_fallback_and_recovery_costs_are_not_hidden": len([key for key in records if key.startswith("f3d_")]) == 7 and max(records[f"f3d_{family}_fallback"]["value"] for family in report["tail_families"]) > 0.9 and records["f3d_nominal_fallback"]["value"] > 0,
        "G09_phase6c_is_supplement_only_and_nonranking": records["f3e_secondary_supplement"]["status"] == "EXCLUDED_FROM_MAIN_RANKING" and report["result_boundary"]["phase6c_placement"] == "SUPPLEMENT_TASK_LOCAL_ONLY",
        "G10_cxxrtl_million_cycle_zero_failure_is_exact": records["f4a_qualified_cycles"]["value"] == 1_000_000 and all(records[key]["value"] == 0 for key in ("f4a_rtl_mismatches", "f4a_undefined_actions", "f4a_silent_overflow")),
        "G11_six_cycle_ii1_is_clock_model_not_board": records["f4b_latency_cycles"]["value"] == 6 and records["f4b_ii"]["value"] == 1 and all(records[key]["status"] in {"CLOCK_MODEL_NOT_BOARD", "ANALYTIC_CLOCK_CONVERSION"} for key in ("f4b_latency_cycles", "f4b_ii", "f4b_27mhz_ns")),
        "G12_pr_has_two_profiles_three_seeds_resources_and_student_boundary": len([key for key in records if key.startswith("f4c_")]) == 6 and len([key for key in records if key.startswith("f4d_")]) == 10 and report["result_boundary"]["student_role"] == "OPTIONAL_ABLATION_SIDECAR_NOT_FAST_ACTION",
        "G13_board_and_v5_hardware_remain_null_or_dropped": records["f4e_board_null"]["value"] == 42 and records["f4e_board_null"]["status"] == "BLOCKED_ALL_NULL" and records["f4e_v5_dropped"]["value"] is None and records["f4e_v5_dropped"]["status"] == "NOT_RUN_DROPPED",
        "G14_claim_references_exist_and_forbidden_promotions_are_complete": all(set(row["claim_ids"]) <= valid_claims for row in records.values()) and report["forbidden_promotions"] == ["Route-A globally best deployable", "broad abrupt/OOD improvement", "positive static-to-oracle gap closure", "V5 hardware implemented", "P&R estimate as board measurement", "measured FPGA speed/power advantage"],
        "G15_source_markdown_and_export_contract_are_live": source_path.is_file() and report["source_data"]["rows"] == len(records) and len(_csv_ids(source_path)) == len(records) and set(_csv_ids(source_path)) == set(records) and markdown_path.is_file() and all(token in markdown_path.read_text(encoding="utf-8") for token in ("strongest deployable = Window", "safety/non-inferiority", "42 个 board fields null", "not run/dropped", "Source data are provided")) and report["export_contract"] == {"backend": "Python/matplotlib only", "width_mm": 183, "svg_text": "editable", "pdf_fonttype": 42, "tiff_dpi": 600, "png_dpi": 300, "outputs": list(FIGURE_OUTPUTS)} and (not check_live_files or _live(report["source_data"]) and _live(report["markdown"])),
        "G16_one_substantive_mutation_per_gate_fails_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 16 and len(report["semantic_mutation_audit"]["cases"]) == 16,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    def record(value: dict[str, Any], record_id: str) -> dict[str, Any]:
        return next(row for row in value["records"] if row["record_id"] == record_id)
    def attempt(name: str, target: str, change: Any) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
        change(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[target]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": target, "rejected": rejected})
    attempt("parent_failure", "G01_all_parent_verifiers_pass_live", lambda x: x["parent_verification"].update(smooth_formal=False))
    attempt("forged_hash", "G02_all_artifacts_and_record_sources_are_live", lambda x: record(x, "f3a_window_map").update(source_ids=[]))
    attempt("wrong_canvas", "G03_two_figures_have_exact_contract_and_panel_maps", lambda x: x["figures"]["Figure 3"].update(height_mm=100))
    attempt("duplicate_record", "G04_records_are_unique_complete_and_hash_traceable", lambda x: x["records"][-1].update(record_id=x["records"][0]["record_id"]))
    attempt("promote_global_best", "G05_smooth_all_methods_and_strongest_window_are_explicit", lambda x: x["result_boundary"].update(route_a_global_best=True))
    attempt("flip_static", "G06_ewma_positive_static_negative_and_oracle_negative_are_exact", lambda x: record(x, "f3b_static_minus_route").update(upper=1e-5))
    attempt("invent_tail_gain", "G07_tail_is_six_family_noninferiority_not_improvement", lambda x: x["result_boundary"].update(broad_tail_improvement=True))
    attempt("hide_fallback", "G08_fallback_and_recovery_costs_are_not_hidden", lambda x: record(x, "f3d_step_calibration_shift_fallback").update(value=0.1))
    attempt("promote_secondary", "G09_phase6c_is_supplement_only_and_nonranking", lambda x: record(x, "f3e_secondary_supplement").update(status="MAIN_RANKING"))
    attempt("erase_cycles", "G10_cxxrtl_million_cycle_zero_failure_is_exact", lambda x: record(x, "f4a_qualified_cycles").update(value=1_000))
    attempt("promote_latency", "G11_six_cycle_ii1_is_clock_model_not_board", lambda x: record(x, "f4b_latency_cycles").update(status="BOARD_MEASURED"))
    attempt("student_fast_action", "G12_pr_has_two_profiles_three_seeds_resources_and_student_boundary", lambda x: x["result_boundary"].update(student_role="PRIMARY_FAST_ACTION"))
    attempt("fill_board", "G13_board_and_v5_hardware_remain_null_or_dropped", lambda x: record(x, "f4e_board_null").update(value=0, status="MEASURED"))
    attempt("remove_forbidden", "G14_claim_references_exist_and_forbidden_promotions_are_complete", lambda x: x.update(forbidden_promotions=[]))
    attempt("switch_backend", "G15_source_markdown_and_export_contract_are_live", lambda x: x["export_contract"].update(backend="R"))
    attempt("forge_mutations", "G16_one_substantive_mutation_per_gate_fails_closed", lambda x: x.update(semantic_mutation_audit={"count": 16, "detected": 15, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report(source_data: Path = DEFAULT_SOURCE_DATA, markdown: Path = DEFAULT_MARKDOWN) -> dict[str, Any]:
    artifacts = {key: _binding(path) for key, path in SOURCES.items()}
    records = _build_records()
    _write_source_data(records, artifacts, source_data)
    report: dict[str, Any] = {
        "task_id": TASK_ID, "schema_version": SCHEMA_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figures": {
            "Figure 3": {"core_conclusion": "V4 has a narrow locked-EWMA aggregate result but loses the strongest deployable and static comparisons; tail evidence is safety/non-inferiority with substantial fallback cost.", "archetype": "quantitative evidence composite", "width_mm": 183, "height_mm": 137, "panel_map": {"a": "smooth all-method LER", "b": "paired contrasts and oracle gap", "c": "abrupt/OOD worst windows", "d": "fallback and recovery cost", "e": "secondary-placement boundary"}},
            "Figure 4": {"core_conclusion": "The current hardware claim is deterministic pre-board correctness and post-route estimation, while every physical-board metric remains null.", "archetype": "quantitative evidence composite", "width_mm": 183, "height_mm": 127, "panel_map": {"a": "million-cycle correctness", "b": "clock-model latency", "c": "three-seed Fmax", "d": "resource context", "e": "board-null and dropped-V5 boundary"}},
        },
        "artifact_registry": artifacts, "parent_verification": _parent_verification(), "records": records,
        "tail_families": ["step_calibration_shift", "telegraph_drift", "burst_outlier", "readout_reset_fault", "leakage_persistence", "compound_ood"],
        "result_boundary": {"strongest_deployable": "window_map", "route_a_global_best": False, "broad_tail_improvement": False, "phase6c_placement": "SUPPLEMENT_TASK_LOCAL_ONLY", "student_role": "OPTIONAL_ABLATION_SIDECAR_NOT_FAST_ACTION", "board_measured": False},
        "forbidden_promotions": ["Route-A globally best deployable", "broad abrupt/OOD improvement", "positive static-to-oracle gap closure", "V5 hardware implemented", "P&R estimate as board measurement", "measured FPGA speed/power advantage"],
        "export_contract": {"backend": "Python/matplotlib only", "width_mm": 183, "svg_text": "editable", "pdf_fonttype": 42, "tiff_dpi": 600, "png_dpi": 300, "outputs": list(FIGURE_OUTPUTS)},
        "source_data": {**_binding(source_data), "rows": len(records)}, "markdown": {"path": _relative(markdown), "sha256": "", "bytes": 0},
        "semantic_mutation_audit": {"count": 16, "detected": 16, "cases": []}, "verdict": VERDICT,
    }
    _atomic_text(_render_markdown(report), markdown)
    report["markdown"] = _binding(markdown)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": [key for key, value in report["gates"].items() if not value]}
    report["verdict"] = VERDICT if not report["gate_summary"]["failed"] else "FAIL_MAIN_FIGURES_3_4_CONTRACT"
    report["analysis_sha256"] = _canonical_sha256({key: report[key] for key in ("figures", "artifact_registry", "parent_verification", "records", "tail_families", "result_boundary", "forbidden_promotions", "export_contract", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict")})
    return report


def verify_report(report: Mapping[str, Any] | None = None, path: Path = DEFAULT_REPORT) -> dict[str, bool]:
    value = dict(report) if report is not None else _load(path)
    gates = evaluate_gates(value)
    expected_hash = _canonical_sha256({key: value[key] for key in ("figures", "artifact_registry", "parent_verification", "records", "tail_families", "result_boundary", "forbidden_promotions", "export_contract", "source_data", "markdown", "semantic_mutation_audit", "gates", "verdict")})
    checks = {"identity": value.get("task_id") == TASK_ID and value.get("schema_version") == SCHEMA_VERSION, "gates": value.get("gates") == gates and all(gates.values()), "verdict": value.get("verdict") == VERDICT, "analysis_hash": value.get("analysis_sha256") == expected_hash}
    if not all(checks.values()):
        raise ValueError(f"T7.1.3 contract verification failed: {[key for key, passed in checks.items() if not passed]}")
    return checks


def verify_bundle(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, bool]:
    manifest = _load(manifest_path)
    outputs = manifest.get("outputs", {})
    checks = {
        "identity": manifest.get("task_id") == TASK_ID and manifest.get("backend") == "Python/matplotlib only",
        "contract_live": manifest.get("contract") == _binding(DEFAULT_REPORT), "source_data_live": manifest.get("source_data") == _binding(DEFAULT_SOURCE_DATA),
        "outputs_exact": set(outputs) == set(FIGURE_OUTPUTS), "outputs_live": set(outputs) == set(FIGURE_OUTPUTS) and all(_live(binding) for binding in outputs.values()),
        "editable_svg": manifest.get("qa", {}).get("svg_text_nodes", 0) >= 50 and manifest.get("qa", {}).get("svg_path_text_promotion") is False,
        "raster_dimensions": all(value >= 3000 for value in manifest.get("qa", {}).get("tiff_min_dimension_px", {}).values()),
        "visual_contract": manifest.get("qa", {}).get("backend_exclusive") is True and manifest.get("qa", {}).get("manual_visual_qa") == "PASS",
    }
    if not all(checks.values()):
        raise ValueError(f"T7.1.3 bundle verification failed: {[key for key, passed in checks.items() if not passed]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        verify_report(path=args.report)
        print(json.dumps({"verified": _relative(args.report), "verdict": VERDICT}, ensure_ascii=False))
        return 0
    report = build_report(args.source_data, args.markdown)
    _atomic_json(report, args.report)
    verify_report(report, args.report)
    print(json.dumps({"output": _relative(args.report), "records": len(report["records"]), "gates": report["gate_summary"], "verdict": report["verdict"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
