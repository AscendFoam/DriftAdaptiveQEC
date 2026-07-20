"""T6.9.3 claim-specific final evidence GO/NO-GO for Route-A."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.9.3"
SCHEMA_VERSION = "t6.9.3-route-a-final-evidence-gate-v1"
DEFAULT_ARTIFACT = ROOT / "docs/t6_9_3_route_a_final_evidence_gate.json"
SOURCE_CSV = ROOT / "docs/t6_9_3_route_a_final_evidence_gate_source_data.csv"
PARENTS = {
    "promotion": ROOT / "docs/t6_7_4_route_a_promotion_gate.json",
    "claim_matrix": ROOT / "docs/t6_8_7_route_a_claim_matrix.json",
    "hardware_pareto": ROOT / "docs/t6_9_1_route_a_hardware_pareto.json",
    "board_blocker": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
}
FINAL_CLAIM_IDS = {
    "CONTRACT_SYSTEM_INTEGRATION",
    "SMOOTH_LOCKED_EWMA_ADVANTAGE",
    "STATIC_GKP_SUPERIORITY",
    "STATIC_K4_HARD_ACTION_EQUIVALENCE",
    "TAIL_SAFETY_AND_IMPROVEMENT",
    "GENERAL_DRIFT_EXTERNAL_COMPARISON",
    "PUVIANI_NMF_SURPASS",
    "FPGA_DETERMINISTIC_ARCHITECTURE",
    "BOARD_MEASURED_CORRECTNESS_LATENCY",
    "FPGA_SPEED_ADVANTAGE",
    "CNN_AND_HMM_ROLE",
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


def _source_claim_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["claim_id"] for row in csv.DictReader(handle)]


def _parent_evidence(key: str, selectors: list[str]) -> dict[str, Any]:
    return {"task_id": _load(PARENTS[key])["task_id"], "artifact": _binding(PARENTS[key]), "selectors": selectors}


def _claim(
    claim_id: str,
    final_state: str,
    phase7_role: str,
    allowed_wording: str,
    forbidden_wording: list[str],
    current_result: Mapping[str, Any],
    evidence: list[Mapping[str, Any]],
    remaining_gate: str,
    revocation_conditions: list[str],
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "final_state": final_state,
        "phase7_role": phase7_role,
        "allowed_wording": allowed_wording,
        "forbidden_wording": forbidden_wording,
        "current_result": dict(current_result),
        "evidence": [dict(row) for row in evidence],
        "remaining_gate": remaining_gate,
        "revocation_conditions": revocation_conditions,
    }


def _claims(parents: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    promotion = parents["promotion"]
    matrix = {row["claim_id"]: row for row in parents["claim_matrix"]["claims"]}
    pareto = parents["hardware_pareto"]
    board = parents["board_blocker"]
    smooth = promotion["scientific_results"]["smooth"]
    tail = promotion["scientific_results"]["tail"]
    rtl = promotion["scientific_results"]["rtl"]
    core_profile = next(row for row in pareto["profiles"] if row["profile_id"] == "route_a_core_no_student")
    core_resources = core_profile["summary"]["resources_max_across_seeds"]
    return [
        _claim(
            "CONTRACT_SYSTEM_INTEGRATION", "SUPPORTED_RESTRICTED_PREBOARD", "restricted_preboard_main",
            "Route-A integrates a locked MAP fast path with regime-aware contract/FSM safety and passes preregistered simulator plus one-million-cycle preboard correctness gates.",
            ["overall best decoder", "broad tail improvement", "measured closed-loop hardware"],
            {"promotion_verdict": promotion["verdict"], "rtl_cycles": rtl["cycles"], "rtl_mismatches": rtl["rtl_mismatches"], "measured_board_latency": rtl["measured_board_latency"]},
            [_parent_evidence("promotion", ["promotion_decision", "scientific_results.rtl"]), _parent_evidence("hardware_pareto", ["pareto_decision", "profiles.route_a_core_no_student"])],
            "T6.9.2 board evidence plus stronger full-system performance is required for a full cross-lane paper.",
            ["any source/config/seed hash drift", "any nonzero RTL/board mismatch", "wording expands beyond simulator/preboard scope"],
        ),
        _claim(
            "SMOOTH_LOCKED_EWMA_ADVANTAGE", "SUPPORTED_PAIRED_OUTCOME", "restricted_preboard_main",
            "On the preregistered smooth matrix, Route-A lowers aggregate paired LER relative to the locked EWMA baseline; periodic drift is the only Holm-confirmed family.",
            ["best smooth decoder", "advantage over static or Window MAP", "all-family drift advantage"],
            {"primary_contrast": smooth["primary_contrast"], "holm_confirmed_families": smooth["holm_confirmed_families"], "route_a_is_global_best": smooth["route_a_is_global_best_deployable"]},
            [_parent_evidence("promotion", ["scientific_results.smooth.primary_contrast", "scientific_results.smooth.holm_confirmed_families"])],
            "A new frozen method must beat the strongest deployable baselines across preregistered families before global LER wording.",
            ["paired LCB becomes non-positive", "periodic-only boundary is omitted", "static/Window rows are removed"],
        ),
        _claim(
            "STATIC_GKP_SUPERIORITY", "FALSIFIED", "mandatory_main_negative",
            "Static joint MAP has lower average LER than Route-A on the frozen same-model smooth benchmark.",
            ["Route-A outperforms static GKP", "static-to-oracle gap closure"],
            matrix["STATIC_GKP_SUPERIORITY"]["current_result"],
            [_parent_evidence("claim_matrix", ["claims.STATIC_GKP_SUPERIORITY"])],
            "Requires a new preregistered same-model experiment with paired improvement CI above zero.",
            ["never promote from p95 alone", "average/worst and paired CI must remain visible"],
        ),
        _claim(
            "STATIC_K4_HARD_ACTION_EQUIVALENCE", "SUPPORTED_PREBOARD_NARROW", "supplement_or_hardware_ablation",
            "For the frozen covariance/prior and complete 10-bit syndrome domain, K4/full static MAP hard actions are identical; resource reductions remain proxies until integrated implementation.",
            ["K4 universally exact", "measured FPGA compression advantage"],
            matrix["STATIC_K4_HARD_ACTION_EQUIVALENCE"]["current_result"],
            [_parent_evidence("claim_matrix", ["claims.STATIC_K4_HARD_ACTION_EQUIVALENCE"])],
            "Re-enumerate every changed model and integrate K4 into RTL/P&R before hardware promotion.",
            ["any hard disagreement", "changed prior without revalidation", "proxy described as measurement"],
        ),
        _claim(
            "TAIL_SAFETY_AND_IMPROVEMENT", "SAFETY_NONINFERIORITY_ONLY", "mandatory_main_limitation",
            "Abrupt/OOD gates establish locked-EWMA safety/non-inferiority under preregistered catastrophic and nominal margins, not broad tail-LER improvement.",
            ["tail advantage", "fault robustness improvement", "low-cost fallback"],
            {"tail_safety_gate_passes": tail["tail_safety_gate_passes"], "confirmed_average_improvement_families": tail["confirmed_average_improvement_families"], "broad_tail_improvement_confirmed": tail["broad_tail_improvement_confirmed"], "exact_equal_average_families": tail["exact_equal_average_families"]},
            [_parent_evidence("promotion", ["scientific_results.tail"])],
            "Requires preregistered induced-error reduction plus acceptable fallback/false-update costs across abrupt/compound families.",
            ["any catastrophic/noninferiority gate failure", "claim says improvement while confirmed family count remains zero"],
        ),
        _claim(
            "GENERAL_DRIFT_EXTERNAL_COMPARISON", "PERFORMANCE_OUTCOME_BUDGET_FAIL", "external_table_with_qualification",
            "Route-A has lower paired LER than the pinned BOCD wrapper on common traces, while that external comparator fails the strict worst-update wall-clock budget; general drift-adaptive SOTA is not established.",
            ["matched-budget external superiority", "general drift-adaptive SOTA", "Bhardwaj exact reproduction"],
            {"paired_outcome": matrix["GENERAL_DRIFT_BOCD_OUTCOME"]["current_result"], "sota_state": matrix["GENERAL_DRIFT_MATCHED_BUDGET_SOTA"]["state"]},
            [_parent_evidence("claim_matrix", ["claims.GENERAL_DRIFT_BOCD_OUTCOME", "claims.GENERAL_DRIFT_MATCHED_BUDGET_SOTA"])],
            "At least two closest external methods must pass identical worst-case compute/wall-clock budgets.",
            ["budget failure omitted", "one external wrapper is generalized to the field", "trace identity fails"],
        ),
        _claim(
            "PUVIANI_NMF_SURPASS", "PROHIBITED_SOURCE_INCOMPLETE", "negative_reproduction_or_supplement",
            "Official GQF paper-exact reproduction is blocked by missing/inconsistent source artifacts; no same-GQF lifetime comparison or NMF-surpass claim is available.",
            ["surpasses Puviani NMF", "longer physical lifetime than NMF", "paper-exact reproduction"],
            matrix["PUVIANI_NMF_SURPASS"]["current_result"],
            [_parent_evidence("claim_matrix", ["claims.PUVIANI_NMF_SURPASS"])],
            "All 15 exact gates plus same-GQF matched lifetime LCB must pass; external missing artifacts are not imputed.",
            ["reduced diagnostic substitutes for exact", "project-internal controller substitutes for official NMF"],
        ),
        _claim(
            "FPGA_DETERMINISTIC_ARCHITECTURE", "SUPPORTED_PR_ESTIMATE", "restricted_preboard_main",
            "The no-student Route-A profile preserves the deterministic six-cycle architecture and passes three-seed open-source P&R at 27 MHz; Fmax/resources/power are estimates, not board measurements.",
            ["measured 222.222 ns", "vendor timing closure", "measured power", "faster than prior FPGA decoders"],
            {"fmax_mhz": core_profile["summary"]["fmax_mhz"], "resources": core_resources, "clock_model_ns": core_profile["source_to_action_latency_model"]["at_enforced_27mhz_ns"], "power_sensitivity_mw": core_profile["dynamic_power_estimate"]["dynamic_power_mw_sensitivity"]},
            [_parent_evidence("hardware_pareto", ["profiles.route_a_core_no_student", "evidence_boundary"])],
            "T6.9.2 must provide same-bitstream layered physical measurements.",
            ["any P&R seed falls below target", "estimate labels removed", "harness path described as measured action path"],
        ),
        _claim(
            "BOARD_MEASURED_CORRECTNESS_LATENCY", "BLOCKED_ALL_FIELDS_NULL", "blocks_phase7_main_freeze",
            "No physical-board correctness, latency, deadline, jitter, resource-readback or power result is currently available.",
            ["zero board deadline misses", "measured source-to-action latency", "measured board power"],
            {"execution_branch": board["execution_branch"], "null_measured_fields": sum(value is None for value in board["measured_results"].values()), "physical_prerequisites_failed": sum(not row["passed"] for row in board["prerequisite_ledger"] if row["kind"] == "physical_external")},
            [_parent_evidence("board_blocker", ["prerequisite_ledger", "measured_results", "claim_boundary"])],
            "Complete all nine T6.9.2 recovery conditions and rerun the final evidence gate.",
            ["any P&R/model value copied into measured fields", "phase7 main freeze before physical evidence"],
        ),
        _claim(
            "FPGA_SPEED_ADVANTAGE", "PROHIBITED_NO_SAME_TASK_BOARD_COMPARATOR", "mandatory_limitation",
            "No FPGA speed advantage is supported because Route-A has no board measurement and the normalized literature table has zero same-task external comparator rows.",
            ["fastest", "SOTA FPGA decoder", "lower latency than existing FPGA decoder"],
            {"same_task_external_comparator_count": matrix["FPGA_SPEED_ADVANTAGE"]["current_result"]["same_task_external_comparator_count"], "board_measured": pareto["evidence_boundary"]["board_measured"], "speed_state": board["claim_boundary"]["fpga_speed_advantage"]},
            [_parent_evidence("claim_matrix", ["claims.FPGA_SPEED_ADVANTAGE"]), _parent_evidence("board_blocker", ["claim_boundary"])],
            "Requires T6.9.2 physical measurement and a T6.8.6 same-task comparable subset.",
            ["cross-code raw latency ranking", "mixed latency boundaries", "P&R estimate used as measurement"],
        ),
        _claim(
            "CNN_AND_HMM_ROLE", "CNN_ABLATION_HMM_SOFTWARE_ONLY", "ablation_or_supplement",
            "CNN/teacher/student remain optional learning ablations, and HMM posterior inference remains a software slow loop; MAP/FSM components own the primary deployable claim.",
            ["CNN-centric decoder", "HMM implemented on FPGA", "learning module causes primary LER gain"],
            {"cnn_state": matrix["CNN_PRIMARY_ROLE"]["state"], "hmm_is_in_rtl": rtl["hmm_is_in_rtl"], "selected_hardware_profile": pareto["pareto_decision"]["selected_profile"]},
            [_parent_evidence("promotion", ["claim_registry.CNN_PRIMARY", "scientific_results.rtl.hmm_is_in_rtl"]), _parent_evidence("hardware_pareto", ["pareto_decision"])],
            "A new matched schema/budget performance gate and functional integrated RTL are required for promotion.",
            ["CNN appears as unconditional title/abstract contribution", "HMM-on-FPGA wording without RTL"],
        ),
    ]


def _write_csv(claims: list[Mapping[str, Any]]) -> None:
    fields = ["claim_id", "final_state", "phase7_role", "allowed_wording", "forbidden_wording_json", "current_result_json", "evidence_json", "remaining_gate", "revocation_conditions_json"]
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in claims:
            writer.writerow({
                "claim_id": row["claim_id"], "final_state": row["final_state"], "phase7_role": row["phase7_role"],
                "allowed_wording": row["allowed_wording"],
                "forbidden_wording_json": json.dumps(row["forbidden_wording"], ensure_ascii=False, separators=(",", ":")),
                "current_result_json": json.dumps(row["current_result"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "evidence_json": json.dumps(row["evidence"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "remaining_gate": row["remaining_gate"],
                "revocation_conditions_json": json.dumps(row["revocation_conditions"], ensure_ascii=False, separators=(",", ":")),
            })


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    claims = {row["claim_id"]: row for row in report["claims"]}
    bindings = list(report["parent_bindings"].values()) + [report["source_data"], report["implementation_binding"]]
    smooth = claims["SMOOTH_LOCKED_EWMA_ADVANTAGE"]["current_result"]
    tail = claims["TAIL_SAFETY_AND_IMPROVEMENT"]["current_result"]
    board = claims["BOARD_MEASURED_CORRECTNESS_LATENCY"]["current_result"]
    return {
        "G01_all_parents_source_data_and_implementation_are_live_hash_bound": set(report["parent_bindings"]) == set(PARENTS) and all(len(row["sha256"]) == 64 for row in bindings) and (not check_live_files or all(_live(row) for row in bindings)),
        "G02_atomic_final_claim_schema_is_complete": set(claims) == FINAL_CLAIM_IDS and len(claims) == 11 and all(row["allowed_wording"] and row["forbidden_wording"] and row["current_result"] and row["evidence"] and row["remaining_gate"] and row["revocation_conditions"] for row in claims.values()),
        "G03_restricted_system_and_smooth_claims_remain_narrow": claims["CONTRACT_SYSTEM_INTEGRATION"]["final_state"] == "SUPPORTED_RESTRICTED_PREBOARD" and claims["SMOOTH_LOCKED_EWMA_ADVANTAGE"]["final_state"] == "SUPPORTED_PAIRED_OUTCOME" and smooth["primary_contrast"]["ci95_low"] > 0 and smooth["holm_confirmed_families"] == ["periodic_drift"] and smooth["route_a_is_global_best"] is False,
        "G04_static_superiority_remains_falsified": claims["STATIC_GKP_SUPERIORITY"]["final_state"] == "FALSIFIED" and claims["STATIC_GKP_SUPERIORITY"]["current_result"]["static_minus_route_a"]["ci95_high"] < 0,
        "G04b_static_k4_equivalence_remains_narrow_and_exhaustive": claims["STATIC_K4_HARD_ACTION_EQUIVALENCE"]["final_state"] == "SUPPORTED_PREBOARD_NARROW" and claims["STATIC_K4_HARD_ACTION_EQUIVALENCE"]["current_result"]["domain_points"] == 1_048_576 and claims["STATIC_K4_HARD_ACTION_EQUIVALENCE"]["current_result"]["hard_action_disagreements"] == 0,
        "G05_tail_is_safety_noninferiority_not_improvement": claims["TAIL_SAFETY_AND_IMPROVEMENT"]["final_state"] == "SAFETY_NONINFERIORITY_ONLY" and tail["tail_safety_gate_passes"] is True and tail["confirmed_average_improvement_families"] == [] and tail["broad_tail_improvement_confirmed"] is False,
        "G06_external_drift_outcome_preserves_performance_and_budget_failure": claims["GENERAL_DRIFT_EXTERNAL_COMPARISON"]["final_state"] == "PERFORMANCE_OUTCOME_BUDGET_FAIL" and claims["GENERAL_DRIFT_EXTERNAL_COMPARISON"]["current_result"]["paired_outcome"]["external_minus_route_a"]["ci95_low"] > 0 and claims["GENERAL_DRIFT_EXTERNAL_COMPARISON"]["current_result"]["paired_outcome"]["external_update_worst_us"] > claims["GENERAL_DRIFT_EXTERNAL_COMPARISON"]["current_result"]["paired_outcome"]["wallclock_cap_us"],
        "G07_puviani_surpass_remains_prohibited": claims["PUVIANI_NMF_SURPASS"]["final_state"] == "PROHIBITED_SOURCE_INCOMPLETE" and claims["PUVIANI_NMF_SURPASS"]["current_result"]["paper_exact_passed"] == 0 and claims["PUVIANI_NMF_SURPASS"]["current_result"]["matched_metric_non_null_count"] == 0,
        "G08_preboard_hardware_is_supported_without_measured_promotion": claims["FPGA_DETERMINISTIC_ARCHITECTURE"]["final_state"] == "SUPPORTED_PR_ESTIMATE" and claims["FPGA_DETERMINISTIC_ARCHITECTURE"]["current_result"]["fmax_mhz"]["minimum"] >= 27.0 and claims["FPGA_DETERMINISTIC_ARCHITECTURE"]["current_result"]["clock_model_ns"] == 222.22222222222223,
        "G09_board_and_speed_claims_fail_closed": claims["BOARD_MEASURED_CORRECTNESS_LATENCY"]["final_state"] == "BLOCKED_ALL_FIELDS_NULL" and board["null_measured_fields"] == 42 and board["physical_prerequisites_failed"] == 6 and claims["FPGA_SPEED_ADVANTAGE"]["final_state"] == "PROHIBITED_NO_SAME_TASK_BOARD_COMPARATOR" and claims["FPGA_SPEED_ADVANTAGE"]["current_result"]["same_task_external_comparator_count"] == 0,
        "G10_learning_roles_remain_ablation_and_software_only": claims["CNN_AND_HMM_ROLE"]["final_state"] == "CNN_ABLATION_HMM_SOFTWARE_ONLY" and claims["CNN_AND_HMM_ROLE"]["current_result"]["cnn_state"] == "ABLATION_ONLY" and claims["CNN_AND_HMM_ROLE"]["current_result"]["hmm_is_in_rtl"] is False,
        "G11_full_high_level_paper_is_no_go_and_phase7_freeze_is_closed": report["paper_decision"]["full_cross_lane_high_level_paper"] == "NO_GO" and report["paper_decision"]["phase7_main_figure_and_prose_freeze_allowed"] is False and set(report["paper_decision"]["blocking_claims"]) == {"STATIC_GKP_SUPERIORITY", "TAIL_SAFETY_AND_IMPROVEMENT", "GENERAL_DRIFT_EXTERNAL_COMPARISON", "PUVIANI_NMF_SURPASS", "BOARD_MEASURED_CORRECTNESS_LATENCY", "FPGA_SPEED_ADVANTAGE"},
        "G12_downgrade_routes_are_explicit_and_not_full_go": report["paper_decision"]["selected_downgrade"] == "RESTRICTED_PREBOARD_SYSTEM_DRAFT" and report["paper_decision"]["allowed_downgrade_components"] == ["locked_EWMA_smooth_paired_outcome", "tail_safety_noninferiority_with_cost_limitations", "deterministic_six_cycle_preboard_architecture", "static_negative_result", "external_BOCD_performance_outcome_with_budget_fail", "official_GQF_negative_reproduction_audit"],
        "G13_figure_table_plan_keeps_blocked_and_negative_panels_visible": len(report["figure_table_plan"]) == 7 and {row["status"] for row in report["figure_table_plan"]} >= {"READY_RESTRICTED", "MANDATORY_NEGATIVE", "BLOCKED"} and next(row for row in report["figure_table_plan"] if row["item_id"] == "HW_MEASURED")["status"] == "BLOCKED",
        "G14_no_global_score_cross_lane_arithmetic_or_sota_wording": report["aggregation_policy"] == {"global_score": "PROHIBITED", "cross_lane_compensation": "PROHIBITED", "missing_evidence_imputation": "PROHIBITED", "overall_sota_fastest_first_surpass": "PROHIBITED"},
        "G15_source_csv_has_all_final_claims": report["source_data"]["rows"] == 11 and set(_source_claim_ids(ROOT / report["source_data"]["path"])) == FINAL_CLAIM_IDS and len(_source_claim_ids(ROOT / report["source_data"]["path"])) == 11 and (not check_live_files or _sha256(ROOT / report["source_data"]["path"]) == report["source_data"]["sha256"]),
        "G16_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 17 and len(report["semantic_mutation_audit"]["cases"]) == 17 and all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]),
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 17, "detected": 17, "cases": [{"rejected": True}] * 17}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    def claim(x: Mapping[str, Any], claim_id: str) -> dict[str, Any]:
        return next(row for row in x["claims"] if row["claim_id"] == claim_id)

    attempt("forge_parent_hash", "G01_all_parents_source_data_and_implementation_are_live_hash_bound", lambda x: x["parent_bindings"]["promotion"].update(sha256="0"))
    attempt("drop_revocation", "G02_atomic_final_claim_schema_is_complete", lambda x: claim(x, "CONTRACT_SYSTEM_INTEGRATION").update(revocation_conditions=[]))
    attempt("claim_all_smooth_families", "G03_restricted_system_and_smooth_claims_remain_narrow", lambda x: claim(x, "SMOOTH_LOCKED_EWMA_ADVANTAGE")["current_result"].update(holm_confirmed_families=["all"]))
    attempt("promote_static", "G04_static_superiority_remains_falsified", lambda x: claim(x, "STATIC_GKP_SUPERIORITY").update(final_state="SUPPORTED"))
    attempt("weaken_k4_domain", "G04b_static_k4_equivalence_remains_narrow_and_exhaustive", lambda x: claim(x, "STATIC_K4_HARD_ACTION_EQUIVALENCE")["current_result"].update(domain_points=1024))
    attempt("promote_tail", "G05_tail_is_safety_noninferiority_not_improvement", lambda x: claim(x, "TAIL_SAFETY_AND_IMPROVEMENT").update(final_state="IMPROVED"))
    attempt("hide_budget_fail", "G06_external_drift_outcome_preserves_performance_and_budget_failure", lambda x: claim(x, "GENERAL_DRIFT_EXTERNAL_COMPARISON")["current_result"]["paired_outcome"].update(external_update_worst_us=1.0))
    attempt("promote_nmf", "G07_puviani_surpass_remains_prohibited", lambda x: claim(x, "PUVIANI_NMF_SURPASS").update(final_state="SURPASSED"))
    attempt("label_hardware_measured", "G08_preboard_hardware_is_supported_without_measured_promotion", lambda x: claim(x, "FPGA_DETERMINISTIC_ARCHITECTURE").update(final_state="MEASURED"))
    attempt("invent_board_result", "G09_board_and_speed_claims_fail_closed", lambda x: claim(x, "BOARD_MEASURED_CORRECTNESS_LATENCY").update(final_state="SUPPORTED"))
    attempt("promote_cnn", "G10_learning_roles_remain_ablation_and_software_only", lambda x: claim(x, "CNN_AND_HMM_ROLE").update(final_state="PRIMARY"))
    attempt("promote_full_paper", "G11_full_high_level_paper_is_no_go_and_phase7_freeze_is_closed", lambda x: x["paper_decision"].update(full_cross_lane_high_level_paper="GO", phase7_main_figure_and_prose_freeze_allowed=True))
    attempt("hide_downgrade", "G12_downgrade_routes_are_explicit_and_not_full_go", lambda x: x["paper_decision"].update(selected_downgrade="FULL_PAPER"))
    attempt("mark_hw_figure_ready", "G13_figure_table_plan_keeps_blocked_and_negative_panels_visible", lambda x: next(row for row in x["figure_table_plan"] if row["item_id"] == "HW_MEASURED").update(status="READY_RESTRICTED"))
    attempt("allow_global_score", "G14_no_global_score_cross_lane_arithmetic_or_sota_wording", lambda x: x["aggregation_policy"].update(global_score="ALLOWED"))
    attempt("forge_csv_rows", "G15_source_csv_has_all_final_claims", lambda x: x["source_data"].update(rows=10))
    attempt("forge_mutation_count", "G16_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 17, "detected": 16, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    parents = {key: _load(path) for key, path in PARENTS.items()}
    claims = _claims(parents)
    _write_csv(claims)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_bindings": {key: _binding(path) for key, path in PARENTS.items()},
        "aggregation_policy": {"global_score": "PROHIBITED", "cross_lane_compensation": "PROHIBITED", "missing_evidence_imputation": "PROHIBITED", "overall_sota_fastest_first_surpass": "PROHIBITED"},
        "claims": claims,
        "paper_decision": {
            "full_cross_lane_high_level_paper": "NO_GO",
            "phase7_main_figure_and_prose_freeze_allowed": False,
            "blocking_claims": ["STATIC_GKP_SUPERIORITY", "TAIL_SAFETY_AND_IMPROVEMENT", "GENERAL_DRIFT_EXTERNAL_COMPARISON", "PUVIANI_NMF_SURPASS", "BOARD_MEASURED_CORRECTNESS_LATENCY", "FPGA_SPEED_ADVANTAGE"],
            "selected_downgrade": "RESTRICTED_PREBOARD_SYSTEM_DRAFT",
            "allowed_downgrade_components": ["locked_EWMA_smooth_paired_outcome", "tail_safety_noninferiority_with_cost_limitations", "deterministic_six_cycle_preboard_architecture", "static_negative_result", "external_BOCD_performance_outcome_with_budget_fail", "official_GQF_negative_reproduction_audit"],
            "resume_condition": "complete T6.9.2 and rerun T6.9.3; performance claims additionally require new preregistered methods/experiments, not narrative changes",
        },
        "figure_table_plan": [
            {"item_id": "ARCH", "role": "contract-centric Route-A architecture", "status": "READY_RESTRICTED", "boundary": "student optional; HMM software; preboard"},
            {"item_id": "SMOOTH", "role": "locked-EWMA paired smooth result", "status": "READY_RESTRICTED", "boundary": "periodic only Holm-confirmed; static/Window visible"},
            {"item_id": "TAIL", "role": "abrupt/OOD safety and cost", "status": "MANDATORY_NEGATIVE", "boundary": "noninferiority; zero broad improvement; high fallback/false update"},
            {"item_id": "STATIC_EXTERNAL", "role": "static and BOCD comparisons", "status": "MANDATORY_NEGATIVE", "boundary": "static wins average; BOCD budget fails"},
            {"item_id": "GQF", "role": "official Puviani exact audit", "status": "MANDATORY_NEGATIVE", "boundary": "0/15 exact; no lifetime comparison"},
            {"item_id": "HW_PREBOARD", "role": "three-seed P&R/resource/Fmax", "status": "READY_RESTRICTED", "boundary": "estimate and analytic power sensitivity only"},
            {"item_id": "HW_MEASURED", "role": "board latency/deadline/power and same-task speed", "status": "BLOCKED", "boundary": "all measured fields null"},
        ],
        "source_data": {**_binding(SOURCE_CSV), "rows": len(claims)},
        "implementation_binding": _binding(Path(__file__)),
    }
    report["semantic_mutation_audit"] = {"count": 17, "detected": 17, "cases": [{"rejected": True}] * 17}
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": sum(not value for value in report["gates"].values())}
    report["verdict"] = "NO_GO_FULL_HIGH_LEVEL_PAPER_RESTRICTED_PREBOARD_DRAFT_ONLY" if report["gate_summary"]["failed"] == 0 else "FAIL_FINAL_EVIDENCE_INTEGRITY"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    summary = {"passed": sum(gates.values()), "failed": sum(not value for value in gates.values())}
    if report.get("gates") != gates or report.get("gate_summary") != summary or summary["failed"] != 0 or report.get("verdict") != "NO_GO_FULL_HIGH_LEVEL_PAPER_RESTRICTED_PREBOARD_DRAFT_ONLY":
        raise ValueError("T6.9.3 final evidence gates/verdict do not pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the T6.9.3 final Route-A evidence gate")
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
    print(json.dumps({"output": _relative(args.output), "verdict": report["verdict"], "gate_summary": report["gate_summary"], "claims": len(report["claims"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
