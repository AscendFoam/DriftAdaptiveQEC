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
TASK_ID = "T6.8.7"
SCHEMA_VERSION = "t6.8.7-route-a-claim-matrix-v1"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_8_7_route_a_claim_matrix.json"
SOURCE_CSV = ROOT / "docs" / "t6_8_7_route_a_claim_matrix_source_data.csv"

REPORT_PATHS = {
    "promotion": ROOT / "docs" / "t6_7_4_route_a_promotion_gate.json",
    "static": ROOT / "docs" / "t6_8_1_static_gkp_same_model_lane.json",
    "drift": ROOT / "docs" / "t6_8_2_external_drift_adaptive_lane.json",
    "gqf_exact": ROOT / "docs" / "t6_8_4_gqf_paper_exact_reproduction.json",
    "gqf_gate": ROOT / "docs" / "t6_8_5_gqf_route_a_matched_comparison_gate.json",
    "fpga": ROOT / "docs" / "t6_8_6_fpga_decoder_normalization.json",
}

OPPONENT_CLASSES = {"static_gkp", "general_drift_adaptive", "puviani_nmf", "fpga_qec_decoder"}
CLAIM_IDS = {
    "CONTRACT_SYSTEM_INTEGRATION",
    "SMOOTH_LOCKED_EWMA_ADVANTAGE",
    "STATIC_GKP_SUPERIORITY",
    "STATIC_K4_HARD_ACTION_EQUIVALENCE",
    "GENERAL_DRIFT_BOCD_OUTCOME",
    "GENERAL_DRIFT_MATCHED_BUDGET_SOTA",
    "PUVIANI_NMF_SURPASS",
    "FPGA_DETERMINISTIC_ARCHITECTURE",
    "FPGA_SPEED_ADVANTAGE",
    "CNN_PRIMARY_ROLE",
}
POSITIVE_STATES = {"SUPPORTED_RESTRICTED", "SUPPORTED_PAIRED_OUTCOME", "SUPPORTED_PREBOARD_NARROW"}
NEGATIVE_STATES = {"FALSIFIED", "NOT_ESTABLISHED", "PROHIBITED", "ABLATION_ONLY"}
FORBIDDEN_RANKING_TOKENS = ("sota", "fastest", "first", "surpass", "超过", "最快", "首个")


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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_binding(key: str, report: Mapping[str, Any]) -> dict[str, Any]:
    path = REPORT_PATHS[key]
    source = report.get("output_source_data_binding") or report.get("source_data")
    if not source and isinstance(report.get("bindings"), Mapping):
        source = report["bindings"].get("source_csv")
    return {
        "task_id": report.get("task_id"),
        "report": _binding(path),
        "source_data": source,
    }


def _evidence(
    *keys: str,
    reports: Mapping[str, Mapping[str, Any]],
    selectors: list[str],
    seed_contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "current_artifacts": [_report_binding(key, reports[key]) for key in keys],
        "selectors": selectors,
        "config": {
            "threshold_lock_sha256": seed_contract["threshold_lock_sha256"],
            "formal_baseline_reselection": seed_contract["formal_baseline_reselection"],
        },
        "seeds": seed_contract["seeds"],
        "t6_9_dependencies": [
            {
                "task_id": "T6.9.1",
                "status": "PENDING",
                "required_artifact": "docs/t6_9_1_route_a_hardware_pareto.json",
                "sha256": None,
                "role": "integrated three-seed P&R/Fmax/resource/power estimate",
            },
            {
                "task_id": "T6.9.2",
                "status": "PENDING_BOARD",
                "required_artifact": "docs/t6_9_2_route_a_board_measurement.json",
                "sha256": None,
                "role": "same-bitstream measured correctness/latency/deadline/resource/power",
            },
            {
                "task_id": "T6.9.3",
                "status": "PENDING",
                "required_artifact": "docs/t6_9_3_route_a_evidence_gate.json",
                "sha256": None,
                "role": "final claim-specific GO/NO-GO",
            },
        ],
    }


def _claim(
    claim_id: str,
    opponent_class: str,
    state: str,
    strongest_supported_wording: str,
    required_evidence: list[str],
    current_result: Mapping[str, Any],
    remaining_gaps: list[str],
    revocation_conditions: list[str],
    evidence: Mapping[str, Any],
    paper_target: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "opponent_class": opponent_class,
        "state": state,
        "strongest_supported_wording": strongest_supported_wording,
        "required_evidence": required_evidence,
        "current_result": dict(current_result),
        "remaining_gaps": remaining_gaps,
        "revocation_conditions": revocation_conditions,
        "evidence": dict(evidence),
        "paper_target": paper_target,
    }


def _build_claims(reports: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    promotion = reports["promotion"]
    static = reports["static"]
    drift = reports["drift"]
    gqf_exact = reports["gqf_exact"]
    gqf_gate = reports["gqf_gate"]
    fpga = reports["fpga"]
    smooth = promotion["scientific_results"]["smooth"]
    rtl = promotion["scientific_results"]["rtl"]
    seed_contract = {
        **promotion["frozen_contract"],
        "seeds": {
            "route_a_formal": "T6.7.1/T6.7.2 frozen formal seed manifests (hash-bound by T6.7.4 parents)",
            "external_drift_formal": drift["split_contract"]["formal_seeds"],
            "gqf_reduced_probe": gqf_exact["reduced_probe"]["configuration"]["seeds"],
        },
    }
    static_contrast = static["paired_static_contrast"]
    static_summary = {row["method_id"]: row for row in static["method_table"]}
    drift_claim = next(row for row in drift["claim_registry"] if row["claim_id"] == "EXTERNAL_BOCD_WRAPPER_PAIRED_OUTCOME")
    drift_summaries = {row["method_id"]: row for row in drift["formal_results"]["method_summaries"]}
    budget = drift["formal_results"]["external_budget"]
    exact = gqf_exact["exact_qualification"]
    fpga_boundary = fpga["claim_boundary"]

    claims = [
        _claim(
            "CONTRACT_SYSTEM_INTEGRATION", "general_drift_adaptive", "SUPPORTED_RESTRICTED",
            "Route-A is a contract-centric, regime-aware safety orchestration whose locked-EWMA smooth primary gate and one-million-cycle preboard fail-closed correctness gate pass; this is not a global LER, tail-improvement, or measured-hardware claim.",
            ["preregistered locked baseline", "smooth paired CI", "tail non-inferiority", "one-million-cycle integer/CXXRTL trace"],
            {"promotion_verdict": promotion["verdict"], "rtl_cycles": rtl["cycles"], "rtl_mismatches": rtl["rtl_mismatches"], "undefined_actions": rtl["undefined_actions"], "silent_overflow": rtl["silent_overflow"], "hmm_is_in_rtl": rtl["hmm_is_in_rtl"], "measured_board_latency": rtl["measured_board_latency"]},
            ["HMM posterior remains in the software slow loop", "T6.9.1 integrated P&R and T6.9.2 board evidence absent", "broad tail improvement is not established"],
            ["any frozen-contract hash drift", "any nonzero RTL mismatch/undefined action/silent overflow", "T6.9.3 final evidence NO-GO"],
            _evidence("promotion", reports=reports, selectors=["scientific_results.smooth", "scientific_results.tail", "scientific_results.rtl", "frozen_contract"], seed_contract=seed_contract),
            "Main system claim; Table Route-A evidence gate and architecture figure",
        ),
        _claim(
            "SMOOTH_LOCKED_EWMA_ADVANTAGE", "general_drift_adaptive", "SUPPORTED_PAIRED_OUTCOME",
            "On the preregistered smooth matrix, Route-A lowers aggregate paired LER relative to the locked EWMA baseline; only periodic drift is Holm-confirmed by family.",
            ["paired trace identity", "95% cluster-bootstrap CI lower bound above zero", "multiple-comparison family report"],
            {"ewma_minus_route_a": smooth["primary_contrast"], "holm_confirmed_families": smooth["holm_confirmed_families"], "all_smooth_families_holm_confirmed": smooth["all_smooth_families_holm_confirmed"]},
            ["not an advantage over static joint MAP or Window MAP", "only one smooth family is Holm-confirmed", "no physical-board result"],
            ["paired CI lower bound becomes non-positive under the frozen analysis", "seed/config/source hash mismatch", "wording expands beyond locked EWMA aggregate"],
            _evidence("promotion", reports=reports, selectors=["scientific_results.smooth.primary_contrast", "scientific_results.smooth.holm_confirmed_families"], seed_contract=seed_contract),
            "Primary smooth-drift panel with paired CI",
        ),
        _claim(
            "STATIC_GKP_SUPERIORITY", "static_gkp", "FALSIFIED",
            "No static-GKP superiority claim is supported: static joint MAP has lower average LER than Route-A on the frozen same-model smooth matrix.",
            ["same physical/statistical model", "same traces and decisions", "paired CI", "average and tail metrics"],
            {"static_minus_route_a": static_contrast, "static_average_ler": static_summary["static_joint_map"]["average_ler_equal_family_seed"], "route_a_average_ler": static_summary["proposed_route_a"]["average_ler_equal_family_seed"], "static_worst_window_ler": static_summary["static_joint_map"]["worst_window_ler"], "route_a_worst_window_ler": static_summary["proposed_route_a"]["worst_window_ler"]},
            ["requires a new method, not narrative relabeling", "must retain current negative result in the main comparison"],
            ["never promote without a new preregistered same-model paired experiment whose improvement CI excludes zero", "any omission of static/Window rows invalidates the claim table"],
            _evidence("promotion", "static", reports=reports, selectors=["paired_static_contrast", "method_table"], seed_contract=seed_contract),
            "Main comparison table as explicit negative result",
        ),
        _claim(
            "STATIC_K4_HARD_ACTION_EQUIVALENCE", "static_gkp", "SUPPORTED_PREBOARD_NARROW",
            "For the frozen covariance/prior and complete 10-bit syndrome domain, top-K=4 and full static MAP produce identical hard actions; retained-bit and serial-cost reductions are preboard proxies.",
            ["complete finite input-domain enumeration", "zero hard-action disagreement", "soft-error bounds", "synthesis/board confirmation before hardware claims"],
            {"domain_points": static["topk_full_exhaustive_equivalence"]["grid_points"], "hard_action_disagreements": static["topk_full_exhaustive_equivalence"]["hard_disagreements"], "retained_bits_k4": 512, "retained_bits_full": 3200},
            ["model-specific frozen covariance/prior only", "no integrated P&R or measured resource/latency result"],
            ["any nonzero disagreement on the declared domain", "changed covariance/prior without re-enumeration", "proxy presented as measured hardware"],
            _evidence("static", reports=reports, selectors=["topk_full_exhaustive_equivalence", "method_table.topk_k4_static_map.cost", "method_table.static_joint_map.cost"], seed_contract=seed_contract),
            "Compression/sensitivity ablation or supplement",
        ),
        _claim(
            "GENERAL_DRIFT_BOCD_OUTCOME", "general_drift_adaptive", "SUPPORTED_PAIRED_OUTCOME",
            "Against the pinned external BOCD-window-EWMA router on common formal traces, Route-A has lower paired LER; the external method fails the preregistered worst-update wall-clock budget and therefore is not a qualified matched-budget comparator.",
            ["pinned external implementation", "identical traces/truth", "paired CI", "strict worst-update compute and wall-clock caps"],
            {"external_minus_route_a": drift_claim["reason"], "route_a_equal_family_ler": drift_summaries["proposed_route_a"]["equal_family_seed_average_ler"], "external_equal_family_ler": drift_summaries["external_bocd_window_ewma_router"]["equal_family_seed_average_ler"], "external_update_worst_us": budget["update_wallclock_worst_us"], "wallclock_cap_us": budget["common_update_wallclock_cap_us"], "deadline_miss_count": budget["deadline_miss_count"]},
            ["one external algorithm is insufficient for general SOTA", "external comparator budget qualification failed", "Bhardwaj paper-exact path unavailable"],
            ["trace/truth mismatch", "paired CI lower bound becomes non-positive", "external wrapper ceases to be pinned/unmodified", "wording omits the budget failure"],
            _evidence("drift", reports=reports, selectors=["formal_results.paired_contrasts", "formal_results.external_budget", "external_upstream", "split_contract.formal_seeds"], seed_contract=seed_contract),
            "External reproducible baseline table with qualification column",
        ),
        _claim(
            "GENERAL_DRIFT_MATCHED_BUDGET_SOTA", "general_drift_adaptive", "NOT_ESTABLISHED",
            "General drift-adaptive SOTA is not established because the only executable external comparator violates the strict worst-update wall-clock ceiling and other literature mappings are not exact same-task reproductions.",
            ["at least two closest external methods", "all comparators pass common worst-case compute/wall-clock budget", "same-task paired statistics"],
            {"qualified_external_comparator_count": 0, "external_updates": budget["update_count"], "external_deadline_misses": budget["deadline_miss_count"], "worst_update_us": budget["update_wallclock_worst_us"], "cap_us": budget["common_update_wallclock_cap_us"]},
            ["second close external implementation missing", "zero budget-qualified external comparators"],
            ["cannot become positive from a performance-only rerun", "requires a newly preregistered and budget-qualified external lane"],
            _evidence("drift", reports=reports, selectors=["claim_registry.GENERAL_DRIFT_ADAPTIVE_SOTA", "formal_results.external_budget"], seed_contract=seed_contract),
            "Claim registry/limitations, not a positive headline",
        ),
        _claim(
            "PUVIANI_NMF_SURPASS", "puviani_nmf", "PROHIBITED",
            "No comparison or surpass claim against Puviani NMF is permitted: the official-source paper-exact gate passes 0/15 criteria and the matched lifetime branch is ineligible, so all 13 comparison metrics remain null.",
            ["official checkpoints/seeds/evaluator", "paper-exact Standard/MF/NMF reproduction", "same-GQF matched lifetime", "paired lifetime-improvement 95% lower bound above zero"],
            {"paper_exact_passed": exact["passed"], "paper_exact_failed": exact["failed"], "exact_status": gqf_exact["exact_reproduction_status"], "matched_branch": gqf_gate["execution_branch"], "matched_metric_non_null_count": sum(value is not None for value in gqf_gate["matched_comparison_metrics"].values())},
            ["official checkpoints, seeds, six-state evaluator and raw trajectories absent", "paper/source architecture and training settings conflict", "full GPU accelerator path unqualified"],
            ["remains prohibited until all T6.8.4 exact gates and T6.8.5 eligibility gates pass", "directional reduced diagnostic may never substitute for exact reproduction"],
            _evidence("gqf_exact", "gqf_gate", reports=reports, selectors=["exact_qualification", "source_discrepancies", "prerequisite_ledger", "matched_comparison_metrics"], seed_contract=seed_contract),
            "Limitations/negative reproduction result; no lifetime ranking figure",
        ),
        _claim(
            "FPGA_DETERMINISTIC_ARCHITECTURE", "fpga_qec_decoder", "SUPPORTED_PREBOARD_NARROW",
            "The integrated Route-A fast path is a deterministic six-cycle preboard architecture with one-million-cycle bit-exact CXXRTL qualification; nanosecond latency, resources, power and board deadline behavior remain unmeasured.",
            ["synthesizable integrated top", "long-sequence bit-exact replay", "P&R for time/resources", "board measurement for physical claims"],
            {"latency_cycles": 6, "qualified_cycles": rtl["cycles"], "rtl_mismatches": rtl["rtl_mismatches"], "undefined_actions": rtl["undefined_actions"], "silent_overflow": rtl["silent_overflow"], "measured_board_latency": rtl["measured_board_latency"]},
            ["T6.9.1 integrated P&R absent", "T6.9.2 board latency/deadline/resource/power absent", "HMM is not in RTL"],
            ["any nonzero bit mismatch/undefined action/silent overflow", "integrated P&R does not preserve the declared cycle contract", "preboard evidence described as measured"],
            _evidence("promotion", "fpga", reports=reports, selectors=["scientific_results.rtl", "rows.project_t6_route_a_integrated_cxxrtl"], seed_contract=seed_contract),
            "Hardware architecture panel and preboard qualification table",
        ),
        _claim(
            "FPGA_SPEED_ADVANTAGE", "fpga_qec_decoder", "PROHIBITED",
            "No speed ranking against prior FPGA QEC decoders is supported: there are zero same-task external comparator rows and no Route-A board source-to-action measurement.",
            ["same code/problem/input/action semantics", "same latency boundary/statistic", "physical-board source-to-action distributions", "deadline, II, resources and power"],
            {"same_task_external_comparator_count": fpga_boundary["same_task_external_comparator_count"], "real_board_source_to_action": fpga_boundary["real_board_source_to_action"], "speed_advantage": fpga_boundary["fpga_speed_advantage"]},
            ["no same-task external comparator", "no integrated board measurement", "cross-code raw latency is non-comparable"],
            ["only T6.9.2 same-task measured evidence may unlock this claim", "any cross-code or mixed-boundary ranking forces immediate withdrawal"],
            _evidence("fpga", reports=reports, selectors=["comparison_eligibility", "claim_boundary", "rows"], seed_contract=seed_contract),
            "Normalization table and limitations; no fastest headline",
        ),
        _claim(
            "CNN_PRIMARY_ROLE", "general_drift_adaptive", "ABLATION_ONLY",
            "Legacy CNN residual and teacher/student modules remain optional learning ablations; the primary deployable LER path is MAP-based and the safety path is contract/FSM-based.",
            ["matched input/output schema", "common compute/wall-clock budget", "paired LER and tail evidence"],
            {"promotion_state": next(row["state"] for row in promotion["claim_registry"] if row["claim_id"] == "CNN_PRIMARY")},
            ["legacy checkpoint fails the matched schema/budget", "no primary claim may depend on CNN"],
            ["promoting CNN in title/abstract without a new matched gate", "removing MAP/FSM attribution from the system claim"],
            _evidence("promotion", reports=reports, selectors=["claim_registry.CNN_PRIMARY"], seed_contract=seed_contract),
            "Ablation table or supplement only",
        ),
    ]
    return claims


def _binding_is_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.is_file() and _sha256(path) == binding["sha256"] and path.stat().st_size == binding["bytes"]


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    claims = report["claims"]
    by_id = {row["claim_id"]: row for row in claims}
    fields = {"claim_id", "opponent_class", "state", "strongest_supported_wording", "required_evidence", "current_result", "remaining_gaps", "revocation_conditions", "evidence", "paper_target"}
    positive_wording = " ".join(row["strongest_supported_wording"].lower() for row in claims if row["state"] in POSITIVE_STATES)
    all_bindings = [item["report"] for row in claims for item in row["evidence"]["current_artifacts"]]
    unique_bindings = {(item["path"], item["sha256"], item["bytes"]): item for item in all_bindings}.values()
    t69 = [dep for row in claims for dep in row["evidence"]["t6_9_dependencies"]]
    source = report["source_data"]
    return {
        "G01_four_opponent_classes_are_explicit": {row["opponent_class"] for row in claims} == OPPONENT_CLASSES,
        "G02_atomic_claim_schema_and_ids_are_complete": len(claims) == len(CLAIM_IDS) and set(by_id) == CLAIM_IDS and all(set(row) == fields for row in claims),
        "G03_every_claim_has_required_result_gap_and_revocation": all(row["required_evidence"] and row["current_result"] and row["remaining_gaps"] and row["revocation_conditions"] and row["paper_target"] for row in claims),
        "G04_positive_wording_contains_no_unqualified_rank_tokens": not any(token in positive_wording for token in FORBIDDEN_RANKING_TOKENS),
        "G05_static_superiority_remains_falsified": by_id["STATIC_GKP_SUPERIORITY"]["state"] == "FALSIFIED" and by_id["STATIC_GKP_SUPERIORITY"]["current_result"]["static_minus_route_a"]["ci95_high"] < 0.0,
        "G06_general_drift_budget_and_sota_fail_closed": by_id["GENERAL_DRIFT_MATCHED_BUDGET_SOTA"]["state"] == "NOT_ESTABLISHED" and by_id["GENERAL_DRIFT_MATCHED_BUDGET_SOTA"]["current_result"]["qualified_external_comparator_count"] == 0 and by_id["GENERAL_DRIFT_MATCHED_BUDGET_SOTA"]["current_result"]["worst_update_us"] > by_id["GENERAL_DRIFT_MATCHED_BUDGET_SOTA"]["current_result"]["cap_us"],
        "G07_puviani_surpass_is_prohibited_with_null_metrics": by_id["PUVIANI_NMF_SURPASS"]["state"] == "PROHIBITED" and by_id["PUVIANI_NMF_SURPASS"]["current_result"]["paper_exact_passed"] == 0 and by_id["PUVIANI_NMF_SURPASS"]["current_result"]["matched_metric_non_null_count"] == 0,
        "G08_fpga_speed_is_prohibited_until_same_task_board_evidence": by_id["FPGA_SPEED_ADVANTAGE"]["state"] == "PROHIBITED" and by_id["FPGA_SPEED_ADVANTAGE"]["current_result"]["same_task_external_comparator_count"] == 0 and by_id["FPGA_SPEED_ADVANTAGE"]["current_result"]["real_board_source_to_action"] == "PENDING_T6.9.2",
        "G09_positive_claims_are_scope_limited": all(row["state"] in POSITIVE_STATES | NEGATIVE_STATES for row in claims) and "global ler advantage" not in positive_wording and "measured hardware advantage" not in positive_wording,
        "G10_current_reports_and_source_data_are_hash_bound": all(len(item["sha256"]) == 64 for item in unique_bindings) and (not check_live_files or all(_binding_is_live(item) for item in unique_bindings)),
        "G11_every_claim_binds_config_seed_selectors_and_t6_9_nulls": all(row["evidence"]["current_artifacts"] and row["evidence"]["selectors"] and len(row["evidence"]["config"]["threshold_lock_sha256"]) == 64 and row["evidence"]["seeds"] and len(row["evidence"]["t6_9_dependencies"]) == 3 for row in claims) and all(dep["status"].startswith("PENDING") and dep["sha256"] is None for dep in t69),
        "G12_no_global_score_or_cross_lane_arithmetic": report["aggregation_policy"] == {"global_score": "PROHIBITED", "cross_simulator_lifetime_subtraction": "PROHIBITED", "cross_code_raw_latency_ranking": "PROHIBITED", "per_claim_evidence_only": True},
        "G13_source_csv_is_complete_and_hash_bound": source["rows"] == len(claims) and len(source["sha256"]) == 64 and (not check_live_files or _sha256(ROOT / source["path"]) == source["sha256"]),
        "G14_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 14,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 14, "detected": 14, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    def row(x: Mapping[str, Any], claim_id: str) -> dict[str, Any]:
        return next(item for item in x["claims"] if item["claim_id"] == claim_id)

    attempt("drop_opponent_class", "G01_four_opponent_classes_are_explicit", lambda x: [item.update(opponent_class="static_gkp") for item in x["claims"]])
    attempt("duplicate_claim_id", "G02_atomic_claim_schema_and_ids_are_complete", lambda x: x["claims"][0].update(claim_id=x["claims"][1]["claim_id"]))
    attempt("erase_revocation", "G03_every_claim_has_required_result_gap_and_revocation", lambda x: x["claims"][0].update(revocation_conditions=[]))
    attempt("insert_fastest", "G04_positive_wording_contains_no_unqualified_rank_tokens", lambda x: row(x, "CONTRACT_SYSTEM_INTEGRATION").update(strongest_supported_wording="fastest FPGA decoder"))
    attempt("promote_static", "G05_static_superiority_remains_falsified", lambda x: row(x, "STATIC_GKP_SUPERIORITY").update(state="SUPPORTED_RESTRICTED"))
    attempt("hide_budget_failure", "G06_general_drift_budget_and_sota_fail_closed", lambda x: row(x, "GENERAL_DRIFT_MATCHED_BUDGET_SOTA")["current_result"].update(qualified_external_comparator_count=1))
    attempt("promote_puviani", "G07_puviani_surpass_is_prohibited_with_null_metrics", lambda x: row(x, "PUVIANI_NMF_SURPASS").update(state="SUPPORTED_PAIRED_OUTCOME"))
    attempt("promote_fpga_speed", "G08_fpga_speed_is_prohibited_until_same_task_board_evidence", lambda x: row(x, "FPGA_SPEED_ADVANTAGE").update(state="SUPPORTED_RESTRICTED"))
    attempt("claim_global", "G09_positive_claims_are_scope_limited", lambda x: row(x, "SMOOTH_LOCKED_EWMA_ADVANTAGE").update(strongest_supported_wording="global LER advantage"))
    attempt("forge_report_hash", "G10_current_reports_and_source_data_are_hash_bound", lambda x: row(x, "STATIC_GKP_SUPERIORITY")["evidence"]["current_artifacts"][0]["report"].update(sha256="0"))
    attempt("invent_t69_hash", "G11_every_claim_binds_config_seed_selectors_and_t6_9_nulls", lambda x: row(x, "FPGA_SPEED_ADVANTAGE")["evidence"]["t6_9_dependencies"][1].update(status="DONE", sha256="a" * 64))
    attempt("allow_global_score", "G12_no_global_score_or_cross_lane_arithmetic", lambda x: x["aggregation_policy"].update(global_score="ALLOWED"))
    attempt("forge_csv_rows", "G13_source_csv_is_complete_and_hash_bound", lambda x: x["source_data"].update(rows=9))
    attempt("forge_mutation_count", "G14_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 14, "detected": 13, "cases": []}))
    return {"count": len(cases), "detected": sum(case["rejected"] for case in cases), "cases": cases}


def _write_csv(claims: list[Mapping[str, Any]]) -> None:
    fieldnames = ["claim_id", "opponent_class", "state", "strongest_supported_wording", "required_evidence_json", "current_result_json", "remaining_gaps_json", "revocation_conditions_json", "evidence_json", "paper_target"]
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for claim in claims:
            writer.writerow({
                "claim_id": claim["claim_id"],
                "opponent_class": claim["opponent_class"],
                "state": claim["state"],
                "strongest_supported_wording": claim["strongest_supported_wording"],
                "required_evidence_json": json.dumps(claim["required_evidence"], ensure_ascii=False, separators=(",", ":")),
                "current_result_json": json.dumps(claim["current_result"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "remaining_gaps_json": json.dumps(claim["remaining_gaps"], ensure_ascii=False, separators=(",", ":")),
                "revocation_conditions_json": json.dumps(claim["revocation_conditions"], ensure_ascii=False, separators=(",", ":")),
                "evidence_json": json.dumps(claim["evidence"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "paper_target": claim["paper_target"],
            })


def build_report() -> dict[str, Any]:
    reports = {key: _load(path) for key, path in REPORT_PATHS.items()}
    claims = _build_claims(reports)
    _write_csv(claims)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Atomic innovation/advantage claim matrix across static GKP, general drift-adaptive decoders, Puviani NMF and FPGA QEC decoders.",
        "aggregation_policy": {"global_score": "PROHIBITED", "cross_simulator_lifetime_subtraction": "PROHIBITED", "cross_code_raw_latency_ranking": "PROHIBITED", "per_claim_evidence_only": True},
        "state_legend": {"positive": sorted(POSITIVE_STATES), "negative_or_limited": sorted(NEGATIVE_STATES)},
        "claims": claims,
        "source_data": {**_binding(SOURCE_CSV), "rows": len(claims)},
        "implementation_binding": _binding(Path(__file__)),
    }
    report["semantic_mutation_audit"] = {"count": 14, "detected": 14, "cases": []}
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": sum(not value for value in report["gates"].values())}
    report["verdict"] = "PASS_ROUTE_A_ATOMIC_CLAIM_MATRIX_WITH_RESTRICTED_POSITIVE_CLAIMS" if report["gate_summary"]["failed"] == 0 else "FAIL_ROUTE_A_CLAIM_MATRIX"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    expected = {"passed": sum(gates.values()), "failed": sum(not value for value in gates.values())}
    if report.get("gates") != gates or report.get("gate_summary") != expected or expected["failed"] != 0 or report.get("verdict") != "PASS_ROUTE_A_ATOMIC_CLAIM_MATRIX_WITH_RESTRICTED_POSITIVE_CLAIMS":
        raise ValueError("T6.8.7 claim-matrix gates/verdict do not pass")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify the T6.8.7 atomic Route-A claim matrix")
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
