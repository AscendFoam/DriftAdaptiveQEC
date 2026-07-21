"""Build the T7.3.4 post-selection/break-even reviewer response contract."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.3.4"
SCHEMA_VERSION = "t7.3.4-postselection-breakeven-reviewer-contract-v1"
VERDICT = "PASS_POSTSELECTION_DIAGNOSTIC_AND_BREAKEVEN_NOT_ESTABLISHED"

CONFIG = ROOT / "configs/phase6d/t7_3_4_postselection_breakeven_reviewer_contract.json"
BOARD = ROOT / "docs/new_task_board.md"
RISKS = ROOT / "docs/new_risks.md"
EXPERIMENT_PLAN = ROOT / "docs/experiment_plan.md"
MANUSCRIPT = ROOT / "docs/paper_notes/Phase6D_Dual_Lane_GKP_manuscript.tex"
MANUSCRIPT_CONTRACT = ROOT / "docs/t7_2_6_phase6d_manuscript_delta.json"
POSTSELECTION = ROOT / "docs/t3_2_4_postselection_validation.json"
LOGICAL_CHANNEL = ROOT / "docs/t5_3_1_logical_channel_reconstruction.json"
FIDELITY = ROOT / "docs/t5_3_2_logical_channel_fidelity.json"
OPERATIONAL = ROOT / "docs/t5_3_3_logical_operational_boundary.json"
COST = ROOT / "docs/t5_3_4_qec_postselection_cost.json"
HEADROOM = ROOT / "docs/t6_20_4_multimode_causal_headroom.json"
FINAL_GATE = ROOT / "docs/t6_26_4_final_dual_lane_gate.json"
POSTERIOR_SOURCE = ROOT / "cnn_fpga/decoder/route_a_regime_posterior.py"
SOURCE_REGISTRY = ROOT / "configs/literature/t6_16_1_secondary_method_sources.json"

DEFAULT_REPORT = ROOT / "docs/t7_3_4_postselection_breakeven_reviewer_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_3_4_postselection_breakeven_reviewer_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/postselection_breakeven_reviewer_response.md"

RESPONSE_STATES = {
    "REVIEWER_CONCERN", "PRIMARY_DENOMINATOR", "OFFLINE_DIAGNOSTIC", "REJECTION_COST",
    "COST_GAP", "OPERATIONAL_BOUNDARY", "COHERENCE_GAIN", "PHYSICAL_BREAK_EVEN",
    "NONTRANSFER", "MANUSCRIPT_CHANGE", "FUTURE_PROMOTION", "RISK_DISCLOSURE", "RESPONSE_WORDING",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path":path.relative_to(ROOT).as_posix(), "sha256":_sha256(path), "bytes":path.stat().st_size}


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.exists() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


def _task_status(board: str, task_id: str) -> str:
    match = re.search(rf"^\|\s*{re.escape(task_id)}\s*\|\s*([^|]+?)\s*\|", board, re.MULTILINE)
    if not match:
        raise ValueError(f"task status not found: {task_id}")
    return match.group(1).strip()


def _response_text() -> str:
    return (
        "We agree that post-selection and break-even are easy to overstate, and we have separated three distinct quantities. "
        "First, no Phase-6D primary metric uses post-selection: all 79,872 registered rounds and all 13 drift families remain in the denominator, and the proposed decoder retains its zero-improvement NO-GO result against static-mixture exact MLD.\n\n"
        "Second, the historical post-selection result is an offline diagnostic, not online correction. Its threshold is fitted on 294,912 training samples and evaluated on 1,572,864 disjoint samples using an observed static-MAP confidence score. "
        "At the 90% target, conditional decision error decreases from 0.013785 to 0.001242 at 0.899108 acceptance. However, accepted failures plus a unit rejection penalty gives total cost 0.102009, compared with raw error 0.013785. All eight targets improve conditional error and all eight become worse at unit rejection cost. "
        "The diagnostic is therefore reported with acceptance, rejection and cost and is ineligible for the primary LER or break-even claim.\n\n"
        "Third, the finite-model result is only a 300-us wall-clock operational boundary: a sustained/cumulative crossover of leakage-inclusive CPTNI average-fidelity curves against matched encoded idle. It uses neither an exponential fit nor a lifetime ratio. "
        "The low-cutoff counterexample is retained, the active short-time rate is unqualified, matched idle is not the best passive physical-qubit encoding, and twelve physical/control cost fields remain null. Consequently, paper-defined simulation-derived coherence gain, full-cost break-even and experimental break-even are all NOT_ESTABLISHED, with no reported gain value.\n\n"
        "Sivak's 2.27±0.07 coherence gain is a literature-reported physical-system result under its own device, best-passive denominator and fitted lifetime protocol. It cannot be transferred to our simulator or RTL. "
        "The manuscript therefore uses the fully qualified term finite-cutoff wall-clock operational boundary only for the historical result and explicitly states that the current work contains no measured logical lifetime or physical break-even result."
    )


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {"generated_at_utc", "analysis_sha256", "semantic_mutation_audit", "source_data", "markdown"}
    return {key:value for key,value in report.items() if key not in omitted}


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{
        "row_id":row["row_id"], "response_state":row["response_state"], "topic":row["topic"],
        "claim":row["claim"], "boundary":row["boundary"], "source_ids_json":_canonical(row["source_ids"]),
        "row_sha256":_canonical_sha256(row),
    } for row in report["response_rows"]]


def _source_data_matches(report: Mapping[str, Any], path: Path = DEFAULT_SOURCE_DATA) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream)) == _source_rows(report)


def build_report(*, generated_at_utc: str | None = None) -> dict[str, Any]:
    config = _load(CONFIG)
    board_text = BOARD.read_text(encoding="utf-8")
    risks_text = RISKS.read_text(encoding="utf-8")
    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    posterior_text = POSTERIOR_SOURCE.read_text(encoding="utf-8")
    post = _load(POSTSELECTION)
    channel = _load(LOGICAL_CHANNEL)
    fidelity = _load(FIDELITY)
    operational = _load(OPERATIONAL)
    cost = _load(COST)
    headroom = _load(HEADROOM)
    final_gate = _load(FINAL_GATE)
    registry = _load(SOURCE_REGISTRY)

    target90 = next(row for row in post["aggregate_by_target_survival"] if row["target_survival"] == 0.9)
    cost90 = next(row for row in cost["postselection_cost_rows"] if row["target_survival"] == 0.9)
    sivak = next(method for method in registry["methods"] if method["method_id"] == "sivak2023_rl_controller")
    sivak_gain = next(metric for metric in sivak["metrics"] if metric["metric_id"] == "break_even_gain")
    lane_events = [lane["event_accounting"] for lane in channel["lanes"].values()]
    boundary_verdict = operational["verdict"]
    cost_verdict = cost["verdict"]
    required_markers = config["required_manuscript_markers"]

    artifact_paths = {
        "implementation":Path(__file__).resolve(), "config":CONFIG, "task_board":BOARD, "new_risks":RISKS,
        "experiment_plan":EXPERIMENT_PLAN, "manuscript":MANUSCRIPT, "manuscript_contract":MANUSCRIPT_CONTRACT,
        "postselection":POSTSELECTION, "logical_channel":LOGICAL_CHANNEL, "fidelity":FIDELITY,
        "operational_boundary":OPERATIONAL, "cost":COST, "headroom":HEADROOM, "final_gate":FINAL_GATE,
        "posterior_source":POSTERIOR_SOURCE, "source_registry":SOURCE_REGISTRY,
    }

    report: dict[str, Any] = {
        "task_id":TASK_ID,
        "schema_version":SCHEMA_VERSION,
        "generated_at_utc":generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "reviewer_context":config["reviewer_context"],
        "task_status":{task:_task_status(board_text,task) for task in ("T7.3.4","T7.3.5")},
        "current_phase6d_primary":{
            "families":headroom["scope"]["families"], "physical_rounds":headroom["scope"]["physical_rounds"],
            "all_registered_families":next(gate["passed"] for gate in headroom["gates"] if gate["gate"] == "all_registered_families"),
            "postselection_used":False, "hard_ood_rows_retained":"selectively discard the hardest/OOD evidence" in posterior_text,
            "verdict":headroom["verdict"], "relative_improvement":headroom["paired_bootstrap"]["relative_improvement_point"],
            "final_verdict":final_gate["verdict"],
        },
        "historical_logical_channel":{
            "lanes":len(lane_events), "source_rows":channel["source_data"]["row_count"],
            "postselected_trajectories":sum(row["postselected_trajectories"] for row in lane_events),
            "discarded_trajectories":sum(row["discarded_trajectories"] for row in lane_events),
            "primary_metric":"leakage-inclusive CPTNI average fidelity",
            "conditional_postselected_forbidden":"conditional postselected channel fidelity" in fidelity["claim_boundary"]["forbidden"],
        },
        "offline_postselection":{
            "status":post["status"], "online_decoder":post["descriptor"]["online_decoder"],
            "primary_metric_eligible":post["descriptor"]["primary_metric_eligible"],
            "observed_inputs":post["descriptor"]["observed_score_inputs"], "hidden_truth_inputs":post["descriptor"]["hidden_truth_score_inputs"],
            "training_samples":post["training_calibration"]["training_samples"], "evaluation_samples":post["aggregate"]["evaluation_samples"],
            "target90":target90,
        },
        "rejection_cost":{
            "target90":cost90,
            "targets":len(cost["postselection_cost_rows"]),
            "targets_lower_conditional":cost_verdict["postselection_targets_with_lower_conditional_error"],
            "targets_worse_unit_penalty":cost_verdict["postselection_targets_worse_at_unit_rejection_penalty"],
            "postselection_joined_to_qec":cost["cost_contract"]["postselection_joined_to_qec"],
            "global_cost_score":cost["cost_contract"]["global_cost_score"],
            "cross_lane_total":cost["cost_contract"]["cross_lane_total"],
            "missing_fields":cost["missing_cost_fields"],
        },
        "break_even_taxonomy":{
            "wall_clock_operational_boundary":boundary_verdict["wall_clock_operational_boundary"],
            "terminal_profiles":boundary_verdict["terminal_cutoff_profiles_qualified"],
            "low_cutoff_counterexample_retained":boundary_verdict["low_cutoff_counterexample_retained"],
            "baseline":operational["boundary_contract"]["baseline"], "fit":operational["boundary_contract"]["fit"], "ratio":operational["boundary_contract"]["ratio"],
            "full_cost_operational_boundary":cost_verdict["full_cost_operational_boundary"],
            "simulation_derived_coherence_gain":cost_verdict["paper_defined_coherence_gain"],
            "coherence_gain_value":boundary_verdict["coherence_gain_value"],
            "postselected_break_even":cost_verdict["postselected_break_even"],
            "experimental_break_even":boundary_verdict["experimental_break_even"],
            "sivak_literature":{"value":sivak_gain["value"],"uncertainty":sivak_gain["uncertainty"],"evidence_grade":sivak_gain["evidence_grade"],"ranking_eligible":sivak_gain["ranking_eligible"],"same_task":sivak["same_task_with_project"]},
        },
        "nontransfer_contract":{
            "conditional_error_to_primary_ler":False, "operational_boundary_to_coherence_gain":False,
            "matched_idle_to_best_passive":False, "sivak_to_project_break_even":False,
            "missing_cost_to_zero":False, "software_or_rtl_to_physical_break_even":False,
        },
        "manuscript_audit":{
            "required_markers":{marker:marker in re.sub(r"\s+"," ",manuscript_text) for marker in required_markers},
            "forbidden_phrases":{phrase:phrase.lower() in manuscript_text.lower() for phrase in ("we demonstrate physical break-even","postselected break-even","simulation-derived coherence gain of")},
            "named_locations":["Related Work: physical lifetime", "Limitations", "Conclusion"],
        },
        "response_package":{
            "strategy":{"overall_posture":"accept metric ambiguity; separate primary denominator, offline diagnostic, finite-model boundary and physical break-even","major_risks":["accepted-only denominator","unpriced rejection","finite-model-to-physical rename","literature transfer"],"suggested_order":["primary denominator","diagnostic and cost","operational boundary","not-established claims","future gates"]},
            "tracker":{"comment_id":config["reviewer_context"]["comment_id"],"concern":config["reviewer_context"]["reviewer_concern"],"severity":config["reviewer_context"]["severity"],"actions":config["reviewer_context"]["actions"],"manuscript_locations":["Methods: denominators", "Results: historical diagnostic", "Limitations", "Conclusion"],"missing_author_input":config["reviewer_context"]["visible_placeholder"]},
            "english_response":_response_text(),
            "manuscript_change_checklist":["State that every Phase-6D round remains in the primary denominator.","Keep post-selection diagnostic, acceptance and rejection cost in one paragraph/table.","Use the full finite-cutoff wall-clock operational-boundary qualifier.","Keep coherence gain, full-cost and physical break-even NOT_ESTABLISHED.","Attribute Sivak 2.27±0.07 only to the cited physical system."],
            "missing_information":[config["reviewer_context"]["visible_placeholder"]], "package_readiness":config["reviewer_context"]["package_readiness"],
        },
        "response_rows":config["response_rows"], "forbidden_response_phrases":config["forbidden_response_phrases"],
        "artifact_registry":{key:_binding(path) for key,path in artifact_paths.items()},
        "risk_audit":{"r_n161_present":"R-N161" in risks_text,"t7_3_4_audit_present":"| 2026-07-21 | T7.3.4 |" in risks_text},
    }
    report["gates"] = evaluate_gates(report, check_live_sources=True)
    report["gate_summary"] = {"passed":sum(report["gates"].values()),"total":len(report["gates"])}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_POSTSELECTION_BREAKEVEN_CONTRACT"
    report["semantic_mutation_audit"] = _run_mutations(report)
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    primary = report["current_phase6d_primary"]
    channel = report["historical_logical_channel"]
    post = report["offline_postselection"]
    rejection = report["rejection_cost"]
    taxonomy = report["break_even_taxonomy"]
    rows = report["response_rows"]
    text = report["response_package"]["english_response"].lower()
    return {
        "G01_identity_and_lifecycle":report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION and report["task_status"]["T7.3.4"] == "Done" and report["task_status"]["T7.3.5"] in {"In Progress", "Done"},
        "G02_preemptive_placeholder_honest":report["reviewer_context"]["comment_id"] == "PRQ-BE-1" and report["response_package"]["package_readiness"] == "draft_with_placeholders" and report["response_package"]["missing_information"] == ["ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING"],
        "G03_phase6d_primary_keeps_full_denominator":primary["families"] == 13 and primary["physical_rounds"] == 79872 and primary["all_registered_families"] and primary["postselection_used"] is False and primary["hard_ood_rows_retained"],
        "G04_phase6d_negative_verdict_not_rescued":primary["verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM" and primary["relative_improvement"] == 0.0 and primary["final_verdict"] == "GO_RTL_ONLY",
        "G05_logical_channel_is_unconditional_cptni":channel == {"lanes":24,"source_rows":17266,"postselected_trajectories":0,"discarded_trajectories":0,"primary_metric":"leakage-inclusive CPTNI average fidelity","conditional_postselected_forbidden":True},
        "G06_postselection_is_offline_observed_and_ineligible":post["status"] == "PASS" and post["online_decoder"] is False and post["primary_metric_eligible"] is False and post["observed_inputs"] == ["static_map_logical_posterior"] and post["hidden_truth_inputs"] == [],
        "G07_postselection_scale_and_disjoint_evaluation_visible":post["training_samples"] == 294912 and post["evaluation_samples"] == 1572864,
        "G08_target90_conditional_gain_reports_acceptance":abs(post["target90"]["raw_error_rate"]-0.013785044352213543)<1e-15 and abs(post["target90"]["conditional_error_rate"]-0.0012424775303850107)<1e-15 and abs(post["target90"]["realized_survival_fraction"]-0.8991082509358723)<1e-15,
        "G09_unit_rejection_penalty_reverses_apparent_gain":abs(rejection["target90"]["total_cost_by_rejection_penalty"]["1.00"]-0.10200887086329927)<1e-15 and rejection["target90"]["total_cost_by_rejection_penalty"]["1.00"] > rejection["target90"]["raw_error_rate"],
        "G10_all_eight_targets_report_both_directions":rejection["targets"] == rejection["targets_lower_conditional"] == rejection["targets_worse_unit_penalty"] == 8,
        "G11_postselection_and_online_qec_ledgers_do_not_mix":rejection["postselection_joined_to_qec"] is False and rejection["global_cost_score"] is None and rejection["cross_lane_total"] is None,
        "G12_twelve_missing_physical_costs_stay_null":len(rejection["missing_fields"]) == 12 and all(row["value"] is None for row in rejection["missing_fields"]),
        "G13_only_finite_model_operational_boundary_is_established":taxonomy["wall_clock_operational_boundary"] == "ESTABLISHED_WITHIN_300US_FINITE_CUTOFF_MODEL" and taxonomy["terminal_profiles"] == 3 and taxonomy["low_cutoff_counterexample_retained"],
        "G14_boundary_uses_matched_idle_without_fit_or_ratio":taxonomy["baseline"] == "matched idle evolution of the same encoded finite-cutoff state" and taxonomy["fit"] is None and taxonomy["ratio"] is None,
        "G15_coherence_and_full_cost_claims_not_established":taxonomy["full_cost_operational_boundary"] == "NOT_ESTABLISHED" and taxonomy["simulation_derived_coherence_gain"] == "NOT_ESTABLISHED" and taxonomy["coherence_gain_value"] is None,
        "G16_postselected_and_experimental_break_even_not_established":taxonomy["postselected_break_even"] == "NOT_ESTABLISHED" and taxonomy["experimental_break_even"] == "NOT_ESTABLISHED",
        "G17_sivak_value_is_literature_only_nonranking_different_task":taxonomy["sivak_literature"] == {"value":2.27,"uncertainty":0.07,"evidence_grade":"LITERATURE_ONLY","ranking_eligible":False,"same_task":False},
        "G18_all_metric_and_evidence_transfers_forbidden":set(report["nontransfer_contract"].values()) == {False} and len(report["nontransfer_contract"]) == 6,
        "G19_manuscript_disclaims_physical_break_even":all(report["manuscript_audit"]["required_markers"].values()) and not any(report["manuscript_audit"]["forbidden_phrases"].values()),
        "G20_response_answers_with_numbers_and_no_overclaim":all(token in text for token in ("all 79,872 registered rounds","0.102009","all eight targets","not_established","2.27±0.07")) and not any(phrase.lower() in text for phrase in report["forbidden_response_phrases"]),
        "G21_rows_are_unique_state_complete_and_lossless":len(rows) == 24 and len({row["row_id"] for row in rows}) == 24 and {row["response_state"] for row in rows} == RESPONSE_STATES and all(row["claim"] and row["boundary"] and row["source_ids"] for row in rows),
        "G22_all_sources_registered_and_live":all(set(row["source_ids"]) <= set(report["artifact_registry"]) for row in rows) and ((not check_live_sources) or all(_binding_live(binding) for binding in report["artifact_registry"].values())),
        "G23_risk_and_task_audit_present":report["risk_audit"] == {"r_n161_present":True,"t7_3_4_audit_present":True},
        "G24_future_promotion_checklist_is_complete":len(report["response_package"]["manuscript_change_checklist"]) == 5 and "Keep coherence gain, full-cost and physical break-even NOT_ESTABLISHED." in report["response_package"]["manuscript_change_checklist"],
    }


def _run_mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    def mutate(path: Sequence[Any], value: Any) -> dict[str, Any]:
        candidate = copy.deepcopy(report)
        target: Any = candidate
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return candidate

    cases = [
        ("G01_identity_and_lifecycle",lambda:mutate(("task_status","T7.3.4"),"In Progress")),
        ("G02_preemptive_placeholder_honest",lambda:mutate(("response_package","package_readiness"),"ready_to_submit")),
        ("G03_phase6d_primary_keeps_full_denominator",lambda:mutate(("current_phase6d_primary","physical_rounds"),70000)),
        ("G04_phase6d_negative_verdict_not_rescued",lambda:mutate(("current_phase6d_primary","relative_improvement"),0.1)),
        ("G05_logical_channel_is_unconditional_cptni",lambda:mutate(("historical_logical_channel","discarded_trajectories"),1)),
        ("G06_postselection_is_offline_observed_and_ineligible",lambda:mutate(("offline_postselection","primary_metric_eligible"),True)),
        ("G07_postselection_scale_and_disjoint_evaluation_visible",lambda:mutate(("offline_postselection","evaluation_samples"),294912)),
        ("G08_target90_conditional_gain_reports_acceptance",lambda:mutate(("offline_postselection","target90","realized_survival_fraction"),1.0)),
        ("G09_unit_rejection_penalty_reverses_apparent_gain",lambda:mutate(("rejection_cost","target90","total_cost_by_rejection_penalty","1.00"),0.001)),
        ("G10_all_eight_targets_report_both_directions",lambda:mutate(("rejection_cost","targets_worse_unit_penalty"),0)),
        ("G11_postselection_and_online_qec_ledgers_do_not_mix",lambda:mutate(("rejection_cost","postselection_joined_to_qec"),True)),
        ("G12_twelve_missing_physical_costs_stay_null",lambda:mutate(("rejection_cost","missing_fields",0,"value"),0)),
        ("G13_only_finite_model_operational_boundary_is_established",lambda:mutate(("break_even_taxonomy","wall_clock_operational_boundary"),"PHYSICAL_BREAK_EVEN")),
        ("G14_boundary_uses_matched_idle_without_fit_or_ratio",lambda:mutate(("break_even_taxonomy","ratio"),2.0)),
        ("G15_coherence_and_full_cost_claims_not_established",lambda:mutate(("break_even_taxonomy","coherence_gain_value"),1.2)),
        ("G16_postselected_and_experimental_break_even_not_established",lambda:mutate(("break_even_taxonomy","experimental_break_even"),"ESTABLISHED")),
        ("G17_sivak_value_is_literature_only_nonranking_different_task",lambda:mutate(("break_even_taxonomy","sivak_literature","same_task"),True)),
        ("G18_all_metric_and_evidence_transfers_forbidden",lambda:mutate(("nontransfer_contract","sivak_to_project_break_even"),True)),
        ("G19_manuscript_disclaims_physical_break_even",lambda:mutate(("manuscript_audit","required_markers","Those claims require a separate protocol-matched experiment"),False)),
        ("G20_response_answers_with_numbers_and_no_overclaim",lambda:mutate(("response_package","english_response"),report["response_package"]["english_response"]+" We demonstrate physical break-even.")),
        ("G21_rows_are_unique_state_complete_and_lossless",lambda:{**copy.deepcopy(report),"response_rows":list(report["response_rows"][:-1])}),
        ("G22_all_sources_registered_and_live",lambda:mutate(("artifact_registry","cost","bytes"),report["artifact_registry"]["cost"]["bytes"]+1)),
        ("G23_risk_and_task_audit_present",lambda:mutate(("risk_audit","r_n161_present"),False)),
        ("G24_future_promotion_checklist_is_complete",lambda:mutate(("response_package","manuscript_change_checklist"),[])),
    ]
    results=[]
    for target_gate,factory in cases:
        mutated=factory()
        detected=not evaluate_gates(mutated,check_live_sources=target_gate=="G22_all_sources_registered_and_live")[target_gate]
        results.append({"mutation_id":f"M{len(results)+1:02d}","target_gate":target_gate,"detected":detected})
    return {"count":len(results),"detected":sum(case["detected"] for case in results),"cases":results}


def _markdown(report: Mapping[str, Any]) -> str:
    taxonomy=report["break_even_taxonomy"]
    lines=["# Reviewer response: post-selection and break-even boundaries","",f"- Task: `{report['task_id']}`",f"- Verdict: `{report['verdict']}`",f"- Package readiness: `{report['response_package']['package_readiness']}`",f"- Gates/mutations: `{report['gate_summary']['passed']}/{report['gate_summary']['total']}` / `{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`","","## Point-by-point response","",report["response_package"]["english_response"],"","## Frozen taxonomy","","| Quantity | Status |","| --- | --- |",f"| 300-us finite-cutoff wall-clock operational boundary | `{taxonomy['wall_clock_operational_boundary']}` |",f"| Full-cost operational boundary | `{taxonomy['full_cost_operational_boundary']}` |",f"| Simulation-derived coherence gain | `{taxonomy['simulation_derived_coherence_gain']}`; value=`{taxonomy['coherence_gain_value']}` |",f"| Postselected break-even | `{taxonomy['postselected_break_even']}` |",f"| Experimental break-even | `{taxonomy['experimental_break_even']}` |","","## Manuscript checklist",""]
    lines.extend(f"- {item}" for item in report["response_package"]["manuscript_change_checklist"])
    lines.extend(["","## Missing author input","",f"- `{report['response_package']['missing_information'][0]}`","","## 中文核对","","主指标不使用 post-selection；历史 post-selection 只作离线诊断且显式计入 rejection。当前只建立 300 us finite-cutoff matched-idle operational boundary；coherence gain、full-cost/postselected/experimental break-even 均未建立。",""])
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    fd,temp_name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as stream:
            stream.write(text)
        os.replace(temp_name,path)
    finally:
        if os.path.exists(temp_name): os.unlink(temp_name)


def write_outputs(report: dict[str, Any]) -> None:
    rows=_source_rows(report)
    fd,temp_name=tempfile.mkstemp(prefix=f".{DEFAULT_SOURCE_DATA.name}.",suffix=".tmp",dir=DEFAULT_SOURCE_DATA.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        os.replace(temp_name,DEFAULT_SOURCE_DATA)
    finally:
        if os.path.exists(temp_name): os.unlink(temp_name)
    report["source_data"]={**_binding(DEFAULT_SOURCE_DATA),"rows":len(rows)}
    _atomic_write_text(DEFAULT_MARKDOWN,_markdown(report)); report["markdown"]=_binding(DEFAULT_MARKDOWN)
    _atomic_write_text(DEFAULT_REPORT,json.dumps(report,ensure_ascii=False,indent=2)+"\n")


def verify_report() -> tuple[bool,dict[str,bool]]:
    if not DEFAULT_REPORT.exists(): return False,{"outputs_exist":False}
    stored=_load(DEFAULT_REPORT); fresh=build_report(generated_at_utc=stored.get("generated_at_utc"))
    checks={"outputs_exist":DEFAULT_SOURCE_DATA.exists() and DEFAULT_MARKDOWN.exists(),"identity":stored.get("task_id")==TASK_ID and stored.get("schema_version")==SCHEMA_VERSION,"verdict":stored.get("verdict")==VERDICT and fresh.get("verdict")==VERDICT,"all_gates":all(evaluate_gates(stored,check_live_sources=True).values()),"all_mutations":stored["semantic_mutation_audit"]["count"]==stored["semantic_mutation_audit"]["detected"]==len(stored["gates"]),"source_data":_source_data_matches(stored),"markdown_live":_binding_live(stored["markdown"]),"analysis_live":stored.get("analysis_sha256")==_canonical_sha256(_analysis_payload(stored)),"fresh_analysis":stored.get("analysis_sha256")==fresh.get("analysis_sha256")}
    return all(checks.values()),checks


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--verify",action="store_true"); args=parser.parse_args(argv)
    if args.verify:
        ok,checks=verify_report(); print(json.dumps(checks,ensure_ascii=False,indent=2)); return 0 if ok else 1
    report=build_report(); write_outputs(report); print(json.dumps({"verdict":report["verdict"],"gates":report["gate_summary"],"mutations":{"detected":report["semantic_mutation_audit"]["detected"],"total":report["semantic_mutation_audit"]["count"]},"source_rows":len(report["response_rows"]),"package_readiness":report["response_package"]["package_readiness"],"analysis_sha256":report["analysis_sha256"]},ensure_ascii=False,indent=2)); return 0 if report["verdict"]==VERDICT else 1


if __name__ == "__main__": raise SystemExit(main())
