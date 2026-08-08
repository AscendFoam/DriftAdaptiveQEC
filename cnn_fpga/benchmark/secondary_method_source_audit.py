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
TASK_ID = "T6.16.1"
SCHEMA_VERSION = "t6.16.1-secondary-method-source-audit-v1"
LEDGER = ROOT / "configs" / "literature" / "t6_16_1_secondary_method_sources.json"
FPGA_REPORT = ROOT / "docs" / "t6_8_6_fpga_decoder_normalization.json"
AQEC_PROJECT_REPORT = ROOT / "docs" / "t3_2_8_autonomous_sbs_wallclock_validation.json"
GQF_EXACT_REPORT = ROOT / "docs" / "t6_8_4_gqf_paper_exact_reproduction.json"
GQF_MATCHED_REPORT = ROOT / "docs" / "t6_8_5_gqf_route_a_matched_comparison_gate.json"
V5_REPORT = ROOT / "docs" / "t6_15_5_route_a_v5_final_evidence_gate.json"
SOURCE_CSV = ROOT / "docs" / "t6_16_1_secondary_method_source_data.csv"
DEFAULT_REPORT = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "secondary_method_source_audit.md"

EVIDENCE_GRADES = {
    "LITERATURE_ONLY",
    "OFFICIAL_CODE_REPRODUCTION",
    "PROJECT_NATIVE_MATCHED",
    "INELIGIBLE",
    "BLOCKED",
    "NEGATIVE",
}
REQUIRED_SCREENSHOT_CATEGORIES = {"CI", "ML/MAP", "Direct NN/RL", "AQEC", "CPD", "Hybrid CNN-FPGA"}
REQUIRED_NORMALIZED_CATEGORIES = {
    "standard_binning",
    "analog_maximum_likelihood",
    "direct_neural_decoder",
    "direct_neural_decoder_fpga",
    "offline_experiment_in_the_loop_rl_controller",
    "experiment_in_the_loop_rl_controller",
    "model_based_feedback_grape_recurrent_controller",
    "autonomous_physical_qec_protocol",
    "structured_lattice_closest_point_decoder",
    "project_preboard_map_event_fast_path",
    "proposed_regime_aware_safe_adaptive_map",
}
LANES = {
    "single_mode_decoder",
    "surface_gkp_gate_outer_code",
    "multimode_structured_lattice_cpd",
    "controller_rl_nmf",
    "aqec_wallclock",
    "fpga_implementation",
}
SOURCE_FIELDS = {
    "source_id", "title", "year", "primary", "version", "formal_identifier",
    "paper_urls", "supplement_urls", "code_url", "code_commit", "code_license",
    "paper_license", "data_url", "availability_note", "local_locators",
}
METHOD_FIELDS = {
    "method_id", "screenshot_category", "normalized_category", "lane_id", "source_id",
    "decision_object", "code_or_modes", "noise_model", "input_history", "output_action",
    "online_privilege", "metrics", "latency", "resources", "same_task_with_project",
    "comparability_notes",
}
METRIC_FIELDS = {
    "metric_id", "name", "value", "uncertainty", "unit", "direction", "denominator",
    "statistic", "evidence_grade", "source_locator", "ranking_eligible", "ineligibility_reason",
}
LATENCY_FIELDS = {
    "core_ns", "source_to_action_ns", "closed_loop_ns", "latency_cycles", "ii_ns",
    "boundary", "statistic", "evidence_grade", "source_locator",
}
RESOURCE_FIELDS = {
    "device", "precision", "lut", "ff", "bram", "dsp", "power_w",
    "evidence_grade", "source_locator",
}
CLAIM_FIELDS = {
    "claim_id", "category", "field", "original_claim", "normalized_value", "unit",
    "verdict", "correction", "source_ids", "source_locators",
}


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


def _method_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["method_id"]): row for row in report["methods"]}


def _metric_map(method: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["metric_id"]): row for row in method["metrics"]}


def _fpga_row(report: Mapping[str, Any], row_id: str) -> Mapping[str, Any]:
    return next(row for row in report["rows"] if row["row_id"] == row_id)


def _numeric_or_null(value: Any) -> bool:
    return value is None or isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bind_live_project_evidence(
    methods: list[dict[str, Any]],
    fpga: Mapping[str, Any],
    v5: Mapping[str, Any],
) -> None:
    method_by_id = {row["method_id"]: row for row in methods}
    source = _fpga_row(fpga, "project_t5_fast_path_core_pr")
    target = method_by_id["project_t5_hybrid_fast_path"]
    target["latency"].update(
        core_ns=source["decoder_core_ns"],
        source_to_action_ns=source["source_to_action_ns"],
        closed_loop_ns=source["closed_loop_ns"],
        latency_cycles=source["latency_cycles"],
        ii_ns=source["ii_ns"],
    )
    target["resources"].update(
        device=source["device"],
        precision=source["precision"],
        lut=source["lut"],
        ff=source["ff"],
        bram=source["bram"],
        dsp=source["dsp"],
        power_w=source["power_w"],
    )
    v5_method = method_by_id["project_v5_route_a"]
    v5_method["comparability_notes"] = (
        f"V5 early stop ({v5['verdict']}); no adaptive LER, RTL, latency or resource claim exists."
    )


def _derived_evidence(methods: list[Mapping[str, Any]]) -> dict[str, Any]:
    methods_by_id = {row["method_id"]: row for row in methods}
    ci = _metric_map(methods_by_id["noh_ci_cnot"])
    ml = _metric_map(methods_by_id["noh_ml_cnot_and_outer"])
    cnot_reductions: dict[str, float] = {}
    for db in (9, 12, 13):
        key = f"cnot_failure_{db}db"
        cnot_reductions[f"{db}_dB"] = (ci[key]["value"] - ml[key]["value"]) / ci[key]["value"]
    cpd = _metric_map(methods_by_id["lin_structured_cpd"])
    cpd_value = cpd["surface_gkp_cpd_threshold"]["value"]
    mwpm_value = cpd["surface_gkp_analog_mwpm_threshold"]["value"]
    aqec = _metric_map(methods_by_id["lachance_aqec_experiment"])
    return {
        "noh_cnot_failure_reduction_by_squeezing": cnot_reductions,
        "noh_about_50_percent_is_not_universal": min(cnot_reductions.values()) < 0.5 < max(cnot_reductions.values()),
        "cpd_same_task_absolute_threshold_delta": cpd_value - mwpm_value,
        "cpd_same_task_relative_threshold_delta": (cpd_value - mwpm_value) / mwpm_value,
        "aqec_reported_gain_ratios": [aqec["method_a_gain"]["value"], aqec["method_b_gain"]["value"]],
    }


def _project_anchor_summary(
    fpga: Mapping[str, Any],
    aqec_project: Mapping[str, Any],
    gqf_exact: Mapping[str, Any],
    gqf_matched: Mapping[str, Any],
    v5: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = [
        lane["comparison"]["autonomous_to_measurement_logical_lifetime_us_ratio"]
        for lane in aqec_project["lanes"].values()
    ]
    return {
        "fpga_same_task_external_comparator_count": fpga["claim_boundary"]["same_task_external_comparator_count"],
        "fpga_fastest_or_sota": fpga["claim_boundary"]["fastest_or_sota"],
        "project_autonomous_to_measurement_wallclock_lifetime_ratio_range": [min(ratios), max(ratios)],
        "project_aqec_scope": aqec_project["scope"],
        "gqf_exact_verdict": gqf_exact["verdict"],
        "gqf_exact_passed": gqf_exact["exact_qualification"]["passed"],
        "gqf_exact_failed": gqf_exact["exact_qualification"]["failed"],
        "gqf_matched_verdict": gqf_matched["verdict"],
        "surpass_puviani_nmf": gqf_matched["claim_boundary"]["surpass_puviani_NMF"],
        "v5_verdict": v5["verdict"],
        "v5_downstream_outputs_found": v5["v5_downstream_outputs_found"],
        "v5_measured_hardware_null_fields": v5["measured_hardware_claim"]["null_fields"],
    }


def _write_source_csv(report: Mapping[str, Any]) -> None:
    fieldnames = [
        "record_type", "record_id", "category", "normalized_category", "lane_id", "source_id",
        "field", "value", "uncertainty", "unit", "denominator", "statistic", "evidence_grade",
        "source_locator", "ranking_eligible", "verdict", "notes",
    ]
    rows: list[dict[str, Any]] = []
    for method in report["methods"]:
        if not method["metrics"]:
            rows.append({
                "record_type": "method", "record_id": method["method_id"],
                "category": method["screenshot_category"], "normalized_category": method["normalized_category"],
                "lane_id": method["lane_id"], "source_id": method["source_id"], "field": "method_contract",
                "value": None, "uncertainty": None, "unit": None, "denominator": None,
                "statistic": None, "evidence_grade": None, "source_locator": None,
                "ranking_eligible": False, "verdict": None, "notes": method["comparability_notes"],
            })
        for metric in method["metrics"]:
            rows.append({
                "record_type": "metric", "record_id": f"{method['method_id']}:{metric['metric_id']}",
                "category": method["screenshot_category"], "normalized_category": method["normalized_category"],
                "lane_id": method["lane_id"], "source_id": method["source_id"], "field": metric["name"],
                "value": metric["value"], "uncertainty": metric["uncertainty"], "unit": metric["unit"],
                "denominator": metric["denominator"], "statistic": metric["statistic"],
                "evidence_grade": metric["evidence_grade"], "source_locator": metric["source_locator"],
                "ranking_eligible": metric["ranking_eligible"], "verdict": None,
                "notes": metric["ineligibility_reason"],
            })
        for kind in ("latency", "resources"):
            contract = method[kind]
            rows.append({
                "record_type": kind, "record_id": f"{method['method_id']}:{kind}",
                "category": method["screenshot_category"], "normalized_category": method["normalized_category"],
                "lane_id": method["lane_id"], "source_id": method["source_id"], "field": kind,
                "value": json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "uncertainty": None, "unit": None, "denominator": None,
                "statistic": contract.get("statistic"), "evidence_grade": contract["evidence_grade"],
                "source_locator": contract.get("source_locator"), "ranking_eligible": False,
                "verdict": None, "notes": method["comparability_notes"],
            })
    for claim in report["claim_audit"]:
        rows.append({
            "record_type": "claim_audit", "record_id": claim["claim_id"], "category": claim["category"],
            "normalized_category": None, "lane_id": None, "source_id": ";".join(claim["source_ids"]),
            "field": claim["field"], "value": claim["normalized_value"], "uncertainty": None,
            "unit": claim["unit"], "denominator": None, "statistic": None, "evidence_grade": None,
            "source_locator": ";".join(claim["source_locators"]), "ranking_eligible": False,
            "verdict": claim["verdict"], "notes": claim["correction"],
        })
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    sources = report["sources"]
    methods = report["methods"]
    claims = report["claim_audit"]
    source_ids = {source["source_id"] for source in sources}
    method_ids = {method["method_id"] for method in methods}
    methods_by_id = _method_map(report)
    claim_by_id = {claim["claim_id"]: claim for claim in claims}
    all_metrics = [metric for method in methods for metric in method["metrics"]]
    numeric_latency_resource_fields = {
        "core_ns", "source_to_action_ns", "closed_loop_ns", "latency_cycles", "ii_ns",
        "lut", "ff", "bram", "dsp", "power_w",
    }
    typed_implementation_numbers = all(
        _numeric_or_null(contract[field])
        for method in methods
        for contract in (method["latency"], method["resources"])
        for field in numeric_latency_resource_fields & set(contract)
    )
    unsourced_qualitative = {"high", "low", "medium", "extremely high", "extremely low", "best", "excellent"}
    accepted_qualitative = [
        claim for claim in claims
        if claim["normalized_value"] is not None
        and isinstance(claim["normalized_value"], str)
        and claim["normalized_value"].strip().lower() in unsourced_qualitative
    ]
    cnot = report["derived_evidence"]["noh_cnot_failure_reduction_by_squeezing"]
    fpga_method = methods_by_id["project_t5_hybrid_fast_path"]
    v5_method = methods_by_id["project_v5_route_a"]
    aqec = methods_by_id["lachance_aqec_experiment"]
    aqec_latency = aqec["latency"]
    cpd_metrics = _metric_map(methods_by_id["lin_structured_cpd"])
    direct_categories = {methods_by_id[key]["normalized_category"] for key in (
        "wang_direct_nn_surface_gkp", "sivak2023_rl_controller", "puviani_nmf_controller"
    )}
    return {
        "G01_primary_source_schema_version_license_and_availability_complete": (
            len(sources) == 11 and len(source_ids) == len(sources)
            and all(set(source) == SOURCE_FIELDS and source["primary"] is True and source["version"]
                    and source["formal_identifier"] and source["availability_note"] for source in sources)
            and all(source["code_commit"] is None or len(source["code_commit"]) == 40 for source in sources)
        ),
        "G02_method_schema_covers_figures_and_normalized_categories": (
            len(methods) == 12 and len(method_ids) == len(methods)
            and {method["screenshot_category"] for method in methods} == REQUIRED_SCREENSHOT_CATEGORIES
            and {method["normalized_category"] for method in methods} == REQUIRED_NORMALIZED_CATEGORIES
            and all(set(method) == METHOD_FIELDS and method["source_id"] in source_ids for method in methods)
        ),
        "G03_six_lanes_and_decision_privilege_contracts_are_explicit": (
            {method["lane_id"] for method in methods} == LANES
            and all(all(str(method[field]).strip() for field in (
                "decision_object", "code_or_modes", "noise_model", "input_history", "output_action",
                "online_privilege", "comparability_notes")) for method in methods)
        ),
        "G04_metrics_have_typed_values_denominators_locators_and_evidence": (
            len(all_metrics) >= 20
            and all(set(metric) == METRIC_FIELDS and metric["evidence_grade"] in EVIDENCE_GRADES
                    and _numeric_or_null(metric["value"]) and _numeric_or_null(metric["uncertainty"])
                    and (metric["value"] is None or all(metric[field] is not None and str(metric[field]).strip()
                                                    for field in ("unit", "direction", "denominator", "statistic", "source_locator")))
                    and (metric["ranking_eligible"] is False or metric["value"] is not None)
                    and (metric["ranking_eligible"] or metric["ineligibility_reason"] is not None)
                    for metric in all_metrics)
        ),
        "G05_latency_and_resource_values_are_implementation_scoped_or_null": (
            typed_implementation_numbers
            and all(set(method["latency"]) == LATENCY_FIELDS and set(method["resources"]) == RESOURCE_FIELDS
                    and method["latency"]["evidence_grade"] in EVIDENCE_GRADES
                    and method["resources"]["evidence_grade"] in EVIDENCE_GRADES
                    and method["latency"]["boundary"] and method["latency"]["statistic"]
                    for method in methods)
            and all(
                not any(method["latency"][field] is not None for field in ("core_ns", "source_to_action_ns", "closed_loop_ns", "latency_cycles", "ii_ns"))
                or method["latency"]["source_locator"]
                for method in methods
            )
        ),
        "G06_qualitative_category_claims_and_unsourced_numbers_are_null": (
            not accepted_qualitative and len(claims) == 24 and all(set(claim) == CLAIM_FIELDS for claim in claims)
            and all(claim["normalized_value"] is None for claim in claims if claim["verdict"].startswith("NULL_"))
        ),
        "G07_9p9db_is_scoped_to_full_surface_gkp_threshold": (
            math.isclose(_metric_map(methods_by_id["noh_ml_cnot_and_outer"])["surface_gkp_threshold_db"]["value"], 9.9)
            and _metric_map(methods_by_id["noh_ml_cnot_and_outer"])["surface_gkp_threshold_db"]["ranking_eligible"] is False
            and claim_by_id["C06"]["normalized_value"] == 9.9
            and "Full concatenated surface-GKP" in claim_by_id["C06"]["correction"]
        ),
        "G08_cnot_failure_reduction_is_recomputed_and_nonuniversal": (
            math.isclose(cnot["9_dB"], (0.101 - 0.0689) / 0.101, abs_tol=1e-15)
            and math.isclose(cnot["12_dB"], (0.00869 - 0.00361) / 0.00869, abs_tol=1e-15)
            and math.isclose(cnot["13_dB"], (0.00260 - 0.000853) / 0.00260, abs_tol=1e-15)
            and report["derived_evidence"]["noh_about_50_percent_is_not_universal"] is True
            and claim_by_id["C05"]["normalized_value"] is None
        ),
        "G09_aqec_gain_is_1p14_and_decoder_latency_is_na_not_zero": (
            [metric["value"] for metric in aqec["metrics"] if metric["metric_id"] in {"method_a_gain", "method_b_gain"}] == [1.14, 1.14]
            and aqec_latency["core_ns"] is aqec_latency["source_to_action_ns"] is aqec_latency["closed_loop_ns"] is None
            and aqec_latency["statistic"] == "N/A, not zero"
            and claim_by_id["C14"]["normalized_value"] == 1.14
            and claim_by_id["C15"]["verdict"] == "N_A_NOT_ZERO"
        ),
        "G10_direct_nn_rl_and_nmf_are_not_merged_and_timing_boundaries_are_exact": (
            len(direct_categories) == 3
            and methods_by_id["overwater_fpga_nn_d5"]["latency"]["core_ns"] == 87.6
            and methods_by_id["yang_fpga_nn_d3"]["latency"]["core_ns"] == 124.0
            and methods_by_id["yang_fpga_nn_d3"]["latency"]["closed_loop_ns"] == 550.0
            and claim_by_id["C12"]["normalized_value"] is None
        ),
        "G11_cpd_threshold_pair_is_same_task_and_not_converted_to_db": (
            cpd_metrics["surface_gkp_cpd_threshold"]["unit"] == "paper_sigma"
            and cpd_metrics["surface_gkp_analog_mwpm_threshold"]["unit"] == "paper_sigma"
            and cpd_metrics["surface_gkp_cpd_threshold"]["value"] == 0.602
            and cpd_metrics["surface_gkp_analog_mwpm_threshold"]["value"] == 0.599
            and math.isclose(report["derived_evidence"]["cpd_same_task_absolute_threshold_delta"], 0.003, abs_tol=1e-15)
        ),
        "G12_project_hybrid_is_live_bound_preboard_and_v5_remains_negative": (
            fpga_method["latency"]["latency_cycles"] == 6
            and math.isclose(fpga_method["latency"]["core_ns"], 222.22222222222223, abs_tol=1e-12)
            and fpga_method["latency"]["source_to_action_ns"] is fpga_method["latency"]["closed_loop_ns"] is None
            and fpga_method["resources"]["lut"] == 3377 and fpga_method["resources"]["ff"] == 865
            and v5_method["latency"]["evidence_grade"] == v5_method["resources"]["evidence_grade"] == "NEGATIVE"
            and report["project_anchor_summary"]["v5_verdict"] == "NO_GO_V5_EARLY_HEADROOM_STOP"
            and report["project_anchor_summary"]["v5_downstream_outputs_found"] == []
        ),
        "G13_external_and_project_evidence_do_not_create_global_or_speed_ranking": (
            report["comparison_policy"] == {
                "global_leaderboard": "PROHIBITED",
                "cross_lane_score": "PROHIBITED",
                "same_task_only_ranking": True,
                "literature_value_is_project_reproduction": False,
                "same_task_external_fpga_comparator_count": 0,
                "fpga_fastest_or_sota": "PROHIBITED",
            }
            and report["project_anchor_summary"]["surpass_puviani_nmf"] == "PROHIBITED"
        ),
        "G14_source_csv_and_all_live_inputs_are_hash_bound": (
            report["source_data"]["rows"] >= 70 and len(report["source_data"]["sha256"]) == 64
            and set(report["bindings"]) == {"implementation", "ledger", "fpga_report", "aqec_project_report", "gqf_exact_report", "gqf_matched_report", "v5_report", "source_csv"}
            and all(len(value["sha256"]) == 64 for value in report["bindings"].values())
        ),
        "G15_targeted_semantic_mutations_fail_closed": (
            report["semantic_mutation_audit"]["count"] == 15
            and report["semantic_mutation_audit"]["detected"] == 15
            and all(case["rejected"] for case in report["semantic_mutation_audit"]["cases"])
        ),
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 15, "detected": 15, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("erase_source_version", "G01_primary_source_schema_version_license_and_availability_complete", lambda x: x["sources"][0].update(version=""))
    attempt("merge_direct_nn_with_rl", "G02_method_schema_covers_figures_and_normalized_categories", lambda x: _method_map(x)["sivak2023_rl_controller"].update(normalized_category="direct_neural_decoder"))
    attempt("erase_online_privilege", "G03_six_lanes_and_decision_privilege_contracts_are_explicit", lambda x: _method_map(x)["puviani_nmf_controller"].update(online_privilege=""))
    attempt("erase_metric_denominator", "G04_metrics_have_typed_values_denominators_locators_and_evidence", lambda x: _metric_map(_method_map(x)["noh_ci_cnot"])["cnot_failure_9db"].update(denominator=None))
    attempt("invent_unsourced_latency", "G05_latency_and_resource_values_are_implementation_scoped_or_null", lambda x: _method_map(x)["noh_ci_cnot"]["latency"].update(core_ns=20.0))
    attempt("promote_qualitative_high", "G06_qualitative_category_claims_and_unsourced_numbers_are_null", lambda x: next(c for c in x["claim_audit"] if c["claim_id"] == "C01").update(normalized_value="high"))
    attempt("move_9p9db_to_gate_rank", "G07_9p9db_is_scoped_to_full_surface_gkp_threshold", lambda x: _metric_map(_method_map(x)["noh_ml_cnot_and_outer"])["surface_gkp_threshold_db"].update(ranking_eligible=True))
    attempt("replace_squeezing_specific_reduction_with_50pct", "G08_cnot_failure_reduction_is_recomputed_and_nonuniversal", lambda x: x["derived_evidence"]["noh_cnot_failure_reduction_by_squeezing"].update({"9_dB": 0.5, "12_dB": 0.5, "13_dB": 0.5}))
    attempt("set_aqec_zero_latency", "G09_aqec_gain_is_1p14_and_decoder_latency_is_na_not_zero", lambda x: _method_map(x)["lachance_aqec_experiment"]["latency"].update(core_ns=0.0, statistic="zero"))
    attempt("replace_yang_closed_loop_with_core", "G10_direct_nn_rl_and_nmf_are_not_merged_and_timing_boundaries_are_exact", lambda x: _method_map(x)["yang_fpga_nn_d3"]["latency"].update(closed_loop_ns=124.0))
    attempt("convert_cpd_threshold_to_db", "G11_cpd_threshold_pair_is_same_task_and_not_converted_to_db", lambda x: _metric_map(_method_map(x)["lin_structured_cpd"])["surface_gkp_cpd_threshold"].update(unit="dB"))
    attempt("invent_v5_rtl", "G12_project_hybrid_is_live_bound_preboard_and_v5_remains_negative", lambda x: _method_map(x)["project_v5_route_a"]["latency"].update(evidence_grade="PROJECT_NATIVE_MATCHED", latency_cycles=6))
    attempt("claim_global_winner", "G13_external_and_project_evidence_do_not_create_global_or_speed_ranking", lambda x: x["comparison_policy"].update(global_leaderboard="PROJECT_WINS"))
    attempt("truncate_live_binding", "G14_source_csv_and_all_live_inputs_are_hash_bound", lambda x: x["bindings"]["fpga_report"].update(sha256="0"))
    attempt("forge_mutation_count", "G15_targeted_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 15, "detected": 14, "cases": []}))
    return {"count": len(cases), "detected": sum(case["rejected"] for case in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    ledger = _load(LEDGER)
    fpga = _load(FPGA_REPORT)
    aqec_project = _load(AQEC_PROJECT_REPORT)
    gqf_exact = _load(GQF_EXACT_REPORT)
    gqf_matched = _load(GQF_MATCHED_REPORT)
    v5 = _load(V5_REPORT)
    methods = deepcopy(ledger["methods"])
    _bind_live_project_evidence(methods, fpga, v5)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_at": ledger["frozen_at"],
        "scope": ledger["scope"],
        "sources": deepcopy(ledger["sources"]),
        "methods": methods,
        "claim_audit": deepcopy(ledger["claim_audit"]),
        "derived_evidence": _derived_evidence(methods),
        "project_anchor_summary": _project_anchor_summary(fpga, aqec_project, gqf_exact, gqf_matched, v5),
        "comparison_policy": {
            "global_leaderboard": "PROHIBITED",
            "cross_lane_score": "PROHIBITED",
            "same_task_only_ranking": True,
            "literature_value_is_project_reproduction": False,
            "same_task_external_fpga_comparator_count": fpga["claim_boundary"]["same_task_external_comparator_count"],
            "fpga_fastest_or_sota": fpga["claim_boundary"]["fastest_or_sota"],
        },
    }
    _write_source_csv(report)
    report["source_data"] = {"path": _relative(SOURCE_CSV), "sha256": _sha256(SOURCE_CSV), "rows": sum(1 for _ in SOURCE_CSV.open(encoding="utf-8")) - 1}
    report["bindings"] = {
        "implementation": _binding(Path(__file__)),
        "ledger": _binding(LEDGER),
        "fpga_report": _binding(FPGA_REPORT),
        "aqec_project_report": _binding(AQEC_PROJECT_REPORT),
        "gqf_exact_report": _binding(GQF_EXACT_REPORT),
        "gqf_matched_report": _binding(GQF_MATCHED_REPORT),
        "v5_report": _binding(V5_REPORT),
        "source_csv": _binding(SOURCE_CSV),
    }
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "failed": [gate for gate, passed in report["gates"].items() if not passed],
    }
    report["verdict"] = (
        "PASS_SECONDARY_METHOD_SOURCE_AUDIT_NO_GLOBAL_RANKING"
        if not report["gate_summary"]["failed"]
        else "FAIL_SECONDARY_METHOD_SOURCE_AUDIT"
    )
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    expected_gates = evaluate_gates(report)
    if dict(report["gates"]) != expected_gates:
        raise ValueError("stored gates do not match recomputation")
    failed = [gate for gate, passed in expected_gates.items() if not passed]
    expected_verdict = "PASS_SECONDARY_METHOD_SOURCE_AUDIT_NO_GLOBAL_RANKING" if not failed else "FAIL_SECONDARY_METHOD_SOURCE_AUDIT"
    if report["verdict"] != expected_verdict or report["gate_summary"] != {"passed": len(expected_gates) - len(failed), "failed": failed}:
        raise ValueError("stored gates/verdict do not match recomputation")


def write_markdown(report: Mapping[str, Any], path: Path = DEFAULT_MARKDOWN) -> None:
    cnot = report["derived_evidence"]["noh_cnot_failure_reduction_by_squeezing"]
    anchors = report["project_anchor_summary"]
    lines = [
        "# T6.16.1 两张异构方法图的一手来源审计",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- gate：`{report['gate_summary']['passed']}/15`",
        f"- 来源：`{len(report['sources'])}`；具体方法/实现：`{len(report['methods'])}`；截图 claim：`{len(report['claim_audit'])}`",
        "- 用途：Phase 6C 非主要比较的 source registry；不是 global leaderboard，也不回写 Phase 6B。",
        "",
        "## 审计后的核心结论",
        "",
        f"1. Noh 的两-GKP CNOT 中，ML 相对 CI 的 failure reduction 分别为 9 dB `{100*cnot['9_dB']:.3f}%`、12 dB `{100*cnot['12_dB']:.3f}%`、13 dB `{100*cnot['13_dB']:.3f}%`；因此“约 50%”不是通用值。",
        "2. `9.9 dB` 是 finite-squeezing-only 条件下、带 analog/history-aware outer decoding 的完整 surface–GKP finite-size threshold；不是 Table-I 门级 ML 阈值，也不是本项目 single-mode threshold。",
        "3. Direct NN、model-free RL 与 Puviani model-based Feedback-GRAPE NMF 是三类不同 decision object。Wang 的“decoding rate +50%”缺少可移植分母，规范化值为 null。",
        "4. NN/FPGA 只能报具体实现边界：Overwater d=5 为 `87.6 ns` post-implementation core estimate；Yang d=3 为 `124 ns` core、`550 ns` end-of-readout-to-feedback real closed loop。不存在 `10--100 us` 类别范围。",
        "5. AQEC 是 physical protocol，不是 syndrome decoder；classical decoder latency 记 N/A 而非 0。实验 lifetime gain 为 `1.14(18)`/`1.14(16)`，即约 14%，不是 universal 20%。",
        "6. Lin structured surface–GKP 的同任务 threshold 是 CPD `0.602` 对 analog-MWPM `0.599`；数值保留 paper sigma，不换算 dB。generic、linear 与 polynomial complexity 必须按 lattice structure 分列。",
        f"7. 项目 T5 仅有 six-cycle、`222.222 ns`、II `37.037 ns` 的 preboard core estimate；external same-task FPGA comparator 为 `{anchors['fpga_same_task_external_comparator_count']}`，所以 faster/SOTA 禁止。V5 仍为 `{anchors['v5_verdict']}`。",
        f"8. 项目 T3.2.8 common-wall-clock simulator 的 autonomous/measurement lifetime ratio 范围为 `{anchors['project_autonomous_to_measurement_wallclock_lifetime_ratio_range'][0]:.6f}--{anchors['project_autonomous_to_measurement_wallclock_lifetime_ratio_range'][1]:.6f}`，是负/不占优的 project-native model result，不能借用 AQEC 论文的 1.14。",
        "",
        "## 分 lane 使用规则",
        "",
        "| lane | 可以比较 | 禁止混排 |",
        "| --- | --- | --- |",
        "| single-mode decoder | 同 syndrome/action/observability/budget 的 LER 与 tail | surface-code threshold、AQEC lifetime、controller gain |",
        "| surface-GKP gate/outer code | 同 CNOT circuit 的 failure；同 family finite-size threshold | single-mode repeated-memory LER |",
        "| multimode structured CPD | 同 lattice/noise/size 的 correctness、threshold、scaling | 把 CPD 当 single-mode 新 comparator |",
        "| controller/RL/NMF | 同 physical protocol、history/action、training/compute budget | direct decoder inference |",
        "| AQEC wall-clock | 同 apparatus/model、wall-clock、duty/event budget的 lifetime | zero-latency decoder claim |",
        "| FPGA implementation | 同 code/input/action/problem size/precision/boundary/evidence | 跨 code family 的纳秒总榜 |",
        "",
        "## 产物与证据状态",
        "",
        f"- machine registry：`docs/t6_16_1_secondary_method_source_audit.json`",
        f"- Source Data：`{report['source_data']['path']}`（`{report['source_data']['rows']}` rows）",
        "- evidence grade 仅允许 `LITERATURE_ONLY/OFFICIAL_CODE_REPRODUCTION/PROJECT_NATIVE_MATCHED/INELIGIBLE/BLOCKED/NEGATIVE`。",
        f"- Puviani exact reproduction：`{anchors['gqf_exact_verdict']}`；matched gate：`{anchors['gqf_matched_verdict']}`；`surpass Puviani NMF` 仍为 `{anchors['surpass_puviani_nmf']}`。",
        "- null 表示没有一手 locator 或不适用；不得用 high/medium/low、0、类别均值或相邻论文插补。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T6.16.1 source-verified secondary-method audit")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report()
    verify_report(report)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps({"verdict": report["verdict"], "gate_summary": report["gate_summary"], "source_rows": report["source_data"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
