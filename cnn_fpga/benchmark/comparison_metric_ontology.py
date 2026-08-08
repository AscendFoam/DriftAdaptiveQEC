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
TASK_ID = "T6.16.2"
SCHEMA_VERSION = "t6.16.2-comparison-metric-ontology-v1"
ONTOLOGY = ROOT / "configs" / "literature" / "t6_16_2_comparison_ontology.json"
SOURCE_AUDIT = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"
T511 = ROOT / "docs" / "t5_1_1_comparison_set_registry.json"
T651 = ROOT / "docs" / "t6_5_1_route_a_claim_contract.json"
T687 = ROOT / "docs" / "t6_8_7_route_a_claim_matrix.json"
SOURCE_CSV = ROOT / "docs" / "t6_16_2_comparison_ontology_source_data.csv"
DEFAULT_REPORT = ROOT / "docs" / "t6_16_2_comparison_ontology.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "comparison_metric_ontology.md"

LANE_IDS = {
    "single_mode_decoder",
    "surface_gkp_gate_outer_code",
    "multimode_structured_lattice_cpd",
    "controller_rl_nmf",
    "aqec_wallclock",
    "fpga_implementation",
}
TIMING_BOUNDARIES = {
    "decoder_core", "update_compute", "transport", "source_to_action", "closed_loop", "initiation_interval"
}
RESOURCE_IDS = {"LUT", "FF", "BRAM", "DSP", "power"}
VALUE_STATES = {
    "MEASURED_VALUE", "ESTIMATE_VALUE", "REPRODUCED_VALUE", "LITERATURE_VALUE",
    "NULL_NOT_REPORTED", "N_A_NOT_APPLICABLE", "FAILED", "NEGATIVE",
}
VALUE_STATE_EVIDENCE = {
    "MEASURED_VALUE": "measured_hardware",
    "ESTIMATE_VALUE": "estimate",
    "REPRODUCED_VALUE": "reproduction",
    "LITERATURE_VALUE": "literature",
    "NULL_NOT_REPORTED": "null_not_reported",
    "N_A_NOT_APPLICABLE": "not_applicable",
    "FAILED": "failed_attempt",
    "NEGATIVE": "negative_result",
}
REQUIRED_METRICS = {
    "p_L", "p_X", "p_Y", "p_Z", "average_ler", "p95_window_ler", "worst_window_ler",
    "cvar95_window_ler", "cnot_failure_probability", "squeezing_threshold_db",
    "noise_threshold_sigma", "logical_lifetime_cycles", "logical_lifetime_us",
    "logical_lifetime_ms", "lifetime_gain_ratio", "coherence_break_even_gain",
    "adaptation_lag_cycles", "adaptation_lag_epochs", "false_update_rate", "fallback_rate",
    "avoided_errors", "induced_errors", "deadline_miss_rate", "latency_ns", "latency_cycles",
    "initiation_interval_ns", "clock_mhz", "lut_count", "ff_count", "bram_count", "dsp_count", "power_w",
}

# Every non-empty metric row in T6.16.1 is either mapped to a typed metric or
# explicitly rejected because its source denominator is undefined.
SOURCE_METRIC_CROSSWALK: dict[tuple[str, str], str | None] = {
    ("noh_ci_cnot", "cnot_failure_9db"): "cnot_failure_probability",
    ("noh_ci_cnot", "cnot_failure_12db"): "cnot_failure_probability",
    ("noh_ci_cnot", "cnot_failure_13db"): "cnot_failure_probability",
    ("noh_ml_cnot_and_outer", "cnot_failure_9db"): "cnot_failure_probability",
    ("noh_ml_cnot_and_outer", "cnot_failure_12db"): "cnot_failure_probability",
    ("noh_ml_cnot_and_outer", "cnot_failure_13db"): "cnot_failure_probability",
    ("noh_ml_cnot_and_outer", "surface_gkp_threshold_db"): "squeezing_threshold_db",
    ("wang_direct_nn_surface_gkp", "threshold_perfect_measurement"): "noise_threshold_sigma",
    ("wang_direct_nn_surface_gkp", "mwpm_threshold_perfect_measurement"): "noise_threshold_sigma",
    ("wang_direct_nn_surface_gkp", "threshold_noisy_measurement"): "noise_threshold_sigma",
    ("wang_direct_nn_surface_gkp", "mwpm_threshold_noisy_measurement"): "noise_threshold_sigma",
    ("wang_direct_nn_surface_gkp", "decoding_rate_improvement"): None,
    ("sivak2023_rl_controller", "break_even_gain"): "coherence_break_even_gain",
    ("sivak2023_rl_controller", "training_epoch_seconds"): "training_epoch_wall_time_s",
    ("sivak2026_rl_drift", "control_only_ler_reduction"): "ler_reduction_fraction",
    ("sivak2026_rl_drift", "control_only_ler_stability"): "ler_stability_ratio",
    ("sivak2026_rl_drift", "control_decoder_ler_reduction"): "ler_reduction_fraction",
    ("sivak2026_rl_drift", "control_decoder_ler_stability"): "ler_stability_ratio",
    ("sivak2026_rl_drift", "calibrated_finetune_ler_reduction"): "ler_reduction_fraction",
    ("sivak2026_rl_drift", "recovery_epochs"): "adaptation_lag_epochs",
    ("puviani_nmf_controller", "low_noise_tz_standard_cycles"): "logical_lifetime_cycles",
    ("puviani_nmf_controller", "low_noise_tz_nmf_cycles"): "logical_lifetime_cycles",
    ("lachance_aqec_experiment", "method_a_gain"): "lifetime_gain_ratio",
    ("lachance_aqec_experiment", "method_b_gain"): "lifetime_gain_ratio",
    ("lachance_aqec_experiment", "method_a_free_lifetime_ms"): "logical_lifetime_ms",
    ("lachance_aqec_experiment", "method_b_free_lifetime_ms"): "logical_lifetime_ms",
    ("lachance_aqec_experiment", "method_a_lifetime_ms"): "logical_lifetime_ms",
    ("lachance_aqec_experiment", "method_b_lifetime_ms"): "logical_lifetime_ms",
    ("lin_structured_cpd", "surface_gkp_cpd_threshold"): "noise_threshold_sigma",
    ("lin_structured_cpd", "surface_gkp_analog_mwpm_threshold"): "noise_threshold_sigma",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maps(ontology: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    return (
        {str(row["lane_id"]): row for row in ontology["lanes"]},
        {str(row["metric_id"]): row for row in ontology["metrics"]},
    )


def validate_observation(record: Mapping[str, Any], ontology: Mapping[str, Any]) -> list[str]:
    """Return fail-closed reasons for one normalized observation."""
    required = {
        "observation_id", "lane_id", "metric_id", "state", "value", "unit", "denominator",
        "statistic", "task_signature", "timing_boundary", "evidence_grade", "source_locator",
        "qualitative_complexity",
    }
    reasons: list[str] = []
    if set(record) != required:
        return ["schema_mismatch"]
    lanes, metrics = _maps(ontology)
    lane_id = record["lane_id"]
    metric_id = record["metric_id"]
    state = record["state"]
    if lane_id not in lanes:
        reasons.append("unknown_lane")
    if metric_id not in metrics:
        reasons.append("unknown_metric")
    if state not in ontology["value_states"]:
        reasons.append("unknown_value_state")
    if reasons:
        return reasons
    metric = metrics[metric_id]
    state_rule = ontology["value_states"][state]
    if lane_id not in metric["allowed_lanes"]:
        reasons.append("wrong_lane")
    value = record["value"]
    if state_rule["value_required"]:
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
            reasons.append("value_required")
        elif isinstance(value, float) and not math.isfinite(value):
            reasons.append("nonfinite_value")
        if not record["unit"] or not record["denominator"] or not record["statistic"] or not record["source_locator"]:
            reasons.append("missing_numeric_contract")
    elif value is not None:
        reasons.append("null_na_failed_negative_must_not_have_value")
    signature = record["task_signature"]
    if not isinstance(signature, Mapping) or set(signature) != set(ontology["task_signature_fields"]):
        reasons.append("task_signature_incomplete")
    elif any(value is None or not str(value).strip() for value in signature.values()):
        reasons.append("task_signature_empty")
    elif signature["evidence_level"] != VALUE_STATE_EVIDENCE[state]:
        reasons.append("value_state_evidence_level_mismatch")
    boundary = record["timing_boundary"]
    if metric["family"] in {"timing", "throughput"}:
        if boundary not in lanes[lane_id]["allowed_timing_boundaries"]:
            reasons.append("timing_boundary_missing_or_wrong_lane")
    elif boundary is not None:
        reasons.append("timing_boundary_on_nontiming_metric")
    complexity = record["qualitative_complexity"]
    if complexity is not None:
        if not isinstance(complexity, Mapping) or set(complexity) != {"claim", "basis", "source_locator"}:
            reasons.append("qualitative_complexity_schema")
        elif complexity["basis"] not in {"recomputable_numeric", "source_asymptotic"} or not complexity["source_locator"]:
            reasons.append("qualitative_complexity_unsupported")
    return reasons


def compare_observations(left: Mapping[str, Any], right: Mapping[str, Any], ontology: Mapping[str, Any]) -> list[str]:
    """Return why a raw rank/comparison is ineligible."""
    reasons = [f"left:{row}" for row in validate_observation(left, ontology)]
    reasons += [f"right:{row}" for row in validate_observation(right, ontology)]
    if reasons:
        return reasons
    for field in ("lane_id", "metric_id", "unit", "denominator", "statistic", "timing_boundary"):
        if left[field] != right[field]:
            reasons.append(f"mismatch:{field}")
    if left["task_signature"] != right["task_signature"]:
        reasons.append("mismatch:task_signature")
    for side, record in (("left", left), ("right", right)):
        if not ontology["value_states"][record["state"]]["ranking_allowed"]:
            reasons.append(f"{side}:state_not_rankable")
    return reasons


def _signature(lane_id: str, evidence_level: str = "reproduction") -> dict[str, str]:
    return {
        "code_family": lane_id,
        "modes_or_distance": "registered_size",
        "decision_target": "registered_target",
        "input_semantics": "registered_input",
        "history_horizon": "registered_horizon",
        "output_action": "registered_action",
        "noise_model": "registered_noise",
        "observability": "registered_observability",
        "online_privilege": "registered_privilege",
        "time_basis": "registered_time_basis",
        "compute_budget": "registered_compute_budget",
        "precision": "registered_precision",
        "evidence_level": evidence_level,
    }


def _observation(
    *, observation_id: str, lane_id: str = "single_mode_decoder", metric_id: str = "p_L",
    state: str = "REPRODUCED_VALUE", value: Any = 0.1, unit: str | None = "probability_per_round",
    denominator: str | None = "100 registered decoded rounds", statistic: str | None = "raw errors/rounds",
    timing_boundary: str | None = None, qualitative_complexity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "observation_id": observation_id,
        "lane_id": lane_id,
        "metric_id": metric_id,
        "state": state,
        "value": value,
        "unit": unit,
        "denominator": denominator,
        "statistic": statistic,
        "task_signature": _signature(lane_id, VALUE_STATE_EVIDENCE[state]),
        "timing_boundary": timing_boundary,
        "evidence_grade": "PROJECT_NATIVE_MATCHED",
        "source_locator": "synthetic ontology self-test locator",
        "qualitative_complexity": qualitative_complexity,
    }


def _contract_self_tests(ontology: Mapping[str, Any]) -> dict[str, Any]:
    valid = _observation(observation_id="valid")
    valid_reasons = validate_observation(valid, ontology)
    cases: list[dict[str, Any]] = []

    def invalid(case_id: str, record: Mapping[str, Any], expected: str) -> None:
        reasons = validate_observation(record, ontology)
        cases.append({"case": case_id, "expected_reason": expected, "reasons": reasons, "rejected": expected in reasons})

    row = _observation(observation_id="wrong_lane", lane_id="aqec_wallclock")
    invalid("wrong_lane_metric", row, "wrong_lane")
    row = _observation(observation_id="null_zero", state="NULL_NOT_REPORTED", value=0.0)
    invalid("null_imputed_zero", row, "null_na_failed_negative_must_not_have_value")
    row = _observation(observation_id="na_zero", state="N_A_NOT_APPLICABLE", value=0.0)
    invalid("na_imputed_zero", row, "null_na_failed_negative_must_not_have_value")
    row = _observation(observation_id="failed_value", state="FAILED", value=0.01)
    invalid("failed_imputed_value", row, "null_na_failed_negative_must_not_have_value")
    row = _observation(observation_id="timing", lane_id="fpga_implementation", metric_id="latency_ns", value=124.0, unit="ns", denominator="one decoder input", statistic="deterministic", timing_boundary=None)
    invalid("latency_without_boundary", row, "timing_boundary_missing_or_wrong_lane")
    row = _observation(observation_id="qualitative", lane_id="multimode_structured_lattice_cpd", metric_id="asymptotic_time_complexity", value="O(n)", unit="big_O", denominator="structured family", statistic="source theorem", qualitative_complexity={"claim":"low", "basis":"opinion", "source_locator":None})
    invalid("unsupported_qualitative_complexity", row, "qualitative_complexity_unsupported")

    compare_cases: list[dict[str, Any]] = []
    right = deepcopy(valid)
    right["observation_id"] = "different_denominator"
    right["denominator"] = "200 unrelated rounds"
    reasons = compare_observations(valid, right, ontology)
    compare_cases.append({"case":"cross_denominator", "reasons":reasons, "rejected":"mismatch:denominator" in reasons})
    right = _observation(observation_id="different_lane", lane_id="surface_gkp_gate_outer_code", metric_id="cnot_failure_probability", value=0.1, unit="probability_per_error_corrected_cnot", denominator="one CNOT", statistic="raw failures/trials")
    reasons = compare_observations(valid, right, ontology)
    compare_cases.append({"case":"cross_family_raw_rank", "reasons":reasons, "rejected":"mismatch:lane_id" in reasons and "mismatch:metric_id" in reasons})
    left_t = _observation(observation_id="core", lane_id="fpga_implementation", metric_id="latency_ns", value=124.0, unit="ns", denominator="one action", statistic="deterministic", timing_boundary="decoder_core")
    right_t = deepcopy(left_t)
    right_t.update(observation_id="closed", value=550.0, timing_boundary="closed_loop")
    reasons = compare_observations(left_t, right_t, ontology)
    compare_cases.append({"case":"cross_latency_boundary", "reasons":reasons, "rejected":"mismatch:timing_boundary" in reasons})
    null_right = deepcopy(valid)
    null_right.update(observation_id="null", state="NULL_NOT_REPORTED", value=None)
    null_right["task_signature"]["evidence_level"] = VALUE_STATE_EVIDENCE["NULL_NOT_REPORTED"]
    reasons = compare_observations(valid, null_right, ontology)
    compare_cases.append({"case":"null_rank", "reasons":reasons, "rejected":"right:state_not_rankable" in reasons})
    return {
        "valid_observation_passed": not valid_reasons,
        "valid_observation_reasons": valid_reasons,
        "invalid_cases": cases,
        "comparison_cases": compare_cases,
        "global_score_api": None,
        "global_score_prohibited": True,
    }


def _crosswalk(source: Mapping[str, Any], ontology: Mapping[str, Any]) -> list[dict[str, Any]]:
    _, metrics = _maps(ontology)
    rows: list[dict[str, Any]] = []
    for method in source["methods"]:
        for source_metric in method["metrics"]:
            key = (method["method_id"], source_metric["metric_id"])
            ontology_metric = SOURCE_METRIC_CROSSWALK.get(key, "__MISSING__")
            rejected = ontology_metric is None
            allowed = False if rejected or ontology_metric == "__MISSING__" else method["lane_id"] in metrics[ontology_metric]["allowed_lanes"]
            rows.append({
                "method_id": method["method_id"],
                "source_metric_id": source_metric["metric_id"],
                "source_lane_id": method["lane_id"],
                "ontology_metric_id": ontology_metric if ontology_metric != "__MISSING__" else None,
                "status": "REJECTED_UNDEFINED_DENOMINATOR" if rejected else ("MAPPED" if allowed else "INVALID"),
                "allowed_in_lane": allowed,
                "source_value_is_null": source_metric["value"] is None,
                "reason": "Wang decoding-rate prose has no portable denominator" if rejected else None,
            })
    return rows


def _write_csv(report: Mapping[str, Any]) -> None:
    fields = ["record_type", "record_id", "lane_id", "family", "unit", "direction", "denominator", "statistic", "allowed_lanes", "state", "notes"]
    rows: list[dict[str, Any]] = []
    for lane in report["ontology"]["lanes"]:
        rows.append({"record_type":"lane", "record_id":lane["lane_id"], "lane_id":lane["lane_id"], "family":None, "unit":None, "direction":None, "denominator":lane["ranking_unit"], "statistic":None, "allowed_lanes":lane["lane_id"], "state":None, "notes":lane["decision_object"]})
    for metric in report["ontology"]["metrics"]:
        rows.append({"record_type":"metric", "record_id":metric["metric_id"], "lane_id":None, "family":metric["family"], "unit":metric["unit"], "direction":metric["direction"], "denominator":metric["denominator_contract"], "statistic":metric["statistic_contract"], "allowed_lanes":";".join(metric["allowed_lanes"]), "state":None, "notes":metric["conversion_policy"]})
    for boundary in report["ontology"]["timing_boundaries"]:
        rows.append({"record_type":"timing_boundary", "record_id":boundary["boundary_id"], "lane_id":None, "family":"timing", "unit":"ns_or_cycles", "direction":None, "denominator":f"{boundary['start_event']} -> {boundary['end_event']}", "statistic":None, "allowed_lanes":None, "state":None, "notes":"includes="+";".join(boundary["includes"])+" excludes="+";".join(boundary["excludes"])})
    for resource in report["ontology"]["resource_dimensions"]:
        rows.append({"record_type":"resource", "record_id":resource["resource_id"], "lane_id":"fpga_implementation", "family":"resources", "unit":resource["resource_id"], "direction":None, "denominator":";".join(resource["required_qualifiers"]), "statistic":None, "allowed_lanes":"fpga_implementation", "state":None, "notes":";".join(resource["allowed_evidence"])})
    for state, rule in report["ontology"]["value_states"].items():
        rows.append({"record_type":"value_state", "record_id":state, "lane_id":None, "family":None, "unit":None, "direction":None, "denominator":None, "statistic":None, "allowed_lanes":None, "state":state, "notes":rule["meaning"]})
    for cross in report["source_metric_crosswalk"]:
        rows.append({"record_type":"source_crosswalk", "record_id":f"{cross['method_id']}:{cross['source_metric_id']}", "lane_id":cross["source_lane_id"], "family":None, "unit":None, "direction":None, "denominator":None, "statistic":None, "allowed_lanes":cross["source_lane_id"] if cross["allowed_in_lane"] else None, "state":cross["status"], "notes":cross["ontology_metric_id"] or cross["reason"]})
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    ontology = report["ontology"]
    lanes, metrics = _maps(ontology)
    source = report["source_audit_summary"]
    crosswalk = report["source_metric_crosswalk"]
    self_tests = report["contract_self_tests"]
    state_rules = ontology["value_states"]
    required_lane_fields = {"lane_id", "label", "decision_object", "required_task_signature", "allowed_metric_families", "allowed_timing_boundaries", "ranking_unit", "forbidden_cross_lane_targets"}
    required_metric_fields = {"metric_id", "family", "value_type", "unit", "direction", "denominator_contract", "statistic_contract", "aggregation", "allowed_lanes", "conversion_policy"}
    return {
        "G01_exactly_six_task_signature_lanes_are_frozen": len(lanes) == 6 and set(lanes) == LANE_IDS and all(set(row) == required_lane_fields for row in ontology["lanes"]),
        "G02_all_required_metric_families_and_board_metrics_are_defined": REQUIRED_METRICS <= set(metrics) and len(metrics) >= 45 and all(set(row) == required_metric_fields for row in ontology["metrics"]),
        "G03_every_metric_has_unit_direction_denominator_statistic_and_conversion_policy": all(all(row[field] and str(row[field]).strip() for field in ("unit", "direction", "denominator_contract", "statistic_contract", "aggregation", "conversion_policy")) for row in ontology["metrics"]),
        "G04_metric_allowed_lanes_are_nonempty_known_and_family_allowed": all(row["allowed_lanes"] and set(row["allowed_lanes"]) <= LANE_IDS and all(row["family"] in lanes[lane]["allowed_metric_families"] for lane in row["allowed_lanes"]) for row in ontology["metrics"]),
        "G05_null_na_failed_negative_and_value_states_are_disjoint": set(state_rules) == VALUE_STATES and all(isinstance(rule["value_required"], bool) and isinstance(rule["ranking_allowed"], bool) and rule["meaning"] for rule in state_rules.values()) and all(not state_rules[state]["ranking_allowed"] and not state_rules[state]["value_required"] for state in ("NULL_NOT_REPORTED", "N_A_NOT_APPLICABLE", "FAILED", "NEGATIVE")),
        "G06_timing_boundaries_are_exact_and_start_end_include_exclude_are_explicit": {row["boundary_id"] for row in ontology["timing_boundaries"]} == TIMING_BOUNDARIES and all(row["start_event"] and row["end_event"] and isinstance(row["includes"], list) and isinstance(row["excludes"], list) and isinstance(row["composable_from"], list) for row in ontology["timing_boundaries"]),
        "G07_latency_and_ii_are_not_interchangeable_or_implicitly_composed": metrics["initiation_interval_ns"]["family"] == "throughput" and metrics["latency_ns"]["family"] == "timing" and next(row for row in ontology["timing_boundaries"] if row["boundary_id"] == "source_to_action")["composable_from"] == ["transport", "decoder_core"] and next(row for row in ontology["timing_boundaries"] if row["boundary_id"] == "closed_loop")["composable_from"] == ["source_to_action"],
        "G08_resource_dimensions_require_device_primitive_tool_stage_or_activity_qualifiers": (
            {row["resource_id"] for row in ontology["resource_dimensions"]} == RESOURCE_IDS
            and {
                row["resource_id"]: set(row["required_qualifiers"])
                for row in ontology["resource_dimensions"]
            } == {
                "LUT": {"device", "primitive_type", "tool", "stage", "seed/profile"},
                "FF": {"device", "primitive_type", "tool", "stage", "seed/profile"},
                "BRAM": {"device", "primitive_bits", "tool", "stage", "seed/profile"},
                "DSP": {"device", "primitive_type", "tool", "stage", "seed/profile"},
                "power": {"device", "voltage", "clock", "activity", "method", "stage"},
            }
            and all(row["allowed_evidence"] for row in ontology["resource_dimensions"])
        ),
        "G09_t6_16_1_all_methods_land_in_exactly_one_known_lane": source["method_count"] == 12 and set(source["method_lane_ids"]) <= LANE_IDS and set(source["method_lane_ids"]) == LANE_IDS,
        "G10_every_t6_16_1_metric_is_mapped_or_explicitly_rejected": len(crosswalk) == source["metric_count"] == len(SOURCE_METRIC_CROSSWALK) and sum(row["status"] == "REJECTED_UNDEFINED_DENOMINATOR" for row in crosswalk) == 1 and all(row["status"] in {"MAPPED", "REJECTED_UNDEFINED_DENOMINATOR"} and (row["allowed_in_lane"] or row["status"].startswith("REJECTED")) for row in crosswalk),
        "G11_observation_validator_rejects_wrong_lane_and_state_imputation": self_tests["valid_observation_passed"] and len(self_tests["invalid_cases"]) == 6 and all(row["rejected"] for row in self_tests["invalid_cases"]),
        "G12_comparator_requires_same_lane_metric_denominator_statistic_boundary_and_task_signature": len(self_tests["comparison_cases"]) == 4 and all(row["rejected"] for row in self_tests["comparison_cases"]),
        "G13_global_score_and_cross_family_raw_ranking_are_absent_and_prohibited": report["ranking_policy"] == {"global_score": "PROHIBITED", "cross_lane_raw_ranking": "PROHIBITED", "same_lane_exact_signature_only": True, "qualitative_complexity_without_numeric_or_asymptotic_locator": "PROHIBITED"} and self_tests["global_score_api"] is None and self_tests["global_score_prohibited"] is True,
        "G14_qualitative_complexity_requires_recomputable_numeric_or_source_asymptotic_basis": all(row["metric_id"] != "asymptotic_time_complexity" or row["value_type"] == "symbolic" and "source locator" in row["statistic_contract"] for row in ontology["metrics"]) and next(row for row in self_tests["invalid_cases"] if row["case"] == "unsupported_qualitative_complexity")["rejected"],
        "G15_parent_contracts_and_source_audit_are_hash_bound_without_claim_promotion": set(report["bindings"]) == {"implementation", "ontology", "t6_16_1", "t5_1_1", "t6_5_1", "t6_8_7", "source_csv"} and all(len(row["sha256"]) == 64 for row in report["bindings"].values()) and report["parent_contracts"] == {"T5.1.1":"PASS", "T6.5.1":"PASS_ROUTE_A_CLAIM_ROLE_AND_LANE_CONTRACT_FROZEN", "T6.8.7":"PASS_ROUTE_A_ATOMIC_CLAIM_MATRIX_WITH_RESTRICTED_POSITIVE_CLAIMS", "T6.16.1":"PASS_SECONDARY_METHOD_SOURCE_AUDIT_NO_GLOBAL_RANKING"},
        "G16_source_data_and_targeted_semantic_mutations_are_complete": report["source_data"]["rows"] >= 95 and len(report["source_data"]["sha256"]) == 64 and report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 16 and all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]),
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count":16, "detected":16, "cases":[]}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case":name, "target_gate":gate, "rejected":rejected})
    attempt("drop_lane", "G01_exactly_six_task_signature_lanes_are_frozen", lambda x: x["ontology"]["lanes"].pop())
    attempt("drop_pY", "G02_all_required_metric_families_and_board_metrics_are_defined", lambda x: x["ontology"]["metrics"].remove(next(row for row in x["ontology"]["metrics"] if row["metric_id"] == "p_Y")))
    attempt("erase_denominator", "G03_every_metric_has_unit_direction_denominator_statistic_and_conversion_policy", lambda x: next(row for row in x["ontology"]["metrics"] if row["metric_id"] == "p_L").update(denominator_contract=""))
    attempt("allow_pL_in_aqec", "G04_metric_allowed_lanes_are_nonempty_known_and_family_allowed", lambda x: next(row for row in x["ontology"]["metrics"] if row["metric_id"] == "p_L")["allowed_lanes"].append("aqec_wallclock"))
    attempt("rank_null", "G05_null_na_failed_negative_and_value_states_are_disjoint", lambda x: x["ontology"]["value_states"]["NULL_NOT_REPORTED"].update(ranking_allowed=True))
    attempt("erase_closed_loop_end", "G06_timing_boundaries_are_exact_and_start_end_include_exclude_are_explicit", lambda x: next(row for row in x["ontology"]["timing_boundaries"] if row["boundary_id"] == "closed_loop").update(end_event=""))
    attempt("rename_ii_as_latency", "G07_latency_and_ii_are_not_interchangeable_or_implicitly_composed", lambda x: next(row for row in x["ontology"]["metrics"] if row["metric_id"] == "initiation_interval_ns").update(family="timing"))
    attempt("erase_power_activity", "G08_resource_dimensions_require_device_primitive_tool_stage_or_activity_qualifiers", lambda x: next(row for row in x["ontology"]["resource_dimensions"] if row["resource_id"] == "power")["required_qualifiers"].remove("activity"))
    attempt("move_aqec_method_to_decoder", "G09_t6_16_1_all_methods_land_in_exactly_one_known_lane", lambda x: x["source_audit_summary"].update(method_lane_ids={"single_mode_decoder"}))
    attempt("drop_source_metric_mapping", "G10_every_t6_16_1_metric_is_mapped_or_explicitly_rejected", lambda x: x["source_metric_crosswalk"].pop())
    attempt("accept_invalid_state_imputation", "G11_observation_validator_rejects_wrong_lane_and_state_imputation", lambda x: x["contract_self_tests"]["invalid_cases"][0].update(rejected=False))
    attempt("accept_cross_denominator", "G12_comparator_requires_same_lane_metric_denominator_statistic_boundary_and_task_signature", lambda x: x["contract_self_tests"]["comparison_cases"][0].update(rejected=False))
    attempt("add_global_score", "G13_global_score_and_cross_family_raw_ranking_are_absent_and_prohibited", lambda x: x["ranking_policy"].update(global_score="WEIGHTED_SUM"))
    attempt("accept_opinion_complexity", "G14_qualitative_complexity_requires_recomputable_numeric_or_source_asymptotic_basis", lambda x: next(row for row in x["contract_self_tests"]["invalid_cases"] if row["case"] == "unsupported_qualitative_complexity").update(rejected=False))
    attempt("truncate_parent_hash", "G15_parent_contracts_and_source_audit_are_hash_bound_without_claim_promotion", lambda x: x["bindings"]["t6_16_1"].update(sha256="0"))
    attempt("forge_mutation_count", "G16_source_data_and_targeted_semantic_mutations_are_complete", lambda x: x.update(semantic_mutation_audit={"count":16,"detected":15,"cases":[]}))
    return {"count":len(cases), "detected":sum(row["rejected"] for row in cases), "cases":cases}


def build_report() -> dict[str, Any]:
    ontology = _load(ONTOLOGY)
    source = _load(SOURCE_AUDIT)
    t511, t651, t687 = _load(T511), _load(T651), _load(T687)
    source_metric_count = sum(len(method["metrics"]) for method in source["methods"])
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ontology": ontology,
        "source_audit_summary": {
            "method_count": len(source["methods"]),
            "metric_count": source_metric_count,
            "method_lane_ids": {method["lane_id"] for method in source["methods"]},
            "claim_count": len(source["claim_audit"]),
        },
        "source_metric_crosswalk": _crosswalk(source, ontology),
        "contract_self_tests": _contract_self_tests(ontology),
        "ranking_policy": {
            "global_score": "PROHIBITED",
            "cross_lane_raw_ranking": "PROHIBITED",
            "same_lane_exact_signature_only": True,
            "qualitative_complexity_without_numeric_or_asymptotic_locator": "PROHIBITED",
        },
        "parent_contracts": {
            "T5.1.1": t511["status"],
            "T6.5.1": t651["verdict"],
            "T6.8.7": t687["verdict"],
            "T6.16.1": source["verdict"],
        },
    }
    # Sets are useful internally but JSON needs stable arrays.
    report["source_audit_summary"]["method_lane_ids"] = sorted(report["source_audit_summary"]["method_lane_ids"])
    _write_csv(report)
    report["source_data"] = {"path":_relative(SOURCE_CSV), "sha256":_sha256(SOURCE_CSV), "rows":sum(1 for _ in SOURCE_CSV.open(encoding="utf-8"))-1}
    report["bindings"] = {
        "implementation": _binding(Path(__file__)),
        "ontology": _binding(ONTOLOGY),
        "t6_16_1": _binding(SOURCE_AUDIT),
        "t5_1_1": _binding(T511),
        "t6_5_1": _binding(T651),
        "t6_8_7": _binding(T687),
        "source_csv": _binding(SOURCE_CSV),
    }
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    failed = [name for name, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {"passed":len(report["gates"])-len(failed), "failed":failed}
    report["verdict"] = "PASS_FAIL_CLOSED_COMPARISON_ONTOLOGY" if not failed else "FAIL_COMPARISON_ONTOLOGY"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    if dict(report["gates"]) != gates:
        raise ValueError("stored gates do not match recomputation")
    failed = [name for name, passed in gates.items() if not passed]
    expected_summary = {"passed":len(gates)-len(failed), "failed":failed}
    expected_verdict = "PASS_FAIL_CLOSED_COMPARISON_ONTOLOGY" if not failed else "FAIL_COMPARISON_ONTOLOGY"
    if report["gate_summary"] != expected_summary or report["verdict"] != expected_verdict:
        raise ValueError("stored gate summary/verdict does not match recomputation")


def write_markdown(report: Mapping[str, Any], path: Path = DEFAULT_MARKDOWN) -> None:
    ontology = report["ontology"]
    lines = [
        "# T6.16.2 comparison-lane / metric / timing-resource ontology",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- lanes/metrics/timing/resources/states：`{len(ontology['lanes'])}/{len(ontology['metrics'])}/{len(ontology['timing_boundaries'])}/{len(ontology['resource_dimensions'])}/{len(ontology['value_states'])}`",
        f"- T6.16.1 metric crosswalk：`{len(report['source_metric_crosswalk'])}` rows；只有 Wang 未定义 denominator 的 decoding-rate prose 被显式拒绝。",
        "- 核心原则：只有同 lane、同 metric、同 denominator/statistic/timing boundary、完全相同 13-field task signature 且状态可排名时，才允许 raw comparison。",
        "",
        "## 六条 lane",
        "",
        "| lane | decision object | ranking unit |",
        "| --- | --- | --- |",
    ]
    for lane in ontology["lanes"]:
        lines.append(f"| `{lane['lane_id']}` | {lane['decision_object']} | {lane['ranking_unit']} |")
    lines += [
        "",
        "## 状态语义",
        "",
        "`NULL_NOT_REPORTED` 是适用但没有值；`N_A_NOT_APPLICABLE` 是不适用；`FAILED` 是已执行但未过门；`NEGATIVE` 是有效 NO-GO。四者 value 必须为 null、不可排名，也不能填 0。literature/reproduced/estimate/measured 仍需分开。",
        "",
        "## timing/resource 边界",
        "",
        "- `decoder_core`、`update_compute`、`transport`、`source_to_action`、`closed_loop`、`initiation_interval` 分开；II 是吞吐，不是 latency。",
        "- 允许声明 `source_to_action` 可由明确的 transport+core 路径构成、closed-loop 可包含 source-to-action，但禁止在缺少边界事件和组件测量时自行相加。",
        "- LUT/FF/BRAM/DSP 必须带 device/primitive/tool/stage/seed-profile；power 还必须带 voltage/clock/activity/method。",
        "",
        "## fail-closed 自检",
        "",
        "wrong-lane、null/N/A/failed 填零、无 boundary latency、无一手依据定性复杂度、跨 denominator、跨 family、core-vs-closed-loop、null 排名均被拒绝；模块没有 global-score API。",
        "",
        "## 产物",
        "",
        "- `configs/literature/t6_16_2_comparison_ontology.json`",
        "- `docs/t6_16_2_comparison_ontology.json`",
        f"- `{report['source_data']['path']}`（{report['source_data']['rows']} rows）",
    ]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build T6.16.2 fail-closed comparison ontology")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report()
    verify_report(report)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps({"verdict":report["verdict"], "gate_summary":report["gate_summary"], "source_rows":report["source_data"]["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
