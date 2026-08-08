from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.16.3"
SCHEMA_VERSION = "t6.16.3-secondary-experiment-preregistration-v1"
CONFIG = ROOT / "configs" / "literature" / "t6_16_3_secondary_preregistration.json"
SOURCE_AUDIT = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"
ONTOLOGY = ROOT / "docs" / "t6_16_2_comparison_ontology.json"
T6101 = ROOT / "docs" / "t6_10_1_causal_headroom.json"
T6155 = ROOT / "docs" / "t6_15_5_route_a_v5_final_evidence_gate.json"
T693 = ROOT / "docs" / "t6_9_3_route_a_final_evidence_gate.json"
BOARD = ROOT / "docs" / "new_task_board.md"
SOURCE_CSV = ROOT / "docs" / "t6_16_3_secondary_preregistration_source_data.csv"
DEFAULT_REPORT = ROOT / "docs" / "t6_16_3_secondary_preregistration.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "secondary_experiment_preregistration.md"

EXPECTED_TASKS = {"T6.17.1", "T6.17.2", "T6.17.3", "T6.18.1", "T6.18.2", "T6.18.3", "T6.19.1", "T6.19.2", "T6.19.3"}
EXPERIMENT_FIELDS = {
    "experiment_id", "task_id", "lane_id", "execution_type", "entry_gate", "sources",
    "source_code", "environment", "adapter", "split", "seeds", "config", "sample_size",
    "pairing", "primary_metrics", "statistics", "stopping_rule", "runtime_budget",
    "failure_branches", "allowed_evidence", "forbidden_actions",
}
EVIDENCE_GRADES = {"LITERATURE_ONLY", "OFFICIAL_CODE_REPRODUCTION", "PROJECT_NATIVE_MATCHED", "INELIGIBLE", "BLOCKED", "NEGATIVE"}
IMMUTABLE_PARENT_PATHS = {
    "t3_2_8_aqec_project": ROOT / "docs" / "t3_2_8_autonomous_sbs_wallclock_validation.json",
    "t4_4_5_student_freeze": ROOT / "docs" / "t4_4_5_teacher_student_branch_freeze.json",
    "t5_5_2_preboard_pr": ROOT / "docs" / "t5_5_2_target_device_synthesis.json",
    "t5_5_4_student_hardware": ROOT / "docs" / "t5_5_4_gru_student_hardware_feasibility.json",
    "t6_8_3_gqf_intake": ROOT / "docs" / "t6_8_3_gqf_official_intake.json",
    "t6_8_4_gqf_exact": ROOT / "docs" / "t6_8_4_gqf_paper_exact_reproduction.json",
    "t6_8_5_gqf_matched": ROOT / "docs" / "t6_8_5_gqf_route_a_matched_comparison_gate.json",
    "t6_8_6_fpga_normalization": ROOT / "docs" / "t6_8_6_fpga_decoder_normalization.json",
}
V5_ABSENT_PATTERNS = (
    "docs/t6_13_3*", "docs/t6_14_1*", "docs/t6_14_2*", "docs/t6_14_3*",
    "docs/t6_15_1*", "docs/t6_15_2*", "docs/t6_15_3*", "docs/t6_15_4*",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_statuses(board_text: str) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for line in board_text.splitlines():
        match = re.match(r"\|\s*(T[^| ]+)\s*\|\s*([^|]+?)\s*\|", line)
        if match:
            statuses[match.group(1)] = match.group(2).strip()
    return statuses


def _v5_semantic(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "execution_path", "headroom_recomputation", "dropped_tasks", "v5_downstream_outputs_found",
        "formal_access", "claim_registry", "phase6c_permission", "measured_hardware_claim", "status", "verdict",
    )
    return {key: report[key] for key in keys}


def _source_semantic(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in ("scope", "sources", "methods", "claim_audit", "derived_evidence", "comparison_policy", "verdict")}


def _ontology_semantic(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in ("ontology", "source_metric_crosswalk", "ranking_policy", "parent_contracts", "verdict")}


def _t693_semantic(report: Mapping[str, Any]) -> dict[str, Any]:
    # Freeze scientific conclusions while permitting the upstream generator to
    # refresh provenance hashes/timestamps after unrelated board bookkeeping.
    # T6.9.3 has no ``state_legend`` field; the paper/figure decisions are the
    # fail-closed objects that Phase 6C must not upgrade.
    return {
        key: report[key]
        for key in ("aggregation_policy", "claims", "paper_decision", "figure_table_plan", "verdict")
    }


def _environment_snapshot() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "pytest", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "tools": {name: shutil.which(name) for name in ("julia", "yosys", "nextpnr-gowin", "c++")},
        "missing_tools_do_not_invalidate_preregistration": True,
        "missing_required_tool_at_execution_uses_frozen_blocked_or_partial_branch": True,
    }


def _source_record_locks(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": row["source_id"],
            "record_sha256": _canonical_sha256(row),
            "version": row["version"],
            "code_url": row["code_url"],
            "code_commit": row["code_commit"],
            "code_license": row["code_license"],
        }
        for row in source["sources"]
    ]


def _absence_proof() -> dict[str, Any]:
    matches: dict[str, list[str]] = {}
    for pattern in V5_ABSENT_PATTERNS:
        matches[pattern] = sorted(_relative(path) for path in ROOT.glob(pattern))
    return {"patterns": matches, "all_absent": all(not rows for rows in matches.values())}


def _phase6b_lock() -> dict[str, Any]:
    v5 = _load(T6155)
    t6101 = _load(T6101)
    t693 = _load(T693)
    board_text = BOARD.read_text(encoding="utf-8")
    statuses = _task_statuses(board_text)
    dropped = [f"T6.{major}.{minor}" for major, end in ((10, 3), (11, 4), (12, 4), (13, 3), (14, 3), (15, 4)) for minor in range(2 if major == 10 else 1, end + 1)]
    # T6.10.2--T6.15.4 is the exact 20-task set stored by T6.15.5.
    dropped = list(v5["dropped_tasks"])
    semantic = _v5_semantic(v5)
    return {
        "t6_10_1": {
            "artifact": _binding(T6101),
            "verdict": t6101["verdict"],
            # T6.10.1 predates the final V5 schema and stores these thresholds
            # under a different contract.  The authoritative frozen values are
            # the recomputed gates in T6.15.5, so bind to those values directly.
            "router_gate": v5["headroom_recomputation"]["router_gate"],
            "action_gate": v5["headroom_recomputation"]["action_gate"],
        },
        "t6_15_5": {"initial_artifact":_binding(T6155), "semantic_sha256":_canonical_sha256(semantic), "semantic_snapshot":semantic},
        "t6_9_3": {"initial_artifact":_binding(T693), "semantic_sha256":_canonical_sha256(_t693_semantic(t693)), "verdict":t693["verdict"]},
        "board_status_snapshot": {task:statuses.get(task) for task in dropped},
        "v5_absence_proof": _absence_proof(),
        "frozen_main_contract": {
            "v5_relative_ler_gate": 0.10,
            "incremental_action_space_gate": 0.12,
            "v5_comparator_selection": "NONE_EARLY_STOP",
            "v5_formal_split": "NOT_CREATED",
            "v5_tail_endpoints": ["step_calibration_shift_worst_window_ler", "telegraph_drift_worst_window_ler"],
            "v5_tail_state": "NOT_RUN_EARLY_STOP",
            "phase6c_may_change_phase6b": False,
            "phase6c_may_rescue_v5": False,
            "phase6c_may_unblock_t6_9_2": False,
        },
    }


def verify_live_phase6b_lock(lock: Mapping[str, Any]) -> dict[str, bool]:
    v5 = _load(T6155)
    t693 = _load(T693)
    statuses = _task_statuses(BOARD.read_text(encoding="utf-8"))
    return {
        "t6_10_1_exact_artifact_unchanged": _sha256(T6101) == lock["t6_10_1"]["artifact"]["sha256"],
        "t6_15_5_semantics_unchanged": _canonical_sha256(_v5_semantic(v5)) == lock["t6_15_5"]["semantic_sha256"],
        "t6_9_3_semantics_unchanged": _canonical_sha256(_t693_semantic(t693)) == lock["t6_9_3"]["semantic_sha256"],
        "all_20_conditional_tasks_remain_dropped": len(lock["board_status_snapshot"]) == 20 and all(statuses.get(task) == "Dropped" for task in lock["board_status_snapshot"]),
        "v5_downstream_outputs_remain_absent": _absence_proof()["all_absent"],
        "phase6c_permissions_remain_readonly": v5["phase6c_permission"] == {"allowed":True,"mode":"READ_ONLY_AUXILIARY_COMPARISONS","may_modify_phase6b_verdict":False,"may_rescue_v5_claim":False},
    }


def _parent_locks() -> dict[str, Any]:
    return {name:_binding(path) for name, path in IMMUTABLE_PARENT_PATHS.items()}


def verify_live_input_locks(report: Mapping[str, Any]) -> dict[str, bool]:
    """Re-evaluate frozen inputs without rejecting provenance-only rebuilds.

    Source-audit and ontology generators legitimately refresh timestamps and
    parent byte hashes as the task board advances, so they are locked by their
    scientific semantic projections.  The preregistration config,
    implementation, Source Data and immutable historical parents are exact-byte
    inputs and must remain unchanged.
    """
    source = _load(SOURCE_AUDIT)
    ontology = _load(ONTOLOGY)
    checks = {
        "implementation_exact_unchanged": _sha256(Path(__file__)) == report["bindings"]["implementation"]["sha256"],
        "config_exact_unchanged": _sha256(CONFIG) == report["bindings"]["config"]["sha256"],
        "source_csv_exact_unchanged": _sha256(SOURCE_CSV) == report["bindings"]["source_csv"]["sha256"],
        "source_audit_semantics_unchanged": _canonical_sha256(_source_semantic(source)) == report["source_audit_semantic_lock"]["semantic_sha256"],
        "ontology_semantics_unchanged": _canonical_sha256(_ontology_semantic(ontology)) == report["ontology_summary"]["semantic_sha256"],
        "source_records_unchanged": _source_record_locks(source) == list(report["source_record_locks"]),
    }
    for name, path in IMMUTABLE_PARENT_PATHS.items():
        checks[f"parent_{name}_exact_unchanged"] = _sha256(path) == report["immutable_parent_locks"][name]["sha256"]
    return checks


def _write_csv(report: Mapping[str, Any]) -> None:
    fields = ["record_type", "record_id", "task_id", "lane_id", "seed", "source_id", "state", "sha256", "details"]
    rows: list[dict[str, Any]] = []
    for experiment in report["experiments"]:
        rows.append({"record_type":"experiment", "record_id":experiment["experiment_id"], "task_id":experiment["task_id"], "lane_id":experiment["lane_id"], "seed":None, "source_id":";".join(experiment["sources"]), "state":experiment["execution_type"], "sha256":_canonical_sha256(experiment), "details":experiment["entry_gate"]})
        for seed in experiment["seeds"]["values"]:
            rows.append({"record_type":"seed", "record_id":f"{experiment['experiment_id']}:{seed}", "task_id":experiment["task_id"], "lane_id":experiment["lane_id"], "seed":seed, "source_id":None, "state":experiment["seeds"]["kind"], "sha256":None, "details":experiment["seeds"]["namespace"]})
    for source in report["source_record_locks"]:
        rows.append({"record_type":"source_lock", "record_id":source["source_id"], "task_id":None, "lane_id":None, "seed":None, "source_id":source["source_id"], "state":"FROZEN", "sha256":source["record_sha256"], "details":source["version"]})
    for name, binding in report["immutable_parent_locks"].items():
        rows.append({"record_type":"parent_lock", "record_id":name, "task_id":None, "lane_id":None, "seed":None, "source_id":None, "state":"FROZEN_EXACT", "sha256":binding["sha256"], "details":binding["path"]})
    for name, passed in report["live_phase6b_lock_checks"].items():
        rows.append({"record_type":"phase6b_lock", "record_id":name, "task_id":"T6.15.5", "lane_id":None, "seed":None, "source_id":None, "state":"PASS" if passed else "FAIL", "sha256":None, "details":"semantic lock tolerates regenerated metadata but not scientific-state changes"})
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    experiments = report["experiments"]
    ontology = report["ontology_summary"]
    source_ids = {row["source_id"] for row in report["source_record_locks"]}
    metric_lanes = ontology["metric_allowed_lanes"]
    all_numeric_seeds = [seed for exp in experiments if exp["seeds"]["namespace"] == "phase6c-secondary-v1" for seed in exp["seeds"]["values"]]
    exp_by_task = {row["task_id"]:row for row in experiments}
    no_favorable = all("favorable" not in row["stopping_rule"].lower() or "never" in row["stopping_rule"].lower() for row in experiments)
    return {
        "G01_nine_downstream_tasks_have_exact_complete_preregistration_schema": len(experiments) == 9 and {row["task_id"] for row in experiments} == EXPECTED_TASKS and all(set(row) == EXPERIMENT_FIELDS for row in experiments),
        "G02_every_experiment_uses_one_known_ontology_lane_and_allowed_metrics": all(row["lane_id"] in ontology["lane_ids"] and all(row["lane_id"] in metric_lanes[metric] for metric in row["primary_metrics"]) for row in experiments),
        "G03_source_records_versions_commits_licenses_and_code_absence_are_frozen": all(set(row["sources"]) <= source_ids and row["source_code"]["state"] for row in experiments) and exp_by_task["T6.18.2"]["source_code"] == {"url":"https://github.com/amazon-science/LatticeAlgorithms.jl","commit":"01f9bf1f6970b3e229b43aac9da3325c75518db8","license":"Apache-2.0","state":"PINNED_NOT_YET_IMPORTED"},
        "G04_environment_and_missing_tool_failure_semantics_are_explicit": report["environment"]["missing_tools_do_not_invalidate_preregistration"] is True and report["environment"]["missing_required_tool_at_execution_uses_frozen_blocked_or_partial_branch"] is True and all(row["environment"]["runtime"] and isinstance(row["environment"]["required_external_tools"], list) for row in experiments),
        "G05_adapters_are_planned_or_existing_but_none_are_claimed_executed": all(row["adapter"]["path"] and row["adapter"]["decision_object"] and any(token in row["adapter"]["state"] for token in ("PLANNED","EXISTING","CONDITIONAL")) for row in experiments),
        "G06_secondary_seed_namespace_is_unique_disjoint_and_no_selection_reuse": len(all_numeric_seeds) == len(set(all_numeric_seeds)) and all(61_710_001 <= seed <= 61_839_999 for seed in all_numeric_seeds) and exp_by_task["T6.17.3"]["config"]["training_allowed"] is False and exp_by_task["T6.17.3"]["config"]["checkpoint_reselection_allowed"] is False,
        "G07_sample_size_crn_pairing_raw_counts_and_cluster_resampling_are_frozen": all(row["sample_size"] and row["pairing"] and row["statistics"] for row in experiments) and report["statistics"]["paired_bootstrap_resamples"] == 20000 and "seed cluster" in report["statistics"]["resampling_unit"] and "raw counts" in report["statistics"]["raw_evidence"],
        "G08_multiplicity_is_within_task_only_and_never_cross_lane": "within each task" in report["statistics"]["multiplicity"] and "never pool" in report["statistics"]["multiplicity"],
        "G09_stopping_rules_are_performance_independent_and_runtime_bounded": no_favorable and all(row["runtime_budget"]["wall_clock_seconds"] > 0 and row["runtime_budget"]["memory_gib"] > 0 and row["runtime_budget"]["failure_on_exceed"] in row["failure_branches"] for row in experiments),
        "G10_every_experiment_has_predeclared_partial_blocked_ineligible_or_negative_failure": all(row["failure_branches"] and any(any(token in branch for token in ("PARTIAL","BLOCKED","INELIGIBLE","NEGATIVE","NOT_RUN","N_A","FAIL")) for branch in row["failure_branches"]) for row in experiments),
        "G11_conditional_multimode_extension_stops_without_rescue_task": exp_by_task["T6.18.3"]["entry_gate"].endswith("NOT_RUN_SCOPE_GATE") and "NOT_RUN_SCOPE_GATE" in exp_by_task["T6.18.3"]["failure_branches"] and any("rescue task" in action for action in exp_by_task["T6.18.3"]["forbidden_actions"]),
        "G12_hardware_profiles_keep_absent_rtl_na_and_measured_fields_null": exp_by_task["T6.19.1"]["config"]["board_measured_fields"] == "NULL_UNTIL_T6.9.2" and "N_A_NO_RTL" in exp_by_task["T6.19.1"]["failure_branches"] and any("demo RTL" in action for action in exp_by_task["T6.19.1"]["forbidden_actions"]),
        "G13_phase6b_semantic_lock_and_v5_absence_proof_are_live": all(report["live_phase6b_lock_checks"].values()) and report["phase6b_lock"]["v5_absence_proof"]["all_absent"] and len(report["phase6b_lock"]["board_status_snapshot"]) == 20,
        "G14_phase6b_10pct_action_tail_comparator_and_permissions_are_immutable": report["phase6b_lock"]["frozen_main_contract"] == {"v5_relative_ler_gate":0.10,"incremental_action_space_gate":0.12,"v5_comparator_selection":"NONE_EARLY_STOP","v5_formal_split":"NOT_CREATED","v5_tail_endpoints":["step_calibration_shift_worst_window_ler","telegraph_drift_worst_window_ler"],"v5_tail_state":"NOT_RUN_EARLY_STOP","phase6c_may_change_phase6b":False,"phase6c_may_rescue_v5":False,"phase6c_may_unblock_t6_9_2":False},
        "G15_evidence_grades_are_closed_and_atlas_cannot_promote_main_or_hardware": all(set(row["allowed_evidence"]) <= EVIDENCE_GRADES for row in experiments) and exp_by_task["T6.19.3"]["config"]["global_score"] is False and exp_by_task["T6.19.3"]["config"]["may_upgrade_phase6b"] is False and exp_by_task["T6.19.3"]["config"]["may_unblock_t6_9_2"] is False,
        "G16_all_parent_source_ontology_config_and_output_data_are_hash_bound": len(report["immutable_parent_locks"]) == len(IMMUTABLE_PARENT_PATHS) and all(len(row["sha256"]) == 64 for row in report["immutable_parent_locks"].values()) and set(report["bindings"]) == {"implementation","config","source_audit_initial","ontology_initial","source_csv"} and all(len(row["sha256"]) == 64 for row in report["bindings"].values()) and all(report["live_input_lock_checks"].values()),
        "G17_source_data_and_targeted_mutations_are_complete": report["source_data"]["rows"] >= 140 and len(report["source_data"]["sha256"]) == 64 and report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 17 and all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]),
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count":17,"detected":17,"cases":[]}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case":name,"target_gate":gate,"rejected":rejected})
    attempt("drop_task_prereg", "G01_nine_downstream_tasks_have_exact_complete_preregistration_schema", lambda x: x["experiments"].pop())
    attempt("wrong_lane_metric", "G02_every_experiment_uses_one_known_ontology_lane_and_allowed_metrics", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.18.1")["primary_metrics"].append("p_X"))
    attempt("change_official_cpd_commit", "G03_source_records_versions_commits_licenses_and_code_absence_are_frozen", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.18.2")["source_code"].update(commit="0"*40))
    attempt("missing_tool_means_silent_substitute", "G04_environment_and_missing_tool_failure_semantics_are_explicit", lambda x: x["environment"].update(missing_required_tool_at_execution_uses_frozen_blocked_or_partial_branch=False))
    attempt("claim_adapter_executed", "G05_adapters_are_planned_or_existing_but_none_are_claimed_executed", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.17.2")["adapter"].update(state="EXECUTED_PASS"))
    attempt("reuse_seed_between_tasks", "G06_secondary_seed_namespace_is_unique_disjoint_and_no_selection_reuse", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.18.1")["seeds"]["values"].append(61720001))
    attempt("individual_round_bootstrap", "G07_sample_size_crn_pairing_raw_counts_and_cluster_resampling_are_frozen", lambda x: x["statistics"].update(resampling_unit="individual rounds"))
    attempt("cross_lane_win_count", "G08_multiplicity_is_within_task_only_and_never_cross_lane", lambda x: x["statistics"].update(multiplicity="global wins across lanes"))
    attempt("remove_runtime_failure_branch", "G09_stopping_rules_are_performance_independent_and_runtime_bounded", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.17.1")["failure_branches"].remove("PARTIAL_RUNTIME_BUDGET_EXCEEDED"))
    attempt("erase_all_failure_branches", "G10_every_experiment_has_predeclared_partial_blocked_ineligible_or_negative_failure", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.17.3").update(failure_branches=[]))
    attempt("rescue_multimode_after_failed_entry", "G11_conditional_multimode_extension_stops_without_rescue_task", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.18.3").update(entry_gate="run anyway"))
    attempt("invent_v5_rtl_profile", "G12_hardware_profiles_keep_absent_rtl_na_and_measured_fields_null", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.19.1")["config"].update(board_measured_fields="ZERO"))
    attempt("alter_v5_semantic_lock", "G13_phase6b_semantic_lock_and_v5_absence_proof_are_live", lambda x: x["live_phase6b_lock_checks"].update(t6_15_5_semantics_unchanged=False))
    attempt("lower_main_gate_after_no_go", "G14_phase6b_10pct_action_tail_comparator_and_permissions_are_immutable", lambda x: x["phase6b_lock"]["frozen_main_contract"].update(v5_relative_ler_gate=0.01))
    attempt("allow_atlas_to_upgrade_phase6b", "G15_evidence_grades_are_closed_and_atlas_cannot_promote_main_or_hardware", lambda x: next(row for row in x["experiments"] if row["task_id"]=="T6.19.3")["config"].update(may_upgrade_phase6b=True))
    attempt("truncate_parent_hash", "G16_all_parent_source_ontology_config_and_output_data_are_hash_bound", lambda x: x["immutable_parent_locks"]["t6_8_4_gqf_exact"].update(sha256="0"))
    attempt("forge_mutation_count", "G17_source_data_and_targeted_mutations_are_complete", lambda x: x.update(semantic_mutation_audit={"count":17,"detected":16,"cases":[]}))
    return {"count":len(cases),"detected":sum(row["rejected"] for row in cases),"cases":cases}


def build_report() -> dict[str, Any]:
    config = _load(CONFIG)
    source = _load(SOURCE_AUDIT)
    ontology_report = _load(ONTOLOGY)
    ontology = ontology_report["ontology"]
    metric_lanes = {row["metric_id"]:row["allowed_lanes"] for row in ontology["metrics"]}
    phase6b_lock = _phase6b_lock()
    report: dict[str, Any] = {
        "task_id":TASK_ID,
        "schema_version":SCHEMA_VERSION,
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "protocol_id":config["protocol_id"],
        "frozen_at":config["frozen_at"],
        "secondary_namespace":config["secondary_namespace"],
        "statistics":deepcopy(config["statistics"]),
        "experiments":deepcopy(config["experiments"]),
        "environment":_environment_snapshot(),
        "source_record_locks":_source_record_locks(source),
        "ontology_summary": {"lane_ids":[row["lane_id"] for row in ontology["lanes"]],"metric_allowed_lanes":metric_lanes,"semantic_sha256":_canonical_sha256(_ontology_semantic(ontology_report)),"initial_artifact_sha256":_sha256(ONTOLOGY)},
        "source_audit_semantic_lock": {"semantic_sha256":_canonical_sha256(_source_semantic(source)),"initial_artifact_sha256":_sha256(SOURCE_AUDIT)},
        "phase6b_lock":phase6b_lock,
        "live_phase6b_lock_checks":verify_live_phase6b_lock(phase6b_lock),
        "immutable_parent_locks":_parent_locks(),
    }
    _write_csv(report)
    report["source_data"] = {"path":_relative(SOURCE_CSV),"sha256":_sha256(SOURCE_CSV),"rows":sum(1 for _ in SOURCE_CSV.open(encoding="utf-8"))-1}
    report["bindings"] = {"implementation":_binding(Path(__file__)),"config":_binding(CONFIG),"source_audit_initial":_binding(SOURCE_AUDIT),"ontology_initial":_binding(ONTOLOGY),"source_csv":_binding(SOURCE_CSV)}
    report["live_input_lock_checks"] = verify_live_input_locks(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    failed = [name for name, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {"passed":len(report["gates"])-len(failed),"failed":failed}
    report["verdict"] = "PASS_PHASE6C_READONLY_SECONDARY_PREREGISTRATION" if not failed else "FAIL_PHASE6C_PREREGISTRATION"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    live = verify_live_phase6b_lock(report["phase6b_lock"])
    live_inputs = verify_live_input_locks(report)
    candidate = deepcopy(report)
    candidate["live_phase6b_lock_checks"] = live
    candidate["live_input_lock_checks"] = live_inputs
    gates = evaluate_gates(candidate)
    if dict(report["live_phase6b_lock_checks"]) != live:
        raise ValueError("stored Phase 6B live lock checks are stale")
    if dict(report["live_input_lock_checks"]) != live_inputs:
        raise ValueError("stored input live lock checks are stale")
    if dict(report["gates"]) != gates:
        raise ValueError("stored gates do not match recomputation")
    failed = [name for name, passed in gates.items() if not passed]
    expected_summary = {"passed":len(gates)-len(failed),"failed":failed}
    expected_verdict = "PASS_PHASE6C_READONLY_SECONDARY_PREREGISTRATION" if not failed else "FAIL_PHASE6C_PREREGISTRATION"
    if report["gate_summary"] != expected_summary or report["verdict"] != expected_verdict:
        raise ValueError("stored summary/verdict does not match recomputation")


def write_markdown(report: Mapping[str, Any], path: Path = DEFAULT_MARKDOWN) -> None:
    lines = [
        "# T6.16.3 Phase 6C 二级实验预注册与只读边界",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- downstream preregistrations：`{len(report['experiments'])}`；gates/mutations：`{report['gate_summary']['passed']}/17`、`{report['semantic_mutation_audit']['detected']}/17`",
        f"- secondary seed namespace：`{report['secondary_namespace']}`",
        "- Phase 6B byte hash 会因任务板元数据重建而变化，因此同时保存 initial artifact hash 与 scientific semantic hash；后续只允许元数据重建，不允许改变 verdict/gates/claims/dropped/absence。",
        "",
        "## 只读锁",
        "",
        "- V5 LER gate 固定 `10%`，incremental action-space gate 固定 `12%`；comparator=`NONE_EARLY_STOP`，formal split=`NOT_CREATED`。",
        "- step-calibration/telegraph worst-window endpoints 固定为 `NOT_RUN_EARLY_STOP`，不得用 Phase 6C secondary 数值代替。",
        "- T6.10.2--T6.15.4 共 20 项保持 Dropped；T6.13.3/T6.14.*/T6.15.1--4 output absence proof 为真。",
        "- Phase 6C 不得改变 T6.15.5、挽救 V5 或解锁 T6.9.2。",
        "",
        "## 二级实验清单",
        "",
        "| task | lane | execution | split/seeds | runtime cap | failure branches |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in report["experiments"]:
        seed_text = f"{row['split']['name']} / {len(row['seeds']['values'])} seeds"
        lines.append(f"| `{row['task_id']}` | `{row['lane_id']}` | `{row['execution_type']}` | {seed_text} | {row['runtime_budget']['wall_clock_seconds']} s | {', '.join(row['failure_branches'])} |")
    lines += [
        "",
        "## 统计与停止规则",
        "",
        "- paired bootstrap=20,000；threshold bootstrap=2,000；resampling unit 是 seed cluster/trajectory，不是相关 round。",
        "- Holm 只在单 task 预声明 endpoint family 内执行，不跨 lane 汇总 p-value、胜场或总分。",
        "- 保存 raw counts/denominator/seed/config/hash/all attempted cells；零事件报告 exact one-sided 95% upper bound。",
        "- 只允许 correctness/source/tool/runtime/entry gate 提前停止；不得因结果有利或不利而改候选、容差、size、sigma grid 或 endpoint。",
        "",
        "## 当前工具状态不是结果",
        "",
        f"- Julia：`{report['environment']['tools']['julia']}`；Yosys：`{report['environment']['tools']['yosys']}`；nextpnr-gowin：`{report['environment']['tools']['nextpnr-gowin']}`。缺失工具在对应 task 使用预注册 BLOCKED/PARTIAL 分支，不能用自写替代实现追求正值。",
        "",
        "## 产物",
        "",
        "- `configs/literature/t6_16_3_secondary_preregistration.json`",
        "- `docs/t6_16_3_secondary_preregistration.json`",
        f"- `{report['source_data']['path']}`（{report['source_data']['rows']} rows）",
    ]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 6C read-only secondary preregistration")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_report()
    verify_report(report)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    write_markdown(report, args.markdown)
    print(json.dumps({"verdict":report["verdict"],"gate_summary":report["gate_summary"],"source_rows":report["source_data"]["rows"]},ensure_ascii=False))


if __name__ == "__main__":
    main()
