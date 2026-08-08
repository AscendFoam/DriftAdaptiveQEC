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
TASK_ID = "T6.8.5"
SCHEMA_VERSION = "t6.8.5-gqf-route-a-matched-comparison-gate-v1"
PARENT = ROOT / "docs" / "t6_8_4_gqf_paper_exact_reproduction.json"
SOURCE_CSV = ROOT / "docs" / "t6_8_5_gqf_route_a_matched_comparison_gate_source_data.csv"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_8_5_gqf_route_a_matched_comparison_gate.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    parent = report["parent_qualification"]
    metrics = report["matched_comparison_metrics"]
    return {
        "G01_parent_exact_report_is_hash_bound_no_go": parent["verdict"] == "COMPLETE_GQF_PAPER_EXACT_ATTEMPT_NO_GO_SOURCE_INCOMPLETE" and parent["exact_passed"] == 0 and parent["exact_failed"] == 15,
        "G02_all_mandatory_prerequisites_are_evaluated": len(report["prerequisite_ledger"]) == 8 and all(row["passed"] is False for row in report["prerequisite_ledger"]),
        "G03_negative_branch_is_selected_before_comparison": report["execution_branch"] == "INELIGIBLE_NEGATIVE_BRANCH_NO_MATCHED_RUN",
        "G04_no_unmatched_result_artifact_is_created": report["comparison_run_manifest"] is None and report["comparison_raw_data"] is None,
        "G05_all_performance_and_cost_metrics_remain_null": all(value is None for value in metrics.values()),
        "G06_independent_project_teacher_student_is_not_substituted": report["non_substitution"]["project_T4_4_or_T2_3_7_used_as_official_NMF"] is False,
        "G07_claims_and_downstream_eligibility_fail_closed": report["claim_boundary"] == {"same_GQF_lifetime_comparison": "NOT_RUN_INELIGIBLE", "paired_lifetime_improvement": "UNDEFINED", "surpass_puviani_NMF": "PROHIBITED", "retention_compression_safety_extension": "NOT_ESTABLISHED_IN_OFFICIAL_GQF"},
        "G08_recovery_conditions_are_explicit": len(report["recovery_conditions"]) == 7 and all(item["status"] == "MISSING" for item in report["recovery_conditions"]),
        "G09_live_inputs_and_outputs_are_hash_bound": all(len(item["sha256"]) == 64 for item in report["bindings"].values()),
        "G10_target_specific_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 10,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 10, "detected": 10, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("promote_parent", "G01_parent_exact_report_is_hash_bound_no_go", lambda x: x["parent_qualification"].update(exact_passed=15, exact_failed=0))
    attempt("forge_prerequisite", "G02_all_mandatory_prerequisites_are_evaluated", lambda x: x["prerequisite_ledger"][0].update(passed=True))
    attempt("select_matched_branch", "G03_negative_branch_is_selected_before_comparison", lambda x: x.update(execution_branch="RUN_MATCHED_COMPARISON"))
    attempt("invent_manifest", "G04_no_unmatched_result_artifact_is_created", lambda x: x.update(comparison_run_manifest="fake.json"))
    attempt("invent_lifetime", "G05_all_performance_and_cost_metrics_remain_null", lambda x: x["matched_comparison_metrics"].update(route_a_T_ch=1000.0))
    attempt("substitute_project_teacher", "G06_independent_project_teacher_student_is_not_substituted", lambda x: x["non_substitution"].update(project_T4_4_or_T2_3_7_used_as_official_NMF=True))
    attempt("claim_surpass", "G07_claims_and_downstream_eligibility_fail_closed", lambda x: x["claim_boundary"].update(surpass_puviani_NMF="ESTABLISHED"))
    attempt("hide_recovery_gap", "G08_recovery_conditions_are_explicit", lambda x: x.update(recovery_conditions=[]))
    attempt("truncate_hash", "G09_live_inputs_and_outputs_are_hash_bound", lambda x: x["bindings"]["source_csv"].update(sha256="0"))
    attempt("forge_mutation_count", "G10_target_specific_semantic_mutations_fail_closed", lambda x: x.update(semantic_mutation_audit={"count": 10, "detected": 9, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    prerequisites = [
        ("paper_exact_reproduction", parent["exact_reproduction_status"] == "PASS_EXACT", "T6.8.4 is NO_GO_SOURCE_INCOMPLETE"),
        ("official_NMF_checkpoint", False, "no official checkpoint published"),
        ("official_MF_checkpoint", False, "no official checkpoint published"),
        ("official_agent_seeds", False, "20 seeds absent"),
        ("paper_matching_architecture", False, "paper/source architecture mismatch"),
        ("six_state_1000_cycle_evaluator", False, "official evaluator/fitter absent"),
        ("matched_training_search_budget", False, "official agent training/search ledger absent"),
        ("current_full_GQF_accelerator", False, "current GPU path unqualified cuSolver fatal"),
    ]
    prerequisite_ledger = [
        {"prerequisite": name, "passed": bool(passed), "reason": reason}
        for name, passed, reason in prerequisites
    ]
    rows = [
        {
            "prerequisite": row["prerequisite"],
            "passed": str(row["passed"]).lower(),
            "reason": row["reason"],
            "execution_branch": "INELIGIBLE_NEGATIVE_BRANCH_NO_MATCHED_RUN",
        }
        for row in prerequisite_ledger
    ]
    with SOURCE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "official_NMF_T_ch": None,
        "route_a_T_ch": None,
        "paired_lifetime_improvement": None,
        "paired_lifetime_improvement_95pct_LCB": None,
        "gain_retention": None,
        "route_a_parameters": None,
        "route_a_MACs": None,
        "route_a_memory_bytes": None,
        "official_NMF_parameters": None,
        "official_NMF_MACs": None,
        "official_NMF_memory_bytes": None,
        "fallback_rate": None,
        "unsafe_action_rate": None,
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_qualification": {
            "path": _relative(PARENT),
            "sha256": _sha256(PARENT),
            "verdict": parent["verdict"],
            "exact_passed": parent["exact_qualification"]["passed"],
            "exact_failed": parent["exact_qualification"]["failed"],
            "t6_8_5_eligible": parent["t6_8_5_eligible"],
        },
        "prerequisite_ledger": prerequisite_ledger,
        "execution_branch": "INELIGIBLE_NEGATIVE_BRANCH_NO_MATCHED_RUN",
        "comparison_run_manifest": None,
        "comparison_raw_data": None,
        "matched_comparison_metrics": metrics,
        "non_substitution": {
            "project_T4_4_or_T2_3_7_used_as_official_NMF": False,
            "reason": "Independent project teacher/student evidence uses a different simulator and cannot repair missing official GQF artifacts.",
        },
        "claim_boundary": {
            "same_GQF_lifetime_comparison": "NOT_RUN_INELIGIBLE",
            "paired_lifetime_improvement": "UNDEFINED",
            "surpass_puviani_NMF": "PROHIBITED",
            "retention_compression_safety_extension": "NOT_ESTABLISHED_IN_OFFICIAL_GQF",
        },
        "recovery_conditions": [
            {"item": "paper-matching official RNN/MF implementation", "status": "MISSING"},
            {"item": "20 official agent checkpoints and seeds", "status": "MISSING"},
            {"item": "official selection ledger", "status": "MISSING"},
            {"item": "six-state 1000-cycle raw evaluation", "status": "MISSING"},
            {"item": "complete lifetime targets and fit protocol", "status": "MISSING"},
            {"item": "qualified full GQF accelerator or feasible exact compute", "status": "MISSING"},
            {"item": "pre-registered common training/search/selection budget", "status": "MISSING"},
        ],
        "bindings": {
            "implementation": {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
            "parent": {"path": _relative(PARENT), "sha256": _sha256(PARENT)},
            "source_csv": {"path": _relative(SOURCE_CSV), "sha256": _sha256(SOURCE_CSV)},
        },
    }
    report["semantic_mutation_audit"] = {"count": 10, "detected": 10, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": sum(not value for value in report["gates"].values())}
    report["verdict"] = "COMPLETE_T6_8_5_INELIGIBLE_NEGATIVE_BRANCH" if all(report["gates"].values()) else "FAIL_T6_8_5_GATE_INTEGRITY"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    expected = "COMPLETE_T6_8_5_INELIGIBLE_NEGATIVE_BRANCH" if all(gates.values()) else "FAIL_T6_8_5_GATE_INTEGRITY"
    if report.get("gates") != gates or report.get("verdict") != expected or not all(gates.values()):
        raise ValueError("T6.8.5 gates/verdict do not recompute")
    for item in report["bindings"].values():
        path = ROOT / item["path"]
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"T6.8.5 bound artifact drifted: {item['path']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    report = build_report()
    args.artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_report(json.loads(args.artifact.read_text(encoding="utf-8")))
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "branch": report["execution_branch"], "surpass": report["claim_boundary"]["surpass_puviani_NMF"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
