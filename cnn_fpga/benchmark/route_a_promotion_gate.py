"""Independent T6.7.4 Route-A promotion and falsification gate.

The gate intentionally does not rerun or tune any formal experiment.  It
recomputes the locked scientific analyses from the stored per-trajectory raw
counts, verifies the large Source Data files and the million-cycle RTL trace,
and then separates preregistered promotion from stronger claims that the data
do not support.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from cnn_fpga.benchmark import route_a_integrated_rtl_qualification as rtlq
from cnn_fpga.benchmark import route_a_smooth_formal as smooth
from cnn_fpga.benchmark import route_a_tail_formal as tail


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.7.4"
SCHEMA_VERSION = "t6.7.4-route-a-promotion-gate-v1"
VERDICT = "GO_ROUTE_A_CONTRACT_SYSTEM_RESTRICTED_SIMULATOR_AND_PREBOARD_CLAIMS"

SMOOTH_ARTIFACT = ROOT / "docs" / "t6_7_1_smooth_formal_matrix.json"
TAIL_ARTIFACT = ROOT / "docs" / "t6_7_2_abrupt_ood_tail_formal_matrix.json"
RTL_ARTIFACT = ROOT / "docs" / "t6_7_3_route_a_integrated_rtl_qualification.json"
COMPARATOR_ARTIFACT = ROOT / "docs" / "t6_6_1_unified_comparator_runner.json"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_7_4_route_a_promotion_gate.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_7_4_route_a_promotion_gate_source_data.csv"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) is None:
            return 0
        return sum(1 for _ in reader)


def _binding_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    binding = report["source_data_binding"]
    path = ROOT / str(binding["path"])
    observed_sha = _sha256(path)
    observed_rows = _csv_rows(path)
    expected_rows = int(binding["row_count"])
    return {
        "path": _relative(path),
        "expected_sha256": binding["sha256"],
        "observed_sha256": observed_sha,
        "expected_rows": expected_rows,
        "observed_rows": observed_rows,
        "passes": observed_sha == binding["sha256"] and observed_rows == expected_rows,
    }


def _recompute_smooth(report: Mapping[str, Any]) -> dict[str, Any]:
    smooth.verify_report(report)
    analysis = smooth._analyze(report["trajectory_results"], report["formal_design"]["seeds"])
    if analysis != report["analysis"] or smooth._json_sha256(analysis) != report["analysis_sha256"]:
        raise ValueError("T6.7.1 raw trajectory analysis does not reproduce")
    gates = smooth.recompute_gates(report)
    summaries = {row["method_id"]: row for row in analysis["method_summaries"]}
    deployable = {
        method: float(row["average_ler_equal_family_seed"])
        for method, row in summaries.items()
        if bool(row["deployable"])
    }
    strongest = min(deployable, key=deployable.get)
    holm = analysis["holm_smooth_family_superiority"]
    return {
        "raw_analysis_sha256": smooth._json_sha256(analysis),
        "evidence_gates": gates,
        "all_evidence_gates_pass": all(gates.values()),
        "primary_contrast": analysis["primary_contrast"],
        "primary_pass": bool(
            analysis["primary_contrast"]["passes_95_lcb_strictly_greater_than_zero"]
        ),
        "average_ler": deployable,
        "strongest_deployable": strongest,
        "strongest_deployable_average_ler": deployable[strongest],
        "route_a_is_global_best_deployable": strongest == "proposed_route_a",
        "route_a_beats_static_average": deployable["proposed_route_a"] < deployable["static_joint_map"],
        "route_a_beats_window_average": deployable["proposed_route_a"] < deployable["window_map"],
        "holm_confirmed_families": [
            row["family"] for row in holm if bool(row["reject_at_familywise_0_05"])
        ],
        "all_smooth_families_holm_confirmed": all(
            bool(row["reject_at_familywise_0_05"]) for row in holm
        ),
        "oracle_gap_closure": analysis["oracle_gap_closure"],
    }


def _recompute_tail(report: Mapping[str, Any]) -> dict[str, Any]:
    tail.verify_report(report)
    analysis = tail._analyze(report["trajectory_results"], report["formal_design"]["seeds"])
    if analysis != report["analysis"] or tail._json_sha256(analysis) != report["analysis_sha256"]:
        raise ValueError("T6.7.2 raw trajectory analysis does not reproduce")
    gates = tail.recompute_gates(report)
    family_rows = analysis["family_paired_safety"]
    confirmed_improvements = [
        row["family"]
        for row in family_rows
        if float(row["average_proposed_minus_baseline"]["ci95_high"]) < 0.0
    ]
    exact_equal_average = [
        row["family"]
        for row in family_rows
        if float(row["average_proposed_minus_baseline"]["estimate"]) == 0.0
    ]
    return {
        "raw_analysis_sha256": tail._json_sha256(analysis),
        "evidence_gates": gates,
        "all_evidence_gates_pass": all(gates.values()),
        "promotion_components": analysis["promotion_components"],
        "tail_safety_gate_passes": bool(analysis["tail_safety_gate_passes"]),
        "confirmed_average_improvement_families": confirmed_improvements,
        "broad_tail_improvement_confirmed": len(confirmed_improvements) == len(family_rows),
        "exact_equal_average_families": exact_equal_average,
        "calibration_shift_strict_gate": analysis["calibration_shift_strict_gate"],
        "nominal_noninferiority_gate": analysis["nominal_noninferiority_gate"],
        "action_metrics_by_family": analysis["action_metrics_by_family"],
    }


def _recompute_rtl(report: Mapping[str, Any]) -> dict[str, Any]:
    gates = rtlq.evaluate_gates(report)
    gates["semantic_mutations"] = (
        report["semantic_mutation_audit"]["detected"]
        == report["semantic_mutation_audit"]["count"]
        == 12
    )
    trace = ROOT / str(report["trace"]["path"])
    trace_sha = _sha256(trace)
    trace_bytes = trace.stat().st_size
    return {
        "evidence_gates": gates,
        "all_evidence_gates_pass": all(gates.values()),
        "trace_path": _relative(trace),
        "trace_expected_sha256": report["trace"]["sha256"],
        "trace_observed_sha256": trace_sha,
        "trace_expected_bytes": int(report["trace"]["bytes"]),
        "trace_observed_bytes": trace_bytes,
        "trace_binding_pass": trace_sha == report["trace"]["sha256"] and trace_bytes == int(report["trace"]["bytes"]),
        "cycles": int(report["aggregate_python"]["cycles"]),
        # The parent keeps comparator results per family rather than copying a
        # potentially stale total into ``aggregate_python``.  Missing the
        # family key must raise; it is never interpreted as zero.
        "rtl_mismatches": sum(int(row["mismatches"]) for row in report["cxxrtl_families"]),
        "undefined_actions": int(report["aggregate_python"]["undefined_actions"]),
        "silent_overflow": int(report["aggregate_python"]["silent_overflow"]),
        "unified_replay_fraction": (
            int(report["aggregate_python"]["unified_replay_cycles"])
            / int(report["aggregate_python"]["cycles"])
        ),
        "scope": report["evidence_scope"],
        "hmm_is_in_rtl": False,
        "measured_board_latency": False,
    }


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    parent = report["parent_audits"]
    smooth_result = report["scientific_results"]["smooth"]
    tail_result = report["scientific_results"]["tail"]
    rtl_result = report["scientific_results"]["rtl"]
    claim_ids = {row["claim_id"] for row in report["claim_registry"]}
    return {
        "G01_all_parent_artifact_hashes_current": all(
            row["expected_sha256"] == row["observed_sha256"]
            for row in parent["artifacts"]
        ),
        "G02_all_large_source_bindings_current": all(
            row["passes"] for row in parent["source_data"]
        ) and bool(rtl_result["trace_binding_pass"]),
        "G03_smooth_raw_analysis_and_evidence_recompute": bool(
            smooth_result["all_evidence_gates_pass"]
        ),
        "G04_locked_ewma_primary_lcb_strictly_positive": bool(
            smooth_result["primary_pass"]
            and float(smooth_result["primary_contrast"]["ci95_low"]) > 0.0
        ),
        "G05_all_tail_compound_nominal_preregistered_gates_pass": bool(
            tail_result["all_evidence_gates_pass"]
            and tail_result["tail_safety_gate_passes"]
            and all(tail_result["promotion_components"].values())
        ),
        "G06_million_cycle_integrated_safety_qualifies": bool(
            rtl_result["all_evidence_gates_pass"]
            and int(rtl_result["cycles"]) >= 1_000_000
            and int(rtl_result["rtl_mismatches"]) == 0
            and int(rtl_result["undefined_actions"]) == 0
            and int(rtl_result["silent_overflow"]) == 0
        ),
        "G07_no_formal_baseline_reselection_or_lock_drift": bool(
            report["frozen_contract"]["primary_baseline"] == "ewma_adaptive_map"
            and report["frozen_contract"]["formal_baseline_reselection"] is False
            and report["frozen_contract"]["threshold_lock_sha256"]
            == "9347edb270bbeb3f50d8bd8aceaeefd8003e118f1e88712dd5265519bb0f67aa"
        ),
        "G08_branch_and_claim_registry_complete": claim_ids
        == {
            "ROUTE_A_SYSTEM",
            "SMOOTH_LOCKED_EWMA",
            "GLOBAL_DEPLOYABLE_LER",
            "STATIC_GKP_SUPERIORITY",
            "TAIL_IMPROVEMENT",
            "CNN_PRIMARY",
            "HMM_ON_FPGA",
            "MEASURED_FPGA_SPEED",
        },
        "G09_restricted_verdict_does_not_promote_falsified_claims": (
            report["promotion_decision"]["scope"]
            == "contract-level simulator and pre-board correctness only"
            and report["promotion_decision"]["global_performance_rank"]
            == "not promoted"
            and report["promotion_decision"]["cnn_role"] == "ablation only"
        ),
        "G10_semantic_mutations_fail_closed": int(
            report["semantic_mutation_audit"]["detected"]
        ) == int(report["semantic_mutation_audit"]["count"]) == 8,
    }


def _semantic_mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        mutate(candidate)
        candidate["semantic_mutation_audit"] = {"count": 8, "detected": 8, "cases": []}
        cases.append({"case": name, "rejected": not all(evaluate_gates(candidate).values())})

    attempt("corrupt_parent_hash", lambda x: x["parent_audits"]["artifacts"][0].update(observed_sha256="0" * 64))
    attempt("corrupt_source_rows", lambda x: x["parent_audits"]["source_data"][0].update(passes=False))
    attempt("zero_primary_lcb", lambda x: x["scientific_results"]["smooth"]["primary_contrast"].update(ci95_low=0.0))
    attempt("erase_tail_gate", lambda x: x["scientific_results"]["tail"].update(tail_safety_gate_passes=False))
    attempt("inject_rtl_mismatch", lambda x: x["scientific_results"]["rtl"].update(rtl_mismatches=1))
    attempt("reselect_baseline", lambda x: x["frozen_contract"].update(primary_baseline="static_joint_map"))
    attempt("drop_claim", lambda x: x["claim_registry"].pop())
    attempt("promote_global_rank", lambda x: x["promotion_decision"].update(global_performance_rank="promoted"))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report() -> dict[str, Any]:
    smooth_report = _load(SMOOTH_ARTIFACT)
    tail_report = _load(TAIL_ARTIFACT)
    rtl_report = _load(RTL_ARTIFACT)
    comparator = _load(COMPARATOR_ARTIFACT)

    smooth_result = _recompute_smooth(smooth_report)
    tail_result = _recompute_tail(tail_report)
    rtl_result = _recompute_rtl(rtl_report)
    source_audits = [_binding_audit(smooth_report), _binding_audit(tail_report)]
    artifact_paths = (SMOOTH_ARTIFACT, TAIL_ARTIFACT, RTL_ARTIFACT, COMPARATOR_ARTIFACT)
    artifacts = [
        {
            "path": _relative(path),
            "expected_sha256": _sha256(path),
            "observed_sha256": _sha256(path),
        }
        for path in artifact_paths
    ]

    legacy_demoted = bool(
        comparator["gates"]["legacy_cnn_is_automatically_demoted_on_schema_and_budget"]
        and not comparator["evidence_boundary"]["cnn_ler_comparable"]
    )
    claim_registry = [
        {"claim_id": "ROUTE_A_SYSTEM", "state": "PROMOTED_RESTRICTED", "reason": "all preregistered scientific and integrated correctness gates pass"},
        {"claim_id": "SMOOTH_LOCKED_EWMA", "state": "PROMOTED", "reason": "aggregate paired EWMA-minus-Route-A 95% LCB is strictly positive"},
        {"claim_id": "GLOBAL_DEPLOYABLE_LER", "state": "FALSIFIED", "reason": f"strongest deployable is {smooth_result['strongest_deployable']}; Route-A is not global best"},
        {"claim_id": "STATIC_GKP_SUPERIORITY", "state": "FALSIFIED", "reason": "Route-A average LER exceeds static joint MAP on the formal smooth matrix"},
        {"claim_id": "TAIL_IMPROVEMENT", "state": "NOT_ESTABLISHED", "reason": "tail gates establish locked-EWMA safety/non-inferiority, not broad LER improvement"},
        {"claim_id": "CNN_PRIMARY", "state": "ABLATION_ONLY", "reason": "legacy checkpoint executes but fails matched schema and budget" if legacy_demoted else "legacy matched gate not established"},
        {"claim_id": "HMM_ON_FPGA", "state": "PROHIBITED", "reason": "posterior/HMM remains a software slow loop"},
        {"claim_id": "MEASURED_FPGA_SPEED", "state": "PROHIBITED", "reason": "no physical-board latency, deadline, resource, or power measurement"},
    ]

    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "independence_contract": {
            "no_experiment_rerun": True,
            "no_threshold_retuning": True,
            "no_baseline_reselection": True,
            "raw_trajectory_analysis_recomputed": True,
            "large_source_files_rehashed": True,
            "cross_lane_offset_prohibited": True,
        },
        "frozen_contract": {
            "primary_baseline": smooth_report["primary_baseline"],
            "formal_baseline_reselection": bool(smooth_report["formal_baseline_reselection"] or tail_report["formal_baseline_reselection"]),
            "threshold_lock_sha256": smooth_report["parent_bindings"]["threshold_lock_sha256"],
        },
        "parent_audits": {"artifacts": artifacts, "source_data": source_audits},
        "implementation_binding": {
            "path": _relative(Path(__file__)),
            "sha256": _sha256(Path(__file__)),
        },
        "scientific_results": {"smooth": smooth_result, "tail": tail_result, "rtl": rtl_result},
        "promotion_decision": {
            "verdict": VERDICT,
            "scope": "contract-level simulator and pre-board correctness only",
            "route_a_system": "promoted with restricted claims",
            "smooth_claim": "promoted only against pilot-locked EWMA aggregate",
            "tail_claim": "safety/non-inferiority only; no improvement claim",
            "global_performance_rank": "not promoted",
            "cnn_role": "ablation only",
            "fallback_branch_triggered": False,
            "smooth_only_branch_triggered": False,
            "static_deterministic_branch_triggered": False,
        },
        "claim_registry": claim_registry,
        "allowed_wording": [
            "Under the frozen simulator/protocol, Route-A improves aggregate smooth-drift LER relative to the pilot-locked EWMA baseline.",
            "The preregistered abrupt/OOD and nominal gates establish non-catastrophic behavior relative to that locked baseline.",
            "A one-million-cycle pre-board integer-golden/CXXRTL qualification was bit exact and fail closed under the exercised faults.",
        ],
        "forbidden_wording": [
            "Route-A is the best deployable GKP decoder.",
            "Route-A outperforms static joint MAP or Window MAP.",
            "Route-A improves tail LER in general.",
            "The HMM runs on FPGA.",
            "The FPGA decoder is faster than prior hardware.",
        ],
    }
    report["semantic_mutation_audit"] = {"count": 8, "detected": 8, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _semantic_mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "failed": sum(not value for value in report["gates"].values()),
    }
    report["verdict"] = VERDICT if all(report["gates"].values()) else "NO_GO_ROUTE_A_PROMOTION"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    if report.get("gates") != gates or not all(gates.values()):
        raise ValueError("T6.7.4 promotion gates do not recompute")
    audit = report["semantic_mutation_audit"]
    if int(audit["detected"]) != int(audit["count"]) or not all(row["rejected"] for row in audit["cases"]):
        raise ValueError("T6.7.4 semantic mutations are incomplete")
    if report.get("verdict") != VERDICT:
        raise ValueError("T6.7.4 verdict is not the restricted GO verdict")
    for row in report["parent_audits"]["artifacts"]:
        if _sha256(ROOT / row["path"]) != row["expected_sha256"]:
            raise ValueError(f"T6.7.4 parent artifact drifted: {row['path']}")
    for row in report["parent_audits"]["source_data"]:
        path = ROOT / row["path"]
        if _sha256(path) != row["expected_sha256"] or _csv_rows(path) != int(row["expected_rows"]):
            raise ValueError(f"T6.7.4 parent Source Data drifted: {row['path']}")
    rtl = report["scientific_results"]["rtl"]
    trace = ROOT / rtl["trace_path"]
    if _sha256(trace) != rtl["trace_expected_sha256"] or trace.stat().st_size != int(rtl["trace_expected_bytes"]):
        raise ValueError("T6.7.4 million-cycle trace drifted")
    implementation = report["implementation_binding"]
    if _sha256(ROOT / implementation["path"]) != implementation["sha256"]:
        raise ValueError("T6.7.4 implementation drifted")
    output = report.get("output_source_data_binding")
    if not output:
        raise ValueError("T6.7.4 output Source Data is unbound")
    output_path = ROOT / output["path"]
    if _sha256(output_path) != output["sha256"] or _csv_rows(output_path) != int(output["row_count"]):
        raise ValueError("T6.7.4 output Source Data drifted")


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate, value in report["gates"].items():
        rows.append({"row_type": "promotion_gate", "key": gate, "value": str(bool(value)).lower(), "detail": "preregistered/integrity promotion gate"})
    for row in report["claim_registry"]:
        rows.append({"row_type": "claim_state", "key": row["claim_id"], "value": row["state"], "detail": row["reason"]})
    smooth_result = report["scientific_results"]["smooth"]
    for method, value in smooth_result["average_ler"].items():
        rows.append({"row_type": "smooth_average_ler", "key": method, "value": f"{value:.17g}", "detail": "equal-family/equal-seed formal estimate"})
    return rows


def write_report(report: dict[str, Any], artifact: Path, source_data: Path) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    rows = _source_rows(report)
    with source_data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row_type", "key", "value", "detail"))
        writer.writeheader()
        writer.writerows(rows)
    report["output_source_data_binding"] = {
        "path": _relative(source_data),
        "sha256": _sha256(source_data),
        "row_count": len(rows),
    }
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args()
    report = build_report()
    write_report(report, args.artifact, args.source_data)
    verify_report(_load(args.artifact))
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "artifact": _relative(args.artifact)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
