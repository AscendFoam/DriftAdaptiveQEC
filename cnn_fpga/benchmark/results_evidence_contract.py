"""T7.2.3 evidence contract for the manuscript Results section.

The human-facing output is the Results prose.  This module makes the ordering,
negative-result disclosure, V4/V5 separation, Phase-6C eligibility, selection
multiplicity, and pre-board/board boundary machine-checkable.  It deliberately
fails closed if a later prose edit turns a diagnostic, literature value, or
post-route estimate into a primary result.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.2.3"
SCHEMA_VERSION = "t7.2.3-results-evidence-contract-v1"
VERDICT = "PASS_RESULTS_COMPLETE_NEGATIVE_AND_SECONDARY_BOUNDARIES"

NOTE_PATH = ROOT / "docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex"
DEFAULT_REPORT = ROOT / "docs/t7_2_3_results_evidence_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_2_3_results_evidence_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/results_evidence_contract.md"

SOURCE_PATHS = {
    "manuscript": NOTE_PATH,
    "claim_matrix": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix.json",
    "posterior_lock": ROOT / "docs/t6_6_3_route_a_posterior_threshold_lock.json",
    "promotion_gate": ROOT / "docs/t6_7_4_route_a_promotion_gate.json",
    "integrated_rtl": ROOT / "docs/t6_7_3_route_a_integrated_rtl_qualification.json",
    "hardware_pareto": ROOT / "docs/t6_9_1_route_a_hardware_pareto.json",
    "board_blocker": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
    "causal_headroom": ROOT / "docs/t6_10_1_causal_headroom.json",
    "v5_final_gate": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json",
    "phase6c_integrity": ROOT / "docs/t6_19_3_secondary_evidence_integrity.json",
    "selection_audit": ROOT / "docs/t5_4_4_multi_agent_seed_selection_audit.json",
    "teacher_validation": ROOT / "docs/t4_4_1_bounded_residual_rnn_teacher_validation.json",
    "student_validation": ROOT / "docs/t4_4_3_low_dimensional_student_validation.json",
    "gain_retention": ROOT / "docs/t4_4_4_teacher_student_gain_retention.json",
    "task_board": ROOT / "docs/new_task_board.md",
    "implementation": Path(__file__).resolve(),
}

RESULT_STATES = (
    "PRIMARY_RESTRICTED",
    "MANDATORY_NEGATIVE",
    "DIAGNOSTIC_STOP",
    "SECONDARY_ELIGIBLE",
    "NULL_OR_CONTEXTUAL",
    "HISTORICAL_EXTENSION",
)

ELIGIBLE_SECONDARY_GRADES = {
    "OFFICIAL_CODE_REPRODUCTION",
    "PROJECT_NATIVE_MATCHED",
}

REQUIRED_SUBSECTIONS = (
    "Result-state and ranking convention",
    "Untouched smooth formal comparison",
    "Action and update diagnostics",
    "Abrupt/OOD and nominal non-inferiority",
    "Deterministic execution and pre-board hardware evidence",
    "Causal-headroom audit and V5 early stop",
    "Independent Phase 6C decoder and comparison lanes",
    "Legacy CNN+FPGA evidence retained as an extension lane",
    "Teacher--student extension evidence",
    "Supplementary validation and failure-domain audit",
)

EXPECTED_PHASE6C_RESULTS = {
    "P6C-SINGLE-CI-CPD-EQUIVALENCE",
    "P6C-SURFACE-NOH-CI-ML",
    "P6C-STRUCTURED-OFFICIAL-CPD",
    "P6C-MULTIMODE-POSTERIOR-WEIGHTED",
    "P6C-AQEC-WALLCLOCK",
}

PROHIBITED_ASSERTIVE_PATTERNS = (
    "route-a is the best deployable decoder",
    "route-a outperforms static joint map",
    "route-a closes the static-to-oracle gap",
    "route-a improves abrupt tail ler",
    "v5 achieved at least 10%",
    "v5 formal verification passed",
    "v5 cxxrtl passed",
    "v5 p&r passed",
    "we surpass puviani",
    "fastest fpga decoder",
    "board-measured 222.222",
    "zero board deadline misses",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return (
        path.is_file()
        and path.stat().st_size == int(binding["bytes"])
        and _sha256(path) == str(binding["sha256"])
    )


def _declared_parent_binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return (
        path.is_file()
        and path.stat().st_size == int(binding["bytes"])
        and _sha256(path) == str(binding["sha256"])
    )


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _extract_section(tex: str, title: str) -> str:
    marker = re.search(rf"\\section\{{{re.escape(title)}\}}", tex)
    if marker is None:
        raise ValueError(f"missing section: {title}")
    tail = tex[marker.end():]
    next_section = re.search(r"\\section\{", tail)
    return tail[: next_section.start() if next_section else len(tail)].strip()


def _headings(text: str, command: str) -> list[str]:
    return re.findall(rf"\\{command}\{{([^}}]+)\}}", text)


def _claim_by_id(claim_matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(claim["claim_id"]): claim for claim in claim_matrix["claims"]}


def _result_rows() -> list[dict[str, str]]:
    raw_rows = [
        ("RES-001", "v4_locked_ewma", "V4 single-mode", "PRIMARY_RESTRICTED", "PROJECT_NATIVE_MATCHED", "POSITIVE", "Locked EWMA aggregate only; periodic is the sole Holm-confirmed family", "T6.7.1/T6.7.4"),
        ("RES-002", "v4_static_ordering", "V4 single-mode", "MANDATORY_NEGATIVE", "NEGATIVE", "NEGATIVE", "Static joint MAP has lower average and calibration worst-window LER", "T6.7.1/T6.7.2/T6.9.3"),
        ("RES-003", "v4_window_ordering", "V4 single-mode", "MANDATORY_NEGATIVE", "NEGATIVE", "NEGATIVE", "Window MAP is the strongest deployable smooth comparator", "T6.7.1/T6.7.4"),
        ("RES-004", "v4_oracle_gap", "V4 single-mode", "MANDATORY_NEGATIVE", "NEGATIVE", "NEGATIVE", "Static-to-oracle gap closure is negative", "T6.7.1/T6.9.3"),
        ("RES-005", "v4_tail_noninferiority", "V4 abrupt/OOD", "MANDATORY_NEGATIVE", "PROJECT_NATIVE_MATCHED", "NEUTRAL", "Five families are exactly equal to EWMA; no broad tail improvement", "T6.7.2/T6.7.4"),
        ("RES-006", "v4_fallback_cost", "V4 abrupt/OOD", "MANDATORY_NEGATIVE", "PROJECT_NATIVE_MATCHED", "NEGATIVE", "High fallback and unnecessary-fallback rates remain visible", "T6.7.2/T6.7.4"),
        ("RES-007", "v4_false_updates", "V4 abrupt/OOD", "MANDATORY_NEGATIVE", "PROJECT_NATIVE_MATCHED", "NEGATIVE", "All family-specific false-update counts remain visible", "T6.7.2/T6.7.4"),
        ("RES-008", "v4_failed_policy_families", "V4 pilot", "MANDATORY_NEGATIVE", "NEGATIVE", "NEGATIVE", "Static-switch and freeze-all each failed all 38 safe tuples", "T6.6.3"),
        ("RES-009", "external_bocd_budget", "V4 external drift", "MANDATORY_NEGATIVE", "PROJECT_NATIVE_MATCHED", "MIXED", "Paired LER outcome is inseparable from the 13,004.1-us budget failure", "T6.8.3/T6.8.7/T6.9.3"),
        ("RES-010", "v4_cxxrtl", "V4 pre-board", "PRIMARY_RESTRICTED", "CXXRTL_PREBOARD", "POSITIVE", "One million cycles and failure branches; no board inference", "T6.7.3/T6.7.4"),
        ("RES-011", "v4_commit_rollback_attempts", "V4 pre-board", "PRIMARY_RESTRICTED", "CXXRTL_PREBOARD", "NEUTRAL", "All 75 commit and 25 rollback attempts stay in the denominator", "T6.7.3"),
        ("RES-012", "v4_post_route", "V4 pre-board", "PRIMARY_RESTRICTED", "POST_ROUTE_ESTIMATE", "POSITIVE", "Six-cycle II=1 and three-seed P&R are estimates", "T6.9.1/T6.9.3"),
        ("RES-013", "physical_board_null", "Physical board", "NULL_OR_CONTEXTUAL", "BLOCKED", "NULL", "All 42 measured fields remain null", "T6.9.2/T6.9.3"),
        ("RES-014", "v5_causal_selector", "V5 entry diagnostic", "DIAGNOSTIC_STOP", "DIAGNOSTIC_ONLY", "NEGATIVE", "Strict causal selector headroom is negative", "T6.10.1/T6.15.5"),
        ("RES-015", "v5_action_headroom", "V5 entry diagnostic", "DIAGNOSTIC_STOP", "DIAGNOSTIC_ONLY", "NEGATIVE", "Expanded action-space headroom is 0.02549%, below 12%", "T6.10.1/T6.15.5"),
        ("RES-016", "v5_downstream_absence", "V5 stopped branch", "NULL_OR_CONTEXTUAL", "BLOCKED", "NULL", "No untouched LER/tail, quantized, formal, CXXRTL, or P&R output exists", "T6.15.5"),
        ("RES-017", "p6c_single_cpd", "Phase 6C single-mode", "SECONDARY_ELIGIBLE", "PROJECT_NATIVE_MATCHED", "EQUIVALENCE", "CI equals Euclidean CPD only in the frozen square/isotropic domain", "T6.17.1/T6.19.3"),
        ("RES-018", "p6c_noh_cnot", "Phase 6C CNOT", "SECONDARY_ELIGIBLE", "PROJECT_NATIVE_MATCHED", "POSITIVE", "ML lowers failure relative to CI on the matched two-GKP CNOT task", "T6.17.2/T6.19.3"),
        ("RES-019", "p6c_official_cpd", "Phase 6C structured CPD", "SECONDARY_ELIGIBLE", "OFFICIAL_CODE_REPRODUCTION", "POSITIVE", "Official data plus partial small-distance CPD reproduction", "T6.18.2/T6.19.3"),
        ("RES-020", "p6c_multimode_adaptive", "Phase 6C multimode CPD", "SECONDARY_ELIGIBLE", "PROJECT_NATIVE_MATCHED", "POSITIVE", "Observed-only adaptive weighting improves aggregate and tail metrics", "T6.18.3/T6.19.3"),
        ("RES-021", "p6c_aqec_project", "Phase 6C AQEC", "SECONDARY_ELIGIBLE", "PROJECT_NATIVE_MATCHED", "NEGATIVE", "Project-native active/autonomous control fails to beat idle; official protocol blocked", "T6.18.1/T6.19.3"),
        ("RES-022", "p6c_learned_zero", "Phase 6C learned", "MANDATORY_NEGATIVE", "INELIGIBLE", "NULL", "Zero of 16 candidate families is same-task eligible", "T6.17.3/T6.19.3"),
        ("RES-023", "p6c_gqf_blocked", "Phase 6C GQF/NMF", "NULL_OR_CONTEXTUAL", "BLOCKED", "NULL", "Zero of 15 exact checks and all 13 matched metrics null", "T6.8.4/T6.8.5/T6.19.3"),
        ("RES-024", "p6c_external_fpga", "Phase 6C FPGA", "NULL_OR_CONTEXTUAL", "LITERATURE_ONLY", "NULL", "18 normalized rows but zero exact same-task comparator", "T6.19.2/T6.19.3"),
        ("RES-025", "legacy_cnn", "Historical T24", "HISTORICAL_EXTENSION", "PROJECT_NATIVE_HISTORICAL", "POSITIVE", "Four-scenario frozen software-HIL result is outside the Route-A ranking", "T24/T7.1.1"),
        ("RES-026", "teacher_student_retention", "Teacher/student", "HISTORICAL_EXTENSION", "PROJECT_NATIVE_SIMULATION", "POSITIVE", "Finite-model gain retention and compression only", "T4.4.4/T4.4.5"),
        ("RES-027", "teacher_student_selection", "Teacher/student", "MANDATORY_NEGATIVE", "SELECTION_AUDIT", "NEGATIVE", "All restarts, cap hits, and test-hindsight reversals remain disclosed", "T4.4.1/T4.4.3/T5.4.4"),
    ]
    return [
        {
            "row_id": row_id,
            "result_id": result_id,
            "lane": lane,
            "result_state": result_state,
            "evidence_grade": evidence_grade,
            "polarity": polarity,
            "boundary": boundary,
            "source_ids": source_ids,
        }
        for row_id, result_id, lane, result_state, evidence_grade, polarity, boundary, source_ids in raw_rows
    ]


def _manuscript_snapshot() -> dict[str, Any]:
    tex = NOTE_PATH.read_text(encoding="utf-8")
    results = _extract_section(tex, "Results")
    normalized = _normalize(results)
    sections = _headings(tex, "section")
    subsections = _headings(results, "subsection")
    positions = {title: subsections.index(title) for title in REQUIRED_SUBSECTIONS if title in subsections}
    checks = {
        "section_order": sections.index("Results") < sections.index("Where the current data show an advantage"),
        "required_subsections": all(title in subsections for title in REQUIRED_SUBSECTIONS),
        "ordered_v4_v5_phase6c_extensions": positions == {title: index for index, title in enumerate(REQUIRED_SUBSECTIONS)},
        "state_convention": all(token in normalized for token in (
            "primary restricted", "mandatory negative", "diagnostic stop",
            "secondary eligible", "null or contextual", "result-state contract",
        )),
        "smooth_complete": all(token in normalized for token in (
            "28,311,552 scored decisions", "2.1687", "2.14", "periodic drift",
            "worse than static joint map", "worse than window map", "-0.03046",
            "strongest deployable method", "hidden-state oracle",
        )),
        "external_bocd_budget_pair": all(token in normalized for token in (
            "bayesian online change-detection wrapper", "1.00184", "13,004.1",
            "5,000", "one deadline miss", "reported together",
        )),
        "action_cost_and_failed_families": all(token in normalized for token in (
            "884,736", "13,824", "661 avoided errors", "47 induced errors",
            "21.6305", "21.5822", "all 38 posterior-safe pilot tuples",
            "-0.037109375", "-0.044921875",
        )),
        "tail_complete": all(token in normalized for token in (
            "43,646,976 scored decisions", "exactly zero", "181/512", "32/512",
            "59.45--95.85", "58.80--94.63", "2,044--3,365", "not be described as freeze-all",
        )),
        "v4_preboard_and_attempts": all(token in normalized for token in (
            "1,000,000 cycles", "zero visible-field mismatch", "six-cycle",
            "75 host-commit attempts", "25 rollback attempts", "99.5802",
            "all 42 physical-board fields remain null", "no measured fpga latency",
        )),
        "v5_diagnostic_and_absence": all(token in normalized for token in (
            "71,958,528", "4,571,136", "-0.2322", "0.4587", "nine errors",
            "0.02549", "12\\% entry gate", "no v5 untouched ler matrix",
            "tail test", "quantized-retention result", "formal proof", "cxxrtl run",
            "p\\&r profile", "reusing v4 hardware artifacts",
        )),
        "phase6c_eligible_table": all(token in normalized for token in (
            "rows eligible for explicitly secondary results", "official\\_code\\_reproduction",
            "project\\_native\\_matched", "1,048,576-point", "3,080,192 trials",
            "0.602456/0.599594", "9.6 million cycles", "six cells $\\times$ 24 clusters",
        )),
        "phase6c_nonranking_ledger": all(token in normalized for token in (
            "non-ranking boundary ledger", "zero of 16", "zero of 15",
            "all 13 matched nmf metrics null", "18 external fpga implementations",
            "zero exact same-task rows", "literature-only latency and lifetime values remain",
        )),
        "multimode_scope": all(token in normalized for token in (
            "0.0646684", "[0.0644128,0.0649264]", "all 32 seed clusters",
            "0.320313 to 0.291016", "0.277271 to 0.240627",
            "independent multimode cpd task rather than to the stopped v5 policy",
        )),
        "historical_extension_scope": all(token in normalized for token in (
            "not part of the new unified", "other two scenarios", "not evidence that cnn is the best",
            "finite-model teacher--student extension", "neither an asymptotic decay constant nor fpga wall-clock time",
            "not an official reproduction or improvement over puviani",
        )),
        "selection_multiplicity": all(token in normalized for token in (
            "three restarts (601, 709, and 811)", "restarts 601 and 811 hit the 320-epoch cap",
            "test-hindsight restart would have differed", "1-, 2-, and 4-state models with three restarts each",
            "all six 2-/4-state fits hit the 900-epoch cap", "evaluation remained excluded",
        )),
        "no_assertive_overclaim": not any(pattern in normalized for pattern in PROHIBITED_ASSERTIVE_PATTERNS),
    }
    return {
        "section_title": "Results",
        "section_order": sections,
        "subsections": subsections,
        "characters": len(results),
        "sha256": hashlib.sha256(results.encode("utf-8")).hexdigest(),
        "checks": checks,
        "prohibited_hits": [pattern for pattern in PROHIBITED_ASSERTIVE_PATTERNS if pattern in normalized],
    }


def _phase6c_bindings_live(report: Mapping[str, Any]) -> bool:
    bindings = list(report["parent_bindings"].values()) + list(report["raw_bindings"].values())
    return all(_declared_parent_binding_live(binding) for binding in bindings)


def _parent_state() -> dict[str, Any]:
    claim_matrix = _load_json(SOURCE_PATHS["claim_matrix"])
    claims = _claim_by_id(claim_matrix)
    posterior = _load_json(SOURCE_PATHS["posterior_lock"])
    promotion = _load_json(SOURCE_PATHS["promotion_gate"])
    rtl = _load_json(SOURCE_PATHS["integrated_rtl"])
    hardware = _load_json(SOURCE_PATHS["hardware_pareto"])
    board = _load_json(SOURCE_PATHS["board_blocker"])
    headroom = _load_json(SOURCE_PATHS["causal_headroom"])
    v5 = _load_json(SOURCE_PATHS["v5_final_gate"])
    phase6c = _load_json(SOURCE_PATHS["phase6c_integrity"])
    selection = _load_json(SOURCE_PATHS["selection_audit"])
    teacher = _load_json(SOURCE_PATHS["teacher_validation"])
    student = _load_json(SOURCE_PATHS["student_validation"])

    smooth = promotion["scientific_results"]["smooth"]
    tail = promotion["scientific_results"]["tail"]
    rtl_result = promotion["scientific_results"]["rtl"]
    nested = headroom["development_audit"]["nested_audit"]
    p6c_claims = [claim for claim_id, claim in claims.items() if claim_id.startswith("P6C-")]
    eligible_results = {
        str(claim["claim_id"])
        for claim in p6c_claims
        if bool(claim["placements"]["results"])
        and str(claim["evidence_grade"]) in ELIGIBLE_SECONDARY_GRADES
    }
    literature_results = {
        str(claim["claim_id"])
        for claim in p6c_claims
        if bool(claim["placements"]["results"])
        and str(claim["evidence_grade"]) == "LITERATURE_ONLY"
    }
    measured = board["measured_results"]
    tail_rows = tail["action_metrics_by_family"]
    teacher_caps = list(teacher["training_cap_hit_indices"])
    student_cap_count = sum(bool(row["best_epoch_reached_training_cap"]) for row in student["training_records"])
    return {
        "verdicts": {
            "promotion": promotion["verdict"],
            "integrated_rtl": rtl["verdict"],
            "hardware": hardware["verdict"],
            "headroom": headroom["verdict"],
            "v5": v5["verdict"],
            "board": board["verdict"],
            "phase6c": phase6c["verdict"],
        },
        "smooth": {
            "primary_estimate": float(smooth["primary_contrast"]["estimate"]),
            "primary_ci_low": float(smooth["primary_contrast"]["ci95_low"]),
            "holm_families": list(smooth["holm_confirmed_families"]),
            "strongest_deployable": smooth["strongest_deployable"],
            "route_a_beats_static": bool(smooth["route_a_beats_static_average"]),
            "route_a_beats_window": bool(smooth["route_a_beats_window_average"]),
            "gap_closure": float(smooth["oracle_gap_closure"]["gap_closure"]),
        },
        "tail": {
            "broad_improvement": bool(tail["broad_tail_improvement_confirmed"]),
            "confirmed_improvement_families": list(tail["confirmed_average_improvement_families"]),
            "exact_equal_families": list(tail["exact_equal_average_families"]),
            "minimum_fallback": min(float(row["fallback_rate"]) for row in tail_rows[:-1]),
            "maximum_fallback": max(float(row["fallback_rate"]) for row in tail_rows[:-1]),
            "minimum_false_updates": min(int(row["false_updates"]) for row in tail_rows[:-1]),
            "maximum_false_updates": max(int(row["false_updates"]) for row in tail_rows[:-1]),
        },
        "failed_policy_families": {
            "static_switch": posterior["protocol_revision_disclosure"]["v2_full_selector_nogo"],
            "freeze_all": posterior["protocol_revision_disclosure"]["v3_full_selector_nogo"],
        },
        "external_bocd": claims["GENERAL_DRIFT_EXTERNAL_COMPARISON"]["current_result"]["paired_outcome"],
        "rtl": {
            "cycles": int(rtl_result["cycles"]),
            "mismatches": int(rtl_result["rtl_mismatches"]),
            "undefined_actions": int(rtl_result["undefined_actions"]),
            "silent_overflow": int(rtl_result["silent_overflow"]),
            "host_commit_attempts": int(rtl["aggregate_python"]["host_commit_attempts"]),
            "rollback_attempts": int(rtl["aggregate_python"]["rollback_attempts"]),
            "measured_board_latency": bool(rtl_result["measured_board_latency"]),
        },
        "hardware": {
            "profile_count": len(hardware["profiles"]),
            "seed_counts": [len(profile["place_route"]) for profile in hardware["profiles"]],
            "evidence_boundary": hardware["evidence_boundary"],
        },
        "board": {
            "field_count": len(measured),
            "nonnull_count": sum(value is not None for value in measured.values()),
        },
        "v5": {
            "formal_decisions": int(headroom["formal_diagnostic_audit"]["scored_decisions"]),
            "development_trajectories": int(headroom["development_audit"]["trajectory_count"]),
            "development_decisions": int(nested["total_decisions"]),
            "strict_causal_headroom": float(v5["headroom_recomputation"]["strict_causal_router_headroom"]),
            "incremental_action_headroom": float(v5["headroom_recomputation"]["incremental_action_space_headroom"]),
            "dropped_tasks": len(v5["dropped_tasks"]),
            "downstream_outputs": len(v5["v5_downstream_outputs_found"]),
            "formal_manifest": bool(v5["formal_access"]["v5_formal_manifest_exists"]),
            "formal_output": bool(v5["formal_access"]["v5_formal_output_exists"]),
        },
        "phase6c": {
            "eligible_result_ids": sorted(eligible_results),
            "literature_result_ids": sorted(literature_results),
            "global_score": phase6c["ranking_policy"]["global_score"],
            "global_winner": phase6c["ranking_policy"]["global_winner"],
            "cells": int(phase6c["source_data"]["rows"]),
            "gates_passed": int(phase6c["gate_summary"]["passed"]),
            "parent_bindings_live": _phase6c_bindings_live(phase6c),
            "recomputations": phase6c["recomputations"],
        },
        "selection": {
            "episodes": int(selection["audit_summary"]["selection_episode_count"]),
            "active_evaluation_selection": int(selection["audit_summary"]["active_selection_episodes_using_evaluation"]),
            "hindsight_disagreements": int(selection["audit_summary"]["hindsight_selection_disagreement_count"]),
            "teacher_restart_seeds": list(teacher["config"]["restart_seeds"]),
            "teacher_cap_indices": teacher_caps,
            "teacher_failed_restarts": list(teacher["failed_restart_indices"]),
            "student_training_records": len(student["training_records"]),
            "student_cap_count": student_cap_count,
        },
        "claim_matrix_parent_verification": all(claim_matrix["parent_verification"].values()),
    }


EXPECTED_VERDICTS = {
    "promotion": "GO_ROUTE_A_CONTRACT_SYSTEM_RESTRICTED_SIMULATOR_AND_PREBOARD_CLAIMS",
    "integrated_rtl": "PASS_ROUTE_A_INTEGRATED_LONG_RTL_QUALIFICATION",
    "hardware": "PASS_ROUTE_A_INTEGRATED_THREE_SEED_PR_ESTIMATE_NOT_BOARD_MEASURED",
    "headroom": "NO_GO_V5_INSUFFICIENT_ACTION_SPACE_HEADROOM",
    "v5": "NO_GO_V5_EARLY_HEADROOM_STOP",
    "board": "BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL",
    "phase6c": "PASS_AUX_COMPARISON_INTEGRITY",
}


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    manuscript = report["manuscript"]
    checks = manuscript["checks"]
    parent = report["parent_state"]
    rows = report["result_rows"]
    states = {row["result_state"] for row in rows}
    p6c = parent["phase6c"]
    source_ok = bool(report["source_integrity_declared"])
    if check_live_sources:
        source_ok = source_ok and all(_binding_live(binding) for binding in report["source_bindings"].values())
    return {
        "G01_identity": report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION,
        "G02_results_order_and_complete_subsections": bool(checks["section_order"] and checks["required_subsections"] and checks["ordered_v4_v5_phase6c_extensions"]),
        "G03_result_state_ontology": bool(checks["state_convention"]) and states == set(RESULT_STATES),
        "G04_v4_smooth_positive_and_stronger_negatives": bool(checks["smooth_complete"]) and parent["smooth"]["primary_ci_low"] > 0 and parent["smooth"]["holm_families"] == ["periodic_drift"] and parent["smooth"]["strongest_deployable"] == "window_map" and not parent["smooth"]["route_a_beats_static"] and not parent["smooth"]["route_a_beats_window"] and parent["smooth"]["gap_closure"] < 0,
        "G05_external_bocd_outcome_with_budget_failure": bool(checks["external_bocd_budget_pair"]) and parent["external_bocd"]["external_update_worst_us"] > parent["external_bocd"]["wallclock_cap_us"] and parent["external_bocd"]["deadline_miss_count"] == 1,
        "G06_action_cost_and_failed_policy_families": bool(checks["action_cost_and_failed_families"]) and all("all 38" in text for text in parent["failed_policy_families"].values()),
        "G07_tail_noninferiority_not_improvement": bool(checks["tail_complete"]) and not parent["tail"]["broad_improvement"] and parent["tail"]["confirmed_improvement_families"] == [] and len(parent["tail"]["exact_equal_families"]) == 5,
        "G08_tail_fallback_and_false_update_costs": parent["tail"]["minimum_fallback"] > 0.59 and parent["tail"]["maximum_fallback"] > 0.95 and parent["tail"]["minimum_false_updates"] == 2044 and parent["tail"]["maximum_false_updates"] == 3365,
        "G09_v4_preboard_execution_and_attempt_denominator": bool(checks["v4_preboard_and_attempts"]) and parent["rtl"] == {"cycles": 1_000_000, "mismatches": 0, "undefined_actions": 0, "silent_overflow": 0, "host_commit_attempts": 75, "rollback_attempts": 25, "measured_board_latency": False},
        "G10_post_route_estimate_and_board_null": parent["hardware"]["profile_count"] == 2 and parent["hardware"]["seed_counts"] == [3, 3] and parent["board"] == {"field_count": 42, "nonnull_count": 0},
        "G11_v5_diagnostic_values": bool(checks["v5_diagnostic_and_absence"]) and parent["v5"]["formal_decisions"] == 71_958_528 and parent["v5"]["development_trajectories"] == 186 and parent["v5"]["development_decisions"] == 4_571_136 and parent["v5"]["strict_causal_headroom"] < 0 and 0 <= parent["v5"]["incremental_action_headroom"] < 0.001,
        "G12_v5_early_stop_absence": parent["v5"]["dropped_tasks"] == 20 and parent["v5"]["downstream_outputs"] == 0 and not parent["v5"]["formal_manifest"] and not parent["v5"]["formal_output"],
        "G13_phase6c_exact_secondary_eligibility": bool(checks["phase6c_eligible_table"]) and set(p6c["eligible_result_ids"]) == EXPECTED_PHASE6C_RESULTS and not p6c["literature_result_ids"],
        "G14_phase6c_integrity_and_no_global_rank": p6c["cells"] == 206 and p6c["gates_passed"] == 24 and p6c["global_score"] is False and p6c["global_winner"] is None and p6c["parent_bindings_live"],
        "G15_phase6c_negative_null_ledger": bool(checks["phase6c_nonranking_ledger"]),
        "G16_multimode_positive_is_task_local": bool(checks["multimode_scope"]) and abs(p6c["recomputations"]["multimode"]["observed_only_posterior_predictive_weighted"]["p_L"] - 0.1722609375) < 1e-15,
        "G17_historical_extensions_remain_bounded": bool(checks["historical_extension_scope"]),
        "G18_restart_cap_and_hindsight_disclosure": bool(checks["selection_multiplicity"]) and parent["selection"]["episodes"] == 6 and parent["selection"]["active_evaluation_selection"] == 0 and parent["selection"]["hindsight_disagreements"] == 2 and parent["selection"]["teacher_restart_seeds"] == [601, 709, 811] and parent["selection"]["teacher_cap_indices"] == [0, 2] and parent["selection"]["teacher_failed_restarts"] == [] and parent["selection"]["student_training_records"] == 9 and parent["selection"]["student_cap_count"] == 6,
        "G19_no_assertive_overclaim": bool(checks["no_assertive_overclaim"]) and manuscript["prohibited_hits"] == [],
        "G20_parent_verdicts_and_live_sources": parent["verdicts"] == EXPECTED_VERDICTS and parent["claim_matrix_parent_verification"] and source_ok,
    }


def _semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    targets = list(evaluate_gates(report))
    cases: list[dict[str, Any]] = []
    check_map = {
        "G02_results_order_and_complete_subsections": "ordered_v4_v5_phase6c_extensions",
        "G04_v4_smooth_positive_and_stronger_negatives": "smooth_complete",
        "G05_external_bocd_outcome_with_budget_failure": "external_bocd_budget_pair",
        "G06_action_cost_and_failed_policy_families": "action_cost_and_failed_families",
        "G07_tail_noninferiority_not_improvement": "tail_complete",
        "G09_v4_preboard_execution_and_attempt_denominator": "v4_preboard_and_attempts",
        "G11_v5_diagnostic_values": "v5_diagnostic_and_absence",
        "G13_phase6c_exact_secondary_eligibility": "phase6c_eligible_table",
        "G15_phase6c_negative_null_ledger": "phase6c_nonranking_ledger",
        "G16_multimode_positive_is_task_local": "multimode_scope",
        "G17_historical_extensions_remain_bounded": "historical_extension_scope",
        "G18_restart_cap_and_hindsight_disclosure": "selection_multiplicity",
        "G19_no_assertive_overclaim": "no_assertive_overclaim",
    }
    for index, target in enumerate(targets):
        mutated = copy.deepcopy(report)
        if target == "G01_identity":
            mutated["task_id"] = "T7.2.X"
        elif target == "G03_result_state_ontology":
            mutated["result_rows"][0]["result_state"] = "UNDECLARED"
        elif target in check_map:
            mutated["manuscript"]["checks"][check_map[target]] = False
        elif target == "G08_tail_fallback_and_false_update_costs":
            mutated["parent_state"]["tail"]["minimum_false_updates"] = 0
        elif target == "G10_post_route_estimate_and_board_null":
            mutated["parent_state"]["board"]["nonnull_count"] = 1
        elif target == "G12_v5_early_stop_absence":
            mutated["parent_state"]["v5"]["downstream_outputs"] = 1
        elif target == "G14_phase6c_integrity_and_no_global_rank":
            mutated["parent_state"]["phase6c"]["global_winner"] = "forged"
        elif target == "G20_parent_verdicts_and_live_sources":
            mutated["source_integrity_declared"] = False
        else:  # pragma: no cover - exhaustive guard for future gates
            raise AssertionError(f"unhandled mutation target: {target}")
        rejected = not evaluate_gates(mutated)[target]
        cases.append({"mutation_id": f"M{index + 1:02d}", "target_gate": target, "rejected": rejected})
    return {"count": len(cases), "detected": sum(case["rejected"] for case in cases), "cases": cases}


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "task_id", "schema_version", "manuscript", "result_rows", "parent_state",
            "source_bindings", "source_integrity_declared", "gates", "verdict",
        )
    }


def build_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manuscript": _manuscript_snapshot(),
        "result_rows": _result_rows(),
        "parent_state": _parent_state(),
        "source_bindings": {name: _binding(path) for name, path in SOURCE_PATHS.items()},
        "source_integrity_declared": True,
    }
    report["gates"] = evaluate_gates(report, check_live_sources=True)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_RESULTS_EVIDENCE_CONTRACT"
    report["semantic_mutation_audit"] = _semantic_mutation_audit(report)
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def _write_source_data(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    fieldnames = [
        "row_id", "result_id", "lane", "result_state", "evidence_grade",
        "polarity", "boundary", "source_ids",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    parent = report["parent_state"]
    lines = [
        "# T7.2.3 Results evidence contract",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- gates：`{report['gate_summary']['passed']}/{report['gate_summary']['total']}`",
        f"- semantic mutations：`{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`",
        f"- result rows：`{len(report['result_rows'])}`",
        f"- V5：`{parent['v5']['dropped_tasks']}` dropped tasks，`{parent['v5']['downstream_outputs']}` downstream outputs",
        f"- Phase 6C：eligible secondary results=`{len(parent['phase6c']['eligible_result_ids'])}`，literature in Results=`{len(parent['phase6c']['literature_result_ids'])}`",
        f"- board：`{parent['board']['field_count']}` measured fields，nonnull=`{parent['board']['nonnull_count']}`",
        "",
        "| result | state | grade | polarity | boundary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["result_rows"]:
        lines.append(
            f"| `{row['result_id']}` | `{row['result_state']}` | `{row['evidence_grade']}` | "
            f"`{row['polarity']}` | {row['boundary']} |"
        )
    lines.extend([
        "",
        "V4 的 locked-EWMA 正对比与 static/Window/oracle-gap/tail/fallback 反证并列；",
        "V5 只报告已执行的 entry diagnostic 和 downstream absence；",
        "Phase 6C 只有 official-code reproduction 或 project-native matched 行可进入明确标注的 secondary Results；",
        "文献值、blocked/null、P&R estimate 和 42 个板测 null 字段均不能生成主排名或实测结论。",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any]) -> None:
    _write_source_data(report["result_rows"], DEFAULT_SOURCE_DATA)
    _atomic_json(report, DEFAULT_REPORT)
    _atomic_text(_markdown(report), DEFAULT_MARKDOWN)


def verify_report(path: Path = DEFAULT_REPORT) -> dict[str, bool]:
    stored = _load_json(path)
    fresh = build_report()
    stored_gates = evaluate_gates(stored, check_live_sources=True)
    return {
        "identity": stored.get("task_id") == TASK_ID and stored.get("schema_version") == SCHEMA_VERSION,
        "live_sources": all(_binding_live(binding) for binding in stored["source_bindings"].values()),
        "all_stored_gates_pass": all(stored_gates.values()),
        "gate_snapshot_matches": stored.get("gates") == stored_gates,
        "mutation_audit_complete": stored["semantic_mutation_audit"]["count"] == stored["semantic_mutation_audit"]["detected"] == len(stored["gates"]),
        "analysis_sha256_live": stored.get("analysis_sha256") == _canonical_sha256(_analysis_payload(stored)),
        "fresh_analysis_matches": stored.get("analysis_sha256") == fresh.get("analysis_sha256"),
        "verdict": stored.get("verdict") == VERDICT,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        checks = verify_report()
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if all(checks.values()) else 1
    report = build_report()
    write_outputs(report)
    print(json.dumps({
        "verdict": report["verdict"],
        "gates": report["gate_summary"],
        "mutations": {
            "detected": report["semantic_mutation_audit"]["detected"],
            "count": report["semantic_mutation_audit"]["count"],
        },
        "analysis_sha256": report["analysis_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
