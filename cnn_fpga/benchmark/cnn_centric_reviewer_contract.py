"""Build the T7.3.2 pre-emptive reviewer response and its fail-closed evidence contract."""

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
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.3.2"
SCHEMA_VERSION = "t7.3.2-cnn-centric-reviewer-contract-v1"
VERDICT = "PASS_CNN_NONCENTRIC_REPLACEABLE_LEARNING_REVIEWER_RESPONSE"

CONFIG = ROOT / "configs/phase6d/t7_3_2_cnn_centric_reviewer_contract.json"
BOARD = ROOT / "docs/new_task_board.md"
RISKS = ROOT / "docs/new_risks.md"
MANUSCRIPT = ROOT / "docs/paper_notes/Phase6D_Dual_Lane_GKP_manuscript.tex"
MANUSCRIPT_CONTRACT = ROOT / "docs/t7_2_6_phase6d_manuscript_delta.json"
CLAIM_FIGURE_DELTA = ROOT / "docs/t7_1_5_phase6d_claim_figure_delta.json"
DUAL_CONTRACT = ROOT / "docs/t6_20_2_dual_evidence_lane_contract.json"
HEADROOM = ROOT / "docs/t6_20_4_multimode_causal_headroom.json"
EVIDENCE_MATRIX = ROOT / "docs/t6_26_3_dual_lane_evidence_matrix.json"
FINAL_GATE = ROOT / "docs/t6_26_4_final_dual_lane_gate.json"
LEGACY_TEACHER = ROOT / "docs/t4_4_1_bounded_residual_rnn_teacher_validation.json"
LEGACY_STUDENT_VALIDATION = ROOT / "docs/t4_4_3_low_dimensional_student_validation.json"
LEGACY_STUDENT = ROOT / "docs/t4_4_3_low_dimensional_student.json"
LEGACY_RETENTION = ROOT / "docs/t4_4_4_teacher_student_gain_retention.json"
LEGACY_MISMATCH = ROOT / "docs/t5_4_6_randomized_model_mismatch.json"
LEGACY_HARDWARE = ROOT / "docs/t5_5_4_gru_student_hardware_feasibility.json"
LEGACY_CNN_ABLATION = ROOT / "docs/t5_4_3_causal_ablation_negative_results.json"
LEARNED_ELIGIBILITY = ROOT / "docs/t6_17_3_learned_model_eligibility_replay.json"

DEFAULT_REPORT = ROOT / "docs/t7_3_2_cnn_centric_reviewer_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_3_2_cnn_centric_reviewer_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/cnn_centric_reviewer_response.md"

RESPONSE_STATES = {
    "REVIEWER_CONCERN",
    "CURRENT_PRIMARY",
    "CURRENT_NEGATIVE",
    "LEGACY_POSITIVE",
    "TASK_SIGNATURE_BOUNDARY",
    "DEPLOYMENT_BOUNDARY",
    "PROMOTION_GATE",
    "MANUSCRIPT_CHANGE",
    "RISK_DISCLOSURE",
    "RESPONSE_WORDING",
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
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.exists() and path.stat().st_size == binding["bytes"] and _sha256(path) == binding["sha256"]


def _task_status(board: str, task_id: str) -> str:
    match = re.search(rf"^\|\s*{re.escape(task_id)}\s*\|\s*([^|]+?)\s*\|", board, re.MULTILINE)
    if not match:
        raise ValueError(f"task status not found: {task_id}")
    return match.group(1).strip()


def _student_candidate(hardware: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    for candidate in hardware["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise ValueError(f"candidate not found: {candidate_id}")


def _response_text() -> str:
    return (
        "We agree that the earlier CNN--FPGA framing could make the learned component appear more central than the evidence supports. "
        "The revised manuscript is not CNN-centric: its two primary evidence lanes are (i) a strongest-baseline-gated multimode software qualification and (ii) an exact single-mode deterministic, atomic and fail-closed RTL qualification. "
        "CNN, teacher and student modules have no independent vote in either lane.\n\n"
        "For Phase 6D, the multimode causal-headroom entry test stopped with the proposed method tied to static-mixture exact MLD, at p_L=0.1119791667 for both methods and a paired relative improvement of 0% [0%, 0%]. "
        "Consequently, no Phase-6D teacher was authorized, T6.26.1 and T6.26.2 were Dropped, and no learned training, checkpoint, quantization or formal-retention result was created. "
        "The final GO_RTL_ONLY verdict therefore does not depend on a CNN or student.\n\n"
        "We have not hidden the favorable historical learning results. In a different finite-cutoff, two-level sBs controller task, a 72,853-parameter GRU teacher was distilled to a four-state, 95-scalar recurrence with evaluation action-imitation MSE 6.083136e-6 and a minimum matched retention point/lower bound of 0.981457/0.944501. "
        "Those results have a different observation, objective, simulator and action signature from the current multimode posterior/MLD task. They are therefore reported only as task-local historical evidence and cannot validate the current algorithm or the exact RTL. The full quantized-GRU route was also dropped after a 72,854-cycle optimistic lower bound without functional RTL or physical-retention evidence.\n\n"
        "We also reran the preserved tiny CNN bit-exact five times on its 206-sample held-out residual-parameter split. It reduced parameter MSE from 8.034045e-6 to 2.414453e-6, but this was a single legacy split without an independent seed-cluster confidence interval and measured neither LER nor control gain. An exhaustive 16-family eligibility registry found zero same-task eligible learned checkpoints. We therefore retain the replay as a diagnostic method detail, not as evidence against simulator overfitting in the current task.\n\n"
        "A future learned module can be reconsidered only as a replaceable approximation to an already authorized posterior, log-likelihood ratio, logical-coset probability or action. It must share the registered split and observed-only information contract, beat a matched classical approximation under the same runtime and memory budget, and pass calibration, action-agreement, LER-retention, worst-family, held-out-OOD, quantization and formal-retention gates while providing a concrete compression or cost benefit. Otherwise it remains an ablation. Thus, the present claims avoid the simulator-overfitting concern by being independent of learning; they do not claim that every future learned model will generalize."
    )


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {"generated_at_utc", "analysis_sha256", "semantic_mutation_audit", "source_data", "markdown"}
    return {key: value for key, value in report.items() if key not in omitted}


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in report["response_rows"]:
        rows.append(
            {
                "row_id": row["row_id"],
                "response_state": row["response_state"],
                "topic": row["topic"],
                "claim": row["claim"],
                "boundary": row["boundary"],
                "source_ids_json": _canonical(row["source_ids"]),
                "row_sha256": _canonical_sha256(row),
            }
        )
    return rows


def _source_data_matches(report: Mapping[str, Any], path: Path = DEFAULT_SOURCE_DATA) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        stored = list(csv.DictReader(stream))
    return stored == _source_rows(report)


def _manuscript_checks(text: str, markers: Sequence[str]) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text)
    sections = [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
        "Supplementary evidence delta",
    ]
    section_presence = {
        section: (
            f"\\section{{{section}}}" in text
            or (section == "Abstract" and "\\begin{abstract}" in text)
            or (section == "Supplementary evidence delta" and "\\section{Supplementary delta:" in text)
        )
        for section in sections
    }
    return {
        "sections": section_presence,
        "markers": {marker: marker in normalized for marker in markers},
        "cnn_title": "CNN" in text.split("\\maketitle", 1)[0],
    }


def build_report(*, generated_at_utc: str | None = None) -> dict[str, Any]:
    config = _load(CONFIG)
    board_text = BOARD.read_text(encoding="utf-8")
    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    final_gate = _load(FINAL_GATE)
    matrix = _load(EVIDENCE_MATRIX)
    dual_contract = _load(DUAL_CONTRACT)
    headroom = _load(HEADROOM)
    teacher = _load(LEGACY_TEACHER)
    student_validation = _load(LEGACY_STUDENT_VALIDATION)
    student = _load(LEGACY_STUDENT)
    retention = _load(LEGACY_RETENTION)
    mismatch = _load(LEGACY_MISMATCH)
    hardware = _load(LEGACY_HARDWARE)
    cnn_ablation = _load(LEGACY_CNN_ABLATION)
    learned_eligibility = _load(LEARNED_ELIGIBILITY)
    selected_restart = teacher["training_restarts"][teacher["selected_restart_index"]]
    quantized_gru = _student_candidate(hardware, "quantized_gru_int8_q14_lower_bound")
    distilled_student = _student_candidate(hardware, "distilled_student_q3_14_state4_serial")

    artifact_paths = {
        "implementation": Path(__file__).resolve(),
        "config": CONFIG,
        "task_board": BOARD,
        "new_risks": RISKS,
        "manuscript": MANUSCRIPT,
        "manuscript_contract": MANUSCRIPT_CONTRACT,
        "claim_figure_delta": CLAIM_FIGURE_DELTA,
        "dual_contract": DUAL_CONTRACT,
        "multimode_headroom": HEADROOM,
        "evidence_matrix": EVIDENCE_MATRIX,
        "final_gate": FINAL_GATE,
        "legacy_teacher": LEGACY_TEACHER,
        "legacy_student_validation": LEGACY_STUDENT_VALIDATION,
        "legacy_student": LEGACY_STUDENT,
        "legacy_retention": LEGACY_RETENTION,
        "legacy_mismatch": LEGACY_MISMATCH,
        "legacy_hardware": LEGACY_HARDWARE,
        "legacy_cnn_ablation": LEGACY_CNN_ABLATION,
        "learned_eligibility": LEARNED_ELIGIBILITY,
    }

    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "reviewer_context": config["reviewer_context"],
        "task_status": {
            "T6.26.1": _task_status(board_text, "T6.26.1"),
            "T6.26.2": _task_status(board_text, "T6.26.2"),
            "T7.3.2": _task_status(board_text, "T7.3.2"),
            "T7.3.3": _task_status(board_text, "T7.3.3"),
        },
        "current_phase6d": {
            "final_verdict": final_gate["verdict"],
            "truth_key": final_gate["truth_key"],
            "multimode": final_gate["lane_decisions"]["MULTIMODE_SOFTWARE_ALGORITHM"],
            "rtl": final_gate["lane_decisions"]["SINGLE_MODE_DETERMINISTIC_RTL"],
            "learning": final_gate["lane_decisions"]["LEARNED_APPROXIMATION_EXTENSION"],
            "publication_boundary": final_gate["publication_boundary"],
            "matrix_learning_outcome": matrix["lane_outcomes"]["LEARNED_APPROXIMATION_EXTENSION"],
            "matrix_learning_primary": matrix["evidence_boundary"]["learning_primary"],
            "headroom_verdict": headroom["verdict"],
            "headroom_strongest_baseline": headroom["strongest_development_baseline_selection"]["selected"],
            "headroom_baseline_p_l": headroom["paired_bootstrap"]["baseline_p_L"],
            "headroom_proposed_p_l": headroom["paired_bootstrap"]["proposed_p_L"],
            "headroom_point": headroom["paired_bootstrap"]["relative_improvement_point"],
            "headroom_lcb": headroom["paired_bootstrap"]["relative_improvement_lcb"],
            "dual_learning_role": next(lane for lane in dual_contract["lanes"] if lane["lane_id"] == "LEARNED_APPROXIMATION_EXTENSION"),
        },
        "legacy_learning": {
            "cnn": {
                "status": cnn_ablation["status"],
                "samples": cnn_ablation["lanes"]["cnn_residual"]["aggregate"]["samples"],
                "active_mse": cnn_ablation["lanes"]["cnn_residual"]["aggregate"]["active_mse"],
                "zero_residual_mse": cnn_ablation["lanes"]["cnn_residual"]["aggregate"]["off_mse"],
                "uncertainty_status": cnn_ablation["lanes"]["cnn_residual"]["uncertainty_status"],
                "claim_decision": cnn_ablation["lanes"]["cnn_residual"]["claim_decision"],
                "scope": cnn_ablation["lanes"]["cnn_residual"]["scope"],
                "eligibility_verdict": learned_eligibility["verdict"],
                "candidate_families": learned_eligibility["eligibility_summary"]["candidate_families"],
                "same_task_eligible": learned_eligibility["eligibility_summary"]["same_task_eligible"],
                "diagnostic_samples": learned_eligibility["diagnostic_replay"]["samples"],
                "repeat_count": learned_eligibility["diagnostic_replay"]["repeat_count"],
                "bit_exact_across_repeats": learned_eligibility["diagnostic_replay"]["bit_exact_across_repeats"],
                "claim_registry": learned_eligibility["claim_registry"],
            },
            "teacher": {
                "status": teacher["status"],
                "scope": teacher["scope"],
                "parameter_count": selected_restart["parameter_count"],
                "architecture": selected_restart["architecture"],
                "selected_restart": teacher["selected_restart_index"],
                "training_cap_hits": teacher["training_cap_hit_indices"],
                "claim_boundary": teacher["claim_boundary"],
            },
            "student": {
                "status": student_validation["status"],
                "scope": student_validation["scope"],
                "state_dimension": student["state_dimension"],
                "stored_trainable_scalars": student["resource_profile"]["stored_trainable_scalars"],
                "evaluation_mse": student_validation["comparisons"]["evaluation"]["selected_student"]["mse"],
                "evaluation_blind": student_validation["selection"]["evaluation_blind"],
                "claim_boundary": student_validation["claim_boundary"],
            },
            "retention": {
                "status": retention["status"],
                "minimum_point": min(metric["point_retention_fraction"] for split in retention["stochastic_retention"].values() for metric in split.values()),
                "minimum_ci_lower": min(metric["ci_95"][0] for split in retention["stochastic_retention"].values() for metric in split.values()),
                "claim_boundary": retention["claim_boundary"],
            },
            "mismatch": {
                "status": mismatch["status"],
                "verdict": mismatch["verdict"],
                "minimum_retention": mismatch["branch_decision"]["observed"]["retention_minimum"],
                "retention_median": mismatch["branch_decision"]["observed"]["retention_median"],
                "claim_boundary": mismatch["claim_boundary"],
            },
            "hardware": {
                "verdict": hardware["verdict"],
                "quantized_gru_cycles": quantized_gru["cxxrtl_lower_bound_cycles"],
                "quantized_gru_latency_us_at_27mhz": quantized_gru["latency_us_at_27mhz_lower_bound"],
                "quantized_gru_functional_rtl": quantized_gru["functional_model"],
                "quantized_gru_physical_gain_retention": quantized_gru["physical_gain_retention"],
                "quantized_gru_eligible": quantized_gru["enhanced_route_eligible"],
                "selected_candidate": hardware["selection"]["candidate_id"],
                "student_cycles": distilled_student["cxxrtl_cycles"],
                "student_present_in_current_rtl": False,
                "evidence_boundary": hardware["evidence_boundary"],
            },
        },
        "task_signature": {
            "legacy_role": "finite_cutoff_two_level_sbs_controller_action_imitation",
            "phase6d_role": "multimode_posterior_or_exact_logical_coset_mld_approximation",
            "same_task": False,
            "migration_allowed": False,
            "authorized_teacher_targets": config["authorized_teacher_targets"],
            "forbidden_online_inputs": ["truth", "scenario_identity", "future_label"],
        },
        "promotion_gate": {
            "required_fields": config["required_promotion_fields"],
            "failure_disposition": "DROPPED_TO_ABLATION",
            "can_change_classical_algorithm_verdict": False,
            "can_change_rtl_verdict": False,
        },
        "manuscript": _manuscript_checks(manuscript_text, config["required_manuscript_markers"]),
        "response_package": {
            "strategy": {
                "overall_posture": "accept framing concern; disclose legacy positives; separate task signatures; retain future promotion gate",
                "major_risks": [
                    "hiding favorable legacy learning evidence",
                    "migrating controller evidence into the multimode decoder task",
                    "calling absent Phase-6D learning a primary contribution",
                    "claiming universal generalization from deletion-invariant current results",
                ],
                "suggested_order": ["current paper center", "Phase-6D disposition", "legacy evidence", "nonmigration", "future gate"],
            },
            "tracker": {
                "comment_id": config["reviewer_context"]["comment_id"],
                "concern": config["reviewer_context"]["reviewer_concern"],
                "category": config["reviewer_context"]["category"],
                "severity": config["reviewer_context"]["severity"],
                "actions": config["reviewer_context"]["actions"],
                "manuscript_locations": ["Abstract", "Introduction", "Methods: Replaceable CNN/student extension", "Discussion", "Supplementary evidence delta"],
                "missing_author_input": config["reviewer_context"]["visible_placeholder"],
            },
            "english_response": _response_text(),
            "manuscript_change_checklist": [
                "Keep the two primary lanes explicit in Abstract and Introduction.",
                "Keep the Phase-6D teacher/training/formal absence explicit in Methods and Results.",
                "Keep legacy positive results task-local and non-migrating in the response and Supplement.",
                "Keep future promotion fields and ablation fallback explicit in Discussion.",
            ],
            "missing_information": [config["reviewer_context"]["visible_placeholder"]],
            "package_readiness": config["reviewer_context"]["package_readiness"],
        },
        "response_rows": config["response_rows"],
        "forbidden_response_phrases": config["forbidden_response_phrases"],
        "artifact_registry": {key: _binding(path) for key, path in artifact_paths.items()},
    }
    report["gates"] = evaluate_gates(report, check_live_sources=False)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_CNN_CENTRIC_REVIEWER_CONTRACT"
    report["semantic_mutation_audit"] = _mutation_audit(report)
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    current = report["current_phase6d"]
    legacy = report["legacy_learning"]
    package = report["response_package"]
    rows = report["response_rows"]
    text = package["english_response"]
    statuses = report["task_status"]
    gates = {
        "G01_identity": report.get("task_id") == TASK_ID and report.get("schema_version") == SCHEMA_VERSION,
        "G02_preemptive_context_and_readiness_honest": report["reviewer_context"]["comment_origin"].startswith("preemptive") and package["package_readiness"] == "draft_with_placeholders" and package["missing_information"] == ["ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING"],
        "G03_task_board_lifecycle": statuses["T6.26.1"] == "Dropped" and statuses["T6.26.2"] == "Dropped" and statuses["T7.3.2"] == "Done" and statuses["T7.3.3"] in {"In Progress", "Done"},
        "G04_final_truth_table_exact": current["final_verdict"] == "GO_RTL_ONLY" and current["truth_key"] == "multimode=false,rtl=true" and current["multimode"]["gate_passed"] is False and current["rtl"]["gate_passed"] is True,
        "G05_learning_dropped_absent_no_vote": current["learning"]["decision"] == "DROPPED_ABLATION_ONLY" and current["learning"]["direct_evidence"]["changes_overall_verdict"] is False and current["matrix_learning_outcome"] == "DROPPED_ABSENT" and current["matrix_learning_primary"] is False,
        "G06_primary_verdict_is_learning_deletion_invariant": package["strategy"]["suggested_order"][0] == "current paper center" and current["publication_boundary"]["learning_primary"] is False and report["promotion_gate"]["can_change_classical_algorithm_verdict"] is False and report["promotion_gate"]["can_change_rtl_verdict"] is False,
        "G07_legacy_teacher_positive_and_scoped": legacy["teacher"]["status"] == "PASS" and legacy["teacher"]["parameter_count"] == 72_853 and legacy["teacher"]["architecture"] == "GRU10-DENSE256-DENSE256-OUT15" and "finite-cutoff two-level" in legacy["teacher"]["claim_boundary"]["allowed"] and len(legacy["teacher"]["training_cap_hits"]) == 2,
        "G08_legacy_student_positive_and_scoped": legacy["student"]["status"] == "PASS" and legacy["student"]["state_dimension"] == 4 and legacy["student"]["stored_trainable_scalars"] == 95 and abs(legacy["student"]["evaluation_mse"] - 6.083136156367311e-06) < 1e-18 and legacy["student"]["evaluation_blind"] is True and "imitation" in legacy["student"]["claim_boundary"]["allowed"],
        "G09_legacy_retention_positive_and_nonmigrating": legacy["retention"]["status"] == "PASS" and abs(legacy["retention"]["minimum_point"] - 0.9814573586937879) < 1e-15 and abs(legacy["retention"]["minimum_ci_lower"] - 0.9445014278749587) < 1e-15 and "finite-model" in legacy["retention"]["claim_boundary"]["allowed"],
        "G10_legacy_ood_is_bounded_not_universal": legacy["mismatch"]["status"] == "PASS" and abs(legacy["mismatch"]["minimum_retention"] - 0.8976304408841681) < 1e-15 and legacy["mismatch"]["claim_boundary"]["forbidden"].find("universal robustness") >= 0,
        "G11_full_gru_dropped_and_legacy_student_not_current_rtl": legacy["hardware"]["verdict"] == "DISTILLED_STUDENT_ONLY_QUANTIZED_GRU_DROPPED_FULL_GRU_OFFLINE_TEACHER" and legacy["hardware"]["quantized_gru_cycles"] == 72_854 and legacy["hardware"]["quantized_gru_functional_rtl"] is False and legacy["hardware"]["quantized_gru_physical_gain_retention"] is None and legacy["hardware"]["quantized_gru_eligible"] is False and legacy["hardware"]["selected_candidate"] == "distilled_student_q3_14_state4_serial" and legacy["hardware"]["student_present_in_current_rtl"] is False,
        "G12_task_signatures_are_explicitly_nonmatching": report["task_signature"]["same_task"] is False and report["task_signature"]["migration_allowed"] is False and report["task_signature"]["legacy_role"] != report["task_signature"]["phase6d_role"],
        "G13_teacher_targets_and_online_privilege_closed": report["task_signature"]["authorized_teacher_targets"] == ["posterior", "log_likelihood_ratio", "logical_coset_probability", "action"] and report["task_signature"]["forbidden_online_inputs"] == ["truth", "scenario_identity", "future_label"],
        "G14_promotion_gate_is_matched_complete_and_fail_closed": len(report["promotion_gate"]["required_fields"]) == 13 and {"matched_classical_approximation_budget", "worst_family_retention", "held_out_ood_retention", "formal_retention_lower_bound", "compression_or_cost_benefit"}.issubset(report["promotion_gate"]["required_fields"]) and report["promotion_gate"]["failure_disposition"] == "DROPPED_TO_ABLATION",
        "G15_manuscript_locations_and_markers_complete": all(report["manuscript"]["sections"].values()) and all(report["manuscript"]["markers"].values()) and report["manuscript"]["cnn_title"] is False and len(package["tracker"]["manuscript_locations"]) == 5,
        "G16_response_directly_accepts_framing_and_answers_centrality": "earlier CNN--FPGA framing could make the learned component appear more central" in text and "revised manuscript is not CNN-centric" in text and "no independent vote" in text,
        "G17_response_discloses_legacy_positive_and_remaining_risk": all(token in text for token in ["72,853-parameter", "four-state, 95-scalar", "6.083136e-6", "0.981457/0.944501", "do not claim that every future learned model will generalize"]),
        "G18_response_contains_no_forbidden_overclaim": not any(phrase.lower() in text.lower() for phrase in report["forbidden_response_phrases"]),
        "G19_artifact_bindings_complete_and_live": len(report["artifact_registry"]) == 19 and (not check_live_sources or all(_binding_live(binding) for binding in report["artifact_registry"].values())),
        "G20_response_rows_are_complete_unique_and_stateful": len(rows) == 24 and len({row["row_id"] for row in rows}) == 24 and {row["response_state"] for row in rows} == RESPONSE_STATES and all(row["claim"] and row["boundary"] and row["source_ids"] for row in rows),
        "G21_source_data_is_lossless": (not check_live_sources) or _source_data_matches(report),
        "G22_response_package_is_traceable_and_actionable": package["tracker"]["comment_id"] == "PRQ-CNN-1" and package["tracker"]["severity"] == "major" and package["tracker"]["actions"] == ["CLARIFY_EXISTING", "SOFTEN_CLAIM", "PARTIAL"] and len(package["manuscript_change_checklist"]) == 4 and bool(package["missing_information"]),
        "G23_legacy_cnn_is_exactly_replayed_but_ineligible": legacy["cnn"]["status"] == "PASS" and legacy["cnn"]["samples"] == 206 and legacy["cnn"]["diagnostic_samples"] == 206 and legacy["cnn"]["repeat_count"] == 5 and legacy["cnn"]["bit_exact_across_repeats"] is True and abs(legacy["cnn"]["active_mse"] - 2.4144528544831194e-06) < 1e-18 and abs(legacy["cnn"]["zero_residual_mse"] - 8.0340452043047e-06) < 1e-18 and legacy["cnn"]["candidate_families"] == 16 and legacy["cnn"]["same_task_eligible"] == 0 and legacy["cnn"]["claim_registry"]["LEGACY_CNN_PARAMETER_REPLAY"] == "DIAGNOSTIC_EXACT_INELIGIBLE" and "not LER or control gain" in legacy["cnn"]["scope"],
    }
    return gates


def _mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[tuple[str, str, Callable[[dict[str, Any]], None]]] = [
        ("M01_wrong_identity", "G01_identity", lambda x: x.update(task_id="T7.3.X")),
        ("M02_hide_placeholder", "G02_preemptive_context_and_readiness_honest", lambda x: x["response_package"].update(package_readiness="ready_to_submit")),
        ("M03_reopen_completed_task", "G03_task_board_lifecycle", lambda x: x["task_status"].update({"T7.3.2": "In Progress"})),
        ("M04_promote_global_go", "G04_final_truth_table_exact", lambda x: x["current_phase6d"].update(final_verdict="GO_TWO_LANE")),
        ("M05_make_learning_primary", "G05_learning_dropped_absent_no_vote", lambda x: x["current_phase6d"].update(matrix_learning_primary=True)),
        ("M06_allow_learning_vote", "G06_primary_verdict_is_learning_deletion_invariant", lambda x: x["promotion_gate"].update(can_change_rtl_verdict=True)),
        ("M07_forge_teacher_size", "G07_legacy_teacher_positive_and_scoped", lambda x: x["legacy_learning"]["teacher"].update(parameter_count=95)),
        ("M08_forge_student_dimension", "G08_legacy_student_positive_and_scoped", lambda x: x["legacy_learning"]["student"].update(state_dimension=2)),
        ("M09_promote_retention", "G09_legacy_retention_positive_and_nonmigrating", lambda x: x["legacy_learning"]["retention"].update(minimum_ci_lower=1.0)),
        ("M10_claim_universal_ood", "G10_legacy_ood_is_bounded_not_universal", lambda x: x["legacy_learning"]["mismatch"]["claim_boundary"].update(forbidden="")),
        ("M11_promote_quantized_gru", "G11_full_gru_dropped_and_legacy_student_not_current_rtl", lambda x: x["legacy_learning"]["hardware"].update(quantized_gru_eligible=True)),
        ("M12_migrate_legacy_task", "G12_task_signatures_are_explicitly_nonmatching", lambda x: x["task_signature"].update(same_task=True, migration_allowed=True)),
        ("M13_add_truth_target", "G13_teacher_targets_and_online_privilege_closed", lambda x: x["task_signature"]["authorized_teacher_targets"].append("truth")),
        ("M14_drop_worst_family_gate", "G14_promotion_gate_is_matched_complete_and_fail_closed", lambda x: x["promotion_gate"]["required_fields"].remove("worst_family_retention")),
        ("M15_erase_manuscript_marker", "G15_manuscript_locations_and_markers_complete", lambda x: x["manuscript"]["markers"].update({next(iter(x["manuscript"]["markers"])): False})),
        ("M16_remove_direct_answer", "G16_response_directly_accepts_framing_and_answers_centrality", lambda x: x["response_package"].update(english_response="We thank the reviewer.")),
        ("M17_hide_legacy_positive", "G17_response_discloses_legacy_positive_and_remaining_risk", lambda x: x["response_package"].update(english_response=x["response_package"]["english_response"].replace("72,853-parameter", "learned"))),
        ("M18_add_overclaim", "G18_response_contains_no_forbidden_overclaim", lambda x: x["response_package"].update(english_response=x["response_package"]["english_response"] + " CNN proves multimode SOTA.")),
        ("M19_corrupt_binding", "G19_artifact_bindings_complete_and_live", lambda x: x["artifact_registry"]["final_gate"].update(sha256="0" * 64)),
        ("M20_duplicate_row", "G20_response_rows_are_complete_unique_and_stateful", lambda x: x["response_rows"][1].update(row_id=x["response_rows"][0]["row_id"])),
        ("M21_corrupt_source_data", "G21_source_data_is_lossless", lambda x: x["response_rows"][0].update(claim="corrupted")),
        ("M22_remove_action_map", "G22_response_package_is_traceable_and_actionable", lambda x: x["response_package"]["tracker"].update(actions=[])),
        ("M23_promote_legacy_cnn", "G23_legacy_cnn_is_exactly_replayed_but_ineligible", lambda x: x["legacy_learning"]["cnn"].update(same_task_eligible=1)),
    ]
    results = []
    for mutation_id, target_gate, mutate in cases:
        mutated = copy.deepcopy(report)
        mutate(mutated)
        rejected = not evaluate_gates(mutated, check_live_sources=(target_gate in {"G19_artifact_bindings_complete_and_live", "G21_source_data_is_lossless"}))[target_gate]
        results.append({"mutation_id": mutation_id, "target_gate": target_gate, "rejected": rejected})
    return {"count": len(results), "detected": sum(case["rejected"] for case in results), "cases": results}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as stream:
        stream.write(text)
        temp = Path(stream.name)
    os.replace(temp, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_source_data(report: Mapping[str, Any], path: Path) -> None:
    rows = _source_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temp = Path(stream.name)
    os.replace(temp, path)


def _markdown(report: Mapping[str, Any]) -> str:
    legacy = report["legacy_learning"]
    current = report["current_phase6d"]
    package = report["response_package"]
    lines = [
        "# T7.3.2：CNN-centric / simulator-overfitting 审稿风险回答",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- gates：`{report['gate_summary']['passed']}/{report['gate_summary']['total']}`",
        f"- semantic mutations：`{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`",
        f"- package readiness：`{package['package_readiness']}`",
        "",
        "## Response strategy summary",
        "",
        "- Decision type: unclear; this is a pre-emptive reviewer-risk package, not a supplied decision letter.",
        "- Overall posture: accept the framing concern, disclose favorable legacy learning evidence, separate task signatures, and retain a strict future promotion gate.",
        "- Major risk: either hiding positive T4.4 evidence or migrating it into the current multimode/RTL task would be misleading.",
        "- Suggested ordering: current paper center -> Phase-6D disposition -> legacy evidence -> nonmigration -> future gate.",
        "",
        "## Comment-response tracker",
        "",
        "| ID | Reviewer concern | Type | Severity | Proposed action | Missing author input |",
        "| --- | --- | --- | --- | --- | --- |",
        "| PRQ-CNN-1 | Is the CNN merely overfitting the simulator, and is the project still CNN-centric? | evidence / methodology / positioning | major | CLARIFY_EXISTING + SOFTEN_CLAIM + PARTIAL | Actual reviewer ID and verbatim wording |",
        "",
        "## Draft point-by-point response letter",
        "",
        "> **Placeholder:** replace `PRQ-CNN-1` with the actual reviewer ID and paste the verbatim reviewer wording before submission.",
        "",
        package["english_response"],
        "",
        "## Evidence audit",
        "",
        f"- Current verdict: `{current['final_verdict']}`; learning=`{current['learning']['decision']}` and changes-overall-verdict=`{str(current['learning']['direct_evidence']['changes_overall_verdict']).lower()}`.",
        f"- Current multimode entry: `{current['headroom_verdict']}`, strongest=`{current['headroom_strongest_baseline']}`, baseline/proposed={current['headroom_baseline_p_l']:.10f}/{current['headroom_proposed_p_l']:.10f}, point/LCB={current['headroom_point']:.1%}/{current['headroom_lcb']:.1%}.",
        f"- Legacy teacher: {legacy['teacher']['parameter_count']:,} parameters, `{legacy['teacher']['architecture']}`, cap hits={legacy['teacher']['training_cap_hits']}.",
        f"- Legacy CNN: {legacy['cnn']['samples']} samples, five bit-exact repeats, active/zero MSE={legacy['cnn']['active_mse']:.9g}/{legacy['cnn']['zero_residual_mse']:.9g}; same-task eligible={legacy['cnn']['same_task_eligible']}/{legacy['cnn']['candidate_families']}.",
        f"- Legacy student: {legacy['student']['state_dimension']} states / {legacy['student']['stored_trainable_scalars']} scalars, evaluation MSE={legacy['student']['evaluation_mse']:.9g}.",
        f"- Legacy retention: minimum point/CI-lower={legacy['retention']['minimum_point']:.6f}/{legacy['retention']['minimum_ci_lower']:.6f}; mismatch minimum={legacy['mismatch']['minimum_retention']:.6f}.",
        f"- Hardware boundary: full quantized GRU lower bound={legacy['hardware']['quantized_gru_cycles']:,} cycles and ineligible; historical selected student={legacy['hardware']['selected_candidate']}, but present-in-current-RTL=false.",
        "",
        "## Manuscript change checklist",
        "",
        *[f"- {item}" for item in package["manuscript_change_checklist"]],
        "",
        "## Missing information / risk flags",
        "",
        "- `ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING`: scientific substance is complete, but the package must not be labelled submission-ready until the actual comment is supplied.",
        "- The response establishes deletion-invariance of current claims with respect to learning; it does not establish universal generalization of any future CNN/student.",
        "",
        "## 中文核对",
        "",
        "- 当前论文不是 CNN-centric：两个主 lane 分别由 multimode strongest-baseline gate 和 exact single-mode RTL gate 决定，learning 没有投票权。",
        "- 不能隐藏 T4.4 的正结果；必须明确它们属于另一 finite-model controller task，不能迁移成 Phase 6D multimode/RTL 证据。",
        "- 实际返修信提交前，只需替换 reviewer ID 并粘贴原始评论；不要虚构编辑决定、reviewer 身份或行号。",
        "",
        "## 原子证据与边界",
        "",
        "| ID | 状态 | 主题 | 主张 | 边界 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["response_rows"]:
        lines.append(f"| {row['row_id']} | `{row['response_state']}` | {row['topic']} | {row['claim']} | {row['boundary']} |")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any]) -> None:
    _write_source_data(report, DEFAULT_SOURCE_DATA)
    report["source_data"] = _binding(DEFAULT_SOURCE_DATA) | {"rows": len(_source_rows(report))}
    report["markdown"] = {"path": DEFAULT_MARKDOWN.relative_to(ROOT).as_posix()}
    _atomic_json(DEFAULT_REPORT, report)
    _atomic_text(DEFAULT_MARKDOWN, _markdown(report))
    report["markdown"] = _binding(DEFAULT_MARKDOWN)
    _atomic_json(DEFAULT_REPORT, report)


def verify_report() -> tuple[bool, dict[str, bool]]:
    if not DEFAULT_REPORT.exists() or not DEFAULT_SOURCE_DATA.exists() or not DEFAULT_MARKDOWN.exists():
        return False, {"outputs_exist": False}
    stored = _load(DEFAULT_REPORT)
    fresh = build_report(generated_at_utc=stored.get("generated_at_utc"))
    checks = {
        "outputs_exist": True,
        "identity": stored.get("task_id") == TASK_ID and stored.get("schema_version") == SCHEMA_VERSION,
        "verdict": stored.get("verdict") == VERDICT and fresh.get("verdict") == VERDICT,
        "all_gates": all(evaluate_gates(stored, check_live_sources=True).values()),
        "all_mutations": stored["semantic_mutation_audit"]["count"] == stored["semantic_mutation_audit"]["detected"] == len(stored["gates"]),
        "source_data": _source_data_matches(stored),
        "markdown_live": _binding_live(stored["markdown"]),
        "analysis_live": stored.get("analysis_sha256") == _canonical_sha256(_analysis_payload(stored)),
        "fresh_analysis": stored.get("analysis_sha256") == fresh.get("analysis_sha256"),
    }
    return all(checks.values()), checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        ok, checks = verify_report()
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    report = build_report()
    write_outputs(report)
    print(json.dumps({
        "verdict": report["verdict"],
        "gates": report["gate_summary"],
        "mutations": {"detected": report["semantic_mutation_audit"]["detected"], "total": report["semantic_mutation_audit"]["count"]},
        "source_rows": len(report["response_rows"]),
        "package_readiness": report["response_package"]["package_readiness"],
        "analysis_sha256": report["analysis_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
