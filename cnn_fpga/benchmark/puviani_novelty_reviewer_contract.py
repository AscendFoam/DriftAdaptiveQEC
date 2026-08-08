"""Build the T7.3.5 Puviani/NMF relationship reviewer-response contract."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T7.3.5"
SCHEMA_VERSION = "t7.3.5-puviani-novelty-reviewer-contract-v1"
VERDICT = "PASS_PUVIANI_RELATIONSHIP_SEPARATED_WITH_SURPASS_PROHIBITED"

CONFIG = ROOT / "configs/phase6d/t7_3_5_puviani_novelty_reviewer_contract.json"
BOARD = ROOT / "docs/new_task_board.md"
RISKS = ROOT / "docs/new_risks.md"
EXPERIMENT_PLAN = ROOT / "docs/experiment_plan.md"
MANUSCRIPT = ROOT / "docs/paper_notes/Phase6D_Dual_Lane_GKP_manuscript.tex"
MANUSCRIPT_CONTRACT = ROOT / "docs/t7_2_6_phase6d_manuscript_delta.json"
GQF_RUNNER = ROOT / "configs/gqf_official/runner_manifest.json"
OFFICIAL_INTAKE = ROOT / "docs/t6_8_3_gqf_official_intake.json"
EXACT_REPRODUCTION = ROOT / "docs/t6_8_4_gqf_paper_exact_reproduction.json"
EXACT_SOURCE_DATA = ROOT / "docs/t6_8_4_gqf_reproduction_source_data.csv"
MATCHED_GATE = ROOT / "docs/t6_8_5_gqf_route_a_matched_comparison_gate.json"
MATCHED_SOURCE_DATA = ROOT / "docs/t6_8_5_gqf_route_a_matched_comparison_gate_source_data.csv"
PROJECT_NMF = ROOT / "docs/t2_3_7_nmf_directional_ranking.json"
PROJECT_NMF_CSV = ROOT / "docs/t2_3_7_nmf_directional_ranking.csv"
PROJECT_NMF_CHECKPOINT = ROOT / "docs/t2_3_7_nmf_directional_ranking_checkpoints.pt"
PUVIANI_PAPER = ROOT / (
    "relative_papers/Non-Markovian_feedback_for_optimized_quantum_error_correction/"
    "Non-Markovian_feedback_for_optimized_quantum_error_correction.md"
)
LEARNED_ELIGIBILITY = ROOT / "docs/t6_17_3_learned_model_eligibility_replay.json"
LEARNING_RESPONSE = ROOT / "docs/t7_3_2_cnn_centric_reviewer_contract.json"
HEADROOM = ROOT / "docs/t6_20_4_multimode_causal_headroom.json"
HEADROOM_SOURCE_DATA = ROOT / "docs/t6_20_4_multimode_causal_headroom_source_data.csv"
TAIL_MATRIX = ROOT / "docs/t6_7_2_abrupt_ood_tail_formal_matrix.json"
TAIL_SOURCE_DATA = ROOT / "docs/t6_7_2_abrupt_ood_tail_formal_matrix_source_data.csv"
RTL_FORMAL = ROOT / "docs/t6_25_2_converged_rtl_formal.json"
RTL_FORMAL_SOURCE_DATA = ROOT / "docs/t6_25_2_converged_rtl_formal_source_data.csv"
RTL_LONG = ROOT / "docs/t6_25_3_converged_long_rtl.json"
RTL_LONG_SOURCE_DATA = ROOT / "docs/t6_25_3_converged_long_rtl_source_data.csv"
RTL_HARDWARE = ROOT / "docs/t6_25_4_converged_hardware.json"
RTL_HARDWARE_SOURCE_DATA = ROOT / "docs/t6_25_4_converged_hardware_source_data.csv"
FINAL_GATE = ROOT / "docs/t6_26_4_final_dual_lane_gate.json"

DEFAULT_REPORT = ROOT / "docs/t7_3_5_puviani_novelty_reviewer_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_3_5_puviani_novelty_reviewer_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/puviani_novelty_reviewer_response.md"

RESPONSE_STATES = {
    "REVIEWER_CONCERN",
    "DIRECT_ANSWER",
    "OFFICIAL_INTAKE",
    "OFFICIAL_EXACT_STATUS",
    "REDUCED_DIAGNOSTIC",
    "MATCHED_COMPARISON",
    "PROJECT_NATIVE_DIRECTIONAL",
    "COUNTEREVIDENCE",
    "TASK_SIGNATURE",
    "CURRENT_PRIMARY",
    "DISTINCT_CONTRIBUTION",
    "LIMITATION",
    "FUTURE_PROMOTION",
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
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "binding_kind": "file_sha256",
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _task_status(board: str, task_id: str) -> str:
    match = re.search(
        rf"^\|\s*{re.escape(task_id)}\s*\|\s*([^|]+?)\s*\|",
        board,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"task status not found: {task_id}")
    return match.group(1).strip()


def _board_projection(text: str) -> dict[str, Any]:
    return {
        "schema_version": "t7.3.5-board-projection-v1",
        "self_done": _task_status(text, "T7.3.5") == "Done",
        "phase9_started": _task_status(text, "T9.1.1") in {"In Progress", "Done"},
        "official_asset_lane_blocked": _task_status(text, "T9.1.2") == "Blocked",
        "phase9_present": "## Phase 9：Performance-first 单模 GKP 多速率非马尔可夫双回路重启" in text,
        "official_and_paper_constrained_split": "official/paper-constrained 双 lane" in text,
    }


def _risk_projection(text: str) -> dict[str, Any]:
    return {
        "schema_version": "t7.3.5-risk-projection-v1",
        "official_asset_risk_present": "R-N162" in text,
        "nmf_conflation_risk_present": "R-N169" in text,
        "task_audit_present": "| 2026-07-22 | T7.3.5 |" in text,
    }


def _plan_projection(text: str) -> dict[str, Any]:
    markers = (
        "Puviani 资产缺失的非阻塞证据合同",
        "OFFICIAL_EXACT_REPRODUCTION",
        "PAPER_CONSTRAINED_REIMPLEMENTATION",
        "双后端、action-conditioned 数字孪生",
        "六态 formal 与高速板 HIL 门",
    )
    return {
        "schema_version": "t7.3.5-plan-projection-v1",
        "markers": {marker: marker in text for marker in markers},
    }


def _contract_projection(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    return {
        key: payload[key]
        for key in ("task_id", "schema_version", "verdict", "gate_summary", "analysis_sha256")
    }


def _semantic_binding(path: Path, selector: str, projection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "binding_kind": "semantic_projection",
        "selector": selector,
        "projection": copy.deepcopy(dict(projection)),
        "sha256": _canonical_sha256(projection),
    }


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    if not path.is_file():
        return False
    if binding.get("binding_kind") == "file_sha256":
        return (
            path.stat().st_size == int(binding["bytes"])
            and _sha256(path) == str(binding["sha256"])
        )
    if binding.get("binding_kind") != "semantic_projection":
        return False
    text = path.read_text(encoding="utf-8")
    projectors = {
        "t7.3.5_board": _board_projection,
        "t7.3.5_risks": _risk_projection,
        "t7.3.5_plan": _plan_projection,
        "t7.3.5_parent_contract": _contract_projection,
    }
    selector = str(binding.get("selector"))
    if selector not in projectors:
        return False
    projection = projectors[selector](text)
    return (
        projection == binding.get("projection")
        and _canonical_sha256(projection) == str(binding["sha256"])
    )


def _mean(summary: Mapping[str, Any], split: str, method: str) -> float:
    return float(summary[split][method]["logical_z_effective_lifetime_cycles"]["mean"])


def _all_exact_outcomes_null(outcomes: Mapping[str, Any]) -> bool:
    fields = ("T_X", "T_Y", "T_Z", "T_ch", "F_avg")
    return set(outcomes) == {"standard", "MF", "NMF"} and all(
        row["status"] == "NOT_RUN_EXACT_PREREQUISITE_FAIL"
        and all(row[field] is None for field in fields)
        for row in outcomes.values()
    )


def _response_text() -> str:
    return (
        "No. More precisely, the manuscript neither presents an official reproduction of Puviani et al. "
        "nor claims to outperform their NMF controller. The conceptual overlap is the use of measurement "
        "history; the decision problems are different. Puviani et al. optimize fifteen physical sBs control "
        "parameters in a single-mode cavity protocol and evaluate six-state logical-channel lifetime. Our "
        "current multimode software lane maps observed syndromes to logical actions and evaluates per-round "
        "LER, while the independent single-mode RTL lane implements a bounded MAP/event transaction path.\n\n"
        "We imported and audited the public GQF source at commit c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d. "
        "The paper-exact qualification passed 0/15 criteria: official checkpoints, twenty-agent seeds, the "
        "selection ledger and the six-state evaluator are unavailable, and all exact Standard/MF/NMF "
        "T_X, T_Y, T_Z, T_ch and F_avg fields remain null. The only official-code-derived execution is a "
        "patched, reduced Standard-path diagnostic at cutoff 8: six states, three seeds, 36 trajectories, "
        "378 environment steps and 756 rows. It contains no MF/NMF training or lifetime comparison. The "
        "matched Route-A comparison therefore follows an ineligible negative branch and all thirteen result "
        "fields remain null.\n\n"
        "For completeness, a separate project-native finite-horizon study found cutoff-12 logical-Z "
        "area-equivalent lifetimes of 2.747662, 6.534671 and 6.740785 cycles for Standard, MF and NMF, with "
        "NMF-minus-MF 0.206114 and paired 95% CI [0.084161, 0.328067]. This is not an official replay. "
        "Moreover, at cutoff 16 the latest-only history-reset ablation reaches 8.271987 cycles, above NMF's "
        "7.708351, so we do not claim a universal memory mechanism.\n\n"
        "The current primary evidence is deliberately more limited. The multimode method ties the strongest "
        "static-mixture exact MLD baseline at p_L=0.111979 over 79,872 development rounds, yielding 0% "
        "relative improvement and a NO-GO algorithm verdict. The tail policy establishes a scoped fail-closed "
        "safety contract, but calibration-step fallback is 0.958546 and its worst window ties the locked "
        "baseline at 181 errors. Historical teacher-to-student retention/compression remains an ablation: "
        "zero of sixteen learned families is same-task eligible and the student is absent from the current RTL.\n\n"
        "The distinct positive result is an exact pre-board digital-system contribution: seventeen formal "
        "gates, twenty-one killed mutants, one million CXXRTL cycles with zero full-vector mismatch, six-cycle "
        "latency and initiation interval one, together with atomic A/B publication, CRC/version checking and "
        "last-known-good fail-closed recovery. It is not an NMF controller, a physical lifetime result, a board "
        "measurement or a fastest-FPGA claim. Any future Puviani-surpass statement requires a protocol-matched, "
        "six-state, no-postselection lifetime experiment with identical observation/action/environment and "
        "training, wall-clock and compute budgets, plus a positive simultaneous paired 95% lower confidence bound."
    )


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    omitted = {
        "generated_at_utc",
        "analysis_sha256",
        "semantic_mutation_audit",
        "source_data",
        "markdown",
    }
    return {key: value for key, value in report.items() if key not in omitted}


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "row_id": row["row_id"],
            "response_state": row["response_state"],
            "topic": row["topic"],
            "claim": row["claim"],
            "boundary": row["boundary"],
            "source_ids_json": _canonical(row["source_ids"]),
            "row_sha256": _canonical_sha256(row),
        }
        for row in report["response_rows"]
    ]


def _source_data_matches(report: Mapping[str, Any], path: Path = DEFAULT_SOURCE_DATA) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream)) == _source_rows(report)


def build_report(*, generated_at_utc: str | None = None) -> dict[str, Any]:
    config = _load(CONFIG)
    board_text = BOARD.read_text(encoding="utf-8")
    risks_text = RISKS.read_text(encoding="utf-8")
    plan_text = EXPERIMENT_PLAN.read_text(encoding="utf-8")
    manuscript_text = MANUSCRIPT.read_text(encoding="utf-8")
    runner = _load(GQF_RUNNER)
    intake = _load(OFFICIAL_INTAKE)
    exact = _load(EXACT_REPRODUCTION)
    matched = _load(MATCHED_GATE)
    project = _load(PROJECT_NMF)
    eligibility = _load(LEARNED_ELIGIBILITY)
    learning = _load(LEARNING_RESPONSE)
    headroom = _load(HEADROOM)
    tail = _load(TAIL_MATRIX)
    formal = _load(RTL_FORMAL)
    long_run = _load(RTL_LONG)
    hardware = _load(RTL_HARDWARE)
    final_gate = _load(FINAL_GATE)
    manuscript_contract = _load(MANUSCRIPT_CONTRACT)

    discrepancy_count = len(exact["source_discrepancies"])
    blocking_discrepancies = sum(row["blocking"] for row in exact["source_discrepancies"])
    agent_fields = ("seed", "checkpoint_sha256", "selection_metric", "T_X", "T_Y", "T_Z", "T_ch", "F_avg")
    agent_rows_all_null = all(
        all(row[field] is None for field in agent_fields)
        and row["status"] == "MISSING_OFFICIAL_AGENT_ARTIFACT"
        for row in exact["agent_ledger"]
    )
    primary = {
        method: _mean(project["summary"], "primary", method)
        for method in ("standard", "mf", "nmf", "nmf_latest_only")
    }
    confirmation = {
        method: _mean(project["summary"], "confirmation", method)
        for method in ("standard", "mf", "nmf", "nmf_latest_only")
    }
    pair = project["paired_bootstrap"]["nmf_minus_mf_logical_z_lifetime"]
    action_metrics = {row["family"]: row for row in tail["analysis"]["action_metrics_by_family"]}
    board_projection = _board_projection(board_text)
    risk_projection = _risk_projection(risks_text)
    plan_projection = _plan_projection(plan_text)

    artifact_paths = {
        "implementation": Path(__file__).resolve(),
        "config": CONFIG,
        "manuscript": MANUSCRIPT,
        "gqf_runner_manifest": GQF_RUNNER,
        "official_intake": OFFICIAL_INTAKE,
        "exact_reproduction": EXACT_REPRODUCTION,
        "exact_source_data": EXACT_SOURCE_DATA,
        "matched_gate": MATCHED_GATE,
        "matched_source_data": MATCHED_SOURCE_DATA,
        "project_nmf": PROJECT_NMF,
        "project_nmf_csv": PROJECT_NMF_CSV,
        "project_nmf_checkpoint": PROJECT_NMF_CHECKPOINT,
        "puviani_paper": PUVIANI_PAPER,
        "learned_eligibility": LEARNED_ELIGIBILITY,
        "headroom": HEADROOM,
        "headroom_source_data": HEADROOM_SOURCE_DATA,
        "tail_matrix": TAIL_MATRIX,
        "tail_source_data": TAIL_SOURCE_DATA,
        "rtl_formal": RTL_FORMAL,
        "rtl_formal_source_data": RTL_FORMAL_SOURCE_DATA,
        "rtl_long": RTL_LONG,
        "rtl_long_source_data": RTL_LONG_SOURCE_DATA,
        "rtl_hardware": RTL_HARDWARE,
        "rtl_hardware_source_data": RTL_HARDWARE_SOURCE_DATA,
        "final_gate": FINAL_GATE,
    }
    artifact_registry = {key: _binding(path) for key, path in artifact_paths.items()}
    artifact_registry.update(
        {
            "task_board": _semantic_binding(BOARD, "t7.3.5_board", board_projection),
            "new_risks": _semantic_binding(RISKS, "t7.3.5_risks", risk_projection),
            "experiment_plan": _semantic_binding(
                EXPERIMENT_PLAN,
                "t7.3.5_plan",
                plan_projection,
            ),
            "manuscript_contract": _semantic_binding(
                MANUSCRIPT_CONTRACT,
                "t7.3.5_parent_contract",
                _contract_projection(MANUSCRIPT_CONTRACT.read_text(encoding="utf-8")),
            ),
            "learning_response": _semantic_binding(
                LEARNING_RESPONSE,
                "t7.3.5_parent_contract",
                _contract_projection(LEARNING_RESPONSE.read_text(encoding="utf-8")),
            ),
        }
    )

    task_signatures = {
        "axes": config["task_signature_axes"],
        "puviani_physical_controller": {
            "observation": "logical-measurement history used by the paper/controller",
            "decision_object": "physical recovery-protocol parameter optimization",
            "action": "15 continuous sBs control parameters",
            "environment": "single-mode finite-energy cavity GKP plus ancillary qubit",
            "endpoint": "six-state logical-channel lifetime T_ch",
            "horizon": "10-cycle training and 1000-cycle evaluation",
            "training_selection_budget": "20 agents, 1000 epochs, best-agent selection",
            "online_compute_budget": "paper controller lane; not matched to project RTL",
            "evidence_level": "literature anchors; local paper-exact replay unqualified",
        },
        "project_multimode_decoder": {
            "observation": "observed causal multimode syndrome history",
            "decision_object": "logical-coset/Pauli decision under drift",
            "action": "bounded logical action and trusted-bank fallback",
            "environment": "synthetic multimode surface-square GKP drift suite",
            "endpoint": "per-round no-postselection p_L and tail LER",
            "horizon": "registered per-round development trajectories",
            "training_selection_budget": "frozen split and strongest deployable denominator",
            "online_compute_budget": "matched deployable software budget",
            "evidence_level": "development-only NO-GO; formal not accessed",
        },
        "project_single_mode_rtl": {
            "observation": "quantized event/syndrome plus integrity and version state",
            "decision_object": "bounded MAP/event action and transactional publication",
            "action": "two-state action, A/B commit, LKG rollback and reset/fallback",
            "environment": "exact digital single-mode top; no cavity or analog transport",
            "endpoint": "cycles, II, mismatch, formal safety and pre-board resources",
            "horizon": "formal properties plus one-million-cycle replay",
            "training_selection_budget": "not a learned-controller training task",
            "online_compute_budget": "six-cycle II=1 digital fast path",
            "evidence_level": "exact pre-board RTL; no board or physical lifetime",
        },
        "same_task": False,
        "numeric_global_leaderboard_allowed": False,
    }

    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "reviewer_context": config["reviewer_context"],
        "lifecycle": board_projection,
        "official_source_intake": {
            "verdict": intake["verdict"],
            "repository": intake["upstream"]["url"],
            "commit": intake["upstream"]["commit"],
            "tracked_files": len(intake["upstream"]["tracked_files"]),
            "tracked_tree_sha256": intake["upstream"]["tracked_tree_sha256"],
            "tracked_status_clean": intake["upstream"]["tracked_status_clean"],
            "license": intake["upstream"]["license"],
            "pristine_syntax_failures": len(intake["syntax_audit"]["upstream_failures"]),
            "isolated_patches": len(intake["patch_manifest"]["patches"]),
            "cpu_smoke": intake["smoke"]["cpu"]["status"],
            "gpu_status": intake["smoke"]["gpu"]["status"],
            "trained_checkpoints_present": runner["scope"]["trained_checkpoints_present"],
            "paper_exact_reproduction": runner["scope"]["paper_exact_reproduction"],
            "claim": runner["scope"]["claim"],
        },
        "official_exact_status": {
            "verdict": exact["verdict"],
            "status": exact["exact_reproduction_status"],
            "qualification": exact["exact_qualification"],
            "discrepancies": discrepancy_count,
            "blocking_discrepancies": blocking_discrepancies,
            "missing_required_fields": exact["missing_required_fields"],
            "guessed_fields": exact["guessed_fields"],
            "agent_rows": len(exact["agent_ledger"]),
            "agent_rows_all_null": agent_rows_all_null,
            "paper_exact_outcomes": exact["paper_exact_outcomes"],
            "all_exact_outcomes_null": _all_exact_outcomes_null(exact["paper_exact_outcomes"]),
            "published_anchors": exact["preregistration"]["published_numeric_anchors"],
            "claim_boundary": exact["claim_boundary"],
        },
        "reduced_diagnostic": {
            "scope": exact["reduced_probe"]["scope"],
            "configuration": exact["reduced_probe"]["configuration"],
            "coverage": exact["reduced_probe"]["coverage"],
            "status": exact["reduced_probe"]["status"],
            "elapsed_s": exact["reduced_probe"]["elapsed_s"],
            "contains_training": False,
            "contains_mf_or_nmf": False,
            "contains_lifetime_fit": False,
        },
        "matched_comparison_status": {
            "verdict": matched["verdict"],
            "prerequisites": matched["prerequisite_ledger"],
            "failed_prerequisites": sum(not row["passed"] for row in matched["prerequisite_ledger"]),
            "execution_branch": matched["execution_branch"],
            "comparison_run_manifest": matched["comparison_run_manifest"],
            "comparison_raw_data": matched["comparison_raw_data"],
            "metrics": matched["matched_comparison_metrics"],
            "metric_count": len(matched["matched_comparison_metrics"]),
            "all_metrics_null": all(value is None for value in matched["matched_comparison_metrics"].values()),
            "non_substitution": matched["non_substitution"],
            "claim_boundary": matched["claim_boundary"],
        },
        "project_native_directional": {
            "status": project["status"],
            "scope": project["scope"],
            "simulator_scope": project["simulator_scope"],
            "primary_logical_z_lifetime_cycles": primary,
            "confirmation_logical_z_lifetime_cycles": confirmation,
            "nmf_minus_mf": pair,
            "history_reset_primary_reduces_nmf": project["gates"]["history_reset_reduces_nmf_lifetime_and_auc"],
            "confirmation_reset_counterexample": confirmation["nmf_latest_only"] > confirmation["nmf"],
            "checkpoint_sha256": project["checkpoint"]["sha256"],
            "claim_boundary": project["claim_boundary"],
            "used_as_official_replacement": False,
        },
        "task_signatures": task_signatures,
        "current_phase6d": {
            "families": headroom["scope"]["families"],
            "physical_rounds": headroom["scope"]["physical_rounds"],
            "strongest_baseline": headroom["strongest_development_baseline_selection"]["selected"],
            "baseline_p_L": headroom["paired_bootstrap"]["baseline_p_L"],
            "proposed_p_L": headroom["paired_bootstrap"]["proposed_p_L"],
            "relative_improvement": headroom["paired_bootstrap"]["relative_improvement_point"],
            "relative_improvement_lcb": headroom["paired_bootstrap"]["relative_improvement_lcb"],
            "formal_or_pilot_accessed": headroom["scope"]["formal_or_pilot_accessed"],
            "verdict": headroom["verdict"],
            "final_verdict": final_gate["verdict"],
            "learning_decision": final_gate["lane_decisions"]["LEARNED_APPROXIMATION_EXTENSION"]["decision"],
        },
        "tail_safety": {
            "verdict": tail["verdict"],
            "formal_seeds": tail["analysis"]["bootstrap_contract"]["cluster_count"],
            "tail_gate_passes": tail["analysis"]["tail_safety_gate_passes"],
            "calibration": tail["analysis"]["calibration_shift_strict_gate"],
            "nominal": tail["analysis"]["nominal_noninferiority_gate"],
            "fallback_rates": {
                family: action_metrics[family]["fallback_rate"]
                for family in ("nominal_static", "telegraph_drift", "step_calibration_shift")
            },
            "not_admitted": tail["claim_boundary"]["not_admitted"],
        },
        "learning_extension": {
            "candidate_families": eligibility["eligibility_summary"]["candidate_families"],
            "same_task_eligible": eligibility["eligibility_summary"]["same_task_eligible"],
            "teacher_parameters": learning["legacy_learning"]["teacher"]["parameter_count"],
            "student_state_dimension": learning["legacy_learning"]["student"]["state_dimension"],
            "student_scalars": learning["legacy_learning"]["student"]["stored_trainable_scalars"],
            "student_mse": learning["legacy_learning"]["student"]["evaluation_mse"],
            "minimum_retention_point": learning["legacy_learning"]["retention"]["minimum_point"],
            "minimum_retention_lcb": learning["legacy_learning"]["retention"]["minimum_ci_lower"],
            "student_present_in_current_rtl": learning["legacy_learning"]["hardware"]["student_present_in_current_rtl"],
            "same_task": learning["task_signature"]["same_task"],
            "current_disposition": final_gate["lane_decisions"]["LEARNED_APPROXIMATION_EXTENSION"]["decision"],
        },
        "rtl_contribution": {
            "formal_verdict": formal["verdict"],
            "formal_gates": formal["gate_summary"],
            "cover_witnesses": formal["cover_summary"],
            "formal_mutations": formal["mutation_summary"],
            "long_verdict": long_run["verdict"],
            "cycles": long_run["aggregate_python"]["cycles"],
            "ii1_input_pairs": long_run["aggregate_python"]["ii1_input_pairs"],
            "ii1_output_pairs": long_run["aggregate_python"]["ii1_output_pairs"],
            "mismatch_count": sum(row["mismatches"] for row in long_run["cxxrtl_families"]),
            "undefined_actions": long_run["aggregate_python"]["undefined_actions"],
            "silent_overflow": long_run["aggregate_python"]["silent_overflow"],
            "latency_cycles": hardware["clock_model"]["cycles"],
            "initiation_interval_cycles": hardware["clock_model"]["initiation_interval_cycles"],
            "minimum_fmax_mhz": hardware["fmax_mhz"]["minimum"],
            "resource_maximum": {
                key: hardware["resource_summary"][key]["maximum"]
                for key in ("LUT4", "DFF", "BSRAM", "MULT18X18", "MULT9X9")
            },
            "measured_fields": hardware["measured_fields"],
            "evidence_boundary": hardware["evidence_boundary"],
            "student_drives_fast_action": hardware["learning_extension"]["drives_fast_action"],
            "claim_boundary": formal["claim_boundary"],
        },
        "nontransfer_contract": {
            "official_intake_to_exact_reproduction": False,
            "reduced_diagnostic_to_nmf_lifetime": False,
            "project_native_to_official_reproduction": False,
            "per_round_ler_to_physical_lifetime": False,
            "tail_safety_to_lifetime_superiority": False,
            "teacher_student_to_current_rtl": False,
            "rtl_cycles_to_board_latency": False,
            "rtl_safety_to_algorithm_sota": False,
            "cross_lane_global_score": False,
        },
        "manuscript_audit": {
            "required_markers": {
                marker: marker in re.sub(r"\s+", " ", manuscript_text)
                for marker in config["required_manuscript_markers"]
            },
            "forbidden_phrases": {
                phrase: phrase.lower() in manuscript_text.lower()
                for phrase in (
                    "we reproduce Puviani",
                    "we surpass Puviani NMF",
                    "our NMF controller",
                )
            },
            "contract_verdict": manuscript_contract["verdict"],
            "contract_gates": manuscript_contract["gate_summary"],
        },
        "future_program": {
            "status": "PLANNED_NOT_CURRENT_EVIDENCE",
            "official_exact_task": "T9.1.2",
            "official_exact_status": _task_status(board_text, "T9.1.2"),
            "paper_constrained_task": "T9.1.3",
            "paper_constrained_status": _task_status(board_text, "T9.1.3"),
            "requirements": config["future_promotion_requirements"],
            "surpass_puviani_nmf": None,
        },
        "response_package": {
            "strategy": {
                "overall_posture": "acknowledge conceptual overlap, disclose exact-reproduction failure, separate task signatures, state scoped contributions and promotion gates",
                "major_risks": [
                    "official intake renamed exact reproduction",
                    "project-native directional evidence substituted for official NMF",
                    "per-round LER, physical lifetime and RTL cycles mixed",
                    "planned Phase 9 work reported as achieved",
                ],
                "suggested_order": [
                    "direct answer and overlap",
                    "official exact status",
                    "project-native evidence and counterexample",
                    "current negative and scoped positive results",
                    "future matched gate",
                ],
            },
            "tracker": {
                "comment_id": config["reviewer_context"]["comment_id"],
                "concern": config["reviewer_context"]["reviewer_concern"],
                "severity": config["reviewer_context"]["severity"],
                "actions": config["reviewer_context"]["actions"],
                "manuscript_locations": [
                    "Related Work: adaptation and physical lifetime",
                    "Results: independent lane outcomes",
                    "Limitations",
                    "Supplementary comparison boundary",
                ],
                "missing_author_input": config["reviewer_context"]["visible_placeholder"],
            },
            "english_response": _response_text(),
            "manuscript_change_checklist": [
                "State conceptual overlap with Puviani without claiming official reproduction.",
                "Report 0/15 exact qualification and keep all exact lifetime fields null.",
                "Label T2.3.7 project-native and retain the cutoff-16 history-reset counterexample.",
                "Keep Phase-6D 0% strongest-baseline headroom and high fallback costs visible.",
                "Label teacher/student as historical ablation absent from current RTL.",
                "Restrict the positive claim to deterministic atomic fail-closed pre-board RTL.",
                "Keep Phase 9 planned and require a protocol-matched paired-CI gate for any surpass claim.",
            ],
            "missing_information": [config["reviewer_context"]["visible_placeholder"]],
            "package_readiness": config["reviewer_context"]["package_readiness"],
        },
        "response_rows": config["response_rows"],
        "forbidden_response_phrases": config["forbidden_response_phrases"],
        "artifact_registry": artifact_registry,
        "risk_audit": risk_projection,
        "plan_audit": plan_projection,
    }
    report["gates"] = evaluate_gates(report, check_live_sources=True)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["verdict"] = "PENDING_SEMANTIC_MUTATION_AUDIT"
    report["semantic_mutation_audit"] = _run_mutations(report)
    report["verdict"] = (
        VERDICT
        if all(report["gates"].values())
        and report["semantic_mutation_audit"]["detected"]
        == report["semantic_mutation_audit"]["count"]
        else "FAIL_PUVIANI_RELATIONSHIP_CONTRACT"
    )
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    intake = report["official_source_intake"]
    exact = report["official_exact_status"]
    reduced = report["reduced_diagnostic"]
    matched = report["matched_comparison_status"]
    project = report["project_native_directional"]
    signatures = report["task_signatures"]
    current = report["current_phase6d"]
    tail = report["tail_safety"]
    learning = report["learning_extension"]
    rtl = report["rtl_contribution"]
    response = report["response_package"]
    rows = report["response_rows"]
    text = response["english_response"].lower()
    signature_names = (
        "puviani_physical_controller",
        "project_multimode_decoder",
        "project_single_mode_rtl",
    )
    signatures_complete = all(
        set(signatures[name]) == set(signatures["axes"])
        for name in signature_names
    )
    return {
        "G01_identity_and_lifecycle": report["task_id"] == TASK_ID
        and report["schema_version"] == SCHEMA_VERSION
        and report["lifecycle"]["self_done"]
        and report["lifecycle"]["phase9_started"],
        "G02_preemptive_placeholder_honest": report["reviewer_context"]["comment_id"] == "PRQ-NMF-1"
        and response["package_readiness"] == "draft_with_placeholders"
        and response["missing_information"] == ["ACTUAL_REVIEWER_ID_AND_VERBATIM_WORDING"],
        "G03_official_intake_is_pinned_but_not_exact": intake["commit"] == "c9ab1ef2b3ff6fa6d6d24cd95fbd06e2872e016d"
        and intake["tracked_files"] == 12
        and intake["tracked_status_clean"]
        and intake["license"] == "MIT"
        and intake["pristine_syntax_failures"] == 1
        and intake["isolated_patches"] == 4
        and intake["cpu_smoke"] == "PASS_CPU_FULL"
        and intake["gpu_status"] == "UNQUALIFIED_CUSOLVER_FATAL"
        and intake["trained_checkpoints_present"] is False
        and intake["paper_exact_reproduction"] is False
        and intake["claim"] == "INTAKE_ONLY_NOT_PAPER_REPRODUCTION",
        "G04_exact_qualification_is_zero_of_fifteen": exact["status"] == "NO_GO_SOURCE_INCOMPLETE"
        and exact["qualification"]["passed"] == 0
        and exact["qualification"]["failed"] == 15
        and set(exact["qualification"]["gates"].values()) == {False}
        and exact["claim_boundary"]["paper_exact_reproduction"] == "PROHIBITED",
        "G05_all_source_gaps_visible_without_guessing": exact["discrepancies"] == exact["blocking_discrepancies"] == 18
        and len(exact["missing_required_fields"]) == 8
        and exact["guessed_fields"] == [],
        "G06_all_twenty_agent_assets_are_null": exact["agent_rows"] == 20
        and exact["agent_rows_all_null"],
        "G07_all_exact_lifetime_outcomes_remain_null": exact["all_exact_outcomes_null"]
        and _all_exact_outcomes_null(exact["paper_exact_outcomes"])
        and exact["claim_boundary"]["surpass_puviani_nmf"] == "PROHIBITED",
        "G08_reduced_probe_is_diagnostic_only": reduced["scope"] == "REDUCED_STANDARD_PATH_DIAGNOSTIC_NOT_PAPER_REPRODUCTION"
        and reduced["configuration"]["cutoff"] == 8
        and reduced["configuration"]["Delta"] == 0.34
        and len(reduced["configuration"]["logical_states"]) == 6
        and len(reduced["configuration"]["seeds"]) == 3
        and reduced["coverage"] == {"rows": 756, "expected_rows": 756, "trajectories": 36, "environment_steps": 378}
        and reduced["status"] == "PASS_REDUCED_STANDARD_PATH_DIAGNOSTIC"
        and not any((reduced["contains_training"], reduced["contains_mf_or_nmf"], reduced["contains_lifetime_fit"])),
        "G09_matched_comparison_takes_ineligible_branch": matched["failed_prerequisites"] == len(matched["prerequisites"]) == 8
        and matched["execution_branch"] == "INELIGIBLE_NEGATIVE_BRANCH_NO_MATCHED_RUN"
        and matched["comparison_run_manifest"] is None
        and matched["comparison_raw_data"] is None,
        "G10_all_thirteen_matched_metrics_are_null": matched["metric_count"] == 13
        and matched["all_metrics_null"]
        and matched["non_substitution"]["project_T4_4_or_T2_3_7_used_as_official_NMF"] is False
        and matched["claim_boundary"]["surpass_puviani_NMF"] == "PROHIBITED",
        "G11_project_native_directional_values_and_ci_exact": project["status"] == "PASS"
        and abs(project["primary_logical_z_lifetime_cycles"]["standard"] - 2.7476620716328606) < 1e-15
        and abs(project["primary_logical_z_lifetime_cycles"]["mf"] - 6.534670655440108) < 1e-15
        and abs(project["primary_logical_z_lifetime_cycles"]["nmf"] - 6.740784780540096) < 1e-15
        and abs(project["nmf_minus_mf"]["mean_difference"] - 0.2061141250999885) < 1e-15
        and abs(project["nmf_minus_mf"]["ci95_low"] - 0.08416109825708099) < 1e-15
        and abs(project["nmf_minus_mf"]["ci95_high"] - 0.32806715194289604) < 1e-15
        and project["used_as_official_replacement"] is False,
        "G12_cutoff_counterexample_blocks_universal_memory_claim": abs(project["confirmation_logical_z_lifetime_cycles"]["standard"] - 5.144171945638508) < 1e-15
        and abs(project["confirmation_logical_z_lifetime_cycles"]["mf"] - 7.245903084242199) < 1e-15
        and abs(project["confirmation_logical_z_lifetime_cycles"]["nmf"] - 7.708350997751258) < 1e-15
        and abs(project["confirmation_logical_z_lifetime_cycles"]["nmf_latest_only"] - 8.271987493616864) < 1e-15
        and project["confirmation_reset_counterexample"],
        "G13_task_signatures_are_complete_and_nonrankable": signatures["axes"] == [
            "observation",
            "decision_object",
            "action",
            "environment",
            "endpoint",
            "horizon",
            "training_selection_budget",
            "online_compute_budget",
            "evidence_level",
        ]
        and signatures_complete
        and signatures["same_task"] is False
        and signatures["numeric_global_leaderboard_allowed"] is False,
        "G14_current_algorithm_no_go_is_not_hidden": current["families"] == 13
        and current["physical_rounds"] == 79_872
        and current["strongest_baseline"] == "static_mixture_exact_mld"
        and current["baseline_p_L"] == current["proposed_p_L"] == 0.11197916666666667
        and current["relative_improvement"] == current["relative_improvement_lcb"] == 0.0
        and current["formal_or_pilot_accessed"] is False
        and current["verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM"
        and current["final_verdict"] == "GO_RTL_ONLY",
        "G15_tail_safety_is_scoped_not_lifetime_superiority": tail["tail_gate_passes"]
        and tail["calibration"]["passes"]
        and tail["nominal"]["passes"]
        and tail["formal_seeds"] == 24
        and "Puviani NMF lifetime superiority" in tail["not_admitted"],
        "G16_high_fallback_cost_and_tie_are_visible": abs(tail["fallback_rates"]["nominal_static"] - 0.001193576388888889) < 1e-15
        and abs(tail["fallback_rates"]["telegraph_drift"] - 0.5944959852430556) < 1e-15
        and abs(tail["fallback_rates"]["step_calibration_shift"] - 0.9585458260995371) < 1e-15
        and tail["calibration"]["baseline_global_worst_error_count"] == tail["calibration"]["proposed_global_worst_error_count"] == 181,
        "G17_teacher_student_retention_is_exact_and_scoped": learning["teacher_parameters"] == 72_853
        and learning["student_state_dimension"] == 4
        and learning["student_scalars"] == 95
        and abs(learning["student_mse"] - 6.083136156367311e-06) < 1e-18
        and abs(learning["minimum_retention_point"] - 0.9814573586937879) < 1e-15
        and abs(learning["minimum_retention_lcb"] - 0.9445014278749587) < 1e-15,
        "G18_learning_is_ineligible_absent_ablation": learning["candidate_families"] == 16
        and learning["same_task_eligible"] == 0
        and learning["same_task"] is False
        and learning["student_present_in_current_rtl"] is False
        and learning["current_disposition"] == "DROPPED_ABLATION_ONLY",
        "G19_exact_rtl_evidence_is_recomputed": rtl["formal_gates"] == {"passed": 17, "total": 17}
        and rtl["cover_witnesses"] == {"reachable": 14, "total": 14}
        and rtl["formal_mutations"]["killed"] == rtl["formal_mutations"]["total"] == 21
        and rtl["cycles"] == 1_000_000
        and rtl["ii1_input_pairs"] == rtl["ii1_output_pairs"] == 998_435
        and rtl["mismatch_count"] == rtl["undefined_actions"] == rtl["silent_overflow"] == 0
        and rtl["latency_cycles"] == 6
        and rtl["initiation_interval_cycles"] == 1,
        "G20_rtl_and_all_evidence_transfers_fail_closed": rtl["evidence_boundary"]["board_measured"] is False
        and rtl["evidence_boundary"]["multimode_decoder_in_rtl"] is False
        and rtl["evidence_boundary"]["fastest_or_sota"] is False
        and rtl["student_drives_fast_action"] is False
        and all(value is None for value in rtl["measured_fields"].values())
        and set(report["nontransfer_contract"].values()) == {False}
        and len(report["nontransfer_contract"]) == 9,
        "G21_response_and_manuscript_are_direct_without_overclaim": all(
            token in text
            for token in (
                "neither presents an official reproduction",
                "0/15",
                "all thirteen result fields remain null",
                "0% relative improvement",
                "8.271987",
                "one million cxxrtl cycles",
            )
        )
        and not any(phrase.lower() in text for phrase in report["forbidden_response_phrases"])
        and all(report["manuscript_audit"]["required_markers"].values())
        and not any(report["manuscript_audit"]["forbidden_phrases"].values())
        and report["manuscript_audit"]["contract_verdict"] == "PASS_PHASE6D_DUAL_LANE_MANUSCRIPT_DELTA_RTL_ONLY"
        and report["manuscript_audit"]["contract_gates"] == {"passed": 27, "total": 27},
        "G22_rows_are_unique_state_complete_and_lossless": len(rows) == 24
        and len({row["row_id"] for row in rows}) == 24
        and {row["response_state"] for row in rows} == RESPONSE_STATES
        and all(row["claim"] and row["boundary"] and row["source_ids"] for row in rows),
        "G23_all_sources_are_registered_and_live": all(
            set(row["source_ids"]) <= set(report["artifact_registry"])
            for row in rows
        )
        and ((not check_live_sources) or all(_binding_live(binding) for binding in report["artifact_registry"].values())),
        "G24_risk_and_future_program_are_complete": report["risk_audit"] == {
            "schema_version": "t7.3.5-risk-projection-v1",
            "official_asset_risk_present": True,
            "nmf_conflation_risk_present": True,
            "task_audit_present": True,
        }
        and all(report["plan_audit"]["markers"].values())
        and report["future_program"]["status"] == "PLANNED_NOT_CURRENT_EVIDENCE"
        and report["future_program"]["official_exact_status"] == "Blocked"
        and report["future_program"]["paper_constrained_status"]
        in {"Todo", "In Progress", "Done"}
        and report["future_program"]["surpass_puviani_nmf"] is None
        and len(report["future_program"]["requirements"]) == 6
        and len(response["manuscript_change_checklist"]) == 7,
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
        ("G01_identity_and_lifecycle", lambda: mutate(("lifecycle", "self_done"), False)),
        ("G02_preemptive_placeholder_honest", lambda: mutate(("response_package", "package_readiness"), "ready_to_submit")),
        ("G03_official_intake_is_pinned_but_not_exact", lambda: mutate(("official_source_intake", "paper_exact_reproduction"), True)),
        ("G04_exact_qualification_is_zero_of_fifteen", lambda: mutate(("official_exact_status", "qualification", "passed"), 15)),
        ("G05_all_source_gaps_visible_without_guessing", lambda: mutate(("official_exact_status", "missing_required_fields"), [])),
        ("G06_all_twenty_agent_assets_are_null", lambda: mutate(("official_exact_status", "agent_rows_all_null"), False)),
        ("G07_all_exact_lifetime_outcomes_remain_null", lambda: mutate(("official_exact_status", "paper_exact_outcomes", "NMF", "T_ch"), 1500.0)),
        ("G08_reduced_probe_is_diagnostic_only", lambda: mutate(("reduced_diagnostic", "scope"), "PAPER_EXACT_REPRODUCTION")),
        ("G09_matched_comparison_takes_ineligible_branch", lambda: mutate(("matched_comparison_status", "comparison_run_manifest"), {"invented": True})),
        ("G10_all_thirteen_matched_metrics_are_null", lambda: mutate(("matched_comparison_status", "all_metrics_null"), False)),
        ("G11_project_native_directional_values_and_ci_exact", lambda: mutate(("project_native_directional", "used_as_official_replacement"), True)),
        ("G12_cutoff_counterexample_blocks_universal_memory_claim", lambda: mutate(("project_native_directional", "confirmation_logical_z_lifetime_cycles", "nmf_latest_only"), 7.0)),
        ("G13_task_signatures_are_complete_and_nonrankable", lambda: mutate(("task_signatures", "same_task"), True)),
        ("G14_current_algorithm_no_go_is_not_hidden", lambda: mutate(("current_phase6d", "relative_improvement"), 0.15)),
        ("G15_tail_safety_is_scoped_not_lifetime_superiority", lambda: mutate(("tail_safety", "not_admitted"), [])),
        ("G16_high_fallback_cost_and_tie_are_visible", lambda: mutate(("tail_safety", "fallback_rates", "step_calibration_shift"), 0.01)),
        ("G17_teacher_student_retention_is_exact_and_scoped", lambda: mutate(("learning_extension", "minimum_retention_lcb"), 1.0)),
        ("G18_learning_is_ineligible_absent_ablation", lambda: mutate(("learning_extension", "student_present_in_current_rtl"), True)),
        ("G19_exact_rtl_evidence_is_recomputed", lambda: mutate(("rtl_contribution", "mismatch_count"), 1)),
        ("G20_rtl_and_all_evidence_transfers_fail_closed", lambda: mutate(("rtl_contribution", "evidence_boundary", "board_measured"), True)),
        ("G21_response_and_manuscript_are_direct_without_overclaim", lambda: mutate(("response_package", "english_response"), report["response_package"]["english_response"] + " We surpass Puviani NMF.")),
        ("G22_rows_are_unique_state_complete_and_lossless", lambda: {**copy.deepcopy(report), "response_rows": list(report["response_rows"][:-1])}),
        ("G23_all_sources_are_registered_and_live", lambda: mutate(("artifact_registry", "exact_reproduction", "sha256"), "0" * 64)),
        ("G24_risk_and_future_program_are_complete", lambda: mutate(("risk_audit", "nmf_conflation_risk_present"), False)),
    ]
    results = []
    for target_gate, factory in cases:
        mutated = factory()
        detected = not evaluate_gates(
            mutated,
            check_live_sources=target_gate == "G23_all_sources_are_registered_and_live",
        )[target_gate]
        results.append(
            {
                "mutation_id": f"M{len(results) + 1:02d}",
                "target_gate": target_gate,
                "detected": detected,
            }
        )
    return {
        "count": len(results),
        "detected": sum(case["detected"] for case in results),
        "cases": results,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    exact = report["official_exact_status"]
    matched = report["matched_comparison_status"]
    project = report["project_native_directional"]
    rtl = report["rtl_contribution"]
    lines = [
        "# Reviewer response: relationship to Puviani non-Markovian feedback",
        "",
        f"- Task: `{report['task_id']}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Package readiness: `{report['response_package']['package_readiness']}`",
        f"- Gates/mutations: `{report['gate_summary']['passed']}/{report['gate_summary']['total']}` / `{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`",
        "",
        "## Point-by-point response",
        "",
        report["response_package"]["english_response"],
        "",
        "## Evidence taxonomy",
        "",
        "| Lane | Status | Numeric boundary |",
        "| --- | --- | --- |",
        f"| Official GQF paper-exact | `{exact['status']}` | `{exact['qualification']['passed']}/{exact['qualification']['passed'] + exact['qualification']['failed']}` gates; all exact lifetime fields null |",
        f"| Official-code reduced diagnostic | `{report['reduced_diagnostic']['scope']}` | 756 rows; no MF/NMF or lifetime fit |",
        f"| Same-GQF matched comparison | `{matched['execution_branch']}` | {matched['metric_count']}/{matched['metric_count']} metrics null |",
        f"| Project-native directional study | `{project['status']}` | NMF-MF={project['nmf_minus_mf']['mean_difference']:.6f}, 95% CI [{project['nmf_minus_mf']['ci95_low']:.6f}, {project['nmf_minus_mf']['ci95_high']:.6f}] |",
        f"| Current multimode algorithm | `{report['current_phase6d']['verdict']}` | 0% improvement over strongest static exact MLD |",
        f"| Current exact RTL | `{report['current_phase6d']['final_verdict']}` | {rtl['latency_cycles']} cycles, II={rtl['initiation_interval_cycles']}, {rtl['cycles']:,} cycles, zero mismatch; pre-board only |",
        "",
        "## Manuscript checklist",
        "",
    ]
    lines.extend(f"- {item}" for item in report["response_package"]["manuscript_change_checklist"])
    lines.extend(
        [
            "",
            "## Missing author input",
            "",
            f"- `{report['response_package']['missing_information'][0]}`",
            "",
            "## 中文核对",
            "",
            "共同使用历史信息不构成同任务复现。官方 exact 为 0/15，所有 Standard/MF/NMF lifetime 字段与 matched comparison 字段均为 null；project-native 十周期方向性结果不能填补官方资产。当前算法结论为 strongest-static 0% NO-GO，正贡献仅限 exact single-mode 六周期、II=1、atomic/fail-closed 的预板数字系统。Phase 9 是未来程序，不是既成结果。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def write_outputs(report: dict[str, Any]) -> None:
    rows = _source_rows(report)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{DEFAULT_SOURCE_DATA.name}.",
        suffix=".tmp",
        dir=DEFAULT_SOURCE_DATA.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, DEFAULT_SOURCE_DATA)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    report["source_data"] = {**_binding(DEFAULT_SOURCE_DATA), "rows": len(rows)}
    _atomic_write_text(DEFAULT_MARKDOWN, _markdown(report))
    report["markdown"] = _binding(DEFAULT_MARKDOWN)
    _atomic_write_text(DEFAULT_REPORT, json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def verify_report() -> tuple[bool, dict[str, bool]]:
    if not DEFAULT_REPORT.is_file():
        return False, {"outputs_exist": False}
    stored = _load(DEFAULT_REPORT)
    fresh = build_report(generated_at_utc=stored.get("generated_at_utc"))
    checks = {
        "outputs_exist": DEFAULT_SOURCE_DATA.is_file() and DEFAULT_MARKDOWN.is_file(),
        "identity": stored.get("task_id") == TASK_ID and stored.get("schema_version") == SCHEMA_VERSION,
        "verdict": stored.get("verdict") == VERDICT and fresh.get("verdict") == VERDICT,
        "all_gates": all(evaluate_gates(stored, check_live_sources=True).values()),
        "all_mutations": stored["semantic_mutation_audit"]["count"]
        == stored["semantic_mutation_audit"]["detected"]
        == len(stored["gates"]),
        "source_data": _source_data_matches(stored),
        "markdown_live": _binding_live(stored["markdown"]),
        "analysis_live": stored.get("analysis_sha256")
        == _canonical_sha256(_analysis_payload(stored)),
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
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "gates": report["gate_summary"],
                "mutations": {
                    "detected": report["semantic_mutation_audit"]["detected"],
                    "total": report["semantic_mutation_audit"]["count"],
                },
                "source_rows": len(report["response_rows"]),
                "package_readiness": report["response_package"]["package_readiness"],
                "analysis_sha256": report["analysis_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
