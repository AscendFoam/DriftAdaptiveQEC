"""T7.2.5 evidence contract for the manuscript Supplementary material.

The contract freezes definitions, parameters, comparators, statistical rules,
negative results, long-sequence RTL evidence, and the Phase-6C source locator.
It rejects attempts to turn N/A or null cells into zero, mix task signatures,
promote pre-board estimates to measurements, or use a secondary result to
rescue the stopped V5 branch.
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
TASK_ID = "T7.2.5"
SCHEMA_VERSION = "t7.2.5-supplementary-evidence-contract-v1"
VERDICT = "PASS_SUPPLEMENT_COMPLETE_REPRODUCIBLE_AND_NONMIXING"

NOTE_PATH = ROOT / "docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex"
DEFAULT_REPORT = ROOT / "docs/t7_2_5_supplementary_evidence_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_2_5_supplementary_evidence_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/supplementary_evidence_contract.md"

SOURCE_PATHS = {
    "manuscript": NOTE_PATH,
    "claim_matrix": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix.json",
    "introduction_contract": ROOT / "docs/t7_2_1_introduction_related_work_contract.json",
    "methods_contract": ROOT / "docs/t7_2_2_methods_evidence_contract.json",
    "results_contract": ROOT / "docs/t7_2_3_results_evidence_contract.json",
    "discussion_contract": ROOT / "docs/t7_2_4_discussion_conclusion_contract.json",
    "supplement_figure_contract": ROOT / "docs/t7_1_4_supplement_figure_contract.json",
    "execution_contract": ROOT / "docs/t6_5_2_unified_execution_contract.json",
    "formal_preregistration": ROOT / "docs/t6_5_3_route_a_preregistration.json",
    "posterior_lock": ROOT / "docs/t6_6_3_route_a_posterior_threshold_lock.json",
    "smooth_formal": ROOT / "docs/t6_7_1_smooth_formal_matrix.json",
    "tail_formal": ROOT / "docs/t6_7_2_abrupt_ood_tail_formal_matrix.json",
    "long_rtl": ROOT / "docs/t6_2_2_long_rtl_qualification.json",
    "integrated_rtl": ROOT / "docs/t6_7_3_route_a_integrated_rtl_qualification.json",
    "preboard_profiles": ROOT / "docs/t6_19_1_project_preboard_profiles.json",
    "board_blocker": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
    "v5_final_gate": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json",
    "ontology": ROOT / "docs/t6_16_2_comparison_ontology.json",
    "secondary_preregistration": ROOT / "docs/t6_16_3_secondary_preregistration.json",
    "single_cpd": ROOT / "docs/t6_17_1_single_mode_cpd_equivalence.json",
    "surface_cnot": ROOT / "docs/t6_17_2_noh_cnot_ci_ml_reproduction.json",
    "learned_eligibility": ROOT / "docs/t6_17_3_learned_model_eligibility_replay.json",
    "gqf_exact": ROOT / "docs/t6_8_4_gqf_paper_exact_reproduction.json",
    "aqec": ROOT / "docs/t6_18_1_aqec_common_wallclock_replay.json",
    "structured_cpd": ROOT / "docs/t6_18_2_official_structured_cpd_reproduction.json",
    "multimode_cpd": ROOT / "docs/t6_18_3_multimode_posterior_weighted_cpd.json",
    "external_fpga": ROOT / "docs/t6_19_2_external_fpga_normalization.json",
    "phase6c_integrity": ROOT / "docs/t6_19_3_secondary_evidence_integrity.json",
    "task_board": ROOT / "docs/new_task_board.md",
    "implementation": Path(__file__).resolve(),
}

SUPPLEMENT_STATES = (
    "DEFINITION",
    "FROZEN_PARAMETER",
    "COMPARATOR",
    "STATISTICAL_RULE",
    "NEGATIVE_OR_FAILURE",
    "RTL_REPRODUCTION",
    "PHASE6C_LOCATOR",
    "NON_MIXING_RULE",
)

REQUIRED_APPENDIX_SECTIONS = (
    "Evidence-bounded supplementary figure suite",
    "Mathematical and decision definitions",
    "Frozen parameters and execution contract",
    "Complete comparators and statistical protocol",
    "Ablations, failure modes, and negative selection",
    "RTL, toolchain, long-sequence, and reproduction procedure",
    "Phase 6C source locator and non-mixing ledger",
    "Primary repository evidence map",
    "Claim wording guardrails",
)

REQUIRED_VALUE_STATES = {
    "MEASURED_VALUE",
    "ESTIMATE_VALUE",
    "REPRODUCED_VALUE",
    "LITERATURE_VALUE",
    "NULL_NOT_REPORTED",
    "N_A_NOT_APPLICABLE",
    "FAILED",
    "NEGATIVE",
}

PROHIBITED_ASSERTIVE_PATTERNS = (
    "n/a values are treated as zero",
    "null values are treated as zero",
    "phase 6c establishes a global winner",
    "phase 6c promotes v5",
    "we outperform puviani nmf",
    "we are faster than existing fpga qec decoders",
    "board-measured latency is 222.222",
    "one million real traces",
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


def _all_tokens(text: str, tokens: Sequence[str]) -> bool:
    return all(token.lower() in text for token in tokens)


def _prior_contract_live(contract: Mapping[str, Any]) -> bool:
    return (
        all(bool(value) for value in contract.get("gates", {}).values())
        and all(_binding_live(binding) for binding in contract.get("source_bindings", {}).values())
    )


def _board_task_status(board_text: str, task_id: str) -> str | None:
    match = re.search(rf"^\|\s*{re.escape(task_id)}\s*\|\s*([^|]+?)\s*\|", board_text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _supplement_rows() -> list[dict[str, str]]:
    raw_rows = [
        ("SUP-001", "folded_syndrome", "DEFINITION", "Methods/T6.17.1", "Half-open square-GKP residue with production tie rule"),
        ("SUP-002", "logical_coset_map", "DEFINITION", "T2/T6.17.1", "Periodic likelihood sum; MAP is not renamed CPD"),
        ("SUP-003", "joint_correlated_map", "DEFINITION", "T6.6.1", "Two-dimensional software comparator differs from phase LUT"),
        ("SUP-004", "pauli_ler", "DEFINITION", "T6.5.3", "p_L equals p_X+p_Y+p_Z; raw denominators retained"),
        ("SUP-005", "window_tail", "DEFINITION", "T6.5.3/T6.7.2", "512-decision p95, worst and CVaR remain finite-horizon metrics"),
        ("SUP-006", "paired_orientation", "DEFINITION", "T6.5.3", "Positive baseline-minus-candidate difference favours candidate"),
        ("SUP-007", "gap_closure", "DEFINITION", "T6.5.3", "Negative static-to-oracle gap closure is not clipped"),
        ("SUP-008", "observed_truth_split", "FROZEN_PARAMETER", "T6.5.2", "Truth is evaluator/oracle only"),
        ("SUP-009", "quantization", "FROZEN_PARAMETER", "T6.5.2", "10-bit ADC, 8-bit address, 2-bit fraction"),
        ("SUP-010", "map_word", "FROZEN_PARAMETER", "T6.5.2/T6.7.3", "Signed Q9.12, ties-to-even, saturation"),
        ("SUP-011", "fast_path", "FROZEN_PARAMETER", "T6.5.2/T6.7.3", "Six cycles and II=1 before physical transport"),
        ("SUP-012", "cadence", "FROZEN_PARAMETER", "T6.5.2/T6.6.2", "32-decision posterior and 4,000-cycle image opportunity"),
        ("SUP-013", "compute_budget", "FROZEN_PARAMETER", "T6.5.2", "8,192 MAC/B/B and 5,000-us update ceilings"),
        ("SUP-014", "hmm_contract", "FROZEN_PARAMETER", "T6.6.2", "Four causal states with 0.1/4.0/2.0 calibration parameters"),
        ("SUP-015", "policy_tuple", "FROZEN_PARAMETER", "T6.6.3", "One tuple selected from 1,728 pilot candidates"),
        ("SUP-016", "bank_transaction", "FROZEN_PARAMETER", "T6.5.2/T6.7.3", "CRC/SHA/CAS A/B bank, six-cycle drain and LKG"),
        ("SUP-017", "formal_split", "FROZEN_PARAMETER", "T6.5.3", "12 calibration, 12 pilot and 24 formal clusters"),
        ("SUP-018", "standard_binning", "COMPARATOR", "T6.6.1", "Weak common-grid reference"),
        ("SUP-019", "static_joint_map", "COMPARATOR", "T6.6.1", "Strong software comparator, not current full joint-MAP RTL"),
        ("SUP-020", "window_map", "COMPARATOR", "T6.6.1/T6.7.1", "Strongest smooth deployable comparator"),
        ("SUP-021", "ewma_map", "COMPARATOR", "T6.6.1/T6.5.3", "Pilot-locked primary comparator"),
        ("SUP-022", "kalman_map", "COMPARATOR", "T6.6.1", "Secondary matched deployable comparator"),
        ("SUP-023", "route_a_v4", "COMPARATOR", "T6.6.2", "Safety router over Window/EWMA shadows"),
        ("SUP-024", "oracle", "COMPARATOR", "T6.6.1", "Truth-privileged non-ranking upper bound"),
        ("SUP-025", "paired_bootstrap", "STATISTICAL_RULE", "T6.5.3/T6.16.3", "20,000 whole-cluster resamples at 95% confidence"),
        ("SUP-026", "multiplicity", "STATISTICAL_RULE", "T6.5.3/T6.16.3", "Holm only within a preregistered endpoint family"),
        ("SUP-027", "failed_policy_families", "NEGATIVE_OR_FAILURE", "T6.6.3", "Static-switch and freeze-all each fail all 38 tuples"),
        ("SUP-028", "static_window_counterevidence", "NEGATIVE_OR_FAILURE", "T6.7.1/T6.9.3", "V4 is not the best deployable decoder"),
        ("SUP-029", "calibration_tail", "NEGATIVE_OR_FAILURE", "T6.7.2", "181/512 versus static 32/512 worst window"),
        ("SUP-030", "bocd_budget", "NEGATIVE_OR_FAILURE", "T6.8.3", "13,004.1-us worst update exceeds 5,000-us cap"),
        ("SUP-031", "v5_entry_stop", "NEGATIVE_OR_FAILURE", "T6.10.1/T6.15.5", "Causal/action headroom fails before implementation"),
        ("SUP-032", "physical_board", "NEGATIVE_OR_FAILURE", "T6.9.2", "All 42 measured fields remain null"),
        ("SUP-033", "generic_long_rtl", "RTL_REPRODUCTION", "T6.2.2", "One million cycles, ten fault families and all 61 attempts"),
        ("SUP-034", "integrated_long_rtl", "RTL_REPRODUCTION", "T6.7.3", "995,802 replay plus 4,198 directed cycles; all 75/25 attempts"),
        ("SUP-035", "toolchain", "RTL_REPRODUCTION", "T6.2.2/T6.7.3", "Trace/model/executable/log hashes and exact tool versions"),
        ("SUP-036", "static_preboard_profile", "RTL_REPRODUCTION", "T6.19.1", "Only static MAP-LUT has eligible current RTL/P&R estimate"),
        ("SUP-037", "single_ci_cpd", "PHASE6C_LOCATOR", "T6.17.1", "Project-native square/isotropic equivalence only"),
        ("SUP-038", "two_gkp_cnot", "PHASE6C_LOCATOR", "T6.17.2", "Project-native matched gate failure, not memory LER"),
        ("SUP-039", "structured_cpd", "PHASE6C_LOCATOR", "T6.18.2", "Official-code reproduction with small-distance caveat"),
        ("SUP-040", "multimode_cpd", "PHASE6C_LOCATOR", "T6.18.3", "Independent project-native d=3 drift result"),
        ("SUP-041", "learned_eligibility", "PHASE6C_LOCATOR", "T6.17.3", "Zero of 16 same-task eligible"),
        ("SUP-042", "gqf_nmf", "PHASE6C_LOCATOR", "T6.8.4/T6.8.5", "Blocked exact reproduction; 13 metrics null"),
        ("SUP-043", "aqec", "PHASE6C_LOCATOR", "T6.18.1", "Negative project replay; official protocol blocked"),
        ("SUP-044", "external_fpga", "PHASE6C_LOCATOR", "T6.19.2", "18 literature rows and zero exact same-task comparator"),
        ("SUP-045", "value_state_semantics", "NON_MIXING_RULE", "T6.16.2", "N/A, null, failed and negative remain distinct"),
        ("SUP-046", "atlas_nonranking", "NON_MIXING_RULE", "T6.19.3", "206 cells, no global score/winner or V5 rescue"),
    ]
    return [
        {"row_id": row_id, "topic": topic, "supplement_state": state, "source_ids": sources, "boundary": boundary}
        for row_id, topic, state, sources, boundary in raw_rows
    ]


def _manuscript_snapshot() -> dict[str, Any]:
    tex = NOTE_PATH.read_text(encoding="utf-8")
    appendix = tex.split("\\appendix", 1)[1]
    normalized = _normalize(appendix)
    sections = re.findall(r"\\section\{([^}]+)\}", appendix)
    positions = {title: sections.index(title) for title in REQUIRED_APPENDIX_SECTIONS if title in sections}
    checks = {
        "required_sections": all(title in sections for title in REQUIRED_APPENDIX_SECTIONS),
        "ordered_sections": positions == {title: index for index, title in enumerate(REQUIRED_APPENDIX_SECTIONS)},
        "map_definitions": _all_tokens(normalized, (
            "r_l(y)", "logical-coset likelihood", "lambda_{b_q,b_p}",
            "p_x+p_y+p_z", "operatorname{cvar}_{95}", "delta_b", "g_{\\rm static}",
        )),
        "metric_boundaries": _all_tokens(normalized, (
            "finite-horizon simulation summaries", "not decay constants or physical lifetimes",
            "positive $\\delta_b$ favours", "negative value means", "truth is used to compute",
        )),
        "frozen_parameter_registry": _all_tokens(normalized, (
            "10-bit adc", "signed q9.12", "five-cycle map plus one event/action cycle",
            "2,048 scalar observations every 4,000 cycles", "8,192 mac",
            "$(0.9,0.2,0.25,192,2,8)$", "1,728 common threshold tuples",
            "12 calibration, 12 pilot, 24 formal", "71,958,528 decisions",
        )),
        "complete_comparators": _all_tokens(normalized, (
            "standard binning", "static joint map", "window map", "ewma adaptive map",
            "kalman adaptive map", "v4 \\routea", "hidden-state oracle", "legacy cnn residual",
            "bocd wrapper", "v5 candidate",
        )),
        "statistical_protocol": _all_tokens(normalized, (
            "28,311,552 scored decisions", "43,646,976 scored decisions",
            "20,000 paired resamples", "paired bootstrap resamples whole seed clusters",
            "holm correction", "non-inferiority margins", "zero-event",
        )),
        "negative_ledger": _all_tokens(normalized, (
            "0/38 pilot-safe tuples", "-0.037109375", "-0.044921875",
            "181/512", "32/512", "13,004.1", "-0.2322", "0.02549",
            "42/42 measured fields remain null",
        )),
        "long_rtl_reproduction": _all_tokens(normalized, (
            "972,386 valid outputs", "all 61 commit attempts", "995,802",
            "4,198", "75 host commit attempts", "25 rollback attempts", "0.4198",
            "yosys 0.67", "g++ 15.1.0", "abstract bounded disturbance model",
        )),
        "preboard_boundary": _all_tokens(normalized, (
            "4,316 accepted equivalence rows", "six cycles, ii=1", "41.024--42.212 mhz",
            "open-tool post-route estimates", "board-only fields null",
        )),
        "phase6c_task_signature": _all_tokens(normalized, (
            "task signature has 13 fields", "code family, modes/distance, decision target",
            "online privilege", "compute budget", "evidence level",
        )),
        "phase6c_locator": _all_tokens(normalized, (
            "1,048,576", "3,080,192 trials", "2,005/2,005", "9.6m cycles",
            "0/16", "0/15", "six cells$\\times$24 clusters", "18 implementations",
            "24/24 gates and mutations",
        )),
        "value_states": _all_tokens(normalized, (
            "measured\\_value", "estimate\\_value", "reproduced\\_value",
            "literature\\_value", "n\\_a\\_not\\_applicable", "null\\_not\\_reported",
            "failed", "negative", "blocked", "ineligible",
        )),
        "nonmixing_boundary": _all_tokens(normalized, (
            "none of these states may be replaced by zero", "206 cells and no global winner",
            "no_go_v5_early_headroom_stop", "42 null measured fields",
        )),
        "prohibited_assertions_absent": not any(pattern in normalized for pattern in PROHIBITED_ASSERTIVE_PATTERNS),
    }
    return {
        "sections": sections,
        "characters": len(appendix),
        "sha256": hashlib.sha256(appendix.encode("utf-8")).hexdigest(),
        "checks": checks,
        "prohibited_hits": [pattern for pattern in PROHIBITED_ASSERTIVE_PATTERNS if pattern in normalized],
    }


def _parent_state() -> dict[str, Any]:
    claim = _load_json(SOURCE_PATHS["claim_matrix"])
    intro = _load_json(SOURCE_PATHS["introduction_contract"])
    methods = _load_json(SOURCE_PATHS["methods_contract"])
    results = _load_json(SOURCE_PATHS["results_contract"])
    discussion = _load_json(SOURCE_PATHS["discussion_contract"])
    figures = _load_json(SOURCE_PATHS["supplement_figure_contract"])
    prereg = _load_json(SOURCE_PATHS["secondary_preregistration"])
    long_rtl = _load_json(SOURCE_PATHS["long_rtl"])
    integrated = _load_json(SOURCE_PATHS["integrated_rtl"])
    preboard = _load_json(SOURCE_PATHS["preboard_profiles"])
    static_profile = next(row for row in preboard["hardware_profiles"] if row["method_id"] == "static_map_lut_if_rtl")
    board = _load_json(SOURCE_PATHS["board_blocker"])
    v5 = _load_json(SOURCE_PATHS["v5_final_gate"])
    ontology = _load_json(SOURCE_PATHS["ontology"])
    single = _load_json(SOURCE_PATHS["single_cpd"])
    cnot = _load_json(SOURCE_PATHS["surface_cnot"])
    learned = _load_json(SOURCE_PATHS["learned_eligibility"])
    gqf = _load_json(SOURCE_PATHS["gqf_exact"])
    aqec = _load_json(SOURCE_PATHS["aqec"])
    structured = _load_json(SOURCE_PATHS["structured_cpd"])
    multimode = _load_json(SOURCE_PATHS["multimode_cpd"])
    external = _load_json(SOURCE_PATHS["external_fpga"])
    atlas = _load_json(SOURCE_PATHS["phase6c_integrity"])
    board_text = SOURCE_PATHS["task_board"].read_text(encoding="utf-8")
    pr_values = [row["achieved_fmax_mhz"] for row in static_profile["place_route"]]
    pr_luts = [row["lut4_count"] for row in static_profile["place_route"]]
    aggregate_multimode = multimode["summaries"]["aggregate"]
    return {
        "verdicts": {
            "claim": claim["verdict"],
            "introduction": intro["verdict"],
            "methods": methods["verdict"],
            "results": results["verdict"],
            "discussion": discussion["verdict"],
            "figures": figures["verdict"],
            "long_rtl": long_rtl["verdict"],
            "integrated_rtl": integrated["verdict"],
            "preboard": preboard["verdict"],
            "board": board["verdict"],
            "v5": v5["verdict"],
            "single": single["verdict"],
            "cnot": cnot["verdict"],
            "learned": learned["verdict"],
            "gqf": gqf["verdict"],
            "aqec": aqec["verdict"],
            "structured": structured["verdict"],
            "multimode": multimode["verdict"],
            "external": external["verdict"],
            "atlas": atlas["verdict"],
        },
        "previous_contracts_live": {
            "introduction": _prior_contract_live(intro),
            "methods": _prior_contract_live(methods),
            "results": _prior_contract_live(results),
            "discussion": _prior_contract_live(discussion),
        },
        "figure_records": len(figures["records"]),
        "secondary_statistics": prereg["statistics"],
        "ontology": {
            "signature_fields": ontology["ontology"]["task_signature_fields"],
            "value_states": sorted(ontology["ontology"]["value_states"]),
        },
        "long_rtl": {
            "families": len(long_rtl["family_names"]),
            "cycles": long_rtl["aggregate_python"]["cycles"],
            "valid": long_rtl["aggregate_python"]["output_valid"],
            "commit_attempts": long_rtl["aggregate_python"]["commit_attempts"],
            "undefined": long_rtl["aggregate_python"]["undefined_actions"],
            "silent_overflow": long_rtl["aggregate_python"]["silent_overflow"],
            "cxx_mismatches": sum(row["mismatches"] for row in long_rtl["cxxrtl_families"]),
        },
        "integrated_rtl": {
            "families": len(integrated["family_names"]),
            "cycles": integrated["aggregate_python"]["cycles"],
            "replay": integrated["aggregate_python"]["unified_replay_cycles"],
            "directed": integrated["aggregate_python"]["directed_boundary_cycles"],
            "host_attempts": integrated["aggregate_python"]["host_commit_attempts"],
            "rollback_attempts": integrated["aggregate_python"]["rollback_attempts"],
            "undefined": integrated["aggregate_python"]["undefined_actions"],
            "silent_overflow": integrated["aggregate_python"]["silent_overflow"],
            "cxx_mismatches": sum(row["mismatches"] for row in integrated["cxxrtl_families"]),
        },
        "preboard": {
            "eligible_profiles": sum(row["ranking_eligible_project_preboard"] for row in preboard["hardware_profiles"]),
            "equivalence_rows": static_profile["equivalence_map_valid_rows"],
            "cycles": static_profile["core_cycles"],
            "ii": static_profile["initiation_interval_cycles"],
            "pr_seeds": len(static_profile["place_route"]),
            "fmax_min": min(pr_values),
            "fmax_max": max(pr_values),
            "lut_min": min(pr_luts),
            "lut_max": max(pr_luts),
        },
        "board": {
            "fields": len(board["measured_results"]),
            "nonnull": sum(value is not None for value in board["measured_results"].values()),
        },
        "v5": {
            "dropped_tasks": len(v5["dropped_tasks"]),
            "downstream_outputs": len(v5["v5_downstream_outputs_found"]),
        },
        "phase6c": {
            "single_domain": single["production_domain"]["points"],
            "single_boundary": single["boundary_audit"]["points"],
            "single_mismatches": single["production_domain"]["cpd_ci_mismatches"] + single["boundary_audit"]["cpd_ci_mismatches"],
            "cnot_trials": sum(row["trials"] for row in cnot["points"]),
            "cnot_reductions": [row["relative_failure_reduction"] for row in cnot["points"]],
            "learned_candidates": learned["eligibility_summary"]["candidate_families"],
            "learned_eligible": learned["eligibility_summary"]["same_task_eligible"],
            "gqf_passed": gqf["exact_qualification"]["passed"],
            "gqf_failed": gqf["exact_qualification"]["failed"],
            "aqec_cells": aqec["project_result"]["cells"],
            "aqec_clusters": aqec["project_result"]["seed_clusters"],
            "aqec_official": aqec["official_protocol_reproduction"]["state"],
            "structured_upstream": structured["upstream_tests"]["deterministic_replay"]["passed"],
            "structured_cells": structured["independent_experiment"]["paired_advantage"]["total_cells"],
            "structured_cpd_wins": structured["independent_experiment"]["paired_advantage"]["cpd_lower_ler_cells"],
            "multimode_cycles": multimode["formal_counts"]["total_physical_cycles"],
            "multimode_decodes": multimode["formal_counts"]["total_comparator_decodes"],
            "multimode_static": aggregate_multimode["static_euclidean"]["p_L"],
            "multimode_adaptive": aggregate_multimode["observed_only_posterior_predictive_weighted"]["p_L"],
            "external_rows": external["comparison_eligibility"]["external_rows"],
            "external_same_task": external["comparison_eligibility"]["same_task_external_comparator_count"],
            "atlas_cells": len(atlas["cells"]),
            "global_score": atlas["ranking_policy"]["global_score"],
            "global_winner": atlas["ranking_policy"]["global_winner"],
        },
        "task_status": _board_task_status(board_text, TASK_ID),
    }


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    checks = report["manuscript"]["checks"]
    parent = report["parent_state"]
    rows = report["supplement_rows"]
    source_live = all(_binding_live(binding) for binding in report["source_bindings"].values()) if check_live_sources else bool(report["source_integrity_declared"])
    phase = parent["phase6c"]
    gates = {
        "G01_identity": report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION,
        "G02_live_source_bindings": source_live,
        "G03_appendix_sections_complete_and_ordered": bool(checks["required_sections"] and checks["ordered_sections"]),
        "G04_map_and_metric_definitions": bool(checks["map_definitions"] and checks["metric_boundaries"]),
        "G05_frozen_parameter_registry": bool(checks["frozen_parameter_registry"]),
        "G06_complete_comparator_registry": bool(checks["complete_comparators"]),
        "G07_statistical_protocol_complete": bool(checks["statistical_protocol"]),
        "G08_negative_and_failure_ledger": bool(checks["negative_ledger"]),
        "G09_long_rtl_reproduction": bool(checks["long_rtl_reproduction"]),
        "G10_preboard_measurement_boundary": bool(checks["preboard_boundary"]),
        "G11_phase6c_task_signature_and_locator": bool(checks["phase6c_task_signature"] and checks["phase6c_locator"]),
        "G12_value_states_and_nonmixing": bool(checks["value_states"] and checks["nonmixing_boundary"]),
        "G13_previous_prose_contracts_live": all(parent["previous_contracts_live"].values()),
        "G14_supplement_figure_parent_complete": parent["figure_records"] == 792,
        "G15_ontology_exact": len(parent["ontology"]["signature_fields"]) == 13 and set(parent["ontology"]["value_states"]) == REQUIRED_VALUE_STATES,
        "G16_long_rtl_parent_exact": parent["long_rtl"] == {"families": 10, "cycles": 1_000_000, "valid": 972_386, "commit_attempts": 61, "undefined": 0, "silent_overflow": 0, "cxx_mismatches": 0},
        "G17_integrated_rtl_parent_exact": parent["integrated_rtl"] == {"families": 10, "cycles": 1_000_000, "replay": 995_802, "directed": 4_198, "host_attempts": 75, "rollback_attempts": 25, "undefined": 0, "silent_overflow": 0, "cxx_mismatches": 0},
        "G18_preboard_parent_and_board_null": parent["preboard"]["eligible_profiles"] == 1 and parent["preboard"]["equivalence_rows"] == 4_316 and parent["preboard"]["cycles"] == 6 and parent["preboard"]["ii"] == 1 and parent["preboard"]["pr_seeds"] == 3 and parent["board"] == {"fields": 42, "nonnull": 0},
        "G19_phase6c_correctness_and_reproduction": phase["single_domain"] == 1_048_576 and phase["single_boundary"] == 1_000_000 and phase["single_mismatches"] == 0 and phase["cnot_trials"] == 3_080_192 and phase["structured_upstream"] == 2_005 and phase["structured_cells"] == phase["structured_cpd_wins"] == 27,
        "G20_phase6c_positive_is_lane_local": phase["multimode_cycles"] == 9_600_000 and phase["multimode_decodes"] == 38_400_000 and phase["multimode_adaptive"] < phase["multimode_static"] and phase["global_score"] is False and phase["global_winner"] is None,
        "G21_phase6c_absence_and_negative_states": phase["learned_candidates"] == 16 and phase["learned_eligible"] == 0 and parent["v5"] == {"dropped_tasks": 20, "downstream_outputs": 0} and phase["aqec_official"] == "BLOCKED_OFFICIAL_PROTOCOL_REPRODUCTION" and phase["external_rows"] == 18 and phase["external_same_task"] == 0,
        "G22_rows_complete_unique_and_stateful": len(rows) == 46 and len({row["row_id"] for row in rows}) == 46 and {row["supplement_state"] for row in rows} == set(SUPPLEMENT_STATES) and all(row["source_ids"] and row["boundary"] for row in rows),
        "G23_task_status": parent["task_status"] in {"In Progress", "Done"},
        "G24_prohibited_assertions_absent": bool(checks["prohibited_assertions_absent"]),
    }
    return gates


def _semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    check_map = {
        "G03_appendix_sections_complete_and_ordered": "ordered_sections",
        "G04_map_and_metric_definitions": "map_definitions",
        "G05_frozen_parameter_registry": "frozen_parameter_registry",
        "G06_complete_comparator_registry": "complete_comparators",
        "G07_statistical_protocol_complete": "statistical_protocol",
        "G08_negative_and_failure_ledger": "negative_ledger",
        "G09_long_rtl_reproduction": "long_rtl_reproduction",
        "G10_preboard_measurement_boundary": "preboard_boundary",
        "G11_phase6c_task_signature_and_locator": "phase6c_locator",
        "G12_value_states_and_nonmixing": "value_states",
        "G24_prohibited_assertions_absent": "prohibited_assertions_absent",
    }
    for index, target in enumerate(evaluate_gates(report)):
        mutated = copy.deepcopy(report)
        if target == "G01_identity":
            mutated["task_id"] = "T7.2.X"
        elif target == "G02_live_source_bindings":
            mutated["source_integrity_declared"] = False
        elif target in check_map:
            mutated["manuscript"]["checks"][check_map[target]] = False
        elif target == "G13_previous_prose_contracts_live":
            mutated["parent_state"]["previous_contracts_live"]["results"] = False
        elif target == "G14_supplement_figure_parent_complete":
            mutated["parent_state"]["figure_records"] = 791
        elif target == "G15_ontology_exact":
            mutated["parent_state"]["ontology"]["value_states"] = mutated["parent_state"]["ontology"]["value_states"][:-1]
        elif target == "G16_long_rtl_parent_exact":
            mutated["parent_state"]["long_rtl"]["commit_attempts"] = 60
        elif target == "G17_integrated_rtl_parent_exact":
            mutated["parent_state"]["integrated_rtl"]["directed"] = 0
        elif target == "G18_preboard_parent_and_board_null":
            mutated["parent_state"]["board"]["nonnull"] = 1
        elif target == "G19_phase6c_correctness_and_reproduction":
            mutated["parent_state"]["phase6c"]["single_mismatches"] = 1
        elif target == "G20_phase6c_positive_is_lane_local":
            mutated["parent_state"]["phase6c"]["global_score"] = True
        elif target == "G21_phase6c_absence_and_negative_states":
            mutated["parent_state"]["phase6c"]["external_same_task"] = 1
        elif target == "G22_rows_complete_unique_and_stateful":
            mutated["supplement_rows"] = mutated["supplement_rows"][:-1]
        elif target == "G23_task_status":
            mutated["parent_state"]["task_status"] = "Todo"
        else:  # pragma: no cover
            raise AssertionError(f"unhandled mutation target: {target}")
        rejected = not evaluate_gates(mutated)[target]
        cases.append({"mutation_id": f"M{index + 1:02d}", "target_gate": target, "rejected": rejected})
    return {"count": len(cases), "detected": sum(case["rejected"] for case in cases), "cases": cases}


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": report["task_id"],
        "schema_version": report["schema_version"],
        "manuscript": report["manuscript"],
        "supplement_rows": report["supplement_rows"],
        "parent_state": report["parent_state"],
        "source_bindings": report["source_bindings"],
        "gates": report["gates"],
        "gate_summary": report["gate_summary"],
        "verdict": report["verdict"],
        "semantic_mutation_audit": report["semantic_mutation_audit"],
    }


def build_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manuscript": _manuscript_snapshot(),
        "supplement_rows": _supplement_rows(),
        "parent_state": _parent_state(),
        "source_bindings": {name: _binding(path) for name, path in SOURCE_PATHS.items()},
        "source_integrity_declared": True,
    }
    report["gates"] = evaluate_gates(report, check_live_sources=True)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_SUPPLEMENTARY_EVIDENCE_CONTRACT"
    report["semantic_mutation_audit"] = _semantic_mutation_audit(report)
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def _write_source_data(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = ("row_id", "topic", "supplement_state", "source_ids", "boundary")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    parent = report["parent_state"]
    lines = [
        "# T7.2.5 Supplementary 证据合同",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- gates：`{report['gate_summary']['passed']}/{report['gate_summary']['total']}`",
        f"- semantic mutations：`{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`",
        f"- evidence rows：`{len(report['supplement_rows'])}`",
        f"- long RTL：`{parent['long_rtl']['cycles']:,}` + `{parent['integrated_rtl']['cycles']:,}` cycles",
        f"- Phase 6C atlas：`{parent['phase6c']['atlas_cells']}` cells，global winner=`{parent['phase6c']['global_winner']}`",
        f"- board measured fields：`{parent['board']['nonnull']}/{parent['board']['fields']}` non-null",
        "",
        "| ID | 主题 | 状态 | 边界 |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['row_id']} | {row['topic']} | `{row['supplement_state']}` | {row['boundary']} |"
        for row in report["supplement_rows"]
    )
    lines.extend([
        "",
        "本合同把公式、冻结参数、完整 baseline/CI、负结果、RTL 长序列和 Phase 6C 来源定位绑定到同一附录；N/A、null、failed、negative、blocked 与 ineligible 不可互换，也不能跨 task signature 排名。",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any]) -> None:
    _write_source_data(report["supplement_rows"], DEFAULT_SOURCE_DATA)
    _atomic_json(report, DEFAULT_REPORT)
    _atomic_text(_markdown(report), DEFAULT_MARKDOWN)


def verify_report() -> tuple[bool, dict[str, bool]]:
    stored = _load_json(DEFAULT_REPORT)
    stored_gates = evaluate_gates(stored, check_live_sources=True)
    fresh = build_report()
    checks = {
        "identity": stored.get("task_id") == TASK_ID and stored.get("schema_version") == SCHEMA_VERSION,
        "live_sources": all(_binding_live(binding) for binding in stored["source_bindings"].values()),
        "all_stored_gates_pass": all(stored_gates.values()),
        "gate_snapshot_matches": stored.get("gates") == stored_gates,
        "mutation_audit_complete": stored["semantic_mutation_audit"]["count"] == stored["semantic_mutation_audit"]["detected"] == len(stored["gates"]),
        "analysis_sha256_live": stored.get("analysis_sha256") == _canonical_sha256(_analysis_payload(stored)),
        "fresh_analysis_matches": stored.get("analysis_sha256") == fresh.get("analysis_sha256"),
        "verdict": stored.get("verdict") == VERDICT,
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
        "mutations": {"detected": report["semantic_mutation_audit"]["detected"], "count": report["semantic_mutation_audit"]["count"]},
        "analysis_sha256": report["analysis_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
