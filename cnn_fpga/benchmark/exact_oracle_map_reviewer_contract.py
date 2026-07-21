"""T7.3.1 reviewer-response contract for exact/oracle MAP.

The contract separates three objects that are often conflated in review:
exact MAP for a frozen deployable likelihood, the truth-privileged simulator
oracle, and a channel/control optimum.  It also prevents a future prose edit
from claiming positive static-to-oracle gap closure when the frozen V4 result
is negative.
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
TASK_ID = "T7.3.1"
SCHEMA_VERSION = "t7.3.1-exact-oracle-map-reviewer-contract-v1"
VERDICT = "PASS_EXACT_ORACLE_MAP_INFORMATION_AND_CLAIM_BOUNDARY"

NOTE_PATH = ROOT / "docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex"
DEFAULT_REPORT = ROOT / "docs/t7_3_1_exact_oracle_map_reviewer_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_3_1_exact_oracle_map_reviewer_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/exact_oracle_map_reviewer_response.md"

SOURCE_PATHS = {
    "manuscript": NOTE_PATH,
    "bibliography": ROOT / "docs/paper_notes/CNN_FPGA_GKP_submission_refs.bib",
    "oracle_implementation": ROOT / "physics/oracle_map.py",
    "oracle_tests": ROOT / "tests/test_oracle_map.py",
    "oracle_validation": ROOT / "docs/t3_1_3_oracle_validation.json",
    "smooth_formal": ROOT / "docs/t6_7_1_smooth_formal_matrix.json",
    "tail_formal": ROOT / "docs/t6_7_2_abrupt_ood_tail_formal_matrix.json",
    "causal_headroom": ROOT / "docs/t6_10_1_causal_headroom.json",
    "claim_matrix": ROOT / "docs/t6_8_7_route_a_claim_matrix.json",
    "results_contract": ROOT / "docs/t7_2_3_results_evidence_contract.json",
    "supplement_contract": ROOT / "docs/t7_2_5_supplementary_evidence_contract.json",
    "task_board": ROOT / "docs/new_task_board.md",
    "implementation": Path(__file__).resolve(),
}

RESPONSE_STATES = (
    "TERM_DEFINITION",
    "INFORMATION_BOUNDARY",
    "DECISION_THEORY",
    "CURRENT_POSITIVE",
    "CURRENT_NEGATIVE",
    "CLAIM_BOUNDARY",
    "FUTURE_PROMOTION_GATE",
)

PROHIBITED_ASSERTIVE_PATTERNS = (
    "route-a closes the static-to-oracle gap",
    "route-a outperforms the oracle",
    "the oracle is deployable",
    "exact map is computationally impossible",
    "static map is an unfair baseline",
    "the hidden-state oracle is the channel-recovery optimum",
    "we omit static map",
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _find_by(items: Sequence[Mapping[str, Any]], key: str, value: str) -> Mapping[str, Any]:
    matches = [item for item in items if item.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"expected one {key}={value!r}, found {len(matches)}")
    return matches[0]


def _reviewer_paragraph(tex: str) -> str:
    start = re.search(r"\\paragraph\{Why not use exact/oracle MAP\?\}", tex)
    if start is None:
        raise ValueError("missing exact/oracle MAP reviewer paragraph")
    tail = tex[start.end():]
    end = tail.find("The Phase~6C results remain useful")
    if end < 0:
        raise ValueError("missing reviewer paragraph end marker")
    return tail[:end].strip()


def _board_task_status(board_text: str, task_id: str) -> str | None:
    match = re.search(rf"^\|\s*{re.escape(task_id)}\s*\|\s*([^|]+?)\s*\|", board_text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _prior_contract_live(report: Mapping[str, Any]) -> bool:
    return (
        all(bool(value) for value in report.get("gates", {}).values())
        and all(_binding_live(binding) for binding in report.get("source_bindings", {}).values())
    )


def _evidence_snapshot() -> dict[str, Any]:
    oracle = _load_json(SOURCE_PATHS["oracle_validation"])
    smooth = _load_json(SOURCE_PATHS["smooth_formal"])
    tail = _load_json(SOURCE_PATHS["tail_formal"])
    causal = _load_json(SOURCE_PATHS["causal_headroom"])
    claims = _load_json(SOURCE_PATHS["claim_matrix"])
    results = _load_json(SOURCE_PATHS["results_contract"])
    supplement = _load_json(SOURCE_PATHS["supplement_contract"])

    smooth_rows = smooth["analysis"]["method_summaries"]
    static = _find_by(smooth_rows, "method_id", "static_joint_map")
    proposed = _find_by(smooth_rows, "method_id", "proposed_route_a")
    hidden = _find_by(smooth_rows, "method_id", "hidden_state_oracle")
    window = _find_by(smooth_rows, "method_id", "window_map")
    ewma = _find_by(smooth_rows, "method_id", "ewma_adaptive_map")
    gap = smooth["analysis"]["oracle_gap_closure"]
    contrast = smooth["analysis"]["primary_contrast"]

    tail_rows = tail["analysis"]["family_method_summaries"]
    step_rows = [row for row in tail_rows if row["family"] == "step_calibration_shift"]
    step_static = _find_by(step_rows, "method_id", "static_joint_map")
    step_proposed = _find_by(step_rows, "method_id", "proposed_route_a")

    nested = causal["development_audit"]["nested_audit"]
    claim_states = {row["claim_id"]: row["state"] for row in claims["claims"]}
    return {
        "oracle_reference": {
            "samples": oracle["regime_matrix"]["samples"],
            "static_ler": oracle["regime_matrix"]["static_error_rate"],
            "oracle_ler": oracle["regime_matrix"]["oracle_error_rate"],
            "static_minus_oracle": oracle["regime_matrix"]["static_minus_oracle"],
            "allowed": oracle["claim_boundary"]["allowed"],
            "forbidden": oracle["claim_boundary"]["forbidden"],
        },
        "smooth": {
            "decisions_per_method": int(static["decisions"]),
            "static_ler": static["average_ler_equal_family_seed"],
            "window_ler": window["average_ler_equal_family_seed"],
            "ewma_ler": ewma["average_ler_equal_family_seed"],
            "proposed_ler": proposed["average_ler_equal_family_seed"],
            "oracle_ler": hidden["average_ler_equal_family_seed"],
            "oracle_deployable": hidden["deployable"],
            "gap_formula": gap["formula"],
            "gap_closure": gap["gap_closure"],
            "gap_ci95": gap["gap_closure_ci95"],
            "denominator": gap["static_minus_oracle"],
            "primary_ewma_contrast": contrast,
        },
        "calibration_shift": {
            "static_average_ler": step_static["average_ler"],
            "proposed_average_ler": step_proposed["average_ler"],
            "static_worst_errors_per_512": step_static["global_worst_window_error_count"],
            "proposed_worst_errors_per_512": step_proposed["global_worst_window_error_count"],
        },
        "causal_headroom": {
            "trajectories": causal["development_audit"]["trajectory_count"],
            "decisions": nested["total_decisions"],
            "selector_relative_headroom": nested["existing_expert_causal_headroom"],
            "fixed_mixture_relative_headroom": nested["heldout_fixed_posterior_mixture"]["relative_headroom"],
            "incremental_action_space_headroom": nested["expanded_candidate_action_oracle"]["incremental_action_space_headroom_vs_baseline"],
            "incremental_errors": nested["expanded_candidate_action_oracle"]["incremental_errors_avoided_beyond_existing_hard_actions"],
            "truth_privileged": nested["expanded_candidate_action_oracle"]["truth_privileged"],
            "verdict": causal["verdict"],
        },
        "claim_states": claim_states,
        "parent_contracts": {
            "results_live": _prior_contract_live(results),
            "supplement_live": _prior_contract_live(supplement),
            "results_verdict": results["verdict"],
            "supplement_verdict": supplement["verdict"],
        },
    }


def _response_rows(evidence: Mapping[str, Any]) -> list[dict[str, str]]:
    smooth = evidence["smooth"]
    calibration = evidence["calibration_shift"]
    causal = evidence["causal_headroom"]
    oracle = evidence["oracle_reference"]
    raw = [
        ("OR-001", "frozen_model_exact_map", "TERM_DEFINITION", "T3.1.2/T6.7.1", "Exact MAP for a specified frozen likelihood is a deployable baseline and is not omitted."),
        ("OR-002", "hidden_state_oracle", "TERM_DEFINITION", "T1.3.2/T3.1.3", "Per-round MAP receives exact simulator theta_t; it is an assumed-model reference."),
        ("OR-003", "channel_control_optimum", "TERM_DEFINITION", "T1.4.5/T3.2.9/T5.3.5", "Decoder oracle is not a channel-recovery or finite-horizon control optimum."),
        ("OR-004", "observed_information", "INFORMATION_BOUNDARY", "T6.5.2", "Deployable methods receive quantized syndrome history, integrity fields, and causal expert state."),
        ("OR-005", "privileged_information", "INFORMATION_BOUNDARY", "T3.1.3", "Exact mean, covariance, outlier mixture, regime, burst state, and labels remain evaluator-only."),
        ("OR-006", "online_calibration", "INFORMATION_BOUNDARY", "spitz2018/wagner2021/sivak2024", "Noise estimation uses finite causal data and does not reveal theta_t instantaneously."),
        ("OR-007", "bayes_risk_role", "DECISION_THEORY", "T1.3.2", "Within the assumed per-round likelihood and zero-one loss, hidden-state MAP minimizes conditional Bayes risk."),
        ("OR-008", "oracle_nonzero", "DECISION_THEORY", "T3.1.3", f"Oracle LER is nonzero ({oracle['oracle_ler']:.8f}); oracle does not mean perfect physical correction."),
        ("OR-009", "locked_ewma_contrast", "CURRENT_POSITIVE", "T6.7.1", f"EWMA minus Route-A is {smooth['primary_ewma_contrast']['estimate']:.10g} with a positive paired interval."),
        ("OR-010", "static_ordering", "CURRENT_NEGATIVE", "T6.7.1", f"Smooth static/Route-A/oracle LER = {smooth['static_ler']:.8g}/{smooth['proposed_ler']:.8g}/{smooth['oracle_ler']:.8g}."),
        ("OR-011", "negative_gap_closure", "CURRENT_NEGATIVE", "T6.7.1", f"Static-to-oracle closure = {smooth['gap_closure']:.8f}, CI [{smooth['gap_ci95'][0]:.8f},{smooth['gap_ci95'][1]:.8f}]."),
        ("OR-012", "window_counterexample", "CURRENT_NEGATIVE", "T6.7.1", f"Window MAP LER {smooth['window_ler']:.8g} is lower than Route-A {smooth['proposed_ler']:.8g}."),
        ("OR-013", "calibration_counterexample", "CURRENT_NEGATIVE", "T6.7.2", f"Calibration worst is {calibration['proposed_worst_errors_per_512']}/512 versus static {calibration['static_worst_errors_per_512']}/512."),
        ("OR-014", "causal_selector_stop", "CURRENT_NEGATIVE", "T6.10.1", f"Nested selector headroom = {100*causal['selector_relative_headroom']:.4f}%; V5 stopped."),
        ("OR-015", "action_oracle_nonpromotion", "CLAIM_BOUNDARY", "T6.10.1", f"Truth-privileged action expansion adds only {causal['incremental_errors']} errors avoided ({100*causal['incremental_action_space_headroom']:.5f}%)."),
        ("OR-016", "allowed_current_claim", "CLAIM_BOUNDARY", "T6.8.7/T7.2.3", "Current claim is restricted EWMA improvement plus information-safe architecture, not positive oracle-gap closure."),
        ("OR-017", "prohibited_claims", "CLAIM_BOUNDARY", "T6.8.7", "Do not claim oracle deployability, oracle superiority, exact-MAP impossibility, or static omission."),
        ("OR-018", "future_gap_gate", "FUTURE_PROMOTION_GATE", "T7.3.1", "Require held-out positive paired gap closure against the strongest deployable comparator."),
        ("OR-019", "new_observed_state", "FUTURE_PROMOTION_GATE", "T8.1/T8.2", "A physically measured calibration variable defines a new observed-input signature with delay and budget."),
        ("OR-020", "population_crossing_audit", "FUTURE_PROMOTION_GATE", "T7.3.1", "Any apparent oracle crossing requires uncertainty, object, model, and implementation audits."),
    ]
    return [
        {"row_id": row_id, "topic": topic, "response_state": state, "source_ids": source, "boundary": boundary}
        for row_id, topic, state, source, boundary in raw
    ]


def _manuscript_snapshot() -> dict[str, Any]:
    tex = NOTE_PATH.read_text(encoding="utf-8")
    paragraph = _reviewer_paragraph(tex)
    normalized = _normalize(paragraph)
    full = _normalize(tex)
    checks = {
        "three_map_objects": all(token in normalized for token in (
            "specified frozen likelihood", "registered hidden-state oracle", "arbitrary channel-recovery",
        )),
        "information_sets_separated": all(token in normalized for token in (
            "simulator's exact", "absent from the observed packet", "truth leakage", "causal observed information",
        )),
        "bayes_role": all(token in normalized for token in (
            "zero--one logical loss", "minimizes conditional bayes risk", "not expected to beat its population risk",
        )),
        "not_compute_strawman": all(token in normalized for token in (
            "not an assertion that exact likelihood evaluation is computationally impossible",
            "full two-dimensional static joint map",
        )),
        "negative_gap_disclosed": all(token in normalized for token in (
            "do \\emph{not} establish positive gap closure", "-0.03046", "-0.04966,-0.01119",
        )),
        "tail_counterexample_disclosed": all(token in normalized for token in ("181/512", "32/512", "static map")),
        "current_claim_restricted": all(token in normalized for token in (
            "only the narrower paired improvement", "locked ewma", "not a completed contribution",
        )),
        "future_gate": all(token in normalized for token in (
            "positive held-out gap closure", "strongest deployable comparator", "new observed-input task signature",
        )),
        "causal_headroom_disclosed": all(token in normalized for token in ("-0.2322", "nine avoided", "0.02549")),
        "citations_present": all(token in paragraph for token in ("spitz2018", "wagner2021", "sivak2024")),
        "prohibited_assertions_absent": not any(pattern in full for pattern in PROHIBITED_ASSERTIVE_PATTERNS),
    }
    return {
        "paragraph_sha256": hashlib.sha256(paragraph.encode("utf-8")).hexdigest(),
        "characters": len(paragraph),
        "checks": checks,
    }


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    evidence = report["evidence"]
    smooth = evidence["smooth"]
    calibration = evidence["calibration_shift"]
    causal = evidence["causal_headroom"]
    oracle = evidence["oracle_reference"]
    parent = evidence["parent_contracts"]
    checks = report["manuscript"]["checks"]
    rows = report["response_rows"]
    states = {row["response_state"] for row in rows}
    source_live = all(_binding_live(binding) for binding in report["source_bindings"].values()) if check_live_sources else bool(report["source_integrity_declared"])
    return {
        "G01_identity": report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION,
        "G02_live_source_bindings": source_live,
        "G03_three_map_objects": bool(checks["three_map_objects"]),
        "G04_information_sets_separated": bool(checks["information_sets_separated"]),
        "G05_conditional_bayes_role": bool(checks["bayes_role"]),
        "G06_no_computational_impossibility_strawman": bool(checks["not_compute_strawman"]),
        "G07_oracle_reference_is_nonperfect_and_nonuniversal": oracle["oracle_ler"] > 0.0 and "nondeployable" in oracle["allowed"] and "channel-recovery optimum" in oracle["forbidden"],
        "G08_smooth_values_and_nondeployability": smooth["decisions_per_method"] == 28_311_552 and smooth["oracle_deployable"] is False and smooth["oracle_ler"] < smooth["static_ler"] < smooth["proposed_ler"],
        "G09_gap_formula_sign_and_interval": smooth["gap_formula"] == "(static_LER-proposed_LER)/(static_LER-oracle_LER)" and smooth["denominator"] > 0.0 and smooth["gap_closure"] < 0.0 and max(smooth["gap_ci95"]) < 0.0,
        "G10_static_window_counterevidence": smooth["static_ler"] < smooth["proposed_ler"] and smooth["window_ler"] < smooth["proposed_ler"] and bool(checks["negative_gap_disclosed"]),
        "G11_calibration_counterexample": calibration["static_worst_errors_per_512"] == 32 and calibration["proposed_worst_errors_per_512"] == 181 and bool(checks["tail_counterexample_disclosed"]),
        "G12_positive_claim_only_locked_ewma": smooth["primary_ewma_contrast"]["ci95_low"] > 0.0 and bool(checks["current_claim_restricted"]),
        "G13_causal_headroom_not_oracle_gain": causal["selector_relative_headroom"] < 0.0 and causal["incremental_errors"] == 9 and causal["incremental_action_space_headroom"] < 0.001 and causal["truth_privileged"] is True and causal["verdict"] == "NO_GO_V5_INSUFFICIENT_ACTION_SPACE_HEADROOM" and bool(checks["causal_headroom_disclosed"]),
        "G14_claim_matrix_falsification_preserved": evidence["claim_states"].get("STATIC_GKP_SUPERIORITY") == "FALSIFIED" and evidence["claim_states"].get("SMOOTH_LOCKED_EWMA_ADVANTAGE") == "SUPPORTED_PAIRED_OUTCOME",
        "G15_future_promotion_is_heldout_and_deployable": bool(checks["future_gate"]),
        "G16_literature_citations_present": bool(checks["citations_present"]),
        "G17_parent_prose_contracts_live": parent == {
            "results_live": True,
            "supplement_live": True,
            "results_verdict": "PASS_RESULTS_COMPLETE_NEGATIVE_AND_SECONDARY_BOUNDARIES",
            "supplement_verdict": "PASS_SUPPLEMENT_COMPLETE_REPRODUCIBLE_AND_NONMIXING",
        },
        "G18_response_rows_complete": len(rows) == 20 and len({row["row_id"] for row in rows}) == 20 and states == set(RESPONSE_STATES),
        "G19_task_board_terminal_and_next": report["task_status"] == {"T7.3.1": "Done", "T7.3.2": "In Progress"},
        "G20_prohibited_assertions_absent": bool(checks["prohibited_assertions_absent"]),
    }


def _semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    check_targets = {
        "G03_three_map_objects": "three_map_objects",
        "G04_information_sets_separated": "information_sets_separated",
        "G05_conditional_bayes_role": "bayes_role",
        "G06_no_computational_impossibility_strawman": "not_compute_strawman",
        "G10_static_window_counterevidence": "negative_gap_disclosed",
        "G11_calibration_counterexample": "tail_counterexample_disclosed",
        "G12_positive_claim_only_locked_ewma": "current_claim_restricted",
        "G13_causal_headroom_not_oracle_gain": "causal_headroom_disclosed",
        "G15_future_promotion_is_heldout_and_deployable": "future_gate",
        "G16_literature_citations_present": "citations_present",
        "G20_prohibited_assertions_absent": "prohibited_assertions_absent",
    }
    cases: list[dict[str, Any]] = []
    for index, target in enumerate(evaluate_gates(report)):
        mutated = copy.deepcopy(report)
        if target == "G01_identity":
            mutated["task_id"] = "T7.3.X"
        elif target == "G02_live_source_bindings":
            mutated["source_integrity_declared"] = False
        elif target in check_targets:
            mutated["manuscript"]["checks"][check_targets[target]] = False
        elif target == "G07_oracle_reference_is_nonperfect_and_nonuniversal":
            mutated["evidence"]["oracle_reference"]["oracle_ler"] = 0.0
        elif target == "G08_smooth_values_and_nondeployability":
            mutated["evidence"]["smooth"]["oracle_deployable"] = True
        elif target == "G09_gap_formula_sign_and_interval":
            mutated["evidence"]["smooth"]["gap_ci95"] = [-0.04, 0.01]
        elif target == "G14_claim_matrix_falsification_preserved":
            mutated["evidence"]["claim_states"]["STATIC_GKP_SUPERIORITY"] = "SUPPORTED"
        elif target == "G17_parent_prose_contracts_live":
            mutated["evidence"]["parent_contracts"]["supplement_live"] = False
        elif target == "G18_response_rows_complete":
            mutated["response_rows"] = mutated["response_rows"][:-1]
        elif target == "G19_task_board_terminal_and_next":
            mutated["task_status"]["T7.3.1"] = "In Progress"
        else:  # pragma: no cover
            raise AssertionError(f"unhandled mutation target: {target}")
        cases.append({
            "mutation_id": f"M{index + 1:02d}",
            "target_gate": target,
            "rejected": not evaluate_gates(mutated)[target],
        })
    return {"count": len(cases), "detected": sum(case["rejected"] for case in cases), "cases": cases}


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": report["task_id"],
        "schema_version": report["schema_version"],
        "reviewer_response": report["reviewer_response"],
        "manuscript": report["manuscript"],
        "evidence": report["evidence"],
        "response_rows": report["response_rows"],
        "task_status": report["task_status"],
        "source_bindings": report["source_bindings"],
        "gates": report["gates"],
        "gate_summary": report["gate_summary"],
        "semantic_mutation_audit": report["semantic_mutation_audit"],
        "verdict": report["verdict"],
    }


def build_report() -> dict[str, Any]:
    evidence = _evidence_snapshot()
    board = SOURCE_PATHS["task_board"].read_text(encoding="utf-8")
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviewer_response": {
            "question": "Why not use exact/oracle MAP?",
            "short_answer": (
                "We do use exact MAP for the frozen model as a strong baseline. The hidden-state oracle "
                "instead receives simulator truth unavailable to a deployable decoder and is therefore an "
                "information upper reference, not an implementation choice. Current V4 data do not close "
                "the static-to-oracle gap; only the locked-EWMA contrast is positive."
            ),
            "allowed_claim": (
                "The oracle quantifies assumed-model headroom; a future adaptive claim requires positive "
                "held-out paired gap closure against the strongest deployable comparator."
            ),
            "current_result": "NEGATIVE_GAP_CLOSURE_RETAINED",
        },
        "manuscript": _manuscript_snapshot(),
        "evidence": evidence,
        "response_rows": _response_rows(evidence),
        "task_status": {
            "T7.3.1": _board_task_status(board, "T7.3.1"),
            "T7.3.2": _board_task_status(board, "T7.3.2"),
        },
        "source_bindings": {name: _binding(path) for name, path in SOURCE_PATHS.items()},
        "source_integrity_declared": True,
    }
    report["gates"] = evaluate_gates(report, check_live_sources=True)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_EXACT_ORACLE_MAP_REVIEWER_CONTRACT"
    report["semantic_mutation_audit"] = _semantic_mutation_audit(report)
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def _write_source_data(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = ("row_id", "topic", "response_state", "source_ids", "boundary")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    smooth = report["evidence"]["smooth"]
    calibration = report["evidence"]["calibration_shift"]
    causal = report["evidence"]["causal_headroom"]
    lines = [
        "# T7.3.1：为何不用 exact/oracle MAP？",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- gates：`{report['gate_summary']['passed']}/{report['gate_summary']['total']}`",
        f"- semantic mutations：`{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`",
        "",
        "## 可直接用于审稿回复的短答",
        "",
        "我们已经把 exact MAP 用作冻结模型下的强 static baseline；没有把它排除。hidden-state oracle 则逐轮读取 simulator 的真实 `theta_t`，其信息不在 deployable observed packet 中，所以它是不可部署的 assumed-model reference，而不是可选择的在线实现。MAP 在已知真实条件分布和 0--1 loss 下给出条件 Bayes 最优，合理目标是用因果可观测信息缩小 static-to-oracle gap，而不是声称超过 oracle。",
        "",
        "当前 V4 也不能声称已经缩小该 gap：",
        "",
        f"- smooth `p_L`：static `{smooth['static_ler']:.8g}`，Route-A `{smooth['proposed_ler']:.8g}`，oracle `{smooth['oracle_ler']:.8g}`；",
        f"- static-to-oracle gap closure：`{smooth['gap_closure']:.8f}`，95% CI `[{smooth['gap_ci95'][0]:.8f}, {smooth['gap_ci95'][1]:.8f}]`；",
        f"- calibration-shift worst-window：Route-A `{calibration['proposed_worst_errors_per_512']}/512`，static `{calibration['static_worst_errors_per_512']}/512`；",
        f"- V5 nested causal selector：`{100*causal['selector_relative_headroom']:.4f}%`；新增 action family 只多避免 `{causal['incremental_errors']}` 个错误，即 `{100*causal['incremental_action_space_headroom']:.5f}%`。",
        "",
        "因此当前只保留相对预注册 locked EWMA 的窄 paired improvement 与 observed-only/fail-closed 架构主张。未来只有在独立 held-out 漂移上，相对最强 deployable comparator 得到正的 paired gap closure，才可升级为自适应 LER 贡献。若未来真实仪器能在线给出校准状态，该变量应进入新的 observed-input task signature，并计入测量延迟、更新预算和误差。",
        "",
        "## 原子证据与边界",
        "",
        "| ID | 主题 | 状态 | 来源 | 边界 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['row_id']} | {row['topic']} | `{row['response_state']}` | {row['source_ids']} | {row['boundary']} |"
        for row in report["response_rows"]
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any]) -> None:
    _write_source_data(report["response_rows"], DEFAULT_SOURCE_DATA)
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
        "mutations": {
            "detected": report["semantic_mutation_audit"]["detected"],
            "count": report["semantic_mutation_audit"]["count"],
        },
        "analysis_sha256": report["analysis_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
