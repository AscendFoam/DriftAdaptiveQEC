"""T7.2.4 evidence contract for Discussion and Conclusion.

The manuscript must interpret the frozen V4/V5/Phase-6C evidence without
promoting a simulator into a cavity/transmon experiment, a post-route estimate
into a board result, or per-round LER into physical break-even.  This module
also freezes the cost inventory, external-validity limitations, and monotonic
path from real-data shadow mode to a registered physical break-even test.
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
TASK_ID = "T7.2.4"
SCHEMA_VERSION = "t7.2.4-discussion-conclusion-contract-v1"
VERDICT = "PASS_DISCUSSION_LIMITATIONS_COST_AND_PHYSICAL_TRANSITION_BOUNDARIES"

NOTE_PATH = ROOT / "docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex"
DEFAULT_REPORT = ROOT / "docs/t7_2_4_discussion_conclusion_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_2_4_discussion_conclusion_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/discussion_conclusion_contract.md"

SOURCE_PATHS = {
    "manuscript": NOTE_PATH,
    "claim_matrix": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix.json",
    "introduction_contract": ROOT / "docs/t7_2_1_introduction_related_work_contract.json",
    "methods_contract": ROOT / "docs/t7_2_2_methods_evidence_contract.json",
    "results_contract": ROOT / "docs/t7_2_3_results_evidence_contract.json",
    "v4_final_gate": ROOT / "docs/t6_9_3_route_a_final_evidence_gate.json",
    "board_blocker": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
    "v5_final_gate": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json",
    "phase6c_integrity": ROOT / "docs/t6_19_3_secondary_evidence_integrity.json",
    "task_board": ROOT / "docs/new_task_board.md",
    "experiment_plan": ROOT / "docs/experiment_plan.md",
    "scope_freeze": ROOT / "docs/tasks/T0.1.1_scope_freeze.md",
    "metric_analysis": ROOT / "docs/deep_research_reports/GKP解码指标分析.md",
    "implementation": Path(__file__).resolve(),
}

DISCUSSION_STATES = (
    "INTERPRETATION_ESTABLISHED",
    "LIMITATION_REQUIRED",
    "COST_REQUIRED",
    "EXTERNAL_VALIDITY_REQUIRED",
    "FUTURE_GATE_ONLY",
    "CONCLUSION_BOUNDARY",
)

REQUIRED_SUBSECTIONS = (
    "Evidence-supported contribution",
    "When regime awareness helps",
    "Safety, performance, and intervention cost",
    "Computational and hardware cost",
    "External validity and missing physical evidence",
    "Relationship to prior decoder classes",
    "Failure-informed design priorities",
    "Staged path to real data, board measurement, and physical break-even",
)

REQUIRED_TRANSITION_STAGES = (
    "Real-data intake",
    "Shadow mode",
    "Board/HIL",
    "Guarded QPU",
    "Physical effectiveness",
    "Break-even",
)

PROHIBITED_ASSERTIVE_PATTERNS = (
    "we demonstrate real beyond-break-even",
    "we achieve physical beyond-break-even",
    "we measured a logical lifetime gain",
    "our calibrated cavity--transmon digital twin",
    "we trained the cnn on the target board",
    "we perform on-board ppo training",
    "board-measured latency is 222.222",
    "we are faster than existing fpga qec decoders",
    "the fastest fpga qec decoder",
    "we establish a surface-code threshold",
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


def _extract_section(tex: str, title: str) -> str:
    marker = re.search(rf"\\section\{{{re.escape(title)}\}}", tex)
    if marker is None:
        raise ValueError(f"missing section: {title}")
    tail = tex[marker.end():]
    next_section = re.search(r"\\section\{", tail)
    return tail[: next_section.start() if next_section else len(tail)].strip()


def _headings(text: str, command: str) -> list[str]:
    return re.findall(rf"\\{command}\{{([^}}]+)\}}", text)


def _all_tokens(text: str, tokens: Sequence[str]) -> bool:
    return all(token.lower() in text for token in tokens)


def _discussion_rows() -> list[dict[str, str]]:
    raw_rows = [
        ("DC-001", "restricted_contribution", "INTERPRETATION_ESTABLISHED", "V4/T7.1.1", "Integration and falsification, not a winning decoder"),
        ("DC-002", "role_separation", "INTERPRETATION_ESTABLISHED", "T6.5--T7.2.3", "MAP, safety FSM, fast path, and learning modules own different claims"),
        ("DC-003", "locked_ewma_positive", "INTERPRETATION_ESTABLISHED", "T6.7.1/T6.7.4", "Positive only against the preregistered EWMA contrast"),
        ("DC-004", "static_window_negative", "LIMITATION_REQUIRED", "T6.7.1/T6.9.3", "Static and Window prevent a best-decoder claim"),
        ("DC-005", "v5_causal_stop", "LIMITATION_REQUIRED", "T6.10.1/T6.15.5", "Observed-only causal/action headroom failed before V5 implementation"),
        ("DC-006", "tail_noninferiority", "INTERPRETATION_ESTABLISHED", "T6.7.2/T6.7.4", "Safety non-inferiority is not broad tail superiority"),
        ("DC-007", "intervention_occupancy", "COST_REQUIRED", "T6.7.2", "Fallback and unnecessary fallback can dominate tail intervals"),
        ("DC-008", "host_update_cost", "COST_REQUIRED", "T6.5.2/T6.8.3", "Slow-loop cadence and compute budget are separate from fast-path cycles"),
        ("DC-009", "fast_path_cost", "COST_REQUIRED", "T6.7.3/T6.9.1", "Six-cycle II=1 and P&R are pre-board evidence"),
        ("DC-010", "offline_learning_only", "LIMITATION_REQUIRED", "T4.4/T5.5", "No board-resident or on-board training"),
        ("DC-011", "board_measurement_null", "LIMITATION_REQUIRED", "T6.9.2", "All 42 physical-board fields remain null"),
        ("DC-012", "single_mode_model_scope", "EXTERNAL_VALIDITY_REQUIRED", "T0.1.1/T2.1", "Single-mode square-lattice syndrome/effective model"),
        ("DC-013", "no_calibrated_cavity_transmon", "EXTERNAL_VALIDITY_REQUIRED", "T0.1.2/T8", "No calibrated cavity/transmon pulse-level device model or experiment"),
        ("DC-014", "synthetic_drift_scope", "EXTERNAL_VALIDITY_REQUIRED", "T1.3/T6.6", "Held-out synthetic generators do not establish real drift prevalence"),
        ("DC-015", "no_physical_lifetime", "LIMITATION_REQUIRED", "T5.5.2/T8", "Per-round simulated LER is not a physical lifetime"),
        ("DC-016", "no_outer_threshold", "LIMITATION_REQUIRED", "T0.1.1", "No surface-code or fault-tolerant threshold from single-mode data"),
        ("DC-017", "task_signature_comparison", "INTERPRETATION_ESTABLISHED", "T6.16--T6.19", "CI/ML/NN/AQEC/CPD/NMF/FPGA lanes have no global leaderboard"),
        ("DC-018", "phase6c_separation", "LIMITATION_REQUIRED", "T6.19.3", "Secondary CPD/CNOT/AQEC evidence cannot rescue V5"),
        ("DC-019", "identifiability_first", "INTERPRETATION_ESTABLISHED", "T6.10.1", "Require prospective causal information before estimator expansion"),
        ("DC-020", "action_value_first", "INTERPRETATION_ESTABLISHED", "T6.10.1", "Require realizable action-value headroom before compiler expansion"),
        ("DC-021", "real_data_intake", "FUTURE_GATE_ONLY", "T8.1.1--T8.1.3", "Immutable metadata, units, labels, permission, and chronological splits"),
        ("DC-022", "shadow_mode", "FUTURE_GATE_ONLY", "T8.2.1", "Prospective output logging without actuation"),
        ("DC-023", "board_hil", "FUTURE_GATE_ONLY", "T6.9.2", "Named-board streaming HIL populates timing, power, and 42 fields"),
        ("DC-024", "guarded_qpu", "FUTURE_GATE_ONLY", "T8.2.2--T8.2.3", "Frame first; displacement only after separate authorization"),
        ("DC-025", "physical_effectiveness", "FUTURE_GATE_ONLY", "T5.5.2/T8", "Matched corrected and best-physical channels on the same device"),
        ("DC-026", "break_even_gate", "FUTURE_GATE_ONLY", "T5.5.2/T8", "Decay-rate ratio with simultaneous lower confidence bound above one"),
        ("DC-027", "balanced_conclusion", "CONCLUSION_BOUNDARY", "T7.1.1/T7.2.3", "Conclusion retains positives, negatives, nulls, and prohibited upgrades"),
    ]
    return [
        {
            "row_id": row_id,
            "topic": topic,
            "discussion_state": state,
            "source_ids": sources,
            "boundary": boundary,
        }
        for row_id, topic, state, sources, boundary in raw_rows
    ]


def _manuscript_snapshot() -> dict[str, Any]:
    tex = NOTE_PATH.read_text(encoding="utf-8")
    discussion = _extract_section(tex, "Discussion")
    conclusion = _extract_section(tex, "Conclusion")
    normalized_discussion = _normalize(discussion)
    normalized_conclusion = _normalize(conclusion)
    normalized_full = _normalize(tex)
    sections = _headings(tex, "section")
    subsections = _headings(discussion, "subsection")
    positions = {title: subsections.index(title) for title in REQUIRED_SUBSECTIONS if title in subsections}
    checks = {
        "section_order": sections.index("Discussion") < sections.index("Reproducibility and evidence availability") < sections.index("Conclusion"),
        "required_subsections": all(title in subsections for title in REQUIRED_SUBSECTIONS),
        "ordered_subsections": positions == {title: index for index, title in enumerate(REQUIRED_SUBSECTIONS)},
        "restricted_contribution": _all_tokens(normalized_discussion, (
            "restricted pre-board contract system", "not a winning adaptive decoder or a physical qec experiment",
            "integration and falsification result", "map owns", "fpga-facing path owns",
        )),
        "regime_interpretation": _all_tokens(normalized_discussion, (
            "static joint map and window map", "reject a best-deployable-decoder claim",
            "prospective identifiability", "action-value headroom", "fresh development split",
        )),
        "safety_and_cost": _all_tokens(normalized_discussion, (
            "59--96\\%", "fail closed", "invariant rather than an ler claim",
            "avoided and induced errors", "unnecessary-fallback rates", "recovery lag",
        )),
        "cost_inventory": _all_tokens(normalized_discussion, (
            "tab:discussion-costs", "4000-cycle cadence", "six-cycle",
            "one-million-cycle integer/cxxrtl equivalence", "training is offline",
            "no ppo, cnn, rnn, hmm", "all 42 measured fields are absent",
        )),
        "external_validity": _all_tokens(normalized_discussion, (
            "single-mode square-lattice", "do not constitute a calibrated driven open-system model",
            "not a measured logical decay rate", "no outer-code scaling result",
            "no microwave, adc/dac, cavity, transmon", "missing experiments",
        )),
        "comparison_boundaries": _all_tokens(normalized_discussion, (
            "does not support a single leaderboard", "relative to static gkp decoding",
            "relative to puviani nmf", "relative to existing fpga qec decoders",
            "none of these task-local results rescues", "global rank",
        )),
        "failure_informed_priorities": _all_tokens(normalized_discussion, (
            "causal information", "realizable trusted bank", "cost-aware policy",
            "chronological, device-separated real records", "same observed-only interface",
        )),
        "transition_ladder": all(stage.lower() in normalized_discussion for stage in REQUIRED_TRANSITION_STAGES) and _all_tokens(
            normalized_discussion,
            ("monotonic evidence ladder", "at least one million cycles", "separate authorization",
             "simultaneous confidence lower bound above one", "correction overhead", "rejected shots"),
        ),
        "explicit_nonclaims": _all_tokens(normalized_discussion + " " + normalized_conclusion, (
            "no real beyond-break-even", "no claim of a calibrated cavity--transmon model",
            "board-resident training", "measured fpga speed/power", "closed-loop quantum-device operation",
        )),
        "balanced_conclusion": _all_tokens(normalized_conclusion, (
            "static joint map and window map outperform", "v5 was stopped before implementation",
            "official gqf/nmf comparison remains blocked", "all 42 board-measured fields remain null",
            "no exact same-task comparator", "restricted simulator/pre-board contract paper",
        )),
        "prohibited_assertions_absent": not any(pattern in normalized_full for pattern in PROHIBITED_ASSERTIVE_PATTERNS),
    }
    return {
        "discussion_sha256": hashlib.sha256(discussion.encode("utf-8")).hexdigest(),
        "conclusion_sha256": hashlib.sha256(conclusion.encode("utf-8")).hexdigest(),
        "discussion_characters": len(discussion),
        "conclusion_characters": len(conclusion),
        "subsections": subsections,
        "transition_stages": list(REQUIRED_TRANSITION_STAGES),
        "checks": checks,
    }


def _prior_contract_live(contract: Mapping[str, Any]) -> bool:
    return (
        all(bool(value) for value in contract.get("gates", {}).values())
        and all(_binding_live(binding) for binding in contract.get("source_bindings", {}).values())
    )


def _board_task_status(board_text: str, task_id: str) -> str | None:
    match = re.search(rf"^\|\s*{re.escape(task_id)}\s*\|\s*([^|]+?)\s*\|", board_text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _parent_state() -> dict[str, Any]:
    claim = _load_json(SOURCE_PATHS["claim_matrix"])
    intro = _load_json(SOURCE_PATHS["introduction_contract"])
    methods = _load_json(SOURCE_PATHS["methods_contract"])
    results = _load_json(SOURCE_PATHS["results_contract"])
    v4 = _load_json(SOURCE_PATHS["v4_final_gate"])
    board = _load_json(SOURCE_PATHS["board_blocker"])
    v5 = _load_json(SOURCE_PATHS["v5_final_gate"])
    phase6c = _load_json(SOURCE_PATHS["phase6c_integrity"])
    board_text = SOURCE_PATHS["task_board"].read_text(encoding="utf-8")
    measured = board["measured_results"]
    result_states = {row["result_state"] for row in results["result_rows"]}
    false_external_prerequisites = [
        item["prerequisite"] for item in board["prerequisite_ledger"]
        if item["kind"] == "physical_external" and not item["passed"]
    ]
    return {
        "verdicts": {
            "claim_matrix": claim["verdict"],
            "introduction": intro["verdict"],
            "methods": methods["verdict"],
            "results": results["verdict"],
            "v4": v4["verdict"],
            "board": board["verdict"],
            "v5": v5["verdict"],
            "phase6c": phase6c["verdict"],
        },
        "previous_contracts_live": {
            "introduction": _prior_contract_live(intro),
            "methods": _prior_contract_live(methods),
            "results": _prior_contract_live(results),
        },
        "claim_count": len(claim["claims"]),
        "result_row_count": len(results["result_rows"]),
        "result_states": sorted(result_states),
        "v5": {
            "dropped_tasks": len(v5["dropped_tasks"]),
            "downstream_outputs": len(v5["v5_downstream_outputs_found"]),
            "formal_artifacts_exist": bool(
                v5["formal_access"]["v5_formal_manifest_exists"]
                or v5["formal_access"]["v5_formal_output_exists"]
            ),
        },
        "board": {
            "measured_field_count": len(measured),
            "nonnull_field_count": sum(value is not None for value in measured.values()),
            "false_external_prerequisites": false_external_prerequisites,
            "same_task_speed_claim": board["claim_boundary"]["fpga_speed_advantage"],
        },
        "task_status": {
            "T7.2.4": _board_task_status(board_text, "T7.2.4"),
            "T8.1.1": _board_task_status(board_text, "T8.1.1"),
            "T8.2.1": _board_task_status(board_text, "T8.2.1"),
            "T6.9.2": _board_task_status(board_text, "T6.9.2"),
        },
    }


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    manuscript = report["manuscript"]
    checks = manuscript["checks"]
    parent = report["parent_state"]
    rows = report["discussion_rows"]
    row_states = {row["discussion_state"] for row in rows}
    source_live = all(_binding_live(binding) for binding in report["source_bindings"].values()) if check_live_sources else bool(report["source_integrity_declared"])
    expected_verdicts = {
        "claim_matrix": "PASS_RESTRICTED_PREBOARD_CLAIM_BOUNDARY_MATRIX",
        "introduction": "PASS_EVIDENCE_BOUNDED_INTRODUCTION_RELATED_WORK",
        "methods": "PASS_EVIDENCE_STATE_BOUNDED_METHODS",
        "results": "PASS_RESULTS_COMPLETE_NEGATIVE_AND_SECONDARY_BOUNDARIES",
        "v4": "NO_GO_FULL_HIGH_LEVEL_PAPER_RESTRICTED_PREBOARD_DRAFT_ONLY",
        "board": "BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL",
        "v5": "NO_GO_V5_EARLY_HEADROOM_STOP",
        "phase6c": "PASS_AUX_COMPARISON_INTEGRITY",
    }
    gates = {
        "G01_identity": report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION,
        "G02_live_source_bindings": source_live,
        "G03_section_order": bool(checks["section_order"]),
        "G04_discussion_subsection_order": bool(checks["required_subsections"] and checks["ordered_subsections"]),
        "G05_restricted_contribution_scope": bool(checks["restricted_contribution"]),
        "G06_regime_awareness_interpretation": bool(checks["regime_interpretation"]),
        "G07_safety_performance_cost_separation": bool(checks["safety_and_cost"]),
        "G08_compute_hardware_cost_inventory": bool(checks["cost_inventory"]),
        "G09_no_onboard_training_claim": bool(checks["explicit_nonclaims"]),
        "G10_external_model_validity": bool(checks["external_validity"]),
        "G11_no_physical_apparatus_or_break_even": bool(checks["explicit_nonclaims"] and checks["external_validity"]),
        "G12_no_outer_code_threshold_upgrade": bool(checks["external_validity"]),
        "G13_task_signature_comparison_boundary": bool(checks["comparison_boundaries"]),
        "G14_failure_informed_priorities": bool(checks["failure_informed_priorities"]),
        "G15_monotonic_physical_transition": bool(checks["transition_ladder"]),
        "G16_physical_break_even_gate": bool(checks["transition_ladder"] and checks["explicit_nonclaims"]),
        "G17_balanced_conclusion": bool(checks["balanced_conclusion"]),
        "G18_board_null_and_blocked": parent["board"]["measured_field_count"] == 42 and parent["board"]["nonnull_field_count"] == 0 and len(parent["board"]["false_external_prerequisites"]) == 6 and parent["task_status"]["T6.9.2"] == "Blocked",
        "G19_v5_stop_and_secondary_nonpromotion": parent["v5"] == {"dropped_tasks": 20, "downstream_outputs": 0, "formal_artifacts_exist": False} and "DIAGNOSTIC_STOP" in parent["result_states"],
        "G20_previous_contracts_and_verdicts": parent["verdicts"] == expected_verdicts and all(parent["previous_contracts_live"].values()),
        "G21_discussion_rows_complete": len(rows) == 27 and row_states == set(DISCUSSION_STATES) and len({row["row_id"] for row in rows}) == len(rows),
        "G22_prohibited_assertions_absent": bool(checks["prohibited_assertions_absent"]),
    }
    return gates


def _semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    check_map = {
        "G03_section_order": "section_order",
        "G04_discussion_subsection_order": "required_subsections",
        "G05_restricted_contribution_scope": "restricted_contribution",
        "G06_regime_awareness_interpretation": "regime_interpretation",
        "G07_safety_performance_cost_separation": "safety_and_cost",
        "G08_compute_hardware_cost_inventory": "cost_inventory",
        "G09_no_onboard_training_claim": "explicit_nonclaims",
        "G10_external_model_validity": "external_validity",
        "G11_no_physical_apparatus_or_break_even": "explicit_nonclaims",
        "G12_no_outer_code_threshold_upgrade": "external_validity",
        "G13_task_signature_comparison_boundary": "comparison_boundaries",
        "G14_failure_informed_priorities": "failure_informed_priorities",
        "G15_monotonic_physical_transition": "transition_ladder",
        "G16_physical_break_even_gate": "transition_ladder",
        "G17_balanced_conclusion": "balanced_conclusion",
        "G22_prohibited_assertions_absent": "prohibited_assertions_absent",
    }
    for index, target in enumerate(evaluate_gates(report)):
        mutated = copy.deepcopy(report)
        if target == "G01_identity":
            mutated["task_id"] = "T7.2.X"
        elif target == "G02_live_source_bindings":
            mutated["source_integrity_declared"] = False
        elif target in check_map:
            mutated["manuscript"]["checks"][check_map[target]] = False
        elif target == "G18_board_null_and_blocked":
            mutated["parent_state"]["board"]["nonnull_field_count"] = 1
        elif target == "G19_v5_stop_and_secondary_nonpromotion":
            mutated["parent_state"]["v5"]["downstream_outputs"] = 1
        elif target == "G20_previous_contracts_and_verdicts":
            mutated["parent_state"]["previous_contracts_live"]["results"] = False
        elif target == "G21_discussion_rows_complete":
            mutated["discussion_rows"] = mutated["discussion_rows"][:-1]
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
        "discussion_rows": report["discussion_rows"],
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
        "discussion_rows": _discussion_rows(),
        "parent_state": _parent_state(),
        "source_bindings": {name: _binding(path) for name, path in SOURCE_PATHS.items()},
        "source_integrity_declared": True,
    }
    report["gates"] = evaluate_gates(report, check_live_sources=True)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_DISCUSSION_CONCLUSION_CONTRACT"
    report["semantic_mutation_audit"] = _semantic_mutation_audit(report)
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def _write_source_data(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = ("row_id", "topic", "discussion_state", "source_ids", "boundary")
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
        "# T7.2.4 Discussion/Conclusion 证据合同",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- gates：`{report['gate_summary']['passed']}/{report['gate_summary']['total']}`",
        f"- semantic mutations：`{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`",
        f"- discussion rows：`{len(report['discussion_rows'])}`",
        f"- board measured fields：`{parent['board']['nonnull_field_count']}/{parent['board']['measured_field_count']}` non-null",
        f"- V5：`{parent['v5']['dropped_tasks']}` dropped，`{parent['v5']['downstream_outputs']}` downstream outputs",
        "",
        "## 结果状态",
        "",
        "| ID | 主题 | 状态 | 边界 |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['row_id']} | {row['topic']} | `{row['discussion_state']}` | {row['boundary']} |"
        for row in report["discussion_rows"]
    )
    lines.extend([
        "",
        "本合同只允许 restricted simulator/pre-board 结论。真实 cavity/transmon、物理 lifetime/beyond-break-even、板上训练、板测 speed/power 与闭环 QPU 均必须经独立 future gate。",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any]) -> None:
    _write_source_data(report["discussion_rows"], DEFAULT_SOURCE_DATA)
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
