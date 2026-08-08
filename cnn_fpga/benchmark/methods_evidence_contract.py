"""T7.2.2 evidence-state contract for the manuscript Methods section.

The contract prevents three common prose failures: treating simulator truth as
an online input, relabelling the implemented V4 HMM/Window/EWMA system as the
stopped V5 IMM/BOCPD design, and promoting pre-board CXXRTL/P&R evidence to a
physical-board measurement.
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
TASK_ID = "T7.2.2"
SCHEMA_VERSION = "t7.2.2-methods-evidence-contract-v1"
VERDICT = "PASS_EVIDENCE_STATE_BOUNDED_METHODS"

NOTE_PATH = ROOT / "docs/paper_notes/Contract_Centric_Regime_Aware_GKP_note_draft.tex"
DEFAULT_REPORT = ROOT / "docs/t7_2_2_methods_evidence_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs/t7_2_2_methods_evidence_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs/methods_evidence_contract.md"

SOURCE_PATHS = {
    "manuscript": NOTE_PATH,
    "protocol_hierarchy": ROOT / "docs/protocol_hierarchy.json",
    "syndrome_stream": ROOT / "physics/syndrome_stream.py",
    "fast_monte_carlo": ROOT / "docs/t2_1_3_fast_monte_carlo.json",
    "unified_contract": ROOT / "docs/t6_5_2_unified_execution_contract.json",
    "posterior_lock": ROOT / "docs/t6_6_3_route_a_posterior_threshold_lock.json",
    "promotion_gate": ROOT / "docs/t6_7_4_route_a_promotion_gate.json",
    "integrated_rtl": ROOT / "docs/t6_7_3_route_a_integrated_rtl_qualification.json",
    "hardware_pareto": ROOT / "docs/t6_9_1_route_a_hardware_pareto.json",
    "causal_headroom": ROOT / "docs/t6_10_1_causal_headroom.json",
    "v5_final_gate": ROOT / "docs/t6_15_5_route_a_v5_final_evidence_gate.json",
    "board_blocker": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
    "task_board": ROOT / "docs/new_task_board.md",
    "implementation": Path(__file__).resolve(),
}

METHOD_STATES = (
    "IMPLEMENTED_EVALUATED",
    "DIAGNOSTIC_ONLY_EXECUTED",
    "CONDITIONALLY_REGISTERED_STOPPED",
    "FUTURE_PHYSICAL_WORK_BLOCKED",
)

REQUIRED_SUBSECTIONS = (
    "Method-state convention",
    "Protocol-aligned simulation stack",
    "System overview and reuse of the dual-loop pipeline",
    "Task formulation",
    "Unified execution contract",
    "Safety state and atomic updates",
    "Matched baselines, oracle, and replaceable learned modules",
    "Evaluation protocol and metrics",
    "Executed V5 entry diagnostic and conditional stopping branch",
    "Fixed-point, RTL, formal, and post-route evidence boundary",
    "Future physical-board procedure",
)

REQUIRED_SUBSUBSECTIONS = (
    "V4 result-blind formal design",
    "Metrics and offline scoring",
    "Evidence grades and stopping rules",
)

PROHIBITED_ASSERTIVE_PATTERNS = (
    "we implemented imm",
    "we implement imm",
    "we implemented bocpd",
    "we implement bocpd",
    "the v5 four-way split was created",
    "v5 formal results demonstrate",
    "v5 cxxrtl qualification passed",
    "v5 formal verification passed",
    "v5 p&r passed",
    "board-measured 222.222 ns",
    "zero deadline miss was measured",
    "measured faster than existing fpga",
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


def _method_rows() -> list[dict[str, str]]:
    rows = [
        ("MTH-001", "protocol_aligned_simulator", "Protocol-aligned syndrome/effective simulator", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "OFFLINE_TRUTH_ONLY_SCORING", "T2.1.1/T2.1.3/T2.2.1/T2.2.2", "Not a calibrated cavity--transmon digital twin"),
        ("MTH-002", "observation_truth_split", "Observed packet and isolated truth record", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "OFFLINE_TRUTH_ONLY_SCORING", "T2.1.1/T6.5.2", "Truth cannot enter an adapter, posterior, image, or action"),
        ("MTH-003", "unified_execution_contract", "Packet/LUT/budget/deadline contract", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "NONE", "T6.5.2/T6.6.1", "A common cap is not equal measured cost"),
        ("MTH-004", "v4_hmm", "Four-state causal HMM and heavy-tail event head", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "OFFLINE_TRUTH_ONLY_SCORING", "T6.6.3", "Must not be renamed IMM, BOCPD, or activation prediction"),
        ("MTH-005", "v4_window_ewma", "Window/EWMA dual-shadow estimators", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "OFFLINE_TRUTH_ONLY_SCORING", "T6.6.2/T6.6.3/T6.7", "Tail behavior establishes EWMA-relative non-inferiority, not improvement"),
        ("MTH-006", "v4_typed_bank", "A/B commit, event/reset, and LKG policy", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "NONE", "T6.6.2/T6.7.3", "V4 transactions are not V5 typed-policy residency evidence"),
        ("MTH-007", "matched_baselines", "Standard/static/Window/EWMA/Kalman comparators", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "OFFLINE_TRUTH_ONLY_SCORING", "T6.6.1/T6.7.1/T6.7.2", "Static joint MAP remains software-only for current RTL"),
        ("MTH-008", "hidden_oracle", "Hidden-state MAP oracle", "IMPLEMENTED_EVALUATED", "NONE", "OFFLINE_TRUTH_ONLY_SCORING", "T6.5.2/T6.7.1", "Nondeployable upper bound; excluded from matched cost/rank"),
        ("MTH-009", "v4_statistics", "Calibration/pilot/formal lock and clustered inference", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "OFFLINE_TRUTH_ONLY_SCORING", "T6.5.3/T6.6.3/T6.7.1/T6.7.2", "Prior-informed V1--V4 history must remain disclosed"),
        ("MTH-010", "v4_integer_cxxrtl", "Integer golden and million-cycle CXXRTL", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "NONE", "T6.2.2/T6.7.3", "Sampled equivalence and mutation coverage are not exhaustive formal proof"),
        ("MTH-011", "v4_post_route", "Two-profile, three-seed open-source P&R", "IMPLEMENTED_EVALUATED", "ONLINE_OBSERVED_ONLY", "NONE", "T6.9.1", "Estimate only; no vendor signoff, transport, or board timing"),
        ("MTH-012", "v5_headroom", "Causal/action-space headroom audit", "DIAGNOSTIC_ONLY_EXECUTED", "ONLINE_OBSERVED_ONLY", "OFFLINE_TRUTH_ONLY_SCORING", "T6.10.1/T6.15.5", "Negative entry audit is not a V5 formal performance result"),
        ("MTH-013", "v5_four_split", "Fresh train/calibration/pilot/formal split", "CONDITIONALLY_REGISTERED_STOPPED", "NONE_NOT_RUN", "NONE_NOT_RUN", "T6.10.3", "No split manifest, power plan, or untouched V5 formal data exists"),
        ("MTH-014", "v5_posterior", "Multiscale features, IMM/BOCPD, activation prediction", "CONDITIONALLY_REGISTERED_STOPPED", "NONE_NOT_RUN", "NONE_NOT_RUN", "T6.11.1--T6.11.4", "Dropped before implementation or calibration"),
        ("MTH-015", "v5_map_risk", "Posterior-predictive MAP and LER/CVaR risk calibration", "CONDITIONALLY_REGISTERED_STOPPED", "NONE_NOT_RUN", "NONE_NOT_RUN", "T6.12.1--T6.12.3", "No compiler image, risk threshold, or quantized action exists"),
        ("MTH-016", "v5_typed_policy", "Typed expert/two-bank V5 policy", "CONDITIONALLY_REGISTERED_STOPPED", "NONE_NOT_RUN", "NONE_NOT_RUN", "T6.12.4", "No V5 resident-bank transaction or event action was executed"),
        ("MTH-017", "v5_qualification", "V5 fixed-point/formal/CXXRTL/P&R", "CONDITIONALLY_REGISTERED_STOPPED", "NONE_NOT_RUN", "NONE_NOT_RUN", "T6.15.1--T6.15.4", "No V5 golden, proof, trace, netlist, resource, or timing result"),
        ("MTH-018", "physical_board", "Physical board/HIL measurement", "FUTURE_PHYSICAL_WORK_BLOCKED", "PHYSICAL_INPUT_PENDING", "PHYSICAL_SCORING_PENDING", "T6.9.2", "All 42 measured fields remain null"),
    ]
    return [
        {
            "row_id": row_id,
            "component_id": component_id,
            "component": component,
            "evidence_state": evidence_state,
            "online_privilege": online_privilege,
            "offline_truth_role": offline_truth_role,
            "source_ids": source_ids,
            "boundary": boundary,
        }
        for row_id, component_id, component, evidence_state, online_privilege, offline_truth_role, source_ids, boundary in rows
    ]


def _parent_state() -> dict[str, Any]:
    unified = _load_json(SOURCE_PATHS["unified_contract"])
    posterior = _load_json(SOURCE_PATHS["posterior_lock"])
    promotion = _load_json(SOURCE_PATHS["promotion_gate"])
    rtl = _load_json(SOURCE_PATHS["integrated_rtl"])
    hardware = _load_json(SOURCE_PATHS["hardware_pareto"])
    headroom = _load_json(SOURCE_PATHS["causal_headroom"])
    v5 = _load_json(SOURCE_PATHS["v5_final_gate"])
    board = _load_json(SOURCE_PATHS["board_blocker"])
    nested = headroom["development_audit"]["nested_audit"]
    recomputed = v5["headroom_recomputation"]
    measured = board["measured_results"]
    profiles = hardware["profiles"]
    return {
        "verdicts": {
            "unified_contract": unified["verdict"],
            "posterior_lock": posterior["verdict"],
            "v4_promotion": promotion["verdict"],
            "integrated_rtl": rtl["verdict"],
            "hardware_pareto": hardware["verdict"],
            "v5_headroom": headroom["verdict"],
            "v5_final": v5["verdict"],
            "board": board["verdict"],
        },
        "posterior_class_order": posterior["posterior_class_order"],
        "v4_formal_accessed_at_lock": bool(posterior["formal_evaluation_accessed"]),
        "v4_rtl_cycles": int(rtl["aggregate_python"]["cycles"]),
        "pr_profile_count": len(profiles),
        "pr_seed_counts": [len(profile["place_route"]) for profile in profiles],
        "v5_development_trajectories": int(headroom["development_audit"]["trajectory_count"]),
        "v5_development_decisions": int(nested["total_decisions"]),
        "strict_causal_router_headroom": float(recomputed["strict_causal_router_headroom"]),
        "incremental_action_space_headroom": float(recomputed["incremental_action_space_headroom"]),
        "v5_dropped_task_count": len(v5["dropped_tasks"]),
        "v5_downstream_output_count": len(v5["v5_downstream_outputs_found"]),
        "v5_formal_manifest_exists": bool(v5["formal_access"]["v5_formal_manifest_exists"]),
        "v5_formal_output_exists": bool(v5["formal_access"]["v5_formal_output_exists"]),
        "board_measured_field_count": len(measured),
        "board_measured_nonnull_count": sum(value is not None for value in measured.values()),
    }


def _manuscript_snapshot() -> dict[str, Any]:
    tex = NOTE_PATH.read_text(encoding="utf-8")
    methods = _extract_section(tex, "Contract-centric dual-loop method")
    normalized = _normalize(methods)
    sections = _headings(tex, "section")
    subsections = _headings(methods, "subsection")
    subsubsections = _headings(methods, "subsubsection")
    state_phrases = {
        "IMPLEMENTED_EVALUATED": "implemented and evaluated",
        "DIAGNOSTIC_ONLY_EXECUTED": "diagnostic only",
        "CONDITIONALLY_REGISTERED_STOPPED": "conditionally registered and stopped",
        "FUTURE_PHYSICAL_WORK_BLOCKED": "future physical work",
    }
    checks = {
        "section_order": sections.index("Contract-centric dual-loop method") < sections.index("Results"),
        "required_subsections": all(title in subsections for title in REQUIRED_SUBSECTIONS) and all(title in subsubsections for title in REQUIRED_SUBSUBSECTIONS),
        "four_method_states": all(phrase in normalized for phrase in state_phrases.values()),
        "simulator_scope": all(token in normalized for token in ("single-mode square-lattice gkp abstraction", "not a calibrated cavity--transmon digital twin", "finite-squeezing layer", "trajectory-clustered monte carlo")),
        "privilege_separation": all(token in normalized for token in ("physically distinct records", "truth is used only", "absent from every online adapter", "offline scores")),
        "unified_contract": all(token in normalized for token in ("10-bit adc", "signed q9.12", "ii=1", "8,192 update macs", "board-measured deadline miss")),
        "v4_hmm_and_policy": all(token in normalized for token in ("four-state causal hmm", "neither an imm nor bocpd", "1,728 pilot tuples", "continuously updated and validated ewma shadow", "monotonic lkg")),
        "matched_baselines": all(token in normalized for token in ("standard binning", "static joint map", "window map", "kalman adaptive map", "hidden-state oracle", "software-only strong baseline")),
        "v4_three_split_statistics": all(token in normalized for token in ("calibration, pilot, and formal", "v4 did not use", "20,000 paired bootstrap", "holm correction")),
        "v5_diagnostic": all(token in normalized for token in ("186 trajectories", "4,571,136 decisions", "-0.2322", "0.02549", "nine avoided errors")),
        "v5_four_split_absent": all(token in normalized for token in ("train/calibration/pilot/formal four-way split", "no four-split manifest", "not implemented modules")),
        "v5_components_stopped": all(token in normalized for token in ("multiscale wrapped features", "static/trend/harmonic imm", "bocpd/telegraph", "activation-horizon prediction", "posterior-predictive map", "ler/cvar", "dropped before implementation")),
        "v4_fixed_point_cxxrtl": all(token in normalized for token in ("independent packed-word integer reference", "ten 100,000-cycle families", "sampled long-sequence equivalence", "not exhaustive formal verification")),
        "formal_absence": all(token in normalized for token in ("actual-parameterized v5 sva/smt proof", "were all stopped", "explicit absent branch")),
        "pr_estimate_boundary": all(token in normalized for token in ("seeds 1, 7, and 19", "estimate", "not vendor signoff or board timing")),
        "board_future_boundary": all(token in normalized for token in ("pre-board candidate only", "all 42 board-measured fields remain null", "streaming or autonomous trace", "faster-than-existing-fpga")),
        "no_assertive_overclaim": not any(pattern in normalized for pattern in PROHIBITED_ASSERTIVE_PATTERNS),
    }
    return {
        "section_title": "Contract-centric dual-loop method",
        "section_order": sections,
        "subsections": subsections,
        "subsubsections": subsubsections,
        "characters": len(methods),
        "sha256": hashlib.sha256(methods.encode("utf-8")).hexdigest(),
        "state_phrases": state_phrases,
        "checks": checks,
        "prohibited_hits": [pattern for pattern in PROHIBITED_ASSERTIVE_PATTERNS if pattern in normalized],
    }


EXPECTED_VERDICTS = {
    "unified_contract": "PASS_UNIFIED_EXECUTION_CONTRACT_FROZEN",
    "posterior_lock": "PASS_ROUTE_A_POSTERIOR_AND_COMMON_THRESHOLD_LOCK",
    "v4_promotion": "GO_ROUTE_A_CONTRACT_SYSTEM_RESTRICTED_SIMULATOR_AND_PREBOARD_CLAIMS",
    "integrated_rtl": "PASS_ROUTE_A_INTEGRATED_LONG_RTL_QUALIFICATION",
    "hardware_pareto": "PASS_ROUTE_A_INTEGRATED_THREE_SEED_PR_ESTIMATE_NOT_BOARD_MEASURED",
    "v5_headroom": "NO_GO_V5_INSUFFICIENT_ACTION_SPACE_HEADROOM",
    "v5_final": "NO_GO_V5_EARLY_HEADROOM_STOP",
    "board": "BLOCKED_T6_9_2_NO_PHYSICAL_BOARD_EVIDENCE_ALL_MEASURED_FIELDS_NULL",
}


def evaluate_gates(report: Mapping[str, Any], *, check_live_sources: bool = False) -> dict[str, bool]:
    manuscript = report["manuscript"]
    checks = manuscript["checks"]
    parent = report["parent_state"]
    rows = report["method_rows"]
    states = {row["evidence_state"] for row in rows}
    parent_ok = (
        parent["verdicts"] == EXPECTED_VERDICTS
        and parent["posterior_class_order"] == ["normal", "smooth", "calibration_shift", "burst"]
        and parent["v4_formal_accessed_at_lock"] is False
        and parent["v4_rtl_cycles"] == 1_000_000
        and parent["pr_profile_count"] == 2
        and parent["pr_seed_counts"] == [3, 3]
        and parent["v5_development_trajectories"] == 186
        and parent["v5_development_decisions"] == 4_571_136
        and parent["strict_causal_router_headroom"] < 0.0
        and 0.0 <= parent["incremental_action_space_headroom"] < 0.001
        and parent["v5_dropped_task_count"] == 20
        and parent["v5_downstream_output_count"] == 0
        and parent["v5_formal_manifest_exists"] is False
        and parent["v5_formal_output_exists"] is False
        and parent["board_measured_field_count"] == 42
        and parent["board_measured_nonnull_count"] == 0
    )
    source_ok = bool(report["source_integrity_declared"])
    if check_live_sources:
        source_ok = source_ok and all(_binding_live(binding) for binding in report["source_bindings"].values())
    return {
        "G01_identity": report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION,
        "G02_method_section_before_results": bool(checks["section_order"]),
        "G03_required_method_subsections": bool(checks["required_subsections"]),
        "G04_four_evidence_states": bool(checks["four_method_states"]) and states == set(METHOD_STATES),
        "G05_protocol_aligned_simulator_scope": bool(checks["simulator_scope"]),
        "G06_online_observed_offline_truth_separation": bool(checks["privilege_separation"]),
        "G07_unified_execution_contract": bool(checks["unified_contract"]),
        "G08_v4_hmm_typed_policy_accuracy": bool(checks["v4_hmm_and_policy"]),
        "G09_matched_baseline_oracle_boundary": bool(checks["matched_baselines"]),
        "G10_v4_three_split_and_statistics": bool(checks["v4_three_split_statistics"]),
        "G11_v5_entry_diagnostic_accuracy": bool(checks["v5_diagnostic"]),
        "G12_v5_four_split_not_instantiated": bool(checks["v5_four_split_absent"]),
        "G13_v5_components_explicitly_stopped": bool(checks["v5_components_stopped"]),
        "G14_v4_fixed_point_cxxrtl_boundary": bool(checks["v4_fixed_point_cxxrtl"]),
        "G15_formal_proof_absence_visible": bool(checks["formal_absence"]),
        "G16_post_route_estimate_boundary": bool(checks["pr_estimate_boundary"]),
        "G17_board_future_work_and_42_null": bool(checks["board_future_boundary"]) and parent["board_measured_nonnull_count"] == 0,
        "G18_parent_and_source_integrity": parent_ok and source_ok and bool(checks["no_assertive_overclaim"]),
    }


def _semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    targets = list(evaluate_gates(report))
    cases: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        mutated = copy.deepcopy(report)
        if target == "G01_identity":
            mutated["task_id"] = "T7.2.X"
        elif target == "G04_four_evidence_states":
            mutated["method_rows"][0]["evidence_state"] = "UNDECLARED"
        elif target == "G17_board_future_work_and_42_null":
            mutated["parent_state"]["board_measured_nonnull_count"] = 1
        elif target == "G18_parent_and_source_integrity":
            mutated["source_integrity_declared"] = False
        else:
            check_key = {
                "G02_method_section_before_results": "section_order",
                "G03_required_method_subsections": "required_subsections",
                "G05_protocol_aligned_simulator_scope": "simulator_scope",
                "G06_online_observed_offline_truth_separation": "privilege_separation",
                "G07_unified_execution_contract": "unified_contract",
                "G08_v4_hmm_typed_policy_accuracy": "v4_hmm_and_policy",
                "G09_matched_baseline_oracle_boundary": "matched_baselines",
                "G10_v4_three_split_and_statistics": "v4_three_split_statistics",
                "G11_v5_entry_diagnostic_accuracy": "v5_diagnostic",
                "G12_v5_four_split_not_instantiated": "v5_four_split_absent",
                "G13_v5_components_explicitly_stopped": "v5_components_stopped",
                "G14_v4_fixed_point_cxxrtl_boundary": "v4_fixed_point_cxxrtl",
                "G15_formal_proof_absence_visible": "formal_absence",
                "G16_post_route_estimate_boundary": "pr_estimate_boundary",
            }[target]
            mutated["manuscript"]["checks"][check_key] = False
        rejected = not evaluate_gates(mutated)[target]
        cases.append({"mutation_id": f"M{index + 1:02d}", "target_gate": target, "rejected": rejected})
    return {"count": len(cases), "detected": sum(case["rejected"] for case in cases), "cases": cases}


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "task_id", "schema_version", "manuscript", "method_rows", "parent_state",
            "source_bindings", "source_integrity_declared", "gates", "verdict",
        )
    }


def build_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manuscript": _manuscript_snapshot(),
        "method_rows": _method_rows(),
        "parent_state": _parent_state(),
        "source_bindings": {name: _binding(path) for name, path in SOURCE_PATHS.items()},
        "source_integrity_declared": True,
    }
    report["gates"] = evaluate_gates(report, check_live_sources=True)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "total": len(report["gates"])}
    report["verdict"] = VERDICT if all(report["gates"].values()) else "FAIL_METHODS_EVIDENCE_CONTRACT"
    report["semantic_mutation_audit"] = _semantic_mutation_audit(report)
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def _write_source_data(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    fieldnames = [
        "row_id", "component_id", "component", "evidence_state", "online_privilege",
        "offline_truth_role", "source_ids", "boundary",
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
        "# T7.2.2 Methods evidence-state contract",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- gates：`{report['gate_summary']['passed']}/{report['gate_summary']['total']}`",
        f"- semantic mutations：`{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`",
        f"- method rows：`{len(report['method_rows'])}`",
        f"- V5：`{parent['v5_dropped_task_count']}` dropped tasks，`{parent['v5_downstream_output_count']}` downstream outputs",
        f"- board：`{parent['board_measured_field_count']}` measured fields，nonnull=`{parent['board_measured_nonnull_count']}`",
        "",
        "| component | state | online | offline truth | boundary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["method_rows"]:
        lines.append(
            f"| `{row['component_id']}` | `{row['evidence_state']}` | `{row['online_privilege']}` | "
            f"`{row['offline_truth_role']}` | {row['boundary']} |"
        )
    lines.extend([
        "",
        "V4 的 simulator、HMM、Window/EWMA、bank、统计、integer/CXXRTL 与 P&R estimate 均按各自证据层描述。",
        "V5 只有 causal/action-space entry diagnostic 被执行；四分割、IMM/BOCPD、activation prediction、",
        "posterior-predictive compiler、LER/CVaR gate、typed V5 bank、formal/CXXRTL/P&R 均在入口 NO-GO 后停止。",
        "真实板卡流程是 future work，不能由 UART candidate、clock model 或 P&R 数值替代。",
        "",
    ])
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any]) -> None:
    _write_source_data(report["method_rows"], DEFAULT_SOURCE_DATA)
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
        "mutations": report["semantic_mutation_audit"],
        "analysis_sha256": report["analysis_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == VERDICT else 1


if __name__ == "__main__":
    raise SystemExit(main())
