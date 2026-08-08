"""Freeze and verify the Phase-9 three-lane performance protocol.

This module is a protocol compiler, not an experiment runner.  It keeps
round-LER, six-state logical lifetime, and raw-IQ digital HIL latency in three
independent task signatures.  The protocol seal may pass while every
performance result remains null.  Future opened outcomes are evaluated by
lane-local GO/NO-GO rules; incomplete evidence never becomes a negative result.
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
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T9.1.1"
SCHEMA_VERSION = "t9.1.1-phase9-three-lane-protocol-report-v1"
CONFIG_SCHEMA_VERSION = "t9.1.1-phase9-three-lane-protocol-config-v1"
PROTOCOL_ID = "PHASE9-THREE-INDEPENDENT-TASK-SIGNATURES-V1"
VERDICT = "PASS_PHASE9_THREE_INDEPENDENT_LANE_PROTOCOL_FROZEN"

DEFAULT_CONFIG = ROOT / "configs" / "phase9" / "t9_1_1_three_lane_protocol.json"
DEFAULT_REPORT = ROOT / "docs" / "t9_1_1_three_lane_protocol.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t9_1_1_three_lane_protocol_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "phase9_three_lane_protocol.md"

LANE_IDS = (
    "ROUND_LER_SINGLE_MODE",
    "SIX_STATE_LOGICAL_LIFETIME",
    "RAW_IQ_DIGITAL_HIL",
)
TASK_SIGNATURE_FIELDS = (
    "code_family",
    "state_family",
    "physical_backend_contract",
    "decision_target",
    "observation_schema",
    "history_horizon",
    "action_set",
    "action_cost_contract",
    "noise_drift_family",
    "observability",
    "online_privilege",
    "cycle_time_contract",
    "primary_estimand",
    "denominator_contract",
    "postselection_policy",
    "baseline_eligibility_contract",
    "compute_budget_contract",
    "wall_clock_budget_contract",
    "precision_contract",
    "split_contract",
    "statistical_unit",
    "multiplicity_contract",
    "missingness_contract",
    "evidence_grade_contract",
)
BASELINE_CLASS_IDS = {
    "MATCHED_DEPLOYABLE_RANKED",
    "CAPACITY_CEILING_NONRANKING",
    "PRIVILEGED_UPPER_BOUND_NONRANKING",
    "PROTOCOL_ANCHOR_NONRANKING",
}
EVIDENCE_GRADE_IDS = {
    "PROTOCOL_ONLY",
    "LITERATURE_ONLY",
    "OFFICIAL_SOURCE_PINNED",
    "OFFICIAL_EXACT_REPRODUCTION",
    "PAPER_CONSTRAINED_REIMPLEMENTATION",
    "PROJECT_NATIVE_DEVELOPMENT_SIMULATION",
    "UNTOUCHED_FORMAL_SIMULATION",
    "INDEPENDENT_BACKEND_REEVALUATION",
    "FIXED_POINT_REFERENCE",
    "CXXRTL_PREBOARD",
    "RTL_PROPERTY_PROOF",
    "POST_ROUTE_ESTIMATE",
    "RECORDED_IQ_HIL_MEASURED",
    "RAW_IQ_HIL_MEASURED",
    "QPU_MEASURED",
}
FORBIDDEN_TRANSFER_IDS = {
    "FT-LIFETIME-TO-LER",
    "FT-LER-TO-LIFETIME",
    "FT-CORE-TO-RAW-IQ",
    "FT-PREBOARD-TO-MEASURED",
    "FT-SIM-TO-PHYSICAL-BREAK-EVEN",
    "FT-PAPER-CONSTRAINED-TO-OFFICIAL",
    "FT-HIDDEN-TO-DEPLOYABLE",
    "FT-CEILING-TO-RANKED",
    "FT-CROSS-LANE-SCORE",
    "FT-POSTSELECTED-TO-FULL",
    "FT-SAFETY-TO-PERFORMANCE",
    "FT-CROSS-CODE-LATENCY",
    "FT-MISSING-AS-ZERO",
}
GATE_IDS = (
    "G01_identity_and_preoutcome_seal",
    "G02_exactly_three_independent_namespaces",
    "G03_signature_schema_has_24_frozen_fields",
    "G04_signatures_are_complete_nonempty_and_distinct",
    "G05_ler_code_state_metrics_are_six_state_single_mode",
    "G06_ler_observation_action_and_denominator_are_causal_full",
    "G07_lifetime_metrics_horizon_and_six_state_aggregation_are_complete",
    "G08_lifetime_inherits_ler_physics_observation_action_and_cost",
    "G09_algorithm_lanes_prohibit_postselection_and_accepted_only_denominators",
    "G10_hil_has_four_boundaries_and_raw_iq_primary",
    "G11_hil_statistics_denominator_and_hardware_cost_are_complete",
    "G12_baseline_classes_keep_only_matched_deployable_ranked",
    "G13_matched_baseline_predicate_is_exact_and_fail_closed",
    "G14_split_is_single_pass_pilot_then_untouched_formal",
    "G15_observed_only_contract_rejects_future_truth_and_scenario_privilege",
    "G16_compute_precision_wallclock_and_deadline_fields_are_nonempty",
    "G17_multiplicity_is_cluster_level_simultaneous_and_closed_family",
    "G18_missingness_retains_failures_and_never_imputes_null_as_zero",
    "G19_evidence_grades_are_scope_sets_not_a_global_rank",
    "G20_puviani_official_and_surpass_slots_remain_local_nulls",
    "G21_physical_break_even_and_raw_iq_speed_remain_null_without_grade",
    "G22_ler_gate_freezes_each_baseline_and_tail_safety_thresholds",
    "G23_lifetime_gate_freezes_six_state_gain_cost_and_horizon",
    "G24_hil_gate_requires_board_chain_three_seeds_million_transactions_and_comparator",
    "G25_future_evaluator_fixtures_cover_go_no_go_incomplete_and_unopened",
    "G26_claim_ladders_have_wording_grades_and_revocation",
    "G27_forbidden_transfer_registry_is_complete",
    "G28_global_score_winner_count_and_cross_lane_rescue_are_prohibited",
    "G29_source_contracts_are_semantically_or_exactly_live",
    "G30_current_performance_results_are_null_not_fake_go_or_no_go",
    "G31_lifetime_does_not_promote_puviani_or_physical_claims",
    "G32_preboard_rtl_does_not_promote_measured_hil_or_speed",
    "G33_independent_backends_each_pass_without_averaging",
    "G34_result_state_machine_is_null_incomplete_then_binary_complete",
    "G35_source_data_and_human_contract_are_lossless_and_live",
    "G36_one_substantive_mutation_per_gate_fails_closed",
)

PLAN_SECTIONS = ("20.1", "20.2", "20.6", "20.7")
BOARD_TASK_IDS = (
    "T9.1.1", "T9.1.2", "T9.1.3", "T9.1.4", "T9.6.1", "T9.6.5",
    "T9.7.2", "T9.7.4", "T9.8.1", "T9.8.2",
)
RISK_IDS = ("R-N162", "R-N167", "R-N168")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _strict_binding(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "selector": "STRICT_FILE_SHA256",
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _normalise_lines(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _extract_section(text: str, number: str) -> str:
    start_match = re.search(rf"^##\s+{re.escape(number)}(?:\s|$).*", text, flags=re.MULTILINE)
    if start_match is None:
        raise ValueError(f"missing experiment-plan section {number}")
    next_match = re.search(
        r"^(?:##\s+\d+\.\d+(?:\s|$).*|\[\d+\]:\s+.*)$",
        text[start_match.end():],
        flags=re.MULTILINE,
    )
    end = start_match.end() + next_match.start() if next_match else len(text)
    return _normalise_lines(text[start_match.start():end])


def _plan_projection(text: str) -> dict[str, str]:
    return {number: _extract_section(text, number) for number in PLAN_SECTIONS}


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _board_projection(text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        cells = _table_cells(line) if line.lstrip().startswith("|") else []
        if len(cells) >= 5 and cells[0] in BOARD_TASK_IDS:
            rows[cells[0]] = {"task_id": cells[0], "task": cells[2], "source": cells[4]}
    if set(rows) != set(BOARD_TASK_IDS):
        raise ValueError("Phase-9 task projection is incomplete")
    return rows


def _risk_projection(text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        cells = _table_cells(line) if line.lstrip().startswith("|") else []
        if len(cells) >= 8 and cells[0] in RISK_IDS:
            rows[cells[0]] = {
                "risk_id": cells[0],
                "risk": cells[2],
                "evidence": cells[5],
                "references": cells[7],
            }
    if set(rows) != set(RISK_IDS):
        raise ValueError("Phase-9 risk projection is incomplete")
    return rows


SEMANTIC_SELECTORS: dict[str, Callable[[str], Any]] = {
    "EXPERIMENT_PLAN_20_1_20_2_20_6_20_7": _plan_projection,
    "PHASE9_TASK_REQUIREMENTS_EXCLUDING_STATUS_AND_OUTPUT": _board_projection,
    "PHASE9_RISK_SEMANTICS_EXCLUDING_MUTABLE_STATUS": _risk_projection,
}


def _semantic_binding(path: Path, selector: str) -> dict[str, Any]:
    payload = SEMANTIC_SELECTORS[selector](path.read_text(encoding="utf-8"))
    return {
        "path": _relative(path),
        "selector": selector,
        "payload": payload,
        "sha256": _canonical_sha256(payload),
    }


def _binding_live(binding: Mapping[str, Any]) -> bool:
    try:
        path = ROOT / str(binding["path"])
        selector = str(binding["selector"])
        if not path.is_file():
            return False
        if selector == "STRICT_FILE_SHA256":
            return path.stat().st_size == binding.get("bytes") and _sha256(path) == binding.get("sha256")
        payload = SEMANTIC_SELECTORS[selector](path.read_text(encoding="utf-8"))
        return payload == binding.get("payload") and _canonical_sha256(payload) == binding.get("sha256")
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _artifact_registry(config_path: Path) -> dict[str, dict[str, Any]]:
    return {
        "config": _strict_binding(config_path),
        "implementation": _strict_binding(Path(__file__).resolve()),
        "experiment_plan": _semantic_binding(
            ROOT / "docs" / "experiment_plan.md",
            "EXPERIMENT_PLAN_20_1_20_2_20_6_20_7",
        ),
        "task_board": _semantic_binding(
            ROOT / "docs" / "new_task_board.md",
            "PHASE9_TASK_REQUIREMENTS_EXCLUDING_STATUS_AND_OUTPUT",
        ),
        "risk_registry": _semantic_binding(
            ROOT / "docs" / "new_risks.md",
            "PHASE9_RISK_SEMANTICS_EXCLUDING_MUTABLE_STATUS",
        ),
    }


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def _lane(rows: Sequence[Mapping[str, Any]], lane_id: str) -> Mapping[str, Any]:
    return next(row for row in rows if row.get("lane_id") == lane_id)


def _claim_metadata(claim_id: str) -> dict[str, list[str]]:
    special = {
        "LER-C4-SOTA": (
            "SOTA on the frozen single-mode round-LER task after GO_LER_SOTA",
            "lifetime, tail-only, safety, or RTL evidence as an LER substitute",
        ),
        "LIFE-C4-PUVIANI-SURPASS": (
            "surpasses official Puviani NMF only after official-exact same-signature qualification",
            "paper-constrained or project-native evidence described as official Puviani surpass",
        ),
        "LIFE-C5-PHYSICAL-BREAK-EVEN": (
            "physical break-even measured on a protocol-matched QPU",
            "simulator lifetime, matched-idle crossover, or accepted-only curve as physical break-even",
        ),
        "HIL-C4-SAME-TASK-SPEED": (
            "faster raw-IQ-source-to-trigger HIL under a same-task measured comparison",
            "core cycles, CXXRTL, P&R, host timing, or cross-code nanoseconds as measured speed",
        ),
    }
    allowed, forbidden = special.get(
        claim_id,
        (f"lane-local wording for {claim_id} at its stated evidence grade", "cross-lane or unqualified evidence promotion"),
    )
    return {
        "allowed_wording": [allowed],
        "forbidden_wording": [forbidden, "weighted-score or winner-count rescue"],
        "revocation_conditions": [
            "task signature, denominator, baseline eligibility, split, evidence grade, or analysis hash changes",
            "required gate or artifact provenance becomes incomplete",
        ],
    }


def _enriched_lanes(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    lanes = copy.deepcopy(config["lanes"])
    for lane in lanes:
        lane["signature_sha256"] = _canonical_sha256(lane["signature"])
        for claim in lane["claim_ladder"]:
            claim.update(_claim_metadata(str(claim["claim_id"])))
    return lanes


def _compare(value: float | int, op: str, threshold: float | int) -> bool:
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    if op == "<":
        return value < threshold
    if op == "==":
        return value == threshold
    raise ValueError(f"unsupported threshold operator: {op}")


def evaluate_result_gate(
    lane_id: str,
    evidence: Mapping[str, Any],
    *,
    lanes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Evaluate an opened future lane without conflating incomplete and negative."""

    lane_rows = lanes if lanes is not None else _enriched_lanes(_load(DEFAULT_CONFIG))
    lane = _lane(lane_rows, lane_id)
    gate = lane["result_gate"]
    if not evidence.get("outcomes_opened", False):
        state = "MISSING_BOARD" if lane_id == "RAW_IQ_DIGITAL_HIL" and evidence.get("missing_board") else "NOT_EVALUATED_NULL"
        return {"evaluation_state": state, "result_verdict": None, "reason_code": state, "supporting_statuses": []}

    required = list(gate["required_boolean_fields"])
    missing_booleans = [key for key in required if key not in evidence]
    missing_numeric = [key for key in gate["numeric_thresholds"] if key not in evidence]
    if missing_booleans or missing_numeric:
        return {
            "evaluation_state": "INCOMPLETE",
            "result_verdict": None,
            "reason_code": "INCOMPLETE_REQUIRED_FIELDS",
            "missing_fields": sorted(missing_booleans + missing_numeric),
            "supporting_statuses": [],
        }

    failed_booleans = [key for key in required if not bool(evidence[key])]
    supporting: list[str] = []
    if lane_id == "RAW_IQ_DIGITAL_HIL" and failed_booleans == ["same_task_measured_comparator_available"]:
        safety_keys = {
            "mismatch_count", "undefined_action_count", "silent_overflow_count",
            "deadline_miss_count", "initiation_interval_cycles", "wcet_minus_deadline_ns",
        }
        if all(
            _compare(evidence[key], gate["numeric_thresholds"][key]["op"], gate["numeric_thresholds"][key]["value"])
            for key in safety_keys
        ):
            supporting.append(str(gate["engineering_only_verdict"]))
    if failed_booleans:
        reason = "INCOMPLETE_BASELINE_OR_COMPARATOR" if any("baseline" in key or "comparator" in key for key in failed_booleans) else "INCOMPLETE_PROTOCOL_OR_EVIDENCE"
        return {
            "evaluation_state": "INCOMPLETE",
            "result_verdict": None,
            "reason_code": reason,
            "failed_boolean_fields": failed_booleans,
            "supporting_statuses": supporting,
        }

    failed_thresholds = [
        key
        for key, spec in gate["numeric_thresholds"].items()
        if not _compare(evidence[key], str(spec["op"]), spec["value"])
    ]
    if failed_thresholds:
        return {
            "evaluation_state": "COMPLETE",
            "result_verdict": gate["no_go_verdict"],
            "reason_code": "COMPLETE_THRESHOLD_FAILURE",
            "failed_thresholds": failed_thresholds,
            "supporting_statuses": supporting,
        }
    return {
        "evaluation_state": "COMPLETE",
        "result_verdict": gate["go_verdict"],
        "reason_code": "ALL_REGISTERED_REQUIREMENTS_PASS",
        "supporting_statuses": supporting,
    }


def _passing_number(spec: Mapping[str, Any]) -> float | int:
    value = spec["value"]
    op = spec["op"]
    if op in {">=", ">"}:
        return value + (1 if abs(value) >= 100 else 0.01)
    if op in {"<=", "<", "=="}:
        return value
    raise ValueError(op)


def _failing_number(spec: Mapping[str, Any]) -> float | int:
    value = spec["value"]
    op = spec["op"]
    if op in {">=", ">"}:
        return value - (1 if abs(value) >= 100 else 0.01)
    if op in {"<=", "<"}:
        return value + (1 if abs(value) >= 100 else 0.01)
    if op == "==":
        return value + 1
    raise ValueError(op)


def _all_pass_evidence(lane: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: True for key in lane["result_gate"]["required_boolean_fields"]}
    result.update({key: _passing_number(spec) for key, spec in lane["result_gate"]["numeric_thresholds"].items()})
    return result


def _verdict_fixtures(lanes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in lanes:
        lane_id = str(lane["lane_id"])
        passing = _all_pass_evidence(lane)
        first_numeric = next(iter(lane["result_gate"]["numeric_thresholds"]))
        failing = copy.deepcopy(passing)
        failing[first_numeric] = _failing_number(lane["result_gate"]["numeric_thresholds"][first_numeric])
        incomplete = copy.deepcopy(passing)
        incomplete.pop(next(key for key in lane["result_gate"]["required_boolean_fields"] if key != "outcomes_opened"))
        cases = [
            ("unopened", {"outcomes_opened": False}),
            ("incomplete", incomplete),
            ("complete_no_go", failing),
            ("complete_go", passing),
        ]
        if lane_id == "RAW_IQ_DIGITAL_HIL":
            engineering = copy.deepcopy(passing)
            engineering["same_task_measured_comparator_available"] = False
            cases.append(("engineering_only_no_comparator", engineering))
        for case_id, evidence in cases:
            rows.append({
                "fixture_id": f"{lane_id}:{case_id}",
                "lane_id": lane_id,
                "evidence": evidence,
                "outcome": evaluate_result_gate(lane_id, evidence, lanes=lanes),
                "scientific_result": "SYNTHETIC_GATE_LOGIC_ONLY_NOT_EXPERIMENTAL_EVIDENCE",
            })
    return rows


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    payloads: list[tuple[str, str, Any]] = []
    for key, value in report["ontology"].items():
        payloads.append(("ontology", key, value))
    for key, value in report["shared_contracts"].items():
        payloads.append(("shared_contract", key, value))
    for key, value in report["external_claim_slots"].items():
        payloads.append(("external_claim_slot", key, value))
    for lane in report["lanes"]:
        payloads.append(("lane", lane["lane_id"], lane))
        for claim in lane["claim_ladder"]:
            payloads.append(("claim", claim["claim_id"], claim))
    for row in report["forbidden_transfers"]:
        payloads.append(("forbidden_transfer", row["transfer_id"], row))
    for row in report["verdict_fixtures"]:
        payloads.append(("verdict_fixture", row["fixture_id"], row))
    for key, value in report["artifact_registry"].items():
        payloads.append(("artifact_binding", key, value))
    rows: list[dict[str, str]] = []
    for category, item_id, payload in payloads:
        payload_json = _canonical_json(payload)
        rows.append({
            "category": category,
            "item_id": str(item_id),
            "payload_json": payload_json,
            "canonical_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
        })
    return rows


def _write_source_data(report: Mapping[str, Any], path: Path) -> int:
    rows = _source_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "item_id", "payload_json", "canonical_sha256"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return len(rows)


def _csv_lossless(report: Mapping[str, Any], path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows == _source_rows(report) and all(
            row["canonical_sha256"] == hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
            for row in rows
        )
    except (OSError, csv.Error, KeyError):
        return False


def _atomic_ids(report: Mapping[str, Any]) -> list[str]:
    ids = list(LANE_IDS) + list(GATE_IDS)
    ids += [row["grade_id"] for row in report["ontology"]["evidence_grades"]]
    ids += [row["class_id"] for row in report["ontology"]["baseline_classes"]]
    ids += list(report["external_claim_slots"])
    ids += [row["transfer_id"] for row in report["forbidden_transfers"]]
    ids += [claim["claim_id"] for lane in report["lanes"] for claim in lane["claim_ladder"]]
    ids += [row["fixture_id"] for row in report["verdict_fixtures"]]
    return ids


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 9 三条独立任务签名与结果门",
        "",
        f"- Task：`{TASK_ID}`",
        f"- 协议 verdict：`{VERDICT}`",
        "- seal：`SEALED_PRE_OUTCOME`；本文件不是性能实验结果。",
        "- 当前 LER、lifetime、HIL speed 的 `result_verdict` 均为 `null`；协议通过不得写成性能 GO。",
        "",
        "## 三条互不补门的 lane",
        "",
        "| Lane | 当前状态 | 主指标 / 主边界 | Signature SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    for lane in report["lanes"]:
        primary = ", ".join(lane["primary_metrics"])
        lines.append(
            f"| `{lane['lane_id']}` | `{lane['current_result']['evaluation_state']}` / `null` | {primary} | `{lane['signature_sha256']}` |"
        )
    lines += [
        "",
        "单轮 LER 不能由 lifetime 补门；lifetime 不能由单轮 LER 推导；HIL speed 的主边界固定为 `raw_iq_source_to_trigger`，不能用 6-cycle core、CXXRTL、P&R 或 host timing 补值。三条 lane 禁止 weighted score、winner count 或全局榜单。",
        "",
        "## 24-field task signature",
        "",
    ]
    lines.extend(f"- `{field}`" for field in TASK_SIGNATURE_FIELDS)
    for lane in report["lanes"]:
        lines += ["", f"### `{lane['lane_id']}`", ""]
        for key, value in lane["signature"].items():
            lines.append(f"- `{key}`：`{value}`")
        lines += ["", "Result gate：", ""]
        for key, spec in lane["result_gate"]["numeric_thresholds"].items():
            lines.append(f"- `{key}` {spec['op']} `{spec['value']}`")
        lines += ["", "Claim ladder：", ""]
        for claim in lane["claim_ladder"]:
            lines.append(
                f"- `{claim['claim_id']}`：`{claim['state']}`；required={','.join(claim['required_grades'])}；allowed={claim['allowed_wording'][0]}；forbidden={claim['forbidden_wording'][0]}。"
            )
    lines += ["", "## Evidence grade（集合资格，不是全局线性等级）", ""]
    for row in report["ontology"]["evidence_grades"]:
        lines.append(f"- `{row['grade_id']}`：{', '.join(row['claim_scope'])}")
    lines += ["", "## Baseline class", ""]
    for row in report["ontology"]["baseline_classes"]:
        lines.append(f"- `{row['class_id']}`：ranked=`{str(row['ranked']).lower()}`，may_support_sota=`{str(row['may_support_sota']).lower()}`")
    lines += ["", "## 外部/null claim slot", ""]
    for key, value in report["external_claim_slots"].items():
        lines.append(f"- `{key}`：state=`{value['state']}`，value=`null`")
    lines += ["", "## 禁止证据迁移", ""]
    for row in report["forbidden_transfers"]:
        lines.append(f"- `{row['transfer_id']}`：`{row['from']}` → `{row['to']}`，拒绝码 `{row['rejection']}`")
    lines += ["", "## 二值结果门的合成逻辑夹具", ""]
    for row in report["verdict_fixtures"]:
        outcome = row["outcome"]
        lines.append(
            f"- `{row['fixture_id']}`：state=`{outcome['evaluation_state']}`，verdict=`{outcome['result_verdict']}`；仅验证 gate logic，不是实验结果。"
        )
    lines += ["", "## Fail-closed gates", ""]
    lines.extend(f"- `{gate_id}`" for gate_id in GATE_IDS)
    lines += [
        "",
        "## 解释边界",
        "",
        "Puviani official asset 缺失只让 `OFFICIAL_PUVIANI_EXACT` / `PUVIANI_NMF_SURPASS` 保持 null，不阻塞 paper-constrained、数字孪生、codebook、model tournament 或 project-native formal。模拟 lifetime 无论多好都不能填 `PHYSICAL_BREAK_EVEN`；pre-board RTL 无论多确定都不能填 `RAW_IQ_HIL_SPEED`。",
        "",
    ]
    return "\n".join(lines)


def _safe(check: Callable[[], Any]) -> bool:
    try:
        return bool(check())
    except (KeyError, TypeError, ValueError, StopIteration, IndexError, OSError):
        return False


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    reference_config = _load(DEFAULT_CONFIG)
    reference_lanes = _enriched_lanes(reference_config)
    lanes = report.get("lanes", [])
    ler = _safe(lambda: _lane(lanes, "ROUND_LER_SINGLE_MODE")) and _lane(lanes, "ROUND_LER_SINGLE_MODE")
    lifetime = _safe(lambda: _lane(lanes, "SIX_STATE_LOGICAL_LIFETIME")) and _lane(lanes, "SIX_STATE_LOGICAL_LIFETIME")
    hil = _safe(lambda: _lane(lanes, "RAW_IQ_DIGITAL_HIL")) and _lane(lanes, "RAW_IQ_DIGITAL_HIL")
    ref_ler = _lane(reference_lanes, "ROUND_LER_SINGLE_MODE")
    ref_lifetime = _lane(reference_lanes, "SIX_STATE_LOGICAL_LIFETIME")
    ref_hil = _lane(reference_lanes, "RAW_IQ_DIGITAL_HIL")

    def signatures_ok() -> bool:
        hashes = []
        for lane in lanes:
            signature = lane["signature"]
            if tuple(signature) != TASK_SIGNATURE_FIELDS or not all(isinstance(value, str) and value for value in signature.values()):
                return False
            digest = _canonical_sha256(signature)
            if lane.get("signature_sha256") != digest:
                return False
            hashes.append(digest)
        return len(hashes) == 3 and len(set(hashes)) == 3

    def inheritance_ok() -> bool:
        if lifetime["inherits_from_lane"] != ler["lane_id"]:
            return False
        fields = lifetime["inheritance_fields"]
        return fields == ref_lifetime["inheritance_fields"] and all(lifetime["signature"][key] == ler["signature"][key] for key in fields)

    def compute_fields_ok() -> bool:
        names = ("compute_budget_contract", "wall_clock_budget_contract", "precision_contract", "online_privilege")
        return all(all(isinstance(lane["signature"][key], str) and lane["signature"][key] for key in names) for lane in lanes)

    def grade_scope_ok() -> bool:
        rows = report["ontology"]["evidence_grades"]
        return (
            {row["grade_id"] for row in rows} == EVIDENCE_GRADE_IDS
            and all(set(row) == {"grade_id", "claim_scope"} and row["claim_scope"] for row in rows)
            and all("rank" not in row and "level" not in row for row in rows)
        )

    def current_results_null() -> bool:
        expected_states = {
            "ROUND_LER_SINGLE_MODE": "NOT_EVALUATED_NULL",
            "SIX_STATE_LOGICAL_LIFETIME": "NOT_EVALUATED_NULL",
            "RAW_IQ_DIGITAL_HIL": "MISSING_BOARD",
        }
        return all(
            lane["current_result"] == {"evaluation_state": expected_states[lane["lane_id"]], "result_verdict": None}
            for lane in lanes
        )

    def claim_ladders_ok() -> bool:
        expected = {lane["lane_id"]: lane["claim_ladder"] for lane in reference_lanes}
        return all(
            lane["claim_ladder"] == expected[lane["lane_id"]]
            and all(claim["allowed_wording"] and claim["forbidden_wording"] and claim["revocation_conditions"] for claim in lane["claim_ladder"])
            for lane in lanes
        )

    def sources_ok() -> bool:
        registry = report["artifact_registry"]
        if set(registry) != {"config", "implementation", "experiment_plan", "task_board", "risk_registry"}:
            return False
        internal = all(
            binding.get("sha256") == (
                _sha256(ROOT / binding["path"])
                if binding.get("selector") == "STRICT_FILE_SHA256" and (ROOT / binding["path"]).is_file()
                else _canonical_sha256(binding.get("payload"))
            )
            for binding in registry.values()
        )
        return internal and (not check_live_files or all(_binding_live(binding) for binding in registry.values()))

    def source_outputs_ok() -> bool:
        source = report["source_data"]
        markdown = report["markdown"]
        basic = (
            source["rows"] == len(_source_rows(report))
            and isinstance(source["path"], str)
            and source["path"].endswith(".csv")
            and isinstance(markdown["path"], str)
            and markdown["path"].endswith(".md")
            and source["path"] != markdown["path"]
        )
        if not basic or not check_live_files:
            return basic
        source_path = ROOT / source["path"]
        markdown_path = ROOT / markdown["path"]
        return (
            _binding_live({**source, "selector": "STRICT_FILE_SHA256"})
            and _binding_live({**markdown, "selector": "STRICT_FILE_SHA256"})
            and _csv_lossless(report, source_path)
            and all(f"`{item_id}`" in markdown_path.read_text(encoding="utf-8") for item_id in _atomic_ids(report))
        )

    gates = {
        "G01_identity_and_preoutcome_seal": _safe(lambda: report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION and report["config_schema_version"] == CONFIG_SCHEMA_VERSION and report["protocol_id"] == PROTOCOL_ID and report["seal_state"] == "SEALED_PRE_OUTCOME"),
        "G02_exactly_three_independent_namespaces": _safe(lambda: len(lanes) == 3 and tuple(lane["lane_id"] for lane in lanes) == LANE_IDS and len({lane["claim_namespace"] for lane in lanes}) == 3),
        "G03_signature_schema_has_24_frozen_fields": _safe(lambda: tuple(report["task_signature_fields"]) == TASK_SIGNATURE_FIELDS and len(TASK_SIGNATURE_FIELDS) == 24),
        "G04_signatures_are_complete_nonempty_and_distinct": _safe(signatures_ok),
        "G05_ler_code_state_metrics_are_six_state_single_mode": _safe(lambda: ler["signature"]["code_family"] == ref_ler["signature"]["code_family"] and ler["signature"]["state_family"] == ref_ler["signature"]["state_family"] and ler["required_state_ensemble"] == ["+X", "-X", "+Y", "-Y", "+Z", "-Z"] and ler["primary_metrics"] == ["p_L", "p_X", "p_Y", "p_Z", "logical_PTM"]),
        "G06_ler_observation_action_and_denominator_are_causal_full": _safe(lambda: all(ler["signature"][key] == ref_ler["signature"][key] for key in ("observation_schema", "history_horizon", "action_set", "denominator_contract", "observability")) and "hidden_truth" not in ler["signature"]["observation_schema"] and "arbitrary_waveform" not in ler["signature"]["action_set"]),
        "G07_lifetime_metrics_horizon_and_six_state_aggregation_are_complete": _safe(lambda: lifetime["required_state_ensemble"] == ["+X", "-X", "+Y", "-Y", "+Z", "-Z"] and lifetime["primary_metrics"] == ref_lifetime["primary_metrics"] and lifetime["minimum_sequence_cycles"] == 10000 and lifetime["six_state_aggregation_contract"] == ref_lifetime["six_state_aggregation_contract"]),
        "G08_lifetime_inherits_ler_physics_observation_action_and_cost": _safe(inheritance_ok),
        "G09_algorithm_lanes_prohibit_postselection_and_accepted_only_denominators": _safe(lambda: all(lane["signature"]["postselection_policy"] == "PROHIBITED_ZERO_REJECTION_PRIMARY" for lane in (ler, lifetime)) and ler["signature"]["denominator_contract"] == ref_ler["signature"]["denominator_contract"] and lifetime["signature"]["denominator_contract"] == ref_lifetime["signature"]["denominator_contract"]),
        "G10_hil_has_four_boundaries_and_raw_iq_primary": _safe(lambda: hil["timing_boundaries"] == ["decoder_core", "discriminator_output_to_action", "adc_last_sample_to_trigger", "raw_iq_source_to_trigger"] and hil["primary_timing_boundary"] == "raw_iq_source_to_trigger"),
        "G11_hil_statistics_denominator_and_hardware_cost_are_complete": _safe(lambda: hil["primary_metrics"] == ref_hil["primary_metrics"] and hil["signature"]["denominator_contract"] == ref_hil["signature"]["denominator_contract"] and all(metric in hil["primary_metrics"] for metric in ("p50", "p95", "p99", "max", "WCET", "II", "deadline_miss", "resource", "power"))),
        "G12_baseline_classes_keep_only_matched_deployable_ranked": _safe(lambda: {row["class_id"] for row in report["ontology"]["baseline_classes"]} == BASELINE_CLASS_IDS and [row["class_id"] for row in report["ontology"]["baseline_classes"] if row["ranked"] or row["may_support_sota"]] == ["MATCHED_DEPLOYABLE_RANKED"]),
        "G13_matched_baseline_predicate_is_exact_and_fail_closed": _safe(lambda: report["shared_contracts"]["baseline_eligibility_contract"] == reference_config["shared_contracts"]["baseline_eligibility_contract"]),
        "G14_split_is_single_pass_pilot_then_untouched_formal": _safe(lambda: report["shared_contracts"]["split_contract"] == reference_config["shared_contracts"]["split_contract"] and report["shared_contracts"]["split_contract"]["pilot_selection_passes"] == 1 and report["shared_contracts"]["split_contract"]["formal_reselection"] == "PROHIBITED"),
        "G15_observed_only_contract_rejects_future_truth_and_scenario_privilege": _safe(lambda: report["shared_contracts"]["observed_only_contract"] == reference_config["shared_contracts"]["observed_only_contract"] and report["shared_contracts"]["observed_only_contract"]["future_suffix"] == "PROHIBITED"),
        "G16_compute_precision_wallclock_and_deadline_fields_are_nonempty": _safe(lambda: compute_fields_ok() and report["shared_contracts"]["cycle_time_ledger_contract"] == reference_config["shared_contracts"]["cycle_time_ledger_contract"] and report["shared_contracts"]["cycle_time_ledger_contract"]["component_values"] is None and report["shared_contracts"]["cycle_time_ledger_contract"]["missing_numeric_mapping"] == "INCOMPLETE_PHYSICAL_TIME_CLAIM_NOT_ZERO"),
        "G17_multiplicity_is_cluster_level_simultaneous_and_closed_family": _safe(lambda: report["shared_contracts"]["multiplicity_contract"] == reference_config["shared_contracts"]["multiplicity_contract"] and report["shared_contracts"]["multiplicity_contract"]["familywise_alpha"] == 0.05 and report["shared_contracts"]["multiplicity_contract"]["minimum_resamples"] >= 50000),
        "G18_missingness_retains_failures_and_never_imputes_null_as_zero": _safe(lambda: report["shared_contracts"]["missingness_contract"] == reference_config["shared_contracts"]["missingness_contract"] and report["shared_contracts"]["missingness_contract"]["zero_imputation"] == "PROHIBITED"),
        "G19_evidence_grades_are_scope_sets_not_a_global_rank": _safe(grade_scope_ok),
        "G20_puviani_official_and_surpass_slots_remain_local_nulls": _safe(lambda: report["external_claim_slots"]["OFFICIAL_PUVIANI_EXACT"] == reference_config["external_claim_slots"]["OFFICIAL_PUVIANI_EXACT"] and report["external_claim_slots"]["PUVIANI_NMF_SURPASS"] == reference_config["external_claim_slots"]["PUVIANI_NMF_SURPASS"] and report["external_claim_slots"]["OFFICIAL_PUVIANI_EXACT"]["only_blocks"] == ["OFFICIAL_EXACT_REPRODUCTION", "PUVIANI_NMF_SURPASS"]),
        "G21_physical_break_even_and_raw_iq_speed_remain_null_without_grade": _safe(lambda: report["external_claim_slots"]["PHYSICAL_BREAK_EVEN"] == reference_config["external_claim_slots"]["PHYSICAL_BREAK_EVEN"] and report["external_claim_slots"]["RAW_IQ_HIL_SPEED"] == reference_config["external_claim_slots"]["RAW_IQ_HIL_SPEED"]),
        "G22_ler_gate_freezes_each_baseline_and_tail_safety_thresholds": _safe(lambda: ler["result_gate"] == ref_ler["result_gate"] and ler["result_gate"]["numeric_thresholds"]["min_relative_improvement_point_each_baseline"] == {"op": ">=", "value": 0.15} and ler["result_gate"]["numeric_thresholds"]["min_simultaneous_relative_lcb_each_baseline"] == {"op": ">=", "value": 0.10}),
        "G23_lifetime_gate_freezes_six_state_gain_cost_and_horizon": _safe(lambda: lifetime["result_gate"] == ref_lifetime["result_gate"] and lifetime["result_gate"]["numeric_thresholds"]["min_six_state_relative_gain_point"] == {"op": ">=", "value": 0.15} and lifetime["result_gate"]["numeric_thresholds"]["minimum_sequence_cycles"] == {"op": ">=", "value": 10000}),
        "G24_hil_gate_requires_board_chain_three_seeds_million_transactions_and_comparator": _safe(lambda: hil["result_gate"] == ref_hil["result_gate"] and hil["result_gate"]["numeric_thresholds"]["implementation_seed_count"] == {"op": ">=", "value": 3} and hil["result_gate"]["numeric_thresholds"]["transaction_count"] == {"op": ">=", "value": 1000000} and "same_task_measured_comparator_available" in hil["result_gate"]["required_boolean_fields"]),
        "G25_future_evaluator_fixtures_cover_go_no_go_incomplete_and_unopened": _safe(lambda: report["verdict_fixtures"] == _verdict_fixtures(lanes) and {row["outcome"]["evaluation_state"] for row in report["verdict_fixtures"]} >= {"NOT_EVALUATED_NULL", "INCOMPLETE", "COMPLETE"}),
        "G26_claim_ladders_have_wording_grades_and_revocation": _safe(claim_ladders_ok),
        "G27_forbidden_transfer_registry_is_complete": _safe(lambda: {row["transfer_id"] for row in report["forbidden_transfers"]} == FORBIDDEN_TRANSFER_IDS and report["forbidden_transfers"] == reference_config["forbidden_transfers"]),
        "G28_global_score_winner_count_and_cross_lane_rescue_are_prohibited": _safe(lambda: report["shared_contracts"]["nontransfer_contract"] == reference_config["shared_contracts"]["nontransfer_contract"] and report["shared_contracts"]["nontransfer_contract"]["aggregate_score"] == "PROHIBITED" and report["shared_contracts"]["nontransfer_contract"]["winner_count"] == "PROHIBITED"),
        "G29_source_contracts_are_semantically_or_exactly_live": _safe(sources_ok),
        "G30_current_performance_results_are_null_not_fake_go_or_no_go": _safe(current_results_null),
        "G31_lifetime_does_not_promote_puviani_or_physical_claims": _safe(lambda: next(claim for claim in lifetime["claim_ladder"] if claim["claim_id"] == "LIFE-C4-PUVIANI-SURPASS")["state"] == "BLOCKED_NULL" and next(claim for claim in lifetime["claim_ladder"] if claim["claim_id"] == "LIFE-C5-PHYSICAL-BREAK-EVEN")["state"] == "BLOCKED_NULL"),
        "G32_preboard_rtl_does_not_promote_measured_hil_or_speed": _safe(lambda: next(claim for claim in hil["claim_ladder"] if claim["claim_id"] == "HIL-C1-FIXED-POINT-CXXRTL")["state"] == "PARENT_RESTRICTED_ONLY" and all(next(claim for claim in hil["claim_ladder"] if claim["claim_id"] == claim_id)["state"] == "BLOCKED_NULL" for claim_id in ("HIL-C3-MEASURED-INTEGRATED-CHAIN", "HIL-C4-SAME-TASK-SPEED"))),
        "G33_independent_backends_each_pass_without_averaging": _safe(lambda: report["shared_contracts"]["independent_backend_contract"] == reference_config["shared_contracts"]["independent_backend_contract"] and report["shared_contracts"]["independent_backend_contract"]["averaging_across_backends"] == "PROHIBITED"),
        "G34_result_state_machine_is_null_incomplete_then_binary_complete": _safe(lambda: report["ontology"]["result_states"] == ["NOT_EVALUATED_NULL", "INCOMPLETE", "GO", "NO_GO"] and all((row["outcome"]["evaluation_state"] == "COMPLETE") == (row["outcome"]["result_verdict"] is not None) for row in report["verdict_fixtures"])),
        "G35_source_data_and_human_contract_are_lossless_and_live": _safe(source_outputs_ok),
        "G36_one_substantive_mutation_per_gate_fails_closed": _safe(lambda: report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == len(GATE_IDS) and len(report["semantic_mutation_audit"]["cases"]) == len(GATE_IDS) and {row["target_gate"] for row in report["semantic_mutation_audit"]["cases"]} == set(GATE_IDS) and all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"])),
    }
    return gates


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, target: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {
            "count": len(GATE_IDS), "detected": len(GATE_IDS),
            "cases": [{"target_gate": gate, "rejected": True} for gate in GATE_IDS],
        }
        mutate(candidate)
        rejected = not evaluate_gates(candidate, check_live_files=False)[target]
        if not rejected:
            raise RuntimeError(f"mutation escaped {target}: {name}")
        cases.append({"mutation_id": name, "target_gate": target, "rejected": True})

    attempt("M01_protocol_identity", GATE_IDS[0], lambda x: x.update(protocol_id="CONFLATED"))
    attempt("M02_duplicate_lane", GATE_IDS[1], lambda x: x["lanes"][1].update(lane_id=x["lanes"][0]["lane_id"]))
    attempt("M03_drop_schema_field", GATE_IDS[2], lambda x: x["task_signature_fields"].pop())
    attempt("M04_drop_signature_field", GATE_IDS[3], lambda x: x["lanes"][0]["signature"].pop("missingness_contract"))
    attempt("M05_drop_sixth_ler_state", GATE_IDS[4], lambda x: x["lanes"][0]["required_state_ensemble"].pop())
    attempt("M06_hidden_truth_observation", GATE_IDS[5], lambda x: x["lanes"][0]["signature"].update(observation_schema="raw_IQ_plus_hidden_truth"))
    attempt("M07_drop_T_ch", GATE_IDS[6], lambda x: x["lanes"][1]["primary_metrics"].remove("T_ch"))
    attempt("M08_lifetime_changes_action", GATE_IDS[7], lambda x: x["lanes"][1]["signature"].update(action_set="arbitrary_waveform"))
    attempt("M09_accepted_only_ler", GATE_IDS[8], lambda x: x["lanes"][0]["signature"].update(postselection_policy="ACCEPTED_ONLY"))
    attempt("M10_core_as_primary_hil", GATE_IDS[9], lambda x: x["lanes"][2].update(primary_timing_boundary="decoder_core"))
    attempt("M11_drop_hil_wcet", GATE_IDS[10], lambda x: x["lanes"][2]["primary_metrics"].remove("WCET"))
    attempt("M12_rank_capacity_ceiling", GATE_IDS[11], lambda x: next(row for row in x["ontology"]["baseline_classes"] if row["class_id"] == "CAPACITY_CEILING_NONRANKING").update(ranked=True))
    attempt("M13_remove_matched_budget_predicate", GATE_IDS[12], lambda x: x["shared_contracts"]["baseline_eligibility_contract"]["all_required_equal"].remove("wall_clock_deadline"))
    attempt("M14_allow_formal_reselection", GATE_IDS[13], lambda x: x["shared_contracts"]["split_contract"].update(formal_reselection="ALLOWED"))
    attempt("M15_allow_future_suffix", GATE_IDS[14], lambda x: x["shared_contracts"]["observed_only_contract"].update(future_suffix="ALLOWED"))
    attempt("M16_blank_compute_budget", GATE_IDS[15], lambda x: x["lanes"][0]["signature"].update(compute_budget_contract=""))
    attempt("M17_uncorrected_alpha", GATE_IDS[16], lambda x: x["shared_contracts"]["multiplicity_contract"].update(familywise_alpha=0.5))
    attempt("M18_zero_impute_missing", GATE_IDS[17], lambda x: x["shared_contracts"]["missingness_contract"].update(zero_imputation="ALLOWED"))
    attempt("M19_global_grade_rank", GATE_IDS[18], lambda x: x["ontology"]["evidence_grades"][0].update(rank=99))
    attempt("M20_fill_puviani_from_project", GATE_IDS[19], lambda x: x["external_claim_slots"]["PUVIANI_NMF_SURPASS"].update(value=True, state="COMPLETE"))
    attempt("M21_fill_physical_from_sim", GATE_IDS[20], lambda x: x["external_claim_slots"]["PHYSICAL_BREAK_EVEN"].update(value=1.2, state="COMPLETE"))
    attempt("M22_lower_ler_lcb", GATE_IDS[21], lambda x: x["lanes"][0]["result_gate"]["numeric_thresholds"]["min_simultaneous_relative_lcb_each_baseline"].update(value=0.0))
    attempt("M23_shorten_lifetime", GATE_IDS[22], lambda x: x["lanes"][1]["result_gate"]["numeric_thresholds"]["minimum_sequence_cycles"].update(value=100))
    attempt("M24_remove_raw_iq_measurement", GATE_IDS[23], lambda x: x["lanes"][2]["result_gate"]["required_boolean_fields"].remove("raw_iq_board_measured"))
    attempt("M25_forge_go_fixture", GATE_IDS[24], lambda x: x["verdict_fixtures"][-1]["outcome"].update(result_verdict="GO_HIL_SPEED"))
    attempt("M26_open_ler_sota_claim", GATE_IDS[25], lambda x: next(claim for claim in x["lanes"][0]["claim_ladder"] if claim["claim_id"] == "LER-C4-SOTA").update(state="SUPPORTED"))
    attempt("M27_remove_transfer", GATE_IDS[26], lambda x: x["forbidden_transfers"].pop())
    attempt("M28_enable_weighted_score", GATE_IDS[27], lambda x: x["shared_contracts"]["nontransfer_contract"].update(aggregate_score="ALLOWED"))
    attempt("M29_forge_semantic_binding", GATE_IDS[28], lambda x: x["artifact_registry"]["risk_registry"].update(sha256="0" * 64))
    attempt("M30_fake_current_ler_go", GATE_IDS[29], lambda x: x["lanes"][0].update(current_result={"evaluation_state": "COMPLETE", "result_verdict": "GO_LER_SOTA"}))
    attempt("M31_open_physical_lifetime", GATE_IDS[30], lambda x: next(claim for claim in x["lanes"][1]["claim_ladder"] if claim["claim_id"] == "LIFE-C5-PHYSICAL-BREAK-EVEN").update(state="SUPPORTED"))
    attempt("M32_open_preboard_as_measured", GATE_IDS[31], lambda x: next(claim for claim in x["lanes"][2]["claim_ladder"] if claim["claim_id"] == "HIL-C3-MEASURED-INTEGRATED-CHAIN").update(state="SUPPORTED"))
    attempt("M33_average_backend_results", GATE_IDS[32], lambda x: x["shared_contracts"]["independent_backend_contract"].update(averaging_across_backends="ALLOWED"))
    attempt("M34_remove_incomplete_state", GATE_IDS[33], lambda x: x["ontology"]["result_states"].remove("INCOMPLETE"))
    attempt("M35_drop_source_row", GATE_IDS[34], lambda x: x["source_data"].update(rows=x["source_data"]["rows"] - 1))
    attempt("M36_forge_mutation_count", GATE_IDS[35], lambda x: x.update(semantic_mutation_audit={"count": len(GATE_IDS), "detected": len(GATE_IDS) - 1, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "task_id", "schema_version", "config_schema_version", "protocol_id", "seal_state",
        "task_signature_fields", "ontology", "shared_contracts", "external_claim_slots",
        "lanes", "forbidden_transfers", "verdict_fixtures", "artifact_registry",
    )
    return {key: report[key] for key in keys}


def build_report(
    *,
    config_path: Path = DEFAULT_CONFIG,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
    markdown_path: Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    config = _load(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION or config.get("task_id") != TASK_ID:
        raise ValueError("T9.1.1 config identity mismatch")
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "config_schema_version": config["schema_version"],
        "protocol_id": config["protocol_id"],
        "frozen_date": config["frozen_date"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seal_state": config["seal_state"],
        "task_signature_fields": list(config["task_signature_fields"]),
        "ontology": copy.deepcopy(config["ontology"]),
        "shared_contracts": copy.deepcopy(config["shared_contracts"]),
        "external_claim_slots": copy.deepcopy(config["external_claim_slots"]),
        "lanes": _enriched_lanes(config),
        "forbidden_transfers": copy.deepcopy(config["forbidden_transfers"]),
        "artifact_registry": _artifact_registry(config_path),
    }
    report["verdict_fixtures"] = _verdict_fixtures(report["lanes"])
    report["source_data"] = {"path": _relative(source_data_path), "rows": len(_source_rows(report))}
    report["markdown"] = {"path": _relative(markdown_path)}
    report["semantic_mutation_audit"] = {
        "count": len(GATE_IDS), "detected": len(GATE_IDS),
        "cases": [{"target_gate": gate, "rejected": True} for gate in GATE_IDS],
    }
    report["semantic_mutation_audit"] = _mutations(report)

    _write_source_data(report, source_data_path)
    _atomic_text(_render_markdown(report), markdown_path)
    report["source_data"] = {
        "path": _relative(source_data_path),
        "rows": len(_source_rows(report)),
        "sha256": _sha256(source_data_path),
        "bytes": source_data_path.stat().st_size,
    }
    report["markdown"] = {
        "path": _relative(markdown_path),
        "sha256": _sha256(markdown_path),
        "bytes": markdown_path.stat().st_size,
    }
    report["gates"] = evaluate_gates(report)
    failed = [key for key, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {"passed": len(report["gates"]) - len(failed), "failed": failed}
    report["verdict"] = VERDICT if not failed else "FAIL_PHASE9_THREE_LANE_PROTOCOL"
    report["analysis_sha256"] = _canonical_sha256(_analysis_payload(report))
    return report


def verify_report(
    report: Mapping[str, Any] | None = None,
    path: Path = DEFAULT_REPORT,
) -> dict[str, bool]:
    value = dict(report) if report is not None else _load(path)
    gates = evaluate_gates(value)
    checks = {
        "identity": value.get("task_id") == TASK_ID and value.get("schema_version") == SCHEMA_VERSION,
        "all_gates": all(gates.values()),
        "gate_cache": value.get("gates") == gates and value.get("gate_summary") == {"passed": len(gates), "failed": []},
        "verdict": value.get("verdict") == VERDICT,
        "analysis_hash": value.get("analysis_sha256") == _canonical_sha256(_analysis_payload(value)),
        "source_data": _csv_lossless(value, ROOT / value["source_data"]["path"]),
        "markdown_live": _binding_live({**value["markdown"], "selector": "STRICT_FILE_SHA256"}),
        "current_results_null": all(lane["current_result"]["result_verdict"] is None for lane in value["lanes"]),
    }
    if not all(checks.values()):
        raise ValueError(f"T9.1.1 verification failed: {checks}; failed_gates={[key for key, passed in gates.items() if not passed]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_report(path=args.report), ensure_ascii=False, indent=2))
        return 0
    report = build_report(config_path=args.config, source_data_path=args.source_data, markdown_path=args.markdown)
    _atomic_json(report, args.report)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "analysis_sha256": report["analysis_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
