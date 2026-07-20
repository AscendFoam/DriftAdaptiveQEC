"""T6.15.5 Phase-6B V5 final/early-stop evidence gate.

The current execution follows the preregistered early-stop branch: T6.10.1
failed both causal-router and incremental-action-space entry gates.  This gate
does not synthesize missing V5 experiments.  It proves that all downstream
conditional tasks were dropped, their outputs are absent, and every affected
claim is revoked before Phase 6C is allowed to run read-only.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.15.5"
SCHEMA_VERSION = "t6.15.5-route-a-v5-final-evidence-gate-v2"
PROTOCOL_ID = "ROUTE-A-V5-EARLY-HEADROOM-STOP-V1"
HEADROOM_PATH = ROOT / "docs" / "t6_10_1_causal_headroom.json"
BOARD_PATH = ROOT / "docs" / "new_task_board.md"
BOARD_GATE_PATH = ROOT / "docs" / "t6_9_2_route_a_board_measurement_blocker.json"
V4_FINAL_PATH = ROOT / "docs" / "t6_9_3_route_a_final_evidence_gate.json"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_15_5_route_a_v5_final_evidence_gate.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_15_5_route_a_v5_final_evidence_gate_source_data.csv"
VERDICT = "NO_GO_V5_EARLY_HEADROOM_STOP"
ROUTER_GATE = 0.10
ACTION_GATE = 0.12

DOWNSTREAM_DROPPED_TASKS = (
    "T6.10.2",
    "T6.10.3",
    *(f"T6.11.{index}" for index in range(1, 5)),
    *(f"T6.12.{index}" for index in range(1, 5)),
    *(f"T6.13.{index}" for index in range(1, 4)),
    *(f"T6.14.{index}" for index in range(1, 4)),
    *(f"T6.15.{index}" for index in range(1, 5)),
)
V5_OUTPUT_PREFIXES = (
    "t6_10_2_",
    "t6_10_3_",
    "t6_11_",
    "t6_12_",
    "t6_13_",
    "t6_14_",
    "t6_15_1_",
    "t6_15_2_",
    "t6_15_3_",
    "t6_15_4_",
)
TASK_ROW = re.compile(
    r"^\|\s*(T(?:-RISK-\d{8}-\d+|\d+(?:\.\d+)+))\s*\|\s*([^|]+?)\s*\|",
    re.MULTILINE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _board_status_binding(statuses: Mapping[str, str]) -> dict[str, Any]:
    """Bind only the preregistered V5 early-stop state, not board prose/history.

    The task board is a living status document.  Hashing the complete Markdown
    made an unrelated next-task pointer or history entry invalidate the frozen
    Phase-6B scientific state.  The evidence gate depends only on the exact 20
    conditional V5 tasks, so the binding is deliberately scoped to those rows.
    """

    scoped = {task: statuses.get(task) for task in DOWNSTREAM_DROPPED_TASKS}
    return {
        "path": str(BOARD_PATH.relative_to(ROOT)).replace("\\", "/"),
        "scope": "T6.10.2--T6.15.4 preregistered conditional-task statuses only",
        "statuses": scoped,
        "semantic_sha256": _canonical_sha256(scoped),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_statuses(board: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for task_id, status in TASK_ROW.findall(board.split("## 进度日志", 1)[0]):
        if task_id in result:
            raise ValueError(f"duplicate authoritative task row: {task_id}")
        result[task_id] = status.strip()
    return result


def _v5_outputs() -> list[str]:
    found: list[str] = []
    for base in (ROOT / "docs", ROOT / "runs"):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            lowered = path.name.lower()
            if any(lowered.startswith(prefix) for prefix in V5_OUTPUT_PREFIXES):
                found.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return sorted(found)


def _claim_rows() -> list[dict[str, object]]:
    return [
        {"claim_id": "V5-ALG-LER-10PCT", "state": "REVOKED", "reason": "strict-causal development headroom is below 10%"},
        {"claim_id": "V5-TAIL-CALIBRATION-TELEGRAPH", "state": "NOT_RUN_EARLY_STOP", "reason": "no V5 candidate entered pilot/formal"},
        {"claim_id": "V5-POSTERIOR-MIXTURE-ACTION", "state": "REVOKED", "reason": "incremental action-space upper bound is below 12%"},
        {"claim_id": "V5-UNTOUCHED-FORMAL", "state": "NOT_RUN_EARLY_STOP", "reason": "formal manifest and outputs intentionally absent"},
        {"claim_id": "V5-QUANTIZED-RETENTION", "state": "NOT_RUN_EARLY_STOP", "reason": "no eligible V5 action to quantize"},
        {"claim_id": "V5-LONG-CXXRTL", "state": "NOT_RUN_EARLY_STOP", "reason": "V4 replay cannot substitute for V5 qualification"},
        {"claim_id": "V5-FORMAL-ATOMIC-SAFETY", "state": "NOT_RUN_EARLY_STOP", "reason": "no V5 RTL/profile exists"},
        {"claim_id": "V5-MULTISEED-PR", "state": "NOT_RUN_EARLY_STOP", "reason": "no V5 RTL/profile exists"},
        {"claim_id": "V5-MEASURED-HARDWARE", "state": "BLOCKED", "reason": "T6.9.2 still has 42 null measured fields"},
        {"claim_id": "PHASE6C-READONLY-AUX", "state": "ALLOWED_AFTER_THIS_GATE", "reason": "secondary lanes cannot modify or rescue Phase 6B"},
    ]


def _core_checks(
    headroom: Mapping[str, Any],
    statuses: Mapping[str, str],
    v5_outputs: Sequence[str],
    board_gate: Mapping[str, Any],
    v4_final: Mapping[str, Any],
    claims: Sequence[Mapping[str, object]],
) -> dict[str, bool]:
    nested = headroom.get("development_audit", {}).get("nested_audit", {})
    baseline = nested.get("nested_strongest_baseline", {})
    selector = nested.get("nested_selector", {})
    hard = nested.get("hard_decision_oracle", {})
    expanded = nested.get("expanded_candidate_action_oracle", {})
    try:
        baseline_errors = int(baseline["errors"])
        selector_errors = int(selector["errors"])
        hard_errors = int(hard["errors"])
        expanded_errors = int(expanded["errors"])
        router = (baseline_errors - selector_errors) / baseline_errors
        action = (hard_errors - expanded_errors) / baseline_errors
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        router = float("nan")
        action = float("nan")
    claim_states = {str(row.get("claim_id")): str(row.get("state")) for row in claims}
    measured = board_gate.get("measured_results", {})
    return {
        "headroom_identity": headroom.get("task_id") == "T6.10.1" and headroom.get("verdict") == "NO_GO_V5_INSUFFICIENT_ACTION_SPACE_HEADROOM",
        "formal_replay_is_exact_and_diagnostic_only": headroom.get("formal_diagnostic_audit", {}).get("trajectory_count") == 1_464 and headroom.get("formal_diagnostic_audit", {}).get("all_parent_replays_exact") is True and headroom.get("formal_diagnostic_audit", {}).get("diagnostic_only") is True,
        "router_headroom_recomputes_below_gate": abs(router - float(nested.get("existing_expert_causal_headroom", 999.0))) < 1e-15 and router < ROUTER_GATE,
        "incremental_action_headroom_recomputes_below_gate": abs(action - float(expanded.get("incremental_action_space_headroom_vs_baseline", 999.0))) < 1e-15 and action < ACTION_GATE,
        "overall_oracle_not_substituted_for_incremental_action": float(expanded.get("overall_relative_headroom_vs_baseline", -1.0)) > ACTION_GATE and action < ACTION_GATE,
        "all_conditional_tasks_are_dropped": all(statuses.get(task) == "Dropped" for task in DOWNSTREAM_DROPPED_TASKS),
        "this_gate_is_active_or_done": statuses.get(TASK_ID) in {"In Progress", "Done"},
        "no_v5_downstream_outputs_exist": len(v5_outputs) == 0,
        "measured_hardware_remains_blocked_and_null": str(board_gate.get("verdict", "")).startswith("BLOCKED_T6_9_2") and isinstance(measured, Mapping) and len(measured) == 42 and all(value is None for value in measured.values()),
        "v4_full_paper_remains_no_go": str(v4_final.get("verdict", "")).startswith("NO_GO_FULL_HIGH_LEVEL_PAPER"),
        "all_v5_performance_and_implementation_claims_closed": all(claim_states.get(claim) in {"REVOKED", "NOT_RUN_EARLY_STOP", "BLOCKED"} for claim in ("V5-ALG-LER-10PCT", "V5-TAIL-CALIBRATION-TELEGRAPH", "V5-POSTERIOR-MIXTURE-ACTION", "V5-UNTOUCHED-FORMAL", "V5-QUANTIZED-RETENTION", "V5-LONG-CXXRTL", "V5-FORMAL-ATOMIC-SAFETY", "V5-MULTISEED-PR", "V5-MEASURED-HARDWARE")),
        "phase6c_is_readonly_only": claim_states.get("PHASE6C-READONLY-AUX") == "ALLOWED_AFTER_THIS_GATE",
    }


def _semantic_mutations(
    headroom: Mapping[str, Any],
    statuses: Mapping[str, str],
    board_gate: Mapping[str, Any],
    v4_final: Mapping[str, Any],
    claims: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    cases: list[tuple[str, dict[str, Any], dict[str, str], list[str], dict[str, Any], dict[str, Any], list[dict[str, object]]]] = []
    mutated = deepcopy(headroom)
    mutated["development_audit"]["nested_audit"]["expanded_candidate_action_oracle"]["incremental_action_space_headroom_vs_baseline"] = mutated["development_audit"]["nested_audit"]["expanded_candidate_action_oracle"]["overall_relative_headroom_vs_baseline"]
    cases.append(("overall_oracle_as_action_gain", mutated, dict(statuses), [], deepcopy(board_gate), deepcopy(v4_final), [dict(row) for row in claims]))
    mutated = deepcopy(headroom)
    mutated["development_audit"]["nested_audit"]["existing_expert_causal_headroom"] = 0.15
    cases.append(("router_headroom_promotion", mutated, dict(statuses), [], deepcopy(board_gate), deepcopy(v4_final), [dict(row) for row in claims]))
    bad_status = dict(statuses)
    bad_status[DOWNSTREAM_DROPPED_TASKS[0]] = "Done"
    cases.append(("dropped_task_promoted", deepcopy(headroom), bad_status, [], deepcopy(board_gate), deepcopy(v4_final), [dict(row) for row in claims]))
    cases.append(("invented_v5_formal_output", deepcopy(headroom), dict(statuses), ["docs/t6_14_1_fake.json"], deepcopy(board_gate), deepcopy(v4_final), [dict(row) for row in claims]))
    bad_board = deepcopy(board_gate)
    bad_board["measured_results"][next(iter(bad_board["measured_results"]))] = 0
    cases.append(("measured_null_imputed", deepcopy(headroom), dict(statuses), [], bad_board, deepcopy(v4_final), [dict(row) for row in claims]))
    bad_claims = [dict(row) for row in claims]
    bad_claims[0]["state"] = "SUPPORTED"
    cases.append(("revoked_claim_promoted", deepcopy(headroom), dict(statuses), [], deepcopy(board_gate), deepcopy(v4_final), bad_claims))
    output: list[dict[str, object]] = []
    for name, local_headroom, local_statuses, outputs, local_board, local_final, local_claims in cases:
        checks = _core_checks(local_headroom, local_statuses, outputs, local_board, local_final, local_claims)
        output.append({"mutation": name, "detected": not all(checks.values()), "failed_checks": [key for key, value in checks.items() if not value]})
    return output


def _write_source_data(path: Path, claims: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("claim_id", "state", "reason"))
        writer.writeheader()
        writer.writerows(claims)
    temporary.replace(path)


def build_report(
    artifact: Path = DEFAULT_ARTIFACT,
    source_data: Path = DEFAULT_SOURCE_DATA,
) -> dict[str, Any]:
    headroom = json.loads(HEADROOM_PATH.read_text(encoding="utf-8"))
    board_text = BOARD_PATH.read_text(encoding="utf-8")
    statuses = _task_statuses(board_text)
    board_gate = json.loads(BOARD_GATE_PATH.read_text(encoding="utf-8"))
    v4_final = json.loads(V4_FINAL_PATH.read_text(encoding="utf-8"))
    outputs = _v5_outputs()
    claims = _claim_rows()
    checks = _core_checks(headroom, statuses, outputs, board_gate, v4_final, claims)
    mutations = _semantic_mutations(headroom, statuses, board_gate, v4_final, claims)
    gates = {
        "G01_t6_10_1_identity_and_negative_verdict": checks["headroom_identity"],
        "G02_all_1464_formal_replays_are_exact_diagnostic_only": checks["formal_replay_is_exact_and_diagnostic_only"],
        "G03_router_headroom_recomputes_below_10_percent": checks["router_headroom_recomputes_below_gate"],
        "G04_incremental_action_headroom_recomputes_below_12_percent": checks["incremental_action_headroom_recomputes_below_gate"],
        "G05_overall_truth_oracle_is_not_action_space_gain": checks["overall_oracle_not_substituted_for_incremental_action"],
        "G06_all_20_conditional_tasks_are_dropped": checks["all_conditional_tasks_are_dropped"] and len(DOWNSTREAM_DROPPED_TASKS) == 20,
        "G07_no_v5_downstream_output_exists": checks["no_v5_downstream_outputs_exist"],
        "G08_all_v5_claims_are_revoked_not_run_or_blocked": checks["all_v5_performance_and_implementation_claims_closed"],
        "G09_measured_hardware_remains_blocked_with_42_nulls": checks["measured_hardware_remains_blocked_and_null"],
        "G10_v4_full_paper_no_go_is_not_overwritten": checks["v4_full_paper_remains_no_go"],
        "G11_phase6c_permission_is_readonly_only": checks["phase6c_is_readonly_only"],
        "G12_all_semantic_mutations_are_detected": all(row["detected"] for row in mutations),
    }
    if not all(gates.values()):
        raise RuntimeError(f"T6.15.5 early-stop evidence gate failed: {[key for key,value in gates.items() if not value]}")
    nested = headroom["development_audit"]["nested_audit"]
    expanded = nested["expanded_candidate_action_oracle"]
    bindings = {
        str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
        for path in (HEADROOM_PATH, BOARD_GATE_PATH, V4_FINAL_PATH)
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_path": "EARLY_STOP_AT_T6.10.1_HEADROOM_GATE",
        "headroom_recomputation": {
            "nested_selector_errors": nested["nested_selector"]["errors"],
            "nested_baseline_errors": nested["nested_strongest_baseline"]["errors"],
            "strict_causal_router_headroom": nested["existing_expert_causal_headroom"],
            "router_gate": ROUTER_GATE,
            "hard_decision_oracle_errors": nested["hard_decision_oracle"]["errors"],
            "expanded_candidate_oracle_errors": expanded["errors"],
            "incremental_action_space_headroom": expanded["incremental_action_space_headroom_vs_baseline"],
            "action_gate": ACTION_GATE,
            "overall_truth_oracle_headroom_not_used_for_gate": expanded["overall_relative_headroom_vs_baseline"],
        },
        "dropped_tasks": list(DOWNSTREAM_DROPPED_TASKS),
        "v5_downstream_outputs_found": outputs,
        "formal_access": {"v5_formal_manifest_exists": False, "v5_formal_output_exists": False, "v4_formal_used_only_as_t6_10_1_diagnostic": True},
        "claim_registry": claims,
        "parent_bindings": bindings,
        "board_status_binding": _board_status_binding(statuses),
        "semantic_mutations": mutations,
        "gates": gates,
        "gate_summary": {"passed": len(gates), "failed": []},
        "phase6c_permission": {"allowed": True, "mode": "READ_ONLY_AUXILIARY_COMPARISONS", "may_modify_phase6b_verdict": False, "may_rescue_v5_claim": False},
        "measured_hardware_claim": {"state": "BLOCKED", "source_task": "T6.9.2", "null_fields": 42},
        "status": "DONE_EARLY_STOP_NEGATIVE_RESULT_PRESERVED",
        "verdict": VERDICT,
    }
    _write_source_data(source_data, claims)
    report["source_data_binding"] = {"path": str(source_data.relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(source_data), "row_count": len(claims)}
    report["analysis_sha256"] = _canonical_sha256({key: report[key] for key in ("execution_path", "headroom_recomputation", "dropped_tasks", "formal_access", "claim_registry", "parent_bindings", "board_status_binding", "semantic_mutations", "gates", "phase6c_permission", "measured_hardware_claim", "verdict")})
    _write_json(artifact, report)
    return report


def validate_report(path: Path = DEFAULT_ARTIFACT) -> dict[str, bool]:
    report = json.loads(path.read_text(encoding="utf-8"))
    source = ROOT / report["source_data_binding"]["path"]
    board = BOARD_PATH.read_text(encoding="utf-8")
    checks = {
        "identity": report.get("task_id") == TASK_ID and report.get("schema_version") == SCHEMA_VERSION and report.get("verdict") == VERDICT,
        "source_data": source.is_file() and _sha256(source) == report["source_data_binding"]["sha256"] and report["source_data_binding"]["row_count"] == len(report["claim_registry"]),
        "parent_hashes": all((ROOT / rel).is_file() and _sha256(ROOT / rel) == digest for rel, digest in report["parent_bindings"].items()),
        "board_status_binding": report.get("board_status_binding") == _board_status_binding(_task_statuses(board)),
        "dropped_statuses": all(_task_statuses(board).get(task) == "Dropped" for task in report["dropped_tasks"]),
        "no_outputs": _v5_outputs() == report["v5_downstream_outputs_found"] == [],
        "gates": report["gate_summary"] == {"passed": len(report["gates"]), "failed": []} and all(report["gates"].values()),
        "analysis_hash": report["analysis_sha256"] == _canonical_sha256({key: report[key] for key in ("execution_path", "headroom_recomputation", "dropped_tasks", "formal_access", "claim_registry", "parent_bindings", "board_status_binding", "semantic_mutations", "gates", "phase6c_permission", "measured_hardware_claim", "verdict")}),
    }
    if not all(checks.values()):
        raise ValueError(f"T6.15.5 artifact validation failed: {[key for key,value in checks.items() if not value]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        print(json.dumps(validate_report(args.artifact), indent=2))
    else:
        report = build_report(args.artifact, args.source_data)
        print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
