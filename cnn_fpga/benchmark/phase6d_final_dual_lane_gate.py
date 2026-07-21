"""Execute the T6.26.4 independent Phase-6D dual-lane final gate."""

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

from cnn_fpga.benchmark import converged_hardware_lane_qualification as hardware_gate
from cnn_fpga.benchmark import converged_long_rtl_qualification as long_gate
from cnn_fpga.benchmark import converged_rtl_formal as formal_gate
from cnn_fpga.benchmark import multimode_causal_headroom as headroom_gate
from cnn_fpga.benchmark import phase6d_dual_lane_evidence_matrix as matrix_gate


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.26.4"
SCHEMA_VERSION = "t6.26.4-final-dual-lane-gate-v1"
RUNNER = Path(__file__).resolve()
CONFIG = ROOT / "configs/phase6d/t6_26_4_final_dual_lane_gate.json"
BOARD = ROOT / "docs/new_task_board.md"
REPORT = ROOT / "docs/t6_26_4_final_dual_lane_gate.json"
SOURCE_DATA = ROOT / "docs/t6_26_4_final_dual_lane_gate_source_data.csv"
MARKDOWN = ROOT / "docs/phase6d_final_dual_lane_gate.md"

MATRIX_REPORT = ROOT / "docs/t6_26_3_dual_lane_evidence_matrix.json"
HEADROOM_REPORT = ROOT / "docs/t6_20_4_multimode_causal_headroom.json"
FORMAL_REPORT = ROOT / "docs/t6_25_2_converged_rtl_formal.json"
LONG_REPORT = ROOT / "docs/t6_25_3_converged_long_rtl.json"
HARDWARE_REPORT = ROOT / "docs/t6_25_4_converged_hardware.json"

ARTIFACT_PATHS = {
    "implementation": RUNNER,
    "self_config": CONFIG,
    "matrix_report": MATRIX_REPORT,
    "matrix_source_data": ROOT / "docs/t6_26_3_dual_lane_evidence_source_data.csv",
    "matrix_config": ROOT / "configs/phase6d/t6_26_3_dual_lane_evidence_matrix.json",
    "matrix_code": ROOT / "cnn_fpga/benchmark/phase6d_dual_lane_evidence_matrix.py",
    "matrix_markdown": ROOT / "docs/phase6d_dual_lane_evidence_matrix.md",
    "cancellation_ledger": ROOT / "docs/phase6d_multimode_v1_cancellation_ledger.md",
    "headroom_report": HEADROOM_REPORT,
    "headroom_raw": ROOT / "runs/t6_20_4_causal_headroom_raw.json",
    "headroom_config": ROOT / "configs/phase6d/t6_20_4_causal_headroom.json",
    "headroom_code": ROOT / "cnn_fpga/benchmark/multimode_causal_headroom.py",
    "headroom_source": ROOT / "scripts/run_t6_20_4_causal_headroom.jl",
    "formal_report": FORMAL_REPORT,
    "formal_source_data": ROOT / "docs/t6_25_2_converged_rtl_formal_source_data.csv",
    "formal_code": ROOT / "cnn_fpga/benchmark/converged_rtl_formal.py",
    "long_report": LONG_REPORT,
    "long_source_data": ROOT / "docs/t6_25_3_converged_long_rtl_source_data.csv",
    "long_code": ROOT / "cnn_fpga/benchmark/converged_long_rtl_qualification.py",
    "hardware_report": HARDWARE_REPORT,
    "hardware_source_data": ROOT / "docs/t6_25_4_converged_hardware_source_data.csv",
    "hardware_config": ROOT / "configs/phase6d/t6_25_4_converged_hardware.json",
    "hardware_code": ROOT / "cnn_fpga/benchmark/converged_hardware_lane_qualification.py",
    "production_rtl": ROOT / "cnn_fpga/rtl/gkp_route_a_converged_production_top.sv",
    "synthesis_rtl": ROOT / "cnn_fpga/rtl/gkp_route_a_converged_synth_top.sv",
    "hardware_netlist": ROOT / "docs/t6_25_4_converged_synth_netlist.json",
    "board_blocker_report": ROOT / "docs/t6_9_2_route_a_board_measurement_blocker.json",
    "board_blocker_code": ROOT / "cnn_fpga/benchmark/route_a_board_measurement_gate.py",
    "historical_claim_snapshot": ROOT / "docs/t7_1_1_claim_evidence_boundary_matrix.json",
    "historical_main_figure_1_2": ROOT / "docs/t7_1_2_main_figure_contract.json",
    "historical_main_figure_3_4": ROOT / "docs/t7_1_3_main_result_figure_contract.json",
}

VERDICTS = {"GO_TWO_LANE", "GO_MULTIMODE_ONLY", "GO_RTL_ONLY", "NO_GO"}
FORMAL_VERDICT = "PASS_CONVERGED_PRODUCTION_TOP_PROPERTY_COVER_MUTATION_CLOSED"


class IntegrityError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"not an object: {path}")
    return value


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing artifact: {path}")
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return path.is_file() and path.stat().st_size == int(binding["bytes"]) and _sha256(path) == binding["sha256"]


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _task_statuses(text: str) -> dict[str, str]:
    rows = re.findall(r"^\| (T[^| ]+) \| ([^|]+) \|", text, flags=re.MULTILINE)
    result: dict[str, str] = {}
    for task, status in rows:
        result.setdefault(task.strip(), status.strip())
    return result


def _board_snapshot() -> dict[str, Any]:
    config = _load(CONFIG)
    statuses = _task_statuses(BOARD.read_text(encoding="utf-8"))
    selected: dict[str, str] = {}
    for task, expected in config["required_task_statuses"].items():
        actual = statuses.get(task, "MISSING")
        if expected == "ACTIVE_OR_DONE" and actual in {"In Progress", "Done"}:
            selected[task] = "ACTIVE_OR_DONE"
        elif expected == "TODO_OR_ACTIVE" and actual in {"Todo", "In Progress"}:
            selected[task] = "TODO_OR_ACTIVE"
        else:
            selected[task] = actual
    return {"path": _relative(BOARD), "statuses": selected, "canonical_sha256": _canonical_sha256(selected)}


def _truth_key(multimode: bool, rtl: bool) -> str:
    return f"multimode={str(multimode).lower()},rtl={str(rtl).lower()}"


def _placements(*sections: str) -> list[str]:
    return list(sections)


CLAIM_RULES = {
    "MM_OPENED_TASK_LOCAL_GAIN": {
        "placements": _placements("Results", "Discussion", "Supplement"),
        "blocking_gaps": ["not the strongest eligible deployable denominator", "opened task-local development evidence", "T6.24.5 not run"],
        "revocation": ["hide T6.20.4 zero-headroom result", "call the result frozen-benchmark or universal SOTA"],
        "final_wording": "Retain only as opened task-local multimode context alongside the mandatory strongest-baseline NO-GO.",
    },
    "MM_V1_CAUSAL_HEADROOM_NO_GO": {
        "placements": _placements("Results", "Discussion", "Supplement"),
        "blocking_gaps": ["relative improvement point is 0%", "paired 95% lower bound is 0%", "pilot/formal unaccessed"],
        "revocation": ["delete static-mixture exact MLD", "delete an unfavorable family", "use RTL evidence to rescue the gate"],
        "final_wording": "Mandatory negative: Phase-6D v1 did not enter implementation or formal because usable causal headroom over the strongest retained static baseline was zero.",
    },
    "MM_FROZEN_BENCHMARK_SOTA_BLOCKED": {
        "placements": _placements("AbstractBoundary", "Results", "Limitations", "Supplement"),
        "blocking_gaps": ["T6.24.5 Dropped", "no untouched formal result", "no simultaneous >10% LCB against every eligible baseline"],
        "revocation": ["write frozen-benchmark SOTA", "write universal GKP SOTA", "promote T6.18.3 into formal"],
        "final_wording": "Frozen-benchmark multimode SOTA is not established.",
    },
    "RTL_DETERMINISTIC_SIX_CYCLE_II1": {
        "placements": _placements("Abstract", "Methods", "Results", "Conclusion", "Supplement"),
        "blocking_gaps": ["physical transport/CDC/pins absent", "nanosecond board latency unmeasured"],
        "revocation": ["source hash drift", "nonzero CXXRTL mismatch", "latency or II contract failure"],
        "final_wording": "The exact single-mode converged RTL supports a six-cycle, II=1 pre-board architecture and passed one-million-cycle full-vector CXXRTL qualification.",
    },
    "RTL_ATOMIC_FAIL_CLOSED": {
        "placements": _placements("Abstract", "Methods", "Results", "Conclusion", "Supplement"),
        "blocking_gaps": ["two-state RTL scope", "physical CDC/metastability absent", "unbounded liveness not claimed"],
        "revocation": ["formal gate or mutation failure", "reintroduce raw config/trust bypass", "cross-top evidence stitching"],
        "final_wording": "The exact converged production top passes the stated pre-board atomic versioned-bank and fail-closed property contract.",
    },
    "RTL_POST_ROUTE_ESTIMATE": {
        "placements": _placements("Methods", "Results", "Supplement"),
        "blocking_gaps": ["open-source estimate, not vendor signoff", "observability fold may dominate critical path", "no physical bitstream or board"],
        "revocation": ["any seed fails 27 MHz", "source/netlist binding drift", "rename harness Fmax as bare-core or board speed"],
        "final_wording": "The exact qualified top passes three-seed 27 MHz open-source P&R; reported Fmax/resources are whole-harness pre-board estimates.",
    },
    "RTL_BOARD_MEASUREMENT_BLOCKED": {
        "placements": _placements("Methods", "Results", "Limitations", "Supplement"),
        "blocking_gaps": ["T6.9.2 Blocked", "no board correctness/latency/jitter/deadline/power", "transport/CDC unimplemented"],
        "revocation": ["populate any measured field from clock model/P&R/analytic power", "mark T6.9.2 Done without physical protocol"],
        "final_wording": "All physical-board fields remain null and unavailable.",
    },
    "RTL_SPEED_ADVANTAGE_PROHIBITED": {
        "placements": _placements("RelatedWork", "Limitations", "Supplement"),
        "blocking_gaps": ["no same-task board comparator", "current Fmax is observability-harness estimate", "external tasks are incommensurate"],
        "revocation": ["write faster/fastest/SOTA latency", "rank raw nanoseconds across code families"],
        "final_wording": "No FPGA speed advantage is claimed.",
    },
    "LEARNING_APPROXIMATION_DROPPED": {
        "placements": _placements("Methods", "Results", "Supplement"),
        "blocking_gaps": ["no authorized Phase-6D teacher", "no distillation/quantization/formal retention", "T6.26.2 Dropped"],
        "revocation": ["make CNN/student a primary contribution", "use legacy CNN to alter either lane verdict"],
        "final_wording": "CNN/student is absent from the primary Phase-6D result and retained only as a dropped/ablation status.",
    },
    "DUAL_LANE_NONTRANSFERABILITY": {
        "placements": _placements("TitleBoundary", "Abstract", "Methods", "Discussion", "Conclusion"),
        "blocking_gaps": ["current RTL does not execute multimode MLD", "lanes have different task signatures and metrics"],
        "revocation": ["add a weighted LER-latency score", "allow one lane to satisfy another lane gate", "draw a deployment arrow without compiler/equivalence proof"],
        "final_wording": "The two evidence lanes are parallel and connected by a contract pattern, not by a common performance denominator or current decoder deployment.",
    },
}


def _final_claims(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    config = _load(CONFIG)
    rows: list[dict[str, Any]] = []
    for parent in matrix["claims"]:
        claim_id = parent["claim_id"]
        rule = CLAIM_RULES[claim_id]
        rows.append({
            "claim_id": claim_id,
            "lane_id": parent["lane_id"],
            "parent_state": parent["state"],
            "final_disposition": config["claim_dispositions"][claim_id],
            "current_evidence": parent["current_result"],
            "blocking_gaps": rule["blocking_gaps"],
            "revocation_conditions": rule["revocation"],
            "paper_placements": rule["placements"],
            "final_wording": rule["final_wording"],
            "forbidden_wording": parent["forbidden_wording"],
            "parent_payload_sha256": _canonical_sha256(parent),
            "parent_evidence_keys": parent["evidence"],
        })
    return rows


def _direct_decisions(
    board: Mapping[str, Any], headroom: Mapping[str, Any], formal: Mapping[str, Any],
    long: Mapping[str, Any], hardware: Mapping[str, Any], matrix: Mapping[str, Any],
) -> dict[str, Any]:
    statuses = board["statuses"]
    boot = headroom["paired_bootstrap"]
    multimode_pass = (
        statuses["T6.24.5"] == "Done"
        and headroom["headroom_gate"]["passed"] is True
        and headroom["scope"]["formal_or_pilot_accessed"] is True
    )
    rtl_pass = (
        statuses["T6.25.4"] == "Done"
        and hardware["verdict"] == hardware_gate.VERDICT
        and hardware["gate_summary"] == {"passed": 16, "total": 16}
        and formal["verdict"] == FORMAL_VERDICT
        and long["verdict"] == long_gate.VERDICT
    )
    learning_state = "DROPPED_ABLATION_ONLY" if statuses["T6.26.2"] == "Dropped" else "OPTIONAL_EXTENSION_REQUIRES_SEPARATE_GATE"
    return {
        "MULTIMODE_SOFTWARE_ALGORITHM": {
            "required_task": "T6.24.5",
            "gate_passed": multimode_pass,
            "decision": "NO_GO",
            "direct_evidence": {
                "task_status": statuses["T6.24.5"],
                "headroom_verdict": headroom["verdict"],
                "strongest_baseline": headroom["strongest_development_baseline_selection"]["selected"],
                "baseline_p_L": boot["baseline_p_L"],
                "proposed_p_L": boot["proposed_p_L"],
                "relative_improvement_point": boot["relative_improvement_point"],
                "relative_improvement_lcb": boot["relative_improvement_lcb"],
                "formal_or_pilot_accessed": headroom["scope"]["formal_or_pilot_accessed"],
            },
            "matrix_context": matrix["lane_outcomes"]["MULTIMODE_SOFTWARE_ALGORITHM"],
            "permitted_claim": "opened task-local context plus mandatory v1 strongest-baseline NO-GO",
        },
        "SINGLE_MODE_DETERMINISTIC_RTL": {
            "required_task": "T6.25.4",
            "gate_passed": rtl_pass,
            "decision": "GO",
            "direct_evidence": {
                "task_status": statuses["T6.25.4"],
                "hardware_verdict": hardware["verdict"],
                "formal_verdict": formal["verdict"],
                "long_verdict": long["verdict"],
                "latency_cycles": hardware["clock_model"]["cycles"],
                "initiation_interval_cycles": hardware["clock_model"]["initiation_interval_cycles"],
                "minimum_fmax_mhz": hardware["fmax_mhz"]["minimum"],
                "wrapper_may_dominate_all": all(row["wrapper_may_dominate"] for row in hardware["critical_paths"]),
                "measured_fields": hardware["measured_fields"],
            },
            "matrix_context": matrix["lane_outcomes"]["SINGLE_MODE_DETERMINISTIC_RTL"],
            "permitted_claim": "exact-top deterministic/atomic/fail-closed pre-board hardware lane",
        },
        "LEARNED_APPROXIMATION_EXTENSION": {
            "required_task": "T6.26.2",
            "gate_passed": False,
            "decision": learning_state,
            "direct_evidence": {"task_status": statuses["T6.26.2"], "changes_overall_verdict": False},
            "matrix_context": matrix["lane_outcomes"]["LEARNED_APPROXIMATION_EXTENSION"],
            "permitted_claim": "dropped/ablation status only",
        },
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(section: str, row_id: str, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        rows.append({
            "section": section,
            "row_id": row_id,
            "lane_or_task": str(payload.get("lane_id", payload.get("required_task", payload.get("task_id", "META")))),
            "state_or_verdict": str(payload.get("final_disposition", payload.get("decision", payload.get("verdict", payload.get("status", "BINDING"))))),
            "payload_json": encoded,
            "payload_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        })

    for lane, decision in report["lane_decisions"].items():
        add("lane_decision", lane, {"lane_id": lane, **decision})
    for claim in report["final_claims"]:
        add("final_claim", str(claim["claim_id"]), claim)
    for task, state in report["phase7_handoff"]["tasks"].items():
        add("phase7_handoff", task, {"task_id": task, "status": state})
    for truth_key, verdict in report["verdict_truth_table"].items():
        add("truth_table", truth_key, {"truth_key": truth_key, "verdict": verdict})
    for key, binding in report["artifact_registry"].items():
        add("artifact", key, {"artifact_id": key, **binding})
    return rows


def _write_source_data(report: Mapping[str, Any]) -> int:
    rows = _source_rows(report)
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _source_data_matches(report: Mapping[str, Any]) -> bool:
    try:
        with SOURCE_DATA.open("r", encoding="utf-8", newline="") as stream:
            actual = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return False
    expected = _source_rows(report)
    return actual == expected and all(
        hashlib.sha256(row["payload_json"].encode()).hexdigest() == row["payload_sha256"]
        for row in actual
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    lanes = report["lane_decisions"]
    claim_rows = "\n".join(
        f"| `{row['claim_id']}` | `{row['final_disposition']}` | {', '.join(row['paper_placements'])} | {row['final_wording']} |"
        for row in report["final_claims"]
    )
    rtl = lanes["SINGLE_MODE_DETERMINISTIC_RTL"]["direct_evidence"]
    mm = lanes["MULTIMODE_SOFTWARE_ALGORITHM"]["direct_evidence"]
    return f"""# T6.26.4 Phase 6D 最终双 lane gate

## 终态

**`{report['verdict']}`**。

- Multimode software lane：`NO_GO`。T6.24.5=`{mm['task_status']}`，strongest baseline=`{mm['strongest_baseline']}`，baseline/proposed `p_L={mm['baseline_p_L']:.6f}/{mm['proposed_p_L']:.6f}`，relative point/LCB=`{mm['relative_improvement_point']:.1%}/{mm['relative_improvement_lcb']:.1%}`，pilot/formal 未访问。T6.18.3 只保留 opened task-local context。
- Single-mode RTL lane：`GO`。exact top 的 property、million-cycle CXXRTL 与三 seed P&R 均通过；6-cycle、II=1，最低 whole-harness Fmax=`{rtl['minimum_fmax_mhz']:.3f}` MHz。该 Fmax 受 observability fold 影响，不是 bare-core/board speed。
- Learning extension：`DROPPED_ABLATION_ONLY`，不参与真值表也不改变 overall verdict。

没有加权总分或跨 lane 补门。T6.9.2 继续 Blocked，board-measured、fastest、multimode-in-RTL 与 frozen-benchmark multimode SOTA 均关闭。

## Claim 移交

| Claim | Final disposition | Placement | Final wording |
| --- | --- | --- | --- |
{claim_rows}

## Phase 7

下一顺序任务是 T7.1.5：保留 T7.1.1--T7.1.4 historical snapshot，新增 Phase6D delta。T7.2.6 等待该 delta；T7.3.8 写 strongest-baseline negative answer，T7.3.9 写 contract bridge 而不是共同性能分母。
"""


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> list[dict[str, Any]]:
    config = _load(CONFIG)
    statuses = report["board_snapshot"]["statuses"]
    lanes = report["lane_decisions"]
    claims = {row["claim_id"]: row for row in report["final_claims"]}
    artifacts = report["artifact_registry"]
    mm = lanes["MULTIMODE_SOFTWARE_ALGORITHM"]
    rtl = lanes["SINGLE_MODE_DETERMINISTIC_RTL"]
    learning = lanes["LEARNED_APPROXIMATION_EXTENSION"]
    truth_key = _truth_key(mm["gate_passed"], rtl["gate_passed"])
    gates = [
        ("identity_config_and_verdict_domain", report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION and report["verdict"] in VERDICTS and report["verdict_truth_table"] == config["verdict_truth_table"]),
        ("all_direct_parent_verifiers_pass", report["parent_verification"] == {"matrix": True, "headroom": True, "formal": True, "long": True, "hardware": True}),
        ("board_states_exact_and_final_task_active_or_done", report["board_snapshot"]["statuses"] == config["required_task_statuses"] and report["board_snapshot"] == _board_snapshot()),
        ("multimode_direct_gate_is_no_go_without_substitution", mm["required_task"] == "T6.24.5" and mm["gate_passed"] is False and mm["decision"] == "NO_GO" and mm["direct_evidence"]["task_status"] == "Dropped" and mm["direct_evidence"]["headroom_verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM" and mm["direct_evidence"]["strongest_baseline"] == "static_mixture_exact_mld" and mm["direct_evidence"]["baseline_p_L"] == mm["direct_evidence"]["proposed_p_L"] and mm["direct_evidence"]["relative_improvement_point"] == mm["direct_evidence"]["relative_improvement_lcb"] == 0.0 and mm["direct_evidence"]["formal_or_pilot_accessed"] is False),
        ("rtl_direct_gate_is_go_from_t6_25_4", rtl["required_task"] == "T6.25.4" and rtl["gate_passed"] is True and rtl["decision"] == "GO" and rtl["direct_evidence"]["task_status"] == "Done" and rtl["direct_evidence"]["hardware_verdict"] == hardware_gate.VERDICT and rtl["direct_evidence"]["formal_verdict"] == FORMAL_VERDICT and rtl["direct_evidence"]["long_verdict"] == long_gate.VERDICT),
        ("rtl_cycle_fmax_wrapper_and_null_boundary_preserved", rtl["direct_evidence"]["latency_cycles"] == 6 and rtl["direct_evidence"]["initiation_interval_cycles"] == 1 and rtl["direct_evidence"]["minimum_fmax_mhz"] >= 27.0 and rtl["direct_evidence"]["wrapper_may_dominate_all"] is True and all(value is None for value in rtl["direct_evidence"]["measured_fields"].values())),
        ("learning_is_excluded_from_truth_table", learning["required_task"] == "T6.26.2" and learning["gate_passed"] is False and learning["decision"] == "DROPPED_ABLATION_ONLY" and learning["direct_evidence"] == {"task_status": "Dropped", "changes_overall_verdict": False}),
        ("truth_table_recomputes_go_rtl_only", truth_key == "multimode=false,rtl=true" and report["truth_key"] == truth_key and report["verdict"] == report["verdict_truth_table"][truth_key] == config["expected_current_verdict"] == "GO_RTL_ONLY"),
        ("no_weighted_score_or_cross_lane_rescue", report["global_weighted_score"] is None and report["decision_policy"] == "INDEPENDENT_BOOLEAN_LANES_NO_WEIGHTED_SCORE_NO_GATE_SUBSTITUTION"),
        ("all_ten_claim_dispositions_are_exact", set(claims) == set(config["claim_dispositions"]) and {key: value["final_disposition"] for key, value in claims.items()} == config["claim_dispositions"]),
        ("every_claim_has_evidence_gap_revocation_and_placement", all("current_evidence" in row and row["blocking_gaps"] and row["revocation_conditions"] and row["paper_placements"] and row["final_wording"] and row["forbidden_wording"] and len(row["parent_payload_sha256"]) == 64 and row["parent_evidence_keys"] for row in report["final_claims"])),
        ("multimode_context_negative_and_blocked_claims_all_visible", claims["MM_OPENED_TASK_LOCAL_GAIN"]["final_disposition"] == "RETAIN_CONTEXT_ONLY" and claims["MM_V1_CAUSAL_HEADROOM_NO_GO"]["final_disposition"] == "MANDATORY_NEGATIVE" and claims["MM_FROZEN_BENCHMARK_SOTA_BLOCKED"]["final_disposition"] == "BLOCKED"),
        ("rtl_restricted_claims_promoted_without_board_or_speed", all(claims[key]["final_disposition"] == "PROMOTED_RESTRICTED" for key in ("RTL_DETERMINISTIC_SIX_CYCLE_II1", "RTL_ATOMIC_FAIL_CLOSED", "RTL_POST_ROUTE_ESTIMATE")) and claims["RTL_BOARD_MEASUREMENT_BLOCKED"]["final_disposition"] == "BLOCKED" and claims["RTL_SPEED_ADVANTAGE_PROHIBITED"]["final_disposition"] == "PROHIBITED_POSITIVE"),
        ("learning_and_nontransferability_dispositions_are_exact", claims["LEARNING_APPROXIMATION_DROPPED"]["final_disposition"] == "DROPPED_ABLATION_ONLY" and claims["DUAL_LANE_NONTRANSFERABILITY"]["final_disposition"] == "MANDATORY_META_BOUNDARY"),
        ("board_task_stays_blocked_and_fields_null", statuses["T6.9.2"] == "Blocked" and all(value is None for value in claims["RTL_BOARD_MEASUREMENT_BLOCKED"]["current_evidence"].values())),
        ("phase7_handoff_is_exact_and_snapshot_is_immutable", report["phase7_handoff"]["tasks"] == config["phase7_handoff"] and report["phase7_handoff"]["next_task"] == "T7.1.5" and report["phase7_handoff"]["historical_snapshot_policy"] == "PRESERVE_T7_1_1_TO_T7_1_4_AND_ADD_DELTA" and report["phase7_handoff"]["old_bundle_publishable_without_delta"] is False),
        ("matrix_payloads_are_preserved", report["matrix_anchor"]["verdict"] == matrix_gate.VERDICT and report["matrix_anchor"]["gates"] == {"passed": 21, "total": 21} and report["matrix_anchor"]["mutations"] == {"detected": 21, "total": 21} and report["matrix_anchor"]["claim_payload_sha256"] == {row["claim_id"]: row["parent_payload_sha256"] for row in report["final_claims"]}),
        ("all_artifact_hash_bindings_are_live", all(len(row["sha256"]) == 64 and int(row["bytes"]) > 0 for row in artifacts.values()) and (not check_live_files or all(_live(row) for row in artifacts.values()))),
        ("lossless_source_data_is_bound_and_reconstructed", report["source_data"]["rows"] == len(_source_rows(report)) and len(report["source_data"]["sha256"]) == 64 and (not check_live_files or (_live(report["source_data"]) and _source_data_matches(report)))),
        ("publication_boundary_is_rtl_only_preboard", report["publication_boundary"] == {"phase6d_verdict": "GO_RTL_ONLY", "multimode_frozen_benchmark_sota": False, "multimode_opened_context_only": True, "single_mode_preboard_deterministic_atomic_fail_closed": True, "board_measured": False, "hardware_fastest_or_sota": False, "multimode_decoder_in_rtl": False, "learning_primary": False}),
    ]
    return [{"gate": name, "passed": bool(passed)} for name, passed in gates]


def semantic_mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def attempt(name: str, mutate: Any) -> None:
        candidate = copy.deepcopy(report)
        mutate(candidate)
        rejected = not all(row["passed"] for row in evaluate_gates(candidate, check_live_files=False))
        rows.append({"mutation": name, "rejected": rejected})

    def lane(candidate: dict[str, Any], lane_id: str) -> dict[str, Any]:
        return candidate["lane_decisions"][lane_id]

    def claim(candidate: dict[str, Any], claim_id: str) -> dict[str, Any]:
        return next(row for row in candidate["final_claims"] if row["claim_id"] == claim_id)

    attempt("promote_multimode_lane", lambda x: lane(x, "MULTIMODE_SOFTWARE_ALGORITHM").update(gate_passed=True, decision="GO"))
    attempt("fail_rtl_lane", lambda x: lane(x, "SINGLE_MODE_DETERMINISTIC_RTL").update(gate_passed=False, decision="NO_GO"))
    attempt("forge_overall_two_lane", lambda x: x.update(verdict="GO_TWO_LANE"))
    attempt("rewrite_truth_table", lambda x: x["verdict_truth_table"].update({"multimode=false,rtl=true": "GO_TWO_LANE"}))
    attempt("let_learning_change_verdict", lambda x: lane(x, "LEARNED_APPROXIMATION_EXTENSION")["direct_evidence"].update(changes_overall_verdict=True))
    attempt("add_weighted_score", lambda x: x.update(global_weighted_score=1.0))
    attempt("mark_t6_24_5_done", lambda x: x["board_snapshot"]["statuses"].update({"T6.24.5": "Done"}))
    attempt("drop_t6_25_4", lambda x: x["board_snapshot"]["statuses"].update({"T6.25.4": "Dropped"}))
    attempt("unblock_board_task", lambda x: x["board_snapshot"]["statuses"].update({"T6.9.2": "Done"}))
    attempt("invent_board_measurement", lambda x: lane(x, "SINGLE_MODE_DETERMINISTIC_RTL")["direct_evidence"]["measured_fields"].update(board_latency_ns=1.0))
    attempt("promote_speed_claim", lambda x: claim(x, "RTL_SPEED_ADVANTAGE_PROHIBITED").update(final_disposition="PROMOTED_RESTRICTED"))
    attempt("promote_multimode_sota_claim", lambda x: claim(x, "MM_FROZEN_BENCHMARK_SOTA_BLOCKED").update(final_disposition="PROMOTED_RESTRICTED"))
    attempt("promote_opened_context", lambda x: claim(x, "MM_OPENED_TASK_LOCAL_GAIN").update(final_disposition="PROMOTED_RESTRICTED"))
    attempt("hide_no_go_claim", lambda x: claim(x, "MM_V1_CAUSAL_HEADROOM_NO_GO").update(final_disposition="OMIT"))
    attempt("block_deterministic_claim", lambda x: claim(x, "RTL_DETERMINISTIC_SIX_CYCLE_II1").update(final_disposition="BLOCKED"))
    attempt("erase_blocking_gap", lambda x: claim(x, "RTL_POST_ROUTE_ESTIMATE").update(blocking_gaps=[]))
    attempt("erase_revocation", lambda x: claim(x, "DUAL_LANE_NONTRANSFERABILITY").update(revocation_conditions=[]))
    attempt("erase_placement", lambda x: claim(x, "LEARNING_APPROXIMATION_DROPPED").update(paper_placements=[]))
    attempt("corrupt_artifact_hash", lambda x: x["artifact_registry"]["implementation"].update(sha256="0"))
    attempt("delay_t7_1_5", lambda x: x["phase7_handoff"]["tasks"].update({"T7.1.5": "WAIT"}))
    attempt("allow_old_snapshot_publish", lambda x: x["phase7_handoff"].update(old_bundle_publishable_without_delta=True))
    attempt("forge_source_rows", lambda x: x["source_data"].update(rows=x["source_data"]["rows"] - 1))
    return {"detected": sum(int(row["rejected"]) for row in rows), "total": len(rows), "mutations": rows}


def build_report() -> dict[str, Any]:
    config = _load(CONFIG)
    matrix = _load(MATRIX_REPORT)
    headroom = _load(HEADROOM_REPORT)
    formal = _load(FORMAL_REPORT)
    long = _load(LONG_REPORT)
    hardware = _load(HARDWARE_REPORT)
    board = _board_snapshot()
    parent_verification = {
        "matrix": bool(matrix_gate.verify()),
        "headroom": bool(headroom_gate.verify()),
        "formal": bool(formal_gate.verify()),
        "long": bool(long_gate.verify()),
        "hardware": bool(hardware_gate.verify()),
    }
    lane_decisions = _direct_decisions(board, headroom, formal, long, hardware, matrix)
    multimode_pass = lane_decisions["MULTIMODE_SOFTWARE_ALGORITHM"]["gate_passed"]
    rtl_pass = lane_decisions["SINGLE_MODE_DETERMINISTIC_RTL"]["gate_passed"]
    truth_key = _truth_key(multimode_pass, rtl_pass)
    final_claims = _final_claims(matrix)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_verification": parent_verification,
        "board_snapshot": board,
        "artifact_registry": {key: _binding(path) for key, path in ARTIFACT_PATHS.items()},
        "matrix_anchor": {
            "verdict": matrix["verdict"],
            "gates": matrix["gate_summary"],
            "mutations": matrix["semantic_mutations"],
            "analysis_sha256": matrix["analysis_sha256"],
            "claim_payload_sha256": {row["claim_id"]: _canonical_sha256(row) for row in matrix["claims"]},
        },
        "lane_decisions": lane_decisions,
        "verdict_truth_table": config["verdict_truth_table"],
        "truth_key": truth_key,
        "global_weighted_score": None,
        "decision_policy": "INDEPENDENT_BOOLEAN_LANES_NO_WEIGHTED_SCORE_NO_GATE_SUBSTITUTION",
        "final_claims": final_claims,
        "phase7_handoff": {
            "next_task": "T7.1.5",
            "tasks": config["phase7_handoff"],
            "historical_snapshot_policy": "PRESERVE_T7_1_1_TO_T7_1_4_AND_ADD_DELTA",
            "old_bundle_publishable_without_delta": False,
        },
        "publication_boundary": {
            "phase6d_verdict": config["verdict_truth_table"][truth_key],
            "multimode_frozen_benchmark_sota": False,
            "multimode_opened_context_only": True,
            "single_mode_preboard_deterministic_atomic_fail_closed": True,
            "board_measured": False,
            "hardware_fastest_or_sota": False,
            "multimode_decoder_in_rtl": False,
            "learning_primary": False,
        },
        "verdict": config["verdict_truth_table"][truth_key],
    }
    rows = _write_source_data(report)
    report["source_data"] = {**_binding(SOURCE_DATA), "rows": rows}
    _atomic_text(MARKDOWN, _render_markdown(report))
    report["markdown"] = _binding(MARKDOWN)
    report["gates"] = evaluate_gates(report)
    audit = semantic_mutation_audit(report)
    report["semantic_mutations"] = {"detected": audit["detected"], "total": audit["total"]}
    report["semantic_mutation_results"] = audit["mutations"]
    report["gates"].append({
        "gate": "all_twenty_two_semantic_mutations_rejected",
        "passed": audit["detected"] == audit["total"] == int(config["semantic_mutation_count"]),
    })
    report["gate_summary"] = {"passed": sum(int(row["passed"]) for row in report["gates"]), "total": len(report["gates"])}
    if not all(row["passed"] for row in report["gates"]):
        report["verdict"] = "FAIL_CLOSED_PHASE6D_FINAL_GATE"
    canonical = copy.deepcopy(report)
    canonical.pop("generated_at_utc", None)
    report["analysis_sha256"] = _canonical_sha256(canonical)
    return report


def _validate(report: Mapping[str, Any], *, check_live_files: bool = True) -> None:
    _require(report["task_id"] == TASK_ID and report["schema_version"] == SCHEMA_VERSION, "identity drift")
    _require(report["verdict"] == "GO_RTL_ONLY", "wrong final verdict")
    _require(report["gate_summary"] == {"passed": 21, "total": 21}, "gate closure failed")
    _require(report["semantic_mutations"] == {"detected": 22, "total": 22}, "mutation closure failed")
    _require(report["gates"][:-1] == evaluate_gates(report, check_live_files=check_live_files), "stored gate mismatch")
    recomputed = semantic_mutation_audit(report)
    _require(report["semantic_mutation_results"] == recomputed["mutations"], "stored mutation mismatch")
    _require(all(row["passed"] for row in report["gates"]), "failed gate")
    if check_live_files:
        _require(_live(report["source_data"]), "source data binding mismatch")
        _require(_live(report["markdown"]), "markdown binding mismatch")


def verify() -> dict[str, Any]:
    report = _load(REPORT)
    _validate(report)
    canonical = copy.deepcopy(report)
    expected = canonical.pop("analysis_sha256")
    canonical.pop("generated_at_utc", None)
    _require(_canonical_sha256(canonical) == expected, "analysis hash mismatch")
    return {"verdict": report["verdict"], "gates": report["gate_summary"], "mutations": report["semantic_mutations"], "analysis_sha256": expected}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return 0
    report = build_report()
    _atomic_text(REPORT, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["verdict"] == "GO_RTL_ONLY":
        _validate(report)
    print(json.dumps({
        "verdict": report["verdict"],
        "truth_key": report["truth_key"],
        "lane_decisions": {key: value["decision"] for key, value in report["lane_decisions"].items()},
        "gates": report["gate_summary"],
        "mutations": report["semantic_mutations"],
        "next_task": report["phase7_handoff"]["next_task"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
