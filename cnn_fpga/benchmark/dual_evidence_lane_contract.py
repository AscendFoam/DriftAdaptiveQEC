"""Build and verify the Phase-6D dual-evidence-lane contract.

The contract deliberately separates a multimode software algorithm claim from
the existing single-mode RTL implementation claim.  It also treats learned
models as replaceable approximations, never as an independent evidence lane.
All promotion rules fail closed and are exercised by semantic mutations.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.runtime import unified_execution_contract as execution_contract


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.20.2"
SCHEMA_VERSION = "t6.20.2-dual-evidence-lane-contract-v1"
VERDICT = "PASS_DUAL_LANE_CONTRACT_FROZEN"

DEFAULT_REPORT = ROOT / "docs" / "t6_20_2_dual_evidence_lane_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_20_2_dual_evidence_lane_contract_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "dual_evidence_lane_contract.md"

PRIMARY_LANE_IDS = {"MULTIMODE_SOFTWARE_ALGORITHM", "SINGLE_MODE_DETERMINISTIC_RTL"}
EXTENSION_LANE_ID = "LEARNED_APPROXIMATION_EXTENSION"
LANE_ROLES = {"PRIMARY_EVIDENCE_LANE", "OPTIONAL_DEPENDENT_EXTENSION"}
TASK_SIGNATURE_FIELDS = (
    "code_family",
    "modes_or_distance",
    "decision_target",
    "input_semantics",
    "history_horizon",
    "output_action",
    "noise_model",
    "observability",
    "online_privilege",
    "time_basis",
    "compute_budget",
    "precision",
    "evidence_level",
)
EVIDENCE_LAYERS = {
    "LITERATURE_ONLY",
    "OFFICIAL_SOURCE_PINNED",
    "OFFICIAL_CODE_REPRODUCTION",
    "PROJECT_NATIVE_DEVELOPMENT_SIMULATION",
    "UNTOUCHED_FORMAL_SIMULATION",
    "FIXED_POINT_INTEGER_REFERENCE",
    "CXXRTL_PREBOARD",
    "RTL_PROPERTY_PROOF",
    "POST_ROUTE_ESTIMATE",
    "BOARD_MEASURED",
}
CLAIM_STATES = {
    "CURRENT_RESTRICTED",
    "CONDITIONAL_FUTURE",
    "OPTIONAL_FUTURE",
    "BLOCKED_NULL",
}
FORBIDDEN_TRANSFER_IDS = {
    "FT-MM-LER-TO-CURRENT-RTL",
    "FT-RTL-LATENCY-TO-MULTIMODE",
    "FT-CNN-TO-ALGORITHM-SOTA",
    "FT-CNN-TO-RTL-SAFETY",
    "FT-PREBOARD-TO-BOARD-MEASURED",
    "FT-OPENED-DEVELOPMENT-TO-FORMAL",
    "FT-SOURCE-PRESENCE-TO-REPRODUCTION",
    "FT-TRUE-METRIC-CPD-TO-EXACT-ORACLE",
    "FT-CROSS-LANE-WEIGHTED-SCORE",
}

ARTIFACT_PATHS = {
    "multimode_baseline_registry": ROOT / "docs" / "multimode_strong_baseline_registry.md",
    "historical_claim_boundary": ROOT / "docs" / "t7_1_1_claim_evidence_boundary_matrix.json",
    "comparison_ontology": ROOT / "docs" / "t6_16_2_comparison_ontology.json",
    "unified_execution_source": ROOT / "cnn_fpga" / "runtime" / "unified_execution_contract.py",
    "domain_timescale_contract": ROOT / "docs" / "two_domains_three_timescales.json",
    "long_rtl_qualification": ROOT / "docs" / "t6_2_2_long_rtl_qualification.json",
    "production_rtl_audit": ROOT / "docs" / "t6_2_1_production_rtl_audit.json",
    "production_rtl": ROOT / "cnn_fpga" / "rtl" / "gkp_fast_path_production_top.sv",
    "official_lattice_gkp_source": ROOT / "third_party" / "LatticeAlgorithms.jl" / "src" / "gkp.jl",
    "implementation": Path(__file__).resolve(),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
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
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding["path"])
    return (
        path.is_file()
        and path.stat().st_size == binding.get("bytes")
        and _sha256(path) == binding.get("sha256")
    )


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", path)


def _signature(**values: str) -> dict[str, str]:
    if set(values) != set(TASK_SIGNATURE_FIELDS):
        raise ValueError("task signature fields are incomplete")
    return {field: values[field] for field in TASK_SIGNATURE_FIELDS}


def _lanes() -> list[dict[str, Any]]:
    return [
        {
            "lane_id": "MULTIMODE_SOFTWARE_ALGORITHM",
            "role": "PRIMARY_EVIDENCE_LANE",
            "code_family": "surface_square_gkp_multimode",
            "observable": "current analog syndrome plus causal observed-only prefix",
            "action": "logical-coset decision per correction round",
            "primary_metrics": ["per_round_p_L"],
            "secondary_metrics": [
                "p_X", "p_Y", "p_Z", "worst_window_LER", "CVaR95_window_LER",
                "adaptation_lag", "calibration", "runtime", "peak_memory", "deadline_miss",
            ],
            "timing_boundaries": ["software_source_to_decision", "software_update_compute"],
            "precision": "float64 reference plus preregistered approximate precision",
            "deployment_status": "SOFTWARE_ONLY_NOT_CURRENT_RTL",
            "required_evidence": [
                "OFFICIAL_CODE_REPRODUCTION", "UNTOUCHED_FORMAL_SIMULATION",
            ],
            "prohibited_claims": [
                "current RTL implements multimode logical-coset MLD",
                "single-mode six-cycle latency applies to this decoder",
                "universal or device-level SOTA",
            ],
            "primary_gate_ids": ["T6.24.5"],
            "task_signature": _signature(
                code_family="surface_square_gkp_multimode",
                modes_or_distance="surface_distance_d3_d5",
                decision_target="logical_coset",
                input_semantics="current_analog_syndrome_and_causal_prefix",
                history_horizon="prefix_through_current_round_no_future_suffix",
                output_action="logical_coset_action",
                noise_model="phase6d_frozen_multimode_suite",
                observability="observed_only",
                online_privilege="causal_budgeted",
                time_basis="per_round_software_source_to_decision",
                compute_budget="matched_cpu_core_memory_update_deadline",
                precision="float64_reference_plus_frozen_approximation",
                evidence_level="untouched_project_native_formal",
            ),
        },
        {
            "lane_id": "SINGLE_MODE_DETERMINISTIC_RTL",
            "role": "PRIMARY_EVIDENCE_LANE",
            "code_family": "single_mode_square_gkp_production_fast_path",
            "observable": "58-bit fixed-point syndrome/event/health input word",
            "action": "bounded frame/event action with version and reason code",
            "primary_metrics": [
                "latency_cycles", "initiation_interval_cycles", "atomic_old_or_new",
                "fail_closed_property",
            ],
            "secondary_metrics": [
                "undefined_action_count", "silent_overflow_count", "LUT", "FF", "BRAM",
                "DSP", "post_route_fmax", "estimated_power",
            ],
            "timing_boundaries": ["rtl_input_accept_to_action_valid", "rtl_initiation_interval"],
            "precision": "production RTL fixed-point formats and saturating arithmetic",
            "deployment_status": "ACTUAL_SINGLE_MODE_RTL_PREBOARD",
            "required_evidence": [
                "FIXED_POINT_INTEGER_REFERENCE", "CXXRTL_PREBOARD", "RTL_PROPERTY_PROOF",
                "POST_ROUTE_ESTIMATE",
            ],
            "prohibited_claims": [
                "multimode decoder latency",
                "board-measured latency or power before physical evidence",
                "LER SOTA inferred from deterministic timing",
            ],
            "primary_gate_ids": ["T6.25.4"],
            "frozen_fast_path": {
                "latency_cycles": 6,
                "initiation_interval_cycles": 1,
                "bank_semantics": "atomic_old_or_new",
                "integrity": ["CRC", "version", "CAS", "LKG_rollback"],
                "failure_semantics": "fail_closed_reason_coded_action",
            },
            "task_signature": _signature(
                code_family="single_mode_square_gkp_production_fast_path",
                modes_or_distance="one_oscillator_two_quadrature_phases",
                decision_target="bounded_fast_path_action",
                input_semantics="t5_5_1_58_bit_fixed_point_wire_word",
                history_horizon="bounded_local_fsm_and_versioned_bank_state",
                output_action="t5_5_1_fast_output_word",
                noise_model="digital_replay_and_fault_injection",
                observability="wire_observed_only",
                online_privilege="local_fail_closed_no_hidden_truth",
                time_basis="input_accept_to_action_valid_cycles",
                compute_budget="six_cycle_ii1_fixed_hardware_budget",
                precision="production_rtl_fixed_point",
                evidence_level="cxxrtl_property_postroute_preboard",
            ),
        },
        {
            "lane_id": EXTENSION_LANE_ID,
            "role": "OPTIONAL_DEPENDENT_EXTENSION",
            "code_family": "replaceable_cnn_or_student_approximation",
            "observable": "exactly the parent algorithm observed-only inputs",
            "action": "posterior/LLR/coset-probability/action approximation",
            "primary_metrics": [],
            "secondary_metrics": [
                "posterior_calibration", "action_agreement", "LER_retention",
                "worst_family_retention", "runtime", "memory", "quantization_error",
            ],
            "timing_boundaries": ["software_inference_only_unless_separately_integrated"],
            "precision": "frozen matched-budget floating or quantized student",
            "deployment_status": "OPTIONAL_NOT_AN_INDEPENDENT_PRIMARY_LANE",
            "required_evidence": ["UNTOUCHED_FORMAL_SIMULATION"],
            "prohibited_claims": [
                "independent algorithm SOTA",
                "replacement for exact decoder formal gate",
                "RTL safety or latency without actual RTL integration proof",
            ],
            "primary_gate_ids": [],
            "depends_on_lane": "MULTIMODE_SOFTWARE_ALGORITHM",
            "promotion_task": "T6.26.2",
            "failure_disposition": "DROPPED_TO_ABLATION",
            "task_signature": _signature(
                code_family="replaceable_cnn_or_student_approximation",
                modes_or_distance="inherits_frozen_multimode_parent",
                decision_target="posterior_llr_coset_probability_or_action",
                input_semantics="same_observed_only_parent_inputs",
                history_horizon="same_causal_parent_horizon",
                output_action="approximation_target_plus_uncertainty",
                noise_model="same_phase6d_frozen_multimode_suite",
                observability="observed_only_no_truth_teacher_at_inference",
                online_privilege="matched_budget_optional",
                time_basis="software_inference_boundary",
                compute_budget="matched_against_classical_approximation",
                precision="frozen_float_or_quantized_student",
                evidence_level="dependent_retention_evidence_only",
            ),
        },
    ]


def _interfaces() -> list[dict[str, Any]]:
    return [
        {
            "interface_id": "IF-MM-POSTERIOR-TO-COSET-ACTION",
            "source": "multimode observed-only posterior provider",
            "sink": "multimode exact/approximate logical-coset decoder",
            "schema": "phase6d_multimode_posterior_action_v1",
            "status": "PLANNED_SOFTWARE_INTERFACE",
            "deployment_implication": "SOFTWARE_ONLY",
        },
        {
            "interface_id": "IF-SLOW-PROPOSAL-TO-CANDIDATE-IMAGE",
            "source": "host estimator or optional learned provider",
            "sink": "schema-specific inactive-bank image adapter",
            "schema": "complete_candidate_parameter_image_with_provenance",
            "status": "CONTRACT_BRIDGE",
            "deployment_implication": "CONTRACT_REUSE_ONLY_REQUIRES_SCHEMA_EQUIVALENCE",
        },
        {
            "interface_id": "IF-CANDIDATE-IMAGE-TO-INACTIVE-BANK",
            "source": "single-mode production image validator",
            "sink": "inactive A/B bank",
            "schema": "T4.3.2_CRC_SHA_CAS_AB_BANK",
            "status": "CURRENT_SINGLE_MODE_PREBOARD",
            "deployment_implication": "SINGLE_MODE_RTL_ONLY",
        },
        {
            "interface_id": "IF-ATOMIC-COMMIT-TO-FAST-PATH",
            "source": "versioned active-bank commit",
            "sink": "six-cycle II=1 event/action datapath",
            "schema": "old_or_new_versioned_parameter_view",
            "status": "CURRENT_SINGLE_MODE_PREBOARD",
            "deployment_implication": "SINGLE_MODE_RTL_ONLY",
        },
    ]


def _claims() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "C-MM-FROZEN-BENCHMARK-LER-SOTA",
            "lane_id": "MULTIMODE_SOFTWARE_ALGORITHM",
            "state": "CONDITIONAL_FUTURE",
            "required_gate": "T6.24.5",
            "required_layers": ["OFFICIAL_CODE_REPRODUCTION", "UNTOUCHED_FORMAL_SIMULATION"],
            "safe_wording": "On the frozen Phase-6D multimode benchmark, the proposed observed-only decoder passes the preregistered relative-LER gate against every eligible strongest deployable baseline.",
            "forbidden_wording": ["universal GKP SOTA", "implemented by the current six-cycle RTL"],
        },
        {
            "claim_id": "C-RTL-HISTORICAL-PREBOARD-IMPLEMENTATION",
            "lane_id": "SINGLE_MODE_DETERMINISTIC_RTL",
            "state": "CURRENT_RESTRICTED",
            "required_gate": "T6.2.2+T6.19.1",
            "required_layers": ["FIXED_POINT_INTEGER_REFERENCE", "CXXRTL_PREBOARD", "POST_ROUTE_ESTIMATE"],
            "safe_wording": "The existing single-mode production RTL has a six-cycle, II=1 pre-board implementation supported by bit-accurate/CXXRTL replay and post-route estimates; this historical evidence does not yet constitute the Phase-6D property qualification.",
            "forbidden_wording": ["board measured", "multimode decoder latency", "formal proof complete"],
        },
        {
            "claim_id": "C-RTL-DETERMINISTIC-ATOMIC-FAIL-CLOSED",
            "lane_id": "SINGLE_MODE_DETERMINISTIC_RTL",
            "state": "CONDITIONAL_FUTURE",
            "required_gate": "T6.25.4",
            "required_layers": ["CXXRTL_PREBOARD", "RTL_PROPERTY_PROOF", "POST_ROUTE_ESTIMATE"],
            "safe_wording": "After T6.25.4 passes, the single-mode production RTL may be claimed to have a six-cycle, II=1 pre-board fast path with atomic versioned-bank and fail-closed properties under the stated CXXRTL/property/P&R boundary.",
            "forbidden_wording": ["board measured", "multimode decoder latency", "fastest FPGA decoder"],
        },
        {
            "claim_id": "C-ML-OPTIONAL-APPROXIMATION",
            "lane_id": EXTENSION_LANE_ID,
            "state": "OPTIONAL_FUTURE",
            "required_gate": "T6.26.2",
            "required_layers": ["UNTOUCHED_FORMAL_SIMULATION"],
            "safe_wording": "A matched-budget learned student may be retained as a replaceable approximation when it preserves the frozen classical decoder result and provides a separately demonstrated cost benefit.",
            "forbidden_wording": ["independent SOTA", "proves RTL safety", "reopens the formal candidate search"],
        },
        {
            "claim_id": "C-BOARD-MEASURED-PERFORMANCE",
            "lane_id": "SINGLE_MODE_DETERMINISTIC_RTL",
            "state": "BLOCKED_NULL",
            "required_gate": "T6.9.2",
            "required_layers": ["BOARD_MEASURED"],
            "safe_wording": "Board-measured latency, deadline, resource-power and transport results remain unavailable until the physical-board protocol is executed.",
            "forbidden_wording": ["measured latency", "measured power", "zero board deadline miss"],
            "current_value": None,
        },
    ]


def _forbidden_transfers() -> list[dict[str, str]]:
    rows = (
        ("FT-MM-LER-TO-CURRENT-RTL", "MULTIMODE_SOFTWARE_ALGORITHM", "SINGLE_MODE_DETERMINISTIC_RTL", "promote multimode LER to current RTL implementation", "different code family, action and precision", "CROSS_LANE_IMPLEMENTATION_PROMOTION"),
        ("FT-RTL-LATENCY-TO-MULTIMODE", "SINGLE_MODE_DETERMINISTIC_RTL", "MULTIMODE_SOFTWARE_ALGORITHM", "attach six-cycle latency to multimode decoder", "multimode compute graph is not the RTL datapath", "CROSS_LANE_TIMING_PROMOTION"),
        ("FT-CNN-TO-ALGORITHM-SOTA", EXTENSION_LANE_ID, "MULTIMODE_SOFTWARE_ALGORITHM", "use student agreement as the algorithm SOTA gate", "agreement cannot replace untouched LER comparison", "SURROGATE_TO_PRIMARY_PROMOTION"),
        ("FT-CNN-TO-RTL-SAFETY", EXTENSION_LANE_ID, "SINGLE_MODE_DETERMINISTIC_RTL", "use model accuracy as RTL safety evidence", "software accuracy does not prove atomicity or fail-closed behavior", "SURROGATE_TO_RTL_PROMOTION"),
        ("FT-PREBOARD-TO-BOARD-MEASURED", "SINGLE_MODE_DETERMINISTIC_RTL", "SINGLE_MODE_DETERMINISTIC_RTL", "rename CXXRTL/P&R estimate as board measurement", "physical bitstream, transport and instrument evidence are absent", "EVIDENCE_LAYER_PROMOTION"),
        ("FT-OPENED-DEVELOPMENT-TO-FORMAL", "MULTIMODE_SOFTWARE_ALGORITHM", "MULTIMODE_SOFTWARE_ALGORITHM", "reuse T6.18.3 opened outcomes as Phase-6D formal", "development and untouched formal evidence must be disjoint", "SPLIT_CONTAMINATION"),
        ("FT-SOURCE-PRESENCE-TO-REPRODUCTION", "MULTIMODE_SOFTWARE_ALGORITHM", "MULTIMODE_SOFTWARE_ALGORITHM", "treat pinned exact-MLD source as a reproduced baseline", "source audit is not execution or anchor agreement", "PROVENANCE_PROMOTION"),
        ("FT-TRUE-METRIC-CPD-TO-EXACT-ORACLE", "MULTIMODE_SOFTWARE_ALGORITHM", "MULTIMODE_SOFTWARE_ALGORITHM", "label true-metric CPD as exact decoding oracle", "metric knowledge does not perform logical-coset probability summation", "ORACLE_HIERARCHY_COLLAPSE"),
        ("FT-CROSS-LANE-WEIGHTED-SCORE", "MULTIMODE_SOFTWARE_ALGORITHM", "SINGLE_MODE_DETERMINISTIC_RTL", "combine LER and latency/safety into one weighted rank", "primary metrics and task signatures are incommensurate", "GLOBAL_SCORE_PROHIBITED"),
    )
    return [
        {
            "transfer_id": transfer_id,
            "source_lane": source,
            "target_lane": target,
            "trigger": trigger,
            "reason": reason,
            "rejection_code": code,
        }
        for transfer_id, source, target, trigger, reason, code in rows
    ]


def _parent_contract_snapshot() -> dict[str, Any]:
    ontology = _load(ARTIFACT_PATHS["comparison_ontology"])["ontology"]
    return {
        "task_signature_fields": list(ontology["task_signature_fields"]),
        "observed_schema_id": execution_contract.OBSERVED_SCHEMA_ID,
        "action_schema_id": execution_contract.ACTION_SCHEMA_ID,
        "map_lut_contract_id": execution_contract.MAP_LUT_CONTRACT_ID,
        "bank_contract_id": execution_contract.BANK_CONTRACT_ID,
        "cross_lane_raw_ranking": "PROHIBITED",
        "global_weighted_score": "PROHIBITED",
    }


def _validate_lane(row: Mapping[str, Any]) -> list[str]:
    common = {
        "lane_id", "role", "code_family", "observable", "action", "primary_metrics",
        "secondary_metrics", "timing_boundaries", "precision", "deployment_status",
        "required_evidence", "prohibited_claims", "primary_gate_ids", "task_signature",
    }
    allowed_extra = {
        "SINGLE_MODE_DETERMINISTIC_RTL": {"frozen_fast_path"},
        EXTENSION_LANE_ID: {"depends_on_lane", "promotion_task", "failure_disposition"},
    }.get(str(row.get("lane_id")), set())
    reasons: list[str] = []
    if set(row) != common | allowed_extra:
        reasons.append("lane_schema_mismatch")
    if row.get("role") not in LANE_ROLES:
        reasons.append("lane_role_invalid")
    signature = row.get("task_signature")
    if not isinstance(signature, Mapping) or tuple(signature) != TASK_SIGNATURE_FIELDS:
        reasons.append("task_signature_schema_mismatch")
    elif any(not isinstance(value, str) or not value for value in signature.values()):
        reasons.append("task_signature_empty")
    evidence = row.get("required_evidence", [])
    if not isinstance(evidence, list) or not set(evidence) <= EVIDENCE_LAYERS:
        reasons.append("evidence_layer_invalid")
    for field in ("observable", "action", "precision", "deployment_status"):
        if not isinstance(row.get(field), str) or not row[field]:
            reasons.append(f"{field}_empty")
    return reasons


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[tuple[str, str, str, Mapping[str, Any]]] = []
    for row in report["lanes"]:
        records.append(("lane", row["lane_id"], row["lane_id"], row))
    for row in report["claims"]:
        records.append(("claim", row["claim_id"], row["lane_id"], row))
    for row in report["interfaces"]:
        records.append(("interface", row["interface_id"], "", row))
    for row in report["forbidden_transfers"]:
        records.append(("forbidden_transfer", row["transfer_id"], row["source_lane"], row))
    for artifact_id, binding in report["artifact_registry"].items():
        records.append(("evidence_binding", artifact_id, "", binding))
    result: list[dict[str, str]] = []
    for record_type, record_id, lane_id, payload in records:
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result.append(
            {
                "record_type": record_type,
                "record_id": record_id,
                "lane_id": lane_id,
                "payload_json": payload_json,
                "canonical_sha256": hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            }
        )
    return result


def _write_source_data(report: Mapping[str, Any], path: Path) -> int:
    rows = _source_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("record_type", "record_id", "lane_id", "payload_json", "canonical_sha256"),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return len(rows)


def _csv_is_lossless(report: Mapping[str, Any], path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    expected = _source_rows(report)
    return actual == expected and all(
        row["canonical_sha256"] == hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
        for row in actual
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 6D 双证据 lane 合同",
        "",
        "> 本文档由 `T6.20.2` 的机器合同生成。两个 primary lane 并列但不互相补门；CNN/student 只是依赖性扩展。",
        "",
        "## Lane 冻结",
        "",
        "| Lane | 角色 | 对象 / action | 主指标 | 时间边界 | 部署状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["lanes"]:
        lines.append(
            f"| `{row['lane_id']}` | {row['role']} | {row['code_family']} / {row['action']} | "
            f"{', '.join(row['primary_metrics']) or 'none'} | {', '.join(row['timing_boundaries'])} | "
            f"{row['deployment_status']} |"
        )
    lines += ["", "## Claim 状态", "", "| Claim | Lane | 状态 | 升级门 | 安全措辞 |", "| --- | --- | --- | --- | --- |"]
    for row in report["claims"]:
        lines.append(
            f"| `{row['claim_id']}` | `{row['lane_id']}` | {row['state']} | `{row['required_gate']}` | "
            f"{row['safe_wording']} |"
        )
    lines += ["", "## Integration bridge", "", "这些接口只复用 schema/事务合同，不把 multimode 软件方法自动提升为当前 RTL 实现。", ""]
    for row in report["interfaces"]:
        lines.append(
            f"- `{row['interface_id']}`: {row['source']} → {row['sink']}；{row['deployment_implication']}。"
        )
    lines += ["", "## 禁止跨 lane 迁移", ""]
    for row in report["forbidden_transfers"]:
        lines.append(
            f"- `{row['transfer_id']}` / `{row['rejection_code']}`：{row['trigger']}。原因：{row['reason']}。"
        )
    lines += [
        "",
        "## 当前证据边界",
        "",
        "multimode LER SOTA 与 Phase-6D RTL property 主张仍是条件性未来 claim；现有 single-mode RTL 只能报告历史 bit-accurate/CXXRTL/P&R pre-board 证据；board-measured 字段保持 null。",
        "",
    ]
    return "\n".join(lines)


def evaluate_gates(report: Mapping[str, Any], *, check_live_files: bool = True) -> dict[str, bool]:
    lanes = {row["lane_id"]: row for row in report["lanes"]}
    claims = {row["claim_id"]: row for row in report["claims"]}
    interfaces = {row["interface_id"]: row for row in report["interfaces"]}
    transfers = {row["transfer_id"]: row for row in report["forbidden_transfers"]}
    source_path = ROOT / str(report["source_data"]["path"])
    markdown_path = ROOT / str(report["markdown"]["path"])
    expected_lane_ids = PRIMARY_LANE_IDS | {EXTENSION_LANE_ID}
    parent = report["parent_contract_snapshot"]
    mm = lanes.get("MULTIMODE_SOFTWARE_ALGORITHM", {})
    rtl = lanes.get("SINGLE_MODE_DETERMINISTIC_RTL", {})
    learned = lanes.get(EXTENSION_LANE_ID, {})
    current_board = claims.get("C-BOARD-MEASURED-PERFORMANCE", {})
    csv_ok = report["source_data"].get("rows") == len(_source_rows(report))
    markdown_contains_all = markdown_path.is_file() and all(
        f"`{key}`" in markdown_path.read_text(encoding="utf-8")
        for key in (*lanes, *claims, *interfaces, *transfers)
    )
    return {
        "G01_schema_and_unique_identity": report.get("schema_version") == SCHEMA_VERSION and len(lanes) == len(report["lanes"]) == 3 and set(lanes) == expected_lane_ids,
        "G02_closed_lane_schema_and_roles": all(not _validate_lane(row) for row in report["lanes"]) and {lanes[key]["role"] for key in PRIMARY_LANE_IDS} == {"PRIMARY_EVIDENCE_LANE"} and learned.get("role") == "OPTIONAL_DEPENDENT_EXTENSION",
        "G03_complete_distinct_task_signatures": parent.get("task_signature_fields") == list(TASK_SIGNATURE_FIELDS) and all(tuple(row["task_signature"]) == TASK_SIGNATURE_FIELDS for row in lanes.values()) and len({_canonical_sha256(row["task_signature"]) for row in lanes.values()}) == 3,
        "G04_multimode_is_software_LER_only": mm.get("primary_metrics") == ["per_round_p_L"] and mm.get("deployment_status") == "SOFTWARE_ONLY_NOT_CURRENT_RTL" and all("rtl" not in value.lower() and "cycle" not in value.lower() for value in mm.get("timing_boundaries", [])),
        "G05_rtl_is_single_mode_six_cycle_ii1": rtl.get("deployment_status") == "ACTUAL_SINGLE_MODE_RTL_PREBOARD" and rtl.get("frozen_fast_path", {}).get("latency_cycles") == 6 and rtl.get("frozen_fast_path", {}).get("initiation_interval_cycles") == 1 and "single_mode" in rtl.get("code_family", "") and "multimode" not in rtl.get("code_family", ""),
        "G06_learning_is_dependent_and_nonprimary": learned.get("depends_on_lane") == "MULTIMODE_SOFTWARE_ALGORITHM" and learned.get("primary_metrics") == [] and learned.get("primary_gate_ids") == [] and learned.get("failure_disposition") == "DROPPED_TO_ABLATION" and learned.get("deployment_status") == "OPTIONAL_NOT_AN_INDEPENDENT_PRIMARY_LANE",
        "G07_integration_bridge_does_not_imply_multimode_deployment": set(interfaces) == {"IF-MM-POSTERIOR-TO-COSET-ACTION", "IF-SLOW-PROPOSAL-TO-CANDIDATE-IMAGE", "IF-CANDIDATE-IMAGE-TO-INACTIVE-BANK", "IF-ATOMIC-COMMIT-TO-FAST-PATH"} and interfaces.get("IF-SLOW-PROPOSAL-TO-CANDIDATE-IMAGE", {}).get("deployment_implication") == "CONTRACT_REUSE_ONLY_REQUIRES_SCHEMA_EQUIVALENCE" and all(row.get("deployment_implication") != "MULTIMODE_CURRENT_RTL" for row in interfaces.values()),
        "G08_all_required_forbidden_transfers_are_closed": set(transfers) == FORBIDDEN_TRANSFER_IDS and len(transfers) == len(report["forbidden_transfers"]) and all(row.get("rejection_code") and row.get("reason") for row in transfers.values()),
        "G09_claim_states_and_upgrade_gates_are_closed": set(claims) == {"C-MM-FROZEN-BENCHMARK-LER-SOTA", "C-RTL-HISTORICAL-PREBOARD-IMPLEMENTATION", "C-RTL-DETERMINISTIC-ATOMIC-FAIL-CLOSED", "C-ML-OPTIONAL-APPROXIMATION", "C-BOARD-MEASURED-PERFORMANCE"} and set(report.get("evidence_layer_ontology", {})) == EVIDENCE_LAYERS and set(report.get("current_evidence_layers", [])) <= EVIDENCE_LAYERS and all(row.get("state") in CLAIM_STATES and row.get("required_gate") and set(row.get("required_layers", [])) <= EVIDENCE_LAYERS and row.get("forbidden_wording") for row in claims.values()) and claims.get("C-MM-FROZEN-BENCHMARK-LER-SOTA", {}).get("state") == "CONDITIONAL_FUTURE" and claims.get("C-RTL-HISTORICAL-PREBOARD-IMPLEMENTATION", {}).get("state") == "CURRENT_RESTRICTED" and claims.get("C-RTL-DETERMINISTIC-ATOMIC-FAIL-CLOSED", {}).get("state") == "CONDITIONAL_FUTURE" and claims.get("C-ML-OPTIONAL-APPROXIMATION", {}).get("state") == "OPTIONAL_FUTURE",
        "G10_board_measurement_remains_blocked_null": current_board.get("state") == "BLOCKED_NULL" and current_board.get("current_value", "missing") is None and current_board.get("required_layers") == ["BOARD_MEASURED"] and "BOARD_MEASURED" not in report.get("current_evidence_layers", []) and "RTL_PROPERTY_PROOF" not in report.get("current_evidence_layers", []),
        "G11_artifact_bindings_are_complete_and_live": set(report["artifact_registry"]) == set(ARTIFACT_PATHS) and all(len(row.get("sha256", "")) == 64 and isinstance(row.get("bytes"), int) and row["bytes"] > 0 for row in report["artifact_registry"].values()) and (not check_live_files or all(_live(row) for row in report["artifact_registry"].values())),
        "G12_source_data_is_lossless_and_hash_bound": csv_ok and source_path.is_file() and (not check_live_files or (_live(report["source_data"]) and _csv_is_lossless(report, source_path))),
        "G13_human_contract_contains_every_atomic_id": markdown_contains_all and (not check_live_files or _live(report["markdown"])),
        "G14_parent_execution_and_ontology_ids_are_frozen": parent == {"task_signature_fields": list(TASK_SIGNATURE_FIELDS), "observed_schema_id": "route-a-observed-syndrome-v1", "action_schema_id": "t5.5.1-fast-output-v1", "map_lut_contract_id": "T421-PHASE-CONDITIONED-MAP-LUT-Q9.12-V1", "bank_contract_id": "T432-CRC-SHA-CAS-AB-BANK-V1", "cross_lane_raw_ranking": "PROHIBITED", "global_weighted_score": "PROHIBITED"},
        "G15_no_cross_lane_score_or_gate_substitution": report.get("ranking_policy") == {"within_exact_task_signature_only": True, "cross_lane_raw_ranking": "PROHIBITED", "global_weighted_score": "PROHIBITED", "one_lane_cannot_satisfy_another_lane_gate": True},
        "G16_one_substantive_mutation_per_gate_fails_closed": report["semantic_mutation_audit"].get("count") == report["semantic_mutation_audit"].get("detected") == 16 and len(report["semantic_mutation_audit"].get("cases", [])) == 16,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def lane(value: dict[str, Any], lane_id: str) -> dict[str, Any]:
        return next(row for row in value["lanes"] if row["lane_id"] == lane_id)

    def claim(value: dict[str, Any], claim_id: str) -> dict[str, Any]:
        return next(row for row in value["claims"] if row["claim_id"] == claim_id)

    def interface(value: dict[str, Any], interface_id: str) -> dict[str, Any]:
        return next(row for row in value["interfaces"] if row["interface_id"] == interface_id)

    def attempt(name: str, target: str, change: Any) -> None:
        candidate = copy.deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
        change(candidate)
        try:
            rejected = not evaluate_gates(candidate, check_live_files=False)[target]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": target, "rejected": rejected})

    attempt("duplicate_lane_identity", "G01_schema_and_unique_identity", lambda x: x["lanes"].append(copy.deepcopy(x["lanes"][0])))
    attempt("remove_required_lane_field", "G02_closed_lane_schema_and_roles", lambda x: lane(x, "MULTIMODE_SOFTWARE_ALGORITHM").pop("precision"))
    attempt("drop_task_signature_field", "G03_complete_distinct_task_signatures", lambda x: lane(x, "MULTIMODE_SOFTWARE_ALGORITHM")["task_signature"].pop("compute_budget"))
    attempt("attach_rtl_cycles_to_multimode", "G04_multimode_is_software_LER_only", lambda x: lane(x, "MULTIMODE_SOFTWARE_ALGORITHM")["timing_boundaries"].append("rtl_latency_cycles"))
    attempt("promote_rtl_to_multimode", "G05_rtl_is_single_mode_six_cycle_ii1", lambda x: lane(x, "SINGLE_MODE_DETERMINISTIC_RTL").update(code_family="multimode_surface_gkp_rtl"))
    attempt("make_student_primary", "G06_learning_is_dependent_and_nonprimary", lambda x: lane(x, EXTENSION_LANE_ID).update(primary_metrics=["per_round_p_L"], primary_gate_ids=["T6.24.5"]))
    attempt("bridge_implies_multimode_deployment", "G07_integration_bridge_does_not_imply_multimode_deployment", lambda x: interface(x, "IF-SLOW-PROPOSAL-TO-CANDIDATE-IMAGE").update(deployment_implication="MULTIMODE_CURRENT_RTL"))
    attempt("remove_cnn_to_rtl_guard", "G08_all_required_forbidden_transfers_are_closed", lambda x: x["forbidden_transfers"].remove(next(row for row in x["forbidden_transfers"] if row["transfer_id"] == "FT-CNN-TO-RTL-SAFETY")))
    attempt("premature_multimode_sota", "G09_claim_states_and_upgrade_gates_are_closed", lambda x: claim(x, "C-MM-FROZEN-BENCHMARK-LER-SOTA").update(state="CURRENT_RESTRICTED"))
    attempt("forge_board_measurement", "G10_board_measurement_remains_blocked_null", lambda x: claim(x, "C-BOARD-MEASURED-PERFORMANCE").update(state="CURRENT_RESTRICTED", current_value={"latency_ns": 1.0}))
    attempt("forge_artifact_hash", "G11_artifact_bindings_are_complete_and_live", lambda x: x["artifact_registry"]["production_rtl"].update(sha256="0"))
    attempt("forge_source_row_count", "G12_source_data_is_lossless_and_hash_bound", lambda x: x["source_data"].update(rows=x["source_data"]["rows"] - 1))
    attempt("disconnect_human_contract", "G13_human_contract_contains_every_atomic_id", lambda x: x["markdown"].update(path="docs/nonexistent_dual_lane_contract.md"))
    attempt("rename_parent_action_schema", "G14_parent_execution_and_ontology_ids_are_frozen", lambda x: x["parent_contract_snapshot"].update(action_schema_id="unfrozen-action"))
    attempt("enable_cross_lane_score", "G15_no_cross_lane_score_or_gate_substitution", lambda x: x["ranking_policy"].update(global_weighted_score="ALLOWED"))
    attempt("forge_mutation_count", "G16_one_substantive_mutation_per_gate_fails_closed", lambda x: x.update(semantic_mutation_audit={"count": 16, "detected": 15, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "paper_argument", "parent_contract_snapshot", "evidence_layer_ontology",
        "current_evidence_layers", "ranking_policy", "artifact_registry", "lanes",
        "interfaces", "claims", "forbidden_transfers", "source_data", "markdown",
        "semantic_mutation_audit", "gates", "verdict",
    )
    return {field: report[field] for field in fields}


def build_report(
    source_data: Path = DEFAULT_SOURCE_DATA,
    markdown: Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_argument": "Two co-primary but non-substitutable evidence lanes: multimode software simulation competes on frozen-benchmark LER, while the actual single-mode RTL establishes deterministic, atomic and fail-closed pre-board properties; CNN/student modules remain replaceable approximations.",
        "parent_contract_snapshot": _parent_contract_snapshot(),
        "evidence_layer_ontology": {
            "LITERATURE_ONLY": "external paper result without local execution",
            "OFFICIAL_SOURCE_PINNED": "audited upstream code at a fixed revision",
            "OFFICIAL_CODE_REPRODUCTION": "pinned upstream implementation rerun against anchors",
            "PROJECT_NATIVE_DEVELOPMENT_SIMULATION": "opened development/headroom evidence",
            "UNTOUCHED_FORMAL_SIMULATION": "preregistered split opened once after freeze",
            "FIXED_POINT_INTEGER_REFERENCE": "bit-accurate integer software reference",
            "CXXRTL_PREBOARD": "cycle-accurate compiled RTL before a physical board",
            "RTL_PROPERTY_PROOF": "exhaustive/bounded property evidence on actual RTL",
            "POST_ROUTE_ESTIMATE": "synthesis/place-and-route timing/resource estimate",
            "BOARD_MEASURED": "same-bitstream physical-board measurement",
        },
        "current_evidence_layers": [
            "LITERATURE_ONLY", "OFFICIAL_SOURCE_PINNED",
            "PROJECT_NATIVE_DEVELOPMENT_SIMULATION", "FIXED_POINT_INTEGER_REFERENCE",
            "CXXRTL_PREBOARD", "POST_ROUTE_ESTIMATE",
        ],
        "ranking_policy": {
            "within_exact_task_signature_only": True,
            "cross_lane_raw_ranking": "PROHIBITED",
            "global_weighted_score": "PROHIBITED",
            "one_lane_cannot_satisfy_another_lane_gate": True,
        },
        "artifact_registry": {key: _binding(path) for key, path in ARTIFACT_PATHS.items()},
        "lanes": _lanes(),
        "interfaces": _interfaces(),
        "claims": _claims(),
        "forbidden_transfers": _forbidden_transfers(),
    }
    _atomic_text(_render_markdown(report), markdown)
    rows = _write_source_data(report, source_data)
    report["source_data"] = {**_binding(source_data), "rows": rows}
    report["markdown"] = _binding(markdown)
    report["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "failed": [key for key, passed in report["gates"].items() if not passed],
    }
    report["verdict"] = VERDICT if not report["gate_summary"]["failed"] else "FAIL_DUAL_LANE_CONTRACT"
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
        "gates": value.get("gates") == gates and all(gates.values()),
        "verdict": value.get("verdict") == VERDICT,
        "analysis_hash": value.get("analysis_sha256") == _canonical_sha256(_analysis_payload(value)),
    }
    if not all(checks.values()):
        raise ValueError(f"T6.20.2 verification failed: {[key for key, passed in checks.items() if not passed]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        verify_report(path=args.report)
        print(json.dumps({"verified": _relative(args.report), "verdict": VERDICT}, ensure_ascii=False))
        return 0
    report = build_report(args.source_data, args.markdown)
    _atomic_json(report, args.report)
    verify_report(report, args.report)
    print(
        json.dumps(
            {
                "output": _relative(args.report),
                "lanes": len(report["lanes"]),
                "forbidden_transfers": len(report["forbidden_transfers"]),
                "gates": report["gate_summary"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
