"""T9.2.1 causal digital-twin contract generator and verifier.

This module freezes interfaces and totality obligations.  It does not qualify
either physics backend, release a recovery codebook, report performance, or
claim hardware measurements.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from cnn_fpga.benchmark import phase9_scoped_claim_amendment
from cnn_fpga.benchmark import phase9_three_lane_protocol
from physics import phase9_twin_contract


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T9.2.1"
PROTOCOL_ID = "PHASE9-CAUSAL-TWIN-CONTRACT-V1"
CONFIG_SCHEMA_VERSION = "t9.2.1-causal-twin-contract-config-v1"
REPORT_SCHEMA_VERSION = "t9.2.1-causal-twin-contract-report-v1"
MANIFEST_SCHEMA_VERSION = "t9.2.1-factorized-totality-manifest-v1"
RELEASE_PIN_SCHEMA_VERSION = "t9.2.1-release-pin-v1"
VERDICT = "PASS_T9_2_1_CAUSAL_TWIN_CONTRACT_FROZEN"

DEFAULT_CONFIG = ROOT / "configs/phase9/t9_2_1_causal_twin_contract.json"
DEFAULT_RELEASE_PIN = ROOT / "configs/phase9/t9_2_1_release_pin.json"
IMPLEMENTATION = (
    ROOT / "cnn_fpga/benchmark/phase9_causal_twin_contract.py"
)
PHYSICS_IMPLEMENTATION = ROOT / "physics/phase9_twin_contract.py"
DEFAULT_REPORT = ROOT / "docs/t9_2_1_causal_twin_contract.json"
DEFAULT_TOTALITY_MANIFEST = (
    ROOT / "docs/t9_2_1_causal_twin_totality_manifest.json"
)
DEFAULT_SOURCE_DATA = (
    ROOT / "docs/t9_2_1_causal_twin_contract_source_data.csv"
)
DEFAULT_MARKDOWN = ROOT / "docs/phase9_causal_twin_contract.md"

PARENT_V1_REPORT = ROOT / "docs/t9_1_1_three_lane_protocol.json"
PARENT_CHILD_REPORT = ROOT / "docs/t9_1_5_scoped_claim_amendment.json"
PARENT_CHILD_RELEASE_PIN = (
    ROOT / "configs/phase9/t9_1_5_release_pin.json"
)

NAMESPACE_IDS = (
    "BACKEND_LATENT",
    "DEPLOYABLE_OBSERVED",
    "CONTROLLER_MEMORY",
    "EVALUATOR_TRUTH",
    "PROVENANCE",
)

GATE_IDS = (
    "G01_identity_and_protocol_only_scope_are_exact",
    "G02_t9_1_1_parent_is_live_semantic_and_byte_exact",
    "G03_t9_1_5_release_pin_is_live_semantic_and_byte_exact",
    "G04_config_generator_and_physics_implementations_are_live",
    "G05_exactly_five_namespaces_are_frozen_and_action_is_separate",
    "G06_namespace_field_schemas_match_the_runtime_contract_exactly",
    "G07_deployable_input_uses_observed_and_memory_allowlists_only",
    "G08_latent_evaluator_future_and_provenance_are_recursively_denied",
    "G09_causal_nodes_are_complete_namespace_bound_and_time_indexed",
    "G10_causal_edges_and_intervention_points_are_exact",
    "G11_forbidden_future_truth_and_reverse_causal_edges_are_absent",
    "G12_six_cycle_ii1_timing_and_old_or_new_sampling_are_frozen",
    "G13_composite_key_and_factorized_n_t_domains_are_exact",
    "G14_finite_cardinalities_and_state_invariants_are_exact",
    "G15_nominal_n_map_is_total_deterministic_and_enumerated",
    "G16_transition_t_map_is_total_deterministic_and_enumerated",
    "G17_totality_fingerprints_and_repeated_enumeration_are_stable",
    "G18_factorized_recurrence_is_not_a_partial_or_host_callback_map",
    "G19_fault_priority_order_is_exact_unique_and_fail_closed",
    "G20_invalid_and_integrity_faults_close_to_lkg_hold_or_reset",
    "G21_crc_version_stale_partial_ood_deadline_faults_are_covered",
    "G22_lkg_republish_is_monotonic_and_never_version_decrement",
    "G23_action_word_is_exactly_80_bits_with_exact_layout",
    "G24_reserved_codes_bounds_and_reason_error_outputs_are_total",
    "G25_base_lane_residual_is_structurally_bit_exact_zero",
    "G26_slow_path_can_nominate_complete_precompiled_packages_only",
    "G27_entry_patch_per_cycle_action_and_freeform_waveform_are_denied",
    "G28_package_commit_is_complete_atomic_versioned_crc_bound",
    "G29_hidden_teacher_is_training_only_and_not_deployable",
    "G30_future_suffix_invariance_and_observed_only_validation_are_frozen",
    "G31_provenance_is_audit_only_and_cannot_enter_policy_inputs",
    "G32_exactly_16_representative_probes_are_frozen_before_codebook",
    "G33_representative_probes_are_noncodebook_nonranking_nonperformance",
    "G34_probes_cover_nominal_boundary_reset_leakage_and_fault_interventions",
    "G35_iq_reset_ack_and_action_conditioning_have_physical_causal_semantics",
    "G36_all_physics_performance_codebook_frontend_claim_rank_fields_are_null",
    "G37_factorized_totality_manifest_is_canonical_live_and_exact",
    "G38_source_data_reconstructs_the_full_analysis_losslessly",
    "G39_markdown_is_canonical_exact_and_contains_all_atomic_ids",
    "G40_one_substantive_mutation_per_gate_is_replayed_and_rejected",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values are prohibited")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _relative(path),
        "bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _binding_live(binding: Mapping[str, Any]) -> bool:
    try:
        path_value = binding["path"]
        byte_value = binding["bytes"]
        sha_value = binding["sha256"]
        if (
            not isinstance(path_value, str)
            or not isinstance(byte_value, int)
            or isinstance(byte_value, bool)
            or byte_value <= 0
            or not _is_sha256(sha_value)
        ):
            return False
        path = ROOT / path_value
        return (
            path.is_file()
            and path.stat().st_size == byte_value
            and _sha256(path) == sha_value
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        path,
    )


def _safe(callback: Callable[[], Any]) -> bool:
    try:
        return bool(callback())
    except (
        AssertionError,
        AttributeError,
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def _parent_summary(
    config: Mapping[str, Any], *, verify_live: bool = True
) -> dict[str, Any]:
    parent_v1 = _load(PARENT_V1_REPORT)
    child = _load(PARENT_CHILD_REPORT)
    release_pin = _load(PARENT_CHILD_RELEASE_PIN)
    if verify_live:
        phase9_three_lane_protocol.verify_report(parent_v1)
        phase9_scoped_claim_amendment.verify_report(
            child,
            expected_analysis_sha256=release_pin["analysis_sha256"],
        )
    summary = {
        "t9_1_1": {
            "task_id": parent_v1["task_id"],
            "protocol_id": parent_v1["protocol_id"],
            "verdict": parent_v1["verdict"],
            "analysis_sha256": parent_v1["analysis_sha256"],
            "report": _binding(PARENT_V1_REPORT),
        },
        "t9_1_5": {
            "task_id": child["task_id"],
            "protocol_id": child["protocol_id"],
            "verdict": child["verdict"],
            "analysis_sha256": child["analysis_sha256"],
            "release_pin": _binding(PARENT_CHILD_RELEASE_PIN),
            "release_pin_payload": release_pin,
            "all_current_states_null": all(
                row["value"] is None
                and row["verdict"] is None
                and row["rank"] is None
                for row in child["current_states"]
            ),
        },
        "ordered_consumer_checks": {
            "verify_t9_1_1_live": verify_live,
            "match_t9_1_1_analysis_and_byte_binding": True,
            "verify_t9_1_5_with_canonical_release_pin": verify_live,
            "match_t9_1_5_release_pin_payload_and_byte_binding": True,
            "retain_all_parent_claims_as_null": True,
        },
    }
    if summary["t9_1_1"] != config["parent_contract"]["t9_1_1"]:
        raise ValueError("T9.1.1 parent byte/semantic binding drift")
    if summary["t9_1_5"] != config["parent_contract"]["t9_1_5"]:
        raise ValueError("T9.1.5 release-pin byte/semantic binding drift")
    return summary


def _runtime_snapshot() -> dict[str, Any]:
    audit = _jsonable(phase9_twin_contract.audit_contract())
    raw_manifest = _jsonable(
        phase9_twin_contract.factorized_map_manifest()
    )
    manifest = {
        "schema_id": raw_manifest["schema_id"],
        "nominal": {
            "expected_count": raw_manifest["nominal_expected_count"],
            "actual_count": raw_manifest["nominal_count"],
            "unique_key_count": raw_manifest["nominal_unique_keys"],
            "total": (
                raw_manifest["nominal_count"]
                == raw_manifest["nominal_expected_count"]
            ),
            "deterministic": raw_manifest["deterministic"],
            "fingerprint_sha256": raw_manifest["nominal_sha256"],
            "repeat_fingerprint_match": raw_manifest["deterministic"],
        },
        "transition": {
            "expected_count": raw_manifest[
                "transition_expected_count"
            ],
            "actual_count": raw_manifest["transition_count"],
            "unique_key_count": raw_manifest[
                "transition_unique_keys"
            ],
            "total": (
                raw_manifest["transition_count"]
                == raw_manifest["transition_expected_count"]
            ),
            "deterministic": raw_manifest["deterministic"],
            "fingerprint_sha256": raw_manifest["transition_sha256"],
            "repeat_fingerprint_match": raw_manifest["deterministic"],
        },
        "composition": {
            "expected_count": raw_manifest[
                "composition_expected_count"
            ],
            "actual_count": raw_manifest["composition_count"],
            "unique_key_count": raw_manifest[
                "composition_unique_keys"
            ],
            "nominal_signature_count": raw_manifest[
                "nominal_signature_count"
            ],
            "full_cartesian_key_count": raw_manifest[
                "full_cartesian_key_count"
            ],
            "quotient_is_lossless": raw_manifest[
                "composition_quotient_is_lossless"
            ],
            "equivalence": raw_manifest[
                "composition_equivalence"
            ],
            "equivalence_witness": raw_manifest[
                "nominal_equivalence_witness"
            ],
            "fingerprint_sha256": raw_manifest[
                "composition_sha256"
            ],
        },
        "combined_sha256": raw_manifest["combined_sha256"],
        "coverage_complete": raw_manifest["coverage_complete"],
        "unique_complete": raw_manifest["unique_complete"],
        "deterministic": raw_manifest["deterministic"],
        "base_residual_zero": raw_manifest["base_residual_zero"],
        "legal_discriminator_count": raw_manifest[
            "legal_discriminator_count"
        ],
        "phase_state_count": raw_manifest["phase_state_count"],
        "event_class_count": raw_manifest["event_class_count"],
        "valid_event_class_count": raw_manifest[
            "valid_event_class_count"
        ],
        "fsm_state_count": raw_manifest["fsm_state_count"],
        "valid_fsm_state_count": raw_manifest[
            "valid_fsm_state_count"
        ],
        "nominal_action_count": raw_manifest["nominal_action_count"],
        "reachable_fsm_count": raw_manifest["reachable_fsm_count"],
        "fsm_reachability_scope": raw_manifest[
            "fsm_reachability_scope"
        ],
        "reset_bfs_covered_count": raw_manifest[
            "reset_bfs_covered_count"
        ],
        "reset_bfs_max_distance": raw_manifest[
            "reset_bfs_max_distance"
        ],
        "reset_bfs_scope": raw_manifest["reset_bfs_scope"],
        "factorized_not_materialized_cartesian": True,
    }
    probes = []
    for row in phase9_twin_contract.representative_action_probes():
        probes.append(
            _jsonable(row.to_dict())
            if hasattr(row, "to_dict")
            else _jsonable(row)
        )
    return {
        "model_scope": _jsonable(phase9_twin_contract.MODEL_SCOPE),
        "claim_boundary": _jsonable(
            phase9_twin_contract.CLAIM_BOUNDARY
        ),
        "namespace_schemas": _jsonable(
            phase9_twin_contract.NAMESPACE_SCHEMAS
        ),
        "truth_provenance_denylist": _jsonable(
            phase9_twin_contract.TRUTH_PROVENANCE_DENYLIST
        ),
        "discriminator_layout": _jsonable(
            phase9_twin_contract.DISCRIMINATOR_LAYOUT
        ),
        "phase_frame_semantics": _jsonable(
            phase9_twin_contract.PHASE_FRAME_SEMANTICS
        ),
        "fsm_encoding": _jsonable(
            phase9_twin_contract.FSM_ENCODING
        ),
        "action_layout": _jsonable(phase9_twin_contract.ACTION_LAYOUT),
        "crc16_contract": _jsonable(
            phase9_twin_contract.CRC16_CONTRACT
        ),
        "action_sideband_contract": _jsonable(
            phase9_twin_contract.ACTION_SIDEBAND_CONTRACT
        ),
        "observation_envelope_boundary": _jsonable(
            phase9_twin_contract.OBSERVATION_ENVELOPE_BOUNDARY
        ),
        "integrity_flag_layout": _jsonable(
            phase9_twin_contract.INTEGRITY_FLAG_LAYOUT
        ),
        "slow_path_boundary": _jsonable(
            phase9_twin_contract.SLOW_PATH_BOUNDARY
        ),
        "fault_priority": list(phase9_twin_contract.FAULT_PRIORITY),
        "fault_response_witnesses": _jsonable(
            phase9_twin_contract.fault_response_witnesses()
        ),
        "nominal_cell_count": phase9_twin_contract.NOMINAL_CELL_COUNT,
        "transition_cell_count": (
            phase9_twin_contract.TRANSITION_CELL_COUNT
        ),
        "factorized_map_manifest_raw": raw_manifest,
        "factorized_map_manifest": manifest,
        "representative_probes": probes,
        "audit": audit,
    }


def _artifact_registry(config_path: Path) -> dict[str, Any]:
    return {
        "config": _binding(config_path),
        "implementation": _binding(IMPLEMENTATION),
        "physics_contract": _binding(PHYSICS_IMPLEMENTATION),
    }


def _build_totality_manifest(
    config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    parent_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "manifest_scope": (
            "FACTORIZED_INTERFACE_TOTALITY_NOT_CODEBOOK_OR_PHYSICS_QUALIFICATION"
        ),
        "frozen_at": config["frozen_at"],
        "parent_analyses": {
            "t9_1_1": parent_summary["t9_1_1"][
                "analysis_sha256"
            ],
            "t9_1_5": parent_summary["t9_1_5"][
                "analysis_sha256"
            ],
        },
        "domain_contract": copy.deepcopy(config["key_contract"]),
        "runtime_factorized_map_manifest": copy.deepcopy(
            runtime["factorized_map_manifest"]
        ),
        "runtime_audit": copy.deepcopy(runtime["audit"]),
        "nominal_cell_count": runtime["nominal_cell_count"],
        "transition_cell_count": runtime["transition_cell_count"],
        "representative_probe_count": len(
            runtime["representative_probes"]
        ),
        "performance_metrics": None,
        "codebook_id": None,
        "codebook_sha256": None,
        "backend_a_qualification": None,
        "backend_b_qualification": None,
        "analysis_sha256": "",
    }


def _manifest_analysis(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "analysis_sha256"
    }


def _expected_null_categories() -> tuple[str, ...]:
    return (
        "physics",
        "performance",
        "codebook",
        "frontend",
        "claim",
        "rank",
    )


def _all_leaf_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _all_leaf_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_leaf_values(item)
    else:
        yield value


def _valid_mutation_placeholder() -> dict[str, Any]:
    return {
        "count": len(GATE_IDS),
        "detected": len(GATE_IDS),
        "all_detected": True,
        "one_per_gate": True,
        "records": [
            {
                "mutation_id": f"M{index:02d}_placeholder",
                "target_gate": gate,
                "detected": True,
                "failed_gates": [gate],
            }
            for index, gate in enumerate(GATE_IDS, start=1)
        ],
    }


def _atomic_ids(report: Mapping[str, Any]) -> list[str]:
    values = list(GATE_IDS)
    values.extend(report["namespace_contract"]["exact_ids"])
    values.extend(
        row["node_id"] for row in report["causal_contract"]["nodes"]
    )
    values.extend(
        row["edge_id"] for row in report["causal_contract"]["edges"]
    )
    values.extend(
        row["probe_id"] for row in report["representative_probes"]
    )
    values.extend(
        f"{category}.{field}"
        for category, fields in report["current_null_state"].items()
        for field in fields
    )
    return values


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "task_id",
        "schema_version",
        "config_schema_version",
        "protocol_id",
        "contract_status",
        "frozen_at",
        "parent_contract",
        "parent_summary",
        "model_scope",
        "claim_boundary",
        "namespace_contract",
        "causal_contract",
        "key_contract",
        "fault_contract",
        "action_word_contract",
        "slow_path_contract",
        "isolation_contract",
        "representative_probe_contract",
        "representative_probes",
        "runtime_contract",
        "current_null_state",
        "protocol_only",
        "downstream_consumption_contract",
        "gate_definitions",
        "mutation_definitions",
        "artifact_registry",
        "totality_manifest",
        "semantic_mutation_audit",
    )
    return {
        field: copy.deepcopy(report[field])
        for field in fields
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(
        record_type: str,
        record_id: str,
        parent_id: str,
        payload: Any,
    ) -> None:
        canonical = _canonical_json(payload)
        rows.append(
            {
                "record_type": record_type,
                "record_id": record_id,
                "parent_id": parent_id,
                "payload_json": canonical,
                "payload_sha256": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
            }
        )

    for field, value in _analysis_payload(report).items():
        add("analysis_field", field, "analysis", value)
    for namespace_id, schema in report["namespace_contract"][
        "schemas"
    ].items():
        add("namespace", namespace_id, "namespace_contract", schema)
    for node in report["causal_contract"]["nodes"]:
        add("causal_node", node["node_id"], "causal_contract", node)
    for edge in report["causal_contract"]["edges"]:
        add("causal_edge", edge["edge_id"], "causal_contract", edge)
    for fault in report["fault_contract"]["priority"]:
        add(
            "fault_priority",
            fault["fault_id"],
            "fault_contract",
            fault,
        )
    for probe in report["representative_probes"]:
        add(
            "representative_probe",
            probe["probe_id"],
            "representative_probe_contract",
            probe,
        )
    for category, fields in report["current_null_state"].items():
        for field, value in fields.items():
            add(
                "typed_null",
                f"{category}.{field}",
                category,
                {"value": value, "type": "null"},
            )
    for definition in report["gate_definitions"]:
        add(
            "gate_definition",
            definition["gate_id"],
            "gates",
            definition,
        )
    for definition in report["mutation_definitions"]:
        add(
            "mutation_definition",
            definition["mutation_id"],
            definition["target_gate"],
            definition,
        )
    for record in report["semantic_mutation_audit"]["records"]:
        add(
            "mutation_result",
            record["mutation_id"],
            record["target_gate"],
            record,
        )
    for partition in ("nominal", "transition"):
        add(
            "totality_partition",
            partition,
            "totality_manifest",
            report["runtime_contract"]["factorized_map_manifest"][
                partition
            ],
        )
    return rows


def _write_source_data(
    report: Mapping[str, Any], path: Path
) -> None:
    rows = _source_rows(report)
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "record_type",
                "record_id",
                "parent_id",
                "payload_json",
                "payload_sha256",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _csv_lossless(
    report: Mapping[str, Any], path: Path
) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            actual = list(csv.DictReader(handle))
        expected = _source_rows(report)
        if actual != expected:
            return False
        for row in actual:
            if (
                hashlib.sha256(
                    row["payload_json"].encode("utf-8")
                ).hexdigest()
                != row["payload_sha256"]
            ):
                return False
            json.loads(row["payload_json"])
        reconstructed = {
            row["record_id"]: json.loads(row["payload_json"])
            for row in actual
            if row["record_type"] == "analysis_field"
        }
        return reconstructed == _analysis_payload(report)
    except (csv.Error, json.JSONDecodeError, OSError, KeyError):
        return False


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# T9.2.1 因果数字孪生接口与有限总递推合同",
        "",
        "## 结论",
        "",
        f"- verdict：`{report['verdict']}`；本 verdict 仅冻结接口合同，不是物理 backend、codebook、frontend、性能或硬件资格。",
        f"- parent：T9.1.1 `{report['parent_summary']['t9_1_1']['analysis_sha256']}`；T9.1.5 `{report['parent_summary']['t9_1_5']['analysis_sha256']}`。",
        f"- 五个 namespace：{', '.join(f'`{value}`' for value in report['namespace_contract']['exact_ids'])}；`ACTION_WORD` 是独立输出，不构成第六输入 namespace。",
        f"- totality：nominal N=`{report['runtime_contract']['nominal_cell_count']}`，transition T=`{report['runtime_contract']['transition_cell_count']}`，lossless composition quotient=`{report['runtime_contract']['factorized_map_manifest']['composition']['actual_count']}`，覆盖 raw Cartesian keys=`{report['runtime_contract']['factorized_map_manifest']['composition']['full_cartesian_key_count']}`。",
        "- action 为 80 bit；`discriminator-out -> action` 的未实测目标为 6 cycles、II=1。raw-IQ/frontend/trigger 均排除在该边界之外，`latency_measured=null`。",
        "- phase-frame 为 2-bit 四态 `(q_byte,p_byte)∈{0,128}²`；当前 two-uint8 RTL adapter 未资格。FSM 为 3-bit mode + 5-bit active-dwell counter 的新 Markov safety state；当前 six-counter FSM adapter 未资格。",
        "- FSM 的 68-state reachability 与 192-state reset/max-3 witness 只属于完整 syntactic T domain（含 reserved/未做 previous-receipt gate 的 event），不是 deployable-causal reachability 结论。",
        f"- gates/mutations：`{report['gate_summary']['passed']}/{report['gate_summary']['total']}`、`{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}`；Source Data `{report['source_data']['rows']}` rows。",
        "",
        "## 因果与权限边界",
        "",
        "- deployable 输入只能来自 `DEPLOYABLE_OBSERVED` 与 `CONTROLLER_MEMORY` 的白名单；`BACKEND_LATENT`、`EVALUATOR_TRUTH`、`PROVENANCE`、future suffix 与 hidden teacher 均结构性拒绝。",
        "- slow path 只能提名预编译、完整、version/CRC/provenance-bound package；禁止逐周期 action、单 entry patch、partial visibility、host callback 与 free-form waveform。",
        "- base lane residual 在序列化的 80-bit action word 内恒为 bit-exact zero；非零 residual 需要独立 amendment 和完整资格链。",
        "- previous action receipt 必须携带 canonical prior K，并由同一个 `F(K)` 重算后逐字段相等；CRC 正确但 recurrence 不可达的 receipt 会 fail closed。",
        "- T9.2.1 只冻结 I/Q 的非空、等长、最大 frame 与 signed-int64 container；sample rate、窗口、Q-format、rounding/saturation 等保持 null，由 T9.2.6 冻结。",
        "",
        "## 有限因子化总递推",
        "",
        "- composite key 逻辑视图为 `(bank_id, discriminator word, phase-frame, event class, leakage/reset FSM state)`；实现分成 nominal `N(bank, word)` 与 transition `T(phase, event, FSM, nominal-action-index)`。",
        "- 每个 legal cell 均有唯一 action/next-state/reason/error。invalid/OOD/CRC/version/stale/partial/deadline 等故障按冻结优先级闭合到 LKG hold 或 reset。",
        "- composition 与实际 `total_recurrence` 共用唯一 action 组装函数；1,024 个 raw nominal keys 到 12 个 signatures 的 class sizes、representatives 与 hashes 均写入 totality manifest。",
        "- LKG 恢复分为 fast hold/reset 与异步完整 image republish；republish 使用更高版本，禁止版本号倒退。",
        "",
        "## Causal graph 原子 ID",
        "",
        "- nodes：" + ", ".join(
            f"`{row['node_id']}`"
            for row in report["causal_contract"]["nodes"]
        ) + "。",
        "- edges：" + ", ".join(
            f"`{row['edge_id']}`"
            for row in report["causal_contract"]["edges"]
        ) + "。",
        "",
        "## Representative probes（非 codebook）",
        "",
    ]
    for probe in report["representative_probes"]:
        lines.append(
            f"- `{probe['probe_id']}`：{probe['intent']}；probe_only=`{str(probe['probe_only']).lower()}`，codebook_candidate=`{str(probe['codebook_candidate']).lower()}`。"
        )
    lines.extend(["", "## Gate 与 mutation", ""])
    mutation_by_gate = {
        row["target_gate"]: row
        for row in report["semantic_mutation_audit"]["records"]
    }
    for gate in GATE_IDS:
        row = mutation_by_gate[gate]
        lines.append(
            f"- `{gate}` = `{str(report['gates'][gate]).lower()}`；`{row['mutation_id']}` detected=`{str(row['detected']).lower()}`。"
        )
    lines.extend(["", "## Typed-null 结果边界", ""])
    for category, fields in report["current_null_state"].items():
        lines.append(
            f"- `{category}`："
            + ", ".join(f"`{category}.{field}`" for field in fields)
            + "（全部 `null`）。"
        )
    lines.extend(
        [
            "",
            "## 后续消费",
            "",
            "- T9.2.2/T9.2.3 实现两个独立 physics backend；T9.2.4 才可做双后端资格对拍；T9.3.3/T9.3.4 才可生成并枚举最终 trusted codebook。",
            "- 下游必须从 canonical release pin 接收 expected analysis SHA，先 live verify，再消费；报告自选路径、seal-only acceptance 和跨 lane promotion 均被禁止。",
            "",
        ]
    )
    lines.extend(["", "## Causal node/edge atomic IDs", ""])
    for node in report["causal_contract"]["nodes"]:
        lines.append(
            f"- node `{node['node_id']}`: namespace=`{node['namespace']}`, "
            f"time=`{node['time_index']}`."
        )
    for edge in report["causal_contract"]["edges"]:
        lines.append(
            f"- edge `{edge['edge_id']}`: `{edge['source']}` -> "
            f"`{edge['target']}`."
        )
    lines.append("")
    return "\n".join(lines)


def _gate_definitions(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return copy.deepcopy(config["gate_definitions"])


def _mutation_definitions(
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return copy.deepcopy(config["mutation_definitions"])


def evaluate_gates(
    report: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    check_live_files: bool = True,
    expected_parent_summary: Mapping[str, Any] | None = None,
    expected_runtime: Mapping[str, Any] | None = None,
    expected_artifacts: Mapping[str, Any] | None = None,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    config = _load(config_path)
    parent = (
        copy.deepcopy(dict(expected_parent_summary))
        if expected_parent_summary is not None
        else _parent_summary(config, verify_live=check_live_files)
    )
    runtime = (
        copy.deepcopy(dict(expected_runtime))
        if expected_runtime is not None
        else _runtime_snapshot()
    )
    artifacts = (
        copy.deepcopy(dict(expected_artifacts))
        if expected_artifacts is not None
        else _artifact_registry(config_path)
    )
    manifest = (
        copy.deepcopy(dict(expected_manifest))
        if expected_manifest is not None
        else _load(DEFAULT_TOTALITY_MANIFEST)
    )

    namespace_contract = report["namespace_contract"]
    causal = report["causal_contract"]
    key_contract = report["key_contract"]
    faults = report["fault_contract"]
    action = report["action_word_contract"]
    slow = report["slow_path_contract"]
    isolation = report["isolation_contract"]
    runtime_manifest = report["runtime_contract"][
        "factorized_map_manifest"
    ]
    probes = report["representative_probes"]

    def mutation_audit_ok() -> bool:
        audit = report["semantic_mutation_audit"]
        records = audit["records"]
        return (
            audit["count"] == len(GATE_IDS)
            and audit["detected"] == len(GATE_IDS)
            and audit["all_detected"] is True
            and audit["one_per_gate"] is True
            and len(records) == len(GATE_IDS)
            and [row["target_gate"] for row in records]
            == list(GATE_IDS)
            and len({row["mutation_id"] for row in records})
            == len(GATE_IDS)
            and all(row["detected"] is True for row in records)
        )

    def output_contract_ok() -> bool:
        basic = (
            report["totality_manifest"]["path"]
            == _relative(DEFAULT_TOTALITY_MANIFEST)
            and report["source_data"]["path"]
            == _relative(DEFAULT_SOURCE_DATA)
            and report["markdown"]["path"] == _relative(DEFAULT_MARKDOWN)
            and report["source_data"]["rows"]
            == len(_source_rows(report))
        )
        if not basic or not check_live_files:
            return basic
        return (
            report["totality_manifest"]
            == _binding(DEFAULT_TOTALITY_MANIFEST)
            and report["source_data"]
            == {
                **_binding(DEFAULT_SOURCE_DATA),
                "rows": len(_source_rows(report)),
            }
            and report["markdown"] == _binding(DEFAULT_MARKDOWN)
        )

    required_faults = set(config["fault_contract"][
        "required_fault_ids"
    ])
    actual_faults = {row["fault_id"] for row in faults["priority"]}
    probe_tags = {
        tag for probe in probes for tag in probe["coverage_tags"]
    }

    gates = {
        GATE_IDS[0]: _safe(
            lambda: report["task_id"] == TASK_ID
            and report["schema_version"] == REPORT_SCHEMA_VERSION
            and report["config_schema_version"] == CONFIG_SCHEMA_VERSION
            and report["protocol_id"] == PROTOCOL_ID
            and report["contract_status"]
            == "FROZEN_PRE_BACKEND_PRE_CODEBOOK_PROTOCOL_ONLY"
            and report["protocol_only"] is True
            and report["model_scope"]
            == config["model_scope"]
            == runtime["model_scope"]
            and report["claim_boundary"]
            == config["claim_boundary"]
            == runtime["claim_boundary"]
            and report["verdict"] in {None, VERDICT}
        ),
        GATE_IDS[1]: _safe(
            lambda: report["parent_contract"]["t9_1_1"]
            == config["parent_contract"]["t9_1_1"]
            and report["parent_summary"]["t9_1_1"]
            == parent["t9_1_1"]
            and parent["t9_1_1"]["analysis_sha256"]
            == "c88110375c358794339e72d672e4624871425fe480e5da091ddd1d6595255e18"
            and (
                not check_live_files
                or _binding_live(parent["t9_1_1"]["report"])
            )
        ),
        GATE_IDS[2]: _safe(
            lambda: report["parent_contract"]["t9_1_5"]
            == config["parent_contract"]["t9_1_5"]
            and report["parent_summary"]["t9_1_5"]
            == parent["t9_1_5"]
            and parent["t9_1_5"]["release_pin_payload"][
                "analysis_sha256"
            ]
            == parent["t9_1_5"]["analysis_sha256"]
            and parent["t9_1_5"]["all_current_states_null"] is True
            and (
                not check_live_files
                or _binding_live(parent["t9_1_5"]["release_pin"])
            )
        ),
        GATE_IDS[3]: _safe(
            lambda: report["artifact_registry"] == artifacts
            and (
                not check_live_files
                or all(_binding_live(row) for row in artifacts.values())
            )
        ),
        GATE_IDS[4]: _safe(
            lambda: tuple(namespace_contract["exact_ids"])
            == NAMESPACE_IDS
            and len(set(namespace_contract["exact_ids"])) == 5
            and namespace_contract["action_word_is_separate"] is True
            and "ACTION_WORD" not in namespace_contract["exact_ids"]
        ),
        GATE_IDS[5]: _safe(
            lambda: namespace_contract["schemas"]
            == runtime["namespace_schemas"]
            and namespace_contract["schemas_sha256"]
            == config["namespace_contract"]["schemas_sha256"]
            == _canonical_sha256(runtime["namespace_schemas"])
        ),
        GATE_IDS[6]: _safe(
            lambda: namespace_contract["deployable_input_namespaces"]
            == ["DEPLOYABLE_OBSERVED", "CONTROLLER_MEMORY"]
            and namespace_contract["deployable_validation"]
            == "EXACT_ALLOWLIST_AND_RECURSIVE_DENYLIST"
        ),
        GATE_IDS[7]: _safe(
            lambda: namespace_contract["forbidden_deployable_namespaces"]
            == [
                "BACKEND_LATENT",
                "EVALUATOR_TRUTH",
                "PROVENANCE",
            ]
            and namespace_contract["truth_provenance_denylist"]
            == runtime["truth_provenance_denylist"]
            and isolation["hidden_or_future_token_policy"]
            == "RECURSIVE_REJECT"
        ),
        GATE_IDS[8]: _safe(
            lambda: causal["nodes"] == config["causal_contract"]["nodes"]
            and len(causal["nodes"])
            == len({row["node_id"] for row in causal["nodes"]})
            and all(
                row["namespace"] in set(NAMESPACE_IDS) | {"ACTION_WORD"}
                and row["time_index"]
                in {"t-1", "t", "t+1", "audit-only"}
                for row in causal["nodes"]
            )
        ),
        GATE_IDS[9]: _safe(
            lambda: causal["edges"] == config["causal_contract"]["edges"]
            and causal["intervention_points"]
            == config["causal_contract"]["intervention_points"]
            and len(causal["edges"])
            == len({row["edge_id"] for row in causal["edges"]})
        ),
        GATE_IDS[10]: _safe(
            lambda: causal["forbidden_edges"]
            == config["causal_contract"]["forbidden_edges"]
            and all(
                row["allowed"] is False
                and row["failure"] == "STRUCTURAL_REJECT"
                for row in causal["forbidden_edges"]
            )
            and causal["future_suffix_policy"]
            == "DEPLOYABLE_PREFIX_OUTPUT_INVARIANT_TO_ANY_FUTURE_SUFFIX"
        ),
        GATE_IDS[11]: _safe(
            lambda: causal["timing"]
            == config["causal_contract"]["timing"]
            and causal["timing"]["discriminator_to_action_cycles"]
            == 6
            and causal["timing"]["initiation_interval_cycles"] == 1
            and causal["timing"]["inflight_bank_visibility"]
            == "EACH_ACCEPTED_REQUEST_OBSERVES_EXACTLY_OLD_OR_NEW_COMPLETE_IMAGE"
        ),
        GATE_IDS[12]: _safe(
            lambda: key_contract == config["key_contract"]
            and key_contract["logical_composite_key_fields"]
            == [
                "bank_id",
                "discriminator_word",
                "phase_frame",
                "event_class",
                "leakage_reset_fsm_state",
            ]
            and key_contract["nominal_n_fields"]
            == ["bank_id", "discriminator_word"]
            and key_contract["transition_t_fields"]
            == [
                "phase_frame",
                "event_class",
                "leakage_reset_fsm_state",
                "nominal_action_index",
            ]
            and key_contract["phase_frame_semantics"]
            == runtime["phase_frame_semantics"]
            and key_contract["fsm_encoding"]
            == runtime["fsm_encoding"]
            and key_contract["phase_frame_semantics"][
                "current_rtl_two_uint8_adapter_qualified"
            ]
            is False
            and key_contract["fsm_encoding"][
                "current_rtl_six_counter_adapter_qualified"
            ]
            is False
        ),
        GATE_IDS[13]: _safe(
            lambda: runtime["nominal_cell_count"]
            == key_contract["nominal_cell_count"]
            == 1024
            and runtime["transition_cell_count"]
            == key_contract["transition_cell_count"]
            == 131072
            and key_contract["legal_discriminator_count"] == 126
            and key_contract["fsm_state_count"] == 256
            and key_contract["valid_fsm_state_count"] == 192
            and runtime_manifest["valid_event_class_count"] == 13
            and runtime_manifest["valid_fsm_state_count"] == 192
            and key_contract["reachable_fsm_state_count"]
            == runtime_manifest["reachable_fsm_count"]
            == 68
            and key_contract["fsm_reachability_scope"]
            == runtime_manifest["fsm_reachability_scope"]
            == "SYNTACTIC_T_DOMAIN_ALL_16_EVENTS_AND_8_NOMINAL_ACTIONS_NOT_DEPLOYABLE_CAUSAL_REACHABILITY"
            and key_contract["reset_witness_covered_state_count"]
            == runtime_manifest["reset_bfs_covered_count"]
            == 192
            and key_contract["reset_witness_max_distance"]
            == runtime_manifest["reset_bfs_max_distance"]
            == 3
            and key_contract["reset_witness_scope"]
            == runtime_manifest["reset_bfs_scope"]
            == "SYNTACTIC_SUCCESS_ACK_THEN_HYSTERESIS_NOT_PREVIOUS_RECEIPT_GATED"
            and key_contract["finite"] is True
            and key_contract["reachable_state_invariants"]
        ),
        GATE_IDS[14]: _safe(
            lambda: runtime_manifest["nominal"]["expected_count"] == 1024
            and runtime_manifest["nominal"]["actual_count"] == 1024
            and runtime_manifest["nominal"]["unique_key_count"] == 1024
            and runtime_manifest["nominal"]["total"] is True
            and runtime_manifest["nominal"]["deterministic"] is True
        ),
        GATE_IDS[15]: _safe(
            lambda: runtime_manifest["transition"]["expected_count"]
            == 131072
            and runtime_manifest["transition"]["actual_count"]
            == 131072
            and runtime_manifest["transition"]["unique_key_count"]
            == 131072
            and runtime_manifest["transition"]["total"] is True
            and runtime_manifest["transition"]["deterministic"] is True
        ),
        GATE_IDS[16]: _safe(
            lambda: _is_sha256(
                runtime_manifest["nominal"]["fingerprint_sha256"]
            )
            and _is_sha256(
                runtime_manifest["transition"]["fingerprint_sha256"]
            )
            and runtime_manifest["nominal"]["repeat_fingerprint_match"]
            is True
            and runtime_manifest["transition"][
                "repeat_fingerprint_match"
            ]
            is True
            and runtime_manifest["composition"]["actual_count"]
            == runtime_manifest["composition"]["expected_count"]
            == runtime_manifest["composition"]["unique_key_count"]
            == 196608
            and runtime_manifest["composition"][
                "full_cartesian_key_count"
            ]
            == 16777216
            and runtime_manifest["composition"][
                "quotient_is_lossless"
            ]
            is True
            and runtime_manifest["composition"][
                "equivalence_witness"
            ]["mapped_nominal_key_count"]
            == 1024
            and runtime_manifest["composition"][
                "equivalence_witness"
            ]["signature_class_size_sum"]
            == 1024
            and _is_sha256(
                runtime_manifest["composition"][
                    "fingerprint_sha256"
                ]
            )
            and _is_sha256(
                runtime_manifest["composition"][
                    "equivalence_witness"
                ]["signatures_sha256"]
            )
            and report["runtime_contract"]["audit"]["all_passed"]
            is True
            and all(
                value is True
                for value in report["runtime_contract"]["audit"][
                    "checks"
                ].values()
            )
        ),
        GATE_IDS[17]: _safe(
            lambda: key_contract["factorization"]
            == "F(K)=T(phase,event,fsm,N(bank,word))"
            and key_contract["partial_map_allowed"] is False
            and key_contract["host_callback_allowed"] is False
            and key_contract["dont_care_output_allowed"] is False
            and runtime_manifest["factorized_not_materialized_cartesian"]
            is True
            and report["runtime_contract"]["audit"]["checks"][
                "future_truth_structurally_absent_from_decision_api"
            ]
            is True
            and report["runtime_contract"]["audit"]["checks"][
                "assembler_is_live_decision_path"
            ]
            is True
            and report["runtime_contract"]["audit"]["checks"][
                "composition_quotient_is_lossless_and_crc_enumerated"
            ]
            is True
        ),
        GATE_IDS[18]: _safe(
            lambda: faults["priority"]
            == config["fault_contract"]["priority"]
            and [row["fault_id"] for row in faults["priority"]]
            == runtime["fault_priority"]
            and [
                {
                    "fault_id": row["fault_id"],
                    "terminal": row["terminal"],
                    "reason_codes": row["reason_codes"],
                    "undefined_action": row["undefined_action"],
                }
                for row in faults["priority"]
            ]
            == [
                {
                    "fault_id": row["fault_id"],
                    "terminal": row["terminal"],
                    "reason_codes": row["reason_codes"],
                    "undefined_action": row["undefined_action"],
                }
                for row in runtime["fault_response_witnesses"]
            ]
            and [row["priority"] for row in faults["priority"]]
            == list(range(len(faults["priority"])))
            and len(actual_faults) == len(faults["priority"])
        ),
        GATE_IDS[19]: _safe(
            lambda: all(
                row["terminal"] in {"LKG_HOLD", "RESET"}
                and row["undefined_action"] is False
                for row in faults["priority"]
            )
            and faults["invalid_key_terminal"] == "LKG_HOLD"
            and faults["persistent_leakage_terminal"] == "RESET"
            and all(
                case["fallback"]
                and (case["hold"] or case["reset_request"])
                for row in runtime["fault_response_witnesses"]
                for case in row["cases"]
            )
        ),
        GATE_IDS[20]: _safe(
            lambda: required_faults <= actual_faults
            and {
                "INVALID_KEY",
                "OOD_WORD",
                "INPUT_CRC",
                "IMAGE_CRC",
                "IMAGE_SHA",
                "UNKNOWN_VERSION",
                "VERSION_MISMATCH",
                "ROLLBACK_VERSION",
                "STALE_PACKAGE",
                "PARTIAL_PACKAGE",
                "DEADLINE_MISS",
                "RESET_ACK_UNEXPECTED",
                "PERSISTENT_LEAKAGE",
            }
            <= actual_faults
        ),
        GATE_IDS[21]: _safe(
            lambda: faults["lkg_semantics"]["fast_path"]
            == "IMMEDIATE_HOLD_OR_RESET"
            and faults["lkg_semantics"]["republish"]
            == "ASYNC_COMPLETE_IMAGE_WITH_STRICTLY_HIGHER_VERSION"
            and faults["lkg_semantics"]["version_decrement_allowed"]
            is False
        ),
        GATE_IDS[22]: _safe(
            lambda: action == config["action_word_contract"]
            and action["layout"]
            == action["fields"]
            == runtime["action_layout"]
            and action["crc16_contract"]
            == runtime["crc16_contract"]
            and action["sideband_contract"]
            == runtime["action_sideband_contract"]
            and action["total_bits"] == 80
            and sum(row["bits"] for row in action["fields"]) == 80
        ),
        GATE_IDS[23]: _safe(
            lambda: action["undefined_output_allowed"] is False
            and action["reserved_code_terminal"]
            in {"LKG_HOLD", "RESET"}
            and action["bounded"] is True
            and {
                "reason_code",
                "error_flags",
                "next_phase_frame",
                "next_fsm_state",
            }
            <= {row["field"] for row in action["fields"]}
        ),
        GATE_IDS[24]: _safe(
            lambda: action["base_lane_residual"]["width_bits"] > 0
            and action["base_lane_residual"]["required_value"] == 0
            and action["base_lane_residual"]["nonzero_representable"]
            is False
            and slow["nonzero_residual_requires_amendment"] is True
        ),
        GATE_IDS[25]: _safe(
            lambda: slow["allowed_operation"]
            == "NOMINATE_COMPLETE_PRECOMPILED_TRUSTED_PACKAGE"
            and slow["per_cycle_action_authority"] is False
            and slow["candidate_must_be_complete"] is True
        ),
        GATE_IDS[26]: _safe(
            lambda: slow["single_entry_patch_allowed"] is False
            and slow["freeform_waveform_allowed"] is False
            and slow["host_callback_in_fast_path_allowed"] is False
            and slow["partial_active_visibility_allowed"] is False
        ),
        GATE_IDS[27]: _safe(
            lambda: slow["package_required_fields"]
            == config["slow_path_contract"]["package_required_fields"]
            and set(slow["package_required_fields"])
            == {
                "schema_id",
                "package_id",
                "bank_target",
                "version",
                "activation_epoch",
                "word_count",
                "crc32",
                "content_sha256",
                "provenance_sha256",
                "release_pin_sha256",
                "entries",
            }
            and slow["runtime_boundary"]
            == runtime["slow_path_boundary"]
            and slow["runtime_boundary"][
                "active_bank_write_authority"
            ]
            is False
            and slow["runtime_boundary"][
                "atomic_bank_integration_qualified"
            ]
            is False
            and slow["commit_semantics"]
            == "ATOMIC_OLD_OR_NEW_COMPLETE_IMAGE"
        ),
        GATE_IDS[28]: _safe(
            lambda: isolation["teacher_scope"]
            == "TRAINING_AND_PRIVILEGED_UPPER_BOUND_ONLY"
            and isolation["teacher_deployable"] is False
            and isolation["hidden_truth_deployable"] is False
        ),
        GATE_IDS[29]: _safe(
            lambda: isolation["future_suffix_invariance_test_required"]
            is True
            and isolation["observed_only_validator"]
            == "TYPED_ALLOWLIST_PLUS_RECURSIVE_DENYLIST"
            and isolation["field_name_blacklist_alone_is_sufficient"]
            is False
            and isolation["observation_envelope_boundary"]
            == runtime["observation_envelope_boundary"]
            and isolation["integrity_flag_layout"]
            == runtime["integrity_flag_layout"]
        ),
        GATE_IDS[30]: _safe(
            lambda: isolation["provenance_scope"] == "AUDIT_ONLY"
            and isolation["provenance_policy_input"] is False
            and isolation["provenance_action_input"] is False
        ),
        GATE_IDS[31]: _safe(
            lambda: probes == runtime["representative_probes"]
            and len(probes)
            == len({row["probe_id"] for row in probes})
            == 16
            and report["representative_probe_contract"][
                "probes_sha256"
            ]
            == config["representative_probe_contract"][
                "probes_sha256"
            ]
            == _canonical_sha256(probes)
        ),
        GATE_IDS[32]: _safe(
            lambda: all(
                row["probe_only"] is True
                and row["codebook_candidate"] is False
                and row["performance_evidence"] is False
                and row["ranking_evidence"] is False
                for row in probes
            )
            and report["representative_probe_contract"]["naming"]
            == "CONSERVATIVE_REPRESENTATIVE_ACTION_PROBE_NOT_CODEBOOK"
        ),
        GATE_IDS[33]: _safe(
            lambda: set(
                config["representative_probe_contract"][
                    "required_coverage_tags"
                ]
            )
            <= probe_tags
        ),
        GATE_IDS[34]: _safe(
            lambda: causal["physics_semantics"][
                "iq_emission_conditioned_on"
            ]
            == [
                "hidden_readout_state",
                "calibration_state",
                "measurement_backaction",
            ]
            and causal["physics_semantics"]["reset_ack_source"]
            == "OBSERVED_PHYSICAL_RESET_OUTCOME_NOT_TRUTH_LABEL"
            and causal["physics_semantics"][
                "transition_accepts_current_action"
            ]
            is True
            and causal["physics_semantics"]["crn_rule"]
            == "SHARE_EXOGENOUS_RANDOMNESS_ONLY_ACTIONS_MAY_DIVERGE_TRAJECTORIES"
        ),
        GATE_IDS[35]: _safe(
            lambda: tuple(report["current_null_state"])
            == _expected_null_categories()
            and report["current_null_state"]
            == config["current_null_state"]
            and all(
                value is None
                for value in _all_leaf_values(
                    report["current_null_state"]
                )
            )
        ),
        GATE_IDS[36]: _safe(
            lambda: report["totality_manifest"]
            == {
                **_binding(DEFAULT_TOTALITY_MANIFEST),
            }
            if check_live_files
            else report["totality_manifest"]["path"]
            == _relative(DEFAULT_TOTALITY_MANIFEST)
        )
        and _safe(
            lambda: manifest["analysis_sha256"]
            == _canonical_sha256(_manifest_analysis(manifest))
            and manifest
            == _build_totality_manifest(config, runtime, parent)
            | {
                "analysis_sha256": _canonical_sha256(
                    _manifest_analysis(
                        _build_totality_manifest(
                            config, runtime, parent
                        )
                    )
                )
            }
        ),
        GATE_IDS[37]: _safe(
            lambda: output_contract_ok()
            and (
                not check_live_files
                or _csv_lossless(report, DEFAULT_SOURCE_DATA)
            )
        ),
        GATE_IDS[38]: _safe(
            lambda: output_contract_ok()
            and (
                not check_live_files
                or DEFAULT_MARKDOWN.read_text(encoding="utf-8")
                == _render_markdown(report)
                and all(
                    f"`{identifier}`"
                    in DEFAULT_MARKDOWN.read_text(encoding="utf-8")
                    for identifier in _atomic_ids(report)
                )
            )
        ),
        GATE_IDS[39]: _safe(mutation_audit_ok),
    }
    if tuple(gates) != GATE_IDS:
        raise AssertionError("gate order drifted")
    return gates


def _semantic_mutation_audit(
    report: Mapping[str, Any],
    *,
    config_path: Path,
    parent: Mapping[str, Any],
    runtime: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []

    def attempt(
        mutation_id: str,
        target_gate: str,
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        candidate = copy.deepcopy(dict(report))
        mutate(candidate)
        gates = evaluate_gates(
            candidate,
            config_path=config_path,
            check_live_files=False,
            expected_parent_summary=parent,
            expected_runtime=runtime,
            expected_artifacts=artifacts,
            expected_manifest=manifest,
        )
        records.append(
            {
                "mutation_id": mutation_id,
                "target_gate": target_gate,
                "detected": gates[target_gate] is False,
                "failed_gates": [
                    key for key, passed in gates.items() if not passed
                ],
            }
        )

    attempts: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
        ("change_protocol_id", lambda x: x.update(protocol_id="BAD")),
        (
            "change_t9_1_1_analysis",
            lambda x: x["parent_summary"]["t9_1_1"].update(
                analysis_sha256="0" * 64
            ),
        ),
        (
            "change_t9_1_5_pin_payload",
            lambda x: x["parent_summary"]["t9_1_5"][
                "release_pin_payload"
            ].update(analysis_sha256="0" * 64),
        ),
        (
            "change_physics_hash",
            lambda x: x["artifact_registry"]["physics_contract"].update(
                sha256="0" * 64
            ),
        ),
        (
            "remove_namespace",
            lambda x: x["namespace_contract"]["exact_ids"].pop(),
        ),
        (
            "change_namespace_schema",
            lambda x: x["namespace_contract"]["schemas"][
                "DEPLOYABLE_OBSERVED"
            ].update(illegal_demo_field={"type": "truth"}),
        ),
        (
            "allow_latent_deployable_input",
            lambda x: x["namespace_contract"][
                "deployable_input_namespaces"
            ].append("BACKEND_LATENT"),
        ),
        (
            "remove_truth_deny_token",
            lambda x: x["namespace_contract"][
                "truth_provenance_denylist"
            ].pop(),
        ),
        (
            "remove_causal_node",
            lambda x: x["causal_contract"]["nodes"].pop(),
        ),
        (
            "reverse_causal_edge",
            lambda x: x["causal_contract"]["edges"][0].update(
                source="future_truth_t_plus_1"
            ),
        ),
        (
            "allow_future_truth_edge",
            lambda x: x["causal_contract"]["forbidden_edges"][0].update(
                allowed=True
            ),
        ),
        (
            "change_six_cycle_boundary",
            lambda x: x["causal_contract"]["timing"].update(
                discriminator_to_action_cycles=7
            ),
        ),
        (
            "remove_key_field",
            lambda x: x["key_contract"][
                "logical_composite_key_fields"
            ].pop(),
        ),
        (
            "reduce_nominal_cardinality",
            lambda x: x["key_contract"].update(nominal_cell_count=1023),
        ),
        (
            "mark_nominal_partial",
            lambda x: x["runtime_contract"][
                "factorized_map_manifest"
            ]["nominal"].update(total=False),
        ),
        (
            "drop_transition_cell",
            lambda x: x["runtime_contract"][
                "factorized_map_manifest"
            ]["transition"].update(actual_count=131071),
        ),
        (
            "forge_repeat_fingerprint",
            lambda x: x["runtime_contract"][
                "factorized_map_manifest"
            ]["nominal"].update(repeat_fingerprint_match=False),
        ),
        (
            "allow_host_callback",
            lambda x: x["key_contract"].update(
                host_callback_allowed=True
            ),
        ),
        (
            "swap_fault_priority",
            lambda x: x["fault_contract"]["priority"][0].update(
                priority=1
            ),
        ),
        (
            "allow_undefined_fault_action",
            lambda x: x["fault_contract"]["priority"][0].update(
                undefined_action=True
            ),
        ),
        (
            "remove_crc_fault",
            lambda x: x["fault_contract"]["priority"].__setitem__(
                slice(None),
                [
                    row
                    for row in x["fault_contract"]["priority"]
                    if row["fault_id"] != "INPUT_CRC"
                ],
            ),
        ),
        (
            "allow_version_decrement",
            lambda x: x["fault_contract"]["lkg_semantics"].update(
                version_decrement_allowed=True
            ),
        ),
        (
            "shorten_action_word",
            lambda x: x["action_word_contract"].update(total_bits=79),
        ),
        (
            "allow_reserved_output",
            lambda x: x["action_word_contract"].update(
                undefined_output_allowed=True
            ),
        ),
        (
            "enable_nonzero_residual",
            lambda x: x["action_word_contract"][
                "base_lane_residual"
            ].update(nonzero_representable=True),
        ),
        (
            "give_slow_path_action_authority",
            lambda x: x["slow_path_contract"].update(
                per_cycle_action_authority=True
            ),
        ),
        (
            "allow_entry_patch",
            lambda x: x["slow_path_contract"].update(
                single_entry_patch_allowed=True
            ),
        ),
        (
            "allow_partial_visibility",
            lambda x: x["slow_path_contract"].update(
                commit_semantics="PARTIAL_VISIBLE"
            ),
        ),
        (
            "deploy_hidden_teacher",
            lambda x: x["isolation_contract"].update(
                teacher_deployable=True
            ),
        ),
        (
            "drop_future_invariance",
            lambda x: x["isolation_contract"].update(
                future_suffix_invariance_test_required=False
            ),
        ),
        (
            "feed_provenance_to_policy",
            lambda x: x["isolation_contract"].update(
                provenance_policy_input=True
            ),
        ),
        (
            "remove_probe",
            lambda x: x["representative_probes"].pop(),
        ),
        (
            "promote_probe_to_codebook",
            lambda x: x["representative_probes"][0].update(
                codebook_candidate=True
            ),
        ),
        (
            "remove_fault_probe_coverage",
            lambda x: [
                row.update(
                    coverage_tags=[
                        tag
                        for tag in row["coverage_tags"]
                        if tag != "crc_version_stale_deadline"
                    ]
                )
                for row in x["representative_probes"]
            ],
        ),
        (
            "source_reset_ack_from_truth",
            lambda x: x["causal_contract"]["physics_semantics"].update(
                reset_ack_source="TRUTH_LABEL"
            ),
        ),
        (
            "fill_claim_null",
            lambda x: x["current_null_state"]["claim"].update(
                registered_best=False
            ),
        ),
        (
            "change_totality_manifest_path",
            lambda x: x["totality_manifest"].update(
                path="docs/alternate_manifest.json"
            ),
        ),
        (
            "change_source_row_count",
            lambda x: x["source_data"].update(
                rows=x["source_data"]["rows"] - 1
            ),
        ),
        (
            "change_markdown_path",
            lambda x: x["markdown"].update(
                path="docs/alternate_contract.md"
            ),
        ),
        (
            "forge_mutation_count",
            lambda x: x["semantic_mutation_audit"].update(
                detected=len(GATE_IDS) - 1
            ),
        ),
    ]
    if len(attempts) != len(GATE_IDS):
        raise AssertionError("mutation/gate cardinality drifted")
    for index, ((slug, mutation), gate) in enumerate(
        zip(attempts, GATE_IDS), start=1
    ):
        attempt(f"M{index:02d}_{slug}", gate, mutation)
    return {
        "count": len(records),
        "detected": sum(row["detected"] for row in records),
        "all_detected": all(row["detected"] for row in records),
        "one_per_gate": [
            row["target_gate"] for row in records
        ]
        == list(GATE_IDS),
        "records": records,
    }


def build_report() -> dict[str, Any]:
    config = _load(DEFAULT_CONFIG)
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("task_id") != TASK_ID
        or config.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("T9.2.1 config identity mismatch")
    if tuple(row["gate_id"] for row in config["gate_definitions"]) != GATE_IDS:
        raise ValueError("gate definition order mismatch")
    if [
        row["target_gate"] for row in config["mutation_definitions"]
    ] != list(GATE_IDS):
        raise ValueError("mutation/gate pairing mismatch")

    parent = _parent_summary(config, verify_live=True)
    runtime = _runtime_snapshot()
    if _canonical_sha256(runtime) != config[
        "runtime_contract_expected_sha256"
    ]:
        raise ValueError("physics runtime contract drifted from frozen config")
    artifacts = _artifact_registry(DEFAULT_CONFIG)

    manifest = _build_totality_manifest(config, runtime, parent)
    manifest["analysis_sha256"] = _canonical_sha256(
        _manifest_analysis(manifest)
    )
    _atomic_json(manifest, DEFAULT_TOTALITY_MANIFEST)

    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "contract_status": config["contract_status"],
        "frozen_at": config["frozen_at"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_contract": copy.deepcopy(config["parent_contract"]),
        "parent_summary": parent,
        "model_scope": copy.deepcopy(config["model_scope"]),
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
        "namespace_contract": {
            **copy.deepcopy(config["namespace_contract"]),
            "schemas": copy.deepcopy(runtime["namespace_schemas"]),
        },
        "causal_contract": copy.deepcopy(config["causal_contract"]),
        "key_contract": copy.deepcopy(config["key_contract"]),
        "fault_contract": copy.deepcopy(config["fault_contract"]),
        "action_word_contract": copy.deepcopy(
            config["action_word_contract"]
        ),
        "slow_path_contract": copy.deepcopy(
            config["slow_path_contract"]
        ),
        "isolation_contract": copy.deepcopy(
            config["isolation_contract"]
        ),
        "representative_probe_contract": copy.deepcopy(
            config["representative_probe_contract"]
        ),
        "representative_probes": copy.deepcopy(
            runtime["representative_probes"]
        ),
        "runtime_contract": runtime,
        "current_null_state": copy.deepcopy(
            config["current_null_state"]
        ),
        "protocol_only": True,
        "downstream_consumption_contract": copy.deepcopy(
            config["downstream_consumption_contract"]
        ),
        "gate_definitions": _gate_definitions(config),
        "mutation_definitions": _mutation_definitions(config),
        "artifact_registry": artifacts,
        "totality_manifest": _binding(DEFAULT_TOTALITY_MANIFEST),
        "source_data": {
            "path": _relative(DEFAULT_SOURCE_DATA),
            "rows": 0,
        },
        "markdown": {"path": _relative(DEFAULT_MARKDOWN)},
        "semantic_mutation_audit": _valid_mutation_placeholder(),
        "gates": {},
        "gate_summary": {},
        "verdict": None,
        "analysis_sha256": "",
    }
    report["source_data"]["rows"] = len(_source_rows(report))
    report["semantic_mutation_audit"] = _semantic_mutation_audit(
        report,
        config_path=DEFAULT_CONFIG,
        parent=parent,
        runtime=runtime,
        artifacts=artifacts,
        manifest=manifest,
    )
    offline_gates = evaluate_gates(
        report,
        config_path=DEFAULT_CONFIG,
        check_live_files=False,
        expected_parent_summary=parent,
        expected_runtime=runtime,
        expected_artifacts=artifacts,
        expected_manifest=manifest,
    )
    if not all(offline_gates.values()):
        raise ValueError(
            "T9.2.1 prepublication gates failed: "
            + ", ".join(
                key
                for key, passed in offline_gates.items()
                if not passed
            )
        )
    report["gates"] = offline_gates
    report["gate_summary"] = {
        "passed": len(GATE_IDS),
        "total": len(GATE_IDS),
        "failed": [],
    }
    report["verdict"] = VERDICT
    report["analysis_sha256"] = _canonical_sha256(
        _analysis_payload(report)
    )

    _write_source_data(report, DEFAULT_SOURCE_DATA)
    report["source_data"] = {
        **_binding(DEFAULT_SOURCE_DATA),
        "rows": len(_source_rows(report)),
    }
    _atomic_text(_render_markdown(report), DEFAULT_MARKDOWN)
    report["markdown"] = _binding(DEFAULT_MARKDOWN)
    report["totality_manifest"] = _binding(
        DEFAULT_TOTALITY_MANIFEST
    )

    live_gates = evaluate_gates(
        report,
        config_path=DEFAULT_CONFIG,
        check_live_files=True,
        expected_parent_summary=parent,
        expected_runtime=runtime,
        expected_artifacts=artifacts,
        expected_manifest=manifest,
    )
    failed = [key for key, passed in live_gates.items() if not passed]
    report["gates"] = live_gates
    report["gate_summary"] = {
        "passed": len(GATE_IDS) - len(failed),
        "total": len(GATE_IDS),
        "failed": failed,
    }
    report["verdict"] = (
        VERDICT if not failed else "FAIL_T9_2_1_CAUSAL_TWIN_CONTRACT"
    )
    if failed:
        raise ValueError(
            "T9.2.1 live gates failed: " + ", ".join(failed)
        )
    if report["analysis_sha256"] != _canonical_sha256(
        _analysis_payload(report)
    ):
        raise ValueError("analysis changed during publication")
    return report


def _release_pin(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RELEASE_PIN_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "analysis_sha256": report["analysis_sha256"],
        "config": _binding(DEFAULT_CONFIG),
        "implementation": _binding(IMPLEMENTATION),
        "physics_contract": _binding(PHYSICS_IMPLEMENTATION),
        "report": _binding(DEFAULT_REPORT),
        "totality_manifest": _binding(DEFAULT_TOTALITY_MANIFEST),
        "source_data": _binding(DEFAULT_SOURCE_DATA),
        "markdown": _binding(DEFAULT_MARKDOWN),
    }


def write_release() -> dict[str, Any]:
    report = build_report()
    _atomic_json(report, DEFAULT_REPORT)
    _atomic_json(_release_pin(report), DEFAULT_RELEASE_PIN)
    verify_report(
        DEFAULT_REPORT,
        expected_analysis_sha256=report["analysis_sha256"],
    )
    return report


def verify_report(
    report_or_path: Mapping[str, Any] | str | Path = DEFAULT_REPORT,
    *,
    expected_analysis_sha256: str | None = None,
) -> dict[str, bool]:
    if not isinstance(report_or_path, Mapping):
        if Path(report_or_path).resolve() != DEFAULT_REPORT.resolve():
            raise ValueError(
                "T9.2.1 verifier accepts only the canonical report path"
            )
        report = _load(DEFAULT_REPORT)
    else:
        report = copy.deepcopy(dict(report_or_path))
    config = _load(DEFAULT_CONFIG)
    parent = _parent_summary(config, verify_live=True)
    runtime = _runtime_snapshot()
    artifacts = _artifact_registry(DEFAULT_CONFIG)
    manifest = _load(DEFAULT_TOTALITY_MANIFEST)
    mutations = _semantic_mutation_audit(
        report,
        config_path=DEFAULT_CONFIG,
        parent=parent,
        runtime=runtime,
        artifacts=artifacts,
        manifest=manifest,
    )
    gates = evaluate_gates(
        report,
        config_path=DEFAULT_CONFIG,
        check_live_files=True,
        expected_parent_summary=parent,
        expected_runtime=runtime,
        expected_artifacts=artifacts,
        expected_manifest=manifest,
    )
    expected_summary = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "failed": [
            key for key, passed in gates.items() if not passed
        ],
    }
    canonical_report = _load(DEFAULT_REPORT)
    checks = {
        "canonical_report_payload": report == canonical_report,
        "identity": report["task_id"] == TASK_ID
        and report["protocol_id"] == PROTOCOL_ID
        and report["schema_version"] == REPORT_SCHEMA_VERSION,
        "parents_live": all(
            parent["ordered_consumer_checks"].values()
        ),
        "runtime_exact": report["runtime_contract"] == runtime
        and _canonical_sha256(runtime)
        == config["runtime_contract_expected_sha256"],
        "mutation_replay": report["semantic_mutation_audit"]
        == mutations,
        "all_gates": all(gates.values()),
        "gate_cache": report["gates"] == gates
        and report["gate_summary"] == expected_summary,
        "verdict": report["verdict"] == VERDICT,
        "analysis_sha256": report["analysis_sha256"]
        == _canonical_sha256(_analysis_payload(report)),
        "caller_expected_analysis": expected_analysis_sha256 is None
        or report["analysis_sha256"] == expected_analysis_sha256,
        "canonical_release_pin": _load(DEFAULT_RELEASE_PIN)
        == _release_pin(report),
        "totality_manifest": report["totality_manifest"]
        == _binding(DEFAULT_TOTALITY_MANIFEST)
        and manifest["analysis_sha256"]
        == _canonical_sha256(_manifest_analysis(manifest)),
        "source_data": report["source_data"]
        == {
            **_binding(DEFAULT_SOURCE_DATA),
            "rows": len(_source_rows(report)),
        }
        and _csv_lossless(report, DEFAULT_SOURCE_DATA),
        "markdown": report["markdown"] == _binding(DEFAULT_MARKDOWN)
        and DEFAULT_MARKDOWN.read_text(encoding="utf-8")
        == _render_markdown(report),
        "all_outcome_fields_null": all(
            value is None
            for value in _all_leaf_values(
                report["current_null_state"]
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "T9.2.1 verification failed: "
            + ", ".join(
                key for key, passed in checks.items() if not passed
            )
            + "; failed_gates="
            + repr(
                [key for key, passed in gates.items() if not passed]
            )
        )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--expected-analysis-sha256")
    arguments = parser.parse_args(argv)
    if arguments.write == arguments.verify:
        parser.error("choose exactly one of --write or --verify")
    if arguments.write:
        report = write_release()
        print(
            json.dumps(
                {
                    "verdict": report["verdict"],
                    "analysis_sha256": report["analysis_sha256"],
                    "gates": report["gate_summary"],
                    "mutations": {
                        key: report["semantic_mutation_audit"][key]
                        for key in ("count", "detected", "all_detected")
                    },
                    "source_rows": report["source_data"]["rows"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                verify_report(
                    DEFAULT_REPORT,
                    expected_analysis_sha256=(
                        arguments.expected_analysis_sha256
                    ),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
