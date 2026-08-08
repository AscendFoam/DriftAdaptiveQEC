"""T6.5.2 validation for the Route-A unified execution contract."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cnn_fpga.runtime.bit_accurate_hardware_reference import decode_input_word
from cnn_fpga.runtime.unified_execution_contract import (
    ACTION_SCHEMA_ID,
    BANK_CONTRACT_ID,
    BUDGET_CONTRACT_ID,
    CADENCE_CONTRACT_ID,
    CONTRACT_ID,
    DEADLINE_CONTRACT_ID,
    DEPLOYABLE_METHOD_IDS,
    MAP_LUT_CONTRACT_ID,
    OBSERVED_SCHEMA_ID,
    ORACLE_METHOD_ID,
    ORACLE_SCHEMA_ID,
    ContractViolation,
    ExecutionAccountingRecord,
    MatchedBudget,
    MethodManifest,
    OracleTruthPacket,
    assert_accounting_conforms,
    assert_method_conforms,
    canonical_method_manifests,
    contract_sha256,
    contract_snapshot,
    oracle_method_manifest,
    validate_observed_mapping_for_deployable,
    validate_wire_roundtrip,
)


TASK_ID = "T6.5.2"
PROTOCOL_ID = "ROUTE-A-UNIFIED-EXECUTION-CONFORMANCE-V1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_5_2_unified_execution_contract.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_5_2_unified_execution_contract_source_data.csv"

SOURCE_PATHS = (
    "cnn_fpga/runtime/unified_execution_contract.py",
    "cnn_fpga/runtime/bit_accurate_hardware_reference.py",
    "cnn_fpga/runtime/parametric_map_lut.py",
    "cnn_fpga/runtime/atomic_parameter_bank.py",
    "cnn_fpga/runtime/three_timescale_cadence.py",
    "docs/t4_2_1_parametric_map_lut_validation.json",
    "docs/t4_3_1_three_timescale_cadence_validation.json",
    "docs/t4_3_2_atomic_parameter_bank_validation.json",
    "docs/t5_5_1_bit_accurate_hardware_reference.json",
    "docs/t6_2_2_long_rtl_qualification.json",
)

COMMON_MUTATION_FIELDS = (
    "online_privilege",
    "input_schema_id",
    "action_schema_id",
    "map_lut_contract_id",
    "bank_contract_id",
    "cadence_contract_id",
    "budget_contract_id",
    "deadline_contract_id",
    "benchmark_deployability",
    "current_rtl_compatibility",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observed_payload() -> dict[str, object]:
    return {
        "schema_id": OBSERVED_SCHEMA_ID,
        "trace_id": "contract-trace-0001",
        "cycle_index": 4000,
        "syndrome_code": 731,
        "syndrome_x": "e",
        "syndrome_z": "g",
        "quadrature_phase_bit": 1,
        "ood_score_code": 23,
        "parameter_age_code": 3999,
        "reset_ack": False,
        "observation_valid": True,
        "deadline_ok": True,
    }


def _oracle_payload() -> dict[str, object]:
    return {
        "schema_id": ORACLE_SCHEMA_ID,
        "trace_id": "contract-trace-0001",
        "cycle_index": 4000,
        "latent_displacement_q": 0.13,
        "latent_displacement_p": -0.21,
        "latent_mean_q": 0.08,
        "latent_mean_p": -0.04,
        "latent_sigma_q": 0.31,
        "latent_sigma_p": 0.37,
        "latent_correlation": 0.22,
        "logical_x": False,
        "logical_z": True,
        "regime_label": "smooth",
    }


def _capture_violation(function: object, *args: object) -> dict[str, object]:
    try:
        function(*args)  # type: ignore[misc]
    except ContractViolation as exc:
        return {"rejected": True, **exc.to_dict()}
    except ValueError as exc:
        return {
            "rejected": True,
            "code": "lower_level_integrity_rejection",
            "field": "packed_wire_word",
            "expected": "valid CRC/canonical word",
            "actual": type(exc).__name__,
            "method_id": None,
        }
    return {
        "rejected": False,
        "code": "not_rejected",
        "field": "unknown",
        "expected": "ContractViolation",
        "actual": "accepted",
        "method_id": None,
    }


def _method_mismatch_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for manifest in canonical_method_manifests():
        for field in COMMON_MUTATION_FIELDS:
            mutated = replace(manifest, **{field: f"MUTATED-{field}"})
            result = _capture_violation(assert_method_conforms, mutated)
            rows.append(
                {
                    "method_id": manifest.method_id,
                    "mutated_field": field,
                    "expected_code": "method_contract_mismatch",
                    **result,
                }
            )
    return rows


def _schema_rejection_matrix() -> list[dict[str, object]]:
    base = _observed_payload()
    cases: list[tuple[str, str, dict[str, object]]] = []
    for method_id in DEPLOYABLE_METHOD_IDS:
        payload = dict(base)
        payload["hidden_drift_state"] = "forbidden"
        cases.append((method_id, "hidden_truth_key", payload))
    missing = dict(base)
    missing.pop("deadline_ok")
    cases.append(("proposed_route_a", "missing_field", missing))
    extra = dict(base)
    extra["debug"] = 1
    cases.append(("proposed_route_a", "extra_field", extra))
    wrong_type = dict(base)
    wrong_type["syndrome_code"] = True
    cases.append(("proposed_route_a", "bool_as_integer", wrong_type))
    out_of_range = dict(base)
    out_of_range["syndrome_code"] = 1024
    cases.append(("proposed_route_a", "code_out_of_range", out_of_range))
    wrong_schema = dict(base)
    wrong_schema["schema_id"] = ORACLE_SCHEMA_ID
    cases.append(("proposed_route_a", "oracle_schema_on_deployable", wrong_schema))
    results = []
    for method_id, case_id, payload in cases:
        result = _capture_violation(validate_observed_mapping_for_deployable, method_id, payload)
        results.append({"method_id": method_id, "case_id": case_id, **result})
    return results


def _accounting_witnesses() -> list[dict[str, object]]:
    rows = []
    for method_id in DEPLOYABLE_METHOD_IDS:
        record = ExecutionAccountingRecord(
            method_id=method_id,
            trace_id="contract-trace-0001",
            cycle_index=4000,
            action_valid_cycle=4006,
            source_to_action_cycles=6,
            logical_deadline_miss=False,
            update_due=False,
            update_macs=0,
            private_model_state_bytes=0,
            transient_workspace_bytes=0,
            host_update_wallclock_us=0.0,
            host_update_deadline_miss=False,
            board_measured_deadline_miss=None,
        )
        assert_accounting_conforms(record)
        rows.append({"method_id": method_id, "accepted": True, **asdict(record)})
    return rows


def _accounting_rejection_matrix() -> list[dict[str, object]]:
    budget = MatchedBudget()
    base = ExecutionAccountingRecord(
        method_id="proposed_route_a",
        trace_id="contract-trace-0001",
        cycle_index=4000,
        action_valid_cycle=4006,
        source_to_action_cycles=6,
        logical_deadline_miss=False,
        update_due=True,
        update_macs=budget.max_algorithm_macs_per_parameter_update,
        private_model_state_bytes=budget.max_private_model_state_bytes,
        transient_workspace_bytes=budget.max_transient_workspace_bytes,
        host_update_wallclock_us=budget.max_host_update_wallclock_us,
        host_update_deadline_miss=False,
        board_measured_deadline_miss=None,
    )
    assert_accounting_conforms(base, budget)
    mutations = (
        ("early_action", replace(base, action_valid_cycle=4005, source_to_action_cycles=5)),
        ("late_action", replace(base, action_valid_cycle=4007, source_to_action_cycles=7, logical_deadline_miss=True)),
        ("mac_overflow", replace(base, update_macs=budget.max_algorithm_macs_per_parameter_update + 1)),
        ("state_overflow", replace(base, private_model_state_bytes=budget.max_private_model_state_bytes + 1)),
        ("workspace_overflow", replace(base, transient_workspace_bytes=budget.max_transient_workspace_bytes + 1)),
        ("host_deadline_flag_missing", replace(base, host_update_wallclock_us=budget.max_host_update_wallclock_us + 0.001)),
        ("board_claim_before_measurement", replace(base, board_measured_deadline_miss=False)),
        ("work_when_update_not_due", replace(base, update_due=False)),
        ("oracle_in_deployable_ledger", replace(base, method_id=ORACLE_METHOD_ID)),
    )
    return [{"case_id": case_id, **_capture_violation(assert_accounting_conforms, record, budget)} for case_id, record in mutations]


def _source_bindings() -> list[dict[str, object]]:
    rows = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        row: dict[str, object] = {"path": relative, "exists": path.is_file()}
        if path.is_file():
            row["sha256"] = _sha256(path)
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                gates = payload.get("gates")
                gate_pass = (
                    isinstance(gates, list)
                    and bool(gates)
                    and all(isinstance(gate, Mapping) and gate.get("passed") is True for gate in gates)
                )
                row["parent_pass"] = (
                    payload.get("status") == "PASS"
                    or str(payload.get("verdict", "")).startswith("PASS")
                    or gate_pass
                )
        rows.append(row)
    return rows


def recompute_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    contract = report["contract"]
    manifests = report["method_conformance"]
    mismatch = report["method_mismatch_fail_fast"]
    schema = report["schema_rejections"]
    accounting = report["accounting"]
    sources = report["source_bindings"]
    method_ids = [row["method_id"] for row in manifests if row["online_privilege"] == "observed_only"]
    static_row = next(row for row in manifests if row["method_id"] == "static_joint_map")
    return {
        "G01_contract_hash_recomputes": report["contract_sha256"] == contract_sha256(contract),
        "G02_exact_deployable_method_set": tuple(method_ids) == DEPLOYABLE_METHOD_IDS,
        "G03_all_canonical_manifests_conform": all(row["conforms"] for row in manifests),
        "G04_all_methods_share_observed_schema": all(row["input_schema_id"] == OBSERVED_SCHEMA_ID for row in manifests if row["method_id"] in DEPLOYABLE_METHOD_IDS),
        "G05_all_methods_share_action_contracts": all(
            row["action_schema_id"] == ACTION_SCHEMA_ID
            and row["map_lut_contract_id"] == MAP_LUT_CONTRACT_ID
            and row["bank_contract_id"] == BANK_CONTRACT_ID
            and row["cadence_contract_id"] == CADENCE_CONTRACT_ID
            and row["budget_contract_id"] == BUDGET_CONTRACT_ID
            and row["deadline_contract_id"] == DEADLINE_CONTRACT_ID
            for row in manifests if row["method_id"] in DEPLOYABLE_METHOD_IDS
        ),
        "G06_oracle_physically_isolated": contract["oracle_schema"]["physically_separate_from_observed"] is True and contract["oracle_schema"]["deployable_access"] is False and next(row for row in manifests if row["method_id"] == ORACLE_METHOD_ID)["input_schema_id"] == ORACLE_SCHEMA_ID,
        "G07_hidden_truth_rejected_for_every_deployable": sum(row["case_id"] == "hidden_truth_key" and row["rejected"] for row in schema) == len(DEPLOYABLE_METHOD_IDS),
        "G08_all_schema_mutations_rejected": len(schema) >= len(DEPLOYABLE_METHOD_IDS) + 5 and all(row["rejected"] for row in schema),
        "G09_each_method_each_contract_dimension_fails_fast": len(mismatch) == len(DEPLOYABLE_METHOD_IDS) * len(COMMON_MUTATION_FIELDS) and all(row["rejected"] and row["code"] == row["expected_code"] for row in mismatch),
        "G10_wire_crc_and_roundtrip_verified": report["wire_checks"]["roundtrip_pass"] is True and report["wire_checks"]["tamper_rejected"] is True,
        "G11_lut_numeric_contract_exact": contract["map_lut"]["llr_format"] == "signed Q9.12" and contract["map_lut"]["rounding"] == "round_to_nearest_ties_to_even" and contract["map_lut"]["entries_per_phase"] == 257,
        "G12_joint_map_rtl_boundary_not_hidden": static_row["current_rtl_compatibility"].startswith("blocked_full_2d_joint_map") and "not_full_2d_joint_MAP" in contract["map_lut"]["capability"],
        "G13_bank_and_cadence_frozen": contract["bank"]["retired_bank_drain_cycles"] == 6 and contract["cadence"]["parameter_update_period_cycles"] == 4000 and contract["cadence"]["regime_update_period_cycles"] == 32,
        "G14_all_accounting_witnesses_accept": len(accounting["accepted_witnesses"]) == len(DEPLOYABLE_METHOD_IDS) and all(row["accepted"] for row in accounting["accepted_witnesses"]),
        "G15_accounting_boundaries_and_mutations_fail_closed": accounting["exact_cap_witness_accepted"] is True and len(accounting["rejections"]) >= 9 and all(row["rejected"] for row in accounting["rejections"]),
        "G16_no_preboard_measured_deadline_claim": all(row["board_measured_deadline_miss"] is None for row in accounting["accepted_witnesses"]) and contract["deadline"]["board_deadline_field_must_be_null_before_measurement"] is True,
        "G17_all_source_bindings_exist_and_parent_pass": all(row["exists"] and ("parent_pass" not in row or row["parent_pass"]) for row in sources),
    }


def build_report() -> dict[str, Any]:
    observed = validate_observed_mapping_for_deployable("proposed_route_a", _observed_payload())
    validate_wire_roundtrip(observed)
    word = observed.to_wire_word()
    tampered = word ^ (1 << 3)
    tamper_decoded = decode_input_word(tampered)
    OracleTruthPacket.from_mapping(_oracle_payload())

    manifests = []
    for manifest in (*canonical_method_manifests(), oracle_method_manifest()):
        assert_method_conforms(manifest)
        manifests.append({**manifest.to_dict(), "conforms": True})

    budget = MatchedBudget()
    exact_cap = ExecutionAccountingRecord(
        method_id="proposed_route_a",
        trace_id="contract-trace-0001",
        cycle_index=4000,
        action_valid_cycle=4006,
        source_to_action_cycles=6,
        logical_deadline_miss=False,
        update_due=True,
        update_macs=budget.max_algorithm_macs_per_parameter_update,
        private_model_state_bytes=budget.max_private_model_state_bytes,
        transient_workspace_bytes=budget.max_transient_workspace_bytes,
        host_update_wallclock_us=budget.max_host_update_wallclock_us,
        host_update_deadline_miss=False,
        board_measured_deadline_miss=None,
    )
    assert_accounting_conforms(exact_cap, budget)

    contract = contract_snapshot()
    report: dict[str, Any] = {
        "schema_version": "t6.5.2-unified-execution-validation-v1",
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "contract_sha256": contract_sha256(contract),
        "method_conformance": manifests,
        "method_mismatch_fail_fast": _method_mismatch_matrix(),
        "schema_rejections": _schema_rejection_matrix(),
        "wire_checks": {
            "observed_packet_sha256": observed.sha256(),
            "packed_word_hex": f"{word:x}",
            "roundtrip_pass": True,
            "tamper_rejected": tamper_decoded.input_crc_ok is False,
        },
        "accounting": {
            "accepted_witnesses": _accounting_witnesses(),
            "exact_cap_witness_accepted": True,
            "exact_cap_witness": asdict(exact_cap),
            "rejections": _accounting_rejection_matrix(),
            "witness_scope": "contract_validator_boundary_witnesses_not_measured_method_costs",
        },
        "source_bindings": _source_bindings(),
        "evidence_boundary": {
            "allowed": "frozen observed-only execution schemas, exact current LUT/bank/cadence capabilities, matched resource ceilings, and fail-fast conformance validation",
            "not_yet_claimed": [
                "matched-budget comparator performance",
                "full joint-MAP equivalence on the current phase-conditioned RTL LUT",
                "legacy CNN checkpoint budget conformance",
                "Route-A policy performance",
                "target-board deadline or latency",
            ],
        },
    }
    gates = recompute_gates(report)
    report["gates"] = [{"gate_id": gate_id, "passed": passed} for gate_id, passed in gates.items()]
    report["status"] = "PASS" if all(gates.values()) else "FAIL"
    report["verdict"] = "PASS_UNIFIED_EXECUTION_CONTRACT_FROZEN" if report["status"] == "PASS" else "FAIL_UNIFIED_EXECUTION_CONTRACT"
    return report


def verify_report(report: Mapping[str, Any], *, verify_sources: bool = True) -> None:
    gates = recompute_gates(report)
    if not verify_sources:
        gates["G17_all_source_bindings_exist_and_parent_pass"] = True
    if report.get("status") != "PASS" or not all(gates.values()):
        failed = [gate for gate, passed in gates.items() if not passed]
        raise ValueError(f"unified execution report failed: {failed}")
    stored = {row["gate_id"]: row["passed"] for row in report["gates"]}
    if stored != recompute_gates(report):
        raise ValueError("stored gate ledger does not match independent recomputation")
    if verify_sources:
        for binding in report["source_bindings"]:
            path = ROOT / binding["path"]
            if not path.is_file() or _sha256(path) != binding["sha256"]:
                raise ValueError(f"source binding stale: {binding['path']}")


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in report["method_conformance"]:
        rows.append({"row_type": "method", "method_id": method["method_id"], "case_id": "canonical", "field": "conforms", "value": method["conforms"], "status": method["benchmark_deployability"]})
    for item in report["method_mismatch_fail_fast"]:
        rows.append({"row_type": "method_mismatch", "method_id": item["method_id"], "case_id": item["mutated_field"], "field": item["field"], "value": item["code"], "status": "rejected" if item["rejected"] else "accepted"})
    for item in report["schema_rejections"]:
        rows.append({"row_type": "schema_rejection", "method_id": item["method_id"], "case_id": item["case_id"], "field": item["field"], "value": item["code"], "status": "rejected" if item["rejected"] else "accepted"})
    for item in report["accounting"]["rejections"]:
        rows.append({"row_type": "accounting_rejection", "method_id": item.get("method_id") or "proposed_route_a", "case_id": item["case_id"], "field": item["field"], "value": item["code"], "status": "rejected" if item["rejected"] else "accepted"})
    for item in report["gates"]:
        rows.append({"row_type": "gate", "method_id": "all", "case_id": item["gate_id"], "field": "passed", "value": item["passed"], "status": "PASS" if item["passed"] else "FAIL"})
    return rows


def write_report(artifact: Path = DEFAULT_ARTIFACT, source_data: Path = DEFAULT_SOURCE_DATA) -> dict[str, Any]:
    report = build_report()
    verify_report(report)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = _source_rows(report)
    with source_data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row_type", "method_id", "case_id", "field", "value", "status"))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    report = write_report(args.artifact, args.source_data)
    print(json.dumps({"status": report["status"], "verdict": report["verdict"], "contract_sha256": report["contract_sha256"], "methods": len(report["method_conformance"]), "mismatch_cases": len(report["method_mismatch_fail_fast"]), "gates": len(report["gates"])}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
