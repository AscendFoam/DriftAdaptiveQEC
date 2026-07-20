from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.unified_execution_contract_validation import (
    COMMON_MUTATION_FIELDS,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    build_report,
    recompute_gates,
    verify_report,
)
from cnn_fpga.runtime.bit_accurate_hardware_reference import decode_input_word
from cnn_fpga.runtime.unified_execution_contract import (
    ACTION_SCHEMA_ID,
    DEPLOYABLE_METHOD_IDS,
    MAP_LUT_CONTRACT_ID,
    OBSERVED_SCHEMA_ID,
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


def _observed() -> dict[str, object]:
    return {
        "schema_id": OBSERVED_SCHEMA_ID,
        "trace_id": "test-trace-1",
        "cycle_index": 17,
        "syndrome_code": 1023,
        "syndrome_x": "leakage",
        "syndrome_z": "e",
        "quadrature_phase_bit": 0,
        "ood_score_code": 255,
        "parameter_age_code": 65535,
        "reset_ack": True,
        "observation_valid": True,
        "deadline_ok": False,
    }


def _accounting() -> ExecutionAccountingRecord:
    budget = MatchedBudget()
    return ExecutionAccountingRecord(
        method_id="proposed_route_a",
        trace_id="test-trace-1",
        cycle_index=17,
        action_valid_cycle=23,
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


def test_contract_freezes_exact_lut_bank_cadence_budget_and_nonmeasurement_boundary() -> None:
    snapshot = contract_snapshot()
    assert len(contract_sha256(snapshot)) == 64
    assert snapshot["map_lut"] == {
        "contract_id": MAP_LUT_CONTRACT_ID,
        "capability": "phase_conditioned_X_Z_LLR_not_full_2d_joint_MAP",
        "adc_bits": 10,
        "address_bits": 8,
        "fraction_bits": 2,
        "entries_per_phase": 257,
        "phase_tables": 2,
        "llr_format": "signed Q9.12",
        "llr_word_bits": 22,
        "rounding": "round_to_nearest_ties_to_even",
        "saturation_codes": [-(1 << 21), (1 << 21) - 1],
        "pipeline_latency_cycles": 5,
        "content_semantics": "same code grid, interpolation, Q-format, CRC and bank image layout; method-specific LLR values may change only through the common cadence and commit protocol",
        "joint_map_boundary": "full_2d_joint_MAP_is_observed_only_software_comparator_until_equivalence_or_new_RTL",
    }
    assert snapshot["action_schema"]["latency_cycles"] == 6
    assert snapshot["bank"]["retired_bank_drain_cycles"] == 6
    assert snapshot["cadence"]["regime_update_period_cycles"] == 32
    assert snapshot["cadence"]["parameter_update_period_cycles"] == 4000
    assert snapshot["budget"]["shared_dual_bank_lut_payload_bits"] == 22616
    assert snapshot["budget"]["shared_current_rtl_mirror_bits"] == 45232
    assert snapshot["deadline"]["board_deadline_field_must_be_null_before_measurement"] is True


def test_observed_schema_is_exact_crc_roundtrips_and_rejects_truth_for_every_method() -> None:
    for method_id in DEPLOYABLE_METHOD_IDS:
        packet = validate_observed_mapping_for_deployable(method_id, _observed())
        validate_wire_roundtrip(packet)
        decoded = decode_input_word(packet.to_wire_word())
        assert decoded.input_crc_ok is True

        leaked = _observed()
        leaked["latent_sigma"] = 0.2
        with pytest.raises(ContractViolation, match="hidden_truth_key_rejected"):
            validate_observed_mapping_for_deployable(method_id, leaked)

    missing = _observed()
    missing.pop("deadline_ok")
    with pytest.raises(ContractViolation, match="schema_keys_mismatch"):
        validate_observed_mapping_for_deployable("proposed_route_a", missing)
    wrong = _observed()
    wrong["syndrome_code"] = True
    with pytest.raises(ContractViolation, match="integer_required"):
        validate_observed_mapping_for_deployable("proposed_route_a", wrong)


def test_oracle_truth_schema_is_distinct_and_cannot_enter_a_deployable_adapter() -> None:
    truth = OracleTruthPacket.from_mapping(
        {
            "schema_id": ORACLE_SCHEMA_ID,
            "trace_id": "test-trace-1",
            "cycle_index": 17,
            "latent_displacement_q": 0.1,
            "latent_displacement_p": -0.2,
            "latent_mean_q": 0.0,
            "latent_mean_p": 0.0,
            "latent_sigma_q": 0.3,
            "latent_sigma_p": 0.4,
            "latent_correlation": -0.7,
            "logical_x": True,
            "logical_z": False,
            "regime_label": "compound",
        }
    )
    assert truth.schema_id != OBSERVED_SCHEMA_ID
    with pytest.raises(ContractViolation):
        validate_observed_mapping_for_deployable("proposed_route_a", asdict(truth))
    oracle = oracle_method_manifest()
    assert_method_conforms(oracle)
    assert oracle.benchmark_deployability == "nondeployable_upper_bound_only"
    assert oracle.current_rtl_compatibility == "prohibited"


def test_every_method_manifest_is_exact_and_every_common_dimension_fails_fast() -> None:
    manifests = canonical_method_manifests()
    assert tuple(row.method_id for row in manifests) == DEPLOYABLE_METHOD_IDS
    for manifest in manifests:
        assert_method_conforms(manifest)
        assert manifest.input_schema_id == OBSERVED_SCHEMA_ID
        assert manifest.action_schema_id == ACTION_SCHEMA_ID
        assert manifest.map_lut_contract_id == MAP_LUT_CONTRACT_ID
        roundtrip = MethodManifest.from_mapping(manifest.to_dict())
        assert roundtrip == manifest
        for field in COMMON_MUTATION_FIELDS:
            with pytest.raises(ContractViolation) as exc:
                assert_method_conforms(replace(manifest, **{field: "mutated"}))
            assert exc.value.code == "method_contract_mismatch"
            assert exc.value.field == field


def test_joint_map_name_does_not_silently_gain_current_rtl_compatibility() -> None:
    manifest = next(row for row in canonical_method_manifests() if row.method_id == "static_joint_map")
    assert manifest.benchmark_deployability == "observed_only_budgeted_candidate"
    assert manifest.current_rtl_compatibility.startswith("blocked_full_2d_joint_map")
    promoted = replace(manifest, current_rtl_compatibility="current_rtl_bit_exact")
    with pytest.raises(ContractViolation, match="method_contract_mismatch"):
        assert_method_conforms(promoted)


def test_accounting_accepts_exact_caps_and_rejects_latency_cost_and_preboard_claims() -> None:
    record = _accounting()
    budget = MatchedBudget()
    assert_accounting_conforms(record, budget)
    mutations = (
        replace(record, action_valid_cycle=24, source_to_action_cycles=7, logical_deadline_miss=True),
        replace(record, update_macs=budget.max_algorithm_macs_per_parameter_update + 1),
        replace(record, private_model_state_bytes=budget.max_private_model_state_bytes + 1),
        replace(record, transient_workspace_bytes=budget.max_transient_workspace_bytes + 1),
        replace(record, host_update_wallclock_us=budget.max_host_update_wallclock_us + 1.0),
        replace(record, board_measured_deadline_miss=False),
        replace(record, method_id="hidden_state_oracle"),
    )
    for mutated in mutations:
        with pytest.raises(ContractViolation):
            assert_accounting_conforms(mutated, budget)

    for field in (
        "cycle_index",
        "action_valid_cycle",
        "source_to_action_cycles",
        "update_macs",
        "private_model_state_bytes",
        "transient_workspace_bytes",
        "host_update_wallclock_us",
    ):
        with pytest.raises(ContractViolation):
            replace(record, **{field: -1})

    no_update = replace(
        record,
        update_due=False,
        update_macs=0,
        private_model_state_bytes=0,
        transient_workspace_bytes=0,
        host_update_wallclock_us=0.0,
    )
    assert_accounting_conforms(no_update, budget)
    with pytest.raises(ContractViolation, match="cost_charged_when_update_not_due"):
        assert_accounting_conforms(replace(no_update, update_macs=1), budget)


def test_machine_report_has_full_per_method_fail_fast_matrix_and_recomputable_gates() -> None:
    report = build_report()
    verify_report(report)
    assert report["verdict"] == "PASS_UNIFIED_EXECUTION_CONTRACT_FROZEN"
    assert len(report["method_conformance"]) == 8
    assert len(report["method_mismatch_fail_fast"]) == 7 * 10
    assert len(report["schema_rejections"]) == 12
    gates = recompute_gates(report)
    assert len(gates) == 17
    assert all(gates.values())
    assert report["accounting"]["witness_scope"].endswith("not_measured_method_costs")


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("contract", "map_lut", "rounding"), "truncate"),
        (("method_conformance", 1, "current_rtl_compatibility"), "current_rtl_bit_exact"),
        (("method_mismatch_fail_fast", 0, "rejected"), False),
        (("schema_rejections", 0, "rejected"), False),
        (("accounting", "rejections", 0, "rejected"), False),
    ),
)
def test_machine_report_semantic_mutations_are_detected(path: tuple[object, ...], value: object) -> None:
    report = build_report()
    mutated = deepcopy(report)
    target: object = mutated
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(ValueError):
        verify_report(mutated, verify_sources=False)


def test_committed_artifacts_are_current_and_source_data_is_not_empty() -> None:
    report = json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))
    verify_report(report)
    assert report["contract_sha256"] == contract_sha256(report["contract"])
    rows = DEFAULT_SOURCE_DATA.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 8 + 70 + 12 + 9 + 17
    assert rows[0] == "row_type,method_id,case_id,field,value,status"
