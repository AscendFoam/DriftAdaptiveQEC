from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9/t9_2_1_causal_twin_contract.json"
REPORT = ROOT / "docs/t9_2_1_causal_twin_contract.json"
RUNTIME = ROOT / "physics/phase9_twin_contract.py"


def test_bootstrap_config_is_visible_and_parent_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["task_id"] == "T9.2.1"
    assert config["protocol_id"] == "PHASE9-CAUSAL-TWIN-CONTRACT-V1"
    assert (
        config["parent_contract"]["t9_1_1"]["analysis_sha256"]
        == "c88110375c358794339e72d672e4624871425fe480e5da091ddd1d6595255e18"
    )
    assert config["parent_contract"]["t9_1_5"]["release_pin"][
        "path"
    ] == "configs/phase9/t9_1_5_release_pin.json"


def test_bootstrap_five_namespace_ids_are_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["namespace_contract"]["exact_ids"] == [
        "BACKEND_LATENT",
        "DEPLOYABLE_OBSERVED",
        "CONTROLLER_MEMORY",
        "EVALUATOR_TRUTH",
        "PROVENANCE",
    ]
    assert config["namespace_contract"]["action_word_is_separate"] is True


def _implementation():
    if not RUNTIME.exists():
        pytest.skip("runtime contract is being implemented by sibling agent")
    from cnn_fpga.benchmark import phase9_causal_twin_contract

    return phase9_causal_twin_contract


def _report() -> dict:
    if not REPORT.exists():
        pytest.skip("canonical report awaits exact runtime snapshot")
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_canonical_release_verifies() -> None:
    implementation = _implementation()
    report = _report()
    checks = implementation.verify_report(
        implementation.DEFAULT_REPORT,
        expected_analysis_sha256=report["analysis_sha256"],
    )
    assert checks and all(checks.values())


def test_report_selected_path_is_rejected() -> None:
    implementation = _implementation()
    with pytest.raises(ValueError, match="canonical report path"):
        implementation.verify_report(ROOT / "docs/alternate.json")


def test_all_current_outcome_fields_are_typed_null() -> None:
    report = _report()
    assert tuple(report["current_null_state"]) == (
        "physics",
        "performance",
        "codebook",
        "frontend",
        "claim",
        "rank",
    )
    assert all(
        value is None
        for fields in report["current_null_state"].values()
        for value in fields.values()
    )


def test_mutation_replay_is_one_to_one_and_complete() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] >= 36
    assert audit["count"] == audit["detected"]
    assert audit["all_detected"] is True
    assert audit["one_per_gate"] is True
    assert [row["target_gate"] for row in audit["records"]] == list(
        report["gates"]
    )
    assert all(
        row["target_gate"] in row["failed_gates"]
        for row in audit["records"]
    )


def test_mapping_input_cannot_select_alternate_outputs() -> None:
    implementation = _implementation()
    report = _report()
    candidate = copy.deepcopy(report)
    candidate["source_data"]["path"] = "docs/alternate.csv"
    with pytest.raises(ValueError):
        implementation.verify_report(candidate)


def test_runtime_totality_is_exhaustive_not_sampled() -> None:
    report = _report()
    manifest = report["runtime_contract"]["factorized_map_manifest"]
    assert manifest["nominal"]["actual_count"] == 1024
    assert manifest["nominal"]["unique_key_count"] == 1024
    assert manifest["transition"]["actual_count"] == 131072
    assert manifest["transition"]["unique_key_count"] == 131072
    assert (
        manifest["composition"]["actual_count"]
        == manifest["composition"]["expected_count"]
        == manifest["composition"]["unique_key_count"]
        == 196608
    )
    assert manifest["composition"]["full_cartesian_key_count"] == 16777216
    assert manifest["composition"]["quotient_is_lossless"] is True
    assert manifest["composition"]["equivalence_witness"][
        "mapped_nominal_key_count"
    ] == 1024
    assert manifest["coverage_complete"] is True
    assert manifest["deterministic"] is True
    assert manifest["reachable_fsm_count"] == 68
    assert manifest["reset_bfs_covered_count"] == 192
    assert manifest["reset_bfs_max_distance"] == 3
    assert manifest["fsm_reachability_scope"].startswith(
        "SYNTACTIC_T_DOMAIN"
    )
    assert manifest["reset_bfs_scope"].startswith(
        "SYNTACTIC_SUCCESS_ACK"
    )


def test_runtime_audit_has_no_soft_failed_check() -> None:
    audit = _report()["runtime_contract"]["audit"]
    assert audit["all_passed"] is True
    assert audit["checks"]
    assert all(value is True for value in audit["checks"].values())


def test_phase_and_fsm_are_explicitly_not_current_rtl_adapters() -> None:
    key = _report()["key_contract"]
    assert key["phase_frame_semantics"]["states"] == {
        "0": {"q_byte": 0, "p_byte": 0},
        "1": {"q_byte": 128, "p_byte": 0},
        "2": {"q_byte": 0, "p_byte": 128},
        "3": {"q_byte": 128, "p_byte": 128},
    }
    assert (
        key["phase_frame_semantics"][
            "current_rtl_two_uint8_adapter_qualified"
        ]
        is False
    )
    assert key["fsm_encoding"]["mode_bits"] == 3
    assert key["fsm_encoding"]["active_dwell_counter_bits"] == 5
    assert (
        key["fsm_encoding"][
            "current_rtl_six_counter_adapter_qualified"
        ]
        is False
    )


def test_action_layout_is_contiguous_80_bits_and_residual_zero() -> None:
    action = _report()["action_word_contract"]
    assert action["total_bits"] == 80
    occupied = set()
    for field in action["fields"]:
        bits = set(range(field["lsb"], field["lsb"] + field["bits"]))
        assert not occupied & bits
        occupied |= bits
    assert occupied == set(range(80))
    assert action["base_lane_residual"] == {
        "width_bits": 2,
        "required_value": 0,
        "nonzero_representable": False,
    }


def test_action_word_roundtrip_and_crc_tamper_rejection() -> None:
    _implementation()
    import physics.phase9_twin_contract as runtime

    result = runtime.total_recurrence(
        runtime.CompositeKey(
            bank_id=runtime.BankId.A,
            discriminator_word=0,
            phase_frame=0,
            event_class=runtime.EventClass.NORMAL,
            leakage_reset_fsm_state=runtime.encode_fsm(
                runtime.FsmMode.NORMAL, 0
            ),
        )
    )
    packed = result.action_word.pack()
    assert packed.bit_length() <= 80
    assert runtime.ActionWord.unpack(packed) == result.action_word
    with pytest.raises(ValueError, match="CRC16"):
        runtime.ActionWord.unpack(packed ^ (1 << 64))


@pytest.mark.parametrize(("field", "value"), [("residual_q", 1), ("residual_p", 1)])
def test_nonzero_residual_is_structurally_rejected(
    field: str, value: int
) -> None:
    _implementation()
    import physics.phase9_twin_contract as runtime

    kwargs = {
        "action_code": 0,
        "correction_enable": False,
        "reset_request": False,
        "fallback": False,
        "hold": False,
        "pauli_dx": 0,
        "pauli_dz": 0,
        "next_phase_frame": 0,
        "next_fsm_state": runtime.encode_fsm(runtime.FsmMode.NORMAL, 0),
        "catalog_action_id": 0,
        "reason_code": 0,
        "error_flags": 0,
        "source_bank_id": 0,
        "factor_tag": 1,
        field: value,
    }
    with pytest.raises(ValueError, match="residual"):
        runtime.ActionWord(**kwargs)


def test_deployable_validator_recursively_rejects_truth() -> None:
    _implementation()
    import physics.phase9_twin_contract as runtime

    observed = runtime._minimal_observed()
    memory = runtime._minimal_memory()
    runtime.validate_deployable_inputs(observed, memory)
    contaminated = copy.deepcopy(observed)
    contaminated["iq_i"] = [{"future": 1}]
    with pytest.raises((TypeError, ValueError)):
        runtime.validate_deployable_inputs(contaminated, memory)


def test_exact_five_namespaces_and_action_output_separation() -> None:
    contract = _report()["namespace_contract"]
    assert tuple(contract["schemas"]) == (
        "BACKEND_LATENT",
        "DEPLOYABLE_OBSERVED",
        "CONTROLLER_MEMORY",
        "EVALUATOR_TRUTH",
        "PROVENANCE",
    )
    assert "ACTION_WORD" not in contract["schemas"]
    assert contract["action_word_is_separate"] is True
    assert contract["deployable_input_namespaces"] == [
        "DEPLOYABLE_OBSERVED",
        "CONTROLLER_MEMORY",
    ]


def test_causal_graph_has_no_privileged_action_edge() -> None:
    causal = _report()["causal_contract"]
    forbidden = {
        (row["source"], row["target"])
        for row in causal["forbidden_edges"]
    }
    assert ("EVALUATOR_TRUTH_T", "ACTION_WORD_T") in forbidden
    assert ("LATENT_T", "ACTION_WORD_T") in forbidden
    assert ("PROVENANCE_AUDIT", "ACTION_WORD_T") in forbidden
    assert all(row["allowed"] is False for row in causal["forbidden_edges"])


def test_slow_path_has_no_fast_action_or_patch_authority() -> None:
    slow = _report()["slow_path_contract"]
    assert slow["allowed_operation"] == (
        "NOMINATE_COMPLETE_PRECOMPILED_TRUSTED_PACKAGE"
    )
    assert slow["per_cycle_action_authority"] is False
    assert slow["single_entry_patch_allowed"] is False
    assert slow["partial_active_visibility_allowed"] is False
    assert slow["commit_semantics"] == "ATOMIC_OLD_OR_NEW_COMPLETE_IMAGE"
    assert set(slow["package_required_fields"]) == {
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
    assert slow["runtime_boundary"][
        "atomic_bank_integration_qualified"
    ] is False


def test_fault_priority_is_unique_complete_and_safe() -> None:
    fault = _report()["fault_contract"]
    assert [row["priority"] for row in fault["priority"]] == list(
        range(14)
    )
    assert {row["fault_id"] for row in fault["priority"]} == set(
        fault["required_fault_ids"]
    )
    assert all(
        row["terminal"] in {"LKG_HOLD", "RESET"}
        and row["undefined_action"] is False
        for row in fault["priority"]
    )
    import physics.phase9_twin_contract as runtime

    assert [row["fault_id"] for row in fault["priority"]] == list(
        runtime.FAULT_PRIORITY
    )
    witnesses = _report()["runtime_contract"][
        "fault_response_witnesses"
    ]
    assert witnesses == list(runtime.fault_response_witnesses())
    assert fault["priority"][0]["reason_codes"] == [
        "INVALID_BANK",
        "INVALID_FSM",
    ]


def test_exactly_sixteen_probes_are_nonpromotional() -> None:
    probes = _report()["representative_probes"]
    assert len(probes) == len({row["probe_id"] for row in probes}) == 16
    assert all(
        row["probe_only"] is True
        and row["codebook_candidate"] is False
        and row["performance_evidence"] is False
        and row["ranking_evidence"] is False
        for row in probes
    )
    import physics.phase9_twin_contract as runtime

    runtime_probes = runtime.representative_action_probes()
    for probe in runtime_probes:
        receipts = runtime.execute_representative_probe(probe)
        assert set(probe.coverage_tags) - {
            "pre_codebook_interface_probe"
        } <= runtime.probe_coverage_witnesses(receipts)


def test_source_data_is_full_analysis_lossless() -> None:
    implementation = _implementation()
    report = _report()
    path = implementation.DEFAULT_SOURCE_DATA
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 246
    reconstructed = {
        row["record_id"]: json.loads(row["payload_json"])
        for row in rows
        if row["record_type"] == "analysis_field"
    }
    assert reconstructed == implementation._analysis_payload(report)
    assert all(
        hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
        == row["payload_sha256"]
        for row in rows
    )


def test_totality_manifest_is_analysis_bound_and_nonperformance() -> None:
    implementation = _implementation()
    manifest = json.loads(
        implementation.DEFAULT_TOTALITY_MANIFEST.read_text(
            encoding="utf-8"
        )
    )
    assert manifest["analysis_sha256"] == implementation._canonical_sha256(
        implementation._manifest_analysis(manifest)
    )
    assert manifest["performance_metrics"] is None
    assert manifest["codebook_id"] is None
    assert manifest["backend_a_qualification"] is None
    assert manifest["backend_b_qualification"] is None


def test_release_pin_binds_every_canonical_artifact() -> None:
    implementation = _implementation()
    report = _report()
    pin = json.loads(
        implementation.DEFAULT_RELEASE_PIN.read_text(encoding="utf-8")
    )
    assert pin == implementation._release_pin(report)
    assert pin["analysis_sha256"] == report["analysis_sha256"]
    assert set(pin) == {
        "schema_version",
        "task_id",
        "protocol_id",
        "analysis_sha256",
        "config",
        "implementation",
        "physics_contract",
        "report",
        "totality_manifest",
        "source_data",
        "markdown",
    }


def test_markdown_is_exact_generated_text() -> None:
    implementation = _implementation()
    report = _report()
    assert implementation.DEFAULT_MARKDOWN.read_text(
        encoding="utf-8"
    ) == implementation._render_markdown(report)
    text = implementation.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    assert all(
        f"`{identifier}`" in text
        for identifier in implementation._atomic_ids(report)
    )


@pytest.mark.parametrize(
    "category",
    ["physics", "performance", "codebook", "frontend", "claim", "rank"],
)
def test_each_result_category_remains_typed_null(category: str) -> None:
    values = _report()["current_null_state"][category]
    assert values
    assert set(values.values()) == {None}
