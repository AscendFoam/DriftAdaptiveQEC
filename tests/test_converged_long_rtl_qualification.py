from __future__ import annotations

import copy
import csv
import hashlib
import json
import zlib

import pytest

from cnn_fpga.benchmark import converged_long_rtl_qualification as subject
from cnn_fpga.runtime.converged_production_reference import (
    ConvergedInputs,
    IndependentParameterBankManager,
    REJECT_ACTIVE_BANK,
    admit_commit,
    image_crc32,
)
from cnn_fpga.runtime.route_a_fixed_policy_reference import ACTION_OPEN


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(subject.REPORT.read_text(encoding="utf-8"))


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_trace_abi_and_config_freeze_the_complete_public_vector() -> None:
    config = json.loads(subject.CONFIG.read_text(encoding="utf-8"))
    assert subject.INPUT_STRUCT.size == config["trace_row_bytes"] == 202
    assert subject.EXPECTED_BYTES == config["expected_public_output_bytes"] == 148
    assert config["cycles_per_family"] == subject.FAMILY_CYCLES == 100_000
    assert config["aggregate_cycles"] == 1_000_000
    assert config["families"] == list(subject.FAMILY_NAMES)


def test_independent_image_crc_matches_standard_byte_packing() -> None:
    words = [0, 1, 0x155555, 1 << 21, (1 << 22) - 1]
    packed = b"".join((word & 0x3FFFFF).to_bytes(3, "little") for word in words)
    assert image_crc32(words) == zlib.crc32(packed) & 0xFFFFFFFF


def test_admission_blocks_host_during_policy_ownership() -> None:
    inputs = ConvergedInputs(
        host_commit_valid=1,
        host_commit_bank=1,
        host_expected_active_version=0,
        host_new_activation_version=1,
    )
    allowed = admit_commit(
        inputs,
        policy_commit_valid=0,
        policy_commit_bank=0,
        policy_commit_version=0,
        policy_commit_pending=0,
        policy_action=ACTION_OPEN,
    )
    assert allowed.host_commit_blocked == 0
    assert allowed.effective_commit_valid == 1
    blocked = admit_commit(
        inputs,
        policy_commit_valid=0,
        policy_commit_bank=0,
        policy_commit_version=0,
        policy_commit_pending=1,
        policy_action=ACTION_OPEN,
    )
    assert blocked.host_commit_blocked == 1
    assert blocked.effective_commit_valid == 0


def test_independent_manager_rejects_active_bank_without_state_mutation() -> None:
    manager = IndependentParameterBankManager()
    inputs = ConvergedInputs(
        cfg_begin_valid=1,
        cfg_begin_bank=0,
        cfg_expected_active_version=0,
        cfg_new_image_version=2,
    )
    admission = admit_commit(
        inputs,
        policy_commit_valid=0,
        policy_commit_bank=0,
        policy_commit_version=0,
        policy_commit_pending=0,
        policy_action=ACTION_OPEN,
    )
    manager.step(
        inputs,
        admission=admission,
        core_active_bank=0,
        core_active_version=0,
        core_commit_ack=0,
    )
    assert manager.pulses["management_reject"] == 1
    assert manager.pulses["management_reject_reason"] == REJECT_ACTIVE_BANK
    assert manager.cfg_session_active == 0
    assert manager.bank_trusted == [1, 1]


def test_qualifying_report_recomputes_all_nineteen_gates(report: dict) -> None:
    assert report["verdict"] == subject.VERDICT
    assert report["gate_summary"] == {"passed": 19, "total": 19}
    assert all(row["passed"] for row in report["gates"])
    assert report["gates"][:-1] == subject.evaluate_gates(report)
    assert report["formal_anchor"]["exact_required_source_bindings"] is True
    assert report["formal_anchor"]["actual_core_atomic_commit_returncode"] == 0
    assert report["formal_anchor"]["near_wrap_witness_found"] is True


def test_trace_and_generated_cxxrtl_are_hash_bound(report: dict) -> None:
    trace = subject.ROOT / report["trace"]["path"]
    assert report["trace"]["rows"] == 1_000_000
    assert trace.stat().st_size == 1_000_000 * subject.INPUT_STRUCT.size == 202_000_000
    assert _sha256(trace) == report["trace"]["sha256"]
    for key in ("model", "executable_binding"):
        binding = report["toolchain"][key]
        path = subject.ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert _sha256(path) == binding["sha256"]


def test_every_family_is_full_scale_and_all_public_bytes_are_exact(report: dict) -> None:
    rows = report["cxxrtl_families"]
    assert len(rows) == 10
    assert sum(row["rows"] for row in rows) == 1_000_000
    assert all(row["rows"] == subject.FAMILY_CYCLES for row in rows)
    assert all(row["mismatches"] == 0 for row in rows)
    assert all(row["actual_digest"] == row["expected_digest"] for row in rows)
    assert sum(row["shadow_mutations"] for row in rows) == subject.EXPECTED_BYTES
    assert sum(row["shadow_mutations_detected"] for row in rows) == subject.EXPECTED_BYTES


def test_six_cycle_ii1_contract_is_count_conserving(report: dict) -> None:
    aggregate = report["aggregate_python"]
    assert aggregate["latency_violations"] == 0
    assert aggregate["map_latency_violations"] == 0
    assert aggregate["route_alignment_violations"] == 0
    assert (
        aggregate["input_valid"]
        == aggregate["output_valid"]
        == aggregate["route_valid"]
        == aggregate["map_valid"]
    )
    assert aggregate["ii1_input_pairs"] == aggregate["ii1_output_pairs"] > 800_000


def test_fault_contract_distinguishes_injectable_from_structurally_blocked(report: dict) -> None:
    faults = report["aggregate_python"]["core_fault_bits"]
    assert all(faults[str(bit)] > 0 for bit in subject.INJECTABLE_CORE_FAULT_BITS)
    assert all(faults[str(bit)] == 0 for bit in subject.COMPOSITION_PROTECTED_CORE_FAULT_BITS)
    assert report["aggregate_python"]["fault_to_clean_recoveries"] > 0
    assert report["aggregate_python"]["route_to_open_recoveries"] > 0


def test_policy_management_and_transport_negative_paths_are_saturated(report: dict) -> None:
    aggregate = report["aggregate_python"]
    assert all(value > 0 for value in aggregate["actions"].values())
    assert all(value > 0 for value in aggregate["reasons"].values())
    assert all(value > 0 for value in aggregate["reject_reasons"].values())
    assert aggregate["cfg_word_acks"] >= 1028
    assert aggregate["host_commit_completes"] > 0
    assert aggregate["policy_commit_completes"] > 0
    transport = report["python_families"][9]["transport"]
    assert transport["overflow_events"] == transport["accounted_overflow_events"] > 0
    assert transport["silent_overflow"] == 0
    assert transport["pending_fifo"] == transport["pending_markers"] == 0
    for key in ("drop_events", "duplicate_events", "reorder_events", "sequence_faults", "deadline_faults"):
        assert transport[key] > 0


def test_semantic_mutations_are_independently_recomputed(report: dict) -> None:
    audit = subject.semantic_mutation_audit(report)
    assert report["semantic_mutations"] == {"detected": 21, "total": 21}
    assert audit["detected"] == audit["total"] == 21
    assert report["semantic_mutation_results"] == audit["mutations"]
    assert all(row["rejected"] for row in audit["mutations"])


def test_validator_rejects_metric_gate_mutation_and_claim_promotion(report: dict) -> None:
    candidate = copy.deepcopy(report)
    candidate["aggregate_python"]["output_valid"] -= 1
    with pytest.raises(subject.IntegrityError):
        subject._validate_report(candidate, check_files=False)
    candidate = copy.deepcopy(report)
    candidate["claim_boundary"]["board_measurement"] = {"latency_ns": 1}
    with pytest.raises(subject.IntegrityError):
        subject._validate_report(candidate, check_files=False)


def test_source_data_is_lossless_for_families_gates_mutations_and_bindings(report: dict) -> None:
    with subject.SOURCE_DATA.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert sum(row["section"] == "family" for row in rows) == 10
    assert sum(row["section"] == "gate" for row in rows) == 19
    assert sum(row["section"] == "mutation" for row in rows) == 21
    assert sum(row["section"] == "binding" for row in rows) == len(report["bindings"])


def test_live_artifact_verification_is_fail_closed() -> None:
    verified = subject.verify()
    assert verified["verdict"] == subject.VERDICT
    assert verified["gates"] == {"passed": 19, "total": 19}
