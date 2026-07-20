from __future__ import annotations

import ast
import copy
import csv
import hashlib
import inspect
import json
import textwrap
from pathlib import Path

import pytest

from cnn_fpga.benchmark import bit_accurate_hardware_reference as validation
from cnn_fpga.runtime.atomic_parameter_bank import AtomicParameterBankConfig
from cnn_fpga.runtime.bit_accurate_hardware_reference import (
    INPUT_SCHEMA,
    OUTPUT_SCHEMA,
    STATE_SCHEMA,
    BitAccurateHardwareReference,
    crc16_ccitt_false,
    decode_input_word,
    encode_input_word,
    hardware_reference_contract,
    pack_parameter_bundle,
    pack_parameter_image,
    unpack_parameter_bundle,
    unpack_parameter_image,
)
from cnn_fpga.runtime.fast_path_fixed_point import (
    BitAccurateFastPath,
    FastPathCodeInput,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/t5_5_1_bit_accurate_hardware_reference.json"
SOURCE = ROOT / "docs/t5_5_1_bit_accurate_hardware_reference_source_data.csv"
TRACE = ROOT / "docs/t5_5_1_bit_accurate_golden_trace.csv"
BUNDLE = ROOT / "docs/t5_5_1_bit_accurate_parameter_bank.bin"


@pytest.fixture(scope="module")
def images():
    return validation.load_frozen_images()


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rehash(changed: dict) -> dict:
    changed["contract_sha256"] = validation._canonical_sha256(
        validation._contract_view(changed)
    )
    return changed


def test_formal_artifact_is_current_and_semantically_valid(report: dict) -> None:
    assert report["status"] == "PASS"
    assert report["verdict"] == (
        "BIT_ACCURATE_PYTHON_RTL_GOLDEN_FROZEN_HARDWARE_UNMEASURED"
    )
    assert report["gate_summary"] == {"passed": 16, "total": 16}
    assert validation.validate_artifact(report) == ()


def test_parent_and_implementation_bindings_are_live(report: dict) -> None:
    assert len(report["parent_bindings"]) == 7
    assert all(row["machine_pass"] for row in report["parent_bindings"])
    for row in report["parent_bindings"] + report["implementation_bindings"]:
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert row["sha256"] == validation._sha256(path)


def test_crc16_known_vector_and_word_widths_are_frozen() -> None:
    assert crc16_ccitt_false(b"123456789") == 0x29B1
    assert INPUT_SCHEMA.word_bits == 58
    assert OUTPUT_SCHEMA.word_bits == 118
    assert STATE_SCHEMA.word_bits == 232
    for schema in (INPUT_SCHEMA, OUTPUT_SCHEMA, STATE_SCHEMA):
        assert [field.offset for field in schema.fields] == [
            sum(item.width for item in schema.fields[:index])
            for index in range(len(schema.fields))
        ]
        assert schema.word_bits == schema.payload_bits + 16


def test_input_word_roundtrip_and_single_bit_fault_detection() -> None:
    word = encode_input_word(
        syndrome_code=1023,
        syndrome_x="e",
        syndrome_z="leakage",
        quadrature_phase_bit=1,
        ood_score_code=255,
        parameter_age_code=65535,
        reset_ack=True,
        observation_valid=True,
        deadline_ok=True,
    )
    decoded = decode_input_word(word)
    assert decoded.input_crc_ok
    assert decoded.syndrome_code == 1023
    assert decoded.syndrome_x == "e"
    assert decoded.syndrome_z == "leakage"
    assert decoded.reset_ack
    assert all(
        not decode_input_word(word ^ (1 << bit)).input_crc_ok
        for bit in range(INPUT_SCHEMA.word_bits)
    )


def test_reserved_observation_code_is_fail_closed_even_with_valid_crc() -> None:
    values = {
        field.name: 0 for field in INPUT_SCHEMA.fields
    }
    values.update(
        syndrome_code=10,
        syndrome_x_code=3,
        syndrome_z_code=0,
        observation_valid=1,
        deadline_ok=1,
    )
    decoded = decode_input_word(INPUT_SCHEMA.pack(values))
    assert decoded.input_crc_ok
    assert decoded.reserved_observation_code
    assert decoded.observation_valid is False


def test_binary_parameter_images_and_bundle_roundtrip_exactly(images) -> None:
    payloads = [pack_parameter_image(image) for image in images]
    assert {len(payload) for payload in payloads} == {1706}
    assert [unpack_parameter_image(payload) for payload in payloads] == list(images)
    bundle = pack_parameter_bundle(images)
    assert len(bundle) == 13_724
    assert unpack_parameter_bundle(bundle) == images
    assert bundle == BUNDLE.read_bytes()


def test_parameter_image_prefix_corruption_and_nonselected_profile_reject(images) -> None:
    payload = pack_parameter_image(images[0])
    with pytest.raises(ValueError, match="exactly"):
        unpack_parameter_image(payload[:-1])
    changed = bytearray(payload)
    changed[500] ^= 1
    with pytest.raises(ValueError, match="CRC32"):
        unpack_parameter_image(bytes(changed))
    from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig
    from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
    from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles

    config = ParametricMAPLUTConfig(adc_bits=8, address_bits=6)
    params = registered_parameter_profiles(config)[0][0]
    image = compile_parametric_map_lut(params, active_bank_version=0, config=config)
    with pytest.raises(ValueError, match="selected"):
        pack_parameter_image(image)


def test_true_pipeline_has_warmup_six_cycle_latency_and_one_output_per_input(images) -> None:
    bank_config = AtomicParameterBankConfig(
        min_residency_cycles=1, max_payload_age_cycles=100
    )
    reference = BitAccurateHardwareReference(images, bank_config=bank_config)
    for cycle in range(20):
        reference.step_word(
            encode_input_word(
                syndrome_code=(cycle * 17) % 1024,
                syndrome_x="g",
                syndrome_z="g",
                quadrature_phase_bit=cycle % 2,
                ood_score_code=0,
                parameter_age_code=0,
            )
        )
    for _ in range(6):
        reference.step_word(None)
    outputs = [row for row in reference.trace if row.output_valid]
    assert len(outputs) == 20
    assert sum(not row.output_valid for row in reference.trace[:6]) == 6
    assert [row.output_source_cycle for row in outputs] == list(range(20))
    assert {row.hardware_cycle - row.output_source_cycle for row in outputs} == {6}
    assert all(row.output_crc_ok and row.state_crc_ok for row in reference.trace)


def test_integrated_first_action_matches_legacy_integer_component(images) -> None:
    code = 700
    word = encode_input_word(
        syndrome_code=code,
        syndrome_x="g",
        syndrome_z="g",
        quadrature_phase_bit=0,
        ood_score_code=0,
        parameter_age_code=0,
    )
    integrated = BitAccurateHardwareReference(images)
    integrated.step_word(word)
    for _ in range(5):
        integrated.step_word(
            encode_input_word(
                syndrome_code=0,
                syndrome_x="g",
                syndrome_z="g",
                quadrature_phase_bit=0,
                ood_score_code=0,
                parameter_age_code=0,
            )
        )
    output = integrated.step_word(
        encode_input_word(
            syndrome_code=0,
            syndrome_x="g",
            syndrome_z="g",
            quadrature_phase_bit=0,
            ood_score_code=0,
            parameter_age_code=0,
        )
    )
    packed, ok, _, _ = OUTPUT_SCHEMA.unpack(int(output.output_word_hex, 16))
    assert ok and packed["output_valid"] == 1

    legacy = BitAccurateFastPath(images)
    image = images[0]
    result = legacy.step_codes(
        FastPathCodeInput(
            cycle_index=5,
            syndrome_code=code,
            syndrome_x="g",
            syndrome_z="g",
            quadrature_phase_bit=0,
            expected_active_bank_version=0,
            reported_image_crc32=image.image_crc32,
            reported_image_sha256=image.image_sha256,
            parameter_age_code=0,
            ood_score_code=0,
        )
    )
    hardware = result.fallback_action.hardware_action
    assert packed["map_action_code"] == {"I": 0, "X": 1, "Z": 2}[
        hardware.map_logical_action
    ]
    assert packed["pauli_frame_x"] == int(hardware.pauli_frame_x)
    assert packed["phase_frame_x_code"] == hardware.phase_frame_x_code


def test_atomic_commit_preserves_inflight_old_image_and_latches_new_image(images) -> None:
    config = AtomicParameterBankConfig(
        min_residency_cycles=1, max_payload_age_cycles=100
    )
    reference = BitAccurateHardwareReference(images, bank_config=config)
    reference.stage_packed_update(
        pack_parameter_image(images[1]),
        transaction_id="test-v1",
        selection_key="v1",
        source_window_id=2,
        created_cycle=0,
        apply_cycle=10,
    )
    for cycle in range(18):
        reference.step_word(
            encode_input_word(
                syndrome_code=(cycle * 73) % 1024,
                syndrome_x="g",
                syndrome_z="g",
                quadrature_phase_bit=cycle % 2,
                ood_score_code=0,
                parameter_age_code=0,
            ),
            safe_boundary=cycle != 10,
        )
    trace = reference.trace
    assert trace[10].commit_status == "deferred"
    assert trace[11].commit_status == "committed"
    by_source = {row.output_source_cycle: row for row in trace if row.output_valid}
    old, ok, _, _ = OUTPUT_SCHEMA.unpack(int(by_source[10].output_word_hex, 16))
    assert ok and old["active_version"] == 0
    new, ok, _, _ = OUTPUT_SCHEMA.unpack(int(by_source[11].output_word_hex, 16))
    assert ok and new["active_version"] == 1


def test_crc_fault_is_latched_with_source_and_blocks_action_six_cycles_later(images) -> None:
    reference = BitAccurateHardwareReference(images)
    for cycle in range(8):
        word = encode_input_word(
            syndrome_code=(cycle * 31) % 1024,
            syndrome_x="g",
            syndrome_z="g",
            quadrature_phase_bit=0,
            ood_score_code=0,
            parameter_age_code=0,
        )
        if cycle == 0:
            word ^= 1
        reference.step_word(word)
    fault = reference.trace[6]
    values, ok, _, _ = OUTPUT_SCHEMA.unpack(int(fault.output_word_hex, 16))
    assert ok and fault.output_source_cycle == 0
    assert values["fault_mask"] != 0
    assert values["correction_enable"] == 0


def test_formal_exhaustive_trace_and_repeatability_evidence(report: dict) -> None:
    exhaustive = report["exhaustive_map_summary"]
    assert exhaustive["rows"] == 16_384
    assert exhaustive["llr_code_mismatch_count"] == 0
    assert exhaustive["action_mismatch_count"] == 0
    trace = report["trace_summary"]
    assert trace["hardware_cycles"] == 4116
    assert trace["output_valid_cycles"] == 4110
    assert trace["source_to_output_latencies"] == [6]
    assert trace["deferred_commit_cycle"] == 4000
    assert trace["committed_cycle"] == 4001
    assert trace["inflight_old_version_at_source_4000"] == 0
    assert trace["post_commit_version_at_source_4001"] == 1
    assert report["repeatability"]["trace_rows_equal"] is True
    assert report["golden_trace"]["chain_valid"] is True
    assert report["golden_trace"]["final_chain_sha256"] == trace[
        "final_trace_sha256"
    ]


def test_online_step_ast_has_no_float_division_or_truth_input() -> None:
    source = textwrap.dedent(inspect.getsource(BitAccurateHardwareReference.step_word))
    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.Div, ast.FloorDiv)) for node in ast.walk(tree))
    assert all(token not in source for token in ("truth", "hidden_state", "drift_state"))
    assert set(inspect.signature(BitAccurateHardwareReference.step_word).parameters) == {
        "self",
        "input_word",
        "safe_boundary",
    }


def test_contract_keeps_hardware_fields_null() -> None:
    contract = hardware_reference_contract()
    assert contract["source_to_output_cycles"] == 6
    assert contract["parameter_bank"]["banks"] == 2
    assert contract["parameter_image"]["binary_runtime_float_operations"] == 0
    hardware = contract["hardware_fields"]
    assert hardware["rtl_generated"] is False
    assert hardware["synthesized"] is False
    assert hardware["board_measured"] is False
    assert all(
        hardware[name] is None
        for name in (
            "fmax_mhz",
            "target_lut_count",
            "target_ff_count",
            "target_bram_count",
            "target_dsp_count",
        )
    )


def test_trace_bundle_and_source_files_are_complete_and_byte_bound(report: dict) -> None:
    with TRACE.open(newline="", encoding="utf-8") as stream:
        trace_rows = list(csv.DictReader(stream))
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    assert len(trace_rows) == report["golden_trace"]["rows"] == 4116
    assert len(source_rows) == report["source_data"]["rows"] == 16_503
    assert hashlib.sha256(TRACE.read_bytes()).hexdigest() == report["golden_trace"][
        "sha256"
    ]
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == report[
        "parameter_bundle"
    ]["sha256"]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == report["source_data"][
        "sha256"
    ]
    assert trace_rows[-1]["trace_chain_sha256"] == report["golden_trace"][
        "final_chain_sha256"
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "input_width",
        "binary_roundtrip",
        "map_mismatch",
        "latency",
        "commit_cycle",
        "inflight_version",
        "hide_fault",
        "repeatability",
        "source_unbound",
        "hardware_claim",
    ),
)
def test_validator_rejects_golden_reference_shortcuts(
    report: dict, mutation: str
) -> None:
    changed = copy.deepcopy(report)
    if mutation == "input_width":
        changed["contract"]["input_word"]["word_bits"] = 57
    elif mutation == "binary_roundtrip":
        changed["binary_parameter_audit"]["all_roundtrip_exact"] = False
    elif mutation == "map_mismatch":
        changed["exhaustive_map_summary"]["llr_code_mismatch_count"] = 1
    elif mutation == "latency":
        changed["trace_summary"]["source_to_output_latencies"] = [5, 6]
    elif mutation == "commit_cycle":
        changed["trace_summary"]["committed_cycle"] = 4000
    elif mutation == "inflight_version":
        changed["trace_summary"]["inflight_old_version_at_source_4000"] = 1
    elif mutation == "hide_fault":
        changed["trace_summary"]["bad_input_crc_output_fault_mask"] = 0
    elif mutation == "repeatability":
        changed["repeatability"]["trace_rows_equal"] = False
    elif mutation == "source_unbound":
        changed["golden_trace"]["sha256"] = None
    elif mutation == "hardware_claim":
        hardware = changed["claim_boundary"]["hardware_fields"]
        hardware["rtl_generated"] = True
        hardware["synthesized"] = True
        hardware["fmax_mhz"] = 120.0
    errors = validation.validate_artifact(_rehash(changed), check_files=False)
    assert errors
    assert any("stored gates" in error or "gate recomputation" in error for error in errors)
