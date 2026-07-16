from __future__ import annotations

import ast
import inspect
import math
import textwrap

import pytest

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.fast_path_fixed_point import (
    BitAccurateFastPath,
    FastPathCodeInput,
    FastPathFixedPointContract,
    build_code_input_from_replay,
    encode_syndrome_replay,
    encode_unit_interval_replay,
    encode_unsigned_age_replay,
)
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig


def _images(config: ParametricMAPLUTConfig | None = None):
    cfg = ParametricMAPLUTConfig() if config is None else config
    return tuple(
        compile_parametric_map_lut(params, active_bank_version=index, config=cfg)
        for index, (params, _) in enumerate(registered_parameter_profiles(cfg))
    )


def test_contract_freezes_all_widths_and_exact_representation_proxy() -> None:
    contract = FastPathFixedPointContract()
    assert contract.llr_word_bits == 22
    assert contract.source_to_action_cycles == 6
    proxy = contract.representation_proxy()
    assert proxy == {
        "rom_entries_per_phase": 257,
        "rom_bits_per_bank": 11_308,
        "double_bank_rom_bits": 22_616,
        "registered_eight_bank_artifact_bits": 90_464,
        "live_event_state_bits": 55,
        "live_health_state_and_input_bits": 182,
        "integrity_metadata_bits_per_image": 288,
    }
    with pytest.raises(ValueError, match="adc_bits-address_bits"):
        FastPathFixedPointContract(interpolation_fraction_bits=3)
    with pytest.raises(ValueError, match=r"5\+1"):
        FastPathFixedPointContract(health_event_register_cycles=2)


def test_replay_quantizers_have_explicit_rounding_saturation_and_half_open_cell() -> None:
    image = _images()[0]
    half = image.config.lattice / 2.0
    assert encode_syndrome_replay(-half, image) == (0, False, True)
    high = encode_syndrome_replay(math.nextafter(half, -half), image)
    assert high == (1023, False, True)
    assert encode_syndrome_replay(half, image) == (1023, True, False)
    assert encode_syndrome_replay(float("nan"), image) == (0, True, False)

    assert encode_unit_interval_replay(0.5 / 255.0, 8) == (0, False)
    assert encode_unit_interval_replay(1.5 / 255.0, 8) == (2, False)
    assert encode_unit_interval_replay(1.2, 8) == (255, True)
    assert encode_unsigned_age_replay(65_535, 16) == (65_535, False)
    assert encode_unsigned_age_replay(65_536, 16) == (65_535, True)


def test_bit_accurate_path_composes_integer_map_health_event_and_frame() -> None:
    images = _images()
    image = images[1]
    path = BitAccurateFastPath(images)
    code_input, saturation = build_code_input_from_replay(
        cycle_index=5,
        syndrome=-0.4 * image.config.lattice,
        syndrome_x="g",
        syndrome_z="g",
        quadrature_phase_bit=0,
        image=image,
        parameter_age_cycles=64,
        ood_score=192.0 / 255.0,
    )
    result = path.step_codes(code_input)
    assert saturation == {
        "syndrome_saturated": False,
        "ood_saturated": False,
        "age_saturated": False,
    }
    assert result.map_decision is not None
    assert result.fallback_action.map_decision_accepted
    assert result.fallback_action.hardware_action.action_cycle == 6
    assert result.fallback_action.hardware_action.source_cycle == 0
    assert result.fallback_action.trusted_active_bank_version == 1
    assert path.history == (result,)


def test_invalid_float_syndrome_becomes_traceable_fallback_not_fake_zero() -> None:
    images = _images()
    image = images[0]
    path = BitAccurateFastPath(images)
    code_input, saturation = build_code_input_from_replay(
        cycle_index=5,
        syndrome=float("nan"),
        syndrome_x="g",
        syndrome_z="g",
        quadrature_phase_bit=0,
        image=image,
        parameter_age_cycles=0,
        ood_score=0.0,
    )
    assert code_input.syndrome_code == 0
    assert not code_input.observation_valid
    assert saturation["syndrome_saturated"]
    result = path.step_codes(code_input)
    assert result.map_decision is None
    assert "observation_invalid" in result.fallback_action.fault_flags
    assert "map_decision_missing" in result.fallback_action.fault_flags
    assert result.fallback_action.conservative_action == "frame_hold"


def test_unknown_version_and_code_width_overflows_fail_closed_transactionally() -> None:
    images = _images()
    path = BitAccurateFastPath(images)
    image = images[0]
    unknown = FastPathCodeInput(
        5,
        0,
        "g",
        "g",
        0,
        8,
        image.image_crc32,
        image.image_sha256,
        0,
        0,
    )
    result = path.step_codes(unknown)
    assert "unknown_bank_version" in result.fallback_action.fault_flags
    assert result.map_decision is None

    path = BitAccurateFastPath(images)
    overflow = FastPathCodeInput(
        5,
        1024,
        "g",
        "g",
        0,
        0,
        image.image_crc32,
        image.image_sha256,
        0,
        0,
    )
    before = path.state
    with pytest.raises(ValueError, match="ADC word width"):
        path.step_codes(overflow)
    assert path.state == before and path.history == ()


def test_reset_restores_cycle_state_and_history() -> None:
    images = _images()
    path = BitAccurateFastPath(images)
    image = images[0]
    event = FastPathCodeInput(
        5, 0, "g", "g", 0, 0, image.image_crc32, image.image_sha256, 0, 0
    )
    path.step_codes(event)
    path.reset()
    assert path.history == () and path.state.cycle_index == 4
    assert path.step_codes(event).fallback_action.hardware_action.action_cycle == 6


def test_online_step_is_integer_only_and_schema_contains_no_truth() -> None:
    fields = set(FastPathCodeInput.__dataclass_fields__)
    assert not fields & {"truth", "logical_truth", "hidden_state", "drift_state"}
    tree = ast.parse(textwrap.dedent(inspect.getsource(BitAccurateFastPath.step_codes)))
    assert not any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) for node in ast.walk(tree))
    source = inspect.getsource(BitAccurateFastPath.step_codes)
    assert not any(token in source for token in ("math.exp", "math.log", "np.exp", "np.log"))
