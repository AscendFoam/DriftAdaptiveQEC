from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import replace

import numpy as np
import pytest

from cnn_fpga.decoder.parametric_map_lut import (
    compile_active_param_bank,
    compile_parametric_map_lut,
    derive_axis_map_model,
    exact_quantized_llr_code,
    source_params_sha256,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTConfig,
    ParametricMAPLUTInput,
    ParametricMAPLUTPipeline,
    ParametricMAPLUTRuntime,
    _rounded_signed_shift,
    resource_contract,
    software_decode_syndrome_code,
    software_encode_syndrome_for_replay,
)
from physics.constants import LATTICE_CONST


def _params(
    *,
    mean: tuple[float, float] = (0.2, -0.1),
    sigma: tuple[float, float] = (0.5, 0.4),
    rho: float = 0.25,
    measurement_variance: float = 0.09,
) -> DecoderRuntimeParams:
    covariance = np.asarray(
        [
            [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
            [rho * sigma[0] * sigma[1], sigma[1] ** 2],
        ],
        dtype=np.float64,
    )
    measurement = np.eye(2, dtype=np.float64) * measurement_variance
    gain = covariance @ np.linalg.inv(covariance + measurement)
    bias = (np.eye(2, dtype=np.float64) - gain) @ np.asarray(mean)
    return DecoderRuntimeParams(
        K=gain,
        b=bias,
        metadata={
            "measurement_cov": measurement.tolist(),
            "alpha_bias": 1.0,
            "source": "t4.2.1-unit-effective-model",
        },
    )


def test_config_and_resource_contract_are_exact_but_not_hardware_measurements() -> None:
    config = ParametricMAPLUTConfig()
    resource = resource_contract(config)
    assert config.table_entries == 257
    assert config.llr_word_bits == 22
    assert resource["single_bank_table_bits"] == 2 * 257 * 22
    assert resource["dual_bank_table_bits"] == 2 * 2 * 257 * 22
    assert resource["worst_case_latency_cycles"] == 5
    assert resource["initiation_interval_cycles"] == 1
    assert resource["runtime_dividers"] == 0
    assert resource["runtime_exp_log_units"] == 0
    assert resource["target_bram_count"] is None
    assert resource["fmax_mhz"] is None
    assert resource["rtl_measured"] is False
    assert resource["board_measured"] is False


def test_active_k_b_inverse_recovers_effective_mean_and_full_covariance() -> None:
    params = _params(mean=(0.31, -0.27), sigma=(0.62, 0.37), rho=-0.55)
    model = derive_axis_map_model(params)
    assert model.mean == pytest.approx((0.31, -0.27), abs=2.0e-15)
    covariance = np.asarray(model.covariance)
    expected = np.asarray(
        [[0.62**2, -0.55 * 0.62 * 0.37], [-0.55 * 0.62 * 0.37, 0.37**2]]
    )
    np.testing.assert_allclose(covariance, expected, rtol=0.0, atol=3.0e-16)
    assert model.derivation_residual_max_abs < 2.0e-15


def test_compiler_binds_live_param_bank_version_and_source_hash() -> None:
    params = _params()
    bank = ParamBank(params)
    first = compile_active_param_bank(bank)
    assert first.active_bank_version == 0
    assert first.source_params_sha256 == source_params_sha256(bank.read_active())
    first.verify()

    changed = _params(mean=(-0.45, 0.38), sigma=(0.55, 0.48), rho=-0.3)
    bank.stage_update(changed, commit_epoch=1)
    bank.commit_if_ready(1)
    second = compile_active_param_bank(bank)
    assert second.active_bank_version == 1
    assert second.source_params_sha256 != first.source_params_sha256
    assert second.image_sha256 != first.image_sha256


def test_runtime_matches_exact_quantized_action_on_exhaustive_adc_grid() -> None:
    image = compile_parametric_map_lut(_params(), active_bank_version=7)
    runtime = ParametricMAPLUTRuntime(image)
    errors: list[int] = []
    for phase in (0, 1):
        for code in range(image.config.adc_levels):
            result = runtime.decode_code(ParametricMAPLUTInput(code, code, phase, 7))
            exact = exact_quantized_llr_code(code, phase, image)
            errors.append(abs(result.llr_code - exact))
            assert result.logical_flip == (exact < 0)
            assert result.logical_action == (("X", "Z")[phase] if exact < 0 else "I")
    assert max(errors) <= 12
    assert float(np.mean(errors)) < 0.7


def test_phase_bit_selects_distinct_axis_model_and_action_label() -> None:
    image = compile_parametric_map_lut(
        _params(mean=(0.52, -0.44), sigma=(0.62, 0.31), rho=0.4),
        active_bank_version=0,
    )
    runtime = ParametricMAPLUTRuntime(image)
    differing = 0
    for code in range(image.config.adc_levels):
        x = runtime.decode_code(ParametricMAPLUTInput(code, code, 0, 0))
        z = runtime.decode_code(ParametricMAPLUTInput(code, code, 1, 0))
        differing += x.llr_code != z.llr_code
        assert x.phase_label == "X"
        assert z.phase_label == "Z"
    assert differing > 0.95 * image.config.adc_levels


def test_interpolation_grid_converges_without_changing_hard_decisions() -> None:
    params = _params(
        mean=(0.35 * LATTICE_CONST, -0.28 * LATTICE_CONST),
        sigma=(0.25 * LATTICE_CONST, 0.18 * LATTICE_CONST),
        rho=-0.55,
    )
    mean_errors = []
    for address_bits in (5, 6, 7, 8):
        config = ParametricMAPLUTConfig(address_bits=address_bits)
        image = compile_parametric_map_lut(params, active_bank_version=0, config=config)
        runtime = ParametricMAPLUTRuntime(image)
        errors = []
        mismatches = 0
        for phase in (0, 1):
            for code in range(config.adc_levels):
                actual = runtime.decode_code(
                    ParametricMAPLUTInput(code, code, phase, 0)
                ).llr_code
                exact = exact_quantized_llr_code(code, phase, image)
                errors.append(abs(actual - exact))
                mismatches += (actual < 0) != (exact < 0)
        mean_errors.append(float(np.mean(errors)))
        assert mismatches == 0
    assert all(a > b for a, b in zip(mean_errors, mean_errors[1:]))
    assert mean_errors[-1] < 0.5


def test_adc_helper_uses_half_open_cell_and_level_centres() -> None:
    config = ParametricMAPLUTConfig(adc_bits=6, address_bits=4)
    half = 0.5 * config.lattice
    assert software_encode_syndrome_for_replay(-half, config) == 0
    assert software_encode_syndrome_for_replay(np.nextafter(half, -np.inf), config) == 63
    values = [software_decode_syndrome_code(code, config) for code in range(64)]
    assert values[0] > -half
    assert values[-1] < half
    assert np.diff(values) == pytest.approx(config.lattice / 64)
    with pytest.raises(ValueError, match="half-open"):
        software_encode_syndrome_for_replay(half, config)
    with pytest.raises(ValueError, match="ADC width"):
        software_decode_syndrome_code(64, config)


def test_signed_shift_uses_ties_to_even_for_positive_and_negative_values() -> None:
    assert _rounded_signed_shift(2, 2) == 0
    assert _rounded_signed_shift(6, 2) == 2
    assert _rounded_signed_shift(-2, 2) == 0
    assert _rounded_signed_shift(-6, 2) == -2
    assert _rounded_signed_shift(7, 2) == 2
    assert _rounded_signed_shift(-7, 2) == -2


def test_pipeline_has_exact_five_cycle_latency_and_initiation_interval_one() -> None:
    image = compile_parametric_map_lut(_params(), active_bank_version=0)
    pipeline = ParametricMAPLUTPipeline(image)
    outputs = []
    for cycle in range(13):
        request = (
            ParametricMAPLUTInput(cycle, 100 + cycle, cycle % 2, 0)
            if cycle < 8
            else None
        )
        output = pipeline.step(cycle, request)
        if output is not None:
            outputs.append(output)
    assert [item.input_cycle for item in outputs] == list(range(8))
    assert [item.valid_cycle for item in outputs] == list(range(5, 13))
    assert all(item.valid_cycle - item.input_cycle == 5 for item in outputs)


def test_pipeline_latches_old_image_for_inflight_request_across_bank_switch() -> None:
    old = compile_parametric_map_lut(_params(mean=(0.3, -0.2)), active_bank_version=0)
    new = compile_parametric_map_lut(_params(mean=(-0.5, 0.4)), active_bank_version=1)
    pipeline = ParametricMAPLUTPipeline(old)
    assert pipeline.step(0, ParametricMAPLUTInput(0, 700, 0, 0)) is None
    pipeline.load_image(new)
    assert pipeline.step(1, ParametricMAPLUTInput(1, 700, 0, 1)) is None
    for cycle in range(2, 5):
        assert pipeline.step(cycle) is None
    first = pipeline.step(5)
    second = pipeline.step(6)
    assert first is not None and first.image_sha256 == old.image_sha256
    assert second is not None and second.image_sha256 == new.image_sha256
    assert first.active_bank_version == 0
    assert second.active_bank_version == 1


def test_runtime_rejects_stale_version_bad_code_phase_and_nonmonotonic_cycle() -> None:
    image = compile_parametric_map_lut(_params(), active_bank_version=3)
    runtime = ParametricMAPLUTRuntime(image)
    with pytest.raises(ValueError, match="version mismatch"):
        runtime.decode_code(ParametricMAPLUTInput(0, 10, 0, 2))
    with pytest.raises(ValueError, match="ADC width"):
        runtime.decode_code(ParametricMAPLUTInput(0, image.config.adc_levels, 0, 3))
    with pytest.raises(ValueError, match="phase"):
        ParametricMAPLUTInput(0, 10, 2, 3)
    pipeline = ParametricMAPLUTPipeline(image)
    pipeline.step(0)
    with pytest.raises(ValueError, match="exactly one"):
        pipeline.step(2)


def test_image_crc_sha_and_source_binding_detect_tamper() -> None:
    image = compile_parametric_map_lut(_params(), active_bank_version=0)
    table = list(image.table_codes[0])
    table[10] += 1
    tampered = replace(image, table_codes=(tuple(table), image.table_codes[1]))
    with pytest.raises(ValueError, match="CRC mismatch"):
        ParametricMAPLUTRuntime(tampered)

    changed = _params(mean=(0.21, -0.1))
    changed_image = compile_parametric_map_lut(changed, active_bank_version=0)
    assert changed_image.source_params_sha256 != image.source_params_sha256
    assert changed_image.image_sha256 != image.image_sha256


@pytest.mark.parametrize(
    "params,match",
    [
        (DecoderRuntimeParams.identity(), "lacks"),
        (
            DecoderRuntimeParams(
                K=np.asarray([[0.5, 0.1], [0.0, 0.5]]),
                b=np.zeros(2),
                metadata={"measurement_cov": np.eye(2).tolist(), "alpha_bias": 1.0},
            ),
            "symmetric",
        ),
        (
            DecoderRuntimeParams(
                K=np.eye(2),
                b=np.zeros(2),
                metadata={"measurement_cov": np.eye(2).tolist(), "alpha_bias": 1.0},
            ),
            "inside",
        ),
    ],
)
def test_compiler_fails_closed_on_unidentified_or_invalid_bank(params, match) -> None:
    with pytest.raises(ValueError, match=match):
        derive_axis_map_model(params)


def test_online_kernel_ast_contains_no_float_division_exp_log_or_sqrt() -> None:
    for function in (ParametricMAPLUTRuntime.decode_code, _rounded_signed_shift):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        assert not any(isinstance(node, ast.Div) for node in ast.walk(tree))
        forbidden_calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not forbidden_calls & {"exp", "log", "sqrt", "pow"}


def test_param_mapper_metadata_now_exposes_bias_contract_for_compiler() -> None:
    from cnn_fpga.decoder.param_mapper import NoisePrediction, ParamMapper, ParamMapperConfig

    mapped = ParamMapper(ParamMapperConfig(alpha_bias=0.75, beta_smoothing=1.0)).map_prediction(
        NoisePrediction(0.45, 0.2, -0.1, 10.0)
    )
    assert mapped.metadata["alpha_bias"] == pytest.approx(0.75)
    model = derive_axis_map_model(mapped)
    assert model.mean == pytest.approx((0.2, -0.1), abs=2.0e-15)
