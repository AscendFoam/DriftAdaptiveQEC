from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.decoder.linear_runtime import LinearRuntime, LinearRuntimeConfig
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from physics.control_imperfections import (
    CONTROL_IMPERFECTION_SCOPE,
    ControlActionRequest,
    ControlImperfectionConfig,
    ControlImperfectionModel,
    ideal_control_imperfection_config,
    run_control_imperfection_validation,
    write_control_imperfection_validation,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(**changes) -> ControlImperfectionConfig:
    base = ControlImperfectionConfig(
        enable_quantization=True,
        awg_amplitude_bits=8,
        awg_phase_bits=9,
        awg_amplitude_full_scale=2.0,
        dac_bits=10,
        dac_full_scale=2.0,
        virtual_rotation_bits=10,
        pulse_gain_matrix=np.eye(2),
        pulse_bias=(0.0, 0.0),
        active_relative_gain_sigma=0.0,
        active_displacement_covariance=np.zeros((2, 2)),
        virtual_rotation_gain=1.0,
        virtual_rotation_bias_rad=0.0,
        virtual_rotation_noise_sigma_rad=0.0,
        latency_drift_per_us=(0.0, 0.0),
        latency_diffusion_covariance_per_us=np.zeros((2, 2)),
        max_latency_us=20.0,
        action_order="displacement_then_virtual_rotation",
        quantization_provenance="unit-test explicit quantizer assumption",
        pulse_provenance="unit-test explicit pulse assumption",
        latency_provenance="unit-test explicit latency assumption",
    )
    return replace(base, **changes)


def _request(
    *,
    correction: tuple[float, float] = (0.4, -0.3),
    rotation: float = 0.2,
    latency: float = 0.0,
    cycle: int = 0,
) -> ControlActionRequest:
    return ControlActionRequest(
        cycle_index=cycle,
        correction_command=correction,
        virtual_rotation_command_rad=rotation,
        latency_us=latency,
        parameter_bank_version=3,
    )


def test_unsigned_awg_quantizer_uses_all_codes_and_reports_clipping() -> None:
    quantize = ControlImperfectionModel._unsigned_quantize
    assert quantize(0.0, bits=4, full_scale=1.0) == (0, 0.0, False)
    assert quantize(1.0, bits=4, full_scale=1.0) == (15, 1.0, False)
    assert quantize(1.2, bits=4, full_scale=1.0) == (15, 1.0, True)


def test_signed_dac_quantizer_is_two_complement_style_and_saturates() -> None:
    quantize = ControlImperfectionModel._signed_quantize
    assert quantize(-1.0, bits=4, full_scale=1.0) == (-8, -1.0, False)
    assert quantize(0.875, bits=4, full_scale=1.0) == (7, 0.875, False)
    assert quantize(2.0, bits=4, full_scale=1.0) == (7, 0.875, True)


def test_phase_quantizer_wraps_periodically_with_integer_code() -> None:
    code, angle = ControlImperfectionModel._phase_quantize(
        5.0 * np.pi / 2.0,
        bits=4,
    )
    assert code == 4
    assert angle == pytest.approx(np.pi / 2.0)


def test_encoding_runs_awg_polar_then_dac_iq_quantization() -> None:
    model = ControlImperfectionModel(
        _config(
            awg_amplitude_bits=4,
            awg_phase_bits=2,
            awg_amplitude_full_scale=1.0,
            dac_bits=4,
            dac_full_scale=1.0,
            virtual_rotation_bits=4,
        )
    )
    record = model.encode(_request(correction=(0.0, 0.5), rotation=np.pi / 2.0))
    assert record.awg_amplitude_code == 8
    assert record.awg_phase_code == 1
    assert record.dac_iq_codes == (0, 4)
    assert record.correction_commanded == pytest.approx((0.0, 0.5), abs=1.0e-15)
    assert record.virtual_rotation_code == 4
    assert not record.displacement_saturated


def test_zero_displacement_has_defined_zero_phase() -> None:
    record = ControlImperfectionModel(_config()).encode(_request(correction=(0.0, 0.0)))
    assert record.awg_amplitude_requested == 0.0
    assert record.awg_phase_requested_rad == 0.0
    assert record.correction_commanded == (0.0, 0.0)


def test_encoding_reports_awg_or_dac_saturation_without_hiding_requested_value() -> None:
    model = ControlImperfectionModel(
        _config(awg_amplitude_full_scale=0.5, dac_full_scale=0.25)
    )
    record = model.encode(_request(correction=(2.0, 0.0)))
    assert record.correction_requested == (2.0, 0.0)
    assert record.displacement_saturated
    assert record.awg_amplitude_command == 0.5
    assert record.correction_commanded[0] < 0.25


def test_affine_pulse_miscalibration_changes_physical_mean_not_command_record() -> None:
    model = ControlImperfectionModel(
        _config(
            enable_quantization=False,
            pulse_gain_matrix=np.asarray([[1.1, 0.2], [-0.1, 0.9]]),
            pulse_bias=(0.03, -0.02),
        )
    )
    step = model.step(
        _request(correction=(0.4, -0.3), rotation=0.0),
        residual_before_latency=(0.0, 0.0),
        seed=1,
    )
    expected = np.asarray([[1.1, 0.2], [-0.1, 0.9]]) @ np.asarray([0.4, -0.3])
    expected += np.asarray([0.03, -0.02])
    assert step.record.correction_commanded == (0.4, -0.3)
    assert step.truth.pulse_mean_displacement == pytest.approx(expected)
    assert step.truth.actual_displacement == pytest.approx(expected)


def test_latency_drift_is_accumulated_before_active_correction() -> None:
    model = ControlImperfectionModel(
        _config(
            enable_quantization=False,
            latency_drift_per_us=(0.01, -0.02),
        )
    )
    step = model.step(
        _request(correction=(0.2, 0.1), rotation=0.0, latency=5.0),
        residual_before_latency=(0.3, 0.4),
        seed=2,
    )
    assert step.truth.latency_drift == pytest.approx((0.05, -0.1))
    assert step.truth.residual_at_action == pytest.approx((0.35, 0.3))
    assert step.truth.residual_after_action == pytest.approx((0.15, 0.2))


def test_latency_diffusion_covariance_scales_linearly_with_reported_latency() -> None:
    covariance_rate = np.asarray([[0.003, 0.001], [0.001, 0.002]])
    model = ControlImperfectionModel(
        _config(
            enable_quantization=False,
            latency_diffusion_covariance_per_us=covariance_rate,
        )
    )
    batch = model.sample_fixed_request(
        _request(correction=(0.0, 0.0), rotation=0.0, latency=4.0),
        residual_before_latency=(0.0, 0.0),
        samples=50_000,
        seed=3,
    )
    empirical = np.cov(batch.latency_diffusion, rowvar=False, ddof=1)
    assert np.allclose(empirical, 4.0 * covariance_rate, rtol=0.025, atol=2.0e-4)


def test_multiplicative_active_error_has_rank_one_command_dependent_covariance() -> None:
    sigma = 0.08
    command = np.asarray([0.7, -0.2])
    model = ControlImperfectionModel(
        _config(
            enable_quantization=False,
            active_relative_gain_sigma=sigma,
        )
    )
    batch = model.sample_fixed_request(
        _request(correction=tuple(command), rotation=0.0),
        residual_before_latency=(0.0, 0.0),
        samples=50_000,
        seed=4,
    )
    empirical = np.cov(batch.actual_displacement, rowvar=False, ddof=1)
    expected = sigma**2 * np.outer(command, command)
    assert np.allclose(empirical, expected, rtol=0.03, atol=2.0e-5)


def test_virtual_rotation_error_rotates_post_displacement_residual() -> None:
    model = ControlImperfectionModel(
        _config(
            enable_quantization=False,
            virtual_rotation_bias_rad=np.pi / 2.0,
        )
    )
    step = model.step(
        _request(correction=(0.2, 0.0), rotation=0.0),
        residual_before_latency=(1.0, 0.0),
        seed=5,
    )
    assert step.truth.virtual_rotation_error_rad == pytest.approx(np.pi / 2.0)
    assert step.truth.residual_after_action == pytest.approx((0.0, 0.8), abs=1.0e-15)


def test_action_order_is_explicit_and_changes_noncommuting_result() -> None:
    base = _config(
        enable_quantization=False,
        virtual_rotation_bias_rad=np.pi / 2.0,
    )
    request = _request(correction=(0.2, 0.0), rotation=0.0)
    after = ControlImperfectionModel(base).step(
        request,
        residual_before_latency=(1.0, 0.0),
        seed=6,
    )
    before = ControlImperfectionModel(
        replace(base, action_order="virtual_rotation_then_displacement")
    ).step(
        request,
        residual_before_latency=(1.0, 0.0),
        seed=6,
    )
    assert after.truth.residual_after_action == pytest.approx((0.0, 0.8), abs=1.0e-15)
    assert before.truth.residual_after_action == pytest.approx((-0.2, 1.0), abs=1.0e-15)


def test_ideal_endpoint_exactly_cancels_residual_and_virtual_frame() -> None:
    model = ControlImperfectionModel(ideal_control_imperfection_config())
    step = model.step(
        _request(correction=(0.7, -0.4), rotation=0.37),
        residual_before_latency=(0.7, -0.4),
        seed=7,
    )
    assert step.truth.virtual_rotation_error_rad == 0.0
    assert step.truth.residual_after_action == (0.0, 0.0)


@pytest.mark.parametrize(
    "action_order",
    [
        "displacement_then_virtual_rotation",
        "virtual_rotation_then_displacement",
    ],
)
def test_exact_analytic_moments_match_vectorized_monte_carlo(action_order: str) -> None:
    model = ControlImperfectionModel(
        _config(
            enable_quantization=True,
            pulse_gain_matrix=np.asarray([[1.02, 0.01], [-0.02, 0.98]]),
            pulse_bias=(0.01, -0.005),
            active_relative_gain_sigma=0.025,
            active_displacement_covariance=np.asarray(
                [[0.0015, 0.0003], [0.0003, 0.001]]
            ),
            virtual_rotation_gain=1.01,
            virtual_rotation_bias_rad=0.01,
            virtual_rotation_noise_sigma_rad=0.03,
            latency_drift_per_us=(0.002, -0.001),
            latency_diffusion_covariance_per_us=np.asarray(
                [[0.0005, 0.0001], [0.0001, 0.0004]]
            ),
            action_order=action_order,
        )
    )
    request = _request(correction=(0.6, -0.35), rotation=0.4, latency=3.0)
    pre = (0.74, -0.22)
    batch = model.sample_fixed_request(
        request,
        residual_before_latency=pre,
        samples=80_000,
        seed=8,
    )
    mean, covariance = model.analytic_residual_moments(
        request,
        residual_before_latency=pre,
    )
    standard_error = np.sqrt(np.diag(covariance) / batch.samples)
    assert np.all(np.abs(batch.empirical_mean - mean) <= 5.0 * standard_error)
    relative = np.linalg.norm(batch.empirical_covariance - covariance) / np.linalg.norm(
        covariance
    )
    assert float(relative) < 0.025


def test_vectorized_batch_seed_replay_and_prefix_are_component_exact() -> None:
    model = ControlImperfectionModel(
        _config(
            active_relative_gain_sigma=0.02,
            active_displacement_covariance=np.eye(2) * 0.001,
            virtual_rotation_noise_sigma_rad=0.01,
            latency_diffusion_covariance_per_us=np.eye(2) * 0.0002,
        )
    )
    request = _request(latency=4.0)
    short = model.sample_fixed_request(
        request,
        residual_before_latency=(0.2, -0.1),
        samples=12,
        seed=9,
    )
    replay = model.sample_fixed_request(
        request,
        residual_before_latency=(0.2, -0.1),
        samples=12,
        seed=9,
    )
    long = model.sample_fixed_request(
        request,
        residual_before_latency=(0.2, -0.1),
        samples=24,
        seed=9,
    )
    for name in (
        "latency_diffusion",
        "active_relative_gain_error",
        "active_additive_error",
        "actual_displacement",
        "virtual_rotation_noise_rad",
        "residual_after_action",
    ):
        assert np.array_equal(getattr(short, name), getattr(replay, name))
        assert np.array_equal(getattr(short, name), getattr(long, name)[:12])


def test_vectorized_truth_arrays_are_read_only() -> None:
    batch = ControlImperfectionModel(_config()).sample_fixed_request(
        _request(),
        residual_before_latency=(0.2, -0.1),
        samples=4,
        seed=10,
    )
    with pytest.raises(ValueError, match="read-only"):
        batch.residual_after_action[0, 0] = 1.0


def test_sequential_trajectory_carries_physical_residual_and_replays() -> None:
    model = ControlImperfectionModel(
        _config(
            enable_quantization=False,
            latency_drift_per_us=(0.01, 0.0),
        )
    )
    requests = (
        _request(correction=(0.1, 0.0), rotation=0.0, latency=1.0, cycle=4),
        _request(correction=(0.1, 0.0), rotation=0.0, latency=1.0, cycle=5),
    )
    first = model.simulate(requests, initial_residual=(0.3, 0.0), seed=11)
    replay = model.simulate(requests, initial_residual=(0.3, 0.0), seed=11)
    assert first == replay
    assert first.steps[0].truth.residual_after_action == pytest.approx((0.21, 0.0))
    assert first.final_residual == pytest.approx((0.12, 0.0))


def test_deployable_record_contains_codes_but_excludes_physical_truth() -> None:
    model = ControlImperfectionModel(
        _config(
            active_relative_gain_sigma=0.1,
            active_displacement_covariance=np.eye(2) * 0.01,
        )
    )
    step = model.step(
        _request(latency=2.0),
        residual_before_latency=(0.5, -0.2),
        seed=12,
    )
    serialized = json.dumps(step.deployable_record())
    assert "dac_iq_codes" in serialized
    for forbidden in (
        "actual_displacement",
        "residual_after_action",
        "gain_error",
        "latency_diffusion",
        "rotation_noise",
        "truth",
    ):
        assert forbidden not in serialized


def test_reported_command_can_feed_control_memory_without_claiming_physical_truth() -> None:
    record = ControlImperfectionModel(_config()).encode(_request())
    assert record.record_scope.endswith("not_physical_realization")
    assert record.correction_commanded != record.correction_requested


def test_existing_q4_20_fast_loop_output_feeds_physical_dac_awg_layer() -> None:
    runtime = LinearRuntime(
        LinearRuntimeConfig(
            fixed_point_spec="Q4.20",
            enable_fixed_point=True,
            syndrome_limit=2.0,
            correction_limit=2.0,
        )
    )
    digital = runtime.decode(
        np.asarray([0.37, -0.29]),
        DecoderRuntimeParams(K=np.eye(2), b=np.zeros(2)),
    )
    model = ControlImperfectionModel(_config())
    record = model.encode(
        _request(correction=tuple(float(value) for value in digital.correction_applied))
    )
    assert tuple(digital.correction_applied) == pytest.approx(
        tuple(digital.syndrome_used)
    )
    assert record.correction_requested == pytest.approx(digital.correction_applied)
    assert record.record_scope.endswith("not_physical_realization")


def test_validation_checks_quantization_analytic_covariance_latency_and_ideal_endpoint() -> None:
    result = run_control_imperfection_validation(samples=100_000, seed=2026071423)
    assert all(bool(value) for value in result.checks.values())
    assert result.maximum_mean_z_score < 5.0
    assert result.covariance_relative_frobenius_error < 0.02
    assert all(
        later < earlier
        for earlier, later in zip(
            result.quantization_rms_error,
            result.quantization_rms_error[1:],
        )
    )
    assert result.pulse_systematic_displacement_error_norm > 0.0
    assert result.virtual_rotation_systematic_error_rad != 0.0
    assert result.latency_covariance_trace == tuple(
        sorted(result.latency_covariance_trace)
    )
    assert result.ideal_endpoint_max_abs_residual == 0.0
    assert result.evidence_scope == CONTROL_IMPERFECTION_SCOPE


def test_validation_writer_round_trips_json() -> None:
    result = run_control_imperfection_validation(samples=10_000, seed=2026071424)
    output = ROOT / "docs" / "_test_control_imperfection_validation.json"
    try:
        path = write_control_imperfection_validation(result, output)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["samples"] == 10_000
        assert all(isinstance(value, bool) for value in payload["checks"].values())
        assert payload["claim_boundary"]["forbidden"].startswith("device-calibrated")
    finally:
        output.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("factory", "pattern"),
    [
        (lambda: _config(dac_bits=1), "at least 2"),
        (lambda: _config(awg_phase_bits=31), "at most 30"),
        (
            lambda: _config(active_displacement_covariance=[[1.0, 2.0], [2.0, 1.0]]),
            "positive semidefinite",
        ),
        (
            lambda: _config(latency_diffusion_covariance_per_us=[[1.0, 0.2], [0.1, 1.0]]),
            "symmetric",
        ),
        (lambda: _config(action_order="hidden_order"), "action_order"),
        (lambda: _config(quantization_provenance=""), "non-empty"),
        (lambda: _request(latency=-1.0), "non-negative"),
        (
            lambda: ControlActionRequest(
                cycle_index=0,
                correction_command=(0.0, 0.0),
                virtual_rotation_command_rad=0.0,
                latency_us=0.0,
                protocol_id="PROTO-SHARPEN-TRIM-XVAL",
            ),
            "protocol_id",
        ),
    ],
)
def test_invalid_configuration_and_requests_fail_closed(factory, pattern: str) -> None:
    with pytest.raises((TypeError, ValueError), match=pattern):
        factory()


def test_max_latency_and_nonconsecutive_trajectory_fail_closed() -> None:
    model = ControlImperfectionModel(_config(max_latency_us=2.0))
    with pytest.raises(ValueError, match="exceeds"):
        model.encode(_request(latency=2.1))
    with pytest.raises(ValueError, match="consecutive"):
        model.simulate(
            (_request(cycle=0), _request(cycle=2)),
            initial_residual=(0.0, 0.0),
            seed=13,
        )


def test_validation_rejects_demo_scale_and_boolean_seed() -> None:
    with pytest.raises(ValueError, match=">= 10000"):
        run_control_imperfection_validation(samples=9_999)
    with pytest.raises(TypeError, match="seed must be an integer"):
        run_control_imperfection_validation(samples=10_000, seed=True)
