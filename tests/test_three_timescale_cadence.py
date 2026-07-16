from __future__ import annotations

import math

import pytest

from cnn_fpga.benchmark.parametric_map_lut_validation import registered_parameter_profiles
from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.fast_path_fixed_point import BitAccurateFastPath, FastPathCodeInput
from cnn_fpga.runtime.parametric_map_lut import ParametricMAPLUTConfig
from cnn_fpga.runtime.three_timescale_cadence import (
    MODEL_SCOPE,
    ThreeTimescaleCadence,
    ThreeTimescaleCadenceConfig,
)


def test_production_reference_ratios_are_exact() -> None:
    config = ThreeTimescaleCadenceConfig()

    assert config.model_scope == MODEL_SCOPE
    assert config.slow_period_cycles == 4000
    assert config.slow_service_cycles == 199
    assert config.recalibration_period_cycles == 12_000_000
    assert config.window_content_us == 10_240.0
    assert config.max_parameter_age_cycles == 8192


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"t_fast_us": 0.0}, ValueError),
        ({"window_size": True}, TypeError),
        ({"window_size": 4001}, ValueError),
        ({"slow_update_period_us": 20_001.0}, ValueError),
        ({"event_register_cycles": 2}, ValueError),
        ({"commit_delay_cycles": 2}, ValueError),
        ({"max_parameter_age_cycles": 7999}, ValueError),
        ({"max_parameter_age_cycles": 65536}, ValueError),
        ({"recalibration_period_us": 60_000_005.0}, ValueError),
    ],
)
def test_invalid_or_unaligned_contracts_fail_closed(
    override: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ThreeTimescaleCadenceConfig(**override)  # type: ignore[arg-type]


def test_reference_trace_phase_has_exact_component_closure() -> None:
    record = ThreeTimescaleCadence().adaptation_schedule(2040)

    assert record.window_start_epoch == 1
    assert record.window_end_epoch == 2048
    assert record.post_change_samples == 9
    assert record.slow_start_epoch == 2048
    assert record.slow_finish_epoch == 2247
    assert record.stage_epoch == 2247
    assert record.commit_epoch == 2248
    assert record.first_use_epoch == 2248
    assert record.total_lag_cycles == 208
    assert record.total_lag_us == 1040.0
    assert record.event_action_epoch == 2041


def test_phase_sweep_enumerates_every_wait_exactly_once() -> None:
    cadence = ThreeTimescaleCadence()
    records = cadence.phase_sweep(evidence_policy="first_influenced_window")

    lags = sorted(record.total_lag_cycles for record in records)
    waits = sorted(record.evidence_wait_cycles for record in records)
    assert len(records) == 4000
    assert waits == list(range(4000))
    assert lags == list(range(200, 4200))
    assert all(record.queue_wait_cycles == 0 for record in records)


def test_full_post_change_policy_never_uses_a_mixed_window() -> None:
    cadence = ThreeTimescaleCadence()
    records = cadence.phase_sweep(evidence_policy="first_full_post_change_window")

    assert min(record.total_lag_cycles for record in records) == 2247
    assert max(record.total_lag_cycles for record in records) == 6246
    assert all(record.post_change_samples == cadence.config.window_size for record in records)
    assert all(record.window_start_epoch >= record.onset_epoch for record in records)


def test_unsupported_evidence_policy_and_bad_epoch_are_rejected() -> None:
    cadence = ThreeTimescaleCadence()
    with pytest.raises(ValueError, match="unsupported evidence_policy"):
        cadence.adaptation_schedule(1, evidence_policy="future_truth")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="onset_epoch"):
        cadence.adaptation_schedule(0)


def test_recalibration_schedule_merges_coincident_minute_and_run_end() -> None:
    cadence = ThreeTimescaleCadence()
    period = cadence.config.recalibration_period_cycles

    triggers = cadence.recalibration_schedule(period)
    assert len(triggers) == 1
    assert triggers[0].epoch == period
    assert triggers[0].kinds == ("periodic_minute", "end_of_run")
    assert triggers[0].time_us == 60_000_000.0


def test_recalibration_can_exclude_end_of_run_and_validates_range() -> None:
    cadence = ThreeTimescaleCadence()
    assert cadence.recalibration_schedule(100, include_end_of_run=False) == ()
    with pytest.raises(ValueError, match="run_end_epoch"):
        cadence.recalibration_schedule(9, run_start_epoch=10)


def _healthy_age_event(path: BitAccurateFastPath, age: int) -> FastPathCodeInput:
    image = next(iter(path._images.values()))
    return FastPathCodeInput(
        cycle_index=5,
        syndrome_code=1 << (image.config.adc_bits - 1),
        syndrome_x="g",
        syndrome_z="g",
        quadrature_phase_bit=1,
        expected_active_bank_version=image.active_bank_version,
        reported_image_crc32=image.image_crc32,
        reported_image_sha256=image.image_sha256,
        parameter_age_code=age,
        ood_score_code=0,
    )


def test_fast_path_age_limit_is_configurable_for_real_slow_cadence() -> None:
    config = ParametricMAPLUTConfig()
    params = registered_parameter_profiles(config)[0][0]
    image = compile_parametric_map_lut(params, active_bank_version=0, config=config)

    short = BitAccurateFastPath((image,))
    short_result = short.step_codes(_healthy_age_event(short, 1000))
    assert "parameter_stale" in short_result.fallback_action.fault_flags

    production = BitAccurateFastPath((image,), max_parameter_age_cycles=8192)
    production_result = production.step_codes(_healthy_age_event(production, 1000))
    assert production_result.fallback_action.fault_flags == ()
    assert production_result.fallback_action.map_decision_accepted is True


def test_fast_path_age_limit_must_fit_the_frozen_age_word() -> None:
    config = ParametricMAPLUTConfig()
    params = registered_parameter_profiles(config)[0][0]
    image = compile_parametric_map_lut(params, active_bank_version=0, config=config)

    with pytest.raises(ValueError, match="parameter-age word width"):
        BitAccurateFastPath((image,), max_parameter_age_cycles=1 << 16)


def test_all_phase_lag_values_are_finite_and_nonnegative() -> None:
    cadence = ThreeTimescaleCadence()
    for policy in ("first_influenced_window", "first_full_post_change_window"):
        for record in cadence.phase_sweep(evidence_policy=policy):  # type: ignore[arg-type]
            assert record.total_lag_cycles >= 0
            assert math.isfinite(record.total_lag_us)

