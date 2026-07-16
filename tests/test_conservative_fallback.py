from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.conservative_fallback import (
    DEGRADED,
    FALLBACK_ACTIVE,
    FAULT_BITS,
    FAULT_ORDER,
    HEALTHY,
    RECOVERING,
    RESET_REQUIRED,
    ConservativeFallbackConfig,
    ConservativeFallbackController,
    ConservativeFallbackInput,
    TrustedParameterImage,
)
from cnn_fpga.runtime.experimental_event_fsm import FALLBACK, ExperimentalEventFSM, ExperimentalEventInput
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.parametric_map_lut import (
    ParametricMAPLUTInput,
    ParametricMAPLUTRuntime,
)


def _image(version: int = 0):
    covariance = np.asarray([[0.25, 0.04], [0.04, 0.16]], dtype=np.float64)
    measurement = np.eye(2) * 0.09
    gain = covariance @ np.linalg.inv(covariance + measurement)
    mean = np.asarray([0.25, -0.18])
    bias = (np.eye(2) - gain) @ mean
    params = DecoderRuntimeParams(
        K=gain,
        b=bias,
        metadata={"measurement_cov": measurement.tolist(), "alpha_bias": 1.0},
    )
    return compile_parametric_map_lut(params, active_bank_version=version)


def _trusted(image) -> TrustedParameterImage:
    return TrustedParameterImage(
        image.active_bank_version, image.image_crc32, image.image_sha256
    )


def _code(runtime: ParametricMAPLUTRuntime, phase: int, flip: bool) -> int:
    for code in range(runtime.image.config.adc_levels):
        decision = runtime.decode_code(
            ParametricMAPLUTInput(0, code, phase, runtime.image.active_bank_version)
        )
        if decision.logical_flip is flip:
            return code
    raise AssertionError("test image lacks requested action")


def _input(
    runtime: ParametricMAPLUTRuntime,
    cycle: int,
    *,
    phase: int = 0,
    flip: bool = False,
    syndrome_x: str = "g",
    syndrome_z: str = "g",
    map_decision="auto",
    expected_version: int | None = None,
    reported_crc: str | None = None,
    reported_sha: str | None = None,
    age: int = 0,
    ood: int = 0,
    reset_ack: bool = False,
    observation_valid: bool = True,
    input_crc_ok: bool = True,
    deadline_ok: bool = True,
) -> ConservativeFallbackInput:
    image = runtime.image
    if map_decision == "auto":
        decision = runtime.decode_code(
            ParametricMAPLUTInput(
                cycle - image.config.pipeline_latency_cycles,
                _code(runtime, phase, flip),
                phase,
                image.active_bank_version,
            )
        )
    else:
        decision = map_decision
    return ConservativeFallbackInput(
        cycle_index=cycle,
        syndrome_x=syndrome_x,
        syndrome_z=syndrome_z,
        quadrature_phase_bit=phase,
        map_decision=decision,
        expected_active_bank_version=(
            image.active_bank_version if expected_version is None else expected_version
        ),
        reported_image_crc32=image.image_crc32 if reported_crc is None else reported_crc,
        reported_image_sha256=image.image_sha256 if reported_sha is None else reported_sha,
        parameter_age_cycles=age,
        ood_score_code=ood,
        reset_ack=reset_ack,
        observation_valid=observation_valid,
        input_crc_ok=input_crc_ok,
        deadline_ok=deadline_ok,
    )


def test_fault_registry_config_and_trusted_image_are_strict() -> None:
    assert len(FAULT_ORDER) == len(FAULT_BITS) == 14
    assert len(set(FAULT_BITS.values())) == len(FAULT_BITS)
    assert sum(FAULT_BITS.values()) < 1 << len(FAULT_ORDER)
    config = ConservativeFallbackConfig()
    assert config.ood_code_max == 255
    assert config.health_counter_max == 255
    with pytest.raises(ValueError, match="exceeds"):
        ConservativeFallbackConfig(ood_score_bits=4, ood_threshold_code=16)
    with pytest.raises(ValueError, match="hexadecimal"):
        TrustedParameterImage(0, "bad", "0" * 64)


def test_healthy_map_is_accepted_and_updates_frame() -> None:
    image = _image()
    runtime = ParametricMAPLUTRuntime(image)
    controller = ConservativeFallbackController([_trusted(image)])
    action = controller.step(_input(runtime, 5, flip=True))
    assert action.status == HEALTHY
    assert action.fault_flags == () and action.fault_mask == 0
    assert action.map_decision_accepted
    assert action.conservative_action == "use_validated_map"
    assert action.hardware_action.pauli_frame_delta_x
    assert action.hardware_action.action_cycle == 6
    assert action.reason_trace == "fsm:no_event_threshold"


@pytest.mark.parametrize(
    ("expected_flag", "kwargs"),
    [
        ("observation_invalid", {"observation_valid": False}),
        ("ood_score_exceeded", {"ood": 193}),
        ("input_crc_mismatch", {"input_crc_ok": False}),
        ("image_crc_mismatch", {"reported_crc": "00000000"}),
        ("image_sha256_mismatch", {"reported_sha": "0" * 64}),
        ("parameter_stale", {"age": 65}),
        ("deadline_miss", {"deadline_ok": False}),
        ("map_decision_missing", {"map_decision": None}),
        ("unexpected_reset_ack", {"reset_ack": True}),
    ],
)
def test_each_operational_fault_falls_back_without_using_map(
    expected_flag: str, kwargs: dict
) -> None:
    image = _image()
    runtime = ParametricMAPLUTRuntime(image)
    controller = ConservativeFallbackController([_trusted(image)])
    action = controller.step(_input(runtime, 5, flip=True, **kwargs))
    assert expected_flag in action.fault_flags
    assert action.status == FALLBACK_ACTIVE
    assert action.hardware_action.mode == FALLBACK
    assert not action.map_decision_accepted
    assert action.conservative_action == "frame_hold"
    assert not action.hardware_action.correction_enable
    assert not action.hardware_action.pauli_frame_delta_x
    assert action.active_profile_id == controller.config.safe_profile_id


def test_alignment_and_logical_action_corruption_are_recorded_not_raised() -> None:
    image = _image()
    runtime = ParametricMAPLUTRuntime(image)
    controller = ConservativeFallbackController([_trusted(image)])
    base = _input(runtime, 5, flip=True)
    bad = replace(base, map_decision=replace(base.map_decision, logical_action="I"))
    action = controller.step(bad)
    assert action.fault_flags == ("map_alignment_or_action_invalid",)
    assert action.status == FALLBACK_ACTIVE
    assert action.reason_trace.endswith("fsm:health_fault:valid")


def test_unknown_mismatched_and_rollback_versions_do_not_replace_trusted_bank() -> None:
    image0 = _image(0)
    image1 = _image(1)
    runtime0 = ParametricMAPLUTRuntime(image0)
    runtime1 = ParametricMAPLUTRuntime(image1)
    controller = ConservativeFallbackController([_trusted(image0), _trusted(image1)])
    first = controller.step(_input(runtime1, 5))
    assert first.map_decision_accepted and first.trusted_active_bank_version == 1

    rollback = controller.step(_input(runtime0, 6))
    assert "bank_version_rollback" in rollback.fault_flags
    assert rollback.trusted_active_bank_version == 1
    assert controller.state.trusted_active_bank_version == 1

    controller = ConservativeFallbackController([_trusted(image0), _trusted(image1)])
    unknown = controller.step(_input(runtime0, 5, expected_version=9))
    assert "unknown_bank_version" in unknown.fault_flags
    assert "bank_version_mismatch" in unknown.fault_flags
    assert unknown.trusted_active_bank_version == 0


def test_leakage_is_degraded_then_reset_required_and_acknowledged() -> None:
    image = _image()
    runtime = ParametricMAPLUTRuntime(image)
    controller = ConservativeFallbackController([_trusted(image)])
    first = controller.step(_input(runtime, 5, syndrome_x="leakage", flip=True))
    second = controller.step(_input(runtime, 6, syndrome_z="leakage", flip=True))
    sticky = controller.step(_input(runtime, 7, flip=True))
    ack = controller.step(_input(runtime, 8, reset_ack=True, flip=True))
    assert first.status == DEGRADED and first.conservative_action == "frame_hold"
    assert first.map_decision_accepted and first.hardware_action.map_action_inhibited
    assert second.status == RESET_REQUIRED and second.conservative_action == "reset_request"
    assert sticky.status == RESET_REQUIRED
    assert ack.status == DEGRADED
    assert ack.hardware_action.reason == "reset_acknowledged_post_reset_hold"
    assert controller.state.leakage_cycle_count == 2


def test_fallback_requires_two_clean_cycles_before_map_resumes() -> None:
    image = _image()
    runtime = ParametricMAPLUTRuntime(image)
    controller = ConservativeFallbackController([_trusted(image)])
    failed = controller.step(_input(runtime, 5, deadline_ok=False, flip=True))
    first_good = controller.step(_input(runtime, 6, flip=True))
    second_good = controller.step(_input(runtime, 7, flip=True))
    assert failed.status == FALLBACK_ACTIVE
    assert first_good.status == RECOVERING
    assert first_good.map_decision_accepted
    assert first_good.hardware_action.map_action_inhibited
    assert second_good.status == HEALTHY
    assert second_good.hardware_action.pauli_frame_delta_x


def test_simultaneous_faults_preserve_all_reasons_priority_and_mask() -> None:
    image = _image()
    runtime = ParametricMAPLUTRuntime(image)
    controller = ConservativeFallbackController([_trusted(image)])
    action = controller.step(
        _input(
            runtime,
            5,
            syndrome_x="leakage",
            map_decision=None,
            expected_version=9,
            reported_crc="0" * 8,
            reported_sha="0" * 64,
            age=65,
            ood=255,
            observation_valid=False,
            input_crc_ok=False,
            deadline_ok=False,
        )
    )
    assert action.primary_reason == "observation_invalid"
    assert len(action.fault_flags) >= 8
    assert action.fault_mask == sum(FAULT_BITS[name] for name in action.fault_flags)
    assert all(name in action.reason_trace for name in action.fault_flags)
    assert action.status == FALLBACK_ACTIVE


def test_health_and_per_reason_counters_saturate_without_wrap() -> None:
    image = _image()
    runtime = ParametricMAPLUTRuntime(image)
    controller = ConservativeFallbackController([_trusted(image)])
    for cycle in range(5, 5 + 260):
        action = controller.step(_input(runtime, cycle, map_decision=None))
    index = FAULT_ORDER.index("map_decision_missing")
    assert action.fault_run == 255
    assert action.fault_cycle_count == 255
    assert action.per_flag_cycle_counts[index] == 255
    assert controller.state.fault_run == 255


def test_structural_failures_are_transactional() -> None:
    image = _image()
    runtime = ParametricMAPLUTRuntime(image)
    controller = ConservativeFallbackController([_trusted(image)])
    before = controller.state
    with pytest.raises(ValueError, match="score width"):
        controller.step(_input(runtime, 5, ood=256))
    assert controller.state == before and controller.history == ()
    with pytest.raises(ValueError, match="sequential"):
        controller.step(_input(runtime, 6))
    assert controller.state == before and controller.history == ()


def test_event_fsm_accepts_missing_map_only_for_fail_closed_health_event() -> None:
    image = _image()
    fsm = ExperimentalEventFSM()
    faulty = ExperimentalEventInput(
        5, "g", "g", 0, None, 0, valid=False
    )
    action = fsm.step(faulty)
    assert action.mode == FALLBACK
    assert action.map_logical_action == "I"
    assert action.map_image_sha256 == ""

    fsm = ExperimentalEventFSM()
    with pytest.raises(ValueError, match="requires"):
        fsm.step(ExperimentalEventInput(5, "g", "g", 0, None, 0))
    assert fsm.history == ()


def test_online_schema_and_controller_source_have_no_truth_fields() -> None:
    fields = set(ConservativeFallbackInput.__dataclass_fields__)
    forbidden = {"truth", "logical_truth", "hidden_state", "drift_state", "recovery_depth"}
    assert not fields & forbidden
    source = inspect.getsource(ConservativeFallbackController)
    assert "DriftState" not in source
    assert "SyndromeTruthStep" not in source
