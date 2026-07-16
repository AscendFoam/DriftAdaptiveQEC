from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from cnn_fpga.decoder.parametric_map_lut import compile_parametric_map_lut
from cnn_fpga.runtime.experimental_event_fsm import (
    EVENT_MODES,
    FALLBACK,
    HOLD,
    NORMAL,
    RESET_REQUEST,
    X_RECOVERY,
    Z_RECOVERY,
    ExperimentalEventFSM,
    ExperimentalEventFSMConfig,
    ExperimentalEventInput,
)
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


def _code_for_action(runtime: ParametricMAPLUTRuntime, phase: int, flip: bool) -> int:
    version = runtime.image.active_bank_version
    for code in range(runtime.image.config.adc_levels):
        decision = runtime.decode_code(ParametricMAPLUTInput(0, code, phase, version))
        if decision.logical_flip is flip:
            return code
    raise AssertionError("registered image did not contain requested action")


def _event(
    runtime: ParametricMAPLUTRuntime,
    cycle: int,
    *,
    phase: int = 0,
    flip: bool = False,
    syndrome_x: str = "g",
    syndrome_z: str = "g",
    reset_ack: bool = False,
    valid: bool = True,
    crc_ok: bool = True,
    parameter_fresh: bool = True,
    deadline_ok: bool = True,
) -> ExperimentalEventInput:
    code = _code_for_action(runtime, phase, flip)
    source_cycle = cycle - runtime.image.config.pipeline_latency_cycles
    decision = runtime.decode_code(
        ParametricMAPLUTInput(
            source_cycle,
            code,
            phase,
            runtime.image.active_bank_version,
        )
    )
    return ExperimentalEventInput(
        cycle_index=cycle,
        syndrome_x=syndrome_x,
        syndrome_z=syndrome_z,
        quadrature_phase_bit=phase,
        map_decision=decision,
        active_bank_version=runtime.image.active_bank_version,
        reset_ack=reset_ack,
        valid=valid,
        crc_ok=crc_ok,
        parameter_fresh=parameter_fresh,
        deadline_ok=deadline_ok,
    )


def test_config_freezes_six_modes_saturating_width_and_frame_modulus() -> None:
    config = ExperimentalEventFSMConfig()
    assert EVENT_MODES == (
        "normal",
        "x_recovery",
        "z_recovery",
        "hold",
        "reset_request",
        "fallback",
    )
    assert config.counter_max == 7
    assert config.phase_modulus == 256
    assert config.logical_half_turn_code == 128
    with pytest.raises(ValueError, match="exceeds"):
        ExperimentalEventFSMConfig(counter_bits=2, reset_request_run=4)
    with pytest.raises(ValueError, match="at least"):
        ExperimentalEventFSMConfig(start_event_cycle=-1)
    with pytest.raises(ValueError, match="exactly one"):
        ExperimentalEventFSMConfig(event_action_latency_cycles=2)


def test_normal_map_x_action_atomically_updates_pauli_and_phase_frames() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    first = fsm.step(_event(runtime, 5, phase=0, flip=True))
    assert first.mode == NORMAL
    assert first.pauli_frame_delta_x is True
    assert first.pauli_frame_delta_z is False
    assert first.pauli_frame_x is True
    assert first.phase_frame_x_code == 128
    assert first.phase_frame_delta_x_code == 128
    assert first.correction_enable is True
    assert first.source_cycle == 0 and first.action_cycle == 6

    second = fsm.step(_event(runtime, 6, phase=0, flip=True))
    assert second.pauli_frame_x is False
    assert second.phase_frame_x_code == 0
    assert fsm.state.frame_update_count == 2


def test_z_map_action_updates_only_z_frame_and_i_action_is_noop() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    z = fsm.step(_event(runtime, 5, phase=1, flip=True))
    assert z.pauli_frame_z and not z.pauli_frame_x
    assert z.phase_frame_z_code == 128 and z.phase_frame_x_code == 0
    idle = fsm.step(_event(runtime, 6, phase=0, flip=False))
    assert idle.pauli_frame_z and not idle.pauli_frame_x
    assert idle.phase_frame_z_code == 128
    assert idle.map_logical_action == "I"


def test_e_run_threshold_enters_axis_recovery_and_single_e_is_not_recovery() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    first = fsm.step(_event(runtime, 5, syndrome_x="e"))
    second = fsm.step(_event(runtime, 6, syndrome_x="e"))
    assert first.mode == NORMAL
    assert second.mode == X_RECOVERY
    assert second.reason == "x_e_run_threshold"
    assert second.x_e_run == 2
    assert second.correction_enable is True

    fsm = ExperimentalEventFSM()
    fsm.step(_event(runtime, 5, syndrome_z="e"))
    z = fsm.step(_event(runtime, 6, syndrome_z="e"))
    assert z.mode == Z_RECOVERY
    assert z.reason == "z_e_run_threshold"


@pytest.mark.parametrize("phase,expected", [(0, X_RECOVERY), (1, Z_RECOVERY)])
def test_simultaneous_e_runs_use_current_phase_tie_break(phase: int, expected: str) -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    fsm.step(_event(runtime, 5, phase=phase, syndrome_x="e", syndrome_z="e"))
    action = fsm.step(
        _event(runtime, 6, phase=phase, syndrome_x="e", syndrome_z="e")
    )
    assert action.mode == expected
    assert action.reason == f"both_e_runs_phase_{'x' if phase == 0 else 'z'}_priority"


def test_three_bit_e_and_leakage_counters_saturate_without_wrap() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    for cycle in range(5, 17):
        action = fsm.step(_event(runtime, cycle, syndrome_x="e"))
    assert action.x_e_run == 7
    assert fsm.state.x_e_run == 7

    fsm = ExperimentalEventFSM()
    for cycle in range(5, 17):
        action = fsm.step(
            _event(
                runtime,
                cycle,
                syndrome_x="leakage",
                syndrome_z="leakage",
            )
        )
    assert action.leakage_run == 7
    assert action.mode == RESET_REQUEST
    assert action.reset_wait_run == 7


def test_leakage_hold_reset_request_sticky_ack_and_clean_hysteresis() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    hold = fsm.step(_event(runtime, 5, syndrome_x="leakage", flip=True))
    request = fsm.step(_event(runtime, 6, syndrome_z="leakage", flip=True))
    sticky = fsm.step(_event(runtime, 7, flip=True))
    ack = fsm.step(_event(runtime, 8, reset_ack=True, flip=True))
    normal = fsm.step(_event(runtime, 9, flip=False))
    assert [item.mode for item in (hold, request, sticky, ack, normal)] == [
        HOLD,
        RESET_REQUEST,
        RESET_REQUEST,
        HOLD,
        NORMAL,
    ]
    assert request.reset_request and sticky.reset_request
    assert sticky.reason == "reset_request_sticky_until_ack"
    assert ack.reason == "reset_acknowledged_post_reset_hold"
    assert normal.reason == "no_event_threshold"
    assert fsm.state.reset_request_count == 2


def test_safe_modes_inhibit_map_action_and_preserve_both_frames() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    before = fsm.state
    hold = fsm.step(_event(runtime, 5, syndrome_x="leakage", flip=True))
    request = fsm.step(_event(runtime, 6, syndrome_x="leakage", flip=True))
    assert hold.map_action_inhibited and request.map_action_inhibited
    assert not hold.correction_enable and not request.correction_enable
    assert hold.pauli_frame_x == before.pauli_frame_x
    assert request.pauli_frame_x == before.pauli_frame_x
    assert fsm.state.frame_update_count == 0


@pytest.mark.parametrize(
    "flag",
    ["valid", "crc_ok", "parameter_fresh", "deadline_ok"],
)
def test_each_health_fault_enters_fallback_and_two_good_cycles_clear(flag: str) -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    kwargs = {flag: False, "syndrome_x": "e", "flip": True}
    failed = fsm.step(_event(runtime, 5, **kwargs))
    first_good = fsm.step(_event(runtime, 6))
    second_good = fsm.step(_event(runtime, 7))
    assert failed.mode == FALLBACK
    assert failed.map_action_inhibited
    assert failed.x_e_run == 0 and failed.health_good_run == 0
    assert first_good.mode == FALLBACK
    assert first_good.reason == "fallback_clear_hysteresis"
    assert second_good.mode == NORMAL


def test_unexpected_reset_ack_fails_closed_without_frame_update() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    action = fsm.step(_event(runtime, 5, reset_ack=True, flip=True))
    assert action.mode == FALLBACK
    assert action.reason == "unexpected_reset_ack"
    assert action.map_action_inhibited
    assert fsm.state.frame_update_count == 0


def test_bank_version_may_advance_but_never_roll_back() -> None:
    runtime0 = ParametricMAPLUTRuntime(_image(0))
    runtime1 = ParametricMAPLUTRuntime(_image(1))
    fsm = ExperimentalEventFSM()
    first = fsm.step(_event(runtime0, 5))
    second = fsm.step(_event(runtime1, 6))
    assert first.active_bank_version == 0
    assert second.active_bank_version == 1
    before = fsm.state
    with pytest.raises(ValueError, match="rollback"):
        fsm.step(_event(runtime0, 7))
    assert fsm.state == before


def test_alignment_phase_version_and_action_mismatch_are_transactional() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    valid = _event(runtime, 5, phase=0, flip=True)
    cases = [
        replace(valid, cycle_index=6),
        replace(valid, quadrature_phase_bit=1),
        replace(valid, active_bank_version=1),
        replace(valid, map_decision=replace(valid.map_decision, logical_action="I")),
        replace(valid, map_decision=replace(valid.map_decision, logical_flip=False)),
    ]
    for case in cases:
        before = fsm.state
        with pytest.raises(ValueError):
            fsm.step(case)
        assert fsm.state == before
        assert fsm.history == ()


def test_reset_restores_state_history_and_start_cycle() -> None:
    runtime = ParametricMAPLUTRuntime(_image())
    fsm = ExperimentalEventFSM()
    fsm.step(_event(runtime, 5, flip=True))
    state = fsm.reset()
    assert state.cycle_index == 4
    assert state.mode == NORMAL
    assert state.pauli_frame_x is False
    assert fsm.history == ()
    assert fsm.step(_event(runtime, 5)).action_cycle == 6


def test_online_input_and_step_signature_have_no_truth_or_hidden_state() -> None:
    fields = set(ExperimentalEventInput.__dataclass_fields__)
    forbidden = {"truth", "logical_truth", "hidden_state", "drift_state", "recovery_depth"}
    assert not fields & forbidden
    assert set(inspect.signature(ExperimentalEventFSM.step).parameters) == {"self", "event"}
    source = inspect.getsource(ExperimentalEventFSM)
    assert "SyndromeTruthStep" not in source
    assert "DriftState" not in source
