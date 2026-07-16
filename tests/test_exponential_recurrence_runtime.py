from __future__ import annotations

from dataclasses import fields

import pytest

from cnn_fpga.runtime.exponential_recurrence import (
    ExponentialEventControllerConfig,
    ExponentialRecurrenceEventController,
    ExponentialSaturationKernel,
)
from cnn_fpga.runtime.run_length_fsm import (
    FALLBACK,
    LEAKAGE_HOLD,
    NORMAL,
    X_RECOVERY,
    RunLengthFSMInput,
)


def _event(cycle: int, *, x: str = "g", z: str = "g", phase: int = 0, **health: bool) -> RunLengthFSMInput:
    return RunLengthFSMInput(
        cycle_index=cycle,
        residual=(0.4, -0.2),
        syndrome_x=x,
        syndrome_z=z,
        quadrature_phase_bit=phase,
        **health,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decay_g": 0.0},
        {"decay_e": 1.0},
        {"saturation_g": -0.1},
        {"recovery_exit": 0.7, "recovery_enter": 0.6},
        {"leakage_exit": 0.7, "leakage_enter": 0.6},
        {"state_fraction_bits": True},
        {"state_total_bits": 13, "state_fraction_bits": 12},
    ],
)
def test_config_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExponentialEventControllerConfig(**kwargs)  # type: ignore[arg-type]


def test_kernel_repeated_branch_matches_closed_form() -> None:
    config = ExponentialEventControllerConfig(initial_state=0.2, decay_e=0.6)
    kernel = ExponentialSaturationKernel(config, "float64")
    for _ in range(4):
        kernel.step("e", "g")
    expected = config.decay_e**4 * config.initial_state + (1.0 - config.decay_e**4) * config.saturation_e
    assert kernel.states[0] == pytest.approx(expected, abs=2.0e-15)


def test_switching_branch_uses_previous_state() -> None:
    config = ExponentialEventControllerConfig(decay_g=0.7, decay_e=0.4)
    kernel = ExponentialSaturationKernel(config, "float64")
    kernel.step("e", "g")
    after_e = 1.0 - config.decay_e
    kernel.step("g", "g")
    assert kernel.states[0] == pytest.approx(config.decay_g * after_e)


def test_float_and_fixed_kernels_track_long_mixed_trace() -> None:
    config = ExponentialEventControllerConfig()
    floating = ExponentialSaturationKernel(config, "float64")
    fixed = ExponentialSaturationKernel(config, "fixed_point")
    pattern = (("g", "g"), ("e", "g"), ("e", "e"), ("leakage", "g"), ("g", "e"))
    maximum = 0.0
    for index in range(500):
        float_state = floating.step(*pattern[index % len(pattern)])
        fixed_state = fixed.step(*pattern[index % len(pattern)])
        maximum = max(maximum, *(abs(a - b) for a, b in zip(float_state, fixed_state, strict=True)))
    assert maximum < 7.5e-4


def test_controller_is_causal_hysteretic_and_uses_atomic_bank() -> None:
    config = ExponentialEventControllerConfig(
        decay_e=0.4,
        decay_g=0.5,
        recovery_enter=0.55,
        recovery_exit=0.20,
    )
    controller = ExponentialRecurrenceEventController(config)
    modes = [
        controller.step(_event(0, x="e")).mode,
        controller.step(_event(1, x="g")).mode,
        controller.step(_event(2, x="g")).mode,
    ]
    assert modes == [X_RECOVERY, X_RECOVERY, NORMAL]
    assert controller.bank_writes == 2
    assert controller.param_bank.active_version == 2


def test_leakage_and_health_fault_have_priority() -> None:
    config = ExponentialEventControllerConfig(decay_leakage=0.2, leakage_enter=0.5)
    controller = ExponentialRecurrenceEventController(config)
    leakage = controller.step(_event(0, x="leakage"))
    fault = controller.step(_event(1, x="leakage", deadline_ok=False))
    assert leakage.mode == LEAKAGE_HOLD
    assert fault.mode == FALLBACK
    assert fault.reason == "health_fault"


def test_fixed_point_controller_has_same_modes_on_margin_trace() -> None:
    config = ExponentialEventControllerConfig()
    floating = ExponentialRecurrenceEventController(config, arithmetic="float64")
    fixed = ExponentialRecurrenceEventController(config, arithmetic="fixed_point")
    pattern = (("g", "g"), ("e", "g"), ("e", "e"), ("leakage", "g"), ("g", "g"))
    for cycle in range(300):
        x, z = pattern[cycle % len(pattern)]
        first = floating.step(_event(cycle, x=x, z=z, phase=cycle & 1))
        second = fixed.step(_event(cycle, x=x, z=z, phase=cycle & 1))
        assert first.mode == second.mode


def test_input_schema_contains_no_truth_or_future_fields() -> None:
    names = {field.name for field in fields(RunLengthFSMInput)}
    assert not any(token in name for name in names for token in ("truth", "hidden", "future", "regime"))
    controller = ExponentialRecurrenceEventController()
    with pytest.raises(ValueError, match="sequential"):
        controller.step(_event(1))
    assert controller.mode == NORMAL

