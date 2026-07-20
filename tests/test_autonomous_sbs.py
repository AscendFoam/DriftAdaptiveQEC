from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cnn_fpga.benchmark.autonomous_sbs_wallclock_baseline import (
    WallClockBenchmarkConfig,
    nonselective_measurement_equivalence_audit,
    zero_noise_duration_invariance_audit,
)
from physics.autonomous_sbs import (
    AUTONOMOUS_TIMING,
    IdleMemoryConfig,
    IdleMemorySimulator,
    MEASUREMENT_TIMING,
    NonselectiveSBSConfig,
    NonselectiveSBSSimulator,
    ProtocolTiming,
    finite_horizon_area_lifetime,
    validate_timing_contract,
)

torch = pytest.importorskip("torch")


def run_small(mode: str, **overrides: object):
    values: dict[str, object] = {
        "mode": mode,
        "full_cycles": 2,
        "cutoff": 6,
        "device": "cpu",
    }
    values.update(overrides)
    return NonselectiveSBSSimulator(NonselectiveSBSConfig(**values)).run()


def run_idle(**overrides: object):
    values: dict[str, object] = {
        "full_cycles": 2,
        "cutoff": 6,
        "device": "cpu",
    }
    values.update(overrides)
    return IdleMemorySimulator(IdleMemoryConfig(**values)).run()


def test_literature_timing_contract_is_exact_and_not_board_measured() -> None:
    assert all(validate_timing_contract().values())
    assert MEASUREMENT_TIMING.half_cycle_duration_ns == 5000
    assert AUTONOMOUS_TIMING.half_cycle_duration_ns == 3500
    assert MEASUREMENT_TIMING.full_cycle_duration_ns == 10_000
    assert AUTONOMOUS_TIMING.full_cycle_duration_ns == 7_000
    assert not MEASUREMENT_TIMING.target_hardware_measured
    assert not AUTONOMOUS_TIMING.target_hardware_measured


def test_invalid_timing_profiles_fail_closed() -> None:
    with pytest.raises(ValueError):
        replace(AUTONOMOUS_TIMING, measurement_and_or_reset_ns=0)
    with pytest.raises(ValueError):
        replace(AUTONOMOUS_TIMING, measurement_events_per_half_cycle=2)
    with pytest.raises(ValueError):
        replace(AUTONOMOUS_TIMING, reset_events_per_half_cycle=0)
    with pytest.raises(ValueError):
        replace(AUTONOMOUS_TIMING, target_hardware_measured=True)


@pytest.mark.parametrize(
    "updates",
    [
        {"mode": "unknown"},
        {"full_cycles": 0},
        {"full_cycles": True},
        {"cutoff": 3},
        {"ancilla_t1_us": 1.0, "ancilla_t2_us": 3.0},
        {"device": "tpu"},
        {"real_dtype": "float16"},
    ],
)
def test_invalid_simulator_configs_fail_closed(updates: dict[str, object]) -> None:
    values: dict[str, object] = {"mode": "autonomous", "full_cycles": 2, "cutoff": 6}
    values.update(updates)
    with pytest.raises(ValueError):
        NonselectiveSBSConfig(**values)


@pytest.mark.parametrize(
    "updates",
    [
        {"full_cycles": 0},
        {"full_cycles": True},
        {"cutoff": 49},
        {"cycle_duration_us": 0.0},
        {"cavity_lifetime_us": float("inf")},
        {"ancilla_t1_us": 1.0, "ancilla_t2_us": 3.0},
        {"device": "tpu"},
        {"real_dtype": "float16"},
    ],
)
def test_invalid_idle_memory_configs_fail_closed(updates: dict[str, object]) -> None:
    values: dict[str, object] = {"full_cycles": 2, "cutoff": 6}
    values.update(updates)
    with pytest.raises(ValueError):
        IdleMemoryConfig(**values)


def test_measurement_feedback_is_exact_all_branch_expectation() -> None:
    audit = nonselective_measurement_equivalence_audit(cutoff=6)
    assert audit["passes"] is True
    assert audit["all_four_branches_positive"] is True
    assert audit["branch_probability_sum"] == pytest.approx(1.0, abs=2.0e-12)
    assert audit["maximum_density_difference"] <= 2.0e-12


def test_duration_difference_vanishes_without_dissipation() -> None:
    audit = zero_noise_duration_invariance_audit(cutoff=6)
    assert audit["passes"] is True
    assert audit["maximum_density_difference"] <= 2.0e-12


def test_protocols_have_distinct_time_grids_but_same_nominal_controls() -> None:
    measurement = run_small("measurement_feedback")
    autonomous = run_small("autonomous")
    assert measurement.time_us.tolist() == [0.0, 10.0, 20.0]
    assert autonomous.time_us.tolist() == [0.0, 7.0, 14.0]
    assert torch.equal(measurement.physical_controls, autonomous.physical_controls)
    assert measurement.physical_controls.shape == (15,)


def test_finite_dissipation_is_rerun_not_a_curve_rescaling() -> None:
    measurement = run_small("measurement_feedback")
    autonomous = run_small("autonomous")
    assert not np.array_equal(measurement.logical_z_signal, autonomous.logical_z_signal)
    assert not torch.equal(measurement.final_cavity_density, autonomous.final_cavity_density)


def test_event_accounting_matches_protocol_semantics() -> None:
    measurement = run_small("measurement_feedback")
    autonomous = run_small("autonomous")
    assert measurement.event_accounting["measurement_events"] == 4
    assert autonomous.event_accounting["measurement_events"] == 0
    assert measurement.event_accounting["reset_events"] == autonomous.event_accounting["reset_events"] == 4
    assert measurement.event_accounting["active_gate_applications"] == autonomous.event_accounting["active_gate_applications"] == 36
    assert autonomous.event_accounting["resets_per_100us"] / measurement.event_accounting["resets_per_100us"] == pytest.approx(10.0 / 7.0)
    assert autonomous.event_accounting["active_gates_per_100us"] / measurement.event_accounting["active_gates_per_100us"] == pytest.approx(10.0 / 7.0)


def test_no_correction_anchor_has_no_control_measurement_reset_or_frame_action() -> None:
    idle = run_idle()
    assert idle.time_us.tolist() == [0.0, 10.0, 20.0]
    for field in (
        "measurement_events",
        "reset_events",
        "active_gate_applications",
        "frame_updates",
        "outcome_dependent_parameter_updates",
    ):
        assert idle.event_accounting[field] == 0
    assert idle.event_accounting["total_physical_time_us"] == 20.0


def test_no_correction_channel_is_not_a_renamed_standard_sbs_curve() -> None:
    idle = run_idle()
    measurement = run_small("measurement_feedback")
    assert np.array_equal(idle.time_us, measurement.time_us)
    assert not np.array_equal(idle.logical_z_signal, measurement.logical_z_signal)
    assert not torch.equal(idle.final_cavity_density, measurement.final_cavity_density)


def test_no_correction_idle_channel_obeys_time_semigroup() -> None:
    one_step = run_idle(full_cycles=1, cycle_duration_us=10.0)
    two_steps = run_idle(full_cycles=2, cycle_duration_us=5.0)
    assert torch.max(
        torch.abs(one_step.final_cavity_density - two_steps.final_cavity_density)
    ).item() <= 2.0e-12


def test_metrics_are_recorded_at_every_full_cycle_and_density_is_healthy() -> None:
    for mode in ("measurement_feedback", "autonomous"):
        result = run_small(mode)
        assert result.fidelity.shape == result.code_survival.shape == (3,)
        assert result.logical_z_signal.shape == result.conditional_logical_z.shape == (3,)
        assert np.all(np.isfinite(result.fidelity))
        assert result.maximum_trace_error <= 2.0e-12
        assert result.maximum_hermiticity_error <= 2.0e-12
        assert result.minimum_final_eigenvalue >= -2.0e-12

    idle = run_idle()
    assert idle.fidelity.shape == idle.code_survival.shape == (3,)
    assert idle.logical_z_signal.shape == idle.conditional_logical_z.shape == (3,)
    assert np.all(np.isfinite(idle.fidelity))
    assert idle.maximum_trace_error <= 2.0e-12
    assert idle.maximum_hermiticity_error <= 2.0e-12
    assert idle.minimum_final_eigenvalue >= -2.0e-12


@pytest.mark.parametrize("lifetime", [3.0, 30.0, 300.0])
def test_area_lifetime_inverts_an_exponential_on_physical_time(lifetime: float) -> None:
    time_us = np.linspace(0.0, 700.0, 7001)
    curve = np.exp(-time_us / lifetime)
    result = finite_horizon_area_lifetime(time_us, curve)
    assert result["area_equivalent_lifetime_us"] == pytest.approx(lifetime, rel=2.0e-4)
    assert result["area_equivalent_lifetime_standard_10us_cycles"] == pytest.approx(lifetime / 10.0, rel=2.0e-4)


@pytest.mark.parametrize(
    "times,curve",
    [
        ([0.0, 1.0], [1.0, 0.5]),
        ([0.0, 2.0, 1.0], [1.0, 0.8, 0.7]),
        ([1.0, 2.0, 3.0], [1.0, 0.8, 0.7]),
        ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]),
        ([0.0, 1.0, 2.0], [1.0, -2.0, -2.0]),
    ],
)
def test_invalid_lifetime_inputs_fail_closed(times: list[float], curve: list[float]) -> None:
    with pytest.raises(ValueError):
        finite_horizon_area_lifetime(np.asarray(times), np.asarray(curve))


def test_common_horizon_requires_integer_cycles_for_both_protocols() -> None:
    assert WallClockBenchmarkConfig(common_horizon_us=700, device="cpu")
    with pytest.raises(ValueError, match="integer number"):
        WallClockBenchmarkConfig(common_horizon_us=701, device="cpu")


def test_result_serialization_keeps_event_and_timing_boundaries() -> None:
    payload = run_small("autonomous").to_dict()
    assert payload["config"]["timing"]["profile_id"] == AUTONOMOUS_TIMING.profile_id
    assert payload["config"]["timing"]["target_hardware_measured"] is False
    assert payload["event_accounting"]["measurement_events"] == 0
    assert "final_cavity_density_real" not in payload
