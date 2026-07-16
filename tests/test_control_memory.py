from __future__ import annotations

import math

import numpy as np
import pytest

from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from physics.constants import LATTICE_CONST
from physics.control_memory import (
    MODEL_SCOPE,
    ControlDecision,
    ControlMemoryConfig,
    ControlMemoryState,
    MultiRoundControlMemory,
)
from physics.drift_processes import DriftState
from physics.sbs_error_space import PauliFrame, SBS_PROTOCOL_ID
from physics.sbs_observation_reset import PairedSyndrome
from physics.syndrome_stream import (
    ObservedSyndromeStep,
    SyndromeStreamConfig,
    generate_syndrome_stream,
)


def _observation(
    cycle: int,
    *,
    residual: tuple[float, float] = (0.0, 0.0),
    analog: tuple[float, float] | None = None,
    x: str = "g",
    z: str = "g",
    x_e_run: int = 0,
    z_e_run: int = 0,
    leakage_run: int = 0,
    valid: bool = True,
) -> ObservedSyndromeStep:
    return ObservedSyndromeStep(
        cycle_index=cycle,
        drift_step=cycle,
        time=float(cycle),
        analog_syndrome=residual if analog is None else analog,
        residual_syndrome=residual,
        syndrome=PairedSyndrome(x=x, z=z),
        quadrature_phases_rad=(0.0, math.pi / 2.0),
        x_e_run=x_e_run,
        z_e_run=z_e_run,
        leakage_run=leakage_run,
        valid=valid,
    )


def _decision(**updates: object) -> ControlDecision:
    values: dict[str, object] = {"parameter_bank_version": 0}
    values.update(updates)
    return ControlDecision(**values)


def test_config_and_constructor_fail_closed() -> None:
    invalid = [
        lambda: ControlMemoryConfig(lattice=0.0),
        lambda: ControlMemoryConfig(counter_max=0),
        lambda: ControlMemoryConfig(counter_max=True),
        lambda: ControlMemoryConfig(start_cycle_index=-1),
        lambda: ControlMemoryConfig(initial_parameter_bank_version=-1),
        lambda: ControlMemoryConfig(residual_consistency_atol=-1.0),
        lambda: ControlMemoryConfig(strict_observed_run_validation=1),
        lambda: ControlMemoryConfig(model_scope="device"),
        lambda: MultiRoundControlMemory(config=object()),  # type: ignore[arg-type]
        lambda: MultiRoundControlMemory(initial_state=object()),  # type: ignore[arg-type]
    ]
    for call in invalid:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_decision_and_state_fail_closed_on_malformed_fields() -> None:
    invalid = [
        lambda: ControlDecision(applied_correction=(0.0,)),
        lambda: ControlDecision(applied_correction=(float("nan"), 0.0)),
        lambda: ControlDecision(confidence=(-0.1, 0.5)),
        lambda: ControlDecision(confidence=(0.5, 1.1)),
        lambda: ControlDecision(pauli_frame_delta=(0, 0)),  # type: ignore[arg-type]
        lambda: ControlDecision(parameter_bank_version=-1),
        lambda: ControlDecision(deadline_missed=1),
        lambda: ControlDecision(control_mode=""),
        lambda: ControlMemoryState(cycle_index=-2),
        lambda: ControlMemoryState(confidence=(1.1, 0.0)),
        lambda: ControlMemoryState(pauli_frame=(0, 0)),  # type: ignore[arg-type]
        lambda: ControlMemoryState(deadline_missed=1),
    ]
    for call in invalid:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_initial_state_contains_every_required_memory_field() -> None:
    state = MultiRoundControlMemory().state
    assert state.cycle_index == -1
    assert state.cycle_count == 0
    assert state.accumulated_residual_shift == (0.0, 0.0)
    assert state.previous_correction == (0.0, 0.0)
    assert state.confidence == (0.0, 0.0)
    assert state.pauli_frame == PauliFrame()
    assert state.phase_frame_rad == (0.0, 0.0)
    assert (state.x_e_run, state.z_e_run, state.leakage_run) == (0, 0, 0)
    assert state.parameter_bank_version == 0
    assert state.deadline_missed is False


def test_residual_nearest_lift_preserves_continuity_across_wrap_boundary() -> None:
    memory = MultiRoundControlMemory()
    memory.update(_observation(0, residual=(0.49 * LATTICE_CONST, 0.0)), _decision())
    update = memory.update(
        _observation(1, residual=(-0.49 * LATTICE_CONST, 0.0)),
        _decision(),
    )
    assert update.residual_alias_indices == (1, 0)
    assert update.lifted_observation_shift[0] == pytest.approx(0.51 * LATTICE_CONST)
    assert memory.state.accumulated_residual_shift[0] == pytest.approx(
        0.51 * LATTICE_CONST
    )


def test_applied_correction_updates_post_action_shift_and_previous_correction() -> None:
    memory = MultiRoundControlMemory()
    correction = (0.2, -0.1)
    memory.update(
        _observation(0, residual=(0.3, -0.4)),
        _decision(applied_correction=correction, confidence=(0.9, 0.7)),
    )
    assert memory.state.accumulated_residual_shift == pytest.approx((0.1, -0.3))
    assert memory.state.previous_correction == correction
    assert memory.state.confidence == (0.9, 0.7)
    assert memory.state.minimum_confidence == pytest.approx(0.7)


def test_pauli_frame_composes_by_xor_and_phase_frame_wraps_modulo_two_pi() -> None:
    memory = MultiRoundControlMemory()
    memory.update(
        _observation(0),
        _decision(
            pauli_frame_delta=PauliFrame(x=1),
            phase_frame_delta_rad=(1.5 * math.pi, -1.5 * math.pi),
        ),
    )
    assert memory.state.pauli_frame == PauliFrame(x=1, z=0)
    assert memory.state.phase_frame_rad == pytest.approx((-0.5 * math.pi, 0.5 * math.pi))
    memory.update(
        _observation(1),
        _decision(
            pauli_frame_delta=PauliFrame(x=1, z=1),
            phase_frame_delta_rad=(0.5 * math.pi, 0.5 * math.pi),
        ),
    )
    assert memory.state.pauli_frame == PauliFrame(x=0, z=1)
    assert memory.state.phase_frame_rad == pytest.approx((0.0, -math.pi))


def test_x_and_z_e_runs_are_recomputed_causally_and_separately() -> None:
    memory = MultiRoundControlMemory()
    observations = [
        _observation(0, x="e", x_e_run=1),
        _observation(1, x="e", x_e_run=2),
        _observation(2, z="e", z_e_run=1),
        _observation(3, z="e", z_e_run=2),
    ]
    trajectory = memory.run(observations, [_decision()] * 4)
    assert [(u.current_state.x_e_run, u.current_state.z_e_run) for u in trajectory.updates] == [
        (1, 0),
        (2, 0),
        (0, 1),
        (0, 2),
    ]


def test_e_and_deadline_counters_saturate_without_integer_growth() -> None:
    memory = MultiRoundControlMemory(ControlMemoryConfig(counter_max=2))
    observations = [
        _observation(index, x="e", x_e_run=index + 1) for index in range(4)
    ]
    decisions = [_decision(deadline_missed=True) for _ in observations]
    trajectory = memory.run(observations, decisions)
    assert [u.current_state.x_e_run for u in trajectory.updates] == [1, 2, 2, 2]
    assert [u.current_state.deadline_miss_run for u in trajectory.updates] == [1, 2, 2, 2]
    assert memory.state.deadline_miss_count == 2


def test_leakage_run_counts_any_constituent_and_resets_on_clean_cycle() -> None:
    memory = MultiRoundControlMemory()
    observations = [
        _observation(0, x="leakage", leakage_run=1),
        _observation(1, z="leakage", leakage_run=2),
        _observation(2),
    ]
    trajectory = memory.run(observations, [_decision()] * 3)
    assert [u.current_state.leakage_run for u in trajectory.updates] == [1, 2, 0]


def test_observed_counter_mismatch_is_rejected_transactionally() -> None:
    memory = MultiRoundControlMemory()
    before = memory.state
    with pytest.raises(ValueError, match="not causal"):
        memory.update(_observation(0, x="e", x_e_run=0), _decision())
    assert memory.state == before
    assert memory.history == ()


def test_run_validation_can_be_disabled_but_memory_still_recomputes() -> None:
    memory = MultiRoundControlMemory(
        ControlMemoryConfig(strict_observed_run_validation=False)
    )
    memory.update(_observation(0, x="e", x_e_run=99), _decision())
    assert memory.state.x_e_run == 1


def test_analog_residual_inconsistency_and_invalid_observation_fail_closed() -> None:
    memory = MultiRoundControlMemory()
    with pytest.raises(ValueError, match="inconsistent"):
        memory.update(
            _observation(0, analog=(0.4, 0.0), residual=(0.3, 0.0)),
            _decision(),
        )
    with pytest.raises(ValueError, match="invalid observation"):
        memory.update(_observation(0, valid=False), _decision())
    assert memory.state.cycle_index == -1


def test_invalid_observation_can_be_explicitly_allowed() -> None:
    memory = MultiRoundControlMemory(ControlMemoryConfig(require_valid_observation=False))
    memory.update(_observation(0, valid=False), _decision())
    assert memory.state.cycle_index == 0


def test_cycle_replay_skip_and_wrong_input_types_are_rejected() -> None:
    memory = MultiRoundControlMemory()
    memory.update(_observation(0), _decision())
    for bad in (_observation(0), _observation(2)):
        with pytest.raises(ValueError, match="cycle_index"):
            memory.update(bad, _decision())
    with pytest.raises(TypeError, match="observation"):
        memory.update(object(), _decision())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="decision"):
        memory.update(_observation(1), object())  # type: ignore[arg-type]
    assert memory.state.cycle_index == 0


def test_parameter_bank_version_is_monotonic_and_change_is_explicit() -> None:
    memory = MultiRoundControlMemory(ControlMemoryConfig(initial_parameter_bank_version=2))
    first = memory.update(_observation(0), _decision(parameter_bank_version=2))
    second = memory.update(_observation(1), _decision(parameter_bank_version=5))
    assert first.parameter_bank_changed is False
    assert second.parameter_bank_changed is True
    assert memory.state.parameter_bank_version == 5
    before = memory.state
    with pytest.raises(ValueError, match="rollback"):
        memory.update(_observation(2), _decision(parameter_bank_version=4))
    assert memory.state == before


def test_real_param_bank_commit_version_can_drive_memory_without_tearing() -> None:
    bank = ParamBank()
    memory = MultiRoundControlMemory()
    memory.update(_observation(0), _decision(parameter_bank_version=bank.active_version))
    bank.stage_update(
        DecoderRuntimeParams(K=np.eye(2) * 0.8, b=np.array([0.1, -0.1])),
        commit_epoch=1,
    )
    result = bank.commit_if_ready(1)
    assert result is not None and result.version == 1
    update = memory.update(
        _observation(1),
        _decision(parameter_bank_version=bank.active_version),
    )
    assert update.parameter_bank_changed is True
    assert update.current_state.parameter_bank_version == bank.active_version == 1


def test_deadline_flag_run_and_total_are_separate_and_recover() -> None:
    memory = MultiRoundControlMemory()
    decisions = [
        _decision(deadline_missed=True),
        _decision(deadline_missed=True),
        _decision(deadline_missed=False),
        _decision(deadline_missed=True),
    ]
    trajectory = memory.run([_observation(i) for i in range(4)], decisions)
    assert [u.current_state.deadline_missed for u in trajectory.updates] == [
        True,
        True,
        False,
        True,
    ]
    assert [u.current_state.deadline_miss_run for u in trajectory.updates] == [1, 2, 0, 1]
    assert [u.current_state.deadline_miss_count for u in trajectory.updates] == [1, 2, 2, 3]


def test_deadline_miss_does_not_erase_actual_fallback_correction() -> None:
    memory = MultiRoundControlMemory()
    memory.update(
        _observation(0, residual=(0.3, 0.0)),
        _decision(
            applied_correction=(0.1, 0.0),
            deadline_missed=True,
            control_mode="local_safe_fallback",
        ),
    )
    assert memory.state.deadline_missed is True
    assert memory.state.previous_correction == (0.1, 0.0)
    assert memory.state.accumulated_residual_shift == pytest.approx((0.2, 0.0))
    assert memory.history[0].decision.control_mode == "local_safe_fallback"


def test_run_length_mismatch_and_nonsequence_inputs_fail_closed() -> None:
    memory = MultiRoundControlMemory()
    with pytest.raises(ValueError, match="equal length"):
        memory.run([_observation(0)], [])
    with pytest.raises(TypeError, match="sequences"):
        memory.run("bad", "bad")  # type: ignore[arg-type]
    assert memory.history == ()


def test_reset_restores_initial_state_and_clears_history() -> None:
    initial = ControlMemoryState(
        cycle_index=4,
        cycle_count=5,
        accumulated_residual_shift=(0.2, -0.1),
        parameter_bank_version=3,
    )
    memory = MultiRoundControlMemory(initial_state=initial)
    memory.update(_observation(5, residual=(0.21, -0.09)), _decision(parameter_bank_version=3))
    assert len(memory.history) == 1
    restored = memory.reset()
    assert restored == initial
    assert memory.history == ()


def test_generated_syndrome_stream_integrates_without_hidden_truth_input() -> None:
    states = [
        DriftState(
            mu_q=0.05 * index,
            mu_p=-0.03 * index,
            sigma_q=0.02,
            sigma_p=0.03,
            regime="private-regime",
        )
        for index in range(8)
    ]
    stream = generate_syndrome_stream(
        states,
        config=SyndromeStreamConfig(
            measurement_sigma=(0.0, 0.0),
            loss_environment_variance=0.0,
            depth_probability_scale=0.0,
            base_leakage_probability=0.0,
            loss_leakage_scale=0.0,
            burst_leakage_bonus=0.0,
            readout_fidelity_g=1.0,
            readout_fidelity_e=1.0,
            seed=81,
        ),
    )
    observations = [step.observed for step in stream.steps]
    decisions = [
        _decision(
            applied_correction=tuple(np.asarray(obs.residual_syndrome)),
            confidence=(0.8, 0.7),
            parameter_bank_version=0 if index < 4 else 1,
        )
        for index, obs in enumerate(observations)
    ]
    trajectory = MultiRoundControlMemory().run(observations, decisions)
    assert trajectory.protocol_id == SBS_PROTOCOL_ID
    assert trajectory.model_scope == MODEL_SCOPE
    assert trajectory.final_state.cycle_count == 8
    assert trajectory.final_state.parameter_bank_version == 1
    assert trajectory.final_state.accumulated_residual_shift == pytest.approx((0.0, 0.0))


def test_full_stream_step_is_rejected_to_prevent_truth_lane_leakage() -> None:
    stream = generate_syndrome_stream(
        [DriftState(sigma_q=0.01, sigma_p=0.01)],
        config=SyndromeStreamConfig(seed=2),
    )
    with pytest.raises(TypeError, match="ObservedSyndromeStep"):
        MultiRoundControlMemory().update(stream.steps[0], _decision())  # type: ignore[arg-type]


def test_deployable_memory_records_have_no_hidden_truth_or_oracle_keys() -> None:
    trajectory = MultiRoundControlMemory().run([_observation(0)], [_decision()])
    record = trajectory.deployable_records()[0]
    required = {
        "accumulated_residual_q",
        "previous_correction_q",
        "confidence_q",
        "pauli_frame_x",
        "phase_frame_x_rad",
        "x_e_run",
        "leakage_run",
        "parameter_bank_version",
        "deadline_missed",
    }
    assert required <= set(record)
    forbidden = ("truth", "hidden", "regime", "outlier", "oracle", "logical_label")
    assert not any(fragment in key for key in record for fragment in forbidden)


def test_two_fresh_memories_replay_identically_and_run_is_appendable() -> None:
    observations = [_observation(index) for index in range(5)]
    decisions = [
        _decision(
            applied_correction=(0.01 * index, -0.02 * index),
            confidence=(0.1 * index, 1.0 - 0.1 * index),
        )
        for index in range(5)
    ]
    first = MultiRoundControlMemory().run(observations, decisions)
    second_memory = MultiRoundControlMemory()
    prefix = second_memory.run(observations[:2], decisions[:2])
    suffix = second_memory.run(observations[2:], decisions[2:])
    assert prefix.final_state == first.updates[1].current_state
    assert suffix.final_state == first.final_state
    assert tuple(second_memory.history) == first.updates


def test_state_phase_boundary_is_canonicalized_to_half_open_interval() -> None:
    state = ControlMemoryState(phase_frame_rad=(math.pi, -math.pi))
    assert state.phase_frame_rad == (-math.pi, -math.pi)
    assert all(-math.pi <= value < math.pi for value in state.phase_frame_rad)


def test_public_physics_exports_resolve() -> None:
    from physics import ControlDecision as PublicDecision
    from physics import MultiRoundControlMemory as PublicMemory

    assert PublicDecision is ControlDecision
    assert PublicMemory is MultiRoundControlMemory
