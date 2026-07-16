from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import physics.sbs_cycle_state_machine as cycle_module
from physics.sbs_cycle_state_machine import (
    TIMING_EVIDENCE_SCOPE,
    CyclePhaseSpec,
    SBSConstituentControl,
    SBSCycleStateMachine,
    SBSCycleTimingProfile,
    sivak_table_s3_profile,
)
from physics.sbs_error_space import PauliFrame
from physics.sbs_observation_reset import make_persistent_leakage_model


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_PHASES = (
    ("enter_cycle", 24),
    ("enter_sbs", 24),
    ("sbs_layer_1", 502),
    ("sbs_layer_2", 708),
    ("sbs_layer_3", 262),
    ("sbs_layer_4", 76),
    ("exit_sbs", 24),
    ("enter_reset", 24),
    ("roundtrip_delay", 300),
    ("readout_acquisition", 1400),
    ("signal_processing", 332),
    ("syndrome_distribution", 100),
    ("branch_and_feedback", 200),
    ("exit_reset", 24),
    ("mixer_matrix_calculation", 400),
    ("mixer_update", 48),
    ("idle", 452),
    ("exit_cycle", 24),
)


def _machine() -> SBSCycleStateMachine:
    return SBSCycleStateMachine(
        virtual_rotation_key_by_observation={
            "g": "theta_g_calibration_slot",
            "e": "theta_e_calibration_slot",
            "leakage": "theta_leakage_assumption_slot",
        },
        virtual_rotation_provenance="unit-test explicit mapping; not device calibrated",
    )


def _observed_cycle(label: str = "K_ge"):
    readout = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    model = make_persistent_leakage_model(
        readout_confusion=readout,
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=1.0,
        higher_reset_success=0.0,
        counter_max=31,
        readout_provenance="unit-test identity readout",
        parameter_provenance="unit-test explicit assumptions",
    )
    return model.step(label, seed=1).observed


def test_table_s3_profile_preserves_exact_phase_order_and_durations() -> None:
    profile = sivak_table_s3_profile()
    assert tuple((phase.phase_id, phase.duration_ns) for phase in profile.phases) == EXPECTED_PHASES
    assert len(profile.phases) == 18
    assert profile.total_duration_ns == 4924
    assert profile.evidence_scope == TIMING_EVIDENCE_SCOPE


def test_scope_specific_sums_preserve_prose_table_discrepancies() -> None:
    profile = sivak_table_s3_profile()
    assert profile.sbs_layer_sum_ns == 1548
    assert profile.prose_sbs_ns == 1546
    assert profile.sbs_layer_sum_ns - profile.prose_sbs_ns == 2
    assert profile.table_sbs_block_ns == 1596
    assert profile.table_sbs_block_ns - profile.sbs_layer_sum_ns == 48
    assert profile.prose_reset_ns == 2332
    assert profile.table_reset_block_ns == 2380
    assert profile.table_reset_block_ns - profile.prose_reset_ns == 48
    assert profile.group_duration_ns("virtual_rotation") == 448
    assert profile.group_duration_ns("idle") == 452
    assert profile.group_duration_ns("cycle_overhead") == 48


def test_timing_profile_anchor_matches_local_primary_source() -> None:
    profile = sivak_table_s3_profile()
    source = ROOT / profile.source_path
    assert source.is_file()
    line = source.read_text(encoding="utf-8").splitlines()[profile.source_line - 1]
    assert profile.source_fragment in line


def test_constituent_runtime_advances_one_legal_phase_at_a_time() -> None:
    machine = _machine()
    control = SBSConstituentControl(
        quadrature="X",
        observed_class="e",
        reset_action="conditional_e_to_g_reset",
        virtual_rotation_key="theta_e_calibration_slot",
    )
    runtime = machine.start_constituent(control, start_ns=100, input_frame=PauliFrame())
    assert runtime.next_phase_id == "enter_cycle"
    with pytest.raises(RuntimeError, match="must complete"):
        _ = runtime.trace

    events = []
    while not runtime.is_complete:
        expected = EXPECTED_PHASES[len(events)][0]
        assert runtime.next_phase_id == expected
        events.append(runtime.advance())
    assert runtime.next_phase_id is None
    with pytest.raises(RuntimeError, match="already complete"):
        runtime.advance()
    trace = runtime.trace
    assert tuple(event.phase_id for event in events) == tuple(name for name, _ in EXPECTED_PHASES)
    assert trace.start_ns == 100
    assert trace.end_ns == 5024
    assert trace.total_duration_ns == 4924


def test_constituent_timeline_is_contiguous_without_overlap_or_gap() -> None:
    control = SBSConstituentControl(
        quadrature="Z",
        observed_class="g",
        reset_action="no_reset_pulse",
        virtual_rotation_key="theta_g_calibration_slot",
    )
    trace = _machine().run_constituent(control, start_ns=37)
    assert trace.events[0].start_ns == 37
    assert trace.events[-1].end_ns == 37 + 4924
    for previous, current in zip(trace.events, trace.events[1:]):
        assert previous.end_ns == current.start_ns
    assert sum(event.duration_ns for event in trace.events) == trace.total_duration_ns
    assert all(event.end_ns - event.start_ns == event.duration_ns for event in trace.events)


def test_branch_and_virtual_rotation_metadata_use_observed_control_only() -> None:
    control = SBSConstituentControl(
        quadrature="X",
        observed_class="leakage",
        reset_action="conditional_f_or_higher_reset_attempt",
        virtual_rotation_key="theta_leakage_assumption_slot",
    )
    trace = _machine().run_constituent(control)
    by_phase = {event.phase_id: event for event in trace.events}
    assert by_phase["signal_processing"].metadata["classified_observation"] == "leakage"
    assert by_phase["syndrome_distribution"].metadata["distributed_observation"] == "leakage"
    assert by_phase["branch_and_feedback"].metadata["reset_action"] == control.reset_action
    mixer = by_phase["mixer_matrix_calculation"].metadata
    assert mixer["virtual_rotation_key"] == "theta_leakage_assumption_slot"
    assert mixer["sbs_quadrature_switch_rad"] == pytest.approx(np.pi / 2.0)
    assert "not device calibrated" in mixer["virtual_rotation_provenance"]

    forbidden = ("hidden", "truth", "carry", "ideal_kraus")
    for event in trace.events:
        keys = " ".join(event.metadata).lower()
        assert not any(token in keys for token in forbidden)


def test_full_xz_cycle_composes_two_4924_ns_constituents_and_frame_updates() -> None:
    observed = _observed_cycle("K_ge")  # chronological X=e, Z=g
    trace = _machine().run_full_xz_cycle(observed, start_ns=200, input_frame=PauliFrame())
    assert trace.x_constituent.control.quadrature == "X"
    assert trace.x_constituent.control.observed_class == "e"
    assert trace.z_constituent.control.quadrature == "Z"
    assert trace.z_constituent.control.observed_class == "g"
    assert trace.x_constituent.start_ns == 200
    assert trace.x_constituent.end_ns == 5124
    assert trace.z_constituent.start_ns == 5124
    assert trace.z_constituent.end_ns == 10048
    assert trace.total_duration_ns == 9848
    assert trace.end_ns - trace.start_ns == 9848
    assert trace.x_constituent.output_frame == PauliFrame(x=1, z=0)
    assert trace.output_frame == PauliFrame(x=1, z=1)


def test_two_full_cycles_are_contiguous_and_pauli_frame_returns() -> None:
    machine = _machine()
    observed = _observed_cycle("K_ee")
    first = machine.run_full_xz_cycle(observed, start_ns=0, input_frame=PauliFrame())
    second = machine.run_full_xz_cycle(
        observed,
        start_ns=first.end_ns,
        input_frame=first.output_frame,
    )
    assert first.end_ns == second.start_ns == 9848
    assert second.end_ns == 19696
    assert second.output_frame == PauliFrame()


def test_all_timing_outputs_are_literature_reference_not_target_measurement() -> None:
    trace = _machine().run_full_xz_cycle(_observed_cycle())
    assert trace.timing_scope == TIMING_EVIDENCE_SCOPE
    assert trace.target_hardware_measured is False
    for constituent in (trace.x_constituent, trace.z_constituent):
        assert constituent.timing_scope == TIMING_EVIDENCE_SCOPE
        assert constituent.target_hardware_measured is False
        assert all(event.timing_scope == TIMING_EVIDENCE_SCOPE for event in constituent.events)


def test_profile_matches_paper_parameter_registry_without_collapsing_cycle_scope() -> None:
    registry = json.loads((ROOT / "docs" / "paper_parameter_registry.json").read_text(encoding="utf-8"))
    items = {item["name"]: item for item in registry["parameters"]}
    profile = sivak_table_s3_profile()
    cycle = items["sivak_composite_xz_cycle"]["value"]
    discrepancy = items["sivak_sbs_duration_source_discrepancy_ns"]["value"]
    reset = items["sivak_measurement_feedback_reset_duration_ns"]["value"]
    assert profile.total_duration_ns / 1000.0 == pytest.approx(cycle["constituent_step_us"])
    assert 2 * profile.total_duration_ns / 1000.0 == pytest.approx(cycle["full_xz_cycle_us"])
    assert profile.prose_sbs_ns == discrepancy["prose_sbs"]
    assert profile.sbs_layer_sum_ns == discrepancy["table_layer_sum"]
    assert profile.prose_reset_ns == reset["prose_subroutine"]
    assert profile.table_reset_block_ns == reset["table_block_with_entry_exit"]

    protocol = json.loads((ROOT / "docs" / "protocol_hierarchy.json").read_text(encoding="utf-8"))
    main = next(item for item in protocol["protocols"] if item["protocol_id"] == "PROTO-SBS-MAIN")
    timeline = main["cycle_contract"]["implemented_reference_timeline"]
    assert timeline == {
        "profile_id": profile.profile_id,
        "constituent_ns": 4924,
        "full_xz_ns": 9848,
        "scope": TIMING_EVIDENCE_SCOPE,
    }
    assert "experimental cycle state machine" not in main["required_future_implementation"]
    assert "T2.0.4" not in main["future_tasks"]
    update = next(item for item in protocol["implementation_updates"] if item["task_id"] == "T2.0.4")
    assert all((ROOT / path).is_file() for path in update["artifacts"])
    assert update["evidence_scope"] == "literature_reference_timeline_state_machine_not_target_board_measurement"


def test_state_machine_contains_no_wall_clock_sleep_or_latency_measurement() -> None:
    source = inspect.getsource(cycle_module)
    assert "sleep(" not in source
    assert "perf_counter" not in source
    assert "time_ns(" not in source
    assert TIMING_EVIDENCE_SCOPE in source


def test_independent_runtimes_do_not_share_transition_state() -> None:
    machine = _machine()
    x = machine.start_constituent(
        SBSConstituentControl("X", "g", "no_reset_pulse", "theta_g_calibration_slot")
    )
    z = machine.start_constituent(
        SBSConstituentControl("Z", "e", "conditional_e_to_g_reset", "theta_e_calibration_slot")
    )
    assert x.advance().phase_id == "enter_cycle"
    assert x.next_phase_id == "enter_sbs"
    assert z.next_phase_id == "enter_cycle"
    assert z.run_to_completion().control.quadrature == "Z"
    assert x.next_phase_id == "enter_sbs"


def test_invalid_controls_mappings_profiles_and_start_times_fail_closed() -> None:
    with pytest.raises(ValueError, match="quadrature"):
        SBSConstituentControl("Y", "g", "none", "theta")
    with pytest.raises(ValueError, match="observed_class"):
        SBSConstituentControl("X", "f", "none", "theta")
    with pytest.raises(ValueError, match="exactly"):
        SBSCycleStateMachine(
            virtual_rotation_key_by_observation={"g": "theta_g", "e": "theta_e"},
            virtual_rotation_provenance="test",
        )
    with pytest.raises(ValueError, match="non-empty"):
        SBSCycleStateMachine(
            virtual_rotation_key_by_observation={"g": "theta_g", "e": "theta_e", "leakage": ""},
            virtual_rotation_provenance="test",
        )
    control = SBSConstituentControl("X", "g", "none", "theta_g")
    with pytest.raises(ValueError, match="non-negative"):
        _machine().start_constituent(control, start_ns=-1)

    phase = CyclePhaseSpec("a", "g", 1, "a")
    with pytest.raises(ValueError, match="unique"):
        SBSCycleTimingProfile(
            profile_id="bad",
            phases=(phase, phase),
            source_path="x",
            source_line=1,
            source_fragment="x",
            prose_sbs_ns=1,
            prose_reset_ns=1,
        )
    with pytest.raises(ValueError, match="evidence_scope"):
        SBSCycleTimingProfile(
            profile_id="bad",
            phases=(phase,),
            source_path="x",
            source_line=1,
            source_fragment="x",
            prose_sbs_ns=1,
            prose_reset_ns=1,
            evidence_scope="measured_board",
        )
