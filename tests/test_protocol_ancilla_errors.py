from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from physics.constants import LATTICE_CONST
from physics.protocol_ancilla_errors import (
    SECONDARY_PROTOCOL_NOISE_REGISTRY,
    SBSAncillaFaultOverlay,
    SBSFaultOverlayConfig,
    SharpenTrimAncillaConfig,
    SharpenTrimAncillaModel,
    SharpenTrimMemory,
    run_protocol_ancilla_validation,
    secondary_protocol_noise_specs,
    write_protocol_ancilla_validation,
)
from physics.sbs_observation_reset import (
    HiddenAncillaMemory,
    make_persistent_leakage_model,
)


ROOT = Path(__file__).resolve().parents[1]


def _sbs_base(
    *,
    readout: np.ndarray | None = None,
    f_reset: float = 1.0,
    higher_reset: float = 1.0,
):
    return make_persistent_leakage_model(
        readout_confusion=(
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            if readout is None
            else readout
        ),
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=f_reset,
        higher_reset_success=higher_reset,
        counter_max=127,
        readout_provenance="unit-test full hidden-to-observed matrix",
        parameter_provenance="unit-test explicit reset assumptions",
    )


def _sbs_config(
    *,
    bit: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
    phase: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
    logical: tuple[float, float] = (0.0, 0.0),
    rotation: float = 0.6,
) -> SBSFaultOverlayConfig:
    return SBSFaultOverlayConfig(
        bit_flip_probabilities=bit or ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        phase_flip_probabilities=phase
        or ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        logical_fault_given_big_cd_bit=logical,
        phase_backaction_scale=(0.011, 0.013),
        small_cd_bit_backaction_scale=(0.021, 0.023),
        misclassification_rotation_max_rad=rotation,
        parameter_provenance="unit-test stage-resolved assumptions",
    )


def _sharp_config(
    *,
    bit: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    phase: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    leakage: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    readout: np.ndarray | None = None,
    correct_reset: tuple[float, float] = (1.0, 1.0),
    wrong_reset: float = 1.0,
    leakage_reset: float = 1.0,
) -> SharpenTrimAncillaConfig:
    return SharpenTrimAncillaConfig(
        bit_flip_probabilities=bit,
        phase_flip_probabilities=phase,
        leakage_injection_probabilities=leakage,
        readout_confusion=(
            np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
            if readout is None
            else readout
        ),
        correct_reset_success=correct_reset,
        wrong_sign_reset_success=wrong_reset,
        leakage_reset_success=leakage_reset,
        peak_feedback_fraction=0.08,
        peak_feedback_asymmetry_fraction=0.01,
        trim_feedback_fraction=0.5,
        lattice=LATTICE_CONST,
        counter_max=127,
        parameter_provenance="unit-test native sharpen-trim assumptions",
        readout_provenance="unit-test full 3x2 matrix",
        reset_provenance="unit-test conditional pi/2 reset assumptions",
    )


def test_sbs_big_cd_bit_flip_toggles_constituent_and_can_be_logical() -> None:
    model = SBSAncillaFaultOverlay(
        _sbs_base(),
        _sbs_config(bit=((0.0, 1.0, 0.0), (0.0, 0.0, 0.0)), logical=(1.0, 0.0)),
    )
    step = model.step("K_gg", seed=12)
    truth = step.fault_truth
    assert truth.original_ideal_kraus_label == "K_gg"
    assert truth.faulted_ideal_kraus_label == "K_ge"
    assert truth.logical_backaction_by_constituent == (True, False)
    assert step.observation_reset.observed.syndrome.as_tuple() == ("e", "g")
    assert [(event.constituent, event.stage) for event in truth.events] == [
        ("X", "big_cd")
    ]


def test_sbs_phase_flip_does_not_toggle_z_basis_but_adds_continuous_backaction() -> None:
    model = SBSAncillaFaultOverlay(
        _sbs_base(),
        _sbs_config(phase=((0.0, 1.0, 0.0), (0.0, 0.0, 0.0))),
    )
    step = model.step("K_gg", seed=13)
    event = step.fault_truth.events[0]
    assert event.fault_type == "phase_flip"
    assert not event.toggles_z_basis_outcome
    assert abs(event.continuous_backaction) == pytest.approx(0.011)
    assert step.fault_truth.faulted_ideal_kraus_label == "K_gg"
    assert step.fault_truth.logical_backaction_by_constituent == (False, False)


def test_sbs_small_cd_bit_flip_is_signed_backaction_and_outcome_toggle() -> None:
    model = SBSAncillaFaultOverlay(
        _sbs_base(),
        _sbs_config(bit=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
    )
    step = model.step("K_gg", seed=14)
    event = step.fault_truth.events[0]
    assert event.stage == "small_cd"
    assert event.toggles_z_basis_outcome
    assert not event.logical_backaction
    assert abs(event.continuous_backaction) == pytest.approx(0.021)
    assert step.fault_truth.faulted_ideal_kraus_label == "K_ge"


def test_sbs_two_bit_flips_cancel_observation_toggle_but_keep_event_provenance() -> None:
    model = SBSAncillaFaultOverlay(
        _sbs_base(),
        _sbs_config(bit=((1.0, 1.0, 0.0), (0.0, 0.0, 0.0))),
    )
    step = model.step("K_gg", seed=15)
    assert step.fault_truth.faulted_ideal_kraus_label == "K_gg"
    assert len(step.fault_truth.events) == 2


def test_sbs_readout_misclassification_creates_bounded_virtual_rotation_truth() -> None:
    flipped = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    model = SBSAncillaFaultOverlay(_sbs_base(readout=flipped), _sbs_config(rotation=0.6))
    step = model.step("K_gg", seed=16)
    assert step.fault_truth.readout_misclassified == (True, True)
    assert all(0.0 < abs(value) <= 0.6 for value in step.fault_truth.virtual_rotation_error_rad)
    assert step.observation_reset.observed.syndrome.as_tuple() == ("e", "e")


def test_sbs_overlay_preserves_base_leakage_and_reset_persistence() -> None:
    model = SBSAncillaFaultOverlay(
        _sbs_base(higher_reset=0.0),
        _sbs_config(),
    )
    trajectory = model.simulate(
        ("K_gg", "K_gg"),
        seed=17,
        initial_hidden_memory=HiddenAncillaMemory(carry_state="higher"),
    )
    assert [step.observation_reset.truth.hidden_pre_readout[0] for step in trajectory.steps] == [
        "higher",
        "higher",
    ]
    assert trajectory.final_hidden_memory.carry_state == "higher"
    assert [record["syndrome_x"] for record in trajectory.deployable_records()] == [
        "leakage",
        "leakage",
    ]


def test_sbs_deployable_record_excludes_fault_and_hidden_truth() -> None:
    model = SBSAncillaFaultOverlay(_sbs_base(), _sbs_config())
    record = model.step("K_ge", seed=18).deployable_record()
    serialized = json.dumps(record)
    for forbidden in (
        "hidden",
        "fault",
        "original",
        "rotation",
        "misclassified",
        "backaction",
    ):
        assert forbidden not in serialized


def test_sbs_seed_replay_and_prefix_are_exact() -> None:
    model = SBSAncillaFaultOverlay(
        _sbs_base(),
        _sbs_config(
            bit=((0.1, 0.2, 0.1), (0.05, 0.1, 0.05)),
            phase=((0.1, 0.1, 0.1), (0.1, 0.1, 0.1)),
        ),
    )
    short = model.simulate(("K_gg",) * 8, seed=19)
    replay = model.simulate(("K_gg",) * 8, seed=19)
    long = model.simulate(("K_gg",) * 16, seed=19)
    assert short == replay
    assert short.steps == long.steps[:8]


def test_sharpen_trim_four_round_schedule_feedback_and_protocol_frame() -> None:
    model = SharpenTrimAncillaModel(_sharp_config())
    trajectory = model.simulate(("+y",) * 4, seed=20)
    assert [item.round_type for item in trajectory.observed_rounds] == [
        "q_peak_sharpen",
        "p_peak_sharpen",
        "q_envelope_trim",
        "p_envelope_trim",
    ]
    assert [item.feedback_axis for item in trajectory.observed_rounds] == ["q", "p", "q", "p"]
    assert [
        (item.pauli_frame_x, item.pauli_frame_z)
        for item in trajectory.observed_rounds
    ] == [(0, 1), (1, 1), (0, 1), (0, 0)]
    assert trajectory.observed_rounds[0].feedback_displacement == pytest.approx(
        0.09 * LATTICE_CONST
    )
    assert trajectory.observed_rounds[2].feedback_displacement == pytest.approx(
        0.5 * LATTICE_CONST
    )


def test_sharpen_trim_phase_flip_toggles_native_y_and_wrong_feedback() -> None:
    model = SharpenTrimAncillaModel(_sharp_config(phase=(1.0, 1.0, 1.0, 1.0)))
    step = model.step("+y", seed=21)
    assert step.truth.phase_flip
    assert not step.truth.bit_flip
    assert step.truth.hidden_pre_readout == "-y"
    assert step.observed.observed_y == "-y"
    assert step.truth.feedback_direction_wrong
    assert step.truth.logical_backaction == "I"


def test_sharpen_trim_simultaneous_bit_and_phase_flip_cancel_y_toggle() -> None:
    model = SharpenTrimAncillaModel(
        _sharp_config(bit=(1.0,) * 4, phase=(1.0,) * 4)
    )
    step = model.step("+y", seed=22)
    assert step.truth.bit_flip and step.truth.phase_flip
    assert step.truth.hidden_pre_readout == "+y"
    assert step.observed.observed_y == "+y"


def test_sharpen_trim_peak_bit_fault_only_has_middle_window_logical_backaction() -> None:
    model = SharpenTrimAncillaModel(_sharp_config(bit=(1.0,) * 4))
    trajectory = model.simulate(("+y",) * 20_000, seed=23)
    logical = np.asarray(
        [truth.logical_backaction != "I" for truth in trajectory.truth_rounds],
        dtype=float,
    )
    # Only half the rounds are peak rounds, and only the middle half of a peak
    # interaction is designated as a logical-backaction region.
    assert float(np.mean(logical)) == pytest.approx(0.25, abs=0.012)
    assert all(
        truth.logical_backaction == "I"
        for truth in trajectory.truth_rounds
        if "envelope_trim" in truth.round_type
    )


def test_sharpen_trim_random_physical_fault_never_leaks_into_deployable_frame() -> None:
    clean = SharpenTrimAncillaModel(_sharp_config())
    faulty = SharpenTrimAncillaModel(_sharp_config(bit=(1.0,) * 4))
    clean_trace = clean.simulate(("+y",) * 8, seed=24)
    faulty_trace = faulty.simulate(("+y",) * 8, seed=24)
    clean_frames = [
        (item.pauli_frame_x, item.pauli_frame_z) for item in clean_trace.observed_rounds
    ]
    faulty_frames = [
        (item.pauli_frame_x, item.pauli_frame_z) for item in faulty_trace.observed_rounds
    ]
    assert clean_frames == faulty_frames
    assert any(item.logical_backaction != "I" for item in faulty_trace.truth_rounds)


def test_sharpen_trim_reset_failure_persists_hidden_y_without_truth_leak() -> None:
    model = SharpenTrimAncillaModel(
        _sharp_config(correct_reset=(0.0, 0.0), wrong_reset=0.0)
    )
    trajectory = model.simulate(("+y", "-y"), seed=25)
    assert [item.hidden_pre_readout for item in trajectory.truth_rounds] == ["+y", "+y"]
    assert trajectory.final_memory.carry_state == "+y"
    assert trajectory.final_memory.reset_failure_run == 2
    second = trajectory.observed_rounds[1]
    assert second.observed_y == "+y"
    assert trajectory.truth_rounds[1].feedback_direction_wrong
    assert "reset_failure_run" not in second.as_deployable_dict()
    assert "carry_state" not in second.as_deployable_dict()


def test_sharpen_trim_leakage_is_binary_observed_and_persists_only_in_truth_memory() -> None:
    model = SharpenTrimAncillaModel(
        _sharp_config(leakage=(1.0, 0.0, 0.0, 0.0), leakage_reset=0.0)
    )
    trajectory = model.simulate(("+y", "+y", "+y"), seed=26)
    assert all(item.hidden_pre_readout == "leakage" for item in trajectory.truth_rounds)
    assert trajectory.final_memory.carry_state == "leakage"
    assert trajectory.final_memory.leakage_run == 3
    assert all(item.observed_y in {"+y", "-y"} for item in trajectory.observed_rounds)
    assert all(
        "leakage" not in json.dumps(item.as_deployable_dict())
        for item in trajectory.observed_rounds
    )


def test_sharpen_trim_full_leakage_confusion_row_is_statistically_calibrated() -> None:
    readout = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.73, 0.27]])
    model = SharpenTrimAncillaModel(
        _sharp_config(
            leakage=(1.0, 0.0, 0.0, 0.0),
            leakage_reset=0.0,
            readout=readout,
        )
    )
    trajectory = model.simulate(("+y",) * 20_000, seed=27)
    plus_rate = np.mean([item.observed_y == "+y" for item in trajectory.observed_rounds])
    assert float(plus_rate) == pytest.approx(0.73, abs=0.012)


def test_sharpen_trim_readout_confusion_selects_wrong_reset_action() -> None:
    flipped = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    model = SharpenTrimAncillaModel(_sharp_config(readout=flipped))
    step = model.step("+y", seed=28)
    assert step.truth.readout_misclassified
    assert step.observed.observed_y == "-y"
    assert step.observed.reset_action == "conditional_pi_over_2_reset_from_minus_y"
    assert step.truth.feedback_direction_wrong


def test_protocol_native_alphabets_are_not_interchangeable() -> None:
    sbs = SBSAncillaFaultOverlay(_sbs_base(), _sbs_config())
    sharp = SharpenTrimAncillaModel(_sharp_config())
    with pytest.raises(ValueError, match="K_"):
        sbs.step("+y", seed=29)
    with pytest.raises(ValueError, match=r"\+y or -y"):
        sharp.step("K_gg", seed=29)
    with pytest.raises(TypeError, match="sequence, not text"):
        sharp.simulate("+y", seed=29)


def test_secondary_protocol_registry_is_explicitly_non_executable() -> None:
    specs = secondary_protocol_noise_specs()
    assert set(SECONDARY_PROTOCOL_NOISE_REGISTRY) == {
        "steane",
        "knill_qunaught",
        "p_steane",
    }
    assert len(specs) == 3
    assert all(not spec.executable for spec in specs)
    assert all(spec.primary_source_required for spec in specs)
    assert {"a", "b", "data_to_ancilla_noise_ratio"} == set(
        SECONDARY_PROTOCOL_NOISE_REGISTRY["p_steane"].allowed_scan_parameters
    )


def test_seeded_validation_checks_analytic_rates_and_non_execution_boundary() -> None:
    result = run_protocol_ancilla_validation(samples=10_000, seed=2026071422)
    assert all(bool(value) for value in result.checks.values())
    assert result.sbs_phase_outcome_toggle_rate == 0.0
    assert result.sharpen_bit_logical_backaction_rate < result.sharpen_bit_middle_window_rate
    assert result.evidence_scope.endswith("not_device_calibrated")


def test_validation_writer_round_trips_json_and_boolean_scalars() -> None:
    result = run_protocol_ancilla_validation(samples=10_000, seed=2026071423)
    output = ROOT / "docs" / "_test_protocol_ancilla_validation.json"
    try:
        written = write_protocol_ancilla_validation(result, output)
        payload = json.loads(written.read_text(encoding="utf-8"))
        assert payload["samples"] == 10_000
        assert all(isinstance(value, bool) for value in payload["checks"].values())
        assert all(not item["executable"] for item in payload["secondary_protocols"])
    finally:
        output.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("factory", "pattern"),
    [
        (
            lambda: _sbs_config(bit=((0.0, 0.0), (0.0, 0.0, 0.0))),
            "numeric matrix",
        ),
        (
            lambda: _sbs_config(phase=((0.0, 0.0, 1.1), (0.0, 0.0, 0.0))),
            r"\[0, 1\]",
        ),
        (
            lambda: _sharp_config(readout=np.ones((3, 2))),
            "sum to 1",
        ),
        (
            lambda: SharpenTrimMemory(carry_state="e"),
            "carry_state",
        ),
        (
            lambda: _sharp_config(correct_reset=(1.0,)),
            "2 probabilities",
        ),
    ],
)
def test_invalid_fault_and_native_protocol_configuration_is_rejected(
    factory,
    pattern: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=pattern):
        factory()


def test_validation_rejects_demo_scale_and_invalid_seed() -> None:
    with pytest.raises(ValueError, match=">= 10000"):
        run_protocol_ancilla_validation(samples=9_999)
    with pytest.raises(TypeError, match="seed must be an integer"):
        run_protocol_ancilla_validation(samples=10_000, seed=True)
