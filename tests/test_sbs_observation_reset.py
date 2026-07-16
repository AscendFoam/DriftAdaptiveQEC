from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from physics.sbs_error_space import make_trickle_down_chain
from physics.sbs_observation_reset import (
    HIDDEN_ANCILLA_STATES,
    OBSERVED_CLASSES,
    HiddenAncillaMemory,
    ObservedSyndromeMemory,
    PairedSyndrome,
    SBSObservationResetModel,
    ideal_syndrome_from_kraus,
    make_persistent_leakage_model,
)


ROOT = Path(__file__).resolve().parents[1]


def _ideal_readout() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _model(
    *,
    readout: np.ndarray | None = None,
    f_g: float = 0.0,
    f_e: float = 0.0,
    higher_g: float = 0.0,
    higher_e: float = 0.0,
    e_reset: float = 1.0,
    f_reset: float = 1.0,
    higher_reset: float = 0.0,
    counter_max: int = 31,
) -> SBSObservationResetModel:
    return make_persistent_leakage_model(
        readout_confusion=_ideal_readout() if readout is None else readout,
        f_injection_given_g=f_g,
        f_injection_given_e=f_e,
        higher_injection_given_g=higher_g,
        higher_injection_given_e=higher_e,
        e_reset_success=e_reset,
        f_reset_success=f_reset,
        higher_reset_success=higher_reset,
        counter_max=counter_max,
        readout_provenance="unit-test synthetic full 4x3 matrix",
        parameter_provenance="unit-test modeling assumptions",
    )


def test_kraus_label_order_is_zx_but_execution_and_pair_storage_are_xz() -> None:
    assert ideal_syndrome_from_kraus("K_gg") == PairedSyndrome(x="g", z="g")
    assert ideal_syndrome_from_kraus("K_ge") == PairedSyndrome(x="e", z="g")
    assert ideal_syndrome_from_kraus("K_eg") == PairedSyndrome(x="g", z="e")
    assert ideal_syndrome_from_kraus("K_ee") == PairedSyndrome(x="e", z="e")
    with pytest.raises(ValueError, match="ideal Kraus label"):
        ideal_syndrome_from_kraus("ge")


def test_ideal_observation_preserves_paired_syndrome_and_resets_e() -> None:
    model = _model()
    step = model.step("K_ge", seed=3)
    assert step.truth.ideal_syndrome == PairedSyndrome(x="e", z="g")
    assert step.truth.hidden_pre_readout == ("e", "g")
    assert step.observed.syndrome == PairedSyndrome(x="e", z="g")
    assert step.observed.reset_actions == (
        "conditional_e_to_g_reset",
        "no_reset_pulse",
    )
    assert step.truth.hidden_post_reset == ("g", "g")
    assert step.hidden_memory.carry_state == "g"


def test_same_quadrature_e_runs_are_tracked_separately_and_saturate() -> None:
    model = _model(counter_max=2)
    trajectory = model.simulate(
        ("K_ge", "K_ge", "K_ge", "K_eg", "K_eg"),
        seed=9,
    )
    assert [cycle.x_e_run for cycle in trajectory.observed_cycles] == [1, 2, 2, 0, 0]
    assert [cycle.z_e_run for cycle in trajectory.observed_cycles] == [0, 0, 0, 1, 2]
    assert trajectory.final_observed_memory.x_e_run == 0
    assert trajectory.final_observed_memory.z_e_run == 2


def test_f_is_observed_as_leakage_and_successfully_reset_without_becoming_fifth_kraus_label() -> None:
    model = _model()
    step = model.step(
        "K_gg",
        hidden_memory=HiddenAncillaMemory(carry_state="f"),
        seed=5,
    )
    assert step.truth.ideal_kraus_label == "K_gg"
    assert step.truth.hidden_pre_readout == ("f", "g")
    assert step.observed.syndrome == PairedSyndrome(x="leakage", z="g")
    assert step.truth.hidden_post_reset == ("g", "g")
    assert step.observed.reset_actions[0] == "conditional_f_or_higher_reset_attempt"
    assert step.hidden_memory.carry_state == "g"


def test_unaddressed_higher_state_creates_hidden_and_observed_leakage_streaks() -> None:
    model = _model(higher_reset=0.0, counter_max=7)
    trajectory = model.simulate(
        ("K_gg", "K_gg", "K_gg", "K_gg"),
        seed=10,
        initial_hidden_memory=HiddenAncillaMemory(carry_state="higher"),
    )
    assert [cycle.syndrome.as_tuple() for cycle in trajectory.observed_cycles] == [
        ("leakage", "leakage")
    ] * 4
    assert [cycle.leakage_cycle_run for cycle in trajectory.observed_cycles] == [1, 2, 3, 4]
    assert [cycle.leakage_constituent_run for cycle in trajectory.observed_cycles] == [2, 4, 6, 7]
    assert [cycle.hidden_higher_run for cycle in trajectory.truth_cycles] == [2, 4, 6, 7]
    assert trajectory.final_hidden_memory.carry_state == "higher"


def test_hidden_higher_can_be_misclassified_as_g_without_leaking_truth_to_counters() -> None:
    readout = _ideal_readout()
    readout[3] = [1.0, 0.0, 0.0]
    model = _model(readout=readout, higher_reset=0.0)
    step = model.step(
        "K_gg",
        hidden_memory=HiddenAncillaMemory(carry_state="higher"),
        seed=1,
    )
    assert step.truth.hidden_pre_readout == ("higher", "higher")
    assert step.truth.hidden_higher_run == 2
    assert step.observed.syndrome == PairedSyndrome(x="g", z="g")
    assert step.observed.leakage_cycle_run == 0
    assert step.observed.leakage_constituent_run == 0


def test_reset_action_is_selected_from_observation_not_hidden_truth() -> None:
    readout = _ideal_readout()
    readout[2] = [0.0, 1.0, 0.0]  # f is misclassified as e.
    model = _model(readout=readout, f_reset=1.0)
    step = model.step(
        "K_gg",
        hidden_memory=HiddenAncillaMemory(carry_state="f"),
        seed=2,
    )
    assert step.truth.hidden_pre_readout == ("f", "f")
    assert step.observed.syndrome == PairedSyndrome(x="e", z="e")
    assert step.observed.reset_actions == (
        "conditional_e_to_g_reset",
        "conditional_e_to_g_reset",
    )
    # e-action does not secretly use hidden truth to invoke the f reset branch.
    assert step.truth.hidden_post_reset == ("f", "f")
    assert step.hidden_memory.carry_state == "f"


def test_readout_confusion_monte_carlo_matches_full_matrix() -> None:
    readout = _ideal_readout()
    readout[0] = [0.80, 0.12, 0.08]
    model = _model(readout=readout)
    cycles = 5000
    trajectory = model.simulate(("K_gg",) * cycles, seed=938)
    counts = Counter(
        observed
        for cycle in trajectory.observed_cycles
        for observed in cycle.syndrome.as_tuple()
    )
    total = 2 * cycles
    assert counts["g"] / total == pytest.approx(0.80, abs=0.015)
    assert counts["e"] / total == pytest.approx(0.12, abs=0.012)
    assert counts["leakage"] / total == pytest.approx(0.08, abs=0.010)


def test_f_reset_success_frequency_matches_explicit_parameter() -> None:
    model = _model(f_reset=0.73)
    samples = 4000
    successes = 0
    for seed in range(samples):
        step = model.step(
            "K_gg",
            hidden_memory=HiddenAncillaMemory(carry_state="f"),
            seed=seed,
        )
        successes += step.truth.hidden_post_reset[0] == "g"
    assert successes / samples == pytest.approx(0.73, abs=0.025)


def test_higher_leakage_streak_survival_matches_explicit_reset_probability() -> None:
    model = _model(higher_reset=0.20)
    samples = 3000
    at_least_four_constituents = 0
    for seed in range(samples):
        trajectory = model.simulate(
            ("K_gg",) * 3,
            seed=seed,
            initial_hidden_memory=HiddenAncillaMemory(carry_state="higher"),
        )
        flattened = [
            observed
            for cycle in trajectory.observed_cycles
            for observed in cycle.syndrome.as_tuple()
        ]
        streak = 0
        for observed in flattened:
            if observed != "leakage":
                break
            streak += 1
        at_least_four_constituents += streak >= 4
    # First higher observation is guaranteed; survival to observations 2--4 requires
    # three consecutive reset failures: (1-0.2)^3.
    assert at_least_four_constituents / samples == pytest.approx(0.8**3, abs=0.03)


def test_seeded_trajectory_is_reproducible_and_accepts_t2_0_2_ideal_sequence() -> None:
    error_space = make_trickle_down_chain(
        max_depth=4,
        one_step_probability=0.6,
        two_step_probability=0.2,
    )
    ideal = error_space.sample_trajectory(initial_subspace="C4", cycles=20, seed=74)
    model = _model()
    first = model.simulate(ideal.outcomes, seed=81)
    second = model.simulate(ideal.outcomes, seed=81)
    assert first == second
    assert tuple(cycle.ideal_kraus_label for cycle in first.truth_cycles) == ideal.outcomes
    assert len(first.observed_cycles) == len(first.truth_cycles) == 20


def test_deployable_records_contain_no_hidden_ideal_or_truth_fields() -> None:
    model = _model()
    trajectory = model.simulate(("K_ge", "K_ee"), seed=7)
    records = trajectory.deployable_records()
    assert len(records) == 2
    forbidden_tokens = ("hidden", "ideal", "truth", "carry")
    for record in records:
        keys = " ".join(record).lower()
        assert not any(token in keys for token in forbidden_tokens)
        assert set(record) == {
            "cycle_index",
            "syndrome_x",
            "syndrome_z",
            "reset_action_x",
            "reset_action_z",
            "x_e_run",
            "z_e_run",
            "leakage_constituent_run",
            "leakage_cycle_run",
            "observation_scope",
        }
    assert model.exposes_hidden_truth_to_deployable_view is False


def test_literature_partial_fidelities_are_not_silently_completed() -> None:
    registry = json.loads((ROOT / "docs" / "paper_parameter_registry.json").read_text(encoding="utf-8"))
    item = next(
        parameter
        for parameter in registry["parameters"]
        if parameter["name"] == "sivak_readout_fidelity_partial_matrix"
    )
    assert item["value"]["F_g"] == 0.9997
    assert item["value"]["F_e"] == 0.9914
    assert item["value"]["F_f"] is None
    assert item["value"]["off_diagonal_transition_matrix"] is None
    assert inspect.signature(make_persistent_leakage_model).parameters["readout_confusion"].default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="shape \\(4, 3\\)"):
        _model(readout=np.eye(3))


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("readout_not_normalized", "rows must sum to 1"),
        ("readout_negative", "non-negative"),
        ("preparation_not_normalized", "rows must sum to 1"),
        ("reset_not_normalized", "rows must sum to 1"),
    ],
)
def test_general_model_rejects_invalid_probability_kernels(mutation: str, message: str) -> None:
    base = _model()
    preparation = np.array(base.preparation_kernel, copy=True)
    readout = np.array(base.readout_confusion, copy=True)
    reset = np.array(base.reset_kernel, copy=True)
    if mutation == "readout_not_normalized":
        readout[0] = [0.5, 0.2, 0.1]
    elif mutation == "readout_negative":
        readout[0] = [1.1, -0.1, 0.0]
    elif mutation == "preparation_not_normalized":
        preparation[0, 0] *= 0.5
    elif mutation == "reset_not_normalized":
        reset[0, 0] *= 0.5
    with pytest.raises(ValueError, match=message):
        SBSObservationResetModel(
            preparation_kernel=preparation,
            readout_confusion=readout,
            reset_kernel=reset,
            reset_action_by_observation=base.reset_action_by_observation,
            counter_max=7,
            preparation_provenance="test",
            readout_provenance="test",
            reset_provenance="test",
        )


def test_builder_rejects_invalid_assumptions_provenance_and_counter() -> None:
    with pytest.raises(ValueError, match="must not exceed 1"):
        _model(f_g=0.7, higher_g=0.5)
    with pytest.raises(ValueError, match="counter_max must be at least 1"):
        _model(counter_max=0)
    with pytest.raises(ValueError, match="readout_provenance"):
        make_persistent_leakage_model(
            readout_confusion=_ideal_readout(),
            f_injection_given_g=0.0,
            f_injection_given_e=0.0,
            higher_injection_given_g=0.0,
            higher_injection_given_e=0.0,
            e_reset_success=1.0,
            f_reset_success=1.0,
            higher_reset_success=0.0,
            counter_max=7,
            readout_provenance="",
            parameter_provenance="test",
        )


def test_memory_cycle_mismatch_and_invalid_public_inputs_fail_closed() -> None:
    model = _model()
    with pytest.raises(ValueError, match="same cycle_index"):
        model.step(
            "K_gg",
            hidden_memory=HiddenAncillaMemory(cycle_index=1),
            observed_memory=ObservedSyndromeMemory(cycle_index=0),
            seed=1,
        )
    with pytest.raises(TypeError, match="sequence of labels"):
        model.simulate("K_gg", seed=1)
    with pytest.raises(ValueError, match="all ideal Kraus labels"):
        model.simulate(("K_gg", "bad"), seed=1)
    with pytest.raises(ValueError):
        PairedSyndrome(x="f", z="g")


def test_protocol_registry_freezes_zx_label_and_xz_execution_order() -> None:
    registry = json.loads((ROOT / "docs" / "protocol_hierarchy.json").read_text(encoding="utf-8"))
    main = next(item for item in registry["protocols"] if item["protocol_id"] == "PROTO-SBS-MAIN")
    observation = main["observation_contract"]
    assert observation["kraus_label_character_order"] == ["Z", "X"]
    assert observation["chronological_constituent_order"] == ["X", "Z"]
    assert tuple(HIDDEN_ANCILLA_STATES) == ("g", "e", "f", "higher")
    assert tuple(OBSERVED_CLASSES) == ("g", "e", "leakage")
    assert main["current_status"] == (
        "error_space_through_finite_cutoff_nmf_directional_ranking_"
        "implemented_long_horizon_memory_robustness_and_device_fidelity_blocked"
    )
    assert "paired syndrome with g/e/f readout-reset branch" not in main["required_future_implementation"]
    assert "T2.0.3" not in main["future_tasks"]
    update = next(item for item in registry["implementation_updates"] if item["task_id"] == "T2.0.3")
    assert all((ROOT / path).is_file() for path in update["artifacts"])
    assert update["evidence_scope"] == "protocol_aligned_hidden_observed_reset_effective_model_not_device_calibrated"
