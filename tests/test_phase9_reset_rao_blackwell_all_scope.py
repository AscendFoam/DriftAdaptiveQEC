from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.phase9_dual_backend_qualification import (
    _action_words,
    _initial_fock_ket,
    _one_step,
)
from cnn_fpga.benchmark.phase9_fresh_twin_qualification import build_simulators
from physics.phase9_reset_rao_blackwell_powered import (
    _boundary_stable_probability,
    evaluate_expected_reset_powered,
    expected_primary_result,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "configs/phase9/t_risk_20260726_01_fresh_twin_qualification.json"
        ).read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("backend", ["A", "B"])
@pytest.mark.parametrize("scope", ["shared", "logical"])
def test_expected_reset_is_primary_for_shared_and_logical_scope(
    backend: str,
    scope: str,
) -> None:
    config = _config()
    simulators = build_simulators(config, 8)
    simulator = simulators[backend]
    if scope == "logical":
        state, evaluator = simulator.initialize_logical("+i")
    else:
        ket, ancilla = _initial_fock_ket("vacuum_f", 8)
        state = simulator.initialize_fock(
            oscillator_ket=ket,
            ancilla_state=ancilla,
        )
        evaluator = None
    action = _action_words()["RESET"]
    evidence = evaluate_expected_reset_powered(
        backend=backend,
        simulator=simulator,
        state=state,
        evaluator=evaluator,
        action=action,
        seed=31_337,
    )
    primary = expected_primary_result(evidence, simulator=simulator)
    assert np.allclose(
        primary.state.joint_density,
        evidence.expected_density,
        rtol=0.0,
        atol=2.0e-12,
    )
    levels = (
        primary.truth.post_reset_level_probabilities
        if backend == "A"
        else primary.truth.post_reset_levels
    )
    assert np.allclose(levels, evidence.expected_levels, rtol=0.0, atol=5e-12)
    assert primary.state.round_index == state.round_index + 1
    assert len(evidence.pre_reset_causal_receipt_sha256) == 64
    assert evidence.sampled_role == "SAMPLED_NATIVE_RESET_BRANCH_NONVOTING_STRESS_ONLY"
    assert evidence.sampled_hidden_outcome in {"success", "failure"}
    if scope == "logical":
        assert primary.logical is not None
        survival = (
            primary.logical.code_survival_probability
            if backend == "A"
            else primary.logical.code_survival
        )
        assert 0.0 <= survival <= 1.0


@pytest.mark.parametrize("backend", ["A", "B"])
def test_expected_reset_state_is_used_by_subsequent_trajectory_round(
    backend: str,
) -> None:
    config = _config()
    simulator = build_simulators(config, 8)[backend]
    state, evaluator = simulator.initialize_logical("-")
    evidence = evaluate_expected_reset_powered(
        backend=backend,
        simulator=simulator,
        state=state,
        evaluator=evaluator,
        action=_action_words()["RESET"],
        seed=70_001,
    )
    primary = expected_primary_result(evidence, simulator=simulator)
    next_evaluator = (
        primary.logical.evaluator_state
        if backend == "A"
        else primary.logical.evaluator
    )
    continued = _one_step(
        backend=backend,
        simulator=simulator,
        state=primary.state,
        evaluator=next_evaluator,
        action=_action_words()["IDLE"],
        seed=70_001,
    )
    assert continued.state.round_index == 2
    assert np.all(np.isfinite(continued.state.joint_density))
    assert abs(np.trace(continued.state.joint_density).real - 1.0) <= 2.0e-9


@pytest.mark.parametrize("backend", ["A", "B"])
def test_pre_reset_causal_receipt_is_replay_stable_and_state_sensitive(
    backend: str,
) -> None:
    config = _config()
    simulator = build_simulators(config, 8)[backend]
    ket, ancilla = _initial_fock_ket("vacuum_f", 8)
    state = simulator.initialize_fock(oscillator_ket=ket, ancilla_state=ancilla)
    kwargs = {
        "backend": backend,
        "simulator": simulator,
        "state": state,
        "evaluator": None,
        "action": _action_words()["RESET"],
    }
    first = evaluate_expected_reset_powered(seed=101, **kwargs)
    replay = evaluate_expected_reset_powered(seed=101, **kwargs)
    changed = evaluate_expected_reset_powered(seed=102, **kwargs)
    assert (
        first.pre_reset_causal_receipt_sha256
        == replay.pre_reset_causal_receipt_sha256
    )
    assert (
        first.pre_reset_causal_receipt_sha256
        != changed.pre_reset_causal_receipt_sha256
    )
    assert np.array_equal(first.expected_density, replay.expected_density)


def test_only_numerically_null_reset_branches_are_snapped() -> None:
    assert _boundary_stable_probability(1.0 - 1.0e-13) == 1.0
    assert _boundary_stable_probability(1.0e-13) == 0.0
    assert _boundary_stable_probability(1.0 - 1.0e-10) == pytest.approx(
        1.0 - 1.0e-10
    )
    assert _boundary_stable_probability(1.0e-10) == pytest.approx(1.0e-10)
    with pytest.raises(RuntimeError, match="probability drift"):
        _boundary_stable_probability(1.0 + 1.0e-8)


@pytest.mark.parametrize("backend", ["A", "B"])
def test_expected_reset_can_be_reapplied_after_expected_state_continuation(
    backend: str,
) -> None:
    config = _config()
    simulator = build_simulators(config, 8)[backend]
    state, evaluator = simulator.initialize_logical("-")
    actions = _action_words()
    sequence = ("X", "Z", "IDLE", "XZ", "RESET", "HOLD") * 2
    reset_probabilities: list[float] = []
    for round_index, action_name in enumerate(sequence):
        action = actions[action_name]
        if action_name == "RESET":
            evidence = evaluate_expected_reset_powered(
                backend=backend,
                simulator=simulator,
                state=state,
                evaluator=evaluator,
                action=action,
                seed=90_000 + round_index,
            )
            result = expected_primary_result(evidence, simulator=simulator)
            reset_probabilities.append(evidence.success_probability)
        else:
            result = _one_step(
                backend=backend,
                simulator=simulator,
                state=state,
                evaluator=evaluator,
                action=action,
                seed=90_000 + round_index,
            )
        state = result.state
        evaluator = (
            result.logical.evaluator_state
            if backend == "A"
            else result.logical.evaluator
        )
    assert len(reset_probabilities) == 2
    assert all(0.0 <= value <= 1.0 for value in reset_probabilities)
    assert state.round_index == 12
    assert np.all(np.isfinite(state.joint_density))
