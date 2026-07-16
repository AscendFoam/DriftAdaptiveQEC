from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from physics.sbs_error_space import (
    FULL_CYCLE_LOGICAL_FLIP,
    MODEL_SCOPE,
    SBS_OUTCOMES,
    PauliFrame,
    SBSErrorSpaceInstrument,
    embed_logical_state,
    make_trickle_down_chain,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def chain() -> SBSErrorSpaceInstrument:
    return make_trickle_down_chain(
        max_depth=4,
        one_step_probability=0.6,
        two_step_probability=0.2,
        ge_fraction=0.5,
    )


def _block_populations(state: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(
        [np.trace(state[2 * index : 2 * index + 2, 2 * index : 2 * index + 2]).real for index in range(count)]
    )


def _custom_multisubspace_instrument() -> SBSErrorSpaceInstrument:
    # C1/C2 同属 first error level，C3 属 second error level；这验证实现不把
    # subspace index 误当 recovery depth。
    size = 4
    transitions = {outcome: np.zeros((size, size)) for outcome in SBS_OUTCOMES}
    transitions["K_gg"][0, 0] = 1.0
    transitions["K_gg"][1, 1] = 0.4
    transitions["K_ge"][0, 1] = 0.6
    transitions["K_gg"][2, 2] = 0.5
    transitions["K_eg"][0, 2] = 0.5
    transitions["K_gg"][3, 3] = 0.2
    transitions["K_ge"][1, 3] = 0.3
    transitions["K_eg"][2, 3] = 0.3
    transitions["K_ee"][0, 3] = 0.2
    return SBSErrorSpaceInstrument(
        subspaces=("C0", "C1", "C2", "C3"),
        depths=(0, 1, 1, 2),
        transition_probabilities=transitions,
    )


def test_grouped_kraus_components_are_exactly_trace_preserving(chain: SBSErrorSpaceInstrument) -> None:
    assert chain.protocol_id == "PROTO-SBS-MAIN"
    assert chain.model_scope == MODEL_SCOPE
    assert tuple(chain.kraus_components) == SBS_OUTCOMES
    assert chain.hilbert_dimension == 10
    assert np.allclose(chain.completeness_matrix(), np.eye(10), rtol=0.0, atol=1.0e-14)
    assert chain.completeness_residual_norm() < 1.0e-14
    assert all(len(chain.kraus_components[outcome]) > 0 for outcome in SBS_OUTCOMES)


def test_every_outcome_branch_has_positive_choi_and_is_trace_nonincreasing(
    chain: SBSErrorSpaceInstrument,
) -> None:
    dimension = chain.hilbert_dimension
    identity = np.eye(dimension, dtype=np.complex128)
    for outcome in SBS_OUTCOMES:
        choi = np.zeros((dimension * dimension, dimension * dimension), dtype=np.complex128)
        effect = np.zeros((dimension, dimension), dtype=np.complex128)
        for component in chain.kraus_components[outcome]:
            vectorized = component.reshape(-1, order="F")
            choi += np.outer(vectorized, vectorized.conj())
            effect += component.conj().T @ component
        assert np.min(np.linalg.eigvalsh(choi)) >= -1.0e-12
        assert np.min(np.linalg.eigvalsh(identity - effect)) >= -1.0e-12


def test_no_error_limit_only_emits_gg_and_frame_correction_restores_logical_state(
    chain: SBSErrorSpaceInstrument,
) -> None:
    plus = np.full((2, 2), 0.5, dtype=np.complex128)
    state = embed_logical_state(plus, subspace_index=0, subspace_count=5)
    result = chain.apply_density_matrix(state)

    assert result.probabilities == {
        "K_gg": pytest.approx(1.0),
        "K_ge": pytest.approx(0.0),
        "K_eg": pytest.approx(0.0),
        "K_ee": pytest.approx(0.0),
    }
    expected_physical = FULL_CYCLE_LOGICAL_FLIP @ plus @ FULL_CYCLE_LOGICAL_FLIP.conj().T
    assert np.allclose(result.unconditional_state[:2, :2], expected_physical)
    assert np.allclose(result.cycle_frame_corrected_state, state, atol=1.0e-14)
    assert result.output_frame == PauliFrame(x=1, z=1)
    assert result.trace_before == pytest.approx(1.0)
    assert result.trace_after == pytest.approx(1.0)


def test_full_cycle_pauli_frame_toggles_both_bits_and_cancels_after_two_cycles() -> None:
    frame0 = PauliFrame()
    frame1 = frame0.after_full_sbs_cycle()
    frame2 = frame1.after_full_sbs_cycle()
    assert frame1 == PauliFrame(x=1, z=1)
    assert frame2 == frame0
    assert np.allclose(frame1.unitary @ frame1.unitary.conj().T, np.eye(2))


def test_chain_branch_semantics_are_gradual_and_never_one_step_project_large_errors(
    chain: SBSErrorSpaceInstrument,
) -> None:
    population = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    result = chain.apply_population(population)
    assert result.probabilities == {
        "K_gg": pytest.approx(0.2),
        "K_ge": pytest.approx(0.3),
        "K_eg": pytest.approx(0.3),
        "K_ee": pytest.approx(0.2),
    }
    assert result.unconditional_population == pytest.approx([0.0, 0.0, 0.2, 0.6, 0.2])
    assert result.unconditional_population[0] == 0.0
    assert chain.expected_depth(population) == pytest.approx(4.0)
    assert chain.expected_depth(result.unconditional_population) == pytest.approx(3.0)

    current = population
    expected_depths = [4.0]
    for _ in range(4):
        current = chain.apply_population(current).unconditional_population
        expected_depths.append(chain.expected_depth(current))
    assert np.all(np.diff(expected_depths) < 0.0)
    assert 0.0 < current[0] < 1.0  # 多轮后部分恢复，但不是假装一步归零。


def test_multiple_ci_subspaces_can_share_a_depth_without_index_depth_conflation() -> None:
    model = _custom_multisubspace_instrument()
    assert model.depths.tolist() == [0, 1, 1, 2]
    assert model.completeness_residual_norm() < 1.0e-14
    population = np.array([0.0, 0.0, 0.0, 1.0])
    result = model.apply_population(population)
    assert result.branch_populations["K_ge"] == pytest.approx([0.0, 0.3, 0.0, 0.0])
    assert result.branch_populations["K_eg"] == pytest.approx([0.0, 0.0, 0.3, 0.0])
    assert result.branch_populations["K_ee"] == pytest.approx([0.2, 0.0, 0.0, 0.0])


def test_density_matrix_and_population_paths_agree_for_random_states(chain: SBSErrorSpaceInstrument) -> None:
    for seed in range(8):
        rng = np.random.default_rng(seed)
        amplitude = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
        state = amplitude @ amplitude.conj().T
        state /= np.trace(state)
        density_result = chain.apply_density_matrix(state)
        input_population = _block_populations(state, 5)
        population_result = chain.apply_population(input_population)
        assert density_result.trace_after == pytest.approx(1.0, abs=1.0e-12)
        assert np.min(np.linalg.eigvalsh(density_result.unconditional_state)) >= -1.0e-12
        assert density_result.probabilities == pytest.approx(population_result.probabilities)
        assert _block_populations(density_result.unconditional_state, 5) == pytest.approx(
            population_result.unconditional_population
        )


def test_grouped_instrument_explicitly_dephases_inter_subspace_coherence(
    chain: SBSErrorSpaceInstrument,
) -> None:
    vector = np.zeros(10, dtype=np.complex128)
    vector[0] = 1.0 / np.sqrt(2.0)  # C0 logical |0>
    vector[2] = 1.0 / np.sqrt(2.0)  # C1 logical |0>
    state = np.outer(vector, vector.conj())
    result = chain.apply_density_matrix(state)
    assert chain.preserves_inter_subspace_coherence is False
    assert chain.device_calibrated is False
    assert result.unconditional_state[0, 2] == pytest.approx(0.0)
    assert result.cycle_frame_corrected_state[0, 2] == pytest.approx(0.0)


def test_seeded_trajectory_is_reproducible_monotone_and_tracks_frame(chain: SBSErrorSpaceInstrument) -> None:
    first = chain.sample_trajectory(initial_subspace="C4", cycles=8, seed=4821)
    second = chain.sample_trajectory(initial_subspace="C4", cycles=8, seed=4821)
    assert first == second
    assert len(first.outcomes) == 8
    assert len(first.subspaces) == len(first.depths) == len(first.frames) == 9
    assert set(first.outcomes) <= set(SBS_OUTCOMES)
    assert np.all(np.diff(first.depths) <= 0)
    for index, frame in enumerate(first.frames):
        expected_bit = index % 2
        assert frame == PauliFrame(x=expected_bit, z=expected_bit)
    assert first.observation_scope == "ideal_kraus_branch_not_noisy_readout"


def test_monte_carlo_branch_frequency_matches_analytic_transition(chain: SBSErrorSpaceInstrument) -> None:
    samples = 4000
    counts = Counter(
        chain.sample_trajectory(initial_subspace="C4", cycles=1, seed=seed).outcomes[0]
        for seed in range(samples)
    )
    empirical = {outcome: counts[outcome] / samples for outcome in SBS_OUTCOMES}
    expected = {"K_gg": 0.2, "K_ge": 0.3, "K_eg": 0.3, "K_ee": 0.2}
    for outcome in SBS_OUTCOMES:
        assert empirical[outcome] == pytest.approx(expected[outcome], abs=0.025)


def test_chain_builder_does_not_clip_or_reallocate_infeasible_boundary_probabilities() -> None:
    model = make_trickle_down_chain(
        max_depth=2,
        one_step_probability=0.55,
        two_step_probability=0.35,
        ge_fraction=0.25,
    )
    # C0 强制 no-error；C1 的 two-step branch 不存在且不会偷偷加给 one-step。
    assert model.transition_probabilities["K_gg"][0, 0] == pytest.approx(1.0)
    assert model.transition_probabilities["K_gg"][1, 1] == pytest.approx(0.45)
    assert model.transition_probabilities["K_ge"][0, 1] == pytest.approx(0.1375)
    assert model.transition_probabilities["K_eg"][0, 1] == pytest.approx(0.4125)
    assert np.sum(model.transition_probabilities["K_ee"][:, 1]) == 0.0
    assert model.transition_probabilities["K_gg"][2, 2] == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"max_depth": 1, "one_step_probability": 0.2, "two_step_probability": 0.1}, ValueError, "at least 2"),
        ({"max_depth": 2, "one_step_probability": 0.8, "two_step_probability": 0.3}, ValueError, "must not exceed 1"),
        ({"max_depth": 2, "one_step_probability": -0.1, "two_step_probability": 0.1}, ValueError, "in \\[0, 1\\]"),
        ({"max_depth": 2, "one_step_probability": 0.2, "two_step_probability": np.inf}, ValueError, "in \\[0, 1\\]"),
        ({"max_depth": True, "one_step_probability": 0.2, "two_step_probability": 0.1}, TypeError, "integer"),
    ],
)
def test_chain_builder_rejects_invalid_modeling_assumptions(kwargs: dict, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        make_trickle_down_chain(**kwargs)


def test_instrument_rejects_non_tp_negative_and_wrong_depth_transitions() -> None:
    base = _custom_multisubspace_instrument()
    matrices = {
        key: np.array(value, copy=True) for key, value in base.transition_probabilities.items()
    }

    non_tp = {key: value.copy() for key, value in matrices.items()}
    non_tp["K_gg"][3, 3] += 0.1
    with pytest.raises(ValueError, match="sum to 1"):
        SBSErrorSpaceInstrument(
            subspaces=base.subspaces,
            depths=base.depths,
            transition_probabilities=non_tp,
        )

    negative = {key: value.copy() for key, value in matrices.items()}
    negative["K_ge"][0, 1] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        SBSErrorSpaceInstrument(
            subspaces=base.subspaces,
            depths=base.depths,
            transition_probabilities=negative,
        )

    wrong_jump = {key: value.copy() for key, value in matrices.items()}
    wrong_jump["K_ge"][1, 3] = 0.0
    wrong_jump["K_ge"][0, 3] = 0.3
    with pytest.raises(ValueError, match="lower recovery depth by exactly 1"):
        SBSErrorSpaceInstrument(
            subspaces=base.subspaces,
            depths=base.depths,
            transition_probabilities=wrong_jump,
        )

    tiny_wrong_jump = {key: value.copy() for key, value in matrices.items()}
    tiny_wrong_jump["K_gg"][3, 3] -= 1.0e-13
    tiny_wrong_jump["K_ge"][0, 3] = 1.0e-13
    with pytest.raises(ValueError, match="lower recovery depth by exactly 1"):
        SBSErrorSpaceInstrument(
            subspaces=base.subspaces,
            depths=base.depths,
            transition_probabilities=tiny_wrong_jump,
        )

    with pytest.raises(ValueError, match="in \\(0, 1e-8\\]"):
        SBSErrorSpaceInstrument(
            subspaces=base.subspaces,
            depths=base.depths,
            transition_probabilities=matrices,
            normalization_tolerance=0.1,
        )


@pytest.mark.parametrize(
    "bad_state",
    [
        np.eye(8) / 7.0,
        np.diag([1.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.ones((7, 7)) / 7.0,
    ],
)
def test_density_path_rejects_nonphysical_or_wrong_shape_states(bad_state: np.ndarray) -> None:
    model = _custom_multisubspace_instrument()
    with pytest.raises(ValueError):
        model.apply_density_matrix(bad_state)


def test_protocol_registry_and_code_use_the_same_main_protocol_id(chain: SBSErrorSpaceInstrument) -> None:
    registry = json.loads((ROOT / "docs" / "protocol_hierarchy.json").read_text(encoding="utf-8"))
    main = next(item for item in registry["protocols"] if item["hierarchy_role"] == "main_digital_twin")
    assert main["protocol_id"] == chain.protocol_id
    assert main["mathematical_object"]["kraus_labels"] == list(SBS_OUTCOMES)
    assert main["current_status"] == (
        "error_space_through_finite_cutoff_nmf_directional_ranking_"
        "implemented_long_horizon_memory_robustness_and_device_fidelity_blocked"
    )
    update = next(item for item in registry["implementation_updates"] if item["task_id"] == "T2.0.2")
    assert update["protocol_id"] == chain.protocol_id
    assert update["evidence_scope"] == "protocol_aligned_coarse_grained_effective_model_not_device_calibrated"
    assert all((ROOT / path).is_file() for path in update["artifacts"])
    assert "C_i gradual trickle-down error hierarchy" not in main["required_future_implementation"]
    assert "T2.0.2" not in main["future_tasks"]
