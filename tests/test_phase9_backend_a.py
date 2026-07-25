from __future__ import annotations

from dataclasses import replace
import inspect
from math import sqrt

import numpy as np
import pytest
from scipy.sparse import eye as sparse_eye

from physics.phase9_backend_a import (
    BACKEND_A_ID,
    BACKEND_A_SCOPE,
    BackendAConfig,
    BackendADriftState,
    BackendAEvaluatorState,
    BackendAExogenous,
    BackendAObservation,
    BackendAQualificationThresholds,
    BackendAState,
    Phase9BackendASimulator,
    backend_a_exogenous,
    diagnostic_action_word,
    run_backend_a_qualification,
)
from physics.phase9_twin_contract import ActionWord, NominalAction


@pytest.fixture(scope="module")
def qualification():
    return run_backend_a_qualification()


@pytest.fixture()
def compact_config() -> BackendAConfig:
    return BackendAConfig(
        cutoff=8,
        substeps_per_segment=2,
        iq_samples=4,
        logical_grid_points=1025,
    )


def noise_free(config: BackendAConfig, **overrides: object) -> BackendAConfig:
    values: dict[str, object] = {
        "oscillator_loss_rate": 0.0,
        "oscillator_dephasing_rate": 0.0,
        "ancilla_ge_relax_rate": 0.0,
        "ancilla_fe_relax_rate": 0.0,
        "ancilla_ge_excitation_rate": 0.0,
        "ancilla_dephasing_rate": 0.0,
        "pulse_leakage_crosstalk": 0.0,
        "measurement_leakage_coupling": 0.0,
        "action_leakage_coupling": 0.0,
        "dispersive_chi": 0.0,
        "self_kerr": 0.0,
        "ramsey_angle": 0.0,
        "sense_duration": 0.0,
        "iq_centers": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        "reset_ack_error": 0.0,
        "drift_retention": (1.0, 1.0, 1.0, 1.0, 1.0),
        "drift_noise_std": (0.0, 0.0, 0.0, 0.0, 0.0),
        "drift_action_kick": 0.0,
        "drift_readout_heating": 0.0,
        "drift_leakage_heating": 0.0,
    }
    values.update(overrides)
    return replace(config, **values)


def zero_exogenous(config: BackendAConfig, round_index: int = 0) -> BackendAExogenous:
    return BackendAExogenous(
        emission_uniform=0.25,
        iq_standard_i=(0.0,) * config.iq_samples,
        iq_standard_q=(0.0,) * config.iq_samples,
        reset_uniform=0.25,
        reset_ack_uniform=0.25,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=7,
        round_index=round_index,
    )


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = 0.5 * ((left - right) + (left - right).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))


def test_config_identity_scope_and_hash_are_stable(compact_config):
    assert compact_config.backend_id == BACKEND_A_ID
    assert compact_config.scope == BACKEND_A_SCOPE
    assert compact_config.semantic_sha256() == BackendAConfig(
        cutoff=8,
        substeps_per_segment=2,
        iq_samples=4,
        logical_grid_points=1025,
    ).semantic_sha256()
    assert len(compact_config.semantic_sha256()) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cutoff", 1),
        ("substeps_per_segment", 0),
        ("iq_samples", 0),
        ("iq_sigma", 0.0),
        ("reset_success_f", 1.1),
        ("leakage_age_threshold", -0.1),
        ("logical_grid_points", 1024),
        ("action_duration", 0.0),
        ("ramsey_pulse_duration", 0.0),
    ],
)
def test_invalid_config_is_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        BackendAConfig(**{field: value})


def test_backend_identity_and_scope_cannot_be_relabelled():
    with pytest.raises(ValueError):
        BackendAConfig(backend_id="device-calibrated")
    with pytest.raises(ValueError):
        BackendAConfig(scope="hardware measured")


def test_exogenous_is_round_addressable_and_seed_deterministic(compact_config):
    first = backend_a_exogenous(seed=19, round_index=3, iq_samples=4)
    replay = backend_a_exogenous(seed=19, round_index=3, iq_samples=4)
    different_round = backend_a_exogenous(seed=19, round_index=4, iq_samples=4)
    assert first == replay
    assert first != different_round
    assert first.round_index == 3


def test_exogenous_rejects_mismatched_iq_arrays():
    with pytest.raises(ValueError, match="equal length"):
        BackendAExogenous(
            emission_uniform=0.2,
            iq_standard_i=(0.0,),
            iq_standard_q=(0.0, 1.0),
            reset_uniform=0.2,
            reset_ack_uniform=0.2,
            drift_standard=(0.0,) * 5,
            seed=0,
            round_index=0,
        )


def test_diagnostic_actions_are_actual_semantic_action_words():
    expected = {
        "IDLE": NominalAction.IDLE,
        "X": NominalAction.X,
        "Z": NominalAction.Z,
        "XZ": NominalAction.XZ,
        "RESET": NominalAction.RESET,
        "HOLD": NominalAction.HOLD,
        "LKG_HOLD": NominalAction.LKG_HOLD,
    }
    for name, code in expected.items():
        word = diagnostic_action_word(name)
        assert isinstance(word, ActionWord)
        assert NominalAction(word.action_code) == code
        assert ActionWord.unpack(word.pack()) == word


def test_step_api_has_no_truth_or_future_decision_input():
    parameters = inspect.signature(Phase9BackendASimulator.step).parameters
    assert tuple(parameters) == (
        "self",
        "state",
        "action",
        "exogenous",
        "evaluator",
    )
    assert "logical_error" not in parameters
    assert "hidden_level" not in parameters
    assert "future" not in parameters


def test_joint_state_rejects_nonpositive_density(compact_config):
    dimension = compact_config.cutoff * 3
    invalid = np.eye(dimension, dtype=np.complex128) / dimension
    invalid[0, 0] = -0.1
    invalid[1, 1] += 0.1 + 1.0 / dimension
    with pytest.raises(ValueError, match="positive semidefinite"):
        BackendAState(invalid, compact_config.cutoff)


def test_partial_traces_are_normalized(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    state = simulator.initialize_fock(ancilla_state="e")
    oscillator = simulator.oscillator_density(state.joint_density)
    ancilla = simulator.ancilla_density(state.joint_density)
    assert np.trace(oscillator.matrix) == pytest.approx(1.0)
    assert np.trace(ancilla) == pytest.approx(1.0)
    assert simulator.level_probabilities(state.joint_density) == pytest.approx(
        (0.0, 1.0, 0.0)
    )


def test_actual_gksl_channel_is_cp_tp_and_hermitian(compact_config):
    small = Phase9BackendASimulator(
        replace(compact_config, cutoff=2, logical_grid_points=1025)
    )
    hamiltonian = small._base_hamiltonian(BackendADriftState())
    diagnostics = small.channel_diagnostics(hamiltonian, 0.07)
    assert diagnostics.cp
    assert diagnostics.tp
    assert diagnostics.choi_minimum_eigenvalue >= -2.0e-10
    assert diagnostics.hermiticity_frobenius <= 2.0e-10
    assert diagnostics.choi_trace == pytest.approx(diagnostics.dimension)


def test_nonhermitian_hamiltonian_is_rejected(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    invalid = np.zeros(
        (simulator.dimension, simulator.dimension),
        dtype=np.complex128,
    )
    invalid[0, 1] = 1.0
    with pytest.raises(ValueError, match="Hermitian"):
        simulator.liouvillian(invalid)


def test_non_trace_preserving_generator_cannot_be_normalized_away(
    compact_config,
    monkeypatch,
):
    simulator = Phase9BackendASimulator(compact_config)
    state = simulator.initialize_fock()
    dimension_squared = simulator.dimension * simulator.dimension
    monkeypatch.setattr(
        simulator,
        "liouvillian",
        lambda _hamiltonian: sparse_eye(
            dimension_squared,
            dtype=np.complex128,
            format="csr",
        ),
    )
    with pytest.raises(RuntimeError, match="trace preservation"):
        simulator._evolve_segment(
            state.joint_density,
            0.1,
            lambda _fraction: np.zeros(
                (simulator.dimension, simulator.dimension),
                dtype=np.complex128,
            ),
        )


def test_measurement_and_reset_instruments_are_complete(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    assert simulator.measurement_completeness_error() <= 1.0e-12
    assert simulator.reset_completeness_error() <= 1.0e-12


def test_zero_noise_idle_is_identity(compact_config):
    config = noise_free(compact_config, substeps_per_segment=32)
    simulator = Phase9BackendASimulator(config)
    initial = simulator.initialize_fock()
    result = simulator.step(
        initial,
        diagnostic_action_word("IDLE"),
        zero_exogenous(config),
    )
    assert trace_distance(initial.joint_density, result.state.joint_density) <= 1.0e-12


def test_ideal_action_matches_explicit_displacement(compact_config):
    config = noise_free(compact_config, substeps_per_segment=64)
    simulator = Phase9BackendASimulator(config)
    initial = simulator.initialize_fock()
    action = diagnostic_action_word("X")
    result = simulator.step(initial, action, zero_exogenous(config))
    alpha = simulator._action_alpha(action)
    displacement = simulator.oscillator.displacement_operator(alpha)
    expected_oscillator = (
        displacement
        @ simulator.oscillator_density(initial.joint_density).matrix
        @ displacement.conj().T
    )
    expected = np.kron(expected_oscillator, simulator.level_projectors[0])
    assert trace_distance(expected, result.state.joint_density) <= 2.0e-5


def test_large_reset_limit_maps_f_to_g(compact_config):
    config = noise_free(
        compact_config,
        reset_success_e=1.0,
        reset_success_f=1.0,
    )
    simulator = Phase9BackendASimulator(config)
    initial = simulator.initialize_fock(ancilla_state="f")
    result = simulator.step(
        initial,
        diagnostic_action_word("RESET"),
        zero_exogenous(config),
    )
    assert simulator.level_probabilities(result.state.joint_density) == pytest.approx(
        (1.0, 0.0, 0.0),
        abs=1.0e-12,
    )
    assert result.truth.reset_hidden_outcome == "success"
    assert result.observation.reset_ack == "success"


def test_reset_failure_and_no_reset_preserve_f(compact_config):
    config = noise_free(
        compact_config,
        reset_success_e=0.0,
        reset_success_f=0.0,
    )
    simulator = Phase9BackendASimulator(config)
    initial = simulator.initialize_fock(ancilla_state="f")
    failed = simulator.step(
        initial,
        diagnostic_action_word("RESET"),
        zero_exogenous(config),
    )
    persistent = simulator.step(
        initial,
        diagnostic_action_word("IDLE"),
        zero_exogenous(config),
    )
    assert simulator.level_probabilities(failed.state.joint_density)[2] == pytest.approx(1.0)
    assert simulator.level_probabilities(persistent.state.joint_density)[2] == pytest.approx(1.0)
    assert failed.truth.reset_hidden_outcome == "failure"
    assert failed.observation.reset_ack == "failure"
    assert persistent.state.leakage_age == 1


def test_reset_ack_error_changes_observation_not_hidden_physics(compact_config):
    config = noise_free(
        compact_config,
        reset_success_e=1.0,
        reset_success_f=1.0,
        reset_ack_error=1.0,
    )
    simulator = Phase9BackendASimulator(config)
    initial = simulator.initialize_fock(ancilla_state="f")
    result = simulator.step(
        initial,
        diagnostic_action_word("RESET"),
        zero_exogenous(config),
    )
    assert result.truth.reset_hidden_outcome == "success"
    assert result.observation.reset_ack == "failure"
    assert simulator.level_probabilities(result.state.joint_density)[0] == pytest.approx(1.0)


def test_iq_likelihood_performs_real_measurement_backaction(compact_config):
    config = noise_free(
        compact_config,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
    )
    simulator = Phase9BackendASimulator(config)
    ge_plus = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / sqrt(2.0)
    initial = simulator.initialize_fock(ancilla_state=ge_plus)
    before = abs(simulator.ancilla_density(initial.joint_density)[0, 1])
    result = simulator.step(
        initial,
        diagnostic_action_word("IDLE"),
        zero_exogenous(config),
    )
    after = abs(simulator.ancilla_density(result.state.joint_density)[0, 1])
    assert before == pytest.approx(0.5)
    assert after < 1.0e-10
    assert max(result.observation.posterior_levels) > 0.999999
    assert result.truth.post_measurement_level_probabilities == pytest.approx(
        result.observation.posterior_levels
    )


def test_ramsey_measurement_is_oscillator_state_dependent(compact_config):
    config = noise_free(
        compact_config,
        ramsey_angle=np.pi / 2.0,
        ramsey_pulse_duration=0.03,
        sense_duration=0.8,
        dispersive_chi=1.0,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
        substeps_per_segment=16,
    )
    simulator = Phase9BackendASimulator(config)
    ket_zero = np.zeros(config.cutoff, dtype=np.complex128)
    ket_one = np.zeros(config.cutoff, dtype=np.complex128)
    ket_zero[0] = 1.0
    ket_one[1] = 1.0
    exogenous = zero_exogenous(config)
    zero = simulator.step(
        simulator.initialize_fock(oscillator_ket=ket_zero),
        diagnostic_action_word("IDLE"),
        exogenous,
    )
    one = simulator.step(
        simulator.initialize_fock(oscillator_ket=ket_one),
        diagnostic_action_word("IDLE"),
        exogenous,
    )
    level_tv = 0.5 * np.sum(
        np.abs(
            np.asarray(zero.truth.pre_measurement_level_probabilities)
            - np.asarray(one.truth.pre_measurement_level_probabilities)
        )
    )
    assert level_tv > 0.05


def test_syndrome_measurement_backacts_on_oscillator(compact_config):
    config = noise_free(
        compact_config,
        ramsey_angle=np.pi / 2.0,
        ramsey_pulse_duration=0.03,
        sense_duration=0.8,
        dispersive_chi=1.0,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
        substeps_per_segment=16,
    )
    simulator = Phase9BackendASimulator(config)
    ket = np.zeros(config.cutoff, dtype=np.complex128)
    ket[0] = ket[1] = 1.0 / sqrt(2.0)
    initial = simulator.initialize_fock(oscillator_ket=ket)
    result = simulator.step(
        initial,
        diagnostic_action_word("IDLE"),
        zero_exogenous(config),
    )
    before = simulator.oscillator_density(initial.joint_density).matrix
    after = simulator.oscillator_density(result.state.joint_density).matrix
    assert trace_distance(before, after) > 0.01


def test_action_coupling_creates_physical_f_population(compact_config):
    config = noise_free(
        compact_config,
        action_leakage_coupling=0.8,
        action_duration=0.1,
        substeps_per_segment=32,
    )
    simulator = Phase9BackendASimulator(config)
    initial = simulator.initialize_fock(ancilla_state="e")
    exogenous = zero_exogenous(config)
    idle = simulator.step(
        initial,
        diagnostic_action_word("IDLE"),
        exogenous,
    )
    acted = simulator.step(
        initial,
        diagnostic_action_word("X"),
        exogenous,
    )
    f_idle = simulator.level_probabilities(idle.state.joint_density)[2]
    f_acted = simulator.level_probabilities(acted.state.joint_density)[2]
    assert f_acted - f_idle > 0.1


def test_f_iq_record_produces_observed_leakage_confidence(compact_config):
    config = noise_free(
        compact_config,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
    )
    simulator = Phase9BackendASimulator(config)
    initial = simulator.initialize_fock(ancilla_state="f")
    result = simulator.step(
        initial,
        diagnostic_action_word("IDLE"),
        zero_exogenous(config),
    )
    assert result.truth.sampled_emission_level == "f"
    assert result.observation.leakage_confidence_analog > 0.999999
    assert result.state.leakage_age == 1


def test_analog_backend_observation_cannot_be_relabelled_live(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    result = simulator.step(
        simulator.initialize_fock(),
        diagnostic_action_word("IDLE"),
        backend_a_exogenous(seed=2, round_index=0, iq_samples=4),
    )
    assert result.observation.source == "synthetic_backend_a_analog_pre_frontend"
    with pytest.raises(ValueError, match="cannot relabel"):
        BackendAObservation(
            iq_i=result.observation.iq_i,
            iq_q=result.observation.iq_q,
            integrated_i=result.observation.integrated_i,
            integrated_q=result.observation.integrated_q,
            log_evidence_density=result.observation.log_evidence_density,
            posterior_levels=result.observation.posterior_levels,
            leakage_confidence_analog=result.observation.leakage_confidence_analog,
            reset_ack=result.observation.reset_ack,
            source="live",
        )


def test_evaluator_truth_cannot_change_physical_transition(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    initial, evaluator_zero = simulator.initialize_logical("0")
    evaluator_one = BackendAEvaluatorState(
        target_label="1",
        target_density=np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    )
    exogenous = backend_a_exogenous(seed=88, round_index=0, iq_samples=4)
    action = diagnostic_action_word("X")
    zero_truth = simulator.step(
        initial,
        action,
        exogenous,
        evaluator=evaluator_zero,
    )
    one_truth = simulator.step(
        initial,
        action,
        exogenous,
        evaluator=evaluator_one,
    )
    no_truth = simulator.step(initial, action, exogenous)
    assert np.array_equal(
        zero_truth.state.joint_density,
        one_truth.state.joint_density,
    )
    assert np.array_equal(
        zero_truth.state.joint_density,
        no_truth.state.joint_density,
    )
    assert np.array_equal(zero_truth.observation.iq_i, one_truth.observation.iq_i)
    assert zero_truth.logical is not None and one_truth.logical is not None
    assert zero_truth.logical.target_fidelity != one_truth.logical.target_fidelity


def test_same_exogenous_different_action_changes_density_and_drift(compact_config):
    config = replace(
        compact_config,
        ramsey_angle=0.0,
        sense_duration=0.0,
        iq_centers=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    simulator = Phase9BackendASimulator(config)
    initial = simulator.initialize_fock()
    exogenous = backend_a_exogenous(seed=31, round_index=0, iq_samples=4)
    idle = simulator.step(
        initial,
        diagnostic_action_word("IDLE"),
        exogenous,
    )
    acted = simulator.step(
        initial,
        diagnostic_action_word("X"),
        exogenous,
    )
    assert idle.exogenous == acted.exogenous
    assert trace_distance(
        idle.state.joint_density,
        acted.state.joint_density,
    ) > 1.0e-3
    assert np.linalg.norm(
        idle.state.drift.vector() - acted.state.drift.vector()
    ) > 1.0e-5


def test_action_conditioned_drift_is_exact_under_zero_noise(compact_config):
    config = noise_free(
        compact_config,
        drift_action_kick=0.01,
        drift_readout_heating=0.02,
        drift_leakage_heating=0.03,
    )
    simulator = Phase9BackendASimulator(config)
    state = simulator.initialize_fock()
    result = simulator.step(
        state,
        diagnostic_action_word("XZ"),
        zero_exogenous(config),
    )
    assert result.state.drift.vector() == pytest.approx(
        (0.01, 0.01, 0.04, -0.02, 0.06)
    )


def test_wrong_round_exogenous_is_rejected(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    state = simulator.initialize_fock()
    with pytest.raises(ValueError, match="round index"):
        simulator.step(
            state,
            diagnostic_action_word("IDLE"),
            backend_a_exogenous(seed=1, round_index=1, iq_samples=4),
        )


def test_seed_replay_is_bit_exact_and_other_seed_changes_iq(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    initial = simulator.initialize_fock()
    actions = (
        diagnostic_action_word("IDLE"),
        diagnostic_action_word("X"),
        diagnostic_action_word("IDLE"),
    )
    first = simulator.simulate(initial, actions, seed=900)
    replay = simulator.simulate(initial, actions, seed=900)
    other = simulator.simulate(initial, actions, seed=901)
    assert np.array_equal(
        first.final_state.joint_density,
        replay.final_state.joint_density,
    )
    for left, right in zip(first.rounds, replay.rounds):
        assert np.array_equal(left.observation.iq_i, right.observation.iq_i)
        assert np.array_equal(left.observation.iq_q, right.observation.iq_q)
    assert not np.array_equal(
        first.rounds[0].observation.iq_i,
        other.rounds[0].observation.iq_i,
    )


def test_future_randomness_does_not_change_prefix(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    initial = simulator.initialize_fock()
    first_action = diagnostic_action_word("IDLE")
    one_round = simulator.simulate(initial, (first_action,), seed=122)
    longer = simulator.simulate(
        initial,
        (
            first_action,
            diagnostic_action_word("X"),
            diagnostic_action_word("Z"),
        ),
        seed=122,
    )
    assert np.array_equal(
        one_round.rounds[0].state.joint_density,
        longer.rounds[0].state.joint_density,
    )
    assert np.array_equal(
        one_round.rounds[0].observation.iq_i,
        longer.rounds[0].observation.iq_i,
    )


def test_six_state_logical_projection_starts_at_unit_fidelity(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    for label in ("0", "1", "+", "-", "+i", "-i"):
        state, evaluator = simulator.initialize_logical(label)
        record = simulator.logical_record(state, evaluator)
        assert record.target_fidelity >= 1.0 - 2.0e-9
        assert record.code_survival_probability >= 1.0 - 2.0e-9
        assert not record.logical_error


def test_evaluator_frame_advances_from_semantic_action(compact_config):
    simulator = Phase9BackendASimulator(compact_config)
    state, evaluator = simulator.initialize_logical("0")
    result = simulator.step(
        state,
        diagnostic_action_word("XZ"),
        backend_a_exogenous(seed=3, round_index=0, iq_samples=4),
        evaluator=evaluator,
    )
    assert result.logical is not None
    assert result.logical.evaluator_state.pauli_x == 1
    assert result.logical.evaluator_state.pauli_z == 1


def test_qualification_passes_every_registered_gate(qualification):
    assert qualification.verdict == "QUALIFIED_BACKEND_A_ONLY"
    assert qualification.passed
    assert len(qualification.checks) >= 24
    assert all(qualification.checks.values())


def test_qualification_has_converged_step_and_cutoff_metrics(qualification):
    metrics = qualification.metrics
    assert metrics["step_size_16_vs_32_trace_distance"] <= 1.5e-4
    assert metrics["step_size_error_ratio"] <= 0.30
    assert metrics["fock_cutoff_8_vs_12_trace_distance"] <= 2.0e-4
    assert metrics["ideal_action_trace_distance"] <= 2.0e-5


def test_qualification_claim_outputs_remain_typed_null(qualification):
    required = {
        "backend_b_qualified",
        "dual_backend_agreement",
        "round_ler",
        "six_state_lifetime",
        "physical_break_even",
        "official_puviani_exact",
        "puviani_nmf_surpass",
        "external_sota",
        "hardware_measured",
        "rank",
    }
    assert set(qualification.claim_state) == required
    assert all(value is None for value in qualification.claim_state.values())


def test_qualification_detects_mutated_nonphysical_gate(qualification):
    mutated = dict(qualification.checks)
    mutated["gksl_channel_cp"] = False
    assert not all(mutated.values())


def test_stricter_prefrozen_threshold_causes_real_no_go(compact_config):
    thresholds = replace(
        BackendAQualificationThresholds(),
        step_size_error_ratio=0.1,
    )
    result = run_backend_a_qualification(compact_config, thresholds)
    assert not result.checks["step_size_convergence"]
    assert result.verdict == "NO_GO_BACKEND_A_QUALIFICATION"
    assert not result.passed
