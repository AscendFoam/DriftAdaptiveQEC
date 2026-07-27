from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from math import exp, sqrt
from pathlib import Path

import numpy as np
import pytest

from physics.phase9_backend_a import backend_a_exogenous
from physics.phase9_backend_b import (
    BACKEND_B_ID,
    BACKEND_B_LIKELIHOOD_ID,
    BACKEND_B_LOGICAL_ID,
    BACKEND_B_RNG_ID,
    BACKEND_B_SCOPE,
    BACKEND_B_SOLVER_ID,
    MAX_EXACT_CHOI_CUTOFF,
    MAX_SUPPORTED_CUTOFF,
    BackendBConfig,
    BackendBDrift,
    BackendBEvaluator,
    BackendBObservation,
    BackendBQualification,
    BackendBQualificationThresholds,
    BackendBRandomRecord,
    BackendBState,
    Phase9BackendBSimulator,
    backend_b_random_record,
    diagnostic_action_word_b,
    run_backend_b_qualification,
)
from physics.phase9_twin_contract import ActionWord


@pytest.fixture(scope="module")
def qualification() -> BackendBQualification:
    return run_backend_b_qualification()


@pytest.fixture()
def compact_config() -> BackendBConfig:
    return BackendBConfig(
        cutoff=8,
        split_steps_per_segment=2,
        iq_samples=4,
    )


def noise_free(config: BackendBConfig, **overrides: object) -> BackendBConfig:
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


def zero_record(
    config: BackendBConfig,
    *,
    round_index: int = 0,
    component_uniform: float = 0.25,
    reset_uniform: float = 0.25,
    ack_uniform: float = 0.25,
) -> BackendBRandomRecord:
    return BackendBRandomRecord(
        component_uniform=component_uniform,
        iq_normal_i=(0.0,) * config.iq_samples,
        iq_normal_q=(0.0,) * config.iq_samples,
        reset_uniform=reset_uniform,
        ack_uniform=ack_uniform,
        drift_normal=(0.0,) * 5,
        seed=7,
        round_index=round_index,
    )


def trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = 0.5 * ((left - right) + (left - right).conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))


def test_backend_b_identity_scope_and_semantic_hash(compact_config):
    assert compact_config.backend_id == BACKEND_B_ID
    assert compact_config.scope == BACKEND_B_SCOPE
    assert len(compact_config.semantic_sha256()) == 64
    assert compact_config.semantic_sha256() == BackendBConfig(
        cutoff=8,
        split_steps_per_segment=2,
        iq_samples=4,
    ).semantic_sha256()
    assert len(
        {
            BACKEND_B_SOLVER_ID,
            BACKEND_B_RNG_ID,
            BACKEND_B_LIKELIHOOD_ID,
            BACKEND_B_LOGICAL_ID,
        }
    ) == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cutoff", 1),
        ("split_steps_per_segment", 0),
        ("iq_samples", 0),
        ("iq_sigma", 0.0),
        ("reset_success_f", 1.1),
        ("action_duration", 0.0),
        ("ramsey_pulse_duration", 0.0),
        ("comb_half_width", 0),
    ],
)
def test_invalid_config_is_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        BackendBConfig(**{field: value})


def test_high_cutoff_contract_accepts_28_and_caps_at_32():
    assert MAX_SUPPORTED_CUTOFF == 32
    assert BackendBConfig(cutoff=28).cutoff == 28
    assert BackendBConfig(cutoff=MAX_SUPPORTED_CUTOFF).cutoff == 32
    dimension = 3 * MAX_SUPPORTED_CUTOFF
    state = BackendBState(
        np.eye(dimension, dtype=np.complex128) / dimension,
        MAX_SUPPORTED_CUTOFF,
    )
    assert state.cutoff == MAX_SUPPORTED_CUTOFF
    with pytest.raises(ValueError, match=r"\[2,32\]"):
        BackendBConfig(cutoff=MAX_SUPPORTED_CUTOFF + 1)
    with pytest.raises(ValueError, match=r"\[2,32\]"):
        BackendBState(
            np.eye(dimension + 3, dtype=np.complex128) / (dimension + 3),
            MAX_SUPPORTED_CUTOFF + 1,
        )


def test_cutoff_28_executes_full_backend_b_step():
    config = BackendBConfig(
        cutoff=28,
        split_steps_per_segment=1,
        iq_samples=1,
    )
    simulator = Phase9BackendBSimulator(config)
    initial = simulator.initialize_fock()
    result = simulator.step(
        initial,
        diagnostic_action_word_b("IDLE"),
        zero_record(config),
    )
    assert result.state.cutoff == 28
    assert result.state.joint_density.shape == (84, 84)
    assert np.isclose(np.trace(result.state.joint_density), 1.0, atol=1.0e-8)
    assert np.all(np.isfinite(result.observation.iq_i))
    assert np.all(np.isfinite(result.observation.iq_q))


def test_backend_identity_and_scope_cannot_be_relabelled():
    with pytest.raises(ValueError, match="immutable"):
        BackendBConfig(backend_id="device")
    with pytest.raises(ValueError, match="immutable"):
        BackendBConfig(scope="hardware measured")


def test_runtime_import_graph_excludes_backend_a_and_shared_physics_kernels():
    path = Path(__file__).parents[1] / "physics" / "phase9_backend_b.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden = {
        "physics.phase9_backend_a",
        "phase9_backend_a",
        "physics.fock_density_model",
        "fock_density_model",
        "physics.fock_sbs_cycle",
        "fock_sbs_cycle",
        "physics.finite_energy_gkp",
        "finite_energy_gkp",
    }
    assert imported.isdisjoint(forbidden)
    assert "default_rng" not in source
    assert "np.random" not in source


def test_random_record_is_addressable_replayable_and_not_backend_a_stream(
    compact_config,
):
    first = backend_b_random_record(seed=19, round_index=3, iq_samples=4)
    replay = backend_b_random_record(seed=19, round_index=3, iq_samples=4)
    later = backend_b_random_record(seed=19, round_index=4, iq_samples=4)
    backend_a = backend_a_exogenous(seed=19, round_index=3, iq_samples=4)
    assert first == replay
    assert first != later
    assert first.rng_id == BACKEND_B_RNG_ID
    assert first.iq_normal_i != backend_a.iq_standard_i


def test_step_api_has_no_truth_hidden_label_or_future_input():
    parameters = inspect.signature(Phase9BackendBSimulator.step).parameters
    assert tuple(parameters) == (
        "self",
        "state",
        "action",
        "random_record",
        "evaluator",
    )
    assert all(
        token not in parameters
        for token in ("truth", "logical_error", "hidden_level", "future")
    )


def test_density_state_rejects_nonpositive_or_nonunit_trace(compact_config):
    dimension = compact_config.cutoff * 3
    invalid = np.eye(dimension, dtype=np.complex128) / dimension
    invalid[0, 0] = -0.1
    invalid[1, 1] += 0.1 + 1.0 / dimension
    with pytest.raises(ValueError, match="positive semidefinite"):
        BackendBState(invalid, compact_config.cutoff)
    with pytest.raises(ValueError, match="unit trace"):
        BackendBState(np.eye(dimension), compact_config.cutoff)


def test_initializers_reject_zero_norm_kets(compact_config):
    simulator = Phase9BackendBSimulator(compact_config)
    with pytest.raises(ValueError, match="nonzero norm"):
        simulator.initialize_fock(
            oscillator_ket=np.zeros(
                compact_config.cutoff,
                dtype=np.complex128,
            )
        )
    with pytest.raises(ValueError, match="nonzero norm"):
        simulator.initialize_fock(
            ancilla_state=np.zeros(3, dtype=np.complex128)
        )


def test_evaluator_rejects_label_density_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        BackendBEvaluator(
            target_label="0",
            target_density=np.array(
                [[0.0, 0.0], [0.0, 1.0]],
                dtype=np.complex128,
            ),
        )


def test_split_channel_is_cp_tp_and_hermitian():
    simulator = Phase9BackendBSimulator(
        BackendBConfig(cutoff=2, split_steps_per_segment=2)
    )
    minimum, tp_error, hermiticity = simulator.split_channel_choi(
        simulator._base_hamiltonian(BackendBDrift()),
        0.07,
    )
    assert minimum >= -3.0e-9
    assert tp_error <= 3.0e-9
    assert hermiticity <= 3.0e-9


def test_high_cutoff_exact_choi_is_rejected_before_dense_expm(monkeypatch):
    simulator = Phase9BackendBSimulator(
        BackendBConfig(
            cutoff=MAX_EXACT_CHOI_CUTOFF + 1,
            split_steps_per_segment=1,
            iq_samples=1,
        )
    )
    monkeypatch.setattr(
        "physics.phase9_backend_b.expm",
        lambda _matrix: pytest.fail("large dense expm must not be allocated"),
    )
    with pytest.raises(RuntimeError, match="exact Choi construction"):
        simulator.split_channel_choi(
            np.zeros(
                (simulator.dimension, simulator.dimension),
                dtype=np.complex128,
            ),
            0.01,
        )


def test_nonhermitian_hamiltonian_is_rejected(compact_config):
    simulator = Phase9BackendBSimulator(compact_config)
    invalid = np.zeros(
        (simulator.dimension, simulator.dimension),
        dtype=np.complex128,
    )
    invalid[0, 1] = 1.0
    state = simulator.initialize_fock()
    with pytest.raises(ValueError, match="Hermitian"):
        simulator._split_segment(
            state.joint_density,
            0.1,
            lambda _fraction: invalid,
        )


def test_non_trace_preserving_channel_cannot_be_normalized_away(
    compact_config,
    monkeypatch,
):
    simulator = Phase9BackendBSimulator(compact_config)
    state = simulator.initialize_fock()
    monkeypatch.setattr(
        simulator,
        "_noise_channels",
        lambda density, _duration: 2.0 * density,
    )
    with pytest.raises(RuntimeError, match="trace preservation"):
        simulator._split_segment(
            state.joint_density,
            0.1,
            lambda _fraction: np.zeros(
                (simulator.dimension, simulator.dimension),
                dtype=np.complex128,
            ),
        )


def test_analytic_kraus_channels_are_complete(compact_config):
    simulator = Phase9BackendBSimulator(compact_config)
    errors = simulator.channel_completeness_errors(0.17)
    assert set(errors) == {
        "pure_loss",
        "ge_relax",
        "fe_relax",
        "ge_excite",
    }
    assert max(errors.values()) <= 1.0e-12
    assert simulator.measurement_completeness_error() <= 1.0e-12
    assert simulator.reset_completeness_error() <= 1.0e-12


def test_pure_loss_matches_closed_form_number_decay(compact_config):
    rate = 0.37
    duration = 0.19
    config = noise_free(
        compact_config,
        oscillator_loss_rate=rate,
        action_duration=duration,
    )
    simulator = Phase9BackendBSimulator(config)
    ket = np.zeros(config.cutoff, dtype=np.complex128)
    ket[3] = 1.0
    state = simulator.initialize_fock(oscillator_ket=ket)
    output = simulator._noise_channels(state.joint_density, duration)
    oscillator = simulator.oscillator_density(output)
    mean = float(np.trace(oscillator @ simulator.number).real)
    assert mean == pytest.approx(3.0 * exp(-rate * duration), abs=2.0e-10)


def test_qutrit_relaxation_matches_closed_form_population(compact_config):
    rate = 0.43
    duration = 0.23
    config = noise_free(
        compact_config,
        ancilla_ge_relax_rate=rate,
        action_duration=duration,
    )
    simulator = Phase9BackendBSimulator(config)
    state = simulator.initialize_fock(ancilla_state="e")
    output = simulator._noise_channels(state.joint_density, duration)
    levels = simulator.level_probabilities(output)
    expected_e = exp(-rate * duration)
    assert levels == pytest.approx((1.0 - expected_e, expected_e, 0.0))


def test_zero_noise_idle_is_identity(compact_config):
    config = noise_free(compact_config, split_steps_per_segment=16)
    simulator = Phase9BackendBSimulator(config)
    initial = simulator.initialize_fock()
    result = simulator.step(
        initial,
        diagnostic_action_word_b("IDLE"),
        zero_record(config),
    )
    assert trace_distance(initial.joint_density, result.state.joint_density) <= 1.0e-12


def test_large_reset_limit_and_failed_reset_have_physical_branches(
    compact_config,
):
    success_config = noise_free(
        compact_config,
        reset_success_e=1.0,
        reset_success_f=1.0,
    )
    success_sim = Phase9BackendBSimulator(success_config)
    success = success_sim.step(
        success_sim.initialize_fock(ancilla_state="f"),
        diagnostic_action_word_b("RESET"),
        zero_record(success_config),
    )
    assert success_sim.level_probabilities(
        success.state.joint_density
    ) == pytest.approx((1.0, 0.0, 0.0))
    assert success.truth.reset_hidden_outcome == "success"

    fail_config = noise_free(
        compact_config,
        reset_success_e=0.0,
        reset_success_f=0.0,
    )
    fail_sim = Phase9BackendBSimulator(fail_config)
    failed = fail_sim.step(
        fail_sim.initialize_fock(ancilla_state="f"),
        diagnostic_action_word_b("RESET"),
        zero_record(fail_config),
    )
    assert fail_sim.level_probabilities(
        failed.state.joint_density
    )[2] == pytest.approx(1.0)
    assert failed.truth.reset_hidden_outcome == "failure"


def test_reset_ack_error_changes_observation_not_hidden_physics(compact_config):
    config = noise_free(
        compact_config,
        reset_success_e=1.0,
        reset_success_f=1.0,
        reset_ack_error=1.0,
    )
    simulator = Phase9BackendBSimulator(config)
    result = simulator.step(
        simulator.initialize_fock(ancilla_state="f"),
        diagnostic_action_word_b("RESET"),
        zero_record(config),
    )
    assert result.truth.reset_hidden_outcome == "success"
    assert result.observation.reset_ack == "failure"
    assert simulator.level_probabilities(
        result.state.joint_density
    )[0] == pytest.approx(1.0)


def test_iq_likelihood_causes_real_backaction(compact_config):
    config = noise_free(
        compact_config,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
    )
    simulator = Phase9BackendBSimulator(config)
    ge_plus = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / sqrt(2.0)
    initial = simulator.initialize_fock(ancilla_state=ge_plus)
    before = abs(simulator.ancilla_density(initial.joint_density)[0, 1])
    result = simulator.step(
        initial,
        diagnostic_action_word_b("IDLE"),
        zero_record(config, component_uniform=0.2),
    )
    after = abs(simulator.ancilla_density(result.state.joint_density)[0, 1])
    assert before == pytest.approx(0.5)
    assert after < 1.0e-10
    assert max(result.observation.posterior_levels) > 0.999999


def test_analog_observation_cannot_be_relabelled_live(compact_config):
    simulator = Phase9BackendBSimulator(compact_config)
    result = simulator.step(
        simulator.initialize_fock(),
        diagnostic_action_word_b("IDLE"),
        backend_b_random_record(seed=2, round_index=0, iq_samples=4),
    )
    with pytest.raises(ValueError, match="cannot relabel"):
        BackendBObservation(
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


def test_evaluator_truth_cannot_change_transition_or_iq(compact_config):
    simulator = Phase9BackendBSimulator(compact_config)
    initial, evaluator_zero = simulator.initialize_logical("0")
    evaluator_one = BackendBEvaluator(
        target_label="1",
        target_density=np.array(
            [[0.0, 0.0], [0.0, 1.0]],
            dtype=np.complex128,
        ),
    )
    random_record = backend_b_random_record(
        seed=88,
        round_index=0,
        iq_samples=4,
    )
    action = diagnostic_action_word_b("X")
    zero = simulator.step(
        initial,
        action,
        random_record,
        evaluator=evaluator_zero,
    )
    one = simulator.step(
        initial,
        action,
        random_record,
        evaluator=evaluator_one,
    )
    absent = simulator.step(initial, action, random_record)
    assert np.array_equal(zero.state.joint_density, one.state.joint_density)
    assert np.array_equal(zero.state.joint_density, absent.state.joint_density)
    assert np.array_equal(zero.observation.iq_i, one.observation.iq_i)
    assert zero.logical is not None and one.logical is not None
    assert zero.logical.target_fidelity != one.logical.target_fidelity


def test_action_changes_physical_density_drift_and_f_population(compact_config):
    config = noise_free(
        compact_config,
        action_leakage_coupling=0.8,
        action_duration=0.1,
        drift_action_kick=0.01,
        split_steps_per_segment=32,
    )
    simulator = Phase9BackendBSimulator(config)
    initial = simulator.initialize_fock(ancilla_state="e")
    record = zero_record(config)
    idle = simulator.step(
        initial,
        diagnostic_action_word_b("IDLE"),
        record,
    )
    acted = simulator.step(
        initial,
        diagnostic_action_word_b("X"),
        record,
    )
    assert trace_distance(
        idle.state.joint_density,
        acted.state.joint_density,
    ) > 1.0e-3
    assert simulator.level_probabilities(
        acted.state.joint_density
    )[2] - simulator.level_probabilities(
        idle.state.joint_density
    )[2] > 0.1
    assert np.linalg.norm(
        acted.state.drift.vector() - idle.state.drift.vector()
    ) > 1.0e-5


def test_wrong_round_record_is_rejected(compact_config):
    simulator = Phase9BackendBSimulator(compact_config)
    with pytest.raises(ValueError, match="round mismatch"):
        simulator.step(
            simulator.initialize_fock(),
            diagnostic_action_word_b("IDLE"),
            backend_b_random_record(seed=1, round_index=1, iq_samples=4),
        )


def test_seed_replay_and_future_prefix_are_bit_exact(compact_config):
    simulator = Phase9BackendBSimulator(compact_config)
    initial = simulator.initialize_fock()
    idle = diagnostic_action_word_b("IDLE")
    actions = (idle, diagnostic_action_word_b("X"), idle)
    first = simulator.simulate(initial, actions, seed=900)
    replay = simulator.simulate(initial, actions, seed=900)
    other = simulator.simulate(initial, actions, seed=901)
    prefix = simulator.simulate(initial, (idle,), seed=900)
    assert np.array_equal(
        first.final_state.joint_density,
        replay.final_state.joint_density,
    )
    assert np.array_equal(
        first.rounds[0].observation.iq_i,
        prefix.rounds[0].observation.iq_i,
    )
    assert not np.array_equal(
        first.rounds[0].observation.iq_i,
        other.rounds[0].observation.iq_i,
    )


def test_six_state_projection_is_evaluator_only_and_initially_exact(
    compact_config,
):
    simulator = Phase9BackendBSimulator(compact_config)
    for label in ("0", "1", "+", "-", "+i", "-i"):
        state, evaluator = simulator.initialize_logical(label)
        logical = simulator.logical_record(state, evaluator)
        assert logical.target_fidelity >= 1.0 - 2.0e-9
        assert logical.code_survival >= 1.0 - 2.0e-9
        assert not logical.logical_error


def test_qualification_passes_all_real_checks_and_keeps_claims_null(
    qualification,
):
    assert qualification.verdict == "QUALIFIED_BACKEND_B_ONLY"
    assert qualification.passed
    assert len(qualification.checks) >= 28
    assert all(qualification.checks.values())
    assert all(value is None for value in qualification.claim_state.values())
    assert qualification.metrics["split_error_ratio"] <= 0.35
    assert (
        qualification.metrics["analytic_loss_mean_error"] <= 2.0e-10
    )


def test_qualification_mappings_are_immutable(qualification):
    with pytest.raises(TypeError):
        qualification.checks["split_convergence"] = False
    with pytest.raises(TypeError):
        qualification.claim_state["external_sota"] = True


def test_stricter_prefrozen_threshold_causes_real_no_go(compact_config):
    thresholds = replace(
        BackendBQualificationThresholds(),
        split_ratio=0.10,
    )
    result = run_backend_b_qualification(compact_config, thresholds)
    assert not result.checks["split_step_convergence"]
    assert result.verdict == "NO_GO_BACKEND_B_QUALIFICATION"
    assert not result.passed
