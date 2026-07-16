from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from physics.fock_density_model import FiniteCutoffDensity, FiniteCutoffFockModel
from physics.fock_sbs_cycle import (
    FOCK_SBS_CYCLE_SCOPE,
    PAPER_CANONICAL_SOURCE_SCALE,
    FockIdleConfig,
    SBSFockCycleConfig,
    SBSFockOneRoundSimulator,
    logical_density,
    run_sbs_fock_cycle_validation,
    write_sbs_fock_cycle_validation,
)
from physics.sbs_error_space import PauliFrame, SBS_PROTOCOL_ID


@pytest.fixture(scope="module")
def clean() -> SBSFockOneRoundSimulator:
    return SBSFockOneRoundSimulator(
        SBSFockCycleConfig(
            cutoff=18,
            projector_delta=0.34,
            grid_points=2049,
            readout_confusion=((1.0, 0.0), (0.0, 1.0)),
        )
    )


@pytest.fixture(scope="module")
def observed() -> SBSFockOneRoundSimulator:
    return SBSFockOneRoundSimulator(
        SBSFockCycleConfig(
            cutoff=18,
            projector_delta=0.34,
            grid_points=2049,
            readout_confusion=((0.985, 0.015), (0.025, 0.975)),
            controller_residual_phase_by_observed=((0.0, 0.03), (0.0, -0.02)),
        )
    )


@pytest.fixture(scope="module")
def observed_exact(observed: SBSFockOneRoundSimulator):
    return observed.run_exact_cycle(observed.initialize("+i"))


def _assert_density(state: FiniteCutoffDensity) -> None:
    assert np.trace(state.matrix) == pytest.approx(1.0, abs=1.0e-10)
    assert np.linalg.norm(state.matrix - state.matrix.conj().T) < 1.0e-10
    assert np.min(np.linalg.eigvalsh(state.matrix)) > -1.0e-9


def _hermitian_function(matrix: np.ndarray, function: object) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    return (vectors * function(values)) @ vectors.conj().T  # type: ignore[operator]


@pytest.mark.parametrize("label", ["0", "1", "+", "-", "+i", "-i", "mixed"])
def test_logical_density_labels_are_physical(label: str) -> None:
    state = logical_density(label)  # type: ignore[arg-type]
    assert state.shape == (2, 2)
    assert np.trace(state) == pytest.approx(1.0)
    assert np.min(np.linalg.eigvalsh(state)) >= -1.0e-12
    assert not state.flags.writeable


def test_logical_density_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        logical_density("cat")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "change",
    [
        {"cutoff": 7},
        {"grid_points": 1024},
        {"projector_delta": 0.0},
        {"readout_confusion": ((0.9, 0.2), (0.0, 1.0))},
        {"readout_confusion": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))},
        {"controller_residual_phase_by_observed": ((0.0, 4.0), (0.0, 0.0))},
        {"controller_residual_phase_by_observed": ((0.0,), (0.0,))},
        {"completion_policy": "none"},
        {"protocol_id": "PROTO-WRONG"},
        {"scope": "overclaim"},
    ],
)
def test_cycle_config_fails_closed(change: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        SBSFockCycleConfig(**change)


def test_idle_config_rejects_unphysical_channel_parameters() -> None:
    with pytest.raises(ValueError):
        FockIdleConfig(loss_transmissivity=1.1)
    with pytest.raises(ValueError):
        FockIdleConfig(thermal_rate_time=-0.1)
    with pytest.raises(ValueError):
        FockIdleConfig(high_fock_proxy_probability=-0.1)
    with pytest.raises(ValueError):
        FockIdleConfig(kerr_strength=float("nan"))


def test_lowdin_code_basis_is_orthonormal_and_canonical_scaled(
    clean: SBSFockOneRoundSimulator,
) -> None:
    basis = clean.code_basis
    assert np.linalg.norm(basis.isometry.conj().T @ basis.isometry - np.eye(2)) < 1.0e-12
    assert np.linalg.norm(basis.projector @ basis.projector - basis.projector) < 1.0e-12
    assert abs(basis.raw_overlap) < 0.03
    # N=18 is intentionally the low end of the cutoff sweep; the production
    # N=24 result is checked separately in the emitted validation artifact.
    assert min(basis.captured_probabilities) > 0.94
    assert basis.source_coordinate_scale == pytest.approx(np.sqrt(2.0))
    assert basis.source_coordinate_scale == PAPER_CANONICAL_SOURCE_SCALE
    assert not basis.isometry.flags.writeable


@pytest.mark.parametrize("label", ["0", "1", "+", "-", "+i", "-i"])
def test_initialization_and_logical_projection_roundtrip(
    clean: SBSFockOneRoundSimulator, label: str
) -> None:
    state = clean.initialize(label)  # type: ignore[arg-type]
    projection = clean.logical_project(state, frame=PauliFrame())
    assert projection.code_survival_probability == pytest.approx(1.0, abs=1.0e-12)
    assert projection.frame_corrected_logical_density == pytest.approx(logical_density(label))
    _assert_density(state)


def test_initialization_respects_existing_pauli_frame(
    clean: SBSFockOneRoundSimulator,
) -> None:
    frame = PauliFrame(x=1, z=1)
    state = clean.initialize("+i", frame=frame)
    projection = clean.logical_project(state, frame=frame)
    assert projection.frame_corrected_logical_density == pytest.approx(logical_density("+i"))


@pytest.mark.parametrize(
    "bad",
    [
        np.eye(3),
        np.array([[1.0, 0.2], [0.0, 0.0]]),
        np.diag([1.2, -0.2]),
        np.diag([0.6, 0.6]),
    ],
)
def test_initialization_rejects_invalid_logical_density(
    clean: SBSFockOneRoundSimulator, bad: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        clean.initialize(bad)


def test_logical_projection_rejects_zero_code_survival(
    clean: SBSFockOneRoundSimulator,
) -> None:
    values, vectors = np.linalg.eigh(clean.code_basis.projector)
    complement = vectors[:, int(np.argmin(values))]
    with pytest.raises(RuntimeError):
        clean.logical_project(FiniteCutoffDensity.from_ket(complement), frame=PauliFrame())


def test_raw_x_kraus_matches_printed_paper_formula(
    clean: SBSFockOneRoundSimulator,
) -> None:
    delta_squared = clean.config.projector_delta**2
    root_pi = np.sqrt(np.pi)
    cosine_p = _hermitian_function(root_pi * clean.p, np.cos)
    sine_p = _hermitian_function(root_pi * clean.p, np.sin)
    cosine_x = _hermitian_function(root_pi * delta_squared * clean.x, np.cos)
    sine_x = _hermitian_function(root_pi * delta_squared * clean.x, np.sin)
    expected_g = cosine_p @ cosine_x + np.sin(np.pi * delta_squared / 2.0) * cosine_p
    expected_e = (
        -np.cos(np.pi * delta_squared / 2.0) * sine_p
        + 1.0j * cosine_p @ sine_x
    )
    actual_g, actual_e = clean.constituent_kraus("X", raw=True)
    assert np.allclose(actual_g, expected_g, atol=1.0e-12)
    assert np.allclose(actual_e, expected_e, atol=1.0e-12)


def test_raw_z_kraus_is_exact_x_to_minus_p_p_to_x_substitution(
    clean: SBSFockOneRoundSimulator,
) -> None:
    delta_squared = clean.config.projector_delta**2
    root_pi = np.sqrt(np.pi)
    cosine_p = _hermitian_function(root_pi * clean.x, np.cos)
    sine_p = _hermitian_function(root_pi * clean.x, np.sin)
    cosine_x = _hermitian_function(-root_pi * delta_squared * clean.p, np.cos)
    sine_x = _hermitian_function(-root_pi * delta_squared * clean.p, np.sin)
    expected_g = cosine_p @ cosine_x + np.sin(np.pi * delta_squared / 2.0) * cosine_p
    expected_e = (
        -np.cos(np.pi * delta_squared / 2.0) * sine_p
        + 1.0j * cosine_p @ sine_x
    )
    actual_g, actual_e = clean.constituent_kraus("Z", raw=True)
    assert np.allclose(actual_g, expected_g, atol=1.0e-12)
    assert np.allclose(actual_e, expected_e, atol=1.0e-12)


@pytest.mark.parametrize("quadrature", ["X", "Z"])
def test_raw_cutoff_defect_is_detected_and_code_restricted(
    clean: SBSFockOneRoundSimulator, quadrature: str
) -> None:
    diagnostics = clean.kraus_diagnostics(quadrature)  # type: ignore[arg-type]
    assert diagnostics.raw_completeness_frobenius_error > 0.5
    assert diagnostics.raw_code_subspace_completeness_error < 0.15
    assert diagnostics.raw_gram_minimum_eigenvalue > 0.5
    assert diagnostics.raw_gram_condition_number < 2.0
    assert diagnostics.completion_pair_frobenius_change > 0.1


@pytest.mark.parametrize("quadrature", ["X", "Z"])
def test_completed_kraus_pair_is_cptp(
    clean: SBSFockOneRoundSimulator, quadrature: str
) -> None:
    pair = clean.constituent_kraus(quadrature)  # type: ignore[arg-type]
    completeness = sum(operator.conj().T @ operator for operator in pair)
    assert np.allclose(completeness, clean.model.identity, atol=1.0e-11)
    assert clean.kraus_diagnostics(
        quadrature  # type: ignore[arg-type]
    ).completed_completeness_frobenius_error < 1.0e-10
    assert all(not operator.flags.writeable for operator in pair)


def test_completion_is_shared_right_inverse_square_root(
    clean: SBSFockOneRoundSimulator,
) -> None:
    raw = clean.constituent_kraus("X", raw=True)
    gram = sum(operator.conj().T @ operator for operator in raw)
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.conj().T))
    inverse_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.conj().T
    expected = tuple(operator @ inverse_sqrt for operator in raw)
    actual = clean.constituent_kraus("X")
    assert np.allclose(actual[0], expected[0], atol=1.0e-12)
    assert np.allclose(actual[1], expected[1], atol=1.0e-12)


def test_identity_idle_is_exact(clean: SBSFockOneRoundSimulator) -> None:
    state = clean.initialize("+")
    assert np.array_equal(clean.apply_idle(state).matrix, state.matrix)


def test_idle_pipeline_matches_explicit_channel_order() -> None:
    config = SBSFockCycleConfig(
        cutoff=14,
        grid_points=2049,
        idle=FockIdleConfig(
            displacement=0.04 - 0.02j,
            loss_transmissivity=0.98,
            thermal_rate_time=0.01,
            thermal_bath_occupation=0.04,
            phase_diffusion_variance=0.002,
            kerr_strength=0.003,
            high_fock_proxy_probability=0.002,
        ),
    )
    simulator = SBSFockOneRoundSimulator(config)
    state = simulator.initialize("0")
    expected = simulator.model.displace(state, config.idle.displacement)
    expected = simulator.model.pure_loss(expected, config.idle.loss_transmissivity)
    expected = simulator.model.thermal_excitation(
        expected,
        rate_time=config.idle.thermal_rate_time,
        bath_occupation=config.idle.thermal_bath_occupation,
    )
    expected = simulator.model.phase_diffusion(expected, config.idle.phase_diffusion_variance)
    expected = simulator.model.kerr(expected, config.idle.kerr_strength)
    expected = simulator.model.high_fock_leakage_proxy(
        expected, config.idle.high_fock_proxy_probability
    )
    actual = simulator.apply_idle(state)
    assert np.allclose(actual.matrix, expected.matrix, atol=1.0e-12)
    _assert_density(actual)


def test_perfect_readout_has_four_physical_kraus_paths(
    clean: SBSFockOneRoundSimulator,
) -> None:
    result = clean.run_exact_cycle(clean.initialize("+"))
    assert len(result.branches) == 4
    assert result.total_probability == pytest.approx(1.0, abs=1.0e-12)
    assert {branch.hidden_kraus_label for branch in result.branches} == {
        "K_gg", "K_ge", "K_eg", "K_ee"
    }


def test_full_cycle_uses_chronological_xz_but_kraus_label_zx(observed_exact) -> None:
    for branch in observed_exact.branches:
        x, z = branch.chronological_observed_outcomes
        assert branch.observed_constituents[0].quadrature == "X"
        assert branch.observed_constituents[1].quadrature == "Z"
        assert branch.observed_kraus_label == f"K_{z}{x}"
        hidden_x, hidden_z = branch.chronological_hidden_outcomes
        assert branch.hidden_kraus_label == f"K_{hidden_z}{hidden_x}"


def test_full_cycle_applies_deterministic_x_then_z_frame(
    clean: SBSFockOneRoundSimulator,
) -> None:
    input_frame = PauliFrame(x=1, z=0)
    initial = clean.initialize("-i", frame=input_frame)
    result = clean.run_exact_cycle(initial, input_frame=input_frame)
    assert result.output_frame == input_frame.after_x_constituent().after_z_constituent()
    assert result.output_frame == input_frame.after_full_sbs_cycle()
    target = logical_density("-i")
    fidelity = np.trace(
        result.unconditional_projection.frame_corrected_logical_density @ target
    ).real
    assert fidelity > 0.99


def test_noisy_readout_creates_all_sixteen_hidden_observed_branches(observed_exact) -> None:
    assert len(observed_exact.branches) == 16
    assert observed_exact.total_probability == pytest.approx(1.0, abs=1.0e-12)
    assert {branch.observed_kraus_label for branch in observed_exact.branches} == {
        "K_gg", "K_ge", "K_eg", "K_ee"
    }


def test_exact_unconditional_state_is_probability_weighted_branch_mixture(
    observed_exact,
) -> None:
    mixture = sum(
        (branch.probability * branch.final_state.matrix for branch in observed_exact.branches),
        np.zeros_like(observed_exact.unconditional_state.matrix),
    )
    assert np.allclose(mixture, observed_exact.unconditional_state.matrix, atol=1.0e-12)
    _assert_density(observed_exact.unconditional_state)
    for branch in observed_exact.branches:
        _assert_density(branch.final_state)


def test_observed_schema_excludes_hidden_truth(observed_exact) -> None:
    observed = asdict(observed_exact.branches[0].observed_constituents[0])
    assert set(observed) == {
        "quadrature",
        "observed_outcome",
        "chronological_index",
        "controller_residual_phase",
        "input_frame",
        "output_frame",
    }
    assert "hidden_outcome" not in observed
    assert "hidden_probability" not in observed


def test_truth_schema_keeps_hidden_kraus_backaction_diagnostics(observed_exact) -> None:
    truth = asdict(observed_exact.branches[0].truth_constituents[0])
    assert truth["hidden_outcome"] in {"g", "e"}
    assert 0.0 <= truth["hidden_probability"] <= 1.0
    assert "code_survival_before" in truth
    assert "code_survival_after_hidden_kraus" in truth


def test_controller_action_depends_on_observed_not_hidden_outcome(observed_exact) -> None:
    saw_hidden_e_observed_g = False
    saw_hidden_g_observed_e = False
    for branch in observed_exact.branches:
        for observed_record, truth in zip(
            branch.observed_constituents, branch.truth_constituents
        ):
            configured = 0.0 if observed_record.observed_outcome == "g" else (
                0.03 if observed_record.quadrature == "X" else -0.02
            )
            assert observed_record.controller_residual_phase == pytest.approx(configured)
            saw_hidden_e_observed_g |= (
                truth.hidden_outcome == "e" and observed_record.observed_outcome == "g"
            )
            saw_hidden_g_observed_e |= (
                truth.hidden_outcome == "g" and observed_record.observed_outcome == "e"
            )
    assert saw_hidden_e_observed_g and saw_hidden_g_observed_e


def test_perfectly_flipped_readout_maps_hidden_to_opposite_observation() -> None:
    simulator = SBSFockOneRoundSimulator(
        SBSFockCycleConfig(
            cutoff=14,
            grid_points=2049,
            readout_confusion=((0.0, 1.0), (1.0, 0.0)),
        )
    )
    result = simulator.run_exact_cycle(simulator.initialize("0"))
    assert len(result.branches) == 4
    for branch in result.branches:
        assert all(
            observed_outcome != hidden_outcome
            for observed_outcome, hidden_outcome in zip(
                branch.chronological_observed_outcomes,
                branch.chronological_hidden_outcomes,
            )
        )


def test_zero_residual_phase_keeps_state_equal_across_observations_for_same_hidden() -> None:
    simulator = SBSFockOneRoundSimulator(
        SBSFockCycleConfig(
            cutoff=14,
            grid_points=2049,
            readout_confusion=((0.5, 0.5), (0.5, 0.5)),
        )
    )
    result = simulator.run_exact_cycle(simulator.initialize("0"))
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for branch in result.branches:
        grouped.setdefault(branch.chronological_hidden_outcomes, []).append(
            branch.final_state.matrix
        )
    assert all(len(states) == 4 for states in grouped.values())
    assert all(
        all(np.allclose(states[0], state) for state in states[1:])
        for states in grouped.values()
    )


def test_nonzero_residual_phase_changes_observation_routed_state(observed_exact) -> None:
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for branch in observed_exact.branches:
        grouped.setdefault(branch.chronological_hidden_outcomes, []).append(
            branch.final_state.matrix
        )
    assert any(
        any(not np.allclose(states[0], state) for state in states[1:])
        for states in grouped.values()
    )


@pytest.mark.parametrize("error_name", ["loss", "gain"])
def test_one_cycle_pumps_first_photon_error_toward_code(
    clean: SBSFockOneRoundSimulator, error_name: str
) -> None:
    base = clean.initialize("0")
    error = clean.model.a if error_name == "loss" else clean.model.adag
    matrix = error @ base.matrix @ error.conj().T
    matrix /= np.trace(matrix).real
    disturbed = FiniteCutoffDensity(matrix, clean.config.cutoff)
    before = np.trace(clean.code_basis.projector @ disturbed.matrix).real
    after = clean.run_exact_cycle(
        disturbed
    ).unconditional_projection.code_survival_probability
    assert after - before > 0.35


def test_sample_branch_is_seed_reproducible(observed_exact) -> None:
    first_rng = np.random.default_rng(144)
    second_rng = np.random.default_rng(144)
    first = [
        SBSFockOneRoundSimulator.sample_branch(observed_exact, first_rng).observed_kraus_label
        for _ in range(30)
    ]
    second = [
        SBSFockOneRoundSimulator.sample_branch(observed_exact, second_rng).observed_kraus_label
        for _ in range(30)
    ]
    assert first == second


def test_sample_branch_rejects_wrong_rng_and_result(observed_exact) -> None:
    with pytest.raises(TypeError):
        SBSFockOneRoundSimulator.sample_branch(object(), np.random.default_rng(1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SBSFockOneRoundSimulator.sample_branch(observed_exact, object())  # type: ignore[arg-type]


def test_state_cutoff_mismatch_and_invalid_quadrature_fail_closed(
    clean: SBSFockOneRoundSimulator,
) -> None:
    wrong = FiniteCutoffFockModel(19).basis_state(0)
    with pytest.raises(ValueError):
        clean.run_exact_cycle(wrong)
    with pytest.raises(ValueError):
        clean.constituent_kraus("Q")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        clean.kraus_diagnostics("Q")  # type: ignore[arg-type]


def test_production_validation_is_formula_cptp_mc_error_and_cutoff_backed() -> None:
    result = run_sbs_fock_cycle_validation(
        cutoff=18,
        projector_delta=0.34,
        grid_points=2049,
        monte_carlo_samples=50_000,
        seed=2301,
    )
    assert result.passed
    assert result.exact_branch_count == 16
    assert result.clean_average_conditional_logical_fidelity > 0.99
    assert result.clean_average_code_weighted_fidelity > result.noisy_average_code_weighted_fidelity
    assert result.one_cycle_photon_error_code_survival_gain > 0.4
    assert [point.cutoff for point in result.cutoff_sweep] == [18, 24, 30, 36, 42]
    assert len(result.checks) == 16


def test_validation_rejects_demo_scale_monte_carlo() -> None:
    with pytest.raises(ValueError):
        run_sbs_fock_cycle_validation(monte_carlo_samples=9999)


def test_validation_writer_emits_scope_and_audited_kraus() -> None:
    result = run_sbs_fock_cycle_validation(
        cutoff=18,
        grid_points=2049,
        monte_carlo_samples=10_000,
        seed=2,
    )
    output = Path.cwd() / ".pytest_t2_3_2_fock_cycle.json"
    try:
        write_sbs_fock_cycle_validation(result, output)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["passed"] is True
        assert payload["scope"] == FOCK_SBS_CYCLE_SCOPE
        assert payload["kraus_diagnostics"][0]["raw_completeness_frobenius_error"] > 0.5
        assert payload["kraus_diagnostics"][0]["completed_completeness_frobenius_error"] < 1e-10
    finally:
        output.unlink(missing_ok=True)


def test_scope_and_protocol_id_block_pulse_device_overclaim(
    clean: SBSFockOneRoundSimulator,
) -> None:
    assert clean.config.protocol_id == SBS_PROTOCOL_ID
    lowered = FOCK_SBS_CYCLE_SCOPE.lower()
    assert "no pulse-level ecd" in lowered
    assert "explicit transmon" in lowered
    assert "device calibration" in lowered
    assert "hardware claim" in lowered
