"""Implementation-level qualification gates for Phase-9 backend A."""

from __future__ import annotations

from dataclasses import replace
from math import pi, sqrt

import numpy as np

from ..fock_density_model import FiniteCutoffDensity
from .schema import (
    BACKEND_A_ID, BACKEND_A_SCOPE, BackendAConfig, BackendADriftState, BackendAExogenous,
    BackendAQualification, BackendAQualificationThresholds, _density_diagnostics,
    _trace_distance, backend_a_exogenous, diagnostic_action_word,
)
from .simulator import Phase9BackendASimulator


def _noise_free_config(
    base: BackendAConfig,
    **overrides: object,
) -> BackendAConfig:
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
    return replace(base, **values)


def _embedded_density_distance(
    low: FiniteCutoffDensity,
    high: FiniteCutoffDensity,
) -> float:
    if low.cutoff >= high.cutoff:
        raise ValueError("low cutoff must be smaller than high cutoff")
    embedded = np.zeros_like(high.matrix)
    embedded[: low.cutoff, : low.cutoff] = low.matrix
    return _trace_distance(embedded, high.matrix)


def run_backend_a_qualification(
    config: BackendAConfig | None = None,
    thresholds: BackendAQualificationThresholds | None = None,
) -> BackendAQualification:
    """Execute the T9.2.2 implementation-level qualification suite."""

    base = BackendAConfig() if config is None else config
    if not isinstance(base, BackendAConfig):
        raise TypeError("config must be BackendAConfig or None")
    limits = (
        BackendAQualificationThresholds()
        if thresholds is None
        else thresholds
    )
    if not isinstance(limits, BackendAQualificationThresholds):
        raise TypeError(
            "thresholds must be BackendAQualificationThresholds or None"
        )
    metrics: dict[str, float | int | str | bool] = {}
    checks: dict[str, bool] = {}

    # 1. Small-system exact Choi test of the actual vectorized GKSL code path.
    cp_config = replace(
        base,
        cutoff=2,
        iq_samples=min(base.iq_samples, 4),
        logical_grid_points=1025,
    )
    cp_simulator = Phase9BackendASimulator(cp_config)
    cp_hamiltonian = cp_simulator._base_hamiltonian(BackendADriftState())
    channel = cp_simulator.channel_diagnostics(cp_hamiltonian, 0.07)
    metrics.update(
        {
            "choi_dimension": channel.dimension,
            "choi_minimum_eigenvalue": channel.choi_minimum_eigenvalue,
            "choi_trace": channel.choi_trace,
            "choi_tp_frobenius": channel.trace_preservation_frobenius,
            "choi_hermiticity_frobenius": channel.hermiticity_frobenius,
        }
    )
    checks["gksl_channel_cp"] = (
        channel.choi_minimum_eigenvalue
        >= limits.choi_minimum_eigenvalue
    )
    checks["gksl_channel_tp"] = (
        channel.trace_preservation_frobenius
        <= limits.choi_tp_frobenius
    )
    checks["gksl_choi_hermitian"] = (
        channel.hermiticity_frobenius
        <= limits.choi_hermiticity_frobenius
    )

    # 2. One full joint round: density and instrument invariants.
    simulator = Phase9BackendASimulator(base)
    initial, evaluator = simulator.initialize_logical("0")
    idle = diagnostic_action_word("IDLE")
    exogenous = backend_a_exogenous(
        seed=731,
        round_index=0,
        iq_samples=base.iq_samples,
    )
    normal = simulator.step(
        initial,
        idle,
        exogenous,
        evaluator=evaluator,
    )
    diagnostics = _density_diagnostics(normal.state.joint_density)
    metrics.update(
        {
            "full_round_trace_error": abs(
                diagnostics["trace_real"] - 1.0
            )
            + abs(diagnostics["trace_imag"]),
            "full_round_hermiticity_frobenius": diagnostics[
                "hermiticity_frobenius"
            ],
            "full_round_minimum_eigenvalue": diagnostics[
                "minimum_eigenvalue"
            ],
            "measurement_completeness_frobenius": simulator.measurement_completeness_error(),
            "reset_completeness_frobenius": simulator.reset_completeness_error(),
            "full_round_posterior_sum_error": abs(
                sum(normal.observation.posterior_levels) - 1.0
            ),
            "full_round_code_survival": (
                normal.logical.code_survival_probability
                if normal.logical is not None
                else -1.0
            ),
        }
    )
    checks["full_round_trace"] = (
        metrics["full_round_trace_error"]
        <= limits.density_trace_error
    )
    checks["full_round_hermiticity"] = (
        diagnostics["hermiticity_frobenius"]
        <= limits.density_hermiticity_frobenius
    )
    checks["full_round_positive"] = (
        diagnostics["minimum_eigenvalue"]
        >= limits.density_minimum_eigenvalue
    )
    checks["measurement_instrument_complete"] = (
        metrics["measurement_completeness_frobenius"]
        <= limits.instrument_completeness_frobenius
    )
    checks["reset_instrument_complete"] = (
        metrics["reset_completeness_frobenius"]
        <= limits.instrument_completeness_frobenius
    )
    checks["probabilities_normalized"] = (
        metrics["full_round_posterior_sum_error"]
        <= limits.probability_sum_error
    )
    checks["logical_tracking_finite"] = (
        normal.logical is not None
        and 0.0 <= normal.logical.code_survival_probability <= 1.0
        and 0.0 <= normal.logical.target_fidelity <= 1.0
    )

    # 3. Zero-noise identity and ideal displacement limits.
    # Use a fine pulse discretization for the continuum ideal-action limit.
    # The separate convergence test below verifies that this is not merely a
    # hand-picked one-grid agreement.
    limit_config = _noise_free_config(base, substeps_per_segment=64)
    limit_simulator = Phase9BackendASimulator(limit_config)
    vacuum = limit_simulator.initialize_fock()
    limit_exogenous = backend_a_exogenous(
        seed=11,
        round_index=0,
        iq_samples=limit_config.iq_samples,
    )
    zero = limit_simulator.step(vacuum, idle, limit_exogenous)
    zero_distance = _trace_distance(
        vacuum.joint_density,
        zero.state.joint_density,
    )
    x_action = diagnostic_action_word("X")
    acted = limit_simulator.step(vacuum, x_action, limit_exogenous)
    alpha = limit_simulator._action_alpha(x_action)
    displacement = limit_simulator.oscillator.displacement_operator(alpha)
    expected = np.kron(
        displacement
        @ limit_simulator.oscillator_density(vacuum.joint_density).matrix
        @ displacement.conj().T,
        limit_simulator.level_projectors[0],
    )
    ideal_action_distance = _trace_distance(
        expected,
        acted.state.joint_density,
    )
    metrics["zero_noise_idle_trace_distance"] = zero_distance
    metrics["ideal_action_trace_distance"] = ideal_action_distance
    checks["zero_noise_idle_limit"] = (
        zero_distance <= limits.zero_noise_idle_trace_distance
    )
    checks["ideal_action_limit"] = (
        ideal_action_distance <= limits.ideal_action_trace_distance
    )

    # 4. Reset success/failure limits and f-state persistence.
    reset_word = diagnostic_action_word("RESET")
    success_config = _noise_free_config(
        base,
        reset_success_e=1.0,
        reset_success_f=1.0,
    )
    success_simulator = Phase9BackendASimulator(success_config)
    f_state = success_simulator.initialize_fock(ancilla_state="f")
    reset_exogenous = BackendAExogenous(
        emission_uniform=0.5,
        iq_standard_i=(0.0,) * success_config.iq_samples,
        iq_standard_q=(0.0,) * success_config.iq_samples,
        reset_uniform=0.5,
        reset_ack_uniform=0.5,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=17,
        round_index=0,
    )
    reset_success = success_simulator.step(
        f_state,
        reset_word,
        reset_exogenous,
    )
    success_g = success_simulator.level_probabilities(
        reset_success.state.joint_density
    )[0]
    failure_config = _noise_free_config(
        base,
        reset_success_e=0.0,
        reset_success_f=0.0,
    )
    failure_simulator = Phase9BackendASimulator(failure_config)
    failure_f = failure_simulator.initialize_fock(ancilla_state="f")
    reset_failure = failure_simulator.step(
        failure_f,
        reset_word,
        reset_exogenous,
    )
    failure_f_probability = failure_simulator.level_probabilities(
        reset_failure.state.joint_density
    )[2]
    persistent = failure_simulator.step(
        failure_f,
        idle,
        reset_exogenous,
    )
    persistent_f_probability = failure_simulator.level_probabilities(
        persistent.state.joint_density
    )[2]
    metrics.update(
        {
            "large_reset_g_probability": success_g,
            "failed_reset_f_probability": failure_f_probability,
            "no_reset_f_persistence_probability": persistent_f_probability,
        }
    )
    checks["large_reset_limit"] = (
        success_g >= limits.limit_population_minimum
        and reset_success.truth.reset_hidden_outcome == "success"
        and reset_success.observation.reset_ack == "success"
    )
    checks["reset_failure_preserves_f"] = (
        failure_f_probability >= limits.limit_population_minimum
        and reset_failure.truth.reset_hidden_outcome == "failure"
        and reset_failure.observation.reset_ack == "failure"
    )
    checks["f_state_persistence"] = (
        persistent_f_probability >= limits.limit_population_minimum
    )

    # 5. IQ backaction must alter the quantum state through a Kraus update.
    measurement_config = _noise_free_config(
        base,
        iq_sigma=0.24,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.6)),
    )
    measurement_simulator = Phase9BackendASimulator(measurement_config)
    ge_plus = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / sqrt(2.0)
    coherent_ancilla = measurement_simulator.initialize_fock(
        ancilla_state=ge_plus,
    )
    measurement_exogenous = BackendAExogenous(
        emission_uniform=0.1,
        iq_standard_i=(0.0,) * measurement_config.iq_samples,
        iq_standard_q=(0.0,) * measurement_config.iq_samples,
        reset_uniform=0.5,
        reset_ack_uniform=0.5,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=23,
        round_index=0,
    )
    measured = measurement_simulator.step(
        coherent_ancilla,
        idle,
        measurement_exogenous,
    )
    before_coherence = abs(
        measurement_simulator.ancilla_density(
            coherent_ancilla.joint_density
        )[0, 1]
    )
    after_coherence = abs(
        measurement_simulator.ancilla_density(
            measured.state.joint_density
        )[0, 1]
    )
    posterior_peak = max(measured.observation.posterior_levels)
    metrics.update(
        {
            "measurement_coherence_before": float(before_coherence),
            "measurement_coherence_after": float(after_coherence),
            "measurement_posterior_peak": posterior_peak,
        }
    )
    checks["iq_drives_measurement_backaction"] = (
        after_coherence
        < before_coherence * limits.measurement_coherence_ratio
        and posterior_peak > limits.measurement_posterior_peak
    )

    # 6. The Ramsey interaction must make the IQ instrument a syndrome
    # measurement of the oscillator, rather than an ancilla-only label sensor.
    syndrome_config = _noise_free_config(
        base,
        ramsey_angle=pi / 2.0,
        ramsey_pulse_duration=0.03,
        sense_duration=0.8,
        dispersive_chi=1.0,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
        substeps_per_segment=16,
    )
    syndrome_simulator = Phase9BackendASimulator(syndrome_config)
    ket_zero = np.zeros(syndrome_config.cutoff, dtype=np.complex128)
    ket_zero[0] = 1.0
    ket_one = np.zeros(syndrome_config.cutoff, dtype=np.complex128)
    ket_one[1] = 1.0
    syndrome_exogenous = BackendAExogenous(
        emission_uniform=0.1,
        iq_standard_i=(0.0,) * syndrome_config.iq_samples,
        iq_standard_q=(0.0,) * syndrome_config.iq_samples,
        reset_uniform=0.5,
        reset_ack_uniform=0.5,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=29,
        round_index=0,
    )
    syndrome_zero = syndrome_simulator.step(
        syndrome_simulator.initialize_fock(oscillator_ket=ket_zero),
        idle,
        syndrome_exogenous,
    )
    syndrome_one = syndrome_simulator.step(
        syndrome_simulator.initialize_fock(oscillator_ket=ket_one),
        idle,
        syndrome_exogenous,
    )
    level_tv = 0.5 * float(
        np.sum(
            np.abs(
                np.asarray(
                    syndrome_zero.truth.pre_measurement_level_probabilities
                )
                - np.asarray(
                    syndrome_one.truth.pre_measurement_level_probabilities
                )
            )
        )
    )
    superposition = (ket_zero + ket_one) / sqrt(2.0)
    superposition_state = syndrome_simulator.initialize_fock(
        oscillator_ket=superposition
    )
    syndrome_measured = syndrome_simulator.step(
        superposition_state,
        idle,
        syndrome_exogenous,
    )
    oscillator_backaction = _trace_distance(
        syndrome_simulator.oscillator_density(
            superposition_state.joint_density
        ).matrix,
        syndrome_simulator.oscillator_density(
            syndrome_measured.state.joint_density
        ).matrix,
    )
    metrics.update(
        {
            "syndrome_fock0_vs_fock1_level_tv": level_tv,
            "syndrome_oscillator_backaction_trace_distance": oscillator_backaction,
        }
    )
    checks["ramsey_syndrome_state_dependence"] = (
        level_tv > limits.syndrome_state_dependence_minimum
    )
    checks["syndrome_measurement_backacts_on_oscillator"] = (
        oscillator_backaction
        > limits.syndrome_backaction_trace_distance_minimum
    )

    # 7. The explicit e<->f Hamiltonian must create action-dependent leakage,
    # not merely increment a classical leakage label.
    leakage_config = _noise_free_config(
        base,
        action_leakage_coupling=0.8,
        action_duration=0.1,
        substeps_per_segment=32,
    )
    leakage_simulator = Phase9BackendASimulator(leakage_config)
    leakage_initial = leakage_simulator.initialize_fock(
        ancilla_state="e"
    )
    leakage_exogenous = BackendAExogenous(
        emission_uniform=0.2,
        iq_standard_i=(0.0,) * leakage_config.iq_samples,
        iq_standard_q=(0.0,) * leakage_config.iq_samples,
        reset_uniform=0.2,
        reset_ack_uniform=0.2,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=30,
        round_index=0,
    )
    leakage_idle = leakage_simulator.step(
        leakage_initial,
        idle,
        leakage_exogenous,
    )
    leakage_x = leakage_simulator.step(
        leakage_initial,
        x_action,
        leakage_exogenous,
    )
    action_f_difference = (
        leakage_simulator.level_probabilities(
            leakage_x.state.joint_density
        )[2]
        - leakage_simulator.level_probabilities(
            leakage_idle.state.joint_density
        )[2]
    )
    metrics["action_induced_f_population_difference"] = action_f_difference
    checks["action_induces_physical_f_population"] = (
        action_f_difference
        > limits.action_induced_f_population_minimum
    )

    # 8. Same addressable randomness, different action: both quantum state and
    # latent drift must change.  This rejects independent label-noise shortcuts.
    intervention_config = replace(
        base,
        ramsey_angle=0.0,
        sense_duration=0.0,
        iq_centers=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    intervention_simulator = Phase9BackendASimulator(intervention_config)
    intervention_initial = intervention_simulator.initialize_fock()
    intervention_exogenous = backend_a_exogenous(
        seed=991,
        round_index=0,
        iq_samples=intervention_config.iq_samples,
    )
    intervention_idle = intervention_simulator.step(
        intervention_initial,
        idle,
        intervention_exogenous,
    )
    intervention_x = intervention_simulator.step(
        intervention_initial,
        x_action,
        intervention_exogenous,
    )
    state_intervention = _trace_distance(
        intervention_idle.state.joint_density,
        intervention_x.state.joint_density,
    )
    drift_intervention = float(
        np.linalg.norm(
            intervention_idle.state.drift.vector()
            - intervention_x.state.drift.vector()
        )
    )
    metrics.update(
        {
            "action_intervention_state_trace_distance": state_intervention,
            "action_intervention_drift_l2": drift_intervention,
            "action_intervention_shared_exogenous": (
                intervention_idle.exogenous == intervention_x.exogenous
            ),
        }
    )
    checks["action_changes_quantum_transition"] = (
        state_intervention
        > limits.action_state_trace_distance_minimum
    )
    checks["action_changes_latent_drift"] = (
        drift_intervention > limits.action_drift_l2_minimum
    )
    checks["intervention_uses_common_randomness"] = (
        intervention_idle.exogenous == intervention_x.exogenous
    )

    # 9. Seed determinism and genuine stochastic sensitivity.
    deterministic_one = simulator.simulate(
        initial,
        (idle, x_action, idle),
        seed=404,
        evaluator=evaluator,
    )
    deterministic_two = simulator.simulate(
        initial,
        (idle, x_action, idle),
        seed=404,
        evaluator=evaluator,
    )
    stochastic_other = simulator.simulate(
        initial,
        (idle, x_action, idle),
        seed=405,
        evaluator=evaluator,
    )
    deterministic_density_error = float(
        np.max(
            np.abs(
                deterministic_one.final_state.joint_density
                - deterministic_two.final_state.joint_density
            )
        )
    )
    deterministic_iq_error = max(
        float(
            np.max(
                np.abs(
                    left.observation.iq_i - right.observation.iq_i
                )
            )
        )
        for left, right in zip(
            deterministic_one.rounds,
            deterministic_two.rounds,
        )
    )
    seed_sensitivity = float(
        np.max(
            np.abs(
                deterministic_one.rounds[0].observation.iq_i
                - stochastic_other.rounds[0].observation.iq_i
            )
        )
    )
    metrics.update(
        {
            "seed_repeat_density_max_error": deterministic_density_error,
            "seed_repeat_iq_max_error": deterministic_iq_error,
            "different_seed_iq_max_difference": seed_sensitivity,
        }
    )
    checks["seed_determinism"] = (
        deterministic_density_error == 0.0
        and deterministic_iq_error == 0.0
    )
    checks["different_seed_changes_observation"] = (
        seed_sensitivity
        > limits.different_seed_iq_difference_minimum
    )

    # 10. Step-size convergence of a non-commuting drift + shaped action pulse.
    convergence_base = _noise_free_config(
        base,
        dispersive_chi=0.23,
        self_kerr=0.015,
        ramsey_angle=0.0,
        sense_duration=0.0,
        drift_action_kick=0.0,
    )
    pre_coarse_simulator = Phase9BackendASimulator(
        replace(convergence_base, substeps_per_segment=8)
    )
    coarse_simulator = Phase9BackendASimulator(
        replace(convergence_base, substeps_per_segment=16)
    )
    fine_simulator = Phase9BackendASimulator(
        replace(convergence_base, substeps_per_segment=32)
    )
    convergence_drift = BackendADriftState(drive_q=0.07, drive_p=-0.04)
    pre_coarse_initial = pre_coarse_simulator.initialize_fock(
        drift=convergence_drift
    )
    coarse_initial = coarse_simulator.initialize_fock(drift=convergence_drift)
    fine_initial = fine_simulator.initialize_fock(drift=convergence_drift)
    convergence_exogenous = BackendAExogenous(
        emission_uniform=0.2,
        iq_standard_i=(0.0,) * convergence_base.iq_samples,
        iq_standard_q=(0.0,) * convergence_base.iq_samples,
        reset_uniform=0.2,
        reset_ack_uniform=0.2,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=55,
        round_index=0,
    )
    pre_coarse = pre_coarse_simulator.step(
        pre_coarse_initial,
        x_action,
        convergence_exogenous,
    )
    coarse = coarse_simulator.step(
        coarse_initial,
        x_action,
        convergence_exogenous,
    )
    fine = fine_simulator.step(
        fine_initial,
        x_action,
        convergence_exogenous,
    )
    step_distance = _trace_distance(
        coarse.state.joint_density,
        fine.state.joint_density,
    )
    previous_step_distance = _trace_distance(
        pre_coarse.state.joint_density,
        coarse.state.joint_density,
    )
    convergence_ratio = step_distance / previous_step_distance
    metrics["step_size_8_vs_16_trace_distance"] = previous_step_distance
    metrics["step_size_16_vs_32_trace_distance"] = step_distance
    metrics["step_size_error_ratio"] = convergence_ratio
    checks["step_size_convergence"] = (
        step_distance <= limits.step_size_trace_distance
        and convergence_ratio <= limits.step_size_error_ratio
    )

    # 11. Fock-cutoff convergence on a low-energy physical trajectory.
    low_config = replace(convergence_base, cutoff=8, substeps_per_segment=12)
    high_config = replace(convergence_base, cutoff=12, substeps_per_segment=12)
    low_simulator = Phase9BackendASimulator(low_config)
    high_simulator = Phase9BackendASimulator(high_config)
    low_initial = low_simulator.initialize_fock(drift=convergence_drift)
    high_initial = high_simulator.initialize_fock(drift=convergence_drift)
    cutoff_exogenous_low = BackendAExogenous(
        emission_uniform=0.2,
        iq_standard_i=(0.0,) * low_config.iq_samples,
        iq_standard_q=(0.0,) * low_config.iq_samples,
        reset_uniform=0.2,
        reset_ack_uniform=0.2,
        drift_standard=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=56,
        round_index=0,
    )
    cutoff_exogenous_high = replace(cutoff_exogenous_low)
    low_result = low_simulator.step(
        low_initial,
        x_action,
        cutoff_exogenous_low,
    )
    high_result = high_simulator.step(
        high_initial,
        x_action,
        cutoff_exogenous_high,
    )
    cutoff_distance = _embedded_density_distance(
        low_simulator.oscillator_density(low_result.state.joint_density),
        high_simulator.oscillator_density(high_result.state.joint_density),
    )
    metrics["fock_cutoff_8_vs_12_trace_distance"] = cutoff_distance
    checks["fock_cutoff_convergence"] = (
        cutoff_distance <= limits.fock_cutoff_trace_distance
    )

    # 12. All six evaluator states initialize with exact logical fidelity.
    initial_fidelities: list[float] = []
    for label in ("0", "1", "+", "-", "+i", "-i"):
        state, truth = simulator.initialize_logical(label)
        record = simulator.logical_record(state, truth)
        initial_fidelities.append(record.target_fidelity)
    minimum_initial_fidelity = min(initial_fidelities)
    metrics["six_state_initial_minimum_fidelity"] = minimum_initial_fidelity
    checks["six_state_logical_projection"] = (
        minimum_initial_fidelity
        >= limits.six_state_initial_fidelity
    )

    claim_state = {
        "backend_b_qualified": None,
        "dual_backend_agreement": None,
        "round_ler": None,
        "six_state_lifetime": None,
        "physical_break_even": None,
        "official_puviani_exact": None,
        "puviani_nmf_surpass": None,
        "external_sota": None,
        "hardware_measured": None,
        "rank": None,
    }
    verdict = (
        "QUALIFIED_BACKEND_A_ONLY"
        if checks and all(checks.values())
        else "NO_GO_BACKEND_A_QUALIFICATION"
    )
    return BackendAQualification(
        backend_id=BACKEND_A_ID,
        scope=BACKEND_A_SCOPE,
        config_sha256=base.semantic_sha256(),
        metrics=metrics,
        checks=checks,
        claim_state=claim_state,
        verdict=verdict,
    )
