"""Numerical qualification gates for Phase-9 backend B."""

from __future__ import annotations

from dataclasses import replace
from math import exp, pi, sqrt

import numpy as np
from scipy.linalg import expm

from .models import (
    BackendBConfig,
    BackendBDrift,
    BackendBQualification,
    BackendBQualificationThresholds,
    BackendBRandomRecord,
    ComplexMatrix,
    _diagnostics,
    _trace_distance,
    backend_b_random_record,
)
from .simulator import Phase9BackendBSimulator, diagnostic_action_word_b


def _noise_free(config: BackendBConfig, **overrides: object) -> BackendBConfig:
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


def _zero_record(config: BackendBConfig, round_index: int = 0) -> BackendBRandomRecord:
    return BackendBRandomRecord(
        component_uniform=0.2,
        iq_normal_i=(0.0,) * config.iq_samples,
        iq_normal_q=(0.0,) * config.iq_samples,
        reset_uniform=0.2,
        ack_uniform=0.2,
        drift_normal=(0.0, 0.0, 0.0, 0.0, 0.0),
        seed=0,
        round_index=round_index,
    )


def _embedded_distance(
    low: ComplexMatrix,
    high: ComplexMatrix,
) -> float:
    embedded = np.zeros_like(high)
    embedded[: low.shape[0], : low.shape[1]] = low
    return _trace_distance(embedded, high)


def run_backend_b_qualification(
    config: BackendBConfig | None = None,
    thresholds: BackendBQualificationThresholds | None = None,
) -> BackendBQualification:
    base = BackendBConfig() if config is None else config
    limits = (
        BackendBQualificationThresholds()
        if thresholds is None
        else thresholds
    )
    if not isinstance(base, BackendBConfig):
        raise TypeError("config must be BackendBConfig")
    if not isinstance(limits, BackendBQualificationThresholds):
        raise TypeError(
            "thresholds must be BackendBQualificationThresholds"
        )
    metrics: dict[str, float | int | str | bool] = {}
    checks: dict[str, bool] = {}

    small_config = replace(base, cutoff=2, iq_samples=min(4, base.iq_samples))
    small = Phase9BackendBSimulator(small_config)
    hamiltonian = small._base_hamiltonian(BackendBDrift())
    choi_minimum, choi_tp, choi_hermiticity = small.split_channel_choi(
        hamiltonian,
        0.07,
    )
    metrics.update(
        {
            "choi_dimension": small.dimension,
            "choi_minimum_eigenvalue": choi_minimum,
            "choi_tp_frobenius": choi_tp,
            "choi_hermiticity_frobenius": choi_hermiticity,
        }
    )
    checks["split_channel_cp"] = (
        choi_minimum >= limits.choi_minimum_eigenvalue
    )
    checks["split_channel_tp"] = choi_tp <= limits.choi_tp_frobenius
    checks["split_channel_hermitian"] = (
        choi_hermiticity <= limits.choi_tp_frobenius
    )

    simulator = Phase9BackendBSimulator(base)
    initial, evaluator = simulator.initialize_logical("0")
    idle = diagnostic_action_word_b("IDLE")
    x_action = diagnostic_action_word_b("X")
    reset_action = diagnostic_action_word_b("RESET")
    record = backend_b_random_record(
        seed=701,
        round_index=0,
        iq_samples=base.iq_samples,
    )
    normal = simulator.step(initial, idle, record, evaluator=evaluator)
    data = _diagnostics(normal.state.joint_density)
    trace_error = abs(data["trace_real"] - 1.0) + abs(data["trace_imag"])
    metrics.update(
        {
            "full_round_trace_error": trace_error,
            "full_round_minimum_eigenvalue": data["minimum_eigenvalue"],
            "full_round_hermiticity_frobenius": data[
                "hermiticity_frobenius"
            ],
            "measurement_completeness": simulator.measurement_completeness_error(),
            "reset_completeness": simulator.reset_completeness_error(),
            "maximum_channel_completeness_error": max(
                simulator.channel_completeness_errors(0.03).values()
            ),
        }
    )
    checks["full_round_trace"] = trace_error <= limits.full_round_trace_error
    checks["full_round_positive"] = (
        data["minimum_eigenvalue"] >= limits.full_round_minimum_eigenvalue
    )
    checks["full_round_hermitian"] = (
        data["hermiticity_frobenius"] <= limits.full_round_trace_error
    )
    checks["analytic_channels_complete"] = (
        metrics["maximum_channel_completeness_error"]
        <= limits.instrument_completeness
    )
    checks["measurement_instrument_complete"] = (
        metrics["measurement_completeness"]
        <= limits.instrument_completeness
    )
    checks["reset_instrument_complete"] = (
        metrics["reset_completeness"] <= limits.instrument_completeness
    )

    # Closed-form references: pure loss mean photon and qutrit e relaxation.
    loss_config = _noise_free(
        base,
        oscillator_loss_rate=0.37,
        action_duration=0.06,
    )
    loss_simulator = Phase9BackendBSimulator(loss_config)
    ket_two = np.zeros(loss_config.cutoff, dtype=np.complex128)
    ket_two[2] = 1.0
    loss_state = loss_simulator.initialize_fock(oscillator_ket=ket_two)
    loss_time = 0.23
    loss_output = loss_simulator._noise_channels(
        loss_state.joint_density,
        loss_time,
    )
    loss_oscillator = loss_simulator.oscillator_density(loss_output)
    loss_mean = float(np.trace(loss_oscillator @ loss_simulator.number).real)
    loss_expected = 2.0 * exp(-0.37 * loss_time)
    loss_error = abs(loss_mean - loss_expected)
    relaxation_config = _noise_free(
        base,
        ancilla_ge_relax_rate=0.41,
    )
    relaxation_simulator = Phase9BackendBSimulator(relaxation_config)
    relaxation_state = relaxation_simulator.initialize_fock(
        ancilla_state="e"
    )
    relaxation_time = 0.19
    relaxation_output = relaxation_simulator._noise_channels(
        relaxation_state.joint_density,
        relaxation_time,
    )
    e_population = relaxation_simulator.level_probabilities(
        relaxation_output
    )[1]
    e_expected = exp(-0.41 * relaxation_time)
    relaxation_error = abs(e_population - e_expected)
    metrics["analytic_loss_mean_error"] = loss_error
    metrics["analytic_relaxation_population_error"] = relaxation_error
    checks["pure_loss_closed_form"] = (
        loss_error <= limits.analytic_loss_mean_error
    )
    checks["relaxation_closed_form"] = (
        relaxation_error <= limits.analytic_relaxation_error
    )

    ideal_config = _noise_free(base, split_steps_per_segment=64)
    ideal = Phase9BackendBSimulator(ideal_config)
    vacuum = ideal.initialize_fock()
    zero = ideal.step(vacuum, idle, _zero_record(ideal_config))
    zero_distance = _trace_distance(
        vacuum.joint_density,
        zero.state.joint_density,
    )
    acted = ideal.step(vacuum, x_action, _zero_record(ideal_config))
    alpha = ideal._action_alpha(x_action)
    displacement = expm(alpha * ideal.adag - alpha.conjugate() * ideal.a)
    expected = np.kron(
        displacement
        @ ideal.oscillator_density(vacuum.joint_density)
        @ displacement.conj().T,
        ideal.level_projectors[0],
    )
    ideal_distance = _trace_distance(expected, acted.state.joint_density)
    metrics["zero_noise_idle_trace_distance"] = zero_distance
    metrics["ideal_action_trace_distance"] = ideal_distance
    checks["zero_noise_idle_limit"] = zero_distance <= 1.0e-12
    checks["ideal_action_limit"] = (
        ideal_distance <= limits.ideal_action_trace_distance
    )

    success_config = _noise_free(
        base,
        reset_success_e=1.0,
        reset_success_f=1.0,
    )
    success_simulator = Phase9BackendBSimulator(success_config)
    f_initial = success_simulator.initialize_fock(ancilla_state="f")
    success = success_simulator.step(
        f_initial,
        reset_action,
        _zero_record(success_config),
    )
    success_g = success_simulator.level_probabilities(
        success.state.joint_density
    )[0]
    failure_config = _noise_free(
        base,
        reset_success_e=0.0,
        reset_success_f=0.0,
    )
    failure_simulator = Phase9BackendBSimulator(failure_config)
    failure_initial = failure_simulator.initialize_fock(ancilla_state="f")
    failed = failure_simulator.step(
        failure_initial,
        reset_action,
        _zero_record(failure_config),
    )
    persisted = failure_simulator.step(
        failure_initial,
        idle,
        _zero_record(failure_config),
    )
    failed_f = failure_simulator.level_probabilities(
        failed.state.joint_density
    )[2]
    persisted_f = failure_simulator.level_probabilities(
        persisted.state.joint_density
    )[2]
    metrics["large_reset_g_probability"] = success_g
    metrics["failed_reset_f_probability"] = failed_f
    metrics["no_reset_f_probability"] = persisted_f
    checks["large_reset_limit"] = (
        success_g >= limits.limit_population_minimum
        and success.truth.reset_hidden_outcome == "success"
    )
    checks["failed_reset_preserves_f"] = (
        failed_f >= limits.limit_population_minimum
        and failed.truth.reset_hidden_outcome == "failure"
    )
    checks["f_state_persistence"] = (
        persisted_f >= limits.limit_population_minimum
    )

    measurement_config = _noise_free(
        base,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
    )
    measurement_simulator = Phase9BackendBSimulator(measurement_config)
    plus_ge = np.array([1.0, 1.0, 0.0], dtype=np.complex128) / sqrt(2.0)
    coherent = measurement_simulator.initialize_fock(
        ancilla_state=plus_ge
    )
    measured = measurement_simulator.step(
        coherent,
        idle,
        _zero_record(measurement_config),
    )
    before_coherence = abs(
        measurement_simulator.ancilla_density(coherent.joint_density)[0, 1]
    )
    after_coherence = abs(
        measurement_simulator.ancilla_density(
            measured.state.joint_density
        )[0, 1]
    )
    posterior_peak = max(measured.observation.posterior_levels)
    metrics["measurement_coherence_ratio"] = float(
        after_coherence / before_coherence
    )
    metrics["measurement_posterior_peak"] = posterior_peak
    checks["iq_kraus_backaction"] = (
        posterior_peak > limits.measurement_posterior_peak
        and after_coherence < before_coherence * 1.0e-3
    )

    syndrome_config = _noise_free(
        base,
        ramsey_angle=pi / 2.0,
        ramsey_pulse_duration=0.03,
        sense_duration=0.8,
        dispersive_chi=1.0,
        iq_sigma=0.2,
        iq_centers=((-1.0, 0.0), (1.0, 0.0), (0.0, 1.5)),
        split_steps_per_segment=32,
    )
    syndrome = Phase9BackendBSimulator(syndrome_config)
    ket_zero = np.zeros(syndrome.cutoff, dtype=np.complex128)
    ket_one = np.zeros(syndrome.cutoff, dtype=np.complex128)
    ket_zero[0] = 1.0
    ket_one[1] = 1.0
    syndrome_record = _zero_record(syndrome_config)
    zero_result = syndrome.step(
        syndrome.initialize_fock(oscillator_ket=ket_zero),
        idle,
        syndrome_record,
    )
    one_result = syndrome.step(
        syndrome.initialize_fock(oscillator_ket=ket_one),
        idle,
        syndrome_record,
    )
    level_tv = 0.5 * float(
        np.sum(
            np.abs(
                np.asarray(zero_result.truth.pre_measurement_levels)
                - np.asarray(one_result.truth.pre_measurement_levels)
            )
        )
    )
    superposition = (ket_zero + ket_one) / sqrt(2.0)
    super_state = syndrome.initialize_fock(oscillator_ket=superposition)
    super_result = syndrome.step(
        super_state,
        idle,
        syndrome_record,
    )
    syndrome_backaction = _trace_distance(
        syndrome.oscillator_density(super_state.joint_density),
        syndrome.oscillator_density(super_result.state.joint_density),
    )
    metrics["syndrome_fock0_vs_fock1_level_tv"] = level_tv
    metrics["syndrome_oscillator_backaction_trace_distance"] = (
        syndrome_backaction
    )
    checks["ramsey_syndrome_state_dependence"] = (
        level_tv > limits.syndrome_state_dependence_minimum
    )
    checks["syndrome_backacts_on_oscillator"] = (
        syndrome_backaction > limits.syndrome_backaction_minimum
    )

    leakage_config = _noise_free(
        base,
        action_leakage_coupling=0.8,
        action_duration=0.1,
        split_steps_per_segment=64,
    )
    leakage = Phase9BackendBSimulator(leakage_config)
    e_initial = leakage.initialize_fock(ancilla_state="e")
    leakage_idle = leakage.step(
        e_initial,
        idle,
        _zero_record(leakage_config),
    )
    leakage_x = leakage.step(
        e_initial,
        x_action,
        _zero_record(leakage_config),
    )
    f_difference = (
        leakage.level_probabilities(leakage_x.state.joint_density)[2]
        - leakage.level_probabilities(leakage_idle.state.joint_density)[2]
    )
    metrics["action_induced_f_population_difference"] = f_difference
    checks["action_induces_f_population"] = (
        f_difference > limits.action_f_population_minimum
    )

    intervention_config = replace(
        base,
        ramsey_angle=0.0,
        sense_duration=0.0,
        iq_centers=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    intervention = Phase9BackendBSimulator(intervention_config)
    intervention_initial = intervention.initialize_fock()
    common_record = backend_b_random_record(
        seed=801,
        round_index=0,
        iq_samples=intervention_config.iq_samples,
    )
    idle_result = intervention.step(
        intervention_initial,
        idle,
        common_record,
    )
    x_result = intervention.step(
        intervention_initial,
        x_action,
        common_record,
    )
    state_distance = _trace_distance(
        idle_result.state.joint_density,
        x_result.state.joint_density,
    )
    drift_distance = float(
        np.linalg.norm(
            idle_result.state.drift.vector()
            - x_result.state.drift.vector()
        )
    )
    metrics["action_intervention_state_trace_distance"] = state_distance
    metrics["action_intervention_drift_l2"] = drift_distance
    metrics["action_intervention_shared_random_record"] = (
        idle_result.random_record == x_result.random_record
    )
    checks["action_changes_quantum_state"] = (
        state_distance > limits.action_state_distance_minimum
    )
    checks["action_changes_drift"] = (
        drift_distance > limits.action_drift_minimum
    )
    checks["common_random_record_intervention"] = (
        idle_result.random_record == x_result.random_record
    )

    actions = (idle, x_action, idle)
    replay_one = simulator.simulate(initial, actions, seed=901, evaluator=evaluator)
    replay_two = simulator.simulate(initial, actions, seed=901, evaluator=evaluator)
    other = simulator.simulate(initial, actions, seed=902, evaluator=evaluator)
    replay_density_error = float(
        np.max(
            np.abs(
                replay_one.final_state.joint_density
                - replay_two.final_state.joint_density
            )
        )
    )
    replay_iq_error = max(
        float(
            np.max(
                np.abs(left.observation.iq_i - right.observation.iq_i)
            )
        )
        for left, right in zip(replay_one.rounds, replay_two.rounds)
    )
    seed_difference = float(
        np.max(
            np.abs(
                replay_one.rounds[0].observation.iq_i
                - other.rounds[0].observation.iq_i
            )
        )
    )
    metrics["seed_replay_density_error"] = replay_density_error
    metrics["seed_replay_iq_error"] = replay_iq_error
    metrics["different_seed_iq_difference"] = seed_difference
    checks["seed_determinism"] = (
        replay_density_error == 0.0 and replay_iq_error == 0.0
    )
    checks["seed_sensitivity"] = (
        seed_difference > limits.rng_sensitivity_minimum
    )

    convergence_base = _noise_free(
        base,
        dispersive_chi=0.23,
        self_kerr=0.015,
        ramsey_angle=0.0,
        sense_duration=0.0,
        drift_action_kick=0.0,
    )
    convergence_drift = BackendBDrift(drive_q=0.07, drive_p=-0.04)
    convergence_record = _zero_record(convergence_base)
    outputs: dict[int, ComplexMatrix] = {}
    for steps in (8, 16, 32):
        active_config = replace(
            convergence_base,
            split_steps_per_segment=steps,
        )
        active = Phase9BackendBSimulator(active_config)
        active_state = active.initialize_fock(drift=convergence_drift)
        outputs[steps] = active.step(
            active_state,
            x_action,
            convergence_record,
        ).state.joint_density
    distance_8_16 = _trace_distance(outputs[8], outputs[16])
    distance_16_32 = _trace_distance(outputs[16], outputs[32])
    ratio = distance_16_32 / distance_8_16
    metrics["split_8_vs_16_trace_distance"] = distance_8_16
    metrics["split_16_vs_32_trace_distance"] = distance_16_32
    metrics["split_error_ratio"] = ratio
    checks["split_step_convergence"] = (
        distance_16_32 <= limits.split_distance
        and ratio <= limits.split_ratio
    )

    cutoff_outputs: dict[int, ComplexMatrix] = {}
    for cutoff in (8, 12):
        active_config = replace(
            convergence_base,
            cutoff=cutoff,
            split_steps_per_segment=24,
        )
        active = Phase9BackendBSimulator(active_config)
        active_state = active.initialize_fock(drift=convergence_drift)
        cutoff_outputs[cutoff] = active.oscillator_density(
            active.step(
                active_state,
                x_action,
                _zero_record(active_config),
            ).state.joint_density
        )
    cutoff_distance = _embedded_distance(
        cutoff_outputs[8],
        cutoff_outputs[12],
    )
    metrics["fock_cutoff_8_vs_12_trace_distance"] = cutoff_distance
    checks["fock_cutoff_convergence"] = (
        cutoff_distance <= limits.cutoff_distance
    )

    fidelities: list[float] = []
    for label in ("0", "1", "+", "-", "+i", "-i"):
        state, truth = simulator.initialize_logical(label)
        fidelities.append(simulator.logical_record(state, truth).target_fidelity)
    minimum_fidelity = min(fidelities)
    metrics["six_state_initial_minimum_fidelity"] = minimum_fidelity
    checks["independent_six_state_logical_projection"] = (
        minimum_fidelity >= limits.six_state_initial_fidelity
    )

    claim_state = {
        "backend_a_b_agreement": None,
        "dual_backend_qualified": None,
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
        "QUALIFIED_BACKEND_B_ONLY"
        if checks and all(checks.values())
        else "NO_GO_BACKEND_B_QUALIFICATION"
    )
    return BackendBQualification(
        config_sha256=base.semantic_sha256(),
        metrics=metrics,
        checks=checks,
        claim_state=claim_state,
        verdict=verdict,
    )

