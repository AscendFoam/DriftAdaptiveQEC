from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import cnn_fpga.benchmark.memory_assisted_bayesian_decoder as module
from cnn_fpga.benchmark.memory_assisted_bayesian_decoder import (
    BayesianObservationBudget,
    MEMORY_BAYES_DESCRIPTOR,
    MODEL_SCOPE,
    PeriodicBayesConfig,
    bayesian_cost_profile,
    bayesian_validation_scenarios,
    decode_observed_episode,
    final_outcome_static_bayes_decode,
    periodic_memory_bayes_decode,
    validate_memory_bayesian_registration,
)
from physics.constants import LATTICE_CONST
from physics.drift_processes import DriftState
from physics.syndrome_stream import SyndromeStreamConfig, generate_syndrome_stream


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_2_1_memory_bayesian_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_2_1_memory_bayesian_source_data.csv"


def _config(
    *,
    grid_size: int = 64,
    history_cycles: int = 8,
    mean: tuple[float, float] = (0.0, 0.0),
    sigma_q: float = 0.16,
    sigma_p: float = 0.20,
    rho: float = 0.45,
    measurement_sigma: float = 0.10,
) -> PeriodicBayesConfig:
    sq = sigma_q * LATTICE_CONST
    sp = sigma_p * LATTICE_CONST
    sm = measurement_sigma * LATTICE_CONST
    return PeriodicBayesConfig(
        grid_size=grid_size,
        process_mean=(mean[0] * LATTICE_CONST, mean[1] * LATTICE_CONST),
        process_covariance=((sq * sq, rho * sq * sp), (rho * sq * sp, sp * sp)),
        measurement_covariance=((sm * sm, 0.0), (0.0, sm * sm)),
        observation_budget=BayesianObservationBudget(history_cycles=history_cycles),
    )


def _implementation_hash() -> str:
    paths = (
        ROOT / "cnn_fpga" / "benchmark" / "memory_assisted_bayesian_decoder.py",
        ROOT / "cnn_fpga" / "benchmark" / "standard_binning_baseline.py",
        ROOT / "physics" / "syndrome_stream.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_descriptor_and_budget_are_observed_only_and_paper_bounded() -> None:
    budget = BayesianObservationBudget()
    assert MEMORY_BAYES_DESCRIPTOR.baseline_id == "periodic_memory_assisted_bayes"
    assert MEMORY_BAYES_DESCRIPTOR.evidence_scope == MODEL_SCOPE
    assert MEMORY_BAYES_DESCRIPTOR.exact_paper_reproduction is False
    assert budget.hidden_truth_inputs == ()
    assert budget.consumed_per_cycle_fields == ("residual_q", "residual_p")
    assert set(budget.consumed_per_cycle_fields) < set(budget.available_per_cycle_fields)
    assert "finite_energy_Glancy_Knill_circuit_fidelity_reproduction" in (
        MEMORY_BAYES_DESCRIPTOR.excluded_claims
    )


def test_budget_and_config_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        BayesianObservationBudget(history_cycles=1)
    with pytest.raises(ValueError, match="subset"):
        BayesianObservationBudget(consumed_per_cycle_fields=("truth",))
    with pytest.raises(ValueError, match="hidden truth"):
        BayesianObservationBudget(hidden_truth_inputs=("logical_truth",))
    for call in (
        lambda: PeriodicBayesConfig(lattice=0.0),
        lambda: PeriodicBayesConfig(grid_size=33),
        lambda: PeriodicBayesConfig(grid_size=514),
        lambda: PeriodicBayesConfig(process_mean=(0.0,)),
        lambda: PeriodicBayesConfig(process_covariance=((1.0, 2.0), (0.0, 1.0))),
        lambda: PeriodicBayesConfig(process_covariance=((1.0, 1.0), (1.0, 1.0))),
        lambda: PeriodicBayesConfig(tail_sigma=4.9),
        lambda: PeriodicBayesConfig(model_scope="device"),
        lambda: PeriodicBayesConfig(
            grid_size=32,
            measurement_covariance=((1.0e-12, 0.0), (0.0, 1.0e-12)),
        ),
    ):
        with pytest.raises((TypeError, ValueError)):
            call()


def test_fft_prediction_matches_explicit_circular_convolution() -> None:
    grid = module._PeriodicBayesGrid(_config(grid_size=32))
    rng = np.random.default_rng(11)
    posterior = rng.random((1, grid.size, grid.size))
    posterior /= posterior.sum(axis=(-2, -1), keepdims=True)
    actual = grid.predict(posterior)[0]
    expected = np.zeros_like(actual)
    for q_shift in range(grid.size):
        for p_shift in range(grid.size):
            expected += grid.transition_kernel[q_shift, p_shift] * np.roll(
                posterior[0], (q_shift, p_shift), axis=(0, 1)
            )
    expected /= expected.sum()
    assert np.allclose(actual, expected, rtol=2.0e-13, atol=2.0e-15)


def test_shifted_likelihood_template_matches_direct_wrapped_gaussian() -> None:
    config = _config(grid_size=64)
    grid = module._PeriodicBayesGrid(config)
    observation = np.asarray([[3.0 * grid.grid_step, -2.0 * grid.grid_step]])
    actual = grid.likelihood(observation)[0]
    expected = module._periodic_gaussian_values(
        observation[0, 0] - grid.q_grid,
        observation[0, 1] - grid.p_grid,
        np.asarray(config.measurement_covariance),
        period=config.lattice,
        tail_sigma=config.tail_sigma,
    )
    expected /= expected.max()
    assert np.allclose(actual, expected, rtol=3.0e-13, atol=2.0e-15)


def test_one_round_memory_and_static_use_identical_prior_and_likelihood() -> None:
    config = _config()
    observations = np.asarray([[0.13, -0.27]])
    memory = periodic_memory_bayes_decode(observations, config)
    static = final_outcome_static_bayes_decode(observations[-1], 1, config)
    assert np.allclose(memory.posterior_mass, static.posterior_mass, atol=2.0e-15)
    assert np.allclose(memory.logical_posterior, static.logical_posterior, atol=2.0e-15)


def test_early_history_changes_memory_posterior_but_not_final_outcome_static() -> None:
    config = _config(history_cycles=8)
    first = np.zeros((8, 2), dtype=np.float64)
    second = first.copy()
    first[:5, 0] = 0.32 * config.lattice
    second[:5, 0] = -0.32 * config.lattice
    first[-1] = second[-1] = (0.07 * config.lattice, -0.04 * config.lattice)
    memory_a = periodic_memory_bayes_decode(first, config)
    memory_b = periodic_memory_bayes_decode(second, config)
    static_a = final_outcome_static_bayes_decode(first[-1], 8, config)
    static_b = final_outcome_static_bayes_decode(second[-1], 8, config)
    assert not np.allclose(memory_a.logical_posterior, memory_b.logical_posterior, atol=1.0e-4)
    assert np.array_equal(static_a.logical_posterior, static_b.logical_posterior)


def test_batch_and_individual_decoding_are_identical() -> None:
    config = _config(history_cycles=8)
    rng = np.random.default_rng(12)
    history = rng.uniform(-config.lattice / 2.0, config.lattice / 2.0, (5, 8, 2))
    batched = periodic_memory_bayes_decode(history, config)
    individual = [periodic_memory_bayes_decode(item, config) for item in history]
    assert np.allclose(
        batched.logical_posterior,
        np.concatenate([item.logical_posterior for item in individual]),
        atol=2.0e-15,
    )
    assert np.array_equal(
        batched.logical_class,
        np.concatenate([item.logical_class for item in individual]),
    )


def test_axis_swap_preserves_joint_correlated_bayesian_result() -> None:
    config = _config(mean=(0.03, -0.05), sigma_q=0.14, sigma_p=0.20, rho=-0.55)
    swapped = _config(mean=(-0.05, 0.03), sigma_q=0.20, sigma_p=0.14, rho=-0.55)
    history = np.asarray(
        [[0.1, -0.2], [0.3, 0.12], [-0.4, 0.2], [0.05, -0.08], [0.11, 0.06], [0.2, -0.1], [-0.3, 0.09], [0.04, 0.02]]
    )
    direct = periodic_memory_bayes_decode(history, config).logical_posterior[0]
    reverse = periodic_memory_bayes_decode(history[:, ::-1], swapped).logical_posterior[0]
    assert np.allclose(reverse, direct[[0, 2, 1, 3]], rtol=2.0e-12, atol=2.0e-14)


def test_correlation_is_not_silently_factorized() -> None:
    correlated = _config(rho=0.75)
    independent = _config(rho=0.0)
    history = np.asarray(
        [[0.32, 0.31], [0.39, 0.36], [-0.41, -0.38], [-0.33, -0.29], [0.28, 0.34], [0.42, 0.40], [-0.37, -0.35], [0.3, 0.27]]
    ) * correlated.lattice
    a = periodic_memory_bayes_decode(history, correlated).logical_posterior
    b = periodic_memory_bayes_decode(history, independent).logical_posterior
    assert np.max(np.abs(a - b)) > 0.02


def test_history_input_validation_and_budget_enforcement() -> None:
    config = _config(history_cycles=8)
    with pytest.raises(ValueError, match="shape"):
        periodic_memory_bayes_decode(np.zeros((8, 3)), config)
    with pytest.raises(ValueError, match="at least one"):
        periodic_memory_bayes_decode(np.empty((0, 2)), config)
    with pytest.raises(ValueError, match="history budget"):
        periodic_memory_bayes_decode(np.zeros((9, 2)), config)
    with pytest.raises(ValueError, match="half-open"):
        periodic_memory_bayes_decode(np.full((2, 2), config.lattice / 2.0), config)
    bad = np.zeros((2, 2))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        periodic_memory_bayes_decode(bad, config)


def test_observed_stream_adapter_matches_array_api_and_rejects_truth_steps() -> None:
    sigma_q = 0.15 * LATTICE_CONST
    sigma_p = 0.18 * LATTICE_CONST
    rho = 0.35
    states = tuple(
        DriftState(
            step=index,
            time=float(index),
            sigma_q=sigma_q,
            sigma_p=sigma_p,
            rho=rho,
            source="t3.2.1-test",
            regime="stationary",
        )
        for index in range(8)
    )
    measurement_sigma = 0.10 * LATTICE_CONST
    stream = generate_syndrome_stream(
        states,
        config=SyndromeStreamConfig(
            measurement_sigma=(measurement_sigma, measurement_sigma),
            loss_environment_variance=0.0,
            depth_probability_scale=0.0,
            recovery_probability=0.0,
            recovery_gain=0.0,
            base_leakage_probability=0.0,
            loss_leakage_scale=0.0,
            burst_leakage_bonus=0.0,
            readout_fidelity_g=1.0,
            readout_fidelity_e=1.0,
            seed=909,
        ),
    )
    config = PeriodicBayesConfig(
        grid_size=64,
        process_covariance=((sigma_q**2, rho * sigma_q * sigma_p), (rho * sigma_q * sigma_p, sigma_p**2)),
        measurement_covariance=((measurement_sigma**2, 0.0), (0.0, measurement_sigma**2)),
        observation_budget=BayesianObservationBudget(history_cycles=8),
    )
    observed = tuple(step.observed for step in stream.steps)
    adapted = decode_observed_episode(observed, config)
    direct = periodic_memory_bayes_decode(
        np.asarray([item.residual_syndrome for item in observed]), config
    )
    assert np.array_equal(adapted.logical_posterior, direct.logical_posterior)
    with pytest.raises(TypeError, match="ObservedSyndromeStep"):
        decode_observed_episode(stream.steps, config)  # type: ignore[arg-type]


def test_observed_adapter_rejects_invalid_order_and_analog_residual_mismatch() -> None:
    sigma = 0.15 * LATTICE_CONST
    states = tuple(DriftState(step=index, time=float(index), sigma_q=sigma, sigma_p=sigma) for index in range(3))
    stream = generate_syndrome_stream(
        states,
        config=SyndromeStreamConfig(
            measurement_sigma=(0.1 * LATTICE_CONST, 0.1 * LATTICE_CONST),
            depth_probability_scale=0.0,
            recovery_probability=0.0,
            base_leakage_probability=0.0,
            loss_leakage_scale=0.0,
            burst_leakage_bonus=0.0,
            readout_fidelity_g=1.0,
            readout_fidelity_e=1.0,
            seed=18,
        ),
    )
    observed = [step.observed for step in stream.steps]
    with pytest.raises(ValueError, match="consecutive"):
        decode_observed_episode((observed[0], observed[2]), _config())
    mutated = module.ObservedSyndromeStep(
        **{**observed[0].__dict__, "residual_syndrome": (0.0, 0.0)}
    )
    with pytest.raises(ValueError, match="does not wrap"):
        decode_observed_episode((mutated,), _config())


def test_cost_profile_is_deterministic_and_keeps_hardware_fields_null() -> None:
    profile = bayesian_cost_profile(_config(grid_size=128, history_cycles=20))
    assert profile.logical_torus_cells == 16_384
    assert profile.raw_history_storage_bits == 960
    assert profile.posterior_storage_bits == 393_216
    assert profile.transition_kernel_storage_bits == 393_216
    assert profile.complex_fft_butterfly_proxy_per_cycle == 458_752
    assert profile.transition_kernel_quadratic_forms_once == 147_456
    assert profile.likelihood_template_quadratic_forms_once == 147_456
    assert profile.likelihood_observation_quantizations_per_cycle == 2
    assert profile.likelihood_table_lookups_per_cycle == 16_384
    assert profile.target_lut is None
    assert profile.target_bram is None
    assert profile.target_dsp is None
    assert profile.target_fmax_hz is None
    assert profile.target_measured is False


def test_t321_comparison_is_registered_with_standard_anchor() -> None:
    gates = validate_memory_bayesian_registration()
    assert "registry:t3_2_1_memory_bayesian_episode_comparison" in gates


def test_production_artifact_is_source_bound_and_non_demo_scale() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_hash()
    assert payload["aggregate"]["episodes"] == 4096
    assert payload["aggregate"]["cycles"] == 81_920
    assert payload["aggregate"]["source_data_rows"] == 32
    assert payload["gate_summary"]["passed"] == 9
    assert payload["gate_summary"]["failed"] == 0
    assert payload["aggregate"]["max_grid_reference_tv_mean"] <= 0.025
    assert payload["aggregate"]["max_grid_reference_error_rate_delta_abs"] <= 0.025
    assert payload["cost_profile"]["target_measured"] is False


def test_source_data_recomputes_scenario_and_aggregate_seed_cluster_gains() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 32
    assert len({row["trace_sha256"] for row in rows}) == 32
    for scenario in payload["scenarios"]:
        selected = [row for row in rows if row["scenario_id"] == scenario["scenario"]["scenario_id"]]
        gains = np.asarray([float(row["static_minus_memory_error_rate"]) for row in selected])
        assert len(selected) == 8
        assert np.mean(gains) == pytest.approx(
            scenario["static_minus_memory_seed_cluster_ci"]["estimate"], abs=1.0e-15
        )
        assert scenario["static_minus_memory_seed_cluster_ci"]["interval_method"] == (
            "two_sided_student_t_cluster_mean"
        )
        assert scenario["static_minus_memory_seed_cluster_ci"]["degrees_freedom"] == 7
        assert all(float(row["memory_nll"]) < float(row["static_nll"]) for row in selected)
        reference_rows = [row for row in selected if row["grid_reference_evaluated"] == "True"]
        assert len(reference_rows) == 1
    seed_suffixes = sorted({int(row["evaluation_seed"]) % 10_000 for row in rows})
    aggregate_seed_means = []
    for suffix in seed_suffixes:
        selected = [row for row in rows if int(row["evaluation_seed"]) % 10_000 == suffix]
        aggregate_seed_means.append(
            np.mean([float(row["static_minus_memory_error_rate"]) for row in selected])
        )
    assert np.mean(aggregate_seed_means) == pytest.approx(
        payload["aggregate"]["static_minus_memory_seed_cluster_ci"]["estimate"],
        abs=1.0e-15,
    )
    assert payload["aggregate"]["static_minus_memory_seed_cluster_ci"][
        "interval_method"
    ] == "two_sided_student_t_cluster_mean"


def test_every_scenario_has_resolved_gain_and_better_proper_scores() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    for scenario in payload["scenarios"]:
        assert scenario["static_minus_memory_seed_cluster_ci"]["ci_low"] > 0.0
        assert scenario["memory_nll"] < scenario["static_nll"]
        assert scenario["memory_brier"] < scenario["static_brier"]
    aggregate = payload["aggregate"]["static_minus_memory_seed_cluster_ci"]
    assert aggregate["estimate"] == pytest.approx(0.303466796875)
    assert aggregate["ci_low"] > 0.29


def test_claim_boundary_rejects_wan_device_and_fpga_overclaim() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    forbidden = payload["claim_boundary"]["forbidden"]
    assert "Wan finite-energy circuit fidelity" in forbidden
    assert "device calibration" in forbidden
    assert "FPGA synthesis" in forbidden
    assert payload["paper_provenance"]["not_transferred"].startswith(
        "finite-energy wavefunction"
    )
