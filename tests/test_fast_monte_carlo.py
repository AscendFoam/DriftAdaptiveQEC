from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pytest

from physics.constants import LATTICE_CONST
from physics.drift_processes import ConstantDriftProcess, DriftState
from physics.fast_monte_carlo import (
    MODEL_SCOPE,
    FastMonteCarloConfig,
    RareEventSpec,
    run_fast_monte_carlo,
    write_fast_monte_carlo_report,
)
from physics.ideal_gkp_decoder import gaussian_logical_flip_probability
from physics.sbs_error_space import SBS_PROTOCOL_ID


def _quiet_state(**updates: object) -> DriftState:
    values: dict[str, object] = {
        "sigma_q": 1.0e-12,
        "sigma_p": 1.0e-12,
        "source": "unit-test",
        "regime": "quiet",
    }
    values.update(updates)
    return DriftState(**values)


def _config(**updates: object) -> FastMonteCarloConfig:
    values: dict[str, object] = {
        "n_trajectories": 400,
        "rounds_per_trajectory": 10,
        "loss_environment_variance": 0.0,
        "depth_probability_scale": 0.0,
        "recovery_probability": 0.0,
        "recovery_gain": 0.0,
        "base_leakage_probability": 0.0,
        "loss_leakage_scale": 0.0,
        "burst_leakage_bonus": 0.0,
        "higher_leakage_fraction": 0.0,
        "leakage_logical_fault_probability": 0.0,
        "bootstrap_replicates": 200,
        "seed": 13,
    }
    values.update(updates)
    return FastMonteCarloConfig(**values)


def _stable_payload(result: object) -> dict:
    payload = result.to_dict()  # type: ignore[attr-defined]
    payload.pop("performance")
    return payload


def test_config_fails_closed_on_invalid_workload_probabilities_and_seed() -> None:
    invalid = [
        lambda: FastMonteCarloConfig(n_trajectories=3),
        lambda: FastMonteCarloConfig(rounds_per_trajectory=0),
        lambda: FastMonteCarloConfig(n_trajectories=1_000_001, rounds_per_trajectory=100),
        lambda: FastMonteCarloConfig(lattice=0.0),
        lambda: FastMonteCarloConfig(loss_environment_variance=-1.0),
        lambda: FastMonteCarloConfig(max_recovery_depth=0),
        lambda: FastMonteCarloConfig(depth_probability_scale=1.1),
        lambda: FastMonteCarloConfig(depth_probability_power=0.0),
        lambda: FastMonteCarloConfig(loss_leakage_scale=-0.1),
        lambda: FastMonteCarloConfig(higher_leakage_mean_duration=1.9),
        lambda: FastMonteCarloConfig(confidence_level=1.0),
        lambda: FastMonteCarloConfig(bootstrap_replicates=199),
        lambda: FastMonteCarloConfig(seed=2**64),
        lambda: FastMonteCarloConfig(model_scope="device"),
    ]
    for call in invalid:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_rare_event_spec_fails_closed() -> None:
    invalid = [
        lambda: RareEventSpec(kind="unknown"),
        lambda: RareEventSpec(true_trajectory_probability=0.0),
        lambda: RareEventSpec(allocation_fraction=1.0),
        lambda: RareEventSpec(mean_duration_cycles=0.9),
        lambda: RareEventSpec(displacement_scale=0.9),
        lambda: RareEventSpec(mean_shift=(0.0,)),
        lambda: RareEventSpec(extra_loss_gamma=-0.1),
        lambda: RareEventSpec(forced_leakage_fault_probability=1.1),
    ]
    for call in invalid:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_source_contract_accepts_process_or_exact_round_sequence() -> None:
    config = _config(rounds_per_trajectory=4)
    process = ConstantDriftProcess(base=_quiet_state(), seed=5)
    by_process = run_fast_monte_carlo(process, config=config)
    by_sequence = run_fast_monte_carlo(process.generate(4), config=config)
    assert by_process.config.total_cycles == by_sequence.config.total_cycles == 1_600
    with pytest.raises(ValueError, match="length"):
        run_fast_monte_carlo(process.generate(3), config=config)
    with pytest.raises(TypeError, match="source"):
        run_fast_monte_carlo("bad", config=config)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="produce"):
        run_fast_monte_carlo([_quiet_state(), object(), _quiet_state(), _quiet_state()], config=config)  # type: ignore[list-item]
    with pytest.raises(TypeError, match="config"):
        run_fast_monte_carlo(process, config=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="rare_event"):
        run_fast_monte_carlo(process, config=config, rare_event=object())  # type: ignore[arg-type]


def test_result_outputs_probability_confidence_interval_seed_and_scope() -> None:
    config = _config(n_trajectories=1_000, rounds_per_trajectory=100, seed=77)
    result = run_fast_monte_carlo([DriftState()] * 100, config=config)
    assert result.config.total_cycles == 100_000
    assert result.config.seed == 77
    assert 0.0 <= result.logical_error_probability <= 1.0
    assert 0.0 <= result.ci_low <= result.logical_error_probability <= result.ci_high <= 1.0
    assert result.confidence_level == pytest.approx(0.95)
    assert "trajectory_cluster" in result.ci_method
    assert result.protocol_id == SBS_PROTOCOL_ID
    assert result.model_scope == MODEL_SCOPE
    assert result.device_calibrated is False
    payload = result.to_dict()
    assert payload["performance"]["at_least_1e5_cycles"] is True
    assert payload["performance"]["one_million_cycle_target_met"] is False


def test_one_million_cycle_target_runs_vectorized_and_reports_throughput() -> None:
    config = _config(
        n_trajectories=10_000,
        rounds_per_trajectory=100,
        bootstrap_replicates=200,
        seed=101,
    )
    result = run_fast_monte_carlo([DriftState()] * 100, config=config)
    assert result.config.total_cycles == 1_000_000
    assert result.to_dict()["performance"]["one_million_cycle_target_met"] is True
    assert result.elapsed_seconds < 30.0
    assert result.cycles_per_second > 20_000.0


def test_same_seed_replays_all_statistical_outputs_and_changed_seed_differs() -> None:
    states = [DriftState(sigma_q=0.4, sigma_p=0.5, rho=0.3)] * 12
    config = _config(n_trajectories=800, rounds_per_trajectory=12, seed=123)
    first = run_fast_monte_carlo(states, config=config)
    second = run_fast_monte_carlo(states, config=config)
    changed = run_fast_monte_carlo(states, config=_config(n_trajectories=800, rounds_per_trajectory=12, seed=124))
    assert _stable_payload(first) == _stable_payload(second)
    assert first.logical_error_probability != changed.logical_error_probability


def test_vectorized_single_round_matches_independent_gaussian_analytic_rate() -> None:
    sigma = 0.30 * LATTICE_CONST
    n = 200_000
    config = _config(n_trajectories=n, rounds_per_trajectory=1, seed=91)
    result = run_fast_monte_carlo(
        [DriftState(sigma_q=sigma, sigma_p=sigma, rho=0.0)],
        config=config,
    )
    per_axis = gaussian_logical_flip_probability(sigma)
    expected = 1.0 - (1.0 - per_axis) ** 2
    standard_error = math.sqrt(expected * (1.0 - expected) / n)
    assert abs(result.logical_error_probability - expected) < 5.0 * standard_error
    assert abs(result.q_error_probability - per_axis) < 5.0 * math.sqrt(
        per_axis * (1.0 - per_axis) / n
    )


def test_zero_event_run_has_nonzero_conservative_upper_bound() -> None:
    config = _config(n_trajectories=1_000, rounds_per_trajectory=10)
    result = run_fast_monte_carlo([_quiet_state()] * 10, config=config)
    assert result.logical_error_probability == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high > 0.0
    assert result.zero_event_trajectory_upper_bound > 0.0


def test_stratified_burst_estimator_uses_target_weight_not_raw_allocation() -> None:
    rounds = 10
    probability = 0.002
    config = _config(n_trajectories=2_000, rounds_per_trajectory=rounds, seed=4)
    rare = RareEventSpec(
        kind="burst",
        true_trajectory_probability=probability,
        allocation_fraction=0.5,
        mean_duration_cycles=1.0,
        displacement_scale=1.0,
        mean_shift=(LATTICE_CONST, 0.0),
        forced_leakage_fault_probability=0.0,
    )
    result = run_fast_monte_carlo([_quiet_state()] * rounds, config=config, rare_event=rare)
    assert result.logical_error_probability == pytest.approx(probability / rounds, abs=1.0e-12)
    assert len(result.strata) == 2
    assert result.strata[1].conditional_cycle_error_rate == pytest.approx(1.0 / rounds)
    raw_fraction = sum(item.logical_event_count for item in result.strata) / config.total_cycles
    assert raw_fraction == pytest.approx(0.5 / rounds)
    assert raw_fraction > 100.0 * result.logical_error_probability


def test_stratified_estimate_is_invariant_to_sampling_allocation_for_deterministic_conditionals() -> None:
    states = [_quiet_state()] * 8
    base = dict(
        kind="burst",
        true_trajectory_probability=0.01,
        mean_duration_cycles=1.0,
        displacement_scale=1.0,
        mean_shift=(LATTICE_CONST, 0.0),
        forced_leakage_fault_probability=0.0,
    )
    low = run_fast_monte_carlo(
        states,
        config=_config(n_trajectories=1_000, rounds_per_trajectory=8),
        rare_event=RareEventSpec(allocation_fraction=0.1, **base),
    )
    high = run_fast_monte_carlo(
        states,
        config=_config(n_trajectories=1_000, rounds_per_trajectory=8),
        rare_event=RareEventSpec(allocation_fraction=0.8, **base),
    )
    assert low.logical_error_probability == pytest.approx(0.01 / 8.0)
    assert high.logical_error_probability == pytest.approx(low.logical_error_probability)
    assert low.strata[1].n_trajectories != high.strata[1].n_trajectories


def test_stratified_leakage_mode_estimates_forced_leakage_faults() -> None:
    rounds = 20
    rare = RareEventSpec(
        kind="leakage",
        true_trajectory_probability=0.005,
        allocation_fraction=0.4,
        mean_duration_cycles=1.0,
        displacement_scale=1.0,
        forced_leakage_fault_probability=1.0,
    )
    result = run_fast_monte_carlo(
        [_quiet_state()] * rounds,
        config=_config(n_trajectories=1_000, rounds_per_trajectory=rounds),
        rare_event=rare,
    )
    assert result.logical_error_probability == pytest.approx(0.005 / rounds)
    assert result.strata[1].leakage_cycle_count == result.strata[1].n_trajectories


def test_burst_and_leakage_strata_have_normalized_target_weights_and_exact_cycle_budget() -> None:
    config = _config(n_trajectories=777, rounds_per_trajectory=9)
    rare = RareEventSpec(true_trajectory_probability=3.0e-4, allocation_fraction=0.23)
    result = run_fast_monte_carlo([DriftState()] * 9, config=config, rare_event=rare)
    assert sum(item.target_weight for item in result.strata) == pytest.approx(1.0)
    assert sum(item.n_trajectories for item in result.strata) == config.n_trajectories
    assert sum(item.simulated_cycles for item in result.strata) == config.total_cycles
    assert result.strata[1].burst_cycle_count > 0
    assert result.strata[1].leakage_cycle_count > 0


def test_background_leakage_fault_probability_enters_logical_event_rate() -> None:
    config = _config(
        n_trajectories=500,
        rounds_per_trajectory=5,
        base_leakage_probability=1.0,
        leakage_logical_fault_probability=1.0,
    )
    result = run_fast_monte_carlo([_quiet_state()] * 5, config=config)
    assert result.logical_error_probability == 1.0
    assert result.strata[0].logical_event_count == config.total_cycles
    assert result.strata[0].leakage_cycle_count == config.total_cycles


def test_multiround_recovery_state_reduces_later_logical_events() -> None:
    states = [_quiet_state(mu_q=0.20 * LATTICE_CONST)] * 10
    no_recovery = run_fast_monte_carlo(
        states,
        config=_config(
            n_trajectories=20_000,
            rounds_per_trajectory=10,
            depth_probability_scale=0.0,
            recovery_probability=0.0,
            recovery_gain=0.0,
            seed=600,
        ),
    )
    active_recovery = run_fast_monte_carlo(
        states,
        config=_config(
            n_trajectories=20_000,
            rounds_per_trajectory=10,
            max_recovery_depth=1,
            depth_probability_scale=1.0,
            depth_probability_power=1.0,
            recovery_probability=1.0,
            recovery_gain=1.0,
            seed=600,
        ),
    )
    assert no_recovery.logical_error_probability > 0.15
    assert active_recovery.logical_error_probability < 0.75 * no_recovery.logical_error_probability


def test_dynamic_hazard_above_one_is_rejected_not_clipped() -> None:
    config = _config(
        base_leakage_probability=0.8,
        burst_leakage_bonus=0.3,
    )
    with pytest.raises(ValueError, match="derived leakage hazard"):
        run_fast_monte_carlo([_quiet_state(burst_active=True)] * 10, config=config)


def test_q_p_and_any_event_counts_have_consistent_bounds() -> None:
    result = run_fast_monte_carlo(
        [DriftState(sigma_q=0.7, sigma_p=0.9, rho=-0.5)] * 7,
        config=_config(n_trajectories=700, rounds_per_trajectory=7),
    )
    stratum = result.strata[0]
    assert 0 <= stratum.logical_event_count <= stratum.simulated_cycles
    assert 0 <= stratum.q_event_count <= stratum.simulated_cycles
    assert 0 <= stratum.p_event_count <= stratum.simulated_cycles
    assert stratum.logical_event_count >= max(stratum.q_event_count, stratum.p_event_count)
    assert stratum.logical_event_count <= stratum.q_event_count + stratum.p_event_count


def test_cluster_bootstrap_interval_is_seeded_and_contains_point_estimate() -> None:
    config = _config(n_trajectories=600, rounds_per_trajectory=30, seed=515)
    states = [DriftState(sigma_q=0.5, sigma_p=0.6, rho=0.2)] * 30
    first = run_fast_monte_carlo(states, config=config)
    second = run_fast_monte_carlo(states, config=config)
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high
    assert first.bootstrap_standard_error == second.bootstrap_standard_error
    assert first.ci_low <= first.logical_error_probability <= first.ci_high


def test_report_writer_round_trips_machine_artifact() -> None:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        path = Path(directory) / "report.json"
        result = run_fast_monte_carlo(
            [_quiet_state()] * 5,
            config=_config(rounds_per_trajectory=5),
        )
        returned = write_fast_monte_carlo_report(result, path)
        assert returned == path
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "t2.1.3-fast-monte-carlo-v1"
        assert payload["seed"] == result.config.seed
        assert payload["logical_error_probability"] == result.logical_error_probability
        assert payload["scope_limits"][2] == "not device calibrated or target-board timed"
        with pytest.raises(TypeError, match="result"):
            write_fast_monte_carlo_report(object(), path)  # type: ignore[arg-type]
