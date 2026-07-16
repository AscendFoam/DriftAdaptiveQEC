from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import physics
import physics.sbs_displacement_fault as displacement_module
from physics.sbs_displacement_fault import (
    MODEL_SCOPE,
    PRIMARY_SOURCE_ANCHORS,
    PRIMARY_SOURCE_PATH,
    DisplacementFaultSweepConfig,
    distance_to_closest_logical_operation,
    run_displacement_fault_sweep,
    write_displacement_fault_report,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def accepted_result():
    config = DisplacementFaultSweepConfig(
        shots=1024,
        bootstrap_replicates=200,
    )
    return run_displacement_fault_sweep(config)


def test_logical_distance_is_periodic_symmetric_and_peaks_at_quarter_lattice() -> None:
    amplitudes = np.array([-0.5, -0.25, 0.0, 0.125, 0.25, 0.375, 0.5, 0.75])
    observed = distance_to_closest_logical_operation(amplitudes)
    assert observed == pytest.approx([0.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.0, 1.0])
    assert distance_to_closest_logical_operation(0.125) == pytest.approx(
        distance_to_closest_logical_operation(0.375)
    )
    with pytest.raises(ValueError, match="finite"):
        distance_to_closest_logical_operation([0.0, np.nan])
    with pytest.raises(ValueError, match="positive"):
        distance_to_closest_logical_operation(0.1, logical_spacing_over_lattice=0.0)


def test_config_preregisters_seed_amplitudes_horizon_and_acceptance_tolerances() -> None:
    config = DisplacementFaultSweepConfig()
    assert config.amplitudes_over_lattice[0] == 0.0
    assert config.amplitudes_over_lattice[-1] == 0.5
    assert 0.25 in config.amplitudes_over_lattice
    assert config.shots == 4096
    assert config.cycles == 20
    assert config.seed != config.bootstrap_seed
    assert config.bootstrap_replicates == 500
    assert config.minimum_midpoint_endpoint_run_margin == 2.0
    assert config.maximum_unaffected_e_probability == 0.06
    assert config.model_scope == MODEL_SCOPE


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"amplitudes_over_lattice": (0.0, 0.1, 0.2, 0.3, 0.5)}, "include"),
        ({"amplitudes_over_lattice": (0.0, 0.25, 0.2, 0.4, 0.5)}, "increasing"),
        ({"shots": 63}, "shots"),
        ({"cycles": 5, "max_recovery_depth": 6}, "cycles"),
        ({"fault_quadrature": "Y"}, "quadrature"),
        ({"e_detection_probability": 1.1}, r"\[0, 1\]"),
        ({"bootstrap_replicates": 99}, "bootstrap_replicates"),
        ({"model_scope": "device_calibrated"}, "model_scope"),
    ],
)
def test_invalid_preregistration_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        DisplacementFaultSweepConfig(**kwargs)


def test_preregistered_sweep_passes_all_nonmonotonic_and_negative_control_gates(
    accepted_result,
) -> None:
    result = accepted_result
    result.require_pass()
    assert result.gate.passed
    assert result.gate.failed_check_ids == ()
    assert len(result.gate.checks) == 10
    assert all(check.passed for check in result.gate.checks)
    assert result.device_calibrated is False
    assert result.experimental_data_digitized_or_fitted is False


def test_midpoint_has_maximum_depth_and_run_while_endpoints_are_near_identity(
    accepted_result,
) -> None:
    points = {point.amplitude_over_lattice: point for point in accepted_result.points}
    midpoint = points[0.25]
    left_endpoint = points[0.0]
    right_endpoint = points[0.5]
    assert midpoint.initial_recovery_depth.mean == pytest.approx(6.0)
    assert left_endpoint.initial_recovery_depth.mean == pytest.approx(0.0)
    assert right_endpoint.initial_recovery_depth.mean == pytest.approx(0.0)
    assert midpoint.observed_same_quadrature_max_e_run.ci_low > (
        max(
            left_endpoint.observed_same_quadrature_max_e_run.ci_high,
            right_endpoint.observed_same_quadrature_max_e_run.ci_high,
        )
        + 2.0
    )
    assert midpoint.ideal_same_quadrature_max_e_run.mean > 4.0


def test_run_trend_increases_then_decreases_and_is_mirror_symmetric(accepted_result) -> None:
    runs = np.array(
        [point.observed_same_quadrature_max_e_run.mean for point in accepted_result.points]
    )
    midpoint = len(runs) // 2
    assert np.all(np.diff(runs[: midpoint + 1]) > 0.0)
    assert np.all(np.diff(runs[midpoint:]) < 0.0)
    assert np.max(np.abs(runs[:midpoint] - runs[:midpoint:-1])) <= 0.30


def test_midpoint_syndrome_heatmap_row_decays_and_unaffected_axis_stays_quiet(
    accepted_result,
) -> None:
    midpoint = next(
        point for point in accepted_result.points if point.amplitude_over_lattice == 0.25
    )
    affected = np.asarray(midpoint.affected_e_probability_by_cycle)
    unaffected = np.asarray(midpoint.unaffected_e_probability_by_cycle)
    assert affected.shape == (accepted_result.config.cycles,)
    assert np.mean(affected[:3]) - np.mean(affected[-3:]) >= 0.25
    assert np.max(unaffected) <= accepted_result.config.maximum_unaffected_e_probability
    assert midpoint.recovered_fraction_by_horizon >= 0.98
    assert midpoint.censored_shots <= 0.02 * accepted_result.config.shots


def test_position_displacement_uses_same_quadrature_eg_semantics_not_pair_mixing(
    accepted_result,
) -> None:
    # fault_quadrature=Z corresponds to chronological X=g,Z=e and Kraus label K_eg.
    config = accepted_result.config
    assert config.fault_quadrature == "Z"
    midpoint = next(
        point for point in accepted_result.points if point.amplitude_over_lattice == 0.25
    )
    assert midpoint.observed_same_quadrature_max_e_run.mean > 4.0
    assert max(midpoint.unaffected_e_probability_by_cycle) < 0.06


def test_position_fault_transition_kernel_is_keg_and_observation_kernel_is_explicit() -> None:
    config = DisplacementFaultSweepConfig(shots=64, bootstrap_replicates=100)
    instrument = displacement_module._make_recovery_instrument(config)
    observation = displacement_module._make_observation_model(config)
    assert instrument.transition_probabilities["K_eg"][0, 1] == pytest.approx(0.88)
    assert instrument.transition_probabilities["K_gg"][1, 1] == pytest.approx(0.12)
    assert np.sum(instrument.transition_probabilities["K_ge"]) == pytest.approx(0.0)
    assert np.sum(instrument.transition_probabilities["K_ee"]) == pytest.approx(0.0)
    assert displacement_module.ideal_syndrome_from_kraus("K_eg").as_tuple() == ("g", "e")
    assert observation.readout_confusion[0] == pytest.approx([0.995, 0.005, 0.0])
    assert observation.readout_confusion[1] == pytest.approx([0.02, 0.98, 0.0])
    assert observation.model_scope.endswith("not_device_calibrated")


def test_sampled_depth_and_recovery_time_match_independent_analytic_expectations(
    accepted_result,
) -> None:
    probability = accepted_result.config.one_step_recovery_probability
    for point in accepted_result.points:
        expected_depth = accepted_result.config.max_recovery_depth * point.logical_distance
        assert point.initial_recovery_depth.mean == pytest.approx(expected_depth, abs=0.10)
        # With no two-step branch, time to remove d levels is negative-binomial with E[T|d]=d/p.
        expected_time = expected_depth / probability
        assert point.restricted_recovery_cycles.mean == pytest.approx(expected_time, abs=0.15)


def test_same_seed_is_bitwise_reproducible_including_bootstrap_intervals() -> None:
    config = DisplacementFaultSweepConfig(shots=256, bootstrap_replicates=100)
    first = run_displacement_fault_sweep(config).to_dict()
    second = run_displacement_fault_sweep(config).to_dict()
    assert first == second


def test_changed_seed_changes_interior_samples_but_keeps_the_physics_gate() -> None:
    config = DisplacementFaultSweepConfig(shots=512, bootstrap_replicates=100)
    first = run_displacement_fault_sweep(config)
    second = run_displacement_fault_sweep(
        replace(config, seed=config.seed + 1, bootstrap_seed=config.bootstrap_seed + 1)
    )
    first_run = first.points[2].observed_same_quadrature_max_e_run.mean
    second_run = second.points[2].observed_same_quadrature_max_e_run.mean
    assert first_run != second_run
    assert first.gate.passed and second.gate.passed


def test_x_and_z_fault_variants_preserve_the_same_nonmonotonic_conclusion() -> None:
    base = DisplacementFaultSweepConfig(shots=512, bootstrap_replicates=100)
    z_result = run_displacement_fault_sweep(base)
    x_result = run_displacement_fault_sweep(replace(base, fault_quadrature="X"))
    z_mid = z_result.points[len(z_result.points) // 2]
    x_mid = x_result.points[len(x_result.points) // 2]
    assert z_result.gate.passed and x_result.gate.passed
    assert abs(
        z_mid.observed_same_quadrature_max_e_run.mean
        - x_mid.observed_same_quadrature_max_e_run.mean
    ) < 0.25


def test_bad_readout_assumption_triggers_named_failure_diagnostics() -> None:
    config = DisplacementFaultSweepConfig(
        shots=256,
        bootstrap_replicates=100,
        false_e_given_g=0.50,
        e_detection_probability=0.50,
    )
    result = run_displacement_fault_sweep(config)
    assert not result.gate.passed
    assert "unaffected_quadrature_negative_control" in result.gate.failed_check_ids
    with pytest.raises(RuntimeError, match="trend gate failed"):
        result.require_pass()


def test_report_writer_preserves_seeds_tolerances_ci_and_failure_diagnostics(
    accepted_result,
) -> None:
    output_dir = ROOT / ".pytest_cache" / "t2_0_5_report_writer"
    json_path = output_dir / "result.json"
    csv_path = output_dir / "trend.csv"
    write_displacement_fault_report(
        accepted_result,
        json_path=json_path,
        csv_path=csv_path,
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config"]["seed"] == accepted_result.config.seed
    assert payload["config"]["bootstrap_seed"] == accepted_result.config.bootstrap_seed
    assert payload["config"]["peak_location_tolerance"] == 0.0625
    assert payload["points"][4]["observed_same_quadrature_max_e_run"]["method"] == (
        "percentile_nonparametric_bootstrap"
    )
    assert len(payload["gate"]["checks"]) == 10
    assert payload["limitations"]
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == len(accepted_result.points) + 1
    assert "observed_run_ci_low" in rows[0]


def test_primary_source_anchors_match_local_paper_exactly() -> None:
    lines = (ROOT / PRIMARY_SOURCE_PATH).read_text(encoding="utf-8").splitlines()
    for anchor in PRIMARY_SOURCE_ANCHORS:
        assert anchor["fragment"] in lines[int(anchor["line"]) - 1]


def test_committed_production_report_is_preregistered_complete_and_passing() -> None:
    payload = json.loads(
        (ROOT / "docs" / "t2_0_5_displacement_fault_trend.json").read_text(encoding="utf-8")
    )
    assert payload["config"]["shots"] == 4096
    assert payload["config"]["cycles"] == 20
    assert payload["config"]["seed"] == 2026071405
    assert payload["config"]["bootstrap_seed"] == 2026071406
    assert payload["gate"]["passed"] is True
    assert len(payload["points"]) == 9
    midpoint = payload["points"][4]
    assert midpoint["amplitude_over_lattice"] == 0.25
    assert midpoint["initial_recovery_depth"]["mean"] == pytest.approx(6.0)
    assert midpoint["observed_same_quadrature_max_e_run"]["ci_low"] > 4.8
    assert payload["device_calibrated"] is False
    assert payload["experimental_data_digitized_or_fitted"] is False
    regenerated = json.loads(json.dumps(run_displacement_fault_sweep().to_dict()))
    assert payload == regenerated
    csv_lines = (
        ROOT / "docs" / "t2_0_5_displacement_fault_trend.csv"
    ).read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 10


def test_protocol_registry_promotes_only_the_verified_fault_trend_layer() -> None:
    registry = json.loads((ROOT / "docs" / "protocol_hierarchy.json").read_text(encoding="utf-8"))
    main = next(item for item in registry["protocols"] if item["protocol_id"] == "PROTO-SBS-MAIN")
    contract = main["fault_diagnostic_contract"]
    assert contract["large_distance_peak_over_lS"] == 0.25
    assert contract["position_fault_full_cycle_label"] == "K_eg"
    assert contract["chronological_pair"] == "X_g_then_Z_e"
    assert "T2.0.5" not in main["future_tasks"]
    # Milestone 2.3 is now closed through the strict-split T2.3.7 ranking.
    # Keep future work pointed at the actual robustness/device tasks instead
    # of retaining a completed T2.3 item merely to satisfy a stale test.
    assert not any(task.startswith("T2.3") for task in main["future_tasks"])
    assert {"T3.2.11", "T4.4.1", "T5.2.1"}.issubset(main["future_tasks"])
    update = next(item for item in registry["implementation_updates"] if item["task_id"] == "T2.0.5")
    assert update["evidence_scope"] == MODEL_SCOPE
    assert all((ROOT / path).is_file() for path in update["artifacts"])
    forbidden = " ".join(main["forbidden_transfers"])
    assert "not claim the qualitative displacement trend is a digitized Fig. 4(c)" in forbidden


def test_public_lazy_exports_include_displacement_fault_contract() -> None:
    assert physics.DisplacementFaultSweepConfig is DisplacementFaultSweepConfig
    assert physics.run_displacement_fault_sweep is run_displacement_fault_sweep
    assert physics.distance_to_closest_logical_operation is distance_to_closest_logical_operation
