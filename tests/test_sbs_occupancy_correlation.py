from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import physics
import physics.sbs_occupancy_correlation as occupancy_module
from physics.sbs_occupancy_correlation import (
    MODEL_SCOPE,
    PRIMARY_SOURCE_ANCHORS,
    PRIMARY_SOURCE_PATH,
    OccupancyCorrelationConfig,
    estimate_leakage_tail_correlation,
    estimate_occupancy_from_syndrome,
    run_occupancy_correlation_validation,
    simulate_occupancy_correlation_dataset,
    write_occupancy_correlation_report,
)
from physics.sbs_observation_reset import HIDDEN_ANCILLA_STATES, OBSERVED_CLASSES


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def production_dataset():
    return simulate_occupancy_correlation_dataset()


@pytest.fixture(scope="module")
def production_result():
    return run_occupancy_correlation_validation()


def test_config_preserves_source_scale_seeds_fit_windows_and_tail_lags() -> None:
    config = OccupancyCorrelationConfig()
    assert config.shots == 600
    assert config.cycles == 1200
    assert config.burn_in_cycles == 200
    assert config.physical_error_probability == pytest.approx(0.13)
    assert config.recovery_probability == pytest.approx(0.5922222222222222)
    assert config.single_cycle_leakage_rate + config.higher_leakage_rate == pytest.approx(6.76e-4)
    assert config.higher_leakage_rate == pytest.approx(1.28e-4)
    assert config.higher_leakage_mean_duration_cycles == pytest.approx(17.2)
    assert config.all_gg_string_lengths == tuple(range(2, 13))
    assert config.tail_lags == tuple(range(40, 201, 20))
    assert config.seed != config.bootstrap_seed
    assert config.model_scope == MODEL_SCOPE


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shots": 99}, "shots"),
        ({"cycles": 99}, "cycles"),
        ({"bootstrap_replicates": 99}, "bootstrap_replicates"),
        ({"physical_error_probability": 1.1}, r"\[0, 1\]"),
        ({"target_no_leakage_occupancy": 1.0}, "strictly"),
        ({"higher_leakage_mean_duration_cycles": 2.0}, "exceed"),
        ({"all_gg_string_lengths": (2, 3, 4)}, "four"),
        ({"tail_lags": (10, 20, 30)}, "four"),
        ({"minimum_retained_fraction": 0.95, "maximum_retained_fraction": 0.75}, "below"),
        ({"model_scope": "device_calibrated"}, "model_scope"),
    ],
)
def test_invalid_configurations_fail_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        OccupancyCorrelationConfig(**kwargs)


def test_observation_model_uses_registered_ge_diagonals_and_keeps_higher_unaddressed() -> None:
    config = OccupancyCorrelationConfig()
    model = occupancy_module._make_observation_model(config)
    assert model.readout_confusion[0] == pytest.approx([0.9997, 0.0003, 0.0])
    assert model.readout_confusion[1] == pytest.approx([0.0086, 0.9914, 0.0])
    assert model.readout_confusion[2] == pytest.approx([0.0, 0.0, 1.0])
    leakage_index = OBSERVED_CLASSES.index("leakage")
    higher_index = HIDDEN_ANCILLA_STATES.index("higher")
    assert model.reset_kernel[leakage_index, higher_index, higher_index] == pytest.approx(1.0)
    assert "not_device" in model.model_scope


def test_dataset_shapes_are_read_only_and_truth_observation_lanes_are_separate(
    production_dataset,
) -> None:
    dataset = production_dataset
    expected = (600, 1200)
    arrays = (
        dataset.hidden_code_occupied,
        dataset.hidden_error_depth,
        dataset.hidden_leakage_kind,
        dataset.observed_x,
        dataset.observed_z,
        dataset.all_gg,
        dataset.non_g_activity,
        dataset.observed_leakage,
    )
    assert all(array.shape == expected for array in arrays)
    assert all(not array.flags.writeable for array in arrays)
    assert dataset.truth_scope == "simulator_hidden_truth_not_deployable_input"
    assert dataset.observation_scope == "observed_syndrome_only"
    assert not np.shares_memory(dataset.hidden_code_occupied, dataset.all_gg)


def test_hidden_code_occupancy_is_consistent_with_start_of_cycle_depth_and_leakage(
    production_dataset,
) -> None:
    expected = (production_dataset.hidden_error_depth == 0) & (
        production_dataset.hidden_leakage_kind == 0
    )
    assert np.array_equal(production_dataset.hidden_code_occupied, expected)
    assert set(np.unique(production_dataset.hidden_leakage_kind)).issubset({0, 1, 2})


def test_observed_all_gg_and_activity_are_derived_without_hidden_truth(
    production_dataset,
) -> None:
    g = OBSERVED_CLASSES.index("g")
    expected_all_gg = (production_dataset.observed_x == g) & (
        production_dataset.observed_z == g
    )
    assert np.array_equal(production_dataset.all_gg, expected_all_gg)
    assert np.array_equal(production_dataset.non_g_activity, ~expected_all_gg)
    assert np.array_equal(
        production_dataset.observed_leakage,
        (production_dataset.observed_x == OBSERVED_CLASSES.index("leakage"))
        | (production_dataset.observed_z == OBSERVED_CLASSES.index("leakage")),
    )


def test_higher_leakage_start_rate_and_duration_match_source_inspired_scale(
    production_dataset,
) -> None:
    higher = production_dataset.hidden_leakage_kind == 2
    starts = higher & ~np.pad(higher[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    start_rate = np.sum(starts) / higher.size
    assert start_rate == pytest.approx(1.28e-4, abs=4.5e-5)
    lengths: list[int] = []
    for row in higher:
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        begin = np.flatnonzero(changes == 1)
        end = np.flatnonzero(changes == -1)
        complete = (begin > 0) & (end < row.size)
        lengths.extend((end[complete] - begin[complete]).tolist())
    assert len(lengths) >= 50
    assert np.mean(lengths) == pytest.approx(17.2, abs=4.0)
    assert min(lengths) >= 2


def test_syndrome_only_estimator_signature_cannot_accept_hidden_truth() -> None:
    parameters = set(inspect.signature(estimate_occupancy_from_syndrome).parameters)
    assert parameters == {
        "observed_all_gg",
        "string_lengths",
        "bootstrap_replicates",
        "bootstrap_seed",
        "confidence_level",
    }
    assert not any("hidden" in name or "truth" in name for name in parameters)


def test_hidden_and_syndrome_occupancy_agree_within_registered_intervals(
    production_result,
) -> None:
    result = production_result
    result.require_pass()
    hidden = result.hidden_occupancy
    syndrome = result.syndrome_occupancy
    assert hidden.mean == pytest.approx(0.8135652777777779)
    assert syndrome.occupancy == pytest.approx(0.8135243036124343)
    assert abs(hidden.mean - syndrome.occupancy) < 1.0e-3
    assert syndrome.occupancy_combined_ci[0] <= hidden.mean <= syndrome.occupancy_combined_ci[1]
    assert hidden.ci_low <= syndrome.occupancy <= hidden.ci_high
    assert syndrome.first_order_model_error_bound == pytest.approx(
        syndrome.physical_error_probability**2
    )


def test_all_gg_fit_recovers_error_rate_and_single_exponential_structure(
    production_result,
) -> None:
    estimate = production_result.syndrome_occupancy
    probabilities = np.asarray(estimate.all_gg_probabilities)
    assert np.all(np.diff(probabilities) < 0.0)
    assert estimate.fitted_lambda == pytest.approx(0.8679893904003879)
    assert estimate.physical_error_probability == pytest.approx(0.13201060959961208)
    assert estimate.r_squared_log_probability > 0.999999
    reconstructed = estimate.fitted_a * estimate.fitted_lambda
    assert reconstructed == pytest.approx(estimate.occupancy)


def test_leakage_removal_shrinks_long_lag_tail_with_paired_ci(production_result) -> None:
    tail = production_result.tail_correlation
    assert tail.retained_shots == 507
    assert tail.retained_fraction == pytest.approx(0.845)
    assert tail.mean_before.mean > 0.002
    assert abs(tail.mean_after.mean) < 5.0e-4
    assert tail.paired_difference.ci_low > 0.001
    assert tail.shrink_ratio > 10.0
    assert len(tail.before_removal) == len(tail.after_removal) == len(tail.lags)


def test_all_registered_acceptance_checks_pass_and_are_named(production_result) -> None:
    assert production_result.gate.passed
    assert production_result.gate.failed_check_ids == ()
    assert len(production_result.gate.checks) == 11
    assert {check.check_id for check in production_result.gate.checks} == {
        "hidden_occupancy_near_reference",
        "syndrome_estimate_matches_hidden_truth",
        "hidden_truth_inside_first_order_syndrome_interval",
        "physical_error_estimate_matches_generator",
        "single_exponential_all_gg_fit",
        "all_gg_probability_strictly_decreases",
        "leakage_removal_retains_expected_fraction",
        "long_lag_tail_detected_before_removal",
        "tail_shrink_paired_ci_positive",
        "tail_shrink_ratio",
        "post_removal_tail_small",
    }


def test_no_higher_leakage_ablation_removes_tail_and_makes_shrink_gate_fail() -> None:
    config = replace(
        OccupancyCorrelationConfig(),
        higher_leakage_rate=0.0,
        bootstrap_replicates=100,
    )
    result = run_occupancy_correlation_validation(config)
    assert result.tail_correlation.retained_fraction == pytest.approx(1.0)
    assert result.tail_correlation.mean_before.mean == pytest.approx(
        result.tail_correlation.mean_after.mean
    )
    assert result.tail_correlation.paired_difference.mean == pytest.approx(0.0)
    assert "tail_shrink_paired_ci_positive" in result.gate.failed_check_ids
    with pytest.raises(RuntimeError, match="occupancy/correlation gate failed"):
        result.require_pass()


def test_fixed_seed_is_bitwise_reproducible_and_changed_seed_keeps_conclusion() -> None:
    config = OccupancyCorrelationConfig(bootstrap_replicates=100)
    first = run_occupancy_correlation_validation(config)
    second = run_occupancy_correlation_validation(config)
    assert first.to_dict() == second.to_dict()
    changed = run_occupancy_correlation_validation(
        replace(config, seed=config.seed + 1, bootstrap_seed=config.bootstrap_seed + 1)
    )
    assert changed.gate.passed
    assert changed.hidden_occupancy.mean != first.hidden_occupancy.mean


def test_estimators_reject_malformed_or_degenerate_inputs() -> None:
    with pytest.raises(TypeError, match="boolean"):
        estimate_occupancy_from_syndrome(
            np.ones((100, 100), dtype=np.int8),
            string_lengths=(2, 3, 4, 5),
            bootstrap_replicates=100,
            bootstrap_seed=1,
            confidence_level=0.95,
        )
    with pytest.raises(ValueError, match="lambda"):
        estimate_occupancy_from_syndrome(
            np.ones((100, 100), dtype=np.bool_),
            string_lengths=(2, 3, 4, 5),
            bootstrap_replicates=100,
            bootstrap_seed=1,
            confidence_level=0.95,
        )
    with pytest.raises(ValueError, match="same 2D"):
        estimate_leakage_tail_correlation(
            np.zeros((100, 100), dtype=np.bool_),
            np.zeros((99, 100), dtype=np.bool_),
            lags=(10, 20, 30, 40),
            bootstrap_replicates=100,
            bootstrap_seed=1,
            confidence_level=0.95,
        )


def test_report_writer_preserves_truth_observed_separation_and_tail_diagnostics(
    production_result,
) -> None:
    output_dir = ROOT / ".pytest_cache" / "t2_0_6_report_writer"
    json_path = output_dir / "result.json"
    csv_path = output_dir / "tail.csv"
    write_occupancy_correlation_report(
        production_result,
        json_path=json_path,
        csv_path=csv_path,
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config"]["seed"] == 2026071407
    assert payload["syndrome_occupancy"]["estimator_inputs"] == [
        "observed_all_gg_boolean_matrix"
    ]
    assert payload["tail_correlation"]["removal_rule"].endswith(">=2 cycles")
    assert payload["gate"]["passed"] is True
    assert payload["limitations"]
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 10


def test_primary_source_anchors_match_local_paper() -> None:
    lines = (ROOT / PRIMARY_SOURCE_PATH).read_text(encoding="utf-8").splitlines()
    for anchor in PRIMARY_SOURCE_ANCHORS:
        assert anchor["fragment"] in lines[int(anchor["line"]) - 1]


def test_committed_production_report_matches_same_seed_regeneration(production_result) -> None:
    payload = json.loads(
        (ROOT / "docs" / "t2_0_6_occupancy_correlation.json").read_text(encoding="utf-8")
    )
    regenerated = json.loads(json.dumps(production_result.to_dict()))
    assert payload == regenerated
    assert payload["config"]["shots"] == 600
    assert payload["config"]["cycles"] == 1200
    assert payload["hidden_occupancy"]["mean"] == pytest.approx(0.8135652777777779)
    assert payload["syndrome_occupancy"]["occupancy"] == pytest.approx(0.8135243036124343)
    assert payload["tail_correlation"]["paired_difference"]["ci_low"] > 0.001
    assert payload["gate"]["passed"] is True
    csv_lines = (ROOT / "docs" / "t2_0_6_correlation_tail.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(csv_lines) == 10


def test_protocol_registry_promotes_only_verified_occupancy_correlation_scope() -> None:
    registry = json.loads((ROOT / "docs" / "protocol_hierarchy.json").read_text(encoding="utf-8"))
    main = next(item for item in registry["protocols"] if item["protocol_id"] == "PROTO-SBS-MAIN")
    contract = main["occupancy_correlation_contract"]
    assert contract["syndrome_only_fit"] == "P([gg]^n)=a*lambda^n"
    assert contract["truth_lane"].endswith("never an estimator input")
    assert contract["leakage_removal_rule"].endswith(">=2 cycles")
    update = next(item for item in registry["implementation_updates"] if item["task_id"] == "T2.0.6")
    assert update["evidence_scope"] == MODEL_SCOPE
    assert all((ROOT / path).is_file() for path in update["artifacts"])
    forbidden = " ".join(main["forbidden_transfers"])
    assert "reproduces experimental raw data" in forbidden


def test_public_lazy_exports_include_occupancy_correlation_contract() -> None:
    assert physics.OccupancyCorrelationConfig is OccupancyCorrelationConfig
    assert physics.estimate_occupancy_from_syndrome is estimate_occupancy_from_syndrome
    assert physics.run_occupancy_correlation_validation is run_occupancy_correlation_validation
