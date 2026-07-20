from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.leakage_reset_causal import (
    CALIBRATION_SEEDS,
    DEFAULT_ARTIFACT,
    EVALUATION_SEEDS,
    FAMILIES,
    IMPLEMENTATION_PATHS,
    LEAKAGE_INJECTION_RATES,
    PARENT_ARTIFACTS,
    PRIMARY_SOURCE_ANCHORS,
    PRIMARY_SOURCE_PATH,
    RESET_FAILURE_RATES,
    SCALAR_METRICS,
    CampaignConfig,
    _channel_spec,
    _run_seed_cell,
    implementation_sha256,
    validate_payload,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((ROOT / DEFAULT_ARTIFACT).read_text(encoding="utf-8"))


def _rows(artifact: dict, family: str) -> list[dict]:
    return [row for row in artifact["summary_rows"] if row["family"] == family]


def _means(rows: list[dict], metric: str) -> list[float]:
    return [float(row[metric]["mean"]) for row in rows]


def test_committed_artifact_is_current_complete_and_passes_all_gates(
    artifact: dict,
) -> None:
    assert validate_payload(artifact) == ()
    assert artifact["status"] == "PASS"
    assert artifact["validation_errors"] == []
    assert artifact["gate_summary"] == {
        "passed": 23,
        "total": 23,
        "failed": [],
    }
    assert all(artifact["gates"].values())
    assert artifact["implementation_sha256"] == implementation_sha256()


def test_parent_implementation_and_source_bindings_are_current(artifact: dict) -> None:
    assert set(artifact["parent_integrity"]) == set(PARENT_ARTIFACTS)
    for task_id, path in PARENT_ARTIFACTS.items():
        record = artifact["parent_integrity"][task_id]
        assert record["path"] == path.as_posix()
        assert record["sha256"] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert record["machine_pass"] is True
    assert [row["path"] for row in artifact["implementation_bindings"]] == [
        path.as_posix() for path in IMPLEMENTATION_PATHS
    ]
    assert all(
        row["sha256"] == hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
        for row in artifact["implementation_bindings"]
    )
    source = artifact["source_binding"]
    assert source["path"] == PRIMARY_SOURCE_PATH
    assert source["sha256"] == hashlib.sha256((ROOT / PRIMARY_SOURCE_PATH).read_bytes()).hexdigest()
    assert source["anchors"] == list(PRIMARY_SOURCE_ANCHORS)
    lines = (ROOT / PRIMARY_SOURCE_PATH).read_text(encoding="utf-8").splitlines()
    assert all(anchor["fragment"] in lines[anchor["line"] - 1] for anchor in source["anchors"])


def test_formal_grids_seed_clusters_and_all_cells_are_exact(artifact: dict) -> None:
    config = artifact["config"]
    assert tuple(config["families"]) == FAMILIES
    assert tuple(config["leakage_injection_rates"]) == LEAKAGE_INJECTION_RATES
    assert tuple(config["reset_failure_rates"]) == RESET_FAILURE_RATES
    assert tuple(config["evaluation_seeds"]) == EVALUATION_SEEDS
    assert not (set(EVALUATION_SEEDS) & set(CALIBRATION_SEEDS))
    assert config["trajectories_per_seed"] == 256
    assert config["evaluation_cycles"] == 512
    expected = {
        (family, seed, rate)
        for family, rates in (
            ("higher_leakage_injection", LEAKAGE_INJECTION_RATES),
            ("higher_reset_failure", RESET_FAILURE_RATES),
        )
        for seed in EVALUATION_SEEDS
        for rate in rates
    }
    assert len(artifact["seed_rows"]) == len(expected) == 96
    assert {
        (row["family"], row["seed"], row["intervention_rate"])
        for row in artifact["seed_rows"]
    } == expected


def test_every_family_changes_only_its_registered_channel(artifact: dict) -> None:
    for row in artifact["seed_rows"]:
        expected = json.loads(
            json.dumps(_channel_spec(row["family"], row["intervention_rate"]))
        )
        assert row["channel_spec"] == expected
        spec = row["channel_spec"]
        if row["family"] == "higher_leakage_injection":
            assert spec["higher_injection_given_g"] == row["intervention_rate"]
            assert spec["higher_injection_given_e"] == row["intervention_rate"]
            assert spec["higher_reset_failure_probability"] == 0.9
        else:
            assert spec["higher_injection_given_g"] == 0.002
            assert spec["higher_injection_given_e"] == 0.002
            assert spec["higher_reset_failure_probability"] == row["intervention_rate"]
    for family in FAMILIES:
        for seed in EVALUATION_SEEDS:
            assert {
                row["paired_stream_id"]
                for row in artifact["seed_rows"]
                if row["family"] == family and row["seed"] == seed
            } == {f"{family}-crn-{seed}"}


def test_leakage_free_ablation_preserves_false_alarms_and_null_semantics(
    artifact: dict,
) -> None:
    row = _rows(artifact, "higher_leakage_injection")[0]
    assert row["intervention_rate"] == 0.0
    assert row["hidden_leakage_occupancy"]["mean"] == 0.0
    assert row["reset_attempts_per_1000_cycles"]["mean"] == 0.0
    assert row["reset_failures_per_1000_cycles"]["mean"] == 0.0
    assert row["detection_probability"]["mean"] is None
    assert row["detection_probability"]["status"] == "NOT_APPLICABLE_NO_TRUE_EPISODES"
    assert row["mean_detection_delay_steps"]["mean"] is None
    assert 5e-5 < row["false_alarm_rate_per_healthy_step"]["mean"] < 4e-4
    assert row["reset_requests_per_1000_cycles"]["mean"] > 0.0


def test_leakage_injection_increases_burden_cost_and_tail(artifact: dict) -> None:
    rows = _rows(artifact, "higher_leakage_injection")
    for metric in (
        "hidden_leakage_occupancy",
        "reset_attempts_per_1000_cycles",
        "reset_failures_per_1000_cycles",
    ):
        values = _means(rows, metric)
        assert all(right > left for left, right in zip(values, values[1:]))
    availability = _means(rows, "safe_normal_action_availability")
    assert all(right < left for left, right in zip(availability, availability[1:]))
    assert rows[-1]["mean_short_lag_correlation"]["mean"] > 0.7
    assert rows[-1]["mean_long_lag_correlation"]["mean"] > 0.2
    assert rows[-1]["mean_long_lag_covariance"]["mean"] > 0.008


def test_reset_failure_increases_persistence_cost_and_unavailability(
    artifact: dict,
) -> None:
    rows = _rows(artifact, "higher_reset_failure")
    for metric in (
        "hidden_leakage_occupancy",
        "mean_hidden_leakage_run_steps",
        "reset_failures_per_1000_cycles",
    ):
        values = _means(rows, metric)
        assert all(right > left for left, right in zip(values, values[1:]))
    availability = _means(rows, "safe_normal_action_availability")
    assert all(right < left for left, right in zip(availability, availability[1:]))
    assert rows[0]["mean_hidden_leakage_run_steps"]["mean"] < 1.1
    assert rows[-1]["mean_hidden_leakage_run_steps"]["mean"] > 19.0
    # At short persistence the registered long lags are at the finite-sample
    # noise floor, so do not pretend all six point estimates are ordered.
    assert rows[-1]["mean_long_lag_correlation"]["mean"] - rows[0][
        "mean_long_lag_correlation"
    ]["mean"] > 0.4
    assert all(
        right > left
        for left, right in zip(
            _means(rows, "mean_long_lag_correlation")[1:],
            _means(rows, "mean_long_lag_correlation")[2:],
        )
    )
    assert rows[-1]["mean_long_lag_correlation"]["mean"] > 0.39


def test_intervention_hazards_and_fixed_channels_are_empirically_calibrated(
    artifact: dict,
) -> None:
    leakage = _rows(artifact, "higher_leakage_injection")
    assert leakage[0]["empirical_higher_injection_probability"]["mean"] == 0.0
    for row in leakage[1:]:
        assert abs(
            row["empirical_higher_injection_probability"]["mean"]
            - row["intervention_rate"]
        ) <= 8e-5
        assert abs(row["empirical_reset_failure_probability"]["mean"] - 0.9) <= 0.02
    for row in _rows(artifact, "higher_reset_failure"):
        assert abs(row["empirical_higher_injection_probability"]["mean"] - 0.002) <= 2e-4
        assert abs(
            row["empirical_reset_failure_probability"]["mean"]
            - row["intervention_rate"]
        ) <= 0.02


def test_detection_delay_false_alarm_and_false_negative_have_correct_denominators(
    artifact: dict,
) -> None:
    for row in artifact["summary_rows"]:
        detection = row["detection_probability"]["mean"]
        if detection is None:
            continue
        assert detection >= 0.99
        assert row["mean_detection_delay_steps"]["mean"] <= 0.12
        assert row["p95_detection_delay_steps"]["mean"] <= 1.0
        assert 0.03 <= row["false_negative_rate_per_leakage_step"]["mean"] <= 0.07
        assert 5e-5 <= row["false_alarm_rate_per_healthy_step"]["mean"] <= 4e-4


def test_availability_cost_and_truth_contracts_remain_separate(artifact: dict) -> None:
    estimand = artifact["estimand_contract"]
    assert estimand["postselection_used_for_primary_metrics"] is False
    assert estimand["combined_availability_cost_score"] == "FORBIDDEN"
    assert estimand["physical_memory_ler"] == "NOT_ESTABLISHED"
    assert artifact["causal_contract"]["truth_visibility"] == (
        "truth_only_scores_detection_false_alarm_and_persistence"
    )
    assert all(row["truth_used_only_for_scoring"] is True for row in artifact["seed_rows"])
    assert artifact["device_calibrated"] is False
    assert artifact["experimental_hardware_used"] is False
    assert artifact["physical_memory_ler_established"] is False


def test_cluster_uncertainty_and_nullable_summaries_are_explicit(artifact: dict) -> None:
    for row in artifact["summary_rows"]:
        for metric in SCALAR_METRICS:
            summary = row[metric]
            assert summary["resampling_unit"] == "whole_seed_cluster"
            assert summary["paired_seed_cluster_count"] == 8
            assert summary["bootstrap_replicates"] == 20000
            if summary["mean"] is None:
                assert summary["ci_low"] is summary["ci_high"] is None
                assert summary["nonnull_seed_cluster_count"] == 0
            else:
                assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]
                assert summary["nonnull_seed_cluster_count"] == 8


def test_source_ledger_has_exact_rows_types_and_hash(artifact: dict) -> None:
    record = artifact["source_data"]
    path = ROOT / record["path"]
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == record["row_count"] == 2508
    assert {row["row_type"] for row in rows} == {
        "parent_artifact",
        "implementation_binding",
        "primary_source_anchor",
        "channel_intervention",
        "seed_metric",
        "seed_cluster_summary",
        "contract_gate",
    }
    assert sum(row["row_type"] == "channel_intervention" for row in rows) == 96
    assert sum(row["row_type"] == "seed_metric" for row in rows) == 96 * len(SCALAR_METRICS)
    assert sum(row["row_type"] == "seed_cluster_summary" for row in rows) == 12 * len(SCALAR_METRICS)


def test_validator_rejects_channel_mixing_truth_leakage_and_fake_zero(
    artifact: dict,
) -> None:
    mixed = copy.deepcopy(artifact)
    row = next(
        row
        for row in mixed["seed_rows"]
        if row["family"] == "higher_leakage_injection" and row["intervention_rate"] == 0.002
    )
    row["channel_spec"]["higher_reset_failure_probability"] = 0.5
    assert "one-channel-at-a-time configuration changed" in validate_payload(mixed)

    leaked = copy.deepcopy(artifact)
    leaked["seed_rows"][0]["truth_used_only_for_scoring"] = False
    assert "truth entered deployable path" in validate_payload(leaked)

    fake = copy.deepcopy(artifact)
    control = next(
        row
        for row in fake["summary_rows"]
        if row["family"] == "higher_leakage_injection" and row["intervention_rate"] == 0.0
    )
    control["detection_probability"].update(mean=0.0, ci_low=0.0, ci_high=0.0)
    assert "not-applicable summary was converted to a fake zero" in validate_payload(fake)


def test_validator_rejects_coherent_direction_rewrite_missing_parent_and_score(
    artifact: dict,
) -> None:
    rewritten = copy.deepcopy(artifact)
    for row in rewritten["seed_rows"]:
        if row["family"] == "higher_reset_failure":
            row["mean_hidden_leakage_run_steps"] = 2.0
    for row in rewritten["summary_rows"]:
        if row["family"] == "higher_reset_failure":
            row["mean_hidden_leakage_run_steps"].update(mean=2.0, ci_low=2.0, ci_high=2.0)
    assert "reset-failure causal direction changed" in validate_payload(rewritten)

    missing = copy.deepcopy(artifact)
    missing["seed_rows"].pop()
    assert "family-seed-rate matrix is incomplete or duplicated" in validate_payload(missing)

    stale = copy.deepcopy(artifact)
    stale["parent_integrity"]["T2.0.6"]["sha256"] = "0" * 64
    assert "parent artifact binding is stale or failed" in validate_payload(stale)

    collapsed = copy.deepcopy(artifact)
    collapsed["global_score"] = 0.5
    assert (
        "forbidden collapsed score or postselected primary field was introduced"
        in validate_payload(collapsed)
    )


def test_small_formal_cells_are_deterministic_and_preserve_null_semantics() -> None:
    config = CampaignConfig()
    cases = (
        ("higher_leakage_injection", 0.0),
        ("higher_leakage_injection", 0.004),
        ("higher_reset_failure", 0.95),
    )
    for family, rate in cases:
        first = _run_seed_cell(family, rate, EVALUATION_SEEDS[0], config=config)
        second = _run_seed_cell(family, rate, EVALUATION_SEEDS[0], config=config)
        assert first == second
        assert len(first["trace_sha256"]) == 64
        if family == "higher_leakage_injection" and rate == 0.0:
            assert first["detection_probability"] is None
            assert first["mean_hidden_leakage_run_steps"] is None
        else:
            assert first["detection_probability"] is not None
            assert first["mean_hidden_leakage_run_steps"] is not None


@pytest.mark.parametrize(
    "change",
    [
        {"families": ("higher_leakage_injection",)},
        {"leakage_injection_rates": (0.0, 0.01)},
        {"reset_failure_rates": (0.0, 0.5)},
        {"evaluation_seeds": EVALUATION_SEEDS[:-1]},
        {"trajectories_per_seed": 64},
        {"burn_in_cycles": 0},
        {"evaluation_cycles": 128},
        {"seed_cluster_bootstrap_replicates": 1000},
        {"false_leakage_alarm_probability": 0.0},
        {"leakage_detection_probability": 1.0},
        {"fixed_reset_failure_for_leakage_family": 0.5},
        {"fixed_leakage_injection_for_reset_family": 0.0},
    ],
)
def test_formal_configuration_rejects_demo_simplification(change: dict) -> None:
    with pytest.raises(ValueError):
        CampaignConfig(**change)
