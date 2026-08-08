from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.ancilla_readout_causal import (
    CALIBRATION_SEEDS,
    DEFAULT_ARTIFACT,
    EVALUATION_SEEDS,
    FAULT_FAMILIES,
    IMPLEMENTATION_PATHS,
    INJECTION_RATES,
    METRICS,
    PARENT_ARTIFACTS,
    PRIMARY_SOURCE_ANCHORS,
    PRIMARY_SOURCE_PATH,
    CampaignConfig,
    _analytic_expectation,
    _channel_spec,
    _run_seed_cell,
    implementation_sha256,
    validate_payload,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((ROOT / DEFAULT_ARTIFACT).read_text(encoding="utf-8"))


def _family_rows(artifact: dict, family: str) -> list[dict]:
    return [row for row in artifact["summary_rows"] if row["family"] == family]


def _means(rows: list[dict], metric: str) -> list[float]:
    return [float(row[metric]["mean"]) for row in rows]


def test_committed_artifact_is_complete_current_and_passes_all_gates(
    artifact: dict,
) -> None:
    assert validate_payload(artifact) == ()
    assert artifact["status"] == "PASS"
    assert artifact["validation_errors"] == []
    assert artifact["gate_summary"] == {
        "passed": 22,
        "total": 22,
        "failed": [],
    }
    assert all(artifact["gates"].values())
    assert artifact["implementation_sha256"] == implementation_sha256()


def test_parent_implementation_and_primary_source_bindings_are_current(
    artifact: dict,
) -> None:
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


def test_formal_grid_seed_clusters_and_cell_membership_are_exact(artifact: dict) -> None:
    config = artifact["config"]
    assert tuple(config["fault_families"]) == FAULT_FAMILIES
    assert tuple(config["injection_rates"]) == INJECTION_RATES
    assert tuple(config["evaluation_seeds"]) == EVALUATION_SEEDS
    assert not (set(EVALUATION_SEEDS) & set(CALIBRATION_SEEDS))
    assert config["cycles_per_seed_rate"] == 4096
    assert config["seed_cluster_bootstrap_replicates"] == 20000
    expected = {
        (family, seed, rate)
        for family in FAULT_FAMILIES
        for seed in EVALUATION_SEEDS
        for rate in INJECTION_RATES
    }
    rows = artifact["seed_rows"]
    assert len(rows) == len(expected) == 144
    assert {
        (row["family"], row["seed"], row["injected_rate"]) for row in rows
    } == expected
    assert all(len(row["trace_sha256"]) == 64 for row in rows)


def test_every_family_changes_exactly_one_registered_channel(artifact: dict) -> None:
    for row in artifact["seed_rows"]:
        family = row["family"]
        rate = row["injected_rate"]
        expected = json.loads(json.dumps(_channel_spec(family, rate)))
        assert row["channel_spec"] == expected
        spec = row["channel_spec"]
        bit = spec["bit_flip_probabilities"][0][1]
        phase = spec["phase_flip_probabilities"][0][1]
        readout = spec["readout_error_probability"]
        assert sum(value != 0.0 for value in (bit, phase, readout)) <= 1
        if rate > 0.0:
            assert sum(value != 0.0 for value in (bit, phase, readout)) == 1
    for family in FAULT_FAMILIES:
        for seed in EVALUATION_SEEDS:
            assert {
                row["paired_stream_id"]
                for row in artifact["seed_rows"]
                if row["family"] == family and row["seed"] == seed
            } == {f"{family}-crn-{seed}"}


def test_zero_rate_controls_and_all_cross_channel_estimands_are_exactly_zero(
    artifact: dict,
) -> None:
    allowed = {
        "ancilla_bit_flip": {
            "bit_event_rate",
            "bit_outcome_toggle_rate",
            "logical_backaction_rate",
            "faulted_label_change_rate",
        },
        "ancilla_phase_flip": {
            "phase_event_rate",
            "phase_nonzero_backaction_rate",
            "mean_abs_continuous_backaction_x",
        },
        "readout_error": {
            "readout_misclassification_rate",
            "nonzero_virtual_rotation_rate",
            "mean_abs_virtual_rotation_rad",
        },
    }
    for row in artifact["summary_rows"]:
        if row["injected_rate"] == 0.0:
            assert all(row[metric]["mean"] == 0.0 for metric in METRICS)
        for metric in set(METRICS) - allowed[row["family"]]:
            assert row[metric]["mean"] == 0.0


def test_bit_only_path_is_monotone_and_matches_analytic_rates(artifact: dict) -> None:
    rows = _family_rows(artifact, "ancilla_bit_flip")
    for metric in (
        "bit_event_rate",
        "bit_outcome_toggle_rate",
        "logical_backaction_rate",
        "faulted_label_change_rate",
    ):
        values = _means(rows, metric)
        assert all(right > left for left, right in zip(values, values[1:]))
        assert max(
            abs(row[metric]["mean"] - _analytic_expectation(row["family"], row["injected_rate"], metric))
            for row in rows
        ) < 0.003
    assert rows[-1]["logical_backaction_rate"]["mean"] > 0.035


def test_phase_only_path_has_small_backaction_without_z_toggle(artifact: dict) -> None:
    rows = _family_rows(artifact, "ancilla_phase_flip")
    for metric in (
        "phase_event_rate",
        "phase_nonzero_backaction_rate",
        "mean_abs_continuous_backaction_x",
    ):
        values = _means(rows, metric)
        assert all(right > left for left, right in zip(values, values[1:]))
    assert all(row["phase_z_basis_outcome_toggle_rate"]["mean"] == 0.0 for row in rows)
    assert all(row["logical_backaction_rate"]["mean"] == 0.0 for row in rows)
    assert all(row["faulted_label_change_rate"]["mean"] == 0.0 for row in rows)
    assert max(
        abs(
            row["mean_abs_continuous_backaction_x"]["mean"]
            - 0.01 * row["phase_event_rate"]["mean"]
        )
        for row in rows
    ) < 1e-15


def test_readout_only_path_has_rotation_without_ancilla_fault_truth(artifact: dict) -> None:
    rows = _family_rows(artifact, "readout_error")
    for metric in (
        "readout_misclassification_rate",
        "nonzero_virtual_rotation_rate",
        "mean_abs_virtual_rotation_rad",
    ):
        values = _means(rows, metric)
        assert all(right > left for left, right in zip(values, values[1:]))
    assert max(
        abs(row["readout_misclassification_rate"]["mean"] - row["injected_rate"])
        for row in rows
    ) < 0.002
    assert max(
        abs(row["mean_abs_virtual_rotation_rad"]["mean"] - 0.3 * row["injected_rate"])
        for row in rows
    ) < 0.001
    for row in rows:
        assert row["bit_event_rate"]["mean"] == 0.0
        assert row["phase_event_rate"]["mean"] == 0.0
        assert row["logical_backaction_rate"]["mean"] == 0.0


def test_uncertainty_uses_whole_seed_clusters_not_individual_cycles(
    artifact: dict,
) -> None:
    for row in artifact["summary_rows"]:
        for metric in METRICS:
            summary = row[metric]
            assert summary["resampling_unit"] == "whole_seed_cluster"
            assert summary["paired_seed_cluster_count"] == 8
            assert summary["bootstrap_replicates"] == 20000
            assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]


def test_deployable_schema_and_claim_contract_fail_closed(artifact: dict) -> None:
    assert all(row["deployable_schema_exact"] is True for row in artifact["seed_rows"])
    assert artifact["causal_contract"]["truth_visibility"] == (
        "simulator_evaluator_only_not_deployable_input"
    )
    assert artifact["estimand_contract"]["global_sensitivity_score"] == "FORBIDDEN"
    assert artifact["estimand_contract"]["numeric_65x_reproduction"] == "NOT_ATTEMPTED"
    assert artifact["estimand_contract"]["physical_memory_ler"] == "NOT_ESTABLISHED"
    assert artifact["device_calibrated"] is False
    assert artifact["experimental_hardware_used"] is False
    assert artifact["physical_memory_ler_established"] is False


def test_source_ledger_has_exact_row_count_types_and_hash(artifact: dict) -> None:
    record = artifact["source_data"]
    path = ROOT / record["path"]
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == record["row_count"] == 1960
    assert {row["row_type"] for row in rows} == {
        "parent_artifact",
        "implementation_binding",
        "primary_source_anchor",
        "channel_intervention",
        "seed_metric",
        "seed_cluster_summary",
        "contract_gate",
    }
    assert sum(row["row_type"] == "channel_intervention" for row in rows) == 144
    assert sum(row["row_type"] == "seed_metric" for row in rows) == 144 * len(METRICS)
    assert sum(row["row_type"] == "seed_cluster_summary" for row in rows) == 18 * len(METRICS)


def test_semantic_validator_rejects_channel_mixing_and_truth_leakage(
    artifact: dict,
) -> None:
    mixed = copy.deepcopy(artifact)
    row = next(row for row in mixed["seed_rows"] if row["family"] == "ancilla_bit_flip" and row["injected_rate"] == 0.04)
    row["channel_spec"]["readout_error_probability"] = 0.04
    assert "one-channel-at-a-time configuration changed" in validate_payload(mixed)

    leaked = copy.deepcopy(artifact)
    leaked["seed_rows"][0]["deployable_schema_exact"] = False
    assert "fault truth leaked into deployable schema" in validate_payload(leaked)


def test_semantic_validator_rejects_coherently_rewritten_phase_logical_path(
    artifact: dict,
) -> None:
    mutated = copy.deepcopy(artifact)
    for row in mutated["seed_rows"]:
        if row["family"] == "ancilla_phase_flip" and row["injected_rate"] > 0.0:
            row["logical_backaction_rate"] = 0.01
    for row in mutated["summary_rows"]:
        if row["family"] == "ancilla_phase_flip" and row["injected_rate"] > 0.0:
            summary = row["logical_backaction_rate"]
            summary["mean"] = summary["ci_low"] = summary["ci_high"] = 0.01
    assert "cross-channel estimands are contaminated" in validate_payload(mutated)


def test_semantic_validator_rejects_missing_cells_stale_parent_and_global_score(
    artifact: dict,
) -> None:
    missing = copy.deepcopy(artifact)
    missing["seed_rows"].pop()
    assert "family-seed-rate matrix is incomplete or duplicated" in validate_payload(missing)

    stale = copy.deepcopy(artifact)
    stale["parent_integrity"]["T2.2.2"]["sha256"] = "0" * 64
    assert "parent artifact binding is stale or failed" in validate_payload(stale)

    collapsed = copy.deepcopy(artifact)
    collapsed["global_score"] = 0.5
    assert "forbidden combined/global sensitivity score was introduced" in validate_payload(collapsed)


def test_small_seed_cells_are_deterministic_and_keep_family_semantics() -> None:
    for family in FAULT_FAMILIES:
        first = _run_seed_cell(family, 0.04, EVALUATION_SEEDS[0], cycles=512)
        second = _run_seed_cell(family, 0.04, EVALUATION_SEEDS[0], cycles=512)
        assert first == second
        assert first["deployable_schema_exact"] is True
        if family != "ancilla_bit_flip":
            assert first["bit_event_rate"] == 0.0
            assert first["logical_backaction_rate"] == 0.0
        if family != "ancilla_phase_flip":
            assert first["phase_event_rate"] == 0.0
        if family != "readout_error":
            assert first["readout_misclassification_rate"] == 0.0


@pytest.mark.parametrize(
    "change",
    [
        {"fault_families": ("ancilla_bit_flip",)},
        {"injection_rates": (0.0, 0.1)},
        {"evaluation_seeds": EVALUATION_SEEDS[:-1]},
        {"cycles_per_seed_rate": 1024},
        {"seed_cluster_bootstrap_replicates": 9999},
        {"bit_logical_given_event": 0.25},
        {"phase_backaction_scale": 0.02},
        {"virtual_rotation_max_rad": 0.3},
    ],
)
def test_preregistered_campaign_configuration_rejects_simplification(change: dict) -> None:
    with pytest.raises(ValueError):
        CampaignConfig(**change)
