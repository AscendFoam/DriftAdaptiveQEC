from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import held_out_ood_validation as ood


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / ood.DEFAULT_ARTIFACT
SOURCE = ROOT / ood.DEFAULT_SOURCE_DATA


def _load() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rehash(payload: dict) -> None:
    payload["contract_sha256"] = ood._canonical_sha256(ood._contract_view(payload))


def test_formal_config_is_frozen_and_seed_groups_are_disjoint() -> None:
    config = ood.HeldOutOODConfig()
    groups = (
        config.drift_evaluation_seeds,
        config.measurement_evaluation_seeds,
        config.leakage_evaluation_seeds,
        config.communication_evaluation_seeds,
    )
    assert all(len(group) == 8 for group in groups)
    assert len(set().union(*(set(group) for group in groups))) == 32
    with pytest.raises(ValueError, match="formal leakage_ood_rates changed"):
        ood.HeldOutOODConfig(leakage_ood_rates=(0.003, 0.006, 0.02))
    with pytest.raises(ValueError, match="formal drift_windows changed"):
        ood.HeldOutOODConfig(drift_windows=16)


def test_drift_scenarios_include_unseen_family_and_true_range_extrapolation() -> None:
    artifact = _load()
    lane = artifact["drift_lane"]
    assert set(lane["registered_ood_scenario_ids"]) == set(ood.DRIFT_SCENARIOS)
    assert not set(ood.DRIFT_SCENARIOS[:2]) & set(lane["parent_scenario_ids"])
    extrapolated = next(
        row
        for row in lane["scenario_aggregates"]
        if row["scenario_id"] == "compound_range_extrapolation"
    )
    assert sum(extrapolated["exceeds_parent_envelope"].values()) >= 5
    assert extrapolated["state_envelope"]["p_outlier_max"] > lane[
        "parent_state_envelope"
    ]["p_outlier_max"]
    assert extrapolated["state_envelope"]["outlier_scale_max"] > lane[
        "parent_state_envelope"
    ]["outlier_scale_max"]


def test_drift_cells_share_trace_across_all_registered_decoders() -> None:
    artifact = _load()
    rows = artifact["drift_lane"]["seed_rows"]
    assert len(rows) == 24
    for row in rows:
        assert row["shared_trace_for_all_methods"] is True
        assert len(row["trace_sha256"]) == 64
        for method in ood.DECODER_METHODS:
            assert 0.0 <= row[f"{method}_error_rate"] <= 1.0
    assert all(
        aggregate["unique_trace_hashes"] == 8
        for aggregate in artifact["drift_lane"]["scenario_aggregates"]
    )


def test_telegraph_negative_result_is_retained() -> None:
    artifact = _load()
    telegraph = next(
        row
        for row in artifact["drift_lane"]["scenario_aggregates"]
        if row["scenario_id"] == "stochastic_telegraph_unseen_family"
    )
    assert telegraph["paired_contrasts"]["static_minus_ewma_error_rate"]["estimate"] < 0.0
    assert telegraph["paired_contrasts"]["static_minus_kalman_error_rate"]["estimate"] < 0.0
    assert artifact["system_robustness_status"] == "NOT_ESTABLISHED_LANE_LOCAL_ONLY"


def test_measurement_confusion_is_asymmetric_protocol_native_and_calibrated() -> None:
    artifact = _load()
    lane = artifact["measurement_confusion_lane"]
    assert lane["registered_confusion_matrices"] == {
        key: [list(row) for row in value]
        for key, value in ood.MEASUREMENT_CONFUSION.items()
    }
    assert len(lane["seed_rows"]) == 24
    assert all(
        row["confusion_rates_within_preregistered_tolerance"]
        for row in lane["scenario_aggregates"]
    )
    assert all(
        row["ancilla_bit_phase_event_count"] == 0
        and row["faulted_label_change_count"] == 0
        and row["deployable_schema_exact"]
        for row in lane["seed_rows"]
    )


def test_leakage_rates_are_unseen_extrapolated_and_burden_is_monotone() -> None:
    artifact = _load()
    lane = artifact["leakage_rate_lane"]
    assert not set(lane["registered_ood_rates"]) & set(lane["parent_rate_grid"])
    assert max(lane["registered_ood_rates"]) > max(lane["parent_rate_grid"])
    occupancy = [
        row["hidden_leakage_occupancy_seed_cluster_ci"]["estimate"]
        for row in lane["rate_aggregates"]
    ]
    assert all(a < b for a, b in zip(occupancy, occupancy[1:]))
    assert all(
        row["injection_rate_within_preregistered_tolerance"]
        for row in lane["rate_aggregates"]
    )


def test_communication_patterns_detect_transitions_and_preserve_integrity() -> None:
    artifact = _load()
    lane = artifact["communication_lane"]
    assert [row["name"] for row in lane["scenarios"]] == [
        "reference",
        *ood.COMMUNICATION_SCENARIOS,
    ]
    assert len(lane["per_seed_results"]) == 32
    for row in lane["per_seed_results"]:
        integrity = row["integrity"]
        assert integrity["active_version_monotonic"]
        assert integrity["maximum_version_step"] <= 1
        assert integrity["all_arrays_finite"]
        assert not integrity["slow_estimator_uses_hidden_truth"]
        if row["scenario"] != "reference":
            assert row["event_counts"]["communication_pause_started"] > 0
            assert row["event_counts"]["communication_pause_ended"] > 0


def test_short_communication_outage_null_and_compound_degradation_are_both_retained() -> None:
    artifact = _load()
    aggregates = {
        row["scenario"]: row
        for row in artifact["communication_lane"]["scenario_aggregates"]
    }
    assert aggregates["periodic_micro_outages"]["paired_ler_minus_reference"]["mean"] == 0.0
    compound = aggregates["communication_jitter_burst_compound"]
    assert compound["paired_ler_minus_reference"]["ci_low"] > 0.0
    assert compound["paired_availability_minus_reference"]["ci_high"] < 0.0


def test_artifact_recomputes_all_gates_and_validates_cleanly() -> None:
    artifact = _load()
    assert ood.validate_artifact(artifact) == ()
    parents = ood.load_parent_artifacts()
    assert artifact["gates"] == ood._compute_gates(artifact, parents)
    assert artifact["gate_summary"] == {"passed": 20, "total": 20}


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    (
        ("measurement", "stored gates do not match recomputed evidence gates"),
        ("leakage", "stored gates do not match recomputed evidence gates"),
        ("communication", "stored gates do not match recomputed evidence gates"),
        ("cross_lane", "forbidden cross-lane robustness/ranking field present"),
    ),
)
def test_mutations_fail_closed(mutation: str, expected_fragment: str) -> None:
    artifact = copy.deepcopy(_load())
    if mutation == "measurement":
        row = artifact["measurement_confusion_lane"]["scenario_aggregates"][0]
        row["empirical_g_to_e"] = 0.99
    elif mutation == "leakage":
        row = artifact["leakage_rate_lane"]["rate_aggregates"][0]
        row["empirical_higher_injection_probability_seed_cluster_ci"]["estimate"] = 0.5
    elif mutation == "communication":
        row = next(
            item
            for item in artifact["communication_lane"]["per_seed_results"]
            if item["scenario"] != "reference"
        )
        row["event_counts"]["communication_pause_started"] = 0
    else:
        artifact["cross_lane_score"] = 1.0
    # Rehashing proves that semantic recomputation, rather than only the
    # top-level contract digest, rejects the mutation.
    _rehash(artifact)
    errors = ood.validate_artifact(artifact)
    assert any(expected_fragment in error for error in errors)


def test_source_data_is_complete_and_byte_hash_bound() -> None:
    artifact = _load()
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == artifact["source_data"]["row_count"] == 280
    assert ood._sha256(SOURCE) == artifact["source_data"]["csv_sha256"]
    assert {
        "parent_binding",
        "drift_seed_method",
        "drift_aggregate",
        "measurement_seed",
        "measurement_aggregate",
        "leakage_seed",
        "leakage_aggregate",
        "communication_seed",
        "communication_aggregate",
        "gate",
    } == {row["row_type"] for row in rows}


def test_no_cross_lane_ranking_or_device_claim_is_present() -> None:
    artifact = _load()
    assert not ood._forbidden_cross_lane_key(artifact)
    assert artifact["device_robustness_status"] == "NOT_ESTABLISHED_NO_TARGET_HARDWARE"
    assert artifact["communication_lane"]["target_hardware_measured"] is False
    assert "uncertainty-gated fallback benefit before T5.4.2" in artifact[
        "claim_boundary"
    ]["forbidden"]
