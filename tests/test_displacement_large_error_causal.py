from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.algorithm_success_falsification import FALLBACK_BRANCH_ID
from cnn_fpga.benchmark.displacement_large_error_causal import (
    AMPLITUDES,
    CALIBRATION_SEEDS,
    CampaignConfig,
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    EVALUATION_SEEDS,
    LOGICAL_NOISE_PROFILES,
    PARENT_ARTIFACTS,
    build_report,
    implementation_sha256,
    inspect_parent_integrity,
    load_parent_artifacts,
    validate_payload,
    write_artifacts,
)


def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def _set_path(payload: dict, path: tuple[object, ...], value: object) -> None:
    target: object = payload
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


def test_committed_artifact_is_current_complete_and_source_bound() -> None:
    payload = _artifact()
    assert payload["task_id"] == "T5.2.1"
    assert payload["status"] == "PASS"
    assert payload["active_algorithm_branch"] == FALLBACK_BRANCH_ID
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["gate_summary"] == {
        "passed": 20,
        "total": 20,
        "failed": [],
    }
    assert len(payload["gates"]) == 20 and all(payload["gates"].values())
    assert payload["source_data"]["row_count"] == 1863
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()
    assert validate_payload(payload) == ()


def test_parent_artifacts_and_implementation_bindings_are_current() -> None:
    payload = _artifact()
    parents = load_parent_artifacts()
    integrity = inspect_parent_integrity(parents)
    assert set(integrity) == set(PARENT_ARTIFACTS)
    for task_id, path in PARENT_ARTIFACTS.items():
        record = payload["parent_integrity"][task_id]
        assert record == integrity[task_id]
        assert record["machine_pass"] is True
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for binding in payload["implementation_bindings"]:
        path = Path(binding["path"])
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_grid_and_seed_clusters_are_disjoint_from_calibration() -> None:
    config = _artifact()["config"]
    assert tuple(config["amplitudes_over_lattice"]) == AMPLITUDES
    assert tuple(config["evaluation_seeds"]) == EVALUATION_SEEDS
    assert set(EVALUATION_SEEDS).isdisjoint(CALIBRATION_SEEDS)
    assert config["shots_per_seed_amplitude"] == 4096
    assert config["cycles"] == 20
    assert config["seed_cluster_bootstrap_replicates"] == 20000
    assert config["logical_noise_profiles"] == [
        {
            "profile_id": profile.profile_id,
            "injection_sigma_over_lattice": profile.injection_sigma_over_lattice,
        }
        for profile in LOGICAL_NOISE_PROFILES
    ]


def test_every_seed_executes_every_amplitude_with_common_random_number_pairing() -> None:
    rows = _artifact()["recovery_seed_rows"]
    assert len(rows) == len(EVALUATION_SEEDS) * len(AMPLITUDES) == 136
    assert {(row["seed"], row["amplitude_over_lattice"]) for row in rows} == {
        (seed, amplitude) for seed in EVALUATION_SEEDS for amplitude in AMPLITUDES
    }
    for seed in EVALUATION_SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        assert {row["paired_stream_id"] for row in selected} == {
            f"recovery-crn-{seed}"
        }
        by_amplitude = {row["amplitude_over_lattice"]: row for row in selected}
        for amplitude in AMPLITUDES[:8]:
            mirror = 0.5 - amplitude
            assert by_amplitude[amplitude]["trace_sha256"] == by_amplitude[mirror][
                "trace_sha256"
            ]


def test_recovery_depth_and_observed_e_run_are_midpoint_peaked_and_symmetric() -> None:
    rows = _artifact()["recovery_summary"]
    assert len(rows) == 17
    depth = [row["mean_initial_recovery_depth"]["mean"] for row in rows]
    e_run = [row["mean_observed_same_quadrature_max_e_run"]["mean"] for row in rows]
    assert depth[:9] == sorted(depth[:9])
    assert depth[8:] == sorted(depth[8:], reverse=True)
    assert e_run[:9] == sorted(e_run[:9])
    assert e_run[8:] == sorted(e_run[8:], reverse=True)
    assert depth == pytest.approx(list(reversed(depth)), abs=0.0)
    assert e_run == pytest.approx(list(reversed(e_run)), abs=0.0)
    midpoint = rows[8]
    assert midpoint["amplitude_over_lattice"] == 0.25
    assert midpoint["logical_distance"] == 1.0
    assert midpoint["mean_initial_recovery_depth"]["mean"] == 6.0
    assert midpoint["mean_observed_same_quadrature_max_e_run"]["mean"] == pytest.approx(
        4.845123291015625
    )


def test_unaffected_quadrature_and_recovery_horizon_remain_separate_controls() -> None:
    rows = _artifact()["recovery_summary"]
    assert max(row["unaffected_e_probability_max"]["ci_high"] for row in rows) < 0.06
    assert all(row["recovered_fraction_by_horizon"]["mean"] == 1.0 for row in rows)
    assert _artifact()["estimand_contract"][
        "logical_failure_is_recovery_censoring"
    ] is False


def test_logical_seed_matrix_uses_two_profiles_and_evaluator_only_targets() -> None:
    payload = _artifact()
    rows = payload["logical_seed_rows"]
    assert len(rows) == 272
    assert {
        (row["seed"], row["profile_id"], row["amplitude_over_lattice"])
        for row in rows
    } == {
        (seed, profile.profile_id, amplitude)
        for seed in EVALUATION_SEEDS
        for profile in LOGICAL_NOISE_PROFILES
        for amplitude in AMPLITUDES
    }
    for seed in EVALUATION_SEEDS:
        for profile in LOGICAL_NOISE_PROFILES:
            selected = [
                row
                for row in rows
                if row["seed"] == seed and row["profile_id"] == profile.profile_id
            ]
            assert len({row["jitter_trace_sha256"] for row in selected}) == 1
    assert payload["causal_contract"]["logical_truth_visibility"] == (
        "evaluator_only_not_controller_input"
    )


@pytest.mark.parametrize(
    ("profile_id", "left_near", "midpoint", "right_near"),
    [
        ("primary_sigma_0p040", 0.219330, 0.495300, 0.215057),
        ("confirmation_sigma_0p025", 0.105927, 0.497864, 0.102692),
    ],
)
def test_nearest_operation_logical_failure_is_midpoint_peaked_in_both_profiles(
    profile_id: str, left_near: float, midpoint: float, right_near: float
) -> None:
    rows = [
        row for row in _artifact()["logical_summary"] if row["profile_id"] == profile_id
    ]
    rates = [row["nearest_operation_logical_failure_rate"]["mean"] for row in rows]
    assert rates[:9] == sorted(rates[:9])
    assert rates[8:] == sorted(rates[8:], reverse=True)
    assert rates[7] == pytest.approx(left_near, abs=5e-7)
    assert rates[8] == pytest.approx(midpoint, abs=5e-7)
    assert rates[9] == pytest.approx(right_near, abs=5e-7)
    assert rows[8]["nearest_operation_logical_failure_rate"]["ci_low"] > 0.48
    assert rows[8]["nearest_operation_logical_failure_rate"]["ci_high"] < 0.52


def test_identity_reference_logical_flip_rate_exposes_right_endpoint_flip() -> None:
    for profile in LOGICAL_NOISE_PROFILES:
        rows = [
            row
            for row in _artifact()["logical_summary"]
            if row["profile_id"] == profile.profile_id
        ]
        rates = [row["identity_reference_logical_flip_rate"]["mean"] for row in rows]
        assert rates == sorted(rates)
        assert rates[0] == 0.0
        assert rates[-1] == 1.0
        assert rows[0]["nearest_nominal_logical_parity"] == 0
        assert rows[-1]["nearest_nominal_logical_parity"] == 1


def test_logical_monte_carlo_agrees_with_independent_boundary_formula() -> None:
    rows = _artifact()["logical_summary"]
    gaps = [
        abs(
            row["nearest_operation_logical_failure_rate"]["mean"]
            - row["analytic_nearest_operation_failure_rate"]["mean"]
        )
        for row in rows
    ]
    assert max(gaps) < 0.005
    # The 34 pointwise cluster intervals are not simultaneous confidence bands;
    # require the exact midpoint identity plus the preregistered uniform error gate.
    for profile in LOGICAL_NOISE_PROFILES:
        midpoint = next(
            row
            for row in rows
            if row["profile_id"] == profile.profile_id
            and row["amplitude_over_lattice"] == 0.25
        )
        assert midpoint["nearest_operation_logical_failure_rate"]["ci_low"] <= 0.5
        assert midpoint["nearest_operation_logical_failure_rate"]["ci_high"] >= 0.5


def test_seed_cluster_ci_never_pretends_shots_are_independent_clusters() -> None:
    for section in ("recovery_summary", "logical_summary"):
        for row in _artifact()[section]:
            for value in row.values():
                if isinstance(value, dict) and "resampling_unit" in value:
                    assert value["paired_seed_cluster_count"] == 8
                    assert value["bootstrap_replicates"] == 20000
                    assert value["resampling_unit"] == "whole_seed_cluster"
                    assert value["ci_low"] <= value["mean"] <= value["ci_high"]


def test_causal_and_claim_contracts_prevent_channel_and_evidence_mixing() -> None:
    payload = _artifact()
    causal = payload["causal_contract"]
    assert causal["changed_channel"] == "nominal_displacement_amplitude_only"
    assert set(causal["fixed_channels"]) == {
        "recovery transition kernel",
        "readout confusion",
        "reset kernel",
        "fault quadrature",
        "horizon",
        "shot count",
    }
    assert payload["device_calibrated"] is False
    assert payload["experimental_hardware_used"] is False
    assert payload["physical_memory_ler_established"] is False
    assert payload["estimand_contract"]["physical_memory_ler"] == "NOT_ESTABLISHED"


def test_source_ledger_has_exact_rows_and_row_types() -> None:
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1863
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["row_type"]] = counts.get(row["row_type"], 0) + 1
    assert counts == {
        "parent_artifact": 4,
        "implementation_binding": 3,
        "recovery_seed_metric": 816,
        "logical_seed_metric": 816,
        "recovery_cluster_summary": 102,
        "logical_cluster_summary": 102,
        "contract_gate": 20,
    }


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("config", "amplitudes_over_lattice", 1), 0.04),
        (("config", "evaluation_seeds", 0), 2026071405),
        (("recovery_seed_rows", 0, "logical_distance"), 0.2),
        (("logical_seed_rows", 0, "nearest_operation_logical_failure_rate"), 1.2),
        (("logical_seed_rows", 0, "nearest_nominal_logical_parity"), 1),
        (("recovery_summary", 0, "mean_initial_recovery_depth", "mean"), 1.0),
        (("logical_summary", 0, "nearest_operation_logical_failure_rate", "mean"), 0.2),
        (("estimand_contract", "logical_failure_is_recovery_censoring"), True),
        (("estimand_contract", "physical_memory_ler"), "ESTABLISHED"),
        (("causal_contract", "changed_channel"), "displacement_and_readout"),
        (("causal_contract", "logical_truth_visibility"), "controller_input"),
        (("active_algorithm_branch",), "learned_decoder"),
        (("device_calibrated",), True),
        (("experimental_hardware_used",), True),
        (("physical_memory_ler_established",), True),
        (("parent_integrity", "T2.0.5", "sha256"), "0" * 64),
        (("implementation_bindings", 0, "sha256"), "0" * 64),
        (("source_binding", "sha256"), "0" * 64),
        (("implementation_sha256",), "0" * 64),
        (("gates", "formal_grid_has_17_points_and_midpoint"), False),
    ],
)
def test_semantic_validator_rejects_evidence_mixing_and_stale_bindings(
    path: tuple[object, ...], replacement: object
) -> None:
    payload = copy.deepcopy(_artifact())
    _set_path(payload, path, replacement)
    assert validate_payload(payload)


def test_semantic_validator_rejects_missing_seed_rows() -> None:
    payload = copy.deepcopy(_artifact())
    payload["recovery_seed_rows"].pop()
    assert validate_payload(payload)


def test_semantic_validator_rejects_coherently_rewritten_midpoint_trend() -> None:
    payload = copy.deepcopy(_artifact())
    for row in payload["logical_seed_rows"]:
        if row["profile_id"] == "primary_sigma_0p040" and row["amplitude_over_lattice"] == 0.25:
            row["nearest_operation_logical_failure_rate"] = 0.1
    midpoint = next(
        row
        for row in payload["logical_summary"]
        if row["profile_id"] == "primary_sigma_0p040"
        and row["amplitude_over_lattice"] == 0.25
    )
    midpoint["nearest_operation_logical_failure_rate"]["mean"] = 0.1
    assert "nearest-operation logical-failure trend changed" in validate_payload(payload)


def test_build_report_fails_closed_when_parent_is_missing() -> None:
    parents = load_parent_artifacts()
    integrity = inspect_parent_integrity(parents)
    parents.pop("T2.0.5")
    with pytest.raises(ValueError, match="missing parent artifacts"):
        build_report(parents, integrity, CampaignConfig(shots_per_seed_amplitude=1024))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"amplitudes_over_lattice": AMPLITUDES[:-1]}, "17-point"),
        ({"evaluation_seeds": EVALUATION_SEEDS[:-1]}, "seed clusters"),
        ({"shots_per_seed_amplitude": 512}, "at least 1024"),
        ({"seed_cluster_bootstrap_replicates": 9999}, "at least 10000"),
        ({"logical_operation_spacing_over_lattice": 1.0}, "l_S/2"),
    ],
)
def test_invalid_preregistration_fails_closed(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        CampaignConfig(**kwargs)


def test_reduced_shot_writer_is_deterministic(tmp_path: Path) -> None:
    config = CampaignConfig(
        shots_per_seed_amplitude=1024,
        seed_cluster_bootstrap_replicates=10000,
    )
    first_json = tmp_path / "first.json"
    first_csv = tmp_path / "first.csv"
    second_json = tmp_path / "second.json"
    second_csv = tmp_path / "second.csv"
    first = write_artifacts(
        artifact_path=first_json,
        source_data_path=first_csv,
        config=config,
    )
    second = write_artifacts(
        artifact_path=second_json,
        source_data_path=second_csv,
        config=config,
    )
    assert first["status"] == second["status"] == "PASS"
    assert first["contract_sha256"] == second["contract_sha256"]
    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first["source_data"]["sha256"] == second["source_data"]["sha256"]
    assert validate_payload(first) == validate_payload(second) == ()
