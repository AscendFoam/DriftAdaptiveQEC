from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import horizon_extrapolation_validation as audit
from cnn_fpga.benchmark import low_dimensional_student_distillation as student_parent
from cnn_fpga.control.low_dimensional_recurrence import (
    RESIDUAL_BOUNDS,
    LowDimensionalRecurrenceArtifact,
)
from physics.nmf_directional_ranking import DirectionalRankingConfig, build_policy


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/t5_4_5_horizon_extrapolation_validation.json"
CHECKPOINT = ROOT / "docs/t5_4_5_horizon_extrapolation_candidates.pt"
SOURCE = ROOT / "docs/t5_4_5_horizon_extrapolation_validation_source_data.csv"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rehash(mutated: dict) -> dict:
    mutated["contract_sha256"] = audit._canonical_sha256(
        audit._contract_view(mutated)
    )
    return mutated


def test_formal_artifact_is_semantically_valid(report: dict) -> None:
    assert report["status"] == "PASS"
    assert report["verdict"] == (
        "QUALIFIED_LONG_RECURRENCE_PASS_PHYSICAL_GAIN_NOT_ESTABLISHED"
    )
    assert report["gate_summary"] == {"passed": 21, "total": 21}
    assert audit.validate_artifact(report) == ()


def test_all_parent_and_implementation_bindings_are_current(report: dict) -> None:
    assert len(report["parent_bindings"]) == 5
    assert all(row["machine_pass"] for row in report["parent_bindings"])
    for row in report["parent_bindings"] + report["implementation_bindings"]:
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert row["sha256"] == audit._sha256(path)


def test_production_design_requires_exact_registered_horizons() -> None:
    audit.validate_production_design(audit.HorizonExtrapolationConfig())
    with pytest.raises(ValueError, match="1e3/1e5/1e6"):
        audit.validate_production_design(
            audit.HorizonExtrapolationConfig(
                deployment_horizons_cycles=(1_000, 10_000, 100_000)
            )
        )


def test_training_horizon_sweep_is_real_and_evaluation_blind(report: dict) -> None:
    sweep = report["training_sweep"]
    assert sweep["fresh_candidate_count"] == 9
    assert len(sweep["candidate_records"]) == 9
    assert set(map(int, sweep["selected_by_horizon"])) == {2, 5, 10, 32}
    assert sweep["evaluation_never_used_for_selection"] is True
    for horizon in (2, 5, 10):
        rows = [
            row
            for row in sweep["candidate_records"]
            if row["training_horizon_cycles"] == horizon
        ]
        assert len(rows) == 3
        assert sum(row["selected_by_validation"] for row in rows) == 1
        assert all(row["selection_split"] == "validation_only" for row in rows)
        assert all(row["evaluation_opened_after_selection"] for row in rows)
    assert sweep["selected_by_horizon"]["32"]["source"] == (
        "frozen_T4.4.3_strict_split_production_student"
    )


def test_training_prefix_datasets_are_hash_bound(report: dict) -> None:
    dataset = report["training_sweep"]["dataset"]
    assert dataset["max_training_half_cycles"] == 64
    assert dataset["trajectories_per_split"] == 256
    assert set(dataset["outcome_hashes"]) == {"training", "validation", "evaluation"}
    assert set(dataset["target_hashes"]) == {"training", "validation", "evaluation"}
    assert len(set(dataset["outcome_hashes"].values())) == 3
    assert "exact prefixes" in dataset["prefix_rule"]


def test_two_cycle_fit_exposes_negative_extrapolation(report: dict) -> None:
    selection = report["training_sweep"]["selected_by_horizon"]
    h2 = selection["2"]["evaluation_32_cycle_metrics"]["mse"]
    h10 = selection["10"]["evaluation_32_cycle_metrics"]["mse"]
    h32 = selection["32"]["evaluation_32_cycle_metrics"]["mse"]
    assert h2 > report["config"]["maximum_imitation_mse"]
    assert h10 < h2 / 5.0
    assert h32 < h2 / 10.0


def test_all_registered_streams_execute_two_million_updates(report: dict) -> None:
    streams = report["stream_registry"]
    assert len(streams) == 8
    assert {row["family"] for row in streams} == set(audit.STREAM_FAMILIES)
    assert all(row["updates_executed"] == 2_000_000 for row in streams)
    assert all(row["cycles_executed"] == 1_000_000 for row in streams)
    assert len({row["stream_id"] for row in streams}) == 8
    assert len({row["outcome_sha256"] for row in streams}) == 8
    execution = report["execution_summary"]
    assert execution["total_teacher_updates_per_precision"] == 16_000_000
    assert execution["total_student_updates_per_precision"] == 64_000_000


def test_sampling_contract_covers_actions_without_shortening_state_scan(
    report: dict,
) -> None:
    sampling = report["sampling_contract"]
    assert sampling["sample_index_count"] >= 13_000
    assert sampling["state_update_sampling"] == (
        "none; every registered half-cycle is executed"
    )
    assert set(map(int, sampling["reset_anchors_half_cycles"])) == {
        1_000,
        100_000,
        1_000_000,
    }


def test_sequence_gru_is_equivalent_to_native_gru_cell() -> None:
    torch = pytest.importorskip("torch")
    config = DirectionalRankingConfig(device="cpu", real_dtype="float64")
    model = build_policy("nmf", config, seed=545001)
    sequence_gru = audit._copy_gru_cell_to_sequence(
        model.gru, dtype=torch.float64, device="cpu"
    )
    rng = np.random.default_rng(545002)
    outcomes = torch.as_tensor(
        rng.integers(0, 2, size=(3, 29)), dtype=torch.float64
    )
    encoded = 2.0 * outcomes.unsqueeze(-1) - 1.0
    with torch.no_grad():
        sequence_states, _ = sequence_gru(encoded)
        hidden = torch.zeros((3, 10), dtype=torch.float64)
        native = []
        for step in range(encoded.shape[1]):
            hidden = model.gru(encoded[:, step], hidden)
            native.append(hidden)
    native_states = torch.stack(native, dim=1)
    assert torch.max(torch.abs(native_states - sequence_states)).item() < 1.0e-12


def test_compiled_student_scan_matches_exported_runtime() -> None:
    artifact = LowDimensionalRecurrenceArtifact.from_dict(
        json.loads(
            (ROOT / "docs/t4_4_3_low_dimensional_student.json").read_text(
                encoding="utf-8"
            )
        )
    )
    parameters = audit._parameters_from_artifact(artifact)
    rng = np.random.default_rng(545003)
    outcomes = rng.integers(0, 2, size=(4, 37), dtype=np.uint8)
    sample_indices = np.arange(38, dtype=np.int64)
    scan = audit._student_long_scan(
        outcomes,
        [parameters],
        sample_indices,
        (37,),
        dtype=np.float64,
    )
    expected = student_parent._predict_exported(artifact, outcomes)
    assert np.max(np.abs(scan["actions"][0] - expected)) < 1.0e-12


def test_hidden_and_student_states_are_actually_bounded(report: dict) -> None:
    rows = report["stability_rows"]
    teacher = [row for row in rows if row["model_id"] == "teacher_gru10"]
    students = [row for row in rows if row["model_id"] != "teacher_gru10"]
    assert len(teacher) == 8 * 3
    assert len(students) == 4 * 8 * 3
    assert max(row["maximum_absolute_state"] for row in teacher) < 0.4
    assert all(
        row["maximum_absolute_state"] <= row["analytic_bound"] + 1.0e-12
        for row in students
    )
    assert max(
        row["maximum_absolute_state"] / row["analytic_bound"] for row in students
    ) <= 1.0


def test_sampled_actions_stay_inside_hard_bounds(report: dict) -> None:
    assert report["claim_boundary"]["residual_bounds"] == list(RESIDUAL_BOUNDS)
    assert max(
        row["maximum_normalized_sampled_action"]
        for row in report["stability_rows"]
    ) < 0.27


def test_float32_shadow_remains_close_through_one_million_cycles(
    report: dict,
) -> None:
    rows = report["numeric_rows"]
    assert len(rows) == 5 * 8 * 3
    assert {row["deployment_horizon_cycles"] for row in rows} == {
        1_000,
        100_000,
        1_000_000,
    }
    maximum = max(row["maximum_float32_float64_action_error"] for row in rows)
    assert maximum < 2.0e-6
    assert maximum < report["config"]["maximum_float32_action_error"]


def test_long_horizon_action_performance_keeps_short_horizon_failure(
    report: dict,
) -> None:
    rows = report["performance_aggregate"]
    assert len(rows) == 4 * 3
    h2 = [row for row in rows if row["model_id"] == "fresh_h2_student"]
    h10 = [row for row in rows if row["model_id"] == "fresh_h10_student"]
    h32 = [row for row in rows if row["model_id"] == "production_h32_student"]
    assert all(row["worst_stream_id"] == "all_e_boundary-seed-0" for row in h2)
    assert all(row["worst_stream_mse"] > 8.0e-4 for row in h2)
    threshold = report["config"]["maximum_imitation_mse"]
    assert all(row["worst_stream_mse"] < threshold for row in h10 + h32)
    assert max(row["mean_stream_mse"] for row in h32) < 5.1e-6


def test_reset_sensitivity_is_intervened_and_recovers(report: dict) -> None:
    rows = report["reset_rows"]
    assert len(rows) == 5 * 8 * 3
    assert all(row["recovered_within_window"] for row in rows)
    assert max(row["recovery_half_cycles"] for row in rows) <= 20
    assert max(row["recovery_half_cycles"] for row in rows) < report["config"][
        "maximum_reset_recovery_half_cycles"
    ]
    assert all(np.isfinite(row["terminal_action_rmse"]) for row in rows)


def test_checkpoint_contains_all_nine_fresh_models(report: dict) -> None:
    torch = pytest.importorskip("torch")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["config_contract_hash"] == report["config_contract_hash"]
    assert len(payload["fresh_models"]) == 9
    assert set(payload["fresh_models"]) == {
        f"h{horizon}_r{restart}" for horizon in (2, 5, 10) for restart in range(3)
    }
    assert report["checkpoint"]["sha256"] == audit._sha256(CHECKPOINT)


def test_source_csv_is_complete_and_byte_bound(report: dict) -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["source_data"]["row_count"] == 521
    assert report["source_data"]["csv_sha256"] == audit._sha256(SOURCE)
    assert {row["row_type"] for row in rows} >= {
        "training_candidate",
        "stream",
        "stream_performance",
        "aggregate_performance",
        "state_stability",
        "numeric_shadow",
        "reset_intervention",
        "gate",
    }
    gate_rows = [row for row in rows if row["row_type"] == "gate"]
    assert len(gate_rows) == 21
    assert all(row["value"] == "1" for row in gate_rows)


def test_claim_boundary_does_not_turn_action_mse_into_physical_gain(
    report: dict,
) -> None:
    boundary = report["claim_boundary"]
    assert boundary["long_horizon_physical_gain_established"] is False
    assert boundary["physical_memory_ler_established"] is False
    assert boundary["leakage_robustness_established"] is False
    assert boundary["device_calibrated"] is False
    assert boundary["hardware_measured"] is False
    assert "not extrapolated by action MSE" in boundary[
        "parent_10_cycle_physical_gain_role"
    ]


@pytest.mark.parametrize(
    "mutation",
    (
        "shorten_stream",
        "evaluation_selection",
        "break_state_bound",
        "hide_worst_stream",
        "break_reset_recovery",
        "promote_physical_gain",
    ),
)
def test_semantic_validator_rejects_demo_shortcuts(
    report: dict, mutation: str
) -> None:
    changed = copy.deepcopy(report)
    if mutation == "shorten_stream":
        changed["stream_registry"][0]["updates_executed"] = 2_000
    elif mutation == "evaluation_selection":
        changed["training_sweep"]["selected_by_horizon"]["10"][
            "evaluation_used_for_selection"
        ] = True
    elif mutation == "break_state_bound":
        row = next(
            row
            for row in changed["stability_rows"]
            if row["model_id"] != "teacher_gru10"
        )
        row["maximum_absolute_state"] = 2.0 * row["analytic_bound"]
    elif mutation == "hide_worst_stream":
        row = next(
            row
            for row in changed["performance_aggregate"]
            if row["model_id"] == "fresh_h10_student"
        )
        row["worst_stream_mse"] = 1.0
        row["worst_stream_id"] = ""
    elif mutation == "break_reset_recovery":
        changed["reset_rows"][0]["recovered_within_window"] = False
        changed["reset_rows"][0]["recovery_half_cycles"] = None
    elif mutation == "promote_physical_gain":
        changed["claim_boundary"]["long_horizon_physical_gain_established"] = True
    errors = audit.validate_artifact(_rehash(changed), check_files=False)
    assert errors
    assert any("stored gates" in error for error in errors)
