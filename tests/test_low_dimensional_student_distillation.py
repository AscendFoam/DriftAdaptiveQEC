from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.low_dimensional_student_distillation import (
    DEFAULT_ARTIFACT,
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_DATA,
    DEFAULT_STUDENT,
    LowDimensionalDistillationConfig,
    _vectorized_recurrence_states,
    implementation_sha256,
    run_low_dimensional_student_distillation,
    validate_production_design,
)
from cnn_fpga.control.low_dimensional_recurrence import (
    RESIDUAL_BOUNDS,
    LowDimensionalRecurrenceArtifact,
)


def _report() -> dict[str, object]:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"candidate_dimensions": (2, 4)}, "start at one"),
        ({"candidate_dimensions": (1, 4, 2)}, "sorted"),
        ({"candidate_dimensions": (1, 17)}, "interpretable cap"),
        ({"restart_seeds": (1, 1)}, "unique"),
        ({"training_seed": 1, "validation_seed": 1}, "disjoint"),
        ({"training_epochs": 4, "validation_interval": 5}, "must not exceed"),
        ({"minimum_zero_mse_reduction_fraction": 1.0}, "must lie"),
        ({"device": "tpu"}, "cpu or cuda"),
    ),
)
def test_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        LowDimensionalDistillationConfig(**kwargs)  # type: ignore[arg-type]


def test_production_design_rejects_demo_or_incomplete_search() -> None:
    with pytest.raises(ValueError, match="half_cycles"):
        validate_production_design(
            LowDimensionalDistillationConfig(
                half_cycles=8,
                trajectories_per_split=8,
                training_epochs=8,
                validation_interval=1,
                candidate_dimensions=(1,),
                restart_seeds=(1,),
                device="cpu",
            )
        )
    with pytest.raises(ValueError, match="1/2/4"):
        validate_production_design(
            LowDimensionalDistillationConfig(candidate_dimensions=(1, 2))
        )
    with pytest.raises(ValueError, match="three restarts"):
        validate_production_design(
            LowDimensionalDistillationConfig(restart_seeds=(1, 2))
        )


def test_vectorized_training_recurrence_matches_literal_step_loop() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(44303)
    initial = torch.randn(4, generator=generator, dtype=torch.float64)
    decays = 0.20 + 0.75 * torch.rand(
        (2, 4), generator=generator, dtype=torch.float64
    )
    saturations = torch.randn((2, 4), generator=generator, dtype=torch.float64)
    outcomes = torch.randint(
        0, 2, (7, 64), generator=generator, dtype=torch.int64
    )
    vectorized = _vectorized_recurrence_states(
        initial, decays, saturations, outcomes
    )
    state = initial[None, :].expand(outcomes.shape[0], -1)
    literal = [state]
    for step in range(outcomes.shape[1]):
        indexed_decay = decays[outcomes[:, step]]
        indexed_saturation = saturations[outcomes[:, step]]
        state = indexed_decay * state + (1.0 - indexed_decay) * indexed_saturation
        literal.append(state)
    torch.testing.assert_close(
        vectorized,
        torch.stack(literal, dim=1),
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_committed_report_passes_every_gate_and_is_source_bound() -> None:
    payload = _report()
    gates = payload["gates"]
    assert payload["status"] == "PASS"
    assert isinstance(gates, dict) and len(gates) >= 16 and all(gates.values())
    assert payload["gate_summary"] == {
        "passed": len(gates),
        "total": len(gates),
        "failed": [],
    }
    assert payload["implementation_sha256"] == implementation_sha256()
    for metadata, path in (
        (payload["checkpoint"], DEFAULT_CHECKPOINT),
        (payload["source_data"], DEFAULT_SOURCE_DATA),
        (payload["student_artifact"], DEFAULT_STUDENT),
    ):
        key = "file_sha256" if path == DEFAULT_STUDENT else "sha256"
        assert metadata[key] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_search_retains_all_dimensions_restarts_epochs_and_gradients() -> None:
    payload = _report()
    config = payload["config"]
    records = payload["training_records"]
    expected = len(config["candidate_dimensions"]) * len(config["restart_seeds"])
    assert len(records) == expected == 9
    assert {(row["dimension"], row["restart_index"]) for row in records} == {
        (dimension, restart)
        for dimension in (1, 2, 4)
        for restart in range(3)
    }
    assert all(len(row["training_curve"]) == config["training_epochs"] for row in records)
    assert all(row["parameter_count"] == 20 * row["dimension"] + 15 for row in records)
    assert all(row["all_parameter_tensors_receive_finite_nonzero_gradient"] for row in records)
    assert all(row["initial_state_sha256"] != row["checkpoint_sha256"] for row in records)
    assert all(row["optimizer_global_convergence_claimed"] is False for row in records)


def test_restart_and_dimension_selection_are_validation_only() -> None:
    payload = _report()
    records = payload["training_records"]
    best = {}
    for dimension in (1, 2, 4):
        best[dimension] = min(
            (row for row in records if row["dimension"] == dimension),
            key=lambda row: row["best_validation_mse"],
        )
    selection = payload["selection"]
    threshold = selection["dimension_eligibility_threshold"]
    eligible = sorted(
        dimension
        for dimension, row in best.items()
        if row["best_validation_mse"] <= threshold
    )
    assert selection["evaluation_blind"] is True
    assert selection["eligible_dimensions"] == eligible
    assert selection["selected_dimension"] == min(eligible)
    assert selection["selected_restart"] == best[min(eligible)]["restart_index"]
    assert "evaluation" not in selection["rule"]


def test_evaluation_reports_error_dimension_and_strong_comparators() -> None:
    payload = _report()
    selected = payload["comparisons"]["evaluation"]["selected_student"]
    zero = payload["comparisons"]["evaluation"]["zero_residual"]
    latest = payload["comparisons"]["evaluation"]["latest_only"]
    legacy = payload["comparisons"]["evaluation"]["legacy_t4_1_5_student"]
    required = payload["config"]["minimum_zero_mse_reduction_fraction"]
    assert selected["mse"] <= (1.0 - required) * zero["mse"]
    assert selected["mse"] < latest["mse"]
    assert selected["mse"] < legacy["mse"]
    assert len(selected["per_parameter_mse"]) == 15
    assert set(payload["candidate_metrics"]) == {"1", "2", "4"}
    assert all(
        "evaluation" in payload["candidate_metrics"][dimension]
        and "resource_profile" in payload["candidate_metrics"][dimension]
        for dimension in ("1", "2", "4")
    )


def test_exported_student_is_small_bounded_exact_and_parent_bound() -> None:
    payload = _report()
    artifact = LowDimensionalRecurrenceArtifact.from_dict(
        json.loads(DEFAULT_STUDENT.read_text(encoding="utf-8"))
    )
    assert artifact.state_dimension == payload["selection"]["selected_dimension"]
    assert artifact.resource_profile.stored_trainable_scalars <= 95
    assert artifact.resource_profile.persistent_state_scalars <= 4
    assert artifact.teacher_checkpoint_sha256 == payload["parent_provenance"][
        "teacher_checkpoint_sha256"
    ]
    assert artifact.teacher_state_sha256 == payload["parent_provenance"][
        "teacher_state_sha256"
    ]
    replay = payload["student_artifact"]["runtime_replay"]
    assert payload["student_artifact"]["torch_export_maximum_error"] < 1.0e-12
    assert replay["maximum_batch_runtime_error"] < 1.0e-12
    assert replay["leakage_exact_zero"] and replay["health_exact_zero"]
    assert tuple(artifact.residual_bounds) == RESIDUAL_BOUNDS


def test_source_data_keeps_every_epoch_split_prediction_and_candidate() -> None:
    payload = _report()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"]
    assert {row["row_type"] for row in rows} == {
        "training_epoch",
        "validation_checkpoint",
        "selected_student_prediction",
        "candidate_summary",
    }
    assert sum(row["row_type"] == "training_epoch" for row in rows) == 9 * 900
    expected_predictions = 3 * payload["config"]["trajectories_per_split"] * (
        payload["config"]["half_cycles"] + 1
    )
    assert sum(row["row_type"] == "selected_student_prediction" for row in rows) == expected_predictions
    assert sum(row["row_type"] == "candidate_summary" for row in rows) == 3
    assert {row["split"] for row in rows} >= {
        "training",
        "validation",
        "evaluation",
        "evaluation_report_only",
    }


def test_online_artifact_module_contains_no_torch_physics_or_teacher_import() -> None:
    source = Path("cnn_fpga/control/low_dimensional_recurrence.py").read_text(
        encoding="utf-8"
    )
    assert "import torch" not in source
    assert "from physics" not in source
    assert "bounded_residual" not in source
    contract = _report()["student_artifact"]["runtime_replay"]["online_contract"]
    assert contract["torch_runtime_dependency"] is False
    assert contract["physics_runtime_dependency"] is False
    assert contract["teacher_runtime_dependency"] is False


def test_pilot_executes_end_to_end_but_fails_production_completeness(tmp_path) -> None:
    pytest.importorskip("torch")
    config = LowDimensionalDistillationConfig(
        half_cycles=8,
        trajectories_per_split=8,
        candidate_dimensions=(1,),
        restart_seeds=(44311,),
        training_epochs=5,
        validation_interval=1,
        minimum_zero_mse_reduction_fraction=0.10,
        device="cpu",
    )
    result = run_low_dimensional_student_distillation(
        config,
        artifact_path=tmp_path / "pilot.json",
        checkpoint_path=tmp_path / "pilot.pt",
        student_path=tmp_path / "student.json",
        source_data_path=tmp_path / "pilot.csv",
        production=False,
        resume=False,
    )
    assert result["status"] == "FAIL"
    assert not result["gates"][
        "one_two_four_dimensions_and_three_restarts_are_fully_retained"
    ]
    assert result["gates"]["json_artifact_roundtrip_matches_torch_candidate"]
    assert result["gates"]["pure_numpy_runtime_replay_is_exact_and_fail_closed"]
    assert "physical gain retention" in result["claim_boundary"]["forbidden"]
