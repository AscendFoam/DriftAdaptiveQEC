from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.bounded_residual_rnn_teacher import (
    ACTION_CONTRACT_ID,
    BoundedResidualTeacherConfig,
    DEFAULT_ARTIFACT,
    DEFAULT_CHECKPOINT,
    DEFAULT_SOURCE_DATA,
    implementation_sha256,
    load_and_verify_teacher_checkpoint,
    run_bounded_residual_teacher_training,
    validate_production_design,
)


def _artifact() -> dict[str, object]:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"restart_seeds": (1, 1, 2)}, "unique"),
        ({"restart_seeds": (1,), "validation_seeds": (1,)}, "disjoint"),
        ({"action_contract_id": "wrong"}, "canonical"),
        ({"training_epochs": 20, "validation_interval": 21}, "must not exceed"),
        ({"minimum_successful_restart_fraction": 0.0}, "must lie"),
    ),
)
def test_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        BoundedResidualTeacherConfig(**kwargs)  # type: ignore[arg-type]


def test_production_design_rejects_demo_scale() -> None:
    config = BoundedResidualTeacherConfig(
        cutoff=4,
        confirmation_cutoff=4,
        full_cycles=2,
        training_epochs=2,
        training_batch_size=1,
        validation_batch_size=1,
        evaluation_batch_size=1,
        confirmation_batch_size=1,
        validation_interval=1,
        restart_seeds=(1,),
        validation_seeds=(2,),
        evaluation_seeds=(3,),
        confirmation_seeds=(4,),
        bootstrap_repetitions=10,
        device="cpu",
    )
    with pytest.raises(ValueError, match="production cutoff"):
        validate_production_design(config)


def test_committed_artifact_passes_every_gate_and_is_source_bound() -> None:
    payload = _artifact()
    assert payload["status"] == "PASS"
    gates = payload["gates"]
    assert isinstance(gates, dict) and len(gates) >= 20 and all(gates.values())
    assert payload["gate_summary"] == {
        "passed": len(gates),
        "total": len(gates),
        "failed": [],
    }
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["checkpoint"]["sha256"] == hashlib.sha256(
        DEFAULT_CHECKPOINT.read_bytes()
    ).hexdigest()
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()


def test_fresh_restart_provenance_cannot_be_old_checkpoint_renaming() -> None:
    payload = _artifact()
    records = payload["training_restarts"]
    parents = payload["parent_provenance"]["t2_3_7"]
    old_hashes = set(parents["nmf_model_sha256s"])
    new_hashes = {record["checkpoint_sha256"] for record in records}
    initial_hashes = {record["initial_state_sha256"] for record in records}
    assert len(records) >= 3
    assert new_hashes.isdisjoint(old_hashes)
    assert initial_hashes.isdisjoint(old_hashes)
    assert new_hashes.isdisjoint(initial_hashes)
    assert set(payload["config"]["restart_seeds"]).isdisjoint(
        parents["training_seeds"]
    )
    assert all(record["parent_checkpoint_loaded"] is False for record in records)
    assert all(record["parameter_count"] == 72_853 for record in records)


def test_validation_only_selection_and_failure_visibility_are_exact() -> None:
    payload = _artifact()
    records = payload["training_restarts"]
    selected = payload["selected_restart_index"]
    expected = max(
        range(len(records)), key=lambda index: records[index]["best_validation_score"]
    )
    assert selected == expected
    assert payload["failed_restart_indices"] == [
        index for index, record in enumerate(records) if not record["restart_success"]
    ]
    assert payload["training_cap_hit_indices"] == [
        index
        for index, record in enumerate(records)
        if record["best_epoch_reached_training_cap"]
    ]
    assert all(record["optimizer_global_convergence_claimed"] is False for record in records)
    assert "evaluation" not in payload["selection_rule"] or "blind" in payload["selection_rule"]


def test_action_contract_is_fifteen_dimensional_nominal_plus_hard_bounds() -> None:
    payload = _artifact()
    contract = payload["action_contract"]
    assert payload["config"]["action_contract_id"] == ACTION_CONTRACT_ID
    assert contract["output_count"] == 15
    assert len(contract["parameter_names"]) == 15
    assert len(contract["nominal_parameters"]) == 15
    assert contract["residual_bounds"] == [2.0] * 14 + [1.0]
    assert contract["maximum_bound_violation"] == 0.0
    assert contract["zero_residual_matches_nominal_max_error"] == 0.0
    assert contract["absolute_zero_physical_vector_is_inside_safe_residual_box"] is False


def test_source_data_retains_all_epochs_splits_and_action_bounds() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"]
    assert {row["split"] for row in rows} == {
        "training",
        "validation",
        "evaluation",
        "confirmation",
        "contract",
    }
    assert sum(row["row_type"] == "training_epoch" for row in rows) == sum(
        record["epochs_executed"] for record in payload["training_restarts"]
    )
    assert sum(row["row_type"] == "action_bound" for row in rows) == 15
    assert all(row["checkpoint_sha256"] for row in rows)


def test_checkpoint_roundtrip_causality_and_gradient_audit() -> None:
    torch = pytest.importorskip("torch")
    model, payload = load_and_verify_teacher_checkpoint()
    config = BoundedResidualTeacherConfig(**payload["config"])
    histories = torch.tensor(
        ((0, 1, 1, 0), (0, 1, 1, 0)), dtype=torch.int64, device=config.device
    )
    with torch.no_grad():
        output = model(histories, 4)
    assert tuple(output.shape) == (2, 15)
    assert torch.equal(output[0], output[1])
    assert payload["causality"]["full_replay_vs_cached_maximum_error"] < 1.0e-12
    assert payload["gradient_coverage"][
        "all_parameter_tensors_have_finite_nonzero_gradient"
    ]


def test_checkpoint_tamper_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    bad_checkpoint = tmp_path / "teacher.pt"
    data = bytearray(DEFAULT_CHECKPOINT.read_bytes())
    data[len(data) // 2] ^= 0x01
    bad_checkpoint.write_bytes(data)
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_and_verify_teacher_checkpoint(bad_checkpoint, DEFAULT_ARTIFACT)


def test_pilot_training_path_fails_closed_without_three_restarts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = BoundedResidualTeacherConfig(
        cutoff=4,
        confirmation_cutoff=4,
        full_cycles=2,
        training_epochs=1,
        training_batch_size=1,
        validation_batch_size=1,
        evaluation_batch_size=1,
        confirmation_batch_size=1,
        validation_interval=1,
        restart_seeds=(4401,),
        validation_seeds=(4411,),
        evaluation_seeds=(4421,),
        confirmation_seeds=(4431,),
        bootstrap_repetitions=100,
        minimum_validation_gain=0.0,
        minimum_primary_score_gain=0.0,
        minimum_confirmation_score_gain=0.0,
        device="cpu",
    )
    result = run_bounded_residual_teacher_training(
        config,
        artifact_path=tmp_path / "pilot.json",
        checkpoint_path=tmp_path / "pilot.pt",
        source_data_path=tmp_path / "pilot.csv",
        production=False,
        resume=False,
    )
    assert result["status"] == "FAIL"
    assert not result["gates"][
        "three_or_more_fresh_72853_parameter_gru_restarts_are_retained"
    ]
    assert "if status is FAIL" in result["claim_boundary"]["failure_branch"]
