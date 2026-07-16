import csv
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.offline_teacher_student_distillation import (
    TeacherStudentValidationConfig,
    _implementation_sha256,
)
from cnn_fpga.control.teacher_student import (
    DistilledRecurrenceStudent,
    DistilledStudentArtifact,
    StudentObservation,
)


JSON_PATH = Path("docs/t4_1_5_teacher_student_validation.json")
CSV_PATH = Path("docs/t4_1_5_teacher_student_source_data.csv")
STUDENT_PATH = Path("docs/t4_1_5_distilled_student_checkpoint.json")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"trajectories_per_split": 64}, "at least"),
        ({"half_cycles": 18}, "20-half-cycle"),
        ({"training_epochs": 400}, "at least"),
        ({"restart_seeds": (1, 1, 2)}, "unique"),
        ({"validation_data_seed": 20261501}, "disjoint"),
    ],
)
def test_validation_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TeacherStudentValidationConfig(**kwargs)


def test_committed_artifact_passes_all_gates() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["gate_summary"]["passed"] == 21
    assert payload["gate_summary"]["failed"] == 0
    assert all(payload["gate_summary"]["gates"].values())


def test_implementation_and_teacher_hashes_are_current() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    assert payload["implementation_sha256"] == _implementation_sha256()
    teacher = payload["teacher_provenance"]
    assert hashlib.sha256(Path(teacher["checkpoint_path"]).read_bytes()).hexdigest() == teacher[
        "checkpoint_sha256"
    ]
    assert hashlib.sha256(Path(teacher["manifest_path"]).read_bytes()).hexdigest() == teacher[
        "manifest_sha256"
    ]
    assert teacher["runtime_allowed"] is False


def test_student_checkpoint_roundtrip_and_online_replay() -> None:
    payload = json.loads(STUDENT_PATH.read_text(encoding="utf-8"))
    artifact = DistilledStudentArtifact.from_dict(payload)
    student = DistilledRecurrenceStudent(artifact)
    initial = student.initial_decision()
    assert initial.student_artifact_sha256 == artifact.artifact_sha256
    assert not initial.used_safe_baseline
    safe = student.step(StudentObservation(0, "leakage"))
    assert safe.used_safe_baseline and safe.raw_control_residual == (0.0,) * 15


def test_source_data_grid_and_scope_are_complete() -> None:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15_360
    assert {row["split"] for row in rows} == {"training", "validation", "evaluation"}
    assert all(row["scope"] == "offline_teacher_target_distillation_source_data" for row in rows)
    assert len([name for name in rows[0] if name.startswith("offline_teacher_target_")]) == 15
    assert len([name for name in rows[0] if name.startswith("student_prediction_")]) == 15


def test_restart_selection_plateau_and_evaluation_blindness() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    restarts = payload["training_restarts"]
    assert len(restarts) == 3
    assert all(row["all_75_trainables_receive_gradient"] for row in restarts)
    assert all(row["validation_plateau_reached"] for row in restarts)
    assert payload["selected_restart"] == min(restarts, key=lambda row: row["validation_mse"])[
        "restart_index"
    ]
    assert payload["dataset"]["evaluation_used_for_training_or_selection"] is False
    serialized_student = json.dumps(payload["student_artifact"], sort_keys=True)
    assert payload["dataset"]["hashes"]["evaluation"] not in serialized_student


def test_student_imitation_beats_both_baselines_but_physical_claim_stays_closed() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    evaluation = payload["metrics"]["evaluation"]
    assert evaluation["student"]["mse"] < evaluation["latest_only"]["mse"]
    assert evaluation["student"]["mse"] < evaluation["zero_safe"]["mse"]
    assert evaluation["student"]["imitation_gain_retention_vs_zero"] > 0.99
    forbidden = payload["claim_boundary"]["forbidden"]
    assert "physical/lifetime/control gain retention" in forbidden
    assert "RTL" in forbidden and "board/device" in forbidden


def test_online_module_has_no_torch_physics_or_teacher_state_dependency() -> None:
    import cnn_fpga.control.teacher_student as module

    source = inspect.getsource(module)
    assert "import torch" not in source
    assert "physics." not in source
    contract = json.loads(JSON_PATH.read_text(encoding="utf-8"))["online_contract"]
    assert contract["teacher_model_runtime_dependency"] is False
    assert contract["simulator_truth_runtime_dependency"] is False
