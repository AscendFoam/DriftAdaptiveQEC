from dataclasses import replace
import json

import numpy as np
import pytest

from cnn_fpga.control.teacher_student import (
    CONTROL_PARAMETER_NAMES,
    DistilledRecurrenceStudent,
    DistilledStudentArtifact,
    StudentObservation,
    StudentResourceProfile,
    online_contract,
)


def _artifact() -> DistilledStudentArtifact:
    return DistilledStudentArtifact.create(
        initial_state=np.linspace(-0.1, 0.1, 15),
        outcome_saturations=(np.full(15, 0.5), np.full(15, -0.4)),
        outcome_decays=(np.full(15, 0.6), np.full(15, 0.7)),
        raw_clip=4.0,
        teacher_checkpoint_sha256="0" * 64,
        teacher_model_sha256s=("1" * 64, "2" * 64, "3" * 64),
        training_dataset_sha256="4" * 64,
        validation_dataset_sha256="5" * 64,
        selected_restart=1,
    )


def test_artifact_roundtrip_is_hash_bound_and_teacher_object_free() -> None:
    artifact = _artifact()
    payload = artifact.to_dict()
    assert DistilledStudentArtifact.from_dict(payload) == artifact
    assert payload["offline_teacher_object_embedded"] is False
    assert "state_dict" not in json.dumps(payload)
    assert len(artifact.artifact_sha256) == 64


def test_artifact_hash_tamper_is_rejected() -> None:
    artifact = _artifact()
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(artifact, initial_state=(9.0,) + artifact.initial_state[1:])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"outcome_decays": ((0.0,) * 15, (0.5,) * 15)}, "decays"),
        ({"raw_clip": 0.0}, "raw_clip"),
        ({"teacher_model_sha256s": ("1" * 64, "1" * 64, "2" * 64)}, "unique"),
    ],
)
def test_artifact_creation_fails_closed(kwargs: dict[str, object], match: str) -> None:
    base = dict(
        initial_state=(0.0,) * 15,
        outcome_saturations=((0.0,) * 15, (0.0,) * 15),
        outcome_decays=((0.5,) * 15, (0.5,) * 15),
        raw_clip=4.0,
        teacher_checkpoint_sha256="0" * 64,
        teacher_model_sha256s=("1" * 64, "2" * 64, "3" * 64),
        training_dataset_sha256="4" * 64,
        validation_dataset_sha256="5" * 64,
        selected_restart=0,
    )
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DistilledStudentArtifact.create(**base)


def test_g_and_e_recurrence_matches_manual_vector_equation() -> None:
    artifact = _artifact()
    student = DistilledRecurrenceStudent(artifact)
    initial = np.asarray(artifact.initial_state)
    first = student.step(StudentObservation(0, "g"))
    expected_first = 0.6 * initial + 0.4 * 0.5
    np.testing.assert_allclose(first.raw_control_residual, expected_first, rtol=0.0, atol=1.0e-15)
    second = student.step(StudentObservation(1, "e"))
    expected_second = 0.7 * expected_first + 0.3 * -0.4
    np.testing.assert_allclose(second.raw_control_residual, expected_second, rtol=0.0, atol=1.0e-15)
    assert not first.used_safe_baseline and not second.used_safe_baseline


@pytest.mark.parametrize("field", ["valid", "crc_ok", "parameter_fresh", "deadline_ok"])
def test_each_health_failure_forces_exact_zero_safe_baseline(field: str) -> None:
    student = DistilledRecurrenceStudent(_artifact())
    kwargs = {"valid": True, "crc_ok": True, "parameter_fresh": True, "deadline_ok": True}
    kwargs[field] = False
    decision = student.step(StudentObservation(0, "g", **kwargs))
    assert decision.used_safe_baseline
    assert decision.raw_control_residual == (0.0,) * 15
    assert field in decision.reason


def test_leakage_forces_exact_zero_safe_baseline() -> None:
    student = DistilledRecurrenceStudent(_artifact())
    decision = student.step(StudentObservation(0, "leakage"))
    assert decision.used_safe_baseline
    assert decision.raw_control_residual == (0.0,) * 15
    assert decision.reason == "observed_leakage_safe_baseline"


def test_online_student_rejects_teacher_or_mapping_objects() -> None:
    with pytest.raises(TypeError, match="not a teacher"):
        DistilledRecurrenceStudent(object())  # type: ignore[arg-type]
    student = DistilledRecurrenceStudent(_artifact())
    with pytest.raises(TypeError, match="StudentObservation"):
        student.step({"cycle_index": 0, "observed_outcome": "g"})  # type: ignore[arg-type]


def test_cycle_sequence_is_contiguous_and_reset_is_deterministic() -> None:
    student = DistilledRecurrenceStudent(_artifact())
    with pytest.raises(ValueError, match="contiguous"):
        student.step(StudentObservation(1, "g"))
    first = student.step(StudentObservation(0, "e"))
    student.reset()
    second = student.step(StudentObservation(0, "e"))
    assert first == second


def test_resource_profile_has_exact_105_scalar_null_hardware_contract() -> None:
    profile = StudentResourceProfile()
    assert profile.stored_scalars == 105
    assert profile.parameter_bytes_float32 == 420
    assert profile.target_latency_cycles is None
    assert not profile.rtl_measured and not profile.board_measured
    with pytest.raises(ValueError, match="exact recurrence"):
        StudentResourceProfile(stored_scalars=104)


def test_online_contract_has_only_observed_health_inputs_and_no_teacher_runtime() -> None:
    contract = online_contract()
    assert contract["teacher_model_runtime_dependency"] is False
    assert contract["torch_runtime_dependency"] is False
    assert contract["simulator_truth_runtime_dependency"] is False
    assert "hidden" not in " ".join(contract["online_input_fields"])
    assert tuple(CONTROL_PARAMETER_NAMES) == tuple(_artifact().to_dict()["control_parameter_names"])


def test_observation_rejects_unknown_outcome_and_nonboolean_health() -> None:
    with pytest.raises(ValueError, match="observed_outcome"):
        StudentObservation(0, "f")
    with pytest.raises(TypeError, match="valid"):
        StudentObservation(0, "g", valid=1)  # type: ignore[arg-type]

