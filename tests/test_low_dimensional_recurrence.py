from __future__ import annotations

import copy

import numpy as np
import pytest

from cnn_fpga.control.low_dimensional_recurrence import (
    CONTROL_PARAMETER_NAMES,
    RESIDUAL_BOUNDS,
    LowDimensionalObservation,
    LowDimensionalRecurrenceArtifact,
    LowDimensionalRecurrenceStudent,
    LowDimensionalResourceProfile,
    online_contract,
)


def _artifact(dimension: int = 2) -> LowDimensionalRecurrenceArtifact:
    return LowDimensionalRecurrenceArtifact.create(
        initial_state=np.linspace(-0.1, 0.1, dimension),
        outcome_decays=np.asarray([[0.5] * dimension, [0.75] * dimension]),
        outcome_saturations=np.asarray([[-0.4] * dimension, [0.6] * dimension]),
        output_weights=np.arange(15 * dimension, dtype=np.float64).reshape(15, dimension)
        / 100.0,
        output_bias=np.linspace(-0.05, 0.05, 15),
        teacher_checkpoint_sha256="0" * 64,
        teacher_state_sha256="1" * 64,
        teacher_analysis_sha256="2" * 64,
        training_dataset_sha256="3" * 64,
        validation_dataset_sha256="4" * 64,
        selected_dimension=dimension,
        selected_restart=1,
        validation_mse=1.0e-5,
    )


def test_resource_profile_is_exact_for_each_candidate_dimension() -> None:
    assert LowDimensionalResourceProfile.exact(1).stored_trainable_scalars == 35
    assert LowDimensionalResourceProfile.exact(2).stored_trainable_scalars == 55
    assert LowDimensionalResourceProfile.exact(4).stored_trainable_scalars == 95
    assert LowDimensionalResourceProfile.exact(4).multiply_adds_per_healthy_step == 87
    with pytest.raises(ValueError, match="does not match"):
        LowDimensionalResourceProfile(
            state_dimension=2,
            stored_trainable_scalars=54,
            persistent_state_scalars=2,
            multiply_adds_per_healthy_step=51,
            stored_scalar_bytes_float32=220,
        )


def test_artifact_roundtrip_is_hash_bound_and_teacher_free() -> None:
    artifact = _artifact()
    payload = artifact.to_dict()
    restored = LowDimensionalRecurrenceArtifact.from_dict(payload)
    assert restored == artifact
    assert payload["offline_teacher_object_embedded"] is False
    assert payload["torch_runtime_dependency"] is False
    tampered = copy.deepcopy(payload)
    tampered["output_bias"][0] += 0.1
    with pytest.raises(ValueError, match="hash mismatch"):
        LowDimensionalRecurrenceArtifact.from_dict(tampered)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    (
        ({"selected_dimension": 0}, "at least"),
        ({"outcome_decays": ((0.0, 0.5), (0.5, 0.5))}, "strictly"),
        ({"teacher_state_sha256": "bad"}, "SHA-256"),
        ({"validation_mse": -1.0}, "nonnegative"),
    ),
)
def test_artifact_creation_fails_closed(kwargs: dict[str, object], match: str) -> None:
    arguments = {
        "initial_state": (0.0, 0.0),
        "outcome_decays": ((0.5, 0.5), (0.7, 0.7)),
        "outcome_saturations": ((-0.5, -0.5), (0.5, 0.5)),
        "output_weights": np.zeros((15, 2)),
        "output_bias": np.zeros(15),
        "teacher_checkpoint_sha256": "0" * 64,
        "teacher_state_sha256": "1" * 64,
        "teacher_analysis_sha256": "2" * 64,
        "training_dataset_sha256": "3" * 64,
        "validation_dataset_sha256": "4" * 64,
        "selected_dimension": 2,
        "selected_restart": 0,
        "validation_mse": 1.0e-5,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        LowDimensionalRecurrenceArtifact.create(**arguments)  # type: ignore[arg-type]


def test_healthy_recurrence_and_bounded_head_match_manual_equation() -> None:
    artifact = _artifact()
    student = LowDimensionalRecurrenceStudent(artifact)
    initial = np.asarray(artifact.initial_state)
    expected_state = 0.5 * initial + 0.5 * np.asarray(artifact.outcome_saturations[0])
    decision = student.step(LowDimensionalObservation(0, "g"))
    np.testing.assert_allclose(decision.state, expected_state, rtol=0.0, atol=1.0e-15)
    expected_raw = np.asarray(artifact.output_weights) @ expected_state + np.asarray(
        artifact.output_bias
    )
    expected_residual = np.asarray(RESIDUAL_BOUNDS) * np.tanh(expected_raw)
    np.testing.assert_allclose(decision.raw_control_residual, expected_raw, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        decision.physical_control_residual, expected_residual, rtol=0.0, atol=1e-15
    )
    assert np.all(np.abs(decision.physical_control_residual) <= np.asarray(RESIDUAL_BOUNDS))


@pytest.mark.parametrize(
    "observation",
    (
        LowDimensionalObservation(0, "leakage"),
        LowDimensionalObservation(0, "g", valid=False),
        LowDimensionalObservation(0, "e", crc_ok=False),
        LowDimensionalObservation(0, "g", parameter_fresh=False),
        LowDimensionalObservation(0, "e", deadline_ok=False),
    ),
)
def test_leakage_and_each_health_failure_reset_and_return_exact_zero(observation) -> None:
    artifact = _artifact()
    student = LowDimensionalRecurrenceStudent(artifact)
    decision = student.step(observation)
    assert decision.used_safe_baseline
    assert decision.raw_control_residual == (0.0,) * 15
    assert decision.physical_control_residual == (0.0,) * 15
    assert decision.state == artifact.initial_state


def test_sequence_must_be_contiguous_and_reset_is_deterministic() -> None:
    student = LowDimensionalRecurrenceStudent(_artifact())
    with pytest.raises(ValueError, match="contiguous"):
        student.step(LowDimensionalObservation(1, "g"))
    first = student.step(LowDimensionalObservation(0, "e"))
    student.reset()
    second = student.step(LowDimensionalObservation(0, "e"))
    assert first == second


def test_observation_and_online_contract_are_strict_and_dependency_free() -> None:
    with pytest.raises(ValueError, match="g, e, or leakage"):
        LowDimensionalObservation(0, "f")
    with pytest.raises(TypeError, match="strict boolean"):
        LowDimensionalObservation(0, "g", valid=1)  # type: ignore[arg-type]
    contract = online_contract()
    assert contract["control_parameter_names"] == CONTROL_PARAMETER_NAMES
    assert contract["torch_runtime_dependency"] is False
    assert contract["physics_runtime_dependency"] is False
    assert contract["teacher_runtime_dependency"] is False
    assert contract["target_latency_cycles"] is None
