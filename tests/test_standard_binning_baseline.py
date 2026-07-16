from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.adaptive_drift_alignment import (
    AdaptiveAlignmentConfig,
    run_adaptive_drift_alignment,
)
from cnn_fpga.benchmark.standard_binning_baseline import (
    STANDARD_BINNING_DESCRIPTOR,
    STANDARD_BINNING_ID,
    MajorComparisonRegistration,
    build_standard_binning_validation,
    major_comparison_registry,
    standard_binning_logical_class,
    standard_binning_paired_outcomes,
    validate_major_comparison_registry,
)
from physics.constants import LATTICE_CONST


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ARTIFACT = ROOT / "docs" / "t3_1_1_standard_binning_validation.json"


def test_descriptor_freezes_no_tuning_and_no_hidden_truth_contract():
    descriptor = STANDARD_BINNING_DESCRIPTOR

    assert descriptor.baseline_id == STANDARD_BINNING_ID
    assert descriptor.deployable
    assert descriptor.hidden_truth_inputs == ()
    assert descriptor.tunable_parameters == ()
    assert descriptor.observation_inputs == (
        "centered_modular_syndrome_q",
        "centered_modular_syndrome_p",
    )


def test_observed_only_decision_is_even_even_for_any_centered_syndrome():
    half = LATTICE_CONST / 2.0
    syndrome = np.array(
        [
            [-half, -half],
            [-0.49 * half, 0.49 * half],
            [0.0, 0.0],
            [np.nextafter(half, -np.inf), np.nextafter(half, -np.inf)],
        ]
    )

    decision = standard_binning_logical_class(syndrome)

    assert decision.dtype == np.int64
    assert decision.shape == (4,)
    assert np.array_equal(decision, np.zeros(4, dtype=np.int64))


@pytest.mark.parametrize(
    ("syndrome", "message"),
    [
        (0.0, "shape"),
        (np.zeros(3), "shape"),
        (np.zeros((2, 3)), "shape"),
        (np.array([[np.nan, 0.0]]), "finite"),
        (np.array([[LATTICE_CONST / 2.0, 0.0]]), "half-open"),
        (np.array([[-LATTICE_CONST / 2.0 - 1.0e-12, 0.0]]), "half-open"),
    ],
)
def test_observed_only_decision_rejects_ambiguous_inputs(syndrome, message):
    with pytest.raises(ValueError, match=message):
        standard_binning_logical_class(syndrome)


@pytest.mark.parametrize("lattice", [0.0, -1.0, np.nan, np.inf])
def test_observed_only_decision_rejects_invalid_lattice(lattice):
    with pytest.raises(ValueError, match="lattice"):
        standard_binning_logical_class(np.zeros((1, 2)), lattice=lattice)


def test_paired_evaluator_keeps_decision_separate_from_hidden_cell_truth():
    lam = LATTICE_CONST
    displacements = np.array(
        [
            [0.10 * lam, -0.20 * lam],
            [1.10 * lam, -0.20 * lam],
            [2.10 * lam, -0.20 * lam],
            [0.10 * lam, -1.20 * lam],
        ]
    )

    decision, truth, failure = standard_binning_paired_outcomes(displacements)

    assert np.array_equal(decision, [0, 0, 0, 0])
    assert np.array_equal(truth, [0, 2, 0, 1])
    assert np.array_equal(failure, [False, True, False, True])
    # Rows 0/1/2 have the same centered syndrome, yet the decoder cannot see
    # their different hidden cell parities.
    assert decision[0] == decision[1] == decision[2]


def test_half_open_boundary_semantics_are_inherited_exactly():
    half = LATTICE_CONST / 2.0
    decision, truth, failure = standard_binning_paired_outcomes(
        np.array([[-half, 0.0], [half, 0.0]])
    )

    assert np.array_equal(decision, [0, 0])
    assert np.array_equal(truth, [0, 2])
    assert np.array_equal(failure, [False, True])


def test_registry_has_one_active_and_one_future_required_schema():
    registry = major_comparison_registry()
    gates = validate_major_comparison_registry(registry)

    required = [entry for entry in registry if entry.standard_binning_policy == "required"]
    assert {entry.lifecycle for entry in required} == {"active", "future_contract"}
    assert all(entry.method_ids.count(STANDARD_BINNING_ID) == 1 for entry in required)
    decoder_entries = [
        entry for entry in registry if entry.comparison_kind == "decoder_algorithm_comparison"
    ]
    assert all(entry.static_anchor_method_id is not None for entry in decoder_entries)
    assert all(entry.reference_anchor_method_id is not None for entry in decoder_entries)
    assert all(
        entry.method_ids.count(entry.static_anchor_method_id) == 1
        for entry in decoder_entries
    )
    assert all(
        entry.method_ids.count(entry.reference_anchor_method_id) == 1
        for entry in decoder_entries
    )
    assert len(gates) == len(registry)


def test_registry_rejects_required_comparison_omission_and_duplicates():
    active = next(
        entry
        for entry in major_comparison_registry()
        if entry.lifecycle == "active" and entry.standard_binning_policy == "required"
    )
    omitted = replace(
        active,
        method_ids=tuple(item for item in active.method_ids if item != STANDARD_BINNING_ID),
    )
    duplicated = replace(active, method_ids=active.method_ids + (STANDARD_BINNING_ID,))

    with pytest.raises(ValueError, match="exactly once"):
        validate_major_comparison_registry((omitted,))
    with pytest.raises(ValueError, match="exactly once"):
        validate_major_comparison_registry((duplicated,))


def test_registry_rejects_semantic_alias_and_duplicate_comparison_id():
    active = next(
        entry
        for entry in major_comparison_registry()
        if entry.lifecycle == "active" and entry.standard_binning_policy == "required"
    )
    mislabeled = MajorComparisonRegistration(
        comparison_id="bad_sensitivity",
        code_path="bad.py",
        comparison_kind="implementation_sensitivity",
        lifecycle="active",
        method_ids=(STANDARD_BINNING_ID,),
        standard_binning_policy="required",
        rationale="invalid on purpose",
    )
    with pytest.raises(ValueError, match="not a decoder comparison"):
        validate_major_comparison_registry((mislabeled,))
    with pytest.raises(ValueError, match="duplicate comparison_id"):
        validate_major_comparison_registry((active, active))


def test_registry_rejects_missing_or_misplaced_task_specific_static_anchor():
    active = next(
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t3_2_1_memory_bayesian_episode_comparison"
    )
    with pytest.raises(ValueError, match="declare one task-specific static anchor"):
        validate_major_comparison_registry((replace(active, static_anchor_method_id=None),))
    with pytest.raises(ValueError, match="static anchor exactly once"):
        validate_major_comparison_registry(
            (replace(active, static_anchor_method_id="static_training_average_map"),)
        )
    sensitivity = next(
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t3_1_5_topk_periodic_map_sensitivity"
    )
    with pytest.raises(ValueError, match="must not declare static/reference anchors"):
        validate_major_comparison_registry(
            (replace(sensitivity, static_anchor_method_id="full_periodic_gaussian_map"),)
        )
    with pytest.raises(ValueError, match="declare one task-specific reference anchor"):
        validate_major_comparison_registry((replace(active, reference_anchor_method_id=None),))
    with pytest.raises(ValueError, match="reference anchor exactly once"):
        validate_major_comparison_registry(
            (replace(active, reference_anchor_method_id="full_state_model_oracle_map"),)
        )


def test_adaptive_comparison_integrates_standard_on_exact_paired_trace():
    config = AdaptiveAlignmentConfig(
        windows=8,
        change_step=4,
        calibration_windows=2,
        observation_samples_per_window=400,
        evaluation_samples_per_window=400,
        histogram_bins=24,
        bootstrap_replicates=200,
        seed=311,
    )
    result = run_adaptive_drift_alignment(config)
    active = next(
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t1_3_4_adaptive_drift_alignment"
    )

    assert result.comparison_method_ids == active.method_ids
    assert result.comparison_method_ids[0] == STANDARD_BINNING_ID
    assert sum(row.standard_failures for row in result.records) == pytest.approx(
        result.standard_error_rate * result.paired_samples,
        abs=1.0e-12,
    )
    assert result.standard_gap.n_samples == result.paired_samples
    assert result.standard_gap.point.static_error_rate == result.standard_error_rate
    assert result.standard_gap.point.dual_error_rate == result.static_error_rate
    assert all(len(row.evaluation_trace_sha256) == 64 for row in result.records)


def test_legacy_p4_static_linear_is_explicitly_not_relabelled():
    legacy = next(
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "legacy_p4_frozen_software_hil"
    )

    assert legacy.lifecycle == "frozen_legacy"
    assert legacy.standard_binning_policy == "not_applicable"
    assert "static_linear" in legacy.method_ids
    assert STANDARD_BINNING_ID not in legacy.method_ids
    assert "not renamed" in legacy.rationale


def _current_implementation_hash() -> str:
    paths = (
        ROOT / "cnn_fpga" / "benchmark" / "standard_binning_baseline.py",
        ROOT / "cnn_fpga" / "benchmark" / "adaptive_drift_alignment.py",
        ROOT / "physics" / "ideal_gkp_decoder.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_production_artifact_is_source_bound_and_preserves_current_paired_ranking():
    payload = json.loads(PRODUCTION_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _current_implementation_hash()
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] == len(payload["gate_summary"]["gates"])
    adaptive = payload["adaptive_alignment"]
    assert adaptive["method_ids"][0] == STANDARD_BINNING_ID
    assert adaptive["paired_samples"] == 72_000
    assert adaptive["standard_minus_static"]["ci_low"] > 0.0
    assert "static row is better" in adaptive["counterevidence"]


def test_validation_builder_reproduces_production_payload_exactly():
    assert build_standard_binning_validation() == json.loads(
        PRODUCTION_ARTIFACT.read_text(encoding="utf-8")
    )
