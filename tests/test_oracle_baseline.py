from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.oracle_baseline import (
    FULL_STATE_ORACLE_ID,
    LEAKAGE_FLAG_ORACLE_ID,
    ORACLE_BASELINE_DESCRIPTOR,
    OracleHiddenContext,
    build_oracle_validation,
    oracle_upper_reference_decision,
    validate_oracle_major_comparisons,
)
from cnn_fpga.benchmark.standard_binning_baseline import major_comparison_registry
from physics.drift_processes import DriftState
from physics.oracle_map import oracle_map_2d, oracle_map_trajectory
from physics.syndrome_stream import SyndromeStreamConfig, generate_syndrome_stream


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_1_3_oracle_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_1_3_oracle_source_data.csv"


def _state(**overrides) -> DriftState:
    values = {
        "mu_q": 0.2,
        "mu_p": -0.1,
        "sigma_q": 0.4,
        "sigma_p": 0.3,
        "rho": 0.2,
        "regime": "quiet",
        "source": "oracle-baseline-test",
    }
    values.update(overrides)
    return DriftState(**values)


def _one_step(*, leakage_probability: float):
    stream = generate_syndrome_stream(
        (_state(burst_active=leakage_probability > 0.0, regime="burst" if leakage_probability > 0 else "quiet"),),
        config=SyndromeStreamConfig(
            seed=313,
            measurement_sigma=(0.0, 0.0),
            base_leakage_probability=leakage_probability,
            burst_leakage_bonus=0.0,
            loss_leakage_scale=0.0,
            higher_leakage_fraction=0.0,
        ),
    )
    return stream.steps[0]


def test_descriptor_is_explicitly_hidden_truth_and_nondeployable():
    descriptor = ORACLE_BASELINE_DESCRIPTOR

    assert descriptor.baseline_id == FULL_STATE_ORACLE_ID
    assert not descriptor.deployable
    assert len(descriptor.hidden_inputs) == 4
    assert "oracle_delayed" in descriptor.forbidden_deployable_aliases
    assert "upper_reference" in descriptor.comparison_role


def test_context_can_only_be_constructed_from_truth_step():
    step = _one_step(leakage_probability=0.0)
    context = OracleHiddenContext.from_truth_step(step.truth)

    assert context.hidden_regime == step.truth.hidden_regime
    assert context.leakage_kind == "none"
    with pytest.raises(TypeError, match="SyndromeTruthStep"):
        OracleHiddenContext.from_truth_step(step.observed.as_deployable_dict())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must match"):
        OracleHiddenContext(
            drift_state=step.truth.drift_state,
            hidden_regime="wrong",
            leakage_kind="none",
        )
    with pytest.raises(ValueError, match="leakage_kind"):
        OracleHiddenContext(
            drift_state=step.truth.drift_state,
            hidden_regime=step.truth.hidden_regime,
            leakage_kind="unknown",  # type: ignore[arg-type]
        )


def test_normal_context_matches_full_state_oracle_and_carries_regime_provenance():
    state = _state(regime="correlated", burst_active=True)
    context = OracleHiddenContext(
        drift_state=state,
        hidden_regime="correlated",
        leakage_kind="none",
    )
    syndrome = np.array([0.1, -0.2])

    decision = oracle_upper_reference_decision(syndrome, context)
    reference = oracle_map_2d(syndrome, state)

    assert decision.reference_id == FULL_STATE_ORACLE_ID
    assert not decision.erasure_flag
    assert decision.logical_class == reference.logical_class
    assert decision.logical_action == reference.logical_action
    assert decision.map_result is not None
    assert np.array_equal(decision.map_result.posterior, reference.posterior)
    assert np.array_equal(decision.map_result.parity, reference.parity)
    assert decision.map_result.state_regime == "correlated"
    assert decision.map_result.burst_active
    assert not decision.deployable


def test_leakage_context_flags_erasure_without_fabricated_pauli_action():
    step = _one_step(leakage_probability=1.0)
    context = OracleHiddenContext.from_truth_step(step.truth)

    decision = oracle_upper_reference_decision(
        step.observed.residual_syndrome,
        context,
    )

    assert context.leakage_kind == "f"
    assert decision.reference_id == LEAKAGE_FLAG_ORACLE_ID
    assert decision.erasure_flag
    assert decision.logical_class is None
    assert decision.logical_action == "FLAG_LEAKAGE"
    assert decision.map_result is None


@pytest.mark.parametrize(
    "syndrome",
    [0.0, [0.0], [0.0, 0.0, 0.0], [np.nan, 0.0]],
)
def test_oracle_upper_reference_rejects_invalid_syndrome(syndrome):
    context = OracleHiddenContext(
        drift_state=_state(),
        hidden_regime="quiet",
        leakage_kind="none",
    )
    with pytest.raises(ValueError, match="exactly two finite"):
        oracle_upper_reference_decision(syndrome, context)


def test_full_state_oracle_validator_respects_task_specific_reference_anchors():
    gates = validate_oracle_major_comparisons()
    full_state = [
        entry
        for entry in major_comparison_registry()
        if entry.reference_anchor_method_id == FULL_STATE_ORACLE_ID
    ]

    assert len(gates) == len(full_state)
    assert all(entry.method_ids.count(FULL_STATE_ORACLE_ID) == 1 for entry in full_state)
    memory = next(
        entry
        for entry in major_comparison_registry()
        if entry.comparison_id == "t3_2_1_memory_bayesian_episode_comparison"
    )
    assert memory.reference_anchor_method_id == "full_episode_logical_truth_reference"
    assert FULL_STATE_ORACLE_ID not in memory.method_ids


def test_production_gate_names_the_declared_full_state_oracle_scope() -> None:
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    gates = payload["gate_summary"]["gates"]
    assert "full_state_oracle_present_in_declared_schemas" in gates
    assert "oracle_present_in_all_required_schemas" not in gates


def test_oracle_map_trajectory_provenance_tracks_regime_and_burst_per_step():
    states = (
        _state(step=0, regime="quiet", burst_active=False),
        _state(step=1, regime="burst", burst_active=True),
    )
    result = oracle_map_trajectory(np.zeros((2, 2)), states)

    assert result.state_regimes == ("quiet", "burst")
    assert np.array_equal(result.burst_active, [False, True])
    assert result.state_sources == ("oracle-baseline-test", "oracle-baseline-test")


def _implementation_hash() -> str:
    paths = (
        ROOT / "cnn_fpga" / "benchmark" / "oracle_baseline.py",
        ROOT / "physics" / "oracle_map.py",
        ROOT / "cnn_fpga" / "benchmark" / "adaptive_drift_alignment.py",
        ROOT / "cnn_fpga" / "benchmark" / "standard_binning_baseline.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_production_artifact_is_source_bound_and_preserves_two_oracle_lanes():
    payload = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_hash()
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] == len(payload["gate_summary"]["gates"])
    regime = payload["regime_matrix"]
    assert regime["samples"] == 320_000
    assert regime["static_minus_oracle"]["ci_low"] > 0.0
    assert regime["oracle_error_rate"] < regime["static_error_rate"]
    leakage = payload["protocol_leakage_flag"]
    assert leakage["cycles"] == 8_000
    assert leakage["leakage_cycles"] > 0
    assert leakage["flag_sensitivity"] == 1.0
    assert leakage["flag_specificity"] == 1.0
    assert leakage["optimistic_perfect_erasure_lower_bound"] < leakage[
        "conservative_leakage_as_failure_rate"
    ]


def test_source_data_has_all_regime_seed_rows_and_leakage_cost_rows():
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    regime_rows = [row for row in rows if row["lane"] == "regime_matrix"]
    leakage_rows = [row for row in rows if row["lane"] == "protocol_leakage_flag"]

    assert len(regime_rows) == 16
    assert len(leakage_rows) == 4
    assert len({row["trace_sha256"] for row in rows}) == 20
    assert {row["scenario"] for row in regime_rows} == {
        "quiet",
        "shifted",
        "correlated",
        "burst_mixture",
    }
    assert all(float(row["static_minus_oracle_ci_low"]) > 0.0 for row in regime_rows)
    assert all(row["leakage_flag_sensitivity"] == "1.0" for row in leakage_rows)


def test_validation_builder_reproduces_persisted_artifacts():
    payload, rows = build_oracle_validation()
    persisted = json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert payload == persisted
    assert len(rows) == len(csv_rows)
    assert [row["trace_sha256"] for row in rows] == [
        row["trace_sha256"] for row in csv_rows
    ]


def test_observed_record_contains_no_oracle_truth_fields():
    observed = _one_step(leakage_probability=1.0).observed.as_deployable_dict()

    assert "drift_state" not in observed
    assert "hidden_regime" not in observed
    assert "leakage_kind" not in observed
    assert "true_logical_bits" not in observed
    assert "target_params" not in observed
