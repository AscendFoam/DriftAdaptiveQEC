from __future__ import annotations

from dataclasses import replace
import math

import pytest

from cnn_fpga.benchmark.adaptive_drift_alignment import (
    AdaptiveAlignmentConfig,
    run_adaptive_drift_alignment,
)
from physics.constants import LATTICE_CONST


@pytest.fixture(scope="module")
def default_alignment():
    """The acceptance run is intentionally shared: it is a 72k-sample benchmark."""

    return run_adaptive_drift_alignment()


def test_default_alignment_preserves_formal_static_counterevidence(default_alignment):
    result = default_alignment

    assert result.primary_method == "ekf_baseline"
    assert result.causal_delay_windows == 1
    assert result.paired_samples == 24 * 3_000
    assert result.static_oracle_gap_exploitable
    assert not result.primary_alignment_gate_passed
    assert result.oracle_error_rate < result.window_error_rate < result.static_error_rate
    assert result.oracle_error_rate < result.static_error_rate < result.ekf_error_rate
    assert result.static_error_rate < result.standard_error_rate
    assert result.standard_gap.static_minus_dual.ci_low > 0.0
    assert result.ekf_gap.static_minus_oracle.ci_low > 0.0
    assert result.ekf_gap.static_minus_dual.ci_low < 0.0 < result.ekf_gap.static_minus_dual.ci_high
    assert result.ekf_gap.point.gap_closed_fraction is not None
    assert result.ekf_gap.point.gap_closed_fraction < 0.0
    assert result.static_prediction.source == "static_training_average_map"
    assert result.static_parameters.training_state_sha256 == result.static_training_state_sha256
    assert result.evidence_scope == "causal_synthetic_existing_baseline_alignment"


def test_failure_accounting_and_trace_provenance_are_exact(default_alignment):
    result = default_alignment
    records = result.records
    sample_count = sum(record.evaluation_samples for record in records)

    assert sample_count == result.paired_samples
    assert sum(record.standard_failures for record in records) / sample_count == pytest.approx(
        result.standard_error_rate,
        abs=0.0,
    )
    assert sum(record.static_failures for record in records) / sample_count == pytest.approx(
        result.static_error_rate,
        abs=0.0,
    )
    assert sum(record.window_failures for record in records) / sample_count == pytest.approx(
        result.window_error_rate,
        abs=0.0,
    )
    assert sum(record.ekf_failures for record in records) / sample_count == pytest.approx(
        result.ekf_error_rate,
        abs=0.0,
    )
    assert sum(record.oracle_failures for record in records) / sample_count == pytest.approx(
        result.oracle_error_rate,
        abs=0.0,
    )
    assert result.ekf_gap.point.static_error_rate == result.static_error_rate
    assert result.standard_gap.point.static_error_rate == result.standard_error_rate
    assert result.standard_gap.point.dual_error_rate == result.static_error_rate
    assert result.ekf_gap.point.dual_error_rate == result.ekf_error_rate
    assert result.ekf_gap.point.oracle_error_rate == result.oracle_error_rate
    assert len(result.trace_sha256) == 64
    assert all(len(record.evaluation_trace_sha256) == 64 for record in records)
    assert len({record.evaluation_trace_sha256 for record in records}) == len(records)


def test_step_change_is_seen_only_by_the_following_window(default_alignment):
    result = default_alignment
    step = result.config.change_step
    at_change = result.records[step]
    after_one_observation = result.records[step + 1]

    assert all(record.regime == "before" for record in result.records[:step])
    assert all(record.regime == "after" for record in result.records[step:])
    assert at_change.window_prediction_used.source == "window_variance"
    assert at_change.ekf_prediction_used.source == "ekf_baseline"
    # The first after-regime decision still uses the last before-regime estimate.
    assert abs(at_change.window_prediction_used.mu_q) < 0.05 * LATTICE_CONST
    assert abs(at_change.window_prediction_used.mu_p) < 0.05 * LATTICE_CONST
    # Only window step+1 may use the histogram collected at the change window.
    assert after_one_observation.window_prediction_used.mu_q > 0.10 * LATTICE_CONST
    assert after_one_observation.window_prediction_used.mu_p < -0.10 * LATTICE_CONST
    assert math.isclose(
        at_change.truth_mu_q,
        result.config.after_mu_q_fraction * LATTICE_CONST,
    )


def test_fixed_seed_has_exact_prefix_causality_and_reproducibility():
    common = AdaptiveAlignmentConfig(
        windows=8,
        change_step=4,
        calibration_windows=2,
        observation_samples_per_window=400,
        evaluation_samples_per_window=400,
        histogram_bins=24,
        bootstrap_replicates=0,
        seed=1234,
    )
    short_first = run_adaptive_drift_alignment(common)
    short_repeat = run_adaptive_drift_alignment(common)
    long = run_adaptive_drift_alignment(replace(common, windows=12))

    assert short_first == short_repeat
    assert short_first.static_prediction == long.static_prediction
    assert short_first.records == long.records[: common.windows]
    assert [record.evaluation_trace_sha256 for record in short_first.records] == [
        record.evaluation_trace_sha256 for record in long.records[: common.windows]
    ]
    assert short_first.trace_sha256 != long.trace_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("windows", 3, "windows must be at least 4"),
        ("change_step", 0, "change_step must lie"),
        ("calibration_windows", 1, "calibration_windows must be at least 2"),
        ("observation_samples_per_window", 199, "must be at least 200"),
        ("evaluation_samples_per_window", 199, "must be at least 200"),
        ("histogram_bins", 15, "must be at least 16"),
        ("histogram_bins", 513, "must not exceed 512"),
        ("seed", 2**64 - 2, "child seeds remain valid"),
        ("sigma_ratio_p", 0.0, "sigma_ratio_p must lie"),
        ("after_mu_q_fraction", 0.5, "must be finite and lie"),
        ("after_sigma_fraction", 0.46, "must lie in"),
        ("after_theta_deg", 90.0, "must lie in"),
    ],
)
def test_configuration_rejects_invalid_or_ambiguous_inputs(field, value, message):
    with pytest.raises((TypeError, ValueError), match=message):
        AdaptiveAlignmentConfig(**{field: value})


def test_configuration_rejects_accidentally_unbounded_workload():
    with pytest.raises(ValueError, match="configured workload"):
        AdaptiveAlignmentConfig(
            windows=10_000,
            observation_samples_per_window=1_000,
            evaluation_samples_per_window=200,
        )


def test_runner_rejects_non_config_objects():
    with pytest.raises(TypeError, match="config must be AdaptiveAlignmentConfig"):
        run_adaptive_drift_alignment(object())  # type: ignore[arg-type]
