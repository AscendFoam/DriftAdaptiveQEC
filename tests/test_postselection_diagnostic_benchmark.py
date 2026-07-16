from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.postselection_diagnostic import (
    POSTSELECTION_DESCRIPTOR,
    PostselectionValidationConfig,
    _implementation_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t3_2_4_postselection_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t3_2_4_postselection_source_data.csv"


def _payload() -> dict[str, object]:
    return json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))


def test_descriptor_excludes_online_primary_and_truth_score() -> None:
    assert POSTSELECTION_DESCRIPTOR.online_decoder is False
    assert POSTSELECTION_DESCRIPTOR.primary_metric_eligible is False
    assert POSTSELECTION_DESCRIPTOR.hidden_truth_score_inputs == ()
    assert POSTSELECTION_DESCRIPTOR.truth_only_evaluator_fields == ("logical_failure",)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"training_seeds": (1, 2)}, "at least three"),
        ({"evaluation_seeds": (1, 2, 3, 4, 5)}, "at least six"),
        ({"training_seeds": (1, 2, 3), "evaluation_seeds": (3, 4, 5, 6, 7, 8)}, "disjoint"),
        ({"target_survivals": (0.9, 0.8, 0.7, 0.6)}, "at least five"),
        ({"target_survivals": (0.9, 0.7, 0.8, 0.6, 0.5)}, "decreasing"),
        ({"rejection_penalties": (0.0, 0.25, 0.5, 0.75)}, "span"),
        ({"primary_diagnostic_survival": 0.85}, "registered"),
    ],
)
def test_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        PostselectionValidationConfig(**kwargs)  # type: ignore[arg-type]


def test_production_artifact_is_source_bound_and_non_demo_scale() -> None:
    payload = _payload()
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_sha256()
    assert payload["aggregate"]["evaluation_samples"] == 1_572_864
    assert payload["aggregate"]["source_data_rows"] == 256
    assert payload["training_calibration"]["training_samples"] == 294_912
    assert payload["gate_summary"]["passed"] == 14
    assert payload["gate_summary"]["failed"] == 0


def test_training_thresholds_are_frozen_monotone_and_truth_free() -> None:
    calibration = _payload()["training_calibration"]
    assert calibration["evaluation_truth_used"] is False
    assert len(calibration["training_trace_sha256"]) == 64
    thresholds = calibration["thresholds"]
    assert [row["target_survival"] for row in thresholds] == [
        0.995,
        0.99,
        0.98,
        0.95,
        0.9,
        0.8,
        0.7,
        0.5,
    ]
    values = [row["score_threshold"] for row in thresholds]
    assert all(values[index] > values[index + 1] for index in range(len(values) - 1))


def test_primary_diagnostic_is_informative_but_not_a_main_gain() -> None:
    payload = _payload()
    primary = [
        row
        for row in payload["scenario_survival_summaries"]
        if row["target_survival"] == 0.9
    ]
    assert len(primary) == 4
    for row in primary:
        assert row["score_auc_seed_cluster_ci"]["ci_low"] > 0.5
        assert row["raw_minus_conditional_seed_cluster_ci"]["ci_low"] > 0.0
        assert row["truth_upper_conditional_error_rate"] <= row["conditional_error_rate"]
        assert row["mean_total_cost_by_rejection_penalty"]["1.00"] >= row["raw_error_rate"]
    assert "online correction gain" in payload["claim_boundary"]["forbidden"]


def test_source_data_grid_recomputes_cost_and_keeps_truth_upper_separate() -> None:
    payload = _payload()
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 256
    assert len({row["trace_sha256"] for row in rows}) == 32
    assert {float(row["target_survival"]) for row in rows} == {
        0.995,
        0.99,
        0.98,
        0.95,
        0.9,
        0.8,
        0.7,
        0.5,
    }
    truth_upper = [float(row["truth_upper_conditional_error_rate"]) for row in rows]
    assert any(value > 0.0 for value in truth_upper)
    assert any(value == 0.0 for value in truth_upper)
    for row in rows:
        survival = float(row["survival_fraction"])
        rejection = float(row["rejection_fraction"])
        accepted = float(row["accepted_failures_per_input"])
        assert survival + rejection == pytest.approx(1.0, abs=1e-12)
        assert float(row["truth_upper_conditional_error_rate"]) <= float(
            row["conditional_error_rate"]
        ) + 1e-15
        assert float(row["random_rejection_expected_conditional_error_rate"]) == float(
            row["raw_error_rate"]
        )
        for penalty in (0.0, 0.25, 0.5, 1.0):
            assert float(row[f"total_cost_penalty_{penalty:.2f}"]) == pytest.approx(
                accepted + penalty * rejection, abs=1e-15
            )
