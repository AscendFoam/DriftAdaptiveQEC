from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.exponential_recurrence_baseline import (
    EventRecurrenceValidationConfig,
    _event_lane,
    implementation_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "t3_2_10_exponential_recurrence_validation.json"
CHECKPOINT = ROOT / "docs" / "t3_2_10_exponential_recurrence_checkpoints.pt"
SOURCE_DATA = ROOT / "docs" / "t3_2_10_exponential_recurrence_source_data.csv"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"training_seeds": (1, 2)},
        {"evaluation_seeds": (3, 4, 5, 6, 7)},
        {"training_cycles": 255},
        {"decay_g_grid": ()},
        {"decay_e_grid": (0.0,)},
        {"recovery_exit": 0.5, "recovery_enter_grid": (0.4,)},
        {"leakage_exit": 0.5, "leakage_enter_grid": (0.4,)},
    ],
)
def test_event_validation_config_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        EventRecurrenceValidationConfig(**kwargs)  # type: ignore[arg-type]


def test_event_smoke_uses_training_only_grid_same_traces_and_fixed_point() -> None:
    settings = EventRecurrenceValidationConfig(
        training_cycles=256,
        evaluation_cycles=512,
        evaluation_seeds=(20261121, 20261122, 20261123, 20261124, 20261125, 20261126),
        decay_g_grid=(0.45,),
        decay_e_grid=(0.55,),
        decay_leakage_grid=(0.15,),
        recovery_enter_grid=(0.45,),
        leakage_enter_grid=(0.35,),
    )
    payload, rows = _event_lane(settings)
    assert payload["training"]["evaluation_truth_used"] is False
    assert payload["training"]["recurrence_grid_size"] == 1
    assert payload["training"]["training_traces"] == 12
    assert payload["evaluation"]["traces"] == 24
    assert payload["evaluation"]["cycles"] == 12_288
    assert payload["evaluation"]["fixed_point"]["maximum_state_error"] < 2.0e-5
    assert len(rows) == 24
    assert len({row["trace_sha256"] for row in rows}) == 24
    assert all(row["recurrence_float_bank_conflicts"] == 0 for row in rows)


def _payload() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_production_artifact_is_source_bound_non_demo_and_passes() -> None:
    payload = _payload()
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] >= 19
    assert payload["artifacts"]["source_data_rows"] > 1500
    assert CHECKPOINT.exists()


def test_physical_lane_is_exact_causal_and_lookup_bounded() -> None:
    payload = _payload()
    contract = payload["recurrence_contract"]
    comparisons = payload["physical_fidelity_lane"]["comparisons"]
    assert contract["formula"] == "pi[t+1] = a[m] * pi[t] + (1-a[m]) * pi_inf[m]"
    assert contract["trainable_scalars"] == 75
    assert contract["stored_scalars_including_leakage"] == 105
    assert comparisons["primary_recurrence_minus_standard_fidelity"] > 0.001
    assert comparisons["primary_lookup_minus_recurrence_fidelity"] >= -2.0e-10
    assert abs(comparisons["primary_fixed_minus_float_fidelity"]) < 0.001
    assert payload["optimization"]["selected_refinement_tail_gain_last_25"] < 2.0e-4


def test_event_cost_is_not_relabelled_as_physical_or_logical_error() -> None:
    payload = _payload()
    event = payload["event_control_lane"]
    assert event["metric_domain"] == "abstract_event_control_cost_not_physical_fidelity_or_LER"
    assert event["evaluation"]["traces"] == 32
    assert event["evaluation"]["cycles"] == 384_000
    assert event["evaluation"]["fixed_point"]["mode_parity"] > 0.99
    assert "logical-error-rate gain" in payload["claim_boundary"]["forbidden"]


def test_source_data_contains_both_lanes_without_mixing_metric_domain() -> None:
    with SOURCE_DATA.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) > 1500
    types = {row["row_type"] for row in rows}
    assert types == {
        "physical_optimization",
        "physical_terminal_branch",
        "event_training_grid",
        "event_evaluation_trace",
    }
    for row in rows:
        if row["row_type"].startswith("physical"):
            assert row["metric_domain"] == "exact_physical_fidelity"
        else:
            assert row["metric_domain"] == "abstract_event_control_cost"

