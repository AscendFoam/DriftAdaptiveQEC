from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.slow_loop_model_selection import (
    DESCRIPTOR,
    SlowLoopSelectionConfig,
    _implementation_sha256,
)
from cnn_fpga.decoder.regime_hmm import REGIME_CLASSES
from cnn_fpga.decoder.slow_loop_model_selection import MODEL_FAMILIES


ROOT = Path(__file__).resolve().parents[1]
JSON_ARTIFACT = ROOT / "docs" / "t4_1_1_slow_loop_model_selection_validation.json"
CSV_ARTIFACT = ROOT / "docs" / "t4_1_1_slow_loop_model_selection_source_data.csv"
CHECKPOINT = ROOT / "docs" / "t4_1_1_slow_loop_model_selection_checkpoints.pt"


def _payload() -> dict[str, object]:
    return json.loads(JSON_ARTIFACT.read_text(encoding="utf-8"))


def test_descriptor_is_observed_estimator_with_no_architecture_prior() -> None:
    assert DESCRIPTOR.online_hidden_truth_input == ()
    assert DESCRIPTOR.model_family_prior.startswith("none")
    assert DESCRIPTOR.evaluation_used_for_selection is False
    assert DESCRIPTOR.controller is False
    assert DESCRIPTOR.logical_decoder is False
    assert DESCRIPTOR.hardware_measured is False


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"training_seeds": (1, 2)}, "at least 3"),
        ({"validation_seeds": (1, 2)}, "at least 3"),
        ({"evaluation_seeds": (1, 2, 3, 4, 5)}, "at least 6"),
        (
            {
                "training_seeds": (1, 2, 3),
                "validation_seeds": (3, 4, 5),
                "evaluation_seeds": (6, 7, 8, 9, 10, 11),
            },
            "pairwise disjoint",
        ),
        ({"windows_per_trajectory": 127}, "at least 128"),
        ({"budget": object()}, "SlowLoopSelectionBudget"),
        ({"neural_restarts": (1,)}, "at least two"),
        ({"neural_epochs": 10, "neural_patience": 10}, "smaller"),
        ({"temperature_grid": ()}, "unique positive"),
        ({"recurrence_decay_grid": (1.0,)}, r"in \[0,1\)"),
        ({"fsm_confidence_grid": (0.2,)}, "uniform"),
    ],
)
def test_production_config_fails_closed(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        SlowLoopSelectionConfig(**kwargs)  # type: ignore[arg-type]


def test_production_artifact_is_source_bound_non_demo_and_all_gates_pass() -> None:
    payload = _payload()
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == _implementation_sha256()
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] == 13
    assert payload["evaluation"]["trajectories"] == 8
    assert payload["evaluation"]["histories_per_trajectory"] == 505
    assert payload["evaluation"]["histories"] == 4040
    assert payload["evaluation"]["source_data_rows"] == 24_240


def test_all_families_share_task_and_fit_resource_envelope() -> None:
    payload = _payload()
    contract = payload["common_task_contract"]
    profiles = payload["resource_profiles"]
    assert tuple(profiles) == MODEL_FAMILIES
    assert contract["evaluation_used_for_model_or_hyperparameter_selection"] is False
    assert "last 8 windows" in contract["history"]
    assert set(contract["excluded_inputs"]) == {
        "hidden regime",
        "future windows",
        "evaluation labels",
        "logical truth",
    }
    for family in MODEL_FAMILIES:
        assert profiles[family]["macs_per_update_proxy"] <= 4096
        assert profiles[family]["model_and_state_bytes"] <= 4096
        assert profiles[family]["transient_workspace_bytes"] <= 4096
        assert 0.0 < profiles[family]["host_batch_median_us_per_update"] < 5000.0


def test_selection_is_validation_locked_and_neural_search_is_multi_restart() -> None:
    payload = _payload()
    selection = payload["training_and_validation_selection"]
    table = selection["selection_table"]
    eligible = [row for row in table if row["eligible"]]
    expected = min(eligible, key=lambda row: tuple(row["selection_key"]))["family"]
    assert selection["selected_family"] == expected
    assert selection["validation_ranking"][0] == expected
    assert set(selection["validation_ranking"]) == set(MODEL_FAMILIES)
    for family in ("causal_tcn", "small_gru"):
        records = selection["family_details"][family]["selection_scan"]
        assert len(records) == 5
        assert len({row["restart_seed"] for row in records}) == 5
        assert all(row["epochs_executed"] >= 37 for row in records)
        assert all(row["best_epoch"] >= 1 for row in records)


def test_evaluation_reports_all_metrics_without_reselecting_winner() -> None:
    payload = _payload()
    evaluation = payload["evaluation"]
    aggregate = evaluation["aggregate"]
    assert tuple(aggregate) == MODEL_FAMILIES
    assert set(evaluation["evaluation_ranking_diagnostic_not_used_for_selection"]) == set(MODEL_FAMILIES)
    assert 1 <= evaluation["validation_winner_rank_on_evaluation"] <= len(MODEL_FAMILIES)
    assert len(evaluation["per_seed"]) == 8 * len(MODEL_FAMILIES)
    for family in MODEL_FAMILIES:
        metrics = aggregate[family]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert metrics["negative_log_likelihood"] > 0.0
        assert metrics["brier_score"] >= 0.0
        assert set(metrics["class_recall"]) == set(REGIME_CLASSES)


def test_source_data_is_long_form_normalized_and_hash_bound() -> None:
    payload = _payload()
    with CSV_ARTIFACT.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24_240
    assert hashlib.sha256(CSV_ARTIFACT.read_bytes()).hexdigest() == payload["source_data"]["sha256"]
    assert {row["family"] for row in rows} == set(MODEL_FAMILIES)
    assert {row["truth_regime"] for row in rows} == set(REGIME_CLASSES)
    assert len({row["evaluation_seed"] for row in rows}) == 8
    for row in rows:
        probabilities = [float(row[f"p_{state}"]) for state in REGIME_CLASSES]
        assert all(value > 0.0 for value in probabilities)
        assert sum(probabilities) == pytest.approx(1.0, abs=1.0e-9)
        assert row["prediction"] in REGIME_CLASSES


def test_checkpoint_contains_replay_state_for_every_family_and_matches_hash() -> None:
    torch = pytest.importorskip("torch")
    payload = _payload()
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == payload["checkpoint"]["sha256"]
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert checkpoint["implementation_sha256"] == _implementation_sha256()
    assert tuple(checkpoint["models"]) == MODEL_FAMILIES
    assert checkpoint["selected_family_from_validation"] == payload["training_and_validation_selection"]["selected_family"]
    for family in ("causal_tcn", "small_gru"):
        assert checkpoint["models"][family]["state_dict"]
        assert len(checkpoint["models"][family]["restart_records"]) == 5
    for family in ("gaussian_hmm", "diagonal_kalman", "exponential_recurrence", "run_length_fsm"):
        assert checkpoint["models"][family]["family"] == family


def test_claim_boundary_forbids_universal_or_hardware_upgrade() -> None:
    boundary = _payload()["claim_boundary"]
    assert "synthetic" in boundary["allowed"]
    for token in ("universal", "logical-error", "bit-accurate", "FPGA"):
        assert token in boundary["forbidden"]
