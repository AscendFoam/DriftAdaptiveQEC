from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.memory_specific_ablation import (
    MemoryAblationConfig,
    implementation_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs" / "t3_2_11_memory_specific_ablation_validation.json"
SOURCE = ROOT / "docs" / "t3_2_11_memory_specific_ablation_source_data.csv"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"truncation_lengths": ()},
        {"truncation_lengths": (2, 4)},
        {"reset_periods": (1, 0)},
        {"shuffle_seeds": (1, 2)},
        {"shuffle_seeds": (1, 1, 2)},
        {"bootstrap_repetitions": 9999},
        {"bootstrap_seed": True},
    ],
)
def test_config_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MemoryAblationConfig(**kwargs)  # type: ignore[arg-type]


def _payload() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_production_artifact_is_live_complete_and_all_gates_pass() -> None:
    payload = _payload()
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["gate_summary"]["failed"] == 0
    assert payload["gate_summary"]["passed"] == 15
    assert payload["source_data"]["row_count"] == 28_230


def test_all_four_memory_specific_ablation_families_are_present() -> None:
    payload = _payload()
    contract = payload["intervention_contract"]
    for key in (
        "history_shuffle",
        "history_truncation",
        "periodic_hidden_reset",
        "last_outcome_only",
    ):
        assert key in contract
    for lane in ("primary", "confirmation"):
        summary = payload["summary"][lane]
        assert "full_history" in summary
        assert "retrained_exact_budget_last_outcome" in summary
        assert "frozen_parent_last_outcome_only" in summary
        assert len([name for name in summary if name.startswith("history_shuffle_seed")]) == 3
        assert {name for name in summary if name.startswith("history_truncation_")} == {
            "history_truncation_L2",
            "history_truncation_L4",
            "history_truncation_L8",
        }
        assert {name for name in summary if name.startswith("periodic_hidden_reset_")} == {
            "periodic_hidden_reset_R2",
            "periodic_hidden_reset_R4",
            "periodic_hidden_reset_R8",
        }


def test_cross_cutoff_negative_mechanism_verdict_is_retained() -> None:
    payload = _payload()
    verdict = payload["mechanism_verdict"]
    assert verdict["primary_support"] is False
    assert verdict["confirmation_support"] is False
    assert verdict["robust_cross_cutoff_support"] is False
    assert verdict["verdict"] == "cross_cutoff_memory_mechanism_not_supported"
    primary = payload["paired_bootstrap_full_minus_ablation"]["primary"]
    confirmation = payload["paired_bootstrap_full_minus_ablation"]["confirmation"]
    # Retrained latest-only is not worse at cutoff 12, while frozen reset/latest
    # views beat full history at cutoff 16.  Both counterexamples must survive.
    assert primary["full_minus_retrained_exact_budget_last_outcome"][
        "logical_z_effective_lifetime_cycles"
    ]["ci95_low"] < 0.0
    assert confirmation["full_minus_frozen_parent_last_outcome_only"][
        "logical_z_effective_lifetime_cycles"
    ]["ci95_high"] < 0.0


def test_frozen_action_interventions_are_real_causal_and_weight_preserving() -> None:
    payload = _payload()
    audit = payload["action_intervention_audit"]
    assert audit["full_view_bit_exact_for_all_agents"] is True
    assert audit["every_intervention_changes_actions"] is True
    assert audit["sampled_histories"] == 128
    assert len(audit["rows"]) == 50
    assert all(row["changed_history_fraction"] > 0.0 for row in audit["rows"])
    assert payload["gate_summary"]["gates"][
        "frozen_interventions_do_not_mutate_parent_weights"
    ] is True
    assert payload["intervention_contract"]["physical_leakage_tokens_observed"] == 0


def test_source_data_has_raw_curves_metrics_and_action_audit() -> None:
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 28_230
    assert {row["row_type"] for row in rows} == {
        "evaluation_curve",
        "evaluation_metric",
        "action_audit",
    }
    assert {row["lane"] for row in rows} == {
        "primary",
        "confirmation",
        "causal_action_probe",
    }
    assert len({(row["training_seed"], row["evaluation_seed"]) for row in rows if row["lane"] == "primary"}) == 40

