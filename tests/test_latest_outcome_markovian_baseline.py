from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark.latest_outcome_markovian_baseline import (
    DEFAULT_ARTIFACT,
    DEFAULT_CHECKPOINT,
    DEFAULT_PARENT_ARTIFACT,
    DEFAULT_PARENT_CHECKPOINT,
    DEFAULT_SOURCE_DATA,
    implementation_sha256,
)
from physics.latest_outcome_markovian import COMPUTE_CONTRACT
from physics.nmf_directional_ranking import state_dict_sha256
from physics.nmf_directional_ranking import _bootstrap_agent_difference

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads((ROOT / DEFAULT_ARTIFACT).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_artifact_passes_every_evidence_gate(artifact: dict) -> None:
    assert artifact["task_id"] == "T3.2.7"
    assert artifact["status"] == "PASS"
    assert artifact["required_gates"] == list(artifact["gates"])
    assert all(artifact["gates"].values())
    assert artifact["implementation_sha256"] == implementation_sha256()


def test_parent_and_output_hashes_are_live(artifact: dict) -> None:
    assert artifact["parent_evidence"]["artifact_sha256"] == sha(ROOT / DEFAULT_PARENT_ARTIFACT)
    assert artifact["parent_evidence"]["checkpoint_sha256"] == sha(ROOT / DEFAULT_PARENT_CHECKPOINT)
    assert artifact["checkpoint"]["sha256"] == sha(ROOT / DEFAULT_CHECKPOINT)
    assert artifact["source_data"]["sha256"] == sha(ROOT / DEFAULT_SOURCE_DATA)


def test_parameter_action_and_mac_budgets_are_exact(artifact: dict) -> None:
    assert artifact["compute_contract"] == {
        "input_classes": 3,
        "static_feature_width": 10,
        "adapter_width": 15,
        "hidden_width": 256,
        "output_width": 15,
        "front_parameter_count": 390,
        "downstream_parameter_count": 72463,
        "total_parameter_count": 72853,
        "front_dense_macs": 330,
        "downstream_dense_macs": 71936,
        "total_dense_macs": 72266,
    }
    assert artifact["compute_contract"] == COMPUTE_CONTRACT.__dict__
    assert all(record["parameter_count"] == 72_853 for record in artifact["training_records"])
    assert all(record["dense_mac_count"] == 72_266 for record in artifact["training_records"])


def test_training_selection_and_evaluation_splits_are_disjoint(artifact: dict) -> None:
    config = artifact["config"]
    training = set(config["training_seeds"])
    validation = set(config["validation_seeds"])
    test = set(config["test_seeds"])
    confirmation = set(config["confirmation_seeds"])
    assert not training & validation & test & confirmation
    assert all(not left & right for i, left in enumerate((training, validation, test, confirmation)) for right in (training, validation, test, confirmation)[i + 1 :])
    for record in artifact["training_records"]:
        assert record["best_validation_epoch"] in {item["epoch"] for item in record["validation_history"]}
        best = max(record["validation_history"], key=lambda item: item["selection_score"])
        assert record["best_validation_score"] == best["selection_score"]
        assert record["validation_seeds_used_for_checkpoint_selection_only"] == config["validation_seeds"]


def test_all_agents_and_seed_level_curves_are_retained(artifact: dict) -> None:
    assert len(artifact["training_records"]) == 5
    for lane, expected_seeds in (("primary", 8), ("confirmation", 4)):
        values = artifact["evaluation"][lane]
        assert len(values["exact_mf"]) == len(values["legacy_mf"]) == len(values["history_nmf"]) == 5
        assert len(values["standard"]["per_seed"]) == expected_seeds
        assert all(len(item["per_seed"]) == expected_seeds for strategy in ("exact_mf", "legacy_mf", "history_nmf") for item in values[strategy])


def test_signed_memory_result_is_reported_without_a_desired_direction_gate(artifact: dict) -> None:
    paired = artifact["paired_bootstrap"]["history_nmf_minus_exact_mf"]
    assert set(paired) == {"mean_difference", "ci95_low", "ci95_high", "probability_positive"}
    assert all(np.isfinite(value) for value in paired.values())
    assert not any("ranks" in name or "positive_memory" in name for name in artifact["required_gates"])


def test_cutoff16_confirmation_preserves_the_direction_reversal(artifact: dict) -> None:
    primary = artifact["paired_bootstrap"]["history_nmf_minus_exact_mf"]
    assert primary["ci95_low"] < 0.0 < primary["ci95_high"]
    summary = artifact["summary"]["confirmation"]
    history = summary["history_nmf"]["logical_z_effective_lifetime_cycles"]["values"]
    exact = summary["exact_mf"]["logical_z_effective_lifetime_cycles"]["values"]
    confirmation = _bootstrap_agent_difference(
        history,
        exact,
        seed=artifact["config"]["bootstrap_seed"] + 329,
        repetitions=artifact["config"]["bootstrap_repetitions"],
    )
    assert confirmation["mean_difference"] == pytest.approx(0.5400815393461595)
    assert confirmation["ci95_low"] > 0.0


def test_latest_only_and_leakage_boundaries_are_machine_readable(artifact: dict) -> None:
    assert all(item["earlier_history_invariant_bit_exact"] for item in artifact["latest_only_behavior_audit"])
    assert all(item["stateless_repeat_bit_exact"] for item in artifact["latest_only_behavior_audit"])
    assert all(item["all_three_tokens_have_distinct_outputs"] for item in artifact["latest_only_behavior_audit"])
    assert artifact["leakage_evidence"] == {
        "interface_token_supported": True,
        "production_two_level_simulator_token_count": 0,
        "multilevel_leakage_training_or_evaluation": False,
    }
    assert "multilevel leakage robustness" in artifact["claim_boundary"]["forbidden"]


def test_checkpoint_contains_exactly_the_five_hash_bound_models(artifact: dict) -> None:
    payload = torch.load(ROOT / DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 1
    assert payload["contract_hash"] == artifact["contract_hash"]
    assert len(payload["models"]) == 5
    assert {item["training_seed"] for item in payload["models"]} == set(artifact["config"]["training_seeds"])
    assert all(state_dict_sha256(item["state_dict"]) == item["checkpoint_sha256"] for item in payload["models"])


def test_source_data_is_nontrivial_and_row_count_bound(artifact: dict) -> None:
    with (ROOT / DEFAULT_SOURCE_DATA).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == artifact["source_data"]["row_count"]
    assert len(rows) > 10_000
    assert {row["row_type"] for row in rows} == {"training", "validation", "evaluation_curve", "evaluation_metric"}
    assert {row["lane"] for row in rows} >= {"train", "validation", "primary", "confirmation"}
    assert {row["strategy"] for row in rows} >= {"standard", "legacy_mf", "exact_mf", "history_nmf"}


def test_summary_is_recomputed_from_agent_evaluations(artifact: dict) -> None:
    for lane in ("primary", "confirmation"):
        for strategy in ("legacy_mf", "exact_mf", "history_nmf"):
            values = [item["metric_means"]["logical_z_effective_lifetime_cycles"] for item in artifact["evaluation"][lane][strategy]]
            summary = artifact["summary"][lane][strategy]["logical_z_effective_lifetime_cycles"]
            assert summary["values"] == pytest.approx(values)
            assert summary["mean"] == pytest.approx(float(np.mean(values)))


def test_gradient_coverage_has_only_the_declared_unobserved_leakage_gap(artifact: dict) -> None:
    for record in artifact["training_records"]:
        coverage = record["gradient_coverage"]
        assert coverage["total_parameter_elements"] == 72_853
        assert coverage["covered_parameter_elements"] >= 72_843
        assert coverage["expected_unobserved_leakage_column_elements"] == 10
