from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark.teacher_student_branch_freeze import (
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    FALLBACK_BRANCH_ID,
    PARENT_ARTIFACTS,
    STRONG_BRANCH_ID,
    current_parent_implementation_hashes,
    decide_branch,
    implementation_sha256,
    inspect_parent_integrity,
    load_parent_artifacts,
)


def _artifact() -> dict[str, object]:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict]:
    parents = load_parent_artifacts()
    implementations = current_parent_implementation_hashes()
    integrity = inspect_parent_integrity(parents)
    return parents, implementations, integrity


def _assert_fallback(result: dict[str, object]) -> None:
    branch = result["active_branch"]
    assert branch["branch_id"] == FALLBACK_BRANCH_ID
    assert branch["strong_branch_activated"] is False
    assert branch["fallback_branch_activated"] is True
    claims = result["claim_registry"]
    assert [item["claim_id"] for item in claims["active_allowed"]] == ["CL-T445-F01"]
    assert result["fallback_contract"]["teacher_or_distillation_claims_retained"] is False
    assert result["status"] == "PASS"
    assert all(result["gates"].values())


def test_committed_artifact_is_current_pass_and_source_bound() -> None:
    payload = _artifact()
    assert payload["task_id"] == "T4.4.5"
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["gate_summary"] == {
        "passed": len(payload["gates"]),
        "total": len(payload["gates"]),
        "failed": [],
    }
    assert len(payload["gates"]) >= 11 and all(payload["gates"].values())
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()


def test_every_parent_is_machine_gate_source_and_file_current() -> None:
    payload = _artifact()
    assert set(payload["parent_provenance"]) == set(PARENT_ARTIFACTS)
    assert all(payload["parent_machine_gate_status"].values())
    assert all(payload["parent_implementation_current"].values())
    assert all(payload["parent_declared_file_integrity"].values())
    for task_id, path in PARENT_ARTIFACTS.items():
        record = payload["parent_provenance"][task_id]
        assert record["artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["machine_gate_count"] >= 16
        assert record["declared_file_bindings"]
        assert all(item["passed"] for item in record["declared_file_bindings"])


def test_qualified_branch_is_narrow_and_retention_specific() -> None:
    payload = _artifact()
    branch = payload["active_branch"]
    assert branch == {
        "branch_id": STRONG_BRANCH_ID,
        "strong_branch_activated": True,
        "fallback_branch_activated": False,
        "failed_evidence_predicates": [],
    }
    assert all(payload["evidence_predicates"].values())
    assert len(payload["retention_metrics"]) == 6
    for row in payload["retention_metrics"]:
        assert row["point"] >= 0.90
        assert row["ci_lower"] >= 0.90
        assert row["passed"] is True
    active_ids = {item["claim_id"] for item in payload["claim_registry"]["active_allowed"]}
    assert active_ids == {"CL-T445-01", "CL-T445-02", "CL-T445-03"}


def test_mf_reversal_and_all_claim_boundaries_remain_explicit() -> None:
    payload = _artifact()
    counter = payload["counterevidence"]
    assert counter["primary_cutoff12_mf_mean_selection_score"] > counter[
        "primary_cutoff12_teacher_selection_score"
    ]
    assert counter["confirmation_cutoff16_teacher_selection_score"] > counter[
        "confirmation_cutoff16_mf_mean_selection_score"
    ]
    assert counter["ordering_reverses_across_cutoffs"] is True
    prohibited = " ".join(payload["claim_registry"]["prohibited"])
    for token in ("universal NMF", "optimizer", "oracle", "leakage", "long-horizon", "FPGA"):
        assert token in prohibited


def test_fallback_and_later_revocation_contract_are_complete() -> None:
    payload = _artifact()
    fallback = payload["fallback_contract"]
    assert fallback["branch_id"] == FALLBACK_BRANCH_ID
    assert "any evidence predicate" in fallback["activation_rule"]
    assert "observed-only" in fallback["algorithmic_scope"]
    assert fallback["teacher_or_distillation_claims_retained"] is False
    tasks = {item["task"] for item in payload["revocation_triggers"]}
    assert tasks >= {"T5.2", "T5.4", "T5.5", "T6", "all"}


def test_source_data_contains_full_parent_gate_and_branch_ledger() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == payload["source_data"]["row_count"] == 112
    row_types = {row["row_type"] for row in rows}
    assert row_types == {
        "allowed_claim",
        "branch_predicate",
        "file_binding",
        "parent_artifact",
        "parent_gate",
        "prohibited_claim",
        "retention_gate",
        "revocation_trigger",
    }
    parents = load_parent_artifacts()
    assert sum(row["row_type"] == "parent_gate" for row in rows) == sum(
        len(payload["gates"]) for payload in parents.values()
    )
    assert sum(row["row_type"] == "retention_gate" for row in rows) == 6
    assert sum(row["row_type"] == "prohibited_claim" for row in rows) == 7


@pytest.mark.parametrize(
    "mutation",
    (
        "parent_status",
        "parent_gate",
        "implementation_hash",
        "declared_file_integrity",
        "retention_threshold",
        "undefined_retention",
        "teacher_not_fresh",
        "student_not_evaluation_blind",
        "mf_reversal_removed",
    ),
)
def test_each_evidence_failure_activates_fallback_without_teacher_claim(
    mutation: str,
) -> None:
    parents, implementations, integrity = _inputs()
    parents = copy.deepcopy(parents)
    implementations = copy.deepcopy(implementations)
    integrity = copy.deepcopy(integrity)
    if mutation == "parent_status":
        parents["T4.4.1"]["status"] = "FAIL"
    elif mutation == "parent_gate":
        gate = next(iter(parents["T4.4.2"]["gates"]))
        parents["T4.4.2"]["gates"][gate] = False
    elif mutation == "implementation_hash":
        implementations["T4.4.3"] = "0" * 64
    elif mutation == "declared_file_integrity":
        integrity["T4.4.4"]["passed"] = False
    elif mutation == "retention_threshold":
        parents["T4.4.4"]["retention_threshold"]["point_fraction"] = 0.89
    elif mutation == "undefined_retention":
        parents["T4.4.4"]["stochastic_retention"]["primary"]["selection_score"][
            "defined"
        ] = False
    elif mutation == "teacher_not_fresh":
        parents["T4.4.1"]["execution"]["fresh_restart_count_in_checkpoint"] = 2
    elif mutation == "student_not_evaluation_blind":
        parents["T4.4.3"]["selection"]["evaluation_blind"] = False
    elif mutation == "mf_reversal_removed":
        parents["T4.4.4"]["stochastic_ten_cycle"]["primary"]["teacher"][
            "selection_score_mean"
        ] = 1.0
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _assert_fallback(decide_branch(parents, implementations, integrity))


def test_decision_hash_is_deterministic_and_does_not_rerun_evaluation() -> None:
    parents, implementations, integrity = _inputs()
    first = decide_branch(copy.deepcopy(parents), implementations, integrity)
    second = decide_branch(copy.deepcopy(parents), implementations, integrity)
    assert first["decision_contract_hash"] == second["decision_contract_hash"]
    assert first["active_branch"] == second["active_branch"]
    assert "execution" not in first
    assert "evaluation" not in first


def test_missing_parent_is_rejected_instead_of_silently_falling_back() -> None:
    parents, implementations, integrity = _inputs()
    parents.pop("T4.4.2")
    with pytest.raises(ValueError, match="missing parent artifacts"):
        decide_branch(parents, implementations, integrity)
