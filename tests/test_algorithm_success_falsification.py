from __future__ import annotations

import copy
import csv
import hashlib
import json

import pytest

from cnn_fpga.benchmark.algorithm_success_falsification import (
    DEFAULT_ARTIFACT,
    DEFAULT_SOURCE_DATA,
    FALLBACK_BRANCH_ID,
    PARENT_ARTIFACTS,
    REQUIRED_REOPEN_GATES,
    STRONG_BRANCH_ID,
    current_parent_composite_hashes,
    decide_branch,
    implementation_sha256,
    inspect_parent_integrity,
    load_parent_artifacts,
    validate_branch_payload,
)


def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict]:
    parents = load_parent_artifacts()
    return parents, inspect_parent_integrity(parents)


def _parent_gate_count(parent: dict) -> int:
    gates = parent.get("gates")
    if isinstance(gates, (dict, list)):
        return len(gates)
    summary_gates = parent.get("gate_summary", {}).get("gates", {})
    return len(summary_gates) if isinstance(summary_gates, dict) else 0


def test_committed_verdict_is_current_fallback_pass_and_source_bound() -> None:
    payload = _artifact()
    assert payload["task_id"] == "T5.1.4"
    assert payload["status"] == "PASS"
    assert payload["implementation_sha256"] == implementation_sha256()
    assert payload["active_branch"]["branch_id"] == FALLBACK_BRANCH_ID
    assert payload["active_branch"]["strong_branch_activated"] is False
    assert payload["active_branch"]["fallback_branch_activated"] is True
    assert payload["gate_summary"] == {
        "passed": len(payload["gates"]),
        "total": len(payload["gates"]),
        "failed": [],
    }
    assert len(payload["gates"]) >= 16 and all(payload["gates"].values())
    assert payload["source_data"]["sha256"] == hashlib.sha256(
        DEFAULT_SOURCE_DATA.read_bytes()
    ).hexdigest()
    assert validate_branch_payload(payload) == ()


def test_every_parent_machine_gate_file_and_composite_binding_is_current() -> None:
    payload = _artifact()
    parents = load_parent_artifacts()
    composites = current_parent_composite_hashes()
    assert set(payload["parent_integrity"]) == set(PARENT_ARTIFACTS)
    for task_id, path in PARENT_ARTIFACTS.items():
        record = payload["parent_integrity"][task_id]
        assert record["machine_pass"] is True
        assert record["machine_gate_count"] >= 8
        assert record["all_declared_files_current"] is True
        assert record["composite_current"] is True
        assert record["passed"] is True
        assert all(item["passed"] for item in record["declared_file_bindings"])
        binding = next(
            row for row in payload["artifact_bindings"] if row["task_id"] == task_id
        )
        assert binding["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        if task_id in composites:
            assert parents[task_id]["implementation_sha256"] == composites[task_id]


def test_strong_branch_fails_every_unmeasured_learned_performance_gate() -> None:
    payload = _artifact()
    predicates = payload["strong_branch_predicates"]
    assert predicates["all_parent_evidence_is_passed_and_current"] is True
    assert predicates["matched_learned_decoder_candidate_executed"] is False
    for name in (
        "candidate_compared_on_same_trace_to_static_and_strong_adaptive_baselines",
        "static_average_p95_and_worst_non_degradation",
        "positive_seed_cluster_effect_against_strong_deployable_baseline",
        "holm_adjusted_repeatable_advantage",
        "no_preregistered_transient_tail_violation",
        "candidate_is_observed_only_causal_and_deployment_scoped",
    ):
        assert predicates[name] is False
    assert payload["evidence_snapshot"]["decoder_lane"][
        "matched_learned_decoder_candidates"
    ] == []
    assert payload["active_branch"]["failed_strong_predicates"] == [
        name for name, value in predicates.items() if not value
    ]


def test_zero_holm_discoveries_and_transient_tail_counterevidence_are_exact() -> None:
    payload = _artifact()
    multiplicity = payload["evidence_snapshot"]["multiplicity"]
    assert multiplicity == {
        "hypotheses": 24,
        "adjustment": "Holm-Bonferroni_two_sided_exact_seed_sign_flip",
        "discoveries": 0,
        "minimum_raw_p_value": 0.03125,
        "minimum_adjusted_p_value": 0.75,
    }
    shift = payload["evidence_snapshot"]["classical_diagnostic"][
        "calibration_shift"
    ]
    assert shift["static_observed_worst_window_ler"] == 37 / 512
    assert shift["kalman_observed_worst_window_ler"] == 55 / 512
    assert shift["transient_tail_violation"] is True
    static = payload["evidence_snapshot"]["classical_diagnostic"]["static_gaussian"]
    assert static["kalman_p_l"] < static["static_p_l"]
    assert static["kalman_observed_worst_window_ler"] < static[
        "static_observed_worst_window_ler"
    ]


def test_fallback_keeps_components_and_software_contracts_without_overclaim() -> None:
    payload = _artifact()
    fallback = payload["fallback_contract"]
    assert fallback["branch_id"] == FALLBACK_BRANCH_ID
    assert all(fallback["prerequisites"].values())
    assert fallback["cnn_or_learned_performance_claim_retained"] is False
    assert fallback["hardware_measurement_claimed"] is False
    assert "not yet one" in fallback["current_integration_status"]
    claims = payload["claim_registry"]
    assert [row["claim_id"] for row in claims["active_allowed"]] == [
        "CL-T514-F01",
        "CL-T514-F02",
        "CL-T514-F03",
    ]
    prohibited = " ".join(claims["prohibited"])
    for token in ("CNN", "universally", "T4.4.5", "T24", "global", "measured FPGA"):
        assert token in prohibited


def test_teacher_student_and_legacy_claims_are_quarantined() -> None:
    payload = _artifact()
    separation = payload["evidence_snapshot"]["teacher_student_separation"]
    assert separation["artifact_task_id"] == "T4.4.5"
    assert separation["controller_matched_model_only"] is True
    assert separation["usable_as_t5_1_decoder_evidence"] is False
    quarantine = payload["claim_registry"]["historical_quarantine"]
    assert "historical frozen-set" in quarantine["T24_PC01"]
    assert "not a CNN or decoder" in quarantine["T4.4.5"]


def test_reopen_contract_requires_new_independent_preregistered_evidence() -> None:
    payload = _artifact()
    reopen = payload["reopen_contract"]
    assert set(reopen["required_gates"]) == set(REQUIRED_REOPEN_GATES)
    assert len(reopen["required_gates"]) == len(REQUIRED_REOPEN_GATES) == 10
    assert reopen["new_seed_registration_timing"] == "before_any_new_evaluation_access"
    assert reopen["existing_1152_windows_may_count_as_independent_seeds"] is False
    required = " ".join(reopen["required_gates"])
    for token in ("holm", "static_average_p95_and_worst", "same_trace", "causal"):
        assert token in required


def test_source_ledger_contains_parent_gates_counterevidence_and_claim_routing() -> None:
    payload = _artifact()
    with DEFAULT_SOURCE_DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == payload["source_data"]["row_count"]
    row_types = {row["row_type"] for row in rows}
    assert row_types == {
        "active_claim",
        "contract_gate",
        "counterevidence",
        "fallback_prerequisite",
        "file_binding",
        "parent_artifact",
        "parent_gate",
        "prohibited_claim",
        "reopen_gate",
        "strong_predicate",
    }
    parents = load_parent_artifacts()
    assert sum(row["row_type"] == "parent_gate" for row in rows) == sum(
        _parent_gate_count(parent) for parent in parents.values()
    )
    assert sum(row["row_type"] == "reopen_gate" for row in rows) == 10
    assert sum(row["row_type"] == "prohibited_claim" for row in rows) == 8


@pytest.mark.parametrize(
    "mutation, expected_error",
    (
        ("force_strong", "active branch"),
        ("active_cnn_claim", "fallback active claim set"),
        ("holm_discovery", "zero-discovery"),
        ("drop_transient", "transient tail counterevidence"),
        ("teacher_promoted", "decoder evidence"),
        ("missing_reopen_holm", "reopen contract"),
        ("window_pseudoreplication", "independent seeds"),
        ("hardware_promotion", "measured hardware"),
    ),
)
def test_semantic_mutations_cannot_rewrite_the_branch(
    mutation: str, expected_error: str
) -> None:
    payload = copy.deepcopy(_artifact())
    if mutation == "force_strong":
        payload["active_branch"] = {
            "branch_id": STRONG_BRANCH_ID,
            "strong_branch_activated": True,
            "fallback_branch_activated": False,
            "failed_strong_predicates": [],
        }
    elif mutation == "active_cnn_claim":
        payload["claim_registry"]["active_allowed"] = [
            {
                "claim_id": "BAD",
                "claim_type": "performance",
                "statement": "CNN wins the expanded matrix",
            }
        ]
    elif mutation == "holm_discovery":
        payload["evidence_snapshot"]["multiplicity"]["discoveries"] = 1
    elif mutation == "drop_transient":
        shift = payload["evidence_snapshot"]["classical_diagnostic"][
            "calibration_shift"
        ]
        shift["kalman_observed_worst_window_ler"] = 0.01
        shift["transient_tail_violation"] = False
    elif mutation == "teacher_promoted":
        payload["evidence_snapshot"]["teacher_student_separation"][
            "usable_as_t5_1_decoder_evidence"
        ] = True
    elif mutation == "missing_reopen_holm":
        payload["reopen_contract"]["required_gates"].remove(
            "holm_adjusted_familywise_discovery"
        )
    elif mutation == "window_pseudoreplication":
        payload["reopen_contract"][
            "existing_1152_windows_may_count_as_independent_seeds"
        ] = True
    elif mutation == "hardware_promotion":
        payload["fallback_contract"]["hardware_measurement_claimed"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)
    assert any(expected_error in error for error in validate_branch_payload(payload))


def test_stale_parent_hash_fails_contract_and_never_activates_strong_branch() -> None:
    parents, integrity = _inputs()
    integrity = copy.deepcopy(integrity)
    integrity["T5.1.3"]["passed"] = False
    result = decide_branch(parents, integrity)
    assert result["status"] == "FAIL"
    assert result["active_branch"]["branch_id"] == FALLBACK_BRANCH_ID
    assert result["strong_branch_predicates"][
        "all_parent_evidence_is_passed_and_current"
    ] is False


def test_decision_is_deterministic_and_does_not_rerun_evaluation() -> None:
    parents, integrity = _inputs()
    first = decide_branch(copy.deepcopy(parents), copy.deepcopy(integrity))
    second = decide_branch(copy.deepcopy(parents), copy.deepcopy(integrity))
    assert first["decision_contract_sha256"] == second["decision_contract_sha256"]
    assert first["active_branch"] == second["active_branch"]
    assert first["determinism_contract"] == {
        "parent_evaluations_rerun": False,
        "new_random_samples_generated": False,
        "decision_rule": "logical_conjunction_of_frozen_strong_predicates",
    }


def test_missing_parent_is_rejected_instead_of_silent_fallback() -> None:
    parents, integrity = _inputs()
    parents.pop("T5.1.3")
    with pytest.raises(ValueError, match="missing parent artifacts"):
        decide_branch(parents, integrity)
