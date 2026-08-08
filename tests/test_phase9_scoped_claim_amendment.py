from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from cnn_fpga.benchmark import phase9_baseline_search_power_registry
from cnn_fpga.benchmark import phase9_scoped_claim_amendment as subject
from cnn_fpga.benchmark import phase9_three_lane_protocol


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9/t9_1_5_scoped_claim_amendment.json"
REPORT = ROOT / "docs/t9_1_5_scoped_claim_amendment.json"
SOURCE = ROOT / "docs/t9_1_5_scoped_claim_amendment_source_data.csv"
MARKDOWN = ROOT / "docs/phase9_scoped_claim_amendment.md"
PARENT_V1 = ROOT / "docs/t9_1_1_three_lane_protocol.json"
PARENT_REGISTRY = (
    ROOT / "docs/t9_1_4_baseline_search_power_registry.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return _load(CONFIG)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return _load(REPORT)


def _definition(
    report: dict[str, Any], state_id: str
) -> dict[str, Any]:
    return next(
        row
        for row in report["state_definitions"]
        if row["state_id"] == state_id
    )


def _offline_gates(
    candidate: dict[str, Any], report: dict[str, Any]
) -> dict[str, bool]:
    return subject.evaluate_gates(
        candidate,
        config_path=CONFIG,
        check_live_files=False,
        expected_parent_summary=report["parent_summary"],
        expected_artifact_registry=report["artifact_registry"],
    )


def test_published_report_live_verifies_all_36_gates(
    report: dict[str, Any],
) -> None:
    checks = subject.verify_report(report)
    assert all(checks.values())
    assert set(checks) == {
        "identity",
        "parents_live",
        "mutation_replay",
        "all_gates",
        "gate_cache",
        "verdict",
        "analysis_sha256",
        "canonical_release_pin",
        "caller_expected_analysis",
        "source_data",
        "markdown",
        "all_current_claims_null",
        "all_performance_fields_null",
    }
    assert tuple(report["gates"]) == subject.GATE_IDS
    assert report["gate_summary"] == {
        "passed": 36,
        "total": 36,
        "failed": [],
    }
    assert report["verdict"] == subject.VERDICT


def test_both_parent_reports_are_live_verified_and_byte_bound(
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    v1 = _load(PARENT_V1)
    registry = _load(PARENT_REGISTRY)
    assert all(phase9_three_lane_protocol.verify_report(v1).values())
    assert all(
        phase9_baseline_search_power_registry.verify_report(
            registry
        ).values()
    )
    assert v1["analysis_sha256"] == (
        "c88110375c358794339e72d672e4624871425fe480e5da091"
        "ddd1d6595255e18"
    )
    assert registry["analysis_sha256"] == (
        "d6c5ac4fd9587854cd6fec7d390c1fd2ddd5300bbe069ff2"
        "db1e16950aa21b7d"
    )
    assert _sha256(PARENT_V1) == config["parent_contract"][
        "parent_report"
    ]["sha256"]
    assert _sha256(PARENT_REGISTRY) == config[
        "registry_parent_contract"
    ]["parent_report"]["sha256"]
    assert report["parent_summary"]["v1"]["raw_report"] == config[
        "parent_contract"
    ]["parent_report"]
    assert report["parent_summary"]["registry"]["raw_report"] == config[
        "registry_parent_contract"
    ]["parent_report"]


def test_migration_table_exactly_covers_dynamic_parent_lane_outputs(
    report: dict[str, Any],
) -> None:
    parent = _load(PARENT_V1)
    expected = set(subject._parent_legacy_outputs(parent))
    actual = {
        (row["legacy_lane"], row["legacy_label"])
        for row in report["legacy_migration_table"]
    }
    assert len(expected) == len(actual) == 29
    assert actual == expected
    assert len(
        {row["migration_id"] for row in report["legacy_migration_table"]}
    ) == 29


def test_only_three_legacy_go_labels_have_narrow_candidate_destinations(
    report: dict[str, Any],
) -> None:
    destinations = {
        (row["legacy_lane"], row["legacy_label"]): row[
            "candidate_destinations"
        ]
        for row in report["legacy_migration_table"]
    }
    assert destinations[
        ("ROUND_LER_SINGLE_MODE", "GO_LER_SOTA")
    ] == ["GO_LER_REGISTERED_BEST"]
    assert destinations[
        ("SIX_STATE_LOGICAL_LIFETIME", "GO_LIFETIME")
    ] == ["GO_LIFETIME_PROJECT_NATIVE"]
    assert destinations[
        ("RAW_IQ_DIGITAL_HIL", "GO_HIL_SPEED")
    ] == ["GO_HIL_INTEGRATED"]
    assert all(
        row["candidate_destinations"] == []
        for row in report["legacy_migration_table"]
        if row["legacy_label"]
        not in {"GO_LER_SOTA", "GO_LIFETIME", "GO_HIL_SPEED"}
    )
    assert all(
        row["automatic_mapping"] is False
        and row["current_mapped_value"] is None
        for row in report["legacy_migration_table"]
    )


def test_scoped_state_inventory_and_every_current_value_are_typed_null(
    report: dict[str, Any],
) -> None:
    assert tuple(
        row["state_id"] for row in report["state_definitions"]
    ) == subject.STATE_IDS
    assert tuple(
        row["state_id"] for row in report["current_states"]
    ) == subject.STATE_IDS
    for row in report["current_states"]:
        assert row["status"].endswith("_NULL")
        assert row["value"] is None
        assert row["verdict"] is None
        assert row["evidence_grade"] is None
        assert row["evidence_refs"] == []
        assert row["numeric_metrics"] is None
        assert row["rank"] is None
        assert row["vote"] is None
        assert row["opened"] is False
        assert row["revocation"] == {
            "status": "NOT_REVOKED",
            "reason": None,
            "evidence_hashes": [],
        }
    assert all(
        value is None
        for value in report["current_claim_state"].values()
    )
    assert all(
        value is None
        for key, value in report["performance_state"].items()
        if key != "protocol_only"
    )
    assert report["performance_state"]["protocol_only"] is True


def test_official_board_external_speed_and_qpu_assets_are_null_not_false(
    report: dict[str, Any],
) -> None:
    assert tuple(report["current_asset_inventory"]) == (
        subject.ASSET_INVENTORY_KEYS
    )
    for row in report["current_asset_inventory"].values():
        assert row["value"] is None
        assert row["verdict"] is None
        assert row["evidence_grade"] is None
        assert row["evidence_refs"] == []
        assert row["status"]
    assert (
        report["current_asset_inventory"]["official_puviani_exact"][
            "status"
        ]
        == "MISSING_EXTERNAL_ASSET"
    )
    assert (
        report["current_asset_inventory"]["board_live_raw_iq_hil"][
            "status"
        ]
        == "MISSING_BOARD"
    )
    assert (
        report["current_asset_inventory"]["qpu_physical_lifetime"][
            "status"
        ]
        == "MISSING_QPU_REAL_GKP"
    )


@pytest.mark.parametrize("state_id", subject.STATE_IDS)
def test_each_state_evaluator_is_nonpromotional_and_distinguishes_null_states(
    report: dict[str, Any],
    state_id: str,
) -> None:
    definition = _definition(report, state_id)
    fixture = subject._passing_evidence(definition)
    fixture_result = subject.evaluate_claim_candidate(
        definition, fixture
    )
    assert fixture_result == {
        "state_id": state_id,
        "status": "FIXTURE_ONLY_NULL",
        "value": None,
        "verdict": None,
        "missing_conditions": [],
        "fixture_only": True,
    }

    passing = subject._passing_evidence(
        definition, fixture_only=False
    )
    complete = subject.evaluate_claim_candidate(
        definition, passing
    )
    assert complete == {
        "state_id": state_id,
        "status": "SCHEMA_COMPLETE_NONPROMOTIONAL_NULL",
        "value": None,
        "verdict": None,
        "missing_conditions": [],
        "fixture_only": False,
    }

    first_true = definition["predicate"].get("true", [None])[0]
    incomplete_evidence = deepcopy(passing)
    if first_true is not None:
        incomplete_evidence[first_true] = False
    else:
        first_non_null = definition["predicate"]["non_null"][0]
        incomplete_evidence[first_non_null] = None
    incomplete = subject.evaluate_claim_candidate(
        definition, incomplete_evidence
    )
    assert incomplete["status"] == "INCOMPLETE_NULL"
    assert incomplete["value"] is None
    assert incomplete["verdict"] is None
    assert incomplete["missing_conditions"]

    revoked = subject.evaluate_claim_candidate(
        definition, passing, revoked=True
    )
    assert revoked["status"] == "REVOKED_NULL"
    assert revoked["value"] is None
    assert revoked["verdict"] is None


def test_external_ler_cannot_open_with_zero_or_unresolved_comparators(
    report: dict[str, Any],
) -> None:
    definition = _definition(report, "GO_LER_EXTERNAL_SOTA")
    evidence = subject._passing_evidence(
        definition, fixture_only=False
    )

    zero = deepcopy(evidence)
    zero["external_same_task_eligible_count"] = 0
    result = subject.evaluate_claim_candidate(definition, zero)
    assert result["status"] == "INCOMPLETE_NULL"
    assert "external_same_task_eligible_count" in result[
        "missing_conditions"
    ]

    unresolved = deepcopy(evidence)
    unresolved["unresolved_eligible_stronger_comparator_count"] = 1
    result = subject.evaluate_claim_candidate(
        definition, unresolved
    )
    assert result["status"] == "INCOMPLETE_NULL"
    assert "unresolved_eligible_stronger_comparator_count" in result[
        "missing_conditions"
    ]
    assert report["external_same_task_contract"][
        "current_external_same_task_eligible_count"
    ] == 0
    assert report["current_claim_state"]["GO_LER_EXTERNAL_SOTA"] is None


def test_registered_ler_schema_complete_does_not_open_external_ler(
    report: dict[str, Any],
) -> None:
    registered = _definition(report, "GO_LER_REGISTERED_BEST")
    external = _definition(report, "GO_LER_EXTERNAL_SOTA")
    registered_complete = subject.evaluate_claim_candidate(
        registered,
        subject._passing_evidence(
            registered, fixture_only=False
        ),
    )
    external_evidence = {
        "go_ler_registered_best": False,
    }
    external_result = subject.evaluate_claim_candidate(
        external, external_evidence
    )
    assert registered_complete["status"] == (
        "SCHEMA_COMPLETE_NONPROMOTIONAL_NULL"
    )
    assert external_result["status"] == "INCOMPLETE_NULL"
    assert external_result["value"] is None


def test_project_native_lifetime_does_not_open_external_physical_or_puviani(
    report: dict[str, Any],
) -> None:
    project = _definition(
        report, "GO_LIFETIME_PROJECT_NATIVE"
    )
    project_complete = subject.evaluate_claim_candidate(
        project,
        subject._passing_evidence(project, fixture_only=False),
    )
    assert project_complete["status"] == (
        "SCHEMA_COMPLETE_NONPROMOTIONAL_NULL"
    )

    for state_id in (
        "GO_LIFETIME_EXTERNAL_SOTA",
        "GO_PUVIANI_NMF_SURPASS",
        "GO_PHYSICAL_LIFETIME",
    ):
        definition = _definition(report, state_id)
        evidence = {
            "go_lifetime_project_native": True,
        }
        result = subject.evaluate_claim_candidate(
            definition, evidence
        )
        assert result["status"] == "INCOMPLETE_NULL"
        assert result["value"] is None


def test_paper_constrained_result_cannot_substitute_official_puviani(
    report: dict[str, Any],
) -> None:
    official = _definition(report, "OFFICIAL_PUVIANI_EXACT")
    paper_constrained = {
        "paper_constrained_reimplementation": True,
        "short_horizon_qualified": True,
    }
    result = subject.evaluate_claim_candidate(
        official, paper_constrained
    )
    assert result["status"] == "INCOMPLETE_NULL"
    assert "official_checkpoint_available" in result[
        "missing_conditions"
    ]
    assert "official_twenty_agent_seed_set_available" in result[
        "missing_conditions"
    ]
    assert "official_selection_ledger_available" in result[
        "missing_conditions"
    ]
    assert "official_six_state_evaluator_available" in result[
        "missing_conditions"
    ]


def test_simulation_cannot_substitute_qpu_physical_lifetime(
    report: dict[str, Any],
) -> None:
    physical = _definition(report, "GO_PHYSICAL_LIFETIME")
    simulation_only = {
        "both_independent_backends_passed": True,
        "simulated_six_state_lifetime": True,
    }
    result = subject.evaluate_claim_candidate(
        physical, simulation_only
    )
    assert result["status"] == "INCOMPLETE_NULL"
    assert "qpu_measured" in result["missing_conditions"]
    assert "real_gkp_state" in result["missing_conditions"]
    assert "raw_measurement_data_sha256" in result[
        "missing_conditions"
    ]


def test_preboard_or_six_cycle_evidence_cannot_substitute_integrated_hil(
    report: dict[str, Any],
) -> None:
    integrated = _definition(report, "GO_HIL_INTEGRATED")
    preboard = {
        "cxxrtl_pass": True,
        "synthesis_pass": True,
        "discriminator_to_action_six_cycles": True,
    }
    result = subject.evaluate_claim_candidate(integrated, preboard)
    assert result["status"] == "INCOMPLETE_NULL"
    assert "real_board_present" in result["missing_conditions"]
    assert "live_adc_to_discriminator_measured" in result[
        "missing_conditions"
    ]
    assert "action_to_trigger_measured" in result[
        "missing_conditions"
    ]
    assert "board_manifest_sha256" in result["missing_conditions"]


def test_recorded_board_replay_cannot_substitute_live_raw_iq_hil(
    report: dict[str, Any],
) -> None:
    integrated = _definition(report, "GO_HIL_INTEGRATED")
    recorded_only = subject._passing_evidence(
        integrated, fixture_only=False
    )
    recorded_only["board_live_raw_iq_hil_qualified"] = False
    recorded_only["live_adc_to_discriminator_measured"] = False
    result = subject.evaluate_claim_candidate(
        integrated, recorded_only
    )
    assert result["status"] == "INCOMPLETE_NULL"
    assert result["value"] is None
    assert "board_live_raw_iq_hil_qualified" in result[
        "missing_conditions"
    ]
    assert "live_adc_to_discriminator_measured" in result[
        "missing_conditions"
    ]


def test_integrated_hil_does_not_open_external_speed(
    report: dict[str, Any],
) -> None:
    integrated = _definition(report, "GO_HIL_INTEGRATED")
    external = _definition(report, "GO_HIL_EXTERNAL_SPEED")
    integrated_complete = subject.evaluate_claim_candidate(
        integrated,
        subject._passing_evidence(
            integrated, fixture_only=False
        ),
    )
    assert integrated_complete["status"] == (
        "SCHEMA_COMPLETE_NONPROMOTIONAL_NULL"
    )
    result = subject.evaluate_claim_candidate(
        external,
        {
            "go_hil_integrated": True,
        },
    )
    assert result["status"] == "INCOMPLETE_NULL"
    assert "external_same_boundary_eligible_count" in result[
        "missing_conditions"
    ]
    assert "hardware_comparator_ledger_sha256" in result[
        "missing_conditions"
    ]


def test_revocation_graph_is_exact_and_propagates_to_null(
    report: dict[str, Any],
) -> None:
    edges = {
        (row["narrower"], row["broader"])
        for row in report["revocation_contract"][
            "prerequisite_edges"
        ]
    }
    assert edges == set(subject.PREREQUISITE_EDGES)
    assert report["revocation_contract"]["mode"] == (
        "FAIL_CLOSED_APPEND_ONLY"
    )
    assert report["revocation_contract"]["current_ledger"] == []
    assert report["revocation_contract"]["current_ledger_anchor"] == (
        subject.revocation_ledger_anchor([])
    )
    assert report["revocation_contract"]["ledger_schema_version"] == (
        subject.REVOCATION_LEDGER_SCHEMA_VERSION
    )
    assert report["revocation_contract"]["genesis_sha256"] == (
        subject.REVOCATION_LEDGER_GENESIS_SHA256
    )
    assert report["revocation_contract"]["trusted_anchor_rule"] == (
        "EVERY_STORED_LEDGER_OR_APPEND_MUST_BE_VERIFIED_AGAINST_AN_EXTERNALLY_PINNED_PRIOR_SNAPSHOT_ANCHOR"
    )
    assert report["revocation_contract"]["on_revocation"][
        "value"
    ] is None
    assert report["revocation_contract"]["on_revocation"][
        "verdict"
    ] is None
    assert report["revocation_contract"]["on_revocation"][
        "status"
    ] == "REVOKED_NULL"
    assert "NULL_NOT_NO_GO" in report["revocation_contract"][
        "propagation"
    ]
    assert len(report["revocation_fixtures"]) == 4
    for fixture in report["revocation_fixtures"]:
        assert (
            fixture["actual_affected_states"]
            == fixture["expected_affected_states"]
        )
        assert fixture["all_affected_null"] is True
        assert fixture["prior_evidence_retained"] is True
        assert fixture["trusted_prior_anchor"] == (
            subject.revocation_ledger_anchor([])
        )
        assert subject.verify_revocation_ledger(
            fixture["ledger"],
            trusted_anchor=fixture["ledger_anchor"],
        )


def _opened_states_for_revocation(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    states = deepcopy(report["current_states"])
    for row in states:
        evidence_hash = hashlib.sha256(
            f"test-open:{row['state_id']}".encode("utf-8")
        ).hexdigest()
        row["status"] = "GO"
        row["value"] = row["state_id"]
        row["verdict"] = row["state_id"]
        row["opened"] = True
        row["evidence_grade"] = "TEST_ONLY"
        row["evidence_refs"] = [evidence_hash]
    return states


def test_revocation_executor_propagates_and_ledger_is_append_only(
    report: dict[str, Any],
) -> None:
    states = _opened_states_for_revocation(report)
    genesis_anchor = subject.revocation_ledger_anchor([])
    by_state = {row["state_id"]: row for row in states}
    project_hash = by_state["GO_LIFETIME_PROJECT_NATIVE"][
        "evidence_refs"
    ][0]
    first_event = {
        "event_id": "REV-TEST-001",
        "revoked_state_id": "GO_LIFETIME_PROJECT_NATIVE",
        "observed_at_utc": "2026-07-25T01:00:00+00:00",
        "prior_evidence_sha256": project_hash,
        "reason_code": "TEST_PROJECT_PREREQUISITE_REVOKED",
    }
    after_first, first_ledger, first_anchor = subject.apply_revocations(
        states,
        prior_ledger=[],
        trusted_prior_anchor=genesis_anchor,
        new_events=[first_event],
    )
    after_first_by_id = {
        row["state_id"]: row for row in after_first
    }
    expected_project_closure = {
        "GO_LIFETIME_PROJECT_NATIVE",
        "GO_LIFETIME_EXTERNAL_SOTA",
        "GO_PUVIANI_NMF_SURPASS",
    }
    assert {
        state_id
        for state_id, row in after_first_by_id.items()
        if row["status"] == "REVOKED_NULL"
    } == expected_project_closure
    assert all(
        after_first_by_id[state_id]["value"] is None
        and after_first_by_id[state_id]["verdict"] is None
        and project_hash
        in after_first_by_id[state_id]["revocation"][
            "evidence_hashes"
        ]
        for state_id in expected_project_closure
    )
    assert subject.verify_revocation_ledger(
        first_ledger, trusted_anchor=first_anchor
    )

    integrated_hash = after_first_by_id["GO_HIL_INTEGRATED"][
        "evidence_refs"
    ][0]
    second_event = {
        "event_id": "REV-TEST-002",
        "revoked_state_id": "GO_HIL_INTEGRATED",
        "observed_at_utc": "2026-07-25T01:01:00+00:00",
        "prior_evidence_sha256": integrated_hash,
        "reason_code": "TEST_HIL_PREREQUISITE_REVOKED",
    }
    after_second, second_ledger, second_anchor = subject.apply_revocations(
        after_first,
        prior_ledger=first_ledger,
        trusted_prior_anchor=first_anchor,
        new_events=[second_event],
    )
    assert subject.verify_revocation_ledger(
        second_ledger, trusted_anchor=second_anchor
    )
    assert subject.verify_revocation_ledger(
        second_ledger,
        trusted_anchor=first_anchor,
        expected_prefix=first_ledger,
        allow_appends=True,
    )
    after_second_by_id = {
        row["state_id"]: row for row in after_second
    }
    assert after_second_by_id["GO_HIL_INTEGRATED"][
        "status"
    ] == "REVOKED_NULL"
    assert after_second_by_id["GO_HIL_EXTERNAL_SPEED"][
        "status"
    ] == "REVOKED_NULL"

    assert not subject.verify_revocation_ledger(
        second_ledger[:-1], trusted_anchor=second_anchor
    )
    assert not subject.verify_revocation_ledger(
        list(reversed(second_ledger)), trusted_anchor=second_anchor
    )
    rewritten = deepcopy(second_ledger)
    rewritten[0]["reason_code"] = "REWRITTEN_HISTORY"
    assert not subject.verify_revocation_ledger(
        rewritten, trusted_anchor=second_anchor
    )

    closure_deleted = deepcopy(first_ledger)
    closure_deleted[0]["propagated_states"] = [
        "GO_LIFETIME_PROJECT_NATIVE"
    ]
    closure_payload = {
        key: value
        for key, value in closure_deleted[0].items()
        if key != "ledger_entry_sha256"
    }
    closure_deleted[0]["ledger_entry_sha256"] = (
        subject._canonical_sha256(closure_payload)
    )
    closure_deleted_anchor = subject.revocation_ledger_anchor(
        closure_deleted
    )
    assert not subject.verify_revocation_ledger(
        closure_deleted, trusted_anchor=closure_deleted_anchor
    )

    rewritten_and_rehashed = deepcopy(second_ledger)
    rewritten_and_rehashed[0]["reason_code"] = "REWRITTEN_HISTORY"
    previous_hash = subject.REVOCATION_LEDGER_GENESIS_SHA256
    for sequence_number, row in enumerate(
        rewritten_and_rehashed, start=1
    ):
        row["sequence_number"] = sequence_number
        row["previous_entry_sha256"] = previous_hash
        payload = {
            key: value
            for key, value in row.items()
            if key != "ledger_entry_sha256"
        }
        row["ledger_entry_sha256"] = subject._canonical_sha256(payload)
        previous_hash = row["ledger_entry_sha256"]
    assert not subject.verify_revocation_ledger(
        rewritten_and_rehashed, trusted_anchor=second_anchor
    )
    with pytest.raises(
        ValueError, match="prior revocation ledger or trusted anchor"
    ):
        subject.apply_revocations(
            after_second,
            prior_ledger=rewritten_and_rehashed,
            trusted_prior_anchor=second_anchor,
            new_events=[],
        )

    with pytest.raises(ValueError, match="invalid or non-append-only"):
        subject.apply_revocations(
            after_second,
            prior_ledger=second_ledger,
            trusted_prior_anchor=second_anchor,
            new_events=[
                {
                    **second_event,
                    "observed_at_utc": "2026-07-25T01:02:00+00:00",
                }
            ],
        )


def test_forbidden_transfers_cover_all_high_risk_rescues(
    report: dict[str, Any],
) -> None:
    transfers = {
        row["transfer_id"]: row
        for row in report["forbidden_transfers"]
    }
    assert len(transfers) == 17
    assert transfers["FT-V2-REGISTERED-TO-EXTERNAL-LER"][
        "rejection"
    ] == "MISSING_EXTERNAL_SAME_TASK_LEDGER_AND_AUDIT"
    assert transfers["FT-V2-PROJECT-TO-PHYSICAL"][
        "rejection"
    ] == "SIMULATION_IS_NOT_QPU_MEASUREMENT"
    assert transfers["FT-V2-PREBOARD-TO-INTEGRATED"][
        "rejection"
    ] == "ESTIMATE_IS_NOT_REAL_BOARD_HIL"
    assert transfers["FT-V2-SIX-CYCLE-TO-END-TO-END"][
        "rejection"
    ] == "LATENCY_BOUNDARY_MISMATCH"
    assert transfers["FT-V2-RECORDED-TO-LIVE-HIL"][
        "rejection"
    ] == "RECORDED_REPLAY_IS_NOT_LIVE_ADC_RAW_IQ"
    assert transfers["FT-V2-PAPER-CONSTRAINED-TO-OFFICIAL"][
        "rejection"
    ] == "MISSING_OFFICIAL_ASSETS"
    assert transfers["FT-V2-MISSING-TO-ZERO"][
        "rejection"
    ] == "TYPED_NULL_REQUIRED"
    assert transfers["FT-V2-NULLABLE-HIL-BLOCKS-ALGORITHM"][
        "rejection"
    ] == "NULLABLE_HIL_DEPENDENCY_MUST_NOT_BLOCK_SOFTWARE_VERDICTS"


def test_external_contract_is_cutoff_ledger_and_nine_field_bound(
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    contract = report["external_same_task_contract"]
    assert contract == config["external_same_task_contract"]
    assert contract["cutoff_inclusive"] == (
        "2026-07-25T01:37:24+08:00"
    )
    assert contract["canonical_work_count"] == 12
    assert contract["current_external_same_task_eligible_count"] == 0
    assert contract["required_signature_fields"] == [
        "input",
        "history",
        "action",
        "physics",
        "online_timing",
        "postselection",
        "denominator",
        "metric",
        "compute",
    ]
    assert contract["registry_best_auto_promotes_external_sota"] is False
    assert contract["current_external_claim_value"] is None


@pytest.mark.parametrize(
    "bad_hash",
    [
        False,
        "",
        "0" * 63,
        "G" * 64,
        object(),
    ],
)
def test_hash_predicates_reject_non_sha256_values(
    report: dict[str, Any],
    bad_hash: Any,
) -> None:
    definition = _definition(report, "GO_LER_EXTERNAL_SOTA")
    evidence = subject._passing_evidence(
        definition, fixture_only=False
    )
    field = definition["predicate"]["non_null"][0]
    evidence[field] = bad_hash
    outcome = subject.evaluate_claim_candidate(
        definition, evidence
    )
    assert outcome["status"] == "INCOMPLETE_NULL"
    assert field in outcome["missing_conditions"]
    assert outcome["value"] is None


@pytest.mark.parametrize(
    "bad_count",
    [math.nan, math.inf, -math.inf, 0.5, True, False, 0, -1],
)
def test_positive_count_predicates_reject_nan_float_bool_and_nonpositive(
    report: dict[str, Any],
    bad_count: Any,
) -> None:
    definition = _definition(report, "GO_LER_EXTERNAL_SOTA")
    evidence = subject._passing_evidence(
        definition, fixture_only=False
    )
    evidence["external_same_task_eligible_count"] = bad_count
    outcome = subject.evaluate_claim_candidate(
        definition, evidence
    )
    assert outcome["status"] == "INCOMPLETE_NULL"
    assert "external_same_task_eligible_count" in outcome[
        "missing_conditions"
    ]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("independent_build_seed_count", 2),
        ("independent_build_seed_count", 2.5),
        ("independent_build_seed_count", True),
        ("measured_transaction_count", 999_999),
        ("measured_transaction_count", math.nan),
        ("bit_mismatch_count", 1),
        ("undefined_action_count", False),
        ("silent_overflow_count", 0.0),
        ("deadline_miss_count", -1),
    ],
)
def test_hil_integer_thresholds_and_zero_counters_are_strict(
    report: dict[str, Any],
    field: str,
    bad_value: Any,
) -> None:
    definition = _definition(report, "GO_HIL_INTEGRATED")
    evidence = subject._passing_evidence(
        definition, fixture_only=False
    )
    evidence[field] = bad_value
    outcome = subject.evaluate_claim_candidate(
        definition, evidence
    )
    assert outcome["status"] == "INCOMPLETE_NULL"
    assert field in outcome["missing_conditions"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: _definition(
            x, "GO_LER_REGISTERED_BEST"
        ).update(allowed_wording="external SOTA"),
        lambda x: _definition(
            x, "GO_LER_REGISTERED_BEST"
        ).update(revocation_triggers=["x"]),
        lambda x: _definition(
            x, "GO_LER_REGISTERED_BEST"
        ).update(current_reason="EVERYTHING_IS_FINE"),
    ],
)
def test_state_wording_revocation_and_reason_are_exact_config_bound(
    report: dict[str, Any],
    mutation: Any,
) -> None:
    candidate = deepcopy(report)
    mutation(candidate)
    gates = _offline_gates(candidate, report)
    assert (
        gates[
            "G08_scoped_state_ids_are_exact_unique_and_lane_bound"
        ]
        is False
    )


def test_nullable_hil_terminal_does_not_block_algorithm_consumers(
    report: dict[str, Any],
) -> None:
    downstream = report["downstream_consumption_contract"]
    assert downstream["direct_consumers"] == [
        "T9.4.5",
        "T9.6.1",
        "T9.6.5",
        "T9.8.1",
    ]
    assert "Done or terminal Blocked/null" in downstream[
        "t9_7_4_nullable_terminal_rule"
    ]
    assert "only HIL states remain null" in downstream[
        "t9_7_4_nullable_terminal_rule"
    ]
    assert downstream["t9_1_2_local_block_rule"].endswith(
        "OFFICIAL_PUVIANI_EXACT and GO_PUVIANI_NMF_SURPASS"
    )
    assert downstream["qpu_local_block_rule"].endswith(
        "only GO_PHYSICAL_LIFETIME"
    )


def test_all_synthetic_fixtures_are_explicitly_non_scientific(
    report: dict[str, Any],
) -> None:
    fixtures = report["predicate_fixtures"]
    assert len(fixtures) == 27
    assert len({row["fixture_id"] for row in fixtures}) == 27
    for state_id in subject.STATE_IDS:
        rows = [
            row for row in fixtures if row["state_id"] == state_id
        ]
        assert len(rows) == 3
        assert {row["outcome"]["status"] for row in rows} == {
            "SYNTHETIC_SCHEMA_COMPLETE_NULL",
            "SYNTHETIC_INCOMPLETE_NULL",
            "SYNTHETIC_REVOKED_NULL",
        }
        assert all(
            row["evidence"]["_fixture_only"] is True for row in rows
        )
        assert all(
            row["outcome"]["fixture_only"] is True for row in rows
        )
        assert all(
            row["outcome"]["value"] is None
            and row["outcome"]["verdict"] is None
            and subject.evaluate_claim_candidate(
                _definition(report, state_id), row["evidence"]
            )["status"]
            == "FIXTURE_ONLY_NULL"
            for row in rows
        )
        assert all(
            row["kind"].startswith("SYNTHETIC_") for row in rows
        )


def test_source_data_is_lossless_and_markdown_contains_atomic_boundaries(
    report: dict[str, Any],
) -> None:
    assert _sha256(SOURCE) == report["source_data"]["sha256"]
    assert SOURCE.stat().st_size == report["source_data"]["bytes"]
    assert subject._csv_lossless(report, SOURCE)
    with SOURCE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"]
    assert len(rows) == len(
        {(row["record_type"], row["record_id"]) for row in rows}
    )
    assert all(
        row["canonical_sha256"]
        == hashlib.sha256(
            row["canonical_json"].encode("utf-8")
        ).hexdigest()
        for row in rows
    )
    reconstructed_analysis = {
        row["record_id"]: json.loads(row["canonical_json"])
        for row in rows
        if row["record_type"] == "analysis_field"
    }
    assert reconstructed_analysis == subject._analysis_payload(report)

    markdown = MARKDOWN.read_text(encoding="utf-8")
    assert markdown == subject._render_markdown(report)
    assert _sha256(MARKDOWN) == report["markdown"]["sha256"]
    for state_id in subject.STATE_IDS:
        assert f"`{state_id}`" in markdown
    for row in report["legacy_migration_table"]:
        assert f"`{row['migration_id']}`" in markdown
    for row in report["forbidden_transfers"]:
        assert f"`{row['transfer_id']}`" in markdown
    assert "不是解码性能" in markdown


def test_timestamp_does_not_change_analysis_but_semantics_do(
    report: dict[str, Any],
) -> None:
    timestamp = deepcopy(report)
    timestamp["generated_at_utc"] = "2099-01-01T00:00:00+00:00"
    assert subject._canonical_sha256(
        subject._analysis_payload(timestamp)
    ) == subject._canonical_sha256(subject._analysis_payload(report))

    semantic = deepcopy(report)
    semantic["external_same_task_contract"][
        "registry_best_auto_promotes_external_sota"
    ] = True
    assert subject._canonical_sha256(
        subject._analysis_payload(semantic)
    ) != subject._canonical_sha256(subject._analysis_payload(report))


def test_every_gate_has_one_replayed_detected_mutation(
    report: dict[str, Any],
) -> None:
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 36
    assert audit["all_detected"] is True
    assert audit["one_per_gate"] is True
    assert len(audit["records"]) == 36
    assert len(
        {row["mutation_id"] for row in audit["records"]}
    ) == 36
    assert [row["target_gate"] for row in audit["records"]] == list(
        subject.GATE_IDS
    )
    assert all(
        row["target_gate"] in row["failed_gates"]
        and row["detected"] is True
        for row in audit["records"]
    )


@pytest.mark.parametrize(
    ("mutation", "target_gate"),
    [
        (
            lambda x: x["current_claim_state"].update(
                GO_LER_EXTERNAL_SOTA="GO_LER_EXTERNAL_SOTA"
            ),
            "G10_all_current_claim_performance_rank_and_vote_values_are_null",
        ),
        (
            lambda x: x["current_asset_inventory"][
                "qpu_physical_lifetime"
            ].update(value=False),
            "G11_official_board_external_speed_and_qpu_inventory_is_typed_null",
        ),
        (
            lambda x: x["hierarchy_contract"].update(
                automatic_transfer=True
            ),
            "G33_registered_project_integrated_states_never_auto_promote_broader_states",
        ),
        (
            lambda x: x["aggregation_contract"].update(
                winner_count="ALLOWED"
            ),
            "G34_weighted_score_winner_count_and_paper_scope_cannot_derive_sota",
        ),
    ],
)
def test_direct_claim_rescue_mutations_fail_closed(
    report: dict[str, Any],
    mutation: Any,
    target_gate: str,
) -> None:
    candidate = deepcopy(report)
    mutation(candidate)
    gates = _offline_gates(candidate, report)
    assert gates[target_gate] is False


def test_tampered_parent_or_registry_binding_fails_closed(
    report: dict[str, Any],
) -> None:
    parent_tamper = deepcopy(report)
    parent_tamper["parent_contract"]["parent_analysis_sha256"] = "0" * 64
    parent_gates = _offline_gates(parent_tamper, report)
    assert (
        parent_gates[
            "G02_t9_1_1_parent_is_live_verified_and_byte_immutable"
        ]
        is False
    )

    registry_tamper = deepcopy(report)
    registry_tamper["registry_parent_contract"][
        "literature_cutoff_inclusive"
    ] = "2099-01-01T00:00:00+00:00"
    registry_gates = _offline_gates(registry_tamper, report)
    assert (
        registry_gates[
            "G04_t9_1_4_registry_is_live_verified_and_exactly_bound"
        ]
        is False
    )


def test_canonical_release_pin_is_external_to_report_selected_paths(
    report: dict[str, Any],
) -> None:
    release_pin = _load(subject.DEFAULT_RELEASE_PIN)
    assert release_pin["analysis_sha256"] == report[
        "analysis_sha256"
    ]
    assert release_pin["config"] == subject._binding(CONFIG)
    assert release_pin["implementation"] == subject._binding(
        subject.IMPLEMENTATION
    )
    assert release_pin["report"] == subject._binding(REPORT)
    assert release_pin["source_data"] == subject._binding(SOURCE)
    assert release_pin["markdown"] == subject._binding(MARKDOWN)
    checks = subject.verify_report(
        report,
        expected_analysis_sha256=report["analysis_sha256"],
    )
    assert checks["canonical_release_pin"] is True
    assert checks["caller_expected_analysis"] is True

    with pytest.raises(
        ValueError, match="accepts only the canonical report path"
    ):
        subject.verify_report(ROOT / "docs" / "alternate.json")
    with pytest.raises(
        ValueError, match="T9.1.5 verification failed"
    ):
        subject.verify_report(
            report, expected_analysis_sha256="0" * 64
        )

    path_tamper = deepcopy(report)
    path_tamper["artifact_registry"]["config"]["path"] = (
        "configs/phase9/alternate.json"
    )
    with pytest.raises(
        ValueError, match="T9.1.5 verification failed"
    ):
        subject.verify_report(path_tamper)


def test_report_verifier_rejects_semantic_tampering(
    report: dict[str, Any],
) -> None:
    tampered = deepcopy(report)
    tampered["current_states"][0]["value"] = (
        "GO_LER_REGISTERED_BEST"
    )
    with pytest.raises(ValueError, match="T9.1.5 verification failed"):
        subject.verify_report(tampered)
