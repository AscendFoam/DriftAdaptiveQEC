from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import pytest

from cnn_fpga.benchmark import multi_agent_seed_selection_audit as audit


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/t5_4_4_multi_agent_seed_selection_audit.json"
SOURCE = ROOT / "docs/t5_4_4_multi_agent_seed_selection_audit_source_data.csv"


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _rehash(mutated: dict) -> dict:
    mutated["contract_sha256"] = audit._canonical_sha256(
        audit._contract_view(mutated)
    )
    return mutated


def test_formal_artifact_is_semantically_valid(report: dict) -> None:
    assert report["status"] == "PASS"
    assert report["audit_summary"]["audit_verdict"] == "PASS_WITH_WARNINGS"
    assert report["gate_summary"] == {"passed": 23, "total": 23}
    assert audit.validate_artifact(report) == ()


def test_all_parents_sources_and_implementations_are_live(report: dict) -> None:
    assert len(report["parent_bindings"]) == 7
    assert all(row["machine_pass"] for row in report["parent_bindings"])
    for row in (
        report["parent_bindings"]
        + report["parent_source_bindings"]
        + report["implementation_bindings"]
    ):
        path = ROOT / row["path"]
        assert path.is_file(), row["path"]
        assert row["sha256"] == audit._sha256(path)


def test_selection_registry_covers_all_six_episodes(report: dict) -> None:
    registry = report["selection_registry"]
    assert len(registry) == 6
    assert {row["lane_id"] for row in registry} == set(report["lanes"])
    assert all(row["evaluation_used_for_selection"] is False for row in registry)


def test_nmf_reports_every_agent_and_seed_without_best_agent_selection(
    report: dict,
) -> None:
    lane = report["lanes"]["nmf_directional"]
    assert lane["agent_selection_rule"] == "none_all_five_paired_agents_retained"
    assert lane["checkpoint_selection_split"] == "validation_only"
    assert lane["evaluation_used_for_agent_selection"] is False
    for split, expected_seeds in (("primary", 8), ("confirmation", 4)):
        section = lane["splits"][split]
        assert len(section["agent_rows"]) == 5
        assert len(section["registered_evaluation_seeds"]) == expected_seeds
        assert len(section["agent_seed_rows"]) == 5 * expected_seeds
        assert section["agent_distribution"]["count"] == 5
        assert len(section["agent_distribution"]["worst_quartile"]) == 2
        assert section["hypothetical_test_best_agent"][
            "inflation_over_all_agent_median"
        ] >= 0.0
        assert section["hypothetical_test_best_agent"][
            "used_for_claim_or_selection"
        ] is False


def test_nmf_distribution_keeps_weak_agents(report: dict) -> None:
    primary = report["lanes"]["nmf_directional"]["splits"]["primary"]
    distribution = primary["agent_distribution"]
    assert distribution["minimum"] == pytest.approx(0.022475147989322153)
    assert distribution["median"] == pytest.approx(0.25721901452195084)
    assert distribution["iqr"] == pytest.approx(0.24201515535490703)
    assert [row["unit_id"] for row in distribution["worst_quartile"]] == [
        "agent-3",
        "agent-0",
    ]


def test_slow_loop_selection_is_validation_only_with_all_families(
    report: dict,
) -> None:
    lane = report["lanes"]["slow_loop_model_selection"]
    assert len(lane["candidate_rows"]) == 6
    assert len(lane["neural_restart_validation_rows"]) == 10
    assert len(lane["evaluation_seed_rows"]) == 48
    assert lane["selected_family"] == "gaussian_hmm"
    diagnostic = lane["hindsight_test_best_diagnostic"]
    assert diagnostic["family"] == "gaussian_hmm"
    assert diagnostic["agrees_with_validation_selection"] is True
    assert diagnostic["postselection_optimism"] == pytest.approx(0.0)
    assert diagnostic["used_to_change_selection"] is False


def test_legacy_student_gap_is_explicit_and_not_promoted(report: dict) -> None:
    lane = report["lanes"]["legacy_student_predecessor"]
    assert len(lane["candidate_validation_rows"]) == 3
    assert lane["selected_restart"] == 0
    assert lane["evaluation_used_for_training_or_selection"] is False
    assert lane["all_candidate_evaluation_available"] is False
    assert "nonselected legacy restart" in lane["missing_evidence"]
    assert "superseded" in lane["role"]


def test_teacher_reports_all_restarts_and_retains_test_ranking_reversal(
    report: dict,
) -> None:
    lane = report["lanes"]["fresh_teacher"]
    assert len(lane["candidate_rows"]) == 3
    assert lane["selected_restart"] == 0
    assert lane["evaluation_used_for_selection"] is False
    assert len(lane["splits"]["primary"]["all_restart_seed_rows"]) == 24
    assert len(lane["splits"]["confirmation"]["all_restart_seed_rows"]) == 12
    for split, expected_gap in (
        ("primary", 0.004127470650250098),
        ("confirmation", 0.0014449750322882426),
    ):
        diagnostic = lane["splits"][split]["hindsight_test_best_diagnostic"]
        assert diagnostic["validation_selected_restart"] == 0
        assert diagnostic["restart_index"] == 2
        assert diagnostic["agrees_with_validation_selection"] is False
        assert diagnostic["postselection_optimism"] == pytest.approx(expected_gap)
        assert diagnostic["used_to_change_selection"] is False


def test_student_reports_all_nine_candidates_and_stays_evaluation_blind(
    report: dict,
) -> None:
    lane = report["lanes"]["low_dimensional_student"]
    assert {(row["dimension"], row["restart_index"]) for row in lane["training_candidate_rows"]} == {
        (dimension, restart)
        for dimension in (1, 2, 4)
        for restart in range(3)
    }
    assert lane["selected_dimension"] == 4
    assert lane["selected_restart"] == 0
    assert lane["evaluation_used_for_selection"] is False
    assert len(lane["best_per_dimension_rows"]) == 3
    diagnostic = lane["hindsight_test_best_diagnostic"]
    assert diagnostic["dimension"] == 4
    assert diagnostic["agrees_with_validation_selection"] is True
    assert diagnostic["postselection_optimism"] == pytest.approx(0.0)


def test_gain_retention_reports_all_strategies_agents_and_seeds(report: dict) -> None:
    lane = report["lanes"]["gain_retention"]
    assert lane["selection_stage"] == "none_frozen_parent_candidates_only"
    assert lane["all_five_mf_agents_retained"] is True
    for split, expected_rows in (("primary", 72), ("confirmation", 36)):
        section = lane["splits"][split]
        assert len(section["strategy_seed_rows"]) == expected_rows
        assert len(section["per_strategy_distributions"]) == 9
        assert section["mf_all_agent_distribution"]["count"] == 5
        assert section["student_gain_retention"]["defined"] is True


def test_every_distribution_has_recomputable_worst_quartile(report: dict) -> None:
    distributions = audit._iter_distributions(report["lanes"])
    assert len(distributions) == report["audit_summary"]["distribution_count"] == 39
    for path, distribution in distributions:
        assert audit._distribution_contract_valid(distribution), path
        assert distribution["worst_quartile_count"] == max(
            1, math.ceil(distribution["count"] / 4)
        )


def test_distribution_direction_changes_which_units_are_worst() -> None:
    rows = [
        {"unit_id": "a", "value": 1.0},
        {"unit_id": "b", "value": 2.0},
        {"unit_id": "c", "value": 3.0},
        {"unit_id": "d", "value": 4.0},
        {"unit_id": "e", "value": 5.0},
    ]
    higher = audit._distribution(rows, higher_is_better=True)
    lower = audit._distribution(rows, higher_is_better=False)
    assert [row["unit_id"] for row in higher["worst_quartile"]] == ["a", "b"]
    assert [row["unit_id"] for row in lower["worst_quartile"]] == ["e", "d"]
    assert higher["median"] == lower["median"] == 3.0
    assert higher["iqr"] == lower["iqr"] == 2.0


def test_audit_summary_counts_and_warnings_are_explicit(report: dict) -> None:
    summary = report["audit_summary"]
    assert summary["selection_episode_count"] == 6
    assert summary["evaluation_unit_row_count"] == 255
    assert summary["active_selection_episodes_using_evaluation"] == 0
    assert summary["hindsight_diagnostic_count"] == 4
    assert summary["hindsight_selection_disagreement_count"] == 2
    assert summary["hindsight_diagnostics_used_to_change_selection"] == 0
    assert "T4.1.5" in summary["legacy_all_candidate_evaluation_gap"]


def test_source_csv_is_complete_and_byte_bound(report: dict) -> None:
    with SOURCE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["source_data"]["row_count"] == 420
    assert report["source_data"]["csv_sha256"] == audit._sha256(SOURCE)
    assert {row["row_type"] for row in rows} >= {
        "nmf_agent",
        "nmf_agent_seed",
        "model_family",
        "model_seed",
        "teacher_restart_seed",
        "student_candidate",
        "retention_strategy_seed",
        "distribution_contract",
        "missing_evidence",
        "gate",
    }


def test_claim_boundary_keeps_revoked_and_hardware_claims_off(report: dict) -> None:
    assert report["gates"]["learned_decoder_performance_branch_remains_revoked"]
    boundary = report["claim_boundary"]
    assert boundary["physical_memory_ler_established"] is False
    assert boundary["device_calibrated"] is False
    assert boundary["hardware_measured"] is False
    assert "best-of-N" in boundary["forbidden"]


@pytest.mark.parametrize(
    "mutation",
    (
        "drop_nmf_agent",
        "select_nmf_after_test",
        "teacher_test_reselection",
        "hide_worst_quartile",
        "hide_legacy_gap",
        "device_claim",
    ),
)
def test_semantic_validator_rejects_selection_bias_mutations(
    report: dict, mutation: str
) -> None:
    changed = copy.deepcopy(report)
    if mutation == "drop_nmf_agent":
        changed["lanes"]["nmf_directional"]["splits"]["primary"]["agent_rows"].pop()
    elif mutation == "select_nmf_after_test":
        changed["lanes"]["nmf_directional"]["evaluation_used_for_agent_selection"] = True
    elif mutation == "teacher_test_reselection":
        changed["lanes"]["fresh_teacher"]["selected_restart"] = 2
        changed["lanes"]["fresh_teacher"]["splits"]["primary"][
            "hindsight_test_best_diagnostic"
        ]["used_to_change_selection"] = True
    elif mutation == "hide_worst_quartile":
        changed["lanes"]["gain_retention"]["splits"]["primary"][
            "mf_all_agent_distribution"
        ]["worst_quartile"] = []
    elif mutation == "hide_legacy_gap":
        changed["lanes"]["legacy_student_predecessor"][
            "all_candidate_evaluation_available"
        ] = True
        changed["lanes"]["legacy_student_predecessor"]["missing_evidence"] = ""
    elif mutation == "device_claim":
        changed["claim_boundary"]["device_calibrated"] = True
        changed["claim_boundary"]["hardware_measured"] = True
    errors = audit.validate_artifact(_rehash(changed))
    assert errors
    assert any(
        "audit lanes" in error or "evidence gates" in error for error in errors
    )
