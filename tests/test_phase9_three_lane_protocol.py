from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import phase9_three_lane_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(protocol.DEFAULT_REPORT.read_text(encoding="utf-8"))


def _lane(report: dict, lane_id: str) -> dict:
    return next(row for row in report["lanes"] if row["lane_id"] == lane_id)


def _claim(lane: dict, claim_id: str) -> dict:
    return next(row for row in lane["claim_ladder"] if row["claim_id"] == claim_id)


def test_repository_protocol_verifies_end_to_end() -> None:
    assert protocol.verify_report() == {
        "identity": True,
        "all_gates": True,
        "gate_cache": True,
        "verdict": True,
        "analysis_hash": True,
        "source_data": True,
        "markdown_live": True,
        "current_results_null": True,
    }
    report = _report()
    assert report["gate_summary"] == {"passed": 36, "failed": []}
    assert report["verdict"] == protocol.VERDICT


def test_three_signatures_use_closed_24_field_schema_and_are_distinct() -> None:
    report = _report()
    assert tuple(report["task_signature_fields"]) == protocol.TASK_SIGNATURE_FIELDS
    assert len(protocol.TASK_SIGNATURE_FIELDS) == 24
    hashes = []
    for lane in report["lanes"]:
        assert tuple(lane["signature"]) == protocol.TASK_SIGNATURE_FIELDS
        assert all(isinstance(value, str) and value for value in lane["signature"].values())
        digest = protocol._canonical_sha256(lane["signature"])
        assert lane["signature_sha256"] == digest
        hashes.append(digest)
    assert tuple(row["lane_id"] for row in report["lanes"]) == protocol.LANE_IDS
    assert len(set(hashes)) == 3


def test_protocol_pass_does_not_forge_any_performance_result() -> None:
    report = _report()
    ler = _lane(report, "ROUND_LER_SINGLE_MODE")
    lifetime = _lane(report, "SIX_STATE_LOGICAL_LIFETIME")
    hil = _lane(report, "RAW_IQ_DIGITAL_HIL")
    assert ler["current_result"] == {"evaluation_state": "NOT_EVALUATED_NULL", "result_verdict": None}
    assert lifetime["current_result"] == {"evaluation_state": "NOT_EVALUATED_NULL", "result_verdict": None}
    assert hil["current_result"] == {"evaluation_state": "MISSING_BOARD", "result_verdict": None}
    assert report["external_claim_slots"]["PUVIANI_NMF_SURPASS"]["value"] is None
    assert report["external_claim_slots"]["PHYSICAL_BREAK_EVEN"]["value"] is None
    assert report["external_claim_slots"]["RAW_IQ_HIL_SPEED"]["value"] is None


@pytest.mark.parametrize("lane_id", protocol.LANE_IDS)
def test_future_evaluator_separates_unopened_incomplete_no_go_and_go(lane_id: str) -> None:
    report = _report()
    lane = _lane(report, lane_id)
    passing = protocol._all_pass_evidence(lane)

    unopened = protocol.evaluate_result_gate(lane_id, {"outcomes_opened": False}, lanes=report["lanes"])
    assert unopened["result_verdict"] is None
    assert unopened["evaluation_state"] == "NOT_EVALUATED_NULL"

    incomplete_evidence = deepcopy(passing)
    incomplete_evidence.pop(next(key for key in lane["result_gate"]["required_boolean_fields"] if key != "outcomes_opened"))
    incomplete = protocol.evaluate_result_gate(lane_id, incomplete_evidence, lanes=report["lanes"])
    assert incomplete["evaluation_state"] == "INCOMPLETE"
    assert incomplete["result_verdict"] is None

    failed = deepcopy(passing)
    first_metric = next(iter(lane["result_gate"]["numeric_thresholds"]))
    failed[first_metric] = protocol._failing_number(lane["result_gate"]["numeric_thresholds"][first_metric])
    no_go = protocol.evaluate_result_gate(lane_id, failed, lanes=report["lanes"])
    assert no_go["evaluation_state"] == "COMPLETE"
    assert no_go["result_verdict"] == lane["result_gate"]["no_go_verdict"]

    go = protocol.evaluate_result_gate(lane_id, passing, lanes=report["lanes"])
    assert go["evaluation_state"] == "COMPLETE"
    assert go["result_verdict"] == lane["result_gate"]["go_verdict"]


def test_hil_engineering_without_same_task_comparator_cannot_be_speed_go() -> None:
    report = _report()
    lane = _lane(report, "RAW_IQ_DIGITAL_HIL")
    evidence = protocol._all_pass_evidence(lane)
    evidence["same_task_measured_comparator_available"] = False
    outcome = protocol.evaluate_result_gate(lane["lane_id"], evidence, lanes=report["lanes"])
    assert outcome["evaluation_state"] == "INCOMPLETE"
    assert outcome["result_verdict"] is None
    assert outcome["supporting_statuses"] == ["GO_HIL_ENGINEERING_NONRANKING"]


def test_lifetime_inherits_same_physics_observation_action_cost_and_precision() -> None:
    report = _report()
    ler = _lane(report, "ROUND_LER_SINGLE_MODE")
    lifetime = _lane(report, "SIX_STATE_LOGICAL_LIFETIME")
    assert lifetime["inherits_from_lane"] == ler["lane_id"]
    assert all(lifetime["signature"][field] == ler["signature"][field] for field in lifetime["inheritance_fields"])
    assert lifetime["minimum_sequence_cycles"] == 10_000
    assert lifetime["required_state_ensemble"] == ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    assert lifetime["six_state_aggregation_contract"]["average_only_promotion"] == "PROHIBITED"


def test_cycle_time_is_a_frozen_action_conditioned_ledger_not_an_invented_constant() -> None:
    report = _report()
    ledger = report["shared_contracts"]["cycle_time_ledger_contract"]
    assert ledger["physical_cycle_formula"] == "t_sBs_base_plus_measurement_readout_plus_action_conditioned_reset_plus_control_plus_fallback_plus_idle"
    assert ledger["component_values"] is None
    assert ledger["component_value_state"] == "MUST_BE_IMMUTABLY_FILLED_FROM_T9.2_T9.3_DEVICE_PROTOCOL_BEFORE_PILOT"
    assert ledger["per_action_per_trajectory_ledger"] == "REQUIRED"
    assert ledger["missing_numeric_mapping"] == "INCOMPLETE_PHYSICAL_TIME_CLAIM_NOT_ZERO"

    tampered = deepcopy(report)
    tampered["shared_contracts"]["cycle_time_ledger_contract"]["component_values"] = {"constant_cycle_us": 9.848}
    assert protocol.evaluate_gates(tampered, check_live_files=False)["G16_compute_precision_wallclock_and_deadline_fields_are_nonempty"] is False


def test_hil_boundaries_are_all_reported_but_only_raw_iq_is_primary() -> None:
    hil = _lane(_report(), "RAW_IQ_DIGITAL_HIL")
    assert hil["timing_boundaries"] == [
        "decoder_core",
        "discriminator_output_to_action",
        "adc_last_sample_to_trigger",
        "raw_iq_source_to_trigger",
    ]
    assert hil["primary_timing_boundary"] == "raw_iq_source_to_trigger"
    thresholds = hil["result_gate"]["numeric_thresholds"]
    assert thresholds["implementation_seed_count"] == {"op": ">=", "value": 3}
    assert thresholds["transaction_count"] == {"op": ">=", "value": 1_000_000}
    assert thresholds["deadline_miss_count"] == {"op": "==", "value": 0}
    assert "same_task_measured_comparator_available" in hil["result_gate"]["required_boolean_fields"]


def test_only_matched_deployable_is_ranked_and_oracles_are_nonranking() -> None:
    rows = _report()["ontology"]["baseline_classes"]
    assert {row["class_id"] for row in rows} == protocol.BASELINE_CLASS_IDS
    ranked = [row for row in rows if row["ranked"] or row["may_support_sota"]]
    assert ranked == [{"class_id": "MATCHED_DEPLOYABLE_RANKED", "ranked": True, "may_support_sota": True}]
    shared = _report()["shared_contracts"]["baseline_eligibility_contract"]
    assert "observation_tokens_and_history" in shared["all_required_equal"]
    assert "memory_compute_cpu_gpu_budget" in shared["all_required_equal"]
    assert shared["mandatory_baseline_failure"] == "RETAIN_ROW_AND_CLOSE_SOTA_GATE"


def test_evidence_grade_is_a_scope_set_not_a_global_maturity_score() -> None:
    rows = _report()["ontology"]["evidence_grades"]
    assert {row["grade_id"] for row in rows} == protocol.EVIDENCE_GRADE_IDS
    assert all(set(row) == {"grade_id", "claim_scope"} for row in rows)
    assert all("rank" not in row and "level" not in row for row in rows)
    assert next(row for row in rows if row["grade_id"] == "QPU_MEASURED")["claim_scope"] == ["physical_qec_and_break_even"]
    assert next(row for row in rows if row["grade_id"] == "RAW_IQ_HIL_MEASURED")["claim_scope"] == ["raw_iq_source_to_trigger_board_hil"]


@pytest.mark.parametrize(
    ("mutation", "gate"),
    [
        ("lifetime_to_ler", "G22_ler_gate_freezes_each_baseline_and_tail_safety_thresholds"),
        ("accepted_only", "G09_algorithm_lanes_prohibit_postselection_and_accepted_only_denominators"),
        ("core_as_raw_iq", "G10_hil_has_four_boundaries_and_raw_iq_primary"),
        ("sim_as_physical", "G21_physical_break_even_and_raw_iq_speed_remain_null_without_grade"),
        ("hidden_ranked", "G12_baseline_classes_keep_only_matched_deployable_ranked"),
    ],
)
def test_direct_claim_rescue_mutations_fail_closed(mutation: str, gate: str) -> None:
    report = deepcopy(_report())
    if mutation == "lifetime_to_ler":
        _lane(report, "ROUND_LER_SINGLE_MODE")["result_gate"] = deepcopy(_lane(report, "SIX_STATE_LOGICAL_LIFETIME")["result_gate"])
    elif mutation == "accepted_only":
        _lane(report, "SIX_STATE_LOGICAL_LIFETIME")["signature"]["postselection_policy"] = "ACCEPTED_ONLY"
    elif mutation == "core_as_raw_iq":
        _lane(report, "RAW_IQ_DIGITAL_HIL")["primary_timing_boundary"] = "decoder_core"
    elif mutation == "sim_as_physical":
        report["external_claim_slots"]["PHYSICAL_BREAK_EVEN"].update(value=2.0, state="COMPLETE")
    else:
        next(row for row in report["ontology"]["baseline_classes"] if row["class_id"] == "PRIVILEGED_UPPER_BOUND_NONRANKING")["ranked"] = True
    assert protocol.evaluate_gates(report, check_live_files=False)[gate] is False
    with pytest.raises(ValueError, match="verification failed"):
        protocol.verify_report(report)


def test_puviani_and_physical_claims_need_their_own_namespaces_and_grades() -> None:
    report = _report()
    slots = report["external_claim_slots"]
    assert slots["OFFICIAL_PUVIANI_EXACT"]["only_blocks"] == ["OFFICIAL_EXACT_REPRODUCTION", "PUVIANI_NMF_SURPASS"]
    assert slots["PUVIANI_NMF_SURPASS"]["required_grade"] == "OFFICIAL_EXACT_REPRODUCTION"
    assert slots["PHYSICAL_BREAK_EVEN"]["required_grade"] == "QPU_MEASURED"
    lifetime = _lane(report, "SIX_STATE_LOGICAL_LIFETIME")
    assert _claim(lifetime, "LIFE-C4-PUVIANI-SURPASS")["state"] == "BLOCKED_NULL"
    assert _claim(lifetime, "LIFE-C5-PHYSICAL-BREAK-EVEN")["state"] == "BLOCKED_NULL"


def test_source_data_is_lossless_and_every_row_hash_recomputes() -> None:
    report = _report()
    with protocol.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == protocol._source_rows(report)
    assert len(rows) == report["source_data"]["rows"]
    for row in rows:
        assert row["canonical_sha256"] == hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
        assert json.loads(row["payload_json"]) is not None


def test_human_contract_contains_every_atomic_identifier() -> None:
    report = _report()
    text = protocol.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    assert all(f"`{item_id}`" in text for item_id in protocol._atomic_ids(report))
    assert "不是性能实验结果" in text
    assert "不能用 6-cycle core" in text
    assert "weighted score" in text


def test_semantic_projections_ignore_status_or_unrelated_text_but_detect_science_changes() -> None:
    board = (ROOT / "docs" / "new_task_board.md").read_text(encoding="utf-8")
    board_projection = protocol._board_projection(board)
    assert protocol._board_projection(board.replace("| T9.1.1 | In Progress |", "| T9.1.1 | Done |", 1)) == board_projection
    assert protocol._board_projection("unrelated timestamp\n" + board) == board_projection
    assert protocol._board_projection(board.replace("冻结单轮 LER", "弱化单轮 LER", 1)) != board_projection

    risks = (ROOT / "docs" / "new_risks.md").read_text(encoding="utf-8")
    risk_projection = protocol._risk_projection(risks)
    assert protocol._risk_projection(risks.replace("| R-N168 | Open |", "| R-N168 | Mitigated |", 1)) == risk_projection

    plan = (ROOT / "docs" / "experiment_plan.md").read_text(encoding="utf-8")
    plan_projection = protocol._plan_projection(plan)
    assert protocol._plan_projection("unrelated timestamp\n" + plan) == plan_projection
    assert protocol._plan_projection(plan.replace("relative improvement point `>=15%`", "relative improvement point `>=1%`", 1)) != plan_projection


def test_timestamp_only_report_change_preserves_analysis_but_gate_change_does_not() -> None:
    report = _report()
    changed = deepcopy(report)
    changed["generated_at_utc"] = "2099-01-01T00:00:00+00:00"
    assert changed["analysis_sha256"] == protocol._canonical_sha256(protocol._analysis_payload(changed))
    assert protocol.verify_report(changed)["analysis_hash"] is True

    tampered = deepcopy(report)
    _lane(tampered, "ROUND_LER_SINGLE_MODE")["result_gate"]["numeric_thresholds"]["min_simultaneous_relative_lcb_each_baseline"]["value"] = 0.0
    assert tampered["analysis_sha256"] != protocol._canonical_sha256(protocol._analysis_payload(tampered))
    assert protocol.evaluate_gates(tampered, check_live_files=False)["G22_ler_gate_freezes_each_baseline_and_tail_safety_thresholds"] is False


def test_every_gate_has_an_independent_detected_mutation() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(protocol.GATE_IDS) == 36
    assert len(audit["cases"]) == 36
    assert {row["target_gate"] for row in audit["cases"]} == set(protocol.GATE_IDS)
    assert len({row["mutation_id"] for row in audit["cases"]}) == 36
    assert all(row["rejected"] for row in audit["cases"])


def test_all_gate_logic_fixtures_are_explicitly_synthetic() -> None:
    fixtures = _report()["verdict_fixtures"]
    assert len(fixtures) == 13
    assert all(row["scientific_result"] == "SYNTHETIC_GATE_LOGIC_ONLY_NOT_EXPERIMENTAL_EVIDENCE" for row in fixtures)
    assert {row["outcome"]["evaluation_state"] for row in fixtures} == {"NOT_EVALUATED_NULL", "INCOMPLETE", "COMPLETE"}
