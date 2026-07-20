from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import external_fpga_decoder_refresh as refresh


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_19_2_external_fpga_normalization.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _rows() -> dict[str, dict]:
    return {row["row_id"]: row for row in _report()["external_rows"]}


def test_live_report_recomputes_all_gates() -> None:
    report = _report()
    refresh.verify_report(report)
    assert report["verdict"] == "PASS_EXTERNAL_FPGA_REFRESH_ZERO_SAME_TASK_NO_SPEED_CLAIM"
    assert report["gate_summary"] == {"passed": 18, "failed": []}


def test_base_report_and_external_ledger_are_both_live_and_revalidated() -> None:
    report = _report()
    assert report["base_import"]["external_rows"] == 8
    assert report["base_import"]["external_rows_equal_source_ledger"] is True
    assert report["base_import"]["report_state"] == "PASS_CURRENT_PROJECT_ANCHOR_AFTER_T6.19.2_REPAIR"
    assert len(report["external_rows"]) == 18
    assert {row["row_id"] for row in report["external_rows"] if row["row_origin"] == "T6.19.2_CUTOFF_REFRESH"} == refresh.REFRESH_ROW_IDS


def test_schema_numeric_types_locators_and_nulls_are_strict() -> None:
    report = _report()
    for row in report["external_rows"]:
        assert set(row) == refresh.REQUIRED_ROW_FIELDS
        assert len(row["task_signature"]) == 9
        for field in refresh.NUMERIC_FIELDS:
            value = row[field]
            if value is None:
                assert field not in row["numeric_sources"]
            else:
                assert isinstance(value, (int, float)) and not isinstance(value, bool)
                assert row["numeric_sources"][field]


def test_direct_nn_profiles_are_descriptive_not_ranked() -> None:
    report = _report()
    rows = _rows()
    assert set(report["descriptive_subsets"]["direct_nn_rows"]) == refresh.DIRECT_NN_ROW_IDS
    assert report["descriptive_subsets"]["direct_nn_ranked_rows"] == []
    assert rows["gnn_d7_max_latency"]["reported_latency_ns"] == 988.8
    assert rows["gnn_d7_max_latency"]["latency_cycles"] == 206
    assert rows["gnn_d7_average_latency"]["reported_latency_ns"] == 846.0
    assert rows["gnn_d7_average_latency"]["latency_cycles"] is None
    rethink = rows["rethink_tcn_d9_hls"]
    assert rethink["latency_cycles"] == 267
    assert "271" in rethink["reported_cycle_conflict"]
    assert rethink["ii_cycles"] == 1
    assert any("modules" in value for value in rethink["caveats"])


def test_qpu_physical_fpga_and_estimates_remain_separate() -> None:
    report = _report()
    rows = _rows()
    assert set(report["descriptive_subsets"]["real_qpu_closed_loop_rows"]) == {
        "caune_stability8_9round_feedback",
        "yang_nn_d3_closed_loop",
    }
    assert rows["micro_blossom_d13"]["physical_board_executed"] is True
    assert rows["micro_blossom_d13"]["qpu_in_loop"] is False
    assert rows["gnn_d7_max_latency"]["physical_board_executed"] is False
    assert rows["rethink_tcn_d9_hls"]["evidence_level"].startswith("HLS_SYNTHESIS")


def test_boundary_specific_values_are_not_silently_interchanged() -> None:
    rows = _rows()
    deconet = rows["deconet_100logical_d5"]
    assert deconet["reported_latency_ns"] == 2400.0
    assert deconet["ii_ns"] == 840.0
    ced = rows["ced_d9_tail"]
    assert ced["reported_latency_ns"] is None
    assert ced["p95_latency_ns"] == 650.0
    assert ced["p99_latency_ns"] == 900.0
    assert ced["lut_count"] is ced["ff_count"] is ced["bram_count"] is None
    assert ced["power_w"] is None
    assert ced["branch_dynamic_power_w"] == 1.2
    assert ced["public_rtl_state"] == "PUBLIC_CYCLE_SIMULATOR_ONLY_RTL_NOT_RELEASED_AS_OF_CUTOFF"
    assert rows["gari24_gross_d12"]["reported_latency_ns"] == 273.0
    assert rows["gari3_gross_d12"]["reported_latency_ns"] == 596.0


def test_zero_same_task_count_prohibits_speed_and_sota_claims() -> None:
    report = _report()
    assert report["comparison_eligibility"]["same_task_external_comparator_count"] == 0
    assert report["comparison_eligibility"]["same_task_external_rows"] == []
    assert all(row["same_task_comparable_to_project"] is False for row in report["external_rows"])
    assert report["ranking"] == {
        "global_score": None,
        "ranked_rows": [],
        "policy": "Exact task-signature subset only; zero eligible external rows means no latency/resource winner is emitted.",
    }
    assert report["claim_boundary"]["fpga_speed_advantage"] == "UNESTABLISHED"
    assert report["claim_boundary"]["fastest_or_sota"] == "PROHIBITED"
    for field in (
        "power_w",
        "jitter_ns",
        "deadline_miss_rate",
        "board_measured_latency_ns",
        "physical_transfer_latency_us",
        "physical_commit_latency_us",
    ):
        assert report["project_anchor"][field] is None


def test_exclusion_ledger_csv_bindings_and_mutations_are_complete() -> None:
    report = _report()
    dispositions = {row["candidate_id"]: row for row in report["candidate_dispositions"]}
    assert set(dispositions) == {
        "QASBA",
        "QUEKUF",
        "SOFT_SYNDROME_QLDPC",
        "DIVERSITY_METHODS_EMULATOR",
        "CED_D15_RESOURCE_ONLY",
        "GKP_SPECIFIC_FPGA",
    }
    assert dispositions["GKP_SPECIFIC_FPGA"]["state"] == "NO_QUALIFYING_PRIMARY_SOURCE_IDENTIFIED_BY_FROZEN_SEARCH"
    with (ROOT / report["source_data"]["path"]).open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == report["source_data"]["rows"] == 18
    assert {row["row_id"] for row in csv_rows} == {row["row_id"] for row in report["external_rows"]}
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 18
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda report: report["claim_boundary"].update(fastest_or_sota="ESTABLISHED"), "gates/verdict"),
        (lambda report: report["project_anchor"].update(board_measured_latency_ns=222.2), "gates/verdict"),
        (lambda report: next(row for row in report["external_rows"] if row["row_id"] == "ced_d9_tail").update(power_w=1.2), "gates/verdict"),
    ],
)
def test_claim_and_evidence_promotions_fail_closed(mutation, message: str) -> None:
    forged = deepcopy(_report())
    mutation(forged)
    with pytest.raises(ValueError, match=message):
        refresh.verify_report(forged)
