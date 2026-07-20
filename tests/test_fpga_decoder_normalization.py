from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import fpga_decoder_normalization as norm


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_8_6_fpga_decoder_normalization.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_current_report_recomputes_and_passes_all_gates() -> None:
    report = _report()
    norm.verify_report(report)
    assert report["verdict"] == "PASS_FPGA_DECODER_NORMALIZATION_NO_SPEED_CLAIM"
    assert report["gate_summary"] == {"passed": 13, "failed": 0}
    assert len(report["rows"]) == 10


def test_every_numeric_value_has_a_locator_and_absence_is_null() -> None:
    report = _report()
    for row in report["rows"]:
        locators = row["numeric_sources"]
        for field in norm.NUMERIC_FIELDS:
            value = row[field]
            if value is None:
                assert field not in locators
            else:
                assert isinstance(value, (int, float)) and not isinstance(value, bool)
                assert locators[field]


def test_latency_boundaries_separate_mean_core_and_closed_loop() -> None:
    rows = {row["row_id"]: row for row in _report()["rows"]}
    assert rows["helios_d21"]["average_per_round_ns"] == 11.5
    assert rows["helios_d21"]["source_to_action_ns"] is None
    assert rows["lilliput_d5_m2"]["decoder_core_ns"] == 42.0
    assert rows["lilliput_d5_m2"]["closed_loop_ns"] is None
    assert rows["yang_nn_d3_closed_loop"]["decoder_core_ns"] == 124.0
    assert rows["yang_nn_d3_closed_loop"]["closed_loop_ns"] == 550.0
    assert rows["caune_stability8_9round_feedback"]["source_to_action_ns"] == 9600.0


def test_no_external_row_is_speed_comparable_and_integrated_values_fail_closed() -> None:
    report = _report()
    rows = {row["row_id"]: row for row in report["rows"]}
    external = [row for row in report["rows"] if row["row_id"] in norm.EXTERNAL_ROW_IDS]
    assert len(external) == 8
    assert all(row["direct_speed_comparable_to_project"] is False for row in external)
    assert report["comparison_eligibility"]["same_task_external_rows"] == []
    assert report["claim_boundary"]["fastest_or_sota"] == "PROHIBITED"
    integrated = rows["project_t6_route_a_integrated_cxxrtl"]
    assert integrated["latency_cycles"] == 6
    assert integrated["reported_latency_ns"] is None
    assert integrated["clock_mhz"] is None
    assert integrated["lut"] is None
    assert integrated["power_w"] is None


def test_source_csv_and_semantic_mutations_are_complete() -> None:
    report = _report()
    with (ROOT / report["source_data"]["path"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 10
    assert {row["row_id"] for row in rows} == {row["row_id"] for row in report["rows"]}
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 13
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])

    forged = deepcopy(report)
    forged["rows"][1]["direct_speed_comparable_to_project"] = True
    with pytest.raises(ValueError, match="gates/verdict"):
        norm.verify_report(forged)
