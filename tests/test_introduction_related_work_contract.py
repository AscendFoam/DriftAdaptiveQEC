from __future__ import annotations

import csv
import json

from cnn_fpga.benchmark import introduction_related_work_contract as contract


def _report() -> dict:
    return json.loads(contract.DEFAULT_REPORT.read_text(encoding="utf-8"))


def test_live_report_and_all_gates_pass() -> None:
    report = _report()
    assert report["verdict"] == contract.VERDICT
    assert all(report["gates"].values())
    assert all(contract.verify_report().values())


def test_introduction_and_related_work_have_full_argument_structure() -> None:
    manuscript = _report()["manuscript"]
    assert 6 <= manuscript["introduction_paragraphs"] <= 7
    assert tuple(manuscript["related_subsections"]) == contract.RELATED_SUBSECTIONS
    assert manuscript["section_order"].index("Introduction") < manuscript["section_order"].index("Related Work")
    assert manuscript["section_order"].index("Related Work") < manuscript["section_order"].index("Contract-centric dual-loop method")


def test_citations_resolve_and_required_primary_sources_are_present() -> None:
    manuscript = _report()["manuscript"]
    assert set(manuscript["citation_keys"]).issubset(set(manuscript["bibliography_keys"]))
    assert contract.REQUIRED_CITATIONS.issubset(set(manuscript["citation_keys"]))
    for subsection in contract.RELATED_SUBSECTIONS[:-1]:
        assert len(manuscript["related_subsection_citations"][subsection]) >= 2


def test_six_task_signature_lanes_never_form_a_global_ranking() -> None:
    comparison = _report()["comparison_contract"]
    assert tuple(comparison["literature_lanes"]) == contract.LITERATURE_LANES
    assert tuple(comparison["task_signature_fields"]) == contract.TASK_SIGNATURE_FIELDS
    assert comparison["task_signature_required"] is True
    assert comparison["global_ranking_allowed"] is False
    assert comparison["literature_values_as_project_results"] is False
    assert comparison["same_task_zero_implies_superiority"] is False


def test_negative_and_blocked_claims_are_visible_in_prose() -> None:
    report = _report()
    intro = contract._normalize(report["manuscript"]["introduction"])
    related = contract._normalize(report["manuscript"]["related_work"])
    assert "static joint map has a lower average error rate" in intro
    assert "window map remains a stronger counterexample" in intro
    assert "stopped before formal or rtl work" in intro
    assert "did not support a paper-exact matched reproduction" in related
    assert "zero learned/controller entries eligible" in related
    assert "a count of zero same-task comparators prevents a fair speed ranking" in related
    assert "it does not imply that the project is faster" in related


def test_source_data_is_lossless_and_every_row_is_traceable() -> None:
    report = _report()
    with contract.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(report["claim_evidence_rows"]) == 14
    by_id = {row["row_id"]: row for row in report["claim_evidence_rows"]}
    assert {row["row_id"] for row in rows} == set(by_id)
    for row in rows:
        expected = by_id[row["row_id"]]
        assert json.loads(row["citation_keys_json"]) == expected["citation_keys"]
        assert json.loads(row["source_ids_json"]) == expected["source_ids"]
        assert expected["boundary"]
        assert expected["citation_keys"] or expected["source_ids"]


def test_each_gate_detects_a_targeted_semantic_mutation() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 18
    assert {case["target_gate"] for case in audit["cases"]} == set(report["gates"])
    assert all(case["rejected"] for case in audit["cases"])


def test_no_assertive_overclaim_occurs_in_intro_or_related_work() -> None:
    manuscript = _report()["manuscript"]
    combined = contract._normalize(manuscript["introduction"] + " " + manuscript["related_work"])
    assert not any(pattern in combined for pattern in contract.PROHIBITED_ASSERTIVE_PATTERNS)
    assert "measured board superiority" not in combined
    assert "global winner" not in combined
