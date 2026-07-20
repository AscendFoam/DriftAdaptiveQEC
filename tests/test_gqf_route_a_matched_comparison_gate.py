from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from cnn_fpga.benchmark import gqf_route_a_matched_comparison_gate as gate


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_8_5_gqf_route_a_matched_comparison_gate.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_current_negative_branch_recomputes_and_passes_integrity() -> None:
    report = _report()
    gate.verify_report(report)
    assert report["gate_summary"] == {"passed": 10, "failed": 0}
    assert report["verdict"] == "COMPLETE_T6_8_5_INELIGIBLE_NEGATIVE_BRANCH"
    assert report["execution_branch"] == "INELIGIBLE_NEGATIVE_BRANCH_NO_MATCHED_RUN"


def test_all_prerequisites_fail_and_recovery_conditions_are_explicit() -> None:
    report = _report()
    assert len(report["prerequisite_ledger"]) == 8
    assert not any(row["passed"] for row in report["prerequisite_ledger"])
    assert len(report["recovery_conditions"]) == 7
    assert all(row["status"] == "MISSING" for row in report["recovery_conditions"])
    with (ROOT / "docs" / "t6_8_5_gqf_route_a_matched_comparison_gate_source_data.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert all(row["passed"] == "false" for row in rows)


def test_no_unmatched_comparison_or_metrics_exist() -> None:
    report = _report()
    assert report["comparison_run_manifest"] is None
    assert report["comparison_raw_data"] is None
    assert all(value is None for value in report["matched_comparison_metrics"].values())
    assert report["non_substitution"]["project_T4_4_or_T2_3_7_used_as_official_NMF"] is False


def test_claim_boundary_prohibits_surpass_and_official_extension() -> None:
    assert _report()["claim_boundary"] == {
        "same_GQF_lifetime_comparison": "NOT_RUN_INELIGIBLE",
        "paired_lifetime_improvement": "UNDEFINED",
        "surpass_puviani_NMF": "PROHIBITED",
        "retention_compression_safety_extension": "NOT_ESTABLISHED_IN_OFFICIAL_GQF",
    }


def test_targeted_mutations_cover_every_gate_and_tamper_fails_live() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == 10
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])

    forged = deepcopy(report)
    forged["matched_comparison_metrics"]["route_a_T_ch"] = 1000.0
    with pytest.raises(ValueError, match="gates/verdict"):
        gate.verify_report(forged)

