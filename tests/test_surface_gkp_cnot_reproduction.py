from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import surface_gkp_cnot_reproduction as audit


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_17_2_noh_cnot_ci_ml_reproduction.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_recomputes_all_gates_and_frozen_source_locks() -> None:
    report = _report()
    audit.verify_report(report)
    assert report["verdict"] == audit.VERDICT
    assert report["gate_summary"] == {"passed": 15, "failed": []}
    assert report["preregistration"]["record_sha256"] == audit._canonical_sha256(audit._preregistered_experiment())
    assert report["preregistration"]["source_record_sha256"] == audit._canonical_sha256(audit._source_record())
    assert report["preregistration"]["seeds"] == list(range(61_720_001, 61_720_033))


def test_source_sufficiency_has_every_equation_and_keeps_outer_code_excluded() -> None:
    source = _report()["source_sufficiency"]
    assert source["status"] == "PASS_SOURCE_SUFFICIENCY"
    assert source["complete_ingredients"] == len(source["ingredients"]) == 9
    assert source["missing_ingredients"] == []
    locators = " ".join(row["locator"] for row in source["ingredients"])
    assert "Eq. (24)" in locators
    assert "Eq. (C19)" in locators
    assert "Algorithm 1" in locators and "Algorithm 2" in locators
    assert "Eqs. (C20)-(C21)" in locators
    assert any("9.9 dB" in row for row in source["explicit_exclusions"])


def test_eight_shift_transform_matches_equation_and_covariance() -> None:
    basis = np.eye(8)
    q, p = audit.net_cnot_shifts(basis, 1.0)
    np.testing.assert_array_equal(q[:, 0], [1, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(q[:, 1], [1, 0, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(p[:, 0], [0, 0, 0, 0, -1, 1, 1, 0])
    np.testing.assert_array_equal(p[:, 1], [0, 0, 0, 0, 1, 0, 0, 1])

    rng = np.random.default_rng(4172)
    q, p = audit.net_cnot_shifts(rng.standard_normal((200_000, 8)), 0.3)
    np.testing.assert_allclose(np.cov(q, rowvar=False) / 0.09, [[2, 1], [1, 3]], atol=0.025)
    np.testing.assert_allclose(np.cov(p, rowvar=False) / 0.09, [[3, -1], [-1, 2]], atol=0.025)
    with pytest.raises(ValueError, match=r"\(n,8\)"):
        audit.net_cnot_shifts(np.zeros((2, 7)), 1.0)
    with pytest.raises(ValueError, match="positive"):
        audit.net_cnot_shifts(np.zeros((2, 8)), 0.0)


def test_paper_voronoi_algorithms_match_independent_25_candidate_likelihood() -> None:
    rng = np.random.default_rng(61_729_777)
    points = rng.uniform(-20 * audit.LATTICE, 20 * audit.LATTICE, size=(100_000, 2))
    np.testing.assert_array_equal(audit.q_ml_decode(points), audit.brute_likelihood_decode(points, "q"))
    np.testing.assert_array_equal(audit.p_ml_decode(points), audit.brute_likelihood_decode(points, "p"))
    with pytest.raises(ValueError, match="quadrature"):
        audit.brute_likelihood_decode(points, "x")
    with pytest.raises(ValueError, match="finite"):
        audit.q_ml_decode([[0.0, np.nan]])


def test_all_points_reach_predeclared_stopping_rule_and_raw_counts_conserve() -> None:
    for point in _report()["points"]:
        assert point["stop_reason"] == "TARGET_FAILURES_BOTH_METHODS"
        assert min(point["ci_failures"], point["ml_failures"]) >= 2000
        assert point["trials"] <= 5_000_000
        assert len(point["clusters"]) == 32
        assert sum(row["trials"] for row in point["clusters"]) == point["trials"]
        assert point["both_fail"] + point["ci_only"] == point["ci_failures"]
        assert point["both_fail"] + point["ml_only"] == point["ml_failures"]
        assert point["both_fail"] + point["ci_only"] + point["ml_only"] + point["neither"] == point["trials"]


def test_crn_and_exact_deterministic_counts_cover_low_failure_anchor() -> None:
    points = {row["squeezing_db"]: row for row in _report()["points"]}
    assert [(points[db]["trials"], points[db]["ci_failures"], points[db]["ml_failures"]) for db in (9.0, 12.0, 13.0)] == [
        (65_536, 6_563, 4_518),
        (589_824, 5_146, 2_158),
        (2_424_832, 6_362, 2_037),
    ]
    for point in points.values():
        assert point["ci_primitive_draw_sha256"] == point["ml_primitive_draw_sha256"]
        assert len(point["ci_primitive_draw_sha256"]) == 64


def test_all_six_anchor_discrepancies_pass_both_frozen_tolerances() -> None:
    rows = _report()["anchor_discrepancies"]
    assert len(rows) == 6
    assert {(row["squeezing_db"], row["method"]) for row in rows} == {
        (9.0, "CI"), (9.0, "ML"), (12.0, "CI"), (12.0, "ML"), (13.0, "CI"), (13.0, "ML")
    }
    assert all(row["absolute_pass"] and row["relative_pass"] and row["joint_pass"] for row in rows)
    assert max(row["relative_discrepancy"] for row in rows) < 0.016


def test_paired_statistics_are_significant_and_cluster_resampled() -> None:
    report = _report()
    assert all(value < 0.05 for value in report["multiplicity"]["holm_adjusted_mcnemar_p"])
    for point in report["points"]:
        assert point["paired_difference"] > 0
        assert point["relative_failure_reduction"] > 0
        assert point["cluster_bootstrap"]["resamples"] == 20_000
        assert point["cluster_bootstrap"]["unit"] == "independent_seed_cluster"
        assert point["cluster_bootstrap"]["ci_minus_ml_95_interval"][0] > 0
        assert point["ci_wilson_95"][0] <= point["ci_probability"] <= point["ci_wilson_95"][1]
        assert point["ml_wilson_95"][0] <= point["ml_probability"] <= point["ml_wilson_95"][1]


def test_boundary_audit_uses_actual_two_sided_crossings_and_zero_mismatch() -> None:
    row = _report()["boundary_audit"]
    assert row["points"] == 100_000
    assert row["boundary_pairs"] == row["one_sided_crossing_pairs"] == 50_000
    assert row["all_pairs_cross_decision_boundary"] is True
    assert row["exact_ties_excluded_due_nonunique_argmin"] is True
    assert row["production_brute_mismatches"] == 0
    assert row["maximum_likelihood_cost_gap"] == 0.0
    assert row["production_action_sha256"] == row["brute_action_sha256"]
    assert {chunk["quadrature"] for chunk in row["chunks"]} == {"q", "p"}


def test_operation_cost_and_claim_registry_do_not_invent_latency_or_threshold() -> None:
    report = _report()
    cost = report["operation_cost"]
    assert cost["ci"]["scalar_round_to_nearest"] == cost["ml"]["scalar_round_to_nearest"] == 4
    assert cost["ml"]["correlated_2d_classifiers"] == 2
    assert cost["measured_latency_ns"] is None
    assert "not latency" in cost["python_runtime_boundary"]
    assert report["claim_registry"]["NOH_9P9DB_OUTER_THRESHOLD"] == "LITERATURE_ONLY_NULL"
    assert report["claim_registry"]["CI_LT50NS"] == "NULL_UNSOURCED"
    assert report["claim_registry"]["HARDWARE_RESOURCES"] == "NULL_NOT_IMPLEMENTED"


def test_source_data_bindings_and_budgets_are_live() -> None:
    report = _report()
    assert report["execution_budget_audit"]["within_runtime_budget"] is True
    assert report["execution_budget_audit"]["within_memory_budget"] is True
    csv_path = ROOT / report["source_data"]["path"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 119
    assert {row["record_type"] for row in rows} == {"source", "seed_cluster", "point_summary", "anchor", "boundary"}
    for name, binding in report["bindings"].items():
        path = ROOT / binding["path"]
        assert path.exists()
        if name not in {"ontology_initial", "source_audit_initial"}:
            assert audit._sha256(path) == binding["sha256"]


def test_mutations_cover_every_gate_and_forged_claims_fail_closed() -> None:
    report = _report()
    mutations = report["semantic_mutation_audit"]
    assert mutations["count"] == mutations["detected"] == 15
    assert {row["target_gate"] for row in mutations["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in mutations["cases"])

    forged = deepcopy(report)
    forged["claim_registry"]["NOH_9P9DB_OUTER_THRESHOLD"] = "PROJECT_REPRODUCED"
    with pytest.raises(ValueError, match="gates"):
        audit.verify_report(forged)

    forged = deepcopy(report)
    forged["anchor_discrepancies"][0].update(
        estimate=0.9,
        absolute_discrepancy=0.0,
        relative_discrepancy=0.0,
        absolute_pass=True,
        relative_pass=True,
        joint_pass=True,
    )
    with pytest.raises(ValueError, match="gates"):
        audit.verify_report(forged)


def test_markdown_is_utf8_and_names_gate_level_boundary() -> None:
    text = (ROOT / "docs" / "noh_cnot_ci_ml_reproduction.md").read_text(encoding="utf-8")
    assert "gate-level" in text
    assert "LITERATURE_ONLY_NULL" in text
    assert "100,000" in text
    assert "�" not in text
