from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cnn_fpga.benchmark import single_mode_cpd_equivalence as audit


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_17_1_single_mode_cpd_equivalence.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _counterexamples(report: dict) -> dict[str, dict]:
    return {row["family"]: row for row in report["counterexamples"]}


def test_report_recomputes_all_gates() -> None:
    report = _report()
    audit.verify_report(report)
    assert report["verdict"] == audit.VERDICT
    assert report["gate_summary"] == {"passed": 15, "failed": []}
    assert report["preregistration"]["seed"] == 61_710_001
    assert report["preregistration"]["sources"] == ["NOH2022_SURFACE_GKP", "LIN2023_CPD"]
    assert [row["source_id"] for row in report["source_scope"]] == report["preregistration"]["sources"]
    assert report["preregistration"]["record_sha256"] == audit._canonical_sha256(
        audit._preregistered_experiment()
    )


def test_analytic_proof_freezes_square_isotropic_half_open_scope() -> None:
    proof = _report()["proof_contract"]
    assert proof["lattice"] == "Lambda=lambda*Z^2"
    assert proof["metric"] == "isotropic Euclidean squared norm"
    assert "no cross term" in proof["separability"]
    assert proof["decision_region"].endswith("[(k_i-1/2)lambda,(k_i+1/2)lambda)")
    assert "larger integer" in proof["tie_rule"]
    assert {"periodic alias/coset summation", "finite-energy state likelihood", "multimode structured lattice"} <= set(proof["excluded"])


def test_independent_cpd_enumeration_matches_ci_at_ties_and_both_sides() -> None:
    below = np.nextafter(0.5, -np.inf)
    above = np.nextafter(0.5, np.inf)
    points = np.asarray([
        [0.5, 0.5],
        [below, -0.5],
        [above, np.nextafter(-0.5, -np.inf)],
        [-7.5, 12.5],
        [2047.5, -2047.5],
    ])
    expected = audit.closest_integer_indices(points)
    actual = audit.brute_force_square_cpd_indices(points)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(expected[0], [1, 1])
    np.testing.assert_array_equal(expected[1], [0, 0])
    np.testing.assert_array_equal(expected[2], [1, -1])
    with pytest.raises(ValueError, match="shape"):
        audit.brute_force_square_cpd_indices([0.0])
    with pytest.raises(ValueError, match="finite"):
        audit.closest_integer_indices([[0.0, np.nan]])
    with pytest.raises(ValueError, match="int64"):
        audit.brute_force_square_cpd_indices([[2.0**62, 0.0]])


def test_complete_q10_pair_domain_is_exhaustive_zero_mismatch() -> None:
    row = _report()["production_domain"]
    assert row["adc_bits_per_axis"] == 10
    assert row["levels_per_axis"] == 1024
    assert row["points"] == 1_048_576
    assert len(row["chunks"]) == 32
    assert sum(chunk["points"] for chunk in row["chunks"]) == row["points"]
    assert all(chunk["mismatches"] == 0 for chunk in row["chunks"])
    assert row["cpd_ci_mismatches"] == 0
    assert row["decision_mismatch_rate"] == 0.0
    assert row["ci_action_sha256"] == row["cpd_action_sha256"]
    assert row["zero_mismatch_one_sided_95_upper_bound"] == pytest.approx(
        audit._zero_event_upper_bound(1_048_576)
    )


def test_canonical_q10_result_is_explicitly_not_used_as_nontrivial_alias_proof() -> None:
    row = _report()["production_domain"]
    assert row["input_is_canonical_centered_cell"] is True
    assert row["canonical_cell_action_is_all_zero"] is True
    assert row["standard_binning_axis_mismatches_from_zero"] == 0
    assert row["positive_endpoint_excluded"] is True
    assert row["maximum_axis_decode_error_lattice_units"] <= 3.0e-16
    assert row["nontrivial_alias_boundary_evidence_is_separate"] is True


def test_one_million_boundary_points_cover_exact_ties_and_two_sides() -> None:
    row = _report()["boundary_audit"]
    assert row["points"] == 1_000_000
    assert row["coordinates"] == 2_000_000
    assert row["seed"] == 61_710_001
    assert row["alias_cell_range_inclusive"] == [-2048, 2048]
    assert len(row["chunks"]) == 10
    assert row["exact_tie_coordinates"] > 300_000
    assert min(row["mode_coordinate_counts"]) > 300_000
    assert row["cpd_ci_mismatches"] == 0
    assert row["maximum_squared_distance_gap"] == 0.0
    assert row["ci_action_sha256"] == row["cpd_action_sha256"]
    assert row["zero_mismatch_one_sided_95_upper_bound"] == pytest.approx(
        audit._zero_event_upper_bound(1_000_000)
    )


def test_biased_and_correlated_likelihoods_are_independently_validated_counterexamples() -> None:
    rows = _counterexamples(_report())
    biased = rows["biased"]
    assert biased["selection_rule"].endswith("not outcome-tuned")
    assert biased["ci_class"] == 0
    assert biased["likelihood_map_class"] == biased["independent_brute"]["class"] == 1
    assert biased["independent_brute"]["weighted_odd"] > biased["independent_brute"]["weighted_even"]

    correlated = rows["correlated"]
    assert correlated["searched_points"] == 10_201
    assert correlated["mismatch_points"] > 0
    assert correlated["ci_class"] == 0
    assert correlated["likelihood_map_class"] == correlated["independent_brute"]["class"] == 2
    np.testing.assert_allclose(
        correlated["project_posterior"],
        correlated["independent_brute"]["posterior"],
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_finite_energy_likelihood_counterexample_reconstructs_from_peak_table() -> None:
    row = _counterexamples(_report())["finite_energy_likelihood"]
    assert row["searched_points"] == 2001
    assert row["mismatch_points"] > 0
    assert row["projector_delta"] == 1.0
    assert row["ci_class"] == 0
    assert row["likelihood_map_class"] == 1
    assert row["maximum_relative_reconstruction_error"] < 1.0e-13
    np.testing.assert_allclose(
        row["logical_state_densities"],
        row["independent_peak_table_densities"],
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_equivalent_cpd_ci_is_counted_once_and_map_remains_distinct() -> None:
    rows = _report()["comparator_registry"]
    euclidean = [row for row in rows if row["equivalence_class"] == "square_isotropic_euclidean_nearest_lattice"]
    assert {row["method_id"] for row in euclidean} == {"closest_integer", "square_euclidean_cpd"}
    assert sum(row["ranking_weight"] for row in euclidean) == 1
    periodic = next(row for row in rows if row["method_id"] == "periodic_coset_map")
    assert periodic["equivalence_class"] == "likelihood_coset_sum"
    claims = _report()["claim_registry"]
    assert claims["CPD_IS_ADDITIONAL_WIN"] == "PROHIBITED_DUPLICATE"
    assert claims["CPD_EQUALS_COSET_MAP"] == "FALSIFIED_BY_THREE_FAMILIES"
    assert claims["SURFACE_GKP_THRESHOLD"] == "NOT_EVALUATED"


def test_runtime_memory_budgets_source_data_and_bindings_are_live() -> None:
    report = _report()
    budget = report["execution_budget_audit"]
    assert budget["within_runtime_budget"] is True
    assert budget["within_memory_budget"] is True
    assert budget["measured_sections_runtime_seconds"] < budget["runtime_budget_seconds"]
    assert budget["peak_tracemalloc_bytes"] < budget["memory_budget_bytes"]
    assert "not decoder or FPGA latency" in budget["boundary_note"]

    csv_path = ROOT / report["source_data"]["path"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"] == 58
    assert {row["record_type"] for row in rows} == {
        "production_chunk", "boundary_chunk", "counterexample", "comparator", "source", "proof"
    }
    for name, binding in report["bindings"].items():
        path = ROOT / binding["path"]
        assert path.exists()
        if name not in {"ontology_initial", "source_audit_initial"}:
            assert audit._sha256(path) == binding["sha256"]
    assert audit._canonical_sha256(audit._ontology_semantic(audit._load(audit.ONTOLOGY))) == report["ontology_semantic_sha256"]
    assert audit._canonical_sha256(audit._source_semantic(audit._load(audit.SOURCE_AUDIT))) == report["source_audit_semantic_sha256"]


def test_mutations_cover_every_gate_and_forged_claims_fail_closed() -> None:
    report = _report()
    mutations = report["semantic_mutation_audit"]
    assert mutations["count"] == mutations["detected"] == 15
    assert {row["target_gate"] for row in mutations["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in mutations["cases"])

    forged = deepcopy(report)
    forged["production_domain"]["cpd_ci_mismatches"] = 1
    with pytest.raises(ValueError, match="gates"):
        audit.verify_report(forged)
    forged = deepcopy(report)
    forged["claim_registry"]["SURFACE_GKP_THRESHOLD"] = "ESTABLISHED"
    with pytest.raises(ValueError, match="gates"):
        audit.verify_report(forged)


def test_markdown_preserves_scope_and_valid_utf8() -> None:
    text = audit.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    assert "single-mode Euclidean CPD 与 CI 等价边界" in text
    assert "canonical-cell 全零" in text
    assert "0.602 threshold" in text
    assert "\ufffd" not in text
