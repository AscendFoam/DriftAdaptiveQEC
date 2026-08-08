from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

from cnn_fpga.benchmark import official_structured_cpd_reproduction as audit


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "t6_18_2_official_structured_cpd_reproduction.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_report_recomputes_all_gates_and_semantic_mutations() -> None:
    report = _report()
    audit.verify_report(report)
    assert report["gate_summary"]["passed"] == report["gate_summary"]["total"] == 16
    assert report["semantic_mutation_audit"]["detected"] == report["semantic_mutation_audit"]["count"] == 17


def test_official_commit_license_environment_and_upstream_caveat_are_preserved() -> None:
    report = _report()
    assert report["official_import"]["head"] == audit.OFFICIAL_COMMIT
    assert report["official_import"]["license_sha256"] == audit.OFFICIAL_LICENSE_SHA256
    assert report["official_import"]["pythoncall_version"] == "0.9.10"
    assert report["upstream_tests"]["standard_pkg_test"]["failed"] == 1
    assert report["upstream_tests"]["deterministic_replay"]["passed"] == 2_005
    assert report["upstream_tests"]["official_source_modified"] is False


def test_exact_correctness_and_logical_classification_are_zero_mismatch() -> None:
    result = _report()["correctness"]
    assert result["exact_correctness"]["generic_samples"] == 312
    assert result["exact_correctness"]["generic_mismatches"] == 0
    assert result["exact_correctness"]["single_mode_mismatches"] == 0
    assert result["exact_correctness"]["surface_d3_mismatches"] == 0
    assert result["final_list_validation"] == {"samples": 384, "mismatches": 0, "passed": True}
    assert result["analog_weight_validation"]["max_probability_difference"] == 0.0


def test_official_aggregate_reanalysis_is_exact_and_not_independent_evidence() -> None:
    result = _report()["official_data_reanalysis"]
    assert result["cpd_threshold_mean"] == 0.6024563484296794
    assert result["analog_threshold_mean"] == 0.5995937637028759
    assert result["anchor_max_abs_gap"] == 0.0
    assert result["declared_samples_per_point"] == 10_000_000
    assert result["evidence_class"] == "OFFICIAL_AGGREGATE_DATA_REANALYSIS_NOT_INDEPENDENT_MONTE_CARLO"


def test_independent_grid_counts_and_pairing_are_complete() -> None:
    result = _report()["independent_experiment"]
    integrity = result["raw_integrity"]
    assert integrity["passed"]
    assert integrity["row_count"] == integrity["unique_key_count"] == 864
    assert integrity["total_paired_trials"] == 1_728_000
    assert integrity["minimum_trials_per_seed_cell"] == integrity["maximum_trials_per_seed_cell"] == 2_000
    assert all(row["trials"] == 64_000 for row in result["curves"])


def test_small_distance_crossings_bootstrap_and_tolerance_recompute() -> None:
    result = _report()["independent_experiment"]
    raw = audit._load(audit.THRESHOLD_RAW)["threshold_simulation"]["rows"]
    for method, anchor in (("cpd", 0.602), ("analog_mwpm", 0.599)):
        stored = result["thresholds"][method]
        assert stored == audit._threshold_summary(raw, method, anchor)
        assert stored["within_preregistered_tolerance"]
        bootstrap = stored["bootstrap"]["summaries"]["mean_adjacent_crossing"]
        assert bootstrap["bootstrap_reps"] == 2_000
        assert bootstrap["missing_fraction"] <= 0.05
        assert bootstrap["ci95"][0] <= stored["central_crossings"]["mean_adjacent_crossing"] <= bootstrap["ci95"][1]


def test_cpd_paired_advantage_is_cellwise_and_not_promoted_to_primary_rank() -> None:
    advantage = _report()["independent_experiment"]["paired_advantage"]
    assert advantage["cpd_lower_ler_cells"] == advantage["total_cells"] == 27
    assert advantage["mean_absolute_ler_difference"] < 0.0
    assert advantage["claim_boundary"] == "PAIRED_SMALL_DISTANCE_STATIONARY_FAMILY_ONLY"


def test_runtime_memory_and_asymptotic_claim_boundary_are_explicit() -> None:
    report = _report()
    budget = report["execution_budget"]
    assert budget["threshold_runtime_seconds"] < budget["runtime_budget_seconds"] == 28_800
    assert budget["sampled_host_working_set_high_water_bytes"] < budget["memory_budget_bytes"] == 16 * (1 << 30)
    runtime = report["independent_experiment"]["runtime_scaling"]
    assert runtime["paper_cpd_runtime_anchor"]["value"] == 3.020
    assert runtime["paper_cpd_runtime_anchor"]["evidence"] == "LITERATURE_ONLY_FIG5D"
    assert all(
        payload["evidence_boundary"] == "THREE_SIZE_EMPIRICAL_DIAGNOSTIC_NOT_ASYMPTOTIC_PROOF"
        for key, payload in runtime.items() if key != "paper_cpd_runtime_anchor"
    )


def test_evidence_boundaries_reject_substitution_and_phase6b_promotion() -> None:
    report = _report()
    assert report["evidence_boundary"] == {
        "official_aggregate_may_fill_independent_threshold": False,
        "analog_adapter_is_official_repository_function": False,
        "single_mode_substitution_allowed": False,
        "paper_threshold_precision_claimed_by_small_distance_run": False,
        "hardware_claim": None,
    }
    assert report["phase6b_boundary"] == {
        "verdict": "NO_GO_V5_EARLY_HEADROOM_STOP",
        "promoted_by_phase6c": False,
        "board_measured_claim": None,
    }


def test_source_data_contains_raw_counts_crossings_runtime_and_literature_boundary() -> None:
    report = _report()
    with (ROOT / report["source_data"]["path"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == report["source_data"]["rows"]
    record_types = {row["record_type"] for row in rows}
    assert {"exact_cvp", "single_mode", "official_aggregate_crossing", "seed_cell", "aggregate_curve", "independent_crossing", "runtime", "literature_runtime"} <= record_types
    assert {row["value_state"] for row in rows if row["record_type"] == "literature_runtime"} == {"LITERATURE_ONLY"}


def test_live_gates_reject_literature_fill_and_official_adapter_upgrade() -> None:
    report = _report()
    forged = deepcopy(report)
    forged["independent_experiment"]["curves"][0]["ler"] = 0.602
    assert not audit.evaluate_gates(forged)["G10_independent_curves_recompute_from_integer_counts_without_literature_fill"]
    forged = deepcopy(report)
    forged["evidence_boundary"]["analog_adapter_is_official_repository_function"] = True
    assert not audit.evaluate_gates(forged)["G14_official_aggregate_independent_monte_carlo_and_adapter_boundaries_are_not_merged"]


def test_all_artifact_bindings_are_live() -> None:
    report = _report()
    for binding in report["bindings"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert audit._sha256(path) == binding["sha256"]
        assert path.stat().st_size == binding["bytes"]
