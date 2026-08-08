from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cnn_fpga.benchmark.multimode_posterior_weighted_cpd import (
    ADAPTIVE,
    METHODS,
    ORACLE,
    ROOT,
    _holm_adjust,
    verify_report,
)


REPORT_PATH = ROOT / "docs" / "t6_18_3_multimode_posterior_weighted_cpd.json"
SOURCE_PATH = ROOT / "docs" / "t6_18_3_multimode_posterior_weighted_cpd_source_data.csv"
CONFIG_PATH = ROOT / "configs" / "literature" / "t6_18_3_multimode_drift.json"
CORRECTNESS_PATH = ROOT / "docs" / "t6_18_3_correctness_raw.json"


def _report() -> dict:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def test_frozen_config_matches_preregistration_scope_and_budget() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["task_id"] == "T6.18.3"
    assert config["official_commit"] == "01f9bf1f6970b3e229b43aac9da3325c75518db8"
    assert config["structured_family"]["distance"] == 3
    assert len(config["formal"]["seeds"]) == 32
    assert config["formal"]["cycles_per_cluster"] == 100_000
    assert config["formal"]["window_cycles"] == 512
    assert config["formal"]["families"] == ["smooth", "calibration_shift", "telegraph"]
    assert tuple(config["formal"]["comparators"]) == METHODS
    assert config["posterior_filter"]["update_cadence_cycles"] == 16
    assert config["pilot"]["performance_selection_allowed"] is False
    assert config["statistics"]["oracle_ranking_eligible"] is False


def test_exact_adapter_and_strict_causality_checks_pass() -> None:
    raw = json.loads(CORRECTNESS_PATH.read_text(encoding="utf-8"))
    checks = raw["correctness"]
    assert checks["passed"] is True
    assert checks["official_euclidean_samples"] == 512
    assert checks["official_euclidean_final_list_mismatches"] == 0
    assert checks["positive_scale_invariance_samples"] == 512
    assert checks["positive_scale_invariance_mismatches"] == 0
    assert checks["strict_causal_prefix_cycles"] == 64
    assert checks["strict_causal_prefix_mismatches"] == 0
    assert checks["mutated_suffix_divergence_detected"] is True
    assert checks["posterior_normalization_max_error"] < 1e-12
    assert checks["minimum_predictive_precision"] > 0.0


def test_machine_report_and_all_bindings_verify() -> None:
    verified = verify_report(REPORT_PATH)
    assert verified["verified"] is True
    assert verified["task_id"] == "T6.18.3"


def test_formal_counts_are_complete_not_demo_scale() -> None:
    report = _report()
    counts = report["formal_counts"]
    assert counts["seed_clusters"] == 32
    assert counts["families"] == 3
    assert counts["seed_family_rows"] == 96
    assert counts["cycles_per_cluster"] == 100_000
    assert counts["total_physical_cycles"] == 9_600_000
    assert counts["total_comparator_decodes"] == 38_400_000
    assert counts["registered_windows_per_method"] == 96 * (100_000 // 512)


def test_all_integrity_and_semantic_mutation_gates_pass() -> None:
    report = _report()
    assert report["gate_summary"]["all_passed"] is True
    assert report["gate_summary"]["passed"] == report["gate_summary"]["total"]
    assert report["semantic_mutation_audit"]["all_detected"] is True
    assert report["semantic_mutation_audit"]["detected"] == report["semantic_mutation_audit"]["total"]
    targets = {row["target_gate"] for row in report["semantic_mutation_audit"]["mutations"]}
    assert {"strict_causal_passed", "phase6b_lock_unchanged", "oracle_nonranking", "bindings_valid"} <= targets


def test_raw_counts_recompute_every_scope_p_l() -> None:
    report = _report()
    shard_paths = [ROOT / binding["path"] for binding in report["bindings"] if "shard_" in binding["path"] and binding["path"].endswith(".json")]
    assert len(shard_paths) == 4
    rows = [row for path in shard_paths for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    for scope in ("aggregate", "smooth", "calibration_shift", "telegraph"):
        selected = rows if scope == "aggregate" else [row for row in rows if row["family"] == scope]
        for method in METHODS:
            errors = sum(int(row["errors"][method]) for row in selected)
            cycles = sum(int(row["cycles"]) for row in selected)
            stored = report["summaries"][scope][method]
            assert stored["errors"] == errors
            assert stored["cycles"] == cycles
            assert stored["p_L"] == errors / cycles


def test_paired_contrast_sign_and_ci_are_raw_recomputable() -> None:
    report = _report()
    shard_paths = [ROOT / binding["path"] for binding in report["bindings"] if "shard_" in binding["path"] and binding["path"].endswith(".json")]
    rows = [row for path in shard_paths for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    by_seed = []
    for seed in sorted({row["seed"] for row in rows}):
        selected = [row for row in rows if row["seed"] == seed]
        static = sum(row["errors"]["static_euclidean"] for row in selected) / sum(row["cycles"] for row in selected)
        adaptive = sum(row["errors"][ADAPTIVE] for row in selected) / sum(row["cycles"] for row in selected)
        by_seed.append(static - adaptive)
    stored = report["comparisons"]["aggregate"]["adaptive_vs_static_euclidean"]["improvement"]
    assert np.mean(by_seed) == stored["mean"]
    assert stored["ci_low"] <= stored["mean"] <= stored["ci_high"]
    assert stored["clusters"] == 32
    assert stored["resamples"] == 20_000


def test_oracle_is_reference_only_and_absent_from_holm_family() -> None:
    report = _report()
    assert report["method_contract"]["oracle_privilege"].startswith("true current theta")
    assert report["performance_gate"]["oracle_ranking_eligible"] is False
    for scope in report["comparisons"].values():
        assert all(ORACLE not in contrast for contrast in scope)
    assert all("oracle" not in name for name in report["comparisons"]["aggregate"])


def test_verdict_exactly_matches_frozen_average_and_tail_gate() -> None:
    report = _report()
    gate = report["performance_gate"]
    expected_go = gate["average_gate"] and gate["tail_passed"]
    assert report["verdict"] == (
        "GO_POSTERIOR_WEIGHTED_CPD_DRIFT_GAIN" if expected_go else "NEGATIVE_NO_DRIFT_GAIN"
    )
    for family in ("calibration_shift", "telegraph"):
        row = gate["tail_gate"][family]
        assert row["baseline"] in {"static_euclidean", "weighted_static"}


def test_runtime_and_memory_are_labeled_and_within_preregistered_caps() -> None:
    report = _report()
    memory = report["memory"]
    assert memory["samples"] > 0
    assert len(memory["process_ids"]) == 4
    assert memory["max_concurrent_working_set_bytes"] < 16 * 1024**3
    assert "sampled high-water" in memory["observation"]
    for method in METHODS:
        summary = report["summaries"]["aggregate"][method]
        assert summary["seconds_per_decode"] > 0.0
        assert summary["allocated_bytes_first_decode_max"] > 0


def test_source_data_row_count_and_oracle_labels() -> None:
    report = _report()
    with SOURCE_PATH.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["formal_counts"]["source_data_rows"]
    assert len(rows) > 100_000
    oracle_rows = [row for row in rows if row["method"] == ORACLE]
    assert oracle_rows
    assert {row["evidence"] for row in oracle_rows} == {"ORACLE_REFERENCE_NONRANKING"}


def test_holm_adjustment_is_monotone_and_never_below_raw() -> None:
    raw = {"a": 0.001, "b": 0.02, "c": 0.4}
    adjusted = _holm_adjust(raw)
    assert all(adjusted[key] >= raw[key] for key in raw)
    ordered = sorted(raw, key=raw.get)
    assert [adjusted[key] for key in ordered] == sorted(adjusted[key] for key in ordered)


def test_claim_boundary_does_not_promote_official_fpga_or_phase6b() -> None:
    report = _report()
    forbidden = " ".join(report["claim_boundary"]["forbidden"])
    assert "Phase 6B" in forbidden
    assert "FPGA" in forbidden
    assert "official Lin" in forbidden
    assert "general multimode" in forbidden

