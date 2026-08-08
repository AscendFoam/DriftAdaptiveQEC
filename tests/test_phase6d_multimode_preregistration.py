from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import pytest

from cnn_fpga.benchmark import phase6d_multimode_preregistration as prereg


ROOT = Path(__file__).resolve().parents[1]


def _report() -> dict:
    return json.loads(prereg.DEFAULT_REPORT.read_text(encoding="utf-8"))


def _splits(report: dict) -> dict[str, dict]:
    return {row["split_id"]: row for row in report["splits"]}


def test_repository_preregistration_verifies_end_to_end() -> None:
    assert prereg.verify_report() == {
        "identity": True,
        "gates": True,
        "verdict": True,
        "analysis_hash": True,
    }
    report = _report()
    assert report["gate_summary"] == {"passed": 20, "failed": []}
    assert report["verdict"] == prereg.VERDICT
    assert report["seal_state"] == "SEALED_PRE_OUTCOME"


def test_exact_four_splits_have_disjoint_seed_namespaces_and_roles() -> None:
    report = _report()
    splits = _splits(report)
    assert tuple(splits) == prereg.SPLIT_IDS
    assert {key: len(value["seeds"]) for key, value in splits.items()} == {
        "train": 24,
        "calibration": 16,
        "pilot": 24,
        "formal": 60,
    }
    all_seeds = [seed for split in splits.values() for seed in split["seeds"]]
    assert len(all_seeds) == len(set(all_seeds)) == 124
    assert all(splits[key]["role"] == prereg.EXPECTED_ROLES[key] for key in prereg.SPLIT_IDS)


def test_opened_t6183_seeds_scales_drift_and_fixed_spatial_pattern_are_absent() -> None:
    audit = _report()["opened_development_audit"]
    assert audit["seed_overlap"] == []
    assert audit["base_sigma_overlap"] == []
    assert audit["amplitude_overlap"] == []
    assert audit["duration_overlap"] == []
    assert audit["opened_fixed_spatial_pattern_matches"] == []
    assert audit["t6_18_3_status"] == "OPENED_DEVELOPMENT_ONLY_NEVER_FORMAL"
    assert audit["opened_distance_three_reuse_is_allowed_only_with_new_factors"] is True


def test_spatial_patterns_are_explicit_balanced_permutations_and_hash_bound() -> None:
    report = _report()
    hashes = []
    for split in report["splits"]:
        for pattern in split["spatial_patterns"]:
            coordinates = 2 * pattern["distance"] * pattern["distance"]
            assert len(pattern["signs"]) == coordinates
            assert sum(pattern["signs"]) == 0
            assert sorted(pattern["permutation"]) == list(range(coordinates))
            payload = {key: value for key, value in pattern.items() if key != "pattern_sha256"}
            assert pattern["pattern_sha256"] == prereg._canonical_sha256(payload)
            hashes.append(pattern["pattern_sha256"])
    assert len(hashes) == len(set(hashes)) == 36


def test_every_registered_factor_is_cross_split_disjoint() -> None:
    disjoint = _report()["factor_disjointness"]
    expected = {
        "seeds", "base_sigmas", "spatial_pattern_keys", "spatial_pattern_sha256",
        "variance_law_ids", "covariance_designs", "transition_rates_per_1000_rounds",
        "amplitudes", "durations_rounds", "aux_noise",
    }
    assert set(disjoint) == expected
    assert all(row["passed"] and row["cross_split_overlap"] == [] for row in disjoint.values())


def test_execution_manifest_has_all_seed_family_cells_and_recomputable_hashes() -> None:
    report = _report()
    cells = report["execution_cells"]
    assert len(cells) == len({row["cell_id"] for row in cells}) == 1612
    per_split = {
        split_id: sum(row["split_id"] == split_id for row in cells)
        for split_id in prereg.SPLIT_IDS
    }
    assert per_split == {"train": 312, "calibration": 208, "pilot": 312, "formal": 780}
    for row in cells:
        payload = {key: value for key, value in row.items() if key != "cell_sha256"}
        assert row["cell_sha256"] == prereg._canonical_sha256(payload)
        assert row["scenario_family"] in prereg.EXPECTED_SCENARIOS


def test_derived_audits_cells_balance_and_power_are_independently_recomputed() -> None:
    report = _report()
    splits = report["splits"]
    assert report["opened_development_audit"] == prereg._opened_audit(report["config_snapshot"], splits)
    assert report["factor_disjointness"] == prereg._factor_disjointness(splits)
    assert report["execution_cells"] == prereg._execution_cells(
        splits, report["config_snapshot"]["scenario_families"]
    )
    assert report["formal_balance"] == prereg._formal_balance(report["execution_cells"])
    assert report["power_analysis"] == prereg._power_analysis(report["config_snapshot"])

    forged = deepcopy(report)
    _splits(forged)["formal"]["seeds"][0] = 61830001
    assert prereg.evaluate_gates(forged, check_live_files=False)[
        "G03_all_seed_namespaces_are_unique_and_opened_disjoint"
    ] is False

    forged_cell = deepcopy(report)
    forged_cell["execution_cells"][0]["distance"] = 7
    assert prereg.evaluate_gates(forged_cell, check_live_files=False)[
        "G07_execution_manifest_is_complete_unique_and_hash_bound"
    ] is False


def test_formal_design_balances_six_distance_sigma_strata_and_all_families() -> None:
    balance = _report()["formal_balance"]
    assert balance["all_strata_equal"] is True
    assert balance["all_families_have_all_formal_clusters"] is True
    assert len(balance["seed_stratum_counts"]) == 6
    assert set(balance["seed_stratum_counts"].values()) == {10}
    assert set(balance["clusters_per_scenario_family"]) == prereg.EXPECTED_SCENARIOS
    assert set(balance["clusters_per_scenario_family"].values()) == {60}


def test_power_recomputes_from_registered_assumptions_and_meets_90_percent() -> None:
    report = _report()
    plan = report["config_snapshot"]["power_plan"]
    power = report["power_analysis"]
    alpha = plan["familywise_alpha"] / plan["bonferroni_comparators"]
    zcrit = NormalDist().inv_cdf(1.0 - alpha)
    zpower = NormalDist().inv_cdf(plan["target_power"])
    required = math.ceil(
        ((zcrit + zpower) * plan["paired_cluster_difference_sd_ceiling"] / plan["target_absolute_difference"]) ** 2
    )
    achieved = NormalDist().cdf(
        math.sqrt(plan["formal_cluster_count"])
        * plan["target_absolute_difference"]
        / plan["paired_cluster_difference_sd_ceiling"]
        - zcrit
    )
    assert power["required_clusters"] == required == 48
    assert power["planned_clusters"] == 60
    assert power["achieved_power_at_registered_sd"] == pytest.approx(achieved)
    assert achieved >= 0.90
    assert plan["no_variance_based_resize_after_pilot"] is True


def test_compute_arithmetic_is_non_demo_and_exact() -> None:
    report = _report()
    compute = report["compute_arithmetic"]
    assert compute["formal_physical_rounds_per_method"] == 60 * 13 * 4096 == 3_194_880
    assert compute["formal_max_method_decodes"] == 3_194_880 * 12 == 38_338_560
    assert compute["all_split_physical_rounds_per_method"] == {
        "train": 638_976,
        "calibration": 638_976,
        "pilot": 1_277_952,
        "formal": 3_194_880,
    }
    caps = report["config_snapshot"]["compute_caps"]
    assert caps["formal_core_hours"] == 12_000
    assert caps["host_memory_gib"] == 64


def test_statistics_selection_tail_and_failure_contracts_are_fail_closed() -> None:
    report = _report()
    statistics = report["statistics"]
    selection = report["config_snapshot"]["selection_contract"]
    missingness = report["missingness"]
    stopping = report["stopping_rules"]
    assert statistics["paired_bootstrap_resamples"] == 50_000
    assert statistics["maximum_ranked_deployable_comparators"] == 12
    assert statistics["formal_sota_relative_ler_95_lcb_min_exclusive"] == 0.10
    assert len(statistics["tail_endpoints"]) == 4
    assert selection["pilot_selection_passes"] == 1
    assert selection["formal_reselection_prohibited"] is True
    assert missingness["missing_cell_fraction_max"] == 0.0
    assert missingness["zero_imputation_prohibited"] is True
    assert stopping["outcome_based_early_stop"] is False
    assert stopping["precision_or_significance_based_extension"] is False


def test_build_refuses_to_seal_when_any_registered_outcome_already_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    config = json.loads(prereg.CONFIG.read_text(encoding="utf-8"))
    config["future_outcome_paths"] = ["README.md"]
    tampered = ROOT / "runs" / "_test_t6_20_3_post_outcome_config.json"
    tampered.parent.mkdir(parents=True, exist_ok=True)
    try:
        tampered.write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setattr(prereg, "CONFIG", tampered)
        with pytest.raises(ValueError, match="cannot create preregistration after outcome access"):
            prereg.build_report()
    finally:
        tampered.unlink(missing_ok=True)


def test_source_data_is_lossless_for_every_split_cell_power_binding_and_absence_row() -> None:
    report = _report()
    with prereg.DEFAULT_SOURCE_DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == prereg._source_rows(report)
    assert len(rows) == report["source_data"]["rows"] == 1634
    assert sum(row["record_type"] == "execution_cell" for row in rows) == 1612
    for row in rows:
        assert row["canonical_sha256"] == hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
        assert json.loads(row["payload_json"])


def test_all_artifact_bindings_are_live_and_outcomes_were_absent_at_seal() -> None:
    report = _report()
    assert set(report["artifact_registry"]) == set(prereg.ARTIFACT_PATHS)
    for binding in report["artifact_registry"].values():
        path = ROOT / binding["path"]
        assert path.is_file()
        assert path.stat().st_size == binding["bytes"] > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    assert all(row["exists_at_seal"] is False for row in report["seal_absence_proof"])


@pytest.mark.parametrize(
    ("mutation", "gate"),
    [
        ("reuse_opened_seed", "G03_all_seed_namespaces_are_unique_and_opened_disjoint"),
        ("allow_resize", "G10_power_is_recomputed_and_fixed_n_meets_target"),
        ("delete_required_baseline", "G14_missingness_is_zero_tolerance_and_fail_closed"),
        ("outcome_extension", "G15_stopping_has_no_outcome_or_precision_extension"),
    ],
)
def test_high_risk_direct_mutations_fail_closed(mutation: str, gate: str) -> None:
    report = deepcopy(_report())
    if mutation == "reuse_opened_seed":
        report["opened_development_audit"]["seed_overlap"] = [61830001]
    elif mutation == "allow_resize":
        report["power_analysis"]["planned_clusters"] = 96
    elif mutation == "delete_required_baseline":
        report["config_snapshot"]["missingness"]["baseline_failure"] = "delete failed baseline"
    else:
        report["config_snapshot"]["stopping_rules"]["precision_or_significance_based_extension"] = True
    assert prereg.evaluate_gates(report, check_live_files=False)[gate] is False
    with pytest.raises(ValueError, match="verification failed"):
        prereg.verify_report(report)


def test_one_independent_semantic_mutation_targets_every_gate() -> None:
    report = _report()
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(report["gates"]) == 20
    assert {row["target_gate"] for row in audit["cases"]} == set(report["gates"])
    assert all(row["rejected"] for row in audit["cases"])


def test_human_report_exposes_power_missingness_and_no_posthoc_extension() -> None:
    text = prereg.DEFAULT_MARKDOWN.read_text(encoding="utf-8")
    assert "60 个 formal seed-cluster" in text
    assert "approximate power=0.9631" in text
    assert "pilot 后不扩样" in text
    assert "required baseline failure 关闭 SOTA" in text
    assert "affects_analysis=false" in text
