from __future__ import annotations

import copy
import json

import pytest

from cnn_fpga.benchmark import multimode_causal_headroom as subject


@pytest.fixture(scope="module")
def payloads():
    config = subject._load(subject.CONFIG)
    raw = subject._load(subject.RAW)
    manifest = subject._load(subject.ROOT / config["source_manifest"])
    return config, raw, manifest


def test_raw_ledger_is_complete_and_train_only(payloads):
    config, raw, manifest = payloads
    result = subject.validate_raw(raw, config, manifest)
    assert result["seed_count"] == 12
    assert result["family_count"] == 13
    assert result["cell_count"] == 156
    assert result["rounds"] == 79_872


def test_official_bsv_and_tjoin_qualification(payloads):
    _, raw, _ = payloads
    assert raw["correctness"]["official_bsv_action_mismatches"] == 0
    assert raw["correctness"]["pure_julia_vs_official_correction_mismatches"] == 0
    assert raw["correctness"]["maximum_official_log10_odds_error"] < 1e-9


def test_alias_normalization_and_coset_cardinality(payloads):
    _, raw, _ = payloads
    assert raw["correctness"]["alias_action_mismatches"] == 0
    assert raw["correctness"]["coset_cardinality_min"] == 16
    assert raw["correctness"]["coset_cardinality_max"] == 16
    assert raw["correctness"]["maximum_probability_normalization_error"] < 1e-12


def test_future_suffix_mutation_is_causal_and_nonvacuous(payloads):
    _, raw, _ = payloads
    audit = raw["causality"]
    assert audit["prefix_action_mismatches"] == 0
    assert audit["prefix_prior_max_abs_error"] == 0.0
    assert audit["mutated_suffix_action_divergence"]
    assert audit["mutated_suffix_prior_divergence"]


def test_decoder_source_denies_hidden_truth_signatures():
    audit = subject._validate_decoder_source()
    assert audit["hidden_truth_signature_hits"] == 0


def test_regret_path_telescopes():
    report = subject.build_report()
    rows = report["regret_decomposition"]
    total = sum(row["absolute_improvement"] for row in rows)
    methods = report["method_summary"]
    expected = methods[rows[0]["before_method"]]["p_L"] - methods[rows[-1]["after_method"]]["p_L"]
    assert total == pytest.approx(expected, abs=1e-15)
    assert [row["component"] for row in rows] == [
        "estimator",
        "metric_likelihood",
        "logical_coset_sum",
        "posterior_marginalization",
        "risk_action",
    ]


def test_strongest_baseline_is_selected_without_deletion():
    report = subject.build_report()
    selection = report["strongest_development_baseline_selection"]
    assert set(selection["candidates_retained"]) == {
        "current_adaptive_weighted_cpd",
        "static_mixture_exact_mld",
    }
    assert selection["selected"] == min(selection["candidate_p_L"], key=selection["candidate_p_L"].get)


def test_headroom_fails_both_frozen_thresholds():
    report = subject.build_report()
    assert report["paired_bootstrap"]["relative_improvement_point"] < 0.15
    assert report["paired_bootstrap"]["relative_improvement_lcb"] < 0.12
    assert report["verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM"


def test_ceiling_and_integrity_repair_boundaries_are_explicit():
    report = subject.build_report()
    assert "not the supremum" in report["scope"]["ceiling_boundary"]
    ledger = report["integrity_repair_ledger"]
    assert ledger["invalidated_outcomes_not_used"] is True
    assert ledger["registered_seeds_families_rounds_baselines_and_gates_changed"] is False
    assert ledger["performance_threshold_tuned_after_outcomes"] is False
    assert len(ledger["repairs"]) == 3


def test_risk_intervention_restores_but_does_not_beat_trusted_baseline():
    report = subject.build_report()
    assert report["risk_action_diagnostics"]["interventions"] > 0
    assert report["risk_action_diagnostics"]["net_errors_reduced_vs_unprotected_posterior_predictive"] > 0
    action = report["regret_decomposition"][-1]
    assert action["component"] == "risk_action"
    assert action["absolute_improvement"] > 0.0
    bootstrap = report["paired_bootstrap"]
    assert bootstrap["baseline_p_L"] == bootstrap["proposed_p_L"]
    assert bootstrap["relative_improvement_point"] == 0.0


@pytest.mark.parametrize(
    ("name", "mutator"),
    [
        ("split", lambda raw: raw.__setitem__("source_split", "pilot")),
        ("missing", lambda raw: raw["rows"].pop()),
        ("duplicate", lambda raw: raw["rows"].__setitem__(-1, copy.deepcopy(raw["rows"][0]))),
        ("method", lambda raw: raw["rows"][0]["errors"].pop("current_adaptive_weighted_cpd")),
        ("trace", lambda raw: raw["rows"][0].__setitem__("physical_trace_sha256", "demo")),
        ("formal", lambda raw: raw["rows"][0].__setitem__("formal", True)),
    ],
)
def test_integrity_mutations_fail_closed(payloads, name, mutator):
    config, raw, manifest = payloads
    candidate = copy.deepcopy(raw)
    mutator(candidate)
    with pytest.raises((subject.IntegrityError, KeyError, TypeError, ValueError), match=".+"):
        subject.validate_raw(candidate, config, manifest, check_file_hashes=False)


def test_all_semantic_mutations_are_caught(payloads):
    config, raw, manifest = payloads
    mutations = subject._semantic_mutations(raw, config, manifest)
    assert len(mutations) >= 15
    assert all(row["caught"] for row in mutations)


def test_report_recomputes_exactly_after_generation():
    result = subject.verify()
    assert result["verdict"] == "NO_GO_MULTIMODE_CAUSAL_HEADROOM"
    assert result["gates"]["integrity_passed"] is True
    assert result["gates"]["headroom_passed"] is False
