from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import tempfile
from typing import Any, Callable

import pytest

from cnn_fpga.benchmark import (
    phase9_baseline_search_power_registry as subject,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase9/t9_1_4_baseline_registry.json"
REPORT = ROOT / "docs/t9_1_4_baseline_search_power_registry.json"
SOURCE = (
    ROOT / "docs/t9_1_4_baseline_search_power_registry_source_data.csv"
)
LITERATURE = ROOT / "docs/t9_1_4_literature_search_ledger.csv"
MARKDOWN = ROOT / "docs/phase9_baseline_search_power_registry.md"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def config() -> dict[str, Any]:
    return _load(CONFIG)


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    return _load(REPORT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _method(
    report: dict[str, Any], method_id: str
) -> dict[str, Any]:
    return next(
        row for row in report["methods"] if row["method_id"] == method_id
    )


def _literature_record(
    report: dict[str, Any], record_id: str
) -> dict[str, Any]:
    return next(
        row
        for row in report["literature_records"]
        if row["record_id"] == record_id
    )


def _offline_gates(
    candidate: dict[str, Any],
    published: dict[str, Any],
) -> dict[str, bool]:
    """Evaluate semantics without repeatedly re-reading live parent evidence."""

    return subject.evaluate_gates(
        candidate,
        config_path=CONFIG,
        check_live_files=False,
        expected_parent_handoff=published["parent_handoff"],
        expected_artifact_registry=published["artifact_registry"],
        expected_implementation_bindings=published[
            "implementation_bindings"
        ],
    )


def test_published_report_live_verifies_all_36_gates(
    report: dict[str, Any],
) -> None:
    checks = subject.verify_report(report)
    assert all(checks.values())
    assert set(checks) == {
        "identity",
        "parent_handoff_live",
        "mutation_replay",
        "all_gates",
        "gate_cache",
        "verdict",
        "analysis_sha256",
        "source_data",
        "literature_ledger",
        "performance_null",
    }
    assert tuple(report["gates"]) == subject.GATE_IDS
    assert report["gate_summary"] == {
        "passed": 36,
        "total": 36,
        "failed": [],
    }
    assert report["verdict"] == subject.VERDICT


def test_registry_is_exact_config_projection_with_expected_cardinality(
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    expected_methods = subject._enrich_methods(
        config, report["parent_handoff"]
    )
    expected_literature, expected_dedup = (
        subject._deduplicate_literature_records(config)
    )
    assert report["methods"] == expected_methods
    assert report["literature_records"] == expected_literature
    assert report["literature_dedup_audit"] == expected_dedup

    classes = Counter(row["registry_class"] for row in report["methods"])
    assert len(report["methods"]) == 31
    assert classes == {
        "MATCHED_DEPLOYABLE_RANKED": 19,
        "UNBUDGETED_CAPACITY_CEILING_NONRANKING": 5,
        "PRIVILEGED_UPPER_BOUND_NONRANKING": 4,
        "PROTOCOL_ANCHOR_NONRANKING": 3,
    }
    mandatory = tuple(
        row["method_id"]
        for row in report["methods"]
        if row["mandatory"] is True
    )
    assert mandatory == subject.MANDATORY_BASELINE_IDS
    assert len(mandatory) == 18
    assert _method(report, subject.PROPOSED_ID)["mandatory"] is False
    assert (
        _method(report, subject.PROPOSED_ID)["registry_class"]
        == "MATCHED_DEPLOYABLE_RANKED"
    )


def test_sbs_protocols_and_ranking_lanes_are_not_conflated(
    report: dict[str, Any],
) -> None:
    measurement = _method(report, "standard_measurement_feedback_sbs")
    autonomous = _method(report, "autonomous_sbs_no_measurement")
    idle = _method(report, "no_correction_idle_memory")
    assert len(
        {
            measurement["method_id"],
            autonomous["method_id"],
            idle["method_id"],
        }
    ) == 3
    assert measurement["registry_class"] == "MATCHED_DEPLOYABLE_RANKED"
    assert measurement["mandatory"] is True
    assert autonomous["registry_class"] == "PROTOCOL_ANCHOR_NONRANKING"
    assert idle["registry_class"] == "PROTOCOL_ANCHOR_NONRANKING"
    assert autonomous["role"] == "NO_MEASUREMENT_PROTOCOL_ANCHOR"
    assert idle["role"] == "ZERO_ACTION_BREAK_EVEN_ANCHOR"
    assert autonomous["claim_boundary"] != idle["claim_boundary"]

    ranked = [
        lane["lane_id"]
        for lane in report["ranking_lanes"]
        if lane["ranked"] is True
    ]
    assert ranked == ["MATCHED_DEPLOYABLE_RANKED"]
    assert tuple(
        lane["lane_id"] for lane in report["ranking_lanes"]
    ) == subject.RANKING_CLASS_IDS


def test_observation_privilege_and_common_contract_are_nontransferable(
    report: dict[str, Any],
) -> None:
    expected_contracts = {
        "task_signature": subject.PROTOCOL_ID,
        "compute": "PHASE9_MATCHED_COMPUTE_V1",
        "selection": "PHASE9_ALL_ATTEMPTS_SELECTION_V1",
        "missingness": "PHASE9_FAILURE_RETENTION_V1",
    }
    for row in report["methods"]:
        assert row["common_contract_ids"] == expected_contracts
        privilege = row["privilege"]
        if row["registry_class"] in {
            "MATCHED_DEPLOYABLE_RANKED",
            "UNBUDGETED_CAPACITY_CEILING_NONRANKING",
        }:
            assert privilege["observed_only"] is True
            assert privilege["hidden_truth_access"] is False
            assert privilege["future_suffix_access"] is False
            assert privilege["scenario_id_access"] is False
            assert privilege["formal_label_access"] is False
            assert (
                privilege["deployable_teacher_call_during_formal"] is False
            )
            assert (
                privilege["privileged_diagnostic_evaluation_on_formal"]
                is False
            )
            assert privilege["assumed_model_branching_access"] is False

    oracle = _method(report, "hidden_state_decoder_oracle")
    teacher = _method(report, "hidden_state_teacher")
    smoother = _method(report, "future_suffix_smoother")
    control_oracle = _method(report, "finite_horizon_control_oracle")
    assert oracle["privilege"]["hidden_truth_access"] is True
    assert teacher["privilege"]["hidden_truth_access"] is True
    assert smoother["privilege"]["future_suffix_access"] is True
    assert (
        control_oracle["privilege"]["assumed_model_branching_access"] is True
    )
    assert all(
        row["privilege"]["deployable"] is False
        and row["current_result"]["rank"] is None
        for row in (oracle, teacher, smoother, control_oracle)
    )


def test_every_current_metric_rank_accounting_and_claim_is_typed_null(
    report: dict[str, Any],
) -> None:
    expected_current = {
        "numeric_metrics": None,
        "rank": None,
        "matched_eligible": False,
        "ceiling_eligible": False,
        "registered_best_vote": False,
        "external_sota_vote": False,
    }
    assert len(subject.ONLINE_ACCOUNTING_FIELDS) == 24
    for row in report["methods"]:
        assert row["current_result"] == expected_current
        assert tuple(row["current_accounting"]) == (
            subject.ONLINE_ACCOUNTING_FIELDS
        )
        assert all(
            value is None for value in row["current_accounting"].values()
        )

    for row in report["literature_records"]:
        assert row["numeric_result"] is None
        assert row["external_rank"] is None
        assert row["external_sota_vote"] is False

    assert report["performance_state"] == {
        "round_ler": None,
        "six_state_lifetime": None,
        "registered_best": None,
        "external_sota": None,
        "protocol_only": True,
    }
    for key in (
        "registered_best",
        "external_sota",
        "official_puviani_exact",
        "puviani_surpass",
        "physical_lifetime",
    ):
        assert report["external_claim_contract"][key] is None

    provenance = _method(report, "paper_constrained_nmf")[
        "t9_1_3_provenance"
    ]
    assert provenance["matched_phase9_ranking_eligible"] is False
    assert provenance["official_exact"] is None
    assert provenance["puviani_surpass"] is None
    assert provenance["paper_scale_lifetime"] is None


def test_compute_selection_and_failure_contract_is_complete(
    report: dict[str, Any],
) -> None:
    compute = report["matched_compute_contract"]
    assert compute["batch_size"] == 1
    assert compute["update_cadence_cycles"] == 256
    assert compute["total_online_deadline_ns"] == 1_000_000
    assert compute["cpu_profile"] == "ONE_PINNED_PHYSICAL_CORE"
    assert compute["threads"] == 1
    assert compute["gpu_allowed_online"] is False
    assert compute["deadline_includes"] == [
        "IQ_or_LLR_preprocessing",
        "feature_extraction",
        "model_inference",
        "state_update",
        "serialization",
        "host_to_bank_transfer",
    ]
    assert compute["max_trainable_parameters"] == 131_072
    assert compute["max_persistent_state_bytes"] == 1_048_576
    assert compute["max_transient_workspace_bytes"] == 16_777_216
    assert compute["max_peak_rss_delta_bytes"] == 268_435_456
    assert compute["max_total_online_macs_per_update"] == 262_144
    assert compute["max_total_online_flops_per_update"] == 1_048_576
    assert compute["precision_profile_payload"] is None
    assert compute["eligibility_if_precision_or_codebook_null"] is False

    selection = report["selection_contract"]
    fields = selection["all_attempts_required_fields"]
    assert len(fields) == len(set(fields)) == 16
    assert {
        "family_id",
        "candidate_id",
        "restart_id",
        "seed",
        "hyperparameters",
        "config_sha256",
        "sample_manifest_sha256",
        "environment_sha256",
        "checkpoint_sha256",
        "validation_metrics",
        "failure_status",
        "selected",
        "selection_reason",
    } <= set(fields)
    assert selection["family_selection_evaluation_cap"] == 64
    assert selection["learning_hyperparameter_config_cap"] == 16
    assert selection["learning_restarts_per_config_cap"] == 4
    assert selection["formal_access_for_selection"] == "PROHIBITED"
    assert (
        selection["best_of_n_without_complete_ledger"]
        == "INELIGIBLE_BEST_OF_N_HIDDEN"
    )
    assert "all_failure_counts" in selection["report_distribution"]

    missing = report["missingness_contract"]
    assert set(missing["status_enum"]) == {
        "COMPLETE",
        "DEADLINE_FALLBACK",
        "TIMEOUT",
        "OOM",
        "NUMERIC_FAILURE",
        "CORRUPT_SHARD",
        "MISSING_TOOL",
        "EXTERNAL_UNAVAILABLE",
        "SIGNATURE_INELIGIBLE",
        "TYPED_NULL_NO_GO",
    }
    assert missing["complete_case_deletion"] == "PROHIBITED"
    assert missing["zero_imputation"] == "PROHIBITED"
    assert missing["retry_count"] == 1


def test_power_is_independently_recomputed_for_claim_margin(
    report: dict[str, Any],
) -> None:
    plan = report["formal_power_plan"]
    power = report["power_analysis"]
    alpha_per = 0.05 / 18
    z_critical = NormalDist().inv_cdf(1.0 - alpha_per)
    z_power = NormalDist().inv_cdf(0.90)
    margin = 0.08 * (0.15 - 0.10)
    required = math.ceil(
        ((z_critical + z_power) * 0.028 / margin) ** 2
    )
    achieved = NormalDist().cdf(
        math.sqrt(808) * margin / 0.028 - z_critical
    )

    assert plan["fixed_comparator_divisor"] == 18
    assert plan["alternative_relative_effect"] == pytest.approx(0.15)
    assert plan["claim_relative_lcb_threshold"] == pytest.approx(0.10)
    assert plan["alternative_relative_effect"] > plan[
        "claim_relative_lcb_threshold"
    ]
    assert alpha_per == pytest.approx(0.002777777777777778)
    assert z_critical == pytest.approx(2.772921294608662)
    assert margin == pytest.approx(0.004)
    assert required == 806
    assert power["alpha_per_comparator"] == pytest.approx(alpha_per)
    assert power["one_sided_z_critical"] == pytest.approx(z_critical)
    assert power["z_target_power"] == pytest.approx(z_power)
    assert power["absolute_claim_margin_at_design_floor"] == pytest.approx(
        margin
    )
    assert power["required_clusters_at_design_floor"] == required
    assert power["planned_clusters_per_backend"] == 808
    assert power["achieved_power_at_design_floor"] == pytest.approx(achieved)
    assert achieved == pytest.approx(0.9010994595676306)
    assert achieved >= 0.90
    assert plan["macro_scenario_families"] == 8
    assert plan["clusters_per_macro_family_per_backend"] == 101
    assert (
        plan["macro_scenario_families"]
        * plan["clusters_per_macro_family_per_backend"]
        == plan["formal_cluster_count_per_backend"]
        == 808
    )
    assert (
        plan["primary_powered_endpoint"]
        == "ROUND_LER_PROMOTION_VS_EACH_MANDATORY_MATCHED_BASELINE"
    )
    assert plan["lifetime_power_status"].startswith("TYPED_NULL_")
    assert (
        plan["joint_family_power_status"]
        == "NO_90PCT_JOINT_POWER_CLAIM_DESIGN_CONTROLS_LER_COMPARATOR_SCREEN_ONLY"
    )

    sensitivity = {
        row["baseline_p_l"]: row for row in power["sensitivity"]
    }
    expected_required = {0.08: 806, 0.10: 516, 0.16: 202}
    for baseline, expected_n in expected_required.items():
        candidate_margin = baseline * (0.15 - 0.10)
        independent_n = math.ceil(
            (
                (z_critical + z_power)
                * 0.028
                / candidate_margin
            )
            ** 2
        )
        independent_power = NormalDist().cdf(
            math.sqrt(808) * candidate_margin / 0.028 - z_critical
        )
        assert independent_n == expected_n
        assert sensitivity[baseline]["required_clusters"] == expected_n
        assert sensitivity[baseline][
            "achieved_power_at_planned_n"
        ] == pytest.approx(independent_power)


def test_cluster_bootstrap_contract_prevents_pseudoreplication(
    report: dict[str, Any],
) -> None:
    statistics = report["statistics_contract"]
    assert statistics["method"] == "paired_cluster_maxT_bootstrap"
    assert statistics["resamples"] == 100_000
    assert statistics["bootstrap_seed"] == 91420260725
    assert (
        statistics["resampling_unit"]
        == "device_cell_id_x_scenario_realization_id_x_trajectory_seed_outer_block"
    )
    assert {
        "all_six_logical_states",
        "all_methods",
        "all_rounds_and_windows",
        "all_reset_leakage_control_fallback_timeout_events",
        "same_initial_state_device_draw_exogenous_innovation_manifest_and_action_opportunities",
        "policy_specific_action_conditioned_transition_and_trace_hashes",
    } == set(statistics["cluster_contains"])
    assert "outer block" in statistics["shared_device_draw_outer_block"]
    assert "averaging or rescue" in statistics["backend_rule"]
    assert len(statistics["closed_families"]) == 3
    assert "recompute baseline and proposed denominator" in statistics[
        "ratio_rule"
    ]
    assert "null/undefined" in statistics["zero_baseline_ler_rule"]
    assert "epsilon substitution prohibited" in statistics[
        "zero_baseline_ler_rule"
    ]
    assert (
        statistics["round_or_cycle_as_independent_sample"] == "PROHIBITED"
    )
    assert (
        statistics["posthoc_family_split_or_endpoint_drop"] == "PROHIBITED"
    )


def test_identifier_normalisation_and_similarity_rules() -> None:
    doi = "10.1103/physrevlett.134.020601"
    assert subject._normalised_doi(f" DOI:{doi.upper()} ") == doi
    assert subject._normalised_doi(f"https://doi.org/{doi}") == doi
    assert subject._normalised_doi(f"http://doi.org/{doi}") == doi
    assert subject._normalised_doi(None) is None
    assert (
        subject._normalised_author("Lachance-Quirion")
        == "lachancequirion"
    )
    assert subject._title_jaccard(
        "Non-Markovian feedback for optimized quantum error correction",
        "Non Markovian Feedback for Optimized Quantum Error Correction",
    ) == pytest.approx(1.0)

    left = {
        "doi": f"DOI:{doi}",
        "arxiv_id": None,
        "title": "A title",
        "first_author": "Puviani",
    }
    right = {
        "doi": f"https://doi.org/{doi.upper()}",
        "arxiv_id": None,
        "title": "A title version",
        "first_author": "Puviani",
    }
    assert subject._same_work(left, right, title_threshold=0.9) is True
    assert (
        subject._same_work_reason(left, right, title_threshold=0.9)
        == "DOI_EXACT"
    )


def test_actual_dedup_pipeline_merges_doi_and_title_author_versions(
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    audit = report["literature_dedup_audit"]
    assert audit["pipeline_executed"] is True
    assert audit["input_bibliographic_hit_count"] == 23
    assert audit["artifact_evidence_count"] == 4
    assert audit["canonical_work_count"] == 12
    assert audit["merged_bibliographic_version_count"] == 11
    assert audit["unique_dedup_key_count"] == 12
    assert len(audit["raw_hit_ledger"]) == 23
    assert len(audit["merge_events"]) == 11
    assert len(
        {row["source_record"] for row in audit["raw_hit_ledger"]}
    ) == 23
    assert {
        row["source_record"] for row in audit["merge_events"]
    } <= {
        row["source_record"] for row in audit["raw_hit_ledger"]
    }
    assert all(
        row["canonical_record_id"]
        in {record["record_id"] for record in report["literature_records"]}
        and row["rule"]
        in {"DOI_EXACT", "ARXIV_EXACT", "TITLE_AUTHOR_JACCARD"}
        for row in audit["merge_events"]
    )
    assert audit["merge_rule_counts"] == {
        "DOI_EXACT": 0,
        "ARXIV_EXACT": 0,
        "TITLE_AUTHOR_JACCARD": 11,
    }

    base = deepcopy(config["literature_records"][0])
    base["raw_records"] = [f"DOI:{base['doi']}"]
    base["artifact_records"] = []
    version = deepcopy(base)
    version["record_id"] = "LIT-DOI-VERSION"
    version["raw_records"] = [
        f"DOI:HTTPS://DOI.ORG/{str(base['doi']).upper()}"
    ]
    mini = deepcopy(config)
    mini["literature_records"] = [base, version]
    rows, mini_audit = subject._deduplicate_literature_records(mini)
    assert len(rows) == 1
    assert set(rows[0]["raw_records"]) == {
        base["raw_records"][0],
        version["raw_records"][0],
    }
    assert mini_audit["input_bibliographic_hit_count"] == 2
    assert mini_audit["canonical_work_count"] == 1
    assert mini_audit["merged_bibliographic_version_count"] == 1
    assert mini_audit["merge_rule_counts"]["DOI_EXACT"] == 1
    assert len(mini_audit["raw_hit_ledger"]) == 2
    assert mini_audit["merge_events"] == [
        {
            "source_record": version["raw_records"][0],
            "canonical_record_id": base["record_id"],
            "rule": "DOI_EXACT",
        }
    ]

    title_version = deepcopy(config)
    title_version["literature_records"] = [
        deepcopy(config["literature_records"][0])
    ]
    rows, title_audit = subject._deduplicate_literature_records(
        title_version
    )
    assert len(rows) == 1
    assert rows[0]["raw_record_count"] == 2
    assert title_audit["merge_rule_counts"]["TITLE_AUTHOR_JACCARD"] == 1


def test_dedup_conflicts_and_artifact_misclassification_fail_closed(
    config: dict[str, Any],
) -> None:
    base = deepcopy(config["literature_records"][0])
    base["artifact_records"] = []
    base["raw_records"] = ["DOI:10.1000/same"]

    conflicting_screen = deepcopy(base)
    conflicting_screen["record_id"] = "LIT-CONFLICT-SCREEN"
    conflicting_screen["raw_records"] = [
        "DOI:https://doi.org/10.1000/same"
    ]
    conflicting_screen["typed_null_reasons"] = [
        *conflicting_screen["typed_null_reasons"],
        "CONTRADICTORY_SCREEN",
    ]
    candidate = deepcopy(config)
    candidate["literature_records"] = [base, conflicting_screen]
    with pytest.raises(
        ValueError, match="conflicting same-task screening"
    ):
        subject._deduplicate_literature_records(candidate)

    conflicting_id = deepcopy(base)
    conflicting_id["record_id"] = "LIT-CONFLICT-ID"
    conflicting_id["raw_records"] = ["DOI:10.1000/different"]
    candidate = deepcopy(config)
    candidate["literature_records"] = [base, conflicting_id]
    with pytest.raises(
        ValueError, match="conflicting non-null identifiers"
    ):
        subject._deduplicate_literature_records(candidate)

    artifact_as_bibliography = deepcopy(base)
    artifact_as_bibliography["raw_records"] = ["GITHUB:owner/repository"]
    candidate = deepcopy(config)
    candidate["literature_records"] = [artifact_as_bibliography]
    with pytest.raises(ValueError, match="move artifact evidence"):
        subject._deduplicate_literature_records(candidate)


def test_literature_screening_and_puviani_boundary_are_complete(
    config: dict[str, Any],
    report: dict[str, Any],
) -> None:
    assert len(report["literature_records"]) == 12
    query_ids = {
        row["query_id"]
        for row in report["literature_search"]["queries"]
    }
    signature_fields = {
        "input",
        "history",
        "action",
        "physics",
        "online_timing",
        "postselection",
        "denominator",
        "metric",
        "compute",
    }
    for row in report["literature_records"]:
        assert set(row["query_ids"]) <= query_ids
        assert row["query_ids"]
        assert set(row["signature_checks"]) == signature_fields
        assert row["eligibility_class"]
        assert row["exclusion_code"]
        assert row["typed_null_reasons"]
        assert row["external_same_task_eligible"] is False
        assert all(
            value.startswith(("DOI:", "ARXIV:"))
            for value in row["raw_records"]
        )
    assert (
        report["external_claim_contract"][
            "current_external_same_task_eligible_count"
        ]
        == 0
    )
    for source in report["literature_search"]["source_status"]:
        if source["status"] == "SOURCE_UNAVAILABLE":
            assert source["raw_result_count"] is None
            assert source["count_reason"]

    puviani = _literature_record(report, "LIT-PUVIANI-2025")
    assert set(puviani["raw_records"]) == {
        "ARXIV:2312.07391",
        "DOI:10.1103/PhysRevLett.134.020601",
    }
    assert puviani["artifact_records"] == [
        "GITHUB:Matteo-Puviani/GQF"
    ]
    assert {
        "MISSING_OFFICIAL_CHECKPOINT",
        "MISSING_AGENT_SEED_LEDGER",
        "MISSING_SELECTION_LEDGER",
        "MISSING_SIX_STATE_EVALUATOR",
    } <= set(puviani["typed_null_reasons"])
    assert report["external_claim_contract"]["official_puviani_exact"] is None
    assert report["external_claim_contract"]["puviani_surpass"] is None

    raw_sources = [
        source
        for row in config["literature_records"]
        for source in row["raw_records"]
    ]
    artifact_sources = [
        source
        for row in config["literature_records"]
        for source in row["artifact_records"]
    ]
    assert report["literature_dedup_audit"][
        "bibliographic_source_ids_sha256"
    ] == subject._canonical_sha256(sorted(raw_sources))
    assert report["literature_dedup_audit"][
        "artifact_evidence_ids_sha256"
    ] == subject._canonical_sha256(sorted(artifact_sources))


def test_source_csv_is_canonical_lossless_and_hash_complete(
    report: dict[str, Any],
) -> None:
    with SOURCE.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == subject._source_rows(report)
    assert len(rows) == report["source_data"]["rows"] == 95
    assert len(
        {(row["record_type"], row["record_id"]) for row in rows}
    ) == 95
    assert Counter(row["record_type"] for row in rows) == {
        "literature_dedup_audit": 1,
        "ranking_lane": 4,
        "method": 31,
        "search_query": 6,
        "search_source": 5,
        "literature_record": 12,
        "power_sensitivity": 3,
        "parent_terminal_option": 2,
        "future_outcome_absence": 5,
        "artifact_binding": 5,
        "implementation_binding": 21,
    }
    for row in rows:
        assert (
            hashlib.sha256(row["payload_json"].encode("utf-8")).hexdigest()
            == row["canonical_sha256"]
        )
        payload = json.loads(row["payload_json"])
        assert subject._canonical_json(payload) == row["payload_json"]
    decoded = {
        (row["record_type"], row["record_id"]): json.loads(
            row["payload_json"]
        )
        for row in rows
    }
    assert decoded[
        ("literature_dedup_audit", "registered-dedup-pipeline")
    ] == report["literature_dedup_audit"]
    assert decoded[("method", "causal_transformer_posterior")] == _method(
        report, "causal_transformer_posterior"
    )


def test_literature_csv_is_exact_and_each_row_hash_recomputes(
    report: dict[str, Any],
) -> None:
    with LITERATURE.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert tuple(reader.fieldnames or ()) == subject.LITERATURE_LEDGER_FIELDS
    typed_rows = {
        row["record_id"]: row
        for row in subject._literature_ledger_rows(report)
    }
    assert len(rows) == report["literature_ledger"]["rows"] == 12
    assert {row["record_id"] for row in rows} == set(typed_rows)
    for row in rows:
        typed = typed_rows[row["record_id"]]
        payload = {
            key: typed[key]
            for key in subject.LITERATURE_LEDGER_FIELDS
            if key != "canonical_sha256"
        }
        assert typed["canonical_sha256"] == subject._canonical_sha256(
            payload
        )
        expected_strings = {
            key: (
                ""
                if value is None
                else "True"
                if value is True
                else "False"
                if value is False
                else str(value)
            )
            for key, value in typed.items()
        }
        assert row == expected_strings


def test_markdown_is_exact_renderer_output_with_all_boundaries(
    report: dict[str, Any],
) -> None:
    text = MARKDOWN.read_text(encoding="utf-8")
    assert text == subject._render_markdown(report)
    assert all(
        f"`{row['method_id']}`" in text for row in report["methods"]
    )
    assert all(
        f"`{row['record_id']}`" in text
        for row in report["literature_records"]
    )
    assert "required N=806" in text
    assert "planned N=808" in text
    assert "100,000" in text
    assert "external-SOTA" in text
    assert "official Puviani" in text
    assert "`null`" in text
    assert (
        report["external_claim_contract"]["safe_future_wording"] in text
    )
    assert "这只是 `SEALED_PRE_OUTCOME` 协议，不是性能实验结果" in text


def test_all_bindings_are_strict_live_contained_and_fragment_qualified(
    report: dict[str, Any],
) -> None:
    assert set(report["artifact_registry"]) == {
        "config",
        "implementation",
        "phase9_parent",
        "t9_1_4_input_contract",
        "historical_power_proxy",
    }
    assert len(report["implementation_bindings"]) == 21
    bindings = [
        *report["artifact_registry"].values(),
        *report["implementation_bindings"],
        report["source_data"],
        report["literature_ledger"],
        report["markdown"],
    ]
    for binding in bindings:
        relative = Path(binding["path"])
        assert relative.is_absolute() is False
        assert ".." not in relative.parts
        path = ROOT / relative
        assert path.is_file()
        assert binding["selector"] == "STRICT_FILE_SHA256"
        assert binding["bytes"] == path.stat().st_size
        assert binding["sha256"] == _sha256(path)
        assert subject._binding_live(binding)

    for binding in report["implementation_bindings"]:
        path = ROOT / binding["path"]
        text = path.read_text(encoding="utf-8", errors="replace").casefold()
        assert all(
            fragment.casefold() in text
            for fragment in binding["expected_fragments"]
        )


def test_parent_handoff_is_live_reconstructed_and_both_branches_release(
    report: dict[str, Any],
) -> None:
    live = subject._consume_parent_handoff()
    assert report["parent_handoff"] == live
    assert all(live["ordered_consumer_checks"].values())
    seal = live["downstream_semantic_seal"]
    assert seal["semantic_sha256"] == subject._canonical_sha256(
        seal["payload"]
    )
    assert live["raw_addendum_hash_consumed"] is False
    assert live["releases_t9_1_4"] is True
    assert live["matched_phase9_ranking_eligible"] is False
    assert live["sota_claim_eligible"] is False

    qualified, no_go = report["parent_terminal_options"]
    assert qualified["terminal_state"] == (
        "QUALIFIED_PAPER_CONSTRAINED_BASELINE"
    )
    assert no_go["terminal_state"] == (
        "NO_GO_PAPER_CONSTRAINED_REIMPLEMENTATION"
    )
    for branch in (qualified, no_go):
        assert branch["releases_t9_1_4"] is True
        assert branch["matched_phase9_ranking_eligible"] is False
        assert branch["typed_payload"]["official_exact"] is None
        assert branch["typed_payload"]["puviani_surpass"] is None
        assert branch["typed_payload"]["paper_scale_lifetime"] is None
    assert no_go["typed_payload"]["numeric_metrics"] is None
    assert no_go["typed_payload"]["rank"] is None


EXPECTED_MUTATION_IDS = {
    "change_protocol_identity",
    "forge_parent_semantic_seal",
    "delete_no_go_parent_release",
    "fill_puviani_surpass",
    "rank_hil_with_algorithm",
    "delete_mandatory_imm",
    "merge_autonomous_into_no_correction",
    "rank_capacity_ceiling",
    "promote_unqualified_method",
    "give_gru_hidden_truth",
    "grant_transformer_extra_iq_stratum",
    "fill_unqualified_codebook",
    "allow_online_gpu",
    "exclude_transfer_from_deadline",
    "hide_selection_attempt_fields",
    "select_checkpoint_on_formal",
    "open_pilot_twice",
    "weaken_lcb_claim_threshold",
    "reuse_old_sixty_cluster_design",
    "resample_rounds_as_independent",
    "replace_maxT_with_pointwise_ci",
    "epsilon_zero_denominator",
    "drop_timeout_status",
    "ceiling_votes_registered_best",
    "oracle_enters_rank",
    "erase_search_cutoff",
    "duplicate_puviani_as_second_comparator",
    "promote_external_without_same_task",
    "erase_external_null_reason",
    "auto_promote_registry_best",
    "claim_puviani_official_assets_complete",
    "pretend_formal_outcome_existed_at_seal",
    "forge_implementation_binding",
    "forge_source_row_count",
    "forge_mutation_count",
    "insert_demo_ler_result",
}


def test_builtin_mutation_audit_covers_and_replays_all_36_gates(
    report: dict[str, Any],
) -> None:
    audit = report["semantic_mutation_audit"]
    assert audit["count"] == audit["detected"] == len(
        subject.GATE_IDS
    ) == 36
    assert audit["all_detected"] is True
    assert len(audit["cases"]) == 36
    assert {
        row["mutation_id"] for row in audit["cases"]
    } == EXPECTED_MUTATION_IDS
    assert {
        row["target_gate"] for row in audit["cases"]
    } == set(subject.GATE_IDS)
    assert len(
        {row["mutation_id"] for row in audit["cases"]}
    ) == len(audit["cases"])
    assert all(
        row["mutation_applied"] is True
        and row["rejection_mode"] == "TARGET_GATE_FALSE"
        and row["rejected"] is True
        for row in audit["cases"]
    )


Mutation = Callable[[dict[str, Any]], None]


def _remove_method(method_id: str) -> Mutation:
    def mutate(value: dict[str, Any]) -> None:
        value["methods"].remove(_method(value, method_id))

    return mutate


DIRECT_MUTATIONS: list[tuple[str, str, Mutation]] = [
    (
        "frozen_timestamp",
        "G01_identity_and_preoutcome_seal",
        lambda value: value.update(frozen_at="2099-01-01T00:00:00+00:00"),
    ),
    (
        "parent_seal",
        "G02_parent_addendum_live_validated_and_semantic_seal_reconstructed",
        lambda value: value["parent_handoff"][
            "downstream_semantic_seal"
        ].update(semantic_sha256="0" * 64),
    ),
    (
        "official_claim",
        "G04_official_external_physical_and_performance_claims_are_null",
        lambda value: value["external_claim_contract"].update(
            external_sota={"rank": 1}
        ),
    ),
    (
        "remove_cnn",
        "G06_exact_mandatory_baselines_and_anchors_are_present",
        _remove_method("cnn_posterior"),
    ),
    (
        "inject_rank",
        "G09_no_method_is_currently_ranked_before_common_harness_qualification",
        lambda value: _method(value, "paper_constrained_nmf")[
            "current_result"
        ].update(rank=1, matched_eligible=True),
    ),
    (
        "hidden_truth",
        "G10_deployable_and_ceiling_rows_are_observed_only",
        lambda value: _method(value, "causal_transformer_posterior")[
            "privilege"
        ].update(hidden_truth_access=True),
    ),
    (
        "delete_accounting",
        "G13_compute_precision_and_accounting_schema_are_complete_and_fail_closed",
        lambda value: _method(value, "gru_posterior")[
            "current_accounting"
        ].pop("timeout_count"),
    ),
    (
        "hide_environment_hash",
        "G15_selection_budget_and_all_attempt_ledger_are_complete",
        lambda value: value["selection_contract"][
            "all_attempts_required_fields"
        ].remove("environment_sha256"),
    ),
    (
        "reopen_pilot",
        "G17_pilot_is_single_pass_balanced_nonclaiming_and_keeps_all_baselines",
        lambda value: value["pilot_contract"].update(open_count=2),
    ),
    (
        "underpowered_n",
        "G19_power_is_recomputed_for_claim_margin_and_planned_n_meets_90pct",
        lambda value: value["power_analysis"].update(
            planned_clusters_per_backend=60
        ),
    ),
    (
        "round_pseudoreplication",
        "G20_cluster_outer_block_and_independent_backend_rules_prevent_pseudoreplication",
        lambda value: value["statistics_contract"].update(
            resampling_unit="round"
        ),
    ),
    (
        "pointwise_ci",
        "G21_joint_maxT_bootstrap_has_fixed_seed_resamples_and_closed_families",
        lambda value: value["statistics_contract"].update(
            method="pointwise_bootstrap"
        ),
    ),
    (
        "epsilon_denominator",
        "G22_relative_zero_denominator_is_null_without_epsilon",
        lambda value: value["statistics_contract"].update(
            zero_baseline_ler_rule="add epsilon"
        ),
    ),
    (
        "drop_timeout",
        "G23_failure_timeout_oom_and_missingness_are_retained_fail_closed",
        lambda value: value["missingness_contract"]["status_enum"].remove(
            "TIMEOUT"
        ),
    ),
    (
        "rank_oracle",
        "G25_hidden_teacher_oracle_and_future_smoother_are_privileged_nonranking",
        lambda value: _method(value, "hidden_state_decoder_oracle")[
            "current_result"
        ].update(rank=1),
    ),
    (
        "duplicate_work",
        "G27_deduplication_merges_versions_by_doi_arxiv_or_title_author",
        lambda value: value["literature_records"].append(
            deepcopy(_literature_record(value, "LIT-PUVIANI-2025"))
        ),
    ),
    (
        "promote_external",
        "G28_every_literature_record_has_complete_same_task_screening",
        lambda value: _literature_record(value, "LIT-SIVAK-2023").update(
            external_same_task_eligible=True
        ),
    ),
    (
        "auto_sota",
        "G30_registry_best_never_auto_promotes_external_sota",
        lambda value: value["external_claim_contract"].update(
            registry_best_auto_promotes_external_sota=True
        ),
    ),
    (
        "erase_puviani_block",
        "G31_puviani_is_one_work_family_and_remains_officially_blocked",
        lambda value: _literature_record(
            value, "LIT-PUVIANI-2025"
        ).update(typed_null_reasons=[]),
    ),
    (
        "future_outcome",
        "G32_configured_future_outcome_paths_are_absent_at_seal",
        lambda value: value["future_outcome_absence"][0].update(
            exists_at_seal=True
        ),
    ),
    (
        "forge_binding",
        "G33_config_code_parent_and_historical_implementations_are_hash_live",
        lambda value: value["implementation_bindings"][0].update(
            sha256="f" * 64
        ),
    ),
    (
        "forge_source_count",
        "G34_source_data_literature_ledger_and_markdown_are_lossless_and_live",
        lambda value: value["source_data"].update(rows=0),
    ),
    (
        "forge_mutation_audit",
        "G35_one_targeted_semantic_mutation_per_gate_fails_closed",
        lambda value: value["semantic_mutation_audit"].update(
            detected=35, all_detected=False
        ),
    ),
    (
        "inject_performance",
        "G36_protocol_pass_has_no_numeric_metric_rank_or_performance_verdict",
        lambda value: value["performance_state"].update(round_ler=0.01),
    ),
]


@pytest.mark.parametrize(
    ("mutation_id", "target_gate", "mutate"),
    DIRECT_MUTATIONS,
    ids=[row[0] for row in DIRECT_MUTATIONS],
)
def test_independent_direct_mutations_fail_their_target_gate(
    mutation_id: str,
    target_gate: str,
    mutate: Mutation,
    report: dict[str, Any],
) -> None:
    del mutation_id
    candidate = deepcopy(report)
    mutate(candidate)
    gates = _offline_gates(candidate, report)
    assert gates[target_gate] is False


def test_frozen_at_and_method_claim_laundering_change_semantic_identity(
    report: dict[str, Any],
) -> None:
    timestamp = deepcopy(report)
    timestamp["frozen_at"] = "2099-01-01T00:00:00+00:00"
    assert _offline_gates(timestamp, report)[
        "G01_identity_and_preoutcome_seal"
    ] is False
    assert subject._canonical_sha256(
        subject._analysis_payload(timestamp)
    ) != report["analysis_sha256"]

    method_claim = deepcopy(report)
    _method(method_claim, "causal_transformer_posterior")[
        "claim_boundary"
    ] = "state of the art"
    assert _offline_gates(method_claim, report)[
        "G06_exact_mandatory_baselines_and_anchors_are_present"
    ] is False
    assert subject._canonical_sha256(
        subject._analysis_payload(method_claim)
    ) != report["analysis_sha256"]

    metadata_only = deepcopy(report)
    metadata_only["generated_at_utc"] = "2099-01-01T00:00:00+00:00"
    assert subject._canonical_sha256(
        subject._analysis_payload(metadata_only)
    ) == report["analysis_sha256"]
    assert all(_offline_gates(metadata_only, report).values())


def test_markdown_claim_laundering_fails_exact_live_gate(
    report: dict[str, Any],
) -> None:
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="_test_t9_1_4_markdown_", dir=ROOT / "runs"
    ) as directory:
        candidate = deepcopy(report)
        markdown = Path(directory) / "laundered.md"
        markdown.write_text(
            subject._render_markdown(candidate)
            + "\n未经证据支持的 external SOTA 声明。\n",
            encoding="utf-8",
        )
        candidate["markdown"] = subject._binding(markdown)
        gates = subject.evaluate_gates(
            candidate,
            config_path=CONFIG,
            check_live_files=True,
            expected_parent_handoff=report["parent_handoff"],
            expected_artifact_registry=report["artifact_registry"],
            expected_implementation_bindings=report[
                "implementation_bindings"
            ],
        )
        assert gates[
            "G34_source_data_literature_ledger_and_markdown_are_lossless_and_live"
        ] is False


def test_future_outcome_presence_fails_before_any_publication(
    config: dict[str, Any],
) -> None:
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="_test_t9_1_4_future_", dir=ROOT / "runs"
    ) as directory:
        work = Path(directory)
        future = work / "future_outcome.json"
        future.write_text('{"forbidden": true}\n', encoding="utf-8")
        candidate_config = deepcopy(config)
        candidate_config["future_outcome_paths"] = [
            future.relative_to(ROOT).as_posix()
        ]
        config_path = work / "config.json"
        config_path.write_text(
            json.dumps(candidate_config, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        source = work / "source.csv"
        literature = work / "literature.csv"
        markdown = work / "report.md"
        with pytest.raises(
            ValueError,
            match="cannot seal T9.1.4 after registered outcome access",
        ):
            subject.build_report(
                config_path=config_path,
                source_data_path=source,
                literature_ledger_path=literature,
                markdown_path=markdown,
            )
        assert source.exists() is False
        assert literature.exists() is False
        assert markdown.exists() is False


def test_verify_entry_point_is_read_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = (REPORT, SOURCE, LITERATURE, MARKDOWN)
    before = {path: path.read_bytes() for path in paths}
    assert subject.main(["--verify"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert all(output.values())
    assert {path: path.read_bytes() for path in paths} == before
