"""Compile and verify the T9.1.4 baseline/search/power preregistration.

This is a pre-outcome protocol compiler, not a performance runner.  Historical
implementations are inventoried but must be requalified under the Phase-9
common IQ/action/compute harness.  Missing implementations and unavailable
external assets remain typed nulls.  Passing this module therefore freezes a
comparison design; it does not produce a rank, an LER/lifetime result, or an
external-SOTA claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

from cnn_fpga.benchmark import (
    t9_1_3_post_outcome_governance_addendum as parent_addendum,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T9.1.4"
CONFIG_SCHEMA_VERSION = "t9.1.4-baseline-search-power-config-v1"
REPORT_SCHEMA_VERSION = "t9.1.4-baseline-search-power-report-v1"
PROTOCOL_ID = "PHASE9-COMMON-HARNESS-BASELINE-REGISTRY-V1"
VERDICT = "PASS_T9_1_4_BASELINE_SEARCH_POWER_REGISTRY_FROZEN"

DEFAULT_CONFIG = ROOT / "configs/phase9/t9_1_4_baseline_registry.json"
DEFAULT_INPUT_CONTRACT = ROOT / "configs/phase9/t9_1_4_input_contract.json"
DEFAULT_REPORT = ROOT / "docs/t9_1_4_baseline_search_power_registry.json"
DEFAULT_SOURCE_DATA = (
    ROOT / "docs/t9_1_4_baseline_search_power_registry_source_data.csv"
)
DEFAULT_LITERATURE_LEDGER = (
    ROOT / "docs/t9_1_4_literature_search_ledger.csv"
)
DEFAULT_MARKDOWN = ROOT / "docs/phase9_baseline_search_power_registry.md"
IMPLEMENTATION = Path(__file__).resolve()

PHASE9_PARENT = ROOT / "docs/t9_1_1_three_lane_protocol.json"
HISTORICAL_POWER_PROXY = (
    ROOT / "configs/phase6d/t6_20_3_multimode_preregistration.json"
)

RANKING_CLASS_IDS = (
    "MATCHED_DEPLOYABLE_RANKED",
    "UNBUDGETED_CAPACITY_CEILING_NONRANKING",
    "PRIVILEGED_UPPER_BOUND_NONRANKING",
    "PROTOCOL_ANCHOR_NONRANKING",
)
MANDATORY_BASELINE_IDS = (
    "standard_measurement_feedback_sbs",
    "static_recovery_map",
    "markovian_feedback_fnn",
    "paper_constrained_nmf",
    "sliding_window_map",
    "ewma_adaptive_map",
    "kalman_adaptive_map",
    "ukf_adaptive_map",
    "regime_hmm",
    "interacting_multiple_model_filter",
    "bocpd",
    "particle_smc",
    "bayesian_predictor",
    "cnn_posterior",
    "gru_posterior",
    "tcn_posterior",
    "causal_ssm_posterior",
    "causal_transformer_posterior",
)
PROTOCOL_ANCHOR_IDS = (
    "standard_binning",
    "autonomous_sbs_no_measurement",
    "no_correction_idle_memory",
)
CAPACITY_CEILING_IDS = (
    "high_particle_smc_ceiling",
    "full_grid_bayesian_ceiling",
    "large_gru_ensemble_ceiling",
    "large_ssm_ceiling",
    "large_transformer_ceiling",
)
PRIVILEGED_IDS = (
    "hidden_state_decoder_oracle",
    "hidden_state_teacher",
    "future_suffix_smoother",
    "finite_horizon_control_oracle",
)
PROPOSED_ID = "proposed_observed_only_posterior_predictive_risk_aware"

ONLINE_ACCOUNTING_FIELDS = (
    "parameter_count",
    "persistent_state_bytes",
    "transient_workspace_bytes",
    "peak_rss_delta_bytes",
    "preprocessing_macs",
    "inference_macs",
    "state_update_macs",
    "serialization_transfer_macs",
    "total_online_macs",
    "total_online_flops",
    "preprocessing_wallclock_ns",
    "inference_wallclock_ns",
    "state_update_wallclock_ns",
    "serialization_transfer_wallclock_ns",
    "total_wallclock_ns",
    "wallclock_p50_ns",
    "wallclock_p95_ns",
    "wallclock_p99_ns",
    "wallclock_max_ns",
    "deadline_miss_count",
    "timeout_count",
    "oom_count",
    "numeric_failure_count",
    "fallback_count",
)

EXPECTED_IMPLEMENTATION_FRAGMENTS: dict[str, tuple[str, ...]] = {
    "cnn_fpga/benchmark/standard_binning_baseline.py": (
        "STANDARD_BINNING_DESCRIPTOR",
    ),
    "physics/autonomous_sbs.py": (
        "class NonselectiveSBSSimulator",
        "class IdleMemorySimulator",
    ),
    "cnn_fpga/benchmark/static_map_baseline.py": (
        "def build_static_map_validation",
    ),
    "cnn_fpga/benchmark/static_protocol_decoder.py": (
        "StaticProtocol",
    ),
    "cnn_fpga/benchmark/latest_outcome_markovian_baseline.py": (
        "def run_benchmark",
    ),
    "cnn_fpga/benchmark/puviani_paper_constrained_artifacts.py": (
        "PAPER_CONSTRAINED_REIMPLEMENTATION",
    ),
    "docs/t9_1_3_puviani_paper_constrained.json": (
        "PASS_ARTIFACT_LANE_AND_EXECUTABLE_REIMPLEMENTATION",
    ),
    "cnn_fpga/decoder/sliding_window_syndrome.py": (
        "class SlidingWindowPeriodicEstimator",
    ),
    "cnn_fpga/benchmark/continuous_adaptive_map.py": (
        "def build_continuous_adaptive_validation",
    ),
    "cnn_fpga/decoder/periodic_adaptive_map.py": (
        "class LatestWindowPeriodicPredictor",
    ),
    "cnn_fpga/decoder/ukf_baseline.py": (
        "class UKFBaseline",
    ),
    "cnn_fpga/benchmark/regime_hmm_baseline.py": (
        "class RegimeHMMDescriptor",
    ),
    "cnn_fpga/benchmark/external_drift_adaptive_lane.py": (
        "class BOCDCandidate",
    ),
    "cnn_fpga/decoder/particle_filter_baseline.py": (
        "class ParticleFilterBaseline",
    ),
    "cnn_fpga/config/p4_particle_filter_smoke.yaml": (
        "particle",
    ),
    "cnn_fpga/benchmark/memory_assisted_bayesian_decoder.py": (
        "class MemoryBayesianDescriptor",
    ),
    "cnn_fpga/model/tiny_cnn.py": (
        "class TinyCNN",
    ),
    "cnn_fpga/benchmark/learned_model_eligibility_replay.py": (
        "def build_report",
    ),
    "cnn_fpga/decoder/slow_loop_model_selection.py": (
        "class CausalTCN",
        "class SmallGRU",
    ),
    "cnn_fpga/benchmark/oracle_baseline.py": (
        "def build_oracle_validation",
    ),
    "physics/trajectory_lookup_control_oracle.py": (
        "def build_policy",
    ),
}

GATE_IDS = (
    "G01_identity_and_preoutcome_seal",
    "G02_parent_addendum_live_validated_and_semantic_seal_reconstructed",
    "G03_qualified_and_no_go_parent_branches_both_release_without_rank",
    "G04_official_external_physical_and_performance_claims_are_null",
    "G05_only_ler_and_lifetime_signatures_are_bound_without_hil_mixing",
    "G06_exact_mandatory_baselines_and_anchors_are_present",
    "G07_standard_autonomous_and_no_correction_sbs_rows_are_distinct",
    "G08_four_ranking_classes_have_exact_nontransfer_semantics",
    "G09_no_method_is_currently_ranked_before_common_harness_qualification",
    "G10_deployable_and_ceiling_rows_are_observed_only",
    "G11_input_representations_are_identical_strata_not_architecture_privilege",
    "G12_samples_actions_codebook_fallback_and_cost_are_common",
    "G13_compute_precision_and_accounting_schema_are_complete_and_fail_closed",
    "G14_preprocessing_state_update_serialization_and_transfer_are_in_deadline",
    "G15_selection_budget_and_all_attempt_ledger_are_complete",
    "G16_hidden_best_of_n_and_formal_reselection_are_prohibited",
    "G17_pilot_is_single_pass_balanced_nonclaiming_and_keeps_all_baselines",
    "G18_formal_effect_targets_point_15pct_and_lcb_10pct",
    "G19_power_is_recomputed_for_claim_margin_and_planned_n_meets_90pct",
    "G20_cluster_outer_block_and_independent_backend_rules_prevent_pseudoreplication",
    "G21_joint_maxT_bootstrap_has_fixed_seed_resamples_and_closed_families",
    "G22_relative_zero_denominator_is_null_without_epsilon",
    "G23_failure_timeout_oom_and_missingness_are_retained_fail_closed",
    "G24_capacity_ceiling_is_unbudgeted_but_nonranking",
    "G25_hidden_teacher_oracle_and_future_smoother_are_privileged_nonranking",
    "G26_literature_cutoff_queries_and_source_failures_are_explicit",
    "G27_deduplication_merges_versions_by_doi_arxiv_or_title_author",
    "G28_every_literature_record_has_complete_same_task_screening",
    "G29_external_exclusions_and_typed_null_reasons_are_nonempty",
    "G30_registry_best_never_auto_promotes_external_sota",
    "G31_puviani_is_one_work_family_and_remains_officially_blocked",
    "G32_configured_future_outcome_paths_are_absent_at_seal",
    "G33_config_code_parent_and_historical_implementations_are_hash_live",
    "G34_source_data_literature_ledger_and_markdown_are_lossless_and_live",
    "G35_one_targeted_semantic_mutation_per_gate_fails_closed",
    "G36_protocol_pass_has_no_numeric_metric_rank_or_performance_verdict",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "selector": "STRICT_FILE_SHA256",
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _binding_live(binding: Mapping[str, Any]) -> bool:
    path = ROOT / str(binding.get("path", ""))
    return (
        binding.get("selector") == "STRICT_FILE_SHA256"
        and path.is_file()
        and path.stat().st_size == binding.get("bytes")
        and _sha256(path) == binding.get("sha256")
    )


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    _atomic_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        path,
    )


def _normalised_title_tokens(title: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(
                re.findall(
                    r"[a-z0-9]+",
                    title.casefold().replace("gottesman-kitaev-preskill", "gkp"),
                )
            )
        )
    )


def _normalised_doi(value: str | None) -> str | None:
    if value is None:
        return None
    return (
        value.strip()
        .casefold()
        .removeprefix("https://doi.org/")
        .removeprefix("http://doi.org/")
        .removeprefix("doi:")
    )


def _dedup_key(record: Mapping[str, Any]) -> str:
    doi = _normalised_doi(record.get("doi"))
    if doi:
        return f"doi:{doi}"
    arxiv = record.get("arxiv_id")
    if arxiv:
        return f"arxiv:{str(arxiv).casefold()}"
    title = "-".join(_normalised_title_tokens(str(record["title"])))
    author = re.sub(r"[^a-z0-9]+", "", str(record["first_author"]).casefold())
    return f"title-author:{title}:{author}"


def _title_jaccard(left: str, right: str) -> float:
    a = set(_normalised_title_tokens(left))
    b = set(_normalised_title_tokens(right))
    return len(a & b) / len(a | b) if a or b else 1.0


def _normalised_author(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _same_work(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    title_threshold: float,
) -> bool:
    left_doi = _normalised_doi(left.get("doi"))
    right_doi = _normalised_doi(right.get("doi"))
    if left_doi and right_doi and left_doi == right_doi:
        return True
    left_arxiv = str(left.get("arxiv_id") or "").strip().casefold()
    right_arxiv = str(right.get("arxiv_id") or "").strip().casefold()
    if left_arxiv and right_arxiv and left_arxiv == right_arxiv:
        return True
    return (
        _normalised_author(str(left["first_author"]))
        == _normalised_author(str(right["first_author"]))
        and _title_jaccard(str(left["title"]), str(right["title"]))
        >= title_threshold
    )


def _same_work_reason(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    title_threshold: float,
) -> str | None:
    left_doi = _normalised_doi(left.get("doi"))
    right_doi = _normalised_doi(right.get("doi"))
    if left_doi and right_doi:
        return "DOI_EXACT" if left_doi == right_doi else None
    left_arxiv = str(left.get("arxiv_id") or "").strip().casefold()
    right_arxiv = str(right.get("arxiv_id") or "").strip().casefold()
    if left_arxiv and right_arxiv:
        return "ARXIV_EXACT" if left_arxiv == right_arxiv else None
    if (
        _normalised_author(str(left["first_author"]))
        == _normalised_author(str(right["first_author"]))
        and _title_jaccard(str(left["title"]), str(right["title"]))
        >= title_threshold
    ):
        return "TITLE_AUTHOR_JACCARD"
    return None


def _conflicting_identifiers_for_same_title(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    title_threshold: float,
) -> bool:
    same_title_author = (
        _normalised_author(str(left["first_author"]))
        == _normalised_author(str(right["first_author"]))
        and _title_jaccard(str(left["title"]), str(right["title"]))
        >= title_threshold
    )
    if not same_title_author:
        return False
    left_doi = _normalised_doi(left.get("doi"))
    right_doi = _normalised_doi(right.get("doi"))
    if left_doi and right_doi and left_doi != right_doi:
        return True
    left_arxiv = str(left.get("arxiv_id") or "").strip().casefold()
    right_arxiv = str(right.get("arxiv_id") or "").strip().casefold()
    return bool(
        left_arxiv and right_arxiv and left_arxiv != right_arxiv
    )


def _power_analysis(config: Mapping[str, Any]) -> dict[str, Any]:
    plan = config["formal_power_plan"]
    alpha_per = plan["familywise_alpha"] / plan["fixed_comparator_divisor"]
    z_critical = NormalDist().inv_cdf(1.0 - alpha_per)
    z_power = NormalDist().inv_cdf(plan["target_power"])
    margin = plan["baseline_p_l_design_floor"] * (
        plan["alternative_relative_effect"]
        - plan["claim_relative_lcb_threshold"]
    )
    sd = plan["paired_cluster_absolute_difference_sd_ceiling"]
    required = math.ceil(((z_critical + z_power) * sd / margin) ** 2)
    planned = plan["formal_cluster_count_per_backend"]
    achieved = NormalDist().cdf(
        math.sqrt(planned) * margin / sd - z_critical
    )
    sensitivity = []
    for baseline_p_l in (0.08, 0.10, 0.16):
        candidate_margin = baseline_p_l * (
            plan["alternative_relative_effect"]
            - plan["claim_relative_lcb_threshold"]
        )
        sensitivity.append(
            {
                "baseline_p_l": baseline_p_l,
                "required_clusters": math.ceil(
                    (
                        (z_critical + z_power)
                        * sd
                        / candidate_margin
                    )
                    ** 2
                ),
                "achieved_power_at_planned_n": NormalDist().cdf(
                    math.sqrt(planned) * candidate_margin / sd - z_critical
                ),
            }
        )
    return {
        "alpha_per_comparator": alpha_per,
        "one_sided_z_critical": z_critical,
        "z_target_power": z_power,
        "absolute_claim_margin_at_design_floor": margin,
        "required_clusters_at_design_floor": required,
        "planned_clusters_per_backend": planned,
        "achieved_power_at_design_floor": achieved,
        "sensitivity": sensitivity,
        "design_only_not_final_inference": True,
    }


def _consume_parent_handoff() -> dict[str, Any]:
    checks = parent_addendum.validate_downstream_handoff(
        parent_addendum.DEFAULT_REPORT,
        input_contract_path=DEFAULT_INPUT_CONTRACT,
    )
    report = _load(parent_addendum.DEFAULT_REPORT)
    seal = copy.deepcopy(report["downstream_semantic_seal"])
    if seal["payload"]["claim_slots"] != {
        "official_exact": None,
        "puviani_surpass": None,
        "paper_scale_lifetime": None,
    }:
        raise ValueError("T9.1.3 downstream claim slots are not null")
    return {
        "source_task_id": "T9.1.3",
        "terminal_branch": report["terminal_branch"],
        "ordered_consumer_checks": checks,
        "downstream_semantic_seal": seal,
        "resolved_terminal_state": seal["payload"]["terminal_resolution"][
            "resolved_terminal_state"
        ],
        "releases_t9_1_4": seal["payload"]["terminal_resolution"][
            "releases_t9_1_4"
        ],
        "matched_phase9_ranking_eligible": seal["payload"][
            "ranking_boundary"
        ]["matched_phase9_ranking_eligible"],
        "sota_claim_eligible": seal["payload"]["ranking_boundary"][
            "sota_claim_eligible"
        ],
        "raw_addendum_hash_consumed": False,
    }


def _parent_terminal_options() -> list[dict[str, Any]]:
    contract = _load(DEFAULT_INPUT_CONTRACT)
    return copy.deepcopy(contract["terminal_state_mapping"])


def _privilege(method: Mapping[str, Any]) -> dict[str, Any]:
    method_id = str(method["method_id"])
    class_id = str(method["registry_class"])
    privileged = class_id == "PRIVILEGED_UPPER_BOUND_NONRANKING"
    return {
        "observed_only": not privileged,
        "hidden_truth_access": method_id
        in {"hidden_state_decoder_oracle", "hidden_state_teacher"},
        "future_suffix_access": method_id == "future_suffix_smoother",
        "scenario_id_access": False,
        "formal_label_access": False,
        "privileged_diagnostic_evaluation_on_formal": privileged,
        "deployable_teacher_call_during_formal": False,
        "assumed_model_branching_access": (
            method_id == "finite_horizon_control_oracle"
        ),
        "deployable": class_id == "MATCHED_DEPLOYABLE_RANKED",
    }


def _current_accounting() -> dict[str, None]:
    return {key: None for key in ONLINE_ACCOUNTING_FIELDS}


def _enrich_methods(
    config: Mapping[str, Any], parent_handoff: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in config["methods"]:
        method = copy.deepcopy(raw)
        class_id = method["registry_class"]
        method["privilege"] = _privilege(method)
        method["common_contract_ids"] = {
            "task_signature": config["protocol_id"],
            "compute": "PHASE9_MATCHED_COMPUTE_V1",
            "selection": "PHASE9_ALL_ATTEMPTS_SELECTION_V1",
            "missingness": "PHASE9_FAILURE_RETENTION_V1",
        }
        method["current_accounting"] = _current_accounting()
        method["current_result"] = {
            "numeric_metrics": None,
            "rank": None,
            "matched_eligible": False,
            "ceiling_eligible": False,
            "registered_best_vote": False,
            "external_sota_vote": False,
        }
        method["target_eligibility"] = {
            "matched_candidate": class_id == "MATCHED_DEPLOYABLE_RANKED",
            "capacity_ceiling_candidate": (
                class_id == "UNBUDGETED_CAPACITY_CEILING_NONRANKING"
            ),
            "privileged_upper_bound": (
                class_id == "PRIVILEGED_UPPER_BOUND_NONRANKING"
            ),
            "protocol_anchor": class_id == "PROTOCOL_ANCHOR_NONRANKING",
        }
        if method["method_id"] == "paper_constrained_nmf":
            method["t9_1_3_provenance"] = {
                "terminal_branch": parent_handoff["terminal_branch"],
                "resolved_terminal_state": parent_handoff[
                    "resolved_terminal_state"
                ],
                "semantic_sha256": parent_handoff[
                    "downstream_semantic_seal"
                ]["semantic_sha256"],
                "matched_phase9_ranking_eligible": False,
                "official_exact": None,
                "puviani_surpass": None,
                "paper_scale_lifetime": None,
            }
        output.append(method)
    return output


def _deduplicate_literature_records(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge raw source hits into one comparator per canonical work.

    The config keeps reviewer-readable canonical rows, each with every raw
    source identifier.  This function deliberately expands those identifiers
    back into a raw hit stream and then performs the registered DOI -> arXiv ->
    title/first-author merge.  Conflicting screening decisions fail closed
    instead of silently selecting the more favourable version.
    """

    threshold = float(
        config["literature_search"]["deduplication"][
            "title_token_jaccard_threshold"
        ]
    )
    raw_hits: list[dict[str, Any]] = []
    for canonical in config["literature_records"]:
        if not canonical["raw_records"]:
            raise ValueError(
                f"{canonical['record_id']} has no retained raw source record"
            )
        for source_record in canonical["raw_records"]:
            hit = copy.deepcopy(canonical)
            hit["raw_records"] = [source_record]
            # A raw source hit carries only the identifier supplied by that
            # source.  DOI/arXiv cross-version linkage must therefore be
            # recovered by the registered identity rules, not by copying the
            # already-curated pair of identifiers into every hit.
            hit["doi"] = None
            hit["arxiv_id"] = None
            prefix, separator, identifier = source_record.partition(":")
            if not separator or not identifier:
                raise ValueError(f"invalid raw source id: {source_record}")
            if prefix.casefold() == "doi":
                hit["doi"] = identifier
            elif prefix.casefold() == "arxiv":
                hit["arxiv_id"] = identifier
            else:
                raise ValueError(
                    "raw_records must contain bibliographic DOI/arXiv hits; "
                    f"move artifact evidence to artifact_records: {source_record}"
                )
            raw_hits.append(hit)
    raw_source_ids = [
        hit["raw_records"][0] for hit in raw_hits
    ]
    if len(raw_source_ids) != len(set(raw_source_ids)):
        raise ValueError("duplicate bibliographic raw source id")

    groups: list[dict[str, Any]] = []
    merge_events: list[dict[str, Any]] = []
    merge_rule_counts = {
        "DOI_EXACT": 0,
        "ARXIV_EXACT": 0,
        "TITLE_AUTHOR_JACCARD": 0,
    }
    immutable_screening_fields = (
        "signature_checks",
        "code_status",
        "data_status",
        "official_artifact_status",
        "eligibility_class",
        "exclusion_code",
        "typed_null_reasons",
        "external_same_task_eligible",
    )
    for hit in raw_hits:
        if any(
            _conflicting_identifiers_for_same_title(
                row, hit, title_threshold=threshold
            )
            for row in groups
        ):
            raise ValueError(
                "conflicting non-null identifiers for a same-title work: "
                f"{hit['record_id']}"
            )
        matches = [
            (
                row,
                _same_work_reason(
                    row, hit, title_threshold=threshold
                ),
            )
            for row in groups
        ]
        matches = [item for item in matches if item[1] is not None]
        if len(matches) > 1:
            raise ValueError(
                f"ambiguous literature identity for {hit['record_id']}"
            )
        if not matches:
            groups.append(copy.deepcopy(hit))
            continue
        canonical, merge_reason = matches[0]
        assert merge_reason is not None
        if any(
            canonical[field] != hit[field]
            for field in immutable_screening_fields
        ):
            raise ValueError(
                "conflicting same-task screening across versions for "
                f"{canonical['record_id']}"
            )
        canonical["raw_records"] = sorted(
            set(canonical["raw_records"]) | set(hit["raw_records"])
        )
        canonical["query_ids"] = sorted(
            set(canonical["query_ids"]) | set(hit["query_ids"])
        )
        canonical["artifact_records"] = sorted(
            set(canonical.get("artifact_records", []))
            | set(hit.get("artifact_records", []))
        )
        # Restore a retained canonical identifier when a later version brings
        # it; the merged work must preserve both DOI and arXiv provenance.
        if canonical.get("doi") is None and hit.get("doi"):
            canonical["doi"] = hit["doi"]
        if canonical.get("arxiv_id") is None and hit.get("arxiv_id"):
            canonical["arxiv_id"] = hit["arxiv_id"]
        if canonical.get("repo_url") is None and hit.get("repo_url"):
            canonical["repo_url"] = hit["repo_url"]
        merge_rule_counts[merge_reason] += 1
        merge_events.append(
            {
                "source_record": hit["raw_records"][0],
                "canonical_record_id": canonical["record_id"],
                "rule": merge_reason,
            }
        )

    rows: list[dict[str, Any]] = []
    for raw in groups:
        row = copy.deepcopy(raw)
        row["doi"] = _normalised_doi(row.get("doi"))
        row["dedup_key"] = _dedup_key(row)
        row["normalised_title_tokens"] = list(
            _normalised_title_tokens(row["title"])
        )
        row["raw_record_count"] = len(row["raw_records"])
        row["artifact_records"] = sorted(
            set(row.get("artifact_records", []))
        )
        row["artifact_record_count"] = len(row["artifact_records"])
        row["numeric_result"] = None
        row["external_rank"] = None
        row["external_sota_vote"] = False
        rows.append(row)
    audit = {
        "pipeline_executed": True,
        "input_bibliographic_hit_count": len(raw_hits),
        "artifact_evidence_count": sum(
            len(row.get("artifact_records", []))
            for row in config["literature_records"]
        ),
        "canonical_work_count": len(rows),
        "merged_bibliographic_version_count": len(raw_hits) - len(rows),
        "unique_dedup_key_count": len({row["dedup_key"] for row in rows}),
        "bibliographic_source_ids_sha256": _canonical_sha256(
            sorted(
                source
                for row in config["literature_records"]
                for source in row["raw_records"]
            )
        ),
        "artifact_evidence_ids_sha256": _canonical_sha256(
            sorted(
                source
                for row in config["literature_records"]
                for source in row.get("artifact_records", [])
            )
        ),
        "merge_rule_counts": merge_rule_counts,
        "raw_hit_ledger": [
            {
                "source_record": hit["raw_records"][0],
                "claimed_record_id": hit["record_id"],
                "doi": _normalised_doi(hit.get("doi")),
                "arxiv_id": hit.get("arxiv_id"),
                "title": hit["title"],
                "first_author": hit["first_author"],
            }
            for hit in raw_hits
        ],
        "merge_events": merge_events,
        "priority": copy.deepcopy(
            config["literature_search"]["deduplication"]["priority"]
        ),
        "title_token_jaccard_threshold": threshold,
        "first_author_match_required": True,
        "conflicting_screening_rule": "FAIL_CLOSED",
    }
    return rows, audit


def _enrich_literature(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows, _ = _deduplicate_literature_records(config)
    return rows


def _implementation_bindings(
    methods: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    paths = sorted(
        {
            str(path)
            for method in methods
            for path in method["implementation_paths"]
        }
    )
    output = []
    for relative in paths:
        path = ROOT / relative
        binding = _binding(path)
        binding["expected_fragments"] = list(
            EXPECTED_IMPLEMENTATION_FRAGMENTS.get(relative, ())
        )
        output.append(binding)
    return output


def _future_absence(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": str(relative), "exists_at_seal": (ROOT / relative).exists()}
        for relative in config["future_outcome_paths"]
    ]


def _artifact_registry(config_path: Path) -> dict[str, dict[str, Any]]:
    return {
        "config": _binding(config_path),
        "implementation": _binding(IMPLEMENTATION),
        "phase9_parent": _binding(PHASE9_PARENT),
        "t9_1_4_input_contract": _binding(DEFAULT_INPUT_CONTRACT),
        "historical_power_proxy": _binding(HISTORICAL_POWER_PROXY),
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, str]]:
    records: list[tuple[str, str, Any]] = []
    records.append(
        (
            "literature_dedup_audit",
            "registered-dedup-pipeline",
            report["literature_dedup_audit"],
        )
    )
    for row in report["ranking_lanes"]:
        records.append(("ranking_lane", row["lane_id"], row))
    for row in report["methods"]:
        records.append(("method", row["method_id"], row))
    for row in report["literature_search"]["queries"]:
        records.append(("search_query", row["query_id"], row))
    for row in report["literature_search"]["source_status"]:
        records.append(("search_source", row["source_id"], row))
    for row in report["literature_records"]:
        records.append(("literature_record", row["record_id"], row))
    for row in report["power_analysis"]["sensitivity"]:
        records.append(
            (
                "power_sensitivity",
                f"baseline_p_l={row['baseline_p_l']:.2f}",
                row,
            )
        )
    for index, row in enumerate(report["parent_terminal_options"], start=1):
        records.append(("parent_terminal_option", f"option-{index}", row))
    for row in report["future_outcome_absence"]:
        records.append(("future_outcome_absence", row["path"], row))
    for name, row in report["artifact_registry"].items():
        records.append(("artifact_binding", name, row))
    for row in report["implementation_bindings"]:
        records.append(("implementation_binding", row["path"], row))
    output: list[dict[str, str]] = []
    for record_type, record_id, payload in records:
        payload_json = _canonical_json(payload)
        output.append(
            {
                "record_type": record_type,
                "record_id": record_id,
                "payload_json": payload_json,
                "canonical_sha256": hashlib.sha256(
                    payload_json.encode("utf-8")
                ).hexdigest(),
            }
        )
    return output


def _write_source_data(report: Mapping[str, Any], path: Path) -> None:
    rows = _source_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "record_type",
                "record_id",
                "payload_json",
                "canonical_sha256",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


LITERATURE_LEDGER_FIELDS = (
    "record_id",
    "title",
    "first_author",
    "year",
    "doi",
    "arxiv_id",
    "primary_url",
    "repo_url",
    "query_ids_json",
    "raw_records_json",
    "artifact_records_json",
    "dedup_key",
    "input_match",
    "history_match",
    "action_match",
    "physics_match",
    "online_timing_match",
    "postselection_match",
    "denominator_match",
    "metric_match",
    "compute_match",
    "code_status",
    "data_status",
    "official_artifact_status",
    "eligibility_class",
    "exclusion_code",
    "typed_null_reasons_json",
    "external_same_task_eligible",
    "numeric_result",
    "external_rank",
    "external_sota_vote",
    "canonical_sha256",
)


def _literature_ledger_rows(
    report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for record in report["literature_records"]:
        checks = record["signature_checks"]
        payload = {
            "record_id": record["record_id"],
            "title": record["title"],
            "first_author": record["first_author"],
            "year": record["year"],
            "doi": record["doi"],
            "arxiv_id": record["arxiv_id"],
            "primary_url": record["primary_url"],
            "repo_url": record["repo_url"],
            "query_ids_json": _canonical_json(record["query_ids"]),
            "raw_records_json": _canonical_json(record["raw_records"]),
            "artifact_records_json": _canonical_json(
                record["artifact_records"]
            ),
            "dedup_key": record["dedup_key"],
            "input_match": checks["input"],
            "history_match": checks["history"],
            "action_match": checks["action"],
            "physics_match": checks["physics"],
            "online_timing_match": checks["online_timing"],
            "postselection_match": checks["postselection"],
            "denominator_match": checks["denominator"],
            "metric_match": checks["metric"],
            "compute_match": checks["compute"],
            "code_status": record["code_status"],
            "data_status": record["data_status"],
            "official_artifact_status": record[
                "official_artifact_status"
            ],
            "eligibility_class": record["eligibility_class"],
            "exclusion_code": record["exclusion_code"],
            "typed_null_reasons_json": _canonical_json(
                record["typed_null_reasons"]
            ),
            "external_same_task_eligible": record[
                "external_same_task_eligible"
            ],
            "numeric_result": record["numeric_result"],
            "external_rank": record["external_rank"],
            "external_sota_vote": record["external_sota_vote"],
        }
        payload["canonical_sha256"] = _canonical_sha256(payload)
        output.append(payload)
    return output


def _write_literature_ledger(report: Mapping[str, Any], path: Path) -> None:
    rows = _literature_ledger_rows(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LITERATURE_LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _csv_lossless(report: Mapping[str, Any], path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows == _source_rows(report)


def _literature_csv_lossless(
    report: Mapping[str, Any], path: Path
) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = []
    for raw in _literature_ledger_rows(report):
        expected.append(
            {
                key: (
                    ""
                    if raw[key] is None
                    else (
                        "True"
                        if raw[key] is True
                        else "False"
                        if raw[key] is False
                        else str(raw[key])
                    )
                )
                for key in LITERATURE_LEDGER_FIELDS
            }
        )
    return rows == expected


def _safe(function: Callable[[], bool]) -> bool:
    try:
        return bool(function())
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        StopIteration,
        OSError,
    ):
        return False


def _method_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {row["method_id"]: row for row in report["methods"]}


def _valid_mutation_placeholder() -> dict[str, Any]:
    return {
        "count": len(GATE_IDS),
        "detected": len(GATE_IDS),
        "all_detected": True,
        "cases": [
            {
                "mutation_id": f"placeholder-{index:02d}",
                "target_gate": gate,
                "mutation_applied": True,
                "rejection_mode": "TARGET_GATE_FALSE",
                "rejected": True,
            }
            for index, gate in enumerate(GATE_IDS, start=1)
        ],
    }


def evaluate_gates(
    report: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    check_live_files: bool = True,
    expected_parent_handoff: Mapping[str, Any] | None = None,
    expected_artifact_registry: Mapping[str, Any] | None = None,
    expected_implementation_bindings: Sequence[Mapping[str, Any]]
    | None = None,
) -> dict[str, bool]:
    config = _load(config_path)
    methods = report.get("methods", [])
    by_id = _method_map(report) if isinstance(methods, list) else {}
    literature = report.get("literature_records", [])
    classes = {
        row.get("lane_id") for row in report.get("ranking_lanes", [])
    }
    mandatory = tuple(
        row["method_id"]
        for row in methods
        if row.get("mandatory") is True
    )
    anchors = tuple(
        row["method_id"]
        for row in methods
        if row.get("registry_class") == "PROTOCOL_ANCHOR_NONRANKING"
    )
    power = report.get("power_analysis", {})
    expected_power = _power_analysis(config)
    expected_methods = _enrich_methods(config, report["parent_handoff"])
    expected_literature, expected_dedup_audit = (
        _deduplicate_literature_records(config)
    )

    def parent_ok() -> bool:
        expected = (
            copy.deepcopy(dict(expected_parent_handoff))
            if expected_parent_handoff is not None
            else _consume_parent_handoff()
        )
        return (
            report["parent_handoff"] == expected
            and all(expected["ordered_consumer_checks"].values())
            and expected["downstream_semantic_seal"]["semantic_sha256"]
            == _canonical_sha256(
                expected["downstream_semantic_seal"]["payload"]
            )
            and expected["raw_addendum_hash_consumed"] is False
        )

    def parent_options_ok() -> bool:
        options = report["parent_terminal_options"]
        if options != _parent_terminal_options() or len(options) != 2:
            return False
        qualified, no_go = options
        return (
            qualified["terminal_state"]
            == "QUALIFIED_PAPER_CONSTRAINED_BASELINE"
            and no_go["terminal_state"]
            == "NO_GO_PAPER_CONSTRAINED_REIMPLEMENTATION"
            and qualified["releases_t9_1_4"] is True
            and no_go["releases_t9_1_4"] is True
            and qualified["matched_phase9_ranking_eligible"] is False
            and no_go["matched_phase9_ranking_eligible"] is False
            and no_go["typed_payload"]["numeric_metrics"] is None
            and no_go["typed_payload"]["rank"] is None
            and by_id["paper_constrained_nmf"]["current_result"][
                "matched_eligible"
            ]
            is False
        )

    def all_current_results_null() -> bool:
        return all(
            row["current_result"]
            == {
                "numeric_metrics": None,
                "rank": None,
                "matched_eligible": False,
                "ceiling_eligible": False,
                "registered_best_vote": False,
                "external_sota_vote": False,
            }
            and all(
                value is None for value in row["current_accounting"].values()
            )
            for row in methods
        )

    def observed_only_ok() -> bool:
        for row in methods:
            if row["registry_class"] not in {
                "MATCHED_DEPLOYABLE_RANKED",
                "UNBUDGETED_CAPACITY_CEILING_NONRANKING",
            }:
                continue
            privilege = row["privilege"]
            if (
                privilege["observed_only"] is not True
                or privilege["hidden_truth_access"]
                or privilege["future_suffix_access"]
                or privilege["scenario_id_access"]
                or privilege["formal_label_access"]
                or privilege["deployable_teacher_call_during_formal"]
                or privilege["privileged_diagnostic_evaluation_on_formal"]
                or privilege["assumed_model_branching_access"]
            ):
                return False
        return True

    def implementation_bindings_ok() -> bool:
        expected = (
            copy.deepcopy(list(expected_implementation_bindings))
            if expected_implementation_bindings is not None
            else _implementation_bindings(config["methods"])
        )
        if report["implementation_bindings"] != expected:
            return False
        if not check_live_files:
            return True
        for binding in expected:
            if not _binding_live(binding):
                return False
            text = (ROOT / binding["path"]).read_text(
                encoding="utf-8", errors="replace"
            )
            if not all(
                fragment.casefold() in text.casefold()
                for fragment in binding["expected_fragments"]
            ):
                return False
        return True

    def artifact_registry_ok() -> bool:
        expected = (
            copy.deepcopy(dict(expected_artifact_registry))
            if expected_artifact_registry is not None
            else _artifact_registry(config_path)
        )
        return report["artifact_registry"] == expected and (
            not check_live_files
            or all(_binding_live(row) for row in expected.values())
        )

    def outputs_ok() -> bool:
        basic = (
            report["source_data"]["rows"] == len(_source_rows(report))
            and report["literature_ledger"]["rows"]
            == len(_literature_ledger_rows(report))
            == len(literature)
            and report["markdown"]["path"].endswith(".md")
        )
        if not basic or not check_live_files:
            return basic
        source_path = ROOT / report["source_data"]["path"]
        literature_path = ROOT / report["literature_ledger"]["path"]
        markdown_path = ROOT / report["markdown"]["path"]
        markdown_text = markdown_path.read_text(encoding="utf-8")
        return (
            _binding_live(
                {**report["source_data"], "selector": "STRICT_FILE_SHA256"}
            )
            and _binding_live(
                {
                    **report["literature_ledger"],
                    "selector": "STRICT_FILE_SHA256",
                }
            )
            and _binding_live(
                {**report["markdown"], "selector": "STRICT_FILE_SHA256"}
            )
            and _csv_lossless(report, source_path)
            and _literature_csv_lossless(report, literature_path)
            and markdown_text == _render_markdown(report)
            and all(
                f"`{row['method_id']}`"
                in markdown_text
                for row in methods
            )
            and all(
                f"`{row['record_id']}`"
                in markdown_text
                for row in literature
            )
        )

    gates = {
        "G01_identity_and_preoutcome_seal": _safe(
            lambda: report["task_id"] == TASK_ID
            and report["schema_version"] == REPORT_SCHEMA_VERSION
            and report["config_schema_version"] == CONFIG_SCHEMA_VERSION
            and report["protocol_id"] == PROTOCOL_ID
            and report["seal_state"] == "SEALED_PRE_OUTCOME"
            and report["frozen_at"] == config["frozen_at"]
        ),
        "G02_parent_addendum_live_validated_and_semantic_seal_reconstructed": _safe(
            parent_ok
        ),
        "G03_qualified_and_no_go_parent_branches_both_release_without_rank": _safe(
            parent_options_ok
        ),
        "G04_official_external_physical_and_performance_claims_are_null": _safe(
            lambda: report["external_claim_contract"]
            == config["external_claim_contract"]
            and all(
                report["external_claim_contract"][key] is None
                for key in (
                    "registered_best",
                    "external_sota",
                    "official_puviani_exact",
                    "puviani_surpass",
                    "physical_lifetime",
                )
            )
        ),
        "G05_only_ler_and_lifetime_signatures_are_bound_without_hil_mixing": _safe(
            lambda: report["task_signature_scope"]
            == config["task_signature_scope"]
            and report["task_signature_scope"]["ranked_lanes"]
            == ["ROUND_LER_SINGLE_MODE", "SIX_STATE_LOGICAL_LIFETIME"]
            and report["task_signature_scope"]["excluded_lane"]
            == "RAW_IQ_DIGITAL_HIL"
        ),
        "G06_exact_mandatory_baselines_and_anchors_are_present": _safe(
            lambda: mandatory == MANDATORY_BASELINE_IDS
            and anchors == PROTOCOL_ANCHOR_IDS
            and methods == expected_methods
            and len(methods) == len({row["method_id"] for row in methods})
            and set(by_id)
            == set(MANDATORY_BASELINE_IDS)
            | set(PROTOCOL_ANCHOR_IDS)
            | set(CAPACITY_CEILING_IDS)
            | set(PRIVILEGED_IDS)
            | {PROPOSED_ID}
        ),
        "G07_standard_autonomous_and_no_correction_sbs_rows_are_distinct": _safe(
            lambda: len(
                {
                    by_id["standard_measurement_feedback_sbs"][
                        "method_id"
                    ],
                    by_id["autonomous_sbs_no_measurement"]["method_id"],
                    by_id["no_correction_idle_memory"]["method_id"],
                }
            )
            == 3
            and by_id["standard_measurement_feedback_sbs"]["registry_class"]
            == "MATCHED_DEPLOYABLE_RANKED"
            and all(
                by_id[method_id]["registry_class"]
                == "PROTOCOL_ANCHOR_NONRANKING"
                for method_id in (
                    "autonomous_sbs_no_measurement",
                    "no_correction_idle_memory",
                )
            )
        ),
        "G08_four_ranking_classes_have_exact_nontransfer_semantics": _safe(
            lambda: report["ranking_lanes"] == config["ranking_lanes"]
            and tuple(row["lane_id"] for row in report["ranking_lanes"])
            == RANKING_CLASS_IDS
            and classes == set(RANKING_CLASS_IDS)
            and [
                row["lane_id"]
                for row in report["ranking_lanes"]
                if row["ranked"]
            ]
            == ["MATCHED_DEPLOYABLE_RANKED"]
        ),
        "G09_no_method_is_currently_ranked_before_common_harness_qualification": _safe(
            all_current_results_null
        ),
        "G10_deployable_and_ceiling_rows_are_observed_only": _safe(
            observed_only_ok
        ),
        "G11_input_representations_are_identical_strata_not_architecture_privilege": _safe(
            lambda: report["task_signature_scope"]["observation_strata"]
            == [
                "RAW_OR_RECORDED_IQ",
                "FIXED_POINT_DERIVED_LLR",
                "BINARY_SYNDROME",
            ]
            and report["task_signature_scope"][
                "primary_observation_stratum"
            ]
            == "FIXED_POINT_DERIVED_LLR"
            and report["task_signature_scope"][
                "secondary_observation_strata"
            ]
            == ["RAW_OR_RECORDED_IQ", "BINARY_SYNDROME"]
            and report["task_signature_scope"]["cross_stratum_selection"]
            == "PROHIBITED_NO_BEST_INPUT_PER_ARCHITECTURE"
            and "frozen T9.2.7 bit-accurate frontend"
            in report["task_signature_scope"]["observation_rule"]
            and "continuous or ideal simulator IQ is privileged ceiling only"
            in report["task_signature_scope"]["raw_iq_ablation_rule"]
            and all(
                row["common_contract_ids"]["task_signature"] == PROTOCOL_ID
                for row in methods
            )
        ),
        "G12_samples_actions_codebook_fallback_and_cost_are_common": _safe(
            lambda: report["task_signature_scope"]["formal_sample_rule"]
            == config["task_signature_scope"]["formal_sample_rule"]
            and report["task_signature_scope"]["action_schema"]
            == config["task_signature_scope"]["action_schema"]
            and report["task_signature_scope"][
                "paired_exogenous_randomness_rule"
            ]
            == "COMMON_RANDOM_NUMBERS_BIND_INITIAL_STATE_DEVICE_DRAW_AND_EXOGENOUS_INNOVATIONS_ONLY"
            and report["task_signature_scope"]["policy_trace_rule"]
            == "ACTION_CONDITIONED_DYNAMICS_MUST_DIVERGE_BY_POLICY_AND_RETAIN_PER_METHOD_TRANSITION_AND_TRACE_HASHES"
            and "post-action physical transitions are policy-specific"
            in report["task_signature_scope"]["formal_sample_rule"]
            and report["task_signature_scope"]["current_codebook_id"] is None
            and report["task_signature_scope"]["current_codebook_sha256"]
            is None
            and report["task_signature_scope"]["null_blocks_matched_rank"]
            is True
        ),
        "G13_compute_precision_and_accounting_schema_are_complete_and_fail_closed": _safe(
            lambda: report["matched_compute_contract"]
            == config["matched_compute_contract"]
            and report["matched_compute_contract"]["batch_size"] == 1
            and report["matched_compute_contract"]["threads"] == 1
            and report["matched_compute_contract"]["gpu_allowed_online"]
            is False
            and report["matched_compute_contract"][
                "eligibility_if_precision_or_codebook_null"
            ]
            is False
            and report["matched_compute_contract"][
                "execution_environment_manifest_payload"
            ]
            is None
            and report["matched_compute_contract"][
                "eligibility_if_environment_manifest_null"
            ]
            is False
            and report["matched_compute_contract"]["precision_profile_payload"]
            is None
            and all(
                tuple(row["current_accounting"]) == ONLINE_ACCOUNTING_FIELDS
                for row in methods
            )
        ),
        "G14_preprocessing_state_update_serialization_and_transfer_are_in_deadline": _safe(
            lambda: report["matched_compute_contract"]["deadline_includes"]
            == [
                "IQ_or_LLR_preprocessing",
                "feature_extraction",
                "model_inference",
                "state_update",
                "serialization",
                "host_to_bank_transfer",
            ]
            and report["matched_compute_contract"]["total_online_deadline_ns"]
            == 1_000_000
            and report["matched_compute_contract"]["fast_path_blocking"]
            == "PROHIBITED_TIMEOUT_MEANS_NO_UPDATE_AND_LKG_FAST_PATH_CONTINUES"
        ),
        "G15_selection_budget_and_all_attempt_ledger_are_complete": _safe(
            lambda: report["selection_contract"]
            == config["selection_contract"]
            and report["selection_contract"]["family_selection_evaluation_cap"]
            == 64
            and len(
                report["selection_contract"]["all_attempts_required_fields"]
            )
            == 16
            and "all_failure_counts"
            in report["selection_contract"]["report_distribution"]
        ),
        "G16_hidden_best_of_n_and_formal_reselection_are_prohibited": _safe(
            lambda: report["selection_contract"][
                "best_of_n_without_complete_ledger"
            ]
            == "INELIGIBLE_BEST_OF_N_HIDDEN"
            and report["selection_contract"]["formal_access_for_selection"]
            == "PROHIBITED"
            and report["selection_contract"][
                "formal_retrain_replace_or_ensemble"
            ]
            == "PROHIBITED"
            and report["pilot_contract"]["formal_reselection"]
            == "PROHIBITED"
        ),
        "G17_pilot_is_single_pass_balanced_nonclaiming_and_keeps_all_baselines": _safe(
            lambda: report["pilot_contract"] == config["pilot_contract"]
            and report["pilot_contract"]["cluster_count"] == 128
            and len(report["pilot_contract"]["macro_scenario_families"])
            * report["pilot_contract"]["clusters_per_macro_family"]
            == 128
            and report["pilot_contract"]["open_count"] == 1
            and report["pilot_contract"]["claim_support"] is False
            and report["pilot_contract"]["baseline_selection"]
            == "PROHIBITED_ALL_MANDATORY_BASELINES_RETAINED"
        ),
        "G18_formal_effect_targets_point_15pct_and_lcb_10pct": _safe(
            lambda: report["formal_power_plan"]
            == config["formal_power_plan"]
            and report["formal_power_plan"]["alternative_relative_effect"]
            == 0.15
            and report["formal_power_plan"][
                "claim_relative_lcb_threshold"
            ]
            == 0.10
            and report["formal_power_plan"]["baseline_p_l_design_floor"]
            == 0.08
            and report["formal_power_plan"]["primary_powered_endpoint"]
            == "ROUND_LER_PROMOTION_VS_EACH_MANDATORY_MATCHED_BASELINE"
            and report["formal_power_plan"]["lifetime_power_status"]
            == "TYPED_NULL_PENDING_T9_6_1_EVENT_RATE_AND_SURVIVAL_POWER_DESIGN"
            and report["formal_power_plan"]["joint_family_power_status"]
            == "NO_90PCT_JOINT_POWER_CLAIM_DESIGN_CONTROLS_LER_COMPARATOR_SCREEN_ONLY"
        ),
        "G19_power_is_recomputed_for_claim_margin_and_planned_n_meets_90pct": _safe(
            lambda: power == expected_power
            and power["required_clusters_at_design_floor"] == 806
            and power["planned_clusters_per_backend"] == 808
            and power["achieved_power_at_design_floor"] >= 0.90
            and report["formal_power_plan"][
                "no_variance_based_resize_after_pilot"
            ]
            is True
            and report["formal_power_plan"][
                "fixed_comparator_divisor"
            ]
            == len(MANDATORY_BASELINE_IDS)
            == 18
            and report["formal_power_plan"]["macro_scenario_families"]
            * report["formal_power_plan"][
                "clusters_per_macro_family_per_backend"
            ]
            == 808
        ),
        "G20_cluster_outer_block_and_independent_backend_rules_prevent_pseudoreplication": _safe(
            lambda: report["statistics_contract"]["resampling_unit"]
            == "device_cell_id_x_scenario_realization_id_x_trajectory_seed_outer_block"
            and "all_six_logical_states"
            in report["statistics_contract"]["cluster_contains"]
            and "all_methods"
            in report["statistics_contract"]["cluster_contains"]
            and "same_initial_state_device_draw_exogenous_innovation_manifest_and_action_opportunities"
            in report["statistics_contract"]["cluster_contains"]
            and "policy_specific_action_conditioned_transition_and_trace_hashes"
            in report["statistics_contract"]["cluster_contains"]
            and "same_physical_trace_and_action_opportunities"
            not in report["statistics_contract"]["cluster_contains"]
            and "outer block"
            in report["statistics_contract"][
                "shared_device_draw_outer_block"
            ]
            and "averaging or rescue"
            in report["statistics_contract"]["backend_rule"]
            and report["statistics_contract"][
                "round_or_cycle_as_independent_sample"
            ]
            == "PROHIBITED"
        ),
        "G21_joint_maxT_bootstrap_has_fixed_seed_resamples_and_closed_families": _safe(
            lambda: report["statistics_contract"]
            == config["statistics_contract"]
            and report["statistics_contract"]["method"]
            == "paired_cluster_maxT_bootstrap"
            and report["statistics_contract"]["resamples"] == 100_000
            and report["statistics_contract"]["bootstrap_seed"]
            == 91420260725
            and len(report["statistics_contract"]["closed_families"]) == 3
            and report["statistics_contract"][
                "posthoc_family_split_or_endpoint_drop"
            ]
            == "PROHIBITED"
        ),
        "G22_relative_zero_denominator_is_null_without_epsilon": _safe(
            lambda: report["statistics_contract"]["zero_baseline_ler_rule"]
            == "relative contrast and CI are null/undefined; absolute contrast only; epsilon substitution prohibited"
            and "recompute baseline and proposed denominator"
            in report["statistics_contract"]["ratio_rule"]
        ),
        "G23_failure_timeout_oom_and_missingness_are_retained_fail_closed": _safe(
            lambda: report["missingness_contract"]
            == config["missingness_contract"]
            and set(report["missingness_contract"]["status_enum"])
            == {
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
            and report["missingness_contract"]["complete_case_deletion"]
            == "PROHIBITED"
            and report["missingness_contract"]["zero_imputation"]
            == "PROHIBITED"
            and "close registered-best and external-SOTA"
            in report["missingness_contract"]["required_baseline_failure"]
        ),
        "G24_capacity_ceiling_is_unbudgeted_but_nonranking": _safe(
            lambda: tuple(
                row["method_id"]
                for row in methods
                if row["registry_class"]
                == "UNBUDGETED_CAPACITY_CEILING_NONRANKING"
            )
            == CAPACITY_CEILING_IDS
            and all(
                by_id[method_id]["current_result"]["rank"] is None
                and by_id[method_id]["current_result"][
                    "registered_best_vote"
                ]
                is False
                and by_id[method_id]["current_result"][
                    "external_sota_vote"
                ]
                is False
                for method_id in CAPACITY_CEILING_IDS
            )
        ),
        "G25_hidden_teacher_oracle_and_future_smoother_are_privileged_nonranking": _safe(
            lambda: tuple(
                row["method_id"]
                for row in methods
                if row["registry_class"]
                == "PRIVILEGED_UPPER_BOUND_NONRANKING"
            )
            == PRIVILEGED_IDS
            and all(
                by_id[method_id]["current_result"]["rank"] is None
                and by_id[method_id]["privilege"]["deployable"] is False
                for method_id in PRIVILEGED_IDS
            )
            and by_id["hidden_state_decoder_oracle"]["privilege"][
                "hidden_truth_access"
            ]
            is True
            and by_id["future_suffix_smoother"]["privilege"][
                "future_suffix_access"
            ]
            is True
            and all(
                by_id[method_id]["privilege"][
                    "deployable_teacher_call_during_formal"
                ]
                is False
                and by_id[method_id]["privilege"][
                    "privileged_diagnostic_evaluation_on_formal"
                ]
                is True
                for method_id in PRIVILEGED_IDS
            )
            and by_id["finite_horizon_control_oracle"]["privilege"][
                "assumed_model_branching_access"
            ]
            is True
        ),
        "G26_literature_cutoff_queries_and_source_failures_are_explicit": _safe(
            lambda: report["literature_cutoff_inclusive"]
            == config["literature_cutoff_inclusive"]
            and report["literature_search"] == config["literature_search"]
            and len(report["literature_search"]["queries"]) == 6
            and all(
                row["status"] != "SOURCE_UNAVAILABLE"
                or row["raw_result_count"] is None
                and row["count_reason"]
                for row in report["literature_search"]["source_status"]
            )
        ),
        "G27_deduplication_merges_versions_by_doi_arxiv_or_title_author": _safe(
            lambda: report["literature_search"]["deduplication"]
            == config["literature_search"]["deduplication"]
            and literature == expected_literature
            and report["literature_dedup_audit"] == expected_dedup_audit
            and report["literature_dedup_audit"]["pipeline_executed"] is True
            and report["literature_dedup_audit"][
                "input_bibliographic_hit_count"
            ]
            > report["literature_dedup_audit"]["canonical_work_count"]
            and report["literature_dedup_audit"][
                "merged_bibliographic_version_count"
            ]
            == report["literature_dedup_audit"][
                "input_bibliographic_hit_count"
            ]
            - report["literature_dedup_audit"]["canonical_work_count"]
            and len(report["literature_dedup_audit"]["raw_hit_ledger"])
            == report["literature_dedup_audit"][
                "input_bibliographic_hit_count"
            ]
            and len(report["literature_dedup_audit"]["merge_events"])
            == report["literature_dedup_audit"][
                "merged_bibliographic_version_count"
            ]
            and all(
                event["rule"]
                in {"DOI_EXACT", "ARXIV_EXACT", "TITLE_AUTHOR_JACCARD"}
                for event in report["literature_dedup_audit"][
                    "merge_events"
                ]
            )
            and len({row["dedup_key"] for row in literature})
            == len(literature)
            and all(
                row["dedup_key"] == _dedup_key(row)
                and row["doi"] == _normalised_doi(row["doi"])
                and row["raw_record_count"] == len(row["raw_records"]) >= 1
                and row["artifact_record_count"]
                == len(row["artifact_records"])
                for row in literature
            )
            and report["literature_search"]["deduplication"][
                "title_token_jaccard_threshold"
            ]
            == 0.9
            and _title_jaccard(
                "Non-Markovian feedback for optimized quantum error correction",
                "Non Markovian Feedback for Optimized Quantum Error Correction",
            )
            >= 0.9
        ),
        "G28_every_literature_record_has_complete_same_task_screening": _safe(
            lambda: len(literature) == 12
            and all(
                set(row["signature_checks"])
                == {
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
                and isinstance(row["external_same_task_eligible"], bool)
                and row["eligibility_class"]
                and row["exclusion_code"]
                for row in literature
            )
            and sum(
                row["external_same_task_eligible"] for row in literature
            )
            == report["external_claim_contract"][
                "current_external_same_task_eligible_count"
            ]
            == 0
        ),
        "G29_external_exclusions_and_typed_null_reasons_are_nonempty": _safe(
            lambda: all(
                row["typed_null_reasons"]
                and row["numeric_result"] is None
                and row["external_rank"] is None
                and row["external_sota_vote"] is False
                for row in literature
            )
        ),
        "G30_registry_best_never_auto_promotes_external_sota": _safe(
            lambda: report["external_claim_contract"][
                "registry_best_auto_promotes_external_sota"
            ]
            is False
            and report["external_claim_contract"][
                "external_point_only_result"
            ]
            == "CONTEXT_ONLY_NOT_PROMOTION_ELIGIBLE"
            and report["external_claim_contract"][
                "unresolved_eligible_stronger_comparator"
            ]
            == "EXTERNAL_SOTA_NULL"
            and report["external_claim_contract"]["external_sota"] is None
        ),
        "G31_puviani_is_one_work_family_and_remains_officially_blocked": _safe(
            lambda: len(
                [
                    row
                    for row in literature
                    if row["record_id"] == "LIT-PUVIANI-2025"
                ]
            )
            == 1
            and set(
                next(
                    row
                    for row in literature
                    if row["record_id"] == "LIT-PUVIANI-2025"
                )["raw_records"]
            )
            == {
                "ARXIV:2312.07391",
                "DOI:10.1103/PhysRevLett.134.020601",
            }
            and next(
                row
                for row in literature
                if row["record_id"] == "LIT-PUVIANI-2025"
            )["artifact_records"]
            == ["GITHUB:Matteo-Puviani/GQF"]
            and {
                "MISSING_OFFICIAL_CHECKPOINT",
                "MISSING_AGENT_SEED_LEDGER",
                "MISSING_SELECTION_LEDGER",
                "MISSING_SIX_STATE_EVALUATOR",
            }
            <= set(
                next(
                    row
                    for row in literature
                    if row["record_id"] == "LIT-PUVIANI-2025"
                )["typed_null_reasons"]
            )
            and report["external_claim_contract"]["official_puviani_exact"]
            is None
            and report["external_claim_contract"]["puviani_surpass"] is None
            and by_id["paper_constrained_nmf"]["t9_1_3_provenance"][
                "matched_phase9_ranking_eligible"
            ]
            is False
        ),
        "G32_configured_future_outcome_paths_are_absent_at_seal": _safe(
            lambda: report["future_outcome_absence"]
            == _future_absence(config)
            and all(
                row["exists_at_seal"] is False
                for row in report["future_outcome_absence"]
            )
        ),
        "G33_config_code_parent_and_historical_implementations_are_hash_live": _safe(
            lambda: artifact_registry_ok() and implementation_bindings_ok()
        ),
        "G34_source_data_literature_ledger_and_markdown_are_lossless_and_live": _safe(
            outputs_ok
        ),
        "G35_one_targeted_semantic_mutation_per_gate_fails_closed": _safe(
            lambda: report["semantic_mutation_audit"]["count"]
            == report["semantic_mutation_audit"]["detected"]
            == len(GATE_IDS)
            and report["semantic_mutation_audit"]["all_detected"] is True
            and len(report["semantic_mutation_audit"]["cases"])
            == len(GATE_IDS)
            and {
                row["target_gate"]
                for row in report["semantic_mutation_audit"]["cases"]
            }
            == set(GATE_IDS)
            and len(
                {
                    row["mutation_id"]
                    for row in report["semantic_mutation_audit"]["cases"]
                }
            )
            == len(GATE_IDS)
            and all(
                row["mutation_applied"] is True
                and row["rejection_mode"] == "TARGET_GATE_FALSE"
                and row["rejected"] is True
                for row in report["semantic_mutation_audit"]["cases"]
            )
        ),
        "G36_protocol_pass_has_no_numeric_metric_rank_or_performance_verdict": _safe(
            lambda: all_current_results_null()
            and all(
                row["numeric_result"] is None
                and row["external_rank"] is None
                for row in literature
            )
            and report["performance_state"]
            == {
                "round_ler": None,
                "six_state_lifetime": None,
                "registered_best": None,
                "external_sota": None,
                "protocol_only": True,
            }
        ),
    }
    if tuple(gates) != GATE_IDS:
        raise AssertionError("gate order drifted")
    return gates


def _semantic_mutation_audit(
    report: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    cached_parent = copy.deepcopy(report["parent_handoff"])
    cached_artifacts = copy.deepcopy(report["artifact_registry"])
    cached_implementations = copy.deepcopy(
        report["implementation_bindings"]
    )
    baseline_gates = evaluate_gates(
        report,
        config_path=config_path,
        check_live_files=False,
        expected_parent_handoff=cached_parent,
        expected_artifact_registry=cached_artifacts,
        expected_implementation_bindings=cached_implementations,
    )
    baseline_failed = [
        gate for gate, passed in baseline_gates.items() if not passed
    ]
    if baseline_failed:
        raise AssertionError(
            "mutation baseline is already invalid: "
            + ", ".join(baseline_failed)
        )

    def attempt(
        mutation_id: str,
        target_gate: str,
        change: Callable[[dict[str, Any]], None],
    ) -> None:
        candidate = copy.deepcopy(dict(report))
        candidate["semantic_mutation_audit"] = _valid_mutation_placeholder()
        try:
            change(candidate)
        except Exception as error:
            raise AssertionError(
                f"mutation {mutation_id} could not be applied"
            ) from error
        gates = evaluate_gates(
            candidate,
            config_path=config_path,
            check_live_files=False,
            expected_parent_handoff=cached_parent,
            expected_artifact_registry=cached_artifacts,
            expected_implementation_bindings=cached_implementations,
        )
        cases.append(
            {
                "mutation_id": mutation_id,
                "target_gate": target_gate,
                "mutation_applied": True,
                "rejection_mode": "TARGET_GATE_FALSE",
                "rejected": gates[target_gate] is False,
            }
        )

    attempt(
        "change_protocol_identity",
        GATE_IDS[0],
        lambda value: value.update(protocol_id="DEMO"),
    )
    attempt(
        "forge_parent_semantic_seal",
        GATE_IDS[1],
        lambda value: value["parent_handoff"][
            "downstream_semantic_seal"
        ].update(semantic_sha256="0" * 64),
    )
    attempt(
        "delete_no_go_parent_release",
        GATE_IDS[2],
        lambda value: value["parent_terminal_options"].pop(),
    )
    attempt(
        "fill_puviani_surpass",
        GATE_IDS[3],
        lambda value: value["external_claim_contract"].update(
            puviani_surpass={"gain": 1.01}
        ),
    )
    attempt(
        "rank_hil_with_algorithm",
        GATE_IDS[4],
        lambda value: value["task_signature_scope"]["ranked_lanes"].append(
            "RAW_IQ_DIGITAL_HIL"
        ),
    )
    attempt(
        "delete_mandatory_imm",
        GATE_IDS[5],
        lambda value: value["methods"].remove(
            next(
                row
                for row in value["methods"]
                if row["method_id"] == "interacting_multiple_model_filter"
            )
        ),
    )
    attempt(
        "merge_autonomous_into_no_correction",
        GATE_IDS[6],
        lambda value: next(
            row
            for row in value["methods"]
            if row["method_id"] == "autonomous_sbs_no_measurement"
        ).update(method_id="no_correction_idle_memory"),
    )
    attempt(
        "rank_capacity_ceiling",
        GATE_IDS[7],
        lambda value: next(
            row
            for row in value["ranking_lanes"]
            if row["lane_id"]
            == "UNBUDGETED_CAPACITY_CEILING_NONRANKING"
        ).update(ranked=True),
    )
    attempt(
        "promote_unqualified_method",
        GATE_IDS[8],
        lambda value: next(
            row
            for row in value["methods"]
            if row["method_id"] == "paper_constrained_nmf"
        )["current_result"].update(matched_eligible=True, rank=1),
    )
    attempt(
        "give_gru_hidden_truth",
        GATE_IDS[9],
        lambda value: next(
            row
            for row in value["methods"]
            if row["method_id"] == "gru_posterior"
        )["privilege"].update(hidden_truth_access=True),
    )
    attempt(
        "grant_transformer_extra_iq_stratum",
        GATE_IDS[10],
        lambda value: next(
            row
            for row in value["methods"]
            if row["method_id"] == "causal_transformer_posterior"
        )["common_contract_ids"].update(task_signature="EXTRA_IQ"),
    )
    attempt(
        "fill_unqualified_codebook",
        GATE_IDS[11],
        lambda value: value["task_signature_scope"].update(
            current_codebook_id="DEMO_CODEBOOK"
        ),
    )
    attempt(
        "allow_online_gpu",
        GATE_IDS[12],
        lambda value: value["matched_compute_contract"].update(
            gpu_allowed_online=True
        ),
    )
    attempt(
        "exclude_transfer_from_deadline",
        GATE_IDS[13],
        lambda value: value["matched_compute_contract"][
            "deadline_includes"
        ].remove("host_to_bank_transfer"),
    )
    attempt(
        "hide_selection_attempt_fields",
        GATE_IDS[14],
        lambda value: value["selection_contract"][
            "all_attempts_required_fields"
        ].remove("failure_status"),
    )
    attempt(
        "select_checkpoint_on_formal",
        GATE_IDS[15],
        lambda value: value["selection_contract"].update(
            formal_access_for_selection="ALLOWED"
        ),
    )
    attempt(
        "open_pilot_twice",
        GATE_IDS[16],
        lambda value: value["pilot_contract"].update(open_count=2),
    )
    attempt(
        "weaken_lcb_claim_threshold",
        GATE_IDS[17],
        lambda value: value["formal_power_plan"].update(
            claim_relative_lcb_threshold=0.0
        ),
    )
    attempt(
        "reuse_old_sixty_cluster_design",
        GATE_IDS[18],
        lambda value: value["power_analysis"].update(
            planned_clusters_per_backend=60
        ),
    )
    attempt(
        "resample_rounds_as_independent",
        GATE_IDS[19],
        lambda value: value["statistics_contract"].update(
            resampling_unit="round"
        ),
    )
    attempt(
        "replace_maxT_with_pointwise_ci",
        GATE_IDS[20],
        lambda value: value["statistics_contract"].update(
            method="pointwise_bootstrap"
        ),
    )
    attempt(
        "epsilon_zero_denominator",
        GATE_IDS[21],
        lambda value: value["statistics_contract"].update(
            zero_baseline_ler_rule="add epsilon"
        ),
    )
    attempt(
        "drop_timeout_status",
        GATE_IDS[22],
        lambda value: value["missingness_contract"]["status_enum"].remove(
            "TIMEOUT"
        ),
    )
    attempt(
        "ceiling_votes_registered_best",
        GATE_IDS[23],
        lambda value: next(
            row
            for row in value["methods"]
            if row["method_id"] == "large_transformer_ceiling"
        )["current_result"].update(registered_best_vote=True),
    )
    attempt(
        "oracle_enters_rank",
        GATE_IDS[24],
        lambda value: next(
            row
            for row in value["methods"]
            if row["method_id"] == "hidden_state_decoder_oracle"
        )["current_result"].update(rank=1),
    )
    attempt(
        "erase_search_cutoff",
        GATE_IDS[25],
        lambda value: value.update(literature_cutoff_inclusive=None),
    )
    attempt(
        "duplicate_puviani_as_second_comparator",
        GATE_IDS[26],
        lambda value: value["literature_records"].append(
            copy.deepcopy(
                next(
                    row
                    for row in value["literature_records"]
                    if row["record_id"] == "LIT-PUVIANI-2025"
                )
            )
        ),
    )
    attempt(
        "promote_external_without_same_task",
        GATE_IDS[27],
        lambda value: next(
            row
            for row in value["literature_records"]
            if row["record_id"] == "LIT-SIVAK-2023"
        ).update(external_same_task_eligible=True),
    )
    attempt(
        "erase_external_null_reason",
        GATE_IDS[28],
        lambda value: next(
            row
            for row in value["literature_records"]
            if row["record_id"] == "LIT-WAN-2020"
        ).update(typed_null_reasons=[]),
    )
    attempt(
        "auto_promote_registry_best",
        GATE_IDS[29],
        lambda value: value["external_claim_contract"].update(
            registry_best_auto_promotes_external_sota=True
        ),
    )
    attempt(
        "claim_puviani_official_assets_complete",
        GATE_IDS[30],
        lambda value: next(
            row
            for row in value["literature_records"]
            if row["record_id"] == "LIT-PUVIANI-2025"
        ).update(typed_null_reasons=[]),
    )
    attempt(
        "pretend_formal_outcome_existed_at_seal",
        GATE_IDS[31],
        lambda value: value["future_outcome_absence"][0].update(
            exists_at_seal=True
        ),
    )
    attempt(
        "forge_implementation_binding",
        GATE_IDS[32],
        lambda value: value["implementation_bindings"][0].update(
            sha256="0" * 64
        ),
    )
    attempt(
        "forge_source_row_count",
        GATE_IDS[33],
        lambda value: value["source_data"].update(rows=0),
    )
    attempt(
        "forge_mutation_count",
        GATE_IDS[34],
        lambda value: value.update(
            semantic_mutation_audit={
                **_valid_mutation_placeholder(),
                "detected": len(GATE_IDS) - 1,
                "all_detected": False,
            }
        ),
    )
    attempt(
        "insert_demo_ler_result",
        GATE_IDS[35],
        lambda value: value["performance_state"].update(round_ler=0.01),
    )
    return {
        "count": len(cases),
        "detected": sum(row["rejected"] for row in cases),
        "all_detected": all(row["rejected"] for row in cases),
        "cases": cases,
    }


def _analysis_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "task_id",
        "schema_version",
        "config_schema_version",
        "protocol_id",
        "seal_state",
        "frozen_at",
        "literature_cutoff_inclusive",
        "parent_handoff",
        "parent_terminal_options",
        "task_signature_scope",
        "ranking_lanes",
        "matched_compute_contract",
        "selection_contract",
        "pilot_contract",
        "formal_power_plan",
        "power_analysis",
        "statistics_contract",
        "missingness_contract",
        "methods",
        "literature_search",
        "literature_dedup_audit",
        "literature_records",
        "external_claim_contract",
        "future_outcome_absence",
        "artifact_registry",
        "implementation_bindings",
        "performance_state",
    )
    return {key: report[key] for key in keys}


def _render_markdown(report: Mapping[str, Any]) -> str:
    power = report["power_analysis"]
    gate_summary = report["gate_summary"]
    power_plan = report["formal_power_plan"]
    dedup = report["literature_dedup_audit"]
    matched = [
        row
        for row in report["methods"]
        if row["registry_class"] == "MATCHED_DEPLOYABLE_RANKED"
    ]
    ceiling = [
        row
        for row in report["methods"]
        if row["registry_class"]
        == "UNBUDGETED_CAPACITY_CEILING_NONRANKING"
    ]
    privileged = [
        row
        for row in report["methods"]
        if row["registry_class"]
        == "PRIVILEGED_UPPER_BOUND_NONRANKING"
    ]
    anchors = [
        row
        for row in report["methods"]
        if row["registry_class"] == "PROTOCOL_ANCHOR_NONRANKING"
    ]

    def method_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
        output = [
            "| ID | family / role | historical state | current Phase-9 state | boundary |",
            "| --- | --- | --- | --- | --- |",
        ]
        output.extend(
            "| `{}` | {} / {} | `{}` | `{}`; metrics/rank=`null` | {} |".format(
                row["method_id"],
                row["family"],
                row["role"],
                row["historical_state"],
                row["current_qualification_state"],
                row["claim_boundary"],
            )
            for row in rows
        )
        return output

    lines = [
        "# T9.1.4 Phase-9 baseline、检索与统计功效注册表",
        "",
        "## 1. 结论",
        "",
        f"机器合同 verdict=`{report['verdict']}`，通过 "
        f"{gate_summary['passed']}/{gate_summary['total']} gates，并杀死 "
        f"{report['semantic_mutation_audit']['detected']}/"
        f"{report['semantic_mutation_audit']['count']} 个针对性语义 mutation。"
        "这只是 `SEALED_PRE_OUTCOME` 协议，不是性能实验结果：所有 LER、lifetime、"
        "rank、registered-best、external-SOTA、official Puviani、physical lifetime "
        "字段仍为 `null`。",
        "",
        "T9.1.3 的 one-way semantic handoff 已按“先 live validate、再重建 canonical "
        "payload、最后比较 seal”的顺序消费；当前 branch 为 "
        f"`{report['parent_handoff']['terminal_branch']}`，但其 "
        "`matched_phase9_ranking_eligible=false`。NO_GO typed-null branch 同样释放本任务，"
        "不会被删行。",
        "",
        "## 2. Matched-deployable 目标榜",
        "",
        *method_table(matched),
        "",
        f"其中 mandatory baseline 恰为 {len(MANDATORY_BASELINE_IDS)} 个。"
        "旧 production/PASS 只证明历史 task 的实现；raw/recorded IQ、同 action/codebook、"
        "同 precision、同 256-cycle cadence、同 1 ms 全链 wall-clock 和同 selection "
        "budget 未重资格前，任何一行都不能排名。",
        "",
        "primary promotion 输入固定为 T9.2.7 同一 bit-accurate frontend 的 "
        "`FIXED_POINT_DERIVED_LLR`；raw IQ 与 binary syndrome 只作分别闭合的输入消融。"
        "paired common randomness 只绑定六态初态、device/scenario draw 与外生 "
        "innovation manifest；每个 policy 执行 action 后必须产生并保存自己的 "
        "transition/trace hash，禁止把同一完整物理轨迹重放给所有 controller。",
        "",
        "## 3. Unbudgeted capacity ceiling",
        "",
        *method_table(ceiling),
        "",
        "这些方法必须使用相同 observation、action 和 formal trace，但允许更大模型、"
        "更多 state/context、GPU 或更多 selection compute；因此 `rank=null`、"
        "`registered_best_vote=false`、`external_sota_vote=false`。",
        "",
        "## 4. Privileged upper bound",
        "",
        *method_table(privileged),
        "",
        "hidden-state teacher/oracle、future-suffix smoother 和 assumed-model short-horizon "
        "control tree 永不进入 deployable 排名。删除 teacher 后，formal deployable action "
        "必须逐 bit 不变。",
        "",
        "## 5. Protocol anchors",
        "",
        *method_table(anchors),
        "",
        "`standard_measurement_feedback_sbs`、`autonomous_sbs_no_measurement` 和 "
        "`no_correction_idle_memory` 是三种不同的 measurement/reset/action/wall-clock "
        "协议，不能用一个含糊的 “sBs/no-feedback” 行合并。",
        "",
        "## 6. Compute、selection 与 failure 合同",
        "",
        "- online：batch=1、单 pinned physical core、1 thread、GPU disabled；"
        "每 256 cycles 更新一次，1,000,000 ns deadline 包含 preprocessing、feature、"
        "inference、state update、serialization 和 transfer；timeout 只导致不更新，"
        "六周期 LKG fast path 不被阻塞。",
        "- envelope：131,072 parameters、1 MiB persistent state、16 MiB workspace、"
        "256 MiB peak RSS delta、262,144 online MAC/update、1,048,576 FLOP/update。"
        "frontend precision 和 codebook hash 仍为 typed null，由 T9.2.7/T9.3.4 在 pilot 前填充；"
        "未填时 matched eligibility 必为 false。",
        "- 每 family 最多 64 个 selection evaluations；learning 最多 16 configs × 4 "
        "restarts。所有 checkpoint、失败、timeout/OOM 与未选中 run 均保留，"
        "`best-of-N` 无完整 ledger 直接不合格。",
        "- failure status 包含 COMPLETE、DEADLINE_FALLBACK、TIMEOUT、OOM、"
        "NUMERIC_FAILURE、CORRUPT_SHARD、MISSING_TOOL、EXTERNAL_UNAVAILABLE、"
        "SIGNATURE_INELIGIBLE、TYPED_NULL_NO_GO；禁止 complete-case deletion 和 zero "
        "imputation。",
        "",
        "## 7. Pilot 与 formal 功效",
        "",
        "- pilot：128 个独立 clusters，8 个 macro scenario × 16；只打开一次，"
        "最多晋升一个 proposed candidate，不选择或删除 baseline，也不支持论文 claim。",
        f"- formal：每个 physics backend 独立 "
        f"{power_plan['formal_cluster_count_per_backend']} clusters，"
        f"{power_plan['macro_scenario_families']} × "
        f"{power_plan['clusters_per_macro_family_per_backend']}；每个 cluster "
        "联合保留六态、全部方法、round/window 和 reset/leakage/control/fallback/timeout。",
        "- 设计目标不是“优于 0”：假定 baseline p_L floor=0.08、真实相对改善 15%，"
        "必须把 simultaneous relative LCB 推过 10%。",
        f"- 固定 {power_plan['fixed_comparator_divisor']} comparator divisor 下 "
        f"required N={power['required_clusters_at_design_floor']}，"
        f"planned N={power['planned_clusters_per_backend']}，approximate design power="
        f"{power['achieved_power_at_design_floor']:.6f}。最终推断仍使用 100,000 次 "
        "paired outer-cluster maxT bootstrap。",
        "- 上述 90% 是 round-LER comparator screen 的设计功效，不是 lifetime 或整个 "
        "multi-endpoint family 的 joint-power 声明；六态 survival/lifetime 功效在 "
        "T9.6.1 取得 event-rate proxy 前保持 typed `null`，最终 maxT 仍控制三组闭合 "
        "family 的错误率。",
        "- pilot 后不按方差、方向或显著性扩样；不足时输出 `UNDERPOWERED_FIXED_N`，"
        "不能补 seed。",
        "",
        "## 8. 文献检索、去重与 same-task eligibility",
        "",
        f"检索截止日为 `{report['literature_cutoff_inclusive']}`。保存 6 条原始检索式；"
        "arXiv、出版商/DOI 和官方仓库完成 targeted primary verification。当前环境没有 "
        "structured CrossRef/Semantic Scholar connector，因此返回数保持 `null` 并记录 "
        "`SOURCE_UNAVAILABLE`，没有伪造 hit count。",
        f"bibliographic raw hits={dedup['input_bibliographic_hit_count']}，"
        f"canonical works={dedup['canonical_work_count']}，"
        f"merged bibliographic versions="
        f"{dedup['merged_bibliographic_version_count']}；另有 "
        f"{dedup['artifact_evidence_count']} 条 repository/local/dataset 证据，"
        "只附着到 work family，不计作检索 hit 或额外 comparator。",
        "",
        "| record | work | eligibility | exclusion / typed-null |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["literature_records"]:
        lines.append(
            f"| `{row['record_id']}` | {row['title']} ({row['year']}) | "
            f"`{row['eligibility_class']}` | `{row['exclusion_code']}`; "
            f"{', '.join(row['typed_null_reasons'])} |"
        )
    lines.extend(
        [
            "",
            "DOI 先规范化后精确去重；无 DOI 时依次用 arXiv、规范化 title + first "
            "author，title token Jaccard 门为 0.90。Puviani 的 PRL、arXiv 和 GQF "
            "只构成一个 work family。逐条筛选 ledger 位于 "
            "`docs/t9_1_4_literature_search_ledger.csv`。",
            "",
            "截至 cutoff，已核验外部工作没有一项同时满足本项目的 input、history、"
            "trusted action、physics backend、online timing、no-postselection denominator、"
            "metric 和 compute signature；所以 external-SOTA 不是 false/negative，而是 "
            "`null`。允许的未来措辞至多是 “best among preregistered matched-deployable "
            "baselines under the frozen T9 task signature”。",
            "",
            "## 9. 反简化审计",
            "",
            "- 删除 mandatory baseline、把 IMM 改名为 HMM、或隐藏 missing implementation "
            "均失败；",
            "- oracle 入榜、future suffix、hidden truth、额外 IQ/history/action 权限均失败；",
            "- 省略 preprocessing/transfer、放宽 GPU/thread/deadline、隐藏 best-of-N 或 "
            "timeout/OOM 均失败；",
            "- 用 60 clusters、round-level pseudoreplication、pointwise CI、缩小 divisor、"
            "zero-denominator epsilon 或 pilot 后扩样均失败；",
            "- 把 registry-best 自动升级 external SOTA、把 T9.1.3 自动升级 matched、"
            "填入 official/Puviani/physical claim 或把同一论文多算 comparator 均失败；",
            f"- 完整 audit 为 `{report['semantic_mutation_audit']['detected']}/"
            f"{report['semantic_mutation_audit']['count']}`。",
            "",
            "## 10. 复现",
            "",
            "```powershell",
            "python -m cnn_fpga.benchmark.phase9_baseline_search_power_registry",
            "python -m cnn_fpga.benchmark.phase9_baseline_search_power_registry --verify",
            "python -m pytest -q tests/test_phase9_baseline_search_power_registry.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    *,
    config_path: Path = DEFAULT_CONFIG,
    source_data_path: Path = DEFAULT_SOURCE_DATA,
    literature_ledger_path: Path = DEFAULT_LITERATURE_LEDGER,
    markdown_path: Path = DEFAULT_MARKDOWN,
) -> dict[str, Any]:
    config = _load(config_path)
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("task_id") != TASK_ID
        or config.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("T9.1.4 config identity mismatch")
    if any((ROOT / path).exists() for path in config["future_outcome_paths"]):
        raise ValueError("cannot seal T9.1.4 after registered outcome access")

    parent_handoff = _consume_parent_handoff()
    methods = _enrich_methods(config, parent_handoff)
    literature, literature_dedup_audit = (
        _deduplicate_literature_records(config)
    )
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "config_schema_version": config["schema_version"],
        "protocol_id": config["protocol_id"],
        "seal_state": config["seal_state"],
        "frozen_at": config["frozen_at"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "literature_cutoff_inclusive": config[
            "literature_cutoff_inclusive"
        ],
        "parent_handoff": parent_handoff,
        "parent_terminal_options": _parent_terminal_options(),
        "task_signature_scope": copy.deepcopy(
            config["task_signature_scope"]
        ),
        "ranking_lanes": copy.deepcopy(config["ranking_lanes"]),
        "matched_compute_contract": copy.deepcopy(
            config["matched_compute_contract"]
        ),
        "selection_contract": copy.deepcopy(config["selection_contract"]),
        "pilot_contract": copy.deepcopy(config["pilot_contract"]),
        "formal_power_plan": copy.deepcopy(config["formal_power_plan"]),
        "power_analysis": _power_analysis(config),
        "statistics_contract": copy.deepcopy(
            config["statistics_contract"]
        ),
        "missingness_contract": copy.deepcopy(
            config["missingness_contract"]
        ),
        "methods": methods,
        "literature_search": copy.deepcopy(config["literature_search"]),
        "literature_dedup_audit": literature_dedup_audit,
        "literature_records": literature,
        "external_claim_contract": copy.deepcopy(
            config["external_claim_contract"]
        ),
        "future_outcome_absence": _future_absence(config),
        "artifact_registry": _artifact_registry(config_path),
        "implementation_bindings": _implementation_bindings(methods),
        "performance_state": {
            "round_ler": None,
            "six_state_lifetime": None,
            "registered_best": None,
            "external_sota": None,
            "protocol_only": True,
        },
        "source_data": {
            "path": _relative(source_data_path),
            "rows": 0,
        },
        "literature_ledger": {
            "path": _relative(literature_ledger_path),
            "rows": len(literature),
        },
        "markdown": {"path": _relative(markdown_path)},
        "semantic_mutation_audit": _valid_mutation_placeholder(),
        "gates": {},
        "gate_summary": {},
        "verdict": None,
        "analysis_sha256": "",
    }
    report["source_data"]["rows"] = len(_source_rows(report))
    report["semantic_mutation_audit"] = _semantic_mutation_audit(
        report, config_path=config_path
    )

    # Prove every non-output gate live before publishing a Markdown PASS.
    # G34 is allowed to be false here because its three outputs have not yet
    # been materialised.  This avoids a failed build leaving a document that
    # falsely claims all gates passed.
    live_prepublication = evaluate_gates(
        report, config_path=config_path, check_live_files=True
    )
    unexpected_prepublication_failures = [
        key
        for key, passed in live_prepublication.items()
        if not passed
        and key
        != "G34_source_data_literature_ledger_and_markdown_are_lossless_and_live"
    ]
    if unexpected_prepublication_failures:
        raise ValueError(
            "T9.1.4 prepublication gates failed: "
            + ", ".join(unexpected_prepublication_failures)
        )
    prospective_gates = dict(live_prepublication)
    prospective_gates[
        "G34_source_data_literature_ledger_and_markdown_are_lossless_and_live"
    ] = True
    report["gates"] = prospective_gates
    report["gate_summary"] = {
        "passed": sum(prospective_gates.values()),
        "total": len(prospective_gates),
        "failed": [
            key for key, passed in prospective_gates.items() if not passed
        ],
    }
    report["verdict"] = VERDICT

    _write_source_data(report, source_data_path)
    _write_literature_ledger(report, literature_ledger_path)
    report["source_data"] = {
        **_binding(source_data_path),
        "rows": len(_source_rows(report)),
    }
    report["literature_ledger"] = {
        **_binding(literature_ledger_path),
        "rows": len(_literature_ledger_rows(report)),
    }
    _atomic_text(_render_markdown(report), markdown_path)
    report["markdown"] = _binding(markdown_path)
    report["gates"] = evaluate_gates(
        report, config_path=config_path, check_live_files=True
    )
    failed = [key for key, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {
        "passed": len(report["gates"]) - len(failed),
        "total": len(report["gates"]),
        "failed": failed,
    }
    report["verdict"] = (
        VERDICT if not failed else "FAIL_T9_1_4_BASELINE_SEARCH_POWER_REGISTRY"
    )
    if failed:
        _atomic_text(
            "# T9.1.4 生成失败\n\n"
            "最终 live gate 未通过，本文件不构成协议冻结或性能证据。"
            "失败 gates："
            + ", ".join(failed)
            + "\n",
            markdown_path,
        )
        raise ValueError(
            "T9.1.4 final live gates failed: " + ", ".join(failed)
        )
    report["analysis_sha256"] = _canonical_sha256(
        _analysis_payload(report)
    )
    return report


def verify_report(
    report_or_path: Mapping[str, Any] | str | Path = DEFAULT_REPORT,
) -> dict[str, bool]:
    report = (
        copy.deepcopy(dict(report_or_path))
        if isinstance(report_or_path, Mapping)
        else _load(Path(report_or_path))
    )
    config_path = ROOT / report["artifact_registry"]["config"]["path"]
    expected_mutations = _semantic_mutation_audit(
        report, config_path=config_path
    )
    gates = evaluate_gates(report, config_path=config_path)
    expected_summary = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "failed": [key for key, passed in gates.items() if not passed],
    }
    checks = {
        "identity": report.get("task_id") == TASK_ID
        and report.get("schema_version") == REPORT_SCHEMA_VERSION,
        "parent_handoff_live": all(
            report["parent_handoff"]["ordered_consumer_checks"].values()
        ),
        "mutation_replay": report["semantic_mutation_audit"]
        == expected_mutations,
        "all_gates": all(gates.values()),
        "gate_cache": report["gates"] == gates
        and report["gate_summary"] == expected_summary,
        "verdict": report["verdict"] == VERDICT,
        "analysis_sha256": report["analysis_sha256"]
        == _canonical_sha256(_analysis_payload(report)),
        "source_data": _csv_lossless(
            report, ROOT / report["source_data"]["path"]
        ),
        "literature_ledger": _literature_csv_lossless(
            report, ROOT / report["literature_ledger"]["path"]
        ),
        "performance_null": report["performance_state"]
        == {
            "round_ler": None,
            "six_state_lifetime": None,
            "registered_best": None,
            "external_sota": None,
            "protocol_only": True,
        },
    }
    if not all(checks.values()):
        raise ValueError(
            "T9.1.4 verification failed: "
            + ", ".join(key for key, passed in checks.items() if not passed)
            + "; failed_gates="
            + repr([key for key, passed in gates.items() if not passed])
        )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--source-data", type=Path, default=DEFAULT_SOURCE_DATA
    )
    parser.add_argument(
        "--literature-ledger", type=Path, default=DEFAULT_LITERATURE_LEDGER
    )
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.verify:
        print(
            json.dumps(
                verify_report(arguments.report),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    report = build_report(
        config_path=arguments.config,
        source_data_path=arguments.source_data,
        literature_ledger_path=arguments.literature_ledger,
        markdown_path=arguments.markdown,
    )
    if report["verdict"] != VERDICT or not all(report["gates"].values()):
        print(
            json.dumps(
                {
                    "verdict": report["verdict"],
                    "failed": report["gate_summary"]["failed"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    _atomic_json(report, arguments.report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "gates": report["gate_summary"],
                "mutations": {
                    "detected": report["semantic_mutation_audit"][
                        "detected"
                    ],
                    "count": report["semantic_mutation_audit"]["count"],
                },
                "analysis_sha256": report["analysis_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
