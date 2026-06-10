"""Build the bounded FR8 statcalib closure pack from preserved artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_DOC = REPO_ROOT / "docs" / "P4_benchmark_formal_protocol.md"
DEFAULT_T64_REPORT = REPO_ROOT / "docs" / "fr8_statcalib_extension_lane_benchmark.md"
DEFAULT_T65_REPORT = REPO_ROOT / "docs" / "fr8_statcalib_extension_lane_consistency_audit.md"
DEFAULT_T66_REPORT = REPO_ROOT / "docs" / "statcalib_sensitivity_bounded_benchmark.md"
DEFAULT_T67_REPORT = REPO_ROOT / "docs" / "statcalib_teacher_anchor_bounded_benchmark.md"
DEFAULT_T68_REPORT = REPO_ROOT / "docs" / "statcalib_generated_only_robustness_bounded_benchmark.md"
DEFAULT_T69_REPORT = REPO_ROOT / "docs" / "statcalib_clean_winner_tiebreak_bounded_benchmark.md"

DEFAULT_T64_REVIEW = REPO_ROOT / "docs" / "review" / "T64_review.md"
DEFAULT_T65_REVIEW = REPO_ROOT / "docs" / "review" / "T65_review.md"
DEFAULT_T66_REVIEW = REPO_ROOT / "docs" / "review" / "T66_review.md"
DEFAULT_T67_REVIEW = REPO_ROOT / "docs" / "review" / "T67_review.md"
DEFAULT_T68_REVIEW = REPO_ROOT / "docs" / "review" / "T68_review.md"
DEFAULT_T69_REVIEW = REPO_ROOT / "docs" / "review" / "T69_review.md"

DEFAULT_T24_RUN_DIR = (
    REPO_ROOT / "runs" / "p4_benchmark" / "T24_formal_software_revalidation_20260510_200743"
)
DEFAULT_T64_RUN_DIR = (
    REPO_ROOT / "runs" / "p4_benchmark" / "T64_fr8_statcalib_extension_lane_20260527_221658"
)
DEFAULT_T66_RUN_DIR = (
    REPO_ROOT / "runs" / "p4_benchmark" / "T66_statcalib_sensitivity_20260529_210906"
)
DEFAULT_T67_RUN_DIR = (
    REPO_ROOT / "runs" / "p4_benchmark" / "T67_statcalib_teacher_anchor_20260601_225718"
)
DEFAULT_T68_RUN_DIR = (
    REPO_ROOT / "runs" / "p4_benchmark" / "T68_statcalib_generated_only_20260605_205723"
)
DEFAULT_T69_RUN_DIR = (
    REPO_ROOT / "runs" / "p4_benchmark" / "T69_statcalib_clean_winner_tiebreak_20260608_160358"
)

DEFAULT_T66_SUMMARY_JSON = (
    DEFAULT_T66_RUN_DIR / "statcalib_sensitivity_summary" / "summary.json"
)
DEFAULT_T67_SUMMARY_JSON = (
    DEFAULT_T67_RUN_DIR / "statcalib_teacher_anchor_summary" / "summary.json"
)
DEFAULT_T68_SUMMARY_JSON = (
    DEFAULT_T68_RUN_DIR / "statcalib_generated_only_summary" / "summary.json"
)
DEFAULT_T69_SUMMARY_JSON = (
    DEFAULT_T69_RUN_DIR / "statcalib_clean_winner_tiebreak_summary" / "summary.json"
)

EXPECTED_SCENARIOS = [
    "static_bias_theta",
    "linear_ramp",
    "step_sigma_theta",
    "periodic_drift",
]
EXPECTED_T24_MODES = [
    "ekf",
    "ukf",
    "constant_residual_mu",
    "rls_residual_b",
    "hybrid_residual_b",
]
EXPECTED_T69_CLEAN_SET = [
    "statcalib_window_variance_t001",
    "statcalib_window_variance_t003",
    "statcalib_window_variance_t005",
]
EXPECTED_T68_FULL_GENERATED_ONLY_WINNERS = [
    "statcalib_window_variance_t001",
    "statcalib_window_variance_t003",
    "statcalib_window_variance_t005",
    "statcalib_ekf_t001",
]
ALLOWED_REVIEW_VERDICTS = {"PASS", "PASS_WITH_WARNINGS"}
VERDICT_PATTERN = re.compile(r"(?i)verdict[\s:`\-]*?(PASS_WITH_WARNINGS|PASS|BLOCK)")


@dataclass(frozen=True)
class ClosureInputs:
    protocol_doc: Path = DEFAULT_PROTOCOL_DOC
    t64_report: Path = DEFAULT_T64_REPORT
    t65_report: Path = DEFAULT_T65_REPORT
    t66_report: Path = DEFAULT_T66_REPORT
    t67_report: Path = DEFAULT_T67_REPORT
    t68_report: Path = DEFAULT_T68_REPORT
    t69_report: Path = DEFAULT_T69_REPORT
    t64_review: Path = DEFAULT_T64_REVIEW
    t65_review: Path = DEFAULT_T65_REVIEW
    t66_review: Path = DEFAULT_T66_REVIEW
    t67_review: Path = DEFAULT_T67_REVIEW
    t68_review: Path = DEFAULT_T68_REVIEW
    t69_review: Path = DEFAULT_T69_REVIEW
    t24_run_dir: Path = DEFAULT_T24_RUN_DIR
    t64_run_dir: Path = DEFAULT_T64_RUN_DIR
    t66_run_dir: Path = DEFAULT_T66_RUN_DIR
    t67_run_dir: Path = DEFAULT_T67_RUN_DIR
    t68_run_dir: Path = DEFAULT_T68_RUN_DIR
    t69_run_dir: Path = DEFAULT_T69_RUN_DIR
    t66_summary_json: Path = DEFAULT_T66_SUMMARY_JSON
    t67_summary_json: Path = DEFAULT_T67_SUMMARY_JSON
    t68_summary_json: Path = DEFAULT_T68_SUMMARY_JSON
    t69_summary_json: Path = DEFAULT_T69_SUMMARY_JSON


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default=None, help="Optional path to write the closure-pack JSON.")
    parser.add_argument("--t69-summary-json", default=None, help="Optional override for the preserved T69 summary JSON.")
    return parser


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _require_phrase(path: Path, *phrases: str) -> None:
    text = _read_text(path)
    for phrase in phrases:
        if not _contains_phrase(text, phrase):
            raise ValueError(f"{path} is missing required phrase: {phrase}")


def _parse_review_verdict(path: Path) -> str:
    text = _read_text(path)
    match = VERDICT_PATTERN.search(text)
    if match is None:
        verdict_section = text.lower().find("verdict")
        if verdict_section != -1:
            snippet = text[verdict_section : verdict_section + 200]
            fallback = re.search(r"(PASS_WITH_WARNINGS|PASS|BLOCK)", snippet)
            if fallback is not None:
                return fallback.group(1)
        raise ValueError(f"Could not parse verdict from {path}")
    return match.group(1)


def _validate_review_chain(inputs: ClosureInputs) -> dict[str, str]:
    verdicts = {
        "T64": _parse_review_verdict(_resolve(inputs.t64_review)),
        "T65": _parse_review_verdict(_resolve(inputs.t65_review)),
        "T66": _parse_review_verdict(_resolve(inputs.t66_review)),
        "T67": _parse_review_verdict(_resolve(inputs.t67_review)),
        "T68": _parse_review_verdict(_resolve(inputs.t68_review)),
        "T69": _parse_review_verdict(_resolve(inputs.t69_review)),
    }
    for task_id, verdict in verdicts.items():
        _require(
            verdict in ALLOWED_REVIEW_VERDICTS,
            f"{task_id} review verdict {verdict} is not an accepted closure input",
        )
    return verdicts


def _validate_boundary_docs(inputs: ClosureInputs) -> None:
    _require_phrase(
        _resolve(inputs.protocol_doc),
        "frozen `T24` ranking set by default",
        "separately labeled extension lane",
    )
    _require_phrase(
        _resolve(inputs.t64_report),
        "mock-backed software-HIL only",
        "separate extension lane only",
        "not a rewrite of `T24`",
    )
    _require_phrase(
        _resolve(inputs.t65_report),
        "bounded extension-lane artifact",
        "not a rewrite of `T24`",
    )
    _require_phrase(
        _resolve(inputs.t66_report),
        "mock-backed software-HIL only",
        "separately labeled extension lane only",
    )
    _require_phrase(
        _resolve(inputs.t67_report),
        "mock-backed software-HIL only",
        "separately labeled extension lane only",
    )
    _require_phrase(
        _resolve(inputs.t68_report),
        "mock-backed software-HIL only",
        "separately labeled extension lane only",
    )
    _require_phrase(
        _resolve(inputs.t69_report),
        "mock-backed software-HIL only",
        "separately labeled extension lane only",
        "persistent_clean_tie_set",
    )


def _normalize_comparison_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "scenario": str(row["scenario"]),
        "mode": str(row["mode"]),
        "completed_repeats": int(row["completed_repeats"]),
        "expected_repeats": int(row["expected_repeats"]),
        "coverage": float(row["coverage"]),
        "final_ler_mean": float(row["final_ler_mean"]),
        "overflow_rate_mean": float(row["overflow_rate_mean"]),
    }


def _validate_t24(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(summary.get("missing_runs", []) == [], f"T24 missing_runs is not empty: {summary.get('missing_runs')}")
    _require(len(rows) == 20, f"T24 comparison row count {len(rows)} is not 20")

    index = {(row["scenario"], row["mode"]): row for row in rows}
    for scenario in EXPECTED_SCENARIOS:
        for mode in EXPECTED_T24_MODES:
            key = (scenario, mode)
            _require(key in index, f"T24 missing comparison row for {scenario}/{mode}")
            row = index[key]
            _require(row["completed_repeats"] == 2, f"T24 {scenario}/{mode} completed_repeats is not 2")
            _require(row["coverage"] == 1.0, f"T24 {scenario}/{mode} coverage is not 1.0")

    scenario_rows = {}
    for scenario in EXPECTED_SCENARIOS:
        ranked = sorted(
            (index[(scenario, mode)] for mode in EXPECTED_T24_MODES),
            key=lambda row: (row["final_ler_mean"], row["mode"]),
        )
        winner = ranked[0]
        runner_up = ranked[1]
        _require(
            winner["mode"] == "hybrid_residual_b",
            f"T24 winner for {scenario} is {winner['mode']} instead of hybrid_residual_b",
        )
        _require(
            runner_up["mode"] == "ukf",
            f"T24 runner-up for {scenario} is {runner_up['mode']} instead of ukf",
        )
        scenario_rows[scenario] = {
            "winner_mode": winner["mode"],
            "winner_final_ler_mean": winner["final_ler_mean"],
            "runner_up_mode": runner_up["mode"],
            "runner_up_final_ler_mean": runner_up["final_ler_mean"],
        }

    return {
        "git_commit": str(summary["git_commit"]),
        "scenario_rows": scenario_rows,
        "all_winners_hybrid_residual_b": True,
        "all_runner_ups_ukf": True,
        "t24_authoritative_frozen_ranked_table": True,
    }


def _validate_t64(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    t24_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(summary.get("missing_runs", []) == [], f"T64 missing_runs is not empty: {summary.get('missing_runs')}")
    _require(len(rows) == 24, f"T64 comparison row count {len(rows)} is not 24")

    t24_index = {(row["scenario"], row["mode"]): row for row in t24_rows}
    t64_index = {(row["scenario"], row["mode"]): row for row in rows}
    frozen_subset_exact_match = True
    for scenario in EXPECTED_SCENARIOS:
        for mode in EXPECTED_T24_MODES:
            t24_row = t24_index[(scenario, mode)]
            t64_row = t64_index[(scenario, mode)]
            _require(t64_row["completed_repeats"] == 2, f"T64 {scenario}/{mode} completed_repeats is not 2")
            _require(t64_row["coverage"] == 1.0, f"T64 {scenario}/{mode} coverage is not 1.0")
            if (
                t64_row["final_ler_mean"] != t24_row["final_ler_mean"]
                or t64_row["overflow_rate_mean"] != t24_row["overflow_rate_mean"]
            ):
                frozen_subset_exact_match = False

    statcalib_wins_all_four = True
    scenario_rows: dict[str, dict[str, Any]] = {}
    for scenario in EXPECTED_SCENARIOS:
        ranked = sorted(
            [t64_index[(scenario, mode)] for mode in EXPECTED_T24_MODES + ["statcalib"]],
            key=lambda row: (row["final_ler_mean"], row["mode"]),
        )
        winner = ranked[0]
        statcalib_row = t64_index[(scenario, "statcalib")]
        if winner["mode"] != "statcalib":
            statcalib_wins_all_four = False
        scenario_rows[scenario] = {
            "winner_mode": winner["mode"],
            "statcalib_final_ler_mean": statcalib_row["final_ler_mean"],
            "ukf_final_ler_mean": t64_index[(scenario, "ukf")]["final_ler_mean"],
            "hybrid_residual_b_final_ler_mean": t64_index[(scenario, "hybrid_residual_b")]["final_ler_mean"],
        }

    _require(frozen_subset_exact_match, "T64 frozen subset no longer matches T24 exactly")
    _require(statcalib_wins_all_four, "T64 statcalib no longer wins all four locked scenarios")

    return {
        "git_commit": str(summary["git_commit"]),
        "frozen_subset_exact_match_vs_t24": True,
        "statcalib_wins_all_four_scenarios": True,
        "scenario_rows": scenario_rows,
    }


def _validate_t66(summary_pack: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        bool(summary_pack.get("best_variant_beats_ukf_all_scenarios")),
        "T66 no longer shows the best statcalib variant beating ukf in all scenarios",
    )
    _require(
        bool(summary_pack.get("best_variant_beats_hybrid_all_scenarios")),
        "T66 no longer shows the best statcalib variant beating hybrid_residual_b in all scenarios",
    )
    rankings = list(summary_pack.get("variant_aggregate_rankings", []))
    _require(rankings, "T66 variant_aggregate_rankings is empty")
    best_by_mean = str(rankings[0]["mode"])
    best_by_stability = str(
        sorted(
            rankings,
            key=lambda row: (-int(row["win_count_within_statcalib"]), float(row["mean_rank_within_statcalib"]), str(row["mode"])),
        )[0]["mode"]
    )
    return {
        "git_commit": str(summary_pack["git_commit"]),
        "best_variant_overall": str(summary_pack["best_variant_overall"]["mode"]),
        "best_variant_beats_ukf_all_scenarios": True,
        "best_variant_beats_hybrid_all_scenarios": True,
        "best_by_mean_ler": best_by_mean,
        "best_by_stability": best_by_stability,
        "local_sensitivity_advantage_persists": True,
    }


def _validate_t67(summary_pack: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        bool(summary_pack.get("any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios")),
        "T67 no longer shows any non-ukf variant beating both frozen anchors in all scenarios",
    )
    qualifying = [str(mode) for mode in summary_pack.get("qualifying_non_ukf_variants_beating_both_frozen_anchors_all_scenarios", [])]
    _require(qualifying, "T67 qualifying non-ukf variant list is empty")
    _require(all("_ukf" not in mode for mode in qualifying), "T67 qualifying list unexpectedly includes ukf-teacher variants")
    parameter_points = list(summary_pack.get("parameter_point_comparisons", []))
    _require(parameter_points, "T67 parameter_point_comparisons is empty")
    return {
        "git_commit": str(summary_pack["git_commit"]),
        "any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios": True,
        "qualifying_non_ukf_variants": qualifying,
        "best_teacher_anchor_by_parameter_point": {
            str(item["parameter_point"]): str(item["best_teacher_anchor"]) for item in parameter_points
        },
        "teacher_anchor_robustness_supported": True,
    }


def _validate_t68(summary_pack: Mapping[str, Any]) -> dict[str, Any]:
    _require(bool(summary_pack.get("any_full_generated_only_winner")), "T68 no longer reports any full generated-only winner")
    full_winners = sorted(str(mode) for mode in summary_pack.get("full_generated_only_winner_modes", []))
    _require(
        full_winners == sorted(EXPECTED_T68_FULL_GENERATED_ONLY_WINNERS),
        f"T68 full generated-only winners changed: {full_winners}",
    )
    mean_best = sorted(str(mode) for mode in summary_pack.get("mean_best_candidates", {}).get("modes", []))
    _require(
        mean_best == sorted(EXPECTED_T69_CLEAN_SET),
        f"T68 mean_best_candidates.modes changed: {mean_best}",
    )
    worst_case_best = sorted(str(mode) for mode in summary_pack.get("worst_case_best_candidates", {}).get("modes", []))
    _require(
        str(summary_pack.get("mean_best_and_worst_case_best_relation")) == "different",
        "T68 mean-best vs worst-case-best relation is no longer different",
    )
    return {
        "git_commit": str(summary_pack["git_commit"]),
        "any_full_generated_only_winner": True,
        "full_generated_only_winner_modes": full_winners,
        "mean_best_candidates": mean_best,
        "worst_case_best_candidates": worst_case_best,
        "mean_best_and_worst_case_best_relation": str(summary_pack["mean_best_and_worst_case_best_relation"]),
        "generated_only_closure_supported": True,
    }


def _validate_t69(summary_pack: Mapping[str, Any]) -> dict[str, Any]:
    final_classification = str(summary_pack.get("final_clean_winner_classification"))
    _require(
        final_classification == "persistent_clean_tie_set",
        f"T69 final_clean_winner_classification changed to {final_classification}",
    )
    current_clean_answer_set = sorted(
        str(mode) for mode in summary_pack.get("current_clean_answer_set", {}).get("modes", [])
    )
    _require(
        current_clean_answer_set == sorted(EXPECTED_T69_CLEAN_SET),
        f"T69 current_clean_answer_set.modes changed to {current_clean_answer_set}",
    )
    _require(
        not bool(summary_pack.get("unique_clean_reference_point_exists")),
        "T69 unexpectedly reports a unique clean reference point",
    )
    _require(
        str(summary_pack.get("t68_clean_tie_set_relation")) == "persists",
        f"T69 t68_clean_tie_set_relation changed to {summary_pack.get('t68_clean_tie_set_relation')}",
    )
    _require(
        str(summary_pack.get("mean_best_and_worst_case_best_relation")) == "same",
        "T69 mean-best vs worst-case-best relation is no longer same",
    )
    return {
        "git_commit": str(summary_pack["git_commit"]),
        "final_clean_winner_classification": final_classification,
        "current_clean_answer_set": current_clean_answer_set,
        "unique_clean_reference_point_exists": False,
        "t68_clean_tie_set_relation": str(summary_pack["t68_clean_tie_set_relation"]),
        "mean_best_and_worst_case_best_relation": str(summary_pack["mean_best_and_worst_case_best_relation"]),
    }


def _read_only_inputs(paths: Sequence[Path]) -> list[str]:
    resolved = [_resolve(path) for path in paths]
    for path in resolved:
        text_path = str(path).lower().replace("\\", "/")
        _require("/.wt/" not in text_path, f"Unexpected worktree path in closure inputs: {path}")
        _require("/runs/sidecar/" not in text_path, f"Unexpected sidecar path in closure inputs: {path}")
    return [str(path) for path in resolved]


def build_closure_pack(inputs: ClosureInputs | None = None) -> dict[str, Any]:
    inputs = ClosureInputs() if inputs is None else inputs
    _validate_boundary_docs(inputs)
    review_verdicts = _validate_review_chain(inputs)

    t24_summary = _read_json(_resolve(inputs.t24_run_dir) / "summary.json")
    t24_rows = [
        _normalize_comparison_row(row)
        for row in _read_csv_rows(_resolve(inputs.t24_run_dir) / "comparison.csv")
    ]
    t24_fact = _validate_t24(t24_summary, t24_rows)

    t64_summary = _read_json(_resolve(inputs.t64_run_dir) / "summary.json")
    t64_rows = [
        _normalize_comparison_row(row)
        for row in _read_csv_rows(_resolve(inputs.t64_run_dir) / "comparison.csv")
    ]
    t64_fact = _validate_t64(t64_summary, t64_rows, t24_rows)

    t66_fact = _validate_t66(_read_json(_resolve(inputs.t66_summary_json)))
    t67_fact = _validate_t67(_read_json(_resolve(inputs.t67_summary_json)))
    t68_fact = _validate_t68(_read_json(_resolve(inputs.t68_summary_json)))
    t69_fact = _validate_t69(_read_json(_resolve(inputs.t69_summary_json)))

    promotion_gate = {
        "verdict": "no_promotion_keep_extension_lane_only",
        "reasons": [
            "T24 remains the authoritative frozen ranked table under the locked formal protocol.",
            "All FR8 artifacts remain mock-backed software-HIL extension-lane evidence only.",
            "The broader predeclared statcalib grid is still not uniformly clean after T68 and T69.",
            "Current accepted docs and reviews keep statcalib outside the frozen ranked set and outside mature-comparator claims.",
        ],
    }
    unique_threshold_gate = {
        "verdict": "future_selection_task_required",
        "reasons": [
            "T69 still ends at persistent_clean_tie_set.",
            "No unique clean reference point exists after the stronger repeats=4 tie-break matrix.",
            "Choosing one threshold now would require a predeclared selection-criterion task rather than a silent retelling of T69.",
        ],
    }
    minimal_future_task_prerequisites = [
        "Predeclare one selection criterion that is not already tied by the current T69 mean/worst-case LER evidence.",
        "Lock the candidate set and the decision rule before any new execution or downstream promotion retelling.",
        "Keep T24 frozen and keep statcalib labeled as an extension lane unless a later promotion gate explicitly changes that status.",
        "If the intended claim is stronger than mock-backed software-HIL evidence, open a separate bounded validation task for that target surface instead of reusing T64-T69 as deployment proof.",
    ]

    supported_claims = [
        "T24 remains the authoritative frozen ranked table and hybrid_residual_b still wins all four frozen scenarios there.",
        "T64 proves one clean bounded statcalib extension-lane win without rewriting the frozen T24 subset.",
        "T66 shows the bounded statcalib advantage survives the predeclared local sensitivity grid.",
        "T67 shows the bounded statcalib advantage is not grossly dependent on teacher_mode=ukf.",
        "T68 shows that full generated-only statcalib winners do exist inside the predeclared bounded grid.",
        "T69 shows that the strongest clean answer remains the persistent window_variance_t001=t003=t005 tie set and that no unique clean reference point emerges.",
    ]
    unsupported_claims = [
        "Rewriting T24 as if FR8 extension-lane evidence replaced the historical frozen ranked table.",
        "Treating the current statcalib lane as a mature calibration comparator rather than a bounded extension lane.",
        "Claiming one unique final threshold is already supported by accepted artifacts.",
        "Upgrading T64-T69 into .tflite, real-board, or paper-grade expanded benchmark evidence.",
    ]

    closure_table = [
        {
            "category": "frozen_anchor_evidence",
            "subject": "T24",
            "summary": "The frozen ranked table remains authoritative; hybrid_residual_b wins all four frozen scenarios and ukf is runner-up in all four.",
        },
        {
            "category": "extension_lane_evidence",
            "subject": "T64/T66/T67/T68/T69",
            "summary": "The extension-lane win exists, survives the local sensitivity grid, is not grossly teacher-anchor dependent, has full generated-only winners, and still ends at a persistent clean tie set.",
        },
        {
            "category": "supported_claims",
            "subject": "Current bounded FR8 answer",
            "summary": "The strongest clean answer after T69 is the persistent window_variance_t001=t003=t005 tie set, with no unique clean reference point.",
        },
        {
            "category": "unsupported_claims",
            "subject": "Out-of-bound retellings",
            "summary": "The current evidence does not support rewriting T24, naming a unique threshold, or upgrading statcalib into mature comparator/.tflite/real-board evidence.",
        },
    ]

    input_paths = _read_only_inputs(
        [
            inputs.protocol_doc,
            inputs.t64_report,
            inputs.t65_report,
            inputs.t66_report,
            inputs.t67_report,
            inputs.t68_report,
            inputs.t69_report,
            inputs.t64_review,
            inputs.t65_review,
            inputs.t66_review,
            inputs.t67_review,
            inputs.t68_review,
            inputs.t69_review,
            inputs.t24_run_dir,
            inputs.t64_run_dir,
            inputs.t66_summary_json,
            inputs.t67_summary_json,
            inputs.t68_summary_json,
            inputs.t69_summary_json,
        ]
    )

    return {
        "task_id": "T70",
        "review_verdict_chain": review_verdicts,
        "historical_inputs_read_only": input_paths,
        "no_new_run_root_created": True,
        "historical_runs_modified": False,
        "sidecar_outputs_used": False,
        "frozen_anchor_evidence": t24_fact,
        "extension_lane_evidence": {
            "t64": t64_fact,
            "t66": t66_fact,
            "t67": t67_fact,
            "t68": t68_fact,
            "t69": t69_fact,
        },
        "final_strongest_clean_answer_set_after_t69": t69_fact["current_clean_answer_set"],
        "unique_clean_reference_point_exists": t69_fact["unique_clean_reference_point_exists"],
        "promotion_gate": promotion_gate,
        "unique_threshold_gate": unique_threshold_gate,
        "minimal_prerequisites_for_later_single_threshold_selection_task": minimal_future_task_prerequisites,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "closure_table": closure_table,
    }


def main() -> int:
    args = _parser().parse_args()
    inputs = ClosureInputs(
        t69_summary_json=DEFAULT_T69_SUMMARY_JSON
        if args.t69_summary_json is None
        else Path(args.t69_summary_json)
    )
    payload = build_closure_pack(inputs)
    if args.output_json is not None:
        _write_json(_resolve(args.output_json), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
