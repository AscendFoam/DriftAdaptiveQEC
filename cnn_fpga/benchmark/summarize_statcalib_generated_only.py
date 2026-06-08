"""Summarize the bounded T68 statcalib generated-only robustness artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SCENARIOS = [
    "static_bias_theta",
    "linear_ramp",
    "step_sigma_theta",
    "periodic_drift",
]
EXPECTED_ANCHOR_MODES = [
    "ukf",
    "hybrid_residual_b",
]
THRESHOLD_METADATA = [
    ("t001", 0.001),
    ("t003", 0.003),
    ("t005", 0.005),
    ("t010", 0.010),
]
EXPECTED_STATCALIB_VARIANTS = [
    {
        "mode": "statcalib_window_variance_t001",
        "teacher_anchor": "window_variance",
        "threshold_key": "t001",
        "signal_threshold": 0.001,
    },
    {
        "mode": "statcalib_window_variance_t003",
        "teacher_anchor": "window_variance",
        "threshold_key": "t003",
        "signal_threshold": 0.003,
    },
    {
        "mode": "statcalib_window_variance_t005",
        "teacher_anchor": "window_variance",
        "threshold_key": "t005",
        "signal_threshold": 0.005,
    },
    {
        "mode": "statcalib_window_variance_t010",
        "teacher_anchor": "window_variance",
        "threshold_key": "t010",
        "signal_threshold": 0.010,
    },
    {
        "mode": "statcalib_ekf_t001",
        "teacher_anchor": "ekf",
        "threshold_key": "t001",
        "signal_threshold": 0.001,
    },
    {
        "mode": "statcalib_ekf_t003",
        "teacher_anchor": "ekf",
        "threshold_key": "t003",
        "signal_threshold": 0.003,
    },
    {
        "mode": "statcalib_ekf_t005",
        "teacher_anchor": "ekf",
        "threshold_key": "t005",
        "signal_threshold": 0.005,
    },
    {
        "mode": "statcalib_ekf_t010",
        "teacher_anchor": "ekf",
        "threshold_key": "t010",
        "signal_threshold": 0.010,
    },
]
EXPECTED_STATCALIB_MODES = [item["mode"] for item in EXPECTED_STATCALIB_VARIANTS]
EXPECTED_ALL_MODES = EXPECTED_ANCHOR_MODES + EXPECTED_STATCALIB_MODES
EXPECTED_TEACHER_ANCHORS = ["window_variance", "ekf"]
DEFAULT_PACK_DIRNAME = "statcalib_generated_only_summary"
TIE_TOLERANCE = 1e-15

VARIANT_METADATA = {
    item["mode"]: {
        "teacher_anchor": item["teacher_anchor"],
        "threshold_key": item["threshold_key"],
        "signal_threshold": item["signal_threshold"],
    }
    for item in EXPECTED_STATCALIB_VARIANTS
}
THRESHOLD_VALUE_BY_KEY = dict(THRESHOLD_METADATA)


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to the T68 run root.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Optional output directory. Defaults to <run-dir>/{DEFAULT_PACK_DIRNAME}.",
    )
    return parser


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mean(values: Sequence[float | None]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _same_metric(lhs: float, rhs: float) -> bool:
    return math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=TIE_TOLERANCE)


def _group_with_ties(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    tie_break_key: str,
) -> list[list[Mapping[str, Any]]]:
    ranked = sorted(rows, key=lambda row: (float(row[value_key]), str(row[tie_break_key])))
    groups: list[list[Mapping[str, Any]]] = []
    for row in ranked:
        if not groups:
            groups.append([row])
            continue
        if _same_metric(float(groups[-1][0][value_key]), float(row[value_key])):
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def _format_tie_ranking(groups: Sequence[Sequence[Mapping[str, Any]]], *, label_key: str) -> str:
    parts = []
    for group in groups:
        labels = sorted(str(row[label_key]) for row in group)
        parts.append(" = ".join(labels))
    return " > ".join(parts)


def _mode_list(group: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(str(row["mode"]) for row in group)


def _status_count(rows: Sequence[Mapping[str, Any]], status: str) -> int:
    return sum(1 for row in rows if str(row["statcalib_status"]) == status)


def _ranking_relation_by_threshold(values: Sequence[float]) -> str:
    non_decreasing = all(values[idx] <= values[idx + 1] or _same_metric(values[idx], values[idx + 1]) for idx in range(len(values) - 1))
    non_increasing = all(values[idx] >= values[idx + 1] or _same_metric(values[idx], values[idx + 1]) for idx in range(len(values) - 1))
    if non_decreasing and non_increasing:
        return "constant"
    if non_decreasing:
        return "monotonic_non_decreasing"
    if non_increasing:
        return "monotonic_non_increasing"
    return "non_monotonic"


def _normalize_row(row: Mapping[str, str]) -> dict[str, Any]:
    mode = str(row["mode"])
    metadata = VARIANT_METADATA.get(mode, {})
    return {
        "scenario": str(row["scenario"]),
        "scenario_label": str(row["scenario_label"]),
        "mode": mode,
        "mode_label": str(row["mode_label"]),
        "completed_repeats": int(row["completed_repeats"]),
        "expected_repeats": int(row["expected_repeats"]),
        "coverage": float(row["coverage"]),
        "final_ler_mean": float(row["final_ler_mean"]),
        "final_ler_std": float(row["final_ler_std"]),
        "overflow_rate_mean": float(row["overflow_rate_mean"]),
        "overflow_rate_std": float(row["overflow_rate_std"]),
        "teacher_anchor": metadata.get("teacher_anchor", ""),
        "threshold_key": metadata.get("threshold_key", ""),
        "signal_threshold": metadata.get("signal_threshold"),
        "statcalib_status": str(row.get("statcalib_status") or ""),
        "statcalib_reason": str(row.get("statcalib_reason") or ""),
        "statcalib_generated_repeats": int(row.get("statcalib_generated_repeats") or 0),
        "statcalib_generated_windows_mean": _optional_float(row.get("statcalib_generated_windows_mean")),
        "statcalib_signal_norm_mean_mean": _optional_float(row.get("statcalib_signal_norm_mean_mean")),
    }


def _validate_run(
    *,
    summary: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    missing_runs = summary.get("missing_runs", [])
    if missing_runs:
        raise ValueError(f"summary.json still reports missing runs: {missing_runs}")
    if list(launch_plan.get("requested_scenarios", [])) != EXPECTED_SCENARIOS:
        raise ValueError(
            f"requested_scenarios={launch_plan.get('requested_scenarios')} does not match expected {EXPECTED_SCENARIOS}"
        )
    if list(launch_plan.get("requested_modes", [])) != EXPECTED_ALL_MODES:
        raise ValueError(f"requested_modes={launch_plan.get('requested_modes')} does not match expected {EXPECTED_ALL_MODES}")
    if not bool(launch_plan.get("paired_seeds")):
        raise ValueError("launch_plan.paired_seeds is not true")
    if int(launch_plan.get("repeats", 0)) != 2:
        raise ValueError(f"launch_plan.repeats={launch_plan.get('repeats')} is not 2")

    protocol = dict(summary.get("protocol", {}))
    if not bool(protocol.get("paired_seeds")):
        raise ValueError("summary.protocol.paired_seeds is not true")
    if int(protocol.get("repeats", 0)) != 2:
        raise ValueError(f"summary.protocol.repeats={protocol.get('repeats')} is not 2")

    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["scenario"]), str(row["mode"]))
        if key in seen_keys:
            raise ValueError(f"duplicate comparison row detected for {key[0]}/{key[1]}")
        seen_keys.add(key)
        if int(row["completed_repeats"]) != 2:
            raise ValueError(f"{key[0]}/{key[1]} completed_repeats={row['completed_repeats']} is not 2")
        if not _same_metric(float(row["coverage"]), 1.0):
            raise ValueError(f"{key[0]}/{key[1]} coverage={row['coverage']} is not 1.0")

    for scenario in EXPECTED_SCENARIOS:
        for mode in EXPECTED_ALL_MODES:
            if (scenario, mode) not in seen_keys:
                raise ValueError(f"missing comparison row for {scenario}/{mode}")


def _candidate_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mode": row["mode"],
        "mode_label": row["mode_label"],
        "teacher_anchor": row["teacher_anchor"],
        "threshold_key": row["threshold_key"],
        "signal_threshold": row["signal_threshold"],
        "mean_final_ler_mean": row["mean_final_ler_mean"],
        "worst_scenario_final_ler_mean": row["worst_scenario_final_ler_mean"],
        "best_scenario_final_ler_mean": row["best_scenario_final_ler_mean"],
        "generated_row_count": row["generated_row_count"],
        "mixed_row_count": row["mixed_row_count"],
        "beats_both_frozen_anchors_count": row["beats_both_frozen_anchors_count"],
        "full_generated_only_winner": row["full_generated_only_winner"],
    }


def _pareto_front_rows(candidate_rows: Sequence[Mapping[str, Any]]) -> set[str]:
    front: set[str] = set()
    for row in candidate_rows:
        dominated = False
        for other in candidate_rows:
            if row["mode"] == other["mode"]:
                continue
            mean_better_or_equal = (
                other["mean_final_ler_mean"] < row["mean_final_ler_mean"]
                or _same_metric(other["mean_final_ler_mean"], row["mean_final_ler_mean"])
            )
            generated_better_or_equal = int(other["generated_row_count"]) >= int(row["generated_row_count"])
            strictly_better = (
                other["mean_final_ler_mean"] < row["mean_final_ler_mean"]
                or int(other["generated_row_count"]) > int(row["generated_row_count"])
            )
            if mean_better_or_equal and generated_better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.add(str(row["mode"]))
    return front


def summarize_run(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    summary = _read_json(run_dir / "summary.json")
    launch_plan = _read_json(run_dir / "launch_plan.json")
    comparison_rows = [_normalize_row(row) for row in _read_csv_rows(run_dir / "comparison.csv")]
    _validate_run(summary=summary, launch_plan=launch_plan, rows=comparison_rows)

    index = {(row["scenario"], row["mode"]): row for row in comparison_rows}
    mode_summary_rows: list[dict[str, Any]] = []
    scenario_summary_rows: list[dict[str, Any]] = []
    candidate_summary_rows: list[dict[str, Any]] = []
    threshold_comparison_rows: list[dict[str, Any]] = []
    teacher_anchor_summary_rows: list[dict[str, Any]] = []
    pareto_summary_rows: list[dict[str, Any]] = []

    for scenario in EXPECTED_SCENARIOS:
        ukf_row = dict(index[(scenario, "ukf")])
        hybrid_row = dict(index[(scenario, "hybrid_residual_b")])
        candidate_rows = [dict(index[(scenario, mode)]) for mode in EXPECTED_STATCALIB_MODES]
        candidate_groups = _group_with_ties(candidate_rows, value_key="final_ler_mean", tie_break_key="mode")
        best_group = candidate_groups[0]

        for row in candidate_rows:
            gap_vs_ukf = ukf_row["final_ler_mean"] - row["final_ler_mean"]
            gap_vs_hybrid = hybrid_row["final_ler_mean"] - row["final_ler_mean"]
            mode_summary_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": row["scenario_label"],
                    "mode": row["mode"],
                    "mode_label": row["mode_label"],
                    "teacher_anchor": row["teacher_anchor"],
                    "threshold_key": row["threshold_key"],
                    "signal_threshold": row["signal_threshold"],
                    "completed_repeats": row["completed_repeats"],
                    "expected_repeats": row["expected_repeats"],
                    "coverage": row["coverage"],
                    "final_ler_mean": row["final_ler_mean"],
                    "final_ler_std": row["final_ler_std"],
                    "overflow_rate_mean": row["overflow_rate_mean"],
                    "overflow_rate_std": row["overflow_rate_std"],
                    "ukf_ler_minus_mode_ler": gap_vs_ukf,
                    "hybrid_ler_minus_mode_ler": gap_vs_hybrid,
                    "beats_ukf": gap_vs_ukf > 0.0,
                    "beats_hybrid_residual_b": gap_vs_hybrid > 0.0,
                    "beats_both_frozen_anchors": gap_vs_ukf > 0.0 and gap_vs_hybrid > 0.0,
                    "statcalib_status": row["statcalib_status"],
                    "statcalib_reason": row["statcalib_reason"],
                    "statcalib_generated_repeats": row["statcalib_generated_repeats"],
                    "statcalib_generated_windows_mean": row["statcalib_generated_windows_mean"],
                    "statcalib_signal_norm_mean_mean": row["statcalib_signal_norm_mean_mean"],
                }
            )

        scenario_summary_rows.append(
            {
                "scenario": scenario,
                "scenario_label": ukf_row["scenario_label"],
                "ukf_final_ler_mean": ukf_row["final_ler_mean"],
                "hybrid_residual_b_final_ler_mean": hybrid_row["final_ler_mean"],
                "best_candidate_modes": "|".join(_mode_list(best_group)),
                "best_candidate_labels": " | ".join(sorted(str(row["mode_label"]) for row in best_group)),
                "best_candidate_teacher_anchors": "|".join(sorted({str(row["teacher_anchor"]) for row in best_group})),
                "best_candidate_threshold_keys": "|".join(sorted({str(row["threshold_key"]) for row in best_group})),
                "best_candidate_final_ler_mean": float(best_group[0]["final_ler_mean"]),
                "best_candidate_status_values": "|".join(sorted({str(row["statcalib_status"]) for row in best_group})),
                "best_candidate_beats_ukf": ukf_row["final_ler_mean"] > float(best_group[0]["final_ler_mean"]),
                "best_candidate_beats_hybrid_residual_b": hybrid_row["final_ler_mean"] > float(best_group[0]["final_ler_mean"]),
                "candidate_ranking_with_ties": _format_tie_ranking(candidate_groups, label_key="mode"),
            }
        )

    for item in EXPECTED_STATCALIB_VARIANTS:
        mode = item["mode"]
        rows = [dict(index[(scenario, mode)]) for scenario in EXPECTED_SCENARIOS]
        beats_both_count = 0
        ukf_gaps: list[float] = []
        hybrid_gaps: list[float] = []
        for row in rows:
            ukf_row = index[(row["scenario"], "ukf")]
            hybrid_row = index[(row["scenario"], "hybrid_residual_b")]
            gap_vs_ukf = ukf_row["final_ler_mean"] - row["final_ler_mean"]
            gap_vs_hybrid = hybrid_row["final_ler_mean"] - row["final_ler_mean"]
            ukf_gaps.append(gap_vs_ukf)
            hybrid_gaps.append(gap_vs_hybrid)
            if gap_vs_ukf > 0.0 and gap_vs_hybrid > 0.0:
                beats_both_count += 1

        best_row = min(rows, key=lambda row: (row["final_ler_mean"], row["scenario"]))
        worst_row = max(rows, key=lambda row: (row["final_ler_mean"], row["scenario"]))
        generated_row_count = _status_count(rows, "generated")
        mixed_row_count = _status_count(rows, "mixed")
        candidate_summary_rows.append(
            {
                "mode": mode,
                "mode_label": rows[0]["mode_label"],
                "teacher_anchor": item["teacher_anchor"],
                "threshold_key": item["threshold_key"],
                "signal_threshold": item["signal_threshold"],
                "scenario_count": len(rows),
                "mean_final_ler_mean": sum(row["final_ler_mean"] for row in rows) / len(rows),
                "worst_scenario": worst_row["scenario"],
                "worst_scenario_final_ler_mean": worst_row["final_ler_mean"],
                "best_scenario": best_row["scenario"],
                "best_scenario_final_ler_mean": best_row["final_ler_mean"],
                "generated_row_count": generated_row_count,
                "mixed_row_count": mixed_row_count,
                "status_values": "|".join(sorted({str(row["statcalib_status"]) for row in rows})),
                "reason_values": "|".join(sorted({str(row["statcalib_reason"]) for row in rows})),
                "beats_both_frozen_anchors_count": beats_both_count,
                "beats_both_frozen_anchors_all_scenarios": beats_both_count == len(rows),
                "full_generated_only_winner": generated_row_count == len(rows) and beats_both_count == len(rows),
                "mean_gap_vs_ukf": sum(ukf_gaps) / len(ukf_gaps),
                "mean_gap_vs_hybrid_residual_b": sum(hybrid_gaps) / len(hybrid_gaps),
                "generated_windows_mean_across_scenarios": _mean([row["statcalib_generated_windows_mean"] for row in rows]),
                "signal_norm_mean_across_scenarios": _mean([row["statcalib_signal_norm_mean_mean"] for row in rows]),
            }
        )

    candidate_summary_rows.sort(
        key=lambda row: (
            row["mean_final_ler_mean"],
            row["worst_scenario_final_ler_mean"],
            row["mode"],
        )
    )
    candidate_by_mode = {row["mode"]: row for row in candidate_summary_rows}

    mean_groups = _group_with_ties(candidate_summary_rows, value_key="mean_final_ler_mean", tie_break_key="mode")
    worst_case_groups = _group_with_ties(
        sorted(
            candidate_summary_rows,
            key=lambda row: (
                row["worst_scenario_final_ler_mean"],
                row["mean_final_ler_mean"],
                row["mode"],
            ),
        ),
        value_key="worst_scenario_final_ler_mean",
        tie_break_key="mode",
    )
    mean_best_group = mean_groups[0]
    worst_case_best_group = worst_case_groups[0]
    mean_best_modes = _mode_list(mean_best_group)
    worst_case_best_modes = _mode_list(worst_case_best_group)

    for threshold_key, signal_threshold in THRESHOLD_METADATA:
        window_row = candidate_by_mode[f"statcalib_window_variance_{threshold_key}"]
        ekf_row = candidate_by_mode[f"statcalib_ekf_{threshold_key}"]
        if _same_metric(window_row["mean_final_ler_mean"], ekf_row["mean_final_ler_mean"]):
            winner_by_mean_ler = "tie"
        elif window_row["mean_final_ler_mean"] < ekf_row["mean_final_ler_mean"]:
            winner_by_mean_ler = "window_variance"
        else:
            winner_by_mean_ler = "ekf"
        threshold_comparison_rows.append(
            {
                "threshold_key": threshold_key,
                "signal_threshold": signal_threshold,
                "window_variance_mode": window_row["mode"],
                "window_variance_mean_final_ler_mean": window_row["mean_final_ler_mean"],
                "window_variance_generated_row_count": window_row["generated_row_count"],
                "window_variance_mixed_row_count": window_row["mixed_row_count"],
                "window_variance_beats_both_count": window_row["beats_both_frozen_anchors_count"],
                "ekf_mode": ekf_row["mode"],
                "ekf_mean_final_ler_mean": ekf_row["mean_final_ler_mean"],
                "ekf_generated_row_count": ekf_row["generated_row_count"],
                "ekf_mixed_row_count": ekf_row["mixed_row_count"],
                "ekf_beats_both_count": ekf_row["beats_both_frozen_anchors_count"],
                "window_minus_ekf_mean_ler": window_row["mean_final_ler_mean"] - ekf_row["mean_final_ler_mean"],
                "winner_by_mean_ler": winner_by_mean_ler,
            }
        )

    for teacher_anchor in EXPECTED_TEACHER_ANCHORS:
        rows = [
            candidate_by_mode[f"statcalib_{teacher_anchor}_{threshold_key}"]
            for threshold_key, _ in THRESHOLD_METADATA
        ]
        ranked_groups = _group_with_ties(rows, value_key="mean_final_ler_mean", tie_break_key="threshold_key")
        threshold_sequence = [row["mean_final_ler_mean"] for row in rows]
        monotonicity = _ranking_relation_by_threshold(threshold_sequence)
        if monotonicity == "non_monotonic":
            sequence_note = "|".join(
                f"{row['threshold_key']}={row['mean_final_ler_mean']:.12f}"
                for row in rows
            )
            monotonicity_note = f"non_monotonic_by_threshold_order:{sequence_note}"
        else:
            monotonicity_note = ""
        teacher_anchor_summary_rows.append(
            {
                "teacher_anchor": teacher_anchor,
                "threshold_ranking_with_ties": _format_tie_ranking(ranked_groups, label_key="threshold_key"),
                "best_threshold_keys": "|".join(sorted({str(row["threshold_key"]) for row in ranked_groups[0]})),
                "best_mean_final_ler_mean": float(ranked_groups[0][0]["mean_final_ler_mean"]),
                "monotonicity": monotonicity,
                "monotonicity_note": monotonicity_note,
                "t001_mean_final_ler_mean": rows[0]["mean_final_ler_mean"],
                "t003_mean_final_ler_mean": rows[1]["mean_final_ler_mean"],
                "t005_mean_final_ler_mean": rows[2]["mean_final_ler_mean"],
                "t010_mean_final_ler_mean": rows[3]["mean_final_ler_mean"],
            }
        )

    pareto_front_modes = _pareto_front_rows(candidate_summary_rows)
    for row in candidate_summary_rows:
        pareto_summary_rows.append(
            {
                "mode": row["mode"],
                "mode_label": row["mode_label"],
                "teacher_anchor": row["teacher_anchor"],
                "threshold_key": row["threshold_key"],
                "mean_final_ler_mean": row["mean_final_ler_mean"],
                "generated_row_count": row["generated_row_count"],
                "mixed_row_count": row["mixed_row_count"],
                "beats_both_frozen_anchors_count": row["beats_both_frozen_anchors_count"],
                "full_generated_only_winner": row["full_generated_only_winner"],
                "pareto_front": row["mode"] in pareto_front_modes,
            }
        )
    pareto_summary_rows.sort(key=lambda row: (-int(row["pareto_front"]), -int(row["generated_row_count"]), row["mean_final_ler_mean"], row["mode"]))

    full_generated_only_winners = [row for row in candidate_summary_rows if row["full_generated_only_winner"]]
    closest_near_miss: dict[str, Any] | None = None
    if not full_generated_only_winners:
        near_miss_rows = []
        for row in candidate_summary_rows:
            generated_gap = len(EXPECTED_SCENARIOS) - int(row["generated_row_count"])
            anchor_gap = len(EXPECTED_SCENARIOS) - int(row["beats_both_frozen_anchors_count"])
            near_miss_rows.append(
                {
                    **row,
                    "generated_gap": generated_gap,
                    "anchor_gap": anchor_gap,
                    "total_gap": generated_gap + anchor_gap,
                }
            )
        near_miss_rows.sort(
            key=lambda row: (
                row["total_gap"],
                row["anchor_gap"],
                row["generated_gap"],
                row["mean_final_ler_mean"],
                row["worst_scenario_final_ler_mean"],
                row["mode"],
            )
        )
        closest_near_miss = {
            **_candidate_fields(near_miss_rows[0]),
            "generated_gap": near_miss_rows[0]["generated_gap"],
            "anchor_gap": near_miss_rows[0]["anchor_gap"],
            "total_gap": near_miss_rows[0]["total_gap"],
        }

    summary_pack = {
        "run_dir": str(run_dir),
        "git_commit": summary.get("git_commit"),
        "launch_plan": str(run_dir / "launch_plan.json"),
        "comparison_csv": str(run_dir / "comparison.csv"),
        "expected_scenarios": EXPECTED_SCENARIOS,
        "expected_anchor_modes": EXPECTED_ANCHOR_MODES,
        "expected_teacher_anchors": EXPECTED_TEACHER_ANCHORS,
        "expected_thresholds": [{"threshold_key": key, "signal_threshold": value} for key, value in THRESHOLD_METADATA],
        "expected_statcalib_modes": EXPECTED_STATCALIB_MODES,
        "scenario_summaries": scenario_summary_rows,
        "candidate_summaries": candidate_summary_rows,
        "threshold_comparisons": threshold_comparison_rows,
        "teacher_anchor_comparisons": teacher_anchor_summary_rows,
        "pareto_summary": pareto_summary_rows,
        "mean_best_candidates": {
            "modes": mean_best_modes,
            "ranking_with_ties": _format_tie_ranking(mean_groups[:1], label_key="mode"),
            "mean_final_ler_mean": float(mean_best_group[0]["mean_final_ler_mean"]),
        },
        "worst_case_best_candidates": {
            "modes": worst_case_best_modes,
            "ranking_with_ties": _format_tie_ranking(worst_case_groups[:1], label_key="mode"),
            "worst_scenario_final_ler_mean": float(worst_case_best_group[0]["worst_scenario_final_ler_mean"]),
        },
        "mean_best_and_worst_case_best_relation": (
            "same" if set(mean_best_modes) == set(worst_case_best_modes) else "different"
        ),
        "full_generated_only_winner_modes": [row["mode"] for row in full_generated_only_winners],
        "any_full_generated_only_winner": bool(full_generated_only_winners),
        "closest_near_miss_candidate": closest_near_miss,
    }

    output_dir = (run_dir / DEFAULT_PACK_DIRNAME) if output_dir is None else output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_summary_path = output_dir / "mode_summary.csv"
    scenario_summary_path = output_dir / "scenario_summary.csv"
    candidate_summary_path = output_dir / "candidate_summary.csv"
    threshold_comparison_path = output_dir / "threshold_comparison.csv"
    teacher_anchor_summary_path = output_dir / "teacher_anchor_summary.csv"
    pareto_summary_path = output_dir / "pareto_summary.csv"
    summary_json_path = output_dir / "summary.json"

    _write_csv(
        mode_summary_path,
        mode_summary_rows,
        [
            "scenario",
            "scenario_label",
            "mode",
            "mode_label",
            "teacher_anchor",
            "threshold_key",
            "signal_threshold",
            "completed_repeats",
            "expected_repeats",
            "coverage",
            "final_ler_mean",
            "final_ler_std",
            "overflow_rate_mean",
            "overflow_rate_std",
            "ukf_ler_minus_mode_ler",
            "hybrid_ler_minus_mode_ler",
            "beats_ukf",
            "beats_hybrid_residual_b",
            "beats_both_frozen_anchors",
            "statcalib_status",
            "statcalib_reason",
            "statcalib_generated_repeats",
            "statcalib_generated_windows_mean",
            "statcalib_signal_norm_mean_mean",
        ],
    )
    _write_csv(
        scenario_summary_path,
        scenario_summary_rows,
        [
            "scenario",
            "scenario_label",
            "ukf_final_ler_mean",
            "hybrid_residual_b_final_ler_mean",
            "best_candidate_modes",
            "best_candidate_labels",
            "best_candidate_teacher_anchors",
            "best_candidate_threshold_keys",
            "best_candidate_final_ler_mean",
            "best_candidate_status_values",
            "best_candidate_beats_ukf",
            "best_candidate_beats_hybrid_residual_b",
            "candidate_ranking_with_ties",
        ],
    )
    _write_csv(
        candidate_summary_path,
        candidate_summary_rows,
        [
            "mode",
            "mode_label",
            "teacher_anchor",
            "threshold_key",
            "signal_threshold",
            "scenario_count",
            "mean_final_ler_mean",
            "worst_scenario",
            "worst_scenario_final_ler_mean",
            "best_scenario",
            "best_scenario_final_ler_mean",
            "generated_row_count",
            "mixed_row_count",
            "status_values",
            "reason_values",
            "beats_both_frozen_anchors_count",
            "beats_both_frozen_anchors_all_scenarios",
            "full_generated_only_winner",
            "mean_gap_vs_ukf",
            "mean_gap_vs_hybrid_residual_b",
            "generated_windows_mean_across_scenarios",
            "signal_norm_mean_across_scenarios",
        ],
    )
    _write_csv(
        threshold_comparison_path,
        threshold_comparison_rows,
        [
            "threshold_key",
            "signal_threshold",
            "window_variance_mode",
            "window_variance_mean_final_ler_mean",
            "window_variance_generated_row_count",
            "window_variance_mixed_row_count",
            "window_variance_beats_both_count",
            "ekf_mode",
            "ekf_mean_final_ler_mean",
            "ekf_generated_row_count",
            "ekf_mixed_row_count",
            "ekf_beats_both_count",
            "window_minus_ekf_mean_ler",
            "winner_by_mean_ler",
        ],
    )
    _write_csv(
        teacher_anchor_summary_path,
        teacher_anchor_summary_rows,
        [
            "teacher_anchor",
            "threshold_ranking_with_ties",
            "best_threshold_keys",
            "best_mean_final_ler_mean",
            "monotonicity",
            "monotonicity_note",
            "t001_mean_final_ler_mean",
            "t003_mean_final_ler_mean",
            "t005_mean_final_ler_mean",
            "t010_mean_final_ler_mean",
        ],
    )
    _write_csv(
        pareto_summary_path,
        pareto_summary_rows,
        [
            "mode",
            "mode_label",
            "teacher_anchor",
            "threshold_key",
            "mean_final_ler_mean",
            "generated_row_count",
            "mixed_row_count",
            "beats_both_frozen_anchors_count",
            "full_generated_only_winner",
            "pareto_front",
        ],
    )
    _write_json(summary_json_path, summary_pack)

    return {
        "mode_summary_rows": mode_summary_rows,
        "scenario_summary_rows": scenario_summary_rows,
        "candidate_summary_rows": candidate_summary_rows,
        "threshold_comparison_rows": threshold_comparison_rows,
        "teacher_anchor_summary_rows": teacher_anchor_summary_rows,
        "pareto_summary_rows": pareto_summary_rows,
        "summary_pack": summary_pack,
        "output_dir": str(output_dir),
        "mode_summary_path": str(mode_summary_path),
        "scenario_summary_path": str(scenario_summary_path),
        "candidate_summary_path": str(candidate_summary_path),
        "threshold_comparison_path": str(threshold_comparison_path),
        "teacher_anchor_summary_path": str(teacher_anchor_summary_path),
        "pareto_summary_path": str(pareto_summary_path),
        "summary_json_path": str(summary_json_path),
    }


def _render_console_summary(result: Mapping[str, Any]) -> str:
    summary_pack = dict(result["summary_pack"])
    mean_best = dict(summary_pack["mean_best_candidates"])
    worst_best = dict(summary_pack["worst_case_best_candidates"])
    lines = [
        "T68 statcalib generated-only summary",
        f"- run_dir: {summary_pack['run_dir']}",
        f"- git_commit: {summary_pack['git_commit']}",
        f"- mean_best_candidates: {', '.join(mean_best['modes'])}",
        f"- worst_case_best_candidates: {', '.join(worst_best['modes'])}",
        f"- mean_best_vs_worst_case_best: {summary_pack['mean_best_and_worst_case_best_relation']}",
        f"- any_full_generated_only_winner: {summary_pack['any_full_generated_only_winner']}",
    ]
    if summary_pack["any_full_generated_only_winner"]:
        lines.append(
            "- full_generated_only_winner_modes: "
            + ", ".join(summary_pack["full_generated_only_winner_modes"])
        )
    else:
        closest = summary_pack["closest_near_miss_candidate"]
        lines.append(
            f"- closest_near_miss: {closest['mode']} generated_gap={closest['generated_gap']} "
            f"anchor_gap={closest['anchor_gap']}"
        )
    lines.append(f"- wrote: {result['summary_json_path']}")
    return "\n".join(lines)


def main() -> int:
    args = _arg_parser().parse_args()
    result = summarize_run(
        run_dir=Path(args.run_dir),
        output_dir=None if args.output_dir is None else Path(args.output_dir),
    )
    print(_render_console_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
