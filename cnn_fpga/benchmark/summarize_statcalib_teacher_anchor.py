"""Summarize the bounded T67 statcalib teacher-anchor benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


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
EXPECTED_PARAMETER_POINTS = [
    "default",
    "high_threshold",
]
EXPECTED_TEACHER_ANCHORS = [
    "ukf",
    "window_variance",
    "ekf",
]
EXPECTED_STATCALIB_VARIANTS = [
    {
        "mode": "statcalib_default_teacher_ukf",
        "parameter_point": "default",
        "teacher_anchor": "ukf",
    },
    {
        "mode": "statcalib_default_teacher_window_variance",
        "parameter_point": "default",
        "teacher_anchor": "window_variance",
    },
    {
        "mode": "statcalib_default_teacher_ekf",
        "parameter_point": "default",
        "teacher_anchor": "ekf",
    },
    {
        "mode": "statcalib_high_threshold_teacher_ukf",
        "parameter_point": "high_threshold",
        "teacher_anchor": "ukf",
    },
    {
        "mode": "statcalib_high_threshold_teacher_window_variance",
        "parameter_point": "high_threshold",
        "teacher_anchor": "window_variance",
    },
    {
        "mode": "statcalib_high_threshold_teacher_ekf",
        "parameter_point": "high_threshold",
        "teacher_anchor": "ekf",
    },
]
EXPECTED_STATCALIB_MODES = [item["mode"] for item in EXPECTED_STATCALIB_VARIANTS]
EXPECTED_ALL_MODES = EXPECTED_ANCHOR_MODES + EXPECTED_STATCALIB_MODES
DEFAULT_PACK_DIRNAME = "statcalib_teacher_anchor_summary"

VARIANT_METADATA = {
    item["mode"]: {
        "parameter_point": item["parameter_point"],
        "teacher_anchor": item["teacher_anchor"],
    }
    for item in EXPECTED_STATCALIB_VARIANTS
}


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to the T67 run root.")
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
        "parameter_point": metadata.get("parameter_point", ""),
        "teacher_anchor": metadata.get("teacher_anchor", ""),
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
        if float(row["coverage"]) != 1.0:
            raise ValueError(f"{key[0]}/{key[1]} coverage={row['coverage']} is not 1.0")

    for scenario in EXPECTED_SCENARIOS:
        for mode in EXPECTED_ALL_MODES:
            if (scenario, mode) not in seen_keys:
                raise ValueError(f"missing comparison row for {scenario}/{mode}")


def summarize_run(run_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    summary = _read_json(run_dir / "summary.json")
    launch_plan = _read_json(run_dir / "launch_plan.json")
    comparison_rows = [_normalize_row(row) for row in _read_csv_rows(run_dir / "comparison.csv")]
    _validate_run(summary=summary, launch_plan=launch_plan, rows=comparison_rows)

    index = {(row["scenario"], row["mode"]): row for row in comparison_rows}
    mode_summary_rows: list[dict[str, Any]] = []
    scenario_summary_rows: list[dict[str, Any]] = []
    variant_summary_rows: list[dict[str, Any]] = []
    teacher_anchor_summary_rows: list[dict[str, Any]] = []
    parameter_point_summary_rows: list[dict[str, Any]] = []

    variant_accumulator = {
        item["mode"]: {
            "mode": item["mode"],
            "parameter_point": item["parameter_point"],
            "teacher_anchor": item["teacher_anchor"],
            "mode_label": index[(EXPECTED_SCENARIOS[0], item["mode"])]["mode_label"],
            "final_ler_means": [],
            "rank_values": [],
            "ukf_gaps": [],
            "hybrid_gaps": [],
            "generated_windows": [],
            "signal_norms": [],
            "status_values": [],
            "reason_values": [],
            "win_count_within_statcalib": 0,
        }
        for item in EXPECTED_STATCALIB_VARIANTS
    }

    for scenario in EXPECTED_SCENARIOS:
        ukf_row = dict(index[(scenario, "ukf")])
        hybrid_row = dict(index[(scenario, "hybrid_residual_b")])
        scenario_mode_rows = [dict(index[(scenario, mode)]) for mode in EXPECTED_ALL_MODES]
        overall_winner = min(scenario_mode_rows, key=lambda row: (row["final_ler_mean"], row["mode"]))

        statcalib_rows = [dict(index[(scenario, mode)]) for mode in EXPECTED_STATCALIB_MODES]
        statcalib_ranked = sorted(statcalib_rows, key=lambda row: (row["final_ler_mean"], row["mode"]))
        statcalib_rank_by_mode = {
            row["mode"]: rank
            for rank, row in enumerate(statcalib_ranked, start=1)
        }
        best_statcalib = statcalib_ranked[0]

        per_parameter_rankings: dict[str, str] = {}
        for parameter_point in EXPECTED_PARAMETER_POINTS:
            ranked = sorted(
                [row for row in statcalib_rows if row["parameter_point"] == parameter_point],
                key=lambda row: (row["final_ler_mean"], row["mode"]),
            )
            per_parameter_rankings[parameter_point] = " > ".join(row["teacher_anchor"] for row in ranked)

        for rank, row in enumerate(statcalib_ranked, start=1):
            bucket = variant_accumulator[row["mode"]]
            bucket["final_ler_means"].append(row["final_ler_mean"])
            bucket["rank_values"].append(rank)
            bucket["ukf_gaps"].append(ukf_row["final_ler_mean"] - row["final_ler_mean"])
            bucket["hybrid_gaps"].append(hybrid_row["final_ler_mean"] - row["final_ler_mean"])
            bucket["generated_windows"].append(row["statcalib_generated_windows_mean"])
            bucket["signal_norms"].append(row["statcalib_signal_norm_mean_mean"])
            bucket["status_values"].append(row["statcalib_status"])
            bucket["reason_values"].append(row["statcalib_reason"])
            if rank == 1:
                bucket["win_count_within_statcalib"] += 1

        for row in scenario_mode_rows:
            gap_vs_ukf = ukf_row["final_ler_mean"] - row["final_ler_mean"]
            gap_vs_hybrid = hybrid_row["final_ler_mean"] - row["final_ler_mean"]
            mode_summary_rows.append(
                {
                    "scenario": row["scenario"],
                    "scenario_label": row["scenario_label"],
                    "mode": row["mode"],
                    "mode_label": row["mode_label"],
                    "parameter_point": row["parameter_point"],
                    "teacher_anchor": row["teacher_anchor"],
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
                    "is_statcalib_variant": row["mode"] in EXPECTED_STATCALIB_MODES,
                    "statcalib_rank_within_scenario": statcalib_rank_by_mode.get(row["mode"]),
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
                "scenario_label": best_statcalib["scenario_label"],
                "overall_winner_mode": overall_winner["mode"],
                "overall_winner_label": overall_winner["mode_label"],
                "overall_winner_final_ler_mean": overall_winner["final_ler_mean"],
                "ukf_final_ler_mean": ukf_row["final_ler_mean"],
                "hybrid_residual_b_final_ler_mean": hybrid_row["final_ler_mean"],
                "best_statcalib_mode": best_statcalib["mode"],
                "best_statcalib_label": best_statcalib["mode_label"],
                "best_statcalib_parameter_point": best_statcalib["parameter_point"],
                "best_statcalib_teacher_anchor": best_statcalib["teacher_anchor"],
                "best_statcalib_final_ler_mean": best_statcalib["final_ler_mean"],
                "best_statcalib_final_ler_std": best_statcalib["final_ler_std"],
                "best_statcalib_ukf_ler_minus_mode_ler": ukf_row["final_ler_mean"] - best_statcalib["final_ler_mean"],
                "best_statcalib_hybrid_ler_minus_mode_ler": hybrid_row["final_ler_mean"] - best_statcalib["final_ler_mean"],
                "best_statcalib_status": best_statcalib["statcalib_status"],
                "best_statcalib_reason": best_statcalib["statcalib_reason"],
                "best_statcalib_generated_windows_mean": best_statcalib["statcalib_generated_windows_mean"],
                "best_statcalib_signal_norm_mean_mean": best_statcalib["statcalib_signal_norm_mean_mean"],
                "default_teacher_anchor_ranking": per_parameter_rankings["default"],
                "high_threshold_teacher_anchor_ranking": per_parameter_rankings["high_threshold"],
                "statcalib_variant_ranking": " > ".join(row["mode"] for row in statcalib_ranked),
            }
        )

    for item in EXPECTED_STATCALIB_VARIANTS:
        mode = item["mode"]
        bucket = variant_accumulator[mode]
        variant_summary_rows.append(
            {
                "mode": mode,
                "mode_label": bucket["mode_label"],
                "parameter_point": bucket["parameter_point"],
                "teacher_anchor": bucket["teacher_anchor"],
                "scenario_count": len(bucket["final_ler_means"]),
                "mean_final_ler_mean": sum(bucket["final_ler_means"]) / len(bucket["final_ler_means"]),
                "mean_rank_within_statcalib": sum(bucket["rank_values"]) / len(bucket["rank_values"]),
                "win_count_within_statcalib": bucket["win_count_within_statcalib"],
                "mean_gap_vs_ukf": sum(bucket["ukf_gaps"]) / len(bucket["ukf_gaps"]),
                "mean_gap_vs_hybrid_residual_b": sum(bucket["hybrid_gaps"]) / len(bucket["hybrid_gaps"]),
                "beats_ukf_all_scenarios": all(gap > 0.0 for gap in bucket["ukf_gaps"]),
                "beats_hybrid_residual_b_all_scenarios": all(gap > 0.0 for gap in bucket["hybrid_gaps"]),
                "beats_both_frozen_anchors_all_scenarios": all(gap > 0.0 for gap in bucket["ukf_gaps"])
                and all(gap > 0.0 for gap in bucket["hybrid_gaps"]),
                "generated_windows_mean_across_scenarios": _mean(bucket["generated_windows"]),
                "signal_norm_mean_across_scenarios": _mean(bucket["signal_norms"]),
                "status_values": "|".join(sorted(set(bucket["status_values"]))),
                "reason_values": "|".join(sorted(set(bucket["reason_values"]))),
            }
        )
    variant_summary_rows.sort(key=lambda row: (row["mean_final_ler_mean"], row["mode"]))
    variant_by_mode = {row["mode"]: row for row in variant_summary_rows}

    for teacher_anchor in EXPECTED_TEACHER_ANCHORS:
        default_row = variant_by_mode[f"statcalib_default_teacher_{teacher_anchor}"]
        high_threshold_row = variant_by_mode[f"statcalib_high_threshold_teacher_{teacher_anchor}"]
        teacher_anchor_summary_rows.append(
            {
                "teacher_anchor": teacher_anchor,
                "default_mode": default_row["mode"],
                "default_mode_label": default_row["mode_label"],
                "default_mean_final_ler_mean": default_row["mean_final_ler_mean"],
                "default_mean_gap_vs_ukf": default_row["mean_gap_vs_ukf"],
                "default_mean_gap_vs_hybrid_residual_b": default_row["mean_gap_vs_hybrid_residual_b"],
                "default_beats_both_frozen_anchors_all_scenarios": default_row["beats_both_frozen_anchors_all_scenarios"],
                "high_threshold_mode": high_threshold_row["mode"],
                "high_threshold_mode_label": high_threshold_row["mode_label"],
                "high_threshold_mean_final_ler_mean": high_threshold_row["mean_final_ler_mean"],
                "high_threshold_mean_gap_vs_ukf": high_threshold_row["mean_gap_vs_ukf"],
                "high_threshold_mean_gap_vs_hybrid_residual_b": high_threshold_row["mean_gap_vs_hybrid_residual_b"],
                "high_threshold_beats_both_frozen_anchors_all_scenarios": high_threshold_row["beats_both_frozen_anchors_all_scenarios"],
                "default_mean_ler_minus_high_threshold_mean_ler": default_row["mean_final_ler_mean"] - high_threshold_row["mean_final_ler_mean"],
                "better_parameter_point_by_mean_ler": (
                    "default"
                    if default_row["mean_final_ler_mean"] < high_threshold_row["mean_final_ler_mean"]
                    else "high_threshold"
                ),
            }
        )

    for parameter_point in EXPECTED_PARAMETER_POINTS:
        rows = [
            variant_by_mode[f"statcalib_{parameter_point}_teacher_{teacher_anchor}"]
            for teacher_anchor in EXPECTED_TEACHER_ANCHORS
        ]
        ranked = sorted(rows, key=lambda row: (row["mean_final_ler_mean"], row["teacher_anchor"]))
        qualifying_non_ukf = [
            row["mode"]
            for row in rows
            if row["teacher_anchor"] != "ukf" and row["beats_both_frozen_anchors_all_scenarios"]
        ]
        parameter_point_summary_rows.append(
            {
                "parameter_point": parameter_point,
                "ukf_teacher_mode": rows[0]["mode"],
                "ukf_teacher_mean_final_ler_mean": next(row["mean_final_ler_mean"] for row in rows if row["teacher_anchor"] == "ukf"),
                "window_variance_teacher_mode": next(row["mode"] for row in rows if row["teacher_anchor"] == "window_variance"),
                "window_variance_teacher_mean_final_ler_mean": next(row["mean_final_ler_mean"] for row in rows if row["teacher_anchor"] == "window_variance"),
                "ekf_teacher_mode": next(row["mode"] for row in rows if row["teacher_anchor"] == "ekf"),
                "ekf_teacher_mean_final_ler_mean": next(row["mean_final_ler_mean"] for row in rows if row["teacher_anchor"] == "ekf"),
                "teacher_anchor_ranking": " > ".join(row["teacher_anchor"] for row in ranked),
                "best_teacher_anchor": ranked[0]["teacher_anchor"],
                "best_mode": ranked[0]["mode"],
                "best_mean_final_ler_mean": ranked[0]["mean_final_ler_mean"],
                "non_ukf_teacher_best": ranked[0]["teacher_anchor"] != "ukf",
                "non_ukf_variants_beating_both_frozen_anchors_all_scenarios": "|".join(qualifying_non_ukf),
                "any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios": bool(qualifying_non_ukf),
            }
        )

    qualifying_non_ukf_variants = [
        row["mode"]
        for row in variant_summary_rows
        if row["teacher_anchor"] != "ukf" and row["beats_both_frozen_anchors_all_scenarios"]
    ]

    summary_pack = {
        "run_dir": str(run_dir),
        "git_commit": summary.get("git_commit"),
        "launch_plan": str(run_dir / "launch_plan.json"),
        "comparison_csv": str(run_dir / "comparison.csv"),
        "expected_scenarios": EXPECTED_SCENARIOS,
        "expected_anchor_modes": EXPECTED_ANCHOR_MODES,
        "expected_parameter_points": EXPECTED_PARAMETER_POINTS,
        "expected_teacher_anchors": EXPECTED_TEACHER_ANCHORS,
        "expected_statcalib_modes": EXPECTED_STATCALIB_MODES,
        "scenario_summaries": scenario_summary_rows,
        "variant_aggregate_rankings": variant_summary_rows,
        "teacher_anchor_comparisons": teacher_anchor_summary_rows,
        "parameter_point_comparisons": parameter_point_summary_rows,
        "qualifying_non_ukf_variants_beating_both_frozen_anchors_all_scenarios": qualifying_non_ukf_variants,
        "any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios": bool(qualifying_non_ukf_variants),
    }

    output_dir = (run_dir / DEFAULT_PACK_DIRNAME) if output_dir is None else output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_summary_path = output_dir / "mode_summary.csv"
    scenario_summary_path = output_dir / "scenario_summary.csv"
    variant_summary_path = output_dir / "variant_summary.csv"
    teacher_anchor_summary_path = output_dir / "teacher_anchor_summary.csv"
    parameter_point_summary_path = output_dir / "parameter_point_summary.csv"
    summary_json_path = output_dir / "summary.json"

    _write_csv(
        mode_summary_path,
        mode_summary_rows,
        [
            "scenario",
            "scenario_label",
            "mode",
            "mode_label",
            "parameter_point",
            "teacher_anchor",
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
            "is_statcalib_variant",
            "statcalib_rank_within_scenario",
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
            "overall_winner_mode",
            "overall_winner_label",
            "overall_winner_final_ler_mean",
            "ukf_final_ler_mean",
            "hybrid_residual_b_final_ler_mean",
            "best_statcalib_mode",
            "best_statcalib_label",
            "best_statcalib_parameter_point",
            "best_statcalib_teacher_anchor",
            "best_statcalib_final_ler_mean",
            "best_statcalib_final_ler_std",
            "best_statcalib_ukf_ler_minus_mode_ler",
            "best_statcalib_hybrid_ler_minus_mode_ler",
            "best_statcalib_status",
            "best_statcalib_reason",
            "best_statcalib_generated_windows_mean",
            "best_statcalib_signal_norm_mean_mean",
            "default_teacher_anchor_ranking",
            "high_threshold_teacher_anchor_ranking",
            "statcalib_variant_ranking",
        ],
    )
    _write_csv(
        variant_summary_path,
        variant_summary_rows,
        [
            "mode",
            "mode_label",
            "parameter_point",
            "teacher_anchor",
            "scenario_count",
            "mean_final_ler_mean",
            "mean_rank_within_statcalib",
            "win_count_within_statcalib",
            "mean_gap_vs_ukf",
            "mean_gap_vs_hybrid_residual_b",
            "beats_ukf_all_scenarios",
            "beats_hybrid_residual_b_all_scenarios",
            "beats_both_frozen_anchors_all_scenarios",
            "generated_windows_mean_across_scenarios",
            "signal_norm_mean_across_scenarios",
            "status_values",
            "reason_values",
        ],
    )
    _write_csv(
        teacher_anchor_summary_path,
        teacher_anchor_summary_rows,
        [
            "teacher_anchor",
            "default_mode",
            "default_mode_label",
            "default_mean_final_ler_mean",
            "default_mean_gap_vs_ukf",
            "default_mean_gap_vs_hybrid_residual_b",
            "default_beats_both_frozen_anchors_all_scenarios",
            "high_threshold_mode",
            "high_threshold_mode_label",
            "high_threshold_mean_final_ler_mean",
            "high_threshold_mean_gap_vs_ukf",
            "high_threshold_mean_gap_vs_hybrid_residual_b",
            "high_threshold_beats_both_frozen_anchors_all_scenarios",
            "default_mean_ler_minus_high_threshold_mean_ler",
            "better_parameter_point_by_mean_ler",
        ],
    )
    _write_csv(
        parameter_point_summary_path,
        parameter_point_summary_rows,
        [
            "parameter_point",
            "ukf_teacher_mode",
            "ukf_teacher_mean_final_ler_mean",
            "window_variance_teacher_mode",
            "window_variance_teacher_mean_final_ler_mean",
            "ekf_teacher_mode",
            "ekf_teacher_mean_final_ler_mean",
            "teacher_anchor_ranking",
            "best_teacher_anchor",
            "best_mode",
            "best_mean_final_ler_mean",
            "non_ukf_teacher_best",
            "non_ukf_variants_beating_both_frozen_anchors_all_scenarios",
            "any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios",
        ],
    )
    _write_json(summary_json_path, summary_pack)

    return {
        "mode_summary_rows": mode_summary_rows,
        "scenario_summary_rows": scenario_summary_rows,
        "variant_summary_rows": variant_summary_rows,
        "teacher_anchor_summary_rows": teacher_anchor_summary_rows,
        "parameter_point_summary_rows": parameter_point_summary_rows,
        "summary_pack": summary_pack,
        "output_dir": str(output_dir),
        "mode_summary_path": str(mode_summary_path),
        "scenario_summary_path": str(scenario_summary_path),
        "variant_summary_path": str(variant_summary_path),
        "teacher_anchor_summary_path": str(teacher_anchor_summary_path),
        "parameter_point_summary_path": str(parameter_point_summary_path),
        "summary_json_path": str(summary_json_path),
    }


def _render_console_summary(result: Mapping[str, Any]) -> str:
    summary_pack = dict(result["summary_pack"])
    lines = [
        "T67 statcalib teacher-anchor summary",
        f"- run_dir: {summary_pack['run_dir']}",
        f"- git_commit: {summary_pack['git_commit']}",
        f"- any_non_ukf_variant_beats_both_all_scenarios: {summary_pack['any_non_ukf_variant_beats_both_frozen_anchors_all_scenarios']}",
        "- qualifying_non_ukf_variants: "
        + (
            ", ".join(summary_pack["qualifying_non_ukf_variants_beating_both_frozen_anchors_all_scenarios"])
            if summary_pack["qualifying_non_ukf_variants_beating_both_frozen_anchors_all_scenarios"]
            else "<none>"
        ),
    ]
    for row in result["scenario_summary_rows"]:
        lines.append(
            f"- {row['scenario']}: best_statcalib={row['best_statcalib_mode']} "
            f"teacher={row['best_statcalib_teacher_anchor']} "
            f"parameter_point={row['best_statcalib_parameter_point']} "
            f"LER={row['best_statcalib_final_ler_mean']:.6f}"
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
