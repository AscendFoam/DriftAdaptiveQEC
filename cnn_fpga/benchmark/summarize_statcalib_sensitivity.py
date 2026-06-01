"""Summarize the bounded T66 statcalib sensitivity benchmark artifacts."""

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
EXPECTED_STATCALIB_MODES = [
    "statcalib_default",
    "statcalib_low_scale",
    "statcalib_high_scale",
    "statcalib_low_clip",
    "statcalib_high_threshold",
]
EXPECTED_ALL_MODES = EXPECTED_ANCHOR_MODES + EXPECTED_STATCALIB_MODES
DEFAULT_PACK_DIRNAME = "statcalib_sensitivity_summary"


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Path to the T66 run root.")
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


def _normalize_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "scenario": str(row["scenario"]),
        "scenario_label": str(row["scenario_label"]),
        "mode": str(row["mode"]),
        "mode_label": str(row["mode_label"]),
        "completed_repeats": int(row["completed_repeats"]),
        "expected_repeats": int(row["expected_repeats"]),
        "coverage": float(row["coverage"]),
        "final_ler_mean": float(row["final_ler_mean"]),
        "final_ler_std": float(row["final_ler_std"]),
        "overflow_rate_mean": float(row["overflow_rate_mean"]),
        "overflow_rate_std": float(row["overflow_rate_std"]),
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
    scenario_json_rows: list[dict[str, Any]] = []
    variant_aggregate_rows: list[dict[str, Any]] = []

    statcalib_accumulator = {
        mode: {
            "mode": mode,
            "mode_label": index[(EXPECTED_SCENARIOS[0], mode)]["mode_label"],
            "final_ler_means": [],
            "rank_values": [],
            "win_count_within_statcalib": 0,
        }
        for mode in EXPECTED_STATCALIB_MODES
    }

    for scenario in EXPECTED_SCENARIOS:
        ukf_row = dict(index[(scenario, "ukf")])
        hybrid_row = dict(index[(scenario, "hybrid_residual_b")])
        scenario_mode_rows = [dict(index[(scenario, mode)]) for mode in EXPECTED_ALL_MODES]
        overall_winner = min(scenario_mode_rows, key=lambda row: row["final_ler_mean"])

        statcalib_rows = [dict(index[(scenario, mode)]) for mode in EXPECTED_STATCALIB_MODES]
        statcalib_ranked = sorted(statcalib_rows, key=lambda row: (row["final_ler_mean"], row["mode"]))
        statcalib_rank_by_mode = {
            row["mode"]: rank
            for rank, row in enumerate(statcalib_ranked, start=1)
        }
        best_statcalib = statcalib_ranked[0]
        best_vs_ukf = ukf_row["final_ler_mean"] - best_statcalib["final_ler_mean"]
        best_vs_hybrid = hybrid_row["final_ler_mean"] - best_statcalib["final_ler_mean"]

        for rank, row in enumerate(statcalib_ranked, start=1):
            bucket = statcalib_accumulator[row["mode"]]
            bucket["final_ler_means"].append(row["final_ler_mean"])
            bucket["rank_values"].append(rank)
            if rank == 1:
                bucket["win_count_within_statcalib"] += 1

        ranked_mode_string = " > ".join(row["mode"] for row in statcalib_ranked)
        ranked_label_string = " > ".join(row["mode_label"] for row in statcalib_ranked)

        for row in scenario_mode_rows:
            mode_summary_rows.append(
                {
                    "scenario": row["scenario"],
                    "scenario_label": row["scenario_label"],
                    "mode": row["mode"],
                    "mode_label": row["mode_label"],
                    "completed_repeats": row["completed_repeats"],
                    "expected_repeats": row["expected_repeats"],
                    "coverage": row["coverage"],
                    "final_ler_mean": row["final_ler_mean"],
                    "final_ler_std": row["final_ler_std"],
                    "overflow_rate_mean": row["overflow_rate_mean"],
                    "overflow_rate_std": row["overflow_rate_std"],
                    "ukf_ler_minus_mode_ler": ukf_row["final_ler_mean"] - row["final_ler_mean"],
                    "hybrid_ler_minus_mode_ler": hybrid_row["final_ler_mean"] - row["final_ler_mean"],
                    "is_statcalib_variant": row["mode"] in EXPECTED_STATCALIB_MODES,
                    "statcalib_rank_within_scenario": statcalib_rank_by_mode.get(row["mode"]),
                    "statcalib_status": row["statcalib_status"],
                    "statcalib_reason": row["statcalib_reason"],
                    "statcalib_generated_repeats": row["statcalib_generated_repeats"],
                    "statcalib_generated_windows_mean": row["statcalib_generated_windows_mean"],
                    "statcalib_signal_norm_mean_mean": row["statcalib_signal_norm_mean_mean"],
                }
            )

        scenario_summary_row = {
            "scenario": scenario,
            "scenario_label": best_statcalib["scenario_label"],
            "overall_winner_mode": overall_winner["mode"],
            "overall_winner_label": overall_winner["mode_label"],
            "overall_winner_final_ler_mean": overall_winner["final_ler_mean"],
            "ukf_final_ler_mean": ukf_row["final_ler_mean"],
            "hybrid_residual_b_final_ler_mean": hybrid_row["final_ler_mean"],
            "best_statcalib_mode": best_statcalib["mode"],
            "best_statcalib_label": best_statcalib["mode_label"],
            "best_statcalib_final_ler_mean": best_statcalib["final_ler_mean"],
            "best_statcalib_final_ler_std": best_statcalib["final_ler_std"],
            "best_statcalib_ukf_ler_minus_mode_ler": best_vs_ukf,
            "best_statcalib_hybrid_ler_minus_mode_ler": best_vs_hybrid,
            "best_statcalib_beats_ukf": best_vs_ukf > 0.0,
            "best_statcalib_beats_hybrid_residual_b": best_vs_hybrid > 0.0,
            "best_statcalib_status": best_statcalib["statcalib_status"],
            "best_statcalib_reason": best_statcalib["statcalib_reason"],
            "best_statcalib_generated_windows_mean": best_statcalib["statcalib_generated_windows_mean"],
            "best_statcalib_signal_norm_mean_mean": best_statcalib["statcalib_signal_norm_mean_mean"],
            "statcalib_mode_ranking": ranked_mode_string,
            "statcalib_label_ranking": ranked_label_string,
        }
        scenario_summary_rows.append(scenario_summary_row)
        scenario_json_rows.append(
            {
                **scenario_summary_row,
                "statcalib_variant_rankings": [
                    {
                        "mode": row["mode"],
                        "mode_label": row["mode_label"],
                        "rank_within_statcalib": rank,
                        "final_ler_mean": row["final_ler_mean"],
                        "final_ler_std": row["final_ler_std"],
                        "ukf_ler_minus_mode_ler": ukf_row["final_ler_mean"] - row["final_ler_mean"],
                        "hybrid_ler_minus_mode_ler": hybrid_row["final_ler_mean"] - row["final_ler_mean"],
                        "statcalib_status": row["statcalib_status"],
                        "statcalib_reason": row["statcalib_reason"],
                        "statcalib_generated_windows_mean": row["statcalib_generated_windows_mean"],
                        "statcalib_signal_norm_mean_mean": row["statcalib_signal_norm_mean_mean"],
                    }
                    for rank, row in enumerate(statcalib_ranked, start=1)
                ],
            }
        )

    for mode in EXPECTED_STATCALIB_MODES:
        bucket = statcalib_accumulator[mode]
        variant_aggregate_rows.append(
            {
                "mode": mode,
                "mode_label": bucket["mode_label"],
                "scenario_count": len(bucket["final_ler_means"]),
                "mean_final_ler_mean": sum(bucket["final_ler_means"]) / len(bucket["final_ler_means"]),
                "mean_rank_within_statcalib": sum(bucket["rank_values"]) / len(bucket["rank_values"]),
                "win_count_within_statcalib": bucket["win_count_within_statcalib"],
            }
        )
    variant_aggregate_rows.sort(key=lambda row: (row["mean_final_ler_mean"], row["mode"]))
    best_variant_overall = dict(variant_aggregate_rows[0])

    best_variant_mode = best_variant_overall["mode"]
    best_variant_rows = [dict(index[(scenario, best_variant_mode)]) for scenario in EXPECTED_SCENARIOS]

    summary_pack = {
        "run_dir": str(run_dir),
        "git_commit": summary.get("git_commit"),
        "launch_plan": str(run_dir / "launch_plan.json"),
        "comparison_csv": str(run_dir / "comparison.csv"),
        "expected_scenarios": EXPECTED_SCENARIOS,
        "expected_anchor_modes": EXPECTED_ANCHOR_MODES,
        "expected_statcalib_modes": EXPECTED_STATCALIB_MODES,
        "scenario_summaries": scenario_json_rows,
        "variant_aggregate_rankings": variant_aggregate_rows,
        "best_variant_overall": best_variant_overall,
        "best_variant_overall_per_scenario": [
            {
                "scenario": row["scenario"],
                "scenario_label": row["scenario_label"],
                "mode": row["mode"],
                "mode_label": row["mode_label"],
                "final_ler_mean": row["final_ler_mean"],
                "ukf_ler_minus_mode_ler": index[(row["scenario"], "ukf")]["final_ler_mean"] - row["final_ler_mean"],
                "hybrid_ler_minus_mode_ler": index[(row["scenario"], "hybrid_residual_b")]["final_ler_mean"] - row["final_ler_mean"],
            }
            for row in best_variant_rows
        ],
        "best_variant_beats_ukf_all_scenarios": all(
            index[(scenario, "ukf")]["final_ler_mean"] > index[(scenario, best_variant_mode)]["final_ler_mean"]
            for scenario in EXPECTED_SCENARIOS
        ),
        "best_variant_beats_hybrid_all_scenarios": all(
            index[(scenario, "hybrid_residual_b")]["final_ler_mean"] > index[(scenario, best_variant_mode)]["final_ler_mean"]
            for scenario in EXPECTED_SCENARIOS
        ),
    }

    output_dir = (run_dir / DEFAULT_PACK_DIRNAME) if output_dir is None else output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_summary_path = output_dir / "mode_summary.csv"
    scenario_summary_path = output_dir / "scenario_summary.csv"
    summary_json_path = output_dir / "summary.json"

    _write_csv(
        mode_summary_path,
        mode_summary_rows,
        [
            "scenario",
            "scenario_label",
            "mode",
            "mode_label",
            "completed_repeats",
            "expected_repeats",
            "coverage",
            "final_ler_mean",
            "final_ler_std",
            "overflow_rate_mean",
            "overflow_rate_std",
            "ukf_ler_minus_mode_ler",
            "hybrid_ler_minus_mode_ler",
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
            "best_statcalib_final_ler_mean",
            "best_statcalib_final_ler_std",
            "best_statcalib_ukf_ler_minus_mode_ler",
            "best_statcalib_hybrid_ler_minus_mode_ler",
            "best_statcalib_beats_ukf",
            "best_statcalib_beats_hybrid_residual_b",
            "best_statcalib_status",
            "best_statcalib_reason",
            "best_statcalib_generated_windows_mean",
            "best_statcalib_signal_norm_mean_mean",
            "statcalib_mode_ranking",
            "statcalib_label_ranking",
        ],
    )
    _write_json(summary_json_path, summary_pack)

    return {
        "mode_summary_rows": mode_summary_rows,
        "scenario_summary_rows": scenario_summary_rows,
        "summary_pack": summary_pack,
        "output_dir": str(output_dir),
        "mode_summary_path": str(mode_summary_path),
        "scenario_summary_path": str(scenario_summary_path),
        "summary_json_path": str(summary_json_path),
    }


def _render_console_summary(result: Mapping[str, Any]) -> str:
    summary_pack = dict(result["summary_pack"])
    best_variant = dict(summary_pack["best_variant_overall"])
    lines = [
        "T66 statcalib sensitivity summary",
        f"- run_dir: {summary_pack['run_dir']}",
        f"- git_commit: {summary_pack['git_commit']}",
        f"- best_variant_overall: {best_variant['mode']} ({best_variant['mode_label']})",
        f"- mean_final_ler_mean: {best_variant['mean_final_ler_mean']:.6f}",
        f"- mean_rank_within_statcalib: {best_variant['mean_rank_within_statcalib']:.3f}",
    ]
    for row in result["scenario_summary_rows"]:
        lines.append(
            f"- {row['scenario']}: best_statcalib={row['best_statcalib_mode']} "
            f"LER={row['best_statcalib_final_ler_mean']:.6f} "
            f"ukf_minus_mode={row['best_statcalib_ukf_ler_minus_mode_ler']:.6f} "
            f"hybrid_minus_mode={row['best_statcalib_hybrid_ler_minus_mode_ler']:.6f}"
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
