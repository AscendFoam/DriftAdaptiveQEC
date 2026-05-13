from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"

PAIR_SUMMARY_DIR = RUNS / "teachrepr_v5_chunked_pair" / "paired_20260427_220702"
PAIR_SUMMARY_CSV = PAIR_SUMMARY_DIR / "summary.csv"
PAIR_SUMMARY_JSON = PAIR_SUMMARY_DIR / "summary.json"

PRIMARY_BASE = RUNS / "teachrepr_v5_chunked" / "p4_benchmark"
SEED_TO_RUN = {
    "20260427": "trp60427_resume",
    "20260428": "trp60428_resume",
    "20260429": "trp60429_resume",
}

LEGACY_20260429_COMPARISON = (
    RUNS
    / "teachrepr"
    / "p4_benchmark"
    / "trp60429_20260427_142013_2a59bc_24060"
    / "comparison.csv"
)

SCENARIO_FOLDERS = {
    "static_bias_theta": "static",
    "linear_ramp": "linear",
    "step_sigma_theta": "stepsi",
    "periodic_drift": "period",
}

MODE_FOLDERS = {
    "hybrid_full": "hybrid",
    "hybrid_gated_teacher_v5": "hybri1",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: str) -> float:
    return float(value)


def _pair_summary_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in _read_csv(PAIR_SUMMARY_CSV):
        rows.append(
            {
                "seed": row["seed"],
                "avg_ler_hybrid_full": _float(row["avg_ler_hybrid_full"]),
                "avg_ler_hybrid_gated_teacher_v5": _float(
                    row["avg_ler_hybrid_gated_teacher_v5"]
                ),
                "avg_gap_hybrid_gated_teacher_v5_vs_hybrid_full": _float(
                    row["avg_gap_hybrid_gated_teacher_v5_vs_hybrid_full"]
                ),
                "benchmark_summary": row["benchmark_summary"],
                "comparison_csv": row["comparison_csv"],
                "teacher_scalar_diagnostics_csv": row["teacher_scalar_diagnostics_csv"],
            }
        )
    return rows


def _comparison_rows(seed: str) -> List[Dict[str, object]]:
    run_dir = PRIMARY_BASE / SEED_TO_RUN[seed]
    rows: List[Dict[str, object]] = []
    for row in _read_csv(run_dir / "comparison.csv"):
        rows.append(
            {
                "scenario": row["scenario"],
                "mode": row["mode"],
                "final_ler_mean": _float(row["final_ler_mean"]),
                "final_ler_std": _float(row["final_ler_std"]),
                "overflow_rate_mean": _float(row["overflow_rate_mean"]),
                "correction_saturation_rate_mean": _float(
                    row["correction_saturation_rate_mean"]
                ),
                "aggressive_param_rate_mean": _float(row["aggressive_param_rate_mean"]),
                "n_commits_applied_mean": _float(row["n_commits_applied_mean"]),
                "slow_update_violation_rate_mean": _float(
                    row["slow_update_violation_rate_mean"]
                ),
                "fast_cycle_violation_rate_mean": _float(
                    row["fast_cycle_violation_rate_mean"]
                ),
                "teacher_contribution_l2_mean_mean": _float(
                    row["teacher_contribution_l2_mean_mean"]
                ),
                "teacher_scalar_abs_mean_mean": _float(
                    row["teacher_scalar_abs_mean_mean"]
                ),
                "teacher_gate_mean_mean": _float(row["teacher_gate_mean_mean"]),
                "teacher_gate_std_mean": _float(row["teacher_gate_std_mean"]),
                "dominant_overflow_source": row["dominant_overflow_source"],
            }
        )
    return rows


def _scenario_gap_table() -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for seed in sorted(SEED_TO_RUN):
        grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
        for row in _comparison_rows(seed):
            grouped.setdefault(row["scenario"], {})[row["mode"]] = row
        for scenario in sorted(grouped):
            full = grouped[scenario]["hybrid_full"]
            gated = grouped[scenario]["hybrid_gated_teacher_v5"]
            out.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "full_ler": full["final_ler_mean"],
                    "gated_v5_ler": gated["final_ler_mean"],
                    "gap_gated_minus_full": gated["final_ler_mean"]
                    - full["final_ler_mean"],
                    "gated_final_ler_std": gated["final_ler_std"],
                    "full_final_ler_std": full["final_ler_std"],
                    "full_overflow_rate_mean": full["overflow_rate_mean"],
                    "gated_overflow_rate_mean": gated["overflow_rate_mean"],
                    "gated_aggressive_param_rate_mean": gated[
                        "aggressive_param_rate_mean"
                    ],
                    "gated_teacher_contribution_l2_mean_mean": gated[
                        "teacher_contribution_l2_mean_mean"
                    ],
                    "gated_teacher_gate_mean_mean": gated["teacher_gate_mean_mean"],
                    "gated_teacher_gate_std_mean": gated["teacher_gate_std_mean"],
                }
            )
    return out


def _teacher_scalar_averages() -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seed, run_name in SEED_TO_RUN.items():
        path = PRIMARY_BASE / run_name / "teacher_scalar_diagnostics.csv"
        accum: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}
        for row in _read_csv(path):
            scalar = row["scalar_name"]
            slot = accum.setdefault(
                scalar,
                {
                    "ablation_l2_mean_avg": 0.0,
                    "gate_delta_l2_mean_avg": 0.0,
                    "raw_std_avg": 0.0,
                    "normalized_std_avg": 0.0,
                },
            )
            slot["ablation_l2_mean_avg"] += _float(row["ablation_l2_mean"])
            slot["gate_delta_l2_mean_avg"] += _float(row["gate_delta_l2_mean"])
            slot["raw_std_avg"] += _float(row["raw_std"])
            slot["normalized_std_avg"] += _float(row["normalized_std"])
            counts[scalar] = counts.get(scalar, 0) + 1
        out[seed] = {}
        for scalar, slot in accum.items():
            n = counts[scalar]
            out[seed][scalar] = {
                key: value / n for key, value in slot.items()
            }
    return out


def _hil_summary_path(scenario: str, mode: str, repeat_name: str) -> Path:
    folder = SCENARIO_FOLDERS[scenario]
    mode_folder = MODE_FOLDERS[mode]
    return (
        PRIMARY_BASE
        / SEED_TO_RUN["20260429"]
        / folder
        / mode_folder
        / repeat_name
        / "hil_summary.json"
    )


def _slow_update_stats_20260429() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for scenario in SCENARIO_FOLDERS:
        out[scenario] = {}
        for mode in MODE_FOLDERS:
            out[scenario][mode] = {}
            for repeat_name in ("repeat_00", "repeat_01"):
                data = _read_json(_hil_summary_path(scenario, mode, repeat_name))
                out[scenario][mode][repeat_name] = {
                    "slow_update_mean_us": data["slow_update_mean_us"],
                    "slow_update_p95_us": data["slow_update_p95_us"],
                    "slow_update_p99_us": data["slow_update_p99_us"],
                    "n_commits_applied": data["n_commits_applied"],
                    "slow_update_violation_rate": data["slow_update_violation_rate"],
                    "fast_cycle_violation_rate": data["fast_cycle_violation_rate"],
                }
    return out


def _repeat_snapshot_summary_20260429() -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for scenario in SCENARIO_FOLDERS:
        out[scenario] = {}
        for mode in MODE_FOLDERS:
            out[scenario][mode] = {}
            for repeat_name in ("repeat_00", "repeat_01"):
                data = _read_json(_hil_summary_path(scenario, mode, repeat_name))
                active_bank = data["final_snapshot"]["param_bank"]["active_bank"]
                params = data["final_snapshot"]["param_bank"]["banks"][active_bank]["params"]
                pred = params["metadata"]["prediction"]
                diag = pred["metadata"]["window_diagnostics"]
                bq, bp = params["b"]
                out[scenario][mode][repeat_name] = {
                    "final_ler": data["final_ler"],
                    "b_q": bq,
                    "b_p": bp,
                    "b_norm": math.sqrt(bq * bq + bp * bp),
                    "teacher_mu_q": pred["mu_q"],
                    "teacher_mu_p": pred["mu_p"],
                    "teacher_theta_deg": pred["theta_deg"],
                    "anisotropy_ratio": pred["metadata"]["observation"][
                        "anisotropy_ratio"
                    ],
                    "window_ler": diag["window_ler"],
                    "mean_correction_utilization": diag["mean_correction_utilization"],
                    "mean_active_param_bias_norm": diag["mean_active_param_bias_norm"],
                    "aggressive_param_rate": data["aggressive_param_rate"],
                }
    return out


def _legacy_context() -> Dict[str, object]:
    rows = _read_csv(LEGACY_20260429_COMPARISON)
    full_rows = [r for r in rows if r["mode"] == "hybrid_full"]
    gated_rows = [r for r in rows if r["mode"] == "hybrid_gated_teacher_v5"]
    avg_full = sum(_float(r["final_ler_mean"]) for r in full_rows) / len(full_rows)
    avg_gated = sum(_float(r["final_ler_mean"]) for r in gated_rows) / len(gated_rows)
    return {
        "path": str(LEGACY_20260429_COMPARISON),
        "avg_full_ler": avg_full,
        "avg_gated_v5_ler": avg_gated,
        "avg_gap_gated_minus_full": avg_gated - avg_full,
        "n_commits_applied_mean_values": sorted(
            {
                _float(r["n_commits_applied_mean"])
                for r in rows
            }
        ),
        "note": (
            "Context only. This older run shape differs from the paired/chunked "
            "rerun and should not be merged into the primary diagnosis table."
        ),
    }


def _repeat_hil_summary_inventory() -> List[str]:
    paths: List[str] = []
    for scenario in SCENARIO_FOLDERS:
        for mode in MODE_FOLDERS:
            for repeat_name in ("repeat_00", "repeat_01"):
                paths.append(str(_hil_summary_path(scenario, mode, repeat_name)))
    return paths


def build_report() -> Dict[str, object]:
    primary_summary = _read_json(PRIMARY_BASE / SEED_TO_RUN["20260429"] / "summary.json")
    return {
        "artifact_inventory": {
            "pair_summary_csv": str(PAIR_SUMMARY_CSV),
            "pair_summary_json": str(PAIR_SUMMARY_JSON),
            "primary_run_summary_json": str(
                PRIMARY_BASE / SEED_TO_RUN["20260429"] / "summary.json"
            ),
            "comparison_csvs": {
                seed: str(PRIMARY_BASE / run_name / "comparison.csv")
                for seed, run_name in SEED_TO_RUN.items()
            },
            "teacher_scalar_diagnostics_csvs": {
                seed: str(PRIMARY_BASE / run_name / "teacher_scalar_diagnostics.csv")
                for seed, run_name in SEED_TO_RUN.items()
            },
            "seed20260429_repeat_hil_summaries": _repeat_hil_summary_inventory(),
            "legacy_context_comparison_csv": str(LEGACY_20260429_COMPARISON),
        },
        "primary_protocol_context": {
            "protocol_id": primary_summary["protocol"]["protocol_id"],
            "repeats": primary_summary["protocol"]["repeats"],
            "paired_seeds": primary_summary["protocol"]["paired_seeds"],
            "frozen_baseline_set": primary_summary["protocol"]["frozen_baseline_set"],
            "scenario_filter": primary_summary["filters"]["scenario"],
        },
        "paired_seed_averages": _pair_summary_rows(),
        "scenario_gap_table": _scenario_gap_table(),
        "teacher_scalar_averages_by_seed": _teacher_scalar_averages(),
        "slow_update_stats_20260429": _slow_update_stats_20260429(),
        "repeat_snapshot_summary_20260429": _repeat_snapshot_summary_20260429(),
        "legacy_context_20260429": _legacy_context(),
        "trace_boundary_note": {
            "full_committed_parameter_timeseries_available": False,
            "reason": (
                "Current artifacts expose aggregate comparison rows and repeat-level "
                "final snapshots, but not the full per-window/per-commit committed "
                "parameter trajectory."
            ),
        },
    }


def main() -> None:
    print(json.dumps(build_report(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
