"""Run frozen P4 multi-scenario benchmark across formal baseline set."""

# 中文说明：
# - 该脚本用于 P4 主线：在固定场景集、固定 repeats、固定报告模板下，
#   比较 `Static Linear / Window Variance / EKF / CNN-FPGA` 四类正式 baseline。
# - 默认口径不包含 `real_board`；真板仍保留为条件性扩展，不阻塞主线统计结论。

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from cnn_fpga.benchmark.run_hil_suite import run_hil_session
from cnn_fpga.utils.config import config_hash, ensure_dir, load_yaml_config, now_tag, open_text, save_json, write_text


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen P4 multi-scenario benchmark.")
    parser.add_argument("--config", required=True, type=str, help="Path to P4 benchmark YAML config")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional scenario name filter. Can be provided multiple times.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Optional benchmark mode filter. Can be provided multiple times.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Optional repeat override for smoke/debug runs.",
    )
    parser.add_argument(
        "--paired-seeds",
        action="store_true",
        help="Force all modes within a scenario/repeat to use the same seed.",
    )
    return parser


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config_with_bases(cfg_path: Path) -> Dict[str, Any]:
    cfg = load_yaml_config(cfg_path)
    base_ref = cfg.get("base_config")
    if not base_ref:
        return cfg
    base_cfg = _load_config_with_bases((cfg_path.parent / str(base_ref)).resolve())
    return _deep_merge(base_cfg, {key: value for key, value in cfg.items() if key != "base_config"})


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _normalize_entries(entries: Sequence[Dict[str, Any]], filter_names: Sequence[str], field_name: str) -> List[Dict[str, Any]]:
    normalized = [dict(item) for item in entries]
    if not normalized:
        raise ValueError(f"No {field_name} configured.")
    if not filter_names:
        return normalized
    filter_set = {str(item) for item in filter_names}
    selected = [item for item in normalized if str(item.get("name")) in filter_set]
    if not selected:
        raise ValueError(f"No {field_name} matched filter: {sorted(filter_set)}")
    return selected


def _build_mode_config(base_config: Dict[str, Any], mode_entry: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(base_config)
    slow_cfg = cfg.setdefault("slow_loop", {})
    artifact_cfg = slow_cfg.setdefault("model_artifact", {})
    infer_cfg = slow_cfg.setdefault("inference_service", {})
    if "slow_loop_mode" in mode_entry:
        slow_cfg["mode"] = str(mode_entry["slow_loop_mode"]).lower()
    if "artifact_selector" in mode_entry:
        artifact_cfg["artifact_selector"] = str(mode_entry["artifact_selector"])
    if "artifact_path" in mode_entry:
        artifact_cfg["path"] = mode_entry["artifact_path"]
    if "inference_backend" in mode_entry:
        infer_cfg["backend"] = str(mode_entry["inference_backend"]).lower()
    if "inference_service_mode" in mode_entry:
        infer_cfg["mode"] = str(mode_entry["inference_service_mode"]).lower()
    if "overrides" in mode_entry:
        cfg = _deep_merge(cfg, dict(mode_entry["overrides"]))
    return cfg


def _build_run_config(base_config: Dict[str, Any], scenario_entry: Dict[str, Any], mode_entry: Dict[str, Any], seed: int) -> Dict[str, Any]:
    cfg = _build_mode_config(base_config, mode_entry)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = int(seed)
    scenario_overrides = dict(scenario_entry.get("overrides", {}))
    if scenario_overrides:
        cfg = _deep_merge(cfg, scenario_overrides)
    return cfg


def _aggregate_metric(items: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    values = np.asarray([float(item[key]) for item in items], dtype=float)
    return {
        f"{key}_mean": float(np.mean(values)),
        f"{key}_std": float(np.std(values)),
    }


def _aggregate_per_scalar(items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float | None]]:
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for item in items:
        diag = item.get("teacher_per_scalar", {})
        if not isinstance(diag, dict):
            continue
        for scalar_name, scalar_stats in diag.items():
            if not isinstance(scalar_stats, dict):
                continue
            bucket = buckets.setdefault(str(scalar_name), {})
            for stat_name, value in scalar_stats.items():
                if value is None:
                    continue
                bucket.setdefault(str(stat_name), []).append(float(value))

    out: Dict[str, Dict[str, float | None]] = {}
    for scalar_name, stats in buckets.items():
        out[scalar_name] = {}
        for stat_name, values in stats.items():
            array = np.asarray(values, dtype=float)
            out[scalar_name][stat_name] = None if array.size == 0 else float(np.mean(array))
    return out


def _dominant_source(items: Sequence[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for item in items:
        source = str(item.get("dominant_overflow_source", "none"))
        counts[source] = counts.get(source, 0) + 1
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return ordered[0][0] if ordered else "none"


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with open_text(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_per_scalar_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "scenario",
        "scenario_label",
        "mode",
        "mode_label",
        "scalar_name",
        "raw_mean",
        "raw_std",
        "raw_min",
        "raw_max",
        "normalized_mean",
        "normalized_std",
        "normalized_min",
        "normalized_max",
        "ablation_l2_mean",
        "ablation_l2_std",
        "gate_delta_mean_mean",
        "gate_delta_l2_mean",
    ]
    flat_rows: List[Dict[str, Any]] = []
    for row in rows:
        per_scalar = row.get("teacher_per_scalar", {})
        if not isinstance(per_scalar, dict):
            continue
        for scalar_name, scalar_stats in per_scalar.items():
            if not isinstance(scalar_stats, dict):
                continue
            flat_rows.append(
                {
                    "scenario": row.get("scenario"),
                    "scenario_label": row.get("scenario_label"),
                    "mode": row.get("mode"),
                    "mode_label": row.get("mode_label"),
                    "scalar_name": scalar_name,
                    **{field: scalar_stats.get(field) for field in fields if field not in {"scenario", "scenario_label", "mode", "mode_label", "scalar_name"}},
                }
            )
    _write_csv(path, flat_rows, fields)


def _write_report(
    path: Path,
    *,
    protocol: Dict[str, Any],
    comparison_rows: List[Dict[str, Any]],
    delta_rows: List[Dict[str, Any]],
    scenario_winners: List[Dict[str, Any]],
) -> None:
    lines = [
        "# P4 Multi-Scenario Benchmark",
        "",
        "## Protocol",
        "",
        f"- protocol_id: `{protocol.get('protocol_id', 'unknown')}`",
        f"- report_template_version: `{protocol.get('report_template_version', 'unknown')}`",
        f"- repeats: `{protocol.get('repeats', 0)}`",
        f"- real_board_policy: `{protocol.get('real_board_policy', 'conditional_extension')}`",
        "",
        "## Scenario Comparison",
        "",
        "| Scenario | Mode | LER Mean | LER Std | Overflow Mean | Hist Sat Mean | Commit Mean | Slow Viol Mean | Fast Viol Mean | Dominant Source | Artifact |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in comparison_rows:
        def _fmt(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        lines.append(
            f"| {row['scenario']} | {row['mode_label']} | {_fmt(row['final_ler_mean'])} | {_fmt(row['final_ler_std'])} | "
            f"{_fmt(row['overflow_rate_mean'])} | {_fmt(row['histogram_input_saturation_rate_mean'])} | "
            f"{_fmt(row['n_commits_applied_mean'])} | {_fmt(row['slow_update_violation_rate_mean'])} | "
            f"{_fmt(row['fast_cycle_violation_rate_mean'])} | {row['dominant_overflow_source']} | "
            f"{row.get('artifact_path') or ''} |"
        )

    lines.extend(
        [
            "",
            "## Delta Summary",
            "",
            "| Scenario | Mode | dLER vs Static | dLER vs CNN | dOverflow vs Static | dOverflow vs CNN |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in delta_rows:
        def _fmt(value: Any) -> str:
            if value is None:
                return ""
            return f"{float(value):.6f}"

        lines.append(
            f"| {row['scenario']} | {row['mode_label']} | {_fmt(row['delta_ler_vs_static_linear'])} | "
            f"{_fmt(row['delta_ler_vs_cnn_fpga'])} | {_fmt(row['delta_overflow_vs_static_linear'])} | "
            f"{_fmt(row['delta_overflow_vs_cnn_fpga'])} |"
        )

    lines.extend(
        [
            "",
            "## Scenario Winners",
            "",
            "| Scenario | Best Mode | Best LER Mean | Runner-Up Gap |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for item in scenario_winners:
        lines.append(
            f"| {item['scenario']} | {item['best_mode_label']} | {float(item['best_ler_mean']):.6f} | "
            f"{float(item['runner_up_gap']):.6f} |"
        )

    write_text(path, "\n".join(lines) + "\n", encoding="utf-8")


def _short_slug(value: Any, default: str, *, limit: int = 8) -> str:
    text = "".join(ch for ch in str(value or default).lower() if ch.isalnum())
    if not text:
        text = default
    return text[:limit]


def main() -> int:
    args = _arg_parser().parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_config_with_bases(cfg_path)
    merged_base = {k: v for k, v in cfg.items() if k != "p4_benchmark"}
    benchmark_cfg = dict(cfg.get("p4_benchmark", {}))
    modes = _normalize_entries(benchmark_cfg.get("modes", []), args.mode, "modes")
    scenarios = _normalize_entries(benchmark_cfg.get("scenarios", []), args.scenario, "scenarios")
    protocol = dict(benchmark_cfg.get("protocol", {}))
    repeats = int(args.repeats if args.repeats is not None else protocol.get("repeats", 1))
    seed_base = int(merged_base.get("experiment", {}).get("seed", 1234))
    scenario_seed_stride = int(protocol.get("scenario_seed_stride", 1000))
    mode_seed_stride = int(protocol.get("mode_seed_stride", 100000))
    seed_pairing = str(protocol.get("seed_pairing", "independent")).lower()
    paired_seeds = bool(args.paired_seeds or seed_pairing in {"paired", "same", "same_across_modes"})

    cfg_hash = config_hash(cfg)
    out_root = ensure_dir(merged_base.get("paths", {}).get("output_root", "runs"))
    exp_slug = _short_slug(merged_base.get("experiment", {}).get("name", "p4_benchmark"), "p4b")
    run_dir = ensure_dir(
        out_root
        / "p4_benchmark"
        / f"{exp_slug}_{now_tag()}_{cfg_hash[:6]}_{os.getpid()}"
    )

    raw_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    scenario_winners: List[Dict[str, Any]] = []

    for scenario_idx, scenario_entry in enumerate(scenarios):
        scenario_name = str(scenario_entry.get("name"))
        scenario_label = str(scenario_entry.get("label", scenario_name))
        for mode_idx, mode_entry in enumerate(modes):
            mode_name = str(mode_entry.get("name"))
            mode_label = str(mode_entry.get("label", mode_name))
            scenario_dir = _short_slug(scenario_name, f"s{scenario_idx}", limit=6)
            mode_dir = _short_slug(mode_name, f"m{mode_idx}", limit=6)
            mode_run_dir = ensure_dir(run_dir / scenario_dir / mode_dir)
            per_repeat: List[Dict[str, Any]] = []
            for repeat_idx in range(repeats):
                mode_offset = 0 if paired_seeds else mode_idx * mode_seed_stride
                seed = seed_base + scenario_idx * scenario_seed_stride + mode_offset + repeat_idx
                run_cfg = _build_run_config(merged_base, scenario_entry, mode_entry, seed)
                run_cfg.setdefault("experiment", {})
                run_cfg["experiment"]["name"] = f"{merged_base.get('experiment', {}).get('name', 'p4_benchmark')}_{scenario_name}_{mode_name}_rep{repeat_idx:02d}"
                repeat_run_dir = ensure_dir(mode_run_dir / f"repeat_{repeat_idx:02d}")
                repeat_summary_path = repeat_run_dir / "hil_summary.json"
                if repeat_summary_path.exists():
                    summary = json.loads(repeat_summary_path.read_text(encoding="utf-8"))
                    print(
                        f"[resume][{scenario_name}][{mode_name}][rep={repeat_idx:02d}] "
                        f"LER={float(summary['final_ler']):.6f}, overflow={float(summary['overflow_rate']):.6f}"
                    )
                else:
                    summary = run_hil_session(run_cfg, repeat_run_dir)
                row = {
                    "scenario": scenario_name,
                    "scenario_label": scenario_label,
                    "mode": mode_name,
                    "mode_label": mode_label,
                    "repeat": repeat_idx,
                    "seed": seed,
                    "artifact_path": summary.get("artifact_path"),
                    "final_ler": float(summary["final_ler"]),
                    "overflow_rate": float(summary["overflow_rate"]),
                    "histogram_input_saturation_rate": float(summary.get("histogram_input_saturation_rate", 0.0)),
                    "correction_saturation_rate": float(summary.get("correction_saturation_rate", 0.0)),
                    "aggressive_param_rate": float(summary.get("aggressive_param_rate", 0.0)),
                    "dominant_overflow_source": summary.get("dominant_overflow_source"),
                    "n_commits_applied": int(summary.get("n_commits_applied", 0)),
                    "slow_update_violation_rate": float(summary.get("slow_update_violation_rate", 0.0)),
                    "fast_cycle_violation_rate": float(summary.get("fast_cycle_violation_rate", 0.0)),
                    "teacher_contribution_l2_mean": float(summary.get("teacher_branch_diagnostics", {}).get("teacher_contribution_l2_mean", 0.0) or 0.0),
                    "teacher_scalar_abs_mean": float(summary.get("teacher_branch_diagnostics", {}).get("teacher_scalar_abs_mean", 0.0) or 0.0),
                    "teacher_gate_mean": float(summary.get("teacher_branch_diagnostics", {}).get("teacher_gate_mean", 0.0) or 0.0),
                    "teacher_gate_std": float(summary.get("teacher_branch_diagnostics", {}).get("teacher_gate_std", 0.0) or 0.0),
                    "teacher_per_scalar": dict(summary.get("teacher_branch_diagnostics", {}).get("per_scalar", {}) or {}),
                    "run_dir": str(repeat_run_dir),
                }
                per_repeat.append(row)
                raw_rows.append(row)
                print(
                    f"[{scenario_name}][{mode_name}][rep={repeat_idx:02d}] "
                    f"LER={row['final_ler']:.6f}, overflow={row['overflow_rate']:.6f}, "
                    f"hist_sat={row['histogram_input_saturation_rate']:.6f}"
                )

            aggregated = {
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "mode": mode_name,
                "mode_label": mode_label,
                "artifact_path": per_repeat[0].get("artifact_path"),
                "dominant_overflow_source": _dominant_source(per_repeat),
                "repeat_run_dirs": [item["run_dir"] for item in per_repeat],
            }
            for metric_key in (
                "final_ler",
                "overflow_rate",
                "histogram_input_saturation_rate",
                "correction_saturation_rate",
                "aggressive_param_rate",
                "n_commits_applied",
                "slow_update_violation_rate",
                "fast_cycle_violation_rate",
                "teacher_contribution_l2_mean",
                "teacher_scalar_abs_mean",
                "teacher_gate_mean",
                "teacher_gate_std",
            ):
                aggregated.update(_aggregate_metric(per_repeat, metric_key))
            aggregated["teacher_per_scalar"] = _aggregate_per_scalar(per_repeat)
            comparison_rows.append(aggregated)

    by_scenario_mode = {(row["scenario"], row["mode"]): row for row in comparison_rows}
    for scenario_entry in scenarios:
        scenario_name = str(scenario_entry.get("name"))
        scenario_rows = [row for row in comparison_rows if row["scenario"] == scenario_name]
        ordered = sorted(scenario_rows, key=lambda item: (float(item["final_ler_mean"]), float(item["overflow_rate_mean"])))
        best = ordered[0]
        runner_up_gap = 0.0 if len(ordered) < 2 else float(ordered[1]["final_ler_mean"] - best["final_ler_mean"])
        scenario_winners.append(
            {
                "scenario": scenario_name,
                "scenario_label": best["scenario_label"],
                "best_mode": best["mode"],
                "best_mode_label": best["mode_label"],
                "best_ler_mean": best["final_ler_mean"],
                "runner_up_gap": runner_up_gap,
            }
        )

    for row in comparison_rows:
        scenario = row["scenario"]
        static_row = by_scenario_mode.get((scenario, "static_linear"))
        cnn_row = by_scenario_mode.get((scenario, "cnn_fpga"))
        delta_rows.append(
            {
                "scenario": scenario,
                "scenario_label": row["scenario_label"],
                "mode": row["mode"],
                "mode_label": row["mode_label"],
                "delta_ler_vs_static_linear": None
                if static_row is None
                else float(row["final_ler_mean"] - static_row["final_ler_mean"]),
                "delta_ler_vs_cnn_fpga": None
                if cnn_row is None
                else float(row["final_ler_mean"] - cnn_row["final_ler_mean"]),
                "delta_overflow_vs_static_linear": None
                if static_row is None
                else float(row["overflow_rate_mean"] - static_row["overflow_rate_mean"]),
                "delta_overflow_vs_cnn_fpga": None
                if cnn_row is None
                else float(row["overflow_rate_mean"] - cnn_row["overflow_rate_mean"]),
            }
        )

    comparison_csv = run_dir / "comparison.csv"
    delta_csv = run_dir / "delta.csv"
    report_md = run_dir / "report.md"
    per_scalar_csv = run_dir / "teacher_scalar_diagnostics.csv"
    _write_csv(
        comparison_csv,
        comparison_rows,
        [
            "scenario",
            "scenario_label",
            "mode",
            "mode_label",
            "artifact_path",
            "final_ler_mean",
            "final_ler_std",
            "overflow_rate_mean",
            "overflow_rate_std",
            "histogram_input_saturation_rate_mean",
            "histogram_input_saturation_rate_std",
            "correction_saturation_rate_mean",
            "aggressive_param_rate_mean",
            "n_commits_applied_mean",
            "slow_update_violation_rate_mean",
            "fast_cycle_violation_rate_mean",
            "teacher_contribution_l2_mean_mean",
            "teacher_scalar_abs_mean_mean",
            "teacher_gate_mean_mean",
            "teacher_gate_std_mean",
            "dominant_overflow_source",
        ],
    )
    _write_per_scalar_csv(per_scalar_csv, comparison_rows)
    _write_csv(
        delta_csv,
        delta_rows,
        [
            "scenario",
            "scenario_label",
            "mode",
            "mode_label",
            "delta_ler_vs_static_linear",
            "delta_ler_vs_cnn_fpga",
            "delta_overflow_vs_static_linear",
            "delta_overflow_vs_cnn_fpga",
        ],
    )
    _write_report(
        report_md,
        protocol={
            "protocol_id": protocol.get("protocol_id", "p4_protocol"),
            "report_template_version": protocol.get("report_template_version", "p4_report_v1"),
            "repeats": repeats,
            "real_board_policy": protocol.get("real_board_policy", "conditional_extension"),
        },
        comparison_rows=comparison_rows,
        delta_rows=delta_rows,
        scenario_winners=scenario_winners,
    )

    save_json(
        run_dir / "summary.json",
        {
            "config_hash": cfg_hash,
            "git_commit": _git_commit(),
            "run_dir": str(run_dir),
            "comparison_csv": str(comparison_csv),
            "delta_csv": str(delta_csv),
            "teacher_scalar_diagnostics_csv": str(per_scalar_csv),
            "report_md": str(report_md),
            "protocol": {
                "protocol_id": protocol.get("protocol_id", "p4_protocol"),
                "report_template_version": protocol.get("report_template_version", "p4_report_v1"),
                "repeats": repeats,
                "scenario_seed_stride": scenario_seed_stride,
                "mode_seed_stride": mode_seed_stride,
                "paired_seeds": paired_seeds,
                "seed_pairing": "paired" if paired_seeds else "independent",
                "real_board_policy": protocol.get("real_board_policy", "conditional_extension"),
                "frozen_baseline_set": list(protocol.get("frozen_baseline_set", [])),
            },
            "comparison_rows": comparison_rows,
            "delta_rows": delta_rows,
            "scenario_winners": scenario_winners,
            "raw_rows": raw_rows,
        },
    )

    print("P4 multi-scenario benchmark complete.")
    print(f"run_dir={run_dir}")
    for item in scenario_winners:
        print(
            f"[winner][{item['scenario']}] mode={item['best_mode']} "
            f"LER={float(item['best_ler_mean']):.6f} gap={float(item['runner_up_gap']):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
