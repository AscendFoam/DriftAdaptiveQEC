"""Compare P2 hardware-emulation results across slow-loop runtime modes."""

# 中文说明：
# - 该脚本复用 P2 行为仿真骨架，在相同随机种子和漂移场景下比较公平基线与模型模式。
# - 输出包含逐模式原始结果、逐场景对比表，以及相对固定基线 / oracle 基线 / 浮点模型的差值。

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cnn_fpga.benchmark.run_hardware_emulation import _aggregate_summaries, _run_repeat
from cnn_fpga.runtime.inference_service import resolve_model_path
from cnn_fpga.runtime.slow_loop_runtime import SlowLoopRuntimeConfig
from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P2 runtime-mode comparison benchmark.")
    parser.add_argument("--config", required=True, type=str, help="Path to hardware_emulation YAML config")
    return parser


def _mode_entries(config: Dict) -> List[Dict]:
    modes = config.get("benchmark_modes", [])
    if not modes:
        raise ValueError("No benchmark_modes configured.")
    return [dict(item) for item in modes]


def _build_mode_config(base_config: Dict, mode_entry: Dict) -> Tuple[Dict, str | None]:
    cfg = deepcopy(base_config)
    slow_cfg = cfg.setdefault("slow_loop", {})
    artifact_cfg = slow_cfg.setdefault("model_artifact", {})

    slow_mode = str(mode_entry.get("slow_loop_mode", "mock")).lower()
    slow_cfg["mode"] = slow_mode

    if slow_mode == "model_artifact":
        if "artifact_selector" in mode_entry:
            artifact_cfg["artifact_selector"] = str(mode_entry["artifact_selector"])
        if "artifact_path" in mode_entry and mode_entry["artifact_path"]:
            artifact_cfg["path"] = str(mode_entry["artifact_path"])
        else:
            artifact_cfg["path"] = None

        runtime_cfg = SlowLoopRuntimeConfig.from_config(cfg)
        resolved = resolve_model_path(
            cfg,
            model_path=runtime_cfg.model_artifact_path,
            use_latest_model_dir_artifact=runtime_cfg.use_latest_model_dir_artifact,
            artifact_selector=runtime_cfg.artifact_selector,
        )
        return cfg, str(resolved)

    artifact_cfg["path"] = None
    return cfg, None


def _write_comparison_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "mode",
        "scenario",
        "final_ler_mean",
        "final_ler_std",
        "commit_count_mean",
        "overflow_rate_mean",
        "fast_cycle_violation_rate_mean",
        "slow_update_violation_rate_mean",
        "artifact_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_delta_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "scenario",
        "mode",
        "delta_ler_vs_fixed_baseline",
        "delta_ler_vs_oracle_delayed",
        "delta_ler_vs_model_artifact",
        "delta_commit_vs_fixed_baseline",
        "delta_commit_vs_model_artifact",
        "delta_overflow_vs_fixed_baseline",
        "delta_overflow_vs_model_artifact",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_report_md(path: Path, comparison_rows: List[Dict], delta_rows: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# P2 Runtime Mode Benchmark")
    lines.append("")
    lines.append("## Mode Comparison")
    lines.append("")
    lines.append("| Mode | Scenario | LER Mean | LER Std | Commit Mean | Overflow Mean | Artifact |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for row in comparison_rows:
        artifact_label = row.get("artifact_path") or row["mode"]
        lines.append(
            "| "
            f"{row['mode']} | {row['scenario']} | {row['final_ler_mean']:.6f} | {row['final_ler_std']:.6f} | "
            f"{row['commit_count_mean']:.2f} | {row['overflow_rate_mean']:.6f} | {artifact_label} |"
        )
    lines.append("")
    lines.append("## Delta Summary")
    lines.append("")
    lines.append("| Scenario | Mode | dLER vs Fixed | dLER vs Oracle | dLER vs Float | dCommit vs Fixed | dOverflow vs Fixed |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in delta_rows:
        def _fmt(value):
            return "" if value is None else f"{value:.6f}"

        lines.append(
            "| "
            f"{row['scenario']} | {row['mode']} | {_fmt(row['delta_ler_vs_fixed_baseline'])} | "
            f"{_fmt(row['delta_ler_vs_oracle_delayed'])} | "
            f"{_fmt(row['delta_ler_vs_model_artifact'])} | {_fmt(row['delta_commit_vs_fixed_baseline'])} | "
            f"{_fmt(row['delta_overflow_vs_fixed_baseline'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _arg_parser().parse_args()
    base_config = load_yaml_config(args.config)
    experiment = base_config.get("experiment", {})
    seed_base = int(experiment.get("seed", 1234))
    repeats = int(base_config.get("simulation", {}).get("repeats", 1))
    scenarios = base_config.get("drift", {}).get("scenarios", [])
    if not scenarios:
        raise ValueError("No drift scenarios configured.")

    mode_entries = _mode_entries(base_config)
    cfg_hash = config_hash(base_config)
    out_root = ensure_dir(get_path(base_config, "output_root", "runs"))
    run_dir = ensure_dir(
        out_root / "p2_mode_benchmark" / f"{experiment.get('name', 'hardware_emulation')}_{now_tag()}_{cfg_hash}"
    )

    comparison_rows: List[Dict] = []
    delta_rows: List[Dict] = []
    per_mode_reports: Dict[str, Dict] = {}

    for mode_entry in mode_entries:
        mode_name = str(mode_entry["name"])
        mode_config, artifact_path = _build_mode_config(base_config, mode_entry)
        mode_run_dir = ensure_dir(run_dir / mode_name)
        print(f"[mode={mode_name}] artifact={artifact_path or mode_name}")

        all_summaries: List[Dict] = []
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_name = str(scenario.get("name", scenario.get("type", "scenario")))
            for repeat_idx in range(repeats):
                seed = seed_base + scenario_idx * 1000 + repeat_idx
                summary = _run_repeat(mode_config, scenario, repeat_idx, mode_run_dir, seed)
                all_summaries.append(summary)
                print(
                    f"[mode={mode_name}][{scenario_name}][rep={repeat_idx:02d}] "
                    f"LER={summary['final_ler']:.6f}, commits={summary['n_commits_applied']}, "
                    f"overflow={summary['overflow_rate']:.6f}"
                )

        aggregated = _aggregate_summaries(mode_config, mode_run_dir, all_summaries)
        aggregated["mode"] = mode_name
        aggregated["artifact_path"] = artifact_path
        save_json(mode_run_dir / "mode_summary.json", aggregated)
        per_mode_reports[mode_name] = aggregated

        raw_by_scenario = {item["scenario"]: item for item in aggregated["scenarios"]}
        raw_repeat_summaries = {}
        for item in all_summaries:
            raw_repeat_summaries.setdefault(item["scenario"], []).append(item)

        for scenario_name, scenario_report in raw_by_scenario.items():
            scenario_items = raw_repeat_summaries[scenario_name]
            comparison_rows.append(
                {
                    "mode": mode_name,
                    "scenario": scenario_name,
                    "final_ler_mean": float(scenario_report["final_ler_mean"]),
                    "final_ler_std": float(scenario_report["final_ler_std"]),
                    "commit_count_mean": float(np.mean([x["n_commits_applied"] for x in scenario_items])),
                    "overflow_rate_mean": float(np.mean([x["overflow_rate"] for x in scenario_items])),
                    "fast_cycle_violation_rate_mean": float(
                        np.mean([x["fast_cycle_violation_rate"] for x in scenario_items])
                    ),
                    "slow_update_violation_rate_mean": float(
                        np.mean([x["slow_update_violation_rate"] for x in scenario_items])
                    ),
                    "artifact_path": artifact_path,
                }
            )

    by_scenario_mode = {(row["scenario"], row["mode"]): row for row in comparison_rows}
    for row in comparison_rows:
        scenario = row["scenario"]
        fixed_row = by_scenario_mode.get((scenario, "fixed_baseline"))
        oracle_row = by_scenario_mode.get((scenario, "oracle_delayed"))
        float_row = by_scenario_mode.get((scenario, "model_artifact"))
        delta_rows.append(
            {
                "scenario": scenario,
                "mode": row["mode"],
                "delta_ler_vs_fixed_baseline": None
                if fixed_row is None
                else float(row["final_ler_mean"] - fixed_row["final_ler_mean"]),
                "delta_ler_vs_oracle_delayed": None
                if oracle_row is None
                else float(row["final_ler_mean"] - oracle_row["final_ler_mean"]),
                "delta_ler_vs_model_artifact": None
                if float_row is None
                else float(row["final_ler_mean"] - float_row["final_ler_mean"]),
                "delta_commit_vs_fixed_baseline": None
                if fixed_row is None
                else float(row["commit_count_mean"] - fixed_row["commit_count_mean"]),
                "delta_commit_vs_model_artifact": None
                if float_row is None
                else float(row["commit_count_mean"] - float_row["commit_count_mean"]),
                "delta_overflow_vs_fixed_baseline": None
                if fixed_row is None
                else float(row["overflow_rate_mean"] - fixed_row["overflow_rate_mean"]),
                "delta_overflow_vs_model_artifact": None
                if float_row is None
                else float(row["overflow_rate_mean"] - float_row["overflow_rate_mean"]),
            }
        )

    comparison_csv = run_dir / "mode_comparison.csv"
    delta_csv = run_dir / "mode_delta.csv"
    report_md = run_dir / "report.md"
    _write_comparison_csv(comparison_csv, comparison_rows)
    _write_delta_csv(delta_csv, delta_rows)
    _write_report_md(report_md, comparison_rows, delta_rows)

    report = {
        "config_hash": cfg_hash,
        "run_dir": str(run_dir),
        "comparison_csv": str(comparison_csv),
        "delta_csv": str(delta_csv),
        "report_md": str(report_md),
        "modes": per_mode_reports,
        "comparison_rows": comparison_rows,
        "delta_rows": delta_rows,
    }
    save_json(run_dir / "summary.json", report)

    print("P2 mode benchmark complete.")
    print(f"run_dir={run_dir}")
    print(f"comparison_csv={comparison_csv}")
    print(f"delta_csv={delta_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
