"""Compare HIL results across mock/float/int8/real-board modes."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from cnn_fpga.benchmark.run_hil_suite import run_hil_session
from cnn_fpga.hwio import BoardBackendUnavailableError
from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HIL mode benchmark.")
    parser.add_argument("--config", required=True, type=str, help="Path to hardware_hil YAML config")
    return parser


def _mode_entries(config: Dict) -> List[Dict]:
    modes = config.get("benchmark_modes", [])
    if not modes:
        raise ValueError("No benchmark_modes configured for HIL mode benchmark.")
    return [dict(item) for item in modes]


def _build_mode_config(base_config: Dict, mode_entry: Dict) -> Dict:
    cfg = deepcopy(base_config)
    hil_cfg = cfg.setdefault("hil", {})
    slow_cfg = cfg.setdefault("slow_loop", {})
    artifact_cfg = slow_cfg.setdefault("model_artifact", {})
    infer_cfg = slow_cfg.setdefault("inference_service", {})

    if "backend" in mode_entry:
        hil_cfg["backend"] = str(mode_entry["backend"])
    if "slow_loop_mode" in mode_entry:
        slow_cfg["mode"] = str(mode_entry["slow_loop_mode"])
    if "artifact_selector" in mode_entry:
        artifact_cfg["artifact_selector"] = str(mode_entry["artifact_selector"])
    if "artifact_path" in mode_entry:
        artifact_cfg["path"] = mode_entry["artifact_path"]
    if "inference_service_mode" in mode_entry:
        infer_cfg["mode"] = str(mode_entry["inference_service_mode"])
    if "inference_backend" in mode_entry:
        infer_cfg["backend"] = str(mode_entry["inference_backend"])
    if "python_executable" in mode_entry:
        infer_cfg["python_executable"] = mode_entry["python_executable"]
    return cfg


def _write_csv(path: Path, rows: List[Dict]) -> None:
    fields = [
        "mode",
        "backend",
        "status",
        "artifact_path",
        "inference_service_mode",
        "final_ler",
        "overflow_rate",
        "histogram_input_saturation_rate",
        "correction_saturation_rate",
        "aggressive_param_rate",
        "dominant_overflow_source",
        "n_slow_updates_finished",
        "n_commits_applied",
        "slow_update_violation_rate",
        "fast_cycle_violation_rate",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_report(path: Path, rows: List[Dict]) -> None:
    lines = [
        "# P3 HIL Mode Benchmark",
        "",
        "| Mode | Backend | Status | LER | Overflow | Hist Sat | Corr Sat | Agg Param | Dominant Source | Commits | Slow Viol | Fast Viol | Artifact | Error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        def _fmt(value):
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        lines.append(
            f"| {row['mode']} | {row['backend']} | {row['status']} | {_fmt(row.get('final_ler'))} | "
            f"{_fmt(row.get('overflow_rate'))} | {_fmt(row.get('histogram_input_saturation_rate'))} | "
            f"{_fmt(row.get('correction_saturation_rate'))} | {_fmt(row.get('aggressive_param_rate'))} | "
            f"{row.get('dominant_overflow_source') or ''} | {_fmt(row.get('n_commits_applied'))} | "
            f"{_fmt(row.get('slow_update_violation_rate'))} | {_fmt(row.get('fast_cycle_violation_rate'))} | "
            f"{row.get('artifact_path') or ''} | {row.get('error') or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _arg_parser().parse_args()
    base_config = load_yaml_config(args.config)
    experiment = base_config.get("experiment", {})
    cfg_hash = config_hash(base_config)
    out_root = ensure_dir(get_path(base_config, "output_root", "runs"))
    run_dir = ensure_dir(out_root / "hil_mode_benchmark" / f"{experiment.get('name', 'hardware_hil')}_{now_tag()}_{cfg_hash}")

    rows: List[Dict] = []
    for mode_entry in _mode_entries(base_config):
        mode_name = str(mode_entry["name"])
        mode_cfg = _build_mode_config(base_config, mode_entry)
        mode_run_dir = ensure_dir(run_dir / mode_name)
        try:
            summary = run_hil_session(mode_cfg, mode_run_dir)
            row = {
                "mode": mode_name,
                "backend": summary.get("backend"),
                "status": "ok",
                "artifact_path": summary.get("artifact_path"),
                "inference_service_mode": summary.get("inference_service_mode"),
                "final_ler": summary.get("final_ler"),
                "overflow_rate": summary.get("overflow_rate"),
                "histogram_input_saturation_rate": summary.get("histogram_input_saturation_rate"),
                "correction_saturation_rate": summary.get("correction_saturation_rate"),
                "aggressive_param_rate": summary.get("aggressive_param_rate"),
                "dominant_overflow_source": summary.get("dominant_overflow_source"),
                "n_slow_updates_finished": summary.get("n_slow_updates_finished"),
                "n_commits_applied": summary.get("n_commits_applied"),
                "slow_update_violation_rate": summary.get("slow_update_violation_rate"),
                "fast_cycle_violation_rate": summary.get("fast_cycle_violation_rate"),
                "error": None,
            }
        except BoardBackendUnavailableError as exc:
            row = {
                "mode": mode_name,
                "backend": str(mode_cfg.get("hil", {}).get("backend", "board")),
                "status": "skipped_unavailable",
                "artifact_path": None,
                "inference_service_mode": str(mode_cfg.get("slow_loop", {}).get("inference_service", {}).get("mode", "inproc")),
                "final_ler": None,
                "overflow_rate": None,
                "histogram_input_saturation_rate": None,
                "correction_saturation_rate": None,
                "aggressive_param_rate": None,
                "dominant_overflow_source": None,
                "n_slow_updates_finished": None,
                "n_commits_applied": None,
                "slow_update_violation_rate": None,
                "fast_cycle_violation_rate": None,
                "error": exc.reason,
            }
            save_json(mode_run_dir / "hil_summary.json", row)
        rows.append(row)
        print(f"[{mode_name}] backend={row['backend']} status={row['status']} commits={row.get('n_commits_applied')}")

    comparison_csv = run_dir / "mode_comparison.csv"
    report_md = run_dir / "report.md"
    _write_csv(comparison_csv, rows)
    _write_report(report_md, rows)
    save_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "comparison_csv": str(comparison_csv),
            "report_md": str(report_md),
            "rows": rows,
        },
    )

    print("HIL mode benchmark complete.")
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
