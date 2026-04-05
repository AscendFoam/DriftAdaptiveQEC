"""Run focused P3 tuning when histogram-input saturation dominates overflow."""

# 中文说明：
# - 该脚本面向“overflow 主要由 histogram_input 主导”的场景。
# - 与 `run_p3_param_sweep.py` 不同，这里重点扫描：
#   1. `fast_loop.syndrome_limit`
#   2. `fast_loop.histogram_range_limit`
#   3. `model.sigma_measurement`
# - 目标是在不明显牺牲 LER 的前提下，优先压低 histogram_input saturation。

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from cnn_fpga.benchmark.run_hil_suite import run_hil_session
from cnn_fpga.utils.config import config_hash, ensure_dir, load_yaml_config, now_tag, save_json


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused P3 tuning for histogram-input saturation.")
    parser.add_argument("--config", required=True, type=str, help="Path to histogram-tuning YAML config")
    return parser


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_base_config(sweep_cfg_path: Path, sweep_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_path = sweep_cfg.get("base_config")
    if not base_path:
        # `load_yaml_config()` may already have resolved `base_config` recursively.
        # In that case the loaded sweep config itself can serve as the merged base.
        merged = deepcopy(sweep_cfg)
        merged.pop("tuning", None)
        return merged
    resolved = (sweep_cfg_path.parent / str(base_path)).resolve()
    return load_yaml_config(resolved)


def _candidate_name(candidate: Dict[str, Any], idx: int) -> str:
    return str(candidate.get("name", f"candidate_{idx:02d}"))


def _normalize_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    """Coerce lightweight YAML-fallback artifacts like '{}' into a real mapping."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in ("", "{}"):
            return {}
    raise ValueError(f"`{field_name}` must be a mapping, got {type(value).__name__}: {value!r}")


def _build_candidate_config(
    base_config: Dict[str, Any],
    candidate: Dict[str, Any],
    timing_override: Dict[str, Any],
    experiment_name: str,
) -> Dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("experiment", {})
    config["experiment"]["name"] = experiment_name
    config = _deep_merge(config, _normalize_mapping(candidate.get("overrides", {}), "candidate.overrides"))
    config.setdefault("timing", {})
    for key, value in timing_override.items():
        config["timing"][key] = value
    return config


def _extract_row(
    name: str,
    candidate: Dict[str, Any],
    phase: str,
    summary: Dict[str, Any],
    baseline_ler: float | None,
    baseline_hist: float | None,
) -> Dict[str, Any]:
    overrides = _normalize_mapping(candidate.get("overrides", {}), "candidate.overrides")
    fast_loop = dict(overrides.get("fast_loop", {}))
    model_cfg = dict(overrides.get("model", {}))
    return {
        "name": name,
        "phase": phase,
        "syndrome_limit": fast_loop.get("syndrome_limit"),
        "histogram_range_limit": fast_loop.get("histogram_range_limit"),
        "sigma_measurement": model_cfg.get("sigma_measurement"),
        "final_ler": float(summary["final_ler"]),
        "overflow_rate": float(summary["overflow_rate"]),
        "histogram_input_saturation_rate": float(summary.get("histogram_input_saturation_rate", 0.0)),
        "correction_saturation_rate": float(summary.get("correction_saturation_rate", 0.0)),
        "aggressive_param_rate": float(summary.get("aggressive_param_rate", 0.0)),
        "dominant_overflow_source": summary.get("dominant_overflow_source"),
        "n_commits_applied": int(summary.get("n_commits_applied", 0)),
        "run_dir": str(summary.get("run_dir", "")),
        "delta_ler_vs_short_baseline": None if baseline_ler is None else float(summary["final_ler"] - baseline_ler),
        "delta_hist_vs_short_baseline": None
        if baseline_hist is None
        else float(summary.get("histogram_input_saturation_rate", 0.0) - baseline_hist),
        "delta_ler_vs_confirm_baseline": None,
        "delta_hist_vs_confirm_baseline": None,
    }


def _write_markdown(path: Path, rows: List[Dict[str, Any]], selected_names: List[str], confirm_names: List[str]) -> None:
    lines = [
        "# P3 Histogram-Input Tuning Sweep",
        "",
        "| Candidate | Phase | syndrome_limit | histogram_range_limit | sigma_measurement | LER | Overflow | Hist Sat | Corr Sat | Agg Param | Dominant Source | dLER vs short baseline | dHist vs short baseline | dLER vs confirm baseline | dHist vs confirm baseline |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def _fmt(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)

        lines.append(
            f"| {row['name']} | {row['phase']} | {_fmt(row['syndrome_limit'])} | {_fmt(row['histogram_range_limit'])} | "
            f"{_fmt(row['sigma_measurement'])} | {_fmt(row['final_ler'])} | {_fmt(row['overflow_rate'])} | "
            f"{_fmt(row['histogram_input_saturation_rate'])} | {_fmt(row['correction_saturation_rate'])} | "
            f"{_fmt(row['aggressive_param_rate'])} | {row.get('dominant_overflow_source') or ''} | "
            f"{_fmt(row['delta_ler_vs_short_baseline'])} | {_fmt(row['delta_hist_vs_short_baseline'])} | "
            f"{_fmt(row['delta_ler_vs_confirm_baseline'])} | {_fmt(row['delta_hist_vs_confirm_baseline'])} |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Shortlist: {', '.join(selected_names) if selected_names else '(none)'}",
            f"- Long confirm: {', '.join(confirm_names) if confirm_names else '(none)'}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _arg_parser().parse_args()
    sweep_cfg_path = Path(args.config).expanduser().resolve()
    sweep_cfg = load_yaml_config(sweep_cfg_path)
    base_config = _load_base_config(sweep_cfg_path, sweep_cfg)

    experiment_cfg = sweep_cfg.get("experiment", {})
    tuning_cfg = sweep_cfg.get("tuning", {})
    candidates = tuning_cfg.get("candidates", [])
    if not candidates:
        raise ValueError("Sweep config must provide `tuning.candidates`.")

    short_timing = dict(tuning_cfg.get("short_timing", {}))
    confirm_timing = dict(tuning_cfg.get("confirm_timing", {}))
    if not short_timing or not confirm_timing:
        raise ValueError("Sweep config must provide both `short_timing` and `confirm_timing`.")

    out_root = ensure_dir(sweep_cfg.get("paths", {}).get("output_root", "runs"))
    cfg_hash = config_hash(sweep_cfg)
    run_dir = ensure_dir(out_root / "p3_histogram_tuning" / f"{experiment_cfg.get('name', 'p3_histogram_tuning')}_{now_tag()}_{cfg_hash}")

    rows: List[Dict[str, Any]] = []
    short_rows: List[Dict[str, Any]] = []
    baseline_candidate_name = str(tuning_cfg.get("baseline_candidate", "baseline_current"))
    baseline_short_ler: float | None = None
    baseline_short_hist: float | None = None

    for idx, candidate in enumerate(candidates):
        name = _candidate_name(candidate, idx)
        short_cfg = _build_candidate_config(base_config, candidate, short_timing, f"{experiment_cfg.get('name', 'p3_histogram_tuning')}_{name}_short")
        short_run_dir = ensure_dir(run_dir / "short" / name)
        summary = run_hil_session(short_cfg, short_run_dir)
        summary["run_dir"] = str(short_run_dir)
        row = _extract_row(name, candidate, "short", summary, baseline_ler=None, baseline_hist=None)
        short_rows.append(row)
        rows.append(row)
        if name == baseline_candidate_name:
            baseline_short_ler = row["final_ler"]
            baseline_short_hist = row["histogram_input_saturation_rate"]
        print(
            f"[short][{name}] "
            f"LER={row['final_ler']:.6f} "
            f"hist_sat={row['histogram_input_saturation_rate']:.6f} "
            f"source={row.get('dominant_overflow_source')}"
        )

    if baseline_short_ler is None or baseline_short_hist is None:
        raise ValueError(f"Baseline candidate `{baseline_candidate_name}` not found in tuning candidates.")

    for row in short_rows:
        row["delta_ler_vs_short_baseline"] = float(row["final_ler"] - baseline_short_ler)
        row["delta_hist_vs_short_baseline"] = float(row["histogram_input_saturation_rate"] - baseline_short_hist)

    ler_tolerance_abs = float(tuning_cfg.get("selection", {}).get("ler_tolerance_abs", 0.02))
    confirm_top_k = int(tuning_cfg.get("selection", {}).get("confirm_top_k", 3))
    eligible = [
        row for row in short_rows if float(row["final_ler"]) <= float(baseline_short_ler + ler_tolerance_abs)
    ]
    eligible.sort(
        key=lambda item: (
            float(item["histogram_input_saturation_rate"]),
            float(item["overflow_rate"]),
            float(item["final_ler"]),
        )
    )
    selected_names = [row["name"] for row in eligible]
    confirm_names = [baseline_candidate_name]
    for row in eligible:
        if row["name"] == baseline_candidate_name:
            continue
        if len(confirm_names) >= confirm_top_k:
            break
        confirm_names.append(row["name"])

    name_to_candidate = {_candidate_name(candidate, idx): candidate for idx, candidate in enumerate(candidates)}
    for name in confirm_names:
        candidate = name_to_candidate[name]
        confirm_cfg = _build_candidate_config(
            base_config,
            candidate,
            confirm_timing,
            f"{experiment_cfg.get('name', 'p3_histogram_tuning')}_{name}_confirm",
        )
        confirm_run_dir = ensure_dir(run_dir / "confirm" / name)
        summary = run_hil_session(confirm_cfg, confirm_run_dir)
        summary["run_dir"] = str(confirm_run_dir)
        row = _extract_row(name, candidate, "confirm", summary, baseline_short_ler, baseline_short_hist)
        rows.append(row)
        print(
            f"[confirm][{name}] "
            f"LER={row['final_ler']:.6f} "
            f"hist_sat={row['histogram_input_saturation_rate']:.6f} "
            f"source={row.get('dominant_overflow_source')}"
        )

    confirm_rows = [row for row in rows if row["phase"] == "confirm"]
    confirm_baseline_ler = None
    confirm_baseline_hist = None
    for row in confirm_rows:
        if row["name"] == baseline_candidate_name:
            confirm_baseline_ler = float(row["final_ler"])
            confirm_baseline_hist = float(row["histogram_input_saturation_rate"])
            break
    if confirm_baseline_ler is not None and confirm_baseline_hist is not None:
        for row in confirm_rows:
            row["delta_ler_vs_confirm_baseline"] = float(row["final_ler"] - confirm_baseline_ler)
            row["delta_hist_vs_confirm_baseline"] = float(
                row["histogram_input_saturation_rate"] - confirm_baseline_hist
            )

    non_baseline_confirm = [row for row in confirm_rows if row["name"] != baseline_candidate_name]
    best_confirm = None
    if non_baseline_confirm:
        best_confirm = sorted(
            non_baseline_confirm,
            key=lambda item: (
                float(item["histogram_input_saturation_rate"]),
                float(item["overflow_rate"]),
                float(item["final_ler"]),
            ),
        )[0]

    summary = {
        "run_dir": str(run_dir),
        "baseline_candidate": baseline_candidate_name,
        "baseline_short_ler": baseline_short_ler,
        "baseline_short_histogram_input_saturation_rate": baseline_short_hist,
        "confirm_baseline_ler": confirm_baseline_ler,
        "confirm_baseline_histogram_input_saturation_rate": confirm_baseline_hist,
        "ler_tolerance_abs": ler_tolerance_abs,
        "selected_names": selected_names,
        "confirm_names": confirm_names,
        "best_confirm": best_confirm,
        "rows": rows,
    }
    save_json(run_dir / "summary.json", summary)
    _write_markdown(run_dir / "report.md", rows, selected_names, confirm_names)

    print("P3 histogram-input tuning complete.")
    print(f"run_dir={run_dir}")
    if best_confirm is not None:
        print(
            "best_confirm="
            f"{best_confirm['name']} "
            f"(LER={best_confirm['final_ler']:.6f}, hist_sat={best_confirm['histogram_input_saturation_rate']:.6f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
