"""Run a focused P3 parameter sweep to trade off LER and overflow rate."""

# 中文说明：
# - 该脚本面向 P3 HIL mock/backend 的参数映射调优，重点扫描 `gain_clip / beta_smoothing / alpha_bias / gain_scale`。
# - 策略采用“两阶段”：
#   1. 先用较短时长做 sweep，快速筛掉明显不合适的参数；
#   2. 再对满足 LER 约束的 top-k 候选做长配置确认，且强制带上 baseline，避免长确认缺少同口径对照。

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from cnn_fpga.benchmark.run_hil_suite import run_hil_session
from cnn_fpga.utils.config import config_hash, ensure_dir, load_yaml_config, now_tag, save_json


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run focused P3 parameter sweep for overflow/LER tradeoff.")
    parser.add_argument("--config", required=True, type=str, help="Path to sweep YAML config")
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
        raise ValueError("Sweep config must provide `base_config`.")
    resolved = (sweep_cfg_path.parent / str(base_path)).resolve()
    return load_yaml_config(resolved)


def _candidate_name(candidate: Dict[str, Any], idx: int) -> str:
    name = candidate.get("name")
    if name:
        return str(name)
    gain_clip = candidate.get("gain_clip", [0.2, 1.2])
    beta = candidate.get("beta_smoothing", 0.2)
    alpha = candidate.get("alpha_bias", 1.0)
    return (
        f"cand_{idx:02d}_g{float(gain_clip[0]):.2f}_{float(gain_clip[1]):.2f}"
        f"_b{float(beta):.2f}_a{float(alpha):.2f}"
    )


def _build_candidate_config(
    base_config: Dict[str, Any],
    candidate: Dict[str, Any],
    timing_override: Dict[str, Any],
    experiment_name: str,
) -> Dict[str, Any]:
    config = deepcopy(base_config)
    config.setdefault("experiment", {})
    config["experiment"]["name"] = experiment_name
    config.setdefault("param_mapping", {})
    if "gain_clip" in candidate:
        config["param_mapping"]["gain_clip"] = [float(candidate["gain_clip"][0]), float(candidate["gain_clip"][1])]
    if "beta_smoothing" in candidate:
        config["param_mapping"]["beta_smoothing"] = float(candidate["beta_smoothing"])
    if "alpha_bias" in candidate:
        config["param_mapping"]["alpha_bias"] = float(candidate["alpha_bias"])
    if "gain_scale" in candidate:
        config["param_mapping"]["gain_scale"] = float(candidate["gain_scale"])
    if "param_mapping" in candidate:
        for key, value in dict(candidate["param_mapping"]).items():
            config["param_mapping"][str(key)] = deepcopy(value)

    config.setdefault("slow_loop", {})
    config["slow_loop"]["mode"] = str(candidate.get("slow_loop_mode", "model_artifact"))
    model_artifact = config["slow_loop"].setdefault("model_artifact", {})
    if "artifact_selector" in candidate:
        model_artifact["artifact_selector"] = str(candidate["artifact_selector"])
    infer_cfg = config["slow_loop"].setdefault("inference_service", {})
    infer_cfg["mode"] = str(candidate.get("inference_service_mode", infer_cfg.get("mode", "subprocess")))
    infer_cfg["backend"] = str(candidate.get("inference_backend", infer_cfg.get("backend", "artifact_npz")))

    config.setdefault("hil", {})
    config["hil"]["backend"] = str(candidate.get("backend", config["hil"].get("backend", "mock")))

    config.setdefault("timing", {})
    for key, value in timing_override.items():
        config["timing"][key] = value
    return config


def _extract_row(name: str, candidate: Dict[str, Any], phase: str, summary: Dict[str, Any], baseline_ler: float | None) -> Dict[str, Any]:
    ler = float(summary["final_ler"])
    overflow = float(summary["overflow_rate"])
    gain_clip_raw = candidate.get("gain_clip", candidate.get("param_mapping", {}).get("gain_clip", [0.0, 0.0]))
    return {
        "name": name,
        "phase": phase,
        "gain_clip": [float(gain_clip_raw[0]), float(gain_clip_raw[1])],
        "gain_scale": float(candidate.get("gain_scale", candidate.get("param_mapping", {}).get("gain_scale", 1.0))),
        "beta_smoothing": float(candidate.get("beta_smoothing", candidate.get("param_mapping", {}).get("beta_smoothing", 0.0))),
        "alpha_bias": float(candidate.get("alpha_bias", candidate.get("param_mapping", {}).get("alpha_bias", 0.0))),
        "final_ler": ler,
        "overflow_rate": overflow,
        "n_commits_applied": int(summary.get("n_commits_applied", 0)),
        "slow_update_violation_rate": float(summary.get("slow_update_violation_rate", 0.0)),
        "fast_cycle_violation_rate": float(summary.get("fast_cycle_violation_rate", 0.0)),
        "artifact_path": summary.get("artifact_path"),
        "run_dir": str(summary.get("run_dir", "")),
        "delta_ler_vs_short_baseline": None if baseline_ler is None else ler - baseline_ler,
        "delta_ler_vs_confirm_baseline": None,
        "delta_overflow_vs_confirm_baseline": None,
    }


def _write_markdown(path: Path, rows: List[Dict[str, Any]], selected_names: List[str], confirm_names: List[str]) -> None:
    lines = [
        "# P3 Overflow Tuning Sweep",
        "",
        "| Candidate | Phase | gain_clip | gain_scale | beta | alpha | LER | Overflow | dLER vs short baseline | dLER vs confirm baseline | dOverflow vs confirm baseline | Commits |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        gain_clip = f"[{row['gain_clip'][0]:.2f}, {row['gain_clip'][1]:.2f}]"
        delta_ler = "" if row["delta_ler_vs_short_baseline"] is None else f"{row['delta_ler_vs_short_baseline']:+.6f}"
        delta_confirm_ler = "" if row["delta_ler_vs_confirm_baseline"] is None else f"{row['delta_ler_vs_confirm_baseline']:+.6f}"
        delta_confirm_overflow = "" if row["delta_overflow_vs_confirm_baseline"] is None else f"{row['delta_overflow_vs_confirm_baseline']:+.6f}"
        lines.append(
            f"| {row['name']} | {row['phase']} | {gain_clip} | {row['gain_scale']:.2f} | {row['beta_smoothing']:.2f} | {row['alpha_bias']:.2f} | "
            f"{row['final_ler']:.6f} | {row['overflow_rate']:.6f} | {delta_ler} | {delta_confirm_ler} | {delta_confirm_overflow} | {row['n_commits_applied']} |"
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
    run_dir = ensure_dir(out_root / "p3_param_sweep" / f"{experiment_cfg.get('name', 'p3_param_sweep')}_{now_tag()}_{cfg_hash}")

    rows: List[Dict[str, Any]] = []
    short_rows: List[Dict[str, Any]] = []
    baseline_candidate_name = str(tuning_cfg.get("baseline_candidate", "baseline_current"))
    baseline_short_ler: float | None = None

    for idx, candidate in enumerate(candidates):
        name = _candidate_name(candidate, idx)
        short_cfg = _build_candidate_config(base_config, candidate, short_timing, f"{experiment_cfg.get('name', 'p3_param_sweep')}_{name}_short")
        short_run_dir = ensure_dir(run_dir / "short" / name)
        summary = run_hil_session(short_cfg, short_run_dir)
        summary["run_dir"] = str(short_run_dir)
        row = _extract_row(name, candidate, "short", summary, baseline_ler=None)
        short_rows.append(row)
        rows.append(row)
        if name == baseline_candidate_name:
            baseline_short_ler = row["final_ler"]
        print(f"[short][{name}] LER={row['final_ler']:.6f} overflow={row['overflow_rate']:.6f}")

    if baseline_short_ler is None:
        raise ValueError(f"Baseline candidate `{baseline_candidate_name}` not found in tuning candidates.")

    for row in short_rows:
        row["delta_ler_vs_short_baseline"] = float(row["final_ler"] - baseline_short_ler)

    ler_tolerance_abs = float(tuning_cfg.get("selection", {}).get("ler_tolerance_abs", 0.02))
    confirm_top_k = int(tuning_cfg.get("selection", {}).get("confirm_top_k", 3))
    eligible = [
        row for row in short_rows if float(row["final_ler"]) <= float(baseline_short_ler + ler_tolerance_abs)
    ]
    eligible.sort(key=lambda item: (float(item["overflow_rate"]), float(item["final_ler"])))
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
            f"{experiment_cfg.get('name', 'p3_param_sweep')}_{name}_confirm",
        )
        confirm_run_dir = ensure_dir(run_dir / "confirm" / name)
        summary = run_hil_session(confirm_cfg, confirm_run_dir)
        summary["run_dir"] = str(confirm_run_dir)
        row = _extract_row(name, candidate, "confirm", summary, baseline_ler=baseline_short_ler)
        rows.append(row)
        print(f"[confirm][{name}] LER={row['final_ler']:.6f} overflow={row['overflow_rate']:.6f}")

    confirm_rows = [row for row in rows if row["phase"] == "confirm"]
    confirm_baseline_ler: float | None = None
    confirm_baseline_overflow: float | None = None
    for row in confirm_rows:
        if row["name"] == baseline_candidate_name:
            confirm_baseline_ler = float(row["final_ler"])
            confirm_baseline_overflow = float(row["overflow_rate"])
            break
    if confirm_baseline_ler is not None and confirm_baseline_overflow is not None:
        for row in confirm_rows:
            row["delta_ler_vs_confirm_baseline"] = float(row["final_ler"] - confirm_baseline_ler)
            row["delta_overflow_vs_confirm_baseline"] = float(row["overflow_rate"] - confirm_baseline_overflow)

    best_confirm = None
    if confirm_rows:
        non_baseline_rows = [row for row in confirm_rows if row["name"] != baseline_candidate_name]
        best_confirm_pool = non_baseline_rows or confirm_rows
        best_confirm = sorted(best_confirm_pool, key=lambda item: (float(item["overflow_rate"]), float(item["final_ler"])))[0]

    summary = {
        "run_dir": str(run_dir),
        "baseline_candidate": baseline_candidate_name,
        "baseline_short_ler": baseline_short_ler,
        "confirm_baseline_ler": confirm_baseline_ler,
        "confirm_baseline_overflow": confirm_baseline_overflow,
        "ler_tolerance_abs": ler_tolerance_abs,
        "selected_names": selected_names,
        "confirm_names": confirm_names,
        "best_confirm": best_confirm,
        "rows": rows,
    }
    save_json(run_dir / "summary.json", summary)
    _write_markdown(run_dir / "report.md", rows, selected_names, confirm_names)

    print("P3 parameter sweep complete.")
    print(f"run_dir={run_dir}")
    if best_confirm is not None:
        print(
            "best_confirm="
            f"{best_confirm['name']} "
            f"(LER={best_confirm['final_ler']:.6f}, overflow={best_confirm['overflow_rate']:.6f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
