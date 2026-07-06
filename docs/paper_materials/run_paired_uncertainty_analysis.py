from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER_MATERIALS = ROOT / "docs" / "paper_materials"
FIG_DIR = ROOT / "docs" / "figure_assets" / "submission_draft_python_figures"
PAIRED_DELTAS_CSV = FIG_DIR / "source_data_fig02_paired_deltas.csv"
CSV_PATH = PAPER_MATERIALS / "submission_draft_paired_uncertainty_analysis.csv"
JSON_PATH = PAPER_MATERIALS / "submission_draft_paired_uncertainty_analysis.json"
REPORT_PATH = PAPER_MATERIALS / "投稿稿paired_uncertainty分析记录.md"

SEED = 20260703 + 73
N_BOOTSTRAP = 20_000


def _read_paired_rows() -> list[dict[str, str]]:
    with PAIRED_DELTAS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _bootstrap_means(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    return np.mean(values[indices], axis=1)


def _summarize_group(label: str, rows: list[dict[str, str]], rng: np.random.Generator) -> dict[str, float | str | int]:
    deltas = np.array([float(row["delta_final_ler_ukf_minus_hybrid"]) for row in rows], dtype=float)
    relative = np.array([float(row["relative_reduction_percent"]) for row in rows], dtype=float)
    bootstrap_means = _bootstrap_means(deltas, rng)
    low, high = _percentile_interval(bootstrap_means)
    return {
        "scenario": label,
        "n_paired_repeats": len(rows),
        "mean_delta_ukf_minus_hybrid": float(np.mean(deltas)),
        "min_delta_ukf_minus_hybrid": float(np.min(deltas)),
        "max_delta_ukf_minus_hybrid": float(np.max(deltas)),
        "paired_bootstrap_span_low": low,
        "paired_bootstrap_span_high": high,
        "directionally_positive_count": int(np.sum(deltas > 0.0)),
        "all_paired_deltas_positive": bool(np.all(deltas > 0.0)),
        "mean_relative_reduction_percent": float(np.mean(relative)),
        "min_relative_reduction_percent": float(np.min(relative)),
        "max_relative_reduction_percent": float(np.max(relative)),
        "boundary": (
            "Descriptive paired repeat summary only. The bootstrap span is "
            "reported for transparency with n=2 per scenario and must not be "
            "interpreted as an inferential confidence interval, p-value, "
            "standard error or distribution-level robustness test."
        ),
    }


def _rows() -> list[dict[str, float | str | int | bool]]:
    source_rows = _read_paired_rows()
    rng = np.random.default_rng(SEED)
    scenarios = []
    for row in source_rows:
        scenario = row["scenario"]
        if scenario not in scenarios:
            scenarios.append(scenario)

    rows: list[dict[str, float | str | int | bool]] = []
    for scenario in scenarios:
        selected = [row for row in source_rows if row["scenario"] == scenario]
        rows.append(_summarize_group(scenario, selected, rng))
    rows.append(_summarize_group("all_scenarios", source_rows, rng))
    return rows


def _write_csv(rows: list[dict[str, float | str | int | bool]]) -> None:
    fields = [
        "scenario",
        "n_paired_repeats",
        "mean_delta_ukf_minus_hybrid",
        "min_delta_ukf_minus_hybrid",
        "max_delta_ukf_minus_hybrid",
        "paired_bootstrap_span_low",
        "paired_bootstrap_span_high",
        "directionally_positive_count",
        "all_paired_deltas_positive",
        "mean_relative_reduction_percent",
        "min_relative_reduction_percent",
        "max_relative_reduction_percent",
        "boundary",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(rows: list[dict[str, float | str | int | bool]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_paired_uncertainty_descriptive_v1",
        "seed": SEED,
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "source_csv": str(PAIRED_DELTAS_CSV),
        "boundary": (
            "Derived from existing paired repeat rows only. Because each scenario "
            "has n=2 paired repeats, the bootstrap span is a descriptive repeat-level "
            "uncertainty marker, not an inferential confidence interval, p-value, "
            "standard error or statistical significance claim."
        ),
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, float | str | int | bool]]) -> None:
    lines = [
        "# 投稿稿 paired uncertainty 分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录从 `source_data_fig02_paired_deltas.csv` 派生 paired repeat uncertainty 摘要。它不重跑 benchmark，不新增实验，不提供显著性检验。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 协议",
        "",
        f"- random seed: `{SEED}`",
        f"- bootstrap resamples: `{N_BOOTSTRAP}`",
        "- source rows: UKF 与 hybrid residual branch 的 paired final_ler deltas",
        "- bootstrap span: paired deltas 的 repeat-level mean bootstrap percentile span",
        "",
        "## 结果摘要",
        "",
        "| Scenario | n | Mean delta | Bootstrap span | Direction | Mean rel. reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        direction = f"{int(row['directionally_positive_count'])}/{int(row['n_paired_repeats'])}"
        lines.append(
            f"| {row['scenario']} | {int(row['n_paired_repeats'])} | "
            f"{float(row['mean_delta_ukf_minus_hybrid']):.6f} | "
            f"[{float(row['paired_bootstrap_span_low']):.6f}, {float(row['paired_bootstrap_span_high']):.6f}] | "
            f"{direction} | {float(row['mean_relative_reduction_percent']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：所有现有 paired repeats 的 UKF-minus-hybrid delta 为正，方向一致。",
            "- 可以写：bootstrap span 是 repeat-level descriptive uncertainty marker，用于透明展示 n=2 的不确定性。",
            "- 不能写：该 span 是 inferential confidence interval、standard error、p-value、significance test 或 distribution-level robustness proof。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _rows()
    _write_csv(rows)
    _write_json(rows)
    _write_report(rows)
    print(json.dumps({"status": "generated", "csv": str(CSV_PATH), "json": str(JSON_PATH), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
