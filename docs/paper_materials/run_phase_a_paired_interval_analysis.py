from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER_MATERIALS = ROOT / "docs" / "paper_materials"
SUMMARY_CSV = PAPER_MATERIALS / "submission_draft_phase_a_repeat_summary.csv"
OUT_CSV = PAPER_MATERIALS / "submission_draft_phase_a_paired_interval_analysis.csv"
OUT_JSON = PAPER_MATERIALS / "submission_draft_phase_a_paired_interval_analysis.json"
REPORT_MD = PAPER_MATERIALS / "投稿稿phase_a_paired_interval分析记录.md"

SEED = 20260706 + 97
N_BOOTSTRAP = 50_000

# Two-sided 95% Student-t critical values for small-n paired mean intervals.
T_CRITICAL_95 = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    11: 2.200985,
    12: 2.178813,
    13: 2.160369,
    14: 2.144787,
    15: 2.131450,
    16: 2.119905,
    17: 2.109816,
    18: 2.100922,
    19: 2.093024,
    20: 2.085963,
    21: 2.079614,
    22: 2.073873,
    23: 2.068658,
    24: 2.063899,
    25: 2.059539,
    26: 2.055529,
    27: 2.051831,
    28: 2.048407,
    29: 2.045230,
    30: 2.042272,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _scenario_order(rows: Iterable[dict[str, str]]) -> list[str]:
    scenarios: list[str] = []
    for row in rows:
        scenario = row["scenario"]
        if scenario not in scenarios:
            scenarios.append(scenario)
    return scenarios


def _formal_delta_rows(rows: list[dict[str, str]], scenario: str) -> list[dict[str, str]]:
    selected = [
        row for row in rows
        if row.get("row_type") == "per_run"
        and row.get("lane") == "formal_length_phase_a_candidate"
        and row.get("scenario") == scenario
        and row.get("mean_delta_ukf_minus_hybrid")
    ]
    return sorted(selected, key=lambda row: int(row["selected_repeat_range"].split("-")[0]))


def _complete_formal_scenarios(rows: list[dict[str, str]]) -> list[str]:
    complete = []
    for row in rows:
        if (
            row.get("row_type") == "cumulative_by_scenario_lane"
            and row.get("lane") == "formal_length_phase_a_candidate"
            and row.get("coverage_status") == "complete_full_repeats"
        ):
            complete.append(row["scenario"])
    order = _scenario_order(rows)
    return [scenario for scenario in order if scenario in complete]


def _t_critical(df: int) -> float:
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    # Normal approximation for larger n; current Phase A target is small-n.
    return 1.959964


def _bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    sample_indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[sample_indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_scenario(scenario: str, rows: list[dict[str, str]], rng: np.random.Generator) -> dict[str, str]:
    delta_rows = _formal_delta_rows(rows, scenario)
    deltas = np.array([float(row["mean_delta_ukf_minus_hybrid"]) for row in delta_rows], dtype=float)
    if len(deltas) < 2:
        raise ValueError(f"Need at least two paired rows for interval analysis: {scenario}")

    n = int(len(deltas))
    mean_delta = float(deltas.mean())
    sample_sd = float(deltas.std(ddof=1))
    standard_error = sample_sd / math.sqrt(n)
    t_critical = _t_critical(n - 1)
    t_low = mean_delta - t_critical * standard_error
    t_high = mean_delta + t_critical * standard_error
    boot_low, boot_high = _bootstrap_interval(deltas, rng)
    lower_positive = t_low > 0.0 and boot_low > 0.0
    repeat_indices = ",".join(row["repeat_indices"] for row in delta_rows)
    source_summaries = ";".join(row["summary_json"] for row in delta_rows)
    positive_pairs = int(np.sum(deltas > 0.0))
    return {
        "scenario": scenario,
        "lane": "formal_length_phase_a_candidate",
        "n_paired_repeats": str(n),
        "repeat_indices": repeat_indices,
        "source_summaries": source_summaries,
        "mean_delta_ukf_minus_hybrid": f"{mean_delta:.12f}",
        "sample_sd_delta": f"{sample_sd:.12f}",
        "standard_error_delta": f"{standard_error:.12f}",
        "min_delta": f"{float(deltas.min()):.12f}",
        "max_delta": f"{float(deltas.max()):.12f}",
        "positive_pairs": f"{positive_pairs}/{n}",
        "paired_t_95_lower": f"{t_low:.12f}",
        "paired_t_95_upper": f"{t_high:.12f}",
        "bootstrap_95_lower": f"{boot_low:.12f}",
        "bootstrap_95_upper": f"{boot_high:.12f}",
        "interval_lower_bounds_positive": str(lower_positive).lower(),
        "claim_status": (
            "completed_scenario_interval_positive_but_all_scenario_gate_blocked"
            if lower_positive else "interval_not_positive"
        ),
        "claim_boundary": (
            "Formal-length paired interval over this completed scenario. "
            "It can support the direction and interval status for this scenario, "
            "but not all-scenario repeat-expanded advantage, holdout robustness, "
            "p-values, hardware latency/resource or deployment readiness."
        ),
    }


def build_rows() -> list[dict[str, str]]:
    summary_rows = read_rows(SUMMARY_CSV)
    scenarios = _complete_formal_scenarios(summary_rows)
    rng = np.random.default_rng(SEED)
    return [summarize_scenario(scenario, summary_rows, rng) for scenario in scenarios]


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "scenario",
        "lane",
        "n_paired_repeats",
        "repeat_indices",
        "source_summaries",
        "mean_delta_ukf_minus_hybrid",
        "sample_sd_delta",
        "standard_error_delta",
        "min_delta",
        "max_delta",
        "positive_pairs",
        "paired_t_95_lower",
        "paired_t_95_upper",
        "bootstrap_95_lower",
        "bootstrap_95_upper",
        "interval_lower_bounds_positive",
        "claim_status",
        "claim_boundary",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, str]]) -> None:
    payload = {
        "analysis_id": "submission_draft_phase_a_paired_interval_v1",
        "status": "generated",
        "seed": SEED,
        "n_bootstrap_resamples": N_BOOTSTRAP,
        "source_csv": str(SUMMARY_CSV),
        "scope": (
            "Paired interval analysis for completed formal-length Phase A "
            "scenario rows discovered in the repeat summary."
        ),
        "non_claims": [
            "does not run a benchmark",
            "does not analyze incomplete scenarios",
            "does not provide p-values or hypothesis-test evidence",
            "does not prove holdout robustness",
            "does not provide hardware timing, resource, power or source-vs-board evidence",
        ],
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(rows: list[dict[str, str]]) -> None:
    positive_scenarios = [
        row["scenario"] for row in rows
        if row["interval_lower_bounds_positive"] == "true"
    ]
    positive_scenario_text = (
        ", ".join(f"`{scenario}`" for scenario in positive_scenarios)
        if positive_scenarios else "none"
    )
    lines = [
        "# 投稿稿 Phase A paired interval 分析记录",
        "",
        "日期：2026-07-06",
        "",
        "本文档从 `submission_draft_phase_a_repeat_summary.csv` 中读取已经完成的 formal-length Phase A 场景行，计算 UKF-minus-Hybrid paired delta 的小样本 paired-t 区间和 paired bootstrap percentile 区间。它不运行 benchmark，不计算 p-value，也不补硬件证据。",
        "",
        "## 生成文件",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "## 结果",
        "",
        "| Scenario | n | Mean delta | Paired-t 95% interval | Bootstrap 95% interval | Positive pairs | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if not rows:
        lines.append("| n/a | 0 | n/a | n/a | n/a | n/a | no completed formal scenario row |")
    for row in rows:
        lines.append(
            f"| `{row['scenario']}` | {row['n_paired_repeats']} | "
            f"{float(row['mean_delta_ukf_minus_hybrid']):.6f} | "
            f"[{float(row['paired_t_95_lower']):.6f}, {float(row['paired_t_95_upper']):.6f}] | "
            f"[{float(row['bootstrap_95_lower']):.6f}, {float(row['bootstrap_95_upper']):.6f}] | "
            f"{row['positive_pairs']} | {row['claim_status']} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            f"- 可以写：已完成且 lower bounds 为正的 formal-length Phase A 场景包括：{positive_scenario_text}。",
            "- 可以写：这些结果只覆盖已完成场景的 formal interval check；四场景 repeat-expanded gate、pooled analysis、holdout drift 和硬件测量仍未完成，除非全部预声明场景均补齐并重新汇总。",
            "- 不能写：已经证明全场景 repeat-expanded advantage、p-value 显著性、holdout robustness、FPGA latency/resource/source-vs-board agreement 或 deployment readiness。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_csv(rows)
    write_json(rows)
    write_report(rows)
    print(json.dumps({"status": "ok", "rows": len(rows), "csv": str(OUT_CSV), "json": str(OUT_JSON)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
