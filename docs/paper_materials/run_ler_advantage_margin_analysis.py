from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_MATERIALS = ROOT / "docs" / "paper_materials"
FIG_DIR = ROOT / "docs" / "figure_assets" / "submission_draft_python_figures"
MAIN_RESULTS_CSV = FIG_DIR / "source_data_fig02_main_results.csv"
PAIRED_DELTAS_CSV = FIG_DIR / "source_data_fig02_paired_deltas.csv"
CSV_PATH = PAPER_MATERIALS / "submission_draft_ler_advantage_margin_analysis.csv"
JSON_PATH = PAPER_MATERIALS / "submission_draft_ler_advantage_margin_analysis.json"
REPORT_PATH = PAPER_MATERIALS / "投稿稿LER优势幅度分析记录.md"

SCENARIO_ORDER = [
    "static_bias_theta",
    "linear_ramp",
    "step_sigma_theta",
    "periodic_drift",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _main_by_key() -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["scenario"], row["mode"]): row
        for row in _read_csv(MAIN_RESULTS_CSV)
    }


def _paired_by_scenario() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_csv(PAIRED_DELTAS_CSV):
        grouped.setdefault(row["scenario"], []).append(row)
    return grouped


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _row_for_scenario(
    scenario: str,
    main_rows: dict[tuple[str, str], dict[str, str]],
    paired_rows: dict[str, list[dict[str, str]]],
) -> dict[str, float | int | str]:
    ukf = main_rows[(scenario, "UKF")]
    hybrid = main_rows[(scenario, "Hybrid-b")]
    paired = paired_rows[scenario]
    deltas = [float(row["delta_final_ler_ukf_minus_hybrid"]) for row in paired]
    relative = [float(row["relative_reduction_percent"]) for row in paired]
    ukf_sd = float(ukf["final_ler_sd"])
    hybrid_sd = float(hybrid["final_ler_sd"])
    max_sd = max(ukf_sd, hybrid_sd)
    pooled_sd = math.sqrt((ukf_sd * ukf_sd + hybrid_sd * hybrid_sd) / 2.0)
    mean_delta = _mean(deltas)
    positive_count = sum(1 for value in deltas if value > 0.0)
    n = len(paired)
    return {
        "scenario": scenario,
        "n_paired_repeats": n,
        "ukf_final_ler_mean": float(ukf["final_ler_mean"]),
        "hybrid_final_ler_mean": float(hybrid["final_ler_mean"]),
        "mean_delta_ukf_minus_hybrid": mean_delta,
        "mean_relative_reduction_percent": _mean(relative),
        "min_paired_delta_ukf_minus_hybrid": min(deltas),
        "max_paired_delta_ukf_minus_hybrid": max(deltas),
        "positive_pair_count": positive_count,
        "paired_direction": f"{positive_count}/{n}",
        "ukf_final_ler_sd": ukf_sd,
        "hybrid_final_ler_sd": hybrid_sd,
        "max_descriptive_sd": max_sd,
        "pooled_descriptive_sd": pooled_sd,
        "delta_over_max_descriptive_sd": mean_delta / max_sd if max_sd else "",
        "delta_over_pooled_descriptive_sd": mean_delta / pooled_sd if pooled_sd else "",
        "boundary": (
            "Descriptive effect-size-style margin over existing paired repeats only; "
            "not an inferential confidence interval, standard error, p-value, "
            "significance test, expanded benchmark or hardware measurement."
        ),
    }


def _rows() -> list[dict[str, float | int | str]]:
    main_rows = _main_by_key()
    paired_rows = _paired_by_scenario()
    return [_row_for_scenario(scenario, main_rows, paired_rows) for scenario in SCENARIO_ORDER]


def _write_csv(rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "scenario",
        "n_paired_repeats",
        "ukf_final_ler_mean",
        "hybrid_final_ler_mean",
        "mean_delta_ukf_minus_hybrid",
        "mean_relative_reduction_percent",
        "min_paired_delta_ukf_minus_hybrid",
        "max_paired_delta_ukf_minus_hybrid",
        "positive_pair_count",
        "paired_direction",
        "ukf_final_ler_sd",
        "hybrid_final_ler_sd",
        "max_descriptive_sd",
        "pooled_descriptive_sd",
        "delta_over_max_descriptive_sd",
        "delta_over_pooled_descriptive_sd",
        "boundary",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(rows: list[dict[str, float | int | str]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_ler_advantage_margin_v1",
        "source_csvs": [
            str(MAIN_RESULTS_CSV),
            str(PAIRED_DELTAS_CSV),
        ],
        "scope": (
            "Source-data-backed descriptive UKF-minus-Hybrid LER advantage "
            "margins for the four predeclared software-HIL scenarios."
        ),
        "non_claims": [
            "not a confidence interval",
            "not a standard error",
            "not a p-value",
            "not a significance test",
            "not an expanded benchmark",
            "not hardware evidence",
        ],
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, float | int | str]]) -> None:
    lines = [
        "# 投稿稿 LER 优势幅度分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录从 Fig. 2 的 `source_data_fig02_main_results.csv` 与 "
        "`source_data_fig02_paired_deltas.csv` 派生 UKF-minus-Hybrid LER 优势幅度。"
        "它不重新运行 benchmark，不新增实验，也不提供统计显著性结论。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 结果摘要",
        "",
        "| Scenario | UKF mean | Hybrid mean | Mean delta | Rel. reduction | Direction | Delta/max SD |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['scenario']}` | "
            f"{float(row['ukf_final_ler_mean']):.6f} | "
            f"{float(row['hybrid_final_ler_mean']):.6f} | "
            f"{float(row['mean_delta_ukf_minus_hybrid']):.6f} | "
            f"{float(row['mean_relative_reduction_percent']):.2f}% | "
            f"{row['paired_direction']} | "
            f"{float(row['delta_over_max_descriptive_sd']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：四个预声明场景中，现有 paired repeats 的 UKF-minus-Hybrid delta 均为正。",
            "- 可以写：mean delta、relative reduction 和 delta/max reported SD 是描述性优势幅度读数。",
            "- 不可以写：这些数字构成置信区间、标准误、p 值、显著性检验、expanded benchmark、holdout robustness 或硬件测量。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _rows()
    _write_csv(rows)
    _write_json(rows)
    _write_report(rows)
    print(
        json.dumps(
            {
                "status": "generated",
                "rows": len(rows),
                "csv": str(CSV_PATH),
                "json": str(JSON_PATH),
                "report": str(REPORT_PATH),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
