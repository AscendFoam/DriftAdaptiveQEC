"""Build a benchmark-expansion protocol table for the submission draft.

The protocol is derived from the current paired UKF-vs-hybrid pilot rows. It
does not run a benchmark and does not turn the current n=2 result into an
inferential claim.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER_MATERIALS = ROOT / "docs" / "paper_materials"
FIG_DIR = ROOT / "docs" / "figure_assets" / "submission_draft_python_figures"
PAIRED_DELTAS_CSV = FIG_DIR / "source_data_fig02_paired_deltas.csv"
OUT_CSV = PAPER_MATERIALS / "submission_draft_benchmark_expansion_protocol.csv"
OUT_JSON = PAPER_MATERIALS / "submission_draft_benchmark_expansion_protocol.json"
REPORT_MD = PAPER_MATERIALS / "投稿稿benchmark_expansion_protocol补强记录.md"

MIN_REPEAT_PAIRS_PER_SCENARIO = 12
TARGET_REPEAT_PAIRS_PER_SCENARIO = 16
MAX_MAIN_CLAIM_REPEAT_PAIRS_PER_SCENARIO = 32
Z_975 = 1.96


def read_paired_rows() -> list[dict[str, str]]:
    with PAIRED_DELTAS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def planning_repeat_count(deltas: np.ndarray) -> int:
    """Planning-only repeat count for lower-bound-positive mean delta.

    With n=2 this is deliberately floored by MIN_REPEAT_PAIRS_PER_SCENARIO,
    because the observed sample variance is too unstable for power analysis.
    """
    if len(deltas) < 2:
        return MIN_REPEAT_PAIRS_PER_SCENARIO
    mean_delta = float(np.mean(deltas))
    sample_sd = float(np.std(deltas, ddof=1))
    if mean_delta <= 0 or sample_sd == 0:
        estimate = MIN_REPEAT_PAIRS_PER_SCENARIO
    else:
        estimate = math.ceil((Z_975 * sample_sd / mean_delta) ** 2)
    return max(MIN_REPEAT_PAIRS_PER_SCENARIO, min(MAX_MAIN_CLAIM_REPEAT_PAIRS_PER_SCENARIO, estimate))


def summarize_deltas(rows: list[dict[str, str]]) -> dict[str, str]:
    deltas = np.array([float(row["delta_final_ler_ukf_minus_hybrid"]) for row in rows])
    sample_sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    return {
        "current_paired_repeats": str(len(rows)),
        "current_mean_delta_ukf_minus_hybrid": f"{float(np.mean(deltas)):.12f}",
        "current_sample_sd_delta": f"{sample_sd:.12f}",
        "current_min_delta": f"{float(np.min(deltas)):.12f}",
        "current_positive_pairs": f"{int(np.sum(deltas > 0.0))}/{len(rows)}",
        "planning_min_pairs_per_scenario": str(planning_repeat_count(deltas)),
        "recommended_target_pairs_per_scenario": str(TARGET_REPEAT_PAIRS_PER_SCENARIO),
    }


def build_rows() -> list[dict[str, str]]:
    source_rows = read_paired_rows()
    scenarios = []
    for row in source_rows:
        if row["scenario"] not in scenarios:
            scenarios.append(row["scenario"])

    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        selected = [row for row in source_rows if row["scenario"] == scenario]
        rows.append(
            {
                "protocol_phase": "Phase A repeat-expanded anchor comparison",
                "scenario_family": scenario,
                "comparison_unit": "paired seed/repeat within scenario",
                "modes": "ukf,hybrid_residual_b",
                **summarize_deltas(selected),
                "inferential_upgrade_gate": (
                    "report paired interval over repeat-level deltas; upgrade wording only if "
                    "the predeclared interval lower bound remains positive for the scenario "
                    "and the pooled all-scenario paired analysis is positive"
                ),
                "non_claim_boundary": (
                    "planning row only; current n=2 data remain descriptive and this row does "
                    "not establish a confidence interval, p-value or robustness claim"
                ),
            }
        )

    pooled = summarize_deltas(source_rows)
    pooled["planning_min_pairs_per_scenario"] = str(MIN_REPEAT_PAIRS_PER_SCENARIO)
    rows.extend(
        [
            {
                "protocol_phase": "Phase A reporting rule",
                "scenario_family": "all_frozen_scenarios",
                "comparison_unit": "paired seed/repeat; scenario reported separately",
                "modes": "ukf,hybrid_residual_b",
                **pooled,
                "inferential_upgrade_gate": (
                    "predeclare paired bootstrap or paired t interval, report scenario-level "
                    "and pooled estimates, and avoid significance language if any scenario "
                    "has a non-positive interval lower bound"
                ),
                "non_claim_boundary": (
                    "pooled row is a planning and reporting rule; scenarios must not be "
                    "treated as independent substitutes for repeat-level samples"
                ),
            },
            {
                "protocol_phase": "Phase B holdout drift expansion",
                "scenario_family": "random_walk_drift,burst_reset_drift,faster_than_window_drift",
                "comparison_unit": "predeclared holdout scenario family",
                "modes": "fixed_affine,ukf,hybrid_residual_b,oracle_affine,wrapped_gaussian_posterior_mean",
                "current_paired_repeats": "0",
                "current_mean_delta_ukf_minus_hybrid": "",
                "current_sample_sd_delta": "",
                "current_min_delta": "",
                "current_positive_pairs": "",
                "planning_min_pairs_per_scenario": str(MIN_REPEAT_PAIRS_PER_SCENARIO),
                "recommended_target_pairs_per_scenario": str(TARGET_REPEAT_PAIRS_PER_SCENARIO),
                "inferential_upgrade_gate": (
                    "treat as a separate expansion lane; report holdout families separately "
                    "from the frozen four-scenario anchor and require complete missing-run accounting"
                ),
                "non_claim_boundary": (
                    "not currently executed in the software-HIL benchmark; existing controlled "
                    "stress diagnostics are not a substitute for this formal holdout lane"
                ),
            },
            {
                "protocol_phase": "Phase C hardware-independent runtime reporting",
                "scenario_family": "same rows as Phase A or B",
                "comparison_unit": "software runtime counter and artifact provenance",
                "modes": "all executed modes",
                "current_paired_repeats": "not_applicable",
                "current_mean_delta_ukf_minus_hybrid": "",
                "current_sample_sd_delta": "",
                "current_min_delta": "",
                "current_positive_pairs": "",
                "planning_min_pairs_per_scenario": "not_applicable",
                "recommended_target_pairs_per_scenario": "not_applicable",
                "inferential_upgrade_gate": (
                    "archive per-row config hash, commit, artifact path, artifact hash, runner "
                    "version and runtime counters before manuscript use"
                ),
                "non_claim_boundary": (
                    "software provenance and runtime counters are not FPGA timing, resource, "
                    "power or source-vs-board evidence"
                ),
            },
        ]
    )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "protocol_phase",
        "scenario_family",
        "comparison_unit",
        "modes",
        "current_paired_repeats",
        "current_mean_delta_ukf_minus_hybrid",
        "current_sample_sd_delta",
        "current_min_delta",
        "current_positive_pairs",
        "planning_min_pairs_per_scenario",
        "recommended_target_pairs_per_scenario",
        "inferential_upgrade_gate",
        "non_claim_boundary",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, str]]) -> None:
    OUT_JSON.write_text(
        json.dumps(
            {
                "analysis_id": "submission_draft_benchmark_expansion_protocol_v1",
                "source_csv": str(PAIRED_DELTAS_CSV),
                "scope": (
                    "Predeclared planning protocol for upgrading the current descriptive "
                    "software-HIL ranking into a repeat-expanded benchmark lane."
                ),
                "planning_constants": {
                    "min_repeat_pairs_per_scenario": MIN_REPEAT_PAIRS_PER_SCENARIO,
                    "target_repeat_pairs_per_scenario": TARGET_REPEAT_PAIRS_PER_SCENARIO,
                    "max_main_claim_repeat_pairs_per_scenario": MAX_MAIN_CLAIM_REPEAT_PAIRS_PER_SCENARIO,
                    "normal_approximation_z": Z_975,
                },
                "non_claims": [
                    "does not run a benchmark",
                    "does not provide inferential confidence intervals for the current n=2 data",
                    "does not establish holdout drift robustness",
                    "does not provide hardware timing, resource, power or source-vs-board evidence",
                ],
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_report(rows: list[dict[str, str]]) -> None:
    phase_a = [row for row in rows if row["protocol_phase"] == "Phase A repeat-expanded anchor comparison"]
    lines = [
        "# 投稿稿 benchmark expansion protocol 补强记录",
        "",
        "日期：2026-07-03",
        "",
        "本文档服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。本轮只把当前 `n=2` paired descriptive 主结果转换为下一步可执行的 repeat-expanded benchmark protocol；不运行 benchmark，不新增实验，不报告 CI / p-value，也不升级 hardware、`.tflite`、real-board 或 deployment 证据等级。",
        "",
        "## 生成文件",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "## Phase A：repeat-expanded anchor comparison",
        "",
        "Phase A 只比较当前主结果最关键的 `ukf` 与 `hybrid_residual_b`，保留现有五模式主表作为 anchor，不把 repeat expansion 静默写成新的五模式 full matrix。",
        "",
        "| Scenario | Current pairs | Mean delta | Sample SD | Min delta | Positive pairs | Planning min pairs | Target pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in phase_a:
        lines.append(
            f"| `{row['scenario_family']}` | {row['current_paired_repeats']} | "
            f"{float(row['current_mean_delta_ukf_minus_hybrid']):.6f} | "
            f"{float(row['current_sample_sd_delta']):.6f} | "
            f"{float(row['current_min_delta']):.6f} | "
            f"{row['current_positive_pairs']} | {row['planning_min_pairs_per_scenario']} | "
            f"{row['recommended_target_pairs_per_scenario']} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：当前稿件已把下一步强统计 benchmark 的 repeat unit、scenario unit、mode subset、minimum repeat budget 和 upgrade gate 机器可读化。",
            "- 可以写：Phase A 的目标是把 UKF-vs-Hybrid 从 descriptive paired deltas 升级为 repeat-expanded paired interval analysis。",
            "- 可以写：Phase B 才会处理 random-walk、burst/reset、faster-than-window 等 holdout drift family。",
            "- 不能写：当前 `n=2` 数据已有 confidence interval、p-value、significance 或 robustness proof。",
            "- 不能写：controlled holdout stress diagnostics 已经等价于正式 software-HIL holdout benchmark。",
            "- 不能写：该 protocol 证明了 FPGA latency/resource、source-vs-board agreement 或硬件有效性。",
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
