"""Summarize the all-scenario UKF-vs-Hybrid runner smoke matrix.

The source run is a smoke-length feasibility pilot for the repeat-expansion
route.  It checks that the formal runner can execute all four manuscript
scenarios for the two anchor modes with paired seeds and two repeats.  It is
not a manuscript performance benchmark or an inferential result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "paper_submission_pilot" / "smoke_all_scenarios_ukf_hybrid_r2_20260703_093537"
SUMMARY_JSON = RUN_DIR / "summary.json"
COMPARISON_CSV = RUN_DIR / "comparison.csv"
LAUNCH_PLAN_JSON = RUN_DIR / "launch_plan.json"
PROGRESS_JSONL = RUN_DIR / "progress.jsonl"

OUT_CSV = ROOT / "docs" / "paper_materials" / "submission_draft_runner_smoke_matrix.csv"
OUT_JSON = ROOT / "docs" / "paper_materials" / "submission_draft_runner_smoke_matrix.json"
REPORT_MD = ROOT / "docs" / "paper_materials" / "投稿稿runner_smoke_matrix记录.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_summary() -> dict[str, Any]:
    if not SUMMARY_JSON.is_file():
        raise FileNotFoundError(SUMMARY_JSON)
    return json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))


def _index_raw_rows(summary: dict[str, Any]) -> dict[tuple[str, str, int], float]:
    indexed: dict[tuple[str, str, int], float] = {}
    for row in summary["raw_rows"]:
        indexed[(str(row["scenario"]), str(row["mode"]), int(row["repeat"]))] = float(row["final_ler"])
    return indexed


def build_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = {(row["scenario"], row["mode"]): row for row in summary["comparison_rows"]}
    raw_index = _index_raw_rows(summary)
    scenarios = [row["scenario"] for row in summary["comparison_rows"] if row["mode"] == "ukf"]
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        ukf = comparison[(scenario, "ukf")]
        hybrid = comparison[(scenario, "hybrid_residual_b")]
        repeats = sorted(
            repeat
            for raw_scenario, raw_mode, repeat in raw_index
            if raw_scenario == scenario and raw_mode == "ukf"
        )
        pair_deltas = [
            raw_index[(scenario, "ukf", repeat)] - raw_index[(scenario, "hybrid_residual_b", repeat)]
            for repeat in repeats
            if (scenario, "hybrid_residual_b", repeat) in raw_index
        ]
        delta = float(ukf["final_ler_mean"]) - float(hybrid["final_ler_mean"])
        relative = 100.0 * delta / float(ukf["final_ler_mean"])
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": ukf["scenario_label"],
                "ukf_final_ler_mean": float(ukf["final_ler_mean"]),
                "hybrid_final_ler_mean": float(hybrid["final_ler_mean"]),
                "ukf_minus_hybrid_final_ler_delta": delta,
                "relative_reduction_percent": relative,
                "completed_pairs": len(pair_deltas),
                "positive_pairs": f"{sum(1 for value in pair_deltas if value > 0.0)}/{len(pair_deltas)}",
                "ukf_completed_repeats": int(ukf["completed_repeats"]),
                "hybrid_completed_repeats": int(hybrid["completed_repeats"]),
                "expected_repeats": int(ukf["expected_repeats"]),
                "coverage": min(float(ukf["coverage"]), float(hybrid["coverage"])),
                "n_slow_updates": 120,
                "n_fast_cycles": 480000,
                "planning_boundary": (
                    "all-scenario smoke-length runner pilot; repeat-expansion feasibility only; "
                    "not the main benchmark, not an expanded benchmark, not CI/p-value evidence, "
                    "not holdout robustness and not hardware timing/resource/source-vs-board evidence"
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "scenario",
        "scenario_label",
        "ukf_final_ler_mean",
        "hybrid_final_ler_mean",
        "ukf_minus_hybrid_final_ler_delta",
        "relative_reduction_percent",
        "completed_pairs",
        "positive_pairs",
        "ukf_completed_repeats",
        "hybrid_completed_repeats",
        "expected_repeats",
        "coverage",
        "n_slow_updates",
        "n_fast_cycles",
        "planning_boundary",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    OUT_JSON.write_text(
        json.dumps(
            {
                "analysis_id": "submission_draft_runner_smoke_matrix_v1",
                "status": "generated",
                "run_dir": str(RUN_DIR),
                "source_files": {
                    "summary_json": {"path": str(SUMMARY_JSON), "sha256": sha256(SUMMARY_JSON)},
                    "comparison_csv": {"path": str(COMPARISON_CSV), "sha256": sha256(COMPARISON_CSV)},
                    "launch_plan_json": {"path": str(LAUNCH_PLAN_JSON), "sha256": sha256(LAUNCH_PLAN_JSON)},
                    "progress_jsonl": {"path": str(PROGRESS_JSONL), "sha256": sha256(PROGRESS_JSONL)},
                },
                "config_hash": summary.get("config_hash"),
                "git_commit": summary.get("git_commit"),
                "protocol": summary.get("protocol"),
                "filters": summary.get("filters"),
                "non_claims": [
                    "not the main manuscript benchmark",
                    "not an expanded benchmark",
                    "not an inferential interval, p-value or standard error",
                    "not a holdout-robustness result",
                    "not a hardware timing/resource/source-vs-board result",
                ],
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_report(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 投稿稿 runner smoke matrix 记录",
        "",
        "日期：2026-07-03",
        "",
        "本文档记录一次用于投稿稿 repeat-expansion 路线的 all-scenario runner pilot：四个预声明 scenario、UKF 与 Hybrid Residual-B 两个 anchor mode、两个 paired repeats、smoke-length 配置。它证明全场景 runner 路径可执行并能产出完整 comparison/summary 文件，但不进入论文主结果层。",
        "",
        "## Source run",
        "",
        f"- run dir: `{RUN_DIR.relative_to(ROOT)}`",
        f"- summary: `{SUMMARY_JSON.relative_to(ROOT)}`",
        f"- comparison: `{COMPARISON_CSV.relative_to(ROOT)}`",
        "",
        "## Pilot matrix",
        "",
        "| Scenario | UKF final LER | Hybrid final LER | UKF-minus-Hybrid delta | Relative reduction | Positive pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['scenario']}` | {row['ukf_final_ler_mean']:.6f} | "
            f"{row['hybrid_final_ler_mean']:.6f} | "
            f"{row['ukf_minus_hybrid_final_ler_delta']:.6f} | "
            f"{row['relative_reduction_percent']:.2f}% | {row['positive_pairs']} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写入投稿材料：repeat-expansion 的全场景 UKF/Hybrid runner pilot 已可执行，并且四个 smoke scenario 都有完整 paired rows。",
            "- 可以用于规划：正式 Phase A repeat-expanded benchmark 的运行成本、source-data 字段和缺失行检查。",
            "- 不能写入主文性能 claim：它使用 smoke-length timing，不是当前主结果 benchmark，不是 expanded benchmark，不是 CI/p-value，不是 holdout robustness，也不是硬件证据。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    summary = read_summary()
    rows = build_rows(summary)
    write_csv(rows)
    write_json(summary, rows)
    write_report(rows)
    print(json.dumps({"status": "ok", "rows": len(rows), "csv": str(OUT_CSV), "json": str(OUT_JSON)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
