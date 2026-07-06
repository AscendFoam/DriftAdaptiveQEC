"""Summarize a bounded runner smoke pair for submission-draft planning.

This script records a completed one-scenario, one-repeat UKF-vs-Hybrid smoke
run from the formal runner.  It is intentionally excluded from the manuscript's
main evidence layer: the output is a feasibility and planning artifact, not an
expanded benchmark, confidence interval, or hardware result.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "paper_submission_pilot" / "smoke_static_ukf_hybrid_r1"
SUMMARY_JSON = RUN_DIR / "summary.json"
COMPARISON_CSV = RUN_DIR / "comparison.csv"
LAUNCH_PLAN_JSON = RUN_DIR / "launch_plan.json"
PROGRESS_JSONL = RUN_DIR / "progress.jsonl"

OUT_CSV = ROOT / "docs" / "paper_materials" / "submission_draft_runner_smoke_pair.csv"
OUT_JSON = ROOT / "docs" / "paper_materials" / "submission_draft_runner_smoke_pair.json"
REPORT_MD = ROOT / "docs" / "paper_materials" / "投稿稿runner_smoke_pair记录.md"


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


def build_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    comparison = {row["mode"]: row for row in summary["comparison_rows"]}
    ukf = comparison["ukf"]
    hybrid = comparison["hybrid_residual_b"]
    delta = float(ukf["final_ler_mean"]) - float(hybrid["final_ler_mean"])
    relative = 100.0 * delta / float(ukf["final_ler_mean"])
    for mode_name, row in comparison.items():
        rows.append(
            {
                "scenario": row["scenario"],
                "mode": mode_name,
                "mode_label": row["mode_label"],
                "completed_repeats": row["completed_repeats"],
                "expected_repeats": row["expected_repeats"],
                "final_ler_mean": row["final_ler_mean"],
                "overflow_rate_mean": row["overflow_rate_mean"],
                "histogram_input_saturation_rate_mean": row["histogram_input_saturation_rate_mean"],
                "n_commits_applied_mean": row["n_commits_applied_mean"],
                "slow_update_violation_rate_mean": row["slow_update_violation_rate_mean"],
                "fast_cycle_violation_rate_mean": row["fast_cycle_violation_rate_mean"],
                "artifact_path": row.get("artifact_path") or "",
                "ukf_minus_hybrid_final_ler_delta": delta,
                "relative_reduction_percent": relative,
                "planning_boundary": (
                    "one-scenario one-repeat smoke pair; feasibility/planning only; "
                    "not an expanded benchmark, CI, p-value, robustness proof or hardware result"
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "scenario",
        "mode",
        "mode_label",
        "completed_repeats",
        "expected_repeats",
        "final_ler_mean",
        "overflow_rate_mean",
        "histogram_input_saturation_rate_mean",
        "n_commits_applied_mean",
        "slow_update_violation_rate_mean",
        "fast_cycle_violation_rate_mean",
        "artifact_path",
        "ukf_minus_hybrid_final_ler_delta",
        "relative_reduction_percent",
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
                "analysis_id": "submission_draft_runner_smoke_pair_v1",
                "status": "generated",
                "run_dir": str(RUN_DIR),
                "source_files": {
                    "summary_json": {
                        "path": str(SUMMARY_JSON),
                        "sha256": sha256(SUMMARY_JSON),
                    },
                    "comparison_csv": {
                        "path": str(COMPARISON_CSV),
                        "sha256": sha256(COMPARISON_CSV),
                    },
                    "launch_plan_json": {
                        "path": str(LAUNCH_PLAN_JSON),
                        "sha256": sha256(LAUNCH_PLAN_JSON),
                    },
                    "progress_jsonl": {
                        "path": str(PROGRESS_JSONL),
                        "sha256": sha256(PROGRESS_JSONL),
                    },
                },
                "config_hash": summary.get("config_hash"),
                "git_commit": summary.get("git_commit"),
                "protocol": summary.get("protocol"),
                "filters": summary.get("filters"),
                "non_claims": [
                    "not an expanded benchmark",
                    "not an inferential interval, p-value or standard error",
                    "not a holdout-robustness result",
                    "not a hardware timing/resource/source-vs-board result",
                    "not used as a main-text performance claim",
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
    ukf_row = next(row for row in rows if row["mode"] == "ukf")
    hybrid_row = next(row for row in rows if row["mode"] == "hybrid_residual_b")
    lines = [
        "# 投稿稿 runner smoke pair 记录",
        "",
        "日期：2026-07-03",
        "",
        "本文档记录一次用于投稿稿规划的 runner smoke pair：单个 scenario、单个 paired repeat、只比较 UKF 与 Hybrid Residual-B。它证明当前 runner 路径可执行并产出可解析的 comparison/summary 文件，但不进入论文主结果层。",
        "",
        "## Source run",
        "",
        f"- run dir: `{RUN_DIR.relative_to(ROOT)}`",
        f"- summary: `{SUMMARY_JSON.relative_to(ROOT)}`",
        f"- comparison: `{COMPARISON_CSV.relative_to(ROOT)}`",
        "",
        "## Pilot result",
        "",
        "| Scenario | UKF final LER | Hybrid final LER | UKF-minus-Hybrid delta | Relative reduction |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| `{ukf_row['scenario']}` | {float(ukf_row['final_ler_mean']):.6f} | "
            f"{float(hybrid_row['final_ler_mean']):.6f} | "
            f"{float(ukf_row['ukf_minus_hybrid_final_ler_delta']):.6f} | "
            f"{float(ukf_row['relative_reduction_percent']):.2f}% |"
        ),
        "",
        "## 可写边界",
        "",
        "- 可以写入内部材料：正式 runner 的 UKF/Hybrid smoke pair 已可执行并可被机器解析。",
        "- 可以用于估算：repeat-expanded benchmark 的运行成本和后续任务规模。",
        "- 不能写入主文性能 claim：它不是 expanded benchmark，不是 CI/p-value，不是 holdout robustness，不是硬件证据。",
    ]
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
