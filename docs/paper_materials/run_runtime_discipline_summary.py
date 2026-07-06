from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CSV = ROOT / "runs" / "p4_benchmark" / "T24_formal_software_revalidation_20260510_200743" / "comparison.csv"
OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_runtime_discipline_summary.csv"
JSON_PATH = OUT_DIR / "submission_draft_runtime_discipline_summary.json"
REPORT_PATH = OUT_DIR / "投稿稿runtime_discipline分析记录.md"


FIELDS = [
    "n_commits_applied_mean",
    "slow_update_violation_rate_mean",
    "fast_cycle_violation_rate_mean",
    "overflow_rate_mean",
    "histogram_input_saturation_rate_mean",
    "correction_saturation_rate_mean",
    "aggressive_param_rate_mean",
]


def _read_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _mean_float(rows: list[dict[str, str]], field: str) -> float:
    return mean(float(row[field]) for row in rows)


def _summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_mode: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode_label"], []).append(row)

    output: list[dict[str, object]] = []
    for mode_label in sorted(by_mode):
        mode_rows = by_mode[mode_label]
        item: dict[str, object] = {
            "mode_label": mode_label,
            "n_scenario_rows": len(mode_rows),
        }
        for field in FIELDS:
            item[field] = _mean_float(mode_rows, field)
        item["dominant_overflow_sources"] = ";".join(sorted({row["dominant_overflow_source"] for row in mode_rows}))
        item["scope"] = "software-in-the-loop runtime counters averaged across scenario-level comparison rows"
        output.append(item)
    return output


def _write_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "mode_label",
        "n_scenario_rows",
        *FIELDS,
        "dominant_overflow_sources",
        "scope",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_json(rows: list[dict[str, object]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_runtime_discipline_summary_v1",
        "source_csv": str(SOURCE_CSV.relative_to(ROOT)),
        "boundary": (
            "Software-in-the-loop runtime-counter summary derived from the preserved comparison CSV. "
            "Not board commit latency, not hardware reliability, not source-vs-board agreement."
        ),
        "fields": FIELDS,
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# 投稿稿 runtime-discipline 分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它从 preserved comparison CSV 派生 software-in-the-loop runtime counters，用于说明稿件中 stage-and-commit / saturation / cycle-violation 口径的当前可写边界。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 结果摘要",
        "",
        "| Mode | Commits applied | Slow-update violation | Fast-cycle violation | Overflow | Correction saturation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode_label']} | "
            f"{float(row['n_commits_applied_mean']):.1f} | "
            f"{float(row['slow_update_violation_rate_mean']):.6g} | "
            f"{float(row['fast_cycle_violation_rate_mean']):.6g} | "
            f"{float(row['overflow_rate_mean']):.6g} | "
            f"{float(row['correction_saturation_rate_mean']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：preserved software-in-the-loop comparison rows include runtime counters for commit activity, slow-update violation, fast-cycle violation, overflow and saturation.",
            "- 可以写：这些 counters 支持 stage-and-commit contract 在软件协议中的可观测性。",
            "- 不能写：这些 counters 是 board commit latency、hardware reliability、rollback proof、source-vs-board agreement 或 FPGA timing/resource evidence。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _summarize(_read_rows())
    _write_csv(rows)
    _write_json(rows)
    _write_report(rows)
    print(json.dumps({"status": "generated", "csv": str(CSV_PATH), "json": str(JSON_PATH), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
