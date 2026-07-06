"""Build validation-threshold gates for the Phase A repeat-expanded route.

The output is a manuscript-facing boundary table.  It states what the current
evidence can say, which evidence class is still missing, and which wording must
remain forbidden until that evidence exists.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_MATERIALS = ROOT / "docs" / "paper_materials"
PLAN_CSV = PAPER_MATERIALS / "submission_draft_phase_a_repeat_plan.csv"
SUMMARY_CSV = PAPER_MATERIALS / "submission_draft_phase_a_repeat_summary.csv"
INTERVAL_CSV = PAPER_MATERIALS / "submission_draft_phase_a_paired_interval_analysis.csv"
EXPANSION_PROTOCOL_CSV = PAPER_MATERIALS / "submission_draft_benchmark_expansion_protocol.csv"
OUT_CSV = PAPER_MATERIALS / "submission_draft_phase_a_upgrade_gate.csv"
OUT_JSON = PAPER_MATERIALS / "submission_draft_phase_a_upgrade_gate.json"
REPORT_MD = PAPER_MATERIALS / "投稿稿phase_a_upgrade_gate记录.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def smoke_summary_status(summary_rows: list[dict[str, str]]) -> str:
    smoke_rows = [row for row in summary_rows if row["lane"] == "smoke_length_feasibility"]
    if not smoke_rows:
        return "not run"
    complete = [row for row in smoke_rows if row["coverage_status"] == "complete"]
    return f"{len(complete)} short-run scenario(s) complete; feasibility only"


def formal_plan_status(
    plan_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    interval_rows: list[dict[str, str]],
) -> str:
    formal_summary_rows = [
        row for row in summary_rows
        if row["lane"] == "formal_length_phase_a_candidate"
        and row.get("row_type") == "cumulative_by_scenario_lane"
    ]
    formal_full_rows = [
        row for row in formal_summary_rows
        if row.get("coverage_status") == "complete_full_repeats"
    ]
    positive_interval_rows = [
        row for row in interval_rows
        if row.get("interval_lower_bounds_positive") == "true"
    ]
    if positive_interval_rows:
        return (
            f"{len(positive_interval_rows)} formal scenario row(s) complete "
            "with positive paired interval; all-scenario gate still incomplete"
        )
    if formal_full_rows:
        return f"{len(formal_full_rows)} formal scenario row(s) complete; interval analysis still required"
    if formal_summary_rows:
        total_pairs = sum(int(row.get("paired_repeats", "0") or 0) for row in formal_summary_rows)
        return f"{total_pairs} unique formal-length paired row(s) complete; no scenario-level repeat expansion complete"
    formal_rows = [row for row in plan_rows if row["lane"] == "formal_length_phase_a"]
    scenarios = sorted({row["scenario"] for row in formal_rows})
    chunks_per_scenario = {
        scenario: len([row for row in formal_rows if row["scenario"] == scenario])
        for scenario in scenarios
    }
    if len(scenarios) == 4 and all(count == 3 for count in chunks_per_scenario.values()):
        return "planned; no formal-length completed rows"
    return "plan incomplete"


def build_rows() -> list[dict[str, str]]:
    plan_rows = read_csv(PLAN_CSV)
    summary_rows = read_csv(SUMMARY_CSV)
    interval_rows = read_csv(INTERVAL_CSV) if INTERVAL_CSV.is_file() else []
    protocol_rows = read_csv(EXPANSION_PROTOCOL_CSV)
    phase_a_protocol = [row for row in protocol_rows if row["protocol_phase"] == "Phase A repeat-expanded anchor comparison"]
    target_repeats = phase_a_protocol[0]["recommended_target_pairs_per_scenario"] if phase_a_protocol else "16"
    min_repeats = phase_a_protocol[0]["planning_min_pairs_per_scenario"] if phase_a_protocol else "12"

    return [
        {
            "evidence_class": "Current descriptive benchmark",
            "current_status": "satisfied for descriptive ranking",
            "allowed_wording": "Hybrid-b has the lowest descriptive mean final LER in the predeclared software-HIL rows.",
            "upgrade_condition": "None; this is the current claim ceiling.",
            "forbidden_inference": "Do not state statistical significance, broad robustness, hardware latency or deployment readiness.",
        },
        {
            "evidence_class": "Short-run repeat rehearsal",
            "current_status": smoke_summary_status(summary_rows),
            "allowed_wording": "The repeat-expansion command path, row accounting and collector logic were rehearsed.",
            "upgrade_condition": "No stronger statement is permitted from short-run rows.",
            "forbidden_inference": "Do not use short-run rows as manuscript performance evidence or inferential uncertainty.",
        },
        {
            "evidence_class": "Formal Phase A repeat expansion",
            "current_status": formal_plan_status(plan_rows, summary_rows, interval_rows),
            "allowed_wording": "Completed formal scenarios may be described as scenario-level positive paired-interval checks; all-scenario and pooled wording remain planned.",
            "upgrade_condition": (
                f"Complete all four scenarios with at least {min_repeats} and preferably "
                f"{target_repeats} paired repeats, no missing paired rows, and positive "
                "predeclared paired-interval lower bounds per scenario plus pooled analysis."
            ),
            "forbidden_inference": "Do not claim repeat-expanded advantage, confidence intervals, p-values or robustness before this gate passes.",
        },
        {
            "evidence_class": "Formal holdout drift expansion",
            "current_status": "planned only",
            "allowed_wording": "Holdout drift families are specified as future validation families.",
            "upgrade_condition": "Run the predeclared random-walk, burst/reset and faster-than-window families with missing-run accounting.",
            "forbidden_inference": "Do not treat controlled stress diagnostics or anchor scenarios as holdout generalization proof.",
        },
        {
            "evidence_class": "Hardware-facing measurements",
            "current_status": "not measured",
            "allowed_wording": "Hardware latency, resource, power and source-vs-board agreement remain measurement targets.",
            "upgrade_condition": "Provide board logs, bitstream or RTL hash, source vectors, measured latency/resource/power and source-vs-board agreement.",
            "forbidden_inference": "Do not claim FPGA timing closure, resource efficiency, source-vs-board agreement or board-level correction success.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "evidence_class",
        "current_status",
        "allowed_wording",
        "upgrade_condition",
        "forbidden_inference",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, str]]) -> None:
    OUT_JSON.write_text(
        json.dumps(
            {
                "analysis_id": "submission_draft_phase_a_upgrade_gate_v1",
                "source_files": {
                "plan_csv": str(PLAN_CSV),
                "summary_csv": str(SUMMARY_CSV),
                "interval_csv": str(INTERVAL_CSV),
                "expansion_protocol_csv": str(EXPANSION_PROTOCOL_CSV),
            },
                "scope": "Manuscript-facing validation-threshold gates for statistical and hardware wording.",
                "non_claims": [
                    "does not run a benchmark",
                    "does not compute confidence intervals or p-values",
                    "does not convert short-run rows into performance evidence",
                    "does not provide holdout robustness or hardware validation",
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
    lines = [
        "# 投稿稿 Phase A validation-threshold gate 记录",
        "",
        "日期：2026-07-06",
        "",
        "本文档服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它把当前描述性结果、short-run rehearsal、formal repeat expansion、holdout drift 与 hardware measurements 分成独立验证门槛，防止把计划或短运行覆盖演练写成主结果。",
        "",
        "## 生成文件",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "## Gate matrix",
        "",
        "| Evidence class | Current status | Upgrade condition | Forbidden inference |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['evidence_class']} | {row['current_status']} | {row['upgrade_condition']} | {row['forbidden_inference']} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：稿件已有明确的 validation-threshold gate，区分 descriptive ranking、short-run rehearsal、formal repeat expansion、holdout expansion 和 hardware measurement。",
            "- 可以写：short-run rehearsal 只验证执行路径和 collector，不改变主文性能结论。",
            "- 不能写：当前材料已经提供 CI、p-value、repeat-expanded advantage、holdout robustness、hardware latency/resource 或 source-vs-board agreement。",
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
