"""Summarize completed Phase A repeat runs, if present.

The collector scans runs/paper_submission_phase_a for benchmark summary.json
files and writes a bounded source-data table.  Missing runs are represented as
an empty, explicitly non-claiming summary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "paper_submission_phase_a"
OUT_DIR = ROOT / "docs" / "paper_materials"
OUT_CSV = OUT_DIR / "submission_draft_phase_a_repeat_summary.csv"
OUT_JSON = OUT_DIR / "submission_draft_phase_a_repeat_summary.json"
REPORT_MD = OUT_DIR / "投稿稿phase_a_repeat_summary记录.md"

REQUIRED_MODES = ("ukf", "hybrid_residual_b")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_summary_files() -> list[Path]:
    if not RUN_ROOT.is_dir():
        return []
    return sorted(RUN_ROOT.rglob("summary.json"))


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def lane_from_summary(summary: dict[str, Any]) -> str:
    protocol = summary.get("protocol", {})
    protocol_id = str(protocol.get("protocol_id", ""))
    if "smoke" in protocol_id:
        return "smoke_length_feasibility"
    return "formal_length_phase_a_candidate"


def summarize_file(path: Path) -> list[dict[str, str]]:
    summary = read_summary(path)
    raw_rows = summary.get("raw_rows", [])
    filters = summary.get("filters", {})
    comparison_rows = summary.get("comparison_rows", [])
    expected_repeats = int(summary.get("protocol", {}).get("repeats", filters.get("repeat_stop", 0)) or 0)
    repeat_start = int(filters.get("repeat_start", 0) or 0)
    repeat_stop = int(filters.get("repeat_stop", expected_repeats) or expected_repeats)
    comparison_by_key = {
        (str(row["scenario"]), str(row["mode"])): row
        for row in comparison_rows
    }
    raw_index = {
        (str(row["scenario"]), str(row["mode"]), int(row["repeat"])): row
        for row in raw_rows
    }
    scenarios = sorted({str(row["scenario"]) for row in raw_rows})
    rows: list[dict[str, str]] = []
    for scenario in scenarios:
        ukf_repeats = {
            repeat
            for raw_scenario, raw_mode, repeat in raw_index
            if raw_scenario == scenario and raw_mode == "ukf"
        }
        hybrid_repeats = {
            repeat
            for raw_scenario, raw_mode, repeat in raw_index
            if raw_scenario == scenario and raw_mode == "hybrid_residual_b"
        }
        paired_repeats = sorted(ukf_repeats & hybrid_repeats)
        deltas = [
            float(raw_index[(scenario, "ukf", repeat)]["final_ler"])
            - float(raw_index[(scenario, "hybrid_residual_b", repeat)]["final_ler"])
            for repeat in paired_repeats
        ]
        mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
        ukf_comparison = comparison_by_key.get((scenario, "ukf"), {})
        hybrid_comparison = comparison_by_key.get((scenario, "hybrid_residual_b"), {})
        ukf_expected = int(ukf_comparison.get("expected_repeats", expected_repeats) or expected_repeats)
        hybrid_expected = int(hybrid_comparison.get("expected_repeats", expected_repeats) or expected_repeats)
        expected_pairs = max(ukf_expected, hybrid_expected, expected_repeats)
        planned_chunk_pairs = max(0, repeat_stop - repeat_start)
        if len(paired_repeats) >= expected_pairs and ukf_repeats == hybrid_repeats:
            coverage = "complete_full_repeats"
        elif len(paired_repeats) >= planned_chunk_pairs and ukf_repeats == hybrid_repeats:
            coverage = "complete_selected_chunk_only"
        else:
            coverage = "incomplete"
        lane = lane_from_summary(summary)
        rows.append(
            {
                "row_type": "per_run",
                "run_dir": str(Path(summary.get("run_dir", path.parent)).relative_to(ROOT)),
                "summary_json": str(path.relative_to(ROOT)),
                "summary_sha256": sha256(path),
                "lane": lane,
                "scenario": scenario,
                "modes": ",".join(REQUIRED_MODES),
                "paired_repeats": str(len(paired_repeats)),
                "expected_repeats": str(expected_pairs),
                "selected_repeat_range": f"{repeat_start}-{repeat_stop}",
                "completed_fraction": f"{len(paired_repeats)}/{expected_pairs}" if expected_pairs else "",
                "repeat_indices": ",".join(str(repeat) for repeat in paired_repeats),
                "mean_delta_ukf_minus_hybrid": "" if not deltas else f"{mean_delta:.12f}",
                "sample_sd_delta": "" if not deltas else f"{sample_sd(deltas):.12f}",
                "min_delta": "" if not deltas else f"{min(deltas):.12f}",
                "positive_pairs": "" if not deltas else f"{sum(1 for value in deltas if value > 0.0)}/{len(deltas)}",
                "coverage_status": coverage,
                "claim_boundary": (
                    "smoke-length feasibility only; not manuscript performance evidence"
                    if lane == "smoke_length_feasibility"
                    else "candidate formal-length source data only; not repeat-expanded evidence and requiring separate paired interval analysis before stronger wording"
                ),
            }
        )
    return rows


def build_cumulative_rows(per_run_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for row in per_run_rows:
        key = (row["lane"], row["scenario"])
        grouped.setdefault(key, {})
        repeats = [int(item) for item in row["repeat_indices"].split(",") if item != ""]
        deltas = [float(row["mean_delta_ukf_minus_hybrid"])] if len(repeats) == 1 else []
        if len(repeats) != 1:
            # Re-read per-repeat deltas from the source summary for multi-repeat rows.
            summary_path = ROOT / row["summary_json"]
            summary = read_summary(summary_path)
            raw_index = {
                (str(raw["scenario"]), str(raw["mode"]), int(raw["repeat"])): raw
                for raw in summary.get("raw_rows", [])
            }
            deltas = [
                float(raw_index[(row["scenario"], "ukf", repeat)]["final_ler"])
                - float(raw_index[(row["scenario"], "hybrid_residual_b", repeat)]["final_ler"])
                for repeat in repeats
                if (row["scenario"], "ukf", repeat) in raw_index
                and (row["scenario"], "hybrid_residual_b", repeat) in raw_index
            ]
        for repeat, delta in zip(repeats, deltas):
            grouped[key].setdefault(
                repeat,
                {
                    "delta": f"{delta:.12f}",
                    "source_summary": row["summary_json"],
                },
            )

    cumulative_rows: list[dict[str, str]] = []
    for (lane, scenario), repeat_map in sorted(grouped.items()):
        repeats = sorted(repeat_map)
        deltas = [float(repeat_map[repeat]["delta"]) for repeat in repeats]
        expected_repeats = "12"
        coverage_status = (
            "complete_full_repeats"
            if len(repeats) >= int(expected_repeats)
            else "incomplete_cumulative"
        )
        if lane == "smoke_length_feasibility":
            claim_boundary = "cumulative smoke-length source data only; not manuscript performance evidence"
        else:
            claim_boundary = (
                "cumulative formal-length source data only; not repeat-expanded evidence "
                "until all expected pairs complete and separate paired interval analysis passes"
            )
        cumulative_rows.append(
            {
                "row_type": "cumulative_by_scenario_lane",
                "run_dir": "multiple",
                "summary_json": ";".join(sorted({repeat_map[repeat]["source_summary"] for repeat in repeats})),
                "summary_sha256": "multiple",
                "lane": lane,
                "scenario": scenario,
                "modes": ",".join(REQUIRED_MODES),
                "paired_repeats": str(len(repeats)),
                "expected_repeats": expected_repeats,
                "selected_repeat_range": "aggregate",
                "completed_fraction": f"{len(repeats)}/{expected_repeats}",
                "repeat_indices": ",".join(str(repeat) for repeat in repeats),
                "mean_delta_ukf_minus_hybrid": "" if not deltas else f"{sum(deltas) / len(deltas):.12f}",
                "sample_sd_delta": "" if not deltas else f"{sample_sd(deltas):.12f}",
                "min_delta": "" if not deltas else f"{min(deltas):.12f}",
                "positive_pairs": "" if not deltas else f"{sum(1 for value in deltas if value > 0.0)}/{len(deltas)}",
                "coverage_status": coverage_status,
                "claim_boundary": claim_boundary,
            }
        )
    return cumulative_rows


def build_rows() -> list[dict[str, str]]:
    per_run_rows: list[dict[str, str]] = []
    for summary_path in find_summary_files():
        per_run_rows.extend(summarize_file(summary_path))
    return per_run_rows + build_cumulative_rows(per_run_rows)


def row_sort_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["scenario"], row["lane"], row["row_type"], row["run_dir"])


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "row_type",
        "run_dir",
        "summary_json",
        "summary_sha256",
        "lane",
        "scenario",
        "modes",
        "paired_repeats",
        "expected_repeats",
        "selected_repeat_range",
        "completed_fraction",
        "repeat_indices",
        "mean_delta_ukf_minus_hybrid",
        "sample_sd_delta",
        "min_delta",
        "positive_pairs",
        "coverage_status",
        "claim_boundary",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=row_sort_key))


def write_json(rows: list[dict[str, str]]) -> None:
    OUT_JSON.write_text(
        json.dumps(
            {
                "analysis_id": "submission_draft_phase_a_repeat_summary_v1",
                "status": "generated" if rows else "no_phase_a_runs_found",
                "run_root": str(RUN_ROOT),
                "scope": (
                    "Collector for completed Phase A repeat runs.  Rows are "
                    "bounded by lane; smoke-length rows are feasibility-only."
                ),
                "non_claims": [
                    "does not run a benchmark",
                    "does not convert smoke-length output into manuscript performance evidence",
                    "does not provide confidence intervals, p-values or statistical significance",
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
        "# 投稿稿 Phase A repeat summary 记录",
        "",
        "日期：2026-07-06",
        "",
        "本文档汇总 `runs/paper_submission_phase_a` 下已经完成的 Phase A repeat runs。它只做 source-data 汇总和边界登记，不运行 benchmark，不报告 CI / p-value，也不把 smoke-length rows 升级成主文性能证据。",
        "",
        "## 生成文件",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_JSON.relative_to(ROOT)}`",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "## 当前状态",
                "",
                "- 未发现 `runs/paper_submission_phase_a/**/summary.json`。",
                "- 因此当前只有 Phase A execution plan，没有 completed Phase A repeat source rows。",
            ]
        )
    else:
        lines.extend(
            [
                "## Completed rows",
                "",
                "| Row type | Lane | Scenario | Paired repeats | Expected repeats | Selected range | Mean delta | Positive pairs | Boundary |",
                "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | --- |",
            ]
        )
        for row in sorted(rows, key=row_sort_key):
            mean_delta = row["mean_delta_ukf_minus_hybrid"] or "n/a"
            lines.append(
                f"| `{row['row_type']}` | `{row['lane']}` | `{row['scenario']}` | {row['paired_repeats']} | "
                f"{row['expected_repeats']} | `{row['selected_repeat_range']}` | "
                f"{mean_delta} | {row['positive_pairs'] or 'n/a'} | {row['claim_boundary']} |"
            )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：collector 会把已完成 Phase A rows 汇总为 scenario-level paired deltas，并保留 source summary hash。",
            "- 可以写：smoke-length rows 只证明 command shape、missing-row accounting 和 source-data collector 可工作。",
            "- 不能写：smoke-length rows 是主文性能证据、expanded benchmark、robustness proof、statistical significance 或硬件证据。",
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
