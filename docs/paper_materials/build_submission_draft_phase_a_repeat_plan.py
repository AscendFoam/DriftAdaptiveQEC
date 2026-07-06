"""Build a Phase A repeat-expanded benchmark execution plan.

This script creates a machine-readable command plan for the UKF-vs-Hybrid
anchor comparison discussed in the submission draft.  It does not run the
benchmark and does not upgrade the current descriptive result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "paper_materials"
OUT_CSV = OUT_DIR / "submission_draft_phase_a_repeat_plan.csv"
OUT_JSON = OUT_DIR / "submission_draft_phase_a_repeat_plan.json"
REPORT_MD = OUT_DIR / "投稿稿phase_a_repeat_plan记录.md"

SCENARIOS = (
    "static_bias_theta",
    "linear_ramp",
    "step_sigma_theta",
    "periodic_drift",
)
MODES = ("ukf", "hybrid_residual_b")

LANES = (
    {
        "lane": "formal_length_phase_a",
        "config": "cnn_fpga/config/p4_multiscenario_strong_baselines.yaml",
        "n_slow_updates": "900",
        "n_fast_cycles": "3600000",
        "repeats": 12,
        "chunks": ((0, 4), (4, 8), (8, 12)),
        "run_root": "runs/paper_submission_phase_a/formal",
        "purpose": "candidate formal repeat-expanded anchor comparison",
        "claim_boundary": (
            "planned execution only until all chunks complete and are audited; "
            "not CI/p-value evidence before post-run paired interval analysis"
        ),
    },
    {
        "lane": "smoke_length_feasibility",
        "config": "cnn_fpga/config/p4_multiscenario_strong_baselines_smoke.yaml",
        "n_slow_updates": "120",
        "n_fast_cycles": "480000",
        "repeats": 12,
        "chunks": ((0, 12),),
        "run_root": "runs/paper_submission_phase_a/smoke",
        "purpose": "runner feasibility and missing-row rehearsal only",
        "claim_boundary": (
            "smoke-length feasibility only; not the main benchmark, not an "
            "expanded benchmark, not holdout robustness and not hardware evidence"
        ),
    },
)


def command_for(lane: dict[str, object], scenario: str, start: int, stop: int, run_dir: str) -> str:
    mode_args = " ".join(f"--mode {mode}" for mode in MODES)
    return (
        "python -m cnn_fpga.benchmark.run_p4_multiscenario_benchmark "
        f"--config {lane['config']} "
        f"--scenario {scenario} "
        f"{mode_args} "
        f"--repeats {lane['repeats']} "
        "--paired-seeds "
        f"--run-dir {run_dir} "
        f"--repeat-start {start} --repeat-stop {stop}"
    )


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for lane in LANES:
        for scenario in SCENARIOS:
            for start, stop in lane["chunks"]:
                run_dir = f"{lane['run_root']}_{scenario}_ukf_hybrid_r{lane['repeats']}_{start:02d}_{stop:02d}"
                rows.append(
                    {
                        "lane": str(lane["lane"]),
                        "scenario": scenario,
                        "modes": ",".join(MODES),
                        "config": str(lane["config"]),
                        "n_slow_updates": str(lane["n_slow_updates"]),
                        "n_fast_cycles": str(lane["n_fast_cycles"]),
                        "paired_seeds": "true",
                        "total_repeats": str(lane["repeats"]),
                        "repeat_start": str(start),
                        "repeat_stop": str(stop),
                        "expected_pairs_in_chunk": str(stop - start),
                        "run_dir": run_dir,
                        "command": command_for(lane, scenario, start, stop, run_dir),
                        "purpose": str(lane["purpose"]),
                        "postrun_required_summary": (
                            "summary.json, comparison.csv, delta.csv, progress.jsonl, "
                            "per-repeat hil_summary.json and repeat_status.json"
                        ),
                        "claim_boundary": str(lane["claim_boundary"]),
                    }
                )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "lane",
        "scenario",
        "modes",
        "config",
        "n_slow_updates",
        "n_fast_cycles",
        "paired_seeds",
        "total_repeats",
        "repeat_start",
        "repeat_stop",
        "expected_pairs_in_chunk",
        "run_dir",
        "command",
        "purpose",
        "postrun_required_summary",
        "claim_boundary",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, str]]) -> None:
    OUT_JSON.write_text(
        json.dumps(
            {
                "analysis_id": "submission_draft_phase_a_repeat_plan_v1",
                "scope": (
                    "Chunked execution plan for a repeat-expanded UKF-vs-Hybrid "
                    "anchor comparison.  The plan is separate from completed "
                    "benchmark evidence."
                ),
                "scenarios": list(SCENARIOS),
                "modes": list(MODES),
                "non_claims": [
                    "does not run a benchmark",
                    "does not provide confidence intervals, p-values or significance claims",
                    "does not establish holdout drift robustness",
                    "does not provide FPGA timing, resource, power or source-vs-board evidence",
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
    formal = [row for row in rows if row["lane"] == "formal_length_phase_a"]
    smoke = [row for row in rows if row["lane"] == "smoke_length_feasibility"]
    lines = [
        "# 投稿稿 Phase A repeat plan 记录",
        "",
        "日期：2026-07-03",
        "",
        "本文档服务 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。本轮只把 UKF-vs-Hybrid repeat-expanded anchor comparison 的执行命令、分块、输出需求和 claim boundary 机器可读化；它不运行 benchmark，不新增统计结论，也不补硬件证据。",
        "",
        "## 生成文件",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_JSON.relative_to(ROOT)}`",
        "",
        "## Plan shape",
        "",
        f"- Formal-length Phase A rows: `{len(formal)}`",
        f"- Smoke-length feasibility rows: `{len(smoke)}`",
        "- Scenario unit: four predeclared scenario families reported separately.",
        "- Comparison unit: paired seed/repeat within scenario.",
        "- Mode subset: `ukf` versus `hybrid_residual_b` only.",
        "",
        "## Formal-length chunks",
        "",
        "| Scenario | Repeats | Chunks | Config | Boundary |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for scenario in SCENARIOS:
        rows_for_scenario = [row for row in formal if row["scenario"] == scenario]
        chunks = ", ".join(f"{row['repeat_start']}-{row['repeat_stop']}" for row in rows_for_scenario)
        lines.append(
            f"| `{scenario}` | 12 | `{chunks}` | `cnn_fpga/config/p4_multiscenario_strong_baselines.yaml` | planned only until completed and audited |"
        )
    lines.extend(
        [
            "",
            "## Smoke feasibility rows",
            "",
            "Smoke-length rows are allowed only to test command shape, missing-row accounting and collector logic. They must not be copied into the main performance claim.",
            "",
            "## 可写边界",
            "",
            "- 可以写：Phase A 的 repeat-expanded execution plan 已经有机器可读 command rows、scenario/mode/repeat units 和 post-run artifact requirements。",
            "- 可以写：正式统计升级必须等待 formal-length run 完成并进行 paired interval analysis。",
            "- 不能写：该 plan 证明了 robustness、statistical significance、hardware latency/resource、source-vs-board agreement 或 deployment readiness。",
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
