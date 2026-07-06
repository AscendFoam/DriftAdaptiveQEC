from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_fast_path_cost_model.csv"
JSON_PATH = OUT_DIR / "submission_draft_fast_path_cost_model.json"
REPORT_PATH = OUT_DIR / "投稿稿fast_path_cost模型记录.md"


ROWS = [
    {
        "decoder": "affine_fast_path",
        "table_label": "Affine fast path",
        "branch_candidates": 1,
        "multiplications_per_shot": 4,
        "additions_per_shot": 4,
        "comparisons_per_shot": 4,
        "exp_per_shot": 0,
        "divisions_per_shot": 0,
        "stored_state_scalars": 6,
        "nonlinear_ops_per_shot": 0,
        "critical_path_class": "single 2x2 matvec plus clip",
        "scope": "analytical count for the manuscript fast-path equation",
    },
    {
        "decoder": "wrapped_gaussian_map_3x3",
        "table_label": "Wrapped MAP, 3x3 branches",
        "branch_candidates": 9,
        "multiplications_per_shot": 49,
        "additions_per_shot": 40,
        "comparisons_per_shot": 12,
        "exp_per_shot": 0,
        "divisions_per_shot": 0,
        "stored_state_scalars": 33,
        "nonlinear_ops_per_shot": 0,
        "critical_path_class": "nine branch scores plus argmin",
        "scope": "one-step known-state wrapped-Gaussian MAP reference",
    },
    {
        "decoder": "wrapped_gaussian_posterior_mean_3x3",
        "table_label": "Wrapped posterior mean, 3x3 branches",
        "branch_candidates": 9,
        "multiplications_per_shot": 99,
        "additions_per_shot": 98,
        "comparisons_per_shot": 4,
        "exp_per_shot": 9,
        "divisions_per_shot": 9,
        "stored_state_scalars": 33,
        "nonlinear_ops_per_shot": 18,
        "critical_path_class": "nine branch scores, softmax and weighted sum",
        "scope": "one-step known-state wrapped-Gaussian posterior-mean reference",
    },
]


def _enrich_rows() -> list[dict[str, object]]:
    affine = ROWS[0]
    affine_mult = float(affine["multiplications_per_shot"])
    affine_add = float(affine["additions_per_shot"])
    affine_state = float(affine["stored_state_scalars"])
    enriched: list[dict[str, object]] = []
    for row in ROWS:
        item = dict(row)
        item["relative_mult_vs_affine"] = float(item["multiplications_per_shot"]) / affine_mult
        item["relative_add_vs_affine"] = float(item["additions_per_shot"]) / affine_add
        item["relative_state_vs_affine"] = float(item["stored_state_scalars"]) / affine_state
        item["arithmetic_ops_per_shot"] = int(item["multiplications_per_shot"]) + int(item["additions_per_shot"])
        enriched.append(item)
    return enriched


def _write_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "decoder",
        "table_label",
        "branch_candidates",
        "multiplications_per_shot",
        "additions_per_shot",
        "arithmetic_ops_per_shot",
        "comparisons_per_shot",
        "exp_per_shot",
        "divisions_per_shot",
        "nonlinear_ops_per_shot",
        "stored_state_scalars",
        "relative_mult_vs_affine",
        "relative_add_vs_affine",
        "relative_state_vs_affine",
        "critical_path_class",
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
        "analysis_id": "submission_draft_fast_path_cost_model_v1",
        "boundary": (
            "Analytical per-shot operation-count model for manuscript positioning. "
            "Not FPGA synthesis, not timing closure, not power/resource measurement, "
            "and not a hardware result."
        ),
        "counting_assumptions": {
            "affine_fast_path": "2x2 matrix-vector product, two bias additions, clipping comparisons; active parameters are treated as already staged.",
            "wrapped_gaussian_map_3x3": "Nine lattice branches, symmetric 2D Mahalanobis score per branch, one selected posterior mean after argmin.",
            "wrapped_gaussian_posterior_mean_3x3": "Nine lattice branches, branch scores, softmax normalization and weighted posterior mean.",
        },
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# 投稿稿 fast-path cost 模型记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它给出 per-shot analytical operation-count，用于支撑稿件中 deterministic affine fast path 的低复杂度定位。",
        "",
        "它不是 FPGA synthesis、不是 timing closure、不是 power/resource 测量，也不是 hardware result。所有真实硬件 latency/resource 仍必须由后续板级实验给出。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 结果摘要",
        "",
        "| Decoder | Branches | Mult. | Add. | Nonlinear ops | Stored scalars |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['table_label']} | {row['branch_candidates']} | "
            f"{row['multiplications_per_shot']} | {row['additions_per_shot']} | "
            f"{row['nonlinear_ops_per_shot']} | {row['stored_state_scalars']} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：affine fast path 的 per-shot arithmetic 和 state footprint 明显小于 3x3 branch wrapped-Gaussian posterior references。",
            "- 可以写：该表支持低延迟/低资源的工程动机，但不等于真实 FPGA timing 或 resource measurement。",
            "- 不能写：已经完成 hardware latency、LUT/FF/DSP/BRAM、power、source-vs-board agreement 或 timing closure。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _enrich_rows()
    _write_csv(rows)
    _write_json(rows)
    _write_report(rows)
    print(json.dumps({"status": "generated", "csv": str(CSV_PATH), "json": str(JSON_PATH), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
