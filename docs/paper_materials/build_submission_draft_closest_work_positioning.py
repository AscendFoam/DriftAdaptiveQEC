"""Build closest-work positioning source data for the submission draft.

The rows summarize nearby literature families, the metric standard each family
sets, and the narrower distinction supported by the manuscript. This is a
literature-positioning table, not a new experiment or leaderboard.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_closest_work_positioning.csv"
JSON_PATH = OUT_DIR / "submission_draft_closest_work_positioning.json"
REPORT_PATH = OUT_DIR / "投稿稿closest_work_positioning记录.md"


ROWS = [
    {
        "row_id": "analog_surface_gkp",
        "closest_work_family": "Analog and surface-GKP decoding",
        "representative_metric_standard": "Squeezing thresholds, logical failure rates and overhead targets",
        "manuscript_distinction": "Recent analog histograms update the physical-layer affine surface rather than an outer-code matching graph or message-passing likelihood.",
        "current_evidence_boundary": "not an end-to-end surface-GKP threshold or overhead result",
        "citation_keys": "fukui2018; noh2020; noh2022; berent2024; borah2025",
        "source_anchor": "zotero_literature_review_cards.md analog GKP and surface-GKP entries",
    },
    {
        "row_id": "calibration_learned_decoders",
        "closest_work_family": "Calibration-aware and learned QEC decoders",
        "representative_metric_standard": "Real-device LER reductions, calibration snapshots, trained decoders and latency measurements",
        "manuscript_distinction": "The learned branch is a slow-loop estimator for a low-dimensional affine surface; the per-shot correction remains deterministic.",
        "current_evidence_boundary": "not a real-processor learned-decoder experiment",
        "citation_keys": "dgr2023; chen2022; sivak2024; bausch2024; stein2026",
        "source_anchor": "zotero_literature_review_cards.md calibration-aware and learned-decoder entries",
    },
    {
        "row_id": "logical_channel_gkp",
        "closest_work_family": "Finite-energy logical-channel analyses",
        "representative_metric_standard": "Logical-channel probabilities, channel infidelity and loss/amplification recovery curves",
        "manuscript_distinction": "The manuscript reports residual-boundary event decompositions and a finite-squeezing toy-channel sanity check as channel-language bridges.",
        "current_evidence_boundary": "not calibrated finite-energy logical-channel fidelity or process tomography",
        "citation_keys": "jafarzadeh2025; hastrup2023; zheng2024",
        "source_anchor": "zotero_literature_review_cards.md logical-channel fidelity and infidelity entries",
    },
    {
        "row_id": "runtime_predecoders",
        "closest_work_family": "Runtime pre-decoders and calibration-conditioned neural modules",
        "representative_metric_standard": "Accuracy-runtime trade-offs for per-shot or near-per-shot learned decoder modules",
        "manuscript_distinction": "The fast path is a clipped affine matrix-vector update; estimation is moved out of the latency-critical correction path.",
        "current_evidence_boundary": "not measured neural-decoder or FPGA runtime",
        "citation_keys": "chamberland2026; stein2026; yang2026",
        "source_anchor": "zotero_literature_review_cards.md learned runtime and FPGA neural-decoder entries",
    },
    {
        "row_id": "real_time_fpga_decoders",
        "closest_work_family": "Real-time FPGA and hardware-tailored decoders",
        "representative_metric_standard": "Closed-loop latency, cycle budget, resource use, memory, area, power and source-to-device evidence",
        "manuscript_distinction": "The present evidence only defines a compact, fixed-point-checkable affine target for a future source-vs-board validation.",
        "current_evidence_boundary": "not FPGA synthesis, resource use, power, real-board latency or source-vs-board agreement",
        "citation_keys": "lilliput2022; helios2023; qldpcfpga2025; caune2024; ziad2024; maurer2025; yang2026",
        "source_anchor": "zotero_literature_review_cards.md FPGA and real-time QEC entries",
    },
]


def validate_rows() -> None:
    seen = set()
    for row in ROWS:
        if row["row_id"] in seen:
            raise ValueError(f"duplicate row_id: {row['row_id']}")
        seen.add(row["row_id"])
        if not row["current_evidence_boundary"].startswith("not "):
            raise ValueError(f"boundary must start with 'not ': {row['row_id']}")
        if not row["citation_keys"] or not row["source_anchor"]:
            raise ValueError(f"missing citation/source anchor: {row['row_id']}")


def write_csv() -> None:
    fieldnames = [
        "row_id",
        "closest_work_family",
        "representative_metric_standard",
        "manuscript_distinction",
        "current_evidence_boundary",
        "citation_keys",
        "source_anchor",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)


def write_json() -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_closest_work_positioning_v1",
        "boundary": (
            "Closest-work positioning source table for the manuscript. It "
            "summarizes adjacent literature standards and the manuscript's "
            "narrower supported distinction; it is not a normalized leaderboard, "
            "not new experiment evidence, not a hardware result and not a claim "
            "that external metrics are reproduced by this manuscript."
        ),
        "rows": ROWS,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report() -> None:
    lines = [
        "# 投稿稿 closest-work positioning 记录",
        "",
        "日期：2026-07-06",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 中的最近邻工作定位表。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 边界",
        "",
        "- 可以写：本稿相对 analog GKP、calibration-aware learned decoders、logical-channel analyses 和 real-time FPGA decoders 的定位差异。",
        "- 不能写：这些外部指标是本稿复现实验结果，或本稿已完成 real-device learned-decoder、finite-energy tomography、FPGA synthesis、resource/power、real-board latency 或 source-vs-board validation。",
        "",
        "## 行摘要",
        "",
        "| Row | Boundary |",
        "| --- | --- |",
    ]
    for row in ROWS:
        lines.append(f"| `{row['row_id']}` | {row['current_evidence_boundary']} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    validate_rows()
    write_csv()
    write_json()
    write_report()
    print(json.dumps({"status": "ok", "rows": len(ROWS), "csv": str(CSV_PATH), "json": str(JSON_PATH)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
