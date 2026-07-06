from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.paper_materials.run_logical_channel_surrogate_analysis import (  # noqa: E402
    JSON_PATH as SOURCE_JSON_PATH,
    CSV_PATH as SOURCE_CSV_PATH,
    N_SAMPLES,
    SEED,
    _rows as logical_channel_rows,
)


OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_lattice_logical_channel_sanity.csv"
JSON_PATH = OUT_DIR / "submission_draft_lattice_logical_channel_sanity.json"
REPORT_PATH = OUT_DIR / "投稿稿lattice_logical_channel_sanity记录.md"

METHOD_ORDER = (
    "fixed_affine",
    "oracle_affine",
    "wrapped_gaussian_posterior_mean",
    "wrapped_gaussian_map",
)

METHOD_LABELS = {
    "fixed_affine": "Fixed affine",
    "oracle_affine": "Oracle affine",
    "wrapped_gaussian_posterior_mean": "Wrapped mean",
    "wrapped_gaussian_map": "Wrapped MAP",
}


def _as_float(row: dict[str, object], key: str) -> float:
    return float(row[key])


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def build_rows() -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in logical_channel_rows():
        method = str(row["method"])
        if method in METHOD_ORDER:
            grouped.setdefault(method, []).append(row)

    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        items = grouped.get(method, [])
        if not items:
            raise ValueError(f"missing logical-channel surrogate rows for {method}")

        p_any = [_as_float(row, "pauli_surrogate_any_crossing") for row in items]
        f_avg = [_as_float(row, "pauli_surrogate_average_fidelity") for row in items]
        qp_asymmetry = [
            abs(
                _as_float(row, "source_q_boundary_crossing_rate")
                - _as_float(row, "source_p_boundary_crossing_rate")
            )
            for row in items
        ]
        worst = max(items, key=lambda row: _as_float(row, "pauli_surrogate_any_crossing"))

        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n_controlled_states": len(items),
                "n_samples_per_state": N_SAMPLES,
                "mean_p_any": _mean(p_any),
                "worst_state": worst["scenario"],
                "worst_state_p_any": _as_float(worst, "pauli_surrogate_any_crossing"),
                "mean_f_avg_surr": _mean(f_avg),
                "worst_state_f_avg_surr": min(f_avg),
                "max_qp_asymmetry": max(qp_asymmetry),
                "source_surrogate_csv": str(SOURCE_CSV_PATH.relative_to(ROOT)),
                "non_claim_boundary": (
                    "not finite-energy GKP logical-channel fidelity; not process "
                    "tomography; not hardware fidelity; not an outer-code logical "
                    "error estimate"
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "method_label",
        "n_controlled_states",
        "n_samples_per_state",
        "mean_p_any",
        "worst_state",
        "worst_state_p_any",
        "mean_f_avg_surr",
        "worst_state_f_avg_surr",
        "max_qp_asymmetry",
        "source_surrogate_csv",
        "non_claim_boundary",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, object]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_lattice_logical_channel_sanity_v1",
        "seed": SEED,
        "source_analysis": "controlled_residual_boundary_pauli_surrogate_v1",
        "source_csv": str(SOURCE_CSV_PATH.relative_to(ROOT)),
        "source_json": str(SOURCE_JSON_PATH.relative_to(ROOT)),
        "scope": (
            "Method-level aggregation over the controlled residual-boundary "
            "Pauli-style surrogate rows. The output reports mean and worst-state "
            "p_any plus the corresponding surrogate average-fidelity readout."
        ),
        "non_claims": [
            "not finite-energy GKP logical-channel fidelity",
            "not process tomography",
            "not hardware fidelity",
            "not an outer-code logical-error estimate",
            "not a replacement for expanded benchmark evidence",
        ],
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# 投稿稿 lattice logical-channel sanity 记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它只把已有 residual-boundary Pauli-style surrogate 按方法聚合成 lattice-level sanity summary，用于给审稿人一个更清楚的 `p_any` / `F_avg^surr` 读数入口。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 结果摘要",
        "",
        "| Method | Mean p_any | Worst state | Worst p_any | Mean F_avg^surr | Worst F_avg^surr |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | "
            f"{float(row['mean_p_any']):.6f} | "
            f"`{row['worst_state']}` | "
            f"{float(row['worst_state_p_any']):.6f} | "
            f"{float(row['mean_f_avg_surr']):.6f} | "
            f"{float(row['worst_state_f_avg_surr']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：该表把四个 controlled local-Gaussian states 上的 residual-boundary surrogate 聚合成方法级 mean / worst-state sanity summary。",
            "- 可以写：`F_avg^surr` 只是由 `p_I^surr=1-p_any` 推出的 Pauli-style surrogate readout，用于审稿可追溯性。",
            "- 不能写：该表完成了 finite-energy GKP logical-channel fidelity、process tomography、hardware fidelity、outer-code logical-error estimate 或正式 expanded benchmark。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_csv(rows)
    write_json(rows)
    write_report(rows)
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
