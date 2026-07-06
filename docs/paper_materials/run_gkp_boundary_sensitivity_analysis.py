from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_gkp_boundary_sensitivity.csv"
JSON_PATH = OUT_DIR / "submission_draft_gkp_boundary_sensitivity.json"
REPORT_PATH = OUT_DIR / "投稿稿gkp_boundary_sensitivity分析记录.md"

LAMBDA = math.sqrt(2.0 * math.pi)
HALF_BOUNDARY = LAMBDA / 2.0
SIGMA_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def _crossing_probability(sigma: float) -> float:
    return math.erfc(HALF_BOUNDARY / (math.sqrt(2.0) * sigma))


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sigma in SIGMA_VALUES:
        p_one = _crossing_probability(sigma)
        p_any = 1.0 - (1.0 - p_one) ** 2
        p_identity = 1.0 - p_any
        f_avg_surrogate = (1.0 + 2.0 * p_identity) / 3.0
        rows.append(
            {
                "effective_residual_sigma": sigma,
                "lambda": LAMBDA,
                "half_lattice_boundary": HALF_BOUNDARY,
                "equivalent_modular_squeezing_db": -10.0 * math.log10(sigma**2),
                "single_quadrature_crossing_probability": p_one,
                "any_qp_crossing_probability": p_any,
                "pauli_surrogate_identity_probability": p_identity,
                "pauli_surrogate_average_fidelity": f_avg_surrogate,
                "pauli_surrogate_infidelity": 1.0 - f_avg_surrogate,
                "boundary": "Analytical Gaussian residual-boundary sensitivity only; not finite-energy logical-channel simulation.",
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]]) -> None:
    fields = [
        "effective_residual_sigma",
        "lambda",
        "half_lattice_boundary",
        "equivalent_modular_squeezing_db",
        "single_quadrature_crossing_probability",
        "any_qp_crossing_probability",
        "pauli_surrogate_identity_probability",
        "pauli_surrogate_average_fidelity",
        "pauli_surrogate_infidelity",
        "boundary",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_json(rows: list[dict[str, object]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_gkp_boundary_sensitivity_v1",
        "formulae": {
            "lambda": "sqrt(2*pi)",
            "half_lattice_boundary": "lambda/2",
            "p_single_quadrature": "erfc((lambda/2)/(sqrt(2)*sigma)) for zero-mean Gaussian residual",
            "p_any_qp": "1 - (1 - p_single_quadrature)^2 assuming independent q/p residuals with equal sigma",
            "f_avg_surrogate": "(1 + 2*p_identity)/3 with p_identity = 1 - p_any_qp",
        },
        "boundary": (
            "Analytical bridge from residual scale to half-lattice crossing probability. "
            "Not a finite-energy GKP logical-channel simulation, not process tomography, "
            "not a hardware result, and not a benchmark rerun."
        ),
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# 投稿稿 GKP boundary-sensitivity 分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它用 zero-mean Gaussian residual 的 half-lattice crossing 公式，连接 residual scale、boundary-crossing probability 和 Pauli-channel-style surrogate fidelity 语言。",
        "",
        "它不是 finite-energy GKP logical-channel simulation，不是 process tomography，不是硬件测量，也不重跑 benchmark。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 结果摘要",
        "",
        "| sigma | squeezing dB | one-quadrature crossing | any q/p crossing | surrogate infidelity |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['effective_residual_sigma']):.2f} | "
            f"{float(row['equivalent_modular_squeezing_db']):.2f} | "
            f"{float(row['single_quadrature_crossing_probability']):.6f} | "
            f"{float(row['any_qp_crossing_probability']):.6f} | "
            f"{float(row['pauli_surrogate_infidelity']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：在 Gaussian residual approximation 下，residual scale 接近 half-lattice boundary 时 crossing probability 和 surrogate infidelity 会迅速上升。",
            "- 可以写：该表解释为什么降低 residual MSE 与降低 \\\\finalLER{} proxy 相关，但二者不是同一个指标。",
            "- 不能写：该表估计了 finite-energy logical-channel fidelity、process fidelity、outer-code LER 或硬件 logical error rate。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _rows()
    _write_csv(rows)
    _write_json(rows)
    _write_report(rows)
    print(json.dumps({"status": "generated", "csv": str(CSV_PATH), "json": str(JSON_PATH), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
