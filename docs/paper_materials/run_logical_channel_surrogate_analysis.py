from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.paper_materials.run_controlled_oracle_affine_analysis import (  # noqa: E402
    CONFIG,
    N_SAMPLES,
    SCENARIOS,
    SEED,
    _controlled_rows,
)


OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_logical_channel_surrogate_analysis.csv"
JSON_PATH = OUT_DIR / "submission_draft_logical_channel_surrogate_analysis.json"
REPORT_PATH = OUT_DIR / "投稿稿logical_channel_surrogate分析记录.md"

METHODS = (
    "fixed_affine",
    "oracle_affine",
    "wrapped_gaussian_posterior_mean",
    "wrapped_gaussian_map",
)


def _as_float(row: dict[str, float | str | int], key: str) -> float:
    return float(row[key])


def _surrogate_row(row: dict[str, float | str | int]) -> dict[str, float | str | int]:
    q_cross = _as_float(row, "q_boundary_crossing_rate")
    p_cross = _as_float(row, "p_boundary_crossing_rate")
    any_cross = _as_float(row, "boundary_crossing_rate")
    both_cross = max(0.0, min(q_cross, p_cross, q_cross + p_cross - any_cross))
    q_only = max(0.0, q_cross - both_cross)
    p_only = max(0.0, p_cross - both_cross)
    identity = max(0.0, 1.0 - any_cross)

    # For a qubit Pauli channel with identity probability p_I, the average
    # fidelity to the identity channel is (1 + 2 p_I) / 3.  Here this is only a
    # residual-boundary surrogate, not logical-channel tomography.
    average_fidelity_surrogate = (1.0 + 2.0 * identity) / 3.0

    return {
        "scenario": row["scenario"],
        "scenario_label": row["scenario_label"],
        "method": row["method"],
        "n_samples": row["n_samples"],
        "pauli_surrogate_identity": identity,
        "pauli_surrogate_q_only": q_only,
        "pauli_surrogate_p_only": p_only,
        "pauli_surrogate_both": both_cross,
        "pauli_surrogate_any_crossing": any_cross,
        "pauli_surrogate_average_fidelity": average_fidelity_surrogate,
        "source_q_boundary_crossing_rate": q_cross,
        "source_p_boundary_crossing_rate": p_cross,
        "source_residual_mse": row["residual_mse"],
    }


def _rows() -> list[dict[str, float | str | int]]:
    controlled = [row for row in _controlled_rows() if row["method"] in METHODS]
    return [_surrogate_row(row) for row in controlled]


def _write_csv(rows: list[dict[str, float | str | int]]) -> None:
    fields = [
        "scenario",
        "scenario_label",
        "method",
        "n_samples",
        "pauli_surrogate_identity",
        "pauli_surrogate_q_only",
        "pauli_surrogate_p_only",
        "pauli_surrogate_both",
        "pauli_surrogate_any_crossing",
        "pauli_surrogate_average_fidelity",
        "source_q_boundary_crossing_rate",
        "source_p_boundary_crossing_rate",
        "source_residual_mse",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(rows: list[dict[str, float | str | int]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "controlled_residual_boundary_pauli_surrogate_v1",
        "seed": SEED,
        "n_samples_per_scenario": N_SAMPLES,
        "source_analysis": "submission_draft_controlled_oracle_affine_analysis",
        "source_csv_expected": str(OUT_DIR / "submission_draft_controlled_oracle_affine_analysis.csv"),
        "boundary": (
            "Controlled residual-boundary Pauli-channel-style surrogate derived "
            "from q/p half-lattice crossing rates. It is a manuscript bridge "
            "between final_ler-style residual-boundary metrics and channel "
            "language, not finite-energy GKP logical-channel tomography, not "
            "process fidelity, not a hardware measurement and not a formal "
            "benchmark upgrade."
        ),
        "formula": {
            "p_identity_surrogate": "1 - p_any_crossing",
            "p_q_only_surrogate": "p_q_cross - p_both_cross",
            "p_p_only_surrogate": "p_p_cross - p_both_cross",
            "p_both_surrogate": "p_q_cross + p_p_cross - p_any_crossing",
            "average_fidelity_surrogate": "(1 + 2 * p_identity_surrogate) / 3",
        },
        "config": CONFIG,
        "scenarios": SCENARIOS,
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, float | str | int]]) -> None:
    by_scenario: dict[str, dict[str, dict[str, float | str | int]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), {})[str(row["method"])] = row

    lines = [
        "# 投稿稿 logical-channel surrogate 分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它从受控 local-Gaussian 样本的 q/p residual half-lattice crossing 率构造一个 Pauli-channel-style surrogate，用于把 `final_ler` 类型的 residual-boundary proxy 和 channel 语言之间的关系讲清楚。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 口径",
        "",
        f"- random seed: `{SEED}`",
        f"- samples per scenario: `{N_SAMPLES}`",
        "- 输入：受控 oracle-affine / wrapped-Gaussian local-Gaussian 分析中的 q/p boundary crossing rate。",
        "- 分解：`p_I=1-p_any`、`p_q_only=p_q-p_both`、`p_p_only=p_p-p_both`、`p_both=p_q+p_p-p_any`。",
        "- 可选解释量：若把该分解仅当作 Pauli-channel-style surrogate，则 `F_avg_surr=(1+2 p_I)/3`。",
        "",
        "## 结果摘要",
        "",
        "| Scenario | Fixed p_any | Oracle p_any | Wrapped mean p_any | Wrapped MAP p_any | Oracle surrogate average fidelity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario, item_by_method in by_scenario.items():
        fixed = item_by_method["fixed_affine"]
        oracle = item_by_method["oracle_affine"]
        wrapped_mean = item_by_method["wrapped_gaussian_posterior_mean"]
        wrapped_map = item_by_method["wrapped_gaussian_map"]
        lines.append(
            f"| {scenario} | "
            f"{float(fixed['pauli_surrogate_any_crossing']):.6f} | "
            f"{float(oracle['pauli_surrogate_any_crossing']):.6f} | "
            f"{float(wrapped_mean['pauli_surrogate_any_crossing']):.6f} | "
            f"{float(wrapped_map['pauli_surrogate_any_crossing']):.6f} | "
            f"{float(oracle['pauli_surrogate_average_fidelity']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：该 surrogate 把 residual-boundary crossing 显式拆成 q-only、p-only、both 和 identity-like 分量，使 `final_ler` proxy 与 channel 语言之间的关系更透明。",
            "- 可以写：surrogate average fidelity 只是在 Pauli-channel-style surrogate 下由 identity-like 分量诱导的解释量，不能与有限能量 GKP logical-channel fidelity 等同。",
            "- 不能写：本分析完成了 logical-channel tomography、process fidelity estimation、finite-energy GKP channel simulation、硬件保真度测量或统计显著性证明。",
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
