from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "paper_materials"

SEQUENCE_CSV = OUT_DIR / "submission_draft_sequence_controlled_baseline_analysis.csv"
HOLDOUT_CSV = OUT_DIR / "submission_draft_holdout_drift_stress_analysis.csv"

CSV_PATH = OUT_DIR / "submission_draft_affine_local_validity_diagnostic.csv"
JSON_PATH = OUT_DIR / "submission_draft_affine_local_validity_diagnostic.json"
REPORT_PATH = OUT_DIR / "投稿稿affine_local_validity诊断记录.md"

SEQUENCE_SCENARIOS = (
    "static_bias_theta",
    "linear_ramp",
    "step_sigma_theta",
    "periodic_drift",
)
HOLDOUT_SCENARIOS = (
    "random_walk_drift",
    "burst_reset_drift",
    "faster_than_window_oscillation",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _percent_gain(fixed: float, oracle: float) -> float:
    if fixed == 0.0:
        return 0.0
    return (fixed - oracle) / fixed * 100.0


def _verdict(
    *,
    oracle_gain_percent: float,
    branch_risk_delta: float,
    lag_risk_delta: float | None,
) -> str:
    if lag_risk_delta is not None and lag_risk_delta > 0.004:
        return "stale_commit_can_erase_gain"
    if oracle_gain_percent > 3.0 and branch_risk_delta >= 0.0:
        return "local_affine_headroom_visible"
    if oracle_gain_percent >= 0.0 and branch_risk_delta >= 0.0:
        return "local_affine_not_dominated"
    return "requires_stronger_baseline_protocol"


def _sequence_rows() -> list[dict[str, str]]:
    source_rows = {
        (row["scenario"], row["method"]): row
        for row in _read_csv(SEQUENCE_CSV)
    }
    rows: list[dict[str, str]] = []
    for scenario in SEQUENCE_SCENARIOS:
        fixed = source_rows[(scenario, "fixed_affine")]
        oracle = source_rows[(scenario, "oracle_affine")]
        wrapped_mean = source_rows[(scenario, "wrapped_gaussian_posterior_mean")]
        fixed_mse = float(fixed["residual_mse"])
        oracle_mse = float(oracle["residual_mse"])
        oracle_gain_percent = _percent_gain(fixed_mse, oracle_mse)
        branch_risk_delta = (
            float(wrapped_mean["sequence_ler_proxy_mean"])
            - float(oracle["sequence_ler_proxy_mean"])
        )
        rows.append(
            {
                "surface": scenario,
                "evidence_layer": "short_sequence_controlled",
                "fixed_residual_mse": f"{fixed_mse:.12g}",
                "oracle_residual_mse": f"{oracle_mse:.12g}",
                "oracle_mse_gain_percent": f"{oracle_gain_percent:.6f}",
                "branch_risk_delta": f"{branch_risk_delta:.6f}",
                "lag_risk_delta": "",
                "validity_readout": _verdict(
                    oracle_gain_percent=oracle_gain_percent,
                    branch_risk_delta=branch_risk_delta,
                    lag_risk_delta=None,
                ),
                "non_claim_boundary": (
                    "not a formal nearest-lattice, known-noise or wrapped-decoder benchmark; "
                    "not a confidence interval; not hardware evidence"
                ),
            }
        )
    return rows


def _holdout_rows() -> list[dict[str, str]]:
    source_rows = {
        (row["scenario"], row["method"]): row
        for row in _read_csv(HOLDOUT_CSV)
    }
    rows: list[dict[str, str]] = []
    for scenario in HOLDOUT_SCENARIOS:
        fixed = source_rows[(scenario, "fixed_affine")]
        oracle = source_rows[(scenario, "oracle_affine")]
        lagged = source_rows[(scenario, "lagged_affine")]
        wrapped_mean = source_rows[(scenario, "wrapped_gaussian_posterior_mean")]
        fixed_mse = float(fixed["residual_mse"])
        oracle_mse = float(oracle["residual_mse"])
        oracle_gain_percent = _percent_gain(fixed_mse, oracle_mse)
        branch_risk_delta = float(wrapped_mean["residual_mse"]) - oracle_mse
        lag_risk_delta = float(lagged["residual_mse"]) - oracle_mse
        rows.append(
            {
                "surface": scenario,
                "evidence_layer": "holdout_stress_controlled",
                "fixed_residual_mse": f"{fixed_mse:.12g}",
                "oracle_residual_mse": f"{oracle_mse:.12g}",
                "oracle_mse_gain_percent": f"{oracle_gain_percent:.6f}",
                "branch_risk_delta": f"{branch_risk_delta:.6f}",
                "lag_risk_delta": f"{lag_risk_delta:.6f}",
                "validity_readout": _verdict(
                    oracle_gain_percent=oracle_gain_percent,
                    branch_risk_delta=branch_risk_delta,
                    lag_risk_delta=lag_risk_delta,
                ),
                "non_claim_boundary": (
                    "not trained-branch holdout generalization; not an expanded benchmark; "
                    "not a confidence interval; not hardware evidence"
                ),
            }
        )
    return rows


def _rows() -> list[dict[str, str]]:
    return _sequence_rows() + _holdout_rows()


def _write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "surface",
        "evidence_layer",
        "fixed_residual_mse",
        "oracle_residual_mse",
        "oracle_mse_gain_percent",
        "branch_risk_delta",
        "lag_risk_delta",
        "validity_readout",
        "non_claim_boundary",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict[str, str]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_affine_local_validity_diagnostic_v1",
        "source_files": [
            str(SEQUENCE_CSV.relative_to(ROOT)),
            str(HOLDOUT_CSV.relative_to(ROOT)),
        ],
        "scope": (
            "Derived diagnostic table that reads existing controlled sequence "
            "and holdout-stress CSVs through local-affine validity criteria."
        ),
        "formulas": {
            "oracle_mse_gain_percent": "(fixed_residual_mse - oracle_residual_mse) / fixed_residual_mse * 100",
            "branch_risk_delta_sequence": "wrapped_mean sequence_ler_proxy_mean - oracle sequence_ler_proxy_mean",
            "branch_risk_delta_holdout": "wrapped_mean residual_mse - oracle residual_mse",
            "lag_risk_delta": "lagged_affine residual_mse - oracle_affine residual_mse",
        },
        "non_claims": [
            "not a new formal benchmark",
            "not a confidence interval or p-value analysis",
            "not a tuned nearest-lattice or wrapped-decoder comparison",
            "not trained-branch holdout generalization",
            "not hardware evidence",
        ],
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, str]]) -> None:
    lines = [
        "# 投稿稿 affine local-validity 诊断记录",
        "",
        "日期：2026-07-06",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`，把已有 controlled sequence 和 holdout-stress source CSV 派生为一张局部仿射有效性诊断表。",
        "",
        "该表只回答一个审稿问题：在受控局部高斯设置和未见漂移压力设置中，oracle affine 是否仍有 MSE headroom，naive wrapped posterior 是否自动支配 affine path，以及 stale commit 是否会吞掉这种 headroom。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 结果摘要",
        "",
        "| Surface | Layer | Oracle gain (%) | Branch risk delta | Lag risk delta | Readout |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lag = row["lag_risk_delta"] or "NA"
        lines.append(
            f"| `{row['surface']}` | `{row['evidence_layer']}` | "
            f"{float(row['oracle_mse_gain_percent']):.2f} | "
            f"{float(row['branch_risk_delta']):.6f} | {lag} | "
            f"`{row['validity_readout']}` |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：该派生表把 oracle-affine MSE headroom、naive wrapped-posterior branch risk 和 stale-commit risk 放在同一个审稿可读框架中。",
            "- 可以写：它支持“affine fast path 有局部有效域且 commit policy 是方法的一部分”的受限解释。",
            "- 不能写：它补齐了正式 nearest-lattice / wrapped-decoder benchmark、CI/p-value、trained-branch holdout generalization、finite-energy logical-channel fidelity 或硬件证据。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _rows()
    _write_csv(rows)
    _write_json(rows)
    _write_report(rows)
    print(json.dumps({"status": "generated", "rows": len(rows), "csv": str(CSV_PATH), "json": str(JSON_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
