from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn_fpga.decoder.param_mapper import NoisePrediction, ParamMapper  # noqa: E402
from physics.constants import LATTICE_CONST  # noqa: E402
from docs.paper_materials.run_holdout_drift_stress_analysis import (  # noqa: E402
    CONFIG,
    N_SEQUENCES,
    N_STEPS,
    SCENARIOS,
    _affine_corrections,
    _prediction,
    _sample_errors,
    _sample_syndromes,
    _state_rows,
)


OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_commit_lag_sweep_analysis.csv"
JSON_PATH = OUT_DIR / "submission_draft_commit_lag_sweep_analysis.json"
REPORT_PATH = OUT_DIR / "投稿稿commit_lag_sweep分析记录.md"

SEED = 20260703 + 73
COMMIT_INTERVAL_STEPS = 64
COMMIT_LAG_STEPS = (0, 8, 16, 32, 64, 128)


def _fixed_prediction() -> NoisePrediction:
    return NoisePrediction(
        sigma=0.22,
        mu_q=0.0,
        mu_p=0.0,
        theta_deg=0.0,
        source="fixed_nominal",
        metadata={"commit_lag_sweep_analysis": True},
    )


def _lagged_state_index(step: int, lag_steps: int) -> int:
    if step <= lag_steps:
        return 0
    visible_step = step - lag_steps
    return (visible_step // COMMIT_INTERVAL_STEPS) * COMMIT_INTERVAL_STEPS


def _summarize(
    *,
    scenario: str,
    lag_steps: int,
    residuals: np.ndarray,
    oracle_residuals: np.ndarray,
    fixed_residuals: np.ndarray,
) -> dict[str, float | str | int]:
    boundary = LATTICE_CONST / 2.0
    any_cross = np.logical_or(
        np.abs(residuals[:, :, 0]) > boundary,
        np.abs(residuals[:, :, 1]) > boundary,
    )
    oracle_any_cross = np.logical_or(
        np.abs(oracle_residuals[:, :, 0]) > boundary,
        np.abs(oracle_residuals[:, :, 1]) > boundary,
    )
    fixed_any_cross = np.logical_or(
        np.abs(fixed_residuals[:, :, 0]) > boundary,
        np.abs(fixed_residuals[:, :, 1]) > boundary,
    )
    residual_mse = float(np.mean(np.sum(residuals**2, axis=2)))
    oracle_mse = float(np.mean(np.sum(oracle_residuals**2, axis=2)))
    fixed_mse = float(np.mean(np.sum(fixed_residuals**2, axis=2)))
    if lag_steps == 0:
        lag_class = "commit_interval_only"
    elif lag_steps <= 16:
        lag_class = "short_lag"
    elif lag_steps <= 64:
        lag_class = "moderate_lag"
    else:
        lag_class = "long_lag"
    return {
        "scenario": scenario,
        "method": "lagged_affine",
        "n_sequences": N_SEQUENCES,
        "n_steps": N_STEPS,
        "commit_interval_steps": COMMIT_INTERVAL_STEPS,
        "commit_lag_steps": lag_steps,
        "lag_class": lag_class,
        "residual_mse": residual_mse,
        "oracle_residual_mse": oracle_mse,
        "fixed_residual_mse": fixed_mse,
        "mse_delta_vs_oracle_affine": residual_mse - oracle_mse,
        "mse_delta_vs_fixed_affine": residual_mse - fixed_mse,
        "relative_mse_penalty_vs_oracle_percent": (
            100.0 * (residual_mse - oracle_mse) / oracle_mse
        ),
        "relative_mse_delta_vs_fixed_percent": (
            100.0 * (residual_mse - fixed_mse) / fixed_mse
        ),
        "any_crossing_rate": float(np.mean(any_cross)),
        "crossing_delta_vs_oracle_affine": float(np.mean(any_cross) - np.mean(oracle_any_cross)),
        "crossing_delta_vs_fixed_affine": float(np.mean(any_cross) - np.mean(fixed_any_cross)),
        "non_claim_boundary": (
            "Controlled non-hardware commit-lag sweep in simulation steps; "
            "not measured FPGA latency, timing closure, source-vs-board agreement, "
            "trained-branch holdout generalization, CI/p-value or logical-channel fidelity."
        ),
    }


def _run_scenario(scenario: str, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    mapper = ParamMapper.from_config(CONFIG)
    fixed_params = mapper.map_prediction(_fixed_prediction())
    state_rows = _state_rows(scenario)
    oracle_params_by_step = [mapper.map_prediction(_prediction(state, "oracle_affine")) for state in state_rows]

    errors_by_step = []
    syndromes_by_step = []
    for state in state_rows:
        errors = _sample_errors(state, rng)
        errors_by_step.append(errors)
        syndromes_by_step.append(_sample_syndromes(errors, rng))

    oracle_residuals = np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float)
    fixed_residuals = np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float)
    for step in range(N_STEPS):
        oracle_residuals[:, step, :] = errors_by_step[step] - _affine_corrections(
            syndromes_by_step[step], oracle_params_by_step[step]
        )
        fixed_residuals[:, step, :] = errors_by_step[step] - _affine_corrections(
            syndromes_by_step[step], fixed_params
        )

    rows: list[dict[str, float | str | int]] = []
    for lag_steps in COMMIT_LAG_STEPS:
        residuals = np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float)
        for step in range(N_STEPS):
            lagged_idx = _lagged_state_index(step, lag_steps)
            residuals[:, step, :] = errors_by_step[step] - _affine_corrections(
                syndromes_by_step[step], oracle_params_by_step[lagged_idx]
            )
        rows.append(
            _summarize(
                scenario=scenario,
                lag_steps=lag_steps,
                residuals=residuals,
                oracle_residuals=oracle_residuals,
                fixed_residuals=fixed_residuals,
            )
        )
    return rows


def _rows() -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str | int]] = []
    for scenario in SCENARIOS:
        rows.extend(_run_scenario(scenario, rng))
    return rows


def _write_csv(rows: list[dict[str, float | str | int]]) -> None:
    fields = [
        "scenario",
        "method",
        "n_sequences",
        "n_steps",
        "commit_interval_steps",
        "commit_lag_steps",
        "lag_class",
        "residual_mse",
        "oracle_residual_mse",
        "fixed_residual_mse",
        "mse_delta_vs_oracle_affine",
        "mse_delta_vs_fixed_affine",
        "relative_mse_penalty_vs_oracle_percent",
        "relative_mse_delta_vs_fixed_percent",
        "any_crossing_rate",
        "crossing_delta_vs_oracle_affine",
        "crossing_delta_vs_fixed_affine",
        "non_claim_boundary",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(rows: list[dict[str, float | str | int]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_commit_lag_sweep_v1",
        "seed": SEED,
        "n_sequences": N_SEQUENCES,
        "n_steps": N_STEPS,
        "commit_interval_steps": COMMIT_INTERVAL_STEPS,
        "commit_lag_steps": list(COMMIT_LAG_STEPS),
        "scenarios": list(SCENARIOS),
        "boundary": (
            "Controlled non-hardware stale-parameter/commit-lag sweep in simulation steps. "
            "The sweep reuses the holdout drift stress families and shared sampled errors per scenario. "
            "It does not measure FPGA latency, timing closure, source-vs-board agreement, trained-branch "
            "holdout generalization, inferential uncertainty or finite-energy logical-channel fidelity."
        ),
        "config": CONFIG,
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, float | str | int]]) -> None:
    lines = [
        "# 投稿稿 commit-lag sweep 分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它在三类 holdout drift stress family 上扫描 lagged affine 的 commit lag，用于把 stale-parameter/commit-latency 风险从叙述性缺口转化为可复核的仿真诊断。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 协议",
        "",
        f"- random seed: `{SEED}`",
        f"- sequences per scenario: `{N_SEQUENCES}`",
        f"- steps per sequence: `{N_STEPS}`",
        f"- fixed commit interval: `{COMMIT_INTERVAL_STEPS}` simulation steps",
        f"- swept commit lag: `{', '.join(str(x) for x in COMMIT_LAG_STEPS)}` simulation steps",
        "- baseline references inside each scenario: oracle affine and fixed nominal affine",
        "- metric: residual MSE and rare half-lattice residual-boundary crossing proxy",
        "",
        "## 结果摘要",
        "",
        "| Scenario | Lag 0 | Lag 8 | Lag 16 | Lag 32 | Lag 64 | Lag 128 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in SCENARIOS:
        selected = {
            int(row["commit_lag_steps"]): row
            for row in rows
            if row["scenario"] == scenario
        }
        lines.append(
            f"| {scenario} | "
            f"{float(selected[0]['residual_mse']):.6f} | "
            f"{float(selected[8]['residual_mse']):.6f} | "
            f"{float(selected[16]['residual_mse']):.6f} | "
            f"{float(selected[32]['residual_mse']):.6f} | "
            f"{float(selected[64]['residual_mse']):.6f} | "
            f"{float(selected[128]['residual_mse']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：该 sweep 将 slow-loop stale parameter 风险变成了 simulation-step 级可审计变量。",
            "- 可以写：短 lag 在 random-walk stress 中仍接近 oracle，但 burst/reset 与 faster-than-window family 对 lag 更敏感。",
            "- 不能写：该 sweep 测得了 FPGA/board latency、timing closure、source-vs-board agreement、trained CNN branch holdout generalization、CI/p-value 或 logical-channel fidelity。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _rows()
    _write_csv(rows)
    _write_json(rows)
    _write_report(rows)
    print(f"Wrote {CSV_PATH.relative_to(ROOT)}")
    print(f"Wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
