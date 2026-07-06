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
from docs.paper_materials.run_sequence_controlled_baseline_analysis import (  # noqa: E402
    CONFIG,
    _measurement_std,
    _scenario_covariance,
    _wrap,
    _wrapped_gaussian_corrections,
)


OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_holdout_drift_stress_analysis.csv"
JSON_PATH = OUT_DIR / "submission_draft_holdout_drift_stress_analysis.json"
REPORT_PATH = OUT_DIR / "投稿稿holdout_drift_stress分析记录.md"

SEED = 20260703 + 41
N_SEQUENCES = 384
N_STEPS = 512
COMMIT_INTERVAL_STEPS = 64
COMMIT_LAG_STEPS = 32

SCENARIOS = (
    "random_walk_drift",
    "burst_reset_drift",
    "faster_than_window_oscillation",
)

METHODS = (
    "fixed_affine",
    "lagged_affine",
    "oracle_affine",
    "wrapped_gaussian_posterior_mean",
    "wrapped_gaussian_map",
)


def _clip_state(row: dict[str, float | str], scenario: str) -> dict[str, float | str]:
    return {
        "scenario": scenario,
        "sigma": float(np.clip(float(row["sigma"]), 0.16, 0.46)),
        "mu_q": float(np.clip(float(row["mu_q"]), -0.070, 0.070)),
        "mu_p": float(np.clip(float(row["mu_p"]), -0.070, 0.070)),
        "theta_deg": float(np.clip(float(row["theta_deg"]), -16.0, 16.0)),
    }


def _state_rows(scenario: str) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(SEED + 17 * (SCENARIOS.index(scenario) + 1))
    if scenario == "random_walk_drift":
        rows: list[dict[str, float | str]] = []
        sigma = 0.25
        mu_q = 0.0
        mu_p = 0.0
        theta = 0.0
        for _ in range(N_STEPS):
            sigma += float(rng.normal(0.0, 0.006))
            mu_q += float(rng.normal(0.0, 0.0025))
            mu_p += float(rng.normal(0.0, 0.0025))
            theta += float(rng.normal(0.0, 0.45))
            rows.append(
                _clip_state(
                    {
                        "sigma": sigma,
                        "mu_q": mu_q,
                        "mu_p": mu_p,
                        "theta_deg": theta,
                    },
                    scenario,
                )
            )
        return rows

    if scenario == "burst_reset_drift":
        rows = []
        for step in range(N_STEPS):
            in_first_burst = 150 <= step < 205
            in_second_burst = 340 <= step < 380
            if in_first_burst:
                raw = {
                    "sigma": 0.41,
                    "mu_q": 0.060,
                    "mu_p": -0.050,
                    "theta_deg": 13.0,
                }
            elif in_second_burst:
                raw = {
                    "sigma": 0.38,
                    "mu_q": -0.055,
                    "mu_p": 0.045,
                    "theta_deg": -11.0,
                }
            else:
                reset_phase = 0.5 + 0.5 * np.sin(2.0 * np.pi * step / N_STEPS)
                raw = {
                    "sigma": 0.22 + 0.025 * reset_phase,
                    "mu_q": 0.012 * np.sin(2.0 * np.pi * step / 160.0),
                    "mu_p": -0.010 * np.sin(2.0 * np.pi * step / 160.0),
                    "theta_deg": 3.0 * np.sin(2.0 * np.pi * step / 192.0),
                }
            rows.append(_clip_state(raw, scenario))
        return rows

    if scenario == "faster_than_window_oscillation":
        rows = []
        for step in range(N_STEPS):
            fast = np.sin(2.0 * np.pi * step / 36.0)
            quadrature = np.sin(2.0 * np.pi * step / 54.0 + np.pi / 5.0)
            rows.append(
                _clip_state(
                    {
                        "sigma": 0.27 + 0.075 * fast,
                        "mu_q": 0.040 * fast,
                        "mu_p": -0.035 * quadrature,
                        "theta_deg": 12.0 * np.sin(2.0 * np.pi * step / 48.0),
                    },
                    scenario,
                )
            )
        return rows

    raise ValueError(f"Unknown holdout scenario: {scenario}")


def _prediction(row: dict[str, float | str], source: str) -> NoisePrediction:
    return NoisePrediction(
        sigma=float(row["sigma"]),
        mu_q=float(row["mu_q"]),
        mu_p=float(row["mu_p"]),
        theta_deg=float(row["theta_deg"]),
        source=source,
        metadata={"holdout_drift_stress_analysis": True},
    )


def _sample_errors(row: dict[str, float | str], rng: np.random.Generator) -> np.ndarray:
    covariance = _scenario_covariance(row)
    mean = np.array([float(row["mu_q"]), float(row["mu_p"])], dtype=float)
    return rng.multivariate_normal(mean, covariance, size=N_SEQUENCES)


def _sample_syndromes(errors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _wrap(errors) + rng.normal(0.0, _measurement_std(), size=errors.shape)


def _affine_corrections(syndromes: np.ndarray, params) -> np.ndarray:
    fast_loop = CONFIG["fast_loop"]
    syndrome_limit = float(fast_loop["syndrome_limit"])
    correction_limit = float(fast_loop["correction_limit"])
    syndrome_used = np.clip(syndromes, -syndrome_limit, syndrome_limit)
    correction = syndrome_used @ params.K.T + params.b
    return np.clip(correction, -correction_limit, correction_limit)


def _lagged_state_index(step: int) -> int:
    if step <= COMMIT_LAG_STEPS:
        return 0
    visible_step = step - COMMIT_LAG_STEPS
    return (visible_step // COMMIT_INTERVAL_STEPS) * COMMIT_INTERVAL_STEPS


def _summarize(
    *,
    scenario: str,
    method: str,
    residuals: np.ndarray,
    oracle_residuals: np.ndarray | None,
) -> dict[str, float | str | int]:
    boundary = LATTICE_CONST / 2.0
    q_cross = np.abs(residuals[:, :, 0]) > boundary
    p_cross = np.abs(residuals[:, :, 1]) > boundary
    any_cross = np.logical_or(q_cross, p_cross)
    per_sequence = np.mean(any_cross, axis=1)
    residual_mse = float(np.mean(np.sum(residuals**2, axis=2)))
    identity_surrogate = float(1.0 - np.mean(any_cross))
    average_fidelity_surrogate = float((1.0 + 2.0 * identity_surrogate) / 3.0)
    if oracle_residuals is None:
        mse_delta_vs_oracle = 0.0
        crossing_delta_vs_oracle = 0.0
    else:
        oracle_any = np.logical_or(
            np.abs(oracle_residuals[:, :, 0]) > boundary,
            np.abs(oracle_residuals[:, :, 1]) > boundary,
        )
        mse_delta_vs_oracle = residual_mse - float(np.mean(np.sum(oracle_residuals**2, axis=2)))
        crossing_delta_vs_oracle = float(np.mean(any_cross) - np.mean(oracle_any))
    return {
        "scenario": scenario,
        "method": method,
        "n_sequences": N_SEQUENCES,
        "n_steps": N_STEPS,
        "commit_interval_steps": COMMIT_INTERVAL_STEPS if method == "lagged_affine" else "",
        "commit_lag_steps": COMMIT_LAG_STEPS if method == "lagged_affine" else "",
        "sequence_ler_proxy_mean": float(np.mean(per_sequence)),
        "sequence_ler_proxy_sd": float(np.std(per_sequence, ddof=1)),
        "residual_mse": residual_mse,
        "q_crossing_rate": float(np.mean(q_cross)),
        "p_crossing_rate": float(np.mean(p_cross)),
        "any_crossing_rate": float(np.mean(any_cross)),
        "pauli_surrogate_identity": identity_surrogate,
        "pauli_surrogate_average_fidelity": average_fidelity_surrogate,
        "mse_delta_vs_oracle_affine": mse_delta_vs_oracle,
        "crossing_delta_vs_oracle_affine": crossing_delta_vs_oracle,
    }


def _run_scenario(scenario: str, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    mapper = ParamMapper.from_config(CONFIG)
    fixed_prediction = NoisePrediction(
        sigma=0.22,
        mu_q=0.0,
        mu_p=0.0,
        theta_deg=0.0,
        source="fixed_nominal",
        metadata={"holdout_drift_stress_analysis": True},
    )
    fixed_params = mapper.map_prediction(fixed_prediction)
    state_rows = _state_rows(scenario)
    oracle_params_by_step = [mapper.map_prediction(_prediction(state, "oracle_affine")) for state in state_rows]
    lagged_params_by_step = [oracle_params_by_step[_lagged_state_index(step)] for step in range(N_STEPS)]

    residuals_by_method = {
        method: np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float)
        for method in METHODS
    }
    for step, state in enumerate(state_rows):
        errors = _sample_errors(state, rng)
        syndromes = _sample_syndromes(errors, rng)
        residuals_by_method["fixed_affine"][:, step, :] = errors - _affine_corrections(syndromes, fixed_params)
        residuals_by_method["oracle_affine"][:, step, :] = errors - _affine_corrections(syndromes, oracle_params_by_step[step])
        residuals_by_method["lagged_affine"][:, step, :] = errors - _affine_corrections(syndromes, lagged_params_by_step[step])
        wrapped_mean, wrapped_map = _wrapped_gaussian_corrections(state, syndromes)
        residuals_by_method["wrapped_gaussian_posterior_mean"][:, step, :] = errors - wrapped_mean
        residuals_by_method["wrapped_gaussian_map"][:, step, :] = errors - wrapped_map

    oracle_residuals = residuals_by_method["oracle_affine"]
    return [
        _summarize(
            scenario=scenario,
            method=method,
            residuals=residuals,
            oracle_residuals=None if method == "oracle_affine" else oracle_residuals,
        )
        for method, residuals in residuals_by_method.items()
    ]


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
        "sequence_ler_proxy_mean",
        "sequence_ler_proxy_sd",
        "residual_mse",
        "q_crossing_rate",
        "p_crossing_rate",
        "any_crossing_rate",
        "pauli_surrogate_identity",
        "pauli_surrogate_average_fidelity",
        "mse_delta_vs_oracle_affine",
        "crossing_delta_vs_oracle_affine",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(rows: list[dict[str, float | str | int]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_holdout_drift_stress_v1",
        "seed": SEED,
        "n_sequences": N_SEQUENCES,
        "n_steps": N_STEPS,
        "commit_interval_steps": COMMIT_INTERVAL_STEPS,
        "commit_lag_steps": COMMIT_LAG_STEPS,
        "scenarios": list(SCENARIOS),
        "methods": list(METHODS),
        "boundary": (
            "Controlled non-hardware holdout/stress analysis for manuscript positioning. "
            "The lagged_affine row is a slow-commit known-state stress reference, not the trained CNN residual branch. "
            "This is not the formal P4 benchmark, not a confidence-interval run, not finite-energy logical-channel "
            "tomography and not hardware validation."
        ),
        "formula": {
            "sequence_ler_proxy_mean": "mean over sequences of per-step q/p half-lattice crossing indicators",
            "pauli_surrogate_identity": "1 - any_crossing_rate",
            "pauli_surrogate_average_fidelity": "(1 + 2 * pauli_surrogate_identity) / 3",
        },
        "config": CONFIG,
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, float | str | int]]) -> None:
    lines = [
        "# 投稿稿 holdout drift stress 分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它补充三类未见漂移压力测试：random-walk drift、burst/reset drift 和 faster-than-window oscillatory drift。",
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
        f"- lagged affine commit interval: `{COMMIT_INTERVAL_STEPS}` steps",
        f"- lagged affine commit lag: `{COMMIT_LAG_STEPS}` steps",
        "- methods: fixed affine, lagged affine, oracle affine, wrapped-Gaussian posterior mean, wrapped-Gaussian MAP",
        "- metric: per-sequence half-lattice residual-boundary crossing proxy plus residual MSE",
        "",
        "## 结果摘要",
        "",
        "| Scenario | Fixed LER proxy | Lagged LER proxy | Oracle LER proxy | Wrapped mean LER proxy | Wrapped MAP LER proxy | Oracle F_avg_surr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in SCENARIOS:
        selected = {str(row["method"]): row for row in rows if row["scenario"] == scenario}
        lines.append(
            f"| {scenario} | "
            f"{float(selected['fixed_affine']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['lagged_affine']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['oracle_affine']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['wrapped_gaussian_posterior_mean']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['wrapped_gaussian_map']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['oracle_affine']['pauli_surrogate_average_fidelity']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：该分析提供了三类未见漂移压力测试，缓解但不完全关闭 holdout-drift 缺口。",
            "- 可以写：`pauli_surrogate_average_fidelity` 是由 residual-boundary crossing rate 派生的 Pauli-channel-style surrogate，不是 finite-energy logical-channel fidelity。",
            "- 可以写：`lagged_affine` 是慢提交 known-state 参数压力参考，不是当前 CNN residual branch。",
            "- 不能写：该分析完成了正式 expanded benchmark、CI/p-value、真实硬件、完整 logical-channel fidelity 或 trained model generalization proof。",
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
