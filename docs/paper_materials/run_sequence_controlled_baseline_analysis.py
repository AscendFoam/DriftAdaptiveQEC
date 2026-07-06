from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn_fpga.decoder.linear_runtime import LinearRuntime  # noqa: E402
from cnn_fpga.decoder.param_mapper import NoisePrediction, ParamMapper  # noqa: E402
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams  # noqa: E402
from physics.constants import LATTICE_CONST  # noqa: E402


OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_sequence_controlled_baseline_analysis.csv"
JSON_PATH = OUT_DIR / "submission_draft_sequence_controlled_baseline_analysis.json"
REPORT_PATH = OUT_DIR / "投稿稿sequence_controlled_baseline分析记录.md"

SEED = 20260703
N_SEQUENCES = 384
N_STEPS = 512

CONFIG = {
    "hardware_defaults": {
        "fixed_point": "Q4.20",
    },
    "fast_loop": {
        "enable_fixed_point": False,
        "syndrome_limit": LATTICE_CONST / 2.0,
        "correction_limit": LATTICE_CONST,
    },
    "measurement": {
        "delta": 0.3,
        "measurement_efficiency": 0.95,
    },
    "model": {
        "sigma_ratio_p": 0.55,
    },
    "param_mapping": {
        "alpha_bias": 1.0,
        "beta_smoothing": 0.2,
        "gain_clip": [0.2, 1.2],
        "gain_scale": 1.0,
        "theta_clip_deg": [-20.0, 20.0],
        "sigma_ratio_p": 0.55,
    },
}


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.mod(values + LATTICE_CONST / 2.0, LATTICE_CONST) - LATTICE_CONST / 2.0


def _measurement_std() -> float:
    measurement = CONFIG["measurement"]
    eta = float(measurement["measurement_efficiency"])
    inefficiency_var = (1.0 - eta) / (2.0 * eta)
    return float(np.sqrt(float(measurement["delta"]) ** 2 + inefficiency_var))


def _state_rows(scenario: str) -> list[dict[str, float | str]]:
    if scenario == "static_bias_theta":
        return [
            {
                "scenario": scenario,
                "sigma": 0.22,
                "mu_q": 0.035,
                "mu_p": -0.025,
                "theta_deg": 4.0,
            }
            for _ in range(N_STEPS)
        ]
    if scenario == "linear_ramp":
        return [
            {
                "scenario": scenario,
                "sigma": 0.20 + 0.12 * step / (N_STEPS - 1),
                "mu_q": -0.025 + 0.050 * step / (N_STEPS - 1),
                "mu_p": 0.025 - 0.050 * step / (N_STEPS - 1),
                "theta_deg": -4.0 + 8.0 * step / (N_STEPS - 1),
            }
            for step in range(N_STEPS)
        ]
    if scenario == "step_sigma_theta":
        return [
            {
                "scenario": scenario,
                "sigma": 0.22 if step < N_STEPS // 2 else 0.36,
                "mu_q": 0.000 if step < N_STEPS // 2 else 0.040,
                "mu_p": 0.000 if step < N_STEPS // 2 else -0.040,
                "theta_deg": 0.0 if step < N_STEPS // 2 else 8.0,
            }
            for step in range(N_STEPS)
        ]
    if scenario == "periodic_drift":
        return [
            {
                "scenario": scenario,
                "sigma": 0.26 + 0.05 * np.sin(2.0 * np.pi * step / N_STEPS),
                "mu_q": 0.020 * np.sin(2.0 * np.pi * step / N_STEPS),
                "mu_p": -0.020 * np.sin(2.0 * np.pi * step / N_STEPS),
                "theta_deg": 7.0 * np.sin(2.0 * np.pi * step / N_STEPS),
            }
            for step in range(N_STEPS)
        ]
    raise ValueError(f"Unknown scenario: {scenario}")


def _prediction(row: dict[str, float | str], source: str) -> NoisePrediction:
    return NoisePrediction(
        sigma=float(row["sigma"]),
        mu_q=float(row["mu_q"]),
        mu_p=float(row["mu_p"]),
        theta_deg=float(row["theta_deg"]),
        source=source,
        metadata={"sequence_controlled_analysis": True},
    )


def _scenario_covariance(row: dict[str, float | str]) -> np.ndarray:
    theta = np.deg2rad(float(row["theta_deg"]))
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    sigma = float(row["sigma"])
    sigma_ratio_p = float(CONFIG["model"]["sigma_ratio_p"])
    diagonal = np.diag([sigma**2, (sigma * sigma_ratio_p) ** 2])
    return rotation @ diagonal @ rotation.T


def _sample_errors(row: dict[str, float | str], rng: np.random.Generator) -> np.ndarray:
    covariance = _scenario_covariance(row)
    mean = np.array([float(row["mu_q"]), float(row["mu_p"])], dtype=float)
    return rng.multivariate_normal(mean, covariance, size=N_SEQUENCES)


def _sample_syndromes(errors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _wrap(errors) + rng.normal(0.0, _measurement_std(), size=errors.shape)


def _wrapped_gaussian_corrections(
    row: dict[str, float | str],
    syndromes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    branches = np.array(
        [[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)],
        dtype=float,
    )
    shifts = branches * LATTICE_CONST
    mean = np.array([float(row["mu_q"]), float(row["mu_p"])], dtype=float)
    covariance = _scenario_covariance(row)
    measurement_covariance = np.eye(2, dtype=float) * (_measurement_std() ** 2)
    observation_covariance = covariance + measurement_covariance
    inv_observation = np.linalg.inv(observation_covariance)
    kalman_gain = covariance @ inv_observation

    observations = syndromes[:, None, :] + shifts[None, :, :]
    diff = observations - mean[None, None, :]
    posterior_means = mean[None, None, :] + np.einsum("nbi,ji->nbj", diff, kalman_gain)
    quadratic = np.einsum("nbi,ij,nbj->nb", diff, inv_observation, diff)
    logits = -0.5 * quadratic
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    posterior_mean = np.sum(weights[:, :, None] * posterior_means, axis=1)
    map_mean = posterior_means[np.arange(syndromes.shape[0]), np.argmax(weights, axis=1)]
    return posterior_mean, map_mean


def _summarize(
    *,
    scenario: str,
    method: str,
    residuals: np.ndarray,
) -> dict[str, float | str | int]:
    boundary = LATTICE_CONST / 2.0
    q_cross = np.abs(residuals[:, :, 0]) > boundary
    p_cross = np.abs(residuals[:, :, 1]) > boundary
    any_cross = np.logical_or(q_cross, p_cross)
    per_sequence = np.mean(any_cross, axis=1)
    return {
        "scenario": scenario,
        "method": method,
        "n_sequences": N_SEQUENCES,
        "n_steps": N_STEPS,
        "sequence_ler_proxy_mean": float(np.mean(per_sequence)),
        "sequence_ler_proxy_sd": float(np.std(per_sequence, ddof=1)),
        "residual_mse": float(np.mean(np.sum(residuals**2, axis=2))),
        "q_crossing_rate": float(np.mean(q_cross)),
        "p_crossing_rate": float(np.mean(p_cross)),
        "any_crossing_rate": float(np.mean(any_cross)),
    }


def _run_scenario(scenario: str, rng: np.random.Generator) -> list[dict[str, float | str | int]]:
    mapper = ParamMapper.from_config(CONFIG)
    runtime = LinearRuntime.from_config(CONFIG)
    fixed_prediction = NoisePrediction(
        sigma=0.22,
        mu_q=0.0,
        mu_p=0.0,
        theta_deg=0.0,
        source="fixed_nominal",
        metadata={"sequence_controlled_analysis": True},
    )
    fixed_params = mapper.map_prediction(fixed_prediction)
    nearest_params = DecoderRuntimeParams(
        K=np.eye(2, dtype=float),
        b=np.zeros(2, dtype=float),
        metadata={"source": "nearest_syndrome"},
    )
    state_rows = _state_rows(scenario)

    residuals_by_method = {
        "nearest_syndrome": np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float),
        "fixed_affine": np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float),
        "oracle_affine": np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float),
        "wrapped_gaussian_posterior_mean": np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float),
        "wrapped_gaussian_map": np.zeros((N_SEQUENCES, N_STEPS, 2), dtype=float),
    }

    for step, state in enumerate(state_rows):
        errors = _sample_errors(state, rng)
        syndromes = _sample_syndromes(errors, rng)
        oracle_params = mapper.map_prediction(_prediction(state, "oracle_affine"))
        nearest_corrections = np.vstack([runtime.decode(syndrome, nearest_params).correction_applied for syndrome in syndromes])
        fixed_corrections = np.vstack([runtime.decode(syndrome, fixed_params).correction_applied for syndrome in syndromes])
        oracle_corrections = np.vstack([runtime.decode(syndrome, oracle_params).correction_applied for syndrome in syndromes])
        wrapped_mean, wrapped_map = _wrapped_gaussian_corrections(state, syndromes)
        residuals_by_method["nearest_syndrome"][:, step, :] = errors - nearest_corrections
        residuals_by_method["fixed_affine"][:, step, :] = errors - fixed_corrections
        residuals_by_method["oracle_affine"][:, step, :] = errors - oracle_corrections
        residuals_by_method["wrapped_gaussian_posterior_mean"][:, step, :] = errors - wrapped_mean
        residuals_by_method["wrapped_gaussian_map"][:, step, :] = errors - wrapped_map

    return [
        _summarize(scenario=scenario, method=method, residuals=residuals)
        for method, residuals in residuals_by_method.items()
    ]


def _rows() -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | str | int]] = []
    for scenario in ("static_bias_theta", "linear_ramp", "step_sigma_theta", "periodic_drift"):
        rows.extend(_run_scenario(scenario, rng))
    return rows


def _write_csv(rows: list[dict[str, float | str | int]]) -> None:
    fields = [
        "scenario",
        "method",
        "n_sequences",
        "n_steps",
        "sequence_ler_proxy_mean",
        "sequence_ler_proxy_sd",
        "residual_mse",
        "q_crossing_rate",
        "p_crossing_rate",
        "any_crossing_rate",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_json(rows: list[dict[str, float | str | int]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_sequence_controlled_baseline_v1",
        "seed": SEED,
        "n_sequences": N_SEQUENCES,
        "n_steps": N_STEPS,
        "boundary": (
            "Controlled sequence-level local-Gaussian baseline analysis for manuscript positioning. "
            "Not the formal P4 benchmark, not hardware, not holdout drift, and not a statistical CI run."
        ),
        "config": CONFIG,
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, float | str | int]]) -> None:
    lines = [
        "# 投稿稿 sequence-controlled baseline 分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它把 one-step oracle-affine / wrapped-Gaussian sanity check 扩展到短序列 controlled local-Gaussian drift setting。",
        "",
        "它不是正式 P4 benchmark、不是 holdout drift、不是 confidence interval run、不是硬件测量，也不改变已有主结果证据等级。",
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
        "- scenarios: static bias/theta, linear ramp, step sigma/theta, periodic drift",
        "- methods: nearest syndrome, fixed affine, oracle affine, wrapped-Gaussian posterior mean, wrapped-Gaussian MAP",
        "- metric: sequence mean of half-lattice residual-boundary crossing proxy plus residual MSE",
        "",
        "## 结果摘要",
        "",
        "| Scenario | Nearest syndrome | Fixed | Oracle | Wrapped mean | Wrapped MAP |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in ("static_bias_theta", "linear_ramp", "step_sigma_theta", "periodic_drift"):
        selected = {str(row["method"]): row for row in rows if row["scenario"] == scenario}
        lines.append(
            f"| {scenario} | "
            f"{float(selected['nearest_syndrome']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['fixed_affine']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['oracle_affine']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['wrapped_gaussian_posterior_mean']['sequence_ler_proxy_mean']):.6f} | "
            f"{float(selected['wrapped_gaussian_map']['sequence_ler_proxy_mean']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：在 controlled sequence setting 中，oracle affine 通常改善 fixed affine；wrapped-Gaussian posterior mean/MAP 并未稳定支配 oracle affine。",
            "- 可以写：该结果比 one-step sanity check 更接近 sequence-level baseline，但仍然只是受控 local-Gaussian positioning analysis。",
            "- 不能写：该分析已经补齐正式 benchmark、CI、holdout drift、logical-channel fidelity 或硬件证据。",
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
