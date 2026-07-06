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
from cnn_fpga.decoder.linear_runtime import LinearRuntime  # noqa: E402
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams  # noqa: E402
from physics.constants import LATTICE_CONST  # noqa: E402


OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_controlled_oracle_affine_analysis.csv"
JSON_PATH = OUT_DIR / "submission_draft_controlled_oracle_affine_analysis.json"
REPORT_PATH = OUT_DIR / "投稿稿oracle_affine受控分析记录.md"

SEED = 20260703
N_SAMPLES = 120_000

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

SCENARIOS = [
    {
        "scenario": "static_bias_theta",
        "label": "Static Bias+Theta",
        "sigma": 0.22,
        "mu_q": 0.035,
        "mu_p": -0.025,
        "theta_deg": 4.0,
    },
    {
        "scenario": "linear_ramp_midpoint",
        "label": "Linear Ramp Midpoint",
        "sigma": 0.26,
        "mu_q": 0.010,
        "mu_p": -0.010,
        "theta_deg": 2.0,
    },
    {
        "scenario": "step_after_jump",
        "label": "Step After Jump",
        "sigma": 0.36,
        "mu_q": 0.040,
        "mu_p": -0.040,
        "theta_deg": 8.0,
    },
    {
        "scenario": "periodic_high_phase",
        "label": "Periodic High Phase",
        "sigma": 0.31,
        "mu_q": 0.015,
        "mu_p": -0.015,
        "theta_deg": 7.0,
    },
]


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.mod(values + LATTICE_CONST / 2.0, LATTICE_CONST) - LATTICE_CONST / 2.0


def _prediction(row: dict[str, float | str], source: str) -> NoisePrediction:
    return NoisePrediction(
        sigma=float(row["sigma"]),
        mu_q=float(row["mu_q"]),
        mu_p=float(row["mu_p"]),
        theta_deg=float(row["theta_deg"]),
        source=source,
        metadata={"controlled_analysis": True},
    )


def _sample_errors(row: dict[str, float | str], rng: np.random.Generator) -> np.ndarray:
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
    base = rng.normal(0.0, [sigma, sigma * sigma_ratio_p], size=(N_SAMPLES, 2))
    mean = np.array([float(row["mu_q"]), float(row["mu_p"])], dtype=float)
    return mean + base @ rotation.T


def _sample_syndromes(errors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    measurement_std = _measurement_std()
    return _wrap(errors) + rng.normal(0.0, measurement_std, size=errors.shape)


def _measurement_std() -> float:
    measurement = CONFIG["measurement"]
    eta = float(measurement["measurement_efficiency"])
    inefficiency_var = (1.0 - eta) / (2.0 * eta)
    return float(np.sqrt(float(measurement["delta"]) ** 2 + inefficiency_var))


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


def _wrapped_gaussian_corrections(
    row: dict[str, float | str],
    syndromes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Known-state wrapped-Gaussian posterior baselines for one-step decoding."""
    branches = np.array(
        [
            [i, j]
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
        ],
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
    scenario: dict[str, float | str],
    method: str,
    params: DecoderRuntimeParams | None,
    corrections: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float | str | int]:
    boundary = LATTICE_CONST / 2.0
    q_cross = np.abs(residuals[:, 0]) > boundary
    p_cross = np.abs(residuals[:, 1]) > boundary
    any_cross = np.logical_or(q_cross, p_cross)
    row: dict[str, float | str | int] = {
        "scenario": str(scenario["scenario"]),
        "scenario_label": str(scenario["label"]),
        "method": method,
        "n_samples": N_SAMPLES,
        "sigma": float(scenario["sigma"]),
        "mu_q": float(scenario["mu_q"]),
        "mu_p": float(scenario["mu_p"]),
        "theta_deg": float(scenario["theta_deg"]),
        "residual_mse": float(np.mean(np.sum(residuals**2, axis=1))),
        "mean_abs_residual_q": float(np.mean(np.abs(residuals[:, 0]))),
        "mean_abs_residual_p": float(np.mean(np.abs(residuals[:, 1]))),
        "boundary_crossing_rate": float(np.mean(any_cross)),
        "q_boundary_crossing_rate": float(np.mean(q_cross)),
        "p_boundary_crossing_rate": float(np.mean(p_cross)),
        "mean_correction_norm": float(np.mean(np.linalg.norm(corrections, axis=1))),
    }
    if params is None:
        row.update({"max_gain": "", "bias_norm": ""})
    else:
        row.update(
            {
                "max_gain": float(np.max(np.abs(np.linalg.eigvalsh(0.5 * (params.K + params.K.T))))),
                "bias_norm": float(np.linalg.norm(params.b)),
            }
        )
    return row


def _controlled_rows() -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(SEED)
    mapper = ParamMapper.from_config(CONFIG)
    runtime = LinearRuntime.from_config(CONFIG)
    fixed_prediction = NoisePrediction(
        sigma=0.22,
        mu_q=0.0,
        mu_p=0.0,
        theta_deg=0.0,
        source="fixed_nominal",
        metadata={"controlled_analysis": True},
    )
    fixed_params = mapper.map_prediction(fixed_prediction)
    identity_params = DecoderRuntimeParams(K=np.eye(2, dtype=float), b=np.zeros(2, dtype=float), metadata={"source": "nearest_syndrome"})
    zero_corrections_params = DecoderRuntimeParams(K=np.zeros((2, 2), dtype=float), b=np.zeros(2, dtype=float), metadata={"source": "no_correction"})

    rows: list[dict[str, float | str | int]] = []
    for scenario in SCENARIOS:
        errors = _sample_errors(scenario, rng)
        syndromes = _sample_syndromes(errors, rng)
        method_params = {
            "no_correction": zero_corrections_params,
            "nearest_syndrome": identity_params,
            "fixed_affine": fixed_params,
            "oracle_affine": mapper.map_prediction(_prediction(scenario, "oracle_affine")),
        }
        for method, params in method_params.items():
            corrections = np.vstack([runtime.decode(syndrome, params).correction_applied for syndrome in syndromes])
            residuals = errors - corrections
            rows.append(
                _summarize(
                    scenario=scenario,
                    method=method,
                    params=None if method == "no_correction" else params,
                    corrections=corrections,
                    residuals=residuals,
                )
            )
        wrapped_mean, wrapped_map = _wrapped_gaussian_corrections(scenario, syndromes)
        for method, corrections in {
            "wrapped_gaussian_posterior_mean": wrapped_mean,
            "wrapped_gaussian_map": wrapped_map,
        }.items():
            rows.append(
                _summarize(
                    scenario=scenario,
                    method=method,
                    params=None,
                    corrections=corrections,
                    residuals=errors - corrections,
                )
            )
    return rows


def _write_csv(rows: list[dict[str, float | str | int]]) -> None:
    fields = [
        "scenario",
        "scenario_label",
        "method",
        "n_samples",
        "sigma",
        "mu_q",
        "mu_p",
        "theta_deg",
        "residual_mse",
        "mean_abs_residual_q",
        "mean_abs_residual_p",
        "boundary_crossing_rate",
        "q_boundary_crossing_rate",
        "p_boundary_crossing_rate",
        "mean_correction_norm",
        "max_gain",
        "bias_norm",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(rows: list[dict[str, float | str | int]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "controlled_oracle_affine_wrapped_gaussian_local_v2",
        "seed": SEED,
        "n_samples_per_scenario": N_SAMPLES,
        "source_csv": str(CSV_PATH),
        "boundary": (
            "Controlled local-Gaussian analysis using the manuscript affine mapper "
            "and one-step wrapped-Gaussian posterior baselines. Not a formal P4 "
            "benchmark, not a sequence-level wrapped decoder, not a hardware measurement."
        ),
        "config": CONFIG,
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_report(rows: list[dict[str, float | str | int]]) -> None:
    by_scenario: dict[str, list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)

    lines = [
        "# 投稿稿 oracle-affine 受控分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex` 的投稿稿补强。"
        "分析目标是补一个小规模、可复现的 oracle-affine / wrapped-Gaussian sanity check，用于区分 affine fast-path 的局部模型上限、slow-loop estimator 误差和 wrapped posterior 参照。"
        "它不是正式 P4 benchmark、不是 sequence-level wrapped decoder、不是 holdout drift、不是硬件测量，也不改变已有主结果的证据等级。",
        "",
        "## 生成文件",
        "",
        f"- `{CSV_PATH.relative_to(ROOT)}`",
        f"- `{JSON_PATH.relative_to(ROOT)}`",
        "",
        "## 协议",
        "",
        f"- random seed: `{SEED}`",
        f"- samples per scenario: `{N_SAMPLES}`",
        "- scenarios: static bias/theta, linear-ramp midpoint, post-step state, periodic high-phase state",
        "- methods: no correction, nearest-syndrome, fixed nominal affine, oracle affine, wrapped-Gaussian posterior mean, wrapped-Gaussian MAP",
        "- metric: one-step residual MSE and half-lattice residual-boundary crossing rate",
        "",
        "## 结果摘要",
        "",
        "| Scenario | Fixed MSE | Oracle MSE | Wrapped mean MSE | Wrapped MAP MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scenario, items in by_scenario.items():
        item_by_method = {str(item["method"]): item for item in items}
        fixed = item_by_method["fixed_affine"]
        oracle = item_by_method["oracle_affine"]
        wrapped_mean = item_by_method["wrapped_gaussian_posterior_mean"]
        wrapped_map = item_by_method["wrapped_gaussian_map"]
        lines.append(
            f"| {scenario} | {float(fixed['residual_mse']):.6f} | "
            f"{float(oracle['residual_mse']):.6f} | "
            f"{float(wrapped_mean['residual_mse']):.6f} | "
            f"{float(wrapped_map['residual_mse']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：在受控 local-Gaussian setting 中，oracle affine 参数通常降低 residual MSE，说明 affine fast path 本身有可解释的局部上限。",
            "- 可以写：wrapped-Gaussian posterior mean 在本受控一步设置中给出混合结果；它只在 static state 略优，其他状态并未优于 oracle affine，说明正式 wrapped baseline 需要独立协议和调参。",
            "- 不能写：该分析已经补齐正式 benchmark、统计显著性、holdout drift、logical-channel fidelity 或硬件证据。",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = _controlled_rows()
    _write_csv(rows)
    _write_json(rows)
    _write_report(rows)
    print(json.dumps({"status": "generated", "csv": str(CSV_PATH), "json": str(JSON_PATH), "report": str(REPORT_PATH)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
