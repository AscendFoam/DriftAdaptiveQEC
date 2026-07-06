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
from physics.constants import LATTICE_CONST  # noqa: E402


OUT_DIR = ROOT / "docs" / "paper_materials"
CSV_PATH = OUT_DIR / "submission_draft_fixed_point_parity_analysis.csv"
JSON_PATH = OUT_DIR / "submission_draft_fixed_point_parity_analysis.json"
REPORT_PATH = OUT_DIR / "投稿稿fixed_point一致性分析记录.md"

SEED = 20260703
N_SAMPLES = 80_000

BASE_CONFIG = {
    "hardware_defaults": {
        "fixed_point": "Q4.20",
    },
    "fast_loop": {
        "fixed_point": "Q4.20",
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
        "sigma": 0.22,
        "mu_q": 0.035,
        "mu_p": -0.025,
        "theta_deg": 4.0,
    },
    {
        "scenario": "linear_ramp_midpoint",
        "sigma": 0.26,
        "mu_q": 0.010,
        "mu_p": -0.010,
        "theta_deg": 2.0,
    },
    {
        "scenario": "step_after_jump",
        "sigma": 0.36,
        "mu_q": 0.040,
        "mu_p": -0.040,
        "theta_deg": 8.0,
    },
    {
        "scenario": "periodic_high_phase",
        "sigma": 0.31,
        "mu_q": 0.015,
        "mu_p": -0.015,
        "theta_deg": 7.0,
    },
]


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.mod(values + LATTICE_CONST / 2.0, LATTICE_CONST) - LATTICE_CONST / 2.0


def _measurement_std() -> float:
    measurement = BASE_CONFIG["measurement"]
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
    sigma_ratio_p = float(BASE_CONFIG["model"]["sigma_ratio_p"])
    diagonal = np.diag([sigma**2, (sigma * sigma_ratio_p) ** 2])
    return rotation @ diagonal @ rotation.T


def _sample_errors(row: dict[str, float | str], rng: np.random.Generator) -> np.ndarray:
    covariance = _scenario_covariance(row)
    mean = np.array([float(row["mu_q"]), float(row["mu_p"])], dtype=float)
    return rng.multivariate_normal(mean, covariance, size=N_SAMPLES)


def _sample_syndromes(errors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return _wrap(errors) + rng.normal(0.0, _measurement_std(), size=errors.shape)


def _prediction(row: dict[str, float | str]) -> NoisePrediction:
    return NoisePrediction(
        sigma=float(row["sigma"]),
        mu_q=float(row["mu_q"]),
        mu_p=float(row["mu_p"]),
        theta_deg=float(row["theta_deg"]),
        source="oracle_affine",
        metadata={"fixed_point_parity_analysis": True},
    )


def _runtime_config(enable_fixed_point: bool) -> dict[str, object]:
    config = json.loads(json.dumps(BASE_CONFIG))
    config["fast_loop"]["enable_fixed_point"] = enable_fixed_point
    return config


def _decode_many(runtime: LinearRuntime, syndromes: np.ndarray, params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrections = []
    clip_saturated = []
    fixed_saturated = []
    for syndrome in syndromes:
        result = runtime.decode(syndrome, params)
        corrections.append(result.correction_applied)
        clip_saturated.append(np.any(result.correction_clip_saturated))
        fixed_saturated.append(np.any(result.correction_fixed_point_saturated))
    return (
        np.asarray(corrections, dtype=float),
        np.asarray(clip_saturated, dtype=bool),
        np.asarray(fixed_saturated, dtype=bool),
    )


def _scenario_row(row: dict[str, float | str], rng: np.random.Generator) -> dict[str, float | str | int]:
    mapper = ParamMapper.from_config(BASE_CONFIG)
    params = mapper.map_prediction(_prediction(row))
    errors = _sample_errors(row, rng)
    syndromes = _sample_syndromes(errors, rng)

    float_runtime = LinearRuntime.from_config(_runtime_config(False))
    fixed_runtime = LinearRuntime.from_config(_runtime_config(True))
    float_corr, float_clip_sat, _ = _decode_many(float_runtime, syndromes, params)
    fixed_corr, fixed_clip_sat, fixed_quant_sat = _decode_many(fixed_runtime, syndromes, params)

    diff = fixed_corr - float_corr
    residual_float = errors - float_corr
    residual_fixed = errors - fixed_corr
    boundary = LATTICE_CONST / 2.0
    float_cross = np.any(np.abs(residual_float) > boundary, axis=1)
    fixed_cross = np.any(np.abs(residual_fixed) > boundary, axis=1)

    return {
        "scenario": str(row["scenario"]),
        "n_samples": N_SAMPLES,
        "fixed_point_spec": "Q4.20",
        "max_abs_correction_diff": float(np.max(np.abs(diff))),
        "p99_abs_correction_diff": float(np.quantile(np.max(np.abs(diff), axis=1), 0.99)),
        "mean_abs_correction_diff": float(np.mean(np.abs(diff))),
        "float_residual_mse": float(np.mean(np.sum(residual_float**2, axis=1))),
        "fixed_residual_mse": float(np.mean(np.sum(residual_fixed**2, axis=1))),
        "residual_mse_delta_fixed_minus_float": float(
            np.mean(np.sum(residual_fixed**2, axis=1)) - np.mean(np.sum(residual_float**2, axis=1))
        ),
        "float_boundary_crossing_rate": float(np.mean(float_cross)),
        "fixed_boundary_crossing_rate": float(np.mean(fixed_cross)),
        "boundary_crossing_delta_fixed_minus_float": float(np.mean(fixed_cross) - np.mean(float_cross)),
        "float_clip_saturation_rate": float(np.mean(float_clip_sat)),
        "fixed_clip_saturation_rate": float(np.mean(fixed_clip_sat)),
        "fixed_point_quant_saturation_rate": float(np.mean(fixed_quant_sat)),
    }


def _rows() -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(SEED)
    return [_scenario_row(row, rng) for row in SCENARIOS]


def _write_csv(rows: list[dict[str, float | str | int]]) -> None:
    fields = [
        "scenario",
        "n_samples",
        "fixed_point_spec",
        "max_abs_correction_diff",
        "p99_abs_correction_diff",
        "mean_abs_correction_diff",
        "float_residual_mse",
        "fixed_residual_mse",
        "residual_mse_delta_fixed_minus_float",
        "float_boundary_crossing_rate",
        "fixed_boundary_crossing_rate",
        "boundary_crossing_delta_fixed_minus_float",
        "float_clip_saturation_rate",
        "fixed_clip_saturation_rate",
        "fixed_point_quant_saturation_rate",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_json(rows: list[dict[str, float | str | int]]) -> None:
    payload = {
        "status": "generated",
        "analysis_id": "submission_draft_fixed_point_parity_v1",
        "seed": SEED,
        "n_samples_per_scenario": N_SAMPLES,
        "boundary": (
            "Software fixed-point emulation parity check for the affine fast path. "
            "This checks Q4.20 numerical degradation against floating-point runtime "
            "under controlled local-Gaussian samples. It is not FPGA synthesis, timing "
            "closure, resource use, power, board execution or logical-channel fidelity."
        ),
        "config": BASE_CONFIG,
        "rows": rows,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_report(rows: list[dict[str, float | str | int]]) -> None:
    lines = [
        "# 投稿稿 fixed-point 一致性分析记录",
        "",
        "日期：2026-07-03",
        "",
        "本记录服务于 `docs/paper_notes/CNN_FPGA_GKP_submission_draft.tex`。它用 `LinearRuntime` 的 Q4.20 fixed-point emulation 对比 floating-point affine fast path，检查固定点量化是否在受控 local-Gaussian 样本上引入可见退化。",
        "",
        "它不是 FPGA synthesis、不是 timing closure、不是 resource/power 测量、不是 real-board execution，也不是 logical-channel fidelity 分析。",
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
        "- runtime pair: floating-point affine fast path vs Q4.20 fixed-point emulation",
        "",
        "## 结果摘要",
        "",
        "| Scenario | Max abs diff | p99 abs diff | MSE delta | Crossing delta | Quant sat. |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | "
            f"{float(row['max_abs_correction_diff']):.9f} | "
            f"{float(row['p99_abs_correction_diff']):.9f} | "
            f"{float(row['residual_mse_delta_fixed_minus_float']):.9e} | "
            f"{float(row['boundary_crossing_delta_fixed_minus_float']):.9e} | "
            f"{float(row['fixed_point_quant_saturation_rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 可写边界",
            "",
            "- 可以写：在受控样本和当前 affine fast-path 参数范围内，Q4.20 emulation 与 floating-point 输出的 correction 差异处于约一个 quantization step 的量级，未产生可见的 residual-boundary crossing 退化。",
            "- 可以写：该结果支持 fixed-point feasibility 的软件数值一致性动机。",
            "- 不能写：该结果已经证明 FPGA timing closure、LUT/FF/DSP/BRAM、power、source-vs-board agreement、real-board latency 或 logical-channel fidelity。",
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
