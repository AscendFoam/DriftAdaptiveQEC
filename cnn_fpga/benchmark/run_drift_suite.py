"""Run drift scenarios for full_qec vs simplified models."""

# 中文说明：
# - 本脚本对不同漂移场景运行多次仿真，比较 full_qec 与 simplified 的 LER 差异。
# - 输出每个场景的曲线 CSV 和汇总 summary.json，便于实验报告复用。

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from physics.error_correction import LinearDecoder
from physics.constants import LATTICE_CONST
from physics.logical_tracking import LogicalErrorTracker
from physics.syndrome_measurement import MeasurementConfig, RealisticSyndromeMeasurement

from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, now_tag, save_json


def _wrap(value: float, lattice: float) -> float:
    return float(np.mod(value + lattice / 2.0, lattice) - lattice / 2.0)


def _build_sigma_trace(rng: np.random.Generator, n_rounds: int, scenario: Dict) -> np.ndarray:
    # 中文注释：根据场景类型生成随时间变化的 sigma(t) 轨迹。
    kind = str(scenario["type"]).lower()
    p = scenario.get("params", {})
    t = np.arange(n_rounds, dtype=float)

    if kind == "linear":
        sigma0 = float(p.get("sigma0", 0.2))
        alpha = float(p.get("alpha", 0.0))
        sigma = sigma0 + alpha * t
    elif kind == "step":
        sigma0 = float(p.get("sigma0", 0.2))
        delta_sigma = float(p.get("delta_sigma", 0.1))
        jump_round = int(p.get("jump_round", n_rounds // 2))
        sigma = np.full(n_rounds, sigma0, dtype=float)
        sigma[jump_round:] += delta_sigma
    elif kind in ("sin", "sine", "periodic"):
        sigma0 = float(p.get("sigma0", 0.2))
        amp = float(p.get("amplitude", 0.05))
        freq = float(p.get("frequency", 0.002))
        phase = float(p.get("phase", 0.0))
        sigma = sigma0 + amp * np.sin(2 * np.pi * freq * t + phase)
    elif kind in ("random_walk", "rw"):
        sigma0 = float(p.get("sigma0", 0.2))
        step_std = float(p.get("step_std", 0.002))
        sigma = np.zeros(n_rounds, dtype=float)
        sigma[0] = sigma0
        for i in range(1, n_rounds):
            sigma[i] = sigma[i - 1] + float(rng.normal(0.0, step_std))
    else:
        raise ValueError(f"Unsupported drift type: {scenario['type']}")

    sigma_min = float(p.get("sigma_min", 0.05))
    sigma_max = float(p.get("sigma_max", 0.8))
    sigma = np.clip(sigma, sigma_min, sigma_max)
    return sigma


def _simulate_trace(
    sigma_trace: np.ndarray,
    model_cfg: Dict,
    meas_cfg: Dict,
    use_full_qec_model: bool,
    seed: int,
) -> np.ndarray:
    # 中文注释：给定一条 sigma 轨迹，逐轮推进并计算累计 LER 曲线。
    np.random.seed(seed)
    tracker = LogicalErrorTracker()
    lattice = LATTICE_CONST

    theta_deg = float(model_cfg.get("theta_deg", 0.0))
    theta = np.deg2rad(theta_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)

    gain = float(model_cfg.get("gain", 0.9))
    sigma_ratio_p = float(model_cfg.get("sigma_ratio_p", 1.0))
    sigma_measurement = float(model_cfg.get("sigma_measurement", 0.05))
    mu_q = float(model_cfg.get("mu_q", 0.0))
    mu_p = float(model_cfg.get("mu_p", 0.0))
    bias = np.array([mu_q, mu_p], dtype=float)

    cumulative_residual = np.zeros(2, dtype=float)

    config = MeasurementConfig(
        delta=float(meas_cfg.get("delta", 0.3)),
        measurement_efficiency=float(meas_cfg.get("measurement_efficiency", 0.95)),
        ancilla_error_rate=float(meas_cfg.get("ancilla_error_rate", 0.01)),
        add_shot_noise=bool(meas_cfg.get("add_shot_noise", True)),
    )
    measurement = RealisticSyndromeMeasurement(config)
    decoder = LinearDecoder(K=gain * rot, b=np.zeros(2, dtype=float))

    curve = np.zeros(sigma_trace.size, dtype=float)
    n_err = 0.0

    for i, sigma in enumerate(sigma_trace):
        base_noise = np.array(
            [
                np.random.normal(0.0, sigma),
                np.random.normal(0.0, sigma * sigma_ratio_p),
            ],
            dtype=float,
        )
        new_error = bias + rot @ base_noise

        if use_full_qec_model:
            # 中文注释：完整闭环模型会携带跨轮残差并经过真实测量模型。
            total_error = cumulative_residual + new_error
            syndrome = measurement.measure(total_error, add_noise=True)
            if sigma_measurement > 0:
                syndrome = syndrome + np.random.normal(0.0, sigma_measurement, size=2)
            correction = decoder.decode(syndrome)
            x_err, z_err = tracker.update(total_error[0], total_error[1], correction[0], correction[1])
            cumulative_residual = np.array([tracker.accumulated_q, tracker.accumulated_p], dtype=float)
        else:
            # 中文注释：简化模型按每轮独立近似处理，便于做对照。
            error_q, error_p = new_error
            syndrome_q = _wrap(error_q, lattice) + float(np.random.normal(0.0, sigma_measurement))
            syndrome_p = _wrap(error_p, lattice) + float(np.random.normal(0.0, sigma_measurement))
            correction_q = gain * syndrome_q
            correction_p = gain * syndrome_p
            x_err, z_err = tracker.update(error_q, error_p, correction_q, correction_p)

        if x_err or z_err:
            n_err += 1.0
        curve[i] = n_err / float(i + 1)

    return curve


def _write_curve_csv(path: Path, rounds: np.ndarray, full_mean: np.ndarray, full_std: np.ndarray, sim_mean: np.ndarray, sim_std: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "full_mean", "full_std", "simplified_mean", "simplified_std", "gap_mean"])
        for i in range(rounds.size):
            writer.writerow(
                [
                    int(rounds[i]),
                    float(full_mean[i]),
                    float(full_std[i]),
                    float(sim_mean[i]),
                    float(sim_std[i]),
                    float(full_mean[i] - sim_mean[i]),
                ]
            )


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run drift scenario suite.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)
    cfg_hash = config_hash(config)

    experiment = config.get("experiment", {})
    seed_base = int(experiment.get("seed", 1234))
    n_rounds = int(config.get("simulation", {}).get("n_rounds", 2000))
    repeats = int(config.get("simulation", {}).get("repeats", 10))

    model_cfg = config.get("model", {})
    meas_cfg = config.get("measurement", {})
    scenarios = config.get("drift", {}).get("scenarios", [])
    if not scenarios:
        raise ValueError("No drift scenarios configured.")

    out_root = ensure_dir(get_path(config, "output_root", "runs"))
    run_dir = ensure_dir(out_root / "drift_suite" / f"{experiment.get('name', 'drift')}_{now_tag()}_{cfg_hash}")
    rounds = np.arange(1, n_rounds + 1, dtype=int)

    # 中文注释：按场景逐个跑完后统一写入总摘要文件。
    scenario_summaries: List[Dict] = []
    for scenario in scenarios:
        scenario_name = str(scenario.get("name", scenario["type"]))
        full_curves = []
        sim_curves = []

        for rep in range(repeats):
            rng = np.random.default_rng(seed_base + rep)
            sigma_trace = _build_sigma_trace(rng, n_rounds=n_rounds, scenario=scenario)
            full_curve = _simulate_trace(
                sigma_trace=sigma_trace,
                model_cfg=model_cfg,
                meas_cfg=meas_cfg,
                use_full_qec_model=True,
                seed=seed_base + rep,
            )
            sim_curve = _simulate_trace(
                sigma_trace=sigma_trace,
                model_cfg=model_cfg,
                meas_cfg=meas_cfg,
                use_full_qec_model=False,
                seed=seed_base + rep,
            )
            full_curves.append(full_curve)
            sim_curves.append(sim_curve)

        full_arr = np.vstack(full_curves)
        sim_arr = np.vstack(sim_curves)

        full_mean = full_arr.mean(axis=0)
        full_std = full_arr.std(axis=0)
        sim_mean = sim_arr.mean(axis=0)
        sim_std = sim_arr.std(axis=0)

        csv_path = run_dir / f"{scenario_name}_curve.csv"
        _write_curve_csv(csv_path, rounds, full_mean, full_std, sim_mean, sim_std)

        summary = {
            "scenario": scenario_name,
            "type": scenario["type"],
            "n_rounds": n_rounds,
            "repeats": repeats,
            "full_final_ler_mean": float(full_mean[-1]),
            "full_final_ler_std": float(full_std[-1]),
            "simplified_final_ler_mean": float(sim_mean[-1]),
            "simplified_final_ler_std": float(sim_std[-1]),
            "final_gap_mean": float(full_mean[-1] - sim_mean[-1]),
            "curve_csv": str(csv_path),
        }
        scenario_summaries.append(summary)
        print(f"[{scenario_name}] full={summary['full_final_ler_mean']:.6f}, simplified={summary['simplified_final_ler_mean']:.6f}, gap={summary['final_gap_mean']:.6f}")

    save_json(
        run_dir / "summary.json",
        {
            "config_hash": cfg_hash,
            "run_dir": str(run_dir),
            "n_rounds": n_rounds,
            "repeats": repeats,
            "scenarios": scenario_summaries,
        },
    )
    print(f"Drift suite complete. Outputs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
