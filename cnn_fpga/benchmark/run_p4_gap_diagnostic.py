"""Diagnose why current CNN-FPGA loses to simple statistical baselines in P4."""

# 中文说明：
# - 该脚本先用一个参考闭环模式采集“原始窗口直方图 + 真实目标参数”，
#   再在完全相同的窗口序列上离线比较 `CNN-FPGA / Window Variance / EKF`。
# - 这样可以把“正式 benchmark 的不同 mode 使用不同 seed、不同闭环轨迹”与
#   “同一批窗口上的预测能力差异”区分开，便于定位 CNN 落后的真实原因。

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from cnn_fpga.benchmark.run_hil_suite import HILSlowJob, _build_mock_noise_provider, _required_fast_cycles
from cnn_fpga.hwio import FPGADriver
from cnn_fpga.runtime import LatencyInjector, SlowLoopRuntime
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.scheduler import SchedulerConfig, WindowFrame
from cnn_fpga.runtime.slow_loop_runtime import NoisePrediction
from cnn_fpga.utils.config import config_hash, ensure_dir, load_yaml_config, now_tag, save_json


DEFAULT_COMPARE_MODES = ("window_variance", "ekf", "cnn_fpga")


@dataclass(frozen=True)
class CapturedWindow:
    """Reference window captured from one closed-loop rollout."""

    window_id: int
    start_epoch: int
    end_epoch: int
    ready_time_us: float
    histogram: np.ndarray
    target_params: Dict[str, float]
    diagnostics: Dict[str, Any]
    window_stats: Dict[str, Any]


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run same-window diagnostic for P4 CNN gap analysis.")
    parser.add_argument("--config", required=True, type=str, help="Path to P4 multiscenario YAML config")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional scenario name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Optional diagnostic mode filter. Defaults to window_variance/ekf/cnn_fpga.",
    )
    parser.add_argument(
        "--reference-mode",
        default="static_linear",
        type=str,
        help="Closed-loop reference mode used to capture the raw windows.",
    )
    parser.add_argument(
        "--n-windows",
        default=96,
        type=int,
        help="Number of slow windows to capture per scenario for same-window analysis.",
    )
    return parser


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_config_with_bases(cfg_path: Path) -> Dict[str, Any]:
    cfg = load_yaml_config(cfg_path)
    base_ref = cfg.get("base_config")
    if not base_ref:
        return cfg
    base_cfg = _load_config_with_bases((cfg_path.parent / str(base_ref)).resolve())
    return _deep_merge(base_cfg, {key: value for key, value in cfg.items() if key != "base_config"})


def _normalize_entries(entries: Sequence[Dict[str, Any]], filter_names: Sequence[str], field_name: str) -> List[Dict[str, Any]]:
    normalized = [dict(item) for item in entries]
    if not normalized:
        raise ValueError(f"No {field_name} configured.")
    if not filter_names:
        return normalized
    selected = [item for item in normalized if str(item.get("name")) in {str(name) for name in filter_names}]
    if not selected:
        raise ValueError(f"No {field_name} matched filter: {list(filter_names)}")
    return selected


def _build_mode_config(base_config: Dict[str, Any], mode_entry: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(base_config)
    slow_cfg = cfg.setdefault("slow_loop", {})
    artifact_cfg = slow_cfg.setdefault("model_artifact", {})
    infer_cfg = slow_cfg.setdefault("inference_service", {})
    if "slow_loop_mode" in mode_entry:
        slow_cfg["mode"] = str(mode_entry["slow_loop_mode"]).lower()
    if "artifact_selector" in mode_entry:
        artifact_cfg["artifact_selector"] = str(mode_entry["artifact_selector"])
    if "artifact_path" in mode_entry:
        artifact_cfg["path"] = mode_entry["artifact_path"]
    if "inference_backend" in mode_entry:
        infer_cfg["backend"] = str(mode_entry["inference_backend"]).lower()
    if "inference_service_mode" in mode_entry:
        infer_cfg["mode"] = str(mode_entry["inference_service_mode"]).lower()
    if "overrides" in mode_entry:
        cfg = _deep_merge(cfg, dict(mode_entry["overrides"]))
    return cfg


def _build_run_config(base_config: Dict[str, Any], scenario_entry: Dict[str, Any], mode_entry: Dict[str, Any], seed: int) -> Dict[str, Any]:
    cfg = _build_mode_config(base_config, mode_entry)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["seed"] = int(seed)
    scenario_overrides = dict(scenario_entry.get("overrides", {}))
    if scenario_overrides:
        cfg = _deep_merge(cfg, scenario_overrides)
    return cfg


def _disable_runtime_failures(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = deepcopy(config)
    err_cfg = cfg.setdefault("error_model", {})
    err_cfg["dma_timeout_prob"] = 0.0
    err_cfg["axi_write_fail_prob"] = 0.0
    err_cfg["cnn_infer_fail_prob"] = 0.0
    slow_cfg = cfg.setdefault("slow_loop", {})
    service_cfg = slow_cfg.setdefault("inference_service", {})
    # 中文注释：诊断阶段只比较算法本身，不引入额外的 subprocess 时延/失败源。
    service_cfg["mode"] = "inproc"
    if str(slow_cfg.get("mode", "")).lower() == "model_artifact" and "backend" not in service_cfg:
        service_cfg["backend"] = "artifact_npz"
    return cfg


def _prediction_from_runtime_output(params: DecoderRuntimeParams) -> Dict[str, float]:
    metadata = dict(params.metadata)
    raw = metadata.get("prediction")
    if raw is None:
        raw = metadata.get("fixed_prediction")
    if raw is None:
        raise KeyError("runtime output does not contain prediction metadata")
    return {
        "sigma": float(raw.get("sigma", 0.0)),
        "mu_q": float(raw.get("mu_q", 0.0)),
        "mu_p": float(raw.get("mu_p", 0.0)),
        "theta_deg": float(raw.get("theta_deg", 0.0)),
    }


def _capture_reference_windows(config: Dict[str, Any], n_windows: int) -> List[CapturedWindow]:
    if n_windows <= 0:
        raise ValueError("n_windows must be positive")

    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", 1234))
    noise_provider = _build_mock_noise_provider(config)
    driver = FPGADriver.from_config(config, noise_provider=noise_provider, seed=seed + 7)
    slow_loop = SlowLoopRuntime.from_config(config, seed=seed + 31)
    host_latency = LatencyInjector.from_config(config, seed=seed + 43)

    captured: List[CapturedWindow] = []
    try:
        driver.start()
        driver.reset_histogram()

        scheduler_cfg = driver.scheduler_config
        requested_cfg = deepcopy(config)
        requested_cfg.setdefault("timing", {})
        requested_cfg["timing"]["n_slow_updates"] = int(n_windows)
        requested_cfg["timing"]["n_fast_cycles"] = int(max(n_windows * scheduler_cfg.resolved_window_stride, scheduler_cfg.window_size))
        max_cycles = _required_fast_cycles(requested_cfg, scheduler_cfg)

        next_slow_start_time_us = 0.0
        slow_job: HILSlowJob | None = None
        awaiting_commit_ack = False
        captures_started = 0

        for _ in range(max_cycles):
            driver.advance_cycles(1)
            status = driver.read_status()
            current_epoch = int(status.get("epoch_id", driver.read_epoch()))
            current_time_us = float(status.get("time_us", driver.read_time_us()))

            if awaiting_commit_ack and bool(status.get("commit_ack", False)):
                awaiting_commit_ack = False

            if slow_job is not None and current_time_us >= slow_job.ready_time_us:
                proposed = slow_loop(slow_job.readout.window, slow_job.active_params)
                driver.stage_params(proposed, metadata={"job_id": slow_job.job_id})
                driver.commit_bank(
                    commit_epoch=current_epoch + scheduler_cfg.commit_delay_cycles,
                    ack_delay_us=slow_job.latency.commit_ack_us,
                    metadata={
                        "job_id": slow_job.job_id,
                        "window_id": slow_job.readout.window.window_id,
                    },
                )
                awaiting_commit_ack = True
                slow_job = None

            if (
                slow_job is None
                and not awaiting_commit_ack
                and not driver.has_pending_commit()
                and current_time_us >= next_slow_start_time_us
                and driver.histogram_available()
                and captures_started < n_windows
            ):
                readout = driver.read_histogram()
                payload = dict(readout.window.payload)
                captured.append(
                    CapturedWindow(
                        window_id=int(readout.window.window_id),
                        start_epoch=int(readout.window.start_epoch),
                        end_epoch=int(readout.window.end_epoch),
                        ready_time_us=float(readout.window.ready_time_us),
                        histogram=np.asarray(payload.get("histogram"), dtype=np.float32),
                        target_params={
                            "sigma": float(payload.get("target_params", {}).get("sigma", 0.0)),
                            "mu_q": float(payload.get("target_params", {}).get("mu_q", 0.0)),
                            "mu_p": float(payload.get("target_params", {}).get("mu_p", 0.0)),
                            "theta_deg": float(payload.get("target_params", {}).get("theta_deg", 0.0)),
                        },
                        diagnostics=dict(payload.get("diagnostics", {})),
                        window_stats=dict(payload.get("window_stats", {})),
                    )
                )
                captures_started += 1

                active_params = driver.read_active_params()
                latency = host_latency.sample_slow_update()
                slow_job = HILSlowJob(
                    job_id=captures_started,
                    readout=readout,
                    active_params=active_params,
                    started_epoch=current_epoch,
                    started_time_us=current_time_us,
                    ready_time_us=current_time_us
                    + latency.dma_us
                    + latency.preprocess_us
                    + latency.inference_us
                    + latency.writeback_us,
                    latency=latency,
                )
                next_slow_start_time_us = current_time_us + scheduler_cfg.slow_update_period_us

            if captures_started >= n_windows and slow_job is None and not awaiting_commit_ack and not driver.has_pending_commit():
                break

        if len(captured) < n_windows:
            raise RuntimeError(f"Only captured {len(captured)} windows, expected {n_windows}")
        return captured
    finally:
        slow_loop.close()
        driver.close()


def _compute_oracle_track(config: Dict[str, Any], windows: Sequence[CapturedWindow]) -> List[DecoderRuntimeParams]:
    runtime = SlowLoopRuntime.from_config(config, seed=int(config.get("experiment", {}).get("seed", 1234)) + 111)
    oracle_params: List[DecoderRuntimeParams] = []
    active_params = DecoderRuntimeParams.identity()
    try:
        for item in windows:
            prediction = NoisePrediction(
                sigma=float(item.target_params["sigma"]),
                mu_q=float(item.target_params["mu_q"]),
                mu_p=float(item.target_params["mu_p"]),
                theta_deg=float(item.target_params["theta_deg"]),
                source="oracle_target",
                metadata={"window_id": int(item.window_id)},
            )
            mapped = runtime.param_mapper.map_prediction(prediction, previous_params=active_params)
            oracle_params.append(mapped.copy())
            active_params = mapped.copy()
        return oracle_params
    finally:
        runtime.close()


def _run_mode_same_window(config: Dict[str, Any], windows: Sequence[CapturedWindow], oracle_track: Sequence[DecoderRuntimeParams]) -> Dict[str, Any]:
    runtime = SlowLoopRuntime.from_config(config, seed=int(config.get("experiment", {}).get("seed", 1234)) + 211)
    active_params = DecoderRuntimeParams.identity()

    per_window_rows: List[Dict[str, Any]] = []
    pred_matrix: List[List[float]] = []
    target_matrix: List[List[float]] = []
    k_rmse_list: List[float] = []
    b_rmse_list: List[float] = []
    max_gain_list: List[float] = []
    bias_norm_list: List[float] = []

    try:
        for idx, item in enumerate(windows):
            window = WindowFrame(
                window_id=int(item.window_id),
                start_epoch=int(item.start_epoch),
                end_epoch=int(item.end_epoch),
                ready_time_us=float(item.ready_time_us),
                payload={
                    "histogram": np.asarray(item.histogram, dtype=np.float32),
                    "target_params": dict(item.target_params),
                    "diagnostics": dict(item.diagnostics),
                    "window_stats": dict(item.window_stats),
                },
            )
            output = runtime(window, active_params)
            active_params = output.copy()
            prediction = _prediction_from_runtime_output(output)
            oracle_params = oracle_track[idx]
            k_rmse = float(np.sqrt(np.mean((output.K - oracle_params.K) ** 2)))
            b_rmse = float(np.sqrt(np.mean((output.b - oracle_params.b) ** 2)))
            aggressiveness = dict(output.metadata.get("param_aggressiveness", {}))

            target = item.target_params
            pred_vec = [prediction["sigma"], prediction["mu_q"], prediction["mu_p"], prediction["theta_deg"]]
            target_vec = [target["sigma"], target["mu_q"], target["mu_p"], target["theta_deg"]]
            pred_matrix.append(pred_vec)
            target_matrix.append(target_vec)
            k_rmse_list.append(k_rmse)
            b_rmse_list.append(b_rmse)
            max_gain_list.append(float(aggressiveness["max_gain"]))
            bias_norm_list.append(float(aggressiveness["bias_norm"]))

            per_window_rows.append(
                {
                    "window_id": int(item.window_id),
                    "target_sigma": float(target["sigma"]),
                    "target_mu_q": float(target["mu_q"]),
                    "target_mu_p": float(target["mu_p"]),
                    "target_theta_deg": float(target["theta_deg"]),
                    "pred_sigma": float(prediction["sigma"]),
                    "pred_mu_q": float(prediction["mu_q"]),
                    "pred_mu_p": float(prediction["mu_p"]),
                    "pred_theta_deg": float(prediction["theta_deg"]),
                    "abs_err_sigma": float(abs(prediction["sigma"] - target["sigma"])),
                    "abs_err_mu_q": float(abs(prediction["mu_q"] - target["mu_q"])),
                    "abs_err_mu_p": float(abs(prediction["mu_p"] - target["mu_p"])),
                    "abs_err_theta_deg": float(abs(prediction["theta_deg"] - target["theta_deg"])),
                    "window_ler_reference": float(item.diagnostics.get("window_ler", 0.0)),
                    "overflow_ratio_reference": float(item.diagnostics.get("overflow_ratio", 0.0)),
                    "hist_sat_ratio_reference": float(item.diagnostics.get("histogram_input_saturation_ratio", 0.0)),
                    "k_rmse_to_oracle": k_rmse,
                    "b_rmse_to_oracle": b_rmse,
                    "max_gain": float(aggressiveness["max_gain"]),
                    "bias_norm": float(aggressiveness["bias_norm"]),
                }
            )

        pred_array = np.asarray(pred_matrix, dtype=float)
        target_array = np.asarray(target_matrix, dtype=float)
        abs_err = np.abs(pred_array - target_array)
        sq_err = (pred_array - target_array) ** 2
        labels = ("sigma", "mu_q", "mu_p", "theta_deg")

        def _sign_mismatch_rate(target_col: np.ndarray, pred_col: np.ndarray) -> float:
            mask = np.abs(target_col) > 1.0e-6
            if not np.any(mask):
                return 0.0
            mismatch = np.sign(target_col[mask]) != np.sign(np.where(np.abs(pred_col[mask]) > 1.0e-6, pred_col[mask], 0.0))
            zero_pred = np.abs(pred_col[mask]) <= 1.0e-6
            mismatch = np.logical_or(mismatch, zero_pred)
            return float(np.mean(mismatch.astype(float)))

        summary = {
            "n_windows": len(per_window_rows),
            "per_label": {
                label: {
                    "mae": float(np.mean(abs_err[:, idx])),
                    "rmse": float(np.sqrt(np.mean(sq_err[:, idx]))),
                }
                for idx, label in enumerate(labels)
            },
            "mean_target": {
                label: float(np.mean(target_array[:, idx])) for idx, label in enumerate(labels)
            },
            "mean_prediction": {
                label: float(np.mean(pred_array[:, idx])) for idx, label in enumerate(labels)
            },
            "mu_sign_mismatch_rate": {
                "mu_q": _sign_mismatch_rate(target_array[:, 1], pred_array[:, 1]),
                "mu_p": _sign_mismatch_rate(target_array[:, 2], pred_array[:, 2]),
            },
            "k_rmse_to_oracle_mean": float(np.mean(k_rmse_list)),
            "b_rmse_to_oracle_mean": float(np.mean(b_rmse_list)),
            "max_gain_mean": float(np.mean(max_gain_list)),
            "bias_norm_mean": float(np.mean(bias_norm_list)),
            "window_rows": per_window_rows,
        }
        return summary
    finally:
        runtime.close()


def _write_report(
    path: Path,
    *,
    summary: Dict[str, Any],
    compare_modes: Sequence[str],
    reference_mode: str,
    n_windows: int,
) -> None:
    lines = [
        "# P4 CNN Gap Diagnostic",
        "",
        f"- reference_mode: `{reference_mode}`",
        f"- compare_modes: `{', '.join(compare_modes)}`",
        f"- n_windows_per_scenario: `{n_windows}`",
        "",
    ]

    for scenario_name, scenario_summary in summary["scenarios"].items():
        lines.extend(
            [
                f"## {scenario_name}",
                "",
                "| Mode | sigma MAE | mu_q MAE | mu_p MAE | theta MAE | K RMSE vs Oracle | b RMSE vs Oracle | Mean Max Gain | Mean Bias Norm |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode_name in compare_modes:
            mode_summary = scenario_summary["modes"][mode_name]
            lines.append(
                f"| {mode_name} | "
                f"{mode_summary['per_label']['sigma']['mae']:.6f} | "
                f"{mode_summary['per_label']['mu_q']['mae']:.6f} | "
                f"{mode_summary['per_label']['mu_p']['mae']:.6f} | "
                f"{mode_summary['per_label']['theta_deg']['mae']:.6f} | "
                f"{mode_summary['k_rmse_to_oracle_mean']:.6f} | "
                f"{mode_summary['b_rmse_to_oracle_mean']:.6f} | "
                f"{mode_summary['max_gain_mean']:.6f} | "
                f"{mode_summary['bias_norm_mean']:.6f} |"
            )
        lines.append("")
        lines.extend(
            [
                "| Mode | mean pred mu_q | mean pred mu_p | mu_q sign mismatch | mu_p sign mismatch |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for mode_name in compare_modes:
            mode_summary = scenario_summary["modes"][mode_name]
            lines.append(
                f"| {mode_name} | "
                f"{mode_summary['mean_prediction']['mu_q']:.6f} | "
                f"{mode_summary['mean_prediction']['mu_p']:.6f} | "
                f"{mode_summary['mu_sign_mismatch_rate']['mu_q']:.6f} | "
                f"{mode_summary['mu_sign_mismatch_rate']['mu_p']:.6f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _arg_parser().parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_config_with_bases(cfg_path)

    benchmark_cfg = dict(cfg.get("p4_benchmark", {}))
    merged_base = {k: v for k, v in cfg.items() if k != "p4_benchmark"}
    all_modes = _normalize_entries(benchmark_cfg.get("modes", []), [], "modes")
    scenario_entries = _normalize_entries(benchmark_cfg.get("scenarios", []), args.scenario, "scenarios")
    compare_mode_names = list(args.mode) if args.mode else list(DEFAULT_COMPARE_MODES)
    compare_modes = _normalize_entries(all_modes, compare_mode_names, "modes")
    reference_mode_entry = _normalize_entries(all_modes, [args.reference_mode], "modes")[0]

    base_seed = int(merged_base.get("experiment", {}).get("seed", 1234))
    scenario_seed_stride = int(benchmark_cfg.get("protocol", {}).get("scenario_seed_stride", 1000))

    run_root = ensure_dir(merged_base.get("paths", {}).get("output_root", "runs"))
    out_dir = ensure_dir(
        run_root / "p4_diagnostic" / f"p4_gap_diag_{now_tag()}_{config_hash(cfg)}"
    )

    top_summary: Dict[str, Any] = {
        "config_path": str(cfg_path),
        "reference_mode": str(args.reference_mode),
        "compare_modes": [str(item.get("name")) for item in compare_modes],
        "n_windows_per_scenario": int(args.n_windows),
        "scenarios": {},
    }

    for scenario_idx, scenario_entry in enumerate(scenario_entries):
        scenario_name = str(scenario_entry.get("name"))
        scenario_seed = base_seed + scenario_idx * scenario_seed_stride
        scenario_dir = ensure_dir(out_dir / scenario_name)

        ref_cfg = _disable_runtime_failures(
            _build_run_config(merged_base, scenario_entry, reference_mode_entry, scenario_seed)
        )
        ref_cfg.setdefault("experiment", {})
        ref_cfg["experiment"]["name"] = f"p4_gap_diag_capture_{scenario_name}"
        captured = _capture_reference_windows(ref_cfg, args.n_windows)

        histograms = np.stack([item.histogram for item in captured], axis=0)
        targets = np.asarray(
            [[item.target_params["sigma"], item.target_params["mu_q"], item.target_params["mu_p"], item.target_params["theta_deg"]] for item in captured],
            dtype=np.float32,
        )
        np.savez_compressed(
            scenario_dir / "reference_windows.npz",
            histograms=histograms,
            targets=targets,
            window_ids=np.asarray([item.window_id for item in captured], dtype=np.int32),
        )

        oracle_cfg = _disable_runtime_failures(
            _build_run_config(merged_base, scenario_entry, reference_mode_entry, scenario_seed)
        )
        oracle_track = _compute_oracle_track(oracle_cfg, captured)

        scenario_summary: Dict[str, Any] = {
            "reference_mode": str(args.reference_mode),
            "n_windows": len(captured),
            "capture_target_means": {
                "sigma": float(np.mean(targets[:, 0])),
                "mu_q": float(np.mean(targets[:, 1])),
                "mu_p": float(np.mean(targets[:, 2])),
                "theta_deg": float(np.mean(targets[:, 3])),
            },
            "modes": {},
        }

        for mode_entry in compare_modes:
            mode_name = str(mode_entry.get("name"))
            mode_cfg = _disable_runtime_failures(
                _build_run_config(merged_base, scenario_entry, mode_entry, scenario_seed)
            )
            mode_cfg.setdefault("experiment", {})
            mode_cfg["experiment"]["name"] = f"p4_gap_diag_{scenario_name}_{mode_name}"
            mode_summary = _run_mode_same_window(mode_cfg, captured, oracle_track)
            scenario_summary["modes"][mode_name] = {
                key: value for key, value in mode_summary.items() if key != "window_rows"
            }
            save_json(scenario_dir / f"{mode_name}_window_summary.json", mode_summary)

        top_summary["scenarios"][scenario_name] = scenario_summary

    save_json(out_dir / "summary.json", top_summary)
    _write_report(
        out_dir / "report.md",
        summary=top_summary,
        compare_modes=[str(item.get("name")) for item in compare_modes],
        reference_mode=str(args.reference_mode),
        n_windows=int(args.n_windows),
    )
    print(f"P4 gap diagnostic complete. Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
