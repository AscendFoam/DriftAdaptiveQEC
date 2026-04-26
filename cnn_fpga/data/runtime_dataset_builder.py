"""Build runtime-consistent window datasets from the real fast/slow-loop emulation chain."""

# 中文说明：
# - 与 `dataset_builder.py` 的静态合成分布不同，这个脚本直接跑快慢回路仿真，
#   从真实窗口链路中采集 histogram。
# - 当前版本支持两类残差标签：
#   1. `residual_mu`: 学 teacher 在 `mu_q / mu_p` 上的偏差；
#   2. `residual_b`: 直接学 teacher 映射后 `b_q / b_p` 的闭环残差。
# - 输入不再局限于“3 帧 histogram 堆叠”，而是可以叠加：
#   histogram 历史、histogram 差分、teacher 当前预测、teacher 当前参数以及 teacher 时间差分。

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from cnn_fpga.benchmark.run_hil_suite import HILSlowJob, _build_mock_noise_provider, _required_fast_cycles
from cnn_fpga.hwio import FPGADriver
from cnn_fpga.decoder import ParamMapper
from cnn_fpga.decoder.param_mapper import NoisePrediction
from cnn_fpga.runtime import LatencyContext, LatencyInjector, SlowLoopRuntime
from cnn_fpga.runtime.feature_builder import (
    RuntimeFeatureConfig,
    build_feature_sample,
    feature_channel_names,
    infer_scalar_feature_dim,
    scalar_feature_names,
)
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.runtime.scheduler import WindowFrame
from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, save_json


LABEL_NAMES_RESIDUAL_MU = ("mu_q", "mu_p")
LABEL_NAMES_RESIDUAL_B = ("b_q", "b_p")


def _log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    try:
        print(f"[runtime_dataset_builder {timestamp}] {message}", flush=True)
    except OSError:
        # Long-running jobs may outlive the attached stdout handle when launched
        # under wrappers/timeouts; logging must not crash dataset generation.
        pass


@dataclass(frozen=True)
class CapturedRuntimeWindow:
    """One runtime-consistent window captured from the real emulation chain."""

    scenario_name: str
    window_id: int
    start_epoch: int
    end_epoch: int
    ready_time_us: float
    histogram: np.ndarray
    target_params: Dict[str, float]
    diagnostics: Dict[str, Any]
    window_stats: Dict[str, Any]


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build runtime-consistent residual-mu dataset.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Optional scenario filter. Can be provided multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print build plan")
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


def _capture_reference_windows(
    config: Dict[str, Any],
    *,
    scenario_name: str,
    n_windows: int,
) -> List[CapturedRuntimeWindow]:
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", 1234))
    t0 = time.perf_counter()
    _log(f"[{scenario_name}] capture start: n_windows={n_windows}, seed={seed}")
    noise_provider = _build_mock_noise_provider(config)
    _log(f"[{scenario_name}] noise provider ready in {time.perf_counter() - t0:.2f}s")
    t_driver = time.perf_counter()
    driver = FPGADriver.from_config(config, noise_provider=noise_provider, seed=seed + 7)
    _log(f"[{scenario_name}] driver ready in {time.perf_counter() - t_driver:.2f}s")
    t_slow = time.perf_counter()
    slow_loop = SlowLoopRuntime.from_config(config, seed=seed + 31)
    _log(f"[{scenario_name}] slow loop ready in {time.perf_counter() - t_slow:.2f}s")
    t_latency = time.perf_counter()
    host_latency = LatencyInjector.from_config(config, seed=seed + 43)
    _log(f"[{scenario_name}] host latency ready in {time.perf_counter() - t_latency:.2f}s")

    captured: List[CapturedRuntimeWindow] = []
    try:
        t_start = time.perf_counter()
        driver.start()
        driver.reset_histogram()
        _log(f"[{scenario_name}] driver started/reset in {time.perf_counter() - t_start:.2f}s")

        scheduler_cfg = driver.scheduler_config
        requested_cfg = deepcopy(config)
        requested_cfg.setdefault("timing", {})
        requested_cfg["timing"]["n_slow_updates"] = int(n_windows)
        requested_cfg["timing"]["n_fast_cycles"] = int(max(n_windows * scheduler_cfg.resolved_window_stride, scheduler_cfg.window_size))
        max_cycles = _required_fast_cycles(requested_cfg, scheduler_cfg)
        _log(
            f"[{scenario_name}] scheduler window_size={scheduler_cfg.window_size}, "
            f"stride={scheduler_cfg.resolved_window_stride}, max_cycles={max_cycles}"
        )

        next_slow_start_time_us = 0.0
        slow_job: HILSlowJob | None = None
        awaiting_commit_ack = False
        captures_started = 0

        progress_interval = max(1, max_cycles // 4)
        loop_started = time.perf_counter()
        for cycle_idx in range(max_cycles):
            driver.advance_cycles(1)
            status = driver.read_status()
            current_epoch = int(status.get("epoch_id", driver.read_epoch()))
            current_time_us = float(status.get("time_us", driver.read_time_us()))

            if cycle_idx == 0:
                _log(
                    f"[{scenario_name}] first cycle ok: epoch={current_epoch}, "
                    f"time_us={current_time_us:.2f}"
                )
            elif (cycle_idx + 1) % progress_interval == 0:
                _log(
                    f"[{scenario_name}] progress cycles={cycle_idx + 1}/{max_cycles}, "
                    f"captures={captures_started}, pending_commit={driver.has_pending_commit()}, "
                    f"hist_ready={driver.histogram_available()}"
                )

            if awaiting_commit_ack and bool(status.get("commit_ack", False)):
                awaiting_commit_ack = False
                _log(f"[{scenario_name}] commit ack observed at epoch={current_epoch}")

            if slow_job is not None and current_time_us >= slow_job.ready_time_us:
                proposed = slow_loop(slow_job.readout.window, slow_job.active_params)
                driver.stage_params(proposed, metadata={"job_id": slow_job.job_id})
                driver.commit_bank(
                    commit_epoch=current_epoch + scheduler_cfg.commit_delay_cycles,
                    ack_delay_us=slow_job.latency.commit_ack_us,
                    metadata={"job_id": slow_job.job_id, "window_id": slow_job.readout.window.window_id},
                )
                awaiting_commit_ack = True
                _log(
                    f"[{scenario_name}] slow job {slow_job.job_id} finished for window "
                    f"{slow_job.readout.window.window_id} at epoch={current_epoch}"
                )
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
                    CapturedRuntimeWindow(
                        scenario_name=scenario_name,
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
                _log(
                    f"[{scenario_name}] captured window {readout.window.window_id} "
                    f"at epoch={current_epoch}, time_us={current_time_us:.2f}"
                )

                active_params = driver.read_active_params()
                latency = host_latency.sample_slow_update(
                    context=LatencyContext(
                        pending_windows=int(status.get("pending_dma_buffers", 0)),
                        slow_job_inflight=False,
                        pending_commit=bool(driver.has_pending_commit()),
                        recent_slow_budget_violations=0,
                    )
                )
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
                _log(
                    f"[{scenario_name}] slow job {slow_job.job_id} scheduled, "
                    f"ready_time_us={slow_job.ready_time_us:.2f}"
                )

            if captures_started >= n_windows and slow_job is None and not awaiting_commit_ack and not driver.has_pending_commit():
                break

        _log(
            f"[{scenario_name}] capture finished with {len(captured)} windows in "
            f"{time.perf_counter() - loop_started:.2f}s"
        )

        if len(captured) < n_windows:
            raise RuntimeError(f"Only captured {len(captured)} windows, expected {n_windows}")
        return captured
    finally:
        slow_loop.close()
        driver.close()
        _log(f"[{scenario_name}] capture teardown complete")


def _make_teacher_mode_entry(teacher_mode: str) -> Dict[str, Any]:
    mode_name = str(teacher_mode).lower()
    if mode_name not in {"window_variance", "ekf", "ukf", "particle_filter"}:
        raise ValueError(f"Unsupported teacher_mode: {teacher_mode}")
    return {"name": mode_name, "slow_loop_mode": mode_name}


def _runtime_feature_config_from_dataset_cfg(runtime_dataset_cfg: Dict[str, Any]) -> RuntimeFeatureConfig:
    return RuntimeFeatureConfig(
        context_windows=int(runtime_dataset_cfg.get("context_windows", 1)),
        context_padding=str(runtime_dataset_cfg.get("context_padding", "edge_repeat")).lower(),
        include_histogram_deltas=bool(runtime_dataset_cfg.get("include_histogram_deltas", False)),
        include_teacher_prediction=bool(runtime_dataset_cfg.get("include_teacher_prediction", False)),
        include_teacher_params=bool(runtime_dataset_cfg.get("include_teacher_params", False)),
        include_teacher_deltas=bool(runtime_dataset_cfg.get("include_teacher_deltas", False)),
        teacher_prediction_layout=str(runtime_dataset_cfg.get("teacher_prediction_layout", "broadcast")).lower(),
        teacher_params_layout=str(runtime_dataset_cfg.get("teacher_params_layout", "broadcast")).lower(),
        teacher_deltas_layout=str(runtime_dataset_cfg.get("teacher_deltas_layout", "broadcast")).lower(),
        teacher_scalar_features=tuple(runtime_dataset_cfg.get("teacher_scalar_features", []) or []),
    )


def _label_names_for_semantics(label_semantics: str) -> tuple[str, str]:
    normalized = str(label_semantics).lower()
    if normalized == "residual_mu":
        return LABEL_NAMES_RESIDUAL_MU
    if normalized == "residual_b":
        return LABEL_NAMES_RESIDUAL_B
    raise ValueError(f"unsupported_label_semantics:{label_semantics}")


def _split_stats(values: np.ndarray) -> Dict[str, List[float]]:
    if values.shape[0] == 0:
        n_labels = int(values.shape[1]) if values.ndim > 1 else 0
        empty = [None] * n_labels
        return {
            "mean": empty,
            "std": empty,
            "min": empty,
            "max": empty,
        }
    return {
        "mean": values.mean(axis=0).astype(np.float64).tolist(),
        "std": values.std(axis=0).astype(np.float64).tolist(),
        "min": values.min(axis=0).astype(np.float64).tolist(),
        "max": values.max(axis=0).astype(np.float64).tolist(),
    }


def _split_indices(n_samples: int, train_ratio: float, val_ratio: float, rng: np.random.Generator):
    indices = rng.permutation(n_samples)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _save_split(
    out_dir: Path,
    split_name: str,
    histograms: np.ndarray,
    scalar_features: np.ndarray,
    labels: np.ndarray,
    label_names: Sequence[str],
    indices: np.ndarray,
    *,
    teacher_predictions: np.ndarray,
    target_params: np.ndarray,
    teacher_runtime_b: np.ndarray,
    target_runtime_b: np.ndarray,
    input_channel_names: Sequence[str],
    scenario_names: np.ndarray,
    window_ids: np.ndarray,
) -> Path:
    split_path = out_dir / f"{split_name}.npz"
    np.savez_compressed(
        split_path,
        histograms=histograms[indices],
        scalar_features=scalar_features[indices],
        labels=labels[indices],
        label_names=np.asarray(label_names),
        teacher_predictions=teacher_predictions[indices],
        target_params=target_params[indices],
        teacher_runtime_b=teacher_runtime_b[indices],
        target_runtime_b=target_runtime_b[indices],
        input_channel_names=np.asarray(input_channel_names),
        scenario_names=scenario_names[indices],
        window_ids=window_ids[indices],
    )
    return split_path


def _target_prediction_from_window(item: CapturedRuntimeWindow) -> NoisePrediction:
    return NoisePrediction(
        sigma=float(item.target_params["sigma"]),
        mu_q=float(item.target_params["mu_q"]),
        mu_p=float(item.target_params["mu_p"]),
        theta_deg=float(item.target_params["theta_deg"]),
        source="window_target",
        metadata={"window_id": int(item.window_id), "scenario_name": item.scenario_name},
    )


def main() -> int:
    args = _arg_parser().parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_config_with_bases(cfg_path)

    runtime_dataset_cfg = dict(cfg.get("runtime_dataset", {}))
    if not runtime_dataset_cfg:
        raise ValueError("Config must define runtime_dataset section.")

    merged_base = {k: v for k, v in cfg.items() if k != "runtime_dataset"}
    p4_cfg = dict(cfg.get("p4_benchmark", {}))
    scenarios_source = p4_cfg.get("scenarios", []) if bool(runtime_dataset_cfg.get("scenarios_from_p4", True)) else runtime_dataset_cfg.get("scenarios", [])
    scenarios = _normalize_entries(scenarios_source, args.scenario, "scenarios")

    reference_mode = str(runtime_dataset_cfg.get("reference_mode", "static_linear")).lower()
    teacher_mode = str(runtime_dataset_cfg.get("teacher_mode", "window_variance")).lower()
    label_semantics = str(runtime_dataset_cfg.get("label_semantics", "residual_mu")).lower()
    feature_cfg = _runtime_feature_config_from_dataset_cfg(runtime_dataset_cfg)
    label_names = _label_names_for_semantics(label_semantics)
    windows_per_scenario = int(runtime_dataset_cfg.get("windows_per_scenario", 128))
    split_cfg = dict(runtime_dataset_cfg.get("split", {"train": 0.8, "val": 0.1, "test": 0.1}))

    cfg_hash = config_hash(cfg)
    seed_base = int(merged_base.get("experiment", {}).get("seed", 1234))
    dataset_dir = ensure_dir(get_path(cfg, "dataset_dir", "artifacts/datasets/runtime_mu_residual"))
    scenario_seed_stride = int(runtime_dataset_cfg.get("scenario_seed_stride", 1000))
    input_channel_names = feature_channel_names(feature_cfg)
    scalar_names = scalar_feature_names(feature_cfg)

    plan = {
        "dataset_dir": str(dataset_dir),
        "config_hash": cfg_hash,
        "reference_mode": reference_mode,
        "teacher_mode": teacher_mode,
        "context_windows": feature_cfg.context_windows,
        "windows_per_scenario": windows_per_scenario,
        "n_scenarios": len(scenarios),
        "label_semantics": label_semantics,
        "input_channels": len(input_channel_names),
        "input_channel_names": input_channel_names,
        "scalar_feature_dim": infer_scalar_feature_dim(feature_cfg),
        "scalar_feature_names": scalar_names,
        "scenarios": [str(item.get("name")) for item in scenarios],
        "split": split_cfg,
    }
    _log(f"Runtime dataset build plan: {plan}")
    if args.dry_run:
        return 0

    reference_mode_entry = {"name": reference_mode, "slow_loop_mode": reference_mode}
    teacher_mode_entry = _make_teacher_mode_entry(teacher_mode)

    histograms: List[np.ndarray] = []
    scalar_features_all: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    teacher_predictions_all: List[np.ndarray] = []
    target_params_all: List[np.ndarray] = []
    teacher_runtime_b_all: List[np.ndarray] = []
    target_runtime_b_all: List[np.ndarray] = []
    scenario_names_all: List[str] = []
    window_ids_all: List[int] = []

    for scenario_idx, scenario_entry in enumerate(scenarios):
        scenario_name = str(scenario_entry.get("name"))
        scenario_seed = seed_base + scenario_idx * scenario_seed_stride
        _log(f"[{scenario_name}] scenario start ({scenario_idx + 1}/{len(scenarios)})")
        reference_cfg = _disable_runtime_failures(
            _build_run_config(merged_base, scenario_entry, reference_mode_entry, scenario_seed)
        )
        reference_cfg.setdefault("experiment", {})
        reference_cfg["experiment"]["name"] = f"runtime_dataset_capture_{scenario_name}"
        captured = _capture_reference_windows(reference_cfg, scenario_name=scenario_name, n_windows=windows_per_scenario)
        _log(f"[{scenario_name}] reference capture complete: {len(captured)} windows")

        teacher_cfg = _disable_runtime_failures(
            _build_run_config(merged_base, scenario_entry, teacher_mode_entry, scenario_seed)
        )
        teacher_cfg.setdefault("experiment", {})
        teacher_cfg["experiment"]["name"] = f"runtime_dataset_teacher_{scenario_name}"
        t_teacher = time.perf_counter()
        teacher_runtime = SlowLoopRuntime.from_config(teacher_cfg, seed=scenario_seed + 211)
        _log(f"[{scenario_name}] teacher runtime ready in {time.perf_counter() - t_teacher:.2f}s")
        param_mapper = ParamMapper.from_config(teacher_cfg)
        try:
            active_params = DecoderRuntimeParams.identity()
            histogram_history: List[np.ndarray] = []
            teacher_prediction_history: List[np.ndarray] = []
            teacher_param_history: List[np.ndarray] = []
            for item in captured:
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
                teacher_output = teacher_runtime(window, active_params)
                teacher_prediction = _prediction_from_runtime_output(teacher_output)
                teacher_prediction_vec = np.array(
                    [
                        teacher_prediction["sigma"],
                        teacher_prediction["mu_q"],
                        teacher_prediction["mu_p"],
                        teacher_prediction["theta_deg"],
                    ],
                    dtype=np.float32,
                )
                target_prediction = _target_prediction_from_window(item)
                target_output = param_mapper.map_prediction(target_prediction, previous_params=active_params)

                histogram_history.append(np.asarray(item.histogram, dtype=np.float32))
                teacher_prediction_history.append(teacher_prediction_vec)
                teacher_param_history.append(np.asarray(teacher_output.b, dtype=np.float32))
                feature_sample = build_feature_sample(
                    histogram_history,
                    feature_cfg,
                    teacher_prediction_history=teacher_prediction_history,
                    teacher_param_history=teacher_param_history,
                )

                if label_semantics == "residual_mu":
                    label_value = np.array(
                        [
                            float(item.target_params["mu_q"] - teacher_prediction["mu_q"]),
                            float(item.target_params["mu_p"] - teacher_prediction["mu_p"]),
                        ],
                        dtype=np.float32,
                    )
                elif label_semantics == "residual_b":
                    label_value = np.asarray(target_output.b - teacher_output.b, dtype=np.float32)
                else:  # pragma: no cover - protected by config validation above.
                    raise ValueError(f"unsupported_label_semantics:{label_semantics}")

                histograms.append(feature_sample.spatial_tensor)
                if feature_sample.scalar_features is None:
                    scalar_features_all.append(np.zeros((infer_scalar_feature_dim(feature_cfg),), dtype=np.float32))
                else:
                    scalar_features_all.append(np.asarray(feature_sample.scalar_features, dtype=np.float32).reshape(-1))
                labels.append(label_value)
                teacher_predictions_all.append(teacher_prediction_vec)
                target_params_all.append(
                    np.array(
                        [
                            item.target_params["sigma"],
                            item.target_params["mu_q"],
                            item.target_params["mu_p"],
                            item.target_params["theta_deg"],
                        ],
                        dtype=np.float32,
                    )
                )
                teacher_runtime_b_all.append(np.asarray(teacher_output.b, dtype=np.float32))
                target_runtime_b_all.append(np.asarray(target_output.b, dtype=np.float32))
                scenario_names_all.append(scenario_name)
                window_ids_all.append(int(item.window_id))
                active_params = teacher_output.copy()
        finally:
            teacher_runtime.close()
            _log(f"[{scenario_name}] teacher feature/label pass complete")

    histogram_array = np.stack(histograms, axis=0).astype(np.float32)
    scalar_feature_array = (
        np.stack(scalar_features_all, axis=0).astype(np.float32)
        if scalar_features_all
        else np.zeros((histogram_array.shape[0], 0), dtype=np.float32)
    )
    label_array = np.stack(labels, axis=0).astype(np.float32)
    teacher_array = np.stack(teacher_predictions_all, axis=0).astype(np.float32)
    target_array = np.stack(target_params_all, axis=0).astype(np.float32)
    teacher_runtime_b_array = np.stack(teacher_runtime_b_all, axis=0).astype(np.float32)
    target_runtime_b_array = np.stack(target_runtime_b_all, axis=0).astype(np.float32)
    scenario_names = np.asarray(scenario_names_all)
    window_ids = np.asarray(window_ids_all, dtype=np.int32)

    rng = np.random.default_rng(seed_base)
    train_idx, val_idx, test_idx = _split_indices(
        n_samples=histogram_array.shape[0],
        train_ratio=float(split_cfg.get("train", 0.8)),
        val_ratio=float(split_cfg.get("val", 0.1)),
        rng=rng,
    )

    train_path = _save_split(
        dataset_dir,
        "train",
        histogram_array,
        scalar_feature_array,
        label_array,
        label_names,
        train_idx,
        teacher_predictions=teacher_array,
        target_params=target_array,
        teacher_runtime_b=teacher_runtime_b_array,
        target_runtime_b=target_runtime_b_array,
        input_channel_names=input_channel_names,
        scenario_names=scenario_names,
        window_ids=window_ids,
    )
    val_path = _save_split(
        dataset_dir,
        "val",
        histogram_array,
        scalar_feature_array,
        label_array,
        label_names,
        val_idx,
        teacher_predictions=teacher_array,
        target_params=target_array,
        teacher_runtime_b=teacher_runtime_b_array,
        target_runtime_b=target_runtime_b_array,
        input_channel_names=input_channel_names,
        scenario_names=scenario_names,
        window_ids=window_ids,
    )
    test_path = _save_split(
        dataset_dir,
        "test",
        histogram_array,
        scalar_feature_array,
        label_array,
        label_names,
        test_idx,
        teacher_predictions=teacher_array,
        target_params=target_array,
        teacher_runtime_b=teacher_runtime_b_array,
        target_runtime_b=target_runtime_b_array,
        input_channel_names=input_channel_names,
        scenario_names=scenario_names,
        window_ids=window_ids,
    )

    split_stats = {
        "train": _split_stats(label_array[train_idx]),
        "val": _split_stats(label_array[val_idx]),
        "test": _split_stats(label_array[test_idx]),
        "all": _split_stats(label_array),
    }
    manifest = {
        "config_hash": cfg_hash,
        "config_path": str(cfg_path),
        "dataset_dir": str(dataset_dir),
        "reference_mode": reference_mode,
        "teacher_mode": teacher_mode,
        "label_semantics": label_semantics,
        "context_windows": feature_cfg.context_windows,
        "context_padding": feature_cfg.context_padding,
        "include_histogram_deltas": feature_cfg.include_histogram_deltas,
        "include_teacher_prediction": feature_cfg.include_teacher_prediction,
        "include_teacher_params": feature_cfg.include_teacher_params,
        "include_teacher_deltas": feature_cfg.include_teacher_deltas,
        "teacher_prediction_layout": feature_cfg.teacher_prediction_layout,
        "teacher_params_layout": feature_cfg.teacher_params_layout,
        "teacher_deltas_layout": feature_cfg.teacher_deltas_layout,
        "teacher_scalar_features": list(feature_cfg.teacher_scalar_features),
        "windows_per_scenario": windows_per_scenario,
        "n_samples_total": int(histogram_array.shape[0]),
        "histogram_shape": list(histogram_array.shape[1:]),
        "input_channel_names": input_channel_names,
        "scalar_feature_dim": int(scalar_feature_array.shape[1]),
        "scalar_feature_names": scalar_names,
        "label_names": list(label_names),
        "teacher_label_names": ["sigma", "mu_q", "mu_p", "theta_deg"],
        "teacher_runtime_param_names": ["b_q", "b_p"],
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "scenarios": [str(item.get("name")) for item in scenarios],
        "split": split_cfg,
        "label_stats": split_stats,
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    save_json(dataset_dir / "manifest.json", manifest)

    _log(f"Runtime dataset written to {dataset_dir}")
    _log(f"Manifest: {dataset_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
