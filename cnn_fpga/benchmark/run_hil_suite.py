"""Run P3 HIL suite through driver/backend abstraction."""

# 中文说明：
# - 该脚本现在把 HIL 主流程拆成“FPGA backend + host slow-loop runtime”两层。
# - mock 后端负责连续快回路、DMA 窗口输出和参数 bank 切换；主脚本负责轮询、推理、写回。

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from cnn_fpga.hwio import DMAReadout, FPGADriver
from cnn_fpga.runtime import LatencyInjector, LatencySample, SchedulerConfig, SlowLoopRuntime
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams
from cnn_fpga.utils.config import config_hash, ensure_dir, get_path, load_yaml_config, now_tag, save_json


@dataclass
class HILSlowJob:
    """Host-side slow-loop job created from one DMA histogram window."""

    job_id: int
    readout: DMAReadout
    active_params: DecoderRuntimeParams
    started_epoch: int
    started_time_us: float
    ready_time_us: float
    latency: LatencySample


def _backend_event_to_dict(event: object) -> Dict:
    if hasattr(event, "to_dict"):
        return getattr(event, "to_dict")()
    raise TypeError(f"Unsupported backend event type: {type(event)!r}")


def _slow_job_to_dict(job: HILSlowJob) -> Dict:
    return {
        "job_id": job.job_id,
        "readout": job.readout.to_dict(),
        "active_params": job.active_params.to_dict(),
        "started_epoch": job.started_epoch,
        "started_time_us": job.started_time_us,
        "ready_time_us": job.ready_time_us,
        "latency": job.latency.to_dict(),
    }


def _build_mock_noise_provider(config: Dict):
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", 1234))
    rng = np.random.default_rng(seed + 17)

    signal_cfg = config.get("mock_signal", {})
    kind = str(signal_cfg.get("type", "random_walk")).lower()

    sigma_state = float(signal_cfg.get("sigma0", 0.24))
    mu_q_state = float(signal_cfg.get("mu_q0", 0.0))
    mu_p_state = float(signal_cfg.get("mu_p0", 0.0))

    sigma_min = float(signal_cfg.get("sigma_min", 0.05))
    sigma_max = float(signal_cfg.get("sigma_max", 0.65))
    sigma_step_std = float(signal_cfg.get("sigma_step_std", 0.004))
    sigma_alpha = float(signal_cfg.get("sigma_alpha", 0.0005))
    sigma_amplitude = float(signal_cfg.get("sigma_amplitude", 0.04))
    sigma_frequency = float(signal_cfg.get("sigma_frequency", 0.015))

    mu_step_std = float(signal_cfg.get("mu_step_std", 0.002))
    theta_deg0 = float(signal_cfg.get("theta_deg0", 0.0))
    theta_amplitude_deg = float(signal_cfg.get("theta_amplitude_deg", 6.0))
    theta_frequency = float(signal_cfg.get("theta_frequency", 0.01))
    theta_step_std_deg = float(signal_cfg.get("theta_step_std_deg", 0.25))
    jump_round = int(signal_cfg.get("jump_round", 0))
    delta_sigma = float(signal_cfg.get("delta_sigma", 0.0))
    delta_mu_q = float(signal_cfg.get("delta_mu_q", 0.0))
    delta_mu_p = float(signal_cfg.get("delta_mu_p", 0.0))
    delta_theta_deg = float(signal_cfg.get("delta_theta_deg", 0.0))
    static_mu_q = float(signal_cfg.get("mu_q0", 0.0))
    static_mu_p = float(signal_cfg.get("mu_p0", 0.0))

    def _provider(epoch_id: int, _time_us: float) -> Dict:
        nonlocal sigma_state, mu_q_state, mu_p_state
        t = float(epoch_id - 1)

        if kind in {"static", "constant"}:
            sigma_value = sigma_state
            mu_q_value = static_mu_q
            mu_p_value = static_mu_p
            theta_value = theta_deg0
        elif kind == "linear":
            sigma_value = sigma_state + sigma_alpha * t
            mu_q_value = mu_q_state
            mu_p_value = mu_p_state
            theta_value = float(theta_deg0 + rng.normal(0.0, theta_step_std_deg))
        elif kind == "step":
            triggered = jump_round > 0 and epoch_id >= jump_round
            sigma_value = sigma_state + (delta_sigma if triggered else 0.0)
            mu_q_value = mu_q_state + (delta_mu_q if triggered else 0.0)
            mu_p_value = mu_p_state + (delta_mu_p if triggered else 0.0)
            theta_value = theta_deg0 + (delta_theta_deg if triggered else 0.0)
        elif kind in {"sin", "sine", "periodic"}:
            sigma_value = sigma_state + sigma_amplitude * np.sin(2.0 * np.pi * sigma_frequency * t)
            mu_q_value = mu_q_state
            mu_p_value = mu_p_state
            theta_value = float(
                theta_deg0
                + theta_amplitude_deg * np.sin(2.0 * np.pi * theta_frequency * t)
                + rng.normal(0.0, theta_step_std_deg)
            )
        else:
            sigma_state = float(sigma_state + rng.normal(0.0, sigma_step_std))
            sigma_value = sigma_state
            mu_q_state = float(mu_q_state + rng.normal(0.0, mu_step_std))
            mu_p_state = float(mu_p_state + rng.normal(0.0, mu_step_std))
            mu_q_value = mu_q_state
            mu_p_value = mu_p_state
            theta_value = float(
                theta_deg0
                + theta_amplitude_deg * np.sin(2.0 * np.pi * theta_frequency * t)
                + rng.normal(0.0, theta_step_std_deg)
            )

        sigma_value = float(np.clip(sigma_value, sigma_min, sigma_max))

        return {
            "sigma": sigma_value,
            "mu_q": float(mu_q_value),
            "mu_p": float(mu_p_value),
            "theta_deg": theta_value,
            "metadata": {"signal_type": kind},
        }

    return _provider


def _extract_reason_counts(failed_events: List[Dict]) -> Dict[str, int]:
    reason_counts: Dict[str, int] = {}
    for event in failed_events:
        reason = str(event.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return reason_counts


def _estimate_fast_stats(driver: FPGADriver) -> Dict[str, float]:
    spec = driver.backend.latency_injector.fast_cycle
    mean = float(spec.mean_us)
    if spec.distribution.lower() in {"constant", "fixed"} or spec.std_us == 0.0:
        return {
            "mean_us": mean,
            "p95_us": mean,
            "p99_us": mean,
        }
    return {
        "mean_us": mean,
        "p95_us": float(mean + 1.645 * spec.std_us),
        "p99_us": float(mean + 2.326 * spec.std_us),
    }


def _required_fast_cycles(config: Dict, scheduler_cfg: SchedulerConfig) -> int:
    timing_cfg = config.get("timing", {})
    requested_fast_cycles = int(timing_cfg.get("n_fast_cycles", 200000))
    requested_slow_updates = int(timing_cfg.get("n_slow_updates", 0))
    if requested_slow_updates <= 0:
        return requested_fast_cycles

    cycles_per_slow = int(np.ceil(scheduler_cfg.slow_update_period_us / scheduler_cfg.t_fast_us))
    cycles_for_window_target = scheduler_cfg.window_size + max(0, requested_slow_updates - 1) * scheduler_cfg.resolved_window_stride
    cycles_for_slow_target = requested_slow_updates * cycles_per_slow
    return max(requested_fast_cycles, cycles_for_slow_target, cycles_for_window_target)


def run_hil_session(config: Dict, run_dir: Path) -> Dict:
    experiment = config.get("experiment", {})
    seed = int(experiment.get("seed", 1234))

    hil_cfg = config.get("hil", {})
    backend_name = str(hil_cfg.get("backend", "mock")).lower()
    noise_provider = _build_mock_noise_provider(config) if backend_name == "mock" else None
    driver = FPGADriver.from_config(config, noise_provider=noise_provider, seed=seed + 7)
    slow_loop = SlowLoopRuntime.from_config(config, seed=seed + 31)
    host_latency = LatencyInjector.from_config(config, seed=seed + 43)
    try:
        driver.start()
        driver.reset_histogram()

        scheduler_cfg = driver.scheduler_config
        n_fast_cycles = _required_fast_cycles(config, scheduler_cfg)
        next_slow_start_time_us = 0.0
        slow_job: HILSlowJob | None = None
        slow_job_counter = 0
        slow_update_budget_violations = 0
        awaiting_commit_ack = False

        backend_events: List[Dict] = []
        host_events: List[Dict] = []
        slow_started: List[Dict] = []
        slow_finished: List[Dict] = []
        slow_failed: List[Dict] = []
        commit_requests: List[Dict] = []
        commit_acks: List[Dict] = []

        for _ in range(n_fast_cycles):
            for event in driver.advance_cycles(1):
                backend_events.append(_backend_event_to_dict(event))

            status = driver.read_status()
            current_epoch = int(status.get("epoch_id", driver.read_epoch()))
            current_time_us = float(status.get("time_us", driver.read_time_us()))

            if awaiting_commit_ack and bool(status.get("commit_ack", False)):
                awaiting_commit_ack = False
                commit_acks.append(
                    {
                        "epoch_id": current_epoch,
                        "time_us": current_time_us,
                        "active_bank": status.get("active_bank"),
                    }
                )

            if slow_job is not None and current_time_us >= slow_job.ready_time_us:
                try:
                    proposed = slow_loop(slow_job.readout.window, slow_job.active_params)
                except Exception as exc:  # pragma: no cover - defensive path
                    error_reason = getattr(exc, "reason", str(exc))
                    failure = {
                        "job_id": slow_job.job_id,
                        "window_id": slow_job.readout.window.window_id,
                        "epoch_id": current_epoch,
                        "time_us": current_time_us,
                        "reason": error_reason,
                    }
                    slow_failed.append(failure)
                    host_events.append({"kind": "slow_update_failed", **failure})
                else:
                    driver.stage_params(proposed, metadata={"job_id": slow_job.job_id})
                    commit_meta = driver.commit_bank(
                        commit_epoch=current_epoch + scheduler_cfg.commit_delay_cycles,
                        ack_delay_us=slow_job.latency.commit_ack_us,
                        metadata={
                            "job_id": slow_job.job_id,
                            "window_id": slow_job.readout.window.window_id,
                        },
                    )
                    finished = {
                        **_slow_job_to_dict(slow_job),
                        "completed_epoch": current_epoch,
                        "completed_time_us": current_time_us,
                        "proposed_params": proposed.to_dict(),
                        "commit": commit_meta,
                    }
                    slow_finished.append(finished)
                    commit_requests.append(
                        {
                            "job_id": slow_job.job_id,
                            "window_id": slow_job.readout.window.window_id,
                            "epoch_id": current_epoch,
                            "time_us": current_time_us,
                            **commit_meta,
                        }
                    )
                    host_events.append({"kind": "slow_update_finished", **finished})
                    awaiting_commit_ack = True
                slow_job = None

            if (
                slow_job is None
                and not awaiting_commit_ack
                and not driver.has_pending_commit()
                and current_time_us >= next_slow_start_time_us
                and driver.histogram_available()
            ):
                readout = driver.read_histogram()
                active_params = driver.read_active_params()
                latency = host_latency.sample_slow_update()
                ready_time_us = (
                    current_time_us
                    + latency.dma_us
                    + latency.preprocess_us
                    + latency.inference_us
                    + latency.writeback_us
                )
                slow_job_counter += 1
                slow_job = HILSlowJob(
                    job_id=slow_job_counter,
                    readout=readout,
                    active_params=active_params,
                    started_epoch=current_epoch,
                    started_time_us=current_time_us,
                    ready_time_us=ready_time_us,
                    latency=latency,
                )
                next_slow_start_time_us = current_time_us + scheduler_cfg.slow_update_period_us
                started = _slow_job_to_dict(slow_job)
                slow_started.append(started)
                host_events.append({"kind": "slow_update_started", **started})
                if latency.total_us > scheduler_cfg.slow_path_budget_us:
                    slow_update_budget_violations += 1
                    host_events.append(
                        {
                            "kind": "slow_budget_violation",
                            "job_id": slow_job.job_id,
                            "latency_us": latency.total_us,
                            "budget_us": scheduler_cfg.slow_path_budget_us,
                            "epoch_id": current_epoch,
                            "time_us": current_time_us,
                        }
                    )

        save_json(
            run_dir / "hil_events.json",
            {
                "backend_events": backend_events,
                "host_events": host_events,
                "commit_requests": commit_requests,
                "commit_acks": commit_acks,
            },
        )

        slow_total = [float(item["latency"]["total_us"]) for item in slow_started]
        if slow_total:
            slow_array = np.asarray(slow_total, dtype=float)
            slow_mean = float(np.mean(slow_array))
            slow_p95 = float(np.percentile(slow_array, 95))
            slow_p99 = float(np.percentile(slow_array, 99))
        else:
            slow_mean = 0.0
            slow_p95 = 0.0
            slow_p99 = 0.0

        if hasattr(driver.backend, "latency_injector"):
            fast_stats = _estimate_fast_stats(driver)
        else:
            fast_stats = {"mean_us": None, "p95_us": None, "p99_us": None}

        reason_counts = _extract_reason_counts(slow_failed)
        backend_event_counts: Dict[str, int] = {}
        for event in backend_events:
            kind = str(event["kind"])
            backend_event_counts[kind] = backend_event_counts.get(kind, 0) + 1

        final_snapshot = driver.snapshot()
        fast_loop_summary = dict(final_snapshot.get("fast_loop", {}))
        summary = {
            "backend": backend_name,
            "requested_n_slow_updates": int(config.get("timing", {}).get("n_slow_updates", 0)),
            "n_fast_cycles": n_fast_cycles,
            "n_windows_ready": int(backend_event_counts.get("window_ready", 0)),
            "n_slow_updates_started": len(slow_started),
            "n_slow_updates_finished": len(slow_finished),
            "n_slow_updates_failed": len(slow_failed),
            "n_commits_requested": len(commit_requests),
            "n_commit_acks_observed": len(commit_acks),
            "n_commits_applied": int(backend_event_counts.get("commit_applied", 0)),
            "slow_update_budget_us": scheduler_cfg.slow_path_budget_us,
            "fast_cycle_budget_us": scheduler_cfg.fast_path_budget_us,
            "slow_update_mean_us": slow_mean,
            "slow_update_p95_us": slow_p95,
            "slow_update_p99_us": slow_p99,
            "slow_update_violation_rate": 0.0
            if not slow_started
            else float(slow_update_budget_violations / len(slow_started)),
            "fast_cycle_mean_us": fast_stats["mean_us"],
            "fast_cycle_p95_us": fast_stats["p95_us"],
            "fast_cycle_p99_us": fast_stats["p99_us"],
            "fast_cycle_violation_rate": float(getattr(driver.backend, "fast_cycle_budget_violations", 0) / max(1, n_fast_cycles)),
            "window_drop_count": int(final_snapshot.get("dropped_dma_buffers", 0)),
            "pending_windows_at_end": int(final_snapshot.get("dma", {}).get("pending_buffers", 0)),
            "dma_timeouts": int(reason_counts.get("dma_timeout", 0)),
            "axi_write_failures": int(reason_counts.get("axi_write_fail", 0)),
            "cnn_infer_failures": int(reason_counts.get("cnn_infer_fail", 0)),
            "failure_reason_counts": reason_counts,
            "artifact_path": None
            if getattr(slow_loop, "artifact_predictor", None) is None
            else str(slow_loop.artifact_predictor.artifact_path),
            "inference_service_mode": str(config.get("slow_loop", {}).get("inference_service", {}).get("mode", "inproc")),
            "final_ler": fast_loop_summary.get("final_logical_error_rate"),
            "overflow_rate": fast_loop_summary.get("overflow_rate"),
            "overflow_any_rate": fast_loop_summary.get("overflow_any_rate"),
            "histogram_input_saturation_rate": fast_loop_summary.get("histogram_input_saturation_rate"),
            "correction_saturation_rate": fast_loop_summary.get("correction_saturation_rate"),
            "correction_clip_saturation_rate": fast_loop_summary.get("correction_clip_saturation_rate"),
            "correction_fixed_point_saturation_rate": fast_loop_summary.get("correction_fixed_point_saturation_rate"),
            "aggressive_param_rate": fast_loop_summary.get("aggressive_param_rate"),
            "aggressive_param_gain_rate": fast_loop_summary.get("aggressive_param_gain_rate"),
            "aggressive_param_bias_rate": fast_loop_summary.get("aggressive_param_bias_rate"),
            "aggressive_param_correction_rate": fast_loop_summary.get("aggressive_param_correction_rate"),
            "dominant_overflow_source": fast_loop_summary.get("dominant_overflow_source"),
            "final_snapshot": final_snapshot,
        }
        save_json(run_dir / "hil_summary.json", summary)
        return summary
    finally:
        slow_loop.close()
        driver.close()


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HIL suite scaffold.")
    parser.add_argument("--config", required=True, type=str, help="Path to YAML config")
    return parser


def _fmt_metric(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6e}"


def main() -> int:
    args = _arg_parser().parse_args()
    config = load_yaml_config(args.config)
    cfg_hash = config_hash(config)

    out_root = ensure_dir(get_path(config, "output_root", "runs"))
    name = str(config.get("experiment", {}).get("name", "hil_suite"))
    run_dir = ensure_dir(out_root / "hil_suite" / f"{name}_{now_tag()}_{cfg_hash}")

    # 中文注释：执行 driver/backend 结构下的 HIL 会话，并把摘要与事件日志写入 run_dir。
    summary = run_hil_session(config, run_dir=run_dir)
    print("HIL suite complete.")
    print(f"run_dir={run_dir}")
    print(f"n_slow_updates_started={summary['n_slow_updates_started']}, n_commits_applied={summary['n_commits_applied']}")
    print(f"slow_update_violation_rate={_fmt_metric(summary['slow_update_violation_rate'])}")
    print(f"fast_cycle_violation_rate={_fmt_metric(summary['fast_cycle_violation_rate'])}")
    print(
        f"dma_timeouts={summary['dma_timeouts']}, "
        f"axi_write_failures={summary['axi_write_failures']}, "
        f"cnn_infer_failures={summary['cnn_infer_failures']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
