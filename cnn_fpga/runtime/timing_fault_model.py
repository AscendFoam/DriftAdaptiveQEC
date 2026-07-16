"""T2.4.2 paired timing-fault stress model built on the real runtime scheduler.

The model exercises queueing and atomic-bank behavior on the same synthetic
physical traces.  It quantifies timing-induced changes in standard-binning
logical error rate (LER) and three availability definitions.  Results remain
software-model evidence; no target-board timestamp is inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.runtime.latency_injector import LatencyInjector, StageLatencySpec
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from cnn_fpga.runtime.scheduler import DualLoopScheduler, SchedulerConfig, WindowFrame
from cnn_fpga.utils.config import load_yaml_config, save_json
from physics.constants import LATTICE_CONST


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "cnn_fpga" / "config" / "hardware_hil.yaml"
DEFAULT_ARTIFACT = ROOT / "docs" / "t2_4_2_timing_fault_validation.json"
DEFAULT_CSV = ROOT / "docs" / "t2_4_2_timing_fault_validation.csv"
CONTRACT_ID = "T242-PAIRED-SCHEDULER-TIMING-FAULT-V1"
MODEL_SCOPE = "paired_synthetic_standard_binning_timing_stress_not_board_measurement"


@dataclass(frozen=True)
class TimingFaultScenario:
    name: str
    slow_mean_scale: float = 1.0
    slow_std_scale: float = 1.0
    fast_mean_us: float = 1.0
    fast_std_us: float = 0.12
    burst_epochs: tuple[int, ...] = ()
    burst_size: int = 0
    communication_pauses: tuple[tuple[int, int], ...] = ()
    inject_conflict_on_internal_stage: bool = False
    max_pending_windows: int = 2
    hold_last_on_fast_miss: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be nonempty")
        if self.slow_mean_scale <= 0 or self.slow_std_scale < 0:
            raise ValueError("slow latency scales must be positive/non-negative")
        if self.fast_mean_us < 0 or self.fast_std_us < 0:
            raise ValueError("fast latency mean/std must be non-negative")
        if self.burst_epochs and self.burst_size < 2:
            raise ValueError("scheduled input bursts require burst_size >= 2")
        if self.max_pending_windows <= 0:
            raise ValueError("max_pending_windows must be positive")
        for start, end in self.communication_pauses:
            if start < 1 or end <= start:
                raise ValueError("communication pauses must satisfy 1 <= start < end")


@dataclass(frozen=True)
class TimingStressConfig:
    n_cycles: int = 64_000
    seeds: tuple[int, ...] = (101, 211, 307, 401, 503, 607, 709, 811)
    measurement_noise_sigma: float = 0.10
    channel_noise_sigma: float = 0.34
    evaluation_warmup_windows: int = 2
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 24_201

    def __post_init__(self) -> None:
        if self.n_cycles <= 0:
            raise ValueError("n_cycles must be positive")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("at least two unique seeds are required")
        if self.measurement_noise_sigma < 0 or self.channel_noise_sigma <= 0:
            raise ValueError("noise sigmas must be non-negative/positive")
        if self.evaluation_warmup_windows < 1:
            raise ValueError("evaluation_warmup_windows must be positive")
        if self.bootstrap_replicates < 1000:
            raise ValueError("bootstrap_replicates must be at least 1000")


def default_scenarios(n_cycles: int) -> tuple[TimingFaultScenario, ...]:
    if n_cycles < 24_000:
        raise ValueError(
            "default timing-fault scenarios require at least 24,000 cycles so "
            "both bursts and the communication pause occur inside the run"
        )
    first_burst = max(12_000, n_cycles // 3)
    second_burst = max(first_burst + 4_000, 2 * n_cycles // 3)
    pause_start = max(16_000, 3 * n_cycles // 8)
    pause_end = min(n_cycles - 2_000, pause_start + 16_000)
    return (
        TimingFaultScenario(name="reference"),
        TimingFaultScenario(
            name="jitter_deadline",
            slow_mean_scale=24.0,
            slow_std_scale=12.0,
            fast_mean_us=1.60,
            fast_std_us=0.45,
        ),
        TimingFaultScenario(
            name="input_burst",
            burst_epochs=(first_burst, second_burst),
            burst_size=4,
            max_pending_windows=12,
        ),
        TimingFaultScenario(
            name="communication_pause",
            communication_pauses=((pause_start, pause_end),),
        ),
        TimingFaultScenario(
            name="parameter_conflict",
            inject_conflict_on_internal_stage=True,
        ),
        TimingFaultScenario(
            name="fifo_overflow",
            burst_epochs=(first_burst, second_burst),
            burst_size=8,
            max_pending_windows=2,
        ),
        TimingFaultScenario(
            name="combined",
            slow_mean_scale=12.0,
            slow_std_scale=10.0,
            fast_mean_us=1.45,
            fast_std_us=0.45,
            burst_epochs=(first_burst, second_burst),
            burst_size=8,
            communication_pauses=((pause_start, pause_end),),
            inject_conflict_on_internal_stage=True,
            max_pending_windows=2,
        ),
    )


def _sha256_sources(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        rel = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _paired_bootstrap(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired bootstrap requires a 1D array with at least two values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "probability_positive": float(np.mean(means > 0.0)),
        "probability_negative": float(np.mean(means < 0.0)),
    }


def _physical_trace(
    n_cycles: int,
    *,
    window_stride: int,
    seed: int,
    measurement_noise_sigma: float,
    channel_noise_sigma: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, n_cycles + 1, dtype=np.float64)
    block = np.floor((epochs - 1) / (4 * window_stride)).astype(np.int64)
    levels = np.asarray([0.00, 0.92, -0.92, 0.68, -0.68, 0.98, -0.45, 0.45])
    drift = levels[np.mod(block, levels.size)]
    drift = drift + 0.10 * np.sin(2 * np.pi * epochs / (11 * window_stride))
    random_walk = np.cumsum(rng.normal(0.0, 1.5e-4, size=n_cycles))
    random_walk -= np.mean(random_walk)
    drift = drift + np.clip(random_walk, -0.12, 0.12)
    observed = drift + rng.normal(0.0, measurement_noise_sigma, size=n_cycles)
    channel_noise = rng.normal(0.0, channel_noise_sigma, size=n_cycles)
    return {
        "drift": drift.astype(np.float64),
        "observed": observed.astype(np.float64),
        "channel_noise": channel_noise.astype(np.float64),
    }


def _is_communication_available(epoch: int, scenario: TimingFaultScenario) -> bool:
    return not any(start <= epoch < end for start, end in scenario.communication_pauses)


def _build_latency_injector(
    config: Mapping[str, Any],
    scenario: TimingFaultScenario,
    *,
    seed: int,
) -> LatencyInjector:
    latency = config["latency_model"]

    def slow_stage(prefix: str) -> StageLatencySpec:
        return StageLatencySpec(
            mean_us=float(latency[f"{prefix}_mean_us"]) * scenario.slow_mean_scale,
            std_us=float(latency[f"{prefix}_std_us"]) * scenario.slow_std_scale,
        )

    return LatencyInjector(
        dma=slow_stage("dma"),
        preprocess=slow_stage("preprocess"),
        inference=slow_stage("inference"),
        writeback=slow_stage("writeback"),
        commit_ack=slow_stage("commit_ack"),
        fast_cycle=StageLatencySpec(
            mean_us=scenario.fast_mean_us,
            std_us=scenario.fast_std_us,
        ),
        seed=seed + 90_001,
    )


def _slow_estimator(window: WindowFrame, active: DecoderRuntimeParams) -> DecoderRuntimeParams:
    if "observed_mean" not in window.payload or "n_valid" not in window.payload:
        raise ValueError("window payload must contain observed_mean and n_valid")
    estimate = float(window.payload["observed_mean"])
    n_valid = int(window.payload["n_valid"])
    if not math.isfinite(estimate) or n_valid <= 0:
        raise ValueError("window estimate must be finite with positive n_valid")
    metadata = dict(active.metadata)
    metadata.update(
        {
            "source_window_id": window.window_id,
            "source_window_end_epoch": window.end_epoch,
            "source_window_ready_time_us": window.ready_time_us,
            "n_valid": n_valid,
            "estimator": "observed_window_mean_no_hidden_truth",
        }
    )
    return DecoderRuntimeParams(
        K=active.K.copy(),
        b=np.asarray([estimate, 0.0], dtype=np.float64),
        metadata=metadata,
    )


def _window_payload(
    observed_prefix: np.ndarray,
    *,
    epoch: int,
    window_size: int,
) -> dict[str, Any]:
    start = max(0, epoch - window_size)
    total = observed_prefix[epoch] - observed_prefix[start]
    n_valid = epoch - start
    return {
        "observed_mean": float(total / n_valid),
        "n_valid": int(n_valid),
        "source_end_epoch": int(epoch),
        "schema": "observed_only_t242_v1",
    }


def simulate_scenario(
    scenario: TimingFaultScenario,
    *,
    config: TimingStressConfig,
    seed: int,
    yaml_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_config = dict(yaml_config or load_yaml_config(DEFAULT_CONFIG))
    base_scheduler = SchedulerConfig.from_config(source_config)
    scheduler_config = SchedulerConfig(
        t_fast_us=base_scheduler.t_fast_us,
        window_size=base_scheduler.window_size,
        slow_update_period_us=base_scheduler.slow_update_period_us,
        window_stride=base_scheduler.resolved_window_stride,
        max_pending_windows=scenario.max_pending_windows,
        commit_delay_cycles=base_scheduler.commit_delay_cycles,
        fast_path_budget_us=base_scheduler.fast_path_budget_us,
        slow_path_budget_us=base_scheduler.slow_path_budget_us,
        guard_cycles_after_commit=base_scheduler.guard_cycles_after_commit,
        window_deadline_us=base_scheduler.resolved_window_deadline_us,
    )
    trace = _physical_trace(
        config.n_cycles,
        window_stride=scheduler_config.resolved_window_stride,
        seed=seed,
        measurement_noise_sigma=config.measurement_noise_sigma,
        channel_noise_sigma=config.channel_noise_sigma,
    )
    observed_prefix = np.concatenate(([0.0], np.cumsum(trace["observed"], dtype=np.float64)))
    scheduler = DualLoopScheduler(
        scheduler_config,
        param_bank=ParamBank(),
        latency_injector=_build_latency_injector(source_config, scenario, seed=seed),
        slow_path_fn=_slow_estimator,
    )

    applied_estimate = np.zeros(config.n_cycles, dtype=np.float64)
    fast_on_time = np.ones(config.n_cycles, dtype=np.bool_)
    parameter_fresh = np.zeros(config.n_cycles, dtype=np.bool_)
    active_versions = np.zeros(config.n_cycles, dtype=np.int64)
    fast_latencies: list[float] = []
    slow_latencies: list[float] = []
    window_ages: list[float] = []
    event_counts: Counter[str] = Counter()
    last_applied = 0.0
    conflict_attempts = 0
    external_update_applied = 0
    burst_epochs = set(scenario.burst_epochs)
    deadline_cycles = int(math.floor(scheduler_config.resolved_window_deadline_us / scheduler_config.t_fast_us))

    for epoch in range(1, config.n_cycles + 1):
        will_emit = (
            epoch >= scheduler_config.window_size
            and (epoch - scheduler_config.window_size) % scheduler_config.resolved_window_stride == 0
        )
        payload = (
            _window_payload(observed_prefix, epoch=epoch, window_size=scheduler_config.window_size)
            if will_emit
            else None
        )
        events = scheduler.tick(
            window_payload=payload,
            communication_available=_is_communication_available(epoch, scenario),
        )
        if epoch in burst_epochs:
            burst_payload = _window_payload(
                observed_prefix,
                epoch=epoch,
                window_size=scheduler_config.window_size,
            )
            events.extend(
                scheduler.inject_window_burst(
                    [dict(burst_payload, burst_member=index) for index in range(scenario.burst_size)]
                )
            )

        if scenario.inject_conflict_on_internal_stage and any(
            event.kind == "params_staged" for event in events
        ):
            conflict_attempts += 1
            active = scheduler.param_bank.read_active()
            conflicting = DecoderRuntimeParams(
                K=active.K,
                b=np.asarray([active.b[0] + 1.25, 0.0]),
                metadata={"external_writer": True},
            )
            pending, conflict_events = scheduler.stage_external_update(
                conflicting,
                commit_epoch=epoch + scheduler_config.commit_delay_cycles,
                metadata={"external_writer": True},
            )
            if pending is not None:
                external_update_applied += 1
            events.extend(conflict_events)

        for event in events:
            event_counts[event.kind] += 1
            if event.kind == "slow_update_started":
                slow_latencies.append(float(event.details["latency"]["total_us"]))
            elif event.kind == "slow_update_finished":
                window_ages.append(float(event.details["window_age_us"]))

        fast_latency = float(scheduler.last_fast_cycle_latency_us or 0.0)
        fast_latencies.append(fast_latency)
        missed = any(event.kind == "fast_budget_violation" for event in events)
        fast_on_time[epoch - 1] = not missed
        current = scheduler.param_bank.read_active()
        requested = float(current.b[0])
        if missed and scenario.hold_last_on_fast_miss:
            applied_estimate[epoch - 1] = last_applied
        else:
            applied_estimate[epoch - 1] = requested
            last_applied = requested
        active_versions[epoch - 1] = scheduler.param_bank.active_version
        source_epoch = current.metadata.get("source_window_end_epoch")
        parameter_fresh[epoch - 1] = (
            source_epoch is not None and 0 <= epoch - int(source_epoch) <= deadline_cycles
        )

    residual = trace["drift"] + trace["channel_noise"] - applied_estimate
    lattice_index = np.floor(residual / float(LATTICE_CONST) + 0.5).astype(np.int64)
    logical_error = np.mod(lattice_index, 2).astype(np.bool_)
    evaluation_start = min(
        config.n_cycles - 1,
        config.evaluation_warmup_windows * scheduler_config.resolved_window_stride,
    )
    eval_slice = slice(evaluation_start, None)
    end_to_end_available = fast_on_time & parameter_fresh
    eval_errors = logical_error[eval_slice]
    eval_available = end_to_end_available[eval_slice]
    available_errors = eval_errors[eval_available]
    unavailable_errors = eval_errors[~eval_available]
    version_diff = np.diff(active_versions)

    return {
        "scenario": scenario.name,
        "seed": int(seed),
        "n_cycles": config.n_cycles,
        "evaluation_start_epoch": evaluation_start + 1,
        "model_scope": MODEL_SCOPE,
        "target_hardware_measured": False,
        "metrics": {
            "logical_error_rate": float(np.mean(eval_errors)),
            "logical_error_count": int(np.count_nonzero(eval_errors)),
            "evaluated_cycles": int(eval_errors.size),
            "fast_action_availability": float(np.mean(fast_on_time[eval_slice])),
            "fresh_parameter_availability": float(np.mean(parameter_fresh[eval_slice])),
            "end_to_end_control_availability": float(np.mean(eval_available)),
            "ler_when_available": None
            if available_errors.size == 0
            else float(np.mean(available_errors)),
            "ler_when_unavailable": None
            if unavailable_errors.size == 0
            else float(np.mean(unavailable_errors)),
            "maximum_pending_windows": int(
                max(
                    [0]
                    + [
                        int(event.details.get("queue_depth_after", 0))
                        for event in scheduler.event_log
                        if event.kind == "window_ready"
                    ]
                )
            ),
        },
        "latency_quantiles_us": {
            "fast_cycle": _quantiles(fast_latencies),
            "slow_service": _quantiles(slow_latencies),
            "window_age_at_finish": _quantiles(window_ages),
        },
        "event_counts": dict(sorted(event_counts.items())),
        "integrity": {
            "active_version_monotonic": bool(np.all(version_diff >= 0)),
            "maximum_version_step": int(np.max(version_diff, initial=0)),
            "conflict_attempts": int(conflict_attempts),
            "external_conflicting_updates_applied": int(external_update_applied),
            "all_arrays_finite": bool(
                np.all(np.isfinite(residual))
                and np.all(np.isfinite(applied_estimate))
                and np.all(np.isfinite(trace["drift"]))
            ),
            "slow_estimator_uses_hidden_truth": False,
        },
        "scheduler_snapshot": scheduler.snapshot(),
    }


def _aggregate_results(
    per_seed: Sequence[Mapping[str, Any]],
    *,
    reference_by_seed: Mapping[int, Mapping[str, Any]],
    config: TimingStressConfig,
    scenario_index: int,
) -> dict[str, Any]:
    scenario = str(per_seed[0]["scenario"])
    seeds = np.asarray([int(row["seed"]) for row in per_seed], dtype=np.int64)
    ler = np.asarray([row["metrics"]["logical_error_rate"] for row in per_seed], dtype=np.float64)
    availability = np.asarray(
        [row["metrics"]["end_to_end_control_availability"] for row in per_seed],
        dtype=np.float64,
    )
    fast_availability = np.asarray(
        [row["metrics"]["fast_action_availability"] for row in per_seed],
        dtype=np.float64,
    )
    fresh_availability = np.asarray(
        [row["metrics"]["fresh_parameter_availability"] for row in per_seed],
        dtype=np.float64,
    )
    reference_ler = np.asarray(
        [reference_by_seed[int(seed)]["metrics"]["logical_error_rate"] for seed in seeds],
        dtype=np.float64,
    )
    reference_availability = np.asarray(
        [
            reference_by_seed[int(seed)]["metrics"]["end_to_end_control_availability"]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    reference_fast_availability = np.asarray(
        [
            reference_by_seed[int(seed)]["metrics"]["fast_action_availability"]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    reference_fresh_availability = np.asarray(
        [
            reference_by_seed[int(seed)]["metrics"]["fresh_parameter_availability"]
            for seed in seeds
        ],
        dtype=np.float64,
    )
    event_totals: Counter[str] = Counter()
    for row in per_seed:
        event_totals.update({key: int(value) for key, value in row["event_counts"].items()})
    return {
        "scenario": scenario,
        "seeds": seeds.tolist(),
        "logical_error_rate": {
            "mean": float(np.mean(ler)),
            "min": float(np.min(ler)),
            "max": float(np.max(ler)),
        },
        "end_to_end_control_availability": {
            "mean": float(np.mean(availability)),
            "min": float(np.min(availability)),
            "max": float(np.max(availability)),
        },
        "fast_action_availability": {
            "mean": float(np.mean(fast_availability)),
            "min": float(np.min(fast_availability)),
            "max": float(np.max(fast_availability)),
        },
        "fresh_parameter_availability": {
            "mean": float(np.mean(fresh_availability)),
            "min": float(np.min(fresh_availability)),
            "max": float(np.max(fresh_availability)),
        },
        "paired_ler_minus_reference": _paired_bootstrap(
            ler - reference_ler,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed + 2 * scenario_index,
        ),
        "paired_availability_minus_reference": _paired_bootstrap(
            availability - reference_availability,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed + 2 * scenario_index + 1,
        ),
        "paired_fast_action_availability_minus_reference": _paired_bootstrap(
            fast_availability - reference_fast_availability,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed + 100 + 2 * scenario_index,
        ),
        "paired_fresh_parameter_availability_minus_reference": _paired_bootstrap(
            fresh_availability - reference_fresh_availability,
            replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed + 101 + 2 * scenario_index,
        ),
        "event_totals": dict(sorted(event_totals.items())),
    }


def run_timing_fault_validation(
    config: TimingStressConfig | None = None,
    *,
    scenarios: Sequence[TimingFaultScenario] | None = None,
    yaml_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or TimingStressConfig()
    source_config = dict(yaml_config or load_yaml_config(DEFAULT_CONFIG))
    scenario_list = tuple(scenarios or default_scenarios(cfg.n_cycles))
    if not scenario_list or scenario_list[0].name != "reference":
        raise ValueError("first scenario must be reference")
    if len({scenario.name for scenario in scenario_list}) != len(scenario_list):
        raise ValueError("scenario names must be unique")

    per_seed: list[dict[str, Any]] = []
    by_scenario: dict[str, list[dict[str, Any]]] = {scenario.name: [] for scenario in scenario_list}
    for scenario in scenario_list:
        for seed in cfg.seeds:
            result = simulate_scenario(
                scenario,
                config=cfg,
                seed=seed,
                yaml_config=source_config,
            )
            per_seed.append(result)
            by_scenario[scenario.name].append(result)

    reference_by_seed = {
        int(row["seed"]): row for row in by_scenario["reference"]
    }
    aggregates = [
        _aggregate_results(
            by_scenario[scenario.name],
            reference_by_seed=reference_by_seed,
            config=cfg,
            scenario_index=index,
        )
        for index, scenario in enumerate(scenario_list)
    ]
    aggregate_map = {row["scenario"]: row for row in aggregates}

    expected_events = {
        "jitter_deadline": ("fast_budget_violation", "slow_budget_violation", "window_deadline_miss"),
        "input_burst": ("input_burst",),
        "communication_pause": (
            "communication_pause_started",
            "communication_pause_ended",
            "window_deadline_miss",
        ),
        "parameter_conflict": ("parameter_update_conflict",),
        "fifo_overflow": ("input_burst", "fifo_overflow", "window_dropped"),
        "combined": (
            "fast_budget_violation",
            "slow_budget_violation",
            "input_burst",
            "communication_pause_started",
            "communication_pause_ended",
            "parameter_update_conflict",
            "fifo_overflow",
            "window_deadline_miss",
        ),
    }
    event_gates = {
        scenario: all(
            row["event_counts"].get(event, 0) > 0
            for row in by_scenario[scenario]
            for event in events
        )
        for scenario, events in expected_events.items()
    }
    all_integrity = all(
        row["integrity"]["active_version_monotonic"]
        and row["integrity"]["maximum_version_step"] <= 1
        and row["integrity"]["all_arrays_finite"]
        and not row["integrity"]["slow_estimator_uses_hidden_truth"]
        for row in per_seed
    )
    conflict_integrity = all(
        row["integrity"]["conflict_attempts"]
        == row["event_counts"].get("parameter_update_conflict", 0)
        and row["integrity"]["external_conflicting_updates_applied"] == 0
        for scenario in ("parameter_conflict", "combined")
        for row in by_scenario[scenario]
    )
    combined_ler = aggregate_map["combined"]["paired_ler_minus_reference"]
    combined_availability = aggregate_map["combined"]["paired_availability_minus_reference"]
    conflict_is_neutral = all(
        row["metrics"]["logical_error_rate"]
        == reference_by_seed[int(row["seed"])]["metrics"]["logical_error_rate"]
        and row["metrics"]["end_to_end_control_availability"]
        == reference_by_seed[int(row["seed"])]["metrics"][
            "end_to_end_control_availability"
        ]
        for row in by_scenario["parameter_conflict"]
    )
    gates = {
        "all_scenarios_all_seeds_executed": len(per_seed)
        == len(scenario_list) * len(cfg.seeds),
        "physical_traces_are_paired_by_seed": all(
            [row["seed"] for row in by_scenario[scenario.name]] == list(cfg.seeds)
            for scenario in scenario_list
        ),
        "jitter_and_deadline_detected": event_gates.get("jitter_deadline", False),
        "input_burst_detected": event_gates.get("input_burst", False),
        "communication_pause_detected": event_gates.get("communication_pause", False),
        "parameter_conflict_detected_and_rejected": event_gates.get("parameter_conflict", False)
        and conflict_integrity,
        "isolated_rejected_conflict_is_fail_closed_neutral": conflict_is_neutral,
        "fifo_overflow_detected_with_drop_provenance": event_gates.get("fifo_overflow", False),
        "combined_faults_all_detected": event_gates.get("combined", False),
        "combined_ler_increase_ci_positive": combined_ler["ci_low"] > 0.0,
        "combined_availability_drop_ci_negative": combined_availability["ci_high"] < 0.0,
        "state_and_numeric_integrity": all_integrity,
        "results_are_not_target_board_measurements": all(
            row["target_hardware_measured"] is False for row in per_seed
        ),
    }
    implementation_sha256 = _sha256_sources(
        [
            ROOT / "cnn_fpga" / "runtime" / "timing_fault_model.py",
            ROOT / "cnn_fpga" / "runtime" / "scheduler.py",
            ROOT / "cnn_fpga" / "runtime" / "param_bank.py",
        ]
    )
    return {
        "contract_id": CONTRACT_ID,
        "task_id": "T2.4.2",
        "model_scope": MODEL_SCOPE,
        "target_hardware_measured": False,
        "source_config": "cnn_fpga/config/hardware_hil.yaml",
        "implementation_sha256": implementation_sha256,
        "config": asdict(cfg),
        "scenarios": [asdict(scenario) for scenario in scenario_list],
        "per_seed_results": per_seed,
        "aggregates": aggregates,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "claim_boundary": {
            "allowed": "paired software-model timing stress with LER and availability impact",
            "forbidden": [
                "target-board measured latency or availability",
                "hard-real-time safety closure",
                "quantum experiment LER",
                "device-calibrated communication or pulse timing",
            ],
        },
    }


def write_per_seed_csv(path: Path, result: Mapping[str, Any]) -> None:
    """Write a flat, auditable per-seed table alongside the full JSON record."""

    event_names = sorted(
        {
            name
            for row in result["per_seed_results"]
            for name in row["event_counts"]
        }
    )
    metric_names = (
        "logical_error_rate",
        "logical_error_count",
        "evaluated_cycles",
        "fast_action_availability",
        "fresh_parameter_availability",
        "end_to_end_control_availability",
        "ler_when_available",
        "ler_when_unavailable",
        "maximum_pending_windows",
    )
    fields = ["scenario", "seed", "n_cycles", *metric_names]
    fields.extend(f"event_{name}" for name in event_names)
    fields.extend(
        [
            "conflict_attempts",
            "external_conflicting_updates_applied",
            "active_version_monotonic",
            "target_hardware_measured",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["per_seed_results"]:
            flat: dict[str, Any] = {
                "scenario": row["scenario"],
                "seed": row["seed"],
                "n_cycles": row["n_cycles"],
                **{name: row["metrics"][name] for name in metric_names},
                **{
                    f"event_{name}": row["event_counts"].get(name, 0)
                    for name in event_names
                },
                "conflict_attempts": row["integrity"]["conflict_attempts"],
                "external_conflicting_updates_applied": row["integrity"][
                    "external_conflicting_updates_applied"
                ],
                "active_version_monotonic": row["integrity"][
                    "active_version_monotonic"
                ],
                "target_hardware_measured": row["target_hardware_measured"],
            }
            writer.writerow(flat)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--cycles", type=int, default=TimingStressConfig.n_cycles)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in TimingStressConfig.seeds),
    )
    parser.add_argument("--bootstrap", type=int, default=TimingStressConfig.bootstrap_replicates)
    args = parser.parse_args(argv)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    result = run_timing_fault_validation(
        TimingStressConfig(
            n_cycles=args.cycles,
            seeds=seeds,
            bootstrap_replicates=args.bootstrap,
        )
    )
    save_json(args.artifact, result)
    write_per_seed_csv(args.csv, result)
    print(json.dumps({
        "status": result["status"],
        "artifact": str(args.artifact),
        "csv": str(args.csv),
        "gates": result["gates"],
        "aggregates": result["aggregates"],
    }, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
