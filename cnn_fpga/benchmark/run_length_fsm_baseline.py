"""T3.2.5 run-length event-FSM and parameter-bank baseline validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import csv
import hashlib
import itertools
import json
from math import isfinite
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from cnn_fpga.benchmark.continuous_adaptive_map import _mean_interval
from cnn_fpga.runtime.param_bank import DecoderRuntimeParams, ParamBank
from cnn_fpga.runtime.run_length_fsm import (
    FALLBACK,
    LEAKAGE_HOLD,
    NORMAL,
    X_RECOVERY,
    Z_RECOVERY,
    RunLengthFSMConfig,
    RunLengthFSMInput,
    RunLengthParameterBankFSM,
    RunLengthParameterTable,
)
from physics.drift_processes import DriftState
from physics.syndrome_stream import SyndromeStream, SyndromeStreamConfig, generate_syndrome_stream


ROOT = Path(__file__).resolve().parents[2]
CONTROLLERS = ("static_safe_normal", "memoryless_event", "run_length_fsm", "truth_oracle")
TARGET_EVENT_MODES = (X_RECOVERY, Z_RECOVERY, LEAKAGE_HOLD)
WRITE_COST = 0.002


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class RunLengthBaselineDescriptor:
    task_id: str = "T3.2.5"
    family: str = "deterministic_event_controller_baseline"
    online_inputs: tuple[str, ...] = (
        "residual_qp",
        "observed_x_z_g_e_leakage",
        "quadrature_phase_bit",
        "valid_crc_fresh_deadline_flags",
    )
    hidden_truth_inputs: tuple[str, ...] = ()
    evaluator_truth_fields: tuple[str, ...] = (
        "leakage_kind",
        "recovery_depth_before_action",
        "recovery_quadrature",
    )
    action_contract: str = "normal_x-recovery_z-recovery_leakage-hold_fallback_parameter-bank_mode"
    primary_metric: str = "event_control_cost_plus_atomic_parameter_write_cost"
    logical_error_metric: bool = False
    fixed_point_or_rtl: bool = False
    target_hardware_measured: bool = False
    evidence_scope: str = "protocol_aligned_synthetic_syndrome_stream_software_validation"


RUN_LENGTH_DESCRIPTOR = RunLengthBaselineDescriptor()


@dataclass(frozen=True)
class EventScenario:
    scenario_id: str
    sigma_q: float
    sigma_p: float
    rho: float
    loss_gamma: float
    p_outlier: float
    outlier_scale: float
    depth_probability_scale: float
    recovery_probability: float
    readout_fidelity_g: float
    readout_fidelity_e: float
    base_leakage_probability: float
    higher_leakage_fraction: float
    higher_leakage_mean_duration: float
    burst_period: int = 0
    burst_width: int = 0
    health_fault_period: int = 0


def event_scenarios() -> tuple[EventScenario, ...]:
    return (
        EventScenario(
            scenario_id="persistent_recovery",
            sigma_q=0.46,
            sigma_p=0.41,
            rho=0.25,
            loss_gamma=0.015,
            p_outlier=0.015,
            outlier_scale=2.5,
            depth_probability_scale=0.55,
            recovery_probability=0.62,
            readout_fidelity_g=0.997,
            readout_fidelity_e=0.985,
            base_leakage_probability=0.0005,
            higher_leakage_fraction=0.4,
            higher_leakage_mean_duration=4.0,
        ),
        EventScenario(
            scenario_id="readout_false_positive",
            sigma_q=0.25,
            sigma_p=0.23,
            rho=-0.15,
            loss_gamma=0.005,
            p_outlier=0.002,
            outlier_scale=2.0,
            depth_probability_scale=0.09,
            recovery_probability=0.78,
            readout_fidelity_g=0.94,
            readout_fidelity_e=0.98,
            base_leakage_probability=0.0002,
            higher_leakage_fraction=0.3,
            higher_leakage_mean_duration=3.0,
        ),
        EventScenario(
            scenario_id="leakage_bursts",
            sigma_q=0.34,
            sigma_p=0.31,
            rho=0.1,
            loss_gamma=0.04,
            p_outlier=0.01,
            outlier_scale=3.0,
            depth_probability_scale=0.28,
            recovery_probability=0.70,
            readout_fidelity_g=0.992,
            readout_fidelity_e=0.98,
            base_leakage_probability=0.014,
            higher_leakage_fraction=0.8,
            higher_leakage_mean_duration=6.0,
        ),
        EventScenario(
            scenario_id="mixed_burst_health",
            sigma_q=0.32,
            sigma_p=0.38,
            rho=-0.35,
            loss_gamma=0.055,
            p_outlier=0.025,
            outlier_scale=4.0,
            depth_probability_scale=0.36,
            recovery_probability=0.67,
            readout_fidelity_g=0.975,
            readout_fidelity_e=0.975,
            base_leakage_probability=0.004,
            higher_leakage_fraction=0.65,
            higher_leakage_mean_duration=5.0,
            burst_period=257,
            burst_width=19,
            health_fault_period=503,
        ),
    )


@dataclass(frozen=True)
class RunLengthValidationConfig:
    training_seeds: tuple[int, ...] = (20261101, 20261102, 20261103)
    evaluation_seeds: tuple[int, ...] = tuple(range(20261121, 20261129))
    training_cycles: int = 4096
    evaluation_cycles: int = 12_000
    # run=1 is the separately reported memoryless comparator, not a member of
    # the non-degenerate run-length family.
    e_enter_grid: tuple[int, ...] = (2, 3, 4)
    leakage_enter_grid: tuple[int, ...] = (1, 2)
    leakage_clear_grid: tuple[int, ...] = (1, 2)
    fallback_clear_grid: tuple[int, ...] = (1, 2)
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        training = tuple(self.training_seeds)
        evaluation = tuple(self.evaluation_seeds)
        if len(training) < 3 or len(set(training)) != len(training):
            raise ValueError("training_seeds must contain at least three unique values")
        if len(evaluation) < 6 or len(set(evaluation)) != len(evaluation):
            raise ValueError("evaluation_seeds must contain at least six unique values")
        if set(training) & set(evaluation):
            raise ValueError("training and evaluation seeds must be disjoint")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or int(seed) < 0
            or int(seed) >= 2**64 - 1_000_000
            for seed in training + evaluation
        ):
            raise ValueError("seeds must be nonnegative uint64-safe integers")
        object.__setattr__(self, "training_seeds", tuple(int(seed) for seed in training))
        object.__setattr__(self, "evaluation_seeds", tuple(int(seed) for seed in evaluation))
        object.__setattr__(self, "training_cycles", _integer(self.training_cycles, "training_cycles", 256))
        object.__setattr__(self, "evaluation_cycles", _integer(self.evaluation_cycles, "evaluation_cycles", 512))
        for name in (
            "e_enter_grid",
            "leakage_enter_grid",
            "leakage_clear_grid",
            "fallback_clear_grid",
        ):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            parsed = tuple(_integer(value, f"{name} value", 1) for value in values)
            if any(value > 7 for value in parsed):
                raise ValueError(f"{name} values must fit the 3-bit counter")
            if name == "e_enter_grid" and any(value < 2 for value in parsed):
                raise ValueError("e_enter_grid excludes run=1, which is the memoryless comparator")
            object.__setattr__(self, name, parsed)
        confidence = float(self.confidence_level)
        if not isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie in (0,1)")
        object.__setattr__(self, "confidence_level", confidence)
        workload = len(event_scenarios()) * (
            len(training) * self.training_cycles + len(evaluation) * self.evaluation_cycles
        )
        if workload > 1_000_000:
            raise ValueError("base trace workload must not exceed 1,000,000 cycles")


@dataclass(frozen=True)
class _Trace:
    scenario_id: str
    base_seed: int
    effective_seed: int
    stream: SyndromeStream
    health: tuple[tuple[bool, bool, bool, bool], ...]
    truth_modes: tuple[str, ...]
    trace_sha256: str


def _states(scenario: EventScenario, cycles: int, seed: int) -> tuple[DriftState, ...]:
    phase = seed % max(1, scenario.burst_period)
    records = []
    for step in range(cycles):
        burst = bool(
            scenario.burst_period
            and (step + phase) % scenario.burst_period < scenario.burst_width
        )
        records.append(
            DriftState(
                step=step,
                time=float(step),
                mu_q=0.035 * np.sin(2.0 * np.pi * step / 701.0),
                mu_p=-0.030 * np.cos(2.0 * np.pi * step / 887.0),
                sigma_q=scenario.sigma_q * (1.30 if burst else 1.0),
                sigma_p=scenario.sigma_p * (1.25 if burst else 1.0),
                rho=scenario.rho,
                loss_gamma=scenario.loss_gamma,
                p_outlier=min(1.0, scenario.p_outlier + (0.08 if burst else 0.0)),
                outlier_scale=scenario.outlier_scale,
                burst_active=burst,
                source=f"t3.2.5:{scenario.scenario_id}",
                regime="burst" if burst else "base",
                seed=seed,
                event_id=step // max(1, scenario.burst_period) if burst else 0,
            )
        )
    return tuple(records)


def _health_trace(
    scenario: EventScenario, cycles: int, seed: int
) -> tuple[tuple[bool, bool, bool, bool], ...]:
    health = []
    offset = seed % max(1, scenario.health_fault_period)
    for cycle in range(cycles):
        flags = [True, True, True, True]
        if scenario.health_fault_period:
            position = (cycle + offset) % scenario.health_fault_period
            if position in (101, 102):
                flags[0] = False
            elif position == 211:
                flags[1] = False
            elif position in (317, 318):
                flags[2] = False
            elif position == 421:
                flags[3] = False
        health.append(tuple(flags))
    return tuple(health)  # type: ignore[return-value]


def _truth_mode(step: object, health: tuple[bool, bool, bool, bool]) -> str:
    if not all(health):
        return FALLBACK
    truth = step.truth  # type: ignore[attr-defined]
    if truth.leakage_kind != "none":
        return LEAKAGE_HOLD
    if truth.recovery_depth_before_action > 0:
        if truth.recovery_quadrature == "X":
            return X_RECOVERY
        if truth.recovery_quadrature == "Z":
            return Z_RECOVERY
    return NORMAL


def _make_trace(
    scenario: EventScenario, base_seed: int, cycles: int, scenario_index: int
) -> _Trace:
    effective_seed = int(base_seed + 100_000 * scenario_index)
    config = SyndromeStreamConfig(
        measurement_sigma=(0.025, 0.025),
        max_recovery_depth=6,
        depth_probability_scale=scenario.depth_probability_scale,
        recovery_probability=scenario.recovery_probability,
        recovery_gain=0.5,
        base_leakage_probability=scenario.base_leakage_probability,
        loss_leakage_scale=0.015,
        burst_leakage_bonus=0.025,
        higher_leakage_fraction=scenario.higher_leakage_fraction,
        higher_leakage_mean_duration=scenario.higher_leakage_mean_duration,
        readout_fidelity_g=scenario.readout_fidelity_g,
        readout_fidelity_e=scenario.readout_fidelity_e,
        seed=effective_seed,
    )
    stream = generate_syndrome_stream(
        _states(scenario, cycles, effective_seed), config=config
    )
    health = _health_trace(scenario, cycles, effective_seed)
    truth_modes = tuple(
        _truth_mode(step, flags) for step, flags in zip(stream.steps, health, strict=True)
    )
    digest = hashlib.sha256()
    digest.update(scenario.scenario_id.encode("utf-8"))
    digest.update(effective_seed.to_bytes(8, "little", signed=False))
    for step, flags in zip(stream.steps, health, strict=True):
        observed = step.observed
        digest.update(
            np.asarray(
                (*observed.residual_syndrome, observed.x_e_run, observed.z_e_run, observed.leakage_run),
                dtype="<f8",
            ).tobytes()
        )
        digest.update((observed.syndrome.x + observed.syndrome.z).encode("ascii"))
        digest.update(bytes(int(value) for value in flags))
    return _Trace(
        scenario_id=scenario.scenario_id,
        base_seed=base_seed,
        effective_seed=effective_seed,
        stream=stream,
        health=health,
        truth_modes=truth_modes,
        trace_sha256=digest.hexdigest(),
    )


def _fsm_input(trace: _Trace, cycle: int) -> RunLengthFSMInput:
    observed = trace.stream.steps[cycle].observed
    valid, crc_ok, parameter_fresh, deadline_ok = trace.health[cycle]
    return RunLengthFSMInput(
        cycle_index=cycle,
        residual=observed.residual_syndrome,
        syndrome_x=observed.syndrome.x,
        syndrome_z=observed.syndrome.z,
        quadrature_phase_bit=cycle & 1,
        valid=valid,
        crc_ok=crc_ok,
        parameter_fresh=parameter_fresh,
        deadline_ok=deadline_ok,
    )


def _static_mode(event: RunLengthFSMInput) -> str:
    return NORMAL if event.health_ok else FALLBACK


def _memoryless_mode(event: RunLengthFSMInput) -> str:
    if not event.health_ok:
        return FALLBACK
    if "leakage" in (event.syndrome_x, event.syndrome_z):
        return LEAKAGE_HOLD
    x_event = event.syndrome_x == "e"
    z_event = event.syndrome_z == "e"
    if x_event and z_event:
        return X_RECOVERY if event.quadrature_phase_bit == 0 else Z_RECOVERY
    if x_event:
        return X_RECOVERY
    if z_event:
        return Z_RECOVERY
    return NORMAL


_EVENT_COST = {
    NORMAL: {NORMAL: 0.0, X_RECOVERY: 0.18, Z_RECOVERY: 0.18, LEAKAGE_HOLD: 0.30, FALLBACK: 0.12},
    X_RECOVERY: {NORMAL: 1.0, X_RECOVERY: 0.0, Z_RECOVERY: 1.25, LEAKAGE_HOLD: 0.90, FALLBACK: 0.55},
    Z_RECOVERY: {NORMAL: 1.0, X_RECOVERY: 1.25, Z_RECOVERY: 0.0, LEAKAGE_HOLD: 0.90, FALLBACK: 0.55},
    LEAKAGE_HOLD: {NORMAL: 1.2, X_RECOVERY: 1.1, Z_RECOVERY: 1.1, LEAKAGE_HOLD: 0.0, FALLBACK: 0.25},
    FALLBACK: {NORMAL: 1.2, X_RECOVERY: 1.2, Z_RECOVERY: 1.2, LEAKAGE_HOLD: 0.25, FALLBACK: 0.0},
}


def event_control_cost(target: str, action: str) -> float:
    try:
        return _EVENT_COST[target][action]
    except KeyError as exc:
        raise ValueError(f"unknown target/action mode pair {target!r}/{action!r}") from exc


def _transition_count(modes: Sequence[str]) -> int:
    return sum(modes[index] != modes[index - 1] for index in range(1, len(modes)))


def _mean_detection_delay(truth: Sequence[str], action: Sequence[str]) -> float:
    delays: list[int] = []
    index = 0
    while index < len(truth):
        target = truth[index]
        end = index + 1
        while end < len(truth) and truth[end] == target:
            end += 1
        if target in TARGET_EVENT_MODES:
            matching = next(
                (cycle for cycle in range(index, end) if action[cycle] == target),
                end,
            )
            delays.append(matching - index)
        index = end
    return float(np.mean(delays)) if delays else 0.0


def _controller_metrics(
    truth: Sequence[str], modes: Sequence[str], *, bank_writes: int
) -> dict[str, float | int]:
    costs = np.asarray(
        [event_control_cost(target, action) for target, action in zip(truth, modes, strict=True)],
        dtype=np.float64,
    )
    cycles = len(truth)
    unsafe_targets = np.asarray(
        [target in (LEAKAGE_HOLD, FALLBACK) for target in truth], dtype=bool
    )
    unsafe_misses = np.asarray(
        [
            target in (LEAKAGE_HOLD, FALLBACK)
            and action not in (LEAKAGE_HOLD, FALLBACK)
            for target, action in zip(truth, modes, strict=True)
        ],
        dtype=bool,
    )
    normal_targets = np.asarray([target == NORMAL for target in truth], dtype=bool)
    false_interventions = np.asarray(
        [target == NORMAL and action != NORMAL for target, action in zip(truth, modes, strict=True)],
        dtype=bool,
    )
    return {
        "event_cost": float(np.mean(costs)),
        "event_plus_write_cost": float(np.mean(costs) + WRITE_COST * bank_writes / cycles),
        "action_accuracy": float(np.mean(np.asarray(truth) == np.asarray(modes))),
        "unsafe_miss_rate": float(np.sum(unsafe_misses) / max(1, np.sum(unsafe_targets))),
        "false_intervention_rate": float(np.sum(false_interventions) / max(1, np.sum(normal_targets))),
        "mean_event_detection_delay_cycles": _mean_detection_delay(truth, modes),
        "transitions": _transition_count(modes),
        "bank_writes": int(bank_writes),
        "fallback_cycles": int(sum(mode == FALLBACK for mode in modes)),
    }


def _evaluate_trace(
    trace: _Trace, config: RunLengthFSMConfig
) -> tuple[dict[str, object], tuple[str, ...]]:
    fsm = RunLengthParameterBankFSM(config)
    static_modes: list[str] = []
    memoryless_modes: list[str] = []
    fsm_modes: list[str] = []
    decisions = []
    corrections_finite = True
    for cycle in range(len(trace.stream.steps)):
        event = _fsm_input(trace, cycle)
        static_modes.append(_static_mode(event))
        memoryless_modes.append(_memoryless_mode(event))
        decision = fsm.step(event)
        decisions.append(decision)
        fsm_modes.append(decision.mode)
        corrections_finite &= all(isfinite(value) for value in decision.correction)
    truth_modes = trace.truth_modes
    mode_sets = {
        "static_safe_normal": tuple(static_modes),
        "memoryless_event": tuple(memoryless_modes),
        "run_length_fsm": tuple(fsm_modes),
        "truth_oracle": truth_modes,
    }
    metrics = {
        name: _controller_metrics(
            truth_modes,
            modes,
            bank_writes=(sum(decision.bank_switched for decision in decisions) if name == "run_length_fsm" else _transition_count(modes)),
        )
        for name, modes in mode_sets.items()
    }
    bank_writes = int(sum(decision.bank_switched for decision in decisions))
    row: dict[str, object] = {
        "scenario_id": trace.scenario_id,
        "base_evaluation_seed": trace.base_seed,
        "effective_seed": trace.effective_seed,
        "cycles": len(trace.stream.steps),
        "trace_sha256": trace.trace_sha256,
        "truth_normal_cycles": sum(mode == NORMAL for mode in truth_modes),
        "truth_x_recovery_cycles": sum(mode == X_RECOVERY for mode in truth_modes),
        "truth_z_recovery_cycles": sum(mode == Z_RECOVERY for mode in truth_modes),
        "truth_leakage_cycles": sum(mode == LEAKAGE_HOLD for mode in truth_modes),
        "truth_health_fallback_cycles": sum(mode == FALLBACK for mode in truth_modes),
        "fsm_final_bank_version": fsm.param_bank.active_version,
        "fsm_bank_conflicts": sum(decision.bank_conflict for decision in decisions),
        "fsm_local_safe_cycles": sum(decision.local_safe_rom_used for decision in decisions),
        "fsm_corrections_finite": corrections_finite,
    }
    for controller, controller_metrics in metrics.items():
        for metric, value in controller_metrics.items():
            row[f"{controller}_{metric}"] = value
    if fsm.param_bank.active_version != bank_writes:
        raise RuntimeError("parameter-bank version does not equal successful FSM writes")
    return row, tuple(fsm_modes)


def _threshold_candidates(settings: RunLengthValidationConfig) -> tuple[RunLengthFSMConfig, ...]:
    return tuple(
        RunLengthFSMConfig(
            counter_bits=3,
            e_enter_run=e_enter,
            leakage_enter_run=leakage_enter,
            leakage_clear_run=leakage_clear,
            fallback_clear_run=fallback_clear,
        )
        for e_enter, leakage_enter, leakage_clear, fallback_clear in itertools.product(
            settings.e_enter_grid,
            settings.leakage_enter_grid,
            settings.leakage_clear_grid,
            settings.fallback_clear_grid,
        )
    )


def _fit_thresholds(
    traces: Sequence[_Trace], settings: RunLengthValidationConfig
) -> tuple[RunLengthFSMConfig, list[dict[str, object]]]:
    grid_rows: list[dict[str, object]] = []
    for candidate in _threshold_candidates(settings):
        trace_scores = []
        trace_writes = []
        for trace in traces:
            row, _ = _evaluate_trace(trace, candidate)
            trace_scores.append(float(row["run_length_fsm_event_plus_write_cost"]))
            trace_writes.append(int(row["run_length_fsm_bank_writes"]))
        grid_rows.append(
            {
                "e_enter_run": candidate.e_enter_run,
                "leakage_enter_run": candidate.leakage_enter_run,
                "leakage_clear_run": candidate.leakage_clear_run,
                "fallback_clear_run": candidate.fallback_clear_run,
                "training_mean_event_plus_write_cost": float(np.mean(trace_scores)),
                "training_mean_bank_writes": float(np.mean(trace_writes)),
                "training_traces": len(trace_scores),
            }
        )
    selected = min(
        grid_rows,
        key=lambda row: (
            row["training_mean_event_plus_write_cost"],
            row["training_mean_bank_writes"],
            row["e_enter_run"],
            row["leakage_enter_run"],
            row["leakage_clear_run"],
            row["fallback_clear_run"],
        ),
    )
    return (
        RunLengthFSMConfig(
            counter_bits=3,
            e_enter_run=int(selected["e_enter_run"]),
            leakage_enter_run=int(selected["leakage_enter_run"]),
            leakage_clear_run=int(selected["leakage_clear_run"]),
            fallback_clear_run=int(selected["fallback_clear_run"]),
        ),
        grid_rows,
    )


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "cnn_fpga/runtime/run_length_fsm.py",
        "cnn_fpga/runtime/param_bank.py",
        "cnn_fpga/benchmark/run_length_fsm_baseline.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _conflict_probe() -> dict[str, object]:
    table = RunLengthParameterTable()
    bank = ParamBank(table.params(NORMAL))
    pending = bank.stage_update(
        DecoderRuntimeParams(
            K=np.diag([9.0, 9.0]),
            b=np.zeros(2),
            metadata={"mode": "external_probe"},
        ),
        commit_epoch=3,
        metadata={"writer": "slow_probe"},
    )
    fsm = RunLengthParameterBankFSM(
        RunLengthFSMConfig(e_enter_run=1, fallback_clear_run=1),
        parameter_table=table,
        param_bank=bank,
    )
    decisions = [
        fsm.step(
            RunLengthFSMInput(
                cycle_index=cycle,
                residual=(0.4, -0.2),
                syndrome_x="e",
                syndrome_z="g",
                quadrature_phase_bit=cycle & 1,
            )
        )
        for cycle in range(4)
    ]
    return {
        "pending_version": pending.version,
        "modes": [decision.mode for decision in decisions],
        "conflicts": [decision.bank_conflict for decision in decisions],
        "local_safe_rom": [decision.local_safe_rom_used for decision in decisions],
        "corrections": [list(decision.correction) for decision in decisions],
        "final_version": fsm.param_bank.active_version,
        "final_active_mode": fsm.param_bank.read_active().metadata.get("mode"),
    }


def _seed_cluster_difference(
    rows: Sequence[dict[str, object]], left: str, right: str
) -> list[float]:
    values = []
    for seed in sorted({int(row["base_evaluation_seed"]) for row in rows}):
        selected = [row for row in rows if int(row["base_evaluation_seed"]) == seed]
        values.append(
            float(
                np.mean(
                    [
                        float(row[f"{left}_event_plus_write_cost"])
                        - float(row[f"{right}_event_plus_write_cost"])
                        for row in selected
                    ]
                )
            )
        )
    return values


def build_run_length_validation(
    config: RunLengthValidationConfig | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings = RunLengthValidationConfig() if config is None else config
    if not isinstance(settings, RunLengthValidationConfig):
        raise TypeError("config must be RunLengthValidationConfig")
    scenarios = event_scenarios()
    training_traces = [
        _make_trace(scenario, seed, settings.training_cycles, scenario_index)
        for scenario_index, scenario in enumerate(scenarios)
        for seed in settings.training_seeds
    ]
    selected_config, training_grid = _fit_thresholds(training_traces, settings)
    evaluation_traces = [
        _make_trace(scenario, seed, settings.evaluation_cycles, scenario_index)
        for scenario_index, scenario in enumerate(scenarios)
        for seed in settings.evaluation_seeds
    ]
    rows = [_evaluate_trace(trace, selected_config)[0] for trace in evaluation_traces]

    scenario_summaries = []
    for scenario in scenarios:
        selected = [row for row in rows if row["scenario_id"] == scenario.scenario_id]
        summary: dict[str, object] = {
            "scenario_id": scenario.scenario_id,
            "seeds": len(selected),
            "cycles": sum(int(row["cycles"]) for row in selected),
        }
        for controller in CONTROLLERS:
            summary[controller] = {
                metric: float(np.mean([float(row[f"{controller}_{metric}"]) for row in selected]))
                for metric in (
                    "event_cost",
                    "event_plus_write_cost",
                    "action_accuracy",
                    "unsafe_miss_rate",
                    "false_intervention_rate",
                    "mean_event_detection_delay_cycles",
                    "bank_writes",
                )
            }
        scenario_summaries.append(summary)

    comparisons = {
        "static_minus_run_length": _mean_interval(
            _seed_cluster_difference(rows, "static_safe_normal", "run_length_fsm"),
            settings.confidence_level,
        ),
        "memoryless_minus_run_length": _mean_interval(
            _seed_cluster_difference(rows, "memoryless_event", "run_length_fsm"),
            settings.confidence_level,
        ),
        "run_length_minus_truth_oracle": _mean_interval(
            _seed_cluster_difference(rows, "run_length_fsm", "truth_oracle"),
            settings.confidence_level,
        ),
    }
    aggregate = {
        controller: {
            metric: float(np.mean([float(row[f"{controller}_{metric}"]) for row in rows]))
            for metric in (
                "event_cost",
                "event_plus_write_cost",
                "action_accuracy",
                "unsafe_miss_rate",
                "false_intervention_rate",
                "mean_event_detection_delay_cycles",
                "bank_writes",
            )
        }
        for controller in CONTROLLERS
    }
    probe = _conflict_probe()
    input_names = {field.name for field in fields(RunLengthFSMInput)}
    expected_rows = len(scenarios) * len(settings.evaluation_seeds)
    expected_grid = (
        len(settings.e_enter_grid)
        * len(settings.leakage_enter_grid)
        * len(settings.leakage_clear_grid)
        * len(settings.fallback_clear_grid)
    )
    gates = {
        "training_and_evaluation_seeds_are_disjoint": not bool(
            set(settings.training_seeds) & set(settings.evaluation_seeds)
        ),
        "threshold_grid_is_complete_and_training_only": (
            len(training_grid) == expected_grid
            and all(row["training_traces"] == len(training_traces) for row in training_grid)
        ),
        "online_fsm_schema_has_no_truth_or_regime": (
            RUN_LENGTH_DESCRIPTOR.hidden_truth_inputs == ()
            and not any("truth" in name or "hidden" in name or "regime" in name for name in input_names)
        ),
        "evaluation_source_grid_is_complete": len(rows) == expected_rows,
        "evaluation_traces_are_unique": len({row["trace_sha256"] for row in rows}) == expected_rows,
        "same_trace_contains_all_four_controllers": all(
            all(f"{controller}_event_plus_write_cost" in row for controller in CONTROLLERS)
            for row in rows
        ),
        "all_scenarios_exercise_recovery_and_leakage": all(
            sum(int(row["truth_x_recovery_cycles"]) + int(row["truth_z_recovery_cycles"]) for row in rows if row["scenario_id"] == scenario.scenario_id) > 0
            and sum(int(row["truth_leakage_cycles"]) for row in rows if row["scenario_id"] == scenario.scenario_id) > 0
            for scenario in scenarios
        ),
        "health_scenario_exercises_fallback": any(int(row["truth_health_fallback_cycles"]) > 0 for row in rows),
        "real_parameter_bank_versions_equal_successful_writes": all(
            int(row["fsm_final_bank_version"]) == int(row["run_length_fsm_bank_writes"])
            for row in rows
        ),
        "normal_evaluation_has_no_bank_conflict_or_nonfinite_action": all(
            int(row["fsm_bank_conflicts"]) == 0
            and int(row["fsm_local_safe_cycles"]) == 0
            and bool(row["fsm_corrections_finite"])
            for row in rows
        ),
        "parameter_writes_are_event_driven_not_per_cycle": all(
            int(row["run_length_fsm_bank_writes"]) < 0.50 * int(row["cycles"])
            for row in rows
        ),
        "truth_oracle_is_cost_lower_bound_on_every_trace": all(
            float(row["truth_oracle_event_plus_write_cost"])
            <= min(
                float(row["static_safe_normal_event_plus_write_cost"]),
                float(row["memoryless_event_event_plus_write_cost"]),
                float(row["run_length_fsm_event_plus_write_cost"]),
            )
            + 1.0e-15
            for row in rows
        ),
        "run_length_improves_static_aggregate": comparisons["static_minus_run_length"]["ci_low"] > 0.0,
        "conflict_probe_stays_local_safe_until_atomic_resync": (
            probe["modes"][:3] == [FALLBACK, FALLBACK, FALLBACK]
            and probe["conflicts"][:3] == [True, True, True]
            and probe["local_safe_rom"][:3] == [True, True, True]
            and probe["final_active_mode"] == X_RECOVERY
            and probe["final_version"] == 2
        ),
        "scope_is_software_event_cost_not_ler_or_hardware_measurement": (
            not RUN_LENGTH_DESCRIPTOR.logical_error_metric
            and not RUN_LENGTH_DESCRIPTOR.fixed_point_or_rtl
            and not RUN_LENGTH_DESCRIPTOR.target_hardware_measured
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    payload: dict[str, object] = {
        "schema_version": "t3.2.5-run-length-fsm-baseline-v1",
        "task_id": "T3.2.5",
        "status": "PASS" if not failed else "FAIL",
        "implementation_sha256": _implementation_sha256(),
        "descriptor": asdict(RUN_LENGTH_DESCRIPTOR),
        "validation_config": asdict(settings),
        "scenario_contracts": [asdict(scenario) for scenario in scenarios],
        "training_selection": {
            "evaluation_truth_used": False,
            "training_traces": len(training_traces),
            "training_cycles": len(training_traces) * settings.training_cycles,
            "training_fsm_replay_cycles": (
                len(training_traces) * settings.training_cycles * len(training_grid)
            ),
            "grid": training_grid,
            "selected_config": asdict(selected_config),
        },
        "scenario_summaries": scenario_summaries,
        "aggregate": {
            "evaluation_traces": len(rows),
            "evaluation_cycles": sum(int(row["cycles"]) for row in rows),
            "source_data_rows": len(rows),
            "controllers": aggregate,
            "paired_seed_cluster_comparisons": comparisons,
        },
        "parameter_bank_conflict_probe": probe,
        "gate_summary": {
            "passed": sum(bool(value) for value in gates.values()),
            "failed": len(failed),
            "gates": gates,
        },
        "claim_boundary": {
            "allowed": (
                "observed-only deterministic run-length event controller with saturating counters, "
                "atomic parameter-bank switching, explicit safety fallback, and paired software event-cost evidence"
            ),
            "forbidden": (
                "logical-error-rate gain, optimal physical recovery, bit-accurate RTL, synthesis/resource/Fmax, "
                "device-calibrated event cost, or target-board measurement"
            ),
        },
    }
    return payload, rows


def write_run_length_validation(
    json_path: str | Path = "docs/t3_2_5_run_length_fsm_validation.json",
    csv_path: str | Path = "docs/t3_2_5_run_length_fsm_source_data.csv",
    config: RunLengthValidationConfig | None = None,
) -> dict[str, object]:
    payload, rows = build_run_length_validation(config)
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not rows:
        raise RuntimeError("run-length validation produced no source rows")
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a reduced validation")
    parser.add_argument("--json", default="docs/t3_2_5_run_length_fsm_validation.json")
    parser.add_argument("--csv", default="docs/t3_2_5_run_length_fsm_source_data.csv")
    args = parser.parse_args(argv)
    config = (
        RunLengthValidationConfig(training_cycles=512, evaluation_cycles=1024)
        if args.smoke
        else None
    )
    payload = write_run_length_validation(args.json, args.csv, config)
    print(json.dumps(payload["gate_summary"], ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUN_LENGTH_DESCRIPTOR",
    "RunLengthBaselineDescriptor",
    "RunLengthValidationConfig",
    "build_run_length_validation",
    "event_control_cost",
    "event_scenarios",
    "write_run_length_validation",
]
