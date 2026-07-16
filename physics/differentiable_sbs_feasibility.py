"""T2.3.6 resource and numerical-feasibility scan for differentiable SBS training.

Every timed point executes a causal 15-output recurrent policy, a sampled joint
cavity--ancilla trajectory, both terms of the Feedback-GRAPE estimator, and an
Adam parameter update.  A point therefore measures a real optimization step rather than
the forward-only resource counters exposed by T2.3.4.  Risky points are run in
fresh subprocesses so an OOM cannot poison the remainder of the scan.

The result is a host-specific, finite-cutoff feasibility envelope.  It is not
evidence of optimizer convergence, lifetime gain, device calibration, pulse
Hamiltonian fidelity, or target-board latency.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np

from .differentiable_sbs_trajectory import (
    PARAMETER_NAMES,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
)
from .sbs_error_space import SBS_PROTOCOL_ID

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - only the minimal recovery env.
    psutil = None  # type: ignore[assignment]

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional in the recovery env.
    torch = None  # type: ignore[assignment]


DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE = (
    "host-specific cutoff/batch/horizon forward-backward-Adam-step resource and numerical "
    "feasibility of the finite-cutoff two-level differentiable SBS model with a "
    "causal 15-output recurrent policy; not optimizer convergence, NMF lifetime "
    "gain, pulse/device calibration, deployment latency, or hardware evidence"
)
POLICY_ARCHITECTURE_ID = "PUVIANI_SCALE_GRU10_MLP256_256_OUT15_AUDIT"
RESULT_MARKER = "T236_RESULT_JSON="


def _require_runtime() -> tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("T2.3.6 requires PyTorch; use the local DLEnv/QuantumEnv")
    if psutil is None:
        raise RuntimeError("T2.3.6 requires psutil for process RSS measurement")
    return torch, psutil


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class RecurrentPolicySpec:
    input_features: int = 3
    hidden_size: int = 10
    dense_widths: tuple[int, int] = (256, 256)
    output_controls: int = 15
    output_scale: float = 0.10
    architecture_id: str = POLICY_ARCHITECTURE_ID

    def __post_init__(self) -> None:
        for name in ("input_features", "hidden_size", "output_controls"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if len(self.dense_widths) != 2:
            raise ValueError("dense_widths must contain exactly two layers")
        widths = tuple(_positive_int(value, "dense_width") for value in self.dense_widths)
        object.__setattr__(self, "dense_widths", widths)
        object.__setattr__(self, "output_scale", _positive_float(self.output_scale, "output_scale"))
        if self.input_features != 3 or self.output_controls != len(PARAMETER_NAMES):
            raise ValueError("T2.3.6 freezes three causal features and fifteen outputs")
        if self.architecture_id != POLICY_ARCHITECTURE_ID:
            raise ValueError("architecture_id must preserve the registered audit architecture")

    @property
    def analytic_parameter_count(self) -> int:
        h = self.hidden_size
        i = self.input_features
        first, second = self.dense_widths
        gru = 3 * h * i + 3 * h * h + 6 * h
        return (
            gru
            + (h + 1) * first
            + (first + 1) * second
            + (second + 1) * self.output_controls
        )


@dataclass(frozen=True)
class TrainingPointConfig:
    cutoff: int = 8
    batch_size: int = 4
    full_cycles: int = 2
    device: str = "cpu"
    real_dtype: str = "float64"
    grid_points: int = 2049
    warmup_steps: int = 1
    repeats: int = 3
    score_baseline: float = 0.35
    learning_rate: float = 1.0e-4
    runtime_budget_seconds: float = 10.0
    preferred_runtime_seconds: float = 2.0
    maximum_memory_fraction: float = 0.75
    trace_tolerance: float = 2.0e-9
    hermiticity_tolerance: float = 2.0e-9
    positivity_tolerance: float = 2.0e-8
    minimum_gradient_norm: float = 1.0e-10
    seed: int = 314159
    policy: RecurrentPolicySpec = RecurrentPolicySpec()
    protocol_id: str = SBS_PROTOCOL_ID
    scope: str = DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE

    def __post_init__(self) -> None:
        cutoff = _positive_int(self.cutoff, "cutoff")
        if not 4 <= cutoff <= 48:
            raise ValueError("cutoff must lie in [4, 48]")
        object.__setattr__(self, "cutoff", cutoff)
        batch = _positive_int(self.batch_size, "batch_size")
        if batch > 4096:
            raise ValueError("batch_size exceeds the simulator feasibility guard")
        object.__setattr__(self, "batch_size", batch)
        cycles = _positive_int(self.full_cycles, "full_cycles")
        if cycles > 10:
            raise ValueError("full_cycles must lie in [1, 10]")
        object.__setattr__(self, "full_cycles", cycles)
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        grid = _positive_int(self.grid_points, "grid_points")
        if grid < 1025 or grid % 2 == 0:
            raise ValueError("grid_points must be odd and at least 1025")
        object.__setattr__(self, "grid_points", grid)
        object.__setattr__(self, "warmup_steps", _nonnegative_int(self.warmup_steps, "warmup_steps"))
        repeats = _positive_int(self.repeats, "repeats")
        if repeats > 20:
            raise ValueError("repeats is capped at 20")
        object.__setattr__(self, "repeats", repeats)
        for name in (
            "runtime_budget_seconds",
            "preferred_runtime_seconds",
            "trace_tolerance",
            "hermiticity_tolerance",
            "positivity_tolerance",
            "minimum_gradient_norm",
            "learning_rate",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        baseline = float(self.score_baseline)
        if not isfinite(baseline):
            raise ValueError("score_baseline must be finite")
        object.__setattr__(self, "score_baseline", baseline)
        fraction = float(self.maximum_memory_fraction)
        if not isfinite(fraction) or not 0.05 <= fraction <= 0.95:
            raise ValueError("maximum_memory_fraction must lie in [0.05, 0.95]")
        object.__setattr__(self, "maximum_memory_fraction", fraction)
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))
        if not isinstance(self.policy, RecurrentPolicySpec):
            raise TypeError("policy must be a RecurrentPolicySpec")
        if self.protocol_id != SBS_PROTOCOL_ID:
            raise ValueError("T2.3.6 scans only the frozen SBS main protocol")
        if self.scope != DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE:
            raise ValueError("scope must preserve the fail-closed feasibility boundary")

    @property
    def half_cycles(self) -> int:
        return 2 * self.full_cycles

    @property
    def point_id(self) -> str:
        return (
            f"{self.device}-c{self.cutoff}-b{self.batch_size}-"
            f"h{self.full_cycles}-{self.real_dtype}"
        )

    @property
    def state_tensor_bytes(self) -> int:
        complex_bytes = 16 if self.real_dtype == "float64" else 8
        joint_dimension = 2 * self.cutoff
        return self.batch_size * joint_dimension * joint_dimension * complex_bytes

    @property
    def autograd_state_lower_bound_bytes(self) -> int:
        return self.state_tensor_bytes * (1 + 7 * self.half_cycles)


@dataclass(frozen=True)
class TrainingPointResult:
    point_id: str
    status: str
    cutoff: int
    batch_size: int
    full_cycles: int
    half_cycles: int
    device: str
    real_dtype: str
    policy_architecture_id: str
    policy_parameter_count: int
    warmup_steps: int
    repeats: int
    runtime_budget_seconds: float
    preferred_runtime_seconds: float
    maximum_memory_fraction: float
    grid_points: int = 2049
    score_baseline: float = 0.35
    learning_rate: float = 1.0e-4
    seed: int = 314159
    numerical_stable: bool = False
    within_runtime_budget: bool = False
    within_memory_budget: bool = False
    feasible: bool = False
    preferred: bool = False
    initialization_seconds: float | None = None
    runtime_seconds: tuple[float, ...] = ()
    runtime_median_seconds: float | None = None
    runtime_p90_seconds: float | None = None
    trajectory_cycle_throughput_per_second: float | None = None
    baseline_rss_bytes: int | None = None
    peak_rss_bytes: int | None = None
    rss_delta_bytes: int | None = None
    system_memory_total_bytes: int | None = None
    system_memory_available_before_bytes: int | None = None
    cuda_peak_allocated_bytes: int | None = None
    cuda_peak_reserved_bytes: int | None = None
    cuda_total_bytes: int | None = None
    cuda_free_before_bytes: int | None = None
    observed_memory_fraction: float | None = None
    state_tensor_bytes: int = 0
    autograd_state_lower_bound_bytes: int = 0
    memory_amplification_over_lower_bound: float | None = None
    mean_reward: float | None = None
    mean_ground_outcome_fraction: float | None = None
    minimum_trajectory_probability: float | None = None
    maximum_trajectory_probability: float | None = None
    minimum_gradient_norm: float | None = None
    maximum_gradient_norm: float | None = None
    minimum_parameter_update_norm: float | None = None
    maximum_parameter_update_norm: float | None = None
    maximum_trace_error: float | None = None
    maximum_hermiticity_error: float | None = None
    minimum_final_eigenvalue: float | None = None
    objective_finite: bool = False
    gradients_finite: bool = False
    failure_kind: str | None = None
    failure_message: str | None = None
    protocol_id: str = SBS_PROTOCOL_ID
    scope: str = DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE


if torch is not None:

    class FeasibilityRecurrentPolicy(torch.nn.Module):
        """Causal GRU10 + 256 + 256 policy with all fifteen physical outputs."""

        def __init__(
            self,
            spec: RecurrentPolicySpec = RecurrentPolicySpec(),
            *,
            device: str = "cpu",
            dtype: Any | None = None,
            seed: int = 0,
        ) -> None:
            super().__init__()
            th = torch
            if not isinstance(spec, RecurrentPolicySpec):
                raise TypeError("spec must be a RecurrentPolicySpec")
            actual_dtype = th.float64 if dtype is None else dtype
            devices = [th.device(device).index or 0] if device == "cuda" else []
            with th.random.fork_rng(devices=devices):
                th.manual_seed(int(seed))
                self.gru = th.nn.GRUCell(spec.input_features, spec.hidden_size)
                self.dense1 = th.nn.Linear(spec.hidden_size, spec.dense_widths[0])
                self.dense2 = th.nn.Linear(spec.dense_widths[0], spec.dense_widths[1])
                self.output = th.nn.Linear(spec.dense_widths[1], spec.output_controls)
            self.spec = spec
            self.to(device=device, dtype=actual_dtype)

        @property
        def parameter_count(self) -> int:
            return sum(parameter.numel() for parameter in self.parameters())

        def forward(self, history: Any, half_index: int) -> Any:
            th = torch
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] != half_index:
                raise ValueError("history width must equal the causal half-cycle index")
            if half_index < 0:
                raise ValueError("half_index must be nonnegative")
            reference = next(self.parameters())
            batch = history.shape[0]
            hidden = th.zeros(
                (batch, self.spec.hidden_size),
                device=reference.device,
                dtype=reference.dtype,
            )
            for index in range(half_index):
                signed = 2.0 * history[:, index].to(reference.dtype) - 1.0
                observed = th.ones_like(signed)
                position = th.full_like(signed, (index + 1.0) / max(half_index, 1))
                features = th.stack((signed, observed, position), dim=-1)
                hidden = self.gru(features, hidden)
            value = th.tanh(self.dense1(hidden))
            value = th.tanh(self.dense2(value))
            return self.spec.output_scale * self.output(value)

else:

    class FeasibilityRecurrentPolicy:  # pragma: no cover - recovery-env error path.
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_runtime()


class _RSSPeakMonitor:
    def __init__(self, process: Any, interval_seconds: float = 0.002) -> None:
        self.process = process
        self.interval_seconds = interval_seconds
        self.baseline = int(process.memory_info().rss)
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, int(self.process.memory_info().rss))
            except (OSError, RuntimeError):
                break
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "_RSSPeakMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.peak = max(self.peak, int(self.process.memory_info().rss))
        except (OSError, RuntimeError):
            pass


def _failure_result(
    config: TrainingPointConfig,
    *,
    kind: str,
    message: str,
    status: str = "exception",
) -> TrainingPointResult:
    return TrainingPointResult(
        point_id=config.point_id,
        status=status,
        cutoff=config.cutoff,
        batch_size=config.batch_size,
        full_cycles=config.full_cycles,
        half_cycles=config.half_cycles,
        device=config.device,
        real_dtype=config.real_dtype,
        policy_architecture_id=config.policy.architecture_id,
        policy_parameter_count=config.policy.analytic_parameter_count,
        warmup_steps=config.warmup_steps,
        repeats=config.repeats,
        runtime_budget_seconds=config.runtime_budget_seconds,
        preferred_runtime_seconds=config.preferred_runtime_seconds,
        maximum_memory_fraction=config.maximum_memory_fraction,
        grid_points=config.grid_points,
        score_baseline=config.score_baseline,
        learning_rate=config.learning_rate,
        seed=config.seed,
        state_tensor_bytes=config.state_tensor_bytes,
        autograd_state_lower_bound_bytes=config.autograd_state_lower_bound_bytes,
        failure_kind=kind,
        failure_message=message[-2000:],
    )


def benchmark_training_point(config: TrainingPointConfig) -> TrainingPointResult:
    """Execute repeated forward + reward/score backward steps in this process."""

    th, ps = _require_runtime()
    if not isinstance(config, TrainingPointConfig):
        raise TypeError("config must be a TrainingPointConfig")
    if config.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = th.device(config.device)
    dtype = th.float64 if config.real_dtype == "float64" else th.float32
    process = ps.Process(os.getpid())
    virtual = ps.virtual_memory()
    system_total = int(virtual.total)
    system_available = int(virtual.available)
    cuda_total = None
    cuda_free = None
    if config.device == "cuda":
        th.cuda.empty_cache()
        cuda_free, cuda_total = (int(value) for value in th.cuda.mem_get_info(device))
        memory_capacity = cuda_total
    else:
        memory_capacity = system_available
    if config.autograd_state_lower_bound_bytes > config.maximum_memory_fraction * memory_capacity:
        return _failure_result(
            config,
            kind="preflight_memory_lower_bound",
            message="analytic autograd state lower bound exceeds the configured memory budget",
            status="preflight_rejected",
        )

    runtime_samples: list[float] = []
    rewards: list[float] = []
    ground_fractions: list[float] = []
    minimum_probabilities: list[float] = []
    maximum_probabilities: list[float] = []
    gradient_norms: list[float] = []
    parameter_update_norms: list[float] = []
    trace_errors: list[float] = []
    hermiticity_errors: list[float] = []
    minimum_eigenvalues: list[float] = []
    objective_finite = True
    gradients_finite = True
    cuda_allocated: list[int] = []
    cuda_reserved: list[int] = []

    with _RSSPeakMonitor(process) as monitor:
        init_start = time.perf_counter()
        simulator = DifferentiableSBSTrajectorySimulator(
            DifferentiableSBSConfig(
                cutoff=config.cutoff,
                full_cycles=config.full_cycles,
                batch_size=config.batch_size,
                grid_points=config.grid_points,
                device=config.device,
                real_dtype=config.real_dtype,
            )
        )
        policy = FeasibilityRecurrentPolicy(
            config.policy,
            device=config.device,
            dtype=dtype,
            seed=config.seed,
        )
        optimizer = th.optim.Adam(policy.parameters(), lr=config.learning_rate)
        initialization_seconds = time.perf_counter() - init_start
        if policy.parameter_count != config.policy.analytic_parameter_count:
            raise RuntimeError("policy parameter count differs from the analytic contract")

        total_steps = config.warmup_steps + config.repeats
        for step_index in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            if config.device == "cuda":
                th.cuda.synchronize(device)
            start = time.perf_counter()
            result = simulator.run(
                control_policy=policy,
                seed=config.seed + 1000 + step_index,
            )
            score_term = th.mean(
                (result.reward.detach() - config.score_baseline)
                * result.log_probability
            )
            objective = result.reward.mean() + score_term
            objective.backward()
            if config.device == "cuda":
                th.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            gradients = [
                parameter.grad.reshape(-1)
                for parameter in policy.parameters()
                if parameter.grad is not None
            ]
            if not gradients:
                raise RuntimeError("recurrent policy received no gradient")
            flat_gradient = th.cat(gradients)
            current_objective_finite = bool(th.isfinite(objective.detach()).cpu())
            current_gradient_finite = bool(th.all(th.isfinite(flat_gradient)).cpu())
            objective_finite = objective_finite and current_objective_finite
            gradients_finite = gradients_finite and current_gradient_finite
            gradient_norm = float(th.linalg.vector_norm(flat_gradient.detach()).cpu())
            parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]
            optimizer.step()
            update_squared = th.zeros((), device=device, dtype=dtype)
            for parameter, before in zip(policy.parameters(), parameters_before):
                update_squared = update_squared + th.sum((parameter.detach() - before) ** 2)
            update_norm = float(th.sqrt(update_squared).cpu())

            if step_index >= config.warmup_steps:
                runtime_samples.append(elapsed)
                rewards.append(float(result.reward.detach().mean().cpu()))
                ground_fractions.append(
                    float((result.outcomes.detach() == 0).to(th.float64).mean().cpu())
                )
                probabilities = result.trajectory_probability.detach()
                minimum_probabilities.append(float(probabilities.min().cpu()))
                maximum_probabilities.append(float(probabilities.max().cpu()))
                gradient_norms.append(gradient_norm)
                parameter_update_norms.append(update_norm)
                trace_errors.append(result.maximum_trace_error)
                hermiticity_errors.append(result.maximum_hermiticity_error)
                minimum_eigenvalues.append(result.minimum_final_eigenvalue)
                if config.device == "cuda":
                    cuda_allocated.append(int(th.cuda.max_memory_allocated(device)))
                    cuda_reserved.append(int(th.cuda.max_memory_reserved(device)))
            del flat_gradient, gradients, parameters_before, update_squared, objective, score_term, result
        del optimizer, policy, simulator
        gc.collect()
        if config.device == "cuda":
            th.cuda.synchronize(device)

    runtime_tuple = tuple(float(value) for value in runtime_samples)
    runtime_median = float(median(runtime_tuple))
    runtime_p90 = float(np.quantile(np.asarray(runtime_tuple), 0.90))
    peak_rss = int(monitor.peak)
    rss_delta = max(0, peak_rss - monitor.baseline)
    cuda_peak_allocated = max(cuda_allocated) if cuda_allocated else None
    cuda_peak_reserved = max(cuda_reserved) if cuda_reserved else None
    observed_memory = cuda_peak_allocated if config.device == "cuda" else peak_rss
    observed_capacity = cuda_total if config.device == "cuda" else system_total
    observed_fraction = float(observed_memory / observed_capacity)
    memory_delta_for_amplification = (
        cuda_peak_allocated if config.device == "cuda" else max(rss_delta, 1)
    )
    amplification = float(
        memory_delta_for_amplification / max(config.autograd_state_lower_bound_bytes, 1)
    )
    max_trace = max(trace_errors)
    max_hermiticity = max(hermiticity_errors)
    min_eigenvalue = min(minimum_eigenvalues)
    min_gradient = min(gradient_norms)
    numerical_stable = (
        objective_finite
        and gradients_finite
        and min_gradient >= config.minimum_gradient_norm
        and min(parameter_update_norms) > 0.0
        and max_trace <= config.trace_tolerance
        and max_hermiticity <= config.hermiticity_tolerance
        and min_eigenvalue >= -config.positivity_tolerance
        and min(minimum_probabilities) > 0.0
        and max(maximum_probabilities) <= 1.0 + config.trace_tolerance
    )
    within_runtime = runtime_median <= config.runtime_budget_seconds
    within_memory = observed_fraction <= config.maximum_memory_fraction
    feasible = numerical_stable and within_runtime and within_memory
    preferred = feasible and runtime_median <= config.preferred_runtime_seconds
    if not numerical_stable:
        status = "numerical_failure"
    elif not within_memory:
        status = "memory_exceeded"
    elif not within_runtime:
        status = "runtime_exceeded"
    else:
        status = "pass"
    return TrainingPointResult(
        point_id=config.point_id,
        status=status,
        cutoff=config.cutoff,
        batch_size=config.batch_size,
        full_cycles=config.full_cycles,
        half_cycles=config.half_cycles,
        device=config.device,
        real_dtype=config.real_dtype,
        policy_architecture_id=config.policy.architecture_id,
        policy_parameter_count=config.policy.analytic_parameter_count,
        warmup_steps=config.warmup_steps,
        repeats=config.repeats,
        runtime_budget_seconds=config.runtime_budget_seconds,
        preferred_runtime_seconds=config.preferred_runtime_seconds,
        maximum_memory_fraction=config.maximum_memory_fraction,
        grid_points=config.grid_points,
        score_baseline=config.score_baseline,
        learning_rate=config.learning_rate,
        seed=config.seed,
        numerical_stable=numerical_stable,
        within_runtime_budget=within_runtime,
        within_memory_budget=within_memory,
        feasible=feasible,
        preferred=preferred,
        initialization_seconds=initialization_seconds,
        runtime_seconds=runtime_tuple,
        runtime_median_seconds=runtime_median,
        runtime_p90_seconds=runtime_p90,
        trajectory_cycle_throughput_per_second=(
            config.batch_size * config.full_cycles / runtime_median
        ),
        baseline_rss_bytes=monitor.baseline,
        peak_rss_bytes=peak_rss,
        rss_delta_bytes=rss_delta,
        system_memory_total_bytes=system_total,
        system_memory_available_before_bytes=system_available,
        cuda_peak_allocated_bytes=cuda_peak_allocated,
        cuda_peak_reserved_bytes=cuda_peak_reserved,
        cuda_total_bytes=cuda_total,
        cuda_free_before_bytes=cuda_free,
        observed_memory_fraction=observed_fraction,
        state_tensor_bytes=config.state_tensor_bytes,
        autograd_state_lower_bound_bytes=config.autograd_state_lower_bound_bytes,
        memory_amplification_over_lower_bound=amplification,
        mean_reward=float(np.mean(rewards)),
        mean_ground_outcome_fraction=float(np.mean(ground_fractions)),
        minimum_trajectory_probability=min(minimum_probabilities),
        maximum_trajectory_probability=max(maximum_probabilities),
        minimum_gradient_norm=min_gradient,
        maximum_gradient_norm=max(gradient_norms),
        minimum_parameter_update_norm=min(parameter_update_norms),
        maximum_parameter_update_norm=max(parameter_update_norms),
        maximum_trace_error=max_trace,
        maximum_hermiticity_error=max_hermiticity,
        minimum_final_eigenvalue=min_eigenvalue,
        objective_finite=objective_finite,
        gradients_finite=gradients_finite,
    )


def safe_benchmark_training_point(config: TrainingPointConfig) -> TrainingPointResult:
    try:
        return benchmark_training_point(config)
    except Exception as exc:  # isolated worker serializes OOM/driver failures.
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.lower()
        if isinstance(exc, MemoryError) or "out of memory" in lowered:
            status = "oom"
            kind = "out_of_memory"
        else:
            status = "exception"
            kind = type(exc).__name__
        if torch is not None and config.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
        return _failure_result(config, kind=kind, message=message, status=status)


def run_point_subprocess(
    config: TrainingPointConfig,
    *,
    timeout_seconds: float = 900.0,
) -> TrainingPointResult:
    """Run one point in a fresh interpreter and parse its structured result."""

    timeout = _positive_float(timeout_seconds, "timeout_seconds")
    command = [
        sys.executable,
        "-m",
        "physics.differentiable_sbs_feasibility",
        "--worker",
        "--cutoff",
        str(config.cutoff),
        "--batch-size",
        str(config.batch_size),
        "--full-cycles",
        str(config.full_cycles),
        "--device",
        config.device,
        "--real-dtype",
        config.real_dtype,
        "--grid-points",
        str(config.grid_points),
        "--warmup-steps",
        str(config.warmup_steps),
        "--repeats",
        str(config.repeats),
        "--score-baseline",
        str(config.score_baseline),
        "--learning-rate",
        str(config.learning_rate),
        "--runtime-budget-seconds",
        str(config.runtime_budget_seconds),
        "--preferred-runtime-seconds",
        str(config.preferred_runtime_seconds),
        "--maximum-memory-fraction",
        str(config.maximum_memory_fraction),
        "--seed",
        str(config.seed),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        return _failure_result(
            config,
            kind="timeout",
            message=f"worker exceeded {timeout:.3f} seconds",
            status="timeout",
        )
    marker_lines = [
        line[len(RESULT_MARKER) :]
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if not marker_lines:
        message = (completed.stderr or completed.stdout or "worker produced no result")[-2000:]
        return _failure_result(
            config,
            kind="worker_exit",
            message=f"exit={completed.returncode}; {message}",
        )
    try:
        payload = json.loads(marker_lines[-1])
        return TrainingPointResult(**payload)
    except (json.JSONDecodeError, TypeError) as exc:
        return _failure_result(
            config,
            kind="invalid_worker_payload",
            message=f"{type(exc).__name__}: {exc}",
        )


def default_scan_points(
    device: str,
    *,
    warmup_steps: int = 1,
    repeats: int = 3,
    runtime_budget_seconds: float = 10.0,
    preferred_runtime_seconds: float = 2.0,
) -> tuple[TrainingPointConfig, ...]:
    """Return the preregistered staged scan; it is not a tiny Cartesian demo."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    common = dict(
        device=device,
        warmup_steps=warmup_steps,
        repeats=repeats,
        runtime_budget_seconds=runtime_budget_seconds,
        preferred_runtime_seconds=preferred_runtime_seconds,
    )
    triples: set[tuple[int, int, int]] = set()
    if device == "cuda":
        # Cutoff and batch axes at a two-cycle anchor.
        triples.update((cutoff, 8, 2) for cutoff in (8, 12, 16, 18, 24, 32, 48))
        triples.update((16, batch, 2) for batch in (1, 4, 8, 16, 32, 64, 128, 256, 512))
        # Every 2--10 cycle horizon at two useful training batches.
        triples.update((16, batch, horizon) for batch in (8, 16) for horizon in range(2, 11))
        # Expansion points test whether even horizons can use a larger batch.
        triples.update((16, 32, horizon) for horizon in (2, 4, 6, 8, 10))
        # High-cutoff long-horizon anchors prevent a cutoff-8-only conclusion.
        triples.update((cutoff, 4, horizon) for cutoff in (18, 24) for horizon in (6, 10))
        # Isolated resource-frontier probes continue until memory/runtime rejection.
        triples.update((16, batch, 10) for batch in (64, 128, 256, 512))
        triples.add((16, 576, 10))
        triples.update((24, batch, 10) for batch in (8, 16, 32, 64))
        triples.update((32, batch, 10) for batch in (4, 8, 16, 32))
        triples.update((48, batch, 10) for batch in (2, 4, 8, 16))
    else:
        # CPU is a representative fallback/RSS lane, not a duplicate 37-point scan.
        triples.update((cutoff, 2, 2) for cutoff in (8, 12, 16))
        triples.update((8, batch, 2) for batch in (1, 4, 8))
        triples.update((8, 4, horizon) for horizon in (2, 6, 10))
        triples.add((16, 2, 10))
    return tuple(
        TrainingPointConfig(cutoff=cutoff, batch_size=batch, full_cycles=horizon, **common)
        for cutoff, batch, horizon in sorted(triples)
    )


def validate_production_design(points: Sequence[TrainingPointConfig], device: str) -> None:
    if not points:
        raise ValueError("scan design must not be empty")
    if any(point.device != device for point in points):
        raise ValueError("all scan points must use the requested device")
    triples = {(point.cutoff, point.batch_size, point.full_cycles) for point in points}
    cutoffs = {point.cutoff for point in points}
    batches = {point.batch_size for point in points}
    horizons = {point.full_cycles for point in points}
    if device == "cuda":
        if not {8, 12, 16, 18, 24, 32, 48}.issubset(cutoffs):
            raise ValueError("CUDA production scan lacks the cutoff axis")
        if not {1, 4, 8, 16, 32, 64, 128, 256, 512}.issubset(batches):
            raise ValueError("CUDA production scan lacks the batch axis")
        if not set(range(2, 11)).issubset(horizons):
            raise ValueError("CUDA production scan must cover every 2--10 cycle horizon")
        required = {
            (16, batch, horizon)
            for batch in (8, 16)
            for horizon in range(2, 11)
        }
        if not required.issubset(triples):
            raise ValueError("CUDA production scan lacks the 2--10 envelope matrix")
        if not {(18, 4, 10), (24, 4, 10)}.issubset(triples):
            raise ValueError("CUDA production scan lacks high-cutoff long-horizon anchors")
        frontier = {
            (16, 512, 10),
            (16, 576, 10),
            (24, 64, 10),
            (32, 32, 10),
            (48, 16, 10),
        }
        if not frontier.issubset(triples):
            raise ValueError("CUDA production scan lacks isolated resource-frontier probes")
    else:
        if len(points) < 8 or max(cutoffs) < 16 or max(batches) < 8 or 10 not in horizons:
            raise ValueError("CPU production lane is too small to report RSS/runtime scaling")


def summarize_scan(results: Sequence[TrainingPointResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("results must not be empty")
    devices = sorted({result.device for result in results})
    primary_device = "cuda" if "cuda" in devices else devices[0]
    primary = [result for result in results if result.device == primary_device]
    largest_batch_by_horizon: dict[str, int | None] = {}
    preferred_batch_by_horizon: dict[str, int | None] = {}
    for horizon in range(2, 11):
        candidates = [
            result.batch_size
            for result in primary
            if result.cutoff == 16
            and result.full_cycles == horizon
            and result.feasible
        ]
        preferred_candidates = [
            result.batch_size
            for result in primary
            if result.cutoff == 16
            and result.full_cycles == horizon
            and result.preferred
        ]
        largest_batch_by_horizon[str(horizon)] = max(candidates) if candidates else None
        preferred_batch_by_horizon[str(horizon)] = (
            max(preferred_candidates) if preferred_candidates else None
        )
    values = list(largest_batch_by_horizon.values())
    all_horizons_observed = all(value is not None for value in values)
    common_batch = min(value for value in values if value is not None) if all_horizons_observed else None
    high_cutoff_long_horizon = [
        result
        for result in primary
        if result.cutoff >= 18
        and result.full_cycles == 10
        and result.batch_size >= 4
        and result.feasible
    ]
    envelope_confirmed = bool(
        all_horizons_observed
        and common_batch is not None
        and common_batch >= 8
        and high_cutoff_long_horizon
    )
    fatal_statuses = {"exception", "timeout", "numerical_failure"}
    fatal = [result.point_id for result in results if result.status in fatal_statuses]
    resource_statuses = {"oom", "preflight_rejected", "memory_exceeded", "runtime_exceeded"}
    resource_boundary = [
        result.point_id for result in results if result.status in resource_statuses
    ]
    decision = (
        "FEASIBLE_2_TO_10_CYCLE_TEACHER_KERNEL"
        if envelope_confirmed
        else "FALSIFIED_AT_REGISTERED_RESOURCE_GATE"
    )
    return {
        "primary_device": primary_device,
        "devices": devices,
        "point_count": len(results),
        "status_counts": {
            status: sum(result.status == status for result in results)
            for status in sorted({result.status for result in results})
        },
        "largest_tested_feasible_batch_by_horizon_at_cutoff16": largest_batch_by_horizon,
        "largest_tested_preferred_batch_by_horizon_at_cutoff16": preferred_batch_by_horizon,
        "common_feasible_batch_for_cycles_2_to_10": common_batch,
        "high_cutoff_long_horizon_points": [result.point_id for result in high_cutoff_long_horizon],
        "two_to_ten_cycle_envelope_confirmed": envelope_confirmed,
        "fatal_point_ids": fatal,
        "resource_boundary_point_ids": resource_boundary,
        "resource_frontier_observed": bool(resource_boundary),
        "decision": decision,
        "claim_boundary": (
            "The decision covers one current-host forward/backward/Adam-update kernel. It does not "
            "prove optimization convergence, seed robustness, lifetime ranking, physical "
            "cutoff convergence, device calibration, or FPGA timing."
        ),
    }


def run_feasibility_scan(
    *,
    devices: Sequence[str] = ("cuda", "cpu"),
    output: str | Path | None = None,
    warmup_steps: int = 1,
    repeats: int = 3,
    runtime_budget_seconds: float = 10.0,
    preferred_runtime_seconds: float = 2.0,
    timeout_seconds: float = 900.0,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the preregistered subprocess-isolated production scan."""

    th, _ = _require_runtime()
    normalized_devices = tuple(dict.fromkeys(devices))
    if not normalized_devices or any(device not in {"cpu", "cuda"} for device in normalized_devices):
        raise ValueError("devices must contain cpu and/or cuda")
    if "cuda" in normalized_devices and not th.cuda.is_available():
        raise RuntimeError("CUDA production lane requested but unavailable")
    points: list[TrainingPointConfig] = []
    for device in normalized_devices:
        designed = default_scan_points(
            device,
            warmup_steps=warmup_steps,
            repeats=repeats,
            runtime_budget_seconds=runtime_budget_seconds,
            preferred_runtime_seconds=preferred_runtime_seconds,
        )
        validate_production_design(designed, device)
        points.extend(designed)
    reusable: dict[str, TrainingPointResult] = {}
    output_path = Path(output) if output is not None else None
    if resume and output_path is not None and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("task_id") != "T2.3.6" or previous.get("scope") != DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE:
            raise ValueError("resume artifact has the wrong task or scope")
        contract = previous.get("measurement_contract", {})
        expected_contract = {
            "warmup_steps": warmup_steps,
            "timed_repeats": repeats,
            "runtime_budget_seconds_per_step": runtime_budget_seconds,
            "preferred_runtime_seconds_per_step": preferred_runtime_seconds,
            "optimizer": "Adam",
            "learning_rate": 1.0e-4,
        }
        for key, expected in expected_contract.items():
            if contract.get(key) != expected:
                raise ValueError(f"resume artifact measurement contract mismatch: {key}")
        for item in previous.get("points", []):
            result = TrainingPointResult(**item)
            reusable[result.point_id] = result
    results: list[TrainingPointResult] = []
    reused_count = 0
    new_count = 0
    for point in points:
        previous_result = reusable.get(point.point_id)
        if (
            previous_result is not None
            and previous_result.warmup_steps == point.warmup_steps
            and previous_result.repeats == point.repeats
            and previous_result.runtime_budget_seconds == point.runtime_budget_seconds
            and previous_result.preferred_runtime_seconds == point.preferred_runtime_seconds
            and previous_result.maximum_memory_fraction == point.maximum_memory_fraction
            and previous_result.grid_points == point.grid_points
            and previous_result.score_baseline == point.score_baseline
            and previous_result.learning_rate == point.learning_rate
            and previous_result.seed == point.seed
        ):
            results.append(previous_result)
            reused_count += 1
        else:
            results.append(run_point_subprocess(point, timeout_seconds=timeout_seconds))
            new_count += 1
    summary = summarize_scan(results)
    checks = {
        "production_design_has_cuda_cutoff_batch_and_every_2_to_10_horizon": (
            "cuda" not in normalized_devices
            or len(default_scan_points("cuda")) >= 50
        ),
        "cpu_rss_fallback_lane_present": "cpu" in normalized_devices,
        "all_points_have_repeated_measurements": all(result.repeats >= 3 for result in results),
        "policy_has_fifteen_outputs": RecurrentPolicySpec().output_controls == 15,
        "policy_parameter_count_matches_gru10_mlp256": (
            RecurrentPolicySpec().analytic_parameter_count == 72913
        ),
        "all_completed_points_have_finite_objective_and_gradients": all(
            result.objective_finite and result.gradients_finite
            for result in results
            if result.status not in {"exception", "timeout", "oom", "preflight_rejected"}
        ),
        "all_completed_points_pass_density_numerics": all(
            result.numerical_stable
            for result in results
            if result.status not in {"exception", "timeout", "oom", "preflight_rejected"}
        ),
        "memory_is_measured_not_only_analytic": all(
            result.peak_rss_bytes is not None
            and (result.device != "cuda" or result.cuda_peak_allocated_bytes is not None)
            for result in results
            if result.status not in {"exception", "timeout", "oom", "preflight_rejected"}
        ),
        "scan_yields_feasibility_or_registered_falsification": summary["decision"] in {
            "FEASIBLE_2_TO_10_CYCLE_TEACHER_KERNEL",
            "FALSIFIED_AT_REGISTERED_RESOURCE_GATE",
        },
        "isolated_resource_frontier_is_observed": (
            "cuda" not in normalized_devices or summary["resource_frontier_observed"]
        ),
        "runtime_frontier_is_observed": (
            "cuda" not in normalized_devices
            or any(result.status == "runtime_exceeded" for result in results)
        ),
        "memory_frontier_is_observed": (
            "cuda" not in normalized_devices
            or any(
                result.status in {"oom", "memory_exceeded", "preflight_rejected"}
                or (
                    result.observed_memory_fraction is not None
                    and result.observed_memory_fraction >= 0.70
                )
                for result in results
                if result.device == "cuda"
            )
        ),
        "no_unisolated_worker_failure": not summary["fatal_point_ids"],
    }
    policy_payload = asdict(RecurrentPolicySpec())
    policy_payload["analytic_parameter_count"] = RecurrentPolicySpec().analytic_parameter_count
    payload = {
        "schema_version": "1.0",
        "task_id": "T2.3.6",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol_id": SBS_PROTOCOL_ID,
        "scope": DIFFERENTIABLE_SBS_FEASIBILITY_SCOPE,
        "policy_spec": policy_payload,
        "measurement_contract": {
            "workload": "causal 15-output recurrent policy + sampled trajectory + reward/score backward + Adam update",
            "warmup_steps": warmup_steps,
            "timed_repeats": repeats,
            "runtime_statistic": "median with p90 and raw repeats",
            "cpu_memory": "2 ms process-RSS peak sampler in an isolated worker",
            "cuda_memory": "torch max allocated/reserved bytes including backward graph",
            "runtime_budget_seconds_per_step": runtime_budget_seconds,
            "preferred_runtime_seconds_per_step": preferred_runtime_seconds,
            "maximum_memory_fraction": 0.75,
            "real_dtype": "float64",
            "grid_points": 2049,
            "constant_score_baseline": 0.35,
            "optimizer": "Adam",
            "learning_rate": 1.0e-4,
            "subprocess_isolation": True,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "logical_cpu_count": os.cpu_count(),
            "torch": th.__version__,
            "cuda_runtime": th.version.cuda,
            "cuda_available": bool(th.cuda.is_available()),
            "cuda_device": (
                th.cuda.get_device_name(0) if th.cuda.is_available() else None
            ),
        },
        "execution": {
            "resume_enabled": bool(resume),
            "reused_contract_identical_points": reused_count,
            "newly_executed_points": new_count,
        },
        "summary": summary,
        "checks": checks,
        "points": [asdict(result) for result in results],
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    return payload


def _config_from_args(args: argparse.Namespace) -> TrainingPointConfig:
    return TrainingPointConfig(
        cutoff=args.cutoff,
        batch_size=args.batch_size,
        full_cycles=args.full_cycles,
        device=args.device,
        real_dtype=args.real_dtype,
        grid_points=args.grid_points,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        score_baseline=args.score_baseline,
        learning_rate=args.learning_rate,
        runtime_budget_seconds=args.runtime_budget_seconds,
        preferred_runtime_seconds=args.preferred_runtime_seconds,
        maximum_memory_fraction=args.maximum_memory_fraction,
        seed=args.seed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cutoff", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--full-cycles", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=("cuda", "cpu"))
    parser.add_argument("--real-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--grid-points", type=int, default=2049)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--score-baseline", type=float, default=0.35)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--runtime-budget-seconds", type=float, default=10.0)
    parser.add_argument("--preferred-runtime-seconds", type=float, default=2.0)
    parser.add_argument("--maximum-memory-fraction", type=float, default=0.75)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        result = safe_benchmark_training_point(_config_from_args(args))
        print(RESULT_MARKER + json.dumps(asdict(result), ensure_ascii=False))
        return 0
    payload = run_feasibility_scan(
        devices=args.devices,
        output=args.output,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        runtime_budget_seconds=args.runtime_budget_seconds,
        preferred_runtime_seconds=args.preferred_runtime_seconds,
        timeout_seconds=args.timeout_seconds,
        resume=args.resume,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
