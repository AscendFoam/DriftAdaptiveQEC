"""Interpretable PRL-inspired exponential-saturation sBs control policy.

For the outcome observed after half-cycle ``t``, the raw 15-vector state is

    pi[t+1] = a[m] * pi[t] + (1-a[m]) * pi_inf[m].

The next action therefore sees only the realized causal prefix.  The learned
physical lane trains the g/e branches in the two-level differentiable sBs
model.  A deterministic leakage branch is present in the API and fixed-point
implementation but is not claimed as physically trained without a multilevel
model.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .differentiable_sbs_trajectory import (
    PARAMETER_NAMES,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
)
from .trajectory_lookup_control_oracle import enumerate_terminal_trajectories

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal interpreter.
    torch = None  # type: ignore[assignment]


OUTCOME_NAMES = ("g", "e", "leakage")
EXPONENTIAL_RECURRENCE_SCOPE = (
    "interpretable causal 15-control exponential-saturation recurrence with "
    "learned g/e branches in the finite-cutoff two-level differentiable sBs "
    "model and an explicit uncalibrated leakage-safe branch; software model, "
    "not multilevel/pulse/device, RTL, target-board, teacher-distillation, or "
    "globally optimal control evidence"
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "exponential recurrence requires PyTorch; use the local DLEnv interpreter"
        )
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class ExponentialRecurrenceConfig:
    full_cycles: int = 2
    cutoff: int = 12
    confirmation_cutoff: int = 16
    projector_delta: float = 0.34
    cavity_lifetime_us: float = 245.0
    ancilla_t1_us: float = 50.0
    ancilla_t2_us: float = 60.0
    phase_one_epochs: int = 300
    refinement_epochs: int = 250
    phase_one_learning_rate: float = 2.0e-2
    refinement_learning_rate: float = 3.0e-3
    gradient_clip_norm: float = 5.0
    initialization_std: float = 3.0e-2
    restart_seeds: tuple[int, ...] = (101, 211, 307)
    decay_minimum: float = 0.02
    decay_maximum: float = 0.995
    initial_decay: float = 0.70
    leakage_decay: float = 0.25
    device: Literal["cpu", "cuda"] = "cuda"
    real_dtype: Literal["float32", "float64"] = "float64"
    scope: str = EXPONENTIAL_RECURRENCE_SCOPE

    def __post_init__(self) -> None:
        for name in (
            "full_cycles",
            "cutoff",
            "confirmation_cutoff",
            "phase_one_epochs",
            "refinement_epochs",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if not 1 <= self.full_cycles <= 3:
            raise ValueError("full_cycles must lie in the exact [1,3] envelope")
        if not 4 <= self.cutoff <= 48 or not 4 <= self.confirmation_cutoff <= 48:
            raise ValueError("cutoffs must lie in [4,48]")
        for name in (
            "projector_delta",
            "cavity_lifetime_us",
            "ancilla_t1_us",
            "ancilla_t2_us",
            "phase_one_learning_rate",
            "refinement_learning_rate",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("decay_minimum", "decay_maximum", "initial_decay", "leakage_decay"):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0,1)")
            object.__setattr__(self, name, value)
        if not self.decay_minimum < self.decay_maximum:
            raise ValueError("decay_minimum must be smaller than decay_maximum")
        if not self.decay_minimum < self.initial_decay < self.decay_maximum:
            raise ValueError("initial_decay must lie strictly inside the learned interval")
        std = float(self.initialization_std)
        if not isfinite(std) or std < 0.0:
            raise ValueError("initialization_std must be finite and nonnegative")
        object.__setattr__(self, "initialization_std", std)
        seeds = tuple(int(seed) for seed in self.restart_seeds)
        if len(seeds) == 0 or len(set(seeds)) != len(seeds):
            raise ValueError("restart_seeds must be nonempty and unique")
        object.__setattr__(self, "restart_seeds", seeds)
        if self.ancilla_t2_us > 2.0 * self.ancilla_t1_us + 1.0e-12:
            raise ValueError("ancilla_t2_us must not exceed 2*T1")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        if self.scope != EXPONENTIAL_RECURRENCE_SCOPE:
            raise ValueError("scope must remain fail closed")

    @property
    def half_cycles(self) -> int:
        return 2 * self.full_cycles

    @property
    def branch_count(self) -> int:
        return 2**self.half_cycles

    def simulator_config(self, *, cutoff: int | None = None) -> DifferentiableSBSConfig:
        return DifferentiableSBSConfig(
            cutoff=self.cutoff if cutoff is None else int(cutoff),
            full_cycles=self.full_cycles,
            batch_size=self.branch_count,
            projector_delta=self.projector_delta,
            cavity_lifetime_us=self.cavity_lifetime_us,
            ancilla_t1_us=self.ancilla_t1_us,
            ancilla_t2_us=self.ancilla_t2_us,
            device=self.device,
            real_dtype=self.real_dtype,
        )


def validate_production_design(config: ExponentialRecurrenceConfig) -> None:
    failures = []
    if config.full_cycles != 2:
        failures.append("production requires the two-cycle exact comparison horizon")
    if config.cutoff < 12 or config.confirmation_cutoff < 16:
        failures.append("production requires cutoff12/cutoff16 or stronger")
    if config.phase_one_epochs < 250 or config.refinement_epochs < 200:
        failures.append("production requires >=250 phase-one and >=200 refinement epochs")
    if len(config.restart_seeds) < 3:
        failures.append("production requires at least three restarts")
    if config.real_dtype != "float64":
        failures.append("production requires float64 physics")
    if failures:
        raise ValueError("; ".join(failures))


def _decay_logit(config: ExponentialRecurrenceConfig, value: float) -> float:
    normalized = (value - config.decay_minimum) / (
        config.decay_maximum - config.decay_minimum
    )
    return float(np.log(normalized / (1.0 - normalized)))


if torch is not None:

    class ExponentialSaturationControlPolicy(torch.nn.Module):
        family = "prl_exponential_recurrence"

        def __init__(
            self,
            config: ExponentialRecurrenceConfig,
            *,
            seed: int,
            initialization_std: float,
        ) -> None:
            super().__init__()
            th = _require_torch()
            dtype = th.float64 if config.real_dtype == "float64" else th.float32
            generator = th.Generator(device="cpu")
            generator.manual_seed(int(seed))
            std = float(initialization_std)
            if not isfinite(std) or std < 0.0:
                raise ValueError("initialization_std must be finite and nonnegative")
            initial = th.zeros((15,), dtype=dtype)
            saturation = th.zeros((2, 15), dtype=dtype)
            if std > 0.0:
                initial += std * th.randn(initial.shape, dtype=dtype, generator=generator)
                saturation += std * th.randn(
                    saturation.shape, dtype=dtype, generator=generator
                )
            logit = _decay_logit(config, config.initial_decay)
            decay_logits = th.full((2, 15), logit, dtype=dtype)
            if std > 0.0:
                decay_logits += std * th.randn(
                    decay_logits.shape, dtype=dtype, generator=generator
                )
            self.initial_raw = th.nn.Parameter(initial.to(config.device))
            self.ge_saturation_raw = th.nn.Parameter(saturation.to(config.device))
            self.ge_decay_logits = th.nn.Parameter(decay_logits.to(config.device))
            self.register_buffer(
                "leakage_saturation_raw",
                th.zeros((15,), dtype=dtype, device=config.device),
            )
            self.register_buffer(
                "leakage_decay",
                th.full((15,), config.leakage_decay, dtype=dtype, device=config.device),
            )
            self.decay_minimum = config.decay_minimum
            self.decay_maximum = config.decay_maximum
            self.half_cycles = config.half_cycles

        @property
        def parameter_count(self) -> int:
            return int(sum(parameter.numel() for parameter in self.parameters()))

        @property
        def stored_scalar_count(self) -> int:
            return self.parameter_count + int(
                self.leakage_saturation_raw.numel() + self.leakage_decay.numel()
            )

        def ge_decay(self) -> Any:
            th = _require_torch()
            return self.decay_minimum + (
                self.decay_maximum - self.decay_minimum
            ) * th.sigmoid(self.ge_decay_logits)

        def all_saturations(self) -> Any:
            return torch.cat(
                (self.ge_saturation_raw, self.leakage_saturation_raw[None, :]), dim=0
            )

        def all_decays(self) -> Any:
            return torch.cat((self.ge_decay(), self.leakage_decay[None, :]), dim=0)

        def state_after_history(self, history: Any) -> Any:
            th = _require_torch()
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] > self.half_cycles:
                raise ValueError("history exceeds the recurrence horizon")
            outcomes = history.to(device=self.initial_raw.device, dtype=th.int64)
            if not bool(th.all((outcomes >= 0) & (outcomes <= 2)).detach().cpu()):
                raise ValueError("history outcomes must encode g=0, e=1, leakage=2")
            state = self.initial_raw[None, :].expand(outcomes.shape[0], -1)
            saturations = self.all_saturations()
            decays = self.all_decays()
            for index in range(outcomes.shape[1]):
                outcome = outcomes[:, index]
                decay = decays[outcome]
                saturation = saturations[outcome]
                state = decay * state + (1.0 - decay) * saturation
            return state

        def forward(self, history: Any, half_index: int) -> Any:
            if history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            if not 0 <= int(half_index) < self.half_cycles:
                raise ValueError("half_index lies outside the configured horizon")
            return self.state_after_history(history)


    class FixedPointExponentialPolicy(torch.nn.Module):
        family = "prl_exponential_recurrence_fixed_point"

        def __init__(
            self,
            source: ExponentialSaturationControlPolicy,
            *,
            state_fraction_bits: int = 14,
            decay_fraction_bits: int = 16,
            state_total_bits: int = 18,
        ) -> None:
            super().__init__()
            th = _require_torch()
            for name, value, lower, upper in (
                ("state_fraction_bits", state_fraction_bits, 4, 24),
                ("decay_fraction_bits", decay_fraction_bits, 8, 24),
                ("state_total_bits", state_total_bits, state_fraction_bits + 2, 31),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                    raise ValueError(f"{name} lies outside the supported range")
            self.state_fraction_bits = state_fraction_bits
            self.decay_fraction_bits = decay_fraction_bits
            self.state_total_bits = state_total_bits
            self.half_cycles = source.half_cycles
            state_scale = 2**state_fraction_bits
            decay_scale = 2**decay_fraction_bits
            state_min = -(2 ** (state_total_bits - 1))
            state_max = 2 ** (state_total_bits - 1) - 1
            with th.no_grad():
                initial = th.clamp(
                    th.round(source.initial_raw.detach() * state_scale), state_min, state_max
                ).to(th.int64)
                saturation = th.clamp(
                    th.round(source.all_saturations().detach() * state_scale),
                    state_min,
                    state_max,
                ).to(th.int64)
                decay = th.clamp(
                    th.round(source.all_decays().detach() * decay_scale),
                    0,
                    decay_scale,
                ).to(th.int64)
            self.register_buffer("initial_code", initial)
            self.register_buffer("saturation_code", saturation)
            self.register_buffer("decay_code", decay)

        @staticmethod
        def _round_divide_signed(numerator: Any, denominator: int) -> Any:
            positive = (numerator + denominator // 2) // denominator
            negative = -((-numerator + denominator // 2) // denominator)
            return torch.where(numerator >= 0, positive, negative)

        def state_codes_after_history(self, history: Any) -> Any:
            th = _require_torch()
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] > self.half_cycles:
                raise ValueError("history exceeds the recurrence horizon")
            outcomes = history.to(device=self.initial_code.device, dtype=th.int64)
            if not bool(th.all((outcomes >= 0) & (outcomes <= 2)).detach().cpu()):
                raise ValueError("history outcomes must encode g=0, e=1, leakage=2")
            state = self.initial_code[None, :].expand(outcomes.shape[0], -1).clone()
            decay_scale = 2**self.decay_fraction_bits
            lower = -(2 ** (self.state_total_bits - 1))
            upper = 2 ** (self.state_total_bits - 1) - 1
            for index in range(outcomes.shape[1]):
                outcome = outcomes[:, index]
                decay = self.decay_code[outcome]
                saturation = self.saturation_code[outcome]
                numerator = decay * state + (decay_scale - decay) * saturation
                state = th.clamp(
                    self._round_divide_signed(numerator, decay_scale), lower, upper
                )
            return state

        def state_after_history(self, history: Any) -> Any:
            """Replay a complete or partial prefix, including the terminal prefix.

            This diagnostic method deliberately accepts ``half_cycles`` observed
            outcomes.  ``forward`` remains stricter because there is no action
            after the terminal observation.
            """

            return self.state_codes_after_history(history).to(torch.float64) / (
                2**self.state_fraction_bits
            )

        def forward(self, history: Any, half_index: int) -> Any:
            if history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            if not 0 <= int(half_index) < self.half_cycles:
                raise ValueError("half_index lies outside the configured horizon")
            return self.state_after_history(history)


else:  # pragma: no cover
    ExponentialSaturationControlPolicy = None  # type: ignore[assignment]
    FixedPointExponentialPolicy = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RecurrenceOptimizationRun:
    seed: int
    phase: str
    initial_expected_fidelity: float
    best_expected_fidelity: float
    final_expected_fidelity: float
    best_epoch: int
    gradient_covered_scalars: int
    changed_scalars: int
    trainable_scalars: int
    final_gradient_norm: float
    wall_time_seconds: float
    trace: tuple[Mapping[str, float | int], ...]


def build_policy(
    config: ExponentialRecurrenceConfig,
    *,
    seed: int,
    initialization_std: float,
) -> Any:
    return ExponentialSaturationControlPolicy(
        config, seed=seed, initialization_std=initialization_std
    )


def _state_dict_cpu(policy: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_policy_state(
    config: ExponentialRecurrenceConfig, state: Mapping[str, Any]
) -> Any:
    policy = build_policy(config, seed=0, initialization_std=0.0)
    expected = policy.state_dict()
    if set(state) != set(expected):
        raise ValueError("recurrence checkpoint keys do not match")
    for name, value in state.items():
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError(f"recurrence checkpoint shape mismatch for {name}")
    policy.load_state_dict(state)
    return policy


def exact_expected_fidelity(
    config: ExponentialRecurrenceConfig, policy: Any, *, cutoff: int | None = None
) -> Any:
    actual_cutoff = config.cutoff if cutoff is None else int(cutoff)
    simulator = DifferentiableSBSTrajectorySimulator(
        config.simulator_config(cutoff=actual_cutoff)
    )
    outcomes = enumerate_terminal_trajectories(
        config.half_cycles, device=config.device
    )
    result = simulator.run(
        control_policy=policy,
        forced_outcomes=outcomes,
        seed=0,
        record_cycle_metrics=False,
    )
    probability_sum = result.trajectory_probability.sum()
    if abs(float(probability_sum.detach().cpu()) - 1.0) > 2.0e-10:
        raise RuntimeError("exact trajectory probabilities do not normalize")
    return torch.sum(result.trajectory_probability * result.reward)


def _flatten_trainable(policy: Any, *, gradients: bool = False) -> Any:
    values = []
    for parameter in policy.parameters():
        value = parameter.grad if gradients else parameter
        if value is None:
            raise RuntimeError("trainable recurrence parameter has no gradient")
        values.append(value.reshape(-1))
    return torch.cat(values)


def optimize_recurrence_once(
    config: ExponentialRecurrenceConfig,
    *,
    seed: int,
    phase: str,
    epochs: int,
    learning_rate: float,
    initialization_std: float,
    initial_state: Mapping[str, Any] | None = None,
) -> tuple[RecurrenceOptimizationRun, Mapping[str, Any]]:
    th = _require_torch()
    policy = build_policy(config, seed=seed, initialization_std=initialization_std)
    if initial_state is not None:
        policy = load_policy_state(config, initial_state)
    initial_vector = _flatten_trainable(policy).detach().clone()
    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    covered = th.zeros_like(initial_vector, dtype=th.bool)
    best_value = -float("inf")
    best_epoch = 0
    best_state = _state_dict_cpu(policy)
    trace = []
    final_gradient_norm = float("nan")
    start = time.perf_counter()
    initial_value = float("nan")
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        expected = exact_expected_fidelity(config, policy)
        value = float(expected.detach().cpu())
        if epoch == 0:
            initial_value = value
        if value > best_value:
            best_value = value
            best_epoch = epoch
            best_state = _state_dict_cpu(policy)
        (-expected).backward()
        gradients = _flatten_trainable(policy, gradients=True)
        if not bool(th.all(th.isfinite(gradients)).detach().cpu()):
            raise RuntimeError("recurrence gradient became non-finite")
        covered |= th.abs(gradients) > 1.0e-14
        final_gradient_norm = float(th.linalg.vector_norm(gradients).detach().cpu())
        trace.append(
            {
                "epoch": epoch,
                "expected_fidelity": value,
                "gradient_norm": final_gradient_norm,
            }
        )
        th.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip_norm)
        optimizer.step()
    with th.no_grad():
        final_value = float(exact_expected_fidelity(config, policy).detach().cpu())
    trace.append(
        {
            "epoch": epochs,
            "expected_fidelity": final_value,
            "gradient_norm": 0.0,
        }
    )
    if final_value > best_value:
        best_value = final_value
        best_epoch = epochs
        best_state = _state_dict_cpu(policy)
    best_policy = load_policy_state(config, best_state)
    changed = int(
        torch.sum(
            torch.abs(_flatten_trainable(best_policy).detach() - initial_vector) > 1.0e-10
        ).cpu()
    )
    run = RecurrenceOptimizationRun(
        seed=int(seed),
        phase=str(phase),
        initial_expected_fidelity=initial_value,
        best_expected_fidelity=best_value,
        final_expected_fidelity=final_value,
        best_epoch=best_epoch,
        gradient_covered_scalars=int(covered.sum().detach().cpu()),
        changed_scalars=changed,
        trainable_scalars=int(initial_vector.numel()),
        final_gradient_norm=final_gradient_norm,
        wall_time_seconds=time.perf_counter() - start,
        trace=tuple(trace),
    )
    return run, best_state


def optimize_recurrence_multistart(
    config: ExponentialRecurrenceConfig,
) -> tuple[
    tuple[RecurrenceOptimizationRun, ...],
    tuple[RecurrenceOptimizationRun, ...],
    Mapping[str, Any],
]:
    phase_runs = []
    phase_states = []
    for index, seed in enumerate(config.restart_seeds):
        run, state = optimize_recurrence_once(
            config,
            seed=seed,
            phase="phase_one",
            epochs=config.phase_one_epochs,
            learning_rate=config.phase_one_learning_rate,
            initialization_std=0.0 if index == 0 else config.initialization_std,
        )
        phase_runs.append(run)
        phase_states.append(state)
    refinement_runs = []
    refinement_states = []
    for seed, state in zip(config.restart_seeds, phase_states, strict=True):
        run, refined = optimize_recurrence_once(
            config,
            seed=seed,
            phase="refinement",
            epochs=config.refinement_epochs,
            learning_rate=config.refinement_learning_rate,
            initialization_std=0.0,
            initial_state=state,
        )
        refinement_runs.append(run)
        refinement_states.append(refined)
    selected_index = int(
        np.argmax([run.best_expected_fidelity for run in refinement_runs])
    )
    selected = {
        "selected_restart_index": selected_index,
        "seed": int(config.restart_seeds[selected_index]),
        "state_dict": refinement_states[selected_index],
        "state_dict_sha256": state_dict_sha256(refinement_states[selected_index]),
        "all_phase_one_states": tuple(phase_states),
        "all_refinement_states": tuple(refinement_states),
    }
    return tuple(phase_runs), tuple(refinement_runs), selected


def config_to_dict(config: ExponentialRecurrenceConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["parameter_names"] = list(PARAMETER_NAMES)
    payload["trainable_scalars"] = 75
    payload["stored_scalars_including_leakage_branch"] = 105
    return payload


__all__ = [
    "OUTCOME_NAMES",
    "EXPONENTIAL_RECURRENCE_SCOPE",
    "ExponentialRecurrenceConfig",
    "ExponentialSaturationControlPolicy",
    "FixedPointExponentialPolicy",
    "RecurrenceOptimizationRun",
    "validate_production_design",
    "build_policy",
    "state_dict_sha256",
    "load_policy_state",
    "exact_expected_fidelity",
    "optimize_recurrence_once",
    "optimize_recurrence_multistart",
    "config_to_dict",
]
