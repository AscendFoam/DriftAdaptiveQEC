"""Finite-horizon causal trajectory-lookup control reference.

The lookup policy assigns an independent bounded 15-parameter sBs action to
every *observed history prefix*.  It never indexes the current or a future
measurement.  At ``H`` half-cycles the causal tree therefore contains
``2**H - 1`` action nodes and ``2**H`` terminal measurement trajectories.

All terminal branches are enumerated exactly in the differentiable joint
cavity--two-level-ancilla simulator.  Optimization maximizes the exact
probability-weighted final fidelity, including the dependence of branch
probabilities on earlier actions.  This is a finite-model, finite-cutoff,
multi-start optimization reference; non-convex numerical optimization does
not certify the global optimum and the result is not a deployable controller.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .differentiable_sbs_trajectory import (
    DIFFERENTIABLE_SBS_SCOPE,
    PARAMETER_NAMES,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
    nominal_sbs_parameters,
)

try:  # The repository's minimal recovery interpreter intentionally lacks torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal-environment path.
    torch = None  # type: ignore[assignment]


PolicyFamily = Literal["time_indexed_open_loop", "causal_history_lookup"]

CONTROL_ORACLE_ROLE_ID = "finite_horizon_control_oracle"
ACTION_CONTRACT_ID = "SBS-NOMINAL-PLUS-BOUNDED-RESIDUAL-15"
LOOKUP_SCOPE = (
    "finite-horizon causal history-prefix lookup optimized by exact branch "
    "enumeration in the finite-cutoff two-level differentiable sBs model; "
    "nondeployable assumed-model control-policy reference, not a decoder "
    "oracle, globally certified optimum, pulse/multilevel/device model, or "
    "target-board implementation"
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "trajectory lookup optimization requires PyTorch; use "
            "C:/ProgramData/anaconda3/envs/DLEnv/python.exe"
        )
    return torch


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def history_node_count(half_cycles: int) -> int:
    count = _positive_int(half_cycles, "half_cycles")
    return 2**count - 1


def terminal_branch_count(half_cycles: int) -> int:
    count = _positive_int(half_cycles, "half_cycles")
    return 2**count


def enumerate_terminal_trajectories(half_cycles: int, *, device: str = "cpu") -> Any:
    th = _require_torch()
    count = _positive_int(half_cycles, "half_cycles")
    if count > 20:
        raise ValueError("explicit binary enumeration is capped at twenty outcomes")
    return th.tensor(
        tuple(product((0, 1), repeat=count)),
        dtype=th.int64,
        device=device,
    )


def resource_growth_row(full_cycles: int, *, cutoff: int = 12) -> dict[str, int]:
    cycles = _positive_int(full_cycles, "full_cycles")
    actual_cutoff = _positive_int(cutoff, "cutoff")
    half_cycles = 2 * cycles
    nodes = history_node_count(half_cycles)
    branches = terminal_branch_count(half_cycles)
    scalars = nodes * len(PARAMETER_NAMES)
    table_bytes = scalars * 8
    joint_dimension = 2 * actual_cutoff
    branch_state_bytes = branches * joint_dimension * joint_dimension * 16
    return {
        "full_cycles": cycles,
        "half_cycles": half_cycles,
        "terminal_branches": branches,
        "causal_history_nodes": nodes,
        "lookup_action_scalars": scalars,
        "float64_table_bytes": table_bytes,
        "adam_parameter_gradient_moment_bytes_lower_bound": 4 * table_bytes,
        "complex128_terminal_state_bytes_lower_bound": branch_state_bytes,
    }


@dataclass(frozen=True)
class TrajectoryLookupConfig:
    full_cycles: int = 2
    cutoff: int = 12
    confirmation_cutoff: int = 16
    projector_delta: float = 0.34
    cavity_lifetime_us: float = 245.0
    ancilla_t1_us: float = 50.0
    ancilla_t2_us: float = 60.0
    epochs: int = 300
    learning_rate: float = 2.0e-2
    refinement_epochs: int = 250
    refinement_learning_rate: float = 3.0e-3
    gradient_clip_norm: float = 5.0
    initialization_std: float = 3.0e-2
    restart_seeds: tuple[int, ...] = (101, 211, 307)
    device: Literal["cpu", "cuda"] = "cuda"
    real_dtype: Literal["float32", "float64"] = "float64"
    role_id: str = CONTROL_ORACLE_ROLE_ID
    action_contract_id: str = ACTION_CONTRACT_ID
    scope: str = LOOKUP_SCOPE

    def __post_init__(self) -> None:
        for name in (
            "full_cycles",
            "cutoff",
            "confirmation_cutoff",
            "epochs",
            "refinement_epochs",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if not 1 <= self.full_cycles <= 3:
            raise ValueError("full_cycles must lie in the explicit [1, 3] lookup envelope")
        if not 4 <= self.cutoff <= 48 or not 4 <= self.confirmation_cutoff <= 48:
            raise ValueError("cutoffs must lie in [4, 48]")
        for name in (
            "projector_delta",
            "cavity_lifetime_us",
            "ancilla_t1_us",
            "ancilla_t2_us",
            "learning_rate",
            "refinement_learning_rate",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        initialization_std = float(self.initialization_std)
        if not isfinite(initialization_std) or initialization_std < 0.0:
            raise ValueError("initialization_std must be finite and nonnegative")
        object.__setattr__(self, "initialization_std", initialization_std)
        if self.ancilla_t2_us > 2.0 * self.ancilla_t1_us + 1.0e-12:
            raise ValueError("ancilla_t2_us must not exceed 2*T1")
        seeds = tuple(int(seed) for seed in self.restart_seeds)
        if len(seeds) == 0 or len(set(seeds)) != len(seeds):
            raise ValueError("restart_seeds must be nonempty and unique")
        object.__setattr__(self, "restart_seeds", seeds)
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        if self.role_id != CONTROL_ORACLE_ROLE_ID:
            raise ValueError("role_id must preserve finite-horizon control-oracle naming")
        if self.action_contract_id != ACTION_CONTRACT_ID:
            raise ValueError("action_contract_id must preserve the 15-control contract")
        if self.scope != LOOKUP_SCOPE:
            raise ValueError("scope must remain fail closed")

    @property
    def half_cycles(self) -> int:
        return 2 * self.full_cycles

    @property
    def branch_count(self) -> int:
        return terminal_branch_count(self.half_cycles)

    @property
    def lookup_node_count(self) -> int:
        return history_node_count(self.half_cycles)

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


def validate_production_design(config: TrajectoryLookupConfig) -> None:
    failures: list[str] = []
    if config.full_cycles != 2:
        failures.append("production lookup must use the paper comparison horizon of two cycles")
    if config.cutoff < 12 or config.confirmation_cutoff < 16:
        failures.append("production requires cutoff >=12 and confirmation cutoff >=16")
    if config.epochs < 250:
        failures.append("production requires at least 250 epochs per restart")
    if config.refinement_epochs < 200:
        failures.append("production requires at least 200 refinement epochs per restart")
    if len(config.restart_seeds) < 3:
        failures.append("production requires at least three independent restarts")
    if config.real_dtype != "float64":
        failures.append("production requires float64/complex128 physics")
    if failures:
        raise ValueError("; ".join(failures))


if torch is not None:

    class CausalHistoryLookupPolicy(torch.nn.Module):
        """Independent action at every causal binary-history prefix."""

        family: PolicyFamily = "causal_history_lookup"

        def __init__(
            self,
            half_cycles: int,
            *,
            device: str,
            dtype: Any,
            seed: int = 0,
            initialization_std: float = 0.0,
        ) -> None:
            super().__init__()
            th = _require_torch()
            self.half_cycles = _positive_int(half_cycles, "half_cycles")
            std = float(initialization_std)
            if not isfinite(std) or std < 0.0:
                raise ValueError("initialization_std must be finite and nonnegative")
            generator = th.Generator(device="cpu")
            generator.manual_seed(int(seed))
            values = th.zeros(
                (history_node_count(self.half_cycles), len(PARAMETER_NAMES)),
                dtype=dtype,
            )
            if std > 0.0:
                values = values + std * th.randn(
                    values.shape, dtype=dtype, generator=generator
                )
            self.raw_table = th.nn.Parameter(values.to(device=device))

        @property
        def action_node_count(self) -> int:
            return int(self.raw_table.shape[0])

        @property
        def parameter_count(self) -> int:
            return int(self.raw_table.numel())

        def node_indices(self, history: Any, half_index: int) -> Any:
            th = _require_torch()
            if isinstance(half_index, bool) or not isinstance(half_index, (int, np.integer)):
                raise TypeError("half_index must be an integer")
            index = int(half_index)
            if not 0 <= index < self.half_cycles:
                raise ValueError("half_index lies outside the lookup horizon")
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] != index:
                raise ValueError("history width must equal half_index")
            bits = history.to(device=self.raw_table.device, dtype=th.int64)
            if not bool(th.all((bits == 0) | (bits == 1)).detach().cpu()):
                raise ValueError("history must encode g=0 or e=1")
            if index == 0:
                return th.zeros((bits.shape[0],), dtype=th.int64, device=bits.device)
            powers = 2 ** th.arange(
                index - 1, -1, -1, dtype=th.int64, device=bits.device
            )
            prefix_code = th.sum(bits * powers[None, :], dim=1)
            return (2**index - 1) + prefix_code

        def forward(self, history: Any, half_index: int) -> Any:
            return self.raw_table[self.node_indices(history, half_index)]


    class TimeIndexedOpenLoopPolicy(torch.nn.Module):
        """Optimized time-dependent action with no measurement-history input."""

        family: PolicyFamily = "time_indexed_open_loop"

        def __init__(
            self,
            half_cycles: int,
            *,
            device: str,
            dtype: Any,
            seed: int = 0,
            initialization_std: float = 0.0,
        ) -> None:
            super().__init__()
            th = _require_torch()
            self.half_cycles = _positive_int(half_cycles, "half_cycles")
            std = float(initialization_std)
            if not isfinite(std) or std < 0.0:
                raise ValueError("initialization_std must be finite and nonnegative")
            generator = th.Generator(device="cpu")
            generator.manual_seed(int(seed))
            values = th.zeros((self.half_cycles, len(PARAMETER_NAMES)), dtype=dtype)
            if std > 0.0:
                values = values + std * th.randn(
                    values.shape, dtype=dtype, generator=generator
                )
            self.raw_table = th.nn.Parameter(values.to(device=device))

        @property
        def action_node_count(self) -> int:
            return self.half_cycles

        @property
        def parameter_count(self) -> int:
            return int(self.raw_table.numel())

        def forward(self, history: Any, half_index: int) -> Any:
            th = _require_torch()
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            if not 0 <= int(half_index) < self.half_cycles:
                raise ValueError("half_index lies outside the open-loop horizon")
            return self.raw_table[int(half_index)][None, :].expand(history.shape[0], -1)


else:  # pragma: no cover - import contract for the minimal interpreter.
    CausalHistoryLookupPolicy = None  # type: ignore[assignment]
    TimeIndexedOpenLoopPolicy = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ExactPolicyEvaluation:
    family: str
    cutoff: int
    expected_fidelity: float
    expected_logical_z_signal: float
    expected_code_survival: float
    expected_ground_outcome_fraction: float
    trajectory_probability_sum: float
    minimum_trajectory_probability: float
    maximum_trajectory_probability: float
    maximum_trace_error: float
    maximum_hermiticity_error: float
    minimum_final_eigenvalue: float
    branch_rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class OptimizationRun:
    family: str
    seed: int
    initialization_std: float
    initial_expected_fidelity: float
    best_expected_fidelity: float
    final_expected_fidelity: float
    best_epoch: int
    final_gradient_norm: float
    gradient_covered_nodes: int
    action_node_count: int
    changed_nodes: int
    wall_time_seconds: float
    trace: tuple[Mapping[str, float | int], ...]


def _policy_class(family: PolicyFamily) -> Any:
    if family == "causal_history_lookup":
        return CausalHistoryLookupPolicy
    if family == "time_indexed_open_loop":
        return TimeIndexedOpenLoopPolicy
    raise ValueError("unknown policy family")


def build_policy(
    config: TrajectoryLookupConfig,
    family: PolicyFamily,
    *,
    seed: int,
    initialization_std: float,
) -> Any:
    th = _require_torch()
    dtype = th.float64 if config.real_dtype == "float64" else th.float32
    cls = _policy_class(family)
    return cls(
        config.half_cycles,
        device=config.device,
        dtype=dtype,
        seed=seed,
        initialization_std=initialization_std,
    )


def _exact_result(config: TrajectoryLookupConfig, policy: Any, *, cutoff: int) -> Any:
    outcomes = enumerate_terminal_trajectories(
        config.half_cycles, device=config.device
    )
    simulator = DifferentiableSBSTrajectorySimulator(
        config.simulator_config(cutoff=cutoff)
    )
    return simulator.run(
        control_policy=policy,
        forced_outcomes=outcomes,
        seed=0,
        record_cycle_metrics=True,
    )


def exact_expected_fidelity_tensor(
    config: TrajectoryLookupConfig,
    policy: Any,
    *,
    cutoff: int | None = None,
) -> Any:
    actual_cutoff = config.cutoff if cutoff is None else int(cutoff)
    result = _exact_result(config, policy, cutoff=actual_cutoff)
    return torch.sum(result.trajectory_probability * result.reward)


def evaluate_exact_policy(
    config: TrajectoryLookupConfig,
    policy: Any,
    *,
    cutoff: int | None = None,
) -> ExactPolicyEvaluation:
    th = _require_torch()
    actual_cutoff = config.cutoff if cutoff is None else int(cutoff)
    with th.no_grad():
        result = _exact_result(config, policy, cutoff=actual_cutoff)
        probability = result.trajectory_probability
        final_logical = result.cycle_logical_z_signal[:, -1]
        final_survival = result.cycle_code_survival[:, -1]
        ground_fraction = (result.outcomes == 0).to(probability.dtype).mean(dim=1)
        expected_fidelity = th.sum(probability * result.reward)
        expected_logical = th.sum(probability * final_logical)
        expected_survival = th.sum(probability * final_survival)
        expected_ground = th.sum(probability * ground_fraction)
        branch_rows = []
        for index in range(config.branch_count):
            branch_rows.append(
                {
                    "trajectory": "".join(
                        "g" if int(value) == 0 else "e"
                        for value in result.outcomes[index].detach().cpu().tolist()
                    ),
                    "probability": float(probability[index].detach().cpu()),
                    "final_fidelity": float(result.reward[index].detach().cpu()),
                    "final_logical_z_signal": float(final_logical[index].detach().cpu()),
                    "final_code_survival": float(final_survival[index].detach().cpu()),
                }
            )
    return ExactPolicyEvaluation(
        family=str(getattr(policy, "family", "standard_nominal")),
        cutoff=actual_cutoff,
        expected_fidelity=float(expected_fidelity.detach().cpu()),
        expected_logical_z_signal=float(expected_logical.detach().cpu()),
        expected_code_survival=float(expected_survival.detach().cpu()),
        expected_ground_outcome_fraction=float(expected_ground.detach().cpu()),
        trajectory_probability_sum=float(probability.sum().detach().cpu()),
        minimum_trajectory_probability=float(probability.min().detach().cpu()),
        maximum_trajectory_probability=float(probability.max().detach().cpu()),
        maximum_trace_error=float(result.maximum_trace_error),
        maximum_hermiticity_error=float(result.maximum_hermiticity_error),
        minimum_final_eigenvalue=float(result.minimum_final_eigenvalue),
        branch_rows=tuple(branch_rows),
    )


def optimize_policy_once(
    config: TrajectoryLookupConfig,
    family: PolicyFamily,
    *,
    seed: int,
    initialization_std: float,
    initial_raw_table: Any | None = None,
) -> tuple[OptimizationRun, Mapping[str, Any]]:
    th = _require_torch()
    policy = build_policy(
        config,
        family,
        seed=seed,
        initialization_std=initialization_std,
    )
    if initial_raw_table is not None:
        table = th.as_tensor(
            initial_raw_table,
            dtype=policy.raw_table.dtype,
            device=policy.raw_table.device,
        )
        if tuple(table.shape) != tuple(policy.raw_table.shape):
            raise ValueError("initial_raw_table shape does not match policy family")
        if not bool(th.all(th.isfinite(table)).detach().cpu()):
            raise ValueError("initial_raw_table must be finite")
        with th.no_grad():
            policy.raw_table.copy_(table)
    initial_table = policy.raw_table.detach().clone()
    optimizer = th.optim.Adam((policy.raw_table,), lr=config.learning_rate)
    outcomes = enumerate_terminal_trajectories(config.half_cycles, device=config.device)
    simulator = DifferentiableSBSTrajectorySimulator(config.simulator_config())
    best_value = -float("inf")
    best_epoch = 0
    best_table = initial_table.clone()
    trace: list[Mapping[str, float | int]] = []
    covered_nodes = th.zeros(
        (policy.action_node_count,), dtype=th.bool, device=policy.raw_table.device
    )
    final_gradient_norm = float("nan")
    start = time.perf_counter()

    def objective() -> Any:
        result = simulator.run(
            control_policy=policy,
            forced_outcomes=outcomes,
            seed=0,
            record_cycle_metrics=False,
        )
        probability_sum = result.trajectory_probability.sum()
        if abs(float(probability_sum.detach().cpu()) - 1.0) > 2.0e-10:
            raise RuntimeError("exact trajectory probabilities do not normalize")
        return th.sum(result.trajectory_probability * result.reward)

    initial_value = float("nan")
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        expected = objective()
        current = float(expected.detach().cpu())
        if epoch == 0:
            initial_value = current
        if current > best_value:
            best_value = current
            best_epoch = epoch
            best_table = policy.raw_table.detach().clone()
        (-expected).backward()
        if policy.raw_table.grad is None:
            raise RuntimeError("lookup policy has no gradient")
        if not bool(th.all(th.isfinite(policy.raw_table.grad)).detach().cpu()):
            raise RuntimeError("lookup policy gradient became non-finite")
        node_norm = th.linalg.vector_norm(policy.raw_table.grad, dim=1)
        covered_nodes |= node_norm > 1.0e-14
        final_gradient_norm = float(th.linalg.vector_norm(policy.raw_table.grad).detach().cpu())
        trace.append(
            {
                "epoch": epoch,
                "expected_fidelity": current,
                "gradient_norm": final_gradient_norm,
            }
        )
        th.nn.utils.clip_grad_norm_((policy.raw_table,), config.gradient_clip_norm)
        optimizer.step()

    with th.no_grad():
        final_value = float(objective().detach().cpu())
    trace.append(
        {
            "epoch": config.epochs,
            "expected_fidelity": final_value,
            "gradient_norm": 0.0,
        }
    )
    if final_value > best_value:
        best_value = final_value
        best_epoch = config.epochs
        best_table = policy.raw_table.detach().clone()
    changed = int(
        th.sum(th.linalg.vector_norm(best_table - initial_table, dim=1) > 1.0e-10)
        .detach()
        .cpu()
    )
    run = OptimizationRun(
        family=family,
        seed=int(seed),
        initialization_std=float(initialization_std),
        initial_expected_fidelity=initial_value,
        best_expected_fidelity=best_value,
        final_expected_fidelity=final_value,
        best_epoch=best_epoch,
        final_gradient_norm=final_gradient_norm,
        gradient_covered_nodes=int(covered_nodes.sum().detach().cpu()),
        action_node_count=policy.action_node_count,
        changed_nodes=changed,
        wall_time_seconds=time.perf_counter() - start,
        trace=tuple(trace),
    )
    state = {
        "family": family,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_expected_fidelity": float(best_value),
        "raw_table": best_table.detach().cpu(),
    }
    return run, state


def optimize_policy_multistart(
    config: TrajectoryLookupConfig,
    family: PolicyFamily,
    *,
    first_initial_raw_table: Any | None = None,
) -> tuple[tuple[OptimizationRun, ...], Mapping[str, Any]]:
    runs: list[OptimizationRun] = []
    states: list[Mapping[str, Any]] = []
    for index, seed in enumerate(config.restart_seeds):
        run, state = optimize_policy_once(
            config,
            family,
            seed=seed,
            initialization_std=0.0 if index == 0 else config.initialization_std,
            initial_raw_table=first_initial_raw_table if index == 0 else None,
        )
        runs.append(run)
        states.append(state)
    selected_index = int(np.argmax([run.best_expected_fidelity for run in runs]))
    selected = dict(states[selected_index])
    selected["selected_restart_index"] = selected_index
    selected["all_restart_states"] = tuple(states)
    return tuple(runs), selected


def expand_open_loop_as_lookup(
    config: TrajectoryLookupConfig,
    open_loop_raw_table: Any,
) -> Any:
    """Embed a time-indexed policy exactly into the causal lookup tree."""

    th = _require_torch()
    source = th.as_tensor(open_loop_raw_table)
    expected = (config.half_cycles, len(PARAMETER_NAMES))
    if tuple(source.shape) != expected:
        raise ValueError(f"open_loop_raw_table must have shape {expected}")
    if not bool(th.all(th.isfinite(source)).detach().cpu()):
        raise ValueError("open_loop_raw_table must be finite")
    expanded = th.empty(
        (config.lookup_node_count, len(PARAMETER_NAMES)), dtype=source.dtype
    )
    for depth in range(config.half_cycles):
        start = 2**depth - 1
        stop = 2 ** (depth + 1) - 1
        expanded[start:stop] = source[depth]
    return expanded


def load_policy_from_state(
    config: TrajectoryLookupConfig,
    state: Mapping[str, Any],
    *,
    device: str | None = None,
) -> Any:
    th = _require_torch()
    family = str(state["family"])
    if family not in {"causal_history_lookup", "time_indexed_open_loop"}:
        raise ValueError("checkpoint family is not registered")
    policy = build_policy(
        config,
        family,  # type: ignore[arg-type]
        seed=int(state.get("seed", 0)),
        initialization_std=0.0,
    )
    table = th.as_tensor(state["raw_table"], dtype=policy.raw_table.dtype)
    if tuple(table.shape) != tuple(policy.raw_table.shape):
        raise ValueError("checkpoint raw_table shape does not match policy family")
    target_device = policy.raw_table.device if device is None else th.device(device)
    with th.no_grad():
        policy.raw_table.copy_(table.to(target_device))
    return policy


def standard_nominal_policy(config: TrajectoryLookupConfig) -> Any:
    policy = build_policy(
        config,
        "time_indexed_open_loop",
        seed=0,
        initialization_std=0.0,
    )
    policy.family = "standard_nominal"
    policy.raw_table.requires_grad_(False)
    return policy


def config_to_dict(config: TrajectoryLookupConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["physics_scope"] = DIFFERENTIABLE_SBS_SCOPE
    payload["parameter_names"] = list(PARAMETER_NAMES)
    payload["nominal_controls"] = (
        nominal_sbs_parameters(device="cpu", dtype=_require_torch().float64)
        .detach()
        .cpu()
        .tolist()
    )
    return payload


__all__ = [
    "ACTION_CONTRACT_ID",
    "CONTROL_ORACLE_ROLE_ID",
    "LOOKUP_SCOPE",
    "CausalHistoryLookupPolicy",
    "TimeIndexedOpenLoopPolicy",
    "TrajectoryLookupConfig",
    "ExactPolicyEvaluation",
    "OptimizationRun",
    "history_node_count",
    "terminal_branch_count",
    "enumerate_terminal_trajectories",
    "resource_growth_row",
    "validate_production_design",
    "build_policy",
    "exact_expected_fidelity_tensor",
    "evaluate_exact_policy",
    "optimize_policy_once",
    "optimize_policy_multistart",
    "expand_open_loop_as_lookup",
    "load_policy_from_state",
    "standard_nominal_policy",
    "config_to_dict",
]
