"""Versioned Puviani/GQF causal adapter for the Phase-9 artifact lane.

This module intentionally leaves the historical T2.3.4/T2.3.7 backend bytes
unchanged.  It adds the source-faithful decision timeline used by the public
GQF implementation:

``d0 -> layers -> m0 -> d1.VR -> d1.layers -> ... -> m(H-1) -> dH.VR``.

There are H measurements and H+1 policy decisions.  ``d0.VR`` and
``dH.layers`` are never physically applied and are therefore excluded from
regularization and from the applied-control fields below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
import time
from typing import Any, Literal, Sequence

import numpy as np

from .differentiable_sbs_trajectory import (
    DIFFERENTIABLE_SBS_SCOPE,
    PARAMETER_NAMES,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
    TrajectoryResourceProfile,
    nominal_sbs_parameters,
)
from .fock_density_model import FiniteCutoffFockModel
from .nmf_directional_ranking import (
    DirectionalRankingConfig,
    PaperScaleMFPolicy as LegacyPaperScaleMFPolicy,
    PaperScaleNMFPolicy as LegacyPaperScaleNMFPolicy,
    _effective_lifetime,
)

try:  # The minimal recovery interpreter deliberately omits torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


PAPER_CAUSAL_FEEDBACK_TIMELINE = (
    "measurement_to_next_interval_with_terminal_vr"
)
GQF_OBSERVATION_ENCODING = "gqf_g_plus1_e_minus1"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "the Phase-9 Puviani adapter requires the DLEnv PyTorch runtime"
        )
    return torch


@dataclass(frozen=True)
class PaperCausalSBSConfig(DifferentiableSBSConfig):
    """Historical simulator configuration plus an immutable causal contract."""

    feedback_timeline: str = PAPER_CAUSAL_FEEDBACK_TIMELINE

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.feedback_timeline != PAPER_CAUSAL_FEEDBACK_TIMELINE:
            raise ValueError("the Phase-9 adapter only implements paper-causal timing")


@dataclass
class PaperCausalTrajectoryResult:
    final_joint_density: Any
    final_cavity_density: Any
    outcomes: Any
    conditional_probabilities: Any
    log_probability: Any
    trajectory_probability: Any
    reward: Any
    decision_physical_controls: Any
    layer_applied_physical_controls: Any
    virtual_rotation_applied_physical_controls: Any
    terminal_virtual_rotation: Any
    cycle_fidelities: Any | None
    cycle_code_survival: Any | None
    cycle_logical_z_signal: Any | None
    cycle_conditional_logical_z: Any | None
    resource_profile: TrajectoryResourceProfile
    maximum_trace_error: float
    maximum_hermiticity_error: float
    minimum_final_eigenvalue: float
    raw_logical_codeword_overlap: complex
    raw_logical_gram_condition_number: float
    feedback_timeline: str = PAPER_CAUSAL_FEEDBACK_TIMELINE
    protocol_id: str = ""
    scope: str = DIFFERENTIABLE_SBS_SCOPE

    def detached_summary(self) -> dict[str, Any]:
        return {
            "outcomes": self.outcomes.detach().cpu().tolist(),
            "conditional_probabilities": (
                self.conditional_probabilities.detach().cpu().tolist()
            ),
            "trajectory_probability": (
                self.trajectory_probability.detach().cpu().tolist()
            ),
            "reward": self.reward.detach().cpu().tolist(),
            "decision_physical_controls": (
                self.decision_physical_controls.detach().cpu().tolist()
            ),
            "layer_applied_physical_controls": (
                self.layer_applied_physical_controls.detach().cpu().tolist()
            ),
            "virtual_rotation_applied_physical_controls": (
                self.virtual_rotation_applied_physical_controls.detach()
                .cpu()
                .tolist()
            ),
            "terminal_virtual_rotation": (
                self.terminal_virtual_rotation.detach().cpu().tolist()
            ),
            "resource_profile": asdict(self.resource_profile),
            "feedback_timeline": self.feedback_timeline,
        }


class PaperCausalSBSTrajectorySimulator(DifferentiableSBSTrajectorySimulator):
    """Source-faithful H-measurement/H+1-decision trajectory simulator."""

    config: PaperCausalSBSConfig

    def __init__(self, config: PaperCausalSBSConfig) -> None:
        if not isinstance(config, PaperCausalSBSConfig):
            raise TypeError("config must be PaperCausalSBSConfig")
        super().__init__(config)
        zero = self.initial_state_vector.detach().cpu().numpy()
        one_preparation = FiniteCutoffFockModel(self.cutoff).prepare_damped_projector_gkp(
            "1",
            self.config.projector_delta,
            grid_points=self.config.grid_points,
            source_coordinate_scale=sqrt(2.0),
        )
        one = np.asarray(one_preparation.coefficients, dtype=np.complex128)
        one = one / np.linalg.norm(one)
        gram = np.asarray(
            [
                [np.vdot(zero, zero), np.vdot(zero, one)],
                [np.vdot(one, zero), np.vdot(one, one)],
            ],
            dtype=np.complex128,
        )
        self.raw_logical_codeword_overlap = complex(gram[0, 1])
        self.raw_logical_gram_condition_number = float(np.linalg.cond(gram))

    def bounded_decision_controls(self, raw_corrections: Any | None = None) -> Any:
        """Map a complete H+1 open-loop decision schedule to physical values."""

        th = _require_torch()
        expected = (
            self.config.batch_size,
            self.config.half_cycles + 1,
            len(PARAMETER_NAMES),
        )
        if raw_corrections is None:
            raw = th.zeros(expected, dtype=self.real_dtype, device=self.device)
        else:
            if not isinstance(raw_corrections, th.Tensor):
                raise TypeError("raw_corrections must be a torch.Tensor")
            raw = raw_corrections.to(device=self.device, dtype=self.real_dtype)
            if raw.ndim == 2 and tuple(raw.shape) == expected[1:]:
                raw = raw.unsqueeze(0).expand(expected)
            if tuple(raw.shape) != expected:
                raise ValueError(
                    f"paper-causal raw_corrections must have shape {expected} "
                    f"or {expected[1:]}"
                )
            if not bool(th.all(th.isfinite(raw)).detach().cpu()):
                raise ValueError("raw_corrections must be finite")
        return self._map_bounded_corrections(raw)

    def _measure_collapsed(
        self,
        state: Any,
        forced_outcome: Any | None,
        generator: Any,
    ) -> tuple[Any, Any, Any]:
        """Project without resetting so the selected branch receives 2.3 us noise."""

        th = _require_torch()
        probabilities, unnormalized_cavity = self._measurement_probabilities(state)
        if forced_outcome is None:
            uniforms = th.rand(
                (self.config.batch_size,), generator=generator, dtype=th.float64
            ).to(self.device)
            outcomes = (uniforms >= probabilities[:, 0].detach()).to(th.int64)
        else:
            outcomes = forced_outcome.to(device=self.device, dtype=th.int64)
            if tuple(outcomes.shape) != (self.config.batch_size,):
                raise ValueError("forced outcome slice must have shape (batch_size,)")
            if not bool(th.all((outcomes == 0) | (outcomes == 1)).detach().cpu()):
                raise ValueError("forced outcomes must encode g=0 or e=1")
        batch = th.arange(self.config.batch_size, device=self.device)
        selected_probability = probabilities[batch, outcomes]
        if bool(
            th.any(selected_probability.detach() <= self.config.probability_floor).cpu()
        ):
            raise RuntimeError("trajectory contains a numerically impossible branch")
        cavity = unnormalized_cavity[batch, outcomes]
        cavity = cavity / selected_probability[:, None, None].to(self.complex_dtype)
        selected_g = outcomes == 0
        collapsed = th.zeros_like(state)
        blocks = collapsed.reshape(
            self.config.batch_size, self.cutoff, 2, self.cutoff, 2
        )
        blocks[selected_g, :, 0, :, 0] = cavity[selected_g]
        blocks[~selected_g, :, 1, :, 1] = cavity[~selected_g]
        return self._stabilize_density(collapsed), outcomes, selected_probability

    def _reset_measured_ancilla(self, state: Any) -> Any:
        th = _require_torch()
        cavity = self._reduce_cavity(state)
        reset = th.einsum("bij,kl->bikjl", cavity, self.g_projector).reshape(
            self.config.batch_size, self.joint_dimension, self.joint_dimension
        )
        return self._stabilize_density(reset)

    def run(
        self,
        raw_corrections: Any | None = None,
        *,
        control_policy: Any | None = None,
        forced_outcomes: Any | Sequence[Sequence[int]] | None = None,
        seed: int = 0,
        record_cycle_metrics: bool = False,
    ) -> PaperCausalTrajectoryResult:
        th = _require_torch()
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        if raw_corrections is not None and control_policy is not None:
            raise ValueError("provide raw_corrections or control_policy, not both")
        open_loop = (
            None
            if control_policy is not None
            else self.bounded_decision_controls(raw_corrections)
        )
        forced = None
        if forced_outcomes is not None:
            forced = th.as_tensor(
                forced_outcomes, dtype=th.int64, device=self.device
            )
            expected = (self.config.batch_size, self.config.half_cycles)
            if tuple(forced.shape) != expected:
                raise ValueError(f"forced_outcomes must have shape {expected}")
            if not bool(th.all((forced == 0) | (forced == 1)).detach().cpu()):
                raise ValueError("forced outcomes must encode g=0 or e=1")

        generator = th.Generator(device="cpu")
        generator.manual_seed(int(seed))
        if self.config.device == "cuda":
            th.cuda.synchronize(self.device)
            th.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        state = self._initial_joint_density()
        if control_policy is not None:
            reset_rollout = getattr(control_policy, "reset_rollout", None)
            if callable(reset_rollout):
                reset_rollout(
                    batch_size=self.config.batch_size,
                    device=self.device,
                    dtype=self.real_dtype,
                )

        outcomes: list[Any] = []
        probabilities: list[Any] = []
        decisions: list[Any] = []
        cycle_fidelities: list[Any] = []
        cycle_code_survival: list[Any] = []
        cycle_logical_z_signal: list[Any] = []
        cycle_conditional_logical_z: list[Any] = []
        if record_cycle_metrics:
            initial = self._cavity_evaluation_metrics(self._reduce_cavity(state))
            cycle_fidelities.append(initial[0])
            cycle_code_survival.append(initial[1])
            cycle_logical_z_signal.append(initial[2])
            cycle_conditional_logical_z.append(initial[3])

        log_probability = th.zeros(
            (self.config.batch_size,), dtype=self.real_dtype, device=self.device
        )
        maximum_trace_error = 0.0
        maximum_hermiticity_error = 0.0

        def history() -> Any:
            if not outcomes:
                return th.empty(
                    (self.config.batch_size, 0),
                    dtype=th.int64,
                    device=self.device,
                )
            return th.stack(outcomes, dim=1)

        def decision(prefix_length: int) -> Any:
            if control_policy is None:
                return open_loop[:, prefix_length, :]
            return self._policy_controls(control_policy, history(), prefix_length)

        current = decision(0)
        decisions.append(current)
        for half_index in range(self.config.half_cycles):
            state = self._apply_idle(state, "entering_cycle")
            for layer in range(1, 5):
                state = self._layer(state, current, layer)

            state, outcome, selected = self._measure_collapsed(
                state,
                None if forced is None else forced[:, half_index],
                generator,
            )
            outcomes.append(outcome)
            probabilities.append(selected)
            log_probability = log_probability + th.log(
                th.clamp(selected, min=self.config.probability_floor)
            )
            following = decision(half_index + 1)
            decisions.append(following)
            state = self._apply_idle(state, "measurement_and_reset")
            state = self._reset_measured_ancilla(state)
            state = self._virtual_rotation(state, following[:, 14])
            state = self._apply_idle(state, "virtual_rotation_and_idle")
            current = following

            trace_error, hermiticity_error, _ = self._diagnostics(state)
            maximum_trace_error = max(maximum_trace_error, trace_error)
            maximum_hermiticity_error = max(
                maximum_hermiticity_error, hermiticity_error
            )
            if record_cycle_metrics and (half_index + 1) % 2 == 0:
                metrics = self._cavity_evaluation_metrics(self._reduce_cavity(state))
                cycle_fidelities.append(metrics[0])
                cycle_code_survival.append(metrics[1])
                cycle_logical_z_signal.append(metrics[2])
                cycle_conditional_logical_z.append(metrics[3])

        cavity = self._reduce_cavity(state)
        reward = self._cavity_evaluation_metrics(cavity)[0]
        trajectory_probability = th.exp(log_probability)
        if self.config.device == "cuda":
            th.cuda.synchronize(self.device)
            peak = int(th.cuda.max_memory_allocated(self.device))
        else:
            peak = None
        wall_time = time.perf_counter() - started
        final_trace, final_hermiticity, minimum_eigenvalue = self._diagnostics(state)
        maximum_trace_error = max(maximum_trace_error, final_trace)
        maximum_hermiticity_error = max(maximum_hermiticity_error, final_hermiticity)

        stacked_decisions = th.stack(decisions, dim=1)
        half_cycles = self.config.half_cycles
        complex_bytes = 16 if self.complex_dtype == th.complex128 else 8
        state_bytes = (
            self.config.batch_size
            * self.joint_dimension
            * self.joint_dimension
            * complex_bytes
        )
        resources = TrajectoryResourceProfile(
            device=str(self.device),
            real_dtype=self.config.real_dtype,
            complex_dtype=str(self.complex_dtype).replace("torch.", ""),
            cutoff=self.cutoff,
            joint_dimension=self.joint_dimension,
            batch_size=self.config.batch_size,
            full_cycles=self.config.full_cycles,
            half_cycles=half_cycles,
            control_source=(
                "history_conditioned_policy"
                if control_policy is not None
                else "paper_causal_open_loop_decisions"
            ),
            trainable_controls=(
                self.config.batch_size
                * (half_cycles + 1)
                * len(PARAMETER_NAMES)
            ),
            matrix_exponentials=self.config.batch_size * half_cycles * 4,
            unitary_applications=self.config.batch_size * half_cycles * 9,
            idle_windows=self.config.batch_size * half_cycles * 7,
            cptp_channel_applications=self.config.batch_size * half_cycles * 21,
            state_tensor_bytes=state_bytes,
            autograd_state_lower_bound_bytes=state_bytes * (1 + 7 * half_cycles),
            wall_time_seconds=wall_time,
            cuda_peak_allocated_bytes=peak,
            timing_profile_id=self.config.timing.profile_id,
            simulated_physical_time_ns=(
                half_cycles * self.config.timing.half_cycle_duration_ns
            ),
        )
        return PaperCausalTrajectoryResult(
            final_joint_density=state,
            final_cavity_density=cavity,
            outcomes=th.stack(outcomes, dim=1),
            conditional_probabilities=th.stack(probabilities, dim=1),
            log_probability=log_probability,
            trajectory_probability=trajectory_probability,
            reward=reward,
            decision_physical_controls=stacked_decisions,
            layer_applied_physical_controls=stacked_decisions[:, :-1, :14],
            virtual_rotation_applied_physical_controls=(
                stacked_decisions[:, 1:, 14:15]
            ),
            terminal_virtual_rotation=stacked_decisions[:, -1, 14],
            cycle_fidelities=(
                th.stack(cycle_fidelities, dim=1) if record_cycle_metrics else None
            ),
            cycle_code_survival=(
                th.stack(cycle_code_survival, dim=1)
                if record_cycle_metrics
                else None
            ),
            cycle_logical_z_signal=(
                th.stack(cycle_logical_z_signal, dim=1)
                if record_cycle_metrics
                else None
            ),
            cycle_conditional_logical_z=(
                th.stack(cycle_conditional_logical_z, dim=1)
                if record_cycle_metrics
                else None
            ),
            resource_profile=resources,
            maximum_trace_error=maximum_trace_error,
            maximum_hermiticity_error=maximum_hermiticity_error,
            minimum_final_eigenvalue=minimum_eigenvalue,
            raw_logical_codeword_overlap=self.raw_logical_codeword_overlap,
            raw_logical_gram_condition_number=(
                self.raw_logical_gram_condition_number
            ),
            protocol_id=self.config.protocol_id,
        )


@dataclass(frozen=True)
class PaperCausalDirectionalRankingConfig(DirectionalRankingConfig):
    feedback_timeline: str = PAPER_CAUSAL_FEEDBACK_TIMELINE
    observation_encoding: str = GQF_OBSERVATION_ENCODING
    consume_initial_zero_sentinel: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.feedback_timeline != PAPER_CAUSAL_FEEDBACK_TIMELINE:
            raise ValueError("paper-causal feedback_timeline is immutable")
        if self.observation_encoding != GQF_OBSERVATION_ENCODING:
            raise ValueError("GQF observations must encode g=+1 and e=-1")
        if self.consume_initial_zero_sentinel is not True:
            raise ValueError("the public GQF GRU consumes an initial zero sentinel")


if torch is not None:

    class PaperCausalMFPolicy(LegacyPaperScaleMFPolicy):
        def forward(self, history: Any, half_index: int) -> Any:
            th = _require_torch()
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            reference = next(self.parameters())
            latest = (
                th.zeros(
                    (history.shape[0], 1),
                    dtype=reference.dtype,
                    device=reference.device,
                )
                if half_index == 0
                else (1.0 - 2.0 * history[:, -1:]).to(reference.dtype)
            )
            value = th.tanh(self.dense1(latest))
            value = th.tanh(self.dense2(value))
            return self.output(value)


    class PaperCausalNMFPolicy(LegacyPaperScaleNMFPolicy):
        def reset_rollout(self, *, batch_size: int, device: Any, dtype: Any) -> None:
            super().reset_rollout(batch_size=batch_size, device=device, dtype=dtype)
            self._next_expected_prefix = 0

        def step_rollout(self, history: Any, half_index: int) -> Any:
            th = _require_torch()
            if self._rollout_hidden is None:
                raise RuntimeError("reset_rollout must be called before step_rollout")
            if half_index != getattr(self, "_next_expected_prefix", None):
                raise RuntimeError("paper-causal GRU prefixes must be consumed exactly once in order")
            if history.ndim != 2 or history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            if history.shape[0] != self._rollout_hidden.shape[0]:
                raise ValueError("rollout batch size changed after reset")
            latest = (
                th.zeros(
                    (history.shape[0], 1),
                    dtype=self._rollout_hidden.dtype,
                    device=self._rollout_hidden.device,
                )
                if half_index == 0
                else (1.0 - 2.0 * history[:, -1:]).to(
                    self._rollout_hidden.dtype
                )
            )
            self._rollout_hidden = self.gru(latest, self._rollout_hidden)
            self._next_expected_prefix += 1
            value = th.tanh(self.dense1(self._rollout_hidden))
            value = th.tanh(self.dense2(value))
            return self.output(value)

        def forward_with_mode(
            self, history: Any, half_index: int, *, latest_only: bool = False
        ) -> Any:
            th = _require_torch()
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            reference = next(self.parameters())
            hidden = th.zeros(
                (history.shape[0], 10),
                dtype=reference.dtype,
                device=reference.device,
            )
            sentinel = th.zeros(
                (history.shape[0], 1),
                dtype=reference.dtype,
                device=reference.device,
            )
            hidden = self.gru(sentinel, hidden)
            start = max(0, half_index - 1) if latest_only else 0
            for index in range(start, half_index):
                token = (1.0 - 2.0 * history[:, index : index + 1]).to(
                    reference.dtype
                )
                hidden = self.gru(token, hidden)
            value = th.tanh(self.dense1(hidden))
            value = th.tanh(self.dense2(value))
            return self.output(value)

        def forward(self, history: Any, half_index: int) -> Any:
            return self.forward_with_mode(history, half_index, latest_only=False)


    class PaperCausalNMFLatestOnlyView:
        def __init__(self, policy: PaperCausalNMFPolicy) -> None:
            self.policy = policy

        def __call__(self, history: Any, half_index: int) -> Any:
            return self.policy.forward_with_mode(history, half_index, latest_only=True)

else:  # pragma: no cover

    class PaperCausalMFPolicy:
        def __init__(self, **_: Any) -> None:
            _require_torch()

    class PaperCausalNMFPolicy:
        def __init__(self, **_: Any) -> None:
            _require_torch()

    class PaperCausalNMFLatestOnlyView:
        def __init__(self, *_: Any) -> None:
            _require_torch()


def _torch_dtype(name: str) -> Any:
    th = _require_torch()
    return th.float64 if name == "float64" else th.float32


def build_policy(
    strategy: Literal["mf", "nmf"],
    config: PaperCausalDirectionalRankingConfig,
    seed: int,
) -> Any:
    kwargs = {
        "device": config.device,
        "dtype": _torch_dtype(config.real_dtype),
        "seed": int(seed),
    }
    if strategy == "mf":
        return PaperCausalMFPolicy(**kwargs)
    if strategy == "nmf":
        return PaperCausalNMFPolicy(**kwargs)
    raise ValueError("strategy must be mf or nmf")


def simulator(
    config: PaperCausalDirectionalRankingConfig,
    *,
    cutoff: int,
    batch_size: int,
) -> PaperCausalSBSTrajectorySimulator:
    return PaperCausalSBSTrajectorySimulator(
        PaperCausalSBSConfig(
            cutoff=cutoff,
            full_cycles=config.full_cycles,
            batch_size=batch_size,
            projector_delta=config.projector_delta,
            cavity_lifetime_us=config.cavity_lifetime_us,
            ancilla_t1_us=config.ancilla_t1_us,
            ancilla_t2_us=config.ancilla_t2_us,
            device=config.device,
            real_dtype=config.real_dtype,
        )
    )


def applied_control_penalties_per_trajectory(
    result: Any, nominal: Any
) -> tuple[Any, Any]:
    """Return residual/slew costs for each retained trajectory."""

    th = _require_torch()
    if result.feedback_timeline != PAPER_CAUSAL_FEEDBACK_TIMELINE:
        raise ValueError("paper-causal penalties require a paper-causal result")
    layer = result.layer_applied_physical_controls - nominal[None, None, :14]
    rotation = (
        result.virtual_rotation_applied_physical_controls
        - nominal[None, None, 14:15]
    )
    residual = (th.sum(layer**2, dim=(1, 2)) + th.sum(rotation**2, dim=(1, 2))) / float(
        layer.shape[1] * layer.shape[2] + rotation.shape[1] * rotation.shape[2]
    )
    decisions = result.decision_physical_controls
    delta = decisions[:, 1:, :] - decisions[:, :-1, :]
    mask = th.ones_like(delta, dtype=th.bool)
    mask[:, 0, 14] = False  # d0.VR was never applied.
    mask[:, -1, :14] = False  # dH.layers were never applied.
    selected_per_trajectory = int(mask[0].sum().detach().cpu())
    slew = (
        th.sum(delta**2 * mask, dim=(1, 2)) / float(selected_per_trajectory)
        if selected_per_trajectory
        else th.zeros_like(residual)
    )
    return residual, slew


def applied_control_penalties(result: Any, nominal: Any) -> tuple[Any, Any]:
    """Regularize exactly the components that entered a physical operation."""

    residual, slew = applied_control_penalties_per_trajectory(result, nominal)
    return residual.mean(), slew.mean()


def evaluate_policy(
    strategy: Literal["standard", "mf", "nmf", "nmf_latest_only"],
    model: Any | None,
    config: PaperCausalDirectionalRankingConfig,
    *,
    cutoff: int,
    batch_size: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Evaluate one frozen paper-causal policy without selection leakage."""

    th = _require_torch()
    if strategy == "standard" and model is not None:
        raise ValueError("standard strategy must not provide a model")
    if strategy != "standard" and model is None:
        raise ValueError("feedback strategy requires a model")
    if strategy == "nmf_latest_only" and not isinstance(
        model, PaperCausalNMFPolicy
    ):
        raise TypeError("nmf_latest_only requires PaperCausalNMFPolicy")
    engine = simulator(config, cutoff=cutoff, batch_size=batch_size)
    if model is not None:
        model.eval()
    nominal = nominal_sbs_parameters(
        device=config.device, dtype=_torch_dtype(config.real_dtype)
    )
    per_seed: list[dict[str, Any]] = []
    with th.no_grad():
        for seed in seeds:
            policy = (
                None
                if strategy == "standard"
                else PaperCausalNMFLatestOnlyView(model)
                if strategy == "nmf_latest_only"
                else model
            )
            result = engine.run(
                control_policy=policy,
                seed=int(seed),
                record_cycle_metrics=True,
            )
            if (
                result.cycle_fidelities is None
                or result.cycle_code_survival is None
                or result.cycle_logical_z_signal is None
            ):
                raise RuntimeError("requested cycle metrics were not returned")
            fidelity_curve = (
                result.cycle_fidelities.mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            trajectory_final_fidelity = (
                result.cycle_fidelities[:, -1]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            survival_curve = (
                result.cycle_code_survival.mean(dim=0)
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            z_values = result.cycle_logical_z_signal
            z_normalized = z_values / th.clamp(
                z_values[:, :1], min=engine.config.probability_floor
            )
            logical_z_curve = (
                z_normalized.mean(dim=0).detach().cpu().numpy().astype(float)
            )
            fidelity_fit = _effective_lifetime(fidelity_curve)
            logical_z_fit = _effective_lifetime(logical_z_curve)
            residual, slew = applied_control_penalties(result, nominal)
            per_seed.append(
                {
                    "seed": int(seed),
                    "trajectory_count": int(batch_size),
                    "fidelity_curve": fidelity_curve.tolist(),
                    "trajectory_final_fidelity": trajectory_final_fidelity.tolist(),
                    "code_survival_curve": survival_curve.tolist(),
                    "logical_z_curve": logical_z_curve.tolist(),
                    "fidelity": fidelity_fit,
                    "logical_z": logical_z_fit,
                    "mean_ground_outcome_probability": float(
                        th.mean((result.outcomes == 0).to(th.float64)).cpu()
                    ),
                    "mean_control_residual_rms": float(th.sqrt(residual).cpu()),
                    "mean_control_slew_rms": float(th.sqrt(slew).cpu()),
                    "maximum_trace_error": result.maximum_trace_error,
                    "maximum_hermiticity_error": result.maximum_hermiticity_error,
                    "minimum_final_eigenvalue": result.minimum_final_eigenvalue,
                }
            )
    metrics = {
        "fidelity_effective_lifetime_cycles": [
            item["fidelity"]["effective_lifetime_cycles"] for item in per_seed
        ],
        "fidelity_normalized_auc": [
            item["fidelity"]["normalized_auc"] for item in per_seed
        ],
        "logical_z_effective_lifetime_cycles": [
            item["logical_z"]["effective_lifetime_cycles"] for item in per_seed
        ],
        "logical_z_normalized_auc": [
            item["logical_z"]["normalized_auc"] for item in per_seed
        ],
    }
    auxiliary = {
        "mean_ground_outcome_probability": [
            item["mean_ground_outcome_probability"] for item in per_seed
        ],
        "mean_control_residual_rms": [
            item["mean_control_residual_rms"] for item in per_seed
        ],
        "mean_control_slew_rms": [item["mean_control_slew_rms"] for item in per_seed],
        "fidelity_log_linear_fit_r_squared_diagnostic": [
            item["fidelity"]["log_linear_fit_r_squared_diagnostic"]
            for item in per_seed
        ],
        "logical_z_log_linear_fit_r_squared_diagnostic": [
            item["logical_z"]["log_linear_fit_r_squared_diagnostic"]
            for item in per_seed
        ],
    }
    return {
        "strategy": strategy,
        "cutoff": int(cutoff),
        "batch_size_per_seed": int(batch_size),
        "seed_count": len(tuple(seeds)),
        "total_trajectories": int(batch_size * len(tuple(seeds))),
        "simulated_physical_time_us": float(config.full_cycles * 10.0),
        "per_seed": per_seed,
        "metric_means": {
            name: float(np.mean(values)) for name, values in metrics.items()
        },
        "auxiliary_means": {
            name: float(np.mean(values)) for name, values in auxiliary.items()
        },
        "selection_score_mean": float(
            0.5
            * (
                np.mean(metrics["fidelity_normalized_auc"])
                + np.mean(metrics["logical_z_normalized_auc"])
            )
        ),
    }


__all__ = [
    "GQF_OBSERVATION_ENCODING",
    "PAPER_CAUSAL_FEEDBACK_TIMELINE",
    "PaperCausalDirectionalRankingConfig",
    "PaperCausalMFPolicy",
    "PaperCausalNMFPolicy",
    "PaperCausalSBSConfig",
    "PaperCausalSBSTrajectorySimulator",
    "PaperCausalTrajectoryResult",
    "applied_control_penalties",
    "build_policy",
    "evaluate_policy",
    "simulator",
]
