"""T2.3.7 standard/MF/NMF directional-lifetime ranking experiment.

The experiment uses the same finite-cutoff joint cavity--two-level-ancilla
simulator, Table-S1 physical timing, and high-noise lifetimes for all three
strategies.  It deliberately separates training, validation checkpoint
selection, and held-out test seeds.  The primary evaluation reports both the
physical target-state fidelity decay and a code-survival-weighted projected
logical-Z signal.  Their fitted time constants are finite-horizon *effective*
lifetimes, not an exact reproduction of the paper's 1000-cycle six-state
logical-channel lifetime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .differentiable_sbs_trajectory import (
    DIFFERENTIABLE_SBS_SCOPE,
    PARAMETER_NAMES,
    POROTTI_S1_PROFILE_ID,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
    nominal_sbs_parameters,
)
from .sbs_error_space import SBS_PROTOCOL_ID

try:  # The recovery interpreter intentionally has no torch installation.
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal-environment path.
    torch = None  # type: ignore[assignment]


StrategyName = Literal["standard", "mf", "nmf", "nmf_latest_only"]

NMF_RANKING_SCOPE = (
    "finite-cutoff joint cavity-two-level-ancilla directional ranking under "
    "Puviani high-noise Table-S1 timing; state-fidelity and projected-logical-Z "
    "10-cycle effective lifetimes; not paper-exact 1000-cycle six-state channel "
    "lifetime, optimizer optimality, multilevel leakage, SPAM, pulse, device, or "
    "hardware evidence"
)
PAPER_DOI = "10.1103/PhysRevLett.134.020601"
PAPER_ARXIV = "2312.07391"
OFFICIAL_CODE_URL = "https://github.com/Matteo-Puviani/GQF"
PAPER_RNN_ARCHITECTURE = "GRU10-DENSE256-DENSE256-OUT15"
PAPER_MF_ARCHITECTURE = "DENSE256-DENSE256-OUT15-LATEST-OUTCOME"
POLICY_INITIALIZATION = (
    "nominal-residual initialization; GRU gain 1.0, dense gain 0.5, output gain "
    "0.05, hidden biases uniform[-0.01,0.01], zero output bias"
)
ANALYSIS_CONTRACT_ID = "T237-V3-STRICT-SPLIT-AREA-EQUIVALENT-10-CYCLE-LIFETIME"
POLICY_INITIALIZATION_ID = "T237-NOMINAL-RESIDUAL-GRU-ORTHOGONAL-V2"
TRAINING_PROTOCOL_ID = "T237-FEEDBACK-GRAPE-TRAIN-ONLY-EMA-BASELINE-V1"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T2.3.7 requires PyTorch; use C:/ProgramData/anaconda3/envs/DLEnv/python.exe"
        )
    return torch


def _tuple_of_unique_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def implementation_sha256() -> str:
    """Fingerprint the two executable sources that define this experiment.

    Resume is deliberately tied to source bytes as well as configuration.  This
    prevents a checkpoint produced by an older simulator or training rule from
    being silently reused after a code change.
    """

    digest = hashlib.sha256()
    source_paths = (
        Path(__file__).resolve(),
        Path(__file__).with_name("differentiable_sbs_trajectory.py").resolve(),
    )
    for path in source_paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _training_trajectory_seed(agent_seed: int, epoch: int) -> int:
    """Return the deterministic train-only rollout seed for warm-up/epoch."""

    if epoch < 0:
        raise ValueError("training epoch must be nonnegative")
    return int(agent_seed) * 1_000_003 + int(epoch)


@dataclass(frozen=True)
class DirectionalRankingConfig:
    cutoff: int = 12
    confirmation_cutoff: int = 16
    full_cycles: int = 10
    train_epochs: int = 320
    train_batch_size: int = 8
    validation_batch_size: int = 32
    test_batch_size: int = 64
    confirmation_batch_size: int = 32
    validation_interval: int = 40
    learning_rate: float = 1.0e-4
    score_baseline_decay: float = 0.95
    gradient_clip_norm: float = 10.0
    residual_l2_weight: float = 1.0e-5
    slew_l2_weight: float = 1.0e-5
    training_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    validation_seeds: tuple[int, ...] = (10007, 10009)
    test_seeds: tuple[int, ...] = (
        20011,
        20021,
        20023,
        20029,
        20047,
        20051,
        20063,
        20071,
    )
    confirmation_seeds: tuple[int, ...] = (30011, 30013, 30029, 30047)
    bootstrap_seed: int = 73191
    bootstrap_repetitions: int = 20_000
    device: Literal["cpu", "cuda"] = "cuda"
    real_dtype: Literal["float32", "float64"] = "float64"
    projector_delta: float = 0.34
    cavity_lifetime_us: float = 245.0
    ancilla_t1_us: float = 50.0
    ancilla_t2_us: float = 60.0
    analysis_contract_id: str = ANALYSIS_CONTRACT_ID
    policy_initialization_id: str = POLICY_INITIALIZATION_ID

    def __post_init__(self) -> None:
        for name in (
            "cutoff",
            "confirmation_cutoff",
            "full_cycles",
            "train_epochs",
            "train_batch_size",
            "validation_batch_size",
            "test_batch_size",
            "confirmation_batch_size",
            "validation_interval",
            "bootstrap_repetitions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if not 4 <= self.cutoff <= 48 or not 4 <= self.confirmation_cutoff <= 48:
            raise ValueError("cutoffs must lie in [4, 48]")
        if not 1 <= self.full_cycles <= 10:
            raise ValueError("full_cycles must lie in the validated [1, 10] envelope")
        if self.validation_interval > self.train_epochs:
            raise ValueError("validation_interval must not exceed train_epochs")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "projector_delta",
            "cavity_lifetime_us",
            "ancilla_t1_us",
            "ancilla_t2_us",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.score_baseline_decay < 1.0:
            raise ValueError("score_baseline_decay must lie in [0, 1)")
        for name in ("residual_l2_weight", "slew_l2_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in (
            "training_seeds",
            "validation_seeds",
            "test_seeds",
            "confirmation_seeds",
        ):
            object.__setattr__(self, name, _tuple_of_unique_ints(getattr(self, name), name))
        all_seed_sets = [
            set(self.training_seeds),
            set(self.validation_seeds),
            set(self.test_seeds),
            set(self.confirmation_seeds),
        ]
        for index, left in enumerate(all_seed_sets):
            for right in all_seed_sets[index + 1 :]:
                if left & right:
                    raise ValueError("training/validation/test/confirmation seeds must be disjoint")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        if self.ancilla_t2_us > 2.0 * self.ancilla_t1_us + 1.0e-12:
            raise ValueError("ancilla_t2_us must not exceed 2*T1")
        if self.analysis_contract_id != ANALYSIS_CONTRACT_ID:
            raise ValueError("analysis_contract_id must preserve the frozen T2.3.7 metric")
        if self.policy_initialization_id != POLICY_INITIALIZATION_ID:
            raise ValueError("policy_initialization_id must preserve the frozen initializer")

    @property
    def train_trajectories_per_family(self) -> int:
        return self.train_epochs * self.train_batch_size * len(self.training_seeds)

    @property
    def test_trajectories_per_agent(self) -> int:
        return self.test_batch_size * len(self.test_seeds)

    @property
    def contract_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_production_design(config: DirectionalRankingConfig) -> None:
    """Reject pilot-sized runs before they can overwrite production evidence."""

    failures: list[str] = []
    if config.full_cycles != 10:
        failures.append("production ranking requires the paper's 10-cycle training horizon")
    if config.cutoff < 12 or config.confirmation_cutoff < 16:
        failures.append("production ranking requires cutoff >=12 and confirmation cutoff >=16")
    if config.train_epochs < 300:
        failures.append("production ranking requires at least 300 epochs per agent")
    if len(config.training_seeds) < 5:
        failures.append("production ranking requires at least five independently trained agents")
    if len(config.validation_seeds) < 2 or len(config.test_seeds) < 8:
        failures.append("production ranking requires >=2 validation and >=8 held-out test seeds")
    if config.test_trajectories_per_agent < 512:
        failures.append("production ranking requires >=512 held-out trajectories per agent")
    if config.real_dtype != "float64":
        failures.append("production ranking requires float64/complex128 physics")
    if failures:
        raise ValueError("; ".join(failures))


if torch is not None:

    class _PaperPolicyBase(torch.nn.Module):
        strategy: str

        @staticmethod
        def _initialize_linear(layer: Any, *, gain: float, generator: Any) -> None:
            th = _require_torch()
            th.nn.init.xavier_uniform_(layer.weight, gain=gain, generator=generator)
            th.nn.init.uniform_(layer.bias, -0.01, 0.01, generator=generator)

        @property
        def parameter_count(self) -> int:
            return int(sum(parameter.numel() for parameter in self.parameters()))


    class PaperScaleMFPolicy(_PaperPolicyBase):
        """Paper Table-S3 latest-outcome-only memoryless feedback policy."""

        strategy = "mf"

        def __init__(self, *, device: str, dtype: Any, seed: int) -> None:
            super().__init__()
            th = _require_torch()
            self.dense1 = th.nn.Linear(1, 256)
            self.dense2 = th.nn.Linear(256, 256)
            self.output = th.nn.Linear(256, len(PARAMETER_NAMES))
            generator = th.Generator(device="cpu")
            generator.manual_seed(int(seed))
            self._initialize_linear(self.dense1, gain=0.50, generator=generator)
            self._initialize_linear(self.dense2, gain=0.50, generator=generator)
            self._initialize_linear(self.output, gain=0.05, generator=generator)
            th.nn.init.zeros_(self.output.bias)
            self.to(device=device, dtype=dtype)

        def forward(self, history: Any, half_index: int) -> Any:
            th = _require_torch()
            if not isinstance(history, th.Tensor) or history.ndim != 2:
                raise TypeError("history must be a rank-two torch.Tensor")
            if history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            if half_index == 0:
                latest = th.zeros(
                    (history.shape[0], 1),
                    dtype=next(self.parameters()).dtype,
                    device=next(self.parameters()).device,
                )
            else:
                latest = (2.0 * history[:, -1:] - 1.0).to(next(self.parameters()).dtype)
            value = th.tanh(self.dense1(latest))
            value = th.tanh(self.dense2(value))
            return self.output(value)


    class PaperScaleNMFPolicy(_PaperPolicyBase):
        """Paper Table-S2 GRU10--256--256--15 full-history policy."""

        strategy = "nmf"

        def __init__(self, *, device: str, dtype: Any, seed: int) -> None:
            super().__init__()
            th = _require_torch()
            self.gru = th.nn.GRUCell(1, 10)
            self.dense1 = th.nn.Linear(10, 256)
            self.dense2 = th.nn.Linear(256, 256)
            self.output = th.nn.Linear(256, len(PARAMETER_NAMES))
            generator = th.Generator(device="cpu")
            generator.manual_seed(int(seed))
            th.nn.init.xavier_uniform_(self.gru.weight_ih, gain=1.00, generator=generator)
            th.nn.init.orthogonal_(self.gru.weight_hh, gain=1.00, generator=generator)
            th.nn.init.uniform_(self.gru.bias_ih, -0.01, 0.01, generator=generator)
            th.nn.init.uniform_(self.gru.bias_hh, -0.01, 0.01, generator=generator)
            self._initialize_linear(self.dense1, gain=0.50, generator=generator)
            self._initialize_linear(self.dense2, gain=0.50, generator=generator)
            self._initialize_linear(self.output, gain=0.05, generator=generator)
            th.nn.init.zeros_(self.output.bias)
            self.to(device=device, dtype=dtype)
            self._rollout_hidden: Any | None = None

        def reset_rollout(self, *, batch_size: int, device: Any, dtype: Any) -> None:
            """Reset the cached hidden state for one simulator rollout."""

            th = _require_torch()
            reference = next(self.parameters())
            requested = th.device(device)
            reference_index = 0 if reference.device.type == "cuda" and reference.device.index is None else reference.device.index
            requested_index = 0 if requested.type == "cuda" and requested.index is None else requested.index
            if (
                reference.device.type != requested.type
                or reference_index != requested_index
                or reference.dtype != dtype
            ):
                raise ValueError("policy and simulator device/dtype must match")
            self._rollout_hidden = th.zeros(
                (int(batch_size), 10), dtype=reference.dtype, device=reference.device
            )

        def step_rollout(self, history: Any, half_index: int) -> Any:
            """O(H) recurrent rollout equivalent to replaying the complete history."""

            th = _require_torch()
            if self._rollout_hidden is None:
                raise RuntimeError("reset_rollout must be called before step_rollout")
            if history.ndim != 2 or history.shape[1] != half_index:
                raise ValueError("history width must equal half_index")
            if history.shape[0] != self._rollout_hidden.shape[0]:
                raise ValueError("rollout batch size changed after reset")
            if half_index > 0:
                latest = (2.0 * history[:, -1:] - 1.0).to(self._rollout_hidden.dtype)
                self._rollout_hidden = self.gru(latest, self._rollout_hidden)
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
                (history.shape[0], 10), dtype=reference.dtype, device=reference.device
            )
            start = max(0, half_index - 1) if latest_only else 0
            for index in range(start, half_index):
                value = (2.0 * history[:, index : index + 1] - 1.0).to(reference.dtype)
                hidden = self.gru(value, hidden)
            value = th.tanh(self.dense1(hidden))
            value = th.tanh(self.dense2(value))
            return self.output(value)

        def forward(self, history: Any, half_index: int) -> Any:
            return self.forward_with_mode(history, half_index, latest_only=False)


    class NMFLatestOnlyView:
        """Read-only hidden-state-reset ablation of a trained NMF policy."""

        def __init__(self, policy: PaperScaleNMFPolicy) -> None:
            self.policy = policy

        def __call__(self, history: Any, half_index: int) -> Any:
            return self.policy.forward_with_mode(history, half_index, latest_only=True)

else:

    class PaperScaleMFPolicy:  # pragma: no cover - recovery interpreter.
        def __init__(self, **_: Any) -> None:
            _require_torch()

    class PaperScaleNMFPolicy:  # pragma: no cover - recovery interpreter.
        def __init__(self, **_: Any) -> None:
            _require_torch()

    class NMFLatestOnlyView:  # pragma: no cover - recovery interpreter.
        def __init__(self, *_: Any) -> None:
            _require_torch()


def _torch_dtype(name: str) -> Any:
    th = _require_torch()
    return th.float64 if name == "float64" else th.float32


def build_policy(strategy: Literal["mf", "nmf"], config: DirectionalRankingConfig, seed: int) -> Any:
    dtype = _torch_dtype(config.real_dtype)
    if strategy == "mf":
        return PaperScaleMFPolicy(device=config.device, dtype=dtype, seed=seed)
    if strategy == "nmf":
        return PaperScaleNMFPolicy(device=config.device, dtype=dtype, seed=seed)
    raise ValueError("strategy must be mf or nmf")


def _simulator(
    config: DirectionalRankingConfig, *, cutoff: int, batch_size: int
) -> DifferentiableSBSTrajectorySimulator:
    return DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(
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


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        array = state_dict[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _area_equivalent_lifetime(normalized_auc: float, horizon: float) -> float:
    """Invert ``A=(T/H)*(1-exp(-H/T))`` for a finite-horizon curve area."""

    area = float(normalized_auc)
    duration = float(horizon)
    if not np.isfinite(area) or not 0.0 < area <= 1.0 + 1.0e-9:
        raise ValueError("normalized AUC must lie in (0, 1] up to numerical tolerance")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if area >= 1.0 - 1.0e-12:
        return 1.0e12 * duration
    lower = 1.0e-12 * duration
    upper = 1.0e6 * duration
    for _ in range(160):
        middle = 0.5 * (lower + upper)
        predicted = (middle / duration) * (1.0 - np.exp(-duration / middle))
        if predicted < area:
            lower = middle
        else:
            upper = middle
    return float(0.5 * (lower + upper))


def _effective_lifetime(curve: Sequence[float]) -> dict[str, Any]:
    """Return an area-equivalent lifetime plus a non-authoritative fit audit.

    Ten-cycle trajectories can contain correction transients and need not be a
    single exponential.  The primary lifetime is therefore the unique
    exponential whose *finite-horizon area* equals the observed normalized
    curve area.  A fixed-origin log-linear fit is retained only to expose when
    direct extrapolation would be unjustified.
    """

    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("lifetime curve must be a finite rank-one array with >=3 points")
    if abs(values[0]) < 1.0e-12:
        raise ValueError("lifetime curve has a numerically zero initial value")
    normalized = values / values[0]
    clipped = np.clip(normalized, 1.0e-8, 1.0)
    time_axis = np.arange(values.size, dtype=np.float64)
    denominator = float(np.dot(time_axis[1:], time_axis[1:]))
    decay_rate = -float(np.dot(time_axis[1:], np.log(clipped[1:]))) / denominator
    log_linear_lifetime = float(1.0 / max(decay_rate, 1.0e-12))
    fitted = np.exp(-time_axis / log_linear_lifetime)
    residual = float(np.sum((normalized - fitted) ** 2))
    centered = float(np.sum((normalized - np.mean(normalized)) ** 2))
    r_squared = float(1.0 - residual / centered) if centered > 1.0e-15 else 1.0
    auc = float(np.trapezoid(normalized, time_axis) / time_axis[-1])
    area_lifetime = _area_equivalent_lifetime(auc, float(time_axis[-1]))
    return {
        "effective_lifetime_cycles": area_lifetime,
        "effective_lifetime_method": "finite_horizon_area_equivalent_exponential",
        "normalized_auc": auc,
        "log_linear_lifetime_cycles_diagnostic": log_linear_lifetime,
        "log_linear_fit_r_squared_diagnostic": r_squared,
        "exponential_fit_is_diagnostic_only": True,
        "initial_value": float(values[0]),
        "final_value": float(values[-1]),
    }


def evaluate_policy(
    strategy: StrategyName,
    model: Any | None,
    config: DirectionalRankingConfig,
    *,
    cutoff: int,
    batch_size: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Evaluate one frozen strategy without gradient or checkpoint selection."""

    th = _require_torch()
    if strategy == "standard" and model is not None:
        raise ValueError("standard strategy must not provide a model")
    if strategy != "standard" and model is None:
        raise ValueError("feedback strategy requires a model")
    if strategy == "nmf_latest_only" and not isinstance(model, PaperScaleNMFPolicy):
        raise TypeError("nmf_latest_only requires a PaperScaleNMFPolicy")
    simulator = _simulator(config, cutoff=cutoff, batch_size=batch_size)
    if model is not None:
        model.eval()
    per_seed: list[dict[str, Any]] = []
    nominal = nominal_sbs_parameters(
        device=config.device, dtype=_torch_dtype(config.real_dtype)
    )
    with th.no_grad():
        for seed in seeds:
            policy = (
                None
                if strategy == "standard"
                else NMFLatestOnlyView(model)
                if strategy == "nmf_latest_only"
                else model
            )
            result = simulator.run(
                control_policy=policy,
                seed=int(seed),
                record_cycle_metrics=True,
            )
            if (
                result.cycle_fidelities is None
                or result.cycle_code_survival is None
                or result.cycle_logical_z_signal is None
            ):
                raise RuntimeError("cycle metrics were requested but not returned")
            fidelity_curve = (
                result.cycle_fidelities.mean(dim=0).detach().cpu().numpy().astype(float)
            )
            survival_curve = (
                result.cycle_code_survival.mean(dim=0).detach().cpu().numpy().astype(float)
            )
            z_values = result.cycle_logical_z_signal
            z_normalized = z_values / th.clamp(
                z_values[:, :1], min=simulator.config.probability_floor
            )
            logical_z_curve = z_normalized.mean(dim=0).detach().cpu().numpy().astype(float)
            fidelity_fit = _effective_lifetime(fidelity_curve)
            logical_z_fit = _effective_lifetime(logical_z_curve)
            residual = result.physical_controls - nominal[None, None, :]
            slew = result.physical_controls[:, 1:, :] - result.physical_controls[:, :-1, :]
            per_seed.append(
                {
                    "seed": int(seed),
                    "trajectory_count": int(batch_size),
                    "fidelity_curve": fidelity_curve.tolist(),
                    "code_survival_curve": survival_curve.tolist(),
                    "logical_z_curve": logical_z_curve.tolist(),
                    "fidelity": fidelity_fit,
                    "logical_z": logical_z_fit,
                    "mean_ground_outcome_probability": float(
                        th.mean((result.outcomes == 0).to(th.float64)).cpu()
                    ),
                    "mean_control_residual_rms": float(th.sqrt(th.mean(residual**2)).cpu()),
                    "mean_control_slew_rms": float(th.sqrt(th.mean(slew**2)).cpu()),
                    "maximum_trace_error": result.maximum_trace_error,
                    "maximum_hermiticity_error": result.maximum_hermiticity_error,
                    "minimum_final_eigenvalue": result.minimum_final_eigenvalue,
                }
            )
    metrics = {
        "fidelity_effective_lifetime_cycles": [
            item["fidelity"]["effective_lifetime_cycles"] for item in per_seed
        ],
        "fidelity_normalized_auc": [item["fidelity"]["normalized_auc"] for item in per_seed],
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
        "mean_control_residual_rms": [item["mean_control_residual_rms"] for item in per_seed],
        "mean_control_slew_rms": [item["mean_control_slew_rms"] for item in per_seed],
        "fidelity_log_linear_fit_r_squared_diagnostic": [
            item["fidelity"]["log_linear_fit_r_squared_diagnostic"] for item in per_seed
        ],
        "logical_z_log_linear_fit_r_squared_diagnostic": [
            item["logical_z"]["log_linear_fit_r_squared_diagnostic"] for item in per_seed
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
        "metric_means": {name: float(np.mean(values)) for name, values in metrics.items()},
        "auxiliary_means": {name: float(np.mean(values)) for name, values in auxiliary.items()},
        "selection_score_mean": float(
            0.5
            * (
                np.mean(metrics["fidelity_normalized_auc"])
                + np.mean(metrics["logical_z_normalized_auc"])
            )
        ),
    }


def train_agent(
    strategy: Literal["mf", "nmf"],
    seed: int,
    config: DirectionalRankingConfig,
) -> tuple[Any, dict[str, Any]]:
    """Train one agent and select its checkpoint on validation seeds only."""

    th = _require_torch()
    model = build_policy(strategy, config, seed)
    model.train()
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    simulator = _simulator(
        config, cutoff=config.cutoff, batch_size=config.train_batch_size
    )
    nominal = nominal_sbs_parameters(
        device=config.device, dtype=_torch_dtype(config.real_dtype)
    )
    validation_history: list[dict[str, Any]] = []
    initial_validation = evaluate_policy(
        strategy,
        model,
        config,
        cutoff=config.cutoff,
        batch_size=config.validation_batch_size,
        seeds=config.validation_seeds,
    )
    best_score = float(initial_validation["selection_score_mean"])
    best_epoch = 0
    best_state = _state_dict_cpu(model)
    validation_history.append(
        {"epoch": 0, "selection_score": best_score, "metric_means": initial_validation["metric_means"]}
    )
    print(
        json.dumps(
            {
                "event": "validation",
                "strategy": strategy,
                "training_seed": int(seed),
                "epoch": 0,
                "selection_score": best_score,
            }
        ),
        flush=True,
    )
    # The score-function baseline must not consume validation/test outcomes.
    # A no-gradient warm-up rollout from the train-only seed family supplies
    # the initial scalar EMA baseline without changing model parameters.
    warmup_seed = _training_trajectory_seed(seed, 0)
    with th.no_grad():
        warmup = simulator.run(
            control_policy=model,
            seed=warmup_seed,
            record_cycle_metrics=False,
        )
    baseline = float(warmup.reward.detach().mean().cpu())
    training_curve: list[dict[str, Any]] = []
    start = time.perf_counter()
    for epoch in range(1, config.train_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        result = simulator.run(
            control_policy=model,
            seed=_training_trajectory_seed(seed, epoch),
            record_cycle_metrics=False,
        )
        advantage = result.reward.detach() - baseline
        reward_path = result.reward.mean()
        score_path = th.mean(advantage * result.log_probability)
        residual = result.physical_controls - nominal[None, None, :]
        slew = result.physical_controls[:, 1:, :] - result.physical_controls[:, :-1, :]
        residual_penalty = th.mean(residual**2)
        slew_penalty = th.mean(slew**2)
        objective = reward_path + score_path
        loss = (
            -objective
            + config.residual_l2_weight * residual_penalty
            + config.slew_l2_weight * slew_penalty
        )
        if not bool(th.isfinite(loss).detach().cpu()):
            raise RuntimeError(f"non-finite training loss for {strategy} seed {seed} epoch {epoch}")
        loss.backward()
        gradient_norm = float(
            th.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm).detach().cpu()
        )
        if not np.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError(f"invalid gradient norm for {strategy} seed {seed} epoch {epoch}")
        optimizer.step()
        reward_mean = float(result.reward.detach().mean().cpu())
        baseline = (
            config.score_baseline_decay * baseline
            + (1.0 - config.score_baseline_decay) * reward_mean
        )
        training_curve.append(
            {
                "epoch": epoch,
                "mean_reward": reward_mean,
                "score_path": float(score_path.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "gradient_norm_before_clip": gradient_norm,
                "baseline": baseline,
                "ground_outcome_fraction": float(
                    th.mean((result.outcomes == 0).to(th.float64)).detach().cpu()
                ),
                "control_residual_rms": float(th.sqrt(residual_penalty).detach().cpu()),
                "control_slew_rms": float(th.sqrt(slew_penalty).detach().cpu()),
            }
        )
        if epoch % config.validation_interval == 0 or epoch == config.train_epochs:
            validation = evaluate_policy(
                strategy,
                model,
                config,
                cutoff=config.cutoff,
                batch_size=config.validation_batch_size,
                seeds=config.validation_seeds,
            )
            score = float(validation["selection_score_mean"])
            validation_history.append(
                {"epoch": epoch, "selection_score": score, "metric_means": validation["metric_means"]}
            )
            print(
                json.dumps(
                    {
                        "event": "validation",
                        "strategy": strategy,
                        "training_seed": int(seed),
                        "epoch": epoch,
                        "selection_score": score,
                    }
                ),
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = _state_dict_cpu(model)
    model.load_state_dict(best_state)
    model.eval()
    record = {
        "strategy": strategy,
        "training_seed": int(seed),
        "parameter_count": int(model.parameter_count),
        "architecture": (
            PAPER_MF_ARCHITECTURE if strategy == "mf" else PAPER_RNN_ARCHITECTURE
        ),
        "epochs_executed": config.train_epochs,
        "training_trajectories": config.train_epochs * config.train_batch_size,
        "baseline_warmup_trajectories": config.train_batch_size,
        "baseline_initialization": "train-only no-gradient warm-up reward mean",
        "baseline_warmup_seed": warmup_seed,
        "training_trajectory_seeds": [
            _training_trajectory_seed(seed, epoch)
            for epoch in range(1, config.train_epochs + 1)
        ],
        "validation_seeds_used_for_checkpoint_selection_only": list(
            config.validation_seeds
        ),
        "best_validation_epoch": best_epoch,
        "best_validation_score": best_score,
        "initial_validation_score": float(initial_validation["selection_score_mean"]),
        "validation_history": validation_history,
        "training_curve": training_curve,
        "checkpoint_sha256": state_dict_sha256(best_state),
        "wall_time_seconds": time.perf_counter() - start,
    }
    return model, record


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be a nonempty finite vector")
    return {
        "values": array.tolist(),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "worst_quartile_mean": float(np.mean(np.sort(array)[: max(1, array.size // 4)])),
    }


def _agent_metric(evaluation: Mapping[str, Any], metric: str) -> float:
    return float(evaluation["metric_means"][metric])


def _bootstrap_agent_difference(
    left: Sequence[float], right: Sequence[float], *, seed: int, repetitions: int
) -> dict[str, float]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("paired agent arrays must have identical rank-one shape")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, left_array.size, size=(repetitions, left_array.size))
    samples = np.mean(left_array[indices] - right_array[indices], axis=1)
    return {
        "mean_difference": float(np.mean(left_array - right_array)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(samples > 0.0)),
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    th = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    th.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_directional_ranking(
    config: DirectionalRankingConfig,
    *,
    artifact_path: str | Path,
    checkpoint_path: str | Path,
    production: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Train all agents, evaluate held-out seeds, and persist auditable evidence."""

    th = _require_torch()
    if production:
        validate_production_design(config)
    if config.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA production run requested but torch.cuda.is_available() is false")
    start = time.perf_counter()
    implementation_hash = implementation_sha256()
    checkpoint_contract_hash = hashlib.sha256(
        f"{config.contract_hash}:{implementation_hash}:{TRAINING_PROTOCOL_ID}".encode(
            "utf-8"
        )
    ).hexdigest()
    trained_models: dict[str, list[Any]] = {"mf": [], "nmf": []}
    training_records: dict[str, list[dict[str, Any]]] = {"mf": [], "nmf": []}
    checkpoint_payload: dict[str, Any] = {
        "schema_version": 3,
        "config_contract_hash": config.contract_hash,
        "checkpoint_contract_hash": checkpoint_contract_hash,
        "implementation_sha256": implementation_hash,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "config": asdict(config),
        "models": {"mf": [], "nmf": []},
        "training_records": {"mf": [], "nmf": []},
    }
    checkpoint = Path(checkpoint_path)
    resumed_agents = 0
    if resume and checkpoint.exists():
        loaded = th.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict) or loaded.get("schema_version") != 3:
            raise ValueError("checkpoint exists but is not a resumable T2.3.7 schema-v3 file")
        if loaded.get("checkpoint_contract_hash") != checkpoint_contract_hash:
            raise ValueError(
                "checkpoint contract hash differs from the requested config/source/training protocol"
            )
        checkpoint_payload = loaded
    for strategy in ("mf", "nmf"):
        existing_models = {
            int(item["training_seed"]): item
            for item in checkpoint_payload["models"][strategy]
        }
        existing_records = {
            int(item["training_seed"]): item
            for item in checkpoint_payload["training_records"][strategy]
        }
        for seed in config.training_seeds:
            if seed in existing_models and seed in existing_records:
                model = build_policy(strategy, config, seed)
                model.load_state_dict(existing_models[seed]["state_dict"])
                model.eval()
                record = existing_records[seed]
                if state_dict_sha256(_state_dict_cpu(model)) != record["checkpoint_sha256"]:
                    raise ValueError("resumed model hash does not match its training record")
                resumed_agents += 1
            else:
                model, record = train_agent(strategy, seed, config)
                checkpoint_payload["models"][strategy].append(
                    {
                        "training_seed": int(seed),
                        "checkpoint_sha256": record["checkpoint_sha256"],
                        "state_dict": _state_dict_cpu(model),
                    }
                )
                checkpoint_payload["training_records"][strategy].append(record)
                _atomic_torch_save(checkpoint_payload, checkpoint)
                print(
                    json.dumps(
                        {
                            "event": "agent_complete",
                            "strategy": strategy,
                            "training_seed": int(seed),
                            "best_validation_epoch": record["best_validation_epoch"],
                            "best_validation_score": record["best_validation_score"],
                            "checkpoint_sha256": record["checkpoint_sha256"],
                        }
                    ),
                    flush=True,
                )
            trained_models[strategy].append(model)
            training_records[strategy].append(record)

    primary: dict[str, Any] = {
        "standard": evaluate_policy(
            "standard",
            None,
            config,
            cutoff=config.cutoff,
            batch_size=config.test_batch_size,
            seeds=config.test_seeds,
        ),
        "mf": [],
        "nmf": [],
        "nmf_latest_only": [],
    }
    confirmation: dict[str, Any] = {
        "standard": evaluate_policy(
            "standard",
            None,
            config,
            cutoff=config.confirmation_cutoff,
            batch_size=config.confirmation_batch_size,
            seeds=config.confirmation_seeds,
        ),
        "mf": [],
        "nmf": [],
        "nmf_latest_only": [],
    }
    for strategy in ("mf", "nmf"):
        for model in trained_models[strategy]:
            primary[strategy].append(
                evaluate_policy(
                    strategy,
                    model,
                    config,
                    cutoff=config.cutoff,
                    batch_size=config.test_batch_size,
                    seeds=config.test_seeds,
                )
            )
            confirmation[strategy].append(
                evaluate_policy(
                    strategy,
                    model,
                    config,
                    cutoff=config.confirmation_cutoff,
                    batch_size=config.confirmation_batch_size,
                    seeds=config.confirmation_seeds,
                )
            )
            if strategy == "nmf":
                primary["nmf_latest_only"].append(
                    evaluate_policy(
                        "nmf_latest_only",
                        model,
                        config,
                        cutoff=config.cutoff,
                        batch_size=config.test_batch_size,
                        seeds=config.test_seeds,
                    )
                )
                confirmation["nmf_latest_only"].append(
                    evaluate_policy(
                        "nmf_latest_only",
                        model,
                        config,
                        cutoff=config.confirmation_cutoff,
                        batch_size=config.confirmation_batch_size,
                        seeds=config.confirmation_seeds,
                    )
                )

    metric_names = (
        "fidelity_effective_lifetime_cycles",
        "fidelity_normalized_auc",
        "logical_z_effective_lifetime_cycles",
        "logical_z_normalized_auc",
    )
    summary: dict[str, Any] = {"primary": {}, "confirmation": {}}
    for lane_name, lane in (("primary", primary), ("confirmation", confirmation)):
        for strategy in ("standard", "mf", "nmf", "nmf_latest_only"):
            evaluations = [lane[strategy]] if strategy == "standard" else lane[strategy]
            summary[lane_name][strategy] = {
                metric: _distribution([_agent_metric(item, metric) for item in evaluations])
                for metric in metric_names
            }
            summary[lane_name][strategy]["auxiliary"] = {
                name: _distribution(
                    [float(item["auxiliary_means"][name]) for item in evaluations]
                )
                for name in (
                    "mean_ground_outcome_probability",
                    "mean_control_residual_rms",
                    "mean_control_slew_rms",
                    "fidelity_log_linear_fit_r_squared_diagnostic",
                    "logical_z_log_linear_fit_r_squared_diagnostic",
                )
            }

    standard_lifetime = summary["primary"]["standard"][
        "logical_z_effective_lifetime_cycles"
    ]["mean"]
    mf_lifetimes = summary["primary"]["mf"]["logical_z_effective_lifetime_cycles"][
        "values"
    ]
    nmf_lifetimes = summary["primary"]["nmf"]["logical_z_effective_lifetime_cycles"][
        "values"
    ]
    ablated_lifetimes = summary["primary"]["nmf_latest_only"][
        "logical_z_effective_lifetime_cycles"
    ]["values"]
    nmf_auc = summary["primary"]["nmf"]["logical_z_normalized_auc"]["values"]
    mf_auc = summary["primary"]["mf"]["logical_z_normalized_auc"]["values"]
    ablated_auc = summary["primary"]["nmf_latest_only"]["logical_z_normalized_auc"][
        "values"
    ]
    confirmation_standard = summary["confirmation"]["standard"][
        "logical_z_effective_lifetime_cycles"
    ]["mean"]
    confirmation_mf = summary["confirmation"]["mf"][
        "logical_z_effective_lifetime_cycles"
    ]["median"]
    confirmation_nmf = summary["confirmation"]["nmf"][
        "logical_z_effective_lifetime_cycles"
    ]["median"]
    paired_nmf_mf = _bootstrap_agent_difference(
        nmf_lifetimes,
        mf_lifetimes,
        seed=config.bootstrap_seed,
        repetitions=config.bootstrap_repetitions,
    )
    paired_nmf_ablation = _bootstrap_agent_difference(
        nmf_lifetimes,
        ablated_lifetimes,
        seed=config.bootstrap_seed + 1,
        repetitions=config.bootstrap_repetitions,
    )
    configured_evaluation_seeds = (
        set(config.validation_seeds)
        | set(config.test_seeds)
        | set(config.confirmation_seeds)
    )
    actual_training_rollout_seeds = {
        int(seed_value)
        for strategy in ("mf", "nmf")
        for record in training_records[strategy]
        for seed_value in (
            [record["baseline_warmup_seed"]]
            + list(record["training_trajectory_seeds"])
        )
    }
    checkpoint_selection_audits: list[bool] = []
    for strategy in ("mf", "nmf"):
        for record in training_records[strategy]:
            history = record["validation_history"]
            best_entry = max(history, key=lambda item: float(item["selection_score"]))
            checkpoint_selection_audits.append(
                record["baseline_initialization"]
                == "train-only no-gradient warm-up reward mean"
                and list(record["validation_seeds_used_for_checkpoint_selection_only"])
                == list(config.validation_seeds)
                and int(record["best_validation_epoch"]) == int(best_entry["epoch"])
                and np.isclose(
                    float(record["best_validation_score"]),
                    float(best_entry["selection_score"]),
                    rtol=0.0,
                    atol=1.0e-14,
                )
            )
    expected_agent_seeds = set(config.training_seeds)
    checkpoint_models_complete = all(
        {
            int(item["training_seed"])
            for item in checkpoint_payload["models"][strategy]
        }
        == expected_agent_seeds
        and {
            int(item["training_seed"])
            for item in checkpoint_payload["training_records"][strategy]
        }
        == expected_agent_seeds
        for strategy in ("mf", "nmf")
    )
    checkpoint_model_hashes_valid = all(
        state_dict_sha256(item["state_dict"]) == item["checkpoint_sha256"]
        for strategy in ("mf", "nmf")
        for item in checkpoint_payload["models"][strategy]
    )
    all_evaluations = {
        "primary": primary,
        "confirmation": confirmation,
    }
    gates = {
        "all_seed_sets_are_disjoint": not (
            set(config.training_seeds)
            & (set(config.validation_seeds) | set(config.test_seeds) | set(config.confirmation_seeds))
            or set(config.validation_seeds)
            & (set(config.test_seeds) | set(config.confirmation_seeds))
            or set(config.test_seeds) & set(config.confirmation_seeds)
        )
        and not (actual_training_rollout_seeds & configured_evaluation_seeds),
        "paper_scale_architectures_used": (
            all(item["parameter_count"] == 70_159 for item in training_records["mf"])
            and all(item["parameter_count"] == 72_853 for item in training_records["nmf"])
        ),
        "checkpoint_selection_uses_validation_only": all(
            checkpoint_selection_audits
        ),
        "primary_logical_z_lifetime_ranks_nmf_above_mf_and_standard": (
            float(np.median(nmf_lifetimes)) > max(float(np.median(mf_lifetimes)), standard_lifetime)
        ),
        "primary_logical_z_auc_ranks_nmf_above_mf_and_standard": (
            float(np.median(nmf_auc))
            > max(
                float(np.median(mf_auc)),
                summary["primary"]["standard"]["logical_z_normalized_auc"]["mean"],
            )
        ),
        "primary_fidelity_lifetime_has_same_nmf_direction": (
            summary["primary"]["nmf"]["fidelity_effective_lifetime_cycles"]["median"]
            > max(
                summary["primary"]["mf"]["fidelity_effective_lifetime_cycles"]["median"],
                summary["primary"]["standard"]["fidelity_effective_lifetime_cycles"]["mean"],
            )
        ),
        "at_least_80_percent_of_nmf_agents_beat_both_comparators": (
            int(
                np.sum(
                    (np.asarray(nmf_lifetimes) > np.asarray(mf_lifetimes))
                    & (np.asarray(nmf_lifetimes) > standard_lifetime)
                )
            )
            >= max(1, int(np.ceil(0.8 * len(nmf_lifetimes))))
        ),
        "history_reset_reduces_nmf_lifetime_and_auc": (
            float(np.median(nmf_lifetimes)) > float(np.median(ablated_lifetimes))
            and float(np.median(nmf_auc)) > float(np.median(ablated_auc))
        ),
        "cutoff16_confirmation_preserves_direction": (
            confirmation_nmf > max(confirmation_mf, confirmation_standard)
        ),
        "nmf_ground_outcome_probability_stays_above_0_75": (
            summary["primary"]["nmf"]["auxiliary"][
                "mean_ground_outcome_probability"
            ]["minimum"]
            >= 0.75
            and summary["confirmation"]["nmf"]["auxiliary"][
                "mean_ground_outcome_probability"
            ]["minimum"]
            >= 0.75
        ),
        "bounded_control_residual_and_slew_are_finite": all(
            np.isfinite(
                summary[lane][strategy]["auxiliary"][metric]["maximum"]
            )
            for lane in ("primary", "confirmation")
            for strategy in ("mf", "nmf", "nmf_latest_only")
            for metric in ("mean_control_residual_rms", "mean_control_slew_rms")
        )
        and all(
            summary[lane][strategy]["auxiliary"]["mean_control_residual_rms"][
                "maximum"
            ]
            <= 2.0
            and summary[lane][strategy]["auxiliary"]["mean_control_slew_rms"][
                "maximum"
            ]
            <= 4.0
            for lane in ("primary", "confirmation")
            for strategy in ("mf", "nmf", "nmf_latest_only")
        ),
        "nonexponential_fit_is_diagnostic_only": all(
            bool(item[metric]["exponential_fit_is_diagnostic_only"])
            and item[metric]["effective_lifetime_method"]
            == "finite_horizon_area_equivalent_exponential"
            for lane in ("primary", "confirmation")
            for strategy in ("standard", "mf", "nmf", "nmf_latest_only")
            for evaluation in (
                [all_evaluations[lane][strategy]]
                if strategy == "standard"
                else all_evaluations[lane][strategy]
            )
            for item in evaluation["per_seed"]
            for metric in ("fidelity", "logical_z")
        ),
        "finite_density_diagnostics_pass": all(
            item["maximum_trace_error"] <= 2.0e-9
            and item["maximum_hermiticity_error"] <= 2.0e-9
            and item["minimum_final_eigenvalue"] >= -2.0e-8
            for lane in ("primary", "confirmation")
            for strategy in ("standard", "mf", "nmf", "nmf_latest_only")
            for evaluation in (
                [all_evaluations[lane][strategy]]
                if strategy == "standard"
                else all_evaluations[lane][strategy]
            )
            for item in evaluation["per_seed"]
        ),
        "checkpoint_contains_all_models_with_matching_hashes": (
            checkpoint_models_complete and checkpoint_model_hashes_valid
        ),
    }
    required_directional_gates = (
        "all_seed_sets_are_disjoint",
        "paper_scale_architectures_used",
        "checkpoint_selection_uses_validation_only",
        "primary_logical_z_lifetime_ranks_nmf_above_mf_and_standard",
        "primary_logical_z_auc_ranks_nmf_above_mf_and_standard",
        "primary_fidelity_lifetime_has_same_nmf_direction",
        "at_least_80_percent_of_nmf_agents_beat_both_comparators",
        "history_reset_reduces_nmf_lifetime_and_auc",
        "cutoff16_confirmation_preserves_direction",
        "nmf_ground_outcome_probability_stays_above_0_75",
        "bounded_control_residual_and_slew_are_finite",
        "nonexponential_fit_is_diagnostic_only",
        "finite_density_diagnostics_pass",
        "checkpoint_contains_all_models_with_matching_hashes",
    )
    status = "PASS" if all(gates[name] for name in required_directional_gates) else "FAIL"
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    artifact: dict[str, Any] = {
        "task_id": "T2.3.7",
        "status": status,
        "scope": NMF_RANKING_SCOPE,
        "protocol_id": SBS_PROTOCOL_ID,
        "simulator_scope": DIFFERENTIABLE_SBS_SCOPE,
        "timing_profile_id": POROTTI_S1_PROFILE_ID,
        "literature": {
            "doi": PAPER_DOI,
            "arxiv": PAPER_ARXIV,
            "official_code": OFFICIAL_CODE_URL,
            "zotero_item_key": "IJ5EDGZF",
            "bibtex_key": "puviani_non-markovian_2025",
        },
        "policy_initialization": POLICY_INITIALIZATION,
        "analysis_contract_id": ANALYSIS_CONTRACT_ID,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "implementation_sha256": implementation_hash,
        "config": asdict(config),
        "contract_hash": config.contract_hash,
        "checkpoint_contract_hash": checkpoint_contract_hash,
        "production_design": bool(production),
        "execution": {
            "resume_enabled": bool(resume),
            "resumed_agents": resumed_agents,
            "newly_trained_agents": 2 * len(config.training_seeds) - resumed_agents,
        },
        "training_records": training_records,
        "evaluation": {"primary": primary, "confirmation": confirmation},
        "summary": summary,
        "paired_bootstrap": {
            "nmf_minus_mf_logical_z_lifetime": paired_nmf_mf,
            "nmf_minus_latest_only_ablation_logical_z_lifetime": paired_nmf_ablation,
        },
        "gates": gates,
        "required_directional_gates": list(required_directional_gates),
        "checkpoint": {
            "path": str(checkpoint).replace("\\", "/"),
            "sha256": checkpoint_sha256,
            "contains_all_training_seed_models": checkpoint_models_complete,
            "all_model_hashes_match": checkpoint_model_hashes_valid,
            "schema_version": 3,
        },
        "claim_boundary": {
            "allowed": (
                "directional 10-cycle state-fidelity and projected-logical-Z finite-horizon "
                "area-equivalent lifetime ranking within the finite-cutoff two-level "
                "high-noise simulator"
            ),
            "forbidden": (
                "paper-exact 1000-cycle six-state channel lifetime, optimizer optimality, "
                "experimental/device/FPGA lifetime, or leakage/SPAM robustness"
            ),
            "failure_branch": (
                "if status is FAIL, NMF remains a literature baseline and teacher-distillation "
                "claims are disabled"
            ),
        },
        "wall_time_seconds": time.perf_counter() - start,
    }
    _atomic_json(artifact, Path(artifact_path))
    return artifact


def _parse_seed_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cutoff", type=int, default=12)
    parser.add_argument("--confirmation-cutoff", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--train-batch", type=int, default=8)
    parser.add_argument("--validation-batch", type=int, default=32)
    parser.add_argument("--test-batch", type=int, default=64)
    parser.add_argument("--confirmation-batch", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=40)
    parser.add_argument("--training-seeds", type=_parse_seed_list, default=(101, 211, 307, 401, 503))
    parser.add_argument("--validation-seeds", type=_parse_seed_list, default=(10007, 10009))
    parser.add_argument(
        "--test-seeds",
        type=_parse_seed_list,
        default=(20011, 20021, 20023, 20029, 20047, 20051, 20063, 20071),
    )
    parser.add_argument(
        "--confirmation-seeds", type=_parse_seed_list, default=(30011, 30013, 30029, 30047)
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    config = DirectionalRankingConfig(
        cutoff=arguments.cutoff,
        confirmation_cutoff=arguments.confirmation_cutoff,
        full_cycles=arguments.cycles,
        train_epochs=arguments.epochs,
        train_batch_size=arguments.train_batch,
        validation_batch_size=arguments.validation_batch,
        test_batch_size=arguments.test_batch,
        confirmation_batch_size=arguments.confirmation_batch,
        validation_interval=arguments.validation_interval,
        training_seeds=arguments.training_seeds,
        validation_seeds=arguments.validation_seeds,
        test_seeds=arguments.test_seeds,
        confirmation_seeds=arguments.confirmation_seeds,
        device=arguments.device,
    )
    artifact = run_directional_ranking(
        config,
        artifact_path=arguments.artifact,
        checkpoint_path=arguments.checkpoint,
        production=not arguments.pilot,
        resume=not arguments.no_resume,
    )
    print(json.dumps({"status": artifact["status"], "gates": artifact["gates"]}, indent=2))
    return 0 if artifact["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DirectionalRankingConfig",
    "ANALYSIS_CONTRACT_ID",
    "NMFLatestOnlyView",
    "NMF_RANKING_SCOPE",
    "PAPER_MF_ARCHITECTURE",
    "PAPER_RNN_ARCHITECTURE",
    "POLICY_INITIALIZATION",
    "POLICY_INITIALIZATION_ID",
    "TRAINING_PROTOCOL_ID",
    "PaperScaleMFPolicy",
    "PaperScaleNMFPolicy",
    "build_policy",
    "evaluate_policy",
    "implementation_sha256",
    "run_directional_ranking",
    "state_dict_sha256",
    "train_agent",
    "validate_production_design",
]
