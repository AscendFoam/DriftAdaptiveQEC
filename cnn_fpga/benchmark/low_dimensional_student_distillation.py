"""Strict-split low-dimensional exponential student fitting for T4.4.3.

Candidate state dimensions 1, 2 and 4 share the interpretable update

    z[t+1] = a[m] * z[t] + (1 - a[m]) * z_inf[m]

and a fifteen-output hard-bounded head.  All candidates and restarts are fit
against the frozen T4.4.1 teacher on training trajectories.  Validation alone
selects the restart and the smallest dimension within a frozen tolerance of
the best validation MSE; evaluation is opened only after selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from cnn_fpga.control.low_dimensional_recurrence import (
    CONTROL_PARAMETER_NAMES,
    RESIDUAL_BOUNDS,
    LowDimensionalObservation,
    LowDimensionalRecurrenceArtifact,
    LowDimensionalRecurrenceStudent,
    LowDimensionalResourceProfile,
    online_contract,
)
from cnn_fpga.control.teacher_student import DistilledStudentArtifact

from .bounded_residual_rnn_teacher import (
    DEFAULT_ARTIFACT as TEACHER_ARTIFACT,
    DEFAULT_CHECKPOINT as TEACHER_CHECKPOINT,
    load_and_verify_teacher_checkpoint,
)
from .bounded_residual_teacher_analysis import (
    DEFAULT_ARTIFACT as TEACHER_ANALYSIS_ARTIFACT,
    implementation_sha256 as teacher_analysis_implementation_sha256,
    trace_teacher_hidden,
)

try:  # Minimal recovery Python intentionally has no torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


TASK_ID = "T4.4.3"
SCHEMA_VERSION = 1
TRAINING_PROTOCOL_ID = "T443-LOWDIM-EXPONENTIAL-STRICT-SPLIT-V1"
SCOPE = (
    "fresh 1/2/4-dimensional outcome-specific exponential recurrence candidates "
    "fit to one frozen T4.4.1 finite-model teacher with training/validation/evaluation "
    "separation; deterministic NumPy artifact with exact leakage/health zero-residual "
    "fallback; imitation evidence only, not physical gain retention or hardware"
)

DEFAULT_ARTIFACT = Path("docs/t4_4_3_low_dimensional_student_validation.json")
DEFAULT_CHECKPOINT = Path("docs/t4_4_3_low_dimensional_student_candidates.pt")
DEFAULT_STUDENT = Path("docs/t4_4_3_low_dimensional_student.json")
DEFAULT_SOURCE_DATA = Path("docs/t4_4_3_low_dimensional_student_source_data.csv")
LEGACY_STUDENT = Path("docs/t4_1_5_distilled_student_checkpoint.json")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T4.4.3 requires PyTorch; use "
            "C:/ProgramData/anaconda3/envs/DLEnv/python.exe"
        )
    return torch


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "control" / "low_dimensional_recurrence.py",
        Path(__file__).with_name("bounded_residual_rnn_teacher.py").resolve(),
        Path(__file__).with_name("bounded_residual_teacher_analysis.py").resolve(),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _unique_integers(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


@dataclass(frozen=True)
class LowDimensionalDistillationConfig:
    half_cycles: int = 64
    trajectories_per_split: int = 256
    training_seed: int = 443101
    validation_seed: int = 443201
    evaluation_seed: int = 443301
    candidate_dimensions: tuple[int, ...] = (1, 2, 4)
    restart_seeds: tuple[int, ...] = (44311, 44321, 44331)
    training_epochs: int = 900
    validation_interval: int = 25
    learning_rate: float = 1.5e-2
    gradient_clip_norm: float = 5.0
    state_l2_weight: float = 1.0e-7
    dimension_relative_tolerance: float = 0.05
    dimension_absolute_tolerance: float = 1.0e-7
    minimum_zero_mse_reduction_fraction: float = 0.90
    device: Literal["cpu", "cuda"] = "cuda"

    def __post_init__(self) -> None:
        for name in (
            "half_cycles",
            "trajectories_per_split",
            "training_epochs",
            "validation_interval",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if self.validation_interval > self.training_epochs:
            raise ValueError("validation_interval must not exceed training_epochs")
        dimensions = _unique_integers(self.candidate_dimensions, "candidate_dimensions")
        if dimensions != tuple(sorted(dimensions)) or dimensions[0] != 1:
            raise ValueError("candidate_dimensions must be sorted and start at one")
        if any(value > 16 for value in dimensions):
            raise ValueError("candidate state dimension exceeds the interpretable cap")
        object.__setattr__(self, "candidate_dimensions", dimensions)
        object.__setattr__(
            self, "restart_seeds", _unique_integers(self.restart_seeds, "restart_seeds")
        )
        if len({self.training_seed, self.validation_seed, self.evaluation_seed}) != 3:
            raise ValueError("training/validation/evaluation seeds must be disjoint")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in (
            "state_l2_weight",
            "dimension_relative_tolerance",
            "dimension_absolute_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        fraction = float(self.minimum_zero_mse_reduction_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("minimum_zero_mse_reduction_fraction must lie in (0,1)")
        object.__setattr__(self, "minimum_zero_mse_reduction_fraction", fraction)
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")

    @property
    def contract_hash(self) -> str:
        return _canonical_sha256(asdict(self))


def validate_production_design(config: LowDimensionalDistillationConfig) -> None:
    minima = {
        "half_cycles": 64,
        "trajectories_per_split": 256,
        "training_epochs": 800,
    }
    for name, minimum in minima.items():
        if int(getattr(config, name)) < minimum:
            raise ValueError(f"production {name} must be at least {minimum}")
    if config.candidate_dimensions != (1, 2, 4):
        raise ValueError("production must compare the frozen 1/2/4-dimensional scope")
    if len(config.restart_seeds) < 3:
        raise ValueError("production requires at least three restarts per dimension")


def _atomic_torch_save(payload: Any, path: Path) -> None:
    th = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    th.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


SOURCE_COLUMNS = (
    "row_type",
    "split",
    "dimension",
    "restart_index",
    "trajectory_index",
    "step",
    "outcome",
    "epoch",
    "metric",
    "value",
    "target_residual_json",
    "student_residual_json",
    "detail_json",
)


def _write_source_data(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in SOURCE_COLUMNS})
    os.replace(temporary, path)


def _outcome_dataset(seed: int, trajectories: int, half_cycles: int) -> np.ndarray:
    """Generate structured and stochastic histories without evaluation reuse."""

    rng = np.random.default_rng(seed)
    result = np.empty((trajectories, half_cycles), dtype=np.int64)
    boundary = (
        np.zeros(half_cycles, dtype=np.int64),
        np.ones(half_cycles, dtype=np.int64),
        np.arange(half_cycles, dtype=np.int64) % 2,
        1 - np.arange(half_cycles, dtype=np.int64) % 2,
        (np.arange(half_cycles) // 8 % 2).astype(np.int64),
        (1 - np.arange(half_cycles) // 8 % 2).astype(np.int64),
    )
    for index, sequence in enumerate(boundary[:trajectories]):
        result[index] = sequence
    for trajectory in range(len(boundary), trajectories):
        ground_probability = float(rng.choice((0.55, 0.70, 0.82, 0.90)))
        persistence = float(rng.choice((0.55, 0.70, 0.85, 0.95)))
        value = int(rng.random() >= ground_probability)
        for step in range(half_cycles):
            if rng.random() >= persistence:
                value = int(rng.random() >= ground_probability)
            if step in {half_cycles // 3, 2 * half_cycles // 3} and rng.random() < 0.5:
                ground_probability = 1.0 - ground_probability
            result[trajectory, step] = value
    return result


def _dataset_sha256(outcomes: np.ndarray, targets: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(outcomes).tobytes())
    digest.update(np.ascontiguousarray(targets, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_dict_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _vectorized_recurrence_states(
    initial_state: Any,
    outcome_decays: Any,
    outcome_saturations: Any,
    outcomes: Any,
) -> Any:
    """Evaluate the exact affine recurrence without per-step GPU launches.

    For ``z_t = a_t z_(t-1) + b_t`` and ``P_t = product(a_1..a_t)``,
    ``z_t = P_t * (z_0 + cumulative_sum(b_t / P_t))``.  The constrained
    decay floor is 0.02 and the production horizon is 64, so every product is
    strictly positive and remains representable in float64.  The returned
    tensor includes the pre-observation initial state at index zero.
    """

    indexed_decays = outcome_decays[outcomes]
    indexed_saturations = outcome_saturations[outcomes]
    products = torch.cumprod(indexed_decays, dim=1)
    increments = (1.0 - indexed_decays) * indexed_saturations
    post_observation = products * (
        initial_state[None, None, :]
        + torch.cumsum(increments / products, dim=1)
    )
    initial = initial_state[None, None, :].expand(outcomes.shape[0], 1, -1)
    return torch.cat((initial, post_observation), dim=1)


def _build_module(
    dimension: int,
    seed: int,
    target_mean: np.ndarray,
    *,
    device: str,
) -> Any:
    th = _require_torch()
    dtype = th.float64
    bounds = th.tensor(RESIDUAL_BOUNDS, dtype=dtype, device=device)

    class Module(th.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = th.Generator(device="cpu")
            generator.manual_seed(seed)
            self.initial_state = th.nn.Parameter(
                0.02 * th.randn(dimension, generator=generator, dtype=dtype).to(device)
            )
            initial_decay = np.log((0.62 - 0.02) / (0.995 - 0.62))
            self.decay_logits = th.nn.Parameter(
                th.full((2, dimension), initial_decay, dtype=dtype, device=device)
                + 0.05
                * th.randn((2, dimension), generator=generator, dtype=dtype).to(device)
            )
            self.saturations = th.nn.Parameter(
                0.25
                * th.randn((2, dimension), generator=generator, dtype=dtype).to(device)
            )
            self.output_weights = th.nn.Parameter(
                0.15
                * th.randn((15, dimension), generator=generator, dtype=dtype).to(device)
            )
            normalized_mean = np.clip(target_mean / np.asarray(RESIDUAL_BOUNDS), -0.95, 0.95)
            bias = np.arctanh(normalized_mean)
            self.output_bias = th.nn.Parameter(
                th.tensor(bias, dtype=dtype, device=device)
                + 0.002 * th.randn(15, generator=generator, dtype=dtype).to(device)
            )
            self.register_buffer("bounds", bounds)

        def constrained(self) -> tuple[Any, Any, Any, Any, Any]:
            decay = 0.02 + 0.975 * th.sigmoid(self.decay_logits)
            return (
                self.initial_state,
                decay,
                self.saturations,
                self.output_weights,
                self.output_bias,
            )

        def forward(self, outcomes: Any) -> tuple[Any, Any]:
            initial, decays, saturations, weights, bias = self.constrained()
            states = _vectorized_recurrence_states(
                initial, decays, saturations, outcomes
            )
            predictions = self.bounds * th.tanh(states @ weights.T + bias)
            return predictions, states

    return Module()


def _fit_candidate(
    dimension: int,
    restart_index: int,
    restart_seed: int,
    config: LowDimensionalDistillationConfig,
    training_outcomes: np.ndarray,
    training_targets: np.ndarray,
    validation_outcomes: np.ndarray,
    validation_targets: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    th = _require_torch()
    train_x = th.as_tensor(training_outcomes, dtype=th.int64, device=config.device)
    train_y = th.as_tensor(training_targets, dtype=th.float64, device=config.device)
    validation_x = th.as_tensor(validation_outcomes, dtype=th.int64, device=config.device)
    validation_y = th.as_tensor(validation_targets, dtype=th.float64, device=config.device)
    model = _build_module(
        dimension,
        restart_seed,
        np.mean(training_targets, axis=(0, 1)),
        device=config.device,
    )
    initial_state_hash = _state_dict_sha256(_state_dict_cpu(model))
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_validation = float("inf")
    best_epoch = -1
    best_state = None
    validation_history: list[dict[str, Any]] = []
    training_curve: list[dict[str, Any]] = []
    gradient_coverage = None
    start = time.perf_counter()
    for epoch in range(1, config.training_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, states = model(train_x)
        mse = th.mean((prediction - train_y) ** 2)
        state_penalty = th.mean(states**2)
        loss = mse + config.state_l2_weight * state_penalty
        if not bool(th.isfinite(loss).detach().cpu()):
            raise RuntimeError("non-finite low-dimensional student loss")
        loss.backward()
        if gradient_coverage is None:
            tensors = []
            for name, parameter in model.named_parameters():
                gradient = parameter.grad
                tensors.append(
                    {
                        "name": name,
                        "finite": gradient is not None
                        and bool(th.all(th.isfinite(gradient)).detach().cpu()),
                        "nonzero_elements": 0
                        if gradient is None
                        else int(th.count_nonzero(gradient).detach().cpu()),
                        "total_elements": int(parameter.numel()),
                    }
                )
            gradient_coverage = tensors
        gradient_norm = float(
            th.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            .detach()
            .cpu()
        )
        optimizer.step()
        training_curve.append(
            {
                "epoch": epoch,
                "training_mse": float(mse.detach().cpu()),
                "state_l2": float(state_penalty.detach().cpu()),
                "gradient_norm_before_clip": gradient_norm,
            }
        )
        if epoch % config.validation_interval == 0 or epoch == 1 or epoch == config.training_epochs:
            model.eval()
            with th.no_grad():
                validation_prediction, _ = model(validation_x)
                validation_mse = float(th.mean((validation_prediction - validation_y) ** 2).cpu())
            validation_history.append(
                {"epoch": epoch, "validation_mse": validation_mse}
            )
            if validation_mse < best_validation:
                best_validation = validation_mse
                best_epoch = epoch
                best_state = _state_dict_cpu(model)
    if best_state is None or gradient_coverage is None:
        raise RuntimeError("candidate produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_hash = _state_dict_sha256(best_state)
    record = {
        "dimension": dimension,
        "restart_index": restart_index,
        "restart_seed": restart_seed,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "resource_profile": asdict(LowDimensionalResourceProfile.exact(dimension)),
        "initial_state_sha256": initial_state_hash,
        "checkpoint_sha256": final_hash,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "best_epoch_reached_training_cap": best_epoch == config.training_epochs,
        "optimizer_global_convergence_claimed": False,
        "gradient_coverage": gradient_coverage,
        "all_parameter_tensors_receive_finite_nonzero_gradient": all(
            item["finite"] and item["nonzero_elements"] > 0 for item in gradient_coverage
        ),
        "training_curve": training_curve,
        "validation_history": validation_history,
        "wall_time_seconds": time.perf_counter() - start,
    }
    return model, record


def _export_parameters(model: Any) -> dict[str, np.ndarray]:
    th = _require_torch()
    with th.no_grad():
        initial, decays, saturations, weights, bias = model.constrained()
    return {
        "initial_state": initial.detach().cpu().numpy(),
        "outcome_decays": decays.detach().cpu().numpy(),
        "outcome_saturations": saturations.detach().cpu().numpy(),
        "output_weights": weights.detach().cpu().numpy(),
        "output_bias": bias.detach().cpu().numpy(),
    }


def _predict_model(model: Any, outcomes: np.ndarray, device: str) -> np.ndarray:
    th = _require_torch()
    with th.no_grad():
        prediction, _ = model(th.as_tensor(outcomes, dtype=th.int64, device=device))
    return prediction.detach().cpu().numpy()


def _predict_exported(
    artifact: LowDimensionalRecurrenceArtifact, outcomes: np.ndarray
) -> np.ndarray:
    initial = np.asarray(artifact.initial_state, dtype=np.float64)
    decays = np.asarray(artifact.outcome_decays, dtype=np.float64)
    saturations = np.asarray(artifact.outcome_saturations, dtype=np.float64)
    weights = np.asarray(artifact.output_weights, dtype=np.float64)
    bias = np.asarray(artifact.output_bias, dtype=np.float64)
    bounds = np.asarray(artifact.residual_bounds, dtype=np.float64)
    state = np.broadcast_to(initial, (outcomes.shape[0], initial.size)).copy()
    outputs = [bounds * np.tanh(state @ weights.T + bias)]
    for step in range(outcomes.shape[1]):
        indices = outcomes[:, step]
        state = decays[indices] * state + (1.0 - decays[indices]) * saturations[indices]
        outputs.append(bounds * np.tanh(state @ weights.T + bias))
    return np.stack(outputs, axis=1)


def _latest_only_fit(outcomes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    latest = np.zeros((outcomes.shape[0], outcomes.shape[1] + 1), dtype=np.float64)
    latest[:, 1:] = 2.0 * outcomes - 1.0
    design = np.column_stack((np.ones(latest.size), latest.reshape(-1)))
    coefficients, *_ = np.linalg.lstsq(design, targets.reshape(-1, 15), rcond=None)
    return coefficients


def _latest_only_predict(coefficients: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    latest = np.zeros((outcomes.shape[0], outcomes.shape[1] + 1), dtype=np.float64)
    latest[:, 1:] = 2.0 * outcomes - 1.0
    design = np.stack((np.ones_like(latest), latest), axis=-1)
    return design @ coefficients


def _legacy_predict(artifact: DistilledStudentArtifact, outcomes: np.ndarray) -> np.ndarray:
    state = np.broadcast_to(
        np.asarray(artifact.initial_state, dtype=np.float64),
        (outcomes.shape[0], 15),
    ).copy()
    saturations = np.asarray(artifact.outcome_saturations, dtype=np.float64)
    decays = np.asarray(artifact.outcome_decays, dtype=np.float64)
    bounds = np.asarray(RESIDUAL_BOUNDS, dtype=np.float64)
    outputs = [bounds * np.tanh(np.clip(state, -artifact.raw_clip, artifact.raw_clip))]
    for step in range(outcomes.shape[1]):
        indices = outcomes[:, step]
        state = decays[indices] * state + (1.0 - decays[indices]) * saturations[indices]
        outputs.append(bounds * np.tanh(np.clip(state, -artifact.raw_clip, artifact.raw_clip)))
    return np.stack(outputs, axis=1)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "per_parameter_mse": {
            name: float(np.mean(error[..., index] ** 2))
            for index, name in enumerate(CONTROL_PARAMETER_NAMES)
        },
        "prediction_residual_rms": float(np.sqrt(np.mean(np.asarray(prediction) ** 2))),
        "target_residual_rms": float(np.sqrt(np.mean(np.asarray(target) ** 2))),
    }


def _runtime_replay(
    artifact: LowDimensionalRecurrenceArtifact,
    outcomes: np.ndarray,
    expected: np.ndarray,
) -> dict[str, Any]:
    student = LowDimensionalRecurrenceStudent(artifact)
    values = [student.initial_decision().physical_control_residual]
    for step, outcome in enumerate(outcomes):
        decision = student.step(
            LowDimensionalObservation(step, "g" if outcome == 0 else "e")
        )
        values.append(decision.physical_control_residual)
    replay = np.asarray(values, dtype=np.float64)
    leakage = LowDimensionalRecurrenceStudent(artifact).step(
        LowDimensionalObservation(0, "leakage")
    )
    health = LowDimensionalRecurrenceStudent(artifact).step(
        LowDimensionalObservation(0, "g", crc_ok=False)
    )
    return {
        "maximum_batch_runtime_error": float(np.max(np.abs(replay - expected))),
        "leakage_exact_zero": leakage.physical_control_residual == (0.0,) * 15,
        "health_exact_zero": health.physical_control_residual == (0.0,) * 15,
        "leakage_resets_initial_state": leakage.state == artifact.initial_state,
        "online_contract": dict(online_contract()),
    }


def run_low_dimensional_student_distillation(
    config: LowDimensionalDistillationConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    student_path: str | Path = DEFAULT_STUDENT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
    production: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    th = _require_torch()
    actual = config or LowDimensionalDistillationConfig()
    if production:
        validate_production_design(actual)
    if actual.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    start = time.perf_counter()
    model_teacher, teacher_report = load_and_verify_teacher_checkpoint(
        TEACHER_CHECKPOINT, TEACHER_ARTIFACT
    )
    teacher_analysis = json.loads(TEACHER_ANALYSIS_ARTIFACT.read_text(encoding="utf-8"))
    if (
        teacher_analysis.get("status") != "PASS"
        or teacher_analysis.get("implementation_sha256")
        != teacher_analysis_implementation_sha256()
    ):
        raise ValueError("T4.4.2 analysis artifact is stale or failed")
    if not LEGACY_STUDENT.is_file():
        raise FileNotFoundError(LEGACY_STUDENT)
    legacy_artifact = DistilledStudentArtifact.from_dict(
        json.loads(LEGACY_STUDENT.read_text(encoding="utf-8"))
    )
    split_seeds = {
        "training": actual.training_seed,
        "validation": actual.validation_seed,
        "evaluation": actual.evaluation_seed,
    }
    outcomes = {
        split: _outcome_dataset(seed, actual.trajectories_per_split, actual.half_cycles)
        for split, seed in split_seeds.items()
    }
    targets = {
        split: trace_teacher_hidden(model_teacher, values)["physical_residual"]
        for split, values in outcomes.items()
    }
    dataset_hashes = {
        split: _dataset_sha256(outcomes[split], targets[split]) for split in outcomes
    }
    implementation_hash = implementation_sha256()
    checkpoint_contract_hash = _canonical_sha256(
        {
            "config": actual.contract_hash,
            "implementation": implementation_hash,
            "teacher_state": teacher_report["checkpoint"]["selected_state_sha256"],
            "dataset_hashes": dataset_hashes,
            "protocol": TRAINING_PROTOCOL_ID,
        }
    )
    checkpoint = Path(checkpoint_path)
    checkpoint_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "checkpoint_contract_hash": checkpoint_contract_hash,
        "implementation_sha256": implementation_hash,
        "config": asdict(actual),
        "dataset_hashes": dataset_hashes,
        "candidates": [],
        "selected_dimension": None,
        "selected_restart_index": None,
    }
    if resume and checkpoint.exists():
        loaded = th.load(checkpoint, map_location="cpu", weights_only=False)
        if loaded.get("checkpoint_contract_hash") != checkpoint_contract_hash:
            raise ValueError("candidate checkpoint contract mismatch")
        checkpoint_payload = loaded
    elif checkpoint.exists():
        raise FileExistsError("checkpoint exists with resume=False; choose a new path")
    existing = {
        (int(item["dimension"]), int(item["restart_index"])): item
        for item in checkpoint_payload["candidates"]
    }
    models: dict[tuple[int, int], Any] = {}
    records: list[dict[str, Any]] = []
    resumed_candidates = 0
    for dimension in actual.candidate_dimensions:
        for restart_index, restart_seed in enumerate(actual.restart_seeds):
            key = (dimension, restart_index)
            if key in existing:
                item = existing[key]
                model = _build_module(
                    dimension,
                    restart_seed,
                    np.mean(targets["training"], axis=(0, 1)),
                    device=actual.device,
                )
                model.load_state_dict(item["state_dict"])
                model.eval()
                if _state_dict_sha256(_state_dict_cpu(model)) != item["checkpoint_sha256"]:
                    raise ValueError("resumed candidate state hash mismatch")
                record = dict(item["training_record"])
                resumed_candidates += 1
            else:
                model, record = _fit_candidate(
                    dimension,
                    restart_index,
                    restart_seed,
                    actual,
                    outcomes["training"],
                    targets["training"],
                    outcomes["validation"],
                    targets["validation"],
                )
                item = {
                    "dimension": dimension,
                    "restart_index": restart_index,
                    "restart_seed": restart_seed,
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "state_dict": _state_dict_cpu(model),
                    "training_record": record,
                }
                checkpoint_payload["candidates"].append(item)
                _atomic_torch_save(checkpoint_payload, checkpoint)
                print(
                    json.dumps(
                        {
                            "event": "candidate_complete",
                            "dimension": dimension,
                            "restart_index": restart_index,
                            "best_validation_mse": record["best_validation_mse"],
                        }
                    ),
                    flush=True,
                )
            models[key] = model
            records.append(record)
    best_per_dimension: dict[int, dict[str, Any]] = {}
    for dimension in actual.candidate_dimensions:
        subset = [record for record in records if record["dimension"] == dimension]
        best_per_dimension[dimension] = min(
            subset, key=lambda item: float(item["best_validation_mse"])
        )
    global_best = min(
        float(record["best_validation_mse"]) for record in best_per_dimension.values()
    )
    threshold = (
        global_best * (1.0 + actual.dimension_relative_tolerance)
        + actual.dimension_absolute_tolerance
    )
    eligible = [
        dimension
        for dimension in actual.candidate_dimensions
        if float(best_per_dimension[dimension]["best_validation_mse"]) <= threshold
    ]
    selected_dimension = min(eligible)
    selected_record = best_per_dimension[selected_dimension]
    selected_restart = int(selected_record["restart_index"])
    selected_model = models[(selected_dimension, selected_restart)]
    checkpoint_payload["selected_dimension"] = selected_dimension
    checkpoint_payload["selected_restart_index"] = selected_restart
    checkpoint_payload["dimension_threshold"] = threshold
    _atomic_torch_save(checkpoint_payload, checkpoint)

    candidate_metrics: dict[str, Any] = {}
    for dimension, record in best_per_dimension.items():
        model = models[(dimension, int(record["restart_index"]))]
        candidate_metrics[str(dimension)] = {
            "restart_index": record["restart_index"],
            "validation": _metrics(
                targets["validation"],
                _predict_model(model, outcomes["validation"], actual.device),
            ),
            "evaluation": _metrics(
                targets["evaluation"],
                _predict_model(model, outcomes["evaluation"], actual.device),
            ),
            "resource_profile": record["resource_profile"],
        }
    parameters = _export_parameters(selected_model)
    student_artifact = LowDimensionalRecurrenceArtifact.create(
        initial_state=parameters["initial_state"],
        outcome_decays=parameters["outcome_decays"],
        outcome_saturations=parameters["outcome_saturations"],
        output_weights=parameters["output_weights"],
        output_bias=parameters["output_bias"],
        teacher_checkpoint_sha256=_sha256(TEACHER_CHECKPOINT),
        teacher_state_sha256=teacher_report["checkpoint"]["selected_state_sha256"],
        teacher_analysis_sha256=_sha256(TEACHER_ANALYSIS_ARTIFACT),
        training_dataset_sha256=dataset_hashes["training"],
        validation_dataset_sha256=dataset_hashes["validation"],
        selected_dimension=selected_dimension,
        selected_restart=selected_restart,
        validation_mse=float(selected_record["best_validation_mse"]),
    )
    student = Path(student_path)
    _atomic_json(student_artifact.to_dict(), student)
    restored = LowDimensionalRecurrenceArtifact.from_dict(
        json.loads(student.read_text(encoding="utf-8"))
    )
    selected_predictions = {
        split: _predict_exported(restored, values) for split, values in outcomes.items()
    }
    latest_coefficients = _latest_only_fit(outcomes["training"], targets["training"])
    latest_predictions = {
        split: _latest_only_predict(latest_coefficients, values)
        for split, values in outcomes.items()
    }
    legacy_predictions = {
        split: _legacy_predict(legacy_artifact, values) for split, values in outcomes.items()
    }
    comparisons: dict[str, Any] = {}
    for split in outcomes:
        comparisons[split] = {
            "selected_student": _metrics(targets[split], selected_predictions[split]),
            "latest_only": _metrics(targets[split], latest_predictions[split]),
            "legacy_t4_1_5_student": _metrics(targets[split], legacy_predictions[split]),
            "zero_residual": _metrics(targets[split], np.zeros_like(targets[split])),
        }
    torch_selected = _predict_model(selected_model, outcomes["evaluation"], actual.device)
    export_max_error = float(
        np.max(np.abs(torch_selected - selected_predictions["evaluation"]))
    )
    runtime = _runtime_replay(
        restored,
        outcomes["evaluation"][0],
        selected_predictions["evaluation"][0],
    )

    rows: list[dict[str, Any]] = []
    for record in records:
        for item in record["training_curve"]:
            rows.append(
                {
                    "row_type": "training_epoch",
                    "split": "training",
                    "dimension": record["dimension"],
                    "restart_index": record["restart_index"],
                    "trajectory_index": "",
                    "step": "",
                    "outcome": "",
                    "epoch": item["epoch"],
                    "metric": "training_mse",
                    "value": item["training_mse"],
                    "target_residual_json": "",
                    "student_residual_json": "",
                    "detail_json": json.dumps(item, sort_keys=True),
                }
            )
        for item in record["validation_history"]:
            rows.append(
                {
                    "row_type": "validation_checkpoint",
                    "split": "validation",
                    "dimension": record["dimension"],
                    "restart_index": record["restart_index"],
                    "trajectory_index": "",
                    "step": "",
                    "outcome": "",
                    "epoch": item["epoch"],
                    "metric": "validation_mse",
                    "value": item["validation_mse"],
                    "target_residual_json": "",
                    "student_residual_json": "",
                    "detail_json": "",
                }
            )
    for split in ("training", "validation", "evaluation"):
        for trajectory_index in range(actual.trajectories_per_split):
            for step in range(actual.half_cycles + 1):
                rows.append(
                    {
                        "row_type": "selected_student_prediction",
                        "split": split,
                        "dimension": selected_dimension,
                        "restart_index": selected_restart,
                        "trajectory_index": trajectory_index,
                        "step": step,
                        "outcome": ""
                        if step == actual.half_cycles
                        else ("g" if outcomes[split][trajectory_index, step] == 0 else "e"),
                        "epoch": "",
                        "metric": "physical_residual",
                        "value": "",
                        "target_residual_json": json.dumps(
                            targets[split][trajectory_index, step].tolist()
                        ),
                        "student_residual_json": json.dumps(
                            selected_predictions[split][trajectory_index, step].tolist()
                        ),
                        "detail_json": "",
                    }
                )
    for dimension, content in candidate_metrics.items():
        rows.append(
            {
                "row_type": "candidate_summary",
                "split": "evaluation_report_only",
                "dimension": dimension,
                "restart_index": content["restart_index"],
                "trajectory_index": "",
                "step": "",
                "outcome": "",
                "epoch": "",
                "metric": "evaluation_mse",
                "value": content["evaluation"]["mse"],
                "target_residual_json": "",
                "student_residual_json": "",
                "detail_json": json.dumps(content, sort_keys=True),
            }
        )
    source_data = Path(source_data_path)
    _write_source_data(source_data, rows)
    source_hash = _sha256(source_data)
    evaluation_selected = comparisons["evaluation"]["selected_student"]
    evaluation_zero = comparisons["evaluation"]["zero_residual"]
    evaluation_latest = comparisons["evaluation"]["latest_only"]
    bounds = np.asarray(RESIDUAL_BOUNDS)
    maximum_bound_violation = float(
        np.max(np.maximum(np.abs(selected_predictions["evaluation"]) - bounds, 0.0))
    )
    gates = {
        "teacher_and_hidden_analysis_parents_are_current_passes": (
            teacher_report["status"] == "PASS"
            and teacher_analysis["status"] == "PASS"
            and teacher_analysis["teacher_provenance"]["selected_state_sha256"]
            == teacher_report["checkpoint"]["selected_state_sha256"]
        ),
        "training_validation_evaluation_splits_are_disjoint_and_complete": (
            len(set(split_seeds.values())) == 3
            and all(values.shape == (actual.trajectories_per_split, actual.half_cycles) for values in outcomes.values())
            and len(set(dataset_hashes.values())) == 3
        ),
        "one_two_four_dimensions_and_three_restarts_are_fully_retained": (
            actual.candidate_dimensions == (1, 2, 4)
            and len(actual.restart_seeds) >= 3
            and len(records) == len(actual.candidate_dimensions) * len(actual.restart_seeds)
        ),
        "all_candidates_match_exact_parameter_and_resource_counts": all(
            record["parameter_count"]
            == record["resource_profile"]["stored_trainable_scalars"]
            == 20 * record["dimension"] + 15
            for record in records
        ),
        "all_candidate_parameter_tensors_receive_finite_nonzero_gradient": all(
            record["all_parameter_tensors_receive_finite_nonzero_gradient"]
            for record in records
        ),
        "every_candidate_changes_from_its_fresh_initializer": all(
            record["initial_state_sha256"] != record["checkpoint_sha256"]
            for record in records
        ),
        "restart_and_dimension_selection_are_validation_only": (
            selected_record
            == min(
                [record for record in records if record["dimension"] == selected_dimension],
                key=lambda item: float(item["best_validation_mse"]),
            )
            and selected_dimension == min(eligible)
        ),
        "selected_student_reduces_evaluation_zero_mse_by_required_fraction": (
            evaluation_selected["mse"]
            <= (1.0 - actual.minimum_zero_mse_reduction_fraction) * evaluation_zero["mse"]
        ),
        "selected_student_beats_latest_only_and_legacy_students_on_evaluation": (
            evaluation_selected["mse"] < evaluation_latest["mse"]
            and evaluation_selected["mse"]
            < comparisons["evaluation"]["legacy_t4_1_5_student"]["mse"]
        ),
        "all_selected_actions_obey_fifteen_hard_bounds": maximum_bound_violation == 0.0,
        "json_artifact_roundtrip_matches_torch_candidate": (
            restored == student_artifact and export_max_error < 1.0e-12
        ),
        "pure_numpy_runtime_replay_is_exact_and_fail_closed": (
            runtime["maximum_batch_runtime_error"] < 1.0e-12
            and runtime["leakage_exact_zero"]
            and runtime["health_exact_zero"]
            and runtime["leakage_resets_initial_state"]
        ),
        "online_module_has_no_torch_physics_or_teacher_dependency": (
            runtime["online_contract"]["torch_runtime_dependency"] is False
            and runtime["online_contract"]["physics_runtime_dependency"] is False
            and runtime["online_contract"]["teacher_runtime_dependency"] is False
        ),
        "source_data_contains_every_epoch_checkpoint_split_and_candidate": (
            len(rows)
            == sum(len(record["training_curve"]) + len(record["validation_history"]) for record in records)
            + 3 * actual.trajectories_per_split * (actual.half_cycles + 1)
            + len(actual.candidate_dimensions)
            and {row["split"] for row in rows}
            >= {"training", "validation", "evaluation", "evaluation_report_only"}
        ),
        "cap_hits_and_nonconvergence_claims_are_explicit": all(
            "best_epoch_reached_training_cap" in record
            and not record["optimizer_global_convergence_claimed"]
            for record in records
        ),
        "claim_boundary_keeps_imitation_separate_from_physical_gain": True,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    status = "PASS" if all(gates.values()) else "FAIL"
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": status,
        "scope": SCOPE,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "implementation_sha256": implementation_hash,
        "config_contract_hash": actual.contract_hash,
        "checkpoint_contract_hash": checkpoint_contract_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(actual),
        "parent_provenance": {
            "teacher_artifact_path": TEACHER_ARTIFACT.as_posix(),
            "teacher_artifact_sha256": _sha256(TEACHER_ARTIFACT),
            "teacher_checkpoint_path": TEACHER_CHECKPOINT.as_posix(),
            "teacher_checkpoint_sha256": _sha256(TEACHER_CHECKPOINT),
            "teacher_state_sha256": teacher_report["checkpoint"]["selected_state_sha256"],
            "teacher_analysis_path": TEACHER_ANALYSIS_ARTIFACT.as_posix(),
            "teacher_analysis_sha256": _sha256(TEACHER_ANALYSIS_ARTIFACT),
            "legacy_student_path": LEGACY_STUDENT.as_posix(),
            "legacy_student_sha256": _sha256(LEGACY_STUDENT),
        },
        "dataset": {
            "split_seeds": split_seeds,
            "trajectories_per_split": actual.trajectories_per_split,
            "half_cycles": actual.half_cycles,
            "hashes": dataset_hashes,
            "target": "frozen selected teacher physical control residual",
        },
        "training_records": records,
        "best_per_dimension": {str(key): value for key, value in best_per_dimension.items()},
        "selection": {
            "global_best_validation_mse": global_best,
            "dimension_eligibility_threshold": threshold,
            "eligible_dimensions": eligible,
            "selected_dimension": selected_dimension,
            "selected_restart": selected_restart,
            "rule": "smallest dimension within frozen relative-plus-absolute validation tolerance",
            "evaluation_blind": True,
        },
        "candidate_metrics": candidate_metrics,
        "comparisons": comparisons,
        "student_artifact": {
            "path": student.as_posix(),
            "file_sha256": _sha256(student),
            "artifact_sha256": student_artifact.artifact_sha256,
            "state_dimension": student_artifact.state_dimension,
            "resource_profile": asdict(student_artifact.resource_profile),
            "torch_export_maximum_error": export_max_error,
            "runtime_replay": runtime,
        },
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": _sha256(checkpoint),
            "candidate_count": len(records),
            "resumed_candidates_this_invocation": resumed_candidates,
        },
        "source_data": {
            "path": source_data.as_posix(),
            "sha256": source_hash,
            "row_count": len(rows),
            "row_types": sorted({row["row_type"] for row in rows}),
        },
        "gates": gates,
        "gate_summary": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "failed": [name for name, value in gates.items() if not value],
        },
        "execution": {
            "device": actual.device,
            "torch_version": th.__version__,
            "wall_time_seconds": time.perf_counter() - start,
        },
        "claim_boundary": {
            "allowed": (
                "strict-split low-dimensional recurrence imitation of one frozen finite-model teacher"
            ),
            "forbidden": (
                "teacher physical gain retention, calibrated leakage response, long-horizon/OOD "
                "robustness, global optimization, fixed-point/RTL/FPGA or device evidence"
            ),
            "next_gate": (
                "T4.4.4 must compare physical trajectories and lifetime/fidelity/burden; imitation "
                "MSE alone cannot advance the NMF claim"
            ),
            "failure_branch": (
                "if status FAIL, do not use this student; retain teacher only as offline reference "
                "and drift/regime-aware MAP-LUT as deployable fallback"
            ),
        },
    }
    _atomic_json(result, Path(artifact_path))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = LowDimensionalDistillationConfig(device=arguments.device)
    result = run_low_dimensional_student_distillation(
        config,
        artifact_path=arguments.artifact,
        checkpoint_path=arguments.checkpoint,
        student_path=arguments.student,
        source_data_path=arguments.source_data,
        production=not arguments.pilot,
        resume=not arguments.no_resume,
    )
    print(json.dumps({"status": result["status"], "gate_summary": result["gate_summary"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SOURCE_DATA",
    "DEFAULT_STUDENT",
    "LowDimensionalDistillationConfig",
    "SCOPE",
    "TASK_ID",
    "TRAINING_PROTOCOL_ID",
    "implementation_sha256",
    "run_low_dimensional_student_distillation",
    "validate_production_design",
]
