"""T5.4.5 training-horizon and long-deployment extrapolation audit.

This module deliberately separates three claims that are easy to conflate:

* a validation-only sweep of low-dimensional students fit on 2/5/10 cycles;
* exact recurrent-state execution through 1e3/1e5/1e6 deployment cycles;
* sampled teacher-action imitation and controlled hidden-state reset recovery.

Every registered half-cycle is actually consumed by the GRU-10 and by each
affine student recurrence.  Dense teacher actions are evaluated at a frozen
set of checkpoints because evaluating a 256x256 head at every one of two
million half-cycles would add cost without strengthening the bounded-state
claim.  No long-horizon Fock-space logical channel is executed here, so this
task cannot establish physical-memory lifetime, logical gain, leakage
robustness, device calibration, or hardware performance.
"""

from __future__ import annotations

import argparse
import copy
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

from cnn_fpga.benchmark import low_dimensional_student_distillation as student_parent
from cnn_fpga.benchmark.bounded_residual_rnn_teacher import (
    DEFAULT_ARTIFACT as TEACHER_ARTIFACT,
    DEFAULT_CHECKPOINT as TEACHER_CHECKPOINT,
    load_and_verify_teacher_checkpoint,
)
from cnn_fpga.benchmark.bounded_residual_teacher_analysis import trace_teacher_hidden
from cnn_fpga.control.low_dimensional_recurrence import (
    RESIDUAL_BOUNDS,
    LowDimensionalRecurrenceArtifact,
)

try:  # The recovery interpreter intentionally has no torch installation.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # Numba keeps the exact 2e6-step student scan practical on the host.
    import numba
except ModuleNotFoundError:  # pragma: no cover
    numba = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.4.5"
SCHEMA_VERSION = "t5.4.5-horizon-extrapolation-v1"
PROTOCOL_ID = "STRICT-SPLIT-REAL-RECURRENCE-LONG-HORIZON-V1"
DEFAULT_ARTIFACT = Path("docs/t5_4_5_horizon_extrapolation_validation.json")
DEFAULT_CHECKPOINT = Path("docs/t5_4_5_horizon_extrapolation_candidates.pt")
DEFAULT_SOURCE_DATA = Path(
    "docs/t5_4_5_horizon_extrapolation_validation_source_data.csv"
)
PRODUCTION_STUDENT = Path("docs/t4_4_3_low_dimensional_student.json")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T4.4.1": Path("docs/t4_4_1_bounded_residual_rnn_teacher_validation.json"),
    "T4.4.3": Path("docs/t4_4_3_low_dimensional_student_validation.json"),
    "T4.4.4": Path("docs/t4_4_4_teacher_student_gain_retention.json"),
    "T4.4.5": Path("docs/t4_4_5_teacher_student_branch_freeze.json"),
    "T5.0.1": Path("docs/t5_0_1_literature_trend_reproduction.json"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/horizon_extrapolation_validation.py"),
    Path("cnn_fpga/benchmark/bounded_residual_rnn_teacher.py"),
    Path("cnn_fpga/benchmark/bounded_residual_teacher_analysis.py"),
    Path("cnn_fpga/benchmark/low_dimensional_student_distillation.py"),
    Path("cnn_fpga/control/low_dimensional_recurrence.py"),
    Path("physics/nmf_directional_ranking.py"),
)

STREAM_FAMILIES = (
    "stationary_nominal",
    "persistent_regime",
    "range_shift",
    "all_g_boundary",
    "all_e_boundary",
)


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T5.4.5 requires PyTorch; use "
            "C:/ProgramData/anaconda3/envs/DLEnv/python.exe"
        )
    return torch


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _unique_positive(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(_positive_integer(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


@dataclass(frozen=True)
class HorizonExtrapolationConfig:
    training_horizons_cycles: tuple[int, ...] = (2, 5, 10, 32)
    fresh_fit_horizons_cycles: tuple[int, ...] = (2, 5, 10)
    deployment_horizons_cycles: tuple[int, ...] = (1_000, 100_000, 1_000_000)
    trajectories_per_split: int = 256
    training_seed: int = 443101
    validation_seed: int = 443201
    evaluation_seed: int = 443301
    restart_seeds: tuple[int, ...] = (44311, 44321, 44331)
    training_epochs: int = 900
    validation_interval: int = 25
    learning_rate: float = 1.5e-2
    gradient_clip_norm: float = 5.0
    state_l2_weight: float = 1.0e-7
    stream_seeds: tuple[int, ...] = (545101, 545103)
    sample_points_per_horizon: int = 4096
    reset_window_half_cycles: int = 256
    reset_recovery_consecutive_points: int = 8
    teacher_chunk_half_cycles: int = 32_768
    maximum_imitation_mse: float = 5.0e-5
    maximum_float32_action_error: float = 1.0e-4
    maximum_reset_recovery_half_cycles: int = 128
    device: Literal["cpu", "cuda"] = "cuda"

    def __post_init__(self) -> None:
        training = _unique_positive(
            self.training_horizons_cycles, "training_horizons_cycles"
        )
        fresh = _unique_positive(
            self.fresh_fit_horizons_cycles, "fresh_fit_horizons_cycles"
        )
        deployment = _unique_positive(
            self.deployment_horizons_cycles, "deployment_horizons_cycles"
        )
        if training != tuple(sorted(training)):
            raise ValueError("training horizons must be strictly increasing")
        if deployment != tuple(sorted(deployment)):
            raise ValueError("deployment horizons must be strictly increasing")
        if not set(fresh) < set(training):
            raise ValueError("fresh horizons must be a proper subset of training horizons")
        if training[-1] * 2 != 64:
            raise ValueError("largest training horizon must preserve the 64-half-cycle parent")
        object.__setattr__(self, "training_horizons_cycles", training)
        object.__setattr__(self, "fresh_fit_horizons_cycles", fresh)
        object.__setattr__(self, "deployment_horizons_cycles", deployment)
        object.__setattr__(
            self, "restart_seeds", _unique_positive(self.restart_seeds, "restart_seeds")
        )
        object.__setattr__(
            self, "stream_seeds", _unique_positive(self.stream_seeds, "stream_seeds")
        )
        for name in (
            "trajectories_per_split",
            "training_epochs",
            "validation_interval",
            "sample_points_per_horizon",
            "reset_window_half_cycles",
            "reset_recovery_consecutive_points",
            "teacher_chunk_half_cycles",
            "maximum_reset_recovery_half_cycles",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if self.validation_interval > self.training_epochs:
            raise ValueError("validation_interval must not exceed training_epochs")
        if len({self.training_seed, self.validation_seed, self.evaluation_seed}) != 3:
            raise ValueError("training/validation/evaluation seeds must be disjoint")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "maximum_imitation_mse",
            "maximum_float32_action_error",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        state_l2 = float(self.state_l2_weight)
        if not np.isfinite(state_l2) or state_l2 < 0.0:
            raise ValueError("state_l2_weight must be finite and nonnegative")
        object.__setattr__(self, "state_l2_weight", state_l2)
        if self.reset_recovery_consecutive_points > self.reset_window_half_cycles + 1:
            raise ValueError("reset recovery run cannot exceed reset window")
        if self.maximum_reset_recovery_half_cycles > self.reset_window_half_cycles:
            raise ValueError("reset recovery threshold cannot exceed reset window")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")

    @property
    def contract_hash(self) -> str:
        return _canonical_sha256(asdict(self))


def validate_production_design(config: HorizonExtrapolationConfig) -> None:
    if config.training_horizons_cycles != (2, 5, 10, 32):
        raise ValueError("production must scan 2/5/10/32-cycle training horizons")
    if config.fresh_fit_horizons_cycles != (2, 5, 10):
        raise ValueError("production must freshly refit 2/5/10-cycle students")
    if config.deployment_horizons_cycles != (1_000, 100_000, 1_000_000):
        raise ValueError("production must execute 1e3/1e5/1e6 cycles")
    if config.trajectories_per_split < 256:
        raise ValueError("production requires at least 256 trajectories per split")
    if len(config.restart_seeds) < 3 or config.training_epochs < 800:
        raise ValueError("production requires >=3 restarts and >=800 epochs")
    if len(config.stream_seeds) < 2:
        raise ValueError("production requires at least two stochastic stream seeds")
    if config.sample_points_per_horizon < 4096:
        raise ValueError("production requires at least 4096 action checkpoints per horizon")
    if config.reset_window_half_cycles < 256:
        raise ValueError("production reset window must be at least 256 half-cycles")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _parent_bindings() -> list[dict[str, Any]]:
    rows = []
    for task_id, path in PARENT_ARTIFACTS.items():
        payload = _load_json(path)
        status = payload.get("status")
        passed = status == "PASS" or payload.get("passed") is True
        gate = payload.get("gate_summary", {})
        if isinstance(gate, Mapping):
            failed = gate.get("failed", gate.get("failed_names", 0))
            if isinstance(failed, Sequence) and not isinstance(failed, (str, bytes)):
                passed = passed and len(failed) == 0
            elif failed is not None:
                passed = passed and int(failed or 0) == 0
        rows.append(
            {
                "task_id": task_id,
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "machine_pass": bool(passed),
            }
        )
    return rows


def _implementation_bindings() -> list[dict[str, Any]]:
    return [
        {"path": path.as_posix(), "sha256": _sha256(path)}
        for path in IMPLEMENTATION_PATHS
    ]


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    th = _require_torch()
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    th.save(payload, temporary)
    os.replace(temporary, target)


SOURCE_COLUMNS = (
    "row_type",
    "lane",
    "model_id",
    "family",
    "seed",
    "training_horizon_cycles",
    "deployment_horizon_cycles",
    "metric",
    "value",
    "detail_json",
)


def _write_source_data(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in SOURCE_COLUMNS})
    os.replace(temporary, target)


def _make_outcome_streams(
    config: HorizonExtrapolationConfig,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    updates = 2 * config.deployment_horizons_cycles[-1]
    values: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for family in STREAM_FAMILIES:
        seeds = config.stream_seeds if family not in {"all_g_boundary", "all_e_boundary"} else (0,)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            if family == "stationary_nominal":
                sequence = (rng.random(updates) >= 0.82).astype(np.uint8)
            elif family == "persistent_regime":
                toggles = rng.random(updates) < 0.0015
                regime = np.bitwise_and(np.cumsum(toggles, dtype=np.int64), 1)
                p_ground = np.where(regime == 0, 0.91, 0.56)
                sequence = (rng.random(updates) >= p_ground).astype(np.uint8)
            elif family == "range_shift":
                fractions = np.arange(updates, dtype=np.int64) * 4 // updates
                p_ground = np.asarray((0.96, 0.44, 0.88, 0.61))[fractions]
                sequence = (rng.random(updates) >= p_ground).astype(np.uint8)
            elif family == "all_g_boundary":
                sequence = np.zeros(updates, dtype=np.uint8)
            elif family == "all_e_boundary":
                sequence = np.ones(updates, dtype=np.uint8)
            else:  # pragma: no cover - frozen registry above.
                raise AssertionError(family)
            stream_id = f"{family}-seed-{seed}"
            values.append(sequence)
            metadata.append(
                {
                    "stream_id": stream_id,
                    "family": family,
                    "seed": int(seed),
                    "updates_executed": int(updates),
                    "cycles_executed": int(updates // 2),
                    "ground_fraction": float(np.mean(sequence == 0)),
                    "outcome_sha256": _array_sha256(sequence),
                }
            )
    return np.stack(values, axis=0), metadata


def _performance_indices(config: HorizonExtrapolationConfig) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for horizon in config.deployment_horizons_cycles:
        updates = 2 * horizon
        count = min(config.sample_points_per_horizon, updates + 1)
        linear = np.linspace(0, updates, num=count, dtype=np.int64)
        logarithmic = np.unique(
            np.rint(
                np.geomspace(1.0, float(updates), num=min(count, updates))
            ).astype(np.int64)
        )
        result[horizon] = np.unique(np.concatenate(([0, updates], linear, logarithmic)))
    return result


def _sample_registry(
    config: HorizonExtrapolationConfig,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    performance = _performance_indices(config)
    reset_anchors: dict[int, int] = {}
    values = [array for array in performance.values()]
    for horizon in config.deployment_horizons_cycles:
        endpoint = 2 * horizon
        anchor = endpoint - config.reset_window_half_cycles
        if anchor <= 0:
            raise ValueError("deployment horizon is too short for reset window")
        reset_anchors[horizon] = anchor
        values.append(
            np.arange(anchor, endpoint + 1, dtype=np.int64)
        )
    union = np.unique(np.concatenate(values))
    positions = {
        horizon: np.searchsorted(union, indices)
        for horizon, indices in performance.items()
    }
    return union, positions, reset_anchors


def _copy_gru_cell_to_sequence(cell: Any, *, dtype: Any, device: str) -> Any:
    th = _require_torch()
    gru = th.nn.GRU(1, int(cell.hidden_size), batch_first=True).to(
        device=device, dtype=dtype
    )
    with th.no_grad():
        gru.weight_ih_l0.copy_(cell.weight_ih.detach().to(device=device, dtype=dtype))
        gru.weight_hh_l0.copy_(cell.weight_hh.detach().to(device=device, dtype=dtype))
        gru.bias_ih_l0.copy_(cell.bias_ih.detach().to(device=device, dtype=dtype))
        gru.bias_hh_l0.copy_(cell.bias_hh.detach().to(device=device, dtype=dtype))
    gru.eval()
    return gru


def _teacher_actions(model: Any, hidden: Any) -> Any:
    th = _require_torch()
    raw = model.output(th.tanh(model.dense2(th.tanh(model.dense1(hidden)))))
    bounds = th.tensor(
        RESIDUAL_BOUNDS, dtype=hidden.dtype, device=hidden.device
    )
    return bounds * th.tanh(raw)


def _teacher_long_scan(
    model: Any,
    outcomes: np.ndarray,
    sample_indices: np.ndarray,
    horizon_updates: Sequence[int],
    *,
    dtype: Any,
    device: str,
    chunk_half_cycles: int,
) -> dict[str, Any]:
    th = _require_torch()
    local_model = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
    gru = _copy_gru_cell_to_sequence(
        local_model.gru, dtype=dtype, device=device
    )
    batch, updates = outcomes.shape
    hidden_size = int(local_model.gru.hidden_size)
    samples = np.empty((batch, sample_indices.size, hidden_size), dtype=np.float64)
    samples[:, 0 if sample_indices[0] == 0 else np.searchsorted(sample_indices, 0), :] = 0.0
    hidden = th.zeros((1, batch, hidden_size), dtype=dtype, device=device)
    running_max = np.zeros(batch, dtype=np.float64)
    max_by_horizon: dict[int, list[float]] = {}
    finite_by_stream = np.ones(batch, dtype=bool)
    endpoint_set = set(int(value) for value in horizon_updates)
    start_time = time.perf_counter()
    start = 0
    with th.no_grad():
        while start < updates:
            next_endpoint = min((value for value in endpoint_set if value > start), default=updates)
            end = min(start + chunk_half_cycles, next_endpoint, updates)
            chunk = th.as_tensor(
                outcomes[:, start:end], dtype=dtype, device=device
            )
            chunk = 2.0 * chunk.unsqueeze(-1) - 1.0
            output, hidden = gru(chunk, hidden)
            finite_chunk = th.all(th.isfinite(output), dim=(1, 2)).cpu().numpy()
            finite_by_stream &= finite_chunk
            chunk_max = th.amax(th.abs(output), dim=(1, 2)).cpu().numpy()
            running_max = np.maximum(running_max, chunk_max)
            left = np.searchsorted(sample_indices, start + 1, side="left")
            right = np.searchsorted(sample_indices, end, side="right")
            if right > left:
                requested = sample_indices[left:right]
                local_indices = th.as_tensor(
                    requested - (start + 1), dtype=th.int64, device=device
                )
                samples[:, left:right, :] = (
                    output.index_select(1, local_indices).detach().cpu().double().numpy()
                )
            start = end
            if start in endpoint_set:
                max_by_horizon[int(start)] = running_max.tolist()
    sample_tensor = th.as_tensor(samples, dtype=dtype, device=device)
    action_parts = []
    with th.no_grad():
        flattened = sample_tensor.reshape(-1, hidden_size)
        for begin in range(0, flattened.shape[0], 8192):
            action_parts.append(
                _teacher_actions(local_model, flattened[begin : begin + 8192])
                .detach()
                .cpu()
                .double()
                .numpy()
            )
    actions = np.concatenate(action_parts, axis=0).reshape(
        batch, sample_indices.size, len(RESIDUAL_BOUNDS)
    )
    return {
        "hidden": samples,
        "actions": actions,
        "max_abs_hidden_by_horizon": max_by_horizon,
        "finite_by_stream": finite_by_stream.tolist(),
        "wall_time_seconds": time.perf_counter() - start_time,
        "updates_per_stream": int(updates),
    }


def _student_scan_impl(
    outcomes: np.ndarray,
    initial: np.ndarray,
    decays: np.ndarray,
    saturations: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    bounds: np.ndarray,
    sample_lookup: np.ndarray,
    sample_count: int,
    horizon_updates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    models, dimension = initial.shape
    batch, updates = outcomes.shape
    states = np.empty((models, batch, dimension), dtype=initial.dtype)
    for model_index in range(models):
        for stream_index in range(batch):
            for state_index in range(dimension):
                states[model_index, stream_index, state_index] = initial[
                    model_index, state_index
                ]
    sampled_states = np.empty(
        (models, batch, sample_count, dimension), dtype=initial.dtype
    )
    sampled_actions = np.empty(
        (models, batch, sample_count, bounds.size), dtype=initial.dtype
    )
    maxima = np.zeros((models, batch), dtype=initial.dtype)
    for model_index in range(models):
        for stream_index in range(batch):
            for state_index in range(dimension):
                absolute = abs(states[model_index, stream_index, state_index])
                if absolute > maxima[model_index, stream_index]:
                    maxima[model_index, stream_index] = absolute
    maxima_by_horizon = np.empty(
        (models, batch, horizon_updates.size), dtype=initial.dtype
    )
    horizon_slot = 0
    for step in range(updates + 1):
        sample_slot = sample_lookup[step]
        if sample_slot >= 0:
            for model_index in range(models):
                for stream_index in range(batch):
                    for state_index in range(dimension):
                        sampled_states[
                            model_index, stream_index, sample_slot, state_index
                        ] = states[model_index, stream_index, state_index]
                    for output_index in range(bounds.size):
                        raw = bias[model_index, output_index]
                        for state_index in range(dimension):
                            raw += (
                                weights[model_index, output_index, state_index]
                                * states[model_index, stream_index, state_index]
                            )
                        sampled_actions[
                            model_index, stream_index, sample_slot, output_index
                        ] = bounds[output_index] * np.tanh(raw)
        if step == updates:
            break
        for model_index in range(models):
            for stream_index in range(batch):
                outcome = outcomes[stream_index, step]
                for state_index in range(dimension):
                    value = (
                        decays[model_index, outcome, state_index]
                        * states[model_index, stream_index, state_index]
                        + (1.0 - decays[model_index, outcome, state_index])
                        * saturations[model_index, outcome, state_index]
                    )
                    states[model_index, stream_index, state_index] = value
                    absolute = abs(value)
                    if absolute > maxima[model_index, stream_index]:
                        maxima[model_index, stream_index] = absolute
        if horizon_slot < horizon_updates.size and step + 1 == horizon_updates[horizon_slot]:
            for model_index in range(models):
                for stream_index in range(batch):
                    maxima_by_horizon[model_index, stream_index, horizon_slot] = maxima[
                        model_index, stream_index
                    ]
            horizon_slot += 1
    return sampled_states, sampled_actions, maxima_by_horizon, states


if numba is not None:
    _student_scan_compiled = numba.njit(cache=True)(_student_scan_impl)
else:  # pragma: no cover
    _student_scan_compiled = _student_scan_impl


def _student_long_scan(
    outcomes: np.ndarray,
    parameter_sets: Sequence[Mapping[str, np.ndarray]],
    sample_indices: np.ndarray,
    horizon_updates: Sequence[int],
    *,
    dtype: Any,
) -> dict[str, Any]:
    initial = np.stack([row["initial_state"] for row in parameter_sets]).astype(dtype)
    decays = np.stack([row["outcome_decays"] for row in parameter_sets]).astype(dtype)
    saturations = np.stack([row["outcome_saturations"] for row in parameter_sets]).astype(dtype)
    weights = np.stack([row["output_weights"] for row in parameter_sets]).astype(dtype)
    bias = np.stack([row["output_bias"] for row in parameter_sets]).astype(dtype)
    bounds = np.asarray(RESIDUAL_BOUNDS, dtype=dtype)
    lookup = np.full(outcomes.shape[1] + 1, -1, dtype=np.int64)
    lookup[sample_indices] = np.arange(sample_indices.size, dtype=np.int64)
    started = time.perf_counter()
    sampled_states, sampled_actions, maxima, final_states = _student_scan_compiled(
        outcomes,
        initial,
        decays,
        saturations,
        weights,
        bias,
        bounds,
        lookup,
        int(sample_indices.size),
        np.asarray(horizon_updates, dtype=np.int64),
    )
    return {
        "states": np.asarray(sampled_states, dtype=np.float64),
        "actions": np.asarray(sampled_actions, dtype=np.float64),
        "max_abs_state_by_horizon": np.asarray(maxima, dtype=np.float64),
        "final_states": np.asarray(final_states, dtype=np.float64),
        "all_finite": bool(
            np.all(np.isfinite(sampled_states))
            and np.all(np.isfinite(sampled_actions))
            and np.all(np.isfinite(maxima))
            and np.all(np.isfinite(final_states))
        ),
        "wall_time_seconds": time.perf_counter() - started,
        "updates_per_stream": int(outcomes.shape[1]),
    }


def _parameters_from_model(model: Any) -> dict[str, np.ndarray]:
    return student_parent._export_parameters(model)


def _parameters_from_artifact(
    artifact: LowDimensionalRecurrenceArtifact,
) -> dict[str, np.ndarray]:
    return {
        "initial_state": np.asarray(artifact.initial_state, dtype=np.float64),
        "outcome_decays": np.asarray(artifact.outcome_decays, dtype=np.float64),
        "outcome_saturations": np.asarray(
            artifact.outcome_saturations, dtype=np.float64
        ),
        "output_weights": np.asarray(artifact.output_weights, dtype=np.float64),
        "output_bias": np.asarray(artifact.output_bias, dtype=np.float64),
    }


def _training_sweep(
    config: HorizonExtrapolationConfig,
    teacher: Any,
) -> tuple[dict[str, Any], list[dict[str, np.ndarray]], dict[str, Any]]:
    max_half_cycles = 2 * config.training_horizons_cycles[-1]
    outcomes = {
        "training": student_parent._outcome_dataset(
            config.training_seed, config.trajectories_per_split, max_half_cycles
        ),
        "validation": student_parent._outcome_dataset(
            config.validation_seed, config.trajectories_per_split, max_half_cycles
        ),
        "evaluation": student_parent._outcome_dataset(
            config.evaluation_seed, config.trajectories_per_split, max_half_cycles
        ),
    }
    targets = {
        split: trace_teacher_hidden(teacher, values)["physical_residual"]
        for split, values in outcomes.items()
    }
    base = student_parent.LowDimensionalDistillationConfig(
        half_cycles=max_half_cycles,
        trajectories_per_split=config.trajectories_per_split,
        training_seed=config.training_seed,
        validation_seed=config.validation_seed,
        evaluation_seed=config.evaluation_seed,
        restart_seeds=config.restart_seeds,
        training_epochs=config.training_epochs,
        validation_interval=config.validation_interval,
        learning_rate=config.learning_rate,
        gradient_clip_norm=config.gradient_clip_norm,
        state_l2_weight=config.state_l2_weight,
        device=config.device,
    )
    records: list[dict[str, Any]] = []
    selected: dict[int, dict[str, Any]] = {}
    checkpoint_models: dict[str, Any] = {}
    long_parameters: list[dict[str, np.ndarray]] = []
    long_model_ids: list[str] = []
    for horizon in config.fresh_fit_horizons_cycles:
        half_cycles = 2 * horizon
        horizon_records = []
        horizon_models = []
        for restart_index, restart_seed in enumerate(config.restart_seeds):
            model, record = student_parent._fit_candidate(
                4,
                restart_index,
                restart_seed,
                base,
                outcomes["training"][:, :half_cycles],
                targets["training"][:, : half_cycles + 1],
                outcomes["validation"][:, :half_cycles],
                targets["validation"][:, : half_cycles + 1],
            )
            prediction = student_parent._predict_model(
                model, outcomes["evaluation"], config.device
            )
            evaluation = student_parent._metrics(targets["evaluation"], prediction)
            summarized = {
                key: value
                for key, value in record.items()
                if key not in {"training_curve", "validation_history"}
            }
            summarized.update(
                {
                    "training_horizon_cycles": int(horizon),
                    "training_half_cycles": int(half_cycles),
                    "selection_split": "validation_only",
                    "evaluation_opened_after_selection": False,
                    "evaluation_32_cycle_metrics": evaluation,
                }
            )
            horizon_records.append(summarized)
            horizon_models.append(model)
            checkpoint_models[f"h{horizon}_r{restart_index}"] = {
                "record": record,
                "state_dict": student_parent._state_dict_cpu(model),
            }
        selected_index = min(
            range(len(horizon_records)),
            key=lambda index: horizon_records[index]["best_validation_mse"],
        )
        for index, row in enumerate(horizon_records):
            row["selected_by_validation"] = index == selected_index
            row["evaluation_opened_after_selection"] = True
        selected_model = horizon_models[selected_index]
        selected_row = horizon_records[selected_index]
        selected[horizon] = {
            "model_id": f"fresh_h{horizon}_student",
            "training_horizon_cycles": int(horizon),
            "selected_restart": int(selected_row["restart_index"]),
            "selected_restart_seed": int(selected_row["restart_seed"]),
            "selection_metric": "validation_teacher_action_mse",
            "selection_value": float(selected_row["best_validation_mse"]),
            "evaluation_32_cycle_metrics": selected_row[
                "evaluation_32_cycle_metrics"
            ],
            "evaluation_used_for_selection": False,
        }
        long_parameters.append(_parameters_from_model(selected_model))
        long_model_ids.append(f"fresh_h{horizon}_student")
        records.extend(horizon_records)

    parent_report = _load_json(student_parent.DEFAULT_ARTIFACT)
    production_artifact = LowDimensionalRecurrenceArtifact.from_dict(
        _load_json(PRODUCTION_STUDENT)
    )
    parent_selection = parent_report["selection"]
    production_horizon = config.training_horizons_cycles[-1]
    selected[production_horizon] = {
        "model_id": "production_h32_student",
        "training_horizon_cycles": int(production_horizon),
        "selected_dimension": int(parent_selection["selected_dimension"]),
        "selected_restart": int(parent_selection["selected_restart"]),
        "selection_metric": "validation_teacher_action_mse",
        "selection_value": float(production_artifact.validation_mse),
        "evaluation_32_cycle_metrics": parent_report["comparisons"]["evaluation"][
            "selected_student"
        ],
        "evaluation_used_for_selection": False,
        "source": "frozen_T4.4.3_strict_split_production_student",
    }
    long_parameters.append(_parameters_from_artifact(production_artifact))
    long_model_ids.append("production_h32_student")
    report = {
        "dataset": {
            "max_training_half_cycles": int(max_half_cycles),
            "trajectories_per_split": int(config.trajectories_per_split),
            "split_seeds": {
                "training": config.training_seed,
                "validation": config.validation_seed,
                "evaluation": config.evaluation_seed,
            },
            "outcome_hashes": {
                split: _array_sha256(value) for split, value in outcomes.items()
            },
            "target_hashes": {
                split: _array_sha256(value) for split, value in targets.items()
            },
            "prefix_rule": (
                "all shorter horizons use exact prefixes of the same split-specific "
                "64-half-cycle histories and frozen teacher targets"
            ),
        },
        "candidate_records": records,
        "selected_by_horizon": {str(key): value for key, value in selected.items()},
        "long_model_ids": long_model_ids,
        "evaluation_never_used_for_selection": True,
        "fresh_candidate_count": len(records),
    }
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "config": asdict(config),
        "config_contract_hash": config.contract_hash,
        "parent_teacher_checkpoint_sha256": _sha256(TEACHER_CHECKPOINT),
        "fresh_models": checkpoint_models,
        "selected_by_horizon": report["selected_by_horizon"],
    }
    return report, long_parameters, checkpoint


def _performance_rows(
    teacher_actions: np.ndarray,
    student_actions: np.ndarray,
    model_ids: Sequence[str],
    streams: Sequence[Mapping[str, Any]],
    positions: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for model_index, model_id in enumerate(model_ids):
        for stream_index, stream in enumerate(streams):
            for horizon, selected_positions in positions.items():
                target = teacher_actions[stream_index, selected_positions]
                prediction = student_actions[
                    model_index, stream_index, selected_positions
                ]
                error = prediction - target
                rows.append(
                    {
                        "model_id": model_id,
                        "family": stream["family"],
                        "seed": stream["seed"],
                        "stream_id": stream["stream_id"],
                        "deployment_horizon_cycles": int(horizon),
                        "sample_count": int(selected_positions.size),
                        "mse": float(np.mean(error**2)),
                        "mae": float(np.mean(np.abs(error))),
                        "maximum_absolute_error": float(np.max(np.abs(error))),
                        "teacher_residual_rms": float(np.sqrt(np.mean(target**2))),
                        "student_residual_rms": float(np.sqrt(np.mean(prediction**2))),
                    }
                )
    return rows


def _aggregate_performance(
    rows: Sequence[Mapping[str, Any]],
    model_ids: Sequence[str],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    result = []
    for model_id in model_ids:
        for horizon in horizons:
            selected = [
                row
                for row in rows
                if row["model_id"] == model_id
                and int(row["deployment_horizon_cycles"]) == horizon
            ]
            mse = np.asarray([float(row["mse"]) for row in selected])
            result.append(
                {
                    "model_id": model_id,
                    "deployment_horizon_cycles": int(horizon),
                    "stream_count": len(selected),
                    "mean_stream_mse": float(np.mean(mse)),
                    "median_stream_mse": float(np.median(mse)),
                    "worst_stream_mse": float(np.max(mse)),
                    "worst_stream_id": selected[int(np.argmax(mse))]["stream_id"],
                }
            )
    return result


def _first_stable_recovery(
    values: np.ndarray, threshold: float, consecutive: int
) -> int | None:
    below = np.asarray(values) <= float(threshold)
    for index in range(0, below.size - consecutive + 1):
        if np.all(below[index : index + consecutive]):
            return int(index)
    return None


def _teacher_reset_actions(model: Any, outcomes: np.ndarray) -> np.ndarray:
    th = _require_torch()
    reference = next(model.parameters())
    batch, updates = outcomes.shape
    hidden = th.zeros(
        (batch, model.gru.hidden_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    rows = []
    with th.no_grad():
        for step in range(updates + 1):
            rows.append(_teacher_actions(model, hidden).detach().cpu().numpy())
            if step < updates:
                token = th.as_tensor(
                    outcomes[:, step : step + 1],
                    dtype=reference.dtype,
                    device=reference.device,
                )
                hidden = model.gru(2.0 * token - 1.0, hidden)
    return np.stack(rows, axis=1)


def _student_reset_actions(
    parameters: Mapping[str, np.ndarray], outcomes: np.ndarray
) -> np.ndarray:
    state = np.broadcast_to(
        parameters["initial_state"],
        (outcomes.shape[0], parameters["initial_state"].size),
    ).copy()
    rows = []
    bounds = np.asarray(RESIDUAL_BOUNDS, dtype=np.float64)
    for step in range(outcomes.shape[1] + 1):
        rows.append(
            bounds
            * np.tanh(state @ parameters["output_weights"].T + parameters["output_bias"])
        )
        if step < outcomes.shape[1]:
            index = outcomes[:, step]
            state = parameters["outcome_decays"][index] * state + (
                1.0 - parameters["outcome_decays"][index]
            ) * parameters["outcome_saturations"][index]
    return np.stack(rows, axis=1)


def _reset_rows(
    model: Any,
    outcomes: np.ndarray,
    teacher_actions: np.ndarray,
    student_actions: np.ndarray,
    parameter_sets: Sequence[Mapping[str, np.ndarray]],
    model_ids: Sequence[str],
    streams: Sequence[Mapping[str, Any]],
    sample_indices: np.ndarray,
    reset_anchors: Mapping[int, int],
    config: HorizonExtrapolationConfig,
) -> list[dict[str, Any]]:
    rows = []
    consecutive = config.reset_recovery_consecutive_points
    for horizon, anchor in reset_anchors.items():
        endpoint = anchor + config.reset_window_half_cycles
        window_outcomes = outcomes[:, anchor:endpoint]
        positions = np.searchsorted(
            sample_indices, np.arange(anchor, endpoint + 1, dtype=np.int64)
        )
        reset_teacher = _teacher_reset_actions(model, window_outcomes)
        for stream_index, stream in enumerate(streams):
            baseline = teacher_actions[stream_index, positions]
            error = np.sqrt(
                np.mean((reset_teacher[stream_index] - baseline) ** 2, axis=1)
            )
            threshold = max(1.0e-4, 0.05 * float(np.sqrt(np.mean(baseline**2))))
            recovery = _first_stable_recovery(error, threshold, consecutive)
            rows.append(
                {
                    "model_id": "teacher_gru10",
                    "family": stream["family"],
                    "seed": stream["seed"],
                    "stream_id": stream["stream_id"],
                    "deployment_horizon_cycles": int(horizon),
                    "reset_anchor_half_cycle": int(anchor),
                    "threshold": float(threshold),
                    "immediate_action_rmse": float(error[0]),
                    "terminal_action_rmse": float(error[-1]),
                    "integrated_action_rmse": float(np.mean(error)),
                    "recovery_half_cycles": recovery,
                    "recovered_within_window": recovery is not None,
                }
            )
        for model_index, (model_id, parameters) in enumerate(
            zip(model_ids, parameter_sets, strict=True)
        ):
            reset_student = _student_reset_actions(parameters, window_outcomes)
            for stream_index, stream in enumerate(streams):
                baseline = student_actions[model_index, stream_index, positions]
                error = np.sqrt(
                    np.mean((reset_student[stream_index] - baseline) ** 2, axis=1)
                )
                threshold = max(
                    1.0e-4, 0.05 * float(np.sqrt(np.mean(baseline**2)))
                )
                recovery = _first_stable_recovery(error, threshold, consecutive)
                rows.append(
                    {
                        "model_id": model_id,
                        "family": stream["family"],
                        "seed": stream["seed"],
                        "stream_id": stream["stream_id"],
                        "deployment_horizon_cycles": int(horizon),
                        "reset_anchor_half_cycle": int(anchor),
                        "threshold": float(threshold),
                        "immediate_action_rmse": float(error[0]),
                        "terminal_action_rmse": float(error[-1]),
                        "integrated_action_rmse": float(np.mean(error)),
                        "recovery_half_cycles": recovery,
                        "recovered_within_window": recovery is not None,
                    }
                )
    return rows


def _stability_rows(
    teacher64: Mapping[str, Any],
    student64: Mapping[str, Any],
    parameter_sets: Sequence[Mapping[str, np.ndarray]],
    model_ids: Sequence[str],
    streams: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    positions: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for horizon_index, horizon in enumerate(horizons):
        endpoint = 2 * horizon
        teacher_values = teacher64["max_abs_hidden_by_horizon"][endpoint]
        for stream_index, stream in enumerate(streams):
            rows.append(
                {
                    "model_id": "teacher_gru10",
                    "family": stream["family"],
                    "seed": stream["seed"],
                    "stream_id": stream["stream_id"],
                    "deployment_horizon_cycles": int(horizon),
                    "maximum_absolute_state": float(teacher_values[stream_index]),
                    "analytic_bound": 1.0,
                    "maximum_normalized_sampled_action": float(
                        np.max(
                            np.abs(
                                teacher64["actions"][
                                    stream_index, positions[horizon]
                                ]
                            )
                            / np.asarray(RESIDUAL_BOUNDS)
                        )
                    ),
                    "finite": bool(teacher64["finite_by_stream"][stream_index]),
                }
            )
        for model_index, model_id in enumerate(model_ids):
            analytic_bound = float(
                np.max(
                    np.abs(
                        np.concatenate(
                            (
                                parameter_sets[model_index]["initial_state"].reshape(-1),
                                parameter_sets[model_index]["outcome_saturations"].reshape(-1),
                            )
                        )
                    )
                )
            )
            for stream_index, stream in enumerate(streams):
                rows.append(
                    {
                        "model_id": model_id,
                        "family": stream["family"],
                        "seed": stream["seed"],
                        "stream_id": stream["stream_id"],
                        "deployment_horizon_cycles": int(horizon),
                        "maximum_absolute_state": float(
                            student64["max_abs_state_by_horizon"][
                                model_index, stream_index, horizon_index
                            ]
                        ),
                        "analytic_bound": analytic_bound,
                        "maximum_normalized_sampled_action": float(
                            np.max(
                                np.abs(
                                    student64["actions"][
                                        model_index,
                                        stream_index,
                                        positions[horizon],
                                    ]
                                )
                                / np.asarray(RESIDUAL_BOUNDS)
                            )
                        ),
                        "finite": bool(student64["all_finite"]),
                    }
                )
    return rows


def _numeric_rows(
    teacher64: Mapping[str, Any],
    teacher32: Mapping[str, Any],
    student64: Mapping[str, Any],
    student32: Mapping[str, Any],
    model_ids: Sequence[str],
    streams: Sequence[Mapping[str, Any]],
    positions: Mapping[int, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for stream_index, stream in enumerate(streams):
        for horizon, selected in positions.items():
            teacher_action_error = np.max(
                np.abs(
                    teacher64["actions"][stream_index, selected]
                    - teacher32["actions"][stream_index, selected]
                )
            )
            teacher_state_error = np.max(
                np.abs(
                    teacher64["hidden"][stream_index, selected]
                    - teacher32["hidden"][stream_index, selected]
                )
            )
            rows.append(
                {
                    "model_id": "teacher_gru10",
                    "family": stream["family"],
                    "seed": stream["seed"],
                    "stream_id": stream["stream_id"],
                    "deployment_horizon_cycles": int(horizon),
                    "maximum_float32_float64_state_error": float(teacher_state_error),
                    "maximum_float32_float64_action_error": float(teacher_action_error),
                }
            )
            for model_index, model_id in enumerate(model_ids):
                state_error = np.max(
                    np.abs(
                        student64["states"][model_index, stream_index, selected]
                        - student32["states"][model_index, stream_index, selected]
                    )
                )
                action_error = np.max(
                    np.abs(
                        student64["actions"][model_index, stream_index, selected]
                        - student32["actions"][model_index, stream_index, selected]
                    )
                )
                rows.append(
                    {
                        "model_id": model_id,
                        "family": stream["family"],
                        "seed": stream["seed"],
                        "stream_id": stream["stream_id"],
                        "deployment_horizon_cycles": int(horizon),
                        "maximum_float32_float64_state_error": float(state_error),
                        "maximum_float32_float64_action_error": float(action_error),
                    }
                )
    return rows


def _contract_view(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "task_id",
        "protocol_id",
        "status",
        "verdict",
        "config",
        "parent_bindings",
        "implementation_bindings",
        "training_sweep",
        "stream_registry",
        "sampling_contract",
        "execution_summary",
        "performance_rows",
        "performance_aggregate",
        "stability_rows",
        "numeric_rows",
        "reset_rows",
        "claim_boundary",
        "gates",
        "gate_summary",
        "checkpoint",
        "source_data",
    )
    return {key: report[key] for key in keys}


def _compute_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    config = report["config"]
    training = report["training_sweep"]
    streams = report["stream_registry"]
    performance = report["performance_aggregate"]
    stability = report["stability_rows"]
    numeric = report["numeric_rows"]
    resets = report["reset_rows"]
    parent_selection = training["selected_by_horizon"]
    required_horizons = {2, 5, 10, 32}
    production_models = {"fresh_h10_student", "production_h32_student"}
    production_performance = [
        row for row in performance if row["model_id"] in production_models
    ]
    maximum_imitation = float(config["maximum_imitation_mse"])
    maximum_numeric = float(config["maximum_float32_action_error"])
    maximum_recovery = int(config["maximum_reset_recovery_half_cycles"])
    expected_streams = 3 * len(config["stream_seeds"]) + 2
    parent32 = parent_selection.get("32", {})
    branch_parent = _load_json(PARENT_ARTIFACTS["T4.4.5"])
    return {
        "all_parent_artifacts_are_live_machine_passes": all(
            row["machine_pass"] for row in report["parent_bindings"]
        ),
        "training_horizon_registry_is_exact_2_5_10_32": set(
            int(key) for key in parent_selection
        )
        == required_horizons,
        "fresh_short_horizons_have_three_restarts_each": training[
            "fresh_candidate_count"
        ]
        >= 9
        and all(
            sum(
                int(row["training_horizon_cycles"]) == horizon
                for row in training["candidate_records"]
            )
            >= 3
            for horizon in (2, 5, 10)
        ),
        "all_horizon_selection_is_validation_only": training[
            "evaluation_never_used_for_selection"
        ]
        and all(
            not row["evaluation_used_for_selection"]
            for row in parent_selection.values()
        ),
        "production_32_cycle_student_is_frozen_parent": parent32.get("source")
        == "frozen_T4.4.3_strict_split_production_student",
        "all_registered_streams_execute_full_two_million_updates": len(streams)
        == expected_streams
        and all(
            row["updates_executed"]
            == 2 * max(config["deployment_horizons_cycles"])
            for row in streams
        ),
        "three_stochastic_and_two_boundary_families_are_present": set(
            row["family"] for row in streams
        )
        == set(STREAM_FAMILIES),
        "all_three_deployment_horizons_are_reported": set(
            int(row["deployment_horizon_cycles"]) for row in performance
        )
        == {1_000, 100_000, 1_000_000},
        "teacher_hidden_is_finite_and_inside_gru_tanh_bound": all(
            row["finite"]
            and (
                row["model_id"] != "teacher_gru10"
                or float(row["maximum_absolute_state"]) <= 1.0 + 1.0e-12
            )
            for row in stability
        ),
        "all_student_states_remain_inside_affine_convex_hull_bound": all(
            row["finite"]
            and float(row["maximum_absolute_state"])
            <= float(row["analytic_bound"]) + 1.0e-12
            for row in stability
            if row["model_id"] != "teacher_gru10"
        ),
        "physical_residual_action_box_is_analytic_and_observed": report[
            "claim_boundary"
        ]["residual_bounds"]
        == list(RESIDUAL_BOUNDS)
        and all(
            float(row["maximum_normalized_sampled_action"]) <= 1.0 + 1.0e-12
            for row in stability
        ),
        "float32_shadow_actions_remain_within_threshold": all(
            float(row["maximum_float32_float64_action_error"]) <= maximum_numeric
            for row in numeric
        ),
        "ten_and_32_cycle_students_retain_sampled_teacher_actions_even_on_worst_stream": all(
            float(row["mean_stream_mse"]) <= maximum_imitation
            and float(row["worst_stream_mse"]) <= maximum_imitation
            for row in production_performance
        ),
        "worst_stream_performance_is_not_hidden": all(
            bool(row["worst_stream_id"]) and row["worst_stream_mse"] >= row["mean_stream_mse"]
            for row in performance
        ),
        "reset_interventions_cover_every_model_stream_and_horizon": len(resets)
        == (1 + len(training["long_model_ids"]))
        * len(streams)
        * len(config["deployment_horizons_cycles"]),
        "all_reset_runs_recover_within_preregistered_window": all(
            row["recovered_within_window"]
            and int(row["recovery_half_cycles"]) <= maximum_recovery
            for row in resets
        ),
        "reset_terminal_error_is_reported_not_assumed_zero": all(
            np.isfinite(float(row["terminal_action_rmse"])) for row in resets
        ),
        "qualified_parent_branch_was_not_silently_reselected": branch_parent[
            "active_branch"
        ]["branch_id"]
        == "qualified_student_retention",
        "long_horizon_physical_gain_remains_not_established": report[
            "claim_boundary"
        ]["long_horizon_physical_gain_established"]
        is False,
        "physical_memory_leakage_device_and_hardware_claims_remain_closed": all(
            report["claim_boundary"][key] is False
            for key in (
                "physical_memory_ler_established",
                "leakage_robustness_established",
                "device_calibrated",
                "hardware_measured",
            )
        ),
        "source_data_and_checkpoint_are_byte_bound": bool(
            report["source_data"].get("csv_sha256")
            and report["checkpoint"].get("sha256")
        ),
    }


def validate_artifact(
    report: Mapping[str, Any], *, check_files: bool = True
) -> tuple[str, ...]:
    errors: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION or report.get("task_id") != TASK_ID:
        errors.append("schema/task mismatch")
    try:
        recomputed = _compute_gates(report)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"gate recomputation failed: {exc}")
        recomputed = {}
    if recomputed != report.get("gates"):
        errors.append("stored gates differ from semantic recomputation")
    expected_summary = {
        "passed": sum(bool(value) for value in recomputed.values()),
        "total": len(recomputed),
    }
    if report.get("gate_summary") != expected_summary:
        errors.append("gate summary mismatch")
    expected_status = "PASS" if recomputed and all(recomputed.values()) else "FAIL"
    if report.get("status") != expected_status:
        errors.append("status does not match gates")
    if report.get("contract_sha256") != _canonical_sha256(_contract_view(report)):
        errors.append("contract hash mismatch")
    if check_files:
        for row in report.get("parent_bindings", ()):
            if not _repo_path(row["path"]).is_file() or row["sha256"] != _sha256(row["path"]):
                errors.append(f"parent binding mismatch: {row.get('task_id')}")
        for row in report.get("implementation_bindings", ()):
            if not _repo_path(row["path"]).is_file() or row["sha256"] != _sha256(row["path"]):
                errors.append(f"implementation binding mismatch: {row.get('path')}")
        for key in ("checkpoint", "source_data"):
            binding = report.get(key, {})
            if not _repo_path(binding.get("path", "")).is_file():
                errors.append(f"{key} file missing")
            elif binding.get("sha256", binding.get("csv_sha256")) != _sha256(
                binding["path"]
            ):
                errors.append(f"{key} hash mismatch")
    return tuple(errors)


def source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append(
        row_type: str,
        lane: str,
        model_id: str,
        family: str,
        seed: Any,
        training_horizon: Any,
        deployment_horizon: Any,
        metric: str,
        value: Any,
        detail: Any,
    ) -> None:
        rows.append(
            {
                "row_type": row_type,
                "lane": lane,
                "model_id": model_id,
                "family": family,
                "seed": seed,
                "training_horizon_cycles": training_horizon,
                "deployment_horizon_cycles": deployment_horizon,
                "metric": metric,
                "value": value,
                "detail_json": json.dumps(detail, sort_keys=True),
            }
        )

    for row in report["parent_bindings"]:
        append("parent_binding", "provenance", row["task_id"], "", "", "", "", "machine_pass", int(row["machine_pass"]), row)
    for row in report["implementation_bindings"]:
        append("implementation_binding", "provenance", row["path"], "", "", "", "", "sha256_bound", 1, row)
    for row in report["stream_registry"]:
        append("stream", "long_recurrence", "observed_outcome_stream", row["family"], row["seed"], "", report["config"]["deployment_horizons_cycles"][-1], "updates_executed", row["updates_executed"], row)
    for row in report["training_sweep"]["candidate_records"]:
        append("training_candidate", "training_horizon_sweep", f"h{row['training_horizon_cycles']}_r{row['restart_index']}", "", row["restart_seed"], row["training_horizon_cycles"], 32, "evaluation_32_cycle_mse", row["evaluation_32_cycle_metrics"]["mse"], row)
    for horizon, row in report["training_sweep"]["selected_by_horizon"].items():
        append("horizon_selection", "training_horizon_sweep", row["model_id"], "", row.get("selected_restart_seed", ""), horizon, 32, "validation_mse", row["selection_value"], row)
    for row in report["performance_rows"]:
        append("stream_performance", "sampled_action_imitation", row["model_id"], row["family"], row["seed"], "", row["deployment_horizon_cycles"], "teacher_action_mse", row["mse"], row)
    for row in report["performance_aggregate"]:
        append("aggregate_performance", "sampled_action_imitation", row["model_id"], "all", "", "", row["deployment_horizon_cycles"], "mean_stream_mse", row["mean_stream_mse"], row)
    for row in report["stability_rows"]:
        append("state_stability", "all_step_state_scan", row["model_id"], row["family"], row["seed"], "", row["deployment_horizon_cycles"], "maximum_absolute_state", row["maximum_absolute_state"], row)
    for row in report["numeric_rows"]:
        append("numeric_shadow", "float32_vs_float64", row["model_id"], row["family"], row["seed"], "", row["deployment_horizon_cycles"], "maximum_action_error", row["maximum_float32_float64_action_error"], row)
    for row in report["reset_rows"]:
        append("reset_intervention", "state_reset_sensitivity", row["model_id"], row["family"], row["seed"], "", row["deployment_horizon_cycles"], "recovery_half_cycles", row["recovery_half_cycles"], row)
    for name, value in report["gates"].items():
        append("gate", "evidence_gate", name, "", "", "", "", "passed", int(value), {"gate": name, "passed": value})
    return rows


def run_horizon_extrapolation_validation(
    config: HorizonExtrapolationConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
    production: bool = True,
) -> dict[str, Any]:
    th = _require_torch()
    actual = config or HorizonExtrapolationConfig()
    if production:
        validate_production_design(actual)
    if actual.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    teacher, teacher_report = load_and_verify_teacher_checkpoint(
        TEACHER_CHECKPOINT, TEACHER_ARTIFACT
    )
    training_sweep, parameter_sets, checkpoint = _training_sweep(actual, teacher)
    _atomic_torch_save(checkpoint, Path(checkpoint_path))
    outcomes, streams = _make_outcome_streams(actual)
    sample_indices, performance_positions, reset_anchors = _sample_registry(actual)
    horizon_updates = tuple(2 * value for value in actual.deployment_horizons_cycles)
    teacher64 = _teacher_long_scan(
        teacher,
        outcomes,
        sample_indices,
        horizon_updates,
        dtype=th.float64,
        device=actual.device,
        chunk_half_cycles=actual.teacher_chunk_half_cycles,
    )
    teacher32 = _teacher_long_scan(
        teacher,
        outcomes,
        sample_indices,
        horizon_updates,
        dtype=th.float32,
        device=actual.device,
        chunk_half_cycles=actual.teacher_chunk_half_cycles,
    )
    student64 = _student_long_scan(
        outcomes,
        parameter_sets,
        sample_indices,
        horizon_updates,
        dtype=np.float64,
    )
    student32 = _student_long_scan(
        outcomes,
        parameter_sets,
        sample_indices,
        horizon_updates,
        dtype=np.float32,
    )
    model_ids = training_sweep["long_model_ids"]
    performance_rows = _performance_rows(
        teacher64["actions"],
        student64["actions"],
        model_ids,
        streams,
        performance_positions,
    )
    performance_aggregate = _aggregate_performance(
        performance_rows, model_ids, actual.deployment_horizons_cycles
    )
    stability_rows = _stability_rows(
        teacher64,
        student64,
        parameter_sets,
        model_ids,
        streams,
        actual.deployment_horizons_cycles,
        performance_positions,
    )
    numeric_rows = _numeric_rows(
        teacher64,
        teacher32,
        student64,
        student32,
        model_ids,
        streams,
        performance_positions,
    )
    reset_rows = _reset_rows(
        teacher,
        outcomes,
        teacher64["actions"],
        student64["actions"],
        parameter_sets,
        model_ids,
        streams,
        sample_indices,
        reset_anchors,
        actual,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PENDING",
        "verdict": "PENDING",
        "config": asdict(actual),
        "config_contract_hash": actual.contract_hash,
        "parent_bindings": _parent_bindings(),
        "implementation_bindings": _implementation_bindings(),
        "training_sweep": training_sweep,
        "stream_registry": streams,
        "sampling_contract": {
            "sample_index_count": int(sample_indices.size),
            "sample_indices_sha256": _array_sha256(sample_indices),
            "action_sampling": (
                "union of linear/log checkpoints at each deployment horizon plus "
                "every point in each reset window"
            ),
            "state_update_sampling": "none; every registered half-cycle is executed",
            "reset_anchors_half_cycles": {
                str(key): int(value) for key, value in reset_anchors.items()
            },
        },
        "execution_summary": {
            "stream_count": len(streams),
            "updates_per_stream": int(outcomes.shape[1]),
            "total_teacher_updates_per_precision": int(outcomes.size),
            "total_student_updates_per_precision": int(
                outcomes.size * len(model_ids)
            ),
            "teacher_float64_seconds": teacher64["wall_time_seconds"],
            "teacher_float32_seconds": teacher32["wall_time_seconds"],
            "student_float64_seconds": student64["wall_time_seconds"],
            "student_float32_seconds": student32["wall_time_seconds"],
            "teacher_architecture": "GRU10-DENSE256-DENSE256-OUT15",
            "student_architecture": "4-state outcome-specific affine recurrence",
            "teacher_gru_sequence_equivalence_checked_by_tests": True,
        },
        "performance_rows": performance_rows,
        "performance_aggregate": performance_aggregate,
        "stability_rows": stability_rows,
        "numeric_rows": numeric_rows,
        "reset_rows": reset_rows,
        "claim_boundary": {
            "allowed": (
                "training-horizon-conditioned teacher-action imitation, exact recurrent "
                "state boundedness, float32 shadow stability, and controlled state-reset "
                "recovery on registered observed g/e streams"
            ),
            "forbidden": (
                "long-horizon physical or logical lifetime/gain, physical-memory LER, "
                "leakage robustness, universal memory benefit, device calibration, RTL, "
                "FPGA, board, or experiment"
            ),
            "residual_bounds": list(RESIDUAL_BOUNDS),
            "long_horizon_physical_gain_established": False,
            "physical_memory_ler_established": False,
            "leakage_robustness_established": False,
            "device_calibrated": False,
            "hardware_measured": False,
            "parent_10_cycle_physical_gain_role": (
                "T4.4.4 finite-model evidence only; not extrapolated by action MSE"
            ),
        },
        "checkpoint": {
            "path": Path(checkpoint_path).as_posix(),
            "sha256": _sha256(checkpoint_path),
            "fresh_model_count": len(checkpoint["fresh_models"]),
        },
        "source_data": {
            "path": Path(source_data_path).as_posix(),
            "row_count": 0,
            "rows_sha256": None,
            "csv_sha256": None,
        },
        "teacher_parent_selected_state_sha256": teacher_report["checkpoint"][
            "selected_state_sha256"
        ],
        "wall_time_seconds": time.perf_counter() - started,
    }
    report["gates"] = _compute_gates(report)
    report["gate_summary"] = {
        "passed": sum(bool(value) for value in report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    report["verdict"] = (
        "QUALIFIED_LONG_RECURRENCE_PASS_PHYSICAL_GAIN_NOT_ESTABLISHED"
        if report["status"] == "PASS"
        else "LONG_HORIZON_GATE_FAILED_REVOKE_QUALIFIED_STUDENT_BRANCH"
    )
    rows = source_rows(report)
    _write_source_data(Path(source_data_path), rows)
    report["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "csv_sha256": _sha256(source_data_path),
    }
    report["gates"] = _compute_gates(report)
    report["gate_summary"] = {
        "passed": sum(bool(value) for value in report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    report["verdict"] = (
        "QUALIFIED_LONG_RECURRENCE_PASS_PHYSICAL_GAIN_NOT_ESTABLISHED"
        if report["status"] == "PASS"
        else "LONG_HORIZON_GATE_FAILED_REVOKE_QUALIFIED_STUDENT_BRANCH"
    )
    # Re-emit the ledger after the source-binding gate becomes true.  This
    # avoids leaving a stale failed gate row in an otherwise passing artifact.
    rows = source_rows(report)
    _write_source_data(Path(source_data_path), rows)
    report["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "csv_sha256": _sha256(source_data_path),
    }
    report["contract_sha256"] = _canonical_sha256(_contract_view(report))
    errors = validate_artifact(report)
    if errors:
        raise RuntimeError("invalid T5.4.5 artifact: " + "; ".join(errors))
    _atomic_json(report, Path(artifact_path))
    return report


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--deployment-horizons", type=_parse_ints, default=(1_000, 100_000, 1_000_000))
    parser.add_argument("--pilot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = HorizonExtrapolationConfig(
        deployment_horizons_cycles=arguments.deployment_horizons,
        device=arguments.device,
    )
    report = run_horizon_extrapolation_validation(
        config,
        artifact_path=arguments.artifact,
        checkpoint_path=arguments.checkpoint,
        source_data_path=arguments.source_data,
        production=not arguments.pilot,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["verdict"],
                "gate_summary": report["gate_summary"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SOURCE_DATA",
    "HorizonExtrapolationConfig",
    "PROTOCOL_ID",
    "SCHEMA_VERSION",
    "TASK_ID",
    "run_horizon_extrapolation_validation",
    "validate_artifact",
    "validate_production_design",
]
