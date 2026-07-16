"""Post-hoc hidden-state and control-trajectory analysis for T4.4.2.

The selected T4.4.1 teacher remains frozen.  Native analysis accepts only the
binary g/e alphabet used during training.  A leakage marker is handled only by
an explicitly labelled out-of-distribution reset-plus-nominal proxy; it is
never passed to the GRU as a fictitious third trained token.
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
from math import ceil, isfinite, log
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from physics.differentiable_sbs_trajectory import (
    PARAMETER_NAMES,
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
    nominal_sbs_parameters,
)

# Import the parent module before torch to retain the Windows DLEnv import order.
from .bounded_residual_rnn_teacher import (
    DEFAULT_ARTIFACT as TEACHER_ARTIFACT,
    DEFAULT_CHECKPOINT as TEACHER_CHECKPOINT,
    load_and_verify_teacher_checkpoint,
)

try:  # The minimal recovery interpreter intentionally has no torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal-environment path.
    torch = None  # type: ignore[assignment]


TASK_ID = "T4.4.2"
SCHEMA_VERSION = 1
ANALYSIS_PROTOCOL_ID = "T442-FROZEN-TEACHER-HIDDEN-CONTROL-PG-V1"
SCOPE = (
    "post-hoc analysis of one frozen T4.4.1 finite-cutoff two-level GRU teacher: "
    "native g/e hidden and control trajectories, observed-history probes, "
    "conditional p(g), PCA, exponential saturation, impulse memory and local "
    "Jacobian; leakage is an OOD reset-plus-nominal proxy, not a trained token"
)

DEFAULT_ARTIFACT = Path("docs/t4_4_2_teacher_hidden_control_analysis.json")
DEFAULT_SOURCE_DATA = Path("docs/t4_4_2_teacher_hidden_control_source_data.csv")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T4.4.2 requires PyTorch; use "
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
        Path(__file__).with_name("bounded_residual_rnn_teacher.py").resolve(),
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


@dataclass(frozen=True)
class TeacherAnalysisConfig:
    analysis_half_cycles: int = 128
    physics_full_cycles: int = 10
    cutoff: int = 12
    probe_training_trajectories: int = 24
    probe_evaluation_trajectories: int = 8
    probe_training_seed: int = 441201
    probe_evaluation_seed: int = 441203
    probe_ground_probability: float = 0.72
    ewma_alpha: float = 0.20
    ridge_strength: float = 1.0e-6
    exponential_decay_grid_size: int = 4096
    memory_thresholds: tuple[float, ...] = (1.0 / np.e, 0.05, 0.01)

    def __post_init__(self) -> None:
        for name in (
            "analysis_half_cycles",
            "physics_full_cycles",
            "cutoff",
            "probe_training_trajectories",
            "probe_evaluation_trajectories",
            "exponential_decay_grid_size",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if not 2 <= self.physics_full_cycles <= 10:
            raise ValueError("physics_full_cycles must lie in [2, 10]")
        if not 4 <= self.cutoff <= 48:
            raise ValueError("cutoff must lie in [4, 48]")
        if self.analysis_half_cycles < 2 * self.physics_full_cycles:
            raise ValueError("analysis_half_cycles must cover the physics horizon")
        if self.probe_training_seed == self.probe_evaluation_seed:
            raise ValueError("probe training and evaluation seeds must be disjoint")
        for name in ("probe_ground_probability", "ewma_alpha"):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
            object.__setattr__(self, name, value)
        ridge = float(self.ridge_strength)
        if not isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge_strength must be finite and positive")
        object.__setattr__(self, "ridge_strength", ridge)
        thresholds = tuple(float(value) for value in self.memory_thresholds)
        if (
            not thresholds
            or any(not isfinite(value) or not 0.0 < value < 1.0 for value in thresholds)
            or tuple(sorted(thresholds, reverse=True)) != thresholds
        ):
            raise ValueError("memory_thresholds must be descending values in (0, 1)")
        object.__setattr__(self, "memory_thresholds", thresholds)

    @property
    def physics_half_cycles(self) -> int:
        return 2 * self.physics_full_cycles

    @property
    def contract_hash(self) -> str:
        return _canonical_sha256(asdict(self))


def validate_production_design(config: TeacherAnalysisConfig) -> None:
    minima = {
        "analysis_half_cycles": 64,
        "physics_full_cycles": 10,
        "cutoff": 12,
        "probe_training_trajectories": 24,
        "probe_evaluation_trajectories": 8,
        "exponential_decay_grid_size": 2048,
    }
    for name, minimum in minima.items():
        if int(getattr(config, name)) < minimum:
            raise ValueError(f"production {name} must be at least {minimum}")


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


SOURCE_COLUMNS = (
    "row_type",
    "sequence",
    "split",
    "trajectory_index",
    "step",
    "outcome",
    "teacher_native",
    "action_source",
    "p_g",
    "hidden_json",
    "raw_residual_json",
    "physical_residual_json",
    "metric",
    "value",
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


def _validate_native_outcomes(outcomes: Any) -> np.ndarray:
    array = np.asarray(outcomes)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("native outcomes must be a nonempty rank-two array")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("native outcomes must use integer g=0/e=1 tokens")
    result = array.astype(np.int64, copy=False)
    if not np.all((result == 0) | (result == 1)):
        raise ValueError("native teacher analysis accepts only g=0/e=1; leakage is OOD")
    return result


def fixed_sequence_library(half_cycles: int, *, seed: int = 442001) -> dict[str, tuple[Any, ...]]:
    horizon = _positive_integer(half_cycles, "half_cycles")
    rng = np.random.default_rng(seed)
    block = max(2, min(16, horizon // 4))
    sequences: dict[str, tuple[Any, ...]] = {
        "all_g": tuple(0 for _ in range(horizon)),
        "all_e": tuple(1 for _ in range(horizon)),
        "alternating_ge": tuple(index % 2 for index in range(horizon)),
        "alternating_eg": tuple(1 - index % 2 for index in range(horizon)),
        "block_ge": tuple((index // block) % 2 for index in range(horizon)),
        "block_eg": tuple(1 - (index // block) % 2 for index in range(horizon)),
        "e_impulse_then_g": (1,) + tuple(0 for _ in range(horizon - 1)),
        "g_impulse_then_e": (0,) + tuple(1 for _ in range(horizon - 1)),
        "isolated_e_mid_g": tuple(
            1 if index == horizon // 2 else 0 for index in range(horizon)
        ),
        "deterministic_mixed": tuple(
            int(value)
            for value in (
                rng.random(horizon) >= 0.72
            )
        ),
    }
    leak_index = horizon // 3
    leakage = [0] * horizon
    leakage[leak_index] = "leak"
    for index in range(leak_index + 1, horizon):
        leakage[index] = 1 if index < 2 * horizon // 3 else index % 2
    sequences["leakage_reset_nominal_proxy"] = tuple(leakage)
    return sequences


def trace_teacher_hidden(model: Any, outcomes: Any) -> dict[str, np.ndarray]:
    """Return hidden/raw/physical traces before each native g/e outcome."""

    th = _require_torch()
    native = _validate_native_outcomes(outcomes)
    reference = next(model.parameters())
    values = th.as_tensor(native, dtype=th.int64, device=reference.device)
    hidden = th.zeros(
        (values.shape[0], model.gru.hidden_size),
        dtype=reference.dtype,
        device=reference.device,
    )
    nominal = nominal_sbs_parameters(device=str(reference.device), dtype=reference.dtype)
    bounds = th.full((len(PARAMETER_NAMES),), 2.0, device=reference.device, dtype=reference.dtype)
    bounds[-1] = 1.0
    hidden_trace = []
    raw_trace = []
    residual_trace = []
    with th.no_grad():
        for step in range(values.shape[1] + 1):
            raw = model.output(th.tanh(model.dense2(th.tanh(model.dense1(hidden)))))
            residual = bounds[None, :] * th.tanh(raw)
            hidden_trace.append(hidden)
            raw_trace.append(raw)
            residual_trace.append(residual)
            if step < values.shape[1]:
                encoded = (2.0 * values[:, step : step + 1] - 1.0).to(reference.dtype)
                hidden = model.gru(encoded, hidden)
    return {
        "hidden": th.stack(hidden_trace, dim=1).cpu().numpy(),
        "raw_residual": th.stack(raw_trace, dim=1).cpu().numpy(),
        "physical_residual": th.stack(residual_trace, dim=1).cpu().numpy(),
        "physical_control": (
            nominal[None, None, :] + th.stack(residual_trace, dim=1)
        ).cpu().numpy(),
    }


def _prefix_replay_error(model: Any, outcomes: np.ndarray, trace: Mapping[str, np.ndarray]) -> float:
    th = _require_torch()
    reference = next(model.parameters())
    values = th.as_tensor(outcomes, dtype=th.int64, device=reference.device)
    maximum = 0.0
    with th.no_grad():
        for step in range(values.shape[1] + 1):
            replay = model(values[:, :step], step).detach().cpu().numpy()
            maximum = max(
                maximum,
                float(np.max(np.abs(replay - trace["raw_residual"][:, step, :]))),
            )
    return maximum


def _leakage_proxy_trace(model: Any, sequence: Sequence[Any]) -> dict[str, Any]:
    th = _require_torch()
    if "leak" not in sequence:
        raise ValueError("leakage proxy sequence must contain a leak marker")
    invalid = [token for token in sequence if token not in (0, 1, "leak")]
    if invalid:
        raise ValueError("leakage proxy sequence contains an unknown token")
    reference = next(model.parameters())
    hidden = th.zeros((1, model.gru.hidden_size), dtype=reference.dtype, device=reference.device)
    bounds = th.full((len(PARAMETER_NAMES),), 2.0, dtype=reference.dtype, device=reference.device)
    bounds[-1] = 1.0
    rows = []
    force_safe = False
    with th.no_grad():
        for step in range(len(sequence) + 1):
            if force_safe:
                raw = th.zeros((1, len(PARAMETER_NAMES)), dtype=reference.dtype, device=reference.device)
                source = "safe_nominal_after_leakage_proxy"
                native = False
                force_safe = False
            else:
                raw = model.output(th.tanh(model.dense2(th.tanh(model.dense1(hidden)))))
                source = "frozen_teacher_native"
                native = True
            residual = bounds[None, :] * th.tanh(raw)
            token = None if step == len(sequence) else sequence[step]
            rows.append(
                {
                    "step": step,
                    "outcome": token,
                    "hidden": hidden.detach().cpu().numpy()[0].tolist(),
                    "raw_residual": raw.detach().cpu().numpy()[0].tolist(),
                    "physical_residual": residual.detach().cpu().numpy()[0].tolist(),
                    "teacher_native": native,
                    "action_source": source,
                }
            )
            if token == "leak":
                hidden = th.zeros_like(hidden)
                force_safe = True
            elif token in (0, 1):
                encoded = th.tensor(
                    [[2.0 * int(token) - 1.0]], dtype=reference.dtype, device=reference.device
                )
                hidden = model.gru(encoded, hidden)
    return {
        "rows": rows,
        "leakage_is_teacher_native": False,
        "policy": "reset hidden and force exactly zero residual for the first post-leakage action",
    }


def _forced_ground_probabilities(
    model: Any,
    outcomes: np.ndarray,
    *,
    cutoff: int,
    full_cycles: int,
    teacher_config: Mapping[str, Any],
) -> dict[str, Any]:
    th = _require_torch()
    native = _validate_native_outcomes(outcomes)
    expected = 2 * full_cycles
    if native.shape[1] != expected:
        raise ValueError(f"forced p(g) outcomes must have exactly {expected} half-cycles")
    simulator = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(
            cutoff=cutoff,
            full_cycles=full_cycles,
            batch_size=native.shape[0],
            projector_delta=float(teacher_config["projector_delta"]),
            cavity_lifetime_us=float(teacher_config["cavity_lifetime_us"]),
            ancilla_t1_us=float(teacher_config["ancilla_t1_us"]),
            ancilla_t2_us=float(teacher_config["ancilla_t2_us"]),
            device=str(teacher_config["device"]),
            real_dtype=str(teacher_config["real_dtype"]),
        )
    )
    forced = th.as_tensor(native, dtype=th.int64, device=str(teacher_config["device"]))
    with th.no_grad():
        result = simulator.run(control_policy=model, forced_outcomes=forced, seed=442003)
    conditional = result.conditional_probabilities.detach().cpu().numpy()
    p_g = np.where(native == 0, conditional, 1.0 - conditional)
    return {
        "p_g": p_g,
        "physical_controls": result.physical_controls.detach().cpu().numpy(),
        "maximum_trace_error": result.maximum_trace_error,
        "maximum_hermiticity_error": result.maximum_hermiticity_error,
        "minimum_final_eigenvalue": result.minimum_final_eigenvalue,
    }


def _observed_features(outcomes: np.ndarray, alpha: float) -> np.ndarray:
    native = _validate_native_outcomes(outcomes)
    trajectories, horizon = native.shape
    features = np.zeros((trajectories, horizon, 5), dtype=np.float64)
    ewma = np.zeros(trajectories, dtype=np.float64)
    cumulative_e = np.zeros(trajectories, dtype=np.float64)
    signed_run = np.zeros(trajectories, dtype=np.float64)
    last = np.zeros(trajectories, dtype=np.float64)
    for step in range(horizon):
        features[:, step, 0] = last
        features[:, step, 1] = signed_run / max(1, horizon)
        features[:, step, 2] = cumulative_e / max(1, step)
        features[:, step, 3] = ewma
        features[:, step, 4] = step / max(1, horizon - 1)
        current = native[:, step]
        signed = 2.0 * current - 1.0
        same = signed == last
        signed_run = np.where(same, signed_run + signed, signed)
        cumulative_e += current
        ewma = alpha * current + (1.0 - alpha) * ewma
        last = signed
    return features


def _fit_ridge_probe(
    train_x: np.ndarray,
    train_p: np.ndarray,
    evaluation_x: np.ndarray,
    evaluation_p: np.ndarray,
    *,
    ridge: float,
) -> dict[str, Any]:
    x_train = np.asarray(train_x, dtype=np.float64)
    x_eval = np.asarray(evaluation_x, dtype=np.float64)
    y_train = np.log(np.clip(train_p, 1.0e-9, 1.0 - 1.0e-9) / np.clip(1.0 - train_p, 1.0e-9, 1.0))
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    train_design = np.column_stack((np.ones(x_train.shape[0]), (x_train - mean) / scale))
    eval_design = np.column_stack((np.ones(x_eval.shape[0]), (x_eval - mean) / scale))
    penalty = np.eye(train_design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        train_design.T @ train_design + penalty, train_design.T @ y_train
    )
    logits = eval_design @ coefficients
    prediction = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    truth = np.asarray(evaluation_p, dtype=np.float64)
    residual = float(np.sum((truth - prediction) ** 2))
    centered = float(np.sum((truth - np.mean(truth)) ** 2))
    r_squared = float(1.0 - residual / centered) if centered > 1.0e-15 else 0.0
    return {
        "feature_count": int(x_train.shape[1]),
        "training_rows": int(x_train.shape[0]),
        "evaluation_rows": int(x_eval.shape[0]),
        "evaluation_r_squared": r_squared,
        "evaluation_mae": float(np.mean(np.abs(truth - prediction))),
        "evaluation_rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "coefficient_l2": float(np.linalg.norm(coefficients[1:])),
        "prediction_minimum": float(np.min(prediction)),
        "prediction_maximum": float(np.max(prediction)),
    }


def _pca_summary(values: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] <= matrix.shape[1]:
        raise ValueError("PCA matrix must be tall and rank two")
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    total = float(np.sum(variance))
    ratios = variance / total if total > 0.0 else np.zeros_like(variance)
    cumulative = np.cumsum(ratios)
    dimensions = {
        str(threshold): int(np.searchsorted(cumulative, threshold, side="left") + 1)
        for threshold in (0.90, 0.95, 0.99)
    }
    return {
        "sample_count": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "feature_names": list(names),
        "explained_variance_ratio": ratios.tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
        "dimensions_for_threshold": dimensions,
        "first_component_loadings": vh[0].tolist(),
        "numerical_rank": int(np.linalg.matrix_rank(centered)),
    }


def _fit_exponential(values: np.ndarray, grid_size: int) -> dict[str, float | bool]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size < 8 or not np.all(np.isfinite(y)):
        raise ValueError("exponential fit requires at least eight finite values")
    time_axis = np.arange(y.size, dtype=np.float64)
    decays = np.linspace(1.0e-4, 0.9999, grid_size)
    best: tuple[float, float, float, float] | None = None
    for decay in decays:
        basis = decay**time_axis
        design = np.column_stack((np.ones(y.size), basis))
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        sse = float(residual @ residual)
        if best is None or sse < best[0]:
            best = (sse, float(decay), float(coefficients[0]), float(coefficients[1]))
    assert best is not None
    centered = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - best[0] / centered) if centered > 1.0e-18 else 1.0
    return {
        "decay": best[1],
        "time_constant_half_cycles": float(-1.0 / log(best[1])),
        "asymptote": best[2],
        "amplitude": best[3],
        "r_squared": r_squared,
        "monotone_single_exponential_is_high_fidelity": r_squared >= 0.95,
    }


def _memory_summary(
    baseline: np.ndarray,
    impulse: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    distance = np.linalg.norm(np.asarray(impulse) - np.asarray(baseline), axis=1)
    peak_index = int(np.argmax(distance))
    peak = float(distance[peak_index])
    normalized = distance / peak if peak > 0.0 else np.zeros_like(distance)
    crossings: dict[str, Any] = {}
    for threshold in thresholds:
        crossing = None
        for index in range(peak_index, len(normalized)):
            if np.all(normalized[index:] <= threshold):
                crossing = index
                break
        crossings[str(threshold)] = {
            "first_persistent_below_step": crossing,
            "censored_at_horizon": crossing is None,
        }
    return {
        "peak_step": peak_index,
        "peak_distance": peak,
        "final_distance": float(distance[-1]),
        "normalized_distance": normalized.tolist(),
        "threshold_crossings": crossings,
    }


def _local_jacobian_summary(model: Any, outcome: int) -> dict[str, Any]:
    th = _require_torch()
    reference = next(model.parameters())
    hidden = th.zeros((1, model.gru.hidden_size), dtype=reference.dtype, device=reference.device)
    encoded = th.tensor([[2.0 * outcome - 1.0]], dtype=reference.dtype, device=reference.device)
    with th.no_grad():
        for _ in range(512):
            hidden = model.gru(encoded, hidden)
    fixed = hidden.detach().clone().requires_grad_(True)

    def transition(vector: Any) -> Any:
        return model.gru(encoded, vector.reshape(1, -1)).reshape(-1)

    jacobian = th.autograd.functional.jacobian(transition, fixed.reshape(-1))
    values = np.linalg.eigvals(jacobian.detach().cpu().numpy())
    radius = float(np.max(np.abs(values)))
    time_constant = float(-1.0 / log(radius)) if 0.0 < radius < 1.0 else None
    with th.no_grad():
        residual = float(th.linalg.vector_norm(model.gru(encoded, fixed) - fixed).cpu())
    return {
        "outcome": "g" if outcome == 0 else "e",
        "spectral_radius": radius,
        "linearized_time_constant_half_cycles": time_constant,
        "fixed_point_residual": residual,
        "eigenvalue_real": np.real(values).tolist(),
        "eigenvalue_imag": np.imag(values).tolist(),
    }


def _trace_rows(
    sequence_name: str,
    sequence: Sequence[Any],
    trace: Mapping[str, np.ndarray],
    p_g: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows = []
    for step in range(len(sequence) + 1):
        rows.append(
            {
                "row_type": "fixed_teacher_trace",
                "sequence": sequence_name,
                "split": "fixed_native",
                "trajectory_index": "",
                "step": step,
                "outcome": "" if step == len(sequence) else ("g" if sequence[step] == 0 else "e"),
                "teacher_native": True,
                "action_source": "frozen_teacher_native",
                "p_g": "" if p_g is None or step >= p_g.shape[0] else p_g[step],
                "hidden_json": json.dumps(trace["hidden"][0, step].tolist()),
                "raw_residual_json": json.dumps(trace["raw_residual"][0, step].tolist()),
                "physical_residual_json": json.dumps(
                    trace["physical_residual"][0, step].tolist()
                ),
                "metric": "",
                "value": "",
                "detail_json": "",
            }
        )
    return rows


def run_teacher_hidden_control_analysis(
    config: TeacherAnalysisConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
    production: bool = True,
) -> dict[str, Any]:
    th = _require_torch()
    actual = config or TeacherAnalysisConfig()
    if production:
        validate_production_design(actual)
    start = time.perf_counter()
    model, teacher_report = load_and_verify_teacher_checkpoint(
        TEACHER_CHECKPOINT, TEACHER_ARTIFACT
    )
    model.eval()
    invalid_native_token_rejected = False
    try:
        trace_teacher_hidden(model, np.asarray([[2]], dtype=np.int64))
    except ValueError as error:
        invalid_native_token_rejected = "only g=0/e=1" in str(error)
    teacher_config = teacher_report["config"]
    device = str(teacher_config["device"])
    if device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("the frozen teacher requires CUDA but CUDA is unavailable")
    implementation_hash = implementation_sha256()
    sequences = fixed_sequence_library(actual.analysis_half_cycles)
    native_names = [name for name in sequences if not name.startswith("leakage_")]
    native = np.asarray([sequences[name] for name in native_names], dtype=np.int64)
    fixed_trace = trace_teacher_hidden(model, native)
    replay_error = _prefix_replay_error(model, native, fixed_trace)
    deterministic_trace = trace_teacher_hidden(model, native)
    deterministic_error = max(
        float(np.max(np.abs(fixed_trace[name] - deterministic_trace[name])))
        for name in fixed_trace
    )

    fixed_physics = _forced_ground_probabilities(
        model,
        native[:, : actual.physics_half_cycles],
        cutoff=actual.cutoff,
        full_cycles=actual.physics_full_cycles,
        teacher_config=teacher_config,
    )
    leakage_name = "leakage_reset_nominal_proxy"
    leakage_trace = _leakage_proxy_trace(model, sequences[leakage_name])

    training_rng = np.random.default_rng(actual.probe_training_seed)
    evaluation_rng = np.random.default_rng(actual.probe_evaluation_seed)
    training_outcomes = (
        training_rng.random(
            (actual.probe_training_trajectories, actual.physics_half_cycles)
        )
        >= actual.probe_ground_probability
    ).astype(np.int64)
    evaluation_outcomes = (
        evaluation_rng.random(
            (actual.probe_evaluation_trajectories, actual.physics_half_cycles)
        )
        >= actual.probe_ground_probability
    ).astype(np.int64)
    probe_outcomes = np.concatenate((training_outcomes, evaluation_outcomes), axis=0)
    probe_trace = trace_teacher_hidden(model, probe_outcomes)
    probe_physics = _forced_ground_probabilities(
        model,
        probe_outcomes,
        cutoff=actual.cutoff,
        full_cycles=actual.physics_full_cycles,
        teacher_config=teacher_config,
    )
    features = _observed_features(probe_outcomes, actual.ewma_alpha)
    train_count = actual.probe_training_trajectories
    hidden_train = probe_trace["hidden"][:train_count, :-1, :].reshape(-1, 10)
    hidden_evaluation = probe_trace["hidden"][train_count:, :-1, :].reshape(-1, 10)
    feature_train = features[:train_count].reshape(-1, features.shape[-1])
    feature_evaluation = features[train_count:].reshape(-1, features.shape[-1])
    p_train = probe_physics["p_g"][:train_count].reshape(-1)
    p_evaluation = probe_physics["p_g"][train_count:].reshape(-1)
    hidden_probe = _fit_ridge_probe(
        hidden_train,
        p_train,
        hidden_evaluation,
        p_evaluation,
        ridge=actual.ridge_strength,
    )
    observed_probe = _fit_ridge_probe(
        feature_train,
        p_train,
        feature_evaluation,
        p_evaluation,
        ridge=actual.ridge_strength,
    )

    pca_hidden_matrix = np.concatenate(
        (
            fixed_trace["hidden"].reshape(-1, 10),
            probe_trace["hidden"].reshape(-1, 10),
        ),
        axis=0,
    )
    pca_control_matrix = np.concatenate(
        (
            fixed_trace["physical_residual"].reshape(-1, 15),
            probe_trace["physical_residual"].reshape(-1, 15),
        ),
        axis=0,
    )
    hidden_pca = _pca_summary(
        pca_hidden_matrix, [f"hidden_{index}" for index in range(10)]
    )
    control_pca = _pca_summary(pca_control_matrix, PARAMETER_NAMES)

    exponential_fits: dict[str, Any] = {}
    for sequence_name in ("all_g", "all_e"):
        sequence_index = native_names.index(sequence_name)
        exponential_fits[sequence_name] = {
            parameter: _fit_exponential(
                fixed_trace["physical_residual"][sequence_index, :, parameter_index],
                actual.exponential_decay_grid_size,
            )
            for parameter_index, parameter in enumerate(PARAMETER_NAMES)
        }
    fit_values = [
        item["r_squared"]
        for sequence in exponential_fits.values()
        for item in sequence.values()
    ]
    exponential_summary = {
        "per_sequence_parameter": exponential_fits,
        "median_r_squared": float(np.median(fit_values)),
        "minimum_r_squared": float(np.min(fit_values)),
        "high_fidelity_fit_count": int(np.count_nonzero(np.asarray(fit_values) >= 0.95)),
        "total_fit_count": len(fit_values),
    }

    all_g_index = native_names.index("all_g")
    all_e_index = native_names.index("all_e")
    e_impulse_index = native_names.index("e_impulse_then_g")
    g_impulse_index = native_names.index("g_impulse_then_e")
    memory = {
        "e_impulse_followed_by_g_hidden": _memory_summary(
            fixed_trace["hidden"][all_g_index],
            fixed_trace["hidden"][e_impulse_index],
            actual.memory_thresholds,
        ),
        "e_impulse_followed_by_g_control": _memory_summary(
            fixed_trace["physical_residual"][all_g_index],
            fixed_trace["physical_residual"][e_impulse_index],
            actual.memory_thresholds,
        ),
        "g_impulse_followed_by_e_hidden": _memory_summary(
            fixed_trace["hidden"][all_e_index],
            fixed_trace["hidden"][g_impulse_index],
            actual.memory_thresholds,
        ),
        "g_impulse_followed_by_e_control": _memory_summary(
            fixed_trace["physical_residual"][all_e_index],
            fixed_trace["physical_residual"][g_impulse_index],
            actual.memory_thresholds,
        ),
    }
    jacobian = {
        "g_fixed_point": _local_jacobian_summary(model, 0),
        "e_fixed_point": _local_jacobian_summary(model, 1),
    }

    rows: list[dict[str, Any]] = []
    for sequence_index, name in enumerate(native_names):
        single_trace = {
            key: value[sequence_index : sequence_index + 1]
            for key, value in fixed_trace.items()
        }
        rows.extend(
            _trace_rows(
                name,
                sequences[name],
                single_trace,
                fixed_physics["p_g"][sequence_index],
            )
        )
    for item in leakage_trace["rows"]:
        rows.append(
            {
                "row_type": "leakage_ood_proxy_trace",
                "sequence": leakage_name,
                "split": "ood_proxy",
                "trajectory_index": "",
                "step": item["step"],
                "outcome": item["outcome"] if item["outcome"] is not None else "",
                "teacher_native": item["teacher_native"],
                "action_source": item["action_source"],
                "p_g": "",
                "hidden_json": json.dumps(item["hidden"]),
                "raw_residual_json": json.dumps(item["raw_residual"]),
                "physical_residual_json": json.dumps(item["physical_residual"]),
                "metric": "",
                "value": "",
                "detail_json": "leakage has no native teacher p(g) semantics",
            }
        )
    for trajectory_index in range(probe_outcomes.shape[0]):
        split = "probe_training" if trajectory_index < train_count else "probe_evaluation"
        split_index = trajectory_index if trajectory_index < train_count else trajectory_index - train_count
        for step in range(actual.physics_half_cycles):
            rows.append(
                {
                    "row_type": "belief_probe_trace",
                    "sequence": "deterministic_forced_probe",
                    "split": split,
                    "trajectory_index": split_index,
                    "step": step,
                    "outcome": "g" if probe_outcomes[trajectory_index, step] == 0 else "e",
                    "teacher_native": True,
                    "action_source": "frozen_teacher_native",
                    "p_g": probe_physics["p_g"][trajectory_index, step],
                    "hidden_json": json.dumps(
                        probe_trace["hidden"][trajectory_index, step].tolist()
                    ),
                    "raw_residual_json": json.dumps(
                        probe_trace["raw_residual"][trajectory_index, step].tolist()
                    ),
                    "physical_residual_json": json.dumps(
                        probe_trace["physical_residual"][trajectory_index, step].tolist()
                    ),
                    "metric": "observed_features",
                    "value": "",
                    "detail_json": json.dumps(features[trajectory_index, step].tolist()),
                }
            )
    for group, content in exponential_fits.items():
        for parameter, fit in content.items():
            rows.append(
                {
                    "row_type": "exponential_fit",
                    "sequence": group,
                    "split": "post_hoc_fit",
                    "trajectory_index": "",
                    "step": "",
                    "outcome": "",
                    "teacher_native": True,
                    "action_source": "frozen_teacher_native",
                    "p_g": "",
                    "hidden_json": "",
                    "raw_residual_json": "",
                    "physical_residual_json": "",
                    "metric": parameter,
                    "value": fit["r_squared"],
                    "detail_json": json.dumps(fit, sort_keys=True),
                }
            )
    source_data = Path(source_data_path)
    _write_source_data(source_data, rows)
    source_hash = _sha256(source_data)

    bound_vector = np.asarray([2.0] * 14 + [1.0], dtype=np.float64)
    maximum_bound_violation = float(
        np.max(
            np.maximum(
                np.abs(
                    np.concatenate(
                        (
                            fixed_trace["physical_residual"].reshape(-1, 15),
                            probe_trace["physical_residual"].reshape(-1, 15),
                        ),
                        axis=0,
                    )
                )
                - bound_vector[None, :],
                0.0,
            )
        )
    )
    post_leak_rows = [
        item
        for item in leakage_trace["rows"]
        if item["action_source"] == "safe_nominal_after_leakage_proxy"
    ]
    probe_seed_sets_disjoint = actual.probe_training_seed != actual.probe_evaluation_seed
    p_g_values = np.concatenate((fixed_physics["p_g"].reshape(-1), probe_physics["p_g"].reshape(-1)))
    all_memory_records = list(memory.values())
    gates = {
        "frozen_teacher_parent_artifact_and_checkpoint_are_current": (
            teacher_report["status"] == "PASS"
            and teacher_report["checkpoint"]["sha256"] == _sha256(TEACHER_CHECKPOINT)
            and teacher_report["checkpoint"]["selected_state_sha256"]
            == teacher_report["checkpoint"]["reload_probe"]["selected_state_sha256"]
        ),
        "native_analysis_rejects_nonbinary_leakage_token": invalid_native_token_rejected,
        "fixed_library_covers_runs_alternation_impulses_and_leakage_proxy": (
            {"all_g", "all_e", "alternating_ge", "alternating_eg", "block_ge", "block_eg", "e_impulse_then_g", "g_impulse_then_e", leakage_name}
            <= set(sequences)
        ),
        "hidden_and_control_dimensions_match_gru10_output15": (
            fixed_trace["hidden"].shape[-1] == 10
            and fixed_trace["raw_residual"].shape[-1] == 15
            and fixed_trace["physical_residual"].shape[-1] == 15
        ),
        "native_prefix_replay_and_repeat_are_exact": (
            replay_error < 1.0e-12 and deterministic_error == 0.0
        ),
        "all_native_and_probe_actions_obey_hard_bounds": maximum_bound_violation == 0.0,
        "conditional_ground_probabilities_are_complete_finite_and_bounded": (
            p_g_values.size
            == native.shape[0] * actual.physics_half_cycles
            + probe_outcomes.shape[0] * actual.physics_half_cycles
            and np.all(np.isfinite(p_g_values))
            and np.all((p_g_values >= 0.0) & (p_g_values <= 1.0))
        ),
        "physics_probability_runs_preserve_density_diagnostics": (
            max(
                fixed_physics["maximum_trace_error"],
                probe_physics["maximum_trace_error"],
            )
            < 2.0e-10
            and max(
                fixed_physics["maximum_hermiticity_error"],
                probe_physics["maximum_hermiticity_error"],
            )
            < 2.0e-10
            and min(
                fixed_physics["minimum_final_eigenvalue"],
                probe_physics["minimum_final_eigenvalue"],
            )
            >= -2.0e-10
        ),
        "belief_probe_training_and_evaluation_trajectories_are_disjoint": (
            probe_seed_sets_disjoint
            and hidden_probe["training_rows"]
            == actual.probe_training_trajectories * actual.physics_half_cycles
            and hidden_probe["evaluation_rows"]
            == actual.probe_evaluation_trajectories * actual.physics_half_cycles
        ),
        "hidden_and_observed_only_belief_probes_are_finite": all(
            isfinite(float(probe[name]))
            for probe in (hidden_probe, observed_probe)
            for name in ("evaluation_r_squared", "evaluation_mae", "evaluation_rmse")
        ),
        "pca_accounts_for_all_hidden_and_control_variance": (
            abs(sum(hidden_pca["explained_variance_ratio"]) - 1.0) < 1.0e-12
            and abs(sum(control_pca["explained_variance_ratio"]) - 1.0) < 1.0e-12
            and hidden_pca["dimensions_for_threshold"]["0.99"] <= 10
            and control_pca["dimensions_for_threshold"]["0.99"] <= 15
        ),
        "all_thirty_run_parameter_exponential_fits_are_reported": (
            exponential_summary["total_fit_count"] == 30
            and all(isfinite(float(value)) for value in fit_values)
        ),
        "bidirectional_impulse_memory_is_measured_with_censor_flags": (
            len(all_memory_records) == 4
            and all(item["peak_distance"] > 0.0 for item in all_memory_records)
            and all(
                set(item["threshold_crossings"])
                == {str(value) for value in actual.memory_thresholds}
                for item in all_memory_records
            )
        ),
        "g_and_e_fixed_point_jacobians_are_finite_and_contracting": all(
            isfinite(item["spectral_radius"])
            and 0.0 <= item["spectral_radius"] < 1.0
            and item["fixed_point_residual"] < 1.0e-10
            for item in jacobian.values()
        ),
        "leakage_is_explicitly_ood_and_first_post_leak_action_is_nominal": (
            leakage_trace["leakage_is_teacher_native"] is False
            and len(post_leak_rows) == 1
            and max(abs(value) for value in post_leak_rows[0]["physical_residual"]) == 0.0
            and post_leak_rows[0]["teacher_native"] is False
        ),
        "source_data_contains_fixed_probe_leakage_and_fit_rows": (
            len(rows) > 1000
            and {row["row_type"] for row in rows}
            == {
                "fixed_teacher_trace",
                "leakage_ood_proxy_trace",
                "belief_probe_trace",
                "exponential_fit",
            }
        ),
        "analysis_claim_boundary_excludes_mechanism_calibration_and_deployment": True,
    }
    # Several gates terminate in ``numpy.bool_`` comparisons.  Normalize the
    # public artifact to strict JSON booleans rather than relying on an encoder
    # extension that could hide other non-serializable scientific scalars.
    gates = {name: bool(value) for name, value in gates.items()}
    status = "PASS" if all(gates.values()) else "FAIL"
    hidden_pc95 = hidden_pca["dimensions_for_threshold"]["0.95"]
    control_pc95 = control_pca["dimensions_for_threshold"]["0.95"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": status,
        "scope": SCOPE,
        "analysis_protocol_id": ANALYSIS_PROTOCOL_ID,
        "implementation_sha256": implementation_hash,
        "config_contract_hash": actual.contract_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(actual),
        "teacher_provenance": {
            "artifact_path": TEACHER_ARTIFACT.as_posix(),
            "artifact_sha256": _sha256(TEACHER_ARTIFACT),
            "checkpoint_path": TEACHER_CHECKPOINT.as_posix(),
            "checkpoint_sha256": _sha256(TEACHER_CHECKPOINT),
            "selected_restart_index": teacher_report["selected_restart_index"],
            "selected_state_sha256": teacher_report["checkpoint"]["selected_state_sha256"],
            "teacher_status": teacher_report["status"],
            "teacher_parameters_frozen": True,
            "optimizer_steps_in_analysis": 0,
        },
        "execution": {
            "device": device,
            "torch_version": th.__version__,
            "wall_time_seconds": time.perf_counter() - start,
        },
        "fixed_sequences": {
            "sequence_names": list(sequences),
            "native_sequence_count": len(native_names),
            "sequence_half_cycles": actual.analysis_half_cycles,
            "physics_p_g_half_cycles": actual.physics_half_cycles,
            "native_prefix_replay_maximum_error": replay_error,
            "deterministic_trace_maximum_error": deterministic_error,
            "maximum_action_bound_violation": maximum_bound_violation,
        },
        "conditional_ground_probability": {
            "definition": "binary forced-path p(g): chosen probability for g, one-minus-chosen probability for e",
            "fixed_minimum": float(np.min(fixed_physics["p_g"])),
            "fixed_maximum": float(np.max(fixed_physics["p_g"])),
            "probe_minimum": float(np.min(probe_physics["p_g"])),
            "probe_maximum": float(np.max(probe_physics["p_g"])),
            "density_diagnostics": {
                "fixed_maximum_trace_error": fixed_physics["maximum_trace_error"],
                "probe_maximum_trace_error": probe_physics["maximum_trace_error"],
                "fixed_minimum_final_eigenvalue": fixed_physics["minimum_final_eigenvalue"],
                "probe_minimum_final_eigenvalue": probe_physics["minimum_final_eigenvalue"],
            },
        },
        "belief_state_proxy": {
            "target": "assumed-model conditional p(g), not hidden physical truth or device calibration",
            "split": {
                "training_seed": actual.probe_training_seed,
                "evaluation_seed": actual.probe_evaluation_seed,
                "training_trajectories": actual.probe_training_trajectories,
                "evaluation_trajectories": actual.probe_evaluation_trajectories,
            },
            "hidden_linear_probe": hidden_probe,
            "observed_history_feature_probe": observed_probe,
            "hidden_minus_observed_r_squared": float(
                hidden_probe["evaluation_r_squared"] - observed_probe["evaluation_r_squared"]
            ),
            "selection_rule": "none; ridge and features frozen before evaluation",
        },
        "low_dimensional_structure": {
            "hidden_pca": hidden_pca,
            "control_residual_pca": control_pca,
            "hidden_pc95_dimensions": hidden_pc95,
            "control_pc95_dimensions": control_pc95,
            "low_dimensional_at_95_percent": hidden_pc95 <= 5 and control_pc95 <= 5,
            "interpretation": (
                "empirical linear subspace of fixed/probe trajectories only; not a unique "
                "physical belief coordinate or sufficient statistic certificate"
            ),
        },
        "exponential_saturation": exponential_summary,
        "effective_memory": memory,
        "local_fixed_point_jacobian": jacobian,
        "leakage_proxy": {
            "teacher_native": False,
            "token_passed_to_teacher": False,
            "policy": leakage_trace["policy"],
            "native_leakage_claim_allowed": False,
            "p_g_defined": False,
        },
        "source_data": {
            "path": source_data.as_posix(),
            "sha256": source_hash,
            "row_count": len(rows),
            "row_types": sorted({row["row_type"] for row in rows}),
            "trace_content_sha256": _canonical_sha256(rows),
        },
        "gates": gates,
        "gate_summary": {
            "passed": sum(bool(value) for value in gates.values()),
            "total": len(gates),
            "failed": [name for name, value in gates.items() if not value],
        },
        "claim_boundary": {
            "allowed": (
                "frozen-teacher g/e response geometry, empirical effective memory, "
                "assumed-model p(g) probe and explicitly labelled leakage OOD proxy"
            ),
            "forbidden": (
                "leakage-trained teacher, calibrated physical belief state, unique mechanism, "
                "causal sufficiency, student gain retention, long-horizon robustness, FPGA or device"
            ),
            "next_gate": (
                "T4.4.3 may fit a low-dimensional student using training data only; T4.4.4 "
                "must independently test physical gain retention"
            ),
        },
    }
    _atomic_json(result, Path(artifact_path))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--pilot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_teacher_hidden_control_analysis(
        artifact_path=arguments.artifact,
        source_data_path=arguments.source_data,
        production=not arguments.pilot,
    )
    print(json.dumps({"status": result["status"], "gate_summary": result["gate_summary"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANALYSIS_PROTOCOL_ID",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "SCOPE",
    "TASK_ID",
    "TeacherAnalysisConfig",
    "fixed_sequence_library",
    "implementation_sha256",
    "run_teacher_hidden_control_analysis",
    "trace_teacher_hidden",
    "validate_production_design",
]
