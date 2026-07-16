"""Train and audit a fresh bounded-residual GRU control teacher for T4.4.1.

This task deliberately does not relabel the T2.3.7 checkpoints.  It creates
new GRU weights from new restart seeds, trains them in the same differentiable
sBs simulator, selects one restart using validation data only, and evaluates
all restarts on held-out primary and confirmation seeds.  The network emits
the raw coordinates of the canonical

    physical_controls = nominal_sBs + residual_bounds * tanh(raw)

map.  Consequently zero *residual* is the nominal sBs action; an all-zero
physical gate vector is neither the initializer nor, for every coordinate,
inside the frozen safety box.

The artifact is an offline finite-model teacher checkpoint.  It is not an
online FPGA policy, a global optimizer certificate, or physical-device
evidence.
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

# Import the physics module before importing torch.  On the Windows DLEnv host
# this keeps scipy/numpy and torch's OpenMP runtimes in their known-good order.
from physics.differentiable_sbs_trajectory import PARAMETER_NAMES, nominal_sbs_parameters
from physics.nmf_directional_ranking import (
    PAPER_RNN_ARCHITECTURE,
    POLICY_INITIALIZATION_ID,
    DirectionalRankingConfig,
    build_policy,
    evaluate_policy,
    state_dict_sha256,
    train_agent,
)
from physics.trajectory_lookup_control_oracle import ACTION_CONTRACT_ID

try:  # The minimal recovery interpreter intentionally has no torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover - minimal-environment path.
    torch = None  # type: ignore[assignment]


TASK_ID = "T4.4.1"
SCHEMA_VERSION = 1
TRAINING_PROTOCOL_ID = "T441-FRESH-BOUNDED-RESIDUAL-GRU-STRICT-SPLIT-V1"
SCOPE = (
    "fresh multi-start GRU10-DENSE256-DENSE256-OUT15 offline teacher trained "
    "with Feedback-GRAPE in the finite-cutoff two-level differentiable sBs "
    "model; nominal-plus-bounded-residual actions; not global optimality, "
    "multilevel leakage/SPAM/pulse calibration, online FPGA, or device evidence"
)

DEFAULT_ARTIFACT = Path("docs/t4_4_1_bounded_residual_rnn_teacher_validation.json")
DEFAULT_CHECKPOINT = Path("docs/t4_4_1_bounded_residual_rnn_teacher_checkpoints.pt")
DEFAULT_SOURCE_DATA = Path("docs/t4_4_1_bounded_residual_rnn_teacher_source_data.csv")
PARENT_T237_MANIFEST = Path("docs/t2_3_7_nmf_directional_ranking.json")
PARENT_T237_CHECKPOINT = Path("docs/t2_3_7_nmf_directional_ranking_checkpoints.pt")
PARENT_T329_MANIFEST = Path("docs/t3_2_9_trajectory_lookup_control_oracle.json")
PARENT_T329_CHECKPOINT = Path("docs/t3_2_9_trajectory_lookup_control_oracle.pt")
PARENT_T415_MANIFEST = Path("docs/t4_1_5_teacher_student_validation.json")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T4.4.1 requires PyTorch; use "
            "C:/ProgramData/anaconda3/envs/DLEnv/python.exe"
        )
    return torch


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    """Fingerprint the executable trainer and its two physics dependencies."""

    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[2] / "physics" / "nmf_directional_ranking.py",
        Path(__file__).resolve().parents[2] / "physics" / "differentiable_sbs_trajectory.py",
    )
    for path in paths:
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


def _unique_seeds(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    result = tuple(int(value) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


@dataclass(frozen=True)
class BoundedResidualTeacherConfig:
    """Strict split and optimizer contract for the fresh T4.4.1 teacher."""

    cutoff: int = 12
    confirmation_cutoff: int = 16
    full_cycles: int = 10
    training_epochs: int = 320
    training_batch_size: int = 6
    validation_batch_size: int = 24
    evaluation_batch_size: int = 64
    confirmation_batch_size: int = 32
    validation_interval: int = 40
    learning_rate: float = 1.0e-4
    score_baseline_decay: float = 0.95
    gradient_clip_norm: float = 10.0
    residual_l2_weight: float = 1.0e-5
    slew_l2_weight: float = 1.0e-5
    restart_seeds: tuple[int, ...] = (601, 709, 811)
    validation_seeds: tuple[int, ...] = (41011, 41017)
    evaluation_seeds: tuple[int, ...] = (
        42013,
        42017,
        42019,
        42023,
        42043,
        42061,
        42071,
        42073,
    )
    confirmation_seeds: tuple[int, ...] = (43003, 43013, 43019, 43037)
    bootstrap_seed: int = 441001
    bootstrap_repetitions: int = 20_000
    minimum_validation_gain: float = 0.10
    minimum_successful_restart_fraction: float = 2.0 / 3.0
    minimum_primary_score_gain: float = 0.05
    minimum_confirmation_score_gain: float = 0.02
    device: Literal["cpu", "cuda"] = "cuda"
    real_dtype: Literal["float32", "float64"] = "float64"
    projector_delta: float = 0.34
    cavity_lifetime_us: float = 245.0
    ancilla_t1_us: float = 50.0
    ancilla_t2_us: float = 60.0
    action_contract_id: str = ACTION_CONTRACT_ID
    policy_initialization_id: str = POLICY_INITIALIZATION_ID

    def __post_init__(self) -> None:
        for name in (
            "cutoff",
            "confirmation_cutoff",
            "full_cycles",
            "training_epochs",
            "training_batch_size",
            "validation_batch_size",
            "evaluation_batch_size",
            "confirmation_batch_size",
            "validation_interval",
            "bootstrap_repetitions",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if not 4 <= self.cutoff <= 48 or not 4 <= self.confirmation_cutoff <= 48:
            raise ValueError("cutoffs must lie in [4, 48]")
        if not 1 <= self.full_cycles <= 10:
            raise ValueError("full_cycles must lie in the validated [1, 10] envelope")
        if self.validation_interval > self.training_epochs:
            raise ValueError("validation_interval must not exceed training_epochs")
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
        for name in (
            "residual_l2_weight",
            "slew_l2_weight",
            "minimum_validation_gain",
            "minimum_primary_score_gain",
            "minimum_confirmation_score_gain",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.score_baseline_decay < 1.0:
            raise ValueError("score_baseline_decay must lie in [0, 1)")
        fraction = float(self.minimum_successful_restart_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("minimum_successful_restart_fraction must lie in (0, 1]")
        object.__setattr__(self, "minimum_successful_restart_fraction", fraction)
        for name in (
            "restart_seeds",
            "validation_seeds",
            "evaluation_seeds",
            "confirmation_seeds",
        ):
            object.__setattr__(self, name, _unique_seeds(getattr(self, name), name))
        seed_sets = tuple(
            set(getattr(self, name))
            for name in (
                "restart_seeds",
                "validation_seeds",
                "evaluation_seeds",
                "confirmation_seeds",
            )
        )
        for index, left in enumerate(seed_sets):
            for right in seed_sets[index + 1 :]:
                if left & right:
                    raise ValueError(
                        "restart/validation/evaluation/confirmation seeds must be disjoint"
                    )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        if self.ancilla_t2_us > 2.0 * self.ancilla_t1_us + 1.0e-12:
            raise ValueError("ancilla_t2_us must not exceed 2*T1")
        if self.action_contract_id != ACTION_CONTRACT_ID:
            raise ValueError("action_contract_id must preserve the canonical 15-control map")
        if self.policy_initialization_id != POLICY_INITIALIZATION_ID:
            raise ValueError("policy_initialization_id must preserve nominal residual initialization")

    @property
    def contract_hash(self) -> str:
        return _canonical_sha256(asdict(self))

    def directional_config(self) -> DirectionalRankingConfig:
        return DirectionalRankingConfig(
            cutoff=self.cutoff,
            confirmation_cutoff=self.confirmation_cutoff,
            full_cycles=self.full_cycles,
            train_epochs=self.training_epochs,
            train_batch_size=self.training_batch_size,
            validation_batch_size=self.validation_batch_size,
            test_batch_size=self.evaluation_batch_size,
            confirmation_batch_size=self.confirmation_batch_size,
            validation_interval=self.validation_interval,
            learning_rate=self.learning_rate,
            score_baseline_decay=self.score_baseline_decay,
            gradient_clip_norm=self.gradient_clip_norm,
            residual_l2_weight=self.residual_l2_weight,
            slew_l2_weight=self.slew_l2_weight,
            training_seeds=self.restart_seeds,
            validation_seeds=self.validation_seeds,
            test_seeds=self.evaluation_seeds,
            confirmation_seeds=self.confirmation_seeds,
            bootstrap_seed=self.bootstrap_seed,
            bootstrap_repetitions=self.bootstrap_repetitions,
            device=self.device,
            real_dtype=self.real_dtype,
            projector_delta=self.projector_delta,
            cavity_lifetime_us=self.cavity_lifetime_us,
            ancilla_t1_us=self.ancilla_t1_us,
            ancilla_t2_us=self.ancilla_t2_us,
        )


def validate_production_design(config: BoundedResidualTeacherConfig) -> None:
    """Reject demo-scale settings before a result can be labelled production."""

    minima = {
        "cutoff": 12,
        "confirmation_cutoff": 16,
        "full_cycles": 10,
        "training_epochs": 300,
        "training_batch_size": 6,
        "validation_batch_size": 24,
        "evaluation_batch_size": 64,
        "confirmation_batch_size": 32,
        "bootstrap_repetitions": 20_000,
    }
    for name, minimum in minima.items():
        if int(getattr(config, name)) < minimum:
            raise ValueError(f"production {name} must be at least {minimum}")
    if len(config.restart_seeds) < 3:
        raise ValueError("production requires at least three fresh restarts")
    if len(config.validation_seeds) < 2 or len(config.evaluation_seeds) < 8:
        raise ValueError("production requires at least 2 validation and 8 evaluation seeds")
    if len(config.confirmation_seeds) < 4:
        raise ValueError("production requires at least 4 confirmation seeds")


def _state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _torch_dtype(name: str) -> Any:
    th = _require_torch()
    return th.float64 if name == "float64" else th.float32


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
    "restart_index",
    "restart_seed",
    "split",
    "cutoff",
    "epoch",
    "metric",
    "value",
    "secondary_value",
    "checkpoint_sha256",
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


def _residual_bounds(*, device: str, dtype: Any) -> Any:
    th = _require_torch()
    result = th.full((len(PARAMETER_NAMES),), 2.0, device=device, dtype=dtype)
    result[-1] = 1.0
    return result


def _initialization_record(
    config: BoundedResidualTeacherConfig, directional: DirectionalRankingConfig, seed: int
) -> dict[str, Any]:
    """Reconstruct the deterministic fresh initializer before any optimizer step."""

    th = _require_torch()
    model = build_policy("nmf", directional, seed)
    dtype = _torch_dtype(config.real_dtype)
    empty = th.empty(
        (config.training_batch_size, 0), dtype=th.int64, device=config.device
    )
    with th.no_grad():
        raw = model(empty, 0)
        bounds = _residual_bounds(device=config.device, dtype=dtype)
        residual = bounds * th.tanh(raw)
        normalized = th.abs(residual / bounds)
    return {
        "initial_state_sha256": state_dict_sha256(_state_dict_cpu(model)),
        "initial_maximum_normalized_residual": float(th.max(normalized).cpu()),
        "initial_residual_rms": float(th.sqrt(th.mean(residual**2)).cpu()),
        "initializer_loads_parent_weights": False,
        "initializer_semantics": (
            "fresh random network weights with zero output bias; raw residual near zero "
            "maps to nominal sBs physical controls"
        ),
    }


def _bound_probe(config: BoundedResidualTeacherConfig) -> dict[str, Any]:
    th = _require_torch()
    dtype = _torch_dtype(config.real_dtype)
    raw_values = th.tensor(
        (-1.0e6, -100.0, -20.0, -5.0, 0.0, 5.0, 20.0, 100.0, 1.0e6),
        dtype=dtype,
        device=config.device,
    )[:, None].expand(-1, len(PARAMETER_NAMES))
    nominal = nominal_sbs_parameters(device=config.device, dtype=dtype)
    bounds = _residual_bounds(device=config.device, dtype=dtype)
    physical = nominal[None, :] + bounds[None, :] * th.tanh(raw_values)
    residual = physical - nominal[None, :]
    violation = th.maximum(th.abs(residual) - bounds[None, :], th.zeros_like(residual))
    zero_residual = nominal + bounds * th.tanh(th.zeros_like(bounds))
    absolute_zero_inside_box = bool(th.all(th.abs(nominal) <= bounds).cpu())
    return {
        "parameter_names": list(PARAMETER_NAMES),
        "output_count": len(PARAMETER_NAMES),
        "nominal_parameters": [float(value) for value in nominal.cpu()],
        "residual_bounds": [float(value) for value in bounds.cpu()],
        "probe_raw_values": [float(value) for value in raw_values[:, 0].cpu()],
        "maximum_bound_violation": float(th.max(violation).cpu()),
        "zero_residual_matches_nominal_max_error": float(
            th.max(th.abs(zero_residual - nominal)).cpu()
        ),
        "absolute_zero_physical_vector_is_inside_safe_residual_box": absolute_zero_inside_box,
        "absolute_zero_exclusion_reason": (
            "layer2_beta_real nominal magnitude exceeds its residual bound"
            if not absolute_zero_inside_box
            else "none"
        ),
    }


def _gradient_coverage(model: Any, config: BoundedResidualTeacherConfig) -> dict[str, Any]:
    th = _require_torch()
    model.train()
    model.zero_grad(set_to_none=True)
    half_index = 8
    values = th.arange(256, device=config.device, dtype=th.int64)
    shifts = th.arange(half_index - 1, -1, -1, device=config.device, dtype=th.int64)
    histories = ((values[:, None] >> shifts[None, :]) & 1).to(th.int64)
    outputs = model(histories, half_index)
    weights = th.linspace(
        0.25, 1.75, len(PARAMETER_NAMES), device=config.device, dtype=outputs.dtype
    )
    loss = th.mean((outputs * weights[None, :]) ** 2) + 1.0e-3 * th.mean(
        outputs * weights[None, :]
    )
    loss.backward()
    tensors: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        finite = gradient is not None and bool(th.all(th.isfinite(gradient)).cpu())
        nonzero = 0 if gradient is None else int(th.count_nonzero(gradient).cpu())
        total = int(parameter.numel())
        tensors.append(
            {
                "name": name,
                "finite": finite,
                "nonzero_elements": nonzero,
                "total_elements": total,
                "nonzero_fraction": float(nonzero / total),
            }
        )
    model.zero_grad(set_to_none=True)
    model.eval()
    return {
        "history_count": int(histories.shape[0]),
        "history_length": half_index,
        "parameter_tensors": tensors,
        "all_parameter_tensors_have_finite_nonzero_gradient": all(
            item["finite"] and item["nonzero_elements"] > 0 for item in tensors
        ),
        "minimum_tensor_nonzero_fraction": min(
            item["nonzero_fraction"] for item in tensors
        ),
    }


def _causality_probe(model: Any, config: BoundedResidualTeacherConfig) -> dict[str, Any]:
    th = _require_torch()
    dtype = _torch_dtype(config.real_dtype)
    sequences = th.tensor(
        (
            (0, 1, 1, 0, 1, 0, 0, 1),
            (0, 1, 1, 0, 0, 1, 1, 0),
            (1, 0, 0, 1, 0, 1, 1, 0),
            (1, 0, 0, 1, 1, 0, 0, 1),
        ),
        dtype=th.int64,
        device=config.device,
    )
    full_outputs: list[Any] = []
    with th.no_grad():
        for half_index in range(sequences.shape[1] + 1):
            full_outputs.append(model(sequences[:, :half_index], half_index).detach())
        model.reset_rollout(batch_size=sequences.shape[0], device=config.device, dtype=dtype)
        cached_outputs = []
        for half_index in range(sequences.shape[1] + 1):
            cached_outputs.append(
                model.step_rollout(sequences[:, :half_index], half_index).detach()
            )
    maximum_cached_error = max(
        float(th.max(th.abs(left - right)).cpu())
        for left, right in zip(full_outputs, cached_outputs)
    )
    # Rows 0/1 and 2/3 share their first four observations but have different
    # suffixes.  At prefix length four their actions must be identical.
    suffix_error = max(
        float(th.max(th.abs(full_outputs[4][0] - full_outputs[4][1])).cpu()),
        float(th.max(th.abs(full_outputs[4][2] - full_outputs[4][3])).cpu()),
    )
    with th.no_grad():
        replay_a = model(sequences[:, :6], 6).detach()
        replay_b = model(sequences[:, :6], 6).detach()
    return {
        "full_replay_vs_cached_maximum_error": maximum_cached_error,
        "shared_prefix_different_suffix_maximum_error": suffix_error,
        "deterministic_replay_maximum_error": float(th.max(th.abs(replay_a - replay_b)).cpu()),
        "future_outcomes_are_not_an_api_input": True,
    }


def _evaluation_rows(
    rows: list[dict[str, Any]],
    evaluation: Mapping[str, Any],
    *,
    split: str,
    restart_index: int,
    restart_seed: int,
    checkpoint_sha256: str,
) -> None:
    for per_seed in evaluation["per_seed"]:
        for metric in (
            "fidelity",
            "logical_z",
        ):
            rows.append(
                {
                    "row_type": "held_out_metric",
                    "restart_index": restart_index,
                    "restart_seed": restart_seed,
                    "split": split,
                    "cutoff": evaluation["cutoff"],
                    "epoch": "",
                    "metric": f"{metric}_effective_lifetime_cycles",
                    "value": per_seed[metric]["effective_lifetime_cycles"],
                    "secondary_value": per_seed[metric]["normalized_auc"],
                    "checkpoint_sha256": checkpoint_sha256,
                    "detail_json": json.dumps(
                        {
                            "data_seed": per_seed["seed"],
                            "trajectory_count": per_seed["trajectory_count"],
                            "maximum_trace_error": per_seed["maximum_trace_error"],
                            "maximum_hermiticity_error": per_seed[
                                "maximum_hermiticity_error"
                            ],
                            "minimum_final_eigenvalue": per_seed[
                                "minimum_final_eigenvalue"
                            ],
                        },
                        sort_keys=True,
                    ),
                }
            )


def _training_rows(
    records: Sequence[Mapping[str, Any]], evaluations: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for restart_index, record in enumerate(records):
        seed = int(record["training_seed"])
        checkpoint_sha = str(record["checkpoint_sha256"])
        for item in record["training_curve"]:
            rows.append(
                {
                    "row_type": "training_epoch",
                    "restart_index": restart_index,
                    "restart_seed": seed,
                    "split": "training",
                    "cutoff": "",
                    "epoch": item["epoch"],
                    "metric": "mean_reward",
                    "value": item["mean_reward"],
                    "secondary_value": item["gradient_norm_before_clip"],
                    "checkpoint_sha256": checkpoint_sha,
                    "detail_json": json.dumps(item, sort_keys=True),
                }
            )
        for item in record["validation_history"]:
            rows.append(
                {
                    "row_type": "validation_checkpoint",
                    "restart_index": restart_index,
                    "restart_seed": seed,
                    "split": "validation",
                    "cutoff": "",
                    "epoch": item["epoch"],
                    "metric": "selection_score",
                    "value": item["selection_score"],
                    "secondary_value": "",
                    "checkpoint_sha256": checkpoint_sha,
                    "detail_json": json.dumps(item["metric_means"], sort_keys=True),
                }
            )
        _evaluation_rows(
            rows,
            evaluations["primary"][restart_index],
            split="evaluation",
            restart_index=restart_index,
            restart_seed=seed,
            checkpoint_sha256=checkpoint_sha,
        )
        _evaluation_rows(
            rows,
            evaluations["confirmation"][restart_index],
            split="confirmation",
            restart_index=restart_index,
            restart_seed=seed,
            checkpoint_sha256=checkpoint_sha,
        )
    return rows


def _parent_provenance() -> dict[str, Any]:
    for path in (
        PARENT_T237_MANIFEST,
        PARENT_T237_CHECKPOINT,
        PARENT_T329_MANIFEST,
        PARENT_T329_CHECKPOINT,
        PARENT_T415_MANIFEST,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required parent evidence is missing: {path}")
    th = _require_torch()
    t237_manifest = json.loads(PARENT_T237_MANIFEST.read_text(encoding="utf-8"))
    t237_checkpoint = th.load(PARENT_T237_CHECKPOINT, map_location="cpu", weights_only=False)
    t329_manifest = json.loads(PARENT_T329_MANIFEST.read_text(encoding="utf-8"))
    t415_manifest = json.loads(PARENT_T415_MANIFEST.read_text(encoding="utf-8"))
    parent_model_hashes = [
        str(item["checkpoint_sha256"]) for item in t237_checkpoint["models"]["nmf"]
    ]
    return {
        "t2_3_7": {
            "role": "architecture_training_and_directional_ranking_parent_only",
            "manifest_path": PARENT_T237_MANIFEST.as_posix(),
            "manifest_sha256": _sha256(PARENT_T237_MANIFEST),
            "checkpoint_path": PARENT_T237_CHECKPOINT.as_posix(),
            "checkpoint_sha256": _sha256(PARENT_T237_CHECKPOINT),
            "status": t237_manifest.get("status"),
            "training_seeds": list(t237_manifest["config"]["training_seeds"]),
            "nmf_model_sha256s": parent_model_hashes,
        },
        "t3_2_9": {
            "role": "bounded_action_contract_and_finite_horizon_control_reference_only",
            "manifest_path": PARENT_T329_MANIFEST.as_posix(),
            "manifest_sha256": _sha256(PARENT_T329_MANIFEST),
            "checkpoint_path": PARENT_T329_CHECKPOINT.as_posix(),
            "checkpoint_sha256": _sha256(PARENT_T329_CHECKPOINT),
            "status": t329_manifest.get("status"),
            "same_bounded_fifteen_action_contract_gate": t329_manifest["gates"].get(
                "same_bounded_fifteen_action_contract_is_used"
            ),
        },
        "t4_1_5": {
            "role": "old_teacher_distillation_provenance_only_not_new_teacher_weights",
            "manifest_path": PARENT_T415_MANIFEST.as_posix(),
            "manifest_sha256": _sha256(PARENT_T415_MANIFEST),
            "status": t415_manifest.get("status"),
            "declared_t2_3_7_checkpoint_sha256": t415_manifest["teacher_provenance"][
                "checkpoint_sha256"
            ],
        },
    }


def _bootstrap_gain(
    teacher_values: Sequence[float],
    standard_values: Sequence[float],
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    left = np.asarray(teacher_values, dtype=np.float64)
    right = np.asarray(standard_values, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired bootstrap values must be equal nonempty vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, left.size, size=(repetitions, left.size))
    samples = np.mean(left[indices] - right[indices], axis=1)
    difference = left - right
    return {
        "mean_difference": float(np.mean(difference)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_positive": float(np.mean(samples > 0.0)),
    }


def _checkpoint_reload_probe(
    checkpoint_path: Path,
    directional: DirectionalRankingConfig,
    config: BoundedResidualTeacherConfig,
    selected_index: int,
) -> dict[str, Any]:
    th = _require_torch()
    payload = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    item = payload["restarts"][selected_index]
    model = build_policy("nmf", directional, int(item["training_seed"]))
    model.load_state_dict(item["state_dict"])
    model.eval()
    history = th.tensor(
        ((0, 1, 1, 0, 1, 0), (1, 0, 0, 1, 0, 1)),
        dtype=th.int64,
        device=config.device,
    )
    with th.no_grad():
        output = model(history, history.shape[1]).detach().cpu()
    return {
        "selected_state_sha256": state_dict_sha256(_state_dict_cpu(model)),
        "saved_state_sha256": item["checkpoint_sha256"],
        "probe_output_sha256": hashlib.sha256(output.numpy().tobytes()).hexdigest(),
        "all_values_finite": bool(th.all(th.isfinite(output)).cpu()),
    }


def run_bounded_residual_teacher_training(
    config: BoundedResidualTeacherConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
    production: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """Train, select, evaluate, persist, and independently reload the teacher."""

    th = _require_torch()
    actual = config or BoundedResidualTeacherConfig()
    if production:
        validate_production_design(actual)
    if actual.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA production run requested but torch.cuda.is_available() is false")
    start = time.perf_counter()
    directional = actual.directional_config()
    implementation_hash = implementation_sha256()
    contract_hash = _canonical_sha256(
        {
            "config_contract_hash": actual.contract_hash,
            "implementation_sha256": implementation_hash,
            "training_protocol_id": TRAINING_PROTOCOL_ID,
        }
    )
    checkpoint = Path(checkpoint_path)
    artifact = Path(artifact_path)
    source_data = Path(source_data_path)
    parents = _parent_provenance()
    parent_hashes = set(parents["t2_3_7"]["nmf_model_sha256s"])

    checkpoint_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "config_contract_hash": actual.contract_hash,
        "checkpoint_contract_hash": contract_hash,
        "implementation_sha256": implementation_hash,
        "config": asdict(actual),
        "parent_t2_3_7_checkpoint_sha256": parents["t2_3_7"]["checkpoint_sha256"],
        "restarts": [],
        "selected_restart_index": None,
    }
    if resume and checkpoint.exists():
        loaded = th.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict) or loaded.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("checkpoint is not a resumable T4.4.1 schema-v1 payload")
        if loaded.get("checkpoint_contract_hash") != contract_hash:
            raise ValueError("checkpoint contract differs from config/source/training protocol")
        checkpoint_payload = loaded
    elif checkpoint.exists():
        raise FileExistsError(
            f"checkpoint already exists and resume=False: {checkpoint}; choose a new path"
        )

    existing = {
        int(item["training_seed"]): item for item in checkpoint_payload["restarts"]
    }
    models: list[Any] = []
    records: list[dict[str, Any]] = []
    resumed_restarts = 0
    for seed in actual.restart_seeds:
        initialization = _initialization_record(actual, directional, seed)
        if seed in existing:
            item = existing[seed]
            model = build_policy("nmf", directional, seed)
            model.load_state_dict(item["state_dict"])
            model.eval()
            if state_dict_sha256(_state_dict_cpu(model)) != item["checkpoint_sha256"]:
                raise ValueError("resumed teacher state hash mismatch")
            record = dict(item["training_record"])
            if record["initial_state_sha256"] != initialization["initial_state_sha256"]:
                raise ValueError("resumed initializer provenance mismatch")
            resumed_restarts += 1
        else:
            model, base_record = train_agent("nmf", seed, directional)
            validation_gain = float(
                base_record["best_validation_score"]
                - base_record["initial_validation_score"]
            )
            record = {
                **base_record,
                **initialization,
                "validation_gain": validation_gain,
                "restart_success": validation_gain >= actual.minimum_validation_gain,
                "best_epoch_reached_training_cap": (
                    int(base_record["best_validation_epoch"]) == actual.training_epochs
                ),
                "optimizer_global_convergence_claimed": False,
                "parent_checkpoint_loaded": False,
            }
            item = {
                "training_seed": int(seed),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "initial_state_sha256": record["initial_state_sha256"],
                "state_dict": _state_dict_cpu(model),
                "training_record": record,
            }
            checkpoint_payload["restarts"].append(item)
            _atomic_torch_save(checkpoint_payload, checkpoint)
            print(
                json.dumps(
                    {
                        "event": "fresh_restart_complete",
                        "seed": seed,
                        "best_epoch": record["best_validation_epoch"],
                        "validation_gain": record["validation_gain"],
                        "success": record["restart_success"],
                        "checkpoint_sha256": record["checkpoint_sha256"],
                    }
                ),
                flush=True,
            )
        models.append(model)
        records.append(record)

    selected_index = max(
        range(len(records)), key=lambda index: float(records[index]["best_validation_score"])
    )
    checkpoint_payload["selected_restart_index"] = selected_index
    checkpoint_payload["selection_rule"] = (
        "maximum best validation selection score; evaluation and confirmation remain blind"
    )
    _atomic_torch_save(checkpoint_payload, checkpoint)

    standard_primary = evaluate_policy(
        "standard",
        None,
        directional,
        cutoff=actual.cutoff,
        batch_size=actual.evaluation_batch_size,
        seeds=actual.evaluation_seeds,
    )
    standard_confirmation = evaluate_policy(
        "standard",
        None,
        directional,
        cutoff=actual.confirmation_cutoff,
        batch_size=actual.confirmation_batch_size,
        seeds=actual.confirmation_seeds,
    )
    primary = [
        evaluate_policy(
            "nmf",
            model,
            directional,
            cutoff=actual.cutoff,
            batch_size=actual.evaluation_batch_size,
            seeds=actual.evaluation_seeds,
        )
        for model in models
    ]
    confirmation = [
        evaluate_policy(
            "nmf",
            model,
            directional,
            cutoff=actual.confirmation_cutoff,
            batch_size=actual.confirmation_batch_size,
            seeds=actual.confirmation_seeds,
        )
        for model in models
    ]
    evaluations = {"primary": primary, "confirmation": confirmation}
    selected_model = models[selected_index]
    selected_record = records[selected_index]
    bound_probe = _bound_probe(actual)
    gradient_coverage = _gradient_coverage(selected_model, actual)
    causality = _causality_probe(selected_model, actual)

    rows = _training_rows(records, evaluations)
    for index, value in enumerate(bound_probe["residual_bounds"]):
        rows.append(
            {
                "row_type": "action_bound",
                "restart_index": selected_index,
                "restart_seed": selected_record["training_seed"],
                "split": "contract",
                "cutoff": "",
                "epoch": "",
                "metric": PARAMETER_NAMES[index],
                "value": value,
                "secondary_value": bound_probe["nominal_parameters"][index],
                "checkpoint_sha256": selected_record["checkpoint_sha256"],
                "detail_json": "",
            }
        )
    _write_source_data(source_data, rows)
    source_hash = _sha256(source_data)
    checkpoint_hash = _sha256(checkpoint)
    reload_probe = _checkpoint_reload_probe(
        checkpoint, directional, actual, selected_index
    )

    selected_primary = primary[selected_index]
    selected_confirmation = confirmation[selected_index]
    primary_gain = float(
        selected_primary["selection_score_mean"] - standard_primary["selection_score_mean"]
    )
    confirmation_gain = float(
        selected_confirmation["selection_score_mean"]
        - standard_confirmation["selection_score_mean"]
    )
    bootstrap = _bootstrap_gain(
        [item["logical_z"]["effective_lifetime_cycles"] for item in selected_primary["per_seed"]],
        [item["logical_z"]["effective_lifetime_cycles"] for item in standard_primary["per_seed"]],
        seed=actual.bootstrap_seed,
        repetitions=actual.bootstrap_repetitions,
    )
    restart_successes = sum(bool(record["restart_success"]) for record in records)
    required_successes = int(
        np.ceil(actual.minimum_successful_restart_fraction * len(records))
    )
    all_per_seed = [
        item
        for evaluation in (*primary, *confirmation)
        for item in evaluation["per_seed"]
    ]
    new_final_hashes = [str(record["checkpoint_sha256"]) for record in records]
    new_initial_hashes = [str(record["initial_state_sha256"]) for record in records]
    parent_seeds = set(parents["t2_3_7"]["training_seeds"])

    gates = {
        "all_required_parent_evidence_is_current_and_passed": (
            parents["t2_3_7"]["status"] == "PASS"
            and parents["t3_2_9"]["status"] == "PASS"
            and parents["t4_1_5"]["status"] == "PASS"
            and parents["t4_1_5"]["declared_t2_3_7_checkpoint_sha256"]
            == parents["t2_3_7"]["checkpoint_sha256"]
            and parents["t3_2_9"]["same_bounded_fifteen_action_contract_gate"] is True
        ),
        "canonical_action_contract_has_exactly_fifteen_outputs": (
            actual.action_contract_id == ACTION_CONTRACT_ID
            and bound_probe["output_count"] == 15
            and len(bound_probe["parameter_names"]) == 15
        ),
        "zero_residual_is_nominal_and_absolute_zero_is_not_substituted": (
            bound_probe["zero_residual_matches_nominal_max_error"] == 0.0
            and not bound_probe[
                "absolute_zero_physical_vector_is_inside_safe_residual_box"
            ]
        ),
        "all_probed_actions_obey_hard_residual_bounds": (
            bound_probe["maximum_bound_violation"] == 0.0
        ),
        "fresh_restarts_and_all_data_splits_are_disjoint": (
            not (set(actual.restart_seeds) & parent_seeds)
            and len(
                set(actual.restart_seeds)
                | set(actual.validation_seeds)
                | set(actual.evaluation_seeds)
                | set(actual.confirmation_seeds)
            )
            == sum(
                len(values)
                for values in (
                    actual.restart_seeds,
                    actual.validation_seeds,
                    actual.evaluation_seeds,
                    actual.confirmation_seeds,
                )
            )
        ),
        "three_or_more_fresh_72853_parameter_gru_restarts_are_retained": (
            len(records) >= 3
            and all(int(record["parameter_count"]) == 72_853 for record in records)
            and all(record["architecture"] == PAPER_RNN_ARCHITECTURE for record in records)
        ),
        "no_parent_checkpoint_was_loaded_or_renamed": (
            all(not record["parent_checkpoint_loaded"] for record in records)
            and not (set(new_final_hashes) & parent_hashes)
            and not (set(new_initial_hashes) & parent_hashes)
            and len(set(new_final_hashes)) == len(new_final_hashes)
        ),
        "every_optimizer_changes_its_fresh_initial_state": all(
            initial != final
            for initial, final in zip(new_initial_hashes, new_final_hashes)
        ),
        "initial_actions_are_close_to_nominal_not_absolute_zero": all(
            float(record["initial_maximum_normalized_residual"]) < 0.01
            for record in records
        ),
        "required_fraction_of_restarts_reaches_validation_gain_gate": (
            restart_successes >= required_successes
        ),
        "restart_selection_is_validation_only": (
            selected_index
            == max(
                range(len(records)),
                key=lambda index: float(records[index]["best_validation_score"]),
            )
        ),
        "selected_teacher_beats_nominal_on_primary_held_out_score": (
            primary_gain >= actual.minimum_primary_score_gain
        ),
        "selected_teacher_beats_nominal_on_confirmation_cutoff": (
            confirmation_gain >= actual.minimum_confirmation_score_gain
        ),
        "selected_teacher_has_positive_paired_logical_z_lifetime_gain": (
            bootstrap["mean_difference"] > 0.0
            and bootstrap["ci95_low"] > 0.0
        ),
        "all_held_out_density_diagnostics_are_finite_and_physical": all(
            np.isfinite(item["maximum_trace_error"])
            and item["maximum_trace_error"] < 2.0e-10
            and np.isfinite(item["maximum_hermiticity_error"])
            and item["maximum_hermiticity_error"] < 2.0e-10
            and item["minimum_final_eigenvalue"] >= -2.0e-10
            for item in all_per_seed
        ),
        "all_selected_teacher_parameter_tensors_receive_gradient": (
            gradient_coverage["all_parameter_tensors_have_finite_nonzero_gradient"]
        ),
        "causal_cached_and_replayed_outputs_are_exact": (
            causality["full_replay_vs_cached_maximum_error"] < 1.0e-12
            and causality["shared_prefix_different_suffix_maximum_error"] == 0.0
            and causality["deterministic_replay_maximum_error"] == 0.0
        ),
        "checkpoint_reload_matches_selected_state_and_is_finite": (
            reload_probe["selected_state_sha256"]
            == reload_probe["saved_state_sha256"]
            == selected_record["checkpoint_sha256"]
            and reload_probe["all_values_finite"]
        ),
        "source_data_contains_full_training_validation_and_held_out_records": (
            len(rows)
            == sum(len(record["training_curve"]) for record in records)
            + sum(len(record["validation_history"]) for record in records)
            + len(records)
            * (
                2 * len(actual.evaluation_seeds)
                + 2 * len(actual.confirmation_seeds)
            )
            + len(PARAMETER_NAMES)
            and {row["split"] for row in rows}
            >= {"training", "validation", "evaluation", "confirmation", "contract"}
        ),
        "failed_restarts_and_training_cap_hits_are_explicit": (
            all("restart_success" in record for record in records)
            and all("best_epoch_reached_training_cap" in record for record in records)
            and all(not record["optimizer_global_convergence_claimed"] for record in records)
        ),
        "claim_boundary_excludes_online_hardware_and_global_optimality": True,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": status,
        "scope": SCOPE,
        "training_protocol_id": TRAINING_PROTOCOL_ID,
        "implementation_sha256": implementation_hash,
        "config_contract_hash": actual.contract_hash,
        "checkpoint_contract_hash": contract_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(actual),
        "execution": {
            "device": actual.device,
            "torch_version": th.__version__,
            "cuda_available": th.cuda.is_available(),
            "cuda_device": (
                th.cuda.get_device_name(0) if th.cuda.is_available() else None
            ),
            "resumed_restarts_this_invocation": resumed_restarts,
            "fresh_restart_count_in_checkpoint": len(records),
            "wall_time_seconds": time.perf_counter() - start,
        },
        "parent_provenance": parents,
        "action_contract": bound_probe,
        "training_restarts": records,
        "failed_restart_indices": [
            index for index, record in enumerate(records) if not record["restart_success"]
        ],
        "training_cap_hit_indices": [
            index
            for index, record in enumerate(records)
            if record["best_epoch_reached_training_cap"]
        ],
        "selected_restart_index": selected_index,
        "selection_rule": checkpoint_payload["selection_rule"],
        "evaluation": {
            "standard_primary": standard_primary,
            "standard_confirmation": standard_confirmation,
            "teacher_primary_all_restarts": primary,
            "teacher_confirmation_all_restarts": confirmation,
            "selected_primary_score_gain": primary_gain,
            "selected_confirmation_score_gain": confirmation_gain,
            "selected_primary_logical_z_lifetime_paired_bootstrap": bootstrap,
        },
        "gradient_coverage": gradient_coverage,
        "causality": causality,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_hash,
            "schema_version": SCHEMA_VERSION,
            "selected_state_sha256": selected_record["checkpoint_sha256"],
            "reload_probe": reload_probe,
        },
        "source_data": {
            "path": source_data.as_posix(),
            "sha256": source_hash,
            "row_count": len(rows),
            "row_types": sorted({row["row_type"] for row in rows}),
        },
        "gates": gates,
        "gate_summary": {
            "passed": sum(bool(value) for value in gates.values()),
            "total": len(gates),
            "failed": [name for name, value in gates.items() if not value],
        },
        "claim_boundary": {
            "allowed": (
                "fresh strict-split bounded-residual GRU teacher within the finite-cutoff "
                "two-level differentiable sBs model"
            ),
            "forbidden": (
                "global optimizer convergence, paper-exact 1000-cycle channel lifetime, "
                "multilevel leakage/SPAM/pulse robustness, online/FPGA deployment, or "
                "experimental device gain"
            ),
            "failure_branch": (
                "if status is FAIL, do not start teacher hidden-state/distillation claims; "
                "retain drift/regime-aware MAP-LUT as the deployable fallback"
            ),
            "optimizer_note": (
                "all budget-cap hits and failed restarts are listed; no global convergence "
                "claim is made even when the production gate passes"
            ),
        },
    }
    _atomic_json(result, artifact)
    return result


def load_and_verify_teacher_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
) -> tuple[Any, Mapping[str, Any]]:
    """Load the selected offline teacher only after all hash/status checks pass."""

    th = _require_torch()
    checkpoint = Path(checkpoint_path)
    artifact = Path(artifact_path)
    report = json.loads(artifact.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or not all(report.get("gates", {}).values()):
        raise ValueError("teacher artifact is not a complete PASS")
    if report.get("implementation_sha256") != implementation_sha256():
        raise ValueError("teacher implementation hash is stale")
    if _sha256(checkpoint) != report["checkpoint"]["sha256"]:
        raise ValueError("teacher checkpoint file hash mismatch")
    payload = th.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_contract_hash") != report["checkpoint_contract_hash"]:
        raise ValueError("teacher checkpoint contract mismatch")
    config = BoundedResidualTeacherConfig(**payload["config"])
    directional = config.directional_config()
    selected_index = int(payload["selected_restart_index"])
    item = payload["restarts"][selected_index]
    model = build_policy("nmf", directional, int(item["training_seed"]))
    model.load_state_dict(item["state_dict"])
    model.eval()
    if state_dict_sha256(_state_dict_cpu(model)) != item["checkpoint_sha256"]:
        raise ValueError("selected teacher state hash mismatch")
    if item["checkpoint_sha256"] != report["checkpoint"]["selected_state_sha256"]:
        raise ValueError("selected teacher state differs from artifact")
    return model, report


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--restart-seeds", type=_parse_seeds, default=(601, 709, 811))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = BoundedResidualTeacherConfig(
        training_epochs=arguments.epochs,
        restart_seeds=arguments.restart_seeds,
        device=arguments.device,
    )
    result = run_bounded_residual_teacher_training(
        config,
        artifact_path=arguments.artifact,
        checkpoint_path=arguments.checkpoint,
        source_data_path=arguments.source_data,
        production=not arguments.pilot,
        resume=not arguments.no_resume,
    )
    print(json.dumps({"status": result["status"], "gate_summary": result["gate_summary"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACTION_CONTRACT_ID",
    "BoundedResidualTeacherConfig",
    "DEFAULT_ARTIFACT",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_SOURCE_DATA",
    "SCOPE",
    "TASK_ID",
    "TRAINING_PROTOCOL_ID",
    "implementation_sha256",
    "load_and_verify_teacher_checkpoint",
    "run_bounded_residual_teacher_training",
    "validate_production_design",
]
