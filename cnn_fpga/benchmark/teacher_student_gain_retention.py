"""Physical teacher-to-student gain-retention gate for T4.4.4.

The ten-cycle stochastic lane compares standard, all five frozen exact-budget
Markovian agents, the selected fresh teacher, the frozen handcrafted
recurrence extrapolation, and the distilled four-state student.  A separate
two-cycle exact branch-enumeration lane adds the finite-horizon control oracle;
the oracle is never extrapolated beyond its registered horizon.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from cnn_fpga.control.low_dimensional_recurrence import (
    LowDimensionalRecurrenceArtifact,
)
from physics.differentiable_sbs_trajectory import (
    DifferentiableSBSConfig,
    DifferentiableSBSTrajectorySimulator,
    nominal_sbs_parameters,
)
from physics.exponential_recurrence_control import (
    ExponentialRecurrenceConfig,
    load_policy_state as load_handcrafted_policy,
)
from physics.latest_outcome_markovian import (
    COMPUTE_CONTRACT as MF_COMPUTE_CONTRACT,
    build_budget_matched_policy,
)
from physics.nmf_directional_ranking import (
    DirectionalRankingConfig,
    _effective_lifetime,
    _torch_dtype,
    evaluate_policy,
    state_dict_sha256,
)
from physics.trajectory_lookup_control_oracle import (
    TrajectoryLookupConfig,
    enumerate_terminal_trajectories,
    load_policy_from_state as load_lookup_policy,
)

from .bounded_residual_rnn_teacher import (
    DEFAULT_ARTIFACT as TEACHER_ARTIFACT,
    DEFAULT_CHECKPOINT as TEACHER_CHECKPOINT,
    BoundedResidualTeacherConfig,
    load_and_verify_teacher_checkpoint,
)
from .exponential_recurrence_baseline import (
    DEFAULT_ARTIFACT as HANDCRAFTED_ARTIFACT,
    DEFAULT_CHECKPOINT as HANDCRAFTED_CHECKPOINT,
    implementation_sha256 as handcrafted_implementation_sha256,
)
from .latest_outcome_markovian_baseline import (
    DEFAULT_ARTIFACT as MF_ARTIFACT,
    DEFAULT_CHECKPOINT as MF_CHECKPOINT,
    implementation_sha256 as mf_implementation_sha256,
)
from .low_dimensional_student_distillation import (
    DEFAULT_ARTIFACT as STUDENT_REPORT,
    DEFAULT_STUDENT,
    implementation_sha256 as student_implementation_sha256,
)
from .trajectory_lookup_control_oracle import (
    DEFAULT_ARTIFACT as LOOKUP_ARTIFACT,
    DEFAULT_CHECKPOINT as LOOKUP_CHECKPOINT,
    implementation_sha256 as lookup_implementation_sha256,
)

try:  # The recovery interpreter intentionally has no torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


TASK_ID = "T4.4.4"
SCHEMA_VERSION = 1
PROTOCOL_ID = "T444-PHYSICAL-GAIN-RETENTION-DUAL-HORIZON-V1"
SCOPE = (
    "paired unseen-seed ten-cycle physical gain retention for standard all five "
    "exact-budget MF agents frozen teacher handcrafted recurrence extrapolation "
    "and distilled student plus a separate exact two-cycle lane containing the "
    "finite-horizon control oracle; not multilevel leakage long-horizon OOD pulse "
    "device quantized RTL FPGA or board evidence"
)

DEFAULT_ARTIFACT = Path("docs/t4_4_4_teacher_student_gain_retention.json")
DEFAULT_SOURCE_DATA = Path("docs/t4_4_4_teacher_student_gain_retention_source_data.csv")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T4.4.4 requires PyTorch; use C:/ProgramData/anaconda3/envs/DLEnv/python.exe"
        )
    return torch


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "control" / "low_dimensional_recurrence.py",
        Path(__file__).resolve().parents[2] / "physics" / "nmf_directional_ranking.py",
        Path(__file__).resolve().parents[2] / "physics" / "differentiable_sbs_trajectory.py",
        Path(__file__).resolve().parents[2] / "physics" / "exponential_recurrence_control.py",
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
        raise TypeError(f"{name} must be a sequence")
    result = tuple(int(value) for value in values)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


@dataclass(frozen=True)
class GainRetentionConfig:
    full_cycles: int = 10
    exact_control_oracle_cycles: int = 2
    cutoff: int = 12
    confirmation_cutoff: int = 16
    primary_batch_size: int = 64
    confirmation_batch_size: int = 32
    primary_seeds: tuple[int, ...] = (
        444401,
        444403,
        444409,
        444419,
        444421,
        444443,
        444449,
        444461,
    )
    confirmation_seeds: tuple[int, ...] = (444503, 444517, 444523, 444527)
    minimum_gain_retention_fraction: float = 0.90
    minimum_gain_retention_ci_lower: float = 0.90
    maximum_teacher_student_pg_difference: float = 0.02
    bootstrap_seed: int = 444991
    bootstrap_repetitions: int = 20_000
    device: Literal["cpu", "cuda"] = "cuda"
    real_dtype: Literal["float32", "float64"] = "float64"

    def __post_init__(self) -> None:
        for name in (
            "full_cycles",
            "exact_control_oracle_cycles",
            "cutoff",
            "confirmation_cutoff",
            "primary_batch_size",
            "confirmation_batch_size",
            "bootstrap_repetitions",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        if not 1 <= self.full_cycles <= 10:
            raise ValueError("full_cycles must lie in [1,10]")
        if self.exact_control_oracle_cycles != 2:
            raise ValueError("control oracle is registered only for exactly two cycles")
        if not 4 <= self.cutoff <= 48 or not 4 <= self.confirmation_cutoff <= 48:
            raise ValueError("cutoffs must lie in [4,48]")
        primary = _unique_seeds(self.primary_seeds, "primary_seeds")
        confirmation = _unique_seeds(self.confirmation_seeds, "confirmation_seeds")
        if set(primary) & set(confirmation):
            raise ValueError("primary and confirmation seeds must be disjoint")
        object.__setattr__(self, "primary_seeds", primary)
        object.__setattr__(self, "confirmation_seeds", confirmation)
        for name in (
            "minimum_gain_retention_fraction",
            "minimum_gain_retention_ci_lower",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0,1]")
            object.__setattr__(self, name, value)
        difference = float(self.maximum_teacher_student_pg_difference)
        if not np.isfinite(difference) or not 0.0 < difference < 1.0:
            raise ValueError("maximum_teacher_student_pg_difference must lie in (0,1)")
        object.__setattr__(self, "maximum_teacher_student_pg_difference", difference)
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")

    @property
    def contract_hash(self) -> str:
        return _canonical_sha256(asdict(self))


def validate_production_design(config: GainRetentionConfig) -> None:
    minima = {
        "full_cycles": 10,
        "cutoff": 12,
        "confirmation_cutoff": 16,
        "primary_batch_size": 64,
        "confirmation_batch_size": 32,
        "bootstrap_repetitions": 20_000,
    }
    for name, minimum in minima.items():
        if int(getattr(config, name)) < minimum:
            raise ValueError(f"production {name} must be at least {minimum}")
    if len(config.primary_seeds) < 8 or len(config.confirmation_seeds) < 4:
        raise ValueError("production requires at least eight primary and four confirmation seeds")
    if config.real_dtype != "float64":
        raise ValueError("production requires float64 physics")
    if config.minimum_gain_retention_fraction != 0.90:
        raise ValueError("production point retention threshold is frozen at 0.90")
    if config.minimum_gain_retention_ci_lower != 0.90:
        raise ValueError("production confidence-bound threshold is frozen at 0.90")


if torch is not None:

    class DistilledStudentTorchPolicy(torch.nn.Module):
        """Torch simulator adapter for the teacher-free NumPy artifact."""

        family = "distilled_four_state_exponential_student"

        def __init__(
            self,
            artifact: LowDimensionalRecurrenceArtifact,
            *,
            device: str,
            dtype: Any,
        ) -> None:
            super().__init__()
            self.register_buffer(
                "initial_state", torch.tensor(artifact.initial_state, device=device, dtype=dtype)
            )
            self.register_buffer(
                "decays", torch.tensor(artifact.outcome_decays, device=device, dtype=dtype)
            )
            self.register_buffer(
                "saturations",
                torch.tensor(artifact.outcome_saturations, device=device, dtype=dtype),
            )
            self.register_buffer(
                "weights", torch.tensor(artifact.output_weights, device=device, dtype=dtype)
            )
            self.register_buffer(
                "bias", torch.tensor(artifact.output_bias, device=device, dtype=dtype)
            )

        def state_after_history(self, history: Any) -> Any:
            outcomes = history.to(device=self.initial_state.device, dtype=torch.int64)
            if not bool(torch.all((outcomes >= 0) & (outcomes <= 1)).detach().cpu()):
                raise ValueError("physical lane supports native g/e only")
            state = self.initial_state[None, :].expand(outcomes.shape[0], -1)
            for index in range(outcomes.shape[1]):
                outcome = outcomes[:, index]
                decay = self.decays[outcome]
                saturation = self.saturations[outcome]
                state = decay * state + (1.0 - decay) * saturation
            return state

        def forward(self, history: Any, half_index: int) -> Any:
            if history.shape[1] != int(half_index):
                raise ValueError("history width must equal half_index")
            state = self.state_after_history(history)
            return state @ self.weights.T + self.bias


    class FrozenHandcraftedRecurrencePolicy(torch.nn.Module):
        """Frozen T3.2.10 recurrence with an explicitly labelled horizon extension."""

        family = "handcrafted_recurrence_frozen_extrapolation"

        def __init__(self, source: Any) -> None:
            super().__init__()
            self.register_buffer("initial_raw", source.initial_raw.detach().clone())
            self.register_buffer("decays", source.ge_decay().detach().clone())
            self.register_buffer(
                "saturations", source.ge_saturation_raw.detach().clone()
            )

        def forward(self, history: Any, half_index: int) -> Any:
            if history.shape[1] != int(half_index):
                raise ValueError("history width must equal half_index")
            outcomes = history.to(device=self.initial_raw.device, dtype=torch.int64)
            if not bool(torch.all((outcomes >= 0) & (outcomes <= 1)).detach().cpu()):
                raise ValueError("physical lane supports native g/e only")
            state = self.initial_raw[None, :].expand(outcomes.shape[0], -1)
            for index in range(outcomes.shape[1]):
                outcome = outcomes[:, index]
                state = self.decays[outcome] * state + (
                    1.0 - self.decays[outcome]
                ) * self.saturations[outcome]
            return state


else:  # pragma: no cover
    DistilledStudentTorchPolicy = None  # type: ignore[assignment]
    FrozenHandcraftedRecurrencePolicy = None  # type: ignore[assignment]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError(f"parent artifact is not PASS: {path}")
    return payload


def _dataclass_kwargs(cls: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in allowed}


def _load_parents(config: GainRetentionConfig) -> dict[str, Any]:
    th = _require_torch()
    dtype = th.float64 if config.real_dtype == "float64" else th.float32

    teacher, teacher_report = load_and_verify_teacher_checkpoint(
        TEACHER_CHECKPOINT, TEACHER_ARTIFACT
    )
    teacher.to(device=config.device, dtype=dtype).eval()
    teacher_config = BoundedResidualTeacherConfig(**teacher_report["config"])

    mf_report = _load_json(MF_ARTIFACT)
    if mf_report["implementation_sha256"] != mf_implementation_sha256():
        raise ValueError("T3.2.7 MF artifact is stale")
    if mf_report["checkpoint"]["sha256"] != _sha256(MF_CHECKPOINT):
        raise ValueError("T3.2.7 MF checkpoint hash mismatch")
    mf_checkpoint = th.load(MF_CHECKPOINT, map_location="cpu", weights_only=False)
    mf_models = []
    for item in mf_checkpoint["models"]:
        model = build_budget_matched_policy(
            device=config.device, dtype=dtype, seed=int(item["training_seed"])
        )
        model.load_state_dict(item["state_dict"])
        model.eval()
        if state_dict_sha256(item["state_dict"]) != item["checkpoint_sha256"]:
            raise ValueError("T3.2.7 MF state hash mismatch")
        mf_models.append((int(item["training_seed"]), model))

    handcrafted_report = _load_json(HANDCRAFTED_ARTIFACT)
    if handcrafted_report["implementation_sha256"] != handcrafted_implementation_sha256():
        raise ValueError("T3.2.10 handcrafted artifact is stale")
    if handcrafted_report["artifacts"]["checkpoint_sha256"] != _sha256(
        HANDCRAFTED_CHECKPOINT
    ):
        raise ValueError("T3.2.10 handcrafted checkpoint hash mismatch")
    handcrafted_checkpoint = th.load(
        HANDCRAFTED_CHECKPOINT, map_location="cpu", weights_only=False
    )
    handcrafted_config = ExponentialRecurrenceConfig(
        **_dataclass_kwargs(ExponentialRecurrenceConfig, handcrafted_report["config"])
    )
    handcrafted_config = replace(
        handcrafted_config, device=config.device, real_dtype=config.real_dtype
    )
    handcrafted_source = load_handcrafted_policy(
        handcrafted_config, handcrafted_checkpoint["selected_state_dict"]
    )
    handcrafted = FrozenHandcraftedRecurrencePolicy(handcrafted_source).eval()

    student_report = _load_json(STUDENT_REPORT)
    if student_report["implementation_sha256"] != student_implementation_sha256():
        raise ValueError("T4.4.3 student report is stale")
    student_artifact = LowDimensionalRecurrenceArtifact.from_dict(
        json.loads(DEFAULT_STUDENT.read_text(encoding="utf-8"))
    )
    if student_report["student_artifact"]["file_sha256"] != _sha256(DEFAULT_STUDENT):
        raise ValueError("T4.4.3 student file hash mismatch")
    student = DistilledStudentTorchPolicy(
        student_artifact, device=config.device, dtype=dtype
    ).eval()

    lookup_report = _load_json(LOOKUP_ARTIFACT)
    if lookup_report["implementation_sha256"] != lookup_implementation_sha256():
        raise ValueError("T3.2.9 control-oracle artifact is stale")
    if lookup_report["checkpoint"]["sha256"] != _sha256(LOOKUP_CHECKPOINT):
        raise ValueError("T3.2.9 control-oracle checkpoint hash mismatch")
    lookup_checkpoint = th.load(LOOKUP_CHECKPOINT, map_location="cpu", weights_only=False)
    lookup_config = TrajectoryLookupConfig(
        **_dataclass_kwargs(TrajectoryLookupConfig, lookup_report["config"])
    )
    lookup_config = replace(
        lookup_config, device=config.device, real_dtype=config.real_dtype
    )
    lookup = load_lookup_policy(lookup_config, lookup_checkpoint["lookup"]).eval()

    return {
        "teacher": teacher,
        "teacher_report": teacher_report,
        "teacher_config": teacher_config,
        "mf_models": mf_models,
        "mf_report": mf_report,
        "handcrafted": handcrafted,
        "handcrafted_report": handcrafted_report,
        "student": student,
        "student_artifact": student_artifact,
        "student_report": student_report,
        "lookup": lookup,
        "lookup_report": lookup_report,
    }


def _directional_config(
    base: BoundedResidualTeacherConfig, actual: GainRetentionConfig
) -> DirectionalRankingConfig:
    return replace(
        base.directional_config(),
        cutoff=actual.cutoff,
        confirmation_cutoff=actual.confirmation_cutoff,
        full_cycles=actual.full_cycles,
        test_batch_size=actual.primary_batch_size,
        confirmation_batch_size=actual.confirmation_batch_size,
        test_seeds=actual.primary_seeds,
        confirmation_seeds=actual.confirmation_seeds,
        bootstrap_seed=actual.bootstrap_seed,
        bootstrap_repetitions=actual.bootstrap_repetitions,
        device=actual.device,
        real_dtype=actual.real_dtype,
    )


def _evaluate_stochastic(
    name: str,
    policy: Any | None,
    config: DirectionalRankingConfig,
    *,
    cutoff: int,
    batch_size: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    strategy = "standard" if policy is None else "mf" if name.startswith("mf_agent") else "nmf"
    payload = evaluate_policy(
        strategy, policy, config, cutoff=cutoff, batch_size=batch_size, seeds=seeds
    )
    payload["strategy"] = name
    return payload


def _weighted_mean(probability: Any, values: Any) -> Any:
    shape = (probability.shape[0],) + (1,) * (values.ndim - 1)
    return torch.sum(probability.reshape(shape) * values, dim=0)


def _evaluate_exact(
    name: str,
    policy: Any | None,
    config: GainRetentionConfig,
    *,
    cutoff: int,
) -> dict[str, Any]:
    th = _require_torch()
    outcomes = enumerate_terminal_trajectories(
        2 * config.exact_control_oracle_cycles, device=config.device
    )
    simulator = DifferentiableSBSTrajectorySimulator(
        DifferentiableSBSConfig(
            cutoff=cutoff,
            full_cycles=config.exact_control_oracle_cycles,
            batch_size=outcomes.shape[0],
            projector_delta=0.34,
            cavity_lifetime_us=245.0,
            ancilla_t1_us=50.0,
            ancilla_t2_us=60.0,
            device=config.device,
            real_dtype=config.real_dtype,
        )
    )
    if policy is not None:
        policy.eval()
    with th.no_grad():
        result = simulator.run(
            control_policy=policy,
            forced_outcomes=outcomes,
            seed=0,
            record_cycle_metrics=True,
        )
        if result.cycle_fidelities is None or result.cycle_logical_z_signal is None:
            raise RuntimeError("exact lane did not return cycle metrics")
        probability = result.trajectory_probability
        fidelity_curve = _weighted_mean(probability, result.cycle_fidelities)
        logical_curve = _weighted_mean(probability, result.cycle_logical_z_signal)
        survival_curve = _weighted_mean(probability, result.cycle_code_survival)
        fidelity = _effective_lifetime(fidelity_curve.detach().cpu().numpy())
        logical = _effective_lifetime(logical_curve.detach().cpu().numpy())
        nominal = nominal_sbs_parameters(device=config.device, dtype=_torch_dtype(config.real_dtype))
        residual = result.physical_controls - nominal[None, None, :]
        slew = result.physical_controls[:, 1:, :] - result.physical_controls[:, :-1, :]
        residual_rms = th.sqrt(_weighted_mean(probability, th.mean(residual**2, dim=(1, 2))))
        slew_rms = th.sqrt(_weighted_mean(probability, th.mean(slew**2, dim=(1, 2))))
        bounds = th.tensor([2.0] * 14 + [1.0], device=config.device, dtype=residual.dtype)
        bound_violation = th.max(th.clamp(th.abs(residual) - bounds, min=0.0))
        ground_by_half = _weighted_mean(
            probability, (result.outcomes == 0).to(probability.dtype)
        )
        branch_rows = []
        for index in range(outcomes.shape[0]):
            branch_rows.append(
                {
                    "trajectory": "".join(
                        "g" if int(value) == 0 else "e"
                        for value in outcomes[index].detach().cpu().tolist()
                    ),
                    "probability": float(probability[index].detach().cpu()),
                    "terminal_fidelity": float(result.reward[index].detach().cpu()),
                    "terminal_logical_z": float(
                        result.cycle_logical_z_signal[index, -1].detach().cpu()
                    ),
                    "terminal_code_survival": float(
                        result.cycle_code_survival[index, -1].detach().cpu()
                    ),
                }
            )
    return {
        "strategy": name,
        "cutoff": cutoff,
        "full_cycles": config.exact_control_oracle_cycles,
        "branch_count": int(outcomes.shape[0]),
        "trajectory_probability_sum": float(probability.sum().detach().cpu()),
        "fidelity_curve": fidelity_curve.detach().cpu().tolist(),
        "logical_z_curve": logical_curve.detach().cpu().tolist(),
        "code_survival_curve": survival_curve.detach().cpu().tolist(),
        "fidelity": fidelity,
        "logical_z": logical,
        "selection_score": float(
            0.5 * (fidelity["normalized_auc"] + logical["normalized_auc"])
        ),
        "terminal_fidelity": float(fidelity_curve[-1].detach().cpu()),
        "terminal_logical_z": float(logical_curve[-1].detach().cpu()),
        "terminal_code_survival": float(survival_curve[-1].detach().cpu()),
        "mean_ground_outcome_probability": float(ground_by_half.mean().detach().cpu()),
        "ground_probability_by_half_cycle": ground_by_half.detach().cpu().tolist(),
        "expected_e_events": float((1.0 - ground_by_half).sum().detach().cpu()),
        "mean_control_residual_rms": float(residual_rms.detach().cpu()),
        "mean_control_slew_rms": float(slew_rms.detach().cpu()),
        "maximum_action_bound_violation": float(bound_violation.detach().cpu()),
        "maximum_trace_error": result.maximum_trace_error,
        "maximum_hermiticity_error": result.maximum_hermiticity_error,
        "minimum_final_eigenvalue": result.minimum_final_eigenvalue,
        "branch_rows": branch_rows,
    }


STOCHASTIC_METRICS = {
    "selection_score": lambda row: 0.5
    * (row["fidelity"]["normalized_auc"] + row["logical_z"]["normalized_auc"]),
    "fidelity_effective_lifetime_cycles": lambda row: row["fidelity"][
        "effective_lifetime_cycles"
    ],
    "logical_z_effective_lifetime_cycles": lambda row: row["logical_z"][
        "effective_lifetime_cycles"
    ],
}


def _metric_values(evaluation: Mapping[str, Any], metric: str) -> np.ndarray:
    return np.asarray(
        [STOCHASTIC_METRICS[metric](row) for row in evaluation["per_seed"]],
        dtype=np.float64,
    )


def _retention_bootstrap(
    standard: Mapping[str, Any],
    teacher: Mapping[str, Any],
    student: Mapping[str, Any],
    metric: str,
    *,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    standard_values = _metric_values(standard, metric)
    teacher_values = _metric_values(teacher, metric)
    student_values = _metric_values(student, metric)
    if not (
        standard_values.shape == teacher_values.shape == student_values.shape
        and standard_values.size >= 2
    ):
        raise ValueError("paired retention arrays are incomplete")
    denominator = float(np.mean(teacher_values) - np.mean(standard_values))
    if denominator <= 0.0:
        return {
            "metric": metric,
            "defined": False,
            "point_retention_fraction": None,
            "ci_95": [None, None],
            "teacher_gain": denominator,
            "student_gain": float(np.mean(student_values) - np.mean(standard_values)),
            "paired_seed_count": int(standard_values.size),
            "bootstrap_repetitions": repetitions,
            "positive_teacher_gain_bootstrap_fraction": 0.0,
        }
    point = float((np.mean(student_values) - np.mean(standard_values)) / denominator)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, standard_values.size, size=(repetitions, standard_values.size)
    )
    standard_means = np.mean(standard_values[indices], axis=1)
    teacher_means = np.mean(teacher_values[indices], axis=1)
    student_means = np.mean(student_values[indices], axis=1)
    denominators = teacher_means - standard_means
    valid = denominators > 0.0
    values = (student_means[valid] - standard_means[valid]) / denominators[valid]
    ci = (
        [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        if values.size
        else [None, None]
    )
    return {
        "metric": metric,
        "defined": bool(values.size),
        "point_retention_fraction": point,
        "ci_95": ci,
        "teacher_gain": denominator,
        "student_gain": float(np.mean(student_values) - np.mean(standard_values)),
        "paired_seed_count": int(standard_values.size),
        "bootstrap_repetitions": repetitions,
        "positive_teacher_gain_bootstrap_fraction": float(np.mean(valid)),
    }


def _exact_retention(
    standard: Mapping[str, Any],
    teacher: Mapping[str, Any],
    student: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    accessors = {
        "selection_score": lambda row: row["selection_score"],
        "fidelity_effective_lifetime_cycles": lambda row: row["fidelity"][
            "effective_lifetime_cycles"
        ],
        "logical_z_effective_lifetime_cycles": lambda row: row["logical_z"][
            "effective_lifetime_cycles"
        ],
        "terminal_fidelity": lambda row: row["terminal_fidelity"],
    }
    result: dict[str, dict[str, Any]] = {}
    for metric, accessor in accessors.items():
        denominator = float(accessor(teacher) - accessor(standard))
        student_gain = float(accessor(student) - accessor(standard))
        result[metric] = {
            "defined": denominator > 0.0,
            "teacher_gain": denominator,
            "student_gain": student_gain,
            "retention_fraction": (
                float(student_gain / denominator) if denominator > 0.0 else None
            ),
        }
    return result


def _aggregate_mf(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = tuple(next(iter(evaluations))["metric_means"])
    auxiliary_names = tuple(next(iter(evaluations))["auxiliary_means"])
    return {
        "agent_count": len(evaluations),
        "training_seeds": [int(item["strategy"].split("_")[-1]) for item in evaluations],
        "metric_mean_across_agents": {
            name: float(np.mean([item["metric_means"][name] for item in evaluations]))
            for name in metric_names
        },
        "metric_median_across_agents": {
            name: float(np.median([item["metric_means"][name] for item in evaluations]))
            for name in metric_names
        },
        "auxiliary_mean_across_agents": {
            name: float(np.mean([item["auxiliary_means"][name] for item in evaluations]))
            for name in auxiliary_names
        },
        "agents": list(evaluations),
    }


def _cost_table(student: LowDimensionalRecurrenceArtifact) -> list[dict[str, Any]]:
    return [
        {
            "strategy": "standard",
            "trainable_scalars": 0,
            "stored_scalars": 15,
            "persistent_state_scalars": 0,
            "analytic_macs_per_half_cycle": 0,
            "cost_scope": "fifteen fixed nominal control constants",
            "deployable": True,
        },
        {
            "strategy": "exact_budget_mf",
            "trainable_scalars": MF_COMPUTE_CONTRACT.total_parameter_count,
            "stored_scalars": MF_COMPUTE_CONTRACT.total_parameter_count,
            "persistent_state_scalars": 0,
            "analytic_macs_per_half_cycle": MF_COMPUTE_CONTRACT.total_dense_macs,
            "cost_scope": "float learned model; no synthesis or board timing",
            "deployable": False,
        },
        {
            "strategy": "fresh_gru_teacher",
            "trainable_scalars": 72_853,
            "stored_scalars": 72_853,
            "persistent_state_scalars": 10,
            "analytic_macs_per_half_cycle": 72_266,
            "cost_scope": "float offline teacher; not an online FPGA path",
            "deployable": False,
        },
        {
            "strategy": "handcrafted_recurrence",
            "trainable_scalars": 75,
            "stored_scalars": 105,
            "persistent_state_scalars": 15,
            "analytic_macs_per_half_cycle": 45,
            "cost_scope": "float affine recurrence; ten-cycle use is frozen extrapolation",
            "deployable": False,
        },
        {
            "strategy": "distilled_student",
            "trainable_scalars": student.resource_profile.stored_trainable_scalars,
            "stored_scalars": student.resource_profile.stored_trainable_scalars,
            "persistent_state_scalars": student.resource_profile.persistent_state_scalars,
            "analytic_macs_per_half_cycle": student.resource_profile.multiply_adds_per_healthy_step,
            "cost_scope": "pure NumPy float artifact; quantized RTL and timing not measured",
            "deployable": False,
        },
        {
            "strategy": "finite_horizon_control_oracle",
            "trainable_scalars": 225,
            "stored_scalars": 225,
            "persistent_state_scalars": 4,
            "analytic_macs_per_half_cycle": 0,
            "cost_scope": "two-cycle 15-node table read only; address and exact optimization excluded; exponential growth",
            "deployable": False,
        },
    ]


SOURCE_COLUMNS = (
    "row_type",
    "lane",
    "cutoff",
    "strategy",
    "agent_seed",
    "evaluation_seed",
    "trajectory",
    "probability",
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
            writer.writerow({key: row.get(key, "") for key in SOURCE_COLUMNS})
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_teacher_student_gain_retention(
    config: GainRetentionConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
    production: bool = True,
) -> dict[str, Any]:
    th = _require_torch()
    actual = config or GainRetentionConfig()
    if production:
        validate_production_design(actual)
    if actual.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    parents = _load_parents(actual)
    directional = _directional_config(parents["teacher_config"], actual)

    modes: list[tuple[str, Any | None]] = [("standard", None)]
    modes.extend(
        (f"mf_agent_{seed}", model) for seed, model in parents["mf_models"]
    )
    modes.extend(
        (
            ("teacher", parents["teacher"]),
            ("handcrafted_recurrence", parents["handcrafted"]),
            ("distilled_student", parents["student"]),
        )
    )

    stochastic: dict[str, Any] = {}
    lane_specs = {
        "primary": (actual.cutoff, actual.primary_batch_size, actual.primary_seeds),
        "confirmation": (
            actual.confirmation_cutoff,
            actual.confirmation_batch_size,
            actual.confirmation_seeds,
        ),
    }
    for lane, (cutoff, batch_size, seeds) in lane_specs.items():
        evaluations = {
            name: _evaluate_stochastic(
                name,
                policy,
                directional,
                cutoff=cutoff,
                batch_size=batch_size,
                seeds=seeds,
            )
            for name, policy in modes
        }
        mf_values = [
            evaluations[name] for name in evaluations if name.startswith("mf_agent_")
        ]
        stochastic[lane] = {
            "cutoff": cutoff,
            "full_cycles": actual.full_cycles,
            "batch_size_per_seed": batch_size,
            "seeds": list(seeds),
            "standard": evaluations["standard"],
            "mf_all_agents": _aggregate_mf(mf_values),
            "teacher": evaluations["teacher"],
            "handcrafted_recurrence": evaluations["handcrafted_recurrence"],
            "distilled_student": evaluations["distilled_student"],
        }

    exact: dict[str, Any] = {}
    exact_modes = list(modes) + [("control_oracle", parents["lookup"])]
    for cutoff in (actual.cutoff, actual.confirmation_cutoff):
        evaluations = {
            name: _evaluate_exact(name, policy, actual, cutoff=cutoff)
            for name, policy in exact_modes
        }
        mf_values = [
            evaluations[name] for name in evaluations if name.startswith("mf_agent_")
        ]
        exact[str(cutoff)] = {
            "cutoff": cutoff,
            "full_cycles": actual.exact_control_oracle_cycles,
            "control_oracle_is_horizon_bound": True,
            "standard": evaluations["standard"],
            "mf_all_agents": {
                "agent_count": len(mf_values),
                "terminal_fidelity_mean": float(
                    np.mean([item["terminal_fidelity"] for item in mf_values])
                ),
                "selection_score_mean": float(
                    np.mean([item["selection_score"] for item in mf_values])
                ),
                "mean_ground_outcome_probability": float(
                    np.mean(
                        [item["mean_ground_outcome_probability"] for item in mf_values]
                    )
                ),
                "agents": mf_values,
            },
            "teacher": evaluations["teacher"],
            "handcrafted_recurrence": evaluations["handcrafted_recurrence"],
            "distilled_student": evaluations["distilled_student"],
            "control_oracle": evaluations["control_oracle"],
        }

    stochastic_retention = {}
    for lane_index, lane in enumerate(("primary", "confirmation")):
        content = stochastic[lane]
        stochastic_retention[lane] = {
            metric: _retention_bootstrap(
                content["standard"],
                content["teacher"],
                content["distilled_student"],
                metric,
                seed=actual.bootstrap_seed + 101 * lane_index + metric_index,
                repetitions=actual.bootstrap_repetitions,
            )
            for metric_index, metric in enumerate(STOCHASTIC_METRICS)
        }
    exact_retention = {
        key: _exact_retention(
            content["standard"], content["teacher"], content["distilled_student"]
        )
        for key, content in exact.items()
    }

    costs = _cost_table(parents["student_artifact"])
    rows: list[dict[str, Any]] = []
    for lane, content in stochastic.items():
        evaluations = [
            content["standard"],
            *content["mf_all_agents"]["agents"],
            content["teacher"],
            content["handcrafted_recurrence"],
            content["distilled_student"],
        ]
        for evaluation in evaluations:
            agent_seed = (
                evaluation["strategy"].split("_")[-1]
                if evaluation["strategy"].startswith("mf_agent_")
                else ""
            )
            for per_seed in evaluation["per_seed"]:
                rows.append(
                    {
                        "row_type": "stochastic_seed_summary",
                        "lane": lane,
                        "cutoff": content["cutoff"],
                        "strategy": evaluation["strategy"],
                        "agent_seed": agent_seed,
                        "evaluation_seed": per_seed["seed"],
                        "trajectory": "",
                        "probability": "",
                        "metric": "selection_score",
                        "value": STOCHASTIC_METRICS["selection_score"](per_seed),
                        "detail_json": json.dumps(per_seed, sort_keys=True),
                    }
                )
    for cutoff, content in exact.items():
        evaluations = [
            content["standard"],
            *content["mf_all_agents"]["agents"],
            content["teacher"],
            content["handcrafted_recurrence"],
            content["distilled_student"],
            content["control_oracle"],
        ]
        for evaluation in evaluations:
            agent_seed = (
                evaluation["strategy"].split("_")[-1]
                if evaluation["strategy"].startswith("mf_agent_")
                else ""
            )
            for branch in evaluation["branch_rows"]:
                rows.append(
                    {
                        "row_type": "exact_branch",
                        "lane": "two_cycle_exact",
                        "cutoff": cutoff,
                        "strategy": evaluation["strategy"],
                        "agent_seed": agent_seed,
                        "evaluation_seed": "",
                        "trajectory": branch["trajectory"],
                        "probability": branch["probability"],
                        "metric": "terminal_fidelity",
                        "value": branch["terminal_fidelity"],
                        "detail_json": json.dumps(branch, sort_keys=True),
                    }
                )
    for lane, metrics in stochastic_retention.items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "row_type": "retention_gate",
                    "lane": lane,
                    "cutoff": stochastic[lane]["cutoff"],
                    "strategy": "distilled_student_vs_teacher",
                    "agent_seed": "",
                    "evaluation_seed": "",
                    "trajectory": "",
                    "probability": "",
                    "metric": metric,
                    "value": value["point_retention_fraction"],
                    "detail_json": json.dumps(value, sort_keys=True),
                }
            )
    for cutoff, metrics in exact_retention.items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "row_type": "retention_gate",
                    "lane": "two_cycle_exact",
                    "cutoff": cutoff,
                    "strategy": "distilled_student_vs_teacher",
                    "agent_seed": "",
                    "evaluation_seed": "",
                    "trajectory": "",
                    "probability": "",
                    "metric": metric,
                    "value": value["retention_fraction"],
                    "detail_json": json.dumps(value, sort_keys=True),
                }
            )
    for cost in costs:
        rows.append(
            {
                "row_type": "cost_summary",
                "lane": "resource_proxy",
                "cutoff": "",
                "strategy": cost["strategy"],
                "agent_seed": "",
                "evaluation_seed": "",
                "trajectory": "",
                "probability": "",
                "metric": "stored_scalars",
                "value": cost["stored_scalars"],
                "detail_json": json.dumps(cost, sort_keys=True),
            }
        )
    source = Path(source_data_path)
    _write_source_data(source, rows)

    parent_seed_sets = [
        set(parents["teacher_report"]["config"][name])
        for name in (
            "restart_seeds",
            "validation_seeds",
            "evaluation_seeds",
            "confirmation_seeds",
        )
    ]
    parent_seed_sets.extend(
        {
            int(parents["student_report"]["dataset"]["split_seeds"][name])
            for name in parents["student_report"]["dataset"]["split_seeds"]
        }
        for _ in (0,)
    )
    current_seeds = set(actual.primary_seeds) | set(actual.confirmation_seeds)
    retention_values = [
        value
        for lane in stochastic_retention.values()
        for value in lane.values()
    ]
    exact_values = [value for lane in exact_retention.values() for value in lane.values()]
    pg_differences = {
        lane: abs(
            content["distilled_student"]["auxiliary_means"][
                "mean_ground_outcome_probability"
            ]
            - content["teacher"]["auxiliary_means"][
                "mean_ground_outcome_probability"
            ]
        )
        for lane, content in stochastic.items()
    }
    exact_pg_differences = {
        cutoff: abs(
            content["distilled_student"]["mean_ground_outcome_probability"]
            - content["teacher"]["mean_ground_outcome_probability"]
        )
        for cutoff, content in exact.items()
    }
    burden_summary: dict[str, Any] = {"stochastic": {}, "exact": {}}
    for lane, content in stochastic.items():
        strategy_values = {
            "standard": content["standard"]["auxiliary_means"][
                "mean_ground_outcome_probability"
            ],
            "exact_budget_mf_agent_mean": content["mf_all_agents"][
                "auxiliary_mean_across_agents"
            ]["mean_ground_outcome_probability"],
            "teacher": content["teacher"]["auxiliary_means"][
                "mean_ground_outcome_probability"
            ],
            "handcrafted_recurrence": content["handcrafted_recurrence"][
                "auxiliary_means"
            ]["mean_ground_outcome_probability"],
            "distilled_student": content["distilled_student"]["auxiliary_means"][
                "mean_ground_outcome_probability"
            ],
        }
        burden_summary["stochastic"][lane] = {
            name: {
                "observed_ground_fraction": float(value),
                "observed_e_fraction": float(1.0 - value),
                "expected_e_events_from_observed_fraction": float(
                    2 * actual.full_cycles * (1.0 - value)
                ),
                "multilevel_leakage_events": None,
            }
            for name, value in strategy_values.items()
        }
    for cutoff, content in exact.items():
        burden_summary["exact"][cutoff] = {
            name: {
                "expected_ground_fraction": float(
                    evaluation["mean_ground_outcome_probability"]
                ),
                "expected_e_fraction": float(
                    1.0 - evaluation["mean_ground_outcome_probability"]
                ),
                "expected_e_events": float(evaluation["expected_e_events"]),
                "multilevel_leakage_events": None,
            }
            for name, evaluation in (
                ("standard", content["standard"]),
                ("teacher", content["teacher"]),
                ("handcrafted_recurrence", content["handcrafted_recurrence"]),
                ("distilled_student", content["distilled_student"]),
                ("control_oracle", content["control_oracle"]),
            )
        }
    expected_rows = (
        (1 + 5 + 3)
        * (len(actual.primary_seeds) + len(actual.confirmation_seeds))
        + len(exact) * (1 + 5 + 4) * 16
        + sum(len(metrics) for metrics in stochastic_retention.values())
        + sum(len(metrics) for metrics in exact_retention.values())
        + len(costs)
    )
    gates = {
        "all_parent_artifacts_and_checkpoints_are_current_passes": True,
        "new_stochastic_seeds_are_disjoint_from_teacher_and_student_fitting": all(
            not (current_seeds & parent) for parent in parent_seed_sets
        ),
        "ten_cycle_lane_retains_standard_all_five_mf_teacher_handcrafted_and_student": all(
            content["mf_all_agents"]["agent_count"] == 5
            and content["full_cycles"] == 10
            for content in stochastic.values()
        ),
        "two_cycle_exact_lane_includes_all_comparators_and_horizon_bound_oracle": all(
            content["control_oracle_is_horizon_bound"]
            and content["control_oracle"]["branch_count"] == 16
            and content["mf_all_agents"]["agent_count"] == 5
            for content in exact.values()
        ),
        "student_point_gain_retention_meets_frozen_ninety_percent_threshold": all(
            value["defined"]
            and value["point_retention_fraction"] is not None
            and value["point_retention_fraction"] >= actual.minimum_gain_retention_fraction
            for value in retention_values
        ),
        "student_gain_retention_lower_confidence_bounds_meet_threshold": all(
            value["defined"]
            and value["positive_teacher_gain_bootstrap_fraction"] == 1.0
            and value["ci_95"][0] is not None
            and value["ci_95"][0] >= actual.minimum_gain_retention_ci_lower
            for value in retention_values
        ),
        "exact_two_cycle_student_gain_retention_meets_threshold": all(
            value["defined"]
            and value["retention_fraction"] is not None
            and value["retention_fraction"] >= actual.minimum_gain_retention_fraction
            for value in exact_values
        ),
        "teacher_student_pg_difference_stays_inside_frozen_tolerance": all(
            value <= actual.maximum_teacher_student_pg_difference
            for value in (*pg_differences.values(), *exact_pg_differences.values())
        ),
        "student_actions_obey_all_fifteen_hard_bounds": all(
            content["distilled_student"]["maximum_action_bound_violation"] == 0.0
            for content in exact.values()
        ),
        "all_physics_lanes_preserve_trace_hermiticity_and_positive_tolerance": all(
            evaluation["maximum_trace_error"] < 1.0e-12
            and evaluation["maximum_hermiticity_error"] < 1.0e-12
            and evaluation["minimum_final_eigenvalue"] > -1.0e-12
            for content in exact.values()
            for evaluation in (
                content["standard"],
                *content["mf_all_agents"]["agents"],
                content["teacher"],
                content["handcrafted_recurrence"],
                content["distilled_student"],
                content["control_oracle"],
            )
        ),
        "all_five_mf_agents_are_reported_without_test_postselection": all(
            content["mf_all_agents"]["agent_count"] == 5 for content in stochastic.values()
        ),
        "student_compresses_teacher_stored_scalars_by_more_than_ninety_nine_percent": (
            1.0
            - parents["student_artifact"].resource_profile.stored_trainable_scalars
            / 72_853
            > 0.99
        ),
        "cost_table_keeps_float_analytic_and_hardware_evidence_separate": (
            len(costs) == 6
            and all(not row["deployable"] for row in costs if row["strategy"] != "standard")
        ),
        "source_data_contains_every_seed_branch_retention_and_cost_row": (
            len(rows) == expected_rows
            and {row["row_type"] for row in rows}
            == {
                "stochastic_seed_summary",
                "exact_branch",
                "retention_gate",
                "cost_summary",
            }
        ),
        "leakage_is_explicitly_unavailable_not_silently_encoded_as_e": True,
        "g_e_and_unavailable_leakage_burdens_are_explicit_for_every_lane": (
            set(burden_summary["stochastic"]) == {"primary", "confirmation"}
            and set(burden_summary["exact"]) == {str(actual.cutoff), str(actual.confirmation_cutoff)}
            and all(
                row["multilevel_leakage_events"] is None
                for family in burden_summary.values()
                for lane in family.values()
                for row in lane.values()
            )
        ),
        "control_oracle_is_never_extrapolated_into_ten_cycle_lane": all(
            "control_oracle" not in content for content in stochastic.values()
        ),
        "claim_boundary_requires_t4_4_5_before_nmf_promotion": True,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    status = "PASS" if all(gates.values()) else "FAIL"

    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": status,
        "scope": SCOPE,
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": implementation_sha256(),
        "config": asdict(actual),
        "config_contract_hash": actual.contract_hash,
        "parent_provenance": {
            "teacher_artifact_sha256": _sha256(TEACHER_ARTIFACT),
            "teacher_checkpoint_sha256": _sha256(TEACHER_CHECKPOINT),
            "teacher_state_sha256": parents["teacher_report"]["checkpoint"][
                "selected_state_sha256"
            ],
            "mf_artifact_sha256": _sha256(MF_ARTIFACT),
            "mf_checkpoint_sha256": _sha256(MF_CHECKPOINT),
            "handcrafted_artifact_sha256": _sha256(HANDCRAFTED_ARTIFACT),
            "handcrafted_checkpoint_sha256": _sha256(HANDCRAFTED_CHECKPOINT),
            "student_report_sha256": _sha256(STUDENT_REPORT),
            "student_artifact_sha256": _sha256(DEFAULT_STUDENT),
            "control_oracle_artifact_sha256": _sha256(LOOKUP_ARTIFACT),
            "control_oracle_checkpoint_sha256": _sha256(LOOKUP_CHECKPOINT),
        },
        "retention_threshold": {
            "point_fraction": actual.minimum_gain_retention_fraction,
            "paired_bootstrap_ci_lower": actual.minimum_gain_retention_ci_lower,
            "applies_to": list(STOCHASTIC_METRICS),
            "frozen_before_physical_evaluation": True,
        },
        "stochastic_ten_cycle": stochastic,
        "stochastic_retention": stochastic_retention,
        "exact_two_cycle": exact,
        "exact_retention": exact_retention,
        "teacher_student_pg_absolute_difference": {
            "stochastic": pg_differences,
            "exact": exact_pg_differences,
        },
        "burden_summary": burden_summary,
        "leakage_burden": {
            "native_multilevel_leakage_available": False,
            "reported_value": None,
            "e_burden_proxy": "one minus native two-level g fraction; never labelled leakage",
            "student_failure_policy": "external leakage token resets state and returns exact zero residual",
        },
        "costs": costs,
        "source_data": {
            "path": source.as_posix(),
            "sha256": _sha256(source),
            "row_count": len(rows),
            "expected_row_count": expected_rows,
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
            "real_dtype": actual.real_dtype,
            "torch_version": th.__version__,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "claim_boundary": {
            "allowed": (
                "student retention of frozen-teacher finite-model physical trajectory gain "
                "on paired unseen ten-cycle seeds and exact two-cycle histories"
            ),
            "forbidden": (
                "T4.4.5 NMF promotion before branch freeze; multilevel leakage robustness; "
                "long-horizon OOD; pulse device quantized RTL FPGA or board performance"
            ),
            "control_oracle_boundary": (
                "the empirical lookup reference appears only in the exact two-cycle lane and "
                "is neither globally certified nor extrapolated to ten cycles"
            ),
            "next_gate": (
                "T4.4.5 must consume this machine result and freeze strong or falsified branch"
            ),
        },
    }
    _atomic_json(Path(artifact_path), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pilot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = GainRetentionConfig(device=arguments.device)
    result = run_teacher_student_gain_retention(
        config,
        artifact_path=arguments.artifact,
        source_data_path=arguments.source_data,
        production=not arguments.pilot,
    )
    print(json.dumps({"status": result["status"], "gate_summary": result["gate_summary"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "GainRetentionConfig",
    "PROTOCOL_ID",
    "SCOPE",
    "TASK_ID",
    "implementation_sha256",
    "run_teacher_student_gain_retention",
    "validate_production_design",
]
