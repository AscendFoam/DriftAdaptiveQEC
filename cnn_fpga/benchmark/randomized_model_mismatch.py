"""T5.4.6 randomized, multi-factor model-mismatch validation.

The campaign keeps four native evidence lanes separate:

* finite-cutoff physical control: random 15-vector gate bias, cavity phase
  diffusion, phase-timing redistribution, and T1/T2/lifetime dynamics;
* protocol-native readout: random full 4x3 confusion matrices;
* persistent leakage/reset: random injection and reset-failure rates; and
* frozen syndrome decoders: random drift vectors and unseen dynamics.

No heterogeneous score is formed.  The physical lane alone decides whether
the previously qualified teacher/student branch retains its status under the
pre-registered mismatch distribution.  A negative decision is a valid task
outcome and routes to the already armed MAP-LUT fallback.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import held_out_ood_validation as ood_parent
from cnn_fpga.benchmark import leakage_reset_causal as leakage_parent
from cnn_fpga.benchmark.bounded_residual_rnn_teacher import (
    DEFAULT_ARTIFACT as TEACHER_ARTIFACT,
    DEFAULT_CHECKPOINT as TEACHER_CHECKPOINT,
    load_and_verify_teacher_checkpoint,
)
from cnn_fpga.benchmark.continuous_adaptive_map import _evaluate_seed, _mean_interval
from cnn_fpga.benchmark.teacher_student_gain_retention import (
    DistilledStudentTorchPolicy,
)
from cnn_fpga.control.low_dimensional_recurrence import (
    LowDimensionalRecurrenceArtifact,
)
from cnn_fpga.decoder.periodic_adaptive_map import PeriodicMomentConfig
from physics.constants import LATTICE_CONST
from physics.differentiable_sbs_trajectory import (
    DifferentiableSBSConfig,
    DifferentiableSBSTimingProfile,
    DifferentiableSBSTrajectorySimulator,
    TrajectoryTimingPhase,
)
from physics.drift_processes import DriftState
from physics.nmf_directional_ranking import _effective_lifetime

try:  # The minimal recovery interpreter intentionally has no torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.4.6"
SCHEMA_VERSION = "t5.4.6-randomized-model-mismatch-v1"
PROTOCOL_ID = "RANDOMIZED-MULTIFACTOR-NATIVE-LANE-MISMATCH-V1"
DEFAULT_ARTIFACT = Path("docs/t5_4_6_randomized_model_mismatch.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_4_6_randomized_model_mismatch_source_data.csv")
PRODUCTION_STUDENT = Path("docs/t4_4_3_low_dimensional_student.json")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T4.4.1": Path("docs/t4_4_1_bounded_residual_rnn_teacher_validation.json"),
    "T4.4.3": Path("docs/t4_4_3_low_dimensional_student_validation.json"),
    "T4.4.4": Path("docs/t4_4_4_teacher_student_gain_retention.json"),
    "T4.4.5": Path("docs/t4_4_5_teacher_student_branch_freeze.json"),
    "T5.1.2": Path("docs/t5_1_2_mixed_scenario_matrix.json"),
    "T5.2.2": Path("docs/t5_2_2_ancilla_readout_causal.json"),
    "T5.2.3": Path("docs/t5_2_3_leakage_reset_causal.json"),
    "T5.4.1": Path("docs/t5_4_1_held_out_ood_validation.json"),
    "T5.4.5": Path("docs/t5_4_5_horizon_extrapolation_validation.json"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/randomized_model_mismatch.py"),
    Path("cnn_fpga/benchmark/held_out_ood_validation.py"),
    Path("cnn_fpga/benchmark/leakage_reset_causal.py"),
    Path("cnn_fpga/benchmark/continuous_adaptive_map.py"),
    Path("cnn_fpga/benchmark/bounded_residual_rnn_teacher.py"),
    Path("cnn_fpga/benchmark/teacher_student_gain_retention.py"),
    Path("physics/differentiable_sbs_trajectory.py"),
    Path("physics/drift_processes.py"),
)

PHYSICAL_FAMILIES = (
    "gate_bias_vector",
    "cavity_dephasing",
    "unseen_timing_dynamics",
    "compound_physical",
)
READOUT_FAMILY = "random_full_readout_confusion"
LEAKAGE_FAMILIES = ("random_leakage_injection", "random_reset_failure")
DRIFT_FAMILY = "random_drift_and_unseen_dynamics"
PHYSICAL_STRATEGIES = ("standard", "teacher", "student")


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "T5.4.6 requires PyTorch; use "
            "C:/ProgramData/anaconda3/envs/DLEnv/python.exe"
        )
    return torch


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _seed_stream(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class RandomizedMismatchConfig:
    physical_families: tuple[str, ...] = PHYSICAL_FAMILIES
    physical_cells_per_family: int = 8
    component_cells_per_family: int = 8
    drift_cells: int = 8
    physical_cutoff: int = 12
    physical_full_cycles: int = 10
    physical_batch_size: int = 16
    readout_cycles_per_cell: int = 32_768
    master_seed: int = 546001
    device: str = "cuda"
    real_dtype: str = "float64"
    minimum_retention_median: float = 0.80
    minimum_retention_q1: float = 0.50
    maximum_teacher_student_abs_gap_p95: float = 0.05
    minimum_positive_teacher_gain_fraction: float = 0.75
    minimum_qualifying_teacher_gain: float = 1.0e-4
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if tuple(self.physical_families) != PHYSICAL_FAMILIES:
            raise ValueError("physical mismatch families are frozen")
        for name in (
            "physical_cells_per_family",
            "component_cells_per_family",
            "drift_cells",
            "physical_cutoff",
            "physical_full_cycles",
            "physical_batch_size",
            "readout_cycles_per_cell",
            "master_seed",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if not 6 <= self.physical_cutoff <= 48:
            raise ValueError("physical cutoff must lie in [6,48]")
        if not 2 <= self.physical_full_cycles <= 10:
            raise ValueError(
                "physical full cycles must lie in [2,10] for lifetime fitting"
            )
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        for name in (
            "minimum_retention_median",
            "minimum_retention_q1",
            "maximum_teacher_student_abs_gap_p95",
            "minimum_positive_teacher_gain_fraction",
            "minimum_qualifying_teacher_gain",
            "confidence_level",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence level must lie in (0,1)")

    @property
    def contract_hash(self) -> str:
        return _canonical_sha256(asdict(self))


def validate_production_design(config: RandomizedMismatchConfig) -> None:
    if config.physical_cells_per_family < 8:
        raise ValueError("production requires >=8 random physical cells per family")
    if config.component_cells_per_family < 8 or config.drift_cells < 8:
        raise ValueError("production requires >=8 random cells in every component lane")
    if config.physical_cutoff < 12 or config.physical_full_cycles != 10:
        raise ValueError("production physical lane requires cutoff>=12 and 10 cycles")
    if config.physical_batch_size < 16:
        raise ValueError("production physical lane requires >=16 trajectories per cell")
    if config.readout_cycles_per_cell < 32_768:
        raise ValueError("production readout cells require >=32768 cycles")
    if config.real_dtype != "float64":
        raise ValueError("production physical lane requires float64")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _parent_pass(payload: Mapping[str, Any]) -> bool:
    if payload.get("status") == "PASS" or payload.get("passed") is True:
        return True
    gate = payload.get("gate")
    return bool(isinstance(gate, Mapping) and gate.get("passed") is True)


def _parent_bindings() -> list[dict[str, Any]]:
    return [
        {
            "task_id": task_id,
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "machine_pass": _parent_pass(_load_json(path)),
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    ]


def _implementation_bindings() -> list[dict[str, str]]:
    return [
        {"path": path.as_posix(), "sha256": _sha256(path)}
        for path in IMPLEMENTATION_PATHS
    ]


def _extract_seeds(value: Any, key_path: str = "") -> set[int]:
    result: set[int] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{key_path}.{key}" if key_path else str(key)
            result.update(_extract_seeds(child, path))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.update(_extract_seeds(child, key_path))
    elif "seed" in key_path.lower() and isinstance(value, int) and not isinstance(value, bool):
        result.add(int(value))
    return result


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


SOURCE_COLUMNS = (
    "row_type",
    "lane",
    "family",
    "cell_id",
    "seed",
    "strategy",
    "metric",
    "value",
    "detail_json",
)


def _write_source(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = _repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in SOURCE_COLUMNS})
    os.replace(temporary, target)


def _duration_profile(rng: np.random.Generator) -> DifferentiableSBSTimingProfile:
    nominal = np.asarray((100, 500, 700, 300, 100, 2300, 1000), dtype=np.float64)
    weights = nominal * np.exp(rng.normal(0.0, 0.40, size=nominal.size))
    available = 5000 - 20 * nominal.size
    scaled = np.floor(20 + available * weights / np.sum(weights)).astype(np.int64)
    remainder = 5000 - int(np.sum(scaled))
    order = np.argsort(-(weights / np.sum(weights) * available % 1.0))
    for index in order[:remainder]:
        scaled[index] += 1
    phase_ids = (
        "entering_cycle",
        "layer_1",
        "layer_2",
        "layer_3",
        "layer_4",
        "measurement_and_reset",
        "virtual_rotation_and_idle",
    )
    gates = (
        "none",
        "R1_then_ECD1",
        "R2_then_ECD2",
        "R3_then_ECD3",
        "R4_then_fixed_D_alpha",
        "measure_g_or_e_then_reset",
        "VR",
    )
    return DifferentiableSBSTimingProfile(
        phases=tuple(
            TrajectoryTimingPhase(name, int(duration), gate)
            for name, duration, gate in zip(phase_ids, scaled, gates, strict=True)
        )
    )


def _physical_cell(
    family: str, cell_index: int, config: RandomizedMismatchConfig
) -> dict[str, Any]:
    seed = _seed_stream(config.master_seed, f"physical:{family}:{cell_index}")
    rng = np.random.default_rng(seed)
    gate_scale = np.asarray(
        (0.08, 0.08, 0.05, 0.05) * 3 + (0.08, 0.08, 0.12),
        dtype=np.float64,
    )
    gate_bias = np.zeros(15, dtype=np.float64)
    cavity_tphi_us: float | None = None
    timing = DifferentiableSBSTimingProfile()
    cavity_lifetime_us = 245.0
    ancilla_t1_us = 50.0
    ancilla_t2_us = 60.0
    if family in {"gate_bias_vector", "compound_physical"}:
        gate_bias = rng.normal(0.0, gate_scale)
    if family in {"cavity_dephasing", "compound_physical"}:
        cavity_tphi_us = float(np.exp(rng.uniform(np.log(80.0), np.log(1000.0))))
    if family in {"unseen_timing_dynamics", "compound_physical"}:
        timing = _duration_profile(rng)
        cavity_lifetime_us = float(rng.uniform(130.0, 380.0))
        ancilla_t1_us = float(rng.uniform(28.0, 90.0))
        ancilla_t2_us = float(rng.uniform(0.55, 0.98) * 2.0 * ancilla_t1_us)
    values = {
        "family": family,
        "cell_id": f"physical-{family}-{cell_index}",
        "seed": int(seed),
        "gate_bias": gate_bias.tolist(),
        "gate_bias_l2": float(np.linalg.norm(gate_bias)),
        "gate_bias_nonzero_dimensions": int(np.count_nonzero(gate_bias)),
        "cavity_tphi_us": cavity_tphi_us,
        "cavity_lifetime_us": cavity_lifetime_us,
        "ancilla_t1_us": ancilla_t1_us,
        "ancilla_t2_us": ancilla_t2_us,
        "timing_durations_ns": [phase.duration_ns for phase in timing.phases],
        "timing_total_ns": timing.half_cycle_duration_ns,
    }
    values["mismatch_vector_sha256"] = _canonical_sha256(values)
    return values


class RandomizedMismatchSimulator(DifferentiableSBSTrajectorySimulator):
    """Base finite-cutoff simulator plus explicit gate bias/phase diffusion."""

    def __init__(
        self,
        config: DifferentiableSBSConfig,
        *,
        gate_bias: Sequence[float],
        cavity_tphi_us: float | None,
    ) -> None:
        super().__init__(config)
        th = _require_torch()
        bias = np.asarray(gate_bias, dtype=np.float64)
        if bias.shape != (15,) or not np.all(np.isfinite(bias)):
            raise ValueError("gate bias must be a finite 15-vector")
        if cavity_tphi_us is not None and (
            not math.isfinite(cavity_tphi_us) or cavity_tphi_us <= 0.0
        ):
            raise ValueError("cavity tphi must be positive or null")
        self.gate_bias = th.tensor(
            bias, dtype=self.real_dtype, device=self.device
        )
        self.cavity_tphi_us = cavity_tphi_us
        self._phase_duration_us = {
            phase.phase_id: phase.duration_ns / 1000.0 for phase in config.timing.phases
        }
        levels = th.arange(config.cutoff, dtype=self.real_dtype, device=self.device)
        self._number_difference_sq = (levels[:, None] - levels[None, :]) ** 2

    def _biased(self, physical: Any) -> Any:
        return physical + self.gate_bias

    def bounded_physical_controls(self, raw_corrections: Any | None = None) -> Any:
        return self._biased(super().bounded_physical_controls(raw_corrections))

    def _policy_controls(self, policy: Any, history: Any, half_index: int) -> Any:
        return self._biased(super()._policy_controls(policy, history, half_index))

    def _apply_idle(self, state: Any, phase_id: str) -> Any:
        result = super()._apply_idle(state, phase_id)
        if self.cavity_tphi_us is None:
            return result
        th = _require_torch()
        duration = self._phase_duration_us[phase_id]
        kernel = th.exp(
            -0.5 * duration / self.cavity_tphi_us * self._number_difference_sq
        )
        blocks = result.reshape(
            self.config.batch_size,
            self.cutoff,
            2,
            self.cutoff,
            2,
        )
        return (blocks * kernel[None, :, None, :, None]).reshape_as(result)

    def phase_kernel_minimum_eigenvalue(self) -> float | None:
        if self.cavity_tphi_us is None:
            return None
        th = _require_torch()
        duration = max(self._phase_duration_us.values())
        kernel = th.exp(
            -0.5 * duration / self.cavity_tphi_us * self._number_difference_sq
        )
        return float(th.min(th.linalg.eigvalsh(kernel)).detach().cpu())


def _simulator_from_cell(
    cell: Mapping[str, Any], config: RandomizedMismatchConfig, *, matched: bool
) -> RandomizedMismatchSimulator:
    timing = (
        DifferentiableSBSTimingProfile()
        if matched
        else DifferentiableSBSTimingProfile(
            phases=tuple(
                TrajectoryTimingPhase(
                    reference.phase_id,
                    int(duration),
                    reference.gate_before_idle,
                )
                for reference, duration in zip(
                    DifferentiableSBSTimingProfile().phases,
                    cell["timing_durations_ns"],
                    strict=True,
                )
            )
        )
    )
    physics = DifferentiableSBSConfig(
        cutoff=config.physical_cutoff,
        full_cycles=config.physical_full_cycles,
        batch_size=config.physical_batch_size,
        cavity_lifetime_us=245.0 if matched else float(cell["cavity_lifetime_us"]),
        ancilla_t1_us=50.0 if matched else float(cell["ancilla_t1_us"]),
        ancilla_t2_us=60.0 if matched else float(cell["ancilla_t2_us"]),
        timing=timing,
        device=config.device,
        real_dtype=config.real_dtype,
    )
    return RandomizedMismatchSimulator(
        physics,
        gate_bias=(0.0,) * 15 if matched else cell["gate_bias"],
        cavity_tphi_us=None if matched else cell["cavity_tphi_us"],
    )


def _physical_metrics(result: Any, simulator: Any) -> dict[str, Any]:
    th = _require_torch()
    fidelity_curve = result.cycle_fidelities.mean(dim=0).detach().cpu().numpy()
    logical = result.cycle_logical_z_signal
    logical_curve = (
        logical / th.clamp(logical[:, :1], min=simulator.config.probability_floor)
    ).mean(dim=0).detach().cpu().numpy()
    fidelity = _effective_lifetime(fidelity_curve)
    logical_fit = _effective_lifetime(logical_curve)
    return {
        "selection_score": float(
            0.5 * (fidelity["normalized_auc"] + logical_fit["normalized_auc"])
        ),
        "fidelity_lifetime_cycles": float(fidelity["effective_lifetime_cycles"]),
        "logical_z_lifetime_cycles": float(logical_fit["effective_lifetime_cycles"]),
        "terminal_fidelity": float(fidelity_curve[-1]),
        "terminal_logical_z": float(logical_curve[-1]),
        "mean_ground_probability": float(
            th.mean((result.outcomes == 0).to(th.float64)).detach().cpu()
        ),
        "maximum_trace_error": float(result.maximum_trace_error),
        "maximum_hermiticity_error": float(result.maximum_hermiticity_error),
        "minimum_final_eigenvalue": float(result.minimum_final_eigenvalue),
    }


def _config_value(config: RandomizedMismatchConfig | Mapping[str, Any], name: str) -> float:
    return float(config[name] if isinstance(config, Mapping) else getattr(config, name))


def _retention_evidence(
    cells: Sequence[Mapping[str, Any]],
    strategy_rows: Sequence[Mapping[str, Any]],
    config: RandomizedMismatchConfig | Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild branch evidence from raw per-strategy mismatch scores."""

    minimum_gain = _config_value(config, "minimum_qualifying_teacher_gain")
    retention_rows: list[dict[str, Any]] = []
    for cell in cells:
        selected = {
            str(row["strategy"]): row
            for row in strategy_rows
            if row["cell_id"] == cell["cell_id"]
        }
        if set(selected) != set(PHYSICAL_STRATEGIES):
            raise ValueError(
                f"cell {cell['cell_id']} must contain each physical strategy once"
            )
        standard = float(selected["standard"]["mismatch"]["selection_score"])
        teacher_score = float(selected["teacher"]["mismatch"]["selection_score"])
        student_score = float(selected["student"]["mismatch"]["selection_score"])
        teacher_gain = teacher_score - standard
        student_gain = student_score - standard
        retention = student_gain / teacher_gain if teacher_gain > minimum_gain else None
        retention_rows.append(
            {
                "family": cell["family"],
                "cell_id": cell["cell_id"],
                "seed": cell["seed"],
                "standard_score": standard,
                "teacher_score": teacher_score,
                "student_score": student_score,
                "teacher_gain_over_standard": teacher_gain,
                "student_gain_over_standard": student_gain,
                "student_teacher_absolute_gap": abs(student_score - teacher_score),
                "gain_retention": retention,
                "qualifying_teacher_gain": retention is not None,
            }
        )
    qualifying = [row for row in retention_rows if row["gain_retention"] is not None]
    retention_values = np.asarray(
        [row["gain_retention"] for row in qualifying], dtype=np.float64
    )
    gap_values = np.asarray(
        [row["student_teacher_absolute_gap"] for row in retention_rows],
        dtype=np.float64,
    )
    compound = np.asarray(
        [
            row["gain_retention"]
            for row in qualifying
            if row["family"] == "compound_physical"
        ],
        dtype=np.float64,
    )
    positive_fraction = len(qualifying) / len(retention_rows)
    summary = {
        "cell_count": len(cells),
        "strategy_cell_rows": len(strategy_rows),
        "qualifying_teacher_gain_cells": len(qualifying),
        "positive_teacher_gain_fraction": positive_fraction,
        "retention_median": None
        if retention_values.size == 0
        else float(np.median(retention_values)),
        "retention_q1": None
        if retention_values.size == 0
        else float(np.quantile(retention_values, 0.25)),
        "retention_minimum": None
        if retention_values.size == 0
        else float(np.min(retention_values)),
        "compound_retention_median": None
        if compound.size == 0
        else float(np.median(compound)),
        "teacher_student_abs_gap_p95": float(np.quantile(gap_values, 0.95)),
    }
    summary["student_mismatch_retention_passed"] = bool(
        positive_fraction
        >= _config_value(config, "minimum_positive_teacher_gain_fraction")
        and summary["retention_median"] is not None
        and summary["retention_median"]
        >= _config_value(config, "minimum_retention_median")
        and summary["retention_q1"]
        >= _config_value(config, "minimum_retention_q1")
        and summary["compound_retention_median"] is not None
        and summary["compound_retention_median"]
        >= _config_value(config, "minimum_retention_q1")
        and summary["teacher_student_abs_gap_p95"]
        <= _config_value(config, "maximum_teacher_student_abs_gap_p95")
    )
    return retention_rows, summary


def _run_physical_lane(config: RandomizedMismatchConfig) -> dict[str, Any]:
    th = _require_torch()
    dtype = th.float64 if config.real_dtype == "float64" else th.float32
    teacher, teacher_report = load_and_verify_teacher_checkpoint(
        TEACHER_CHECKPOINT, TEACHER_ARTIFACT
    )
    teacher.to(device=config.device, dtype=dtype).eval()
    student_artifact = LowDimensionalRecurrenceArtifact.from_dict(
        _load_json(PRODUCTION_STUDENT)
    )
    student = DistilledStudentTorchPolicy(
        student_artifact, device=config.device, dtype=dtype
    ).eval()
    policies = {"standard": None, "teacher": teacher, "student": student}
    cells = [
        _physical_cell(family, index, config)
        for family in PHYSICAL_FAMILIES
        for index in range(config.physical_cells_per_family)
    ]
    rows: list[dict[str, Any]] = []
    cell_diagnostics = []
    with th.no_grad():
        for cell in cells:
            mismatch_simulator = _simulator_from_cell(cell, config, matched=False)
            cell_diagnostics.append(
                {
                    "cell_id": cell["cell_id"],
                    "phase_kernel_minimum_eigenvalue": (
                        mismatch_simulator.phase_kernel_minimum_eigenvalue()
                    ),
                    "idle_completeness_max": max(
                        mismatch_simulator.idle_completeness_residuals().values()
                    ),
                }
            )
            for strategy, policy in policies.items():
                paired = {}
                for condition in ("matched", "mismatch"):
                    simulator = (
                        _simulator_from_cell(cell, config, matched=True)
                        if condition == "matched"
                        else mismatch_simulator
                    )
                    result = simulator.run(
                        control_policy=policy,
                        seed=int(cell["seed"]),
                        record_cycle_metrics=True,
                    )
                    paired[condition] = _physical_metrics(result, simulator)
                rows.append(
                    {
                        "family": cell["family"],
                        "cell_id": cell["cell_id"],
                        "seed": cell["seed"],
                        "strategy": strategy,
                        "matched": paired["matched"],
                        "mismatch": paired["mismatch"],
                        "selection_score_degradation": float(
                            paired["matched"]["selection_score"]
                            - paired["mismatch"]["selection_score"]
                        ),
                    }
                )
    retention_rows, summary = _retention_evidence(cells, rows, config)
    aggregates = []
    for family in PHYSICAL_FAMILIES:
        for strategy in PHYSICAL_STRATEGIES:
            selected = [
                row
                for row in rows
                if row["family"] == family and row["strategy"] == strategy
            ]
            degradation = np.asarray(
                [row["selection_score_degradation"] for row in selected]
            )
            mismatch_scores = np.asarray(
                [row["mismatch"]["selection_score"] for row in selected]
            )
            aggregates.append(
                {
                    "family": family,
                    "strategy": strategy,
                    "cell_count": len(selected),
                    "mismatch_score_median": float(np.median(mismatch_scores)),
                    "mismatch_score_iqr": float(
                        np.quantile(mismatch_scores, 0.75)
                        - np.quantile(mismatch_scores, 0.25)
                    ),
                    "degradation_median": float(np.median(degradation)),
                    "degradation_worst": float(np.max(degradation)),
                    "worst_cell_id": selected[int(np.argmax(degradation))]["cell_id"],
                }
            )
    return {
        "lane_id": "finite_cutoff_random_physical_control",
        "scope": (
            f"cutoff-{config.physical_cutoff} two-level "
            f"{config.physical_full_cycles}-cycle randomized "
            "control/noise/timing mismatch; "
            "not multilevel leakage, device calibration, or long-memory channel"
        ),
        "teacher_selected_state_sha256": teacher_report["checkpoint"][
            "selected_state_sha256"
        ],
        "student_artifact_sha256": student_artifact.artifact_sha256,
        "cells": cells,
        "cell_diagnostics": cell_diagnostics,
        "strategy_rows": rows,
        "retention_rows": retention_rows,
        "aggregates": aggregates,
        "retention_summary": summary,
        "evaluation_used_for_selection": False,
    }


def _random_confusion_matrix(rng: np.random.Generator) -> list[list[float]]:
    g_to_e = float(rng.uniform(0.002, 0.24))
    e_to_g = float(rng.uniform(0.002, 0.24))
    healthy_to_leak = float(rng.uniform(0.0, 0.01))
    detection_f = float(rng.uniform(0.70, 0.995))
    detection_higher = float(rng.uniform(0.70, 0.995))
    f_to_g = float(rng.uniform(0.0, 1.0 - detection_f))
    higher_to_g = float(rng.uniform(0.0, 1.0 - detection_higher))
    return [
        [1.0 - g_to_e - healthy_to_leak, g_to_e, healthy_to_leak],
        [e_to_g, 1.0 - e_to_g - healthy_to_leak, healthy_to_leak],
        [f_to_g, 1.0 - detection_f - f_to_g, detection_f],
        [higher_to_g, 1.0 - detection_higher - higher_to_g, detection_higher],
    ]


def _binomial_tolerance(probability: float, trials: int) -> float:
    return 5.0 * math.sqrt(
        max(probability * (1.0 - probability), 1.0e-12) / trials
    ) + 1.0 / trials


def _audit_full_confusion_matrix(
    matrix: Sequence[Sequence[float]], seed: int, trials_per_hidden_state: int
) -> dict[str, Any]:
    """Exercise and calibrate every hidden-state row of a 4x3 readout matrix."""

    probabilities = np.asarray(matrix, dtype=np.float64)
    if probabilities.shape != (4, 3):
        raise ValueError("readout confusion matrix must have shape (4,3)")
    if np.any(probabilities < 0.0) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0
    ):
        raise ValueError("readout confusion rows must be nonnegative and stochastic")
    trials = _positive_int(trials_per_hidden_state, "trials_per_hidden_state")
    rng = np.random.default_rng(_seed_stream(seed, "full-4x3-row-calibration"))
    counts = np.zeros((4, 3), dtype=np.int64)
    digest = hashlib.sha256()
    for hidden_index, row in enumerate(probabilities):
        observations = rng.choice(3, size=trials, p=row)
        counts[hidden_index] = np.bincount(observations, minlength=3)
        digest.update(hidden_index.to_bytes(1, "little"))
        digest.update(observations.astype(np.uint8, copy=False).tobytes())
    empirical = counts / float(trials)
    tolerances = np.asarray(
        [
            [_binomial_tolerance(float(value), trials) for value in row]
            for row in probabilities
        ],
        dtype=np.float64,
    )
    return {
        "hidden_state_order": ["g", "e", "f", "higher"],
        "observed_class_order": ["g", "e", "leakage"],
        "trials_per_hidden_state": trials,
        "counts": counts.tolist(),
        "empirical_matrix": empirical.tolist(),
        "entrywise_tolerances": tolerances.tolist(),
        "maximum_absolute_entry_error": float(
            np.max(np.abs(empirical - probabilities))
        ),
        "all_four_hidden_rows_exercised": bool(np.all(counts.sum(axis=1) == trials)),
        "full_matrix_calibrated": bool(
            np.all(np.abs(empirical - probabilities) <= tolerances)
        ),
        "trace_sha256": digest.hexdigest(),
    }


def _run_readout_lane(config: RandomizedMismatchConfig) -> dict[str, Any]:
    rows = []
    for index in range(config.component_cells_per_family):
        seed = _seed_stream(config.master_seed, f"readout:{index}")
        matrix = _random_confusion_matrix(np.random.default_rng(seed))
        row = ood_parent._run_measurement_seed(
            f"random-readout-{index}",
            matrix,
            seed,
            config.readout_cycles_per_cell,
        )
        full_matrix_audit = _audit_full_confusion_matrix(
            matrix, seed, config.readout_cycles_per_cell
        )
        g_target = matrix[0][1]
        e_target = matrix[1][0]
        row.update(
            {
                "family": READOUT_FAMILY,
                "cell_id": f"readout-{index}",
                "target_g_to_e": g_target,
                "target_e_to_g": e_target,
                "g_tolerance": _binomial_tolerance(
                    g_target, int(row["g_observation_count"])
                ),
                "e_tolerance": _binomial_tolerance(
                    e_target, int(row["e_observation_count"])
                ),
                "mismatch_vector_sha256": _canonical_sha256(matrix),
                "full_matrix_audit": full_matrix_audit,
            }
        )
        row["confusion_calibrated"] = bool(
            abs(row["empirical_g_to_e"] - g_target) <= row["g_tolerance"]
            and abs(row["empirical_e_to_g"] - e_target) <= row["e_tolerance"]
            and full_matrix_audit["all_four_hidden_rows_exercised"]
            and full_matrix_audit["full_matrix_calibrated"]
        )
        rows.append(row)
    return {
        "lane_id": "random_protocol_readout_confusion",
        "scope": "protocol-native effective readout confusion; not device measurement calibration",
        "rows": rows,
        "cell_count": len(rows),
        "unique_matrix_hashes": len({row["mismatch_vector_sha256"] for row in rows}),
        "misclassification_rate_interval": _mean_interval(
            [row["misclassification_rate"] for row in rows], config.confidence_level
        ),
        "evaluation_used_for_selection": False,
    }


def _run_leakage_lane(config: RandomizedMismatchConfig) -> dict[str, Any]:
    parent_config = leakage_parent.CampaignConfig()
    rows = []
    for family in LEAKAGE_FAMILIES:
        for index in range(config.component_cells_per_family):
            seed = _seed_stream(config.master_seed, f"leakage:{family}:{index}")
            rng = np.random.default_rng(seed)
            if family == "random_leakage_injection":
                parent_family = "higher_leakage_injection"
                rate = float(np.exp(rng.uniform(np.log(1.0e-4), np.log(1.5e-2))))
                target_metric = "empirical_higher_injection_probability"
            else:
                parent_family = "higher_reset_failure"
                rate = float(rng.uniform(0.05, 0.98))
                target_metric = "empirical_reset_failure_probability"
            result = leakage_parent._run_seed_cell(
                parent_family, rate, seed, config=parent_config
            )
            result.update(
                {
                    "family": family,
                    "cell_id": f"{family}-{index}",
                    "target_rate": rate,
                    "target_metric": target_metric,
                    "mismatch_vector_sha256": _canonical_sha256(
                        {"family": family, "rate": rate}
                    ),
                }
            )
            empirical = result[target_metric]
            cycle_denominator = int(result["trajectories"]) * int(
                result["evaluation_cycles"]
            )
            if family == "random_leakage_injection":
                empirical_injection = float(
                    result["empirical_higher_injection_probability"]
                )
                trials = int(
                    round(result["injection_episode_count"] / empirical_injection)
                )
                trial_source = (
                    "injection_episode_count / "
                    "empirical_higher_injection_probability"
                )
            else:
                trials = int(
                    round(
                        result["reset_attempts_per_1000_cycles"]
                        * cycle_denominator
                        / 1000.0
                    )
                )
                trial_source = (
                    "reset_attempts_per_1000_cycles * "
                    "trajectories * evaluation_cycles / 1000"
                )
            trials = max(1, trials)
            result["calibration_trial_count"] = trials
            result["calibration_trial_count_source"] = trial_source
            result["calibration_tolerance"] = _binomial_tolerance(rate, trials)
            result["channel_calibrated"] = bool(
                empirical is not None
                and abs(float(empirical) - rate) <= result["calibration_tolerance"]
            )
            rows.append(result)
    aggregates = []
    for family in LEAKAGE_FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        aggregates.append(
            {
                "family": family,
                "cell_count": len(selected),
                "target_rate_min": min(row["target_rate"] for row in selected),
                "target_rate_max": max(row["target_rate"] for row in selected),
                "hidden_occupancy_interval": _mean_interval(
                    [row["hidden_leakage_occupancy"] for row in selected],
                    config.confidence_level,
                ),
                "safe_availability_interval": _mean_interval(
                    [row["safe_normal_action_availability"] for row in selected],
                    config.confidence_level,
                ),
            }
        )
    return {
        "lane_id": "random_persistent_leakage_reset",
        "scope": "effective persistent hidden leakage/reset kernel; not multilevel master equation or device rate",
        "rows": rows,
        "aggregates": aggregates,
        "evaluation_used_for_selection": False,
    }


@dataclass(frozen=True)
class _RandomDriftScenario:
    scenario_id: str
    state_values: tuple[DriftState, ...]

    def states(self, windows: int) -> tuple[DriftState, ...]:
        if windows != len(self.state_values):
            raise ValueError("random drift scenario window mismatch")
        return self.state_values


def _random_drift_states(
    seed: int, windows: int, *, dynamics: str | None = None
) -> tuple[tuple[DriftState, ...], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    lam = float(LATTICE_CONST)
    supported_dynamics = ("chirped_sinusoid", "random_telegraph", "ramp_burst")
    dynamics = str(rng.choice(supported_dynamics)) if dynamics is None else dynamics
    if dynamics not in supported_dynamics:
        raise ValueError(f"unsupported randomized dynamics: {dynamics}")
    mu_q_amp = float(rng.uniform(0.05, 0.34))
    mu_p_amp = float(rng.uniform(0.05, 0.34))
    sigma_center = float(rng.uniform(0.11, 0.27))
    sigma_amp = float(rng.uniform(0.01, min(0.11, sigma_center - 0.04)))
    rho_amp = float(rng.uniform(0.05, 0.90))
    outlier_probability = float(rng.uniform(0.0, 0.16))
    outlier_scale = float(rng.uniform(1.5, 6.5))
    phase0 = float(rng.uniform(0.0, 2.0 * math.pi))
    telegraph = 1.0
    states = []
    for step in range(windows):
        progress = step / max(1, windows - 1)
        if dynamics == "chirped_sinusoid":
            phase = phase0 + 2.0 * math.pi * (progress + 2.5 * progress**2)
            signal_q = math.sin(phase)
            signal_p = math.cos(0.8 * phase + 0.3)
        elif dynamics == "random_telegraph":
            if rng.random() < 0.18:
                telegraph *= -1.0
            signal_q = telegraph
            signal_p = -telegraph + 0.15 * math.sin(7.0 * progress)
            phase = phase0 + 4.0 * math.pi * progress
        else:
            burst = 1.0 if step % 11 in (0, 1, 2) else 0.0
            signal_q = 2.0 * progress - 1.0 + 0.25 * burst
            signal_p = 1.0 - 2.0 * progress - 0.20 * burst
            phase = phase0 + 6.0 * math.pi * progress
        sigma_q = (sigma_center + sigma_amp * math.sin(phase)) * lam
        sigma_p = (sigma_center - sigma_amp * math.sin(phase)) * lam
        p_outlier = outlier_probability if step % 13 in (0, 1, 2) else 0.25 * outlier_probability
        states.append(
            DriftState(
                step=step,
                time=float(step),
                mu_q=mu_q_amp * signal_q * lam,
                mu_p=mu_p_amp * signal_p * lam,
                sigma_q=sigma_q,
                sigma_p=sigma_p,
                rho=rho_amp * math.sin(phase),
                p_outlier=p_outlier,
                outlier_scale=outlier_scale,
                burst_active=p_outlier > 0.08,
                source="t5.4.6-randomized-mismatch",
                regime=dynamics,
                seed=seed,
                event_id=step // 11,
            )
        )
    parameters = {
        "dynamics": dynamics,
        "mu_q_amplitude_fraction": mu_q_amp,
        "mu_p_amplitude_fraction": mu_p_amp,
        "sigma_center_fraction": sigma_center,
        "sigma_amplitude_fraction": sigma_amp,
        "rho_amplitude": rho_amp,
        "outlier_probability": outlier_probability,
        "outlier_scale": outlier_scale,
        "phase0": phase0,
    }
    return tuple(states), parameters


def _run_drift_lane(config: RandomizedMismatchConfig) -> dict[str, Any]:
    parent = _load_json(PARENT_ARTIFACTS["T5.1.2"])
    held_config = ood_parent.HeldOutOODConfig()
    settings, frozen, static_map, frozen_binding = ood_parent._restore_decoder_parent(
        parent, held_config
    )
    moment = PeriodicMomentConfig(
        minimum_samples=min(64, settings.observation_samples_per_window)
    )
    rows = []
    registered_dynamics = ("chirped_sinusoid", "random_telegraph", "ramp_burst")
    for index in range(config.drift_cells):
        seed = _seed_stream(config.master_seed, f"drift:{index}")
        states, parameters = _random_drift_states(
            seed,
            settings.windows,
            dynamics=registered_dynamics[index % len(registered_dynamics)],
        )
        scenario = _RandomDriftScenario(f"random-drift-{index}", states)
        row = dict(
            _evaluate_seed(
                scenario,
                160 + index,
                seed,
                settings,
                frozen,
                static_map,
                moment,
            )
        )
        row.update(
            {
                "family": DRIFT_FAMILY,
                "cell_id": f"drift-{index}",
                "parameters": parameters,
                "dynamic_seed": seed,
                "mismatch_vector_sha256": _canonical_sha256(parameters),
                "shared_trace_for_all_methods": True,
            }
        )
        rows.append(row)
    methods = ("standard", "static", "window", "ewma", "kalman", "oracle")
    aggregates = {
        method: {
            "error_rate_interval": _mean_interval(
                [row[f"{method}_error_rate"] for row in rows],
                config.confidence_level,
            ),
            "worst_error_rate": max(row[f"{method}_error_rate"] for row in rows),
        }
        for method in methods
    }
    return {
        "lane_id": "frozen_decoder_random_drift_dynamics",
        "scope": "frozen syndrome decoders under random synthetic drift; not physical-memory/device robustness",
        "frozen_parent_binding": frozen_binding,
        "rows": rows,
        "aggregates": aggregates,
        "dynamics_families": sorted({row["parameters"]["dynamics"] for row in rows}),
        "dynamics_assignment": "index-stratified; within-family parameters randomized",
        "evaluation_used_for_selection": False,
    }


def _mismatch_registry(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell in report["physical_lane"]["cells"]:
        rows.append(
            {
                "lane": "physical",
                "family": cell["family"],
                "cell_id": cell["cell_id"],
                "seed": cell["seed"],
                "vector_sha256": cell["mismatch_vector_sha256"],
            }
        )
    for lane_name in ("readout_lane", "leakage_reset_lane", "drift_lane"):
        lane = report[lane_name]
        for row in lane["rows"]:
            rows.append(
                {
                    "lane": lane_name.removesuffix("_lane"),
                    "family": row["family"],
                    "cell_id": row["cell_id"],
                    "seed": int(row.get("seed", row.get("dynamic_seed"))),
                    "vector_sha256": row["mismatch_vector_sha256"],
                }
            )
    return rows


def _branch_decision(
    physical_lane: Mapping[str, Any],
    config: RandomizedMismatchConfig | Mapping[str, Any],
) -> dict[str, Any]:
    summary = physical_lane["retention_summary"]
    retained = bool(summary["student_mismatch_retention_passed"])
    return {
        "input_branch": "qualified_student_retention",
        "student_mismatch_retention_passed": retained,
        "output_branch": (
            "qualified_student_retention"
            if retained
            else "drift_regime_aware_map_lut"
        ),
        "fallback_activated": not retained,
        "thresholds": {
            "minimum_retention_median": _config_value(
                config, "minimum_retention_median"
            ),
            "minimum_retention_q1": _config_value(config, "minimum_retention_q1"),
            "maximum_teacher_student_abs_gap_p95": _config_value(
                config, "maximum_teacher_student_abs_gap_p95"
            ),
            "minimum_positive_teacher_gain_fraction": _config_value(
                config, "minimum_positive_teacher_gain_fraction"
            ),
            "minimum_qualifying_teacher_gain": _config_value(
                config, "minimum_qualifying_teacher_gain"
            ),
        },
        "observed": summary,
        "evaluation_used_to_change_thresholds": False,
    }


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
        "mismatch_registry",
        "physical_lane",
        "readout_lane",
        "leakage_reset_lane",
        "drift_lane",
        "branch_decision",
        "claim_boundary",
        "gates",
        "gate_summary",
        "source_data",
    )
    return {key: report[key] for key in keys}


def _compute_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    config = report["config"]
    physical = report["physical_lane"]
    readout = report["readout_lane"]
    leakage = report["leakage_reset_lane"]
    drift = report["drift_lane"]
    registry = report["mismatch_registry"]
    branch = report["branch_decision"]
    expected_registry = (
        len(PHYSICAL_FAMILIES) * int(config["physical_cells_per_family"])
        + int(config["component_cells_per_family"])
        + len(LEAKAGE_FAMILIES) * int(config["component_cells_per_family"])
        + int(config["drift_cells"])
    )
    expected_physical_rows = (
        len(PHYSICAL_FAMILIES)
        * int(config["physical_cells_per_family"])
        * len(PHYSICAL_STRATEGIES)
    )
    physical_hashes = [cell["mismatch_vector_sha256"] for cell in physical["cells"]]
    try:
        recomputed_retention_rows, recomputed_retention_summary = _retention_evidence(
            physical["cells"], physical["strategy_rows"], config
        )
        recomputed_branch = _branch_decision(
            {"retention_summary": recomputed_retention_summary}, config
        )
        retention_evidence_consistent = (
            physical["retention_rows"] == recomputed_retention_rows
            and physical["retention_summary"] == recomputed_retention_summary
            and branch == recomputed_branch
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        retention_evidence_consistent = False

    physical_vectors_bound = all(
        cell["mismatch_vector_sha256"]
        == _canonical_sha256(
            {key: value for key, value in cell.items() if key != "mismatch_vector_sha256"}
        )
        for cell in physical["cells"]
    )

    def readout_row_valid(row: Mapping[str, Any]) -> bool:
        matrix = np.asarray(row["confusion_matrix"], dtype=np.float64)
        audit = row["full_matrix_audit"]
        counts = np.asarray(audit["counts"], dtype=np.int64)
        trials = int(audit["trials_per_hidden_state"])
        empirical = counts / float(trials)
        tolerances = np.asarray(
            [
                [_binomial_tolerance(float(value), trials) for value in values]
                for values in matrix
            ]
        )
        g_tolerance = _binomial_tolerance(
            float(matrix[0, 1]), int(row["g_observation_count"])
        )
        e_tolerance = _binomial_tolerance(
            float(matrix[1, 0]), int(row["e_observation_count"])
        )
        return bool(
            matrix.shape == (4, 3)
            and np.all(matrix >= 0.0)
            and np.allclose(matrix.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0)
            and counts.shape == (4, 3)
            and np.all(counts.sum(axis=1) == trials)
            and np.allclose(audit["empirical_matrix"], empirical, atol=1.0e-15)
            and np.allclose(audit["entrywise_tolerances"], tolerances, atol=1.0e-15)
            and np.all(np.abs(empirical - matrix) <= tolerances)
            and abs(float(row["empirical_g_to_e"]) - matrix[0, 1]) <= g_tolerance
            and abs(float(row["empirical_e_to_g"]) - matrix[1, 0]) <= e_tolerance
            and row["target_g_to_e"] == matrix[0, 1]
            and row["target_e_to_g"] == matrix[1, 0]
            and row["g_tolerance"] == g_tolerance
            and row["e_tolerance"] == e_tolerance
            and audit["all_four_hidden_rows_exercised"] is True
            and audit["full_matrix_calibrated"] is True
            and row["confusion_calibrated"] is True
            and row["mismatch_vector_sha256"] == _canonical_sha256(matrix.tolist())
        )

    def leakage_row_valid(row: Mapping[str, Any]) -> bool:
        empirical = row[row["target_metric"]]
        target = float(row["target_rate"])
        trials = int(row["calibration_trial_count"])
        tolerance = _binomial_tolerance(target, trials)
        return bool(
            empirical is not None
            and trials > 0
            and row["calibration_tolerance"] == tolerance
            and abs(float(empirical) - target) <= tolerance
            and row["channel_calibrated"] is True
            and row["mismatch_vector_sha256"]
            == _canonical_sha256({"family": row["family"], "rate": target})
        )

    drift_dynamics = sorted(
        {str(row["parameters"]["dynamics"]) for row in drift["rows"]}
    )
    drift_vectors_bound = all(
        row["mismatch_vector_sha256"] == _canonical_sha256(row["parameters"])
        for row in drift["rows"]
    )
    return {
        "all_parent_artifacts_are_live_machine_passes": all(
            row["machine_pass"] for row in report["parent_bindings"]
        ),
        "all_random_evaluation_seeds_are_parent_disjoint": report[
            "seed_audit"
        ]["overlap_with_parent_seeds"]
        == [],
        "registry_covers_every_randomized_cell_once": registry
        == _mismatch_registry(report)
        and len(registry)
        == expected_registry
        and len({(row["lane"], row["cell_id"]) for row in registry})
        == expected_registry,
        "randomized_vectors_are_distinct_not_one_fixed_bias": len(
            {row["vector_sha256"] for row in registry}
        )
        == expected_registry
        and physical_vectors_bound
        and all(readout_row_valid(row) for row in readout["rows"])
        and all(leakage_row_valid(row) for row in leakage["rows"])
        and drift_vectors_bound,
        "all_four_physical_mismatch_families_have_eight_or_more_cells": set(
            cell["family"] for cell in physical["cells"]
        )
        == set(PHYSICAL_FAMILIES)
        and all(
            sum(cell["family"] == family for cell in physical["cells"]) >= 8
            for family in PHYSICAL_FAMILIES
        ),
        "physical_lane_has_matched_and_mismatch_all_three_strategies": len(
            physical["strategy_rows"]
        )
        == expected_physical_rows
        and all(
            set(row) >= {"matched", "mismatch", "selection_score_degradation"}
            for row in physical["strategy_rows"]
        )
        and all(
            {
                row["strategy"]
                for row in physical["strategy_rows"]
                if row["cell_id"] == cell["cell_id"]
            }
            == set(PHYSICAL_STRATEGIES)
            for cell in physical["cells"]
        ),
        "gate_bias_cells_use_random_full_15_vectors": all(
            cell["gate_bias_nonzero_dimensions"] == 15
            for cell in physical["cells"]
            if cell["family"] in {"gate_bias_vector", "compound_physical"}
        )
        and len(
            {
                tuple(cell["gate_bias"])
                for cell in physical["cells"]
                if cell["family"] in {"gate_bias_vector", "compound_physical"}
            }
        )
        >= 16,
        "cavity_dephasing_is_nonzero_and_phase_kernels_are_psd": all(
            diagnostic["phase_kernel_minimum_eigenvalue"] is None
            or diagnostic["phase_kernel_minimum_eigenvalue"] >= -1.0e-8
            for diagnostic in physical["cell_diagnostics"]
        )
        and all(
            cell["cavity_tphi_us"] is not None
            for cell in physical["cells"]
            if cell["family"] in {"cavity_dephasing", "compound_physical"}
        )
        and all(
            diagnostic["idle_completeness_max"] < 1.0e-10
            for diagnostic in physical["cell_diagnostics"]
        ),
        "timing_profiles_preserve_total_but_randomize_phase_allocation": all(
            cell["timing_total_ns"] == 5000 for cell in physical["cells"]
        )
        and len(
            {
                tuple(cell["timing_durations_ns"])
                for cell in physical["cells"]
                if cell["family"] in {"unseen_timing_dynamics", "compound_physical"}
            }
        )
        >= 16,
        "physical_outputs_are_finite_and_trace_stable": all(
            all(
                np.isfinite(float(metrics[key]))
                for metrics in (row["matched"], row["mismatch"])
                for key in (
                    "selection_score",
                    "fidelity_lifetime_cycles",
                    "logical_z_lifetime_cycles",
                    "terminal_fidelity",
                )
            )
            and all(
                metrics["maximum_trace_error"] < 1.0e-7
                and metrics["maximum_hermiticity_error"] < 1.0e-7
                and metrics["minimum_final_eigenvalue"] >= -1.0e-8
                for metrics in (row["matched"], row["mismatch"])
            )
            for row in physical["strategy_rows"]
        ),
        "branch_decision_is_threshold_bound_and_fail_closed": retention_evidence_consistent,
        "readout_lane_has_random_full_matrices_and_calibrated_ge_confusion": readout[
            "cell_count"
        ]
        >= 8
        and readout["unique_matrix_hashes"] == readout["cell_count"]
        and all(readout_row_valid(row) for row in readout["rows"]),
        "leakage_and_reset_rates_are_randomized_and_calibrated": set(
            row["family"] for row in leakage["rows"]
        )
        == set(LEAKAGE_FAMILIES)
        and all(leakage_row_valid(row) for row in leakage["rows"]),
        "drift_lane_uses_frozen_parent_decoders_and_shared_traces": drift[
            "frozen_parent_binding"
        ]["hyperparameters_reselected_on_ood"]
        is False
        and drift["frozen_parent_binding"]["static_parameters_refit_on_ood"] is False
        and len(drift["rows"]) >= 8
        and all(row["shared_trace_for_all_methods"] for row in drift["rows"]),
        "drift_vectors_span_multiple_unseen_dynamics": drift["dynamics_families"]
        == drift_dynamics
        and len(drift_dynamics) >= 2
        and drift_vectors_bound,
        "all_lanes_are_evaluation_only_without_reselection": all(
            report[name]["evaluation_used_for_selection"] is False
            for name in (
                "physical_lane",
                "readout_lane",
                "leakage_reset_lane",
                "drift_lane",
            )
        ),
        "no_cross_lane_global_score_or_universal_claim": report[
            "cross_lane_aggregate"
        ]
        is None
        and report["global_ranking"] is None,
        "physical_memory_device_and_hardware_claims_remain_closed": all(
            report["claim_boundary"][key] is False
            for key in (
                "physical_memory_ler_established",
                "device_calibrated",
                "hardware_measured",
                "experimental_claim",
            )
        ),
        "source_data_is_byte_bound": bool(report["source_data"].get("csv_sha256")),
    }


def validate_artifact(
    report: Mapping[str, Any], *, check_files: bool = True
) -> tuple[str, ...]:
    errors = []
    if report.get("schema_version") != SCHEMA_VERSION or report.get("task_id") != TASK_ID:
        errors.append("schema/task mismatch")
    try:
        gates = _compute_gates(report)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"gate recomputation failed: {exc}")
        gates = {}
    if gates != report.get("gates"):
        errors.append("stored gates differ from semantic recomputation")
    summary = {"passed": sum(gates.values()), "total": len(gates)}
    if report.get("gate_summary") != summary:
        errors.append("gate summary mismatch")
    status = "PASS" if gates and all(gates.values()) else "FAIL"
    if report.get("status") != status:
        errors.append("status mismatch")
    if report.get("contract_sha256") != _canonical_sha256(_contract_view(report)):
        errors.append("contract hash mismatch")
    if check_files:
        for row in report.get("parent_bindings", ()):
            if row["sha256"] != _sha256(row["path"]):
                errors.append(f"parent hash mismatch: {row['task_id']}")
        for row in report.get("implementation_bindings", ()):
            if row["sha256"] != _sha256(row["path"]):
                errors.append(f"implementation hash mismatch: {row['path']}")
        source = report.get("source_data", {})
        if not _repo_path(source.get("path", "")).is_file():
            errors.append("source file missing")
        elif source.get("csv_sha256") != _sha256(source["path"]):
            errors.append("source hash mismatch")
    return tuple(errors)


def source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def append(
        row_type: str,
        lane: str,
        family: str,
        cell_id: str,
        seed: Any,
        strategy: str,
        metric: str,
        value: Any,
        detail: Any,
    ) -> None:
        rows.append(
            {
                "row_type": row_type,
                "lane": lane,
                "family": family,
                "cell_id": cell_id,
                "seed": seed,
                "strategy": strategy,
                "metric": metric,
                "value": value,
                "detail_json": json.dumps(detail, sort_keys=True),
            }
        )

    for row in report["parent_bindings"]:
        append("parent_binding", "provenance", "", row["task_id"], "", "", "machine_pass", int(row["machine_pass"]), row)
    for row in report["implementation_bindings"]:
        append("implementation_binding", "provenance", "", row["path"], "", "", "sha256_bound", 1, row)
    for row in report["mismatch_registry"]:
        append("mismatch_registry", row["lane"], row["family"], row["cell_id"], row["seed"], "", "registered", 1, row)
    for row in report["physical_lane"]["strategy_rows"]:
        append("physical_strategy_cell", "physical", row["family"], row["cell_id"], row["seed"], row["strategy"], "mismatch_selection_score", row["mismatch"]["selection_score"], row)
    for row in report["physical_lane"]["retention_rows"]:
        append("physical_retention", "physical", row["family"], row["cell_id"], row["seed"], "student", "gain_retention", row["gain_retention"], row)
    for row in report["physical_lane"]["aggregates"]:
        append("physical_aggregate", "physical", row["family"], "all", "", row["strategy"], "degradation_median", row["degradation_median"], row)
    for row in report["readout_lane"]["rows"]:
        append("readout_cell", "readout", row["family"], row["cell_id"], row["seed"], "", "misclassification_rate", row["misclassification_rate"], row)
    for row in report["leakage_reset_lane"]["rows"]:
        append("leakage_reset_cell", "leakage_reset", row["family"], row["cell_id"], row["seed"], "", row["target_metric"], row[row["target_metric"]], row)
    for row in report["drift_lane"]["rows"]:
        append("drift_cell", "drift", row["family"], row["cell_id"], row["dynamic_seed"], "kalman", "error_rate", row["kalman_error_rate"], row)
    append("branch_decision", "physical", "", "qualified-student", "", "student", "retained", int(report["branch_decision"]["student_mismatch_retention_passed"]), report["branch_decision"])
    for name, value in report["gates"].items():
        append("gate", "evidence_gate", "", name, "", "", "passed", int(value), {"gate": name, "passed": value})
    return rows


def run_randomized_model_mismatch(
    config: RandomizedMismatchConfig | None = None,
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
    production: bool = True,
) -> dict[str, Any]:
    th = _require_torch()
    actual = config or RandomizedMismatchConfig()
    if production:
        validate_production_design(actual)
    if actual.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    started = time.perf_counter()
    parent_payloads = [_load_json(path) for path in PARENT_ARTIFACTS.values()]
    parent_seeds = set().union(*(_extract_seeds(value) for value in parent_payloads))
    physical = _run_physical_lane(actual)
    readout = _run_readout_lane(actual)
    leakage = _run_leakage_lane(actual)
    drift = _run_drift_lane(actual)
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
        "physical_lane": physical,
        "readout_lane": readout,
        "leakage_reset_lane": leakage,
        "drift_lane": drift,
        "cross_lane_aggregate": None,
        "global_ranking": None,
        "claim_boundary": {
            "allowed": (
                "lane-local randomized mismatch distributions, matched physical-control "
                "degradation, frozen-decoder errors, readout/leakage/reset burdens, and "
                "threshold-bound student branch decision"
            ),
            "forbidden": (
                "single-vector robustness, cross-lane global score, universal robustness, "
                "physical-memory LER, device calibration, RTL, FPGA, board, or experiment"
            ),
            "physical_memory_ler_established": False,
            "device_calibrated": False,
            "hardware_measured": False,
            "experimental_claim": False,
        },
        "source_data": {
            "path": Path(source_data_path).as_posix(),
            "row_count": 0,
            "rows_sha256": None,
            "csv_sha256": None,
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    report["mismatch_registry"] = _mismatch_registry(report)
    evaluation_seeds = {row["seed"] for row in report["mismatch_registry"]}
    report["seed_audit"] = {
        "evaluation_seed_count": len(evaluation_seeds),
        "parent_seed_count": len(parent_seeds),
        "overlap_with_parent_seeds": sorted(evaluation_seeds & parent_seeds),
    }
    report["branch_decision"] = _branch_decision(physical, actual)
    report["gates"] = _compute_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    rows = source_rows(report)
    _write_source(Path(source_data_path), rows)
    report["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "csv_sha256": _sha256(source_data_path),
    }
    report["gates"] = _compute_gates(report)
    report["gate_summary"] = {
        "passed": sum(report["gates"].values()),
        "total": len(report["gates"]),
    }
    report["status"] = "PASS" if all(report["gates"].values()) else "FAIL"
    retained = report["branch_decision"]["student_mismatch_retention_passed"]
    report["verdict"] = (
        (
            "EVIDENCE_COMPLETE_QUALIFIED_STUDENT_BRANCH_RETAINED"
            if retained
            else "EVIDENCE_COMPLETE_QUALIFIED_STUDENT_BRANCH_REVOKED_TO_MAP_LUT"
        )
        if report["status"] == "PASS"
        else "INCOMPLETE_EVIDENCE_GATE_FAILURE"
    )
    rows = source_rows(report)
    _write_source(Path(source_data_path), rows)
    report["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "row_count": len(rows),
        "rows_sha256": _canonical_sha256(rows),
        "csv_sha256": _sha256(source_data_path),
    }
    report["contract_sha256"] = _canonical_sha256(_contract_view(report))
    errors = validate_artifact(report)
    if errors:
        raise RuntimeError("invalid T5.4.6 artifact: " + "; ".join(errors))
    _atomic_json(report, Path(artifact_path))
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--pilot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = (
        RandomizedMismatchConfig(
            physical_cells_per_family=1,
            component_cells_per_family=2,
            drift_cells=2,
            physical_cutoff=6,
            physical_full_cycles=2,
            physical_batch_size=2,
            readout_cycles_per_cell=512,
            device=arguments.device,
        )
        if arguments.pilot
        else RandomizedMismatchConfig(device=arguments.device)
    )
    artifact = arguments.artifact
    source_data = arguments.source_data
    if arguments.pilot and artifact == DEFAULT_ARTIFACT:
        artifact = Path(".tmp_t546/pilot.json")
    if arguments.pilot and source_data == DEFAULT_SOURCE_DATA:
        source_data = Path(".tmp_t546/pilot.csv")
    report = run_randomized_model_mismatch(
        config,
        artifact_path=artifact,
        source_data_path=source_data,
        production=not arguments.pilot,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["verdict"],
                "gate_summary": report["gate_summary"],
                "branch_decision": report["branch_decision"]["output_branch"],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "PROTOCOL_ID",
    "RandomizedMismatchConfig",
    "RandomizedMismatchSimulator",
    "SCHEMA_VERSION",
    "TASK_ID",
    "run_randomized_model_mismatch",
    "validate_artifact",
    "validate_production_design",
]
