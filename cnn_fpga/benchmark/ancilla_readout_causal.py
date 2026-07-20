"""T5.2.2 independent ancilla-bit, ancilla-phase, and readout campaign.

The campaign changes exactly one protocol-native channel at a time.  It uses
the T2.2.2 sBs fault overlay and the T2.0.3 hidden/observed/reset kernel, but
keeps simulator truth outside the deployable record.  The reported quantities
are effective-model sensitivity diagnostics, not device-calibrated rates or a
physical-memory logical error rate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.algorithm_success_falsification import FALLBACK_BRANCH_ID
from physics.protocol_ancilla_errors import SBSAncillaFaultOverlay, SBSFaultOverlayConfig
from physics.sbs_observation_reset import make_persistent_leakage_model


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T5.2.2"
SCHEMA_VERSION = "t5.2.2-independent-ancilla-readout-causal-v1"
PROTOCOL_ID = "SBS-INDEPENDENT-ANCILLA-READOUT-INJECTION-V1"
DEFAULT_ARTIFACT = Path("docs/t5_2_2_ancilla_readout_causal.json")
DEFAULT_SOURCE_DATA = Path("docs/t5_2_2_ancilla_readout_source_data.csv")

PARENT_ARTIFACTS: dict[str, Path] = {
    "T2.2.2": Path("docs/t2_2_2_protocol_ancilla_validation.json"),
    "T5.1.2": Path("docs/t5_1_2_mixed_scenario_matrix.json"),
    "T5.1.6": Path("docs/t5_1_6_experimental_feasibility.json"),
    "T5.2.1": Path("docs/t5_2_1_displacement_large_error_causal.json"),
}

IMPLEMENTATION_PATHS = (
    Path("cnn_fpga/benchmark/ancilla_readout_causal.py"),
    Path("physics/protocol_ancilla_errors.py"),
    Path("physics/sbs_observation_reset.py"),
)

PRIMARY_SOURCE_PATH = (
    "relative_papers/Real-time_quantum_error_correction_beyond_break-even/"
    "Real-time_quantum_error_correction_beyond_break-even.md"
)
PRIMARY_SOURCE_ANCHORS = (
    {
        "line": 1121,
        "fragment": "QEC circuit is fault-tolerant with respect to ancilla phase flips",
        "role": "phase_flip_protocol_direction",
    },
    {
        "line": 1157,
        "fragment": "generate a logical error. This mechanism accounts",
        "role": "big_cd_bit_flip_logical_path",
    },
    {
        "line": 1157,
        "fragment": "misclassification of the ancilla state generates rotational errors",
        "role": "readout_virtual_rotation_path",
    },
    {
        "line": 1159,
        "fragment": "selectively increase the transmon phase-flip rate",
        "role": "independent_noise_injection_design",
    },
    {
        "line": 1165,
        "fragment": "phase flips is 65 times smaller than to bit flips",
        "role": "qualitative_sensitivity_direction_not_numeric_reproduction_target",
    },
)

FAULT_FAMILIES = ("ancilla_bit_flip", "ancilla_phase_flip", "readout_error")
INJECTION_RATES = (0.0, 0.005, 0.01, 0.02, 0.04, 0.08)
EVALUATION_SEEDS = tuple(202607162201 + index for index in range(8))
CALIBRATION_SEEDS = (
    2026071422,
    2026071423,
    20260716301,
    20260716302,
    20260716303,
    20260716304,
    20260716321,
)

BIT_LOGICAL_GIVEN_EVENT = 0.5
PHASE_BACKACTION_SCALE = 0.01
SMALL_CD_BIT_BACKACTION_SCALE = 0.02
VIRTUAL_ROTATION_MAX_RAD = 0.6

METRICS = (
    "bit_event_rate",
    "bit_outcome_toggle_rate",
    "logical_backaction_rate",
    "phase_event_rate",
    "phase_nonzero_backaction_rate",
    "mean_abs_continuous_backaction_x",
    "phase_z_basis_outcome_toggle_rate",
    "readout_misclassification_rate",
    "nonzero_virtual_rotation_rate",
    "mean_abs_virtual_rotation_rad",
    "faulted_label_change_rate",
)


@dataclass(frozen=True)
class CampaignConfig:
    fault_families: tuple[str, ...] = FAULT_FAMILIES
    injection_rates: tuple[float, ...] = INJECTION_RATES
    evaluation_seeds: tuple[int, ...] = EVALUATION_SEEDS
    cycles_per_seed_rate: int = 4096
    seed_cluster_bootstrap_replicates: int = 20000
    bootstrap_seed: int = 202607162299
    confidence_level: float = 0.95
    bit_logical_given_event: float = BIT_LOGICAL_GIVEN_EVENT
    phase_backaction_scale: float = PHASE_BACKACTION_SCALE
    small_cd_bit_backaction_scale: float = SMALL_CD_BIT_BACKACTION_SCALE
    virtual_rotation_max_rad: float = VIRTUAL_ROTATION_MAX_RAD

    def __post_init__(self) -> None:
        if tuple(self.fault_families) != FAULT_FAMILIES:
            raise ValueError("formal fault families changed")
        if tuple(self.injection_rates) != INJECTION_RATES:
            raise ValueError("formal injection-rate grid changed")
        if tuple(self.evaluation_seeds) != EVALUATION_SEEDS:
            raise ValueError("formal evaluation seed clusters changed")
        if set(self.evaluation_seeds) & set(CALIBRATION_SEEDS):
            raise ValueError("evaluation seeds overlap calibration/pilot seeds")
        if self.cycles_per_seed_rate != 4096:
            raise ValueError("formal campaign requires 4096 cycles per seed-rate cell")
        if self.seed_cluster_bootstrap_replicates != 20000:
            raise ValueError("formal campaign requires 20000 cluster bootstraps")
        if self.confidence_level != 0.95:
            raise ValueError("formal confidence level changed")
        if self.bit_logical_given_event != BIT_LOGICAL_GIVEN_EVENT:
            raise ValueError("bit logical-backaction assumption changed")
        if self.phase_backaction_scale != PHASE_BACKACTION_SCALE:
            raise ValueError("phase backaction assumption changed")
        if self.small_cd_bit_backaction_scale != SMALL_CD_BIT_BACKACTION_SCALE:
            raise ValueError("small-CD bit backaction assumption changed")
        if self.virtual_rotation_max_rad != VIRTUAL_ROTATION_MAX_RAD:
            raise ValueError("virtual-rotation range changed")


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _machine_pass(task_id: str, payload: Mapping[str, Any]) -> bool:
    if task_id == "T2.2.2":
        checks = payload.get("checks")
        return bool(
            isinstance(checks, Mapping)
            and checks
            and all(value is True for value in checks.values())
        )
    gates = payload.get("gates")
    return bool(
        payload.get("status") == "PASS"
        and isinstance(gates, Mapping)
        and gates
        and all(value is True for value in gates.values())
    )


def load_parent_artifacts(
    paths: Mapping[str, str | Path] = PARENT_ARTIFACTS,
) -> dict[str, dict[str, Any]]:
    return {
        task_id: json.loads(_repo_path(path).read_text(encoding="utf-8"))
        for task_id, path in paths.items()
    }


def inspect_parent_integrity(
    parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    return {
        task_id: {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "machine_pass": _machine_pass(task_id, parents[task_id]),
        }
        for task_id, path in PARENT_ARTIFACTS.items()
    }


def _source_anchors_current() -> bool:
    lines = _repo_path(PRIMARY_SOURCE_PATH).read_text(encoding="utf-8").splitlines()
    return all(
        0 < int(anchor["line"]) <= len(lines)
        and str(anchor["fragment"]) in lines[int(anchor["line"]) - 1]
        for anchor in PRIMARY_SOURCE_ANCHORS
    )


def _seed_stream(seed: int, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stream}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "little")


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _readout_base(readout_error: float):
    confusion = np.asarray(
        [
            [1.0 - readout_error, readout_error, 0.0],
            [readout_error, 1.0 - readout_error, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return make_persistent_leakage_model(
        readout_confusion=confusion,
        f_injection_given_g=0.0,
        f_injection_given_e=0.0,
        higher_injection_given_g=0.0,
        higher_injection_given_e=0.0,
        e_reset_success=1.0,
        f_reset_success=1.0,
        higher_reset_success=1.0,
        counter_max=2**31 - 1,
        readout_provenance="T5.2.2 isolated symmetric g/e confusion assumption",
        parameter_provenance="T5.2.2 no-leakage registered reset isolation",
    )


def _channel_spec(family: str, rate: float) -> dict[str, Any]:
    if family not in FAULT_FAMILIES:
        raise ValueError(f"unknown fault family: {family}")
    bit_rate = rate if family == "ancilla_bit_flip" else 0.0
    phase_rate = rate if family == "ancilla_phase_flip" else 0.0
    readout_error = rate if family == "readout_error" else 0.0
    return {
        "changed_channel": family,
        "bit_flip_probabilities": ((0.0, bit_rate, 0.0), (0.0, 0.0, 0.0)),
        "phase_flip_probabilities": ((0.0, phase_rate, 0.0), (0.0, 0.0, 0.0)),
        "readout_error_probability": readout_error,
        "logical_fault_given_big_cd_bit": (BIT_LOGICAL_GIVEN_EVENT, 0.0),
        "phase_backaction_scale": (PHASE_BACKACTION_SCALE, PHASE_BACKACTION_SCALE),
        "small_cd_bit_backaction_scale": (
            SMALL_CD_BIT_BACKACTION_SCALE,
            SMALL_CD_BIT_BACKACTION_SCALE,
        ),
        "misclassification_rotation_max_rad": VIRTUAL_ROTATION_MAX_RAD,
    }


def _overlay(family: str, rate: float) -> SBSAncillaFaultOverlay:
    spec = _channel_spec(family, rate)
    return SBSAncillaFaultOverlay(
        _readout_base(float(spec["readout_error_probability"])),
        SBSFaultOverlayConfig(
            bit_flip_probabilities=spec["bit_flip_probabilities"],
            phase_flip_probabilities=spec["phase_flip_probabilities"],
            logical_fault_given_big_cd_bit=spec[
                "logical_fault_given_big_cd_bit"
            ],
            phase_backaction_scale=spec["phase_backaction_scale"],
            small_cd_bit_backaction_scale=spec[
                "small_cd_bit_backaction_scale"
            ],
            misclassification_rotation_max_rad=spec[
                "misclassification_rotation_max_rad"
            ],
            parameter_provenance=(
                "T5.2.2 one-channel-at-a-time protocol-native causal campaign"
            ),
        ),
    )


def _ideal_labels(cycles: int) -> tuple[str, ...]:
    pattern = ("K_gg", "K_ge", "K_eg", "K_ee")
    return tuple(pattern[index % len(pattern)] for index in range(cycles))


def _run_seed_cell(
    family: str,
    rate: float,
    seed: int,
    *,
    cycles: int,
) -> dict[str, Any]:
    spec = _channel_spec(family, rate)
    trajectory = _overlay(family, rate).simulate(
        _ideal_labels(cycles),
        seed=_seed_stream(seed, f"{family}:paired-rate-stream"),
    )
    bit = np.zeros(cycles, dtype=np.bool_)
    bit_toggle = np.zeros(cycles, dtype=np.bool_)
    logical = np.zeros(cycles, dtype=np.bool_)
    phase = np.zeros(cycles, dtype=np.bool_)
    phase_backaction = np.zeros(cycles, dtype=np.bool_)
    continuous = np.zeros(cycles, dtype=np.float64)
    phase_z_toggle = np.zeros(cycles, dtype=np.bool_)
    readout_mismatch = np.zeros((cycles, 2), dtype=np.bool_)
    virtual_rotation = np.zeros((cycles, 2), dtype=np.float64)
    label_change = np.zeros(cycles, dtype=np.bool_)
    deployable_keys: set[str] | None = None

    for index, step in enumerate(trajectory.steps):
        truth = step.fault_truth
        x_bit_events = [
            event
            for event in truth.events
            if event.constituent == "X"
            and event.fault_type == "bit_flip"
            and event.stage == "big_cd"
        ]
        x_phase_events = [
            event
            for event in truth.events
            if event.constituent == "X"
            and event.fault_type == "phase_flip"
            and event.stage == "big_cd"
        ]
        bit[index] = bool(x_bit_events)
        bit_toggle[index] = any(event.toggles_z_basis_outcome for event in x_bit_events)
        logical[index] = truth.logical_backaction_by_constituent[0]
        phase[index] = bool(x_phase_events)
        phase_backaction[index] = any(
            event.continuous_backaction != 0.0 for event in x_phase_events
        )
        continuous[index] = truth.continuous_backaction_by_constituent[0]
        phase_z_toggle[index] = any(
            event.toggles_z_basis_outcome for event in x_phase_events
        )
        readout_mismatch[index] = truth.readout_misclassified
        virtual_rotation[index] = truth.virtual_rotation_error_rad
        label_change[index] = (
            truth.faulted_ideal_kraus_label != truth.original_ideal_kraus_label
        )
        keys = set(step.deployable_record())
        deployable_keys = keys if deployable_keys is None else deployable_keys | keys

    trace_sha256 = _array_hash(
        bit,
        bit_toggle,
        logical,
        phase,
        phase_backaction,
        continuous,
        phase_z_toggle,
        readout_mismatch,
        virtual_rotation,
        label_change,
    )
    expected_deployable = {
        "cycle_index",
        "syndrome_x",
        "syndrome_z",
        "reset_action_x",
        "reset_action_z",
        "x_e_run",
        "z_e_run",
        "leakage_constituent_run",
        "leakage_cycle_run",
        "observation_scope",
    }
    return {
        "family": family,
        "injected_rate": rate,
        "seed": seed,
        "cycles": cycles,
        "paired_stream_id": f"{family}-crn-{seed}",
        "channel_spec": spec,
        "bit_event_rate": float(np.mean(bit)),
        "bit_outcome_toggle_rate": float(np.mean(bit_toggle)),
        "logical_backaction_rate": float(np.mean(logical)),
        "phase_event_rate": float(np.mean(phase)),
        "phase_nonzero_backaction_rate": float(np.mean(phase_backaction)),
        "mean_abs_continuous_backaction_x": float(np.mean(np.abs(continuous))),
        "phase_z_basis_outcome_toggle_rate": float(np.mean(phase_z_toggle)),
        "readout_misclassification_rate": float(np.mean(readout_mismatch)),
        "nonzero_virtual_rotation_rate": float(
            np.mean(np.abs(virtual_rotation) > 0.0)
        ),
        "mean_abs_virtual_rotation_rad": float(
            np.mean(np.abs(virtual_rotation))
        ),
        "faulted_label_change_rate": float(np.mean(label_change)),
        "deployable_schema_exact": deployable_keys == expected_deployable,
        "trace_sha256": trace_sha256,
    }


def _run_seed_rows(config: CampaignConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in config.fault_families:
        for seed in config.evaluation_seeds:
            for rate in config.injection_rates:
                rows.append(
                    _run_seed_cell(
                        family,
                        rate,
                        seed,
                        cycles=config.cycles_per_seed_rate,
                    )
                )
    return rows


def _cluster_summary(
    values: Sequence[float],
    *,
    config: CampaignConfig,
    key: str,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (len(config.evaluation_seeds),) or not np.all(np.isfinite(data)):
        raise ValueError("cluster summary requires one finite value per seed cluster")
    rng = np.random.default_rng(_seed_stream(config.bootstrap_seed, key))
    indices = rng.integers(
        0,
        data.size,
        size=(config.seed_cluster_bootstrap_replicates, data.size),
    )
    bootstrap = np.mean(data[indices], axis=1)
    tail = 0.5 * (1.0 - config.confidence_level)
    low, high = np.quantile(bootstrap, [tail, 1.0 - tail])
    return {
        "mean": float(np.mean(data)),
        "ci_low": float(low),
        "ci_high": float(high),
        "paired_seed_cluster_count": int(data.size),
        "bootstrap_replicates": config.seed_cluster_bootstrap_replicates,
        "confidence_level": config.confidence_level,
        "resampling_unit": "whole_seed_cluster",
    }


def _summarize(
    seed_rows: Sequence[Mapping[str, Any]], config: CampaignConfig
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for family in config.fault_families:
        for rate in config.injection_rates:
            selected = [
                row
                for row in seed_rows
                if row["family"] == family and row["injected_rate"] == rate
            ]
            record: dict[str, Any] = {"family": family, "injected_rate": rate}
            for metric in METRICS:
                record[metric] = _cluster_summary(
                    [float(row[metric]) for row in selected],
                    config=config,
                    key=f"{family}:{rate}:{metric}",
                )
            summaries.append(record)
    return summaries


def _family_summary(
    summaries: Sequence[Mapping[str, Any]], family: str
) -> list[Mapping[str, Any]]:
    return [row for row in summaries if row["family"] == family]


def _means(records: Sequence[Mapping[str, Any]], metric: str) -> np.ndarray:
    return np.asarray([row[metric]["mean"] for row in records], dtype=np.float64)


def _strictly_increasing(records: Sequence[Mapping[str, Any]], metric: str) -> bool:
    return bool(np.all(np.diff(_means(records, metric)) > 0.0))


def _analytic_expectation(family: str, rate: float, metric: str) -> float:
    if family == "ancilla_bit_flip":
        if metric in ("bit_event_rate", "bit_outcome_toggle_rate", "faulted_label_change_rate"):
            return rate
        if metric == "logical_backaction_rate":
            return BIT_LOGICAL_GIVEN_EVENT * rate
    elif family == "ancilla_phase_flip":
        if metric in ("phase_event_rate", "phase_nonzero_backaction_rate"):
            return rate
        if metric == "mean_abs_continuous_backaction_x":
            return PHASE_BACKACTION_SCALE * rate
    elif family == "readout_error":
        if metric in ("readout_misclassification_rate", "nonzero_virtual_rotation_rate"):
            return rate
        if metric == "mean_abs_virtual_rotation_rad":
            return 0.5 * VIRTUAL_ROTATION_MAX_RAD * rate
    return 0.0


def _rate_tolerance(
    probability: float, *, trials: int, multiplier: float = 6.0
) -> float:
    return multiplier * sqrt(max(probability * (1.0 - probability), 0.0) / trials) + 1.0 / trials


def _analytic_gaps_within_tolerance(
    summaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> bool:
    cycles = int(config["cycles_per_seed_rate"])
    clusters = len(config["evaluation_seeds"])
    for row in summaries:
        family = str(row["family"])
        rate = float(row["injected_rate"])
        metrics = {
            "ancilla_bit_flip": (
                "bit_event_rate",
                "bit_outcome_toggle_rate",
                "logical_backaction_rate",
                "faulted_label_change_rate",
            ),
            "ancilla_phase_flip": (
                "phase_event_rate",
                "phase_nonzero_backaction_rate",
                "mean_abs_continuous_backaction_x",
            ),
            "readout_error": (
                "readout_misclassification_rate",
                "nonzero_virtual_rotation_rate",
                "mean_abs_virtual_rotation_rad",
            ),
        }[family]
        for metric in metrics:
            expected = _analytic_expectation(family, rate, metric)
            observed = float(row[metric]["mean"])
            if metric == "logical_backaction_rate":
                tolerance = _rate_tolerance(
                    expected, trials=cycles * clusters
                )
            elif metric in ("readout_misclassification_rate", "nonzero_virtual_rotation_rate"):
                tolerance = _rate_tolerance(
                    expected, trials=2 * cycles * clusters
                )
            elif metric == "mean_abs_virtual_rotation_rad":
                tolerance = 0.003
            elif metric == "mean_abs_continuous_backaction_x":
                tolerance = PHASE_BACKACTION_SCALE * _rate_tolerance(
                    rate, trials=cycles * clusters
                )
            else:
                tolerance = _rate_tolerance(rate, trials=cycles * clusters)
            if abs(observed - expected) > tolerance:
                return False
    return True


def _zero_control_clean(summaries: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        all(float(row[metric]["mean"]) == 0.0 for metric in METRICS)
        for row in summaries
        if row["injected_rate"] == 0.0
    )


def _cross_channels_clean(summaries: Sequence[Mapping[str, Any]]) -> bool:
    allowed_nonzero = {
        "ancilla_bit_flip": {
            "bit_event_rate",
            "bit_outcome_toggle_rate",
            "logical_backaction_rate",
            "faulted_label_change_rate",
        },
        "ancilla_phase_flip": {
            "phase_event_rate",
            "phase_nonzero_backaction_rate",
            "mean_abs_continuous_backaction_x",
        },
        "readout_error": {
            "readout_misclassification_rate",
            "nonzero_virtual_rotation_rate",
            "mean_abs_virtual_rotation_rad",
        },
    }
    return all(
        float(row[metric]["mean"]) == 0.0
        for row in summaries
        for metric in METRICS
        if metric not in allowed_nonzero[str(row["family"])]
    )


def _channel_specs_exact(seed_rows: Sequence[Mapping[str, Any]]) -> bool:
    return all(
        _canonical_sha256(row.get("channel_spec"))
        == _canonical_sha256(
            _channel_spec(
                str(row.get("family")), float(row.get("injected_rate", -1.0))
            )
        )
        for row in seed_rows
    )


def _expected_config_dict() -> dict[str, Any]:
    return asdict(CampaignConfig())


def _evaluate_gates(result: Mapping[str, Any]) -> dict[str, bool]:
    summaries = result["summary_rows"]
    bit = _family_summary(summaries, "ancilla_bit_flip")
    phase = _family_summary(summaries, "ancilla_phase_flip")
    readout = _family_summary(summaries, "readout_error")
    rows = result["seed_rows"]
    return {
        "all_parent_artifacts_are_current_and_machine_pass": all(
            record["machine_pass"] for record in result["parent_integrity"].values()
        ),
        "implementation_bindings_are_current": all(
            row["sha256"] == _sha256(row["path"])
            for row in result["implementation_bindings"]
        ),
        "primary_source_anchors_are_line_and_hash_bound": (
            result["source_binding"]["sha256"]
            == _sha256(result["source_binding"]["path"])
            and _source_anchors_current()
        ),
        "three_independent_families_and_six_rates_are_frozen": (
            tuple(result["config"]["fault_families"]) == FAULT_FAMILIES
            and tuple(result["config"]["injection_rates"]) == INJECTION_RATES
        ),
        "eight_evaluation_seed_clusters_are_disjoint_from_calibration": (
            tuple(result["config"]["evaluation_seeds"]) == EVALUATION_SEEDS
            and not (set(EVALUATION_SEEDS) & set(CALIBRATION_SEEDS))
        ),
        "every_family_seed_rate_cell_executes": len(rows)
        == len(FAULT_FAMILIES) * len(EVALUATION_SEEDS) * len(INJECTION_RATES),
        "only_one_registered_channel_changes_per_family": _channel_specs_exact(rows),
        "common_random_numbers_pair_rates_within_seed_family": all(
            len(
                {
                    row["paired_stream_id"]
                    for row in rows
                    if row["family"] == family and row["seed"] == seed
                }
            )
            == 1
            for family in FAULT_FAMILIES
            for seed in EVALUATION_SEEDS
        ),
        "zero_rate_controls_are_exactly_clean": _zero_control_clean(summaries),
        "bit_event_toggle_and_logical_paths_increase": (
            _strictly_increasing(bit, "bit_event_rate")
            and _strictly_increasing(bit, "bit_outcome_toggle_rate")
            and _strictly_increasing(bit, "logical_backaction_rate")
        ),
        "phase_event_and_small_backaction_paths_increase": (
            _strictly_increasing(phase, "phase_event_rate")
            and _strictly_increasing(phase, "phase_nonzero_backaction_rate")
            and _strictly_increasing(phase, "mean_abs_continuous_backaction_x")
        ),
        "phase_flip_never_toggles_z_basis_or_creates_logical_truth": all(
            row["phase_z_basis_outcome_toggle_rate"]["mean"] == 0.0
            and row["logical_backaction_rate"]["mean"] == 0.0
            and row["faulted_label_change_rate"]["mean"] == 0.0
            for row in phase
        ),
        "readout_misclassification_and_virtual_rotation_increase": (
            _strictly_increasing(readout, "readout_misclassification_rate")
            and _strictly_increasing(readout, "nonzero_virtual_rotation_rate")
            and _strictly_increasing(readout, "mean_abs_virtual_rotation_rad")
        ),
        "readout_only_never_injects_ancilla_or_logical_faults": all(
            row["bit_event_rate"]["mean"] == 0.0
            and row["phase_event_rate"]["mean"] == 0.0
            and row["logical_backaction_rate"]["mean"] == 0.0
            and row["faulted_label_change_rate"]["mean"] == 0.0
            for row in readout
        ),
        "all_primary_rates_match_independent_analytic_expectations": (
            _analytic_gaps_within_tolerance(summaries, result["config"])
        ),
        "all_cross_channel_estimands_remain_exactly_zero": _cross_channels_clean(
            summaries
        ),
        "deployable_schema_never_exposes_fault_truth": all(
            row["deployable_schema_exact"] is True for row in rows
        )
        and result["causal_contract"]["truth_visibility"]
        == "simulator_evaluator_only_not_deployable_input",
        "uncertainty_resamples_whole_seed_clusters": all(
            row[metric]["resampling_unit"] == "whole_seed_cluster"
            and row[metric]["paired_seed_cluster_count"] == len(EVALUATION_SEEDS)
            for row in summaries
            for metric in METRICS
        ),
        "separate_estimands_are_not_collapsed_to_global_score": (
            result["estimand_contract"]["global_sensitivity_score"]
            == "FORBIDDEN"
            and result["estimand_contract"]["physical_memory_ler"]
            == "NOT_ESTABLISHED"
        ),
        "active_fallback_branch_is_preserved": result["active_algorithm_branch"]
        == FALLBACK_BRANCH_ID,
        "device_and_experimental_claims_remain_false": (
            result["device_calibrated"] is False
            and result["experimental_hardware_used"] is False
            and result["physical_memory_ler_established"] is False
        ),
        "semantic_validator_accepts_only_complete_nonmixing_campaign": validate_payload(
            result
        )
        == (),
    }


def _summary_values(
    rows: Sequence[Mapping[str, Any]], family: str, rate: float, metric: str
) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if row["family"] == family and row["injected_rate"] == rate
    ]


def _has_forbidden_global_score(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key) in {"combined_fault_rate", "global_score", "global_sensitivity"}
            or _has_forbidden_global_score(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_has_forbidden_global_score(child) for child in value)
    return False


def validate_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("task_id") != TASK_ID or payload.get("protocol_id") != PROTOCOL_ID:
        errors.append("task or protocol identity changed")
    config = payload.get("config", {})
    expected_config = _expected_config_dict()
    normalized_config = dict(config)
    for key in ("fault_families", "injection_rates", "evaluation_seeds"):
        if key in normalized_config:
            normalized_config[key] = tuple(normalized_config[key])
    if normalized_config != expected_config:
        errors.append("formal campaign configuration changed")
    if set(config.get("evaluation_seeds", ())) & set(CALIBRATION_SEEDS):
        errors.append("evaluation/calibration seed overlap")

    rows = payload.get("seed_rows", ())
    expected_keys = {
        (family, seed, rate)
        for family in FAULT_FAMILIES
        for seed in EVALUATION_SEEDS
        for rate in INJECTION_RATES
    }
    actual_keys = {
        (row.get("family"), row.get("seed"), row.get("injected_rate"))
        for row in rows
    }
    if len(rows) != len(expected_keys) or actual_keys != expected_keys:
        errors.append("family-seed-rate matrix is incomplete or duplicated")
    if rows and not _channel_specs_exact(rows):
        errors.append("one-channel-at-a-time configuration changed")
    for row in rows:
        if row.get("cycles") != 4096:
            errors.append("seed-rate cell cycle count changed")
            break
        if row.get("paired_stream_id") != f"{row.get('family')}-crn-{row.get('seed')}":
            errors.append("paired stream identity changed")
            break
        if row.get("deployable_schema_exact") is not True:
            errors.append("fault truth leaked into deployable schema")
            break
        trace = row.get("trace_sha256", "")
        if not isinstance(trace, str) or len(trace) != 64:
            errors.append("trace hash is missing or malformed")
            break
        if any(not 0.0 <= float(row.get(metric, -1.0)) <= 1.0 for metric in METRICS):
            errors.append("seed metric is outside [0,1]")
            break

    summaries = payload.get("summary_rows", ())
    expected_summary_keys = {
        (family, rate) for family in FAULT_FAMILIES for rate in INJECTION_RATES
    }
    if len(summaries) != len(expected_summary_keys) or {
        (row.get("family"), row.get("injected_rate")) for row in summaries
    } != expected_summary_keys:
        errors.append("family-rate summary matrix is incomplete or duplicated")
    for summary in summaries:
        family = summary.get("family")
        rate = summary.get("injected_rate")
        for metric in METRICS:
            values = _summary_values(rows, family, rate, metric)
            record = summary.get(metric, {})
            if not values or abs(float(record.get("mean", -1.0)) - float(np.mean(values))) > 1e-15:
                errors.append("summary mean no longer matches seed clusters")
                break
            if not (
                record.get("paired_seed_cluster_count") == len(EVALUATION_SEEDS)
                and record.get("bootstrap_replicates") == 20000
                and record.get("confidence_level") == 0.95
                and record.get("resampling_unit") == "whole_seed_cluster"
                and float(record.get("ci_low", 2.0))
                <= float(record.get("mean", -1.0))
                <= float(record.get("ci_high", -2.0))
            ):
                errors.append("cluster uncertainty contract changed")
                break

    if len(summaries) == len(expected_summary_keys):
        bit = _family_summary(summaries, "ancilla_bit_flip")
        phase = _family_summary(summaries, "ancilla_phase_flip")
        readout = _family_summary(summaries, "readout_error")
        if not _zero_control_clean(summaries):
            errors.append("zero-rate negative controls are contaminated")
        if not _cross_channels_clean(summaries):
            errors.append("cross-channel estimands are contaminated")
        if not (
            _strictly_increasing(bit, "bit_event_rate")
            and _strictly_increasing(bit, "logical_backaction_rate")
        ):
            errors.append("bit-flip sensitivity direction changed")
        if not (
            _strictly_increasing(phase, "phase_event_rate")
            and _strictly_increasing(phase, "mean_abs_continuous_backaction_x")
        ):
            errors.append("phase-flip sensitivity direction changed")
        if not (
            _strictly_increasing(readout, "readout_misclassification_rate")
            and _strictly_increasing(readout, "mean_abs_virtual_rotation_rad")
        ):
            errors.append("readout sensitivity direction changed")
        if not _analytic_gaps_within_tolerance(summaries, config):
            errors.append("primary rates no longer match analytic expectations")

    causal = payload.get("causal_contract", {})
    if not (
        causal.get("intervention_rule") == "exactly_one_registered_channel_changes"
        and causal.get("truth_visibility")
        == "simulator_evaluator_only_not_deployable_input"
        and tuple(causal.get("fixed_channels", ()))
        == (
            "ideal balanced K_gg/K_ge/K_eg/K_ee label schedule",
            "no leakage injection",
            "registered conditional reset kernel",
            "bit-to-logical conditional probability",
            "phase and small-CD backaction scales",
            "virtual-rotation range",
        )
    ):
        errors.append("causal isolation or truth-visibility contract changed")
    estimand = payload.get("estimand_contract", {})
    if not (
        estimand.get("bit_path")
        == "big-CD bit event -> Z-basis toggle and evaluator-only logical backaction"
        and estimand.get("phase_path")
        == "big-CD phase event -> small continuous backaction without Z-basis toggle"
        and estimand.get("readout_path")
        == "g/e classifier error -> wrong observed feedback and virtual rotation"
        and estimand.get("global_sensitivity_score") == "FORBIDDEN"
        and estimand.get("physical_memory_ler") == "NOT_ESTABLISHED"
    ):
        errors.append("separate estimand contract changed")
    if _has_forbidden_global_score(
        {key: value for key, value in payload.items() if key != "estimand_contract"}
    ):
        errors.append("forbidden combined/global sensitivity score was introduced")

    if payload.get("active_algorithm_branch") != FALLBACK_BRANCH_ID:
        errors.append("active fallback branch changed")
    if (
        payload.get("device_calibrated") is not False
        or payload.get("experimental_hardware_used") is not False
        or payload.get("physical_memory_ler_established") is not False
    ):
        errors.append("effective simulation was promoted to device or physical LER evidence")

    integrity = payload.get("parent_integrity", {})
    if set(integrity) != set(PARENT_ARTIFACTS):
        errors.append("parent artifact membership changed")
    else:
        for task_id, path in PARENT_ARTIFACTS.items():
            record = integrity[task_id]
            if not (
                record.get("path") == path.as_posix()
                and record.get("sha256") == _sha256(path)
                and record.get("machine_pass") is True
            ):
                errors.append("parent artifact binding is stale or failed")
                break
    bindings = payload.get("implementation_bindings", ())
    if len(bindings) != len(IMPLEMENTATION_PATHS) or any(
        row.get("path") != path.as_posix() or row.get("sha256") != _sha256(path)
        for row, path in zip(bindings, IMPLEMENTATION_PATHS)
    ):
        errors.append("implementation binding is stale or incomplete")
    source = payload.get("source_binding", {})
    if not (
        source.get("path") == PRIMARY_SOURCE_PATH
        and source.get("sha256") == _sha256(PRIMARY_SOURCE_PATH)
        and tuple(source.get("anchors", ())) == PRIMARY_SOURCE_ANCHORS
        and _source_anchors_current()
    ):
        errors.append("primary source binding or exact line anchor changed")
    if "gates" in payload and (
        payload.get("status") != "PASS"
        or not payload.get("gates")
        or not all(value is True for value in payload["gates"].values())
    ):
        errors.append("committed machine gate status is not PASS")
    if "implementation_sha256" in payload and payload.get(
        "implementation_sha256"
    ) != implementation_sha256():
        errors.append("campaign implementation hash is stale")
    if "source_data" in payload:
        source_data = payload["source_data"]
        source_path = _repo_path(source_data.get("path", ""))
        if not source_path.is_file() or source_data.get("sha256") != hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest():
            errors.append("source-data binding is stale")
    return tuple(errors)


def build_report(
    parents: Mapping[str, Mapping[str, Any]],
    integrity: Mapping[str, Mapping[str, Any]],
    config: CampaignConfig | None = None,
) -> dict[str, Any]:
    actual = CampaignConfig() if config is None else config
    if not isinstance(actual, CampaignConfig):
        raise TypeError("config must be CampaignConfig or None")
    missing = set(PARENT_ARTIFACTS) - set(parents)
    if missing:
        raise ValueError(f"missing parent artifacts: {sorted(missing)}")
    seed_rows = _run_seed_rows(actual)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "pass_semantics": (
            "three independent protocol-native effective-fault sensitivities pass; "
            "not a device calibration, numeric reproduction of the experimental "
            "65x ratio, or physical-memory LER"
        ),
        "active_algorithm_branch": FALLBACK_BRANCH_ID,
        "config": asdict(actual),
        "parent_integrity": dict(integrity),
        "implementation_bindings": [
            {"path": path.as_posix(), "sha256": _sha256(path)}
            for path in IMPLEMENTATION_PATHS
        ],
        "source_binding": {
            "path": PRIMARY_SOURCE_PATH,
            "sha256": _sha256(PRIMARY_SOURCE_PATH),
            "anchors": list(PRIMARY_SOURCE_ANCHORS),
        },
        "causal_contract": {
            "intervention_rule": "exactly_one_registered_channel_changes",
            "paired_randomness": (
                "common random numbers across rates within each family/seed; "
                "eight seeds are independent resampling clusters"
            ),
            "fixed_channels": [
                "ideal balanced K_gg/K_ge/K_eg/K_ee label schedule",
                "no leakage injection",
                "registered conditional reset kernel",
                "bit-to-logical conditional probability",
                "phase and small-CD backaction scales",
                "virtual-rotation range",
            ],
            "truth_visibility": "simulator_evaluator_only_not_deployable_input",
            "readout_consequence_note": (
                "a classifier error may select the wrong registered reset action; "
                "this is a downstream consequence, not a second reset-channel injection"
            ),
        },
        "estimand_contract": {
            "bit_path": (
                "big-CD bit event -> Z-basis toggle and evaluator-only logical backaction"
            ),
            "phase_path": (
                "big-CD phase event -> small continuous backaction without Z-basis toggle"
            ),
            "readout_path": (
                "g/e classifier error -> wrong observed feedback and virtual rotation"
            ),
            "global_sensitivity_score": "FORBIDDEN",
            "numeric_65x_reproduction": "NOT_ATTEMPTED",
            "physical_memory_ler": "NOT_ESTABLISHED",
        },
        "seed_rows": seed_rows,
        "summary_rows": _summarize(seed_rows, actual),
        "device_calibrated": False,
        "experimental_hardware_used": False,
        "physical_memory_ler_established": False,
        "limitations": [
            "effective stochastic overlay rather than cavity-transmon master equation",
            "fault rates, backaction scales and virtual-rotation range are project assumptions",
            "logical backaction is evaluator truth and is not a repeated-memory LER",
            "no target-board, ADC/AWG, waveform, transmon or quantum-device calibration",
            "the experimental 65x bit/phase sensitivity ratio is a source anchor, not a numeric target",
        ],
    }
    errors = validate_payload(result)
    result["validation_errors"] = list(errors)
    gates = _evaluate_gates(result)
    result["gates"] = gates
    result["gate_summary"] = {
        "passed": sum(gates.values()),
        "total": len(gates),
        "failed": [name for name, value in gates.items() if not value],
    }
    result["status"] = "PASS" if all(gates.values()) else "FAIL"
    result["contract_sha256"] = _canonical_sha256(
        {
            "protocol_id": PROTOCOL_ID,
            "config": result["config"],
            "causal_contract": result["causal_contract"],
            "estimand_contract": result["estimand_contract"],
            "summary_rows": result["summary_rows"],
            "limitations": result["limitations"],
        }
    )
    return result


CSV_FIELDS = (
    "row_type",
    "family",
    "seed",
    "injected_rate",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "status_or_scope",
    "trace_sha256",
    "source_task",
)


def _source_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(row_type: str, **values: Any) -> None:
        row = {field: "" for field in CSV_FIELDS}
        row.update({"row_type": row_type, **values})
        rows.append(row)

    for task_id, parent in result["parent_integrity"].items():
        add(
            "parent_artifact",
            metric=parent["path"],
            value=parent["sha256"],
            status_or_scope=parent["machine_pass"],
            source_task=task_id,
        )
    for binding in result["implementation_bindings"]:
        add(
            "implementation_binding",
            metric=binding["path"],
            value=binding["sha256"],
            status_or_scope="current",
            source_task=TASK_ID,
        )
    for anchor in result["source_binding"]["anchors"]:
        add(
            "primary_source_anchor",
            metric=f"line:{anchor['line']}:{anchor['role']}",
            value=anchor["fragment"],
            status_or_scope="exact_line_fragment",
            source_task="Sivak2023",
        )
    for row in result["seed_rows"]:
        add(
            "channel_intervention",
            family=row["family"],
            seed=row["seed"],
            injected_rate=row["injected_rate"],
            metric="channel_spec_sha256",
            value=_canonical_sha256(row["channel_spec"]),
            status_or_scope="exactly_one_registered_channel_changes",
            trace_sha256=row["trace_sha256"],
            source_task=TASK_ID,
        )
        for metric in METRICS:
            add(
                "seed_metric",
                family=row["family"],
                seed=row["seed"],
                injected_rate=row["injected_rate"],
                metric=metric,
                value=row[metric],
                status_or_scope="simulator_evaluator_truth",
                trace_sha256=row["trace_sha256"],
                source_task=TASK_ID,
            )
    for row in result["summary_rows"]:
        for metric in METRICS:
            summary = row[metric]
            add(
                "seed_cluster_summary",
                family=row["family"],
                injected_rate=row["injected_rate"],
                metric=metric,
                value=summary["mean"],
                ci_low=summary["ci_low"],
                ci_high=summary["ci_high"],
                status_or_scope="whole_seed_cluster_bootstrap",
                source_task=TASK_ID,
            )
    for name, passed in result["gates"].items():
        add(
            "contract_gate",
            metric=name,
            value=passed,
            status_or_scope="PASS" if passed else "FAIL",
            source_task=TASK_ID,
        )
    return rows


def write_artifacts(
    *,
    artifact_path: str | Path = DEFAULT_ARTIFACT,
    source_data_path: str | Path = DEFAULT_SOURCE_DATA,
    config: CampaignConfig | None = None,
) -> dict[str, Any]:
    parents = load_parent_artifacts()
    integrity = inspect_parent_integrity(parents)
    result = build_report(parents, integrity, config)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["implementation_sha256"] = implementation_sha256()
    rows = _source_rows(result)
    source = _repo_path(source_data_path)
    source.parent.mkdir(parents=True, exist_ok=True)
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    result["source_data"] = {
        "path": Path(source_data_path).as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "row_count": len(rows),
        "row_types": sorted({row["row_type"] for row in rows}),
    }
    artifact = _repo_path(artifact_path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    args = parser.parse_args(argv)
    result = write_artifacts(
        artifact_path=args.artifact,
        source_data_path=args.source_data,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed_rows": len(result["seed_rows"]),
                "summary_rows": len(result["summary_rows"]),
                "gate_summary": result["gate_summary"],
                "source_rows": result["source_data"]["row_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_SEEDS",
    "CampaignConfig",
    "DEFAULT_ARTIFACT",
    "DEFAULT_SOURCE_DATA",
    "EVALUATION_SEEDS",
    "FAULT_FAMILIES",
    "INJECTION_RATES",
    "METRICS",
    "PARENT_ARTIFACTS",
    "PRIMARY_SOURCE_ANCHORS",
    "PRIMARY_SOURCE_PATH",
    "build_report",
    "implementation_sha256",
    "inspect_parent_integrity",
    "load_parent_artifacts",
    "validate_payload",
    "write_artifacts",
]
