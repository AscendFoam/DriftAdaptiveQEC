"""Independent coverage calibration for the Phase-9 paired-cluster UCB.

This design transaction never opens a formal cell ledger or raw archive.
It selects a multiplier-radius inflation factor on a calibration split and
then evaluates that frozen factor on a disjoint validation split containing
valid physical density matrices with analytically known population means.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.phase9_paired_cluster_uq import (
    half_trace_norm,
    paired_density_trace_ucb,
)


TASK_ID = "T-RISK-20260727-01"
CONFIG_PATH = "configs/phase9/t_risk_20260727_01_uq_calibration.json"
REPORT_PATH = "docs/t_risk_20260727_01_uq_calibration.json"
SOURCE_PATH = "docs/t_risk_20260727_01_uq_calibration_source_data.csv"
SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-CALIBRATION-REPORT-V1"
PASS_VERDICT = "PASS_PAIRED_CLUSTER_UQ_COVERAGE_CALIBRATED"
NO_GO_VERDICT = "NO_GO_PAIRED_CLUSTER_UQ_CALIBRATION"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version")
        != "PHASE9-PAIRED-CLUSTER-UQ-CALIBRATION-CONFIG-V1"
        or config.get("formal_outcomes_used_for_factor_selection") is not False
        or config.get("selected_factor") is not None
        or config.get("selected_formal_clusters_per_state") is not None
    ):
        raise ValueError("UQ calibration config identity or outcome firewall invalid")
    splits = config["splits"]
    calibration = splits["calibration"]
    validation = splits["validation"]
    namespaces = {
        int(calibration["seed_base"]),
        int(validation["seed_base"]),
        int(splits["multiplier_seed_base"]),
    }
    if (
        splits.get("all_intervals_disjoint") is not True
        or len(namespaces) != 3
        or min(namespaces) <= 1076767
    ):
        raise ValueError("calibration/validation/multiplier namespace firewall invalid")
    return config


def _seed(base: int, *parts: object) -> int:
    digest = sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    # The namespace occupies all bits above the low 64-bit digest.  Distinct
    # registered bases are therefore disjoint by construction, not merely
    # unlikely to collide after addition.
    return (int(base) << 64) | int.from_bytes(digest[:8], "big")


def _center_probabilities(dimension: int) -> np.ndarray:
    if dimension < 4:
        raise ValueError("calibration dimension must be at least four")
    center = np.full(dimension, 0.1 / (dimension - 2), dtype=np.float64)
    center[0] = 0.45
    center[1] = 0.45
    return center


def _fourier_unitary(dimension: int) -> np.ndarray:
    indices = np.arange(dimension)
    return np.exp(
        2j * np.pi * np.outer(indices, indices) / float(dimension)
    ) / math.sqrt(dimension)


def _phase_pure_noise(
    rng: np.random.Generator,
    *,
    count: int,
    center: np.ndarray,
    rare_probability: float,
) -> np.ndarray:
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(count, len(center)))
    kets = np.sqrt(center)[None, :] * np.exp(1j * phases)
    pure = np.einsum("ni,nj->nij", kets, kets.conj(), optimize=True)
    diagonal_center = np.diag(center).astype(np.complex128)
    if rare_probability < 1.0:
        active = rng.random(count) < rare_probability
        pure[~active] = diagonal_center
    return pure


def physical_density_trial(
    *,
    dimension: int,
    count: int,
    true_distance: float,
    family: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a valid density pair with an analytic population distance."""

    if not 0.0 <= true_distance <= 0.2:
        raise ValueError("true_distance outside calibration design")
    center = _center_probabilities(dimension)
    weight_left = float(family["left_noise_weight"])
    weight_right = float(family["right_noise_weight"])
    denominator = (2.0 - weight_left - weight_right) / 2.0
    shift = 0.0 if true_distance == 0.0 else true_distance / denominator
    if shift >= 0.85:
        raise ValueError("requested population shift violates PSD design envelope")
    direction = np.zeros(dimension, dtype=np.float64)
    direction[0] = shift / 2.0
    direction[1] = -shift / 2.0
    base_left = np.diag(center + direction).astype(np.complex128)
    base_right = np.diag(center - direction).astype(np.complex128)
    center_density = np.diag(center).astype(np.complex128)

    rng = np.random.default_rng(seed)
    noise_left = _phase_pure_noise(
        rng,
        count=count,
        center=center,
        rare_probability=float(family["rare_probability"]),
    )
    noise_right = _phase_pure_noise(
        rng,
        count=count,
        center=center,
        rare_probability=float(family["rare_probability"]),
    )
    left = (1.0 - weight_left) * base_left[None, :, :] + weight_left * noise_left
    right = (
        (1.0 - weight_right) * base_right[None, :, :]
        + weight_right * noise_right
    )
    population_left = (
        (1.0 - weight_left) * base_left + weight_left * center_density
    )
    population_right = (
        (1.0 - weight_right) * base_right + weight_right * center_density
    )
    analytic_distance = half_trace_norm(population_left - population_right)
    if abs(analytic_distance - true_distance) > 1e-12:
        raise RuntimeError("analytic density calibration target drift")

    if bool(family["coherent_unitary"]):
        unitary = _fourier_unitary(dimension)
        left = np.einsum(
            "ij,njk,lk->nil", unitary, left, unitary.conj(), optimize=True
        )
        right = np.einsum(
            "ij,njk,lk->nil", unitary, right, unitary.conj(), optimize=True
        )
    return left, right, analytic_distance


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid Wilson inputs")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    )
    return (center - radius) / denominator


def _simulate_split(
    config: Mapping[str, Any],
    *,
    split_name: str,
) -> list[dict[str, object]]:
    split = config["splits"][split_name]
    records: list[dict[str, object]] = []
    for family_name, family in config["families"].items():
        for dimension in config["dimensions"]:
            for count in config["cluster_counts_per_state"]:
                for true_distance in config["true_trace_distances"]:
                    for trial in range(int(split["trial_count"])):
                        trial_seed = _seed(
                            int(split["seed_base"]),
                            split_name,
                            family_name,
                            dimension,
                            count,
                            true_distance,
                            trial,
                        )
                        multiplier_seed = _seed(
                            int(config["splits"]["multiplier_seed_base"]),
                            split_name,
                            family_name,
                            dimension,
                            count,
                            true_distance,
                            trial,
                        )
                        left, right, truth = physical_density_trial(
                            dimension=int(dimension),
                            count=int(count),
                            true_distance=float(true_distance),
                            family=family,
                            seed=trial_seed,
                        )
                        ucb = paired_density_trace_ucb(
                            left,
                            right,
                            confidence=float(config["confidence"]),
                            multiplier_replicates=int(
                                config["multiplier_replicates"]
                            ),
                            seed=multiplier_seed,
                            calibration_factor=1.0,
                        )
                        records.append(
                            {
                                "split": split_name,
                                "family": family_name,
                                "dimension": int(dimension),
                                "cluster_count": int(count),
                                "true_distance": truth,
                                "trial": trial,
                                "trial_seed": trial_seed,
                                "multiplier_seed": multiplier_seed,
                                "estimate": ucb.estimate,
                                "raw_radius": ucb.raw_radius,
                                "power_primary": bool(family["power_primary"]),
                            }
                        )
    return records


def _coverage_cells(
    records: Sequence[Mapping[str, object]],
    *,
    factor: float,
    margin: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in records:
        key = (
            row["split"],
            row["family"],
            row["dimension"],
            row["cluster_count"],
            row["true_distance"],
        )
        grouped.setdefault(key, []).append(row)
    cells: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        bounds = [
            float(row["estimate"]) + factor * float(row["raw_radius"])
            for row in rows
        ]
        truth = float(key[4])
        coverage = sum(bound + 1e-12 >= truth for bound in bounds)
        equivalence = sum(bound <= margin for bound in bounds)
        cells.append(
            {
                "split": key[0],
                "family": key[1],
                "dimension": key[2],
                "cluster_count": key[3],
                "true_distance": truth,
                "trials": len(rows),
                "coverage_count": coverage,
                "coverage_rate": coverage / len(rows),
                "coverage_wilson_lcb": _wilson_lower(coverage, len(rows)),
                "equivalence_count": equivalence,
                "equivalence_rate": equivalence / len(rows),
                "factor": factor,
                "power_primary": bool(rows[0]["power_primary"]),
            }
        )
    return cells


def select_factor(
    config: Mapping[str, Any],
    calibration_records: Sequence[Mapping[str, object]],
) -> tuple[float | None, list[dict[str, object]]]:
    gates = config["selection_gates"]
    diagnostics: list[dict[str, object]] = []
    selected: float | None = None
    for factor in config["candidate_calibration_factors"]:
        cells = _coverage_cells(
            calibration_records,
            factor=float(factor),
            margin=float(config["margin"]),
        )
        min_rate = min(float(cell["coverage_rate"]) for cell in cells)
        min_lcb = min(float(cell["coverage_wilson_lcb"]) for cell in cells)
        passed = (
            min_rate >= float(gates["calibration_min_cell_coverage"])
            and min_lcb >= float(gates["calibration_min_cell_wilson_lcb"])
        )
        diagnostics.append(
            {
                "factor": float(factor),
                "minimum_cell_coverage": min_rate,
                "minimum_cell_wilson_lcb": min_lcb,
                "passed": passed,
            }
        )
        if selected is None and passed:
            selected = float(factor)
    return selected, diagnostics


def select_cluster_count(
    config: Mapping[str, Any],
    validation_cells: Sequence[Mapping[str, object]],
) -> tuple[int | None, list[dict[str, object]]]:
    gates = config["selection_gates"]
    diagnostics: list[dict[str, object]] = []
    selected: int | None = None
    for count in config["cluster_counts_per_state"]:
        primary = [
            row
            for row in validation_cells
            if int(row["cluster_count"]) == int(count)
            and bool(row["power_primary"])
        ]
        rates: dict[float, float] = {}
        for effect in config["true_trace_distances"]:
            values = [
                float(row["equivalence_rate"])
                for row in primary
                if abs(float(row["true_distance"]) - float(effect)) < 1e-12
            ]
            rates[float(effect)] = min(values)
        passed = (
            rates[0.0] >= float(gates["null_min_equivalence_rate"])
            and rates[0.05] >= float(gates["local_005_min_equivalence_rate"])
            and rates[0.1] <= float(gates["boundary_max_equivalence_rate"])
            and rates[0.12] <= float(gates["outside_max_equivalence_rate"])
        )
        diagnostics.append(
            {
                "cluster_count_per_state": int(count),
                "worst_primary_equivalence_rate_by_true_distance": {
                    str(key): value for key, value in sorted(rates.items())
                },
                "passed": passed,
            }
        )
        if selected is None and passed:
            selected = int(count)
    return selected, diagnostics


def build_report(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    config = load_config(root)
    calibration_records = _simulate_split(config, split_name="calibration")
    selected_factor, factor_diagnostics = select_factor(
        config, calibration_records
    )
    validation_records = _simulate_split(config, split_name="validation")
    calibration_trial_seeds = {
        int(row["trial_seed"]) for row in calibration_records
    }
    validation_trial_seeds = {
        int(row["trial_seed"]) for row in validation_records
    }
    multiplier_seeds = {
        int(row["multiplier_seed"])
        for row in (*calibration_records, *validation_records)
    }
    if (
        len(calibration_trial_seeds) != len(calibration_records)
        or len(validation_trial_seeds) != len(validation_records)
        or len(multiplier_seeds)
        != len(calibration_records) + len(validation_records)
        or calibration_trial_seeds & validation_trial_seeds
        or calibration_trial_seeds & multiplier_seeds
        or validation_trial_seeds & multiplier_seeds
    ):
        raise RuntimeError("materialized UQ calibration seed collision")
    if selected_factor is None:
        selected_factor = float(
            max(config["candidate_calibration_factors"])
        )
        factor_selection_passed = False
    else:
        factor_selection_passed = True
    validation_cells = _coverage_cells(
        validation_records,
        factor=selected_factor,
        margin=float(config["margin"]),
    )
    gates = config["selection_gates"]
    coverage_passed = all(
        float(row["coverage_rate"])
        >= float(gates["validation_min_cell_coverage"])
        and float(row["coverage_wilson_lcb"])
        >= float(gates["validation_min_cell_wilson_lcb"])
        for row in validation_cells
    )
    selected_count, count_diagnostics = select_cluster_count(
        config, validation_cells
    )
    verdict = (
        PASS_VERDICT
        if factor_selection_passed
        and coverage_passed
        and selected_count is not None
        else NO_GO_VERDICT
    )
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "verdict": verdict,
        "formal_outcomes_accessed": False,
        "factor_selected_from_split": "calibration",
        "factor_validated_on_split": "validation",
        "selected_calibration_factor": (
            selected_factor if factor_selection_passed else None
        ),
        "selected_formal_clusters_per_state": (
            selected_count if verdict == PASS_VERDICT else None
        ),
        "statistical_contract": {
            "estimand": "0.5*trace_norm(E[rho_left-rho_right])",
            "radius": (
                "Rademacher paired-cluster multiplier quantile of "
                "0.5*trace_norm(centered matrix mean error)"
            ),
            "ucb": "point + calibrated_radius + quantization_bound",
            "confidence": config["confidence"],
            "multiplier_replicates_design": config["multiplier_replicates"],
            "formal_replicates_minimum": 1999,
            "aggregation": "state×scenario×backend IUT/max; no state average rescue",
        },
        "factor_diagnostics": factor_diagnostics,
        "validation_coverage_summary": {
            "minimum_cell_coverage": min(
                float(row["coverage_rate"]) for row in validation_cells
            ),
            "minimum_cell_wilson_lcb": min(
                float(row["coverage_wilson_lcb"]) for row in validation_cells
            ),
            "all_cells_passed": coverage_passed,
        },
        "cluster_count_diagnostics": count_diagnostics,
        "density_validity": {
            "psd_by_construction": True,
            "trace_one_by_construction": True,
            "analytic_population_distance": True,
            "families": sorted(config["families"]),
            "dimensions": config["dimensions"],
        },
        "split_firewall": {
            **config["splits"],
            "materialized_calibration_trial_seed_count": len(
                calibration_trial_seeds
            ),
            "materialized_validation_trial_seed_count": len(
                validation_trial_seeds
            ),
            "materialized_multiplier_seed_count": len(multiplier_seeds),
            "materialized_collision_count": 0,
        },
        "claim_state": dict(config["claim_boundary"]),
        "bindings": {
            "config": _binding(root / CONFIG_PATH, root),
            "calibration_source": _binding(Path(__file__).resolve(), root),
            "paired_cluster_uq_source": _binding(
                root
                / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py",
                root,
            ),
        },
        "generator_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    report["analysis_sha256"] = _sha(report)
    source_rows = [
        {
            **row,
            "selected_factor": selected_factor,
            "calibrated_upper_bound": float(row["estimate"])
            + selected_factor * float(row["raw_radius"]),
            "covered": (
                float(row["estimate"])
                + selected_factor * float(row["raw_radius"])
                + 1e-12
                >= float(row["true_distance"])
            ),
            "equivalent": (
                float(row["estimate"])
                + selected_factor * float(row["raw_radius"])
                <= float(config["margin"])
            ),
        }
        for row in (*calibration_records, *validation_records)
    ]
    return report, source_rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    report, rows = build_report(base)
    _atomic_text(
        base / REPORT_PATH,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        _atomic_text(base / SOURCE_PATH, stream.read())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate the Phase-9 paired-cluster density UCB."
    )
    parser.parse_args(argv)
    report = write_artifacts()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "selected_calibration_factor": report[
                    "selected_calibration_factor"
                ],
                "selected_formal_clusters_per_state": report[
                    "selected_formal_clusters_per_state"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "NO_GO_VERDICT",
    "PASS_VERDICT",
    "REPORT_PATH",
    "SOURCE_PATH",
    "build_report",
    "load_config",
    "physical_density_trial",
    "select_cluster_count",
    "select_factor",
    "write_artifacts",
]
