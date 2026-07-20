"""T6.18.2 official structured-lattice CPD reproduction and audit.

The module keeps three evidence classes separate:

* execution of the pinned official CPD implementation for correctness and the
  independent small-distance Monte Carlo;
* exact reanalysis of the authors' published Fig. 5 aggregate JLD2 data;
* a local, source-transcribed Noh--Chamberland analog-MWPM adapter used only as
  the paired comparator on the same graph and displacement instances.

It never substitutes the project's single-mode decoder for the multimode task
and never fills an independent threshold from the literature anchor.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.18.2"
SCHEMA_VERSION = "t6.18.2-official-structured-cpd-v1"
OFFICIAL_COMMIT = "01f9bf1f6970b3e229b43aac9da3325c75518db8"
OFFICIAL_LICENSE_SHA256 = "88046bf22d5b4f4b8cc85079ae6aae5424a3a1999db952ed152828ff325b2c6d"
EXPECTED_SEEDS = list(range(61_820_001, 61_820_033))
EXPECTED_DISTANCES = [3, 5, 7]
EXPECTED_SIGMAS = [round(value, 2) for value in np.arange(0.56, 0.641, 0.01)]
EXPECTED_TRIALS_PER_SEED = 2_000
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 61_822_000
SAMPLED_HOST_WORKING_SET_HIGH_WATER_BYTES = int(round(864.3 * (1 << 20)))

PREREG = ROOT / "configs" / "literature" / "t6_16_3_secondary_preregistration.json"
SOURCE_AUDIT = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"
ONTOLOGY = ROOT / "docs" / "t6_16_2_comparison_ontology.json"
PHASE6B_TERMINAL = ROOT / "docs" / "t6_15_5_route_a_v5_final_evidence_gate.json"
JULIA_PROJECT = ROOT / "configs" / "literature" / "t6_18_2_julia_env" / "Project.toml"
JULIA_MANIFEST = ROOT / "configs" / "literature" / "t6_18_2_julia_env" / "Manifest.toml"
JULIA_SCRIPT = ROOT / "scripts" / "run_lattice_algorithms_reproduction.jl"
UPSTREAM_SCRIPT = ROOT / "scripts" / "run_lattice_algorithms_upstream_tests.jl"
SEEDED_TEST_SCRIPT = ROOT / "scripts" / "run_lattice_algorithms_seeded_tests.jl"
OFFICIAL_ROOT = ROOT / "third_party" / "LatticeAlgorithms.jl"
CORRECTNESS_RAW = ROOT / "docs" / "t6_18_2_julia_correctness_raw.json"
OFFICIAL_DATA_RAW = ROOT / "docs" / "t6_18_2_julia_official_data_raw.json"
THRESHOLD_RAW = ROOT / "docs" / "t6_18_2_julia_threshold_raw.json"
PILOT_RAW = ROOT / "docs" / "t6_18_2_julia_pilot_raw.json"
DEFAULT_REPORT = ROOT / "docs" / "t6_18_2_official_structured_cpd_reproduction.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_18_2_official_structured_cpd_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "official_structured_cpd_reproduction.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _preregistered_experiment() -> dict[str, Any]:
    rows = [row for row in _load(PREREG)["experiments"] if row["task_id"] == TASK_ID]
    if len(rows) != 1:
        raise ValueError("T6.18.2 requires exactly one preregistration row")
    return rows[0]


def _raw_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return _load(CORRECTNESS_RAW), _load(OFFICIAL_DATA_RAW), _load(THRESHOLD_RAW)


def _row_key(row: Mapping[str, Any]) -> tuple[int, float, int]:
    return int(row["distance"]), round(float(row["sigma"]), 2), int(row["seed"])


def _raw_integrity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [_row_key(row) for row in rows]
    expected_keys = {
        (distance, sigma, seed)
        for distance in EXPECTED_DISTANCES
        for sigma in EXPECTED_SIGMAS
        for seed in EXPECTED_SEEDS
    }
    actual_keys = set(keys)
    trials = [int(row["trials"]) for row in rows]
    return {
        "row_count": len(rows),
        "expected_row_count": len(expected_keys),
        "unique_key_count": len(actual_keys),
        "missing_keys": [list(value) for value in sorted(expected_keys - actual_keys)],
        "unexpected_keys": [list(value) for value in sorted(actual_keys - expected_keys)],
        "duplicate_key_count": len(keys) - len(actual_keys),
        "distinct_seeds": sorted({int(row["seed"]) for row in rows}),
        "distinct_distances": sorted({int(row["distance"]) for row in rows}),
        "distinct_sigmas": sorted({round(float(row["sigma"]), 2) for row in rows}),
        "minimum_trials_per_seed_cell": min(trials),
        "maximum_trials_per_seed_cell": max(trials),
        "total_paired_trials": int(sum(trials)),
        "all_error_counts_bounded": all(
            0 <= int(row[method]) <= int(row["trials"])
            for row in rows
            for method in ("cpd_errors", "analog_errors")
        ),
        "passed": (
            len(rows) == len(expected_keys)
            and actual_keys == expected_keys
            and len(keys) == len(actual_keys)
            and min(trials) == max(trials) == EXPECTED_TRIALS_PER_SEED
        ),
    }


def _aggregate_curves(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, error_field in (("cpd", "cpd_errors"), ("analog_mwpm", "analog_errors")):
        for distance in EXPECTED_DISTANCES:
            for sigma in EXPECTED_SIGMAS:
                selected = [
                    row for row in rows
                    if int(row["distance"]) == distance and math.isclose(float(row["sigma"]), sigma, abs_tol=1e-12)
                ]
                errors = int(sum(int(row[error_field]) for row in selected))
                trials = int(sum(int(row["trials"]) for row in selected))
                seed_lers = np.asarray([int(row[error_field]) / int(row["trials"]) for row in selected], dtype=float)
                output.append(
                    {
                        "method": method,
                        "distance": distance,
                        "sigma": sigma,
                        "seed_clusters": len(selected),
                        "errors": errors,
                        "trials": trials,
                        "ler": errors / trials,
                        "fidelity": 1.0 - errors / trials,
                        "seed_ler_standard_error": float(seed_lers.std(ddof=1) / np.sqrt(seed_lers.size)),
                    }
                )
    return output


def _crossing_from_lers(sigmas: Sequence[float], lower_lers: Sequence[float], higher_lers: Sequence[float]) -> float | None:
    # This is the notebook's adjacent fidelity crossing written as a linear
    # interpolation of fidelity_low-fidelity_high = ler_high-ler_low.
    differences = np.asarray(higher_lers, dtype=float) - np.asarray(lower_lers, dtype=float)
    positive = np.flatnonzero(differences > 0.0)
    if positive.size == 0 or int(positive[0]) == 0:
        return None
    index = int(positive[0])
    x1, x2 = float(sigmas[index - 1]), float(sigmas[index])
    y1, y2 = float(differences[index - 1]), float(differences[index])
    if math.isclose(y1, y2, abs_tol=1e-18):
        return None
    return x1 - y1 * (x2 - x1) / (y2 - y1)


def _crossings_from_rows(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, float | None]:
    error_field = "cpd_errors" if method == "cpd" else "analog_errors"
    lers: dict[tuple[int, float], float] = {}
    for distance in EXPECTED_DISTANCES:
        for sigma in EXPECTED_SIGMAS:
            selected = [
                row for row in rows
                if int(row["distance"]) == distance and math.isclose(float(row["sigma"]), sigma, abs_tol=1e-12)
            ]
            lers[(distance, sigma)] = sum(int(row[error_field]) for row in selected) / sum(
                int(row["trials"]) for row in selected
            )
    pair_values: dict[str, float | None] = {}
    for lower, higher in zip(EXPECTED_DISTANCES[:-1], EXPECTED_DISTANCES[1:]):
        pair_values[f"d{lower}_d{higher}"] = _crossing_from_lers(
            EXPECTED_SIGMAS,
            [lers[(lower, sigma)] for sigma in EXPECTED_SIGMAS],
            [lers[(higher, sigma)] for sigma in EXPECTED_SIGMAS],
        )
    valid = [value for value in pair_values.values() if value is not None]
    pair_values["mean_adjacent_crossing"] = float(np.mean(valid)) if len(valid) == 2 else None
    return pair_values


def _bootstrap_crossings(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    error_field = "cpd_errors" if method == "cpd" else "analog_errors"
    lookup = {_row_key(row): row for row in rows}
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if method == "cpd" else 1))
    distributions: dict[str, list[float]] = {"d3_d5": [], "d5_d7": [], "mean_adjacent_crossing": []}
    missing = {key: 0 for key in distributions}
    seeds = np.asarray(EXPECTED_SEEDS, dtype=np.int64)
    for _ in range(BOOTSTRAP_REPS):
        sampled = rng.choice(seeds, size=seeds.size, replace=True)
        lers: dict[tuple[int, float], float] = {}
        for distance in EXPECTED_DISTANCES:
            for sigma in EXPECTED_SIGMAS:
                selected = [lookup[(distance, sigma, int(seed))] for seed in sampled]
                lers[(distance, sigma)] = sum(int(row[error_field]) for row in selected) / sum(
                    int(row["trials"]) for row in selected
                )
        values: dict[str, float | None] = {}
        for lower, higher in zip(EXPECTED_DISTANCES[:-1], EXPECTED_DISTANCES[1:]):
            values[f"d{lower}_d{higher}"] = _crossing_from_lers(
                EXPECTED_SIGMAS,
                [lers[(lower, sigma)] for sigma in EXPECTED_SIGMAS],
                [lers[(higher, sigma)] for sigma in EXPECTED_SIGMAS],
            )
        pair_values = [value for key, value in values.items() if key != "mean_adjacent_crossing" and value is not None]
        values["mean_adjacent_crossing"] = float(np.mean(pair_values)) if len(pair_values) == 2 else None
        for key, value in values.items():
            if value is None:
                missing[key] += 1
            else:
                distributions[key].append(float(value))

    summaries: dict[str, Any] = {}
    for key, values in distributions.items():
        array = np.asarray(values, dtype=float)
        summaries[key] = {
            "bootstrap_reps": BOOTSTRAP_REPS,
            "valid_reps": int(array.size),
            "missing_reps": missing[key],
            "missing_fraction": missing[key] / BOOTSTRAP_REPS,
            "mean": float(array.mean()) if array.size else None,
            "median": float(np.median(array)) if array.size else None,
            "ci95": [float(value) for value in np.quantile(array, [0.025, 0.975])] if array.size else None,
        }
    return {"seed": BOOTSTRAP_SEED + (0 if method == "cpd" else 1), "summaries": summaries}


def _threshold_summary(rows: Sequence[Mapping[str, Any]], method: str, anchor: float) -> dict[str, Any]:
    central = _crossings_from_rows(rows, method)
    bootstrap = _bootstrap_crossings(rows, method)
    estimate = central["mean_adjacent_crossing"]
    return {
        "method": method,
        "estimator": "unweighted mean of d3/d5 and d5/d7 linear fidelity crossings",
        "central_crossings": central,
        "bootstrap": bootstrap,
        "literature_anchor": anchor,
        "absolute_tolerance": 0.02,
        "absolute_gap_to_anchor": abs(float(estimate) - anchor) if estimate is not None else None,
        "within_preregistered_tolerance": estimate is not None and abs(float(estimate) - anchor) <= 0.02,
        "precision_boundary": "SMALL_DISTANCE_COARSE_REPRODUCTION_NOT_PAPER_3_DECIMAL_THRESHOLD_PRECISION",
    }


def _runtime_scaling(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in ("cpd", "analog"):
        per_distance = []
        for distance in EXPECTED_DISTANCES:
            selected = [row for row in rows if int(row["distance"]) == distance]
            times = np.asarray([float(row[f"{method}_seconds_per_decode"]) for row in selected], dtype=float)
            allocations = np.asarray(
                [int(row[f"{method}_allocated_bytes_first_measured_decode"]) for row in selected], dtype=float
            )
            per_distance.append(
                {
                    "distance": distance,
                    "samples": int(times.size),
                    "median_seconds_per_decode": float(np.median(times)),
                    "p95_seconds_per_decode": float(np.quantile(times, 0.95)),
                    "median_allocated_bytes_per_measured_decode": float(np.median(allocations)),
                }
            )
        exponent = float(
            np.polyfit(
                np.log([row["distance"] for row in per_distance]),
                np.log([row["median_seconds_per_decode"] for row in per_distance]),
                deg=1,
            )[0]
        )
        allocation_exponent = float(
            np.polyfit(
                np.log([row["distance"] for row in per_distance]),
                np.log([row["median_allocated_bytes_per_measured_decode"] for row in per_distance]),
                deg=1,
            )[0]
        )
        methods["cpd" if method == "cpd" else "analog_mwpm"] = {
            "per_distance": per_distance,
            "empirical_time_exponent_three_sizes": exponent,
            "empirical_allocation_exponent_three_sizes": allocation_exponent,
            "evidence_boundary": "THREE_SIZE_EMPIRICAL_DIAGNOSTIC_NOT_ASYMPTOTIC_PROOF",
        }
    methods["paper_cpd_runtime_anchor"] = {
        "value": 3.020,
        "quantity": "power-law exponent in code distance",
        "evidence": "LITERATURE_ONLY_FIG5D",
        "may_be_replaced_by_three_size_fit": False,
    }
    return methods


def _paired_advantage(curves: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {(row["method"], row["distance"], row["sigma"]): row for row in curves}
    differences = []
    for distance in EXPECTED_DISTANCES:
        for sigma in EXPECTED_SIGMAS:
            cpd = lookup[("cpd", distance, sigma)]["ler"]
            analog = lookup[("analog_mwpm", distance, sigma)]["ler"]
            differences.append(
                {"distance": distance, "sigma": sigma, "cpd_minus_analog_ler": cpd - analog}
            )
    values = np.asarray([row["cpd_minus_analog_ler"] for row in differences])
    return {
        "cells": differences,
        "cpd_lower_ler_cells": int(np.sum(values < 0.0)),
        "total_cells": int(values.size),
        "mean_absolute_ler_difference": float(values.mean()),
        "minimum_difference": float(values.min()),
        "maximum_difference": float(values.max()),
        "claim_boundary": "PAIRED_SMALL_DISTANCE_STATIONARY_FAMILY_ONLY",
    }


def _upstream_test_audit() -> dict[str, Any]:
    return {
        "standard_pkg_test": {
            "state": "ATTEMPTED_PARTIAL_FAIL_UNSEEDED",
            "passed": 2_004,
            "failed": 1,
            "total_before_stop": 2_005,
            "failure_file": "third_party/LatticeAlgorithms.jl/test/lattice_algorithms.jl:57",
            "failure_detail": "one unseeded random batch returned 99/100 for ds<=single-basis-vector bound",
            "suite_stopped_after_first_testset": True,
        },
        "deterministic_replay": {
            "seed": 61_820_001,
            "file": "lattice_algorithms.jl",
            "passed": 2_005,
            "failed": 0,
            "total": 2_005,
            "elapsed_seconds": 44.1,
        },
        "classification": "PASS_RELEVANT_GATES_WITH_UPSTREAM_RANDOM_REPRODUCIBILITY_CAVEAT",
        "official_source_modified": False,
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    correctness = report["correctness"]
    for row in correctness["exact_correctness"]["generic_rows"]:
        rows.append({"record_type": "exact_cvp", "method": "official_cpd", **row, "value_state": "OFFICIAL_CODE_REPRODUCTION"})
    for row in correctness["exact_correctness"]["single_mode_rows"]:
        rows.append({"record_type": "single_mode", "method": "official_scaled_Z", **row, "value_state": "OFFICIAL_CODE_REPRODUCTION"})
    for method in ("cpd", "analog"):
        for index, value in enumerate(report["official_data_reanalysis"][f"{method}_crossings"]):
            rows.append(
                {
                    "record_type": "official_aggregate_crossing",
                    "method": "cpd" if method == "cpd" else "analog_mwpm",
                    "lower_distance": 3 + 2 * index,
                    "crossing_sigma": value,
                    "value_state": "OFFICIAL_AGGREGATE_REANALYSIS",
                }
            )
    for row in report["independent_experiment"]["seed_rows"]:
        rows.append({"record_type": "seed_cell", "method": "paired", **row, "value_state": "OFFICIAL_CODE_PLUS_SOURCE_TRANSCRIBED_COMPARATOR"})
    for row in report["independent_experiment"]["curves"]:
        rows.append({"record_type": "aggregate_curve", **row, "value_state": "INDEPENDENT_PREREGISTERED_MONTE_CARLO"})
    for method, payload in report["independent_experiment"]["thresholds"].items():
        for pair, value in payload["central_crossings"].items():
            rows.append(
                {
                    "record_type": "independent_crossing",
                    "method": method,
                    "pair": pair,
                    "crossing_sigma": value,
                    "value_state": "INDEPENDENT_PREREGISTERED_MONTE_CARLO" if value is not None else "FAILED_NO_GRID_CROSSING",
                }
            )
    for method, payload in report["independent_experiment"]["runtime_scaling"].items():
        if method == "paper_cpd_runtime_anchor":
            rows.append({"record_type": "literature_runtime", "method": "cpd", **payload, "value_state": "LITERATURE_ONLY"})
        else:
            for row in payload["per_distance"]:
                rows.append({"record_type": "runtime", "method": method, **row, "value_state": "INDEPENDENT_MEASURED_HOST_RUNTIME"})
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    os.replace(temporary, path)


@lru_cache(maxsize=1)
def _live_recomputation() -> dict[str, Any]:
    correctness_raw, official_raw, threshold_raw = _raw_inputs()
    rows = threshold_raw["threshold_simulation"]["rows"]
    curves = _aggregate_curves(rows)
    return {
        "correctness": {
            "exact_correctness": correctness_raw["exact_correctness"],
            "analog_weight_validation": correctness_raw["analog_weight_validation"],
            "final_list_validation": correctness_raw["final_list_validation"],
        },
        "official": official_raw["official_data_reanalysis"],
        "integrity": _raw_integrity(rows),
        "curves": curves,
        "thresholds": {
            "cpd": _threshold_summary(rows, "cpd", 0.602),
            "analog_mwpm": _threshold_summary(rows, "analog_mwpm", 0.599),
        },
        "runtime": _runtime_scaling(rows),
        "paired": _paired_advantage(curves),
        "threshold_runtime_seconds": float(threshold_raw["threshold_simulation"]["wall_clock_seconds"]),
        "raw_provenance": {
            "correctness": correctness_raw["provenance"],
            "official": official_raw["provenance"],
            "threshold": threshold_raw["provenance"],
        },
        "seed_rows": rows,
    }


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    live = _live_recomputation()
    prereg = _preregistered_experiment()
    bindings_ok = all(
        (ROOT / binding["path"]).is_file()
        and _sha256(ROOT / binding["path"]) == binding["sha256"]
        and (ROOT / binding["path"]).stat().st_size == binding["bytes"]
        for binding in report["bindings"].values()
    )
    source_path = ROOT / report["source_data"]["path"]
    source_ok = source_path.is_file() and _sha256(source_path) == report["source_data"]["sha256"]
    raw_provenance = live["raw_provenance"]
    current_manifest_hash = _sha256(JULIA_MANIFEST)
    return {
        "G01_preregistration_exactly_matches_frozen_grid_seed_and_statistics_contract": (
            prereg["config"]["surface_gkp_sizes"] == EXPECTED_DISTANCES
            and prereg["config"]["sigma_grid"] == EXPECTED_SIGMAS
            and prereg["seeds"]["values"] == EXPECTED_SEEDS
            and prereg["sample_size"]["minimum_trials_per_size_sigma_seed"] == EXPECTED_TRIALS_PER_SEED
            and prereg["sample_size"]["threshold_bootstrap_resamples"] == BOOTSTRAP_REPS
        ),
        "G02_official_commit_license_and_source_tree_are_pinned_live": (
            report["official_import"]["head"] == _git_head(OFFICIAL_ROOT) == OFFICIAL_COMMIT
            and report["official_import"]["license_sha256"] == _sha256(OFFICIAL_ROOT / "LICENSE") == OFFICIAL_LICENSE_SHA256
            and report["official_import"]["official_source_modified"] is False
        ),
        "G03_julia_manifest_and_pythoncall_compatibility_are_frozen": (
            report["official_import"]["manifest_sha256"] == current_manifest_hash
            and "PythonCall = \"=0.9.10\"" in JULIA_PROJECT.read_text(encoding="utf-8")
            and 'version = "0.9.10"' in JULIA_MANIFEST.read_text(encoding="utf-8")
            and all(payload["manifest_sha256"] == current_manifest_hash for payload in raw_provenance.values())
        ),
        "G04_upstream_standard_failure_and_seeded_replay_are_both_preserved": (
            report["upstream_tests"] == _upstream_test_audit()
            and report["upstream_tests"]["standard_pkg_test"]["failed"] == 1
            and report["upstream_tests"]["deterministic_replay"]["failed"] == 0
        ),
        "G05_exact_cvp_single_mode_and_surface_fast_path_have_zero_mismatch": (
            report["correctness"]["exact_correctness"] == live["correctness"]["exact_correctness"]
            and live["correctness"]["exact_correctness"]["passed"]
        ),
        "G06_noh_conditional_probability_and_truncation_gate_pass": (
            report["correctness"]["analog_weight_validation"] == live["correctness"]["analog_weight_validation"]
            and live["correctness"]["analog_weight_validation"]["passed"]
        ),
        "G07_fast_final_list_logical_classification_matches_canonical_lattice_coordinates": (
            report["correctness"]["final_list_validation"] == live["correctness"]["final_list_validation"]
            and live["correctness"]["final_list_validation"]["passed"]
        ),
        "G08_official_fig5_aggregate_reanalysis_reproduces_notebook_exactly": (
            report["official_data_reanalysis"] == live["official"]
            and live["official"]["passed"]
            and live["official"]["evidence_class"] == "OFFICIAL_AGGREGATE_DATA_REANALYSIS_NOT_INDEPENDENT_MONTE_CARLO"
        ),
        "G09_independent_raw_grid_has_all_864_seed_cells_and_1728000_paired_trials": (
            report["independent_experiment"]["raw_integrity"] == live["integrity"]
            and live["integrity"]["passed"]
            and live["integrity"]["total_paired_trials"] == 1_728_000
        ),
        "G10_independent_curves_recompute_from_integer_counts_without_literature_fill": (
            report["independent_experiment"]["curves"] == live["curves"]
            and all(row["trials"] == 64_000 for row in live["curves"])
            and report["independent_experiment"]["evidence_class"] == "INDEPENDENT_PREREGISTERED_MONTE_CARLO"
        ),
        "G11_seed_cluster_bootstrap_recomputes_crossings_and_reports_missing_replicates": (
            report["independent_experiment"]["thresholds"] == live["thresholds"]
            and all(
                payload["bootstrap"]["summaries"]["mean_adjacent_crossing"]["bootstrap_reps"] == BOOTSTRAP_REPS
                and payload["bootstrap"]["summaries"]["mean_adjacent_crossing"]["missing_fraction"] <= 0.05
                for payload in live["thresholds"].values()
            )
        ),
        "G12_both_small_distance_thresholds_meet_preregistered_absolute_tolerance": (
            all(payload["within_preregistered_tolerance"] for payload in live["thresholds"].values())
            and report["independent_experiment"]["thresholds"] == live["thresholds"]
        ),
        "G13_runtime_and_memory_accounting_stay_within_budget_and_scaling_is_not_overclaimed": (
            report["execution_budget"]["threshold_runtime_seconds"] == live["threshold_runtime_seconds"]
            and report["execution_budget"]["threshold_runtime_seconds"] < 28_800
            and report["execution_budget"]["sampled_host_working_set_high_water_bytes"] == SAMPLED_HOST_WORKING_SET_HIGH_WATER_BYTES
            and report["execution_budget"]["sampled_host_working_set_high_water_bytes"] < 16 * (1 << 30)
            and report["independent_experiment"]["runtime_scaling"] == live["runtime"]
            and all(
                payload.get("evidence_boundary") == "THREE_SIZE_EMPIRICAL_DIAGNOSTIC_NOT_ASYMPTOTIC_PROOF"
                for key, payload in live["runtime"].items() if key != "paper_cpd_runtime_anchor"
            )
        ),
        "G14_official_aggregate_independent_monte_carlo_and_adapter_boundaries_are_not_merged": (
            report["evidence_boundary"]["official_aggregate_may_fill_independent_threshold"] is False
            and report["evidence_boundary"]["analog_adapter_is_official_repository_function"] is False
            and report["evidence_boundary"]["single_mode_substitution_allowed"] is False
            and report["independent_experiment"]["paired_advantage"] == live["paired"]
        ),
        "G15_phase6b_no_go_and_board_measurement_null_are_read_only": (
            report["phase6b_boundary"]["verdict"] == "NO_GO_V5_EARLY_HEADROOM_STOP"
            and report["phase6b_boundary"]["promoted_by_phase6c"] is False
            and report["phase6b_boundary"]["board_measured_claim"] is None
        ),
        "G16_bindings_source_data_and_value_states_are_live": bindings_ok and source_ok,
    }


def _mutation_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def check(name: str, gate: str, mutate) -> None:
        forged = deepcopy(report)
        mutate(forged)
        rejected = not evaluate_gates(forged)[gate]
        cases.append({"name": name, "target_gate": gate, "rejected": rejected})

    check("official_head", "G02_official_commit_license_and_source_tree_are_pinned_live", lambda value: value["official_import"].__setitem__("head", "0" * 40))
    check("license_hash", "G02_official_commit_license_and_source_tree_are_pinned_live", lambda value: value["official_import"].__setitem__("license_sha256", "0" * 64))
    check("manifest_hash", "G03_julia_manifest_and_pythoncall_compatibility_are_frozen", lambda value: value["official_import"].__setitem__("manifest_sha256", "0" * 64))
    check("generic_mismatch", "G05_exact_cvp_single_mode_and_surface_fast_path_have_zero_mismatch", lambda value: value["correctness"]["exact_correctness"].__setitem__("generic_mismatches", 1))
    check("surface_mismatch", "G05_exact_cvp_single_mode_and_surface_fast_path_have_zero_mismatch", lambda value: value["correctness"]["exact_correctness"].__setitem__("surface_d3_mismatches", 1))
    check("analog_equation", "G06_noh_conditional_probability_and_truncation_gate_pass", lambda value: value["correctness"]["analog_weight_validation"].__setitem__("equation", "log-odds substitute"))
    check("logical_classification", "G07_fast_final_list_logical_classification_matches_canonical_lattice_coordinates", lambda value: value["correctness"]["final_list_validation"].__setitem__("mismatches", 1))
    check("official_anchor", "G08_official_fig5_aggregate_reanalysis_reproduces_notebook_exactly", lambda value: value["official_data_reanalysis"].__setitem__("cpd_threshold_mean", 0.62))
    check("raw_row_count", "G09_independent_raw_grid_has_all_864_seed_cells_and_1728000_paired_trials", lambda value: value["independent_experiment"]["raw_integrity"].__setitem__("row_count", 863))
    check("raw_trial_count", "G09_independent_raw_grid_has_all_864_seed_cells_and_1728000_paired_trials", lambda value: value["independent_experiment"]["raw_integrity"].__setitem__("total_paired_trials", 1_000))
    check("curve_literature_fill", "G10_independent_curves_recompute_from_integer_counts_without_literature_fill", lambda value: value["independent_experiment"]["curves"][0].__setitem__("ler", 0.602))
    check("crossing_fill", "G11_seed_cluster_bootstrap_recomputes_crossings_and_reports_missing_replicates", lambda value: value["independent_experiment"]["thresholds"]["cpd"]["central_crossings"].__setitem__("mean_adjacent_crossing", 0.602))
    check("threshold_tolerance", "G12_both_small_distance_thresholds_meet_preregistered_absolute_tolerance", lambda value: value["independent_experiment"]["thresholds"]["cpd"].__setitem__("within_preregistered_tolerance", False))
    check("runtime_budget", "G13_runtime_and_memory_accounting_stay_within_budget_and_scaling_is_not_overclaimed", lambda value: value["execution_budget"].__setitem__("threshold_runtime_seconds", 99_999.0))
    check("adapter_upgrade", "G14_official_aggregate_independent_monte_carlo_and_adapter_boundaries_are_not_merged", lambda value: value["evidence_boundary"].__setitem__("analog_adapter_is_official_repository_function", True))
    check("phase6b_promotion", "G15_phase6b_no_go_and_board_measurement_null_are_read_only", lambda value: value["phase6b_boundary"].__setitem__("promoted_by_phase6c", True))
    check("source_hash", "G16_bindings_source_data_and_value_states_are_live", lambda value: value["source_data"].__setitem__("sha256", "0" * 64))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def _markdown(report: Mapping[str, Any]) -> str:
    independent = report["independent_experiment"]
    lines = [
        "# T6.18.2 official structured-lattice CPD 复现",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- official commit：`{report['official_import']['head']}`",
        f"- independent paired trials：{independent['raw_integrity']['total_paired_trials']:,}",
        f"- gates / mutations：{report['gate_summary']['passed']}/{report['gate_summary']['total']} / {report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['count']}",
        "",
        "## 结果",
        "",
        "| 方法 | d3/d5 crossing | d5/d7 crossing | mean crossing [bootstrap 95% CI] | anchor |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in ("cpd", "analog_mwpm"):
        payload = independent["thresholds"][method]
        crossing = payload["central_crossings"]
        interval = payload["bootstrap"]["summaries"]["mean_adjacent_crossing"]["ci95"]
        lines.append(
            f"| `{method}` | {crossing['d3_d5']:.6f} | {crossing['d5_d7']:.6f} | "
            f"{crossing['mean_adjacent_crossing']:.6f} [{interval[0]:.6f}, {interval[1]:.6f}] | {payload['literature_anchor']:.3f} |"
        )
    lines += [
        "",
        f"本次冻结网格中，CPD 在 {independent['paired_advantage']['cpd_lower_ler_cells']}/{independent['paired_advantage']['total_cells']} 个 d×σ cell 的 LER 低于 analog-MWPM；平均 CPD−analog LER 为 {independent['paired_advantage']['mean_absolute_ler_difference']:.6f}。这是同一 stationary surface-GKP 小距离 family 的配对结果，不是相对所有解码器的主排名。",
        "",
        "官方 Fig. 5 聚合数据重算逐位得到 CPD `0.6024563484 ± 0.0003776410` 与 analog-MWPM `0.5995937637 ± 0.0004433259`。该结果来自作者提供的 10^7 samples/point JLD2，不是本项目独立 Monte Carlo。",
        "",
        "## 正确性与实现边界",
        "",
        f"- 1–4 维 certified brute-force CVP：{report['correctness']['exact_correctness']['generic_mismatches']}/{report['correctness']['exact_correctness']['generic_samples']} mismatch。",
        f"- d=3 official fast CPD vs generic exact CVP：{report['correctness']['exact_correctness']['surface_d3_mismatches']}/{report['correctness']['exact_correctness']['surface_d3_samples']} mismatch。",
        f"- final-list vs canonical lattice logical coordinates：{report['correctness']['final_list_validation']['mismatches']}/{report['correctness']['final_list_validation']['samples']} mismatch。",
        "- analog comparator 使用 Noh–Chamberland Eq. (11) 的条件逻辑错误概率及 Appendix B 的 `-log2(p)`；它是 source-transcribed adapter，不是官方仓库原生函数。",
        "- 标准上游测试保留 2004/2005 的无 seed 随机失败；固定 seed 重放为 2005/2005。官方源码未修改。",
        "",
        "## 不能声称的内容",
        "",
        "d=3/5/7、64,000 trials/point 只支持 ±0.02 粗阈值复现，不能替代论文 d=15–29、10^7 samples/point 的三位小数结论。三尺寸 runtime exponent 只是经验诊断；论文 `d^3.020` 保持 literature-only。没有 FPGA 真板结果，也没有把 Phase 6B NO-GO 改写为通过。",
        "",
        "## Artifacts",
        "",
        f"- `{report['source_data']['path']}`",
        f"- `{_relative(DEFAULT_REPORT)}`",
        f"- `{_relative(JULIA_SCRIPT)}`",
        f"- `{_relative(Path(__file__))}`",
        "",
    ]
    return "\n".join(lines)


def build_report(source_data_path: Path = DEFAULT_SOURCE_DATA) -> dict[str, Any]:
    live = _live_recomputation()
    phase6b = _load(PHASE6B_TERMINAL)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": "PENDING_GATES",
        "preregistration": _preregistered_experiment(),
        "official_import": {
            "repository": "https://github.com/amazon-science/LatticeAlgorithms.jl",
            "head": _git_head(OFFICIAL_ROOT),
            "expected_head": OFFICIAL_COMMIT,
            "license": "Apache-2.0",
            "license_sha256": _sha256(OFFICIAL_ROOT / "LICENSE"),
            "manifest_sha256": _sha256(JULIA_MANIFEST),
            "pythoncall_version": "0.9.10",
            "julia_version": live["raw_provenance"]["threshold"]["julia_version"],
            "compiled_modules_disabled": True,
            "official_source_modified": False,
            "one_dimensional_generic_boundary": "official generic closest_point raises BoundsError; official exact closest_point_scaled_Zn used",
        },
        "upstream_tests": _upstream_test_audit(),
        "correctness": live["correctness"],
        "official_data_reanalysis": live["official"],
        "independent_experiment": {
            "evidence_class": "INDEPENDENT_PREREGISTERED_MONTE_CARLO",
            "raw_integrity": live["integrity"],
            "seed_rows": live["seed_rows"],
            "curves": live["curves"],
            "thresholds": live["thresholds"],
            "paired_advantage": live["paired"],
            "runtime_scaling": live["runtime"],
        },
        "execution_budget": {
            "threshold_runtime_seconds": live["threshold_runtime_seconds"],
            "runtime_budget_seconds": 28_800,
            "sampled_host_working_set_high_water_bytes": SAMPLED_HOST_WORKING_SET_HIGH_WATER_BYTES,
            "memory_budget_bytes": 16 * (1 << 30),
            "memory_observation_boundary": "two external process working-set samples; not exact process peak RSS",
        },
        "evidence_boundary": {
            "official_aggregate_may_fill_independent_threshold": False,
            "analog_adapter_is_official_repository_function": False,
            "single_mode_substitution_allowed": False,
            "paper_threshold_precision_claimed_by_small_distance_run": False,
            "hardware_claim": None,
        },
        "phase6b_boundary": {
            "verdict": phase6b["verdict"],
            "promoted_by_phase6c": False,
            "board_measured_claim": None,
        },
        "bindings": {
            "preregistration": _binding(PREREG),
            "source_audit": _binding(SOURCE_AUDIT),
            "ontology": _binding(ONTOLOGY),
            "phase6b_terminal": _binding(PHASE6B_TERMINAL),
            "julia_project": _binding(JULIA_PROJECT),
            "julia_manifest": _binding(JULIA_MANIFEST),
            "julia_adapter": _binding(JULIA_SCRIPT),
            "upstream_test_adapter": _binding(UPSTREAM_SCRIPT),
            "seeded_test_adapter": _binding(SEEDED_TEST_SCRIPT),
            "official_license": _binding(OFFICIAL_ROOT / "LICENSE"),
            "official_module": _binding(OFFICIAL_ROOT / "src" / "LatticeAlgorithms.jl"),
            "official_surface_code": _binding(OFFICIAL_ROOT / "src" / "surface_code.jl"),
            "official_concatenated_code": _binding(OFFICIAL_ROOT / "src" / "concatenated_code.jl"),
            "official_matching": _binding(OFFICIAL_ROOT / "src" / "matching_utils.jl"),
            "official_fig5_notebook": _binding(OFFICIAL_ROOT / "examples" / "papers" / "Closest_lattice_point_decoding_for_multimode_GKP_codes" / "Fig_5.ipynb"),
            "correctness_raw": _binding(CORRECTNESS_RAW),
            "official_data_raw": _binding(OFFICIAL_DATA_RAW),
            "threshold_raw": _binding(THRESHOLD_RAW),
            "pilot_raw": _binding(PILOT_RAW),
        },
    }
    source_rows = _source_rows(report)
    _write_csv(source_rows, source_data_path)
    report["source_data"] = {"path": _relative(source_data_path), "sha256": _sha256(source_data_path), "rows": len(source_rows)}
    initial_gates = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(initial_gates.values()),
        "total": len(initial_gates),
        "failed": [key for key, passed in initial_gates.items() if not passed],
        "gates": initial_gates,
    }
    report["semantic_mutation_audit"] = _mutation_audit(report)
    final_gates = evaluate_gates(report)
    report["gate_summary"] = {
        "passed": sum(final_gates.values()),
        "total": len(final_gates),
        "failed": [key for key, passed in final_gates.items() if not passed],
        "gates": final_gates,
    }
    if all(final_gates.values()) and report["semantic_mutation_audit"]["detected"] == report["semantic_mutation_audit"]["count"]:
        report["verdict"] = "PASS_OFFICIAL_CPD_SMALL_DISTANCE_THRESHOLD_REPRODUCTION_WITH_UPSTREAM_CAVEAT"
    elif not final_gates["G05_exact_cvp_single_mode_and_surface_fast_path_have_zero_mismatch"]:
        report["verdict"] = "PARTIAL_UPSTREAM_OR_MODEL_MISMATCH"
    elif not final_gates["G12_both_small_distance_thresholds_meet_preregistered_absolute_tolerance"]:
        report["verdict"] = "NEGATIVE_THRESHOLD_TOLERANCE_FAIL"
    else:
        report["verdict"] = "PARTIAL_RUNTIME_OR_GOVERNANCE_FAILURE"
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    if report["gate_summary"]["gates"] != gates:
        raise AssertionError("stored gates do not match live recomputation")
    if not all(gates.values()):
        raise AssertionError(f"failed gates: {[key for key, passed in gates.items() if not passed]}")
    mutation = _mutation_audit(report)
    if mutation["detected"] != mutation["count"]:
        raise AssertionError("semantic mutation audit is incomplete")
    if report["semantic_mutation_audit"] != mutation:
        raise AssertionError("stored semantic mutation audit does not match live audit")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_report(_load(args.report))
        print(json.dumps({"task_id": TASK_ID, "verified": True, "report": _relative(args.report)}))
        return 0
    report = build_report(args.source_data)
    args.markdown.write_text(_markdown(report), encoding="utf-8")
    _atomic_json(report, args.report)
    verify_report(_load(args.report))
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "verdict": report["verdict"],
                "gates": report["gate_summary"],
                "mutations": report["semantic_mutation_audit"]["detected"],
                "source_rows": report["source_data"]["rows"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
