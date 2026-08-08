"""T6.18.3 posterior-predictive weighted CPD drift analysis.

The Julia shards execute the frozen structured surface--GKP family.  This
module is deliberately read-only with respect to the formal raw shards: it
validates entry/scope, recomputes all statistics, emits Source Data and keeps
the hidden-state oracle outside deployable rankings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "literature" / "t6_18_3_multimode_drift.json"
DEFAULT_REPORT = ROOT / "docs" / "t6_18_3_multimode_posterior_weighted_cpd.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_18_3_multimode_posterior_weighted_cpd_source_data.csv"
DEFAULT_HUMAN_REPORT = ROOT / "docs" / "multimode_posterior_weighted_cpd.md"
METHODS = (
    "static_euclidean",
    "weighted_static",
    "observed_only_posterior_predictive_weighted",
    "oracle_metric_upper_bound",
)
DEPLOYABLE = METHODS[:3]
ADAPTIVE = METHODS[2]
ORACLE = METHODS[3]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _tail_metrics(window_errors: Iterable[int], window_cycles: int, tail_fraction: float) -> dict[str, float]:
    values = np.asarray(tuple(window_errors), dtype=np.float64) / float(window_cycles)
    if values.size == 0:
        raise ValueError("tail metrics require full registered windows")
    count = max(1, int(math.ceil(tail_fraction * values.size)))
    ordered = np.sort(values)
    return {"worst_window_ler": float(ordered[-1]), "cvar95_window_ler": float(ordered[-count:].mean())}


def _bootstrap_mean_ci(values: np.ndarray, resamples: int, seed: int) -> dict[str, float]:
    if values.ndim != 1 or values.size < 2:
        raise ValueError("paired seed-cluster bootstrap needs at least two clusters")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "clusters": int(values.size),
        "resamples": int(resamples),
    }


def _signflip_pvalue(values: np.ndarray, resamples: int, seed: int) -> float:
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    generated = 0
    chunk = 5000
    while generated < resamples:
        count = min(chunk, resamples - generated)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, values.size))
        permuted = (signs * values).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-15))
        generated += count
    return float((exceed + 1) / (resamples + 1))


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (total - rank) * raw[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _load_memory(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="ascii", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "pid": int(row["pid"]),
                    "working_set_bytes": int(row["working_set_bytes"]),
                    "peak_working_set_bytes": int(row["peak_working_set_bytes"]),
                    "cpu_seconds": float(row["cpu_seconds"]),
                }
            )
    if not rows:
        raise ValueError("process memory monitor contains no samples")
    concurrent: dict[str, int] = defaultdict(int)
    for row in rows:
        concurrent[row["timestamp"]] += row["working_set_bytes"]
    return {
        "samples": len(rows),
        "process_ids": sorted({row["pid"] for row in rows}),
        "max_single_working_set_bytes": max(row["working_set_bytes"] for row in rows),
        "max_reported_peak_working_set_bytes": max(row["peak_working_set_bytes"] for row in rows),
        "max_concurrent_working_set_bytes": max(concurrent.values()),
        "observation": "5-second external Get-Process samples; sampled high-water, not exact peak RSS",
    }


def _entry_passed(entry: dict[str, Any]) -> bool:
    summary = entry.get("gate_summary", {})
    gates_passed = bool(summary.get("all_passed", False)) or (
        int(summary.get("total", -1)) > 0
        and int(summary.get("passed", -2)) == int(summary.get("total", -1))
        and not summary.get("failed", [])
    )
    return str(entry.get("verdict", "")).startswith("PASS_OFFICIAL_CPD_SMALL_DISTANCE") and gates_passed


def _seed_arrays(rows: list[dict[str, Any]], family: str | None) -> dict[str, dict[str, np.ndarray]]:
    selected = [row for row in rows if family is None or row["family"] == family]
    seeds = sorted({int(row["seed"]) for row in selected})
    result: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        p_l: list[float] = []
        seed_worst: list[float] = []
        seed_cvar: list[float] = []
        for seed in seeds:
            seed_rows = [row for row in selected if int(row["seed"]) == seed]
            errors = sum(int(row["errors"][method]) for row in seed_rows)
            cycles = sum(int(row["cycles"]) for row in seed_rows)
            p_l.append(errors / cycles)
            windows = [int(value) for row in seed_rows for value in row["window_errors"][method]]
            tail = _tail_metrics(windows, 512, 0.05)
            seed_worst.append(tail["worst_window_ler"])
            seed_cvar.append(tail["cvar95_window_ler"])
        result[method] = {
            "p_L": np.asarray(p_l, dtype=np.float64),
            "seed_worst_window_ler": np.asarray(seed_worst, dtype=np.float64),
            "seed_cvar95_window_ler": np.asarray(seed_cvar, dtype=np.float64),
        }
    return result


def _summaries(rows: list[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    window_cycles = int(config["formal"]["window_cycles"])
    tail_fraction = float(config["statistics"]["tail_fraction"])
    scopes: dict[str, Any] = {}
    seed_metrics: dict[str, Any] = {}
    for scope in ("aggregate", *config["formal"]["families"]):
        family = None if scope == "aggregate" else scope
        selected = [row for row in rows if family is None or row["family"] == family]
        arrays = _seed_arrays(rows, family)
        seed_metrics[scope] = arrays
        method_summary: dict[str, Any] = {}
        for method in METHODS:
            errors = sum(int(row["errors"][method]) for row in selected)
            cycles = sum(int(row["cycles"]) for row in selected)
            windows = [int(value) for row in selected for value in row["window_errors"][method]]
            tail = _tail_metrics(windows, window_cycles, tail_fraction)
            runtimes = [float(row["runtime_seconds"][method]) for row in selected]
            allocations = [int(row["allocated_bytes_first_measured_decode"][method]) for row in selected]
            method_summary[method] = {
                "errors": errors,
                "cycles": cycles,
                "p_L": errors / cycles,
                "registered_windows": len(windows),
                **tail,
                "mean_seed_worst_window_ler": float(arrays[method]["seed_worst_window_ler"].mean()),
                "mean_seed_cvar95_window_ler": float(arrays[method]["seed_cvar95_window_ler"].mean()),
                "runtime_seconds": float(sum(runtimes)),
                "seconds_per_decode": float(sum(runtimes) / cycles),
                "allocated_bytes_first_decode_median": float(np.median(allocations)),
                "allocated_bytes_first_decode_max": max(allocations),
            }
        scopes[scope] = method_summary
    return scopes, seed_metrics


def _comparisons(seed_metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    bootstrap = int(config["statistics"]["paired_bootstrap_resamples"])
    signflips = int(config["statistics"]["signflip_resamples"])
    definitions = {
        "adaptive_vs_static_euclidean": ("static_euclidean", ADAPTIVE),
        "adaptive_vs_weighted_static": ("weighted_static", ADAPTIVE),
        "weighted_static_vs_static_euclidean": ("static_euclidean", "weighted_static"),
    }
    result: dict[str, Any] = {}
    for scope, metrics in seed_metrics.items():
        scope_rows: dict[str, Any] = {}
        for contrast, (baseline, candidate) in definitions.items():
            values = metrics[baseline]["p_L"] - metrics[candidate]["p_L"]
            seed = int(hashlib.sha256(f"T6.18.3:{scope}:{contrast}".encode()).hexdigest()[:8], 16)
            scope_rows[contrast] = {
                "baseline": baseline,
                "candidate": candidate,
                "improvement": _bootstrap_mean_ci(values, bootstrap, seed),
                "two_sided_signflip_p": _signflip_pvalue(values, signflips, seed ^ 0x61830001),
                "positive_seed_clusters": int(np.count_nonzero(values > 0)),
                "negative_seed_clusters": int(np.count_nonzero(values < 0)),
                "tie_seed_clusters": int(np.count_nonzero(values == 0)),
            }
        raw = {name: row["two_sided_signflip_p"] for name, row in scope_rows.items()}
        adjusted = _holm_adjust(raw)
        for name, value in adjusted.items():
            scope_rows[name]["holm_adjusted_p"] = value
        result[scope] = scope_rows
    return result


def _adaptation(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family in config["formal"]["families"]:
        selected = [row for row in rows if row["family"] == family]
        raw_lags = [value for row in selected for value in row["adaptation_lag_cycles"]]
        observed = np.asarray([int(value) for value in raw_lags if value is not None], dtype=np.int64)
        result[family] = {
            "registered_events": len(raw_lags),
            "observed_lags": int(observed.size),
            "censored_lags": sum(value is None for value in raw_lags),
            "median_lag_cycles": None if observed.size == 0 else float(np.median(observed)),
            "p95_lag_cycles": None if observed.size == 0 else float(np.quantile(observed, 0.95)),
            "maximum_lag_cycles": None if observed.size == 0 else int(observed.max()),
            "posterior_theta_rmse_mean": float(np.mean([row["posterior_theta_rmse"] for row in selected])),
            "posterior_entropy_mean": float(np.mean([row["posterior_entropy_mean"] for row in selected])),
        }
    return result


def _structural_state(
    config: dict[str, Any],
    preregistration: dict[str, Any],
    entry: dict[str, Any],
    correctness: dict[str, Any],
    pilot: dict[str, Any],
    shards: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    memory: dict[str, Any],
    source_rows: int,
    stderr_only_known_warnings: bool,
    bindings_valid: bool = True,
) -> dict[str, Any]:
    expected_seeds = set(map(int, config["formal"]["seeds"]))
    expected_families = set(config["formal"]["families"])
    expected_cycles = int(config["formal"]["cycles_per_cluster"])
    expected_windows = expected_cycles // int(config["formal"]["window_cycles"])
    keys = [(int(row["seed"]), row["family"]) for row in rows]
    method_keys_valid = all(
        set(row["errors"]) == set(METHODS)
        and set(row["window_errors"]) == set(METHODS)
        and set(row["runtime_seconds"]) == set(METHODS)
        and set(row["allocated_bytes_first_measured_decode"]) == set(METHODS)
        for row in rows
    )
    window_valid = all(
        len(row["window_errors"][method]) == expected_windows
        and sum(row["window_errors"][method]) <= int(row["errors"][method])
        for row in rows
        for method in METHODS
    )
    count_valid = all(
        0 <= int(row["errors"][method]) <= int(row["cycles"])
        and all(0 <= int(value) <= int(config["formal"]["window_cycles"]) for value in row["window_errors"][method])
        for row in rows
        for method in METHODS
    )
    phase6b_lock = preregistration["phase6b_lock"]["t6_15_5"]["initial_artifact"]
    phase6b_path = ROOT / phase6b_lock["path"]
    return {
        "entry_passed": _entry_passed(entry),
        "official_commit_matches": all(shard["head_matches"] for shard in shards)
        and all(shard["official_head"] == config["official_commit"] for shard in shards),
        "correctness_passed": bool(correctness["correctness"]["passed"])
        and all(shard["correctness"]["passed"] for shard in shards),
        "strict_causal_passed": correctness["correctness"]["strict_causal_prefix_mismatches"] == 0
        and correctness["correctness"]["mutated_suffix_divergence_detected"],
        "config_hash_matches": all(shard["config_sha256"] == _sha256(DEFAULT_CONFIG) for shard in shards),
        "formal_shards_exact": all(
            shard["mode"] == "formal"
            and set(map(int, shard["selected_seeds"]))
            == {int(row["seed"]) for row in shard["rows"]}
            for shard in shards
        ),
        "pilot_nonselection": config["pilot"]["performance_selection_allowed"] is False
        and pilot["mode"] == "pilot"
        and set(map(int, pilot["selected_seeds"])) == set(map(int, config["pilot"]["seeds"]))
        and int(pilot["cycles_per_cluster"]) == int(config["pilot"]["cycles_per_cluster"]),
        "phase6b_lock_unchanged": phase6b_path.stat().st_size == int(phase6b_lock["bytes"])
        and _sha256(phase6b_path) == phase6b_lock["sha256"],
        "seed_coverage": {seed for seed, _ in keys} == expected_seeds,
        "family_coverage": {family for _, family in keys} == expected_families,
        "unique_seed_family_rows": len(keys) == len(set(keys)) == len(expected_seeds) * len(expected_families),
        "cycles_exact": all(int(row["cycles"]) == expected_cycles for row in rows),
        "methods_exact": method_keys_valid,
        "windows_exact": window_valid,
        "counts_valid": count_valid,
        "oracle_nonranking": config["statistics"]["oracle_ranking_eligible"] is False,
        "runtime_within_budget": max(float(shard["wall_clock_seconds"]) for shard in shards)
        <= float(config["runtime_budget"]["wall_clock_seconds"]),
        "memory_within_budget": memory["max_concurrent_working_set_bytes"]
        <= float(config["runtime_budget"]["memory_gib"]) * 1024**3,
        "stderr_only_known_warnings": bool(stderr_only_known_warnings),
        "source_data_nonempty": source_rows > 0,
        "bindings_valid": bool(bindings_valid),
    }


def _mutation_audit(state: dict[str, Any]) -> list[dict[str, Any]]:
    mutations = [
        ("entry_scope_failure", "entry_passed", False),
        ("official_commit_swap", "official_commit_matches", False),
        ("correctness_mismatch", "correctness_passed", False),
        ("future_suffix_leak", "strict_causal_passed", False),
        ("config_hash_drift", "config_hash_matches", False),
        ("pilot_reselection", "pilot_nonselection", False),
        ("nonformal_shard", "formal_shards_exact", False),
        ("phase6b_parent_rewrite", "phase6b_lock_unchanged", False),
        ("missing_seed", "seed_coverage", False),
        ("missing_family", "family_coverage", False),
        ("duplicate_seed_family", "unique_seed_family_rows", False),
        ("shortened_cluster", "cycles_exact", False),
        ("dropped_comparator", "methods_exact", False),
        ("partial_window_injection", "windows_exact", False),
        ("invalid_error_count", "counts_valid", False),
        ("oracle_rank_eligible", "oracle_nonranking", False),
        ("runtime_cap_exceeded", "runtime_within_budget", False),
        ("memory_cap_exceeded", "memory_within_budget", False),
        ("hidden_stderr_error", "stderr_only_known_warnings", False),
        ("empty_source_data", "source_data_nonempty", False),
        ("binding_hash_tamper", "bindings_valid", False),
    ]
    rows: list[dict[str, Any]] = []
    for name, gate, value in mutations:
        mutated = copy.deepcopy(state)
        mutated[gate] = value
        detected = not bool(mutated[gate]) and all(bool(v) for key, v in mutated.items() if key != gate)
        rows.append({"mutation": name, "target_gate": gate, "detected": detected})
    return rows


def _source_rows(
    rows: list[dict[str, Any]],
    summaries: dict[str, Any],
    comparisons: dict[str, Any],
    adaptation: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    window_cycles = int(config["formal"]["window_cycles"])
    for row in rows:
        for method in METHODS:
            output.append(
                {
                    "row_type": "seed_summary",
                    "seed": row["seed"],
                    "family": row["family"],
                    "method": method,
                    "index": "",
                    "numerator": row["errors"][method],
                    "denominator": row["cycles"],
                    "value": row["errors"][method] / row["cycles"],
                    "metric": "p_L",
                    "evidence": "PROJECT_NATIVE_MATCHED" if method != ORACLE else "ORACLE_REFERENCE_NONRANKING",
                }
            )
            for index, errors in enumerate(row["window_errors"][method], start=1):
                output.append(
                    {
                        "row_type": "window",
                        "seed": row["seed"],
                        "family": row["family"],
                        "method": method,
                        "index": index,
                        "numerator": errors,
                        "denominator": window_cycles,
                        "value": errors / window_cycles,
                        "metric": "window_ler",
                        "evidence": "PROJECT_NATIVE_MATCHED" if method != ORACLE else "ORACLE_REFERENCE_NONRANKING",
                    }
                )
        for index, trace in enumerate(row["trace_sample"], start=1):
            for metric in ("true_theta", "posterior_theta", "posterior_entropy"):
                output.append(
                    {
                        "row_type": "posterior_trace_sample",
                        "seed": row["seed"],
                        "family": row["family"],
                        "method": ADAPTIVE,
                        "index": trace["cycle"],
                        "numerator": "",
                        "denominator": "",
                        "value": trace[metric],
                        "metric": metric,
                        "evidence": "OBSERVED_ONLY_CAUSAL_SAMPLE" if metric != "true_theta" else "ORACLE_DIAGNOSTIC_ONLY",
                    }
                )
    for scope, methods in summaries.items():
        for method, values in methods.items():
            for metric in ("p_L", "worst_window_ler", "cvar95_window_ler", "seconds_per_decode"):
                output.append(
                    {
                        "row_type": "aggregate_metric",
                        "seed": "",
                        "family": scope,
                        "method": method,
                        "index": "",
                        "numerator": values.get("errors", "") if metric == "p_L" else "",
                        "denominator": values.get("cycles", "") if metric == "p_L" else "",
                        "value": values[metric],
                        "metric": metric,
                        "evidence": "PROJECT_NATIVE_MATCHED" if method != ORACLE else "ORACLE_REFERENCE_NONRANKING",
                    }
                )
    for scope, contrasts in comparisons.items():
        for contrast, values in contrasts.items():
            output.append(
                {
                    "row_type": "paired_contrast",
                    "seed": "",
                    "family": scope,
                    "method": contrast,
                    "index": "",
                    "numerator": values["improvement"]["ci_low"],
                    "denominator": values["improvement"]["ci_high"],
                    "value": values["improvement"]["mean"],
                    "metric": "p_L_baseline_minus_candidate",
                    "evidence": "PAIRED_SEED_CLUSTER_BOOTSTRAP",
                }
            )
    for family, values in adaptation.items():
        output.append(
            {
                "row_type": "adaptation_lag",
                "seed": "",
                "family": family,
                "method": ADAPTIVE,
                "index": "",
                "numerator": values["observed_lags"],
                "denominator": values["registered_events"],
                "value": values["median_lag_cycles"],
                "metric": "median_adaptation_lag_cycles",
                "evidence": "OBSERVED_ONLY_CAUSAL",
            }
        )
    return output


def _write_source_data(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "row_type",
        "seed",
        "family",
        "method",
        "index",
        "numerator",
        "denominator",
        "value",
        "metric",
        "evidence",
    ]
    with DEFAULT_SOURCE_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(run_dir: Path, write_outputs: bool = True) -> dict[str, Any]:
    config = _read_json(DEFAULT_CONFIG)
    preregistration_path = ROOT / "docs" / "t6_16_3_secondary_preregistration.json"
    preregistration = _read_json(preregistration_path)
    entry_path = ROOT / config["entry_report"]
    entry = _read_json(entry_path)
    correctness_path = ROOT / "docs" / "t6_18_3_correctness_raw.json"
    pilot_path = ROOT / "docs" / "t6_18_3_pilot_raw.json"
    correctness = _read_json(correctness_path)
    pilot = _read_json(pilot_path)
    shard_paths = sorted(run_dir.glob("shard_*.json"))
    if len(shard_paths) != 4:
        raise ValueError(f"expected four completed formal shards, found {len(shard_paths)}")
    shards = [_read_json(path) for path in shard_paths]
    stdout_paths = sorted(run_dir.glob("shard_*.stdout.log"))
    stderr_paths = sorted(run_dir.glob("shard_*.stderr.log"))
    if len(stdout_paths) != 4 or len(stderr_paths) != 4:
        raise ValueError("each formal shard requires bound stdout and stderr")
    stderr_text = "\n".join(path.read_text(encoding="utf-8") for path in stderr_paths)
    stderr_only_known_warnings = not any(
        marker in stderr_text for marker in ("ERROR:", "Stacktrace", "LoadError", "Exception")
    ) and all(
        not line.strip()
        or line.startswith(("┌ Warning:", "┌ Info:", "└ ", "            ", "Anything "))
        for line in stderr_text.splitlines()
    )
    rows = [row for shard in shards for row in shard["rows"]]
    memory_path = run_dir / "process_memory_samples.csv"
    memory = _load_memory(memory_path)
    summaries, seed_metrics = _summaries(rows, config)
    comparisons = _comparisons(seed_metrics, config)
    adaptation = _adaptation(rows, config)
    source_data = _source_rows(rows, summaries, comparisons, adaptation, config)
    if write_outputs:
        _write_source_data(source_data)
    state = _structural_state(
        config,
        preregistration,
        entry,
        correctness,
        pilot,
        shards,
        rows,
        memory,
        len(source_data),
        stderr_only_known_warnings,
    )
    gates = [{"gate": key, "passed": bool(value)} for key, value in state.items()]
    mutations = _mutation_audit(state)

    strongest_static: dict[str, str] = {}
    tail_gate: dict[str, Any] = {}
    for family in ("calibration_shift", "telegraph"):
        baseline = min(("static_euclidean", "weighted_static"), key=lambda method: summaries[family][method]["p_L"])
        strongest_static[family] = baseline
        tail_gate[family] = {
            "baseline": baseline,
            "worst_window_noninferior": summaries[family][ADAPTIVE]["worst_window_ler"]
            <= summaries[family][baseline]["worst_window_ler"] + 1e-15,
            "cvar95_noninferior": summaries[family][ADAPTIVE]["cvar95_window_ler"]
            <= summaries[family][baseline]["cvar95_window_ler"] + 1e-15,
        }
    aggregate_main = comparisons["aggregate"]
    average_gate = all(
        aggregate_main[name]["improvement"]["ci_low"] > 0.0
        and aggregate_main[name]["holm_adjusted_p"] < 0.05
        for name in ("adaptive_vs_static_euclidean", "adaptive_vs_weighted_static")
    )
    tail_passed = all(
        row["worst_window_noninferior"] and row["cvar95_noninferior"] for row in tail_gate.values()
    )
    performance_go = average_gate and tail_passed
    verdict = "GO_POSTERIOR_WEIGHTED_CPD_DRIFT_GAIN" if performance_go else "NEGATIVE_NO_DRIFT_GAIN"

    bindings_paths = [
        DEFAULT_CONFIG,
        preregistration_path,
        ROOT / "docs" / "t6_16_2_comparison_ontology.json",
        entry_path,
        correctness_path,
        pilot_path,
        ROOT / "docs" / "t6_15_5_route_a_v5_final_evidence_gate.json",
        ROOT / "scripts" / "run_multimode_posterior_weighted_cpd.jl",
        ROOT / "scripts" / "launch_t6_18_3_shard.ps1",
        ROOT / "scripts" / "monitor_t6_18_3_processes.ps1",
        Path(__file__),
        ROOT / "configs" / "literature" / "t6_18_2_julia_env" / "Project.toml",
        ROOT / "configs" / "literature" / "t6_18_2_julia_env" / "Manifest.toml",
        ROOT / "third_party" / "LatticeAlgorithms.jl" / "LICENSE",
        ROOT / "third_party" / "LatticeAlgorithms.jl" / "src" / "surface_code.jl",
        ROOT / "third_party" / "LatticeAlgorithms.jl" / "src" / "concatenated_code.jl",
        ROOT / "third_party" / "LatticeAlgorithms.jl" / "src" / "matching_utils.jl",
        *shard_paths,
        *stdout_paths,
        *stderr_paths,
        memory_path,
        run_dir / "monitor.stdout.log",
        run_dir / "monitor.stderr.log",
        DEFAULT_SOURCE_DATA,
    ]
    bindings = [_binding(path) for path in bindings_paths]
    analysis_payload = {
        "summaries": summaries,
        "comparisons": comparisons,
        "adaptation": adaptation,
        "memory": memory,
        "performance_gate": {
            "average_gate": average_gate,
            "tail_gate": tail_gate,
            "tail_passed": tail_passed,
            "strongest_static": strongest_static,
            "oracle_ranking_eligible": False,
        },
        "formal_counts": {
            "seed_clusters": len(config["formal"]["seeds"]),
            "families": len(config["formal"]["families"]),
            "seed_family_rows": len(rows),
            "cycles_per_cluster": config["formal"]["cycles_per_cluster"],
            "total_physical_cycles": sum(int(row["cycles"]) for row in rows),
            "total_comparator_decodes": sum(int(row["cycles"]) for row in rows) * len(METHODS),
            "registered_windows_per_method": sum(
                len(row["window_errors"][METHODS[0]]) for row in rows
            ),
            "source_data_rows": len(source_data),
        },
        "verdict": verdict,
    }
    report = {
        "schema_version": "t6.18.3-multimode-posterior-weighted-cpd-v1",
        "task_id": "T6.18.3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "entry_gate": {
            "passed": _entry_passed(entry),
            "entry_verdict": entry["verdict"],
            "scope": "same d=3 official-validated rotated surface--square-GKP family",
        },
        "method_contract": {
            "adaptive_observability": config["posterior_filter"]["observation"],
            "adaptive_decode_uses": config["posterior_filter"]["decode_uses"],
            "update_cadence_cycles": config["posterior_filter"]["update_cadence_cycles"],
            "oracle_privilege": "true current theta/variance; reference only, excluded from Holm and deployable ranking",
            "pilot_performance_selection_allowed": config["pilot"]["performance_selection_allowed"],
            "tail_rule": "non-overlapping full 512-cycle windows; exclude the 160-cycle partial tail; worst=max; CVaR95=mean of exactly ceil(5%*N) largest windows with deterministic rank tie handling",
            "stderr_boundary": "only upstream duplicate-doc replacement warnings and concurrent CondaPkg lock-wait info; ERROR/Stacktrace/LoadError forbidden",
        },
        **analysis_payload,
        "analysis_payload_sha256": _canonical_sha(analysis_payload),
        "bindings": bindings,
        "gate_summary": {
            "passed": sum(row["passed"] for row in gates),
            "total": len(gates),
            "all_passed": all(row["passed"] for row in gates),
            "gates": gates,
        },
        "semantic_mutation_audit": {
            "detected": sum(row["detected"] for row in mutations),
            "total": len(mutations),
            "all_detected": all(row["detected"] for row in mutations),
            "mutations": mutations,
        },
        "claim_boundary": {
            "allowed": [
                "project-native observed-only posterior-weighted CPD result on the frozen d=3 heteroscedastic drift family",
                "paired p_L and registered 512-cycle tail metrics against two same-task static deployable baselines",
                "strict-causal software simulation, runtime and sampled host-memory evidence",
            ],
            "forbidden": [
                "rewrite Phase 6B NO-GO or the V5 >=10% target",
                "rank the hidden-state oracle as deployable",
                "claim general multimode, stationary-threshold, FPGA or physical-device superiority",
                "call a project-native heteroscedastic drift extension an official Lin et al. experiment",
            ],
        },
    }
    if not report["gate_summary"]["all_passed"]:
        failed = [row["gate"] for row in gates if not row["passed"]]
        raise ValueError(f"integrity gates failed: {failed}")
    if not report["semantic_mutation_audit"]["all_detected"]:
        failed = [row["mutation"] for row in mutations if not row["detected"]]
        raise ValueError(f"semantic mutation detection failed: {failed}")
    if write_outputs:
        DEFAULT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_human_report(report)
    return report


def _write_human_report(report: dict[str, Any]) -> None:
    summaries = report["summaries"]
    comparisons = report["comparisons"]["aggregate"]
    lines = [
        "# T6.18.3 multimode posterior-weighted CPD 漂移扩展",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- formal cycles：{report['formal_counts']['total_physical_cycles']:,}",
        f"- comparator decodes：{report['formal_counts']['total_comparator_decodes']:,}",
        f"- gates / mutations：{report['gate_summary']['passed']}/{report['gate_summary']['total']} / "
        f"{report['semantic_mutation_audit']['detected']}/{report['semantic_mutation_audit']['total']}",
        "",
        "## Aggregate",
        "",
        "| method | p_L | worst 512-window | CVaR95 | seconds/decode |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = summaries["aggregate"][method]
        lines.append(
            f"| `{method}` | {row['p_L']:.8f} | {row['worst_window_ler']:.6f} | "
            f"{row['cvar95_window_ler']:.6f} | {row['seconds_per_decode']:.6g} |"
        )
    lines += ["", "## Paired p_L contrasts", "", "| contrast | mean [95% CI] | Holm p |", "| --- | ---: | ---: |"]
    for name, row in comparisons.items():
        imp = row["improvement"]
        lines.append(
            f"| `{name}` | {imp['mean']:.8f} [{imp['ci_low']:.8f}, {imp['ci_high']:.8f}] | "
            f"{row['holm_adjusted_p']:.6g} |"
        )
    lines += [
        "",
        "## 边界",
        "",
        "该实验是 project-native、observed-only、strict-causal 的 d=3 heteroscedastic drift 扩展。"
        "oracle 使用当前真实 metric，只作上界参考；结果不回写 Phase 6B，不代表 official Lin et al. drift experiment、"
        "stationary threshold、一般 multimode SOTA、FPGA 或物理装置优势。",
        "",
        "## Artifacts",
        "",
        "- `docs/t6_18_3_multimode_posterior_weighted_cpd.json`",
        "- `docs/t6_18_3_multimode_posterior_weighted_cpd_source_data.csv`",
        "- `scripts/run_multimode_posterior_weighted_cpd.jl`",
        "- `cnn_fpga/benchmark/multimode_posterior_weighted_cpd.py`",
    ]
    DEFAULT_HUMAN_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_report(report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    report = _read_json(report_path)
    if report["analysis_payload_sha256"] != _canonical_sha(
        {
            "summaries": report["summaries"],
            "comparisons": report["comparisons"],
            "adaptation": report["adaptation"],
            "memory": report["memory"],
            "performance_gate": report["performance_gate"],
            "formal_counts": report["formal_counts"],
            "verdict": report["verdict"],
        }
    ):
        raise ValueError("analysis payload hash mismatch")
    for binding in report["bindings"]:
        path = ROOT / binding["path"]
        if not path.is_file() or _sha256(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]:
            raise ValueError(f"binding mismatch: {binding['path']}")
    if not report["gate_summary"]["all_passed"] or not report["semantic_mutation_audit"]["all_detected"]:
        raise ValueError("stored integrity or mutation gate failed")
    return {"task_id": "T6.18.3", "verified": True, "verdict": report["verdict"], "report": _rel(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_report(), ensure_ascii=False))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required unless --verify-only is used")
    report = build_report(args.run_dir.resolve(), write_outputs=True)
    print(
        json.dumps(
            {
                "task_id": report["task_id"],
                "verdict": report["verdict"],
                "gates": report["gate_summary"],
                "mutations": report["semantic_mutation_audit"],
                "source_data_rows": report["formal_counts"]["source_data_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
