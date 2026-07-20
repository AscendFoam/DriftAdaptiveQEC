"""T6.20.4 development-only causal headroom and regret audit.

The Julia runner owns trace generation and decoding.  This module never
simulates missing outcomes: it validates the raw ledger, independently
recomputes every aggregate, performs the paired seed-cluster bootstrap and
emits the fail-closed GO/NO-GO evidence bundle.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "phase6d" / "t6_20_4_causal_headroom.json"
RAW = ROOT / "runs" / "t6_20_4_causal_headroom_raw.json"
REPORT = ROOT / "docs" / "t6_20_4_multimode_causal_headroom.json"
SOURCE_DATA = ROOT / "docs" / "t6_20_4_multimode_causal_headroom_source_data.csv"
MARKDOWN = ROOT / "docs" / "multimode_causal_headroom.md"
RUNNER = ROOT / "scripts" / "run_t6_20_4_causal_headroom.jl"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


class IntegrityError(RuntimeError):
    """Raised when a development raw ledger violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def _expected_cells(manifest: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    split = config["source_split"]
    distance = int(config["distance"])
    return [
        row
        for row in manifest["execution_cells"]
        if row["split_id"] == split and int(row["distance"]) == distance
    ]


def _validate_decoder_source() -> dict[str, Any]:
    source = RUNNER.read_text(encoding="utf-8")
    signatures = {}
    decision_functions = (
        "decode_cpd",
        "periodic_mwpm_weights",
        "exact_state_scores",
        "mixture_action",
        "plugin_action",
        "robust_action",
        "causal_probe",
    )
    for name in decision_functions:
        match = re.search(rf"^function\s+{name}\(([^\n]*)", source, flags=re.MULTILINE)
        _require(match is not None, f"missing decision function {name}")
        signature = match.group(1)
        signatures[name] = signature
        lowered = signature.lower()
        for forbidden in (
            "true_theta",
            "scenario_family",
            "formal_label",
            "future_suffix",
            "pattern",
            "variance_law",
            "loadings",
        ):
            _require(forbidden not in lowered, f"{name} signature exposes {forbidden}")

    _require(
        "q_scores0, q_scores1, fast.posterior" in source
        and "p_scores0, p_scores1, fast.posterior" in source,
        "posterior-predictive action is not bound to the predecision posterior",
    )
    _require(
        source.index("q_pp, q_confidence = mixture_action")
        < source.index("update_filter!(fast, fast_observation)"),
        "posterior update occurs before current-round decoding",
    )
    _require(
        "minimum_tjoin(graph, syndrome_with_boundary)" in source,
        "long-run path does not use the qualified pure-Julia T-join",
    )
    _require("qloadings" not in source and "ploadings" not in source, "latent spatial loadings reach decision code")
    _require(
        source.count("spatial_loadings(") == 2,
        "spatial pattern must appear only in its generator and physical trace construction",
    )
    _require(
        "fast_sigmas, fast_means = circular_parameters(fast_moments)" in source
        and "predictive_precision(fast, fast_sigmas)" in source,
        "modewise estimator is not observation-derived",
    )
    return {
        "decision_function_signatures": signatures,
        "predecision_update_order": "decode current round, then update for next round",
        "hidden_truth_signature_hits": 0,
        "spatial_metadata_decision_hits": 0,
    }


def validate_raw(
    raw: dict[str, Any],
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    check_file_hashes: bool = True,
) -> dict[str, Any]:
    _require(raw["task_id"] == "T6.20.4", "wrong raw task id")
    _require(raw["schema_version"] == "t6.20.4-causal-headroom-raw-v1", "wrong raw schema")
    _require(raw["source_split"] == "train", "non-train split accessed")
    _require(config["eligibility_boundary"]["spatial_pattern_or_variance_law_decoder_access"] is False, "spatial metadata privilege not denied")
    _require(raw["source_split"] in config["eligibility_boundary"]["allowed_split_ids"], "split not allowed")
    _require(
        raw["source_split"] not in config["eligibility_boundary"]["forbidden_split_ids"],
        "forbidden split accessed",
    )
    if check_file_hashes:
        _require(raw["config_sha256"] == _sha256(CONFIG), "raw config hash is stale")
        manifest_path = ROOT / raw["source_manifest_path"]
        _require(raw["source_manifest_sha256"] == _sha256(manifest_path), "raw source manifest hash is stale")
    _require(raw["correctness"]["passed"] is True, "exact/coset correctness failed")
    _require(raw["causality"]["passed"] is True, "causal prefix mutation failed")
    correctness = raw["correctness"]
    expected_correctness = config["correctness"]
    _require(correctness["official_bsv_samples"] == expected_correctness["official_bsv_samples"], "wrong BSV sample count")
    _require(correctness["official_bsv_action_mismatches"] == 0, "BSV decision mismatch")
    _require(correctness["pure_julia_vs_official_correction_mismatches"] == 0, "T-join mismatch")
    _require(correctness["alias_action_mismatches"] == 0, "alias truncation changes action")
    _require(
        correctness["maximum_official_log10_odds_error"] <= expected_correctness["maximum_log_odds_error"],
        "official log-odds tolerance exceeded",
    )
    _require(
        correctness["maximum_probability_normalization_error"]
        <= expected_correctness["maximum_probability_normalization_error"],
        "coset probability normalization failed",
    )
    _require(correctness["coset_cardinality_min"] == correctness["coset_cardinality_max"] == 16, "wrong d=3 coset cardinality")

    expected = _expected_cells(manifest, config)
    expected_by_id = {row["cell_id"]: row for row in expected}
    rows = raw["rows"]
    _require(raw["rounds_per_cell"] == config["rounds_per_cell"], "raw rounds differ from frozen config")
    _require(len(rows) == len(expected), "raw cell count is incomplete")
    _require(raw["selected_cell_count"] == len(expected), "selected cell count is inconsistent")
    _require(raw["selected_family_count"] == len(manifest["config_snapshot"]["scenario_families"]), "family count incomplete")
    ids = [row["cell_id"] for row in rows]
    _require(len(ids) == len(set(ids)), "duplicate raw cell")
    _require(set(ids) == set(expected_by_id), "raw cells do not exactly equal the d=3 train manifest")

    methods = list(config["methods"])
    per_seed_families: dict[int, set[str]] = defaultdict(set)
    trace_hashes: set[str] = set()
    for row in rows:
        expected_row = expected_by_id[row["cell_id"]]
        for key in (
            "seed",
            "scenario_family",
            "distance",
            "base_sigma",
            "variance_law_id",
            "spatial_pattern_sha256",
            "cell_sha256",
        ):
            raw_key = {"scenario_family": "family", "cell_sha256": "source_cell_sha256"}.get(key, key)
            _require(row[raw_key] == expected_row[key], f"cell metadata mismatch: {row['cell_id']}:{key}")
        _require(row["rounds"] == config["rounds_per_cell"], "row rounds mismatch")
        _require(set(row["errors"]) == set(methods), "method set mismatch")
        _require(set(row["x_only_errors"]) == set(methods), "X method set mismatch")
        _require(set(row["z_only_errors"]) == set(methods), "Z method set mismatch")
        _require(set(row["y_errors"]) == set(methods), "Y method set mismatch")
        for method in methods:
            values = [
                row["errors"][method],
                row["x_only_errors"][method],
                row["z_only_errors"][method],
                row["y_errors"][method],
            ]
            _require(all(isinstance(value, int) and 0 <= value <= row["rounds"] for value in values), "invalid error count")
            _require(sum(values[1:]) == values[0], "Pauli-class counts do not sum to pL count")
        _require(re.fullmatch(r"[0-9a-f]{64}", row["physical_trace_sha256"]) is not None, "invalid trace hash")
        _require(re.fullmatch(r"[0-9a-f]{64}", row["predecision_posterior_sha256"]) is not None, "invalid posterior hash")
        _require("formal" not in row and "pilot" not in row and "true_theta_trace" not in row, "forbidden label/latent trace in row")
        _require(row["physical_trace_sha256"] not in trace_hashes, "duplicate physical trace hash")
        trace_hashes.add(row["physical_trace_sha256"])
        per_seed_families[int(row["seed"])].add(row["family"])

    expected_families = set(manifest["config_snapshot"]["scenario_families"])
    _require(all(families == expected_families for families in per_seed_families.values()), "family deletion/reweighting detected")
    _require(len(per_seed_families) == raw["selected_seed_count"], "seed count mismatch")
    _require(len(per_seed_families) == 12, "d=3 train cluster count changed")
    return {
        "seed_count": len(per_seed_families),
        "family_count": len(expected_families),
        "cell_count": len(rows),
        "rounds": sum(int(row["rounds"]) for row in rows),
        "trace_hash_count": len(trace_hashes),
        "methods": methods,
    }


def _aggregate(raw: dict[str, Any], methods: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[int, Any]]:
    totals = {method: {"errors": 0, "rounds": 0, "x": 0, "y": 0, "z": 0} for method in methods}
    families: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {method: {"errors": 0, "rounds": 0} for method in methods}
    )
    clusters: dict[int, dict[str, dict[str, int]]] = defaultdict(
        lambda: {method: {"errors": 0, "rounds": 0} for method in methods}
    )
    for row in raw["rows"]:
        seed = int(row["seed"])
        family = row["family"]
        for method in methods:
            totals[method]["errors"] += int(row["errors"][method])
            totals[method]["rounds"] += int(row["rounds"])
            totals[method]["x"] += int(row["x_only_errors"][method])
            totals[method]["y"] += int(row["y_errors"][method])
            totals[method]["z"] += int(row["z_only_errors"][method])
            families[family][method]["errors"] += int(row["errors"][method])
            families[family][method]["rounds"] += int(row["rounds"])
            clusters[seed][method]["errors"] += int(row["errors"][method])
            clusters[seed][method]["rounds"] += int(row["rounds"])
    method_summary = {
        method: {
            **counts,
            "p_L": counts["errors"] / counts["rounds"],
            "p_X": counts["x"] / counts["rounds"],
            "p_Y": counts["y"] / counts["rounds"],
            "p_Z": counts["z"] / counts["rounds"],
        }
        for method, counts in totals.items()
    }
    family_summary = {
        family: {
            method: {
                **counts,
                "p_L": counts["errors"] / counts["rounds"],
            }
            for method, counts in method_rows.items()
        }
        for family, method_rows in sorted(families.items())
    }
    return method_summary, family_summary, dict(clusters)


def _bootstrap(
    clusters: dict[int, dict[str, dict[str, int]]],
    baseline: str,
    proposed: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    seed_rows = [clusters[seed] for seed in sorted(clusters)]
    baseline_errors = np.asarray([row[baseline]["errors"] for row in seed_rows], dtype=np.float64)
    proposed_errors = np.asarray([row[proposed]["errors"] for row in seed_rows], dtype=np.float64)
    rounds = np.asarray([row[baseline]["rounds"] for row in seed_rows], dtype=np.float64)
    _require(np.array_equal(rounds, [row[proposed]["rounds"] for row in seed_rows]), "unpaired cluster denominators")
    stats = config["statistics"]
    rng = np.random.default_rng(int(stats["bootstrap_seed"]))
    resamples = int(stats["bootstrap_resamples"])
    n = len(seed_rows)
    relative = np.empty(resamples, dtype=np.float64)
    absolute = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 5000):
        stop = min(start + 5000, resamples)
        indices = rng.integers(0, n, size=(stop - start, n))
        b = baseline_errors[indices].sum(axis=1) / rounds[indices].sum(axis=1)
        p = proposed_errors[indices].sum(axis=1) / rounds[indices].sum(axis=1)
        relative[start:stop] = np.where(b > 0, (b - p) / b, np.nan)
        absolute[start:stop] = b - p
    _require(np.isfinite(relative).all(), "zero or invalid bootstrap denominator")
    baseline_point = baseline_errors.sum() / rounds.sum()
    proposed_point = proposed_errors.sum() / rounds.sum()
    alpha = 1.0 - float(stats["confidence_level"])
    return {
        "cluster_count": n,
        "resamples": resamples,
        "bootstrap_seed": int(stats["bootstrap_seed"]),
        "baseline": baseline,
        "proposed": proposed,
        "baseline_p_L": float(baseline_point),
        "proposed_p_L": float(proposed_point),
        "relative_improvement_point": float((baseline_point - proposed_point) / baseline_point),
        "relative_improvement_lcb": float(np.quantile(relative, alpha / 2)),
        "relative_improvement_ucb": float(np.quantile(relative, 1 - alpha / 2)),
        "absolute_improvement_point": float(baseline_point - proposed_point),
        "absolute_improvement_lcb": float(np.quantile(absolute, alpha / 2)),
        "absolute_improvement_ucb": float(np.quantile(absolute, 1 - alpha / 2)),
    }


def _regret_decomposition(method_summary: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    path = list(config["nested_regret_path"])
    labels = ["estimator", "metric_likelihood", "logical_coset_sum", "posterior_marginalization", "risk_action"]
    rows = []
    start = method_summary[path[0]]["p_L"]
    telescoping = 0.0
    for label, before, after in zip(labels, path[:-1], path[1:]):
        absolute = method_summary[before]["p_L"] - method_summary[after]["p_L"]
        telescoping += absolute
        rows.append(
            {
                "component": label,
                "before_method": before,
                "after_method": after,
                "before_p_L": method_summary[before]["p_L"],
                "after_p_L": method_summary[after]["p_L"],
                "absolute_improvement": absolute,
                "relative_to_path_start": absolute / start,
            }
        )
    expected = method_summary[path[0]]["p_L"] - method_summary[path[-1]]["p_L"]
    _require(math.isclose(telescoping, expected, rel_tol=0, abs_tol=1e-15), "regret path does not telescope")
    return rows


def _semantic_mutations(raw: dict[str, Any], config: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    mutations: list[tuple[str, Any]] = []

    def add(name: str, mutate: Any) -> None:
        candidate = copy.deepcopy(raw)
        mutate(candidate)
        caught = False
        try:
            validate_raw(candidate, config, manifest, check_file_hashes=False)
        except (IntegrityError, KeyError, TypeError, ValueError):
            caught = True
        mutations.append((name, caught))

    add("pilot_split_poisoning", lambda value: value.__setitem__("source_split", "pilot"))
    add("missing_cell", lambda value: value["rows"].pop())
    add("duplicate_cell", lambda value: value["rows"].__setitem__(-1, copy.deepcopy(value["rows"][0])))
    add("wrong_rounds", lambda value: value["rows"][0].__setitem__("rounds", 511))
    add("delete_method", lambda value: value["rows"][0]["errors"].pop(config["methods"][0]))
    add("inflate_error_count", lambda value: value["rows"][0]["errors"].__setitem__(config["methods"][0], 513))
    add("break_pauli_sum", lambda value: value["rows"][0]["y_errors"].__setitem__(config["methods"][0], 7))
    add("forged_correctness", lambda value: value["correctness"].__setitem__("official_bsv_action_mismatches", 1))
    add("forged_tjoin", lambda value: value["correctness"].__setitem__("pure_julia_vs_official_correction_mismatches", 1))
    add("alias_failure", lambda value: value["correctness"].__setitem__("alias_action_mismatches", 1))
    add("causal_failure", lambda value: value["causality"].__setitem__("passed", False))
    add("invalid_trace_hash", lambda value: value["rows"][0].__setitem__("physical_trace_sha256", "demo"))
    add("duplicate_trace_hash", lambda value: value["rows"][1].__setitem__("physical_trace_sha256", value["rows"][0]["physical_trace_sha256"]))
    add("formal_label_injection", lambda value: value["rows"][0].__setitem__("formal", True))
    add("metadata_relabel", lambda value: value["rows"][0].__setitem__("family", "favorable_only"))
    return [{"mutation": name, "caught": caught} for name, caught in mutations]


def build_report() -> dict[str, Any]:
    config = _load(CONFIG)
    raw = _load(RAW)
    manifest_path = ROOT / config["source_manifest"]
    manifest = _load(manifest_path)
    validation = validate_raw(raw, config, manifest)
    source_audit = _validate_decoder_source()
    methods = validation["methods"]
    method_summary, family_summary, clusters = _aggregate(raw, methods)
    baseline_candidates = list(config["strongest_development_baseline_candidates"])
    strongest = min(baseline_candidates, key=lambda method: method_summary[method]["p_L"])
    proposed = "risk_aware_observed_only_action"
    bootstrap = _bootstrap(clusters, strongest, proposed, config)
    regret = _regret_decomposition(method_summary, config)
    mutations = _semantic_mutations(raw, config, manifest)
    stats = config["statistics"]
    headroom_pass = (
        bootstrap["relative_improvement_point"] >= float(stats["headroom_point_min"])
        and bootstrap["relative_improvement_lcb"] >= float(stats["headroom_lcb_min"])
    )
    gates = [
        {"gate": "raw_integrity_complete", "passed": validation["cell_count"] == 156},
        {"gate": "train_only", "passed": raw["source_split"] == "train"},
        {"gate": "all_registered_families", "passed": validation["family_count"] == 13},
        {"gate": "paired_trace_count", "passed": validation["trace_hash_count"] == validation["cell_count"]},
        {"gate": "official_bsv_zero_decision_mismatch", "passed": raw["correctness"]["official_bsv_action_mismatches"] == 0},
        {"gate": "official_log_odds_tolerance", "passed": raw["correctness"]["maximum_official_log10_odds_error"] <= config["correctness"]["maximum_log_odds_error"]},
        {"gate": "pure_julia_tjoin_zero_mismatch", "passed": raw["correctness"]["pure_julia_vs_official_correction_mismatches"] == 0},
        {"gate": "alias_convergence", "passed": raw["correctness"]["alias_action_mismatches"] == 0},
        {"gate": "coset_normalization", "passed": raw["correctness"]["maximum_probability_normalization_error"] <= config["correctness"]["maximum_probability_normalization_error"]},
        {"gate": "causal_prefix_mutation", "passed": raw["causality"]["passed"]},
        {"gate": "decoder_source_privilege_audit", "passed": source_audit["hidden_truth_signature_hits"] == 0 and source_audit["spatial_metadata_decision_hits"] == 0},
        {"gate": "semantic_mutations_caught", "passed": all(row["caught"] for row in mutations)},
        {"gate": "headroom_point_ge_15pct", "passed": bootstrap["relative_improvement_point"] >= float(stats["headroom_point_min"])},
        {"gate": "headroom_lcb_ge_12pct", "passed": bootstrap["relative_improvement_lcb"] >= float(stats["headroom_lcb_min"])},
    ]
    integrity_pass = all(row["passed"] for row in gates if not row["gate"].startswith("headroom_"))
    verdict = config["pass_verdict"] if integrity_pass and headroom_pass else config["failure_verdict"]
    family_headroom = {
        family: {
            "baseline": strongest,
            "baseline_p_L": rows[strongest]["p_L"],
            "proposed_p_L": rows[proposed]["p_L"],
            "relative_improvement": (rows[strongest]["p_L"] - rows[proposed]["p_L"]) / rows[strongest]["p_L"],
        }
        for family, rows in family_summary.items()
    }
    report = {
        "task_id": "T6.20.4",
        "schema_version": "t6.20.4-causal-headroom-report-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "split": "train",
            "development_only": True,
            "distance": config["distance"],
            "seed_clusters": validation["seed_count"],
            "families": validation["family_count"],
            "cells": validation["cell_count"],
            "physical_rounds": validation["rounds"],
            "formal_or_pilot_accessed": False,
            "claim_limit": config["eligibility_boundary"]["claim_limit"],
            "ceiling_boundary": "registered finite observed-only diagnostic stack; not the supremum over all causal decoders",
        },
        "integrity_repair_ledger": {
            "invalidated_outcomes_not_used": True,
            "registered_seeds_families_rounds_baselines_and_gates_changed": False,
            "performance_threshold_tuned_after_outcomes": False,
            "repairs": [
                {
                    "issue": "official pymatching through PythonCall crashed with a Windows native access violation",
                    "resolution": "replace only the long-run execution path with exhaustive d=3 T-join after 128 zero-mismatch official qualifications",
                },
                {
                    "issue": "the Gaussian CPD edge gap used sign(0), producing a zero gap at the cell centre",
                    "resolution": "replace it with the exact closed form 1-2*abs(r)",
                },
                {
                    "issue": "an invalid exploratory run passed generator-only spatial-pattern and variance-law loadings into decoder state",
                    "resolution": "discard that run and restrict the final decoder to past folded-residual moments, current observation and nominal base sigma",
                },
            ],
        },
        "correctness": raw["correctness"],
        "causality": raw["causality"],
        "source_privilege_audit": source_audit,
        "strongest_development_baseline_selection": {
            "candidates_retained": baseline_candidates,
            "selection_rule": "minimum aggregate train-only p_L; no candidate deleted",
            "selected": strongest,
            "candidate_p_L": {method: method_summary[method]["p_L"] for method in baseline_candidates},
            "boundary": "not yet the T6.22.4 strongest eligible formal denominator",
        },
        "method_summary": method_summary,
        "family_summary": family_summary,
        "family_headroom": family_headroom,
        "regret_decomposition": regret,
        "paired_bootstrap": bootstrap,
        "risk_action_diagnostics": {
            "interventions": sum(int(row["robust_interventions"]) for row in raw["rows"]),
            "rounds": validation["rounds"],
            "intervention_rate": sum(int(row["robust_interventions"]) for row in raw["rows"]) / validation["rounds"],
            "net_errors_reduced_vs_unprotected_posterior_predictive": (
                method_summary["posterior_predictive_exact_mld"]["errors"]
                - method_summary["risk_aware_observed_only_action"]["errors"]
            ),
            "interpretation": "trusted-bank robust action prevents induced adaptive errors, but final p_L only equals the strongest static baseline and therefore contributes zero usable headroom",
        },
        "headroom_gate": {
            "point_threshold": stats["headroom_point_min"],
            "lcb_threshold": stats["headroom_lcb_min"],
            "point_pass": bootstrap["relative_improvement_point"] >= float(stats["headroom_point_min"]),
            "lcb_pass": bootstrap["relative_improvement_lcb"] >= float(stats["headroom_lcb_min"]),
            "passed": headroom_pass,
            "failure_action": "do not enter T6.21 under Phase 6D v1; do not rescue by scenario deletion, denominator deletion or pilot/formal access",
        },
        "semantic_mutation_audit": mutations,
        "gates": gates,
        "gate_summary": {
            "passed": sum(bool(row["passed"]) for row in gates),
            "total": len(gates),
            "integrity_passed": integrity_pass,
            "headroom_passed": headroom_pass,
        },
        "bindings": [
            _binding(CONFIG),
            _binding(RAW),
            _binding(manifest_path),
            _binding(RUNNER),
            _binding(Path(__file__).resolve()),
            _binding(ROOT / "third_party" / "LatticeAlgorithms.jl" / "src" / "surface_code.jl"),
            _binding(ROOT / "third_party" / "LatticeAlgorithms.jl" / "src" / "matching_utils.jl"),
        ],
        "verdict": verdict,
    }
    canonical = copy.deepcopy(report)
    canonical.pop("generated_at_utc", None)
    report["analysis_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return report


def _source_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, values in report["method_summary"].items():
        for metric in ("errors", "rounds", "p_L", "p_X", "p_Y", "p_Z"):
            rows.append({"section": "method", "key": method, "metric": metric, "value": values[metric], "detail": "all train d=3 families"})
    for family, values in report["family_headroom"].items():
        for metric in ("baseline_p_L", "proposed_p_L", "relative_improvement"):
            rows.append({"section": "family_headroom", "key": family, "metric": metric, "value": values[metric], "detail": values["baseline"]})
    for component in report["regret_decomposition"]:
        for metric in ("before_p_L", "after_p_L", "absolute_improvement", "relative_to_path_start"):
            rows.append({"section": "regret", "key": component["component"], "metric": metric, "value": component[metric], "detail": f"{component['before_method']} -> {component['after_method']}"})
    for metric in (
        "baseline_p_L",
        "proposed_p_L",
        "relative_improvement_point",
        "relative_improvement_lcb",
        "relative_improvement_ucb",
        "absolute_improvement_point",
        "absolute_improvement_lcb",
        "absolute_improvement_ucb",
    ):
        rows.append({"section": "bootstrap", "key": report["paired_bootstrap"]["baseline"], "metric": metric, "value": report["paired_bootstrap"][metric], "detail": report["paired_bootstrap"]["proposed"]})
    for gate in report["gates"]:
        rows.append({"section": "gate", "key": gate["gate"], "metric": "passed", "value": gate["passed"], "detail": report["verdict"]})
    for mutation in report["semantic_mutation_audit"]:
        rows.append({"section": "mutation", "key": mutation["mutation"], "metric": "caught", "value": mutation["caught"], "detail": "fail-closed"})
    for binding in report["bindings"]:
        rows.append({"section": "binding", "key": binding["path"], "metric": "sha256", "value": binding["sha256"], "detail": binding["bytes"]})
    return rows


def _write_outputs(report: dict[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = _source_rows(report)
    with SOURCE_DATA.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["section", "key", "metric", "value", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    b = report["paired_bootstrap"]
    regret_lines = "\n".join(
        f"| {row['component']} | {row['before_p_L']:.6f} | {row['after_p_L']:.6f} | {row['absolute_improvement']:+.6f} |"
        for row in report["regret_decomposition"]
    )
    family_lines = "\n".join(
        f"| {family} | {values['baseline_p_L']:.6f} | {values['proposed_p_L']:.6f} | {100*values['relative_improvement']:+.2f}% |"
        for family, values in report["family_headroom"].items()
    )
    text = f"""# T6.20.4 Multimode causal headroom（development-only）

## 结论

**`{report['verdict']}`**。本任务只使用 T6.20.3 的 `train` split：{report['scope']['seed_clusters']} 个独立 seed-cluster、{report['scope']['families']} 个完整 family、{report['scope']['physical_rounds']:,} 轮；未访问 calibration、pilot 或 formal。

最强当前可执行 development baseline 是 `{b['baseline']}`（$p_L={b['baseline_p_L']:.6f}$），observed-only causal ceiling `{b['proposed']}` 为 $p_L={b['proposed_p_L']:.6f}$。相对改善点估计为 **{100*b['relative_improvement_point']:.2f}%**，paired seed-cluster bootstrap 95% CI 为 **[{100*b['relative_improvement_lcb']:.2f}%, {100*b['relative_improvement_ucb']:.2f}%]**，没有达到预注册的 `point >= 15%` 且 `LCB >= 12%`。

这个结果不能写成 SOTA，也不能授权进入 T6.21。它只说明：在当前 train task-signature 下，把 CPD 逐步换成 exact/posterior-predictive backend 的可用因果 headroom 太小，不能支撑预期的 10% formal 优势。

这里的“causal ceiling”只指本任务注册的有限 observed-only 诊断候选栈，不是所有因果解码器上的数学上确界，也不排除使用新机制和全新前瞻 split 的 v2。

## 五段 regret

| 组件 | 替换前 pL | 替换后 pL | 绝对改善 |
| --- | ---: | ---: | ---: |
{regret_lines}

正数表示降低 LER，负数表示退化。observed-only modewise estimator/likelihood/coset 路径明显退化；trusted-bank robust action 产生 {report['risk_action_diagnostics']['interventions']} 次干预，并相对未保护的 posterior-predictive arm 净减少 {report['risk_action_diagnostics']['net_errors_reduced_vs_unprotected_posterior_predictive']} 个逻辑错误，但最终只回到 strongest static baseline，仍然没有可用 headroom。这是安全回退价值，不是 LER 优势。

## 不删场景的 family 结果

| family | strongest baseline pL | causal ceiling pL | 相对改善 |
| --- | ---: | ---: | ---: |
{family_lines}

## 正确性与反简化检查

- explicit d=3 coset sum 对 official BSV：{report['correctness']['official_bsv_samples']} 个样本零 action mismatch，最大 log10-odds 误差 {report['correctness']['maximum_official_log10_odds_error']:.3e}。
- 纯 Julia exhaustive T-join 对 official `pymatching`：{report['correctness']['pure_julia_tjoin_samples']} 个样本零 correction mismatch；正式长跑不再依赖会崩溃的 PythonCall 路径。
- alias truncation：{report['correctness']['alias_convergence_samples']} 个样本零 action mismatch；概率归一最大误差 {report['correctness']['maximum_probability_normalization_error']:.3e}。
- future-suffix mutation：prefix action mismatch={report['causality']['prefix_action_mismatches']}、prefix prior max error={report['causality']['prefix_prior_max_abs_error']:.1e}，且 mutated suffix 后 action/posterior 均真实分叉。
- {len(report['semantic_mutation_audit'])}/{len(report['semantic_mutation_audit'])} 个完整性 mutation 被 fail-closed 捕获；所有 13 family 与两个 baseline candidate 均保留。
- 完整性修复账本保留三项被发现并修正的问题；含 generator-only spatial/variance-law privilege 的探索性结果已作废，最终报告没有使用。修复没有改变 seeds、families、rounds、baseline 候选、统计门，也没有在看见结果后调 performance threshold。

## 失败后的约束

Phase 6D v1 不得通过删去不利 family、删除 `static_mixture_exact_mld`/adaptive CPD、改阈值或访问 pilot/formal 来“救”此门。允许的后续是：(1) 将 T6.21--T6.24 标记为本路线 v1 不进入；(2) 继续独立的 single-mode RTL lane；(3) 若未来重开 multimode，必须形成新的、前瞻注册的机制假设，而不是对本数据调参。
"""
    MARKDOWN.write_text(text, encoding="utf-8")


def verify() -> dict[str, Any]:
    stored = _load(REPORT)
    rebuilt = build_report()
    for payload in (stored, rebuilt):
        payload.pop("generated_at_utc", None)
    _require(stored == rebuilt, "stored report differs from independent recomputation")
    _require(SOURCE_DATA.exists() and SOURCE_DATA.stat().st_size > 0, "source data missing")
    _require(MARKDOWN.exists() and MARKDOWN.stat().st_size > 0, "markdown report missing")
    return {
        "verdict": stored["verdict"],
        "gates": stored["gate_summary"],
        "relative_improvement_point": stored["paired_bootstrap"]["relative_improvement_point"],
        "relative_improvement_lcb": stored["paired_bootstrap"]["relative_improvement_lcb"],
        "analysis_sha256": stored["analysis_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
        return
    report = build_report()
    _write_outputs(report)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "report": REPORT.as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
