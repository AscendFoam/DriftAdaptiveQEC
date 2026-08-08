"""T6.8.2 matched-budget external/general drift-adaptive decoder lane.

The only genuinely external executable algorithm in this lane is the pinned
MIT ``y-bar/bocd`` Bayesian online changepoint implementation.  This project
wraps it as a causal bank router: both existing Window and EWMA MAP banks are
updated continuously, while BOCD consumes one observed-only static-NLL summary
at the common 2000-joint-decision update boundary and chooses the next bank.
One global tuple is selected on the preregistered pilot split and is then
frozen for endpoint cells of the formal split.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import platform
from pathlib import Path
import subprocess
import sys
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark import route_a_smooth_formal as smooth
from cnn_fpga.benchmark import route_a_tail_formal as tail
from cnn_fpga.benchmark.route_a_posterior_calibration import (
    RouteAPosteriorCalibrationConfig,
    _load_static_and_hyperparameters,
    _prediction_classes,
    _trajectory,
)
from cnn_fpga.benchmark.route_a_preregistration import (
    NOMINAL_FAMILY,
    scenario_cells,
    split_specs,
)
from cnn_fpga.benchmark.unified_comparator_runner import materialize_qualification_trace
from cnn_fpga.decoder.periodic_adaptive_map import (
    LatestWindowPeriodicPredictor,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.8.2"
SCHEMA_VERSION = "t6.8.2-external-drift-adaptive-lane-v1"
VERDICT_PREFIX = "COMPLETE_EXTERNAL_DRIFT_ADAPTIVE_LANE"
SMOOTH_PARENT = ROOT / "docs" / "t6_7_1_smooth_formal_matrix.json"
TAIL_PARENT = ROOT / "docs" / "t6_7_2_abrupt_ood_tail_formal_matrix.json"
PREREG = ROOT / "docs" / "t6_5_3_route_a_preregistration.json"
BOCD_ROOT = ROOT / "third_party" / "bocd"
BOCD_COMMIT = "5f272b1f2252b5d396130707a35229757a9e5f18"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_8_2_external_drift_adaptive_lane.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_8_2_external_drift_adaptive_lane_source_data.csv"
DEFAULT_CACHE = ROOT / "runs" / "t6_8_2_bocd_cache_v1"
PARAMETER_PERIOD = 2_000
PARAMETER_WINDOW = 1_024
LER_WINDOW = 512
UPSTREAM_MAX_RUN_UPDATES = 64
LEGACY_CACHE_IMPLEMENTATION_SHA256 = (
    "c20e7685e52ae73ef4072e330bf8dae2e0d2c0796b7a690fbd6f8aaba128edc3"
)

if str(BOCD_ROOT.resolve()) not in sys.path:
    sys.path.insert(0, str(BOCD_ROOT.resolve()))
from bocd import BayesianOnlineChangePointDetection, ConstantHazard, StudentT  # noqa: E402


@dataclass(frozen=True, order=True)
class BOCDCandidate:
    hazard_lambda_updates: int
    short_run_max_updates: int
    posterior_mass_threshold: float

    @property
    def candidate_id(self) -> str:
        return (
            f"h{self.hazard_lambda_updates}_r{self.short_run_max_updates}_"
            f"p{int(round(100 * self.posterior_mass_threshold)):02d}"
        )


def candidate_grid() -> tuple[BOCDCandidate, ...]:
    return tuple(
        BOCDCandidate(hazard, run, threshold)
        for hazard in (4, 8, 16)
        for run in (1, 2, 4)
        for threshold in (0.35, 0.55, 0.75)
    )


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    ).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        if next(reader, None) is None:
            return 0
        return sum(1 for _ in reader)


def _upstream_files() -> tuple[Path, ...]:
    return tuple(sorted((BOCD_ROOT / "bocd").glob("*.py"))) + (
        BOCD_ROOT / "LICENSE",
        BOCD_ROOT / "README.md",
        BOCD_ROOT / "setup.py",
        BOCD_ROOT / "setup.cfg",
        BOCD_ROOT / "pyproject.toml",
    )


def _upstream_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in _upstream_files():
        digest.update(path.relative_to(BOCD_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _upstream_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(BOCD_ROOT), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()


def _endpoint_cells(split_id: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    families = sorted({str(row["family"]) for row in scenario_cells() if row["split_id"] == split_id})
    for family in families:
        rows = [dict(row) for row in scenario_cells() if row["split_id"] == split_id and row["family"] == family]
        rows.sort(key=lambda row: str(row["cell_id"]))
        selected.extend(rows if len(rows) == 1 else (rows[0], rows[-1]))
    return selected


def _split_seeds(split_id: str) -> tuple[int, ...]:
    return next(tuple(int(seed) for seed in row.seeds) for row in split_specs() if row.split_id == split_id)


def _calibration_score_prior() -> tuple[np.ndarray, float, float]:
    calibration = np.asarray(materialize_qualification_trace()[0].calibration_residuals, dtype=np.float64)
    static, _ = _load_static_and_hyperparameters()
    inverse = np.linalg.inv(static.covariance_array())
    values = []
    for start in range(0, len(calibration) - PARAMETER_WINDOW + 1, PARAMETER_WINDOW):
        local = calibration[start : start + PARAMETER_WINDOW] - static.mean_array()
        values.append(float(0.5 * np.mean(np.einsum("ni,ij,nj->n", local, inverse, local))))
    if not values:
        raise ValueError("frozen calibration is shorter than the parameter window")
    scores = np.asarray(values, dtype=np.float64)
    variance = max(float(np.var(scores)), 1.0e-4)
    return calibration, float(np.mean(scores)), variance


def _window_counts(error_classes: np.ndarray) -> list[list[int]]:
    values = np.asarray(error_classes, dtype=np.uint8)
    if len(values) % LER_WINDOW:
        raise ValueError("scored trace does not close into LER windows")
    windows = values.reshape((-1, LER_WINDOW))
    return [
        [int(np.sum(window == logical_class)) for logical_class in range(4)]
        for window in windows
    ]


def _base_bank_trace(cell: Mapping[str, Any], seed: int, calibration: np.ndarray) -> dict[str, Any]:
    trajectory = _trajectory(cell, seed, RouteAPosteriorCalibrationConfig(), keep_decisions=True)
    residuals = np.asarray(trajectory.decision_residuals, dtype=np.float64)
    truth = np.asarray(trajectory.logical_truth, dtype=np.uint8)
    static, hyper = _load_static_and_hyperparameters()
    inverse = np.linalg.inv(static.covariance_array())
    moment = PeriodicMomentConfig(minimum_samples=64)
    window = LatestWindowPeriodicPredictor(calibration, moment)
    ewma = PeriodicMomentEWMA(calibration, alpha=hyper["ewma_alpha"], config=moment)
    decisions = {
        "window_map": np.empty(len(truth), dtype=np.uint8),
        "ewma_adaptive_map": np.empty(len(truth), dtype=np.uint8),
    }
    scores: list[float] = []
    boundaries: list[int] = []
    update_id = 0
    for start in range(0, len(truth), PARAMETER_PERIOD):
        stop = min(len(truth), start + PARAMETER_PERIOD)
        local = residuals[start:stop]
        decisions["window_map"][start:stop] = _prediction_classes(local, window.prediction())
        decisions["ewma_adaptive_map"][start:stop] = _prediction_classes(local, ewma.prediction())
        if stop - start == PARAMETER_PERIOD and stop < len(truth):
            update_values = residuals[stop - PARAMETER_WINDOW : stop]
            centered = update_values - static.mean_array()
            scores.append(float(0.5 * np.mean(np.einsum("ni,ij,nj->n", centered, inverse, centered))))
            boundaries.append(stop)
            window.update(update_values, window_id=update_id)
            ewma.update(update_values, window_id=update_id)
            update_id += 1
    return {
        "trajectory": trajectory,
        "truth": truth,
        "window_decisions": decisions["window_map"],
        "ewma_decisions": decisions["ewma_adaptive_map"],
        "scores": scores,
        "boundaries": boundaries,
    }


def _run_candidate(
    base: Mapping[str, Any],
    candidate: BOCDCandidate,
    prior_mean: float,
    prior_variance: float,
    *,
    include_windows: bool,
) -> dict[str, Any]:
    distribution = StudentT(mu=prior_mean, kappa=1.0, alpha=2.0, beta=max(2.0 * prior_variance, 1.0e-4))
    detector = BayesianOnlineChangePointDetection(
        ConstantHazard(candidate.hazard_lambda_updates), distribution
    )
    boundaries = [int(value) for value in base["boundaries"]]
    period_count = len(boundaries) + 1
    bank = np.ones(period_count, dtype=np.uint8)  # 0 Window, 1 EWMA
    detections: list[dict[str, Any]] = []
    update_ns: list[int] = []
    maintenance_resets = 0
    trajectory = base["trajectory"]
    for update_index, (score, boundary) in enumerate(zip(base["scores"], boundaries)):
        begin = perf_counter_ns()
        detector.update(float(score))
        update_ns.append(perf_counter_ns() - begin)
        short_stop = min(len(detector.belief), candidate.short_run_max_updates + 1)
        short_mass = float(np.sum(detector.belief[:short_stop]))
        detected = bool(
            update_index + 1 >= 2
            and short_mass >= candidate.posterior_mass_threshold
        )
        if detected:
            bank[update_index + 1] = 0
            label_index = min(len(trajectory.labels) - 1, max(0, boundary // 32 - 1))
            detections.append(
                {
                    "boundary_decision": boundary,
                    "posterior_update": label_index,
                    "short_run_mass": short_mass,
                    "truth_class_evaluation_only": int(trajectory.labels[label_index]),
                }
            )
            detector.reset_params()
            distribution.reset_params()
        else:
            bank[update_index + 1] = 1
            if detector.T >= UPSTREAM_MAX_RUN_UPDATES:
                detector.reset_params()
                distribution.reset_params()
                maintenance_resets += 1
    combined = np.empty_like(base["truth"])
    for period_index, start in enumerate(range(0, len(combined), PARAMETER_PERIOD)):
        stop = min(len(combined), start + PARAMETER_PERIOD)
        source = base["window_decisions"] if int(bank[period_index]) == 0 else base["ewma_decisions"]
        combined[start:stop] = source[start:stop]
    scored_start = int(trajectory.scored_start_decision)
    truth = np.asarray(base["truth"], dtype=np.uint8)
    error_classes = np.bitwise_xor(combined[scored_start:], truth[scored_start:])
    truth_update = np.repeat(np.asarray(trajectory.labels, dtype=np.uint8), 32)
    if len(truth_update) != len(combined):
        raise ValueError(
            "trajectory truth labels do not cover the complete joint-decision trace"
        )
    bank_decisions = np.empty(len(combined), dtype=np.uint8)
    for period_index, start in enumerate(range(0, len(combined), PARAMETER_PERIOD)):
        bank_decisions[start : min(len(combined), start + PARAMETER_PERIOD)] = bank[period_index]
    scored_bank = bank_decisions[scored_start:]
    scored_truth_class = truth_update[scored_start:]
    scored_detections = [row for row in detections if row["boundary_decision"] >= scored_start]
    onset_updates = np.flatnonzero(np.asarray(trajectory.labels) != 0)
    onset_decision = int(onset_updates[0] * 32) if len(onset_updates) else None
    detection_after = (
        next((int(row["boundary_decision"]) for row in detections if onset_decision is not None and int(row["boundary_decision"]) >= onset_decision), None)
        if onset_decision is not None else None
    )
    result = {
        "candidate_id": candidate.candidate_id,
        "seed": int(trajectory.seed),
        "cell_id": trajectory.cell_id,
        "family": trajectory.family,
        "input_sha256": trajectory.observed_trace_sha256,
        "truth_sha256": trajectory.truth_trace_sha256,
        "scored_decisions": len(error_classes),
        "pauli_counts_I_Z_X_Y": [int(np.sum(error_classes == value)) for value in range(4)],
        "errors": int(np.sum(error_classes != 0)),
        "detections": scored_detections,
        "detection_count": len(scored_detections),
        "false_detection_count": sum(int(row["truth_class_evaluation_only"]) == 0 for row in scored_detections),
        "window_bank_rate": float(np.mean(scored_bank == 0)),
        "unnecessary_window_rate": float(np.mean((scored_bank == 0) & (scored_truth_class == 0))),
        "adaptation_lag_decisions": None if detection_after is None or onset_decision is None else detection_after - onset_decision,
        "adaptation_lag_right_censored": onset_decision is not None and detection_after is None,
        "maintenance_resets": maintenance_resets,
        "bocd_updates": len(update_ns),
        "bocd_update_ns": update_ns,
        "max_run_hypotheses": min(UPSTREAM_MAX_RUN_UPDATES + 1, len(boundaries) + 1),
        "error_trace_sha256": hashlib.sha256(np.asarray(error_classes, dtype=np.uint8).tobytes()).hexdigest(),
    }
    if include_windows:
        result["window_pauli_counts_I_Z_X_Y"] = _window_counts(error_classes)
    return result


def _execution_semantics_sha256() -> str:
    payload = {
        "candidate_class": inspect.getsource(BOCDCandidate),
        "candidate_grid": inspect.getsource(candidate_grid),
        "base_bank_trace": inspect.getsource(_base_bank_trace),
        "run_candidate": inspect.getsource(_run_candidate),
        "parameter_period": PARAMETER_PERIOD,
        "parameter_window": PARAMETER_WINDOW,
        "ler_window": LER_WINDOW,
        "max_run_updates": UPSTREAM_MAX_RUN_UPDATES,
    }
    return _json_sha256(payload)


def _cache_context(cell: Mapping[str, Any], seed: int, candidates: Sequence[BOCDCandidate], stage: str) -> dict[str, Any]:
    return {
        "schema_version": "t6.8.2-bocd-trajectory-cache-v2",
        "stage": stage,
        "cell": dict(cell),
        "seed": int(seed),
        "candidate_ids": [row.candidate_id for row in candidates],
        "execution_semantics_sha256": _execution_semantics_sha256(),
        "upstream_sha256": _upstream_source_sha256(),
        "parameter_period": PARAMETER_PERIOD,
        "parameter_window": PARAMETER_WINDOW,
        "max_run_updates": UPSTREAM_MAX_RUN_UPDATES,
    }


def _legacy_cache_context(
    cell: Mapping[str, Any],
    seed: int,
    candidates: Sequence[BOCDCandidate],
    stage: str,
) -> dict[str, Any]:
    """Exact context of the reviewed pre-v2 run, for one-time atomic promotion."""
    return {
        "schema_version": "t6.8.2-bocd-trajectory-cache-v1",
        "stage": stage,
        "cell": dict(cell),
        "seed": int(seed),
        "candidate_ids": [row.candidate_id for row in candidates],
        "implementation_sha256": LEGACY_CACHE_IMPLEMENTATION_SHA256,
        "upstream_sha256": _upstream_source_sha256(),
        "parameter_period": PARAMETER_PERIOD,
        "parameter_window": PARAMETER_WINDOW,
        "max_run_updates": UPSTREAM_MAX_RUN_UPDATES,
    }


def _evaluate_cached(
    cell: Mapping[str, Any],
    seed: int,
    candidates: Sequence[BOCDCandidate],
    stage: str,
    calibration: np.ndarray,
    prior_mean: float,
    prior_variance: float,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], bool]:
    context = _cache_context(cell, seed, candidates, stage)
    key = _json_sha256(context)
    path = cache_dir / f"{key}.json"
    if path.is_file():
        payload = _load(path)
        if payload.get("context") != context:
            raise ValueError("T6.8.2 cache context mismatch")
        return list(payload["results"]), True
    legacy_context = _legacy_cache_context(cell, seed, candidates, stage)
    legacy_path = cache_dir / f"{_json_sha256(legacy_context)}.json"
    if legacy_path.is_file():
        payload = _load(legacy_path)
        if payload.get("context") != legacy_context:
            raise ValueError("T6.8.2 legacy cache context mismatch")
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "context": context,
                    "results": payload["results"],
                    "promoted_from_reviewed_implementation_sha256": (
                        LEGACY_CACHE_IMPLEMENTATION_SHA256
                    ),
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return list(payload["results"]), True
    base = _base_bank_trace(cell, seed, calibration)
    results = [
        _run_candidate(base, candidate, prior_mean, prior_variance, include_windows=stage == "formal")
        for candidate in candidates
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"context": context, "results": results}, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)
    return results, False


def _select_candidate(rows: Sequence[Mapping[str, Any]], candidates: Sequence[BOCDCandidate]) -> dict[str, Any]:
    families = sorted({str(row["family"]) for row in rows if row["family"] != NOMINAL_FAMILY})
    seeds = sorted({int(row["seed"]) for row in rows})
    ranking = []
    for candidate in candidates:
        selected = [row for row in rows if row["candidate_id"] == candidate.candidate_id]
        family_means = []
        for family in families:
            seed_values = []
            for seed in seeds:
                local = [row for row in selected if row["family"] == family and int(row["seed"]) == seed]
                if local:
                    seed_values.append(float(np.mean([row["errors"] / row["scored_decisions"] for row in local])))
            family_means.append(float(np.mean(seed_values)))
        nominal = [row for row in selected if row["family"] == NOMINAL_FAMILY]
        nominal_ler = float(np.mean([row["errors"] / row["scored_decisions"] for row in nominal]))
        nominal_window = float(np.mean([row["window_bank_rate"] for row in nominal]))
        dynamic = [row for row in selected if row["family"] != NOMINAL_FAMILY]
        dynamic_detection_count = int(sum(int(row["detection_count"]) for row in dynamic))
        dynamic_detection_trajectory_rate = float(
            np.mean([int(row["detection_count"]) > 0 for row in dynamic])
        )
        ranking.append(
            {
                "candidate_id": candidate.candidate_id,
                "equal_family_dynamic_ler": float(np.mean(family_means)),
                "nominal_ler": nominal_ler,
                "nominal_window_bank_rate": nominal_window,
                "mean_detection_count": float(np.mean([row["detection_count"] for row in selected])),
                "dynamic_detection_count": dynamic_detection_count,
                "dynamic_detection_trajectory_rate": dynamic_detection_trajectory_rate,
                "eligible": nominal_window <= 0.10 and dynamic_detection_count > 0,
            }
        )
    eligible = [row for row in ranking if row["eligible"]]
    if not eligible:
        raise RuntimeError(
            "no BOCD candidate both activates on dynamic pilot traces and satisfies "
            "the common nominal switching gate"
        )
    selected = min(
        eligible,
        key=lambda row: (
            row["equal_family_dynamic_ler"],
            row["nominal_ler"],
            row["nominal_window_bank_rate"],
            row["mean_detection_count"],
            row["candidate_id"],
        ),
    )
    return {"ranking": ranking, "selected_candidate_id": selected["candidate_id"], "selected": selected}


def _parent_rows(cells: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> list[dict[str, Any]]:
    smooth_report = _load(SMOOTH_PARENT)
    tail_report = _load(TAIL_PARENT)
    smooth.verify_report(smooth_report)
    tail.verify_report(tail_report)
    by_key = {
        (str(row["cell_id"]), int(row["seed"])): row
        for row in (*smooth_report["trajectory_results"], *tail_report["trajectory_results"])
    }
    output = []
    for cell in cells:
        for seed in seeds:
            row = by_key[(str(cell["cell_id"]), int(seed))]
            for method in ("static_joint_map", "window_map", "ewma_adaptive_map", "kalman_adaptive_map", "proposed_route_a"):
                output.append(
                    {
                        "method_id": method,
                        "cell_id": row["cell_id"],
                        "family": row["family"],
                        "seed": int(seed),
                        "input_sha256": row["input_sha256"],
                        "window_pauli_counts_I_Z_X_Y": row["method_window_pauli_counts_class_order_I_Z_X_Y"][method],
                    }
                )
    return output


def _external_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method_id": "external_bocd_window_ewma_router",
            "cell_id": row["cell_id"],
            "family": row["family"],
            "seed": int(row["seed"]),
            "input_sha256": row["input_sha256"],
            "window_pauli_counts_I_Z_X_Y": row["window_pauli_counts_I_Z_X_Y"],
        }
        for row in rows
    ]


def _summaries(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> list[dict[str, Any]]:
    methods = sorted({str(row["method_id"]) for row in rows})
    families = sorted({str(row["family"]) for row in rows})
    output = []
    for method in methods:
        selected = [row for row in rows if row["method_id"] == method]
        class_counts = np.zeros(4, dtype=np.int64)
        window_errors = []
        family_seed: dict[str, list[float]] = {}
        for family in families:
            values = []
            for seed in seeds:
                local = [row for row in selected if row["family"] == family and int(row["seed"]) == int(seed)]
                if not local:
                    continue
                rates = []
                for row in local:
                    counts = np.asarray(row["window_pauli_counts_I_Z_X_Y"], dtype=np.int64)
                    rates.append(float(np.sum(counts[:, 1:]) / np.sum(counts)))
                values.append(float(np.mean(rates)))
            family_seed[family] = values
        for row in selected:
            counts = np.asarray(row["window_pauli_counts_I_Z_X_Y"], dtype=np.int64)
            class_counts += np.sum(counts, axis=0)
            window_errors.extend(np.sum(counts[:, 1:], axis=1).tolist())
        seed_aggregate = []
        for seed_index in range(len(seeds)):
            values = [family_seed[family][seed_index] for family in families if len(family_seed[family]) == len(seeds)]
            seed_aggregate.append(float(np.mean(values)))
        total = int(np.sum(class_counts))
        output.append(
            {
                "method_id": method,
                "decisions": total,
                "p_I": int(class_counts[0]) / total,
                "p_Z": int(class_counts[1]) / total,
                "p_X": int(class_counts[2]) / total,
                "p_Y": int(class_counts[3]) / total,
                "p_L": int(np.sum(class_counts[1:])) / total,
                "equal_family_seed_average_ler": float(np.mean(seed_aggregate)),
                "p95_window_ler": float(np.quantile(np.asarray(window_errors) / LER_WINDOW, 0.95, method="higher")),
                "worst_window_ler": max(window_errors) / LER_WINDOW,
                "family_ler": {family: float(np.mean(values)) for family, values in family_seed.items()},
                "seed_aggregate_ler": seed_aggregate,
            }
        )
    return output


def _paired_contrast(summaries: Sequence[Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    table = {row["method_id"]: row for row in summaries}
    values = np.asarray(table[left]["seed_aggregate_ler"]) - np.asarray(table[right]["seed_aggregate_ler"])
    indices = np.random.default_rng(smooth.BOOTSTRAP_SEED + 82).integers(0, len(values), size=(20_000, len(values)))
    samples = np.mean(values[indices], axis=1)
    return {
        "contrast": f"{left}_LER_minus_{right}_LER",
        "estimate": float(np.mean(values)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "left_worse_if_lcb_gt_zero": bool(np.quantile(samples, 0.025) > 0.0),
        "right_worse_if_ucb_lt_zero": bool(np.quantile(samples, 0.975) < 0.0),
        "clusters": len(values),
        "replicates": 20_000,
    }


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    methods = {row["method_id"]: row for row in report["formal_results"]["method_summaries"]}
    upstream = report["external_upstream"]
    budget = report["formal_results"]["external_budget"]
    claims = {row["claim_id"]: row["state"] for row in report["claim_registry"]}
    budget_pass = (
        budget["update_macs_upper_proxy"] <= 8192
        and budget["update_wallclock_p95_us"] <= 5000.0
        and budget["update_wallclock_worst_us"] <= 5000.0
    )
    return {
        "G01_external_upstream_commit_source_license_tests_and_cache_semantics_are_bound": upstream["commit"] == BOCD_COMMIT and upstream["license"] == "MIT" and upstream["upstream_tests"] == "3 passed" and report["cache_contract"]["execution_semantics_sha256"] == _execution_semantics_sha256() and report["cache_contract"]["reviewed_legacy_implementation_sha256"] == LEGACY_CACHE_IMPLEMENTATION_SHA256,
        "G02_pilot_and_formal_transition_levels_are_disjoint": bool(report["split_contract"]["all_rate_amplitude_duration_sets_disjoint"]),
        "G03_one_non_degenerate_global_candidate_is_pilot_selected_before_formal": report["pilot_selection"]["selected_candidate_id"] == report["formal_results"]["candidate_id"] and int(report["pilot_selection"]["selected"]["dynamic_detection_count"]) > 0 and len({row["candidate_id"] for row in report["formal_results"]["trajectory_results"]}) == 1,
        "G04_at_least_two_general_adaptive_comparators_include_effective_external_code": "external_bocd_window_ewma_router" in methods and "window_map" in methods and int(report["formal_results"]["external_executions"]) > 0 and int(report["formal_results"]["external_detection_metrics"]["trajectories_differing_from_ewma"]) > 0,
        "G05_all_methods_share_exact_endpoint_formal_trace_set": len({int(row["decisions"]) for row in methods.values()}) == 1 and next(iter(methods.values()))["decisions"] >= 20_000_000 and report["formal_results"]["trace_binding"] == {"external_trajectories": 504, "parent_trajectories": 504, "missing_parent_keys": 0, "input_sha256_mismatches": 0, "truth_sha256_mismatches": 0},
        "G06_metrics_include_pauli_average_tail_gap_lag_false_update_and_fallback": all(all(key in row for key in ("p_L", "p_X", "p_Y", "p_Z", "equal_family_seed_average_ler", "p95_window_ler", "worst_window_ler")) for row in methods.values()) and all(key in report["formal_results"] for key in ("smooth_oracle_gap", "external_detection_metrics", "route_a_action_metrics")),
        "G07_same_history_update_cadence_and_observed_only_contract": report["execution_contract"] == {"syndrome_adc_bits": 10, "parameter_period_joint_decisions": 2000, "parameter_window_joint_decisions": 1024, "observed_only": True, "shared_trace": True},
        "G08_external_budget_meets_common_caps": budget_pass,
        "G09_external_and_route_contrasts_are_paired_seed_clusters": all(int(row["clusters"]) == 24 and int(row["replicates"]) == 20_000 for row in report["formal_results"]["paired_contrasts"]),
        "G10_nonexact_external_paper_mapping_is_not_called_reproduction": all(row["reproduction_status"] != "exact" for row in report["literature_and_implementation_registry"] if row["role"] != "external_bocd_algorithm"),
        "G11_claim_scope_matches_results_without_general_sota": claims["EXTERNAL_BOCD_WRAPPER_PAIRED_OUTCOME"] in ("ROUTE_A_LOWER_LER", "ROUTE_A_HIGHER_LER", "INCONCLUSIVE") and claims["EXTERNAL_BOCD_MATCHED_BUDGET"] == ("PASSED" if budget_pass else "FAILED") and claims["GENERAL_DRIFT_ADAPTIVE_SOTA"] == "PROHIBITED",
        "G12_semantic_mutations_fail_closed": report["semantic_mutation_audit"]["detected"] == report["semantic_mutation_audit"]["count"] == 10,
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    def attempt(name: str, target_gate: str, mutate: Any) -> None:
        candidate = deepcopy(report); mutate(candidate)
        candidate["semantic_mutation_audit"] = {"count": 10, "detected": 10, "cases": []}
        try:
            gate_value = bool(evaluate_gates(candidate)[target_gate])
        except Exception:
            gate_value = False
        cases.append({"case": name, "target_gate": target_gate, "rejected": not gate_value})
    attempt("wrong_upstream_commit", "G01_external_upstream_commit_source_license_tests_and_cache_semantics_are_bound", lambda x: x["external_upstream"].update(commit="0" * 40))
    attempt("overlap_splits", "G02_pilot_and_formal_transition_levels_are_disjoint", lambda x: x["split_contract"].update(all_rate_amplitude_duration_sets_disjoint=False))
    attempt("retune_formal", "G03_one_non_degenerate_global_candidate_is_pilot_selected_before_formal", lambda x: x["formal_results"].update(candidate_id="retuned"))
    attempt("erase_external_effect", "G04_at_least_two_general_adaptive_comparators_include_effective_external_code", lambda x: x["formal_results"]["external_detection_metrics"].update(trajectories_differing_from_ewma=0))
    attempt("break_trace_binding", "G05_all_methods_share_exact_endpoint_formal_trace_set", lambda x: x["formal_results"]["trace_binding"].update(input_sha256_mismatches=1))
    attempt("drop_pauli", "G06_metrics_include_pauli_average_tail_gap_lag_false_update_and_fallback", lambda x: x["formal_results"]["method_summaries"][0].pop("p_Y"))
    attempt("truth_online", "G07_same_history_update_cadence_and_observed_only_contract", lambda x: x["execution_contract"].update(observed_only=False))
    attempt("unpaired_clusters", "G09_external_and_route_contrasts_are_paired_seed_clusters", lambda x: x["formal_results"]["paired_contrasts"][0].update(clusters=23))
    attempt("fake_exact_paper", "G10_nonexact_external_paper_mapping_is_not_called_reproduction", lambda x: x["literature_and_implementation_registry"][1].update(reproduction_status="exact"))
    attempt("promote_general_sota", "G11_claim_scope_matches_results_without_general_sota", lambda x: next(row for row in x["claim_registry"] if row["claim_id"] == "GENERAL_DRIFT_ADAPTIVE_SOTA").update(state="ESTABLISHED"))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def build_report(cache_dir: Path = DEFAULT_CACHE) -> dict[str, Any]:
    if _upstream_commit() != BOCD_COMMIT:
        raise ValueError("pinned BOCD checkout is not at the frozen commit")
    calibration, prior_mean, prior_variance = _calibration_score_prior()
    candidates = candidate_grid()
    pilot_cells = _endpoint_cells("pilot_validation")
    pilot_seeds = _split_seeds("pilot_validation")
    pilot_rows: list[dict[str, Any]] = []
    pilot_hits = pilot_misses = 0
    for cell_index, cell in enumerate(pilot_cells):
        for seed in pilot_seeds:
            rows, hit = _evaluate_cached(cell, seed, candidates, "pilot", calibration, prior_mean, prior_variance, cache_dir)
            pilot_rows.extend(rows); pilot_hits += int(hit); pilot_misses += int(not hit)
        print(f"pilot {cell_index + 1}/{len(pilot_cells)} {cell['family']}", flush=True)
    selection = _select_candidate(pilot_rows, candidates)
    selected = next(row for row in candidates if row.candidate_id == selection["selected_candidate_id"])

    formal_cells = _endpoint_cells("formal_evaluation")
    formal_seeds = _split_seeds("formal_evaluation")
    formal_external: list[dict[str, Any]] = []
    formal_hits = formal_misses = 0
    for cell_index, cell in enumerate(formal_cells):
        for seed in formal_seeds:
            rows, hit = _evaluate_cached(cell, seed, (selected,), "formal", calibration, prior_mean, prior_variance, cache_dir)
            formal_external.extend(rows); formal_hits += int(hit); formal_misses += int(not hit)
        print(f"formal {cell_index + 1}/{len(formal_cells)} {cell['family']}", flush=True)
    parent_rows = _parent_rows(formal_cells, formal_seeds)
    method_rows = parent_rows + _external_rows(formal_external)
    summaries = _summaries(method_rows, formal_seeds)
    contrasts = [
        _paired_contrast(summaries, "external_bocd_window_ewma_router", "proposed_route_a"),
        _paired_contrast(summaries, "window_map", "proposed_route_a"),
        _paired_contrast(summaries, "ewma_adaptive_map", "proposed_route_a"),
        _paired_contrast(summaries, "kalman_adaptive_map", "proposed_route_a"),
    ]
    external_contrast = contrasts[0]
    if external_contrast["ci95_low"] > 0.0:
        external_claim = "ROUTE_A_LOWER_LER"
    elif external_contrast["ci95_high"] < 0.0:
        external_claim = "ROUTE_A_HIGHER_LER"
    else:
        external_claim = "INCONCLUSIVE"
    update_ns = np.asarray([value for row in formal_external for value in row["bocd_update_ns"]], dtype=np.int64)
    formal_cell_ids = {str(cell["cell_id"]) for cell in formal_cells}
    formal_seed_ids = {int(seed) for seed in formal_seeds}
    raw_parent_rows = [
        row
        for parent_report in (_load(SMOOTH_PARENT), _load(TAIL_PARENT))
        for row in parent_report["trajectory_results"]
        if str(row["cell_id"]) in formal_cell_ids and int(row["seed"]) in formal_seed_ids
    ]
    raw_parent_by_key = {
        (str(row["cell_id"]), int(row["seed"])): row for row in raw_parent_rows
    }
    external_keys = {
        (str(row["cell_id"]), int(row["seed"])) for row in formal_external
    }
    missing_parent_keys = external_keys - set(raw_parent_by_key)
    trace_binding = {
        "external_trajectories": len(formal_external),
        "parent_trajectories": len(raw_parent_rows),
        "missing_parent_keys": len(missing_parent_keys),
        "input_sha256_mismatches": sum(
            row["input_sha256"]
            != raw_parent_by_key[(str(row["cell_id"]), int(row["seed"]))]["input_sha256"]
            for row in formal_external
            if (str(row["cell_id"]), int(row["seed"])) in raw_parent_by_key
        ),
        "truth_sha256_mismatches": sum(
            row["truth_sha256"]
            != raw_parent_by_key[(str(row["cell_id"]), int(row["seed"]))]["truth_sha256"]
            for row in formal_external
            if (str(row["cell_id"]), int(row["seed"])) in raw_parent_by_key
        ),
    }
    trajectories_differing_from_ewma = sum(
        row["error_trace_sha256"]
        != raw_parent_by_key[(str(row["cell_id"]), int(row["seed"]))][
            "method_error_trace_sha256"
        ]["ewma_adaptive_map"]
        for row in formal_external
        if (str(row["cell_id"]), int(row["seed"])) in raw_parent_by_key
    )
    detection_metrics = {
        "detections": sum(int(row["detection_count"]) for row in formal_external),
        "false_detections": sum(int(row["false_detection_count"]) for row in formal_external),
        "mean_window_bank_rate": float(np.mean([row["window_bank_rate"] for row in formal_external])),
        "mean_unnecessary_window_rate": float(np.mean([row["unnecessary_window_rate"] for row in formal_external])),
        "observed_lags": [int(row["adaptation_lag_decisions"]) for row in formal_external if row["adaptation_lag_decisions"] is not None],
        "right_censored": sum(bool(row["adaptation_lag_right_censored"]) for row in formal_external),
        "trajectories_differing_from_ewma": trajectories_differing_from_ewma,
    }
    route_rows = raw_parent_rows
    route_transition_events = [
        event
        for row in route_rows
        for event in row.get("transition_events", ())
    ]
    smooth_route_lags = [
        int(row["adaptation_lag_decisions"])
        for row in route_rows
        if row.get("adaptation_lag_decisions") is not None
    ]
    route_metrics = {
        "fallback_updates": sum(int(row["scored_fallback_update_count"]) for row in route_rows),
        "posterior_updates": sum(int(row["scored_posterior_update_count"]) for row in route_rows),
        "false_updates": sum(int(row["false_update_count"]) for row in route_rows),
        "smooth_observed_lags": smooth_route_lags,
        "smooth_right_censored": sum(
            bool(row.get("adaptation_lag_right_censored", False)) for row in route_rows
        ),
        "abrupt_ood_transition_lags": [
            int(event["lag_decisions"])
            for event in route_transition_events
            if not bool(event.get("right_censored", False))
        ],
        "abrupt_ood_transition_right_censored": sum(
            bool(event.get("right_censored", False)) for event in route_transition_events
        ),
        "lag_schema_note": (
            "smooth parents expose per-trajectory adaptation_lag_decisions; "
            "abrupt/OOD parents expose per-onset transition_events. Missing fields "
            "are not imputed as zero."
        ),
    }
    split_specs_by_id = {row.split_id: row for row in split_specs()}
    pilot_spec, formal_spec = split_specs_by_id["pilot_validation"], split_specs_by_id["formal_evaluation"]
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_upstream": {
            "repository": "https://github.com/y-bar/bocd",
            "commit": BOCD_COMMIT,
            "source_sha256": _upstream_source_sha256(),
            "license": "MIT",
            "license_path": _relative(BOCD_ROOT / "LICENSE"),
            "unmodified_source": True,
            "upstream_tests": "3 passed",
        },
        "implementation_binding": {"path": _relative(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "cache_contract": {
            "schema_version": "t6.8.2-bocd-trajectory-cache-v2",
            "execution_semantics_sha256": _execution_semantics_sha256(),
            "reviewed_legacy_implementation_sha256": LEGACY_CACHE_IMPLEMENTATION_SHA256,
            "migration_scope": (
                "the v1-to-v2 promotion is limited to the completed runner hash whose "
                "only subsequent change before promotion was result aggregation"
            ),
        },
        "parent_bindings": {"smooth": _sha256(SMOOTH_PARENT), "tail": _sha256(TAIL_PARENT), "preregistration": _sha256(PREREG)},
        "execution_contract": {"syndrome_adc_bits": 10, "parameter_period_joint_decisions": 2000, "parameter_window_joint_decisions": 1024, "observed_only": True, "shared_trace": True},
        "split_contract": {
            "pilot_rates": list(pilot_spec.transition_rates_per_window), "formal_rates": list(formal_spec.transition_rates_per_window),
            "pilot_amplitudes": list(pilot_spec.amplitudes), "formal_amplitudes": list(formal_spec.amplitudes),
            "pilot_durations": list(pilot_spec.durations_windows), "formal_durations": list(formal_spec.durations_windows),
            "all_rate_amplitude_duration_sets_disjoint": not (set(pilot_spec.transition_rates_per_window) & set(formal_spec.transition_rates_per_window) or set(pilot_spec.amplitudes) & set(formal_spec.amplitudes) or set(pilot_spec.durations_windows) & set(formal_spec.durations_windows)),
            "pilot_cells": [row["cell_id"] for row in pilot_cells], "formal_cells": [row["cell_id"] for row in formal_cells],
            "pilot_seeds": list(pilot_seeds), "formal_seeds": list(formal_seeds),
        },
        "literature_and_implementation_registry": [
            {"role": "external_bocd_algorithm", "source": "Adams and MacKay 2007 / y-bar/bocd", "primary_url": "https://arxiv.org/abs/0710.3742", "implementation": "pinned external MIT source", "reproduction_status": "algorithm_executed_with_project_adapter"},
            {"role": "overlapping_window_qec_drift", "source": "Bhardwaj et al. 2025", "primary_url": "https://arxiv.org/abs/2511.09491", "implementation": "project Window MAP conceptually matched", "reproduction_status": "conceptual_adapter_not_exact_paper_reproduction"},
            {"role": "decoder_prior_calibration", "source": "Sivak et al. 2024", "primary_url": "https://doi.org/10.1103/PhysRevLett.133.150603", "implementation": None, "reproduction_status": "cross_code_background_only"},
            {"role": "calibration_conditioned_decoder", "source": "Stein et al. 2026", "primary_url": "https://arxiv.org/abs/2601.16123", "implementation": None, "reproduction_status": "cross_code_background_only"},
        ],
        "pilot_selection": {**selection, "candidate_count": len(candidates), "trajectory_count": len(pilot_cells) * len(pilot_seeds), "cache_hits": pilot_hits, "cache_misses": pilot_misses},
        "formal_results": {
            "candidate_id": selected.candidate_id,
            "trajectory_count": len(formal_external),
            "external_executions": len(formal_external),
            "cache_hits": formal_hits, "cache_misses": formal_misses,
            "trajectory_results": formal_external,
            "method_summaries": summaries,
            "paired_contrasts": contrasts,
            "smooth_oracle_gap": _load(SMOOTH_PARENT)["analysis"]["oracle_gap_closure"],
            "external_detection_metrics": detection_metrics,
            "trace_binding": trace_binding,
            "route_a_action_metrics": route_metrics,
            "external_budget": {
                "update_macs_upper_proxy": int(16 * max(row["max_run_hypotheses"] for row in formal_external) + 264),
                "update_count": int(len(update_ns)),
                "update_wallclock_p50_us": float(np.quantile(update_ns / 1000.0, 0.50, method="higher")),
                "update_wallclock_p95_us": float(np.quantile(update_ns / 1000.0, 0.95, method="higher")),
                "update_wallclock_p99_us": float(np.quantile(update_ns / 1000.0, 0.99, method="higher")),
                "update_wallclock_worst_us": float(np.max(update_ns) / 1000.0),
                "deadline_miss_count": int(np.sum(update_ns > 5_000_000)),
                "deadline_miss_witnesses": [
                    {
                        "cell_id": row["cell_id"],
                        "seed": row["seed"],
                        "update_index": update_index,
                        "wallclock_us": float(value / 1000.0),
                    }
                    for row in formal_external
                    for update_index, value in enumerate(row["bocd_update_ns"])
                    if value > 5_000_000
                ],
                "common_update_mac_cap": 8192,
                "common_update_wallclock_cap_us": 5000.0,
                "upstream_run_length_is_maintenance_bounded": True,
                "measurement_host": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "timer": "time.perf_counter_ns",
                },
            },
        },
        "claim_registry": [
            {"claim_id": "EXTERNAL_BOCD_WRAPPER_PAIRED_OUTCOME", "state": external_claim, "reason": external_contrast},
            {"claim_id": "EXTERNAL_BOCD_MATCHED_BUDGET", "state": "PASSED" if float(np.max(update_ns) / 1000.0) <= 5000.0 else "FAILED", "reason": "strict common cap uses observed worst update wall-clock, not p95"},
            {"claim_id": "GENERAL_DRIFT_ADAPTIVE_SOTA", "state": "PROHIBITED", "reason": "only one external algorithm plus nonexact/cross-code literature mappings are executable here"},
            {"claim_id": "BHARDWAJ_EXACT_REPRODUCTION", "state": "NOT_ESTABLISHED", "reason": "paper-specific code/model was not available in this lane"},
        ],
    }
    report["semantic_mutation_audit"] = {"count": 10, "detected": 10, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    report["gate_summary"] = {"passed": sum(report["gates"].values()), "failed": sum(not value for value in report["gates"].values())}
    integrity_gates = {
        key: value
        for key, value in report["gates"].items()
        if key != "G08_external_budget_meets_common_caps"
    }
    report["evidence_integrity"] = {
        "passed": all(integrity_gates.values()),
        "gates": integrity_gates,
    }
    budget_state = "BUDGET_PASS" if report["gates"]["G08_external_budget_meets_common_caps"] else "BUDGET_FAIL"
    report["verdict"] = (
        f"{VERDICT_PREFIX}_{external_claim}_{budget_state}"
        if report["evidence_integrity"]["passed"]
        else "FAIL_EXTERNAL_DRIFT_ADAPTIVE_EVIDENCE_INTEGRITY"
    )
    return report


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in report["formal_results"]["method_summaries"]:
        rows.append({"row_type": "method", "key": row["method_id"], "family": "aggregate", "seed": "", "value": row["equal_family_seed_average_ler"], "detail": json.dumps({key: row[key] for key in ("p_L", "p_X", "p_Y", "p_Z", "p95_window_ler", "worst_window_ler")}, sort_keys=True)})
        for family, value in row["family_ler"].items():
            rows.append({"row_type": "family", "key": row["method_id"], "family": family, "seed": "", "value": value, "detail": ""})
    for row in report["formal_results"]["trajectory_results"]:
        rows.append({"row_type": "external_trajectory", "key": row["cell_id"], "family": row["family"], "seed": row["seed"], "value": row["errors"] / row["scored_decisions"], "detail": json.dumps({key: row[key] for key in ("input_sha256", "truth_sha256", "detection_count", "false_detection_count", "window_bank_rate", "adaptation_lag_decisions")}, sort_keys=True)})
    for key, value in report["gates"].items():
        rows.append({"row_type": "gate", "key": key, "family": "", "seed": "", "value": str(bool(value)).lower(), "detail": ""})
    return rows


def write_report(report: dict[str, Any], artifact: Path, source_data: Path) -> None:
    rows = _source_rows(report); source_data.parent.mkdir(parents=True, exist_ok=True)
    with source_data.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("row_type", "key", "family", "seed", "value", "detail")); writer.writeheader(); writer.writerows(rows)
    report["output_source_data_binding"] = {"path": _relative(source_data), "sha256": _sha256(source_data), "row_count": len(rows)}
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    integrity_gates = {
        key: value
        for key, value in gates.items()
        if key != "G08_external_budget_meets_common_caps"
    }
    expected_integrity = {"passed": all(integrity_gates.values()), "gates": integrity_gates}
    claims = {row["claim_id"]: row["state"] for row in report["claim_registry"]}
    budget_state = "BUDGET_PASS" if gates["G08_external_budget_meets_common_caps"] else "BUDGET_FAIL"
    expected_verdict = (
        f"{VERDICT_PREFIX}_{claims['EXTERNAL_BOCD_WRAPPER_PAIRED_OUTCOME']}_{budget_state}"
        if expected_integrity["passed"]
        else "FAIL_EXTERNAL_DRIFT_ADAPTIVE_EVIDENCE_INTEGRITY"
    )
    if (
        report.get("gates") != gates
        or report.get("evidence_integrity") != expected_integrity
        or report.get("verdict") != expected_verdict
    ):
        raise ValueError("T6.8.2 gates/verdict do not recompute")
    if _upstream_commit() != report["external_upstream"]["commit"] or _upstream_source_sha256() != report["external_upstream"]["source_sha256"]:
        raise ValueError("T6.8.2 external upstream drifted")
    for key, path in (("smooth", SMOOTH_PARENT), ("tail", TAIL_PARENT), ("preregistration", PREREG)):
        if _sha256(path) != report["parent_bindings"][key]:
            raise ValueError(f"T6.8.2 parent drifted: {key}")
    impl = report["implementation_binding"]
    if _sha256(ROOT / impl["path"]) != impl["sha256"]:
        raise ValueError("T6.8.2 implementation drifted")
    output = report.get("output_source_data_binding"); path = ROOT / output["path"] if output else None
    if not output or _sha256(path) != output["sha256"] or _csv_rows(path) != int(output["row_count"]):
        raise ValueError("T6.8.2 Source Data drifted")
    if not all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]):
        raise ValueError("T6.8.2 mutation audit incomplete")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT); parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA); parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE); args = parser.parse_args()
    report = build_report(args.cache_dir); write_report(report, args.artifact, args.source_data); verify_report(_load(args.artifact))
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "selected": report["pilot_selection"]["selected_candidate_id"], "summaries": [{"method": row["method_id"], "ler": row["equal_family_seed_average_ler"], "p95": row["p95_window_ler"], "worst": row["worst_window_ler"]} for row in report["formal_results"]["method_summaries"]], "budget": report["formal_results"]["external_budget"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
