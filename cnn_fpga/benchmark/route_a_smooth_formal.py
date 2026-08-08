"""T6.7.1 untouched smooth-drift formal evaluation for Route A.

This runner is deliberately confirmatory.  It reads the immutable T6.5.3
formal split and the T6.6.3 V4 lock, evaluates every registered smooth cell and
seed, and never exposes hidden simulator state to a deployable method.  The
hidden-state Gaussian oracle is reconstructed in a physically separate
evaluation lane solely for static-to-oracle gap accounting.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from math import ceil, pi
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit

from cnn_fpga.benchmark.route_a_posterior_calibration import (
    CLASS_TO_INDEX,
    DUAL_BANK_EXPERTS,
    PREQUENTIAL_SCORE_MEMORY,
    SMOOTH_BANK_POSTERIOR_MIN,
    RouteAPosteriorCalibrationConfig,
    _causal_gaussian_predictive_score,
    _canonical_sha256,
    _load_static_and_hyperparameters,
    _prediction_classes,
    _selected_action_trace,
    _trajectory,
    verify_report as verify_threshold_lock_report,
)
from cnn_fpga.benchmark.route_a_preregistration import (
    DEFAULT_ARTIFACT as PREREG_ARTIFACT,
    SMOOTH_FAMILIES,
    protocol_payload,
    scenario_cells,
    split_specs,
    validate_protocol,
)
from cnn_fpga.benchmark.unified_comparator_runner import (
    derive_method_costs,
    materialize_qualification_trace,
)
from cnn_fpga.decoder.periodic_adaptive_map import (
    ConstantVelocityPeriodicKalman,
    LatestWindowPeriodicPredictor,
    PeriodicMomentConfig,
    PeriodicMomentEWMA,
    scaled_periodic_kalman_config,
)
from cnn_fpga.decoder.route_a_regime_posterior import (
    ObservedTailEventModel,
    RouteAPosteriorModel,
    temperature_scale,
)
from physics.constants import LATTICE_CONST
from physics.ideal_gkp_decoder import map_decode_2d


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.7.1"
PROTOCOL_ID = "ROUTE-A-SMOOTH-FORMAL-V1"
SCHEMA_VERSION = "t6.7.1-route-a-smooth-formal-v1"
DEFAULT_LOCK = ROOT / "docs" / "t6_6_3_route_a_posterior_threshold_lock.json"
DEFAULT_ARTIFACT = ROOT / "docs" / "t6_7_1_smooth_formal_matrix.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_7_1_smooth_formal_matrix_source_data.csv"
DEFAULT_ACCESS_LEDGER = ROOT / "runs" / "t6_7_1_formal_access_ledger.json"
DEFAULT_CACHE_DIR = ROOT / "runs" / "t6_7_1_smooth_formal_cache_v1"

METHODS = (
    "standard_binning",
    "static_joint_map",
    "window_map",
    "ewma_adaptive_map",
    "kalman_adaptive_map",
    "proposed_route_a",
    "hidden_state_oracle",
)
DEPLOYABLE_METHODS = METHODS[:-1]
PRIMARY_BASELINE = "ewma_adaptive_map"
LER_WINDOW_DECISIONS = 512
POSTERIOR_WINDOW_DECISIONS = 32
PARAMETER_PERIOD_DECISIONS = 2_000
PARAMETER_WINDOW_DECISIONS = 1_024
PAULI_CLASS_ORDER = ("I", "Z", "X", "Y")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 202607176999


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes()).hexdigest()


def _algorithm_contract() -> dict[str, object]:
    """Numerical cache contract; changes require a new cache namespace."""

    return {
        "schema_version": "t6.7.1-smooth-formal-cell-cache-v1",
        "methods": list(METHODS),
        "pauli_class_order": list(PAULI_CLASS_ORDER),
        "parameter_period_decisions": PARAMETER_PERIOD_DECISIONS,
        "parameter_window_decisions": PARAMETER_WINDOW_DECISIONS,
        "ler_window_decisions": LER_WINDOW_DECISIONS,
        "router": (
            "continuously update Window/EWMA shadows; at each complete period "
            "promote Window only after pre-update NLL win, OPEN action and "
            "smooth posterior threshold; otherwise promote EWMA"
        ),
        "oracle": (
            "evaluation-only per-decision correlated Gaussian periodic MAP, "
            "tail_sigma=10 and uniform logical prior"
        ),
        "aggregation": "cell mean within seed/family; equal family within seed",
    }


ALGORITHM_CONTRACT_SHA256 = _json_sha256(_algorithm_contract())


def _load_parents(lock_path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    prereg = json.loads(PREREG_ARTIFACT.read_text(encoding="utf-8"))
    # Validate the immutable payload stored at freeze time; the historical
    # formal-artifact-absence field is evidence about that time, not a live
    # condition to be re-evaluated after formal execution.
    protocol = {key: prereg[key] for key in protocol_payload()}
    validate_protocol(protocol)
    if prereg.get("protocol_sha256") != _json_sha256(protocol):
        raise ValueError("T6.5.3 preregistration protocol hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_threshold_lock_report(lock)
    core = lock["threshold_lock"]["lock_core"]
    if lock["threshold_lock"]["lock_sha256"] != _canonical_sha256(core):
        raise ValueError("T6.6.3 threshold lock hash mismatch")
    if core["strongest_deployable_baseline"] != PRIMARY_BASELINE:
        raise ValueError("formal primary baseline is not the pilot-locked EWMA")
    if tuple(core["deployable_expert_banks"]) != DUAL_BANK_EXPERTS:
        raise ValueError("formal dual-bank experts differ from the V4 lock")
    if float(core["smooth_bank_posterior_min"]) != SMOOTH_BANK_POSTERIOR_MIN:
        raise ValueError("formal smooth bank threshold differs from the V4 lock")
    if float(core["prequential_score_memory"]) != PREQUENTIAL_SCORE_MEMORY:
        raise ValueError("formal prequential score memory differs from the V4 lock")
    return {"preregistration": prereg, "threshold_lock": lock}


def _formal_cells_and_seeds() -> tuple[tuple[dict[str, object], ...], tuple[int, ...]]:
    cells = tuple(
        row
        for row in scenario_cells()
        if row["split_id"] == "formal_evaluation" and row["family"] in SMOOTH_FAMILIES
    )
    formal = next(row for row in split_specs() if row.split_id == "formal_evaluation")
    if len(cells) != 24 or {str(row["family"]) for row in cells} != set(SMOOTH_FAMILIES):
        raise ValueError("formal smooth cell matrix is not the frozen 4x6 design")
    if len(formal.seeds) != 24:
        raise ValueError("formal seed cluster count is not 24")
    return cells, formal.seeds


def _record_formal_access(parents: Mapping[str, Any], path: Path) -> dict[str, object]:
    lock = parents["threshold_lock"]
    payload = {
        "schema_version": "t6.7.1-formal-access-ledger-v1",
        "task_id": TASK_ID,
        "first_access_is_irreversible": True,
        "formal_split": "formal_evaluation/smooth",
        "preregistration_path": PREREG_ARTIFACT.relative_to(ROOT).as_posix(),
        "preregistration_sha256": _sha256(PREREG_ARTIFACT),
        "threshold_lock_path": DEFAULT_LOCK.relative_to(ROOT).as_posix(),
        "threshold_lock_artifact_sha256": _sha256(DEFAULT_LOCK),
        "threshold_lock_sha256": lock["threshold_lock"]["lock_sha256"],
        "primary_baseline": PRIMARY_BASELINE,
        "prohibitions": [
            "no formal baseline reselection",
            "no threshold/router retuning",
            "no family/cell/seed deletion",
            "no result-conditioned reweighting",
        ],
    }
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if {key: stored[key] for key in payload} != payload:
            raise ValueError("formal access ledger conflicts with the frozen parents")
        return stored
    payload["first_accessed_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def _hidden_smooth_parameters(
    cell: Mapping[str, object],
    settings: RouteAPosteriorCalibrationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the registered hidden Gaussian parameters for oracle use."""

    family = str(cell["family"])
    if family not in SMOOTH_FAMILIES:
        raise ValueError("smooth oracle parameter reconstruction received a tail family")
    preamble = int(cell["nominal_preamble_windows"])
    scored = int(cell["scored_windows"])
    updates = (preamble + scored) * settings.posterior_updates_per_ler_window
    ler_index = np.repeat(
        np.arange(preamble + scored), settings.posterior_updates_per_ler_window
    )
    local_update = np.tile(
        np.arange(settings.posterior_updates_per_ler_window), preamble + scored
    )
    smooth = ler_index >= preamble
    progress = np.clip(
        (
            ler_index
            - preamble
            + local_update / settings.posterior_updates_per_ler_window
        )
        / max(1, scored - 1),
        0.0,
        1.0,
    )
    amplitude = float(cell["amplitude"])
    duration = int(cell["duration_windows"])
    base_sigma = 0.28
    mu_q = np.zeros(updates, dtype=np.float64)
    mu_p = np.zeros(updates, dtype=np.float64)
    sigma_q = np.full(updates, base_sigma, dtype=np.float64)
    sigma_p = np.full(updates, base_sigma * 1.05, dtype=np.float64)
    rho = np.full(updates, 0.05, dtype=np.float64)
    if family == "mean_drift":
        mu_q[smooth] = amplitude * LATTICE_CONST * (2.0 * progress[smooth] - 1.0)
        mu_p[smooth] = -0.7 * mu_q[smooth]
    elif family == "variance_drift":
        sigma_q[smooth] *= 1.0 + 2.0 * amplitude * progress[smooth]
        sigma_p[smooth] *= 1.0 + 1.4 * amplitude * progress[smooth]
    elif family == "correlation_drift":
        rho[smooth] = 3.0 * amplitude * (2.0 * progress[smooth] - 1.0)
    elif family == "periodic_drift":
        phase = (
            2.0
            * pi
            * (
                ler_index
                + local_update / settings.posterior_updates_per_ler_window
            )
            / max(1, duration)
        )
        mu_q[smooth] = amplitude * LATTICE_CONST * np.sin(phase[smooth])
        mu_p[smooth] = (
            0.75 * amplitude * LATTICE_CONST * np.cos(phase[smooth] + 0.3)
        )
        sigma_q[smooth] *= 1.0 + amplitude * np.cos(phase[smooth] - 0.2)
        sigma_p[smooth] *= 1.0 + amplitude * np.sin(phase[smooth] + 0.4)
        rho[smooth] = np.clip(
            2.5 * amplitude * np.sin(phase[smooth]), -0.85, 0.85
        )
    repeat = settings.posterior_window_cycles
    return tuple(np.repeat(value, repeat) for value in (mu_q, mu_p, sigma_q, sigma_p, rho))  # type: ignore[return-value]


@njit(cache=True)
def _variable_gaussian_oracle_kernel(
    residuals: np.ndarray,
    mu_q: np.ndarray,
    mu_p: np.ndarray,
    sigma_q: np.ndarray,
    sigma_p: np.ndarray,
    rho: np.ndarray,
    lattice: float,
) -> np.ndarray:
    """Exact hard-decision counterpart of ``map_decode_2d`` for varying states."""

    output = np.empty(len(residuals), dtype=np.uint8)
    for sample in range(len(residuals)):
        sq = sigma_q[sample]
        sp = sigma_p[sample]
        covariance = rho[sample] * sq * sp
        determinant = sq * sq * sp * sp - covariance * covariance
        inv_qq = sp * sp / determinant
        inv_pp = sq * sq / determinant
        inv_qp = -covariance / determinant
        nearest_q = int(np.floor((mu_q[sample] - residuals[sample, 0]) / lattice + 0.5))
        nearest_p = int(np.floor((mu_p[sample] - residuals[sample, 1]) / lattice + 0.5))
        radius_q = max(2, int(ceil(10.0 * sq / lattice)) + 2)
        radius_p = max(2, int(ceil(10.0 * sp / lattice)) + 2)
        maxima = np.full(4, -np.inf)
        for offset_q in range(-radius_q, radius_q + 1):
            alias_q = nearest_q + offset_q
            dq = residuals[sample, 0] + alias_q * lattice - mu_q[sample]
            for offset_p in range(-radius_p, radius_p + 1):
                alias_p = nearest_p + offset_p
                dp = residuals[sample, 1] + alias_p * lattice - mu_p[sample]
                logical_class = (alias_q & 1) * 2 + (alias_p & 1)
                log_weight = -0.5 * (
                    inv_qq * dq * dq + 2.0 * inv_qp * dq * dp + inv_pp * dp * dp
                )
                if log_weight > maxima[logical_class]:
                    maxima[logical_class] = log_weight
        sums = np.zeros(4, dtype=np.float64)
        for offset_q in range(-radius_q, radius_q + 1):
            alias_q = nearest_q + offset_q
            dq = residuals[sample, 0] + alias_q * lattice - mu_q[sample]
            for offset_p in range(-radius_p, radius_p + 1):
                alias_p = nearest_p + offset_p
                dp = residuals[sample, 1] + alias_p * lattice - mu_p[sample]
                logical_class = (alias_q & 1) * 2 + (alias_p & 1)
                log_weight = -0.5 * (
                    inv_qq * dq * dq + 2.0 * inv_qp * dq * dp + inv_pp * dp * dp
                )
                sums[logical_class] += np.exp(log_weight - maxima[logical_class])
        best_class = 0
        best_score = maxima[0] + np.log(sums[0])
        for logical_class in range(1, 4):
            score = maxima[logical_class] + np.log(sums[logical_class])
            if score > best_score:
                best_score = score
                best_class = logical_class
        output[sample] = best_class
    return output


def variable_gaussian_oracle_decisions(
    residuals: np.ndarray,
    parameters: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    values = np.asarray(residuals, dtype=np.float64)
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in parameters)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("oracle residuals must have shape (n,2)")
    if any(value.shape != (len(values),) for value in arrays):
        raise ValueError("oracle hidden parameters are not decision-aligned")
    if not np.all(np.isfinite(values)) or any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("oracle inputs must be finite")
    if np.any(arrays[2] <= 0.0) or np.any(arrays[3] <= 0.0) or np.any(np.abs(arrays[4]) >= 1.0):
        raise ValueError("oracle covariance parameters are invalid")
    return _variable_gaussian_oracle_kernel(values, *arrays, float(LATTICE_CONST))


def _window_class_counts(error_classes: np.ndarray) -> np.ndarray:
    values = np.asarray(error_classes, dtype=np.uint8)
    if values.ndim != 1 or len(values) % LER_WINDOW_DECISIONS or np.any(values > 3):
        raise ValueError("scored Pauli error trace is invalid")
    windows = values.reshape((-1, LER_WINDOW_DECISIONS))
    counts = np.empty((len(windows), 4), dtype=np.int16)
    for logical_class in range(4):
        counts[:, logical_class] = np.sum(windows == logical_class, axis=1)
    if not np.all(np.sum(counts, axis=1) == LER_WINDOW_DECISIONS):
        raise RuntimeError("Pauli class counts do not close")
    return counts


def _load_models(lock: Mapping[str, Any]) -> tuple[RouteAPosteriorModel, ObservedTailEventModel, float, dict[str, object]]:
    model = RouteAPosteriorModel.from_payload(lock["posterior_model"])
    event = ObservedTailEventModel.from_payload(lock["event_model"])
    if model.sha256 != lock["posterior_model_sha256"] or event.sha256 != lock["event_model_sha256"]:
        raise ValueError("formal posterior/event checkpoint hash mismatch")
    core = lock["threshold_lock"]["lock_core"]
    return model, event, float(core["posterior_temperature"]), dict(core["selected_threshold_tuple"])


def _cache_context(
    trajectory: Any,
    cell: Mapping[str, object],
    parents: Mapping[str, Any],
    calibration_sha256: str,
) -> dict[str, object]:
    lock = parents["threshold_lock"]
    return {
        **_algorithm_contract(),
        "algorithm_contract_sha256": ALGORITHM_CONTRACT_SHA256,
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": _sha256(PREREG_ARTIFACT),
        "threshold_lock_artifact_sha256": _sha256(DEFAULT_LOCK),
        "threshold_lock_sha256": lock["threshold_lock"]["lock_sha256"],
        "baseline_selection_sha256": lock["threshold_lock"]["lock_core"]["baseline_selection_sha256"],
        "calibration_sha256": calibration_sha256,
        "cell": dict(cell),
        "seed": int(trajectory.seed),
        "observed_trace_sha256": trajectory.observed_trace_sha256,
        "truth_trace_sha256": trajectory.truth_trace_sha256,
        "scored_start_decision": int(trajectory.scored_start_decision),
    }


def _run_trajectory(
    cell: Mapping[str, object],
    seed: int,
    parents: Mapping[str, Any],
    calibration: np.ndarray,
    cache_dir: Path,
) -> tuple[dict[str, object], bool]:
    settings = RouteAPosteriorCalibrationConfig()
    trajectory = _trajectory(cell, seed, settings, keep_decisions=True)
    residuals = np.asarray(trajectory.decision_residuals, dtype=np.float64)
    truth = np.asarray(trajectory.logical_truth, dtype=np.uint8)
    calibration_sha256 = _array_sha256(calibration, "<f8")
    context = _cache_context(trajectory, cell, parents, calibration_sha256)
    cache_key = _json_sha256(context)
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if str(cached["context_json"].item()) != json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=True):
                raise ValueError(f"formal cache context mismatch: {cache_path}")
            result = json.loads(str(cached["result_json"].item()))
        _validate_trajectory_result(result)
        return result, True

    lock = parents["threshold_lock"]
    model, event_model, temperature, thresholds = _load_models(lock)
    static, hyper = _load_static_and_hyperparameters()
    moment = PeriodicMomentConfig(minimum_samples=64)
    latest = LatestWindowPeriodicPredictor(calibration, moment)
    ewma = PeriodicMomentEWMA(calibration, alpha=hyper["ewma_alpha"], config=moment)
    kalman = ConstantVelocityPeriodicKalman(
        calibration,
        moment_config=moment,
        kalman_config=scaled_periodic_kalman_config(
            process_scale=hyper["kalman_process_scale"],
            measurement_scale=hyper["kalman_measurement_scale"],
        ),
    )
    predictors = {
        "window_map": latest,
        "ewma_adaptive_map": ewma,
        "kalman_adaptive_map": kalman,
    }
    decisions = {method: np.empty(len(truth), dtype=np.uint8) for method in METHODS}
    decisions["standard_binning"].fill(0)
    wallclock_ns = {method: 0 for method in METHODS}
    preferred_next: list[int] = []
    parameter_window_id = 0
    posterior_begin = perf_counter_ns()
    posterior = temperature_scale(model.filter_base(trajectory.features), temperature)
    deterministic_ood = np.asarray(trajectory.ood_score_codes, dtype=np.uint8)
    event_ood = event_model.score_codes(trajectory.features)
    ood = np.maximum(deterministic_ood, event_ood)
    actions = _selected_action_trace(
        posterior,
        ood,
        np.asarray([0], dtype=np.int64),
        np.asarray([len(posterior)], dtype=np.int64),
        thresholds,
    )
    route_model_ns = perf_counter_ns() - posterior_begin

    for start in range(0, len(truth), PARAMETER_PERIOD_DECISIONS):
        stop = min(len(truth), start + PARAMETER_PERIOD_DECISIONS)
        local = residuals[start:stop]
        begin = perf_counter_ns()
        static_result = map_decode_2d(
            local, static.covariance_array(), mean=static.mean_array()
        )
        decisions["static_joint_map"][start:stop] = np.asarray(
            static_result.logical_class, dtype=np.uint8
        )
        wallclock_ns["static_joint_map"] += perf_counter_ns() - begin
        predictions: dict[str, Any] = {}
        for method, predictor in predictors.items():
            prediction = predictor.prediction()
            predictions[method] = prediction
            begin = perf_counter_ns()
            decisions[method][start:stop] = _prediction_classes(local, prediction)
            wallclock_ns[method] += perf_counter_ns() - begin
        if stop - start == PARAMETER_PERIOD_DECISIONS and stop < len(truth):
            update_values = residuals[stop - PARAMETER_WINDOW_DECISIONS : stop]
            scores = np.asarray(
                [
                    _causal_gaussian_predictive_score(update_values, predictions[method])
                    for method in DUAL_BANK_EXPERTS
                ],
                dtype=np.float64,
            )
            preferred_next.append(int(np.argmin(scores)))
            for method, predictor in predictors.items():
                begin = perf_counter_ns()
                predictor.update(update_values, window_id=parameter_window_id)
                wallclock_ns[method] += perf_counter_ns() - begin
            parameter_window_id += 1

    period_count = (len(truth) + PARAMETER_PERIOD_DECISIONS - 1) // PARAMETER_PERIOD_DECISIONS
    active_expert = np.ones(period_count, dtype=np.uint8)
    commit_rows: list[dict[str, object]] = []
    for period_index, preferred in enumerate(preferred_next):
        stop = (period_index + 1) * PARAMETER_PERIOD_DECISIONS
        boundary_update = min(len(actions) - 1, max(0, stop // POSTERIOR_WINDOW_DECISIONS - 1))
        window_allowed = bool(
            preferred == 0
            and posterior[boundary_update, CLASS_TO_INDEX["smooth"]]
            >= SMOOTH_BANK_POSTERIOR_MIN
            and actions[boundary_update] == 0
        )
        selected_expert = 0 if window_allowed else 1
        active_expert[period_index + 1] = selected_expert
        commit_rows.append(
            {
                "commit_decision": stop,
                "boundary_posterior_update": boundary_update,
                "preferred_expert": DUAL_BANK_EXPERTS[preferred],
                "selected_expert": DUAL_BANK_EXPERTS[selected_expert],
                "policy_action": int(actions[boundary_update]),
                "smooth_posterior": float(posterior[boundary_update, CLASS_TO_INDEX["smooth"]]),
                "truth_class": int(trajectory.labels[boundary_update]),
                "accepted_and_acknowledged": True,
            }
        )
    router_begin = perf_counter_ns()
    for period_index, start in enumerate(range(0, len(truth), PARAMETER_PERIOD_DECISIONS)):
        stop = min(len(truth), start + PARAMETER_PERIOD_DECISIONS)
        source = DUAL_BANK_EXPERTS[int(active_expert[period_index])]
        decisions["proposed_route_a"][start:stop] = decisions[source][start:stop]
    wallclock_ns["proposed_route_a"] = (
        route_model_ns
        + wallclock_ns["window_map"]
        + wallclock_ns["ewma_adaptive_map"]
        + perf_counter_ns()
        - router_begin
    )

    oracle_begin = perf_counter_ns()
    decisions["hidden_state_oracle"] = variable_gaussian_oracle_decisions(
        residuals, _hidden_smooth_parameters(cell, settings)
    )
    wallclock_ns["hidden_state_oracle"] = perf_counter_ns() - oracle_begin

    scored_start = int(trajectory.scored_start_decision)
    scored_slice = slice(scored_start, len(truth))
    scored_truth = truth[scored_slice]
    if len(scored_truth) != int(cell["scored_windows"]) * LER_WINDOW_DECISIONS:
        raise RuntimeError("formal scored decision count differs from preregistration")
    method_counts: dict[str, list[list[int]]] = {}
    method_error_hashes: dict[str, str] = {}
    method_error_vectors: dict[str, np.ndarray] = {}
    for method in METHODS:
        error_classes = np.bitwise_xor(decisions[method][scored_slice], scored_truth)
        method_error_vectors[method] = error_classes != 0
        method_counts[method] = _window_class_counts(error_classes).astype(int).tolist()
        method_error_hashes[method] = _array_sha256(error_classes, "u1")
    off_error = method_error_vectors[PRIMARY_BASELINE]
    on_error = method_error_vectors["proposed_route_a"]
    fallback_decisions = np.repeat(actions, POSTERIOR_WINDOW_DECISIONS)[scored_slice] != 0
    joint_window_counts = np.column_stack(
        (
            ~(off_error | on_error),
            off_error & ~on_error,
            ~off_error & on_error,
            off_error & on_error,
        )
    ).reshape((-1, LER_WINDOW_DECISIONS, 4)).sum(axis=1).astype(np.int16)
    fallback_window_counts = fallback_decisions.reshape((-1, LER_WINDOW_DECISIONS)).sum(axis=1).astype(np.int16)
    unnecessary_fallback_window_counts = (
        fallback_decisions & ~off_error & ~on_error
    ).reshape((-1, LER_WINDOW_DECISIONS)).sum(axis=1).astype(np.int16)

    scored_commits = [row for row in commit_rows if int(row["commit_decision"]) >= scored_start]
    transition_onset = scored_start
    first_commit = next(
        (row for row in scored_commits if int(row["commit_decision"]) >= transition_onset),
        None,
    )
    false_updates = sum(
        int(row["truth_class"])
        not in (CLASS_TO_INDEX["normal"], CLASS_TO_INDEX["smooth"])
        for row in scored_commits
    )
    result = {
        "seed": int(seed),
        "cell_id": str(cell["cell_id"]),
        "family": str(cell["family"]),
        "cell_parameters": {
            key: cell[key]
            for key in (
                "transition_rate_per_window",
                "amplitude",
                "duration_windows",
                "scored_windows",
                "nominal_preamble_windows",
            )
        },
        "input_sha256": trajectory.observed_trace_sha256,
        "truth_sha256": trajectory.truth_trace_sha256,
        "scored_start_decision": scored_start,
        "scored_decisions": len(scored_truth),
        "method_window_pauli_counts_class_order_I_Z_X_Y": method_counts,
        "method_error_trace_sha256": method_error_hashes,
        "paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong": joint_window_counts.astype(int).tolist(),
        "fallback_window_decision_counts": fallback_window_counts.astype(int).tolist(),
        "unnecessary_fallback_window_decision_counts": unnecessary_fallback_window_counts.astype(int).tolist(),
        "posterior_update_count": len(actions),
        "scored_posterior_update_count": len(actions) - scored_start // POSTERIOR_WINDOW_DECISIONS,
        "scored_fallback_update_count": int(np.sum(actions[scored_start // POSTERIOR_WINDOW_DECISIONS :] != 0)),
        "scored_tail_action_count": int(np.sum(actions[scored_start // POSTERIOR_WINDOW_DECISIONS :] == 1)),
        "scored_uncertain_action_count": int(np.sum(actions[scored_start // POSTERIOR_WINDOW_DECISIONS :] == 2)),
        "commit_count": len(commit_rows),
        "scored_commit_count": len(scored_commits),
        "window_bank_scored_commit_count": sum(row["selected_expert"] == "window_map" for row in scored_commits),
        "ewma_bank_scored_commit_count": sum(row["selected_expert"] == PRIMARY_BASELINE for row in scored_commits),
        "false_update_count": int(false_updates),
        "adaptation_lag_decisions": (
            None if first_commit is None else int(first_commit["commit_decision"]) - transition_onset
        ),
        "adaptation_lag_right_censored": first_commit is None,
        "commit_rows": commit_rows,
        "wallclock_ns": {key: int(value) for key, value in wallclock_ns.items()},
        "route_model_and_action_wallclock_ns": int(route_model_ns),
        "cache_key": cache_key,
        "cache_context_sha256": _json_sha256(context),
    }
    _validate_trajectory_result(result)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            context_json=np.asarray(json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=True)),
            result_json=np.asarray(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)),
        )
    temporary.replace(cache_path)
    return result, False


def _validate_trajectory_result(row: Mapping[str, Any]) -> None:
    if row.get("family") not in SMOOTH_FAMILIES:
        raise ValueError("formal trajectory result has an unknown smooth family")
    if int(row.get("scored_decisions", 0)) != 96 * LER_WINDOW_DECISIONS:
        raise ValueError("formal trajectory scored decision count is invalid")
    tables = row.get("method_window_pauli_counts_class_order_I_Z_X_Y")
    # JSON cache serialization is canonicalized with ``sort_keys=True``; map
    # insertion order is therefore not a scientific contract.  Exact method
    # membership is, and every downstream loop uses the frozen ``METHODS``
    # tuple rather than iterating the stored mapping.
    if not isinstance(tables, Mapping) or set(tables) != set(METHODS) or len(tables) != len(METHODS):
        raise ValueError("formal trajectory method table is incomplete")
    for method in METHODS:
        counts = np.asarray(tables[method], dtype=np.int64)
        if counts.shape != (96, 4) or np.any(counts < 0) or not np.all(np.sum(counts, axis=1) == LER_WINDOW_DECISIONS):
            raise ValueError(f"formal trajectory Pauli counts are invalid for {method}")
    paired = np.asarray(
        row["paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong"],
        dtype=np.int64,
    )
    if paired.shape != (96, 4) or np.any(paired < 0) or not np.all(np.sum(paired, axis=1) == LER_WINDOW_DECISIONS):
        raise ValueError("formal paired outcome counts are invalid")
    for field in ("fallback_window_decision_counts", "unnecessary_fallback_window_decision_counts"):
        values = np.asarray(row[field], dtype=np.int64)
        if values.shape != (96,) or np.any(values < 0) or np.any(values > LER_WINDOW_DECISIONS):
            raise ValueError(f"formal {field} is invalid")
    if int(row["scored_fallback_update_count"]) > int(row["scored_posterior_update_count"]):
        raise ValueError("formal fallback update count exceeds observed updates")
    if int(row["false_update_count"]) != 0:
        raise ValueError("smooth-only formal trace contains a false update")


def _bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != indices.shape[1] or not np.all(np.isfinite(array)):
        raise ValueError("cluster bootstrap input is invalid")
    samples = np.mean(array[indices], axis=1)
    return {
        "estimate": float(np.mean(array)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "one_sided_p_nonpositive": float((1 + np.sum(samples <= 0.0)) / (len(samples) + 1)),
    }


def _holm(rows: Sequence[Mapping[str, object]], alpha: float = 0.05) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: (float(row["raw_p"]), str(row["family"])))
    running = 0.0
    output: list[dict[str, object]] = []
    for rank, row in enumerate(ordered, start=1):
        adjusted = min(1.0, (len(ordered) - rank + 1) * float(row["raw_p"]))
        running = max(running, adjusted)
        output.append(
            {
                **row,
                "holm_rank": rank,
                "holm_adjusted_p": running,
                "reject_at_familywise_0_05": running <= alpha,
            }
        )
    return sorted(output, key=lambda row: SMOOTH_FAMILIES.index(str(row["family"])))


def _method_family_seed_ler(
    rows: Sequence[Mapping[str, Any]], method: str, family: str, seeds: Sequence[int]
) -> np.ndarray:
    output = []
    for seed in seeds:
        selected = [row for row in rows if int(row["seed"]) == seed and row["family"] == family]
        if len(selected) != 6:
            raise ValueError("formal family/seed does not contain exactly six cells")
        cell_ler = []
        for row in selected:
            counts = np.asarray(row["method_window_pauli_counts_class_order_I_Z_X_Y"][method], dtype=np.int64)
            cell_ler.append(float(np.sum(counts[:, 1:]) / np.sum(counts)))
        output.append(float(np.mean(cell_ler)))
    return np.asarray(output, dtype=np.float64)


def _summarize_method(rows: Sequence[Mapping[str, Any]], method: str, seeds: Sequence[int]) -> dict[str, object]:
    family_seed = {
        family: _method_family_seed_ler(rows, method, family, seeds)
        for family in SMOOTH_FAMILIES
    }
    aggregate_seed = np.mean(np.vstack([family_seed[family] for family in SMOOTH_FAMILIES]), axis=0)
    class_counts = np.zeros(4, dtype=np.int64)
    window_errors: list[int] = []
    worst_locator: dict[str, object] | None = None
    worst_count = -1
    seed_worst: dict[str, int] = {}
    wallclock_ns = 0
    for row in rows:
        counts = np.asarray(row["method_window_pauli_counts_class_order_I_Z_X_Y"][method], dtype=np.int64)
        class_counts += np.sum(counts, axis=0)
        local_errors = np.sum(counts[:, 1:], axis=1)
        window_errors.extend(int(value) for value in local_errors)
        local_max_index = int(np.argmax(local_errors))
        local_max = int(local_errors[local_max_index])
        if local_max > worst_count:
            worst_count = local_max
            worst_locator = {
                "seed": int(row["seed"]),
                "cell_id": row["cell_id"],
                "family": row["family"],
                "window_id": local_max_index,
                "errors": local_max,
                "denominator": LER_WINDOW_DECISIONS,
            }
        key = str(row["seed"])
        seed_worst[key] = max(seed_worst.get(key, 0), local_max)
        wallclock_ns += int(row["wallclock_ns"][method])
    total = int(np.sum(class_counts))
    errors = int(np.sum(class_counts[1:]))
    window_array = np.asarray(window_errors, dtype=np.int64)
    # Secondary Pauli intervals use the same formal-seed clusters and the same
    # equal-family/equal-cell estimator as the primary LER.  They are
    # descriptive only and never enter promotion.
    seed_pauli = np.empty((len(seeds), 4), dtype=np.float64)
    for seed_index, seed in enumerate(seeds):
        family_rates = []
        for family in SMOOTH_FAMILIES:
            selected = [
                row
                for row in rows
                if int(row["seed"]) == int(seed) and row["family"] == family
            ]
            cell_rates = []
            for row in selected:
                counts = np.asarray(
                    row["method_window_pauli_counts_class_order_I_Z_X_Y"][method],
                    dtype=np.int64,
                )
                cell_rates.append(np.sum(counts, axis=0) / np.sum(counts))
            family_rates.append(np.mean(np.vstack(cell_rates), axis=0))
        seed_pauli[seed_index] = np.mean(np.vstack(family_rates), axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(seeds), size=(BOOTSTRAP_REPLICATES, len(seeds))
    )
    bootstrap_pauli = np.mean(seed_pauli[indices], axis=1)
    bootstrap_ler = 1.0 - bootstrap_pauli[:, 0]
    pauli_intervals = {
        "p_I": [
            float(np.quantile(bootstrap_pauli[:, 0], 0.025)),
            float(np.quantile(bootstrap_pauli[:, 0], 0.975)),
        ],
        "p_Z": [
            float(np.quantile(bootstrap_pauli[:, 1], 0.025)),
            float(np.quantile(bootstrap_pauli[:, 1], 0.975)),
        ],
        "p_X": [
            float(np.quantile(bootstrap_pauli[:, 2], 0.025)),
            float(np.quantile(bootstrap_pauli[:, 2], 0.975)),
        ],
        "p_Y": [
            float(np.quantile(bootstrap_pauli[:, 3], 0.025)),
            float(np.quantile(bootstrap_pauli[:, 3], 0.975)),
        ],
        "p_L": [
            float(np.quantile(bootstrap_ler, 0.025)),
            float(np.quantile(bootstrap_ler, 0.975)),
        ],
    }
    return {
        "method_id": method,
        "deployable": method != "hidden_state_oracle",
        "average_ler_equal_family_seed": float(np.mean(aggregate_seed)),
        "pooled_decision_ler_descriptive_only": errors / total,
        "p_I": int(class_counts[0]) / total,
        "p_X": int(class_counts[2]) / total,
        "p_Y": int(class_counts[3]) / total,
        "p_Z": int(class_counts[1]) / total,
        "p_L": errors / total,
        "paired_formal_seed_cluster_ci95": pauli_intervals,
        "pauli_counts_class_order_I_Z_X_Y": [int(value) for value in class_counts],
        "decisions": total,
        "errors": errors,
        "p95_window_ler": float(np.quantile(window_array / LER_WINDOW_DECISIONS, 0.95, method="higher")),
        "global_worst_window_ler": worst_count / LER_WINDOW_DECISIONS,
        "global_worst_window": worst_locator,
        "seed_worst_window_error_counts": seed_worst,
        "family_ler_equal_seed_cell": {
            family: float(np.mean(values)) for family, values in family_seed.items()
        },
        "wallclock_seconds_total": wallclock_ns / 1.0e9,
        "wallclock_us_per_decision": wallclock_ns / 1.0e3 / total,
    }


def _analyze(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, object]:
    for row in rows:
        _validate_trajectory_result(row)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(0, len(seeds), size=(BOOTSTRAP_REPLICATES, len(seeds)))
    method_summaries = [_summarize_method(rows, method, seeds) for method in METHODS]
    family_contrasts: list[dict[str, object]] = []
    for family in SMOOTH_FAMILIES:
        baseline = _method_family_seed_ler(rows, PRIMARY_BASELINE, family, seeds)
        proposed = _method_family_seed_ler(rows, "proposed_route_a", family, seeds)
        interval = _bootstrap_mean(baseline - proposed, bootstrap_indices)
        family_contrasts.append({"family": family, **interval})
    baseline_aggregate = np.mean(
        np.vstack([_method_family_seed_ler(rows, PRIMARY_BASELINE, family, seeds) for family in SMOOTH_FAMILIES]),
        axis=0,
    )
    proposed_aggregate = np.mean(
        np.vstack([_method_family_seed_ler(rows, "proposed_route_a", family, seeds) for family in SMOOTH_FAMILIES]),
        axis=0,
    )
    static_aggregate = np.mean(
        np.vstack([_method_family_seed_ler(rows, "static_joint_map", family, seeds) for family in SMOOTH_FAMILIES]),
        axis=0,
    )
    oracle_aggregate = np.mean(
        np.vstack([_method_family_seed_ler(rows, "hidden_state_oracle", family, seeds) for family in SMOOTH_FAMILIES]),
        axis=0,
    )
    primary = _bootstrap_mean(baseline_aggregate - proposed_aggregate, bootstrap_indices)
    denominator_samples = np.mean((static_aggregate - oracle_aggregate)[bootstrap_indices], axis=1)
    denominator_ci = (
        float(np.quantile(denominator_samples, 0.025)),
        float(np.quantile(denominator_samples, 0.975)),
    )
    numerator = static_aggregate - proposed_aggregate
    denominator = static_aggregate - oracle_aggregate
    stable = denominator_ci[0] > 0.0
    ratio_samples = np.mean(numerator[bootstrap_indices], axis=1) / denominator_samples
    oracle_gap = {
        "formula": "(static_LER-proposed_LER)/(static_LER-oracle_LER)",
        "static_minus_oracle": float(np.mean(denominator)),
        "static_minus_oracle_ci95": list(denominator_ci),
        "denominator_strictly_positive": stable,
        "gap_closure": float(np.mean(numerator) / np.mean(denominator)) if stable else None,
        "gap_closure_ci95": (
            [float(np.quantile(ratio_samples, 0.025)), float(np.quantile(ratio_samples, 0.975))]
            if stable
            else None
        ),
        "flags": [] if stable else ["static_oracle_denominator_not_strictly_positive"],
    }
    total_scored_updates = sum(int(row["scored_posterior_update_count"]) for row in rows)
    total_fallback_updates = sum(int(row["scored_fallback_update_count"]) for row in rows)
    total_decisions = sum(int(row["scored_decisions"]) for row in rows)
    paired = np.sum(
        [
            np.sum(
                np.asarray(
                    row["paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong"],
                    dtype=np.int64,
                ),
                axis=0,
            )
            for row in rows
        ],
        axis=0,
    )
    lags = [int(row["adaptation_lag_decisions"]) for row in rows if row["adaptation_lag_decisions"] is not None]
    costs = derive_method_costs()
    route_hmm_macs = costs["proposed_route_a"].update_macs - costs["kalman_adaptive_map"].update_macs
    cost_ledger = {
        method: {
            "update_macs": (None if method == "hidden_state_oracle" else costs[method].update_macs),
            "private_model_state_bytes": (None if method == "hidden_state_oracle" else costs[method].private_model_state_bytes),
            "transient_workspace_bytes": (None if method == "hidden_state_oracle" else costs[method].transient_workspace_bytes),
            "role": "nondeployable_upper_bound_only" if method == "hidden_state_oracle" else "deployable_comparator",
        }
        for method in METHODS
    }
    cost_ledger["proposed_route_a"].update(
        {
            "v4_registered_full_update_macs": (
                costs["window_map"].update_macs
                + costs["ewma_adaptive_map"].update_macs
                + route_hmm_macs
                + 2 * 14
            ),
            "cap": 8192,
            "source_to_action_latency_cycles": 6,
            "board_measured": False,
        }
    )
    holm = _holm(
        [{"family": row["family"], "raw_p": row["one_sided_p_nonpositive"]} for row in family_contrasts]
    )
    return {
        "method_summaries": method_summaries,
        "primary_contrast": {
            "contrast": "ewma_adaptive_map_LER_minus_proposed_route_a_LER",
            **primary,
            "passes_95_lcb_strictly_greater_than_zero": primary["ci95_low"] > 0.0,
        },
        "per_family_contrasts": family_contrasts,
        "holm_smooth_family_superiority": holm,
        "oracle_gap_closure": oracle_gap,
        "action_and_update_metrics": {
            "scored_posterior_updates": total_scored_updates,
            "fallback_updates": total_fallback_updates,
            "fallback_rate": total_fallback_updates / total_scored_updates,
            "false_updates": sum(int(row["false_update_count"]) for row in rows),
            "commits": sum(int(row["scored_commit_count"]) for row in rows),
            "window_bank_commits": sum(int(row["window_bank_scored_commit_count"]) for row in rows),
            "ewma_bank_commits": sum(int(row["ewma_bank_scored_commit_count"]) for row in rows),
            "avoided_errors": int(paired[1]),
            "induced_errors": int(paired[2]),
            "unnecessary_fallback_decisions": sum(sum(row["unnecessary_fallback_window_decision_counts"]) for row in rows),
            "unnecessary_fallback_rate": sum(sum(row["unnecessary_fallback_window_decision_counts"]) for row in rows) / total_decisions,
            "adaptation_lag_decisions": {
                "observed": len(lags),
                "right_censored": sum(bool(row["adaptation_lag_right_censored"]) for row in rows),
                "mean": float(np.mean(lags)) if lags else None,
                "p95_higher": float(np.quantile(lags, 0.95, method="higher")) if lags else None,
                "max": max(lags) if lags else None,
            },
        },
        "cost_ledger": cost_ledger,
        "bootstrap_contract": {
            "independent_cluster": "formal seed",
            "cluster_count": len(seeds),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "equal_family_weighting": True,
        },
    }


def recompute_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    rows = report["trajectory_results"]
    seeds = report["formal_design"]["seeds"]
    analysis = _analyze(rows, seeds)
    stored_analysis_matches = _json_sha256(analysis) == report.get("analysis_sha256") and analysis == report.get("analysis")
    expected_pairs = {
        (int(seed), str(cell["cell_id"]))
        for seed in seeds
        for cell in report["formal_design"]["cells"]
    }
    actual_pairs = {(int(row["seed"]), str(row["cell_id"])) for row in rows}
    source = ROOT / report["source_data_binding"]["path"]
    source_bindings_current = all(
        (ROOT / row["path"]).is_file()
        and _sha256(ROOT / row["path"]) == row["sha256"]
        for row in report["source_bindings"]
    )
    return {
        "G01_parent_preregistration_v4_lock_and_source_hashes_match": report["parent_bindings"]["preregistration_sha256"] == _sha256(PREREG_ARTIFACT) and report["parent_bindings"]["threshold_lock_artifact_sha256"] == _sha256(DEFAULT_LOCK) and report["parent_bindings"]["threshold_lock_sha256"] == "9347edb270bbeb3f50d8bd8aceaeefd8003e118f1e88712dd5265519bb0f67aa" and source_bindings_current,
        "G02_primary_baseline_remains_pilot_locked_ewma": report["primary_baseline"] == PRIMARY_BASELINE and report["formal_baseline_reselection"] is False,
        "G03_exact_24_seed_by_24_cell_matrix_executed": len(rows) == 576 and actual_pairs == expected_pairs,
        "G04_all_four_smooth_families_and_six_cells_each_present": set(row["family"] for row in rows) == set(SMOOTH_FAMILIES) and all(sum(row["family"] == family for row in rows) == 144 for family in SMOOTH_FAMILIES),
        "G05_every_trajectory_has_all_methods_and_96_complete_windows": all(set(row["method_window_pauli_counts_class_order_I_Z_X_Y"]) == set(METHODS) and len(row["method_window_pauli_counts_class_order_I_Z_X_Y"]) == len(METHODS) and int(row["scored_decisions"]) == 96 * 512 for row in rows),
        "G06_oracle_is_separate_non_deployable_evaluation_lane": report["oracle_contract"]["deployable"] is False and report["oracle_contract"]["online_route_input"] is False,
        "G07_analysis_recomputes_exactly_from_raw_trajectory_counts": stored_analysis_matches,
        "G08_paired_seed_cluster_bootstrap_and_equal_family_weights_frozen": analysis["bootstrap_contract"] == {"independent_cluster": "formal seed", "cluster_count": 24, "replicates": 20000, "seed": 202607176999, "equal_family_weighting": True},
        "G09_source_data_is_complete_and_hash_bound": source.is_file() and _sha256(source) == report["source_data_binding"]["sha256"] and int(report["source_data_binding"]["row_count"]) == 498240,
        "G10_no_false_update_or_undefined_metric_in_smooth_matrix": analysis["action_and_update_metrics"]["false_updates"] == 0 and all(np.isfinite(float(row["analysis_value"])) for row in report["finite_metric_audit"]),
        "G11_cache_context_binds_every_trace_to_lock_and_algorithm": len(report["cache_audit"]["cache_keys"]) == 576 and len(set(report["cache_audit"]["cache_keys"])) == 576 and report["cache_audit"]["algorithm_contract_sha256"] == ALGORITHM_CONTRACT_SHA256,
        "G12_semantic_mutations_all_rejected": len(report.get("semantic_mutations", [])) == 6 and all(row["rejected"] for row in report.get("semantic_mutations", [])),
    }


def _validate_core(report: Mapping[str, Any], *, verify_source: bool = True) -> None:
    rows = report.get("trajectory_results")
    if not isinstance(rows, list):
        raise ValueError("formal raw trajectory results are absent")
    for row in rows:
        _validate_trajectory_result(row)
    analysis = _analyze(rows, report["formal_design"]["seeds"])
    if report.get("analysis") != analysis or report.get("analysis_sha256") != _json_sha256(analysis):
        raise ValueError("formal analysis does not recompute from raw trajectory results")
    if report["primary_baseline"] != PRIMARY_BASELINE or report["formal_baseline_reselection"] is not False:
        raise ValueError("formal primary baseline was reselected")
    if report["parent_bindings"]["threshold_lock_sha256"] != "9347edb270bbeb3f50d8bd8aceaeefd8003e118f1e88712dd5265519bb0f67aa":
        raise ValueError("formal threshold lock was replaced")
    if len(rows) != 576:
        raise ValueError("formal trajectory matrix is incomplete")
    if report["oracle_contract"]["deployable"] is not False or report["oracle_contract"]["online_route_input"] is not False:
        raise ValueError("hidden-state oracle entered a deployable path")
    for binding in report["source_bindings"]:
        path = ROOT / binding["path"]
        if not path.is_file() or _sha256(path) != binding["sha256"]:
            raise ValueError(f"formal implementation source binding is stale: {binding['path']}")
    if verify_source:
        source = ROOT / report["source_data_binding"]["path"]
        if not source.is_file() or _sha256(source) != report["source_data_binding"]["sha256"]:
            raise ValueError("formal Source Data binding is stale")


def _semantic_mutations(report: Mapping[str, Any]) -> list[dict[str, object]]:
    cases = (
        ("replace_lock", ("parent_bindings", "threshold_lock_sha256"), "0" * 64),
        ("reselect_baseline", ("primary_baseline",), "kalman_adaptive_map"),
        ("claim_reselection", ("formal_baseline_reselection",), True),
        ("drop_trajectory", ("trajectory_results",), report["trajectory_results"][:-1]),
        ("mutate_primary_result", ("analysis", "primary_contrast", "ci95_low"), 1.0),
        ("promote_oracle", ("oracle_contract", "deployable"), True),
    )
    rows = []
    for mutation, path, value in cases:
        mutated = deepcopy(report)
        target: Any = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        try:
            _validate_core(mutated, verify_source=False)
        except (TypeError, ValueError, KeyError) as exc:
            rows.append({"mutation": mutation, "rejected": True, "detail": str(exc)})
        else:
            rows.append({"mutation": mutation, "rejected": False, "detail": "accepted"})
    return rows


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in report["trajectory_results"]:
        for method in METHODS:
            counts = row["method_window_pauli_counts_class_order_I_Z_X_Y"][method]
            for window_id, values in enumerate(counts):
                output.append(
                    {
                        "row_type": "formal_window",
                        "seed": row["seed"],
                        "cell_id": row["cell_id"],
                        "family": row["family"],
                        "method_id": method,
                        "window_id": window_id,
                        "n_I": values[0],
                        "n_X": values[2],
                        "n_Y": values[3],
                        "n_Z": values[1],
                        "denominator": LER_WINDOW_DECISIONS,
                        "input_sha256": row["input_sha256"],
                        "detail": row["method_error_trace_sha256"][method],
                    }
                )
        output.append(
            {
                "row_type": "trajectory_action",
                "seed": row["seed"],
                "cell_id": row["cell_id"],
                "family": row["family"],
                "method_id": "proposed_route_a",
                "window_id": -1,
                "n_I": row["scored_fallback_update_count"],
                "n_X": row["window_bank_scored_commit_count"],
                "n_Y": row["ewma_bank_scored_commit_count"],
                "n_Z": row["false_update_count"],
                "denominator": row["scored_posterior_update_count"],
                "input_sha256": row["input_sha256"],
                "detail": f"lag={row['adaptation_lag_decisions']};censored={row['adaptation_lag_right_censored']}",
            }
        )
        paired_rows = row[
            "paired_ewma_proposed_window_counts_order_both_correct_avoided_induced_both_wrong"
        ]
        fallback_rows = row["fallback_window_decision_counts"]
        unnecessary_rows = row["unnecessary_fallback_window_decision_counts"]
        for window_id, (paired, fallback, unnecessary) in enumerate(
            zip(paired_rows, fallback_rows, unnecessary_rows, strict=True)
        ):
            output.append(
                {
                    "row_type": "paired_outcome_window",
                    "seed": row["seed"],
                    "cell_id": row["cell_id"],
                    "family": row["family"],
                    "method_id": "ewma_adaptive_map_vs_proposed_route_a",
                    "window_id": window_id,
                    "n_I": paired[0],
                    "n_X": paired[1],
                    "n_Y": paired[2],
                    "n_Z": paired[3],
                    "denominator": LER_WINDOW_DECISIONS,
                    "input_sha256": row["input_sha256"],
                    "detail": "both_correct|avoided|induced|both_wrong",
                }
            )
            output.append(
                {
                    "row_type": "action_window",
                    "seed": row["seed"],
                    "cell_id": row["cell_id"],
                    "family": row["family"],
                    "method_id": "proposed_route_a",
                    "window_id": window_id,
                    "n_I": fallback,
                    "n_X": unnecessary,
                    "n_Y": 0,
                    "n_Z": 0,
                    "denominator": LER_WINDOW_DECISIONS,
                    "input_sha256": row["input_sha256"],
                    "detail": "fallback_decisions|unnecessary_fallback_decisions",
                }
            )
    return output


def build_report(
    *,
    lock_path: Path = DEFAULT_LOCK,
    access_ledger: Path = DEFAULT_ACCESS_LEDGER,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    parents = _load_parents(lock_path)
    ledger = _record_formal_access(parents, access_ledger)
    cells, seeds = _formal_cells_and_seeds()
    calibration = materialize_qualification_trace()[0].calibration_residuals
    cache_dir.mkdir(parents=True, exist_ok=True)
    trajectory_results: list[dict[str, object]] = []
    cache_hits = 0
    cache_misses = 0
    total = len(cells) * len(seeds)
    completed = 0
    for seed in seeds:
        for cell in cells:
            row, hit = _run_trajectory(cell, seed, parents, calibration, cache_dir)
            trajectory_results.append(row)
            cache_hits += int(hit)
            cache_misses += int(not hit)
            completed += 1
            if completed % 12 == 0 or completed == total:
                print(
                    json.dumps(
                        {
                            "task": TASK_ID,
                            "completed": completed,
                            "total": total,
                            "cache_hits": cache_hits,
                            "cache_misses": cache_misses,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    analysis = _analyze(trajectory_results, seeds)
    finite_values = []
    for method in analysis["method_summaries"]:
        for key in ("average_ler_equal_family_seed", "p_X", "p_Y", "p_Z", "p_L", "p95_window_ler", "global_worst_window_ler", "wallclock_us_per_decision"):
            finite_values.append({"metric": f"{method['method_id']}:{key}", "analysis_value": method[key]})
    lock = parents["threshold_lock"]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "protocol_id": PROTOCOL_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_access_ledger": ledger,
        "parent_bindings": {
            "preregistration_path": PREREG_ARTIFACT.relative_to(ROOT).as_posix(),
            "preregistration_sha256": _sha256(PREREG_ARTIFACT),
            "preregistration_protocol_sha256": parents["preregistration"]["protocol_sha256"],
            "threshold_lock_path": lock_path.relative_to(ROOT).as_posix(),
            "threshold_lock_artifact_sha256": _sha256(lock_path),
            "threshold_lock_sha256": lock["threshold_lock"]["lock_sha256"],
            "posterior_model_sha256": lock["posterior_model_sha256"],
            "event_model_sha256": lock["event_model_sha256"],
            "baseline_selection_sha256": lock["pilot_baseline_qualification"]["selection_sha256"],
        },
        "primary_baseline": PRIMARY_BASELINE,
        "formal_baseline_reselection": False,
        "formal_design": {"split_id": "formal_evaluation", "families": list(SMOOTH_FAMILIES), "cells": list(cells), "seeds": list(seeds)},
        "method_contract": {"methods": list(METHODS), "deployable_methods": list(DEPLOYABLE_METHODS), "pauli_class_encoding": {"0": "I", "1": "Z", "2": "X", "3": "Y"}},
        "oracle_contract": {"method_id": "hidden_state_oracle", "deployable": False, "online_route_input": False, "truth_scope": "per-decision hidden mean/covariance reconstructed only in evaluation lane", "outlier_scope": "smooth families have p_outlier=0", "same_quantized_syndrome": True},
        "trajectory_results": trajectory_results,
        "analysis": analysis,
        "analysis_sha256": _json_sha256(analysis),
        "finite_metric_audit": finite_values,
        "cache_audit": {
            "directory": cache_dir.relative_to(ROOT).as_posix(),
            "schema_version": _algorithm_contract()["schema_version"],
            "algorithm_contract_sha256": ALGORITHM_CONTRACT_SHA256,
            "hits": cache_hits,
            "misses": cache_misses,
            "cache_keys": [str(row["cache_key"]) for row in trajectory_results],
            "all_entries_hash_bind_trace_lock_and_algorithm": True,
        },
        "source_bindings": [
            {"path": relative, "sha256": _sha256(ROOT / relative)}
            for relative in (
                "cnn_fpga/benchmark/route_a_smooth_formal.py",
                "cnn_fpga/benchmark/route_a_posterior_calibration.py",
                "cnn_fpga/benchmark/route_a_preregistration.py",
                "cnn_fpga/decoder/route_a_regime_posterior.py",
                "cnn_fpga/decoder/periodic_adaptive_map.py",
                "physics/ideal_gkp_decoder.py",
            )
        ],
        "source_data_binding": {"path": DEFAULT_SOURCE_DATA.relative_to(ROOT).as_posix(), "sha256": None, "row_count": 0},
        "semantic_mutations": [],
        "claim_boundary": {
            "admitted_if_primary_passes": "untouched formal smooth aggregate advantage over the pilot-locked EWMA baseline under this simulator/protocol",
            "not_admitted": ["abrupt/OOD safety", "physical break-even", "Puviani NMF lifetime superiority", "measured FPGA latency or power"],
        },
    }
    return report


def write_report(
    artifact: Path = DEFAULT_ARTIFACT,
    source_data: Path = DEFAULT_SOURCE_DATA,
    *,
    lock_path: Path = DEFAULT_LOCK,
    access_ledger: Path = DEFAULT_ACCESS_LEDGER,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> dict[str, Any]:
    report = build_report(lock_path=lock_path, access_ledger=access_ledger, cache_dir=cache_dir)
    source_rows = _source_rows(report)
    source_data.parent.mkdir(parents=True, exist_ok=True)
    with source_data.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ("row_type", "seed", "cell_id", "family", "method_id", "window_id", "n_I", "n_X", "n_Y", "n_Z", "denominator", "input_sha256", "detail")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)
    report["source_data_binding"] = {
        "path": source_data.relative_to(ROOT).as_posix(),
        "sha256": _sha256(source_data),
        "row_count": len(source_rows),
    }
    report["semantic_mutations"] = _semantic_mutations(report)
    gates = recompute_gates(report)
    report["gate_summary"] = {"passed": sum(gates.values()), "failed": sum(not value for value in gates.values()), "gates": gates}
    primary_pass = bool(report["analysis"]["primary_contrast"]["passes_95_lcb_strictly_greater_than_zero"])
    report["status"] = "PASS" if all(gates.values()) else "FAIL"
    report["verdict"] = (
        "PASS_VALID_FORMAL_RESULT_PRIMARY_SMOOTH_GATE_PASSED"
        if all(gates.values()) and primary_pass
        else "PASS_VALID_FORMAL_RESULT_PRIMARY_SMOOTH_GATE_FAILED"
        if all(gates.values())
        else "FAIL_INVALID_FORMAL_RESULT"
    )
    _validate_core(report)
    if not all(gates.values()):
        raise ValueError(f"T6.7.1 formal evidence gates failed: {[key for key, value in gates.items() if not value]}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    _validate_core(report)
    gates = recompute_gates(report)
    if report.get("gate_summary", {}).get("gates") != gates or not all(gates.values()):
        raise ValueError("stored T6.7.1 evidence gates do not recompute")
    if len(report.get("semantic_mutations", [])) != 6 or not all(row["rejected"] for row in report["semantic_mutations"]):
        raise ValueError("T6.7.1 semantic mutation audit is incomplete")
    primary_pass = report["analysis"]["primary_contrast"]["passes_95_lcb_strictly_greater_than_zero"]
    expected = "PASS_VALID_FORMAL_RESULT_PRIMARY_SMOOTH_GATE_PASSED" if primary_pass else "PASS_VALID_FORMAL_RESULT_PRIMARY_SMOOTH_GATE_FAILED"
    if report.get("status") != "PASS" or report.get("verdict") != expected:
        raise ValueError("T6.7.1 verdict does not match the recomputed primary gate")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--access-ledger", type=Path, default=DEFAULT_ACCESS_LEDGER)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        report = json.loads(args.artifact.read_text(encoding="utf-8"))
        verify_report(report)
    else:
        report = write_report(
            args.artifact,
            args.source_data,
            lock_path=args.lock,
            access_ledger=args.access_ledger,
            cache_dir=args.cache_dir,
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report["verdict"],
                "primary": report["analysis"]["primary_contrast"],
                "gates": report["gate_summary"],
                "cache": {key: report["cache_audit"][key] for key in ("hits", "misses")},
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
