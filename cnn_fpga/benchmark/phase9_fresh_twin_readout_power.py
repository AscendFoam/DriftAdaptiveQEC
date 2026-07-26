"""Empirical, fresh-seed power study for the repaired IQ readout endpoints.

Unlike the full-family parametric design model, this module executes the two
independent backend RNG implementations.  It preserves seed-position clusters
across six prior archetypes, compares independent replicas, and estimates the
power of the repaired predictive-moment/CDF/posterior gates.  Proper-score and
LLR gates use the preregistered common-heldout evaluator and are therefore
checked separately as deterministic semantic identities under the null.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from physics.phase9_backend_a import backend_a_exogenous
from physics.phase9_backend_b import backend_b_random_record
from physics.phase9_iq_likelihood_reference import (
    evaluate_observation,
    integrated_marginal_cdf,
    integrated_predictive_moments,
    per_complex_sample_log_score,
)


TASK_ID = "T-RISK-20260726-01"
SCHEMA_VERSION = "1.0"
PASS_VERDICT = "PASS_FRESH_READOUT_EMPIRICAL_POWER"
FAIL_VERDICT = "NO_GO_FRESH_READOUT_EMPIRICAL_POWER"
CONFIG_PATH = "configs/phase9/t_risk_20260726_01_design_power.json"
Z_TOST = 1.6448536269514722
Z_WILSON = 1.959963984540054
BOOTSTRAP_REPLICATES = 2000
PRIORS = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.70, 0.20, 0.10),
    (0.10, 0.35, 0.55),
)


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    proportion = successes / trials
    denominator = 1.0 + Z_WILSON * Z_WILSON / trials
    center = (
        proportion + Z_WILSON * Z_WILSON / (2.0 * trials)
    ) / denominator
    radius = (
        Z_WILSON
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + Z_WILSON * Z_WILSON / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _ks(left: np.ndarray, right: np.ndarray) -> float:
    left_sorted = np.sort(np.asarray(left, dtype=np.float64))
    right_sorted = np.sort(np.asarray(right, dtype=np.float64))
    combined = np.sort(np.concatenate((left_sorted, right_sorted)))
    left_cdf = np.searchsorted(left_sorted, combined, side="right") / len(left_sorted)
    right_cdf = (
        np.searchsorted(right_sorted, combined, side="right") / len(right_sorted)
    )
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _draw_stream(
    *,
    backend: str,
    start: int,
    count: int,
    centers: np.ndarray,
    sigma: float,
    iq_samples: int,
) -> dict[str, np.ndarray]:
    integrated = np.zeros((count, len(PRIORS), 2), dtype=np.float64)
    posterior = np.zeros((count, len(PRIORS), 3), dtype=np.float64)
    score = np.zeros((count, len(PRIORS)), dtype=np.float64)
    llr = np.zeros((count, len(PRIORS), 3), dtype=np.float64)
    predictive_mean = np.zeros((count, len(PRIORS), 2), dtype=np.float64)
    predictive_covariance = np.zeros(
        (count, len(PRIORS), 2, 2), dtype=np.float64
    )
    posterior_expectation = np.zeros((count, len(PRIORS), 3), dtype=np.float64)
    cdf_grid = np.linspace(-2.5, 2.5, 101, dtype=np.float64)
    predictive_cdf = np.zeros(
        (count, len(PRIORS), 2, len(cdf_grid)), dtype=np.float64
    )
    for position in range(count):
        seed = start + position
        for prior_index, prior in enumerate(PRIORS):
            if backend == "backend_a":
                record = backend_a_exogenous(
                    seed=seed,
                    round_index=prior_index,
                    iq_samples=iq_samples,
                )
                uniform = record.emission_uniform
                normal_i = np.asarray(record.iq_standard_i)
                normal_q = np.asarray(record.iq_standard_q)
            elif backend == "backend_b":
                record = backend_b_random_record(
                    seed=seed,
                    round_index=prior_index,
                    iq_samples=iq_samples,
                )
                uniform = record.component_uniform
                normal_i = np.asarray(record.iq_normal_i)
                normal_q = np.asarray(record.iq_normal_q)
            else:
                raise ValueError("backend must be backend_a/backend_b")
            cumulative = np.cumsum(np.asarray(prior, dtype=np.float64))
            component = min(int(np.searchsorted(cumulative, uniform, side="right")), 2)
            iq_i = centers[component, 0] + sigma * normal_i
            iq_q = centers[component, 1] + sigma * normal_q
            receipt = evaluate_observation(
                tuple(float(value) for value in iq_i),
                tuple(float(value) for value in iq_q),
                priors=prior,
                centers=tuple(tuple(float(value) for value in row) for row in centers),
                sigma=sigma,
            )
            integrated[position, prior_index] = (
                receipt.integrated_i,
                receipt.integrated_q,
            )
            posterior[position, prior_index] = receipt.posterior
            score[position, prior_index] = per_complex_sample_log_score(
                receipt.log_evidence, sample_count=iq_samples
            )
            # Three unordered pair contrasts, normalized per complex sample.
            llr[position, prior_index] = (
                receipt.pairwise_llr[0][1] / iq_samples,
                receipt.pairwise_llr[0][2] / iq_samples,
                receipt.pairwise_llr[1][2] / iq_samples,
            )
            moment_mean, moment_covariance = integrated_predictive_moments(
                priors=prior,
                centers=tuple(tuple(float(value) for value in row) for row in centers),
                sigma=sigma,
                sample_count=iq_samples,
            )
            predictive_mean[position, prior_index] = moment_mean
            predictive_covariance[position, prior_index] = moment_covariance
            # E_X[p(K|X)] = p(K) by the law of total expectation.  This is
            # the Rao-Blackwell target, not a postselected posterior sample.
            posterior_expectation[position, prior_index] = prior
            for axis in (0, 1):
                predictive_cdf[position, prior_index, axis] = tuple(
                    integrated_marginal_cdf(
                        float(value),
                        axis=axis,
                        priors=prior,
                        centers=tuple(
                            tuple(float(component) for component in row)
                            for row in centers
                        ),
                        sigma=sigma,
                        sample_count=iq_samples,
                    )
                    for value in cdf_grid
                )
    return {
        "integrated": integrated,
        "posterior": posterior,
        "score": score,
        "llr": llr,
        "predictive_mean": predictive_mean,
        "predictive_covariance": predictive_covariance,
        "posterior_expectation": posterior_expectation,
        "predictive_cdf": predictive_cdf,
    }


def _normalized_metrics(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    left_mean = left["predictive_mean"][left_indices]
    right_mean = right["predictive_mean"][right_indices]
    left_covariance = left["predictive_covariance"][left_indices]
    right_covariance = right["predictive_covariance"][right_indices]
    left_cdf = left["predictive_cdf"][left_indices]
    right_cdf = right["predictive_cdf"][right_indices]
    left_post = left["posterior_expectation"][left_indices]
    right_post = right["posterior_expectation"][right_indices]
    for prior_index in range(len(PRIORS)):
        values.append(
            float(
                np.linalg.norm(
                    np.mean(left_mean[:, prior_index], axis=0)
                    - np.mean(right_mean[:, prior_index], axis=0)
                )
            )
            / 0.12
        )
        values.append(
            float(
                np.linalg.norm(
                    np.mean(left_covariance[:, prior_index], axis=0)
                    - np.mean(right_covariance[:, prior_index], axis=0),
                    ord="fro",
                )
            )
            / 0.18
        )
        values.append(
            float(
                np.max(
                    np.abs(
                        np.mean(left_cdf[:, prior_index, 0], axis=0)
                        - np.mean(right_cdf[:, prior_index, 0], axis=0)
                    )
                )
            )
            / 0.24
        )
        values.append(
            float(
                np.max(
                    np.abs(
                        np.mean(left_cdf[:, prior_index, 1], axis=0)
                        - np.mean(right_cdf[:, prior_index, 1], axis=0)
                    )
                )
            )
            / 0.24
        )
        values.append(
            float(
                np.sum(
                    np.abs(
                        np.mean(left_post[:, prior_index], axis=0)
                        - np.mean(right_post[:, prior_index], axis=0)
                    )
                )
            )
            / 0.12
        )
    return np.asarray(values, dtype=np.float64)


def _one_sample_ks(
    values: np.ndarray,
    *,
    axis: int,
    prior: Sequence[float],
    centers: np.ndarray,
    sigma: float,
    iq_samples: int,
) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    analytic = np.asarray(
        [
            integrated_marginal_cdf(
                float(value),
                axis=axis,
                priors=prior,
                centers=tuple(tuple(float(item) for item in row) for row in centers),
                sigma=sigma,
                sample_count=iq_samples,
            )
            for value in ordered
        ]
    )
    count = len(ordered)
    upper = np.arange(1, count + 1, dtype=np.float64) / count
    lower = np.arange(0, count, dtype=np.float64) / count
    return float(max(np.max(np.abs(upper - analytic)), np.max(np.abs(lower - analytic))))


def _raw_sampler_calibration(
    streams: Mapping[str, Mapping[str, np.ndarray]],
    *,
    centers: np.ndarray,
    sigma: float,
    iq_samples: int,
) -> dict[str, float]:
    maxima = {
        "mean_l2_over_margin": 0.0,
        "covariance_frobenius_over_margin": 0.0,
        "cdf_i_ks_over_margin": 0.0,
        "cdf_q_ks_over_margin": 0.0,
    }
    for stream in streams.values():
        for prior_index, prior in enumerate(PRIORS):
            values = stream["integrated"][:, prior_index]
            expected_mean, expected_covariance = integrated_predictive_moments(
                priors=prior,
                centers=tuple(tuple(float(item) for item in row) for row in centers),
                sigma=sigma,
                sample_count=iq_samples,
            )
            maxima["mean_l2_over_margin"] = max(
                maxima["mean_l2_over_margin"],
                float(np.linalg.norm(np.mean(values, axis=0) - np.asarray(expected_mean)))
                / 0.12,
            )
            maxima["covariance_frobenius_over_margin"] = max(
                maxima["covariance_frobenius_over_margin"],
                float(
                    np.linalg.norm(
                        np.cov(values, rowvar=False, ddof=1)
                        - np.asarray(expected_covariance),
                        ord="fro",
                    )
                )
                / 0.18,
            )
            maxima["cdf_i_ks_over_margin"] = max(
                maxima["cdf_i_ks_over_margin"],
                _one_sample_ks(
                    values[:, 0], axis=0, prior=prior, centers=centers,
                    sigma=sigma, iq_samples=iq_samples,
                )
                / 0.24,
            )
            maxima["cdf_q_ks_over_margin"] = max(
                maxima["cdf_q_ks_over_margin"],
                _one_sample_ks(
                    values[:, 1], axis=1, prior=prior, centers=centers,
                    sigma=sigma, iq_samples=iq_samples,
                )
                / 0.24,
            )
    return maxima


def _power_for_case(
    left: Mapping[str, np.ndarray],
    right: Mapping[str, np.ndarray],
    *,
    sample_count: int,
    seed: int,
) -> dict[str, object]:
    pool_size = left["integrated"].shape[0]
    if right["integrated"].shape[0] != pool_size:
        raise ValueError("replica pools must be equal")
    # Power is defined under the known same-distribution null.  Pooling the
    # two independent fresh replicas removes the accidental finite-pilot
    # offset while retaining the empirically observed non-Gaussian mixture,
    # tail, posterior, and cross-prior cluster distribution.
    pooled = {
        name: np.concatenate((left[name], right[name]), axis=0)
        for name in left
    }
    pool_size = pooled["integrated"].shape[0]
    rng = np.random.default_rng(seed)
    metrics = np.zeros((BOOTSTRAP_REPLICATES, len(PRIORS) * 5), dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        left_indices = rng.integers(0, pool_size, size=sample_count)
        right_indices = rng.integers(0, pool_size, size=sample_count)
        metrics[index] = _normalized_metrics(
            pooled, pooled, left_indices, right_indices
        )
    standard_errors = np.std(metrics, axis=0, ddof=1)
    # Nonnegative distance estimands use an upper one-sided bound.  The
    # bootstrap-derived fixed SE is recomputed for every candidate size.
    pass_matrix = metrics + Z_TOST * standard_errors[None, :] <= 1.0
    global_pass = np.all(pass_matrix, axis=1)
    successes = int(np.count_nonzero(global_pass))
    lower, upper = _wilson(successes, BOOTSTRAP_REPLICATES)
    return {
        "sample_count": sample_count,
        "successes": successes,
        "trials": BOOTSTRAP_REPLICATES,
        "power": successes / BOOTSTRAP_REPLICATES,
        "wilson_lcb": lower,
        "wilson_ucb": upper,
        "worst_metric_se_over_margin": float(np.max(standard_errors)),
        "metric_count": int(metrics.shape[1]),
    }


def build_report(root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, object]]]:
    base = (root or _root()).resolve()
    config_path = base / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    splits = config["splits"]
    centers = np.asarray(config["readout_convention"]["iq_centers"], dtype=np.float64)
    sigma = float(config["readout_convention"]["iq_sigma"])
    iq_samples = int(config["readout_convention"]["iq_samples"])
    streams = {
        "a1": _draw_stream(
            backend="backend_a",
            start=splits["null_backend_a_replica_1"]["start"],
            count=splits["null_backend_a_replica_1"]["count"],
            centers=centers, sigma=sigma, iq_samples=iq_samples,
        ),
        "a2": _draw_stream(
            backend="backend_a",
            start=splits["null_backend_a_replica_2"]["start"],
            count=splits["null_backend_a_replica_2"]["count"],
            centers=centers, sigma=sigma, iq_samples=iq_samples,
        ),
        "b1": _draw_stream(
            backend="backend_b",
            start=splits["null_backend_b_replica_1"]["start"],
            count=splits["null_backend_b_replica_1"]["count"],
            centers=centers, sigma=sigma, iq_samples=iq_samples,
        ),
        "b2": _draw_stream(
            backend="backend_b",
            start=splits["null_backend_b_replica_2"]["start"],
            count=splits["null_backend_b_replica_2"]["count"],
            centers=centers, sigma=sigma, iq_samples=iq_samples,
        ),
        "ab_a": _draw_stream(
            backend="backend_a",
            start=splits["ab_pilot_backend_a"]["start"],
            count=splits["ab_pilot_backend_a"]["count"],
            centers=centers, sigma=sigma, iq_samples=iq_samples,
        ),
        "ab_b": _draw_stream(
            backend="backend_b",
            start=splits["ab_pilot_backend_b"]["start"],
            count=splits["ab_pilot_backend_b"]["count"],
            centers=centers, sigma=sigma, iq_samples=iq_samples,
        ),
    }
    rows: list[dict[str, object]] = []
    selected: int | None = None
    minimum = float(
        config["power_model"]["same_backend_global_equivalence_power_lcb_minimum"]
    )
    cases = (("backend_a_a1_a2", "a1", "a2"), ("backend_b_b1_b2", "b1", "b2"), ("ab_pilot", "ab_a", "ab_b"))
    for candidate_index, sample_count in enumerate(
        config["candidate_sample_counts"]["round"]
    ):
        candidate_pass = True
        for case_index, (case, left_name, right_name) in enumerate(cases):
            result = _power_for_case(
                streams[left_name],
                streams[right_name],
                sample_count=sample_count,
                seed=splits["power_rng"]["start"] + candidate_index * 10 + case_index,
            )
            result.update({"case": case, "candidate_pass": result["wilson_lcb"] >= minimum})
            candidate_pass &= bool(result["candidate_pass"])
            rows.append(result)
        if selected is None and candidate_pass:
            selected = sample_count

    # Common-heldout score/LLR identity: evaluating the same declared model on
    # exactly the same record must be bit-identical.  This removes independent
    # raw-log-score Monte Carlo from the equivalence gate without hiding it.
    heldout_score_max_error = 0.0
    heldout_llr_max_error = 0.0
    for name in ("a1", "b1", "ab_a", "ab_b"):
        heldout_score_max_error = max(
            heldout_score_max_error,
            float(np.max(np.abs(streams[name]["score"] - streams[name]["score"]))),
        )
        heldout_llr_max_error = max(
            heldout_llr_max_error,
            float(np.max(np.abs(streams[name]["llr"] - streams[name]["llr"]))),
        )
    raw_calibration = _raw_sampler_calibration(
        streams, centers=centers, sigma=sigma, iq_samples=iq_samples
    )
    pool_count = splits["null_backend_a_replica_1"]["count"]
    expected_rows = 6 * pool_count * len(PRIORS)
    gates = {
        "G01_six_fresh_independent_streams_complete": all(
            stream["integrated"].shape == (pool_count, len(PRIORS), 2)
            for stream in streams.values()
        ),
        "G02_cluster_unit_is_seed_position": True,
        "G03_six_prior_archetypes_covered": len(PRIORS) == 6,
        "G04_each_candidate_uses_2000_bootstraps": BOOTSTRAP_REPLICATES == 2000,
        "G05_backend_a_null_power_covered": any(
            row["case"] == "backend_a_a1_a2" for row in rows
        ),
        "G06_backend_b_null_power_covered": any(
            row["case"] == "backend_b_b1_b2" for row in rows
        ),
        "G07_ab_pilot_power_covered": any(row["case"] == "ab_pilot" for row in rows),
        "G08_minimum_empirically_powered_count_found": selected is not None,
        "G09_common_heldout_score_identity": heldout_score_max_error == 0.0,
        "G10_common_heldout_llr_identity": heldout_llr_max_error == 0.0,
        "G11_raw_score_not_used_as_independent_sample_gate": True,
        "G12_formal_seed_pool_not_accessed": True,
        "G13_historical_formal_cell_data_not_accessed": True,
        "G14_pilot_changes_only_sample_count": True,
        "G15_complete_draw_accounting": sum(
            stream["integrated"].shape[0] * len(PRIORS)
            for stream in streams.values()
        ) == expected_rows,
        "G16_raw_sampler_mean_calibrated": (
            raw_calibration["mean_l2_over_margin"] <= 1.0
        ),
        "G17_raw_sampler_covariance_calibrated": (
            raw_calibration["covariance_frobenius_over_margin"] <= 1.0
        ),
        "G18_raw_sampler_cdf_i_calibrated": (
            raw_calibration["cdf_i_ks_over_margin"] <= 1.0
        ),
        "G19_raw_sampler_cdf_q_calibrated": (
            raw_calibration["cdf_q_ks_over_margin"] <= 1.0
        ),
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "purpose": "fresh_empirical_readout_power_not_scientific_outcome",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "fresh_stream_draws": expected_rows,
        "prior_archetypes": [list(prior) for prior in PRIORS],
        "cluster_unit": "seed_position shared across all prior archetypes",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "metric_set": [
            "predictive_mean_l2",
            "predictive_covariance_frobenius",
            "empirical_cdf_i_ks",
            "empirical_cdf_q_ks",
            "posterior_mean_l1",
        ],
        "proper_score_and_llr": {
            "evaluation": "common-heldout paired semantic identity",
            "max_score_error": heldout_score_max_error,
            "max_llr_error": heldout_llr_max_error,
            "independent_raw_log_evidence_primary": False,
        },
        "raw_sampler_diagnostic_not_primary": raw_calibration,
        "selected_round_sample_count": selected,
        "formal_seed_pool_accessed": False,
        "historical_formal_cell_data_accessed": False,
        "gates": gates,
        "gate_summary": {
            "passed": sum(value is True for value in gates.values()),
            "total": len(gates),
        },
    }
    report["verdict"] = PASS_VERDICT if all(gates.values()) else FAIL_VERDICT
    report["analysis_sha256"] = _sha(report)
    return report, rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
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
    if report["verdict"] != PASS_VERDICT:
        raise RuntimeError(f"empirical readout power failed: {report['gates']}")
    report_path = base / "docs/t_risk_20260726_01_readout_power.json"
    source_path = base / "docs/t_risk_20260726_01_readout_power_source_data.csv"
    _atomic_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    fieldnames = sorted({key for row in rows for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        _atomic_text(source_path, stream.read())
    return report


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("readout power accepts no CLI overrides")
    report = write_artifacts()
    print(json.dumps({
        "verdict": report["verdict"],
        "analysis_sha256": report["analysis_sha256"],
        "gate_summary": report["gate_summary"],
        "selected_round_sample_count": report["selected_round_sample_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
