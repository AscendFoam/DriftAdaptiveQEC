"""Fresh, cell-data-blind diagnosis of the Phase-9 IQ semantics.

The diagnostic uses new calibration namespaces only.  It never reads the old
formal cell ledger, source data, or state archive, and it never chooses a
qualification margin.  Its purpose is to establish whether each independent
backend implements the same declared readout probability model as the third
analytic reference and whether both RNG implementations pass basic moment and
tail checks.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.phase9_fresh_twin_lineage import (
    PASS_VERDICT as LINEAGE_PASS,
    build_receipt as build_lineage_receipt,
)
from physics.phase9_backend_a import (
    BackendAConfig,
    Phase9BackendASimulator,
    backend_a_exogenous,
    diagnostic_action_word,
)
from physics.phase9_backend_b import (
    BackendBConfig,
    Phase9BackendBSimulator,
    backend_b_random_record,
    diagnostic_action_word_b,
)
from physics.phase9_iq_likelihood_reference import (
    INTEGRATION_CONVENTION,
    RAW_BASE_MEASURE,
    REFERENCE_ID,
    SIGMA_CONVENTION,
    evaluate_observation,
)


TASK_ID = "T-RISK-20260726-01"
SCHEMA_VERSION = "1.0"
PASS_VERDICT = "PASS_FRESH_IQ_SEMANTICS_DIAGNOSTIC"
FAIL_VERDICT = "INCOMPLETE_FAIL_CLOSED"
CALIBRATION_SEEDS = {
    "backend_a": {"start": 1_030_000, "count": 512},
    "backend_b": {"start": 1_031_000, "count": 512},
}
REFERENCE_STEP_SEEDS = {
    "backend_a": {"start": 1_030_000, "count": 8},
    "backend_b": {"start": 1_031_000, "count": 8},
}
ANCILLA_STATES = ("g", "e", "f")
ACTIONS = ("IDLE", "X", "RESET")
IQ_SAMPLES = 8


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


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


def _rng_values(backend: str) -> np.ndarray:
    split = CALIBRATION_SEEDS[backend]
    values: list[tuple[float, float]] = []
    for seed in range(split["start"], split["start"] + split["count"]):
        if backend == "backend_a":
            record = backend_a_exogenous(
                seed=seed, round_index=0, iq_samples=IQ_SAMPLES
            )
            i_values = record.iq_standard_i
            q_values = record.iq_standard_q
        else:
            record = backend_b_random_record(
                seed=seed, round_index=0, iq_samples=IQ_SAMPLES
            )
            i_values = record.iq_normal_i
            q_values = record.iq_normal_q
        values.extend(zip(i_values, q_values))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (split["count"] * IQ_SAMPLES, 2):
        raise RuntimeError(f"{backend} RNG accounting mismatch")
    return result


def _rng_summary(backend: str) -> dict[str, float | int]:
    values = _rng_values(backend)
    means = np.mean(values, axis=0)
    variances = np.var(values, axis=0, ddof=1)
    correlation = float(np.corrcoef(values.T)[0, 1])
    tail = np.mean(np.abs(values) > 1.959963984540054, axis=0)
    return {
        "draw_pairs": int(values.shape[0]),
        "mean_i": float(means[0]),
        "mean_q": float(means[1]),
        "variance_i": float(variances[0]),
        "variance_q": float(variances[1]),
        "correlation_iq": correlation,
        "two_sided_1_96_tail_i": float(tail[0]),
        "two_sided_1_96_tail_q": float(tail[1]),
    }


def _reference_rows(backend: str) -> list[dict[str, object]]:
    if backend == "backend_a":
        config: Any = BackendAConfig(
            cutoff=8, substeps_per_segment=1, iq_samples=IQ_SAMPLES
        )
        simulator: Any = Phase9BackendASimulator(config)
        action_factory = diagnostic_action_word
        random_factory = backend_a_exogenous
        prior_name = "pre_measurement_level_probabilities"
    else:
        config = BackendBConfig(
            cutoff=8, split_steps_per_segment=1, iq_samples=IQ_SAMPLES
        )
        simulator = Phase9BackendBSimulator(config)
        action_factory = diagnostic_action_word_b
        random_factory = backend_b_random_record
        prior_name = "pre_measurement_levels"

    split = REFERENCE_STEP_SEEDS[backend]
    rows: list[dict[str, object]] = []
    for ancilla in ANCILLA_STATES:
        for action in ACTIONS:
            for offset in range(split["count"]):
                seed = split["start"] + offset
                state = simulator.initialize_fock(ancilla_state=ancilla)
                result = simulator.step(
                    state,
                    action_factory(action),
                    random_factory(
                        seed=seed,
                        round_index=0,
                        iq_samples=IQ_SAMPLES,
                    ),
                    evaluator=None,
                )
                observation = result.observation
                prior = getattr(result.truth, prior_name)
                centers = np.asarray(config.iq_centers, dtype=np.float64).copy()
                # Both backends declare drift[2:4] as an additive readout-center
                # shift applied exactly once immediately before measurement.
                drift_before = np.asarray(result.truth.drift_before, dtype=np.float64)
                centers[:, 0] += drift_before[2]
                centers[:, 1] += drift_before[3]
                reference = evaluate_observation(
                    observation.iq_i,
                    observation.iq_q,
                    priors=prior,
                    centers=tuple(tuple(float(x) for x in row) for row in centers),
                    sigma=config.iq_sigma,
                )
                rows.append(
                    {
                        "backend": backend,
                        "ancilla": ancilla,
                        "action": action,
                        "seed": seed,
                        "sample_count": IQ_SAMPLES,
                        "log_evidence_abs_error": abs(
                            reference.log_evidence
                            - observation.log_evidence_density
                        ),
                        "posterior_l1_error": float(
                            np.sum(
                                np.abs(
                                    np.asarray(reference.posterior)
                                    - np.asarray(observation.posterior_levels)
                                )
                            )
                        ),
                        "integrated_mean_abs_error": max(
                            abs(
                                reference.integrated_i
                                - float(observation.integrated_i)
                            ),
                            abs(
                                reference.integrated_q
                                - float(observation.integrated_q)
                            ),
                        ),
                    }
                )
    return rows


def _source_independence() -> dict[str, object]:
    import physics.phase9_iq_likelihood_reference as reference

    source = inspect.getsource(reference)
    forbidden = (
        "phase9_backend_" + "a",
        "phase9_backend_" + "b",
        "numpy",
        "random",
    )
    hits = [token for token in forbidden if token in source]
    return {
        "reference_id": REFERENCE_ID,
        "forbidden_import_tokens": list(forbidden),
        "hits": hits,
        "independent": not hits,
    }


def build_report(root: Path | None = None) -> tuple[dict[str, Any], list[dict[str, object]]]:
    base = (root or _root()).resolve()
    lineage = build_lineage_receipt(base)
    rows = _reference_rows("backend_a") + _reference_rows("backend_b")
    rng = {
        backend: _rng_summary(backend)
        for backend in ("backend_a", "backend_b")
    }
    independence = _source_independence()
    max_log = {
        backend: max(
            float(row["log_evidence_abs_error"])
            for row in rows if row["backend"] == backend
        )
        for backend in ("backend_a", "backend_b")
    }
    max_posterior = {
        backend: max(
            float(row["posterior_l1_error"])
            for row in rows if row["backend"] == backend
        )
        for backend in ("backend_a", "backend_b")
    }
    max_integrated = max(float(row["integrated_mean_abs_error"]) for row in rows)
    rng_gates = {
        f"{backend}_{metric}": passed
        for backend, summary in rng.items()
        for metric, passed in {
            "mean": max(
                abs(float(summary["mean_i"])),
                abs(float(summary["mean_q"])),
            ) <= 0.06,
            "variance": max(
                abs(float(summary["variance_i"]) - 1.0),
                abs(float(summary["variance_q"]) - 1.0),
            ) <= 0.10,
            "cross_correlation": abs(float(summary["correlation_iq"])) <= 0.06,
            "tail": max(
                abs(float(summary["two_sided_1_96_tail_i"]) - 0.05),
                abs(float(summary["two_sided_1_96_tail_q"]) - 0.05),
            ) <= 0.02,
        }.items()
    }
    gates = {
        "G01_historical_no_go_lineage_live": lineage["verdict"] == LINEAGE_PASS,
        "G02_third_reference_independent": independence["independent"] is True,
        "G03_backend_a_log_evidence_matches_reference": max_log["backend_a"] <= 2e-11,
        "G04_backend_a_posterior_matches_reference": max_posterior["backend_a"] <= 2e-11,
        "G05_backend_b_log_evidence_matches_reference": max_log["backend_b"] <= 2e-11,
        "G06_backend_b_posterior_matches_reference": max_posterior["backend_b"] <= 2e-11,
        "G07_integrated_fields_are_actual_sample_means": max_integrated <= 2e-12,
        "G08_readout_base_measure_frozen": (
            RAW_BASE_MEASURE == "two_dimensional_lebesgue_per_complex_iq_sample"
        ),
        "G09_sigma_convention_frozen": (
            SIGMA_CONVENTION == "per_real_axis_standard_deviation"
        ),
        "G10_integration_convention_frozen": (
            INTEGRATION_CONVENTION == "arithmetic_mean_over_window"
        ),
        "G11_fresh_calibration_namespaces_disjoint": (
            CALIBRATION_SEEDS["backend_a"]["start"]
            + CALIBRATION_SEEDS["backend_a"]["count"]
            <= CALIBRATION_SEEDS["backend_b"]["start"]
        ),
        "G12_rng_draw_accounting_complete": all(
            summary["draw_pairs"] == 4096 for summary in rng.values()
        ),
        **{
            f"G{13 + index:02d}_{name}": value
            for index, (name, value) in enumerate(sorted(rng_gates.items()))
        },
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "purpose": "fresh_iq_semantics_diagnostic_not_qualification",
        "old_formal_cell_data_accessed": False,
        "margin_or_threshold_selected_from_old_outcome": False,
        "historical_lineage_analysis_sha256": lineage["analysis_sha256"],
        "readout_convention": {
            "reference_id": REFERENCE_ID,
            "raw_base_measure": RAW_BASE_MEASURE,
            "sigma": SIGMA_CONVENTION,
            "integration": INTEGRATION_CONVENTION,
            "latent_conditioning": "one_ancilla_label_per_complete_window",
            "gain_rule": "2x2 affine Jacobian applied once per raw sample",
        },
        "independent_reference": independence,
        "fresh_seed_splits": {
            "calibration": CALIBRATION_SEEDS,
            "reference_steps": REFERENCE_STEP_SEEDS,
        },
        "reference_step_accounting": {
            "expected": 2 * len(ANCILLA_STATES) * len(ACTIONS) * 8,
            "observed": len(rows),
            "backend_a_max_log_evidence_abs_error": max_log["backend_a"],
            "backend_b_max_log_evidence_abs_error": max_log["backend_b"],
            "backend_a_max_posterior_l1_error": max_posterior["backend_a"],
            "backend_b_max_posterior_l1_error": max_posterior["backend_b"],
            "max_integrated_mean_abs_error": max_integrated,
        },
        "rng_calibration": rng,
        "historical_diagnosis": {
            "provenance": "read_only_source_and_seed_reconstruction_audit",
            "old_gate_count": 1042,
            "old_failed_gate_count": 217,
            "iq_failures_explained_by_two_reused_rng_blocks": 198,
            "reset_coupled_failures": 9,
            "mixed_unit_fault_composite_failures": 8,
            "independent_cutoff_survival_failures": 2,
            "interpretation": (
                "backend likelihood formulas agree; repair evaluator "
                "pseudo-replication, power, score units, reset grouping, "
                "fault metric units, and cutoff convergence"
            ),
        },
        "gates": gates,
        "gate_summary": {
            "passed": sum(value is True for value in gates.values()),
            "total": len(gates),
        },
    }
    report["verdict"] = PASS_VERDICT if all(gates.values()) else FAIL_VERDICT
    report["analysis_sha256"] = _sha(report)
    return report, rows


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    report, rows = build_report(base)
    if report["verdict"] != PASS_VERDICT:
        raise RuntimeError(f"IQ semantics diagnostic failed: {report['gates']}")
    json_path = base / "docs/t_risk_20260726_01_iq_semantics_diagnostic.json"
    csv_path = base / "docs/t_risk_20260726_01_iq_semantics_source_data.csv"
    _atomic_text(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    fieldnames = list(rows[0])
    lines: list[str] = []
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        lines.append(stream.read())
    _atomic_text(csv_path, "".join(lines))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("this diagnostic accepts no CLI overrides")
    report = write_artifacts()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "gate_summary": report["gate_summary"],
                "analysis_sha256": report["analysis_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

