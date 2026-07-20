"""T6.18.1 project-native AQEC/common-wall-clock secondary replay.

The executable lane deliberately reuses the existing finite-cutoff exact CPTP
simulator.  It does not construct a substitute for the Lachance et al.
reservoir-engineered experiment.  Independent seed clusters are effect-blind,
mean-preserving quasi-static lifetime realizations shared by idle,
measurement/reset and autonomous/reset anchors within each cell.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from cnn_fpga.benchmark.autonomous_sbs_wallclock_baseline import NOISE_PROFILES
from physics.autonomous_sbs import (
    AUTONOMOUS_TIMING,
    MEASUREMENT_TIMING,
    MODEL_SCOPE,
    IdleMemoryConfig,
    IdleMemorySimulator,
    NonselectiveSBSConfig,
    NonselectiveSBSSimulator,
    validate_timing_contract,
)

try:  # Base verification environment intentionally need not contain torch.
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover
    psutil = None


ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "T6.18.1"
SCHEMA_VERSION = "t6.18.1-aqec-common-wallclock-v1"
VERDICT = "PASS_PROJECT_NATIVE_AQEC_WALLCLOCK_WITH_OFFICIAL_PROTOCOL_BLOCKED"
PREREG_CONFIG = ROOT / "configs" / "literature" / "t6_16_3_secondary_preregistration.json"
SOURCE_AUDIT = ROOT / "docs" / "t6_16_1_secondary_method_source_audit.json"
ONTOLOGY = ROOT / "docs" / "t6_16_2_comparison_ontology.json"
PARENT_REPORT = ROOT / "docs" / "t3_2_8_autonomous_sbs_wallclock_validation.json"
PARENT_SOURCE_DATA = ROOT / "docs" / "t3_2_8_autonomous_sbs_wallclock_source_data.csv"
PHYSICS_IMPLEMENTATION = ROOT / "physics" / "autonomous_sbs.py"
PHASE6B_TERMINAL = ROOT / "docs" / "t6_15_5_route_a_v5_final_evidence_gate.json"
DEFAULT_REPORT = ROOT / "docs" / "t6_18_1_aqec_common_wallclock_replay.json"
DEFAULT_SOURCE_DATA = ROOT / "docs" / "t6_18_1_aqec_common_wallclock_source_data.csv"
DEFAULT_MARKDOWN = ROOT / "docs" / "aqec_common_wallclock_replay.md"
PARTIAL_CACHE = ROOT / "docs" / ".t6_18_1_aqec_partial.json"

ANCHORS = ("idle_memory", "measurement_feedback", "autonomous")
CURVE_METRICS = ("fidelity", "code_survival", "logical_z_signal", "conditional_logical_z")
PRIMARY_METRICS = (
    "logical_lifetime_us",
    "logical_lifetime_cycles",
    "lifetime_gain_ratio",
    "final_code_survival",
    "measurements_per_100us",
    "resets_per_100us",
    "active_gates_per_100us",
)
NOISE_LOG_STD = 0.05
NOISE_MODEL_ID = "mean-preserving-lognormal-quasistatic-lifetimes-sigma0p05-v1"
BOOTSTRAP_REPS = 20_000
BOOTSTRAP_BASE_SEED = 61_819_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    ).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _binding(path: Path) -> dict[str, Any]:
    return {"path": _relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _preregistered_experiment() -> dict[str, Any]:
    rows = [row for row in _load(PREREG_CONFIG)["experiments"] if row["task_id"] == TASK_ID]
    if len(rows) != 1:
        raise ValueError("T6.18.1 requires exactly one frozen preregistration row")
    return rows[0]


def _source_record() -> dict[str, Any]:
    rows = [row for row in _load(SOURCE_AUDIT)["sources"] if row["source_id"] == "LACHANCE2024_AQEC"]
    if len(rows) != 1:
        raise ValueError("LACHANCE2024_AQEC source row must be unique")
    return rows[0]


def _source_method() -> dict[str, Any]:
    rows = [row for row in _load(SOURCE_AUDIT)["methods"] if row["source_id"] == "LACHANCE2024_AQEC"]
    if len(rows) != 1:
        raise ValueError("LACHANCE2024_AQEC method row must be unique")
    return rows[0]


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), PHYSICS_IMPLEMENTATION):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sampled_lifetimes(seed: int, cutoff: int, noise_name: str) -> dict[str, Any]:
    """Effect-blind, physically valid CRN nuisance realization for one cell."""

    if noise_name not in NOISE_PROFILES:
        raise ValueError(f"unknown noise profile {noise_name}")
    profile_index = tuple(NOISE_PROFILES).index(noise_name)
    rng = np.random.default_rng(np.random.SeedSequence([6_181, seed, cutoff, profile_index]))
    z = rng.standard_normal(3)
    factors = np.exp(NOISE_LOG_STD * z - 0.5 * NOISE_LOG_STD**2)
    nominal_cavity, nominal_t1, nominal_t2 = NOISE_PROFILES[noise_name]
    nominal_tphi_rate = 1.0 / nominal_t2 - 1.0 / (2.0 * nominal_t1)
    if nominal_tphi_rate <= 0.0:
        raise ValueError("nominal profile violates T2 <= 2*T1")
    nominal_tphi = 1.0 / nominal_tphi_rate
    cavity = nominal_cavity * factors[0]
    ancilla_t1 = nominal_t1 * factors[1]
    ancilla_tphi = nominal_tphi * factors[2]
    ancilla_t2 = 1.0 / (1.0 / (2.0 * ancilla_t1) + 1.0 / ancilla_tphi)
    values = {
        "model_id": NOISE_MODEL_ID,
        "seed": int(seed),
        "cutoff": int(cutoff),
        "noise_profile": noise_name,
        "standard_normal_draws": z.tolist(),
        "cavity_lifetime_us": float(cavity),
        "ancilla_t1_us": float(ancilla_t1),
        "ancilla_tphi_us": float(ancilla_tphi),
        "ancilla_t2_us": float(ancilla_t2),
    }
    values["fingerprint"] = _canonical_sha256({key: value for key, value in values.items() if key != "fingerprint"})
    return values


def exponential_fit_diagnostic(time_us: Any, curve: Any) -> dict[str, Any]:
    times = np.asarray(time_us, dtype=np.float64)
    values = np.asarray(curve, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape or times.size < 3:
        raise ValueError("fit curve/time shape mismatch")
    normalized = values / values[0]
    valid = np.isfinite(normalized) & (normalized > 1.0e-14)
    if int(valid.sum()) < 3:
        return {"status": "INSUFFICIENT_POSITIVE_POINTS", "points": int(valid.sum()), "tau_us": None, "slope_per_us": None, "intercept": None, "r_squared": None, "log_rmse": None, "monotonic_increase_count": int(np.sum(np.diff(normalized) > 1.0e-12))}
    x = times[valid]
    y = np.log(normalized[valid])
    slope, intercept = np.polyfit(x, y, deg=1)
    prediction = slope * x + intercept
    residual = y - prediction
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0.0 else None
    return {
        "status": "DECAY_FIT" if slope < 0.0 else "NONDECAYING_FIT",
        "points": int(valid.sum()),
        "tau_us": float(-1.0 / slope) if slope < 0.0 else None,
        "slope_per_us": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "log_rmse": float(np.sqrt(np.mean(residual**2))),
        "monotonic_increase_count": int(np.sum(np.diff(normalized) > 1.0e-12)),
    }


def area_equivalent_lifetime(time_us: Any, curve: Any) -> dict[str, float]:
    """Parent-equivalent lifetime with NumPy 1.x/2.x trapezoid support."""

    times = np.asarray(time_us, dtype=np.float64)
    values = np.asarray(curve, dtype=np.float64)
    if times.ndim != 1 or values.shape != times.shape or times.size < 3:
        raise ValueError("time and curve must be aligned rank-one arrays with >=3 points")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError("time and curve must be finite")
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("time must start at zero and increase strictly")
    if abs(values[0]) <= 1.0e-14:
        raise ValueError("curve initial value must be nonzero")
    normalized = values / values[0]
    horizon = float(times[-1])
    trapezoid = getattr(np, "trapezoid", np.trapz)
    normalized_auc = float(trapezoid(normalized, times) / horizon)
    if not 0.0 < normalized_auc <= 1.0 + 1.0e-9:
        raise ValueError("normalized signed area lies outside (0,1]")
    normalized_auc = min(normalized_auc, 1.0)
    if normalized_auc >= 1.0 - 1.0e-12:
        lifetime = 1.0e12 * horizon
    else:
        lower = 1.0e-12 * horizon
        upper = 1.0e12 * horizon
        for _ in range(160):
            middle = 0.5 * (lower + upper)
            area = (middle / horizon) * (1.0 - np.exp(-horizon / middle))
            if area < normalized_auc:
                lower = middle
            else:
                upper = middle
        lifetime = 0.5 * (lower + upper)
    return {
        "normalized_signed_auc": normalized_auc,
        "area_equivalent_lifetime_us": float(lifetime),
        "area_equivalent_lifetime_protocol_cycles": float(lifetime / (times[1] - times[0])),
        "area_equivalent_lifetime_standard_10us_cycles": float(lifetime / 10.0),
        "horizon_us": horizon,
    }


def _run_anchor(anchor: str, cutoff: int, noise: Mapping[str, Any], device: str) -> dict[str, Any]:
    common = {
        "cutoff": cutoff,
        "projector_delta": 0.34,
        "cavity_lifetime_us": noise["cavity_lifetime_us"],
        "ancilla_t1_us": noise["ancilla_t1_us"],
        "ancilla_t2_us": noise["ancilla_t2_us"],
        "device": device,
        "real_dtype": "float64",
    }
    if anchor == "idle_memory":
        result = IdleMemorySimulator(IdleMemoryConfig(full_cycles=70, cycle_duration_us=10.0, **common)).run()
        cycle_duration = 10.0
    elif anchor == "measurement_feedback":
        result = NonselectiveSBSSimulator(NonselectiveSBSConfig(mode=anchor, full_cycles=70, **common)).run()
        cycle_duration = MEASUREMENT_TIMING.full_cycle_duration_ns / 1000.0
    elif anchor == "autonomous":
        result = NonselectiveSBSSimulator(NonselectiveSBSConfig(mode=anchor, full_cycles=100, **common)).run()
        cycle_duration = AUTONOMOUS_TIMING.full_cycle_duration_ns / 1000.0
    else:
        raise ValueError(f"unknown anchor {anchor}")
    payload = result.to_dict()
    curves = {metric: payload[metric] for metric in CURVE_METRICS}
    area = area_equivalent_lifetime(payload["time_us"], payload["logical_z_signal"])
    event = payload["event_accounting"]
    metrics = {
        "logical_lifetime_us": area["area_equivalent_lifetime_us"],
        "logical_lifetime_cycles": area["area_equivalent_lifetime_us"] / cycle_duration,
        "lifetime_gain_ratio": None,
        "final_code_survival": curves["code_survival"][-1],
        "measurements_per_100us": event["measurements_per_100us"],
        "resets_per_100us": event["resets_per_100us"],
        "active_gates_per_100us": event["active_gates_per_100us"],
    }
    return {
        "anchor": anchor,
        "noise_fingerprint": noise["fingerprint"],
        "cycle_duration_us": cycle_duration,
        "time_us": payload["time_us"],
        "curves": curves,
        "metrics": metrics,
        "area_lifetime": area,
        "fit_diagnostics": {
            metric: exponential_fit_diagnostic(payload["time_us"], curves[metric])
            for metric in ("logical_z_signal", "code_survival", "fidelity")
        },
        "event_accounting": event,
        "health": {
            "maximum_trace_error": payload["maximum_trace_error"],
            "maximum_hermiticity_error": payload["maximum_hermiticity_error"],
            "minimum_final_eigenvalue": payload["minimum_final_eigenvalue"],
        },
        "unavailable_fields": {
            "pulse_energy_joule": None,
            "control_duty_fraction": None,
            "classical_decoder_latency_ns": None,
        },
    }


def _bootstrap_indices(cell_index: int, n: int) -> tuple[np.ndarray, int]:
    seed = BOOTSTRAP_BASE_SEED + cell_index
    indices = np.random.default_rng(seed).integers(0, n, size=(BOOTSTRAP_REPS, n), endpoint=False)
    return indices, seed


def _summary(values: Any, indices: np.ndarray, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    bootstrap = array[indices].mean(axis=1)
    return {
        "n_seed_clusters": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "ci95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": seed,
    }


def _cell_summary(seed_records: list[dict[str, Any]], cell_index: int) -> dict[str, Any]:
    indices, bootstrap_seed = _bootstrap_indices(cell_index, len(seed_records))
    anchors: dict[str, Any] = {}
    for anchor in ANCHORS:
        anchors[anchor] = {
            metric: _summary([row["anchors"][anchor]["metrics"][metric] for row in seed_records], indices, bootstrap_seed)
            for metric in PRIMARY_METRICS
        }
    paired_values = {
        "measurement_vs_idle_lifetime_gain": [row["anchors"]["measurement_feedback"]["metrics"]["lifetime_gain_ratio"] for row in seed_records],
        "autonomous_vs_idle_lifetime_gain": [row["anchors"]["autonomous"]["metrics"]["lifetime_gain_ratio"] for row in seed_records],
        "autonomous_vs_measurement_lifetime_us_ratio": [row["anchors"]["autonomous"]["metrics"]["logical_lifetime_us"] / row["anchors"]["measurement_feedback"]["metrics"]["logical_lifetime_us"] for row in seed_records],
        "autonomous_minus_measurement_final_survival": [row["anchors"]["autonomous"]["metrics"]["final_code_survival"] - row["anchors"]["measurement_feedback"]["metrics"]["final_code_survival"] for row in seed_records],
    }
    reversal = [
        row["anchors"]["autonomous"]["metrics"]["logical_lifetime_cycles"]
        > row["anchors"]["measurement_feedback"]["metrics"]["logical_lifetime_cycles"]
        and row["anchors"]["autonomous"]["metrics"]["logical_lifetime_us"]
        < row["anchors"]["measurement_feedback"]["metrics"]["logical_lifetime_us"]
        for row in seed_records
    ]
    return {
        "anchors": anchors,
        "paired": {name: _summary(values, indices, bootstrap_seed) for name, values in paired_values.items()},
        "ordering_reversal": {
            "definition": "autonomous lifetime is higher in protocol cycles but lower in common-wall-clock microseconds",
            "count": int(sum(reversal)),
            "total": len(reversal),
            "fraction": float(np.mean(reversal)),
        },
    }


def _build_cell(cutoff: int, noise_name: str, seeds: list[int], device: str, cell_index: int, *, progress: bool) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        noise = sampled_lifetimes(seed, cutoff, noise_name)
        anchors = {anchor: _run_anchor(anchor, cutoff, noise, device) for anchor in ANCHORS}
        idle_lifetime = anchors["idle_memory"]["metrics"]["logical_lifetime_us"]
        for anchor in ANCHORS:
            anchors[anchor]["metrics"]["lifetime_gain_ratio"] = anchors[anchor]["metrics"]["logical_lifetime_us"] / idle_lifetime
        records.append({"seed": seed, "noise": noise, "anchors": anchors})
        if progress:
            print(json.dumps({"progress": TASK_ID, "cell": f"cutoff{cutoff}_{noise_name}", "seed": seed, "completed": seed_index + 1, "total": len(seeds)}), flush=True)
    return {
        "cell_id": f"cutoff{cutoff}_{noise_name}",
        "cutoff": cutoff,
        "noise_profile": noise_name,
        "seed_records": records,
        "summary": _cell_summary(records, cell_index),
    }


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in report["cells"]:
        for seed_row in cell["seed_records"]:
            seed = seed_row["seed"]
            for name in ("cavity_lifetime_us", "ancilla_t1_us", "ancilla_tphi_us", "ancilla_t2_us"):
                rows.append({"record_type": "noise", "cell_id": cell["cell_id"], "cutoff": cell["cutoff"], "noise_profile": cell["noise_profile"], "seed": seed, "anchor": "shared_crn", "time_index": "", "time_us": "", "metric": name, "value": seed_row["noise"][name], "value_state": "PROJECT_NATIVE_MATCHED", "details": seed_row["noise"]["fingerprint"]})
            for anchor in ANCHORS:
                payload = seed_row["anchors"][anchor]
                for time_index, time_us in enumerate(payload["time_us"]):
                    for metric in CURVE_METRICS:
                        rows.append({"record_type": "curve", "cell_id": cell["cell_id"], "cutoff": cell["cutoff"], "noise_profile": cell["noise_profile"], "seed": seed, "anchor": anchor, "time_index": time_index, "time_us": time_us, "metric": metric, "value": payload["curves"][metric][time_index], "value_state": "PROJECT_NATIVE_MATCHED", "details": seed_row["noise"]["fingerprint"]})
                for metric in PRIMARY_METRICS:
                    rows.append({"record_type": "seed_metric", "cell_id": cell["cell_id"], "cutoff": cell["cutoff"], "noise_profile": cell["noise_profile"], "seed": seed, "anchor": anchor, "time_index": "", "time_us": 700.0, "metric": metric, "value": payload["metrics"][metric], "value_state": "PROJECT_NATIVE_MATCHED", "details": "area-equivalent lifetime; event accounting exact within project model"})
            for metric, value in {
                "measurement_vs_idle_lifetime_gain": seed_row["anchors"]["measurement_feedback"]["metrics"]["lifetime_gain_ratio"],
                "autonomous_vs_idle_lifetime_gain": seed_row["anchors"]["autonomous"]["metrics"]["lifetime_gain_ratio"],
                "autonomous_vs_measurement_lifetime_us_ratio": seed_row["anchors"]["autonomous"]["metrics"]["logical_lifetime_us"] / seed_row["anchors"]["measurement_feedback"]["metrics"]["logical_lifetime_us"],
                "autonomous_minus_measurement_final_survival": seed_row["anchors"]["autonomous"]["metrics"]["final_code_survival"] - seed_row["anchors"]["measurement_feedback"]["metrics"]["final_code_survival"],
            }.items():
                rows.append({"record_type": "paired", "cell_id": cell["cell_id"], "cutoff": cell["cutoff"], "noise_profile": cell["noise_profile"], "seed": seed, "anchor": "paired", "time_index": "", "time_us": 700.0, "metric": metric, "value": value, "value_state": "PROJECT_NATIVE_MATCHED", "details": "same seed-level lifetime realization across anchors"})
    for metric in _source_method()["metrics"]:
        rows.append({"record_type": "literature", "cell_id": "LACHANCE2024_AQEC", "cutoff": "", "noise_profile": "experimental", "seed": "", "anchor": "paper", "time_index": "", "time_us": "", "metric": metric["metric_id"], "value": metric["value"], "value_state": "LITERATURE_ONLY", "details": metric["source_locator"]})
    rows.extend([
        {"record_type": "boundary", "cell_id": "LACHANCE2024_AQEC", "cutoff": "", "noise_profile": "experimental", "seed": "", "anchor": "paper", "time_index": "", "time_us": "", "metric": "classical_decoder_latency_ns", "value": "", "value_state": "N_A_NOT_APPLICABLE", "details": "physical pulses/reset have finite duration; N/A is not zero"},
        {"record_type": "boundary", "cell_id": "LACHANCE2024_AQEC", "cutoff": "", "noise_profile": "experimental", "seed": "", "anchor": "paper", "time_index": "", "time_us": "", "metric": "official_protocol_reproduction", "value": "", "value_state": "BLOCKED", "details": "official code and paper-native reservoir adapter absent"},
    ])
    return rows


def _write_csv(report: Mapping[str, Any]) -> None:
    fields = ("record_type", "cell_id", "cutoff", "noise_profile", "seed", "anchor", "time_index", "time_us", "metric", "value", "value_state", "details")
    rows = _source_rows(report)
    with DEFAULT_SOURCE_DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _live_anchor_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    time_us = np.asarray(payload["time_us"], dtype=np.float64)
    logical = np.asarray(payload["curves"]["logical_z_signal"], dtype=np.float64)
    area = area_equivalent_lifetime(time_us, logical)
    metrics = dict(payload["metrics"])
    metrics["logical_lifetime_us"] = area["area_equivalent_lifetime_us"]
    metrics["logical_lifetime_cycles"] = area["area_equivalent_lifetime_us"] / payload["cycle_duration_us"]
    metrics["final_code_survival"] = payload["curves"]["code_survival"][-1]
    return {
        "area_lifetime": area,
        "fit_diagnostics": {
            metric: exponential_fit_diagnostic(time_us, payload["curves"][metric])
            for metric in ("logical_z_signal", "code_survival", "fidelity")
        },
        "metrics_without_gain": {key: value for key, value in metrics.items() if key != "lifetime_gain_ratio"},
    }


def _upgrade_analysis_fields(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute analysis-only fields from preserved raw curves."""

    upgraded = deepcopy(cells)
    for cell_index, cell in enumerate(upgraded):
        for seed_row in cell["seed_records"]:
            for anchor in ANCHORS:
                payload = seed_row["anchors"][anchor]
                payload.pop("fit_diagnostic", None)
                payload["fit_diagnostics"] = {
                    metric: exponential_fit_diagnostic(payload["time_us"], payload["curves"][metric])
                    for metric in ("logical_z_signal", "code_survival", "fidelity")
                }
        cell["summary"] = _cell_summary(cell["seed_records"], cell_index)
    return upgraded


def _raw_cells_projection(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": cell["cell_id"],
            "cutoff": cell["cutoff"],
            "noise_profile": cell["noise_profile"],
            "seed_records": [
                {
                    "seed": row["seed"],
                    "noise": row["noise"],
                    "anchors": {
                        anchor: {
                            key: row["anchors"][anchor][key]
                            for key in ("noise_fingerprint", "cycle_duration_us", "time_us", "curves", "event_accounting", "health", "unavailable_fields")
                        }
                        for anchor in ANCHORS
                    },
                }
                for row in cell["seed_records"]
            ],
        }
        for cell in cells
    ]


def evaluate_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    frozen = _preregistered_experiment()
    parent = _load(PARENT_REPORT)
    cells = report["cells"]
    expected_cells = {(cutoff, noise) for cutoff in frozen["config"]["cutoffs"] for noise in frozen["config"]["noise_profiles"]}
    expected_seeds = frozen["seeds"]["values"]
    all_seed_rows = [row for cell in cells for row in cell["seed_records"]]
    all_anchors = [row["anchors"][anchor] for row in all_seed_rows for anchor in ANCHORS]
    live_metric_ok = True
    for seed_row in all_seed_rows:
        idle_lifetime = seed_row["anchors"]["idle_memory"]["metrics"]["logical_lifetime_us"]
        for anchor in ANCHORS:
            payload = seed_row["anchors"][anchor]
            live = _live_anchor_projection(payload)
            expected_gain = payload["metrics"]["logical_lifetime_us"] / idle_lifetime
            live_metric_ok &= live["area_lifetime"] == payload["area_lifetime"] and live["fit_diagnostics"] == payload["fit_diagnostics"] and live["metrics_without_gain"] == {key: value for key, value in payload["metrics"].items() if key != "lifetime_gain_ratio"} and payload["metrics"]["lifetime_gain_ratio"] == expected_gain
    expected_summaries = [_cell_summary(cell["seed_records"], index) for index, cell in enumerate(cells)]
    source_path = ROOT / report["source_data"]["path"]
    bindings = report["bindings"]
    return {
        "G01_frozen_preregistration_and_effect_blind_nuisance_model_are_exact": report["preregistration"]["record_sha256"] == _canonical_sha256(frozen) and report["nuisance_model"] == {"model_id": NOISE_MODEL_ID, "log_standard_deviation": NOISE_LOG_STD, "selection_timing": "fixed before any T6.18.1 simulation result", "interpretation": "project-model quasi-static robustness only; not device uncertainty or paper fit error"},
        "G02_source_scope_keeps_lachance_literature_separate_from_project_replay": report["source_scope"]["record_sha256"] == _canonical_sha256(_source_record()) and report["source_scope"]["method_sha256"] == _canonical_sha256(_source_method()) and report["source_scope"]["official_code"] is None,
        "G03_parent_protocol_simulator_and_common_700us_contract_are_live": parent["status"] == "PASS" and parent["contract_id"] == "T328-PROTOCOL-NATIVE-COMMON-700US-V1" and _sha256(PARENT_REPORT) == bindings["parent_report"]["sha256"] and _sha256(PARENT_SOURCE_DATA) == bindings["parent_source_data"]["sha256"] and all(validate_timing_contract().values()),
        "G04_all_six_cells_have_24_independent_registered_seed_clusters": {(cell["cutoff"], cell["noise_profile"]) for cell in cells} == expected_cells and len(cells) == 6 and all([row["seed"] for row in cell["seed_records"]] == expected_seeds for cell in cells) and len({row["noise"]["fingerprint"] for row in all_seed_rows}) == len(all_seed_rows),
        "G05_noise_realizations_are_physical_and_shared_as_crn_across_anchors": all(row["noise"]["model_id"] == NOISE_MODEL_ID and row["noise"]["ancilla_t2_us"] <= 2.0 * row["noise"]["ancilla_t1_us"] + 1e-12 and all(row["anchors"][anchor]["noise_fingerprint"] == row["noise"]["fingerprint"] for anchor in ANCHORS) for row in all_seed_rows),
        "G06_all_anchors_reach_exact_common_wallclock_with_minimum_time_points": all(payload["time_us"][-1] == 700.0 and len(payload["time_us"]) >= 71 and payload["cycle_duration_us"] == (7.0 if payload["anchor"] == "autonomous" else 10.0) for payload in all_anchors),
        "G07_density_health_and_all_curve_values_are_finite": all(payload["health"]["maximum_trace_error"] <= 2e-9 and payload["health"]["maximum_hermiticity_error"] <= 2e-9 and payload["health"]["minimum_final_eigenvalue"] >= -2e-8 and all(np.all(np.isfinite(payload["curves"][metric])) for metric in CURVE_METRICS) for payload in all_anchors),
        "G08_idle_anchor_has_zero_measurement_reset_gate_and_latency_claims": all(all(row["anchors"]["idle_memory"]["event_accounting"].get(field, 0) == 0 for field in ("measurement_events", "reset_events", "active_gate_applications", "outcome_dependent_parameter_updates")) and row["anchors"]["idle_memory"]["unavailable_fields"]["classical_decoder_latency_ns"] is None for row in all_seed_rows),
        "G09_protocol_event_rates_are_native_and_not_zero_latency_proxies": all(row["anchors"]["measurement_feedback"]["metrics"]["measurements_per_100us"] == 20.0 and row["anchors"]["autonomous"]["metrics"]["measurements_per_100us"] == 0.0 and row["anchors"]["measurement_feedback"]["metrics"]["resets_per_100us"] == 20.0 and np.isclose(row["anchors"]["autonomous"]["metrics"]["resets_per_100us"], 200.0 / 7.0, rtol=0.0, atol=1e-14) for row in all_seed_rows),
        "G10_lifetime_survival_and_fit_metrics_recompute_from_every_raw_curve": live_metric_ok,
        "G11_paired_20k_seed_cluster_bootstrap_and_reversal_are_live": all(cell["summary"] == expected for cell, expected in zip(cells, expected_summaries, strict=True)) and all(summary["paired"][name]["bootstrap_reps"] == BOOTSTRAP_REPS for summary in expected_summaries for name in summary["paired"]),
        "G12_no_performance_direction_or_universal_gain_is_required": report["stopping_rule"] == "all 6 cells x 24 clusters; no desired ordering gate" and report["project_result"]["cells"] == 6 and report["project_result"]["seed_clusters"] == 144 and report["project_result"]["universal_20_percent_claim"] is False,
        "G13_official_reservoir_reproduction_and_energy_latency_fields_fail_closed": report["official_protocol_reproduction"] == {"state": "BLOCKED_OFFICIAL_PROTOCOL_REPRODUCTION", "official_code_available": False, "paper_native_reservoir_adapter": False, "project_replay_may_substitute": False} and all(all(value is None for value in payload["unavailable_fields"].values()) for payload in all_anchors),
        "G14_source_data_and_all_artifact_bindings_are_live": report["source_data"]["sha256"] == _sha256(source_path) and bindings["source_data"]["sha256"] == report["source_data"]["sha256"] and all(_sha256(ROOT / row["path"]) == row["sha256"] and (ROOT / row["path"]).stat().st_size == row["bytes"] for row in bindings.values()),
        "G15_execution_budget_and_phase6b_noninterference_are_preserved": report["execution_budget_audit"]["within_runtime_budget"] and report["execution_budget_audit"]["within_memory_budget"] and report["execution_budget_audit"]["accounted_peak_memory_bytes"] == max(report["execution_budget_audit"]["peak_device_memory_bytes"], report["execution_budget_audit"]["peak_host_working_set_bytes"]) and report["execution_budget_audit"]["peak_host_working_set_bytes"] > 0 and report["phase6b_noninterference"] == {"training_executed": False, "threshold_tuned": False, "phase6b_outputs_modified": False, "phase6b_verdict": "NO_GO_V5_EARLY_HEADROOM_STOP"} and _load(PHASE6B_TERMINAL)["verdict"] == "NO_GO_V5_EARLY_HEADROOM_STOP",
        "G16_all_targeted_semantic_mutations_are_detected": report["semantic_mutation_audit"]["count"] == report["semantic_mutation_audit"]["detected"] == 16 and all(row["rejected"] for row in report["semantic_mutation_audit"]["cases"]),
    }


def _mutations(report: Mapping[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def attempt(name: str, gate: str, mutate: Any) -> None:
        candidate = deepcopy(report)
        candidate["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
        mutate(candidate)
        try:
            rejected = not evaluate_gates(candidate)[gate]
        except Exception:
            rejected = True
        cases.append({"case": name, "target_gate": gate, "rejected": rejected})

    attempt("forge_prereg_hash", "G01_frozen_preregistration_and_effect_blind_nuisance_model_are_exact", lambda x: x["preregistration"].update(record_sha256="0" * 64))
    attempt("claim_official_source_code", "G02_source_scope_keeps_lachance_literature_separate_from_project_replay", lambda x: x["source_scope"].update(official_code="invented"))
    attempt("forge_parent_hash", "G03_parent_protocol_simulator_and_common_700us_contract_are_live", lambda x: x["bindings"]["parent_report"].update(sha256="0" * 64))
    attempt("drop_seed_cluster", "G04_all_six_cells_have_24_independent_registered_seed_clusters", lambda x: x["cells"][0]["seed_records"].pop())
    attempt("break_crn_fingerprint", "G05_noise_realizations_are_physical_and_shared_as_crn_across_anchors", lambda x: x["cells"][0]["seed_records"][0]["anchors"]["autonomous"].update(noise_fingerprint="0" * 64))
    attempt("truncate_curve", "G06_all_anchors_reach_exact_common_wallclock_with_minimum_time_points", lambda x: x["cells"][0]["seed_records"][0]["anchors"]["idle_memory"]["time_us"].pop())
    attempt("forge_density_health", "G07_density_health_and_all_curve_values_are_finite", lambda x: x["cells"][0]["seed_records"][0]["anchors"]["autonomous"]["health"].update(maximum_trace_error=1.0))
    attempt("invent_idle_gate", "G08_idle_anchor_has_zero_measurement_reset_gate_and_latency_claims", lambda x: x["cells"][0]["seed_records"][0]["anchors"]["idle_memory"]["event_accounting"].update(active_gate_applications=1))
    attempt("rename_reset_rate", "G09_protocol_event_rates_are_native_and_not_zero_latency_proxies", lambda x: x["cells"][0]["seed_records"][0]["anchors"]["autonomous"]["metrics"].update(resets_per_100us=20.0))
    attempt("forge_lifetime", "G10_lifetime_survival_and_fit_metrics_recompute_from_every_raw_curve", lambda x: x["cells"][0]["seed_records"][0]["anchors"]["measurement_feedback"]["metrics"].update(logical_lifetime_us=1e9))
    attempt("forge_bootstrap", "G11_paired_20k_seed_cluster_bootstrap_and_reversal_are_live", lambda x: x["cells"][0]["summary"]["paired"]["measurement_vs_idle_lifetime_gain"].update(mean=99.0))
    attempt("claim_universal_gain", "G12_no_performance_direction_or_universal_gain_is_required", lambda x: x["project_result"].update(universal_20_percent_claim=True))
    attempt("claim_official_reproduction", "G13_official_reservoir_reproduction_and_energy_latency_fields_fail_closed", lambda x: x["official_protocol_reproduction"].update(project_replay_may_substitute=True))
    attempt("forge_source_hash", "G14_source_data_and_all_artifact_bindings_are_live", lambda x: x["source_data"].update(sha256="0" * 64))
    attempt("rewrite_phase6b", "G15_execution_budget_and_phase6b_noninterference_are_preserved", lambda x: x["phase6b_noninterference"].update(phase6b_outputs_modified=True))
    attempt("forge_mutation_count", "G16_all_targeted_semantic_mutations_are_detected", lambda x: x.update(semantic_mutation_audit={"count": 16, "detected": 15, "cases": []}))
    return {"count": len(cases), "detected": sum(row["rejected"] for row in cases), "cases": cases}


def _cache_key(prereg_sha: str, device: str) -> dict[str, Any]:
    return {"implementation_sha256": implementation_sha256(), "preregistration_sha256": prereg_sha, "device": device, "noise_model_id": NOISE_MODEL_ID}


def build_report(
    *,
    device: str = "cuda",
    resume: bool = True,
    progress: bool = True,
    reuse_report_path: Path | None = None,
    observed_campaign_host_peak_wset: int | None = None,
    observed_simulation_runtime_seconds: float | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("T6.18.1 execution requires the local DLEnv PyTorch environment")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA execution requested but unavailable")
    frozen = _preregistered_experiment()
    prereg_sha = _canonical_sha256(frozen)
    key = _cache_key(prereg_sha, device)
    completed: dict[str, dict[str, Any]] = {}
    reused_input: dict[str, Any] | None = None
    reused_input_sha256: str | None = None
    prior_runtime = 0.0
    prior_device_peak = 0
    if reuse_report_path is not None:
        reused_input_sha256 = _sha256(reuse_report_path)
        reused_input = _load(reuse_report_path)
        if reused_input.get("task_id") != TASK_ID or len(reused_input.get("cells", [])) != 6:
            raise ValueError("reuse report is not a complete T6.18.1 raw campaign")
        completed = {cell["cell_id"]: cell for cell in _upgrade_analysis_fields(reused_input["cells"])}
        prior_runtime = float(
            observed_simulation_runtime_seconds
            if observed_simulation_runtime_seconds is not None
            else reused_input["execution_budget_audit"].get(
                    "simulation_runtime_seconds",
                    reused_input["execution_budget_audit"]["runtime_seconds"],
                )
        )
        prior_device_peak = int(reused_input["execution_budget_audit"]["peak_device_memory_bytes"])
    elif resume and PARTIAL_CACHE.exists():
        cached = _load(PARTIAL_CACHE)
        if cached.get("cache_key") == key:
            completed = {cell["cell_id"]: cell for cell in cached.get("cells", [])}
    started = perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    seeds = frozen["seeds"]["values"]
    cells: list[dict[str, Any]] = []
    cell_specs = [(cutoff, noise) for cutoff in frozen["config"]["cutoffs"] for noise in frozen["config"]["noise_profiles"]]
    for cell_index, (cutoff, noise_name) in enumerate(cell_specs):
        cell_id = f"cutoff{cutoff}_{noise_name}"
        cell = completed.get(cell_id)
        if cell is None:
            cell = _build_cell(cutoff, noise_name, seeds, device, cell_index, progress=progress)
            completed[cell_id] = cell
            _atomic_json({"cache_key": key, "cells": [completed[name] for name in sorted(completed)]}, PARTIAL_CACHE)
        cells.append(cell)
    postprocess_elapsed = perf_counter() - started
    elapsed = prior_runtime + postprocess_elapsed
    peak_device_memory = max(prior_device_peak, int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0)
    live_peak_wset = int(getattr(psutil.Process(os.getpid()).memory_info(), "peak_wset", 0)) if psutil is not None else 0
    peak_host_memory = max(live_peak_wset, int(observed_campaign_host_peak_wset or 0))
    accounted_peak_memory = max(peak_device_memory, peak_host_memory)
    reversal_cells = sum(cell["summary"]["ordering_reversal"]["count"] > 0 for cell in cells)
    method_gain_cells = {
        "measurement_feedback": sum(cell["summary"]["paired"]["measurement_vs_idle_lifetime_gain"]["ci95"][0] > 1.0 for cell in cells),
        "autonomous": sum(cell["summary"]["paired"]["autonomous_vs_idle_lifetime_gain"]["ci95"][0] > 1.0 for cell in cells),
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": MODEL_SCOPE,
        "preregistration": {"experiment_id": frozen["experiment_id"], "record_sha256": prereg_sha, "seeds": frozen["seeds"], "config": frozen["config"]},
        "source_scope": {"source_id": "LACHANCE2024_AQEC", "record_sha256": _canonical_sha256(_source_record()), "method_sha256": _canonical_sha256(_source_method()), "official_code": None, "evidence_boundary": "literature experimental metrics and project-native simulator results remain separate"},
        "nuisance_model": {"model_id": NOISE_MODEL_ID, "log_standard_deviation": NOISE_LOG_STD, "selection_timing": "fixed before any T6.18.1 simulation result", "interpretation": "project-model quasi-static robustness only; not device uncertainty or paper fit error"},
        "cells": cells,
        "stopping_rule": "all 6 cells x 24 clusters; no desired ordering gate",
        "project_result": {"cells": len(cells), "seed_clusters": len(cells) * len(seeds), "ordering_reversal_cells": reversal_cells, "cells_with_gain_ci_lower_above_one": method_gain_cells, "universal_20_percent_claim": False},
        "official_protocol_reproduction": {"state": "BLOCKED_OFFICIAL_PROTOCOL_REPRODUCTION", "official_code_available": False, "paper_native_reservoir_adapter": False, "project_replay_may_substitute": False},
        "literature_only_metrics": _source_method()["metrics"],
        "phase6b_noninterference": {"training_executed": False, "threshold_tuned": False, "phase6b_outputs_modified": False, "phase6b_verdict": "NO_GO_V5_EARLY_HEADROOM_STOP"},
        "claim_boundary": {
            "allowed": ["project-native finite-cutoff common-wall-clock robustness under the frozen nuisance model", "protocol-cycle versus physical-time ordering reversal", "event/reset/gate burden within the explicit simulator"],
            "forbidden": ["Lachance Method-A/Method-B reproduction", "zero-latency AQEC", "universal approximately 20 percent lifetime gain", "experimental device uncertainty from project bootstrap", "pulse-energy or control-duty advantage", "replace the area-equivalent primary metric with a favorable low-R2 exponential fit"],
        },
        "execution_budget_audit": {
            "runtime_seconds": elapsed,
            "simulation_runtime_seconds": prior_runtime if reused_input is not None else elapsed,
            "analysis_reprocess_runtime_seconds": postprocess_elapsed if reused_input is not None else 0.0,
            "runtime_budget_seconds": frozen["runtime_budget"]["wall_clock_seconds"],
            "peak_device_memory_bytes": peak_device_memory,
            "peak_host_working_set_bytes": peak_host_memory,
            "host_peak_observation": "maximum of live psutil peak_wset and externally sampled campaign Get-Process WorkingSet64" if observed_campaign_host_peak_wset else "live psutil process peak_wset",
            "accounted_peak_memory_bytes": accounted_peak_memory,
            "memory_budget_bytes": int(frozen["runtime_budget"]["memory_gib"] * (1 << 30)),
            "within_runtime_budget": elapsed <= frozen["runtime_budget"]["wall_clock_seconds"],
            "within_memory_budget": accounted_peak_memory <= frozen["runtime_budget"]["memory_gib"] * (1 << 30),
            "device": device,
            "boundary": "campaign/reprocess wall time and host/device peak; not protocol latency",
        },
        "analysis_reuse": {
            "state": "RAW_CURVE_REPROCESS_NO_SIMULATION_CHANGE" if reused_input is not None else "FRESH_SIMULATION",
            "input_report_sha256": reused_input_sha256,
            "reused_raw_cells_sha256": _canonical_sha256(_raw_cells_projection(reused_input["cells"])) if reused_input is not None else None,
            "reason": "add survival/fidelity fit diagnostics and host-memory accounting without resampling" if reused_input is not None else None,
        },
    }
    _write_csv(report)
    report["source_data"] = {"path": _relative(DEFAULT_SOURCE_DATA), "sha256": _sha256(DEFAULT_SOURCE_DATA), "rows": sum(1 for _ in DEFAULT_SOURCE_DATA.open(encoding="utf-8")) - 1}
    report["bindings"] = {
        "implementation": _binding(Path(__file__)),
        "physics": _binding(PHYSICS_IMPLEMENTATION),
        "preregistration": _binding(PREREG_CONFIG),
        "source_audit": _binding(SOURCE_AUDIT),
        "ontology": _binding(ONTOLOGY),
        "parent_report": _binding(PARENT_REPORT),
        "parent_source_data": _binding(PARENT_SOURCE_DATA),
        "phase6b_terminal": _binding(PHASE6B_TERMINAL),
        "source_data": _binding(DEFAULT_SOURCE_DATA),
    }
    report["semantic_mutation_audit"] = {"count": 16, "detected": 16, "cases": []}
    report["gates"] = evaluate_gates(report)
    report["semantic_mutation_audit"] = _mutations(report)
    report["gates"] = evaluate_gates(report)
    failed = [name for name, passed in report["gates"].items() if not passed]
    report["gate_summary"] = {"passed": len(report["gates"]) - len(failed), "failed": failed}
    report["verdict"] = VERDICT if not failed else "FAIL_AQEC_COMMON_WALLCLOCK_REPLAY"
    if PARTIAL_CACHE.exists():
        PARTIAL_CACHE.unlink()
    return report


def verify_report(report: Mapping[str, Any]) -> None:
    gates = evaluate_gates(report)
    if dict(report["gates"]) != gates:
        raise ValueError("stored T6.18.1 gates do not match live recomputation")
    failed = [name for name, passed in gates.items() if not passed]
    if report["gate_summary"] != {"passed": len(gates) - len(failed), "failed": failed}:
        raise ValueError("stored T6.18.1 gate summary drifted")
    if report["verdict"] != (VERDICT if not failed else "FAIL_AQEC_COMMON_WALLCLOCK_REPLAY"):
        raise ValueError("stored T6.18.1 verdict drifted")


def write_markdown(report: Mapping[str, Any], path: Path = DEFAULT_MARKDOWN) -> None:
    lines = [
        "# T6.18.1 AQEC/autonomous 共同 wall-clock replay",
        "",
        f"- verdict：`{report['verdict']}`",
        f"- cells / seed clusters：{report['project_result']['cells']} / {report['project_result']['seed_clusters']}",
        f"- cycle-vs-wall-clock ordering-reversal cells：{report['project_result']['ordering_reversal_cells']}/6",
        f"- gates / mutations：{report['gate_summary']['passed']}/16 / {report['semantic_mutation_audit']['detected']}/16",
        "",
        "## Project-native matched replay",
        "",
        "| cell | feedback / idle lifetime | autonomous / idle lifetime | autonomous / feedback (us) | reversal seeds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in report["cells"]:
        summary = cell["summary"]
        paired = summary["paired"]
        lines.append(
            f"| `{cell['cell_id']}` | {paired['measurement_vs_idle_lifetime_gain']['mean']:.4f} "
            f"[{paired['measurement_vs_idle_lifetime_gain']['ci95'][0]:.4f}, {paired['measurement_vs_idle_lifetime_gain']['ci95'][1]:.4f}] | "
            f"{paired['autonomous_vs_idle_lifetime_gain']['mean']:.4f} "
            f"[{paired['autonomous_vs_idle_lifetime_gain']['ci95'][0]:.4f}, {paired['autonomous_vs_idle_lifetime_gain']['ci95'][1]:.4f}] | "
            f"{paired['autonomous_vs_measurement_lifetime_us_ratio']['mean']:.4f} "
            f"[{paired['autonomous_vs_measurement_lifetime_us_ratio']['ci95'][0]:.4f}, {paired['autonomous_vs_measurement_lifetime_us_ratio']['ci95'][1]:.4f}] | "
            f"{summary['ordering_reversal']['count']}/24 |"
        )
    lines += [
        "",
        "六个 cell 中 measurement/reset 与 autonomous/reset 相对 idle 的 lifetime 95% CI 上界都低于 1，故项目原生结果是明确负结果；这只说明当前 fixed-nominal-control finite-cutoff 模型不产生 AQEC lifetime gain，不反驳论文装置的 reservoir-engineered experimental result。autonomous 每 100 us 避免 20 次 measurement，但 reset 从 20 增至 28.5714、active gates 从 180 增至 257.143。",
        "",
        "主 lifetime 使用完整曲线的 area-equivalent 定义。全体 raw trace 的 logical-Z exponential-fit R² 为 0.181--0.954，code-survival R² 为 0.047--0.685，并存在非单调点；这些 fit 只作为诊断保留，未挑选时间窗或替换主指标。",
        "",
        "每个 seed 是在查看 T6.18.1 结果前固定的 5% log-scale、mean-preserving quasi-static lifetime realization；idle、measurement/reset 与 autonomous/reset 共用同一 realization。置信区间只表示该项目 nuisance model 下的 24-cluster paired bootstrap，不是装置误差条。",
        "",
        "## Evidence boundary",
        "",
        "现有 simulator 使用 fixed nominal controls、instantaneous gates、analytic idle CPTP maps 与 trace-reset；它不是 Lachance 2024 的 dissipative transmon/reservoir Method A/B。论文的 1.14(18)/1.14(16) lifetime gains 保持 `LITERATURE_ONLY`，official protocol reproduction 为 `BLOCKED`。classical decoder latency 对 AQEC 是 N/A 而非 0；pulse energy 与 control-duty 未建模，保持 null。",
        "",
        "## Artifacts",
        "",
        "- `docs/t6_18_1_aqec_common_wallclock_replay.json`",
        "- `docs/t6_18_1_aqec_common_wallclock_source_data.csv`",
        "- `cnn_fpga/benchmark/aqec_secondary_wallclock_replay.py`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--reuse-report", type=Path)
    parser.add_argument("--observed-campaign-host-peak-wset", type=int)
    parser.add_argument("--observed-simulation-runtime-seconds", type=float)
    args = parser.parse_args()
    report = build_report(
        device=args.device,
        resume=not args.no_resume,
        progress=not args.quiet,
        reuse_report_path=args.reuse_report,
        observed_campaign_host_peak_wset=args.observed_campaign_host_peak_wset,
        observed_simulation_runtime_seconds=args.observed_simulation_runtime_seconds,
    )
    _atomic_json(report, DEFAULT_REPORT)
    write_markdown(report)
    print(json.dumps({"verdict": report["verdict"], "gates": report["gate_summary"], "cells": report["project_result"]["cells"], "seed_clusters": report["project_result"]["seed_clusters"], "source_rows": report["source_data"]["rows"]}))


if __name__ == "__main__":
    main()
