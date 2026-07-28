"""Outcome-free T04 statistical preregistration preflight.

This module deliberately does not import, inspect, or execute a T04 formal
writer.  It performs four design-only operations:

1. expand the complete old-seven-family plus bounded-repair gate union;
2. qualify the frozen paired-density multiplier UCB at dimensions 120/132;
3. construct one correlated, studentized, closed-family maxT critical value
   and test its global-IUT power without deleting difficult gates; and
4. forecast the full 518-chunk T04 resource envelope.

No physics, LER, lifetime, hardware, official-comparator or SOTA claim is
produced here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from cnn_fpga.benchmark.phase9_paired_cluster_uq_hardened_confirmation import (
    _physical_density_trial,
    paired_density_trace_ucb_physical,
)


TASK_ID = "T-RISK-20260728-05"
CONFIG_PATH = "configs/phase9/t_risk_20260728_05_highdim_joint_maxt_preflight.json"
PASS_VERDICT = "PASS_T04_STATISTICAL_PREREGISTRATION_RELEASED"
FAIL_VERDICT = "FAIL_T04_STATISTICAL_PREREGISTRATION_BLOCKED"
REPORT_SCHEMA = "PHASE9-HIGHDIM-JOINT-MAXT-PREFLIGHT-REPORT-V1"
BLUEPRINT_SCHEMA = "PHASE9-T04-FULL-GATE-BLUEPRINT-V1"
SOURCE_FIELDS = (
    "row_type",
    "cell_id",
    "family",
    "dimension",
    "cluster_count",
    "true_distance",
    "trial",
    "trial_seed",
    "multiplier_seed",
    "estimate",
    "raw_radius",
    "upper_bound",
    "covered",
    "equivalence_pass",
    "replicate",
    "max_statistic",
    "case",
    "effect_ratio",
    "successes",
    "trials",
    "point",
    "wilson_lcb",
    "wilson_ucb",
    "gate_pass",
)


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
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n")


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("analysis_sha256", None)
    return _sha(body)


def _higher_quantile(values: np.ndarray, confidence: float) -> float:
    return float(np.quantile(values, confidence, method="higher"))


def _normal_quantile(probability: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(probability)


def _wilson(
    successes: int,
    trials: int,
    *,
    confidence: float,
    comparisons: int = 1,
) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials or comparisons < 1:
        raise ValueError("invalid Wilson inputs")
    alpha = 1.0 - confidence
    z = _normal_quantile(1.0 - alpha / (2.0 * comparisons))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class Gate:
    gate_id: str
    family: str
    source_contract: str
    stage: str
    metric: str
    margin: float
    direction: str
    normalized_sd: float
    cluster_count: int
    cluster_scope: str
    deterministic: bool = False


def _old_gate(
    old: Mapping[str, Any],
    matrix: Mapping[str, Any],
    *,
    gate_id: str,
    family: str,
    metric: str,
    cluster_scope: str,
    deterministic: bool = False,
) -> Gate:
    family_spec = old["families"][family]
    margin_spec = old["margins"][metric]
    stage = str(family_spec["stage"])
    if deterministic:
        count = 0
    elif stage == "trajectory":
        count = (
            int(matrix["fault_clusters_per_state"])
            * int(matrix["fault_states_per_cell"])
        )
    else:
        count = int(matrix["round_clusters_per_cell"])
    return Gate(
        gate_id=gate_id,
        family=family,
        source_contract="old_seven_family",
        stage=stage,
        metric=metric,
        margin=float(margin_spec["value"]),
        direction=str(margin_spec.get("direction", "upper")),
        normalized_sd=float(family_spec["normalized_sd_envelope"]),
        cluster_count=count,
        cluster_scope=cluster_scope,
        deterministic=deterministic,
    )


def build_blueprint(
    config: Mapping[str, Any],
    old: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> list[Gate]:
    """Materialize the exact all-required T04 gate union."""

    m = config["formal_matrix"]
    cutoffs = [int(value) for value in m["cutoffs"]]
    states = list(m["shared_fock_states"])
    labels = list(m["logical_labels"])
    actions = list(m["actions"])
    faults = list(m["fault_scenarios"])
    probes = list(m["representative_probes"])
    gates: list[Gate] = []

    def add_old(
        gate_id: str,
        family: str,
        metric: str,
        scope: str,
        deterministic: bool = False,
    ) -> None:
        gates.append(_old_gate(
            old, m, gate_id=gate_id, family=family, metric=metric,
            cluster_scope=scope, deterministic=deterministic,
        ))

    physical = old["families"]["physical_state_channel"]["metrics"]
    iq = old["families"]["iq_conditional_distribution"]["metrics"]
    likelihood = old["families"]["likelihood_score_posterior"]["metrics"]
    fault_metrics = old["families"]["fault_trajectory_tail"]["metrics"]
    for cutoff in cutoffs:
        for state in states:
            for action in actions:
                scope = f"round/c{cutoff}/shared/{state}/{action}"
                prefix = f"ab/c{cutoff}/shared/{state}/{action}"
                for metric in physical:
                    add_old(f"{prefix}/{metric}", "physical_state_channel", metric, scope)
                for metric in iq:
                    add_old(f"{prefix}/{metric}", "iq_conditional_distribution", metric, scope)
                for metric in likelihood:
                    add_old(
                        f"{prefix}/{metric}", "likelihood_score_posterior", metric, scope
                    )
        for label in labels:
            for action in actions:
                scope = f"round/c{cutoff}/logical/{label}/{action}"
                add_old(
                    f"ab/c{cutoff}/logical/{label}/{action}/logical_survival",
                    "logical_ptm_survival", "logical_survival", scope,
                )
        for action in actions:
            add_old(
                f"ab/c{cutoff}/logical/PTM/{action}/logical_ptm_entry",
                "logical_ptm_survival", "logical_ptm_entry",
                f"round/c{cutoff}/logical/PTM/{action}",
            )
        for state in states:
            scope = f"round/c{cutoff}/reset/{state}"
            for metric in ("rao_blackwell_reset_success", "leakage_residence_rate"):
                add_old(
                    f"ab/c{cutoff}/reset/{state}/{metric}",
                    "reset_leakage", metric, scope,
                )
        for scenario in faults:
            scope = f"trajectory/c{cutoff}/fault/{scenario}/all_states"
            for metric in fault_metrics:
                add_old(
                    f"ab/c{cutoff}/fault/{scenario}/{metric}",
                    "fault_trajectory_tail", metric, scope,
                )
        add_old(
            f"mapping/c{cutoff}/principal_singular", "cutoff_mapping",
            "principal_singular", f"deterministic/c{cutoff}", True,
        )
        add_old(
            f"mapping/c{cutoff}/projector_frobenius", "cutoff_mapping",
            "projector_frobenius", f"deterministic/c{cutoff}", True,
        )

    probe_cutoff = int(m["probe_cutoff"])
    for probe in probes:
        scope = f"round/c{probe_cutoff}/probe/{probe}"
        prefix = f"ab/c{probe_cutoff}/probe/{probe}"
        for metric in physical:
            add_old(f"{prefix}/{metric}", "physical_state_channel", metric, scope)
        for metric in iq:
            add_old(f"{prefix}/{metric}", "iq_conditional_distribution", metric, scope)
        for metric in likelihood:
            add_old(
                f"{prefix}/{metric}", "likelihood_score_posterior", metric, scope
            )
    for probe in ("P09_LEAK_RESET", "P10_RESET_OK", "P11_RESET_FAIL"):
        for metric in ("rao_blackwell_reset_success", "leakage_residence_rate"):
            add_old(
                f"ab/c{probe_cutoff}/probe/{probe}/{metric}",
                "reset_leakage", metric,
                f"round/c{probe_cutoff}/probe/{probe}",
            )
    for lower, upper in m["primary_cutoff_increments"]:
        for state in states:
            for action in m["cutoff_actions"]:
                add_old(
                    f"cutoff/{lower}-{upper}/shared/{state}/{action}/density",
                    "cutoff_mapping", "cutoff_density_trace_distance",
                    f"round/cutoff/{lower}-{upper}/shared/{state}/{action}",
                )
        for label in labels:
            for action in m["cutoff_actions"]:
                add_old(
                    f"cutoff/{lower}-{upper}/logical/{label}/{action}/survival",
                    "cutoff_mapping", "cutoff_logical_survival",
                    f"round/cutoff/{lower}-{upper}/logical/{label}/{action}",
                )
        for scenario in faults:
            add_old(
                f"cutoff/{lower}-{upper}/fault/{scenario}/terminal_density",
                "cutoff_mapping", "cutoff_density_trace_distance",
                f"trajectory/cutoff/{lower}-{upper}/fault/{scenario}",
            )
            add_old(
                f"cutoff/{lower}-{upper}/fault/{scenario}/survival",
                "cutoff_mapping", "cutoff_logical_survival",
                f"trajectory/cutoff/{lower}-{upper}/fault/{scenario}",
            )

    # Bounded-repair estimands are deliberately state-conditioned and remain
    # additional gates; the older aggregate trajectory gates do not replace them.
    contract = repair["diagnostic_contract"]
    margins = contract["margins"]
    state_schedule = list(repair["logical_state_schedule"])
    scalar_margin_keys = {
        "mean_photon": "cutoff_terminal_mean_photon_difference",
        "level_probability_l1": "cutoff_terminal_level_probability_l1",
        "logical_survival": "cutoff_terminal_logical_survival_difference",
    }

    def add_repair(
        gate_id: str,
        family: str,
        metric: str,
        margin: float,
        scope: str,
        *,
        stage: str,
        clusters: int,
    ) -> None:
        gates.append(Gate(
            gate_id=gate_id,
            family=family,
            source_contract="t03_bounded_repair",
            stage=stage,
            metric=metric,
            margin=float(margin),
            direction="upper",
            normalized_sd=2.0,
            cluster_count=int(clusters),
            cluster_scope=scope,
            deterministic=False,
        ))

    fault_clusters = int(m["fault_clusters_per_state"])
    for lower, upper in m["primary_cutoff_increments"]:
        for scenario in faults:
            for backend in ("A", "B"):
                for state in state_schedule:
                    scope = f"trajectory/{scenario}/{backend}/{state}"
                    add_repair(
                        f"fault/density/{lower}->{upper}/{scenario}/{backend}/{state}",
                        "fault_density", "density_trace_distance",
                        float(contract["design_density_point_threshold"]), scope,
                        stage="terminal", clusters=fault_clusters,
                    )
                    for stage_name in repair["stage_partition"][scenario]:
                        for metric, margin_key in scalar_margin_keys.items():
                            add_repair(
                                f"fault/scalar/{metric}/{lower}->{upper}/"
                                f"{scenario}/{backend}/{state}/{stage_name}",
                                "fault_scalar", metric, float(margins[margin_key]),
                                scope, stage=str(stage_name), clusters=fault_clusters,
                            )
    terminal = int(repair["absolute_tail_cutoff"])
    tail_metrics = (
        "top1_fock_mass", "top2_fock_mass", "top4_fock_mass",
        "normalized_mean_photon", "commutator_defect",
    )
    for scenario in faults:
        for backend in ("A", "B"):
            for state in state_schedule:
                scope = f"trajectory/{scenario}/{backend}/{state}"
                for metric in tail_metrics:
                    add_repair(
                        f"fault/tail/{metric}/c{terminal}/{scenario}/{backend}/{state}",
                        "fault_absolute_tail", metric,
                        float(margins[f"absolute_terminal_{metric}"]), scope,
                        stage="terminal", clusters=fault_clusters,
                    )

    shared_scope = "round/shared_vacuum_f_expected_reset"
    round_clusters = int(m["round_clusters_per_cell"])
    for lower, upper in m["primary_cutoff_increments"]:
        for backend in ("A", "B"):
            add_repair(
                f"shared/density/{lower}->{upper}/{backend}", "shared_density",
                "density_trace_distance",
                float(margins["cutoff_terminal_density_trace_distance"]),
                shared_scope, stage="terminal", clusters=round_clusters,
            )
            for metric, margin_key in scalar_margin_keys.items():
                add_repair(
                    f"shared/scalar/{metric}/{lower}->{upper}/{backend}",
                    "shared_scalar", metric, float(margins[margin_key]),
                    shared_scope, stage="terminal", clusters=round_clusters,
                )
    ab_margin_keys = {
        "mean_photon": "ab_terminal_mean_photon_difference",
        "level_probability_l1": "ab_terminal_level_probability_l1",
        "logical_survival": "ab_terminal_logical_survival_difference",
    }
    for cutoff in cutoffs:
        add_repair(
            f"shared/density/ab/c{cutoff}", "shared_density",
            "density_trace_distance",
            float(margins["ab_terminal_density_trace_distance"]),
            shared_scope, stage="terminal", clusters=round_clusters,
        )
        for metric, margin_key in ab_margin_keys.items():
            add_repair(
                f"shared/scalar/{metric}/ab/c{cutoff}", "shared_scalar",
                metric, float(margins[margin_key]), shared_scope,
                stage="terminal", clusters=round_clusters,
            )
    for backend in ("A", "B"):
        for metric in tail_metrics:
            add_repair(
                f"shared/tail/{metric}/c{terminal}/{backend}",
                "shared_absolute_tail", metric,
                float(margins[f"absolute_terminal_{metric}"]), shared_scope,
                stage="terminal", clusters=round_clusters,
            )

    identifiers = [gate.gate_id for gate in gates]
    if len(identifiers) != len(set(identifiers)):
        duplicate = next(value for value in identifiers if identifiers.count(value) > 1)
        raise RuntimeError(f"duplicate gate ID: {duplicate}")
    gates.sort(key=lambda item: item.gate_id)
    return gates


def _density_cells(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    design = config["density_uq"]
    cells: list[dict[str, Any]] = []
    index = 0
    for family in sorted(design["families"]):
        for dimension in design["dimensions"]:
            for effect in design["true_trace_distances"]:
                cells.append({
                    "cell_index": index,
                    "cell_id": f"{family}__d{dimension}__n{design['clusters_per_state']}"
                    f"__effect_{float(effect):.3f}",
                    "family": family,
                    "dimension": int(dimension),
                    "cluster_count": int(design["clusters_per_state"]),
                    "true_distance": float(effect),
                })
                index += 1
    if len(cells) != 24:
        raise RuntimeError("density cell denominator drift")
    return cells


def _simulate_density_cell(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = payload["config"]
    cell = payload["cell"]
    design = config["density_uq"]
    trials = int(design["trial_count_per_cell"])
    records: list[dict[str, Any]] = []
    family = design["families"][cell["family"]]
    for trial in range(trials):
        address = int(cell["cell_index"]) * trials + trial
        trial_seed = int(design["trial_seed_base"]) + address
        multiplier_seed = int(design["multiplier_seed_base"]) + address
        left, right, truth = _physical_density_trial(
            dimension=int(cell["dimension"]),
            count=int(cell["cluster_count"]),
            true_distance=float(cell["true_distance"]),
            family=family,
            seed=trial_seed,
        )
        ucb = paired_density_trace_ucb_physical(
            left,
            right,
            confidence=float(design["confidence"]),
            multiplier_replicates=int(design["multiplier_replicates"]),
            seed=multiplier_seed,
            calibration_factor=float(design["calibration_factor"]),
        )
        records.append({
            "row_type": "density_trial",
            "cell_id": cell["cell_id"],
            "family": cell["family"],
            "dimension": int(cell["dimension"]),
            "cluster_count": int(cell["cluster_count"]),
            "true_distance": float(truth),
            "trial": trial,
            "trial_seed": trial_seed,
            "multiplier_seed": multiplier_seed,
            "estimate": ucb.estimate,
            "raw_radius": ucb.raw_radius,
            "upper_bound": ucb.upper_bound,
            "covered": bool(ucb.upper_bound + 1e-15 >= truth),
            "equivalence_pass": bool(
                ucb.upper_bound <= float(design["equivalence_margin"])
            ),
        })
    return {"cell": dict(cell), "records": records}


def _valid_chunk(
    path: Path,
    *,
    config_sha: str,
    cell: Mapping[str, Any],
    trial_count: int,
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        value = _load_json(path)
        if (
            value.get("task_id") != TASK_ID
            or value.get("config_analysis_sha256") != config_sha
            or value.get("cell") != dict(cell)
            or value.get("record_count") != trial_count
            or len(value.get("records", [])) != trial_count
            or value.get("analysis_sha256") != _self_hash(value)
        ):
            return None
        return list(value["records"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def run_density(
    root: Path,
    config: Mapping[str, Any],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    paths = config["artifact_paths"]
    chunk_dir = root / paths["chunk_directory"]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    config_sha = _sha(config)
    trial_count = int(config["density_uq"]["trial_count_per_cell"])
    cells = _density_cells(config)
    all_records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for cell in cells:
        path = chunk_dir / f"{cell['cell_id']}.json"
        records = _valid_chunk(
            path, config_sha=config_sha, cell=cell, trial_count=trial_count
        )
        if records is None:
            missing.append(cell)
        else:
            all_records.extend(records)

    started = time.time()
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_simulate_density_cell, {
                    "config": dict(config), "cell": cell,
                }): cell
                for cell in missing
            }
            completed = len(cells) - len(missing)
            for future in as_completed(futures):
                result = future.result()
                cell = result["cell"]
                records = result["records"]
                chunk: dict[str, Any] = {
                    "task_id": TASK_ID,
                    "schema_version": "PHASE9-HIGHDIM-DENSITY-CHUNK-V1",
                    "config_analysis_sha256": config_sha,
                    "cell": cell,
                    "record_count": len(records),
                    "records": records,
                }
                chunk["analysis_sha256"] = _self_hash(chunk)
                _atomic_json(chunk_dir / f"{cell['cell_id']}.json", chunk)
                all_records.extend(records)
                completed += 1
                heartbeat: dict[str, Any] = {
                    "task_id": TASK_ID,
                    "phase": "density_uq",
                    "completed_cells": completed,
                    "total_cells": len(cells),
                    "record_count": len(all_records),
                    "elapsed_seconds": time.time() - started,
                    "fresh_unix": time.time(),
                }
                heartbeat["analysis_sha256"] = _self_hash(heartbeat)
                _atomic_json(root / paths["heartbeat"], heartbeat)

    expected = len(cells) * trial_count
    if len(all_records) != expected:
        raise RuntimeError(f"density row denominator drift: {len(all_records)} != {expected}")
    seeds = [int(row["trial_seed"]) for row in all_records]
    multiplier = [int(row["multiplier_seed"]) for row in all_records]
    if len(seeds) != len(set(seeds)) or len(multiplier) != len(set(multiplier)):
        raise RuntimeError("density seed address collision")
    return sorted(
        all_records,
        key=lambda row: (str(row["cell_id"]), int(row["trial"])),
    )


def summarize_density(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    design = config["density_uq"]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_id"]), []).append(row)
    comparisons = int(design["simultaneous_wilson"]["comparisons"])
    confidence = float(design["simultaneous_wilson"]["confidence"])
    gates = design["gates"]
    summaries: list[dict[str, Any]] = []
    decisions: dict[str, bool] = {}
    for cell in _density_cells(config):
        cell_rows = grouped[cell["cell_id"]]
        trials = len(cell_rows)
        coverage = sum(bool(row["covered"]) for row in cell_rows)
        equivalence = sum(bool(row["equivalence_pass"]) for row in cell_rows)
        coverage_lcb, coverage_ucb = _wilson(
            coverage, trials, confidence=confidence, comparisons=comparisons
        )
        eq_lcb, eq_ucb = _wilson(
            equivalence, trials, confidence=confidence, comparisons=comparisons
        )
        effect = float(cell["true_distance"])
        coverage_pass = (
            coverage / trials >= float(gates["minimum_cell_coverage_rate"])
            and coverage_lcb >= float(gates["minimum_cell_coverage_wilson_lcb"])
        )
        if effect == 0.0:
            power_pass = eq_lcb >= float(gates["null_equivalence_wilson_lcb"])
        elif effect == 0.05:
            power_pass = (
                eq_lcb >= float(gates["local_005_equivalence_wilson_lcb"])
            )
        elif effect == 0.1:
            power_pass = (
                eq_ucb <= float(gates["boundary_equivalence_wilson_ucb"])
            )
        elif effect == 0.12:
            power_pass = (
                eq_ucb <= float(gates["outside_equivalence_wilson_ucb"])
            )
        else:
            raise RuntimeError("unregistered density effect")
        decisions[f"density_coverage::{cell['cell_id']}"] = coverage_pass
        decisions[f"density_power::{cell['cell_id']}"] = power_pass
        summaries.append({
            **cell,
            "trials": trials,
            "coverage_successes": coverage,
            "coverage_rate": coverage / trials,
            "coverage_wilson_lcb": coverage_lcb,
            "coverage_wilson_ucb": coverage_ucb,
            "equivalence_successes": equivalence,
            "equivalence_rate": equivalence / trials,
            "equivalence_wilson_lcb": eq_lcb,
            "equivalence_wilson_ucb": eq_ucb,
            "coverage_gate_pass": coverage_pass,
            "power_gate_pass": power_pass,
        })
    return summaries, decisions


def _scope_token(gate: Gate) -> str:
    return gate.cluster_scope


def _standardized_influences(
    gates: Sequence[Gate],
    *,
    count: int,
    family_rho: float,
    scope_rho: float,
    seed: int,
) -> np.ndarray:
    selected = [gate for gate in gates if gate.cluster_count == count]
    families = sorted({gate.family for gate in selected})
    scopes = sorted({_scope_token(gate) for gate in selected})
    rng = np.random.default_rng(seed + count)
    family_values = {
        key: rng.standard_normal(count) for key in families
    }
    scope_values = {
        key: rng.standard_normal(count) for key in scopes
    }
    matrix = np.empty((count, len(selected)), dtype=np.float64)
    residual_rho = 1.0 - family_rho - scope_rho
    if residual_rho <= 0.0:
        raise ValueError("correlation components must sum below one")
    for index, gate in enumerate(selected):
        values = (
            math.sqrt(family_rho) * family_values[gate.family]
            + math.sqrt(scope_rho) * scope_values[_scope_token(gate)]
            + math.sqrt(residual_rho) * rng.standard_normal(count)
        )
        values -= float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        if not math.isfinite(sd) or sd <= 0.0:
            raise RuntimeError("degenerate maxT influence")
        matrix[:, index] = values / sd
    return matrix


def build_joint_maxt(
    config: Mapping[str, Any],
    blueprint: Sequence[Gate],
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    contract = config["joint_maxt"]
    stochastic = [gate for gate in blueprint if not gate.deterministic]
    correlation = contract["correlation_model"]
    b = int(contract["multiplier_replicates"])
    replicate_max = np.zeros(b, dtype=np.float64)
    influence_fingerprints: dict[str, str] = {}
    for count in sorted({gate.cluster_count for gate in stochastic}):
        selected = [gate for gate in stochastic if gate.cluster_count == count]
        influence = _standardized_influences(
            stochastic,
            count=count,
            family_rho=float(correlation["family_shared"]),
            scope_rho=float(correlation["scope_shared"]),
            seed=int(contract["influence_seed_base"]),
        )
        rng = np.random.default_rng(int(contract["rademacher_seed_base"]) + count)
        weights = rng.integers(0, 2, size=(b, count), dtype=np.int8)
        signs = weights.astype(np.float64) * 2.0 - 1.0
        standardized = signs @ influence / math.sqrt(count)
        replicate_max = np.maximum(
            replicate_max, np.max(np.abs(standardized), axis=1)
        )
        influence_fingerprints[str(count)] = _sha({
            "shape": list(influence.shape),
            "head": influence[:4, :4].tolist(),
            "selected_gate_ids": [gate.gate_id for gate in selected],
        })
    critical = (
        float(contract["calibration_factor"])
        * _higher_quantile(replicate_max, float(contract["confidence"]))
    )
    max_rows = [
        {
            "row_type": "maxt_replicate",
            "replicate": index,
            "max_statistic": float(value),
        }
        for index, value in enumerate(replicate_max)
    ]

    power = contract["power"]
    trials = int(power["pseudoexperiments"])
    rng = np.random.default_rng(int(contract["power_seed_base"]))
    families = sorted({gate.family for gate in stochastic})
    scopes = sorted({_scope_token(gate) for gate in stochastic})
    family_index = {value: index for index, value in enumerate(families)}
    scope_index = {value: index for index, value in enumerate(scopes)}
    family_noise = rng.standard_normal((trials, len(families)))
    scope_noise = rng.standard_normal((trials, len(scopes)))
    passed = np.empty((trials, len(stochastic)), dtype=np.bool_)
    estimates = np.empty((trials, len(stochastic)), dtype=np.float32)
    family_rho = float(correlation["family_shared"])
    scope_rho = float(correlation["scope_shared"])
    residual_rho = float(correlation["idiosyncratic"])
    for index, gate in enumerate(stochastic):
        se_ratio = gate.normalized_sd * math.sqrt(2.0 / gate.cluster_count)
        noise = (
            math.sqrt(family_rho) * family_noise[:, family_index[gate.family]]
            + math.sqrt(scope_rho) * scope_noise[:, scope_index[_scope_token(gate)]]
            + math.sqrt(residual_rho) * rng.standard_normal(trials)
        )
        estimate = se_ratio * noise
        estimates[:, index] = estimate
        passed[:, index] = np.abs(estimate) + critical * se_ratio <= 1.0
    fail_counts = np.sum(~passed, axis=1)
    null_success = int(np.count_nonzero(fail_counts == 0))
    confidence = float(power["confidence_for_bounds"])
    power_rows: list[dict[str, Any]] = []

    def row(
        case: str, family: str, effect_ratio: float, successes: int
    ) -> dict[str, Any]:
        lower, upper = _wilson(
            successes, trials, confidence=confidence, comparisons=1
        )
        return {
            "row_type": "maxt_power",
            "case": case,
            "family": family,
            "effect_ratio": effect_ratio,
            "successes": successes,
            "trials": trials,
            "point": successes / trials,
            "wilson_lcb": lower,
            "wilson_ucb": upper,
        }

    null_row = row("null_global_equivalence", "all", 0.0, null_success)
    null_row["gate_pass"] = (
        null_row["wilson_lcb"]
        >= float(power["null_global_equivalence_lcb_minimum"])
    )
    power_rows.append(null_row)
    decisions: dict[str, bool] = {"maxt_null_global_power": bool(null_row["gate_pass"])}
    selected_injections: dict[str, str] = {}
    for family in families:
        candidates = [
            (index, gate) for index, gate in enumerate(stochastic)
            if gate.family == family
        ]
        injection_index, injection_gate = sorted(
            candidates,
            key=lambda item: (
                -item[1].normalized_sd * math.sqrt(2.0 / item[1].cluster_count),
                item[1].gate_id,
            ),
        )[0]
        selected_injections[family] = injection_gate.gate_id
        other_pass = fail_counts - (~passed[:, injection_index]).astype(np.int64) == 0
        se_ratio = (
            injection_gate.normalized_sd
            * math.sqrt(2.0 / injection_gate.cluster_count)
        )
        base_estimate = estimates[:, injection_index].astype(np.float64)
        for case, ratio, bound_key, lower_required in (
            (
                "local_single_gate_equivalence",
                float(power["local_effect_ratio"]),
                "local_single_gate_equivalence_lcb_minimum",
                True,
            ),
            (
                "boundary_false_equivalence",
                float(power["boundary_effect_ratio"]),
                "boundary_false_equivalence_ucb_maximum",
                False,
            ),
            (
                "outside_false_equivalence",
                float(power["outside_effect_ratio"]),
                "outside_false_equivalence_ucb_maximum",
                False,
            ),
        ):
            injected_pass = (
                np.abs(base_estimate + ratio) + critical * se_ratio <= 1.0
            )
            successes = int(np.count_nonzero(other_pass & injected_pass))
            power_row = row(case, family, ratio, successes)
            if lower_required:
                power_row["gate_pass"] = (
                    power_row["wilson_lcb"] >= float(power[bound_key])
                )
            else:
                power_row["gate_pass"] = (
                    power_row["wilson_ucb"] <= float(power[bound_key])
                )
            power_row["injection_gate_id"] = injection_gate.gate_id
            power_rows.append(power_row)
            decisions[f"maxt_power::{case}::{family}"] = bool(
                power_row["gate_pass"]
            )
    decisions["maxt_critical_finite_and_above_pointwise"] = (
        math.isfinite(critical) and critical > 1.6448536269514722
    )
    meta = [{
        "row_type": "maxt_meta",
        "case": "critical",
        "point": critical,
        "family": "all",
        "trials": b,
        "gate_pass": decisions["maxt_critical_finite_and_above_pointwise"],
        "influence_fingerprints": influence_fingerprints,
        "selected_injections": selected_injections,
    }]
    return critical, max_rows + meta, power_rows, decisions


def resource_forecast(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    contract = config["resource_forecast"]
    parent_path = root / config["parent_contracts"]["t03_resource_evidence"]
    parent = _load_json(parent_path)
    parent_rows = 21168
    parent_estimated_wall = float(parent["estimated_wall_seconds_with_safety_factor"])
    parent_estimated_artifact = float(parent["estimated_artifact_bytes"])
    rows = int(contract["full_t04_rows"])
    row_ratio = rows / parent_rows
    worker_candidates: list[dict[str, Any]] = []
    for workers in contract["max_workers_candidates"]:
        estimated_wall = (
            parent_estimated_wall
            * row_ratio
            * 2.0
            / int(workers)
        )
        worker_candidates.append({
            "workers": int(workers),
            "blas_threads_per_worker": int(contract["blas_threads_per_worker"]),
            "estimated_wall_seconds": estimated_wall,
            "wall_gate_pass": (
                estimated_wall <= float(contract["maximum_estimated_wall_seconds"])
            ),
        })
    selected = next(
        (item for item in reversed(worker_candidates) if item["wall_gate_pass"]),
        None,
    )
    estimated_artifact = (
        parent_estimated_artifact
        * row_ratio
        * float(contract["artifact_safety_factor"])
    )
    dimension = int(config["formal_matrix"]["cutoff_dimensions"]["44"])
    clusters = int(config["formal_matrix"]["round_clusters_per_cell"])
    # Four concurrent complex128 density stacks cover primary/conditional RESET
    # compression plus one scratch stack.  The factor is intentionally larger
    # than T03 endpoint RSS and is checked again by fresh T04 benchmarks.
    per_worker_density_peak = 5 * clusters * dimension * dimension * 16
    estimated_rss = (
        int(selected["workers"]) * per_worker_density_peak
        * float(contract["rss_safety_factor"])
        if selected is not None else math.inf
    )
    free_disk = shutil.disk_usage(root).free
    decisions = {
        "resource_parent_preflight_live_pass": parent.get("passed") is True,
        "resource_worker_candidate_exists": selected is not None,
        "resource_artifact_under_limit": (
            estimated_artifact
            <= float(contract["maximum_estimated_artifact_bytes"])
        ),
        "resource_rss_under_limit": (
            estimated_rss <= float(contract["maximum_estimated_total_rss_bytes"])
        ),
        "resource_disk_headroom": (
            free_disk - estimated_artifact
            >= float(contract["minimum_free_disk_bytes"])
        ),
        "resource_t04_fresh_benchmark_still_required": (
            contract["t04_must_run_fresh_resource_benchmarks"] is True
            and contract["forecast_does_not_authorize_scientific_chunk"] is True
        ),
    }
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-T04-RESOURCE-FORECAST-V1",
        "parent_binding": _binding(parent_path, root),
        "parent_rows": parent_rows,
        "full_t04_rows": rows,
        "row_ratio": row_ratio,
        "worker_candidates": worker_candidates,
        "selected_candidate": selected,
        "estimated_artifact_bytes": int(math.ceil(estimated_artifact)),
        "estimated_total_rss_bytes": int(math.ceil(estimated_rss)),
        "free_disk_bytes": int(free_disk),
        "limits": {
            "wall_seconds": int(contract["maximum_estimated_wall_seconds"]),
            "artifact_bytes": int(contract["maximum_estimated_artifact_bytes"]),
            "rss_bytes": int(contract["maximum_estimated_total_rss_bytes"]),
            "minimum_free_disk_bytes": int(contract["minimum_free_disk_bytes"]),
        },
        "fresh_t04_resource_preflight_required": True,
        "gates": decisions,
        "passed": all(decisions.values()),
        "qualified_claim": None,
    }
    report["analysis_sha256"] = _self_hash(report)
    return report, decisions


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=SOURCE_FIELDS, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in SOURCE_FIELDS})
    return buffer.getvalue().encode("utf-8")


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("task_id") != TASK_ID:
        raise ValueError("task ID drift")
    access = config["formal_outcome_access"]
    if (
        access["t04_run_exists"] is not False
        or access["t04_formal_outcomes_accessed"] is not False
        or access["outcomes_may_change_factor_margin_family_seed_count_or_quantile"]
        is not False
    ):
        raise ValueError("formal outcome firewall is open")
    d = config["density_uq"]
    if (
        d["dimensions"] != [120, 132]
        or int(d["clusters_per_state"]) != 384
        or int(d["multiplier_replicates"]) != 199
        or float(d["calibration_factor"]) != 1.0
        or d["quantile_method"] != "higher"
    ):
        raise ValueError("frozen high-dimensional UQ contract drift")
    j = config["joint_maxt"]
    if (
        int(j["multiplier_replicates"]) != 199
        or float(j["calibration_factor"]) != 1.0
        or j["quantile_method"] != "higher"
        or j["aggregate_rescue"] is not False
        or j["gate_deletion"] is not False
        or j["cross_state_averaging"] is not False
        or j["pointwise_z_substitution"] is not False
    ):
        raise ValueError("joint maxT contract drift")
    if config["formal_matrix"]["cell_accounting"] != {
        "shared_chunks": 210,
        "logical_chunks": 252,
        "fault_chunks": 24,
        "probe_chunks": 32,
        "total_chunks": 518,
        "primary_rows": 1042944,
    }:
        raise ValueError("formal matrix denominator drift")
    claims = config["claim_boundary"]
    for key in (
        "twin_qualification", "ler", "lifetime", "physical_break_even",
        "official_puviani_exact", "puviani_nmf_surpass", "external_sota",
        "hardware_measured",
    ):
        if claims.get(key) is not None:
            raise ValueError(f"prohibited claim populated: {key}")


def write_artifacts(
    root: Path | None = None,
    *,
    workers: int = 4,
) -> dict[str, Any]:
    base = (root or _root()).resolve()
    config_path = base / CONFIG_PATH
    config = _load_json(config_path)
    _validate_config(config)
    if workers not in config["resource_forecast"]["max_workers_candidates"]:
        raise ValueError("workers outside preregistered candidates")
    paths = config["artifact_paths"]
    lock_path = base / paths["run_directory"] / "supervisor.owner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError("another T05 supervisor may be active") from exc
    with lock_handle:
        lock_handle.write(json.dumps({"pid": os.getpid(), "started": time.time()}))
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
    try:
        old_path = base / config["parent_contracts"]["old_seven_family_design"]
        repair_path = base / config["parent_contracts"]["bounded_repair"]
        old = _load_json(old_path)
        repair = _load_json(repair_path)
        blueprint = build_blueprint(config, old, repair)
        blueprint_rows = [asdict(gate) for gate in blueprint]
        family_counts = {
            family: sum(gate.family == family for gate in blueprint)
            for family in sorted({gate.family for gate in blueprint})
        }
        source_counts = {
            source: sum(gate.source_contract == source for gate in blueprint)
            for source in ("old_seven_family", "t03_bounded_repair")
        }
        deterministic = sum(gate.deterministic for gate in blueprint)
        blueprint_payload: dict[str, Any] = {
            "task_id": TASK_ID,
            "schema_version": BLUEPRINT_SCHEMA,
            "formal_outcomes_accessed": False,
            "gate_count": len(blueprint),
            "stochastic_gate_count": len(blueprint) - deterministic,
            "deterministic_gate_count": deterministic,
            "family_counts": family_counts,
            "source_counts": source_counts,
            "gates": blueprint_rows,
        }
        blueprint_payload["analysis_sha256"] = _self_hash(blueprint_payload)
        _atomic_json(base / paths["blueprint"], blueprint_payload)

        density_rows = run_density(base, config, workers=workers)
        density_summary, density_decisions = summarize_density(config, density_rows)
        critical, maxt_rows, power_rows, maxt_decisions = build_joint_maxt(
            config, blueprint
        )
        resource, resource_decisions = resource_forecast(base, config)
        _atomic_json(base / paths["resource_preflight"], resource)
        all_source_rows = density_rows + maxt_rows + power_rows
        source_payload = _csv_bytes(all_source_rows)
        _atomic_bytes(base / paths["source_data"], source_payload)

        gates: dict[str, bool] = {
            "G01_formal_outcome_firewall_closed": (
                config["formal_outcome_access"]["t04_run_exists"] is False
                and config["formal_outcome_access"]["t04_formal_outcomes_accessed"]
                is False
            ),
            "G02_exact_full_matrix_accounting": (
                config["formal_matrix"]["cell_accounting"]["total_chunks"] == 518
                and config["formal_matrix"]["cell_accounting"]["primary_rows"]
                == 1042944
            ),
            "G03_exact_unique_blueprint_materialized": (
                len(blueprint) == len({gate.gate_id for gate in blueprint})
                and len(blueprint) == 3043
            ),
            "G04_old_seven_family_denominator_complete": (
                source_counts["old_seven_family"] == 1589
            ),
            "G05_t03_repair_denominator_complete": (
                source_counts["t03_bounded_repair"] == 1454
            ),
            "G06_stochastic_deterministic_partition": (
                deterministic == 6 and len(blueprint) - deterministic == 3037
            ),
            "G07_all_margins_positive": all(gate.margin > 0.0 for gate in blueprint),
            "G08_all_stochastic_cluster_units_registered": (
                {gate.cluster_count for gate in blueprint if not gate.deterministic}
                == {384, 768, 2304}
            ),
            "G09_no_aggregate_rescue_or_gate_deletion": (
                config["joint_maxt"]["aggregate_rescue"] is False
                and config["joint_maxt"]["gate_deletion"] is False
                and config["joint_maxt"]["cross_state_averaging"] is False
            ),
            "G10_density_raw_denominator_complete": (
                len(density_rows)
                == 24 * int(config["density_uq"]["trial_count_per_cell"])
            ),
            "G11_density_seed_address_injective": (
                len({row["trial_seed"] for row in density_rows}) == len(density_rows)
                and len({row["multiplier_seed"] for row in density_rows})
                == len(density_rows)
            ),
            "G12_density_all_coverage_and_power_gates": all(
                density_decisions.values()
            ),
            "G13_joint_maxt_uses_closed_stochastic_family": (
                len([gate for gate in blueprint if not gate.deterministic]) == 3037
            ),
            "G14_joint_maxt_higher_B199_factor1": (
                config["joint_maxt"]["quantile_method"] == "higher"
                and config["joint_maxt"]["multiplier_replicates"] == 199
                and config["joint_maxt"]["calibration_factor"] == 1.0
            ),
            "G15_joint_maxt_critical_and_all_power_gates": all(
                maxt_decisions.values()
            ),
            "G16_joint_maxt_not_pointwise_z": critical > 1.6448536269514722,
            "G17_resource_forecast_all_gates": all(resource_decisions.values()),
            "G18_t04_fresh_resource_benchmark_remains_mandatory": (
                resource["fresh_t04_resource_preflight_required"] is True
            ),
            "G19_all_claim_fields_null": all(
                config["claim_boundary"][key] is None
                for key in (
                    "twin_qualification", "ler", "lifetime",
                    "physical_break_even", "official_puviani_exact",
                    "puviani_nmf_surpass", "external_sota", "hardware_measured",
                )
            ),
        }
        report: dict[str, Any] = {
            "task_id": TASK_ID,
            "schema_version": REPORT_SCHEMA,
            "purpose": "statistical_preregistration_only",
            "formal_outcomes_accessed": False,
            "bindings": {
                "config": _binding(config_path, base),
                "old_design": _binding(old_path, base),
                "repair": _binding(repair_path, base),
                "blueprint": _binding(base / paths["blueprint"], base),
                "source_data": _binding(base / paths["source_data"], base),
                "resource_forecast": _binding(
                    base / paths["resource_preflight"], base
                ),
                "writer_source": _binding(Path(__file__), base),
            },
            "blueprint": {
                "gate_count": len(blueprint),
                "stochastic_gate_count": len(blueprint) - deterministic,
                "deterministic_gate_count": deterministic,
                "family_counts": family_counts,
                "source_counts": source_counts,
                "analysis_sha256": blueprint_payload["analysis_sha256"],
            },
            "density_uq": {
                "row_count": len(density_rows),
                "cell_count": len(density_summary),
                "factor": config["density_uq"]["calibration_factor"],
                "multiplier_replicates": config["density_uq"][
                    "multiplier_replicates"
                ],
                "dimensions": config["density_uq"]["dimensions"],
                "cluster_count": config["density_uq"]["clusters_per_state"],
                "cell_summaries": density_summary,
            },
            "joint_maxt": {
                "critical": critical,
                "multiplier_replicates": len([
                    row for row in maxt_rows
                    if row["row_type"] == "maxt_replicate"
                ]),
                "stochastic_gate_count": len(blueprint) - deterministic,
                "power_rows": power_rows,
                "contract": config["joint_maxt"],
            },
            "resource_forecast": resource,
            "gates": gates,
            "gate_summary": {
                "passed": sum(value is True for value in gates.values()),
                "total": len(gates),
            },
            "release": {
                "t04_preregistration_released": all(gates.values()),
                "t04_scientific_execution_released": False,
                "requires_independent_verifier_pass": True,
            },
            "claim_boundary": {
                **config["claim_boundary"],
                "t04_preregistration_released": all(gates.values()),
            },
            "verdict": PASS_VERDICT if all(gates.values()) else FAIL_VERDICT,
            "qualified_claim": None,
        }
        report["analysis_sha256"] = _self_hash(report)
        _atomic_json(base / paths["report"], report)
        live = _load_json(base / paths["report"])
        if live != report:
            raise RuntimeError("report atomic write drift")
        return report
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    report = write_artifacts(workers=args.workers)
    print(json.dumps({
        "verdict": report["verdict"],
        "gate_summary": report["gate_summary"],
        "blueprint": report["blueprint"],
        "density_rows": report["density_uq"]["row_count"],
        "joint_maxt_critical": report["joint_maxt"]["critical"],
        "report": CONFIG_PATH.replace(
            "configs/phase9/t_risk_20260728_05_highdim_joint_maxt_preflight.json",
            "docs/t_risk_20260728_05_highdim_joint_maxt_preflight.json",
        ),
    }, ensure_ascii=False, indent=2))
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())
