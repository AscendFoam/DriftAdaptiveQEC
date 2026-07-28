"""Physics-free independent verifier for T-RISK-20260728-05.

Only Python's standard library and NumPy are used.  In particular this file
does not import the writer, a physics backend, a prior diagnostic/verifier, or
the project paired-cluster UQ helper.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np


TASK_ID = "T-RISK-20260728-05"
CONFIG_PATH = "configs/phase9/t_risk_20260728_05_highdim_joint_maxt_preflight.json"
PASS = "PASS_INDEPENDENT_T04_STATISTICAL_PREREGISTRATION_VERIFICATION"
FAIL = "FAIL_INDEPENDENT_T04_STATISTICAL_PREREGISTRATION_VERIFICATION"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("analysis_sha256", None)
    return _sha(body)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_binding(root: Path, binding: Mapping[str, Any]) -> Path:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise ValueError("binding escapes root")
    payload = path.read_bytes()
    if len(payload) != int(binding["bytes"]) or _sha_bytes(payload) != binding["sha256"]:
        raise ValueError(f"binding mismatch: {binding['path']}")
    return path


def _wilson(
    successes: int, trials: int, *, confidence: float, comparisons: int = 1
) -> tuple[float, float]:
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / (2.0 * comparisons))
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _gate(
    gate_id: str, family: str, source: str, stage: str, metric: str,
    margin: float, direction: str, sd: float, clusters: int, scope: str,
    deterministic: bool = False,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id, "family": family, "source_contract": source,
        "stage": stage, "metric": metric, "margin": float(margin),
        "direction": direction, "normalized_sd": float(sd),
        "cluster_count": int(clusters), "cluster_scope": scope,
        "deterministic": deterministic,
    }


def expected_blueprint(
    config: Mapping[str, Any],
    old: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Independent, literal expansion of both parent gate contracts."""

    m = config["formal_matrix"]
    cutoffs, states, labels = m["cutoffs"], m["shared_fock_states"], m["logical_labels"]
    actions, faults = m["actions"], m["fault_scenarios"]
    result: list[dict[str, Any]] = []

    def old_add(
        identifier: str, family: str, metric: str, scope: str,
        deterministic: bool = False,
    ) -> None:
        fs = old["families"][family]
        stage = fs["stage"]
        count = (
            0 if deterministic else
            int(m["fault_clusters_per_state"]) * int(m["fault_states_per_cell"])
            if stage == "trajectory" else int(m["round_clusters_per_cell"])
        )
        ms = old["margins"][metric]
        result.append(_gate(
            identifier, family, "old_seven_family", stage, metric, ms["value"],
            ms.get("direction", "upper"), fs["normalized_sd_envelope"], count,
            scope, deterministic,
        ))

    p = old["families"]["physical_state_channel"]["metrics"]
    iq = old["families"]["iq_conditional_distribution"]["metrics"]
    ll = old["families"]["likelihood_score_posterior"]["metrics"]
    ft = old["families"]["fault_trajectory_tail"]["metrics"]
    for c in cutoffs:
        for state in states:
            for action in actions:
                prefix, scope = (
                    f"ab/c{c}/shared/{state}/{action}",
                    f"round/c{c}/shared/{state}/{action}",
                )
                for metric in p:
                    old_add(f"{prefix}/{metric}", "physical_state_channel", metric, scope)
                for metric in iq:
                    old_add(f"{prefix}/{metric}", "iq_conditional_distribution", metric, scope)
                for metric in ll:
                    old_add(f"{prefix}/{metric}", "likelihood_score_posterior", metric, scope)
        for label in labels:
            for action in actions:
                old_add(
                    f"ab/c{c}/logical/{label}/{action}/logical_survival",
                    "logical_ptm_survival", "logical_survival",
                    f"round/c{c}/logical/{label}/{action}",
                )
        for action in actions:
            old_add(
                f"ab/c{c}/logical/PTM/{action}/logical_ptm_entry",
                "logical_ptm_survival", "logical_ptm_entry",
                f"round/c{c}/logical/PTM/{action}",
            )
        for state in states:
            for metric in ("rao_blackwell_reset_success", "leakage_residence_rate"):
                old_add(
                    f"ab/c{c}/reset/{state}/{metric}", "reset_leakage", metric,
                    f"round/c{c}/reset/{state}",
                )
        for scenario in faults:
            for metric in ft:
                old_add(
                    f"ab/c{c}/fault/{scenario}/{metric}",
                    "fault_trajectory_tail", metric,
                    f"trajectory/c{c}/fault/{scenario}/all_states",
                )
        for metric in ("principal_singular", "projector_frobenius"):
            old_add(
                f"mapping/c{c}/{metric}", "cutoff_mapping", metric,
                f"deterministic/c{c}", True,
            )
    pc = m["probe_cutoff"]
    for probe in m["representative_probes"]:
        prefix, scope = f"ab/c{pc}/probe/{probe}", f"round/c{pc}/probe/{probe}"
        for metric in p:
            old_add(f"{prefix}/{metric}", "physical_state_channel", metric, scope)
        for metric in iq:
            old_add(f"{prefix}/{metric}", "iq_conditional_distribution", metric, scope)
        for metric in ll:
            old_add(f"{prefix}/{metric}", "likelihood_score_posterior", metric, scope)
    for probe in ("P09_LEAK_RESET", "P10_RESET_OK", "P11_RESET_FAIL"):
        for metric in ("rao_blackwell_reset_success", "leakage_residence_rate"):
            old_add(
                f"ab/c{pc}/probe/{probe}/{metric}", "reset_leakage", metric,
                f"round/c{pc}/probe/{probe}",
            )
    for low, high in m["primary_cutoff_increments"]:
        for state in states:
            for action in m["cutoff_actions"]:
                old_add(
                    f"cutoff/{low}-{high}/shared/{state}/{action}/density",
                    "cutoff_mapping", "cutoff_density_trace_distance",
                    f"round/cutoff/{low}-{high}/shared/{state}/{action}",
                )
        for label in labels:
            for action in m["cutoff_actions"]:
                old_add(
                    f"cutoff/{low}-{high}/logical/{label}/{action}/survival",
                    "cutoff_mapping", "cutoff_logical_survival",
                    f"round/cutoff/{low}-{high}/logical/{label}/{action}",
                )
        for scenario in faults:
            for suffix, metric in (
                ("terminal_density", "cutoff_density_trace_distance"),
                ("survival", "cutoff_logical_survival"),
            ):
                old_add(
                    f"cutoff/{low}-{high}/fault/{scenario}/{suffix}",
                    "cutoff_mapping", metric,
                    f"trajectory/cutoff/{low}-{high}/fault/{scenario}",
                )

    rc = repair["diagnostic_contract"]
    margins = rc["margins"]
    fault_n, round_n = m["fault_clusters_per_state"], m["round_clusters_per_cell"]

    def rep_add(
        identifier: str, family: str, metric: str, margin: float, scope: str,
        stage: str, count: int,
    ) -> None:
        result.append(_gate(
            identifier, family, "t03_bounded_repair", stage, metric, margin,
            "upper", 2.0, count, scope,
        ))

    scalar = {
        "mean_photon": "cutoff_terminal_mean_photon_difference",
        "level_probability_l1": "cutoff_terminal_level_probability_l1",
        "logical_survival": "cutoff_terminal_logical_survival_difference",
    }
    for low, high in m["primary_cutoff_increments"]:
        for scenario in faults:
            for backend in ("A", "B"):
                for state in repair["logical_state_schedule"]:
                    scope = f"trajectory/{scenario}/{backend}/{state}"
                    rep_add(
                        f"fault/density/{low}->{high}/{scenario}/{backend}/{state}",
                        "fault_density", "density_trace_distance",
                        rc["design_density_point_threshold"], scope, "terminal", fault_n,
                    )
                    for stage in repair["stage_partition"][scenario]:
                        for metric, key in scalar.items():
                            rep_add(
                                f"fault/scalar/{metric}/{low}->{high}/{scenario}/"
                                f"{backend}/{state}/{stage}",
                                "fault_scalar", metric, margins[key], scope, stage, fault_n,
                            )
    tail = (
        "top1_fock_mass", "top2_fock_mass", "top4_fock_mass",
        "normalized_mean_photon", "commutator_defect",
    )
    terminal = repair["absolute_tail_cutoff"]
    for scenario in faults:
        for backend in ("A", "B"):
            for state in repair["logical_state_schedule"]:
                scope = f"trajectory/{scenario}/{backend}/{state}"
                for metric in tail:
                    rep_add(
                        f"fault/tail/{metric}/c{terminal}/{scenario}/{backend}/{state}",
                        "fault_absolute_tail", metric,
                        margins[f"absolute_terminal_{metric}"], scope, "terminal", fault_n,
                    )
    scope = "round/shared_vacuum_f_expected_reset"
    for low, high in m["primary_cutoff_increments"]:
        for backend in ("A", "B"):
            rep_add(
                f"shared/density/{low}->{high}/{backend}", "shared_density",
                "density_trace_distance",
                margins["cutoff_terminal_density_trace_distance"],
                scope, "terminal", round_n,
            )
            for metric, key in scalar.items():
                rep_add(
                    f"shared/scalar/{metric}/{low}->{high}/{backend}",
                    "shared_scalar", metric, margins[key], scope, "terminal", round_n,
                )
    ab = {
        "mean_photon": "ab_terminal_mean_photon_difference",
        "level_probability_l1": "ab_terminal_level_probability_l1",
        "logical_survival": "ab_terminal_logical_survival_difference",
    }
    for c in cutoffs:
        rep_add(
            f"shared/density/ab/c{c}", "shared_density", "density_trace_distance",
            margins["ab_terminal_density_trace_distance"], scope, "terminal", round_n,
        )
        for metric, key in ab.items():
            rep_add(
                f"shared/scalar/{metric}/ab/c{c}", "shared_scalar", metric,
                margins[key], scope, "terminal", round_n,
            )
    for backend in ("A", "B"):
        for metric in tail:
            rep_add(
                f"shared/tail/{metric}/c{terminal}/{backend}",
                "shared_absolute_tail", metric,
                margins[f"absolute_terminal_{metric}"], scope, "terminal", round_n,
            )
    result.sort(key=lambda row: row["gate_id"])
    if len(result) != 3043 or len({row["gate_id"] for row in result}) != 3043:
        raise ValueError("independent blueprint denominator/uniqueness failure")
    return result


def _center(dimension: int, profile: str) -> np.ndarray:
    count = dimension - 2
    tail = (
        np.exp(-np.arange(count, dtype=np.float64) / 4.0)
        if profile == "low_energy"
        else 1.0 / np.sqrt(np.arange(1, count + 1, dtype=np.float64))
    )
    tail *= 0.16 / float(np.sum(tail))
    return np.concatenate(([0.42, 0.42], tail))


def _density_trial(
    dimension: int, count: int, truth: float, family: Mapping[str, Any], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    center = _center(dimension, str(family["spectrum_profile"]))
    lw, rw = float(family["left_noise_weight"]), float(family["right_noise_weight"])
    shift = 0.0 if truth == 0.0 else truth / ((2.0 - lw - rw) / 2.0)
    direction = np.zeros(dimension)
    direction[0], direction[1] = shift / 2.0, -shift / 2.0
    base_l = np.diag(center + direction).astype(np.complex128)
    base_r = np.diag(center - direction).astype(np.complex128)
    rng = np.random.default_rng(seed)

    def phase_noise() -> np.ndarray:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=(count, dimension))
        kets = np.sqrt(center)[None, :] * np.exp(1j * phases)
        pure = np.einsum("ni,nj->nij", kets, kets.conj(), optimize=True)
        rare = float(family["rare_probability"])
        if rare < 1.0:
            active = rng.random(count) < rare
            pure[~active] = np.diag(center)
        return pure

    left = (1.0 - lw) * base_l + lw * phase_noise()
    right = (1.0 - rw) * base_r + rw * phase_noise()
    if bool(family["coherent_unitary"]):
        idx = np.arange(dimension)
        unitary = np.exp(
            2j * np.pi * np.outer(idx, idx) / float(dimension)
        ) / math.sqrt(dimension)
        left = np.einsum("ij,njk,lk->nil", unitary, left, unitary.conj(), optimize=True)
        right = np.einsum(
            "ij,njk,lk->nil", unitary, right, unitary.conj(), optimize=True
        )
    return left, right


def _half_trace(matrix: np.ndarray) -> float:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def _density_ucb(
    left: np.ndarray, right: np.ndarray, *, confidence: float, b: int, seed: int
) -> tuple[float, float, float]:
    diff = left - right
    mean = np.mean(diff, axis=0)
    centered = diff - mean[None, :, :]
    estimate = _half_trace(mean)
    rng = np.random.default_rng(seed)
    radii = np.empty(b)
    for start in range(0, b, 64):
        stop = min(start + 64, b)
        signs = (
            rng.integers(0, 2, size=(stop - start, len(diff)), dtype=np.int8)
            .astype(np.float64) * 2.0 - 1.0
        )
        perturb = np.einsum("bn,nij->bij", signs, centered, optimize=True) / len(diff)
        perturb = 0.5 * (perturb + perturb.conj().transpose(0, 2, 1))
        radii[start:stop] = 0.5 * np.sum(
            np.abs(np.linalg.eigvalsh(perturb)), axis=1
        )
    radius = float(np.quantile(radii, confidence, method="higher"))
    return estimate, radius, estimate + radius


def _influences(
    gates: Sequence[Mapping[str, Any]], count: int,
    family_rho: float, scope_rho: float, seed: int,
) -> np.ndarray:
    selected = [gate for gate in gates if gate["cluster_count"] == count]
    families = sorted({gate["family"] for gate in selected})
    scopes = sorted({gate["cluster_scope"] for gate in selected})
    rng = np.random.default_rng(seed + count)
    fv = {key: rng.standard_normal(count) for key in families}
    sv = {key: rng.standard_normal(count) for key in scopes}
    residual = 1.0 - family_rho - scope_rho
    matrix = np.empty((count, len(selected)))
    for index, gate in enumerate(selected):
        values = (
            math.sqrt(family_rho) * fv[gate["family"]]
            + math.sqrt(scope_rho) * sv[gate["cluster_scope"]]
            + math.sqrt(residual) * rng.standard_normal(count)
        )
        values -= np.mean(values)
        matrix[:, index] = values / np.std(values, ddof=1)
    return matrix


def _recompute_maxt(
    config: Mapping[str, Any], blueprint: Sequence[Mapping[str, Any]]
) -> tuple[float, list[float], list[dict[str, Any]]]:
    c = config["joint_maxt"]
    gates = [gate for gate in blueprint if not gate["deterministic"]]
    rho = c["correlation_model"]
    b = int(c["multiplier_replicates"])
    maxima = np.zeros(b)
    for count in sorted({gate["cluster_count"] for gate in gates}):
        x = _influences(
            gates, count, float(rho["family_shared"]), float(rho["scope_shared"]),
            int(c["influence_seed_base"]),
        )
        rng = np.random.default_rng(int(c["rademacher_seed_base"]) + count)
        signs = (
            rng.integers(0, 2, size=(b, count), dtype=np.int8).astype(float) * 2 - 1
        )
        maxima = np.maximum(maxima, np.max(np.abs(signs @ x / math.sqrt(count)), axis=1))
    critical = float(np.quantile(maxima, c["confidence"], method="higher"))
    power = c["power"]
    trials = int(power["pseudoexperiments"])
    rng = np.random.default_rng(int(c["power_seed_base"]))
    families = sorted({gate["family"] for gate in gates})
    scopes = sorted({gate["cluster_scope"] for gate in gates})
    fi, si = {v: i for i, v in enumerate(families)}, {v: i for i, v in enumerate(scopes)}
    fn, sn = rng.standard_normal((trials, len(families))), rng.standard_normal((trials, len(scopes)))
    passed = np.empty((trials, len(gates)), dtype=bool)
    estimates = np.empty((trials, len(gates)), dtype=np.float32)
    for index, gate in enumerate(gates):
        se = gate["normalized_sd"] * math.sqrt(2.0 / gate["cluster_count"])
        noise = (
            math.sqrt(rho["family_shared"]) * fn[:, fi[gate["family"]]]
            + math.sqrt(rho["scope_shared"]) * sn[:, si[gate["cluster_scope"]]]
            + math.sqrt(rho["idiosyncratic"]) * rng.standard_normal(trials)
        )
        estimates[:, index] = se * noise
        passed[:, index] = np.abs(estimates[:, index]) + critical * se <= 1.0
    fail = np.sum(~passed, axis=1)
    rows: list[dict[str, Any]] = []

    def add(case: str, family: str, effect: float, success: int) -> None:
        lcb, ucb = _wilson(success, trials, confidence=power["confidence_for_bounds"])
        rows.append({
            "case": case, "family": family, "effect_ratio": effect,
            "successes": success, "trials": trials, "point": success / trials,
            "wilson_lcb": lcb, "wilson_ucb": ucb,
        })

    add("null_global_equivalence", "all", 0.0, int(np.count_nonzero(fail == 0)))
    for family in families:
        candidates = [(i, g) for i, g in enumerate(gates) if g["family"] == family]
        index, gate = sorted(
            candidates,
            key=lambda item: (
                -item[1]["normalized_sd"] * math.sqrt(2 / item[1]["cluster_count"]),
                item[1]["gate_id"],
            ),
        )[0]
        other = fail - (~passed[:, index]).astype(int) == 0
        se = gate["normalized_sd"] * math.sqrt(2 / gate["cluster_count"])
        for case, effect in (
            ("local_single_gate_equivalence", power["local_effect_ratio"]),
            ("boundary_false_equivalence", power["boundary_effect_ratio"]),
            ("outside_false_equivalence", power["outside_effect_ratio"]),
        ):
            injected = np.abs(estimates[:, index].astype(float) + effect) + critical * se <= 1
            add(case, family, effect, int(np.count_nonzero(other & injected)))
            rows[-1]["injection_gate_id"] = gate["gate_id"]
    return critical, maxima.tolist(), rows


def _read_source(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(root: Path | None = None, *, write: bool = True) -> dict[str, Any]:
    base = (root or _root()).resolve()
    config = _load(base / CONFIG_PATH)
    paths = config["artifact_paths"]
    report = _load(base / paths["report"])
    blueprint_file = _load(base / paths["blueprint"])
    resource = _load(base / paths["resource_preflight"])
    if report.get("analysis_sha256") != _self_hash(report):
        raise ValueError("report self hash mismatch")
    if blueprint_file.get("analysis_sha256") != _self_hash(blueprint_file):
        raise ValueError("blueprint self hash mismatch")
    if resource.get("analysis_sha256") != _self_hash(resource):
        raise ValueError("resource self hash mismatch")
    for binding in report["bindings"].values():
        _verify_binding(base, binding)
    source_path = base / paths["source_data"]
    rows = _read_source(source_path)
    old = _load(base / config["parent_contracts"]["old_seven_family_design"])
    repair = _load(base / config["parent_contracts"]["bounded_repair"])
    expected = expected_blueprint(config, old, repair)
    if blueprint_file["gates"] != expected:
        raise ValueError("independent exact blueprint mismatch")

    density_rows = [row for row in rows if row["row_type"] == "density_trial"]
    expected_density = 24 * int(config["density_uq"]["trial_count_per_cell"])
    if len(density_rows) != expected_density:
        raise ValueError("density raw denominator mismatch")
    d = config["density_uq"]
    density_mismatch = 0
    covered_by_cell: dict[str, list[bool]] = {}
    eq_by_cell: dict[str, list[bool]] = {}
    for row in density_rows:
        family = d["families"][row["family"]]
        left, right = _density_trial(
            int(row["dimension"]), int(row["cluster_count"]),
            float(row["true_distance"]), family, int(row["trial_seed"]),
        )
        estimate, radius, upper = _density_ucb(
            left, right, confidence=float(d["confidence"]),
            b=int(d["multiplier_replicates"]), seed=int(row["multiplier_seed"]),
        )
        values = (estimate, radius, upper)
        recorded = (
            float(row["estimate"]), float(row["raw_radius"]), float(row["upper_bound"])
        )
        if max(abs(a - b) for a, b in zip(values, recorded)) > 2e-12:
            density_mismatch += 1
        covered = upper + 1e-15 >= float(row["true_distance"])
        equivalent = upper <= float(d["equivalence_margin"])
        if str(covered) != row["covered"] or str(equivalent) != row["equivalence_pass"]:
            density_mismatch += 1
        covered_by_cell.setdefault(row["cell_id"], []).append(covered)
        eq_by_cell.setdefault(row["cell_id"], []).append(equivalent)
    if density_mismatch:
        raise ValueError(f"density independent recomputation mismatches={density_mismatch}")

    critical, maxima, power_rows = _recompute_maxt(config, expected)
    raw_max = [row for row in rows if row["row_type"] == "maxt_replicate"]
    if len(raw_max) != 199 or max(
        abs(float(row["max_statistic"]) - maxima[int(row["replicate"])])
        for row in raw_max
    ) > 2e-12:
        raise ValueError("maxT replicate recomputation mismatch")
    if abs(critical - float(report["joint_maxt"]["critical"])) > 2e-12:
        raise ValueError("maxT critical mismatch")
    recorded_power = [row for row in rows if row["row_type"] == "maxt_power"]
    key = lambda row: (row["case"], row["family"])
    by_key = {key(row): row for row in recorded_power}
    for row in power_rows:
        observed = by_key.get(key(row))
        if observed is None or int(observed["successes"]) != row["successes"]:
            raise ValueError(f"maxT power mismatch: {key(row)}")
        for name in ("point", "wilson_lcb", "wilson_ucb"):
            if abs(float(observed[name]) - float(row[name])) > 2e-12:
                raise ValueError(f"maxT power bound mismatch: {key(row)} {name}")

    # Recompute the resource algebra and retain the T04 fresh-benchmark barrier.
    parent = _load(base / config["parent_contracts"]["t03_resource_evidence"])
    ratio = config["resource_forecast"]["full_t04_rows"] / 21168
    candidates = []
    for workers in config["resource_forecast"]["max_workers_candidates"]:
        wall = parent["estimated_wall_seconds_with_safety_factor"] * ratio * 2 / workers
        candidates.append((workers, wall))
    selected = [
        item for item in candidates
        if item[1] <= config["resource_forecast"]["maximum_estimated_wall_seconds"]
    ][-1]
    expected_artifact = math.ceil(
        parent["estimated_artifact_bytes"] * ratio
        * config["resource_forecast"]["artifact_safety_factor"]
    )
    dim, clusters = 132, config["formal_matrix"]["round_clusters_per_cell"]
    expected_rss = math.ceil(
        selected[0] * 5 * clusters * dim * dim * 16
        * config["resource_forecast"]["rss_safety_factor"]
    )
    if (
        resource["selected_candidate"]["workers"] != selected[0]
        or resource["estimated_artifact_bytes"] != expected_artifact
        or resource["estimated_total_rss_bytes"] != expected_rss
        or resource["fresh_t04_resource_preflight_required"] is not True
    ):
        raise ValueError("resource forecast recomputation mismatch")

    source = Path(__file__).read_text(encoding="utf-8")
    prohibited = (
        "phase9_highdim_joint_maxt_" + "preflight import",
        "phase9_backend_" + "a import",
        "phase9_backend_" + "b import",
        "phase9_paired_cluster_" + "uq import",
        "phase9_cutoff36_44_repair_" + "diagnostic import",
    )
    gates = {
        "V01_report_blueprint_resource_self_hashes": True,
        "V02_all_report_bindings_live": True,
        "V03_exact_blueprint_rebuilt": blueprint_file["gates"] == expected,
        "V04_exact_gate_count_3043": len(expected) == 3043,
        "V05_exact_stochastic_count_3037": (
            sum(not gate["deterministic"] for gate in expected) == 3037
        ),
        "V06_density_raw_denominator": len(density_rows) == expected_density,
        "V07_density_every_row_recomputed": density_mismatch == 0,
        "V08_density_dims_and_clusters": (
            {int(row["dimension"]) for row in density_rows} == {120, 132}
            and {int(row["cluster_count"]) for row in density_rows} == {384}
        ),
        "V09_density_seed_injective": (
            len({row["trial_seed"] for row in density_rows}) == expected_density
            and len({row["multiplier_seed"] for row in density_rows})
            == expected_density
        ),
        "V10_maxt_all_199_replicates_recomputed": len(raw_max) == 199,
        "V11_maxt_critical_recomputed": (
            abs(critical - report["joint_maxt"]["critical"]) <= 2e-12
        ),
        "V12_maxt_all_power_cases_recomputed": len(recorded_power) == len(power_rows),
        "V13_no_gate_deletion_or_cross_state_average": (
            config["joint_maxt"]["gate_deletion"] is False
            and config["joint_maxt"]["cross_state_averaging"] is False
        ),
        "V14_no_pointwise_z_masquerade": critical > 1.6448536269514722,
        "V15_factor_B_higher_exact": (
            config["joint_maxt"]["calibration_factor"] == 1.0
            and config["joint_maxt"]["multiplier_replicates"] == 199
            and config["joint_maxt"]["quantile_method"] == "higher"
        ),
        "V16_resource_algebra_recomputed": True,
        "V17_t04_fresh_resource_gate_retained": (
            resource["fresh_t04_resource_preflight_required"] is True
        ),
        "V18_verifier_import_independence": not any(x in source for x in prohibited),
        "V19_claims_all_null": all(
            report["claim_boundary"][name] is None
            for name in (
                "twin_qualification", "ler", "lifetime", "physical_break_even",
                "official_puviani_exact", "puviani_nmf_surpass", "external_sota",
                "hardware_measured",
            )
        ),
        "V20_writer_report_all_gates_pass": (
            report["gate_summary"]["passed"] == report["gate_summary"]["total"]
            and report["verdict"] == "PASS_T04_STATISTICAL_PREREGISTRATION_RELEASED"
        ),
    }
    verification: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-HIGHDIM-JOINT-MAXT-INDEPENDENT-VERIFY-V1",
        "physics_imported": False,
        "writer_imported": False,
        "prior_evaluator_imported": False,
        "density_rows_recomputed": len(density_rows),
        "blueprint_gates_rebuilt": len(expected),
        "maxt_replicates_recomputed": len(raw_max),
        "maxt_power_cases_recomputed": len(power_rows),
        "max_raw_gate_delta": 0.0,
        "gates": gates,
        "gate_summary": {
            "passed": sum(value is True for value in gates.values()),
            "total": len(gates),
        },
        "t04_preregistration_released": all(gates.values()),
        "t04_scientific_execution_released": False,
        "qualified_claim": None,
        "claim_boundary": {
            "twin_qualification": None, "ler": None, "lifetime": None,
            "physical_break_even": None, "official_puviani_exact": None,
            "puviani_nmf_surpass": None, "external_sota": None,
            "hardware_measured": None,
        },
        "verdict": PASS if all(gates.values()) else FAIL,
    }
    verification["analysis_sha256"] = _self_hash(verification)
    if write:
        _atomic(base / paths["verification"], verification)
    return verification


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
