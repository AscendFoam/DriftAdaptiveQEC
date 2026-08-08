"""Physics-free independent verifier for T-RISK-20260728-06.

This verifier intentionally uses only the Python standard library and NumPy.
It does not import the T05/T06 writers, a physics backend, an existing
verifier, or the project paired-cluster UQ helper.  Every density trial and
every correlated maxT/power replicate is reconstructed from frozen seeds.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TASK_ID = "T-RISK-20260728-06"
CONFIG_PATH = "configs/phase9/t_risk_20260728_06_count_selection_confirmation.json"
PASS = "PASS_INDEPENDENT_COUNT_SELECTION_AND_CONFIRMATION"
VERIFIED_NO_GO = "PASS_INDEPENDENT_TERMINAL_COUNT_NO_GO"
FAIL = "FAIL_INDEPENDENT_COUNT_SELECTION_VERIFICATION"
WRITER_PASS = "PASS_COUNT_SELECTED_AND_UNTOUCHED_CONFIRMED"
WRITER_NO_GO = "TERMINAL_NO_GO_COUNT_SELECTION_OR_CONFIRMATION"
SCHEMA = "PHASE9-COUNT-SELECTION-CONFIRMATION-INDEPENDENT-VERIFY-V1"

NULL_CLAIMS = (
    "twin_qualification",
    "ler",
    "lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "hardware_measured",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "cnn_fpga.benchmark.phase9_count_selection_confirmation",
    "cnn_fpga.benchmark.phase9_highdim_joint_maxt_preflight",
    "cnn_fpga.benchmark.phase9_highdim_joint_maxt_preflight_verify",
    "cnn_fpga.benchmark.phase9_paired_cluster_uq",
    "physics",
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


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("analysis_sha256", None)
    return _sha(body)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _binding(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _safe_bound_path(root: Path, binding: Mapping[str, Any]) -> Path:
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("binding path missing")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("binding escapes repository root")
    payload = path.read_bytes()
    if (
        isinstance(binding.get("bytes"), bool)
        or int(binding.get("bytes", -1)) != len(payload)
        or binding.get("sha256") != _sha_bytes(payload)
    ):
        raise ValueError(f"binding mismatch: {relative}")
    return path


def _bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{name} is not a strict boolean")


def _float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} is boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} is boolean")
    result = int(value)
    if str(result) != str(value) and not isinstance(value, int):
        try:
            if float(value) != result:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} is not integral") from exc
    return result


def _assert_close(actual: object, expected: object, name: str, tol: float = 2e-12) -> float:
    delta = abs(_float(actual, name) - _float(expected, name))
    if delta > tol:
        raise ValueError(f"{name} mismatch: {delta}")
    return delta


def _intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version")
        != "PHASE9-COUNT-SELECTION-CONFIRMATION-CONFIG-V1"
    ):
        raise ValueError("T06 config identity drift")
    firewall = config.get("outcome_firewall")
    if not isinstance(firewall, dict) or (
        firewall.get("t04_run_exists") is not False
        or firewall.get("t04_formal_outcomes_accessed") is not False
        or firewall.get("t05_no_go_rewritten") is not False
        or firewall.get("selection_may_choose") != ["linked_cluster_count"]
        or firewall.get("confirmation_may_choose_nothing") is not True
        or firewall.get("candidate_exhaustion")
        != "TERMINAL_NO_GO_NO_FURTHER_COUNT_LADDER"
    ):
        raise ValueError("outcome firewall drift")
    forbidden = set(firewall.get("selection_may_not_choose", ()))
    if forbidden != {
        "factor", "B", "quantile", "margin", "family", "gate", "correlation",
        "effect", "confidence", "confirmation_seed", "cutoff",
    }:
        raise ValueError("selection forbidden-field firewall drift")
    grid = config.get("linked_count_grid")
    expected_grid = [
        {
            "scale": 1.5, "state_clusters": 576, "round_clusters": 1152,
            "aggregate_fault_clusters": 3456,
        },
        {
            "scale": 2.0, "state_clusters": 768, "round_clusters": 1536,
            "aggregate_fault_clusters": 4608,
        },
        {
            "scale": 2.5, "state_clusters": 960, "round_clusters": 1920,
            "aggregate_fault_clusters": 5760,
        },
        {
            "scale": 3.0, "state_clusters": 1152, "round_clusters": 2304,
            "aggregate_fault_clusters": 6912,
        },
    ]
    if grid != expected_grid:
        raise ValueError("finite linked count grid drift")
    if config.get("selection_rule") != (
        "lexicographically smallest linked_count_grid row passing every "
        "density-selection and full-blueprint joint-maxT selection gate; "
        "no skipped candidate"
    ):
        raise ValueError("smallest-candidate selection rule drift")
    density = config["density"]
    if (
        density["dimensions"] != [120, 132]
        or density["selection_family"] != "heteroskedastic_coherent"
        or density["selection_effect"] != 0.05
        or sorted(density["confirmation_families"]) != [
            "heavy_tail_rare_coherent",
            "heteroskedastic_coherent",
            "low_energy_balanced",
        ]
        or density["confirmation_effects"] != [0.0, 0.05, 0.1, 0.12]
        or density["margin"] != 0.1
        or density["confidence"] != 0.95
        or density["factor"] != 1.0
        or density["multiplier_replicates"] != 199
        or density["quantile"] != "higher"
        or density["selection_trials_per_cell"] != 128
        or density["confirmation_trials_per_cell"] != 256
        or "injective" not in density["seed_address"]
        or "no modulo" not in density["seed_address"]
    ):
        raise ValueError("density selection/confirmation contract drift")
    if {
        name: density[name] for name in (
            "selection_trial_seed_base",
            "selection_multiplier_seed_base",
            "confirmation_trial_seed_base",
            "confirmation_multiplier_seed_base",
        )
    } != {
        "selection_trial_seed_base": 201000000,
        "selection_multiplier_seed_base": 202000000,
        "confirmation_trial_seed_base": 203000000,
        "confirmation_multiplier_seed_base": 204000000,
    }:
        raise ValueError("density seed namespace freeze drift")
    if density["selection_wilson"] != {
        "confidence": 0.95,
        "comparisons": 8,
        "coverage_lcb_minimum": 0.9,
        "local_equivalence_lcb_minimum": 0.65,
    } or density["confirmation_wilson"] != {
        "confidence": 0.95,
        "comparisons": 24,
        "coverage_rate_minimum": 0.94,
        "coverage_lcb_minimum": 0.9,
        "null_equivalence_lcb_minimum": 0.8,
        "local_equivalence_lcb_minimum": 0.65,
        "boundary_equivalence_ucb_maximum": 0.1,
        "outside_equivalence_ucb_maximum": 0.05,
    }:
        raise ValueError("density Wilson gate contract drift")
    maxt = config["joint_maxt"]
    if (
        maxt["stochastic_gate_count"] != 3037
        or maxt["deterministic_gate_count"] != 6
        or maxt["closed_gate_count"] != 3043
        or maxt["confidence"] != 0.95
        or maxt["factor"] != 1.0
        or maxt["multiplier_replicates"] != 199
        or maxt["quantile"] != "higher"
        or maxt["global_test"] != "intersection_union_equivalence"
        or maxt["aggregate_rescue"] is not False
        or maxt["gate_deletion"] is not False
        or maxt["cross_state_averaging"] is not False
        or maxt["pointwise_z_substitution"] is not False
        or maxt["correlation_model"]
        != {"family_shared": 0.55, "scope_shared": 0.25, "idiosyncratic": 0.2}
    ):
        raise ValueError("joint maxT contract drift")
    if abs(sum(maxt["correlation_model"].values()) - 1.0) > 1e-15:
        raise ValueError("maxT correlation components do not sum to one")
    if {
        name: maxt[name] for name in (
            "selection_influence_seed_base",
            "selection_rademacher_seed_base",
            "selection_power_seed_base",
            "confirmation_influence_seed_base",
            "confirmation_rademacher_seed_base",
            "confirmation_power_seed_base",
        )
    } != {
        "selection_influence_seed_base": 205000000,
        "selection_rademacher_seed_base": 206000000,
        "selection_power_seed_base": 207000000,
        "confirmation_influence_seed_base": 208000000,
        "confirmation_rademacher_seed_base": 209000000,
        "confirmation_power_seed_base": 210000000,
    }:
        raise ValueError("maxT seed namespace freeze drift")
    if maxt["power"] != {
        "pseudoexperiments": 4000,
        "null_global_equivalence_lcb_minimum": 0.9,
        "local_effect_ratio": 0.5,
        "local_single_gate_equivalence_lcb_minimum": 0.6,
        "boundary_effect_ratio": 1.0,
        "boundary_false_equivalence_ucb_maximum": 0.05,
        "outside_effect_ratio": 1.25,
        "outside_false_equivalence_ucb_maximum": 0.01,
        "confidence_for_bounds": 0.95,
    }:
        raise ValueError("maxT power contract drift")
    if config["parent_artifacts"] != {
        "t05_report": "docs/t_risk_20260728_05_highdim_joint_maxt_preflight.json",
        "t05_verification": (
            "docs/t_risk_20260728_05_highdim_joint_maxt_verification.json"
        ),
        "t05_blueprint": "docs/t_risk_20260728_05_full_gate_blueprint.json",
        "t05_config": (
            "configs/phase9/t_risk_20260728_05_highdim_joint_maxt_preflight.json"
        ),
    }:
        raise ValueError("T05 parent path contract drift")
    resource = config["resource"]
    if resource != {
        "base_t04_rows": 1042944,
        "base_wall_seconds_4_workers": 230440.09880259438,
        "base_artifact_bytes": 58513386935,
        "base_rss_bytes": 8564244480,
        "workers": 4,
        "blas_threads_per_worker": 1,
        "maximum_wall_seconds": 1209600,
        "maximum_artifact_bytes": 171798691840,
        "maximum_rss_bytes": 34359738368,
        "minimum_free_disk_after_estimate_bytes": 34359738368,
        "t04_fresh_layer_benchmark_required": True,
        "forecast_only": True,
    }:
        raise ValueError("resource forecast contract drift")
    if config["artifact_paths"] != {
        "run_directory": "runs/t_risk_20260728_06_count_selection_confirmation",
        "selection_chunks": (
            "runs/t_risk_20260728_06_count_selection_confirmation/selection_chunks"
        ),
        "confirmation_chunks": (
            "runs/t_risk_20260728_06_count_selection_confirmation/confirmation_chunks"
        ),
        "heartbeat": (
            "runs/t_risk_20260728_06_count_selection_confirmation/heartbeat.json"
        ),
        "report": "docs/t_risk_20260728_06_count_selection_confirmation.json",
        "source_data": (
            "docs/t_risk_20260728_06_count_selection_confirmation_source_data.csv"
        ),
        "selected_blueprint": "docs/t_risk_20260728_06_selected_gate_blueprint.json",
        "verification": (
            "docs/t_risk_20260728_06_count_selection_confirmation_verification.json"
        ),
    }:
        raise ValueError("artifact path contract drift")

    intervals = [
        (
            int(density["selection_trial_seed_base"]),
            int(density["selection_trial_seed_base"]) + 8 * 128,
        ),
        (
            int(density["selection_multiplier_seed_base"]),
            int(density["selection_multiplier_seed_base"]) + 8 * 128,
        ),
        (
            int(density["confirmation_trial_seed_base"]),
            int(density["confirmation_trial_seed_base"]) + 24 * 256,
        ),
        (
            int(density["confirmation_multiplier_seed_base"]),
            int(density["confirmation_multiplier_seed_base"]) + 24 * 256,
        ),
    ]
    point_names = (
        "selection_influence_seed_base", "selection_rademacher_seed_base",
        "selection_power_seed_base", "confirmation_influence_seed_base",
        "confirmation_rademacher_seed_base", "confirmation_power_seed_base",
    )
    points = [int(maxt[name]) for name in point_names]
    if len(points) != len(set(points)):
        raise ValueError("maxT seed namespace collision")
    max_counts = {
        int(row["state_clusters"]) for row in grid
    } | {
        int(row["round_clusters"]) for row in grid
    } | {
        int(row["aggregate_fault_clusters"]) for row in grid
    }
    maxt_used: set[int] = set()
    for split in ("selection", "confirmation"):
        influence_base = int(maxt[f"{split}_influence_seed_base"])
        rademacher_base = int(maxt[f"{split}_rademacher_seed_base"])
        power_seed = int(maxt[f"{split}_power_seed_base"])
        used = (
            {influence_base + count for count in max_counts}
            | {rademacher_base + count for count in max_counts}
            | {power_seed}
        )
        if len(used) != 2 * len(max_counts) + 1 or maxt_used & used:
            raise ValueError("maxT used-seed namespace collision")
        maxt_used |= used
    for index, left in enumerate(intervals):
        if any(_intervals_overlap(left, right) for right in intervals[index + 1:]):
            raise ValueError("density seed range collision")
        if any(left[0] <= point < left[1] for point in maxt_used):
            raise ValueError("density/maxT seed namespace collision")
    if any(config["claim_boundary"].get(name) is not None for name in NULL_CLAIMS):
        raise ValueError("prohibited config claim populated")
    if (
        config["claim_boundary"].get("count_design_only") is not True
        or config["claim_boundary"].get("t04_preregistration_released") is not False
        or config["claim_boundary"].get("t04_scientific_execution_released") is not False
    ):
        raise ValueError("config release boundary drift")


def _validate_import_independence() -> bool:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    return not any(
        name == prefix or name.startswith(prefix + ".")
        for name in imported for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def _wilson(
    successes: int,
    trials: int,
    *,
    confidence: float,
    comparisons: int,
) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials or comparisons < 1:
        raise ValueError("invalid Wilson inputs")
    z = NormalDist().inv_cdf(
        1.0 - (1.0 - confidence) / (2.0 * comparisons)
    )
    point = successes / trials
    denominator = 1.0 + z * z / trials
    center = (point + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        point * (1.0 - point) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _center_probabilities(dimension: int, profile: str) -> np.ndarray:
    if dimension < 4:
        raise ValueError("density dimension too small")
    count = dimension - 2
    if profile == "low_energy":
        tail = np.exp(-np.arange(count, dtype=np.float64) / 4.0)
    elif profile == "heavy_tail":
        tail = 1.0 / np.sqrt(np.arange(1, count + 1, dtype=np.float64))
    else:
        raise ValueError("unknown spectrum profile")
    tail *= 0.16 / float(np.sum(tail))
    center = np.concatenate(([0.42, 0.42], tail))
    if abs(float(np.sum(center)) - 1.0) > 1e-12 or np.min(center) <= 0:
        raise ValueError("density center invalid")
    return center


def _phase_noise(
    rng: np.random.Generator,
    center: np.ndarray,
    count: int,
    rare_probability: float,
) -> np.ndarray:
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(count, len(center)))
    kets = np.sqrt(center)[None, :] * np.exp(1j * phases)
    pure = np.einsum("ni,nj->nij", kets, kets.conj(), optimize=True)
    if rare_probability < 1.0:
        active = rng.random(count) < rare_probability
        pure[~active] = np.diag(center)
    return pure


def _density_trial(
    *,
    dimension: int,
    count: int,
    truth: float,
    family: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if count < 2 or truth not in (0.0, 0.05, 0.1, 0.12):
        raise ValueError("density trial design drift")
    center = _center_probabilities(dimension, str(family["spectrum_profile"]))
    left_weight = float(family["left_noise_weight"])
    right_weight = float(family["right_noise_weight"])
    shift = 0.0 if truth == 0.0 else truth / (
        (2.0 - left_weight - right_weight) / 2.0
    )
    direction = np.zeros(dimension, dtype=np.float64)
    direction[0], direction[1] = shift / 2.0, -shift / 2.0
    if np.min(center - np.abs(direction)) < -1e-15:
        raise ValueError("density effect violates PSD envelope")
    base_left = np.diag(center + direction).astype(np.complex128)
    base_right = np.diag(center - direction).astype(np.complex128)
    population = np.diag(center).astype(np.complex128)
    rng = np.random.default_rng(seed)
    rare = float(family["rare_probability"])
    left = (
        (1.0 - left_weight) * base_left
        + left_weight * _phase_noise(rng, center, count, rare)
    )
    right = (
        (1.0 - right_weight) * base_right
        + right_weight * _phase_noise(rng, center, count, rare)
    )
    population_left = (1.0 - left_weight) * base_left + left_weight * population
    population_right = (
        (1.0 - right_weight) * base_right + right_weight * population
    )
    analytic = _half_trace_norm(population_left - population_right)
    if abs(analytic - truth) > 1e-12:
        raise ValueError("analytic density truth drift")
    if bool(family["coherent_unitary"]):
        indices = np.arange(dimension)
        unitary = np.exp(
            2j * np.pi * np.outer(indices, indices) / float(dimension)
        ) / math.sqrt(dimension)
        left = np.einsum(
            "ij,njk,lk->nil", unitary, left, unitary.conj(), optimize=True
        )
        right = np.einsum(
            "ij,njk,lk->nil", unitary, right, unitary.conj(), optimize=True
        )
    _validate_density_stack(left, "left")
    _validate_density_stack(right, "right")
    return left, right, analytic


def _validate_density_stack(stack: np.ndarray, name: str) -> None:
    value = np.asarray(stack, dtype=np.complex128)
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError(f"{name} density shape invalid")
    hermitian = 0.5 * (value + value.conj().transpose(0, 2, 1))
    traces = np.trace(hermitian, axis1=1, axis2=2)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if (
        not np.all(np.isfinite(value.real))
        or not np.all(np.isfinite(value.imag))
        or float(np.max(np.abs(value - hermitian))) > 1e-9
        or float(np.max(np.abs(traces - 1.0))) > 1e-9
        or float(np.min(eigenvalues)) < -1e-9
    ):
        raise ValueError(f"{name} contains a non-physical density")


def _half_trace_norm(matrix: np.ndarray) -> float:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def _density_ucb(
    left: np.ndarray,
    right: np.ndarray,
    *,
    confidence: float,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    differences = left - right
    mean = np.mean(differences, axis=0)
    centered = differences - mean[None, :, :]
    estimate = _half_trace_norm(mean)
    rng = np.random.default_rng(seed)
    radii = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 64):
        stop = min(start + 64, replicates)
        signs = (
            rng.integers(
                0, 2, size=(stop - start, len(differences)), dtype=np.int8
            ).astype(np.float64)
            * 2.0
            - 1.0
        )
        perturbation = np.einsum(
            "bn,nij->bij", signs, centered, optimize=True
        ) / len(differences)
        perturbation = 0.5 * (
            perturbation + perturbation.conj().transpose(0, 2, 1)
        )
        radii[start:stop] = 0.5 * np.sum(
            np.abs(np.linalg.eigvalsh(perturbation)), axis=1
        )
    radius = float(np.quantile(radii, confidence, method="higher"))
    return estimate, radius, estimate + radius


def _density_worker(
    payload: tuple[dict[str, str], dict[str, Any], dict[str, Any]],
) -> tuple[float, float, float, bool, bool]:
    row, family, design = payload
    left, right, truth = _density_trial(
        dimension=int(row["dimension"]),
        count=int(row["cluster_count"]),
        # The source value is a recomputed physical quantity and may be a few
        # ulps away from the categorical design point.  Generate from the
        # independently reconstructed canonical spec; _validate_density_rows
        # separately checks that the saved value agrees with that spec.
        truth=float(design["effect"]),
        family=family,
        seed=int(row["trial_seed"]),
    )
    estimate, radius, upper = _density_ucb(
        left,
        right,
        confidence=float(design["confidence"]),
        replicates=int(design["multiplier_replicates"]),
        seed=int(row["multiplier_seed"]),
    )
    return (
        estimate,
        radius,
        upper,
        upper + 1e-15 >= truth,
        upper <= float(design["margin"]),
    )


def scale_blueprint(
    t05_blueprint: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (
        t05_blueprint.get("task_id") != "T-RISK-20260728-05"
        or t05_blueprint.get("schema_version")
        != "PHASE9-T04-FULL-GATE-BLUEPRINT-V1"
        or t05_blueprint.get("analysis_sha256") != _self_hash(t05_blueprint)
        or t05_blueprint.get("gate_count") != 3043
        or t05_blueprint.get("stochastic_gate_count") != 3037
        or t05_blueprint.get("deterministic_gate_count") != 6
    ):
        raise ValueError("T05 blueprint identity/count/hash drift")
    gates = t05_blueprint.get("gates")
    if not isinstance(gates, list) or len(gates) != 3043:
        raise ValueError("T05 blueprint gate denominator drift")
    identifiers = [gate.get("gate_id") for gate in gates if isinstance(gate, dict)]
    if len(identifiers) != 3043 or len(set(identifiers)) != 3043:
        raise ValueError("T05 blueprint gate IDs duplicate or missing")
    mapping = {
        384: int(candidate["state_clusters"]),
        768: int(candidate["round_clusters"]),
        2304: int(candidate["aggregate_fault_clusters"]),
        0: 0,
    }
    output: list[dict[str, Any]] = []
    for raw in gates:
        if not isinstance(raw, dict):
            raise ValueError("T05 blueprint gate is not an object")
        count = raw.get("cluster_count")
        if isinstance(count, bool) or int(count) not in mapping:
            raise ValueError("T05 blueprint contains unlinked cluster count")
        gate = dict(raw)
        gate["cluster_count"] = mapping[int(count)]
        if (
            not isinstance(gate.get("gate_id"), str)
            or not isinstance(gate.get("family"), str)
            or not isinstance(gate.get("cluster_scope"), str)
            or gate.get("direction") not in ("upper", "lower")
            or not isinstance(gate.get("deterministic"), bool)
            or not math.isfinite(float(gate.get("normalized_sd", math.nan)))
            or float(gate["normalized_sd"]) <= 0
            or not math.isfinite(float(gate.get("margin", math.nan)))
            or float(gate["margin"]) <= 0
        ):
            raise ValueError("T05 blueprint gate contract malformed")
        output.append(gate)
    if (
        len(output) != 3043
        or sum(not gate["deterministic"] for gate in output) != 3037
        or sum(gate["deterministic"] for gate in output) != 6
        or {gate["cluster_count"] for gate in output}
        != {
            0,
            int(candidate["state_clusters"]),
            int(candidate["round_clusters"]),
            int(candidate["aggregate_fault_clusters"]),
        }
    ):
        raise ValueError("linked blueprint mapping/count drift")
    return output


def _read_source(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        rows = list(reader)
    if len(header) != len(set(header)) or "row_type" not in header:
        raise ValueError("source data header invalid")
    allowed = {
        "density_trial",
        "maxt_replicate",
        "maxt_meta",
        "maxt_power",
        "maxt_selection_summary",
        "density_selection_summary",
        "density_confirmation_summary",
    }
    for row in rows:
        if not row.get("row_type"):
            if row.get("split") == "selection" and row.get("dimension"):
                row["row_type"] = "density_selection_summary"
            elif str(row.get("cell_id", "")).startswith("confirmation__"):
                row["row_type"] = "density_confirmation_summary"
        if row.get("row_type") not in allowed:
            raise ValueError(f"unknown source row type: {row.get('row_type')}")
    return header, rows


def _density_specs(
    config: Mapping[str, Any],
    *,
    split: str,
    selected: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    density = config["density"]
    specs: list[dict[str, Any]] = []
    if split == "selection":
        index = 0
        for candidate in config["linked_count_grid"]:
            for dimension in density["dimensions"]:
                specs.append({
                    "split": split,
                    "candidate_scale": float(candidate["scale"]),
                    "cluster_count": int(candidate["state_clusters"]),
                    "cell_index": index,
                    "cell_id": (
                        f"selection__n{candidate['state_clusters']}__d{dimension}"
                        "__heteroskedastic__effect_0.050"
                    ),
                    "family": density["selection_family"],
                    "dimension": int(dimension),
                    "effect": float(density["selection_effect"]),
                    "trials": int(density["selection_trials_per_cell"]),
                    "trial_seed_base": int(density["selection_trial_seed_base"]),
                    "multiplier_seed_base": int(
                        density["selection_multiplier_seed_base"]
                    ),
                })
                index += 1
    elif split == "confirmation":
        if selected is None:
            raise ValueError("confirmation density requires selected count")
        index = 0
        for family in sorted(density["confirmation_families"]):
            for dimension in density["dimensions"]:
                for effect in density["confirmation_effects"]:
                    specs.append({
                        "split": split,
                        "candidate_scale": float(selected["scale"]),
                        "cluster_count": int(selected["state_clusters"]),
                        "cell_index": index,
                        "cell_id": (
                            f"confirmation__n{selected['state_clusters']}__"
                            f"d{dimension}__{family}__effect_{effect:.3f}"
                        ),
                        "family": family,
                        "dimension": int(dimension),
                        "effect": float(effect),
                        "trials": int(density["confirmation_trials_per_cell"]),
                        "trial_seed_base": int(
                            density["confirmation_trial_seed_base"]
                        ),
                        "multiplier_seed_base": int(
                            density["confirmation_multiplier_seed_base"]
                        ),
                    })
                    index += 1
    else:
        raise ValueError("unknown density split")
    return specs


def _validate_density_rows(
    rows: Sequence[Mapping[str, str]],
    specs: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for spec in specs:
        for trial in range(int(spec["trials"])):
            expected[(str(spec["cell_id"]), trial)] = {
                **spec,
                "trial": trial,
                "trial_seed": (
                    int(spec["trial_seed_base"])
                    + int(spec["cell_index"]) * int(spec["trials"])
                    + trial
                ),
                "multiplier_seed": (
                    int(spec["multiplier_seed_base"])
                    + int(spec["cell_index"]) * int(spec["trials"])
                    + trial
                ),
            }
    if len(rows) != len(expected):
        raise ValueError("density source denominator drift")
    result: list[tuple[dict[str, str], dict[str, Any]]] = []
    seen: set[tuple[str, int]] = set()
    for raw in rows:
        row = dict(raw)
        key = (str(row.get("cell_id")), _int(row.get("trial"), "density trial"))
        if key in seen or key not in expected:
            raise ValueError("density source duplicate/unexpected row")
        seen.add(key)
        spec = expected[key]
        checks = {
            "split": spec["split"],
            "family": spec["family"],
        }
        if any(row.get(name) != value for name, value in checks.items()):
            raise ValueError("density row categorical field drift")
        numeric = {
            "candidate_scale": spec["candidate_scale"],
            "cluster_count": spec["cluster_count"],
            "dimension": spec["dimension"],
            "true_distance": spec["effect"],
            "trial_seed": spec["trial_seed"],
            "multiplier_seed": spec["multiplier_seed"],
        }
        if any(
            abs(_float(row.get(name), name) - float(value)) > 1e-15
            for name, value in numeric.items()
        ):
            raise ValueError("density row design/seed field drift")
        result.append((row, spec))
    if seen != set(expected):
        raise ValueError("density source missing expected row")
    trial_seeds = [int(spec["trial_seed"]) for _, spec in result]
    multiplier_seeds = [int(spec["multiplier_seed"]) for _, spec in result]
    if (
        len(trial_seeds) != len(set(trial_seeds))
        or len(multiplier_seeds) != len(set(multiplier_seeds))
        or set(trial_seeds) & set(multiplier_seeds)
    ):
        raise ValueError("density source seed reuse")
    return result


def _density_summary(
    config: Mapping[str, Any],
    split: str,
    recomputed: Sequence[tuple[dict[str, str], dict[str, Any], tuple[Any, ...]]],
) -> tuple[list[dict[str, Any]], dict[float, bool] | bool]:
    by_cell: dict[str, list[tuple[dict[str, str], dict[str, Any], tuple[Any, ...]]]] = {}
    for item in recomputed:
        by_cell.setdefault(str(item[0]["cell_id"]), []).append(item)
    summaries: list[dict[str, Any]] = []
    if split == "selection":
        contract = config["density"]["selection_wilson"]
        decisions = {
            float(candidate["scale"]): True
            for candidate in config["linked_count_grid"]
        }
    else:
        contract = config["density"]["confirmation_wilson"]
        all_pass = True
    for cell_id, values in sorted(by_cell.items()):
        spec = values[0][1]
        trials = len(values)
        covered = sum(bool(item[2][3]) for item in values)
        equivalent = sum(bool(item[2][4]) for item in values)
        cov_lcb, cov_ucb = _wilson(
            covered,
            trials,
            confidence=float(contract["confidence"]),
            comparisons=int(contract["comparisons"]),
        )
        eq_lcb, eq_ucb = _wilson(
            equivalent,
            trials,
            confidence=float(contract["confidence"]),
            comparisons=int(contract["comparisons"]),
        )
        if split == "selection":
            passed = (
                cov_lcb >= float(contract["coverage_lcb_minimum"])
                and eq_lcb >= float(contract["local_equivalence_lcb_minimum"])
            )
            decisions[float(spec["candidate_scale"])] &= passed
            summaries.append({
                "split": split,
                "candidate_scale": float(spec["candidate_scale"]),
                "cluster_count": int(spec["cluster_count"]),
                "dimension": int(spec["dimension"]),
                "trials": trials,
                "coverage_successes": covered,
                "coverage_lcb": cov_lcb,
                "equivalence_successes": equivalent,
                "equivalence_rate": equivalent / trials,
                "equivalence_lcb": eq_lcb,
                "equivalence_ucb": eq_ucb,
                "gate_pass": passed,
            })
        else:
            coverage_pass = (
                covered / trials >= float(contract["coverage_rate_minimum"])
                and cov_lcb >= float(contract["coverage_lcb_minimum"])
            )
            effect = float(spec["effect"])
            if effect == 0.0:
                power_pass = (
                    eq_lcb >= float(contract["null_equivalence_lcb_minimum"])
                )
            elif effect == 0.05:
                power_pass = (
                    eq_lcb >= float(contract["local_equivalence_lcb_minimum"])
                )
            elif effect == 0.1:
                power_pass = (
                    eq_ucb <= float(contract["boundary_equivalence_ucb_maximum"])
                )
            elif effect == 0.12:
                power_pass = (
                    eq_ucb <= float(contract["outside_equivalence_ucb_maximum"])
                )
            else:
                raise ValueError("unregistered confirmation effect")
            passed = coverage_pass and power_pass
            all_pass &= passed
            summaries.append({
                "cell_id": cell_id,
                "family": spec["family"],
                "dimension": int(spec["dimension"]),
                "true_distance": effect,
                "cluster_count": int(spec["cluster_count"]),
                "trials": trials,
                "coverage_successes": covered,
                "coverage_rate": covered / trials,
                "coverage_lcb": cov_lcb,
                "coverage_ucb": cov_ucb,
                "equivalence_successes": equivalent,
                "equivalence_rate": equivalent / trials,
                "equivalence_lcb": eq_lcb,
                "equivalence_ucb": eq_ucb,
                "coverage_gate_pass": coverage_pass,
                "power_gate_pass": power_pass,
                "gate_pass": passed,
            })
    return summaries, decisions if split == "selection" else all_pass


def _influences(
    gates: Sequence[Mapping[str, Any]],
    count: int,
    *,
    family_rho: float,
    scope_rho: float,
    seed: int,
) -> np.ndarray:
    selected = [gate for gate in gates if int(gate["cluster_count"]) == count]
    families = sorted({str(gate["family"]) for gate in selected})
    scopes = sorted({str(gate["cluster_scope"]) for gate in selected})
    rng = np.random.default_rng(seed + count)
    family_values = {
        name: rng.standard_normal(count) for name in families
    }
    scope_values = {
        name: rng.standard_normal(count) for name in scopes
    }
    residual = 1.0 - family_rho - scope_rho
    if residual <= 0:
        raise ValueError("invalid maxT residual correlation")
    matrix = np.empty((count, len(selected)), dtype=np.float64)
    for index, gate in enumerate(selected):
        values = (
            math.sqrt(family_rho) * family_values[str(gate["family"])]
            + math.sqrt(scope_rho) * scope_values[str(gate["cluster_scope"])]
            + math.sqrt(residual) * rng.standard_normal(count)
        )
        values -= float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0:
            raise ValueError("degenerate maxT influence")
        matrix[:, index] = values / standard_deviation
    return matrix


def recompute_maxt(
    config: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> tuple[float, list[float], list[dict[str, Any]], dict[str, bool]]:
    contract = config["joint_maxt"]
    stochastic = [gate for gate in gates if not gate["deterministic"]]
    if len(stochastic) != 3037:
        raise ValueError("maxT stochastic gate denominator drift")
    correlation = contract["correlation_model"]
    replicates = int(contract["multiplier_replicates"])
    maxima = np.zeros(replicates, dtype=np.float64)
    influence_seed = int(contract[f"{split}_influence_seed_base"])
    rademacher_seed = int(contract[f"{split}_rademacher_seed_base"])
    for count in sorted({int(gate["cluster_count"]) for gate in stochastic}):
        selected = [
            gate for gate in stochastic if int(gate["cluster_count"]) == count
        ]
        influence = _influences(
            stochastic,
            count,
            family_rho=float(correlation["family_shared"]),
            scope_rho=float(correlation["scope_shared"]),
            seed=influence_seed,
        )
        if influence.shape != (count, len(selected)):
            raise ValueError("maxT influence shape drift")
        rng = np.random.default_rng(rademacher_seed + count)
        signs = (
            rng.integers(
                0, 2, size=(replicates, count), dtype=np.int8
            ).astype(np.float64)
            * 2.0
            - 1.0
        )
        standardized = signs @ influence / math.sqrt(count)
        maxima = np.maximum(
            maxima, np.max(np.abs(standardized), axis=1)
        )
    critical = float(
        contract["factor"]
        * np.quantile(
            maxima, float(contract["confidence"]), method="higher"
        )
    )
    power = contract["power"]
    trials = int(power["pseudoexperiments"])
    rng = np.random.default_rng(int(contract[f"{split}_power_seed_base"]))
    families = sorted({str(gate["family"]) for gate in stochastic})
    scopes = sorted({str(gate["cluster_scope"]) for gate in stochastic})
    family_index = {name: index for index, name in enumerate(families)}
    scope_index = {name: index for index, name in enumerate(scopes)}
    family_noise = rng.standard_normal((trials, len(families)))
    scope_noise = rng.standard_normal((trials, len(scopes)))
    passed = np.empty((trials, len(stochastic)), dtype=np.bool_)
    estimates = np.empty((trials, len(stochastic)), dtype=np.float32)
    for index, gate in enumerate(stochastic):
        standard_error = float(gate["normalized_sd"]) * math.sqrt(
            2.0 / int(gate["cluster_count"])
        )
        noise = (
            math.sqrt(float(correlation["family_shared"]))
            * family_noise[:, family_index[str(gate["family"])]]
            + math.sqrt(float(correlation["scope_shared"]))
            * scope_noise[:, scope_index[str(gate["cluster_scope"])]]
            + math.sqrt(float(correlation["idiosyncratic"]))
            * rng.standard_normal(trials)
        )
        estimate = standard_error * noise
        estimates[:, index] = estimate
        passed[:, index] = (
            np.abs(estimate) + critical * standard_error <= 1.0
        )
    failures = np.sum(~passed, axis=1)
    rows: list[dict[str, Any]] = []
    decisions: dict[str, bool] = {}

    def add(
        case: str,
        family: str,
        effect: float,
        successes: int,
        injection: str | None = None,
    ) -> None:
        lower, upper = _wilson(
            successes,
            trials,
            confidence=float(power["confidence_for_bounds"]),
            comparisons=1,
        )
        row = {
            "row_type": "maxt_power",
            "case": case,
            "family": family,
            "effect_ratio": effect,
            "successes": successes,
            "trials": trials,
            "point": successes / trials,
            "wilson_lcb": lower,
            "wilson_ucb": upper,
        }
        if injection is not None:
            row["injection_gate_id"] = injection
        rows.append(row)

    add(
        "null_global_equivalence",
        "all",
        0.0,
        int(np.count_nonzero(failures == 0)),
    )
    decisions["maxt_null_global_power"] = (
        rows[-1]["wilson_lcb"]
        >= float(power["null_global_equivalence_lcb_minimum"])
    )
    for family in families:
        candidates = [
            (index, gate)
            for index, gate in enumerate(stochastic)
            if str(gate["family"]) == family
        ]
        injection_index, injection_gate = sorted(
            candidates,
            key=lambda item: (
                -float(item[1]["normalized_sd"])
                * math.sqrt(2.0 / int(item[1]["cluster_count"])),
                str(item[1]["gate_id"]),
            ),
        )[0]
        other_pass = (
            failures - (~passed[:, injection_index]).astype(np.int64) == 0
        )
        standard_error = float(
            injection_gate["normalized_sd"]
        ) * math.sqrt(2.0 / int(injection_gate["cluster_count"]))
        base_estimate = estimates[:, injection_index].astype(np.float64)
        cases = (
            (
                "local_single_gate_equivalence",
                float(power["local_effect_ratio"]),
                "local_single_gate_equivalence_lcb_minimum",
                "lower",
            ),
            (
                "boundary_false_equivalence",
                float(power["boundary_effect_ratio"]),
                "boundary_false_equivalence_ucb_maximum",
                "upper",
            ),
            (
                "outside_false_equivalence",
                float(power["outside_effect_ratio"]),
                "outside_false_equivalence_ucb_maximum",
                "upper",
            ),
        )
        for case, effect, threshold, direction in cases:
            injected = (
                np.abs(base_estimate + effect)
                + critical * standard_error
                <= 1.0
            )
            successes = int(np.count_nonzero(other_pass & injected))
            add(
                case,
                family,
                effect,
                successes,
                str(injection_gate["gate_id"]),
            )
            if direction == "lower":
                decision = (
                    rows[-1]["wilson_lcb"] >= float(power[threshold])
                )
            else:
                decision = (
                    rows[-1]["wilson_ucb"] <= float(power[threshold])
                )
            decisions[f"maxt_power::{case}::{family}"] = decision
    decisions["maxt_critical_finite_and_above_pointwise"] = (
        math.isfinite(critical) and critical > 1.6448536269514722
    )
    return critical, maxima.tolist(), rows, decisions


def _max_numeric_delta(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    names: Iterable[str],
    context: str,
) -> float:
    maximum = 0.0
    for name in names:
        if name not in observed:
            raise ValueError(f"{context} missing {name}")
        maximum = max(
            maximum,
            _assert_close(observed[name], expected[name], f"{context}/{name}"),
        )
    return maximum


def _compare_maxt_source(
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    scale: float,
    critical: float,
    maxima: Sequence[float],
    power_rows: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, bool],
) -> float:
    scoped = [
        row for row in rows
        if row.get("split") == split
        and abs(_float(row.get("candidate_scale"), "candidate scale") - scale)
        < 1e-15
    ]
    raw = [row for row in scoped if row["row_type"] == "maxt_replicate"]
    if (
        len(raw) != 199
        or {int(row["replicate"]) for row in raw} != set(range(199))
    ):
        raise ValueError(f"{split}/{scale} maxT replicate denominator drift")
    maximum = max(
        abs(_float(row["max_statistic"], "max statistic") - maxima[int(row["replicate"])])
        for row in raw
    )
    if maximum > 2e-12:
        raise ValueError(f"{split}/{scale} maxT replicate mismatch: {maximum}")
    observed_power = {
        (row["case"], row["family"]): row
        for row in scoped if row["row_type"] == "maxt_power"
    }
    if len(observed_power) != len(power_rows):
        raise ValueError(f"{split}/{scale} maxT power denominator drift")
    for expected in power_rows:
        key = (expected["case"], expected["family"])
        observed = observed_power.get(key)
        if observed is None:
            raise ValueError(f"{split}/{scale} missing maxT power {key}")
        if (
            int(observed["successes"]) != int(expected["successes"])
            or int(observed["trials"]) != int(expected["trials"])
            or _bool(observed["gate_pass"], "maxT power gate")
            is not bool(
                decisions[
                    "maxt_null_global_power"
                    if key == ("null_global_equivalence", "all")
                    else f"maxt_power::{key[0]}::{key[1]}"
                ]
            )
        ):
            raise ValueError(f"{split}/{scale} maxT power count/gate mismatch {key}")
        maximum = max(
            maximum,
            _max_numeric_delta(
                expected,
                observed,
                ("effect_ratio", "point", "wilson_lcb", "wilson_ucb"),
                f"{split}/{scale}/{key}",
            ),
        )
        if key != ("null_global_equivalence", "all") and (
            observed.get("injection_gate_id") != expected["injection_gate_id"]
        ):
            raise ValueError(f"{split}/{scale} injection gate drift {key}")
    meta = [row for row in scoped if row["row_type"] == "maxt_meta"]
    if len(meta) != 1 or _assert_close(
        meta[0]["point"], critical, f"{split}/{scale}/critical"
    ) > 2e-12:
        raise ValueError(f"{split}/{scale} maxT meta/critical drift")
    return maximum


def _summary_key(row: Mapping[str, Any], split: str) -> tuple[Any, ...]:
    if split == "selection":
        return (float(row["candidate_scale"]), int(row["dimension"]))
    return (str(row["cell_id"]),)


def _compare_density_summaries(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> float:
    by_key = {_summary_key(row, split): row for row in observed}
    if len(by_key) != len(expected):
        raise ValueError(f"{split} density summary denominator drift")
    maximum = 0.0
    numeric_names = (
        (
            "candidate_scale", "cluster_count", "dimension", "trials",
            "coverage_successes", "coverage_lcb", "equivalence_successes",
            "equivalence_rate", "equivalence_lcb", "equivalence_ucb",
        )
        if split == "selection"
        else (
            "dimension", "true_distance", "cluster_count", "trials",
            "coverage_successes", "coverage_rate", "coverage_lcb",
            "coverage_ucb", "equivalence_successes", "equivalence_rate",
            "equivalence_lcb", "equivalence_ucb",
        )
    )
    boolean_names = (
        ("gate_pass",)
        if split == "selection"
        else ("coverage_gate_pass", "power_gate_pass", "gate_pass")
    )
    for item in expected:
        key = _summary_key(item, split)
        actual = by_key.get(key)
        if actual is None:
            raise ValueError(f"{split} missing density summary {key}")
        maximum = max(
            maximum,
            _max_numeric_delta(
                item, actual, numeric_names, f"{split}/density-summary/{key}"
            ),
        )
        for name in boolean_names:
            if _bool(actual[name], f"{split}/{key}/{name}") is not bool(item[name]):
                raise ValueError(f"{split} density summary boolean drift {key}/{name}")
        if split == "confirmation" and (
            actual.get("family") != item["family"]
            or actual.get("cell_id") != item["cell_id"]
        ):
            raise ValueError("confirmation density summary identity drift")
    return maximum


def _resource_summary(
    config: Mapping[str, Any],
    selected: Mapping[str, Any] | None,
    recorded_free_disk: int | None,
) -> tuple[dict[str, Any], bool]:
    resource = config["resource"]
    if selected is None:
        return {
            "selected_scale": None,
            "exact_t04_rows": None,
            "estimated_wall_seconds": None,
            "estimated_artifact_bytes": None,
            "estimated_rss_bytes": None,
            "free_disk_bytes": recorded_free_disk,
            "fresh_t04_layer_benchmark_required": True,
        }, False
    scale = float(selected["scale"])
    summary = {
        "selected_scale": scale,
        "exact_t04_rows": int(round(int(resource["base_t04_rows"]) * scale)),
        "estimated_wall_seconds": (
            float(resource["base_wall_seconds_4_workers"]) * scale
        ),
        "estimated_artifact_bytes": math.ceil(
            int(resource["base_artifact_bytes"]) * scale
        ),
        "estimated_rss_bytes": math.ceil(
            int(resource["base_rss_bytes"]) * scale
        ),
        "free_disk_bytes": recorded_free_disk,
        "fresh_t04_layer_benchmark_required": True,
    }
    passed = (
        summary["estimated_wall_seconds"] <= resource["maximum_wall_seconds"]
        and summary["estimated_artifact_bytes"]
        <= resource["maximum_artifact_bytes"]
        and summary["estimated_rss_bytes"] <= resource["maximum_rss_bytes"]
        and recorded_free_disk is not None
        and recorded_free_disk - summary["estimated_artifact_bytes"]
        >= resource["minimum_free_disk_after_estimate_bytes"]
    )
    return summary, bool(passed)


def select_smallest(
    config: Mapping[str, Any],
    density_decisions: Mapping[float, bool],
    maxt_decisions: Mapping[float, bool],
) -> Mapping[str, Any] | None:
    return next(
        (
            candidate for candidate in config["linked_count_grid"]
            if density_decisions.get(float(candidate["scale"])) is True
            and maxt_decisions.get(float(candidate["scale"])) is True
        ),
        None,
    )


def _check_report_claims(report: Mapping[str, Any]) -> None:
    if report.get("qualified_claim") is not None:
        raise ValueError("writer report qualified claim must remain null")
    boundary = report.get("claim_boundary")
    if not isinstance(boundary, dict) or any(
        boundary.get(name) is not None for name in NULL_CLAIMS
    ):
        raise ValueError("writer report populated prohibited claim")
    if boundary.get("t04_scientific_execution_released") is not False:
        raise ValueError("T04 scientific execution release is forbidden")
    if boundary.get("count_design_only") is not True:
        raise ValueError("writer report count-design boundary drift")


def verify(
    root: Path | None = None,
    *,
    workers: int = 4,
    write: bool = True,
) -> dict[str, Any]:
    if workers not in (1, 2, 3, 4):
        raise ValueError("verifier workers must be in 1..4")
    base = (root or _root()).resolve()
    config_path = base / CONFIG_PATH
    config = _load(config_path)
    validate_config(config)
    if not _validate_import_independence():
        raise ValueError("verifier imports a prohibited project module")
    paths = config["artifact_paths"]
    report_path = base / paths["report"]
    source_path = base / paths["source_data"]
    report = _load(report_path)
    if (
        report.get("task_id") != TASK_ID
        or report.get("schema_version")
        != "PHASE9-COUNT-SELECTION-CONFIRMATION-REPORT-V1"
        or report.get("analysis_sha256") != _self_hash(report)
        or report.get("formal_outcomes_accessed") is not False
    ):
        raise ValueError("T06 writer report identity/hash/firewall drift")
    _check_report_claims(report)

    bindings = report.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("writer report bindings missing")
    required_bindings = {
        "config",
        *config["parent_artifacts"].keys(),
        "source_data",
        "writer_source",
        "selected_blueprint",
    }
    if set(bindings) != required_bindings:
        raise ValueError("writer report binding set drift")
    for name, binding in bindings.items():
        if name == "selected_blueprint" and binding is None:
            continue
        if not isinstance(binding, dict):
            raise ValueError(f"writer binding {name} malformed")
        live = _safe_bound_path(base, binding)
        if name == "config" and live != config_path.resolve():
            raise ValueError("writer config binding path drift")
        if name in config["parent_artifacts"] and live != (
            base / config["parent_artifacts"][name]
        ).resolve():
            raise ValueError(f"writer parent binding path drift: {name}")
        if name == "source_data" and live != source_path.resolve():
            raise ValueError("writer source-data binding path drift")
        if name == "writer_source" and live != (
            base / "cnn_fpga/benchmark/phase9_count_selection_confirmation.py"
        ).resolve():
            raise ValueError("writer source binding path drift")

    t05_report = _load(base / config["parent_artifacts"]["t05_report"])
    t05_verification = _load(
        base / config["parent_artifacts"]["t05_verification"]
    )
    t05_blueprint = _load(base / config["parent_artifacts"]["t05_blueprint"])
    t05_config = _load(base / config["parent_artifacts"]["t05_config"])
    if (
        t05_report.get("analysis_sha256") != _self_hash(t05_report)
        or t05_verification.get("analysis_sha256")
        != _self_hash(t05_verification)
        or t05_blueprint.get("analysis_sha256") != _self_hash(t05_blueprint)
        or t05_report.get("verdict")
        != "FAIL_T04_STATISTICAL_PREREGISTRATION_BLOCKED"
        or t05_verification.get("verdict")
        != "PASS_INDEPENDENT_T04_STATISTICAL_NO_GO_VERIFICATION"
        or t05_verification.get("t04_preregistration_released") is not False
    ):
        raise ValueError("T05 verified NO-GO lineage drift")
    if any(
        t05_report.get("claim_boundary", {}).get(name) is not None
        or t05_verification.get("claim_boundary", {}).get(name) is not None
        for name in NULL_CLAIMS
    ):
        raise ValueError("T05 parent claim boundary drift")
    families = t05_config.get("density_uq", {}).get("families")
    if not isinstance(families, dict) or set(families) != {
        "heavy_tail_rare_coherent",
        "heteroskedastic_coherent",
        "low_energy_balanced",
    }:
        raise ValueError("T05 density family contract drift")

    header, source_rows = _read_source(source_path)
    selection_density_raw = [
        row for row in source_rows
        if row["row_type"] == "density_trial" and row.get("split") == "selection"
    ]
    selection_pairs = _validate_density_rows(
        selection_density_raw, _density_specs(config, split="selection")
    )
    density_design = {
        "confidence": config["density"]["confidence"],
        "multiplier_replicates": config["density"]["multiplier_replicates"],
        "margin": config["density"]["margin"],
    }
    payloads = [
        (
            row,
            dict(families[spec["family"]]),
            {**density_design, "effect": float(spec["effect"])},
        )
        for row, spec in selection_pairs
    ]
    selection_recomputed: list[
        tuple[dict[str, str], dict[str, Any], tuple[Any, ...]]
    ] = []
    max_density_delta = 0.0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (row, spec), values in zip(
            selection_pairs,
            pool.map(_density_worker, payloads, chunksize=1),
        ):
            delta = max(
                abs(values[0] - _float(row["estimate"], "density estimate")),
                abs(values[1] - _float(row["raw_radius"], "density radius")),
                abs(values[2] - _float(row["upper_bound"], "density upper")),
            )
            if delta > 2e-12:
                raise ValueError(f"selection density raw mismatch: {delta}")
            if (
                _bool(row["covered"], "selection covered") is not values[3]
                or _bool(row["equivalence_pass"], "selection equivalence")
                is not values[4]
            ):
                raise ValueError("selection density boolean mismatch")
            max_density_delta = max(max_density_delta, delta)
            selection_recomputed.append((row, spec, values))
    selection_summary, density_decisions_raw = _density_summary(
        config, "selection", selection_recomputed
    )
    assert isinstance(density_decisions_raw, dict)
    density_decisions = density_decisions_raw
    max_summary_delta = _compare_density_summaries(
        selection_summary,
        report["selection"]["density_summaries"],
        split="selection",
    )
    max_summary_delta = max(
        max_summary_delta,
        _compare_density_summaries(
            selection_summary,
            [
                row for row in source_rows
                if row["row_type"] == "density_selection_summary"
            ],
            split="selection",
        ),
    )

    maxt_decisions: dict[float, bool] = {}
    max_maxt_delta = 0.0
    expected_selection_maxt_summaries: list[dict[str, Any]] = []
    for candidate in config["linked_count_grid"]:
        scale = float(candidate["scale"])
        gates = scale_blueprint(t05_blueprint, candidate)
        critical, maxima, power_rows, decisions = recompute_maxt(
            config, gates, split="selection"
        )
        passed = all(decisions.values())
        maxt_decisions[scale] = passed
        max_maxt_delta = max(
            max_maxt_delta,
            _compare_maxt_source(
                source_rows,
                split="selection",
                scale=scale,
                critical=critical,
                maxima=maxima,
                power_rows=power_rows,
                decisions=decisions,
            ),
        )
        expected_selection_maxt_summaries.append({
            "row_type": "maxt_selection_summary",
            "split": "selection",
            "candidate_scale": scale,
            "cluster_count": int(candidate["state_clusters"]),
            "critical": critical,
            "gate_pass": passed,
            "failed_gate_count": sum(not value for value in decisions.values()),
        })
    observed_maxt_summaries = report["selection"]["maxt_summaries"]
    source_maxt_summaries = [
        row for row in source_rows
        if row["row_type"] == "maxt_selection_summary"
    ]
    for observed_collection in (observed_maxt_summaries, source_maxt_summaries):
        if len(observed_collection) != 4:
            raise ValueError("selection maxT summary denominator drift")
        by_scale = {
            float(row["candidate_scale"]): row for row in observed_collection
        }
        for expected in expected_selection_maxt_summaries:
            actual = by_scale.get(expected["candidate_scale"])
            if actual is None:
                raise ValueError("selection maxT candidate summary missing")
            max_summary_delta = max(
                max_summary_delta,
                _max_numeric_delta(
                    expected,
                    actual,
                    (
                        "candidate_scale", "cluster_count", "critical",
                        "failed_gate_count",
                    ),
                    "selection/maxT-summary",
                ),
            )
            if _bool(actual["gate_pass"], "selection maxT gate") is not expected[
                "gate_pass"
            ]:
                raise ValueError("selection maxT summary decision drift")

    selected = select_smallest(config, density_decisions, maxt_decisions)
    if report["selection"].get("selected") != selected:
        raise ValueError("writer did not select the smallest passing candidate")
    if report["selection"].get("rule") != config["selection_rule"]:
        raise ValueError("writer selection rule receipt drift")

    confirmation_density_raw = [
        row for row in source_rows
        if row["row_type"] == "density_trial"
        and row.get("split") == "confirmation"
    ]
    confirmation_summary: list[dict[str, Any]] = []
    confirmation_density_pass = False
    confirmation_maxt_pass = False
    confirmation_critical: float | None = None
    confirmation_power_rows: list[dict[str, Any]] = []
    selected_blueprint: dict[str, Any] | None = None
    if selected is None:
        if (
            confirmation_density_raw
            or report["confirmation"]["density_summaries"]
            or report["confirmation"]["maxt_power_rows"]
            or bindings["selected_blueprint"] is not None
        ):
            raise ValueError("confirmation evidence exists without a selected count")
    else:
        confirmation_pairs = _validate_density_rows(
            confirmation_density_raw,
            _density_specs(config, split="confirmation", selected=selected),
        )
        selection_seed_set = {
            int(spec["trial_seed"]) for _, spec in selection_pairs
        } | {
            int(spec["multiplier_seed"]) for _, spec in selection_pairs
        }
        confirmation_seed_set = {
            int(spec["trial_seed"]) for _, spec in confirmation_pairs
        } | {
            int(spec["multiplier_seed"]) for _, spec in confirmation_pairs
        }
        if selection_seed_set & confirmation_seed_set:
            raise ValueError("selection/confirmation density seed reuse")
        payloads = [
            (
                row,
                dict(families[spec["family"]]),
                {**density_design, "effect": float(spec["effect"])},
            )
            for row, spec in confirmation_pairs
        ]
        confirmation_recomputed = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for (row, spec), values in zip(
                confirmation_pairs,
                pool.map(_density_worker, payloads, chunksize=1),
            ):
                delta = max(
                    abs(values[0] - _float(row["estimate"], "density estimate")),
                    abs(values[1] - _float(row["raw_radius"], "density radius")),
                    abs(values[2] - _float(row["upper_bound"], "density upper")),
                )
                if delta > 2e-12:
                    raise ValueError(f"confirmation density raw mismatch: {delta}")
                if (
                    _bool(row["covered"], "confirmation covered") is not values[3]
                    or _bool(
                        row["equivalence_pass"], "confirmation equivalence"
                    )
                    is not values[4]
                ):
                    raise ValueError("confirmation density boolean mismatch")
                max_density_delta = max(max_density_delta, delta)
                confirmation_recomputed.append((row, spec, values))
        confirmation_summary, confirmation_density_pass_raw = _density_summary(
            config, "confirmation", confirmation_recomputed
        )
        assert isinstance(confirmation_density_pass_raw, bool)
        confirmation_density_pass = confirmation_density_pass_raw
        max_summary_delta = max(
            max_summary_delta,
            _compare_density_summaries(
                confirmation_summary,
                report["confirmation"]["density_summaries"],
                split="confirmation",
            ),
            _compare_density_summaries(
                confirmation_summary,
                [
                    row for row in source_rows
                    if row["row_type"] == "density_confirmation_summary"
                ],
                split="confirmation",
            ),
        )
        selected_gates = scale_blueprint(t05_blueprint, selected)
        confirmation_critical, maxima, confirmation_power_rows, decisions = (
            recompute_maxt(config, selected_gates, split="confirmation")
        )
        confirmation_maxt_pass = all(decisions.values())
        max_maxt_delta = max(
            max_maxt_delta,
            _compare_maxt_source(
                source_rows,
                split="confirmation",
                scale=float(selected["scale"]),
                critical=confirmation_critical,
                maxima=maxima,
                power_rows=confirmation_power_rows,
                decisions=decisions,
            ),
        )
        selected_path = base / paths["selected_blueprint"]
        selected_blueprint = _load(selected_path)
        if (
            selected_blueprint.get("task_id") != TASK_ID
            or selected_blueprint.get("schema_version")
            != "PHASE9-T06-SELECTED-BLUEPRINT-V1"
            or selected_blueprint.get("analysis_sha256")
            != _self_hash(selected_blueprint)
            or selected_blueprint.get("selected_count") != selected
            or selected_blueprint.get("gate_count") != 3043
            or selected_blueprint.get("stochastic_gate_count") != 3037
            or selected_blueprint.get("gates") != selected_gates
        ):
            raise ValueError("selected linked blueprint drift")
        if bindings["selected_blueprint"] != _binding(selected_path, base):
            raise ValueError("selected blueprint live binding drift")

    if report["confirmation"]["density_row_count"] != len(
        confirmation_density_raw
    ):
        raise ValueError("confirmation density report denominator drift")
    if selected is None:
        if report["confirmation"]["maxt_critical"] is not None:
            raise ValueError("NO-GO report contains confirmation critical")
    else:
        max_summary_delta = max(
            max_summary_delta,
            _assert_close(
                report["confirmation"]["maxt_critical"],
                confirmation_critical,
                "report confirmation critical",
            ),
        )
        recorded_power = {
            (row["case"], row["family"]): row
            for row in report["confirmation"]["maxt_power_rows"]
        }
        if len(recorded_power) != len(confirmation_power_rows):
            raise ValueError("report confirmation power denominator drift")
        for expected in confirmation_power_rows:
            key = (expected["case"], expected["family"])
            actual = recorded_power.get(key)
            if actual is None or int(actual["successes"]) != expected["successes"]:
                raise ValueError(f"report confirmation power mismatch {key}")
            max_summary_delta = max(
                max_summary_delta,
                _max_numeric_delta(
                    expected,
                    actual,
                    ("effect_ratio", "point", "wilson_lcb", "wilson_ucb"),
                    f"report/confirmation-power/{key}",
                ),
            )

    recorded_free = report.get("resource", {}).get("free_disk_bytes")
    free_disk = (
        None if recorded_free is None
        else _int(recorded_free, "recorded free disk")
    )
    expected_resource, resource_pass = _resource_summary(
        config, selected, free_disk
    )
    if set(report["resource"]) != set(expected_resource):
        raise ValueError("resource summary field set drift")
    for name, value in expected_resource.items():
        actual = report["resource"][name]
        if isinstance(value, bool) or value is None:
            if actual is not value:
                raise ValueError(f"resource {name} drift")
        else:
            max_summary_delta = max(
                max_summary_delta,
                _assert_close(actual, value, f"resource/{name}"),
            )

    expected_gates = {
        "G01_t05_verified_no_go_live": True,
        "G02_t04_outcome_firewall_closed": True,
        "G03_finite_grid_complete": True,
        "G04_selection_seed_ranges_injective": True,
        "G05_all_candidates_density_evaluated": len(selection_summary) == 8,
        "G06_all_candidates_full_maxt_evaluated": (
            len(expected_selection_maxt_summaries) == 4
        ),
        "G07_smallest_joint_passing_candidate_selected": selected is not None,
        "G08_confirmation_seed_ranges_disjoint": selected is not None,
        "G09_full_24_cell_density_confirmation": (
            selected is not None
            and len(confirmation_summary) == 24
            and len(confirmation_density_raw) == 24 * 256
        ),
        "G10_density_confirmation_all_pass": confirmation_density_pass,
        "G11_full_3043_gate_maxt_confirmation": (
            selected_blueprint is not None
            and selected_blueprint["gate_count"] == 3043
            and selected_blueprint["stochastic_gate_count"] == 3037
        ),
        "G12_maxt_confirmation_all_pass": confirmation_maxt_pass,
        "G13_resource_forecast_pass": resource_pass,
        "G14_t04_fresh_resource_gate_retained": (
            config["resource"]["t04_fresh_layer_benchmark_required"] is True
        ),
        "G15_claims_null": True,
    }
    if report.get("gates") != expected_gates:
        raise ValueError("writer report gate ledger differs from independent recompute")
    writer_pass = all(expected_gates.values())
    expected_writer_verdict = WRITER_PASS if writer_pass else WRITER_NO_GO
    if (
        report.get("verdict") != expected_writer_verdict
        or report.get("t04_preregistration_released") is not writer_pass
        or report.get("t04_scientific_execution_released") is not False
        or report["claim_boundary"].get("t04_preregistration_released")
        is not writer_pass
        or report.get("gate_summary")
        != {
            "passed": sum(expected_gates.values()),
            "total": len(expected_gates),
        }
    ):
        raise ValueError("writer verdict/release is inconsistent")

    verification_gates = {
        "V01_config_outcome_firewall_and_claims": True,
        "V02_all_report_and_parent_bindings_live": True,
        "V03_t05_verified_no_go_lineage_preserved": True,
        "V04_t05_blueprint_exact_3043_3037_6": True,
        "V05_all_four_linked_blueprints_mapped": True,
        "V06_selection_density_all_1024_rows_recomputed": (
            len(selection_density_raw) == 8 * 128
        ),
        "V07_density_complex_psd_trace_norm_b199_higher": True,
        "V08_selection_density_summaries_recomputed": len(selection_summary) == 8,
        "V09_all_four_selection_maxt_199_replicates_recomputed": True,
        "V10_all_four_selection_power_ledgers_recomputed": True,
        "V11_smallest_passing_candidate_recomputed": (
            report["selection"]["selected"] == selected
        ),
        "V12_confirmation_full_24_cells_or_verified_no_go": (
            len(confirmation_summary) == 24 if selected is not None else True
        ),
        "V13_confirmation_maxt_and_power_or_verified_no_go": (
            confirmation_critical is not None if selected is not None else True
        ),
        "V14_selection_confirmation_seed_namespaces_disjoint": True,
        "V15_factor_B_quantile_and_joint_correlation_exact": True,
        "V16_no_gate_deletion_posthoc_scale_or_pointwise_z": True,
        "V17_resource_algebra_and_fresh_gate_recomputed": True,
        "V18_source_and_summary_values_recomputed": True,
        "V19_verifier_stdlib_numpy_only": _validate_import_independence(),
        "V20_claims_null_and_scientific_execution_blocked": True,
        "V21_release_matches_independent_pass_only": True,
    }
    valid = all(verification_gates.values())
    verification = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "verifier_imports_writer": False,
        "verifier_imports_physics": False,
        "verifier_imports_prior_verifier": False,
        "verifier_imports_paired_uq": False,
        "bindings": {
            "config": _binding(config_path, base),
            "writer_report": _binding(report_path, base),
            "source_data": _binding(source_path, base),
            "t05_blueprint": _binding(
                base / config["parent_artifacts"]["t05_blueprint"], base
            ),
            "selected_blueprint": (
                _binding(base / paths["selected_blueprint"], base)
                if selected is not None else None
            ),
            "verifier_source": _binding(Path(__file__), base),
        },
        "selected_count": selected,
        "selection_density_rows_recomputed": len(selection_density_raw),
        "confirmation_density_rows_recomputed": len(confirmation_density_raw),
        "linked_blueprints_rebuilt": 4,
        "selection_maxt_replicates_recomputed": 4 * 199,
        "selection_maxt_power_cases_recomputed": 4 * 40,
        "confirmation_maxt_replicates_recomputed": (
            199 if selected is not None else 0
        ),
        "confirmation_maxt_power_cases_recomputed": (
            40 if selected is not None else 0
        ),
        "maximum_density_raw_delta": max_density_delta,
        "maximum_maxt_raw_delta": max_maxt_delta,
        "maximum_summary_resource_delta": max_summary_delta,
        "maximum_numeric_delta": max(
            max_density_delta, max_maxt_delta, max_summary_delta
        ),
        "gates": verification_gates,
        "gate_summary": {
            "passed": sum(verification_gates.values()),
            "total": len(verification_gates),
        },
        "t04_preregistration_released": bool(valid and writer_pass),
        "t04_scientific_execution_released": False,
        "qualified_claim": None,
        "claim_boundary": {
            "twin_qualification": None,
            "ler": None,
            "lifetime": None,
            "physical_break_even": None,
            "official_puviani_exact": None,
            "puviani_nmf_surpass": None,
            "external_sota": None,
            "hardware_measured": None,
        },
        "verdict": (
            PASS if valid and writer_pass
            else VERIFIED_NO_GO if valid
            else FAIL
        ),
        "source_header_sha256": _sha(list(header)),
    }
    verification["analysis_sha256"] = _self_hash(verification)
    if write:
        _atomic(base / paths["verification"], verification)
    return verification


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args(argv)
    result = verify(workers=arguments.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] in (PASS, VERIFIED_NO_GO) else 2


if __name__ == "__main__":
    raise SystemExit(main())
