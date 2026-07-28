"""Fresh three-split calibration of the scalar/vector-norm UCB factor.

The failed factor=1.0 run is an immutable diagnostic only.  This runner
creates two independent selection folds, seals the smallest factor that
passes both folds, and only then generates an untouched confirmation split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from statistics import NormalDist
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from cnn_fpga.benchmark.phase9_paired_cluster_uq import paired_vector_norm_ucb


TASK_ID = "T-RISK-20260728-02"
CONFIG_PATH = "configs/phase9/t_risk_20260728_02_scalar_uq_calibration.json"
CONFIG_SCHEMA = "PHASE9-SCALAR-UQ-THREE-SPLIT-CALIBRATION-CONFIG-V2"
REPORT_SCHEMA = "PHASE9-SCALAR-UQ-THREE-SPLIT-CALIBRATION-REPORT-V2"
RUN_IDENTITY_SCHEMA = "PHASE9-SCALAR-UQ-THREE-SPLIT-RUN-IDENTITY-V2"
SELECTION_RECEIPT_SCHEMA = "PHASE9-SCALAR-UQ-SELECTION-RECEIPT-V2"
PASS_VERDICT = "PASS_SCALAR_UQ_THREE_SPLIT_CALIBRATION"
NO_GO_SELECTION = "NO_GO_SCALAR_UQ_FACTOR_SELECTION"
NO_GO_CONFIRMATION = "NO_GO_SCALAR_UQ_UNTOUCHED_CONFIRMATION"
WORKER_BLAS_THREADS = 1
FACTOR_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
MARGINS = [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.25]
CLUSTER_COUNTS = [12, 384]
EFFECT_RATIOS = [0.0, 0.5, 1.0, 1.2]
FAMILIES = {
    "gaussian_balanced": {
        "left_scale_ratio": 0.25,
        "right_scale_ratio": 0.25,
        "rare_probability": 0.0,
        "rare_scale_ratio": 0.0,
        "power_primary": True,
    },
    "rare_heavy_tail": {
        "left_scale_ratio": 0.2,
        "right_scale_ratio": 0.2,
        "rare_probability": 0.12,
        "rare_scale_ratio": 1.25,
        "power_primary": True,
    },
    "heteroskedastic": {
        "left_scale_ratio": 0.15,
        "right_scale_ratio": 0.45,
        "rare_probability": 0.0,
        "rare_scale_ratio": 0.0,
        "power_primary": False,
    },
}
GATES = {
    "minimum_cell_coverage_rate": 0.94,
    "minimum_cell_coverage_wilson_lcb": 0.9,
    "null_equivalence_wilson_lcb": 0.8,
    "local_half_margin_equivalence_wilson_lcb": 0.65,
    "boundary_equivalence_wilson_ucb": 0.1,
    "outside_equivalence_wilson_ucb": 0.05,
}
CLAIM_BOUNDARY = {
    "scalar_uq_calibration_only": True,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}
SPLIT_NAMES = ("selection_a", "selection_b", "confirmation")
RAW_FIELDS = [
    "split",
    "cell_id",
    "family",
    "margin",
    "cluster_count",
    "effect_ratio",
    "true_difference",
    "trial_index",
    "trial_seed",
    "multiplier_seed",
    "estimate",
    "raw_radius",
]


@dataclass(frozen=True)
class Cell:
    family: str
    margin: float
    cluster_count: int
    effect_ratio: float

    @property
    def cell_id(self) -> str:
        return (
            f"{self.family}__m{self.margin:.6f}__n{self.cluster_count}"
            f"__r{self.effect_ratio:.3f}"
        ).replace(".", "p")


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
    return sha256(_canonical(value)).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _verify_binding(root: Path, binding: Mapping[str, Any], label: str) -> None:
    if set(binding) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label} binding schema drift")
    path = (root / str(binding["path"])).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError(f"{label} binding path invalid")
    if _binding(path, root) != dict(binding):
        raise RuntimeError(f"{label} live binding drift")


def _with_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["analysis_sha256"] = _sha(result)
    return result


def _verify_self_hash(payload: Mapping[str, Any], label: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop("analysis_sha256", None)
    if not isinstance(observed, str) or observed != _sha(unsigned):
        raise RuntimeError(f"{label} self-hash drift")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _cells(config: Mapping[str, Any]) -> list[Cell]:
    cells = [
        Cell(family, margin, count, ratio)
        for family in config["families"]
        for margin in config["margins"]
        for count in config["cluster_counts"]
        for ratio in config["effect_ratios"]
    ]
    if len(cells) != 192 or len({cell.cell_id for cell in cells}) != 192:
        raise RuntimeError("three-split scalar UQ cell denominator drift")
    return cells


def _split_seed(
    base: int, cell_index: int, trial_index: int, trial_count: int
) -> int:
    if (
        base < 0
        or not 0 <= cell_index < 192
        or not 0 <= trial_index < trial_count
        or trial_count < 1
    ):
        raise ValueError("three-split scalar UQ seed address outside frozen domain")
    return int(base) + cell_index * trial_count + trial_index


def _resource_seed(base: int, family_index: int, count_index: int) -> int:
    if base < 0 or not 0 <= family_index < 3 or not 0 <= count_index < 2:
        raise ValueError("three-split scalar UQ resource seed outside frozen domain")
    return int(base) + family_index * 2 + count_index


def _validate_parent_diagnostic(root: Path, config: Mapping[str, Any]) -> None:
    parent = config.get("diagnostic_only_parent")
    if (
        not isinstance(parent, Mapping)
        or parent.get("verdict") != "INCOMPLETE_CUTOFF32_36_SCALAR_UQ_PREFLIGHT"
        or parent.get(
            "may_inform_repair_design_but_not_enter_selection_or_confirmation_evidence"
        )
        is not True
    ):
        raise ValueError("diagnostic-only parent contract drift")
    names = {
        "report",
        "source_data",
    }
    if not names.issubset(parent):
        raise ValueError("diagnostic-only parent bindings missing")
    for name in names:
        binding = parent[name]
        if not isinstance(binding, Mapping):
            raise ValueError(f"diagnostic-only parent binding invalid: {name}")
        _verify_binding(root, binding, f"diagnostic parent/{name}")
    report = _read_json(root / str(parent["report"]["path"]))
    if (
        report.get("verdict") != parent["verdict"]
        or report.get("raw_trial_count") != 55296
        or report.get("coverage_all_passed") is not False
        or report.get("power_all_passed") is not False
        or report.get("design_outcomes_accessed") is not False
        or report.get("qualified_claim") is not None
        or report.get("claim_state", {}).get("scalar_uq_preflight_only") is not True
        or any(
            value is not None
            for key, value in report.get("claim_state", {}).items()
            if key != "scalar_uq_preflight_only"
        )
    ):
        raise ValueError("diagnostic-only parent semantic drift")
    source_binding = report.get("bindings", {}).get("source_data")
    if source_binding != parent["source_data"]:
        raise ValueError("diagnostic-only parent source binding drift")
    source_path = root / str(parent["source_data"]["path"])
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        if sum(1 for _ in stream) != 55297:
            raise ValueError("diagnostic-only parent raw denominator drift")


def _validate_infrastructure_failure_parent(
    root: Path, config: Mapping[str, Any]
) -> None:
    parent = config.get("infrastructure_failure_parent")
    if (
        not isinstance(parent, Mapping)
        or parent.get("terminal_state")
        != "FAILED_INFRASTRUCTURE_SEED_COLLISION_BEFORE_SELECTION_SEAL"
        or parent.get(
            "v1_outcomes_used_as_v2_selection_or_confirmation_evidence"
        )
        is not False
    ):
        raise ValueError("v1 seed-collision parent contract drift")
    names = {
        "failure_report",
        "selection_a_source_data",
        "resource_preflight",
        "run_identity",
    }
    if not names.issubset(parent):
        raise ValueError("v1 seed-collision parent bindings missing")
    for name in names:
        binding = parent[name]
        if not isinstance(binding, Mapping):
            raise ValueError(f"v1 seed-collision parent binding invalid: {name}")
        _verify_binding(root, binding, f"v1 seed-collision parent/{name}")
    failure = _read_json(root / str(parent["failure_report"]["path"]))
    if (
        failure.get("terminal_state") != parent["terminal_state"]
        or failure.get("selection_receipt_published") is not False
        or failure.get("confirmation_outcomes_accessed") is not False
        or failure.get("final_report_published") is not False
        or failure.get(
            "v1_outcomes_used_as_v2_selection_or_confirmation_evidence"
        )
        is not False
        or failure.get("claim_state") != CLAIM_BOUNDARY
        or failure.get("bindings", {}).get("selection_a_source_data")
        != parent["selection_a_source_data"]
        or failure.get("bindings", {}).get("resource_preflight")
        != parent["resource_preflight"]
        or failure.get("bindings", {}).get("run_identity")
        != parent["run_identity"]
    ):
        raise ValueError("v1 seed-collision parent semantic drift")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    expected_numeric = {
        "blas_threads_per_worker": 1,
        "threading_policy": (
            "one BLAS thread inside each ProcessPool worker; "
            "process-level parallelism only"
        ),
        "scientific_design_unchanged": True,
    }
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("margins") != MARGINS
        or config.get("cluster_counts") != CLUSTER_COUNTS
        or config.get("effect_ratios") != EFFECT_RATIOS
        or config.get("families") != FAMILIES
        or config.get("factor_grid") != FACTOR_GRID
        or config.get("confidence") != 0.95
        or config.get("multiplier_replicates") != 199
        or config.get("trial_count_per_cell") != 2048
        or config.get("max_workers") != 4
        or config.get("numeric_execution") != expected_numeric
        or config.get("gates") != GATES
        or config.get("design_outcomes_accessed") is not False
        or config.get("claim_boundary") != CLAIM_BOUNDARY
        or config.get("selection_rule")
        != (
            "choose the smallest factor that passes every coverage and "
            "primary-power gate in both selection_a and selection_b; never "
            "inspect confirmation before selection is atomically sealed"
        )
    ):
        raise ValueError("three-split scalar UQ config drift")
    splits = config.get("splits")
    expected_splits = {
        "selection_a": {
            "role": "factor_selection",
            "trial_seed_base": 2000000,
            "multiplier_seed_base": 3000000,
        },
        "selection_b": {
            "role": "factor_selection",
            "trial_seed_base": 4000000,
            "multiplier_seed_base": 5000000,
        },
        "confirmation": {
            "role": "untouched_confirmation",
            "trial_seed_base": 6000000,
            "multiplier_seed_base": 7000000,
        },
    }
    if splits != expected_splits:
        raise ValueError("three-split scalar UQ split drift")
    simultaneous = config.get("simultaneous_wilson")
    if simultaneous != {
        "confidence": 0.95,
        "comparisons_per_split": 256,
        "method": (
            "per-split Bonferroni-adjusted two-sided Wilson bounds across "
            "192 coverage and 64 primary-power strata"
        ),
    }:
        raise ValueError("three-split scalar UQ simultaneous-Wilson drift")
    resource = config.get("resource_preflight", {})
    if (
        resource.get("trial_seed_base") != 8000000
        or resource.get("multiplier_seed_base") != 9000000
        or resource.get("safety_factor") != 2.0
        or resource.get("maximum_estimated_wall_seconds") != 14400
        or resource.get("maximum_estimated_rss_bytes") != 4294967296
    ):
        raise ValueError("three-split scalar UQ resource contract drift")
    all_seed_bases = [
        value[key]
        for value in splits.values()
        for key in ("trial_seed_base", "multiplier_seed_base")
    ] + [resource["trial_seed_base"], resource["multiplier_seed_base"]]
    if len(all_seed_bases) != len(set(all_seed_bases)):
        raise ValueError("three-split scalar UQ seed firewall drift")
    addressing = config.get("seed_addressing")
    if addressing != {
        "scheme": "injective_cell_major_v2",
        "formula": "base + cell_index * trial_count_per_cell + trial_index",
        "cell_count": 192,
        "trial_count_per_cell": 2048,
        "maximum_offset": 393215,
        "all_six_scientific_ranges_pairwise_disjoint": True,
        "v1_hash_modulo_addressing_forbidden": True,
    }:
        raise ValueError("three-split scalar UQ seed-addressing drift")
    scientific_ranges = sorted(
        (
            int(value[key]),
            int(value[key]) + int(addressing["maximum_offset"]),
        )
        for value in splits.values()
        for key in ("trial_seed_base", "multiplier_seed_base")
    )
    if any(
        left[1] >= right[0]
        for left, right in zip(scientific_ranges, scientific_ranges[1:])
    ):
        raise ValueError("three-split scalar UQ scientific seed ranges overlap")
    if set(config.get("artifact_paths", {})) != {
        "run_directory",
        "owner_lock",
        "run_identity",
        "selection_receipt",
        "report",
        "selection_a_source_data",
        "selection_b_source_data",
        "confirmation_source_data",
    }:
        raise ValueError("three-split scalar UQ artifact path drift")
    _validate_parent_diagnostic(root, config)
    _validate_infrastructure_failure_parent(root, config)
    _cells(config)
    return config


def _one_trial_raw(
    split: str,
    cell: Cell,
    family: Mapping[str, Any],
    *,
    trial_index: int,
    trial_seed: int,
    multiplier_seed: int,
    confidence: float,
    replicates: int,
) -> dict[str, object]:
    rng = np.random.default_rng(trial_seed)
    count = cell.cluster_count
    margin = cell.margin
    common = 0.2 * margin * rng.standard_normal(count)
    left = common + float(family["left_scale_ratio"]) * margin * rng.standard_normal(
        count
    )
    right = (
        common
        - cell.effect_ratio * margin
        + float(family["right_scale_ratio"])
        * margin
        * rng.standard_normal(count)
    )
    rare_probability = float(family["rare_probability"])
    if rare_probability:
        rare_scale = float(family["rare_scale_ratio"]) * margin
        left += (
            rng.random(count) < rare_probability
        ) * rare_scale * rng.standard_normal(count)
        right += (
            rng.random(count) < rare_probability
        ) * rare_scale * rng.standard_normal(count)
    result = paired_vector_norm_ucb(
        left,
        right,
        ord_value=2,
        confidence=confidence,
        multiplier_replicates=replicates,
        seed=multiplier_seed,
        calibration_factor=1.0,
    )
    values = (
        float(result.estimate),
        float(result.raw_radius),
        float(cell.effect_ratio * margin),
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("three-split scalar UQ non-finite trial")
    return {
        "split": split,
        "cell_id": cell.cell_id,
        "family": cell.family,
        "margin": margin,
        "cluster_count": count,
        "effect_ratio": cell.effect_ratio,
        "true_difference": values[2],
        "trial_index": trial_index,
        "trial_seed": trial_seed,
        "multiplier_seed": multiplier_seed,
        "estimate": values[0],
        "raw_radius": values[1],
    }


def _simulate_cell_thread_limited(
    split: str, cell: Cell, config: Mapping[str, Any]
) -> list[dict[str, object]]:
    spec = config["splits"][split]
    cell_index = _cells(config).index(cell)
    rows = []
    for trial in range(int(config["trial_count_per_cell"])):
        rows.append(
            _one_trial_raw(
                split,
                cell,
                config["families"][cell.family],
                trial_index=trial,
                trial_seed=_split_seed(
                    int(spec["trial_seed_base"]),
                    cell_index,
                    trial,
                    int(config["trial_count_per_cell"]),
                ),
                multiplier_seed=_split_seed(
                    int(spec["multiplier_seed_base"]),
                    cell_index,
                    trial,
                    int(config["trial_count_per_cell"]),
                ),
                confidence=float(config["confidence"]),
                replicates=int(config["multiplier_replicates"]),
            )
        )
    return rows


def _simulate_cell(
    split: str, cell: Cell, config: Mapping[str, Any]
) -> dict[str, Any]:
    with threadpool_limits(limits=WORKER_BLAS_THREADS, user_api="blas"):
        libraries = [
            info for info in threadpool_info() if info.get("user_api") == "blas"
        ]
        if not libraries or any(
            int(info.get("num_threads", 0)) != WORKER_BLAS_THREADS
            for info in libraries
        ):
            raise RuntimeError("three-split scalar UQ worker BLAS contract drift")
        rows = _simulate_cell_thread_limited(split, cell, config)
        attestation = [
            {
                "internal_api": info.get("internal_api"),
                "prefix": info.get("prefix"),
                "num_threads": int(info.get("num_threads", 0)),
            }
            for info in libraries
        ]
    return {"rows": rows, "numeric_worker_attestation": attestation}


def _wilson(
    successes: int, total: int, config: Mapping[str, Any]
) -> tuple[float, float]:
    alpha = (1.0 - float(config["simultaneous_wilson"]["confidence"])) / int(
        config["simultaneous_wilson"]["comparisons_per_split"]
    )
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _evaluate_cell(
    cell: Cell,
    rows: Sequence[Mapping[str, Any]],
    factor: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = int(config["trial_count_per_cell"])
    if len(rows) != expected:
        raise RuntimeError("three-split scalar UQ cell trial denominator drift")
    expected_fields = set(RAW_FIELDS)
    expected_trials = set(range(expected))
    observed_trials: set[int] = set()
    observed_trial_seeds: set[int] = set()
    observed_multiplier_seeds: set[int] = set()
    for row in rows:
        if set(row) != expected_fields:
            raise RuntimeError("three-split scalar UQ raw-row schema drift")
        trial_index = int(row["trial_index"])
        observed_trials.add(trial_index)
        observed_trial_seeds.add(int(row["trial_seed"]))
        observed_multiplier_seeds.add(int(row["multiplier_seed"]))
        numeric = (
            float(row["margin"]),
            float(row["effect_ratio"]),
            float(row["true_difference"]),
            float(row["estimate"]),
            float(row["raw_radius"]),
        )
        if (
            str(row["cell_id"]) != cell.cell_id
            or str(row["family"]) != cell.family
            or float(row["margin"]) != cell.margin
            or int(row["cluster_count"]) != cell.cluster_count
            or float(row["effect_ratio"]) != cell.effect_ratio
            or float(row["true_difference"]) != cell.effect_ratio * cell.margin
            or not all(math.isfinite(value) for value in numeric)
            or float(row["estimate"]) < 0.0
            or float(row["raw_radius"]) < 0.0
        ):
            raise RuntimeError("three-split scalar UQ raw-row semantic drift")
    if (
        observed_trials != expected_trials
        or len(observed_trial_seeds) != expected
        or len(observed_multiplier_seeds) != expected
    ):
        raise RuntimeError("three-split scalar UQ raw-row seed/index drift")
    coverage = 0
    equivalence = 0
    for row in rows:
        upper = float(row["estimate"]) + factor * float(row["raw_radius"])
        truth = float(row["true_difference"])
        coverage += upper + 1e-15 >= truth
        equivalence += upper <= cell.margin
    coverage_lcb, coverage_ucb = _wilson(coverage, expected, config)
    equivalence_lcb, equivalence_ucb = _wilson(equivalence, expected, config)
    return {
        "cell_id": cell.cell_id,
        "family": cell.family,
        "margin": cell.margin,
        "cluster_count": cell.cluster_count,
        "effect_ratio": cell.effect_ratio,
        "power_primary": bool(config["families"][cell.family]["power_primary"]),
        "coverage_successes": coverage,
        "coverage_rate": coverage / expected,
        "coverage_wilson_lcb": coverage_lcb,
        "coverage_wilson_ucb": coverage_ucb,
        "equivalence_successes": equivalence,
        "equivalence_rate": equivalence / expected,
        "equivalence_wilson_lcb": equivalence_lcb,
        "equivalence_wilson_ucb": equivalence_ucb,
    }


def _factor_gate(
    factor: float,
    summaries: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(summaries) != 192:
        raise RuntimeError("three-split scalar UQ summary denominator drift")
    coverage_failed = [
        str(row["cell_id"])
        for row in summaries
        if (
            float(row["coverage_rate"]) < GATES["minimum_cell_coverage_rate"]
            or float(row["coverage_wilson_lcb"])
            < GATES["minimum_cell_coverage_wilson_lcb"]
        )
    ]
    rules = {
        0.0: ("lcb", GATES["null_equivalence_wilson_lcb"]),
        0.5: ("lcb", GATES["local_half_margin_equivalence_wilson_lcb"]),
        1.0: ("ucb", GATES["boundary_equivalence_wilson_ucb"]),
        1.2: ("ucb", GATES["outside_equivalence_wilson_ucb"]),
    }
    power_ledger = []
    for ratio, (bound, threshold) in rules.items():
        strata = [
            row
            for row in summaries
            if int(row["cluster_count"]) == 384
            and bool(row["power_primary"])
            and float(row["effect_ratio"]) == ratio
        ]
        if len(strata) != 16:
            raise RuntimeError("three-split scalar UQ power denominator drift")
        failed = [
            str(row["cell_id"])
            for row in strata
            if (
                float(row["equivalence_wilson_lcb"]) < threshold
                if bound == "lcb"
                else float(row["equivalence_wilson_ucb"]) > threshold
            )
        ]
        power_ledger.append(
            {
                "effect_ratio": ratio,
                "bound": bound,
                "threshold": threshold,
                "stratum_count": len(strata),
                "failed_strata": failed,
                "global_iut_pass": not failed,
            }
        )
    coverage_pass = not coverage_failed
    power_pass = all(row["global_iut_pass"] for row in power_ledger)
    return {
        "factor": factor,
        "coverage_pass": coverage_pass,
        "coverage_failed_cells": coverage_failed,
        "power_pass": power_pass,
        "power_ledger": power_ledger,
        "global_pass": coverage_pass and power_pass,
        "minimum_coverage_rate": min(float(row["coverage_rate"]) for row in summaries),
        "minimum_coverage_wilson_lcb": min(
            float(row["coverage_wilson_lcb"]) for row in summaries
        ),
        "summaries_sha256": _sha(list(summaries)),
    }


def _write_split(
    root: Path,
    config: Mapping[str, Any],
    split: str,
    factors: Sequence[float],
) -> dict[str, Any]:
    if split not in SPLIT_NAMES:
        raise ValueError("unknown scalar UQ split")
    if not factors or any(factor not in FACTOR_GRID for factor in factors):
        raise ValueError("scalar UQ factor evaluation outside frozen grid")
    path = root / config["artifact_paths"][f"{split}_source_data"]
    if path.exists():
        raise RuntimeError(f"{split} canonical source already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    cells = _cells(config)
    summaries: dict[float, list[dict[str, Any]]] = {
        factor: [] for factor in factors
    }
    attestations: list[list[dict[str, Any]]] = []
    raw_count = 0
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS, lineterminator="\n")
            writer.writeheader()
            with ProcessPoolExecutor(max_workers=int(config["max_workers"])) as executor:
                for start in range(0, len(cells), int(config["max_workers"])):
                    batch = cells[start : start + int(config["max_workers"])]
                    futures = [
                        executor.submit(_simulate_cell, split, cell, config)
                        for cell in batch
                    ]
                    for cell, future in zip(batch, futures, strict=True):
                        payload = future.result()
                        rows = payload["rows"]
                        if len(rows) != int(config["trial_count_per_cell"]):
                            raise RuntimeError("scalar UQ worker denominator drift")
                        writer.writerows(rows)
                        raw_count += len(rows)
                        attestations.append(payload["numeric_worker_attestation"])
                        for factor in factors:
                            summaries[factor].append(
                                _evaluate_cell(cell, rows, factor, config)
                            )
            stream.flush()
            os.fsync(stream.fileno())
        expected = len(cells) * int(config["trial_count_per_cell"])
        if raw_count != expected:
            raise RuntimeError("three-split scalar UQ raw denominator drift")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    if len(attestations) != 192 or any(
        not attestation
        or any(int(item["num_threads"]) != 1 for item in attestation)
        for attestation in attestations
    ):
        raise RuntimeError("three-split scalar UQ numeric attestation drift")
    gates = {
        f"{factor:.1f}": _factor_gate(factor, summaries[factor], config)
        for factor in factors
    }
    return {
        "split": split,
        "role": config["splits"][split]["role"],
        "raw_trial_count": raw_count,
        "cell_count": len(cells),
        "trial_count_per_cell": int(config["trial_count_per_cell"]),
        "evaluated_factors": list(factors),
        "factor_gates": gates,
        "source_data_binding": _binding(path, root),
        "numeric_worker_attestation_sha256": _sha(attestations),
        "all_workers_single_blas_thread": True,
    }


def _select_factor(
    selection_a: Mapping[str, Any],
    selection_b: Mapping[str, Any],
) -> float | None:
    if (
        selection_a.get("split") != "selection_a"
        or selection_b.get("split") != "selection_b"
        or selection_a.get("evaluated_factors") != FACTOR_GRID
        or selection_b.get("evaluated_factors") != FACTOR_GRID
    ):
        raise RuntimeError("scalar UQ selection split contract drift")
    for factor in FACTOR_GRID:
        key = f"{factor:.1f}"
        if (
            selection_a["factor_gates"][key]["global_pass"] is True
            and selection_b["factor_gates"][key]["global_pass"] is True
        ):
            return factor
    return None


def _resource_preflight(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["resource_preflight"]
    process = psutil.Process(os.getpid())
    baseline = int(process.memory_info().rss)
    timings = []
    with threadpool_limits(limits=WORKER_BLAS_THREADS, user_api="blas"):
        for family_index, family in enumerate(config["families"]):
            for count_index, count in enumerate(config["cluster_counts"]):
                cell = Cell(family, 0.1, count, 0.5)
                started = time.perf_counter()
                _one_trial_raw(
                    "resource_preflight",
                    cell,
                    config["families"][family],
                    trial_index=0,
                    trial_seed=_resource_seed(
                        contract["trial_seed_base"], family_index, count_index
                    ),
                    multiplier_seed=_resource_seed(
                        contract["multiplier_seed_base"],
                        family_index,
                        count_index,
                    ),
                    confidence=config["confidence"],
                    replicates=config["multiplier_replicates"],
                )
                timings.append(
                    {
                        "family": family,
                        "cluster_count": count,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
    estimated_wall = (
        sum(item["elapsed_seconds"] for item in timings)
        * len(config["margins"])
        * len(config["effect_ratios"])
        * int(config["trial_count_per_cell"])
        * len(SPLIT_NAMES)
        / int(config["max_workers"])
        * float(contract["safety_factor"])
    )
    observed = int(process.memory_info().rss)
    estimated_rss = baseline + max(observed - baseline, 67108864) * int(
        config["max_workers"]
    )
    report = _with_self_hash(
        {
            "task_id": TASK_ID,
            "schema_version": "PHASE9-SCALAR-UQ-THREE-SPLIT-RESOURCE-V1",
            "timings": timings,
            "estimated_wall_seconds": estimated_wall,
            "estimated_rss_bytes": estimated_rss,
            "wall_limit_seconds": contract["maximum_estimated_wall_seconds"],
            "rss_limit_bytes": contract["maximum_estimated_rss_bytes"],
            "blas_threads_per_worker": WORKER_BLAS_THREADS,
            "split_count": len(SPLIT_NAMES),
            "design_outcomes_accessed": False,
            "passed": (
                estimated_wall <= contract["maximum_estimated_wall_seconds"]
                and estimated_rss <= contract["maximum_estimated_rss_bytes"]
            ),
        }
    )
    _atomic_write(
        root / contract["artifact"],
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if not report["passed"]:
        raise RuntimeError("three-split scalar UQ resource preflight failed")
    return report


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if len(head) != 40:
        raise RuntimeError("scalar UQ release commit invalid")
    return head


def _assert_release_frozen(root: Path) -> str:
    paths = [
        CONFIG_PATH,
        Path(__file__).resolve().relative_to(root).as_posix(),
        "cnn_fpga/benchmark/phase9_paired_cluster_uq.py",
    ]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("scalar UQ release inputs are not committed")
    return _git_head(root)


def _run_identity(
    root: Path, config: Mapping[str, Any], release_commit: str
) -> dict[str, Any]:
    paths = config["artifact_paths"]
    identity_path = root / paths["run_identity"]
    if identity_path.exists():
        raise RuntimeError("scalar UQ run identity already exists")
    payload = _with_self_hash(
        {
            "task_id": TASK_ID,
            "schema_version": RUN_IDENTITY_SCHEMA,
            "run_id": str(uuid.uuid4()),
            "release_commit": release_commit,
            "created_utc": time.time(),
            "factor_grid": list(config["factor_grid"]),
            "splits": json.loads(json.dumps(config["splits"])),
            "seed_addressing": dict(config["seed_addressing"]),
            "trial_count_per_cell": config["trial_count_per_cell"],
            "design_outcomes_accessed": False,
            "claim_state": dict(CLAIM_BOUNDARY),
            "input_bindings": {
                "config": _binding(root / CONFIG_PATH, root),
                "implementation": _binding(Path(__file__).resolve(), root),
                "paired_uq": _binding(
                    root / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py", root
                ),
            },
        }
    )
    _atomic_write(
        identity_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def _selection_receipt(
    root: Path,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    selection_a: Mapping[str, Any],
    selection_b: Mapping[str, Any],
    selected_factor: float | None,
) -> dict[str, Any]:
    path = root / config["artifact_paths"]["selection_receipt"]
    payload = _with_self_hash(
        {
            "task_id": TASK_ID,
            "schema_version": SELECTION_RECEIPT_SCHEMA,
            "run_id": identity["run_id"],
            "run_identity_analysis_sha256": identity["analysis_sha256"],
            "selection_rule": config["selection_rule"],
            "factor_grid": list(config["factor_grid"]),
            "selected_factor": selected_factor,
            "selection_passed": selected_factor is not None,
            "confirmation_outcomes_accessed": False,
            "selection_a": selection_a,
            "selection_b": selection_b,
            "claim_state": dict(CLAIM_BOUNDARY),
        }
    )
    _atomic_write(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    live = _read_json(path)
    _verify_self_hash(live, "scalar UQ selection receipt")
    if live != payload or live.get("confirmation_outcomes_accessed") is not False:
        raise RuntimeError("scalar UQ selection receipt publication drift")
    _verify_binding(root, selection_a["source_data_binding"], "selection_a")
    _verify_binding(root, selection_b["source_data_binding"], "selection_b")
    return payload


@contextmanager
def _owner_lock(root: Path, config: Mapping[str, Any]):
    path = root / config["artifact_paths"]["owner_lock"]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    closed = False
    try:
        payload = {
            "task_id": TASK_ID,
            "pid": os.getpid(),
            "created_utc": time.time(),
        }
        os.write(descriptor, _canonical(payload))
        os.close(descriptor)
        closed = True
        yield
    finally:
        if not closed:
            try:
                os.close(descriptor)
            except OSError:
                pass
        path.unlink(missing_ok=True)


def _build_final_report(
    root: Path,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
    resource: Mapping[str, Any],
    receipt: Mapping[str, Any],
    confirmation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected = receipt["selected_factor"]
    if selected is None:
        verdict = NO_GO_SELECTION
        confirmation_passed = None
    else:
        if confirmation is None:
            raise RuntimeError("selected scalar UQ factor lacks confirmation")
        if confirmation.get("evaluated_factors") != [selected]:
            raise RuntimeError("confirmation attempted factor reselection")
        key = f"{selected:.1f}"
        confirmation_passed = confirmation["factor_gates"][key]["global_pass"]
        verdict = PASS_VERDICT if confirmation_passed else NO_GO_CONFIRMATION
    bindings: dict[str, Any] = {
        "config": _binding(root / CONFIG_PATH, root),
        "implementation": _binding(Path(__file__).resolve(), root),
        "paired_uq": _binding(
            root / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py", root
        ),
        "run_identity": _binding(
            root / config["artifact_paths"]["run_identity"], root
        ),
        "selection_receipt": _binding(
            root / config["artifact_paths"]["selection_receipt"], root
        ),
        "resource_preflight": _binding(
            root / config["resource_preflight"]["artifact"], root
        ),
        "selection_a_source_data": receipt["selection_a"]["source_data_binding"],
        "selection_b_source_data": receipt["selection_b"]["source_data_binding"],
    }
    for name, binding in config["diagnostic_only_parent"].items():
        if isinstance(binding, Mapping) and set(binding) == {"path", "bytes", "sha256"}:
            bindings[f"diagnostic_parent_{name}"] = dict(binding)
    for name, binding in config["infrastructure_failure_parent"].items():
        if isinstance(binding, Mapping) and set(binding) == {
            "path",
            "bytes",
            "sha256",
        }:
            bindings[f"infrastructure_failure_parent_{name}"] = dict(binding)
    if confirmation is not None:
        bindings["confirmation_source_data"] = confirmation["source_data_binding"]
    report = _with_self_hash(
        {
            "task_id": TASK_ID,
            "schema_version": REPORT_SCHEMA,
            "verdict": verdict,
            "qualified_claim": (
                "COVERAGE_CALIBRATED_SCALAR_UQ_FACTOR_FOR_T_RISK_20260728_01"
                if verdict == PASS_VERDICT
                else None
            ),
            "run_id": identity["run_id"],
            "release_commit": identity["release_commit"],
            "run_identity_analysis_sha256": identity["analysis_sha256"],
            "resource_preflight_analysis_sha256": resource["analysis_sha256"],
            "selection_receipt_analysis_sha256": receipt["analysis_sha256"],
            "factor_grid": list(config["factor_grid"]),
            "selected_factor": selected,
            "selection_passed": selected is not None,
            "confirmation_passed": confirmation_passed,
            "confirmation_evaluated_factor_count": (
                0 if confirmation is None else len(confirmation["evaluated_factors"])
            ),
            "trial_count_per_cell_per_split": config["trial_count_per_cell"],
            "expected_cells_per_split": len(_cells(config)),
            "expected_raw_trials_per_split": (
                len(_cells(config)) * int(config["trial_count_per_cell"])
            ),
            "selection_a": receipt["selection_a"],
            "selection_b": receipt["selection_b"],
            "confirmation": confirmation,
            "diagnostic_parent_used_as_selection_or_confirmation_evidence": False,
            "repair_contract_frozen_after_diagnostic": True,
            "v1_failure_used_as_v2_selection_or_confirmation_evidence": False,
            "seed_addressing": dict(config["seed_addressing"]),
            "design_outcomes_accessed": False,
            "claim_state": dict(CLAIM_BOUNDARY),
            "bindings": bindings,
        }
    )
    return report


def _verify_finalized_report(
    root: Path, config: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    _verify_self_hash(report, "scalar UQ finalized report")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("verdict")
        not in {PASS_VERDICT, NO_GO_SELECTION, NO_GO_CONFIRMATION}
        or report.get("design_outcomes_accessed") is not False
        or report.get("diagnostic_parent_used_as_selection_or_confirmation_evidence")
        is not False
        or report.get("repair_contract_frozen_after_diagnostic") is not True
        or report.get(
            "v1_failure_used_as_v2_selection_or_confirmation_evidence"
        )
        is not False
        or report.get("seed_addressing") != config["seed_addressing"]
        or report.get("claim_state") != CLAIM_BOUNDARY
        or report.get("expected_cells_per_split") != 192
        or report.get("trial_count_per_cell_per_split") != 2048
        or report.get("expected_raw_trials_per_split") != 393216
    ):
        raise RuntimeError("scalar UQ finalized semantic drift")
    passed = report["verdict"] == PASS_VERDICT
    if passed:
        if (
            report.get("selection_passed") is not True
            or report.get("confirmation_passed") is not True
            or report.get("selected_factor") not in FACTOR_GRID
            or report.get("confirmation_evaluated_factor_count") != 1
            or report.get("qualified_claim")
            != "COVERAGE_CALIBRATED_SCALAR_UQ_FACTOR_FOR_T_RISK_20260728_01"
        ):
            raise RuntimeError("scalar UQ PASS semantics drift")
    elif report.get("qualified_claim") is not None:
        raise RuntimeError("scalar UQ NO-GO leaked qualified claim")
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("scalar UQ finalized bindings missing")
    expected_binding_names = {
        "config",
        "implementation",
        "paired_uq",
        "run_identity",
        "selection_receipt",
        "resource_preflight",
        "selection_a_source_data",
        "selection_b_source_data",
        "diagnostic_parent_report",
        "diagnostic_parent_source_data",
        "infrastructure_failure_parent_failure_report",
        "infrastructure_failure_parent_selection_a_source_data",
        "infrastructure_failure_parent_resource_preflight",
        "infrastructure_failure_parent_run_identity",
    }
    if report.get("confirmation") is not None:
        expected_binding_names.add("confirmation_source_data")
    if set(bindings) != expected_binding_names:
        raise RuntimeError("scalar UQ finalized binding-name drift")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"scalar UQ finalized binding type drift: {name}")
        _verify_binding(root, binding, f"final report/{name}")
    expected_direct_bindings = {
        "config": _binding(root / CONFIG_PATH, root),
        "implementation": _binding(Path(__file__).resolve(), root),
        "paired_uq": _binding(
            root / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py", root
        ),
        "run_identity": _binding(
            root / config["artifact_paths"]["run_identity"], root
        ),
        "selection_receipt": _binding(
            root / config["artifact_paths"]["selection_receipt"], root
        ),
        "resource_preflight": _binding(
            root / config["resource_preflight"]["artifact"], root
        ),
        "selection_a_source_data": _binding(
            root / config["artifact_paths"]["selection_a_source_data"], root
        ),
        "selection_b_source_data": _binding(
            root / config["artifact_paths"]["selection_b_source_data"], root
        ),
    }
    if report.get("confirmation") is not None:
        expected_direct_bindings["confirmation_source_data"] = _binding(
            root / config["artifact_paths"]["confirmation_source_data"], root
        )
    for name, expected in expected_direct_bindings.items():
        if bindings.get(name) != expected:
            raise RuntimeError(f"scalar UQ finalized expected binding drift: {name}")
    for name, parent_binding in config["diagnostic_only_parent"].items():
        if isinstance(parent_binding, Mapping) and set(parent_binding) == {
            "path",
            "bytes",
            "sha256",
        }:
            if bindings.get(f"diagnostic_parent_{name}") != parent_binding:
                raise RuntimeError(
                    f"scalar UQ diagnostic parent binding drift: {name}"
                )
    for name, parent_binding in config["infrastructure_failure_parent"].items():
        if isinstance(parent_binding, Mapping) and set(parent_binding) == {
            "path",
            "bytes",
            "sha256",
        }:
            if bindings.get(f"infrastructure_failure_parent_{name}") != parent_binding:
                raise RuntimeError(
                    f"scalar UQ infrastructure failure binding drift: {name}"
                )
    identity = _read_json(root / config["artifact_paths"]["run_identity"])
    _verify_self_hash(identity, "scalar UQ run identity")
    if (
        identity.get("schema_version") != RUN_IDENTITY_SCHEMA
        or identity.get("run_id") != report.get("run_id")
        or identity.get("analysis_sha256")
        != report.get("run_identity_analysis_sha256")
        or identity.get("release_commit") != report.get("release_commit")
        or identity.get("design_outcomes_accessed") is not False
        or identity.get("claim_state") != CLAIM_BOUNDARY
        or identity.get("seed_addressing") != config["seed_addressing"]
        or identity.get("input_bindings")
        != {
            "config": expected_direct_bindings["config"],
            "implementation": expected_direct_bindings["implementation"],
            "paired_uq": expected_direct_bindings["paired_uq"],
        }
    ):
        raise RuntimeError("scalar UQ run identity semantic drift")
    receipt = _read_json(root / config["artifact_paths"]["selection_receipt"])
    _verify_self_hash(receipt, "scalar UQ finalized selection receipt")
    if (
        receipt.get("schema_version") != SELECTION_RECEIPT_SCHEMA
        or receipt.get("run_id") != report.get("run_id")
        or receipt.get("analysis_sha256")
        != report.get("selection_receipt_analysis_sha256")
        or receipt.get("selected_factor") != report.get("selected_factor")
        or receipt.get("confirmation_outcomes_accessed") is not False
        or receipt.get("selection_a") != report.get("selection_a")
        or receipt.get("selection_b") != report.get("selection_b")
        or receipt.get("claim_state") != CLAIM_BOUNDARY
    ):
        raise RuntimeError("scalar UQ selection receipt semantic drift")
    resource = _read_json(root / config["resource_preflight"]["artifact"])
    _verify_self_hash(resource, "scalar UQ finalized resource report")
    if (
        resource.get("passed") is not True
        or resource.get("analysis_sha256")
        != report.get("resource_preflight_analysis_sha256")
        or resource.get("design_outcomes_accessed") is not False
        or resource.get("blas_threads_per_worker") != 1
    ):
        raise RuntimeError("scalar UQ resource report semantic drift")
    expected_lines = int(report["expected_raw_trials_per_split"]) + 1
    source_names = ["selection_a_source_data", "selection_b_source_data"]
    if report.get("confirmation") is not None:
        source_names.append("confirmation_source_data")
    for name in source_names:
        path = root / bindings[name]["path"]
        with path.open("r", encoding="utf-8", newline="") as stream:
            if sum(1 for _ in stream) != expected_lines:
                raise RuntimeError(f"scalar UQ source denominator drift: {name}")


def run(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    config = load_config(base)
    paths = config["artifact_paths"]
    must_not_exist = [
        base / paths["run_directory"],
        base / paths["report"],
        base / paths["selection_a_source_data"],
        base / paths["selection_b_source_data"],
        base / paths["confirmation_source_data"],
    ]
    if any(path.exists() for path in must_not_exist):
        raise RuntimeError("three-split scalar UQ canonical output already exists")
    release_commit = _assert_release_frozen(base)
    with _owner_lock(base, config):
        resource = _resource_preflight(base, config)
        identity = _run_identity(base, config, release_commit)
        selection_a = _write_split(base, config, "selection_a", FACTOR_GRID)
        selection_b = _write_split(base, config, "selection_b", FACTOR_GRID)
        selected = _select_factor(selection_a, selection_b)
        receipt = _selection_receipt(
            base, config, identity, selection_a, selection_b, selected
        )
        confirmation = None
        if selected is not None:
            live_receipt = _read_json(base / paths["selection_receipt"])
            _verify_self_hash(live_receipt, "pre-confirmation selection receipt")
            if (
                live_receipt["analysis_sha256"] != receipt["analysis_sha256"]
                or live_receipt["selected_factor"] != selected
                or live_receipt["confirmation_outcomes_accessed"] is not False
            ):
                raise RuntimeError("scalar UQ selection seal changed before confirmation")
            confirmation = _write_split(
                base, config, "confirmation", [selected]
            )
        report = _build_final_report(
            base, config, identity, resource, receipt, confirmation
        )
        report_path = base / paths["report"]
        _atomic_write(
            report_path,
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        live = _read_json(report_path)
        if live != report:
            raise RuntimeError("scalar UQ finalized report publication drift")
        _verify_finalized_report(base, config, live)
        return live


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = run()
    print(
        json.dumps(
            {
                "analysis_sha256": report["analysis_sha256"],
                "verdict": report["verdict"],
                "selected_factor": report["selected_factor"],
                "confirmation_passed": report["confirmation_passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Cell",
    "FACTOR_GRID",
    "NO_GO_CONFIRMATION",
    "NO_GO_SELECTION",
    "PASS_VERDICT",
    "load_config",
    "run",
]
