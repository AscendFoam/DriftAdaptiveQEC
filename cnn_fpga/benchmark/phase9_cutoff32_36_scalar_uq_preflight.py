"""Outcome-free scalar/tail UCB coverage and power preflight."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import time
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from cnn_fpga.benchmark.phase9_paired_cluster_uq import paired_vector_norm_ucb


TASK_ID = "T-RISK-20260728-01"
CONFIG_PATH = "configs/phase9/t_risk_20260728_01_scalar_uq_preflight.json"
CONFIG_SCHEMA = "PHASE9-CUTOFF32-36-SCALAR-UQ-PREFLIGHT-CONFIG-V1"
REPORT_SCHEMA = "PHASE9-CUTOFF32-36-SCALAR-UQ-PREFLIGHT-V1"
PASS_VERDICT = "PASS_CUTOFF32_36_SCALAR_UQ_PREFLIGHT"
NO_GO_VERDICT = "INCOMPLETE_CUTOFF32_36_SCALAR_UQ_PREFLIGHT"
WORKER_BLAS_THREADS = 1
CLAIM_BOUNDARY = {
    "scalar_uq_preflight_only": True,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}
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
    ).encode()


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


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


def load_config(root: Path) -> dict[str, Any]:
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("margins")
        != [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.25]
        or config.get("cluster_counts") != [12, 384]
        or config.get("effect_ratios") != [0.0, 0.5, 1.0, 1.2]
        or config.get("families") != FAMILIES
        or config.get("frozen_calibration_factor") != 1.0
        or config.get("confidence") != 0.95
        or config.get("multiplier_replicates") != 199
        or config.get("trial_count_per_cell") != 288
        or config.get("trial_seed_base") != 1560000
        or config.get("multiplier_seed_base") != 1570000
        or config.get("max_workers") != 4
        or config.get("numeric_execution")
        != {
            "blas_threads_per_worker": 1,
            "threading_policy": (
                "one BLAS thread inside each ProcessPool worker; "
                "process-level parallelism only"
            ),
            "scientific_design_unchanged": True,
        }
        or config.get("gates") != GATES
        or config.get("design_outcomes_accessed") is not False
        or config.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("scalar UQ preflight config drift")
    simultaneous = config.get("simultaneous_wilson", {})
    if (
        simultaneous.get("confidence") != 0.95
        or simultaneous.get("comparisons") != 256
        or simultaneous.get("method")
        != (
            "single-family Bonferroni-adjusted two-sided Wilson bounds "
            "across 192 coverage and 64 primary-power strata"
        )
    ):
        raise ValueError("scalar UQ simultaneous-Wilson drift")
    resource = config.get("resource_preflight", {})
    if (
        resource.get("trial_seed_base") != 1580000
        or resource.get("multiplier_seed_base") != 1590000
        or resource.get("safety_factor") != 2.0
        or resource.get("maximum_estimated_wall_seconds") != 14400
        or resource.get("maximum_estimated_rss_bytes") != 4294967296
        or len(
            {
                config["trial_seed_base"],
                config["multiplier_seed_base"],
                resource["trial_seed_base"],
                resource["multiplier_seed_base"],
            }
        )
        != 4
    ):
        raise ValueError("scalar UQ resource/seed firewall drift")
    if set(config.get("artifact_paths", {})) != {
        "run_directory",
        "owner_lock",
        "report",
        "source_data",
    }:
        raise ValueError("scalar UQ artifact-path drift")
    return config


def _cells(config: Mapping[str, Any]) -> list[Cell]:
    cells = [
        Cell(family, margin, count, ratio)
        for family in config["families"]
        for margin in config["margins"]
        for count in config["cluster_counts"]
        for ratio in config["effect_ratios"]
    ]
    if len(cells) != 192 or len({cell.cell_id for cell in cells}) != 192:
        raise RuntimeError("scalar UQ cell denominator drift")
    return cells


def _address(base: int, *parts: object) -> int:
    digest = sha256("|".join(map(str, parts)).encode()).digest()
    return int(base) + int.from_bytes(digest[:8], "big") % 1_000_000_000


def _one_trial(
    cell: Cell,
    family: Mapping[str, Any],
    *,
    trial_seed: int,
    multiplier_seed: int,
    confidence: float,
    replicates: int,
    factor: float,
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
        calibration_factor=factor,
    )
    truth = cell.effect_ratio * margin
    return {
        "cell_id": cell.cell_id,
        "family": cell.family,
        "margin": margin,
        "cluster_count": count,
        "effect_ratio": cell.effect_ratio,
        "true_difference": truth,
        "trial_seed": trial_seed,
        "multiplier_seed": multiplier_seed,
        "estimate": result.estimate,
        "raw_radius": result.raw_radius,
        "upper_bound": result.upper_bound,
        "covers_truth": result.upper_bound + 1e-15 >= truth,
        "declares_equivalent": result.upper_bound <= margin,
        "power_primary": bool(family["power_primary"]),
    }


def _simulate_cell_thread_limited(
    cell: Cell,
    config: Mapping[str, Any],
) -> list[dict[str, object]]:
    rows = []
    for trial in range(int(config["trial_count_per_cell"])):
        rows.append(
            _one_trial(
                cell,
                config["families"][cell.family],
                trial_seed=_address(
                    int(config["trial_seed_base"]), cell.cell_id, trial
                ),
                multiplier_seed=_address(
                    int(config["multiplier_seed_base"]), cell.cell_id, trial
                ),
                confidence=float(config["confidence"]),
                replicates=int(config["multiplier_replicates"]),
                factor=float(config["frozen_calibration_factor"]),
            )
        )
    return rows


def _simulate_cell(
    cell: Cell,
    config: Mapping[str, Any],
) -> list[dict[str, object]]:
    with threadpool_limits(limits=WORKER_BLAS_THREADS, user_api="blas"):
        libraries = [
            info
            for info in threadpool_info()
            if info.get("user_api") == "blas"
        ]
        if not libraries or any(
            int(info.get("num_threads", 0)) != WORKER_BLAS_THREADS
            for info in libraries
        ):
            raise RuntimeError("scalar UQ worker BLAS thread contract drift")
        return _simulate_cell_thread_limited(cell, config)


def _wilson(successes: int, total: int, config: Mapping[str, Any]) -> tuple[float, float]:
    alpha = (1.0 - float(config["simultaneous_wilson"]["confidence"])) / int(
        config["simultaneous_wilson"]["comparisons"]
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


def _resource_preflight(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["resource_preflight"]
    process = psutil.Process(os.getpid())
    baseline = int(process.memory_info().rss)
    timings = []
    with threadpool_limits(limits=WORKER_BLAS_THREADS, user_api="blas"):
        for family in config["families"]:
            for count in config["cluster_counts"]:
                cell = Cell(family, 0.1, count, 0.5)
                started = time.perf_counter()
                _one_trial(
                    cell,
                    config["families"][family],
                    trial_seed=_address(
                        contract["trial_seed_base"], family, count
                    ),
                    multiplier_seed=_address(
                        contract["multiplier_seed_base"], family, count
                    ),
                    confidence=config["confidence"],
                    replicates=config["multiplier_replicates"],
                    factor=config["frozen_calibration_factor"],
                )
                timings.append(
                    {
                        "family": family,
                        "cluster_count": count,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
    estimated_wall = (
        sum(row["elapsed_seconds"] for row in timings)
        * len(config["margins"])
        * len(config["effect_ratios"])
        * int(config["trial_count_per_cell"])
        / int(config["max_workers"])
        * float(contract["safety_factor"])
    )
    observed_rss = int(process.memory_info().rss)
    estimated_rss = baseline + max(observed_rss - baseline, 67108864) * int(
        config["max_workers"]
    )
    report = {
        "timings": timings,
        "estimated_wall_seconds": estimated_wall,
        "estimated_rss_bytes": estimated_rss,
        "wall_limit_seconds": contract["maximum_estimated_wall_seconds"],
        "rss_limit_bytes": contract["maximum_estimated_rss_bytes"],
        "blas_threads_per_worker": WORKER_BLAS_THREADS,
        "passed": (
            estimated_wall <= contract["maximum_estimated_wall_seconds"]
            and estimated_rss <= contract["maximum_estimated_rss_bytes"]
        ),
    }
    report["analysis_sha256"] = _sha(report)
    _atomic_write(
        root / contract["artifact"],
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    if not report["passed"]:
        raise RuntimeError("scalar UQ resource preflight failed")
    return report


@contextmanager
def _owner_lock(root: Path, config: Mapping[str, Any]):
    path = root / config["artifact_paths"]["owner_lock"]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        payload = {"task_id": TASK_ID, "pid": os.getpid()}
        os.write(descriptor, _canonical(payload))
        os.close(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        path.unlink(missing_ok=True)


def _build_report(
    root: Path,
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    resource: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected = len(_cells(config)) * int(config["trial_count_per_cell"])
    if len(rows) != expected:
        raise RuntimeError("scalar UQ raw denominator drift")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["cell_id"]), []).append(row)
    summaries = []
    for cell in _cells(config):
        values = grouped.get(cell.cell_id, [])
        if len(values) != int(config["trial_count_per_cell"]):
            raise RuntimeError("scalar UQ cell denominator drift")
        coverage = sum(bool(row["covers_truth"]) for row in values)
        equivalence = sum(bool(row["declares_equivalent"]) for row in values)
        coverage_lcb, coverage_ucb = _wilson(coverage, len(values), config)
        equivalence_lcb, equivalence_ucb = _wilson(
            equivalence, len(values), config
        )
        summaries.append(
            {
                "cell_id": cell.cell_id,
                "family": cell.family,
                "margin": cell.margin,
                "cluster_count": cell.cluster_count,
                "effect_ratio": cell.effect_ratio,
                "power_primary": bool(
                    config["families"][cell.family]["power_primary"]
                ),
                "coverage_rate": coverage / len(values),
                "coverage_wilson_lcb": coverage_lcb,
                "coverage_wilson_ucb": coverage_ucb,
                "equivalence_rate": equivalence / len(values),
                "equivalence_wilson_lcb": equivalence_lcb,
                "equivalence_wilson_ucb": equivalence_ucb,
            }
        )
    coverage_pass = all(
        row["coverage_rate"] >= GATES["minimum_cell_coverage_rate"]
        and row["coverage_wilson_lcb"]
        >= GATES["minimum_cell_coverage_wilson_lcb"]
        for row in summaries
    )
    power_rules = {
        0.0: ("lcb", GATES["null_equivalence_wilson_lcb"]),
        0.5: ("lcb", GATES["local_half_margin_equivalence_wilson_lcb"]),
        1.0: ("ucb", GATES["boundary_equivalence_wilson_ucb"]),
        1.2: ("ucb", GATES["outside_equivalence_wilson_ucb"]),
    }
    power = []
    for ratio, (bound_name, threshold) in power_rules.items():
        strata = [
            row
            for row in summaries
            if row["cluster_count"] == 384
            and row["power_primary"]
            and row["effect_ratio"] == ratio
        ]
        if len(strata) != 16:
            raise RuntimeError("scalar UQ power denominator drift")
        failed = [
            row["cell_id"]
            for row in strata
            if (
                row["equivalence_wilson_lcb"] < threshold
                if bound_name == "lcb"
                else row["equivalence_wilson_ucb"] > threshold
            )
        ]
        power.append(
            {
                "effect_ratio": ratio,
                "bound": bound_name,
                "threshold": threshold,
                "stratum_count": len(strata),
                "failed_strata": failed,
                "global_iut_pass": not failed,
            }
        )
    passed = coverage_pass and all(row["global_iut_pass"] for row in power)
    report = {
        "task_id": TASK_ID,
        "schema_version": REPORT_SCHEMA,
        "verdict": PASS_VERDICT if passed else NO_GO_VERDICT,
        "qualified_claim": None,
        "factor": config["frozen_calibration_factor"],
        "multiplier_replicates": config["multiplier_replicates"],
        "margins": list(config["margins"]),
        "cluster_counts": list(config["cluster_counts"]),
        "coverage_all_passed": coverage_pass,
        "power_all_passed": all(row["global_iut_pass"] for row in power),
        "coverage_cell_count": len(summaries),
        "raw_trial_count": len(rows),
        "power_ledger": power,
        "resource_preflight_analysis_sha256": resource["analysis_sha256"],
        "design_outcomes_accessed": False,
        "claim_state": dict(CLAIM_BOUNDARY),
        "bindings": {
            "config": _binding(root / CONFIG_PATH, root),
            "implementation": _binding(Path(__file__).resolve(), root),
            "paired_uq": _binding(
                root / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py", root
            ),
        },
    }
    report["analysis_sha256"] = _sha(report)
    return report, summaries


def run(root: Path | None = None) -> dict[str, Any]:
    base = _root() if root is None else root.resolve()
    config = load_config(base)
    report_path = base / config["artifact_paths"]["report"]
    source_path = base / config["artifact_paths"]["source_data"]
    with _owner_lock(base, config):
        if report_path.exists() or source_path.exists():
            raise RuntimeError("scalar UQ canonical output already exists")
        resource = _resource_preflight(base, config)
        rows = []
        with ProcessPoolExecutor(max_workers=config["max_workers"]) as executor:
            futures = {
                executor.submit(_simulate_cell, cell, config): cell
                for cell in _cells(config)
            }
            for future in as_completed(futures):
                rows.extend(future.result())
        rows.sort(key=lambda row: (str(row["cell_id"]), int(row["trial_seed"])))
        report, summaries = _build_report(base, config, rows, resource)
        fields = list(rows[0])
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            delete=False,
            dir=source_path.parent,
            prefix=f".{source_path.name}.",
        )
        try:
            with temporary as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary.name, source_path)
        except BaseException:
            Path(temporary.name).unlink(missing_ok=True)
            raise
        report["bindings"]["source_data"] = _binding(source_path, base)
        report["coverage_summary_sha256"] = _sha(summaries)
        report["analysis_sha256"] = _sha(
            {key: value for key, value in report.items() if key != "analysis_sha256"}
        )
        _atomic_write(
            report_path,
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
        )
        live = json.loads(report_path.read_text(encoding="utf-8"))
        unsigned = dict(live)
        analysis = unsigned.pop("analysis_sha256")
        if analysis != _sha(unsigned):
            raise RuntimeError("scalar UQ report publication drift")
        return live


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    report = run()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "raw_trial_count": report["raw_trial_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())
