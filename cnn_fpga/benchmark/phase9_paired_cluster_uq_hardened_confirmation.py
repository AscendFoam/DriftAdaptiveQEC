"""Hardened third-split confirmation for the paired-cluster density UCB.

This child transaction preserves the earlier calibration NO-GO and power
extension verbatim.  It validates the frozen factor/count on a new split over
the dimensions actually produced by c16/c20/c24/c28 joint densities.  Count
qualification uses Wilson bounds, never point rates, and all paper claims stay
typed null.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from statistics import NormalDist
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np
import psutil

from cnn_fpga.benchmark.phase9_paired_cluster_uq import (
    NormUCB,
    half_trace_norm,
    paired_density_trace_ucb,
)


TASK_ID = "T-RISK-20260727-01"
CONFIG_PATH = "configs/phase9/t_risk_20260727_01_uq_hardened_confirmation.json"
CONFIG_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-CONFIRMATION-CONFIG-V3"
CHUNK_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-CHUNK-V2"
REPORT_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-CONFIRMATION-V2"
LOCK_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-OWNER-LOCK-V1"
RUN_IDENTITY_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-RUN-IDENTITY-V1"
PREFLIGHT_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-RESOURCE-PREFLIGHT-V2"
CONFIRMATION_SOURCE_SHA256_AT_IMPORT = sha256(Path(__file__).read_bytes()).hexdigest()
PAIRED_UQ_SOURCE_PATH = "cnn_fpga/benchmark/phase9_paired_cluster_uq.py"
PAIRED_UQ_SOURCE_SHA256_AT_IMPORT = sha256(
    (Path(__file__).resolve().parents[2] / PAIRED_UQ_SOURCE_PATH).read_bytes()
).hexdigest()
PASS_VERDICT = "PASS_PAIRED_CLUSTER_UQ_HARDENED_CONFIRMATION"
NO_GO_VERDICT = "NO_GO_PAIRED_CLUSTER_UQ_HARDENED_CONFIRMATION"
CLAIM_BOUNDARY = {
    "hardened_confirmation_only": True,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}
PARENT_CALIBRATION_CLAIM_BOUNDARY = {
    "calibration_only": True,
    "external_sota": None,
    "hardware_measured": None,
    "ler": None,
    "lifetime": None,
    "official_puviani_exact": None,
    "physical_break_even": None,
    "puviani_nmf_surpass": None,
    "twin_qualification": None,
}
PARENT_EXTENSION_CLAIM_BOUNDARY = {
    "external_sota": None,
    "hardware_measured": None,
    "ler": None,
    "lifetime": None,
    "official_puviani_exact": None,
    "physical_break_even": None,
    "power_extension_only": True,
    "puviani_nmf_surpass": None,
    "twin_qualification": None,
}
EXPECTED_FAMILIES = {
    "low_energy_balanced": {
        "spectrum_profile": "low_energy",
        "left_noise_weight": 0.4,
        "right_noise_weight": 0.4,
        "rare_probability": 1.0,
        "coherent_unitary": False,
        "power_primary": True,
    },
    "heavy_tail_rare_coherent": {
        "spectrum_profile": "heavy_tail",
        "left_noise_weight": 0.75,
        "right_noise_weight": 0.75,
        "rare_probability": 0.12,
        "coherent_unitary": True,
        "power_primary": True,
    },
    "heteroskedastic_coherent": {
        "spectrum_profile": "low_energy",
        "left_noise_weight": 0.25,
        "right_noise_weight": 0.75,
        "rare_probability": 1.0,
        "coherent_unitary": True,
        "power_primary": False,
    },
}
EXPECTED_GATES = {
    "minimum_cell_coverage_rate": 0.94,
    "minimum_cell_coverage_wilson_lcb": 0.9,
    "null_equivalence_wilson_lcb": 0.8,
    "local_005_equivalence_wilson_lcb": 0.65,
    "boundary_equivalence_wilson_ucb": 0.1,
    "outside_equivalence_wilson_ucb": 0.05,
}


@dataclass(frozen=True)
class CellSpec:
    family: str
    dimension: int
    cluster_count: int
    true_distance: float

    @property
    def cell_id(self) -> str:
        effect = str(self.true_distance).replace(".", "p")
        return (
            f"{self.family}__d{self.dimension}__n{self.cluster_count}"
            f"__delta{effect}"
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
    return sha256(_canonical(value)).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _verify_binding(root: Path, binding: Mapping[str, Any], *, label: str) -> Path:
    if set(binding) < {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} binding schema drift")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} binding escapes root") from exc
    expected = _binding(path, root)
    if any(binding.get(key) != expected[key] for key in expected):
        raise ValueError(f"{label} byte binding drift")
    return path


def _verify_self_hash(value: Mapping[str, Any], label: str) -> None:
    unsigned = dict(value)
    analysis = unsigned.pop("analysis_sha256", None)
    if not isinstance(analysis, str) or analysis != _sha(unsigned):
        raise ValueError(f"{label} self-hash drift")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _seed(base: int, *parts: object) -> int:
    digest = sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return (int(base) << 64) | int.from_bytes(digest[:8], "big")


def _wilson_bounds(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
    comparisons: int = 1,
) -> tuple[float, float]:
    if (
        total <= 0
        or not 0 <= successes <= total
        or not 0.5 < confidence < 1.0
        or comparisons < 1
    ):
        raise ValueError("invalid Wilson inputs")
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / (2.0 * comparisons))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return (
        max(0.0, (center - radius) / denominator),
        min(1.0, (center + radius) / denominator),
    )


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = root / CONFIG_PATH
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("claim_boundary") != CLAIM_BOUNDARY
        or config.get("formal_outcomes_accessed") is not False
        or config.get("dimensions") != [48, 60, 72, 84]
        or config.get("pilot_clusters_per_state") != 12
        or config.get("frozen_formal_clusters_per_state") != 384
        or config.get("true_trace_distances") != [0.0, 0.05, 0.1, 0.12]
        or config.get("margin") != 0.1
        or config.get("frozen_calibration_factor") != 1.0
        or config.get("confidence") != 0.95
        or int(config.get("multiplier_replicates", 0)) != 199
        or int(config.get("max_workers", 0)) != 4
    ):
        raise ValueError("hardened UQ config identity/claim/domain drift")
    if (
        config.get("families") != EXPECTED_FAMILIES
        or config.get("gates") != EXPECTED_GATES
    ):
        raise ValueError("hardened UQ preregistered family/gate drift")
    split = config.get("confirmation_split", {})
    if (
        split.get("independent_of_parent_splits") is not True
        or int(split.get("trial_count_per_cell", 0)) != 256
        or int(split.get("trial_seed_base", 0)) != 1430000
        or int(split.get("multiplier_seed_base", 0)) != 1440000
    ):
        raise ValueError("hardened UQ confirmation split drift")
    if sorted(config.get("families", {})) != [
        "heavy_tail_rare_coherent",
        "heteroskedastic_coherent",
        "low_energy_balanced",
    ]:
        raise ValueError("hardened UQ family set drift")
    if config.get("power_primary_families") != [
        "heavy_tail_rare_coherent",
        "low_energy_balanced",
    ]:
        raise ValueError("hardened UQ primary-family registry drift")
    for name, family in config["families"].items():
        required = {
            "spectrum_profile",
            "left_noise_weight",
            "right_noise_weight",
            "rare_probability",
            "coherent_unitary",
            "power_primary",
        }
        if set(family) != required:
            raise ValueError(f"hardened UQ family schema drift: {name}")
        if family["spectrum_profile"] not in {"low_energy", "heavy_tail"}:
            raise ValueError(f"hardened UQ spectrum profile drift: {name}")
        for key in ("left_noise_weight", "right_noise_weight"):
            if not 0.0 <= float(family[key]) < 1.0:
                raise ValueError(f"hardened UQ family weight drift: {name}/{key}")
        if not 0.0 < float(family["rare_probability"]) <= 1.0:
            raise ValueError(f"hardened UQ rare probability drift: {name}")
        if not isinstance(family["coherent_unitary"], bool) or not isinstance(
            family["power_primary"], bool
        ):
            raise ValueError(f"hardened UQ family Boolean drift: {name}")
        if bool(family["power_primary"]) is (
            name not in config["power_primary_families"]
        ):
            raise ValueError(f"hardened UQ primary-family flag drift: {name}")
    preflight = config.get("resource_preflight", {})
    if (
        preflight.get("required") is not True
        or preflight.get("disjoint_from_scientific_and_parent_seeds") is not True
        or int(preflight.get("trial_seed_base", 0)) != 1450000
        or int(preflight.get("multiplier_seed_base", 0)) != 1460000
        or float(preflight.get("eta_safety_factor", 0.0)) != 2.0
        or int(preflight.get("maximum_estimated_wall_seconds", 0)) != 7200
        or int(preflight.get("maximum_estimated_worker_pool_rss_bytes", 0))
        != 2147483648
        or int(preflight.get("maximum_estimated_total_task_rss_bytes", 0)) != 2684354560
        or preflight.get("rss_scope")
        != "worker_pool and supervisor are gated separately and jointly"
        or int(preflight["trial_seed_base"])
        in {
            int(split["trial_seed_base"]),
            int(split["multiplier_seed_base"]),
        }
        or int(preflight["multiplier_seed_base"])
        in {
            int(split["trial_seed_base"]),
            int(split["multiplier_seed_base"]),
            int(preflight["trial_seed_base"]),
        }
    ):
        raise ValueError("hardened UQ resource-preflight contract drift")
    simultaneous = config.get("simultaneous_wilson", {})
    if (
        simultaneous.get("confidence") != 0.95
        or simultaneous.get("global_comparisons") != 128
        or simultaneous.get("coverage_cell_comparisons") != 128
        or simultaneous.get("power_stratum_comparisons") != 128
        or simultaneous.get("method")
        != (
            "single-family Bonferroni-adjusted two-sided Wilson bounds "
            "across all 96 coverage and 32 power intervals"
        )
    ):
        raise ValueError("hardened UQ simultaneous-Wilson contract drift")
    artifacts = config.get("artifact_paths", {})
    if set(artifacts) != {
        "run_directory",
        "chunk_directory",
        "owner_lock",
        "run_identity",
        "heartbeat",
        "report",
        "source_data",
    }:
        raise ValueError("hardened UQ artifact-path schema drift")

    parents = config.get("parent_artifacts", {})
    expected_parent_names = {
        "calibration_report",
        "calibration_source_data",
        "calibration_config",
        "extension_report",
        "extension_source_data",
        "extension_config",
    }
    if set(parents) != expected_parent_names:
        raise ValueError("hardened UQ parent artifact set drift")
    parent_paths = {
        name: _verify_binding(root, binding, label=f"parent/{name}")
        for name, binding in parents.items()
    }
    calibration = json.loads(
        parent_paths["calibration_report"].read_text(encoding="utf-8")
    )
    extension = json.loads(parent_paths["extension_report"].read_text(encoding="utf-8"))
    _verify_self_hash(calibration, "parent calibration")
    _verify_self_hash(extension, "parent extension")
    for label, report in (
        ("parent calibration", calibration),
        ("parent extension", extension),
    ):
        bindings = report.get("bindings")
        if not isinstance(bindings, Mapping) or not bindings:
            raise ValueError(f"{label} live bindings missing")
        for name, binding in bindings.items():
            if not isinstance(binding, Mapping):
                raise ValueError(f"{label}/{name} binding type drift")
            _verify_binding(root, binding, label=f"{label}/{name}")
    if (
        calibration.get("analysis_sha256")
        != parents["calibration_report"]["analysis_sha256"]
        or calibration.get("verdict") != "NO_GO_PAIRED_CLUSTER_UQ_CALIBRATION"
        or calibration.get("selected_calibration_factor") != 1.0
        or calibration.get("formal_outcomes_accessed") is not False
        or calibration.get("claim_state") != PARENT_CALIBRATION_CLAIM_BOUNDARY
        or extension.get("analysis_sha256")
        != parents["extension_report"]["analysis_sha256"]
        or extension.get("parent_analysis_sha256") != calibration.get("analysis_sha256")
        or extension.get("verdict") != "PASS_PAIRED_CLUSTER_UQ_POWER_EXTENSION"
        or extension.get("selected_formal_clusters_per_state") != 384
        or extension.get("frozen_parent_calibration_factor") != 1.0
        or extension.get("parent_no_go_preserved") is not True
        or extension.get("claim_state") != PARENT_EXTENSION_CLAIM_BOUNDARY
    ):
        raise ValueError("hardened UQ parent semantic binding drift")
    return config, calibration, extension


def _execution_bindings(
    root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, object]]:
    live_confirmation_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    live_paired_sha = sha256((root / PAIRED_UQ_SOURCE_PATH).read_bytes()).hexdigest()
    if (
        live_confirmation_sha != CONFIRMATION_SOURCE_SHA256_AT_IMPORT
        or live_paired_sha != PAIRED_UQ_SOURCE_SHA256_AT_IMPORT
    ):
        raise RuntimeError(
            "hardened UQ loaded-source/live-file drift; fresh process required"
        )
    return {
        "config": _binding(root / CONFIG_PATH, root),
        "confirmation_source": _binding(Path(__file__).resolve(), root),
        "paired_cluster_uq_source": _binding(root / PAIRED_UQ_SOURCE_PATH, root),
        **{
            f"parent_{name}": _binding(root / str(binding["path"]), root)
            for name, binding in config["parent_artifacts"].items()
        },
    }


def _worker_source_attestation(
    expected_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    confirmation_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    paired_sha = sha256(
        (Path(__file__).resolve().parents[2] / PAIRED_UQ_SOURCE_PATH).read_bytes()
    ).hexdigest()
    if (
        confirmation_sha != CONFIRMATION_SOURCE_SHA256_AT_IMPORT
        or paired_sha != PAIRED_UQ_SOURCE_SHA256_AT_IMPORT
        or confirmation_sha != expected_bindings["confirmation_source"].get("sha256")
        or paired_sha != expected_bindings["paired_cluster_uq_source"].get("sha256")
    ):
        raise RuntimeError("hardened UQ worker loaded/live source attestation drift")
    return {
        "worker_pid": os.getpid(),
        "confirmation_source_sha256": confirmation_sha,
        "paired_cluster_uq_source_sha256": paired_sha,
    }


def _verify_execution_identity_live(
    root: Path,
    config: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    _verify_self_hash(identity, "hardened UQ run identity")
    expected_keys = {
        "task_id",
        "schema_version",
        "run_id",
        "config_analysis_sha256",
        "execution_bindings",
        "claim_state",
        "created_utc",
        "analysis_sha256",
    }
    try:
        UUID(str(identity.get("run_id")))
    except ValueError as exc:
        raise RuntimeError("hardened UQ run identity UUID drift") from exc
    if (
        set(identity) != expected_keys
        or identity.get("task_id") != TASK_ID
        or identity.get("schema_version") != RUN_IDENTITY_SCHEMA
        or identity.get("config_analysis_sha256") != _sha(config)
        or identity.get("execution_bindings") != _execution_bindings(root, config)
        or identity.get("claim_state") != CLAIM_BOUNDARY
        or not isinstance(identity.get("created_utc"), str)
    ):
        raise RuntimeError("hardened UQ execution identity/live byte drift")


def _load_or_create_run_identity(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / str(config["artifact_paths"]["run_identity"])
    if path.exists():
        identity = json.loads(path.read_text(encoding="utf-8"))
    else:
        identity = {
            "task_id": TASK_ID,
            "schema_version": RUN_IDENTITY_SCHEMA,
            "run_id": str(uuid4()),
            "config_analysis_sha256": _sha(config),
            "execution_bindings": _execution_bindings(root, config),
            "claim_state": dict(config["claim_boundary"]),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        identity["analysis_sha256"] = _sha(identity)
        _atomic_text(
            path,
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    live = json.loads(path.read_text(encoding="utf-8"))
    if live != identity:
        raise RuntimeError("hardened UQ run identity atomic binding drift")
    _verify_execution_identity_live(root, config, live)
    return live


def _center_probabilities(dimension: int, profile: str) -> np.ndarray:
    if dimension < 4:
        raise ValueError("density dimension must be at least four")
    tail_count = dimension - 2
    if profile == "low_energy":
        tail = np.exp(-np.arange(tail_count, dtype=np.float64) / 4.0)
    elif profile == "heavy_tail":
        tail = 1.0 / np.sqrt(np.arange(1, tail_count + 1, dtype=np.float64))
    else:
        raise ValueError("unknown spectrum profile")
    tail *= 0.16 / float(np.sum(tail))
    center = np.concatenate(([0.42, 0.42], tail))
    if abs(float(np.sum(center)) - 1.0) > 1e-12 or np.min(center) <= 0.0:
        raise RuntimeError("density center construction drift")
    return center


@lru_cache(maxsize=None)
def _fourier_unitary(dimension: int) -> np.ndarray:
    indices = np.arange(dimension)
    return np.exp(
        2j * np.pi * np.outer(indices, indices) / float(dimension)
    ) / math.sqrt(dimension)


def _phase_noise(
    rng: np.random.Generator,
    *,
    count: int,
    center: np.ndarray,
    rare_probability: float,
) -> np.ndarray:
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(count, len(center)))
    kets = np.sqrt(center)[None, :] * np.exp(1j * phases)
    pure = np.einsum("ni,nj->nij", kets, kets.conj(), optimize=True)
    if rare_probability < 1.0:
        active = rng.random(count) < rare_probability
        pure[~active] = np.diag(center)
    return pure


def _physical_density_trial(
    *,
    dimension: int,
    count: int,
    true_distance: float,
    family: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if count < 2 or not 0.0 <= true_distance <= 0.12:
        raise ValueError("density trial count/effect outside design")
    center = _center_probabilities(dimension, str(family["spectrum_profile"]))
    left_weight = float(family["left_noise_weight"])
    right_weight = float(family["right_noise_weight"])
    denominator = (2.0 - left_weight - right_weight) / 2.0
    shift = 0.0 if true_distance == 0.0 else true_distance / denominator
    direction = np.zeros(dimension, dtype=np.float64)
    direction[0] = shift / 2.0
    direction[1] = -shift / 2.0
    if np.min(center - np.abs(direction)) < -1e-15:
        raise ValueError("requested effect violates PSD design envelope")
    base_left = np.diag(center + direction).astype(np.complex128)
    base_right = np.diag(center - direction).astype(np.complex128)
    center_density = np.diag(center).astype(np.complex128)
    rng = np.random.default_rng(seed)
    rare_probability = float(family["rare_probability"])
    noise_left = _phase_noise(
        rng, count=count, center=center, rare_probability=rare_probability
    )
    noise_right = _phase_noise(
        rng, count=count, center=center, rare_probability=rare_probability
    )
    left = (1.0 - left_weight) * base_left + left_weight * noise_left
    right = (1.0 - right_weight) * base_right + right_weight * noise_right
    population_left = (1.0 - left_weight) * base_left + left_weight * center_density
    population_right = (1.0 - right_weight) * base_right + right_weight * center_density
    truth = half_trace_norm(population_left - population_right)
    if abs(truth - true_distance) > 1e-12:
        raise RuntimeError("analytic density effect drift")
    if bool(family["coherent_unitary"]):
        unitary = _fourier_unitary(dimension)
        left = np.einsum("ij,njk,lk->nil", unitary, left, unitary.conj(), optimize=True)
        right = np.einsum(
            "ij,njk,lk->nil", unitary, right, unitary.conj(), optimize=True
        )
    return left, right, truth


def _validate_density_stack(stack: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(stack, dtype=np.complex128)
    if value.ndim != 3 or value.shape[1] != value.shape[2]:
        raise ValueError(f"{name} density stack shape drift")
    hermitian = 0.5 * (value + value.conj().transpose(0, 2, 1))
    traces = np.trace(hermitian, axis1=1, axis2=2)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if (
        not np.all(np.isfinite(value))
        or float(np.max(np.abs(value - hermitian))) > 1e-9
        or float(np.max(np.abs(traces - 1.0))) > 1e-9
        or float(np.min(eigenvalues)) < -1e-9
    ):
        raise ValueError(f"{name} contains a non-physical density")
    return hermitian


def paired_density_trace_ucb_physical(
    left: np.ndarray, right: np.ndarray, **kwargs: Any
) -> NormUCB:
    left_valid = _validate_density_stack(left, "left")
    right_valid = _validate_density_stack(right, "right")
    return paired_density_trace_ucb(left_valid, right_valid, **kwargs)


def _cells(config: Mapping[str, Any]) -> list[CellSpec]:
    cells = [
        CellSpec(family, int(dimension), int(count), float(effect))
        for family in config["families"]
        for dimension in config["dimensions"]
        for count in (
            int(config["pilot_clusters_per_state"]),
            int(config["frozen_formal_clusters_per_state"]),
        )
        for effect in config["true_trace_distances"]
    ]
    if len(cells) != 96 or len({cell.cell_id for cell in cells}) != 96:
        raise RuntimeError("hardened UQ cell accounting drift")
    return cells


def _measure_preflight_trial(
    config: Mapping[str, Any],
    expected_execution_bindings: Mapping[str, Mapping[str, Any]],
    *,
    family_name: str,
    dimension: int,
    cluster_count: int,
) -> dict[str, Any]:
    attestation_at_start = _worker_source_attestation(expected_execution_bindings)
    contract = config["resource_preflight"]
    trial_seed = _seed(
        int(contract["trial_seed_base"]),
        "resource_preflight",
        family_name,
        dimension,
        cluster_count,
    )
    multiplier_seed = _seed(
        int(contract["multiplier_seed_base"]),
        "resource_preflight",
        family_name,
        dimension,
        cluster_count,
    )
    process = psutil.Process(os.getpid())
    peak = [int(process.memory_info().rss)]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.01):
            peak[0] = max(peak[0], int(process.memory_info().rss))

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        left, right, _ = _physical_density_trial(
            dimension=dimension,
            count=cluster_count,
            true_distance=0.05,
            family=config["families"][family_name],
            seed=trial_seed,
        )
        ucb = paired_density_trace_ucb_physical(
            left,
            right,
            confidence=float(config["confidence"]),
            multiplier_replicates=int(config["multiplier_replicates"]),
            seed=multiplier_seed,
            calibration_factor=float(config["frozen_calibration_factor"]),
        )
    finally:
        stop.set()
        sampler.join(timeout=1.0)
        peak[0] = max(peak[0], int(process.memory_info().rss))
    attestation_at_end = _worker_source_attestation(expected_execution_bindings)
    if attestation_at_end != attestation_at_start:
        raise RuntimeError("resource-preflight worker source changed during trial")
    return {
        "family": family_name,
        "dimension": dimension,
        "cluster_count": cluster_count,
        "true_distance": 0.05,
        "trial_seed": trial_seed,
        "multiplier_seed": multiplier_seed,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": peak[0],
        "ucb_upper_bound": ucb.upper_bound,
        "worker_attestation": attestation_at_end,
    }


def _wilson_feasibility(config: Mapping[str, Any]) -> dict[str, Any]:
    trials = int(config["confirmation_split"]["trial_count_per_cell"])
    simultaneous = config["simultaneous_wilson"]
    confidence = float(simultaneous["confidence"])
    comparisons = int(simultaneous["global_comparisons"])
    best_lcb, _ = _wilson_bounds(
        trials,
        trials,
        confidence=confidence,
        comparisons=comparisons,
    )
    _, zero_ucb = _wilson_bounds(
        0,
        trials,
        confidence=confidence,
        comparisons=comparisons,
    )
    gates = config["gates"]
    required_lcb = max(
        float(gates["minimum_cell_coverage_wilson_lcb"]),
        float(gates["null_equivalence_wilson_lcb"]),
        float(gates["local_005_equivalence_wilson_lcb"]),
    )
    required_ucb = min(
        float(gates["boundary_equivalence_wilson_ucb"]),
        float(gates["outside_equivalence_wilson_ucb"]),
    )
    return {
        "trial_count": trials,
        "confidence": confidence,
        "global_comparisons": comparisons,
        "all_successes_wilson_lcb": best_lcb,
        "zero_successes_wilson_ucb": zero_ucb,
        "maximum_required_lcb": required_lcb,
        "minimum_required_ucb": required_ucb,
        "attainable": best_lcb >= required_lcb and zero_ucb <= required_ucb,
    }


def _validate_resource_preflight(
    root: Path,
    config: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    _verify_execution_identity_live(root, config, run_identity)
    _verify_self_hash(report, "resource preflight")
    records = report.get("records")
    contract = config["resource_preflight"]
    feasibility = _wilson_feasibility(config)
    if (
        report.get("task_id") != TASK_ID
        or report.get("schema_version") != PREFLIGHT_SCHEMA
        or report.get("config_analysis_sha256") != _sha(config)
        or report.get("run_id") != run_identity["run_id"]
        or report.get("run_identity_analysis_sha256") != run_identity["analysis_sha256"]
        or report.get("execution_bindings") != run_identity["execution_bindings"]
        or report.get("scientific_outcomes_accessed") is not False
        or report.get("claim_state") != CLAIM_BOUNDARY
        or report.get("wilson_feasibility") != feasibility
        or report.get("eta_method")
        != (
            "max(serial-child-work/workers, observed ProcessPool wall throughput) "
            "times full scientific trial ratio and safety factor"
        )
        or report.get("rss_scope") != contract["rss_scope"]
        or not isinstance(records, list)
        or len(records) != 24
    ):
        raise RuntimeError("resource preflight identity/gate drift")
    identities = {
        (
            row["family"],
            int(row["dimension"]),
            int(row["cluster_count"]),
        )
        for row in records
    }
    expected = {
        (family, int(dimension), int(count))
        for family in config["families"]
        for dimension in config["dimensions"]
        for count in (
            config["pilot_clusters_per_state"],
            config["frozen_formal_clusters_per_state"],
        )
    }
    if identities != expected:
        raise RuntimeError("resource preflight cell coverage drift")
    confirmation_sha = run_identity["execution_bindings"]["confirmation_source"][
        "sha256"
    ]
    paired_sha = run_identity["execution_bindings"]["paired_cluster_uq_source"][
        "sha256"
    ]
    for row in records:
        expected_trial = _seed(
            int(contract["trial_seed_base"]),
            "resource_preflight",
            row["family"],
            row["dimension"],
            row["cluster_count"],
        )
        expected_multiplier = _seed(
            int(contract["multiplier_seed_base"]),
            "resource_preflight",
            row["family"],
            row["dimension"],
            row["cluster_count"],
        )
        if (
            int(row["trial_seed"]) != expected_trial
            or int(row["multiplier_seed"]) != expected_multiplier
            or not math.isfinite(float(row["elapsed_seconds"]))
            or float(row["elapsed_seconds"]) <= 0.0
            or int(row["peak_process_rss_bytes"]) <= 0
            or row.get("worker_attestation", {}).get("confirmation_source_sha256")
            != confirmation_sha
            or row.get("worker_attestation", {}).get("paired_cluster_uq_source_sha256")
            != paired_sha
            or int(row.get("worker_attestation", {}).get("worker_pid", 0)) <= 0
        ):
            raise RuntimeError("resource preflight record drift")
    workers = int(config["max_workers"])
    full_trial_ratio = len(config["true_trace_distances"]) * int(
        config["confirmation_split"]["trial_count_per_cell"]
    )
    expected_serial_projection = (
        sum(float(row["elapsed_seconds"]) for row in records)
        * full_trial_ratio
        / workers
    )
    observed_wall = float(report.get("observed_process_pool_wall_seconds", math.nan))
    observed_supervisor_rss = int(report.get("observed_supervisor_peak_rss_bytes", -1))
    if (
        not math.isfinite(observed_wall)
        or observed_wall <= 0.0
        or observed_supervisor_rss <= 0
    ):
        raise RuntimeError("resource preflight observed resource drift")
    expected_pool_projection = observed_wall * full_trial_ratio
    expected_wall = max(expected_serial_projection, expected_pool_projection) * float(
        contract["eta_safety_factor"]
    )
    expected_worker_pool_rss = (
        max(int(row["peak_process_rss_bytes"]) for row in records) * workers
    )
    expected_total_task_rss = expected_worker_pool_rss + observed_supervisor_rss
    expected_passed = (
        expected_wall <= int(contract["maximum_estimated_wall_seconds"])
        and expected_worker_pool_rss
        <= int(contract["maximum_estimated_worker_pool_rss_bytes"])
        and expected_total_task_rss
        <= int(contract["maximum_estimated_total_task_rss_bytes"])
        and feasibility["attainable"] is True
    )
    float_fields = {
        "serial_child_work_projection_seconds": expected_serial_projection,
        "observed_pool_projection_seconds": expected_pool_projection,
        "estimated_wall_seconds_with_safety_factor": expected_wall,
    }
    if any(
        not math.isclose(
            float(report.get(name, math.nan)),
            expected_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for name, expected_value in float_fields.items()
    ) or (
        int(report.get("estimated_worker_pool_rss_bytes", -1))
        != expected_worker_pool_rss
        or int(report.get("estimated_total_task_rss_bytes", -1))
        != expected_total_task_rss
        or report.get("wall_limit_seconds")
        != int(contract["maximum_estimated_wall_seconds"])
        or report.get("worker_pool_rss_limit_bytes")
        != int(contract["maximum_estimated_worker_pool_rss_bytes"])
        or report.get("total_task_rss_limit_bytes")
        != int(contract["maximum_estimated_total_task_rss_bytes"])
        or report.get("passed") is not expected_passed
    ):
        raise RuntimeError("resource preflight derived summary drift")


def run_resource_preflight(
    root: Path,
    config: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    owner: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_owner_lock(root, config, owner)
    _verify_execution_identity_live(root, config, run_identity)
    contract = config["resource_preflight"]
    path = root / str(contract["artifact"])
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate_resource_preflight(root, config, run_identity, report)
        if report["passed"] is not True:
            raise RuntimeError("resource preflight ETA/RSS/feasibility gate failed")
        return report
    jobs = [
        (family, int(dimension), int(count))
        for family in config["families"]
        for dimension in config["dimensions"]
        for count in (
            config["pilot_clusters_per_state"],
            config["frozen_formal_clusters_per_state"],
        )
    ]
    workers = int(config["max_workers"])
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    supervisor = psutil.Process(os.getpid())
    supervisor_peak = [int(supervisor.memory_info().rss)]
    supervisor_stop = threading.Event()

    def sample_supervisor_rss() -> None:
        while not supervisor_stop.wait(0.01):
            supervisor_peak[0] = max(
                supervisor_peak[0], int(supervisor.memory_info().rss)
            )

    supervisor_sampler = threading.Thread(
        target=sample_supervisor_rss,
        daemon=True,
    )
    supervisor_sampler.start()
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _measure_preflight_trial,
                    config,
                    run_identity["execution_bindings"],
                    family_name=family,
                    dimension=dimension,
                    cluster_count=count,
                ): (family, dimension, count)
                for family, dimension, count in jobs
            }
            for future in as_completed(futures):
                records.append(future.result())
    finally:
        supervisor_stop.set()
        supervisor_sampler.join(timeout=1.0)
        supervisor_peak[0] = max(supervisor_peak[0], int(supervisor.memory_info().rss))
    _validate_owner_lock(root, config, owner)
    observed_pool_wall = time.perf_counter() - started
    records.sort(
        key=lambda row: (
            str(row["family"]),
            int(row["dimension"]),
            int(row["cluster_count"]),
        )
    )
    _verify_execution_identity_live(root, config, run_identity)
    trial_count = int(config["confirmation_split"]["trial_count_per_cell"])
    safety_factor = float(contract["eta_safety_factor"])
    full_trial_ratio = len(config["true_trace_distances"]) * trial_count
    serial_work_projection = (
        sum(float(row["elapsed_seconds"]) for row in records)
        * full_trial_ratio
        / workers
    )
    observed_pool_projection = observed_pool_wall * full_trial_ratio
    estimated_wall = (
        max(serial_work_projection, observed_pool_projection) * safety_factor
    )
    estimated_worker_pool_rss = (
        max(int(row["peak_process_rss_bytes"]) for row in records) * workers
    )
    supervisor_rss = supervisor_peak[0]
    estimated_total_task_rss = estimated_worker_pool_rss + supervisor_rss
    feasibility = _wilson_feasibility(config)
    passed = (
        estimated_wall <= float(contract["maximum_estimated_wall_seconds"])
        and estimated_worker_pool_rss
        <= int(contract["maximum_estimated_worker_pool_rss_bytes"])
        and estimated_total_task_rss
        <= int(contract["maximum_estimated_total_task_rss_bytes"])
        and feasibility["attainable"] is True
    )
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": PREFLIGHT_SCHEMA,
        "config_analysis_sha256": _sha(config),
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "execution_bindings": run_identity["execution_bindings"],
        "scientific_outcomes_accessed": False,
        "claim_state": dict(config["claim_boundary"]),
        "records": records,
        "observed_process_pool_wall_seconds": observed_pool_wall,
        "serial_child_work_projection_seconds": serial_work_projection,
        "observed_pool_projection_seconds": observed_pool_projection,
        "eta_method": (
            "max(serial-child-work/workers, observed ProcessPool wall throughput) "
            "times full scientific trial ratio and safety factor"
        ),
        "estimated_wall_seconds_with_safety_factor": estimated_wall,
        "estimated_worker_pool_rss_bytes": estimated_worker_pool_rss,
        "observed_supervisor_peak_rss_bytes": supervisor_rss,
        "estimated_total_task_rss_bytes": estimated_total_task_rss,
        "rss_scope": contract["rss_scope"],
        "wall_limit_seconds": int(contract["maximum_estimated_wall_seconds"]),
        "worker_pool_rss_limit_bytes": int(
            contract["maximum_estimated_worker_pool_rss_bytes"]
        ),
        "total_task_rss_limit_bytes": int(
            contract["maximum_estimated_total_task_rss_bytes"]
        ),
        "wilson_feasibility": feasibility,
        "passed": passed,
    }
    report["analysis_sha256"] = _sha(report)
    _validate_owner_lock(root, config, owner)
    _atomic_text(
        path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _validate_owner_lock(root, config, owner)
    live = json.loads(path.read_text(encoding="utf-8"))
    if live != report:
        raise RuntimeError("resource preflight atomic commit drift")
    _validate_resource_preflight(root, config, run_identity, live)
    if not passed:
        raise RuntimeError("resource preflight ETA/RSS/feasibility gate failed")
    return report


def _simulate_cell(
    config: Mapping[str, Any],
    cell_payload: Mapping[str, Any],
    expected_execution_bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    attestation_at_start = _worker_source_attestation(expected_execution_bindings)
    cell = CellSpec(**cell_payload)
    split = config["confirmation_split"]
    family = config["families"][cell.family]
    records: list[dict[str, Any]] = []
    for trial in range(int(split["trial_count_per_cell"])):
        trial_seed = _seed(
            int(split["trial_seed_base"]),
            "confirmation",
            cell.family,
            cell.dimension,
            cell.cluster_count,
            cell.true_distance,
            trial,
        )
        multiplier_seed = _seed(
            int(split["multiplier_seed_base"]),
            "confirmation",
            cell.family,
            cell.dimension,
            cell.cluster_count,
            cell.true_distance,
            trial,
        )
        left, right, truth = _physical_density_trial(
            dimension=cell.dimension,
            count=cell.cluster_count,
            true_distance=cell.true_distance,
            family=family,
            seed=trial_seed,
        )
        ucb = paired_density_trace_ucb_physical(
            left,
            right,
            confidence=float(config["confidence"]),
            multiplier_replicates=int(config["multiplier_replicates"]),
            seed=multiplier_seed,
            calibration_factor=float(config["frozen_calibration_factor"]),
        )
        records.append(
            {
                "split": "confirmation",
                "family": cell.family,
                "dimension": cell.dimension,
                "cluster_count": cell.cluster_count,
                "true_distance": truth,
                "trial": trial,
                "trial_seed": trial_seed,
                "multiplier_seed": multiplier_seed,
                "estimate": ucb.estimate,
                "raw_radius": ucb.raw_radius,
                "power_primary": bool(family["power_primary"]),
            }
        )
    attestation_at_end = _worker_source_attestation(expected_execution_bindings)
    if attestation_at_end != attestation_at_start:
        raise RuntimeError("scientific worker source changed during cell")
    return {
        "records": records,
        "worker_attestation": attestation_at_end,
    }


def _chunk_path(root: Path, config: Mapping[str, Any], cell: CellSpec) -> Path:
    return (
        root / str(config["artifact_paths"]["chunk_directory"]) / f"{cell.cell_id}.json"
    )


def _chunk_payload(
    config: Mapping[str, Any],
    cell: CellSpec,
    records: Sequence[Mapping[str, Any]],
    run_identity: Mapping[str, Any],
    worker_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": CHUNK_SCHEMA,
        "config_analysis_sha256": _sha(config),
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "worker_attestation": dict(worker_attestation),
        "cell": asdict(cell),
        "record_count": len(records),
        "records": list(records),
    }
    payload["analysis_sha256"] = _sha(payload)
    return payload


def _validate_chunk(
    root: Path,
    config: Mapping[str, Any],
    cell: CellSpec,
    chunk: Mapping[str, Any],
    run_identity: Mapping[str, Any],
) -> None:
    _verify_execution_identity_live(root, config, run_identity)
    _verify_self_hash(chunk, f"chunk/{cell.cell_id}")
    expected_keys = {
        "task_id",
        "schema_version",
        "config_analysis_sha256",
        "run_id",
        "run_identity_analysis_sha256",
        "worker_attestation",
        "cell",
        "record_count",
        "records",
        "analysis_sha256",
    }
    records = chunk.get("records")
    expected_count = int(config["confirmation_split"]["trial_count_per_cell"])
    if (
        set(chunk) != expected_keys
        or chunk.get("task_id") != TASK_ID
        or chunk.get("schema_version") != CHUNK_SCHEMA
        or chunk.get("config_analysis_sha256") != _sha(config)
        or chunk.get("run_id") != run_identity["run_id"]
        or chunk.get("run_identity_analysis_sha256") != run_identity["analysis_sha256"]
        or chunk.get("worker_attestation", {}).get("confirmation_source_sha256")
        != run_identity["execution_bindings"]["confirmation_source"]["sha256"]
        or chunk.get("worker_attestation", {}).get("paired_cluster_uq_source_sha256")
        != run_identity["execution_bindings"]["paired_cluster_uq_source"]["sha256"]
        or int(chunk.get("worker_attestation", {}).get("worker_pid", 0)) <= 0
        or chunk.get("cell") != asdict(cell)
        or chunk.get("record_count") != expected_count
        or not isinstance(records, list)
        or len(records) != expected_count
    ):
        raise ValueError(f"chunk identity/accounting drift: {cell.cell_id}")
    seen: set[tuple[int, int]] = set()
    split = config["confirmation_split"]
    record_keys = {
        "split",
        "family",
        "dimension",
        "cluster_count",
        "true_distance",
        "trial",
        "trial_seed",
        "multiplier_seed",
        "estimate",
        "raw_radius",
        "power_primary",
    }
    for trial, record in enumerate(records):
        expected_trial_seed = _seed(
            int(split["trial_seed_base"]),
            "confirmation",
            cell.family,
            cell.dimension,
            cell.cluster_count,
            cell.true_distance,
            trial,
        )
        expected_multiplier_seed = _seed(
            int(split["multiplier_seed_base"]),
            "confirmation",
            cell.family,
            cell.dimension,
            cell.cluster_count,
            cell.true_distance,
            trial,
        )
        identity = (
            int(record.get("trial_seed", -1)),
            int(record.get("multiplier_seed", -1)),
        )
        if (
            set(record) != record_keys
            or record.get("split") != "confirmation"
            or record.get("family") != cell.family
            or record.get("dimension") != cell.dimension
            or record.get("cluster_count") != cell.cluster_count
            or abs(float(record.get("true_distance", -1)) - cell.true_distance) > 1e-12
            or record.get("trial") != trial
            or identity != (expected_trial_seed, expected_multiplier_seed)
            or identity in seen
            or record.get("power_primary")
            is not bool(config["families"][cell.family]["power_primary"])
            or not math.isfinite(float(record.get("estimate", math.nan)))
            or not math.isfinite(float(record.get("raw_radius", math.nan)))
            or float(record["estimate"]) < 0.0
            or float(record["raw_radius"]) < 0.0
        ):
            raise ValueError(f"chunk record drift: {cell.cell_id}/{trial}")
        seen.add(identity)


@contextmanager
def _owner_lock(root: Path, config: Mapping[str, Any]) -> Any:
    path = root / str(config["artifact_paths"]["owner_lock"])
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    payload = {
        "task_id": TASK_ID,
        "schema_version": LOCK_SCHEMA,
        "owner_token": token,
        "pid": os.getpid(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_analysis_sha256": _sha(config),
    }
    payload["analysis_sha256"] = _sha(payload)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("hardened UQ owner lock already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        yield payload
    finally:
        if not path.exists():
            raise RuntimeError("hardened UQ owner lock disappeared while active")
        _validate_owner_lock(root, config, payload)
        path.unlink()


def _validate_owner_lock(
    root: Path,
    config: Mapping[str, Any],
    owner: Mapping[str, Any],
) -> None:
    path = root / str(config["artifact_paths"]["owner_lock"])
    if not path.exists():
        raise RuntimeError("hardened UQ owner lock ownership lost")
    live = json.loads(path.read_text(encoding="utf-8"))
    _verify_self_hash(live, "owner lock")
    if (
        live != owner
        or live.get("owner_token") != owner.get("owner_token")
        or live.get("config_analysis_sha256") != _sha(config)
        or live.get("pid") != os.getpid()
    ):
        raise RuntimeError("hardened UQ owner lock ownership drift")


def _heartbeat(
    root: Path,
    config: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    owner: Mapping[str, Any],
    *,
    completed: int,
    total: int,
    active: bool,
    state: str,
    stage: str,
    error_type: str | None = None,
) -> None:
    if state not in {"RUNNING", "PREFLIGHT_PASS", "PASS", "NO_GO", "FAILED"}:
        raise ValueError("hardened UQ heartbeat state drift")
    if stage not in {"PRECHECK", "PREFLIGHT", "SCIENCE", "FINALIZE", "COMPLETE"}:
        raise ValueError("hardened UQ heartbeat stage drift")
    owner_lock_valid = True
    try:
        _validate_owner_lock(root, config, owner)
    except BaseException:
        owner_lock_valid = False
        if state != "FAILED":
            raise
    payload = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-HEARTBEAT-V2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "owner_token": owner["owner_token"],
        "owner_lock_valid": owner_lock_valid,
        "config_analysis_sha256": _sha(config),
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "completed_cells": completed,
        "expected_cells": total,
        "active": active,
        "state": state,
        "stage": stage,
        "error_type": error_type,
    }
    payload["analysis_sha256"] = _sha(payload)
    _atomic_text(
        root / str(config["artifact_paths"]["heartbeat"]),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def simulate_confirmation(
    root: Path,
    config: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    owner: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_owner_lock(root, config, owner)
    _verify_execution_identity_live(root, config, run_identity)
    cells = _cells(config)
    chunks: dict[str, dict[str, Any]] = {}
    pending: list[CellSpec] = []
    for cell in cells:
        path = _chunk_path(root, config, cell)
        if path.exists():
            chunk = json.loads(path.read_text(encoding="utf-8"))
            _validate_chunk(root, config, cell, chunk, run_identity)
            chunks[cell.cell_id] = chunk
        else:
            pending.append(cell)
    _heartbeat(
        root,
        config,
        run_identity,
        owner,
        completed=len(chunks),
        total=len(cells),
        active=True,
        state="RUNNING",
        stage="SCIENCE",
    )
    if pending:
        with ProcessPoolExecutor(max_workers=int(config["max_workers"])) as executor:
            futures = {
                executor.submit(
                    _simulate_cell,
                    config,
                    asdict(cell),
                    run_identity["execution_bindings"],
                ): cell
                for cell in pending
            }
            for future in as_completed(futures):
                _validate_owner_lock(root, config, owner)
                _verify_execution_identity_live(root, config, run_identity)
                cell = futures[future]
                worker_result = future.result()
                attestation = worker_result.get("worker_attestation", {})
                if (
                    attestation.get("confirmation_source_sha256")
                    != run_identity["execution_bindings"]["confirmation_source"][
                        "sha256"
                    ]
                    or attestation.get("paired_cluster_uq_source_sha256")
                    != run_identity["execution_bindings"]["paired_cluster_uq_source"][
                        "sha256"
                    ]
                    or int(attestation.get("worker_pid", 0)) <= 0
                ):
                    raise RuntimeError("scientific worker source attestation drift")
                records = worker_result.get("records")
                if not isinstance(records, list):
                    raise RuntimeError("scientific worker record payload drift")
                chunk = _chunk_payload(
                    config,
                    cell,
                    records,
                    run_identity,
                    attestation,
                )
                path = _chunk_path(root, config, cell)
                _atomic_text(
                    path,
                    json.dumps(chunk, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                )
                live = json.loads(path.read_text(encoding="utf-8"))
                _validate_chunk(root, config, cell, live, run_identity)
                chunks[cell.cell_id] = live
                _heartbeat(
                    root,
                    config,
                    run_identity,
                    owner,
                    completed=len(chunks),
                    total=len(cells),
                    active=True,
                    state="RUNNING",
                    stage="SCIENCE",
                )
    ordered = [chunks[cell.cell_id] for cell in cells]
    records = [dict(record) for chunk in ordered for record in chunk["records"]]
    bindings = [_binding(_chunk_path(root, config, cell), root) for cell in cells]
    _heartbeat(
        root,
        config,
        run_identity,
        owner,
        completed=len(cells),
        total=len(cells),
        active=True,
        state="RUNNING",
        stage="SCIENCE",
    )
    return records, bindings


def _coverage_cells(
    records: Sequence[Mapping[str, Any]],
    *,
    factor: float,
    margin: float,
    expected_trials: int,
    confidence: float,
    coverage_comparisons: int,
    power_comparisons: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in records:
        key = (
            row["family"],
            int(row["dimension"]),
            int(row["cluster_count"]),
            float(row["true_distance"]),
        )
        grouped.setdefault(key, []).append(row)
    cells: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(rows) != expected_trials:
            raise ValueError(f"hardened UQ cell trial denominator drift: {key}")
        bounds = [
            float(row["estimate"]) + factor * float(row["raw_radius"]) for row in rows
        ]
        truth = float(key[3])
        coverage = sum(bound + 1e-12 >= truth for bound in bounds)
        equivalence = sum(bound <= margin for bound in bounds)
        coverage_lcb, coverage_ucb = _wilson_bounds(
            coverage,
            len(rows),
            confidence=confidence,
            comparisons=coverage_comparisons,
        )
        equivalence_lcb, equivalence_ucb = _wilson_bounds(
            equivalence,
            len(rows),
            confidence=confidence,
            comparisons=power_comparisons,
        )
        cells.append(
            {
                "split": "confirmation",
                "family": key[0],
                "dimension": key[1],
                "cluster_count": key[2],
                "true_distance": truth,
                "trials": len(rows),
                "coverage_count": coverage,
                "coverage_rate": coverage / len(rows),
                "coverage_wilson_lcb": coverage_lcb,
                "coverage_wilson_ucb": coverage_ucb,
                "equivalence_count": equivalence,
                "equivalence_rate": equivalence / len(rows),
                "equivalence_wilson_lcb": equivalence_lcb,
                "equivalence_wilson_ucb": equivalence_ucb,
                "power_primary": bool(rows[0]["power_primary"]),
            }
        )
    return cells


def _parent_seed_sets(
    root: Path, config: Mapping[str, Any]
) -> tuple[set[int], set[int]]:
    trial_seeds: set[int] = set()
    multiplier_seeds: set[int] = set()
    for name in ("calibration_source_data", "extension_source_data"):
        binding = config["parent_artifacts"][name]
        path = _verify_binding(root, binding, label=f"seed parent/{name}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                trial_seeds.add(int(row["trial_seed"]))
                multiplier_seeds.add(int(row["multiplier_seed"]))
    return trial_seeds, multiplier_seeds


def build_report(
    root: Path,
    config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    extension: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    resource_preflight: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    chunk_bindings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _verify_execution_identity_live(root, config, run_identity)
    if len(chunk_bindings) != len(_cells(config)):
        raise ValueError("hardened UQ chunk-binding denominator drift")
    if resource_preflight.get("passed") is not True or not isinstance(
        resource_preflight.get("analysis_sha256"), str
    ):
        raise ValueError("hardened UQ resource preflight is not released")
    factor = float(config["frozen_calibration_factor"])
    margin = float(config["margin"])
    expected_records = len(_cells(config)) * int(
        config["confirmation_split"]["trial_count_per_cell"]
    )
    if len(records) != expected_records:
        raise ValueError("hardened UQ record denominator drift")
    cells = _coverage_cells(
        records,
        factor=factor,
        margin=margin,
        expected_trials=int(config["confirmation_split"]["trial_count_per_cell"]),
        confidence=float(config["simultaneous_wilson"]["confidence"]),
        coverage_comparisons=int(
            config["simultaneous_wilson"]["coverage_cell_comparisons"]
        ),
        power_comparisons=int(
            config["simultaneous_wilson"]["power_stratum_comparisons"]
        ),
    )
    if len(cells) != 96:
        raise ValueError("hardened UQ coverage-cell denominator drift")
    gates = config["gates"]
    coverage_rows: list[dict[str, Any]] = []
    for row in cells:
        passed = float(row["coverage_rate"]) >= float(
            gates["minimum_cell_coverage_rate"]
        ) and float(row["coverage_wilson_lcb"]) >= float(
            gates["minimum_cell_coverage_wilson_lcb"]
        )
        coverage_rows.append({**row, "coverage_gate_passed": passed})
    coverage_by_count: dict[int, bool] = {}
    for count in (12, 384):
        count_rows = [
            row for row in coverage_rows if int(row["cluster_count"]) == count
        ]
        if len(count_rows) != 48:
            raise ValueError(
                f"hardened UQ per-count coverage denominator drift: {count}"
            )
        coverage_by_count[count] = all(
            bool(row["coverage_gate_passed"]) for row in count_rows
        )
    primary = [
        row
        for row in coverage_rows
        if int(row["cluster_count"]) == 384 and bool(row["power_primary"])
    ]
    power_rules = {
        0.0: (
            "lcb",
            float(gates["null_equivalence_wilson_lcb"]),
            lambda row, threshold: float(row["equivalence_wilson_lcb"]) >= threshold,
        ),
        0.05: (
            "lcb",
            float(gates["local_005_equivalence_wilson_lcb"]),
            lambda row, threshold: float(row["equivalence_wilson_lcb"]) >= threshold,
        ),
        0.1: (
            "ucb",
            float(gates["boundary_equivalence_wilson_ucb"]),
            lambda row, threshold: float(row["equivalence_wilson_ucb"]) <= threshold,
        ),
        0.12: (
            "ucb",
            float(gates["outside_equivalence_wilson_ucb"]),
            lambda row, threshold: float(row["equivalence_wilson_ucb"]) <= threshold,
        ),
    }
    power_ledger: list[dict[str, Any]] = []
    for effect, (bound, threshold, predicate) in power_rules.items():
        strata = [
            row for row in primary if abs(float(row["true_distance"]) - effect) < 1e-12
        ]
        if len(strata) != 8:
            raise ValueError(f"hardened UQ power stratum denominator drift: {effect}")
        failures = [
            f"{row['family']}/d{row['dimension']}"
            for row in strata
            if not predicate(row, threshold)
        ]
        power_ledger.append(
            {
                "true_distance": effect,
                "required_bound": bound,
                "threshold": threshold,
                "stratum_count": len(strata),
                "failed_strata": failures,
                "global_iut_pass": not failures,
            }
        )
    power_passed = all(row["global_iut_pass"] for row in power_ledger)

    child_trial_seeds = {int(row["trial_seed"]) for row in records}
    child_multiplier_seeds = {int(row["multiplier_seed"]) for row in records}
    parent_trial_seeds, parent_multiplier_seeds = _parent_seed_sets(root, config)
    seed_firewall_passed = (
        len(child_trial_seeds) == expected_records
        and len(child_multiplier_seeds) == expected_records
        and not child_trial_seeds & child_multiplier_seeds
        and not child_trial_seeds & parent_trial_seeds
        and not child_trial_seeds & parent_multiplier_seeds
        and not child_multiplier_seeds & parent_trial_seeds
        and not child_multiplier_seeds & parent_multiplier_seeds
    )
    if not seed_firewall_passed:
        raise RuntimeError("hardened UQ materialized seed firewall failed")

    passed = coverage_by_count[12] and coverage_by_count[384] and power_passed
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": REPORT_SCHEMA,
        "verdict": PASS_VERDICT if passed else NO_GO_VERDICT,
        "qualified_claim": None,
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "frozen_parent_calibration_factor": factor,
        "frozen_parent_selected_clusters_per_state": 384,
        "selected_formal_clusters_per_state": 384 if passed else None,
        "pilot_domain_factor_coverage_calibrated": coverage_by_count[12],
        "formal_domain_factor_coverage_calibrated": coverage_by_count[384],
        "confirmation_power_passed": power_passed,
        "confirmation_split_role": (
            "independent third split; validates frozen factor/count only"
        ),
        "simultaneous_wilson_contract": dict(config["simultaneous_wilson"]),
        "parent_calibration_analysis_sha256": calibration["analysis_sha256"],
        "parent_extension_analysis_sha256": extension["analysis_sha256"],
        "parent_outcomes_preserved": True,
        "resource_preflight_analysis_sha256": resource_preflight["analysis_sha256"],
        "coverage_summary": {
            "minimum_rate": min(float(row["coverage_rate"]) for row in coverage_rows),
            "minimum_wilson_lcb": min(
                float(row["coverage_wilson_lcb"]) for row in coverage_rows
            ),
            "pilot_count_all_passed": coverage_by_count[12],
            "formal_count_all_passed": coverage_by_count[384],
        },
        "power_ledger": power_ledger,
        "seed_firewall": {
            "passed": seed_firewall_passed,
            "child_trial_seed_count": len(child_trial_seeds),
            "child_multiplier_seed_count": len(child_multiplier_seeds),
            "parent_trial_seed_count": len(parent_trial_seeds),
            "parent_multiplier_seed_count": len(parent_multiplier_seeds),
            "collision_count": 0,
        },
        "domain": {
            "dimensions": list(config["dimensions"]),
            "cluster_counts_per_state": [12, 384],
            "families": sorted(config["families"]),
            "trial_count_per_cell": int(
                config["confirmation_split"]["trial_count_per_cell"]
            ),
            "multiplier_replicates": int(config["multiplier_replicates"]),
            "record_count": len(records),
            "cell_count": len(cells),
        },
        "claim_state": dict(config["claim_boundary"]),
        "formal_outcomes_accessed": False,
        "bindings": dict(run_identity["execution_bindings"]),
        "chunk_bindings": list(chunk_bindings),
    }
    return report, coverage_rows


def _serialize_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = sorted({key for row in rows for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.seek(0)
        return stream.read()


def _verify_finalized_report(
    root: Path,
    config: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    report_path: Path,
    source_path: Path,
) -> None:
    _verify_execution_identity_live(root, config, run_identity)
    live = json.loads(report_path.read_text(encoding="utf-8"))
    _verify_self_hash(live, "hardened UQ report")
    if (
        live != report
        or live.get("run_id") != run_identity["run_id"]
        or live.get("run_identity_analysis_sha256") != run_identity["analysis_sha256"]
        or live.get("claim_state") != CLAIM_BOUNDARY
        or live.get("formal_outcomes_accessed") is not False
        or live.get("qualified_claim") is not None
        or live.get("verdict") not in {PASS_VERDICT, NO_GO_VERDICT}
    ):
        raise RuntimeError("hardened UQ finalized report semantic drift")
    bindings = live.get("bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("hardened UQ finalized bindings missing")
    expected_binding_names = {
        *run_identity["execution_bindings"],
        "run_identity",
        "source_data",
        "resource_preflight",
    }
    if set(bindings) != expected_binding_names:
        raise RuntimeError("hardened UQ finalized binding set drift")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"hardened UQ finalized binding type drift: {name}")
        _verify_binding(root, binding, label=f"final report/{name}")
    if bindings["source_data"] != _binding(source_path, root):
        raise RuntimeError("hardened UQ finalized source-data binding drift")
    expected_chunks = [
        _binding(_chunk_path(root, config, cell), root) for cell in _cells(config)
    ]
    if live.get("chunk_bindings") != expected_chunks:
        raise RuntimeError("hardened UQ finalized chunk live-binding drift")


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    initial_config, _, _ = load_config(base)
    with _owner_lock(base, initial_config) as owner:
        config, calibration, extension = load_config(base)
        if config != initial_config:
            raise RuntimeError("hardened UQ config changed during lock acquisition")
        run_identity = _load_or_create_run_identity(base, config)
        stage = "PRECHECK"
        _heartbeat(
            base,
            config,
            run_identity,
            owner,
            completed=0,
            total=len(_cells(config)),
            active=True,
            state="RUNNING",
            stage=stage,
        )
        try:
            _verify_execution_identity_live(base, config, run_identity)
            stage = "PREFLIGHT"
            _heartbeat(
                base,
                config,
                run_identity,
                owner,
                completed=0,
                total=len(_cells(config)),
                active=True,
                state="RUNNING",
                stage=stage,
            )
            resource_preflight = run_resource_preflight(
                base,
                config,
                run_identity,
                owner,
            )
            stage = "SCIENCE"
            records, chunk_bindings = simulate_confirmation(
                base, config, run_identity, owner
            )
            stage = "FINALIZE"
            _heartbeat(
                base,
                config,
                run_identity,
                owner,
                completed=len(_cells(config)),
                total=len(_cells(config)),
                active=True,
                state="RUNNING",
                stage=stage,
            )
            report, coverage_rows = build_report(
                base,
                config,
                calibration,
                extension,
                run_identity,
                resource_preflight,
                records,
                chunk_bindings,
            )
            source_path = base / str(config["artifact_paths"]["source_data"])
            report_path = base / str(config["artifact_paths"]["report"])
            _atomic_text(source_path, _serialize_csv(records))
            report["bindings"]["source_data"] = _binding(source_path, base)
            report["bindings"]["run_identity"] = _binding(
                base / str(config["artifact_paths"]["run_identity"]), base
            )
            report["bindings"]["resource_preflight"] = _binding(
                base / str(config["resource_preflight"]["artifact"]), base
            )
            report["transaction"] = {
                "write_order": ["chunks", "source_data", "report"],
                "source_committed_before_report": True,
                "coverage_cell_count": len(coverage_rows),
            }
            report["analysis_sha256"] = _sha(report)
            _atomic_text(
                report_path,
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            _verify_finalized_report(
                base,
                config,
                run_identity,
                report,
                report_path=report_path,
                source_path=source_path,
            )
            stage = "COMPLETE"
            _heartbeat(
                base,
                config,
                run_identity,
                owner,
                completed=len(_cells(config)),
                total=len(_cells(config)),
                active=False,
                state="PASS" if report["verdict"] == PASS_VERDICT else "NO_GO",
                stage=stage,
            )
        except BaseException as exc:
            if "report_path" in locals():
                report_path.unlink(missing_ok=True)
            chunk_directory = base / str(config["artifact_paths"]["chunk_directory"])
            completed = (
                len(list(chunk_directory.glob("*.json")))
                if chunk_directory.exists()
                else 0
            )
            _heartbeat(
                base,
                config,
                run_identity,
                owner,
                completed=completed,
                total=len(_cells(config)),
                active=False,
                state="FAILED",
                stage=stage,
                error_type=type(exc).__name__,
            )
            raise
        return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run hardened third-split paired-cluster UQ confirmation."
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run only the disjoint-seed ETA/RSS resource preflight.",
    )
    arguments = parser.parse_args(argv)
    if arguments.preflight_only:
        base = _root().resolve()
        initial_config, _, _ = load_config(base)
        with _owner_lock(base, initial_config) as owner:
            config, _, _ = load_config(base)
            if config != initial_config:
                raise RuntimeError(
                    "hardened UQ config changed during preflight lock acquisition"
                )
            run_identity = _load_or_create_run_identity(base, config)
            _heartbeat(
                base,
                config,
                run_identity,
                owner,
                completed=0,
                total=len(_cells(config)),
                active=True,
                state="RUNNING",
                stage="PREFLIGHT",
            )
            try:
                report = run_resource_preflight(
                    base,
                    config,
                    run_identity,
                    owner,
                )
            except BaseException as exc:
                _heartbeat(
                    base,
                    config,
                    run_identity,
                    owner,
                    completed=0,
                    total=len(_cells(config)),
                    active=False,
                    state="FAILED",
                    stage="PREFLIGHT",
                    error_type=type(exc).__name__,
                )
                raise
            _heartbeat(
                base,
                config,
                run_identity,
                owner,
                completed=0,
                total=len(_cells(config)),
                active=False,
                state="PREFLIGHT_PASS",
                stage="PREFLIGHT",
            )
        print(
            json.dumps(
                {
                    "analysis_sha256": report["analysis_sha256"],
                    "passed": report["passed"],
                    "estimated_wall_seconds_with_safety_factor": report[
                        "estimated_wall_seconds_with_safety_factor"
                    ],
                    "estimated_worker_pool_rss_bytes": report[
                        "estimated_worker_pool_rss_bytes"
                    ],
                    "estimated_total_task_rss_bytes": report[
                        "estimated_total_task_rss_bytes"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    report = write_artifacts()
    print(
        json.dumps(
            {
                "analysis_sha256": report["analysis_sha256"],
                "verdict": report["verdict"],
                "selected_formal_clusters_per_state": report[
                    "selected_formal_clusters_per_state"
                ],
                "pilot_domain_factor_coverage_calibrated": report[
                    "pilot_domain_factor_coverage_calibrated"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CellSpec",
    "NO_GO_VERDICT",
    "PASS_VERDICT",
    "build_report",
    "load_config",
    "paired_density_trace_ucb_physical",
    "simulate_confirmation",
    "write_artifacts",
]
