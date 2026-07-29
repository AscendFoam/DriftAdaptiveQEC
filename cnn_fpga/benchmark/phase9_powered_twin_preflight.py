"""Fail-closed resource preflight for the T04 powered-twin transaction.

The preflight deliberately has a different run, seed and artifact namespace
from formal evidence.  It executes the five full-denominator resource profile
cells frozen in the T04 config, measures four-worker concurrency continuously,
exercises a 3,037-by-199 streaming statistics kernel, and inventories the
content-addressed worker objects without copying them into an archive.

This module owns no scientific verdict.  Every claim field in its report is
literal ``null`` and none of its receipts are accepted by the formal run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import sys
from threading import Event, Lock, Thread
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import uuid4

import psutil

from cnn_fpga.benchmark.phase9_immutable_object_store import (
    ImmutableObjectStore,
    append_attempt_event,
)
from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    CONFIG_PATH,
    EXPECTED_CLAIM_FIELDS,
    T04CellSpec,
    build_cell_plan,
    plan_payload,
    runtime_source_snapshot,
    validate_config,
)
from cnn_fpga.benchmark.phase9_powered_twin_runtime import (
    HeartbeatService,
    OwnerLease,
)


PREFLIGHT_SCHEMA = "PHASE9-POWERED-TWIN-RESOURCE-PREFLIGHT-V1"
SAMPLING_SCHEMA = "PHASE9-POWERED-TWIN-RESOURCE-SAMPLING-V1"
PROJECTION_SCHEMA = "PHASE9-POWERED-TWIN-STRATIFIED-PROJECTION-V1"
STATS_DRY_RUN_SCHEMA = "PHASE9-POWERED-TWIN-STATS-DRY-RUN-V1"
RUNNER_ID = "phase9_powered_twin_resource_preflight_v1"
PASS_VERDICT = "PASS_RESOURCE_PREFLIGHT"
FAIL_VERDICT = "INCOMPLETE_RESOURCE_FAIL_CLOSED"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete bytes with atomic fail-if-exists semantics.

    A fully written and fsynced same-directory temporary file is hard-linked
    to the destination.  ``os.link`` cannot replace an existing name, closing
    the exists-then-replace race that would let a competing writer overwrite
    immutable evidence.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"immutable preflight target is not a regular file: {path}"
                )
            existing = path.read_bytes()
            if existing != payload:
                raise RuntimeError(
                    f"conflicting immutable preflight report: {path}"
                )
        if path.read_bytes() != payload:
            raise RuntimeError("immutable preflight publication recheck failed")
    finally:
        if temporary.exists():
            temporary.unlink()


def _inside(path: Path, root: Path, name: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} escapes repository") from exc
    return resolved


def _claims_null() -> dict[str, None]:
    return {name: None for name in EXPECTED_CLAIM_FIELDS}


def validate_preflight_lineage(
    root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    plan_sha256: str,
    source_snapshot_sha256: str,
) -> dict[str, Any]:
    """Recompute every release binding at supervisor entry."""

    for name, value in (
        ("config_sha256", config_sha256),
        ("plan_sha256", plan_sha256),
        ("source_snapshot_sha256", source_snapshot_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value == "0" * 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"invalid {name}")
    config_path = _inside(root / CONFIG_PATH, root, "T04 config")
    live_config_sha = _sha_file(config_path)[1]
    if live_config_sha != config_sha256:
        raise RuntimeError("resource preflight config binding drift")
    live_plan_sha = str(plan_payload(config)["canonical_plan_sha256"])
    if (
        live_plan_sha != plan_sha256
        or str(config["plan_contract"]["canonical_plan_sha256"]) != plan_sha256
    ):
        raise RuntimeError("resource preflight plan binding drift")
    snapshot = runtime_source_snapshot(root, config)
    if snapshot["source_snapshot_sha256"] != source_snapshot_sha256:
        raise RuntimeError("resource preflight source binding drift")
    claim_boundary = config.get("claim_boundary")
    if (
        not isinstance(claim_boundary, Mapping)
        or set(claim_boundary) != set(EXPECTED_CLAIM_FIELDS)
        or any(value is not None for value in claim_boundary.values())
    ):
        raise RuntimeError("resource preflight claim boundary is not literal-null")
    evidence: dict[str, Any] = {
        "config_sha256": config_sha256,
        "plan_sha256": plan_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "runtime_source_count": snapshot["runtime_source_count"],
        "validation_source_count": snapshot["validation_source_count"],
        "claim_boundary_all_literal_null": True,
        "passed": True,
    }
    evidence["analysis_sha256"] = _sha(evidence)
    return evidence


def _record_attempt(
    path: Path,
    *,
    task_id: str,
    run_id: str,
    event: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the hash-chain and an immutable per-sequence event witness."""

    record = append_attempt_event(
        path,
        task_id=task_id,
        run_id=run_id,
        event=event,
        payload=payload,
    )
    event_path = (
        path.parent
        / "attempt_events"
        / f"{int(record['sequence']):08d}.json"
    )
    _immutable_json(event_path, record)
    return record


def _json_binding(root: Path, path: Path) -> dict[str, Any]:
    size, digest = _sha_file(path)
    return {
        "path": _relative(root, path),
        "bytes": size,
        "sha256": digest,
    }


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _disjoint(a: Path, b: Path) -> bool:
    a = a.resolve()
    b = b.resolve()
    return a != b and a not in b.parents and b not in a.parents


def isolated_preflight_paths(
    root: Path,
    config: Mapping[str, Any],
    *,
    run_id: str,
) -> tuple[Path, dict[str, str]]:
    """Return a sibling preflight namespace and prove it cannot touch formal raw data."""

    if not run_id or any(character not in "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for character in run_id):
        raise ValueError("run_id contains unsafe characters")
    preflight_root = _inside(
        root / "runs" / f"t04_resource_preflight_{run_id}",
        root,
        "preflight root",
    )
    paths = {
        "object_store": _relative(root, preflight_root / "objects" / "sha256"),
        "staging_directory": _relative(root, preflight_root / "staging"),
        "receipt_directory": _relative(root, preflight_root / "receipts"),
    }
    formal = config["artifact_paths"]
    for key, relative in paths.items():
        candidate = root / relative
        for formal_key in ("object_store", "staging_directory", "receipt_directory"):
            if not _disjoint(candidate, root / str(formal[formal_key])):
                raise RuntimeError(
                    f"preflight {key} overlaps formal {formal_key}"
                )
    return preflight_root, paths


def assert_seed_firewall(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the resource interval is disjoint from every formal seed interval."""

    registry = config["seed_registry"]
    resource = registry["resource_preflight"]
    resource_interval = (
        int(resource["start"]),
        int(resource["start"]) + int(resource["count"]),
    )
    formal_names = ("physical", "heldout", "joint_maxt_rademacher")
    formal_intervals: dict[str, list[int]] = {}
    for name in formal_names:
        entry = registry[name]
        interval = (int(entry["start"]), int(entry["start"]) + int(entry["count"]))
        formal_intervals[name] = [interval[0], interval[1]]
        if max(resource_interval[0], interval[0]) < min(
            resource_interval[1], interval[1]
        ):
            raise RuntimeError(f"resource seed namespace overlaps formal {name}")
    for offset_name in ("physical_offset", "heldout_offset"):
        address = resource_interval[0] + int(resource[offset_name])
        if not resource_interval[0] <= address < resource_interval[1]:
            raise RuntimeError(f"resource {offset_name} escapes its interval")
    maximum_positions = int(registry["maximum_cluster_positions"])
    maximum_horizon = int(registry["maximum_horizon"])
    pair_groups = int(registry["pair_group_count"])
    maximum_resource_addresses = {
        "physical": (
            resource_interval[0]
            + int(resource["physical_offset"])
            + 2 * pair_groups * maximum_positions
            - 1
        ),
        "heldout": (
            resource_interval[0]
            + int(resource["heldout_offset"])
            + pair_groups * maximum_positions * maximum_horizon
            - 1
        ),
    }
    if any(
        not resource_interval[0] <= address < resource_interval[1]
        for address in maximum_resource_addresses.values()
    ):
        raise RuntimeError("full preflight seed address range escapes allocation")
    return {
        "resource_interval_half_open": list(resource_interval),
        "formal_intervals_half_open": formal_intervals,
        "formal_seed_addresses_accessed": False,
        "maximum_resource_addresses": maximum_resource_addresses,
        "seed_namespace_pass": True,
    }


def profile_cells(
    config: Mapping[str, Any],
    cells: Sequence[T04CellSpec],
) -> tuple[list[T04CellSpec], T04CellSpec]:
    profile = config["resource_contract"]["profile_plan"]
    concurrent_profile = profile["four_worker_concurrent_peak"]
    singleton_profile = profile["backend_a_representative"]
    if (
        concurrent_profile.get("full_frozen_denominator") is not True
        or singleton_profile.get("full_frozen_denominator") is not True
    ):
        raise RuntimeError("resource profiles must use the full frozen denominator")
    concurrent_indices = list(concurrent_profile["plan_indices"])
    singleton_indices = list(singleton_profile["plan_indices"])
    if len(concurrent_indices) != 4 or len(singleton_indices) != 1:
        raise RuntimeError("resource profile must contain exactly four concurrent and one A cell")
    indices = concurrent_indices + singleton_indices
    if len(indices) != len(set(indices)) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(cells)
        for index in indices
    ):
        raise RuntimeError("invalid or duplicate resource profile plan index")
    selected = [cells[index] for index in concurrent_indices]
    singleton = cells[singleton_indices[0]]
    expected_counts = list(concurrent_profile["sample_counts"])
    if [cell.sample_count for cell in selected] != expected_counts:
        raise RuntimeError("four-worker profile denominator drift")
    if [singleton.sample_count] != list(
        singleton_profile["sample_counts"]
    ):
        raise RuntimeError("backend-A profile denominator drift")
    if [cell.plan_index for cell in selected] != [389, 403, 507, 485]:
        raise RuntimeError("frozen four-worker profile identity drift")
    if singleton.plan_index != 388 or singleton.backend != "A":
        raise RuntimeError("frozen backend-A profile identity drift")
    return selected, singleton


class ResourcePreflightFailure(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(FAIL_VERDICT)


class ResourceSampler:
    """Continuously sample parent plus live child RSS; never endpoint-only."""

    def __init__(
        self,
        *,
        evidence_path: Path,
        child_pids: Callable[[], Sequence[int]],
        stage: Callable[[], str] | None = None,
        interval_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        self.evidence_path = evidence_path.resolve()
        self.child_pids = child_pids
        self.stage = stage or (lambda: "unspecified")
        self.interval_seconds = float(interval_seconds)
        self.clock = clock
        self.started = clock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._lock = Lock()
        self._sample_serial = Lock()
        self._count = 0
        self._active_count = 0
        self._peak_rss = 0
        self._peak_children = 0
        self._stage_peaks: dict[str, int] = {}
        self._first: dict[str, Any] | None = None
        self._last: dict[str, Any] | None = None
        self._peak: dict[str, Any] | None = None
        self._chain = "0" * 64
        self.error: BaseException | None = None

    @staticmethod
    def _rss(pid: int) -> int:
        try:
            return int(psutil.Process(pid).memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return 0

    def sample_once(self) -> dict[str, Any]:
        with self._sample_serial:
            return self._sample_once_serial()

    def _sample_once_serial(self) -> dict[str, Any]:
        pids = sorted({int(pid) for pid in self.child_pids() if int(pid) > 0})
        parent_rss = self._rss(os.getpid())
        seen = {os.getpid()}
        child_rss: dict[str, int] = {}
        child_process_tree_pids: dict[str, list[int]] = {}
        for pid in pids:
            try:
                process = psutil.Process(pid)
                tree = [process, *process.children(recursive=True)]
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            tree_pids: list[int] = []
            tree_rss = 0
            for member in tree:
                if member.pid in seen:
                    continue
                rss = self._rss(member.pid)
                if rss <= 0:
                    continue
                seen.add(member.pid)
                tree_pids.append(member.pid)
                tree_rss += rss
            if tree_rss:
                child_rss[str(pid)] = tree_rss
                child_process_tree_pids[str(pid)] = sorted(tree_pids)
        record: dict[str, Any] = {
            "schema_version": SAMPLING_SCHEMA,
            "sequence": self._count,
            "monotonic_seconds": float(self.clock() - self.started),
            "parent_pid": os.getpid(),
            "parent_rss_bytes": parent_rss,
            "child_rss_bytes": child_rss,
            "child_process_tree_pids": child_process_tree_pids,
            "live_child_count": len(child_rss),
            "aggregate_rss_bytes": parent_rss + sum(child_rss.values()),
            "stage": str(self.stage()),
            "previous_sample_sha256": self._chain,
        }
        record["sample_sha256"] = _sha(record)
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with self.evidence_path.open("ab") as handle:
            handle.write(_canonical(record) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        with self._lock:
            self._count += 1
            if child_rss:
                self._active_count += 1
            self._peak_children = max(self._peak_children, len(child_rss))
            sample_stage = str(record["stage"])
            self._stage_peaks[sample_stage] = max(
                self._stage_peaks.get(sample_stage, 0),
                int(record["aggregate_rss_bytes"]),
            )
            if record["aggregate_rss_bytes"] >= self._peak_rss:
                self._peak_rss = int(record["aggregate_rss_bytes"])
                self._peak = dict(record)
            if self._first is None:
                self._first = dict(record)
            self._last = dict(record)
            self._chain = str(record["sample_sha256"])
        return record

    def _run(self) -> None:
        try:
            self.sample_once()
            while not self._stop.wait(self.interval_seconds):
                self.sample_once()
        except BaseException as exc:
            self.error = exc
            self._stop.set()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource sampler already started")
        if self.evidence_path.exists():
            raise RuntimeError(
                "resource sampling evidence already exists; fresh run_id required"
            )
        self._thread = Thread(target=self._run, name="t04-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, 2.0 * self.interval_seconds))
            if self._thread.is_alive():
                raise RuntimeError("resource sampler did not stop")
        if self.error is not None:
            raise RuntimeError("resource sampler failed") from self.error

    def summary(self) -> dict[str, Any]:
        with self._sample_serial:
            with self._lock:
                summary: dict[str, Any] = {
                    "schema_version": SAMPLING_SCHEMA,
                    "sample_count": self._count,
                    "active_child_sample_count": self._active_count,
                    "peak_aggregate_rss_bytes": self._peak_rss,
                    "maximum_observed_live_children": self._peak_children,
                    "stage_peak_aggregate_rss_bytes": dict(self._stage_peaks),
                    "first_sample": self._first,
                    "last_sample": self._last,
                    "peak_sample": self._peak,
                    "sample_chain_tip_sha256": self._chain,
                }
            if self.evidence_path.exists():
                size, digest = _sha_file(self.evidence_path)
                summary["evidence"] = {
                    "path": self.evidence_path.as_posix(),
                    "bytes": size,
                    "sha256": digest,
                }
        summary["summary_sha256"] = _sha(summary)
        return summary


def validate_continuous_sampling(
    summary: Mapping[str, Any],
    *,
    required_concurrency: int = 4,
) -> None:
    if int(summary.get("sample_count", 0)) < 3:
        raise RuntimeError("endpoint-only RSS evidence: fewer than three samples")
    if int(summary.get("active_child_sample_count", 0)) < 2:
        raise RuntimeError("endpoint-only RSS evidence: fewer than two active samples")
    if int(summary.get("maximum_observed_live_children", 0)) < required_concurrency:
        raise RuntimeError("four-worker concurrency was not actually observed")
    first = summary.get("first_sample")
    last = summary.get("last_sample")
    if not isinstance(first, Mapping) or not isinstance(last, Mapping):
        raise RuntimeError("RSS sampling endpoints missing")
    if float(last["monotonic_seconds"]) <= float(first["monotonic_seconds"]):
        raise RuntimeError("RSS sampling has no positive observation span")


def _worker_entry(result_queue: Any, kwargs: Mapping[str, Any]) -> None:
    started = time.monotonic()
    try:
        from cnn_fpga.benchmark.phase9_powered_twin_qualification import (
            execute_cell_to_store,
        )

        receipt = execute_cell_to_store(**dict(kwargs))
        result_queue.put(
            {
                "ok": True,
                "pid": os.getpid(),
                "wall_seconds": time.monotonic() - started,
                "receipt": receipt,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "pid": os.getpid(),
                "wall_seconds": time.monotonic() - started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        raise


def _receipt_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    receipt = result["receipt"]
    objects = receipt["objects"]
    unique = {str(binding["sha256"]): int(binding["bytes"]) for binding in objects}
    return {
        "chunk_id": receipt["cell"]["chunk_id"],
        "plan_index": int(receipt["cell"]["plan_index"]),
        "pid": int(result["pid"]),
        "wall_seconds": float(result["wall_seconds"]),
        "object_bytes_unique": sum(unique.values()),
        "object_bytes_by_role": {
            str(binding["role"]): int(binding["bytes"]) for binding in objects
        },
        "object_bindings": [
            {
                "role": str(binding["role"]),
                "bytes": int(binding["bytes"]),
                "sha256": str(binding["sha256"]),
            }
            for binding in objects
        ],
        "expected_rows": int(receipt["diagnostics"]["expected_rows"]),
        "reset_rows": int(receipt["diagnostics"]["reset_rows"]),
        "receipt_sha256": receipt["receipt_sha256"],
    }


def execute_process_group(
    worker_kwargs: Sequence[Mapping[str, Any]],
    *,
    active_pids: set[int],
    active_lock: Lock,
    start_method: str = "spawn",
    sample_callback: Callable[[], object] | None = None,
) -> list[dict[str, Any]]:
    """Execute one process group and surface every child failure."""

    context = mp.get_context(start_method)
    result_queue = context.Queue()
    processes: list[mp.Process] = []
    try:
        for kwargs in worker_kwargs:
            process = context.Process(
                target=_worker_entry,
                args=(result_queue, dict(kwargs)),
                name=f"t04-preflight-{kwargs['cell'].plan_index}",
            )
            process.start()
            processes.append(process)
            with active_lock:
                if process.pid is not None:
                    active_pids.add(int(process.pid))
        if sample_callback is not None:
            sample_callback()
        results: list[dict[str, Any]] = []
        while any(process.is_alive() for process in processes):
            try:
                results.append(result_queue.get(timeout=0.1))
            except queue.Empty:
                pass
        for process in processes:
            process.join()
        result_deadline = time.monotonic() + 5.0
        while len(results) < len(processes) and time.monotonic() < result_deadline:
            try:
                results.append(result_queue.get(timeout=0.1))
            except queue.Empty:
                pass
        failures = [
            {
                "pid": process.pid,
                "exitcode": process.exitcode,
            }
            for process in processes
            if process.exitcode != 0
        ]
        failures.extend(result for result in results if not result.get("ok"))
        if failures or len(results) != len(processes):
            raise RuntimeError(
                "resource profile worker failure: "
                + json.dumps(failures, sort_keys=True)
            )
        return sorted(
            (_receipt_metrics(result) for result in results),
            key=lambda value: value["plan_index"],
        )
    finally:
        with active_lock:
            for process in processes:
                if process.pid is not None:
                    active_pids.discard(int(process.pid))
        result_queue.close()
        result_queue.join_thread()


def streaming_statistics_dry_run(
    config: Mapping[str, Any],
    *,
    gate_kernel: Callable[[int, int, int], float] | None = None,
    sample_callback: Callable[[], object] | None = None,
    sign_matrix_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Exercise the frozen 3,037 x 199 streaming shape without scientific data.

    The scalar sweep alone is not a sufficient resource witness.  The
    independent verifier's largest density gate holds a ``199 x 132 x 132``
    complex perturbation accumulator while a same-sized matrix-product result
    is live.  We therefore execute one full 4,608-root density multiplier
    kernel, including all 199 trace-norm finalizations, before sweeping the
    remaining gate identities.  Values are deterministic resource-only
    fixtures and never enter a scientific decision.
    """

    import numpy as np

    specification = config["resource_contract"]["profile_plan"]["joint_maxt_3037x199"]
    gates = int(specification["gate_count"])
    replicates = int(specification["replicates"])
    clusters = int(specification["largest_cluster_count"])
    dimension = int(specification["largest_density_dimension"])
    if gates != 3037 or replicates != 199:
        raise RuntimeError("streaming statistics dry-run shape drift")
    namespace = config["seed_registry"]["resource_preflight"]
    seed = int(namespace["start"]) + int(namespace["count"]) - 1
    resource_start = int(namespace["start"])
    resource_stop = resource_start + int(namespace["count"])
    for name, candidate in config["seed_registry"].items():
        if name == "resource_preflight" or not isinstance(candidate, Mapping):
            continue
        if "start" not in candidate or "count" not in candidate:
            continue
        candidate_start = int(candidate["start"])
        candidate_stop = candidate_start + int(candidate["count"])
        if max(resource_start, candidate_start) < min(
            resource_stop,
            candidate_stop,
        ):
            raise RuntimeError(
                "resource statistics seed namespace overlaps "
                f"{name}"
            )
    physicality_profile = _retained_density_physicality_dry_run(
        config,
        resource_seed=seed,
        sample_callback=sample_callback,
    )
    started = time.monotonic()
    maxima = np.full(replicates, -np.inf, dtype=np.float64)
    # One largest-shape work buffer is reused; no gate tensor family coexists.
    work = np.empty((clusters, dimension), dtype=np.float64)
    production_generator = sign_matrix_factory is None
    if production_generator:
        from cnn_fpga.benchmark.phase9_powered_twin_statistics import (
            rademacher_matrix,
        )

        sign_matrix_factory = rademacher_matrix
    group_sizes = [1536] * 93 + [4608] * 4
    sign_cache = []
    for group_index, group_size in enumerate(group_sizes):
        roots = [
            f"resource/group={group_index:02d}/cluster={index:04d}"
            for index in range(group_size)
        ]
        signs = np.asarray(
            sign_matrix_factory(
                seed=seed,
                replicates=replicates,
                cluster_root_ids=roots,
            ),
            dtype=np.int8,
        )
        if signs.shape != (replicates, group_size) or not np.all(
            (signs == -1) | (signs == 1)
        ):
            raise RuntimeError("statistics sign-matrix factory shape/value drift")
        sign_cache.append(signs)
    l1_maxima = np.full(replicates, -np.inf, dtype=np.float64)
    checksum = sha256()
    base_replicates: list[tuple[np.ndarray, np.ndarray]] = []
    for group_index, signs in enumerate(sign_cache):
        influence_a = (
            ((np.arange(signs.shape[1], dtype=np.float64) + group_index) % 19.0)
            - 9.0
        ) / 19.0
        influence_b = (
            (
                (
                    np.arange(signs.shape[1], dtype=np.float64)
                    + 3 * group_index
                    + 1
                )
                % 23.0
            )
            - 11.0
        ) / 23.0
        base_replicates.append(
            (
                signs @ influence_a / np.sqrt(float(signs.shape[1])),
                signs @ influence_b / np.sqrt(float(signs.shape[1])),
            )
        )
    persistent_components = {
        "gate_work": int(work.nbytes),
        "maxima": int(maxima.nbytes),
        "l1_maxima": int(l1_maxima.nbytes),
        "sign_cache": int(sum(signs.nbytes for signs in sign_cache)),
        "base_replicates": int(
            sum(a.nbytes + b.nbytes for a, b in base_replicates)
        ),
    }
    persistent_working_set = sum(persistent_components.values())
    # Exercise the actual worst-shape density path used by
    # ``phase9_powered_twin_verifier.evaluate_material``.  Keeping this local
    # avoids importing the verifier (and therefore prevents a resource
    # preflight from becoming an alternate scientific evaluator).
    density_roots = group_sizes[-1]
    density_signs = sign_cache[-1]
    density_block_rows = 32
    density_total = np.zeros((dimension, dimension), dtype=np.complex128)

    def density_block(start: int, end: int) -> np.ndarray:
        count = end - start
        row_ids = np.arange(start, end, dtype=np.float64)
        result = np.zeros(
            (count, dimension, dimension),
            dtype=np.complex128,
        )
        diagonal = np.arange(dimension)
        result[:, diagonal, diagonal] = (
            (
                (row_ids[:, None] + 3.0)
                * (diagonal[None, :] + 5.0)
            )
            % 257.0
        ) / (257.0 * dimension)
        if dimension > 1:
            adjacent = np.arange(dimension - 1)
            real = (
                (
                    row_ids[:, None]
                    + 2.0 * adjacent[None, :]
                    + 1.0
                )
                % 29.0
                - 14.0
            ) / (4096.0 * dimension)
            imag = (
                (
                    3.0 * row_ids[:, None]
                    + adjacent[None, :]
                    + 2.0
                )
                % 31.0
                - 15.0
            ) / (4096.0 * dimension)
            off_diagonal = real + 1j * imag
            result[:, adjacent, adjacent + 1] = off_diagonal
            result[:, adjacent + 1, adjacent] = np.conjugate(off_diagonal)
        return result

    for start in range(0, density_roots, density_block_rows):
        source_left = density_block(
            start,
            min(density_roots, start + density_block_rows),
        )
        # The verifier's density material generator keeps both source
        # archives live while yielding their difference.  A single synthetic
        # block would understate the callback-time RSS by two full blocks.
        source_right = np.zeros_like(source_left)
        block = source_left - source_right
        density_total += np.sum(block, axis=0, dtype=np.complex128)
    del source_left, source_right
    density_mean = density_total / float(density_roots)
    density_perturbation = np.zeros(
        (replicates, dimension, dimension),
        dtype=np.complex128,
    )
    density_kernel_peak = int(
        density_perturbation.nbytes
        + density_mean.nbytes
        + density_total.nbytes
    )
    density_kernel_peak_components: dict[str, int] = {}
    density_block_count = 0
    density_rss_callback_count = 0
    for start in range(0, density_roots, density_block_rows):
        end = min(density_roots, start + density_block_rows)
        source_left = density_block(start, end)
        source_right = np.zeros_like(source_left)
        block = source_left - source_right
        centered = (block - density_mean).reshape(end - start, -1)
        signed = density_signs[:, start:end].astype(
            np.float64,
            copy=True,
        )
        update = (
            signed @ centered / float(density_roots)
        ).reshape(replicates, dimension, dimension)
        if (
            update.shape != density_perturbation.shape
            or update.dtype != np.dtype(np.complex128)
        ):
            raise RuntimeError("largest density update shape/dtype drift")
        live_components = {
            "perturbation": int(density_perturbation.nbytes),
            "update": int(update.nbytes),
            "density_mean": int(density_mean.nbytes),
            "density_total": int(density_total.nbytes),
            "source_left": int(source_left.nbytes),
            "source_right": int(source_right.nbytes),
            "material_block": int(block.nbytes),
            "centered": int(centered.nbytes),
            "signed": int(signed.nbytes),
        }
        live_total = sum(live_components.values())
        if live_total > density_kernel_peak:
            density_kernel_peak = live_total
            density_kernel_peak_components = live_components
        density_block_count += 1
        if sample_callback is not None:
            # Sample while both the accumulator and the matrix-product result
            # are live, together with the verifier's two source blocks,
            # yielded material block and centered block.
            sample_callback()
            density_rss_callback_count += 1
        density_perturbation += update
    del source_left, source_right
    density_functionals = np.empty(replicates, dtype=np.float64)
    density_trace_norm_evaluations = 0
    for replicate in range(replicates):
        candidate = density_mean + density_perturbation[replicate]
        hermitian = (candidate + candidate.conj().T) / 2.0
        density_functionals[replicate] = (
            np.sum(np.abs(np.linalg.eigvalsh(hermitian))) / 2.0
        )
        density_trace_norm_evaluations += 1
    if not np.all(np.isfinite(density_functionals)):
        raise RuntimeError("largest density statistics kernel became non-finite")
    density_kernel_sha256 = sha256(
        np.asarray(density_functionals, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    peak_working_set = int(
        persistent_working_set + density_kernel_peak
    )
    if sample_callback is not None:
        sample_callback()
    for gate_index in range(gates):
        if gate_index == 0:
            rows = np.arange(clusters, dtype=np.float64)[:, None]
            columns = np.arange(dimension, dtype=np.float64)[None, :]
            work[:] = ((rows + 1.0) * (columns + 3.0)) % 257.0
            work *= 1.0 / 257.0
            checksum.update(
                np.asarray(
                    [work[0, 0], work[-1, -1], float(work.mean())],
                    dtype="<f8",
                ).tobytes()
            )
        if gate_kernel is None:
            leg_a, leg_b = base_replicates[
                gate_index % len(base_replicates)
            ]
            scale_a = 1.0 + (gate_index % 17) * 1.0e-4
            scale_b = 1.0 + (gate_index % 13) * 1.0e-4
            absolute_a = np.abs(leg_a * scale_a)
            absolute_b = np.abs(leg_b * scale_b)
            values = np.maximum(absolute_a, absolute_b)
            l1_values = absolute_a + absolute_b
            np.maximum(l1_maxima, l1_values, out=l1_maxima)
            checksum.update(
                np.asarray(l1_values, dtype="<f8").tobytes(order="C")
            )
        else:
            values = np.asarray(
                [
                    gate_kernel(gate_index, replicate, seed)
                    for replicate in range(replicates)
                ],
                dtype=np.float64,
            )
            l1_values = np.abs(values)
            np.maximum(l1_maxima, l1_values, out=l1_maxima)
        if values.shape != (replicates,) or not np.all(np.isfinite(values)):
            raise RuntimeError("statistics dry-run kernel returned invalid values")
        np.maximum(maxima, values, out=maxima)
        checksum.update(np.asarray(values, dtype="<f8").tobytes(order="C"))
    wall = time.monotonic() - started
    report: dict[str, Any] = {
        "schema_version": STATS_DRY_RUN_SCHEMA,
        "gate_count": gates,
        "replicates": replicates,
        "largest_cluster_count": clusters,
        "largest_density_dimension": dimension,
        "streaming": True,
        "maximum_coexisting_gate_buffers": 1,
        "cached_cluster_root_groups": len(sign_cache),
        "cached_sign_bytes": sum(signs.nbytes for signs in sign_cache),
        "production_rademacher_generator_exercised": production_generator,
        "conservative_dual_leg_max_exercised": gate_kernel is None,
        "dual_leg_evaluation_count": 2 * gates if gate_kernel is None else 0,
        "l1_accumulation_exercised": True,
        "largest_density_kernel_exercised": True,
        "largest_density_root_count": density_roots,
        "largest_density_block_rows": density_block_rows,
        "largest_density_block_count": density_block_count,
        "largest_density_source_buffer_count": 2,
        "largest_density_rss_callback_count": (
            density_rss_callback_count
        ),
        "largest_density_perturbation_shape": [
            replicates,
            dimension,
            dimension,
        ],
        "largest_density_perturbation_bytes": int(
            density_perturbation.nbytes
        ),
        "largest_density_update_bytes": int(
            replicates * dimension * dimension
            * np.dtype(np.complex128).itemsize
        ),
        "largest_density_trace_norm_evaluations": (
            density_trace_norm_evaluations
        ),
        "largest_density_kernel_sha256": density_kernel_sha256,
        "persistent_working_set_components": persistent_components,
        "persistent_working_set_bytes": persistent_working_set,
        "largest_density_peak_live_components": (
            density_kernel_peak_components
        ),
        "largest_density_peak_live_bytes": density_kernel_peak,
        "l1_maxima_sha256": sha256(
            np.asarray(l1_maxima, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "peak_explicit_working_set_bytes": peak_working_set,
        "peak_analysis_scratch_bytes": max(
            peak_working_set,
            int(physicality_profile["peak_explicit_live_bytes"]),
        ),
        "wall_seconds": wall,
        "kernel_trace_sha256": checksum.hexdigest(),
        "retained_density_physicality_profile": physicality_profile,
        "seed_namespace": "resource_preflight",
        "seed_address": seed,
        "formal_seed_addresses_accessed": False,
        "scientific_influences_used": False,
        "scientific_verdict": None,
        "qualified_claim": None,
    }
    report["analysis_sha256"] = _sha(report)
    return report


def _retained_density_physicality_dry_run(
    config: Mapping[str, Any],
    *,
    resource_seed: int,
    sample_callback: Callable[[], object] | None,
) -> dict[str, Any]:
    """Measure the verifier's full retained-density audit without raw data."""

    import numpy as np

    dimension = int(
        config["resource_contract"]["profile_plan"][
            "joint_maxt_3037x199"
        ]["largest_density_dimension"]
    )
    specification = config["resource_contract"]["profile_plan"][
        "retained_density_physicality_full_482304"
    ]
    full_count = int(specification["full_retained_count"])
    block_size = int(specification["block_size"])
    fixture_count = int(specification["fixture_matrix_count"])
    timed_repeats = int(specification["timed_repeats"])
    if dimension != 132 or full_count != 482_304:
        raise RuntimeError("retained density resource profile shape drift")
    if (
        int(specification["largest_dimension"]) != dimension
        or specification["full_coverage_required"] is not True
        or specification["sampled"] is not False
        or full_count
        != int(config["plan_contract"]["primary_density_count"])
        or block_size != 8
        or fixture_count != 256
        or timed_repeats != 3
    ):
        raise RuntimeError(
            "retained density frozen resource specification drift"
        )
    if fixture_count % block_size:
        raise RuntimeError("retained density fixture does not close blocks")

    # Strictly diagonally dominant Hermitian complex64 fixtures exercise the
    # exact conversion and batched LAPACK path without using scientific raw
    # densities.  Random phases are addressed only by resource_preflight.
    fixture = np.zeros(
        (fixture_count, dimension, dimension),
        dtype=np.complex64,
    )
    diagonal = np.arange(dimension)
    fixture[:, diagonal, diagonal] = np.float32(1.0 / dimension)
    adjacent = np.arange(dimension - 1)
    rng = np.random.default_rng(
        np.random.SeedSequence([resource_seed, 0xD31517])
    )
    phase = rng.uniform(
        -np.pi,
        np.pi,
        size=(fixture_count, dimension - 1),
    )
    amplitude = np.float32(1.0 / (16.0 * dimension))
    off_diagonal = (
        amplitude * np.exp(1j * phase)
    ).astype(np.complex64)
    fixture[:, adjacent, adjacent + 1] = off_diagonal
    fixture[:, adjacent + 1, adjacent] = np.conjugate(off_diagonal)

    original_trace = np.zeros(fixture_count, dtype=np.float64)
    original_hermiticity = np.zeros(fixture_count, dtype=np.float64)
    # The random phases are removable by a diagonal unitary, so every fixture
    # has the eigenvalues of the same Hermitian Toeplitz tridiagonal matrix.
    original_minimum = np.full(
        fixture_count,
        float(
            np.float32(1.0 / dimension)
            + 2.0
            * amplitude
            * np.cos(dimension * np.pi / (dimension + 1))
        ),
        dtype=np.float64,
    )
    certified = np.full(fixture_count, 2.0e-6, dtype=np.float64)
    tolerance = 5.0e-12
    trial_seconds: list[float] = []
    eigvalsh_matrix_count = 0
    rss_callback_count = 0
    checksum = sha256()
    peak_components: dict[str, int] = {}
    for _ in range(timed_repeats):
        trial_started = time.perf_counter()
        for start in range(0, fixture_count, block_size):
            stop = start + block_size
            stack = np.asarray(
                fixture[start:stop],
                dtype=np.complex128,
            )
            adjoint = np.swapaxes(stack.conj(), 1, 2)
            hermitian = 0.5 * (stack + adjoint)
            traces = np.trace(stack, axis1=1, axis2=2)
            trace_error = (
                np.abs(traces.real - 1.0) + np.abs(traces.imag)
            )
            difference = stack - adjoint
            hermiticity = np.linalg.norm(
                difference.reshape(stop - start, -1),
                axis=1,
            )
            eigenvalues = np.linalg.eigvalsh(hermitian)
            minimum = eigenvalues[:, 0]
            eigvalsh_matrix_count += stop - start
            q = certified[start:stop]
            if (
                np.any(original_trace[start:stop] > 5e-8)
                or np.any(original_hermiticity[start:stop] > 5e-8)
                or np.any(original_minimum[start:stop] < -5e-8)
                or np.any(
                    trace_error
                    > original_trace[start:stop]
                    + np.sqrt(dimension) * q
                    + tolerance
                )
                or np.any(
                    hermiticity
                    > original_hermiticity[start:stop]
                    + 2.0 * q
                    + tolerance
                )
                or np.any(
                    np.abs(minimum - original_minimum[start:stop])
                    > q + tolerance
                )
                or np.any(minimum < -5e-8 - q - tolerance)
            ):
                raise RuntimeError(
                    "retained density resource physicality/Weyl drift"
                )
            checksum.update(
                np.asarray(minimum, dtype="<f8").tobytes(order="C")
            )
            live_components = {
                "fixture_complex64": int(fixture.nbytes),
                "stack_complex128": int(stack.nbytes),
                "adjoint_complex128": int(adjoint.nbytes),
                "hermitian_complex128": int(hermitian.nbytes),
                "hermiticity_difference_complex128": int(
                    difference.nbytes
                ),
                "trace_complex128": int(traces.nbytes),
                "trace_error_float64": int(trace_error.nbytes),
                "hermiticity_float64": int(hermiticity.nbytes),
                # minimum is a view into this full batched eigvalsh result.
                "eigenvalues_float64": int(eigenvalues.nbytes),
            }
            if sum(live_components.values()) > sum(
                peak_components.values()
            ):
                peak_components = live_components
            if sample_callback is not None:
                sample_callback()
                rss_callback_count += 1
        trial_seconds.append(time.perf_counter() - trial_started)
    if (
        len(trial_seconds) != timed_repeats
        or not all(
            np.isfinite(value) and value > 0.0
            for value in trial_seconds
        )
        or sum(trial_seconds) < 0.02
    ):
        raise RuntimeError(
            "retained density resource timing was not measurable"
        )
    worst_seconds_per_matrix = (
        max(trial_seconds) / float(fixture_count)
    )
    projected = worst_seconds_per_matrix * float(full_count)
    mean_trial = float(np.mean(trial_seconds))
    coefficient_of_variation = float(
        np.std(trial_seconds) / mean_trial
    )
    report: dict[str, Any] = {
        "schema_version": (
            "PHASE9-RETAINED-DENSITY-PHYSICALITY-RESOURCE-PROFILE-V1"
        ),
        "matrix_dimension": dimension,
        "block_size": block_size,
        "fixture_matrix_count": fixture_count,
        "fixture_bytes": int(fixture.nbytes),
        "timed_repeats": timed_repeats,
        "timed_matrix_evaluations": eigvalsh_matrix_count,
        "trial_wall_seconds": trial_seconds,
        "measured_total_wall_seconds": float(sum(trial_seconds)),
        "trial_coefficient_of_variation": coefficient_of_variation,
        "worst_seconds_per_matrix": worst_seconds_per_matrix,
        "projected_full_retained_count": full_count,
        "projected_full_serial_wall_seconds": projected,
        "full_fixture_generated": False,
        "complex64_to_complex128_exercised": True,
        "trace_recomputed": True,
        "hermiticity_frobenius_recomputed": True,
        "batched_minimum_eigvalsh_recomputed": True,
        "weyl_certificate_checked": True,
        "rss_callback_count": rss_callback_count,
        "peak_explicit_live_components": peak_components,
        "peak_explicit_live_bytes": int(sum(peak_components.values())),
        "kernel_sha256": checksum.hexdigest(),
        "seed_namespace": "resource_preflight",
        "seed_address": resource_seed,
        "formal_seed_addresses_accessed": False,
        "scientific_data_used": False,
        "scientific_verdict": None,
        "qualified_claim": None,
    }
    report["analysis_sha256"] = _sha(report)
    return report


def validate_statistics_profile(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    specification = config["resource_contract"]["profile_plan"][
        "joint_maxt_3037x199"
    ]
    required = {
        "gate_count": int(specification["gate_count"]),
        "replicates": int(specification["replicates"]),
        "largest_cluster_count": int(specification["largest_cluster_count"]),
        "largest_density_dimension": int(
            specification["largest_density_dimension"]
        ),
    }
    if any(profile.get(name) != value for name, value in required.items()):
        raise RuntimeError("joint maxT resource profile shape drift")
    replicates = int(specification["replicates"])
    clusters = int(specification["largest_cluster_count"])
    dimension = int(specification["largest_density_dimension"])
    block_rows = 32
    block_count = (clusters + block_rows - 1) // block_rows
    full_block_bytes = block_rows * dimension * dimension * 16
    density_matrix_bytes = dimension * dimension * 16
    perturbation_bytes = replicates * density_matrix_bytes
    expected_persistent = {
        "gate_work": clusters * dimension * 8,
        "maxima": replicates * 8,
        "l1_maxima": replicates * 8,
        "sign_cache": (93 * 1536 + 4 * 4608) * replicates,
        "base_replicates": 97 * 2 * replicates * 8,
    }
    expected_density_live = {
        "perturbation": perturbation_bytes,
        "update": perturbation_bytes,
        "density_mean": density_matrix_bytes,
        "density_total": density_matrix_bytes,
        "source_left": full_block_bytes,
        "source_right": full_block_bytes,
        "material_block": full_block_bytes,
        "centered": full_block_bytes,
        "signed": replicates * block_rows * 8,
    }
    physicality = profile.get("retained_density_physicality_profile")
    explicit_accounting_valid = (
        profile.get("persistent_working_set_components")
        == expected_persistent
        and profile.get("persistent_working_set_bytes")
        == sum(expected_persistent.values())
        and profile.get("largest_density_peak_live_components")
        == expected_density_live
        and profile.get("largest_density_peak_live_bytes")
        == sum(expected_density_live.values())
        and profile.get("peak_explicit_working_set_bytes")
        == sum(expected_persistent.values())
        + sum(expected_density_live.values())
        and profile.get("peak_analysis_scratch_bytes")
        == max(
            sum(expected_persistent.values())
            + sum(expected_density_live.values()),
            int(
                physicality.get("peak_explicit_live_bytes", -1)
                if isinstance(physicality, Mapping)
                else -1
            ),
        )
    )
    physicality_valid = False
    if isinstance(physicality, Mapping):
        physicality_specification = config["resource_contract"][
            "profile_plan"
        ]["retained_density_physicality_full_482304"]
        fixture_count = int(
            physicality_specification["fixture_matrix_count"]
        )
        physicality_block = int(
            physicality_specification["block_size"]
        )
        repeats = int(physicality_specification["timed_repeats"])
        full_retained = int(
            physicality_specification["full_retained_count"]
        )
        trial_seconds = physicality.get("trial_wall_seconds")
        expected_physicality_live = {
            "fixture_complex64": fixture_count
            * dimension
            * dimension
            * 8,
            "stack_complex128": physicality_block
            * dimension
            * dimension
            * 16,
            "adjoint_complex128": physicality_block
            * dimension
            * dimension
            * 16,
            "hermitian_complex128": physicality_block
            * dimension
            * dimension
            * 16,
            "hermiticity_difference_complex128": physicality_block
            * dimension
            * dimension
            * 16,
            "trace_complex128": physicality_block * 16,
            "trace_error_float64": physicality_block * 8,
            "hermiticity_float64": physicality_block * 8,
            "eigenvalues_float64": physicality_block
            * dimension
            * 8,
        }
        if (
            isinstance(trial_seconds, list)
            and len(trial_seconds) == repeats
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in trial_seconds
            )
        ):
            measured_total = sum(float(value) for value in trial_seconds)
            worst_per_matrix = max(
                float(value) for value in trial_seconds
            ) / fixture_count
            expected_projection = worst_per_matrix * full_retained
            unsigned = dict(physicality)
            claimed = unsigned.pop("analysis_sha256", None)
            physicality_valid = (
                physicality.get("schema_version")
                == (
                    "PHASE9-RETAINED-DENSITY-PHYSICALITY-"
                    "RESOURCE-PROFILE-V1"
                )
                and physicality.get("matrix_dimension") == dimension
                and physicality.get("block_size") == physicality_block
                and physicality.get("fixture_matrix_count")
                == fixture_count
                and physicality.get("fixture_bytes")
                == expected_physicality_live["fixture_complex64"]
                and physicality.get("timed_repeats") == repeats
                and physicality.get("timed_matrix_evaluations")
                == fixture_count * repeats
                and math.isclose(
                    float(
                        physicality.get(
                            "measured_total_wall_seconds",
                            math.nan,
                        )
                    ),
                    measured_total,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                and measured_total >= 0.02
                and math.isfinite(
                    float(
                        physicality.get(
                            "trial_coefficient_of_variation",
                            math.nan,
                        )
                    )
                )
                and float(
                    physicality.get(
                        "trial_coefficient_of_variation",
                        math.inf,
                    )
                )
                <= 1.0
                and math.isclose(
                    float(
                        physicality.get(
                            "worst_seconds_per_matrix",
                            math.nan,
                        )
                    ),
                    worst_per_matrix,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                and physicality.get("projected_full_retained_count")
                == full_retained
                and math.isclose(
                    float(
                        physicality.get(
                            "projected_full_serial_wall_seconds",
                            math.nan,
                        )
                    ),
                    expected_projection,
                    rel_tol=1e-12,
                    abs_tol=1e-9,
                )
                and physicality.get("full_fixture_generated") is False
                and physicality.get(
                    "complex64_to_complex128_exercised"
                )
                is True
                and physicality.get("trace_recomputed") is True
                and physicality.get(
                    "hermiticity_frobenius_recomputed"
                )
                is True
                and physicality.get(
                    "batched_minimum_eigvalsh_recomputed"
                )
                is True
                and physicality.get("weyl_certificate_checked") is True
                and physicality.get("rss_callback_count")
                == (fixture_count // physicality_block) * repeats
                and physicality.get("peak_explicit_live_components")
                == expected_physicality_live
                and physicality.get("peak_explicit_live_bytes")
                == sum(expected_physicality_live.values())
                and isinstance(physicality.get("kernel_sha256"), str)
                and len(str(physicality.get("kernel_sha256"))) == 64
                and physicality.get("seed_namespace")
                == "resource_preflight"
                and physicality.get("seed_address")
                == profile.get("seed_address")
                and physicality.get(
                    "formal_seed_addresses_accessed"
                )
                is False
                and physicality.get("scientific_data_used") is False
                and physicality.get("scientific_verdict") is None
                and physicality.get("qualified_claim") is None
                and claimed == _sha(unsigned)
            )
    if (
        profile.get("streaming") is not True
        or profile.get("maximum_coexisting_gate_buffers") != 1
        or profile.get("cached_cluster_root_groups") != 97
        or profile.get("production_rademacher_generator_exercised") is not True
        or profile.get("conservative_dual_leg_max_exercised") is not True
        or profile.get("dual_leg_evaluation_count") != 6074
        or profile.get("l1_accumulation_exercised") is not True
        or profile.get("largest_density_kernel_exercised") is not True
        or profile.get("largest_density_root_count")
        != int(specification["largest_cluster_count"])
        or profile.get("largest_density_block_rows") != 32
        or profile.get("largest_density_block_count") != block_count
        or profile.get("largest_density_source_buffer_count") != 2
        or profile.get("largest_density_rss_callback_count")
        != block_count
        or profile.get("largest_density_perturbation_shape")
        != [
            int(specification["replicates"]),
            int(specification["largest_density_dimension"]),
            int(specification["largest_density_dimension"]),
        ]
        or profile.get("largest_density_perturbation_bytes")
        != (
            int(specification["replicates"])
            * int(specification["largest_density_dimension"]) ** 2
            * 16
        )
        or profile.get("largest_density_update_bytes")
        != profile.get("largest_density_perturbation_bytes")
        or profile.get("largest_density_trace_norm_evaluations")
        != int(specification["replicates"])
        or explicit_accounting_valid is not True
        or physicality_valid is not True
        or not isinstance(profile.get("largest_density_kernel_sha256"), str)
        or len(str(profile.get("largest_density_kernel_sha256"))) != 64
        or profile.get("formal_seed_addresses_accessed") is not False
        or profile.get("scientific_influences_used") is not False
        or profile.get("scientific_verdict") is not None
        or profile.get("qualified_claim") is not None
    ):
        raise RuntimeError("joint maxT resource profile implementation drift")
    resource = config["seed_registry"]["resource_preflight"]
    seed = profile.get("seed_address")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not int(resource["start"])
        <= seed
        < int(resource["start"]) + int(resource["count"])
    ):
        raise RuntimeError("joint maxT dry-run seed escaped resource namespace")


def _reset_events(config: Mapping[str, Any], cell: T04CellSpec) -> int:
    if cell.layer != "fault":
        return cell.sample_count if cell.action == "RESET" else 0
    sequence = config["formal_matrix"]["fault_action_sequences"][cell.scenario]
    reset_rounds = sum(
        sequence[index % len(sequence)] == "RESET"
        for index in range(cell.horizon)
    )
    return cell.sample_count * reset_rounds


def _work_components(config: Mapping[str, Any], cell: T04CellSpec) -> dict[str, float]:
    dimension = float(3 * cell.cutoff)
    rows = float(cell.expected_rows)
    densities = float(cell.sample_count if cell.density_retention != "none" else 0)
    resets = float(_reset_events(config, cell))
    return {
        "round_steps": rows,
        "density_elements": densities * dimension * dimension,
        "reset_density_elements": resets * dimension * dimension,
        "physics_cubic": float(cell.sample_count) * dimension**3,
        "trajectory_cubic": rows * dimension**3,
    }


def stratified_projection(
    config: Mapping[str, Any],
    cells: Sequence[T04CellSpec],
    measurements: Sequence[Mapping[str, Any]],
    *,
    stats_wall_seconds: float,
    retained_density_physicality_wall_seconds: float,
) -> dict[str, Any]:
    """Project by layer/backend and structural components, never one row ratio."""

    measured = {int(item["plan_index"]): item for item in measurements}
    required = {388, 389, 403, 485, 507}
    if set(measured) != required:
        raise RuntimeError("projection requires all five frozen resource profiles")
    backend_a_wall_factor = max(
        1.0,
        float(measured[388]["wall_seconds"])
        / max(1.0e-12, float(measured[389]["wall_seconds"])),
    )
    backend_a_byte_factor = max(
        1.0,
        float(measured[388]["object_bytes_unique"])
        / max(1.0, float(measured[389]["object_bytes_unique"])),
    )

    def representative(cell: T04CellSpec) -> int:
        if cell.layer == "fault":
            return 485
        if cell.layer == "logical":
            return 403
        if cell.layer == "probe":
            return 507
        return 388 if cell.backend == "A" else 389

    layer_rows: dict[str, dict[str, float]] = {}
    projected_bytes = 0.0
    projected_worker_wall = 0.0
    cell_projections: list[dict[str, Any]] = []
    row_roles = {"round_ledger_csv", "raw_iq_npy", "heldout_iq_npy"}

    def component_bytes(rep: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
        bindings = rep.get("object_bindings")
        if isinstance(bindings, list) and bindings:
            by_digest: dict[str, dict[str, Any]] = {}
            for binding in bindings:
                digest = str(binding["sha256"])
                entry = by_digest.setdefault(
                    digest,
                    {"bytes": int(binding["bytes"]), "roles": set()},
                )
                if entry["bytes"] != int(binding["bytes"]):
                    raise RuntimeError("measured digest size drift")
                entry["roles"].add(str(binding["role"]))
            totals = {
                "row": 0,
                "primary": 0,
                "reset_scalar": 0,
                "reset_density": 0,
                "other": 0,
            }
            for entry in by_digest.values():
                roles_for_digest = entry["roles"]
                size = int(entry["bytes"])
                if roles_for_digest & row_roles:
                    totals["row"] += size
                elif "primary_density_npy" in roles_for_digest:
                    # rb_expected_density may alias the primary object.
                    totals["primary"] += size
                elif any(
                    role.startswith("rb_") and "density" in role
                    for role in roles_for_digest
                ):
                    totals["reset_density"] += size
                elif any(role.startswith("rb_") for role in roles_for_digest):
                    totals["reset_scalar"] += size
                else:
                    totals["other"] += size
            if sum(totals.values()) != int(rep["object_bytes_unique"]):
                raise RuntimeError("component projection double-counted an object")
            return (
                totals["row"],
                totals["primary"],
                totals["reset_scalar"],
                totals["reset_density"],
                totals["other"],
            )
        roles = rep["object_bytes_by_role"]
        row = sum(int(roles.get(role, 0)) for role in row_roles)
        primary = int(roles.get("primary_density_npy", 0))
        reset_scalar = sum(
            int(size)
            for role, size in roles.items()
            if str(role).startswith("rb_") and "density" not in str(role)
        )
        reset_density = sum(
            int(size)
            for role, size in roles.items()
            if str(role).startswith("rb_") and "density" in str(role)
        )
        other = max(
            0,
            int(rep["object_bytes_unique"])
            - row
            - primary
            - reset_scalar
            - reset_density,
        )
        return row, primary, reset_scalar, reset_density, other

    for cell in cells:
        rep_index = representative(cell)
        rep = measured[rep_index]
        rep_cell = cells[rep_index]
        target_components = _work_components(config, cell)
        rep_components = _work_components(config, rep_cell)
        (
            row_bytes,
            primary_bytes,
            reset_scalar,
            reset_density,
            other_bytes,
        ) = component_bytes(rep)
        row_ratio = target_components["round_steps"] / max(
            1.0, rep_components["round_steps"]
        )
        density_ratio = target_components["density_elements"] / max(
            1.0, rep_components["density_elements"]
        ) if target_components["density_elements"] else 0.0
        reset_ratio = target_components["reset_density_elements"] / max(
            1.0, rep_components["reset_density_elements"]
        ) if target_components["reset_density_elements"] else 0.0
        cell_bytes = (
            row_bytes * row_ratio
            + primary_bytes * density_ratio
            + reset_scalar * reset_ratio
            + reset_density * reset_ratio
            + other_bytes
        )
        if cell.backend == "A" and rep_index != 388:
            cell_bytes *= backend_a_byte_factor
        # Wall uses two distinct structural terms.  This intentionally avoids
        # applying a single observed seconds/row ratio to unlike cells.
        trajectory_ratio = target_components["trajectory_cubic"] / max(
            1.0, rep_components["trajectory_cubic"]
        )
        reset_work_ratio = (
            target_components["reset_density_elements"]
            / max(1.0, rep_components["reset_density_elements"])
            if target_components["reset_density_elements"]
            else 0.0
        )
        wall_ratio = max(trajectory_ratio, reset_work_ratio)
        cell_wall = float(rep["wall_seconds"]) * wall_ratio
        if cell.backend == "A" and rep_index != 388:
            cell_wall *= backend_a_wall_factor
        projected_bytes += cell_bytes
        projected_worker_wall += cell_wall
        cell_projections.append(
            {
                "plan_index": cell.plan_index,
                "chunk_id": cell.chunk_id,
                "representative_plan_index": rep_index,
                "projected_object_bytes": int(cell_bytes + 0.999999),
                "projected_wall_seconds": cell_wall,
            }
        )
        bucket = layer_rows.setdefault(
            cell.layer,
            {
                "cell_count": 0.0,
                "rows": 0.0,
                "density_elements": 0.0,
                "reset_density_elements": 0.0,
                "projected_bytes": 0.0,
                "projected_worker_wall_seconds": 0.0,
            },
        )
        bucket["cell_count"] += 1
        bucket["rows"] += cell.expected_rows
        bucket["density_elements"] += target_components["density_elements"]
        bucket["reset_density_elements"] += target_components[
            "reset_density_elements"
        ]
        bucket["projected_bytes"] += cell_bytes
        bucket["projected_worker_wall_seconds"] += cell_wall
    max_workers = int(config["runtime_contract"]["max_workers"])
    physicality_wall = float(
        retained_density_physicality_wall_seconds
    )
    if (
        not math.isfinite(physicality_wall)
        or physicality_wall <= 0.0
    ):
        raise RuntimeError(
            "retained density physicality wall projection invalid"
        )
    projected_wall = (
        projected_worker_wall / max_workers
        + float(stats_wall_seconds)
        + physicality_wall
    )
    report: dict[str, Any] = {
        "schema_version": PROJECTION_SCHEMA,
        "method": (
            "layer/backend representative with separate round, density, "
            "reset-density, cutoff-dimension and trajectory-horizon components"
        ),
        "uniform_row_ratio_used": False,
        "representative_plan_indices": sorted(required),
        "backend_a_conservative_wall_factor": backend_a_wall_factor,
        "backend_a_conservative_byte_factor": backend_a_byte_factor,
        "cell_projections": cell_projections,
        "layers": layer_rows,
        "projected_formal_artifact_bytes": int(projected_bytes + 0.999999),
        "projected_formal_worker_wall_seconds": projected_worker_wall,
        "projected_formal_wall_seconds_at_frozen_concurrency": projected_wall,
        "frozen_concurrency": max_workers,
        "statistics_wall_seconds": float(stats_wall_seconds),
        "retained_density_physicality_serial_wall_seconds": (
            physicality_wall
        ),
    }
    report["projection_sha256"] = _sha(report)
    return report


def _object_tree_snapshot(object_root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(object_root).as_posix(): _sha_file(path)
        for path in sorted(object_root.rglob("*"))
        if path.is_file()
    }


def no_copy_inventory(
    store: ImmutableObjectStore,
    cells: Sequence[T04CellSpec],
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _object_tree_snapshot(store.object_root)
    started = time.monotonic()
    inventory = store.inventory([asdict(cell) for cell in cells])
    wall = time.monotonic() - started
    after = _object_tree_snapshot(store.object_root)
    if before != after:
        raise RuntimeError("inventory finalize copied or mutated raw objects")
    forbidden = [
        path.as_posix()
        for path in store.repository_root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() == ".zip"
            or path.name.lower() in {"merged.csv", "full.csv", "all_rows.csv"}
        )
        and (
            path == store.object_root
            or store.object_root in path.parents
            or path == store.receipt_root
            or store.receipt_root in path.parents
            or path == store.staging_root
            or store.staging_root in path.parents
        )
    ]
    if forbidden:
        raise RuntimeError(f"monolithic preflight archive found: {forbidden[:3]}")
    evidence: dict[str, Any] = {
        "receipt_count": inventory["receipt_count"],
        "unique_object_count": inventory["unique_object_count"],
        "object_bytes_unique": inventory["totals"]["object_bytes_unique"],
        "object_tree_unchanged": True,
        "object_tree_sha256": _sha(before),
        "finalize_wall_seconds": wall,
        "monolithic_archive": None,
        "merged_full_csv": None,
        "raw_payload_bytes_copied_during_finalize": 0,
    }
    evidence["analysis_sha256"] = _sha(evidence)
    return inventory, evidence


def resource_gate_decision(
    config: Mapping[str, Any],
    *,
    sampling: Mapping[str, Any],
    projection: Mapping[str, Any],
    inventory: Mapping[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    contract = config["resource_contract"]
    free = int(shutil.disk_usage(run_directory).free)
    projected_bytes = int(projection["projected_formal_artifact_bytes"])
    post_projection_free = free - projected_bytes
    checks = {
        "rss": int(sampling["peak_aggregate_rss_bytes"])
        <= int(contract["maximum_peak_rss_bytes"]),
        "artifact": projected_bytes <= int(contract["maximum_artifact_bytes"]),
        "disk": post_projection_free
        >= int(contract["minimum_post_projection_free_bytes"]),
        "wall": float(projection["projected_formal_wall_seconds_at_frozen_concurrency"])
        <= float(contract["maximum_wall_seconds"]),
        "inventory": (
            inventory["raw_status"]
            == "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT"
            and int(inventory["receipt_count"]) == 5
            and int(inventory["totals"]["observed_rows"])
            == int(inventory["totals"]["expected_rows"])
            and int(inventory["totals"]["exception_rows"]) == 0
            and int(inventory["totals"]["missing_rows"]) == 0
            and int(inventory["totals"]["conservation_failures"]) == 0
            and inventory["monolithic_archive"] is None
            and inventory["merged_full_csv"] is None
        ),
    }
    decision: dict[str, Any] = {
        "checks": checks,
        "passed": all(checks.values()),
        "disk_free_bytes": free,
        "projected_post_formal_free_bytes": post_projection_free,
        "limits": {
            "maximum_peak_rss_bytes": int(contract["maximum_peak_rss_bytes"]),
            "maximum_artifact_bytes": int(contract["maximum_artifact_bytes"]),
            "minimum_post_projection_free_bytes": int(
                contract["minimum_post_projection_free_bytes"]
            ),
            "maximum_wall_seconds": float(contract["maximum_wall_seconds"]),
        },
    }
    decision["decision_sha256"] = _sha(decision)
    return decision


@dataclass
class ResourcePreflightSupervisor:
    root: Path
    config: Mapping[str, Any]
    config_sha256: str
    plan_sha256: str
    run_id: str
    source_snapshot_sha256: str
    sample_interval_seconds: float = 5.0
    heartbeat_period_seconds: float = 30.0
    process_group_runner: Callable[..., list[dict[str, Any]]] = execute_process_group
    stats_runner: Callable[..., dict[str, Any]] = streaming_statistics_dry_run
    lineage_validator: Callable[..., dict[str, Any]] = validate_preflight_lineage

    def run(self) -> dict[str, Any]:
        root = self.root.resolve()
        cells = build_cell_plan(self.config)
        concurrent, singleton = profile_cells(self.config, cells)
        selected = concurrent + [singleton]
        preflight_root, artifact_paths = isolated_preflight_paths(
            root, self.config, run_id=self.run_id
        )
        if (
            preflight_root.exists()
            and not (preflight_root / "owner.lock").exists()
            and any(preflight_root.iterdir())
        ):
            raise RuntimeError(
                "preflight namespace already contains evidence; fresh run_id required"
            )
        preflight_root.mkdir(parents=True, exist_ok=True)
        owner = OwnerLease(
            preflight_root / "owner.lock",
            run_id=self.run_id,
            config_sha256=self.config_sha256,
            plan_sha256=self.plan_sha256,
        )
        owner.acquire()
        attempt_path = preflight_root / "attempts.jsonl"
        active_pids: set[int] = set()
        active_lock = Lock()

        def child_pids() -> list[int]:
            with active_lock:
                return sorted(active_pids)

        sampler = ResourceSampler(
            evidence_path=preflight_root / "resource_samples.jsonl",
            child_pids=child_pids,
            stage=lambda: str(state["stage"]),
            interval_seconds=self.sample_interval_seconds,
        )
        state: dict[str, Any] = {
            "stage": "starting",
            "child_pids": [],
            "profiles_completed": 0,
        }
        heartbeat = HeartbeatService(
            path=preflight_root / "heartbeat.json",
            owner=owner,
            period_seconds=self.heartbeat_period_seconds,
            snapshot=lambda: {**state, "child_pids": child_pids()},
        )
        sampler_started = False
        heartbeat_started = False
        terminal_recorded = False
        try:
            lineage = self.lineage_validator(
                root,
                self.config,
                self.config_sha256,
                self.plan_sha256,
                self.source_snapshot_sha256,
            )
            _record_attempt(
                attempt_path,
                task_id=str(self.config["task_id"]),
                run_id=self.run_id,
                event="START_RESOURCE_PREFLIGHT",
                payload={
                    "formal_seed_addresses_accessed": False,
                    "artifact_namespace": artifact_paths,
                },
            )
            sampler.start()
            sampler_started = True
            heartbeat.start()
            heartbeat_started = True
            seed_firewall = assert_seed_firewall(self.config)

            def kwargs_for(cell: T04CellSpec) -> dict[str, Any]:
                return {
                    "root": root,
                    "t04": self.config,
                    "config_sha256": self.config_sha256,
                    "plan_sha256": self.plan_sha256,
                    "run_id": self.run_id,
                    "cell": cell,
                    "source_snapshot_sha256": self.source_snapshot_sha256,
                    "sample_count_override": None,
                    "seed_namespace": "resource_preflight",
                    "artifact_paths_override": artifact_paths,
                }

            state["stage"] = "four_worker_concurrent_peak"
            measurements = self.process_group_runner(
                [kwargs_for(cell) for cell in concurrent],
                active_pids=active_pids,
                active_lock=active_lock,
                sample_callback=sampler.sample_once,
            )
            concurrent_peak = sampler.summary()[
                "stage_peak_aggregate_rss_bytes"
            ].get("four_worker_concurrent_peak", 0)
            for measurement in measurements:
                measurement["profile_peak_aggregate_rss_bytes"] = concurrent_peak
            state["profiles_completed"] = len(measurements)
            state["stage"] = "backend_a_representative"
            measurements.extend(
                self.process_group_runner(
                    [kwargs_for(singleton)],
                    active_pids=active_pids,
                    active_lock=active_lock,
                    sample_callback=sampler.sample_once,
                )
            )
            singleton_peak = sampler.summary()[
                "stage_peak_aggregate_rss_bytes"
            ].get("backend_a_representative", 0)
            for measurement in measurements:
                if int(measurement["plan_index"]) == singleton.plan_index:
                    measurement["profile_peak_aggregate_rss_bytes"] = singleton_peak
            state["profiles_completed"] = len(measurements)
            state["stage"] = "joint_maxt_3037x199"
            stats = self.stats_runner(
                self.config,
                sample_callback=sampler.sample_once,
            )
            validate_statistics_profile(self.config, stats)
            stats["profile_peak_aggregate_rss_bytes"] = sampler.summary()[
                "stage_peak_aggregate_rss_bytes"
            ].get("joint_maxt_3037x199", 0)
            state["stage"] = "inventory_finalize_no_copy"
            store = ImmutableObjectStore(
                repository_root=root,
                object_root=root / artifact_paths["object_store"],
                staging_root=root / artifact_paths["staging_directory"],
                receipt_root=root / artifact_paths["receipt_directory"],
                task_id=str(self.config["task_id"]),
                run_id=self.run_id,
                config_sha256=self.config_sha256,
                plan_sha256=self.plan_sha256,
            )
            inventory, inventory_evidence = no_copy_inventory(store, selected)
            inventory_path = preflight_root / "inventory.json"
            _immutable_json(inventory_path, inventory)
            inventory_binding = _json_binding(root, inventory_path)
            # Ensure at least one final sample sees the post-finalize parent.
            sampler.sample_once()
            heartbeat.stop()
            heartbeat_started = False
            sampler.stop()
            sampler_started = False
            sampling = sampler.summary()
            sampling["evidence"] = _json_binding(
                root, preflight_root / "resource_samples.jsonl"
            )
            sampling.pop("summary_sha256", None)
            sampling["summary_sha256"] = _sha(sampling)
            validate_continuous_sampling(sampling)
            physicality_profile = stats[
                "retained_density_physicality_profile"
            ]
            statistics_wall = (
                float(stats["wall_seconds"])
                - float(
                    physicality_profile[
                        "measured_total_wall_seconds"
                    ]
                )
            )
            if statistics_wall <= 0.0:
                raise RuntimeError(
                    "statistics wall does not exceed physicality sample"
                )
            projection = stratified_projection(
                self.config,
                cells,
                measurements,
                stats_wall_seconds=statistics_wall,
                retained_density_physicality_wall_seconds=float(
                    physicality_profile[
                        "projected_full_serial_wall_seconds"
                    ]
                ),
            )
            decision = resource_gate_decision(
                self.config,
                sampling=sampling,
                projection=projection,
                inventory=inventory,
                run_directory=preflight_root,
            )
            if not decision["passed"]:
                raise RuntimeError(
                    "resource gates failed: "
                    + ",".join(
                        name for name, passed in decision["checks"].items() if not passed
                    )
                )
            heartbeat_live = json.loads(
                (preflight_root / "heartbeat.json").read_text(encoding="utf-8")
            )
            heartbeat_binding = _json_binding(
                root, preflight_root / "heartbeat.json"
            )
            observed_span = (
                float(sampling["last_sample"]["monotonic_seconds"])
                - float(sampling["first_sample"]["monotonic_seconds"])
            )
            if (
                int(heartbeat_live.get("sequence", -1)) < 1
                or observed_span < self.heartbeat_period_seconds
            ):
                raise RuntimeError(
                    "independent heartbeat did not span one full frozen period"
                )
            report: dict[str, Any] = {
                "schema_version": PREFLIGHT_SCHEMA,
                "task_id": self.config["task_id"],
                "run_id": self.run_id,
                "runner_id": RUNNER_ID,
                "verdict": PASS_VERDICT,
                "config_sha256": self.config_sha256,
                "plan_sha256": self.plan_sha256,
                "source_snapshot_sha256": self.source_snapshot_sha256,
                "lineage_validation": lineage,
                "seed_firewall": seed_firewall,
                "artifact_namespace": artifact_paths,
                "formal_artifact_namespace_accessed": False,
                "full_size_receipt_count": inventory["receipt_count"],
                "profile_measurements": sorted(
                    measurements, key=lambda value: value["plan_index"]
                ),
                "actual_peak_concurrency": sampling[
                    "maximum_observed_live_children"
                ],
                "maximum_observed_worker_overlap": sampling[
                    "maximum_observed_live_children"
                ],
                "resource_sample_count": sampling["sample_count"],
                "sampling": sampling,
                "heartbeat": {
                    "path": _relative(root, preflight_root / "heartbeat.json"),
                    "binding": heartbeat_binding,
                    "period_seconds": self.heartbeat_period_seconds,
                    "latest_sequence": heartbeat_live["sequence"],
                    "observed_sampling_span_seconds": observed_span,
                    "latest_child_pids": heartbeat_live["snapshot"].get(
                        "child_pids", []
                    ),
                    "independent_of_chunk_completion": True,
                },
                "streaming_statistics_dry_run": stats,
                "joint_maxt_profile": stats,
                "projection": projection,
                "cell_projections": projection["cell_projections"],
                "maximum_inflight_temp_bytes": sum(
                    sorted(
                        (
                            int(item["object_bytes_unique"])
                            for item in measurements
                        ),
                        reverse=True,
                    )[: int(self.config["runtime_contract"]["max_workers"])]
                ),
                "analysis_scratch_bytes": int(
                    stats["peak_analysis_scratch_bytes"]
                ),
                "formal_projected_object_bytes": projection[
                    "projected_formal_artifact_bytes"
                ],
                "formal_projected_wall_seconds": projection[
                    "projected_formal_wall_seconds_at_frozen_concurrency"
                ],
                "inventory": inventory,
                "inventory_binding": inventory_binding,
                "inventory_no_copy_evidence": inventory_evidence,
                "resource_gate_decision": decision,
                "scientific_verdict": None,
                "qualified_claim": None,
                "claim_boundary": _claims_null(),
                "attempt_witnesses_before_terminal": [
                    _json_binding(
                        root,
                        preflight_root / "attempt_events" / "00000000.json",
                    )
                ],
            }
            report["analysis_sha256"] = _sha(report)
            official = root / str(self.config["artifact_paths"]["resource_preflight"])
            _immutable_json(official, report)
            _record_attempt(
                attempt_path,
                task_id=str(self.config["task_id"]),
                run_id=self.run_id,
                event="PASS_RESOURCE_PREFLIGHT",
                payload={
                    "analysis_sha256": report["analysis_sha256"],
                    "formal_seed_addresses_accessed": False,
                },
            )
            terminal_recorded = True
            return report
        except BaseException as exc:
            cleanup_errors: list[str] = []
            if heartbeat_started:
                try:
                    heartbeat.stop()
                except BaseException as cleanup_exc:
                    cleanup_errors.append(
                        f"heartbeat:{type(cleanup_exc).__name__}:{cleanup_exc}"
                    )
                heartbeat_started = False
            if sampler_started:
                try:
                    sampler.stop()
                except BaseException as cleanup_exc:
                    cleanup_errors.append(
                        f"sampler:{type(cleanup_exc).__name__}:{cleanup_exc}"
                    )
                sampler_started = False
            sampling = sampler.summary()
            sample_path = preflight_root / "resource_samples.jsonl"
            if sample_path.exists():
                sampling["evidence"] = _json_binding(root, sample_path)
                sampling.pop("summary_sha256", None)
                sampling["summary_sha256"] = _sha(sampling)
            failure: dict[str, Any] = {
                "schema_version": PREFLIGHT_SCHEMA,
                "task_id": self.config["task_id"],
                "run_id": self.run_id,
                "runner_id": RUNNER_ID,
                "verdict": FAIL_VERDICT,
                "config_sha256": self.config_sha256,
                "plan_sha256": self.plan_sha256,
                "source_snapshot_sha256": self.source_snapshot_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "cleanup_errors": cleanup_errors,
                "formal_seed_addresses_accessed": False,
                "formal_artifact_namespace_accessed": False,
                "artifact_namespace": artifact_paths,
                "sampling": sampling,
                "scientific_verdict": None,
                "qualified_claim": None,
                "claim_boundary": _claims_null(),
            }
            failure["analysis_sha256"] = _sha(failure)
            _record_attempt(
                attempt_path,
                task_id=str(self.config["task_id"]),
                run_id=self.run_id,
                event="FAIL_RESOURCE_PREFLIGHT",
                payload={
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "analysis_sha256": failure["analysis_sha256"],
                },
            )
            terminal_recorded = True
            _immutable_json(preflight_root / "resource_preflight_failed.json", failure)
            raise ResourcePreflightFailure(failure) from exc
        finally:
            state["stage"] = "terminal"
            if heartbeat_started:
                heartbeat.stop()
            if sampler_started:
                sampler.stop()
            if not terminal_recorded:
                _record_attempt(
                    attempt_path,
                    task_id=str(self.config["task_id"]),
                    run_id=self.run_id,
                    event="FAIL_RESOURCE_PREFLIGHT_CLEANUP",
                    payload={"formal_seed_addresses_accessed": False},
                )
            owner.release()


def run_resource_preflight(
    *,
    root: Path,
    config_path: Path,
    run_id: str,
    sample_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = _inside(config_path, root, "T04 config")
    raw = config_path.read_bytes()
    config = json.loads(
        raw,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite config token {token}")
        ),
    )
    if not isinstance(config, dict):
        raise ValueError("T04 config must be one object")
    validate_config(config, root=root)
    plan = plan_payload(config)
    supervisor = ResourcePreflightSupervisor(
        root=root,
        config=config,
        config_sha256=sha256(raw).hexdigest(),
        plan_sha256=str(plan["canonical_plan_sha256"]),
        run_id=run_id,
        source_snapshot_sha256=str(
            runtime_source_snapshot(root, config)["source_snapshot_sha256"]
        ),
        sample_interval_seconds=sample_interval_seconds,
        heartbeat_period_seconds=float(
            config["runtime_contract"]["heartbeat_period_seconds"]
        ),
    )
    return supervisor.run()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full-denominator T04 resource preflight."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--confirm-full-size",
        required=True,
        choices=("T04-FULL-RESOURCE-PREFLIGHT",),
        help="Explicit guard against accidental full profile execution.",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    try:
        run_resource_preflight(
            root=arguments.root,
            config_path=arguments.root / arguments.config,
            run_id=arguments.run_id,
        )
    except ResourcePreflightFailure:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FAIL_VERDICT",
    "PASS_VERDICT",
    "PREFLIGHT_SCHEMA",
    "ResourcePreflightFailure",
    "ResourcePreflightSupervisor",
    "ResourceSampler",
    "assert_seed_firewall",
    "execute_process_group",
    "isolated_preflight_paths",
    "main",
    "no_copy_inventory",
    "profile_cells",
    "resource_gate_decision",
    "run_resource_preflight",
    "stratified_projection",
    "streaming_statistics_dry_run",
    "validate_continuous_sampling",
]
