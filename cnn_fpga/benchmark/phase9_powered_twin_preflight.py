"""Fail-closed resource preflight for the T04 powered-twin transaction.

The preflight deliberately has a different run, seed and artifact namespace
from formal evidence.  It executes eight full-denominator resource profile
cells frozen in the T04 config, including the exact formal LPT four-cell
prefix, measures four-worker concurrency continuously,
exercises a 3,037-by-199 streaming statistics kernel, and inventories the
content-addressed worker objects without copying them into an archive.

This module owns no scientific verdict.  Every claim field in its report is
literal ``null`` and none of its receipts are accepted by the formal run.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version as package_version
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
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
    cluster_root_id,
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
RAW_SEED_AUDIT_SCHEMA = "PHASE9-POWERED-TWIN-RESOURCE-RAW-SEED-AUDIT-V1"
RUNNER_ID = "phase9_powered_twin_resource_preflight_v1"
RAW_RUNNER_ID = "PHASE9-POWERED-TWIN-RAW-RUNNER-V1"
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
) -> tuple[list[T04CellSpec], list[T04CellSpec]]:
    profile = config["resource_contract"]["profile_plan"]
    peak_profile = profile["formal_lpt_four_worker_peak"]
    representative_profile = profile[
        "representative_four_worker_profiles"
    ]
    if (
        peak_profile.get("full_frozen_denominator") is not True
        or peak_profile.get("matches_formal_lpt_prefix") is not True
        or representative_profile.get("full_frozen_denominator") is not True
    ):
        raise RuntimeError("resource profiles must use the full frozen denominator")
    peak_indices = list(peak_profile["plan_indices"])
    representative_indices = list(representative_profile["plan_indices"])
    if len(peak_indices) != 4 or len(representative_indices) != 4:
        raise RuntimeError(
            "resource profile must contain exact four LPT and four "
            "representative cells"
        )
    indices = peak_indices + representative_indices
    if len(indices) != len(set(indices)) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(cells)
        for index in indices
    ):
        raise RuntimeError("invalid or duplicate resource profile plan index")
    peak = [cells[index] for index in peak_indices]
    representatives = [cells[index] for index in representative_indices]
    if [cell.sample_count for cell in peak] != list(
        peak_profile["sample_counts"]
    ):
        raise RuntimeError("formal-LPT profile denominator drift")
    if [cell.sample_count for cell in representatives] != list(
        representative_profile["sample_counts"]
    ):
        raise RuntimeError("representative profile denominator drift")
    if [cell.plan_index for cell in peak] != [478, 480, 482, 484]:
        raise RuntimeError("frozen formal-LPT profile identity drift")
    if [cell.plan_index for cell in representatives] != [388, 389, 403, 507]:
        raise RuntimeError("frozen representative profile identity drift")
    return peak, representatives


def _strict_csv_integer(
    row: Mapping[str | None, Any],
    field: str,
    *,
    expected: int,
    row_number: int,
) -> None:
    value = row.get(field)
    canonical = str(expected)
    if not isinstance(value, str) or value != canonical:
        raise RuntimeError(
            f"resource ledger {field} drift at row {row_number}: "
            f"expected={canonical!r} observed={value!r}"
        )


def _resource_seed_addresses(
    config: Mapping[str, Any],
    cell: T04CellSpec,
    *,
    position: int,
    round_index: int,
) -> tuple[int, int]:
    registry = config["seed_registry"]
    namespace = registry["resource_preflight"]
    maximum = int(registry["maximum_cluster_positions"])
    maximum_horizon = int(registry["maximum_horizon"])
    backend_index = 0 if cell.backend == "A" else 1
    physical = (
        int(namespace["start"])
        + int(namespace["physical_offset"])
        + backend_index * 97 * maximum
        + cell.pair_group_index * maximum
        + position
    )
    heldout = (
        int(namespace["start"])
        + int(namespace["heldout_offset"])
        + cell.pair_group_index * maximum * maximum_horizon
        + position * maximum_horizon
        + round_index
    )
    return physical, heldout


def _audit_resource_seed_ledger(
    path: Path,
    *,
    config: Mapping[str, Any],
    cell: T04CellSpec,
    expected_header: Sequence[str],
    heldout_iq: Any | None = None,
    global_row_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Stream one raw CSV and prove every physical/heldout seed address.

    The receipt fingerprint is only a declaration.  This audit recomputes the
    address for every archived row from the frozen cell identity, position,
    round and resource namespace.  It therefore catches a coordinated
    object/receipt/inventory/report rehash that substitutes formal seeds.
    """

    namespace = config["seed_registry"]["resource_preflight"]
    resource_start = int(namespace["start"])
    resource_stop = resource_start + int(namespace["count"])
    formal_intervals = [
        (
            str(name),
            int(config["seed_registry"][name]["start"]),
            int(config["seed_registry"][name]["start"])
            + int(config["seed_registry"][name]["count"]),
        )
        for name in ("physical", "heldout", "joint_maxt_rademacher")
    ]
    observed = 0
    physical_min: int | None = None
    physical_max: int | None = None
    heldout_min: int | None = None
    heldout_max: int | None = None
    row_ids: set[str] = set()
    row_id_digest = sha256()
    hex_characters = frozenset("0123456789abcdef")
    hash_fields = {
        name
        for name in expected_header
        if name.endswith("_sha256")
    }
    finite_if_present = {
        "pre_readout_i",
        "pre_readout_q",
        "pre_measurement_g",
        "pre_measurement_e",
        "pre_measurement_f",
        "pre_reset_g",
        "pre_reset_e",
        "pre_reset_f",
        "integrated_i",
        "integrated_q",
        "raw_log_evidence",
        "raw_reference_log_evidence",
        "raw_within_window_residual",
        "posterior_g",
        "posterior_e",
        "posterior_f",
        "level_g",
        "level_e",
        "level_f",
        "mean_photon",
        "leakage_residence_probability",
        "predictive_mean_i",
        "predictive_mean_q",
        "predictive_cov_ii",
        "predictive_cov_iq",
        "predictive_cov_qq",
        "heldout_reference_log_evidence",
        "heldout_proper_score_per_sample",
        "heldout_llr_ge_per_sample",
        "heldout_llr_gf_per_sample",
        "heldout_llr_ef_per_sample",
        "logical_survival",
        "density_trace_error",
        "density_hermiticity_frobenius",
        "density_minimum_eigenvalue",
        "density_quantization_frobenius_error",
        "density_quantization_certified_frobenius_bound",
        "density_quantization_trace_distance_bound",
        "posterior_normalization_error",
        "level_normalization_error",
        "reference_posterior_l1_error",
        "reference_log_evidence_error",
        *{f"drift_{index}" for index in range(5)},
        *{f"pre_intervention_drift_{index}" for index in range(5)},
        *{f"input_intervention_drift_{index}" for index in range(5)},
        *{
            f"logical_block_{row}{column}_{part}"
            for row in range(2)
            for column in range(2)
            for part in ("real", "imag")
        },
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_header):
            raise RuntimeError("resource ledger exact header drift")
        for observed, row in enumerate(reader, start=1):
            row_index = observed - 1
            if None in row:
                raise RuntimeError(
                    f"resource ledger has excess columns at row {row_index}"
                )
            if any(value is None for value in row.values()):
                raise RuntimeError(
                    f"resource ledger has missing columns at row {row_index}"
                )
            position = row_index // cell.horizon
            round_index = row_index % cell.horizon
            if position >= cell.sample_count:
                raise RuntimeError("resource ledger exceeds frozen denominator")
            physical, heldout = _resource_seed_addresses(
                config,
                cell,
                position=position,
                round_index=round_index,
            )
            for field, expected in (
                ("seed", physical),
                ("physical_seed_address", physical),
                ("heldout_seed_address", heldout),
                ("seed_position", position),
                ("round_index", round_index),
                ("archive_row_index", row_index),
                ("raw_iq_index", row_index),
                ("heldout_iq_index", row_index),
                ("cutoff", cell.cutoff),
            ):
                _strict_csv_integer(
                    row,
                    field,
                    expected=expected,
                    row_number=row_index,
                )
            expected_action = (
                config["formal_matrix"]["fault_action_sequences"][
                    cell.scenario
                ][
                    round_index
                    % len(
                        config["formal_matrix"]["fault_action_sequences"][
                            cell.scenario
                        ]
                    )
                ]
                if cell.layer == "fault"
                else cell.action
            )
            trajectory_id = (
                f"{cell.cell_base}|c{cell.cutoff}|{cell.backend}|"
                f"p{position:04d}"
                if cell.layer == "fault"
                else ""
            )
            expected_row_id = (
                f"{trajectory_id}|r{round_index:03d}"
                if trajectory_id
                else (
                    f"{cell.layer}|c{cell.cutoff}|{cell.cell_base}|"
                    f"{cell.backend}|p{position:04d}"
                )
            )
            if cell.layer == "shared":
                expected_cell_id = (
                    f"ab/c{cell.cutoff}/shared/{cell.initial_state}/"
                    f"{cell.action}"
                )
            elif cell.layer == "probe":
                expected_cell_id = (
                    f"ab/c{cell.cutoff}/probe/{cell.probe_id}"
                )
            elif cell.layer == "logical":
                expected_cell_id = (
                    f"ab/c{cell.cutoff}/logical/{cell.logical_label}/"
                    f"{cell.action}"
                )
            else:
                expected_cell_id = (
                    f"ab/c{cell.cutoff}/fault/{cell.scenario}"
                )
            terminal = round_index == cell.horizon - 1
            expected_density_index = (
                position
                if cell.density_retention != "none"
                and (cell.layer != "fault" or terminal)
                else -1
            )
            expected_strings = {
                "row_id": expected_row_id,
                "row_schema": (
                    "PHASE9-POWERED-TWIN-ROUND-LEDGER-V1"
                ),
                "layer": cell.layer,
                "cell_base": cell.cell_base,
                "cell_id": expected_cell_id,
                "backend": cell.backend,
                "backend_id": (
                    "PHASE9-BACKEND-A-JOINT-FOCK-QUTRIT-GKSL-V1"
                    if cell.backend == "A"
                    else "PHASE9-BACKEND-B-DENSE-STRANG-ANALYTIC-KRAUS-V1"
                ),
                "convergence_role": cell.convergence_role,
                "trajectory_id": trajectory_id,
                "action": expected_action,
                "probe_id": cell.probe_id,
                "scenario": cell.scenario,
                "initial_state": cell.initial_state,
                "logical_label": (
                    config["formal_matrix"]["logical_labels"][
                        position
                        // int(
                            config["formal_matrix"][
                                "fault_clusters_per_state"
                            ]
                        )
                    ]
                    if cell.layer == "fault"
                    else cell.logical_label
                ),
                "archive_chunk": cell.chunk_id,
                "rng_namespace": (
                    "NUMPY_SEEDSEQUENCE_ADDRESSED"
                    if cell.backend == "A"
                    else "BLAKE2B_ADDRESS_PYTHON_RANDOM_BOX_MULLER"
                ),
                "cluster_root_id": cluster_root_id(
                    config, cell, position
                ),
                "terminal_round": str(terminal),
                "fault_state_index": (
                    str(
                        position
                        // int(
                            config["formal_matrix"][
                                "fault_clusters_per_state"
                            ]
                        )
                    )
                    if cell.layer == "fault"
                    else ""
                ),
                "fault_within_state_index": (
                    str(
                        position
                        % int(
                            config["formal_matrix"][
                                "fault_clusters_per_state"
                            ]
                        )
                    )
                    if cell.layer == "fault"
                    else ""
                ),
            }
            for field, expected in expected_strings.items():
                if row.get(field) != expected:
                    raise RuntimeError(
                        f"resource ledger {field} drift at row {row_index}"
                    )
            if (
                row.get("conservation_pass") != "True"
                or row.get("exception_type") != ""
                or row.get("exception_message") != ""
            ):
                raise RuntimeError(
                    f"resource ledger terminal row status drift at {row_index}"
                )
            _strict_csv_integer(
                row,
                "density_index",
                expected=expected_density_index,
                row_number=row_index,
            )
            for field in finite_if_present:
                value = row.get(field)
                if value not in (None, ""):
                    try:
                        numeric = float(str(value))
                    except ValueError as exc:
                        raise RuntimeError(
                            f"resource ledger nonnumeric {field} at "
                            f"{row_index}"
                        ) from exc
                    if not math.isfinite(numeric):
                        raise RuntimeError(
                            f"resource ledger nonfinite {field} at "
                            f"{row_index}"
                        )
            for field in hash_fields:
                value = row.get(field)
                if value not in (None, "") and (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in hex_characters for character in value)
                ):
                    raise RuntimeError(
                        f"resource ledger hash {field} drift at {row_index}"
                    )
            if heldout_iq is not None:
                import numpy as np

                expected_heldout_hash = sha256(
                    np.asarray(
                        heldout_iq[row_index],
                        dtype="<f8",
                    ).tobytes(order="C")
                ).hexdigest()
                if row.get("heldout_window_sha256") != expected_heldout_hash:
                    raise RuntimeError(
                        "resource ledger heldout-IQ hash drift at "
                        f"{row_index}"
                    )
            row_id = row.get("row_id")
            if not isinstance(row_id, str) or not row_id or row_id in row_ids:
                raise RuntimeError(
                    f"resource ledger row_id coverage drift at row {row_index}"
                )
            row_ids.add(row_id)
            if global_row_ids is not None:
                if row_id in global_row_ids:
                    raise RuntimeError(
                        "resource ledger global row_id collision at "
                        f"{row_index}"
                    )
                global_row_ids.add(row_id)
            row_id_digest.update(row_id.encode("utf-8") + b"\0")
            if not (
                resource_start <= physical < resource_stop
                and resource_start <= heldout < resource_stop
            ):
                raise RuntimeError("resource ledger seed escaped resource interval")
            for name, start, stop in formal_intervals:
                if start <= physical < stop or start <= heldout < stop:
                    raise RuntimeError(
                        f"resource ledger accessed formal {name} seed interval"
                    )
            physical_min = (
                physical
                if physical_min is None
                else min(physical_min, physical)
            )
            physical_max = (
                physical
                if physical_max is None
                else max(physical_max, physical)
            )
            heldout_min = (
                heldout if heldout_min is None else min(heldout_min, heldout)
            )
            heldout_max = (
                heldout if heldout_max is None else max(heldout_max, heldout)
            )
    if observed != cell.expected_rows or len(row_ids) != cell.expected_rows:
        raise RuntimeError(
            "resource ledger frozen denominator/identity coverage drift"
        )
    return {
        "plan_index": cell.plan_index,
        "chunk_id": cell.chunk_id,
        "expected_rows": cell.expected_rows,
        "observed_rows": observed,
        "physical_seed_min": physical_min,
        "physical_seed_max": physical_max,
        "heldout_seed_min": heldout_min,
        "heldout_seed_max": heldout_max,
        "row_id_sequence_sha256": row_id_digest.hexdigest(),
        "formal_seed_addresses_accessed": False,
    }


def _validate_npy_payload(
    root: Path,
    binding: Mapping[str, Any],
    *,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    import numpy as np

    path = (root / str(binding["path"])).resolve()
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    try:
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise RuntimeError(
                f"resource NPY shape/dtype drift for {binding['role']}"
            )
        expected_file_bytes = int(value.offset) + int(value.nbytes)
        if path.stat().st_size != expected_file_bytes:
            raise RuntimeError(
                f"resource NPY trailing/truncated bytes for {binding['role']}"
            )
        role = str(binding["role"])
        if value.dtype.kind in {"f", "c"}:
            block = max(1, min(64, value.shape[0] if value.ndim else 1))
            if value.ndim == 0:
                finite = bool(np.isfinite(value))
            else:
                finite = all(
                    bool(np.all(np.isfinite(value[start:start + block])))
                    for start in range(0, value.shape[0], block)
                )
            if not finite:
                raise RuntimeError(
                    f"resource NPY nonfinite payload for {role}"
                )
        if role == "rb_success_probability_npy" and (
            bool(np.any(value < 0.0)) or bool(np.any(value > 1.0))
        ):
            raise RuntimeError("resource RB probability outside [0,1]")
        if role in {
            "rb_branch_trace_distance_npy",
            "rb_sampled_match_trace_distance_npy",
        } and bool(np.any(value < 0.0)):
            raise RuntimeError("resource RB distance is negative")
        if role == "rb_sampled_hidden_outcome_npy" and not bool(
            np.all((value == 0) | (value == 1))
        ):
            raise RuntimeError("resource RB hidden outcome domain drift")
        if role == "rb_sampled_reset_ack_npy" and not set(
            np.asarray(value).tolist()
        ).issubset({b"success", b"failure"}):
            raise RuntimeError("resource RB reset ack domain drift")
    finally:
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _expected_resource_object_specs(
    config: Mapping[str, Any],
    cell: T04CellSpec,
) -> dict[str, tuple[tuple[int, ...], str] | None]:
    rows = cell.expected_rows
    samples = cell.sample_count
    dimension = 3 * cell.cutoff
    iq_samples = int(config["resource_contract"]["expected_iq_samples"])
    reset_rows = _reset_events(config, cell)
    specifications: dict[
        str, tuple[tuple[int, ...], str] | None
    ] = {
        "round_ledger_csv": None,
        "raw_iq_npy": ((rows, iq_samples, 2), "<f8"),
        "heldout_iq_npy": ((rows, iq_samples, 2), "<f8"),
    }
    if cell.density_retention != "none":
        specifications["primary_density_npy"] = (
            (samples, dimension, dimension),
            "<c8",
        )
    if reset_rows:
        specifications.update(
            {
                "rb_valid_npy": ((reset_rows,), "?"),
                "rb_row_index_npy": ((reset_rows,), "<i8"),
                "rb_success_probability_npy": ((reset_rows,), "<f8"),
                "rb_success_present_npy": ((reset_rows,), "?"),
                "rb_failure_present_npy": ((reset_rows,), "?"),
                "rb_expected_density_npy": (
                    (reset_rows, dimension, dimension),
                    "<c8",
                ),
                "rb_conditional_success_density_npy": (
                    (reset_rows, dimension, dimension),
                    "<c8",
                ),
                "rb_conditional_failure_density_npy": (
                    (reset_rows, dimension, dimension),
                    "<c8",
                ),
                "rb_sampled_stress_density_npy": (
                    (reset_rows, dimension, dimension),
                    "<c8",
                ),
                "rb_sampled_hidden_outcome_npy": ((reset_rows,), "u1"),
                "rb_sampled_reset_ack_npy": ((reset_rows,), "S16"),
                "rb_branch_trace_distance_npy": ((reset_rows,), "<f8"),
                "rb_sampled_match_trace_distance_npy": (
                    (reset_rows,),
                    "<f8",
                ),
                "rb_pre_reset_receipt_npy": ((reset_rows,), "S64"),
            }
        )
    return specifications


def audit_resource_profile_receipts(
    root: Path,
    config: Mapping[str, Any],
    store: ImmutableObjectStore,
    cells: Sequence[T04CellSpec],
) -> dict[str, Any]:
    """Reopen all eight receipts and every one of 227,328 seed rows."""

    from cnn_fpga.benchmark.phase9_fresh_twin_qualification import (
        LEDGER_FIELDS,
    )
    from cnn_fpga.benchmark.phase9_powered_twin_qualification import (
        EXTRA_FIELDS,
        RUNNER_ID as QUALIFICATION_RUNNER_ID,
    )
    import numpy as np

    if QUALIFICATION_RUNNER_ID != RAW_RUNNER_ID:
        raise RuntimeError("resource/qualification raw runner identity drift")
    if [cell.plan_index for cell in cells] != [
        478, 480, 482, 484, 388, 389, 403, 507
    ]:
        raise RuntimeError("resource raw audit profile identity drift")
    expected_header = tuple(LEDGER_FIELDS) + tuple(EXTRA_FIELDS)
    receipt_evidence: list[dict[str, Any]] = []
    referenced_objects: set[Path] = set()
    total_rows = 0
    total_reset_rows = 0
    runtime_platform: str | None = None
    global_row_ids: set[str] = set()
    for cell in cells:
        receipt = store.verify_receipt(
            store.receipt_path(cell.chunk_id),
            expected_cell=asdict(cell),
        )
        fingerprint = receipt["runtime_fingerprint"]
        if (
            fingerprint["python"]
            != config["runtime_environment"]["python"]
            or fingerprint["numpy"]
            != config["runtime_environment"]["numpy"]
            or fingerprint["scipy"]
            != config["runtime_environment"]["scipy"]
            or fingerprint["psutil"]
            != config["runtime_environment"]["psutil"]
            or fingerprint["thread_environment"]
            != config["runtime_contract"]["preimport_thread_environment"]
            or fingerprint["python"] != list(sys.version_info[:3])
            or fingerprint["numpy"] != np.__version__
            or fingerprint["scipy"] != package_version("scipy")
            or fingerprint["psutil"] != package_version("psutil")
            or fingerprint["platform"] != platform.platform()
        ):
            raise RuntimeError(
                f"resource receipt numerical runtime drift for {cell.chunk_id}"
            )
        if runtime_platform is None:
            runtime_platform = str(fingerprint["platform"])
        elif fingerprint["platform"] != runtime_platform:
            raise RuntimeError("resource receipt platform drift across profiles")
        objects = receipt["objects"]
        by_role = {str(item["role"]): item for item in objects}
        expected_specs = _expected_resource_object_specs(config, cell)
        expected_reset = _reset_events(config, cell)
        if set(by_role) != set(expected_specs):
            raise RuntimeError(
                f"resource object role coverage drift for {cell.chunk_id}"
            )
        if cell.layer in {"shared", "probe"}:
            primary_binding = by_role.get("primary_density_npy")
            expected_binding = by_role.get("rb_expected_density_npy")
            if (
                primary_binding is None
                or expected_binding is None
                or primary_binding["path"] != expected_binding["path"]
                or primary_binding["sha256"]
                != expected_binding["sha256"]
                or primary_binding["bytes"] != expected_binding["bytes"]
            ):
                raise RuntimeError(
                    f"resource explicit expected-density alias drift for "
                    f"{cell.chunk_id}"
                )
        for role, specification in expected_specs.items():
            binding = by_role[role]
            expected_media = (
                "text/csv"
                if role == "round_ledger_csv"
                else "application/x-npy"
            )
            if binding.get("media_type") != expected_media:
                raise RuntimeError(
                    f"resource object media type drift for {role}"
                )
            referenced_objects.add(
                (root / str(binding["path"])).resolve()
            )
            if specification is not None:
                shape, dtype = specification
                _validate_npy_payload(
                    root,
                    binding,
                    shape=shape,
                    dtype=dtype,
                )
        if expected_reset:
            import numpy as np

            row_index_array = np.load(
                root / str(by_role["rb_row_index_npy"]["path"]),
                allow_pickle=False,
                mmap_mode="r",
            )
            valid_array = np.load(
                root / str(by_role["rb_valid_npy"]["path"]),
                allow_pickle=False,
                mmap_mode="r",
            )
            try:
                expected_reset_indices = [
                    position * cell.horizon + round_index
                    for position in range(cell.sample_count)
                    for round_index in range(cell.horizon)
                    if (
                        (
                            config["formal_matrix"][
                                "fault_action_sequences"
                            ][cell.scenario][
                                round_index
                                % len(
                                    config["formal_matrix"][
                                        "fault_action_sequences"
                                    ][cell.scenario]
                                )
                            ]
                            if cell.layer == "fault"
                            else cell.action
                        )
                        == "RESET"
                    )
                ]
                if (
                    not np.array_equal(
                        np.asarray(row_index_array, dtype=np.int64),
                        np.asarray(expected_reset_indices, dtype=np.int64),
                    )
                    or not bool(np.all(valid_array))
                ):
                    raise RuntimeError(
                        f"resource RB row-index/valid coverage drift for "
                        f"{cell.chunk_id}"
                    )
            finally:
                for array in (row_index_array, valid_array):
                    mapping = getattr(array, "_mmap", None)
                    if mapping is not None:
                        mapping.close()
        heldout_array = np.load(
            root / str(by_role["heldout_iq_npy"]["path"]),
            allow_pickle=False,
            mmap_mode="r",
        )
        try:
            ledger = _audit_resource_seed_ledger(
                root / str(by_role["round_ledger_csv"]["path"]),
                config=config,
                cell=cell,
                expected_header=expected_header,
                heldout_iq=heldout_array,
                global_row_ids=global_row_ids,
            )
        finally:
            mapping = getattr(heldout_array, "_mmap", None)
            if mapping is not None:
                mapping.close()
        diagnostics = receipt["diagnostics"]
        if (
            diagnostics["expected_rows"] != cell.expected_rows
            or diagnostics["observed_rows"] != cell.expected_rows
            or diagnostics["exception_rows"] != 0
            or diagnostics["missing_rows"] != 0
            or diagnostics["conservation_failures"] != 0
            or diagnostics["reset_rows"] != expected_reset
            or diagnostics["reset_sidecar_rows"] != expected_reset
        ):
            raise RuntimeError(
                f"resource receipt denominator drift for {cell.chunk_id}"
            )
        total_rows += cell.expected_rows
        total_reset_rows += expected_reset
        receipt_evidence.append(
            {
                **ledger,
                "reset_rows": expected_reset,
                "receipt_sha256": receipt["receipt_sha256"],
                "object_role_count": len(objects),
                "object_roles": sorted(by_role),
            }
        )
    live_objects = {
        path.resolve()
        for path in store.object_root.rglob("*")
        if path.is_file()
    }
    if live_objects != referenced_objects:
        raise RuntimeError("resource object tree has orphan or unreferenced bytes")
    if any(path.is_file() for path in store.staging_root.rglob("*")):
        raise RuntimeError("resource staging is not empty after receipt commit")
    if total_rows != 227_328 or total_reset_rows != 15_360:
        raise RuntimeError("resource profile aggregate denominator drift")
    if len(global_row_ids) != total_rows:
        raise RuntimeError("resource profile global row identity drift")
    evidence: dict[str, Any] = {
        "schema_version": RAW_SEED_AUDIT_SCHEMA,
        "profile_plan_indices": [cell.plan_index for cell in cells],
        "ledger_column_count": len(expected_header),
        "ledger_header_sha256": _sha(list(expected_header)),
        "expected_rows": 227_328,
        "observed_rows": total_rows,
        "expected_reset_rows": 15_360,
        "observed_reset_rows": total_reset_rows,
        "receipt_count": len(receipt_evidence),
        "receipts": receipt_evidence,
        "runtime_platform": runtime_platform,
        "object_tree_exactly_receipt_referenced": True,
        "staging_empty": True,
        "formal_seed_addresses_accessed": False,
        "qualified_claim": None,
        "scientific_verdict": None,
    }
    evidence["analysis_sha256"] = _sha(evidence)
    return evidence


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
        # Publish the initial stage synchronously.  A daemon thread may not be
        # scheduled before the supervisor advances state, which would make a
        # genuine run nondeterministically lose its ``starting`` witness.
        self.sample_once()
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
    by_role = {str(binding["role"]): binding for binding in objects}
    explicit_alias_bytes = 0
    primary = by_role.get("primary_density_npy")
    expected = by_role.get("rb_expected_density_npy")
    if (
        receipt["cell"].get("layer") in {"shared", "probe"}
        and primary is not None
        and expected is not None
        and primary["path"] == expected["path"]
        and primary["sha256"] == expected["sha256"]
        and primary["bytes"] == expected["bytes"]
    ):
        explicit_alias_bytes = int(expected["bytes"])
    conservative_payload_bytes = (
        sum(int(binding["bytes"]) for binding in objects)
        - explicit_alias_bytes
    )
    return {
        "chunk_id": receipt["cell"]["chunk_id"],
        "plan_index": int(receipt["cell"]["plan_index"]),
        "pid": int(result["pid"]),
        "wall_seconds": float(result["wall_seconds"]),
        "object_bytes_unique": sum(unique.values()),
        "explicit_alias_bytes": explicit_alias_bytes,
        "conservative_payload_bytes": conservative_payload_bytes,
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
    """Time a frozen fixture and conservatively project the formal full audit."""

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
        or specification["coverage_mode"]
        != "fixture_timed_conservative_projection"
        or specification["resource_profile_is_full_coverage"] is not False
        or specification["formal_full_coverage_required"] is not True
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
        "coverage_mode": "fixture_timed_conservative_projection",
        "resource_profile_is_full_coverage": False,
        "formal_full_coverage_required": True,
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
                and physicality.get("coverage_mode")
                == "fixture_timed_conservative_projection"
                and physicality.get("resource_profile_is_full_coverage")
                is False
                and physicality.get("formal_full_coverage_required") is True
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
    inventory_finalize_wall_seconds: float,
    inventory_profile_object_bytes: int,
    inventory_profile_receipt_count: int,
) -> dict[str, Any]:
    """Project by layer/backend and structural components, never one row ratio."""

    measured = {int(item["plan_index"]): item for item in measurements}
    expected_measured = {388, 389, 403, 478, 480, 482, 484, 507}
    required = {388, 389, 403, 484, 507}
    if set(measured) != expected_measured:
        raise RuntimeError(
            "projection requires all eight frozen resource profiles"
        )
    backend_a_wall_factor = max(
        1.0,
        float(measured[388]["wall_seconds"])
        / max(1.0e-12, float(measured[389]["wall_seconds"])),
    )
    backend_a_byte_factor = max(
        1.0,
        float(measured[388]["conservative_payload_bytes"])
        / max(
            1.0,
            float(measured[389]["conservative_payload_bytes"]),
        ),
    )

    def representative(cell: T04CellSpec) -> int:
        if cell.layer == "fault":
            return 484
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

    def mapping_anchor_bytes(cell: T04CellSpec) -> int:
        anchors = config["formal_matrix"]["mapping_anchor_plan_indices"]
        if anchors.get(str(cell.cutoff)) != cell.plan_index:
            return 0
        import numpy as np

        total = 0
        for shape in (
            (cell.cutoff, 2),
            (cell.cutoff, 2),
            (cell.cutoff, cell.cutoff),
            (cell.cutoff, cell.cutoff),
        ):
            buffer = io.BytesIO()
            np.save(
                buffer,
                np.zeros(shape, dtype=np.complex128),
                allow_pickle=False,
            )
            total += len(buffer.getbuffer())
        return total

    def component_bytes(
        rep: Mapping[str, Any],
        rep_cell: T04CellSpec,
    ) -> tuple[int, int, int, int, int]:
        bindings = rep.get("object_bindings")
        if isinstance(bindings, list) and bindings:
            by_role: dict[str, Mapping[str, Any]] = {}
            for binding in bindings:
                role = str(binding["role"])
                if role in by_role:
                    raise RuntimeError("measured duplicate object role")
                by_role[role] = binding
            alias_roles: set[str] = set()
            primary_binding = by_role.get("primary_density_npy")
            expected_binding = by_role.get("rb_expected_density_npy")
            if (
                rep_cell.layer in {"shared", "probe"}
                and primary_binding is not None
                and expected_binding is not None
                and primary_binding["sha256"] == expected_binding["sha256"]
                and int(primary_binding["bytes"])
                == int(expected_binding["bytes"])
            ):
                alias_roles.add("rb_expected_density_npy")
            totals = {
                "row": 0,
                "primary": 0,
                "reset_scalar": 0,
                "reset_density": 0,
                "other": 0,
            }
            for role, entry in by_role.items():
                if role in alias_roles:
                    continue
                size = int(entry["bytes"])
                if role in row_roles:
                    totals["row"] += size
                elif role == "primary_density_npy":
                    totals["primary"] += size
                elif role.startswith("rb_") and "density" in role:
                    totals["reset_density"] += size
                elif role.startswith("rb_"):
                    totals["reset_scalar"] += size
                else:
                    totals["other"] += size
            if sum(totals.values()) != int(
                rep["conservative_payload_bytes"]
            ):
                raise RuntimeError(
                    "component projection conservative byte accounting drift"
                )
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
        explicit_alias_bytes = int(rep.get("explicit_alias_bytes", 0))
        reset_density = max(0, reset_density - explicit_alias_bytes)
        conservative_payload_bytes = int(
            rep.get(
                "conservative_payload_bytes",
                sum(int(size) for size in roles.values())
                - explicit_alias_bytes,
            )
        )
        other = max(
            0,
            conservative_payload_bytes
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
        ) = component_bytes(rep, rep_cell)
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
        anchor_bytes = mapping_anchor_bytes(cell)
        cell_bytes += anchor_bytes
        if cell.backend == "A" and rep_cell.backend != "A":
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
        if cell.backend == "A" and rep_cell.backend != "A":
            cell_wall *= backend_a_wall_factor
        projected_cell_bytes = int(math.ceil(cell_bytes))
        projected_bytes += cell_bytes
        projected_worker_wall += cell_wall
        cell_projections.append(
            {
                "plan_index": cell.plan_index,
                "chunk_id": cell.chunk_id,
                "representative_plan_index": rep_index,
                "projected_mapping_anchor_bytes": anchor_bytes,
                "projected_object_bytes": projected_cell_bytes,
                # Every object is first written below staging before atomic
                # adoption.  At cell granularity the conservative payload is
                # therefore also the transient disk requirement.  Keeping a
                # distinct field prevents persistent bytes from being
                # silently substituted for concurrent staging admission.
                "projected_transient_bytes": projected_cell_bytes,
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
    # Formal production schedules the frozen cells by descending projected
    # cost.  Replay that exact deterministic LPT policy instead of the
    # impossible ``sum/max_workers`` lower bound: heterogeneous cells can
    # never finish before the longest assigned worker load.
    worker_loads = [0.0] * max_workers
    for item in sorted(
        cell_projections,
        key=lambda value: (
            -float(value["projected_wall_seconds"]),
            int(value["plan_index"]),
        ),
    ):
        worker = min(
            range(max_workers),
            key=lambda index: (worker_loads[index], index),
        )
        worker_loads[worker] += float(item["projected_wall_seconds"])
    projected_raw_lpt_wall = max(worker_loads, default=0.0)
    projected_artifact_bytes = sum(
        int(item["projected_object_bytes"])
        for item in cell_projections
    )
    inventory_wall = float(inventory_finalize_wall_seconds)
    inventory_bytes = int(inventory_profile_object_bytes)
    inventory_receipts = int(inventory_profile_receipt_count)
    if (
        not math.isfinite(inventory_wall)
        or inventory_wall <= 0.0
        or inventory_bytes <= 0
        or inventory_receipts != 8
    ):
        raise RuntimeError("inventory finalize wall projection invalid")
    inventory_scale = max(
        projected_artifact_bytes / inventory_bytes,
        len(cells) / inventory_receipts,
    )
    projected_inventory_wall = inventory_wall * inventory_scale
    projected_wall = (
        projected_raw_lpt_wall
        + float(stats_wall_seconds)
        + physicality_wall
        + projected_inventory_wall
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
        "projected_formal_artifact_bytes": projected_artifact_bytes,
        "projected_formal_worker_wall_seconds": projected_worker_wall,
        "projected_raw_lpt_worker_load_seconds": worker_loads,
        "projected_raw_lpt_wall_seconds": projected_raw_lpt_wall,
        "projected_formal_wall_seconds_at_frozen_concurrency": projected_wall,
        "frozen_concurrency": max_workers,
        "statistics_wall_seconds": float(stats_wall_seconds),
        "retained_density_physicality_serial_wall_seconds": (
            physicality_wall
        ),
        "inventory_profile_finalize_wall_seconds": inventory_wall,
        "inventory_profile_object_bytes": inventory_bytes,
        "inventory_profile_receipt_count": inventory_receipts,
        "inventory_finalize_projection_scale": inventory_scale,
        "projected_inventory_finalize_wall_seconds": (
            projected_inventory_wall
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
    maximum_inflight_temp_bytes: int = 0,
    analysis_scratch_bytes: int = 0,
) -> dict[str, Any]:
    contract = config["resource_contract"]
    free = int(shutil.disk_usage(run_directory).free)
    projected_bytes = int(projection["projected_formal_artifact_bytes"])
    maximum_inflight_temp_bytes = int(maximum_inflight_temp_bytes)
    analysis_scratch_bytes = int(analysis_scratch_bytes)
    if maximum_inflight_temp_bytes < 0 or analysis_scratch_bytes < 0:
        raise ValueError("resource scratch/inflight bytes must be nonnegative")
    post_projection_free = (
        free
        - projected_bytes
        - maximum_inflight_temp_bytes
        - analysis_scratch_bytes
    )
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
            and int(inventory["receipt_count"]) == 8
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
        "maximum_inflight_temp_bytes": maximum_inflight_temp_bytes,
        "analysis_scratch_bytes": analysis_scratch_bytes,
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
        if self.sample_interval_seconds != float(
            self.config["resource_contract"]["sample_interval_seconds"]
        ):
            raise RuntimeError("resource sampling interval contract drift")
        cells = build_cell_plan(self.config)
        formal_peak, representatives = profile_cells(self.config, cells)
        selected = formal_peak + representatives
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
        owner_identity = owner.acquire()
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
        # Keep a stable, fail-closed snapshot of every stage that may have
        # completed before a later resource gate rejects the run.  These
        # values are deliberately initialized before the guarded execution so
        # early failures remain explicit instead of being mistaken for absent
        # reporting, while late failures preserve the already-computed
        # projection and gate decision for independent diagnosis.
        lineage: dict[str, Any] | None = None
        seed_firewall: dict[str, Any] | None = None
        measurements: list[dict[str, Any]] = []
        stats: dict[str, Any] | None = None
        raw_seed_audit: dict[str, Any] | None = None
        inventory: dict[str, Any] | None = None
        inventory_evidence: dict[str, Any] | None = None
        inventory_binding: dict[str, Any] | None = None
        projection: dict[str, Any] | None = None
        decision: dict[str, Any] | None = None
        maximum_inflight_temp_bytes: int | None = None
        analysis_scratch_bytes: int | None = None
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
                    "owner_token": owner_identity.owner_token,
                    "owner_pid": owner_identity.pid,
                    "process_creation_time": (
                        owner_identity.process_creation_time
                    ),
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

            state["stage"] = "formal_lpt_four_worker_peak"
            sampler.sample_once()
            measurements = self.process_group_runner(
                [kwargs_for(cell) for cell in formal_peak],
                active_pids=active_pids,
                active_lock=active_lock,
                sample_callback=sampler.sample_once,
            )
            sampler.sample_once()
            concurrent_peak = sampler.summary()[
                "stage_peak_aggregate_rss_bytes"
            ].get("formal_lpt_four_worker_peak", 0)
            for measurement in measurements:
                measurement["profile_peak_aggregate_rss_bytes"] = concurrent_peak
            state["profiles_completed"] = len(measurements)
            state["stage"] = "representative_four_worker_profiles"
            sampler.sample_once()
            measurements.extend(
                self.process_group_runner(
                    [kwargs_for(cell) for cell in representatives],
                    active_pids=active_pids,
                    active_lock=active_lock,
                    sample_callback=sampler.sample_once,
                )
            )
            sampler.sample_once()
            singleton_peak = sampler.summary()[
                "stage_peak_aggregate_rss_bytes"
            ].get("representative_four_worker_profiles", 0)
            for measurement in measurements:
                if int(measurement["plan_index"]) in {
                    cell.plan_index for cell in representatives
                }:
                    measurement["profile_peak_aggregate_rss_bytes"] = singleton_peak
            state["profiles_completed"] = len(measurements)
            state["stage"] = "joint_maxt_3037x199"
            sampler.sample_once()
            stats = self.stats_runner(
                self.config,
                sample_callback=sampler.sample_once,
            )
            sampler.sample_once()
            validate_statistics_profile(self.config, stats)
            stats["profile_peak_aggregate_rss_bytes"] = sampler.summary()[
                "stage_peak_aggregate_rss_bytes"
            ].get("joint_maxt_3037x199", 0)
            stats.pop("analysis_sha256", None)
            stats["analysis_sha256"] = _sha(stats)
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
                source_snapshot_sha256=self.source_snapshot_sha256,
                seed_namespace="resource_preflight",
                runner_id=RAW_RUNNER_ID,
            )
            raw_seed_audit = audit_resource_profile_receipts(
                root,
                self.config,
                store,
                selected,
            )
            inventory, inventory_evidence = no_copy_inventory(store, selected)
            inventory_path = preflight_root / "inventory.json"
            _immutable_json(inventory_path, inventory)
            inventory_binding = _json_binding(root, inventory_path)
            # Ensure at least one final sample sees the post-finalize parent.
            sampler.sample_once()
            # Likewise freeze a final heartbeat at the exact finalize stage;
            # a periodic heartbeat alone may still describe the preceding
            # statistics stage when finalize completes in under one period.
            heartbeat.write_once()
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
            # ``streaming_statistics_dry_run.wall_seconds`` starts after the
            # retained-density timing fixture completes.  Subtracting the
            # fixture again would understate the statistics stage.
            statistics_wall = float(stats["wall_seconds"])
            if statistics_wall <= 0.0:
                raise RuntimeError("statistics wall is not positive")
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
                inventory_finalize_wall_seconds=float(
                    inventory_evidence["finalize_wall_seconds"]
                ),
                inventory_profile_object_bytes=int(
                    inventory["totals"]["object_bytes_unique"]
                ),
                inventory_profile_receipt_count=int(
                    inventory["receipt_count"]
                ),
            )
            maximum_inflight_temp_bytes = sum(
                sorted(
                    (
                        int(item["projected_transient_bytes"])
                        for item in projection["cell_projections"]
                    ),
                    reverse=True,
                )[: int(self.config["runtime_contract"]["max_workers"])]
            )
            analysis_scratch_bytes = int(
                stats["peak_analysis_scratch_bytes"]
            )
            decision = resource_gate_decision(
                self.config,
                sampling=sampling,
                projection=projection,
                inventory=inventory,
                run_directory=preflight_root,
                maximum_inflight_temp_bytes=(
                    maximum_inflight_temp_bytes
                ),
                analysis_scratch_bytes=analysis_scratch_bytes,
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
                "sample_interval_seconds": self.sample_interval_seconds,
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
                "raw_seed_audit": raw_seed_audit,
                "projection": projection,
                "cell_projections": projection["cell_projections"],
                "maximum_inflight_temp_bytes": (
                    maximum_inflight_temp_bytes
                ),
                "analysis_scratch_bytes": analysis_scratch_bytes,
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
                "completed_stage_evidence": {
                    "lineage_validation": lineage,
                    "seed_firewall": seed_firewall,
                    # Preserve the already-returned process-group order.  Do
                    # not perform additional validation while serializing an
                    # exception: a malformed partial measurement must not
                    # mask the original fail-closed cause.
                    "profile_measurements": list(measurements),
                    "streaming_statistics_dry_run": stats,
                    "raw_seed_audit": raw_seed_audit,
                    "inventory": inventory,
                    "inventory_binding": inventory_binding,
                    "inventory_no_copy_evidence": inventory_evidence,
                    "projection": projection,
                    "resource_gate_decision": decision,
                    "maximum_inflight_temp_bytes": (
                        maximum_inflight_temp_bytes
                    ),
                    "analysis_scratch_bytes": analysis_scratch_bytes,
                },
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
    "RAW_RUNNER_ID",
    "RAW_SEED_AUDIT_SCHEMA",
    "ResourcePreflightFailure",
    "ResourcePreflightSupervisor",
    "ResourceSampler",
    "assert_seed_firewall",
    "audit_resource_profile_receipts",
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
