"""Fresh powered T04 raw-evidence runner.

The runner writes one content-addressed transaction per frozen cell and never
computes a scientific verdict.  It fixes three simplifications in the legacy
runner: state-major six-state fault sampling, explicit logical evaluator carry,
and Rao--Blackwell expected RESET continuation in every reset scope.

Large arrays are streamed directly to ``.npy`` memmaps in the same-volume
staging area, then atomically adopted by the immutable object store.  No
monolithic ZIP and no merged 2,085,888-row CSV are created.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
from dataclasses import asdict, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from threading import Lock
import time
from typing import Any, Iterable, Mapping

import numpy as np
import psutil

from cnn_fpga.benchmark.phase9_immutable_object_store import (
    ImmutableObjectStore,
    ObjectBinding,
    append_attempt_event,
    publish_inventory_and_manifest,
)
from cnn_fpga.benchmark.phase9_powered_twin_contract import (
    EXPECTED_CLAIM_FIELDS,
    T04CellSpec,
    build_cell_plan,
    cluster_root_id,
    heldout_seed,
    load_config,
    physical_seed,
    plan_payload,
    runtime_source_snapshot,
)
from cnn_fpga.benchmark.phase9_powered_twin_runtime import (
    HeartbeatService,
    OwnerLease,
    ResourceWatchdog,
)


RUNNER_ID = "PHASE9-POWERED-TWIN-RAW-RUNNER-V1"
ROW_SCHEMA = "PHASE9-POWERED-TWIN-ROUND-LEDGER-V1"
RB_SIDECAR_SCHEMA = "PHASE9-POWERED-ALL-SCOPE-RB-SIDECAR-V1"
EXTRA_FIELDS = (
    "cluster_root_id",
    "physical_seed_address",
    "heldout_seed_address",
    "primary_reset_estimand",
    "sampled_reset_nonvoting",
    "pre_reset_causal_receipt_sha256",
    "fault_state_index",
    "fault_within_state_index",
    "pre_intervention_state_sha256",
    "intervention_delta_sha256",
    "intervention_applied",
    "pre_intervention_drift_0",
    "pre_intervention_drift_1",
    "pre_intervention_drift_2",
    "pre_intervention_drift_3",
    "pre_intervention_drift_4",
    "input_intervention_drift_0",
    "input_intervention_drift_1",
    "input_intervention_drift_2",
    "input_intervention_drift_3",
    "input_intervention_drift_4",
    "pre_intervention_non_drift_state_sha256",
    "input_non_drift_state_sha256",
    "intervention_application_receipt_sha256",
    "input_state_sha256",
    "input_evaluator_sha256",
    "output_state_sha256",
    "output_evaluator_sha256",
    "expected_reset_ancestor_receipt_sha256",
)
PRODUCTION_SUPERVISOR_ID = "PHASE9-POWERED-TWIN-PRODUCTION-SUPERVISOR-V1"


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _semantic_object_digest(value: object | None) -> str:
    """Stable recursive digest for evaluator/frame provenance."""

    digest = sha256()

    def update(item: object) -> None:
        if item is None:
            digest.update(b"N;")
        elif isinstance(item, Enum):
            digest.update(b"E")
            update(item.value)
        elif isinstance(item, (bool, int, str)):
            digest.update(type(item).__name__.encode("ascii") + b":")
            digest.update(str(item).encode("utf-8") + b";")
        elif isinstance(item, float):
            digest.update(b"F" + float(item).hex().encode("ascii") + b";")
        elif isinstance(item, complex):
            digest.update(
                b"C"
                + float(item.real).hex().encode("ascii")
                + b","
                + float(item.imag).hex().encode("ascii")
                + b";"
            )
        elif isinstance(item, np.generic):
            update(item.item())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(
                b"A"
                + array.dtype.str.encode("ascii")
                + b":"
                + ",".join(str(value) for value in array.shape).encode("ascii")
                + b":"
            )
            digest.update(array.tobytes(order="C"))
        elif is_dataclass(item):
            digest.update(
                b"D" + type(item).__qualname__.encode("utf-8") + b"{"
            )
            for field in fields(item):
                digest.update(field.name.encode("utf-8") + b"=")
                update(getattr(item, field.name))
            digest.update(b"}")
        elif isinstance(item, Mapping):
            digest.update(b"M{")
            for key in sorted(item, key=lambda value: str(value)):
                update(key)
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(b"T[" if isinstance(item, tuple) else b"L[")
            for child in item:
                update(child)
            digest.update(b"]")
        elif hasattr(item, "__dict__"):
            digest.update(
                b"O" + type(item).__qualname__.encode("utf-8") + b"{"
            )
            update(vars(item))
            digest.update(b"}")
        else:
            raise TypeError(
                f"unsupported evaluator digest type {type(item).__qualname__}"
            )

    update(value)
    return digest.hexdigest()


def _state_digest(state: object | None) -> str:
    if state is None:
        return ""
    digest = sha256()
    density = np.asarray(state.joint_density, dtype="<c16")
    digest.update(density.tobytes(order="C"))
    digest.update(
        json.dumps(
            {
                "cutoff": int(state.cutoff),
                "round_index": int(state.round_index),
                "leakage_age": int(state.leakage_age),
                "drift": [float(value) for value in state.drift.vector()],
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _state_non_drift_digest(state: object | None) -> str:
    """Digest the state fields which an intervention must not mutate."""

    if state is None:
        return ""
    digest = sha256()
    digest.update(b"PHASE9-INTERVENTION-NON-DRIFT-STATE-V1\0")
    density = np.asarray(state.joint_density, dtype="<c16")
    digest.update(density.tobytes(order="C"))
    digest.update(
        json.dumps(
            {
                "cutoff": int(state.cutoff),
                "round_index": int(state.round_index),
                "leakage_age": int(state.leakage_age),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _activate_execution_modules(
    root: Path,
) -> tuple[dict[str, Any], Any, Any, Any]:
    """Activate the byte-verified c44 runtime before importing physics users."""

    repair = importlib.import_module(
        "cnn_fpga.benchmark.phase9_cutoff36_44_repair"
    )
    repair_config, base = repair.load_config(root)
    repair._activate_verified_modules(root, repair_config)
    repair._assert_verified_modules()
    runner = importlib.import_module(
        "cnn_fpga.benchmark.phase9_fresh_twin_qualification"
    )
    dual = importlib.import_module(
        "cnn_fpga.benchmark.phase9_dual_backend_qualification"
    )
    powered_name = "physics.phase9_reset_rao_blackwell_powered"
    stale_powered = sys.modules.pop(powered_name, None)
    physics_package = sys.modules.get("physics")
    if (
        stale_powered is not None
        and physics_package is not None
        and getattr(
            physics_package,
            "phase9_reset_rao_blackwell_powered",
            None,
        )
        is stale_powered
    ):
        delattr(physics_package, "phase9_reset_rao_blackwell_powered")
    # Import a fresh additive wrapper only after the byte-verified T03 runtime
    # is active.  Pre-collected test functions keep their old, internally
    # consistent module graph; production receives the verified graph.
    powered_reset = importlib.import_module(powered_name)
    return base, runner, dual, powered_reset


def _execution_config(
    base: Mapping[str, Any],
    t04: Mapping[str, Any],
) -> dict[str, Any]:
    value = json.loads(json.dumps(base))
    matrix = value["formal_matrix"]
    frozen = t04["formal_matrix"]
    matrix["same_cutoff_ab"] = list(frozen["cutoffs"])
    matrix["cutoff_ladder"] = list(frozen["cutoffs"])
    matrix["primary_cutoff_increments"] = list(
        frozen["primary_cutoff_increments"]
    )
    matrix["round_sample_count"] = int(frozen["round_clusters_per_cell"])
    matrix["trajectory_sample_count"] = int(
        frozen["aggregate_fault_clusters_per_cell"]
    )
    matrix["fault_scenarios"] = json.loads(
        json.dumps(frozen["fault_scenario_parameters"])
    )
    matrix["fault_logical_label_schedule"] = list(frozen["logical_labels"])
    return value


def _heldout_window(
    config: Mapping[str, Any],
    t04: Mapping[str, Any],
    cell: T04CellSpec,
    position: int,
    round_index: int,
    *,
    seed_namespace: str,
) -> tuple[int, np.ndarray]:
    if seed_namespace == "formal":
        seed = heldout_seed(t04, cell, position, round_index)
    else:
        registry = t04["seed_registry"]
        namespace = registry[seed_namespace]
        maximum = int(registry["maximum_cluster_positions"])
        horizon = int(registry["maximum_horizon"])
        seed = (
            int(namespace["start"])
            + int(namespace["heldout_offset"])
            + cell.pair_group_index * maximum * horizon
            + position * horizon
            + round_index
        )
        if seed >= int(namespace["start"]) + int(namespace["count"]):
            raise RuntimeError("preflight heldout seed escaped its namespace")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x704]))
    readout = config["readout_semantics"]
    prior = np.asarray(readout["heldout_component_prior"], dtype=np.float64)
    centers = np.asarray(readout["heldout_centers"], dtype=np.float64)
    sigma = float(readout["heldout_sigma"])
    component = int(rng.choice(3, p=prior))
    count = int(config["common_physics"]["iq_samples"])
    window = centers[component] + sigma * rng.standard_normal((count, 2))
    return seed, np.asarray(window, dtype=np.float64)


def _fault_label(
    config: Mapping[str, Any],
    position: int,
) -> tuple[str, int, int]:
    per_state = int(config["formal_matrix"]["fault_clusters_per_state"])
    state_index, within = divmod(position, per_state)
    labels = config["formal_matrix"]["logical_labels"]
    return str(labels[state_index]), state_index, within


def _next_evaluator(result: object, backend: str) -> object | None:
    if result.logical is None:
        return None
    return (
        result.logical.evaluator_state
        if backend == "A"
        else result.logical.evaluator
    )


def _rb_event_count(config: Mapping[str, Any], cell: T04CellSpec) -> int:
    if cell.reset_estimand_scope == "none":
        return 0
    if cell.layer != "fault":
        return cell.sample_count
    sequence = config["formal_matrix"]["fault_action_sequences"][cell.scenario]
    resets_per_trajectory = sum(
        sequence[index % len(sequence)] == "RESET"
        for index in range(cell.horizon)
    )
    return cell.sample_count * resets_per_trajectory


def _new_memmap(
    store: ImmutableObjectStore,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | str,
    *,
    registry: list[np.memmap] | None = None,
) -> tuple[Path, np.memmap]:
    path = store.new_staging_path(suffix=".npy")
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=shape,
    )
    if registry is not None:
        registry.append(array)
    return path, array


def _close_memmap(array: np.memmap) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not bool(getattr(mapping, "closed", False)):
        array.flush()
        mapping.flush()
        mapping.close()


def _identity(
    runner: Any,
    cell: T04CellSpec,
    *,
    seed: int,
    position: int,
    row_index: int,
    action: str,
    round_index: int = 0,
    terminal_round: bool = True,
    logical_label: str | None = None,
) -> dict[str, object]:
    value = runner._identity(
        cell,
        seed=seed,
        position=position,
        row_index=row_index,
        action=action,
        round_index=round_index,
        terminal_round=terminal_round,
        logical_label=logical_label,
    )
    value["row_schema"] = ROW_SCHEMA
    return value


def _augment_row(
    row: dict[str, object],
    *,
    t04: Mapping[str, Any],
    cell: T04CellSpec,
    position: int,
    physical: int,
    heldout: int,
    reset_evidence: object | None,
    fault_state_index: int | str = "",
    fault_within_state_index: int | str = "",
    pre_intervention_state_sha256: str = "",
    intervention_evidence: Mapping[str, object],
    input_state_sha256: str = "",
    input_evaluator_sha256: str = "",
    output_state_sha256: str = "",
    output_evaluator_sha256: str = "",
    expected_reset_ancestor_receipt_sha256: str = "",
) -> dict[str, object]:
    row.update(
        {
            "cluster_root_id": cluster_root_id(t04, cell, position),
            "physical_seed_address": physical,
            "heldout_seed_address": heldout,
            "primary_reset_estimand": (
                "RAO_BLACKWELLIZED_EXPECTED_POST_RESET_DENSITY_AND_LEVELS_V1"
                if reset_evidence is not None
                else ""
            ),
            "sampled_reset_nonvoting": reset_evidence is not None,
            "pre_reset_causal_receipt_sha256": (
                reset_evidence.pre_reset_causal_receipt_sha256
                if reset_evidence is not None
                else ""
            ),
            "fault_state_index": fault_state_index,
            "fault_within_state_index": fault_within_state_index,
            "pre_intervention_state_sha256": pre_intervention_state_sha256,
            "input_state_sha256": input_state_sha256,
            "input_evaluator_sha256": input_evaluator_sha256,
            "output_state_sha256": output_state_sha256,
            "output_evaluator_sha256": output_evaluator_sha256,
            "expected_reset_ancestor_receipt_sha256": (
                expected_reset_ancestor_receipt_sha256
            ),
        }
    )
    evidence_fields = set(intervention_evidence)
    required_fields = set(_INTERVENTION_EVIDENCE_FIELDS)
    if evidence_fields != required_fields:
        raise RuntimeError(
            "intervention evidence schema mismatch "
            f"missing={sorted(required_fields - evidence_fields)} "
            f"unexpected={sorted(evidence_fields - required_fields)}"
        )
    row.update(intervention_evidence)
    return row


def _intervention_witness(delta: object) -> tuple[str, bool]:
    """Return the canonical byte witness and application predicate."""

    canonical = np.asarray(delta, dtype="<f8")
    if canonical.shape != (5,):
        raise RuntimeError(
            "fault intervention delta must have canonical shape (5,)"
        )
    if not np.all(np.isfinite(canonical)):
        raise RuntimeError("fault intervention delta must be finite")
    canonical = np.ascontiguousarray(canonical)
    return (
        sha256(canonical.tobytes(order="C")).hexdigest(),
        bool(np.any(canonical != 0.0)),
    )


ZERO_INTERVENTION_DELTA_SHA256, _ZERO_INTERVENTION_APPLIED = (
    _intervention_witness(np.zeros(5, dtype="<f8"))
)
if _ZERO_INTERVENTION_APPLIED:
    raise RuntimeError("zero intervention witness classified as applied")


_PRE_DRIFT_FIELDS = tuple(
    f"pre_intervention_drift_{index}" for index in range(5)
)
_INPUT_DRIFT_FIELDS = tuple(
    f"input_intervention_drift_{index}" for index in range(5)
)
_INTERVENTION_EVIDENCE_FIELDS = (
    "intervention_delta_sha256",
    "intervention_applied",
    *_PRE_DRIFT_FIELDS,
    *_INPUT_DRIFT_FIELDS,
    "pre_intervention_non_drift_state_sha256",
    "input_non_drift_state_sha256",
    "intervention_application_receipt_sha256",
)
_APPLICATION_RECEIPT_SCHEMA = (
    "PHASE9-INTERVENTION-APPLICATION-RECEIPT-V1"
)


def _canonical_application_receipt(
    row: Mapping[str, object],
) -> str:
    def vector_hex(names: tuple[str, ...]) -> list[str]:
        return [float(row[name]).hex() for name in names]

    payload = {
        "schema": _APPLICATION_RECEIPT_SCHEMA,
        "row_id": str(row["row_id"]),
        "scenario": str(row["scenario"]),
        "round_index": int(row["round_index"]),
        "intervention_delta_sha256": str(
            row["intervention_delta_sha256"]
        ),
        "intervention_applied": _canonical_csv_bool(
            row["intervention_applied"]
        ),
        "pre_intervention_drift_hex": vector_hex(_PRE_DRIFT_FIELDS),
        "input_intervention_drift_hex": vector_hex(
            _INPUT_DRIFT_FIELDS
        ),
        "pre_intervention_non_drift_state_sha256": str(
            row["pre_intervention_non_drift_state_sha256"]
        ),
        "input_non_drift_state_sha256": str(
            row["input_non_drift_state_sha256"]
        ),
    }
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_csv_bool(value: object) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise RuntimeError(f"non-canonical intervention bool {value!r}")


def _validate_intervention_application_fields(
    row: Mapping[str, object],
    expected_delta: object,
) -> None:
    delta = np.asarray(expected_delta, dtype="<f8")
    if delta.shape != (5,) or not np.all(np.isfinite(delta)):
        raise RuntimeError("expected intervention delta must be finite shape (5,)")
    pre = np.asarray(
        [float(row[name]) for name in _PRE_DRIFT_FIELDS],
        dtype="<f8",
    )
    input_vector = np.asarray(
        [float(row[name]) for name in _INPUT_DRIFT_FIELDS],
        dtype="<f8",
    )
    if not np.all(np.isfinite(pre)) or not np.all(np.isfinite(input_vector)):
        raise RuntimeError("intervention drift witness must be finite")
    delta_sha256, applied = _intervention_witness(delta)
    if str(row["intervention_delta_sha256"]) != delta_sha256:
        raise RuntimeError("intervention delta digest mismatch")
    if _canonical_csv_bool(row["intervention_applied"]) is not applied:
        raise RuntimeError("intervention applied predicate mismatch")
    expected_input = pre + delta if applied else pre
    if (
        input_vector.tobytes(order="C")
        != np.asarray(expected_input, dtype="<f8").tobytes(order="C")
    ):
        raise RuntimeError("intervention drift application mismatch")
    pre_non_drift = str(
        row["pre_intervention_non_drift_state_sha256"]
    )
    input_non_drift = str(row["input_non_drift_state_sha256"])
    if (
        len(pre_non_drift) != 64
        or any(value not in "0123456789abcdef" for value in pre_non_drift)
        or input_non_drift != pre_non_drift
    ):
        raise RuntimeError("intervention mutated non-drift state")
    expected_receipt = _canonical_application_receipt(row)
    if (
        str(row["intervention_application_receipt_sha256"])
        != expected_receipt
    ):
        raise RuntimeError("intervention application receipt mismatch")


def _intervention_application_evidence(
    *,
    identity: Mapping[str, object],
    pre_state: object,
    input_state: object,
    delta: object,
) -> dict[str, object]:
    pre = np.asarray(pre_state.drift.vector(), dtype="<f8")
    input_vector = np.asarray(input_state.drift.vector(), dtype="<f8")
    if (
        pre.shape != (5,)
        or input_vector.shape != (5,)
        or not np.all(np.isfinite(pre))
        or not np.all(np.isfinite(input_vector))
    ):
        raise RuntimeError("state drift vectors must be finite shape (5,)")
    delta_sha256, applied = _intervention_witness(delta)
    evidence: dict[str, object] = {
        "intervention_delta_sha256": delta_sha256,
        "intervention_applied": applied,
        **{
            name: float(pre[index])
            for index, name in enumerate(_PRE_DRIFT_FIELDS)
        },
        **{
            name: float(input_vector[index])
            for index, name in enumerate(_INPUT_DRIFT_FIELDS)
        },
        "pre_intervention_non_drift_state_sha256": (
            _state_non_drift_digest(pre_state)
        ),
        "input_non_drift_state_sha256": (
            _state_non_drift_digest(input_state)
        ),
    }
    receipt_row = {**identity, **evidence}
    evidence["intervention_application_receipt_sha256"] = (
        _canonical_application_receipt(receipt_row)
    )
    _validate_intervention_application_fields(
        {**identity, **evidence},
        delta,
    )
    return evidence


def _unavailable_intervention_evidence(
    delta: object,
) -> dict[str, object]:
    delta_sha256, applied = _intervention_witness(delta)
    return {
        "intervention_delta_sha256": delta_sha256,
        "intervention_applied": applied,
        **{name: "" for name in _PRE_DRIFT_FIELDS},
        **{name: "" for name in _INPUT_DRIFT_FIELDS},
        "pre_intervention_non_drift_state_sha256": "",
        "input_non_drift_state_sha256": "",
        "intervention_application_receipt_sha256": "",
    }


def _rb_arrays(
    store: ImmutableObjectStore,
    *,
    count: int,
    dimension: int,
    expected_alias: np.memmap | None,
    registry: list[np.memmap] | None = None,
) -> tuple[dict[str, Path], dict[str, np.memmap]]:
    if count == 0:
        return {}, {}
    specifications: dict[str, tuple[tuple[int, ...], str]] = {
        "valid": ((count,), "?"),
        "row_index": ((count,), "<i8"),
        "success_probability": ((count,), "<f8"),
        "success_present": ((count,), "?"),
        "failure_present": ((count,), "?"),
        "conditional_success_density": ((count, dimension, dimension), "<c8"),
        "conditional_failure_density": ((count, dimension, dimension), "<c8"),
        "sampled_stress_density": ((count, dimension, dimension), "<c8"),
        "sampled_hidden_outcome": ((count,), "u1"),
        "sampled_reset_ack": ((count,), "S16"),
        "branch_trace_distance": ((count,), "<f8"),
        "sampled_match_trace_distance": ((count,), "<f8"),
        "pre_reset_receipt": ((count,), "S64"),
    }
    paths: dict[str, Path] = {}
    arrays: dict[str, np.memmap] = {}
    for name, (shape, dtype) in specifications.items():
        paths[name], arrays[name] = _new_memmap(
            store,
            shape,
            dtype,
            registry=registry,
        )
    if expected_alias is not None:
        arrays["expected_density"] = expected_alias
    else:
        paths["expected_density"], arrays["expected_density"] = _new_memmap(
            store,
            (count, dimension, dimension),
            "<c8",
            registry=registry,
        )
    return paths, arrays


def _record_rb(
    arrays: Mapping[str, np.memmap],
    index: int,
    *,
    row_index: int,
    evidence: object,
) -> None:
    arrays["valid"][index] = True
    arrays["row_index"][index] = row_index
    arrays["success_probability"][index] = evidence.success_probability
    arrays["success_present"][index] = evidence.success_density is not None
    arrays["failure_present"][index] = evidence.failure_density is not None
    arrays["expected_density"][index] = np.asarray(
        evidence.expected_density, dtype=np.complex64
    )
    arrays["conditional_success_density"][index] = (
        np.asarray(evidence.success_density, dtype=np.complex64)
        if evidence.success_density is not None
        else np.zeros_like(evidence.expected_density, dtype=np.complex64)
    )
    arrays["conditional_failure_density"][index] = (
        np.asarray(evidence.failure_density, dtype=np.complex64)
        if evidence.failure_density is not None
        else np.zeros_like(evidence.expected_density, dtype=np.complex64)
    )
    arrays["sampled_stress_density"][index] = np.asarray(
        evidence.sampled_density, dtype=np.complex64
    )
    arrays["sampled_hidden_outcome"][index] = (
        1 if evidence.sampled_hidden_outcome == "success" else 0
    )
    arrays["sampled_reset_ack"][index] = (
        str(evidence.sampled_result.observation.reset_ack).encode("ascii")
    )
    arrays["branch_trace_distance"][index] = evidence.branch_trace_distance
    arrays["sampled_match_trace_distance"][
        index
    ] = evidence.sampled_matches_forced_branch_trace_distance
    arrays["pre_reset_receipt"][index] = (
        evidence.pre_reset_causal_receipt_sha256.encode("ascii")
    )


def _record_rb_failure(
    arrays: Mapping[str, np.memmap],
    index: int,
    *,
    row_index: int,
) -> None:
    """Preserve an expected RESET sidecar position after a physics exception."""

    arrays["valid"][index] = False
    arrays["row_index"][index] = row_index
    arrays["success_probability"][index] = np.nan
    arrays["success_present"][index] = False
    arrays["failure_present"][index] = False
    for name in (
        "expected_density",
        "conditional_success_density",
        "conditional_failure_density",
        "sampled_stress_density",
    ):
        arrays[name][index] = np.nan + 1j * np.nan
    arrays["sampled_hidden_outcome"][index] = 255
    arrays["sampled_reset_ack"][index] = b""
    arrays["branch_trace_distance"][index] = np.nan
    arrays["sampled_match_trace_distance"][index] = np.nan
    arrays["pre_reset_receipt"][index] = b""


def _validate_npy(path: Path, *, shape: tuple[int, ...], dtype: str) -> None:
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    try:
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise RuntimeError(f"staged NPY shape/dtype drift: {path}")
    finally:
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _validate_rb_mixture(
    arrays: Mapping[str, np.memmap],
    count: int,
) -> None:
    for start in range(0, count, 16):
        stop = min(count, start + 16)
        valid = np.asarray(arrays["valid"][start:stop], dtype=bool)
        if not np.any(valid):
            continue
        probability = np.asarray(
            arrays["success_probability"][start:stop], dtype=np.float64
        )[valid]
        success_present = np.asarray(
            arrays["success_present"][start:stop]
        )[valid]
        failure_present = np.asarray(
            arrays["failure_present"][start:stop]
        )[valid]
        success = np.asarray(
            arrays["conditional_success_density"][start:stop],
            dtype=np.complex128,
        )[valid]
        failure = np.asarray(
            arrays["conditional_failure_density"][start:stop],
            dtype=np.complex128,
        )[valid]
        expected = np.asarray(
            arrays["expected_density"][start:stop],
            dtype=np.complex128,
        )[valid]
        mixture = (
            probability[:, None, None] * success
            + (1.0 - probability[:, None, None]) * failure
        )
        # A degenerate absent branch has zero weight and a zero placeholder.
        if np.any((~success_present) & (probability > 2.0e-12)) or np.any(
            (~failure_present) & (probability < 1.0 - 2.0e-12)
        ):
            raise RuntimeError("RB conditional branch presence drift")
        errors = np.linalg.norm(expected - mixture, axis=(1, 2))
        if float(np.max(errors, initial=0.0)) > 3.0e-6:
            raise RuntimeError("RB expected density/branch mixture drift")
        if np.any(
            np.asarray(
                arrays["sampled_match_trace_distance"][start:stop]
            )[valid]
            > 2.0e-10
        ):
            raise RuntimeError("RB sampled stress does not match forced branch")


def _adopt_array(
    store: ImmutableObjectStore,
    path: Path,
    *,
    role: str,
) -> ObjectBinding:
    return store.adopt_staged_file(
        path,
        role=role,
        media_type="application/x-npy",
    )


def _stage_mapping_arrays(
    store: ImmutableObjectStore,
    *,
    runner: Any,
    simulators: Mapping[str, object],
    cutoff: int,
) -> dict[str, Path]:
    arrays = runner._mapping_arrays({cutoff: simulators})
    selected = {
        "mapping_isometry_a_npy": arrays[
            f"mapping_isometry_a_cutoff_{cutoff}"
        ],
        "mapping_isometry_b_npy": arrays[
            f"mapping_isometry_b_cutoff_{cutoff}"
        ],
        "mapping_projector_a_npy": arrays[
            f"mapping_projector_a_cutoff_{cutoff}"
        ],
        "mapping_projector_b_npy": arrays[
            f"mapping_projector_b_cutoff_{cutoff}"
        ],
    }
    paths: dict[str, Path] = {}
    for role, value in selected.items():
        path = store.new_staging_path(suffix=".npy")
        with path.open("wb") as handle:
            np.save(
                handle,
                np.asarray(value, dtype=np.complex128),
                allow_pickle=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        paths[role] = path
    return paths


def _validate_mapping_anchor(
    paths: Mapping[str, Path],
    *,
    cutoff: int,
) -> None:
    expected_roles = {
        "mapping_isometry_a_npy",
        "mapping_isometry_b_npy",
        "mapping_projector_a_npy",
        "mapping_projector_b_npy",
    }
    if set(paths) != expected_roles:
        raise RuntimeError("mapping anchor role family drift")
    arrays: dict[str, np.ndarray] = {}
    for role, path in paths.items():
        value = np.load(path, allow_pickle=False, mmap_mode="r")
        try:
            arrays[role] = np.array(value, dtype=np.complex128, copy=True)
        finally:
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()
    identity = np.eye(2, dtype=np.complex128)
    for backend in ("a", "b"):
        isometry = arrays[f"mapping_isometry_{backend}_npy"]
        projector = arrays[f"mapping_projector_{backend}_npy"]
        if (
            isometry.shape != (cutoff, 2)
            or projector.shape != (cutoff, cutoff)
            or not np.all(np.isfinite(isometry))
            or not np.all(np.isfinite(projector))
            or not np.allclose(
                isometry.conj().T @ isometry,
                identity,
                rtol=0.0,
                atol=2.0e-10,
            )
            or not np.allclose(
                projector,
                isometry @ isometry.conj().T,
                rtol=0.0,
                atol=2.0e-13,
            )
            or not np.allclose(
                projector,
                projector.conj().T,
                rtol=0.0,
                atol=2.0e-13,
            )
        ):
            raise RuntimeError(
                f"mapping anchor semantic drift for backend {backend.upper()}"
            )


def _sanitize_expected_reset_primary(
    row: dict[str, object],
    evidence: object,
) -> None:
    """Remove sampled branch/ack fields from the unconditional primary row."""

    row["reset_hidden_success"] = ""
    row["reset_ack"] = "marginalized"
    row["rao_blackwell_reset_success"] = float(
        evidence.success_probability
    )


def _live_binding(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    payload = resolved.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _validate_preformal_seal(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    plan_sha256: str,
    source_snapshot_sha256: str,
    expected_file_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    """Revalidate the complete immutable release lineage in each worker."""

    seal_path = root / str(config["artifact_paths"]["preformal_seal"])
    if not seal_path.exists():
        raise RuntimeError("formal cell execution requires preformal seal")
    seal_bytes = seal_path.read_bytes()
    seal_file_sha256 = _sha_bytes(seal_bytes)
    if (
        expected_file_sha256 is None
        or expected_file_sha256 != seal_file_sha256
    ):
        raise RuntimeError("formal preformal-seal file binding mismatch")
    seal = _strict_json(seal_path)
    _canonical_json_sha(seal, "analysis_sha256")
    live_snapshot = runtime_source_snapshot(root, config)
    if (
        seal.get("schema_version")
        != "PHASE9-POWERED-TWIN-PREFORMAL-SEAL-V1"
        or seal.get("task_id") != config.get("task_id")
        or seal.get("verdict") != "PASS_PREFORMAL_RELEASE"
        or seal.get("raw_execution_released") is not True
        or seal.get("scientific_verdict_released") is not False
        or seal.get("formal_outcomes_accessed") is not False
        or seal.get("config_sha256") != config_sha256
        or seal.get("plan_sha256") != plan_sha256
        or seal.get("source_snapshot_sha256")
        != source_snapshot_sha256
        or seal.get("source_snapshot") != live_snapshot
        or live_snapshot.get("source_snapshot_sha256")
        != source_snapshot_sha256
    ):
        raise RuntimeError("formal source/config/plan seal mismatch")
    bindings = seal.get("bindings")
    expected_bindings = {
        "contract_preflight": config["artifact_paths"]["contract_preflight"],
        "resource_preflight": config["artifact_paths"]["resource_preflight"],
        "preformal_validation": config["artifact_paths"]["preformal_validation"],
    }
    if not isinstance(bindings, Mapping) or set(bindings) != set(
        expected_bindings
    ):
        raise RuntimeError("preformal seal binding family mismatch")
    for name, relative in expected_bindings.items():
        registered = bindings[name]
        if (
            not isinstance(registered, Mapping)
            or dict(registered)
            != _live_binding(root / str(relative), root)
        ):
            raise RuntimeError(f"preformal seal live binding drift: {name}")
    claims = seal.get("claim_boundary")
    if (
        not isinstance(claims, Mapping)
        or set(claims) != set(EXPECTED_CLAIM_FIELDS)
        or any(claims.get(name) is not None for name in EXPECTED_CLAIM_FIELDS)
        or seal.get("scientific_verdict") is not None
        or seal.get("qualified_claim") is not None
        or seal.get("official_puviani_surpass") is not None
    ):
        raise RuntimeError("preformal seal claim boundary drift")
    return seal, seal_file_sha256


def _execute_cell_to_store_impl(
    *,
    root: Path,
    t04: Mapping[str, Any],
    config_sha256: str,
    plan_sha256: str,
    run_id: str,
    cell: T04CellSpec,
    source_snapshot_sha256: str,
    sample_count_override: int | None = None,
    seed_namespace: str = "formal",
    artifact_paths_override: Mapping[str, str] | None = None,
    preformal_seal_file_sha256: str | None = None,
    _memmap_registry: list[np.memmap],
) -> dict[str, Any]:
    """Execute and atomically commit one complete cell denominator."""

    base, runner, dual, powered_reset = _activate_execution_modules(root)
    execution = _execution_config(base, t04)
    simulators = runner.build_simulators(execution, cell.cutoff)
    simulator = simulators[cell.backend]
    actions = dual._action_words()
    if (
        not isinstance(source_snapshot_sha256, str)
        or len(source_snapshot_sha256) != 64
        or source_snapshot_sha256 == "0" * 64
        or any(
            character not in "0123456789abcdef"
            for character in source_snapshot_sha256
        )
    ):
        raise ValueError("a nonzero lowercase source snapshot SHA-256 is required")
    if seed_namespace == "formal":
        _validate_preformal_seal(
            root=root,
            config=t04,
            config_sha256=config_sha256,
            plan_sha256=plan_sha256,
            source_snapshot_sha256=source_snapshot_sha256,
            expected_file_sha256=preformal_seal_file_sha256,
        )
    paths = (
        dict(artifact_paths_override)
        if artifact_paths_override is not None
        else t04["artifact_paths"]
    )
    if artifact_paths_override is not None:
        formal_roots = [
            (root / str(t04["artifact_paths"][name])).resolve()
            for name in (
                "object_store",
                "staging_directory",
                "receipt_directory",
            )
        ]
        override_roots = [
            (root / str(paths[name])).resolve()
            for name in (
                "object_store",
                "staging_directory",
                "receipt_directory",
            )
        ]
        if len(set(override_roots)) != 3:
            raise ValueError("isolated artifact roots must be distinct")
        for override in override_roots:
            for formal in formal_roots:
                if (
                    override == formal
                    or override in formal.parents
                    or formal in override.parents
                ):
                    raise ValueError(
                        "preflight artifact namespace overlaps formal evidence"
                    )
    store = ImmutableObjectStore(
        repository_root=root,
        object_root=root / str(paths["object_store"]),
        staging_root=root / str(paths["staging_directory"]),
        receipt_root=root / str(paths["receipt_directory"]),
        task_id=str(t04["task_id"]),
        run_id=run_id,
        config_sha256=config_sha256,
        plan_sha256=plan_sha256,
    )
    if sample_count_override is not None:
        if seed_namespace == "formal":
            raise ValueError("formal cells forbid sample_count_override")
        if artifact_paths_override is None:
            raise ValueError(
                "preflight sample override requires an isolated artifact namespace"
            )
        if sample_count_override <= 0 or sample_count_override > cell.sample_count:
            raise ValueError("sample_count_override outside frozen upper bound")
        cell = replace(
            cell,
            sample_count=int(sample_count_override),
            expected_rows=int(sample_count_override) * cell.horizon,
        )
    if seed_namespace not in {"formal", "resource_preflight", "capability_preflight"}:
        raise ValueError("unknown seed namespace")
    if seed_namespace == "formal":
        def seed_function(position: int) -> int:
            return physical_seed(t04, cell, position)
    else:
        namespace = t04["seed_registry"][seed_namespace]
        def seed_function(position: int) -> int:
            value = (
                int(namespace["start"])
                + int(namespace["physical_offset"])
                + (0 if cell.backend == "A" else 1)
                * 97
                * int(t04["seed_registry"]["maximum_cluster_positions"])
                + cell.pair_group_index
                * int(t04["seed_registry"]["maximum_cluster_positions"])
                + position
            )
            if value >= int(namespace["start"]) + int(namespace["count"]):
                raise RuntimeError("preflight physical seed escaped its namespace")
            return value

    dimension = 3 * cell.cutoff
    primary_count = (
        cell.sample_count if cell.density_retention != "none" else 0
    )
    rb_count = _rb_event_count(t04, cell)
    row_path = store.new_staging_path(suffix=".csv")
    raw_path, raw_iq = _new_memmap(
        store,
        (cell.expected_rows, int(execution["common_physics"]["iq_samples"]), 2),
        "<f8",
        registry=_memmap_registry,
    )
    heldout_path, heldout_iq = _new_memmap(
        store,
        (cell.expected_rows, int(execution["common_physics"]["iq_samples"]), 2),
        "<f8",
        registry=_memmap_registry,
    )
    density_path: Path | None = None
    primary_density: np.memmap | None = None
    if primary_count:
        density_path, primary_density = _new_memmap(
            store,
            (primary_count, dimension, dimension),
            "<c8",
            registry=_memmap_registry,
        )
    expected_alias = (
        primary_density
        if rb_count
        and primary_density is not None
        and cell.layer in {"shared", "probe"}
        and rb_count == primary_count
        else None
    )
    rb_paths, rb = _rb_arrays(
        store,
        count=rb_count,
        dimension=dimension,
        expected_alias=expected_alias,
        registry=_memmap_registry,
    )
    mapping_paths: dict[str, Path] = {}
    if t04["formal_matrix"]["mapping_anchor_plan_indices"].get(
        str(cell.cutoff)
    ) == cell.plan_index:
        mapping_paths = _stage_mapping_arrays(
            store,
            runner=runner,
            simulators=simulators,
            cutoff=cell.cutoff,
        )
    fields = tuple(runner.LEDGER_FIELDS) + EXTRA_FIELDS
    exception_rows = 0
    conservation_failures = 0
    row_index = 0
    density_index = 0
    rb_index = 0
    with row_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for position in range(cell.sample_count):
            physical = seed_function(position)
            if cell.layer != "fault":
                heldout_address, heldout = _heldout_window(
                    execution,
                    t04,
                    cell,
                    position,
                    0,
                    seed_namespace=seed_namespace,
                )
                identity = _identity(
                    runner,
                    cell,
                    seed=physical,
                    position=position,
                    row_index=row_index,
                    action=cell.action,
                )
                reset_evidence = None
                result = None
                success = False
                pre_intervention_digest = ""
                input_state_digest = ""
                input_evaluator_digest = ""
                output_state_digest = ""
                output_evaluator_digest = ""
                zero_delta = np.zeros(5, dtype="<f8")
                intervention_evidence = _unavailable_intervention_evidence(
                    zero_delta
                )
                try:
                    state, evaluator = runner._initial_state(cell, simulator)
                    pre_intervention_digest = _state_digest(state)
                    input_state_digest = pre_intervention_digest
                    intervention_evidence = (
                        _intervention_application_evidence(
                            identity=identity,
                            pre_state=state,
                            input_state=state,
                            delta=zero_delta,
                        )
                    )
                    input_evaluator_digest = (
                        _semantic_object_digest(evaluator)
                        if evaluator is not None
                        else ""
                    )
                    action_word = actions[cell.action]
                    if cell.action == "RESET":
                        reset_evidence = (
                            powered_reset.evaluate_expected_reset_powered(
                                backend=cell.backend,
                                simulator=simulator,
                                state=state,
                                evaluator=evaluator,
                                action=action_word,
                                seed=physical,
                            )
                        )
                        result = powered_reset.expected_primary_result(
                            reset_evidence,
                            simulator=simulator,
                        )
                    else:
                        result = dual._one_step(
                            backend=cell.backend,
                            simulator=simulator,
                            state=state,
                            evaluator=evaluator,
                            action=action_word,
                            seed=physical,
                        )
                    retain = primary_density is not None
                    row, raw = runner._success_row(
                        config=execution,
                        identity=identity,
                        simulator=simulator,
                        result=result,
                        action=action_word,
                        heldout=heldout,
                        density_index=(density_index if retain else -1),
                    )
                    if reset_evidence is not None:
                        _sanitize_expected_reset_primary(row, reset_evidence)
                    output_state_digest = _state_digest(result.state)
                    next_evaluator = _next_evaluator(result, cell.backend)
                    output_evaluator_digest = (
                        _semantic_object_digest(next_evaluator)
                        if next_evaluator is not None
                        else ""
                    )
                    success = True
                except Exception as exc:
                    row = runner._exception_row(identity, exc, heldout)
                    raw = np.full_like(heldout, np.nan)
                    exception_rows += 1
                if primary_density is not None:
                    if success:
                        primary_density[density_index] = np.asarray(
                            result.state.joint_density, dtype=np.complex64
                        )
                    else:
                        primary_density[density_index] = np.nan + 1j * np.nan
                    density_index += 1
                if cell.action == "RESET":
                    if success and reset_evidence is not None:
                        _record_rb(
                            rb,
                            rb_index,
                            row_index=row_index,
                            evidence=reset_evidence,
                        )
                    else:
                        _record_rb_failure(
                            rb,
                            rb_index,
                            row_index=row_index,
                        )
                    rb_index += 1
                row = _augment_row(
                    row,
                    t04=t04,
                    cell=cell,
                    position=position,
                    physical=physical,
                    heldout=heldout_address,
                    reset_evidence=(reset_evidence if success else None),
                    pre_intervention_state_sha256=pre_intervention_digest,
                    intervention_evidence=intervention_evidence,
                    input_state_sha256=input_state_digest,
                    input_evaluator_sha256=input_evaluator_digest,
                    output_state_sha256=output_state_digest,
                    output_evaluator_sha256=output_evaluator_digest,
                    expected_reset_ancestor_receipt_sha256=(
                        reset_evidence.pre_reset_causal_receipt_sha256
                        if success and reset_evidence is not None
                        else ""
                    ),
                )
                conservation_failures += int(
                    not bool(row.get("conservation_pass", False))
                )
                raw_iq[row_index] = raw
                heldout_iq[row_index] = heldout
                writer.writerow(row)
                row_index += 1
                continue

            label, state_index, within_state = _fault_label(t04, position)
            try:
                state, evaluator = simulator.initialize_logical(label)
                failed: BaseException | None = None
            except Exception as exc:
                state, evaluator, failed = None, None, exc
            last_expected_reset_ancestor = ""
            specification = execution["formal_matrix"]["fault_scenarios"][
                cell.scenario
            ]
            sequence = t04["formal_matrix"]["fault_action_sequences"][
                cell.scenario
            ]
            for round_number in range(cell.horizon):
                row: dict[str, object] | None = None
                raw: np.ndarray | None = None
                action_name = sequence[round_number % len(sequence)]
                heldout_address, heldout = _heldout_window(
                    execution,
                    t04,
                    cell,
                    position,
                    round_number,
                    seed_namespace=seed_namespace,
                )
                terminal = round_number == cell.horizon - 1
                identity = _identity(
                    runner,
                    cell,
                    seed=physical,
                    position=position,
                    row_index=row_index,
                    action=action_name,
                    round_index=round_number,
                    terminal_round=terminal,
                    logical_label=label,
                )
                reset_evidence = None
                result = None
                success = False
                pre_intervention_digest = _state_digest(state)
                input_state_digest = pre_intervention_digest
                input_evaluator_digest = (
                    _semantic_object_digest(evaluator)
                    if evaluator is not None
                    else ""
                )
                output_state_digest = ""
                output_evaluator_digest = ""
                row_reset_ancestor = last_expected_reset_ancestor
                candidate_reset_ancestor = last_expected_reset_ancestor
                pre_intervention_state = state
                delta = dual._fault_delta_for_round(
                    cell.scenario, specification, round_number
                )
                _, intervention_applied = _intervention_witness(delta)
                intervention_evidence = _unavailable_intervention_evidence(
                    delta
                )
                if failed is None:
                    try:
                        if intervention_applied:
                            state = dual._apply_intervention(
                                state, cell.backend, delta
                            )
                        input_state_digest = _state_digest(state)
                        intervention_evidence = (
                            _intervention_application_evidence(
                                identity=identity,
                                pre_state=pre_intervention_state,
                                input_state=state,
                                delta=delta,
                            )
                        )
                        action_word = actions[action_name]
                        if action_name == "RESET":
                            reset_evidence = (
                                powered_reset.evaluate_expected_reset_powered(
                                    backend=cell.backend,
                                    simulator=simulator,
                                    state=state,
                                    evaluator=evaluator,
                                    action=action_word,
                                    seed=physical,
                                )
                            )
                            result = powered_reset.expected_primary_result(
                                reset_evidence,
                                simulator=simulator,
                            )
                        else:
                            result = dual._one_step(
                                backend=cell.backend,
                                simulator=simulator,
                                state=state,
                                evaluator=evaluator,
                                action=action_word,
                                seed=physical,
                            )
                        row, raw = runner._success_row(
                            config=execution,
                            identity=identity,
                            simulator=simulator,
                            result=result,
                            action=action_word,
                            heldout=heldout,
                            density_index=(
                                density_index if terminal else -1
                            ),
                        )
                        if reset_evidence is not None:
                            _sanitize_expected_reset_primary(
                                row,
                                reset_evidence,
                            )
                            candidate_reset_ancestor = (
                                reset_evidence.pre_reset_causal_receipt_sha256
                            )
                        next_state = result.state
                        next_evaluator = _next_evaluator(result, cell.backend)
                        if result.logical is None or next_evaluator is None:
                            raise RuntimeError(
                                "fault logical evaluator carry became undefined"
                            )
                        output_state_digest = _state_digest(next_state)
                        output_evaluator_digest = _semantic_object_digest(
                            next_evaluator
                        )
                        success = True
                        state = next_state
                        evaluator = next_evaluator
                        if reset_evidence is not None:
                            row_reset_ancestor = candidate_reset_ancestor
                            last_expected_reset_ancestor = (
                                candidate_reset_ancestor
                            )
                    except Exception as exc:
                        failed = exc
                if failed is not None and not success:
                    row = runner._exception_row(identity, failed, heldout)
                    raw = np.full_like(heldout, np.nan)
                    exception_rows += 1
                if terminal and primary_density is not None:
                    if success:
                        primary_density[density_index] = np.asarray(
                            result.state.joint_density,
                            dtype=np.complex64,
                        )
                    else:
                        primary_density[density_index] = np.nan + 1j * np.nan
                    density_index += 1
                if action_name == "RESET":
                    if success and reset_evidence is not None:
                        _record_rb(
                            rb,
                            rb_index,
                            row_index=row_index,
                            evidence=reset_evidence,
                        )
                    else:
                        _record_rb_failure(
                            rb,
                            rb_index,
                            row_index=row_index,
                        )
                    rb_index += 1
                if row is None or raw is None:
                    raise RuntimeError("fault round produced neither success nor exception")
                row = _augment_row(
                    row,
                    t04=t04,
                    cell=cell,
                    position=position,
                    physical=physical,
                    heldout=heldout_address,
                    reset_evidence=(reset_evidence if success else None),
                    fault_state_index=state_index,
                    fault_within_state_index=within_state,
                    pre_intervention_state_sha256=pre_intervention_digest,
                    intervention_evidence=intervention_evidence,
                    input_state_sha256=input_state_digest,
                    input_evaluator_sha256=input_evaluator_digest,
                    output_state_sha256=output_state_digest,
                    output_evaluator_sha256=output_evaluator_digest,
                    expected_reset_ancestor_receipt_sha256=(
                        row_reset_ancestor
                    ),
                )
                conservation_failures += int(
                    not bool(row.get("conservation_pass", False))
                )
                raw_iq[row_index] = raw
                heldout_iq[row_index] = heldout
                writer.writerow(row)
                row_index += 1
        handle.flush()
        os.fsync(handle.fileno())

    if (
        row_index != cell.expected_rows
        or density_index != primary_count
        or rb_index != rb_count
    ):
        raise RuntimeError(
            "cell denominator drift "
            f"rows={row_index}/{cell.expected_rows} "
            f"density={density_index}/{primary_count} rb={rb_index}/{rb_count}"
        )
    if rb_count:
        _validate_rb_mixture(rb, rb_count)
    rb_metadata = {
        name: (array.shape, array.dtype.str) for name, array in rb.items()
    }
    arrays_to_close: list[np.memmap] = [raw_iq, heldout_iq]
    arrays_to_close.extend(rb.values())
    if primary_density is not None and all(
        primary_density is not array for array in rb.values()
    ):
        arrays_to_close.append(primary_density)
    seen_arrays: set[int] = set()
    for array in arrays_to_close:
        if id(array) not in seen_arrays:
            _close_memmap(array)
            seen_arrays.add(id(array))
    _validate_npy(
        raw_path,
        shape=(cell.expected_rows, int(execution["common_physics"]["iq_samples"]), 2),
        dtype="<f8",
    )
    _validate_npy(
        heldout_path,
        shape=(cell.expected_rows, int(execution["common_physics"]["iq_samples"]), 2),
        dtype="<f8",
    )
    objects: list[ObjectBinding] = [
        store.adopt_staged_file(
            row_path, role="round_ledger_csv", media_type="text/csv"
        ),
        _adopt_array(store, raw_path, role="raw_iq_npy"),
        _adopt_array(store, heldout_path, role="heldout_iq_npy"),
    ]
    expected_binding: ObjectBinding | None = None
    if density_path is not None:
        _validate_npy(
            density_path,
            shape=(primary_count, dimension, dimension),
            dtype="<c8",
        )
        primary_binding = _adopt_array(
            store, density_path, role="primary_density_npy"
        )
        objects.append(primary_binding)
        if expected_alias is not None:
            expected_binding = replace(
                primary_binding, role="rb_expected_density_npy"
            )
            objects.append(expected_binding)
    if rb_count:
        for name, path in rb_paths.items():
            shape, dtype = rb_metadata[name]
            _validate_npy(path, shape=shape, dtype=dtype)
            objects.append(
                _adopt_array(store, path, role=f"rb_{name}_npy")
            )
    if mapping_paths:
        _validate_mapping_anchor(mapping_paths, cutoff=cell.cutoff)
    for role, path in sorted(mapping_paths.items()):
        objects.append(_adopt_array(store, path, role=role))
    missing_rows = max(0, cell.expected_rows - row_index)
    receipt = store.commit_receipt(
        cell=asdict(cell),
        objects=objects,
        expected_rows=cell.expected_rows,
        observed_rows=row_index,
        exception_rows=exception_rows,
        missing_rows=missing_rows,
        conservation_failures=conservation_failures,
        source_snapshot_sha256=source_snapshot_sha256,
        runtime_fingerprint={
            "runner_id": RUNNER_ID,
            "python": list(sys.version_info[:3]),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "seed_namespace": seed_namespace,
        },
        reset_rows=rb_count,
        reset_sidecar_rows=rb_count,
    )
    return receipt


def execute_cell_to_store(
    *,
    root: Path,
    t04: Mapping[str, Any],
    config_sha256: str,
    plan_sha256: str,
    run_id: str,
    cell: T04CellSpec,
    source_snapshot_sha256: str,
    sample_count_override: int | None = None,
    seed_namespace: str = "formal",
    artifact_paths_override: Mapping[str, str] | None = None,
    preformal_seal_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute one cell and deterministically release every mapped file.

    The inner transaction intentionally leaves content-addressed orphan files
    after a publication failure, but Windows file handles must never survive
    the call: otherwise a fail-closed fresh rerun cannot archive the staging
    namespace.
    """

    memmaps: list[np.memmap] = []
    try:
        return _execute_cell_to_store_impl(
            root=root,
            t04=t04,
            config_sha256=config_sha256,
            plan_sha256=plan_sha256,
            run_id=run_id,
            cell=cell,
            source_snapshot_sha256=source_snapshot_sha256,
            sample_count_override=sample_count_override,
            seed_namespace=seed_namespace,
            artifact_paths_override=artifact_paths_override,
            preformal_seal_file_sha256=preformal_seal_file_sha256,
            _memmap_registry=memmaps,
        )
    finally:
        cleanup_error: BaseException | None = None
        seen: set[int] = set()
        for array in reversed(memmaps):
            if id(array) in seen:
                continue
            seen.add(id(array))
            try:
                _close_memmap(array)
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if cleanup_error is not None and sys.exc_info()[0] is None:
            raise RuntimeError("cell memmap cleanup failed") from cleanup_error


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token {token} in {path}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _canonical_json_sha(value: Mapping[str, Any], field: str) -> None:
    claimed = value.get(field)
    unsigned = dict(value)
    unsigned.pop(field, None)
    observed = sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if claimed != observed:
        raise RuntimeError(f"{field} mismatch")


def _git_release_attestation(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, object]:
    protected = [
        "configs/phase9/t_risk_20260728_04_powered_twin_qualification.json",
        *config["runtime_sources"]["paths"],
        *config["runtime_sources"]["validation_paths"],
        str(config["artifact_paths"]["contract_preflight"]),
        str(config["artifact_paths"]["resource_preflight"]),
        str(config["artifact_paths"]["preformal_validation"]),
        str(config["artifact_paths"]["preformal_seal"]),
    ]

    def git_result(*arguments: str) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            text=True,
            encoding="utf-8",
            errors="strict",
            capture_output=True,
            check=False,
        )
        return completed

    def git(*arguments: str) -> str:
        completed = git_result(*arguments)
        if completed.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    local = git("rev-parse", "HEAD")
    upstream = git("rev-parse", "@{upstream}")
    if local != upstream:
        raise RuntimeError("production launch commit does not equal upstream")
    untracked: list[str] = []
    for value in protected:
        completed = git_result(
            "ls-files",
            "--error-unmatch",
            "--",
            value,
        )
        if completed.returncode == 1:
            untracked.append(value)
        elif completed.returncode != 0:
            raise RuntimeError(
                "git ls-files attestation failed for "
                f"{value}: {completed.stderr.strip()}"
            )
    if untracked:
        raise RuntimeError(f"production protected paths are untracked: {untracked}")
    status = git("status", "--porcelain", "--", *protected)
    if status:
        raise RuntimeError("production protected source/artifact paths are dirty")
    return {
        "branch": git("branch", "--show-current"),
        "local_commit": local,
        "upstream_commit": upstream,
        "protected_path_count": len(protected),
        "protected_paths_clean": True,
    }


def _production_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(payload["root"])).resolve()
    config, binding = load_config(root)
    plan = build_cell_plan(config)
    index = int(payload["plan_index"])
    started = time.time_ns()
    receipt = execute_cell_to_store(
        root=root,
        t04=config,
        config_sha256=str(binding["sha256"]),
        plan_sha256=str(payload["plan_sha256"]),
        run_id=str(payload["run_id"]),
        cell=plan[index],
        source_snapshot_sha256=str(payload["source_snapshot_sha256"]),
        seed_namespace="formal",
        preformal_seal_file_sha256=str(
            payload["preformal_seal_file_sha256"]
        ),
    )
    return {
        "plan_index": index,
        "chunk_id": plan[index].chunk_id,
        "pid": os.getpid(),
        "started_ns": started,
        "finished_ns": time.time_ns(),
        "receipt": receipt,
    }


def _formal_object_bytes(root: Path, object_root: Path) -> int:
    resolved = object_root.resolve()
    resolved.relative_to(root.resolve())
    if not resolved.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in resolved.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _process_tree_rss_snapshot() -> dict[str, object]:
    """Sample the supervisor and every currently live recursive child."""

    process = psutil.Process(os.getpid())
    processes = [process, *process.children(recursive=True)]
    aggregate = 0
    rows: list[dict[str, object]] = []
    for member in processes:
        try:
            rss = int(member.memory_info().rss)
            aggregate += rss
            rows.append(
                {
                    "pid": member.pid,
                    "create_time": float(member.create_time()),
                    "rss_bytes": rss,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows.sort(key=lambda row: (int(row["pid"]), float(row["create_time"])))
    return {
        "process_tree": rows,
        "aggregate_rss_bytes": aggregate,
    }


def run_production(root: Path) -> dict[str, Any]:
    """Run the one sealed 518-cell transaction under an exclusive supervisor."""

    config, binding = load_config(root)
    paths = config["artifact_paths"]
    seal_path = root / str(paths["preformal_seal"])
    seal = _strict_json(seal_path)
    _canonical_json_sha(seal, "analysis_sha256")
    plan = build_cell_plan(config)
    plan_sha = str(plan_payload(config)["canonical_plan_sha256"])
    snapshot = runtime_source_snapshot(root, config)
    seal, seal_file_sha256 = _validate_preformal_seal(
        root=root,
        config=config,
        config_sha256=str(binding["sha256"]),
        plan_sha256=plan_sha,
        source_snapshot_sha256=str(snapshot["source_snapshot_sha256"]),
        expected_file_sha256=_sha_bytes(seal_path.read_bytes()),
    )
    resource = _strict_json(root / str(paths["resource_preflight"]))
    _canonical_json_sha(resource, "analysis_sha256")
    projections = resource.get("cell_projections")
    if not isinstance(projections, list) or len(projections) != len(plan):
        raise RuntimeError("resource report lacks exact 518-cell projections")
    projection_by_index = {
        int(row["plan_index"]): row
        for row in projections
        if isinstance(row, Mapping)
    }
    if set(projection_by_index) != set(range(len(plan))):
        raise RuntimeError("resource projection plan-index coverage drift")
    git_attestation = _git_release_attestation(root, config)
    run_id = str(config["runtime_contract"]["production_run_id"])
    receipt_root = root / str(paths["receipt_directory"])
    object_root = root / str(paths["object_store"])
    staging_root = root / str(paths["staging_directory"])
    if any(receipt_root.glob("*.json")) if receipt_root.exists() else False:
        raise RuntimeError("partial receipt resume is forbidden")
    if object_root.exists() and any(
        path.is_file() for path in object_root.rglob("*")
    ):
        raise RuntimeError("preexisting formal object requires fresh amendment")
    if staging_root.exists() and any(
        path.is_file() for path in staging_root.rglob("*")
    ):
        raise RuntimeError("preexisting formal staging requires fresh amendment")
    owner = OwnerLease(
        root / str(paths["owner_lock"]),
        run_id=run_id,
        config_sha256=str(binding["sha256"]),
        plan_sha256=plan_sha,
    )
    attempt_path = root / str(paths["attempt_ledger"])
    state_lock = Lock()
    state: dict[str, Any] = {
        "phase": "production",
        "scheduled_chunks": 0,
        "committed_chunks": 0,
        "active_chunk_ids": [],
        "exception_rows": 0,
        "resource_sample_count": 0,
        "peak_process_tree_rss_bytes": 0,
    }

    def heartbeat_snapshot() -> Mapping[str, Any]:
        resource = _process_tree_rss_snapshot()
        rss = int(resource["aggregate_rss_bytes"])
        with state_lock:
            state["resource_sample_count"] += 1
            state["peak_process_tree_rss_bytes"] = max(
                int(state["peak_process_tree_rss_bytes"]),
                rss,
            )
            return {**state, **resource}

    heartbeat = HeartbeatService(
        path=root / str(paths["heartbeat"]),
        owner=owner,
        period_seconds=float(
            config["runtime_contract"]["heartbeat_period_seconds"]
        ),
        snapshot=heartbeat_snapshot,
    )
    watchdog = ResourceWatchdog(
        run_directory=root / str(paths["run_directory"]),
        maximum_wall_seconds=float(
            config["resource_contract"]["maximum_wall_seconds"]
        ),
        maximum_peak_rss_bytes=int(
            config["resource_contract"]["maximum_peak_rss_bytes"]
        ),
        minimum_post_projection_free_bytes=int(
            config["resource_contract"]["minimum_post_projection_free_bytes"]
        ),
        maximum_artifact_bytes=int(
            config["resource_contract"]["maximum_artifact_bytes"]
        ),
    )
    # The cost key is outcome-blind and frozen: projected resource cost, then
    # plan index for deterministic tie-breaking.
    queue = sorted(
        plan,
        key=lambda cell: (
            -float(
                projection_by_index[cell.plan_index][
                    "projected_wall_seconds"
                ]
            ),
            cell.plan_index,
        ),
    )
    committed_indices: set[int] = set()
    active: dict[object, T04CellSpec] = {}
    stop_reason: str | None = None
    maximum_workers = int(config["runtime_contract"]["max_workers"])
    identity = owner.acquire()
    heartbeat_started = True
    try:
        append_attempt_event(
            attempt_path,
            task_id=str(config["task_id"]),
            run_id=run_id,
            event="START",
            payload={
                "supervisor_id": PRODUCTION_SUPERVISOR_ID,
                "owner_token": identity.owner_token,
                "source_snapshot_sha256": snapshot["source_snapshot_sha256"],
                "preformal_seal_file_sha256": seal_file_sha256,
                "git_attestation": git_attestation,
                "formal_outcomes_accessed": False,
            },
        )
        heartbeat.start()
        with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
            while queue or active:
                while queue and len(active) < maximum_workers and stop_reason is None:
                    if heartbeat.error is not None:
                        stop_reason = (
                            "INCOMPLETE_FAIL_CLOSED_HEARTBEAT:"
                            f"{type(heartbeat.error).__name__}:"
                            f"{heartbeat.error}"
                        )
                        break
                    live_resource = heartbeat_snapshot()
                    if int(live_resource["aggregate_rss_bytes"]) > int(
                        config["resource_contract"][
                            "maximum_peak_rss_bytes"
                        ]
                    ):
                        stop_reason = (
                            "INCOMPLETE_RESOURCE_FAIL_CLOSED_PROCESS_TREE_RSS"
                        )
                        break
                    remaining = [*queue, *active.values()]
                    projected_bytes = sum(
                        int(
                            projection_by_index[cell.plan_index][
                                "projected_object_bytes"
                            ]
                        )
                        for cell in remaining
                    )
                    wall_values = [
                        float(
                            projection_by_index[cell.plan_index][
                                "projected_wall_seconds"
                            ]
                        )
                        for cell in remaining
                    ]
                    projected_wall = max(
                        max(wall_values, default=0.0),
                        sum(wall_values) / maximum_workers,
                    )
                    watchdog.check(
                        committed_bytes=_formal_object_bytes(
                            root, object_root
                        ),
                        projected_remaining_bytes=projected_bytes,
                        maximum_inflight_temp_bytes=int(
                            resource["maximum_inflight_temp_bytes"]
                        ),
                        analysis_scratch_bytes=int(
                            resource["analysis_scratch_bytes"]
                        ),
                        projected_remaining_wall_seconds=projected_wall,
                    )
                    cell = queue.pop(0)
                    future = executor.submit(
                        _production_worker,
                        {
                            "root": str(root),
                            "plan_index": cell.plan_index,
                            "plan_sha256": plan_sha,
                            "run_id": run_id,
                            "source_snapshot_sha256": snapshot[
                                "source_snapshot_sha256"
                            ],
                            "preformal_seal_file_sha256": (
                                seal_file_sha256
                            ),
                        },
                    )
                    active[future] = cell
                    with state_lock:
                        state["scheduled_chunks"] += 1
                        state["active_chunk_ids"] = [
                            value.chunk_id for value in active.values()
                        ]
                if not active:
                    break
                completed, _ = wait(
                    tuple(active),
                    return_when=FIRST_COMPLETED,
                )
                for future in completed:
                    cell = active.pop(future)
                    try:
                        result = future.result()
                        receipt = result["receipt"]
                        diagnostics = receipt["diagnostics"]
                        committed_indices.add(cell.plan_index)
                        append_attempt_event(
                            attempt_path,
                            task_id=str(config["task_id"]),
                            run_id=run_id,
                            event="CELL_COMMITTED",
                            payload={
                                "plan_index": cell.plan_index,
                                "chunk_id": cell.chunk_id,
                                "receipt_sha256": receipt["receipt_sha256"],
                                "worker_pid": result["pid"],
                                "started_ns": result["started_ns"],
                                "finished_ns": result["finished_ns"],
                                "diagnostics": diagnostics,
                            },
                        )
                        if (
                            int(diagnostics["exception_rows"]) != 0
                            or int(diagnostics["missing_rows"]) != 0
                            or int(diagnostics["conservation_failures"]) != 0
                        ):
                            stop_reason = (
                                "INCOMPLETE_FAIL_CLOSED_CELL_DIAGNOSTIC"
                            )
                        with state_lock:
                            state["committed_chunks"] = len(committed_indices)
                            state["exception_rows"] += int(
                                diagnostics["exception_rows"]
                            )
                    except BaseException as exc:
                        stop_reason = (
                            "INCOMPLETE_FAIL_CLOSED_WORKER_EXCEPTION:"
                            f"{type(exc).__name__}:{exc}"
                        )
                    with state_lock:
                        state["active_chunk_ids"] = [
                            value.chunk_id for value in active.values()
                        ]
                if stop_reason is not None:
                    for future in active:
                        future.cancel()
                    # Running workers cannot be killed without risking partial
                    # filesystem writes; the context waits for them, but no new
                    # chunk is admitted and this attempt remains terminal.
                    queue.clear()
        final_resource = heartbeat_snapshot()
        if heartbeat.error is not None:
            stop_reason = (
                "INCOMPLETE_FAIL_CLOSED_HEARTBEAT:"
                f"{type(heartbeat.error).__name__}:{heartbeat.error}"
            )
        if int(final_resource["peak_process_tree_rss_bytes"]) > int(
            config["resource_contract"]["maximum_peak_rss_bytes"]
        ):
            stop_reason = "INCOMPLETE_RESOURCE_FAIL_CLOSED_PROCESS_TREE_RSS"
        if stop_reason is not None or len(committed_indices) != len(plan):
            raise RuntimeError(
                stop_reason
                or "INCOMPLETE_FAIL_CLOSED_RECEIPT_COVERAGE"
            )
        store = ImmutableObjectStore(
            repository_root=root,
            object_root=object_root,
            staging_root=staging_root,
            receipt_root=receipt_root,
            task_id=str(config["task_id"]),
            run_id=run_id,
            config_sha256=str(binding["sha256"]),
            plan_sha256=plan_sha,
        )
        inventory = store.inventory([asdict(cell) for cell in plan])
        if inventory["raw_status"] != "RAW_EVIDENCE_COMPLETE_NO_SCIENTIFIC_VERDICT":
            raise RuntimeError("formal inventory is incomplete")
        _, manifest = publish_inventory_and_manifest(
            repository_root=root,
            inventory_path=root / str(paths["inventory"]),
            manifest_path=root / str(paths["execution_manifest"]),
            inventory=inventory,
            claim_fields=EXPECTED_CLAIM_FIELDS,
        )
        append_attempt_event(
            attempt_path,
            task_id=str(config["task_id"]),
            run_id=run_id,
            event="RAW_TRANSACTION_COMPLETE",
            payload={
                "receipt_count": len(plan),
                "inventory_sha256": inventory["inventory_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "scientific_verdict": None,
                "qualified_claim": None,
            },
        )
        return manifest
    except BaseException as exc:
        append_attempt_event(
            attempt_path,
            task_id=str(config["task_id"]),
            run_id=run_id,
            event="TERMINAL_INCOMPLETE",
            payload={
                "reason": f"{type(exc).__name__}:{exc}",
                "committed_chunks": len(committed_indices),
                "automatic_retry": False,
                "scientific_verdict": None,
                "qualified_claim": None,
            },
        )
        raise
    finally:
        heartbeat_error: BaseException | None = None
        if heartbeat_started:
            try:
                heartbeat.stop()
            except BaseException as exc:
                heartbeat_error = exc
        try:
            owner.release()
        finally:
            if heartbeat_error is not None and sys.exc_info()[0] is None:
                raise RuntimeError(
                    "heartbeat failed during supervisor teardown"
                ) from heartbeat_error


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the sealed T04 powered raw-evidence transaction."
    )
    parser.add_argument(
        "--launch",
        action="store_true",
        help="launch the single sealed production supervisor",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    root = _root()
    if not arguments.launch:
        raise RuntimeError("production requires explicit --launch")
    manifest = run_production(root)
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
