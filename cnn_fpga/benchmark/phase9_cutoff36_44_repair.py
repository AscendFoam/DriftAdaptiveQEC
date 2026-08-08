"""Bounded fresh cutoff-36/40/44 repair with expected RESET outputs.

This writer is the only raw-evidence producer for T-RISK-20260728-03.  It
loads every execution module from config-pinned source bytes, activates the
dedicated cap-44 adapter, runs resource/capability preflights before any
scientific chunk, and then commits a fresh 36/40/44 denominator cell by cell.

Fault cells reuse the frozen Phase-9 trajectory executor.  Shared RESET cells
replace the sampled branch as the *primary* density/level estimand with the
native-backend Rao--Blackwell expectation.  Conditional success/failure and
the unmodified sampled branch are archived in a bound sidecar and cannot vote
in the later diagnostic.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from hashlib import sha256
import importlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import psutil
import scipy


TASK_ID = "T-RISK-20260728-03"
CONFIG_PATH = "configs/phase9/t_risk_20260728_03_cutoff36_44_repair.json"
CONFIG_BYTES = 11057
CONFIG_SHA256 = "1b85f03e1b7650f36f41b2411894b88c802bb826a18b0760a143c6c5590e752e"
CONFIG_SCHEMA = "PHASE9-CUTOFF36-44-BOUNDED-REPAIR-CONFIG-V1"
MANIFEST_SCHEMA = "PHASE9-CUTOFF36-44-BOUNDED-REPAIR-MANIFEST-V1"
RECEIPT_SCHEMA = "PHASE9-CUTOFF36-44-BOUNDED-REPAIR-RECEIPT-V1"
RUN_IDENTITY_SCHEMA = "PHASE9-CUTOFF36-44-BOUNDED-REPAIR-RUN-IDENTITY-V1"
HEARTBEAT_SCHEMA = "PHASE9-CUTOFF36-44-BOUNDED-REPAIR-HEARTBEAT-V1"
RESOURCE_SCHEMA = "PHASE9-CUTOFF36-44-RESOURCE-PREFLIGHT-V1"
CAPABILITY_SCHEMA = "PHASE9-CUTOFF36-44-CAPABILITY-PREFLIGHT-V1"
RB_SIDECAR_SCHEMA = "PHASE9-RAO-BLACKWELL-RESET-SIDECAR-V1"
VERIFIED_LOADER_CONTRACT = "PHASE9-VERIFIED-SOURCE-BYTES-LOADER-V1"
STATUS = "CUTOFF36_44_BOUNDED_REPAIR_RAW_EVIDENCE_COMPLETE"
REJECTED_STATUS = "CUTOFF36_44_BOUNDED_REPAIR_RAW_EVIDENCE_REJECTED"
CLAIM_BOUNDARY = {
    "design_repair_only": True,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}
_SOURCE_AT_IMPORT = sha256(Path(__file__).read_bytes()).hexdigest()
_EXECUTION_MODULE_NAMES = {
    "backend_a": "physics.phase9_backend_a",
    "backend_b": "physics.phase9_backend_b",
    "cutoff44_adapter": "physics.phase9_cutoff44_runtime_adapter",
    "backend_b_bridge": "physics.phase9_backend_b_logical_bridge",
    "dual_backend_kernel": "cnn_fpga.benchmark.phase9_dual_backend_qualification",
    "fresh_runner": "cnn_fpga.benchmark.phase9_fresh_twin_qualification",
    "reset_rao_blackwell": "physics.phase9_reset_rao_blackwell",
    "iq_reference": "physics.phase9_iq_likelihood_reference",
    "twin_contract": "physics.phase9_twin_contract",
}
_VERIFIED_MODULES: dict[str, object] = {}
_VERIFIED_BINDINGS: dict[str, dict[str, object]] = {}
_ADAPTER_RECEIPT: dict[str, Any] = {}
runner: Any = None
rb_reset: Any = None


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


def _self_hash(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("analysis_sha256", None)
    return _sha(unsigned)


def _binding(path: Path, root: Path) -> dict[str, object]:
    resolved = path.resolve()
    relative = resolved.relative_to(root.resolve()).as_posix()
    payload = resolved.read_bytes()
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _read_bound_bytes(
    root: Path,
    binding: Mapping[str, Any],
) -> tuple[Path, bytes]:
    if set(binding) - {"path", "bytes", "sha256", "analysis_sha256"}:
        raise RuntimeError("bound artifact schema drift")
    path = (root / str(binding["path"])).resolve()
    path.relative_to(root.resolve())
    payload = path.read_bytes()
    if (
        len(payload) != int(binding["bytes"])
        or sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise RuntimeError(f"bound artifact drift: {binding['path']}")
    if "analysis_sha256" in binding:
        document = json.loads(payload)
        if (
            document.get("analysis_sha256") != binding["analysis_sha256"]
            or _self_hash(document) != binding["analysis_sha256"]
        ):
            raise RuntimeError(f"bound semantic hash drift: {binding['path']}")
    return path, payload


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_text = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


class _VerifiedBytesLoader(importlib.abc.Loader):
    def __init__(
        self,
        fullname: str,
        path: Path,
        payload: bytes,
        digest: str,
    ) -> None:
        self.fullname = fullname
        self.path = path
        self.payload = payload
        self.digest = digest

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: object) -> None:
        namespace = vars(module)
        namespace["__file__"] = str(self.path)
        namespace["__verified_source_sha256__"] = self.digest
        namespace["__verified_bootstrap_contract__"] = VERIFIED_LOADER_CONTRACT
        code = compile(self.payload, str(self.path), "exec", dont_inherit=True)
        exec(code, namespace)
        namespace["__verified_source_sha256__"] = self.digest
        namespace["__verified_bootstrap_contract__"] = VERIFIED_LOADER_CONTRACT


class _VerifiedBytesFinder(importlib.abc.MetaPathFinder):
    def __init__(
        self,
        frozen: Mapping[str, tuple[Path, bytes, str]],
    ) -> None:
        self.frozen = dict(frozen)

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> object:
        source = self.frozen.get(fullname)
        if source is None:
            return None
        source_path, payload, digest = source
        return importlib.util.spec_from_loader(
            fullname,
            _VerifiedBytesLoader(fullname, source_path, payload, digest),
            origin=str(source_path),
        )


def _drop_preloaded_modules() -> None:
    for module_name in reversed(tuple(_EXECUTION_MODULE_NAMES.values())):
        module = sys.modules.pop(module_name, None)
        if module is None or "." not in module_name:
            continue
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, attribute, None) is module:
            delattr(parent, attribute)


def _activate_verified_modules(
    root: Path,
    config: Mapping[str, Any],
) -> None:
    global runner, rb_reset, _ADAPTER_RECEIPT
    expected: dict[str, dict[str, object]] = {}
    frozen: dict[str, tuple[Path, bytes, str]] = {}
    sources = config["source_bindings"]
    for name, module_name in _EXECUTION_MODULE_NAMES.items():
        binding = sources.get(name)
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"verified source absent: {name}")
        path, payload = _read_bound_bytes(root, binding)
        digest = sha256(payload).hexdigest()
        expected[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
        }
        frozen[module_name] = (path, payload, digest)
    if _VERIFIED_BINDINGS:
        if _VERIFIED_BINDINGS != expected:
            raise RuntimeError("verified execution source-set drift")
        _assert_verified_modules()
        return

    _drop_preloaded_modules()
    finder = _VerifiedBytesFinder(frozen)
    sys.meta_path.insert(0, finder)
    loaded: dict[str, object] = {}
    try:
        for name, module_name in _EXECUTION_MODULE_NAMES.items():
            module = importlib.import_module(module_name)
            if (
                getattr(module, "__verified_source_sha256__", None)
                != expected[name]["sha256"]
            ):
                raise RuntimeError(f"module not loaded from verified bytes: {name}")
            loaded[name] = module
            if name == "cutoff44_adapter":
                if set(loaded) != {
                    "backend_a",
                    "backend_b",
                    "cutoff44_adapter",
                }:
                    raise RuntimeError("cutoff44 adapter activation-order drift")
                _ADAPTER_RECEIPT = module.enable_verified_cutoff44(
                    loaded["backend_a"],
                    loaded["backend_b"],
                )
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
    _VERIFIED_BINDINGS.update(expected)
    _VERIFIED_MODULES.update(loaded)
    runner = loaded["fresh_runner"]
    rb_reset = loaded["reset_rao_blackwell"]
    _assert_verified_modules()


def _assert_verified_modules() -> None:
    if set(_VERIFIED_BINDINGS) != set(_EXECUTION_MODULE_NAMES) or set(
        _VERIFIED_MODULES
    ) != set(_EXECUTION_MODULE_NAMES):
        raise RuntimeError("verified execution module set incomplete")
    for name, module_name in _EXECUTION_MODULE_NAMES.items():
        module = _VERIFIED_MODULES[name]
        if (
            sys.modules.get(module_name) is not module
            or getattr(module, "__verified_source_sha256__", None)
            != _VERIFIED_BINDINGS[name]["sha256"]
        ):
            raise RuntimeError(f"verified execution attestation drift: {name}")
    _VERIFIED_MODULES["cutoff44_adapter"].assert_verified_cutoff44(
        _VERIFIED_MODULES["backend_a"],
        _VERIFIED_MODULES["backend_b"],
        _ADAPTER_RECEIPT,
    )


def _validate_config(root: Path, config: Mapping[str, Any]) -> None:
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("claim_boundary") != CLAIM_BOUNDARY
        or config.get("analysis_sha256") != _self_hash(config)
        or config.get("cutoffs") != [36, 40, 44]
        or config.get("required_consecutive_increments") != [[36, 40], [40, 44]]
        or config.get("absolute_tail_cutoff") != 44
        or config.get("automatic_cutoff_extension_beyond_44") is not False
        or config.get("terminal_if_cutoff44_fails") is not True
        or config.get("logical_state_schedule") != ["0", "1", "+", "-", "+i", "-i"]
        or config.get("scenario_names") != ["step", "telegraph", "burst", "compound"]
        or config.get("trajectory_count") != 72
        or config.get("clusters_per_state") != 12
        or config.get("max_workers") != 2
    ):
        raise RuntimeError("bounded repair identity/claim contract drift")
    expected_accounting = {
        "fault_cells": 24,
        "shared_reset_cells": 6,
        "total_cells": 30,
        "fault_rows": 20736,
        "shared_rows": 432,
        "total_rows": 21168,
        "fault_terminal_densities": 1728,
        "shared_expected_densities": 432,
        "total_primary_densities": 2160,
        "shared_conditional_success_densities": 432,
        "shared_conditional_failure_densities": 432,
        "shared_sampled_stress_densities": 432,
    }
    if config.get("matrix_accounting") != expected_accounting:
        raise RuntimeError("bounded repair matrix accounting drift")
    reset = config.get("shared_reset", {})
    if (
        reset.get("initial_state") != "vacuum_f"
        or reset.get("action") != "RESET"
        or reset.get("primary_estimand")
        != "RAO_BLACKWELLIZED_EXPECTED_POST_RESET_DENSITY_AND_LEVELS_V1"
        or reset.get("sampled_branch_role")
        != "SAMPLED_NATIVE_RESET_BRANCH_NONVOTING_STRESS_ONLY"
        or reset.get("same_native_record_for_counterfactual_branches") is not True
        or reset.get("conditional_branch_density_archive_required") is not True
        or reset.get("sampled_branch_density_archive_required") is not True
        or reset.get("nondegenerate_success_probability_required") is not True
        or reset.get("backend_shared_transition_or_rng") is not False
    ):
        raise RuntimeError("Rao-Blackwell RESET contract drift")
    diagnostic = config.get("diagnostic_contract", {})
    if (
        diagnostic.get("required_consecutive_increments") != [[36, 40], [40, 44]]
        or diagnostic.get("absolute_tail_cutoff") != 44
        or diagnostic.get("expected_gate_count") != 1454
        or diagnostic.get("all_1454_gates_must_pass") is not True
        or diagnostic.get("powered_formal_release") is not False
        or diagnostic.get("old_gate_or_raw_composition_forbidden") is not True
    ):
        raise RuntimeError("bounded repair diagnostic contract drift")
    for scenario in config["scenario_names"]:
        partition = config["stage_partition"][scenario]
        indices = [item for values in partition.values() for item in values]
        if sorted(indices) != list(range(12)) or len(indices) != 12:
            raise RuntimeError(f"stage partition drift: {scenario}")
    intervals: list[set[int]] = []
    splits = config["seed_splits"]
    for name in (
        "trajectory_backend_a",
        "trajectory_backend_b",
        "heldout_common",
        "round_backend_a",
        "round_backend_b",
    ):
        start = int(splits[name]["start"])
        count = int(splits[name]["count"])
        if count != 72:
            raise RuntimeError("scientific seed denominator drift")
        intervals.append(set(range(start, start + count)))
    if (
        splits.get("all_intervals_disjoint") is not True
        or splits.get("fresh_from_prior_design_and_uq") is not True
        or any(
            intervals[left] & intervals[right]
            for left in range(len(intervals))
            for right in range(left)
        )
    ):
        raise RuntimeError("scientific seed firewall drift")
    resource = config.get("resource_preflight", {})
    if (
        resource.get("required_before_scientific_chunk") is not True
        or resource.get("benchmark_cutoffs") != [40, 44]
        or resource.get("benchmark_trajectories_per_state") != 1
        or resource.get("include_rao_blackwell_shared_reset") is not True
        or resource.get("design_outcomes_accessed") is not False
    ):
        raise RuntimeError("resource preflight contract drift")
    capability = config.get("capability_preflight", {})
    if (
        capability.get("required_before_scientific_chunk") is not True
        or capability.get("cutoffs") != [40, 44]
        or capability.get("integration_segment_steps") != [8, 16, 32]
        or capability.get("cutoff45_rejected_before_allocation") is not True
    ):
        raise RuntimeError("capability preflight contract drift")
    future = config.get("future_powered_formal", {})
    if (
        future.get("released") is not False
        or future.get("automatic_execution_from_this_task") is not False
        or future.get("old_design_rows_vote") is not False
        or future.get("official_puviani_sota_claims") is not None
    ):
        raise RuntimeError("powered-formal release firewall drift")
    trigger = config.get("repair_trigger", {})
    if (
        trigger.get("old_raw_rows_vote") is not False
        or trigger.get("old_passing_gates_vote") is not False
        or trigger.get("required_diagnosis_verdict")
        != "PHYSICS_AND_ESTIMAND_REPAIR_REQUIRED"
        or trigger.get("required_scientific_verdict") != "NO_GO_HIGH_CUTOFF_DESIGN"
    ):
        raise RuntimeError("prior NO-GO composition firewall drift")
    for binding in (
        config["base_config"],
        *config["repair_trigger"].values(),
        *config["source_bindings"].values(),
    ):
        if isinstance(binding, Mapping) and "path" in binding:
            _read_bound_bytes(root, binding)
    _, diagnosis_payload = _read_bound_bytes(
        root,
        config["repair_trigger"]["diagnosis"],
    )
    diagnosis = json.loads(diagnosis_payload)
    if (
        diagnosis.get("diagnosis_verdict")
        != config["repair_trigger"]["required_diagnosis_verdict"]
        or diagnosis.get("scientific_verdict_unchanged")
        != config["repair_trigger"]["required_scientific_verdict"]
        or diagnosis.get("bounded_repair_preregistration", {}).get("fresh_cutoffs")
        != [36, 40, 44]
    ):
        raise RuntimeError("repair trigger semantic drift")


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / CONFIG_PATH
    payload = path.read_bytes()
    if len(payload) != CONFIG_BYTES or sha256(payload).hexdigest() != CONFIG_SHA256:
        raise RuntimeError("bounded repair config byte binding drift")
    config = json.loads(payload)
    _validate_config(root, config)
    _, base_payload = _read_bound_bytes(root, config["base_config"])
    base = json.loads(base_payload)
    return config, base


def materialize_execution(
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    *,
    resource: bool = False,
) -> dict[str, Any]:
    execution = json.loads(json.dumps(base))
    source = (
        config["resource_preflight"]["seed_splits"]
        if resource
        else config["seed_splits"]
    )
    count = 6 if resource else int(config["trajectory_count"])
    execution["formal_matrix"]["trajectory_sample_count"] = count
    execution["formal_matrix"]["cutoff_ladder"] = [36, 40, 44]
    for key in (
        "trajectory_backend_a",
        "trajectory_backend_b",
        "heldout_common",
        "round_backend_a",
        "round_backend_b",
    ):
        execution["formal_splits"][key] = {
            "start": int(source[key]["start"]),
            "count": int(source[key]["count"]),
        }
    execution["artifact_paths"]["chunk_directory"] = config["artifact_paths"][
        "chunk_directory"
    ]
    return execution


def _chunk_id(identity: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in identity)
    return f"{safe}__{sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def build_cells(
    config: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    resource: bool = False,
) -> list[Any]:
    count = 6 if resource else int(config["trajectory_count"])
    cutoffs = (
        config["resource_preflight"]["benchmark_cutoffs"]
        if resource
        else config["cutoffs"]
    )
    cells: list[Any] = []
    for cutoff in cutoffs:
        for scenario in config["scenario_names"]:
            horizon = int(
                execution["formal_matrix"]["fault_scenarios"][scenario]["horizon"]
            )
            for backend in ("A", "B"):
                identity = (
                    f"{'resource' if resource else 'repair'}|c{cutoff}|"
                    f"fault|{scenario}|{backend}"
                )
                cells.append(
                    runner.CellSpec(
                        chunk_id=_chunk_id(identity),
                        layer="fault",
                        cell_base=f"fault|{scenario}",
                        cutoff=int(cutoff),
                        backend=backend,
                        sample_count=count,
                        convergence_role=(
                            "resource_preflight_only"
                            if resource
                            else "fresh_cutoff36_40_44_bounded_repair"
                        ),
                        scenario=scenario,
                        horizon=horizon,
                    )
                )
    reset = config["shared_reset"]
    for cutoff in cutoffs:
        for backend in ("A", "B"):
            identity = (
                f"{'resource' if resource else 'repair'}|c{cutoff}|"
                f"shared|{reset['initial_state']}|{reset['action']}|{backend}"
            )
            cells.append(
                runner.CellSpec(
                    chunk_id=_chunk_id(identity),
                    layer="shared",
                    cell_base=(f"shared|{reset['initial_state']}|{reset['action']}"),
                    cutoff=int(cutoff),
                    backend=backend,
                    sample_count=count,
                    convergence_role=(
                        "resource_preflight_only"
                        if resource
                        else "rao_blackwell_expected_reset_repair"
                    ),
                    action=reset["action"],
                    initial_state=reset["initial_state"],
                    horizon=1,
                )
            )
    expected = 20 if resource else 30
    expected_rows = 16 * count * 12 + 4 * count if resource else 21168
    if (
        len(cells) != expected
        or len({cell.chunk_id for cell in cells}) != expected
        or sum(cell.expected_rows for cell in cells) != expected_rows
    ):
        raise RuntimeError("repair cell accounting drift")
    return cells


def _density_quantization(
    density: np.ndarray,
) -> tuple[float, float, float]:
    restored = density.astype(np.complex64).astype(np.complex128)
    exact = float(np.linalg.norm(density - restored, ord="fro"))
    unit_roundoff = 2.0**-24
    certified = float(
        unit_roundoff / (1.0 - unit_roundoff) * np.linalg.norm(restored, ord="fro")
        + np.sqrt(2.0 * density.size) * 2.0**-150
    )
    if exact > certified:
        raise RuntimeError("expected RESET quantization exceeded certificate")
    trace_bound = float(0.5 * np.sqrt(density.shape[0]) * certified)
    return exact, certified, trace_bound


def _mean_photon(simulator: object, density: np.ndarray, cutoff: int) -> float:
    oscillator = simulator.oscillator_density(density)
    matrix = oscillator.matrix if hasattr(oscillator, "matrix") else oscillator
    return float(np.trace(np.diag(np.arange(cutoff, dtype=np.float64)) @ matrix).real)


def _execute_shared_rb(
    config: Mapping[str, Any],
    execution: Mapping[str, Any],
    cell: Any,
    simulator: object,
) -> tuple[Any, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    densities: list[np.ndarray] = []
    density_ids: list[str] = []
    raw_windows: list[np.ndarray] = []
    heldout_windows: list[np.ndarray] = []
    probabilities: list[float] = []
    success_densities: list[np.ndarray] = []
    failure_densities: list[np.ndarray] = []
    sampled_densities: list[np.ndarray] = []
    outcomes: list[str] = []
    branch_distances: list[float] = []
    sampled_match: list[float] = []
    actions = runner._action_words()
    for position in range(cell.sample_count):
        seed = runner._seed_for(execution, cell, position)
        heldout = runner._heldout_window(
            execution,
            cell_base=cell.cell_base,
            cutoff=cell.cutoff,
            position=position,
            round_index=0,
        )
        identity = runner._identity(
            cell,
            seed=seed,
            position=position,
            row_index=position,
            action=cell.action,
        )
        state, evaluator = runner._initial_state(cell, simulator)
        evidence = rb_reset.evaluate_expected_reset(
            backend=cell.backend,
            simulator=simulator,
            state=state,
            evaluator=evaluator,
            action=actions[cell.action],
            seed=seed,
        )
        if (
            evidence.success_density is None
            or evidence.failure_density is None
            or not 0.0 < evidence.success_probability < 1.0
        ):
            raise RuntimeError("shared RESET nondegenerate branch contract drift")
        row, raw_iq = runner._success_row(
            config=execution,
            identity=identity,
            simulator=simulator,
            result=evidence.sampled_result,
            action=actions[cell.action],
            heldout=heldout,
            density_index=len(densities),
        )
        density = np.asarray(evidence.expected_density, dtype=np.complex128)
        diagnostics = runner._density_diagnostics(density)
        exact, certified, trace_bound = _density_quantization(density)
        row.update(
            {
                "level_g": evidence.expected_levels[0],
                "level_e": evidence.expected_levels[1],
                "level_f": evidence.expected_levels[2],
                "mean_photon": _mean_photon(simulator, density, cell.cutoff),
                "rao_blackwell_reset_success": evidence.success_probability,
                "leakage_residence_probability": evidence.expected_levels[2],
                "density_trace_error": diagnostics[0],
                "density_hermiticity_frobenius": diagnostics[1],
                "density_minimum_eigenvalue": diagnostics[2],
                "density_quantization_frobenius_error": exact,
                "density_quantization_certified_frobenius_bound": certified,
                "density_quantization_trace_distance_bound": trace_bound,
                "level_normalization_error": abs(sum(evidence.expected_levels) - 1.0),
                "conservation_pass": (
                    bool(row["conservation_pass"])
                    and diagnostics[0] <= 5.0e-8
                    and diagnostics[1] <= 5.0e-8
                    and diagnostics[2] >= -5.0e-8
                    and abs(sum(evidence.expected_levels) - 1.0) <= 5.0e-8
                ),
            }
        )
        rows.append(row)
        densities.append(density.astype(np.complex64))
        density_ids.append(str(row["row_id"]))
        raw_windows.append(raw_iq)
        heldout_windows.append(heldout)
        probabilities.append(evidence.success_probability)
        success_densities.append(evidence.success_density.astype(np.complex64))
        failure_densities.append(evidence.failure_density.astype(np.complex64))
        sampled_densities.append(evidence.sampled_density.astype(np.complex64))
        outcomes.append(evidence.sampled_hidden_outcome)
        branch_distances.append(evidence.branch_trace_distance)
        sampled_match.append(evidence.sampled_matches_forced_branch_trace_distance)
    chunk = runner.ChunkEvidence(
        rows=rows,
        densities=densities,
        density_row_ids=density_ids,
        raw_iq=np.stack(raw_windows),
        heldout_iq=np.stack(heldout_windows),
    )
    width = max(len(value) for value in density_ids)
    sidecar = {
        "schema": np.asarray([RB_SIDECAR_SCHEMA]),
        "chunk_id": np.asarray([cell.chunk_id]),
        "cutoff": np.asarray([cell.cutoff], dtype=np.int64),
        "row_ids": np.asarray(density_ids, dtype=f"<U{width}"),
        "success_probability": np.asarray(probabilities, dtype=np.float64),
        "conditional_success_densities": np.stack(success_densities),
        "conditional_failure_densities": np.stack(failure_densities),
        "sampled_stress_densities": np.stack(sampled_densities),
        "sampled_hidden_outcome": np.asarray(outcomes, dtype="<U7"),
        "branch_trace_distance": np.asarray(branch_distances, dtype=np.float64),
        "sampled_match_trace_distance": np.asarray(sampled_match, dtype=np.float64),
    }
    return chunk, sidecar


def _rb_sidecar_path(
    root: Path,
    config: Mapping[str, Any],
    cell: Any,
) -> Path:
    return (
        root
        / str(config["artifact_paths"]["chunk_directory"])
        / f"{cell.chunk_id}.rb.npz"
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_rb_sidecar(
    root: Path,
    receipt: Mapping[str, Any],
    cell: Any,
) -> None:
    binding = receipt.get("rao_blackwell_sidecar")
    if cell.layer != "shared":
        if binding is not None:
            raise RuntimeError("fault cell has prohibited RESET sidecar")
        return
    if not isinstance(binding, Mapping):
        raise RuntimeError("shared cell RESET sidecar missing")
    path, payload = _read_bound_bytes(root, binding)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        expected_fields = {
            "schema",
            "chunk_id",
            "cutoff",
            "row_ids",
            "success_probability",
            "conditional_success_densities",
            "conditional_failure_densities",
            "sampled_stress_densities",
            "sampled_hidden_outcome",
            "branch_trace_distance",
            "sampled_match_trace_distance",
        }
        if set(archive.files) != expected_fields:
            raise RuntimeError("RESET sidecar schema drift")
        schema = archive["schema"].tolist()
        chunk_ids = archive["chunk_id"].tolist()
        cutoffs = archive["cutoff"].tolist()
        row_ids = archive["row_ids"].tolist()
        probabilities = np.asarray(archive["success_probability"])
        success = np.asarray(archive["conditional_success_densities"])
        failure = np.asarray(archive["conditional_failure_densities"])
        sampled = np.asarray(archive["sampled_stress_densities"])
        outcomes = archive["sampled_hidden_outcome"].tolist()
        branch = np.asarray(archive["branch_trace_distance"])
        matches = np.asarray(archive["sampled_match_trace_distance"])
    if (
        schema != [RB_SIDECAR_SCHEMA]
        or chunk_ids != [cell.chunk_id]
        or cutoffs != [cell.cutoff]
        or len(row_ids) != cell.sample_count
        or len(set(row_ids)) != cell.sample_count
        or probabilities.shape != (cell.sample_count,)
        or success.shape != (cell.sample_count, 3 * cell.cutoff, 3 * cell.cutoff)
        or failure.shape != success.shape
        or sampled.shape != success.shape
        or success.dtype != np.complex64
        or failure.dtype != np.complex64
        or sampled.dtype != np.complex64
        or not np.all((probabilities > 0.0) & (probabilities < 1.0))
        or not np.all(np.isfinite(probabilities))
        or not np.all(np.isfinite(success))
        or not np.all(np.isfinite(failure))
        or not np.all(np.isfinite(sampled))
        or set(outcomes) - {"success", "failure"}
        or branch.shape != (cell.sample_count,)
        or matches.shape != (cell.sample_count,)
        or np.any(branch < 0.0)
        or np.any(matches > 2.0e-10)
    ):
        raise RuntimeError("RESET sidecar shape/semantic drift")
    npz_binding = receipt["npz"]
    _, main_payload = _read_bound_bytes(root, npz_binding)
    with np.load(io.BytesIO(main_payload), allow_pickle=False) as main:
        primary_ids = main["density_row_ids"].tolist()
        primary = np.asarray(main["densities"], dtype=np.complex128)
    mixture = probabilities[:, None, None] * success.astype(np.complex128) + (
        1.0 - probabilities[:, None, None]
    ) * failure.astype(np.complex128)
    mixture_error = np.asarray(
        [
            np.linalg.norm(left - right, ord="fro")
            for left, right in zip(primary, mixture, strict=True)
        ]
    )
    selected = np.stack(
        [
            success[index] if outcome == "success" else failure[index]
            for index, outcome in enumerate(outcomes)
        ]
    ).astype(np.complex128)
    sampled_error = np.asarray(
        [
            np.linalg.norm(left - right, ord="fro")
            for left, right in zip(sampled.astype(np.complex128), selected, strict=True)
        ]
    )
    if (
        row_ids != primary_ids
        or float(np.max(mixture_error)) > 3.0e-6
        or float(np.max(sampled_error)) > 3.0e-6
    ):
        raise RuntimeError("RESET sidecar branch-mixture alignment drift")
    if path != (root / str(binding["path"])).resolve():
        raise RuntimeError("RESET sidecar path drift")


def _receipt_path(
    root: Path,
    config: Mapping[str, Any],
    cell: Any,
) -> Path:
    return (
        root
        / str(config["artifact_paths"]["receipt_directory"])
        / f"{cell.chunk_id}.json"
    )


def _execute_and_commit_cell(
    root: Path,
    config: Mapping[str, Any],
    execution: Mapping[str, Any],
    cell: Any,
    run_identity: Mapping[str, Any],
    input_snapshot_sha256: str,
) -> dict[str, Any]:
    _assert_snapshot(root, config, input_snapshot_sha256)
    receipt_path = _receipt_path(root, config, cell)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_bytes())
        _validate_receipt(
            root,
            config,
            execution,
            cell,
            receipt,
            run_identity,
            input_snapshot_sha256,
        )
        return receipt
    simulator = runner.build_simulators(execution, cell.cutoff)[cell.backend]
    if cell.layer == "shared":
        evidence, sidecar = _execute_shared_rb(
            config,
            execution,
            cell,
            simulator,
        )
    else:
        evidence = runner.execute_cell(
            execution,
            cell,
            simulator,
            runner._action_words(),
        )
        sidecar = None
    _assert_snapshot(root, config, input_snapshot_sha256)
    chunk = runner.write_chunk(root, execution, cell, evidence)
    receipt: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": RECEIPT_SCHEMA,
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "config_analysis_sha256": config["analysis_sha256"],
        "input_snapshot_analysis_sha256": input_snapshot_sha256,
        "writer_source_sha256": _SOURCE_AT_IMPORT,
        "cell": asdict(cell),
        **chunk,
    }
    if sidecar is not None:
        sidecar_path = _rb_sidecar_path(root, config, cell)
        _write_npz(sidecar_path, sidecar)
        receipt["rao_blackwell_sidecar"] = _binding(sidecar_path, root)
    receipt["analysis_sha256"] = _self_hash(receipt)
    _atomic_json(receipt_path, receipt)
    _validate_receipt(
        root,
        config,
        execution,
        cell,
        receipt,
        run_identity,
        input_snapshot_sha256,
    )
    return receipt


def _validate_receipt(
    root: Path,
    config: Mapping[str, Any],
    execution: Mapping[str, Any],
    cell: Any,
    receipt: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    input_snapshot_sha256: str,
) -> None:
    if (
        receipt.get("task_id") != TASK_ID
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("analysis_sha256") != _self_hash(receipt)
        or receipt.get("run_id") != run_identity["run_id"]
        or receipt.get("run_identity_analysis_sha256")
        != run_identity["analysis_sha256"]
        or receipt.get("config_analysis_sha256") != config["analysis_sha256"]
        or receipt.get("input_snapshot_analysis_sha256") != input_snapshot_sha256
        or receipt.get("writer_source_sha256") != _SOURCE_AT_IMPORT
        or receipt.get("cell") != asdict(cell)
        or receipt.get("exception_rows") != 0
        or receipt.get("observed_rows") != cell.expected_rows
        or receipt.get("expected_rows") != cell.expected_rows
    ):
        raise RuntimeError("repair receipt identity/completeness drift")
    runner._validate_chunk_files(root, receipt, cell)
    _validate_rb_sidecar(root, receipt, cell)
    path = _receipt_path(root, config, cell)
    if path.exists() and json.loads(path.read_bytes()) != receipt:
        raise RuntimeError("repair receipt file/content drift")


def _input_snapshot(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "config": _binding(root / CONFIG_PATH, root),
        "writer": _binding(Path(__file__).resolve(), root),
        "base_config": dict(config["base_config"]),
        "diagnosis": dict(config["repair_trigger"]["diagnosis"]),
        "independent_no_go_verification": dict(
            config["repair_trigger"]["independent_no_go_verification"]
        ),
    }
    snapshot.update(
        {
            f"source/{name}": dict(binding)
            for name, binding in config["source_bindings"].items()
        }
    )
    snapshot["analysis_sha256"] = _self_hash(snapshot)
    return snapshot


def _assert_snapshot(
    root: Path,
    config: Mapping[str, Any],
    expected_sha256: str,
) -> None:
    live = _input_snapshot(root, config)
    if live["analysis_sha256"] != expected_sha256:
        raise RuntimeError("repair input snapshot drift")
    _assert_verified_modules()


def _git_state(root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    return {
        "head": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "status_short": command("status", "--short"),
    }


def _load_or_create_run_identity(
    root: Path,
    config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / str(config["artifact_paths"]["run_identity"])
    if path.exists():
        identity = json.loads(path.read_bytes())
    else:
        identity = {
            "task_id": TASK_ID,
            "schema_version": RUN_IDENTITY_SCHEMA,
            "run_id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_analysis_sha256": config["analysis_sha256"],
            "input_snapshot_analysis_sha256": snapshot["analysis_sha256"],
            "writer_source_sha256": _SOURCE_AT_IMPORT,
            "git": _git_state(root),
            "claim_state": dict(CLAIM_BOUNDARY),
        }
        identity["analysis_sha256"] = _self_hash(identity)
        _atomic_json(path, identity)
    if (
        identity.get("task_id") != TASK_ID
        or identity.get("schema_version") != RUN_IDENTITY_SCHEMA
        or identity.get("analysis_sha256") != _self_hash(identity)
        or identity.get("config_analysis_sha256") != config["analysis_sha256"]
        or identity.get("input_snapshot_analysis_sha256") != snapshot["analysis_sha256"]
        or identity.get("writer_source_sha256") != _SOURCE_AT_IMPORT
        or identity.get("claim_state") != CLAIM_BOUNDARY
    ):
        raise RuntimeError("repair run identity drift")
    return identity


@contextmanager
def _owner_lock(
    root: Path,
    config: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    path = root / str(config["artifact_paths"]["owner_lock"])
    path.parent.mkdir(parents=True, exist_ok=True)
    token = str(uuid4())
    document = {
        "task_id": TASK_ID,
        "pid": os.getpid(),
        "token": token,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        )
    except FileExistsError as exc:
        raise RuntimeError("repair owner lock already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(document))
            stream.flush()
            os.fsync(stream.fileno())
        yield document
    finally:
        if path.exists():
            live = json.loads(path.read_bytes())
            if live.get("token") != token or live.get("pid") != os.getpid():
                raise RuntimeError("repair owner lock ownership drift")
            path.unlink()


def _heartbeat(
    root: Path,
    config: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    *,
    state: str,
    completed: int,
    total: int,
    error: BaseException | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    document: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": HEARTBEAT_SCHEMA,
        "run_id": run_identity["run_id"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "state": state,
        "completed_cells": completed,
        "total_cells": total,
        "active": state in {"PREFLIGHT", "RUNNING"},
        "manifest_analysis_sha256": (
            manifest.get("analysis_sha256") if manifest is not None else None
        ),
        "error_type": type(error).__name__ if error is not None else None,
        "claim_state": dict(CLAIM_BOUNDARY),
    }
    document["analysis_sha256"] = _self_hash(document)
    _atomic_json(
        root / str(config["artifact_paths"]["heartbeat"]),
        document,
    )


def _half_trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.sum(np.linalg.svd(left - right, compute_uv=False)))


def _run_capability_preflight(
    root: Path,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / str(config["artifact_paths"]["capability_preflight"])
    if path.exists():
        report = json.loads(path.read_bytes())
        if (
            report.get("schema_version") != CAPABILITY_SCHEMA
            or report.get("analysis_sha256") != _self_hash(report)
            or report.get("passed") is not True
            or report.get("design_outcomes_accessed") is not False
        ):
            raise RuntimeError("existing capability preflight drift")
        return report
    contract = config["capability_preflight"]
    actions = runner._action_words()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for cutoff in contract["cutoffs"]:
        grids: dict[tuple[str, str, str, int], np.ndarray] = {}
        for segment_steps in contract["integration_segment_steps"]:
            execution = json.loads(json.dumps(base))
            execution["common_physics"]["segment_steps"] = segment_steps
            simulators = runner.build_simulators(execution, cutoff)
            for backend in contract["backends"]:
                simulator = simulators[backend]
                for state_index, label in enumerate(contract["logical_states"]):
                    for action_index, action_name in enumerate(contract["actions"]):
                        state, evaluator = simulator.initialize_logical(label)
                        result = _VERIFIED_MODULES["dual_backend_kernel"]._one_step(
                            backend=backend,
                            simulator=simulator,
                            state=state,
                            evaluator=evaluator,
                            action=actions[action_name],
                            seed=(
                                1_791_000
                                + 10_000 * cutoff
                                + 100 * state_index
                                + action_index
                            ),
                        )
                        density = np.asarray(
                            result.state.joint_density, dtype=np.complex128
                        )
                        if (
                            density.shape != (3 * cutoff, 3 * cutoff)
                            or not np.all(np.isfinite(density))
                            or abs(np.trace(density) - 1.0) > 5.0e-8
                        ):
                            raise RuntimeError("capability preflight physicality drift")
                        grids[(backend, label, action_name, segment_steps)] = density
        for backend in contract["backends"]:
            for label in contract["logical_states"]:
                for action_name in contract["actions"]:
                    coarse = _half_trace_distance(
                        grids[(backend, label, action_name, 8)],
                        grids[(backend, label, action_name, 16)],
                    )
                    fine = _half_trace_distance(
                        grids[(backend, label, action_name, 16)],
                        grids[(backend, label, action_name, 32)],
                    )
                    ratio = fine / max(coarse, np.finfo(np.float64).eps)
                    passed = (
                        coarse <= contract["coarse_to_middle_max_trace_distance"]
                        and fine <= contract["middle_to_fine_max_trace_distance"]
                        and ratio <= contract["refinement_ratio_max"]
                    )
                    rows.append(
                        {
                            "cutoff": cutoff,
                            "backend": backend,
                            "state": label,
                            "action": action_name,
                            "trace_distance_8_to_16": coarse,
                            "trace_distance_16_to_32": fine,
                            "refinement_ratio": ratio,
                            "passed": passed,
                        }
                    )
    for module, config_name in (
        (_VERIFIED_MODULES["backend_a"], "BackendAConfig"),
        (_VERIFIED_MODULES["backend_b"], "BackendBConfig"),
    ):
        try:
            getattr(module, config_name)(cutoff=45)
        except ValueError:
            pass
        else:
            raise RuntimeError("capability preflight accepted cutoff45")
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": CAPABILITY_SCHEMA,
        "cutoffs": list(contract["cutoffs"]),
        "rows": rows,
        "row_count": len(rows),
        "cutoff45_rejected": True,
        "passed": all(row["passed"] for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "design_outcomes_accessed": False,
        "qualified_claim": None,
        "claim_state": dict(CLAIM_BOUNDARY),
    }
    if report["row_count"] != 144 or report["passed"] is not True:
        raise RuntimeError("capability preflight failed")
    report["analysis_sha256"] = _self_hash(report)
    _atomic_json(path, report)
    return report


def _evidence_nbytes(evidence: Any, sidecar: Mapping[str, np.ndarray] | None) -> int:
    value = int(evidence.raw_iq.nbytes + evidence.heldout_iq.nbytes)
    value += sum(int(density.nbytes) for density in evidence.densities)
    if sidecar is not None:
        value += sum(
            int(array.nbytes)
            for name, array in sidecar.items()
            if name
            in {
                "conditional_success_densities",
                "conditional_failure_densities",
                "sampled_stress_densities",
            }
        )
    return value


def _run_resource_preflight(
    root: Path,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / str(config["artifact_paths"]["resource_preflight"])
    if path.exists():
        report = json.loads(path.read_bytes())
        if (
            report.get("schema_version") != RESOURCE_SCHEMA
            or report.get("analysis_sha256") != _self_hash(report)
            or report.get("passed") is not True
            or report.get("design_outcomes_accessed") is not False
        ):
            raise RuntimeError("existing resource preflight drift")
        return report
    contract = config["resource_preflight"]
    execution = materialize_execution(config, base, resource=True)
    cells = build_cells(config, execution, resource=True)
    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    max_delta = 0
    artifact_bytes = 0
    records: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for cell in cells:
        started = time.perf_counter()
        before = process.memory_info().rss
        simulator = runner.build_simulators(execution, cell.cutoff)[cell.backend]
        if cell.layer == "shared":
            evidence, sidecar = _execute_shared_rb(
                config,
                execution,
                cell,
                simulator,
            )
        else:
            evidence = runner.execute_cell(
                execution,
                cell,
                simulator,
                runner._action_words(),
            )
            sidecar = None
        elapsed = time.perf_counter() - started
        after = process.memory_info().rss
        max_delta = max(
            max_delta,
            max(0, after - before),
            max(0, after - baseline_rss),
        )
        bytes_value = _evidence_nbytes(evidence, sidecar)
        artifact_bytes += bytes_value
        exception_rows = sum(bool(row["exception_type"]) for row in evidence.rows)
        conservation_failures = sum(
            not bool(row["conservation_pass"])
            for row in evidence.rows
            if not row["exception_type"]
        )
        records.append(
            {
                "cell": asdict(cell),
                "elapsed_seconds": elapsed,
                "rss_before_bytes": before,
                "rss_after_bytes": after,
                "estimated_uncompressed_artifact_bytes": bytes_value,
                "exception_rows": exception_rows,
                "conservation_failure_rows": conservation_failures,
            }
        )
        if exception_rows or conservation_failures:
            raise RuntimeError("resource preflight scientific path failure")
        del simulator, evidence, sidecar
        gc.collect()
    elapsed_total = time.perf_counter() - started_all
    scale = int(config["trajectory_count"]) / 6.0
    estimated_wall = (
        elapsed_total
        * scale
        / int(config["max_workers"])
        * float(contract["wall_safety_factor"])
    )
    estimated_rss = int(
        baseline_rss
        + max(
            max_delta,
            int(contract["minimum_per_worker_delta_bytes"]),
        )
        * int(config["max_workers"])
        * float(contract["rss_delta_safety_factor"])
    )
    estimated_artifact = int(
        artifact_bytes * scale * float(contract["artifact_safety_factor"])
    )
    free_disk = shutil.disk_usage(root).free
    passed = (
        estimated_wall <= int(contract["maximum_estimated_wall_seconds"])
        and estimated_rss <= int(contract["maximum_estimated_total_rss_bytes"])
        and estimated_artifact <= int(contract["maximum_estimated_artifact_bytes"])
        and free_disk >= int(contract["minimum_free_disk_bytes"])
    )
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": RESOURCE_SCHEMA,
        "benchmark_records": records,
        "benchmark_cell_count": len(records),
        "baseline_rss_bytes": baseline_rss,
        "maximum_observed_rss_delta_bytes": max_delta,
        "estimated_wall_seconds_with_safety_factor": estimated_wall,
        "estimated_total_rss_bytes": estimated_rss,
        "estimated_artifact_bytes": estimated_artifact,
        "free_disk_bytes": free_disk,
        "limits": {
            "wall_seconds": contract["maximum_estimated_wall_seconds"],
            "rss_bytes": contract["maximum_estimated_total_rss_bytes"],
            "artifact_bytes": contract["maximum_estimated_artifact_bytes"],
            "minimum_free_disk_bytes": contract["minimum_free_disk_bytes"],
        },
        "configured_max_workers": config["max_workers"],
        "design_outcomes_accessed": False,
        "passed": passed,
        "qualified_claim": None,
        "claim_state": dict(CLAIM_BOUNDARY),
    }
    if len(records) != 20 or not passed:
        raise RuntimeError("resource preflight failed")
    report["analysis_sha256"] = _self_hash(report)
    _atomic_json(path, report)
    return report


def _verify_manifest(
    root: Path,
    config: Mapping[str, Any],
    execution: Mapping[str, Any],
    cells: Sequence[Any],
    run_identity: Mapping[str, Any],
    snapshot_sha256: str,
    manifest: Mapping[str, Any],
) -> None:
    if (
        manifest.get("task_id") != TASK_ID
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("analysis_sha256") != _self_hash(manifest)
        or manifest.get("status") != STATUS
        or manifest.get("scientific_verdict") is not None
        or manifest.get("qualified_claim") is not None
        or manifest.get("claim_state") != CLAIM_BOUNDARY
        or manifest.get("run_id") != run_identity["run_id"]
        or manifest.get("observed_cells") != 30
        or manifest.get("observed_rows") != 21168
        or manifest.get("primary_density_count") != 2160
        or manifest.get("rao_blackwell_sidecar_density_count") != 1296
        or manifest.get("exception_rows") != 0
        or manifest.get("conservation_failure_rows") != 0
        or manifest.get("input_snapshot_analysis_sha256") != snapshot_sha256
        or manifest.get("writer_source_sha256") != _SOURCE_AT_IMPORT
        or manifest.get("old_raw_or_gate_composition") is not False
        or manifest.get("powered_formal_released") is not False
    ):
        raise RuntimeError("repair manifest identity/claim drift")
    receipts = manifest.get("chunk_receipts")
    bindings = manifest.get("receipt_bindings")
    if (
        not isinstance(receipts, list)
        or not isinstance(bindings, list)
        or len(receipts) != len(cells)
        or len(bindings) != len(cells)
    ):
        raise RuntimeError("repair manifest receipt denominator drift")
    for cell, receipt, binding in zip(cells, receipts, bindings, strict=True):
        _validate_receipt(
            root,
            config,
            execution,
            cell,
            receipt,
            run_identity,
            snapshot_sha256,
        )
        _, payload = _read_bound_bytes(root, binding)
        if json.loads(payload) != receipt:
            raise RuntimeError("manifest receipt binding/content drift")
    for name in ("resource_preflight", "capability_preflight"):
        binding = manifest["bindings"][name]
        _read_bound_bytes(root, binding)


def run_repair(
    root: Path | None = None,
    *,
    preflight_only: bool = False,
) -> dict[str, Any]:
    root = (root or _root()).resolve()
    config, base = load_config(root)
    _activate_verified_modules(root, config)
    snapshot = _input_snapshot(root, config)
    _assert_snapshot(root, config, snapshot["analysis_sha256"])
    execution = materialize_execution(config, base)
    cells = build_cells(config, execution)
    with _owner_lock(root, config):
        identity = _load_or_create_run_identity(root, config, snapshot)
        manifest_path = root / str(config["artifact_paths"]["execution_manifest"])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_bytes())
            _verify_manifest(
                root,
                config,
                execution,
                cells,
                identity,
                snapshot["analysis_sha256"],
                manifest,
            )
            return manifest
        _heartbeat(
            root,
            config,
            identity,
            state="PREFLIGHT",
            completed=0,
            total=len(cells),
        )
        try:
            resource = _run_resource_preflight(root, config, base)
            capability = _run_capability_preflight(root, config, base)
            if preflight_only:
                report = {
                    "task_id": TASK_ID,
                    "preflight_only": True,
                    "resource_preflight": resource,
                    "capability_preflight": capability,
                    "claim_state": dict(CLAIM_BOUNDARY),
                }
                report["analysis_sha256"] = _self_hash(report)
                _heartbeat(
                    root,
                    config,
                    identity,
                    state="PREFLIGHT_COMPLETE",
                    completed=0,
                    total=len(cells),
                )
                return report
            _heartbeat(
                root,
                config,
                identity,
                state="RUNNING",
                completed=0,
                total=len(cells),
            )
            receipts: list[dict[str, Any]] = []
            with ThreadPoolExecutor(
                max_workers=int(config["max_workers"]),
                thread_name_prefix="cutoff36_44",
            ) as executor:
                futures = {
                    executor.submit(
                        _execute_and_commit_cell,
                        root,
                        config,
                        execution,
                        cell,
                        identity,
                        snapshot["analysis_sha256"],
                    ): cell
                    for cell in cells
                }
                for future in as_completed(futures):
                    receipts.append(future.result())
                    _assert_snapshot(
                        root,
                        config,
                        snapshot["analysis_sha256"],
                    )
                    _heartbeat(
                        root,
                        config,
                        identity,
                        state="RUNNING",
                        completed=len(receipts),
                        total=len(cells),
                    )
            by_id = {receipt["cell"]["chunk_id"]: receipt for receipt in receipts}
            if (
                len(receipts) != 30
                or len(by_id) != 30
                or set(by_id) != {cell.chunk_id for cell in cells}
            ):
                raise RuntimeError("repair receipt cell-set drift")
            ordered = [by_id[cell.chunk_id] for cell in cells]
            receipt_bindings = [
                _binding(_receipt_path(root, config, cell), root) for cell in cells
            ]
            manifest: dict[str, Any] = {
                "task_id": TASK_ID,
                "schema_version": MANIFEST_SCHEMA,
                "status": STATUS,
                "scientific_verdict": None,
                "qualified_claim": None,
                "run_id": identity["run_id"],
                "run_identity_analysis_sha256": identity["analysis_sha256"],
                "config_analysis_sha256": config["analysis_sha256"],
                "input_snapshot_analysis_sha256": snapshot["analysis_sha256"],
                "writer_source_sha256": _SOURCE_AT_IMPORT,
                "observed_cells": 30,
                "observed_rows": 21168,
                "primary_density_count": 2160,
                "rao_blackwell_sidecar_density_count": 1296,
                "exception_rows": 0,
                "conservation_failure_rows": 0,
                "old_raw_or_gate_composition": False,
                "powered_formal_released": False,
                "chunk_receipts": ordered,
                "receipt_bindings": receipt_bindings,
                "capability_preflight": capability,
                "resource_preflight": resource,
                "bindings": {
                    "config": _binding(root / CONFIG_PATH, root),
                    "writer": _binding(Path(__file__).resolve(), root),
                    "run_identity": _binding(
                        root / str(config["artifact_paths"]["run_identity"]),
                        root,
                    ),
                    "resource_preflight": _binding(
                        root / str(config["artifact_paths"]["resource_preflight"]),
                        root,
                    ),
                    "capability_preflight": _binding(
                        root / str(config["artifact_paths"]["capability_preflight"]),
                        root,
                    ),
                    "diagnosis": dict(config["repair_trigger"]["diagnosis"]),
                    "independent_no_go_verification": dict(
                        config["repair_trigger"]["independent_no_go_verification"]
                    ),
                    **{
                        f"source/{name}": dict(binding)
                        for name, binding in config["source_bindings"].items()
                    },
                },
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "platform": platform.platform(),
                },
                "claim_state": dict(CLAIM_BOUNDARY),
            }
            manifest["analysis_sha256"] = _self_hash(manifest)
            _verify_manifest(
                root,
                config,
                execution,
                cells,
                identity,
                snapshot["analysis_sha256"],
                manifest,
            )
            _atomic_json(manifest_path, manifest)
            live = json.loads(manifest_path.read_bytes())
            _verify_manifest(
                root,
                config,
                execution,
                cells,
                identity,
                snapshot["analysis_sha256"],
                live,
            )
            _heartbeat(
                root,
                config,
                identity,
                state="COMPLETE",
                completed=30,
                total=30,
                manifest=live,
            )
            return live
        except BaseException as exc:
            manifest_path.unlink(missing_ok=True)
            _heartbeat(
                root,
                config,
                identity,
                state="FAILED",
                completed=0,
                total=len(cells),
                error=exc,
            )
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded fresh cutoff-36/40/44 repair. "
            "No scientific verdict is emitted by this raw writer."
        )
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run and publish only resource/capability preflights.",
    )
    args = parser.parse_args(argv)
    result = run_repair(preflight_only=args.preflight_only)
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": result.get("status", "PREFLIGHT_COMPLETE"),
                "analysis_sha256": result["analysis_sha256"],
                "scientific_verdict": result.get("scientific_verdict"),
                "qualified_claim": result.get("qualified_claim"),
                "claim_state": result["claim_state"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
