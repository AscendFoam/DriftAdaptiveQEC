"""Resumable state-conditioned high-cutoff design pilot.

The pilot reuses the already validated physics execution kernel but has a new,
disjoint seed namespace and a deliberately small denominator.  It produces
raw chunks only for designing the subsequent formal matrix.  It cannot emit
a twin-qualification verdict or release any blocked downstream task.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np
import scipy

from cnn_fpga.benchmark import phase9_fresh_twin_qualification as runner


TASK_ID = "T-RISK-20260727-01"
CONFIG_PATH = "configs/phase9/t_risk_20260727_01_high_cutoff_design_pilot_fresh2.json"
CONFIG_SCHEMA = "PHASE9-HIGH-CUTOFF-STATE-DESIGN-PILOT-CONFIG-V2"
MANIFEST_SCHEMA = "PHASE9-HIGH-CUTOFF-STATE-DESIGN-PILOT-MANIFEST-V2"
RECEIPT_SCHEMA = "PHASE9-HIGH-CUTOFF-PILOT-CHUNK-RECEIPT-V2"
RUN_IDENTITY_SCHEMA = "PHASE9-HIGH-CUTOFF-PILOT-RUN-IDENTITY-V1"
LOCK_SCHEMA = "PHASE9-HIGH-CUTOFF-PILOT-OWNER-LOCK-V1"
HARDENED_CONFIRMATION_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-CONFIRMATION-V2"
STATUS = "DESIGN_PILOT_RAW_EVIDENCE_COMPLETE"
REJECTED_STATUS = "DESIGN_PILOT_RAW_EVIDENCE_REJECTED"
CLAIM_BOUNDARY = {
    "design_pilot_only": True,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}
HARDENED_CONFIRMATION_CLAIM_BOUNDARY = {
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


def _self_hash(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    analysis = unsigned.pop("analysis_sha256", None)
    if not isinstance(analysis, str) or analysis != _sha(unsigned):
        raise RuntimeError("artifact self-hash drift")
    return analysis


def _require_binding(
    root: Path, binding: Mapping[str, Any], *, expected_path: str | None = None
) -> Path:
    if set(binding) != {"path", "bytes", "sha256"}:
        raise RuntimeError("artifact binding schema drift")
    relative = str(binding["path"])
    if expected_path is not None and relative != expected_path:
        raise RuntimeError("artifact binding path drift")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("artifact binding escapes repository root") from exc
    if dict(binding) != _binding(path, root):
        raise RuntimeError(f"artifact byte binding drift: {relative}")
    return path


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


def _validate_hardened_confirmation(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    contract = config.get("hardened_confirmation_source")
    expected_keys = {
        "report",
        "source_data",
        "required_analysis_sha256",
        "required_verdict",
        "required_selected_formal_clusters_per_state",
        "required_pilot_domain_factor_coverage_calibrated",
        "required_formal_domain_factor_coverage_calibrated",
        "required_confirmation_power_passed",
    }
    if not isinstance(contract, Mapping) or set(contract) != expected_keys:
        raise RuntimeError("hardened confirmation release contract drift")
    required_analysis = contract.get("required_analysis_sha256")
    if not isinstance(required_analysis, str) or len(required_analysis) != 64:
        raise RuntimeError("hardened confirmation is pending and unreleased")
    report_binding = contract.get("report")
    source_binding = contract.get("source_data")
    if not isinstance(report_binding, Mapping) or not isinstance(
        source_binding, Mapping
    ):
        raise RuntimeError("hardened confirmation artifact binding missing")
    report_path = _require_binding(
        root,
        report_binding,
        expected_path=str(report_binding.get("path")),
    )
    source_path = _require_binding(
        root,
        source_binding,
        expected_path=str(source_binding.get("path")),
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _self_hash(report)
    if (
        report.get("task_id") != TASK_ID
        or report.get("schema_version") != HARDENED_CONFIRMATION_SCHEMA
        or report.get("analysis_sha256") != required_analysis
        or report.get("verdict") != contract["required_verdict"]
        or report.get("qualified_claim") is not None
        or report.get("selected_formal_clusters_per_state")
        != contract["required_selected_formal_clusters_per_state"]
        or report.get("pilot_domain_factor_coverage_calibrated")
        is not contract["required_pilot_domain_factor_coverage_calibrated"]
        or report.get("formal_domain_factor_coverage_calibrated")
        is not contract["required_formal_domain_factor_coverage_calibrated"]
        or report.get("confirmation_power_passed")
        is not contract["required_confirmation_power_passed"]
        or report.get("claim_state") != HARDENED_CONFIRMATION_CLAIM_BOUNDARY
        or report.get("formal_outcomes_accessed") is not False
        or report.get("domain", {}).get("multiplier_replicates")
        != config["diagnostic_contract"]["multiplier_replicates"]
    ):
        raise RuntimeError("hardened confirmation release semantics drift")
    report_source_binding = report.get("bindings", {}).get("source_data")
    if (
        not isinstance(report_source_binding, Mapping)
        or dict(report_source_binding) != dict(source_binding)
        or report_source_binding.get("sha256")
        != sha256(source_path.read_bytes()).hexdigest()
    ):
        raise RuntimeError("hardened confirmation source-data binding drift")
    for name, binding in report.get("bindings", {}).items():
        if not isinstance(binding, Mapping):
            raise RuntimeError(
                f"hardened confirmation report binding type drift: {name}"
            )
        path = root / str(binding.get("path"))
        live = _binding(path, root)
        if any(binding.get(key) != live[key] for key in live):
            raise RuntimeError(f"hardened confirmation live binding drift: {name}")
    chunk_bindings = report.get("chunk_bindings")
    if not isinstance(chunk_bindings, list) or len(chunk_bindings) != 96:
        raise RuntimeError("hardened confirmation chunk binding count drift")
    for index, binding in enumerate(chunk_bindings):
        if not isinstance(binding, Mapping):
            raise RuntimeError("hardened confirmation chunk binding type drift")
        path = root / str(binding.get("path"))
        live = _binding(path, root)
        if dict(binding) != live:
            raise RuntimeError(f"hardened confirmation chunk binding drift: {index}")
    return report


def load_pilot_config(
    root: Path, *, require_hardened: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("high-cutoff pilot identity/claim firewall invalid")
    if (
        config.get("cutoffs") != [16, 20, 24, 28]
        or int(config.get("trajectory_count", 0))
        != 6 * int(config.get("clusters_per_state", -1))
        or sorted(config.get("scenario_names", []))
        != ["burst", "compound", "step", "telegraph"]
    ):
        raise ValueError("high-cutoff pilot matrix drift")
    diagnostic = config.get("diagnostic_contract", {})
    if (
        diagnostic.get("confidence") != 0.95
        or diagnostic.get("multiplier_replicates") != 199
        or diagnostic.get("multiplier_seed_namespace") != 1420000
        or diagnostic.get("formal_rescue_forbidden") is not True
    ):
        raise ValueError("high-cutoff pilot diagnostic contract drift")
    for scenario in config["scenario_names"]:
        partition = config.get("stage_partition", {}).get(scenario)
        if (
            not isinstance(partition, Mapping)
            or sorted(
                round_index for indices in partition.values() for round_index in indices
            )
            != list(range(12))
            or sum(len(indices) for indices in partition.values()) != 12
        ):
            raise ValueError(f"high-cutoff stage partition drift: {scenario}")
    splits = config["seed_splits"]
    intervals = []
    for key in ("trajectory_backend_a", "trajectory_backend_b", "heldout_common"):
        start = int(splits[key]["start"])
        count = int(splits[key]["count"])
        intervals.append(set(range(start, start + count)))
    if (
        splits.get("all_intervals_disjoint") is not True
        or splits.get("disjoint_from_20260726_formal") is not True
        or any(intervals[i] & intervals[j] for i in range(3) for j in range(i))
        or min(min(interval) for interval in intervals) <= 1360511
    ):
        raise ValueError("high-cutoff pilot seed firewall invalid")
    for binding in config["source_bindings"].values():
        path = root / str(binding["path"])
        if _binding(path, root)["sha256"] != binding["sha256"]:
            raise ValueError(f"pilot source binding drift: {binding['path']}")
    base_binding = config["base_config"]
    base_path = root / str(base_binding["path"])
    if _binding(base_path, root)["sha256"] != base_binding["sha256"]:
        raise ValueError("pilot base config binding drift")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if require_hardened:
        _validate_hardened_confirmation(root, config)
    return config, base


def materialize_execution_config(
    pilot: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    execution = json.loads(json.dumps(base))
    count = int(pilot["trajectory_count"])
    execution["formal_matrix"]["trajectory_sample_count"] = count
    execution["formal_matrix"]["cutoff_ladder"] = list(pilot["cutoffs"])
    execution["formal_splits"]["trajectory_backend_a"] = dict(
        pilot["seed_splits"]["trajectory_backend_a"]
    )
    execution["formal_splits"]["trajectory_backend_b"] = dict(
        pilot["seed_splits"]["trajectory_backend_b"]
    )
    execution["formal_splits"]["heldout_common"] = dict(
        pilot["seed_splits"]["heldout_common"]
    )
    execution["artifact_paths"]["chunk_directory"] = pilot["artifact_paths"][
        "chunk_directory"
    ]
    return execution


def build_pilot_cells(
    pilot: Mapping[str, Any], execution: Mapping[str, Any]
) -> list[runner.CellSpec]:
    cells: list[runner.CellSpec] = []
    count = int(pilot["trajectory_count"])
    for cutoff in pilot["cutoffs"]:
        for scenario in pilot["scenario_names"]:
            horizon = int(
                execution["formal_matrix"]["fault_scenarios"][scenario]["horizon"]
            )
            for backend in ("A", "B"):
                identity = f"pilot|c{cutoff}|fault|{scenario}|{backend}"
                chunk_id = (
                    "".join(
                        character if character.isalnum() else "_"
                        for character in identity
                    )
                    + "__"
                    + sha256(identity.encode("utf-8")).hexdigest()[:16]
                )
                cells.append(
                    runner.CellSpec(
                        chunk_id=chunk_id,
                        layer="fault",
                        cell_base=f"fault|{scenario}",
                        cutoff=int(cutoff),
                        backend=backend,
                        sample_count=count,
                        convergence_role="high_cutoff_state_design_pilot",
                        scenario=scenario,
                        horizon=horizon,
                    )
                )
    if (
        len(cells) != 32
        or len({cell.chunk_id for cell in cells}) != 32
        or sum(cell.expected_rows for cell in cells) != 32 * count * 12
    ):
        raise RuntimeError("high-cutoff pilot accounting drift")
    return cells


def _receipt_path(root: Path, pilot: Mapping[str, Any], cell: runner.CellSpec) -> Path:
    return (
        root
        / str(pilot["artifact_paths"]["receipt_directory"])
        / f"{cell.chunk_id}.json"
    )


def _validate_receipt(
    root: Path,
    pilot: Mapping[str, Any],
    cell: runner.CellSpec,
    receipt: Mapping[str, Any],
    *,
    run_identity: Mapping[str, Any],
    execution_analysis_sha256: str,
) -> None:
    unsigned = dict(receipt)
    analysis = unsigned.pop("analysis_sha256", None)
    expected_keys = {
        "task_id",
        "schema_version",
        "run_id",
        "run_identity_analysis_sha256",
        "config_analysis_sha256",
        "execution_analysis_sha256",
        "pilot_source_sha256",
        "cell",
        "chunk_id",
        "cell_base",
        "layer",
        "backend",
        "cutoff",
        "expected_rows",
        "observed_rows",
        "exception_rows",
        "csv",
        "npz",
        "analysis_sha256",
    }
    if (
        set(receipt) != expected_keys
        or receipt.get("task_id") != TASK_ID
        or receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("run_id") != run_identity["run_id"]
        or receipt.get("run_identity_analysis_sha256")
        != run_identity["analysis_sha256"]
        or receipt.get("config_analysis_sha256") != _sha(pilot)
        or receipt.get("execution_analysis_sha256") != execution_analysis_sha256
        or receipt.get("pilot_source_sha256")
        != sha256(Path(__file__).read_bytes()).hexdigest()
        or receipt.get("cell") != asdict(cell)
        or analysis != _sha(unsigned)
    ):
        raise RuntimeError("pilot receipt identity drift")
    runner._validate_chunk_files(root, receipt, cell)
    for key in ("csv", "npz"):
        binding = receipt.get(key)
        if not isinstance(binding, Mapping) or dict(binding) != _binding(
            root / str(binding.get("path")), root
        ):
            raise RuntimeError(f"pilot {key} binding drift")


def _validate_receipt_file(
    root: Path,
    pilot: Mapping[str, Any],
    cell: runner.CellSpec,
    receipt: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
    *,
    run_identity: Mapping[str, Any],
    execution_analysis_sha256: str,
) -> None:
    receipt_path = _receipt_path(root, pilot, cell)
    _require_binding(
        root,
        receipt_binding,
        expected_path=receipt_path.resolve().relative_to(root.resolve()).as_posix(),
    )
    live = json.loads(receipt_path.read_text(encoding="utf-8"))
    if live != receipt:
        raise RuntimeError("manifest/live receipt content drift")
    _validate_receipt(
        root,
        pilot,
        cell,
        live,
        run_identity=run_identity,
        execution_analysis_sha256=execution_analysis_sha256,
    )


def _worker(
    root_text: str,
    pilot: Mapping[str, Any],
    execution: Mapping[str, Any],
    cell_payload: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    execution_analysis_sha256: str,
) -> dict[str, Any]:
    root = Path(root_text).resolve()
    cell = runner.CellSpec(**cell_payload)
    receipt_path = _receipt_path(root, pilot, cell)
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _validate_receipt(
            root,
            pilot,
            cell,
            receipt,
            run_identity=run_identity,
            execution_analysis_sha256=execution_analysis_sha256,
        )
        return receipt
    simulator = runner.build_simulators(execution, cell.cutoff)[cell.backend]
    evidence = runner.execute_cell(execution, cell, simulator, runner._action_words())
    chunk = runner.write_chunk(root, execution, cell, evidence)
    receipt = {
        "task_id": TASK_ID,
        "schema_version": RECEIPT_SCHEMA,
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "config_analysis_sha256": _sha(pilot),
        "execution_analysis_sha256": execution_analysis_sha256,
        "pilot_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "cell": asdict(cell),
        **chunk,
    }
    receipt["analysis_sha256"] = _sha(receipt)
    _atomic_text(
        receipt_path,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _validate_receipt(
        root,
        pilot,
        cell,
        receipt,
        run_identity=run_identity,
        execution_analysis_sha256=execution_analysis_sha256,
    )
    return receipt


@contextmanager
def _exclusive_owner_lock(root: Path, pilot: Mapping[str, Any]) -> Any:
    lock_path = root / str(pilot["artifact_paths"]["owner_lock"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_token = uuid4().hex
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": LOCK_SCHEMA,
        "owner_token": owner_token,
        "pid": os.getpid(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_analysis_sha256": _sha(pilot),
    }
    payload["analysis_sha256"] = _sha(payload)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "high-cutoff pilot owner lock already exists; concurrent or "
            "unclean prior supervisor must be resolved explicitly"
        ) from exc
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        yield payload
    finally:
        if lock_path.exists():
            live = json.loads(lock_path.read_text(encoding="utf-8"))
            if (
                live.get("owner_token") != owner_token
                or _self_hash(live) != payload["analysis_sha256"]
            ):
                raise RuntimeError("owner lock changed while supervisor was active")
            lock_path.unlink()


def _load_or_create_run_identity(
    root: Path,
    pilot: Mapping[str, Any],
    execution_analysis_sha256: str,
) -> dict[str, Any]:
    path = root / str(pilot["artifact_paths"]["run_identity"])
    if path.exists():
        identity = json.loads(path.read_text(encoding="utf-8"))
        _self_hash(identity)
    else:
        identity = {
            "task_id": TASK_ID,
            "schema_version": RUN_IDENTITY_SCHEMA,
            "run_id": str(uuid4()),
            "config_analysis_sha256": _sha(pilot),
            "execution_analysis_sha256": execution_analysis_sha256,
            "pilot_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        identity["analysis_sha256"] = _sha(identity)
        _atomic_text(
            path,
            json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    try:
        UUID(str(identity.get("run_id")))
    except ValueError as exc:
        raise RuntimeError("run identity UUID drift") from exc
    if (
        identity.get("task_id") != TASK_ID
        or identity.get("schema_version") != RUN_IDENTITY_SCHEMA
        or identity.get("config_analysis_sha256") != _sha(pilot)
        or identity.get("execution_analysis_sha256") != execution_analysis_sha256
        or identity.get("pilot_source_sha256")
        != sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise RuntimeError("run identity binding drift")
    return identity


def _chunk_health(root: Path, receipt: Mapping[str, Any]) -> tuple[int, int]:
    exception_rows = 0
    conservation_failures = 0
    with (root / str(receipt["csv"]["path"])).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            exception = bool(row["exception_type"])
            exception_rows += int(exception)
            conservation = row["conservation_pass"]
            if conservation not in {"True", "False"}:
                raise RuntimeError("conservation_pass is not a strict boolean")
            conservation_failures += int(not exception and conservation != "True")
    return exception_rows, conservation_failures


def _heartbeat(
    root: Path,
    pilot: Mapping[str, Any],
    *,
    completed: int,
    total: int,
    active: bool,
    state: str,
    error_type: str | None = None,
) -> None:
    if state not in {"RUNNING", "COMPLETE", "REJECTED", "FAILED"}:
        raise ValueError("pilot heartbeat state drift")
    payload = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-HIGH-CUTOFF-PILOT-HEARTBEAT-V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "completed_cells": completed,
        "expected_cells": total,
        "active": active,
        "state": state,
        "error_type": error_type,
    }
    payload["analysis_sha256"] = _sha(payload)
    _atomic_text(
        root / str(pilot["artifact_paths"]["heartbeat"]),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _verify_manifest(
    root: Path,
    pilot: Mapping[str, Any],
    execution: Mapping[str, Any],
    cells: Sequence[runner.CellSpec],
    run_identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    _self_hash(manifest)
    expected_keys = {
        "task_id",
        "schema_version",
        "status",
        "scientific_verdict",
        "qualified_claim",
        "run_id",
        "run_identity_analysis_sha256",
        "config_analysis_sha256",
        "execution_analysis_sha256",
        "pilot_source_sha256",
        "observed_cells",
        "observed_rows",
        "exception_rows",
        "conservation_failure_rows",
        "chunk_receipts",
        "receipt_bindings",
        "claim_state",
        "bindings",
        "runtime",
        "analysis_sha256",
    }
    execution_analysis_sha256 = _sha(execution)
    if (
        set(manifest) != expected_keys
        or manifest.get("task_id") != TASK_ID
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != STATUS
        or manifest.get("scientific_verdict") is not None
        or manifest.get("qualified_claim") is not None
        or manifest.get("run_id") != run_identity["run_id"]
        or manifest.get("run_identity_analysis_sha256")
        != run_identity["analysis_sha256"]
        or manifest.get("config_analysis_sha256") != _sha(pilot)
        or manifest.get("execution_analysis_sha256") != execution_analysis_sha256
        or manifest.get("pilot_source_sha256")
        != sha256(Path(__file__).read_bytes()).hexdigest()
        or manifest.get("observed_cells") != len(cells)
        or manifest.get("observed_rows") != sum(cell.expected_rows for cell in cells)
        or manifest.get("exception_rows") != 0
        or manifest.get("conservation_failure_rows") != 0
        or manifest.get("claim_state") != CLAIM_BOUNDARY
    ):
        raise RuntimeError("existing pilot manifest semantic drift")
    receipts = manifest.get("chunk_receipts")
    receipt_bindings = manifest.get("receipt_bindings")
    if (
        not isinstance(receipts, list)
        or not isinstance(receipt_bindings, list)
        or len(receipts) != len(cells)
        or len(receipt_bindings) != len(cells)
    ):
        raise RuntimeError("existing pilot manifest receipt accounting drift")
    receipt_ids = [
        receipt.get("cell", {}).get("chunk_id")
        for receipt in receipts
        if isinstance(receipt, Mapping)
    ]
    expected_ids = [cell.chunk_id for cell in cells]
    if receipt_ids != expected_ids or len(set(receipt_ids)) != len(receipt_ids):
        raise RuntimeError("existing pilot manifest ordered cell set drift")
    exceptions = 0
    conservation_failures = 0
    for cell, receipt, receipt_binding in zip(cells, receipts, receipt_bindings):
        if not isinstance(receipt, Mapping) or not isinstance(receipt_binding, Mapping):
            raise RuntimeError("existing pilot manifest receipt type drift")
        _validate_receipt_file(
            root,
            pilot,
            cell,
            receipt,
            receipt_binding,
            run_identity=run_identity,
            execution_analysis_sha256=execution_analysis_sha256,
        )
        chunk_exceptions, chunk_conservation_failures = _chunk_health(root, receipt)
        exceptions += chunk_exceptions
        conservation_failures += chunk_conservation_failures
    if exceptions != 0 or conservation_failures != 0:
        raise RuntimeError("existing pilot chunks fail health revalidation")
    expected_bindings = {
        "config": _binding(root / CONFIG_PATH, root),
        "base_config": _binding(root / str(pilot["base_config"]["path"]), root),
        "pilot_source": _binding(Path(__file__).resolve(), root),
        "run_identity": _binding(
            root / str(pilot["artifact_paths"]["run_identity"]), root
        ),
        "hardened_confirmation_report": _binding(
            root / str(pilot["hardened_confirmation_source"]["report"]["path"]),
            root,
        ),
        "hardened_confirmation_source_data": _binding(
            root / str(pilot["hardened_confirmation_source"]["source_data"]["path"]),
            root,
        ),
        **{
            name: _binding(root / str(binding["path"]), root)
            for name, binding in pilot["source_bindings"].items()
        },
    }
    if manifest.get("bindings") != expected_bindings:
        raise RuntimeError("existing pilot manifest live binding drift")


def run_pilot(root: Path) -> dict[str, Any]:
    root = root.resolve()
    pilot, base = load_pilot_config(root, require_hardened=True)
    execution = materialize_execution_config(pilot, base)
    execution_analysis_sha256 = _sha(execution)
    cells = build_pilot_cells(pilot, execution)
    with _exclusive_owner_lock(root, pilot):
        run_identity = _load_or_create_run_identity(
            root, pilot, execution_analysis_sha256
        )
        manifest_path = root / str(pilot["artifact_paths"]["execution_manifest"])
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _verify_manifest(
                root,
                pilot,
                execution,
                cells,
                run_identity,
                manifest,
            )
            return manifest

        _heartbeat(
            root,
            pilot,
            completed=0,
            total=len(cells),
            active=True,
            state="RUNNING",
        )
        receipts: list[dict[str, Any]] = []
        try:
            with ProcessPoolExecutor(max_workers=int(pilot["max_workers"])) as executor:
                futures = {
                    executor.submit(
                        _worker,
                        str(root),
                        pilot,
                        execution,
                        asdict(cell),
                        run_identity,
                        execution_analysis_sha256,
                    ): cell
                    for cell in cells
                }
                for future in as_completed(futures):
                    receipts.append(future.result())
                    _heartbeat(
                        root,
                        pilot,
                        completed=len(receipts),
                        total=len(cells),
                        active=True,
                        state="RUNNING",
                    )
            by_id = {receipt["cell"]["chunk_id"]: receipt for receipt in receipts}
            if (
                len(receipts) != len(cells)
                or len(by_id) != len(receipts)
                or set(by_id) != {cell.chunk_id for cell in cells}
            ):
                raise RuntimeError("pilot receipt cell set drift")
            ordered = [by_id[cell.chunk_id] for cell in cells]
            receipt_bindings: list[dict[str, Any]] = []
            exception_rows = 0
            conservation_failure_rows = 0
            for cell, receipt in zip(cells, ordered):
                receipt_binding = _binding(_receipt_path(root, pilot, cell), root)
                _validate_receipt_file(
                    root,
                    pilot,
                    cell,
                    receipt,
                    receipt_binding,
                    run_identity=run_identity,
                    execution_analysis_sha256=execution_analysis_sha256,
                )
                receipt_bindings.append(receipt_binding)
                exceptions, conservation_failures = _chunk_health(root, receipt)
                exception_rows += exceptions
                conservation_failure_rows += conservation_failures
            healthy = exception_rows == 0 and conservation_failure_rows == 0
            manifest: dict[str, Any] = {
                "task_id": TASK_ID,
                "schema_version": MANIFEST_SCHEMA,
                "status": STATUS if healthy else REJECTED_STATUS,
                "scientific_verdict": None,
                "qualified_claim": None,
                "run_id": run_identity["run_id"],
                "run_identity_analysis_sha256": run_identity["analysis_sha256"],
                "config_analysis_sha256": _sha(pilot),
                "execution_analysis_sha256": execution_analysis_sha256,
                "pilot_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "observed_cells": len(ordered),
                "observed_rows": sum(cell.expected_rows for cell in cells),
                "exception_rows": exception_rows,
                "conservation_failure_rows": conservation_failure_rows,
                "chunk_receipts": ordered,
                "receipt_bindings": receipt_bindings,
                "claim_state": dict(pilot["claim_boundary"]),
                "bindings": {
                    "config": _binding(root / CONFIG_PATH, root),
                    "base_config": _binding(
                        root / str(pilot["base_config"]["path"]), root
                    ),
                    "pilot_source": _binding(Path(__file__).resolve(), root),
                    "run_identity": _binding(
                        root / str(pilot["artifact_paths"]["run_identity"]),
                        root,
                    ),
                    "hardened_confirmation_report": _binding(
                        root
                        / str(pilot["hardened_confirmation_source"]["report"]["path"]),
                        root,
                    ),
                    "hardened_confirmation_source_data": _binding(
                        root
                        / str(
                            pilot["hardened_confirmation_source"]["source_data"]["path"]
                        ),
                        root,
                    ),
                    **{
                        name: _binding(root / str(binding["path"]), root)
                        for name, binding in pilot["source_bindings"].items()
                    },
                },
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "platform": platform.platform(),
                },
            }
            manifest["analysis_sha256"] = _sha(manifest)
            if not healthy:
                _atomic_text(
                    manifest_path,
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                )
                _heartbeat(
                    root,
                    pilot,
                    completed=len(cells),
                    total=len(cells),
                    active=False,
                    state="REJECTED",
                )
            else:
                # Validate the complete in-memory document before publication,
                # then re-read and validate the exact bytes that became
                # visible.  A COMPLETE heartbeat is the final commit marker
                # and is never emitted before both validations succeed.
                _verify_manifest(
                    root,
                    pilot,
                    execution,
                    cells,
                    run_identity,
                    manifest,
                )
                _atomic_text(
                    manifest_path,
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                )
                live_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                _verify_manifest(
                    root,
                    pilot,
                    execution,
                    cells,
                    run_identity,
                    live_manifest,
                )
                _heartbeat(
                    root,
                    pilot,
                    completed=len(cells),
                    total=len(cells),
                    active=False,
                    state="COMPLETE",
                )
                return live_manifest
        except BaseException as exc:
            # The manifest is a commit artifact.  If any worker/finalization
            # step fails, remove a potentially published-but-unverified copy
            # before recording the terminal FAILED state.
            manifest_path.unlink(missing_ok=True)
            _heartbeat(
                root,
                pilot,
                completed=len(receipts),
                total=len(cells),
                active=False,
                state="FAILED",
                error_type=type(exc).__name__,
            )
            raise
        raise RuntimeError("pilot evidence rejected: exception or conservation failure")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the state-conditioned high-cutoff design pilot."
    )
    parser.parse_args(argv)
    report = run_pilot(_root())
    print(
        json.dumps(
            {
                "status": report["status"],
                "analysis_sha256": report["analysis_sha256"],
                "observed_cells": report["observed_cells"],
                "observed_rows": report["observed_rows"],
                "exception_rows": report["exception_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "STATUS",
    "build_pilot_cells",
    "load_pilot_config",
    "materialize_execution_config",
    "run_pilot",
]
