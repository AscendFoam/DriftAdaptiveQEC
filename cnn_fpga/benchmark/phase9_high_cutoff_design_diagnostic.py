"""Independent state/stage diagnostic for the high-cutoff design pilot."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.abc
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np


TASK_ID = "T-RISK-20260727-01"
EXTERNAL_LAUNCHER_SHA256 = (
    "e732eb9fec98ed5955f1864feeffde33a364fc79b011ac4b83ae48c872e97be6"
)
LAUNCH_META_SCHEMA = "PHASE9-TRUSTED-OPERATOR-LAUNCH-META-V2"
LAUNCHER_ASSURANCE = {
    "scope": "trusted_operator_preregistered_command_and_accidental_drift",
    "trusted_operator_required": True,
    "preexecution_arbitrary_code": "OUT_OF_SCOPE",
    "adversarial_local_operator_resistance": None,
    "cryptographic_process_origin_attestation": None,
    "os_native_signed_launcher_receipt": None,
}
CONFIG_PATH = (
    "configs/phase9/" "t_risk_20260727_01_high_cutoff_design_pilot_fresh3_released.json"
)
UQ_REPORT_PATH = "docs/t_risk_20260727_01_uq_calibration.json"
UQ_EXTENSION_PATH = "docs/t_risk_20260727_01_uq_power_extension.json"
REPORT_PATH = "docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3.json"
SOURCE_PATH = (
    "docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3_source_data.csv"
)
SCHEMA = "PHASE9-HIGH-CUTOFF-STATE-DESIGN-DIAGNOSTIC-V5"
STATUS = "HIGH_CUTOFF_STATE_STAGE_DESIGN_DIAGNOSTIC_COMPLETE"
DIAGNOSTIC_LOCK_SCHEMA = "PHASE9-HIGH-CUTOFF-DIAGNOSTIC-OWNER-LOCK-V1"
DIAGNOSTIC_COMPLETION_SCHEMA = "PHASE9-HIGH-CUTOFF-DIAGNOSTIC-COMPLETION-RECEIPT-V1"
DIAGNOSTIC_LOCK_PATH = (
    "runs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3/" "diagnostic.owner.lock"
)
COMPLETION_PATH = (
    "docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3_completion.json"
)
RISK_VERDICT = "EXPLORATORY_RISK_SIGNAL"
INCONCLUSIVE_VERDICT = "NO_LARGE_SIGNAL_INCONCLUSIVE"
INCOMPLETE_VERDICT = "INCOMPLETE"
VERIFIED_LOADER_CONTRACT = "PHASE9-VERIFIED-SOURCE-BYTES-LOADER-V1"
UQ_CLAIM_BOUNDARY = {
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
EXTENSION_CLAIM_BOUNDARY = {
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
_DIAGNOSTIC_SOURCE_SHA256_AT_IMPORT = sha256(Path(__file__).read_bytes()).hexdigest()
_DIAGNOSTIC_MODULE_NAMES = {
    "pilot_source": "cnn_fpga.benchmark.phase9_high_cutoff_design_pilot",
    "paired_cluster_uq_source": "cnn_fpga.benchmark.phase9_paired_cluster_uq",
}
_PILOT_BOOTSTRAP_PATH = "cnn_fpga/benchmark/phase9_high_cutoff_design_pilot.py"
_PILOT_BOOTSTRAP_SHA256 = (
    "57e9fe8a9541aa76c577a5418423e96806b0ef883330a83628eb45e5649d01c4"
)
_RELEASED_CHILD_BYTES = 2821
_RELEASED_CHILD_SHA256 = (
    "e8e301e0ac2f718b1a51839adb8ccf8de929af5c23a73d5883f6853e60f89a61"
)
pilot_runner: Any = None
NormUCB: Any = Any
paired_density_trace_ucb: Any = None
paired_vector_norm_ucb: Any = None
_VERIFIED_DIAGNOSTIC_BINDINGS: dict[str, dict[str, object]] = {}
_VERIFIED_DIAGNOSTIC_MODULES: dict[str, object] = {}


class _DiagnosticVerifiedLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, path: Path, payload: bytes, digest: str) -> None:
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


class _DiagnosticVerifiedFinder(importlib.abc.MetaPathFinder):
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
        loader = _DiagnosticVerifiedLoader(
            fullname,
            source_path,
            payload,
            digest,
        )
        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=str(source_path),
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


def _diagnostic_source_sha256() -> str:
    live = sha256(Path(__file__).read_bytes()).hexdigest()
    if live != _DIAGNOSTIC_SOURCE_SHA256_AT_IMPORT:
        raise RuntimeError("diagnostic source changed after module import")
    return live


def _require_verified_self_import() -> None:
    module = sys.modules.get(__name__)
    bootstrap_binding = (
        None
        if module is None
        else getattr(module, "__verified_bootstrap_source_binding__", None)
    )
    launch_meta_binding = (
        None
        if module is None
        else getattr(module, "__verified_launch_meta_binding__", None)
    )
    launcher_sha256 = (
        None
        if module is None
        else getattr(module, "__verified_external_launcher_sha256__", None)
    )
    launch_meta_payload = (
        None
        if module is None
        else getattr(module, "__verified_launch_meta_payload__", None)
    )
    if (
        module is None
        or getattr(module, "__verified_source_sha256__", None)
        != _DIAGNOSTIC_SOURCE_SHA256_AT_IMPORT
        or getattr(module, "__verified_bootstrap_contract__", None)
        != VERIFIED_LOADER_CONTRACT
        or launcher_sha256 != EXTERNAL_LAUNCHER_SHA256
        or not isinstance(bootstrap_binding, Mapping)
        or set(bootstrap_binding) != {"path", "bytes", "sha256"}
        or not isinstance(launch_meta_binding, Mapping)
        or set(launch_meta_binding) != {"path", "bytes", "sha256"}
        or not isinstance(launch_meta_payload, Mapping)
        or launch_meta_payload.get("task_id") != TASK_ID
        or launch_meta_payload.get("schema_version") != LAUNCH_META_SCHEMA
        or launch_meta_payload.get("mode") != "diagnostic"
        or launch_meta_payload.get("external_launcher_sha256") != launcher_sha256
        or launch_meta_payload.get("launcher_assurance") != LAUNCHER_ASSURANCE
        or launch_meta_payload.get("isolation_flags") != ["-I", "-S"]
        or launch_meta_payload.get("bootstrap") != dict(bootstrap_binding)
        or launch_meta_payload.get("bootstrap_load_protocol")
        != "read_once_sha256_then_compile_exec"
        or launch_meta_payload.get("child_process_policy")
        != "same_verified_process_thread_workers_only"
        or launch_meta_payload.get("qualified_claim") is not None
        or launch_meta_payload.get("downstream_release") is not False
        or launch_meta_payload.get("analysis_sha256")
        != _sha(
            {
                key: value
                for key, value in launch_meta_payload.items()
                if key != "analysis_sha256"
            }
        )
    ):
        raise RuntimeError(
            "diagnostic must be imported by the preregistered trusted-operator bootstrap"
        )


def _assert_diagnostic_import_snapshot(
    root: Path,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(_VERIFIED_DIAGNOSTIC_BINDINGS) != set(_DIAGNOSTIC_MODULE_NAMES):
        raise RuntimeError("verified diagnostic module set is incomplete")
    if set(_VERIFIED_DIAGNOSTIC_MODULES) != set(_DIAGNOSTIC_MODULE_NAMES):
        raise RuntimeError("verified diagnostic module object set is incomplete")
    for name, module_name in _DIAGNOSTIC_MODULE_NAMES.items():
        imported = _VERIFIED_DIAGNOSTIC_BINDINGS[name]
        binding = snapshot.get(name)
        module = _VERIFIED_DIAGNOSTIC_MODULES[name]
        if (
            not isinstance(binding, Mapping)
            or binding.get("sha256") != imported["sha256"]
            or binding.get("path") != imported["relative_path"]
            or str((root / str(binding.get("path"))).resolve())
            != imported["absolute_path"]
            or sys.modules.get(module_name) is not module
            or getattr(module, "__verified_source_sha256__", None) != imported["sha256"]
        ):
            raise RuntimeError(f"verified diagnostic module attestation drift: {name}")


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _verify_self_hash(payload: Mapping[str, Any], label: str) -> None:
    unsigned = dict(payload)
    analysis = unsigned.pop("analysis_sha256", None)
    if not isinstance(analysis, str) or analysis != _sha(unsigned):
        raise ValueError(f"{label} self-hash drift")


def _read_exact_bootstrap_bytes(
    root: Path,
    binding: Mapping[str, Any],
    label: str,
) -> tuple[Path, bytes]:
    if set(binding) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label} bootstrap binding schema drift")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} bootstrap binding escapes root") from exc
    payload = path.read_bytes()
    if (
        len(payload) != binding["bytes"]
        or sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise RuntimeError(f"{label} bootstrap byte binding drift")
    return path, payload


def _diagnostic_frozen_sources(
    root: Path,
) -> tuple[
    dict[str, tuple[Path, bytes, str]],
    dict[str, dict[str, object]],
]:
    child_binding = {
        "path": CONFIG_PATH,
        "bytes": _RELEASED_CHILD_BYTES,
        "sha256": _RELEASED_CHILD_SHA256,
    }
    _, child_bytes = _read_exact_bootstrap_bytes(
        root,
        child_binding,
        "released child",
    )
    child = json.loads(child_bytes)
    if not isinstance(child, dict):
        raise RuntimeError("released child bootstrap JSON is not an object")
    _verify_self_hash(child, "released child bootstrap")
    pending_binding = child.get("pending_parent")
    if not isinstance(pending_binding, Mapping):
        raise RuntimeError("released child pending-parent bootstrap pin missing")
    _, pending_bytes = _read_exact_bootstrap_bytes(
        root,
        pending_binding,
        "pending parent",
    )
    pending = json.loads(pending_bytes)
    if not isinstance(pending, dict):
        raise RuntimeError("pending-parent bootstrap JSON is not an object")

    calibration_contract = pending.get("diagnostic_contract", {}).get(
        "calibration_factor_source", {}
    )
    uq_path = calibration_contract.get("path")
    required_uq_analysis = calibration_contract.get("required_analysis_sha256")
    if uq_path != UQ_REPORT_PATH or not isinstance(required_uq_analysis, str):
        raise RuntimeError("diagnostic UQ bootstrap contract drift")
    uq_bytes = (root / UQ_REPORT_PATH).read_bytes()
    uq_report = json.loads(uq_bytes)
    if not isinstance(uq_report, dict):
        raise RuntimeError("diagnostic UQ bootstrap JSON is not an object")
    _verify_self_hash(uq_report, "diagnostic UQ bootstrap")
    if uq_report.get("analysis_sha256") != required_uq_analysis:
        raise RuntimeError("diagnostic UQ bootstrap analysis drift")
    paired_binding = uq_report.get("bindings", {}).get("paired_cluster_uq_source")
    if not isinstance(paired_binding, Mapping):
        raise RuntimeError("paired-cluster UQ bootstrap pin missing")
    paired_path, paired_bytes = _read_exact_bootstrap_bytes(
        root,
        paired_binding,
        "paired-cluster UQ source",
    )

    pilot_binding = {
        "path": _PILOT_BOOTSTRAP_PATH,
        "bytes": (root / _PILOT_BOOTSTRAP_PATH).stat().st_size,
        "sha256": _PILOT_BOOTSTRAP_SHA256,
    }
    pilot_path, pilot_bytes = _read_exact_bootstrap_bytes(
        root,
        pilot_binding,
        "pilot source",
    )
    sources = {
        _DIAGNOSTIC_MODULE_NAMES["pilot_source"]: (
            pilot_path,
            pilot_bytes,
            sha256(pilot_bytes).hexdigest(),
        ),
        _DIAGNOSTIC_MODULE_NAMES["paired_cluster_uq_source"]: (
            paired_path,
            paired_bytes,
            sha256(paired_bytes).hexdigest(),
        ),
    }
    bindings = {
        "pilot_source": {
            "absolute_path": str(pilot_path),
            "relative_path": _PILOT_BOOTSTRAP_PATH,
            "sha256": sha256(pilot_bytes).hexdigest(),
        },
        "paired_cluster_uq_source": {
            "absolute_path": str(paired_path),
            "relative_path": str(paired_binding["path"]),
            "sha256": sha256(paired_bytes).hexdigest(),
        },
    }
    return sources, bindings


def _drop_preloaded_diagnostic_modules() -> None:
    for module_name in reversed(tuple(_DIAGNOSTIC_MODULE_NAMES.values())):
        module = sys.modules.pop(module_name, None)
        if module is None or "." not in module_name:
            continue
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, attribute, None) is module:
            delattr(parent, attribute)


def _activate_verified_diagnostic_modules(root: Path) -> None:
    global pilot_runner
    global NormUCB
    global paired_density_trace_ucb
    global paired_vector_norm_ucb
    global _VERIFIED_DIAGNOSTIC_BINDINGS
    global _VERIFIED_DIAGNOSTIC_MODULES

    frozen, bindings = _diagnostic_frozen_sources(root)
    if _VERIFIED_DIAGNOSTIC_BINDINGS:
        if _VERIFIED_DIAGNOSTIC_BINDINGS != bindings:
            raise RuntimeError("verified diagnostic source set drift")
        modules = _VERIFIED_DIAGNOSTIC_MODULES
    else:
        _drop_preloaded_diagnostic_modules()
        finder = _DiagnosticVerifiedFinder(frozen)
        sys.meta_path.insert(0, finder)
        modules: dict[str, object] = {}
        try:
            for name, module_name in _DIAGNOSTIC_MODULE_NAMES.items():
                module = importlib.import_module(module_name)
                if (
                    getattr(module, "__verified_source_sha256__", None)
                    != bindings[name]["sha256"]
                ):
                    raise RuntimeError(
                        f"diagnostic module was not loaded from verified bytes: {name}"
                    )
                modules[name] = module
        finally:
            try:
                sys.meta_path.remove(finder)
            except ValueError:
                pass
        _VERIFIED_DIAGNOSTIC_BINDINGS = bindings
        _VERIFIED_DIAGNOSTIC_MODULES = modules

    pilot_runner = modules["pilot_source"]
    uq_module = modules["paired_cluster_uq_source"]
    current_module = sys.modules.get(__name__)
    if current_module is not None:
        launcher_sha256 = getattr(
            current_module,
            "__verified_external_launcher_sha256__",
            None,
        )
        bootstrap_binding = getattr(
            current_module,
            "__verified_bootstrap_source_binding__",
            None,
        )
        if isinstance(launcher_sha256, str) and isinstance(
            bootstrap_binding,
            Mapping,
        ):
            pilot_runner.__verified_external_launcher_sha256__ = launcher_sha256
            pilot_runner.__verified_bootstrap_source_binding__ = dict(bootstrap_binding)
    NormUCB = uq_module.NormUCB
    paired_density_trace_ucb = uq_module.paired_density_trace_ucb
    paired_vector_norm_ucb = uq_module.paired_vector_norm_ucb


def _verify_live_binding(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{label} binding schema drift")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} binding escapes root") from exc
    if dict(binding) != _binding(path, root):
        raise ValueError(f"{label} live binding drift")
    return path


def _verify_report_bindings(root: Path, report: Mapping[str, Any], label: str) -> None:
    bindings = report.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ValueError(f"{label} bindings missing")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"{label}/{name} binding type drift")
        _verify_live_binding(root, binding, f"{label}/{name}")


def _seed(namespace: int, gate_id: str) -> int:
    return (namespace << 64) | int.from_bytes(
        sha256(gate_id.encode("utf-8")).digest()[:8], "big"
    )


def _embed_density(matrix: np.ndarray, lower: int, upper: int) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (3 * lower, 3 * lower) or not 0 < lower < upper:
        raise ValueError("cutoff density embedding mismatch")
    output = np.zeros((3 * upper, 3 * upper), dtype=np.complex128)
    output[: 3 * lower, : 3 * lower] = value
    return output


def _load_inputs(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, object]],
]:
    _activate_verified_diagnostic_modules(root)
    config, base = pilot_runner.load_pilot_config(root, require_hardened=True)
    if pilot_runner.CONFIG_PATH != CONFIG_PATH:
        raise ValueError("diagnostic/pilot config path drift")
    execution = pilot_runner.materialize_execution_config(config, base)
    if (root / pilot_runner.OWNER_LOCK_PATH).exists():
        raise ValueError("pilot supervisor is still active or stale")
    run_identity_path = root / str(config["artifact_paths"]["run_identity"])
    run_identity_binding = _binding(run_identity_path, root)
    _, run_identity = pilot_runner._read_bound_json(
        root,
        run_identity_binding,
    )
    _verify_self_hash(run_identity, "pilot run identity")
    pilot_snapshot = run_identity.get("input_snapshot")
    if not isinstance(pilot_snapshot, Mapping):
        raise ValueError("pilot input snapshot missing from run identity")
    pilot_runner._assert_input_snapshot(root, pilot_snapshot)
    pilot_runner._activate_verified_execution_modules(root, pilot_snapshot)
    pilot_runner._assert_input_snapshot(root, pilot_snapshot)
    cells = pilot_runner.build_pilot_cells(config, execution)
    manifest_path = root / str(config["artifact_paths"]["execution_manifest"])
    manifest_binding = _binding(manifest_path, root)
    _, manifest = pilot_runner._read_bound_json(root, manifest_binding)
    _verify_self_hash(manifest, "pilot manifest")
    pilot_runner._verify_manifest(
        root, config, execution, cells, run_identity, manifest
    )
    pilot_runner._verify_complete_marker(root, config, run_identity, manifest)
    if (
        manifest.get("task_id") != TASK_ID
        or manifest.get("schema_version") != pilot_runner.MANIFEST_SCHEMA
        or manifest.get("status") != pilot_runner.STATUS
        or manifest.get("scientific_verdict") is not None
        or manifest.get("qualified_claim") is not None
        or manifest.get("exception_rows") != 0
        or manifest.get("conservation_failure_rows") != 0
        or manifest.get("observed_cells") != 32
        or manifest.get("observed_rows") != 27648
        or manifest.get("claim_state") != pilot_runner.CLAIM_BOUNDARY
    ):
        raise ValueError("high-cutoff pilot manifest incomplete or contaminated")
    uq_binding = _binding(root / UQ_REPORT_PATH, root)
    _, uq = pilot_runner._read_bound_json(root, uq_binding)
    _verify_self_hash(uq, "UQ calibration report")
    _verify_report_bindings(root, uq, "UQ calibration report")
    required = config["diagnostic_contract"]["calibration_factor_source"]
    if (
        uq.get("analysis_sha256") != required["required_analysis_sha256"]
        or uq.get("claim_state") != UQ_CLAIM_BOUNDARY
        or uq.get("selected_calibration_factor") != required["required_factor"]
        or uq.get("validation_coverage_summary", {}).get("all_cells_passed")
        is not required["required_coverage_all_passed"]
    ):
        raise ValueError("coverage-calibrated factor binding drift")
    extension_binding = _binding(root / UQ_EXTENSION_PATH, root)
    _, extension = pilot_runner._read_bound_json(root, extension_binding)
    _verify_self_hash(extension, "UQ power extension report")
    _verify_report_bindings(root, extension, "UQ power extension report")
    if (
        extension.get("claim_state") != EXTENSION_CLAIM_BOUNDARY
        or extension.get("qualified_claim") is not None
        or extension.get("parent_analysis_sha256") != uq["analysis_sha256"]
    ):
        raise ValueError("historical UQ extension lineage drift")
    hardened = pilot_runner._validate_hardened_confirmation(root, config)
    snapshot: dict[str, dict[str, object]] = {
        str(name): dict(binding)
        for name, binding in pilot_snapshot.items()
        if isinstance(binding, Mapping)
    }
    if len(snapshot) != len(pilot_snapshot):
        raise ValueError("pilot input snapshot entry drift")
    snapshot.update(
        {
            "pilot_run_identity": run_identity_binding,
            "pilot_manifest": manifest_binding,
            "pilot_complete_marker": _binding(
                root / str(config["artifact_paths"]["heartbeat"]),
                root,
            ),
            "uq_calibration": uq_binding,
            "uq_power_extension": extension_binding,
            "diagnostic_source": _binding(Path(__file__).resolve(), root),
            "paired_cluster_uq_source": _binding(
                root / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py",
                root,
            ),
        }
    )
    current_module = sys.modules.get(__name__)
    if current_module is None:
        raise ValueError("diagnostic module identity missing")
    for snapshot_name, attribute in (
        (
            "verified_diagnostic_bootstrap_source",
            "__verified_bootstrap_source_binding__",
        ),
        (
            "verified_diagnostic_external_launch_meta",
            "__verified_launch_meta_binding__",
        ),
    ):
        binding = getattr(current_module, attribute, None)
        if not isinstance(binding, Mapping):
            raise ValueError(f"{snapshot_name} binding missing")
        snapshot[snapshot_name] = dict(binding)
    launch_binding = snapshot["verified_diagnostic_external_launch_meta"]
    launch_payload = getattr(current_module, "__verified_launch_meta_payload__", None)
    _, disk_launch_payload = pilot_runner._read_bound_json(root, launch_binding)
    if not isinstance(launch_payload, Mapping) or disk_launch_payload != dict(
        launch_payload
    ):
        raise ValueError("verified diagnostic launch meta content drift")
    for report_name, evidence_report in (
        ("uq_calibration", uq),
        ("uq_power_extension", extension),
        ("uq_hardened_confirmation", hardened),
    ):
        bindings = evidence_report.get("bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError(f"{report_name} evidence bindings missing")
        for name, binding in bindings.items():
            if not isinstance(binding, Mapping):
                raise ValueError(f"{report_name}/{name} binding type drift")
            snapshot[f"{report_name}_binding/{name}"] = dict(binding)
        chunk_bindings = evidence_report.get("chunk_bindings", [])
        if not isinstance(chunk_bindings, list):
            raise ValueError(f"{report_name} chunk bindings drift")
        for index, binding in enumerate(chunk_bindings):
            if not isinstance(binding, Mapping):
                raise ValueError(f"{report_name} chunk binding type drift")
            snapshot[f"{report_name}_chunk/{index:03d}"] = dict(binding)
    if snapshot["diagnostic_source"]["sha256"] != _diagnostic_source_sha256():
        raise ValueError("diagnostic source snapshot drift")
    receipts = manifest.get("chunk_receipts")
    receipt_bindings = manifest.get("receipt_bindings")
    if not isinstance(receipts, list) or not isinstance(receipt_bindings, list):
        raise ValueError("pilot receipt snapshot accounting missing")
    for index, (receipt, receipt_binding) in enumerate(
        zip(receipts, receipt_bindings, strict=True)
    ):
        if not isinstance(receipt, Mapping) or not isinstance(receipt_binding, Mapping):
            raise ValueError("pilot receipt snapshot type drift")
        snapshot[f"pilot_receipt/{index:02d}"] = dict(receipt_binding)
        for artifact in ("csv", "npz"):
            binding = receipt.get(artifact)
            if not isinstance(binding, Mapping):
                raise ValueError(f"pilot {artifact} snapshot binding missing")
            snapshot[f"pilot_{artifact}/{index:02d}"] = dict(binding)
    pilot_runner._assert_input_snapshot(root, snapshot)
    _assert_diagnostic_import_snapshot(root, snapshot)
    return config, manifest, uq, extension, hardened, snapshot


def _parse_chunk(
    root: Path,
    receipt: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    _verify_self_hash(receipt, "pilot receipt")
    cell_payload = receipt.get("cell")
    if not isinstance(cell_payload, Mapping):
        raise ValueError("pilot receipt cell missing")
    chunk_id = str(cell_payload.get("chunk_id"))
    expected_receipt_path = (
        root / str(config["artifact_paths"]["receipt_directory"]) / f"{chunk_id}.json"
    )
    live_receipt_path, live_receipt = pilot_runner._read_bound_json(
        root,
        receipt_binding,
        expected_path=expected_receipt_path.relative_to(root).as_posix(),
    )
    if live_receipt_path != expected_receipt_path.resolve():
        raise ValueError("pilot receipt path identity drift")
    if live_receipt != receipt:
        raise ValueError("pilot manifest/live receipt drift")
    if (
        receipt.get("task_id") != TASK_ID
        or receipt.get("schema_version") != pilot_runner.RECEIPT_SCHEMA
        or receipt.get("run_id") != manifest.get("run_id")
        or receipt.get("run_identity_analysis_sha256")
        != manifest.get("run_identity_analysis_sha256")
        or receipt.get("config_analysis_sha256")
        != manifest.get("config_analysis_sha256")
        or receipt.get("execution_analysis_sha256")
        != manifest.get("execution_analysis_sha256")
        or receipt.get("input_snapshot_analysis_sha256")
        != manifest.get("input_snapshot_analysis_sha256")
        or receipt.get("pilot_source_sha256") != manifest.get("pilot_source_sha256")
        or receipt.get("chunk_id") != chunk_id
    ):
        raise ValueError("pilot receipt identity drift")
    csv_binding = receipt["csv"]
    npz_binding = receipt["npz"]
    _, csv_bytes = pilot_runner._read_bound_bytes(root, csv_binding)
    _, npz_bytes = pilot_runner._read_bound_bytes(root, npz_binding)
    rows: list[dict[str, Any]] = []
    with io.StringIO(csv_bytes.decode("utf-8"), newline="") as stream:
        for raw in csv.DictReader(stream):
            if raw["exception_type"]:
                raise ValueError("pilot contains exception row")
            if raw["conservation_pass"] != "True":
                raise ValueError("pilot contains conservation failure")
            rows.append(
                {
                    **raw,
                    "cutoff": int(raw["cutoff"]),
                    "seed_position": int(raw["seed_position"]),
                    "round_index": int(raw["round_index"]),
                    "terminal_round": raw["terminal_round"].lower() == "true",
                    "mean_photon": float(raw["mean_photon"]),
                    "level_g": float(raw["level_g"]),
                    "level_e": float(raw["level_e"]),
                    "level_f": float(raw["level_f"]),
                    "logical_survival": float(raw["logical_survival"]),
                    "density_quantization_trace_distance_bound": float(
                        raw["density_quantization_trace_distance_bound"]
                    ),
                }
            )
    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as archive:
        density_ids = [str(value) for value in archive["density_row_ids"].tolist()]
        densities = np.asarray(archive["densities"], dtype=np.complex128)
    if len(density_ids) != len(densities):
        raise ValueError("pilot density row alignment drift")
    if len(densities):
        hermitian_error = float(
            np.max(np.abs(densities - densities.conj().transpose(0, 2, 1)))
        )
        trace_error = float(np.max(np.abs(np.trace(densities, axis1=1, axis2=2) - 1.0)))
        minimum_eigenvalue = min(
            float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min())
            for matrix in densities
        )
        if hermitian_error > 5e-5 or trace_error > 5e-5 or minimum_eigenvalue < -5e-5:
            raise ValueError("pilot density physicality drift")
    return rows, dict(zip(density_ids, densities))


def load_pilot_evidence(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    input_snapshot: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    pilot_runner._assert_input_snapshot(root, input_snapshot)
    rows: list[dict[str, Any]] = []
    densities: dict[str, np.ndarray] = {}
    receipts = manifest["chunk_receipts"]
    receipt_bindings = manifest["receipt_bindings"]
    if len(receipts) != len(receipt_bindings):
        raise ValueError("pilot receipt binding denominator drift")
    for receipt, receipt_binding in zip(receipts, receipt_bindings, strict=True):
        chunk_rows, chunk_densities = _parse_chunk(
            root,
            receipt,
            receipt_binding,
            config=config,
            manifest=manifest,
        )
        rows.extend(chunk_rows)
        if set(densities) & set(chunk_densities):
            raise ValueError("duplicate pilot density row id")
        densities.update(chunk_densities)
        pilot_runner._assert_input_snapshot(root, input_snapshot)
    if len(rows) != manifest["observed_rows"]:
        raise ValueError("pilot diagnostic row denominator drift")
    terminal_ids = {str(row["row_id"]) for row in rows if row["terminal_round"]}
    if set(densities) != terminal_ids:
        raise ValueError("pilot terminal density coverage drift")
    pilot_runner._assert_input_snapshot(root, input_snapshot)
    return rows, densities


def _indexed_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["cutoff"]),
            str(row["scenario"]),
            str(row["backend"]),
            str(row["logical_label"]),
            int(row["seed_position"]),
        )
        grouped.setdefault(key, []).append(dict(row))
    for key, values in grouped.items():
        values.sort(key=lambda row: int(row["round_index"]))
        expected_label = ("0", "1", "+", "-", "+i", "-i")[key[4] % 6]
        if key[3] != expected_label or [row["round_index"] for row in values] != list(
            range(12)
        ):
            raise ValueError("pilot state schedule or round coverage drift")
    return grouped


def _state_positions(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
) -> list[int]:
    values = sorted(
        key[4] for key in grouped if key[:4] == (cutoff, scenario, backend, state)
    )
    if len(values) != 12 or any(
        position % 6 != ("0", "1", "+", "-", "+i", "-i").index(state)
        for position in values
    ):
        raise ValueError("pilot per-state cluster denominator drift")
    return values


def _stage_matrix(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    *,
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
    rounds: Sequence[int],
    fields: Sequence[str],
) -> tuple[np.ndarray, list[int]]:
    positions = _state_positions(grouped, cutoff, scenario, backend, state)
    matrix = []
    selected = set(int(value) for value in rounds)
    for position in positions:
        values = [
            row
            for row in grouped[(cutoff, scenario, backend, state, position)]
            if int(row["round_index"]) in selected
        ]
        if len(values) != len(selected):
            raise ValueError("pilot stage round denominator drift")
        matrix.append(
            [float(np.mean([float(row[field]) for row in values])) for field in fields]
        )
    return np.asarray(matrix, dtype=np.float64), positions


def _terminal_density_stack(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    densities: Mapping[str, np.ndarray],
    *,
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    positions = _state_positions(grouped, cutoff, scenario, backend, state)
    stack = []
    quantization = []
    for position in positions:
        terminal = grouped[(cutoff, scenario, backend, state, position)][-1]
        if not terminal["terminal_round"]:
            raise ValueError("pilot terminal row drift")
        stack.append(densities[str(terminal["row_id"])])
        quantization.append(
            float(terminal["density_quantization_trace_distance_bound"])
        )
    return (
        np.asarray(stack, dtype=np.complex128),
        np.asarray(quantization, dtype=np.float64),
        positions,
    )


def _terminal_truncation_features(
    densities: np.ndarray,
    *,
    cutoff: int,
) -> dict[str, np.ndarray]:
    """Return state-weighted Fock-boundary diagnostics per trajectory cluster."""

    stack = np.asarray(densities, dtype=np.complex128)
    expected_dimension = 3 * int(cutoff)
    if (
        stack.ndim != 3
        or stack.shape[0] < 2
        or stack.shape[1:] != (expected_dimension, expected_dimension)
    ):
        raise ValueError("terminal truncation density shape drift")
    joint = stack.reshape(len(stack), cutoff, 3, cutoff, 3)
    oscillator = np.trace(joint, axis1=2, axis2=4)
    diagonal = np.diagonal(oscillator, axis1=1, axis2=2)
    if (
        not np.all(np.isfinite(diagonal.real))
        or not np.all(np.isfinite(diagonal.imag))
        or float(np.max(np.abs(diagonal.imag))) > 5.0e-8
    ):
        raise ValueError("terminal oscillator population drift")
    populations = diagonal.real
    if (
        float(np.min(populations)) < -5.0e-5
        or float(np.max(np.abs(np.sum(populations, axis=1) - 1.0))) > 5.0e-5
    ):
        raise ValueError("terminal oscillator population physicality drift")
    fock_index = np.arange(cutoff, dtype=np.float64)
    top1 = np.sum(populations[:, -1:], axis=1)
    return {
        "top1_fock_mass": top1,
        "top2_fock_mass": np.sum(populations[:, -min(2, cutoff) :], axis=1),
        "top4_fock_mass": np.sum(populations[:, -min(4, cutoff) :], axis=1),
        "normalized_mean_photon": populations @ fock_index / float(cutoff - 1),
        # [a,a†] = I - cutoff |cutoff-1><cutoff-1| in the truncated basis.
        "commutator_defect": float(cutoff) * top1,
    }


def _result_row(
    *,
    gate_id: str,
    contrast: str,
    scenario: str,
    state: str,
    stage: str,
    metric: str,
    margin: float,
    ucb: NormUCB,
    cutoff: str,
    backend: str,
) -> dict[str, object]:
    point_signal = (
        "STRONG_EXPLORATORY_RISK_SIGNAL"
        if ucb.estimate > 2.0 * margin
        else (
            "CANDIDATE_EXPLORATORY_RISK_SIGNAL"
            if ucb.estimate > margin
            else "NO_LARGE_SIGNAL_INCONCLUSIVE"
        )
    )
    return {
        "gate_id": gate_id,
        "contrast": contrast,
        "scenario": scenario,
        "logical_state": state,
        "stage": stage,
        "metric": metric,
        "cutoff_or_increment": cutoff,
        "backend_or_pair": backend,
        "estimate": ucb.estimate,
        "raw_radius": ucb.raw_radius,
        "calibrated_radius": ucb.calibrated_radius,
        "quantization_bound": ucb.quantization_bound,
        "upper_bound": ucb.upper_bound,
        "margin": margin,
        "signal_class": point_signal,
        "signal_repeated": False,
        "negative_interpretation": "inconclusive",
        "qualification_effect": None,
        "cluster_count": ucb.cluster_count,
        "multiplier_replicates": ucb.multiplier_replicates,
        "multiplier_seed": ucb.seed,
        "design_pilot_only": True,
    }


def _promote_repeated_risk_signals(
    rows: Sequence[dict[str, object]],
) -> None:
    signal_rows = [
        row
        for row in rows
        if row["signal_class"]
        in {
            "CANDIDATE_EXPLORATORY_RISK_SIGNAL",
            "STRONG_EXPLORATORY_RISK_SIGNAL",
        }
    ]
    for row in signal_rows:
        peers = [
            peer
            for peer in signal_rows
            if peer is not row
            and peer["scenario"] == row["scenario"]
            and peer["logical_state"] == row["logical_state"]
            and peer["stage"] == row["stage"]
            and peer["metric"] == row["metric"]
            and peer["contrast"] == row["contrast"]
        ]
        repeated = False
        if row["contrast"] == "within_backend_cutoff":
            repeated = any(
                peer["cutoff_or_increment"] == row["cutoff_or_increment"]
                and peer["backend_or_pair"] != row["backend_or_pair"]
                for peer in peers
            )
            increment = str(row["cutoff_or_increment"]).split("->")
            if len(increment) == 2:
                lower, upper = (int(value) for value in increment)
                for peer in peers:
                    peer_increment = str(peer["cutoff_or_increment"]).split("->")
                    if (
                        peer["backend_or_pair"] == row["backend_or_pair"]
                        and len(peer_increment) == 2
                    ):
                        peer_lower, peer_upper = (
                            int(value) for value in peer_increment
                        )
                        repeated = repeated or (
                            upper == peer_lower or peer_upper == lower
                        )
        if repeated:
            row["signal_repeated"] = True
            if row["signal_class"] == "CANDIDATE_EXPLORATORY_RISK_SIGNAL":
                row["signal_class"] = "STRONG_EXPLORATORY_RISK_SIGNAL"


def evaluate_diagnostics(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    densities: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    grouped = _indexed_rows(rows)
    contract = config["diagnostic_contract"]
    factor = float(contract["calibration_factor_source"]["required_factor"])
    confidence = float(contract["confidence"])
    replicates = int(contract["multiplier_replicates"])
    namespace = int(contract["multiplier_seed_namespace"])
    margins = contract["margins"]
    states = list(config["logical_state_schedule"])
    results: list[dict[str, object]] = []

    def vector_result(
        *,
        gate_id: str,
        left: np.ndarray,
        right: np.ndarray,
        ord_value: float,
        margin_key: str,
        contrast: str,
        scenario: str,
        state: str,
        stage: str,
        metric: str,
        cutoff: str,
        backend: str,
    ) -> None:
        ucb = paired_vector_norm_ucb(
            left,
            right,
            ord_value=ord_value,
            confidence=confidence,
            multiplier_replicates=replicates,
            seed=_seed(namespace, gate_id),
            calibration_factor=factor,
        )
        results.append(
            _result_row(
                gate_id=gate_id,
                contrast=contrast,
                scenario=scenario,
                state=state,
                stage=stage,
                metric=metric,
                margin=float(margins[margin_key]),
                ucb=ucb,
                cutoff=cutoff,
                backend=backend,
            )
        )

    def absolute_truncation_result(
        *,
        gate_id: str,
        values: np.ndarray,
        quantization_bounds: np.ndarray,
        quantization_scale: float,
        margin_key: str,
        scenario: str,
        state: str,
        metric: str,
        cutoff: int,
        backend: str,
    ) -> None:
        base = paired_vector_norm_ucb(
            values,
            np.zeros_like(values),
            ord_value=1,
            confidence=confidence,
            multiplier_replicates=replicates,
            seed=_seed(namespace, gate_id),
            calibration_factor=factor,
        )
        quantization_bound = float(
            quantization_scale * np.mean(quantization_bounds)
        )
        ucb = NormUCB(
            estimate=base.estimate,
            raw_radius=base.raw_radius,
            calibrated_radius=base.calibrated_radius,
            quantization_bound=quantization_bound,
            upper_bound=base.upper_bound + quantization_bound,
            confidence=base.confidence,
            multiplier_replicates=base.multiplier_replicates,
            cluster_count=base.cluster_count,
            calibration_factor=base.calibration_factor,
            seed=base.seed,
        )
        results.append(
            _result_row(
                gate_id=gate_id,
                contrast="absolute_truncation_risk",
                scenario=scenario,
                state=state,
                stage="terminal",
                metric=metric,
                margin=float(margins[margin_key]),
                ucb=ucb,
                cutoff=str(cutoff),
                backend=backend,
            )
        )

    for cutoff in config["cutoffs"]:
        for scenario in config["scenario_names"]:
            for state in states:
                density_a, quant_a, positions_a = _terminal_density_stack(
                    grouped,
                    densities,
                    cutoff=int(cutoff),
                    scenario=scenario,
                    backend="A",
                    state=state,
                )
                density_b, quant_b, positions_b = _terminal_density_stack(
                    grouped,
                    densities,
                    cutoff=int(cutoff),
                    scenario=scenario,
                    backend="B",
                    state=state,
                )
                if positions_a != positions_b:
                    raise ValueError("pilot A/B state positions drift")
                gate_id = f"ab/c{cutoff}/{scenario}/{state}/terminal_density"
                density_ucb = paired_density_trace_ucb(
                    density_a,
                    density_b,
                    confidence=confidence,
                    multiplier_replicates=replicates,
                    seed=_seed(namespace, gate_id),
                    calibration_factor=factor,
                    quantization_bounds=quant_a + quant_b,
                )
                results.append(
                    _result_row(
                        gate_id=gate_id,
                        contrast="same_cutoff_ab",
                        scenario=scenario,
                        state=state,
                        stage="terminal",
                        metric="density_trace_distance",
                        margin=float(margins["ab_terminal_density_trace_distance"]),
                        ucb=density_ucb,
                        cutoff=str(cutoff),
                        backend="A/B",
                    )
                )
                for backend, density_stack, quantization in (
                    ("A", density_a, quant_a),
                    ("B", density_b, quant_b),
                ):
                    features = _terminal_truncation_features(
                        density_stack,
                        cutoff=int(cutoff),
                    )
                    for metric, margin_key, quantization_scale in (
                        (
                            "top1_fock_mass",
                            "absolute_terminal_top1_fock_mass",
                            1.0,
                        ),
                        (
                            "top2_fock_mass",
                            "absolute_terminal_top2_fock_mass",
                            1.0,
                        ),
                        (
                            "top4_fock_mass",
                            "absolute_terminal_top4_fock_mass",
                            1.0,
                        ),
                        (
                            "normalized_mean_photon",
                            "absolute_terminal_normalized_mean_photon",
                            1.0,
                        ),
                        (
                            "commutator_defect",
                            "absolute_terminal_commutator_defect",
                            float(cutoff),
                        ),
                    ):
                        tail_gate = (
                            f"tail/c{cutoff}/{backend}/{scenario}/{state}/{metric}"
                        )
                        absolute_truncation_result(
                            gate_id=tail_gate,
                            values=features[metric],
                            quantization_bounds=quantization,
                            quantization_scale=quantization_scale,
                            margin_key=margin_key,
                            scenario=scenario,
                            state=state,
                            metric=metric,
                            cutoff=int(cutoff),
                            backend=backend,
                        )
                for stage, stage_rounds in config["stage_partition"][scenario].items():
                    a_values, a_positions = _stage_matrix(
                        grouped,
                        cutoff=int(cutoff),
                        scenario=scenario,
                        backend="A",
                        state=state,
                        rounds=stage_rounds,
                        fields=(
                            "mean_photon",
                            "level_g",
                            "level_e",
                            "level_f",
                            "logical_survival",
                        ),
                    )
                    b_values, b_positions = _stage_matrix(
                        grouped,
                        cutoff=int(cutoff),
                        scenario=scenario,
                        backend="B",
                        state=state,
                        rounds=stage_rounds,
                        fields=(
                            "mean_photon",
                            "level_g",
                            "level_e",
                            "level_f",
                            "logical_survival",
                        ),
                    )
                    if a_positions != b_positions:
                        raise ValueError("pilot A/B stage positions drift")
                    specs = (
                        (
                            "mean_photon",
                            a_values[:, 0],
                            b_values[:, 0],
                            1,
                            "ab_terminal_mean_photon_difference",
                        ),
                        (
                            "level_probability_l1",
                            a_values[:, 1:4],
                            b_values[:, 1:4],
                            1,
                            "ab_terminal_level_probability_l1",
                        ),
                        (
                            "logical_survival",
                            a_values[:, 4],
                            b_values[:, 4],
                            1,
                            "ab_terminal_logical_survival_difference",
                        ),
                    )
                    for metric, left, right, ord_value, margin_key in specs:
                        stage_gate = f"ab/c{cutoff}/{scenario}/{state}/{stage}/{metric}"
                        vector_result(
                            gate_id=stage_gate,
                            left=left,
                            right=right,
                            ord_value=ord_value,
                            margin_key=margin_key,
                            contrast="same_cutoff_ab",
                            scenario=scenario,
                            state=state,
                            stage=stage,
                            metric=metric,
                            cutoff=str(cutoff),
                            backend="A/B",
                        )

    for lower, upper in zip(config["cutoffs"][:-1], config["cutoffs"][1:]):
        for scenario in config["scenario_names"]:
            for state in states:
                for backend in ("A", "B"):
                    low_density, low_quant, low_positions = _terminal_density_stack(
                        grouped,
                        densities,
                        cutoff=int(lower),
                        scenario=scenario,
                        backend=backend,
                        state=state,
                    )
                    high_density, high_quant, high_positions = _terminal_density_stack(
                        grouped,
                        densities,
                        cutoff=int(upper),
                        scenario=scenario,
                        backend=backend,
                        state=state,
                    )
                    if low_positions != high_positions:
                        raise ValueError("pilot cutoff state positions drift")
                    embedded = np.asarray(
                        [
                            _embed_density(value, int(lower), int(upper))
                            for value in low_density
                        ]
                    )
                    gate_id = (
                        f"cutoff/{lower}-{upper}/{backend}/{scenario}/{state}/"
                        "terminal_density"
                    )
                    density_ucb = paired_density_trace_ucb(
                        embedded,
                        high_density,
                        confidence=confidence,
                        multiplier_replicates=replicates,
                        seed=_seed(namespace, gate_id),
                        calibration_factor=factor,
                        quantization_bounds=low_quant + high_quant,
                    )
                    results.append(
                        _result_row(
                            gate_id=gate_id,
                            contrast="within_backend_cutoff",
                            scenario=scenario,
                            state=state,
                            stage="terminal",
                            metric="density_trace_distance",
                            margin=float(
                                margins["cutoff_terminal_density_trace_distance"]
                            ),
                            ucb=density_ucb,
                            cutoff=f"{lower}->{upper}",
                            backend=backend,
                        )
                    )
                    for stage, stage_rounds in config["stage_partition"][
                        scenario
                    ].items():
                        low_values, low_stage_positions = _stage_matrix(
                            grouped,
                            cutoff=int(lower),
                            scenario=scenario,
                            backend=backend,
                            state=state,
                            rounds=stage_rounds,
                            fields=(
                                "mean_photon",
                                "level_g",
                                "level_e",
                                "level_f",
                                "logical_survival",
                            ),
                        )
                        high_values, high_stage_positions = _stage_matrix(
                            grouped,
                            cutoff=int(upper),
                            scenario=scenario,
                            backend=backend,
                            state=state,
                            rounds=stage_rounds,
                            fields=(
                                "mean_photon",
                                "level_g",
                                "level_e",
                                "level_f",
                                "logical_survival",
                            ),
                        )
                        if low_stage_positions != high_stage_positions:
                            raise ValueError("pilot cutoff stage positions drift")
                        specs = (
                            (
                                "mean_photon",
                                low_values[:, 0],
                                high_values[:, 0],
                                1,
                                "cutoff_terminal_mean_photon_difference",
                            ),
                            (
                                "level_probability_l1",
                                low_values[:, 1:4],
                                high_values[:, 1:4],
                                1,
                                "cutoff_terminal_level_probability_l1",
                            ),
                            (
                                "logical_survival",
                                low_values[:, 4],
                                high_values[:, 4],
                                1,
                                "cutoff_terminal_logical_survival_difference",
                            ),
                        )
                        for metric, left, right, ord_value, margin_key in specs:
                            stage_gate = (
                                f"cutoff/{lower}-{upper}/{backend}/{scenario}/"
                                f"{state}/{stage}/{metric}"
                            )
                            vector_result(
                                gate_id=stage_gate,
                                left=left,
                                right=right,
                                ord_value=ord_value,
                                margin_key=margin_key,
                                contrast="within_backend_cutoff",
                                scenario=scenario,
                                state=state,
                                stage=stage,
                                metric=metric,
                                cutoff=f"{lower}->{upper}",
                                backend=backend,
                            )
    identifiers = [str(row["gate_id"]) for row in results]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate pilot diagnostic gate id")
    _promote_repeated_risk_signals(results)
    return results


def _build_report_core(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    config, manifest, uq, extension, hardened, input_snapshot = _load_inputs(root)
    rows, densities = load_pilot_evidence(
        root,
        config,
        manifest,
        input_snapshot,
    )
    diagnostics = evaluate_diagnostics(config, rows, densities)
    cutoff_24_28 = [
        row
        for row in diagnostics
        if row["contrast"] == "within_backend_cutoff"
        and row["metric"] == "density_trace_distance"
        and row["cutoff_or_increment"] == "24->28"
    ]
    cutoff_28_tail = [
        row
        for row in diagnostics
        if row["contrast"] == "absolute_truncation_risk"
        and row["cutoff_or_increment"] == "28"
    ]
    trigger_threshold = 0.075
    density_trigger = any(
        float(row["estimate"]) > trigger_threshold for row in cutoff_24_28
    )
    tail_trigger_ids = [
        str(row["gate_id"])
        for row in cutoff_28_tail
        if float(row["estimate"]) > float(row["margin"])
    ]
    trigger_32 = density_trigger or bool(tail_trigger_ids)
    aggregate_ledger: list[dict[str, Any]] = []
    for contrast, metric in sorted(
        {(str(row["contrast"]), str(row["metric"])) for row in diagnostics}
    ):
        strata = [
            row
            for row in diagnostics
            if row["contrast"] == contrast and row["metric"] == metric
        ]
        worst = max(strata, key=lambda row: float(row["estimate"]))
        candidate = [
            str(row["gate_id"])
            for row in strata
            if row["signal_class"] == "CANDIDATE_EXPLORATORY_RISK_SIGNAL"
        ]
        strong = [
            str(row["gate_id"])
            for row in strata
            if row["signal_class"] == "STRONG_EXPLORATORY_RISK_SIGNAL"
        ]
        aggregate_ledger.append(
            {
                "contrast": contrast,
                "metric": metric,
                "stratum_count": len(strata),
                "candidate_signal_count": len(candidate),
                "candidate_gate_ids": candidate,
                "strong_signal_count": len(strong),
                "strong_gate_ids": strong,
                "worst_gate_id": str(worst["gate_id"]),
                "worst_point_estimate": float(worst["estimate"]),
                "global_iut_pass": None,
                "design_pilot_only": True,
            }
        )
    candidate_ids = [
        str(row["gate_id"])
        for row in diagnostics
        if row["signal_class"] == "CANDIDATE_EXPLORATORY_RISK_SIGNAL"
    ]
    strong_ids = [
        str(row["gate_id"])
        for row in diagnostics
        if row["signal_class"] == "STRONG_EXPLORATORY_RISK_SIGNAL"
    ]
    verdict = RISK_VERDICT if candidate_ids or strong_ids else INCONCLUSIVE_VERDICT
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "status": STATUS,
        "scientific_verdict": verdict,
        "qualified_claim": None,
        "design_pilot_only": True,
        "authorization_state": pilot_runner.NARROW_AUTHORIZATION_STATE,
        "input_snapshot": input_snapshot,
        "input_snapshot_analysis_sha256": _sha(input_snapshot),
        "diagnostic_count": len(diagnostics),
        "exploratory_signal_summary": {
            "aggregation": "state×scenario×stage exploratory localization",
            "stratum_count": len(diagnostics),
            "candidate_signal_count": len(candidate_ids),
            "candidate_gate_ids": candidate_ids,
            "strong_signal_count": len(strong_ids),
            "strong_gate_ids": strong_ids,
            "global_iut_pass": None,
            "aggregate_ledger": aggregate_ledger,
            "formal_qualification_effect": None,
            "negative_result_interpretation": "inconclusive",
        },
        "maxima": {
            metric: {
                "point_estimate": max(
                    float(row["estimate"])
                    for row in diagnostics
                    if row["metric"] == metric
                ),
                "upper_bound": max(
                    float(row["upper_bound"])
                    for row in diagnostics
                    if row["metric"] == metric
                ),
            }
            for metric in sorted({str(row["metric"]) for row in diagnostics})
        },
        "cutoff_32_exploratory_candidate": {
            "registered_trigger": (
                "any state×scenario×backend 24->28 terminal-density point > "
                "0.075 or any cutoff-28 absolute truncation-risk point exceeds "
                "its preregistered margin"
            ),
            "density_threshold": trigger_threshold,
            "observed_maximum_density_point": max(
                float(row["estimate"]) for row in cutoff_24_28
            ),
            "density_triggered": density_trigger,
            "tail_margin_by_metric": {
                metric: float(config["diagnostic_contract"]["margins"][margin_key])
                for metric, margin_key in (
                    ("top1_fock_mass", "absolute_terminal_top1_fock_mass"),
                    ("top2_fock_mass", "absolute_terminal_top2_fock_mass"),
                    ("top4_fock_mass", "absolute_terminal_top4_fock_mass"),
                    (
                        "normalized_mean_photon",
                        "absolute_terminal_normalized_mean_photon",
                    ),
                    ("commutator_defect", "absolute_terminal_commutator_defect"),
                )
            },
            "tail_trigger_gate_ids": tail_trigger_ids,
            "candidate_for_followup": trigger_32,
            "selected": None,
            "selection_effect": None,
        },
        "formal_design": None,
        "scope_guard": {
            "purpose": "unpowered exploratory localization",
            "n12_role": "localization_only",
            "calibration_factor": hardened["frozen_parent_calibration_factor"],
            "factor_interpretation": (
                "synthetic coverage calibration only; no physical coverage guarantee"
            ),
            "hardened_confirmation_analysis_sha256": hardened["analysis_sha256"],
            "design_multiplier_replicates": 199,
            "positive_signal_effect": (
                "may only exclude a candidate or trigger physics repair and "
                "powered confirmation"
            ),
            "aggregation": "state×scenario×stage exploratory localization",
            "absence_effect": "inconclusive; cannot qualify any candidate",
            "equivalence_conclusion": None,
            "formal_cutoff_selection": None,
            "formal_sample_count_selection": None,
        },
        "preserved_no_go": {
            "historical_t9_2_4": "NO_GO_PRESERVED",
            "fresh_twin_qualification": "NO_GO_PRESERVED",
        },
        "downstream_state": {
            task: "BLOCKED"
            for task in (
                "T9.2.5",
                "T9.2.7",
                "T9.3.1",
                "T9.3.4",
                "T9.6.2",
                "T9.6.5",
            )
        },
        "claim_state": dict(config["claim_boundary"]),
        "bindings": {
            "config": _binding(root / CONFIG_PATH, root),
            "pilot_manifest": _binding(
                root / str(config["artifact_paths"]["execution_manifest"]), root
            ),
            "uq_calibration": _binding(root / UQ_REPORT_PATH, root),
            "uq_power_extension": _binding(root / UQ_EXTENSION_PATH, root),
            "uq_hardened_confirmation": _binding(
                root / str(config["hardened_confirmation_source"]["report"]["path"]),
                root,
            ),
            "uq_hardened_confirmation_source_data": _binding(
                root
                / str(config["hardened_confirmation_source"]["source_data"]["path"]),
                root,
            ),
            "diagnostic_source": _binding(Path(__file__).resolve(), root),
            "paired_cluster_uq_source": _binding(
                root / "cnn_fpga/benchmark/phase9_paired_cluster_uq.py", root
            ),
        },
    }
    pilot_runner._assert_input_snapshot(root, input_snapshot)
    _assert_diagnostic_import_snapshot(root, input_snapshot)
    return report, diagnostics


def _finalized_report_document(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    report, diagnostics = _build_report_core(root)
    source_path = root / SOURCE_PATH
    if not source_path.is_file():
        raise ValueError("diagnostic source data is absent")
    report["bindings"]["source_data"] = _binding(source_path, root)
    report["transaction"] = _transaction_contract()
    report["analysis_sha256"] = _sha(report)
    return report, diagnostics


def _assert_diagnostic_lock(
    root: Path,
    expected: Mapping[str, Any],
) -> None:
    path = root / DIAGNOSTIC_LOCK_PATH
    try:
        live = json.loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise RuntimeError("diagnostic owner lock disappeared") from exc
    _verify_self_hash(live, "diagnostic owner lock")
    if live != dict(expected):
        raise RuntimeError("diagnostic owner lock changed")
    if expected.get("diagnostic_source_sha256") != _diagnostic_source_sha256():
        raise RuntimeError("diagnostic source changed while lock was held")


@contextmanager
def _exclusive_diagnostic_lock(root: Path) -> Any:
    path = root / DIAGNOSTIC_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": DIAGNOSTIC_LOCK_SCHEMA,
        "owner_token": uuid4().hex,
        "pid": os.getpid(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_source_sha256": _diagnostic_source_sha256(),
    }
    payload["analysis_sha256"] = _sha(payload)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("diagnostic owner lock already exists") from exc
    try:
        encoded = (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _assert_diagnostic_lock(root, payload)
        yield payload
    finally:
        _assert_diagnostic_lock(root, payload)
        path.unlink()


def _transaction_contract() -> dict[str, Any]:
    return {
        "write_order": [
            "source_data",
            "report",
            "completion_receipt",
        ],
        "source_committed_before_report": True,
        "completion_receipt_is_final_commit_marker": True,
    }


def _source_csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    fields = sorted({key for row in rows for key in row})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _verify_published_report_and_source(
    root: Path,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
    *,
    report_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> None:
    _, disk_report = pilot_runner._read_bound_json(
        root,
        report_binding,
        expected_path=REPORT_PATH,
    )
    _verify_self_hash(disk_report, "published diagnostic report")
    if disk_report != dict(report):
        raise RuntimeError("published diagnostic report content drift")
    if (
        disk_report.get("schema_version") != SCHEMA
        or disk_report.get("status") != STATUS
        or disk_report.get("scientific_verdict")
        not in {RISK_VERDICT, INCONCLUSIVE_VERDICT}
        or disk_report.get("qualified_claim") is not None
    ):
        raise RuntimeError("published diagnostic report firewall drift")
    disk_claim_state = disk_report.get("claim_state")
    if not isinstance(disk_claim_state, Mapping) or any(
        value is not None
        for key, value in disk_claim_state.items()
        if key != "design_pilot_only"
    ):
        raise RuntimeError("published diagnostic claim state drift")
    disk_source_binding = disk_report.get("bindings", {}).get("source_data")
    if disk_source_binding != dict(source_binding):
        raise RuntimeError("published diagnostic source binding drift")
    _, disk_source = pilot_runner._read_bound_bytes(
        root,
        source_binding,
        expected_path=SOURCE_PATH,
    )
    if disk_source != _source_csv_bytes(rows):
        raise RuntimeError("published diagnostic source content drift")


def _completion_payload(
    root: Path,
    report: Mapping[str, Any],
    *,
    report_binding: Mapping[str, Any] | None = None,
    source_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bound_report = dict(report_binding or _binding(root / REPORT_PATH, root))
    bound_source = dict(source_binding or _binding(root / SOURCE_PATH, root))
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": DIAGNOSTIC_COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "report": bound_report,
        "source_data": bound_source,
        "report_analysis_sha256": report["analysis_sha256"],
        "input_snapshot_analysis_sha256": report["input_snapshot_analysis_sha256"],
        "pilot_manifest": dict(report["bindings"]["pilot_manifest"]),
        "scientific_verdict": report["scientific_verdict"],
        "qualified_claim": None,
        "claim_state": dict(report["claim_state"]),
    }
    payload["analysis_sha256"] = _sha(payload)
    return payload


def _verify_diagnostic_completion(
    root: Path,
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    path = root / COMPLETION_PATH
    completion = json.loads(path.read_bytes())
    if not isinstance(completion, dict):
        raise RuntimeError("diagnostic completion receipt is not an object")
    _verify_self_hash(completion, "diagnostic completion receipt")
    report_binding = completion.get("report")
    source_binding = completion.get("source_data")
    if not isinstance(report_binding, Mapping) or not isinstance(
        source_binding, Mapping
    ):
        raise RuntimeError("diagnostic completion artifact binding drift")
    expected = _completion_payload(
        root,
        report,
        report_binding=report_binding,
        source_binding=source_binding,
    )
    if set(completion) != set(expected) or completion != expected:
        raise RuntimeError("diagnostic completion receipt drift")
    _verify_published_report_and_source(
        root,
        report,
        rows,
        report_binding=report_binding,
        source_binding=source_binding,
    )
    return completion


def build_report(root: Path) -> tuple[dict[str, Any], list[dict[str, object]]]:
    """Rebuild a finalized report and require its final completion receipt."""

    _require_verified_self_import()
    report, diagnostics = _finalized_report_document(root)
    _verify_diagnostic_completion(root, report, diagnostics)
    return report, diagnostics


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


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    _require_verified_self_import()
    base = (root or _root()).resolve()
    with _exclusive_diagnostic_lock(base):
        completion_path = base / COMPLETION_PATH
        report_path = base / REPORT_PATH
        if completion_path.is_file():
            report, _rows = build_report(base)
            return report
        report_path.unlink(missing_ok=True)
        try:
            report, rows = _build_report_core(base)
            _atomic_text(
                base / SOURCE_PATH,
                _source_csv_bytes(rows).decode("utf-8"),
            )
            report["bindings"]["source_data"] = _binding(
                base / SOURCE_PATH,
                base,
            )
            report["transaction"] = _transaction_contract()
            report["analysis_sha256"] = _sha(report)
            _atomic_text(
                report_path,
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            rebuilt, rebuilt_rows = _finalized_report_document(base)
            if rebuilt != report or rebuilt_rows != rows:
                raise RuntimeError("finalized diagnostic live validation drift")
            report_binding = _binding(report_path, base)
            source_binding = dict(report["bindings"]["source_data"])
            _verify_published_report_and_source(
                base,
                report,
                rows,
                report_binding=report_binding,
                source_binding=source_binding,
            )
            completion = _completion_payload(
                base,
                report,
                report_binding=report_binding,
                source_binding=source_binding,
            )
            _atomic_text(
                completion_path,
                json.dumps(
                    completion,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
            _verify_diagnostic_completion(base, report, rows)
            final_report, final_rows = build_report(base)
            if final_report != report or final_rows != rows:
                raise RuntimeError("diagnostic completion rebuild drift")
            return report
        except BaseException:
            report_path.unlink(missing_ok=True)
            completion_path.unlink(missing_ok=True)
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the high-cutoff state/stage design pilot."
    )
    parser.parse_args(argv)
    try:
        report = write_artifacts()
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(
            json.dumps(
                {
                    "scientific_verdict": INCOMPLETE_VERDICT,
                    "qualified_claim": None,
                    "error_type": type(exc).__name__,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "analysis_sha256": report["analysis_sha256"],
                "diagnostic_count": report["diagnostic_count"],
                "scientific_verdict": report["scientific_verdict"],
                "cutoff_32_candidate_for_followup": report[
                    "cutoff_32_exploratory_candidate"
                ]["candidate_for_followup"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REPORT_PATH",
    "SOURCE_PATH",
    "STATUS",
    "build_report",
    "evaluate_diagnostics",
    "load_pilot_evidence",
    "write_artifacts",
]
