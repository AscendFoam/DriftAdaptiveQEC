"""Resumable cutoff-32/36 state-conditioned design extension.

The extension consumes the immutable cutoff-28 fresh3 evidence and reruns the
same seed positions at cutoffs 32 and 36.  It is still an unpowered design
stage: it may authorize one separately frozen powered qualification, but it
cannot itself qualify the twin or release any blocked downstream task.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np
import psutil
import scipy


TASK_ID = "T-RISK-20260728-01"
EXTERNAL_LAUNCHER_SHA256 = (
    "ca1b63693509f00bb2270776ded73e58c927c08569747f500468e05022663892"
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
    "configs/phase9/"
    "t_risk_20260728_01_cutoff32_36_design_extension_released.json"
)
PENDING_CONFIG_PATH = (
    "configs/phase9/t_risk_20260728_01_cutoff32_36_design_extension.json"
)
RELEASE_RECEIPT_PATH = (
    "docs/t_risk_20260728_01_cutoff32_36_design_extension_release_receipt.json"
)
OWNER_LOCK_PATH = (
    "runs/t_risk_20260728_01_cutoff32_36_design_extension/"
    "supervisor.owner.lock"
)
CONFIG_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-CONFIG-V1"
RELEASED_CHILD_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-RELEASED-CHILD-V1"
RELEASE_RECEIPT_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-RELEASE-RECEIPT-V1"
PENDING_CONFIG_BYTES = 15879
PENDING_CONFIG_SHA256 = (
    "1f47a4fc8a12e823967a146fdd55eef6254f9397196218fe717feb685f89ecdd"
)
RELEASED_CHILD_BYTES = 4907
RELEASED_CHILD_SHA256 = (
    "486d75a15f629de7aac63221c4cbdcc485b90e1e5bb51cb39a69f8d5e30cd400"
)
RELEASED_CHILD_ANALYSIS_SHA256 = (
    "924d67a7286c0499cb46552e3c9b8ec4036d4f3aac02281222f2e80695a2bf23"
)
RELEASE_RECEIPT_BYTES = 4511
RELEASE_RECEIPT_SHA256 = (
    "310aca1bf9bd9fdec55b5844870c74b3cedd9de49fca0e9bb02fd3466101066a"
)
RELEASE_RECEIPT_ANALYSIS_SHA256 = (
    "e5bc5b41ed1651e5197622074f7fb88cf64f591e6e96781f49dc628b3068b2da"
)
MANIFEST_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-MANIFEST-V1"
RECEIPT_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-CHUNK-RECEIPT-V1"
RUN_IDENTITY_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-RUN-IDENTITY-V1"
LOCK_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-OWNER-LOCK-V1"
HEARTBEAT_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-HEARTBEAT-V1"
PREFLIGHT_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-EXTENSION-PREFLIGHT-V1"
HARDENED_CONFIRMATION_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-CONFIRMATION-V2"
HARDENED_CONFIRMATION_ANALYSIS_SHA256 = (
    "5a798e45c0306d4bf591c971e52c68e4faf0dce276eafc74cb69d66ef6abe5a5"
)
HARDENED_CONFIRMATION_PASS = "PASS_PAIRED_CLUSTER_UQ_HARDENED_CONFIRMATION"
HARDENED_CONFIRMATION_SOURCE_TASK_ID = "T-RISK-20260727-01"
DENSITY_UQ_REPORT_SCHEMA = "PHASE9-CUTOFF32-36-DENSITY-UQ-PREFLIGHT-V1"
SCALAR_UQ_REPORT_SCHEMA = "PHASE9-SCALAR-UQ-THREE-SPLIT-CALIBRATION-REPORT-V2"
SCALAR_UQ_VERIFICATION_SCHEMA = (
    "PHASE9-SCALAR-UQ-THREE-SPLIT-INDEPENDENT-VERIFIER-V1"
)
SCALAR_UQ_SOURCE_TASK_ID = "T-RISK-20260728-02"
SCALAR_UQ_PASS = "PASS_SCALAR_UQ_THREE_SPLIT_CALIBRATION"
SCALAR_UQ_VERIFIED_PASS = "VERIFIED_PASS_SCALAR_UQ_THREE_SPLIT_CALIBRATION"
SCALAR_UQ_QUALIFIED_CLAIM = (
    "COVERAGE_CALIBRATED_SCALAR_UQ_FACTOR_FOR_T_RISK_20260728_01"
)
DENSITY_UQ_CLAIM_BOUNDARY = {
    "density_uq_preflight_only": True,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}
SCALAR_UQ_CLAIM_BOUNDARY = {
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
REFERENCE_TASK_ID = "T-RISK-20260727-01"
REFERENCE_MANIFEST = {
    "path": "docs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3_manifest.json",
    "bytes": 85625,
    "sha256": "37e7606d2d8ad1d06b46dfec9b21580a2f325eae7a0b552a4109dd85f5dfef91",
    "analysis_sha256": "8e48faad8cc5f94204a52e9f96c8c42850f7f8e0f937dfb8d188d925f91c4904",
}
REFERENCE_DIAGNOSTIC = {
    "path": "docs/t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3.json",
    "bytes": 190977,
    "sha256": "7f5e542e8734249bed5580ba7e648200965c59f64c609d9599325755fea127fb",
    "analysis_sha256": "c4ddfba3b060775000521dfd4a9c233ec3958acd12df36462dbfa3aef8b9a0ac",
}
REFERENCE_DIAGNOSTIC_SOURCE = {
    "path": (
        "docs/"
        "t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3_source_data.csv"
    ),
    "bytes": 1161786,
    "sha256": "69cea45f7439bb7efeab6beb63f24b2944bc558838cf929895f38f338f32fee0",
}
REFERENCE_DIAGNOSTIC_COMPLETION = {
    "path": (
        "docs/"
        "t_risk_20260727_01_high_cutoff_design_diagnostic_fresh3_completion.json"
    ),
    "bytes": 1423,
    "sha256": "89aed9983dd2a1978030f9da64b86bc773b662464667c63105a7da2c18639985",
    "analysis_sha256": "b2e3232434ddbb6f749e47aa6ecbcc340beac5c4beeb7a95fcd9c7272b85d1a1",
}
NARROW_AUTHORIZATION_STATE = "NARROW_UNPOWERED_CUTOFF32_36_DESIGN_ONLY"
NARROW_SCOPE = {
    "purpose": "unpowered cutoff32/36 convergence design extension",
    "synthetic_coverage_only": True,
    "physical_coverage_guarantee": None,
    "n12_role": "localization_only",
    "negative_result_interpretation": "inconclusive",
    "equivalence_conclusion": None,
    "formal_cutoff_selection": "pre_frozen_36_if_and_only_if_design_gate_passes",
    "formal_sample_count_selection": "pre_frozen_384_clusters_per_state",
    "downstream_release": False,
    "allowed_diagnostic_verdicts": [
        "DESIGN_GATE_PASS_AUTHORIZES_POWERED_FORMAL",
        "NO_GO_HIGH_CUTOFF_DESIGN",
        "INCOMPLETE",
    ],
    "required_followup": [
        "powered_confirmation_only_after_design_pass",
        "physics_repair_after_design_no_go",
    ],
}
STATUS = "CUTOFF32_36_DESIGN_EXTENSION_RAW_EVIDENCE_COMPLETE"
REJECTED_STATUS = "CUTOFF32_36_DESIGN_EXTENSION_RAW_EVIDENCE_REJECTED"
CLAIM_BOUNDARY = {
    "design_extension_only": True,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}
VERIFIED_LOADER_CONTRACT = "PHASE9-VERIFIED-SOURCE-BYTES-LOADER-V1"
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
_PILOT_SOURCE_SHA256_AT_IMPORT = sha256(Path(__file__).read_bytes()).hexdigest()
_EXECUTION_MODULE_NAMES = {
    "backend_a": "physics.phase9_backend_a",
    "backend_b": "physics.phase9_backend_b",
    "high_cutoff_adapter": "physics.phase9_high_cutoff_runtime_adapter",
    "backend_b_bridge": "physics.phase9_backend_b_logical_bridge",
    "dual_backend_kernel": ("cnn_fpga.benchmark.phase9_dual_backend_qualification"),
    "fresh_runner": "cnn_fpga.benchmark.phase9_fresh_twin_qualification",
    "iq_reference": "physics.phase9_iq_likelihood_reference",
    "twin_contract": "physics.phase9_twin_contract",
}
runner: Any = None
_VERIFIED_EXECUTION_BINDINGS: dict[str, dict[str, object]] = {}
_VERIFIED_EXECUTION_MODULES: dict[str, object] = {}
_HIGH_CUTOFF_ADAPTER_RECEIPT: dict[str, Any] = {}


class _VerifiedBytesLoader(importlib.abc.Loader):
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
        loader = _VerifiedBytesLoader(fullname, source_path, payload, digest)
        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=str(source_path),
        )


def _drop_preloaded_execution_modules() -> None:
    for module_name in reversed(tuple(_EXECUTION_MODULE_NAMES.values())):
        module = sys.modules.pop(module_name, None)
        if module is None or "." not in module_name:
            continue
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, attribute, None) is module:
            delattr(parent, attribute)


def _activate_verified_execution_modules(
    root: Path,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    global runner
    global _VERIFIED_EXECUTION_BINDINGS
    global _VERIFIED_EXECUTION_MODULES
    global _HIGH_CUTOFF_ADAPTER_RECEIPT

    expected: dict[str, dict[str, object]] = {}
    frozen: dict[str, tuple[Path, bytes, str]] = {}
    for name, module_name in _EXECUTION_MODULE_NAMES.items():
        binding = snapshot.get(f"source/{name}")
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"verified execution source is absent: {name}")
        source_path, payload = _read_bound_bytes(root, binding)
        digest = sha256(payload).hexdigest()
        expected[name] = {
            "absolute_path": str(source_path),
            "relative_path": str(binding["path"]),
            "sha256": digest,
        }
        frozen[module_name] = (source_path, payload, digest)

    if _VERIFIED_EXECUTION_BINDINGS:
        if _VERIFIED_EXECUTION_BINDINGS != expected:
            raise RuntimeError("verified execution source set drift")
        _assert_verified_execution_modules(snapshot)
        runner = _VERIFIED_EXECUTION_MODULES["fresh_runner"]
        return

    _drop_preloaded_execution_modules()
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
                raise RuntimeError(
                    f"execution module was not loaded from verified bytes: {name}"
                )
            loaded[name] = module
            if name == "high_cutoff_adapter":
                if set(loaded) != {
                    "backend_a",
                    "backend_b",
                    "high_cutoff_adapter",
                }:
                    raise RuntimeError(
                        "high-cutoff adapter activation order drift"
                    )
                _HIGH_CUTOFF_ADAPTER_RECEIPT = (
                    module.enable_verified_high_cutoff(
                        loaded["backend_a"],
                        loaded["backend_b"],
                    )
                )
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
    _VERIFIED_EXECUTION_BINDINGS = expected
    _VERIFIED_EXECUTION_MODULES = loaded
    runner = loaded["fresh_runner"]
    _assert_verified_execution_modules(snapshot)


def _assert_verified_execution_modules(
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(_VERIFIED_EXECUTION_BINDINGS) != set(_EXECUTION_MODULE_NAMES):
        raise RuntimeError("verified execution module set is incomplete")
    if set(_VERIFIED_EXECUTION_MODULES) != set(_EXECUTION_MODULE_NAMES):
        raise RuntimeError("verified execution module object set is incomplete")
    for name, module_name in _EXECUTION_MODULE_NAMES.items():
        expected = _VERIFIED_EXECUTION_BINDINGS[name]
        binding = snapshot.get(f"source/{name}")
        module = _VERIFIED_EXECUTION_MODULES[name]
        if (
            not isinstance(binding, Mapping)
            or binding.get("sha256") != expected["sha256"]
            or binding.get("path") != expected["relative_path"]
            or sys.modules.get(module_name) is not module
            or getattr(module, "__verified_source_sha256__", None) != expected["sha256"]
        ):
            raise RuntimeError(f"verified execution module attestation drift: {name}")
    adapter = _VERIFIED_EXECUTION_MODULES["high_cutoff_adapter"]
    adapter.assert_verified_high_cutoff(
        _VERIFIED_EXECUTION_MODULES["backend_a"],
        _VERIFIED_EXECUTION_MODULES["backend_b"],
        _HIGH_CUTOFF_ADAPTER_RECEIPT,
    )


class ReleasedPilotConfig(dict[str, Any]):
    """Materialized pending parent with out-of-band immutable release lineage."""

    def __init__(
        self,
        value: Mapping[str, Any],
        *,
        release_lineage: Mapping[str, Any],
    ) -> None:
        super().__init__(value)
        self.release_lineage = json.loads(json.dumps(release_lineage))


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


def _pilot_source_sha256() -> str:
    live = sha256(Path(__file__).read_bytes()).hexdigest()
    if live != _PILOT_SOURCE_SHA256_AT_IMPORT:
        raise RuntimeError("pilot source changed after module import")
    return live


def _require_verified_self_import(*, expected_mode: str = "pilot") -> None:
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
        != _PILOT_SOURCE_SHA256_AT_IMPORT
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
        or launch_meta_payload.get("mode") != expected_mode
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
            "pilot must be imported by the preregistered trusted-operator bootstrap"
        )


def _release_lineage(config: Mapping[str, Any]) -> dict[str, Any]:
    lineage = getattr(config, "release_lineage", None)
    if not isinstance(lineage, Mapping):
        raise RuntimeError("released pilot lineage is absent")
    return json.loads(json.dumps(lineage))


def _leaf_differences(
    left: object,
    right: object,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                differences.add((*prefix, str(key)))
            else:
                differences.update(
                    _leaf_differences(left[key], right[key], (*prefix, str(key)))
                )
        return differences
    return set() if left == right else {prefix}


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


def _read_bound_bytes(
    root: Path,
    binding: Mapping[str, Any],
    *,
    expected_path: str | None = None,
) -> tuple[Path, bytes]:
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
    payload = path.read_bytes()
    if (
        len(payload) != binding["bytes"]
        or sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise RuntimeError(f"artifact byte binding drift: {relative}")
    return path, payload


def _read_bound_json(
    root: Path,
    binding: Mapping[str, Any],
    *,
    expected_path: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    path, payload = _read_bound_bytes(
        root,
        binding,
        expected_path=expected_path,
    )
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError(f"bound JSON is not an object: {path}")
    return path, value


def _bound_part(value: Mapping[str, Any]) -> dict[str, object]:
    return {
        "path": value["path"],
        "bytes": value["bytes"],
        "sha256": value["sha256"],
    }


def _reference_input_bindings(
    root: Path,
    pilot: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    contract = pilot.get("reference_cutoff_28_evidence")
    expected = {
        "manifest": REFERENCE_MANIFEST,
        "diagnostic": REFERENCE_DIAGNOSTIC,
        "diagnostic_source_data": REFERENCE_DIAGNOSTIC_SOURCE,
        "diagnostic_completion": REFERENCE_DIAGNOSTIC_COMPLETION,
    }
    if not isinstance(contract, Mapping) or set(contract) != set(expected):
        raise RuntimeError("cutoff-28 reference contract drift")
    for name, frozen in expected.items():
        if contract.get(name) != frozen:
            raise RuntimeError(f"cutoff-28 reference binding drift: {name}")

    bindings = {
        name: _bound_part(value)
        for name, value in expected.items()
    }
    _, manifest = _read_bound_json(
        root,
        bindings["manifest"],
        expected_path=str(REFERENCE_MANIFEST["path"]),
    )
    _, diagnostic = _read_bound_json(
        root,
        bindings["diagnostic"],
        expected_path=str(REFERENCE_DIAGNOSTIC["path"]),
    )
    _read_bound_bytes(
        root,
        bindings["diagnostic_source_data"],
        expected_path=str(REFERENCE_DIAGNOSTIC_SOURCE["path"]),
    )
    _, completion = _read_bound_json(
        root,
        bindings["diagnostic_completion"],
        expected_path=str(REFERENCE_DIAGNOSTIC_COMPLETION["path"]),
    )
    if (
        _self_hash(manifest) != REFERENCE_MANIFEST["analysis_sha256"]
        or manifest.get("task_id") != REFERENCE_TASK_ID
        or manifest.get("status") != "DESIGN_PILOT_RAW_EVIDENCE_COMPLETE"
        or manifest.get("observed_cells") != 32
        or manifest.get("observed_rows") != 27648
        or manifest.get("exception_rows") != 0
        or manifest.get("conservation_failure_rows") != 0
        or manifest.get("qualified_claim") is not None
        or set(manifest.get("claim_state", {}).values())
        != {True, None}
    ):
        raise RuntimeError("cutoff-28 reference manifest semantic drift")
    candidate = diagnostic.get("cutoff_32_exploratory_candidate", {})
    if (
        _self_hash(diagnostic) != REFERENCE_DIAGNOSTIC["analysis_sha256"]
        or diagnostic.get("task_id") != REFERENCE_TASK_ID
        or diagnostic.get("scientific_verdict") != "EXPLORATORY_RISK_SIGNAL"
        or diagnostic.get("qualified_claim") is not None
        or candidate.get("candidate_for_followup") is not True
        or candidate.get("density_threshold") != 0.075
        or float(candidate.get("observed_maximum_density_point", 0.0))
        != 0.09134328001208851
        or len(candidate.get("tail_trigger_gate_ids", [])) != 6
    ):
        raise RuntimeError("cutoff-28 reference diagnostic semantic drift")
    if (
        _self_hash(completion)
        != REFERENCE_DIAGNOSTIC_COMPLETION["analysis_sha256"]
        or completion.get("task_id") != REFERENCE_TASK_ID
        or completion.get("status") != "COMPLETE"
        or completion.get("scientific_verdict") != "EXPLORATORY_RISK_SIGNAL"
        or completion.get("qualified_claim") is not None
        or completion.get("pilot_manifest") != bindings["manifest"]
        or completion.get("report") != bindings["diagnostic"]
        or completion.get("source_data")
        != bindings["diagnostic_source_data"]
    ):
        raise RuntimeError("cutoff-28 diagnostic completion semantic drift")

    receipts = manifest.get("chunk_receipts")
    receipt_bindings = manifest.get("receipt_bindings")
    if (
        not isinstance(receipts, list)
        or not isinstance(receipt_bindings, list)
        or len(receipts) != 32
        or len(receipt_bindings) != 32
    ):
        raise RuntimeError("cutoff-28 reference receipt ledger drift")
    cutoff_28_count = 0
    for index, (receipt, receipt_binding) in enumerate(
        zip(receipts, receipt_bindings, strict=True)
    ):
        if not isinstance(receipt, Mapping) or not isinstance(
            receipt_binding, Mapping
        ):
            raise RuntimeError("cutoff-28 reference receipt type drift")
        _, live_receipt = _read_bound_json(root, receipt_binding)
        if live_receipt != receipt:
            raise RuntimeError("cutoff-28 reference live receipt drift")
        if int(receipt.get("cell", {}).get("cutoff", -1)) != 28:
            continue
        if (
            _self_hash(receipt) != receipt.get("analysis_sha256")
            or receipt.get("task_id") != REFERENCE_TASK_ID
            or receipt.get("exception_rows") != 0
            or receipt.get("observed_rows") != 864
            or receipt.get("expected_rows") != 864
        ):
            raise RuntimeError("cutoff-28 reference receipt semantic drift")
        cutoff_28_count += 1
        bindings[f"cutoff28_receipt_{cutoff_28_count:02d}"] = dict(
            receipt_binding
        )
        for kind in ("csv", "npz"):
            chunk_binding = receipt.get(kind)
            if not isinstance(chunk_binding, Mapping):
                raise RuntimeError("cutoff-28 reference chunk binding missing")
            _read_bound_bytes(root, chunk_binding)
            bindings[
                f"cutoff28_{kind}_{cutoff_28_count:02d}"
            ] = dict(chunk_binding)
    if cutoff_28_count != 8:
        raise RuntimeError("cutoff-28 reference cell coverage drift")
    return bindings


def _build_input_snapshot(
    root: Path,
    pilot: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    lineage = _release_lineage(pilot)
    expected: dict[str, dict[str, object]] = {
        "released_child": {
            "path": CONFIG_PATH,
            "bytes": RELEASED_CHILD_BYTES,
            "sha256": RELEASED_CHILD_SHA256,
        },
        "pending_parent": dict(lineage["pending_parent"]),
        "release_receipt": dict(lineage["release_receipt"]),
        "hardened_confirmation_report": dict(lineage["hardened_confirmation_report"]),
        "hardened_confirmation_source_data": dict(
            lineage["hardened_confirmation_source_data"]
        ),
        "base_config": _binding(root / str(pilot["base_config"]["path"]), root),
        "pilot_source": _binding(Path(__file__).resolve(), root),
    }
    module = sys.modules.get(__name__)
    if module is None:
        raise RuntimeError("verified pilot module identity missing")
    for snapshot_name, attribute in (
        (
            "verified_bootstrap_source",
            "__verified_bootstrap_source_binding__",
        ),
        (
            "verified_external_launch_meta",
            "__verified_launch_meta_binding__",
        ),
    ):
        binding = getattr(module, attribute, None)
        if not isinstance(binding, Mapping) or set(binding) != {
            "path",
            "bytes",
            "sha256",
        }:
            raise RuntimeError(f"{snapshot_name} binding missing or malformed")
        expected[snapshot_name] = dict(binding)
    launch_binding = expected["verified_external_launch_meta"]
    launch_payload = (
        None
        if module is None
        else getattr(module, "__verified_launch_meta_payload__", None)
    )
    _, disk_launch_payload = _read_bound_json(root, launch_binding)
    if not isinstance(launch_payload, Mapping) or disk_launch_payload != dict(
        launch_payload
    ):
        raise RuntimeError("verified external launch meta content drift")
    if expected["base_config"]["sha256"] != pilot["base_config"]["sha256"]:
        raise RuntimeError("pilot base config changed before input snapshot")
    if expected["pilot_source"]["sha256"] != _pilot_source_sha256():
        raise RuntimeError("pilot source snapshot drift")
    for name, pinned in pilot["source_bindings"].items():
        live = _binding(root / str(pinned["path"]), root)
        if live["sha256"] != pinned["sha256"]:
            raise RuntimeError(f"pilot source changed before snapshot: {name}")
        expected[f"source/{name}"] = live
    for name, binding in _reference_input_bindings(root, pilot).items():
        expected[f"reference/{name}"] = binding
    for name, binding in _validate_uq_preflight_sources(root, pilot).items():
        expected[f"uq/{name}"] = binding
    _assert_input_snapshot(root, expected)
    return expected


def _assert_input_snapshot(
    root: Path,
    snapshot: Mapping[str, Mapping[str, Any]],
) -> None:
    if not snapshot or len(snapshot) != len(set(snapshot)):
        raise RuntimeError("pilot input snapshot is empty or duplicated")
    for name, binding in snapshot.items():
        if not isinstance(name, str) or not isinstance(binding, Mapping):
            raise RuntimeError("pilot input snapshot schema drift")
        _read_bound_bytes(root, binding)
    pilot_source = snapshot.get("pilot_source")
    if (
        not isinstance(pilot_source, Mapping)
        or pilot_source.get("sha256") != _pilot_source_sha256()
    ):
        raise RuntimeError("pilot input snapshot source-at-import drift")
    if _VERIFIED_EXECUTION_BINDINGS:
        _assert_verified_execution_modules(snapshot)


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


def _uq_release_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project the fully pinned UQ parents into the release envelope."""

    contract = config.get("uq_preflight_sources")
    if not isinstance(contract, Mapping):
        raise RuntimeError("UQ release evidence contract missing")
    density = contract.get("density")
    scalar = contract.get("scalar")
    if not isinstance(density, Mapping) or not isinstance(scalar, Mapping):
        raise RuntimeError("UQ release evidence lane missing")
    scalar_sources = scalar.get("source_data")
    if not isinstance(scalar_sources, Mapping):
        raise RuntimeError("scalar UQ release source ledger missing")
    return {
        "density": {
            "report": dict(density["report"]),
            "source_data": dict(density["source_data"]),
            "analysis_sha256": density["required_analysis_sha256"],
            "verdict": density["required_verdict"],
            "source_pass_is_authorization": False,
        },
        "scalar": {
            "report": dict(scalar["report"]),
            "independent_verification": dict(
                scalar["independent_verification"]
            ),
            "selection_a_source_data": dict(scalar_sources["selection_a"]),
            "selection_b_source_data": dict(scalar_sources["selection_b"]),
            "confirmation_source_data": dict(scalar_sources["confirmation"]),
            "analysis_sha256": scalar["required_analysis_sha256"],
            "verdict": scalar["required_verdict"],
            "verification_analysis_sha256": scalar[
                "required_verification_analysis_sha256"
            ],
            "verification_verdict": scalar[
                "required_verification_verdict"
            ],
            "qualified_claim": scalar["required_qualified_claim"],
            "source_pass_is_authorization": False,
        },
    }


def _validate_release_receipt(
    receipt: Mapping[str, Any],
    *,
    pending_binding: Mapping[str, Any],
    report_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    uq_evidence: Mapping[str, Any],
) -> None:
    if _self_hash(receipt) != RELEASE_RECEIPT_ANALYSIS_SHA256:
        raise RuntimeError("high-cutoff release receipt analysis anchor drift")
    if (
        set(receipt)
        != {
            "task_id",
            "schema_version",
            "authorization_state",
            "pending_parent",
            "hardened_confirmation",
            "uq_preflights",
            "narrow_scope",
            "qualified_claim",
            "claim_state",
            "analysis_sha256",
        }
        or receipt.get("task_id") != TASK_ID
        or receipt.get("schema_version") != RELEASE_RECEIPT_SCHEMA
        or receipt.get("authorization_state") != NARROW_AUTHORIZATION_STATE
        or receipt.get("pending_parent") != pending_binding
        or receipt.get("uq_preflights") != uq_evidence
        or receipt.get("narrow_scope") != NARROW_SCOPE
        or receipt.get("qualified_claim") is not None
        or receipt.get("claim_state") != CLAIM_BOUNDARY
    ):
        raise RuntimeError("high-cutoff release receipt semantic drift")
    hardened = receipt.get("hardened_confirmation")
    if (
        not isinstance(hardened, Mapping)
        or set(hardened)
        != {
            "report",
            "source_data",
            "analysis_sha256",
            "source_report_verdict",
            "source_pass_is_authorization",
        }
        or hardened.get("report") != report_binding
        or hardened.get("source_data") != source_binding
        or hardened.get("analysis_sha256") != HARDENED_CONFIRMATION_ANALYSIS_SHA256
        or hardened.get("source_report_verdict") != HARDENED_CONFIRMATION_PASS
        or hardened.get("source_pass_is_authorization") is not False
    ):
        raise RuntimeError("high-cutoff release receipt evidence drift")


def _materialize_released_parent(
    root: Path,
    child: Mapping[str, Any],
) -> ReleasedPilotConfig:
    if _self_hash(child) != RELEASED_CHILD_ANALYSIS_SHA256:
        raise RuntimeError("released high-cutoff child analysis anchor drift")
    expected_child_keys = {
        "task_id",
        "schema_version",
        "purpose",
        "authorization_state",
        "pending_parent",
        "release_receipt",
        "hardened_confirmation",
        "uq_preflights",
        "narrow_scope",
        "qualified_claim",
        "claim_state",
        "analysis_sha256",
    }
    if (
        set(child) != expected_child_keys
        or child.get("task_id") != TASK_ID
        or child.get("schema_version") != RELEASED_CHILD_SCHEMA
        or child.get("authorization_state") != NARROW_AUTHORIZATION_STATE
        or child.get("narrow_scope") != NARROW_SCOPE
        or child.get("qualified_claim") is not None
        or child.get("claim_state") != CLAIM_BOUNDARY
    ):
        raise RuntimeError("released high-cutoff child semantic drift")

    pending_binding = child.get("pending_parent")
    receipt_binding = child.get("release_receipt")
    hardened = child.get("hardened_confirmation")
    child_uq_evidence = child.get("uq_preflights")
    if (
        not isinstance(pending_binding, Mapping)
        or not isinstance(receipt_binding, Mapping)
        or not isinstance(hardened, Mapping)
        or not isinstance(child_uq_evidence, Mapping)
        or set(hardened)
        != {
            "report",
            "source_data",
            "analysis_sha256",
            "source_report_verdict",
            "source_pass_is_authorization",
        }
        or not isinstance(hardened.get("report"), Mapping)
        or not isinstance(hardened.get("source_data"), Mapping)
    ):
        raise RuntimeError("released high-cutoff child binding schema drift")
    if dict(pending_binding) != {
        "path": PENDING_CONFIG_PATH,
        "bytes": PENDING_CONFIG_BYTES,
        "sha256": PENDING_CONFIG_SHA256,
    }:
        raise RuntimeError("pending high-cutoff parent immutable binding drift")
    if dict(receipt_binding) != {
        "path": RELEASE_RECEIPT_PATH,
        "bytes": RELEASE_RECEIPT_BYTES,
        "sha256": RELEASE_RECEIPT_SHA256,
    }:
        raise RuntimeError("high-cutoff release receipt immutable binding drift")
    pending_path = _require_binding(
        root, pending_binding, expected_path=PENDING_CONFIG_PATH
    )
    receipt_path, receipt = _read_bound_json(
        root, receipt_binding, expected_path=RELEASE_RECEIPT_PATH
    )
    report_binding = hardened["report"]
    source_binding = hardened["source_data"]
    report_path, report = _read_bound_json(
        root,
        report_binding,
        expected_path="docs/t_risk_20260727_01_uq_hardened_confirmation.json",
    )
    _require_binding(
        root,
        source_binding,
        expected_path=(
            "docs/t_risk_20260727_01_uq_hardened_confirmation_source_data.csv"
        ),
    )
    if (
        hardened.get("analysis_sha256") != HARDENED_CONFIRMATION_ANALYSIS_SHA256
        or hardened.get("source_report_verdict") != HARDENED_CONFIRMATION_PASS
        or hardened.get("source_pass_is_authorization") is not False
    ):
        raise RuntimeError("released high-cutoff child PASS evidence drift")

    _validate_release_receipt(
        receipt,
        pending_binding=pending_binding,
        report_binding=report_binding,
        source_binding=source_binding,
        uq_evidence=child_uq_evidence,
    )
    _self_hash(report)
    if (
        report.get("analysis_sha256") != HARDENED_CONFIRMATION_ANALYSIS_SHA256
        or report.get("verdict") != HARDENED_CONFIRMATION_PASS
        or report.get("qualified_claim") is not None
        or report.get("claim_state") != HARDENED_CONFIRMATION_CLAIM_BOUNDARY
        or report.get("formal_outcomes_accessed") is not False
    ):
        raise RuntimeError("released high-cutoff hardened PASS semantics drift")

    _, pending = _read_bound_json(
        root,
        pending_binding,
        expected_path=PENDING_CONFIG_PATH,
    )
    pins = pending.get("hardened_confirmation_source")
    if (
        not isinstance(pins, Mapping)
        or dict(pins.get("report", {})) != dict(report_binding)
        or dict(pins.get("source_data", {})) != dict(source_binding)
        or pins.get("required_analysis_sha256")
        != HARDENED_CONFIRMATION_ANALYSIS_SHA256
    ):
        raise RuntimeError("pending high-cutoff hardened pins drift")
    expected_uq_evidence = _uq_release_evidence(pending)
    if child_uq_evidence != expected_uq_evidence:
        raise RuntimeError("released high-cutoff UQ evidence drift")
    if _leaf_differences(pending, pending):
        raise RuntimeError("released child changed fully pinned parent")

    lineage = {
        "released_child": _binding(root / CONFIG_PATH, root),
        "pending_parent": dict(pending_binding),
        "release_receipt": dict(receipt_binding),
        "release_receipt_analysis_sha256": receipt["analysis_sha256"],
        "hardened_confirmation_report": dict(report_binding),
        "hardened_confirmation_source_data": dict(source_binding),
        "hardened_confirmation_analysis_sha256": (
            HARDENED_CONFIRMATION_ANALYSIS_SHA256
        ),
        "uq_preflights": json.loads(json.dumps(expected_uq_evidence)),
        "authorization_state": NARROW_AUTHORIZATION_STATE,
        "narrow_scope": dict(NARROW_SCOPE),
    }
    return ReleasedPilotConfig(pending, release_lineage=lineage)


def _validate_hardened_confirmation(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    contract = config.get("hardened_confirmation_source")
    expected_keys = {
        "report",
        "source_data",
        "source_task_id",
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
    report_path, report = _read_bound_json(
        root,
        report_binding,
        expected_path=str(report_binding.get("path")),
    )
    source_path, source_bytes = _read_bound_bytes(
        root,
        source_binding,
        expected_path=str(source_binding.get("path")),
    )
    _self_hash(report)
    if (
        contract.get("source_task_id") != HARDENED_CONFIRMATION_SOURCE_TASK_ID
        or report.get("task_id") != HARDENED_CONFIRMATION_SOURCE_TASK_ID
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
        or report_source_binding.get("sha256") != sha256(source_bytes).hexdigest()
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


def _validate_uq_preflight_sources(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Bind both UQ gates, including the independent scalar recomputation.

    The density lane is the frozen T-RISK-20260728-01 preflight.  The scalar
    lane is the later T-RISK-20260728-02 two-selection-fold plus untouched
    confirmation repair.  The failed one-split scalar diagnostic is never
    accepted as the design release.
    """

    contract = config.get("uq_preflight_sources")
    expected_top_keys = {
        "density",
        "scalar",
        "required_dimensions",
        "required_cluster_counts",
        "required_factor",
        "required_multiplier_replicates",
        "failure_effect",
    }
    if (
        not isinstance(contract, Mapping)
        or set(contract) != expected_top_keys
        or contract.get("required_dimensions") != [84, 96, 108]
        or contract.get("required_cluster_counts") != [12, 384]
        or contract.get("required_factor") != 1.0
        or contract.get("required_multiplier_replicates") != 199
        or contract.get("failure_effect") != "INCOMPLETE_UQ_NO_SCIENTIFIC_RUN"
    ):
        raise RuntimeError("cutoff32/36 UQ preflight contract drift")

    expected_density_keys = {
        "report",
        "source_data",
        "required_analysis_sha256",
        "required_verdict",
    }
    bindings: dict[str, dict[str, object]] = {}
    density_contract = contract.get("density")
    if (
        not isinstance(density_contract, Mapping)
        or set(density_contract) != expected_density_keys
    ):
        raise RuntimeError("density UQ preflight lane contract drift")
    density_analysis = density_contract.get("required_analysis_sha256")
    if (
        not isinstance(density_analysis, str)
        or len(density_analysis) != 64
        or any(
            character not in "0123456789abcdef"
            for character in density_analysis
        )
    ):
        raise RuntimeError("density UQ preflight is pending and unreleased")
    density_report_binding = density_contract.get("report")
    density_source_binding = density_contract.get("source_data")
    if not isinstance(density_report_binding, Mapping) or not isinstance(
        density_source_binding, Mapping
    ):
        raise RuntimeError("density UQ preflight binding missing")
    _, density = _read_bound_json(
        root,
        density_report_binding,
        expected_path=str(density_report_binding.get("path")),
    )
    _, density_source_payload = _read_bound_bytes(
        root,
        density_source_binding,
        expected_path=str(density_source_binding.get("path")),
    )
    if (
        _self_hash(density) != density.get("analysis_sha256")
        or density.get("analysis_sha256") != density_analysis
        or density.get("task_id") != TASK_ID
        or density.get("verdict") != density_contract["required_verdict"]
        or density.get("qualified_claim") is not None
    ):
        raise RuntimeError("density UQ preflight PASS identity drift")
    density_report_source = density.get("bindings", {}).get("source_data")
    if (
        not isinstance(density_report_source, Mapping)
        or dict(density_report_source) != dict(density_source_binding)
        or density_report_source.get("sha256")
        != sha256(density_source_payload).hexdigest()
    ):
        raise RuntimeError("density UQ source-data binding drift")
    density_report_bindings = density.get("bindings")
    if not isinstance(density_report_bindings, Mapping) or not density_report_bindings:
        raise RuntimeError("density UQ report binding ledger missing")
    for name, binding in density_report_bindings.items():
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"density UQ binding type drift: {name}")
        live = _binding(root / str(binding.get("path")), root)
        if dict(binding) != live:
            raise RuntimeError(f"density UQ live binding drift: {name}")

    density_domain = density.get("domain", {})
    if (
        density.get("schema_version") != DENSITY_UQ_REPORT_SCHEMA
        or density.get("frozen_parent_calibration_factor")
        != contract["required_factor"]
        or density.get("selected_formal_clusters_per_state") != 384
        or density.get("pilot_domain_factor_coverage_calibrated") is not True
        or density.get("formal_domain_factor_coverage_calibrated") is not True
        or density.get("confirmation_power_passed") is not True
        or density.get("formal_outcomes_accessed") is not False
        or density.get("claim_state") != DENSITY_UQ_CLAIM_BOUNDARY
        or density.get("seed_firewall", {}).get("passed") is not True
        or density_domain.get("dimensions") != contract["required_dimensions"]
        or density_domain.get("cluster_counts_per_state")
        != contract["required_cluster_counts"]
        or density_domain.get("multiplier_replicates")
        != contract["required_multiplier_replicates"]
        or density_domain.get("cell_count") != 72
        or density_domain.get("record_count") != 72 * 256
        or density_source_payload.count(b"\n") != 72 * 256 + 1
    ):
        raise RuntimeError("density UQ preflight PASS semantics drift")
    bindings["density_uq_report"] = dict(density_report_binding)
    bindings["density_uq_source_data"] = dict(density_source_binding)

    scalar_contract = contract.get("scalar")
    expected_scalar_keys = {
        "report",
        "independent_verification",
        "source_data",
        "required_analysis_sha256",
        "required_verdict",
        "required_verification_analysis_sha256",
        "required_verification_verdict",
        "required_qualified_claim",
    }
    if (
        not isinstance(scalar_contract, Mapping)
        or set(scalar_contract) != expected_scalar_keys
        or scalar_contract.get("required_verdict") != SCALAR_UQ_PASS
        or scalar_contract.get("required_verification_verdict")
        != SCALAR_UQ_VERIFIED_PASS
        or scalar_contract.get("required_qualified_claim")
        != SCALAR_UQ_QUALIFIED_CLAIM
    ):
        raise RuntimeError("scalar UQ calibration lane contract drift")
    scalar_report_binding = scalar_contract.get("report")
    scalar_verification_binding = scalar_contract.get("independent_verification")
    scalar_source_bindings = scalar_contract.get("source_data")
    if (
        not isinstance(scalar_report_binding, Mapping)
        or not isinstance(scalar_verification_binding, Mapping)
        or not isinstance(scalar_source_bindings, Mapping)
        or set(scalar_source_bindings)
        != {"selection_a", "selection_b", "confirmation"}
    ):
        raise RuntimeError("scalar UQ calibration bindings missing")
    _, scalar = _read_bound_json(
        root,
        scalar_report_binding,
        expected_path=str(scalar_report_binding.get("path")),
    )
    _, scalar_verification = _read_bound_json(
        root,
        scalar_verification_binding,
        expected_path=str(scalar_verification_binding.get("path")),
    )
    scalar_source_payloads: dict[str, bytes] = {}
    for split, source_binding in scalar_source_bindings.items():
        if not isinstance(source_binding, Mapping):
            raise RuntimeError(f"scalar UQ {split} binding type drift")
        _, payload = _read_bound_bytes(
            root,
            source_binding,
            expected_path=str(source_binding.get("path")),
        )
        scalar_source_payloads[str(split)] = payload
        bindings[f"scalar_uq_{split}_source_data"] = dict(source_binding)

    required_scalar_analysis = scalar_contract["required_analysis_sha256"]
    required_verification_analysis = scalar_contract[
        "required_verification_analysis_sha256"
    ]
    if (
        _self_hash(scalar) != required_scalar_analysis
        or scalar.get("analysis_sha256") != required_scalar_analysis
        or scalar.get("task_id") != SCALAR_UQ_SOURCE_TASK_ID
        or scalar.get("schema_version") != SCALAR_UQ_REPORT_SCHEMA
        or scalar.get("verdict") != SCALAR_UQ_PASS
        or scalar.get("qualified_claim") != SCALAR_UQ_QUALIFIED_CLAIM
        or _self_hash(scalar_verification) != required_verification_analysis
        or scalar_verification.get("analysis_sha256")
        != required_verification_analysis
        or scalar_verification.get("task_id") != SCALAR_UQ_SOURCE_TASK_ID
        or scalar_verification.get("schema_version")
        != SCALAR_UQ_VERIFICATION_SCHEMA
        or scalar_verification.get("verdict") != SCALAR_UQ_VERIFIED_PASS
    ):
        raise RuntimeError("scalar UQ calibrated PASS identity drift")

    scalar_report_bindings = scalar.get("bindings")
    verifier_bindings = scalar_verification.get("bindings")
    if (
        not isinstance(scalar_report_bindings, Mapping)
        or not scalar_report_bindings
        or not isinstance(verifier_bindings, Mapping)
        or not verifier_bindings
    ):
        raise RuntimeError("scalar UQ binding ledgers missing")
    for ledger_name, ledger in (
        ("report", scalar_report_bindings),
        ("verification", verifier_bindings),
    ):
        for name, binding in ledger.items():
            if not isinstance(binding, Mapping):
                raise RuntimeError(
                    f"scalar UQ {ledger_name} binding type drift: {name}"
                )
            live = _binding(root / str(binding.get("path")), root)
            if dict(binding) != live:
                raise RuntimeError(
                    f"scalar UQ {ledger_name} live binding drift: {name}"
                )
    for split, report_key in (
        ("selection_a", "selection_a_source_data"),
        ("selection_b", "selection_b_source_data"),
        ("confirmation", "confirmation_source_data"),
    ):
        if (
            dict(scalar_report_bindings.get(report_key, {}))
            != dict(scalar_source_bindings[split])
            or dict(verifier_bindings.get(report_key, {}))
            != dict(scalar_source_bindings[split])
            or scalar_source_payloads[split].count(b"\n") != 393216 + 1
        ):
            raise RuntimeError(f"scalar UQ {split} source-data drift")

    scalar_config_binding = scalar_report_bindings.get("config")
    if not isinstance(scalar_config_binding, Mapping):
        raise RuntimeError("scalar UQ source config binding missing")
    _, scalar_source_config = _read_bound_json(
        root,
        scalar_config_binding,
        expected_path=str(scalar_config_binding.get("path")),
    )
    expected_margins = sorted(
        {
            float(value)
            for value in config["diagnostic_contract"]["margins"].values()
        }
    )
    if (
        scalar_source_config.get("margins") != expected_margins
        or scalar_source_config.get("cluster_counts")
        != contract["required_cluster_counts"]
        or scalar_source_config.get("factor_grid")
        != [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        or scalar_source_config.get("multiplier_replicates")
        != contract["required_multiplier_replicates"]
        or scalar_source_config.get("trial_count_per_cell") != 2048
        or scalar_source_config.get("design_outcomes_accessed") is not False
        or scalar_source_config.get("claim_boundary")
        != SCALAR_UQ_CLAIM_BOUNDARY
    ):
        raise RuntimeError("scalar UQ source config semantic drift")

    def _split_passes(
        split: Mapping[str, Any],
        *,
        role: str,
        evaluated_factors: list[float],
    ) -> bool:
        factor_gates = split.get("factor_gates")
        if not isinstance(factor_gates, Mapping):
            return False
        selected_gate = factor_gates.get("1.0")
        if not isinstance(selected_gate, Mapping):
            return False
        power_ledger = selected_gate.get("power_ledger")
        return bool(
            split.get("role") == role
            and split.get("cell_count") == 192
            and split.get("raw_trial_count") == 393216
            and split.get("trial_count_per_cell") == 2048
            and split.get("evaluated_factors") == evaluated_factors
            and split.get("all_workers_single_blas_thread") is True
            and selected_gate.get("factor") == 1.0
            and selected_gate.get("coverage_pass") is True
            and selected_gate.get("coverage_failed_cells") == []
            and selected_gate.get("power_pass") is True
            and selected_gate.get("global_pass") is True
            and isinstance(power_ledger, list)
            and len(power_ledger) == 4
            and all(
                isinstance(entry, Mapping)
                and entry.get("global_iut_pass") is True
                and entry.get("failed_strata") == []
                and entry.get("stratum_count") == 16
                for entry in power_ledger
            )
        )

    if (
        scalar.get("selected_factor") != contract["required_factor"]
        or scalar.get("factor_grid") != [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
        or scalar.get("expected_cells_per_split") != 192
        or scalar.get("expected_raw_trials_per_split") != 393216
        or scalar.get("trial_count_per_cell_per_split") != 2048
        or scalar.get("confirmation_evaluated_factor_count") != 1
        or scalar.get("selection_passed") is not True
        or scalar.get("confirmation_passed") is not True
        or scalar.get("design_outcomes_accessed") is not False
        or scalar.get("claim_state") != SCALAR_UQ_CLAIM_BOUNDARY
        or scalar.get(
            "diagnostic_parent_used_as_selection_or_confirmation_evidence"
        )
        is not False
        or scalar.get(
            "v1_failure_used_as_v2_selection_or_confirmation_evidence"
        )
        is not False
        or not _split_passes(
            scalar.get("selection_a", {}),
            role="factor_selection",
            evaluated_factors=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        )
        or not _split_passes(
            scalar.get("selection_b", {}),
            role="factor_selection",
            evaluated_factors=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        )
        or not _split_passes(
            scalar.get("confirmation", {}),
            role="untouched_confirmation",
            evaluated_factors=[1.0],
        )
    ):
        raise RuntimeError("scalar UQ calibrated PASS semantics drift")
    if (
        scalar_verification.get("target_analysis_sha256")
        != required_scalar_analysis
        or scalar_verification.get("target_verdict") != SCALAR_UQ_PASS
        or scalar_verification.get("selected_factor") != 1.0
        or scalar_verification.get("raw_rows_recomputed") != 3 * 393216
        or scalar_verification.get("selection_recomputed") is not True
        or scalar_verification.get("confirmation_recomputed") is not True
        or scalar_verification.get("factor_gate_rows_exact") is not True
        or scalar_verification.get("seed_rows_exact") is not True
        or scalar_verification.get("design_outcomes_accessed") is not False
        or scalar_verification.get("claim_state")
        != SCALAR_UQ_CLAIM_BOUNDARY
        or dict(verifier_bindings.get("target_report", {}))
        != dict(scalar_report_binding)
    ):
        raise RuntimeError("scalar UQ independent verification semantics drift")
    bindings["scalar_uq_report"] = dict(scalar_report_binding)
    bindings["scalar_uq_independent_verification"] = dict(
        scalar_verification_binding
    )
    return bindings


def load_pilot_config(
    root: Path, *, require_hardened: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / CONFIG_PATH
    child_binding = {
        "path": CONFIG_PATH,
        "bytes": RELEASED_CHILD_BYTES,
        "sha256": RELEASED_CHILD_SHA256,
    }
    _, child = _read_bound_json(root, child_binding, expected_path=CONFIG_PATH)
    config = _materialize_released_parent(root, child)
    if (
        config.get("task_id") != TASK_ID
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise ValueError("cutoff32/36 extension identity/claim firewall invalid")
    if (
        config.get("cutoffs") != [32, 36]
        or config.get("reference_cutoff") != 28
        or int(config.get("trajectory_count", 0))
        != 6 * int(config.get("clusters_per_state", -1))
        or int(config.get("clusters_per_state", 0)) != 12
        or config.get("shared_repair_cells")
        != [
            {
                "initial_state": "vacuum_f",
                "action": "RESET",
                "cutoffs": [28, 32, 36],
                "backends": ["A", "B"],
                "reason": (
                    "replace the three preregistered shared-vacuum_f RESET "
                    "density failures without deleting or rewriting them"
                ),
            }
        ]
        or sorted(config.get("scenario_names", []))
        != ["burst", "compound", "step", "telegraph"]
    ):
        raise ValueError("cutoff32/36 extension matrix drift")
    diagnostic = config.get("diagnostic_contract", {})
    if (
        diagnostic.get("confidence") != 0.95
        or diagnostic.get("multiplier_replicates") != 199
        or diagnostic.get("multiplier_seed_namespace") != 1480000
        or diagnostic.get("absolute_truncation_diagnostics_are_localization_only")
        is not True
        or diagnostic.get("formal_rescue_forbidden") is not True
        or diagnostic.get("design_density_point_threshold") != 0.075
        or diagnostic.get("required_consecutive_increments")
        != [[28, 32], [32, 36]]
        or diagnostic.get("expected_gate_count") != 1454
        or diagnostic.get("gate_accounting")
        != {
            "fault_density": 96,
            "fault_scalar": 1080,
            "fault_absolute_tail": 240,
            "shared_density": 7,
            "shared_scalar": 21,
            "shared_absolute_tail": 10,
        }
        or diagnostic.get("decision_statistic")
        != (
            "conservative unpowered point estimate including archived density "
            "quantization bound; no confidence or equivalence claim"
        )
        or diagnostic.get("design_pass_authorizes_only_powered_formal")
        is not True
    ):
        raise ValueError("cutoff32/36 extension diagnostic contract drift")
    expected_tail_margins = {
        "absolute_terminal_top1_fock_mass": 0.005,
        "absolute_terminal_top2_fock_mass": 0.01,
        "absolute_terminal_top4_fock_mass": 0.02,
        "absolute_terminal_normalized_mean_photon": 0.25,
        "absolute_terminal_commutator_defect": 0.05,
    }
    margins = diagnostic.get("margins")
    if not isinstance(margins, Mapping) or any(
        margins.get(key) != value for key, value in expected_tail_margins.items()
    ):
        raise ValueError("cutoff32/36 absolute truncation contract drift")
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
            raise ValueError(f"cutoff32/36 stage partition drift: {scenario}")
    splits = config["seed_splits"]
    intervals = []
    for key in (
        "round_backend_a",
        "round_backend_b",
        "trajectory_backend_a",
        "trajectory_backend_b",
        "heldout_common",
    ):
        start = int(splits[key]["start"])
        count = int(splits[key]["count"])
        intervals.append(set(range(start, start + count)))
    if (
        splits.get("all_intervals_disjoint") is not True
        or splits.get("disjoint_from_20260726_formal") is not True
        or splits.get(
            "fault_trajectory_intervals_identical_to_fresh3_for_cutoff_pairing"
        )
        is not True
        or splits.get("shared_repair_round_intervals_are_fresh") is not True
        or splits["round_backend_a"] != {"start": 1433000, "count": 72}
        or splits["round_backend_b"] != {"start": 1434000, "count": 72}
        or splits["trajectory_backend_a"] != {"start": 1430000, "count": 72}
        or splits["trajectory_backend_b"] != {"start": 1431000, "count": 72}
        or splits["heldout_common"] != {"start": 1432000, "count": 72}
        or any(
            intervals[i] & intervals[j]
            for i in range(len(intervals))
            for j in range(i)
        )
        or min(min(interval) for interval in intervals) <= 1360511
    ):
        raise ValueError("cutoff32/36 extension seed firewall invalid")
    resource = config.get("resource_preflight", {})
    if (
        resource.get("required") is not True
        or resource.get("benchmark_cutoffs") != [32, 36]
        or resource.get("benchmark_backends") != ["A", "B"]
        or resource.get("benchmark_scenarios")
        != ["step", "telegraph", "burst", "compound"]
        or resource.get("benchmark_trajectories_per_state") != 1
        or resource.get("seed_splits")
        != {
            "trajectory_backend_a": {"start": 1600000, "count": 6},
            "trajectory_backend_b": {"start": 1601000, "count": 6},
            "heldout_common": {"start": 1602000, "count": 6},
            "round_backend_a": {"start": 1603000, "count": 6},
            "round_backend_b": {"start": 1604000, "count": 6},
            "disjoint_from_design_and_powered_formal": True,
        }
        or resource.get("wall_safety_factor") != 2.0
        or resource.get("rss_delta_safety_factor") != 2.0
        or resource.get("artifact_safety_factor") != 2.0
        or resource.get("minimum_per_worker_delta_bytes") != 67108864
        or resource.get("maximum_estimated_wall_seconds") != 43200
        or resource.get("maximum_estimated_total_rss_bytes") != 17179869184
        or resource.get("maximum_estimated_artifact_bytes") != 2147483648
        or resource.get("thread_pool_accounting")
        != "baseline_rss + max_single_benchmark_delta * max_workers"
        or resource.get("fail_before_scientific_chunk") is not True
    ):
        raise ValueError("cutoff32/36 resource preflight contract drift")
    formal = config.get("powered_formal_preregistration", {})
    if (
        formal.get("release_condition")
        != "DESIGN_GATE_PASS_AUTHORIZES_POWERED_FORMAL"
        or formal.get("cutoffs") != [28, 32, 36]
        or formal.get("fresh_rerun_all_cutoffs") is not True
        or formal.get("clusters_per_state") != 384
        or formal.get("trajectory_count") != 2304
        or formal.get("logical_state_schedule")
        != ["0", "1", "+", "-", "+i", "-i"]
        or formal.get("scenario_names")
        != ["step", "telegraph", "burst", "compound"]
        or formal.get("seed_splits")
        != {
            "trajectory_backend_a": {"start": 1500000, "count": 2304},
            "trajectory_backend_b": {"start": 1503000, "count": 2304},
            "heldout_common": {"start": 1506000, "count": 2304},
            "round_backend_a": {"start": 1510000, "count": 2304},
            "round_backend_b": {"start": 1513000, "count": 2304},
        }
        or formal.get("multiplier_seed_namespace") != 1516000
        or formal.get("calibration_factor") != 1.0
        or formal.get("multiplier_replicates") != 199
        or formal.get("density_point_threshold") != 0.075
        or formal.get("absolute_tail_gate")
        != (
            "primary one-sided coverage-calibrated UCB at cutoff36; "
            "localization-only semantics do not carry into formal"
        )
        or formal.get("design_outcome_may_not_change_contract") is not True
        or formal.get("official_puviani_sota_claims") is not None
    ):
        raise ValueError("powered formal preregistration drift")
    prior = formal.get("prior_passing_gate_composition")
    fresh = formal.get("full_fresh_qualification_required")
    if (
        not isinstance(prior, Mapping)
        or prior.get("allowed") is not False
        or prior.get("physics_repair_invalidates_composition") is not True
        or prior.get("composite_pass_scope") is not None
        or prior.get("expected_prior_gate_counts")
        != {
            "total": 1589,
            "passed": 1562,
            "failed_replaced_only_by_fresh_high_cutoff_formal": 27,
        }
        or not isinstance(fresh, Mapping)
        or fresh
        != {
            "required": True,
            "same_runtime_source_set_for_every_gate": True,
            "all_previous_gate_families_rerun": True,
            "all_high_cutoff_design_gates_rerun_powered": True,
            "old_1562_passing_gates_vote": False,
            "old_27_failing_gates_vote": False,
            "independent_final_verifier_required": True,
            "failure_effect": "NO_GO_TWIN_QUALIFICATION_NO_SEED_EXTENSION",
        }
    ):
        raise ValueError("powered full-fresh supersession contract drift")
    _reference_input_bindings(root, config)
    _validate_uq_preflight_sources(root, config)
    for binding in config["source_bindings"].values():
        path = root / str(binding["path"])
        if _binding(path, root)["sha256"] != binding["sha256"]:
            raise ValueError(f"pilot source binding drift: {binding['path']}")
    base_binding = config["base_config"]
    base_path = root / str(base_binding["path"])
    base_bytes = base_path.read_bytes()
    if sha256(base_bytes).hexdigest() != base_binding["sha256"]:
        raise ValueError("pilot base config binding drift")
    base = json.loads(base_bytes)
    if require_hardened:
        _validate_hardened_confirmation(root, config)
    return config, base


def materialize_execution_config(
    pilot: Mapping[str, Any], base: Mapping[str, Any]
) -> dict[str, Any]:
    execution = json.loads(json.dumps(base))
    count = int(pilot["trajectory_count"])
    execution["formal_matrix"]["trajectory_sample_count"] = count
    execution["formal_matrix"]["cutoff_ladder"] = sorted(
        {
            int(pilot["reference_cutoff"]),
            *[int(value) for value in pilot["cutoffs"]],
        }
    )
    execution["formal_splits"]["trajectory_backend_a"] = dict(
        pilot["seed_splits"]["trajectory_backend_a"]
    )
    execution["formal_splits"]["trajectory_backend_b"] = dict(
        pilot["seed_splits"]["trajectory_backend_b"]
    )
    execution["formal_splits"]["heldout_common"] = dict(
        pilot["seed_splits"]["heldout_common"]
    )
    execution["formal_splits"]["round_backend_a"] = dict(
        pilot["seed_splits"]["round_backend_a"]
    )
    execution["formal_splits"]["round_backend_b"] = dict(
        pilot["seed_splits"]["round_backend_b"]
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
    for specification in pilot["shared_repair_cells"]:
        for cutoff in specification["cutoffs"]:
            for backend in specification["backends"]:
                identity = (
                    f"pilot|c{cutoff}|shared|{specification['initial_state']}|"
                    f"{specification['action']}|{backend}"
                )
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
                        layer="shared",
                        cell_base=(
                            f"shared|{specification['initial_state']}|"
                            f"{specification['action']}"
                        ),
                        cutoff=int(cutoff),
                        backend=str(backend),
                        sample_count=count,
                        convergence_role="shared_reset_high_cutoff_repair",
                        action=str(specification["action"]),
                        initial_state=str(specification["initial_state"]),
                        horizon=1,
                    )
                )
    if (
        len(cells) != 22
        or len({cell.chunk_id for cell in cells}) != 22
        or sum(cell.expected_rows for cell in cells) != 16 * count * 12 + 6 * count
    ):
        raise RuntimeError("cutoff32/36 extension accounting drift")
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
        "input_snapshot_analysis_sha256",
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
        or receipt.get("input_snapshot_analysis_sha256")
        != run_identity.get("input_snapshot_analysis_sha256")
        or receipt.get("pilot_source_sha256") != _pilot_source_sha256()
        or receipt.get("pilot_source_sha256") != run_identity.get("pilot_source_sha256")
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
    _, live = _read_bound_json(
        root,
        receipt_binding,
        expected_path=receipt_path.resolve().relative_to(root.resolve()).as_posix(),
    )
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
    owner_lock: Mapping[str, Any],
    input_snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require_verified_self_import()
    root = Path(root_text).resolve()
    _assert_owner_lock(root, pilot, owner_lock)
    _assert_input_snapshot(root, input_snapshot)
    _activate_verified_execution_modules(root, input_snapshot)
    _assert_input_snapshot(root, input_snapshot)
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
        _assert_owner_lock(root, pilot, owner_lock)
        _assert_input_snapshot(root, input_snapshot)
        return receipt
    simulator = runner.build_simulators(execution, cell.cutoff)[cell.backend]
    evidence = runner.execute_cell(execution, cell, simulator, runner._action_words())
    _assert_owner_lock(root, pilot, owner_lock)
    _assert_input_snapshot(root, input_snapshot)
    chunk = runner.write_chunk(root, execution, cell, evidence)
    receipt = {
        "task_id": TASK_ID,
        "schema_version": RECEIPT_SCHEMA,
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "config_analysis_sha256": _sha(pilot),
        "execution_analysis_sha256": execution_analysis_sha256,
        "input_snapshot_analysis_sha256": _sha(input_snapshot),
        "pilot_source_sha256": _pilot_source_sha256(),
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
    _assert_owner_lock(root, pilot, owner_lock)
    _assert_input_snapshot(root, input_snapshot)
    return receipt


def _verified_thread_worker_probe(
    root_text: str,
    input_snapshot: Mapping[str, Mapping[str, Any]],
    launch_mode: str = "pilot",
) -> dict[str, object]:
    """Exercise the exact production trust checks on a thread worker."""

    _require_verified_self_import(expected_mode=launch_mode)
    root = Path(root_text).resolve()
    _assert_input_snapshot(root, input_snapshot)
    _activate_verified_execution_modules(root, input_snapshot)
    _assert_input_snapshot(root, input_snapshot)
    return {
        "thread_id": threading.get_ident(),
        "execution_module_count": len(_VERIFIED_EXECUTION_MODULES),
        "fresh_runner_sha256": getattr(
            runner,
            "__verified_source_sha256__",
            None,
        ),
    }


def _assert_owner_lock(
    root: Path,
    pilot: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> None:
    lock_path = root / (
        OWNER_LOCK_PATH if pilot is None else str(pilot["artifact_paths"]["owner_lock"])
    )
    try:
        live = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            "owner lock disappeared while supervisor was active"
        ) from exc
    if live != dict(expected) or _self_hash(live) != expected.get("analysis_sha256"):
        raise RuntimeError("owner lock changed while supervisor was active")


@contextmanager
def _exclusive_owner_lock(
    root: Path,
    pilot: Mapping[str, Any] | None = None,
) -> Any:
    lock_relative = (
        OWNER_LOCK_PATH if pilot is None else str(pilot["artifact_paths"]["owner_lock"])
    )
    lock_path = root / lock_relative
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    owner_token = uuid4().hex
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": LOCK_SCHEMA,
        "owner_token": owner_token,
        "pid": os.getpid(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lock_path": lock_relative,
        "released_child_sha256": (RELEASED_CHILD_SHA256 if pilot is None else None),
        "fixture_config_analysis_sha256": (None if pilot is None else _sha(pilot)),
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
        _assert_owner_lock(root, pilot, payload)
        yield payload
    finally:
        _assert_owner_lock(root, pilot, payload)
        lock_path.unlink()


def _load_or_create_run_identity(
    root: Path,
    pilot: Mapping[str, Any],
    execution_analysis_sha256: str,
    input_snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path = root / str(pilot["artifact_paths"]["run_identity"])
    release_lineage = _release_lineage(pilot)
    if path.exists():
        identity = json.loads(path.read_bytes())
        _self_hash(identity)
    else:
        identity = {
            "task_id": TASK_ID,
            "schema_version": RUN_IDENTITY_SCHEMA,
            "run_id": str(uuid4()),
            "config_analysis_sha256": _sha(pilot),
            "execution_analysis_sha256": execution_analysis_sha256,
            "input_snapshot": json.loads(json.dumps(input_snapshot)),
            "input_snapshot_analysis_sha256": _sha(input_snapshot),
            "pilot_source_sha256": _pilot_source_sha256(),
            "release_lineage": release_lineage,
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
        or identity.get("input_snapshot") != input_snapshot
        or identity.get("input_snapshot_analysis_sha256") != _sha(input_snapshot)
        or identity.get("pilot_source_sha256") != _pilot_source_sha256()
        or identity.get("release_lineage") != release_lineage
    ):
        raise RuntimeError("run identity binding drift")
    _assert_input_snapshot(root, input_snapshot)
    return identity


def _chunk_health(root: Path, receipt: Mapping[str, Any]) -> tuple[int, int]:
    exception_rows = 0
    conservation_failures = 0
    binding = receipt.get("csv")
    if not isinstance(binding, Mapping):
        raise RuntimeError("pilot CSV binding missing")
    _, payload = _read_bound_bytes(root, binding)
    with io.StringIO(payload.decode("utf-8"), newline="") as stream:
        for row in csv.DictReader(stream):
            exception = bool(row["exception_type"])
            exception_rows += int(exception)
            conservation = row["conservation_pass"]
            if conservation not in {"True", "False"}:
                raise RuntimeError("conservation_pass is not a strict boolean")
            conservation_failures += int(not exception and conservation != "True")
    return exception_rows, conservation_failures


def _state_physicality(density: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(density, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise RuntimeError("preflight state density shape drift")
    hermitian = 0.5 * (matrix + matrix.conj().T)
    result = {
        "trace_error": abs(float(np.trace(matrix).real) - 1.0)
        + abs(float(np.trace(matrix).imag)),
        "hermiticity_frobenius": float(
            np.linalg.norm(matrix - matrix.conj().T, ord="fro")
        ),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(hermitian).min()),
    }
    if (
        not all(np.isfinite(value) for value in result.values())
        or result["trace_error"] > 5.0e-8
        or result["hermiticity_frobenius"] > 5.0e-8
        or result["minimum_eigenvalue"] < -5.0e-8
    ):
        raise RuntimeError("high-cutoff preflight state physicality failure")
    return result


def _half_trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.complex128) - np.asarray(
        right,
        dtype=np.complex128,
    )
    if (
        difference.ndim != 2
        or difference.shape[0] != difference.shape[1]
        or not np.all(np.isfinite(difference.real))
        or not np.all(np.isfinite(difference.imag))
    ):
        raise RuntimeError("integration-convergence density shape drift")
    hermitian = 0.5 * (difference + difference.conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))


def _validate_high_cutoff_preflight(
    pilot: Mapping[str, Any],
    execution: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    _self_hash(report)
    expected_cutoffs = sorted(int(value) for value in pilot["cutoffs"])
    checks = report.get("checks")
    if (
        set(report)
        != {
            "task_id",
            "schema_version",
            "status",
            "evaluated_cutoffs",
            "production_segment_steps",
            "production_iq_samples",
            "logical_labels",
            "high_energy_cutoffs",
            "high_energy_actions",
            "checks",
            "integration_convergence",
            "integration_convergence_contract",
            "resource_preflight",
            "configured_max_workers",
            "elapsed_seconds",
            "qualified_claim",
            "claim_state",
            "analysis_sha256",
        }
        or report.get("task_id") != TASK_ID
        or report.get("schema_version") != PREFLIGHT_SCHEMA
        or report.get("status") != "PASS"
        or report.get("evaluated_cutoffs") != expected_cutoffs
        or report.get("production_segment_steps")
        != int(execution["common_physics"]["segment_steps"])
        or report.get("production_segment_steps") != 8
        or report.get("production_iq_samples")
        != int(execution["common_physics"]["iq_samples"])
        or report.get("production_iq_samples") != 8
        or report.get("logical_labels") != ["0", "1", "+", "-", "+i", "-i"]
        or report.get("high_energy_cutoffs") != [32, 36]
        or report.get("high_energy_actions")
        != ["IDLE", "X", "Z", "XZ", "HOLD", "RESET"]
        or not isinstance(checks, list)
        or len(checks) != 2 * len(expected_cutoffs)
        or report.get("integration_convergence_contract")
        != {
            "segment_steps": [8, 16, 32],
            "coarse_to_middle_max_trace_distance": 0.005,
            "middle_to_fine_max_trace_distance": 0.0015,
            "refinement_ratio_max": 0.5,
            "states": ["0", "1", "+", "-", "+i", "-i"],
            "actions": ["IDLE", "X", "Z", "XZ", "HOLD", "RESET"],
        }
        or report.get("configured_max_workers") != int(pilot["max_workers"])
        or report.get("qualified_claim") is not None
        or report.get("claim_state") != CLAIM_BOUNDARY
        or isinstance(report.get("elapsed_seconds"), bool)
        or not isinstance(report.get("elapsed_seconds"), (int, float))
        or not np.isfinite(float(report["elapsed_seconds"]))
        or float(report["elapsed_seconds"]) <= 0.0
    ):
        raise RuntimeError("high-cutoff capability preflight contract drift")
    identities: set[tuple[int, str]] = set()
    for check in checks:
        if not isinstance(check, Mapping):
            raise RuntimeError("high-cutoff preflight check type drift")
        identity = (int(check.get("cutoff", -1)), str(check.get("backend", "")))
        if identity in identities:
            raise RuntimeError("duplicate high-cutoff preflight check")
        identities.add(identity)
        expected_actions = 6
        if (
            identity[0] not in expected_cutoffs
            or identity[1] not in {"A", "B"}
            or check.get("dimension") != 3 * identity[0]
            or check.get("logical_states_initialized") != 6
            or check.get("high_energy_actions_executed") != expected_actions
            or check.get("all_checks_passed") is not True
            or float(check.get("maximum_trace_error", float("inf"))) > 5.0e-8
            or float(
                check.get("maximum_hermiticity_frobenius", float("inf"))
            )
            > 5.0e-8
            or float(check.get("minimum_eigenvalue", float("-inf"))) < -5.0e-8
        ):
            raise RuntimeError("high-cutoff preflight check failed")
    if identities != {
        (cutoff, backend)
        for cutoff in expected_cutoffs
        for backend in ("A", "B")
    }:
        raise RuntimeError("high-cutoff preflight coverage drift")
    convergence = report.get("integration_convergence")
    if not isinstance(convergence, list) or len(convergence) != 144:
        raise RuntimeError("high-cutoff convergence coverage drift")
    convergence_identities: set[tuple[int, str, str, str]] = set()
    for check in convergence:
        if not isinstance(check, Mapping):
            raise RuntimeError("high-cutoff convergence check type drift")
        identity = (
            int(check.get("cutoff", -1)),
            str(check.get("backend", "")),
            str(check.get("state", "")),
            str(check.get("action", "")),
        )
        if identity in convergence_identities:
            raise RuntimeError("duplicate high-cutoff convergence check")
        convergence_identities.add(identity)
        coarse = float(check.get("trace_distance_8_to_16", float("inf")))
        fine = float(check.get("trace_distance_16_to_32", float("inf")))
        ratio = float(check.get("refinement_ratio", float("inf")))
        if (
            identity[0] not in {32, 36}
            or identity[1] not in {"A", "B"}
            or identity[2] not in {"0", "1", "+", "-", "+i", "-i"}
            or identity[3] not in {"IDLE", "X", "Z", "XZ", "HOLD", "RESET"}
            or check.get("all_checks_passed") is not True
            or not all(np.isfinite(value) for value in (coarse, fine, ratio))
            or coarse > 0.005
            or fine > 0.0015
            or ratio > 0.5
        ):
            raise RuntimeError("high-cutoff integration convergence failed")
    if convergence_identities != {
        (cutoff, backend, state, action)
        for cutoff in (32, 36)
        for backend in ("A", "B")
        for state in ("0", "1", "+", "-", "+i", "-i")
        for action in ("IDLE", "X", "Z", "XZ", "HOLD", "RESET")
    }:
        raise RuntimeError("high-cutoff convergence identity drift")
    _validate_resource_preflight(pilot, report.get("resource_preflight"))


def _resource_preflight_identity(
    record: Mapping[str, Any],
) -> tuple[int, str, str, str | None, str | None, str | None]:
    return (
        int(record.get("cutoff", -1)),
        str(record.get("backend", "")),
        str(record.get("layer", "")),
        None if record.get("scenario") is None else str(record["scenario"]),
        (
            None
            if record.get("initial_state") is None
            else str(record["initial_state"])
        ),
        None if record.get("action") is None else str(record["action"]),
    )


def _expected_resource_preflight_identities(
    pilot: Mapping[str, Any],
) -> set[tuple[int, str, str, str | None, str | None, str | None]]:
    contract = pilot["resource_preflight"]
    identities = {
        (
            int(cutoff),
            str(backend),
            "fault",
            str(scenario),
            None,
            None,
        )
        for cutoff in contract["benchmark_cutoffs"]
        for backend in contract["benchmark_backends"]
        for scenario in contract["benchmark_scenarios"]
    }
    identities.update(
        {
            (
                int(cutoff),
                str(backend),
                "shared",
                None,
                str(specification["initial_state"]),
                str(specification["action"]),
            )
            for specification in pilot["shared_repair_cells"]
            for cutoff in specification["cutoffs"]
            for backend in specification["backends"]
        }
    )
    if len(identities) != 22:
        raise RuntimeError("resource preflight preregistered identity drift")
    return identities


def _validate_resource_preflight(
    pilot: Mapping[str, Any],
    resource: object,
) -> None:
    contract = pilot["resource_preflight"]
    expected_resource_keys = {
        "benchmark_records",
        "baseline_rss_bytes",
        "maximum_single_benchmark_rss_delta_bytes",
        "minimum_accounted_per_worker_delta_bytes",
        "estimated_wall_seconds_with_safety_factor",
        "estimated_total_rss_bytes",
        "estimated_artifact_bytes",
        "wall_limit_seconds",
        "rss_limit_bytes",
        "artifact_limit_bytes",
        "configured_max_workers",
        "seed_splits",
        "design_outcomes_accessed",
        "passed",
        "analysis_sha256",
    }
    records = (
        resource.get("benchmark_records")
        if isinstance(resource, Mapping)
        else None
    )
    if (
        not isinstance(resource, Mapping)
        or set(resource) != expected_resource_keys
        or _self_hash(resource) != resource.get("analysis_sha256")
        or resource.get("configured_max_workers") != int(pilot["max_workers"])
        or resource.get("seed_splits")
        != pilot["resource_preflight"]["seed_splits"]
        or resource.get("design_outcomes_accessed") is not False
        or resource.get("wall_limit_seconds")
        != int(contract["maximum_estimated_wall_seconds"])
        or resource.get("rss_limit_bytes")
        != int(contract["maximum_estimated_total_rss_bytes"])
        or resource.get("artifact_limit_bytes")
        != int(contract["maximum_estimated_artifact_bytes"])
        or resource.get("passed") is not True
        or not isinstance(records, list)
        or len(records) != 22
        or float(resource.get("estimated_wall_seconds_with_safety_factor", -1))
        <= 0.0
        or int(resource.get("estimated_total_rss_bytes", -1)) <= 0
        or float(resource["estimated_wall_seconds_with_safety_factor"])
        > int(contract["maximum_estimated_wall_seconds"])
        or int(resource["estimated_total_rss_bytes"])
        > int(contract["maximum_estimated_total_rss_bytes"])
        or int(resource.get("estimated_artifact_bytes", -1)) <= 0
        or int(resource["estimated_artifact_bytes"])
        > int(contract["maximum_estimated_artifact_bytes"])
    ):
        raise RuntimeError("cutoff32/36 resource preflight failed")
    expected_record_keys = {
        "cutoff",
        "backend",
        "layer",
        "scenario",
        "initial_state",
        "action",
        "sample_count",
        "observed_rows",
        "terminal_density_count",
        "elapsed_seconds",
        "peak_process_rss_bytes",
        "rss_delta_bytes",
    }
    expected_identities = _expected_resource_preflight_identities(pilot)
    observed_identities: set[
        tuple[int, str, str, str | None, str | None, str | None]
    ] = set()
    expected_sample_count = 6 * int(
        contract["benchmark_trajectories_per_state"]
    )
    for record in records:
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_record_keys
        ):
            raise RuntimeError("resource preflight record schema drift")
        identity = _resource_preflight_identity(record)
        if identity in observed_identities:
            raise RuntimeError("duplicate resource preflight identity")
        observed_identities.add(identity)
        expected_rows = (
            12 * expected_sample_count
            if identity[2] == "fault"
            else expected_sample_count
        )
        numeric_values = (
            record.get("elapsed_seconds"),
            record.get("peak_process_rss_bytes"),
            record.get("rss_delta_bytes"),
        )
        if (
            identity not in expected_identities
            or record.get("sample_count") != expected_sample_count
            or record.get("terminal_density_count") != expected_sample_count
            or record.get("observed_rows") != expected_rows
            or any(isinstance(value, bool) for value in numeric_values)
            or not all(
                isinstance(value, (int, float)) for value in numeric_values
            )
            or not all(np.isfinite(float(value)) for value in numeric_values)
            or float(record["elapsed_seconds"]) <= 0.0
            or int(record["peak_process_rss_bytes"]) <= 0
            or int(record["rss_delta_bytes"]) < 0
        ):
            raise RuntimeError("resource preflight record semantic drift")
    if observed_identities != expected_identities:
        raise RuntimeError("resource preflight exact identity coverage drift")


def _resource_preflight_path(
    root: Path,
    pilot: Mapping[str, Any],
) -> Path:
    return (
        root
        / str(pilot["artifact_paths"]["run_directory"])
        / "resource_preflight.json"
    )


def _publish_resource_preflight(
    root: Path,
    pilot: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    path = _resource_preflight_path(root, pilot)
    _atomic_text(
        path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    binding = _binding(path, root)
    _, live = _read_bound_json(root, binding)
    _validate_resource_preflight(pilot, live)
    return live


def _run_resource_preflight(
    pilot: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    contract = pilot["resource_preflight"]
    process = psutil.Process(os.getpid())
    baseline_rss = int(process.memory_info().rss)
    records: list[dict[str, object]] = []
    sample_count = 6 * int(contract["benchmark_trajectories_per_state"])
    resource_execution = json.loads(json.dumps(execution))
    for split_name in (
        "round_backend_a",
        "round_backend_b",
        "trajectory_backend_a",
        "trajectory_backend_b",
        "heldout_common",
    ):
        resource_execution["formal_splits"][split_name] = dict(
            contract["seed_splits"][split_name]
        )

    def benchmark_cell(cell: runner.CellSpec) -> None:
        simulator = runner.build_simulators(
            resource_execution, cell.cutoff
        )[cell.backend]
        peak = [int(process.memory_info().rss)]
        stop = threading.Event()

        def sample_rss() -> None:
            while not stop.wait(0.01):
                peak[0] = max(peak[0], int(process.memory_info().rss))

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        started = time.perf_counter()
        try:
            evidence = runner.execute_cell(
                resource_execution,
                cell,
                simulator,
                runner._action_words(),
            )
        finally:
            stop.set()
            sampler.join(timeout=2.0)
            peak[0] = max(peak[0], int(process.memory_info().rss))
        elapsed = time.perf_counter() - started
        if (
            len(evidence.rows) != cell.expected_rows
            or len(evidence.densities) != sample_count
            or any(row["exception_type"] for row in evidence.rows)
            or any(row["conservation_pass"] is not True for row in evidence.rows)
        ):
            raise RuntimeError("resource preflight scientific path failure")
        for density in evidence.densities:
            _state_physicality(density)
        records.append(
            {
                "cutoff": cell.cutoff,
                "backend": cell.backend,
                "layer": cell.layer,
                "scenario": cell.scenario,
                "initial_state": cell.initial_state,
                "action": cell.action,
                "sample_count": sample_count,
                "observed_rows": len(evidence.rows),
                "terminal_density_count": len(evidence.densities),
                "elapsed_seconds": elapsed,
                "peak_process_rss_bytes": peak[0],
                "rss_delta_bytes": max(0, peak[0] - baseline_rss),
            }
        )

    for cutoff in contract["benchmark_cutoffs"]:
        for backend in contract["benchmark_backends"]:
            for scenario in contract["benchmark_scenarios"]:
                benchmark_cell(
                    runner.CellSpec(
                        chunk_id=f"resource_c{cutoff}_{backend}_{scenario}",
                        layer="fault",
                        cell_base=f"fault|{scenario}",
                        cutoff=int(cutoff),
                        backend=str(backend),
                        sample_count=sample_count,
                        convergence_role="resource_preflight_only",
                        scenario=str(scenario),
                        horizon=12,
                    )
                )
    for specification in pilot["shared_repair_cells"]:
        for cutoff in specification["cutoffs"]:
            for backend in specification["backends"]:
                benchmark_cell(
                    runner.CellSpec(
                        chunk_id=f"resource_shared_c{cutoff}_{backend}",
                        layer="shared",
                        cell_base=(
                            f"shared|{specification['initial_state']}|"
                            f"{specification['action']}"
                        ),
                        cutoff=int(cutoff),
                        backend=str(backend),
                        sample_count=sample_count,
                        convergence_role="resource_preflight_only",
                        initial_state=str(specification["initial_state"]),
                        action=str(specification["action"]),
                        horizon=1,
                    )
                )
    workers = int(pilot["max_workers"])
    full_scale = (
        int(pilot["trajectory_count"])
        / (6 * int(contract["benchmark_trajectories_per_state"]))
    )
    estimated_wall = (
        sum(float(row["elapsed_seconds"]) * full_scale for row in records)
        / workers
        * float(contract["wall_safety_factor"])
    )
    maximum_delta = max(int(row["rss_delta_bytes"]) for row in records)
    accounted_delta = max(
        maximum_delta,
        int(contract["minimum_per_worker_delta_bytes"]),
    )
    estimated_rss = baseline_rss + int(
        accounted_delta
        * workers
        * float(contract["rss_delta_safety_factor"])
    )
    reference_average_csv_bytes = 1211989.25
    reference_average_npz_bytes = 3272848.375
    estimated_artifact_bytes = int(
        float(contract["artifact_safety_factor"])
        * (
            sum(
                8
                * (
                    reference_average_csv_bytes
                    + reference_average_npz_bytes * (cutoff / 28.0) ** 2
                )
                for cutoff in contract["benchmark_cutoffs"]
            )
            + 6
            * (
                reference_average_csv_bytes / 12.0
                + reference_average_npz_bytes
                * (max(contract["benchmark_cutoffs"]) / 28.0) ** 2
            )
        )
    )
    report: dict[str, Any] = {
        "benchmark_records": records,
        "baseline_rss_bytes": baseline_rss,
        "maximum_single_benchmark_rss_delta_bytes": maximum_delta,
        "minimum_accounted_per_worker_delta_bytes": accounted_delta,
        "estimated_wall_seconds_with_safety_factor": estimated_wall,
        "estimated_total_rss_bytes": estimated_rss,
        "estimated_artifact_bytes": estimated_artifact_bytes,
        "wall_limit_seconds": int(contract["maximum_estimated_wall_seconds"]),
        "rss_limit_bytes": int(contract["maximum_estimated_total_rss_bytes"]),
        "artifact_limit_bytes": int(
            contract["maximum_estimated_artifact_bytes"]
        ),
        "configured_max_workers": workers,
        "seed_splits": dict(contract["seed_splits"]),
        "design_outcomes_accessed": False,
        "passed": (
            estimated_wall
            <= float(contract["maximum_estimated_wall_seconds"])
            and estimated_rss
            <= int(contract["maximum_estimated_total_rss_bytes"])
            and estimated_artifact_bytes
            <= int(contract["maximum_estimated_artifact_bytes"])
        ),
    }
    report["analysis_sha256"] = _sha(report)
    return report


def _run_high_cutoff_preflight(
    root: Path,
    pilot: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Exercise the full production simulator factory before worker launch.

    This preflight intentionally uses the frozen production integration depth
    and IQ width.  It initializes all six logical states at every registered
    cutoff and executes IDLE/XZ/RESET from a top-four-Fock superposition at
    cutoff 32 and 36.  It therefore catches constructor-only support, bridge
    limits, state validators, and action paths before a long transaction writes
    any scientific chunk.
    """

    started = time.perf_counter()
    labels = ["0", "1", "+", "-", "+i", "-i"]
    actions = runner._action_words()
    preflight_actions = ["IDLE", "X", "Z", "XZ", "HOLD", "RESET"]
    cutoffs = sorted(int(value) for value in pilot["cutoffs"])
    resource_preflight = _publish_resource_preflight(
        root,
        pilot,
        _run_resource_preflight(pilot, execution),
    )
    checks: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        simulators = runner.build_simulators(execution, cutoff)
        if set(simulators) != {"A", "B"}:
            raise RuntimeError("high-cutoff preflight simulator set drift")
        for backend in ("A", "B"):
            simulator = simulators[backend]
            metrics: list[dict[str, float]] = []
            for label in labels:
                logical_state, _evaluator = simulator.initialize_logical(label)
                if (
                    logical_state.cutoff != cutoff
                    or logical_state.joint_density.shape != (3 * cutoff, 3 * cutoff)
                ):
                    raise RuntimeError("high-cutoff logical initialization drift")
                metrics.append(_state_physicality(logical_state.joint_density))
            action_count = 0
            if cutoff in {32, 36}:
                high_energy = np.zeros(cutoff, dtype=np.complex128)
                high_energy[-4:] = 0.5
                state = simulator.initialize_fock(
                    oscillator_ket=high_energy,
                    ancilla_state="f",
                )
                for action_name in preflight_actions:
                    result = runner._one_step(
                        backend=backend,
                        simulator=simulator,
                        state=state,
                        evaluator=None,
                        action=actions[action_name],
                        seed=1_439_000 + 100 * cutoff + action_count,
                    )
                    metrics.append(_state_physicality(result.state.joint_density))
                    if (
                        result.state.cutoff != cutoff
                        or result.state.joint_density.shape
                        != (3 * cutoff, 3 * cutoff)
                        or not np.all(np.isfinite(result.observation.iq_i))
                        or not np.all(np.isfinite(result.observation.iq_q))
                    ):
                        raise RuntimeError("high-cutoff action preflight drift")
                    state = result.state
                    action_count += 1
            checks.append(
                {
                    "cutoff": cutoff,
                    "backend": backend,
                    "dimension": 3 * cutoff,
                    "logical_states_initialized": len(labels),
                    "high_energy_actions_executed": action_count,
                    "maximum_trace_error": max(
                        value["trace_error"] for value in metrics
                    ),
                    "maximum_hermiticity_frobenius": max(
                        value["hermiticity_frobenius"] for value in metrics
                    ),
                    "minimum_eigenvalue": min(
                        value["minimum_eigenvalue"] for value in metrics
                    ),
                    "all_checks_passed": True,
                }
            )
    integration_convergence: list[dict[str, Any]] = []
    convergence_actions = list(preflight_actions)
    for cutoff in (32, 36):
        for backend in ("A", "B"):
            density_grid: dict[tuple[str, str, int], np.ndarray] = {}
            for segment_steps in (8, 16, 32):
                refined_execution = json.loads(json.dumps(execution))
                refined_execution["common_physics"]["segment_steps"] = segment_steps
                simulator = runner.build_simulators(refined_execution, cutoff)[backend]
                for state_index, label in enumerate(labels):
                    for action_index, action_name in enumerate(convergence_actions):
                        logical_state, evaluator = simulator.initialize_logical(label)
                        result = runner._one_step(
                            backend=backend,
                            simulator=simulator,
                            state=logical_state,
                            evaluator=evaluator,
                            action=actions[action_name],
                            seed=(
                                1_439_500
                                + 10_000 * cutoff
                                + 100 * state_index
                                + action_index
                            ),
                        )
                        _state_physicality(result.state.joint_density)
                        if (
                            result.state.cutoff != cutoff
                            or not np.all(np.isfinite(result.observation.iq_i))
                            or not np.all(np.isfinite(result.observation.iq_q))
                        ):
                            raise RuntimeError(
                                "high-cutoff convergence execution drift"
                            )
                        density_grid[
                            (label, action_name, segment_steps)
                        ] = result.state.joint_density
            for label in labels:
                for action_name in convergence_actions:
                    coarse = _half_trace_distance(
                        density_grid[(label, action_name, 8)],
                        density_grid[(label, action_name, 16)],
                    )
                    fine = _half_trace_distance(
                        density_grid[(label, action_name, 16)],
                        density_grid[(label, action_name, 32)],
                    )
                    ratio = fine / max(coarse, np.finfo(np.float64).eps)
                    integration_convergence.append(
                        {
                            "cutoff": cutoff,
                            "backend": backend,
                            "state": label,
                            "action": action_name,
                            "trace_distance_8_to_16": coarse,
                            "trace_distance_16_to_32": fine,
                            "refinement_ratio": ratio,
                            "all_checks_passed": (
                                coarse <= 0.005
                                and fine <= 0.0015
                                and ratio <= 0.5
                            ),
                        }
                    )
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "PASS",
        "evaluated_cutoffs": cutoffs,
        "production_segment_steps": int(
            execution["common_physics"]["segment_steps"]
        ),
        "production_iq_samples": int(execution["common_physics"]["iq_samples"]),
        "logical_labels": labels,
        "high_energy_cutoffs": [32, 36],
        "high_energy_actions": preflight_actions,
        "checks": checks,
        "integration_convergence": integration_convergence,
        "integration_convergence_contract": {
            "segment_steps": [8, 16, 32],
            "coarse_to_middle_max_trace_distance": 0.005,
            "middle_to_fine_max_trace_distance": 0.0015,
            "refinement_ratio_max": 0.5,
            "states": labels,
            "actions": convergence_actions,
        },
        "resource_preflight": resource_preflight,
        "configured_max_workers": int(pilot["max_workers"]),
        "elapsed_seconds": time.perf_counter() - started,
        "qualified_claim": None,
        "claim_state": dict(CLAIM_BOUNDARY),
    }
    report["analysis_sha256"] = _sha(report)
    _validate_high_cutoff_preflight(pilot, execution, report)
    return report


def _heartbeat(
    root: Path,
    pilot: Mapping[str, Any],
    *,
    completed: int,
    total: int,
    active: bool,
    state: str,
    run_identity: Mapping[str, Any],
    input_snapshot: Mapping[str, Mapping[str, Any]],
    manifest_binding: Mapping[str, Any] | None = None,
    manifest_analysis_sha256: str | None = None,
    error_type: str | None = None,
    owner_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in {"RUNNING", "COMPLETE", "REJECTED", "FAILED"}:
        raise ValueError("pilot heartbeat state drift")
    terminal_with_manifest = state in {"COMPLETE", "REJECTED"}
    if terminal_with_manifest != (manifest_binding is not None):
        raise ValueError("pilot heartbeat manifest commit-marker drift")
    if terminal_with_manifest != (manifest_analysis_sha256 is not None):
        raise ValueError("pilot heartbeat manifest analysis drift")
    if owner_lock is not None:
        _assert_owner_lock(root, pilot, owner_lock)
    _assert_input_snapshot(root, input_snapshot)
    payload = {
        "task_id": TASK_ID,
        "schema_version": HEARTBEAT_SCHEMA,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "run_id": run_identity["run_id"],
        "run_identity_analysis_sha256": run_identity["analysis_sha256"],
        "input_snapshot_analysis_sha256": _sha(input_snapshot),
        "completed_cells": completed,
        "expected_cells": total,
        "active": active,
        "state": state,
        "error_type": error_type,
        "manifest": None if manifest_binding is None else dict(manifest_binding),
        "manifest_analysis_sha256": manifest_analysis_sha256,
    }
    payload["analysis_sha256"] = _sha(payload)
    _atomic_text(
        root / str(pilot["artifact_paths"]["heartbeat"]),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if owner_lock is not None:
        _assert_owner_lock(root, pilot, owner_lock)
    _assert_input_snapshot(root, input_snapshot)
    live = json.loads((root / str(pilot["artifact_paths"]["heartbeat"])).read_bytes())
    if live != payload or _self_hash(live) != payload["analysis_sha256"]:
        raise RuntimeError("pilot heartbeat live commit drift")
    return payload


def _verify_complete_marker(
    root: Path,
    pilot: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / str(pilot["artifact_paths"]["heartbeat"])
    marker = json.loads(path.read_bytes())
    _self_hash(marker)
    manifest_path = root / str(pilot["artifact_paths"]["execution_manifest"])
    manifest_binding = _binding(manifest_path, root)
    if (
        set(marker)
        != {
            "task_id",
            "schema_version",
            "timestamp_utc",
            "pid",
            "run_id",
            "run_identity_analysis_sha256",
            "input_snapshot_analysis_sha256",
            "completed_cells",
            "expected_cells",
            "active",
            "state",
            "error_type",
            "manifest",
            "manifest_analysis_sha256",
            "analysis_sha256",
        }
        or marker.get("task_id") != TASK_ID
        or marker.get("schema_version") != HEARTBEAT_SCHEMA
        or marker.get("run_id") != run_identity["run_id"]
        or marker.get("run_identity_analysis_sha256") != run_identity["analysis_sha256"]
        or marker.get("input_snapshot_analysis_sha256")
        != run_identity["input_snapshot_analysis_sha256"]
        or marker.get("completed_cells") != manifest["observed_cells"]
        or marker.get("expected_cells") != manifest["observed_cells"]
        or marker.get("active") is not False
        or marker.get("state") != "COMPLETE"
        or marker.get("error_type") is not None
        or marker.get("manifest") != manifest_binding
        or marker.get("manifest_analysis_sha256") != manifest["analysis_sha256"]
    ):
        raise RuntimeError("pilot COMPLETE commit marker drift")
    return marker


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
        "input_snapshot_analysis_sha256",
        "pilot_source_sha256",
        "release_lineage",
        "observed_cells",
        "observed_rows",
        "exception_rows",
        "conservation_failure_rows",
        "chunk_receipts",
        "receipt_bindings",
        "capability_preflight",
        "claim_state",
        "bindings",
        "runtime",
        "analysis_sha256",
    }
    execution_analysis_sha256 = _sha(execution)
    release_lineage = _release_lineage(pilot)
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
        or manifest.get("input_snapshot_analysis_sha256")
        != run_identity.get("input_snapshot_analysis_sha256")
        or manifest.get("pilot_source_sha256") != _pilot_source_sha256()
        or manifest.get("pilot_source_sha256")
        != run_identity.get("pilot_source_sha256")
        or manifest.get("release_lineage") != release_lineage
        or manifest.get("observed_cells") != len(cells)
        or manifest.get("observed_rows") != sum(cell.expected_rows for cell in cells)
        or manifest.get("exception_rows") != 0
        or manifest.get("conservation_failure_rows") != 0
        or manifest.get("claim_state") != CLAIM_BOUNDARY
    ):
        raise RuntimeError("existing pilot manifest semantic drift")
    input_snapshot = run_identity.get("input_snapshot")
    if not isinstance(input_snapshot, Mapping):
        raise RuntimeError("run identity input snapshot missing")
    _assert_input_snapshot(root, input_snapshot)
    preflight = manifest.get("capability_preflight")
    if not isinstance(preflight, Mapping):
        raise RuntimeError("existing pilot capability preflight missing")
    _validate_high_cutoff_preflight(pilot, execution, preflight)
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
        "pending_config": dict(release_lineage["pending_parent"]),
        "release_receipt": dict(release_lineage["release_receipt"]),
        "base_config": _binding(root / str(pilot["base_config"]["path"]), root),
        "pilot_source": _binding(Path(__file__).resolve(), root),
        "run_identity": _binding(
            root / str(pilot["artifact_paths"]["run_identity"]), root
        ),
        "resource_preflight": _binding(
            _resource_preflight_path(root, pilot), root
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
        **{
            name: dict(binding)
            for name, binding in _validate_uq_preflight_sources(
                root, pilot
            ).items()
        },
    }
    for name in (
        "verified_bootstrap_source",
        "verified_external_launch_meta",
    ):
        binding = input_snapshot.get(name)
        if not isinstance(binding, Mapping):
            raise RuntimeError(f"existing pilot manifest missing {name}")
        expected_bindings[name] = dict(binding)
    if manifest.get("bindings") != expected_bindings:
        raise RuntimeError("existing pilot manifest live binding drift")


def run_pilot(root: Path) -> dict[str, Any]:
    _require_verified_self_import()
    root = root.resolve()
    with _exclusive_owner_lock(root) as owner_lock:
        pilot, base = load_pilot_config(root, require_hardened=True)
        if str(pilot["artifact_paths"]["owner_lock"]) != OWNER_LOCK_PATH:
            raise RuntimeError("pilot owner-lock path drift")
        execution = materialize_execution_config(pilot, base)
        execution_analysis_sha256 = _sha(execution)
        input_snapshot = _build_input_snapshot(root, pilot)
        _assert_owner_lock(root, None, owner_lock)
        _activate_verified_execution_modules(root, input_snapshot)
        _assert_input_snapshot(root, input_snapshot)
        cells = build_pilot_cells(pilot, execution)
        run_identity = _load_or_create_run_identity(
            root,
            pilot,
            execution_analysis_sha256,
            input_snapshot,
        )
        manifest_path = root / str(pilot["artifact_paths"]["execution_manifest"])
        if manifest_path.exists():
            manifest_binding = _binding(manifest_path, root)
            _, manifest = _read_bound_json(root, manifest_binding)
            _verify_manifest(
                root,
                pilot,
                execution,
                cells,
                run_identity,
                manifest,
            )
            try:
                _verify_complete_marker(root, pilot, run_identity, manifest)
            except (FileNotFoundError, RuntimeError, ValueError, KeyError):
                _heartbeat(
                    root,
                    pilot,
                    completed=len(cells),
                    total=len(cells),
                    active=False,
                    state="COMPLETE",
                    run_identity=run_identity,
                    input_snapshot=input_snapshot,
                    manifest_binding=manifest_binding,
                    manifest_analysis_sha256=manifest["analysis_sha256"],
                    owner_lock=owner_lock,
                )
            _verify_complete_marker(root, pilot, run_identity, manifest)
            _assert_input_snapshot(root, input_snapshot)
            return manifest

        _heartbeat(
            root,
            pilot,
            completed=0,
            total=len(cells),
            active=True,
            state="RUNNING",
            run_identity=run_identity,
            input_snapshot=input_snapshot,
            owner_lock=owner_lock,
        )
        receipts: list[dict[str, Any]] = []
        try:
            capability_preflight = _run_high_cutoff_preflight(
                root, pilot, execution
            )
            _assert_owner_lock(root, pilot, owner_lock)
            _assert_input_snapshot(root, input_snapshot)
            with ThreadPoolExecutor(max_workers=int(pilot["max_workers"])) as executor:
                futures = {
                    executor.submit(
                        _worker,
                        str(root),
                        pilot,
                        execution,
                        asdict(cell),
                        run_identity,
                        execution_analysis_sha256,
                        owner_lock,
                        input_snapshot,
                    ): cell
                    for cell in cells
                }
                for future in as_completed(futures):
                    receipts.append(future.result())
                    _assert_input_snapshot(root, input_snapshot)
                    _heartbeat(
                        root,
                        pilot,
                        completed=len(receipts),
                        total=len(cells),
                        active=True,
                        state="RUNNING",
                        run_identity=run_identity,
                        input_snapshot=input_snapshot,
                        owner_lock=owner_lock,
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
                _assert_input_snapshot(root, input_snapshot)
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
                "input_snapshot_analysis_sha256": _sha(input_snapshot),
                "pilot_source_sha256": _pilot_source_sha256(),
                "release_lineage": _release_lineage(pilot),
                "observed_cells": len(ordered),
                "observed_rows": sum(cell.expected_rows for cell in cells),
                "exception_rows": exception_rows,
                "conservation_failure_rows": conservation_failure_rows,
                "chunk_receipts": ordered,
                "receipt_bindings": receipt_bindings,
                "capability_preflight": capability_preflight,
                "claim_state": dict(pilot["claim_boundary"]),
                "bindings": {
                    "config": _binding(root / CONFIG_PATH, root),
                    "pending_config": dict(_release_lineage(pilot)["pending_parent"]),
                    "release_receipt": dict(_release_lineage(pilot)["release_receipt"]),
                    "base_config": _binding(
                        root / str(pilot["base_config"]["path"]), root
                    ),
                    "pilot_source": _binding(Path(__file__).resolve(), root),
                    "run_identity": _binding(
                        root / str(pilot["artifact_paths"]["run_identity"]),
                        root,
                    ),
                    "resource_preflight": _binding(
                        _resource_preflight_path(root, pilot),
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
                    **{
                        name: dict(binding)
                        for name, binding in _validate_uq_preflight_sources(
                            root, pilot
                        ).items()
                    },
                    **{
                        name: dict(input_snapshot[name])
                        for name in (
                            "verified_bootstrap_source",
                            "verified_external_launch_meta",
                        )
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
            _assert_input_snapshot(root, input_snapshot)
            if not healthy:
                _atomic_text(
                    manifest_path,
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                )
                rejected_binding = _binding(manifest_path, root)
                _heartbeat(
                    root,
                    pilot,
                    completed=len(cells),
                    total=len(cells),
                    active=False,
                    state="REJECTED",
                    run_identity=run_identity,
                    input_snapshot=input_snapshot,
                    manifest_binding=rejected_binding,
                    manifest_analysis_sha256=manifest["analysis_sha256"],
                    owner_lock=owner_lock,
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
                manifest_binding = _binding(manifest_path, root)
                _, live_manifest = _read_bound_json(
                    root,
                    manifest_binding,
                )
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
                    run_identity=run_identity,
                    input_snapshot=input_snapshot,
                    manifest_binding=manifest_binding,
                    manifest_analysis_sha256=live_manifest["analysis_sha256"],
                    owner_lock=owner_lock,
                )
                _verify_complete_marker(root, pilot, run_identity, live_manifest)
                _assert_input_snapshot(root, input_snapshot)
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
                run_identity=run_identity,
                input_snapshot=input_snapshot,
                error_type=type(exc).__name__,
                owner_lock=owner_lock,
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
