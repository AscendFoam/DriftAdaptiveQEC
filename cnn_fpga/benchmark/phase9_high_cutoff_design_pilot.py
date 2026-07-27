"""Resumable state-conditioned high-cutoff design pilot.

The pilot reuses the already validated physics execution kernel but has a new,
disjoint seed namespace and a deliberately small denominator.  It produces
raw chunks only for designing the subsequent formal matrix.  It cannot emit
a twin-qualification verdict or release any blocked downstream task.
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
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

import numpy as np
import scipy


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
    "configs/phase9/" "t_risk_20260727_01_high_cutoff_design_pilot_fresh2_released.json"
)
PENDING_CONFIG_PATH = (
    "configs/phase9/t_risk_20260727_01_high_cutoff_design_pilot_fresh2.json"
)
RELEASE_RECEIPT_PATH = (
    "docs/t_risk_20260727_01_high_cutoff_design_pilot_fresh2_release_receipt.json"
)
OWNER_LOCK_PATH = (
    "runs/t_risk_20260727_01_high_cutoff_design_pilot_fresh2/" "supervisor.owner.lock"
)
CONFIG_SCHEMA = "PHASE9-HIGH-CUTOFF-STATE-DESIGN-PILOT-CONFIG-V2"
RELEASED_CHILD_SCHEMA = "PHASE9-HIGH-CUTOFF-DESIGN-PILOT-RELEASED-CHILD-V1"
RELEASE_RECEIPT_SCHEMA = "PHASE9-HIGH-CUTOFF-DESIGN-PILOT-RELEASE-RECEIPT-V1"
PENDING_CONFIG_BYTES = 5701
PENDING_CONFIG_SHA256 = (
    "0e32c27a72f4105bf9ce51a65935586deaafde72bb520a961716965f9e8c6329"
)
RELEASED_CHILD_BYTES = 2821
RELEASED_CHILD_SHA256 = (
    "248e8cabe2f4e1264cd5256fc2d3e5f3b60c54bdfe2afb11e2880163ed6e6992"
)
RELEASED_CHILD_ANALYSIS_SHA256 = (
    "d2cc70dd7bbac1071dca2f23acb55c8804168dea84279f3fca962b2b8f6ee0b6"
)
RELEASE_RECEIPT_BYTES = 2092
RELEASE_RECEIPT_SHA256 = (
    "c286f3ab73bfcec2971f506e1182c337679d70dfa30c66dcd54835e4332a9ffa"
)
RELEASE_RECEIPT_ANALYSIS_SHA256 = (
    "f2165c13d81e948d77a74033a6f6d13b06de88253ada4b4487e86239ba7ac301"
)
MANIFEST_SCHEMA = "PHASE9-HIGH-CUTOFF-STATE-DESIGN-PILOT-MANIFEST-V4"
RECEIPT_SCHEMA = "PHASE9-HIGH-CUTOFF-PILOT-CHUNK-RECEIPT-V3"
RUN_IDENTITY_SCHEMA = "PHASE9-HIGH-CUTOFF-PILOT-RUN-IDENTITY-V3"
LOCK_SCHEMA = "PHASE9-HIGH-CUTOFF-PILOT-OWNER-LOCK-V2"
HEARTBEAT_SCHEMA = "PHASE9-HIGH-CUTOFF-PILOT-HEARTBEAT-V2"
HARDENED_CONFIRMATION_SCHEMA = "PHASE9-PAIRED-CLUSTER-UQ-HARDENED-CONFIRMATION-V2"
HARDENED_CONFIRMATION_ANALYSIS_SHA256 = (
    "5a798e45c0306d4bf591c971e52c68e4faf0dce276eafc74cb69d66ef6abe5a5"
)
HARDENED_CONFIRMATION_PASS = "PASS_PAIRED_CLUSTER_UQ_HARDENED_CONFIRMATION"
NARROW_AUTHORIZATION_STATE = "NARROW_UNPOWERED_EXPLORATORY_LOCALIZATION_ONLY"
NARROW_SCOPE = {
    "purpose": "unpowered exploratory localization",
    "synthetic_coverage_only": True,
    "physical_coverage_guarantee": None,
    "n12_role": "localization_only",
    "negative_result_interpretation": "inconclusive",
    "equivalence_conclusion": None,
    "formal_cutoff_selection": None,
    "formal_sample_count_selection": None,
    "downstream_release": False,
    "allowed_diagnostic_verdicts": [
        "EXPLORATORY_RISK_SIGNAL",
        "NO_LARGE_SIGNAL_INCONCLUSIVE",
        "INCOMPLETE",
    ],
    "required_followup_for_any_signal": [
        "physics_repair",
        "powered_confirmation",
    ],
}
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
    "backend_b_bridge": "physics.phase9_backend_b_logical_bridge",
    "dual_backend_kernel": ("cnn_fpga.benchmark.phase9_dual_backend_qualification"),
    "fresh_runner": "cnn_fpga.benchmark.phase9_fresh_twin_qualification",
    "iq_reference": "physics.phase9_iq_likelihood_reference",
    "twin_contract": "physics.phase9_twin_contract",
}
runner: Any = None
_VERIFIED_EXECUTION_BINDINGS: dict[str, dict[str, object]] = {}
_VERIFIED_EXECUTION_MODULES: dict[str, object] = {}


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


def _validate_release_receipt(
    receipt: Mapping[str, Any],
    *,
    pending_binding: Mapping[str, Any],
    report_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
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
            "narrow_scope",
            "qualified_claim",
            "claim_state",
            "analysis_sha256",
        }
        or receipt.get("task_id") != TASK_ID
        or receipt.get("schema_version") != RELEASE_RECEIPT_SCHEMA
        or receipt.get("authorization_state") != NARROW_AUTHORIZATION_STATE
        or receipt.get("pending_parent") != pending_binding
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
        "pin_patch",
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
    if (
        not isinstance(pending_binding, Mapping)
        or not isinstance(receipt_binding, Mapping)
        or not isinstance(hardened, Mapping)
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
        or pins.get("report", {}).get("bytes") is not None
        or pins.get("report", {}).get("sha256") is not None
        or pins.get("source_data", {}).get("bytes") is not None
        or pins.get("source_data", {}).get("sha256") is not None
        or pins.get("required_analysis_sha256") is not None
    ):
        raise RuntimeError("pending high-cutoff parent pins are not immutable nulls")
    patch = child.get("pin_patch")
    if (
        not isinstance(patch, Mapping)
        or set(patch) != {"report", "source_data", "required_analysis_sha256"}
        or not isinstance(patch.get("report"), Mapping)
        or set(patch["report"]) != {"bytes", "sha256"}
        or not isinstance(patch.get("source_data"), Mapping)
        or set(patch["source_data"]) != {"bytes", "sha256"}
        or patch["report"]
        != {
            "bytes": report_binding["bytes"],
            "sha256": report_binding["sha256"],
        }
        or patch["source_data"]
        != {
            "bytes": source_binding["bytes"],
            "sha256": source_binding["sha256"],
        }
        or patch.get("required_analysis_sha256")
        != HARDENED_CONFIRMATION_ANALYSIS_SHA256
    ):
        raise RuntimeError("released high-cutoff pin patch whitelist drift")

    materialized = json.loads(json.dumps(pending))
    materialized_pins = materialized["hardened_confirmation_source"]
    materialized_pins["report"].update(patch["report"])
    materialized_pins["source_data"].update(patch["source_data"])
    materialized_pins["required_analysis_sha256"] = patch["required_analysis_sha256"]
    expected_differences = {
        ("hardened_confirmation_source", "report", "bytes"),
        ("hardened_confirmation_source", "report", "sha256"),
        ("hardened_confirmation_source", "source_data", "bytes"),
        ("hardened_confirmation_source", "source_data", "sha256"),
        ("hardened_confirmation_source", "required_analysis_sha256"),
    }
    if _leaf_differences(pending, materialized) != expected_differences:
        raise RuntimeError("released child changed non-whitelisted parent fields")

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
        "authorization_state": NARROW_AUTHORIZATION_STATE,
        "narrow_scope": dict(NARROW_SCOPE),
    }
    return ReleasedPilotConfig(materialized, release_lineage=lineage)


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
