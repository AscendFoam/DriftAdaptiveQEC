"""Audited validation-cap extension for the frozen Phase-9 physics kernels.

The previously qualified backend-A/backend-B source bytes cap construction at
cutoff 32.  The numerical kernels themselves derive all oscillator and joint
dimensions from ``config.cutoff``.  This adapter changes only the two verified
module objects' validation caps from 32 to 36 inside the dedicated
cutoff-32/36 process.  It does not edit either backend source, replace a
transition kernel, or retroactively alter prior evidence.

The caller must first load both backends from byte-pinned sources.  The
adapter refuses unknown modules, a non-native starting cap, an altered exact
Choi cap, missing verified-source attestations, or repeated activation with a
different extension.  Cutoff 37 remains rejected before simulator allocation.
"""

from __future__ import annotations

from hashlib import sha256
import json
from types import ModuleType
from typing import Any, Mapping

import numpy as np


ADAPTER_ID = "PHASE9-HIGH-CUTOFF-RUNTIME-VALIDATION-ADAPTER-V1"
NATIVE_MAX_SUPPORTED_CUTOFF = 32
EXTENDED_MAX_SUPPORTED_CUTOFF = 36
EXPECTED_EXACT_CHOI_CUTOFF = 8
CLAIM_BOUNDARY = {
    "runtime_validation_cap_extension_only": True,
    "physics_kernel_change": False,
    "twin_qualification": None,
    "ler": None,
    "lifetime": None,
    "physical_break_even": None,
    "official_puviani_exact": None,
    "puviani_nmf_surpass": None,
    "external_sota": None,
    "hardware_measured": None,
}


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


def _module_attestation(module: ModuleType, expected_name: str) -> dict[str, str]:
    digest = getattr(module, "__verified_source_sha256__", None)
    if (
        module.__name__ != expected_name
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"unverified high-cutoff backend module: {expected_name}")
    return {"module": expected_name, "verified_source_sha256": digest}


def _validate_native_contract(module: ModuleType, *, backend: str) -> None:
    if (
        getattr(module, "MAX_SUPPORTED_CUTOFF", None)
        != NATIVE_MAX_SUPPORTED_CUTOFF
        or getattr(module, "MAX_EXACT_CHOI_CUTOFF", None)
        != EXPECTED_EXACT_CHOI_CUTOFF
    ):
        raise RuntimeError(f"native backend-{backend} cutoff contract drift")


def _validate_extended_constructors(
    backend_a: ModuleType,
    backend_b: ModuleType,
) -> None:
    """Exercise config/state caps without constructing a simulator kernel."""

    for module, config_name, state_name in (
        (backend_a, "BackendAConfig", "BackendAState"),
        (backend_b, "BackendBConfig", "BackendBState"),
    ):
        config_type = getattr(module, config_name)
        state_type = getattr(module, state_name)
        config = config_type(cutoff=EXTENDED_MAX_SUPPORTED_CUTOFF)
        dimension = 3 * EXTENDED_MAX_SUPPORTED_CUTOFF
        state = state_type(
            np.eye(dimension, dtype=np.complex128) / dimension,
            EXTENDED_MAX_SUPPORTED_CUTOFF,
        )
        if (
            int(config.cutoff) != EXTENDED_MAX_SUPPORTED_CUTOFF
            or int(state.cutoff) != EXTENDED_MAX_SUPPORTED_CUTOFF
            or state.joint_density.shape != (dimension, dimension)
        ):
            raise RuntimeError("extended backend constructor contract drift")
        for constructor, arguments in (
            (config_type, {"cutoff": EXTENDED_MAX_SUPPORTED_CUTOFF + 1}),
            (
                state_type,
                {
                    "joint_density": np.eye(
                        dimension + 3, dtype=np.complex128
                    )
                    / (dimension + 3),
                    "cutoff": EXTENDED_MAX_SUPPORTED_CUTOFF + 1,
                },
            ),
        ):
            try:
                constructor(**arguments)
            except ValueError:
                pass
            else:
                raise RuntimeError("cutoff 37 was not rejected before allocation")


def enable_verified_high_cutoff(
    backend_a: ModuleType,
    backend_b: ModuleType,
) -> dict[str, Any]:
    """Enable cutoff 36 on exact, byte-attested backend module objects."""

    if not isinstance(backend_a, ModuleType) or not isinstance(
        backend_b, ModuleType
    ):
        raise TypeError("backend modules must be ModuleType instances")
    module_bindings = {
        "backend_a": _module_attestation(
            backend_a, "physics.phase9_backend_a"
        ),
        "backend_b": _module_attestation(
            backend_b, "physics.phase9_backend_b"
        ),
    }
    _validate_native_contract(backend_a, backend="A")
    _validate_native_contract(backend_b, backend="B")
    backend_a.MAX_SUPPORTED_CUTOFF = EXTENDED_MAX_SUPPORTED_CUTOFF
    backend_b.MAX_SUPPORTED_CUTOFF = EXTENDED_MAX_SUPPORTED_CUTOFF
    _validate_extended_constructors(backend_a, backend_b)
    receipt: dict[str, Any] = {
        "adapter_id": ADAPTER_ID,
        "native_max_supported_cutoff": NATIVE_MAX_SUPPORTED_CUTOFF,
        "extended_max_supported_cutoff": EXTENDED_MAX_SUPPORTED_CUTOFF,
        "exact_choi_cutoff_unchanged": EXPECTED_EXACT_CHOI_CUTOFF,
        "module_bindings": module_bindings,
        "numerical_kernel_source_changed": False,
        "cutoff_37_rejected": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    receipt["analysis_sha256"] = _sha(receipt)
    return receipt


def assert_verified_high_cutoff(
    backend_a: ModuleType,
    backend_b: ModuleType,
    receipt: Mapping[str, Any],
) -> None:
    unsigned = dict(receipt)
    analysis = unsigned.pop("analysis_sha256", None)
    if (
        analysis != _sha(unsigned)
        or receipt.get("adapter_id") != ADAPTER_ID
        or receipt.get("native_max_supported_cutoff")
        != NATIVE_MAX_SUPPORTED_CUTOFF
        or receipt.get("extended_max_supported_cutoff")
        != EXTENDED_MAX_SUPPORTED_CUTOFF
        or receipt.get("exact_choi_cutoff_unchanged")
        != EXPECTED_EXACT_CHOI_CUTOFF
        or receipt.get("numerical_kernel_source_changed") is not False
        or receipt.get("cutoff_37_rejected") is not True
        or receipt.get("claim_boundary") != CLAIM_BOUNDARY
        or getattr(backend_a, "MAX_SUPPORTED_CUTOFF", None)
        != EXTENDED_MAX_SUPPORTED_CUTOFF
        or getattr(backend_b, "MAX_SUPPORTED_CUTOFF", None)
        != EXTENDED_MAX_SUPPORTED_CUTOFF
        or getattr(backend_a, "MAX_EXACT_CHOI_CUTOFF", None)
        != EXPECTED_EXACT_CHOI_CUTOFF
        or getattr(backend_b, "MAX_EXACT_CHOI_CUTOFF", None)
        != EXPECTED_EXACT_CHOI_CUTOFF
    ):
        raise RuntimeError("high-cutoff runtime adapter attestation drift")
    live_bindings = {
        "backend_a": _module_attestation(
            backend_a, "physics.phase9_backend_a"
        ),
        "backend_b": _module_attestation(
            backend_b, "physics.phase9_backend_b"
        ),
    }
    if receipt.get("module_bindings") != live_bindings:
        raise RuntimeError("high-cutoff backend source binding drift")


__all__ = [
    "ADAPTER_ID",
    "CLAIM_BOUNDARY",
    "EXTENDED_MAX_SUPPORTED_CUTOFF",
    "NATIVE_MAX_SUPPORTED_CUTOFF",
    "assert_verified_high_cutoff",
    "enable_verified_high_cutoff",
]
