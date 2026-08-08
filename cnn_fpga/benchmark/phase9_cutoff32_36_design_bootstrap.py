"""Verified trusted-operator entrypoint for cutoff32/36 design and diagnostic.

The bootstrap is loaded by an isolated ``python -I -S -c`` literal.  That
literal reads these exact bytes once, verifies SHA-256 before compilation, and
then this module loads the released child, design runner and diagnostic from
their preregistered bytes.  Production workers remain threads in the same
verified interpreter.

The trusted OS operator is in scope.  Arbitrary code already controlling that
operator or interpreter before launch is explicitly out of scope; this is
source-drift/TOCTOU protection, not process-origin attestation.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence


TASK_ID = "T-RISK-20260728-01"
VERIFIED_LOADER_CONTRACT = "PHASE9-VERIFIED-SOURCE-BYTES-LOADER-V1"
LAUNCH_META_SCHEMA = "PHASE9-TRUSTED-OPERATOR-LAUNCH-META-V2"
LAUNCHER_ASSURANCE = {
    "scope": "trusted_operator_preregistered_command_and_accidental_drift",
    "trusted_operator_required": True,
    "preexecution_arbitrary_code": "OUT_OF_SCOPE",
    "adversarial_local_operator_resistance": None,
    "cryptographic_process_origin_attestation": None,
    "os_native_signed_launcher_receipt": None,
}
EXTERNAL_LAUNCHER_SOURCE = (
    "import hashlib,json,os,pathlib,sys\n"
    "root=pathlib.Path(sys.argv[1]).resolve()\n"
    "expected=sys.argv[2]\n"
    "launcher_sha=sys.argv[3]\n"
    "mode=sys.argv[4]\n"
    "actual=sys.orig_argv[sys.orig_argv.index('-c')+1]\n"
    "assert hashlib.sha256(actual.encode('utf-8')).hexdigest()==launcher_sha\n"
    "path=root/'cnn_fpga/benchmark/phase9_cutoff32_36_design_bootstrap.py'\n"
    "payload=path.read_bytes()\n"
    "assert hashlib.sha256(payload).hexdigest()==expected\n"
    "sys.path.insert(0,str(root))\n"
    "sys.path.append(str(pathlib.Path(sys.base_prefix)/'Lib'/'site-packages'))\n"
    "dll=pathlib.Path(sys.base_prefix)/'Library'/'bin'\n"
    "dll_handle=os.add_dll_directory(str(dll)) if dll.is_dir() else None\n"
    "binding={'path':path.relative_to(root).as_posix(),"
    "'bytes':len(payload),'sha256':expected}\n"
    "namespace={'__name__':'__main__','__file__':str(path),"
    "'__package__':'cnn_fpga.benchmark',"
    "'__verified_source_sha256__':expected,"
    "'__verified_external_launcher_sha256__':launcher_sha,"
    "'__verified_external_launcher_source__':actual,"
    "'__verified_external_launcher_flags__':('-I','-S'),"
    "'__verified_bootstrap_source_binding__':binding}\n"
    "sys.argv=[str(path),mode]\n"
    "exec(compile(payload,str(path),'exec',dont_inherit=True),namespace)\n"
)
EXTERNAL_LAUNCHER_SHA256 = (
    "ca1b63693509f00bb2270776ded73e58c927c08569747f500468e05022663892"
)
RELEASED_CHILD_PATH = (
    "configs/phase9/"
    "t_risk_20260728_01_cutoff32_36_design_extension_released.json"
)
RELEASED_CHILD_BYTES = 4907
RELEASED_CHILD_SHA256 = (
    "a84e91a771116d1ee21072796fc4fd72613118ef90416a7c38ee91a08fd372f0"
)
RUNNER_MODULE = (
    "cnn_fpga.benchmark.phase9_cutoff32_36_design_extension"
)
RUNNER_PATH = "cnn_fpga/benchmark/phase9_cutoff32_36_design_extension.py"
RUNNER_SHA256 = (
    "b00f13141c92fff992ae90f487497a4d8da3b36698fa4584d1a2aa0c3c75d8c3"
)
DIAGNOSTIC_MODULE = (
    "cnn_fpga.benchmark.phase9_cutoff32_36_design_diagnostic"
)
DIAGNOSTIC_PATH = (
    "cnn_fpga/benchmark/phase9_cutoff32_36_design_diagnostic.py"
)
DIAGNOSTIC_SHA256 = (
    "616d87b18b2ff33471af54e104ad4b54c27d06846b2b9bb8f911ee0f25e22ebe"
)
PILOT_LAUNCH_META_PATH = (
    "runs/t_risk_20260728_01_cutoff32_36_design_extension_fresh2/"
    "verified_pilot_launch_meta.json"
)
DIAGNOSTIC_LAUNCH_META_PATH = (
    "runs/t_risk_20260728_01_cutoff32_36_design_extension_fresh2/"
    "verified_diagnostic_launch_meta.json"
)
PROBE_LAUNCH_META_PATH = (
    "runs/t_risk_20260728_01_cutoff32_36_design_probe/"
    "verified_probe_launch_meta.json"
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


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _external_launcher_contract(
    mode: str,
) -> tuple[str, dict[str, object]]:
    launcher_sha = globals().get("__verified_external_launcher_sha256__")
    launcher_source = globals().get("__verified_external_launcher_source__")
    bootstrap_binding = globals().get("__verified_bootstrap_source_binding__")
    isolation_flags = globals().get("__verified_external_launcher_flags__")
    source_sha = globals().get("__verified_source_sha256__")
    root = _root()
    live_binding = _binding(Path(__file__).resolve(), root)
    expected_tail = [
        "-I",
        "-S",
        "-c",
        EXTERNAL_LAUNCHER_SOURCE,
        str(root),
        str(source_sha),
        EXTERNAL_LAUNCHER_SHA256,
        mode,
    ]
    if (
        mode not in {"pilot", "diagnostic", "probe"}
        or launcher_sha != EXTERNAL_LAUNCHER_SHA256
        or launcher_source != EXTERNAL_LAUNCHER_SOURCE
        or _sha256_bytes(EXTERNAL_LAUNCHER_SOURCE.encode("utf-8"))
        != EXTERNAL_LAUNCHER_SHA256
        or not isinstance(bootstrap_binding, dict)
        or bootstrap_binding != live_binding
        or isolation_flags != ("-I", "-S")
        or list(sys.orig_argv[1:]) != expected_tail
    ):
        raise RuntimeError("bootstrap requires the exact isolated launcher")
    return str(launcher_sha), dict(bootstrap_binding)


def _verify_release_root(root: Path) -> None:
    binding = _binding(root / RELEASED_CHILD_PATH, root)
    if binding != {
        "path": RELEASED_CHILD_PATH,
        "bytes": RELEASED_CHILD_BYTES,
        "sha256": RELEASED_CHILD_SHA256,
    }:
        raise RuntimeError("bootstrap released-child byte drift")
    child = json.loads((root / RELEASED_CHILD_PATH).read_bytes())
    unsigned = dict(child)
    analysis = unsigned.pop("analysis_sha256", None)
    if (
        child.get("task_id") != TASK_ID
        or analysis != _sha256_bytes(_canonical(unsigned))
        or child.get("authorization_state")
        != "NARROW_UNPOWERED_CUTOFF32_36_DESIGN_ONLY"
        or child.get("qualified_claim") is not None
    ):
        raise RuntimeError("bootstrap released-child semantic drift")


def _commit_launch_meta(
    mode: str,
) -> tuple[dict[str, object], dict[str, object]]:
    launcher_sha, bootstrap_binding = _external_launcher_contract(mode)
    relative = {
        "pilot": PILOT_LAUNCH_META_PATH,
        "diagnostic": DIAGNOSTIC_LAUNCH_META_PATH,
        "probe": PROBE_LAUNCH_META_PATH,
    }[mode]
    payload: dict[str, object] = {
        "task_id": TASK_ID,
        "schema_version": LAUNCH_META_SCHEMA,
        "mode": mode,
        "external_launcher_sha256": launcher_sha,
        "launcher_assurance": dict(LAUNCHER_ASSURANCE),
        "isolation_flags": ["-I", "-S"],
        "bootstrap": bootstrap_binding,
        "bootstrap_load_protocol": "read_once_sha256_then_compile_exec",
        "child_process_policy": "same_verified_process_thread_workers_only",
        "qualified_claim": None,
        "downstream_release": False,
    }
    payload["analysis_sha256"] = _sha256_bytes(_canonical(payload))
    path = _ROOT / relative
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError("verified launch-meta drift")
    else:
        _atomic_bytes(path, encoded)
    return _binding(path, _ROOT), payload


class _FrozenLoader(importlib.abc.Loader):
    def __init__(self, fullname: str, path: Path, payload: bytes, digest: str):
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
        exec(
            compile(
                self.payload, str(self.path), "exec", dont_inherit=True
            ),
            namespace,
        )
        namespace["__verified_source_sha256__"] = self.digest
        namespace["__verified_bootstrap_contract__"] = VERIFIED_LOADER_CONTRACT


class _FrozenFinder(importlib.abc.MetaPathFinder):
    def __init__(self, fullname: str, path: Path, payload: bytes, digest: str):
        self.fullname = fullname
        self.path = path
        self.payload = payload
        self.digest = digest

    def find_spec(
        self, fullname: str, path: object = None, target: object = None
    ) -> object:
        if fullname != self.fullname:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _FrozenLoader(
                fullname, self.path, self.payload, self.digest
            ),
            origin=str(self.path),
        )


def _drop_module(fullname: str) -> None:
    module = sys.modules.pop(fullname, None)
    if module is None or "." not in fullname:
        return
    parent_name, attribute = fullname.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None and getattr(parent, attribute, None) is module:
        delattr(parent, attribute)


def _load_verified(
    fullname: str, relative: str, expected_sha256: str
) -> Any:
    path = (_ROOT / relative).resolve()
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest != expected_sha256:
        raise RuntimeError(f"verified source byte drift: {fullname}")
    _drop_module(fullname)
    finder = _FrozenFinder(fullname, path, payload, digest)
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module(fullname)
    finally:
        sys.meta_path.remove(finder)
    if (
        getattr(module, "__verified_source_sha256__", None) != digest
        or getattr(module, "__verified_bootstrap_contract__", None)
        != VERIFIED_LOADER_CONTRACT
    ):
        raise RuntimeError(f"verified source attestation drift: {fullname}")
    module.__verified_external_launcher_sha256__ = globals().get(
        "__verified_external_launcher_sha256__"
    )
    module.__verified_bootstrap_source_binding__ = dict(
        globals()["__verified_bootstrap_source_binding__"]
    )
    return module


_ROOT = _root()
_RUNNER: Any = None


def _initialize(mode: str) -> Any:
    global _RUNNER

    _external_launcher_contract(mode)
    _verify_release_root(_ROOT)
    if _RUNNER is None:
        _RUNNER = _load_verified(RUNNER_MODULE, RUNNER_PATH, RUNNER_SHA256)
    return _RUNNER


def _probe() -> dict[str, object]:
    launch_binding, launch_payload = _commit_launch_meta("probe")
    _RUNNER.__verified_launch_meta_binding__ = launch_binding
    _RUNNER.__verified_launch_meta_payload__ = launch_payload
    try:
        _RUNNER._require_verified_self_import(expected_mode="probe")
        config, _base = _RUNNER.load_pilot_config(
            _ROOT, require_hardened=True
        )
        snapshot = _RUNNER._build_input_snapshot(_ROOT, config)
        _RUNNER._activate_verified_execution_modules(_ROOT, snapshot)
        with _RUNNER.ThreadPoolExecutor(max_workers=6) as executor:
            attestations = list(
                executor.map(
                    _RUNNER._verified_thread_worker_probe,
                    [str(_ROOT)] * 6,
                    [snapshot] * 6,
                    ["probe"] * 6,
                )
            )
        if len(attestations) != 6:
            raise RuntimeError("verified worker probe denominator drift")
        return {
            "runner_sha256": getattr(
                _RUNNER, "__verified_source_sha256__", None
            ),
            "bootstrap_contract": getattr(
                _RUNNER, "__verified_bootstrap_contract__", None
            ),
            "worker_attestations": len(attestations),
            "high_cutoff_adapter_receipt": dict(
                _RUNNER._HIGH_CUTOFF_ADAPTER_RECEIPT
            ),
        }
    finally:
        (_ROOT / PROBE_LAUNCH_META_PATH).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verified cutoff32/36 design entrypoint."
    )
    parser.add_argument(
        "mode",
        choices=("pilot", "diagnostic", "probe"),
        nargs="?",
        default="pilot",
    )
    args = parser.parse_args(argv)
    _initialize(args.mode)
    if args.mode == "probe":
        print(json.dumps(_probe(), sort_keys=True))
        return 0
    if args.mode == "pilot":
        binding, payload = _commit_launch_meta("pilot")
        _RUNNER.__verified_launch_meta_binding__ = binding
        _RUNNER.__verified_launch_meta_payload__ = payload
        _RUNNER._require_verified_self_import()
        return int(_RUNNER.main([]))
    diagnostic = _load_verified(
        DIAGNOSTIC_MODULE, DIAGNOSTIC_PATH, DIAGNOSTIC_SHA256
    )
    binding, payload = _commit_launch_meta("diagnostic")
    diagnostic.__verified_launch_meta_binding__ = binding
    diagnostic.__verified_launch_meta_payload__ = payload
    diagnostic.pilot_runner = _RUNNER
    diagnostic._require_verified_self_import()
    return int(diagnostic.main([]))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EXTERNAL_LAUNCHER_SOURCE", "main"]
