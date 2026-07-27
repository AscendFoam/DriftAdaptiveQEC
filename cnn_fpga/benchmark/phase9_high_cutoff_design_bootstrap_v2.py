"""Trusted-operator launch guard for the repaired high-cutoff diagnostic.

The pilot and diagnostic must never be imported from mutable project files by
the preregistered execution path. This bootstrap reads their exact bytes,
verifies SHA-256 before compilation, and executes the frozen bytes through a
dedicated loader. Production workers are threads inside this verified
interpreter; no child interpreter re-imports mutable project modules.

Threat boundary: the OS-level operator invoking the preregistered ``-I -S -c``
command is trusted. This guard detects accidental source drift, target-module
preload, and source TOCTOU within that path. It is not a sandbox, a
cryptographic process-origin attestation, or a defense against arbitrary code
that already controls the interpreter before this file executes.
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


TASK_ID = "T-RISK-20260727-01"
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
_SHA256_INITIAL = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
_SHA256_ROUND = (
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
)
EXTERNAL_LAUNCHER_SOURCE = (
    "import hashlib,json,os,pathlib,sys\n"
    "root=pathlib.Path(sys.argv[1]).resolve()\n"
    "expected=sys.argv[2]\n"
    "launcher_sha=sys.argv[3]\n"
    "mode=sys.argv[4]\n"
    "actual_source=sys.orig_argv[sys.orig_argv.index('-c')+1]\n"
    "assert hashlib.sha256(actual_source.encode('utf-8')).hexdigest()==launcher_sha\n"
    "path=root/'cnn_fpga/benchmark/phase9_high_cutoff_design_bootstrap_v2.py'\n"
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
    "'__verified_external_launcher_source__':actual_source,"
    "'__verified_external_launcher_flags__':('-I','-S'),"
    "'__verified_bootstrap_source_binding__':binding}\n"
    "sys.argv=[str(path),mode]\n"
    "exec(compile(payload,str(path),'exec',dont_inherit=True),namespace)\n"
)
EXTERNAL_LAUNCHER_SHA256 = (
    "0d9561d3faa0292e69cf49408069eae465a1750d584be3467d306e617e804dff"
)
RELEASED_CHILD_PATH = (
    "configs/phase9/" "t_risk_20260727_01_high_cutoff_design_pilot_fresh3_released.json"
)
RELEASED_CHILD_BYTES = 2821
RELEASED_CHILD_SHA256 = (
    "e8e301e0ac2f718b1a51839adb8ccf8de929af5c23a73d5883f6853e60f89a61"
)
PILOT_MODULE = "cnn_fpga.benchmark.phase9_high_cutoff_design_pilot"
PILOT_PATH = "cnn_fpga/benchmark/phase9_high_cutoff_design_pilot.py"
PILOT_SHA256 = "57e9fe8a9541aa76c577a5418423e96806b0ef883330a83628eb45e5649d01c4"
DIAGNOSTIC_MODULE = "cnn_fpga.benchmark.phase9_high_cutoff_design_diagnostic"
DIAGNOSTIC_PATH = "cnn_fpga/benchmark/phase9_high_cutoff_design_diagnostic.py"
DIAGNOSTIC_SHA256 = "94d35b1a3913b1ade3a41deb0a4fa4a046028aec212ecedea645325065e2b480"
PILOT_LAUNCH_META_PATH = (
    "runs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3/"
    "verified_pilot_launch_meta.json"
)
DIAGNOSTIC_LAUNCH_META_PATH = (
    "runs/t_risk_20260727_01_high_cutoff_design_pilot_fresh3/"
    "verified_diagnostic_v2_launch_meta.json"
)
PROBE_LAUNCH_META_PATH = (
    "runs/t_risk_20260727_01_test_audit/" "verified_external_probe_launch_meta.json"
)


def _rotate_right(value: int, amount: int) -> int:
    return ((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF


def _sha256_bytes(payload: bytes) -> str:
    """Independent SHA-256 used before any project module is trusted."""

    message = bytearray(payload)
    bit_length = len(message) * 8
    message.append(0x80)
    while len(message) % 64 != 56:
        message.append(0)
    message.extend(bit_length.to_bytes(8, "big"))
    state = list(_SHA256_INITIAL)
    for offset in range(0, len(message), 64):
        words = [
            int.from_bytes(message[index : index + 4], "big")
            for index in range(offset, offset + 64, 4)
        ]
        for index in range(16, 64):
            left = words[index - 15]
            right = words[index - 2]
            sigma0 = _rotate_right(left, 7) ^ _rotate_right(left, 18) ^ (left >> 3)
            sigma1 = _rotate_right(right, 17) ^ _rotate_right(right, 19) ^ (right >> 10)
            words.append(
                (words[index - 16] + sigma0 + words[index - 7] + sigma1) & 0xFFFFFFFF
            )
        a, b, c, d, e, f, g, h = state
        for index in range(64):
            choose = (e & f) ^ ((~e) & g)
            majority = (a & b) ^ (a & c) ^ (b & c)
            sum0 = _rotate_right(a, 2) ^ _rotate_right(a, 13) ^ _rotate_right(a, 22)
            sum1 = _rotate_right(e, 6) ^ _rotate_right(e, 11) ^ _rotate_right(e, 25)
            temporary1 = (
                h + sum1 + choose + _SHA256_ROUND[index] + words[index]
            ) & 0xFFFFFFFF
            temporary2 = (sum0 + majority) & 0xFFFFFFFF
            h, g, f, e, d, c, b, a = (
                g,
                f,
                e,
                (d + temporary1) & 0xFFFFFFFF,
                c,
                b,
                a,
                (temporary1 + temporary2) & 0xFFFFFFFF,
            )
        state = [
            (current + update) & 0xFFFFFFFF
            for current, update in zip(state, (a, b, c, d, e, f, g, h), strict=True)
        ]
    return "".join(f"{value:08x}" for value in state)


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


def _verify_release_root(root: Path) -> None:
    path = (root / RELEASED_CHILD_PATH).resolve()
    payload = path.read_bytes()
    if (
        len(payload) != RELEASED_CHILD_BYTES
        or _sha256_bytes(payload) != RELEASED_CHILD_SHA256
    ):
        raise RuntimeError("verified bootstrap released-child byte drift")
    child = json.loads(payload)
    if not isinstance(child, dict):
        raise RuntimeError("verified bootstrap released child is not an object")
    unsigned = dict(child)
    analysis = unsigned.pop("analysis_sha256", None)
    if (
        child.get("task_id") != TASK_ID
        or analysis != _sha256_bytes(_canonical(unsigned))
        or child.get("authorization_state")
        != "NARROW_UNPOWERED_EXPLORATORY_LOCALIZATION_ONLY"
        or child.get("qualified_claim") is not None
    ):
        raise RuntimeError("verified bootstrap released-child semantic drift")


def _external_launcher_contract(mode: str) -> tuple[str, dict[str, object]]:
    launcher_sha256 = globals().get("__verified_external_launcher_sha256__")
    launcher_source = globals().get("__verified_external_launcher_source__")
    bootstrap_binding = globals().get("__verified_bootstrap_source_binding__")
    isolation_flags = globals().get("__verified_external_launcher_flags__")
    source_sha256 = globals().get("__verified_source_sha256__")
    root = _root()
    bootstrap_path = Path(__file__).resolve()
    live_binding = _binding(bootstrap_path, root)
    expected_argv_tail = [
        "-I",
        "-S",
        "-c",
        EXTERNAL_LAUNCHER_SOURCE,
        str(root),
        str(source_sha256),
        EXTERNAL_LAUNCHER_SHA256,
        mode,
    ]
    if (
        mode not in {"pilot", "diagnostic", "probe"}
        or launcher_sha256 != EXTERNAL_LAUNCHER_SHA256
        or launcher_source != EXTERNAL_LAUNCHER_SOURCE
        or _sha256_bytes(EXTERNAL_LAUNCHER_SOURCE.encode("utf-8"))
        != EXTERNAL_LAUNCHER_SHA256
        or not isinstance(bootstrap_binding, dict)
        or set(bootstrap_binding) != {"path", "bytes", "sha256"}
        or bootstrap_binding != live_binding
        or source_sha256 != live_binding["sha256"]
        or isolation_flags != ("-I", "-S")
        or list(sys.orig_argv[1:]) != expected_argv_tail
    ):
        raise RuntimeError(
            "bootstrap must use the preregistered trusted-operator isolated launcher"
        )
    return launcher_sha256, dict(bootstrap_binding)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _commit_launch_meta(
    mode: str,
) -> tuple[dict[str, object], dict[str, object]]:
    launcher_sha256, bootstrap_binding = _external_launcher_contract(mode)
    relative = {
        "pilot": PILOT_LAUNCH_META_PATH,
        "diagnostic": DIAGNOSTIC_LAUNCH_META_PATH,
        "probe": PROBE_LAUNCH_META_PATH,
    }.get(mode)
    if relative is None:
        raise RuntimeError("unsupported verified launch-meta mode")
    payload: dict[str, object] = {
        "task_id": TASK_ID,
        "schema_version": LAUNCH_META_SCHEMA,
        "mode": mode,
        "external_launcher_sha256": launcher_sha256,
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
    if path.is_file():
        if path.read_bytes() != encoded:
            raise RuntimeError("verified launch meta drift")
    else:
        _atomic_bytes(path, encoded)
    return _binding(path, _ROOT), payload


class _FrozenModuleLoader(importlib.abc.Loader):
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


class _SingleFrozenFinder(importlib.abc.MetaPathFinder):
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

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> object:
        if fullname != self.fullname:
            return None
        loader = _FrozenModuleLoader(
            fullname,
            self.path,
            self.payload,
            self.digest,
        )
        return importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=str(self.path),
        )


def _drop_preloaded_module(fullname: str) -> None:
    module = sys.modules.pop(fullname, None)
    if module is None or "." not in fullname:
        return
    parent_name, attribute = fullname.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None and getattr(parent, attribute, None) is module:
        delattr(parent, attribute)


def _load_verified_module(
    root: Path,
    fullname: str,
    relative_path: str,
    expected_sha256: str,
) -> Any:
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("verified bootstrap source escapes root") from exc
    payload = path.read_bytes()
    digest = _sha256_bytes(payload)
    if digest != expected_sha256:
        raise RuntimeError(f"verified bootstrap source byte drift: {fullname}")
    _drop_preloaded_module(fullname)
    finder = _SingleFrozenFinder(fullname, path, payload, digest)
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module(fullname)
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
    if (
        getattr(module, "__verified_source_sha256__", None) != digest
        or getattr(module, "__verified_bootstrap_contract__", None)
        != VERIFIED_LOADER_CONTRACT
    ):
        raise RuntimeError(f"verified bootstrap import attestation drift: {fullname}")
    launcher_sha256 = globals().get("__verified_external_launcher_sha256__")
    bootstrap_binding = globals().get("__verified_bootstrap_source_binding__")
    if isinstance(launcher_sha256, str) and isinstance(bootstrap_binding, dict):
        module.__verified_external_launcher_sha256__ = launcher_sha256
        module.__verified_bootstrap_source_binding__ = dict(bootstrap_binding)
    return module


_ROOT = _root()
_PILOT: Any = None


def _initialize_verified_pilot(mode: str) -> Any:
    global _PILOT

    _external_launcher_contract(mode)
    _verify_release_root(_ROOT)
    if _PILOT is None:
        _PILOT = _load_verified_module(
            _ROOT,
            PILOT_MODULE,
            PILOT_PATH,
            PILOT_SHA256,
        )
    return _PILOT


def _attestation_probe() -> dict[str, object]:
    launcher_sha256, bootstrap_binding = _external_launcher_contract("probe")
    launch_binding, launch_payload = _commit_launch_meta("probe")
    _PILOT.__verified_launch_meta_binding__ = launch_binding
    _PILOT.__verified_launch_meta_payload__ = launch_payload
    try:
        _PILOT._require_verified_self_import(expected_mode="probe")
        pilot, _base = _PILOT.load_pilot_config(_ROOT)
        snapshot = _PILOT._build_input_snapshot(_ROOT, pilot)
        _PILOT._activate_verified_execution_modules(_ROOT, snapshot)
        _PILOT._assert_input_snapshot(_ROOT, snapshot)
        with _PILOT.ThreadPoolExecutor(max_workers=6) as executor:
            thread_attestations = list(
                executor.map(
                    _PILOT._verified_thread_worker_probe,
                    [str(_ROOT)] * 6,
                    [snapshot] * 6,
                    ["probe"] * 6,
                )
            )
        expected_runner_sha256 = snapshot["source/fresh_runner"]["sha256"]
        if len(thread_attestations) != 6 or any(
            attestation["execution_module_count"] != len(_PILOT._EXECUTION_MODULE_NAMES)
            or attestation["fresh_runner_sha256"] != expected_runner_sha256
            for attestation in thread_attestations
        ):
            raise RuntimeError("verified thread-worker attestation drift")
        return {
            "pilot_sha256": getattr(_PILOT, "__verified_source_sha256__", None),
            "external_launcher_sha256": launcher_sha256,
            "bootstrap_sha256": bootstrap_binding["sha256"],
            "launcher_assurance": dict(LAUNCHER_ASSURANCE),
            "bootstrap_contract": getattr(
                _PILOT,
                "__verified_bootstrap_contract__",
                None,
            ),
            "execution_module_count": len(_PILOT._VERIFIED_EXECUTION_MODULES),
            "fresh_runner_sha256": getattr(
                _PILOT.runner,
                "__verified_source_sha256__",
                None,
            ),
            "expected_fresh_runner_sha256": snapshot["source/fresh_runner"]["sha256"],
            "thread_worker_attestations": len(thread_attestations),
        }
    finally:
        (_ROOT / PROBE_LAUNCH_META_PATH).unlink(missing_ok=True)


def _install_terminal_quantization_parser(diagnostic: object) -> None:
    """Install the V2 parser while keeping the V1 diagnostic bytes immutable.

    The pilot ledger intentionally emits the density quantization certificate
    only on terminal rows, because only those rows have archived densities.
    V1 attempted ``float("")`` on every non-terminal row.  This replacement
    enforces both sides of the schema: terminal rows require a finite,
    non-negative certificate and non-terminal rows must leave it blank.
    """

    def parse_chunk(
        root: Path,
        receipt: dict[str, Any],
        receipt_binding: dict[str, Any],
        *,
        config: dict[str, Any],
        manifest: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, object]]:
        diagnostic._verify_self_hash(receipt, "pilot receipt")
        cell_payload = receipt.get("cell")
        if not isinstance(cell_payload, dict):
            raise ValueError("pilot receipt cell missing")
        chunk_id = str(cell_payload.get("chunk_id"))
        expected_receipt_path = (
            root
            / str(config["artifact_paths"]["receipt_directory"])
            / f"{chunk_id}.json"
        )
        live_receipt_path, live_receipt = (
            diagnostic.pilot_runner._read_bound_json(
                root,
                receipt_binding,
                expected_path=expected_receipt_path.relative_to(root).as_posix(),
            )
        )
        if live_receipt_path != expected_receipt_path.resolve():
            raise ValueError("pilot receipt path identity drift")
        if live_receipt != receipt:
            raise ValueError("pilot manifest/live receipt drift")
        if (
            receipt.get("task_id") != TASK_ID
            or receipt.get("schema_version")
            != diagnostic.pilot_runner.RECEIPT_SCHEMA
            or receipt.get("run_id") != manifest.get("run_id")
            or receipt.get("run_identity_analysis_sha256")
            != manifest.get("run_identity_analysis_sha256")
            or receipt.get("config_analysis_sha256")
            != manifest.get("config_analysis_sha256")
            or receipt.get("execution_analysis_sha256")
            != manifest.get("execution_analysis_sha256")
            or receipt.get("input_snapshot_analysis_sha256")
            != manifest.get("input_snapshot_analysis_sha256")
            or receipt.get("pilot_source_sha256")
            != manifest.get("pilot_source_sha256")
            or receipt.get("chunk_id") != chunk_id
        ):
            raise ValueError("pilot receipt identity drift")
        _, csv_bytes = diagnostic.pilot_runner._read_bound_bytes(
            root, receipt["csv"]
        )
        _, npz_bytes = diagnostic.pilot_runner._read_bound_bytes(
            root, receipt["npz"]
        )
        rows: list[dict[str, Any]] = []
        with diagnostic.io.StringIO(
            csv_bytes.decode("utf-8"), newline=""
        ) as stream:
            for raw in diagnostic.csv.DictReader(stream):
                if raw["exception_type"]:
                    raise ValueError("pilot contains exception row")
                if raw["conservation_pass"] != "True":
                    raise ValueError("pilot contains conservation failure")
                terminal_text = raw["terminal_round"].lower()
                if terminal_text not in {"true", "false"}:
                    raise ValueError("pilot terminal_round is not a strict boolean")
                terminal_round = terminal_text == "true"
                quantization_text = raw[
                    "density_quantization_trace_distance_bound"
                ]
                if terminal_round:
                    if not quantization_text:
                        raise ValueError(
                            "terminal density quantization certificate missing"
                        )
                    quantization_bound = float(quantization_text)
                    if (
                        not diagnostic.np.isfinite(quantization_bound)
                        or quantization_bound < 0.0
                    ):
                        raise ValueError(
                            "terminal density quantization certificate invalid"
                        )
                else:
                    if quantization_text:
                        raise ValueError(
                            "non-terminal density quantization certificate present"
                        )
                    quantization_bound = None
                rows.append(
                    {
                        **raw,
                        "cutoff": int(raw["cutoff"]),
                        "seed_position": int(raw["seed_position"]),
                        "round_index": int(raw["round_index"]),
                        "terminal_round": terminal_round,
                        "mean_photon": float(raw["mean_photon"]),
                        "level_g": float(raw["level_g"]),
                        "level_e": float(raw["level_e"]),
                        "level_f": float(raw["level_f"]),
                        "logical_survival": float(raw["logical_survival"]),
                        "density_quantization_trace_distance_bound": (
                            quantization_bound
                        ),
                    }
                )
        with diagnostic.np.load(
            diagnostic.io.BytesIO(npz_bytes), allow_pickle=False
        ) as archive:
            density_ids = [
                str(value) for value in archive["density_row_ids"].tolist()
            ]
            densities = diagnostic.np.asarray(
                archive["densities"], dtype=diagnostic.np.complex128
            )
        if len(density_ids) != len(densities):
            raise ValueError("pilot density row alignment drift")
        if len(densities):
            hermitian_error = float(
                diagnostic.np.max(
                    diagnostic.np.abs(
                        densities - densities.conj().transpose(0, 2, 1)
                    )
                )
            )
            trace_error = float(
                diagnostic.np.max(
                    diagnostic.np.abs(
                        diagnostic.np.trace(densities, axis1=1, axis2=2) - 1.0
                    )
                )
            )
            minimum_eigenvalue = min(
                float(
                    diagnostic.np.linalg.eigvalsh(
                        (matrix + matrix.conj().T) / 2
                    ).min()
                )
                for matrix in densities
            )
            if (
                hermitian_error > 5e-5
                or trace_error > 5e-5
                or minimum_eigenvalue < -5e-5
            ):
                raise ValueError("pilot density physicality drift")
        return rows, dict(zip(density_ids, densities, strict=True))

    diagnostic._parse_chunk = parse_chunk


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verified entrypoint for the high-cutoff pilot or diagnostic."
    )
    parser.add_argument(
        "mode",
        choices=("pilot", "diagnostic", "probe"),
        nargs="?",
        default="pilot",
    )
    args = parser.parse_args(argv)
    _initialize_verified_pilot(args.mode)
    if args.mode == "probe":
        print(json.dumps(_attestation_probe(), sort_keys=True))
        return 0
    if args.mode == "pilot":
        launch_binding, launch_payload = _commit_launch_meta("pilot")
        _PILOT.__verified_launch_meta_binding__ = launch_binding
        _PILOT.__verified_launch_meta_payload__ = launch_payload
        _PILOT._require_verified_self_import()
        return int(_PILOT.main([]))
    diagnostic = _load_verified_module(
        _ROOT,
        DIAGNOSTIC_MODULE,
        DIAGNOSTIC_PATH,
        DIAGNOSTIC_SHA256,
    )
    launch_binding, launch_payload = _commit_launch_meta("diagnostic")
    diagnostic.__verified_launch_meta_binding__ = launch_binding
    diagnostic.__verified_launch_meta_payload__ = launch_payload
    diagnostic._require_verified_self_import()
    _install_terminal_quantization_parser(diagnostic)
    return int(diagnostic.main([]))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
