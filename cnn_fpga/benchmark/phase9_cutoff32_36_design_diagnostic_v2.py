"""Hardened point-screen diagnostic for the cutoff-32/36 design extension.

This module consumes the sealed cutoff-28 fault reference and the fresh
cutoff-32/36 design transaction.  It evaluates every preregistered
state/scenario/backend/stage stratum plus the shared vacuum-f/RESET repair
cells.  The result is only a design screen:

* PASS authorizes launching the separately frozen, powered, full-fresh
  qualification;
* NO-GO requires a new physics/design transaction;
* neither result qualifies the twin or supports LER, lifetime, hardware,
  physical break-even, Puviani, SOTA or rank claims.

All density point estimates include the archived float32 trace-distance
quantization certificate.  Statistical UCBs are intentionally reserved for
the powered formal stage; this n=12/state screen never claims equivalence.

V2 preserves every V1 gate and margin.  It only repairs the live evidence
reader after the immutable V1 launch correctly failed closed:

* JSON bindings may carry the already-preregistered ``analysis_sha256`` and
  that semantic hash is independently checked;
* production NPZ archives are consumed with their complete eight-field schema
  and exact CSV/NPZ row, density, IQ, cutoff and chunk alignment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np


TASK_ID = "T-RISK-20260728-01"
SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-DIAGNOSTIC-V2"
COMPLETION_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-DIAGNOSTIC-COMPLETION-V2"
LOCK_SCHEMA = "PHASE9-CUTOFF32-36-DESIGN-DIAGNOSTIC-LOCK-V2"
LAUNCH_META_SCHEMA = "PHASE9-TRUSTED-OPERATOR-LAUNCH-META-V2"
VERIFIED_LOADER_CONTRACT = "PHASE9-VERIFIED-SOURCE-BYTES-LOADER-V1"
EXTERNAL_LAUNCHER_SHA256 = (
    "f95bc4e947416bd8461d7e1fb26ef6e9071f66243623116847230a369619252d"
)
PASS_VERDICT = "DESIGN_GATE_PASS_AUTHORIZES_POWERED_FORMAL"
NO_GO_VERDICT = "NO_GO_HIGH_CUTOFF_DESIGN"
INCOMPLETE_VERDICT = "INCOMPLETE"
STATUS = "CUTOFF32_36_DESIGN_DIAGNOSTIC_COMPLETE"
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
SOURCE_FIELDS = (
    "gate_id",
    "family",
    "contrast",
    "scenario",
    "logical_state",
    "stage",
    "metric",
    "cutoff_or_increment",
    "backend_or_pair",
    "estimate",
    "quantization_bound",
    "conservative_point",
    "margin",
    "passed",
    "cluster_count",
    "statistical_role",
    "qualification_effect",
)
_SOURCE_SHA256_AT_IMPORT = sha256(Path(__file__).read_bytes()).hexdigest()
pilot_runner: Any = None


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


def _self_hash(value: Mapping[str, Any], label: str) -> str:
    unsigned = dict(value)
    analysis = unsigned.pop("analysis_sha256", None)
    if not isinstance(analysis, str) or analysis != _sha(unsigned):
        raise RuntimeError(f"{label} self-hash drift")
    return analysis


def _binding(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }


def _read_bound_bytes(root: Path, binding: Mapping[str, Any]) -> tuple[Path, bytes]:
    base_fields = {"path", "bytes", "sha256"}
    fields = set(binding)
    if fields not in (base_fields, base_fields | {"analysis_sha256"}):
        raise RuntimeError("diagnostic binding schema drift")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError("diagnostic binding escapes repository") from exc
    payload = path.read_bytes()
    if (
        len(payload) != int(binding["bytes"])
        or sha256(payload).hexdigest() != binding["sha256"]
    ):
        raise RuntimeError(f"diagnostic bound artifact drift: {binding['path']}")
    if "analysis_sha256" in binding:
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "analysis-bound diagnostic artifact is not JSON"
            ) from exc
        if not isinstance(document, Mapping):
            raise RuntimeError(
                "analysis-bound diagnostic artifact is not a JSON object"
            )
        analysis = _self_hash(document, f"analysis-bound artifact {binding['path']}")
        if analysis != binding["analysis_sha256"]:
            raise RuntimeError(f"diagnostic semantic binding drift: {binding['path']}")
    return path, payload


def _require_verified_self_import() -> None:
    module = sys.modules.get(__name__)
    bootstrap = (
        None
        if module is None
        else getattr(module, "__verified_bootstrap_source_binding__", None)
    )
    launch_binding = (
        None
        if module is None
        else getattr(module, "__verified_launch_meta_binding__", None)
    )
    launch_payload = (
        None
        if module is None
        else getattr(module, "__verified_launch_meta_payload__", None)
    )
    if (
        module is None
        or getattr(module, "__verified_source_sha256__", None)
        != _SOURCE_SHA256_AT_IMPORT
        or getattr(module, "__verified_bootstrap_contract__", None)
        != VERIFIED_LOADER_CONTRACT
        or getattr(module, "__verified_external_launcher_sha256__", None)
        != EXTERNAL_LAUNCHER_SHA256
        or not isinstance(bootstrap, Mapping)
        or set(bootstrap) != {"path", "bytes", "sha256"}
        or not isinstance(launch_binding, Mapping)
        or set(launch_binding) != {"path", "bytes", "sha256"}
        or not isinstance(launch_payload, Mapping)
        or launch_payload.get("task_id") != TASK_ID
        or launch_payload.get("schema_version") != LAUNCH_META_SCHEMA
        or launch_payload.get("mode") != "diagnostic"
        or launch_payload.get("qualified_claim") is not None
        or launch_payload.get("downstream_release") is not False
        or launch_payload.get("analysis_sha256")
        != _sha(
            {
                key: value
                for key, value in launch_payload.items()
                if key != "analysis_sha256"
            }
        )
    ):
        raise RuntimeError(
            "diagnostic must be imported by the verified trusted-operator bootstrap"
        )
    if pilot_runner is None:
        raise RuntimeError("verified design runner was not injected")


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


@contextmanager
def _exclusive_lock(path: Path) -> Iterable[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": LOCK_SCHEMA,
        "owner_token": uuid4().hex,
        "pid": os.getpid(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                (
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        yield payload
    finally:
        live = json.loads(path.read_text(encoding="utf-8"))
        if live != payload:
            raise RuntimeError("diagnostic owner lock drift")
        path.unlink()


def _physicality(stack: np.ndarray, cutoff: int) -> None:
    value = np.asarray(stack, dtype=np.complex128)
    if value.ndim != 3 or value.shape[1:] != (3 * cutoff, 3 * cutoff):
        raise RuntimeError("diagnostic density shape drift")
    if not np.all(np.isfinite(value)):
        raise RuntimeError("diagnostic density contains NaN/Inf")
    if float(np.max(np.abs(value - value.conj().transpose(0, 2, 1)))) > 5e-5:
        raise RuntimeError("diagnostic density Hermiticity drift")
    if float(np.max(np.abs(np.trace(value, axis1=1, axis2=2) - 1.0))) > 5e-5:
        raise RuntimeError("diagnostic density trace drift")
    if (
        min(
            float(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min())
            for matrix in value
        )
        < -5e-5
    ):
        raise RuntimeError("diagnostic density positivity drift")


def _parse_receipt(
    root: Path,
    receipt: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    _self_hash(receipt, "raw receipt")
    _, receipt_bytes = _read_bound_bytes(root, receipt_binding)
    if json.loads(receipt_bytes) != receipt:
        raise RuntimeError("raw receipt binding/content mismatch")
    cell = receipt.get("cell")
    if not isinstance(cell, Mapping):
        raise RuntimeError("raw receipt cell missing")
    if (
        receipt.get("exception_rows") != 0
        or receipt.get("observed_rows") != receipt.get("expected_rows")
        or receipt.get("chunk_id") != cell.get("chunk_id")
    ):
        raise RuntimeError("raw receipt completeness drift")
    csv_binding = receipt.get("csv")
    npz_binding = receipt.get("npz")
    if not isinstance(csv_binding, Mapping) or not isinstance(npz_binding, Mapping):
        raise RuntimeError("raw receipt chunk binding missing")
    _, csv_payload = _read_bound_bytes(root, csv_binding)
    _, npz_payload = _read_bound_bytes(root, npz_binding)
    rows: list[dict[str, Any]] = []
    raw_row_ids: list[str] = []
    raw_row_id_set: set[str] = set()
    terminal_row_ids: list[str] = []
    with io.StringIO(csv_payload.decode("utf-8"), newline="") as stream:
        reader = csv.DictReader(stream)
        for archive_row_index, raw in enumerate(reader):
            if raw["exception_type"] or raw["conservation_pass"] != "True":
                raise RuntimeError("raw row exception/conservation failure")
            row_id = str(raw["row_id"])
            if (
                not row_id
                or row_id in raw_row_id_set
                or raw["archive_chunk"] != cell["chunk_id"]
                or int(raw["archive_row_index"]) != archive_row_index
                or int(raw["raw_iq_index"]) != archive_row_index
                or int(raw["heldout_iq_index"]) != archive_row_index
            ):
                raise RuntimeError("raw CSV archive alignment drift")
            raw_row_ids.append(row_id)
            raw_row_id_set.add(row_id)
            terminal_text = raw["terminal_round"].lower()
            if terminal_text not in {"true", "false"}:
                raise RuntimeError("raw terminal flag drift")
            terminal = terminal_text == "true"
            quantization_text = raw["density_quantization_trace_distance_bound"]
            if terminal:
                if not quantization_text:
                    raise RuntimeError("terminal density certificate missing")
                if int(raw["density_index"]) != len(terminal_row_ids):
                    raise RuntimeError("terminal density index drift")
                terminal_row_ids.append(row_id)
                quantization: float | None = float(quantization_text)
                if not np.isfinite(quantization) or quantization < 0.0:
                    raise RuntimeError("terminal density certificate invalid")
            else:
                if quantization_text:
                    raise RuntimeError("nonterminal density certificate must be blank")
                if int(raw["density_index"]) != -1:
                    raise RuntimeError("nonterminal density index drift")
                quantization = None
            logical_text = raw["logical_survival"]
            if raw["layer"] == "fault":
                if not logical_text:
                    raise RuntimeError("fault logical-survival value missing")
                logical_survival: float | None = float(logical_text)
                if (
                    not np.isfinite(logical_survival)
                    or logical_survival < -5e-6
                    or logical_survival > 1.0 + 5e-6
                ):
                    raise RuntimeError("fault logical-survival value invalid")
            elif raw["layer"] == "shared":
                if logical_text:
                    raise RuntimeError(
                        "shared logical survival must be density-derived"
                    )
                logical_survival = None
            else:
                raise RuntimeError("raw layer drift")
            rows.append(
                {
                    **raw,
                    "cutoff": int(raw["cutoff"]),
                    "seed_position": int(raw["seed_position"]),
                    "round_index": int(raw["round_index"]),
                    "terminal_round": terminal,
                    "mean_photon": float(raw["mean_photon"]),
                    "level_g": float(raw["level_g"]),
                    "level_e": float(raw["level_e"]),
                    "level_f": float(raw["level_f"]),
                    "logical_survival": logical_survival,
                    "density_quantization_trace_distance_bound": quantization,
                }
            )
    if len(rows) != int(receipt["expected_rows"]):
        raise RuntimeError("raw CSV denominator drift")
    with np.load(io.BytesIO(npz_payload), allow_pickle=False) as archive:
        expected_archive_fields = {
            "schema",
            "chunk_id",
            "cutoff",
            "row_ids",
            "density_row_ids",
            "densities",
            "raw_iq",
            "heldout_iq",
        }
        if set(archive.files) != expected_archive_fields:
            raise RuntimeError("raw density archive schema drift")
        schema = np.asarray(archive["schema"])
        chunk_ids = np.asarray(archive["chunk_id"])
        cutoffs = np.asarray(archive["cutoff"])
        archive_row_ids = [str(value) for value in archive["row_ids"]]
        density_ids = [str(value) for value in archive["density_row_ids"]]
        density_stack = np.asarray(archive["densities"], dtype=np.complex128)
        raw_iq = np.asarray(archive["raw_iq"], dtype=np.float64)
        heldout_iq = np.asarray(archive["heldout_iq"], dtype=np.float64)
    cutoff = int(cell["cutoff"])
    if (
        schema.shape != (1,)
        or str(schema[0]) != "PHASE9-FRESH-TWIN-CHUNKED-RAW-ARCHIVE-V1"
        or chunk_ids.shape != (1,)
        or str(chunk_ids[0]) != cell["chunk_id"]
        or cutoffs.shape != (1,)
        or int(cutoffs[0]) != cutoff
        or archive_row_ids != raw_row_ids
        or density_ids != terminal_row_ids
        or len(density_ids) != len(set(density_ids))
        or len(density_ids) != len(density_stack)
        or raw_iq.shape != (len(rows), 8, 2)
        or heldout_iq.shape != (len(rows), 8, 2)
        or not np.all(np.isfinite(raw_iq))
        or not np.all(np.isfinite(heldout_iq))
    ):
        raise RuntimeError("raw density row alignment drift")
    _physicality(density_stack, cutoff)
    return rows, dict(zip(density_ids, density_stack, strict=True))


def _populate_and_verify_logical_survival(
    execution: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    densities: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Cross-check fault values and derive the undefined shared CSV field.

    The raw runner only emits logical records for trajectories initialized in
    a logical state.  Shared vacuum-f/RESET cells therefore archive a joint
    density but intentionally leave ``logical_survival`` blank.  The frozen
    diagnostic blueprint nevertheless requires code-space projection for
    those cells.  V2 derives it from the archived density using each backend's
    native, source-bound logical isometry.  Before using that route for shared
    cells, every archived fault terminal is compared with the independently
    recorded CSV value under its float32 density quantization certificate.
    """

    if pilot_runner is None or not hasattr(pilot_runner, "runner"):
        raise RuntimeError("verified fresh runner is unavailable")
    projectors: dict[tuple[int, str], np.ndarray] = {}
    projector_errors: dict[str, float] = {}
    for cutoff in (28, 32, 36):
        built = pilot_runner.runner.build_simulators(execution, cutoff)
        if set(built) != {"A", "B"}:
            raise RuntimeError("logical projection simulator coverage drift")
        for backend in ("A", "B"):
            simulator = built[backend]
            if backend == "A":
                isometry = np.asarray(
                    simulator._logical_engine().code_basis.isometry,
                    dtype=np.complex128,
                )
            else:
                isometry = np.asarray(simulator._comb_isometry(), dtype=np.complex128)
            if isometry.shape != (cutoff, 2):
                raise RuntimeError("logical projection isometry shape drift")
            error = float(
                np.linalg.norm(
                    isometry.conj().T @ isometry - np.eye(2, dtype=np.complex128),
                    ord="fro",
                )
            )
            if not np.isfinite(error) or error > 1e-10:
                raise RuntimeError("logical projection isometry orthonormality drift")
            projectors[(cutoff, backend)] = isometry @ isometry.conj().T
            projector_errors[f"c{cutoff}/{backend}"] = error

    fault_count = 0
    shared_count = 0
    maximum_fault_delta = 0.0
    maximum_fault_allowance = 0.0
    shared_values: list[float] = []
    for row in rows:
        if not row["terminal_round"]:
            continue
        row_id = str(row["row_id"])
        if row_id not in densities:
            raise RuntimeError("terminal logical-projection density missing")
        cutoff = int(row["cutoff"])
        backend = str(row["backend"])
        joint = np.asarray(densities[row_id], dtype=np.complex128)
        if joint.shape != (3 * cutoff, 3 * cutoff):
            raise RuntimeError("logical-projection joint density shape drift")
        # Both source-bound backends use the verified oscillator-major,
        # ancilla-minor tensor layout.  Do the linear partial trace directly:
        # the backend-B convenience method rejects the legitimate <=3.8e-8
        # float32 archive trace error at its simulation-time 1e-8 threshold.
        joint_tensor = joint.reshape(cutoff, 3, cutoff, 3)
        oscillator_matrix = np.trace(joint_tensor, axis1=1, axis2=3)
        oscillator_matrix = (oscillator_matrix + oscillator_matrix.conj().T) / 2.0
        survival = float(
            np.trace(projectors[(cutoff, backend)] @ oscillator_matrix).real
        )
        if not np.isfinite(survival) or survival < -5e-6 or survival > 1.0 + 5e-6:
            raise RuntimeError("density-derived logical survival is invalid")
        survival = min(max(survival, 0.0), 1.0)
        if row["layer"] == "fault":
            recorded = row["logical_survival"]
            if recorded is None:
                raise RuntimeError("fault logical-survival value missing")
            certificate = row["density_quantization_trace_distance_bound"]
            if certificate is None:
                raise RuntimeError(
                    "fault logical-survival quantization certificate missing"
                )
            allowance = float(certificate) + 1e-6
            delta = abs(float(recorded) - survival)
            maximum_fault_delta = max(maximum_fault_delta, delta)
            maximum_fault_allowance = max(maximum_fault_allowance, allowance)
            if delta > allowance:
                raise RuntimeError(
                    "density-derived fault logical survival disagrees with CSV"
                )
            fault_count += 1
        elif row["layer"] == "shared":
            if row["logical_survival"] is not None:
                raise RuntimeError(
                    "shared logical survival was not blank before derivation"
                )
            row["logical_survival"] = survival
            shared_values.append(survival)
            shared_count += 1
        else:
            raise RuntimeError("logical-projection layer drift")

    if fault_count != 1728 or shared_count != 432:
        raise RuntimeError("logical-projection terminal denominator drift")
    return {
        "schema_version": ("PHASE9-DIAGNOSTIC-DENSITY-LOGICAL-PROJECTION-AUDIT-V1"),
        "source": "archived_joint_density_and_native_backend_isometry",
        "fault_terminal_cross_checks": fault_count,
        "shared_terminal_derivations": shared_count,
        "maximum_fault_absolute_delta": maximum_fault_delta,
        "maximum_fault_allowed_delta": maximum_fault_allowance,
        "shared_minimum": min(shared_values),
        "shared_maximum": max(shared_values),
        "projector_orthonormality_frobenius": projector_errors,
    }


def _selected_reference_receipts(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    binding = config["reference_cutoff_28_evidence"]["manifest"]
    _, payload = _read_bound_bytes(root, binding)
    manifest = json.loads(payload)
    _self_hash(manifest, "cutoff28 reference manifest")
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    receipts = manifest.get("chunk_receipts")
    receipt_bindings = manifest.get("receipt_bindings")
    if not isinstance(receipts, list) or not isinstance(receipt_bindings, list):
        raise RuntimeError("cutoff28 reference receipt ledger missing")
    for receipt, receipt_binding in zip(receipts, receipt_bindings, strict=True):
        cell = receipt.get("cell", {})
        if int(cell.get("cutoff", -1)) == 28 and cell.get("layer") == "fault":
            selected.append((receipt, receipt_binding))
    expected = {
        (scenario, backend)
        for scenario in ("step", "telegraph", "burst", "compound")
        for backend in ("A", "B")
    }
    observed = {
        (str(receipt["cell"]["scenario"]), str(receipt["cell"]["backend"]))
        for receipt, _ in selected
    }
    if len(selected) != 8 or observed != expected:
        raise RuntimeError("cutoff28 reference fault coverage drift")
    return manifest, selected


def _load_evidence(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    list[dict[str, object]],
    dict[str, Any],
]:
    _require_verified_self_import()
    config, base = pilot_runner.load_pilot_config(root, require_hardened=True)
    execution = pilot_runner.materialize_execution_config(config, base)
    owner_path = root / str(config["artifact_paths"]["owner_lock"])
    if owner_path.exists():
        raise RuntimeError("design supervisor is active or stale")
    run_identity_path = root / str(config["artifact_paths"]["run_identity"])
    run_identity_binding = _binding(run_identity_path, root)
    _, run_payload = _read_bound_bytes(root, run_identity_binding)
    run_identity = json.loads(run_payload)
    _self_hash(run_identity, "design run identity")
    snapshot = run_identity.get("input_snapshot")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError("design input snapshot missing")
    pilot_runner._assert_input_snapshot(root, snapshot)
    pilot_runner._activate_verified_execution_modules(root, snapshot)
    cells = pilot_runner.build_pilot_cells(config, execution)
    manifest_path = root / str(config["artifact_paths"]["execution_manifest"])
    manifest_binding = _binding(manifest_path, root)
    _, manifest_payload = _read_bound_bytes(root, manifest_binding)
    manifest = json.loads(manifest_payload)
    _self_hash(manifest, "design manifest")
    pilot_runner._verify_manifest(
        root, config, execution, cells, run_identity, manifest
    )
    pilot_runner._verify_complete_marker(root, config, run_identity, manifest)
    if (
        manifest.get("status") != pilot_runner.STATUS
        or manifest.get("scientific_verdict") is not None
        or manifest.get("qualified_claim") is not None
        or manifest.get("observed_cells") != 22
        or manifest.get("observed_rows") != 14256
        or manifest.get("exception_rows") != 0
        or manifest.get("conservation_failure_rows") != 0
        or manifest.get("claim_state") != pilot_runner.CLAIM_BOUNDARY
    ):
        raise RuntimeError("design manifest incomplete or contaminated")

    reference_manifest, reference_receipts = _selected_reference_receipts(root, config)
    receipt_pairs = list(
        zip(
            manifest["chunk_receipts"],
            manifest["receipt_bindings"],
            strict=True,
        )
    )
    rows: list[dict[str, Any]] = []
    densities: dict[str, np.ndarray] = {}
    raw_bindings: list[dict[str, object]] = []
    for receipt, receipt_binding in [*reference_receipts, *receipt_pairs]:
        chunk_rows, chunk_densities = _parse_receipt(root, receipt, receipt_binding)
        rows.extend(chunk_rows)
        if set(densities) & set(chunk_densities):
            raise RuntimeError("duplicate raw density row id")
        densities.update(chunk_densities)
        raw_bindings.extend(
            [
                dict(receipt_binding),
                dict(receipt["csv"]),
                dict(receipt["npz"]),
            ]
        )
    if len(rows) != 21168 or len(densities) != 2160:
        raise RuntimeError("combined design evidence denominator drift")
    if len({str(row["row_id"]) for row in rows}) != len(rows):
        raise RuntimeError("combined design row IDs are not unique")
    logical_projection_audit = _populate_and_verify_logical_survival(
        execution, rows, densities
    )
    pilot_runner._assert_input_snapshot(root, snapshot)
    return (
        config,
        manifest,
        reference_manifest,
        rows,
        densities,
        raw_bindings,
        logical_projection_audit,
    )


def _embed_density(matrix: np.ndarray, lower: int, upper: int) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.complex128)
    if value.shape != (3 * lower, 3 * lower) or not 0 < lower < upper:
        raise RuntimeError("cutoff density embedding drift")
    output = np.zeros((3 * upper, 3 * upper), dtype=np.complex128)
    output[: 3 * lower, : 3 * lower] = value
    return output


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = (left - right + (left - right).conj().T) / 2
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(difference))))


def _tail_features(stack: np.ndarray, cutoff: int) -> dict[str, np.ndarray]:
    value = np.asarray(stack, dtype=np.complex128)
    _physicality(value, cutoff)
    joint = value.reshape(len(value), cutoff, 3, cutoff, 3)
    oscillator = np.trace(joint, axis1=2, axis2=4)
    populations = np.diagonal(oscillator, axis1=1, axis2=2).real
    if (
        float(np.min(populations)) < -5e-5
        or float(np.max(np.abs(np.sum(populations, axis=1) - 1.0))) > 5e-5
    ):
        raise RuntimeError("tail population physicality drift")
    top1 = np.sum(populations[:, -1:], axis=1)
    return {
        "top1_fock_mass": top1,
        "top2_fock_mass": np.sum(populations[:, -2:], axis=1),
        "top4_fock_mass": np.sum(populations[:, -4:], axis=1),
        "normalized_mean_photon": (
            populations @ np.arange(cutoff, dtype=np.float64) / float(cutoff - 1)
        ),
        "commutator_defect": float(cutoff) * top1,
    }


def _tail_quantization_bound(
    metric: str, quantization: np.ndarray, cutoff: int
) -> float:
    """Lift per-density trace-distance certificates to tail observables.

    Projector probabilities and normalized photon number are effects with
    operator norm at most one, so their expectation error is bounded by trace
    distance.  ``commutator_defect`` is ``cutoff * top1_fock_mass`` and
    therefore carries the corresponding cutoff factor.
    """

    values = np.asarray(quantization, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or cutoff <= 0
    ):
        raise RuntimeError("tail quantization certificate drift")
    if metric in {
        "top1_fock_mass",
        "top2_fock_mass",
        "top4_fock_mass",
        "normalized_mean_photon",
    }:
        factor = 1.0
    elif metric == "commutator_defect":
        factor = float(cutoff)
    else:
        raise RuntimeError("unknown tail metric quantization rule")
    return factor * float(np.mean(values))


def _gate(
    *,
    gate_id: str,
    family: str,
    contrast: str,
    scenario: str,
    state: str,
    stage: str,
    metric: str,
    cutoff: str,
    backend: str,
    estimate: float,
    quantization: float,
    margin: float,
    clusters: int,
) -> dict[str, object]:
    values = (estimate, quantization, margin)
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise RuntimeError("diagnostic gate contains invalid numeric value")
    conservative = float(estimate + quantization)
    return {
        "gate_id": gate_id,
        "family": family,
        "contrast": contrast,
        "scenario": scenario,
        "logical_state": state,
        "stage": stage,
        "metric": metric,
        "cutoff_or_increment": cutoff,
        "backend_or_pair": backend,
        "estimate": float(estimate),
        "quantization_bound": float(quantization),
        "conservative_point": conservative,
        "margin": float(margin),
        "passed": bool(conservative <= margin),
        "cluster_count": int(clusters),
        "statistical_role": "UNPOWERED_POINT_SCREEN_NO_EQUIVALENCE_CLAIM",
        "qualification_effect": None,
    }


def _fault_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, str, str, int], list[dict[str, Any]]] = {}
    schedule = ("0", "1", "+", "-", "+i", "-i")
    for row in rows:
        if row["layer"] != "fault":
            continue
        key = (
            int(row["cutoff"]),
            str(row["scenario"]),
            str(row["backend"]),
            str(row["logical_label"]),
            int(row["seed_position"]),
        )
        grouped.setdefault(key, []).append(dict(row))
    expected_cells = {
        (cutoff, scenario, backend)
        for cutoff in (28, 32, 36)
        for scenario in ("step", "telegraph", "burst", "compound")
        for backend in ("A", "B")
    }
    observed_cells = {key[:3] for key in grouped}
    if observed_cells != expected_cells:
        raise RuntimeError("fault cell coverage drift")
    for key, values in grouped.items():
        values.sort(key=lambda row: int(row["round_index"]))
        if key[3] != schedule[key[4] % 6] or [
            row["round_index"] for row in values
        ] != list(range(12)):
            raise RuntimeError("fault state schedule/round coverage drift")
    if len(grouped) != 3 * 4 * 2 * 72:
        raise RuntimeError("fault trajectory denominator drift")
    return grouped


def _fault_positions(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
) -> list[int]:
    positions = sorted(
        key[4] for key in grouped if key[:4] == (cutoff, scenario, backend, state)
    )
    if len(positions) != 12:
        raise RuntimeError("fault per-state denominator drift")
    return positions


def _fault_stage(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    *,
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
    rounds: Sequence[int],
) -> np.ndarray:
    positions = _fault_positions(grouped, cutoff, scenario, backend, state)
    fields = ("mean_photon", "level_g", "level_e", "level_f", "logical_survival")
    selected = set(rounds)
    matrix = []
    for position in positions:
        values = [
            row
            for row in grouped[(cutoff, scenario, backend, state, position)]
            if row["round_index"] in selected
        ]
        if len(values) != len(selected):
            raise RuntimeError("fault stage denominator drift")
        matrix.append(
            [float(np.mean([float(row[field]) for row in values])) for field in fields]
        )
    return np.asarray(matrix, dtype=np.float64)


def _fault_terminal(
    grouped: Mapping[tuple[int, str, str, str, int], Sequence[Mapping[str, Any]]],
    densities: Mapping[str, np.ndarray],
    *,
    cutoff: int,
    scenario: str,
    backend: str,
    state: str,
) -> tuple[np.ndarray, np.ndarray]:
    stack = []
    quantization = []
    for position in _fault_positions(grouped, cutoff, scenario, backend, state):
        row = grouped[(cutoff, scenario, backend, state, position)][-1]
        if not row["terminal_round"]:
            raise RuntimeError("fault terminal row drift")
        stack.append(densities[str(row["row_id"])])
        quantization.append(float(row["density_quantization_trace_distance_bound"]))
    return (
        np.asarray(stack, dtype=np.complex128),
        np.asarray(quantization, dtype=np.float64),
    )


def _shared_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["layer"] != "shared":
            continue
        if (
            row["initial_state"] != "vacuum_f"
            or row["action"] != "RESET"
            or not row["terminal_round"]
            or row["round_index"] != 0
        ):
            raise RuntimeError("shared repair row identity drift")
        grouped.setdefault((int(row["cutoff"]), str(row["backend"])), []).append(
            dict(row)
        )
    expected = {(cutoff, backend) for cutoff in (28, 32, 36) for backend in ("A", "B")}
    if set(grouped) != expected:
        raise RuntimeError("shared repair cell coverage drift")
    for values in grouped.values():
        values.sort(key=lambda row: int(row["seed_position"]))
        if [row["seed_position"] for row in values] != list(range(72)):
            raise RuntimeError("shared repair denominator drift")
    return grouped


def _density_point(
    lower_stack: np.ndarray,
    upper_stack: np.ndarray,
    lower_cutoff: int,
    upper_cutoff: int,
) -> float:
    embedded = np.asarray(
        [_embed_density(matrix, lower_cutoff, upper_cutoff) for matrix in lower_stack]
    )
    return _trace_distance(np.mean(embedded, axis=0), np.mean(upper_stack, axis=0))


def _scalar_points(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_mean = np.mean(left, axis=0)
    right_mean = np.mean(right, axis=0)
    return {
        "mean_photon": abs(float(left_mean[0] - right_mean[0])),
        "level_probability_l1": float(np.sum(np.abs(left_mean[1:4] - right_mean[1:4]))),
        "logical_survival": abs(float(left_mean[4] - right_mean[4])),
    }


def evaluate(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    densities: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    contract = config["diagnostic_contract"]
    margins = contract["margins"]
    grouped = _fault_index(rows)
    output: list[dict[str, object]] = []
    states = ("0", "1", "+", "-", "+i", "-i")
    for lower, upper in ((28, 32), (32, 36)):
        for scenario in config["scenario_names"]:
            for backend in ("A", "B"):
                for state in states:
                    lower_density, lower_q = _fault_terminal(
                        grouped,
                        densities,
                        cutoff=lower,
                        scenario=scenario,
                        backend=backend,
                        state=state,
                    )
                    upper_density, upper_q = _fault_terminal(
                        grouped,
                        densities,
                        cutoff=upper,
                        scenario=scenario,
                        backend=backend,
                        state=state,
                    )
                    output.append(
                        _gate(
                            gate_id=(
                                f"fault/density/{lower}->{upper}/"
                                f"{scenario}/{backend}/{state}"
                            ),
                            family="fault_density",
                            contrast="within_backend_cutoff",
                            scenario=scenario,
                            state=state,
                            stage="terminal",
                            metric="density_trace_distance",
                            cutoff=f"{lower}->{upper}",
                            backend=backend,
                            estimate=_density_point(
                                lower_density,
                                upper_density,
                                lower,
                                upper,
                            ),
                            quantization=float(np.mean(lower_q) + np.mean(upper_q)),
                            margin=float(contract["design_density_point_threshold"]),
                            clusters=12,
                        )
                    )
                    for stage, stage_rounds in config["stage_partition"][
                        scenario
                    ].items():
                        lower_stage = _fault_stage(
                            grouped,
                            cutoff=lower,
                            scenario=scenario,
                            backend=backend,
                            state=state,
                            rounds=stage_rounds,
                        )
                        upper_stage = _fault_stage(
                            grouped,
                            cutoff=upper,
                            scenario=scenario,
                            backend=backend,
                            state=state,
                            rounds=stage_rounds,
                        )
                        for metric, estimate in _scalar_points(
                            lower_stage, upper_stage
                        ).items():
                            margin_key = {
                                "mean_photon": (
                                    "cutoff_terminal_mean_photon_difference"
                                ),
                                "level_probability_l1": (
                                    "cutoff_terminal_level_probability_l1"
                                ),
                                "logical_survival": (
                                    "cutoff_terminal_logical_survival_difference"
                                ),
                            }[metric]
                            output.append(
                                _gate(
                                    gate_id=(
                                        f"fault/scalar/{metric}/{lower}->{upper}/"
                                        f"{scenario}/{backend}/{state}/{stage}"
                                    ),
                                    family="fault_scalar",
                                    contrast="within_backend_cutoff",
                                    scenario=scenario,
                                    state=state,
                                    stage=stage,
                                    metric=metric,
                                    cutoff=f"{lower}->{upper}",
                                    backend=backend,
                                    estimate=estimate,
                                    quantization=0.0,
                                    margin=float(margins[margin_key]),
                                    clusters=12,
                                )
                            )
    for scenario in config["scenario_names"]:
        for backend in ("A", "B"):
            for state in states:
                stack, quantization = _fault_terminal(
                    grouped,
                    densities,
                    cutoff=36,
                    scenario=scenario,
                    backend=backend,
                    state=state,
                )
                for metric, values in _tail_features(stack, 36).items():
                    margin_key = f"absolute_terminal_{metric}"
                    output.append(
                        _gate(
                            gate_id=(
                                f"fault/tail/{metric}/c36/"
                                f"{scenario}/{backend}/{state}"
                            ),
                            family="fault_absolute_tail",
                            contrast="absolute_cutoff36_tail",
                            scenario=scenario,
                            state=state,
                            stage="terminal",
                            metric=metric,
                            cutoff="36",
                            backend=backend,
                            estimate=float(np.mean(values)),
                            quantization=_tail_quantization_bound(
                                metric, quantization, 36
                            ),
                            margin=float(margins[margin_key]),
                            clusters=12,
                        )
                    )

    shared = _shared_index(rows)

    def shared_stack(cutoff: int, backend: str) -> tuple[np.ndarray, np.ndarray]:
        values = shared[(cutoff, backend)]
        return (
            np.asarray(
                [densities[str(row["row_id"])] for row in values],
                dtype=np.complex128,
            ),
            np.asarray(
                [
                    float(row["density_quantization_trace_distance_bound"])
                    for row in values
                ],
                dtype=np.float64,
            ),
        )

    def shared_scalar(cutoff: int, backend: str) -> np.ndarray:
        return np.asarray(
            [
                [
                    row["mean_photon"],
                    row["level_g"],
                    row["level_e"],
                    row["level_f"],
                    row["logical_survival"],
                ]
                for row in shared[(cutoff, backend)]
            ],
            dtype=np.float64,
        )

    for lower, upper in ((28, 32), (32, 36)):
        for backend in ("A", "B"):
            lower_stack, lower_q = shared_stack(lower, backend)
            upper_stack, upper_q = shared_stack(upper, backend)
            output.append(
                _gate(
                    gate_id=f"shared/density/{lower}->{upper}/{backend}",
                    family="shared_density",
                    contrast="within_backend_cutoff",
                    scenario="shared_vacuum_f_reset",
                    state="not_applicable",
                    stage="terminal",
                    metric="density_trace_distance",
                    cutoff=f"{lower}->{upper}",
                    backend=backend,
                    estimate=_density_point(lower_stack, upper_stack, lower, upper),
                    quantization=float(np.mean(lower_q) + np.mean(upper_q)),
                    margin=float(margins["cutoff_terminal_density_trace_distance"]),
                    clusters=72,
                )
            )
            for metric, estimate in _scalar_points(
                shared_scalar(lower, backend),
                shared_scalar(upper, backend),
            ).items():
                margin_key = {
                    "mean_photon": ("cutoff_terminal_mean_photon_difference"),
                    "level_probability_l1": ("cutoff_terminal_level_probability_l1"),
                    "logical_survival": ("cutoff_terminal_logical_survival_difference"),
                }[metric]
                output.append(
                    _gate(
                        gate_id=(
                            f"shared/scalar/{metric}/" f"{lower}->{upper}/{backend}"
                        ),
                        family="shared_scalar",
                        contrast="within_backend_cutoff",
                        scenario="shared_vacuum_f_reset",
                        state="not_applicable",
                        stage="terminal",
                        metric=metric,
                        cutoff=f"{lower}->{upper}",
                        backend=backend,
                        estimate=estimate,
                        quantization=(
                            float(np.mean(lower_q) + np.mean(upper_q))
                            if metric == "logical_survival"
                            else 0.0
                        ),
                        margin=float(margins[margin_key]),
                        clusters=72,
                    )
                )
    for cutoff in (28, 32, 36):
        stack_a, q_a = shared_stack(cutoff, "A")
        stack_b, q_b = shared_stack(cutoff, "B")
        output.append(
            _gate(
                gate_id=f"shared/density/ab/c{cutoff}",
                family="shared_density",
                contrast="same_cutoff_ab",
                scenario="shared_vacuum_f_reset",
                state="not_applicable",
                stage="terminal",
                metric="density_trace_distance",
                cutoff=str(cutoff),
                backend="A/B",
                estimate=_trace_distance(
                    np.mean(stack_a, axis=0), np.mean(stack_b, axis=0)
                ),
                quantization=float(np.mean(q_a) + np.mean(q_b)),
                margin=float(margins["ab_terminal_density_trace_distance"]),
                clusters=72,
            )
        )
        for metric, estimate in _scalar_points(
            shared_scalar(cutoff, "A"), shared_scalar(cutoff, "B")
        ).items():
            margin_key = {
                "mean_photon": "ab_terminal_mean_photon_difference",
                "level_probability_l1": ("ab_terminal_level_probability_l1"),
                "logical_survival": ("ab_terminal_logical_survival_difference"),
            }[metric]
            output.append(
                _gate(
                    gate_id=f"shared/scalar/{metric}/ab/c{cutoff}",
                    family="shared_scalar",
                    contrast="same_cutoff_ab",
                    scenario="shared_vacuum_f_reset",
                    state="not_applicable",
                    stage="terminal",
                    metric=metric,
                    cutoff=str(cutoff),
                    backend="A/B",
                    estimate=estimate,
                    quantization=(
                        float(np.mean(q_a) + np.mean(q_b))
                        if metric == "logical_survival"
                        else 0.0
                    ),
                    margin=float(margins[margin_key]),
                    clusters=72,
                )
            )
    for backend in ("A", "B"):
        stack, quantization = shared_stack(36, backend)
        for metric, values in _tail_features(stack, 36).items():
            output.append(
                _gate(
                    gate_id=f"shared/tail/{metric}/c36/{backend}",
                    family="shared_absolute_tail",
                    contrast="absolute_cutoff36_tail",
                    scenario="shared_vacuum_f_reset",
                    state="not_applicable",
                    stage="terminal",
                    metric=metric,
                    cutoff="36",
                    backend=backend,
                    estimate=float(np.mean(values)),
                    quantization=_tail_quantization_bound(metric, quantization, 36),
                    margin=float(margins[f"absolute_terminal_{metric}"]),
                    clusters=72,
                )
            )

    if len(output) != int(contract["expected_gate_count"]):
        raise RuntimeError("diagnostic gate denominator drift")
    gate_ids = [str(row["gate_id"]) for row in output]
    if len(gate_ids) != len(set(gate_ids)):
        raise RuntimeError("diagnostic gate IDs are not unique")
    observed_accounting: dict[str, int] = {}
    for row in output:
        family = str(row["family"])
        observed_accounting[family] = observed_accounting.get(family, 0) + 1
    if observed_accounting != contract["gate_accounting"]:
        raise RuntimeError("diagnostic gate-family accounting drift")
    return sorted(output, key=lambda row: str(row["gate_id"]))


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SOURCE_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in SOURCE_FIELDS})
    return buffer.getvalue().encode("utf-8")


def _report(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    reference_manifest: Mapping[str, Any],
    gates: Sequence[Mapping[str, object]],
    raw_bindings: Sequence[Mapping[str, object]],
    logical_projection_audit: Mapping[str, Any],
    source_binding: Mapping[str, object],
) -> dict[str, Any]:
    passed = [row for row in gates if row["passed"] is True]
    failed = [row for row in gates if row["passed"] is False]
    verdict = PASS_VERDICT if not failed else NO_GO_VERDICT
    family_counts: dict[str, dict[str, int]] = {}
    for row in gates:
        family = str(row["family"])
        ledger = family_counts.setdefault(
            family, {"total": 0, "passed": 0, "failed": 0}
        )
        ledger["total"] += 1
        ledger["passed" if row["passed"] else "failed"] += 1
    config_binding = _binding(root / pilot_runner.PENDING_CONFIG_PATH, root)
    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA,
        "status": STATUS,
        "scientific_verdict": verdict,
        "authorization_effect": (
            "POWERED_FORMAL_MAY_BE_ATOMICALLY_RELEASED"
            if verdict == PASS_VERDICT
            else "POWERED_FORMAL_REMAINS_UNRELEASED"
        ),
        "design_screen_only": True,
        "equivalence_claim": None,
        "gate_count": len(gates),
        "passed_gate_count": len(passed),
        "failed_gate_count": len(failed),
        "family_counts": family_counts,
        "maximum_margin_ratio": max(
            float(row["conservative_point"]) / float(row["margin"]) for row in gates
        ),
        "failed_gate_ids": [str(row["gate_id"]) for row in failed],
        "gate_rows": list(gates),
        "bindings": {
            "pending_config": config_binding,
            "released_child": _binding(root / pilot_runner.CONFIG_PATH, root),
            "release_receipt": _binding(root / pilot_runner.RELEASE_RECEIPT_PATH, root),
            "design_manifest": _binding(
                root / str(config["artifact_paths"]["execution_manifest"]),
                root,
            ),
            "reference_manifest": dict(
                config["reference_cutoff_28_evidence"]["manifest"]
            ),
            "density_uq_report": dict(
                config["uq_preflight_sources"]["density"]["report"]
            ),
            "scalar_uq_report": dict(
                config["uq_preflight_sources"]["scalar"]["report"]
            ),
            "scalar_uq_independent_verification": dict(
                config["uq_preflight_sources"]["scalar"]["independent_verification"]
            ),
            "diagnostic_source": _binding(Path(__file__).resolve(), root),
            "source_data": dict(source_binding),
        },
        "raw_binding_count": len(raw_bindings),
        "raw_bindings": [dict(binding) for binding in raw_bindings],
        "raw_bindings_sha256": _sha(list(raw_bindings)),
        "logical_projection_audit": dict(logical_projection_audit),
        "design_manifest_analysis_sha256": manifest["analysis_sha256"],
        "reference_manifest_analysis_sha256": reference_manifest["analysis_sha256"],
        "formal_outcomes_accessed": False,
        "qualified_claim": None,
        "claim_state": dict(CLAIM_BOUNDARY),
    }
    report["analysis_sha256"] = _sha(report)
    return report


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    _require_verified_self_import()
    repository = _root() if root is None else root.resolve()
    (
        config,
        manifest,
        reference,
        rows,
        densities,
        raw_bindings,
        logical_projection_audit,
    ) = _load_evidence(repository)
    lock_path = repository / str(config["artifact_paths"]["diagnostic_owner_lock"])
    report_path = repository / str(config["artifact_paths"]["diagnostic_report"])
    source_path = repository / str(config["artifact_paths"]["diagnostic_source_data"])
    completion_path = repository / str(
        config["artifact_paths"]["diagnostic_completion"]
    )
    with _exclusive_lock(lock_path):
        if any(path.exists() for path in (report_path, source_path, completion_path)):
            if not all(
                path.exists() for path in (report_path, source_path, completion_path)
            ):
                raise RuntimeError("partial diagnostic publication exists")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            _self_hash(report, "published diagnostic report")
            _self_hash(completion, "published diagnostic completion")
            if (
                completion.get("report") != _binding(report_path, repository)
                or completion.get("source_data") != _binding(source_path, repository)
                or completion.get("scientific_verdict")
                != report.get("scientific_verdict")
            ):
                raise RuntimeError("published diagnostic receipt drift")
            return report

        gates = evaluate(config, rows, densities)
        source_payload = _csv_bytes(gates)
        _atomic_bytes(source_path, source_payload)
        source_binding = _binding(source_path, repository)
        report = _report(
            repository,
            config,
            manifest,
            reference,
            gates,
            raw_bindings,
            logical_projection_audit,
            source_binding,
        )
        _atomic_bytes(
            report_path,
            (
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        )
        completion: dict[str, Any] = {
            "task_id": TASK_ID,
            "schema_version": COMPLETION_SCHEMA,
            "status": "COMPLETE",
            "scientific_verdict": report["scientific_verdict"],
            "report": _binding(report_path, repository),
            "source_data": source_binding,
            "design_manifest": _binding(
                repository / str(config["artifact_paths"]["execution_manifest"]),
                repository,
            ),
            "qualified_claim": None,
            "claim_state": dict(CLAIM_BOUNDARY),
        }
        completion["analysis_sha256"] = _sha(completion)
        _atomic_bytes(
            completion_path,
            (
                json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            ).encode("utf-8"),
        )
        return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the sealed cutoff32/36 design extension."
    )
    parser.parse_args(argv)
    report = write_artifacts()
    print(
        json.dumps(
            {
                "status": report["status"],
                "scientific_verdict": report["scientific_verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "gate_count": report["gate_count"],
                "failed_gate_count": report["failed_gate_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["scientific_verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_BOUNDARY",
    "NO_GO_VERDICT",
    "PASS_VERDICT",
    "evaluate",
    "main",
    "write_artifacts",
]
