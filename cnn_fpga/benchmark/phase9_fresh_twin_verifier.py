"""Physics-free verifier for the fresh Phase-9 twin qualification.

The formal runner is an evidence producer only.  This module does not import
the runner, either physics backend, the analytic reference module, or the old
T9.2.4 verifier.  It reconstructs every scientific estimand from the fresh
CSV ledger and ZIP/NPZ raw archive, applies the preregistered IUT/TOST rule,
and emits one of three disjoint outcomes:

* ``PASS_FRESH_TWIN_QUALIFIED``;
* ``NO_GO_FRESH_TWIN_QUALIFICATION`` (complete evidence, scientific miss);
* ``INCOMPLETE_FAIL_CLOSED`` (missing/corrupt/non-finite/exception evidence).

Raw log evidence is checked as a diagnostic only.  IQ primary quantities are
Rao--Blackwellized from the pre-measurement prior.  Reset success is likewise
recomputed from the pre-reset levels.  Fault observables are never combined
across units.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from scipy.special import ndtr


TASK_ID = "T-RISK-20260726-01"
VERIFIER_ID = "PHASE9-FRESH-TWIN-PHYSICS-FREE-VERIFIER-V1"
CONFIG_SCHEMA = "PHASE9-FRESH-TWIN-QUALIFICATION-CONFIG-V1"
ROW_SCHEMA = "PHASE9-FRESH-TWIN-ROUND-LEDGER-V1"
ARCHIVE_SCHEMA = "PHASE9-FRESH-TWIN-CHUNKED-RAW-ARCHIVE-V1"
ARCHIVE_MANIFEST_SCHEMA = ARCHIVE_SCHEMA
EXECUTION_MANIFEST_SCHEMA = "PHASE9-FRESH-TWIN-EXECUTION-MANIFEST-V1"
PREFORMAL_SEAL_SCHEMA = "PHASE9-FRESH-TWIN-PREFORMAL-SEAL-V1"
ATTEMPT_SCHEMA = "PHASE9-FRESH-TWIN-ATTEMPT-EVENT-V1"
PASS_VERDICT = "PASS_FRESH_TWIN_QUALIFIED"
NO_GO_VERDICT = "NO_GO_FRESH_TWIN_QUALIFICATION"
INCOMPLETE_VERDICT = "INCOMPLETE_FAIL_CLOSED"
QUALIFIED_CLAIM = "dual_backend_agreement_for_fresh_repaired_synthetic_task"
Z_TOST = 1.6448536269514722

TYPED_NULL_FIELDS = (
    "frontend_performance",
    "synthetic_iq_qualification",
    "recorded_iq_qualification",
    "live_raw_iq_qualification",
    "board_measured_latency",
    "board_resources",
    "board_power",
    "external_same_task_speed",
    "round_ler",
    "six_state_lifetime",
    "physical_break_even",
    "official_puviani_exact",
    "puviani_nmf_surpass",
    "external_sota",
    "rank",
)
DOWNSTREAM_TASKS = (
    "T9.2.5",
    "T9.2.7",
    "T9.3.1",
    "T9.3.4",
    "T9.6.2",
    "T9.6.5",
)

# Split string literals keep policy scanners able to detect the deny-list
# behaviour without accidentally treating this verifier as a consumer.
PROHIBITED_HISTORICAL_BASENAMES = frozenset(
    {
        "t9_2_4_dual_backend_" + "cell_ledger.csv",
        "t9_2_4_dual_backend_" + "qualification_source_data.csv",
        "t9_2_4_dual_backend_" + "state_archive.npz",
    }
)

DEFAULT_CONFIG_PATH = (
    "configs/phase9/t_risk_20260726_01_fresh_twin_qualification.json"
)
DEFAULT_DESIGN_CONFIG_PATH = (
    "configs/phase9/t_risk_20260726_01_design_power.json"
)
DEFAULT_LEDGER_PATH = "docs/t_risk_20260726_01_fresh_cell_ledger.csv"
DEFAULT_ARCHIVE_PATH = "docs/t_risk_20260726_01_fresh_raw_archive.zip"
DEFAULT_MANIFEST_PATH = "docs/t_risk_20260726_01_fresh_execution_manifest.json"
DEFAULT_REPORT_PATH = "docs/t_risk_20260726_01_fresh_verification.json"
DEFAULT_QUALIFICATION_PATH = (
    "docs/t_risk_20260726_01_fresh_qualification.json"
)
DEFAULT_SOURCE_PATH = (
    "docs/t_risk_20260726_01_fresh_qualification_source_data.csv"
)
DEFAULT_GATE_LEDGER_PATH = (
    "docs/t_risk_20260726_01_fresh_gate_ledger.csv"
)
DEFAULT_RELEASE_PATH = "docs/t_risk_20260726_01_fresh_release.json"
DEFAULT_RELEASE_PIN_PATH = (
    "configs/phase9/t_risk_20260726_01_fresh_release_pin.json"
)

REQUIRED_LEDGER_FIELDS = frozenset(
    {
        "row_id",
        "row_schema",
        "layer",
        "cell_base",
        "cell_id",
        "backend",
        "backend_id",
        "cutoff",
        "seed",
        "seed_position",
        "trajectory_id",
        "round_index",
        "terminal_round",
        "action",
        "probe_id",
        "scenario",
        "initial_state",
        "logical_label",
        "rng_namespace",
        "archive_chunk",
        "archive_row_index",
        "density_index",
        "raw_iq_index",
        "heldout_iq_index",
        "heldout_window_sha256",
        "pre_measurement_g",
        "pre_measurement_e",
        "pre_measurement_f",
        "pre_reset_g",
        "pre_reset_e",
        "pre_reset_f",
        "pre_readout_i",
        "pre_readout_q",
        "integrated_i",
        "integrated_q",
        "integrated_i_mean_error",
        "integrated_q_mean_error",
        "raw_log_evidence",
        "raw_reference_log_evidence",
        "raw_within_window_residual",
        "posterior_g",
        "posterior_e",
        "posterior_f",
        "level_g",
        "level_e",
        "level_f",
        "mean_photon",
        "reset_requested",
        "reset_hidden_success",
        "reset_ack",
        "rao_blackwell_reset_success",
        "leakage_resident",
        "leakage_residence_probability",
        "leakage_age",
        "drift_0",
        "drift_1",
        "drift_2",
        "drift_3",
        "drift_4",
        "logical_survival",
        "logical_block_00_real",
        "logical_block_00_imag",
        "logical_block_01_real",
        "logical_block_01_imag",
        "logical_block_10_real",
        "logical_block_10_imag",
        "logical_block_11_real",
        "logical_block_11_imag",
        "density_trace_error",
        "density_hermiticity_frobenius",
        "density_minimum_eigenvalue",
        "density_quantization_frobenius_error",
        "density_quantization_certified_frobenius_bound",
        "density_quantization_trace_distance_bound",
        "posterior_normalization_error",
        "level_normalization_error",
        "reference_posterior_l1_error",
        "reference_log_evidence_error",
        "conservation_pass",
        "exception_type",
        "exception_message",
    }
)
EXPECTED_LEDGER_FIELDS = (
    "row_id", "row_schema", "layer", "cell_base", "cell_id", "backend",
    "backend_id", "cutoff", "convergence_role", "seed", "seed_position",
    "trajectory_id", "round_index", "terminal_round", "action", "probe_id",
    "scenario", "initial_state", "logical_label", "rng_namespace",
    "archive_chunk", "archive_row_index", "density_index", "raw_iq_index",
    "heldout_iq_index", "heldout_window_sha256", "pre_readout_i",
    "pre_readout_q", "pre_measurement_g", "pre_measurement_e",
    "pre_measurement_f", "pre_reset_g", "pre_reset_e", "pre_reset_f",
    "integrated_i", "integrated_q", "integrated_i_mean_error",
    "integrated_q_mean_error", "raw_log_evidence",
    "raw_reference_log_evidence", "raw_within_window_residual", "posterior_g",
    "posterior_e", "posterior_f", "level_g", "level_e", "level_f",
    "mean_photon", "reset_requested", "reset_hidden_success", "reset_ack",
    "rao_blackwell_reset_success", "leakage_resident",
    "leakage_residence_probability", "leakage_age", "predictive_mean_i",
    "predictive_mean_q", "predictive_cov_ii", "predictive_cov_iq",
    "predictive_cov_qq", "heldout_reference_log_evidence",
    "heldout_proper_score_per_sample", "heldout_llr_ge_per_sample",
    "heldout_llr_gf_per_sample", "heldout_llr_ef_per_sample", "drift_0",
    "drift_1", "drift_2", "drift_3", "drift_4", "logical_survival",
    "logical_block_00_real", "logical_block_00_imag",
    "logical_block_01_real", "logical_block_01_imag",
    "logical_block_10_real", "logical_block_10_imag",
    "logical_block_11_real", "logical_block_11_imag",
    "density_trace_error", "density_hermiticity_frobenius",
    "density_minimum_eigenvalue", "density_quantization_frobenius_error",
    "density_quantization_certified_frobenius_bound",
    "density_quantization_trace_distance_bound",
    "posterior_normalization_error", "level_normalization_error",
    "reference_posterior_l1_error", "reference_log_evidence_error",
    "conservation_pass", "exception_type", "exception_message",
)
INT_FIELDS = frozenset(
    {
        "cutoff",
        "seed",
        "seed_position",
        "round_index",
        "archive_row_index",
        "density_index",
        "raw_iq_index",
        "heldout_iq_index",
        "leakage_age",
    }
)
BOOL_FIELDS = frozenset(
    {
        "terminal_round",
        "reset_requested",
        "reset_hidden_success",
        "leakage_resident",
        "conservation_pass",
    }
)
FLOAT_FIELDS = frozenset(
    REQUIRED_LEDGER_FIELDS
    - INT_FIELDS
    - BOOL_FIELDS
    - {
        "row_id",
        "row_schema",
        "layer",
        "cell_base",
        "cell_id",
        "backend",
        "backend_id",
        "trajectory_id",
        "action",
        "probe_id",
        "scenario",
        "initial_state",
        "logical_label",
        "rng_namespace",
        "archive_chunk",
        "heldout_window_sha256",
        "reset_ack",
        "exception_type",
        "exception_message",
    }
)


class EvidenceIncomplete(RuntimeError):
    """Structural or numerical failure that must not become a scientific miss."""


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    family: str
    stage: str
    metric: str
    margin: float
    direction: str = "upper"
    deterministic: bool = False


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    family: str
    stage: str
    metric: str
    direction: str
    estimate: float
    standard_error: float
    bound: float
    margin: float
    cluster_count: int
    passed: bool
    denominator: str

    def to_dict(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "family": self.family,
            "stage": self.stage,
            "metric": self.metric,
            "direction": self.direction,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "bound": self.bound,
            "margin": self.margin,
            "cluster_count": self.cluster_count,
            "passed": self.passed,
            "denominator": self.denominator,
        }


class _LazyRowArrayMap(Mapping[str, np.ndarray]):
    def __init__(self, archive: "ArchiveEvidence", kind: str) -> None:
        self.archive = archive
        self.kind = kind

    def __len__(self) -> int:
        return len(
            self.archive.density_locator_by_row
            if self.kind == "density"
            else self.archive.locator_by_row
        )

    def __iter__(self) -> Iterator[str]:
        return iter(
            self.archive.density_locator_by_row
            if self.kind == "density"
            else self.archive.locator_by_row
        )

    def __getitem__(self, row_id: str) -> np.ndarray:
        if self.kind == "density":
            try:
                chunk_id, index = self.archive.density_locator_by_row[row_id]
            except KeyError as exc:
                raise KeyError(row_id) from exc
            return np.asarray(
                self.archive._load_chunk(chunk_id)["densities"][index],
                dtype=np.complex128,
            )
        try:
            chunk_id, index = self.archive.locator_by_row[row_id]
        except KeyError as exc:
            raise KeyError(row_id) from exc
        array_name = "raw_iq" if self.kind == "raw" else "heldout_iq"
        return np.asarray(
            self.archive._load_chunk(chunk_id)[array_name][index],
            dtype=np.float64,
        )


@dataclass
class ArchiveEvidence:
    path: Path
    locator_by_row: dict[str, tuple[str, int]]
    density_locator_by_row: dict[str, tuple[str, int]]
    chunk_members: dict[str, str]
    chunk_bindings_by_id: dict[str, dict[str, object]]
    mapping_arrays: dict[str, np.ndarray]
    chunk_bindings: tuple[dict[str, object], ...]
    archive_manifest: dict[str, Any]

    def __post_init__(self) -> None:
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.raw_iq_by_row = _LazyRowArrayMap(self, "raw")
        self.heldout_iq_by_row = _LazyRowArrayMap(self, "heldout")
        self.density_by_row = _LazyRowArrayMap(self, "density")

    def _load_chunk(self, chunk_id: str) -> dict[str, np.ndarray]:
        if chunk_id in self._cache:
            value = self._cache.pop(chunk_id)
            self._cache[chunk_id] = value
            return value
        try:
            member = self.chunk_members[chunk_id]
            expected = self.chunk_bindings_by_id[chunk_id]
            with zipfile.ZipFile(self.path, mode="r") as archive:
                payload = archive.read(member)
        except (KeyError, OSError, zipfile.BadZipFile) as exc:
            raise EvidenceIncomplete("lazy archive chunk unavailable") from exc
        if (
            len(payload) != expected["bytes"]
            or _sha_bytes(payload) != expected["sha256"]
        ):
            raise EvidenceIncomplete("lazy archive chunk binding drift")
        arrays = _npz_arrays(payload, member)
        self._cache[chunk_id] = arrays
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return arrays


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


def _sha_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _sha(value: object) -> str:
    return _sha_bytes(_canonical(value))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceIncomplete(f"cannot load JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceIncomplete(f"{path.name} must contain a JSON object")
    return value


def _safe_relative(root: Path, value: object, *, purpose: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceIncomplete(f"{purpose} path must be a non-empty string")
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise EvidenceIncomplete(f"{purpose} path escapes repository")
    if pure.name.lower() in PROHIBITED_HISTORICAL_BASENAMES:
        raise EvidenceIncomplete(f"{purpose} attempts historical-cell access")
    candidate = (root / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceIncomplete(f"{purpose} path escapes repository") from exc
    return candidate


def _binding(path: Path, root: Path) -> dict[str, object]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvidenceIncomplete(f"missing bound artifact {path.name}") from exc
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": len(payload),
        "sha256": _sha_bytes(payload),
    }


def _verify_binding(
    root: Path,
    value: object,
    *,
    purpose: str,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(value, dict):
        raise EvidenceIncomplete(f"{purpose} binding missing")
    if set(value) < {"path", "bytes", "sha256"}:
        raise EvidenceIncomplete(f"{purpose} binding incomplete")
    path = _safe_relative(root, value["path"], purpose=purpose)
    if expected_path is not None and path.resolve() != expected_path.resolve():
        raise EvidenceIncomplete(f"{purpose} binding points to wrong artifact")
    actual = _binding(path, root)
    expected = {
        "path": str(value["path"]).replace("\\", "/"),
        "bytes": value["bytes"],
        "sha256": value["sha256"],
    }
    if actual != expected:
        raise EvidenceIncomplete(f"{purpose} binding drift")
    return path


def _strict_int(value: str, field: str, *, allow_blank: bool = False) -> int | None:
    if value == "" and allow_blank:
        return None
    if value.strip() != value or not value or any(c not in "-0123456789" for c in value):
        raise EvidenceIncomplete(f"{field} is not an exact integer")
    try:
        result = int(value)
    except ValueError as exc:
        raise EvidenceIncomplete(f"{field} is not an exact integer") from exc
    if str(result) != value and not (value == "-0" and result == 0):
        raise EvidenceIncomplete(f"{field} has non-canonical integer syntax")
    return result


def _strict_bool(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise EvidenceIncomplete(f"{field} is not an exact bool")


def _strict_float(value: str, field: str, *, allow_blank: bool = False) -> float:
    if value == "" and allow_blank:
        return float("nan")
    try:
        result = float(value)
    except ValueError as exc:
        raise EvidenceIncomplete(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise EvidenceIncomplete(f"{field} is non-finite")
    return result


def load_ledger(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise EvidenceIncomplete("fresh cell ledger missing") from exc
    with handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if not fields or len(fields) != len(set(fields)):
            raise EvidenceIncomplete("ledger header missing or duplicated")
        if fields != EXPECTED_LEDGER_FIELDS:
            raise EvidenceIncomplete("ledger header/order differs from frozen schema")
        missing = REQUIRED_LEDGER_FIELDS - set(fields)
        if missing:
            raise EvidenceIncomplete(
                "ledger required fields missing: " + ",".join(sorted(missing))
            )
        rows: list[dict[str, Any]] = []
        row_ids: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise EvidenceIncomplete(f"ledger row {line_number} has extra cells")
            row: dict[str, Any] = dict(raw)
            if row["row_schema"] != ROW_SCHEMA:
                raise EvidenceIncomplete(f"ledger row {line_number} schema drift")
            row_id = row["row_id"]
            if not row_id or row_id in row_ids:
                raise EvidenceIncomplete("blank or duplicate row_id")
            row_ids.add(row_id)
            if row["backend"] not in {"A", "B"}:
                raise EvidenceIncomplete("backend must be exactly A or B")
            for field in INT_FIELDS:
                row[field] = _strict_int(
                    row[field], field, allow_blank=field in {"density_index"}
                )
            for field in BOOL_FIELDS:
                row[field] = _strict_bool(row[field], field)
            for field in FLOAT_FIELDS:
                allow_blank = field.startswith("logical_block_") or field in {
                    "logical_survival",
                    "rao_blackwell_reset_success",
                    "density_quantization_frobenius_error",
                    "density_quantization_certified_frobenius_bound",
                    "density_quantization_trace_distance_bound",
                }
                row[field] = _strict_float(
                    row[field], field, allow_blank=allow_blank
                )
            if row["exception_type"] or row["exception_message"]:
                raise EvidenceIncomplete("formal exception row present")
            if row["conservation_pass"] is not True:
                raise EvidenceIncomplete("conservation gate failed in raw evidence")
            rows.append(row)
    if not rows:
        raise EvidenceIncomplete("fresh cell ledger is empty")
    return rows, fields


def _zip_member_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceIncomplete("archive member name invalid")
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or pure.name == "":
        raise EvidenceIncomplete("archive member path traversal")
    return pure.as_posix()


def _npz_arrays(payload: bytes, member: str) -> dict[str, np.ndarray]:
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except (OSError, ValueError, KeyError) as exc:
        raise EvidenceIncomplete(f"invalid NPZ chunk {member}") from exc
    for name, value in arrays.items():
        if value.dtype.hasobject:
            raise EvidenceIncomplete(f"object array forbidden in {member}:{name}")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise EvidenceIncomplete(f"non-finite archive array {member}:{name}")
    return arrays


def _string_vector(value: np.ndarray, name: str) -> list[str]:
    if value.ndim != 1 or value.dtype.kind not in {"U", "S"}:
        raise EvidenceIncomplete(f"{name} must be a one-dimensional string array")
    result = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in value.tolist()
    ]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise EvidenceIncomplete(f"{name} contains blank/duplicate identifiers")
    return result


def load_archive(path: Path) -> ArchiveEvidence:
    try:
        handle = zipfile.ZipFile(path, mode="r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise EvidenceIncomplete("fresh raw archive missing/corrupt") from exc
    locator_by_row: dict[str, tuple[str, int]] = {}
    density_locator_by_row: dict[str, tuple[str, int]] = {}
    chunk_members: dict[str, str] = {}
    chunk_bindings_by_id: dict[str, dict[str, object]] = {}
    mapping: dict[str, np.ndarray] = {}
    bindings: list[dict[str, object]] = []
    with handle:
        infos = handle.infolist()
        names = [_zip_member_name(info.filename) for info in infos]
        if len(names) != len(set(names)):
            raise EvidenceIncomplete("duplicate archive member")
        if "archive_manifest.json" not in names:
            raise EvidenceIncomplete("archive manifest missing")
        try:
            manifest = json.loads(
                handle.read("archive_manifest.json").decode("utf-8")
            )
        except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceIncomplete("archive manifest invalid") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("task_id") != TASK_ID
            or manifest.get("schema_version") != ARCHIVE_MANIFEST_SCHEMA
        ):
            raise EvidenceIncomplete("archive manifest schema mismatch")
        analysis = manifest.get("analysis_sha256")
        unsigned_manifest = dict(manifest)
        unsigned_manifest.pop("analysis_sha256", None)
        if analysis != _sha_bytes(_canonical(unsigned_manifest)):
            raise EvidenceIncomplete("archive manifest analysis hash invalid")
        chunks = manifest.get("entries")
        if not isinstance(chunks, list) or not chunks:
            raise EvidenceIncomplete("archive chunk registry empty")
        mapping_member = _zip_member_name(manifest.get("mapping_member"))
        expected_names = {"archive_manifest.json", mapping_member}
        for entry in chunks:
            if not isinstance(entry, dict):
                raise EvidenceIncomplete("archive chunk binding invalid")
            member = _zip_member_name(entry.get("member"))
            expected_names.add(member)
            try:
                payload = handle.read(member)
            except KeyError as exc:
                raise EvidenceIncomplete(f"archive chunk missing: {member}") from exc
            actual = {
                "path": member,
                "bytes": len(payload),
                "sha256": _sha_bytes(payload),
            }
            source = entry.get("source")
            if not isinstance(source, dict):
                raise EvidenceIncomplete("archive chunk source binding missing")
            expected = {
                "path": member,
                "bytes": source.get("bytes"),
                "sha256": source.get("sha256"),
            }
            if actual != expected:
                raise EvidenceIncomplete(f"archive chunk binding drift: {member}")
            bindings.append(actual)
            arrays = _npz_arrays(payload, member)
            schema = arrays.get("schema")
            if schema is None or schema.tolist() != [ARCHIVE_SCHEMA]:
                raise EvidenceIncomplete(f"cell chunk schema mismatch: {member}")
            required = {"row_ids", "raw_iq", "heldout_iq"}
            if not required <= set(arrays):
                raise EvidenceIncomplete(f"cell chunk arrays missing in {member}")
            row_ids = _string_vector(arrays["row_ids"], f"{member}:row_ids")
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise EvidenceIncomplete("archive chunk id missing")
            if chunk_id in chunk_members:
                raise EvidenceIncomplete("duplicate archive chunk id")
            chunk_members[chunk_id] = member
            chunk_bindings_by_id[chunk_id] = actual
            raw = arrays["raw_iq"]
            heldout = arrays["heldout_iq"]
            if (
                raw.ndim != 3
                or heldout.ndim != 3
                or raw.shape != heldout.shape
                or raw.shape[0] != len(row_ids)
                or raw.shape[2] != 2
            ):
                raise EvidenceIncomplete(f"IQ archive shape mismatch in {member}")
            for index, row_id in enumerate(row_ids):
                if row_id in locator_by_row:
                    raise EvidenceIncomplete("row appears in multiple archive chunks")
                locator_by_row[row_id] = (chunk_id, index)
            if "densities" in arrays or "density_row_ids" in arrays:
                if not {"densities", "density_row_ids"} <= set(arrays):
                    raise EvidenceIncomplete("partial density chunk")
                densities = arrays["densities"]
                if arrays["densities"].dtype != np.dtype(np.complex64):
                    raise EvidenceIncomplete("retained densities must be complex64")
                density_ids = _string_vector(
                    arrays["density_row_ids"], f"{member}:density_row_ids"
                ) if len(arrays["density_row_ids"]) else []
                if (
                    densities.ndim != 3
                    or densities.shape[1] != densities.shape[2]
                    or len(density_ids) != len(densities)
                ):
                    raise EvidenceIncomplete("density archive shape mismatch")
                if not set(density_ids) <= set(row_ids):
                    raise EvidenceIncomplete("density row id absent from cell chunk")
                for density_index, row_id in enumerate(density_ids):
                    if row_id in density_locator_by_row:
                        raise EvidenceIncomplete("duplicate archived density")
                    density_locator_by_row[row_id] = (
                        chunk_id, density_index
                    )
        try:
            mapping_payload = handle.read(mapping_member)
        except KeyError as exc:
            raise EvidenceIncomplete("mapping archive member missing") from exc
        mapping_source = manifest.get("mapping_source")
        if not isinstance(mapping_source, dict) or {
            "bytes", "sha256"
        } - set(mapping_source):
            raise EvidenceIncomplete("mapping source binding missing")
        if (
            len(mapping_payload) != mapping_source["bytes"]
            or _sha_bytes(mapping_payload) != mapping_source["sha256"]
        ):
            raise EvidenceIncomplete("mapping member binding drift")
        mapping_chunk = _npz_arrays(mapping_payload, mapping_member)
        mapping_schema = mapping_chunk.pop("schema", None)
        if mapping_schema is None or mapping_schema.tolist() != [ARCHIVE_SCHEMA]:
            raise EvidenceIncomplete("mapping chunk schema mismatch")
        mapping_chunk.pop("iq_reference_id", None)
        mapping.update(mapping_chunk)
        if set(names) != expected_names:
            raise EvidenceIncomplete("unmanifested archive member present")
        if (
            manifest.get("chunk_count") not in (None, len(chunks))
            or manifest.get("row_count") not in (None, len(locator_by_row))
        ):
            raise EvidenceIncomplete("archive manifest count mismatch")
    return ArchiveEvidence(
        path=path.resolve(),
        locator_by_row=locator_by_row,
        density_locator_by_row=density_locator_by_row,
        chunk_members=chunk_members,
        chunk_bindings_by_id=chunk_bindings_by_id,
        mapping_arrays=mapping,
        chunk_bindings=tuple(bindings),
        archive_manifest=manifest,
    )


def _validate_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if (
        vector.shape[-1] != 3
        or not np.all(np.isfinite(vector))
        or np.any(vector < -1.0e-12)
        or np.any(vector > 1.0 + 1.0e-12)
        or np.any(np.abs(np.sum(vector, axis=-1) - 1.0) > 5.0e-8)
    ):
        raise EvidenceIncomplete(f"{name} is not a normalized qutrit probability")
    return np.clip(vector, 0.0, 1.0)


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != left.shape[1]:
        raise EvidenceIncomplete("density dimensions disagree")
    delta = 0.5 * ((left - right) + (left - right).conj().T)
    value = 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(delta))))
    if not math.isfinite(value):
        raise EvidenceIncomplete("density trace distance non-finite")
    return value


def _density_quantization_bound(
    row: Mapping[str, Any], density: np.ndarray
) -> float:
    """Validate and return the one-sided trace-distance quantization bound."""

    frobenius_error = float(row["density_quantization_frobenius_error"])
    certified = float(row["density_quantization_certified_frobenius_bound"])
    trace_bound = float(row["density_quantization_trace_distance_bound"])
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in (
            frobenius_error, certified, trace_bound
        ))
        or frobenius_error > certified + 1.0e-15
    ):
        raise EvidenceIncomplete("density quantization certificate invalid")
    expected = 0.5 * math.sqrt(density.shape[0]) * certified
    if abs(trace_bound - expected) > max(1.0e-15, 2.0e-12 * expected):
        raise EvidenceIncomplete("density trace-distance quantization bound drift")
    return trace_bound


def _embed_density(matrix: np.ndarray, lower: int, upper: int) -> np.ndarray:
    if matrix.shape != (3 * lower, 3 * lower) or not 0 < lower < upper:
        raise EvidenceIncomplete("cutoff density embedding mismatch")
    output = np.zeros((3 * upper, 3 * upper), dtype=np.complex128)
    output[: 3 * lower, : 3 * lower] = matrix
    return output


def _effective_centers(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> np.ndarray:
    base = np.asarray(config["common_physics"]["iq_centers"], dtype=np.float64)
    if base.shape != (3, 2) or not np.all(np.isfinite(base)):
        raise EvidenceIncomplete("IQ centers invalid")
    offsets = np.asarray(
        [[row["pre_readout_i"], row["pre_readout_q"]] for row in rows],
        dtype=np.float64,
    )
    return base[None, :, :] + offsets[:, None, :]


def _predictive_features(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    priors = _validate_probabilities(
        np.asarray(
            [
                [
                    row["pre_measurement_g"],
                    row["pre_measurement_e"],
                    row["pre_measurement_f"],
                ]
                for row in rows
            ]
        ),
        "pre-measurement prior",
    )
    centers = _effective_centers(rows, config)
    sigma = float(config["common_physics"]["iq_sigma"])
    count = int(config["common_physics"]["iq_samples"])
    if not math.isfinite(sigma) or sigma <= 0.0 or count <= 0:
        raise EvidenceIncomplete("IQ scale/count invalid")
    means = np.einsum("nk,nkd->nd", priors, centers)
    centered = centers - means[:, None, :]
    covariances = np.einsum(
        "nk,nki,nkj->nij", priors, centered, centered
    )
    covariances[:, 0, 0] += sigma * sigma / count
    covariances[:, 1, 1] += sigma * sigma / count
    return priors, centers, means, covariances


def _mixture_cdf_matrix(
    priors: np.ndarray,
    centers: np.ndarray,
    *,
    axis: int,
    grid: np.ndarray,
    sigma: float,
    sample_count: int,
) -> np.ndarray:
    z = (
        grid[None, None, :] - centers[:, :, axis, None]
    ) / (sigma / math.sqrt(sample_count))
    return np.sum(priors[:, :, None] * ndtr(z), axis=1)


def _component_logs(
    observations: np.ndarray,
    centers: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if observations.ndim != 3 or observations.shape[2] != 2:
        raise EvidenceIncomplete("heldout IQ has wrong shape")
    if centers.shape != (len(observations), 3, 2):
        raise EvidenceIncomplete("heldout center shape mismatch")
    count = observations.shape[1]
    delta = observations[:, None, :, :] - centers[:, :, None, :]
    squared = np.sum(delta * delta, axis=(2, 3))
    return (
        -count * math.log(2.0 * math.pi * sigma * sigma)
        - squared / (2.0 * sigma * sigma)
    )


def _logsumexp_rows(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum[:, None]), axis=1)
    )
    if not np.all(np.isfinite(result)):
        raise EvidenceIncomplete("logsumexp non-finite")
    return result


def _posterior_and_score(
    rows: Sequence[Mapping[str, Any]],
    archive: ArchiveEvidence,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    priors, centers, _, _ = _predictive_features(rows, config)
    observations = np.stack(
        [archive.heldout_iq_by_row[str(row["row_id"])] for row in rows]
    )
    count = int(config["common_physics"]["iq_samples"])
    if observations.shape[1] != count:
        raise EvidenceIncomplete("heldout IQ sample-count mismatch")
    sigma = float(config["common_physics"]["iq_sigma"])
    logs = _component_logs(observations, centers, sigma)
    weighted = np.where(
        priors > 0.0, logs + np.log(np.maximum(priors, 1.0e-300)), -np.inf
    )
    evidence = _logsumexp_rows(weighted)
    posterior = np.exp(weighted - evidence[:, None])
    posterior = _validate_probabilities(posterior, "heldout posterior")
    llr = np.stack(
        (
            logs[:, 0] - logs[:, 1],
            logs[:, 0] - logs[:, 2],
            logs[:, 1] - logs[:, 2],
        ),
        axis=1,
    ) / count
    return evidence / count, llr, posterior


def _assert_common_heldout(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    archive: ArchiveEvidence,
) -> None:
    if len(left) != len(right):
        raise EvidenceIncomplete("heldout A/B row count mismatch")
    for a, b in zip(left, right):
        window_a = archive.heldout_iq_by_row[str(a["row_id"])]
        window_b = archive.heldout_iq_by_row[str(b["row_id"])]
        digest_a = _sha_bytes(
            np.asarray(window_a, dtype="<f8").tobytes(order="C")
        )
        digest_b = _sha_bytes(
            np.asarray(window_b, dtype="<f8").tobytes(order="C")
        )
        if (
            not np.array_equal(window_a, window_b)
            or digest_a != digest_b
            or a["heldout_window_sha256"] != digest_a
            or b["heldout_window_sha256"] != digest_b
        ):
            raise EvidenceIncomplete(
                "proper score/LLR did not use one common heldout window"
            )


def _jackknife_scalar(values: np.ndarray, reducer: Any) -> tuple[float, float]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    count = len(matrix)
    if count < 2 or not np.all(np.isfinite(matrix)):
        raise EvidenceIncomplete("at least two finite seed clusters required")
    estimate = float(reducer(np.mean(matrix, axis=0)))
    leave = (np.sum(matrix, axis=0)[None, :] - matrix) / (count - 1)
    leave_values = np.asarray([reducer(row) for row in leave], dtype=np.float64)
    standard_error = math.sqrt(
        (count - 1) / count
        * float(np.sum((leave_values - np.mean(leave_values)) ** 2))
    )
    if not math.isfinite(estimate) or not math.isfinite(standard_error):
        raise EvidenceIncomplete("jackknife produced non-finite result")
    return estimate, standard_error


def _signed_difference(
    left: np.ndarray, right: np.ndarray
) -> tuple[float, float]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return _jackknife_scalar(difference, lambda value: abs(float(value[0])))


def _norm_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    ord_value: int | float | str = 2,
) -> tuple[float, float]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return _jackknife_scalar(
        difference,
        lambda value: float(np.linalg.norm(value, ord=ord_value)),
    )


def _positive_mean(values: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return _jackknife_scalar(vector, lambda value: float(value[0]))


def _ensemble_trace_distance(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    quantization_bounds: Sequence[float],
) -> tuple[float, float]:
    """Ensemble trace-distance point estimate with a conservative cluster SE."""

    if len(left) != len(right) or len(left) != len(quantization_bounds) or len(left) < 2:
        raise EvidenceIncomplete("ensemble density cluster coverage invalid")
    left_stack = np.stack(left).astype(np.complex128, copy=False)
    right_stack = np.stack(right).astype(np.complex128, copy=False)
    if left_stack.shape != right_stack.shape:
        raise EvidenceIncomplete("ensemble density shapes disagree")
    differences = left_stack - right_stack
    mean_difference = np.mean(differences, axis=0)
    hermitian = 0.5 * (mean_difference + mean_difference.conj().T)
    quant = np.asarray(quantization_bounds, dtype=np.float64)
    if not np.all(np.isfinite(quant)) or np.any(quant < 0.0):
        raise EvidenceIncomplete("ensemble quantization bound invalid")
    estimate = (
        0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(hermitian))))
        + float(np.mean(quant))
    )
    count = len(differences)
    centered = differences - mean_difference[None, :, :]
    frobenius_mean_se = math.sqrt(
        float(np.sum(np.abs(centered) ** 2)) / (count * (count - 1))
    )
    dimension = differences.shape[1]
    matrix_se_bound = 0.5 * math.sqrt(dimension) * frobenius_mean_se
    quant_se = float(np.std(quant, ddof=1) / math.sqrt(count))
    return estimate, matrix_se_bound + quant_se


def _row_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["cell_id"]), str(row["backend"])), []).append(
            dict(row)
        )
    for values in grouped.values():
        values.sort(key=lambda row: (row["seed_position"], row["round_index"]))
    return grouped


def _paired_rows(
    grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    scope: str,
    *,
    terminal_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left = [dict(row) for row in grouped.get((scope, "A"), ())]
    right = [dict(row) for row in grouped.get((scope, "B"), ())]
    if terminal_only:
        left = [row for row in left if row["terminal_round"] is True]
        right = [row for row in right if row["terminal_round"] is True]
    if not left or not right:
        raise EvidenceIncomplete(f"missing A/B scope {scope}")
    left_index = {(row["seed_position"], row["round_index"]): row for row in left}
    right_index = {
        (row["seed_position"], row["round_index"]): row for row in right
    }
    if (
        len(left_index) != len(left)
        or len(right_index) != len(right)
        or set(left_index) != set(right_index)
    ):
        raise EvidenceIncomplete(f"A/B cluster coverage mismatch in {scope}")
    keys = sorted(left_index)
    paired_left = [left_index[key] for key in keys]
    paired_right = [right_index[key] for key in keys]
    if "/fault/" in scope:
        schedule = ("0", "1", "+", "-", "+i", "-i")
        for a, b in zip(paired_left, paired_right):
            expected = schedule[int(a["seed_position"]) % len(schedule)]
            if (
                a["logical_label"] != expected
                or b["logical_label"] != expected
                or a["logical_label"] != b["logical_label"]
            ):
                raise EvidenceIncomplete(
                    "fault trajectory six-state label schedule mismatch"
                )
        positions = {int(row["seed_position"]) for row in paired_left}
        counts = [
            sum(position % len(schedule) == index for position in positions)
            for index in range(len(schedule))
        ]
        if not all(counts) or max(counts) - min(counts) > 1:
            raise EvidenceIncomplete("fault six-state denominator imbalance")
    return paired_left, paired_right


def _gate_scope(gate_id: str, metric: str) -> str:
    suffixes = {
        "density_trace_distance": "/density_trace_distance",
        "mean_photon_difference": "/mean_photon_difference",
        "level_probability_l1": "/level_probability_l1",
        "predictive_mean_l2": "/predictive_mean_l2",
        "predictive_covariance_frobenius": "/predictive_covariance_frobenius",
        "predictive_cdf_i_sup": "/predictive_cdf_i_sup",
        "predictive_cdf_q_sup": "/predictive_cdf_q_sup",
        "paired_proper_score_per_sample": "/paired_proper_score_per_sample",
        "pairwise_llr_per_sample": "/pairwise_llr_per_sample",
        "posterior_mean_l1": "/posterior_mean_l1",
        "logical_survival": "/logical_survival",
        "rao_blackwell_reset_success": "/rao_blackwell_reset_success",
        "leakage_residence_rate": "/leakage_residence_rate",
        "terminal_density_trace_distance": "/terminal_density_trace_distance",
        "terminal_mean_photon_difference": "/terminal_mean_photon_difference",
        "terminal_level_probability_l1": "/terminal_level_probability_l1",
        "terminal_predictive_mean_l2": "/terminal_predictive_mean_l2",
        "terminal_predictive_covariance_frobenius": (
            "/terminal_predictive_covariance_frobenius"
        ),
        "terminal_paired_proper_score_per_sample": (
            "/terminal_paired_proper_score_per_sample"
        ),
        "terminal_leakage_residence_rate": (
            "/terminal_leakage_residence_rate"
        ),
        "terminal_drift_normalized_l2": "/terminal_drift_normalized_l2",
        "terminal_logical_survival": "/terminal_logical_survival",
    }
    suffix = suffixes.get(metric)
    if suffix is None or not gate_id.endswith(suffix):
        raise EvidenceIncomplete(f"cannot derive scope for {gate_id}")
    return gate_id[: -len(suffix)]


def _build_blueprint(
    design: Mapping[str, Any],
    qualification: Mapping[str, Any],
) -> list[GateSpec]:
    blueprint = qualification.get("gate_blueprint")
    if isinstance(blueprint, dict) and isinstance(blueprint.get("rows"), list):
        specs: list[GateSpec] = []
        for raw in blueprint["rows"]:
            if not isinstance(raw, dict):
                raise EvidenceIncomplete("gate blueprint row invalid")
            if (
                not isinstance(raw.get("gate_id"), str)
                or not raw["gate_id"]
                or not isinstance(raw.get("family"), str)
                or not isinstance(raw.get("stage"), str)
                or not isinstance(raw.get("metric"), str)
                or raw.get("direction") not in {"upper", "lower"}
                or not isinstance(raw.get("deterministic"), bool)
                or isinstance(raw.get("margin"), bool)
                or not isinstance(raw.get("margin"), (int, float))
                or not math.isfinite(float(raw["margin"]))
                or float(raw["margin"]) <= 0.0
            ):
                raise EvidenceIncomplete("gate blueprint row types invalid")
            try:
                spec = GateSpec(
                    gate_id=str(raw["gate_id"]),
                    family=str(raw["family"]),
                    stage=str(raw["stage"]),
                    metric=str(raw["metric"]),
                    margin=float(raw["margin"]),
                    direction=str(raw.get("direction", "upper")),
                    deterministic=raw["deterministic"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise EvidenceIncomplete("gate blueprint row malformed") from exc
            specs.append(spec)
        if not specs:
            raise EvidenceIncomplete("gate blueprint empty")
        identifiers = [spec.gate_id for spec in specs]
        if len(identifiers) != len(set(identifiers)):
            raise EvidenceIncomplete("duplicate gate blueprint identifier")
        if blueprint.get("row_count") != len(specs):
            raise EvidenceIncomplete("gate blueprint row count mismatch")
        expected_hash = blueprint.get("canonical_blueprint_sha256")
        if expected_hash is not None and _sha(blueprint["rows"]) != expected_hash:
            raise EvidenceIncomplete("gate blueprint hash mismatch")
        return specs
    raise EvidenceIncomplete("qualification config lacks frozen gate blueprint")


def _logical_block(row: Mapping[str, Any]) -> np.ndarray:
    values = np.array(
        [
            [
                row["logical_block_00_real"]
                + 1.0j * row["logical_block_00_imag"],
                row["logical_block_01_real"]
                + 1.0j * row["logical_block_01_imag"],
            ],
            [
                row["logical_block_10_real"]
                + 1.0j * row["logical_block_10_imag"],
                row["logical_block_11_real"]
                + 1.0j * row["logical_block_11_imag"],
            ],
        ],
        dtype=np.complex128,
    )
    if not np.all(np.isfinite(values)):
        raise EvidenceIncomplete("logical block missing/non-finite")
    return values


def _ptm_seed_vectors(
    grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    cutoff: int,
    action: str,
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = ("0", "1", "+", "-", "+i", "-i")
    by_label: dict[str, dict[int, Mapping[str, Any]]] = {}
    for label in labels:
        scope = f"ab/c{cutoff}/logical/{label}/{action}"
        rows = grouped.get((scope, backend), ())
        index = {int(row["seed_position"]): row for row in rows}
        if len(index) != len(rows) or not index:
            raise EvidenceIncomplete(f"PTM source missing for {scope}/{backend}")
        by_label[label] = index
    positions = set.intersection(*(set(value) for value in by_label.values()))
    if any(set(value) != positions for value in by_label.values()):
        raise EvidenceIncomplete("PTM six-state seed coverage mismatch")
    paulis = (
        np.eye(2, dtype=np.complex128),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
    )
    vectors: list[np.ndarray] = []
    for position in sorted(positions):
        coordinates: dict[str, np.ndarray] = {}
        for label in labels:
            block = _logical_block(by_label[label][position])
            coordinates[label] = np.asarray(
                [np.trace(operator @ block).real for operator in paulis]
            )
        matrix = np.zeros((4, 4), dtype=np.float64)
        axes = (("+", "-", 1), ("+i", "-i", 2), ("0", "1", 3))
        identities = []
        for plus, minus, column in axes:
            identities.append(0.5 * (coordinates[plus] + coordinates[minus]))
            matrix[:, column] = 0.5 * (
                coordinates[plus] - coordinates[minus]
            )
        matrix[:, 0] = np.mean(np.stack(identities), axis=0)
        vectors.append(matrix.reshape(-1))
    return np.asarray(sorted(positions), dtype=np.int64), np.stack(vectors)


def _result(
    spec: GateSpec,
    estimate: float,
    standard_error: float,
    count: int,
    denominator: str,
) -> GateResult:
    if (
        not math.isfinite(estimate)
        or not math.isfinite(standard_error)
        or standard_error < 0.0
        or not math.isfinite(spec.margin)
        or spec.margin <= 0.0
    ):
        raise EvidenceIncomplete(f"invalid gate numeric result {spec.gate_id}")
    if spec.direction == "upper":
        bound = estimate + (0.0 if spec.deterministic else Z_TOST * standard_error)
        passed = bound <= spec.margin
    elif spec.direction == "lower":
        bound = estimate - (0.0 if spec.deterministic else Z_TOST * standard_error)
        passed = bound >= spec.margin
    else:
        raise EvidenceIncomplete(f"invalid gate direction {spec.gate_id}")
    return GateResult(
        gate_id=spec.gate_id,
        family=spec.family,
        stage=spec.stage,
        metric=spec.metric,
        direction=spec.direction,
        estimate=float(estimate),
        standard_error=float(standard_error),
        bound=float(bound),
        margin=float(spec.margin),
        cluster_count=int(count),
        passed=bool(passed),
        denominator=denominator,
    )


def _evaluate_standard_gate(
    spec: GateSpec,
    grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    archive: ArchiveEvidence,
    config: Mapping[str, Any],
) -> GateResult:
    metric = spec.metric
    terminal = metric.startswith("terminal_")
    scope = _gate_scope(spec.gate_id, metric)
    if "/reset/" in scope:
        parts = scope.split("/")
        if len(parts) != 4 or parts[0] != "ab" or parts[2] != "reset":
            raise EvidenceIncomplete("reset gate scope malformed")
        scope = f"{parts[0]}/{parts[1]}/shared/{parts[3]}/RESET"
    left, right = _paired_rows(grouped, scope, terminal_only=terminal)
    count = len(left)
    if count < 2:
        raise EvidenceIncomplete(f"underpowered scope {scope}")

    if metric in {"density_trace_distance", "terminal_density_trace_distance"}:
        densities_a = [
            archive.density_by_row[str(row["row_id"])] for row in left
        ]
        densities_b = [
            archive.density_by_row[str(row["row_id"])] for row in right
        ]
        quantization = [
            _density_quantization_bound(a, density_a)
            + _density_quantization_bound(b, density_b)
            for a, b, density_a, density_b in zip(
                left, right, densities_a, densities_b
            )
        ]
        estimate, se = _ensemble_trace_distance(
            densities_a, densities_b, quantization
        )
    elif metric in {
        "mean_photon_difference",
        "terminal_mean_photon_difference",
    }:
        estimate, se = _signed_difference(
            np.asarray([row["mean_photon"] for row in left]),
            np.asarray([row["mean_photon"] for row in right]),
        )
    elif metric in {"level_probability_l1", "terminal_level_probability_l1"}:
        fields = ("level_g", "level_e", "level_f")
        estimate, se = _norm_difference(
            np.asarray([[row[field] for field in fields] for row in left]),
            np.asarray([[row[field] for field in fields] for row in right]),
            ord_value=1,
        )
    elif metric in {
        "predictive_mean_l2",
        "terminal_predictive_mean_l2",
    }:
        _, _, mean_a, _ = _predictive_features(left, config)
        _, _, mean_b, _ = _predictive_features(right, config)
        estimate, se = _norm_difference(mean_a, mean_b)
    elif metric in {
        "predictive_covariance_frobenius",
        "terminal_predictive_covariance_frobenius",
    }:
        _, _, _, cov_a = _predictive_features(left, config)
        _, _, _, cov_b = _predictive_features(right, config)
        estimate, se = _norm_difference(
            cov_a.reshape(count, -1),
            cov_b.reshape(count, -1),
        )
    elif metric in {"predictive_cdf_i_sup", "predictive_cdf_q_sup"}:
        axis = 0 if metric.endswith("i_sup") else 1
        prior_a, centers_a, _, _ = _predictive_features(left, config)
        prior_b, centers_b, _, _ = _predictive_features(right, config)
        sigma = float(config["common_physics"]["iq_sigma"])
        samples = int(config["common_physics"]["iq_samples"])
        all_centers = np.concatenate(
            (centers_a[:, :, axis].reshape(-1), centers_b[:, :, axis].reshape(-1))
        )
        span = 8.0 * sigma / math.sqrt(samples)
        grid = np.linspace(
            float(np.min(all_centers) - span),
            float(np.max(all_centers) + span),
            513,
        )
        differences = _mixture_cdf_matrix(
            prior_a, centers_a, axis=axis, grid=grid, sigma=sigma,
            sample_count=samples,
        ) - _mixture_cdf_matrix(
            prior_b, centers_b, axis=axis, grid=grid, sigma=sigma,
            sample_count=samples,
        )
        estimate, se = _jackknife_scalar(
            differences,
            lambda value: float(np.max(np.abs(value))),
        )
    elif metric in {
        "paired_proper_score_per_sample",
        "terminal_paired_proper_score_per_sample",
    }:
        _assert_common_heldout(left, right, archive)
        score_a, _, _ = _posterior_and_score(left, archive, config)
        score_b, _, _ = _posterior_and_score(right, archive, config)
        estimate, se = _signed_difference(score_a, score_b)
    elif metric == "pairwise_llr_per_sample":
        _assert_common_heldout(left, right, archive)
        _, llr_a, _ = _posterior_and_score(left, archive, config)
        _, llr_b, _ = _posterior_and_score(right, archive, config)
        estimate, se = _norm_difference(llr_a, llr_b, ord_value=np.inf)
    elif metric == "posterior_mean_l1":
        _assert_common_heldout(left, right, archive)
        _, _, posterior_a = _posterior_and_score(left, archive, config)
        _, _, posterior_b = _posterior_and_score(right, archive, config)
        estimate, se = _norm_difference(
            posterior_a, posterior_b, ord_value=1
        )
    elif metric in {
        "rao_blackwell_reset_success",
    }:
        reset_e = float(config["common_physics"]["reset_success_e"])
        reset_f = float(config["common_physics"]["reset_success_f"])

        def rb(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
            probabilities = _validate_probabilities(
                np.asarray(
                    [
                        [row["pre_reset_g"], row["pre_reset_e"], row["pre_reset_f"]]
                        for row in rows
                    ]
                ),
                "pre-reset levels",
            )
            return (
                probabilities[:, 0]
                + reset_e * probabilities[:, 1]
                + reset_f * probabilities[:, 2]
            )

        estimate, se = _signed_difference(rb(left), rb(right))
    elif metric in {
        "leakage_residence_rate",
        "terminal_leakage_residence_rate",
    }:
        estimate, se = _signed_difference(
            np.asarray([float(row["leakage_resident"]) for row in left]),
            np.asarray([float(row["leakage_resident"]) for row in right]),
        )
    elif metric in {"logical_survival", "terminal_logical_survival"}:
        estimate, se = _signed_difference(
            np.asarray([row["logical_survival"] for row in left]),
            np.asarray([row["logical_survival"] for row in right]),
        )
    elif metric == "terminal_drift_normalized_l2":
        normalization = np.asarray(
            config["verification_contract"]["drift_normalization"],
            dtype=np.float64,
        )
        if (
            normalization.shape != (5,)
            or not np.all(np.isfinite(normalization))
            or np.any(normalization <= 0.0)
        ):
            raise EvidenceIncomplete("drift normalization invalid")
        fields = tuple(f"drift_{index}" for index in range(5))
        a = np.asarray([[row[field] for field in fields] for row in left])
        b = np.asarray([[row[field] for field in fields] for row in right])
        estimate, se = _norm_difference(a / normalization, b / normalization)
    else:
        raise EvidenceIncomplete(f"unsupported standard metric {metric}")
    return _result(spec, estimate, se, count, f"{scope}: paired seed_position")


def _evaluate_ptm_gate(
    spec: GateSpec,
    grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> GateResult:
    parts = spec.gate_id.split("/")
    if len(parts) < 6 or parts[0] != "ab" or parts[2] != "logical":
        raise EvidenceIncomplete("PTM gate identifier malformed")
    cutoff = int(parts[1][1:])
    action = parts[4]
    positions_a, vectors_a = _ptm_seed_vectors(grouped, cutoff, action, "A")
    positions_b, vectors_b = _ptm_seed_vectors(grouped, cutoff, action, "B")
    if not np.array_equal(positions_a, positions_b):
        raise EvidenceIncomplete("PTM A/B seed coverage mismatch")
    estimate, se = _norm_difference(
        vectors_a,
        vectors_b,
        ord_value=np.inf,
    )
    return _result(
        spec,
        estimate,
        se,
        len(positions_a),
        f"c{cutoff}/{action}: six-state seed_position PTM",
    )


def _evaluate_mapping_gate(
    spec: GateSpec,
    archive: ArchiveEvidence,
) -> GateResult:
    parts = spec.gate_id.split("/")
    if len(parts) != 3 or parts[0] != "mapping":
        raise EvidenceIncomplete("mapping gate identifier malformed")
    cutoff = int(parts[1][1:])
    key_a = f"mapping_isometry_a_cutoff_{cutoff}"
    key_b = f"mapping_isometry_b_cutoff_{cutoff}"
    projector_a = f"mapping_projector_a_cutoff_{cutoff}"
    projector_b = f"mapping_projector_b_cutoff_{cutoff}"
    try:
        isometry_a = archive.mapping_arrays[key_a]
        isometry_b = archive.mapping_arrays[key_b]
        proj_a = archive.mapping_arrays[projector_a]
        proj_b = archive.mapping_arrays[projector_b]
    except KeyError as exc:
        raise EvidenceIncomplete("mapping archive array missing") from exc
    if spec.metric == "principal_singular":
        overlap = isometry_a.conj().T @ isometry_b
        estimate = float(np.min(np.linalg.svd(overlap, compute_uv=False)))
    elif spec.metric == "projector_frobenius":
        estimate = float(np.linalg.norm(proj_a - proj_b, ord="fro"))
    else:
        raise EvidenceIncomplete("unknown deterministic mapping metric")
    return _result(spec, estimate, 0.0, 0, f"mapping cutoff {cutoff}")


def _cutoff_scope(
    gate_id: str,
) -> tuple[int, int, str, str]:
    parts = gate_id.split("/")
    if len(parts) < 5 or parts[0] != "cutoff":
        raise EvidenceIncomplete("cutoff gate identifier malformed")
    try:
        lower, upper = (int(value) for value in parts[1].split("-"))
    except (ValueError, TypeError) as exc:
        raise EvidenceIncomplete("cutoff increment malformed") from exc
    if parts[2] == "shared":
        source = f"shared/{parts[3]}/{parts[4]}"
        kind = "density"
    elif parts[2] == "logical":
        source = f"logical/{parts[3]}/{parts[4]}"
        kind = "survival"
    elif parts[2] == "fault":
        source = f"fault/{parts[3]}"
        kind = "density" if parts[4] == "terminal_density" else "survival"
    else:
        raise EvidenceIncomplete("unknown cutoff source layer")
    return lower, upper, source, kind


def _evaluate_cutoff_gate(
    spec: GateSpec,
    grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    archive: ArchiveEvidence,
) -> GateResult:
    lower, upper, source, kind = _cutoff_scope(spec.gate_id)
    backend_estimates: list[float] = []
    backend_standard_errors: list[float] = []
    positions_reference: np.ndarray | None = None
    for backend in ("A", "B"):
        scope_lower = f"ab/c{lower}/{source}"
        scope_upper = f"ab/c{upper}/{source}"
        rows_lower = list(grouped.get((scope_lower, backend), ()))
        rows_upper = list(grouped.get((scope_upper, backend), ()))
        if source.startswith("fault/"):
            rows_lower = [row for row in rows_lower if row["terminal_round"]]
            rows_upper = [row for row in rows_upper if row["terminal_round"]]
        left = {int(row["seed_position"]): row for row in rows_lower}
        right = {int(row["seed_position"]): row for row in rows_upper}
        if (
            not left
            or len(left) != len(rows_lower)
            or len(right) != len(rows_upper)
            or set(left) != set(right)
        ):
            raise EvidenceIncomplete(
                f"cutoff seed coverage mismatch {scope_lower}/{scope_upper}/{backend}"
            )
        positions = np.asarray(sorted(left), dtype=np.int64)
        if positions_reference is None:
            positions_reference = positions
        elif not np.array_equal(positions_reference, positions):
            raise EvidenceIncomplete("cutoff A/B position mismatch")
        values: list[float] = []
        low_densities: list[np.ndarray] = []
        high_densities: list[np.ndarray] = []
        quantization: list[float] = []
        for position in positions:
            low_row = left[int(position)]
            high_row = right[int(position)]
            if kind == "density":
                low_density = archive.density_by_row[str(low_row["row_id"])]
                high_density = archive.density_by_row[str(high_row["row_id"])]
                embedded = _embed_density(low_density, lower, upper)
                low_densities.append(embedded)
                high_densities.append(high_density)
                quantization.append(
                    _density_quantization_bound(low_row, low_density)
                    + _density_quantization_bound(high_row, high_density)
                )
            else:
                values.append(
                    abs(
                        float(low_row["logical_survival"])
                        - float(high_row["logical_survival"])
                    )
                )
        if kind == "density":
            estimate_backend, se_backend = _ensemble_trace_distance(
                low_densities, high_densities, quantization
            )
        else:
            estimate_backend, se_backend = _positive_mean(np.asarray(values))
        backend_estimates.append(estimate_backend)
        backend_standard_errors.append(se_backend)
    assert positions_reference is not None
    estimate = max(backend_estimates)
    # This produces an upper bound no smaller than either backend-specific
    # TOST upper bound and therefore cannot rescue a failing convergence leg.
    se = max(backend_standard_errors)
    return _result(
        spec,
        estimate,
        se,
        len(positions_reference),
        f"cutoff {lower}->{upper}: max(A,B), paired seed_position",
    )


def evaluate_gates(
    rows: Sequence[Mapping[str, Any]],
    archive: ArchiveEvidence,
    qualification: Mapping[str, Any],
    design: Mapping[str, Any],
) -> list[GateResult]:
    specs = _build_blueprint(design, qualification)
    grouped = _row_map(rows)
    results: list[GateResult] = []
    for spec in specs:
        if spec.metric in {"principal_singular", "projector_frobenius"}:
            result = _evaluate_mapping_gate(spec, archive)
        elif spec.metric in {
            "cutoff_density_trace_distance",
            "cutoff_logical_survival",
        }:
            result = _evaluate_cutoff_gate(spec, grouped, archive)
        elif spec.metric == "logical_ptm_entry":
            result = _evaluate_ptm_gate(spec, grouped)
        else:
            result = _evaluate_standard_gate(
                spec, grouped, archive, qualification
            )
        results.append(result)
    if len(results) != len(specs):
        raise EvidenceIncomplete("gate evaluation count mismatch")
    return results


def _validate_config(
    qualification: Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    allow_test_fixture: bool,
) -> None:
    if (
        qualification.get("task_id") != TASK_ID
        or qualification.get("schema_version") != CONFIG_SCHEMA
        or qualification.get("formal_result_accessed_before_freeze") is not False
    ):
        raise EvidenceIncomplete("qualification config identity/prefreeze drift")
    policy = qualification.get("historical_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("historical_no_go_rewritten") is not False
        or policy.get("historical_formal_cell_data_access_allowed") is not False
    ):
        raise EvidenceIncomplete("historical NO-GO isolation policy drift")
    if design.get("task_id") != TASK_ID:
        raise EvidenceIncomplete("design config task mismatch")
    matrix = qualification.get("formal_matrix")
    if (
        not isinstance(matrix, dict)
        or matrix.get("fault_initialization")
        != "logical_six_state_balanced_cycle"
        or matrix.get("fault_logical_label_schedule")
        != ["0", "1", "+", "-", "+i", "-i"]
    ):
        if not allow_test_fixture:
            raise EvidenceIncomplete("fault six-state initialization drift")
    procedure = design.get("statistical_procedure")
    if (
        not isinstance(procedure, dict)
        or procedure.get("global_test") != "intersection_union_equivalence"
        or procedure.get("cell_test") != "two_one_sided_tests"
        or procedure.get("cluster_unit")
        != "independent seed position; all rows sharing a seed remain together"
        or procedure.get("raw_log_evidence")
        != "diagnostic only; never a cross-gain primary gate"
        or procedure.get("mixed_unit_vector_max") is not False
        or procedure.get("missing_nonfinite_exception")
        != INCOMPLETE_VERDICT
    ):
        raise EvidenceIncomplete("IUT/TOST procedure drift")
    contract = qualification.get("verification_contract")
    if not isinstance(contract, dict):
        raise EvidenceIncomplete("verification contract missing")
    blueprint = qualification.get("gate_blueprint")
    specs = blueprint.get("rows") if isinstance(blueprint, dict) else None
    if not isinstance(specs, list):
        raise EvidenceIncomplete("verification gate blueprint missing")
    if not allow_test_fixture and len(specs) != 1589:
        raise EvidenceIncomplete("production blueprint must contain 1589 gates")
    if contract.get("cluster_unit") != "seed_position":
        raise EvidenceIncomplete("verification cluster unit drift")
    if (
        contract.get("global_test") != "intersection_union_equivalence"
        or contract.get("cell_test") != "two_one_sided_tests"
        or contract.get("cell_confidence_interval") != 0.9
        or contract.get("tost_z") != Z_TOST
        or contract.get("aggregate_rescue_forbidden") is not True
        or contract.get("missing_nonfinite_exception") != INCOMPLETE_VERDICT
    ):
        raise EvidenceIncomplete("verification IUT/TOST contract drift")
    if contract.get("raw_log_evidence_primary") is not False:
        raise EvidenceIncomplete("raw log evidence promoted to primary")
    if contract.get("fault_mixed_unit_composite") is not False:
        raise EvidenceIncomplete("fault mixed-unit composite forbidden")
    if contract.get("density_quantization_bound_must_be_added") is not True:
        raise EvidenceIncomplete("density quantization UCB requirement missing")


def _validate_design_lineage(
    root: Path,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    specification = qualification.get("design_power")
    if not isinstance(specification, dict):
        raise EvidenceIncomplete("design-power specification missing")
    path = _safe_relative(
        root, specification.get("path"), purpose="fresh design-power report"
    )
    report = _load_json(path)
    analysis = report.get("analysis_sha256")
    unsigned = dict(report)
    unsigned.pop("analysis_sha256", None)
    if (
        analysis != _sha(unsigned)
        or report.get("task_id") != TASK_ID
        or report.get("schema_version") != specification.get("schema_version")
        or report.get("verdict") != specification.get("required_verdict")
    ):
        raise EvidenceIncomplete("design-power lineage mismatch")
    blueprint = report.get("blueprint")
    contract = qualification.get("verification_contract")
    if (
        not isinstance(blueprint, dict)
        or not isinstance(contract, dict)
        or qualification.get("gate_blueprint", {}).get(
            "source_design_blueprint_sha256"
        )
        != blueprint.get("sha256")
    ):
        raise EvidenceIncomplete("source design blueprint binding mismatch")
    return report


def _validate_preformal_seal(
    root: Path,
    config_path: Path,
    qualification: Mapping[str, Any],
) -> dict[str, Any]:
    seal_spec = qualification.get("preformal_seal")
    if not isinstance(seal_spec, dict):
        raise EvidenceIncomplete("preformal seal specification missing")
    seal_path = _safe_relative(root, seal_spec.get("path"), purpose="preformal seal")
    seal = _load_json(seal_path)
    if (
        seal.get("task_id") != TASK_ID
        or seal.get("schema_version") != PREFORMAL_SEAL_SCHEMA
        or seal.get("status") != seal_spec.get("required_status")
        or seal.get("formal_result_accessed") is not False
        or seal.get("historical_formal_cell_data_accessed") is not False
        or seal.get("all_gates_passed") is not True
        or seal.get("all_mutations_detected") is not True
        or seal.get("scientific_verdict") is not None
    ):
        raise EvidenceIncomplete("preformal seal status invalid")
    analysis = seal.get("analysis_sha256")
    unsigned = dict(seal)
    unsigned.pop("analysis_sha256", None)
    if analysis != _sha(unsigned):
        raise EvidenceIncomplete("preformal seal analysis hash invalid")
    bindings = seal.get("live_bindings")
    if not isinstance(bindings, dict):
        raise EvidenceIncomplete("preformal seal live bindings missing")
    required_binding_names = {
        "fresh_config",
        "fresh_runner",
        "historical_lineage_receipt",
        "design_power_report",
        "preformal_audit",
    }
    if not required_binding_names <= set(bindings):
        raise EvidenceIncomplete("preformal seal omits a required live binding")
    for name, binding in bindings.items():
        _verify_binding(root, binding, purpose=f"sealed live input {name}")
    sealed_paths = {
        str(binding["path"]).replace("\\", "/")
        for binding in bindings.values()
        if isinstance(binding, dict) and isinstance(binding.get("path"), str)
    }
    runtime = qualification.get("runtime_dependencies")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("paths"), list):
        raise EvidenceIncomplete("runtime dependency registry missing")
    if not {
        str(path).replace("\\", "/") for path in runtime["paths"]
    } <= sealed_paths:
        raise EvidenceIncomplete("seal omits a runtime dependency")
    config_binding = bindings.get("fresh_config")
    _verify_binding(
        root,
        config_binding,
        purpose="sealed qualification config",
        expected_path=config_path,
    )
    return seal


def _validate_manifest(
    root: Path,
    manifest_path: Path,
    config_path: Path,
    ledger_path: Path,
    archive_path: Path,
    seal: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if (
        manifest.get("task_id") != TASK_ID
        or manifest.get("schema_version") != EXECUTION_MANIFEST_SCHEMA
        or manifest.get("status") != "FORMAL_RAW_EVIDENCE_COMPLETE"
        or manifest.get("formal") is not True
        or manifest.get("scientific_verdict") is not None
        or manifest.get("expected_cells") != manifest.get("observed_cells")
        or manifest.get("expected_rows") != manifest.get("observed_rows")
        or manifest.get("exception_rows") != 0
    ):
        raise EvidenceIncomplete("execution manifest status invalid")
    _verify_binding(
        root, manifest.get("config"), purpose="manifest config",
        expected_path=config_path,
    )
    _verify_binding(
        root, manifest.get("cell_ledger"), purpose="manifest ledger",
        expected_path=ledger_path,
    )
    _verify_binding(
        root, manifest.get("raw_archive"), purpose="manifest archive",
        expected_path=archive_path,
    )
    seal_binding = manifest.get("preformal_seal")
    seal_path = _safe_relative(
        root,
        str(seal_binding.get("path")) if isinstance(seal_binding, dict) else "",
        purpose="manifest preformal seal",
    )
    _verify_binding(
        root, seal_binding, purpose="manifest preformal seal",
        expected_path=seal_path,
    )
    if seal.get("analysis_sha256") != manifest.get(
        "preformal_seal_analysis_sha256"
    ):
        raise EvidenceIncomplete("manifest preformal analysis mismatch")
    analysis = manifest.get("execution_sha256")
    unsigned = dict(manifest)
    unsigned.pop("execution_sha256", None)
    if analysis != _sha(unsigned):
        raise EvidenceIncomplete("execution manifest analysis hash invalid")
    return manifest


def _validate_attempt_ledger(
    root: Path,
    qualification: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    prefix = manifest.get("attempt_ledger_prefix")
    if not isinstance(prefix, dict):
        raise EvidenceIncomplete("attempt-ledger prefix receipt missing")
    path = _safe_relative(
        root,
        qualification["artifact_paths"]["attempt_ledger"],
        purpose="fresh attempt ledger",
    )
    if str(prefix.get("path", "")).replace("\\", "/") != path.relative_to(
        root
    ).as_posix():
        raise EvidenceIncomplete("attempt-ledger prefix path mismatch")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvidenceIncomplete("attempt ledger missing") from exc
    prefix_bytes = prefix.get("bytes")
    if (
        isinstance(prefix_bytes, bool)
        or not isinstance(prefix_bytes, int)
        or prefix_bytes <= 0
        or prefix_bytes >= len(payload)
        or _sha_bytes(payload[:prefix_bytes]) != prefix.get("sha256")
        or not payload[:prefix_bytes].endswith(b"\n")
        or not payload.endswith(b"\n")
    ):
        raise EvidenceIncomplete("attempt-ledger prefix binding invalid")
    events: list[dict[str, Any]] = []
    previous = "0" * 64
    for index, line in enumerate(payload.splitlines()):
        try:
            event = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceIncomplete("attempt ledger has invalid JSON") from exc
        if not isinstance(event, dict):
            raise EvidenceIncomplete("attempt event is not an object")
        claimed = event.get("event_sha256")
        unsigned = dict(event)
        unsigned.pop("event_sha256", None)
        if (
            event.get("event_schema") != ATTEMPT_SCHEMA
            or event.get("event_index") != index
            or event.get("previous_event_sha256") != previous
            or claimed != _sha(unsigned)
        ):
            raise EvidenceIncomplete("attempt ledger hash-chain mismatch")
        previous = str(claimed)
        events.append(event)
    prefix_events = payload[:prefix_bytes].splitlines()
    if (
        prefix.get("last_event_index") != len(prefix_events) - 1
        or prefix.get("last_event_sha256")
        != events[len(prefix_events) - 1]["event_sha256"]
        or len(events) != len(prefix_events) + 1
        or events[-1].get("event_kind") != "FINALIZED"
    ):
        raise EvidenceIncomplete("attempt-ledger finalization placement invalid")
    committed = [
        event for event in events if event.get("event_kind") == "CHUNK_COMMITTED"
    ]
    chunk_ids = [
        event.get("chunk", {}).get("chunk_id")
        if isinstance(event.get("chunk"), dict)
        else None
        for event in committed
    ]
    error_indices = [
        index for index, event in enumerate(events)
        if event.get("event_kind") == "RUN_ERROR"
    ]
    resume_indices = [
        index for index, event in enumerate(events)
        if event.get("event_kind") == "RESUME_STARTED"
    ]
    unrecovered_error = any(
        not any(resume > error for resume in resume_indices)
        for error in error_indices
    )
    if (
        len(committed) != manifest.get("observed_cells")
        or len(chunk_ids) != len(set(chunk_ids))
        or any(value is None for value in chunk_ids)
        or unrecovered_error
    ):
        raise EvidenceIncomplete("attempt-ledger chunk accounting invalid")


def _archive_ledger_coverage(
    rows: Sequence[Mapping[str, Any]],
    archive: ArchiveEvidence,
    qualification: Mapping[str, Any],
) -> None:
    count = int(qualification["common_physics"]["iq_samples"])
    row_ids = {str(row["row_id"]) for row in rows}
    if set(archive.raw_iq_by_row) != row_ids:
        raise EvidenceIncomplete("raw IQ archive/ledger row coverage mismatch")
    if set(archive.heldout_iq_by_row) != row_ids:
        raise EvidenceIncomplete("heldout IQ archive/ledger row coverage mismatch")
    for row in rows:
        row_id = str(row["row_id"])
        if archive.raw_iq_by_row[row_id].shape != (count, 2):
            raise EvidenceIncomplete("raw IQ sample-count mismatch")
        if archive.heldout_iq_by_row[row_id].shape != (count, 2):
            raise EvidenceIncomplete("heldout IQ sample-count mismatch")
        expected_locator = archive.locator_by_row.get(row_id)
        if expected_locator != (
            str(row["archive_chunk"]),
            int(row["archive_row_index"]),
        ):
            raise EvidenceIncomplete("ledger/archive row locator mismatch")
        if (
            int(row["raw_iq_index"]) != int(row["archive_row_index"])
            or int(row["heldout_iq_index"]) != int(row["archive_row_index"])
        ):
            raise EvidenceIncomplete("ledger IQ archive index mismatch")
        raw_mean = np.mean(archive.raw_iq_by_row[row_id], axis=0)
        errors = np.abs(
            raw_mean
            - np.asarray([row["integrated_i"], row["integrated_q"]])
        )
        if (
            np.max(errors) > 5.0e-12
            or abs(errors[0] - float(row["integrated_i_mean_error"])) > 1.0e-15
            or abs(errors[1] - float(row["integrated_q_mean_error"])) > 1.0e-15
        ):
            raise EvidenceIncomplete("raw IQ integrated mean mismatch")
        heldout_digest = _sha_bytes(
            np.asarray(
                archive.heldout_iq_by_row[row_id], dtype="<f8"
            ).tobytes(order="C")
        )
        if heldout_digest != row["heldout_window_sha256"]:
            raise EvidenceIncomplete("heldout IQ window hash mismatch")
        if int(row["density_index"]) >= 0 and row_id not in archive.density_by_row:
            raise EvidenceIncomplete("ledger density reference missing")
    if not set(archive.density_by_row) <= row_ids:
        raise EvidenceIncomplete("archive contains unknown density row")


def _diagnostic_raw_log_evidence(
    rows: Sequence[Mapping[str, Any]],
    archive: ArchiveEvidence,
    qualification: Mapping[str, Any],
) -> dict[str, float | int | bool]:
    sigma = float(qualification["common_physics"]["iq_sigma"])
    maximum_error = 0.0
    batch_size = 4096
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        priors, centers, _, _ = _predictive_features(batch, qualification)
        raw = np.stack(
            [archive.raw_iq_by_row[str(row["row_id"])] for row in batch]
        )
        logs = _component_logs(raw, centers, sigma)
        weighted = np.where(
            priors > 0.0,
            logs + np.log(np.maximum(priors, 1.0e-300)),
            -np.inf,
        )
        reference = _logsumexp_rows(weighted)
        observed = np.asarray(
            [float(row["raw_log_evidence"]) for row in batch]
        )
        maximum_error = max(
            maximum_error, float(np.max(np.abs(reference - observed)))
        )
    return {
        "primary": False,
        "rows_checked": len(rows),
        "maximum_absolute_reference_error": maximum_error,
        "reference_consistent": maximum_error <= 5.0e-10,
    }


def _validate_semantic_receipts(
    rows: Sequence[Mapping[str, Any]],
    archive: ArchiveEvidence,
    qualification: Mapping[str, Any],
) -> dict[str, float | int]:
    """Independently check every runner-side semantic receipt in batches."""

    sigma = float(qualification["common_physics"]["iq_sigma"])
    sample_count = int(qualification["common_physics"]["iq_samples"])
    reset_e = float(qualification["common_physics"]["reset_success_e"])
    reset_f = float(qualification["common_physics"]["reset_success_f"])
    maxima = {
        "raw_log_evidence": 0.0,
        "raw_posterior_l1": 0.0,
        "raw_residual": 0.0,
        "predictive_mean": 0.0,
        "predictive_covariance": 0.0,
        "heldout_log_evidence": 0.0,
        "heldout_score": 0.0,
        "heldout_llr": 0.0,
        "rao_blackwell_reset": 0.0,
    }
    for offset in range(0, len(rows), 4096):
        batch = rows[offset : offset + 4096]
        priors, centers, means, covariances = _predictive_features(
            batch, qualification
        )
        raw = np.stack(
            [archive.raw_iq_by_row[str(row["row_id"])] for row in batch]
        )
        heldout = np.stack(
            [archive.heldout_iq_by_row[str(row["row_id"])] for row in batch]
        )
        raw_logs = _component_logs(raw, centers, sigma)
        heldout_logs = _component_logs(heldout, centers, sigma)
        log_prior = np.where(
            priors > 0.0, np.log(np.maximum(priors, 1.0e-300)), -np.inf
        )
        raw_evidence = _logsumexp_rows(raw_logs + log_prior)
        heldout_evidence = _logsumexp_rows(heldout_logs + log_prior)
        raw_posterior = np.exp(
            raw_logs + log_prior - raw_evidence[:, None]
        )
        raw_posterior = _validate_probabilities(
            raw_posterior, "raw reference posterior"
        )
        residual = np.sum(
            (raw - np.mean(raw, axis=1, keepdims=True)) ** 2,
            axis=(1, 2),
        )
        observed_posterior = np.asarray(
            [
                [row["posterior_g"], row["posterior_e"], row["posterior_f"]]
                for row in batch
            ],
            dtype=np.float64,
        )
        stored_means = np.asarray(
            [
                [float(row["predictive_mean_i"]), float(row["predictive_mean_q"])]
                for row in batch
            ]
        )
        stored_cov = np.asarray(
            [
                [
                    [float(row["predictive_cov_ii"]), float(row["predictive_cov_iq"])],
                    [float(row["predictive_cov_iq"]), float(row["predictive_cov_qq"])],
                ]
                for row in batch
            ]
        )
        heldout_llr = np.stack(
            (
                heldout_logs[:, 0] - heldout_logs[:, 1],
                heldout_logs[:, 0] - heldout_logs[:, 2],
                heldout_logs[:, 1] - heldout_logs[:, 2],
            ),
            axis=1,
        ) / sample_count
        stored_llr = np.asarray(
            [
                [
                    float(row["heldout_llr_ge_per_sample"]),
                    float(row["heldout_llr_gf_per_sample"]),
                    float(row["heldout_llr_ef_per_sample"]),
                ]
                for row in batch
            ]
        )
        comparisons = {
            "raw_log_evidence": np.abs(
                raw_evidence
                - np.asarray(
                    [float(row["raw_reference_log_evidence"]) for row in batch]
                )
            ),
            "raw_posterior_l1": np.sum(
                np.abs(raw_posterior - observed_posterior), axis=1
            ),
            "raw_residual": np.abs(
                residual
                - np.asarray(
                    [float(row["raw_within_window_residual"]) for row in batch]
                )
            ),
            "predictive_mean": np.linalg.norm(
                means - stored_means, axis=1
            ),
            "predictive_covariance": np.linalg.norm(
                covariances - stored_cov, axis=(1, 2)
            ),
            "heldout_log_evidence": np.abs(
                heldout_evidence
                - np.asarray(
                    [
                        float(row["heldout_reference_log_evidence"])
                        for row in batch
                    ]
                )
            ),
            "heldout_score": np.abs(
                heldout_evidence / sample_count
                - np.asarray(
                    [
                        float(row["heldout_proper_score_per_sample"])
                        for row in batch
                    ]
                )
            ),
            "heldout_llr": np.max(np.abs(heldout_llr - stored_llr), axis=1),
        }
        for name, values in comparisons.items():
            if not np.all(np.isfinite(values)):
                raise EvidenceIncomplete(f"semantic receipt {name} non-finite")
            maxima[name] = max(maxima[name], float(np.max(values)))
        for index, row in enumerate(batch):
            if row["reset_requested"]:
                probabilities = _validate_probabilities(
                    np.asarray(
                        [[row["pre_reset_g"], row["pre_reset_e"], row["pre_reset_f"]]]
                    ),
                    "pre-reset levels",
                )[0]
                expected = (
                    probabilities[0]
                    + reset_e * probabilities[1]
                    + reset_f * probabilities[2]
                )
                maxima["rao_blackwell_reset"] = max(
                    maxima["rao_blackwell_reset"],
                    abs(expected - float(row["rao_blackwell_reset_success"])),
                )
            elif not math.isnan(float(row["rao_blackwell_reset_success"])):
                raise EvidenceIncomplete("non-reset row contains reset estimand")
    tolerances = {
        "raw_log_evidence": 5.0e-10,
        "raw_posterior_l1": 5.0e-10,
        "raw_residual": 5.0e-11,
        "predictive_mean": 5.0e-12,
        "predictive_covariance": 5.0e-12,
        "heldout_log_evidence": 5.0e-10,
        "heldout_score": 1.0e-10,
        "heldout_llr": 1.0e-10,
        "rao_blackwell_reset": 5.0e-12,
    }
    for name, maximum in maxima.items():
        if maximum > tolerances[name]:
            raise EvidenceIncomplete(
                f"semantic receipt mismatch {name}: {maximum}"
            )
    return {"rows_checked": len(rows), **maxima}


def _typed_null_claim_state() -> dict[str, None]:
    return {field: None for field in TYPED_NULL_FIELDS}


def _release_payload(
    *,
    verdict: str,
    analysis_sha256: str | None,
) -> dict[str, Any]:
    passed = verdict == PASS_VERDICT
    payload: dict[str, Any] = {
        "task_id": TASK_ID,
        "schema_version": "PHASE9-FRESH-TWIN-RELEASE-V1",
        "verdict": verdict,
        "qualified_claim": QUALIFIED_CLAIM if passed else None,
        "qualification_analysis_sha256": analysis_sha256 if passed else None,
        "historical_t9_2_4_no_go_preserved": True,
        "downstream_release": {
            task: {
                "released": passed,
                "reason": (
                    "fresh repaired synthetic-task twin qualification passed"
                    if passed
                    else "fresh twin qualification did not pass"
                ),
            }
            for task in DOWNSTREAM_TASKS
        },
        "claim_state": _typed_null_claim_state(),
    }
    payload["analysis_sha256"] = _sha(payload)
    return payload


def verify_bundle(
    root: Path,
    *,
    config_path: Path,
    design_config_path: Path,
    ledger_path: Path,
    archive_path: Path,
    manifest_path: Path,
    allow_test_fixture: bool = False,
) -> tuple[dict[str, Any], list[dict[str, object]], dict[str, Any]]:
    base = root.resolve()
    try:
        qualification = _load_json(config_path)
        design = _load_json(design_config_path)
        _validate_config(
            qualification, design, allow_test_fixture=allow_test_fixture
        )
        _validate_design_lineage(base, qualification)
        seal = _validate_preformal_seal(base, config_path, qualification)
        manifest = _validate_manifest(
            base,
            manifest_path,
            config_path,
            ledger_path,
            archive_path,
            seal,
        )
        if not allow_test_fixture and (
            manifest.get("observed_cells") != 592
            or manifest.get("observed_rows") != 528384
        ):
            raise EvidenceIncomplete("production formal denominator count drift")
        _validate_attempt_ledger(base, qualification, manifest)
        rows, header = load_ledger(ledger_path)
        if manifest.get("observed_rows") != len(rows):
            raise EvidenceIncomplete("execution manifest row count mismatch")
        archive = load_archive(archive_path)
        _archive_ledger_coverage(rows, archive, qualification)
        semantic_receipts = _validate_semantic_receipts(
            rows, archive, qualification
        )
        gates = evaluate_gates(rows, archive, qualification, design)
        expected_gate_count = len(
            qualification["gate_blueprint"]["rows"]
        )
        if len(gates) != expected_gate_count:
            raise EvidenceIncomplete("verification gate count mismatch")
        diagnostic = _diagnostic_raw_log_evidence(
            rows, archive, qualification
        )
        # The raw score remains non-primary, but a mismatch against the frozen
        # semantic reference is evidence corruption/implementation drift.
        if diagnostic["reference_consistent"] is not True:
            raise EvidenceIncomplete("raw log-evidence semantic mismatch")
        verdict = (
            PASS_VERDICT if all(gate.passed for gate in gates) else NO_GO_VERDICT
        )
        report: dict[str, Any] = {
            "task_id": TASK_ID,
            "verifier_id": VERIFIER_ID,
            "schema_version": "PHASE9-FRESH-TWIN-VERIFICATION-REPORT-V1",
            "verdict": verdict,
            "qualified_claim": (
                QUALIFIED_CLAIM if verdict == PASS_VERDICT else None
            ),
            "historical_t9_2_4_no_go_preserved": True,
            "old_formal_cell_data_accessed": False,
            "runner_or_physics_imported": False,
            "statistical_procedure": {
                "global": "intersection_union_equivalence",
                "cell": "two_one_sided_tests",
                "confidence_interval": 0.90,
                "cluster_unit": "seed_position",
                "all_cells_required": True,
                "postselection": False,
            },
            "raw_log_evidence_diagnostic": diagnostic,
            "independent_semantic_receipt_check": semantic_receipts,
            "row_count": len(rows),
            "ledger_header_sha256": _sha(list(header)),
            "archive_chunk_count": len(archive.chunk_bindings),
            "gate_summary": {
                "passed": sum(gate.passed for gate in gates),
                "failed": sum(not gate.passed for gate in gates),
                "total": len(gates),
                "all_passed": all(gate.passed for gate in gates),
            },
            "family_summary": {
                family: {
                    "passed": sum(
                        gate.passed for gate in gates if gate.family == family
                    ),
                    "total": sum(gate.family == family for gate in gates),
                }
                for family in sorted({gate.family for gate in gates})
            },
            "failed_gate_ids": [
                gate.gate_id for gate in gates if not gate.passed
            ],
            "claim_state": _typed_null_claim_state(),
            "bindings": {
                "config": _binding(config_path, base),
                "design_config": _binding(design_config_path, base),
                "preformal_seal": _binding(
                    _safe_relative(
                        base,
                        qualification["preformal_seal"]["path"],
                        purpose="preformal seal",
                    ),
                    base,
                ),
                "execution_manifest": _binding(manifest_path, base),
                "cell_ledger": _binding(ledger_path, base),
                "raw_archive": _binding(archive_path, base),
            },
        }
        report["analysis_sha256"] = _sha(report)
        gate_rows = [gate.to_dict() for gate in gates]
        release = _release_payload(
            verdict=verdict,
            analysis_sha256=(
                report["analysis_sha256"] if verdict == PASS_VERDICT else None
            ),
        )
        return report, gate_rows, release
    except EvidenceIncomplete as exc:
        report = {
            "task_id": TASK_ID,
            "verifier_id": VERIFIER_ID,
            "schema_version": "PHASE9-FRESH-TWIN-VERIFICATION-REPORT-V1",
            "verdict": INCOMPLETE_VERDICT,
            "qualified_claim": None,
            "historical_t9_2_4_no_go_preserved": True,
            "old_formal_cell_data_accessed": False,
            "runner_or_physics_imported": False,
            "incomplete_reason": str(exc),
            "gate_summary": {
                "passed": 0,
                "failed": 0,
                "total": 0,
                "all_passed": False,
            },
            "claim_state": _typed_null_claim_state(),
        }
        report["analysis_sha256"] = _sha(report)
        return report, [], _release_payload(
            verdict=INCOMPLETE_VERDICT, analysis_sha256=None
        )


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


def _csv_text(rows: Iterable[Mapping[str, object]]) -> str:
    values = list(rows)
    fields = (
        "gate_id",
        "family",
        "stage",
        "metric",
        "direction",
        "estimate",
        "standard_error",
        "bound",
        "margin",
        "cluster_count",
        "passed",
        "denominator",
    )
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
        stream.seek(0)
        return stream.read()


def write_artifacts(root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    report, gates, release = verify_bundle(
        base,
        config_path=base / DEFAULT_CONFIG_PATH,
        design_config_path=base / DEFAULT_DESIGN_CONFIG_PATH,
        ledger_path=base / DEFAULT_LEDGER_PATH,
        archive_path=base / DEFAULT_ARCHIVE_PATH,
        manifest_path=base / DEFAULT_MANIFEST_PATH,
    )
    qualification_path = base / DEFAULT_QUALIFICATION_PATH
    source_path = base / DEFAULT_SOURCE_PATH
    gate_ledger_path = base / DEFAULT_GATE_LEDGER_PATH
    release_path = base / DEFAULT_RELEASE_PATH
    release_pin_path = base / DEFAULT_RELEASE_PIN_PATH
    qualification_text = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    source_text = _csv_text(gates)
    release_text = (
        json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _atomic_text(qualification_path, qualification_text)
    _atomic_text(source_path, source_text)
    _atomic_text(gate_ledger_path, source_text)
    _atomic_text(release_path, release_text)
    _atomic_text(release_pin_path, release_text)
    verification = {
        "task_id": TASK_ID,
        "verifier_id": VERIFIER_ID,
        "schema_version": "PHASE9-FRESH-TWIN-INDEPENDENT-VERIFICATION-V1",
        "verdict": report["verdict"],
        "qualified_claim": report["qualified_claim"],
        "qualification_analysis_sha256": report["analysis_sha256"],
        "historical_t9_2_4_no_go_preserved": True,
        "claim_state": _typed_null_claim_state(),
        "bindings": {
            "qualification": _binding(qualification_path, base),
            "qualification_source_data": _binding(source_path, base),
            "gate_ledger": _binding(gate_ledger_path, base),
            "release": _binding(release_path, base),
            "release_pin": _binding(release_pin_path, base),
        },
    }
    verification["analysis_sha256"] = _sha(verification)
    _atomic_text(
        base / DEFAULT_REPORT_PATH,
        json.dumps(
            verification, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("fresh verifier accepts no CLI overrides")
    report = write_artifacts()
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "analysis_sha256": report["analysis_sha256"],
                "gate_summary": report["gate_summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["verdict"] == PASS_VERDICT else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_MANIFEST_SCHEMA",
    "ARCHIVE_SCHEMA",
    "CONFIG_SCHEMA",
    "DOWNSTREAM_TASKS",
    "EvidenceIncomplete",
    "GateResult",
    "GateSpec",
    "INCOMPLETE_VERDICT",
    "NO_GO_VERDICT",
    "PASS_VERDICT",
    "QUALIFIED_CLAIM",
    "REQUIRED_LEDGER_FIELDS",
    "ROW_SCHEMA",
    "TYPED_NULL_FIELDS",
    "evaluate_gates",
    "load_archive",
    "load_ledger",
    "verify_bundle",
    "write_artifacts",
]
